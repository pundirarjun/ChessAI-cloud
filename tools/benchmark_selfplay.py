"""Self-play engine benchmark for the 2x Tesla T4 gates.

Run from project root (Kaggle notebook / shell):

    python tools/benchmark_selfplay.py --engine cpp --sims 100,200,400,800

For each simulation count it reports:
  * full self-play throughput (games/hour) via training.self_play.play_games
  * raw MCTS throughput (simulations/sec, NN evaluations/sec) from a fixed
    search microbenchmark with an eval counter
  * GPU utilization / memory and CPU load sampled while running

Gate: >= 50 games/hour at 400 sims (100+ is good).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from az.reference import gpu_from_board  # noqa: E402
from environment.gpu_chess import GPUChess  # noqa: E402
from model.chess_net import ChessNet  # noqa: E402


class _GpuSampler(threading.Thread):
    """Poll nvidia-smi for GPU utilization / memory while a benchmark runs."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        self.samples.append(
                            (float(parts[0]), float(parts[1]))
                        )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)

    @property
    def report(self) -> dict:
        if not self.samples:
            return {"gpu_util_avg": None, "gpu_util_peak": None,
                    "gpu_mem_peak_mib": None}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "gpu_util_avg": round(sum(utils) / len(utils), 1),
            "gpu_util_peak": max(utils),
            "gpu_mem_peak_mib": max(mems),
        }


def _cpu_load() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def _fixed_states(device, games: int) -> GPUChess:
    """Diverse fixed root positions for the search microbenchmark."""
    fens = [
        "startpos",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2k5/8/8/8/8/2K5/4R3 w - - 0 1",
    ]
    boards = [
        chess.Board() if f == "startpos" else chess.Board(f) for f in fens
    ]
    while len(boards) < games:
        boards.extend(boards[: games - len(boards)])
    boards = boards[:games]

    states = GPUChess(device, games)
    for i, board in enumerate(boards):
        one = gpu_from_board(board, device)
        states.pieces[i] = one.pieces[0]
        states.turn[i] = one.turn[0]
        states.castling[i] = one.castling[0]
        states.ep_square[i] = one.ep_square[0]
        states.halfmove_clock[i] = one.halfmove_clock[0]
        states.fullmove_number[i] = one.fullmove_number[0]
    return states


def benchmark_search(
    engine: str, model, device, sims: int, games: int, batch_size: int
) -> dict:
    """Fixed-root search microbenchmark: sims/sec and NN evals/sec."""
    from mcts.cpp_mcts import CppMctsAdapter
    from mcts.gpu_mcts import GPUMCTS

    states = _fixed_states(device, games)
    eval_count = 0

    if engine == "cpp":
        engine_obj = CppMctsAdapter(
            model=model, device=device, c_puct=1.5, seed=1234
        )
        inner = engine_obj._evaluate

        def counted(planes):
            nonlocal eval_count
            eval_count += planes.shape[0]
            return inner(planes)

        engine_obj._evaluate = counted
        eval_fn = counted
    else:
        engine_obj = GPUMCTS(model=model, device=device, c_puct=1.5)
        inner = engine_obj._evaluate

        def counted(node_ids):
            nonlocal eval_count
            eval_count += node_ids.numel()
            return inner(node_ids)

        engine_obj._evaluate = counted

    if device.type == "cuda":
        torch.cuda.synchronize()
    sampler = _GpuSampler()
    sampler.start()
    load_before = _cpu_load()
    start = time.perf_counter()
    if engine == "cpp":
        engine_obj.search(
            states,
            num_simulations=sims,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=batch_size,
            repetition_history=torch.zeros((games, 0), dtype=torch.int64,
                                           device=device),
        )
    else:
        engine_obj.search(
            states,
            num_simulations=sims,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=batch_size,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    sampler.stop()
    load_after = _cpu_load()

    total_sims = games * sims
    return {
        "sims_per_sec": round(total_sims / elapsed, 1),
        "nn_evals_per_sec": round(eval_count / elapsed, 1),
        "search_seconds": round(elapsed, 3),
        "games": games,
        "simulations": sims,
        "cpu_load_avg": (
            None
            if load_before is None or load_after is None
            else round((load_before + load_after) / 2, 2)
        ),
        **sampler.report,
    }


def benchmark_selfplay(engine: str, model, args) -> dict:
    from training.self_play import play_games

    if device_of(model).type != "cuda":
        return {"error": "self-play benchmark requires CUDA"}
    games = args.selfplay_games
    start = time.perf_counter()
    results = play_games(
        model=model,
        num_games=games,
        num_simulations=args.selfplay_sims,
        max_moves=args.max_moves,
        temperature=1.0,
        temperature_moves=args.temperature_moves,
        late_temperature=0.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=args.batch_size,
        engine=engine,
        c_puct=1.5,
        seed=1234,
    )
    elapsed = time.perf_counter() - start
    moves = sum(r.moves_played for r in results)
    completed = sum(1 for r in results if r.completed)
    return {
        "games": games,
        "completed": completed,
        "moves": moves,
        "elapsed_seconds": round(elapsed, 2),
        "games_per_hour": round(3600.0 * games / elapsed, 1),
        "moves_per_sec": round(moves / elapsed, 2),
        "mean_game_moves": round(moves / max(games, 1), 1),
    }


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="cpp",
                        choices=["cpp", "python_gpu_reference"])
    parser.add_argument("--sims", default="100,200,400,800",
                        help="comma-separated simulation counts for the "
                             "search microbenchmark")
    parser.add_argument("--search-games", type=int, default=8,
                        help="root positions in the search microbenchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--selfplay", action="store_true",
                        help="also run the full self-play games/hour gate")
    parser.add_argument("--selfplay-games", type=int, default=8)
    parser.add_argument("--selfplay-sims", type=int, default=400)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--checkpoint", default=None,
                        help="optional checkpoint to benchmark a trained model")
    parser.add_argument("--require-games-per-hour", type=float, default=0.0,
                        help="exit non-zero if the self-play gate measures "
                             "below this games/hour (50 for the T4 gate)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Engine: {args.engine}")

    model = ChessNet(action_space_size=4544).to(device).eval()
    model.to(memory_format=torch.channels_last)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

    report: dict = {"engine": args.engine, "gpu": torch.cuda.get_device_name(0)}

    # Warmup (JIT/cuDNN/allocator) before timing.
    benchmark_search(args.engine, model, device, sims=10,
                     games=2, batch_size=4)

    search_results = []
    for sims in [int(s) for s in args.sims.split(",") if s.strip()]:
        print(f"-- search benchmark: {sims} sims ...", flush=True)
        result = benchmark_search(
            args.engine, model, device, sims=sims,
            games=args.search_games, batch_size=args.batch_size,
        )
        search_results.append(result)
        print(json.dumps(result), flush=True)
    report["search"] = search_results

    if args.selfplay:
        print(f"-- self-play: {args.selfplay_games} games @ "
              f"{args.selfplay_sims} sims ...", flush=True)
        sampler = _GpuSampler()
        sampler.start()
        sp = benchmark_selfplay(args.engine, model, args)
        sampler.stop()
        sp.update(sampler.report)
        report["selfplay"] = sp
        print(json.dumps(sp), flush=True)
        if "games_per_hour" in sp:
            gate = sp["games_per_hour"] >= 50.0
            print(f"games/hour gate (>=50): {'PASS' if gate else 'FAIL'} "
                  f"({sp['games_per_hour']})")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json_out}")

    if args.require_games_per_hour > 0:
        gph = report.get("selfplay", {}).get("games_per_hour")
        if gph is None:
            raise SystemExit("self-play benchmark did not run "
                             "(need --selfplay with CUDA)")
        if gph < args.require_games_per_hour:
            raise SystemExit(
                f"GATE FAIL: {gph} games/hour < "
                f"{args.require_games_per_hour}"
            )
        print(f"GATE PASS: {gph} games/hour >= "
              f"{args.require_games_per_hour}")


if __name__ == "__main__":
    main()
