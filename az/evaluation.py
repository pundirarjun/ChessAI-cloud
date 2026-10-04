"""Reproducible, color-balanced candidate-vs-best evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp
import shutil
import tempfile

import torch

from az.checkpoints import load_checkpoint
from az.config import RunConfig
from az.gating import EvaluationGame, GateDecision, evaluate_gate
from az.model import build_fresh_model
from az.provenance import RunManifest, load_manifest
from az.reproducibility import seed_everything
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


def evaluate_candidate(
    *,
    root: str | Path,
    config: RunConfig,
    manifest: RunManifest,
) -> tuple[list[EvaluationGame], GateDecision]:
    """Evaluate candidate and best at identical search settings.

    The first ``temperature_moves`` plies are sampled at ``temperature`` so
    games diversify instead of collapsing into identical deterministic
    shuffles; later play is deterministic. Evaluation has no Dirichlet noise.
    Every game has an assigned color and seed, and max-move truncations
    remain None rather than becoming draws.
    """

    config.validate()
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and (config.runtime.num_gpus or 1) >= 2
    ):
        games = _evaluate_multi_gpu(root, config)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        games = _evaluate_local(root, config, manifest, device, range(config.evaluation.games))
    print(
        f"[eval] {len(games)} games done (engine={config.mcts.engine}, "
        f"{config.evaluation.simulations} sims)",
        flush=True,
    )
    decision = evaluate_gate(
        games,
        promotion_score=config.evaluation.promotion_score,
        confidence_level=config.evaluation.confidence_level,
        min_completed_games=config.evaluation.min_completed_games,
    )
    return games, decision


def _evaluate_local(
    root: str | Path,
    config: RunConfig,
    manifest: RunManifest,
    device: torch.device,
    game_indices: range | list[int],
) -> list[EvaluationGame]:
    best = build_fresh_model(config.architecture, config.runtime, device)
    candidate = build_fresh_model(config.architecture, config.runtime, device)
    load_checkpoint(root=root, manifest=manifest, role="best", model=best, map_location=device)
    load_checkpoint(root=root, manifest=manifest, role="candidate", model=candidate, map_location=device)
    best.eval()
    candidate.eval()
    print(
        f"[eval] playing {len(list(game_indices))} games on {device} "
        f"@ {config.evaluation.simulations} sims "
        f"(engine={config.mcts.engine})",
        flush=True,
    )
    games = []
    for game_id in game_indices:
        game = _play_one(
            candidate=candidate,
            best=best,
            config=config,
            game_id=game_id,
            device=device,
        )
        games.append(game)
        print(
            f"[eval] game {game.game_id} ({game.candidate_color}): "
            f"{game.termination} result={game.result_for_candidate}",
            flush=True,
        )
    return games


def _play_one(
    *,
    candidate: torch.nn.Module,
    best: torch.nn.Module,
    config: RunConfig,
    game_id: int,
    device: torch.device,
) -> EvaluationGame:
    candidate_is_white = game_id % 2 == 0
    seed = config.runtime.seed + 1_000_000 + game_id
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    state = GPUChess(device, 1)
    repetitions: dict[int, int] = {}
    history: list[int] = []
    for _move in range(config.evaluation.max_moves):
        state_key = int(state.state_hash()[0].item())
        history.append(state_key)
        repetitions[state_key] = repetitions.get(state_key, 0) + 1
        if repetitions[state_key] >= 3:
            white_result, termination = 0, "THREEFOLD_REPETITION"
            break
        terminal = _terminal_result(state)
        if terminal is not None:
            white_result, termination = terminal
            break
        white_to_move = not bool(state.turn[0].item())
        active_model = candidate if white_to_move == candidate_is_white else best
        search = _make_search(active_model, config, device, seed + _move)
        repetition_history = torch.tensor([history], dtype=torch.int64, device=device)
        search.search(
            state,
            num_simulations=config.evaluation.simulations,
            batch_size=config.mcts.batch_size,
            repetition_history=repetition_history,
        )
        action = search.select_actions(
            config.evaluation.temperature
            if _move < config.evaluation.temperature_moves
            else 0.0
        )
        state = search.advance(action)
    else:
        white_result, termination = None, "MAX_MOVES"
    candidate_result = (
        None if white_result is None
        else int(white_result if candidate_is_white else -white_result)
    )
    return EvaluationGame(
        game_id=game_id,
        candidate_color="white" if candidate_is_white else "black",
        result_for_candidate=candidate_result,
        termination=termination,
        seed=seed,
        simulations=config.evaluation.simulations,
        temperature=config.evaluation.temperature,
    )


def _make_search(model, config: RunConfig, device: torch.device, seed: int):
    """Evaluation uses the configured engine; both are behaviorally identical
    (trajectory-proven in tools/cpp_selfplay_smoke.py --parity)."""
    engine = config.mcts.engine
    if engine == "cpp":
        from mcts.cpp_mcts import CppMctsAdapter

        return CppMctsAdapter(
            model=model, device=device, c_puct=config.mcts.c_puct, seed=seed
        )
    if engine == "python_gpu_reference":
        return GPUMCTS(model=model, device=device, c_puct=config.mcts.c_puct)
    raise ValueError(f"Unknown MCTS engine: {engine!r}")


def _terminal_result(state: GPUChess) -> tuple[int, str] | None:
    legal = state.legal_move_mask()[0]
    if not bool(legal.any().item()):
        if bool(state.is_in_check()[0].item()):
            # turn=False is White to move, so White has been checkmated.
            return (1 if bool(state.turn[0].item()) else -1), "CHECKMATE"
        return 0, "STALEMATE"
    if bool((state.halfmove_clock[0] >= 100).item()):
        return 0, "FIFTY_MOVE"
    if bool(state.insufficient_material()[0].item()):
        return 0, "INSUFFICIENT_MATERIAL"
    return None


def _evaluation_worker(
    rank: int,
    device_id: int,
    root: str,
    config: RunConfig,
    game_indices: list[int],
    output_path: str,
) -> None:
    torch.cuda.set_device(device_id)
    seed_everything(config.runtime, rank=rank)
    manifest = load_manifest(root)
    games = _evaluate_local(
        root,
        config,
        manifest,
        torch.device(f"cuda:{device_id}"),
        game_indices,
    )
    torch.save([asdict(game) for game in games], output_path)


def _evaluate_multi_gpu(root: str | Path, config: RunConfig) -> list[EvaluationGame]:
    root = Path(root).resolve()
    available = torch.cuda.device_count()
    workers = min(available, config.evaluation.games // 2)
    # Each worker receives an even contiguous count, so both workers process
    # white and black candidate assignments.
    if config.evaluation.games % workers:
        return _evaluate_local(
            root, config, load_manifest(root), torch.device("cuda:0"), range(config.evaluation.games)
        )
    games_per_worker = config.evaluation.games // workers
    if games_per_worker % 2:
        return _evaluate_local(
            root, config, load_manifest(root), torch.device("cuda:0"), range(config.evaluation.games)
        )
    temp_dir = Path(tempfile.mkdtemp(prefix="clean_az_eval_", dir=root / "logs"))
    context = mp.get_context("spawn")
    processes = []
    outputs = []
    try:
        for rank in range(workers):
            indices = list(range(rank * games_per_worker, (rank + 1) * games_per_worker))
            output = temp_dir / f"worker_{rank}.pt"
            process = context.Process(
                target=_evaluation_worker,
                args=(rank, rank, str(root), config, indices, str(output)),
            )
            process.start()
            processes.append(process)
            outputs.append(output)
        for process in processes:
            process.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Evaluation worker failure: {failed}")
        games = [
            EvaluationGame(**raw)
            for output in outputs
            for raw in torch.load(output, map_location="cpu", weights_only=False)
        ]
        games.sort(key=lambda game: game.game_id)
        return games
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
