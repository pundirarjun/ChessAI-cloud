"""CUDA smoke test for the full C++-engine self-play path.

Run from project root (Kaggle):

    python tools/cpp_selfplay_smoke.py
    python tools/cpp_selfplay_smoke.py --parity   # also cross-check against
                                                   # the reference engine

Requires CUDA; exits 0 with a SKIP message when unavailable (local CPU boxes).

Checks, for engine="cpp" through training.self_play.play_games:
  * games run to a real termination (completed, moves_played > 0)
  * training samples are well-formed: state (18,8,8), policy (4544) finite
    and summing to 1, value in {-1, 0, 1}

With --parity it additionally runs the reference engine with identical
parameters (temperature=0, no Dirichlet noise) and asserts the move sequences
(identical state trajectories), terminations, and sample counts match — a
full-game extension of the exact root-policy parity in tests/test_cpp_adapter.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.chess_net import ChessNet  # noqa: E402
from training.self_play import play_games  # noqa: E402


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name} FAILED" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}")


def run_games(model, engine: str, args):
    return play_games(
        model=model,
        num_games=args.games,
        num_simulations=args.sims,
        max_moves=args.max_moves,
        temperature=0.0,
        temperature_moves=0,
        late_temperature=0.0,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        batch_size=args.batch_size,
        engine=engine,
        c_puct=1.5,
        seed=1234,
    )


def check_sanity(tag: str, results):
    completed = 0
    total_samples = 0
    for i, r in enumerate(results):
        assert r is not None, f"{tag}: game {i} missing"
        check(
            f"{tag}: game {i} completed with moves",
            r.completed and r.moves_played > 0,
            f"completed={r.completed} moves={r.moves_played} "
            f"termination={r.termination}",
        )
        check(
            f"{tag}: game {i} termination is a real outcome",
            r.termination in {"CHECKMATE", "STALEMATE", "FIFTY_MOVE",
                              "INSUFFICIENT_MATERIAL", "THREEFOLD_REPETITION"},
            r.termination,
        )
        check(f"{tag}: game {i} result in {{0,1,2}}", r.result in (0, 1, 2),
              str(r.result))
        completed += int(r.completed)
        total_samples += len(r.training_data)
        for state, policy, value in r.training_data:
            assert state.shape == (18, 8, 8), f"{tag}: bad state {state.shape}"
            assert policy.shape == (4544,), f"{tag}: bad policy {policy.shape}"
            assert np.isfinite(policy).all(), f"{tag}: non-finite policy"
            assert abs(float(policy.sum()) - 1.0) < 1e-4, (
                f"{tag}: policy sum {policy.sum()}"
            )
            assert value in (-1.0, 0.0, 1.0), f"{tag}: value {value}"
    check(f"{tag}: all games completed", completed == len(results),
          f"{completed}/{len(results)}")
    check(f"{tag}: training samples produced", total_samples > 0)
    return total_samples


def trajectories(results):
    """Per-game list of state hashes (a fingerprint of the move sequence)."""
    out = []
    for r in results:
        out.append(
            [np.asarray(s, dtype=np.float32).tobytes()
             for s, _p, _v in r.training_data]
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parity", action="store_true",
                        help="cross-check the full game trajectory against "
                             "the reference engine")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    try:
        import az_cpp_mcts  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"az_cpp_mcts is not importable: {exc}")

    model = ChessNet(action_space_size=4544).to(device).eval()
    model.to(memory_format=torch.channels_last)

    # Warmup for the cpp path (module already imported by play_games).
    t0 = time.perf_counter()
    cpp_results = run_games(model, "cpp", args)
    cpp_elapsed = time.perf_counter() - t0
    print(f"cpp self-play: {cpp_elapsed:.1f}s")
    check_sanity("cpp", cpp_results)

    if args.parity:
        t0 = time.perf_counter()
        ref_results = run_games(model, "python_gpu_reference", args)
        print(f"reference self-play: {time.perf_counter() - t0:.1f}s")
        check_sanity("reference", ref_results)

        cpp_traj = trajectories(cpp_results)
        ref_traj = trajectories(ref_results)
        check(
            "identical move counts",
            [len(t) for t in cpp_traj] == [len(t) for t in ref_traj],
            f"{[len(t) for t in cpp_traj]} vs {[len(t) for t in ref_traj]}",
        )
        for i, (a, b) in enumerate(zip(cpp_traj, ref_traj)):
            check(f"game {i} trajectory identical", a == b)
        check(
            "identical terminations",
            [r.termination for r in cpp_results]
            == [r.termination for r in ref_results],
        )
        check(
            "identical results",
            [r.result for r in cpp_results] == [r.result for r in ref_results],
        )

    print()
    print("=" * 60)
    print("CPP SELF-PLAY SMOKE TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
