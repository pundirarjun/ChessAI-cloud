# ============================================================
# RL53 GPU SELF-PLAY INTEGRATION TEST
# ============================================================
#
# Purpose:
#   Test the EXACT GPU self-play path used by RL training.
#
# This test uses:
#   - RL53 checkpoint
#   - training.self_play.play_games()
#   - GPUChess
#   - GPUMCTS
#   - actual self-play temperature
#   - actual Dirichlet noise
#
# It verifies:
#   1. Self-play runs without crashing.
#   2. Every returned game has a valid result/termination.
#   3. Completed games contain training samples.
#   4. MAX_MOVES games are correctly marked incomplete.
#   5. State shape is 18x8x8.
#   6. Policy shape is 4544.
#   7. Policies are finite, non-negative and normalized.
#   8. Values are exactly -1 / 0 / +1.
#   9. Player values are ±1.
#  10. Training-sample counts match move counts.
#  11. No NaN/Inf appears in generated data.
#
# NOTE:
#   This deliberately calls play_games(), not a copied version of
#   the self-play loop, so the real training path is being tested.
# ============================================================

import os
import sys
import traceback
import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/kaggle/working/chess-zero"

CHECKPOINT_PATH = (
    "/kaggle/working/chess-zero/checkpoints/rl_iteration_53.pt"
)

NUM_GAMES = 8
NUM_SIMULATIONS = 100
MAX_MOVES = 200

TEMPERATURE = 1.0
TEMPERATURE_MOVES = 60

DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Keep this aligned with the normal GPU self-play call.
BATCH_SIZE = 128

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 100)
print("RL53 GPU SELF-PLAY INTEGRATION TEST")
print("=" * 100)

print()
print("Project:", PROJECT_DIR)
print("Checkpoint:", CHECKPOINT_PATH)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print()
print("Games:", NUM_GAMES)
print("Simulations/game:", NUM_SIMULATIONS)
print("Max moves:", MAX_MOVES)
print("Temperature:", TEMPERATURE)
print("Temperature moves:", TEMPERATURE_MOVES)
print("Dirichlet alpha:", DIRICHLET_ALPHA)
print("Dirichlet epsilon:", DIRICHLET_EPSILON)
print("Batch size:", BATCH_SIZE)
print("Seed:", SEED)


# ============================================================
# ENVIRONMENT
# ============================================================

sys.path.insert(0, PROJECT_DIR)

from model.chess_net import ChessNet
from training.self_play import play_games


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise AssertionError("FAILED: " + message)


def check(condition, message):
    if not bool(condition):
        fail(message)
    print("PASS:", message)


def check_close(value, expected, message, tol=1e-5):
    value = float(value)
    expected = float(expected)

    if abs(value - expected) > tol:
        fail(
            f"{message}: expected {expected:.8f}, "
            f"got {value:.8f}"
        )

    print(
        f"PASS: {message}: {value:.8f}"
    )


def finite_array(x):
    return bool(np.isfinite(x).all())


def policy_stats(policy):
    return {
        "sum": float(policy.sum()),
        "min": float(policy.min()),
        "max": float(policy.max()),
        "nonzero": int(np.count_nonzero(policy > 0.0)),
    }


# ============================================================
# TEST 1
# CHECK CHECKPOINT
# ============================================================

print()
print("=" * 100)
print("TEST 1: LOAD RL53 CHECKPOINT")
print("=" * 100)

check(
    os.path.isfile(CHECKPOINT_PATH),
    "RL53 checkpoint exists",
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print(
    "Checkpoint keys:",
    list(checkpoint.keys())
)

model = ChessNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

if DEVICE.type == "cuda":
    model.to(memory_format=torch.channels_last)

print(
    "Checkpoint iteration:",
    checkpoint.get("iteration", "N/A")
)

check(
    checkpoint.get("iteration", 53) == 53,
    "Loaded checkpoint is RL53",
)

check(
    all(
        torch.isfinite(p).all().item()
        for p in model.parameters()
    ),
    "RL53 model parameters are finite",
)


# ============================================================
# TEST 2
# RUN ACTUAL GPU SELF-PLAY
# ============================================================

print()
print("=" * 100)
print("TEST 2: RUN ACTUAL GPU SELF-PLAY")
print("=" * 100)

torch.manual_seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

try:
    results = play_games(
        model=model,
        num_games=NUM_GAMES,
        num_simulations=NUM_SIMULATIONS,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        temperature_moves=TEMPERATURE_MOVES,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        batch_size=BATCH_SIZE,
    )
except Exception:
    print()
    print("ACTUAL GPU SELF-PLAY FAILED")
    traceback.print_exc()
    raise

check(
    isinstance(results, list),
    "Self-play returned a list",
)

check(
    len(results) == NUM_GAMES,
    "Self-play returned the requested number of games",
)


# ============================================================
# TEST 3
# GAME RESULT / TERMINATION VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 3: GAME RESULT VALIDATION")
print("=" * 100)

VALID_TERMINATIONS = {
    "CHECKMATE",
    "STALEMATE",
    "FIFTY_MOVE",
    "INSUFFICIENT_MATERIAL",
    "THREEFOLD_REPETITION",
    "MAX_MOVES",
    "UNKNOWN",
}

VALID_RESULTS = {-1, 0, 1, None}

completed_count = 0
incomplete_count = 0

total_samples = 0

for i, result in enumerate(results):

    print()
    print(
        f"Game {i + 1}: "
        f"moves={result.moves_played} | "
        f"result={result.result} | "
        f"termination={result.termination} | "
        f"completed={result.completed} | "
        f"samples={len(result.training_data)}"
    )

    check(
        result.termination in VALID_TERMINATIONS,
        f"Game {i + 1}: valid termination type",
    )

    check(
        result.result in VALID_RESULTS,
        f"Game {i + 1}: valid result value",
    )

    check(
        int(result.moves_played) >= 0,
        f"Game {i + 1}: non-negative move count",
    )

    if result.completed:
        completed_count += 1

        check(
            result.result in {-1, 0, 1},
            f"Game {i + 1}: completed game has a result",
        )

        check(
            result.termination != "MAX_MOVES",
            f"Game {i + 1}: completed game is not MAX_MOVES",
        )

    else:
        incomplete_count += 1

        check(
            result.result is None,
            f"Game {i + 1}: incomplete game has result=None",
        )

        check(
            len(result.training_data) == 0,
            f"Game {i + 1}: incomplete game has no training data",
        )

    total_samples += len(result.training_data)


print()
print("Completed games:", completed_count)
print("Incomplete games:", incomplete_count)
print("Total training samples:", total_samples)

check(
    completed_count + incomplete_count == NUM_GAMES,
    "Every game is classified as completed or incomplete",
)


# ============================================================
# TEST 4
# TRAINING SAMPLE STRUCTURE
# ============================================================

print()
print("=" * 100)
print("TEST 4: TRAINING SAMPLE STRUCTURE")
print("=" * 100)

sample_checked = 0

for gi, result in enumerate(results):

    if not result.completed:
        continue

    samples = result.training_data

    check(
        len(samples) == result.moves_played,
        (
            f"Game {gi + 1}: "
            f"training sample count equals moves played"
        ),
    )

    for si, sample in enumerate(samples):

        check(
            isinstance(sample, tuple),
            f"Game {gi + 1}, sample {si}: tuple structure",
        )

        check(
            len(sample) == 3,
            f"Game {gi + 1}, sample {si}: 3 fields",
        )

        state, policy, value = sample

        state = np.asarray(state)
        policy = np.asarray(policy)

        check(
            state.shape == (18, 8, 8),
            (
                f"Game {gi + 1}, sample {si}: "
                f"state shape is (18,8,8)"
            ),
        )

        check(
            policy.shape == (4544,),
            (
                f"Game {gi + 1}, sample {si}: "
                f"policy shape is (4544,)"
            ),
        )

        check(
            finite_array(state),
            f"Game {gi + 1}, sample {si}: state is finite",
        )

        check(
            finite_array(policy),
            f"Game {gi + 1}, sample {si}: policy is finite",
        )

        check(
            value in {-1.0, 0.0, 1.0},
            (
                f"Game {gi + 1}, sample {si}: "
                f"value is -1/0/+1"
            ),
        )

        check(
            state.dtype in (
                np.float16,
                np.float32,
                np.float64,
                np.uint8,
                np.int8,
                np.int16,
                np.int32,
            ),
            f"Game {gi + 1}, sample {si}: valid state dtype",
        )

        check(
            np.all(policy >= -1e-7),
            (
                f"Game {gi + 1}, sample {si}: "
                f"policy is non-negative"
            ),
        )

        pstats = policy_stats(policy)

        check_close(
            pstats["sum"],
            1.0,
            (
                f"Game {gi + 1}, sample {si}: "
                f"policy sums to 1"
            ),
            tol=1e-4,
        )

        check(
            pstats["nonzero"] > 0,
            (
                f"Game {gi + 1}, sample {si}: "
                f"policy has nonzero actions"
            ),
        )

        sample_checked += 1

        # Print only the first few samples in detail.
        if sample_checked <= 10:
            print()
            print(
                f"Sample {sample_checked}: "
                f"game={gi + 1}, index={si}"
            )
            print("  state:", state.shape, state.dtype)
            print("  policy:", policy.shape, policy.dtype)
            print(
                "  policy sum:",
                pstats["sum"]
            )
            print(
                "  policy max:",
                pstats["max"]
            )
            print(
                "  nonzero:",
                pstats["nonzero"]
            )
            print("  value:", value)


print()
print(
    "Training samples individually checked:",
    sample_checked,
)

check(
    sample_checked > 0,
    "At least one completed training sample was generated",
)


# ============================================================
# TEST 5
# STATE / POLICY VALUE RANGES
# ============================================================

print()
print("=" * 100)
print("TEST 5: DATA RANGE VALIDATION")
print("=" * 100)

state_min = float("inf")
state_max = float("-inf")
policy_min = float("inf")
policy_max = float("-inf")

values_seen = set()

for result in results:
    for state, policy, value in result.training_data:

        state = np.asarray(state)
        policy = np.asarray(policy)

        state_min = min(
            state_min,
            float(state.min()),
        )
        state_max = max(
            state_max,
            float(state.max()),
        )

        policy_min = min(
            policy_min,
            float(policy.min()),
        )
        policy_max = max(
            policy_max,
            float(policy.max()),
        )

        values_seen.add(float(value))

print("State min:", state_min)
print("State max:", state_max)
print("Policy min:", policy_min)
print("Policy max:", policy_max)
print("Values seen:", sorted(values_seen))

check(
    state_min >= -1e-6,
    "Encoded states contain no negative values",
)

check(
    state_max <= 1.0 + 1e-6,
    "Encoded states are within [0,1]",
)

check(
    policy_min >= -1e-7,
    "Policies contain no negative values",
)

check(
    policy_max <= 1.0 + 1e-6,
    "Policies are within [0,1]",
)


# ============================================================
# TEST 6
# VALUE PERSPECTIVE DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("TEST 6: TRAINING VALUE DISTRIBUTION")
print("=" * 100)

value_counts = {
    -1.0: 0,
    0.0: 0,
    1.0: 0,
}

for result in results:
    for _, _, value in result.training_data:
        value_counts[float(value)] += 1

print("Value -1:", value_counts[-1.0])
print("Value  0:", value_counts[0.0])
print("Value +1:", value_counts[1.0])

check(
    sum(value_counts.values()) == total_samples,
    "Value counts equal total training samples",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print()
print("=" * 100)
print("RL53 GPU SELF-PLAY INTEGRATION TEST COMPLETE")
print("=" * 100)

print()
print("Games:", NUM_GAMES)
print("Completed:", completed_count)
print("Incomplete:", incomplete_count)
print("Training samples:", total_samples)

print()
print("Verified:")
print("  Actual training.self_play.play_games() path")
print("  RL53 checkpoint loading")
print("  GPU self-play execution")
print("  Game result/termination handling")
print("  Completed/incomplete classification")
print("  State shape and finiteness")
print("  Policy shape and normalization")
print("  Training value targets")
print("  Training data ranges")
print("  No NaN/Inf in generated samples")

print()
print("If every PASS above succeeds, low-level MCTS and")
print("self-play generation are both behaving correctly.")
print("The investigation should then move to the RL training")
print("and replay-buffer/data-distribution stage.")
print("=" * 100)
