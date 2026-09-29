# ============================================================
# RL53 GPU SELF-PLAY INTEGRATION TEST
# ============================================================
#
# Tests the ACTUAL GPU self-play function currently present in
# training/self_play.py:
#
#     _play_games_gpu(...)
#
# This avoids the play_games import mismatch.
#
# Checks:
#   1. RL53 checkpoint loads correctly
#   2. GPU self-play actually runs
#   3. Requested games are returned
#   4. Game termination/result fields are valid
#   5. Completed games produce training samples
#   6. Incomplete MAX_MOVES games produce no training data
#   7. States have shape (18, 8, 8)
#   8. Policies have shape (4544,)
#   9. Policies are finite, non-negative and sum to 1
#  10. Values are -1 / 0 / +1
#  11. State/policy data contains no NaN/Inf
#  12. Training-data count matches moves played
#
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

BATCH_SIZE = 128

SEED = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("RL53 GPU SELF-PLAY INTEGRATION TEST")
print("=" * 100)

print()
print("Project:", PROJECT_DIR)
print("Checkpoint:", CHECKPOINT_PATH)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
else:
    DEVICE = torch.device("cpu")
    print("Device:", DEVICE)

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
# PROJECT IMPORTS
# ============================================================

sys.path.insert(0, PROJECT_DIR)

from model.chess_net import ChessNet

# IMPORTANT:
# Your current self_play.py exposes _play_games_gpu()
# rather than the play_games() wrapper.
from training.self_play import _play_games_gpu


# ============================================================
# HELPERS
# ============================================================

def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAILED: " + message)

    print("PASS:", message)


def check_close(value, expected, message, tolerance=1e-4):
    value = float(value)
    expected = float(expected)

    if abs(value - expected) > tolerance:
        raise AssertionError(
            f"FAILED: {message}: "
            f"expected {expected:.8f}, got {value:.8f}"
        )

    print(
        f"PASS: {message}: "
        f"{value:.8f}"
    )


def is_finite(x):
    return bool(np.isfinite(x).all())


def policy_stats(policy):
    policy = np.asarray(policy)

    return {
        "sum": float(policy.sum()),
        "min": float(policy.min()),
        "max": float(policy.max()),
        "nonzero": int(np.count_nonzero(policy > 0.0)),
    }


# ============================================================
# TEST 1
# CHECK ENVIRONMENT
# ============================================================

print()
print("=" * 100)
print("TEST 1: ENVIRONMENT")
print("=" * 100)

check(
    torch.cuda.is_available(),
    "CUDA is available",
)

check(
    DEVICE.type == "cuda",
    "Test is running on CUDA",
)

check(
    os.path.isdir(PROJECT_DIR),
    "Project directory exists",
)

check(
    os.path.isfile(CHECKPOINT_PATH),
    "RL53 checkpoint exists",
)


# ============================================================
# TEST 2
# LOAD RL53 CHECKPOINT
# ============================================================

print()
print("=" * 100)
print("TEST 2: LOAD RL53 CHECKPOINT")
print("=" * 100)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print()
print("Checkpoint keys:")
for key in checkpoint.keys():
    print(" ", key)

model = ChessNet().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

if DEVICE.type == "cuda":
    model.to(memory_format=torch.channels_last)

iteration = checkpoint.get("iteration", None)

print()
print("Checkpoint iteration:", iteration)

if iteration is not None:
    check(
        int(iteration) == 53,
        "Loaded checkpoint is RL53",
    )

for name, parameter in model.named_parameters():
    check(
        bool(torch.isfinite(parameter).all().item()),
        f"Model parameter finite: {name}",
    )

print()
print("RL53 model loaded successfully.")


# ============================================================
# TEST 3
# RUN ACTUAL GPU SELF-PLAY
# ============================================================

print()
print("=" * 100)
print("TEST 3: ACTUAL GPU SELF-PLAY")
print("=" * 100)

print()
print("Calling:")
print("    training.self_play._play_games_gpu()")
print()

# Reproducibility
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:

    results = _play_games_gpu(
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
    print("=" * 100)
    print("GPU SELF-PLAY CRASHED")
    print("=" * 100)

    traceback.print_exc()

    raise


# ============================================================
# TEST 4
# BASIC RETURN VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 4: RETURN VALUE")
print("=" * 100)

check(
    isinstance(results, list),
    "Self-play returned a list",
)

check(
    len(results) == NUM_GAMES,
    "Returned exactly the requested number of games",
)


# ============================================================
# TEST 5
# GAME RESULT VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 5: GAME RESULT VALIDATION")
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

VALID_RESULTS = {
    -1,
    0,
    1,
    None,
}

completed_games = 0
incomplete_games = 0
total_samples = 0

for game_index, result in enumerate(results):

    game_number = game_index + 1

    print()
    print(
        f"Game {game_number}:"
    )
    print(
        f"  moves       = {result.moves_played}"
    )
    print(
        f"  result      = {result.result}"
    )
    print(
        f"  termination = {result.termination}"
    )
    print(
        f"  completed   = {result.completed}"
    )
    print(
        f"  samples     = {len(result.training_data)}"
    )

    check(
        result.termination in VALID_TERMINATIONS,
        f"Game {game_number}: valid termination",
    )

    check(
        result.result in VALID_RESULTS,
        f"Game {game_number}: valid result",
    )

    check(
        int(result.moves_played) >= 0,
        f"Game {game_number}: non-negative move count",
    )

    if result.completed:

        completed_games += 1

        check(
            result.result in {-1, 0, 1},
            f"Game {game_number}: completed game has valid result",
        )

        check(
            result.termination != "MAX_MOVES",
            f"Game {game_number}: completed game did not hit MAX_MOVES",
        )

    else:

        incomplete_games += 1

        check(
            result.result is None,
            f"Game {game_number}: incomplete result is None",
        )

        check(
            len(result.training_data) == 0,
            f"Game {game_number}: incomplete game has no training data",
        )

    total_samples += len(result.training_data)


print()
print("Completed games:", completed_games)
print("Incomplete games:", incomplete_games)
print("Total samples:", total_samples)

check(
    completed_games + incomplete_games == NUM_GAMES,
    "Every game is classified",
)


# ============================================================
# TEST 6
# TRAINING SAMPLE COUNT
# ============================================================

print()
print("=" * 100)
print("TEST 6: TRAINING SAMPLE COUNT")
print("=" * 100)

for game_index, result in enumerate(results):

    game_number = game_index + 1

    if not result.completed:
        continue

    check(
        len(result.training_data) == result.moves_played,
        (
            f"Game {game_number}: "
            f"samples == moves played"
        ),
    )


# ============================================================
# TEST 7
# TRAINING SAMPLE CONTENT
# ============================================================

print()
print("=" * 100)
print("TEST 7: TRAINING SAMPLE CONTENT")
print("=" * 100)

samples_checked = 0

for game_index, result in enumerate(results):

    game_number = game_index + 1

    if not result.completed:
        continue

    for sample_index, sample in enumerate(
        result.training_data
    ):

        check(
            isinstance(sample, tuple),
            f"Game {game_number}, sample {sample_index}: tuple",
        )

        check(
            len(sample) == 3,
            f"Game {game_number}, sample {sample_index}: 3 fields",
        )

        state, policy, value = sample

        state = np.asarray(state)
        policy = np.asarray(policy)

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        check(
            state.shape == (18, 8, 8),
            (
                f"Game {game_number}, sample {sample_index}: "
                f"state shape = (18,8,8)"
            ),
        )

        check(
            is_finite(state),
            (
                f"Game {game_number}, sample {sample_index}: "
                f"state finite"
            ),
        )

        # ----------------------------------------------------
        # POLICY
        # ----------------------------------------------------

        check(
            policy.shape == (4544,),
            (
                f"Game {game_number}, sample {sample_index}: "
                f"policy shape = (4544,)"
            ),
        )

        check(
            is_finite(policy),
            (
                f"Game {game_number}, sample {sample_index}: "
                f"policy finite"
            ),
        )

        check(
            bool(np.all(policy >= -1e-7)),
            (
                f"Game {game_number}, sample {sample_index}: "
                f"policy non-negative"
            ),
        )

        stats = policy_stats(policy)

        check_close(
            stats["sum"],
            1.0,
            (
                f"Game {game_number}, sample {sample_index}: "
                f"policy sums to 1"
            ),
            tolerance=1e-4,
        )

        check(
            stats["nonzero"] > 0,
            (
                f"Game {game_number}, sample {sample_index}: "
                f"policy has nonzero actions"
            ),
        )

        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------

        check(
            float(value) in {-1.0, 0.0, 1.0},
            (
                f"Game {game_number}, sample {sample_index}: "
                f"value is -1/0/+1"
            ),
        )

        samples_checked += 1

        # Print details for first 10 samples only.
        if samples_checked <= 10:

            print()
            print(
                f"Sample {samples_checked}"
            )
            print(
                "  game:",
                game_number
            )
            print(
                "  index:",
                sample_index
            )
            print(
                "  state shape:",
                state.shape
            )
            print(
                "  state dtype:",
                state.dtype
            )
            print(
                "  state min:",
                float(state.min())
            )
            print(
                "  state max:",
                float(state.max())
            )
            print(
                "  policy shape:",
                policy.shape
            )
            print(
                "  policy dtype:",
                policy.dtype
            )
            print(
                "  policy sum:",
                stats["sum"]
            )
            print(
                "  policy max:",
                stats["max"]
            )
            print(
                "  policy nonzero:",
                stats["nonzero"]
            )
            print(
                "  value:",
                value
            )


print()
print(
    "Training samples checked:",
    samples_checked
)

check(
    samples_checked > 0,
    "At least one training sample exists",
)


# ============================================================
# TEST 8
# GLOBAL DATA RANGE
# ============================================================

print()
print("=" * 100)
print("TEST 8: GLOBAL DATA RANGE")
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
            float(state.min())
        )

        state_max = max(
            state_max,
            float(state.max())
        )

        policy_min = min(
            policy_min,
            float(policy.min())
        )

        policy_max = max(
            policy_max,
            float(policy.max())
        )

        values_seen.add(float(value))


print("State min:", state_min)
print("State max:", state_max)

print("Policy min:", policy_min)
print("Policy max:", policy_max)

print("Values seen:", sorted(values_seen))

check(
    state_min >= -1e-6,
    "States contain no negative values",
)

check(
    state_max <= 1.0 + 1e-6,
    "States are within [0,1]",
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
# TEST 9
# VALUE DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("TEST 9: VALUE TARGET DISTRIBUTION")
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
    "Value counts match total training samples",
)


# ============================================================
# TEST 10
# CUDA CLEANUP / FINAL CHECK
# ============================================================

print()
print("=" * 100)
print("TEST 10: CUDA FINAL CHECK")
print("=" * 100)

if torch.cuda.is_available():

    torch.cuda.synchronize()

    allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    reserved_mb = (
        torch.cuda.memory_reserved()
        / 1024**2
    )

    print(
        f"CUDA allocated: {allocated_mb:.2f} MB"
    )

    print(
        f"CUDA reserved:  {reserved_mb:.2f} MB"
    )

    check(
        torch.cuda.is_available(),
        "CUDA remains available after self-play",
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("RL53 GPU SELF-PLAY INTEGRATION TEST COMPLETE")
print("=" * 100)

print()
print("Games requested:", NUM_GAMES)
print("Games completed:", completed_games)
print("Games incomplete:", incomplete_games)
print("Training samples:", total_samples)

print()
print("Verified:")
print("  [OK] RL53 checkpoint loading")
print("  [OK] CUDA execution")
print("  [OK] Actual _play_games_gpu() path")
print("  [OK] Game result handling")
print("  [OK] Game termination handling")
print("  [OK] Completed/incomplete handling")
print("  [OK] Training sample count")
print("  [OK] State shape: 18 x 8 x 8")
print("  [OK] Policy shape: 4544")
print("  [OK] Policy normalization")
print("  [OK] Policy non-negativity")
print("  [OK] Value targets")
print("  [OK] NaN/Inf checks")
print("  [OK] Data ranges")
print("  [OK] CUDA final state")

print()
print("=" * 100)
print("NEXT STEP:")
print("If all tests PASS, low-level MCTS + actual GPU self-play")
print("are validated. Then we move to the RL training/replay")
print("distribution side of the investigation.")
print("=" * 100)
