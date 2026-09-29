# ============================================================
# RL53 SELF-PLAY INTEGRATION TEST - ROBUST ENTRYPOINT VERSION
# ============================================================
#
# IMPORTANT:
# Your Kaggle training/self_play.py is NOT the same revision as
# the copied source we previously inspected. Therefore this test
# does NOT assume that play_games() or _play_games_gpu() exists.
#
# It imports the ACTUAL training/self_play.py on Kaggle and detects
# the available self-play entrypoint.
#
# Preferred order:
#   1. play_games_multi_gpu
#   2. _play_games_gpu
#   3. play_games
#
# The test then validates the returned SelfPlayResult objects.
#
# ============================================================

import os
import sys
import traceback
import inspect
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
# PROJECT PATH
# ============================================================

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# TEST 1
# ENVIRONMENT
# ============================================================

print()
print("=" * 100)
print("TEST 1: ENVIRONMENT")
print("=" * 100)

assert torch.cuda.is_available(), "CUDA is required for this test."

assert os.path.isdir(
    PROJECT_DIR
), f"Project directory not found: {PROJECT_DIR}"

assert os.path.isfile(
    CHECKPOINT_PATH
), f"Checkpoint not found: {CHECKPOINT_PATH}"

print("PASS: CUDA available")
print("PASS: Project directory exists")
print("PASS: RL53 checkpoint exists")


# ============================================================
# TEST 2
# LOAD THE ACTUAL self_play MODULE
# ============================================================

print()
print("=" * 100)
print("TEST 2: ACTUAL training.self_play MODULE")
print("=" * 100)

# Import the module itself rather than importing a guessed function.
import training.self_play as self_play

print()
print("Loaded module:")
print(self_play.__file__)

assert os.path.abspath(self_play.__file__) == os.path.abspath(
    os.path.join(PROJECT_DIR, "training", "self_play.py")
), (
    "Python loaded a different self_play.py than the Kaggle project file."
)

print()
print("Available self-play callables:")

candidate_names = [
    "play_games_multi_gpu",
    "_play_games_gpu",
    "play_games",
]

available = []

for name in candidate_names:
    obj = getattr(self_play, name, None)

    if callable(obj):
        available.append(name)

        print()
        print(f"FOUND: {name}")
        try:
            print(
                inspect.signature(obj)
            )
        except Exception:
            print("Signature unavailable.")
    else:
        print(f"NOT FOUND: {name}")


print()

assert available, (
    "No supported self-play entrypoint exists in the actual "
    "training/self_play.py. "
    f"Available module names containing 'play': "
    f"{[x for x in dir(self_play) if 'play' in x.lower()]}"
)

# Use the highest-level entrypoint available.
#
# play_games_multi_gpu is preferred because this is the function
# used by the multi-GPU RL training pipeline.
if "play_games_multi_gpu" in available:
    ENTRYPOINT_NAME = "play_games_multi_gpu"
    ENTRYPOINT = self_play.play_games_multi_gpu

elif "_play_games_gpu" in available:
    ENTRYPOINT_NAME = "_play_games_gpu"
    ENTRYPOINT = self_play._play_games_gpu

else:
    ENTRYPOINT_NAME = "play_games"
    ENTRYPOINT = self_play.play_games

print(
    "SELECTED ENTRYPOINT:",
    ENTRYPOINT_NAME
)


# ============================================================
# TEST 3
# LOAD RL53 MODEL
# ============================================================

print()
print("=" * 100)
print("TEST 3: LOAD RL53 MODEL")
print("=" * 100)

from model.chess_net import ChessNet

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
    assert int(iteration) == 53, (
        f"Expected RL53 checkpoint, got iteration {iteration}"
    )

print("PASS: RL53 model loaded")


# ============================================================
# TEST 4
# RUN THE ACTUAL SELF-PLAY ENTRYPOINT
# ============================================================

print()
print("=" * 100)
print("TEST 4: RUN ACTUAL SELF-PLAY")
print("=" * 100)

print()
print("Using:", ENTRYPOINT_NAME)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


try:

    if ENTRYPOINT_NAME == "play_games_multi_gpu":

        # This is the actual high-level multi-GPU function.
        results = ENTRYPOINT(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            num_games=NUM_GAMES,
            num_simulations=NUM_SIMULATIONS,
            max_moves=MAX_MOVES,
            temperature=TEMPERATURE,
            temperature_moves=TEMPERATURE_MOVES,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )

    else:

        # Direct GPU self-play fallback.
        results = ENTRYPOINT(
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
    print("SELF-PLAY FAILED")
    print("=" * 100)

    traceback.print_exc()

    raise


# ============================================================
# TEST 5
# BASIC RETURN VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 5: RETURN VALUE")
print("=" * 100)

assert isinstance(results, list), (
    f"Expected list, got {type(results)}"
)

assert len(results) == NUM_GAMES, (
    f"Expected {NUM_GAMES} games, got {len(results)}"
)

print(
    f"PASS: returned {len(results)} games"
)


# ============================================================
# TEST 6
# GAME RESULT VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 6: GAME RESULT VALIDATION")
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

for i, result in enumerate(results):

    game = i + 1

    print()
    print(
        f"Game {game}:"
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

    assert result.termination in VALID_TERMINATIONS, (
        f"Game {game}: invalid termination "
        f"{result.termination}"
    )

    assert result.result in VALID_RESULTS, (
        f"Game {game}: invalid result "
        f"{result.result}"
    )

    assert int(result.moves_played) >= 0, (
        f"Game {game}: negative move count"
    )

    if result.completed:

        completed_games += 1

        assert result.result in {-1, 0, 1}, (
            f"Game {game}: completed game has invalid result"
        )

        assert result.termination != "MAX_MOVES", (
            f"Game {game}: completed game marked MAX_MOVES"
        )

    else:

        incomplete_games += 1

        assert result.result is None, (
            f"Game {game}: incomplete result should be None"
        )

        assert len(result.training_data) == 0, (
            f"Game {game}: incomplete game contains training data"
        )

    total_samples += len(result.training_data)

    print(
        f"PASS: Game {game}"
    )


print()
print("Completed games:", completed_games)
print("Incomplete games:", incomplete_games)
print("Total training samples:", total_samples)


# ============================================================
# TEST 7
# SAMPLE COUNT
# ============================================================

print()
print("=" * 100)
print("TEST 7: TRAINING SAMPLE COUNT")
print("=" * 100)

for i, result in enumerate(results):

    game = i + 1

    if not result.completed:
        continue

    assert len(result.training_data) == result.moves_played, (
        f"Game {game}: "
        f"samples={len(result.training_data)} "
        f"but moves={result.moves_played}"
    )

    print(
        f"PASS: Game {game}: "
        f"{len(result.training_data)} samples "
        f"for {result.moves_played} moves"
    )


# ============================================================
# TEST 8
# SAMPLE SHAPES / VALUES / POLICIES
# ============================================================

print()
print("=" * 100)
print("TEST 8: TRAINING SAMPLE CONTENT")
print("=" * 100)

samples_checked = 0

for i, result in enumerate(results):

    game = i + 1

    for j, sample in enumerate(result.training_data):

        state, policy, value = sample

        state = np.asarray(state)
        policy = np.asarray(policy)

        assert state.shape == (
            18,
            8,
            8,
        ), (
            f"Game {game}, sample {j}: "
            f"bad state shape {state.shape}"
        )

        assert policy.shape == (
            4544,
        ), (
            f"Game {game}, sample {j}: "
            f"bad policy shape {policy.shape}"
        )

        assert np.isfinite(state).all(), (
            f"Game {game}, sample {j}: "
            f"state contains NaN/Inf"
        )

        assert np.isfinite(policy).all(), (
            f"Game {game}, sample {j}: "
            f"policy contains NaN/Inf"
        )

        assert np.all(policy >= -1e-7), (
            f"Game {game}, sample {j}: "
            f"negative policy value"
        )

        policy_sum = float(policy.sum())

        assert abs(policy_sum - 1.0) <= 1e-4, (
            f"Game {game}, sample {j}: "
            f"policy sum={policy_sum}"
        )

        assert int(np.count_nonzero(policy > 0.0)) > 0, (
            f"Game {game}, sample {j}: "
            f"policy has no nonzero actions"
        )

        assert float(value) in {
            -1.0,
            0.0,
            1.0,
        }, (
            f"Game {game}, sample {j}: "
            f"invalid value target {value}"
        )

        samples_checked += 1

        if samples_checked <= 10:

            print()
            print(
                f"Sample {samples_checked}"
            )

            print(
                "  game:",
                game
            )

            print(
                "  sample:",
                j
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
                "  state range:",
                float(state.min()),
                "to",
                float(state.max())
            )

            print(
                "  policy shape:",
                policy.shape
            )

            print(
                "  policy sum:",
                policy_sum
            )

            print(
                "  policy max:",
                float(policy.max())
            )

            print(
                "  policy nonzero:",
                int(np.count_nonzero(policy > 0.0))
            )

            print(
                "  value:",
                value
            )

            print(
                "PASS: sample"
            )


assert samples_checked > 0, (
    "No training samples were generated."
)

print()
print(
    "Total samples checked:",
    samples_checked
)


# ============================================================
# TEST 9
# GLOBAL DATA RANGE
# ============================================================

print()
print("=" * 100)
print("TEST 9: GLOBAL DATA RANGE")
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

        values_seen.add(
            float(value)
        )

print(
    "State range:",
    state_min,
    "to",
    state_max
)

print(
    "Policy range:",
    policy_min,
    "to",
    policy_max
)

print(
    "Values seen:",
    sorted(values_seen)
)

assert state_min >= -1e-6
assert state_max <= 1.0 + 1e-6

assert policy_min >= -1e-7
assert policy_max <= 1.0 + 1e-6

print("PASS: state range")
print("PASS: policy range")


# ============================================================
# TEST 10
# VALUE TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("TEST 10: VALUE TARGET DISTRIBUTION")
print("=" * 100)

value_counts = {
    -1.0: 0,
    0.0: 0,
    1.0: 0,
}

for result in results:

    for _, _, value in result.training_data:

        value_counts[
            float(value)
        ] += 1

print(
    "Value -1:",
    value_counts[-1.0]
)

print(
    "Value  0:",
    value_counts[0.0]
)

print(
    "Value +1:",
    value_counts[1.0]
)

assert sum(
    value_counts.values()
) == total_samples

print(
    "PASS: value counts match sample count"
)


# ============================================================
# FINAL CUDA CHECK
# ============================================================

print()
print("=" * 100)
print("FINAL CUDA CHECK")
print("=" * 100)

torch.cuda.synchronize()

print(
    "CUDA allocated:",
    f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
)

print(
    "CUDA reserved:",
    f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
)

print(
    "PASS: CUDA synchronized successfully"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("RL53 GPU SELF-PLAY INTEGRATION TEST PASSED")
print("=" * 100)

print()
print("Entrypoint:", ENTRYPOINT_NAME)
print("Games:", NUM_GAMES)
print("Completed:", completed_games)
print("Incomplete:", incomplete_games)
print("Training samples:", total_samples)

print()
print("Verified:")
print("  [PASS] Actual Kaggle training.self_play.py loaded")
print("  [PASS] Correct self-play entrypoint detected")
print("  [PASS] RL53 checkpoint")
print("  [PASS] CUDA execution")
print("  [PASS] Self-play execution")
print("  [PASS] Game termination/result handling")
print("  [PASS] Training sample count")
print("  [PASS] State shape 18x8x8")
print("  [PASS] Policy shape 4544")
print("  [PASS] Policy normalization")
print("  [PASS] Policy non-negativity")
print("  [PASS] Value targets")
print("  [PASS] NaN/Inf checks")
print("  [PASS] Data ranges")
print("  [PASS] CUDA synchronization")

print()
print("=" * 100)
