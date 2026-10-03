"""GPU-first reinforcement-learning training for the chess AI.

Usage on Kaggle:
    python training/rl_training.py

Set RL_ITERATION to the next number in the chain. The script loads the prior
iteration, runs GPU-native self-play/MCTS, appends completed games to the replay
buffer, and trains the unchanged 18x8x8 -> 4544 policy/value network on CUDA.

Memory policy:
    - Replay buffer remains in CPU RAM.
    - Only one training batch is transferred to CUDA at a time.
    - The complete replay buffer is never materialized on the GPU.
"""

from __future__ import annotations

import os
import sys
import random
import re

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from training.self_play import play_games_multi_gpu
from training.replay_buffer import ReplayBuffer
from training.train_step import train_from_replay_buffer


# ---------------------------------------------------------------------------
# Iteration / paths
# ---------------------------------------------------------------------------

RL_ITERATION = 57

PREVIOUS_ITERATION = RL_ITERATION - 1

KAGGLE_CHECKPOINT_ROOT = (
    "/kaggle/working/checkpoints"
)

LOCAL_CHECKPOINT_ROOT = "checkpoints"

PREVIOUS_CHECKPOINT = os.path.join(
    KAGGLE_CHECKPOINT_ROOT,
    f"rl_iteration_{PREVIOUS_ITERATION}.pt",
)

PREVIOUS_REPLAY_BUFFER = os.path.join(
    KAGGLE_CHECKPOINT_ROOT,
    f"replay_buffer_rl{PREVIOUS_ITERATION}.pt",
)

OUTPUT_CHECKPOINT = os.path.join(
    LOCAL_CHECKPOINT_ROOT,
    f"rl_iteration_{RL_ITERATION}.pt",
)

OUTPUT_REPLAY_BUFFER = os.path.join(
    LOCAL_CHECKPOINT_ROOT,
    f"replay_buffer_rl{RL_ITERATION}.pt",
)


# ---------------------------------------------------------------------------
# Self-play configuration
# ---------------------------------------------------------------------------

NUM_SELF_PLAY_GAMES = 256   

NUM_SIMULATIONS = 100

MAX_MOVES = 400

MCTS_BATCH_SIZE = 64

TEMPERATURE = 1.0

TEMPERATURE_MOVES = 60

DIRICHLET_ALPHA = 0.3

DIRICHLET_EPSILON = 0.25


# ---------------------------------------------------------------------------
# Replay / training configuration
# ---------------------------------------------------------------------------

REPLAY_BUFFER_CAPACITY = 150000

TRAINING_BATCH_SIZE = 256

TRAINING_STEPS = 1200

LEARNING_RATE = 5e-5

SEED = 42


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def configure_device():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":

        torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True

        torch.set_float32_matmul_precision("high")

    return device


device = configure_device()


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def create_model():

    encoder = ActionEncoder()

    model = ChessNet(
        action_space_size=encoder.size()
    ).to(device)

    if device.type == "cuda":
        model.to(
            memory_format=torch.channels_last
        )

    return model, encoder


def create_optimizer(model):

    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(path):

    if os.path.exists(path):
        return path

    local = os.path.join(
        LOCAL_CHECKPOINT_ROOT,
        os.path.basename(path),
    )

    if os.path.exists(local):
        return local

    raise FileNotFoundError(
        f"Required file not found: {path}"
    )


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_previous_checkpoint(
    model,
    optimizer,
    path,
):

    path = _resolve_path(path)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_iteration = checkpoint.get("iteration")
    if checkpoint_iteration is None:
        raise RuntimeError(
            "Previous checkpoint is missing its iteration metadata; "
            f"expected RL{PREVIOUS_ITERATION}."
        )
    if int(checkpoint_iteration) != PREVIOUS_ITERATION:
        raise RuntimeError(
            "Checkpoint iteration does not match the configured predecessor: "
            f"got RL{checkpoint_iteration}, expected RL{PREVIOUS_ITERATION}."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )

    if optimizer_state is not None:

        optimizer.load_state_dict(
            optimizer_state
        )

        print(
            "Loaded previous optimizer state."
        )

    else:

        print(
            "Warning: previous checkpoint has no "
            "optimizer state; using a fresh Adam optimizer."
        )

    # Keep this iteration's configured LR.
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE

    model.to(device)

    if device.type == "cuda":

        model.to(
            memory_format=torch.channels_last
        )

    model.eval()

    return checkpoint, path


# ---------------------------------------------------------------------------
# Load replay buffer
# ---------------------------------------------------------------------------

def load_previous_replay_buffer(path):

    path = _resolve_path(path)

    match = re.fullmatch(r"replay_buffer_rl(\d+)\.pt", os.path.basename(path))
    if match is None:
        raise RuntimeError(
            "Replay buffer filename must identify its source iteration: "
            f"{path}"
        )
    if int(match.group(1)) != PREVIOUS_ITERATION:
        raise RuntimeError(
            "Replay buffer iteration does not match the configured predecessor: "
            f"got RL{match.group(1)}, expected RL{PREVIOUS_ITERATION}."
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, list):

        raise RuntimeError(
            "Invalid replay buffer format: expected a list."
        )

    replay = ReplayBuffer(
        REPLAY_BUFFER_CAPACITY
    )

    replay.add(data)

    # The deque now owns the sample references.
    del data

    provenance = replay.validate_provenance(
        expected_max_iteration=PREVIOUS_ITERATION,
    )
    print("Replay provenance validated:", provenance)

    return replay, path


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def generate_self_play_data(
    model,
    replay_buffer,
    checkpoint_path,
):

    print("\n==============================")
    print("GPU SELF-PLAY")
    print("==============================")

    print(
        f"Games: {NUM_SELF_PLAY_GAMES}"
    )

    print(
        f"Simulations/game: {NUM_SIMULATIONS}"
    )

    print(
        f"Max moves: {MAX_MOVES}"
    )

    results = play_games_multi_gpu(
        model=model,
        checkpoint_path=checkpoint_path,
        num_games=NUM_SELF_PLAY_GAMES,
        num_simulations=NUM_SIMULATIONS,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        temperature_moves=TEMPERATURE_MOVES,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        batch_size=MCTS_BATCH_SIZE,
    )

    completed = 0

    incomplete = 0

    white_wins = 0

    black_wins = 0

    draws = 0

    new_samples = 0

    termination_counts = {}

    for result in results:

        termination_counts[
            result.termination
        ] = (
            termination_counts.get(
                result.termination,
                0,
            )
            + 1
        )

        if not result.completed:

            incomplete += 1

            continue

        completed += 1

        replay_buffer.add(
            result.training_data,
            source_iteration=RL_ITERATION,
        )

        new_samples += len(
            result.training_data
        )

        if result.result == 1:

            white_wins += 1

        elif result.result == -1:

            black_wins += 1

        else:

            draws += 1

    stats = {

        "completed_games": completed,

        "incomplete_games": incomplete,

        "white_wins": white_wins,

        "black_wins": black_wins,

        "draws": draws,

        "new_samples": new_samples,

        "termination_counts": termination_counts,

        "replay_buffer_size": len(
            replay_buffer
        ),
    }

    print(
        "\nSelf-play summary:",
        stats,
    )

    # Worker results can be large. Release them before replay saving/training.
    del results

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return stats


# ---------------------------------------------------------------------------
# Save replay buffer
# ---------------------------------------------------------------------------

def save_replay_buffer(
    replay_buffer,
    path,
):

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    print(
        "Saving replay buffer:",
        path,
    )

    print(
        "Replay samples:",
        len(replay_buffer),
    )

    torch.save(
        list(replay_buffer),
        path,
    )

    print(
        "Replay buffer saved:",
        path,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    optimizer,
    replay_buffer,
):

    if len(replay_buffer) < TRAINING_BATCH_SIZE:

        raise RuntimeError(
            f"Replay buffer has {len(replay_buffer)} samples; "
            f"need {TRAINING_BATCH_SIZE}."
        )

    print("\n==============================")
    print("GPU TRAINING")
    print("==============================")

    print(
        f"CPU replay samples: {len(replay_buffer)}"
    )

    print(
        f"GPU batch size: {TRAINING_BATCH_SIZE}"
    )

    print(
        "Only one training batch is transferred "
        "to GPU at a time."
    )

    return train_from_replay_buffer(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=TRAINING_BATCH_SIZE,
        training_steps=TRAINING_STEPS,
    )


# ---------------------------------------------------------------------------
# Save RL checkpoint
# ---------------------------------------------------------------------------

def save_rl_checkpoint(
    model,
    optimizer,
    replay_buffer,
    self_play_stats,
    training_stats,
):

    os.makedirs(
        os.path.dirname(OUTPUT_CHECKPOINT)
        or ".",
        exist_ok=True,
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "iteration":
            RL_ITERATION,

        "previous_checkpoint":
            PREVIOUS_CHECKPOINT,

        "replay_buffer_provenance": {
            "filename": os.path.basename(OUTPUT_REPLAY_BUFFER),
            "sample_count": len(replay_buffer),
            "source_distribution": dict(replay_buffer.iteration_distribution()),
        },

        **self_play_stats,

        "num_simulations":
            NUM_SIMULATIONS,

        "mcts_batch_size":
            MCTS_BATCH_SIZE,

        "temperature":
            TEMPERATURE,

        "temperature_moves":
            TEMPERATURE_MOVES,

        "dirichlet_alpha":
            DIRICHLET_ALPHA,

        "dirichlet_epsilon":
            DIRICHLET_EPSILON,

        "max_moves":
            MAX_MOVES,

        "training_batch_size":
            TRAINING_BATCH_SIZE,

        "training_steps":
            TRAINING_STEPS,

        "learning_rate":
            LEARNING_RATE,

        **training_stats,
    }

    torch.save(
        checkpoint,
        OUTPUT_CHECKPOINT,
    )

    print(
        "Model checkpoint saved:",
        OUTPUT_CHECKPOINT,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if device.type == "cuda":

        torch.cuda.manual_seed_all(
            SEED
        )

    print(
        "=========================================="
    )

    print(
        "GPU RL TRAINING"
    )

    print(
        "=========================================="
    )

    print(
        "Iteration:",
        RL_ITERATION,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

        print(
            "FP16 AMP: enabled"
        )

        print(
            "Channels last: enabled"
        )

        print(
            "TF32: enabled"
        )

    print(
        "Previous checkpoint:",
        PREVIOUS_CHECKPOINT,
    )

    print(
        "Previous replay buffer:",
        PREVIOUS_REPLAY_BUFFER,
    )

    print(
        "Output checkpoint:",
        OUTPUT_CHECKPOINT,
    )

    print(
        "Output replay buffer:",
        OUTPUT_REPLAY_BUFFER,
    )

    model, encoder = create_model()

    optimizer = create_optimizer(
        model
    )

    print(
        "Action space size:",
        encoder.size(),
    )

    checkpoint, resolved_checkpoint = (
        load_previous_checkpoint(
            model,
            optimizer,
            PREVIOUS_CHECKPOINT,
        )
    )

    print(
        "Loaded previous checkpoint:",
        resolved_checkpoint,
    )

    print(
        "Previous iteration:",
        checkpoint.get("iteration"),
    )

    replay, resolved_replay = (
        load_previous_replay_buffer(
            PREVIOUS_REPLAY_BUFFER
        )
    )

    print(
        "Loaded replay buffer:",
        resolved_replay,
    )

    print(
        "Previous samples:",
        len(replay),
    )

    # Show the source-iteration distribution before new self-play data is added.
    print("\nReplay buffer before self-play:")
    replay.print_iteration_distribution()

    # -------------------------------------------------------
    # SELF-PLAY
    # -------------------------------------------------------

    self_play_stats = (
        generate_self_play_data(
            model,
            replay,
            resolved_checkpoint,
        )
    )

    # -------------------------------------------------------
    # SAVE REPLAY
    # -------------------------------------------------------

    # Show exactly what survived FIFO after adding this iteration's samples.
    print("\nReplay buffer after self-play:")
    replay.print_iteration_distribution()

    save_replay_buffer(
        replay,
        OUTPUT_REPLAY_BUFFER,
    )

    # -------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------

    training_stats = train_model(
        model,
        optimizer,
        replay,
    )

    # -------------------------------------------------------
    # SAVE CHECKPOINT
    # -------------------------------------------------------

    save_rl_checkpoint(
        model,
        optimizer,
        replay,
        self_play_stats,
        training_stats,
    )

    print(
        "\n=========================================="
    )

    print(
        f"RL ITERATION {RL_ITERATION} COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
