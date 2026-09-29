from __future__ import annotations

import os
import sys
import time
import re
import shutil
import tempfile
import multiprocessing as mp

import torch


# ==========================================================
# MAKE PROJECT ROOT IMPORTABLE
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORTS
# ==========================================================

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_GAMES = 300

NUM_SIMULATIONS = 100

MAX_MOVES = 400

# 0.0 = deterministic evaluation.
EVALUATION_TEMPERATURE = 0.25

ACTION_SPACE_SIZE = 4544


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# CHECKPOINTS
# ==========================================================

MODEL_A_CHECKPOINT = (
    "/kaggle/input/datasets/arjunthakur9999/checkpoints/chess_checkpoints (3)/rl_iteration_48.pt"
)




MODEL_B_CHECKPOINT = (
    "/kaggle/input/datasets/arjunthakur9999/checkpoints/chess_checkpoints (4)/rl_iteration_52.pt"
)


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model(path, device):

    model = ChessNet(
        action_space_size=ACTION_SPACE_SIZE
    ).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    if device.type == "cuda":

        model.to(
            memory_format=torch.channels_last
        )

    return model, checkpoint


# ==========================================================
# TERMINAL INFORMATION
# ==========================================================

def get_terminal_results(
    states: GPUChess,
):
    """
    Determine terminal status for a batch of GPU states.

    Returns:
        terminal: bool tensor [batch]
        results:  int tensor [batch]
                   +1 = White win
                    0 = Draw
                   -1 = Black win
        termination: list[str | None]
    """

    batch_size = states.pieces.shape[0]

    terminal = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=states.device
    )

    results = torch.zeros(
        batch_size,
        dtype=torch.int8,
        device=states.device
    )

    termination = [None] * batch_size

    # ------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------

    legal = states.legal_move_mask()

    has_legal_moves = legal.any(dim=1)

    in_check = states.is_in_check()

    checkmate = (
        ~has_legal_moves
        & in_check
    )

    stalemate = (
        ~has_legal_moves
        & ~in_check
    )

    # ------------------------------------------------------
    # Checkmate
    # ------------------------------------------------------

    if bool(checkmate.any().item()):

        terminal |= checkmate

        # turn=False -> White to move -> White is mated
        #               -> Black wins (-1)
        #
        # turn=True  -> Black to move -> Black is mated
        #               -> White wins (+1)

        checkmate_results = torch.where(
            states.turn,
            torch.ones_like(results),
            -torch.ones_like(results)
        )

        results = torch.where(
            checkmate,
            checkmate_results,
            results
        )

        indices = (
            torch.nonzero(
                checkmate,
                as_tuple=False
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

        for i in indices:
            termination[i] = "CHECKMATE"

    # ------------------------------------------------------
    # Stalemate
    # ------------------------------------------------------

    if bool(stalemate.any().item()):

        terminal |= stalemate

        indices = (
            torch.nonzero(
                stalemate,
                as_tuple=False
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

        for i in indices:
            termination[i] = "STALEMATE"

    # ------------------------------------------------------
    # Fifty-move rule
    # ------------------------------------------------------

    fifty_move = (
        states.halfmove_clock >= 100
    )

    # Only assign if not already terminal.
    fifty_new = (
        fifty_move
        & ~terminal
    )

    if bool(fifty_new.any().item()):

        terminal |= fifty_new

        indices = (
            torch.nonzero(
                fifty_new,
                as_tuple=False
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

        for i in indices:
            termination[i] = "FIFTY_MOVE"

    # ------------------------------------------------------
    # Insufficient material
    # ------------------------------------------------------

    insufficient = (
        states.insufficient_material()
    )

    insufficient_new = (
        insufficient
        & ~terminal
    )

    if bool(insufficient_new.any().item()):

        terminal |= insufficient_new

        indices = (
            torch.nonzero(
                insufficient_new,
                as_tuple=False
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

        for i in indices:
            termination[i] = "INSUFFICIENT_MATERIAL"

    return (
        terminal,
        results,
        termination
    )


# ==========================================================
# COPY STATES
# ==========================================================

def copy_states(
    destination: GPUChess,
    destination_indices: torch.Tensor,
    source: GPUChess,
):
    """
    Copy a batch of GPUChess states into selected positions
    of the master state tensor.
    """

    destination.pieces[destination_indices] = (
        source.pieces
    )

    destination.turn[destination_indices] = (
        source.turn
    )

    destination.castling[destination_indices] = (
        source.castling
    )

    destination.ep_square[destination_indices] = (
        source.ep_square
    )

    destination.halfmove_clock[destination_indices] = (
        source.halfmove_clock
    )

    destination.fullmove_number[destination_indices] = (
        source.fullmove_number
    )


# ==========================================================
# RUN ONE GPU MCTS BATCH
# ==========================================================

def search_batch(
    model,
    states: GPUChess,
    device,
):
    """
    Run one batched GPU MCTS search.

    Every position in states uses the same model.
    """

    if states.pieces.shape[0] == 0:
        return None

    search = GPUMCTS(
        model=model,
        device=device,
    )

    search.search(
        states,
        num_simulations=NUM_SIMULATIONS
    )

    actions = search.select_actions(
        temperature=EVALUATION_TEMPERATURE
    )

    next_states = search.advance(
        actions
    )

    return actions, next_states


# ==========================================================
# PLAY ALL GAMES IN PARALLEL
# ==========================================================

def play_games(
    white_model,
    black_model,
    device,
    game_offset=0,
    num_games=None,
):
    """
    Play num_games games simultaneously.

    Games are grouped by side-to-move so positions using the same
    neural network can be searched together.
    """

    if device.type != "cuda":
        raise RuntimeError(
            "GPU evaluation requires CUDA."
        )

    # ------------------------------------------------------
    # Resolve number of games for this worker
    # ------------------------------------------------------

    if num_games is None:
        num_games = NUM_GAMES

    # ------------------------------------------------------
    # Master state
    # ------------------------------------------------------

    states = GPUChess(
        device,
        num_games
    )

    # ------------------------------------------------------
    # Per-game information
    # ------------------------------------------------------

    active = torch.ones(
        num_games,
        dtype=torch.bool,
        device=device
    )

    move_counts = [0] * num_games

    repetition = [
        {}
        for _ in range(num_games)
    ]

    results = [None] * num_games

    termination = [None] * num_games

    # ------------------------------------------------------
    # Game color assignment
    #
    # Odd games:
    #   model A = White
    #   model B = Black
    #
    # Even games:
    #   model B = White
    #   model A = Black
    # ------------------------------------------------------

    model_a_is_white = [
        (game_offset + game) % 2 == 0
        for game in range(num_games)
    ]

    round_no = 0

    # ======================================================
    # MAIN GAME LOOP
    # ======================================================

    while bool(active.any().item()):

        round_no += 1

        # --------------------------------------------------
        # Active game indices
        # --------------------------------------------------

        active_indices = (
            torch.nonzero(
                active,
                as_tuple=False
            )
            .flatten()
        )

        if active_indices.numel() == 0:
            break

        active_states = states.select(
            active_indices
        )

        # --------------------------------------------------
        # THREEFOLD REPETITION
        # --------------------------------------------------

        hashes = (
            active_states
            .state_hash()
            .detach()
            .cpu()
            .tolist()
        )

        active_list = (
            active_indices
            .detach()
            .cpu()
            .tolist()
        )

        repetition_finished = []

        for local, game_index in enumerate(
            active_list
        ):

            key = int(hashes[local])

            count = (
                repetition[game_index]
                .get(key, 0)
                + 1
            )

            repetition[game_index][key] = count

            if count >= 3:

                results[game_index] = 0

                termination[game_index] = (
                    "THREEFOLD_REPETITION"
                )

                active[game_index] = False

                repetition_finished.append(
                    game_index
                )

        # --------------------------------------------------
        # Refresh active indices
        # --------------------------------------------------

        active_indices = (
            torch.nonzero(
                active,
                as_tuple=False
            )
            .flatten()
        )

        if active_indices.numel() == 0:
            break

        active_states = states.select(
            active_indices
        )

        # --------------------------------------------------
        # CHECK TERMINAL STATES
        # --------------------------------------------------

        terminal, terminal_results, terminal_names = (
            get_terminal_results(
                active_states
            )
        )

        terminal_local = (
            torch.nonzero(
                terminal,
                as_tuple=False
            )
            .flatten()
        )

        if terminal_local.numel() > 0:

            terminal_global = (
                active_indices[terminal_local]
            )

            for local, global_index in zip(
                terminal_local.detach().cpu().tolist(),
                terminal_global.detach().cpu().tolist()
            ):

                results[global_index] = int(
                    terminal_results[local].item()
                )

                termination[global_index] = (
                    terminal_names[local]
                )

                active[global_index] = False

        # --------------------------------------------------
        # Refresh after terminal states
        # --------------------------------------------------

        active_indices = (
            torch.nonzero(
                active,
                as_tuple=False
            )
            .flatten()
        )

        if active_indices.numel() == 0:
            break

        active_states = states.select(
            active_indices
        )

        # --------------------------------------------------
        # MAX MOVE LIMIT
        # --------------------------------------------------

        too_long = []

        for game_index in (
            active_indices
            .detach()
            .cpu()
            .tolist()
        ):

            if move_counts[game_index] >= MAX_MOVES:

                results[game_index] = None

                termination[game_index] = (
                    "MAX_MOVES"
                )

                active[game_index] = False

                too_long.append(
                    game_index
                )

        # --------------------------------------------------
        # Refresh after max-move removal
        # --------------------------------------------------

        active_indices = (
            torch.nonzero(
                active,
                as_tuple=False
            )
            .flatten()
        )

        if active_indices.numel() == 0:
            break

        # ==================================================
        # DETERMINE WHICH MODEL PLAYS EACH POSITION
        # ==================================================

        active_turns = states.turn[
            active_indices
        ]

        # turn=False = White
        # turn=True  = Black

        white_games = []
        black_games = []

        active_list = (
            active_indices
            .detach()
            .cpu()
            .tolist()
        )

        for local, game_index in enumerate(
            active_list
        ):

            is_black = bool(
                active_turns[local].item()
            )

            if not is_black:
                white_games.append(
                    local
                )
            else:
                black_games.append(
                    local
                )

        # ==================================================
        # WHITE MODEL BATCH
        # ==================================================

        if len(white_games) > 0:

            local_indices = torch.tensor(
                white_games,
                dtype=torch.long,
                device=device
            )

            global_indices = (
                active_indices[
                    local_indices
                ]
            )

            white_states = states.select(
                global_indices
            )

            # ------------------------------------------------
            # Select model according to game color assignment
            # ------------------------------------------------

            model_a_indices = []
            model_b_indices = []

            global_list = (
                global_indices
                .detach()
                .cpu()
                .tolist()
            )

            for local, game_index in enumerate(
                global_list
            ):

                if model_a_is_white[
                    game_index
                ]:
                    model_a_indices.append(
                        local
                    )
                else:
                    model_b_indices.append(
                        local
                    )

            # ----------------------------------------------
            # Model A playing White
            # ----------------------------------------------

            if len(model_a_indices) > 0:

                subset = torch.tensor(
                    model_a_indices,
                    dtype=torch.long,
                    device=device
                )

                subset_global = (
                    global_indices[subset]
                )

                subset_states = states.select(
                    subset_global
                )

                _, next_states = search_batch(
                    white_model,
                    subset_states,
                    device,
                )

                copy_states(
                    states,
                    subset_global,
                    next_states
                )

                for game_index in (
                    subset_global
                    .detach()
                    .cpu()
                    .tolist()
                ):
                    move_counts[game_index] += 1

            # ----------------------------------------------
            # Model B playing White
            # ----------------------------------------------

            if len(model_b_indices) > 0:

                subset = torch.tensor(
                    model_b_indices,
                    dtype=torch.long,
                    device=device
                )

                subset_global = (
                    global_indices[subset]
                )

                subset_states = states.select(
                    subset_global
                )

                _, next_states = search_batch(
                    black_model,
                    subset_states,
                    device,
                )

                copy_states(
                    states,
                    subset_global,
                    next_states
                )

                for game_index in (
                    subset_global
                    .detach()
                    .cpu()
                    .tolist()
                ):
                    move_counts[game_index] += 1

        # ==================================================
        # BLACK MODEL BATCH
        # ==================================================

        if len(black_games) > 0:

            local_indices = torch.tensor(
                black_games,
                dtype=torch.long,
                device=device
            )

            global_indices = (
                active_indices[
                    local_indices
                ]
            )

            model_a_indices = []
            model_b_indices = []

            global_list = (
                global_indices
                .detach()
                .cpu()
                .tolist()
            )

            for local, game_index in enumerate(
                global_list
            ):

                if not model_a_is_white[
                    game_index
                ]:
                    model_a_indices.append(
                        local
                    )
                else:
                    model_b_indices.append(
                        local
                    )

            # ----------------------------------------------
            # Model A playing Black
            # ----------------------------------------------

            if len(model_a_indices) > 0:

                subset = torch.tensor(
                    model_a_indices,
                    dtype=torch.long,
                    device=device
                )

                subset_global = (
                    global_indices[subset]
                )

                subset_states = states.select(
                    subset_global
                )

                _, next_states = search_batch(
                    black_model,
                    subset_states,
                    device,
                )

                copy_states(
                    states,
                    subset_global,
                    next_states
                )

                for game_index in (
                    subset_global
                    .detach()
                    .cpu()
                    .tolist()
                ):
                    move_counts[game_index] += 1

            # ----------------------------------------------
            # Model B playing Black
            # ----------------------------------------------

            if len(model_b_indices) > 0:

                subset = torch.tensor(
                    model_b_indices,
                    dtype=torch.long,
                    device=device
                )

                subset_global = (
                    global_indices[subset]
                )

                subset_states = states.select(
                    subset_global
                )

                _, next_states = search_batch(
                    black_model,
                    subset_states,
                    device,
                )

                copy_states(
                    states,
                    subset_global,
                    next_states
                )

                for game_index in (
                    subset_global
                    .detach()
                    .cpu()
                    .tolist()
                ):
                    move_counts[game_index] += 1

        # ==================================================
        # CHECK RESULTS AFTER MOVES
        # ==================================================

        active_indices = (
            torch.nonzero(
                active,
                as_tuple=False
            )
            .flatten()
        )

        if active_indices.numel() > 0:

            current_states = states.select(
                active_indices
            )

            terminal, terminal_results, terminal_names = (
                get_terminal_results(
                    current_states
                )
            )

            terminal_local = (
                torch.nonzero(
                    terminal,
                    as_tuple=False
                )
                .flatten()
            )

            if terminal_local.numel() > 0:

                terminal_global = (
                    active_indices[
                        terminal_local
                    ]
                )

                for local, global_index in zip(
                    terminal_local.detach().cpu().tolist(),
                    terminal_global.detach().cpu().tolist()
                ):

                    results[global_index] = int(
                        terminal_results[
                            local
                        ].item()
                    )

                    termination[
                        global_index
                    ] = terminal_names[local]

                    active[global_index] = False

        # ==================================================
        # PROGRESS
        # ==================================================

        # print(
        #     f"Evaluation round {round_no} | "
        #     f"active games: "
        #     f"{int(active.sum().item())}/{NUM_GAMES}",
        #     flush=True,
        # )

    # ======================================================
    # BUILD FINAL RESULTS
    # ======================================================

    final_results = []

    for game_index in range(num_games):

        final_results.append(
            {
                "result": results[game_index],
                "termination": termination[game_index],
                "moves": move_counts[game_index],
            }
        )

    return final_results



# ==========================================================
# MULTI-GPU EVALUATION
# ==========================================================

def _evaluation_worker(
    rank,
    device_id,
    num_games,
    game_offset,
    model_a_checkpoint,
    model_b_checkpoint,
    output_path,
):
    """Run one evaluation shard entirely on one GPU.

    Each GPU owns both models and an independent set of games.
    This is the same strategy used by multi-GPU self-play:
    one process per GPU, with no cross-GPU state transfers.
    """

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}")

    print(
        f"[Evaluation worker {rank}] "
        f"GPU {device_id}: {torch.cuda.get_device_name(device_id)} | "
        f"games={num_games}",
        flush=True,
    )

    model_a, _ = create_model(
        model_a_checkpoint,
        device,
    )

    model_b, _ = create_model(
        model_b_checkpoint,
        device,
    )

    results = play_games(
        white_model=model_a,
        black_model=model_b,
        device=device,
        game_offset=game_offset,
        num_games=num_games,
    )

    torch.save(results, output_path)

    print(
        f"[Evaluation worker {rank}] "
        f"finished {len(results)} games",
        flush=True,
    )


def play_games_multi_gpu(
    model_a_checkpoint,
    model_b_checkpoint,
):
    """Run the evaluation across every visible CUDA GPU.

    For two GPUs and NUM_GAMES=10:
        GPU 0 -> 5 games
        GPU 1 -> 5 games

    Each worker loads both checkpoints onto its own GPU and runs
    GPUChess + GPUMCTS locally. Therefore both GPUs are active
    concurrently rather than one GPU handling all games.
    """

    gpu_count = torch.cuda.device_count()

    if gpu_count < 2:
        raise RuntimeError(
            f"Expected at least 2 GPUs, but only {gpu_count} "
            "CUDA device(s) are visible."
        )

    workers = min(gpu_count, NUM_GAMES)

    game_counts = [NUM_GAMES // workers] * workers

    for i in range(NUM_GAMES % workers):
        game_counts[i] += 1

    print()
    print("=" * 60)
    print("MULTI-GPU EVALUATION")
    print("=" * 60)
    print("Visible GPUs:", gpu_count)
    print("Workers:", workers)
    print("Games per GPU:", game_counts)

    for device_id in range(gpu_count):
        print(
            f"GPU {device_id}: "
            f"{torch.cuda.get_device_name(device_id)}"
        )

    temp_dir = tempfile.mkdtemp(
        prefix="chess_eval_multi_gpu_"
    )

    ctx = mp.get_context("spawn")
    processes = []
    output_paths = []

    try:
        game_offset = 0

        for rank, (device_id, worker_games) in enumerate(
            enumerate(game_counts)
        ):
            output_path = os.path.join(
                temp_dir,
                f"worker_{rank}.pt",
            )

            output_paths.append(output_path)

            process = ctx.Process(
                target=_evaluation_worker,
                args=(
                    rank,
                    device_id,
                    worker_games,
                    game_offset,
                    model_a_checkpoint,
                    model_b_checkpoint,
                    output_path,
                ),
            )

            process.start()
            processes.append(process)

            game_offset += worker_games

        for process in processes:
            process.join()

        failed = [
            (rank, process.exitcode)
            for rank, process in enumerate(processes)
            if process.exitcode != 0
        ]

        if failed:
            raise RuntimeError(
                f"Evaluation worker failure: {failed}"
            )

        results = []

        for output_path in output_paths:
            worker_results = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )

            results.extend(worker_results)

        if len(results) != NUM_GAMES:
            raise RuntimeError(
                f"Expected {NUM_GAMES} results, "
                f"but received {len(results)}."
            )

        print(
            f"MULTI-GPU EVALUATION COMPLETE: "
            f"{len(results)} games"
        )

        return results

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ==========================================================
# EVALUATE MODELS
# ==========================================================

def evaluate_models(
    results,
    name_a,
    name_b,
    elapsed,
):

    a_wins = 0
    b_wins = 0
    draws = 0
    truncated = 0
    total_moves = 0

    termination_counts = {}

    # ======================================================
    # PRINT INDIVIDUAL GAMES
    # ======================================================

    for game_index, result in enumerate(
        results,
        start=1
    ):

        if game_index % 2 == 1:
            a_color = "White"
        else:
            a_color = "Black"

        print(
            f"Game {game_index}/{NUM_GAMES}: "
            f"{name_a}={a_color} | "
            f"{result}"
        )

        total_moves += result["moves"]

        reason = result["termination"]

        termination_counts[reason] = (
            termination_counts.get(reason, 0)
            + 1
        )

        # --------------------------------------------------
        # Truncated
        # --------------------------------------------------

        if result["result"] is None:

            truncated += 1

        # --------------------------------------------------
        # Draw
        # --------------------------------------------------

        elif result["result"] == 0:

            draws += 1

        # --------------------------------------------------
        # Model A win
        # --------------------------------------------------

        elif (
            result["result"] == 1
            and a_color == "White"
        ) or (
            result["result"] == -1
            and a_color == "Black"
        ):

            a_wins += 1

        # --------------------------------------------------
        # Model B win
        # --------------------------------------------------

        else:

            b_wins += 1

    # ======================================================
    # SCORES
    # ======================================================

    completed_games = (
        a_wins
        + b_wins
        + draws
    )

    if completed_games > 0:

        a_score = (
            a_wins
            + 0.5 * draws
        ) / completed_games

        b_score = (
            b_wins
            + 0.5 * draws
        ) / completed_games

    else:

        a_score = 0.0
        b_score = 0.0

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ======================================================
    # FINAL REPORT
    # ======================================================

    print()
    print("=" * 60)
    print("GPU-BATCHED EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"{name_a} wins:",
        a_wins
    )

    print(
        f"{name_b} wins:",
        b_wins
    )

    print(
        "Draws:",
        draws
    )

    print(
        "Truncated:",
        truncated
    )

    print(
        "Completed games:",
        completed_games
    )

    print(
        "Average moves:",
        f"{total_moves / NUM_GAMES:.1f}"
        if NUM_GAMES
        else "0.0"
    )

    print(
        f"{name_a} score:",
        f"{a_score:.3f}"
    )

    print(
        f"{name_b} score:",
        f"{b_score:.3f}"
    )

    print(
        "Termination reasons:"
    )

    for reason, count in (
        termination_counts.items()
    ):

        print(
            f"  {reason}: {count}"
        )

    print(
        "Elapsed:",
        f"{elapsed:.2f}s"
    )

    print(
        "Games/second:",
        f"{NUM_GAMES / elapsed:.3f}"
        if elapsed > 0
        else "0.000"
    )

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "truncated": truncated,
        "completed_games": completed_games,
        "a_score": a_score,
        "b_score": b_score,
        "termination_counts": termination_counts,
        "elapsed": elapsed,
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU evaluation requires CUDA."
        )

    gpu_count = torch.cuda.device_count()

    if gpu_count < 2:
        raise RuntimeError(
            f"Expected 2 GPUs for this evaluation, "
            f"but only {gpu_count} CUDA GPU(s) are visible."
        )

    print(
        "Evaluation GPUs:",
        gpu_count,
    )

    for device_id in range(gpu_count):
        print(
            f"GPU {device_id}:",
            torch.cuda.get_device_name(device_id),
        )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "Games:",
        NUM_GAMES,
    )

    print(
        "Simulations/game:",
        NUM_SIMULATIONS,
    )

    print(
        "Max moves:",
        MAX_MOVES,
    )

    print(
        "Temperature:",
        EVALUATION_TEMPERATURE,
    )

    # ------------------------------------------------------
    # AUTOMATIC MODEL NAMES FROM CHECKPOINT PATHS
    # ------------------------------------------------------

    def checkpoint_name(path):
        """
        Create a readable model name from the checkpoint filename.

        Examples:
            rl_iteration_4.pt  -> RL iteration 4
            rl_iteration_12.pt -> RL iteration 12
            pretrained_phase1.pt -> Pretrained phase 1
        """
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]

        match = re.search(
            r"rl_iteration[_-]?(\d+)",
            stem,
            re.IGNORECASE,
        )

        if match:
            return f"RL iteration {match.group(1)}"

        match = re.search(
            r"pretrained[_-]?phase[_-]?(\d+)",
            stem,
            re.IGNORECASE,
        )

        if match:
            return f"Pretrained phase {match.group(1)}"

        return (
            stem
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

    name_a = checkpoint_name(
        MODEL_A_CHECKPOINT
    )

    name_b = checkpoint_name(
        MODEL_B_CHECKPOINT
    )

    print()
    print("Model A checkpoint:")
    print(MODEL_A_CHECKPOINT)
    print("Model A:", name_a)

    print()
    print("Model B checkpoint:")
    print(MODEL_B_CHECKPOINT)
    print("Model B:", name_b)

    # ------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------

    start_time = time.perf_counter()

    results = play_games_multi_gpu(
        model_a_checkpoint=MODEL_A_CHECKPOINT,
        model_b_checkpoint=MODEL_B_CHECKPOINT,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    evaluate_models(
        results,
        name_a,
        name_b,
        elapsed,
    )
