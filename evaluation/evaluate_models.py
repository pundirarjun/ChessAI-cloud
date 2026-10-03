from __future__ import annotations

import os
import sys
import time
import re
import shutil
import tempfile
import multiprocessing as mp
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

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

# 0.0 = deterministic move selection.
EVALUATION_TEMPERATURE = 0.25

# Keep the existing stochastic default, while making reproducible benchmark
# runs an explicit opt-in.  BENCHMARK_SEED is intentionally configurable rather
# than baked into the evaluation logic.
EVALUATION_SEED: Optional[int] = 123
BENCHMARK_MODE = False
BENCHMARK_SEED: Optional[int] = 42

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
    "/kaggle/input/datasets/arjunthakur9999/checkpoints/chess_checkpoints (4)/rl_iteration_50.pt"
)


# ==========================================================
# EVALUATION IDENTITY / REPRODUCIBILITY
# ==========================================================

@dataclass(frozen=True)
class EvaluationSettings:
    temperature: float
    seed: Optional[int]
    benchmark_mode: bool


@dataclass(frozen=True)
class GameAssignment:
    """The intended model identity for one globally numbered evaluation game."""

    global_game_index: int
    model_a_is_white: bool

    @property
    def model_a_color(self) -> str:
        return "White" if self.model_a_is_white else "Black"

    @property
    def white_model_id(self) -> str:
        return "A" if self.model_a_is_white else "B"

    @property
    def black_model_id(self) -> str:
        return "B" if self.model_a_is_white else "A"


def resolve_evaluation_settings(
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    benchmark_mode: Optional[bool] = None,
) -> EvaluationSettings:
    """Resolve the explicit benchmark mode without changing stochastic defaults."""
    benchmark = BENCHMARK_MODE if benchmark_mode is None else bool(benchmark_mode)
    selected_temperature = EVALUATION_TEMPERATURE if temperature is None else float(temperature)
    selected_seed = EVALUATION_SEED if seed is None else int(seed)

    if benchmark:
        selected_temperature = 0.0
        if selected_seed is None:
            selected_seed = BENCHMARK_SEED

    if not 0.0 <= selected_temperature:
        raise ValueError("Evaluation temperature must be non-negative.")
    if benchmark and selected_seed is None:
        raise ValueError("Deterministic benchmark mode requires a configured seed.")

    return EvaluationSettings(
        temperature=selected_temperature,
        seed=selected_seed,
        benchmark_mode=benchmark,
    )


def build_game_assignments(num_games: int, game_offset: int = 0) -> list[GameAssignment]:
    """Assign alternating global games so the overall match is color balanced."""
    if num_games < 0:
        raise ValueError("num_games must be non-negative")
    return [
        GameAssignment(
            global_game_index=game_offset + game,
            model_a_is_white=(game_offset + game) % 2 == 0,
        )
        for game in range(num_games)
    ]


def _set_model_identity(model, model_id: str, checkpoint_path: str):
    if model_id not in {"A", "B"}:
        raise ValueError(f"Unknown evaluation model id: {model_id}")
    model._evaluation_model_id = model_id
    model._evaluation_checkpoint = os.path.abspath(checkpoint_path)
    return model


def _require_model_identity(model, expected_model_id: str):
    actual_model_id = getattr(model, "_evaluation_model_id", None)
    checkpoint = getattr(model, "_evaluation_checkpoint", None)
    if actual_model_id != expected_model_id or not checkpoint:
        raise RuntimeError(
            "Evaluation model identity mismatch: "
            f"expected {expected_model_id}, got {actual_model_id!r}."
        )


def select_assigned_model(model_a, model_b, assignment: GameAssignment, is_black: bool):
    """Return the network dictated by model identity, not by a color variable."""
    expected_model_id = assignment.black_model_id if is_black else assignment.white_model_id
    model = model_a if expected_model_id == "A" else model_b
    _require_model_identity(model, expected_model_id)
    return model, expected_model_id


def _seed_worker(seed: Optional[int], rank: int):
    if seed is None:
        return
    worker_seed = int(seed) + int(rank)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model(path, device, model_id):

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

    _set_model_identity(model, model_id, path)
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
    temperature,
    repetition_history=None,
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
        num_simulations=NUM_SIMULATIONS,
        repetition_history=repetition_history,
    )

    actions = search.select_actions(
        temperature=temperature
    )

    next_states = search.advance(
        actions
    )

    return actions, next_states


# ==========================================================
# PLAY ALL GAMES IN PARALLEL
# ==========================================================

def play_games(
    model_a,
    model_b,
    device,
    game_offset=0,
    num_games=None,
    temperature=EVALUATION_TEMPERATURE,
):
    """Play games with explicit A/B identity checks on every searched side."""
    if device.type != "cuda":
        raise RuntimeError("GPU evaluation requires CUDA.")
    if num_games is None:
        num_games = NUM_GAMES

    _require_model_identity(model_a, "A")
    _require_model_identity(model_b, "B")
    assignments = build_game_assignments(num_games, game_offset)
    assignment_by_local_game = {a.global_game_index - game_offset: a for a in assignments}

    states = GPUChess(device, num_games)
    active = torch.ones(num_games, dtype=torch.bool, device=device)
    move_counts = [0] * num_games
    repetition = [{} for _ in range(num_games)]
    # This history is retained on CUDA so GPUMCTS can recognize internal tree
    # leaves that would be the third occurrence of an actual game position.
    history = torch.empty((num_games, MAX_MOVES + 1), dtype=torch.int64, device=device)
    results = [None] * num_games
    termination = [None] * num_games

    def finish_terminal(active_indices, terminal, terminal_results, terminal_names):
        terminal_local = torch.nonzero(terminal, as_tuple=False).flatten()
        if terminal_local.numel() == 0:
            return
        terminal_global = active_indices[terminal_local]
        for local, global_index in zip(
            terminal_local.detach().cpu().tolist(),
            terminal_global.detach().cpu().tolist(),
        ):
            results[global_index] = int(terminal_results[local].item())
            termination[global_index] = terminal_names[local]
            active[global_index] = False

    def advance_subset(subset_global, is_black: bool):
        """Search one identity-homogeneous side batch and assert every choice."""
        if subset_global.numel() == 0:
            return
        local_games = subset_global.detach().cpu().tolist()
        assignments_for_subset = [assignment_by_local_game[g] for g in local_games]
        selected_models = [
            select_assigned_model(model_a, model_b, assignment, is_black)[0]
            for assignment in assignments_for_subset
        ]
        selected_ids = [
            select_assigned_model(model_a, model_b, assignment, is_black)[1]
            for assignment in assignments_for_subset
        ]
        if len(set(selected_ids)) != 1 or any(model is not selected_models[0] for model in selected_models):
            raise RuntimeError("Evaluation batch mixed model identities unexpectedly.")

        subset_states = states.select(subset_global)
        _, next_states = search_batch(
            selected_models[0],
            subset_states,
            device,
            temperature=temperature,
            repetition_history=history[subset_global, :round_no],
        )
        copy_states(states, subset_global, next_states)
        for game_index in local_games:
            move_counts[game_index] += 1

    round_no = 0
    while bool(active.any().item()):
        round_no += 1
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        active_states = states.select(active_indices)

        # Threefold detection remains the authoritative game-level rule.
        hash_tensor = active_states.state_hash()
        history[active_indices, round_no - 1] = hash_tensor
        hashes = hash_tensor.detach().cpu().tolist()
        for local, game_index in enumerate(active_indices.detach().cpu().tolist()):
            key = int(hashes[local])
            repetition[game_index][key] = repetition[game_index].get(key, 0) + 1
            if repetition[game_index][key] >= 3:
                results[game_index] = 0
                termination[game_index] = "THREEFOLD_REPETITION"
                active[game_index] = False

        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        active_states = states.select(active_indices)
        terminal, terminal_results, terminal_names = get_terminal_results(active_states)
        finish_terminal(active_indices, terminal, terminal_results, terminal_names)

        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        for game_index in active_indices.detach().cpu().tolist():
            if move_counts[game_index] >= MAX_MOVES:
                results[game_index] = None
                termination[game_index] = "MAX_MOVES"
                active[game_index] = False

        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break

        active_turns = states.turn[active_indices]
        white_local = [i for i, turn in enumerate(active_turns.detach().cpu().tolist()) if not turn]
        black_local = [i for i, turn in enumerate(active_turns.detach().cpu().tolist()) if turn]

        # Split each side-to-move batch by the model identity dictated by its
        # assignment.  In particular, A-as-Black uses model_a, never model_b.
        for is_black, local_indices in ((False, white_local), (True, black_local)):
            if not local_indices:
                continue
            side_global = active_indices[
                torch.tensor(local_indices, dtype=torch.long, device=device)
            ]
            a_local = []
            b_local = []
            for local, game_index in enumerate(side_global.detach().cpu().tolist()):
                assignment = assignment_by_local_game[game_index]
                expected = assignment.black_model_id if is_black else assignment.white_model_id
                (a_local if expected == "A" else b_local).append(local)
            for model_local in (a_local, b_local):
                if model_local:
                    subset_global = side_global[
                        torch.tensor(model_local, dtype=torch.long, device=device)
                    ]
                    advance_subset(subset_global, is_black=is_black)

        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() > 0:
            current_states = states.select(active_indices)
            terminal, terminal_results, terminal_names = get_terminal_results(current_states)
            finish_terminal(active_indices, terminal, terminal_results, terminal_names)

    model_a_checkpoint = getattr(model_a, "_evaluation_checkpoint")
    model_b_checkpoint = getattr(model_b, "_evaluation_checkpoint")
    return [
        {
            "global_game_index": assignment.global_game_index,
            "result": results[local_game_index],
            "termination": termination[local_game_index],
            "moves": move_counts[local_game_index],
            "model_a_color": assignment.model_a_color,
            "white_model_id": assignment.white_model_id,
            "black_model_id": assignment.black_model_id,
            "model_a_checkpoint": model_a_checkpoint,
            "model_b_checkpoint": model_b_checkpoint,
        }
        for local_game_index, assignment in enumerate(assignments)
    ]



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
    settings,
):
    """Run one evaluation shard entirely on one GPU.

    Each GPU owns both models and an independent set of games.
    This is the same strategy used by multi-GPU self-play:
    one process per GPU, with no cross-GPU state transfers.
    """

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.cuda.set_device(device_id)
    _seed_worker(settings.seed, rank)

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
        "A",
    )

    model_b, _ = create_model(
        model_b_checkpoint,
        device,
        "B",
    )

    results = play_games(
        model_a=model_a,
        model_b=model_b,
        device=device,
        game_offset=game_offset,
        num_games=num_games,
        temperature=settings.temperature,
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
    *,
    temperature=None,
    seed=None,
    benchmark_mode=None,
):
    """Run the evaluation across every visible CUDA GPU.

    For two GPUs and NUM_GAMES=10:
        GPU 0 -> 5 games
        GPU 1 -> 5 games

    Each worker loads both checkpoints onto its own GPU and runs
    GPUChess + GPUMCTS locally. Therefore both GPUs are active
    concurrently rather than one GPU handling all games.
    """

    settings = resolve_evaluation_settings(
        temperature=temperature,
        seed=seed,
        benchmark_mode=benchmark_mode,
    )
    if NUM_GAMES % 2 != 0:
        raise ValueError(
            "Color-balanced evaluation requires an even NUM_GAMES value."
        )
    assignments = build_game_assignments(NUM_GAMES)
    if sum(a.model_a_is_white for a in assignments) != NUM_GAMES // 2:
        raise RuntimeError("Model A color assignment is not balanced.")

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
    print("Benchmark mode:", settings.benchmark_mode)
    print("Temperature:", settings.temperature)
    print("Seed:", settings.seed)
    print("Model A colors: White=", NUM_GAMES // 2, "Black=", NUM_GAMES // 2)
    print("Model B colors: White=", NUM_GAMES // 2, "Black=", NUM_GAMES // 2)

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
                    settings,
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

    color_counts = Counter()
    identity_pairs = Counter()

    for game_index, result in enumerate(
        results,
        start=1
    ):
        a_color = result["model_a_color"]
        white_model_id = result["white_model_id"]
        black_model_id = result["black_model_id"]
        if {white_model_id, black_model_id} != {"A", "B"}:
            raise RuntimeError(
                f"Invalid evaluation identity pair in game {game_index}: "
                f"White={white_model_id}, Black={black_model_id}."
            )
        expected_a_color = "White" if white_model_id == "A" else "Black"
        if a_color != expected_a_color:
            raise RuntimeError(f"Game {game_index} has inconsistent Model A color metadata.")
        color_counts[a_color] += 1
        identity_pairs[(white_model_id, black_model_id)] += 1

        print(
            f"Game {game_index}/{NUM_GAMES}: "
            f"{name_a}={a_color} | White={white_model_id} "
            f"Black={black_model_id} | result={result['result']} "
            f"termination={result['termination']}"
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

    if color_counts["White"] != color_counts["Black"]:
        raise RuntimeError(f"Unbalanced colors in results: {dict(color_counts)}")
    if identity_pairs[("A", "B")] != identity_pairs[("B", "A")]:
        raise RuntimeError(f"Unbalanced model pairings in results: {dict(identity_pairs)}")

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
        "Model identity / color audit:",
        f"A White/B Black={identity_pairs[('A', 'B')]} | "
        f"B White/A Black={identity_pairs[('B', 'A')]}"
    )

    print(
        "Score formula:",
        "(wins + 0.5 * draws) / completed games; truncations excluded"
    )

    print(
        "Average moves:",
        f"{total_moves / len(results):.1f}"
        if results
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
        f"{len(results) / elapsed:.3f}"
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

    evaluation_settings = resolve_evaluation_settings()

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
        evaluation_settings.temperature,
    )

    print(
        "Benchmark mode:",
        evaluation_settings.benchmark_mode,
    )

    print(
        "Seed:",
        evaluation_settings.seed,
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
        temperature=evaluation_settings.temperature,
        seed=evaluation_settings.seed,
        benchmark_mode=evaluation_settings.benchmark_mode,
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
