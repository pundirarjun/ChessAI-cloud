"""GPU-first AlphaZero-style self-play.

The training self-play path uses the tensorized GPU chess engine and GPU MCTS
when the model is on CUDA.  A small legacy CPU fallback is retained for local
CPU use and for tools that still rely on python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import random
import shutil
import tempfile
import multiprocessing as mp

import numpy as np
import torch

from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


@dataclass
class TrainingSample:
    state: np.ndarray
    policy: np.ndarray
    player: int


@dataclass
class SelfPlayResult:
    training_data: list
    result: int | None
    termination: str
    moves_played: int
    completed: bool


class SelfPlayGame:
    def __init__(self):
        self.samples = []

    def add_position(self, state: np.ndarray, policy: np.ndarray, player: int):
        self.samples.append(
            TrainingSample(
                state=np.asarray(state, dtype=np.float32),
                policy=np.asarray(policy, dtype=np.float32),
                player=int(player),
            )
        )

    def get_training_data(self, result):
        data = []
        for sample in self.samples:
            if result == 0:
                value = 0.0
            elif result == 1:
                value = float(sample.player)
            else:
                value = float(-sample.player)
            data.append((sample.state, sample.policy, value))
        return data


def _terminal_result(states: GPUChess, local_index: int) -> tuple[int, str, bool]:
    """Return result/termination for one terminal GPU state."""
    one = states.select(torch.tensor([local_index], device=states.device))
    legal = one.legal_move_mask()[0]
    in_check = bool(one.is_in_check()[0].item())
    if not bool(legal.any().item()):
        if in_check:
            # Side to move has been checkmated.
            result = -1 if bool(one.turn[0].item()) is False else 1
            return result, "CHECKMATE", True
        return 0, "STALEMATE", True
    if bool((one.halfmove_clock[0] >= 100).item()):
        return 0, "FIFTY_MOVE", True
    if bool(one.insufficient_material()[0].item()):
        return 0, "INSUFFICIENT_MATERIAL", True
    return 0, "UNKNOWN", False


def _play_games_gpu(
    model,
    num_games=8,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    late_temperature=0.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    """GPU-resident self-play.

    The hot path keeps boards, repetition history, training states, policies and
    players on CUDA.  CPU is used only after a game finishes (or for final
    checkpoint/replay serialization).  ``batch_size`` is retained for API
    compatibility; MCTS receives the complete active batch.
    """
    if num_games <= 0:
        return []
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("GPU self-play requires a CUDA model")
    _ = batch_size

    states = GPUChess(device, num_games)
    search = GPUMCTS(model=model, device=device)

    # All per-move training information remains on the GPU.  0/1 board planes
    # are stored as uint8 because they are exact and much smaller than float32.
    sample_states = torch.empty(
        (num_games, max_moves, 18, 8, 8), dtype=torch.uint8, device=device
    )
    sample_policies = torch.empty(
        (num_games, max_moves, 4544), dtype=torch.float32, device=device
    )
    sample_players = torch.empty((num_games, max_moves), dtype=torch.int8, device=device)

    move_numbers = torch.ones((num_games,), dtype=torch.int32, device=device)
    active = torch.ones((num_games,), dtype=torch.bool, device=device)
    result_tensor = torch.zeros((num_games,), dtype=torch.int8, device=device)
    termination_code = torch.zeros((num_games,), dtype=torch.int8, device=device)
    completed_tensor = torch.zeros((num_games,), dtype=torch.bool, device=device)

    # Full repetition history on GPU.  This replaces state_hash().cpu().tolist()
    # on every move and therefore removes a major synchronization point.
    history = torch.empty((num_games, max_moves + 1), dtype=torch.int64, device=device)

    # Codes: 1 checkmate, 2 stalemate, 3 fifty-move, 4 insufficient,
    # 5 threefold, 6 max-moves, 7 unknown.
    CODE_CHECKMATE = 1
    CODE_STALEMATE = 2
    CODE_FIFTY = 3
    CODE_INSUFFICIENT = 4
    CODE_THREEFOLD = 5
    CODE_MAX_MOVES = 6
    CODE_UNKNOWN = 7

    # ------------------------------------------------------------------
    # MCTS target diagnostics
    # ------------------------------------------------------------------
    # These statistics are diagnostic only. They do NOT change the MCTS
    # search, move selection, or training targets.
    diagnostic_positions = 0
    diagnostic_entropy_sum = 0.0
    diagnostic_normalized_entropy_sum = 0.0
    diagnostic_max_prob_sum = 0.0
    diagnostic_legal_moves_sum = 0.0
    diagnostic_max_prob_bins = {
        "<0.20": 0,
        "0.20-0.40": 0,
        "0.40-0.60": 0,
        "0.60-0.80": 0,
        ">=0.80": 0,
    }

    round_no = 0
    while round_no < max_moves:
        round_no += 1
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            break
        active_states = states.select(active_idx)

        # Repetition detection stays entirely on CUDA.
        keys = active_states.state_hash()
        history[active_idx, round_no - 1] = keys
        previous = history[active_idx, :round_no]
        repeated = (previous == keys[:, None]).sum(dim=1) >= 3
        rep_local = torch.nonzero(repeated, as_tuple=False).flatten()
        if rep_local.numel() > 0:
            rep_global = active_idx[rep_local]
            result_tensor[rep_global] = 0
            termination_code[rep_global] = CODE_THREEFOLD
            completed_tensor[rep_global] = True
            active[rep_global] = False

        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            continue
        active_states = states.select(active_idx)

        # Save the position BEFORE the selected move.  This is still entirely
        # GPU-resident; no numpy/CPU conversion occurs here.
        sample_pos = (move_numbers[active_idx] - 1).long()
        model_input = active_states.to_model_input().to(torch.uint8)
        sample_states[active_idx, sample_pos] = model_input
        sample_players[active_idx, sample_pos] = (~active_states.turn).to(torch.int8).mul(2).sub(1)

        search.search(
            active_states,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
            repetition_history=history[active_idx, :round_no],
        )
        policies = search.root_visit_policy()

        # --------------------------------------------------------------
        # MCTS TARGET DIAGNOSTICS
        # --------------------------------------------------------------
        # root_visit_policy() is already a probability distribution over
        # the 4544-action space.  We measure its concentration without
        # changing the policy used for self-play.
        with torch.no_grad():
            eps = 1e-12
            p = policies.clamp_min(eps)
            entropy = -(policies * p.log()).sum(dim=1)
            max_prob = policies.max(dim=1).values

            # Exact legal-move count. This is diagnostic only and therefore
            # intentionally kept out of the normal training logic.
            legal_mask = active_states.legal_move_mask()
            legal_count = legal_mask.sum(dim=1).to(torch.float32)

            # Normalize entropy by log(number of legal actions). A value
            # near 1 means a very spread-out target; near 0 means highly
            # concentrated.
            normalized_entropy = entropy / legal_count.clamp_min(2.0).log()

            n = int(policies.shape[0])
            diagnostic_positions += n
            diagnostic_entropy_sum += float(entropy.sum().item())
            diagnostic_normalized_entropy_sum += float(normalized_entropy.sum().item())
            diagnostic_max_prob_sum += float(max_prob.sum().item())
            diagnostic_legal_moves_sum += float(legal_count.sum().item())

            diagnostic_max_prob_bins["<0.20"] += int((max_prob < 0.20).sum().item())
            diagnostic_max_prob_bins["0.20-0.40"] += int(((max_prob >= 0.20) & (max_prob < 0.40)).sum().item())
            diagnostic_max_prob_bins["0.40-0.60"] += int(((max_prob >= 0.40) & (max_prob < 0.60)).sum().item())
            diagnostic_max_prob_bins["0.60-0.80"] += int(((max_prob >= 0.60) & (max_prob < 0.80)).sum().item())
            diagnostic_max_prob_bins[">=0.80"] += int((max_prob >= 0.80).sum().item())

        current_temperature = (
            temperature if round_no <= temperature_moves else late_temperature
        )
        actions = search.select_actions(current_temperature)
        next_states = search.advance(actions)
        sample_policies[active_idx, sample_pos] = policies

        # Advance actual game states on CUDA.
        states.pieces[active_idx] = next_states.pieces
        states.turn[active_idx] = next_states.turn
        states.castling[active_idx] = next_states.castling
        states.ep_square[active_idx] = next_states.ep_square
        states.halfmove_clock[active_idx] = next_states.halfmove_clock
        states.fullmove_number[active_idx] = next_states.fullmove_number
        move_numbers[active_idx] += 1

        # Terminal detection is batched on CUDA.  Only the small terminal index
        # list is copied to CPU for constructing Python result objects.
        terminal, terminal_value = next_states.terminal_info()
        insufficient_all = next_states.insufficient_material()
        fifty_all = next_states.halfmove_clock >= 100
        # terminal_info already computed legal moves. Its value is -1 only for
        # checkmate, so do not regenerate the expensive legal mask here.
        checkmate_all = terminal & (terminal_value < 0)
        stalemate_all = terminal & ~checkmate_all & ~fifty_all & ~insufficient_all
        terminal_all = terminal | insufficient_all
        term_local = torch.nonzero(terminal_all, as_tuple=False).flatten()
        if term_local.numel() > 0:
            checkmate = checkmate_all[term_local]
            fifty = fifty_all[term_local]
            insufficient = insufficient_all[term_local]
            stalemate = stalemate_all[term_local]
            term_code = torch.where(
                checkmate,
                torch.full_like(term_local, CODE_CHECKMATE, dtype=torch.int64),
                torch.where(
                    fifty,
                    torch.full_like(term_local, CODE_FIFTY, dtype=torch.int64),
                    torch.where(
                        insufficient,
                        torch.full_like(term_local, CODE_INSUFFICIENT, dtype=torch.int64),
                        torch.where(
                            stalemate,
                            torch.full_like(term_local, CODE_STALEMATE, dtype=torch.int64),
                            torch.full_like(term_local, CODE_UNKNOWN, dtype=torch.int64),
                        ),
                    ),
                ),
            )
            term_turn = next_states.turn[term_local]
            winner = torch.where(term_turn, torch.ones_like(term_local), -torch.ones_like(term_local))
            term_result = torch.where(checkmate, winner, torch.zeros_like(term_local))
            global_idx = active_idx[term_local]
            result_tensor[global_idx] = term_result.to(torch.int8)
            termination_code[global_idx] = term_code.to(torch.int8)
            completed_tensor[global_idx] = True
            active[global_idx] = False

        # Progress: print every self-play round.
        # flush=True makes the update appear immediately in Kaggle/terminal output.
        print(
            f"GPU self-play round {round_no} | "
            f"active games: {int(active.sum().item())}/{num_games}",
            flush=True,
        )

    # Any game still active reached the move budget and is deliberately not
    # converted into training data.
    max_idx = torch.nonzero(active, as_tuple=False).flatten()
    if max_idx.numel() > 0:
        termination_code[max_idx] = CODE_MAX_MOVES
        completed_tensor[max_idx] = False
        active[max_idx] = False

    # One CPU transfer per completed game, instead of one transfer per move.
    result_cpu = result_tensor.detach().cpu().tolist()
    code_cpu = termination_code.detach().cpu().tolist()
    completed_cpu = completed_tensor.detach().cpu().tolist()
    move_cpu = move_numbers.detach().cpu().tolist()

    code_names = {
        CODE_CHECKMATE: "CHECKMATE",
        CODE_STALEMATE: "STALEMATE",
        CODE_FIFTY: "FIFTY_MOVE",
        CODE_INSUFFICIENT: "INSUFFICIENT_MATERIAL",
        CODE_THREEFOLD: "THREEFOLD_REPETITION",
        CODE_MAX_MOVES: "MAX_MOVES",
        CODE_UNKNOWN: "UNKNOWN",
    }

    results: list[Optional[SelfPlayResult]] = [None] * num_games
    for gi in range(num_games):
        completed = bool(completed_cpu[gi])
        code = int(code_cpu[gi])
        result = int(result_cpu[gi]) if completed else None
        # A game that terminates before making a move has zero samples.  The
        # number of stored positions is move_numbers-1, except for repetition
        # which is checked before the move and therefore has the same count.
        sample_count = max(0, int(move_cpu[gi]) - 1)
        if completed and sample_count > 0:
            s = sample_states[gi, :sample_count].detach().cpu().numpy().astype(np.float32, copy=False)
            p = sample_policies[gi, :sample_count].detach().cpu().numpy()
            pl = sample_players[gi, :sample_count].detach().cpu().numpy()
            data = []
            for j in range(sample_count):
                player = int(pl[j])
                if result == 0:
                    value = 0.0
                elif result == 1:
                    value = float(player)
                else:
                    value = float(-player)
                data.append((s[j], p[j], value))
        else:
            data = []

        results[gi] = SelfPlayResult(
            training_data=data,
            result=result,
            termination=code_names.get(code, "UNKNOWN"),
            moves_played=sample_count,
            completed=completed,
        )

    print("\\n" + "=" * 60)
    print("GPU SELF-PLAY COMPLETE")
    print("=" * 60)
    for i, result in enumerate(results):
        print(
            f"Game {i + 1}: moves={result.moves_played} | "
            f"result={result.result} | termination={result.termination} | "
            f"completed={result.completed}"
        )
    total_samples = sum(len(r.training_data) for r in results if r is not None)
    print("Training samples:", total_samples)

    # --------------------------------------------------------------
    # MCTS TARGET DIAGNOSTIC SUMMARY
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MCTS TARGET DIAGNOSTICS")
    print("=" * 60)

    if diagnostic_positions > 0:
        print(f"Positions measured:           {diagnostic_positions:,}")
        print(f"Average legal moves:          {diagnostic_legal_moves_sum / diagnostic_positions:.2f}")
        print(f"Average policy entropy:       {diagnostic_entropy_sum / diagnostic_positions:.4f}")
        print(f"Average normalized entropy:   {diagnostic_normalized_entropy_sum / diagnostic_positions:.4f}")
        print(f"Average max visit probability:{diagnostic_max_prob_sum / diagnostic_positions:.4f}")
        print("\nMax visit probability distribution:")
        for label, count in diagnostic_max_prob_bins.items():
            pct = 100.0 * count / diagnostic_positions
            print(f"  {label:>8}: {count:7,} ({pct:6.2f}%)")
    else:
        print("No MCTS diagnostic positions were collected.")

    print("=" * 60)

    return results



def _multi_gpu_self_play_worker(
    rank: int,
    device_id: int,
    num_games: int,
    checkpoint_path: str,
    output_path: str,
    num_simulations: int,
    max_moves: int,
    temperature: float,
    temperature_moves: int,
    late_temperature: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    batch_size: int,
    seed: int,
    model_kwargs: Optional[dict] = None,
):
    """Run one independent self-play shard on one CUDA device.

    Each worker owns its model, GPU chess state and MCTS tree. Results are
    written to a temporary file instead of being sent through multiprocessing
    IPC, because completed self-play data contains large 4544-action policies.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.cuda.set_device(device_id)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from model.chess_net import ChessNet
    from environment.action_encoder import ActionEncoder

    encoder = ActionEncoder()
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("action_space_size", encoder.size())
    model = ChessNet(**model_kwargs).to(
        torch.device(f"cuda:{device_id}")
    )
    model.to(memory_format=torch.channels_last)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[Self-play worker {rank}] GPU {device_id}: "
        f"{torch.cuda.get_device_name(device_id)} | games={num_games}",
        flush=True,
    )

    results = play_games(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature=temperature,
        temperature_moves=temperature_moves,
        late_temperature=late_temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
    )

    torch.save(results, output_path)
    print(
        f"[Self-play worker {rank}] finished and saved {len(results)} games",
        flush=True,
    )


def play_games_multi_gpu(
    model,
    checkpoint_path,
    num_games=8,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    late_temperature=0.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
    seed=42,
    model_kwargs: Optional[dict] = None,
):
    """Split self-play across all visible CUDA GPUs.

    The parent process does not run MCTS. One spawned process is created per
    GPU, and each process runs an independent half/batch of the games. This
    targets the expensive self-play stage only; neural-network training remains
    unchanged in the parent process.
    """
    if num_games <= 0:
        return []

    if next(model.parameters()).device.type != "cuda":
        return play_games(
            model=model,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            temperature=temperature,
            temperature_moves=temperature_moves,
            late_temperature=late_temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return play_games(
            model=model,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            temperature=temperature,
            temperature_moves=temperature_moves,
            late_temperature=late_temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
        )

    # Use every visible GPU. For your current 2-GPU setup this becomes
    # 128 games on GPU 0 + 128 games on GPU 1 when num_games=256.
    workers = min(gpu_count, num_games)
    game_counts = [num_games // workers] * workers
    for i in range(num_games % workers):
        game_counts[i] += 1

    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    temp_dir = tempfile.mkdtemp(prefix="chess_selfplay_multi_gpu_")
    ctx = mp.get_context("spawn")
    processes = []
    output_paths = []

    try:
        print("\nMULTI-GPU SELF-PLAY", flush=True)
        print(f"Visible GPUs: {gpu_count}", flush=True)
        print(f"Self-play workers: {workers}", flush=True)
        print(f"Games per GPU: {game_counts}", flush=True)

        # Free the parent's model VRAM while workers run. The model is moved
        # back to CUDA by rl_training.py after self-play for the fast training
        # phase.
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for rank, (device_id, worker_games) in enumerate(enumerate(game_counts)):
            output_path = os.path.join(temp_dir, f"worker_{rank}.pt")
            output_paths.append(output_path)
            process = ctx.Process(
                target=_multi_gpu_self_play_worker,
                args=(
                    rank,
                    device_id,
                    worker_games,
                    checkpoint_path,
                    output_path,
                    num_simulations,
                    max_moves,
                    temperature,
                    temperature_moves,
                    late_temperature,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    batch_size,
                    seed + rank,
                    model_kwargs,
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failed = [
            (rank, process.exitcode)
            for rank, process in enumerate(processes)
            if process.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"Multi-GPU self-play worker failure: {failed}")

        results = []
        for output_path in output_paths:
            worker_results = torch.load(
                output_path, map_location="cpu", weights_only=False
            )
            results.extend(worker_results)

        print(
            f"MULTI-GPU SELF-PLAY COMPLETE: {len(results)} games",
            flush=True,
        )
        return results
    finally:
        # Restore the parent's model to its original CUDA device so the
        # existing training code can continue unchanged.
        original_device = next(model.parameters()).device
        if original_device.type == "cpu":
            model.to("cuda:0")
            model.to(memory_format=torch.channels_last)
        shutil.rmtree(temp_dir, ignore_errors=True)

def play_games(
    model,
    num_games=8,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    late_temperature=0.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    """Run GPU-native self-play when the model is CUDA; otherwise use legacy path."""
    if next(model.parameters()).device.type == "cuda":
        return _play_games_gpu(
            model=model,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            temperature=temperature,
            temperature_moves=temperature_moves,
            late_temperature=late_temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
        )

    # CPU fallback imports are lazy so CUDA runs never import python-chess.
    from training.self_play_legacy import play_games as legacy_play_games
    return legacy_play_games(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature=temperature,
        temperature_moves=temperature_moves,
        late_temperature=late_temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
    )


def play_game(
    model,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    late_temperature=0.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    return play_games(
        model=model,
        num_games=1,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature=temperature,
        temperature_moves=temperature_moves,
        late_temperature=late_temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
    )[0]
