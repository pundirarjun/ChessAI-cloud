"""Clean-run self-play adapter with explicit result/provenance accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from az.config import ArchitectureConfig, RunConfig
from az.replay import CleanReplayBuffer
from training.self_play import SelfPlayResult, play_games, play_games_multi_gpu


@dataclass(frozen=True)
class SelfPlaySummary:
    games: int
    completed_games: int
    incomplete_games: int
    white_wins: int
    black_wins: int
    draws: int
    samples_added: int
    termination_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def architecture_kwargs(architecture: ArchitectureConfig) -> dict[str, int]:
    return {
        "input_channels": architecture.input_channels,
        "channels": architecture.channels,
        "num_blocks": architecture.residual_blocks,
        "policy_channels": architecture.policy_channels,
        "action_space_size": architecture.action_space_size,
        "value_channels": architecture.value_channels,
    }


def generate_self_play(
    *,
    config: RunConfig,
    model: torch.nn.Module,
    best_checkpoint: str | Path,
    replay: CleanReplayBuffer,
    iteration: int,
) -> SelfPlaySummary:
    """Generate best-model games and insert only completed-game samples."""

    config.validate()
    if iteration < 1:
        raise ValueError("Clean self-play iteration must be >= 1.")
    device = next(model.parameters()).device
    settings = config.self_play
    use_multi_gpu = (
        device.type == "cuda"
        and torch.cuda.device_count() >= 2
        and (config.runtime.num_gpus or 1) >= 2
    )
    if use_multi_gpu:
        results = play_games_multi_gpu(
            model=model,
            checkpoint_path=str(best_checkpoint),
            num_games=settings.num_games,
            num_simulations=config.mcts.simulations,
            max_moves=settings.max_moves,
            temperature=settings.temperature,
            temperature_moves=settings.temperature_moves,
            late_temperature=settings.late_temperature,
            dirichlet_alpha=settings.dirichlet_alpha,
            dirichlet_epsilon=settings.dirichlet_epsilon,
            batch_size=config.mcts.batch_size,
            seed=config.runtime.seed + iteration * 100_000,
            model_kwargs=architecture_kwargs(config.architecture),
        )
    else:
        results = play_games(
            model=model,
            num_games=settings.num_games,
            num_simulations=config.mcts.simulations,
            max_moves=settings.max_moves,
            temperature=settings.temperature,
            temperature_moves=settings.temperature_moves,
            late_temperature=settings.late_temperature,
            dirichlet_alpha=settings.dirichlet_alpha,
            dirichlet_epsilon=settings.dirichlet_epsilon,
            batch_size=config.mcts.batch_size,
        )
    return _record_results(results, replay=replay, iteration=iteration)


def _record_results(
    results: list[SelfPlayResult],
    *,
    replay: CleanReplayBuffer,
    iteration: int,
) -> SelfPlaySummary:
    completed = incomplete = white_wins = black_wins = draws = samples_added = 0
    reasons: dict[str, int] = {}
    for game_id, result in enumerate(results):
        reasons[result.termination] = reasons.get(result.termination, 0) + 1
        if not result.completed:
            incomplete += 1
            # Incomplete/truncated games always provide zero samples to clean replay.
            if result.training_data:
                raise RuntimeError("Incomplete game unexpectedly contains training data.")
            continue
        completed += 1
        samples_added += replay.add_completed_game(
            iteration=iteration,
            game_id=game_id,
            samples=result.training_data,
            completed=True,
        )
        if result.result == 1:
            white_wins += 1
        elif result.result == -1:
            black_wins += 1
        elif result.result == 0:
            draws += 1
        else:
            raise RuntimeError("Completed self-play game has no terminal result.")
    return SelfPlaySummary(
        games=len(results),
        completed_games=completed,
        incomplete_games=incomplete,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        samples_added=samples_added,
        termination_reasons=reasons,
    )
