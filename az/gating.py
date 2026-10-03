"""Candidate-vs-best promotion rules with explicit truncation accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Iterable


@dataclass(frozen=True)
class EvaluationGame:
    game_id: int
    candidate_color: str
    result_for_candidate: int | None
    termination: str
    seed: int
    simulations: int
    temperature: float

    def __post_init__(self) -> None:
        if self.candidate_color not in {"white", "black"}:
            raise ValueError("candidate_color must be 'white' or 'black'.")
        if self.result_for_candidate not in {-1, 0, 1, None}:
            raise ValueError("result_for_candidate must be -1, 0, 1, or None.")


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    completed_games: int
    truncations: int
    wins: int
    losses: int
    draws: int
    score: float | None
    wilson_lower_bound: float | None
    threshold: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_gate(
    games: Iterable[EvaluationGame],
    *,
    promotion_score: float,
    confidence_level: float,
    min_completed_games: int,
    require_color_balance: bool = True,
) -> GateDecision:
    """Decide promotion from completed games only.

    A win contributes 1, a draw .5, and a loss 0. The Wilson lower confidence
    bound is evaluated against the configured score threshold, preventing a
    single lucky game from promoting a candidate.
    """

    records = list(games)
    candidate_colors = [r.candidate_color for r in records]
    if require_color_balance and candidate_colors.count("white") != candidate_colors.count("black"):
        raise RuntimeError("Candidate evaluation is not color balanced.")
    wins = sum(r.result_for_candidate == 1 for r in records)
    losses = sum(r.result_for_candidate == -1 for r in records)
    draws = sum(r.result_for_candidate == 0 for r in records)
    truncations = sum(r.result_for_candidate is None for r in records)
    completed = wins + losses + draws
    if completed < min_completed_games:
        return GateDecision(
            promote=False,
            completed_games=completed,
            truncations=truncations,
            wins=wins,
            losses=losses,
            draws=draws,
            score=None,
            wilson_lower_bound=None,
            threshold=promotion_score,
            reason="insufficient_completed_games",
        )
    score = (wins + 0.5 * draws) / completed
    lower = wilson_lower_bound(
        successes=wins + 0.5 * draws,
        trials=completed,
        confidence_level=confidence_level,
    )
    return GateDecision(
        promote=lower >= promotion_score,
        completed_games=completed,
        truncations=truncations,
        wins=wins,
        losses=losses,
        draws=draws,
        score=score,
        wilson_lower_bound=lower,
        threshold=promotion_score,
        reason="promoted" if lower >= promotion_score else "lower_bound_below_threshold",
    )


def wilson_lower_bound(
    *,
    successes: float,
    trials: int,
    confidence_level: float,
) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials].")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must be in (0, 1).")
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * ((p * (1 - p) + z * z / (4 * trials)) / trials) ** 0.5
    return (centre - margin) / denominator
