import pytest

from az.gating import EvaluationGame, evaluate_gate


def _games(results):
    return [
        EvaluationGame(
            game_id=index,
            candidate_color="white" if index % 2 == 0 else "black",
            result_for_candidate=result,
            termination="CHECKMATE" if result is not None else "MAX_MOVES",
            seed=index,
            simulations=400,
            temperature=0.0,
        )
        for index, result in enumerate(results)
    ]


def test_gate_excludes_truncations_and_needs_enough_completed_games():
    decision = evaluate_gate(
        _games([1, 1, None, None]),
        promotion_score=0.55,
        confidence_level=0.95,
        min_completed_games=4,
    )
    assert not decision.promote
    assert decision.reason == "insufficient_completed_games"
    assert decision.truncations == 2


def test_gate_rejects_unbalanced_color_assignments():
    games = _games([1, 1])
    games[1] = EvaluationGame(
        game_id=1, candidate_color="white", result_for_candidate=1,
        termination="CHECKMATE", seed=1, simulations=400, temperature=0.0,
    )
    with pytest.raises(RuntimeError, match="color balanced"):
        evaluate_gate(games, promotion_score=0.5, confidence_level=0.95, min_completed_games=2)
