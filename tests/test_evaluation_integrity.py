import torch

from evaluation.evaluate_models import (
    _set_model_identity,
    build_game_assignments,
    evaluate_models,
    resolve_evaluation_settings,
    select_assigned_model,
)


def test_assignments_are_balanced_and_dispatch_by_identity():
    assignments = build_game_assignments(300)
    assert sum(assignment.model_a_is_white for assignment in assignments) == 150

    model_a = _set_model_identity(torch.nn.Linear(1, 1), "A", "a.pt")
    model_b = _set_model_identity(torch.nn.Linear(1, 1), "B", "b.pt")
    for assignment in assignments:
        white, white_id = select_assigned_model(model_a, model_b, assignment, False)
        black, black_id = select_assigned_model(model_a, model_b, assignment, True)
        assert white_id == assignment.white_model_id
        assert black_id == assignment.black_model_id
        assert white is (model_a if white_id == "A" else model_b)
        assert black is (model_a if black_id == "A" else model_b)


def test_benchmark_mode_is_reproducible_without_changing_default_temperature():
    default = resolve_evaluation_settings()
    benchmark = resolve_evaluation_settings(temperature=0.75, seed=7, benchmark_mode=True)

    assert default.temperature == 0.25
    assert default.seed is None
    assert benchmark.temperature == 0.0
    assert benchmark.seed == 7


def test_evaluation_excludes_truncations_from_score():
    results = [
        {
            "result": 1,
            "termination": "CHECKMATE",
            "moves": 10,
            "model_a_color": "White",
            "white_model_id": "A",
            "black_model_id": "B",
        },
        {
            "result": None,
            "termination": "MAX_MOVES",
            "moves": 400,
            "model_a_color": "Black",
            "white_model_id": "B",
            "black_model_id": "A",
        },
    ]

    summary = evaluate_models(results, "A", "B", elapsed=2.0)

    assert summary["a_wins"] == 1
    assert summary["truncated"] == 1
    assert summary["completed_games"] == 1
    assert summary["a_score"] == 1.0
