from dataclasses import replace

import pytest
import torch

import az.evaluation as evaluation
from az.config import RunConfig, StorageConfig
from az.evaluation import _play_one


def test_play_one_temperature_schedule(tmp_path, monkeypatch):
    """Sampled opening moves, then deterministic play."""
    config = replace(
        RunConfig(),
        run_name="unit_run",
        storage=StorageConfig(run_root=str(tmp_path)),
        evaluation=replace(
            RunConfig().evaluation,
            temperature=1.0,
            temperature_moves=3,
            max_moves=5,
        ),
    )
    seen: list[float] = []

    class FakeSearch:
        def search(self, state, **kwargs):
            self.state = state

        def select_actions(self, temperature):
            seen.append(float(temperature))
            legal = self.state.legal_move_mask()[0]
            return int(torch.nonzero(legal, as_tuple=False)[0].item())

        def advance(self, action):
            return self.state.push_actions(
                torch.tensor([action], dtype=torch.long, device=self.state.device)
            )

    monkeypatch.setattr(
        evaluation, "_make_search", lambda model, cfg, device, seed: FakeSearch()
    )

    game = _play_one(
        candidate=None,
        best=None,
        config=config,
        game_id=0,
        device=torch.device("cpu"),
    )

    assert seen == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert game.temperature == 1.0
    assert game.termination == "MAX_MOVES"


def test_evaluation_temperature_moves_validation():
    config = RunConfig()
    with pytest.raises(ValueError, match="temperature_moves"):
        replace(
            config, evaluation=replace(config.evaluation, temperature_moves=-1)
        ).validate()
