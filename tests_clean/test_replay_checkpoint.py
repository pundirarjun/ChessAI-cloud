from dataclasses import replace

import numpy as np
import pytest
import torch

from az.checkpoints import load_checkpoint, save_checkpoint
from az.config import RunConfig, StorageConfig
from az.model import build_fresh_model
from az.provenance import initialize_run
from az.replay import CleanReplayBuffer
from az.training import build_optimizer


def _sample(value=0.0):
    policy = np.zeros((4544,), dtype=np.float32)
    policy[17] = 1.0
    return np.zeros((18, 8, 8), dtype=np.float32), policy, value


def _run(tmp_path):
    config = replace(RunConfig(), run_name="unit_run", storage=StorageConfig(run_root=str(tmp_path), replay_capacity=8))
    return config, initialize_run(config)


def test_replay_requires_completed_games_and_persists_provenance(tmp_path):
    config, manifest = _run(tmp_path)
    replay = CleanReplayBuffer(8, manifest)
    assert replay.add_completed_game(iteration=1, game_id=0, samples=[_sample()], completed=False) == 0
    assert replay.add_completed_game(iteration=1, game_id=0, samples=[_sample(1.0)], completed=True) == 1
    replay.save(config.root)
    restored = CleanReplayBuffer.load(config.root, manifest)
    assert len(restored) == 1
    assert restored.validate()["by_iteration"] == {1: 1}


def test_checkpoint_cannot_be_loaded_as_a_different_role(tmp_path):
    config, manifest = _run(tmp_path)
    model = build_fresh_model(config.architecture, config.runtime, "cpu")
    optimizer = build_optimizer(model, config.training)
    save_checkpoint(
        root=config.root, manifest=manifest, role="best", model=model,
        optimizer=optimizer, iteration=0, training_step=0,
    )
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            root=config.root, manifest=manifest, role="candidate",
            model=model, optimizer=optimizer,
        )
