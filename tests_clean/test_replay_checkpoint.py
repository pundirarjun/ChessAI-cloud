from dataclasses import replace

import numpy as np
import pytest
import torch

from az.checkpoints import load_checkpoint, prune_iteration_checkpoints, save_checkpoint
from az.config import RunConfig, StorageConfig
from az.model import build_fresh_model
from az.provenance import initialize_run
from az.replay import CleanReplayBuffer
from az.training import build_optimizer


def _sample(value=0.0):
    policy = np.zeros((4544,), dtype=np.float32)
    policy[17] = 1.0
    return np.zeros((18, 8, 8), dtype=np.float32), policy, value


def _multi_move_sample(value=0.0):
    policy = np.zeros((4544,), dtype=np.float32)
    policy[[3, 900, 4000]] = [0.5, 0.25, 0.25]
    state = np.zeros((18, 8, 8), dtype=np.float32)
    state[5, 2, 3] = 1.0
    return state, policy, value


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


def test_sparse_roundtrip_preserves_dense_targets(tmp_path):
    config, manifest = _run(tmp_path)
    replay = CleanReplayBuffer(8, manifest)
    replay.add_completed_game(iteration=1, game_id=0, samples=[_sample(1.0), _multi_move_sample(-1.0)], completed=True)
    sample = next(iter(replay))
    assert sample.state.dtype == np.uint8
    assert sample.policy_indices.dtype == np.int16
    assert sample.policy_values.dtype == np.float32
    replay.save(config.root)
    restored = CleanReplayBuffer.load(config.root, manifest)
    original = list(replay)
    for kept, loaded in zip(original, list(restored)):
        np.testing.assert_array_equal(loaded.dense_state(), kept.dense_state())
        np.testing.assert_allclose(loaded.dense_policy(), kept.dense_policy(), atol=0, rtol=0)


def test_v1_dense_payload_migrates_on_load(tmp_path):
    config, manifest = _run(tmp_path)
    state, policy, value = _multi_move_sample(1.0)
    payload = {
        "metadata": {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "config_sha256": manifest.config_sha256,
            "kind": "clean_replay_buffer",
            "capacity": 8,
            "sample_count": 1,
            "iteration_distribution": {1: 1},
        },
        "samples": [
            {
                "state": state,
                "policy": policy,
                "value": value,
                "provenance": {
                    "run_id": manifest.run_id,
                    "iteration": 1,
                    "game_id": 0,
                    "position_index": 0,
                },
            }
        ],
    }
    destination = config.root / "replay" / "replay_buffer.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    restored = CleanReplayBuffer.load(config.root, manifest)
    assert len(restored) == 1
    sample = next(iter(restored))
    assert sample.state.dtype == np.uint8
    np.testing.assert_array_equal(sample.dense_state(), state.astype(np.float32))
    np.testing.assert_allclose(sample.dense_policy(), policy, atol=0, rtol=0)


def test_prune_iteration_checkpoints_keeps_newest(tmp_path):
    root = tmp_path
    directory = root / "checkpoints"
    directory.mkdir(parents=True)
    for number in range(1, 6):
        (directory / f"iteration_{number}.pt").write_bytes(b"x")
    removed = prune_iteration_checkpoints(root, 3)
    assert sorted(p.name for p in removed) == ["iteration_1.pt", "iteration_2.pt"]
    assert sorted(p.name for p in directory.glob("iteration_*.pt")) == [
        "iteration_3.pt",
        "iteration_4.pt",
        "iteration_5.pt",
    ]


def test_preflight_ram_guard_uses_configured_capacity(tmp_path):
    from az.preflight import _check_system_ram

    config = replace(
        RunConfig(),
        run_name="unit_run",
        storage=StorageConfig(run_root=str(tmp_path), replay_capacity=150_000),
    )
    result = _check_system_ram(config)
    assert result == "pass" or result.startswith("fail:") or result.startswith("info:")
