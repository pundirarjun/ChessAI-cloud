from dataclasses import replace

import pytest

from az.config import RunConfig, StorageConfig
from az.provenance import initialize_run, load_manifest, owned_artifact_path, reject_legacy_artifact


def test_clean_run_manifest_and_legacy_rejection(tmp_path):
    config = replace(
        RunConfig(),
        run_name="unit_run",
        storage=StorageConfig(run_root=str(tmp_path), replay_capacity=8),
    )
    manifest = initialize_run(config)
    assert load_manifest(config.root).run_id == manifest.run_id
    assert owned_artifact_path(config.root, "checkpoints/best_model.pt").parent.name == "checkpoints"
    with pytest.raises(RuntimeError, match="Legacy"):
        owned_artifact_path(config.root, "checkpoints/rl_iteration_56.pt")
    with pytest.raises(RuntimeError, match="Legacy"):
        reject_legacy_artifact("pretrained_phase1.pt")


def test_config_rejects_wrong_baseline_action_shape():
    with pytest.raises(ValueError, match="4544"):
        replace(RunConfig(), architecture=replace(RunConfig().architecture, action_space_size=4096)).validate()
