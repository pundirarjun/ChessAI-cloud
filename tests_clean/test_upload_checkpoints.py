"""Tests for the manual checkpoint export tool (tools/upload_checkpoints.py)."""

from __future__ import annotations

import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

from tools import upload_checkpoints


def _fake_run(root: Path) -> Path:
    run = root / "run"
    (run / "checkpoints" / "nested").mkdir(parents=True)
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "run_state.json").write_text('{"iteration": 5}', encoding="utf-8")
    (run / "replay").mkdir()
    (run / "replay" / "replay_buffer.pt").write_bytes(b"replay")
    (run / "checkpoints" / "iteration_5.pt").write_bytes(b"weights")
    (run / "checkpoints" / "best_model.pt").write_bytes(b"weights")
    (run / "checkpoints" / "nested" / "extra.pt").write_bytes(b"weights")
    return run


def test_build_zip_includes_checkpoints_and_provenance(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    out = tmp_path / "checkpoints.zip"
    count = upload_checkpoints.build_zip(run, out)
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert names == {
        "manifest.json",
        "run_state.json",
        "checkpoints/iteration_5.pt",
        "checkpoints/best_model.pt",
        "checkpoints/nested/extra.pt",
    }
    assert count == 5


def test_build_zip_tolerates_missing_provenance_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "iteration_1.pt").write_bytes(b"weights")
    out = tmp_path / "checkpoints.zip"
    count = upload_checkpoints.build_zip(run, out)
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == ["checkpoints/iteration_1.pt"]
    assert count == 1


def test_resolve_credentials_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
    monkeypatch.setenv("KAGGLE_KEY", "envkey")
    upload_checkpoints.resolve_credentials()
    assert os.environ["KAGGLE_USERNAME"] == "envuser"


def test_resolve_credentials_fails_without_any_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle/kaggle.json
    fake = types.ModuleType("kaggle_secrets")

    class _Blocked:
        def __init__(self) -> None:
            raise RuntimeError("secrets blocked in test")

    fake.UserSecretsClient = _Blocked
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake)
    with pytest.raises(SystemExit):
        upload_checkpoints.resolve_credentials()


def _recorder(probe_rc: int = 0, probe_writes: bool = True, probe_content=None):
    """Fake subprocess.run for _upload; returns (fake_run, seen_commands).

    probe_content mimics what `kaggle datasets metadata` (2.0.2) actually
    writes: title and friends but no `id`/`id_no`.
    """

    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        argv = [str(item) for item in command]
        seen.append(argv)
        if argv[1:3] == ["datasets", "metadata"]:
            if probe_rc == 0 and probe_writes:
                dest = Path(argv[argv.index("-p") + 1])
                content = {"title": "original title"} if probe_content is None else probe_content
                dest.joinpath("dataset-metadata.json").write_text(
                    json.dumps(content),
                    encoding="utf-8",
                )
            return types.SimpleNamespace(returncode=probe_rc, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake_run, seen


def test_upload_keeps_probed_metadata_and_injects_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, seen = _recorder()
    monkeypatch.setattr(upload_checkpoints.subprocess, "run", fake_run)
    upload_checkpoints._upload(tmp_path, "owner/ds", "iteration 5 checkpoints", False, "kaggle")
    data = json.loads((tmp_path / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert data["title"] == "original title"
    assert data["id"] == "owner/ds"
    assert seen[0][1:3] == ["datasets", "metadata"]
    assert seen[1][1:3] == ["datasets", "version"]
    assert seen[1][seen[1].index("-p") + 1] == str(tmp_path)
    assert "-d" not in seen[1]


def test_upload_leaves_existing_id_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, _ = _recorder(probe_content={"id": "owner/real-ds", "title": "original title"})
    monkeypatch.setattr(upload_checkpoints.subprocess, "run", fake_run)
    upload_checkpoints._upload(tmp_path, "owner/ds", "iteration 5 checkpoints", False, "kaggle")
    data = json.loads((tmp_path / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert data["id"] == "owner/real-ds"
    assert data["title"] == "original title"


def test_ensure_metadata_identity_replaces_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset-metadata.json"
    path.write_text("not json", encoding="utf-8")
    upload_checkpoints._ensure_metadata_identity(path, "owner/ds")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"] == "owner/ds"
    assert data["title"] == "ChessAI clean AZ checkpoints"


def test_upload_falls_back_when_probe_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, seen = _recorder(probe_writes=False)
    monkeypatch.setattr(upload_checkpoints.subprocess, "run", fake_run)
    upload_checkpoints._upload(tmp_path, "owner/ds", "iteration 5 checkpoints", False, "kaggle")
    data = json.loads((tmp_path / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert data["title"] == "ChessAI clean AZ checkpoints"
    assert data["id"] == "owner/ds"
    assert seen[1][1:3] == ["datasets", "version"]


def test_upload_creates_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, seen = _recorder(probe_rc=1)
    monkeypatch.setattr(upload_checkpoints.subprocess, "run", fake_run)
    upload_checkpoints._upload(tmp_path, "owner/ds", "iteration 5 checkpoints", False, "kaggle")
    assert (tmp_path / "dataset-metadata.json").is_file()
    assert seen[1][1:3] == ["datasets", "create"]


def test_upload_prune_old_adds_delete_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, seen = _recorder()
    monkeypatch.setattr(upload_checkpoints.subprocess, "run", fake_run)
    upload_checkpoints._upload(tmp_path, "owner/ds", "iteration 5 checkpoints", True, "kaggle")
    assert seen[1][1:3] == ["datasets", "version"]
    assert seen[1][-1] == "-d"
