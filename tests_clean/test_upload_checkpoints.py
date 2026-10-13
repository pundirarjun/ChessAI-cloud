"""Tests for the manual checkpoint export tool (tools/upload_checkpoints.py)."""

from __future__ import annotations

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
