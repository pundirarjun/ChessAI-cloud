"""Run manifest and strict clean-artifact provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re
import uuid

from az.config import RunConfig


MANIFEST_NAME = "run_manifest.json"
_LEGACY_ARTIFACT = re.compile(
    r"(?i)(?:^|[_-])(?:rl(?:[_-]?iteration)?[_-]?\d+|"
    r"pretrained(?:[_-]?phase)?[_-]?\d*|replay[_-]?buffer[_-]?rl\d+)(?:$|[_\-.])"
)


def _resolved_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Artifact path escapes run root: {resolved}") from error
    return resolved


def reject_legacy_artifact(path: str | Path) -> None:
    """Reject the names used by discarded training chains."""

    candidate = Path(path)
    if any(_LEGACY_ARTIFACT.search(part) for part in candidate.parts):
        raise RuntimeError(
            f"Legacy training artifact is forbidden in a clean run: {candidate}"
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    run_name: str
    created_at_utc: str
    config_sha256: str
    configuration: dict[str, Any]
    clean_start: bool
    forbidden_artifact_patterns: tuple[str, ...]

    @classmethod
    def create(cls, config: RunConfig) -> "RunManifest":
        config.validate()
        return cls(
            schema_version=1,
            run_id=str(uuid.uuid4()),
            run_name=config.run_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            config_sha256=hashlib.sha256(
                config.canonical_json().encode("utf-8")
            ).hexdigest(),
            configuration=config.to_dict(),
            clean_start=True,
            forbidden_artifact_patterns=(r"RLn", r"rl_iteration_n", r"pretrained_phase_n"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "created_at_utc": self.created_at_utc,
            "config_sha256": self.config_sha256,
            "configuration": self.configuration,
            "clean_start": self.clean_start,
            "forbidden_artifact_patterns": list(self.forbidden_artifact_patterns),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunManifest":
        return cls(
            schema_version=int(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            run_name=str(raw["run_name"]),
            created_at_utc=str(raw["created_at_utc"]),
            config_sha256=str(raw["config_sha256"]),
            configuration=dict(raw["configuration"]),
            clean_start=bool(raw["clean_start"]),
            forbidden_artifact_patterns=tuple(raw["forbidden_artifact_patterns"]),
        )


def initialize_run(config: RunConfig) -> RunManifest:
    """Create exactly one empty clean-run namespace and manifest."""

    config.validate()
    reject_legacy_artifact(config.run_name)
    root = config.root
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"Clean run already exists at {root}; load its manifest instead."
        )
    root.mkdir(parents=True, exist_ok=False)
    for child in ("checkpoints", "replay", "logs", "benchmarks"):
        (root / child).mkdir()
    manifest = RunManifest.create(config)
    _write_json_atomic(manifest_path, manifest.to_dict())
    config.save_json(root / "config.json")
    return manifest


def load_manifest(root: str | Path) -> RunManifest:
    root = Path(root).resolve()
    with (root / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        manifest = RunManifest.from_dict(json.load(handle))
    if manifest.schema_version != 1 or not manifest.clean_start:
        raise RuntimeError("Not a valid clean-start AlphaZero run manifest.")
    return manifest


def owned_artifact_path(root: str | Path, relative: str | Path) -> Path:
    root = Path(root).resolve()
    relative = Path(relative)
    reject_legacy_artifact(relative)
    path = _resolved_inside(root / relative, root)
    return path


def assert_artifact_matches_run(
    artifact_metadata: dict[str, Any],
    manifest: RunManifest,
    artifact_path: str | Path,
) -> None:
    reject_legacy_artifact(artifact_path)
    if artifact_metadata.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported clean artifact schema: {artifact_path}")
    if artifact_metadata.get("run_id") != manifest.run_id:
        raise RuntimeError(
            f"Artifact does not belong to clean run {manifest.run_id}: {artifact_path}"
        )
    if artifact_metadata.get("config_sha256") != manifest.config_sha256:
        raise RuntimeError(f"Artifact configuration provenance mismatch: {artifact_path}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
