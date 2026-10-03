"""Role-based, provenance-checked clean-run checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import os
import random
import uuid

import numpy as np
import torch

from az.provenance import (
    RunManifest,
    assert_artifact_matches_run,
    owned_artifact_path,
)


CHECKPOINT_ROLES = {"best", "candidate", "iteration"}


def checkpoint_relative_path(role: str, iteration: int | None = None) -> str:
    if role not in CHECKPOINT_ROLES:
        raise ValueError(f"Unknown checkpoint role: {role}")
    if role == "iteration":
        if iteration is None or iteration < 0:
            raise ValueError("iteration checkpoint requires non-negative iteration.")
        return f"checkpoints/iteration_{iteration}.pt"
    if iteration is not None:
        raise ValueError(f"{role} checkpoint must not have an iteration suffix.")
    return f"checkpoints/{role}_model.pt"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def save_checkpoint(
    *,
    root: str | Path,
    manifest: RunManifest,
    role: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    training_step: int,
    scheduler: Any = None,
    scaler: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint with explicit role, provenance, and training state."""

    if iteration < 0 or training_step < 0:
        raise ValueError("iteration and training_step must be non-negative.")
    relative = checkpoint_relative_path(role, iteration if role == "iteration" else None)
    path = owned_artifact_path(root, relative)
    metadata = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "config_sha256": manifest.config_sha256,
        "kind": "clean_checkpoint",
        "role": role,
        "iteration": int(iteration),
        "training_step": int(training_step),
    }
    payload = {
        "metadata": metadata,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "configuration": manifest.configuration,
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    *,
    root: str | Path,
    manifest: RunManifest,
    role: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    iteration: int | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only a same-run, same-role clean artifact."""

    relative = checkpoint_relative_path(role, iteration if role == "iteration" else None)
    path = owned_artifact_path(root, relative)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise RuntimeError(f"Checkpoint is not a clean artifact: {path}")
    metadata = payload["metadata"]
    assert_artifact_matches_run(metadata, manifest, path)
    if metadata.get("kind") != "clean_checkpoint" or metadata.get("role") != role:
        raise RuntimeError(f"Checkpoint role mismatch for {path}.")
    if role == "iteration" and metadata.get("iteration") != iteration:
        raise RuntimeError(f"Iteration checkpoint metadata mismatch for {path}.")
    _unwrap(model).load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        if payload.get("optimizer_state_dict") is None:
            raise RuntimeError(f"Checkpoint has no optimizer state: {path}")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
