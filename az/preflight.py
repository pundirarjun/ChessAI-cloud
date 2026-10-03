"""Pre-training validation; a failed gate blocks the clean training command."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util

import torch

from az.config import RunConfig
from az.provenance import RunManifest, load_manifest


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    checks: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def run_preflight(
    *,
    config: RunConfig,
    root: str | Path,
    require_two_gpus: bool,
    require_cpp_engine: bool,
) -> PreflightReport:
    """Validate static run invariants without starting training."""

    checks: dict[str, str] = {}
    try:
        config.validate()
        checks["configuration"] = "pass"
    except Exception as error:
        checks["configuration"] = f"fail: {error}"
    try:
        manifest = load_manifest(root)
        if manifest.config_sha256 != __import__("hashlib").sha256(
            config.canonical_json().encode("utf-8")
        ).hexdigest():
            raise RuntimeError("config does not match the run manifest.")
        checks["clean_manifest"] = "pass"
    except Exception as error:
        checks["clean_manifest"] = f"fail: {error}"
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_two_gpus:
        checks["two_gpu_visibility"] = (
            "pass" if gpu_count >= 2 else f"fail: expected two CUDA GPUs, found {gpu_count}"
        )
    else:
        checks["gpu_visibility"] = f"info: {gpu_count} CUDA GPU(s) visible"
    if require_cpp_engine:
        available = importlib.util.find_spec("az_cpp_mcts") is not None
        checks["cpp_engine"] = (
            "pass" if available else "fail: az_cpp_mcts extension is not available"
        )
    else:
        checks["cpp_engine"] = "not required for this reference-only preflight"
    # CPU correctness remains a first-class gate. This verifies imports and
    # action-space identity; full pytest is invoked by the documented command.
    try:
        from environment.action_encoder import ActionEncoder
        from environment.state_encoder import StateEncoder
        from model.chess_net import ChessNet

        if ActionEncoder().size() != 4544:
            raise RuntimeError("action encoder size is not 4544")
        model = ChessNet()
        policy, value = model(torch.zeros((1, 18, 8, 8)))
        if policy.shape != (1, 4544) or value.shape != (1, 1):
            raise RuntimeError("network output shape mismatch")
        checks["baseline_contract"] = "pass"
    except Exception as error:
        checks["baseline_contract"] = f"fail: {error}"
    passed = all(value == "pass" or value.startswith("info:") or value.startswith("not required") for value in checks.values())
    return PreflightReport(passed=passed, checks=checks)
