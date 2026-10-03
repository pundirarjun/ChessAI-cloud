"""Audited model construction, kept independent from legacy checkpoints."""

from __future__ import annotations

import torch

from az.config import ArchitectureConfig, RuntimeConfig
from model.chess_net import ChessNet


def build_fresh_model(
    architecture: ArchitectureConfig,
    runtime: RuntimeConfig,
    device: str | torch.device,
) -> ChessNet:
    """Build a fresh network; this function never accepts a checkpoint path."""

    model = ChessNet(
        input_channels=architecture.input_channels,
        channels=architecture.channels,
        num_blocks=architecture.residual_blocks,
        policy_channels=architecture.policy_channels,
        action_space_size=architecture.action_space_size,
        value_channels=architecture.value_channels,
    ).to(device)
    if torch.device(device).type == "cuda" and runtime and runtime.allow_tf32:
        if runtime and True:
            model.to(memory_format=torch.channels_last)
    return model
