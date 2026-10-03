"""Clean AlphaZero iteration lifecycle: best -> self-play -> candidate -> gate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid

import torch

from az.checkpoints import load_checkpoint, save_checkpoint
from az.config import RunConfig
from az.evaluation import evaluate_candidate
from az.model import build_fresh_model
from az.provenance import RunManifest, initialize_run, load_manifest
from az.replay import CleanReplayBuffer
from az.reproducibility import seed_everything
from az.selfplay import generate_self_play
from az.training import (
    build_optimizer,
    close_distributed,
    distributed_context,
    train_candidate,
)


STATE_NAME = "run_state.json"


def initialize_clean_run(config: RunConfig) -> tuple[RunManifest, Path]:
    """Initialize a brand-new random best model and an empty replay buffer."""

    manifest = initialize_run(config)
    root = config.root
    seed_everything(config.runtime)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_fresh_model(config.architecture, config.runtime, device)
    optimizer = build_optimizer(model, config.training)
    save_checkpoint(
        root=root,
        manifest=manifest,
        role="best",
        model=model,
        optimizer=optimizer,
        iteration=0,
        training_step=0,
        extra={"origin": "fresh_random_initialization", "candidate_promoted": True},
    )
    replay = CleanReplayBuffer(config.storage.replay_capacity, manifest)
    replay.save(root)
    _write_state(root, {"iteration": 0, "best_iteration": 0, "training_step": 0})
    return manifest, root


def train_candidate_worker(config: RunConfig) -> dict:
    """Torchrun entry point; each rank participates in synchronized training."""

    root = config.root
    manifest = load_manifest(root)
    context = distributed_context(config.runtime)
    try:
        seed_everything(config.runtime, rank=context.rank)
        replay = CleanReplayBuffer.load(root, manifest)
        candidate = build_fresh_model(config.architecture, config.runtime, context.device)
        optimizer = build_optimizer(candidate, config.training)
        best_payload = load_checkpoint(
            root=root,
            manifest=manifest,
            role="best",
            model=candidate,
            optimizer=optimizer,
            map_location=context.device,
        )
        model, optimizer, metrics, scaler = train_candidate(
            model=candidate,
            replay=replay,
            config=config.training,
            runtime=config.runtime,
            context=context,
            optimizer=optimizer,
        )
        if context.world_size > 1:
            torch.distributed.barrier()
        if context.is_primary:
            state = _read_state(root)
            iteration = state["iteration"] + 1
            save_checkpoint(
                root=root,
                manifest=manifest,
                role="candidate",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                training_step=state["training_step"] + metrics.steps,
                scaler=scaler,
                extra={
                    "parent_best_iteration": best_payload["metadata"]["iteration"],
                    "training_metrics": metrics.to_dict(),
                },
            )
            return metrics.to_dict()
        return {"non_primary_rank": context.rank}
    finally:
        close_distributed(context)


def run_iteration(config: RunConfig, *, entry_script: str | Path) -> dict:
    """Run a complete iteration without ever loading legacy artifacts."""

    root = config.root
    manifest = load_manifest(root)
    state = _read_state(root)
    iteration = int(state["iteration"]) + 1
    seed_everything(config.runtime)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = build_fresh_model(config.architecture, config.runtime, device)
    best_optimizer = build_optimizer(best_model, config.training)
    load_checkpoint(
        root=root,
        manifest=manifest,
        role="best",
        model=best_model,
        optimizer=best_optimizer,
        map_location=device,
    )
    best_path = root / "checkpoints" / "best_model.pt"
    replay = CleanReplayBuffer.load(root, manifest)
    self_play = generate_self_play(
        config=config,
        model=best_model,
        best_checkpoint=best_path,
        replay=replay,
        iteration=iteration,
    )
    replay.save(root)
    if len(replay) < config.training.batch_size:
        raise RuntimeError(
            f"Iteration {iteration} generated only {len(replay)} replay samples; "
            f"need {config.training.batch_size} before candidate training."
        )
    _launch_distributed_training(config, entry_script)
    games, decision = evaluate_candidate(root=root, config=config, manifest=manifest)

    candidate = build_fresh_model(config.architecture, config.runtime, "cpu")
    candidate_optimizer = build_optimizer(candidate, config.training)
    candidate_payload = load_checkpoint(
        root=root,
        manifest=manifest,
        role="candidate",
        model=candidate,
        optimizer=candidate_optimizer,
        map_location="cpu",
    )
    next_step = int(candidate_payload["metadata"]["training_step"])
    if decision.promote:
        save_checkpoint(
            root=root,
            manifest=manifest,
            role="best",
            model=candidate,
            optimizer=candidate_optimizer,
            iteration=iteration,
            training_step=next_step,
            extra={"promotion": decision.to_dict()},
        )
        best_iteration = iteration
    else:
        best_iteration = int(state["best_iteration"])
    save_checkpoint(
        root=root,
        manifest=manifest,
        role="iteration",
        model=candidate,
        optimizer=candidate_optimizer,
        iteration=iteration,
        training_step=next_step,
        extra={
            "self_play": self_play.to_dict(),
            "evaluation_games": [asdict(game) for game in games],
            "promotion": decision.to_dict(),
        },
    )
    new_state = {
        "iteration": iteration,
        "best_iteration": best_iteration,
        "training_step": next_step,
    }
    _write_state(root, new_state)
    return {
        "iteration": iteration,
        "self_play": self_play.to_dict(),
        "replay": replay.validate(),
        "promotion": decision.to_dict(),
        "best_iteration": best_iteration,
    }


def _launch_distributed_training(config: RunConfig, entry_script: str | Path) -> None:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_ddp = device_count >= 2 and (config.runtime.num_gpus or 1) >= 2
    command = [sys.executable]
    if use_ddp:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    command += [str(Path(entry_script).resolve()), "train-candidate", "--config", str(config.root / "config.json")]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Candidate training failed with exit code {result.returncode}.")


def _read_state(root: str | Path) -> dict:
    with (Path(root) / STATE_NAME).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_state(root: str | Path, state: dict) -> None:
    root = Path(root)
    destination = root / STATE_NAME
    temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
