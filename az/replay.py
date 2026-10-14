"""CPU-resident replay with mandatory clean-run/game/position provenance.

Storage format (schema v2): states are uint8 (exact 0/1 planes) and policies
are stored sparsely as (indices, values) for the ~28 legal-move entries instead
of a dense 4544-vector.  This reduces per-sample memory from ~22.8 KB to
~1.3 KB, which is what makes a 150k-sample buffer fit comfortably alongside
DDP training copies on Kaggle's ~29 GB system RAM.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator
import math
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


STATE_SHAPE = (18, 8, 8)
POLICY_SHAPE = (4544,)
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SampleProvenance:
    run_id: str
    iteration: int
    game_id: int
    position_index: int


@dataclass(frozen=True)
class ReplaySample:
    """Compact replay sample: uint8 state + sparse policy."""

    state: np.ndarray  # uint8 (18, 8, 8)
    policy_indices: np.ndarray  # int16 (N,) action indices with non-zero policy
    policy_values: np.ndarray  # float32 (N,) policy probabilities at those indices
    value: float
    provenance: SampleProvenance

    def dense_state(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def dense_policy(self) -> np.ndarray:
        policy = np.zeros(POLICY_SHAPE, dtype=np.float32)
        policy[self.policy_indices] = self.policy_values
        return policy


def _compact_state(state) -> np.ndarray:
    return np.asarray(state, dtype=np.uint8)


def _compact_policy(policy) -> tuple[np.ndarray, np.ndarray]:
    policy = np.asarray(policy, dtype=np.float32)
    nonzero = np.nonzero(policy)[0]
    return nonzero.astype(np.int16), policy[nonzero].astype(np.float32).copy()


class CleanReplayBuffer:
    """A bounded CPU-only replay buffer whose samples are never anonymous."""

    def __init__(self, capacity: int, manifest: RunManifest):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.manifest = manifest
        self._samples: deque[ReplaySample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ReplaySample]:
        return iter(self._samples)

    def add_completed_game(
        self,
        *,
        iteration: int,
        game_id: int,
        samples: Iterable[tuple[np.ndarray, np.ndarray, float]],
        completed: bool,
    ) -> int:
        """Add only completed games and attach immutable sample provenance.

        ``samples`` is an iterable of ``(state, policy, value)`` tuples where
        ``state`` may be uint8 or float32 (0/1 planes) and ``policy`` is a
        dense float32 vector.  Both are compacted on insertion.
        """

        if not completed:
            return 0
        if iteration < 1 or game_id < 0:
            raise ValueError("iteration must be >= 1 and game_id must be >= 0.")
        added = 0
        for position_index, (state, policy, value) in enumerate(samples):
            policy_indices, policy_values = _compact_policy(policy)
            sample = ReplaySample(
                state=_compact_state(state),
                policy_indices=policy_indices,
                policy_values=policy_values,
                value=float(value),
                provenance=SampleProvenance(
                    run_id=self.manifest.run_id,
                    iteration=int(iteration),
                    game_id=int(game_id),
                    position_index=position_index,
                ),
            )
            self._validate_sample(sample)
            self._samples.append(sample)
            added += 1
        return added

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[ReplaySample]:
        if batch_size <= 0 or batch_size > len(self._samples):
            raise ValueError(
                f"batch_size must be in [1, {len(self._samples)}], got {batch_size}."
            )
        chooser = rng if rng is not None else random
        indices = chooser.sample(range(len(self._samples)), batch_size)
        return [self._samples[i] for i in indices]

    def iteration_distribution(self) -> dict[int, int]:
        return dict(sorted(Counter(s.provenance.iteration for s in self._samples).items()))

    def validate(self, *, expected_max_iteration: int | None = None) -> dict:
        for index, sample in enumerate(self._samples):
            self._validate_sample(sample, index)
            if expected_max_iteration is not None and sample.provenance.iteration > expected_max_iteration:
                raise RuntimeError(
                    f"Replay sample {index} claims future iteration "
                    f"{sample.provenance.iteration}."
                )
        return {
            "run_id": self.manifest.run_id,
            "capacity": self.capacity,
            "size": len(self),
            "by_iteration": self.iteration_distribution(),
        }

    def save(self, root: str | Path, relative_path: str = "replay/replay_buffer.pt") -> Path:
        """Atomically save payload and provenance under this run root."""

        self.validate()
        destination = owned_artifact_path(root, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.manifest.run_id,
            "config_sha256": self.manifest.config_sha256,
            "kind": "clean_replay_buffer",
            "capacity": self.capacity,
            "sample_count": len(self),
            "iteration_distribution": self.iteration_distribution(),
        }
        payload = {
            "metadata": metadata,
            "samples": [
                {
                    "state": sample.state,
                    "policy_indices": sample.policy_indices,
                    "policy_values": sample.policy_values,
                    "value": sample.value,
                    "provenance": asdict(sample.provenance),
                }
                for sample in self._samples
            ],
        }
        temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        dist = self.iteration_distribution()
        print(f"[REPLAY] Saved: {len(self)} samples -> {destination}, iterations={dict(dist)}", flush=True)
        return destination

    @classmethod
    def load(
        cls,
        root: str | Path,
        manifest: RunManifest,
        relative_path: str = "replay/replay_buffer.pt",
    ) -> "CleanReplayBuffer":
        path = owned_artifact_path(root, relative_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "metadata" not in payload:
            raise RuntimeError(f"Replay is not a clean artifact: {path}")
        assert_artifact_matches_run(payload["metadata"], manifest, path)
        if payload["metadata"].get("kind") != "clean_replay_buffer":
            raise RuntimeError(f"Wrong clean artifact type for replay: {path}")
        schema_version = int(payload["metadata"].get("schema_version", 1))
        replay = cls(int(payload["metadata"]["capacity"]), manifest)
        for raw in payload.get("samples", []):
            if schema_version == 1:
                state = np.asarray(raw["state"], dtype=np.float32).astype(np.uint8)
                policy_indices, policy_values = _compact_policy(raw["policy"])
            else:
                state = np.asarray(raw["state"], dtype=np.uint8)
                policy_indices = np.asarray(raw["policy_indices"], dtype=np.int16)
                policy_values = np.asarray(raw["policy_values"], dtype=np.float32)
            replay._samples.append(
                ReplaySample(
                    state=state,
                    policy_indices=policy_indices,
                    policy_values=policy_values,
                    value=float(raw["value"]),
                    provenance=SampleProvenance(**raw["provenance"]),
                )
            )
        replay.validate()
        if len(replay) != int(payload["metadata"]["sample_count"]):
            raise RuntimeError(f"Replay sample-count mismatch: {path}")
        dist = replay.iteration_distribution()
        print(f"[REPLAY] Loaded: {len(replay)} samples from {path}, iterations={dict(dist)}", flush=True)
        return replay

    def _validate_sample(self, sample: ReplaySample, index: int | None = None) -> None:
        label = f"Replay sample {index}" if index is not None else "Replay sample"
        if sample.provenance.run_id != self.manifest.run_id:
            raise RuntimeError(f"{label} has foreign run provenance.")
        if sample.provenance.iteration < 1:
            raise RuntimeError(f"{label} has invalid iteration provenance.")
        if sample.state.shape != STATE_SHAPE or sample.state.dtype != np.uint8:
            raise RuntimeError(f"{label} state must be uint8{STATE_SHAPE}.")
        if sample.policy_indices.shape != sample.policy_values.shape:
            raise RuntimeError(f"{label} policy indices/values shape mismatch.")
        if sample.policy_indices.dtype != np.int16 or sample.policy_values.dtype != np.float32:
            raise RuntimeError(f"{label} policy must be int16 indices + float32 values.")
        if len(sample.policy_indices) > 0:
            if sample.policy_indices.min() < 0 or sample.policy_indices.max() >= POLICY_SHAPE[0]:
                raise RuntimeError(f"{label} policy indices out of range.")
        if not np.isfinite(sample.policy_values).all():
            raise RuntimeError(f"{label} contains non-finite policy values.")
        if np.any(sample.policy_values < 0) or not np.isclose(sample.policy_values.sum(), 1.0, atol=1e-4):
            raise RuntimeError(f"{label} policy is not normalized.")
        if not math.isfinite(sample.value) or sample.value not in (-1.0, 0.0, 1.0):
            raise RuntimeError(f"{label} value must be one of -1, 0, 1.")
