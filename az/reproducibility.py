"""Central seed and CUDA reproducibility controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import random

import numpy as np
import torch

from az.config import RuntimeConfig


@dataclass(frozen=True)
class SeedReport:
    seed: int
    rank: int
    worker_seed: int
    cuda_available: bool
    deterministic: bool

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(runtime: RuntimeConfig, *, rank: int = 0) -> SeedReport:
    """Seed Python, NumPy, PyTorch, CUDA, and data-worker derivation."""

    worker_seed = int(runtime.seed) + int(rank)
    os.environ.setdefault("PYTHONHASHSEED", str(worker_seed))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

    torch.backends.cuda.matmul.allow_tf32 = bool(runtime.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(runtime.allow_tf32)
    if runtime.deterministic:
        # Users must set this before CUDA kernels are created. It deliberately
        # remains documented as best-effort because not every GPU operation has
        # a deterministic implementation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
    return SeedReport(
        seed=int(runtime.seed),
        rank=int(rank),
        worker_seed=worker_seed,
        cuda_available=torch.cuda.is_available(),
        deterministic=bool(runtime.deterministic),
    )


def worker_seed(base_seed: int, worker_id: int, rank: int = 0) -> int:
    return int(base_seed) + int(rank) * 10_000 + int(worker_id)


def seed_data_worker(worker_id: int) -> None:
    # DataLoader invokes this in the worker process. torch.initial_seed already
    # includes the rank-specific generator state.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
