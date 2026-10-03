"""DDP-aware AlphaZero candidate training over CPU-resident clean replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from az.config import RuntimeConfig, TrainingConfig
from az.replay import CleanReplayBuffer, ReplaySample
from az.reproducibility import seed_data_worker
from training.loss import alpha_zero_loss


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class TrainingMetrics:
    steps: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float
    learning_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


class _ReplayDataset(Dataset):
    def __init__(self, replay: CleanReplayBuffer):
        self.samples = list(replay)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample: ReplaySample = self.samples[index]
        return (
            torch.from_numpy(sample.state),
            torch.from_numpy(sample.policy),
            torch.tensor(sample.value, dtype=torch.float32),
        )


def distributed_context(runtime: RuntimeConfig) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if runtime.num_gpus is not None and visible < min(runtime.num_gpus, world_size):
            raise RuntimeError(
                f"Configured for {runtime.num_gpus} GPUs but only {visible} are visible."
            )
        if local_rank >= visible:
            raise RuntimeError(f"LOCAL_RANK {local_rank} has no visible CUDA device.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if world_size != 1:
            raise RuntimeError("Distributed training requires CUDA in this project.")
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def close_distributed(context: DistributedContext) -> None:
    if context.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def train_candidate(
    *,
    model: torch.nn.Module,
    replay: CleanReplayBuffer,
    config: TrainingConfig,
    runtime: RuntimeConfig,
    context: DistributedContext,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, TrainingMetrics, torch.amp.GradScaler | None]:
    """Train synchronized model replicas; replay data stays on CPU."""

    replay.validate()
    if len(replay) < config.batch_size * context.world_size:
        raise RuntimeError(
            f"Need at least batch_size * world_size samples "
            f"({config.batch_size * context.world_size}), got {len(replay)}."
        )
    dataset = _ReplayDataset(replay)
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=runtime.seed,
        drop_last=True,
    ) if context.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        pin_memory=context.device.type == "cuda",
        num_workers=0,
        worker_init_fn=seed_data_worker,
    )
    if len(loader) == 0:
        raise RuntimeError("Replay does not produce a complete training batch.")
    model = model.to(context.device)
    if context.device.type == "cuda" and config.channels_last:
        model.to(memory_format=torch.channels_last)
    if optimizer is None:
        optimizer = build_optimizer(model, config)
    wrapped = (
        DDP(model, device_ids=[context.local_rank], output_device=context.local_rank)
        if context.world_size > 1
        else model
    )
    use_amp = context.device.type == "cuda" and config.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator: Iterator = iter(loader)
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)
    epoch = 0
    for step in range(config.steps):
        try:
            states, policies, values = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            states, policies, values = next(iterator)
        states = states.to(context.device, non_blocking=use_amp)
        policies = policies.to(context.device, non_blocking=use_amp)
        values = values.to(context.device, non_blocking=use_amp)
        if context.device.type == "cuda" and config.channels_last:
            states = states.contiguous(memory_format=torch.channels_last)
        _validate_training_batch(policies, values)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, predicted_values = wrapped(states)
            total, policy, value = alpha_zero_loss(logits, predicted_values, policies, values)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            wrapped.parameters(), config.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        totals += torch.tensor(
            [total.detach(), policy.detach(), value.detach(), gradient_norm.detach()],
            dtype=torch.float64,
            device=context.device,
        )
    if context.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        totals /= context.world_size
    totals /= config.steps
    metrics = TrainingMetrics(
        steps=config.steps,
        total_loss=float(totals[0].item()),
        policy_loss=float(totals[1].item()),
        value_loss=float(totals[2].item()),
        gradient_norm=float(totals[3].item()),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )
    return wrapped.module if isinstance(wrapped, DDP) else wrapped, optimizer, metrics, scaler


def _validate_training_batch(policies: torch.Tensor, values: torch.Tensor) -> None:
    if policies.ndim != 2 or policies.shape[1] != 4544:
        raise RuntimeError("Policy targets must be shaped [batch, 4544].")
    if not torch.isfinite(policies).all() or not torch.isfinite(values).all():
        raise RuntimeError("Training targets contain non-finite values.")
    if torch.any(policies < 0) or not torch.allclose(
        policies.sum(dim=1), torch.ones_like(policies[:, 0]), atol=1e-4, rtol=0
    ):
        raise RuntimeError("Policy targets must be normalized non-negative distributions.")
    if torch.any((values < -1) | (values > 1)):
        raise RuntimeError("Value targets must be in [-1, 1].")
