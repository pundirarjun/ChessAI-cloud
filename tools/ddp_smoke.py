"""Two-GPU DDP smoke test for the clean 18-plane policy/value model.

Run on Kaggle:
    torchrun --standalone --nproc_per_node=2 tools/ddp_smoke.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.chess_net import ChessNet


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("This smoke test requires two visible CUDA GPUs.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(100 + rank)
        model = ChessNet().to(device, memory_format=torch.channels_last)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        states = torch.randn((2, 18, 8, 8), device=device).contiguous(
            memory_format=torch.channels_last
        )
        logits, values = model(states)
        loss = logits.square().mean() + values.square().mean()
        loss.backward()
        optimizer.step()
        checksum = torch.stack(
            [parameter.detach().float().sum() for parameter in model.module.parameters()]
        ).sum()
        gathered = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, checksum)
        if not torch.allclose(gathered[0], gathered[1], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"DDP replicas diverged: {[value.item() for value in gathered]}")
        if rank == 0:
            print("PASS: GPUs 0 and 1 initialized; gradients synchronized; optimizer stepped.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
