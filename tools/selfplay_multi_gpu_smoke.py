"""Smoke-test GPU-local self-play worker sharding on two CUDA devices."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from model.chess_net import ChessNet
from training.self_play import play_games_multi_gpu


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("This smoke test requires two visible CUDA GPUs.")
    with tempfile.TemporaryDirectory(prefix="clean_az_selfplay_smoke_") as directory:
        checkpoint = Path(directory) / "fresh_best.pt"
        model = ChessNet().to("cuda:0").eval()
        torch.save({"model_state_dict": model.state_dict()}, checkpoint)
        results = play_games_multi_gpu(
            model=model,
            checkpoint_path=checkpoint,
            num_games=2,
            num_simulations=2,
            max_moves=2,
            batch_size=2,
            model_kwargs={"action_space_size": 4544},
        )
    if len(results) != 2:
        raise RuntimeError(f"Expected 2 results, received {len(results)}.")
    print("PASS: one self-play worker completed on each visible GPU.")


if __name__ == "__main__":
    main()
