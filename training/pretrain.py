"""Supervised pretraining of ChessNet on PGN games.

Trains the network on human games (policy = one-hot of played move,
value = game outcome) before AlphaZero self-play. Overwrites
best_model.pt in the run dir with pretrained weights.

Usage:
    python clean_az.py pretrain --config configs/clean_az.json \
        --pgn-glob /kaggle/input/datasets/pearsejim01/pgn-mentor-dataset/**/*.pgn
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.checkpoints import save_checkpoint
from az.config import RunConfig
from az.model import build_fresh_model
from az.provenance import load_manifest
from az.reproducibility import seed_everything
from environment.action_encoder import ActionEncoder
from environment.state_encoder import StateEncoder
from training.loss import alpha_zero_loss


class PGNPretrainDataset(Dataset):
    def __init__(self, positions: list[np.ndarray], policy_targets: list[int], value_targets: list[float]):
        self.positions = positions
        self.policy_targets = policy_targets
        self.value_targets = value_targets

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.positions[idx])
        policy_target = self.policy_targets[idx]
        value_target = self.value_targets[idx]
        return state, policy_target, value_target


def parse_pgn_games(
    pgn_files: list[str],
    state_encoder: StateEncoder,
    action_encoder: ActionEncoder,
    max_games: int,
    max_positions: int,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    positions: list[np.ndarray] = []
    policy_targets: list[int] = []
    value_targets: list[float] = []
    games_parsed = 0

    for pgn_file in pgn_files:
        if games_parsed >= max_games or len(positions) >= max_positions:
            break
        with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                if games_parsed >= max_games or len(positions) >= max_positions:
                    break
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games_parsed += 1
                if game.headers.get("Result", "*") == "*":
                    continue
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    outcome = 1.0
                elif result == "0-1":
                    outcome = -1.0
                elif result in ("1/2-1/2", "1/2"):
                    outcome = 0.0
                else:
                    continue
                board = game.board()
                for move in game.mainline_moves():
                    if len(positions) >= max_positions:
                        break
                    try:
                        action_id = action_encoder.encode(move)
                    except KeyError:
                        board.push(move)
                        continue
                    state = state_encoder.encode(board).astype(np.uint8)
                    positions.append(state)
                    policy_targets.append(action_id)
                    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
                    value_targets.append(outcome * side_to_move)
                    board.push(move)

    print(f"[PRETRAIN] Parsed {games_parsed} games, {len(positions)} positions", flush=True)
    return positions, policy_targets, value_targets


def run_pretrain(config: RunConfig, args: argparse.Namespace) -> None:
    seed_everything(config.runtime)
    manifest = load_manifest(config.root)

    if args.pgn_glob:
        pgn_files = globlib.glob(args.pgn_glob, recursive=True)
    else:
        pgn_dir = Path(args.pgn_dir)
        pgn_files = sorted(pgn_dir.glob("**/*.pgn"))
    if not pgn_files:
        raise SystemExit(f"No PGN files found: {args.pgn_glob or args.pgn_dir}")
    print(f"[PRETRAIN] Found {len(pgn_files)} PGN files", flush=True)

    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    positions, policy_targets, value_targets = parse_pgn_games(
        pgn_files, state_encoder, action_encoder, args.max_games, args.max_positions
    )
    if not positions:
        raise SystemExit("No positions extracted from PGNs")

    dataset = PGNPretrainDataset(positions, policy_targets, value_targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fresh_model(config.architecture, config.runtime, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    total_steps = len(loader) * args.epochs
    print(f"[PRETRAIN] Starting: {len(positions)} positions, {args.epochs} epochs, {total_steps} steps on {device}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        n_batches = 0
        for states, policy_idx, values in loader:
            states = states.to(device, non_blocking=True)
            policy_idx = policy_idx.to(device)
            values = values.to(device)
            target_policy = torch.zeros(len(policy_idx), config.architecture.action_space_size, device=device)
            target_policy.scatter_(1, policy_idx.unsqueeze(1), 1.0)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=config.training.mixed_precision and device.type == "cuda"):
                policy_logits, value_pred = model(states)
                total, policy_loss, value_loss = alpha_zero_loss(
                    policy_logits, value_pred.squeeze(-1), target_policy, values
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += total.item()
            n_batches += 1
            if n_batches % 100 == 0:
                avg = total_loss / n_batches
                print(f"[PRETRAIN] Epoch {epoch+1}, step {n_batches}/{len(loader)}: loss={avg:.4f} p={total_policy_loss/n_batches:.4f} v={total_value_loss/n_batches:.4f}", flush=True)
        avg_loss = total_loss / n_batches
        print(f"[PRETRAIN] Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}", flush=True)

    path = save_checkpoint(
        root=config.root,
        manifest=manifest,
        role="best",
        model=model,
        optimizer=optimizer,
        iteration=0,
        training_step=0,
        extra={"pretrained": True, "epochs": args.epochs, "positions": len(positions)},
    )
    print(f"[PRETRAIN] Saved pretrained weights to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised pretraining of ChessNet on PGN data.")
    parser.add_argument("--config", required=True, help="Path to clean run JSON config.")
    parser.add_argument("--pgn-dir", default=None, help="Directory containing PGN files.")
    parser.add_argument("--pgn-glob", default=None, help="Glob pattern for PGN files (recursive).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--max-positions", type=int, default=2000000)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.pgn_dir and not args.pgn_glob:
        raise SystemExit("Provide --pgn-dir or --pgn-glob")

    config = RunConfig.load_json(args.config)
    run_pretrain(config, args)


if __name__ == "__main__":
    main()
