"""Differential-test the tensorized chess engine against python-chess.

Example:
    python tools/chess_differential.py --positions 1000 --max-plies 200 --seed 42
"""

from __future__ import annotations

import argparse
import random

import chess
import torch

from az.reference import board_from_gpu, gpu_from_board
from environment.action_encoder import ActionEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=1_000)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    board = chess.Board()
    encoder = ActionEncoder()
    for position_index in range(args.positions):
        if board.is_game_over(claim_draw=True) or board.ply() >= args.max_plies:
            board.reset()
        gpu = gpu_from_board(board, "cpu")
        actual = set(torch.nonzero(gpu.legal_move_mask()[0]).flatten().tolist())
        expected = {encoder.encode(move) for move in board.legal_moves}
        if actual != expected:
            missing = expected - actual
            unexpected = actual - expected
            raise AssertionError(
                f"Legal-action mismatch at position {position_index}\n"
                f"FEN: {board.fen()}\nmissing={sorted(missing)}\nunexpected={sorted(unexpected)}"
            )
        restored = board_from_gpu(gpu)
        if restored.fen(en_passant="fen") != board.fen(en_passant="fen"):
            raise AssertionError(
                f"State-transition metadata mismatch at position {position_index}\n"
                f"expected={board.fen(en_passant='fen')}\nactual={restored.fen(en_passant='fen')}"
            )
        move = rng.choice(list(board.legal_moves))
        action = encoder.encode(move)
        after_gpu = gpu.push_actions(torch.tensor([action]))
        after_board = board.copy(stack=False)
        after_board.push(move)
        after_restored = board_from_gpu(after_gpu)
        if after_restored.fen(en_passant="fen") != after_board.fen(en_passant="fen"):
            raise AssertionError(
                f"Move-application mismatch at position {position_index}, move {move.uci()}\n"
                f"expected={after_board.fen(en_passant='fen')}\n"
                f"actual={after_restored.fen(en_passant='fen')}"
            )
        board.push(move)
    print(
        f"PASS: {args.positions} random legal positions/transitions matched python-chess "
        f"(seed={args.seed})."
    )


if __name__ == "__main__":
    main()
