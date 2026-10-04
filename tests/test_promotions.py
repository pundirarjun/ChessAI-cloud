"""Promotion action ids must agree between ActionEncoder and GPUChess.

Regression: the encoder used to emit black promotions first (ids 4032..4287)
while GPUChess emitted white promotions first, disagreeing on all 512 ids.
"""

import chess
import pytest
import torch

from az.reference import gpu_from_board
from environment.action_encoder import ActionEncoder

_ENCODER = ActionEncoder()

CASES = [
    ("white push and capture promos",
     "3n2k1/4P3/8/8/8/8/8/4K3 w - - 0 1"),
    ("white underpromotions blocked queen",
     "rn5k/1P6/8/8/8/8/8/K7 w - - 0 1"),
    ("black push and capture promos",
     "4k3/8/8/8/8/8/4p3/3N2K1 b - - 0 1"),
    ("black underpromotions",
     "k7/8/8/8/8/8/1p6/RN5K b - - 0 1"),
    ("white push promo with enemy pawn near promotion",
     "6k1/4P3/8/8/8/8/4p3/4K3 w - - 0 1"),
]


@pytest.mark.parametrize("name,fen", CASES, ids=[c[0] for c in CASES])
def test_promotion_actions_match_python_chess(name, fen):
    board = chess.Board(fen)
    assert any(chess.square_rank(m.to_square) in (0, 7)
               and m.promotion for m in board.legal_moves), "case must contain promotions"
    gpu = gpu_from_board(board, "cpu")
    actual = set(torch.nonzero(gpu.legal_move_mask()[0]).flatten().tolist())
    expected = {_ENCODER.encode(m) for m in board.legal_moves}
    assert actual == expected, (
        f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
    )
