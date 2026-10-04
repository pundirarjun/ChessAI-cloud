"""Rook/queen slides that share action ids with castling must stay legal.

Regression: pseudo_legal_mask used to clear actions (4,6)/(4,2)/(60,62)/(60,58)
for every state, dropping genuine slides (e.g. Re1-g1 with the king elsewhere),
and _apply_flat relocated the h1/a1 rook for any piece sliding on those ids.
"""

import chess
import pytest
import torch

from az.reference import board_from_gpu, gpu_from_board
from environment.action_encoder import ActionEncoder

_ENCODER = ActionEncoder()

CASES = [
    ("white rook e1 slides", "4k3/8/8/8/8/8/4K3/4R3 w - - 0 1", {"e1g1", "e1c1"}),
    ("black rook e8 slides", "4r2k/8/8/8/8/8/8/K7 b - - 0 1", {"e8g8", "e8c8"}),
    ("white queen e1 slides", "4k3/8/8/8/8/8/4K3/4Q3 w - - 0 1", {"e1g1", "e1c1"}),
    ("black queen e8 slides", "4q2k/8/8/8/8/8/8/K7 b - - 0 1", {"e8g8", "e8c8"}),
    # King still at home: castling-geometry slides are impossible there, but
    # the ordinary castling candidates must keep working.
    ("white king home castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", {"e1g1", "e1c1"}),
    ("black king home castling", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", {"e8g8", "e8c8"}),
]


@pytest.mark.parametrize("name,fen,moves", CASES, ids=[c[0] for c in CASES])
def test_castle_geometry_moves_match_python_chess(name, fen, moves):
    board = chess.Board(fen)
    gpu = gpu_from_board(board, "cpu")
    actual = set(torch.nonzero(gpu.legal_move_mask()[0]).flatten().tolist())
    expected = {_ENCODER.encode(m) for m in board.legal_moves}
    assert actual == expected, (
        f"{name}: missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
    )
    uci = {m.uci() for m in board.legal_moves}
    assert moves <= uci


@pytest.mark.parametrize("name,fen,moves", CASES, ids=[c[0] for c in CASES])
def test_castle_geometry_apply_does_not_corrupt(name, fen, moves):
    """Every move must transition to the exact python-chess position."""
    board = chess.Board(fen)
    for uci in sorted(moves):
        move = board.parse_uci(uci)
        if not board.is_legal(move):
            continue
        gpu = gpu_from_board(board, "cpu")
        action = _ENCODER.encode(move)
        assert bool(gpu.legal_move_mask()[0, action].item())
        after_gpu = board_from_gpu(gpu.push_actions(torch.tensor([action])))
        expected = board.copy(stack=False)
        expected.push(move)
        assert after_gpu.fen(en_passant="fen") == expected.fen(en_passant="fen"), uci
