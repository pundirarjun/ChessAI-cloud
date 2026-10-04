"""GPUChess.insufficient_material must match python-chess is_insufficient_material.

Regression for two bugs: bishop square-color used file parity instead of
(rank + file) parity, and minor counting used piece *types* not pieces.
"""

import random

import chess
import pytest
import torch

from environment.gpu_chess import GPUChess, _signed_u64
from az.reference import gpu_from_board

_PIECE_CHARS = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


def _board(pieces):
    """pieces: list of (char, square) with uppercase = white."""
    board = chess.Board(None)
    for char, square in pieces:
        color = chess.WHITE if char.isupper() else chess.BLACK
        board.set_piece_at(chess.parse_square(square), chess.Piece(_PIECE_CHARS[char.upper()], color))
    return board


CURATED = [
    ("bare kings", [("K", "e1"), ("k", "e8")]),
    ("K vs K+N", [("K", "e1"), ("k", "e8"), ("n", "d5")]),
    ("K vs K+B", [("K", "e1"), ("k", "e8"), ("b", "d5")]),
    ("K+N vs K", [("K", "e1"), ("k", "e8"), ("N", "d4")]),
    ("K+B vs K", [("K", "e1"), ("k", "e8"), ("B", "d4")]),
    # Bug 1: both bishops on dark squares (b2 and g7) -> dead, same color.
    ("K+B(dark) vs K+B(dark)", [("K", "e1"), ("k", "e8"), ("B", "b2"), ("b", "g7")]),
    ("K+B(dark) vs K+B(light)", [("K", "e1"), ("k", "e8"), ("B", "b2"), ("b", "f7")]),
    ("K+B(light) vs K+B(light)", [("K", "e1"), ("k", "e8"), ("B", "f1"), ("b", "c8")]),
    # Bug 2: two bishops of one side are NOT dead material.
    ("K+2B vs K", [("K", "e1"), ("k", "e8"), ("B", "c1"), ("B", "f1")]),
    ("K+3B(one color) vs K", [("K", "e1"), ("k", "e8"), ("B", "f1"), ("B", "a4"), ("B", "c6")]),
    ("K+N vs K+N", [("K", "e1"), ("k", "e8"), ("N", "d4"), ("n", "d5")]),
    ("K+N vs K+B", [("K", "e1"), ("k", "e8"), ("N", "d4"), ("b", "d5")]),
    ("K+B vs K+N", [("K", "e1"), ("k", "e8"), ("B", "d4"), ("n", "d5")]),
    ("K+2N vs K", [("K", "e1"), ("k", "e8"), ("N", "d4"), ("N", "e4")]),
    ("K+P vs K", [("K", "e1"), ("k", "e8"), ("P", "e2")]),
    ("K+Q vs K", [("K", "e1"), ("k", "e8"), ("Q", "d4")]),
    ("K+R vs K", [("K", "e1"), ("k", "e8"), ("R", "d4")]),
    ("K+B+N vs K", [("K", "e1"), ("k", "e8"), ("B", "d4"), ("N", "e4")]),
    ("K+N+P vs K", [("K", "e1"), ("k", "e8"), ("N", "d4"), ("P", "e2")]),
    ("K vs K+B+N", [("K", "e1"), ("k", "e8"), ("b", "d5"), ("n", "e5")]),
]


@pytest.mark.parametrize("name,pieces", CURATED, ids=[c[0] for c in CURATED])
def test_curated_matches_python_chess(name, pieces):
    board = _board(pieces)
    expected = board.is_insufficient_material()
    state = gpu_from_board(board, "cpu")
    got = bool(state.insufficient_material()[0].item())
    assert got == expected, f"{name}: GPUChess={got} python-chess={expected}"


def _random_material_board(rng: random.Random) -> chess.Board:
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    used = {chess.E1, chess.E8}
    for color in (chess.WHITE, chess.BLACK):
        counts = [
            (chess.PAWN, rng.choice([0, 0, 0, 1, 2])),
            (chess.KNIGHT, rng.choice([0, 0, 1, 2, 3])),
            (chess.BISHOP, rng.choice([0, 0, 1, 2, 3])),
            (chess.ROOK, rng.choice([0, 0, 0, 1])),
            (chess.QUEEN, rng.choice([0, 0, 1])),
        ]
        for piece_type, n in counts:
            for _ in range(n):
                free = [s for s in range(64) if s not in used]
                square = rng.choice(free)
                used.add(square)
                board.set_piece_at(square, chess.Piece(piece_type, color))
    return board


def test_randomized_differential_matches_python_chess():
    rng = random.Random(20261004)
    boards = [_random_material_board(rng) for _ in range(400)]

    expected = torch.tensor([b.is_insufficient_material() for b in boards])

    # One batched state: insufficient_material reads only `pieces`.
    state = GPUChess("cpu", len(boards))
    state.pieces.zero_()
    for i, board in enumerate(boards):
        for square, piece in board.piece_map().items():
            plane = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}[piece.piece_type]
            if not piece.color:
                plane += 6
            state.pieces[i, plane] |= _signed_u64(1 << square)

    got = state.insufficient_material()
    mismatches = (got != expected).nonzero().flatten().tolist()
    detail = []
    for i in mismatches[:5]:
        detail.append(f"  case {i} fen={boards[i].fen()} expected={bool(expected[i])} got={bool(got[i])}")
    assert not mismatches, f"{len(mismatches)}/400 mismatches:\n" + "\n".join(detail)
