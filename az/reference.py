"""Conversions used only by CPU correctness/differential tests."""

from __future__ import annotations

import chess
import torch

from environment.gpu_chess import BK, BQ, WK, WQ, GPUChess, _signed_u64


_TYPE_TO_INTERNAL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_INTERNAL_TO_TYPE = {value: key for key, value in _TYPE_TO_INTERNAL.items()}


def gpu_from_board(board: chess.Board, device: str | torch.device = "cpu") -> GPUChess:
    """Make a one-position GPUChess test state from a python-chess board."""

    state = GPUChess(device, 1)
    state.pieces.zero_()
    for square, piece in board.piece_map().items():
        plane = _TYPE_TO_INTERNAL[piece.piece_type] + (0 if piece.color else 6)
        state.pieces[0, plane] |= _signed_u64(1 << square)
    state.turn[0] = not board.turn
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= WK
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= BK
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= BQ
    state.castling[0] = castling
    state.ep_square[0] = -1 if board.ep_square is None else board.ep_square
    state.halfmove_clock[0] = board.halfmove_clock
    state.fullmove_number[0] = board.fullmove_number
    return state


def board_from_gpu(state: GPUChess, index: int = 0) -> chess.Board:
    """Convert a one-state GPUChess test position to python-chess."""

    board = chess.Board(None)
    for plane in range(12):
        bits = int(state.pieces[index, plane].item()) & ((1 << 64) - 1)
        color = chess.WHITE if plane < 6 else chess.BLACK
        piece_type = _INTERNAL_TO_TYPE[plane % 6]
        for square in chess.scan_reversed(bits):
            board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = not bool(state.turn[index].item())
    rights = 0
    castling = int(state.castling[index].item())
    if castling & WK:
        rights |= chess.BB_H1
    if castling & WQ:
        rights |= chess.BB_A1
    if castling & BK:
        rights |= chess.BB_H8
    if castling & BQ:
        rights |= chess.BB_A8
    board.castling_rights = rights
    ep = int(state.ep_square[index].item())
    board.ep_square = None if ep < 0 else ep
    board.halfmove_clock = int(state.halfmove_clock[index].item())
    board.fullmove_number = int(state.fullmove_number[index].item())
    board.clear_stack()
    return board
