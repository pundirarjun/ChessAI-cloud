import random

import chess
import numpy as np
import torch

from az.reference import board_from_gpu, gpu_from_board
from environment.action_encoder import ActionEncoder
from environment.state_encoder import StateEncoder
from environment.gpu_chess import _action_tables


def test_action_encoder_is_bijective_and_has_baseline_size():
    encoder = ActionEncoder()
    assert encoder.size() == 4544
    for index in range(encoder.size()):
        assert encoder.encode(encoder.decode(index)) == index


def test_action_encoder_matches_gpu_action_tables():
    """Encoder and GPUChess must agree on every one of the 4544 action ids.

    Regression: promotion blocks were once ordered black-first in the encoder
    and white-first in GPUChess, disagreeing on all 512 promotion ids.
    """
    encoder = ActionEncoder()
    tables = _action_tables()
    assert encoder.size() == 4544
    for index in range(4544):
        move = encoder.decode(index)
        expected_from = int(tables["from_sq"][index])
        expected_to = int(tables["to_sq"][index])
        expected_promo = int(tables["promo"][index])
        assert move.from_square == expected_from, index
        assert move.to_square == expected_to, index
        # python-chess promotion constants are KNIGHT=2..QUEEN=5, which are
        # exactly GPUChess's promotion codes.
        promo_code = 0 if move.promotion is None else move.promotion
        assert promo_code == expected_promo, (
            f"action {index}: encoder move={move} gpu=({expected_from},"
            f"{expected_to},{expected_promo})"
        )


def test_gpu_state_and_legal_actions_match_python_chess_randomly():
    encoder = ActionEncoder()
    board = chess.Board()
    rng = random.Random(19)
    for _ in range(24):
        gpu = gpu_from_board(board)
        restored = board_from_gpu(gpu)
        assert restored.board_fen() == board.board_fen()
        assert restored.turn == board.turn
        assert restored.castling_rights == board.castling_rights
        assert restored.ep_square == board.ep_square
        gpu_actions = set(torch.nonzero(gpu.legal_move_mask()[0]).flatten().tolist())
        reference_actions = {encoder.encode(move) for move in board.legal_moves}
        assert gpu_actions == reference_actions
        state = StateEncoder.encode(board)
        assert np.array_equal(state, gpu.to_model_input()[0].numpy())
        move = rng.choice(list(board.legal_moves))
        board.push(move)
