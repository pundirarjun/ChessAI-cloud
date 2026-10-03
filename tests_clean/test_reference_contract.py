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
