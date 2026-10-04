import chess
import torch

from environment.action_encoder import ActionEncoder
from mcts.mcts import MCTS
from mcts.node import Node
from mcts.policy import policy_from_logits
from training.self_play import SelfPlayGame


def _child(parent, move):
    board = parent.board.copy()
    board.push(move)
    result = Node(board, parent=parent, move=move, prior=0.5)
    parent.children[move] = result
    return result


def test_backup_negates_values_and_clears_virtual_visits():
    root = Node(chess.Board())
    child = _child(root, chess.Move.from_uci("e2e4"))
    root.virtual_visit_count = child.virtual_visit_count = 1
    child.backup(0.75)
    assert child.visit_count == root.visit_count == 1
    assert child.value == 0.75
    assert root.value == -0.75
    assert child.virtual_visit_count == root.virtual_visit_count == 0


def test_puct_uses_parent_perspective_and_virtual_visits():
    root = Node(chess.Board())
    first = _child(root, chess.Move.from_uci("e2e4"))
    second = _child(root, chess.Move.from_uci("d2d4"))
    root.visit_count = 16
    first.visit_count, first.value_sum, first.prior = 4, -2.0, 0.3
    second.visit_count, second.value_sum, second.prior = 4, 2.0, 0.7
    # First child is good for its parent because child value is negative.
    assert root.select_child(c_puct=0.1)[1] is first
    # Virtual loss adds temporary visits AND a temporary value sum; the PUCT
    # score must use both, matching Node.apply_virtual_loss semantics.
    for _ in range(100):
        first.apply_virtual_loss(1.0)
    assert first.virtual_visit_count == 100
    assert first.virtual_value_sum == 100.0
    assert root.select_child(c_puct=0.1)[1] is second


def test_policy_mask_exposes_only_legal_actions():
    board = chess.Board()
    encoder = ActionEncoder()
    logits = torch.full((4544,), 100.0)
    policy = policy_from_logits(board, logits, encoder)
    assert set(policy) == set(board.legal_moves)
    assert abs(sum(policy.values()) - 1.0) < 1e-6


def test_terminal_checkmate_value_is_side_to_move_perspective():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    node = Node(board)
    assert node.is_terminal()
    value = MCTS(model=None, action_encoder=ActionEncoder()).get_terminal_value(node)
    assert value == -1.0


def test_self_play_values_are_from_the_stored_player_perspective():
    game = SelfPlayGame()
    state = torch.zeros((18, 8, 8)).numpy()
    policy = torch.zeros((4544,)).numpy()
    policy[0] = 1
    game.add_position(state, policy, player=1)
    game.add_position(state, policy, player=-1)
    assert [sample[2] for sample in game.get_training_data(1)] == [1.0, -1.0]
    assert [sample[2] for sample in game.get_training_data(-1)] == [-1.0, 1.0]
