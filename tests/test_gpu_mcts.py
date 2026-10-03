import torch

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


def test_gpu_mcts_pipeline_cpu_validation():
    model = ChessNet(action_space_size=4544).eval()
    states = GPUChess("cpu", 2)
    search = GPUMCTS(model=model, device="cpu")
    search.search(states, num_simulations=2)
    policy = search.root_visit_policy()
    actions = search.select_actions(temperature=0.0)
    next_states = search.advance(actions)
    assert policy.shape == (2, 4544)
    assert torch.allclose(policy.sum(1), torch.ones(2), atol=1e-5)
    assert actions.shape == (2,)
    assert next_states.legal_move_mask().sum(1).min().item() > 0
    assert search.last_duplicate_leaf_count >= 0
    assert search.total_duplicate_leaf_count >= search.last_duplicate_leaf_count


def test_mcts_marks_a_third_repetition_on_the_selected_tree_path():
    model = ChessNet(action_space_size=4544).eval()
    states = GPUChess("cpu", 1)
    search = GPUMCTS(model=model, device="cpu")
    search._allocate(num_games=1, num_simulations=2)
    root = search._root_ids
    child = root + 1
    # A compact tree state is enough for this focused history test: the child
    # represents a branch returning to the root position.
    search._write_states(root, states)
    search._write_states(child, states)
    root_hash = states.state_hash().reshape(1, 1)
    paths = torch.stack((root, child), dim=1).reshape(1, 1, 2)

    repeated = search._third_repetition_paths(paths, root_hash.repeat(1, 2))

    assert bool(repeated[0, 0])
