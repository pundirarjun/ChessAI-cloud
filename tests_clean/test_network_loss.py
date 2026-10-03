import torch

from model.chess_net import ChessNet
from training.loss import alpha_zero_loss


def test_network_output_contract_and_loss_backward():
    model = ChessNet()
    states = torch.zeros((2, 18, 8, 8))
    logits, values = model(states)
    policy = torch.zeros((2, 4544))
    policy[:, 0] = 1
    target_values = torch.tensor([1.0, -1.0])
    total, policy_loss, value_loss = alpha_zero_loss(logits, values, policy, target_values)
    total.backward()
    assert logits.shape == (2, 4544)
    assert values.shape == (2, 1)
    assert torch.all(values <= 1) and torch.all(values >= -1)
    assert policy_loss > 0 and value_loss >= 0
