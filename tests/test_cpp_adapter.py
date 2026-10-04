"""Contract tests: CppMctsAdapter must match GPUMCTS under a deterministic model.

With noise disabled and a deterministic CPU stub model, tree growth is fully
deterministic for both engines, so root policies and selected actions must be
identical. This is the strongest cross-validation of the C++ MCTS semantics
(PUCT, virtual loss, backups, prior softmax, batch rounds, dedup fan-out).
"""

import chess
import numpy as np
import pytest
import torch

from az.reference import gpu_from_board
from environment.gpu_chess import GPUChess
from environment.action_encoder import ActionEncoder

m = pytest.importorskip("az_cpp_mcts", reason="az_cpp_mcts extension not built")

from mcts.cpp_mcts import CppMctsAdapter  # noqa: E402
from mcts.gpu_mcts import GPUMCTS  # noqa: E402


ENCODER = ActionEncoder()

COMPARE_FENS = [
    "startpos: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "sicilian: rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "castling: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame: 8/2k5/8/8/8/8/2K5/4R3 w - - 0 1",
    "in_check: 7k/6Q1/8/8/8/8/8/K7 b - - 0 1",
]

SOLVED_FENS = [
    c.split(": ", 1)[1] for c in COMPARE_FENS
]


class StubModel(torch.nn.Module):
    """Deterministic frozen model: distinct logits, no randomness."""

    def __init__(self, seed: int = 1234):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("w_log", 0.5 * torch.randn(1152, 4544, generator=gen))
        self.register_buffer("w_val", torch.randn(1152, generator=gen))

    def forward(self, x: torch.Tensor):
        flat = x.reshape(x.shape[0], -1).float()
        logits = flat @ self.w_log
        values = (flat * self.w_val).sum(dim=1, keepdim=True) * 0.1
        return logits, values


def _state_batch(fens: list[str]) -> GPUChess:
    boards = [chess.Board("startpos" if f == "startpos" else f) for f in fens]
    states = GPUChess("cpu", len(boards))
    for i, board in enumerate(boards):
        one = gpu_from_board(board, "cpu")
        states.pieces[i] = one.pieces[0]
        states.turn[i] = one.turn[0]
        states.castling[i] = one.castling[0]
        states.ep_square[i] = one.ep_square[0]
        states.halfmove_clock[i] = one.halfmove_clock[0]
        states.fullmove_number[i] = one.fullmove_number[0]
    return states


def _run_both(states: GPUChess, num_simulations: int, batch_size: int):
    model = StubModel()

    gpu = GPUMCTS(model=model, device=torch.device("cpu"))
    gpu.search(
        states,
        num_simulations=num_simulations,
        dirichlet_alpha=None,
        batch_size=batch_size,
    )

    cpp = CppMctsAdapter(model=model, device=torch.device("cpu"), seed=99)
    cpp.search(
        states,
        num_simulations=num_simulations,
        dirichlet_alpha=None,
        batch_size=batch_size,
    )
    return gpu, cpp


@pytest.mark.parametrize("num_simulations,batch_size", [(64, 8), (256, 16)])
def test_root_policy_matches_gpumcts_exactly(num_simulations, batch_size):
    states = _state_batch(SOLVED_FENS)
    gpu, cpp = _run_both(states, num_simulations, batch_size)
    a = gpu.root_visit_policy()
    b = cpp.root_visit_policy()
    assert a.shape == b.shape == (len(SOLVED_FENS), 4544)
    assert torch.equal(a, b), (
        f"root policy divergence: max abs diff "
        f"{(a - b).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_select_actions_matches_gpumcts(temperature):
    states = _state_batch(SOLVED_FENS)
    gpu, cpp = _run_both(states, 128, 16)
    a = gpu.select_actions(temperature=temperature)
    b = cpp.select_actions(temperature=temperature)
    # temperature=0 is deterministic argmax; temperature=1 samples with
    # different RNGs, so only legality is asserted there.
    if temperature == 0.0:
        assert torch.equal(a, b), f"{a.tolist()} != {b.tolist()}"
    if temperature > 0:
        for g, fen in enumerate(SOLVED_FENS):
            board = chess.Board(fen)
            allowed = {ENCODER.encode(mv) for mv in board.legal_moves}
            assert int(b[g]) in allowed, (fen, int(b[g]))


def test_adapter_policies_have_legal_support_and_normalize():
    states = _state_batch(SOLVED_FENS)
    _, cpp = _run_both(states, 200, 16)
    policy = cpp.root_visit_policy().numpy()
    for g, fen in enumerate(SOLVED_FENS):
        board = chess.Board(fen)
        allowed = np.zeros(4544, dtype=bool)
        for mv in board.legal_moves:
            allowed[ENCODER.encode(mv)] = True
        assert np.all(policy[g, ~allowed] == 0.0), fen
        assert policy[g].sum() == pytest.approx(1.0, abs=1e-6), fen
        assert np.all(policy[g] >= 0.0), fen


def test_adapter_advance_matches_gpuchess_push_actions():
    states = _state_batch(SOLVED_FENS)
    _, cpp = _run_both(states, 64, 8)
    actions = cpp.select_actions(temperature=0.0)
    got = cpp.advance(actions)
    expected = states.push_actions(actions.cpu())
    assert got.pieces.shape == expected.pieces.shape
    assert torch.equal(got.pieces, expected.pieces)
    assert torch.equal(got.turn, expected.turn)
    assert torch.equal(got.castling, expected.castling)
    assert torch.equal(got.ep_square, expected.ep_square)
    assert torch.equal(got.halfmove_clock, expected.halfmove_clock)
    assert torch.equal(got.fullmove_number, expected.fullmove_number)


def test_adapter_is_deterministic_for_same_seed():
    states = _state_batch(SOLVED_FENS)
    model = StubModel()

    def run() -> np.ndarray:
        adapter = CppMctsAdapter(
            model=model, device=torch.device("cpu"), seed=7
        )
        adapter.search(
            states,
            num_simulations=128,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=16,
        )
        return adapter.root_visit_policy().numpy()

    assert np.array_equal(run(), run())


def test_adapter_dirichlet_noise_stays_normalized():
    states = _state_batch(SOLVED_FENS)
    model = StubModel()
    adapter = CppMctsAdapter(model=model, device=torch.device("cpu"), seed=3)
    adapter.search(
        states,
        num_simulations=100,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        batch_size=16,
    )
    policy = adapter.root_visit_policy().numpy()
    for g, fen in enumerate(SOLVED_FENS):
        board = chess.Board(fen)
        allowed = np.zeros(4544, dtype=bool)
        for mv in board.legal_moves:
            allowed[ENCODER.encode(mv)] = True
        assert np.all(policy[g, ~allowed] == 0.0), fen
        assert policy[g].sum() == pytest.approx(1.0, abs=1e-6), fen


def test_adapter_repetition_history_changes_terminal_values():
    """A depth-1 leaf that repeats a twice-seen position must back up as a draw.

    The first selected leaves of a fresh tree are always depth-1 children, so
    the history contains every root-child hash twice: whichever child is
    selected first, `history (2) + branch (1) >= 3` fires and its value becomes
    the draw value 0 instead of the stub's nonzero network value.
    """
    root = chess.Board()
    states = GPUChess("cpu", 1)
    one = gpu_from_board(root, "cpu")
    states.pieces[0] = one.pieces[0]
    states.turn[0] = one.turn[0]
    states.castling[0] = one.castling[0]
    states.ep_square[0] = one.ep_square[0]
    states.halfmove_clock[0] = one.halfmove_clock[0]
    states.fullmove_number[0] = one.fullmove_number[0]

    root_p = m.Position.from_fen(root.fen(en_passant="fen"))
    child_hashes = [
        root_p.apply(ENCODER.encode(mv)).state_hash() for mv in root.legal_moves
    ]

    model = StubModel()

    def run(history: torch.Tensor) -> np.ndarray:
        adapter = CppMctsAdapter(model=model, device=torch.device("cpu"), seed=11)
        adapter.search(
            states,
            num_simulations=150,
            dirichlet_alpha=None,
            batch_size=16,
            repetition_history=history,
        )
        return adapter.root_visit_policy().numpy()

    doubled = list(child_hashes) + list(child_hashes)
    with_history = run(torch.tensor([doubled], dtype=torch.int64))
    without = run(torch.zeros((1, 0), dtype=torch.int64))
    assert with_history.shape == without.shape == (1, 4544)
    assert not np.allclose(with_history, without, atol=1e-6), (
        "repetition history had no effect on the search"
    )
