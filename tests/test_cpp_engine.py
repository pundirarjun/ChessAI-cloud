"""Differential tests: az_cpp_mcts (C++) vs python-chess / GPUChess.

The C++ engine must be behaviorally identical to the Python reference:
legal move sets, FEN transitions, state hashes, plane encoding, material
draws, terminal detection, and the canonical 4544 action tables.
"""

import random

import chess
import numpy as np
import pytest
import torch

from az.reference import gpu_from_board
from environment.action_encoder import ActionEncoder
from environment.gpu_chess import BK, BQ, WK, WQ

m = pytest.importorskip("az_cpp_mcts", reason="az_cpp_mcts extension not built")

_ENCODER = ActionEncoder()

CURATED_FENS = [
    "startpos",
    # Standard middlegames / tactical.
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbq1rk1/pp2ppbp/3p1np1/2p5/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # En passant availability (legal and pinned-legal variants).
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 4",
    "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
    "8/8/8/8/3pP3/8/8/4K2k w - d3 0 1",
    # Castling rights edge cases.
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    # Promotion.
    "8/P7/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/p7/K6k b - - 0 1",
    # Check / mate / stalemate.
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1",
    # 50-move boundary.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 50",
    # Insufficient material with legal moves.
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/4k3/8/8/8/4K1N1 w - - 0 1",
    "8/2b5/8/4k3/8/8/8/4K3 w - - 0 1",
]


def _position(fen: str):
    if fen == "startpos":
        return m.Position.start(), chess.Board()
    return m.Position.from_fen(fen), chess.Board(fen)


def _cpp_legal(p) -> set[int]:
    return {int(a) for a in p.legal_actions()}


def _board_legal(board: chess.Board) -> set[int]:
    return {_ENCODER.encode(move) for move in board.legal_moves}


def test_action_tables_match_action_encoder():
    from_cpp, to_cpp, promo_cpp = m.action_tables()
    assert len(from_cpp) == len(to_cpp) == len(promo_cpp) == 4544
    for move, action_id in _ENCODER.move_to_id.items():
        assert int(from_cpp[action_id]) == move.from_square, action_id
        assert int(to_cpp[action_id]) == move.to_square, action_id
        assert int(promo_cpp[action_id]) == (move.promotion or 0), action_id


def test_action_tables_are_well_formed():
    from_cpp, to_cpp, promo_cpp = m.action_tables()
    for action_id in range(4544):
        f, t, p = m.decode_action(action_id)
        assert int(from_cpp[action_id]) == f
        assert int(to_cpp[action_id]) == t
        assert int(promo_cpp[action_id]) == p
        assert 0 <= f < 64 and 0 <= t < 64 and f != t
        if p:
            assert p in (2, 3, 4, 5)
        assert m.encode_action(f, t, p) == action_id
        assert _ENCODER.decode(action_id) == chess.Move(f, t, promotion=p or None)


@pytest.mark.parametrize("fen", CURATED_FENS, ids=range(len(CURATED_FENS)))
def test_legal_actions_match_python_chess(fen):
    p, board = _position(fen)
    assert _cpp_legal(p) == _board_legal(board)


@pytest.mark.parametrize("fen", CURATED_FENS, ids=range(len(CURATED_FENS)))
def test_fen_roundtrip(fen):
    p, board = _position(fen)
    expected = board.fen(en_passant="fen")
    assert p.to_fen() == expected


@pytest.mark.parametrize("fen", CURATED_FENS, ids=range(len(CURATED_FENS)))
def test_state_hash_matches_gpuchess(fen):
    p, board = _position(fen)
    gpu = gpu_from_board(board, "cpu")
    assert int(p.state_hash()) == int(gpu.state_hash()[0].item())


@pytest.mark.parametrize("fen", CURATED_FENS, ids=range(len(CURATED_FENS)))
def test_planes_match_gpuchess(fen):
    p, board = _position(fen)
    gpu = gpu_from_board(board, "cpu")
    cpp_planes = torch.from_numpy(np.array(p.to_model_input(), copy=True))
    assert torch.equal(cpp_planes, gpu.to_model_input()[0])


@pytest.mark.parametrize("fen", CURATED_FENS, ids=range(len(CURATED_FENS)))
def test_apply_matches_python_chess(fen):
    p, board = _position(fen)
    for action in sorted(_cpp_legal(p)):
        move = _ENCODER.decode(action)
        q = p.apply(action)
        after_board = board.copy(stack=False)
        after_board.push(move)
        assert q.to_fen() == after_board.fen(en_passant="fen"), (
            f"{fen} action {action} ({move.uci()})"
        )


@pytest.mark.parametrize(
    "fen,expected_terminal,expected_value",
    [
        # Checkmate: side to move is mated.
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", True, -1),
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", True, -1),
        # Stalemate.
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", True, 0),
        ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", True, 0),
        # Halfmove clock.
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 50", True, 0),
        # Insufficient material is NOT terminal here: like GPUChess.terminal_info
        # the caller ORs it in (self-play: terminal | insufficient_material).
        ("8/8/8/4k3/8/8/8/4K3 w - - 0 1", False, 0),
        # Normal position.
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", False, 0),
    ],
)
def test_terminal_info(fen, expected_terminal, expected_value):
    p, _ = _position(fen)
    terminal, value = p.terminal_info()
    assert bool(terminal) == expected_terminal, fen
    assert int(value) == expected_value, fen


def test_terminal_info_matches_python_chess_on_walk():
    rng = random.Random(4321)
    # Seeds include terminal and dead positions so every event class is
    # actually visited (random play alone almost never reaches them).
    seeds = [
        chess.Board(),
        chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
        chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1"),
    ]
    checked = {"terminal": 0, "check": 0, "insufficient": 0}

    def verify(board: chess.Board) -> None:
        p = m.Position.from_fen(board.fen(en_passant="fen"))
        terminal, value = p.terminal_info()
        no_moves = not any(board.legal_moves)
        # Mirrors GPUChess.terminal_info: insufficient material is applied by
        # the caller (self-play ORs terminal | insufficient_material).
        expected_terminal = no_moves or board.halfmove_clock >= 100
        assert bool(terminal) == expected_terminal, board.fen()
        expected_value = -1 if no_moves and board.is_check() else 0
        assert int(value) == expected_value, board.fen()
        assert bool(p.insufficient_material()) == board.is_insufficient_material(), (
            board.fen()
        )
        if no_moves:
            checked["terminal"] += 1
        if board.is_check():
            checked["check"] += 1
        if board.is_insufficient_material():
            checked["insufficient"] += 1

    # Deterministic sweep over the seeds for terminal/insufficient coverage.
    for _ in range(10):
        for seed in seeds:
            verify(seed)

    # Random walk for in-check coverage and general position variety.
    board = seeds[0]
    steps = 0
    while checked["check"] < 40 and steps < 2000:
        steps += 1
        verify(board)
        if (
            not any(board.legal_moves)
            or board.is_game_over(claim_draw=True)
            or board.ply() >= 200
        ):
            board = rng.choice(seeds).copy(stack=False)
        else:
            board.push(rng.choice(list(board.legal_moves)))

    assert checked["terminal"] >= 30, checked
    assert checked["check"] >= 40, checked
    assert checked["insufficient"] >= 10, checked


def test_insufficient_material_randomized():
    def random_material_board(rng: random.Random) -> chess.Board:
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

    rng = random.Random(20261005)
    mismatches = []
    for i in range(400):
        board = random_material_board(rng)
        p = m.Position.from_fen(board.fen())
        if bool(p.insufficient_material()) != board.is_insufficient_material():
            mismatches.append(board.fen())
    assert not mismatches, f"{len(mismatches)}/400 mismatches: {mismatches[:5]}"


def test_random_walk_differential():
    """Positions, transitions, hashes, and planes over random legal play."""
    rng = random.Random(1999)
    starts = [
        chess.Board(),
        # Promotion seed first so the (short) walk reaches it after the
        # first reset.
        chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1"),
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        chess.Board("rnbq1rk1/pp2ppbp/3p1np1/2p5/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8"),
    ]
    board = starts[0]
    checked = 0
    ep_seen = 0
    promo_seen = 0
    for step in range(120):
        if board.is_game_over(claim_draw=True) or board.ply() >= 80:
            board = starts[step % len(starts)]

        fen = board.fen(en_passant="fen")
        p = m.Position.from_fen(fen)
        gpu = gpu_from_board(board, "cpu")

        assert _cpp_legal(p) == _board_legal(board), f"legal mismatch at {fen}"
        assert p.to_fen() == fen, f"fen mismatch at {fen}"
        assert int(p.state_hash()) == int(gpu.state_hash()[0].item()), (
            f"hash mismatch at {fen}"
        )
        cpp_planes = torch.from_numpy(np.array(p.to_model_input(), copy=True))
        assert torch.equal(cpp_planes, gpu.to_model_input()[0]), f"planes at {fen}"
        if board.ep_square is not None:
            ep_seen += 1

        promotions = [mv for mv in board.legal_moves if mv.promotion]
        if promotions:
            move = rng.choice(promotions)
            promo_seen += 1
        else:
            move = rng.choice(list(board.legal_moves))
        action = _ENCODER.encode(move)
        p = p.apply(action)
        board.push(move)
        checked += 1
    assert checked == 120
    assert ep_seen > 0, "walk never reached an en passant square"
    assert promo_seen > 0, "walk never reached a promotion"


def _start_arrays(games: int):
    start = m.Position.start()
    pieces = np.tile(np.array(start.pieces(), dtype=np.int64), (games, 1))
    turn = np.full(games, int(start.turn), dtype=np.int8)
    castling = np.full(games, int(start.castling), dtype=np.int16)
    ep = np.full(games, int(start.ep), dtype=np.int16)
    halfmove = np.full(games, int(start.halfmove), dtype=np.int16)
    fullmove = np.full(games, int(start.fullmove), dtype=np.int16)
    history = np.zeros((games, 0), dtype=np.int64)
    return pieces, turn, castling, ep, halfmove, fullmove, history


def _zero_eval(planes):
    n = planes.shape[0]
    return np.zeros((n, 4544), dtype=np.float32), np.zeros((n,), dtype=np.float32)


def test_mcts_smoke_search_is_consistent():
    games = 3
    arrays = _start_arrays(games)
    mcts = m.BatchMcts()
    mcts.set_seed(7)
    mcts.set_c_puct(1.5)
    mcts.search(
        *arrays,
        num_simulations=64,
        dirichlet_alpha=-1.0,
        dirichlet_epsilon=0.0,
        batch_size=4,
        eval_fn=_zero_eval,
    )
    policy = mcts.root_visit_policy()
    assert policy.shape == (games, 4544)
    assert np.all(np.isfinite(policy))
    assert np.all(policy >= 0.0)
    legal = _board_legal(chess.Board())
    for g in range(games):
        row = policy[g]
        mask = np.ones(4544, dtype=bool)
        mask[list(legal)] = False
        assert np.all(row[mask] == 0.0), "policy mass on illegal actions"
        assert row[list(legal)].sum() == pytest.approx(1.0), "policy not normalized"

    actions = mcts.select_actions(0.0)
    assert actions.shape == (games,)
    for a in actions:
        assert int(a) in legal

    pieces, turn, castling, ep, halfmove, fullmove = mcts.advance(actions)
    assert pieces.shape == (games, 12)
    for g in range(games):
        board = chess.Board()
        board.push(_ENCODER.decode(int(actions[g])))
        expected_pieces = np.zeros(12, dtype=np.int64)
        for square, piece in board.piece_map().items():
            plane = {
                chess.PAWN: 0,
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK: 3,
                chess.QUEEN: 4,
                chess.KING: 5,
            }[piece.piece_type] + (0 if piece.color else 6)
            expected_pieces[plane] |= np.int64(1) << square
        assert np.array_equal(pieces[g], expected_pieces)
        assert int(turn[g]) == int(not board.turn)
        expected_castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            expected_castling |= WK
        if board.has_queenside_castling_rights(chess.WHITE):
            expected_castling |= WQ
        if board.has_kingside_castling_rights(chess.BLACK):
            expected_castling |= BK
        if board.has_queenside_castling_rights(chess.BLACK):
            expected_castling |= BQ
        assert int(castling[g]) == expected_castling
        assert int(ep[g]) == (-1 if board.ep_square is None else board.ep_square)
        assert int(halfmove[g]) == board.halfmove_clock
        assert int(fullmove[g]) == board.fullmove_number


def test_mcts_deterministic_with_same_seed():
    def run(seed: int) -> np.ndarray:
        mcts = m.BatchMcts()
        mcts.set_seed(seed)
        mcts.search(
            *_start_arrays(2),
            num_simulations=48,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            batch_size=4,
            eval_fn=_zero_eval,
        )
        return mcts.root_visit_policy()

    first = run(123)
    second = run(123)
    assert np.array_equal(first, second)


def test_mcts_eval_callback_receives_correct_shapes():
    seen = {}

    def eval_fn(planes):
        planes = np.asarray(planes)
        n = planes.shape[0]
        seen["planes"] = planes.shape
        assert planes.dtype == np.float32
        logits = np.zeros((n, 4544), dtype=np.float32)
        logits[:, 10] = 1.0
        values = np.full((n,), 0.25, dtype=np.float32)
        return logits, values

    mcts = m.BatchMcts()
    mcts.set_seed(5)
    mcts.search(
        *_start_arrays(4),
        num_simulations=8,
        dirichlet_alpha=-1.0,
        dirichlet_epsilon=0.0,
        batch_size=2,
        eval_fn=eval_fn,
    )
    assert seen["planes"][1] == 18 * 64
    assert seen["planes"][0] <= 8  # at most batch_size x games rows per call
    policy = mcts.root_visit_policy()
    assert np.isfinite(policy).all()
