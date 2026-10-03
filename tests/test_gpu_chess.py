import torch

from environment.gpu_chess import (
    BK,
    BQ,
    WK,
    WQ,
    GPUChess,
    _action_tables,
    _signed_u64,
)


def action_id(from_sq, to_sq, promotion=0):
    t = _action_tables()
    hits = [i for i, (f, to, p) in enumerate(zip(t["from_sq"], t["to_sq"], t["promo"])) if int(f) == from_sq and int(to) == to_sq and int(p) == promotion]
    assert len(hits) == 1
    return hits[0]


def test_initial_position():
    g = GPUChess("cpu", 1)
    legal = g.legal_move_mask()[0]
    assert int(legal.sum()) == 20
    assert not bool(g.is_in_check()[0])


def test_e2_e4():
    g = GPUChess("cpu", 1)
    g = g.push_actions(torch.tensor([action_id(12, 28)]))
    assert bool(g.turn[0])
    assert int(g.ep_square[0]) == 20
    assert int(g.legal_move_mask()[0].sum()) == 20


def test_pinned_piece():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)   # white king e1
    g.pieces[0, 3] = _signed_u64(1 << 12)  # white rook e2
    g.pieces[0, 9] = _signed_u64(1 << 60)  # black rook e8
    g.pieces[0, 11] = _signed_u64(1 << 56) # black king a8
    g.castling.zero_()
    legal = g.legal_move_mask()[0]
    t = _action_tables()
    for a in legal.nonzero().flatten().tolist():
        if int(t["from_sq"][a]) == 12:
            assert int(t["to_sq"][a]) % 8 == 4


def test_illegal_en_passant_exposes_king():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)    # white king e1
    g.pieces[0, 0] = _signed_u64(1 << 36)   # white pawn e5
    g.pieces[0, 6] = _signed_u64(1 << 35)   # black pawn d5
    g.pieces[0, 9] = _signed_u64(1 << 60)   # black rook e8
    g.pieces[0, 11] = _signed_u64(1 << 56)
    g.castling.zero_()
    g.ep_square[0] = 43  # d6
    legal = g.legal_move_mask()[0]
    ep = action_id(36, 43)
    assert not bool(legal[ep])


def test_promotion():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)
    g.pieces[0, 0] = _signed_u64(1 << 48)
    g.pieces[0, 11] = _signed_u64(1 << 60)
    g.castling.zero_()
    legal = g.legal_move_mask()[0]
    assert all(bool(legal[action_id(48, 56, p)]) for p in (2, 3, 4, 5))


def test_model_state_shape():
    g = GPUChess("cpu", 2)
    x = g.to_model_input()
    assert x.shape == (2, 18, 8, 8)
    assert x.dtype == torch.float32


def _minimal_position():
    game = GPUChess("cpu", 1)
    game.pieces.zero_()
    game.pieces[0, 5] = _signed_u64(1 << 4)   # white king e1
    game.pieces[0, 11] = _signed_u64(1 << 60) # black king e8
    game.castling.zero_()
    game.ep_square.fill_(-1)
    game.halfmove_clock.zero_()
    return game


def test_state_hash_ignores_unavailable_en_passant_target():
    unavailable = _minimal_position()
    unavailable.pieces[0, 0] = _signed_u64(1 << 36)  # white pawn e5
    unavailable.ep_square[0] = 43  # d6, but no black pawn at d5

    no_target = _minimal_position()
    no_target.pieces[0, 0] = _signed_u64(1 << 36)

    assert torch.equal(unavailable.state_hash(), no_target.state_hash())


def test_state_hash_retains_legal_en_passant_target_and_capture():
    game = _minimal_position()
    game.pieces[0, 0] = _signed_u64(1 << 36)  # white pawn e5
    game.pieces[0, 6] = _signed_u64(1 << 35)  # black pawn d5
    game.ep_square[0] = 43  # d6

    without_ep = _minimal_position()
    without_ep.pieces[0, 0] = _signed_u64(1 << 36)
    without_ep.pieces[0, 6] = _signed_u64(1 << 35)

    assert not torch.equal(game.state_hash(), without_ep.state_hash())
    after = game.push_actions(torch.tensor([action_id(36, 43)]))
    assert int(after.pieces[0, 6] & _signed_u64(1 << 35)) == 0
    assert int(after.pieces[0, 0] & _signed_u64(1 << 43)) != 0


def test_rook_capture_only_clears_the_matching_castling_right():
    game = _minimal_position()
    game.pieces[0, 3] = _signed_u64((1 << 0) | (1 << 7))
    game.pieces[0, 9] = _signed_u64((1 << 56) | (1 << 63))
    game.castling[0] = WK | WQ | BK | BQ

    # Black rook from a8 captures the original white rook on a1.
    game.turn[0] = True
    after_white_a_rook = game.apply_actions_unchecked(
        torch.tensor([0]), torch.tensor([action_id(56, 0)])
    )
    assert int(after_white_a_rook.castling[0]) & WQ == 0
    assert int(after_white_a_rook.castling[0]) & WK == WK

    # White rook from h1 captures the original black rook on h8.
    game.turn[0] = False
    after_black_h_rook = game.apply_actions_unchecked(
        torch.tensor([0]), torch.tensor([action_id(7, 63)])
    )
    assert int(after_black_h_rook.castling[0]) & BK == 0
    assert int(after_black_h_rook.castling[0]) & BQ == BQ


def test_promoted_rook_capture_does_not_clear_home_rook_rights():
    game = _minimal_position()
    game.pieces[0, 3] = _signed_u64((1 << 0) | (1 << 7) | (1 << 28))
    game.pieces[0, 9] = _signed_u64(1 << 56)
    game.castling[0] = WK | WQ | BK | BQ
    game.turn[0] = True

    after = game.apply_actions_unchecked(
        torch.tensor([0]), torch.tensor([action_id(56, 28)])
    )
    assert int(after.castling[0]) & (WK | WQ) == (WK | WQ)
