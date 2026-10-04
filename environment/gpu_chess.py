"""GPU-native chess state, move generation, and move application.

This module deliberately does not use python-chess in the hot path.  It keeps
chess positions as bitboards in torch tensors and performs move generation and
legality checks on CUDA when the supplied tensors are on CUDA.

The 4544 action ordering is identical to environment.action_encoder.ActionEncoder:
4032 normal from/to moves followed by 512 promotion moves (white promotions,
then black promotions; knight, bishop, rook, queen).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_PIECE_TYPES = 6
NUM_PLANES = 12
ACTION_SPACE_SIZE = 4544
MAX_LEGAL_MOVES = 256

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WHITE = 0
BLACK = 1

# Castling-right bits: WK, WQ, BK, BQ.
WK = 1
WQ = 2
BK = 4
BQ = 8

# Piece-plane index: white 0..5, black 6..11.


def _signed_u64(value: int) -> int:
    """Return a uint64 value represented in torch int64 two's complement."""
    value &= (1 << 64) - 1
    return value if value < (1 << 63) else value - (1 << 64)


@lru_cache(maxsize=1)
def _action_tables():
    """Build the exact 4544 action ordering without importing python-chess."""
    from_sq = []
    to_sq = []
    promo = []

    # Normal moves: ActionEncoder iterates from_square, then to_square,
    # skipping from_square == to_square.
    for f in range(64):
        for t in range(64):
            if f == t:
                continue
            from_sq.append(f); to_sq.append(t); promo.append(0)

    # Promotion moves: white from rank 7 to rank 8, then black from rank 2
    # to rank 1, with N/B/R/Q in that order.
    for f in range(48, 56):
        for t in range(56, 64):
            for p in (2, 3, 4, 5):
                from_sq.append(f); to_sq.append(t); promo.append(p)
    for f in range(8, 16):
        for t in range(0, 8):
            for p in (2, 3, 4, 5):
                from_sq.append(f); to_sq.append(t); promo.append(p)

    from_sq = np.asarray(from_sq, dtype=np.int64)
    to_sq = np.asarray(to_sq, dtype=np.int64)
    promo = np.asarray(promo, dtype=np.int64)
    assert len(from_sq) == ACTION_SPACE_SIZE

    geom = np.zeros((6, ACTION_SPACE_SIZE), dtype=np.bool_)
    between = np.zeros(ACTION_SPACE_SIZE, dtype=np.int64)
    from_bits = np.empty(ACTION_SPACE_SIZE, dtype=np.int64)
    to_bits = np.empty(ACTION_SPACE_SIZE, dtype=np.int64)

    def sq_rank(s): return s // 8
    def sq_file(s): return s % 8

    for a in range(ACTION_SPACE_SIZE):
        f = int(from_sq[a]); t = int(to_sq[a])
        fr, ff = sq_rank(f), sq_file(f)
        tr, tf = sq_rank(t), sq_file(t)
        dr, df = tr - fr, tf - ff
        from_bits[a] = _signed_u64(1 << f)
        to_bits[a] = _signed_u64(1 << t)
        geom[KNIGHT, a] = (abs(dr), abs(df)) in ((1, 2), (2, 1))
        geom[KING, a] = max(abs(dr), abs(df)) == 1
        geom[BISHOP, a] = abs(dr) == abs(df) and dr != 0
        geom[ROOK, a] = (dr == 0 and df != 0) or (df == 0 and dr != 0)
        geom[QUEEN, a] = geom[BISHOP, a] or geom[ROOK, a]
        if geom[QUEEN, a]:
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_f = 0 if df == 0 else (1 if df > 0 else -1)
            r, c = fr + step_r, ff + step_f
            mask = 0
            while (r, c) != (tr, tf):
                mask |= 1 << (r * 8 + c)
                r += step_r; c += step_f
            between[a] = _signed_u64(mask) if mask else 0

    pawn_masks = {
        "w_single": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
        "w_double": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
        "w_capture": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
        "b_single": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
        "b_double": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
        "b_capture": np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_),
    }
    for a in range(ACTION_SPACE_SIZE):
        f = int(from_sq[a]); t = int(to_sq[a])
        fr, ff = sq_rank(f), sq_file(f); tr, tf = sq_rank(t), sq_file(t)
        dr, df = tr - fr, tf - ff; is_promo = promo[a] != 0
        pawn_masks["w_single"][a] = dr == 1 and df == 0 and fr <= 6 and (fr != 6 or is_promo)
        pawn_masks["w_double"][a] = dr == 2 and df == 0 and fr == 1 and not is_promo
        pawn_masks["w_capture"][a] = dr == 1 and abs(df) == 1 and fr <= 6 and (fr != 6 or is_promo)
        pawn_masks["b_single"][a] = dr == -1 and df == 0 and fr >= 1 and (fr != 1 or is_promo)
        pawn_masks["b_double"][a] = dr == -2 and df == 0 and fr == 6 and not is_promo
        pawn_masks["b_capture"][a] = dr == -1 and abs(df) == 1 and fr >= 1 and (fr != 1 or is_promo)

    castle_kind = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    for a in range(ACTION_SPACE_SIZE):
        f = int(from_sq[a]); t = int(to_sq[a])
        if (f, t) == (4, 6): castle_kind[a] = 1
        elif (f, t) == (4, 2): castle_kind[a] = 2
        elif (f, t) == (60, 62): castle_kind[a] = 3
        elif (f, t) == (60, 58): castle_kind[a] = 4

    return {
        "from_sq": from_sq, "to_sq": to_sq, "promo": promo, "geom": geom,
        "between": between, "from_bits": from_bits, "to_bits": to_bits,
        **pawn_masks, "castle_kind": castle_kind,
    }


class GPUChess:
    """Batch of chess positions stored entirely as torch tensors."""

    def __init__(self, device: torch.device | str = "cuda", batch_size: int = 1):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        t = _action_tables()
        self.from_sq = torch.tensor(t["from_sq"], device=self.device, dtype=torch.long)
        self.to_sq = torch.tensor(t["to_sq"], device=self.device, dtype=torch.long)
        self.promo = torch.tensor(t["promo"], device=self.device, dtype=torch.long)
        self.from_bits = torch.tensor(t["from_bits"], device=self.device, dtype=torch.int64)
        self.to_bits = torch.tensor(t["to_bits"], device=self.device, dtype=torch.int64)
        self.between = torch.tensor(t["between"], device=self.device, dtype=torch.int64)
        self.geom = torch.tensor(t["geom"], device=self.device, dtype=torch.bool)
        self.w_single = torch.tensor(t["w_single"], device=self.device, dtype=torch.bool)
        self.w_double = torch.tensor(t["w_double"], device=self.device, dtype=torch.bool)
        self.w_capture = torch.tensor(t["w_capture"], device=self.device, dtype=torch.bool)
        self.b_single = torch.tensor(t["b_single"], device=self.device, dtype=torch.bool)
        self.b_double = torch.tensor(t["b_double"], device=self.device, dtype=torch.bool)
        self.b_capture = torch.tensor(t["b_capture"], device=self.device, dtype=torch.bool)
        self.castle_kind = torch.tensor(t["castle_kind"], device=self.device, dtype=torch.int8)
        # Precompute pawn double-move intermediate squares once.
        mid_sq = self.from_sq + torch.where(
            self.from_sq < 32,
            torch.full_like(self.from_sq, 8),
            torch.full_like(self.from_sq, -8),
        )
        self.mid_sq = mid_sq.clamp(0, 63)

        square_bits = [
            _signed_u64(1 << s) for s in range(64)
        ]
        self.square_bits = torch.tensor(square_bits, device=self.device, dtype=torch.int64)

        # Precomputed non-sliding attacks used by attack detection.
        self.knight_attacks = self._build_jump_attacks(knight=True)
        self.king_attacks = self._build_jump_attacks(knight=False)
        self.white_pawn_attacks = self._build_pawn_attacks(WHITE)
        self.black_pawn_attacks = self._build_pawn_attacks(BLACK)
        self.ray_squares = self._build_rays()

        self.reset(self.batch_size)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_jump_attacks(self, knight: bool) -> torch.Tensor:
        table = torch.zeros((64,), dtype=torch.int64, device=self.device)
        offsets = ((1, 2), (2, 1), (-1, 2), (-2, 1),
                   (1, -2), (2, -1), (-1, -2), (-2, -1)) if knight else (
                   (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1))
        vals = []
        for s in range(64):
            r, c = divmod(s, 8)
            mask = 0
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    mask |= 1 << (rr * 8 + cc)
            vals.append(_signed_u64(mask))
        return torch.tensor(vals, dtype=torch.int64, device=self.device)

    def _build_pawn_attacks(self, color: int) -> torch.Tensor:
        vals = []
        for s in range(64):
            r, c = divmod(s, 8)
            dr = 1 if color == WHITE else -1
            mask = 0
            for dc in (-1, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    mask |= 1 << (rr * 8 + cc)
            vals.append(_signed_u64(mask))
        return torch.tensor(vals, dtype=torch.int64, device=self.device)

    def _build_rays(self) -> torch.Tensor:
        # Direction order: N,S,E,W,NE,NW,SE,SW.  Unused slots are -1.
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1))
        out = torch.full((64, 8, 7), -1, dtype=torch.long, device=self.device)
        for s in range(64):
            r, c = divmod(s, 8)
            for d, (dr, dc) in enumerate(dirs):
                rr, cc = r + dr, c + dc
                k = 0
                while 0 <= rr < 8 and 0 <= cc < 8 and k < 7:
                    out[s, d, k] = rr * 8 + cc
                    rr += dr; cc += dc; k += 1
        return out

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------
    def reset(self, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = int(batch_size)
        b = self.batch_size
        self.pieces = torch.zeros((b, 12), dtype=torch.int64, device=self.device)
        # Initial position.
        self.pieces[:, WHITE * 6 + PAWN] = self._bb_rank(1)
        self.pieces[:, WHITE * 6 + KNIGHT] = self._bb_squares((1, 6))
        self.pieces[:, WHITE * 6 + BISHOP] = self._bb_squares((2, 5))
        self.pieces[:, WHITE * 6 + ROOK] = self._bb_squares((0, 7))
        self.pieces[:, WHITE * 6 + QUEEN] = self._bb_squares((3,))
        self.pieces[:, WHITE * 6 + KING] = self._bb_squares((4,))
        self.pieces[:, BLACK * 6 + PAWN] = self._bb_rank(6)
        self.pieces[:, BLACK * 6 + KNIGHT] = self._bb_squares((57, 62))
        self.pieces[:, BLACK * 6 + BISHOP] = self._bb_squares((58, 61))
        self.pieces[:, BLACK * 6 + ROOK] = self._bb_squares((56, 63))
        self.pieces[:, BLACK * 6 + QUEEN] = self._bb_squares((59,))
        self.pieces[:, BLACK * 6 + KING] = self._bb_squares((60,))
        self.turn = torch.zeros((b,), dtype=torch.bool, device=self.device)  # False=white
        self.castling = torch.full((b,), WK | WQ | BK | BQ, dtype=torch.int16, device=self.device)
        self.ep_square = torch.full((b,), -1, dtype=torch.int16, device=self.device)
        self.halfmove_clock = torch.zeros((b,), dtype=torch.int16, device=self.device)
        self.fullmove_number = torch.ones((b,), dtype=torch.int16, device=self.device)
        return self

    def _bb_rank(self, rank: int) -> int:
        return _signed_u64(0xFF << (rank * 8))

    def _bb_squares(self, squares) -> int:
        value = 0
        for s in squares:
            value |= 1 << int(s)
        return _signed_u64(value)

    def clone(self) -> "GPUChess":
        other = object.__new__(GPUChess)
        other.__dict__ = self.__dict__.copy()
        for name in ("pieces", "turn", "castling", "ep_square", "halfmove_clock", "fullmove_number"):
            setattr(other, name, getattr(self, name).clone())
        return other

    def select(self, indices: torch.Tensor) -> "GPUChess":
        """Create a lightweight view over selected batch elements."""
        indices = indices.to(device=self.device, dtype=torch.long)
        other = object.__new__(GPUChess)
        other.__dict__ = self.__dict__.copy()
        other.batch_size = int(indices.numel())
        other.pieces = self.pieces[indices]
        other.turn = self.turn[indices]
        other.castling = self.castling[indices]
        other.ep_square = self.ep_square[indices]
        other.halfmove_clock = self.halfmove_clock[indices]
        other.fullmove_number = self.fullmove_number[indices]
        return other

    @property
    def batch_size_current(self):
        return self.pieces.shape[0]

    # ------------------------------------------------------------------
    # Board queries
    # ------------------------------------------------------------------
    def occupancies(self):
        # Reduce explicitly for broad PyTorch-version compatibility.
        white = self.pieces[:, 0].clone()
        for i in range(1, 6):
            white = white | self.pieces[:, i]
        black = self.pieces[:, 6].clone()
        for i in range(7, 12):
            black = black | self.pieces[:, i]
        return white, black, white | black

    def _king_square(self, color: int) -> torch.Tensor:
        bb = self.pieces[:, color * 6 + KING]
        # Exactly one king is required in a valid position. Convert to square
        # by matching against the precomputed bit values.
        return (bb[:, None] == self.square_bits[None, :]).to(torch.long).argmax(dim=1)

    def to_model_input(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the same 18x8x8 orientation as StateEncoder, without CPU work."""
        pieces = self.pieces if indices is None else self.pieces[indices]
        b = pieces.shape[0]
        planes = torch.zeros((b, 18, 8, 8), dtype=torch.float32, device=self.device)
        # squares: [64], model index: row=7-rank, col=file.
        for p in range(12):
            bits = pieces[:, p]
            occ = (bits[:, None] & self.square_bits[None, :]) != 0
            plane = occ.reshape(b, 8, 8).flip(1)
            planes[:, p] = plane.float()
        white_turn = ~self.turn
        planes[:, 12] = white_turn[:, None, None].float()
        planes[:, 13] = ((self.castling & WK) != 0)[:, None, None].float()
        planes[:, 14] = ((self.castling & WQ) != 0)[:, None, None].float()
        planes[:, 15] = ((self.castling & BK) != 0)[:, None, None].float()
        planes[:, 16] = ((self.castling & BQ) != 0)[:, None, None].float()
        ep = self.ep_square if indices is None else self.ep_square[indices]
        ep_mask = ep[:, None] == torch.arange(64, device=self.device)[None, :]
        planes[:, 17] = ep_mask.reshape(b, 8, 8).flip(1).float()
        return planes

    # ------------------------------------------------------------------
    # Attack detection
    # ------------------------------------------------------------------
    def _attacked(self, state_indices: torch.Tensor, squares: torch.Tensor, by_color: torch.Tensor) -> torch.Tensor:
        """Whether each selected square is attacked in its corresponding state."""
        # state_indices and squares are [N], by_color is [N].
        p = self.pieces[state_indices]
        occ = p[:, 0].clone()
        for i in range(1, 12):
            occ = occ | p[:, i]

        result = torch.zeros_like(squares, dtype=torch.bool)

        # Pawn attacks: to test whether target is attacked, use inverse attack
        # relation (white pawn must be one rank below target, etc.).
        target_bit = self.square_bits[squares]
        target_r = squares // 8
        target_f = squares % 8
        white_pawns = p[:, PAWN]
        black_pawns = p[:, 6 + PAWN]
        white_src_mask = torch.zeros_like(target_bit)
        black_src_mask = torch.zeros_like(target_bit)
        for df in (-1, 1):
            wf = target_f - df
            valid = (target_r >= 1) & (wf >= 0) & (wf < 8)
            ws = (target_r - 1) * 8 + wf
            ws = torch.clamp(ws, 0, 63)
            white_src_mask = white_src_mask | (self.square_bits[ws] * valid.to(torch.int64))
            bf = target_f - df
            validb = (target_r <= 6) & (bf >= 0) & (bf < 8)
            bs = (target_r + 1) * 8 + bf
            bs = torch.clamp(bs, 0, 63)
            black_src_mask = black_src_mask | (self.square_bits[bs] * validb.to(torch.int64))
        result |= torch.where(by_color == WHITE, (white_pawns & white_src_mask) != 0, (black_pawns & black_src_mask) != 0)

        # Knights and kings via lookup table.
        knight_bb = self.knight_attacks[squares]
        king_bb = self.king_attacks[squares]
        enemy_knights = torch.where(by_color == WHITE, p[:, KNIGHT], p[:, 6 + KNIGHT])
        enemy_kings = torch.where(by_color == WHITE, p[:, KING], p[:, 6 + KING])
        result |= (enemy_knights & knight_bb) != 0
        result |= (enemy_kings & king_bb) != 0

        # Sliding attacks. Scan each of 8 rays from target until first blocker.
        for d in range(8):
            ray = self.ray_squares[squares, d]  # [N,7]
            valid = ray >= 0
            safe_ray = torch.clamp(ray, 0, 63)
            ray_bits = self.square_bits[safe_ray]
            blockers = (occ[:, None] & ray_bits) != 0
            any_blocker = blockers & valid
            # First blocker: argmax is safe because blockers are ordered nearest-first.
            first_pos = any_blocker.to(torch.int8).argmax(dim=1)
            has = any_blocker.any(dim=1)
            first_sq = safe_ray[torch.arange(safe_ray.shape[0], device=self.device), first_pos]
            first_bit = self.square_bits[first_sq]
            if d < 4:
                enemy_slider = torch.where(by_color == WHITE, p[:, ROOK] | p[:, QUEEN], p[:, 6 + ROOK] | p[:, 6 + QUEEN])
            else:
                enemy_slider = torch.where(by_color == WHITE, p[:, BISHOP] | p[:, QUEEN], p[:, 6 + BISHOP] | p[:, 6 + QUEEN])
            result |= has & ((enemy_slider & first_bit) != 0)

        return result

    def is_in_check(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is None:
            indices = torch.arange(self.pieces.shape[0], device=self.device)
        p = self.pieces[indices]
        turn = self.turn[indices]
        king_bb = p[torch.arange(p.shape[0], device=self.device), (turn.long() * 6) + KING]
        king_sq = (king_bb[:, None] == self.square_bits[None, :]).to(torch.long).argmax(dim=1)
        by_color = (~turn).long()
        return self._attacked(indices, king_sq, by_color)

    # ------------------------------------------------------------------
    # Pseudo-legal moves
    # ------------------------------------------------------------------
    def pseudo_legal_mask(self) -> torch.Tensor:
        b = self.pieces.shape[0]
        white_occ, black_occ, occ = self.occupancies()
        own = torch.where(self.turn, black_occ, white_occ)
        enemy = torch.where(self.turn, white_occ, black_occ)
        mask = torch.zeros((b, ACTION_SPACE_SIZE), dtype=torch.bool, device=self.device)

        # Build source-piece occupancy tests for each piece type.
        for t in (KNIGHT, BISHOP, ROOK, QUEEN, KING):
            piece_bb = torch.where(self.turn, self.pieces[:, 6 + t], self.pieces[:, t])
            has_source = (piece_bb[:, None] & self.from_bits[None, :]) != 0
            valid = has_source & self.geom[t][None, :]
            valid &= (self.promo == 0)[None, :]
            valid &= (own[:, None] & self.to_bits[None, :]) == 0
            if t in (BISHOP, ROOK, QUEEN):
                valid &= (occ[:, None] & self.between[None, :]) == 0
            mask |= valid

        # Pawns.
        pawn_bb = torch.where(self.turn, self.pieces[:, 6 + PAWN], self.pieces[:, PAWN])
        has_source = (pawn_bb[:, None] & self.from_bits[None, :]) != 0
        is_white = ~self.turn
        target_empty = (occ[:, None] & self.to_bits[None, :]) == 0
        target_enemy = (enemy[:, None] & self.to_bits[None, :]) != 0
        ep_target = self.ep_square[:, None] == self.to_sq[None, :]

        w_single = has_source & self.w_single[None, :] & target_empty
        w_double = has_source & self.w_double[None, :] & target_empty
        # Double pawn move also requires intermediate square empty.
        mid_bit = self.square_bits[self.mid_sq]
        w_double &= (occ[:, None] & mid_bit[None, :]) == 0
        w_capture = has_source & self.w_capture[None, :] & (target_enemy | ep_target)

        b_single = has_source & self.b_single[None, :] & target_empty
        b_double = has_source & self.b_double[None, :] & target_empty
        b_double &= (occ[:, None] & mid_bit[None, :]) == 0
        b_capture = has_source & self.b_capture[None, :] & (target_enemy | ep_target)

        pawn_valid = torch.where(is_white[:, None], w_single | w_double | w_capture, b_single | b_double | b_capture)
        # A promotion action must have a promotion piece; ordinary pawn moves
        # to the last rank are excluded by the precomputed categories.
        mask |= pawn_valid

        # A legal chess move may never capture the opponent king.
        enemy_king = torch.where(self.turn, self.pieces[:, KING], self.pieces[:, 6 + KING])
        mask &= (enemy_king[:, None] & self.to_bits[None, :]) == 0

        # Castling-specific conditions.
        ck = self.castle_kind[None, :]
        castle = ck > 0
        castle &= ((self.promo == 0)[None, :])
        # White/black king and rook presence.
        white_king = self.pieces[:, KING]
        black_king = self.pieces[:, 6 + KING]
        white_rook = self.pieces[:, ROOK]
        black_rook = self.pieces[:, 6 + ROOK]
        e1 = self.square_bits[4]; a1 = self.square_bits[0]; h1 = self.square_bits[7]
        e8 = self.square_bits[60]; a8 = self.square_bits[56]; h8 = self.square_bits[63]
        for kind, right, king_bb, rook_bb, king_sq, rook_sq, empty_sqs in (
            (1, WK, white_king, white_rook, e1, h1, (5, 6)),
            (2, WQ, white_king, white_rook, e1, a1, (1, 2, 3)),
            (3, BK, black_king, black_rook, e8, h8, (61, 62)),
            (4, BQ, black_king, black_rook, e8, a8, (57, 58, 59)),
        ):
            candidate = (ck == kind) & (((self.castling & right) != 0)[:, None])
            candidate &= (king_bb[:, None] == king_sq)
            candidate &= ((rook_bb & rook_sq) != 0)[:, None]
            for sq in empty_sqs:
                candidate &= ((occ & self.square_bits[sq]) == 0)[:, None]
            # Side must match the castle kind.
            candidate &= ((~self.turn) if kind <= 2 else self.turn)[:, None]
            # Only allow if king is not currently in check and transit squares
            # are not attacked.  These are checked on the current board.
            idx = torch.arange(b, device=self.device)
            current_check = self.is_in_check()
            transit = 5 if kind == 1 else 3 if kind == 2 else 61 if kind == 3 else 59
            final = 6 if kind == 1 else 2 if kind == 2 else 62 if kind == 3 else 58
            transit_attacked = self._attacked(idx, torch.full((b,), transit, device=self.device, dtype=torch.long), torch.where(self.turn, torch.zeros(b, dtype=torch.long, device=self.device), torch.ones(b, dtype=torch.long, device=self.device)))
            final_attacked = self._attacked(idx, torch.full((b,), final, device=self.device, dtype=torch.long), torch.where(self.turn, torch.zeros(b, dtype=torch.long, device=self.device), torch.ones(b, dtype=torch.long, device=self.device)))
            candidate &= (~current_check & ~transit_attacked & ~final_attacked)[:, None]
            # A castling-geometry action may also be a genuine rook/queen slide
            # from the castling square (e.g. Re1-g1 with the king elsewhere).
            # The two cases are mutually exclusive per state: castling requires
            # the king on the home square while a slide requires a rook or
            # queen there.  Never clear generic slides for failed candidates.
            mask |= candidate

        return mask

    def legal_move_mask(self) -> torch.Tensor:
        """Generate legal moves by pseudo generation + GPU king-safety filtering."""
        pseudo = self.pseudo_legal_mask()
        b = self.pieces.shape[0]
        flat = torch.nonzero(pseudo, as_tuple=False)
        if flat.numel() == 0:
            return pseudo
        state_idx = flat[:, 0]
        actions = flat[:, 1]
        legal = self._validate_actions(state_idx, actions)
        out = torch.zeros_like(pseudo)
        out[state_idx, actions] = legal
        return out

    # ------------------------------------------------------------------
    # Move application and legality
    # ------------------------------------------------------------------
    def _apply_flat(self, state_idx: torch.Tensor, actions: torch.Tensor):
        """Apply one action to each selected state using batched CUDA ops.

        The implementation intentionally avoids per-piece ``if ... .any()``
        branches. Those branches synchronize the CPU with CUDA repeatedly and
        were a major source of the low GPU utilization seen during self-play.
        """
        n = state_idx.numel()
        new = object.__new__(GPUChess)
        new.__dict__ = self.__dict__.copy()
        original = self.pieces[state_idx]
        new.pieces = original.clone()
        new.turn = self.turn[state_idx].clone()
        new.castling = self.castling[state_idx].clone()
        new.ep_square = self.ep_square[state_idx].clone()
        new.halfmove_clock = self.halfmove_clock[state_idx].clone()
        new.fullmove_number = self.fullmove_number[state_idx].clone()

        frm = self.from_sq[actions]
        to = self.to_sq[actions]
        fb = self.square_bits[frm]
        tb = self.square_bits[to]
        promo = self.promo[actions]
        turn = new.turn
        color = turn.long()

        # Determine the moving piece from the six piece planes of the side to move.
        own = torch.where(turn[:, None], original[:, 6:12], original[:, 0:6])
        hit = (own & fb[:, None]) != 0
        moving = hit.to(torch.int8).argmax(dim=1).long()

        # Remove moving piece from its source and clear every destination square
        # before placing the mover/captured result.
        moving_idx = color * 6 + moving
        moving_onehot = torch.nn.functional.one_hot(moving_idx, num_classes=12).to(torch.bool)
        cleared_source = new.pieces & ~fb[:, None]
        new.pieces = torch.where(moving_onehot, cleared_source, new.pieces)
        new.pieces = new.pieces & ~tb[:, None]

        # En-passant capture removes the pawn behind the destination square.
        is_pawn = moving == PAWN
        df = (to % 8 - frm % 8).abs()
        is_ep = is_pawn & (df == 1) & (to == self.ep_square[state_idx])
        captured_sq = to + torch.where(
            turn,
            torch.full_like(to, 8),
            torch.full_like(to, -8),
        )
        captured_bit = self.square_bits[captured_sq.clamp(0, 63)]
        ep_white = is_ep & turn
        ep_black = is_ep & (~turn)
        new.pieces[:, PAWN] = torch.where(
            ep_white,
            new.pieces[:, PAWN] & ~captured_bit,
            new.pieces[:, PAWN],
        )
        new.pieces[:, 6 + PAWN] = torch.where(
            ep_black,
            new.pieces[:, 6 + PAWN] & ~captured_bit,
            new.pieces[:, 6 + PAWN],
        )

        # Normal move or promotion. Promotion encoding 2..5 maps to internal
        # piece types 1..4 (N/B/R/Q).
        target_type = torch.where(promo == 0, moving, promo - 1)
        target_idx = color * 6 + target_type
        target_onehot = torch.nn.functional.one_hot(target_idx, num_classes=12).to(torch.bool)
        new.pieces = torch.where(
            target_onehot,
            new.pieces | tb[:, None],
            new.pieces,
        )

        # Castling: vectorized rook relocation.
        ck = self.castle_kind[actions]
        rook_from = torch.where(
            ck == 1, torch.full_like(ck, 7, dtype=torch.long),
            torch.where(
                ck == 2, torch.full_like(ck, 0, dtype=torch.long),
                torch.where(
                    ck == 3, torch.full_like(ck, 63, dtype=torch.long),
                    torch.where(ck == 4, torch.full_like(ck, 56, dtype=torch.long), torch.zeros_like(ck, dtype=torch.long)),
                ),
            ),
        )
        rook_to = torch.where(
            ck == 1, torch.full_like(ck, 5, dtype=torch.long),
            torch.where(
                ck == 2, torch.full_like(ck, 3, dtype=torch.long),
                torch.where(
                    ck == 3, torch.full_like(ck, 61, dtype=torch.long),
                    torch.where(ck == 4, torch.full_like(ck, 59, dtype=torch.long), torch.zeros_like(ck, dtype=torch.long)),
                ),
            ),
        )
        rfb = self.square_bits[rook_from]
        rtb = self.square_bits[rook_to]
        # Only a king move on a castling-geometry action is a real castle.
        # A rook/queen sharing that action id (e.g. Re1-g1) must not trigger
        # the rook relocation.
        castle = (ck > 0) & (moving == KING)
        new.pieces[:, ROOK] = torch.where(
            castle & (~turn),
            (new.pieces[:, ROOK] & ~rfb) | rtb,
            new.pieces[:, ROOK],
        )
        new.pieces[:, 6 + ROOK] = torch.where(
            castle & turn,
            (new.pieces[:, 6 + ROOK] & ~rfb) | rtb,
            new.pieces[:, 6 + ROOK],
        )

        # Castling rights update.
        rights = new.castling.clone()
        rights = torch.where((moving == KING) & (~turn), rights & ~(WK | WQ), rights)
        rights = torch.where((moving == KING) & turn, rights & ~(BK | BQ), rights)
        rights = torch.where((moving == ROOK) & (~turn) & (frm == 0), rights & ~WQ, rights)
        rights = torch.where((moving == ROOK) & (~turn) & (frm == 7), rights & ~WK, rights)
        rights = torch.where((moving == ROOK) & turn & (frm == 56), rights & ~BQ, rights)
        rights = torch.where((moving == ROOK) & turn & (frm == 63), rights & ~BK, rights)

        # A captured rook removes only the castling right associated with that
        # original home square.  In particular, capturing a promoted rook must
        # not erase the right belonging to an untouched a/h-file rook.
        captured_white_rook = (original[:, ROOK] & tb) != 0
        captured_black_rook = (original[:, 6 + ROOK] & tb) != 0
        rights = torch.where(
            captured_white_rook & (to == 0), rights & ~WQ, rights
        )
        rights = torch.where(
            captured_white_rook & (to == 7), rights & ~WK, rights
        )
        rights = torch.where(
            captured_black_rook & (to == 56), rights & ~BQ, rights
        )
        rights = torch.where(
            captured_black_rook & (to == 63), rights & ~BK, rights
        )
        new.castling = rights

        # En-passant target after a double pawn move.
        new.ep_square = torch.full_like(new.ep_square, -1)
        double = is_pawn & ((to - frm).abs() == 16)
        ep_new = (frm + to) // 2
        new.ep_square = torch.where(double, ep_new.to(torch.int16), new.ep_square)

        # Halfmove/fullmove clocks.
        capture = ((original & tb[:, None]) != 0).any(dim=1) | is_ep
        new.halfmove_clock = torch.where(
            is_pawn | capture,
            torch.zeros_like(new.halfmove_clock),
            new.halfmove_clock + 1,
        )
        new.fullmove_number = new.fullmove_number + turn.to(torch.int16)
        new.turn = ~new.turn
        return new

    def _validate_actions(self, state_idx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        candidate = self._apply_flat(state_idx, actions)
        # After the move, the opponent is to move. We need to test whether the
        # mover's king is attacked by the opponent.
        mover_color = self.turn[state_idx].long()
        mover_king_piece = mover_color * 6 + KING
        king_bb = candidate.pieces[torch.arange(actions.numel(), device=self.device), mover_king_piece]
        king_sq = (king_bb[:, None] == self.square_bits[None, :]).to(torch.long).argmax(dim=1)
        attacked = candidate._attacked(
            torch.arange(actions.numel(), device=self.device),
            king_sq,
            (~self.turn[state_idx]).long(),
        )
        return ~attacked

    def push_actions(self, actions: torch.Tensor) -> "GPUChess":
        """Apply one action per batch element; validates legality."""
        if actions.ndim != 1 or actions.shape[0] != self.pieces.shape[0]:
            raise ValueError("actions must have shape [batch]")
        legal = self.legal_move_mask()
        chosen = legal[torch.arange(actions.shape[0], device=self.device), actions]
        if not bool(chosen.all()):
            bad = torch.nonzero(~chosen, as_tuple=False).flatten().tolist()
            raise ValueError(f"Illegal GPU chess action at batch indices {bad}")
        return self._apply_flat(torch.arange(actions.shape[0], device=self.device), actions)

    def apply_actions_unchecked(self, state_idx: torch.Tensor, actions: torch.Tensor) -> "GPUChess":
        return self._apply_flat(state_idx, actions)

    # ------------------------------------------------------------------
    # Terminal / result helpers
    # ------------------------------------------------------------------
    def terminal_info(self) -> tuple[torch.Tensor, torch.Tensor]:
        legal = self.legal_move_mask()
        no_moves = ~legal.any(dim=1)
        in_check = self.is_in_check()
        checkmate = no_moves & in_check
        # 50-move rule is represented by the state even though threefold
        # repetition needs game history and is therefore tracked by self-play.
        fifty = self.halfmove_clock >= 100
        terminal = no_moves | fifty
        # Value from side-to-move perspective.
        value = torch.where(checkmate, torch.full_like(self.halfmove_clock, -1, dtype=torch.int16), torch.zeros_like(self.halfmove_clock))
        return terminal, value

    def insufficient_material(self) -> torch.Tensor:
        """FIDE dead-position test matching python-chess is_insufficient_material.

        Matches ``all(has_insufficient_material(c) for c in COLORS)``: a color
        without pawn/rook/queen is insufficient when it has no piece (bare
        king), a lone knight whose side has at most 2 pieces while the opponent
        has no knight or bishop, or only bishops when every bishop (both sides)
        sits on one square color and neither side has a knight.
        """
        p = self.pieces
        white_occ = p[:, 0].clone()
        for i in range(1, 6):
            white_occ |= p[:, i]
        black_occ = p[:, 6].clone()
        for i in range(7, 12):
            black_occ |= p[:, i]

        # Safe popcount of the 64-bit two's-complement pattern: bit k of
        # (x >> k) equals bit k of x regardless of arithmetic/logical shift.
        shifts = torch.arange(64, device=self.device, dtype=torch.int64)
        white_cnt = ((white_occ.unsqueeze(1) >> shifts) & 1).sum(dim=1)
        black_cnt = ((black_occ.unsqueeze(1) >> shifts) & 1).sum(dim=1)

        white_prq = (p[:, PAWN] | p[:, ROOK] | p[:, QUEEN]) != 0
        black_prq = (p[:, 6 + PAWN] | p[:, 6 + ROOK] | p[:, 6 + QUEEN]) != 0

        white_n = p[:, KNIGHT] != 0
        white_b = p[:, BISHOP] != 0
        black_n = p[:, 6 + KNIGHT] != 0
        black_b = p[:, 6 + BISHOP] != 0

        # All bishops on one square color: light = (rank + file) parity == 1.
        # Byte pattern by rank: rank0 AA, rank1 55, ... -> 0x55AA55AA55AA55AA.
        light = _signed_u64(0x55AA55AA55AA55AA)
        dark = _signed_u64(0xAA55AA55AA55AA55)
        bishops = p[:, BISHOP] | p[:, 6 + BISHOP]
        has_light = (bishops & light) != 0
        has_dark = (bishops & dark) != 0
        one_color = ~(has_light & has_dark)
        knights_any = white_n | black_n

        # python-chess order: knight branch wins if the color has any knight.
        white_ins = ~white_prq & torch.where(
            white_n,
            (white_cnt <= 2) & ~black_n & ~black_b,
            torch.where(
                white_b,
                one_color & ~knights_any,
                torch.ones_like(white_n),
            ),
        )
        black_ins = ~black_prq & torch.where(
            black_n,
            (black_cnt <= 2) & ~white_n & ~white_b,
            torch.where(
                black_b,
                one_color & ~knights_any,
                torch.ones_like(black_n),
            ),
        )
        return white_ins & black_ins

    def _legal_en_passant_available(self) -> torch.Tensor:
        """Whether an en-passant capture is legal in each current position.

        FIDE repetition identity retains en-passant state only when that right
        changes the legal moves.  This tests only candidate EP captures and the
        usual king-safety validation, rather than materializing every legal move.
        """
        b = self.pieces.shape[0]
        has_ep = self.ep_square >= 0
        if not bool(has_ep.any().item()):
            return torch.zeros((b,), dtype=torch.bool, device=self.device)

        pawn_bb = torch.where(self.turn, self.pieces[:, 6 + PAWN], self.pieces[:, PAWN])
        has_source = (pawn_bb[:, None] & self.from_bits[None, :]) != 0
        candidate_geometry = torch.where(
            (~self.turn)[:, None], self.w_capture[None, :], self.b_capture[None, :]
        )
        candidates = has_source & candidate_geometry
        candidates &= self.to_sq[None, :] == self.ep_square[:, None]

        # Verify that the pawn which may be captured actually exists behind the
        # EP target. Valid game states guarantee this, but retaining the check
        # makes the repetition key robust for manually constructed test states.
        captured_sq = self.ep_square.to(torch.long) + torch.where(
            self.turn,
            torch.full_like(self.ep_square, 8, dtype=torch.long),
            torch.full_like(self.ep_square, -8, dtype=torch.long),
        )
        captured_sq = captured_sq.clamp(0, 63)
        captured_bit = self.square_bits[captured_sq]
        enemy_pawns = torch.where(self.turn, self.pieces[:, PAWN], self.pieces[:, 6 + PAWN])
        candidates &= ((enemy_pawns & captured_bit) != 0)[:, None]

        flat = torch.nonzero(candidates, as_tuple=False)
        available = torch.zeros((b,), dtype=torch.bool, device=self.device)
        if flat.numel() == 0:
            return available
        legal = self._validate_actions(flat[:, 0], flat[:, 1])
        available[flat[:, 0][legal]] = True
        return available

    def state_hash(self) -> torch.Tensor:
        """Deterministic compact repetition key with legal EP semantics."""
        coeffs = [
            0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B,
            0x94D049BB133111EB, 0x369DEA0F31A53F85,
            0xD6E8FEB86659FD93, 0xA24BAED4963EE407,
            0x9FB21C651E98DF25, 0xC13FA9A902A6328F,
            0x165667B19E3779F9, 0x85EBCA77C2B2AE63,
            0x27D4EB2F165667C5, 0x2545F4914F6CDD1D,
        ]
        h = torch.zeros((self.pieces.shape[0],), dtype=torch.int64, device=self.device)
        for i, c in enumerate(coeffs):
            h = h ^ (self.pieces[:, i] * _signed_u64(c))
        h = h ^ (self.castling.to(torch.int64) * _signed_u64(0x517CC1B727220A95))
        # An unavailable EP target is ignored: it does not create a different
        # position for threefold-repetition purposes.
        legal_ep = self._legal_en_passant_available()
        ep_for_hash = torch.where(
            legal_ep,
            self.ep_square.to(torch.int64) + 1,
            torch.zeros_like(self.ep_square, dtype=torch.int64),
        )
        h = h ^ (ep_for_hash * _signed_u64(0x6A09E667F3BCC909))
        h = h ^ (self.turn.to(torch.int64) * _signed_u64(0xBB67AE8584CAA73B))
        return h
