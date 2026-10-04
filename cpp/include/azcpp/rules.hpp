// Chess rules core: bitboards, move generation, application, hashing,
// terminal detection, and model-input encoding.
//
// Semantics match python-chess (the trusted oracle) and are cross-checked by
// differential tests.  Action ids follow the project's canonical 4544 encoding:
//   0..4031    normal from/to (skip from == to)
//   4032..4287 white promotions  (from 48..55 to 56..63, N,B,R,Q)
//   4288..4543 black promotions  (from 8..15 to 0..7,  N,B,R,Q)
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace azcpp {

using U64 = std::uint64_t;

constexpr int kActionSpace = 4544;
constexpr int kNumPlanes = 12;
constexpr int kNumModelPlanes = 18;

enum PieceType : int { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };
enum Color : int { WHITE = 0, BLACK = 1 };

constexpr int kWK = 1, kWQ = 2, kBK = 4, kBQ = 8;

// ---------------------------------------------------------------------------
// Action encoding
// ---------------------------------------------------------------------------

inline int encode_action(int from, int to, int promo_code) {
  if (promo_code == 0) {
    return from * 63 + to - (to > from ? 1 : 0);
  }
  if (from >= 48 && from < 56 && to >= 56 && to < 64) {  // white promotion
    return 4032 + (from - 48) * 32 + (to - 56) * 4 + (promo_code - 2);
  }
  // black promotion
  return 4288 + (from - 8) * 32 + to * 4 + (promo_code - 2);
}

inline void decode_action(int action, int& from, int& to, int& promo_code) {
  if (action < 4032) {
    from = action / 63;
    const int rem = action - from * 63;
    to = rem < from ? rem : rem + 1;
    promo_code = 0;
  } else if (action < 4288) {
    const int base = action - 4032;
    from = 48 + base / 32;
    const int rem = base - (from - 48) * 32;
    to = 56 + rem / 4;
    promo_code = 2 + rem % 4;
  } else {
    const int base = action - 4288;
    from = 8 + base / 32;
    const int rem = base - (from - 8) * 32;
    to = rem / 4;
    promo_code = 2 + rem % 4;
  }
}

// ---------------------------------------------------------------------------
// Static attack tables
// ---------------------------------------------------------------------------

struct Tables {
  U64 knight[64];
  U64 king[64];
  U64 pawn[2][64];   // pawn[color][square]: squares attacked by a pawn there
  U64 rank_mask[8];
  U64 file_mask[8];
  // Sliding rays: dir order N,S,E,W,NE,NW,SE,SW. sq_rays[s][d][k] = square or -1.
  std::int8_t sq_rays[64][8][7];
  // Squares strictly between two squares on a line; 0 if not aligned.
  U64 between[64][64];

  Tables() {
    static const int knight_dr[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    static const int knight_df[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    static const int dirs[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int s = 0; s < 64; ++s) {
      const int r = s / 8, f = s % 8;
      knight[s] = 0;
      for (int i = 0; i < 8; ++i) {
        const int rr = r + knight_dr[i], ff = f + knight_df[i];
        if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) knight[s] |= 1ULL << (rr * 8 + ff);
      }
      king[s] = 0;
      for (int dr = -1; dr <= 1; ++dr)
        for (int df = -1; df <= 1; ++df) {
          if (!dr && !df) continue;
          const int rr = r + dr, ff = f + df;
          if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) king[s] |= 1ULL << (rr * 8 + ff);
        }
      for (int color = 0; color < 2; ++color) {
        pawn[color][s] = 0;
        const int dr = (color == WHITE) ? 1 : -1;
        for (int df = -1; df <= 1; df += 2) {
          const int rr = r + dr, ff = f + df;
          if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) pawn[color][s] |= 1ULL << (rr * 8 + ff);
        }
      }
      for (int d = 0; d < 8; ++d) {
        int rr = r + dirs[d][0], ff = f + dirs[d][1], k = 0;
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8 && k < 7) {
          sq_rays[s][d][k] = static_cast<std::int8_t>(rr * 8 + ff);
          rr += dirs[d][0]; ff += dirs[d][1]; ++k;
        }
        while (k < 7) { sq_rays[s][d][k] = -1; ++k; }
      }
    }
    for (int r = 0; r < 8; ++r) rank_mask[r] = 0xFFULL << (r * 8);
    for (int f = 0; f < 8; ++f) file_mask[f] = 0x0101010101010101ULL << f;

    std::memset(between, 0, sizeof(between));
    static const int dirs8[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int a = 0; a < 64; ++a) {
      for (int d = 0; d < 8; ++d) {
        int rr = a / 8 + dirs8[d][0], ff = a % 8 + dirs8[d][1];
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
          const int b = rr * 8 + ff;
          between[a][b] = 1ULL << b;  // provisional: first step only
          rr += dirs8[d][0]; ff += dirs8[d][1];
        }
      }
    }
    // Rebuild properly: accumulate all squares strictly between a and b.
    std::memset(between, 0, sizeof(between));
    for (int a = 0; a < 64; ++a) {
      for (int d = 0; d < 8; ++d) {
        int rr = a / 8 + dirs8[d][0], ff = a % 8 + dirs8[d][1];
        U64 mask = 0;
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
          const int b = rr * 8 + ff;
          between[a][b] = mask;
          mask |= 1ULL << b;
          rr += dirs8[d][0]; ff += dirs8[d][1];
        }
      }
    }
  }
};

inline const Tables& tables() {
  static const Tables t;
  return t;
}

inline int lsb_index(U64 bb) {
#if defined(_MSC_VER)
  unsigned long idx = 0;
  _BitScanForward64(&idx, bb);
  return static_cast<int>(idx);
#else
  return __builtin_ctzll(bb);
#endif
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

struct Position {
  U64 occ[12] = {0,0,0,0,0,0,0,0,0,0,0,0};  // [color * 6 + piece_type]
  std::int16_t turn = WHITE;
  std::int16_t castling = 0;
  std::int16_t ep = -1;
  std::int16_t halfmove = 0;
  std::int16_t fullmove = 1;

  U64 color_occ(int color) const {
    U64 bb = occ[color * 6];
    for (int i = 1; i < 6; ++i) bb |= occ[color * 6 + i];
    return bb;
  }
  U64 all_occ() const { return color_occ(WHITE) | color_occ(BLACK); }

  int king_sq(int color) const { return lsb_index(occ[color * 6 + KING]); }

  // Is `square` attacked by `by` color?
  bool attacked(int square, int by) const {
    const Tables& t = tables();
    const int r = square / 8, f = square % 8;

    // Pawn attacks: inverse relation from the target square.
    const int pawn_dir = (by == WHITE) ? -1 : 1;  // attacker sits this row offset
    const int pr = r + pawn_dir;
    if (pr >= 0 && pr < 8) {
      const U64 pawn_bb = occ[by * 6 + PAWN];
      if (f > 0 && (pawn_bb & (1ULL << (pr * 8 + f - 1)))) return true;
      if (f < 7 && (pawn_bb & (1ULL << (pr * 8 + f + 1)))) return true;
    }
    if (occ[by * 6 + KNIGHT] & t.knight[square]) return true;
    if (occ[by * 6 + KING] & t.king[square]) return true;

    static const int rook_dirs[4] = {0, 1, 2, 3};
    static const int bishop_dirs[4] = {4, 5, 6, 7};
    const U64 occ_all = all_occ();
    const U64 rq = occ[by * 6 + ROOK] | occ[by * 6 + QUEEN];
    const U64 bq = occ[by * 6 + BISHOP] | occ[by * 6 + QUEEN];

    for (int d : rook_dirs) {
      for (int k = 0; k < 7; ++k) {
        const std::int8_t sq = t.sq_rays[square][d][k];
        if (sq < 0) break;
        const U64 bit = 1ULL << sq;
        if (occ_all & bit) {
          if (rq & bit) return true;
          break;
        }
      }
    }
    for (int d : bishop_dirs) {
      for (int k = 0; k < 7; ++k) {
        const std::int8_t sq = t.sq_rays[square][d][k];
        if (sq < 0) break;
        const U64 bit = 1ULL << sq;
        if (occ_all & bit) {
          if (bq & bit) return true;
          break;
        }
      }
    }
    return false;
  }

  bool in_check() const {
    const int them = turn;  // side to move
    return attacked(king_sq(them), 1 - them);
  }

  // King of `color` attacked in a hypothetical position where `occ` is used.
  bool king_attacked_with(int color, U64 (&pieces)[12]) const {
    Position tmp;
    tmp.turn = turn;
    std::memcpy(tmp.occ, pieces, sizeof(tmp.occ));
    return tmp.attacked(tmp.king_sq(color), 1 - color);
  }

  // ------------------------------------------------------------------
  // Pseudo-legal generation: append action ids to `out` (unsorted).
  // ------------------------------------------------------------------
  void pseudo_legal(std::vector<int>& out) const {
    const Tables& t = tables();
    const int color = turn;
    const int enemy = 1 - color;
    const U64 own = color_occ(color);
    const U64 opp = color_occ(enemy);
    const U64 occ_all = own | opp;
    const U64 enemy_king = occ[enemy * 6 + KING];

    auto add_moves = [&](U64 targets, int from) {
      while (targets) {
        const int to = lsb_index(targets);
        targets &= targets - 1;
        if (enemy_king & (1ULL << to)) continue;  // never capture the king
        out.push_back(encode_action(from, to, 0));
      }
    };

    // Knights, bishops, rooks, queen, king (non-castling).
    for (int pt = KNIGHT; pt <= KING; ++pt) {
      U64 bb = occ[color * 6 + pt];
      while (bb) {
        const int from = lsb_index(bb);
        bb &= bb - 1;
        U64 targets = 0;
        switch (pt) {
          case KNIGHT: targets = t.knight[from] & ~own; break;
          case BISHOP: targets = slide(from, 4, occ_all) & ~own; break;
          case ROOK:   targets = slide(from, 0, occ_all) & ~own; break;
          case QUEEN:  targets = (slide(from, 0, occ_all) | slide(from, 4, occ_all)) & ~own; break;
          case KING:   targets = t.king[from] & ~own; break;
        }
        add_moves(targets, from);
      }
    }

    // Pawns.
    U64 pawns = occ[color * 6 + PAWN];
    const int push = (color == WHITE) ? 8 : -8;
    const int start_rank = (color == WHITE) ? 1 : 6;
    const int promo_rank = (color == WHITE) ? 6 : 1;  // rank index from which promotion happens
    while (pawns) {
      const int from = lsb_index(pawns);
      pawns &= pawns - 1;
      const int fr = from / 8;

      const int single = from + push;
      const bool single_ok = (single >= 0 && single < 64) &&
                             !(occ_all & (1ULL << single));
      const bool can_promo = (fr == promo_rank);

      if (can_promo) {
        if (single_ok) add_promotions(from, single, out);
        // Diagonal captures with promotion.
        for (int df = -1; df <= 1; df += 2) {
          const int ff = from % 8 + df;
          if (ff < 0 || ff > 7) continue;
          const int to = from + push + df;
          if (to < 0 || to > 63) continue;
          if ((opp & (1ULL << to)) && !(enemy_king & (1ULL << to))) {
            add_promotions(from, to, out);
          }
        }
      } else {
        if (single_ok) out.push_back(encode_action(from, single, 0));
        // Double push.
        if (fr == start_rank && single_ok &&
            !(occ_all & (1ULL << (from + 2 * push)))) {
          out.push_back(encode_action(from, from + 2 * push, 0));
        }
        // Captures (including en-passant target square).
        for (int df = -1; df <= 1; df += 2) {
          const int ff = from % 8 + df;
          if (ff < 0 || ff > 7) continue;
          const int to = from + push + df;
          if (to < 0 || to > 63) continue;
          const U64 to_bit = 1ULL << to;
          const bool is_ep = (to == ep);
          if (((opp & to_bit) && !(enemy_king & to_bit)) || is_ep) {
            out.push_back(encode_action(from, to, 0));
          }
        }
      }
    }

    // Castling.
    add_castling(out);
  }

  // Sliding attack ray from `from` along directions starting at `dir0`
  // (dir0, dir0+1, ... dir0+3): 0=N,1=S,2=E,3=W (rook); 4..7 bishop.
  U64 slide(int from, int dir0, U64 occ_all) const {
    const Tables& t = tables();
    U64 bb = 0;
    for (int d = dir0; d < dir0 + 4; ++d) {
      for (int k = 0; k < 7; ++k) {
        const std::int8_t sq = t.sq_rays[from][d][k];
        if (sq < 0) break;
        const U64 bit = 1ULL << sq;
        bb |= bit;
        if (occ_all & bit) break;
      }
    }
    return bb;
  }

  static void add_promotions(int from, int to, std::vector<int>& out) {
    for (int p = 2; p <= 5; ++p) out.push_back(encode_action(from, to, p));
  }

  void add_castling(std::vector<int>& out) const {
    const U64 occ_all = all_occ();
    const int color = turn;
    const int enemy = 1 - color;

    auto clear = [&](int king_sq_expected, int rook_sq, int right, int king_to) {
      if (!(castling & right)) return;
      if (king_sq(color) != king_sq_expected) return;
      if (!(occ[color * 6 + ROOK] & (1ULL << rook_sq))) return;
      const int transit = (king_to + king_sq_expected) / 2;
      // Empty squares between king and rook (exclusive of both).
      const int lo = king_sq_expected < rook_sq ? king_sq_expected : rook_sq;
      const int hi = king_sq_expected < rook_sq ? rook_sq : king_sq_expected;
      for (int s = lo + 1; s < hi; ++s) {
        if (occ_all & (1ULL << s)) return;
      }
      if (attacked(king_sq_expected, enemy)) return;
      if (attacked(transit, enemy)) return;
      if (attacked(king_to, enemy)) return;
      out.push_back(encode_action(king_sq_expected, king_to, 0));
    };

    if (color == WHITE) {
      clear(4, 7, kWK, 6);   // e1 -> g1
      clear(4, 0, kWQ, 2);   // e1 -> c1
    } else {
      clear(60, 63, kBK, 62);  // e8 -> g8
      clear(60, 56, kBQ, 58);  // e8 -> c8
    }
  }

  // ------------------------------------------------------------------
  // Legal moves
  // ------------------------------------------------------------------
  void legal_actions(std::vector<int>& out) const {
    out.clear();
    std::vector<int> pseudo;
    pseudo_legal(pseudo);
    const int color = turn;
    for (int action : pseudo) {
      Position child = *this;
      child.apply(action);
      if (!child.attacked(child.king_sq(color), 1 - color)) out.push_back(action);
    }
  }

  int legal_action_count() const {
    std::vector<int> tmp;
    legal_actions(tmp);
    return static_cast<int>(tmp.size());
  }

  // ------------------------------------------------------------------
  // Apply (must be a pseudo-legal action; validates nothing).
  // ------------------------------------------------------------------
  void apply(int action) {
    int from, to, promo;
    decode_action(action, from, to, promo);
    const int color = turn;
    const int enemy = 1 - color;

    // Identify the moving piece.
    int moving = -1;
    const U64 from_bit = 1ULL << from;
    for (int pt = 0; pt < 6; ++pt) {
      if (occ[color * 6 + pt] & from_bit) { moving = pt; break; }
    }

    // Detect en-passant before clearing (pawn diagonal onto ep square).
    const bool is_ep = (moving == PAWN && (from % 8) != (to % 8) && to == ep);

    // Capture detection (piece standing on the target square).
    int captured_pt = -1;
    const U64 to_bit = 1ULL << to;
    for (int pt = 0; pt < 6; ++pt) {
      if (occ[enemy * 6 + pt] & to_bit) { captured_pt = pt; break; }
    }

    // Move the piece (promotion replaces the type).
    occ[color * 6 + moving] &= ~from_bit;
    for (int i = 0; i < 12; ++i) occ[i] &= ~to_bit;
    const int placed = (promo != 0) ? (promo - 1) : moving;
    occ[color * 6 + placed] |= to_bit;

    // En-passant: remove the pawn behind the target square.
    if (is_ep) {
      const int cap_sq = to + ((color == WHITE) ? -8 : 8);
      if (cap_sq >= 0 && cap_sq < 64) {
        occ[enemy * 6 + PAWN] &= ~(1ULL << cap_sq);
      }
    }

    // Castling: a king moving on a castling-geometry action relocates the rook.
    const int castle_kind = castle_kind_of(from, to);
    if (castle_kind != 0 && moving == KING) {
      int rook_from = -1, rook_to = -1;
      switch (castle_kind) {
        case 1: rook_from = 7;  rook_to = 5;  break;  // white O-O
        case 2: rook_from = 0;  rook_to = 3;  break;  // white O-O-O
        case 3: rook_from = 63; rook_to = 61; break;  // black O-O
        case 4: rook_from = 56; rook_to = 59; break;  // black O-O-O
      }
      const U64 rf = 1ULL << rook_from, rt = 1ULL << rook_to;
      if (occ[color * 6 + ROOK] & rf) {
        occ[color * 6 + ROOK] &= ~rf;
        occ[color * 6 + ROOK] |= rt;
      }
    }

    // Castling rights.
    int rights = castling;
    if (moving == KING) {
      if (color == WHITE) rights &= ~(kWK | kWQ);
      else rights &= ~(kBK | kBQ);
    }
    if (moving == ROOK) {
      if (color == WHITE) {
        if (from == 0) rights &= ~kWQ;
        if (from == 7) rights &= ~kWK;
      } else {
        if (from == 56) rights &= ~kBQ;
        if (from == 63) rights &= ~kBK;
      }
    }
    if (captured_pt == ROOK) {  // captured rook on its home square
      if (to == 0)  rights &= ~kWQ;
      if (to == 7)  rights &= ~kWK;
      if (to == 56) rights &= ~kBQ;
      if (to == 63) rights &= ~kBK;
    }
    castling = static_cast<std::int16_t>(rights);

    // En-passant target.
    ep = -1;
    if (moving == PAWN && (to - from == 16 || to - from == -16)) {
      ep = static_cast<std::int16_t>((from + to) / 2);
    }

    // Clocks.
    const bool capture = (captured_pt >= 0) || is_ep;
    if (moving == PAWN || capture) halfmove = 0;
    else halfmove = static_cast<std::int16_t>(halfmove + 1);
    if (color == BLACK) fullmove = static_cast<std::int16_t>(fullmove + 1);

    turn = static_cast<std::int16_t>(enemy);
  }

  static int castle_kind_of(int from, int to) {
    if (from == 4 && to == 6) return 1;
    if (from == 4 && to == 2) return 2;
    if (from == 60 && to == 62) return 3;
    if (from == 60 && to == 58) return 4;
    return 0;
  }

  // ------------------------------------------------------------------
  // Terminal detection
  // ------------------------------------------------------------------
  bool insufficient_material() const {
    // Mirrors python-chess: all(has_insufficient_material(c) for c in COLORS).
    const U64 white = color_occ(WHITE);
    const U64 black = color_occ(BLACK);
    const int white_cnt = popcount(white);
    const int black_cnt = popcount(black);

    const bool white_prq = (occ[0 + PAWN] | occ[0 + ROOK] | occ[0 + QUEEN]) != 0;
    const bool black_prq = (occ[6 + PAWN] | occ[6 + ROOK] | occ[6 + QUEEN]) != 0;

    const bool white_n = occ[0 + KNIGHT] != 0;
    const bool white_b = occ[0 + BISHOP] != 0;
    const bool black_n = occ[6 + KNIGHT] != 0;
    const bool black_b = occ[6 + BISHOP] != 0;

    // All bishops on one square color: light = (rank + file) parity == 1.
    static const U64 kLight = 0x55AA55AA55AA55AAULL;
    static const U64 kDark = 0xAA55AA55AA55AA55ULL;
    const U64 bishops = occ[0 + BISHOP] | occ[6 + BISHOP];
    const bool one_color = !(((bishops & kLight) != 0) && ((bishops & kDark) != 0));
    const bool knights_any = white_n || black_n;

    auto insufficient = [&](bool has_prq, bool has_n, bool has_b, int own_cnt,
                            bool opp_n, bool opp_b) {
      if (has_prq) return false;
      if (has_n) return own_cnt <= 2 && !opp_n && !opp_b;
      if (has_b) return one_color && !knights_any;
      return true;  // bare king
    };
    return insufficient(white_prq, white_n, white_b, white_cnt, black_n, black_b) &&
           insufficient(black_prq, black_n, black_b, black_cnt, white_n, white_b);
  }

  // (terminal, value-from-side-to-move): mirrors GPUChess.terminal_info.
  // Threefold repetition is a game-history concern handled by the caller.
  void terminal_info(bool& terminal, int& value) const {
    const int count = legal_action_count();
    const bool in_chk = in_check();
    const bool mate = count == 0 && in_chk;
    terminal = (count == 0) || halfmove >= 100;
    value = mate ? -1 : 0;
  }

  // Value for a leaf position: -1 checkmate, else 0 (stalemate / draws).
  float leaf_terminal_value() const {
    const int count = legal_action_count();
    if (count == 0 && in_check()) return -1.0f;
    return 0.0f;
  }

  // ------------------------------------------------------------------
  // Repetition hash: bit-identical to GPUChess.state_hash.
  // ------------------------------------------------------------------
  U64 state_hash() const {
    static const U64 kCoeffs[12] = {
        0x9E3779B97F4A7C15ULL, 0x0BF58476D1CE4E5BULL, 0x94D049BB133111EBULL,
        0x369DEA0F31A53F85ULL, 0xD6E8FEB86659FD93ULL, 0xA24BAED4963EE407ULL,
        0x9FB21C651E98DF25ULL, 0xC13FA9A902A6328FULL, 0x165667B19E3779F9ULL,
        0x85EBCA77C2B2AE63ULL, 0x27D4EB2F165667C5ULL, 0x2545F4914F6CDD1DULL};
    U64 h = 0;
    for (int i = 0; i < 12; ++i) h ^= occ[i] * kCoeffs[i];
    h ^= static_cast<U64>(static_cast<std::uint16_t>(castling)) * 0x517CC1B727220A95ULL;
    const int ep_val = legal_en_passant_available() ? (static_cast<int>(ep) + 1) : 0;
    h ^= static_cast<U64>(static_cast<std::uint16_t>(ep_val)) * 0x6A09E667F3BCC909ULL;
    h ^= static_cast<U64>(static_cast<std::uint16_t>(turn)) * 0xBB67AE8584CAA73BULL;
    return h;
  }

  // Whether a legal en-passant capture exists (GPUChess semantics).
  bool legal_en_passant_available() const {
    if (ep < 0) return false;
    const int color = turn;
    const int enemy = 1 - color;
    const int ep_sq = ep;

    // Enemy pawn that just double-pushed sits behind the ep square.
    const int behind = ep_sq + ((color == WHITE) ? -8 : 8);
    if (behind < 0 || behind > 63) return false;
    if (!(occ[enemy * 6 + PAWN] & (1ULL << behind))) return false;

    // A side pawn must be able to capture onto the ep square diagonally.
    const int dir = (color == WHITE) ? 1 : -1;  // attacker relative row
    const int er = ep_sq / 8, ef = ep_sq % 8;
    int from_sq = -1;
    for (int df = -1; df <= 1; df += 2) {
      const int rr = er - dir, ff = ef + df;
      if (rr < 0 || rr > 7 || ff < 0 || ff > 7) continue;
      const int sq = rr * 8 + ff;
      if (occ[color * 6 + PAWN] & (1ULL << sq)) { from_sq = sq; break; }
    }
    if (from_sq < 0) return false;

    // The capture must be legal (king safety after the capture).
    Position child = *this;
    child.apply(encode_action(from_sq, ep_sq, 0));
    return !child.attacked(child.king_sq(color), enemy);
  }

  // ------------------------------------------------------------------
  // Model input: 18x8x8 float32 planes, identical to GPUChess.to_model_input.
  // ------------------------------------------------------------------
  void encode_planes(float* out) const {
    std::memset(out, 0, sizeof(float) * kNumModelPlanes * 8 * 8);
    for (int p = 0; p < 12; ++p) {
      U64 bb = occ[p];
      while (bb) {
        const int s = lsb_index(bb);
        bb &= bb - 1;
        const int row = 7 - (s / 8);   // model row = 7 - rank
        const int col = s % 8;
        out[p * 64 + row * 8 + col] = 1.0f;
      }
    }
    const bool white_to_move = (turn == WHITE);
    const float stm = white_to_move ? 1.0f : 0.0f;
    for (int i = 0; i < 64; ++i) out[12 * 64 + i] = stm;
    const int rights[4] = {kWK, kWQ, kBK, kBQ};
    for (int r = 0; r < 4; ++r) {
      const float v = (castling & rights[r]) ? 1.0f : 0.0f;
      if (v != 0.0f) {
        for (int i = 0; i < 64; ++i) out[(13 + r) * 64 + i] = v;
      }
    }
    if (ep >= 0) {
      const int row = 7 - (ep / 8);
      const int col = ep % 8;
      out[17 * 64 + row * 8 + col] = 1.0f;
    }
  }

  // ------------------------------------------------------------------
  // FEN
  // ------------------------------------------------------------------
  static Position from_fen(const std::string& fen) {
    Position p;
    p.occ[0] = 0;
    std::memset(p.occ, 0, sizeof(p.occ));
    int sq = 56;  // start at a8
    size_t i = 0;
    for (; i < fen.size() && fen[i] != ' '; ++i) {
      const char c = fen[i];
      if (c == '/') { sq -= 16; continue; }
      if (c >= '1' && c <= '8') { sq += c - '0'; continue; }
      int color = (c >= 'a' && c <= 'z') ? BLACK : WHITE;
      int pt;
      switch (std::tolower(c)) {
        case 'p': pt = PAWN; break;
        case 'n': pt = KNIGHT; break;
        case 'b': pt = BISHOP; break;
        case 'r': pt = ROOK; break;
        case 'q': pt = QUEEN; break;
        case 'k': pt = KING; break;
        default: pt = PAWN; break;
      }
      p.occ[color * 6 + pt] |= 1ULL << sq;
      ++sq;
    }
    // Side to move.
    while (i < fen.size() && fen[i] == ' ') ++i;
    if (i < fen.size()) { p.turn = (fen[i] == 'b') ? BLACK : WHITE; ++i; }
    // Castling.
    while (i < fen.size() && fen[i] == ' ') ++i;
    int rights = 0;
    for (; i < fen.size() && fen[i] != ' '; ++i) {
      switch (fen[i]) {
        case 'K': rights |= kWK; break;
        case 'Q': rights |= kWQ; break;
        case 'k': rights |= kBK; break;
        case 'q': rights |= kBQ; break;
      }
    }
    p.castling = static_cast<std::int16_t>(rights);
    // En-passant.
    while (i < fen.size() && fen[i] == ' ') ++i;
    if (i < fen.size() && fen[i] != '-') {
      const int f = fen[i] - 'a';
      const int r = fen[i + 1] - '1';
      p.ep = static_cast<std::int16_t>(r * 8 + f);
      i += 2;
    } else if (i < fen.size()) {
      ++i;
    }
    // Clocks.
    while (i < fen.size() && fen[i] == ' ') ++i;
    int half = 0;
    while (i < fen.size() && fen[i] >= '0' && fen[i] <= '9') {
      half = half * 10 + (fen[i] - '0'); ++i;
    }
    p.halfmove = static_cast<std::int16_t>(half);
    while (i < fen.size() && fen[i] == ' ') ++i;
    int full = 0;
    bool have_full = false;
    while (i < fen.size() && fen[i] >= '0' && fen[i] <= '9') {
      full = full * 10 + (fen[i] - '0');
      have_full = true;
      ++i;
    }
    if (!have_full) full = 1;
    p.fullmove = static_cast<std::int16_t>(full);
    return p;
  }

  std::string to_fen() const {
    std::string fen;
    for (int r = 7; r >= 0; --r) {
      int empty = 0;
      for (int f = 0; f < 8; ++f) {
        const int sq = r * 8 + f;
        const U64 bit = 1ULL << sq;
        int piece = -1;
        for (int i = 0; i < 12; ++i) {
          if (occ[i] & bit) { piece = i; break; }
        }
        if (piece < 0) { ++empty; continue; }
        if (empty) { fen += static_cast<char>('0' + empty); empty = 0; }
        static const char kChars[12] = {'P','N','B','R','Q','K','p','n','b','r','q','k'};
        fen += kChars[piece];
      }
      if (empty) fen += static_cast<char>('0' + empty);
      if (r) fen += '/';
    }
    fen += turn == WHITE ? " w " : " b ";
    if (castling == 0) fen += '-';
    else {
      if (castling & kWK) fen += 'K';
      if (castling & kWQ) fen += 'Q';
      if (castling & kBK) fen += 'k';
      if (castling & kBQ) fen += 'q';
    }
    fen += ' ';
    if (ep >= 0) {
      fen += static_cast<char>('a' + ep % 8);
      fen += static_cast<char>('1' + ep / 8);
    } else {
      fen += '-';
    }
    fen += ' ';
    fen += std::to_string(halfmove);
    fen += ' ';
    fen += std::to_string(fullmove);
    return fen;
  }

  static int popcount(U64 bb) {
    int n = 0;
    while (bb) { bb &= bb - 1; ++n; }
    return n;
  }
};

}  // namespace azcpp
