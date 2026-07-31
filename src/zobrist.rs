//! Zobrist hashing: the five-key family's tables and folding helpers
//! (design/04-position-state.md §3.4, D9).
//!
//! One const-evaluated table set, [`ZOBRIST`], feeds every key the engine
//! maintains incrementally: the main position `key`, `pawn_key`,
//! `material_key`, `minor_key`, and the per-color `non_pawn_key`s. All of
//! Stockfish's hashing tricks are wired from day one because they are
//! near-impossible to retrofit (D9):
//!
//! - **Zeroed pawn promotion ranks** — pawns can never stand on their
//!   promotion rank, so a promotion XORs only the new piece at `to`.
//! - **Castling keyed by the whole 4-bit mask** — any rights change is two
//!   XORs, never per-right bookkeeping.
//! - **Material key via count slots** — no separate table; piece counts
//!   reuse psq square indices `8 + count` ([`Tables::material`]).
//! - **Rule50 folding** — the TT-facing key is perturbed as the halfmove
//!   clock nears the 50-move rule ([`fold_rule50`]), mitigating
//!   graph-history interaction; repetition detection compares raw keys.
//!
//! Key *deltas* live in `make_move` (SF-style), NOT in
//! `put_piece`/`remove_piece` — the five-key family updates differently per
//! piece type and move kind, and the ep/castling/side keys never route
//! through placement at all (doc 04 §3.3, §9.1).

use crate::attacks::Prng;
use crate::types::{Piece, PieceType, Square};

pub use crate::types::Key;

/// Stockfish's seed for the Zobrist table stream (a citable fact, D20),
/// driving the shared xorshift64* generator (doc 03 §3.6). Our draw ORDER
/// differs from SF's, so our keys don't equal SF's — nothing requires that.
const ZOBRIST_SEED: u64 = 1_070_372;

/// The five Zobrist key families (design/04 §3.4).
///
/// Fields are public — call sites in `make_move` are dense XOR chains and
/// index the tables directly; the one non-obvious access pattern (material
/// count slots) gets the [`Tables::material`] helper.
pub struct Tables {
    /// Piece-on-square keys: `psq[Piece::index()][sq.index()]`.
    ///
    /// Two deliberate quirks (doc 04 §3.4 tricks 1 and 3):
    /// - the pawn rows are ZERO on their promotion ranks
    ///   (`psq[white pawn][56..64]`, `psq[black pawn][0..8]`), so promotion
    ///   key updates need no pawn-clear XOR at `to`;
    /// - square indices `8 + count` double as material-key count slots
    ///   ([`Tables::material`]) — separate hash spaces never mix, so reusing
    ///   real square keys is fine.
    pub psq: [[Key; Square::COUNT]; Piece::COUNT],
    /// By FILE of the en-passant square; XORed in only when a legal ep
    /// capture actually exists (D9).
    pub en_passant: [Key; 8],
    /// By the WHOLE 4-bit rights mask
    /// ([`CastlingRights::bits`](crate::board::CastlingRights::bits)): any
    /// rights change is `castling[old] ^ castling[new]`. `castling[0]` is a
    /// real random key, not zero — harmless and simpler.
    pub castling: [Key; 16],
    /// XORed in when Black is to move.
    pub side: Key,
    /// `pawn_key` baseline for a pawnless board, so "no pawns at all"
    /// hashes distinctly (pawn correction history depends on it later).
    pub no_pawns: Key,
}

impl Tables {
    /// Material-key count slot: the key for "one more piece of kind `pc`",
    /// reusing `psq[pc][8 + count]` — no separate table (doc 04 §3.4
    /// trick 3).
    ///
    /// Invariant: `material_key` is the XOR over all pieces `pc`, for `i`
    /// in `0..count(pc)`, of `material(pc, i)`. Adding a piece whose
    /// PRE-add count is `c` XORs `material(pc, c)`; removing one whose
    /// POST-removal count is `c` XORs the same slot.
    ///
    /// Live slots run to `count == 9`: promotions legally allow 10
    /// knights/bishops/rooks (or 9 queens) of one color; pawns stop at
    /// board count 8, i.e. slot 7. The debug bound below is deliberately
    /// exactly that — asserting `count <= 8` would pass every perft-suite
    /// position yet panic in legal promotion-heavy games.
    #[inline]
    pub const fn material(&self, pc: Piece, count: usize) -> Key {
        debug_assert!(
            count
                <= match pc.piece_type {
                    PieceType::Pawn => 7,
                    _ => 9,
                }
        );
        // Perf study 56 mechanism B: the outer index folds (`Piece::index` is
        // enum arithmetic, provably `< 12`), but `8 + count` is an unbounded
        // `usize` as far as the optimizer is concerned, so this one line was
        // the second-largest bounds-check site in the shipped binary — NINE
        // machine edges, all inside `make_move`/`do_move`. The debug_assert
        // above is the real invariant (`count <= 9`); `& 63` states it where
        // codegen can use it and is the identity for every count the assert
        // admits.
        self.psq[pc.index()][(8 + count) & (Square::COUNT - 1)]
    }
}

/// The one table set, const-evaluated at compile time. A plain `static` —
/// unlike the magics there is no runtime search here, so no `LazyLock`.
pub static ZOBRIST: Tables = build_tables();

/// Fill order (FIXED — the determinism pins in the tests notice any change,
/// and changing it invalidates every stored key): `psq` row by row in
/// `Piece::index()` order (White P,N,B,R,Q,K then Black P..K), each row
/// square a1..h8; then `en_passant[0..8]`; then `castling[0..16]`; then
/// `side`; then `no_pawns`. The promotion-rank zeroing overwrites AFTER all
/// draws, so it does not shift the stream (doc 04 §3.4).
const fn build_tables() -> Tables {
    let mut rng = Prng::new(ZOBRIST_SEED);

    let mut psq = [[0; Square::COUNT]; Piece::COUNT];
    let mut pc = 0;
    while pc < Piece::COUNT {
        let mut sq = 0;
        while sq < Square::COUNT {
            psq[pc][sq] = rng.rand64();
            sq += 1;
        }
        pc += 1;
    }

    let mut en_passant = [0; 8];
    let mut file = 0;
    while file < 8 {
        en_passant[file] = rng.rand64();
        file += 1;
    }

    let mut castling = [0; 16];
    let mut mask = 0;
    while mask < 16 {
        castling[mask] = rng.rand64();
        mask += 1;
    }

    let side = rng.rand64();
    let no_pawns = rng.rand64();

    // Trick 1 (D9): zero the pawn rows on the promotion ranks. White pawn
    // row is Piece::index() 0, rank 8 is squares 56..64; Black pawn row is
    // 6, rank 1 is squares 0..8.
    const WHITE_PAWN: usize = Piece::new(crate::types::Color::White, PieceType::Pawn).index();
    const BLACK_PAWN: usize = Piece::new(crate::types::Color::Black, PieceType::Pawn).index();
    let mut sq = Square::A8.index() as usize;
    while sq < Square::COUNT {
        psq[WHITE_PAWN][sq] = 0;
        sq += 1;
    }
    let mut sq = 0;
    while sq < Square::A2.index() as usize {
        psq[BLACK_PAWN][sq] = 0;
        sq += 1;
    }

    Tables {
        psq,
        en_passant,
        castling,
        side,
        no_pawns,
    }
}

/// SF's `make_key`: cheap avalanche of a small integer into key space
/// (multiplier/addend are SF-current facts, D20). Backs the rule50 folding
/// below.
pub const fn make_key(seed: u64) -> Key {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Rule50 folding (graph-history mitigation, D9): perturb the raw key once
/// the halfmove clock nears the 50-move rule, in 8-step buckets starting at
/// 14 — clocks 0..=13 hash raw, 14..=21 XOR `make_key(0)`, 22..=29
/// `make_key(1)`, and so on.
///
/// `Board::key()` (the TT probe key) delegates here as `adjust_key50`;
/// repetition detection compares RAW keys — folding applies only to the
/// exposed key (doc 04 §3.4).
pub const fn fold_rule50(k: Key, rule50: i32) -> Key {
    if rule50 < 14 {
        k
    } else {
        k ^ make_key(((rule50 - 14) / 8) as u64)
    }
}

/// After-move folding variant, threshold 13:
/// `fold_rule50_after(k, r) == fold_rule50(k, r + 1)` — models "parent
/// rule50 + 1", valid only for quiet non-pawn moves.
///
/// Exists EXCLUSIVELY for the speculative `prefetch_key` prediction (doc 04
/// §3.10); inside `make_move` the child clock is already exact, so hook A
/// folds with [`fold_rule50`] (doc 04 §3.8 step 9).
pub const fn fold_rule50_after(k: Key, rule50: i32) -> Key {
    if rule50 < 13 {
        k
    } else {
        k ^ make_key(((rule50 - 13) / 8) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Color;

    const WHITE_PAWN_ROW: usize = Piece::new(Color::White, PieceType::Pawn).index();
    const BLACK_PAWN_ROW: usize = Piece::new(Color::Black, PieceType::Pawn).index();

    // Doc 04 §4 step 2: determinism across calls — rebuilding produces the
    // exact same tables the static was const-evaluated to.
    #[test]
    fn tables_deterministic_across_builds() {
        let rebuilt = build_tables();
        assert_eq!(rebuilt.psq, ZOBRIST.psq);
        assert_eq!(rebuilt.en_passant, ZOBRIST.en_passant);
        assert_eq!(rebuilt.castling, ZOBRIST.castling);
        assert_eq!(rebuilt.side, ZOBRIST.side);
        assert_eq!(rebuilt.no_pawns, ZOBRIST.no_pawns);
    }

    // Regression pins, verified against an independent Python model of the
    // xorshift64* stream (seed 1070372, doc 04 §3.4 fill order). These
    // change ONLY if the generator, seed, or fill order changes — all of
    // which invalidate every stored key and must be deliberate.
    #[test]
    fn determinism_pins() {
        assert_eq!(ZOBRIST.psq[0][Square::A1.idx()], 0x083610FB1CD7C6A5);
        assert_eq!(ZOBRIST.psq[0][Square::A2.idx()], 0x3F00957BF619FABD);
        assert_eq!(ZOBRIST.psq[11][Square::H8.idx()], 0x2FE9A4D5AA8D43F6);
        assert_eq!(ZOBRIST.en_passant[0], 0x7D56D658294A9988);
        assert_eq!(ZOBRIST.en_passant[7], 0x586CDAC18FC14DF7);
        assert_eq!(ZOBRIST.castling[0], 0x67E85E44A0C80F99);
        assert_eq!(ZOBRIST.castling[15], 0x6CEF81B4350535C6);
        assert_eq!(ZOBRIST.side, 0xA0F520A4C9FA5BCC);
        assert_eq!(ZOBRIST.no_pawns, 0xA3626919C9115B2A);
    }

    // Doc 04 §4 step 2: the zeroed slots are EXACTLY psq[wp][56..64] and
    // psq[bp][0..8] — no more (a stray zero elsewhere would silently drop a
    // piece from the hash), no fewer.
    #[test]
    fn promotion_rank_pawn_slots_zeroed_exactly() {
        for (row, row_keys) in ZOBRIST.psq.iter().enumerate() {
            for (sq, &key) in row_keys.iter().enumerate() {
                let should_be_zero =
                    (row == WHITE_PAWN_ROW && sq >= 56) || (row == BLACK_PAWN_ROW && sq < 8);
                assert_eq!(key == 0, should_be_zero, "psq[{row}][{sq}] = {key:#x}");
            }
        }
    }

    // Doc 04 §4 step 2 (strengthened from "spot-check"): every live key —
    // all non-zeroed psq entries, all 8 ep files, all 16 castling masks
    // (including castling[0]), side, no_pawns — is nonzero and pairwise
    // distinct.
    #[test]
    fn live_keys_nonzero_and_pairwise_distinct() {
        let mut keys: Vec<Key> = Vec::new();
        for (row, row_keys) in ZOBRIST.psq.iter().enumerate() {
            for (sq, &key) in row_keys.iter().enumerate() {
                let zeroed =
                    (row == WHITE_PAWN_ROW && sq >= 56) || (row == BLACK_PAWN_ROW && sq < 8);
                if !zeroed {
                    keys.push(key);
                }
            }
        }
        keys.extend_from_slice(&ZOBRIST.en_passant);
        keys.extend_from_slice(&ZOBRIST.castling);
        keys.push(ZOBRIST.side);
        keys.push(ZOBRIST.no_pawns);

        assert_eq!(keys.len(), 12 * 64 - 16 + 8 + 16 + 2);
        assert!(keys.iter().all(|&k| k != 0), "live keys must be nonzero");
        keys.sort_unstable();
        assert!(
            keys.windows(2).all(|w| w[0] != w[1]),
            "live keys must be pairwise distinct"
        );
    }

    // Doc 04 §3.4 trick 3: count slots are psq[pc][8 + count], live through
    // count 9 for non-pawns and count 7 for pawns, and never collide with
    // the zeroed promotion-rank region.
    #[test]
    fn material_count_slots() {
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let pc = Piece::new(color, piece_type);
                let max_count = if piece_type == PieceType::Pawn { 7 } else { 9 };
                for count in 0..=max_count {
                    let key = ZOBRIST.material(pc, count);
                    assert_eq!(key, ZOBRIST.psq[pc.index()][8 + count]);
                    assert_ne!(key, 0, "material slot {count} of {pc:?} is zero");
                }
            }
        }
    }

    // Doc 04 §4 step 2: fold bucket boundaries — 13 -> raw, 14 and 21 in
    // the same bucket, 22 in the next; and the after-move variant is
    // exactly "fold at parent clock + 1".
    #[test]
    fn rule50_fold_buckets() {
        let k: Key = 0x9E37_79B9_7F4A_7C15;
        for r in 0..=13 {
            assert_eq!(fold_rule50(k, r), k, "clock {r} must hash raw");
        }
        assert_eq!(fold_rule50(k, 14), k ^ make_key(0));
        assert_ne!(fold_rule50(k, 14), k);
        assert_eq!(fold_rule50(k, 14), fold_rule50(k, 21));
        assert_eq!(fold_rule50(k, 22), k ^ make_key(1));
        assert_ne!(fold_rule50(k, 22), fold_rule50(k, 21));

        for r in 0..=110 {
            assert_eq!(
                fold_rule50_after(k, r),
                fold_rule50(k, r + 1),
                "after-variant must model parent clock + 1 (r = {r})"
            );
        }
    }

    // make_key constants are SF-current facts (D20); pin them.
    #[test]
    fn make_key_pins() {
        assert_eq!(make_key(0), 1442695040888963407);
        assert_eq!(make_key(1), 0x6C576FAC43FD007C);
        assert_ne!(make_key(0), make_key(1));
    }

    // Everything here is const-evaluable: ZOBRIST itself is a static built
    // by a const fn (its existence proves compile-time evaluation); the
    // helpers work in const context too.
    #[test]
    fn helpers_are_const_evaluable() {
        const FOLDED: Key = fold_rule50(0xABCD, 30);
        assert_eq!(FOLDED, 0xABCD ^ make_key(2));
        const {
            assert!(make_key(0) != 0);
            assert!(fold_rule50(7, 0) == 7);
        }
        assert_ne!(ZOBRIST.side, 0);
    }
}
