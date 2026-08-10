//! Zobrist hashing: the five-key family's tables and folding helpers
//! (design/04 §3.4, D9).
//!
//! One const-evaluated table set, [`ZOBRIST`], feeds every key the engine
//! maintains incrementally: the main `key`, `pawn_key`, `material_key`,
//! `minor_key`, and the per-color `non_pawn_key`s. Pawn rows are ZEROED on
//! promotion ranks; castling is keyed by the whole 4-bit mask; the material
//! key reuses psq indices `8 + count` as count slots; the TT-facing key is
//! rule50-folded while repetition detection compares raw keys.
//!
//! Key *deltas* live in `make_move`, NOT in the placement mutators — the
//! ep/castling/side keys never route through placement (doc 04 §3.3).

use crate::attacks::Prng;
use crate::types::{Piece, PieceType, Square};

pub use crate::types::Key;

/// Stockfish's table-stream seed; our draw order differs, so keys differ.
const ZOBRIST_SEED: u64 = 1_070_372;

/// The five Zobrist key families (design/04 §3.4). Fields are public —
/// `make_move`'s dense XOR chains index the tables directly; material
/// count slots go through [`Tables::material`].
pub struct Tables {
    /// Piece-on-square keys: `psq[Piece::index()][sq.index()]`. Pawn rows
    /// are ZERO on their promotion ranks (promotions need no pawn-clear
    /// XOR at `to`); indices `8 + count` double as material-key count
    /// slots ([`Tables::material`]).
    pub psq: [[Key; Square::COUNT]; Piece::COUNT],
    /// By FILE of the en-passant square; XORed in only when a legal ep
    /// capture actually exists (D9).
    pub en_passant: [Key; 8],
    /// By the WHOLE 4-bit rights mask: any rights change is
    /// `castling[old] ^ castling[new]`. `castling[0]` is a real key.
    pub castling: [Key; 16],
    /// XORed in when Black is to move.
    pub side: Key,
    /// `pawn_key` baseline for a pawnless board.
    pub no_pawns: Key,
}

impl Tables {
    /// Material-key count slot: the key for "one more piece of kind `pc`",
    /// reusing `psq[pc][8 + count]`. `material_key` is the XOR of
    /// `material(pc, i)` for `i` in `0..count(pc)`: adding at PRE-add count
    /// `c` and removing at POST-removal count `c` XOR the same slot. Live
    /// slots reach `count == 9` (promotions allow 10 like non-pawn pieces).
    #[inline]
    pub const fn material(&self, pc: Piece, count: usize) -> Key {
        debug_assert!(
            count
                <= match pc.piece_type {
                    PieceType::Pawn => 7,
                    _ => 9,
                }
        );
        // NOTE: `& 63` states the debug_assert's bound where codegen can use
        // it (identity for every admitted count) — keeps make_move check-free.
        self.psq[pc.index()][(8 + count) & (Square::COUNT - 1)]
    }
}

/// The one table set, const-evaluated at compile time.
pub static ZOBRIST: Tables = build_tables();

/// Fill order (FIXED — changing it invalidates every stored key): `psq`
/// rows in `Piece::index()` order, each row a1..h8; then `en_passant`,
/// `castling`, `side`, `no_pawns`. The zeroing overwrites AFTER all draws.
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

    // Zero the pawn rows on the promotion ranks.
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

/// Mirrors SF's `make_key`: avalanche a small integer into key space.
pub const fn make_key(seed: u64) -> Key {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Rule50 folding (graph-history mitigation, D9): perturb the raw key in
/// 8-step buckets — clocks 0..=13 hash raw, 14..=21 XOR `make_key(0)`, etc.
/// `Board::key()` (the TT probe key) delegates here; repetition detection
/// compares RAW keys.
pub const fn fold_rule50(k: Key, rule50: i32) -> Key {
    if rule50 < 14 {
        k
    } else {
        k ^ make_key(((rule50 - 14) / 8) as u64)
    }
}

/// After-move variant: `fold_rule50_after(k, r) == fold_rule50(k, r + 1)`
/// — models "parent rule50 + 1", valid only for quiet non-pawn moves.
/// Exists EXCLUSIVELY for the speculative `prefetch_key`; `make_move`
/// folds the exact child clock with [`fold_rule50`].
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

    // Rebuilding reproduces the const-evaluated static exactly.
    #[test]
    fn tables_deterministic_across_builds() {
        let rebuilt = build_tables();
        assert_eq!(rebuilt.psq, ZOBRIST.psq);
        assert_eq!(rebuilt.en_passant, ZOBRIST.en_passant);
        assert_eq!(rebuilt.castling, ZOBRIST.castling);
        assert_eq!(rebuilt.side, ZOBRIST.side);
        assert_eq!(rebuilt.no_pawns, ZOBRIST.no_pawns);
    }

    // Regression pins: these change ONLY if the generator, seed, or fill
    // order changes — which invalidates every stored key.
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

    // The zeroed slots are EXACTLY psq[wp][56..64] and psq[bp][0..8].
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

    // Every live key is nonzero and pairwise distinct.
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

    // Count slots live through count 9 (non-pawns) / 7 (pawns).
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

    // Fold bucket boundaries; the after-variant is "parent clock + 1".
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

    // Pin the make_key constants.
    #[test]
    fn make_key_pins() {
        assert_eq!(make_key(0), 1442695040888963407);
        assert_eq!(make_key(1), 0x6C576FAC43FD007C);
        assert_ne!(make_key(0), make_key(1));
    }

    // Everything here is const-evaluable, helpers included.
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
