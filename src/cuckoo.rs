//! Marcel van Kervinck's cuckoo tables — the machinery behind
//! `upcoming_repetition` (design/04 §3.9, design/10 item 19 "N1").
//!
//! This is the last search feature the P9 waves left deferred: item 19 landed
//! its fail-low bonuses, ttPv backprop and TT no-moves store, but the
//! upcoming-repetition early draw was blocked on these tables never being
//! built.
//!
//! **What it buys.** A side that can reach a position already on the path can
//! force a draw, so the search may return a draw score immediately instead of
//! exploring the subtree — a large pruning win in blocked or fortress-like
//! positions, and the reason SF checks it before every other early exit.
//!
//! **The scheme.** Every REVERSIBLE non-pawn move `s1 -> s2` changes the
//! position key by exactly `psq[pc][s1] ^ psq[pc][s2] ^ side`, independent of
//! everything else on the board. Precompute that delta for all such moves into
//! a hash table, and a repetition test becomes: XOR two path keys together and
//! ask whether the difference is a single reversible move.
//!
//! Two slots per key (`H1`/`H2`) with cuckoo displacement gives worst-case
//! two-probe lookup. The build is exact — Stockfish's own table holds exactly
//! **3668** entries and [`tables`] asserts the same count, which is a hard
//! oracle on the enumeration being right.
//!
//! Built lazily alongside the magics (D12): it needs empty-board slider
//! attacks, so it cannot be a `const` table.

use std::sync::LazyLock;

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square};
use crate::zobrist::{Key, ZOBRIST};

/// Table size. Must be a power of two — the slot functions mask with it.
const SIZE: usize = 8192;
const MASK: u64 = (SIZE as u64) - 1;

/// Stockfish's exact entry count for the same enumeration. A build that
/// disagrees has the piece set, the attack sets, or the pair ordering wrong.
const EXPECTED_ENTRIES: usize = 3668;

/// First slot: the low 13 bits.
#[inline]
fn h1(k: Key) -> usize {
    (k & MASK) as usize
}

/// Second slot: bits 16..29. Disjoint from `h1`'s bits so a displaced entry
/// lands somewhere unrelated.
#[inline]
fn h2(k: Key) -> usize {
    ((k >> 16) & MASK) as usize
}

/// The two parallel tables: key deltas and the move each delta encodes.
pub struct Cuckoo {
    keys: [Key; SIZE],
    moves: [Move; SIZE],
}

impl Cuckoo {
    /// The move whose key-delta is `key`, if any. Two probes, worst case.
    #[inline]
    pub fn lookup(&self, key: Key) -> Option<Move> {
        for slot in [h1(key), h2(key)] {
            if self.keys[slot] == key {
                return Some(self.moves[slot]);
            }
        }
        None
    }
}

static CUCKOO: LazyLock<Cuckoo> = LazyLock::new(build);

/// The cuckoo tables, built on first use (needs the magic slider attacks).
#[inline]
pub fn tables() -> &'static Cuckoo {
    &CUCKOO
}

fn build() -> Cuckoo {
    let mut keys = [0 as Key; SIZE];
    let mut moves = [Move::NULL; SIZE];
    let mut count = 0usize;

    for color in [Color::White, Color::Black] {
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            // Pawns are excluded: their moves are irreversible, so they can
            // never close a repetition.
            let pc = Piece::new(color, pt);
            for s1 in 0..64u8 {
                let from = Square::new(s1);
                // Empty-board attack set: reversibility means nothing may be
                // in the way, and the emptiness of the path is re-checked at
                // query time against the ACTUAL occupancy.
                let targets = attacks::attacks(pt, from, Bitboard::EMPTY);
                for to in targets {
                    // Each unordered pair once: s1 < s2.
                    if to.index() <= s1 {
                        continue;
                    }
                    let delta = ZOBRIST.psq[pc.index()][usize::from(from.index())]
                        ^ ZOBRIST.psq[pc.index()][usize::from(to.index())]
                        ^ ZOBRIST.side;
                    // Cuckoo insertion: sit in h1; whoever is evicted moves to
                    // its own alternate slot, repeating until an empty one.
                    let mut k = delta;
                    let mut mv = Move::normal(from, to);
                    let mut slot = h1(k);
                    loop {
                        std::mem::swap(&mut keys[slot], &mut k);
                        std::mem::swap(&mut moves[slot], &mut mv);
                        if k == 0 {
                            break; // displaced an empty slot: done
                        }
                        // Send the evictee to its OTHER slot.
                        slot = if slot == h1(k) { h2(k) } else { h1(k) };
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(
        count, EXPECTED_ENTRIES,
        "cuckoo table built {count} entries, expected {EXPECTED_ENTRIES} \
         (piece set, attack sets or pair ordering is wrong)"
    );
    Cuckoo { keys, moves }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_matches_stockfishs_exact_entry_count() {
        // The assert lives in build(); touching the table runs it.
        let t = tables();
        // And the tables must actually be populated.
        let filled = t.keys.iter().filter(|k| **k != 0).count();
        assert_eq!(filled, EXPECTED_ENTRIES, "occupied slots");
    }

    #[test]
    fn every_reversible_move_delta_is_findable() {
        let t = tables();
        // Spot-check across piece types: the delta of a legal empty-board
        // move must resolve back to that move (or its mirror — the table
        // stores each unordered pair once, in s1<s2 order).
        for (pt, from, to) in [
            (PieceType::Knight, Square::G1, Square::F3),
            (PieceType::Bishop, Square::C1, Square::H6),
            (PieceType::Rook, Square::A1, Square::A8),
            (PieceType::Queen, Square::D1, Square::D8),
            (PieceType::King, Square::E1, Square::E2),
        ] {
            let pc = Piece::new(Color::White, pt);
            let delta = ZOBRIST.psq[pc.index()][usize::from(from.index())]
                ^ ZOBRIST.psq[pc.index()][usize::from(to.index())]
                ^ ZOBRIST.side;
            let found = t.lookup(delta).expect("reversible move delta missing");
            let (a, b) = (found.from(), found.to());
            assert!(
                (a == from && b == to) || (a == to && b == from),
                "{pt:?} {from}{to} resolved to {a}{b}"
            );
        }
    }

    #[test]
    fn a_pawn_push_delta_is_absent() {
        // Pawn moves are irreversible and must never appear.
        let pc = Piece::new(Color::White, PieceType::Pawn);
        let delta = ZOBRIST.psq[pc.index()][usize::from(Square::E2.index())]
            ^ ZOBRIST.psq[pc.index()][usize::from(Square::E4.index())]
            ^ ZOBRIST.side;
        assert!(tables().lookup(delta).is_none());
    }

    #[test]
    fn a_nonsense_delta_is_absent() {
        assert!(tables().lookup(0x0123_4567_89ab_cdef).is_none());
    }
}
