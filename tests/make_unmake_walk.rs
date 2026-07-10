//! Randomized movegen-driven make/unmake stress — the key-invariance walk
//! of doc 04 §8.2 (its §5 "scripted (later: randomized, once movegen
//! exists)" test, now in its randomized form).
//!
//! Over ≥ 10,000 positions reached by seeded-random legal walks from every
//! `tests/data/perft.epd` root, verify after every `make_move` that the
//! incremental keys equal a from-scratch recompute (all six keys: main,
//! pawn, material, minor, non_pawn[2] — `assert_state_keys_consistent`,
//! doc 04 §3.12; in debug builds `make_move` additionally runs the full
//! `assert_consistent` oracle at every node), and on unwinding that every
//! `unmake_move` restores a `Board` equal to the pre-move snapshot
//! (derived `PartialEq`, the doc 06 §3.3 invariant-5 comparison).
//!
//! Deep random walks complement the perft oracles: the perft tree only
//! explores a few plies from the 23 fixed roots, while a 120-ply walk
//! stresses make/unmake on late-game material distributions (promoted
//! pieces, bare kings, dead castling rights) that no shallow tree reaches.
//! Deterministic seed: failures reproduce exactly; no RNG dependency (D15).

mod common;

use chessengine::movegen::generate_legal;
use chessengine::{Board, Move, MoveList};

/// Doc 04 §8.2 floor: at least this many make/unmake round-trips.
const TARGET_STEPS: u64 = 10_240;

/// Plies per walk before restarting from the root. Long enough to leave
/// the opening/tactical melee the perft roots encode; short enough that
/// dozens of restarts keep position diversity high.
const MAX_PLIES: usize = 120;

/// Deterministic SplitMix64 — same tiny generator as `tests/check_info.rs`
/// (public-domain construction from Steele et al.).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-enough index in `0..bound` (bias is irrelevant here).
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

#[test]
fn randomized_walks_keep_keys_and_roundtrip_make_unmake() {
    let records = common::load_perft_epd().unwrap_or_else(|e| panic!("{e}"));
    assert!(!records.is_empty(), "perft.epd contains no cases");

    let mut rng = SplitMix64(0xC0FF_EE00_D0C4_0802);
    let mut steps = 0u64;

    'outer: loop {
        for record in &records {
            let mut board = common::board(&record.fen);
            // Pre-move snapshots + moves, unwound in reverse below.
            let mut trail: Vec<(Board, Move)> = Vec::with_capacity(MAX_PLIES);

            for _ in 0..MAX_PLIES {
                let mut list = MoveList::new();
                generate_legal(&board, &mut list);
                if list.is_empty() {
                    break; // checkmate or stalemate: restart from the next root
                }
                let m = list[rng.below(list.len())];
                let snapshot = board.clone();
                let gives_check = board.gives_check(m);
                board.make_move(m, gives_check);

                // All six incremental keys vs a from-scratch recompute
                // (compiled out of release builds along with the oracle).
                #[cfg(debug_assertions)]
                board.assert_state_keys_consistent();

                trail.push((snapshot, m));
                steps += 1;
            }

            // Unwind the whole walk: every unmake must restore the exact
            // pre-move board (placement, clocks, rights, keys, state stack).
            while let Some((snapshot, m)) = trail.pop() {
                board.unmake_move(m);
                assert!(
                    board == snapshot,
                    "unmake_move failed to restore the position after {m} \
                     (walk root {:?}, step {})",
                    record.fen,
                    trail.len(),
                );
            }

            if steps >= TARGET_STEPS {
                break 'outer;
            }
        }
    }

    assert!(steps >= TARGET_STEPS, "walk budget miscounted: {steps}");
}
