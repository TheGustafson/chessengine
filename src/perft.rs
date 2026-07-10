//! Perft — the correctness oracle for movegen and make/unmake (D24).
//!
//! API and semantics are specified in `design/06-perft-validation.md`
//! §3.1 (bulk-counting `perft`), §3.2 (`divide` and its SF-format output),
//! and §3.3 (`perft_checked`). Perft counts leaf nodes of the legal move
//! tree — no 50-move, repetition, or insufficient-material cutoffs, and no
//! TT acceleration (the oracle must not depend on the Zobrist keys it
//! validates).

use crate::board::Board;
use crate::movegen;
use crate::moves::{Move, MoveList};
use std::fmt;

/// Count all legal move paths of exactly `depth` plies from `board`.
/// `perft(0) == 1` by convention.
///
/// design/06 §3.1: bulk counting at the horizon (at `depth == 1` the answer
/// is the legal move list's length — no make/unmake), `&mut Board` with
/// strict make/unmake pairing (never clone-per-node), so this exercises
/// exactly the code path search will use, and the `gives_check` O(1)
/// check-prediction machinery is validated before search relies on it.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1; // by convention (design/06 §3.1)
    }
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    if depth == 1 {
        return list.len() as u64; // bulk counting: no make/unmake at horizon
    }
    let mut nodes = 0;
    for m in &list {
        let gives_check = board.gives_check(m); // D10: caller supplies it
        board.make_move(m, gives_check);
        nodes += perft(board, depth - 1);
        board.unmake_move(m);
    }
    nodes
}

/// Perft that validates board invariants at every node. ~10-50x slower
/// than [`perft`]; used by tests and for bug hunts, never for timing.
///
/// design/06 §3.3, adapted to the landed doc 04 invariant surface: after
/// each `make_move` it runs `Board::assert_consistent` — which covers layer
/// consistency, all five Zobrist keys against a from-scratch recomputation
/// (doc 04 §3.12's `assert_state_keys_consistent`), and the derived
/// checkers/blockers/pinners/check_squares recompute — plus an independent
/// `gives_check`-agreement check here; after each `unmake_move` it checks
/// round-trip identity against a snapshot (`Board: PartialEq`).
///
/// Compiled always, but the `assert_consistent` oracle only exists in
/// test/debug builds (`#[cfg(any(test, debug_assertions))]`); the
/// gives_check and round-trip asserts run in every build.
pub fn perft_checked(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1; // same convention as `perft` (design/06 §3.1)
    }
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for m in &list {
        let snapshot = board.clone();
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);

        // Invariants 1, 2, 4 of design/06 §3.3: layers, keys, derived state.
        #[cfg(any(test, debug_assertions))]
        board.assert_consistent();

        // Invariant 3: the pre-move `gives_check` prediction must equal
        // reality. Recomputed independently — make_move trusts the flag to
        // set `checkers` cheaply, so asking `board.checkers()` alone would
        // partly test the flag against itself.
        let us = board.side_to_move();
        let fresh_checkers = board.attackers_to(board.king_square(us), board.all_occupancy())
            & board.occupancy(us.flip());
        assert_eq!(
            gives_check,
            !fresh_checkers.is_empty(),
            "gives_check mispredicted for {m}"
        );
        assert_eq!(
            board.checkers(),
            fresh_checkers,
            "stored checkers drifted after {m}"
        );

        nodes += perft_checked(board, depth - 1);
        board.unmake_move(m);

        // Invariant 5: unmake round-trip identity (placement, side to move,
        // clocks, castling geometry, the whole state stack — one derived
        // PartialEq comparison).
        assert!(
            *board == snapshot,
            "unmake_move failed to restore the position after {m}"
        );
    }
    nodes
}

/// One root move and the size of its subtree (design/06 §3.2).
pub struct DivideEntry {
    pub mv: Move,
    pub nodes: u64,
}

/// Per-root-move counts plus the total. Entries in generation order
/// (design/06 §3.2).
pub struct DivideResult {
    pub entries: Vec<DivideEntry>,
    pub total: u64,
}

/// Perft divide: per-root-move subtree counts, for descending into a
/// disagreement with the known-count tables (design/06 §3.2).
///
/// Requires `depth >= 1`.
pub fn divide(board: &mut Board, depth: u32) -> DivideResult {
    debug_assert!(depth >= 1);
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    let mut entries = Vec::with_capacity(list.len());
    let mut total = 0;
    for m in &list {
        let nodes = if depth == 1 {
            1
        } else {
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            let n = perft(board, depth - 1);
            board.unmake_move(m);
            n
        };
        entries.push(DivideEntry { mv: m, nodes });
        total += nodes;
    }
    DivideResult { entries, total }
}

/// Stockfish-format divide output, parsed by `tools/perft_diff.py`
/// (design/06 §3.2): one `<move>: <count>` line per root move in entry
/// order, a blank line, then `Nodes searched: <total>`.
impl fmt::Display for DivideResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}: {}", entry.mv, entry.nodes)?;
        }
        writeln!(f)?;
        writeln!(f, "Nodes searched: {}", self.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    /// Parse a FEN, panicking with context on failure.
    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap_or_else(|e| panic!("bad test FEN {fen:?}: {e}"))
    }

    #[test]
    fn perft_zero_is_one() {
        // design/06 §5: perft(0) == 1.
        let mut board = Board::startpos();
        assert_eq!(perft(&mut board, 0), 1);
        assert_eq!(perft_checked(&mut board, 0), 1);
    }

    #[test]
    fn divide_display_is_sf_format() {
        // design/06 §3.2: `move: count` lines, blank line, total. The
        // format is load-bearing — tools/perft_diff.py parses it.
        let result = DivideResult {
            entries: vec![
                DivideEntry {
                    mv: Move::normal(Square::A2, Square::A3),
                    nodes: 44,
                },
                DivideEntry {
                    mv: Move::normal(Square::E2, Square::E4),
                    nodes: 44,
                },
            ],
            total: 2039,
        };
        assert_eq!(
            result.to_string(),
            "a2a3: 44\ne2e4: 44\n\nNodes searched: 2039\n"
        );
    }

    #[test]
    fn perft_fast_smoke_startpos() {
        // design/06 §3.4 startpos ladder, fast depths (✔ python-chess).
        let mut board = Board::startpos();
        assert_eq!(perft(&mut board, 1), 20);
        assert_eq!(perft(&mut board, 2), 400);
        assert_eq!(perft(&mut board, 3), 8_902);
        assert_eq!(perft(&mut board, 4), 197_281);
    }

    #[test]
    fn divide_total_equals_perft_kiwipete_d3() {
        // design/06 §5: divide's total equals perft, and its entries sum
        // to the total (kiwipete d3 = 97,862 ✔).
        const KIWIPETE: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut board = Board::from_fen(KIWIPETE).expect("kiwipete FEN is valid");
        let result = divide(&mut board, 3);
        assert_eq!(result.total, 97_862);
        assert_eq!(perft(&mut board, 3), result.total);
        let sum: u64 = result.entries.iter().map(|e| e.nodes).sum();
        assert_eq!(sum, result.total);
    }

    /// The checked mini-tier (design/06 §3.5 tier 1): `perft_checked` — with
    /// the full per-node invariant oracle — reproduces the exact known
    /// counts on a bug-class-diverse subset at shallow depth (~90 K nodes
    /// total with checking). Counts from tests/data/perft.epd
    /// (✔ python-chess 1.11.2).
    #[test]
    fn perft_checked_mini_tier() {
        const MINI_TIER: &[(&str, u32, u64)] = &[
            // startpos: breadth.
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                3,
                8_902,
            ),
            // kiwipete: castling, ep, promotions, discovered/double checks.
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2,
                2_039,
            ),
            // ep-pin-h: ep-discovered horizontal pin.
            ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 3, 1_670),
            // ep-pin-rank: ep target set but exd3 illegal (rank pin).
            ("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1", 3, 863),
            // castle-rights: rights revoked when a rook is captured.
            ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 2, 1_141),
            // castle-prevented: attacked transit squares, rights persist.
            ("r3k2r/8/3Q4/8/8/8/8/R3K2R b KQkq - 0 1", 2, 872),
            // promo-storm: mass promotions/underpromotions, both colors.
            ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 3, 9_483),
            // promo-out-of-check: promotion as a check evasion.
            ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 3, 1_442),
            // double-check: only king moves legal at the root.
            ("4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1", 4, 4_755),
            // stale-checkmate: stalemate and checkmate leaves mixed.
            ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 4, 926),
            // cpw5: the make/unmake killer (un-promote cleanly).
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                2,
                1_486,
            ),
        ];
        for &(fen, depth, expected) in MINI_TIER {
            let mut b = board(fen);
            assert_eq!(
                perft_checked(&mut b, depth),
                expected,
                "perft_checked mismatch on {fen} depth {depth}"
            );
        }
    }
}
