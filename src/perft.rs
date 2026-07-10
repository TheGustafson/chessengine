//! Perft — the correctness oracle for movegen and make/unmake (D24).
//!
//! API and semantics are specified in `design/06-perft-validation.md`
//! §3.1 (bulk-counting `perft`), §3.2 (`divide` and its SF-format output),
//! and §3.3 (`perft_checked`). The bodies are `todo!` stubs until P3 lands
//! `movegen::generate_legal` and P2 lands `make_move`/`unmake_move`; the
//! signatures below are the contract the rest of the engine (and the RED
//! test suite in `tests/perft_suite.rs`) compiles against.

use crate::board::Board;
use crate::moves::Move;
use std::fmt;

/// Count all legal move paths of exactly `depth` plies from `board`.
/// `perft(0) == 1` by convention.
///
/// design/06 §3.1: bulk counting at the horizon (at `depth == 1` the answer
/// is the legal move list's length — no make/unmake), `&mut Board` with
/// strict make/unmake pairing (never clone-per-node), no 50-move /
/// repetition / insufficient-material cutoffs, and no TT acceleration.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1; // by convention (design/06 §3.1)
    }
    let _ = board;
    todo!("P3: movegen")
}

/// Perft that validates board invariants at every node. ~10-50x slower
/// than [`perft`]; used by tests and for bug hunts, never for timing.
///
/// design/06 §3.3: after each `make_move` it checks layer consistency,
/// all five Zobrist keys against a from-scratch recomputation,
/// `gives_check` agreement, and derived check/pin state; after each
/// `unmake_move` it checks round-trip identity against a snapshot.
pub fn perft_checked(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1; // same convention as `perft` (design/06 §3.1)
    }
    let _ = board;
    todo!("P3: movegen")
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
    let _ = (board, depth);
    todo!("P3: movegen")
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

    #[test]
    fn perft_zero_is_one() {
        // design/06 §5: perft(0) == 1. Reachable today: the depth-0
        // short-circuit precedes the movegen stub.
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
    #[ignore = "RED: un-ignore when P3 movegen lands"]
    fn perft_fast_smoke_startpos() {
        // design/06 §3.4 startpos ladder, fast depths (✔ python-chess).
        let mut board = Board::startpos();
        assert_eq!(perft(&mut board, 1), 20);
        assert_eq!(perft(&mut board, 2), 400);
        assert_eq!(perft(&mut board, 3), 8_902);
        assert_eq!(perft(&mut board, 4), 197_281);
    }

    #[test]
    #[ignore = "RED: un-ignore when P3 movegen lands"]
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
}
