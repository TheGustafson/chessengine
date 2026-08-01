//! A UCI chess engine in Rust, aiming at Stockfish-class architecture and,
//! ultimately, strength — written the idiomatic Rust way, not as a C++
//! transliteration.
//!
//! Everything here is governed by the design suite in `design/`:
//! `design/decisions.md` pins decisions D1–D27 (decisions change there first;
//! docs follow; code follows docs) and `design/00-master-plan.md` orders the
//! phases P0–P15. Working conventions live in `CLAUDE.md`; test conventions
//! in `TESTING.md`. Perft is the correctness oracle, `bench` the search
//! regression oracle (D24).
//!
//! Layout: library crate + thin binary (`main.rs`, the UCI entry point).
//! Modules below are re-exported at the crate root.

// D14: unsafe is banned crate-wide for now; later phases relax this
// per-module (`#[allow(unsafe_code)]`) only with a measured, documented
// justification and a `// SAFETY:` comment on each block.
#![deny(unsafe_code)]

/// Perf study 68's **layout null**: a cold, never-called function that exists
/// only to shift `.text` by a few dozen bytes and move every downstream symbol's
/// L1i set mapping, while changing **nothing** the search executes.
///
/// It exists because `ic_cache_fill_l2` is not trustworthy at the few-per-cent
/// level in this engine and the study needed to know HOW untrustworthy. Two
/// points were already in hand — a byte-identical rebuild (±0.5 %) and a
/// zero-semantic basic-block relayout (+1.103 %) — and neither is a *small*
/// shift, which is the perturbation the landed arms actually apply. This is
/// that third point, and it is the only honest way to say whether an arm's
/// fill number is the arm or the lottery.
///
/// `#[cold]` + `#[inline(never)]` keep it out of the hot code; the `#[used]`
/// function-pointer static below is what keeps fat LTO from deleting it
/// (`no_mangle` would be the usual trick, but this crate is
/// `#![deny(unsafe_code)]` and `#[unsafe(no_mangle)]` trips it — `#[used]` is
/// safe and does the same job).
///
/// Default OFF; it must never ship.
#[cfg(feature = "pad68")]
#[cold]
#[inline(never)]
pub fn archon_s68_layout_null(x: u64) -> u64 {
    // Enough arithmetic to occupy a few dozen bytes, and dependent enough that
    // the optimiser cannot fold it to a constant or a `ret`.
    let mut a = x ^ 0x9E37_79B9_7F4A_7C15;
    a = a.rotate_left(31).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    a ^= a >> 27;
    a = a.wrapping_mul(0x94D0_49BB_1331_11EB);
    a ^ (a >> 31)
}

#[cfg(feature = "pad68")]
#[used]
static ARCHON_S68_LAYOUT_NULL: fn(u64) -> u64 = archon_s68_layout_null;

/// Perf study 70's runtime-configurable move-ordering ARM BENCH.
#[cfg(feature = "arm70")]
pub mod arm70;
pub mod attacks;
pub mod bench;
pub mod bitboard;
pub mod board;
/// Perf study 70's move-ordering QUALITY census (instrument only).
#[cfg(feature = "census70")]
pub mod census70;
pub mod cuckoo;
pub mod eval;
pub mod fen;
pub mod history;
pub mod movegen;
pub mod movepick;
pub mod moves;
#[cfg(feature = "datagen")]
pub mod obs;
pub mod perft;
pub mod search;
#[cfg(feature = "search-control")]
pub mod search_control;
#[cfg(feature = "search-control")]
pub mod search_intervention;
#[cfg(feature = "search-control")]
pub mod search_trace;
pub mod threads;
pub mod timeman;
pub mod tt;
/// Perf study 42's transposition-table census (instrument only; relaxed
/// atomics on paths that make no search decision, so the tree is unmoved).
#[cfg(feature = "ttcensus42")]
pub mod ttcensus42;
#[macro_use]
mod tune;
pub mod types;
pub mod uci;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use board::{Board, CastlingRights};
pub use fen::{FenError, START_FEN};
pub use moves::{Move, MoveKind, MoveList};
pub use types::{Color, Direction, Piece, PieceType, Square};
