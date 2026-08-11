//! A UCI chess engine in Rust, aiming at Stockfish-class architecture and,
//! ultimately, strength — written the idiomatic Rust way, not as a C++
//! transliteration.
//!
//! Everything here is governed by the design suite in `design/`:
//! `design/decisions.md` pins the decisions and `design/00-master-plan.md`
//! orders the phases. Working conventions live in `CLAUDE.md`; test conventions
//! in `TESTING.md`. Perft is the correctness oracle, `bench` the search
//! regression oracle (D24).
//!
//! Layout: library crate + thin binary (`main.rs`, the UCI entry point).
//! Modules below are re-exported at the crate root.

// D14: unsafe is banned crate-wide; relaxing it per-module
// (`#[allow(unsafe_code)]`) requires a measured, documented justification and a
// `// SAFETY:` comment on each block.
#![deny(unsafe_code)]

/// Instrument-only **layout null**: a cold, never-called function that shifts
/// `.text` by a few dozen bytes and moves every downstream symbol's L1i set
/// mapping, while changing **nothing** the search executes — so an arm's
/// icache-fill number can be compared against a pure-relayout baseline.
///
/// `#[cold]` + `#[inline(never)]` keep it out of the hot code; the `#[used]`
/// function-pointer static below keeps fat LTO from deleting it (`no_mangle`
/// would be the usual trick, but this crate is `#![deny(unsafe_code)]` and
/// `#[unsafe(no_mangle)]` trips it).
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

/// Runtime-configurable move-ordering ARM BENCH (instrument only).
#[cfg(feature = "arm70")]
pub mod arm70;
pub mod attacks;
pub mod bench;
pub mod bench_positions_archon;
pub mod bitboard;
pub mod board;
/// Move-ordering QUALITY census (instrument only).
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
