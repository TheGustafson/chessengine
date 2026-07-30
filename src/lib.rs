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

pub mod attacks;
pub mod bench;
pub mod bitboard;
pub mod board;
/// Perf study 34's move-ordering census (instrument only).
#[cfg(feature = "census34")]
pub mod census34;
pub mod cuckoo;
/// Perf study 40's dose amplifier for the movepick partial sort
/// (instrument only; the search tree is untouched at every dose).
#[cfg(feature = "dosesort40")]
pub mod dose40;
pub mod eval;
pub mod fen;
pub mod history;
pub mod movegen;
pub mod movepick;
pub mod moves;
#[cfg(feature = "datagen")]
pub mod obs;
pub mod perft;
#[cfg(feature = "policy")]
pub mod policy;
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
