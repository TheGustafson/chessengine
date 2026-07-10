//! A UCI chess engine in Rust, aiming at Stockfish-class architecture and,
//! ultimately, strength — written the idiomatic Rust way, not as a C++
//! transliteration.
//!
//! Everything here is governed by the design suite in `design/`:
//! `design/decisions.md` pins decisions D1–D27 (decisions change there first;
//! docs follow; code follows docs) and `design/00-master-plan.md` orders the
//! phases P0–P14. Working conventions live in `CLAUDE.md`; test conventions
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
pub mod eval;
pub mod fen;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod search;
pub mod timeman;
pub mod types;
pub mod uci;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use board::{Board, CastlingRights};
pub use fen::{FenError, START_FEN};
pub use moves::{Move, MoveKind, MoveList};
pub use types::{Color, Direction, Piece, PieceType, Square};
