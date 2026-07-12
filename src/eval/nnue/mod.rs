//! NNUE inference (design/11 §3.4) — the Phase C evaluation path.
//!
//! **Scope (P13 step 1).** This is the *inference skeleton*: the quantized
//! network container (CENN, [`loader`]), the perspective feature transformer
//! and its incremental accumulator ([`accumulator`]), and a SCALAR quantized
//! SCReLU/CReLU forward pass ([`simd`]). It is purely additive — the active
//! evaluation (`eval::evaluate`) still returns the PSQT value; the NNUE
//! forward pass is exercised only by tests and the `nnue-eval` dev
//! subcommand. Wiring NNUE in as the eval, the AVX2 backend, and a trained
//! net all land in later steps (design/11 §11).
//!
//! **Architecture** (`768 → 2×256 → 1`, single perspective pair, SCReLU):
//! 768 sparse binary inputs (`2 colors × 6 piece types × 64 squares`) feed a
//! shared feature transformer producing two [`L1`]-wide accumulators (one per
//! perspective). The output layer concatenates the two activated
//! accumulators — **side-to-move half first** — and dots with the output
//! weights (design/11 §3.4.1).
//!
//! **Quantization** (bullet-standard, design/11 §3.4.3): feature-transformer
//! weights and bias scaled by [`QA`], output weights by [`QB`], centipawn
//! output by [`SCALE`]. The scalar forward accumulates in `i64` and is the
//! bit-exact reference the future AVX2 path (step 2) must match.

pub mod accumulator;
pub mod loader;
pub mod network;
pub mod simd;

// --- Architecture dimensions (design/11 §3.4.1) ----------------------------

/// Number of sparse input features: `2 colors × 6 piece types × 64 squares`.
pub const INPUT: usize = 768;

/// Feature-transformer / accumulator width (the "L1" hidden size). The first
/// net is deliberately small (design/11 §11 locked choice); the upgrade path
/// grows this to 512/1024.
pub const L1: usize = 256;

// --- Quantization constants (design/11 §3.4.3) -----------------------------

/// Feature-transformer scale: `w0_q = round(w0 · QA)`, stored `i16`. Also the
/// SCReLU clamp ceiling (`clamp(acc, 0, QA)`).
pub const QA: i32 = 255;

/// Output-layer weight scale: `w1_q = round(w1 · QB)`, stored `i16`.
pub const QB: i32 = 64;

/// Centipawn output scale.
pub const SCALE: i32 = 400;

// --- Re-exported public surface --------------------------------------------

pub use accumulator::{AccumulatorEntry, NnueContext, evaluate_forward};
pub use loader::{NnueError, load, load_bytes, serialize};
pub use network::{Activation, Network, feature_index};
pub use simd::{Aligned64, KERNELS, Kernels};

// `DirtyPiece` is defined in `board` (its data is a board-mutation fact and
// its constructor reads `Board`'s private castling geometry — design/11
// §3.4.4 layering note). Re-exported here so the design-doc-facing path
// `eval::nnue::DirtyPiece` still resolves.
pub use crate::board::DirtyPiece;
