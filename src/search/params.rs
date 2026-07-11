//! Tuned search parameters — every SPSA-movable constant, named and
//! greppable in one file (D20; conventions: design/01 §3.8).
//!
//! Values are SF-current starting points from Stockfish commit `eca43a97`
//! (2026-07-06) unless noted. See
//! `stockfish_reference/stockfish_search_notes.md`.
//!
//! What does NOT live here: *structural* constants (bit widths,
//! `VALUE_MATE`, array sizes) are semantics and live with their types;
//! piece values are `types::PIECE_VALUES` (design/02 §3.9); history
//! gravity bounds are const generics in `history.rs` (design/09 §3.4) —
//! only their multipliers are tunables.
//!
//! Naming: `TECHNIQUE_ROLE` screaming-snake, grouped by technique in
//! design/07 / design/10 section order. Each constant documents only its
//! role; the formula *shape* is documented at the use site in `search`,
//! citing the owning design section. P0 seeds the aspiration and razoring
//! groups (design/01 §4 step 4); the full inventory lands with designs
//! 07, 09, and 10. The rule for reviewers: any bare numeric literal in
//! `search/`, `movepick.rs`, or `see.rs` that a tuner could move is a
//! defect (design/01 §3.8).

use crate::types::{Depth, Value};

// --- Aspiration windows (design/07 §3.6) ---

/// Base half-width of the window around the root move's average score.
pub const ASPIRATION_BASE_DELTA: Value = 5;
/// Divisor of `mean_squared_score`: unstable scores widen the base delta.
pub const ASPIRATION_MSS_DIVISOR: i64 = 10588;
/// Numerator of the per-re-search widening `delta += delta * 44 / 128`.
pub const ASPIRATION_WIDEN_NUM: Value = 44;
/// Denominator of the per-re-search widening (~x1.34 per fail).
pub const ASPIRATION_WIDEN_DEN: Value = 128;

// --- Razoring (design/10 §3.1; node step N9) ---

/// Flat margin below alpha at which shallow nodes drop to qsearch.
pub const RAZOR_BASE: Value = 465;
/// Depth-quadratic term of the razoring margin (`* depth^2`).
pub const RAZOR_DEPTH_MULT: Value = 300;

// --- Transposition-table interaction (design/07 §3.7.2 steps 7/14) ---

/// TT cutoffs are refused once the halfmove clock reaches this (the
/// graph-history guard of design/07 §3.7.2 step 7 / design/08 §3.12).
/// `i32`: matches `value_from_tt`'s rule50 parameter (doc 02).
pub const TT_CUTOFF_RULE50_LIMIT: i32 = 96;
/// Extra depth credited to the TT store of a no-legal-moves node
/// (design/07 §3.7.2 step 14): a mate/stalemate terminal is exact far
/// beyond its nominal depth, so make the entry hard to displace.
pub const TT_NO_MOVES_DEPTH_BONUS: Depth = 6;

// --- Stop cadence (design/07 §3.9) ---

/// `check_time` is called at every main-search node entry but early-outs
/// on a countdown, doing real work every this many calls. Under a
/// `go nodes` limit the reset shrinks to `min(512, limits.nodes / 1024)`
/// (owned by the time-management layer, design/12).
pub const CHECK_TIME_CALLS: i32 = 512;
