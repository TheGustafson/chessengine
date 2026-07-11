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
/// Numerator of the search-again depth damping term
/// `3 * (search_again_counter + 1) / 4` in the aspiration loop's
/// `adjusted_depth` (design/07 §3.6, design/12 §3.9 — the counter itself
/// is TM-owned and 0 until doc 12 step 8).
pub const ASPIRATION_SEARCH_AGAIN_NUM: Depth = 3;
/// Denominator of the search-again depth damping.
pub const ASPIRATION_SEARCH_AGAIN_DEN: Depth = 4;

// --- Move ordering: capture scoring (design/09 §3.7) ---

/// MVV weight on the victim value in the capture ordering score
/// `CAPTURE_MVV_FACTOR * victim_value + capture_history` (design/09 §3.7):
/// pure most-valuable-victim plus capture history, with no attacker (LVA)
/// term — SEE handles attacker risk at the good/bad split gate.
pub const CAPTURE_MVV_FACTOR: Value = 7;

// --- Move ordering: good/bad capture split (design/09 §3.6 stage 3) ---

/// Divisor of the score-dependent SEE gate `see_ge(m, -score / 18)` that
/// splits captures into good and bad (design/09 §3.6): a capture with a
/// high ordering score may lose a little material and still count as good;
/// a capture with terrible capture-history must win material to qualify.
pub const GOOD_CAPTURE_SEE_DIVISOR: Value = 18;

// --- Move ordering: quiet scoring (design/09 §3.7) ---

/// Weight on butterfly (main) history in the quiet ordering score.
pub const QUIET_MAIN_HISTORY_WEIGHT: Value = 2;
/// Weight on pawn-structure history in the quiet ordering score.
pub const QUIET_PAWN_HISTORY_WEIGHT: Value = 2;
/// Flat bonus for a quiet that gives check, gated by the SEE margin below
/// so a check that hangs material earns nothing.
pub const QUIET_CHECK_BONUS: Value = 16384;
/// SEE margin gating the check bonus: a checking quiet earns it only if the
/// move does not lose more than this much material (`see_ge(m, -75)`).
pub const QUIET_CHECK_SEE_MARGIN: Value = -75;
/// Per-piece-value multiplier of the threat escape/entry term: fleeing a
/// square attacked by a strictly cheaper enemy piece earns
/// `QUIET_THREAT_FACTOR * piece_value`, stepping onto one costs the same.
pub const QUIET_THREAT_FACTOR: Value = 20;
/// Weight on low-ply history in the quiet ordering score (plies 0..4 only,
/// scaled by `/ (1 + ply)`).
pub const QUIET_LOW_PLY_WEIGHT: Value = 8;

// --- Move ordering: quiet good/bad split + sort (design/09 §3.6) ---

/// Good/bad quiet partition threshold (design/09 §3.6 stages 5/7): quiets
/// scoring strictly above this are "good" (emitted before bad captures);
/// the exact complement are "bad" (emitted last). One shared threshold
/// partitions the quiet region with no dedup bookkeeping.
pub const GOOD_QUIET_THRESHOLD: Value = -14000;
/// Depth multiplier of the quiet partial-sort limit `-3560 * depth`
/// (design/09 §3.6 stage 4 / §3.8): deeper nodes sort more of the list, so
/// sorting work scales with what the search actually consumes.
pub const QUIET_SORT_DEPTH_FACTOR: Value = -3560;

// --- Move ordering: evasion scoring (design/09 §3.7) ---

/// Constant lifting every capturing evasion above every quiet evasion
/// (design/09 §3.7): `2^28` dwarfs any history-based quiet-evasion score.
pub const EVASION_CAPTURE_BONUS: Value = 1 << 28;

// --- Quiescence search (design/07 §3.8) ---

/// Futility headroom over the corrected static eval:
/// `futility_base = static_eval + 335` (§3.8 step 5). A capture whose
/// victim cannot lift this base above alpha is skipped.
pub const QS_FUTILITY_MARGIN: Value = 335;
/// Stand-pat fail-high softening `(467·best + 557·beta) / 1024` — applied
/// to non-decisive stand pats at or above beta (§3.8 step 5).
pub const QS_STANDPAT_BLEND: (Value, Value, Value) = (467, 557, 1024);
/// Node-exit fail-high softening `(481·best + 543·beta) / 1024` — applied
/// to non-decisive best values strictly above beta (§3.8 step 11).
pub const QS_FAILHIGH_BLEND: (Value, Value, Value) = (481, 543, 1024);
/// The hard SEE floor: any move whose static exchange value cannot reach
/// this bar is never searched in qsearch (§3.8 step 7).
pub const QS_SEE_FLOOR: Value = -74;
/// Late-move cutoff: after this many non-checking moves not aimed at the
/// previous move's destination, stop trying such moves entirely
/// (`move_count > 2`, §3.8 step 7).
pub const QS_LATE_MOVE_LIMIT: i32 = 2;

// --- Equal-value dither (design/07 §3.7.2 step 12) ---

/// Node-counter mask of the equal-value dither: near the leaves an
/// equal-to-best value is promoted to best only when `(nodes & 14) == 0`
/// (nodes ≡ 0 or 1 mod 16, ~1 in 8 made moves) — deliberate diversity so
/// equal lines do not all tie on the first one found.
pub const DITHER_NODE_MASK: u64 = 14;
/// Leaf-proximity gate of the dither: it fires only when
/// `ply + 2 >= root_depth`.
pub const DITHER_PLY_MARGIN: usize = 2;

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

// --- Time management: soft stop (design/12 §3.9) ---

/// Single-reply cap on the soft time budget (`min(561.7, total_time)`,
/// design/12 §3.9): with exactly one legal move at the root the search
/// result cannot change the move played, so never sit on it longer than
/// this — a viewer-experience cap, not a strength knob. In TM units:
/// milliseconds, or nodes under `nodestime` (doc 12 §3.11), like every
/// budget the soft stop compares against.
pub const ONE_REPLY_TOTAL_TIME_CAP: f64 = 561.7;

// --- Stop cadence (design/07 §3.9) ---

/// `check_time` is called at every main-search node entry but early-outs
/// on a countdown, doing real work every this many calls. Under a
/// `go nodes` limit the reset shrinks to `min(512, limits.nodes / 1024)`
/// (owned by the time-management layer, design/12).
pub const CHECK_TIME_CALLS: i32 = 512;
