//! Named evaluation constants (D20): every tunable number in `src/eval/`
//! lives here, greppable — a bare numeric literal a tuner could move is a
//! defect (design/01 §3.8).
//!
//! The outer-shape constants below are SF-current starting points recorded
//! verbatim from `evaluate.cpp` (design/11 §3.6; commit per
//! `stockfish_reference/stockfish_search_notes.md` §10). The formula SHAPES
//! are what we keep; the values are tuner-owned. All are `i64` because the
//! outer shape computes in 64-bit — `v * (EVAL_BLEND_BASE + material)`
//! overflows `i32` (design/11 §3.6 notes).
//!
//! What does NOT live here: PeSTO piece values and tables are provenance-
//! bound transcriptions in `psqt.rs`; game-phase increments are structural
//! to the tapering scheme and live in `score.rs`; value bands
//! (`VALUE_TB_*`) are semantics owned by `types`.

crate::tunables! {
    /// Pawn weight in the outer-shape material term:
    /// `material = MATERIAL_PAWN_WEIGHT * pawn_count + non_pawn_material_total`.
    pub MATERIAL_PAWN_WEIGHT: i64 = 534;

    /// Base of the eval/optimism material blend — the denominator, and the
    /// material-independent weight on the eval side:
    /// `v = (v * (EVAL_BLEND_BASE + material) + opt * (OPTIMISM_BLEND_BASE + material)) / EVAL_BLEND_BASE`.
    /// Heavier boards trust the eval more; lighter boards trust optimism
    /// relatively more.
    pub EVAL_BLEND_BASE: i64 = 77871;

    /// Material-independent weight on the optimism side of the blend.
    pub OPTIMISM_BLEND_BASE: i64 = 7191;

    /// Rule-50 damping divisor: `v -= v * rule50 / RULE50_DAMPING_DEN` shuffles
    /// the eval toward zero as the halfmove clock grows.
    pub RULE50_DAMPING_DEN: i64 = 199;

    /// Complexity-blend divisor on the optimism side:
    /// `opt += opt * complexity / COMPLEXITY_OPTIMISM_DIV` (design/11 §3.6).
    /// Inactive until Phase C's psqt/positional split exists — `complexity`
    /// is identically 0 in Phases A/B — but pinned here so the whole outer
    /// shape reads from one place.
    pub COMPLEXITY_OPTIMISM_DIV: i64 = 476;

    /// Complexity-blend divisor on the eval side:
    /// `v -= v * complexity / COMPLEXITY_EVAL_DIV`. Inactive in Phases A/B,
    /// like [`COMPLEXITY_OPTIMISM_DIV`].
    pub COMPLEXITY_EVAL_DIV: i64 = 18236;

}
