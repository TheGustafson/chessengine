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
    /// **The eval-frame scale, in 1024ths.** 1024 = 1.0 = exact identity, so
    /// the shipped engine is unchanged and a sweep has a clean control.
    ///
    /// Why this exists (docs/research/eval-frame-scale-model-2026-07-27.md
    /// §10). Our static eval measures at ≈0.58× SF18's internal frame —
    /// three independent estimates, 0.55 from game outcomes and 0.579/0.605
    /// from static eval — while every cp-denominated margin we have sits at
    /// Stockfish's RAW value (razor 465 vs 485, capture-futility base 231 vs
    /// 232, aspiration delta 5 vs 5). So the margins are ~1.7× too large
    /// relative to the eval they are compared against.
    ///
    /// The obvious fix is to scale ~40 margin constants down by 0.58. That
    /// was built and tested and it LOST (global negative; PRUNE-only and
    /// HISTMG both washes). This is the DUAL: scale the eval UP instead and
    /// leave every constant alone. For a pure eval margin the two are
    /// algebraically identical — only the margin/eval ratio matters — but the
    /// dual also repairs the one class the constant-side fix provably cannot.
    ///
    /// That class is material-vs-eval. `PIECE_VALUES` are Stockfish's exactly
    /// (pawn 208) while our eval is 0.58× theirs, so a captured piece weighs
    /// **1.72× more against eval here than in SF**. The error rides on the
    /// piece-value term, which varies by piece, so no margin constant can
    /// absorb it — but scaling the eval restores the proportion in one move.
    ///
    /// Safe where it matters: a positive scalar is monotone, so the net's
    /// RANKING of positions is untouched and the same move is best at every
    /// node. Only the eval's interaction with cp thresholds moves, which is
    /// exactly what is under test. The outer shape's existing TB/mate clamp
    /// keeps a large scale off the decisive bands.
    ///
    /// Theory puts the interesting region near 1764 (= 1/0.58); 1024 is the
    /// control. Reported `cp` scales with it, which is cosmetic — normalise
    /// before reading a score.
    pub EVAL_SCALE: i64 = 1024;

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
