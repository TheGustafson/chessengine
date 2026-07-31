//! Search — the iterative-deepening driver, PVS nodes, and quiescence.
//!
//! Contracts: `design/07-search-core.md` (structure) and
//! `design/10-pruning-extensions.md` (the heuristics). Where reality and the
//! docs disagree, the docs are updated in the same commit (D-process).
//!
//! # Reading order
//!
//! 1. [`Worker::iterative_deepening`] — the ID loop: aspiration windows, the
//!    root move loop, the four-factor soft stop, and the `info` callback seam.
//! 2. [`Worker::search`] — one PVS node, in the Stockfish step order: TT probe,
//!    tablebase-free draw/mate handling, static eval + correction history,
//!    razoring / RFP / null move / ProbCut, IIR, the move loop with singular
//!    extensions and LMR, and the node-exit TT store.
//! 3. [`Worker::qsearch`] — stand-pat, capture/check generation, qsearch futility.
//! 4. [`params`] — every tunable, with its provenance. Read a knob's doc comment
//!    BEFORE changing it: several record experiments that were run and refuted.
//!
//! Supporting modules: [`stack`] (per-ply state, PV lines), [`rootmoves`] (root
//! bookkeeping), `crate::movepick` (staged ordering), `crate::history` (the
//! butterfly/continuation/pawn/capture/correction tables).
//!
//! # Invariants worth knowing before editing
//!
//! * **`Bench:` (D24).** Any change that can affect the tree must state the new
//!   node count from `./target/release/chessengine bench 16 1 6 default depth`;
//!   an intended non-functional change must reproduce the old one bit-for-bit.
//!   `bash tools/check.sh` prints the line to paste.
//! * **The bench pin is BLIND to NNUE work** — it is a default-feature build, so
//!   the whole `sota` path is absent. Anything under `src/eval/nnue/` needs
//!   `CHECK_NNUE_GATE=1 bash tools/check.sh`, which reads the canonical net and
//!   arch from `engines/canonical/`.
//! * **Scores are OUR internal frame**, not Stockfish printed cp — ours is about
//!   2.07x SF's printed value. Never hand-compare; use
//!   `searchopt/tools/uciprobe.py`'s frame helpers.
//! * **Tunables are compile-time constants by default.** They only become
//!   `setoption`-able under `--features tune`; without it every "arm" you set is
//!   silently ignored and all measurements come back identical.
//!
//! # Current state (2026-07-28)
//!
//! Feature-complete relative to the P9 pruning waves: TT, PVS + aspiration,
//! staged movepick with the full history stack, razoring/RFP/NMP/ProbCut/LMP/
//! futility/SEE pruning, LMR with singular (and double/triple) extensions,
//! correction history, and the D30 threat map. The canonical eval is **D48X**
//! (gen-2, 1280 accumulator) — see `engines/canonical/MANIFEST.md`.
//!
//! **Known weakness, and the reason to touch this file:** on the losses that are
//! genuinely ours the engine is *confidently wrong* — it separates the top two
//! root moves by ~55 cp where Stockfish separates them by ~10, i.e. it commits
//! to a losing move rather than failing to discriminate. The right move is
//! present and correctly valued once forced; it is discarded too early. The
//! method, the corpus, and every claim that survived its own controls are in
//! `searchopt/`. Read `searchopt/README.md` before optimizing anything here.

pub mod params;
pub mod rootmoves;
pub mod stack;

pub use crate::history::ContHistId;
pub use rootmoves::{RootMove, RootMoves, build_root_moves, sort_descending};
pub use stack::{PvLine, SS_OFFSET, STACK_SIZE, Stack, init_stack, ss, update_pv};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::board::{Board, DirtyPiece, MakeMoveHooks};
use crate::eval::{EvalContext, evaluate};
use crate::history::Histories;
use crate::movegen::generate_legal;
use crate::movepick::MovePicker;
use crate::moves::{MAX_MOVES, Move, MoveKind, MoveList};
use crate::threads::{Counters, Job, PoolShared, SetupJob};
use crate::timeman::{Limits, TimeManager, TimePersist};
use crate::tt::{Bound, TranspositionTable};
use crate::types::{
    DEPTH_QS, DEPTH_UNSEARCHED, Depth, Key, MAX_PLY, Piece, PieceType, VALUE_DRAW, VALUE_INFINITE,
    VALUE_MATE, VALUE_NONE, VALUE_ZERO, Value, is_decisive, is_loss, is_mate, is_mated, is_valid,
    is_win, mate_in, mated_in, value_from_tt, value_to_tt,
};
use crate::uci::Options;

/// The draw dither (design/07 §3.1 — the one value function search owns):
/// draw-ish returns in the MAIN search are never exactly [`VALUE_DRAW`];
/// they carry a ±1 pseudo-random component from the node counter so
/// repetition lines are not all mutually equal ("3-fold blindness"
/// damping). Evaluates to −1 or +1.
///
/// Qsearch's ordinary draw/max-ply return deliberately uses plain
/// `VALUE_DRAW` instead (SF parity); only its upcoming-repetition path —
/// a later phase — dithers.
#[inline]
pub fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// [`value_draw`] with CONTEMPT: how much worse than equality a draw is judged,
/// from the ROOT side's point of view.
///
/// 2026-07-28, walking loss P2 to its bottom. At the error node ARCHON plays
/// `Rxh7+`, a genuine perpetual worth exactly −1, instead of `Qh3` which mates
/// at d18. That −1 then becomes alpha and STARVES the winning sibling:
/// `searchmoves={Qh3,Rxh7+}` spends 52,407 nodes and finds nothing, while
/// `searchmoves={Qh3}` finds the mate in 260,666 — an order of magnitude of
/// allocation lost to a move worth a draw. Stockfish never plays that blunder
/// at ANY budget from 50k up; we still play it at 1M and need 2M to stop.
///
/// A draw scored slightly below equality cannot set alpha as cheaply, so the
/// winning alternative keeps a usable window. Negamax returns values from the
/// node's own perspective, so the sign alternates with ply parity: at even ply
/// the side to move IS the root side and a draw is worth −contempt to it.
///
/// DEFAULT 0 ⇒ bit-identical to before; only a `--features tune` build moves it.
#[inline]
pub fn value_draw_contempt(nodes: u64, ply: usize) -> Value {
    let c = params::DRAW_CONTEMPT();
    value_draw(nodes) + if ply.is_multiple_of(2) { -c } else { c }
}

/// Map the search/TT bound to the Observation-Store bound (design/16 §3.5.5).
/// The two enums disagree on discriminants; `tt::Bound::None` (only seen when
/// `!tt_hit`, where the field is not meaningful) maps to the store's default.
#[cfg(feature = "datagen")]
fn obs_bound_from_tt(bound: Bound) -> crate::obs::Bound {
    match bound {
        Bound::Lower => crate::obs::Bound::Lower,
        Bound::Upper => crate::obs::Bound::Upper,
        Bound::Exact => crate::obs::Bound::Exact,
        Bound::None => crate::obs::Bound::Exact,
    }
}

/// The reverse-futility margin (design/10 §3.4, node step N10): the
/// static-eval headroom over beta a non-PV node must clear to prune. The
/// per-ply multiplier `min(RFP_MULT_BASE + RFP_MULT_SLOPE·depth,
/// RFP_MULT_CAP)` (docked by [`params::RFP_MULT_NO_TTHIT()`] on a TT miss)
/// scales the margin with depth, then the two trend flags discount it:
/// an *improving* node or a *worsening* opponent both signal the fail-high
/// is real, so less headroom is demanded.
///
/// ```text
/// futility_mult = min(40 + 4·depth, 80) - 20·(!tt_hit)
/// margin        = futility_mult·depth
///               - (2934·improving + 343·opponent_worsening)·futility_mult / 1024
/// ```
///
/// Pure integer arithmetic — division truncates toward zero exactly like
/// the Stockfish source this re-expresses (D27), so the term order is
/// preserved. The Wave-F correction-history widening term
/// (`|correction_value| / 182069`) is deliberately absent until correction
/// history is plumbed into this suite (design/10 §4 Wave F).
#[inline]
fn rfp_margin(
    depth: Depth,
    tt_hit: bool,
    improving: bool,
    opponent_worsening: bool,
    correction_value: i32,
) -> Value {
    let futility_mult = (params::RFP_MULT_BASE() + params::RFP_MULT_SLOPE() * depth)
        .min(params::RFP_MULT_CAP())
        - params::RFP_MULT_NO_TTHIT() * Value::from(!tt_hit);
    futility_mult * depth
        - (params::RFP_IMPROVING() * Value::from(improving)
            + params::RFP_OPP_WORSENING() * Value::from(opponent_worsening))
            * futility_mult
            / params::RFP_TREND_DEN()
        + correction_value.abs() / params::RFP_CORR_DIV()
}

/// The N10 RFP prune margin with the threat-coupled relaxation folded in
/// (design/10 §3.4, Wave-G item 23, D30): the base [`rfp_margin`] minus
/// `RFP_THREAT_MARGIN` when NONE of our pieces is currently threatened — a
/// position with nothing hanging is safe enough to prune a little more
/// readily. The caller supplies the predicate
/// (`(all_threats() & occupancy(us)).is_empty()`, itself pinned by
/// `rfp_threat_coupling_is_disarmed_when_an_own_piece_is_attacked`); keeping
/// the subtraction in a pure function pins its SIGN and MAGNITUDE, so a wiring
/// regression that dropped the term or flipped its sign changes this returned
/// margin and a unit test catches it apart from the make/prune control flow.
#[inline]
fn rfp_prune_margin(
    depth: Depth,
    tt_hit: bool,
    improving: bool,
    opponent_worsening: bool,
    correction_value: i32,
    no_own_threat: bool,
) -> Value {
    rfp_margin(
        depth,
        tt_hit,
        improving,
        opponent_worsening,
        correction_value,
    ) - params::RFP_THREAT_MARGIN() * Value::from(no_own_threat)
}

/// EXP-UNC-1 (search study bet 1): default EMA shift for the running mean value
/// uncertainty `σ̄` (α = 2⁻¹⁰ = 1/1024). Env-overridable via `ARCHON_UNC_EMA_SHIFT`
/// so Gate 2 can sweep it — D33 reviewer B argued a fast α self-centers within a
/// search and subtracts out the cross-phase (endgame-low-σ̂) component, likely the
/// stronger Elo half, so a slower α (shift ≥ 13) or a fixed σ̄ is worth trying.
#[cfg(feature = "sota")]
const UNC_EMA_SHIFT_DEFAULT: i32 = 10;

/// EXP-UNC-1: the position-conditioned RFP/futility margin scale from the value
/// uncertainty `σ̂` relative to its running mean `σ̄` (design: consumer 1). The
/// eval is trusted MORE (tighter margin, prune more — where the Elo lives) where
/// σ̂ is below average, and LESS (looser margin, prune less — safety) where it is
/// above. Returns exactly `1.0` (no change) when disabled (`c1 == 0`), when there
/// is no fresh σ̂ this node (`sigma <= 0`), or before σ̄ has warmed
/// (`sigma_bar <= 0`) — so a variance-less net or the default coefficient is
/// bit-identical. Clamped to [0.7, 1.5] to bound the effect (study formula).
#[cfg(feature = "sota")]
#[inline]
fn unc_margin_scale(sigma: f32, sigma_bar: f32, c1: f32) -> f32 {
    if c1 == 0.0 || sigma <= 0.0 || sigma_bar <= 0.0 {
        return 1.0;
    }
    (1.0 + c1 * (sigma - sigma_bar) / sigma_bar).clamp(0.7, 1.5)
}

/// EXP-UNC-1 consumer 2: the TM soft-stop time scale from the ROOT value
/// uncertainty. Think LONGER (factor > 1) at a sharp/unclear root, SHORTER at a
/// reliable one — ARCHON's TM otherwise has zero position content. Returns
/// exactly `1.0` (bit-identical) when disabled / no root σ̂ / σ̄ unwarmed;
/// clamped to [0.8, 1.25] (design). Narrower bound than the margin scale — time
/// is a scarcer, riskier lever than a prune margin. Returns `f64` to fold into
/// `soft_stop`'s `total_time` without a precision hop.
#[cfg(feature = "sota")]
#[inline]
fn unc_tm_factor(root_sigma: f32, sigma_bar: f32, c2: f32) -> f64 {
    if c2 == 0.0 || root_sigma <= 0.0 || sigma_bar <= 0.0 {
        return 1.0;
    }
    f64::from((1.0 + c2 * (root_sigma - sigma_bar) / sigma_bar).clamp(0.8, 1.25))
}

/// Head-free TM soft-stop **contestedness** scale (2026-07-28, searchopt):
/// scale the clock by how *undecided* the SEARCH's own root score is.
///
/// This is [`wdl_tm_factor`]'s idea without the net dependency. That one reads
/// `|P(win) − P(loss)|` from a WDL sidecar and [`unc_tm_factor`] reads a σ̂
/// head; **D48X carries neither, so both are measurably inert** — with the
/// coefficients forced on, `ARCHON_TM_TRACE` emits byte-identical soft-stop
/// sequences. The settled root score is already in hand at every soft-stop
/// decision, so this factor works on any net and needs no retrain.
///
/// # Why it keys on CONTESTEDNESS and not decisiveness
///
/// The first cut of this was shrink-only on decided roots, mirroring
/// `wdl_tm_factor`. **That has the wrong sign for the measurement that
/// motivated it**, a point the NNUE architecture study made about
/// `ARCHON_WDL_TM_C3` and which applies here identically. Measured at 10+0.1,
/// normalised by each engine's own mean, we under-spend against SF18 at BOTH
/// extremes — 0.559 vs 0.679 when balanced, 0.546 vs 0.819 when clearly winning
/// — while over-spending in the 1.5–3.0 band. Shrinking decided roots pushes
/// the winning band *further* from SF and only reaches the balanced band
/// indirectly, by banking clock for later moves.
///
/// So the factor keys on `contested = 1 − min(|score|, KNEE)/KNEE`, which is 1
/// at a dead-level root and 0 once the score reaches the knee. **The sign of
/// `TM_SCORE_C` then selects the hypothesis** rather than the code assuming
/// one: positive spends more on contested roots (the direct reading of the
/// measurement), negative spends less (the bank-it-for-later reading). Both are
/// reachable from one tunable, so the game decides which is right instead of me.
///
/// Returns exactly `1.0` when disabled (`TM_SCORE_C == 0`, the default) or on a
/// non-finite score, so `total_time` is bit-identical and the four-factor
/// golden tests hold.
#[inline]
fn score_tm_factor(best_value: Value, c_num: i32, knee: Value) -> f64 {
    if c_num == 0 || knee <= 0 || !is_valid(best_value) {
        return 1.0;
    }
    // `saturating_abs` rather than `abs`: a mate score sits near `Value::MIN`
    // and a plain negation would overflow in debug. Saturating is also the
    // right semantics here — a mate IS maximally decided, and clamping to the
    // knee below discards the magnitude anyway.
    let contested = 1.0 - f64::from(best_value.saturating_abs().min(knee)) / f64::from(knee);
    (1.0 + f64::from(c_num) / 1024.0 * contested)
        .clamp(params::TM_SCORE_FLOOR(), params::TM_SCORE_CEIL())
}

/// Multi-head consumer (D41 rich-net, 2026-07-21): TM soft-stop decisiveness
/// scale from the ROOT WDL head. A root the head already calls decided
/// (|P(win) − P(loss)| high) earns LESS clock — bank it for genuinely
/// contested positions. Shrink-only ([0.8, 1.0]): the σ̂ factor above owns
/// "unclear → think longer", so the two heads pull one lever from independent,
/// independently-tunable directions. Returns exactly `1.0` (bit-identical)
/// when disabled (`ARCHON_WDL_TM_C3`=0, the default), when the loaded net has
/// no WDL sidecar, or before a root capture.
#[cfg(feature = "sota")]
#[inline]
fn wdl_tm_factor(root_decisiveness: f32, c3: f32) -> f64 {
    if c3 == 0.0 || root_decisiveness <= 0.0 {
        return 1.0;
    }
    f64::from((1.0 - c3 * root_decisiveness).clamp(0.8, 1.0))
}

/// Feature-gated qsearch decision trace (`--features search-control`,
/// enabled at runtime by `ARCHON_QTRACE=1`).
///
/// Compiled out of every shipped build, so it is a debugger rather than
/// logging: it answers the one question reading the source cannot, namely
/// which filter rejects which move at runtime. Aggregate node counts say we
/// expand ~1.75x Stockfish's qsearch on identical positions with identical
/// windows and verified-identical code; only a per-decision trace can say
/// which branch differs.
#[cfg(feature = "search-control")]
#[inline]
fn qtrace(ply: usize, alpha: Value, beta: Value, stand: Value, mv: &str, verdict: &str) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("ARCHON_QTRACE").is_some()) {
        eprintln!("QT ply={ply} a={alpha} b={beta} sp={stand} {mv:<6} {verdict}");
    }
}

#[cfg(not(feature = "search-control"))]
#[inline(always)]
fn qtrace(_: usize, _: Value, _: Value, _: Value, _: &str, _: &str) {}

/// The late-move-pruning move-count threshold (design/10 §3.5, the move
/// loop's LMP): once the running `move_count` reaches
/// `(LMP_BASE + depth²) / (2 - improving)`, the picker is told to stop
/// emitting quiets ([`MovePicker::skip_quiet_moves`]). The threshold is
/// quadratic in depth — a deeper node trusts its move ordering less, so it
/// tries more moves before movecount pruning bites — and the `improving`
/// flag halves the divisor: a node whose eval is not improving distrusts its
/// late quiets and prunes them roughly twice as early.
///
/// Integer division truncates toward zero exactly like the Stockfish source
/// this re-expresses (D27); the numerator `LMP_BASE + depth²` is always
/// positive, so the truncation is a plain floor.
#[inline]
fn lmp_move_count(depth: Depth, improving: bool) -> i32 {
    // The depth² term carries a coefficient (identity at 1024) because it,
    // not LMP_BASE, is what actually sets this threshold — see LMP_DEPTH_MULT.
    (params::LMP_BASE() + params::scale_1024(depth * depth, params::LMP_DEPTH_MULT()))
        / (params::LMP_IMPROVING_DEN() - i32::from(improving))
}

/// The parent-futility bound (design/10 §3.5, the move loop's quiet branch):
/// the most optimistic value a quiet move could deliver, against which a
/// shallow non-check node writes it off when the bound cannot reach alpha.
///
/// ```text
/// static_eval + PFUT_BASE
///             + PFUT_NO_BESTMOVE       · (no best move found yet)
///             + PFUT_LMRDEPTH_MULT     · lmr_depth
///             + PFUT_EVAL_ABOVE_ALPHA  · (static_eval > alpha)
/// ```
///
/// `lmr_depth` is the predicted post-LMR depth `new_depth - r / 1024` and can
/// be negative at shallow nodes (the `PFUT_LMRDEPTH_MULT · lmr_depth` term
/// then lowers the bound) — pure `i32`, so the term order and truncation match
/// the Stockfish source this re-expresses (D27). The caller supplies the two
/// predicates; keeping the bound a pure function makes its algebra unit-
/// testable (§5) apart from the make/prune control flow.
#[inline]
fn parent_futility_value(
    static_eval: Value,
    no_best_move: bool,
    lmr_depth: i32,
    eval_above_alpha: bool,
) -> Value {
    static_eval
        + params::PFUT_BASE()
        + params::PFUT_NO_BESTMOVE() * Value::from(no_best_move)
        + params::PFUT_LMRDEPTH_MULT() * lmr_depth
        + params::PFUT_EVAL_ABOVE_ALPHA() * Value::from(eval_above_alpha)
}

/// The capture-futility optimistic value (design/10 §3.5, the move loop's
/// capture branch): the most a non-checking capture could plausibly yield —
/// the static eval, a depth-scaled slack, the victim's value, and a
/// capture-history nudge. The capture is pruned when this cannot reach alpha.
///
/// ```text
/// static_eval + CFUT_BASE + CFUT_LMRDEPTH_MULT·lmr_depth
///             + captured_value + CFUT_HIST_MULT·capt_hist / CAPT_HIST_DEN
/// ```
///
/// `captured_value` is 0 for en passant (no piece on `to`) — SF parity; the
/// `CFUT_LMRDEPTH_MULT·lmr_depth` term LOWERS the value at shallow nodes (it
/// can be negative). Pure `i32`, so the term order and truncation match the
/// Stockfish source this re-expresses (D27): `CFUT_HIST_MULT·capt_hist` is
/// formed before the divide, exactly like `131 * captHist / 1024`. Keeping the
/// bound a pure function makes its algebra unit-testable (§5) apart from the
/// make/prune control flow.
#[inline]
fn capture_futility_value(
    static_eval: Value,
    lmr_depth: i32,
    captured_value: Value,
    capt_hist: i32,
) -> Value {
    static_eval
        + params::CFUT_BASE()
        + params::CFUT_LMRDEPTH_MULT() * lmr_depth
        // Class 4: `captured_value` is in PIECE_VALUES (Stockfish's material
        // frame) while `static_eval` is ours, ~0.64x of it. Identity at 1024.
        + params::scale_1024(captured_value, params::CFUT_PIECE_MULT())
        + params::CFUT_HIST_MULT() * capt_hist / params::CAPT_HIST_DEN()
}

/// The capture/check SEE margin (design/10 §3.5): a capture or checking move
/// whose static exchange cannot reach `-margin` is pruned.
///
/// ```text
/// CAP_SEE_DEPTH_MULT·depth + CAP_SEE_HIST_MULT·capt_hist / CAPT_HIST_DEN
/// ```
///
/// Pure `i32`; the capture-history term is formed before the divide, matching
/// the Stockfish source's `captHist * 34 / 1024` truncation (D27).
#[inline]
fn capture_see_margin(depth: Depth, capt_hist: i32) -> Value {
    params::CAP_SEE_DEPTH_MULT() * depth
        + params::CAP_SEE_HIST_MULT() * capt_hist / params::CAPT_HIST_DEN()
}

/// The singular-verification window centre (design/10 §3.6, Wave-D item 13):
/// the exclusion search runs at the zero window `[singular_beta - 1,
/// singular_beta]` with the ttMove excluded, where
///
/// ```text
/// singular_beta = tt_value - (SE_BETA_MULT + SE_BETA_TTPV_MULT·ttpv_non_pv)·depth / SE_BETA_DIV
/// ```
///
/// `ttpv_non_pv` is `ss.tt_pv && !PV`. Pure `i32`; the `(…)·depth` product is
/// formed before the divide, matching the SF truncation this re-expresses
/// (D27). Keeping it a pure function makes the algebra unit-testable (§5)
/// apart from the exclusion-search control flow.
#[inline]
fn singular_verification_beta(tt_value: Value, depth: Depth, ttpv_non_pv: bool) -> Value {
    tt_value
        - (params::SE_BETA_MULT() + params::SE_BETA_TTPV_MULT() * Value::from(ttpv_non_pv)) * depth
            / params::SE_BETA_DIV()
}

/// The singular DOUBLE-extension margin (design/10 §3.6, Wave-D item 13): when
/// the exclusion value falls below `singular_beta - double_margin` the ttMove
/// earns a second extension ply.
///
/// ```text
/// SE_DOUBLE_BASE + SE_DOUBLE_PV·pv - SE_DOUBLE_NON_TTCAP·(!tt_capture)
///                - corr_adj - SE_DOUBLE_TMH_NUM·tmh / SE_DOUBLE_TMH_DEN
///                - SE_DOUBLE_DEEP·deep
/// ```
///
/// `corr_adj = |correction_value| / SE_CORR_DIV` (Wave-F item 21) tightens the
/// margin under eval uncertainty; `tmh` is the tt_move_history scalar; `deep`
/// is `ply > root_depth`. Pure `i32`; the `tmh` term is formed before its
/// divide, matching the SF truncation this re-expresses (D27).
#[inline]
fn singular_double_margin(
    pv: bool,
    tt_capture: bool,
    corr_adj: Value,
    tmh: i32,
    deep: bool,
) -> Value {
    params::SE_DOUBLE_BASE() + params::SE_DOUBLE_PV() * Value::from(pv)
        - params::SE_DOUBLE_NON_TTCAP() * Value::from(!tt_capture)
        - corr_adj
        - params::SE_DOUBLE_TMH_NUM() * tmh / params::SE_DOUBLE_TMH_DEN()
        - params::SE_DOUBLE_DEEP() * Value::from(deep)
}

/// The singular TRIPLE-extension margin (design/10 §3.6, Wave-D item 13): when
/// the exclusion value falls below `singular_beta - triple_margin` the ttMove
/// earns a third extension ply.
///
/// ```text
/// SE_TRIPLE_BASE + SE_TRIPLE_PV·pv - SE_TRIPLE_NON_TTCAP·(!tt_capture)
///                + SE_TRIPLE_TTPV·ttpv - corr_adj - SE_TRIPLE_DEEP·deep
/// ```
///
/// Same `corr_adj`/`deep` as [`singular_double_margin`]; `ttpv` is `ss.tt_pv`.
/// Pure `i32`, so term order and truncation match the SF source (D27).
#[inline]
fn singular_triple_margin(
    pv: bool,
    tt_capture: bool,
    ttpv: bool,
    corr_adj: Value,
    deep: bool,
) -> Value {
    params::SE_TRIPLE_BASE() + params::SE_TRIPLE_PV() * Value::from(pv)
        - params::SE_TRIPLE_NON_TTCAP() * Value::from(!tt_capture)
        + params::SE_TRIPLE_TTPV() * Value::from(ttpv)
        - corr_adj
        - params::SE_TRIPLE_DEEP() * Value::from(deep)
}

/// The continuation-history pruning predicate (design/10 §3.5, the move loop's
/// quiet branch, Wave-C item 11): a quiet whose combined recent history
/// `cont_history = contHist(1)[pc][to] + contHist(2)[pc][to] +
/// pawnHistory[pc][to]` falls below the depth-scaled bar `-QHIST_PRUNE_MULT ·
/// depth` is written off without a search — the offset-1/offset-2 continuation
/// slabs plus the pawn-structure slot say the move has been failing here.
///
/// A LINEAR depth scale (unlike the quadratic quiet-SEE bar): a deeper node
/// demands a proportionally more damning history. Pure `i32`, so the
/// `-4313 · depth` boundary matches the Stockfish source this re-expresses
/// (D27); keeping the predicate a free function makes that boundary
/// unit-testable (§5) apart from the make/prune control flow.
#[inline]
fn continuation_history_prunes(cont_history: i32, depth: Depth) -> bool {
    cont_history < -params::QHIST_PRUNE_MULT() * depth
}

/// The TT prefetch hook (doc 04 §3.10 hook A, doc 08 §3.15 call site 1):
/// wired into [`Worker::do_move`] so every made move issues a prefetch for
/// the child's cluster the moment the child key is final — the DRAM fetch
/// overlaps the rest of `make_move` and the child's node entry, hiding
/// most of the probe's cache-miss latency. Monomorphized away to a single
/// `prefetcht0` instruction; hook B stays a no-op until the history
/// tables (doc 09) want their own prefetches.
struct TtPrefetch<'a> {
    tt: &'a TranspositionTable,
}

impl MakeMoveHooks for TtPrefetch<'_> {
    #[inline]
    fn on_key(&mut self, key: Key) {
        self.tt.prefetch(key);
    }
}

/// Perf study 54 arm `hookc54`: [`TtPrefetch`] plus Hook C. `WANTS_DIRTY` is
/// `true` only on THIS type, so the ProbCut verification probe — which still
/// makes and unmakes through the plain [`TtPrefetch`] — pays nothing, and
/// `NoHooks` (perft, `Board::make_move`) monomorphizes the record away
/// entirely. What it does NOT dodge is the PSQT path: the const cannot see
/// `nnue_active()`, so a classical search builds a record it never reads.
/// That is the cost study 35 §4 named, and it is why the `nnue_active()`
/// guard in `do_move` cannot be pushed down into the make.
#[cfg(feature = "hookc54")]
struct TtPrefetchDirty<'a> {
    tt: &'a TranspositionTable,
    dp: DirtyPiece,
}

#[cfg(feature = "hookc54")]
impl MakeMoveHooks for TtPrefetchDirty<'_> {
    const WANTS_DIRTY: bool = true;

    #[inline]
    fn on_key(&mut self, key: Key) {
        self.tt.prefetch(key);
    }

    #[inline]
    fn on_dirty(&mut self, dp: &DirtyPiece) {
        self.dp = *dp;
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Compile-time node class (design/07 §3.7.1, D17): sealed marker types
/// with associated consts monomorphize `search` into Root/Pv/NonPv
/// instances on stable Rust — the effect of Stockfish's `NodeType`
/// template parameter, with the PV-only code paths (PV-line writes,
/// selDepth) compiled out of NonPV instances.
pub trait NodeType: sealed::Sealed {
    const ROOT: bool;
    const PV: bool;
}

/// The root node: ply 0, always a PV node.
pub enum Root {}
/// An interior PV node: full-window child of a PV node.
pub enum Pv {}
/// A zero-window node (`beta == alpha + 1` — asserted at node entry).
pub enum NonPv {}

impl sealed::Sealed for Root {}
impl sealed::Sealed for Pv {}
impl sealed::Sealed for NonPv {}

impl NodeType for Root {
    const ROOT: bool = true;
    const PV: bool = true;
}
impl NodeType for Pv {
    const ROOT: bool = false;
    const PV: bool = true;
}
impl NodeType for NonPv {
    const ROOT: bool = false;
    const PV: bool = false;
}

/// One completed PV line's report (design/07 §3.5): everything the UCI
/// layer needs to format an `info depth ... pv ...` line (design/12 §3.13),
/// borrowed straight out of the worker's root-move bookkeeping. The UCI
/// layer adds `time`/`nps` itself from `Limits::start_time` and formats
/// `score` (mate/cp split, design/12 §3.14).
#[derive(Clone, Copy)]
pub struct SearchInfo<'a> {
    /// Root iteration depth.
    pub depth: Depth,
    /// Max ply this line reached, counted from 1 (§3.9).
    pub sel_depth: i32,
    /// 1-based MultiPV line index (always 1 in v1, §3.5).
    pub multipv: usize,
    /// The line's `uci_score`: clamped to the aspiration window on a
    /// bound (§3.4). v1 emits one info line per COMPLETED aspiration
    /// loop, whose front move is exact by construction, so the flags
    /// below stay false until fail-high/low progress reports land with
    /// the UCI layer's 3-second info cadence (doc 12 §3.13).
    pub score: Value,
    pub lowerbound: bool,
    pub upperbound: bool,
    /// Nodes searched so far this `go` (single worker until SMP, D23).
    pub nodes: u64,
    /// TT occupancy in permille (`tt.hashfull(0)`, doc 08 §3.10), sampled
    /// once per completed line — UCI-info cadence, never per node.
    pub hashfull: u32,
    /// The line itself; `pv[0]` is the root move.
    pub pv: &'a [Move],
}

/// The type-erased UCI output sink (design/13 §3.3/§3.7 G). The threading
/// layer holds this as `Arc<dyn UciSink>` so the pool stays free of the
/// output stream's `W: Write` type parameter; the UCI layer implements it
/// (formatting lives there, `uci.rs`). Only the MAIN worker ever writes
/// through it, from its own thread, so every method must be internally
/// line-atomic. `Send + Sync` because the handle travels into the search
/// threads and is shared by all of them.
pub trait UciSink: Send + Sync {
    /// One completed `info` line (design/12 §3.13).
    fn emit_info(&self, info: &SearchInfo, elapsed_ms: i64);
    /// An arbitrary already-formatted line (the `info depth 0 score ...`
    /// mate/stalemate report and diagnostics). Written verbatim + newline.
    fn emit_raw(&self, line: &str);
    /// The final block, emitted atomically (design/12 §3.2 step 4): an
    /// optional trailing `info` line whose PV matches `best`, then
    /// `bestmove <best> [ponder <p>]`.
    fn emit_bestmove(&self, tail: Option<(SearchInfo, i64)>, best: Move, ponder: Option<Move>);
    /// `bestmove (none)` — the root had no legal move.
    fn emit_bestmove_none(&self);
}

/// One worker's end-of-search summary, published under its shell for voting
/// (design/13 §3.8). Built from `root_moves[0]` as the last act of a search
/// job, then read by the main worker after every helper has parked.
#[derive(Clone, Debug)]
pub struct SearchReport {
    /// `root_moves[0].pv`; `pv[0]` is the move this thread would play.
    pub pv: Vec<Move>,
    /// `root_moves[0].score` (ply-relative, may be a bound after an abort).
    pub score: Value,
    /// `root_moves[0].uci_score` (window-clamped) — what the winner's
    /// republished `info` line prints.
    pub uci_score: Value,
    pub score_lowerbound: bool,
    pub score_upperbound: bool,
    /// `root_moves[0].average_score` — the running EMA the winner carries into
    /// the next move's TM persist (design/13 §3.7 G; SF's
    /// `bestThread->rootMoves[0].averageScore`).
    pub average_score: Value,
    /// Deepest fully-completed iteration for this thread (a vote weight).
    pub completed_depth: Depth,
    pub sel_depth: i32,
}

impl SearchReport {
    /// A report whose score is decisive AND trusted (not a bound left by an
    /// aborted iteration) — the §3.8 decisive-protection predicate.
    fn is_trusted_decisive(&self) -> bool {
        self.score != -VALUE_INFINITE
            && is_decisive(self.score)
            && !(self.score_lowerbound || self.score_upperbound)
    }
}

/// Best-thread voting (design/13 §3.8; SF `get_best_thread`, re-expressed).
/// Returns the index of the winning report. Callers disable voting (and take
/// thread 0 directly) for a single thread or a fixed-depth `go` (§3.8 step 1).
///
/// Each thread votes for its own best move with weight
/// `score - min_score + VOTE_TEMPO`, so even the weakest thread's opinion
/// counts and identical moves accumulate. A *trusted decisive* incumbent is
/// only displaced by a trusted decisive candidate with strictly larger
/// magnitude (a shorter mate / faster TB win). Against a NON-decisive
/// incumbent, a *trusted decisive* candidate — a proven, non-bound mate/TB
/// result, WIN **or** LOSS — is trusted outright (SF `get_best_thread`'s
/// `newThreadDecisive` short-circuit: an exact proof of a lost position, whose
/// PV is then the longest defence, is more reliable than another thread's fuzzy
/// heuristic score). Only among NON-decisive candidates does the vote decide —
/// more votes, or equal votes and a longer PV (a depth proxy) — and there a
/// losing (`is_loss`) or never-finished (`-VALUE_INFINITE`) candidate can never
/// win.
pub fn best_thread(reports: &[SearchReport]) -> usize {
    use std::collections::HashMap;
    debug_assert!(!reports.is_empty());
    if reports.len() == 1 {
        return 0;
    }
    let min_score = reports.iter().map(|r| r.score).min().unwrap_or(VALUE_DRAW);
    let mut votes: HashMap<Move, i64> = HashMap::new();
    for r in reports {
        if let Some(&mv) = r.pv.first() {
            *votes.entry(mv).or_insert(0) += i64::from(r.score - min_score + params::VOTE_TEMPO);
        }
    }
    let vote_of = |r: &SearchReport| -> i64 {
        r.pv.first()
            .and_then(|m| votes.get(m))
            .copied()
            .unwrap_or(0)
    };
    let mut best = 0;
    for i in 1..reports.len() {
        let b = &reports[best];
        let c = &reports[i];
        let b_dec = b.is_trusted_decisive();
        let c_dec = c.is_trusted_decisive();
        if b_dec {
            // Only a strictly shorter mate / faster win displaces a decisive
            // incumbent; votes are ignored entirely.
            if c_dec && c.score.abs() > b.score.abs() {
                best = i;
            }
        } else if c_dec
            || (!is_loss(c.score)
                && (vote_of(c) > vote_of(b)
                    || (vote_of(c) == vote_of(b) && c.pv.len() > b.pv.len())))
        {
            best = i;
        }
    }
    best
}

/// Elect the winning worker index (design/13 §3.8 step 1). Voting is disabled —
/// thread 0 wins directly — for a `single_thread` pool or a `fixed_depth` `go`,
/// the two cases that must stay bit-reproducible (D24, acceptance criterion 3);
/// otherwise the best-thread vote decides.
pub fn elect_winner(reports: &[SearchReport], single_thread: bool, fixed_depth: bool) -> usize {
    if single_thread || fixed_depth {
        0
    } else {
        best_thread(reports)
    }
}

/// The 1-ply-PV ponder fallback (design/12 §3.2 step 4): make `best` on a
/// scratch board and probe the TT there; offer the stored move only if the
/// resulting position is not an immediate draw and the move is legal (the doc
/// 08 §3.5 validation, discharged by legal-list membership: a collided or
/// garbage move simply matches nothing). Cold path (once per `go`).
pub(crate) fn ponder_from_tt(tt: &TranspositionTable, board: &Board, best: Move) -> Option<Move> {
    let mut b = board.clone();
    b.make_move(best, b.gives_check(best));
    if b.is_draw(1) {
        return None;
    }
    let probe = tt.probe(b.key());
    if !probe.hit || probe.data.mv == Move::NULL {
        return None;
    }
    let mut legal = MoveList::new();
    generate_legal(&b, &mut legal);
    legal.contains(&probe.data.mv).then_some(probe.data.mv)
}

/// One aspiration attempt's window snapshot — test-only instrumentation
/// behind the design/07 §8 criterion-7 trace test. `delta`/`alpha`/
/// `beta`/`failed_high_cnt` are the values the attempt SEARCHED with;
/// `value` is what the root search returned against them.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct AspirationAttempt {
    root_depth: Depth,
    adjusted_depth: Depth,
    delta: Value,
    alpha: Value,
    beta: Value,
    failed_high_cnt: i32,
    value: Value,
}

/// Length of the soft-stop `fallingEval` ring (design/12 §3.9): the scores
/// of the previous four completed iterations. A power of two so the cursor
/// advance is a single mask (`& (ITER_VALUE_RING - 1)`); named once here so
/// the array size and the mask can never silently disagree (doc 12 §8.4).
const ITER_VALUE_RING: usize = 4;
const _: () = assert!(ITER_VALUE_RING.is_power_of_two());

/// Main-worker-only state (design/13 §3.3), gathered so the search code
/// stays branch-light: the null-object payload behind the main worker's
/// `manager: Some(..)` (helpers hold `None`). Holds the per-GAME time state
/// that must survive across `go`s ([`TimePersist`]) and the ponder-hit
/// verdict. Never touched cross-thread — the UCI thread flips only
/// `PoolShared.ponder`, never this (§3.3).
pub struct SearchManager {
    /// Per-game TM state, seeded into the ID loop each `go` and updated at
    /// search end; reset by `ucinewgame` ([`Worker::clear`]).
    pub persist: TimePersist,
    /// `stop_on_ponderhit` (design/12 step 9 / §3.3): set true when the soft
    /// budget elapses *while pondering*, so a later `ponderhit` stops at once
    /// instead of burning the whole clock. Cleared each `go` (`setup`).
    pub stop_on_ponderhit: bool,
}

impl SearchManager {
    pub fn new() -> SearchManager {
        SearchManager {
            persist: TimePersist::default(),
            stop_on_ponderhit: false,
        }
    }
}

impl Default for SearchManager {
    fn default() -> SearchManager {
        SearchManager::new()
    }
}

/// Per-thread search state (design/07 §3.10, design/13 §3.2): one worker owns
/// its board copy (D8, D23), stack, PV table, histories and reduction table —
/// none observable from another thread. The only cross-thread-visible handles
/// are the atomic [`Counters`] and the `Arc<PoolShared>` (flags + sibling
/// shells). Moved into its thread at spawn, never shared (§3.12).
pub struct Worker {
    /// The worker's own board copy, history Vec included (D8).
    pub board: Board,
    /// Root-move bookkeeping (§3.4); populated by the iterative-deepening
    /// driver (plan step 5).
    pub root_moves: RootMoves,
    /// The per-ply stack at offset [`SS_OFFSET`] (§3.2, D18).
    pub stack: Box<[Stack; STACK_SIZE]>,
    /// Per-ply PV lines (§3.3); only PV/Root nodes touch them.
    pub pv_table: Box<[PvLine; MAX_PLY + 1]>,
    /// Made-move counter (§3.9): bumped in [`Worker::do_move`], matching
    /// SF and the `bench` fingerprint (D24). Plain `u64` until SMP lands
    /// (D23), same call sites.
    pub nodes: u64,
    /// Qsearch slice of `nodes` (bumped at the qsearch `do_move` site only).
    /// Diagnostic: read by the env-gated ARCHON_SEARCH_STATS report; a plain
    /// add per qsearch node, no search decision reads it (bench-neutral).
    pub qnodes: u64,
    /// Search-anatomy diagnostics (ARCHON_SEARCH_STATS; plain adds on paths
    /// that already write stack state — bench-neutral, no decision reads them):
    /// main-search beta cutoffs, cutoffs on the FIRST move tried, total moves
    /// tried at cutoff nodes, main-search TT-cutoff returns, net extension
    /// plies granted, aspiration re-searches.
    pub st_fh: u64,
    pub st_fh_first: u64,
    pub st_fh_moves: u64,
    pub st_ttcut: u64,
    pub st_ext: i64,
    pub st_research: u64,
    pub st_lmr_n: u64,
    pub st_lmr_plies: i64,
    pub st_nmp_try: u64,
    pub st_nmp_cut: u64,
    /// Max ply reached on the current PV line, counted from 1 (§3.9).
    pub sel_depth: i32,
    /// Depth of the current iteration (§3.5).
    pub root_depth: Depth,
    /// Current root aspiration window width `beta - alpha`, stored before
    /// every root search — re-searches included (§3.6); feeds doc 10's
    /// reduction formula.
    pub root_delta: Value,
    /// The per-worker log reduction table (design/10 §3.3): `reductions[i]
    /// = int(RED_TABLE_NUM / RED_TABLE_DEN * ln(i))` in 1024ths of a ply,
    /// indexed by *either* depth or move number (`MAX_MOVES = 256`, and
    /// `MAX_PLY < 256`, so depth always fits). Built once at construction —
    /// runtime `f64::ln` is fine there: it runs once, deterministically,
    /// before any search. Read only by [`Worker::reduction`], which stays
    /// unused until Wave-A item 4 wires LMR (design/10 §4).
    reductions: [i32; MAX_MOVES],
    /// Null-move pruning floor (design/10 §3.4 N11): NMP is disabled while
    /// `ply < nmp_min_ply`. Normally 0; a high-depth NMP verification search
    /// raises it for the duration of that re-search so the subtree cannot
    /// recursively null-move its way back to the same (possibly zugzwang)
    /// cutoff, then resets it to 0. Per-worker (D23); reset to 0 each `go`.
    nmp_min_ply: i32,
    /// Per-color eval optimism (design/11 §3.6's consumer, S0): set at
    /// aspiration setup from the root move's average score —
    /// `OPTIMISM_NUM·avg / (|avg| + OPTIMISM_DEN_BASE)` for the root side
    /// to move, negated for the opponent — and passed to every in-search
    /// `evaluate()` call, indexed by the node's side to move. Zero until
    /// the first aspiration setup (bench/tests that bypass it see the
    /// pre-S0 eval).
    optimism: [Value; 2],
    /// MultiPV line currently searched (§3.5; the outer loop index).
    pub pv_idx: usize,
    /// Number of PV lines to search this `go` (design/12 §3.12, step 10):
    /// the `MultiPV` option, set by the session layer before
    /// [`Worker::iterative_deepening`], clamped there to `root_moves.len()`
    /// (min 1). Default 1 keeps single-PV behavior (bench, D24).
    pub multi_pv: usize,
    /// Depth of the last FULLY completed ID iteration (design/12 §3.8;
    /// Reckless robustness bundle D31). 0 until iteration 1 finishes —
    /// [`Worker::check_time`] refuses any hard stop while it is 0 so a real
    /// (if shallow) bestmove is always produced under a tiny budget. Also
    /// the future SMP best-thread vote weight (doc 13).
    pub completed_depth: Depth,
    /// What `go` asked for (design/12 §3.6).
    pub limits: Limits,
    /// Per-`go` time budgets (design/12 §3.7). This is main-worker state
    /// per doc 12 §3.8's `SearchManager` — the UCI layer calls
    /// `tm.init(...)` at the top of every search session (on the search
    /// thread), [`Worker::check_time`] enforces `maximum()` as the hard
    /// clock wall, and [`Worker::soft_stop`] compares `optimum()` between
    /// iterations (v1, doc 12 step 7). `stop_on_ponderhit` and the
    /// soft-stop factor stack are doc 12 steps 8–9 (P10/P11).
    pub tm: TimeManager,
    /// Best PV of the last iteration whose result was trusted (`!stop`)
    /// AND not a forgotten mate, §3.5. Persists across iterations so a
    /// `bestmove` is always printable; [`Worker::root_hygiene`] both
    /// maintains it and rolls back to it.
    pub last_best_move_pv: Vec<Move>,
    /// The previous iteration's PV for the line being searched (SF's
    /// `lastIterationIdxPV`, search.cpp:367): snapshotted from
    /// `root_moves[pv_idx].previous_pv` before every aspiration search, read
    /// by the node-entry `follow_pv` recurrence when
    /// [`params::FOLLOW_PV_MODE`] is on. Empty on iteration 1, so `follow_pv`
    /// marks only the root there — same as SF.
    last_iter_pv: Vec<Move>,
    /// Score belonging to [`Self::last_best_move_pv`] (§3.5).
    pub last_best_move_score: Value,
    /// Iteration at which the best root move last changed — the
    /// time-management stability signal (§3.5; consumed by doc 12 §3.9's
    /// four-factor soft stop, step 8/P10).
    pub last_best_move_depth: Depth,
    /// Soft-stop `fallingEval` ring (design/12 §3.9): the scores of the
    /// previous four iterations. Re-seeded per `go` from
    /// `persist.best_previous_score` (or `VALUE_DRAW` on the first search
    /// of a game), never carried over — so `fallingEval` is well-defined
    /// from iteration 1.
    iter_value: [Value; ITER_VALUE_RING],
    /// Ring cursor into [`Self::iter_value`]; `iter_value[iter_idx]` is the
    /// score from four iterations ago at read time.
    iter_idx: usize,
    /// Working copy of `persist.previous_time_reduction` — the *numerator*
    /// of the soft-stop `reduction` factor (design/12 §3.9). Seeded from
    /// persist at `go` start and held CONSTANT for the whole search (SF's
    /// `mainThread->previousTimeReduction`, `search.cpp:580`): every
    /// iteration divides `1.48 + this` by the current iteration's freshly
    /// recomputed `time_reduction`. The running value that replaces it for
    /// the *next move* is [`Worker::tm_time_reduction`], not this field.
    tm_previous_time_reduction: f64,
    /// The running `time_reduction` (SF's local `timeReduction`,
    /// `search.cpp:279`/`576`/`619`): seeded to
    /// [`crate::timeman::TIME_REDUCTION_INIT`] (1.0) at `go` start,
    /// recomputed by [`Worker::soft_stop`] each iteration, and written back
    /// to `persist.previous_time_reduction` at search end so it becomes the
    /// next move's numerator. Distinct from [`Worker::tm_previous_time_reduction`]
    /// so the numerator stays a per-move constant, not the prior iteration's
    /// value.
    tm_time_reduction: f64,
    /// Working copy of `persist.best_previous_average_score` for the
    /// soft-stop `fallingEval` factor (design/12 §3.9): seeded from persist
    /// at `go` start.
    tm_best_previous_average_score: Value,
    /// Per-thread eval state (design/11 §3.1); empty in Phase A.
    eval_ctx: EvalContext,
    /// EXP-UNC-1: RFP/futility uncertainty coefficient (env `ARCHON_UNC_RFP_C1`,
    /// default 0.0 = off → bit-identical). Read once at Worker construction.
    #[cfg(feature = "sota")]
    unc_rfp_c1: f32,
    /// EXP-UNC-1: running mean value-uncertainty σ̄ (EMA), 0.0 until the first
    /// fresh σ̂. Per-worker; the ratio `(σ̂−σ̄)/σ̄` self-calibrates as it warms.
    #[cfg(feature = "sota")]
    sigma_bar: f32,
    /// EXP-UNC-1 consumer 2: TM soft-stop uncertainty coefficient (env
    /// `ARCHON_UNC_TM_C2`, default 0.0 = off). Read once at construction.
    #[cfg(feature = "sota")]
    unc_tm_c2: f32,
    /// EXP-UNC-1 consumer 2: the ROOT position's value uncertainty σ̂ (ply-0
    /// fresh eval), read by [`Worker::soft_stop`]. 0.0 until a root eval lands.
    #[cfg(feature = "sota")]
    root_sigma: f32,
    /// Multi-head WDL TM consumer: soft-stop decisiveness coefficient (env
    /// `ARCHON_WDL_TM_C3`, default 0.0 = off). Read once at construction.
    #[cfg(feature = "sota")]
    wdl_tm_c3: f32,
    /// Multi-head WDL TM consumer: the ROOT position's decisiveness
    /// |P(win) − P(loss)| from the WDL head (ply-0 fresh eval), read by
    /// [`Worker::soft_stop`]. 0.0 until a root eval lands.
    #[cfg(feature = "sota")]
    root_wdl_d: f32,
    /// EXP-UNC-1: the σ̄ EMA rate α (env `ARCHON_UNC_EMA_SHIFT`, default
    /// [`UNC_EMA_SHIFT_DEFAULT`]). Read once at construction; swept in Gate 2.
    #[cfg(feature = "sota")]
    unc_ema_alpha: f32,
    /// Per-thread move-ordering statistics (design/09 §3.4, D23): butterfly/
    /// low-ply/capture/continuation/pawn/correction history. Read by the
    /// [`MovePicker`] scoring and written by [`Histories::update_all_stats`]
    /// at node exit. `new_search()` decays it each `go`; `ucinewgame` clears
    /// it (uci.rs). ~20 MiB behind a `Box` so it never transits the stack.
    pub histories: Box<Histories>,
    /// Per-search snapshot of the D47 policy head (perf study 13 §4.1).
    ///
    /// The two hot call sites used to reach for
    /// [`crate::eval::nnue::active_policy_head`] directly, which takes a
    /// **process-wide `RwLock` read lock** — two atomic read-modify-writes and
    /// a poison check — at every interior node with no TT move **at depth >= 3**,
    /// and once more per node whenever `CPL_ORD_W` is set (it defaults to 0, so
    /// the measured −4.8 i/n is the first site alone). Nets are
    /// installed once per `go` (`refresh_eval_net`), never mid-search, so the
    /// value is a per-search constant; [`SotaContext`] already snapshots the
    /// same handle at net-set time, and this is that pattern applied to the
    /// search. `None` (the production case, no `.cpol` sidecar) collapses both
    /// sites to one predictable `test`/`jne`.
    ///
    /// [`SotaContext`]: crate::eval::nnue::sota_context::SotaContext
    #[cfg(feature = "sota")]
    policy_head: Option<&'static crate::eval::nnue::policy_head::PolicyHead>,
    /// Poll-cadence bitmask for [`Worker::check_time`] (design/12 §3.8;
    /// Reckless bundle D31): the real clock/limit check runs when `nodes &
    /// mask == mask`. Recomputed per `go` — tighter (a smaller power of
    /// two) under a `go nodes` limit so tiny budgets are honored.
    check_time_mask: u64,

    // ---- Cross-thread-visible handles (design/13 §3.2) ----
    /// 0 = main; feeds the aspiration jitter and gates helper-only behavior.
    pub thread_idx: usize,
    /// This thread's atomic counters (nodes/tb_hits/best_move_changes),
    /// shared with its shell so the pool can read them while the worker runs
    /// (§3.10). Written owner-only; a plain `self.nodes` mirror stays the
    /// deterministic driver of every search decision (D24).
    pub counters: Arc<Counters>,
    /// The shared pool block: control flags (`stop`/`increase_depth`/
    /// `ponder`) and every sibling shell. The main worker (thread 0) uses it
    /// to wake helpers, wait for them, aggregate counters and read reports;
    /// helpers only read the flags (§3.5).
    pub pool: Arc<PoolShared>,

    // ---- Main-thread-only state (None for helpers, design/13 §3.3) ----
    /// `Some` iff this is the main worker (thread 0): the null-object payload
    /// (per-game TM persist + ponder-hit verdict). `is_main()` reads its
    /// presence.
    pub manager: Option<Box<SearchManager>>,
    /// The UCI output sink, installed by `setup`; only the main worker writes
    /// through it. `None` on a detached worker (bench/tests never emit).
    output: Option<Arc<dyn UciSink>>,
    /// The options snapshot for this `go`, installed by `setup`; the main
    /// worker consumes it in `TimeManager::init`. Defaults on a detached
    /// worker.
    options: Options,
    /// `Minimal` report mode (design/12 §3.13): suppress per-iteration `info`
    /// lines. Installed by `setup`; main worker only.
    minimal: bool,

    /// Experiment (design/17 Thrust II): the shared learned move-prior, cloned
    /// into each node's [`MovePicker`]. `None` unless `POLICY_NET` is set; the
    /// default build does not compile the field.
    #[cfg(feature = "policy")]
    policy_net: Option<Arc<crate::policy::PolicyNet>>,

    /// Main-worker-only, bounded observational LMR telemetry. It is compiled
    /// only for search-control experiments and has no authority over search.
    #[cfg(feature = "search-control")]
    search_trace: Option<crate::search_trace::SearchTrace>,
    /// One explicit deterministic replay intervention, never enabled in play.
    #[cfg(feature = "search-control")]
    lmr_intervention: Option<crate::search_intervention::LmrIntervention>,

    /// In-search observation capture (design/16 §3.5.5, D45a): the per-`Worker`
    /// buffer + shard writer. `None` unless `ObsCapturePath` is set on the main
    /// worker at `Threads == 1`. Observation-only — the sample gate is a pure
    /// function of the position key, so capture never perturbs search. Compiled
    /// out of the default build (bench neutrality, §6).
    #[cfg(feature = "datagen")]
    obs_recorder: Option<Box<crate::obs::recorder::ObsRecorder>>,

    /// Per-attempt aspiration trace (§8 criterion 7) — test builds only,
    /// cleared per `go`, appended by [`Worker::aspiration_search`].
    #[cfg(test)]
    asp_trace: Vec<AspirationAttempt>,
}

impl Worker {
    pub fn new(board: Board) -> Worker {
        // `[CONST; N]` array-repeat works for the non-Copy PvLine because
        // the repeated operand is a `const` item, evaluated per element.
        const EMPTY_PV: PvLine = PvLine::new();
        // The log reduction table (design/10 §3.3), built once here. Index 0
        // is never read (depth/move_number are >= 1 at every call site) and
        // stays 0; `ln(1) == 0` makes `reductions[1]` exactly 0. The `as
        // i32` cast truncates toward zero exactly like Stockfish's `int(...)`
        // — since `ln(i) >= 0` for `i >= 1` this is a floor, and the divisor
        // is a power of two so the table is bit-reproducible (D24).
        let mut reductions = [0i32; MAX_MOVES];
        for (i, r) in reductions.iter_mut().enumerate().skip(1) {
            // The additive OFFSET (default 0 — adding literal 0.0 is exact,
            // so the table is bit-identical at identity) is the one new
            // degree of freedom the product form admits; see RED_TABLE_OFFSET.
            *r = (params::RED_TABLE_NUM() as f64 / params::RED_TABLE_DEN() as f64 * (i as f64).ln()
                + params::RED_TABLE_OFFSET() as f64 / params::RED_TABLE_DEN() as f64)
                as i32;
        }
        // A detached worker (bench, unit tests) is its own 1-thread pool:
        // it is `is_main()` (manager present), shares its counters with its
        // shell, and never emits (`output = None`). thread_idx 0 ⇒ no
        // aspiration jitter, so its search is bit-identical to Threads=1
        // (D24).
        let counters = Arc::new(Counters::new());
        let pool = PoolShared::detached(Arc::clone(&counters));
        Worker {
            board,
            root_moves: RootMoves::new(),
            stack: Box::new([Stack::default(); STACK_SIZE]),
            pv_table: Box::new([EMPTY_PV; MAX_PLY + 1]),
            nodes: 0,
            qnodes: 0,
            st_fh: 0,
            st_fh_first: 0,
            st_fh_moves: 0,
            st_ttcut: 0,
            st_ext: 0,
            st_research: 0,
            st_lmr_n: 0,
            st_lmr_plies: 0,
            st_nmp_try: 0,
            st_nmp_cut: 0,
            sel_depth: 0,
            root_depth: 0,
            root_delta: 2 * VALUE_INFINITE,
            reductions,
            nmp_min_ply: 0,
            optimism: [VALUE_ZERO; 2],
            pv_idx: 0,
            multi_pv: 1,
            completed_depth: 0,
            limits: Limits::new(),
            tm: TimeManager::new(),
            last_best_move_pv: Vec::new(),
            last_iter_pv: Vec::new(),
            last_best_move_score: -VALUE_INFINITE,
            last_best_move_depth: 0,
            iter_value: [VALUE_DRAW; ITER_VALUE_RING],
            iter_idx: 0,
            tm_previous_time_reduction: crate::timeman::PREVIOUS_TIME_REDUCTION_INIT,
            tm_time_reduction: crate::timeman::TIME_REDUCTION_INIT,
            tm_best_previous_average_score: VALUE_INFINITE,
            eval_ctx: EvalContext::new(),
            #[cfg(feature = "sota")]
            unc_rfp_c1: std::env::var("ARCHON_UNC_RFP_C1")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            #[cfg(feature = "sota")]
            sigma_bar: 0.0,
            #[cfg(feature = "sota")]
            unc_tm_c2: std::env::var("ARCHON_UNC_TM_C2")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            #[cfg(feature = "sota")]
            root_sigma: 0.0,
            #[cfg(feature = "sota")]
            wdl_tm_c3: std::env::var("ARCHON_WDL_TM_C3")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            #[cfg(feature = "sota")]
            root_wdl_d: 0.0,
            #[cfg(feature = "sota")]
            unc_ema_alpha: 2.0f32.powi(
                -std::env::var("ARCHON_UNC_EMA_SHIFT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(UNC_EMA_SHIFT_DEFAULT),
            ),
            histories: Histories::new(),
            #[cfg(feature = "sota")]
            policy_head: None,
            check_time_mask: params::CHECK_TIME_CALLS as u64 - 1,
            thread_idx: 0,
            counters,
            pool,
            manager: Some(Box::new(SearchManager::new())),
            output: None,
            options: Options::new(),
            minimal: false,
            #[cfg(feature = "policy")]
            policy_net: crate::policy::shared_from_env(),
            #[cfg(feature = "search-control")]
            search_trace: None,
            #[cfg(feature = "search-control")]
            lmr_intervention: None,
            #[cfg(feature = "datagen")]
            obs_recorder: None,
            #[cfg(test)]
            asp_trace: Vec::new(),
        }
    }

    /// Construct a pooled worker on its own thread (design/13 §3.4): `idx`
    /// selects main (0) vs helper, `counters` is this thread's shell's atomic
    /// block, and `pool` is the shared handle. The main worker gets a
    /// [`SearchManager`]; helpers get `None` (the null object). All heavy
    /// tables are allocated here so their pages are first-touch-local (§3.4).
    pub fn new_in_pool(idx: usize, counters: Arc<Counters>, pool: Arc<PoolShared>) -> Worker {
        let mut w = Worker::new(Board::startpos());
        w.thread_idx = idx;
        w.counters = counters;
        w.pool = pool;
        w.manager = if idx == 0 {
            Some(Box::new(SearchManager::new()))
        } else {
            None
        };
        w
    }

    /// True iff this is the main worker (thread 0). The null-object gate
    /// (design/13 §3.3): main-only bookkeeping (TM, `info`/`bestmove`
    /// output, voting) is `if self.is_main()`-guarded; helpers pay one
    /// perfectly-predicted branch and never touch that state. Keyed on the
    /// manager's presence (helpers hold `None`).
    #[inline]
    pub fn is_main(&self) -> bool {
        self.manager.is_some()
    }

    /// `Setup` job (design/13 §3.6 step 4): install this `go`'s per-thread
    /// state on the worker's own thread, and zero the per-`go` counters and
    /// this thread's report slot. `nmp_min_ply`/`root_depth`/... are reset by
    /// [`Worker::iterative_deepening`] itself, so this only covers what the
    /// driver assumes already-set.
    pub fn setup(&mut self, job: SetupJob) {
        self.board = job.board;
        self.limits = job.limits;
        self.options = job.options;
        self.output = Some(job.output);
        self.multi_pv = self.options.multipv().max(1) as usize;
        self.minimal = self.options.minimal();
        self.nodes = 0;
        self.qnodes = 0;
        self.st_fh = 0;
        self.st_fh_first = 0;
        self.st_fh_moves = 0;
        self.st_ttcut = 0;
        self.st_ext = 0;
        self.st_research = 0;
        self.st_lmr_n = 0;
        self.st_lmr_plies = 0;
        self.st_nmp_try = 0;
        self.st_nmp_cut = 0;
        self.counters.reset();
        *self.pool.shells[self.thread_idx]
            .report
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        if let Some(m) = &mut self.manager {
            m.stop_on_ponderhit = false;
        }
    }

    /// `Clear` job (`ucinewgame`, design/13 §3.4): reset the per-thread
    /// histories and the per-game TM state to their new-game defaults.
    pub fn clear(&mut self) {
        self.histories.clear();
        self.tm.new_game();
        if let Some(m) = &mut self.manager {
            m.persist = TimePersist::default();
        }
        // In-search capture (design/16 §3.5.5): `ucinewgame` is the game
        // boundary — flush the buffered episode, if any, before the histories
        // reset begins the next game. Best-effort; an I/O error must not crash
        // the engine (§3.3).
        #[cfg(feature = "datagen")]
        if let Some(rec) = self.obs_recorder.as_mut()
            && let Err(error) = rec.flush_game()
        {
            eprintln!("ARCHON obs capture: game flush failed: {error}");
        }
    }

    /// A `Search` job entry (design/13 §3.4): the main worker orchestrates
    /// (TM, helper wake, output, voting); a helper just deepens until `stop`.
    pub fn start_searching(&mut self, tt: &Arc<TranspositionTable>) {
        // Install this thread's eval from the process-wide active net (the
        // `EvalFile` option, design/11 §3.4.5) — read once per `go` at the
        // search boundary, before iterative deepening refreshes the root
        // accumulator. `None` ⇒ PSQT (default; bench-neutral).
        self.refresh_eval_net();
        #[cfg(feature = "datagen")]
        if self.is_main() {
            self.obs_capture_setup();
        }
        #[cfg(feature = "search-control")]
        {
            self.search_trace = if self.is_main() {
                crate::search_trace::SearchTrace::from_env(self.board.key())
            } else {
                None
            };
            self.lmr_intervention = if self.is_main() {
                match crate::search_intervention::LmrIntervention::from_env() {
                    Ok(intervention) => intervention,
                    Err(error) => {
                        eprintln!("ARCHON LMR intervention disabled: {error}");
                        None
                    }
                }
            } else {
                None
            };
        }
        if self.is_main() {
            self.main_search(tt);
        } else {
            self.helper_search(tt);
        }
        #[cfg(feature = "search-control")]
        if let Some(trace) = self.search_trace.as_mut() {
            if let Err(error) = trace.flush() {
                eprintln!("ARCHON search trace write failed: {error}");
            }
        }
        #[cfg(feature = "search-control")]
        if let Some(intervention) = self.lmr_intervention.as_ref() {
            if let Err(error) = intervention.flush() {
                eprintln!("ARCHON LMR intervention result write failed: {error}");
            }
        }
        // Release this `go`'s output sink so the UCI layer regains sole
        // ownership of the stream once the search is joined (the type-erased
        // `Arc<dyn UciSink>` clone must not outlive the search; a `Setup` for
        // the next `go` reinstalls it).
        self.output = None;
    }

    /// Install this thread's eval from the process-wide active NNUE net
    /// (design/11 §3.4.5), read once at the search boundary. `None` ⇒ PSQT.
    /// Called by `start_searching` (pool path) and by `bench` (direct path)
    /// before iterative deepening; unit tests inject a net into `eval_ctx`
    /// directly to stay independent of the global.
    pub fn refresh_eval_net(&mut self) {
        self.eval_ctx.set_net(crate::eval::nnue::active_net());
        #[cfg(feature = "sota")]
        self.eval_ctx
            .set_sota_net(crate::eval::nnue::active_sota_net());
        #[cfg(feature = "sota")]
        self.eval_ctx
            .set_simple_halfka_net(crate::eval::nnue::active_simple_halfka_net());
        #[cfg(feature = "sota")]
        self.eval_ctx
            .set_protected_halfka_net(crate::eval::nnue::active_protected_halfka_net());
        // Perf study 13 §4.1: snapshot the policy head here, with the nets, so
        // the per-node sites never take the process-wide `RwLock`.
        #[cfg(feature = "sota")]
        {
            self.policy_head = crate::eval::nnue::active_policy_head();
        }
    }

    /// Lazily create the in-search observation recorder (design/16 §3.5.5) when
    /// `ObsCapturePath` is set. Runs once per `go` on the main worker; the
    /// recorder persists across the game's `go`s (idempotent — a second call
    /// with a recorder already open is a no-op). Refuses (with an `info string`)
    /// when `Threads != 1`, keeping single-threaded capture determinism. Silent
    /// on success so fastchess sees no unexpected between-game stdout.
    #[cfg(feature = "datagen")]
    fn obs_capture_setup(&mut self) {
        let path = self.options.obs_capture_path();
        if path.is_empty() || self.obs_recorder.is_some() {
            return; // capture off, or already open
        }
        if self.options.threads() != 1 {
            if let Some(out) = &self.output {
                out.emit_raw(&format!(
                    "info string ObsCapture: requires Threads==1 (have {}); capture disabled",
                    self.options.threads()
                ));
            }
            return;
        }
        let eval_file = self.options.eval_file();
        let net_version = (!eval_file.is_empty()).then(|| eval_file.to_string());
        let search_config = format!(
            "Threads=1 Hash={} EvalFile={} obs[min_depth={} sample_bits={} per_game_cap={}]",
            self.options.hash_mb(),
            if eval_file.is_empty() {
                "<none>"
            } else {
                eval_file
            },
            self.options.obs_capture_min_depth(),
            self.options.obs_capture_sample_bits(),
            self.options.obs_capture_per_game_cap(),
        );
        // Concurrent matches hand every engine process the same option value,
        // and the shard writer is a plain truncating create — suffix the pid
        // so each process owns a distinct shard (obs-tool merges + dedups
        // multi-shard input by canonical key).
        let pid_path = {
            let pid = std::process::id();
            match path.strip_suffix(".ceobs") {
                Some(stem) => format!("{stem}.p{pid}.ceobs"),
                None => format!("{path}.p{pid}.ceobs"),
            }
        };
        let config = crate::obs::recorder::ObsRecorderConfig {
            path: std::path::PathBuf::from(pid_path),
            min_depth: self
                .options
                .obs_capture_min_depth()
                .clamp(0, MAX_PLY as i64) as Depth,
            sample_bits: self.options.obs_capture_sample_bits().clamp(0, 63) as u32,
            per_game_cap: self.options.obs_capture_per_game_cap().max(0) as usize,
            engine_version: format!(
                "ARCHON {} (in-search-capture v0)",
                env!("CARGO_PKG_VERSION")
            ),
            search_config,
            net_version,
            hardware_identity: crate::obs::recorder::hardware_identity(),
        };
        match crate::obs::recorder::ObsRecorder::enable(config) {
            Ok(recorder) => self.obs_recorder = Some(Box::new(recorder)),
            Err(error) => {
                if let Some(out) = &self.output {
                    out.emit_raw(&format!("info string ObsCapture: disabled ({error})"));
                }
            }
        }
    }

    /// Node-entry gate for in-search capture (design/16 §3.5.5). Returns `Some`
    /// only when the recorder is open, the node is deep enough, and its position
    /// key passes the deterministic sample gate with buffer room. When capture
    /// is DISABLED (`obs_recorder` is `None`) this is one predictable, always-not
    /// -taken branch per node — the `?` short-circuit — and reads nothing else.
    #[cfg(feature = "datagen")]
    #[inline]
    fn obs_begin_node(
        &mut self,
        depth: Depth,
        alpha: Value,
        beta: Value,
    ) -> Option<crate::obs::recorder::ObsNodeCapture> {
        let recorder = self.obs_recorder.as_mut()?;
        if depth < recorder.min_depth() {
            return None;
        }
        let sample_key = self.board.raw_key();
        if !recorder.sample_hit(sample_key) {
            return None;
        }
        if !recorder.has_room() {
            recorder.note_dropped();
            return None;
        }
        Some(crate::obs::recorder::ObsNodeCapture::new(
            self.nodes, depth, alpha, beta,
        ))
    }

    /// Main-worker orchestration (design/13 §3.7): TM init + the single TT
    /// generation bump, wake the helpers, run our own iterative deepening
    /// with `info` output, honor the ponder/infinite busy-wait, stop and
    /// collect every worker, vote, and emit the exactly-once `bestmove`.
    fn main_search(&mut self, tt: &Arc<TranspositionTable>) {
        let pool = Arc::clone(&self.pool);
        let n = pool.shells.len();

        // B. TM init + the single per-`go` TT generation bump. `persist`
        // lives in the manager; move it into a local so the borrow checker
        // lets the driver mutate it (the manager box stays present, so
        // `is_main()` is still true throughout).
        let mut persist = std::mem::take(&mut self.manager.as_mut().expect("main").persist);
        let mut limits = self.limits.clone();
        let (us, game_ply) = (self.board.side_to_move(), self.board.game_ply());
        let options = self.options.clone();
        self.tm
            .init(&mut limits, us, game_ply, &options, &mut persist);
        self.limits = limits;
        let start_time = self.limits.start_time;
        // The single per-`go` generation bump, after TM init, before helpers
        // can observe the table (the `submit` mutex publishes it, §3.7 B).
        tt.new_search();

        // C. Wake the helpers; each runs the identical `iterative_deepening`.
        for shell in pool.shells.iter().skip(1) {
            shell.submit(Job::Search { tt: Arc::clone(tt) });
        }

        // Run our own iterative deepening with `info` emission. The closure
        // captures owned handles only (no `self` borrow), so it coexists with
        // the `&mut self` the driver needs.
        let output = self.output.clone();
        let minimal = self.minimal;
        let mut last_sent: Option<Move> = None;
        self.iterative_deepening(tt, &pool.stop, &mut persist, &mut |info| {
            if minimal {
                return;
            }
            if let Some(sink) = output.as_ref() {
                sink.emit_info(info, start_time.elapsed().as_millis() as i64);
            }
            if info.multipv == 1 {
                last_sent = info.pv.first().copied();
            }
        });
        // The per-game persist is restored AFTER voting (below), so its
        // winner-derived seeds can come from the voted thread, not just main.

        // §3.11 post-search `nodestime` accounting (doc 12 step 11): book the
        // nodes consumed minus the increment credit. Gate on `npmsec` (set by
        // `init` only when it converted OUR clock to a node budget).
        if self.limits.npmsec > 0 {
            let inc_nodes = self.limits.inc[us as usize];
            self.tm.advance_nodes_time(self.nodes as i64 - inc_nodes);
        }

        // Publish our final node count for the aggregate before the epilogue.
        self.counters.nodes.store(self.nodes, Ordering::Relaxed);

        // D. Never answer early: while pondering or infinite, spin until
        // `stop`/`ponderhit` (design/13 §3.7 D). Burns one core for the
        // (typically tiny) window — SF makes the same trade.
        while !pool.stop.load(Ordering::Relaxed)
            && (pool.ponder.load(Ordering::Relaxed) || self.limits.infinite)
        {
            std::thread::yield_now();
        }
        // E. Stop and collect: signal every worker, then wait for the helpers
        // to park — the mutex hand-off publishes their reports/root scores.
        pool.stop.store(true, Ordering::Relaxed);
        for shell in pool.shells.iter().skip(1) {
            shell.wait_idle();
        }

        // Gather reports: index 0 is ours (built inline), 1.. from the shells
        // (each helper published exactly one before parking).
        let mut reports: Vec<SearchReport> = Vec::with_capacity(n);
        reports.push(self.build_report());
        for shell in pool.shells.iter().skip(1) {
            if let Some(r) = shell
                .report
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
            {
                reports.push(r);
            }
        }

        // F. Vote (§3.8) — disabled under a single thread or fixed-depth
        // `go` (bench reproducibility, D24), where thread 0 wins directly.
        let winner = elect_winner(&reports, n == 1, self.limits.depth.is_some());

        // Seed the next `go`'s TM persist from the VOTED winner (design/13
        // §3.7 G; SF search.cpp: `bestPreviousScore/AverageScore =
        // bestThread->rootMoves[0]...`). `iterative_deepening` already filled
        // `persist` from OUR (main's) last iteration; overwrite only the two
        // winner-derived seeds when a helper won. Under fixed-depth `go` the
        // winner is forced to 0, so this never fires there — matching SF's
        // "only when `limits.depth` unset" and keeping Threads=1 bit-identical
        // (D24). The seeds feed only next move's aspiration/TM inputs, never
        // the played move.
        if winner != 0 {
            persist.best_previous_score = reports[winner].score;
            persist.best_previous_average_score = reports[winner].average_score;
        }
        self.manager.as_mut().expect("main").persist = persist;

        // G. Emit the exactly-once final block.
        self.emit_bestmove(tt, &pool, &reports[winner], winner, last_sent, start_time);
    }

    /// A helper worker (design/13 §3.7): run the identical iterative deepening
    /// until `stop`, then publish the node count and a [`SearchReport`] for
    /// the main worker's vote. Helpers never emit output or evaluate time.
    fn helper_search(&mut self, tt: &Arc<TranspositionTable>) {
        let pool = Arc::clone(&self.pool);
        // Helpers do not persist TM state across `go`s.
        let mut persist = TimePersist::default();
        self.iterative_deepening(tt, &pool.stop, &mut persist, &mut |_| {});
        self.counters.nodes.store(self.nodes, Ordering::Relaxed);
        self.publish_report();
    }

    /// Build this worker's end-of-search [`SearchReport`] from `root_moves[0]`
    /// (design/13 §3.8). An empty root move list (mate/stalemate) reports an
    /// empty PV and a `-VALUE_INFINITE` score, which voting never elects.
    fn build_report(&self) -> SearchReport {
        match self.root_moves.first() {
            Some(rm) => SearchReport {
                pv: rm.pv.clone(),
                score: rm.score,
                uci_score: rm.uci_score,
                score_lowerbound: rm.score_lowerbound,
                score_upperbound: rm.score_upperbound,
                average_score: rm.average_score,
                completed_depth: self.completed_depth,
                sel_depth: rm.sel_depth,
            },
            None => SearchReport {
                pv: Vec::new(),
                score: -VALUE_INFINITE,
                uci_score: -VALUE_INFINITE,
                score_lowerbound: false,
                score_upperbound: false,
                average_score: -VALUE_INFINITE,
                completed_depth: 0,
                sel_depth: 0,
            },
        }
    }

    /// Publish this worker's report into its shell for the main worker's vote
    /// (design/13 §3.8). Helper worker only; the main worker builds its report
    /// inline.
    fn publish_report(&self) {
        let report = self.build_report();
        *self.pool.shells[self.thread_idx]
            .report
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(report);
    }

    /// Emit the winning worker's final block (design/13 §3.7 G): republish an
    /// `info` line when the GUI's last-seen PV differs from the played move
    /// (always, when the winner is a helper), then the exactly-once
    /// `bestmove [ponder]`. `bestmove (none)` when the root had no legal move.
    fn emit_bestmove(
        &self,
        tt: &TranspositionTable,
        pool: &PoolShared,
        winner: &SearchReport,
        winner_idx: usize,
        last_sent: Option<Move>,
        start_time: std::time::Instant,
    ) {
        let Some(sink) = self.output.as_ref() else {
            return; // a detached worker never emits
        };
        let Some(best) = winner.pv.first().copied() else {
            // Mate/stalemate at the root (design/12 §3.2 step 2).
            let score = if self.board.checkers().is_empty() {
                "cp 0"
            } else {
                "mate 0"
            };
            sink.emit_raw(&format!("info depth 0 score {score}"));
            sink.emit_bestmove_none();
            return;
        };
        let elapsed_ms = start_time.elapsed().as_millis() as i64;
        // Republish the winner's line if the GUI has not seen it as the front
        // move (a mid-iteration promotion, or any helper win), and it has a
        // trusted score to show.
        let need_tail =
            (winner_idx != 0 || last_sent != Some(best)) && winner.uci_score != -VALUE_INFINITE;
        let tail = need_tail.then(|| {
            (
                SearchInfo {
                    // The winner's own deepest completed iteration — so depth,
                    // sel_depth, score and PV all describe the SAME thread
                    // (design/13 §3.7 G; SF prints the winner's completedDepth).
                    // At Threads=1 the winner is thread 0, so this equals main's
                    // completed depth (info-only, bench-neutral, D24).
                    depth: winner.completed_depth.max(1),
                    sel_depth: winner.sel_depth,
                    multipv: 1,
                    score: winner.uci_score,
                    lowerbound: winner.score_lowerbound,
                    upperbound: winner.score_upperbound,
                    nodes: pool.nodes_searched(),
                    hashfull: tt.hashfull(0),
                    pv: &winner.pv,
                },
                elapsed_ms,
            )
        });
        // Ponder = pv[1] when the line is long enough; else probe the TT after
        // playing `best`.
        let ponder = winner
            .pv
            .get(1)
            .copied()
            .or_else(|| ponder_from_tt(tt, &self.board, best));
        sink.emit_bestmove(tail, best, ponder);
    }

    /// The iterative-deepening driver (design/07 §3.5), v1 scope: the
    /// per-`go` entry point for every search mode. Builds the root move
    /// list (with the `searchmoves` filter, design/12 §3.6), then deepens
    /// one aspirated root search at a time ([`Worker::aspiration_search`],
    /// §3.6) with the per-iteration bookkeeping this phase owns:
    /// previous-score snapshots, stable re-sorting, lastBestMove tracking
    /// and the instability accumulator.
    ///
    /// `tt` is the shared transposition table, borrowed through the search
    /// entry (doc 07 §3.10). The session layer owns the `Arc<TT>` handle
    /// for the job's duration and has already bumped `tt.new_search()`
    /// once for this `go` (doc 13 §3.7 B) — the driver never bumps it, so
    /// helpers can share one generation when SMP lands (D23).
    ///
    /// `stop` is the §3.9 stop seam: the UCI layer flips it from another
    /// thread ("stop"/"quit"), `check_time` flips it on hard limits; a
    /// search that observes it returns untrusted values, so every consumer
    /// here is `!stop`-guarded. The flag rides the call stack as a shared
    /// borrow — the Rust-safe form of SF's global.
    ///
    /// `on_info` is called once per completed PV line (never after a
    /// stop); the UCI layer formats it into an `info` line (design/12
    /// §3.13), so `go depth N` emits monotone depth reports for free. A
    /// `&mut dyn FnMut` trait object: the callback fires once per
    /// iteration (cold), and dynamic dispatch keeps `Worker` free of a
    /// type parameter.
    ///
    /// On return, `root_moves[0]` holds the best move/PV/score — trusted
    /// even after a mid-iteration stop, because root bookkeeping only
    /// publishes completed subtree results (§3.9) and the stable sort
    /// keeps stale-but-real scores ordered (§3.4). With no legal moves
    /// (`root_moves` empty) it returns immediately; the UCI layer emits
    /// `info depth 0` + `bestmove (none)` itself (design/12 §3.2 step 2).
    pub fn iterative_deepening(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        persist: &mut TimePersist,
        on_info: &mut dyn FnMut(&SearchInfo),
    ) {
        // Per-`go` reset (§3.2.4, §3.9): stack, counters, root moves.
        init_stack(&mut self.stack);
        self.nodes = 0;
        self.qnodes = 0;
        self.st_fh = 0;
        self.st_fh_first = 0;
        self.st_fh_moves = 0;
        self.st_ttcut = 0;
        self.st_ext = 0;
        self.st_research = 0;
        self.st_lmr_n = 0;
        self.st_lmr_plies = 0;
        self.st_nmp_try = 0;
        self.st_nmp_cut = 0;
        self.sel_depth = 0;
        self.root_depth = 0;
        self.completed_depth = 0;
        self.pv_idx = 0;
        // The cross-thread counters (design/13 §3.10) are reset here so a
        // reused worker on the DIRECT path (bench/tests, which never call
        // `setup`) starts clean; the pool path also zeroes them in `setup`.
        // `nodes` mirrors `self.nodes` and must start at 0 so `nodes_searched`
        // is accurate from the first `info` line, before the `do_move` mirror
        // catches up.
        self.counters.nodes.store(0, Ordering::Relaxed);
        self.counters.best_move_changes.store(0, Ordering::Relaxed);
        self.nmp_min_ply = 0;
        self.check_time_mask = self.compute_check_time_mask();
        self.last_best_move_pv.clear();
        self.last_best_move_score = -VALUE_INFINITE;
        self.last_best_move_depth = 0;
        // TUNE-BUILD ONLY: rebuild the reductions table from the live
        // tunables at search start. RED_TABLE_NUM/DEN bake in at Worker
        // construction, which predates setoption in fastchess/SPSA drivers —
        // the 2026-07-26 shapeA screen came back a PERFECT MIRROR (Ptnml
        // [0,0,100,0,0]) because the option silently never applied. Release
        // builds keep the constructed table (zero cost, D24-identical).
        #[cfg(feature = "tune")]
        {
            for (i, r) in self.reductions.iter_mut().enumerate().skip(1) {
                *r = (params::RED_TABLE_NUM() as f64 / params::RED_TABLE_DEN() as f64
                    * (i as f64).ln()
                    + params::RED_TABLE_OFFSET() as f64 / params::RED_TABLE_DEN() as f64)
                    as i32;
            }
        }
        // Soft-stop factor state (doc 12 §3.9, step 8). `increase_depth`
        // starts `true` so fixed-depth/fixed-node search never damps its
        // depth (bench-neutral, D24). The `iter_value` ring is seeded per
        // `go` from persist (never carried over), and the persist-derived
        // factor inputs are latched into working copies here.
        //
        // Only the MAIN worker writes `increase_depth` (design/13 §3.7): the
        // pool already stores `true` in `start_thinking` before any search
        // begins, so a helper re-storing it here could clobber a soft-stop
        // verdict main published earlier this `go`. Detached/bench workers are
        // `is_main()`, so their needed per-`go` reset is preserved on the
        // direct path (bench-neutral, D24).
        if self.is_main() {
            self.pool.increase_depth.store(true, Ordering::Relaxed);
        }
        self.iter_idx = 0;
        let iter_seed = if persist.best_previous_score != VALUE_INFINITE {
            persist.best_previous_score
        } else {
            VALUE_DRAW // first search of the game (§3.9)
        };
        self.iter_value = [iter_seed; ITER_VALUE_RING];
        self.tm_previous_time_reduction = persist.previous_time_reduction;
        // The running time_reduction seeds to 1.0 per `go` (SF search.cpp:279),
        // NOT to the persisted numerator: if the factor block never runs this
        // move, next move's numerator falls back to 1.0.
        self.tm_time_reduction = crate::timeman::TIME_REDUCTION_INIT;
        self.tm_best_previous_average_score = persist.best_previous_average_score;
        #[cfg(test)]
        self.asp_trace.clear();
        self.root_moves = build_root_moves(&self.board, &self.limits.searchmoves);
        if self.root_moves.is_empty() {
            return; // mate/stalemate at the root: nothing to search
        }
        // Refresh before the optional root-policy consumer. Frozen-classic
        // heads reuse this protected accumulator and have no board-refresh
        // trunk of their own.
        self.eval_ctx.refresh_root(&self.board);
        // EXP-UNC-1 (D33 review, reviewers A/B/C): reset the root uncertainty
        // per search. The persistent TT means the root usually hits a cached
        // eval on move N+1, so a fresh root σ̂ isn't recaptured every search;
        // without this reset `root_sigma` would leak a PRIOR move's value into
        // the TM soft-stop. Worst case now: 0.0 → `unc_tm_factor` returns 1.0
        // (consumer 2 self-disables that search) rather than mis-scaling on a
        // different position's uncertainty.
        #[cfg(feature = "sota")]
        {
            self.root_sigma = 0.0;
            // Same leak guard for the WDL decisiveness: a TT-cached root eval
            // on move N+1 would otherwise leave move N's |P(w)−P(l)| scaling
            // move N+1's clock. 0.0 → wdl_tm_factor self-disables this search.
            self.root_wdl_d = 0.0;
        }
        #[cfg(feature = "policy")]
        if crate::policy::root_order_enabled() {
            let scores = self.policy_net.as_ref().and_then(|net| {
                if net.uses_classic_accumulator() {
                    self.eval_ctx
                        .classic_policy_accumulators(&self.board, 0)
                        .and_then(|(stm, ntm, activation)| {
                            net.scores_from_classic(stm, ntm, activation, self.board.side_to_move())
                        })
                } else {
                    net.scores(&self.board)
                }
            });
            if let Some(scores) = scores {
                for root_move in &mut self.root_moves {
                    let mv = root_move.pv[0];
                    root_move.policy_score =
                        (scores.move_score(mv.from(), mv.to()) * 1024.0) as i32;
                }
                sort_descending(&mut self.root_moves);
            }
        }
        // Per-`go` history refresh (design/09 §3.4): decay the butterfly
        // history toward zero so recent searches dominate, and refill the
        // low-ply history. Correction/capture/continuation tables persist
        // untouched across the game (cleared only by `ucinewgame`).
        self.histories.new_search();
        // NNUE root accumulator (design/11 §3.4.4): refresh `stack[0]` from the
        // root position, both perspectives, once per search — the root stays
        // valid regardless of how much game history UCI replayed. Refreshes only
        // the already-installed net (the active net is installed at the search
        // boundary — `start_searching`/`bench` — not here, so unit tests driving
        // `iterative_deepening` directly stay on PSQT); a no-op when NNUE is off.
        // MultiPV (doc 12 §3.12, step 10): the number of PV lines to
        // report, the option clamped to the legal-move count (min 1). At
        // MultiPV 1 the loop below is exactly single-PV (bench, D24).
        let multi_pv = self.multi_pv.clamp(1, self.root_moves.len());
        // Time-management instability input (doc 12 §3.9): halved every
        // iteration, harvested from `best_move_changes`, consumed by the
        // four-factor soft stop.
        let mut tot_best_move_changes = 0.0f64;
        // TM-driven re-search damping (doc 12 §3.9): bumped at the top of
        // an iteration when the previous soft stop said "don't deepen"
        // (`increase_depth == false`). Stays 0 whenever the soft stop never
        // runs (fixed depth/nodes), so `3 * (0 + 1) / 4 == 0` leaves the
        // §3.6 adjusted-depth formula untouched (bench-neutral, D24).
        let mut search_again_counter: i32 = 0;

        while self.root_depth + 1 < MAX_PLY as Depth
            && !stop.load(Ordering::Relaxed)
            // `go depth d` (design/12 §3.6): stop iterating once reached — but
            // only on the MAIN worker (SF search.cpp gates the depth cap on
            // `mainThread`; design/13 §3.7 has helpers ignore `limits.depth`).
            // Helpers keep deepening past `d` to enrich the shared TT and are
            // halted by main's step-E `stop`; voting is disabled under
            // `go depth` so thread 0's move is chosen regardless. Vacuous at
            // Threads=1 (`is_main()` always true → the cap always applies,
            // bench-neutral, D24).
            && (!self.is_main() || self.limits.depth.is_none_or(|d| self.root_depth < d))
        {
            self.root_depth += 1;
            // §3.9 search-again: a soft budget more than half spent leaves
            // `increase_depth == false`, so this iteration re-searches near
            // the same effective depth (refreshing PV/TT) instead of
            // starting a deeper pass that would be aborted mid-way. The flag
            // is shared (design/13 §3.7): helpers read the main worker's
            // verdict; each worker keeps its own `search_again_counter`.
            if !self.pool.increase_depth.load(Ordering::Relaxed) {
                search_again_counter += 1;
            }
            // Age the instability metric (SF: /2). Our divisor is 2.5 — the
            // depth-redenomination of 2026-07-26: we run ~3.3 fewer
            // iterations per search than SF at equal time, so SF's /2 leaves
            // the same root flip ~10× hotter at our stop decision (rationale
            // at params::INSTABILITY_CHANGES_DECAY).
            tot_best_move_changes /= params::INSTABILITY_CHANGES_DECAY();

            // Snapshot last iteration (§3.5).
            for (idx, rm) in self.root_moves.iter_mut().enumerate() {
                rm.previous_score = rm.score;
                rm.previous_pv.clone_from(&rm.pv);
                rm.previous_score_exact = idx < multi_pv;
            }

            // SF's `bestValue`: the value the LAST MultiPV line's root
            // search settled on (reassigned every line, so after the loop it
            // holds the pv_idx == multi_pv-1 line's return). Feeds the
            // soft-stop `fallingEval` + `iter_value` ring. Identical to
            // `root_moves[0].score` at MultiPV 1 (bench-neutral); the weakest
            // line at MultiPV > 1, matching search.cpp.
            let mut root_best_value = -VALUE_INFINITE;
            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;
                self.sel_depth = 0; // per PV line (§3.9)
                // SF search.cpp:367 — arm the follow-PV recurrence with THIS
                // line's previous-iteration PV. Unconditional (the snapshot is
                // inert while FOLLOW_PV_MODE is 0, whose gate lives at the
                // node-entry write).
                self.last_iter_pv
                    .clone_from(&self.root_moves[pv_idx].previous_pv);
                root_best_value = self.aspiration_search(tt, stop, search_again_counter);
                // Order the completed lines among themselves (§3.5) — a
                // no-op at MultiPV 1, but the shape is load-bearing for
                // the MultiPV loop.
                sort_descending(&mut self.root_moves[..=pv_idx]);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // §3.13 emission: below the node throttle publish the block
                // only when the iteration's LAST line finishes; above it,
                // after every line. Either way republish exactly the lines
                // already searched THIS iteration (`0..=pv_idx`, matching
                // SF's `i <= pvIdx`) — a higher-index line still holds last
                // iteration's score/pv, so surfacing it under the current
                // depth label would make the block non-monotonic. On the
                // last line `0..=pv_idx == 0..multi_pv`, the full coherent
                // block; at MultiPV 1 a single line per iteration (unchanged,
                // bench-neutral).
                // Helpers pass a no-op `on_info` and never emit, so skip the
                // whole block for them — building a `SearchInfo` eagerly calls
                // `pool.nodes_searched()` (a Σ over all shells) and samples
                // `tt.hashfull` per root line per iteration, side-effecting
                // reads the compiler cannot elide behind the empty callback.
                // Gating on `is_main` costs the (perfectly predicted) branch and
                // is bench-neutral: the detached bench worker and the Threads=1
                // pool main are both `is_main`, so their emission is unchanged.
                if self.is_main()
                    && (pv_idx + 1 == multi_pv || self.nodes > params::INFO_THROTTLE_NODES())
                {
                    for line in 0..=pv_idx {
                        let rm = &self.root_moves[line];
                        on_info(&SearchInfo {
                            depth: self.root_depth,
                            sel_depth: rm.sel_depth,
                            multipv: line + 1,
                            score: rm.uci_score,
                            lowerbound: rm.score_lowerbound,
                            upperbound: rm.score_upperbound,
                            // Aggregate across all threads (design/13 §3.10);
                            // equals `self.nodes` at Threads=1 (bench-neutral).
                            nodes: self.pool.nodes_searched(),
                            hashfull: tt.hashfull(0),
                            pv: &rm.pv,
                        });
                    }
                }
            }

            // lastBestMove tracking with the forgotten-mate guard, plus
            // the aborted-loss repair (§3.5, plan step 10).
            self.root_hygiene(stop.load(Ordering::Relaxed));

            // Mark the iteration fully completed (design/12 §3.8): unblocks
            // the `check_time` hard stops (the `completed_depth == 0`
            // guard) now that a real bestmove exists at this depth.
            if !stop.load(Ordering::Relaxed) {
                self.completed_depth = self.root_depth;
            }

            // Harvest the per-iteration instability signal across ALL threads
            // (design/13 §3.10): only the main worker aggregates, summing and
            // resetting every thread's atomic counter with `swap(0)` so no
            // count is lost. Helpers never reach `soft_stop`, so their local
            // `tot_best_move_changes` stays 0 (unused). At Threads=1 this is
            // exactly the main worker's own count (bench-neutral, D24).
            if self.is_main() {
                let sum: u64 = self
                    .pool
                    .shells
                    .iter()
                    .map(|s| s.counters.best_move_changes.swap(0, Ordering::Relaxed))
                    .sum();
                tot_best_move_changes += sum as f64;
            }

            // `go mate n` (§3.5; doc 12 §3.9's mate-limit stop,
            // independent of TM): a COMPLETED iteration whose front score
            // proves a mate — ours or theirs — within n full moves ends
            // the search. `2` converts the UCI limit's full moves to the
            // mate score's plies (structural, not tunable).
            if !stop.load(Ordering::Relaxed)
                && self.limits.mate.is_some_and(|n| {
                    let s = self.root_moves[0].score;
                    (is_mate(s) || is_mated(s)) && VALUE_MATE - s.abs() <= 2 * n
                })
            {
                stop.store(true, Ordering::Relaxed);
            }

            // TM soft stop between completed iterations (doc 12 §3.9, full
            // four-factor form — doc 12 step 8). Sets `increase_depth` for
            // the next iteration and maintains the `iter_value` ring.
            self.soft_stop(stop, multi_pv, tot_best_move_changes, root_best_value);
        }
        // Persist the per-game TM state for the next `go` of this game
        // (design/12 §3.7/§3.9): the front move's score and running average
        // seed next move's `iter_value` and `fallingEval`; this move's
        // `time_reduction` becomes next move's `previous_time_reduction`.
        // Only a truly completed iteration (real, non-sentinel score) may
        // overwrite the seeds — an aborted depth-0 search leaves them.
        if let Some(front) = self.root_moves.first().filter(|_| self.completed_depth > 0) {
            persist.best_previous_score = front.score;
            persist.best_previous_average_score = front.average_score;
        }
        // SF search.cpp:619 `previousTimeReduction = timeReduction`: the
        // running value (this move's last iteration, or the 1.0 seed if the
        // factor block never ran) becomes next move's numerator.
        persist.previous_time_reduction = self.tm_time_reduction;

        // Diagnostic node anatomy (env-gated, off by default => bench-neutral;
        // 2026-07-26 search-gap investigation): one stderr line per completed
        // search splitting `nodes` into main-search vs qsearch makes.
        if std::env::var_os("ARCHON_SEARCH_STATS").is_some() {
            eprintln!(
                "SSTATS nodes={} qnodes={} qshare={:.3} depth={} fh={} fhf={} \
                 fhf_rate={:.3} avg_moves_at_cut={:.2} ttcut={} ext={} research={}",
                self.nodes,
                self.qnodes,
                self.qnodes as f64 / self.nodes.max(1) as f64,
                self.completed_depth,
                self.st_fh,
                self.st_fh_first,
                self.st_fh_first as f64 / self.st_fh.max(1) as f64,
                self.st_fh_moves as f64 / self.st_fh.max(1) as f64,
                self.st_ttcut,
                self.st_ext,
                self.st_research
            );
            eprintln!(
                "SSTATS2 lmr_n={} lmr_avg_plies={:.3} nmp_try={} nmp_cut={} nmp_rate={:.3}",
                self.st_lmr_n,
                self.st_lmr_plies as f64 / self.st_lmr_n.max(1) as f64,
                self.st_nmp_try,
                self.st_nmp_cut,
                self.st_nmp_cut as f64 / self.st_nmp_try.max(1) as f64
            );
        }
    }

    /// End-of-iteration root hygiene (design/07 §3.5, plan step 10):
    /// lastBestMove tracking with the forgotten-mate guard, then the
    /// rollback/repair pass. Called once per ID iteration, after the
    /// MultiPV loop; `stopped` is the stop flag as observed there.
    /// Requires a non-empty `root_moves` (the driver returns before its
    /// first iteration otherwise).
    ///
    /// - **Forgotten mate**: once an iteration proves `mate n`, a later
    ///   iteration returning a weaker (smaller-magnitude) or bound score
    ///   — aspiration instability, pruning noise — must not replace the
    ///   proven line: the tracking refuses the update and the front root
    ///   move is rolled back to the remembered PV/score.
    /// - **Aborted loss**: an *exact* mated/TB-loss score for the whole
    ///   root from a search stopped mid-iteration is untrusted —
    ///   unsearched root moves might refute or delay it. Restore the
    ///   previous iteration's move/score, or (no previous iteration) mark
    ///   the score as a lower bound so the GUI never sees a fake forced
    ///   loss (doc 12 §3.12). The `pv_idx == 0` guard keeps an aborted
    ///   LATER MultiPV line from leapfrogging a proven loss in an earlier
    ///   line (structural now; exercised when MultiPV lands, doc 12 step
    ///   10).
    fn root_hygiene(&mut self, stopped: bool) {
        let front = &self.root_moves[0];
        let forgotten_mate = self.last_best_move_score != -VALUE_INFINITE
            && (is_mate(self.last_best_move_score) || is_mated(self.last_best_move_score))
            && (front.score.abs() < self.last_best_move_score.abs() || front.score_is_bound());

        if !stopped {
            // The TM stability signal: the iteration at which the best
            // move last changed (§3.5, doc 12 §3.9).
            if self.last_best_move_pv.first() != front.pv.first() {
                self.last_best_move_depth = self.root_depth;
            }
            // Never replace a proven (shorter) mate with a weaker or
            // bound re-score.
            if !forgotten_mate {
                self.last_best_move_pv.clone_from(&self.root_moves[0].pv);
                self.last_best_move_score = self.root_moves[0].score;
            }
        }

        let aborted_loss = stopped && self.pv_idx == 0 && self.root_moves[0].score_is_exact_loss();
        if aborted_loss || (self.root_moves[0].score != -VALUE_INFINITE && forgotten_mate) {
            if let Some(&remembered) = self.last_best_move_pv.first() {
                // Move the remembered line's root move to the front (the
                // rotate preserves the others' relative order — the next
                // iteration's move ordering, §3.4) and republish it.
                let idx = self
                    .root_moves
                    .iter()
                    .position(|rm| rm.pv[0] == remembered)
                    .expect("the remembered best move is a root move");
                self.root_moves[..=idx].rotate_right(1);
                let rm = &mut self.root_moves[0];
                rm.score = self.last_best_move_score;
                rm.uci_score = self.last_best_move_score;
                rm.pv.clone_from(&self.last_best_move_pv);
                rm.score_lowerbound = false;
                rm.score_upperbound = false;
            } else if aborted_loss {
                // Aborted depth-1 search: nothing to restore — label the
                // loss as a lower bound (unsearched moves may beat it).
                self.root_moves[0].score_lowerbound = true;
            }
        }
    }

    /// The aspiration loop (design/07 §3.6): one iteration's root search
    /// for the current PV line, run in a window centered on that line's
    /// `average_score` and re-searched with a widening window until the
    /// value lands strictly inside it (or the search is stopped).
    ///
    /// Mechanics, in doc order (constants in [`params`], D20):
    ///
    /// - Initial half-width `delta = 5 + |mean_squared_score| / 10588`
    ///   (i64 EMA, §3.4). First iterations start from the
    ///   `-VALUE_INFINITE²` sentinel, so `|mss|/10588 ≈ 96_700` and the
    ///   window clamps to `(-INF, INF)` — depth 1 is effectively a
    ///   full-window search, no special-casing. The lazy-SMP diversity
    ///   term `+ thread_idx % 8` is vacuous single-threaded and joins
    ///   with doc 13 (P11); the eval-doc hook
    ///   `optimism = OPTIMISM_NUM·avg / (|avg| + OPTIMISM_DEN_BASE)` is
    ///   ATTACHED (S0, 2026-07-26) — see the setup below and design/11 §3.6.
    /// - `adjusted_depth` shrinks by one ply per CONSECUTIVE fail-high
    ///   (fail-highs are usually resolved by move ordering, not depth —
    ///   this caps re-search cost) and by the TM's search-again damping
    ///   (doc 12 §3.9; counter 0 until its producer lands).
    /// - FAIL LOW (`best <= alpha`): the window slides down — beta drops
    ///   exactly to the old alpha, alpha backs off delta below the
    ///   returned value — and `failed_high_cnt` resets, restoring the
    ///   full re-search depth. TM seam: a root fail low must also revoke
    ///   `stop_on_ponderhit` (the intended move may be refuted — the one
    ///   case where burning past the soft budget after a ponderhit is
    ///   clearly right); the flag lands with doc 12 step 9 (P9) and the
    ///   revoke goes on this branch then.
    /// - FAIL HIGH (`best >= beta`): alpha backs off at most to the old
    ///   `beta - delta` and never below its current value — the window
    ///   stays tight from below — while beta chases the returned value.
    /// - Every re-search widens `delta += 44·delta/128` (~×1.34).
    ///
    /// `self.root_delta` is stored before EVERY root search, re-searches
    /// included — doc 10's reduction formula divides by it (§3.6).
    ///
    /// Returns the raw value the root search settled on (SF's `bestValue`).
    /// The MultiPV driver keeps the LAST line's return as the soft-stop
    /// `fallingEval` input, exactly as SF does — identical to
    /// `root_moves[0].score` at MultiPV 1, the weakest line at MultiPV > 1.
    fn aspiration_search(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        search_again_counter: i32,
    ) -> Value {
        let rm = &self.root_moves[self.pv_idx];
        // Lazy-SMP aspiration jitter (design/13 §3.1): the `thread_idx % 8`
        // term makes helpers search slightly different windows so they diverge
        // and seed the shared TT differently. Vacuous at `thread_idx == 0`
        // (main / detached / bench), keeping Threads=1 bit-identical (D24).
        let mut delta = (params::ASPIRATION_BASE_DELTA() as i64
            + (self.thread_idx % params::ASPIRATION_JITTER_PERIOD()) as i64
            + rm.mean_squared_score.abs() / params::ASPIRATION_MSS_DIVISOR())
            as Value;
        let avg = rm.average_score;
        // Optimism (design/11 §3.6's consumer, S0): lean the eval a few cp
        // toward the side the search already favors, by root average score.
        // SF-parity shape incl. the unguarded first-iteration sentinel
        // (avg = -VALUE_INFINITE saturates to ≈ -OPTIMISM_NUM; iteration 1
        // is full-window and self-corrects). Priors 75/45 = SF's 137/81
        // scaled by the measured winner-slope ~0.55 (our engine frame is
        // ~2× compressed vs SF's reported cp; loss_forensics 2026-07-26);
        // S2 SPSA owns the refinement.
        let us = self.board.side_to_move();
        let opt = params::OPTIMISM_NUM() * avg / (avg.abs() + params::OPTIMISM_DEN_BASE());
        self.optimism[us as usize] = opt;
        self.optimism[us.flip() as usize] = -opt;
        let mut alpha = (avg - delta).max(-VALUE_INFINITE);
        let mut beta = (avg + delta).min(VALUE_INFINITE);
        let mut failed_high_cnt: i32 = 0;
        // Assigned at the top of every loop iteration before any break, so
        // definite-assignment holds at the post-loop return.
        let mut best;

        loop {
            let adjusted_depth = (self.root_depth
                - params::ASPIRATION_FAILHIGH_DEPTH_MULT() * failed_high_cnt
                - params::ASPIRATION_SEARCH_AGAIN_NUM() * (search_again_counter + 1)
                    / params::ASPIRATION_SEARCH_AGAIN_DEN())
            .max(1);
            self.root_delta = beta - alpha;
            best = self.search::<Root>(tt, stop, alpha, beta, adjusted_depth, false, 0);

            // Stable re-sort per attempt (§3.4): the searched PV move
            // floats to the front, everything else keeps last iteration's
            // order. Even on stop the results still sort safely — they
            // refer to the previous iteration.
            sort_descending(&mut self.root_moves[self.pv_idx..]);

            #[cfg(test)]
            self.asp_trace.push(AspirationAttempt {
                root_depth: self.root_depth,
                adjusted_depth,
                delta,
                alpha,
                beta,
                failed_high_cnt,
                value: best,
            });

            if stop.load(Ordering::Relaxed) {
                break;
            }

            if best <= alpha {
                // FAIL LOW: slide the window down; the old alpha caps it.
                beta = alpha;
                alpha = (best - delta).max(-VALUE_INFINITE);
                failed_high_cnt = 0;
                self.st_research += 1;
                // A root fail-low may mean the pondered move is refuted;
                // burning past the soft budget to find a replacement is right,
                // so revoke any pending ponderhit stop (design/12 step 9).
                if let Some(m) = &mut self.manager {
                    m.stop_on_ponderhit = false;
                }
            } else if best >= beta {
                // FAIL HIGH: alpha never loosens past its current value.
                alpha = (beta - delta).max(alpha);
                beta = (best + delta).min(VALUE_INFINITE);
                failed_high_cnt += 1;
                self.st_research += 1;
            } else {
                break; // inside the window: done
            }

            // Widen ~×1.34 per re-search. The cap is semantics-preserving:
            // once delta >= 2·VALUE_INFINITE both window edges clamp to
            // ±INF for every representable score and further growth is
            // inert — while unbounded ×1.34 growth would eventually
            // overflow the i32 in a long enough re-search storm.
            delta = (delta
                + params::ASPIRATION_WIDEN_NUM() * delta / params::ASPIRATION_WIDEN_DEN())
            .min(2 * VALUE_INFINITE);
        }
        best
    }

    /// The best root line of the current `go` (`root_moves[0].pv`; empty
    /// if the root has no legal moves). Trusted even after a stop (§3.5).
    pub fn root_pv(&self) -> &[Move] {
        self.root_moves.first().map_or(&[][..], |rm| &rm.pv)
    }

    /// The best root move of the current `go`, if any. Always printable
    /// once `iterative_deepening` returns with legal moves present: the
    /// root move list exists before any search happens (§3.9).
    pub fn best_move(&self) -> Option<Move> {
        self.root_pv().first().copied()
    }

    /// The main search node (design/07 §3.7.2), v1 subset: node order
    /// steps 2/3/4/5/6/7/9/10/11/12/13/14, plus the first ⟨D10⟩ pruning
    /// heuristics (step 8, design/10 §3.4): N9 razoring, N10 reverse futility
    /// pruning and N11 null move pruning with its verification scheme, all in
    /// the not-in-check branch after the TT cutoff (LMP/singular remain doc
    /// 10's later Wave-B/D items). Step 12's equal-value dither and step
    /// 14's fail-high softening landed with plan step 10. Step 10 is the
    /// PVS three-call structure (plan step 7) with reductions pinned at
    /// r = 0 — LMR is doc 10's (P9). Zero-window scouts are NonPV and
    /// never write PV lines, so `pv_table[1]` stays fresh for every root
    /// move's step-12 PV rebuild (§3.3), and the step-7 NonPV TT cutoff
    /// below is live: zero-window subtrees may resolve straight from the
    /// table — the source of the step-7 node-count drop the Bench:
    /// trailer records (D24).
    // The parameter list is the doc 07 §3.7.1 node signature plus the two
    // plumbed-through borrows (`tt` per doc 07 §3.10, `stop` per §3.9) —
    // SF's search carries the same state; bundling would only rename it.
    #[allow(clippy::too_many_arguments)]
    fn search<NT: NodeType>(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        mut alpha: Value,
        mut beta: Value,
        mut depth: Depth,
        cut_node: bool,
        ply: usize,
    ) -> Value {
        // Step 1 — upcoming repetition (design/10 item 19 "N1", SF parity).
        // If a draw is available in one move and we are currently worse than
        // a draw, the draw is a LOWER BOUND on this node: raise alpha to it,
        // and if that already beats beta the subtree needs no search at all.
        // Cheap enough to sit above the qsearch dive (two table probes, no
        // move generation) and placed there deliberately — SF runs it as the
        // very first statement of search(), before the depth-0 boundary, so
        // shallow nodes get the cutoff too.
        if !NT::ROOT
            && self.board.rule50_count() >= 3
            && alpha < VALUE_DRAW
            && self.board.upcoming_repetition(ply as i32)
        {
            alpha = value_draw_contempt(self.nodes, ply);
            if alpha >= beta {
                return alpha;
            }
        }

        // Depth-0 boundary: drop into qsearch, preserving PV-ness.
        if depth <= 0 {
            return if NT::PV {
                self.qsearch::<Pv>(tt, stop, alpha, beta, ply)
            } else {
                self.qsearch::<NonPv>(tt, stop, alpha, beta, ply)
            };
        }
        // N0 — depth clamp: singular/hindsight extensions can overshoot the
        // stack (design/10 §3.4). Bounds the child's ply below MAX_PLY.
        depth = depth.min(MAX_PLY as Depth - 1);

        // Node-entry invariants (§3.7.1).
        debug_assert!(!(NT::PV && cut_node), "PV nodes are never cut nodes");
        debug_assert!(NT::PV || alpha == beta - 1, "NonPV requires a zero window");
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(NT::ROOT == (ply == 0));

        // Step 2 — init. `cur ± k` indexing stays in bounds for all
        // ply <= MAX_PLY (§3.2.1).
        let cur = ss(ply);
        let in_check = !self.board.checkers().is_empty();
        self.stack[cur].in_check = in_check;
        self.stack[cur].move_count = 0;
        self.stack[cur].stat_score = 0;
        // followPV (SF search.cpp:752-755, toggled by FOLLOW_PV_MODE): this
        // node sits on the previous iteration's PV path iff its parent did
        // AND the parent's move was that PV's move at the parent's ply. Read
        // by N12's IIR exemption and the quiet-pruning PV-path exemption.
        // At mode 0 the flag stays `false` everywhere — bit-identical to the
        // pre-toggle tree (the P9 item-9 parked state).
        self.stack[cur].follow_pv = params::FOLLOW_PV_MODE() != 0
            && (NT::ROOT
                || (self.stack[cur - 1].follow_pv
                    && ply - 1 < self.last_iter_pv.len()
                    && self.stack[cur - 1].current_move == self.last_iter_pv[ply - 1]));
        // Parent-move facts (design/10 §3.4 N2): `prev_sq` is the parent's
        // destination (None after a null move / at the root), `prior_capture`
        // whether the parent's move captured. Read by N6b's TT-cutoff parent
        // penalty and N8's eval-diff nudge.
        let prev_move = self.stack[cur - 1].current_move;
        let prev_sq = prev_move.is_ok().then(|| prev_move.to());
        let prior_capture = self.board.captured_piece().is_some();
        // The UNSCALED correction sum (design/09 §3.10, Wave-F item 21): its
        // `/131072` is the eval delta, and the pruning couplings below divide
        // this raw value by their own big divisors. Computed once per node.
        let correction_value = self
            .histories
            .correction_value(&self.board, &self.stack[..], cur);
        // Consumed-and-reset slot (design/10 §3.4 N6 hindsight): capture the
        // parent's LMR reduction, then reset it so it cannot go stale.
        let prior_reduction = self.stack[cur - 1].reduction;
        self.stack[cur - 1].reduction = 0;
        self.stack[cur + 2].cutoff_cnt = 0;
        if NT::PV {
            self.pv_table[ply].clear();
            self.sel_depth = self.sel_depth.max(ply as i32 + 1);
        }
        // §3.9 cadence: every main-search node entry; qsearch relies on
        // these checks above it. (`follow_pv` semantics arrive with doc 10.)
        self.check_time(stop);

        if !NT::ROOT {
            // Step 3 — abort/draw/max-ply. A stop observed here returns
            // the (untrusted) dithered draw; the parent's step-11 check
            // discards it.
            if stop.load(Ordering::Relaxed) || self.board.is_draw(ply as i32) || ply >= MAX_PLY {
                return if ply >= MAX_PLY && !in_check {
                    evaluate(
                        &self.board,
                        &mut self.eval_ctx,
                        self.optimism[self.board.side_to_move() as usize],
                        ply,
                    )
                } else {
                    value_draw_contempt(self.nodes, ply)
                };
            }

            // Step 4 — mate-distance pruning: even an immediate mate here
            // cannot beat a shorter mate already found higher up.
            alpha = alpha.max(mated_in(ply as i32));
            beta = beta.min(mate_in(ply as i32 + 1));
            if alpha >= beta {
                return alpha;
            }
        }

        // Step 5 — TT probe (doc 08 §3.5 API). The single read-side
        // conversion boundary: a hit's value goes through `value_from_tt`
        // (§3.12) — with the 50-move downgrade — before any use; a
        // tag-matched-but-unoccupied entry normalizes to VALUE_NONE here.
        // The ttMove needs no pseudo_legal/legal validation in this phase:
        // it is only ever matched against `generate_legal`'s output in
        // [`order_moves`], which subsumes the doc 08 §3.5 contract.
        let key = self.board.key();
        let generation = tt.generation(); // once per node (doc 08 §3.9)
        let probe = tt.probe(key);
        let (tt_hit, tt_data) = (probe.hit, probe.data);
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply as i32, self.board.rule50_count() as i32)
        } else {
            VALUE_NONE
        };
        self.stack[cur].tt_hit = tt_hit;
        // Sticky "was ever on the PV" flag (stored back at node exit). Inside a
        // singular verification (excluded_move set) keep the enclosing node's
        // flag — this is the SAME ply (design/10 §3.4 N5).
        if self.stack[cur].excluded_move == Move::NULL {
            self.stack[cur].tt_pv = NT::PV || (tt_hit && tt_data.is_pv);
        }
        // A capturing ttMove does not veto RFP (N10) and shapes the singular
        // extension margins (§3.6); a quiet one is a standing reason to search.
        let tt_capture = tt_data.mv != Move::NULL && self.board.capture_stage(tt_data.mv);

        // Steps 6+7 — the static-eval slot, then the TT cutoff (doc 07
        // deliberately orders the cutoff AFTER the eval slot so doc 10's
        // hindsight depth adjustments can feed it). In check both are
        // skipped: the node borrows the grandparent's eval and jumps
        // straight to the move loop — no pruning happens in check.
        let raw_eval;
        // EXP-UNC-1: this node's value uncertainty σ̂ (0.0 = none this node);
        // set on a fresh sota eval, read at the RFP margin below.
        #[cfg(feature = "sota")]
        let mut node_sigma: f32 = 0.0;
        // `improving` is hoisted out of the not-in-check branch: doc 10's LMR
        // reduction (below, in the move loop) reads it even when this node is
        // in check — where SF pins it `false` (the pre-check eval bears no
        // useful trend). RFP/NMP still consume the computed value inside the
        // branch.
        let mut improving;
        if in_check {
            raw_eval = VALUE_NONE;
            self.stack[cur].static_eval = self.stack[cur - 2].static_eval;
            improving = false;
        } else {
            // Step 6 — eval. A singular verification (excluded_move set) REUSES
            // the enclosing node's corrected static eval — same ply — and
            // stores no eval-cache entry (design/10 §3.6 / §3.4 N6 excluded
            // path). Otherwise: reuse a valid cached RAW eval (D22) if present,
            // else call evaluate(); a TT miss caches the raw eval at
            // DEPTH_UNSEARCHED so transpositions skip the eval (§3.11).
            if self.stack[cur].excluded_move != Move::NULL {
                // `raw_eval` is DEAD on this path — every TT store of it is
                // `excluded_move == NULL`-gated — and `ss.static_eval` holds
                // the enclosing node's CORRECTED eval, which must never be
                // mislabeled as raw (D22). The sentinel keeps any future
                // un-gated store loud instead of silently corrupting the TT
                // eval field (D33 review round 1, personas A + C).
                raw_eval = VALUE_NONE;
            } else {
                raw_eval = if tt_hit && is_valid(tt_data.eval) {
                    tt_data.eval
                } else {
                    evaluate(
                        &self.board,
                        &mut self.eval_ctx,
                        self.optimism[self.board.side_to_move() as usize],
                        ply,
                    )
                };
                // EXP-UNC-1: on a FRESH sota eval, capture this node's value
                // uncertainty σ̂ and fold it into the running mean σ̄. Env-gated —
                // skipped entirely unless a consumer coefficient (ARCHON_UNC_RFP_C1
                // for the RFP margin, ARCHON_UNC_TM_C2 for the TM soft-stop) is set,
                // so zero overhead and bit-identical by default. `last_sigma()` is
                // invalidated at the top of `evaluate()`, so a material-draw /
                // non-SOTA node yields None here (no stale fold) — D33 review.
                #[cfg(feature = "sota")]
                if (self.unc_rfp_c1 != 0.0 || self.unc_tm_c2 != 0.0)
                    && !(tt_hit && is_valid(tt_data.eval))
                    && let Some(s) = self.eval_ctx.last_sigma()
                {
                    node_sigma = s;
                    self.sigma_bar = if self.sigma_bar > 0.0 {
                        self.sigma_bar + (s - self.sigma_bar) * self.unc_ema_alpha
                    } else {
                        s
                    };
                    // Consumer 2 reads the ROOT position's uncertainty in soft_stop.
                    if ply == 0 {
                        self.root_sigma = s;
                    }
                }
                // Multi-head WDL TM consumer: capture the root decisiveness from
                // the WDL head's fresh ply-0 eval (same freshness rules as σ̂:
                // `last_wdl()` is invalidated at the top of `evaluate()`, so a
                // TT-served root never folds a stale distribution).
                #[cfg(feature = "sota")]
                if self.wdl_tm_c3 != 0.0
                    && ply == 0
                    && !(tt_hit && is_valid(tt_data.eval))
                    && let Some([l, _d, w]) = self.eval_ctx.last_wdl()
                {
                    self.root_wdl_d = (w - l).abs();
                }
                // S1 (design/10 N6, D22): `ss.static_eval` is the CORRECTED
                // eval — raw plus the correction-history delta — and every
                // downstream consumer (improving/opponent_worsening, hindsight,
                // razoring/RFP/NMP margins, the step-9 error signal) reads it.
                // The TT keeps the RAW eval (both stores below): corrections
                // are per-path (continuation keys), so a cached corrected
                // value would smuggle one path's correction into another.
                self.stack[cur].static_eval =
                    crate::history::to_corrected_static_eval(raw_eval, correction_value);
                if !tt_hit {
                    probe.writer.write(
                        key,
                        VALUE_NONE,
                        self.stack[cur].tt_pv,
                        Bound::None,
                        DEPTH_UNSEARCHED,
                        Move::NULL,
                        raw_eval,
                        generation,
                    );
                }
            }

            // `improving`: our eval beats our eval two plies ago (our last
            // turn). Uses `ss.static_eval` (never the substituted `eval`); an
            // unset predecessor is the large `VALUE_NONE` sentinel (SF parity).
            // Pinned false in check (set above).
            improving = self.stack[cur].static_eval > self.stack[cur - 2].static_eval;
        }

        // N6 trend flag + hindsight depth adjustments (design/10 §3.11):
        // ordered BEFORE the N7 in-check jump, so they run for ALL nodes —
        // including in check, where `ss.static_eval` is the grandparent's eval.
        // `opponent_worsening`: our eval beats the negation of the parent's —
        // the position is turning our way. A hard parent reduction
        // (`>= HINDSIGHT_RED_UP`) that did NOT turn the position the opponent's
        // way looks unjustified — deepen the whole node; extending a
        // hard-reduced check evasion is exactly where this pays, so it must not
        // be gated out of check. A moderate reduction with a healthy combined
        // eval sum — shallow it.
        let opponent_worsening = self.stack[cur].static_eval > -self.stack[cur - 1].static_eval;
        if prior_reduction >= params::HINDSIGHT_RED_UP() && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= params::HINDSIGHT_RED_DOWN()
            && depth >= 2
            && self.stack[cur].static_eval + self.stack[cur - 1].static_eval
                > params::HINDSIGHT_EVAL_SUM()
        {
            depth -= 1;
        }

        // N6b — the NonPV TT cutoff runs for ALL nodes, in check or not
        // (design/10 §3.11 / SF search.cpp: the early TT cutoff precedes the
        // `if (ss->inCheck) goto moves_loop`). A TT cutoff keys only on the
        // stored value/depth/bound, never on the static eval, so it is exactly
        // as sound in check as out of it — and in-check nodes are common, so
        // gating it out silently searches whole evasion subtrees SF prunes.
        // The child-TT verification's make/unmake is safe in check: it
        // re-checks `pseudo_legal && legal` on the ttMove first.
        {
            // N6b — the NonPV TT cutoff (design/10 §3.4): enough stored depth
            // (one ply of slack for upper-bound values), a valid value, the
            // bound pointing the cutoff's way, the cutNode-agreement guard at
            // shallow depth (item 20), the rule-50 guard (§3.12), and — deep —
            // a child-TT re-verification. Live since plan step 7's zero-window
            // scouts create NonPV nodes.
            //
            // The rule-50 graph-history guard (§3.12) wraps only the cutoff
            // RETURN, not the N6b move-ordering updates: SF (search.cpp) runs
            // the quiet-ttMove bonus / parent malus whenever the base cutoff
            // shape holds and skips only the `return` in the near-50-move
            // regime. `tt_cut_base` is the shared skeleton both the cutoff and
            // the item-20 wrong-direction penalize build on, so the two
            // predicates cannot silently drift.
            let tt_cut_base = !NT::PV
                && self.stack[cur].excluded_move == Move::NULL
                && is_valid(tt_value)
                && tt_data.depth
                    > depth - params::TTCUT_BOUND_SLACK() * Depth::from(tt_value <= beta);
            let base_cutoff = tt_cut_base
                && (if tt_value >= beta {
                    tt_data.bound.includes_lower()
                } else {
                    tt_data.bound.includes_upper()
                })
                && (cut_node == (tt_value >= beta) || depth > params::TTCUT_AGREE_DEPTH());
            if base_cutoff {
                // N6b extras (design/10 §3.4, Wave-C item 12): a beta-side
                // cutoff means the stored ttMove "worked" here. These run
                // regardless of rule50 — SF gates only the cutoff return.
                let ttm = tt_data.mv;
                if ttm != Move::NULL && tt_value >= beta {
                    let us = self.board.side_to_move();
                    // Reward a quiet ttMove. The from-square guard keeps a rare
                    // colliding TT move from mis-indexing the histories.
                    if !self.board.capture_stage(ttm)
                        && self
                            .board
                            .piece_on(ttm.from())
                            .is_some_and(|p| p.color == us)
                    {
                        let bonus =
                            (params::TTCUT_BONUS_MULT() * depth).min(params::TTCUT_BONUS_CAP());
                        self.histories.update_quiet_histories(
                            &self.board,
                            &self.stack[..],
                            cur,
                            ttm,
                            bonus,
                        );
                    }
                    // Penalize the parent's early non-capturing quiet.
                    if let Some(sq) = prev_sq
                        && self.stack[cur - 1].move_count < params::TTCUT_PARENT_MOVECOUNT()
                        && !prior_capture
                        && let Some(pc) = self.board.piece_on(sq)
                    {
                        self.histories.update_continuation_histories(
                            &self.stack[..],
                            cur - 1,
                            pc,
                            sq,
                            params::TTCUT_PARENT_MALUS(),
                        );
                    }
                }
                // The cutoff RETURN itself is gated by the rule-50 guard
                // (§3.12); at rule50 >= limit the node falls through to a real
                // search even though the bonus/malus above already fired.
                if (self.board.rule50_count() as i32) < params::TT_CUTOFF_RULE50_LIMIT() {
                    // N6b child-TT verification (item 20): a deep cutoff on a
                    // non-decisive value re-probes the ttMove's child entry —
                    // only an invalid child value or one AGREEING with the
                    // cutoff direction confirms it. A contradicting child falls
                    // through to a real search.
                    if depth >= params::TTCUT_VERIFY_DEPTH()
                        && !is_decisive(tt_value)
                        && tt_data.mv != Move::NULL
                        && self.board.pseudo_legal(tt_data.mv)
                        && self.board.legal(tt_data.mv)
                    {
                        let m = tt_data.mv;
                        // Board-only make/unmake (SF `pos.do_move`,
                        // search.cpp): this verification probe is NOT a search
                        // node, so it must NOT bump `self.nodes` (which would
                        // inflate the bench fingerprint, the `go nodes` budget,
                        // and node-based TM) nor stamp the search stack. The
                        // prefetch hook still rides along so the immediate child
                        // probe below is warm.
                        let gc = self.board.gives_check(m);
                        // No NNUE accumulator push here: this probe only reads
                        // the child's TT entry and never `evaluate`s it, and it
                        // unmakes immediately — the next real `do_move` at this
                        // node re-pushes and invalidates `stack[ply + 1]`
                        // (design/11 §3.4.4), so the accumulator stays correct.
                        self.board.make_move_with(m, gc, &mut TtPrefetch { tt });
                        let child = tt.probe(self.board.key());
                        let child_value = if child.hit {
                            value_from_tt(
                                child.data.value,
                                ply as i32 + 1,
                                self.board.rule50_count() as i32,
                            )
                        } else {
                            VALUE_NONE
                        };
                        self.board.unmake_move(m);
                        if !is_valid(child_value) || (tt_value >= beta) == (-child_value >= beta) {
                            self.st_ttcut += 1;
                            return tt_value;
                        }
                    } else {
                        self.st_ttcut += 1;
                        return tt_value;
                    }
                }
            } else if tt_cut_base
                && tt_data.bound != Bound::Exact
                && (if tt_value >= beta {
                    tt_data.bound.includes_upper()
                } else {
                    tt_data.bound.includes_lower()
                })
                && depth > params::TTCUT_PENALIZE_DEPTH()
            {
                // Wrong-direction bound (item 20): the only reason the cutoff
                // failed is a bound pointing the wrong way — age the mismatching
                // entry out faster.
                probe.writer.penalize(1);
            }
        }

        // Steps N8–N13 (design/10 §3.11) are skipped in check: the node
        // borrowed the grandparent's eval and jumps straight to N14 / the move
        // loop, since no eval-based pruning is sound while in check.
        if !in_check {
            // The pruning value `eval` (design/10 §3.4 N6): the corrected
            // static eval, then the TT-value-as-better-eval substitution —
            // a valid stored value whose bound points the right way is a
            // tighter position estimate than the raw eval. The substitution is
            // fenced to `excluded_move == NULL`: SF (search.cpp) takes a
            // dedicated `else if (excludedMove)` branch that sets
            // `eval = ss->staticEval` and does NOT fall through to the ttHit
            // substitution — so a singular verification razors/RFPs on the
            // enclosing node's corrected static eval, not on the (necessarily
            // present) stored value of the very move it is excluding. Without this guard the re-probe hits
            // the excluded move's own entry and inflates `eval` toward
            // `tt_value`, corrupting the exclusion-search return that decides
            // singular vs multicut vs negative extension.
            let mut eval = self.stack[cur].static_eval;
            if self.stack[cur].excluded_move == Move::NULL
                && is_valid(tt_value)
                && (if tt_value > eval {
                    tt_data.bound.includes_lower()
                } else {
                    tt_data.bound.includes_upper()
                })
            {
                eval = tt_value;
            }

            // N8 — eval-diff history nudge for the parent's quiet move
            // (design/10 §3.4, Wave-C item 12): grade the opponent's last
            // quiet by the static-eval swing it produced. `prev_sq` is Some
            // only for a real (non-null) parent move.
            if let Some(sq) = prev_sq
                && !self.stack[cur - 1].in_check
                && !prior_capture
            {
                let them = self.board.side_to_move().flip();
                let eval_diff = (-self.stack[cur - 1].static_eval - self.stack[cur].static_eval)
                    .clamp(params::EVALDIFF_CLAMP_LO(), params::EVALDIFF_CLAMP_HI())
                    + params::EVALDIFF_OFFSET();
                self.histories.main[them as usize][prev_move.raw16() as usize]
                    .update(params::EVALDIFF_MAIN_MULT() * eval_diff);
                if let Some(pc) = self.board.piece_on(sq)
                    && !tt_hit
                    && pc.piece_type != PieceType::Pawn
                    && prev_move.kind() != MoveKind::Promotion
                {
                    let pawn_key = self.board.pawn_key();
                    self.histories.pawn_slot_mut(pawn_key)[pc.index()][sq.index() as usize]
                        .update(params::EVALDIFF_PAWN_MULT() * eval_diff);
                }
            }

            // N9 — razoring (design/10 §3.4). A non-PV node whose corrected
            // `eval` sits so far below alpha that even the quadratic razoring
            // margin (`RAZOR_BASE + RAZOR_DEPTH_MULT·depth²`) cannot lift it
            // toward the window skips the full-width search entirely and drops
            // straight to qsearch, returning whatever tactic the captures-only
            // search uncovers — the qsearch IS the verification (a real
            // resource found there is >= alpha and returned; otherwise the
            // pessimistic fail-low stands). Uses the TT-substituted `eval`
            // (NOT `ss.static_eval` — the doc marks this site). No depth cap:
            // the margin grows as depth² so deep nodes effectively never
            // razor, making the heuristic self-limiting. Gated `!PV` so a mate
            // score along a PV line is never masked behind a qsearch value
            // (the SF "guard against mates being returned"); non-PV nodes have
            // no such exposure — the returned value stays inside the window,
            // and a losing `eval` only ever drops us into a qsearch that
            // confirms the loss. No `is_decisive` guard is needed: a winning
            // `eval` fails the `<` test outright, so wins are never razored.
            if !NT::PV
                && eval
                    < alpha
                        - params::RAZOR_BASE()
                        - params::RAZOR_DEPTH_MULT() * depth * depth
                        // Widen where our eval has historically been wrong.
                        - correction_value.abs() / params::RAZOR_CORR_DIV()
            {
                return self.qsearch::<NonPv>(tt, stop, alpha, beta, ply);
            }

            // N10 — reverse futility pruning ("futility: child", design/10
            // §3.4). Gated on `!ss.tt_pv` (a PV-flagged node is worth a real
            // search — and PV nodes always carry the flag, so this subsumes
            // `!PV`), a shallow depth (deep nodes never RFP — the depth
            // guard is what keeps mates findable), an eval already at/above
            // beta, the ttMove/ttCapture guard, and the two decisive-score
            // guards (`!is_loss(beta)`, `!is_win(eval)`) that keep the mate
            // bands out (§8.11). When the eval clears beta by the whole
            // margin, return a beta-ward blend rather than the raw eval —
            // static-eval instability damping (the same idiom as the
            // node-exit fail-high soften). The correction-history term of
            // the margin joins in Wave F (design/10 §4).
            if !self.stack[cur].tt_pv
                && depth < params::RFP_MAX_DEPTH()
                && eval >= beta
                && (tt_data.mv == Move::NULL || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
            {
                // Threat-coupled relaxation (Wave-G item 23, D30, Reckless R8):
                // a position where none of our pieces is currently threatened
                // is safe enough to prune a little more readily. Keys on ALL
                // our pieces — pawns included — matching the doc's "no own
                // piece is currently threatened" and Reckless's `all_threats()
                // & colors(stm)`; a hanging pawn still blocks the relaxation.
                // …but the threat map is only ever ABLE to change this node's
                // answer inside one `RFP_THREAT_MARGIN`-wide band, so it is
                // computed lazily (perf study 52 §5.5).
                //
                // `rfp_prune_margin` subtracts `RFP_THREAT_MARGIN * flag`, and
                // `RFP_THREAT_MARGIN` is positive (55), so `m(true) <= m(false)`
                // unconditionally — the flag can only ever make pruning EASIER.
                // Writing `C(f)` for `eval - m(f) >= beta`, that ordering gives
                // `C(false) => C(true)`, and therefore
                //
                //     C(flag)  ==  C(false) || (flag && C(true))
                //
                // exactly, for both values of `flag`. The right-hand side never
                // evaluates the flag when `C(false)` already prunes or when
                // `C(true)` cannot, which confines `all_threats()` to the band
                // where the relaxation decides. The equivalence survives the
                // EXP-UNC-1 rescale below because that scale is non-negative and
                // `as Value` truncation is monotone, so it cannot reorder the
                // two margins.
                //
                // Why this is worth doing: `Board::ensure_threats` memoises on
                // the position key, but RFP runs BEFORE any child search, so the
                // entry is cold here and this is a full `compute_threats()`
                // sweep. Measured on the precise (IBS) instrument: the
                // `all_threats` phase is 3.17% of `search::<NonPv>`'s executed
                // micro-ops — 0.90% of ALL engine ops — at this one call site.
                // Nodes that prune on the un-relaxed margin, or that go on to
                // fail high before `MovePicker::init_quiets` asks for the map,
                // now never build it.
                //
                // NOTE for anyone re-tightening this: the win is only real
                // while the map is genuinely unused on the paths that skip it.
                // It is a MEMO, so deferring never duplicates work — the first
                // later consumer pays what this site used to.
                let rfp_margin = |flag: bool| -> Value {
                    let margin = rfp_prune_margin(
                        depth,
                        tt_hit,
                        improving,
                        opponent_worsening,
                        correction_value,
                        flag,
                    );
                    // EXP-UNC-1: position-condition the RFP margin by the value
                    // uncertainty (consumer 1). unc_margin_scale returns exactly
                    // 1.0 when disabled or when there is no fresh σ̂ this node,
                    // and the else keeps the integer margin untouched →
                    // bit-identical by default (verified: sota bench 147691 with
                    // the coeff unset).
                    #[cfg(feature = "sota")]
                    let margin = if self.unc_rfp_c1 != 0.0 {
                        (margin as f32
                            * unc_margin_scale(node_sigma, self.sigma_bar, self.unc_rfp_c1))
                            as Value
                    } else {
                        margin
                    };
                    margin
                };
                let prune = eval - rfp_margin(false) >= beta
                    || (eval - rfp_margin(true) >= beta && {
                        let us = self.board.side_to_move();
                        (self.board.all_threats() & self.board.occupancy(us)).is_empty()
                    });
                if prune {
                    return (params::RFP_RET_BETA_W() * beta + params::RFP_RET_EVAL_W() * eval)
                        / params::RFP_RET_DEN();
                }
            }

            // N11 — null move pruning with verification (cutNode-only,
            // design/10 §3.4). A cut node whose *static* eval already clears
            // a beta-relative bar assumes it can afford to "pass" and still
            // fail high: it searches a null move at reduced depth `depth - R`
            // and, if that fails high, returns without searching a real move.
            // The gate reads `ss.static_eval`, NOT the TT-substituted `eval`
            // RFP uses (the doc marks this site). Zugzwang — where passing is
            // illusory — is fenced three ways: NMP runs only with non-pawn
            // material for the side to move, never trusts an unproven mate
            // (`!is_win(null_value)`), and at high depth re-confirms the
            // fail-high with a verification search that disables NMP up to
            // `nmp_min_ply` (recursion- and zugzwang-proof).
            let us = self.board.side_to_move();
            // DIAGNOSTIC A/B (2026-07-26 search-gap study, env-gated, default
            // off => bench-neutral): SF16 gates NMP on !PvNode; the donor
            // (eca43a97) on cutNode — but the donor's OTHER machinery (IIR,
            // hotter cutNode LMR) shrinks the trees that gate leaves behind,
            // and we did not port those. ARCHON_NMP_ALLNODE=1 tries the
            // SF16-style gate for measurement; landing is a post-D47 decision.
            let nmp_node_ok = if crate::search::nmp_allnode_probe() {
                !NT::PV
            } else {
                cut_node
            };
            if nmp_node_ok
                && self.stack[cur].static_eval
                    >= beta
                        - params::NMP_EVAL_DEPTH() * depth
                        - params::NMP_EVAL_IMPROVING() * Value::from(improving)
                        + params::NMP_EVAL_BASE()
                        // Raise the bar where our eval has historically been
                        // wrong: NMP's premise is that the static eval can be
                        // trusted enough to skip a move.
                        + correction_value.abs() / params::NMP_CORR_DIV()
                && self.stack[cur].excluded_move == Move::NULL
                && self.board.non_pawn_material(us) != 0
                && ply as i32 >= self.nmp_min_ply
                && !is_loss(beta)
            {
                // The parent is never itself a null move: a null move's child
                // is searched as a non-cut (all) node, so NMP — cutNode only —
                // cannot fire two plies deep off a single pass (SF's assert).
                debug_assert!(
                    self.stack[cur - 1].current_move != Move::NULL,
                    "no two null moves in a row"
                );
                let r = params::NMP_R_BASE() + depth / params::NMP_R_DEPTH_DIV();

                // Make the null move by hand (not `do_move`): a pass searches
                // no node, so it does NOT bump the node counter (SF `do_null_
                // move` parity), and it stamps the null-move stack markers the
                // child reads as its parent — `Move::NULL` (⇒ no `prev_sq`)
                // and the inert continuation-history SENTINEL.
                self.stack[cur].current_move = Move::NULL;
                self.stack[cur].cont_hist = ContHistId::SENTINEL;
                // NNUE (design/11 §3.4.4): a null move leaves every piece in
                // place, so the child accumulator IS the parent's — push no
                // entry at all and let the child index the same one (the us/them
                // perspective swap happens at `evaluate`, off the board). This
                // is Stockfish's shape: `do_null_move` has no
                // `accumulatorStack.push()` and `undo_null_move` no `pop()`
                // (`eca43a97:src/search.cpp:657-669`). No-op when NNUE is off.
                // Perf study 15 arm 1 — what it replaced was an EMPTY `DirtyPiece`
                // pushed into BOTH stacks, whose fold was a 2560-byte copy per
                // perspective per stack plus a `ThreatPosition` snapshot.
                self.eval_ctx.push_null();
                self.st_nmp_try += 1;
                self.board.make_null_move();
                let null_value =
                    -self.search::<NonPv>(tt, stop, -beta, -beta + 1, depth - r, false, ply + 1);
                self.board.unmake_null_move();
                self.eval_ctx.pop_null();

                // Do not return an unproven mate/TB score: a null-move
                // "mate" is an artifact of the illegal pass, never real.
                if null_value >= beta && !is_win(null_value) {
                    self.st_nmp_cut += 1;
                    // Below the verification depth — or already inside a
                    // verification (`nmp_min_ply != 0`, where recursive
                    // verification is disallowed) — trust the null value.
                    if self.nmp_min_ply != 0 || depth < params::NMP_VERIFY_DEPTH() {
                        return null_value;
                    }
                    // Verification search at THIS node (same ply, real moves)
                    // with NMP disabled up to `nmp_min_ply`; only a fail-high
                    // here confirms the cutoff. The disable window spans the
                    // whole verification subtree, so it cannot recursively
                    // null-move back to the same (possibly zugzwang) cutoff.
                    self.nmp_min_ply = ply as i32
                        + params::NMP_VERIFY_NUM() * (depth - r) / params::NMP_VERIFY_DEN();
                    let v = self.search::<NonPv>(tt, stop, beta - 1, beta, depth - r, false, ply);
                    self.nmp_min_ply = 0;
                    if v >= beta {
                        return null_value;
                    }
                }
            }

            // N11b — improving late-OR (design/10 §3.4, Wave-E item 17): from
            // here on, "improving" also means "eval already beats beta". Feeds
            // ProbCut and the move-loop LMP/reduction. SF places it between NMP
            // and IIR.
            improving |= self.stack[cur].static_eval >= beta;

            // N12 — internal iterative reduction (design/10 §3.4; S0 re-land
            // of reverted item 9 in the doc's narrow shape): a PV or cut node
            // (= !all_node) with no ttMove searches one ply shallower rather
            // than paying full price for a tree the TT can't seed. The P9
            // revert was the broad follow-PV-shaped form under PSQT eval;
            // `follow_pv` is live iff FOLLOW_PV_MODE=1 (node-entry write),
            // exempting the previous iteration's PV path exactly like SF.
            if !self.stack[cur].follow_pv
                && (NT::PV || cut_node)
                && depth >= params::IIR_MIN_DEPTH()
                && tt_data.mv == Move::NULL
            {
                depth -= params::IIR_REDUCTION();
            }

            // N13 — ProbCut (design/10 §3.4, Wave-E item 16): a cheap search
            // proving a capture beats a raised beta lets the node fail high
            // early. Skipped inside a singular verification; a stored value
            // already below the threshold says "don't bother".
            let probcut_beta = beta + params::PROBCUT_BETA_BASE()
                - params::PROBCUT_BETA_IMPROVING() * Value::from(improving);
            // Our `excluded_move == NULL` fence looks like a deviation from
            // SF (which has no such gate on the block, search.cpp:1037) but
            // is a DE FACTO no-op: inside a verification, beta is
            // `tt_value - margin`, so `probcut_beta ≈ tt_value - margin +
            // PROBCUT_BETA_BASE`, and with margin ≪ base at any practical
            // depth the `tt_value < probcut_beta` clause below blocks the
            // block in BOTH engines. Toggled + measured 2026-07-27
            // (PROBCUT_IN_VERIF): bit-identical trees at 50k nodes and at
            // depth 13 on wl50 — the toggle was deleted as dead surface.
            if depth >= params::PROBCUT_MIN_DEPTH()
                && !is_decisive(beta)
                && self.stack[cur].excluded_move == Move::NULL
                && !(is_valid(tt_value) && tt_value < probcut_beta)
            {
                let probcut_depth = depth
                    - params::PROBCUT_DEPTH_SUB()
                    - params::PROBCUT_IMPROVING_SUB() * Depth::from(improving);
                // Class 4: an EVAL-frame gap fed to `see_ge`, whose threshold
                // is in PIECE_VALUES. Identity at 1024.
                let threshold = params::scale_1024(
                    probcut_beta - self.stack[cur].static_eval,
                    params::SEE_THRESH_MULT(),
                );
                let mut mp = MovePicker::new_probcut(&self.board, tt_data.mv, threshold);
                while let Some(m) = mp.next_move(&self.board, &self.histories) {
                    if m == self.stack[cur].excluded_move || !self.board.legal(m) {
                        continue;
                    }
                    let gives_check = self.board.gives_check(m);
                    // ProbCut only ever emits captures and queen promotions, but
                    // `capture_stage` is asked anyway rather than assumed: this
                    // is a cold path and an assumption here would be a silent
                    // continuation-history mis-key if the picker ever changed.
                    let capture = self.board.capture_stage(m);
                    let moved_piece = self.board.piece_on(m.from());
                    self.do_move(tt, m, gives_check, capture, moved_piece, ply);
                    // Cheap qsearch verification first, then — if it held — a
                    // real reduced search.
                    let mut value =
                        -self.qsearch::<NonPv>(tt, stop, -probcut_beta, -probcut_beta + 1, ply + 1);
                    if value >= probcut_beta && probcut_depth > 0 {
                        value = -self.search::<NonPv>(
                            tt,
                            stop,
                            -probcut_beta,
                            -probcut_beta + 1,
                            probcut_depth,
                            !cut_node,
                            ply + 1,
                        );
                    }
                    self.undo_move(m);
                    if stop.load(Ordering::Relaxed) {
                        return VALUE_ZERO;
                    }
                    if value >= probcut_beta {
                        probe.writer.write(
                            key,
                            value_to_tt(value, ply as i32),
                            self.stack[cur].tt_pv,
                            Bound::Lower,
                            probcut_depth + 1,
                            m,
                            raw_eval,
                            generation,
                        );
                        if !is_decisive(value) {
                            // Shift the surplus back onto our window.
                            return value - (probcut_beta - beta);
                        }
                    }
                }
            }
        }

        // N14 — small in-check ProbCut (design/10 §3.4, Wave-E item 16): sits
        // after the in-check jump target, so it runs even in check. A deep
        // stored LOWER bound already above a large raised beta lets the node
        // fail high without entering the move loop.
        let check_probcut_beta = beta + params::PROBCUT_CHECK_BETA();
        if !is_decisive(beta)
            && self.stack[cur].excluded_move == Move::NULL
            && tt_hit
            && tt_data.bound.includes_lower()
            && tt_data.depth >= depth - params::PROBCUT_DEPTH_SUB()
            && is_valid(tt_value)
            && !is_decisive(tt_value)
            && tt_value >= check_probcut_beta
        {
            return check_probcut_beta;
        }

        // Step 9 — move loop. Interior nodes draw from the staged
        // MovePicker (doc 09), which emits pseudo-legal moves best-first
        // (TT move → good captures → good quiets → bad captures → bad
        // quiets); legality is checked per move instead of up front.
        // (The `excluded_move` skip lives at the top of the loop below —
        // doc 10's singular verification writes it during an exclusion search.)
        //
        // At the root the loop iterates `root_moves[pv_idx..]` itself
        // (doc 07 §3.7.2 step 9's "skip moves outside the root list",
        // v1 form): the ID loop's stable sort has last iteration's best
        // in front — the root form of the step-5 ttMove override, so the
        // picker is not consulted here. Everything in `root_moves` is
        // legal by construction, and `pv_last = root_moves.len()` until
        // MultiPV.
        let mut root_list = MoveList::new();
        if NT::ROOT {
            for rm in &self.root_moves[self.pv_idx..] {
                root_list.push(rm.pv[0]);
            }
        }
        // The six predecessor continuation-history keys the quiet scorer
        // rewards continuing (doc 09 §3.7): `stack[cur-1..=cur-6].cont_hist`,
        // in-bounds since `cur >= SS_OFFSET`. The picker is built for every
        // node but only POLLED at interior nodes; at the root it sits inert
        // (a cheap once-per-visit 2 KiB buffer init, §6).
        let cont_hist = [
            self.stack[cur - 1].cont_hist,
            self.stack[cur - 2].cont_hist,
            self.stack[cur - 3].cont_hist,
            self.stack[cur - 4].cont_hist,
            self.stack[cur - 5].cont_hist,
            self.stack[cur - 6].cont_hist,
        ];
        // D47 policy ordering (v1): when the TT offers no move, fill the slot
        // with the policy-argmax — legal by construction; the picker treats it
        // exactly like a TT move (pseudo-legality re-checked there). The
        // Zobrist-stamped accessor guarantees the activation belongs to THIS
        // node (a cheap-path/skipped eval yields None — correct, not stale).
        // Depth-gated to keep movegen+dot off the shallow tree. Without a
        // .cpol sidecar `active_policy_head()` is None => bit-identical.
        #[cfg(feature = "sota")]
        let picker_tt_move = if tt_data.mv == Move::NULL
            && depth >= 3
            && let Some(head) = self.policy_head
            && let Some(l2) = self.eval_ctx.last_l2_for(&self.board)
        {
            head.best_move(&self.board, l2).unwrap_or(Move::NULL)
        } else {
            tt_data.mv
        };
        #[cfg(not(feature = "sota"))]
        let picker_tt_move = tt_data.mv;
        let mut mp = MovePicker::new(&self.board, picker_tt_move, depth, ply, cont_hist);
        // P-ORD: attach the trained head's quiet-move prior. The activation is
        // copied NOW — the Zobrist-stamped stash dies at the first child eval,
        // long before QuietInit. `CPL_ORD_W` unset/0 ⇒ never constructed ⇒
        // ordering bit-identical.
        #[cfg(feature = "sota")]
        if crate::eval::nnue::policy_head::ord_weight() != 0.0
            && depth >= crate::eval::nnue::policy_head::ord_min_depth()
            && let Some(head) = self.policy_head
            && let Some(l2) = self.eval_ctx.last_l2_for(&self.board)
        {
            mp.set_quiet_prior(head.quiet_prior(
                &self.board,
                l2,
                crate::eval::nnue::policy_head::ord_weight(),
            ));
        }
        #[cfg(feature = "policy")]
        {
            mp.set_policy_net(self.policy_net.clone());
            if crate::policy::enabled_at(depth, ply) {
                if let Some(net) = self
                    .policy_net
                    .as_ref()
                    .filter(|net| net.uses_classic_accumulator())
                {
                    let scores = self
                        .eval_ctx
                        .classic_policy_accumulators(&self.board, ply)
                        .and_then(|(stm, ntm, activation)| {
                            net.scores_from_classic(stm, ntm, activation, self.board.side_to_move())
                        });
                    mp.set_policy_scores(scores);
                }
            }
        }

        let mut best_value = -VALUE_INFINITE;
        let mut best_move = Move::NULL;
        let mut move_count = 0;
        // Tried non-best moves, split by kind, for the node-result history
        // burst (doc 09 §3.9). Fixed 32-slot buffers — no heap on the hot
        // path (§6); anything past 32 is simply not recorded, matching SF.
        let mut quiets_searched = [Move::NULL; 32];
        let mut captures_searched = [Move::NULL; 32];
        let mut n_quiets = 0usize;
        let mut n_captures = 0usize;

        // In-search observation capture (design/16 §3.5.5): decide ONCE, at the
        // node's move-loop entry, whether this node is captured. The window
        // (`alpha`/`beta`) and `depth` are snapshotted here because both mutate
        // during the loop (alpha via move updates, depth via singular
        // extension), and the recorded context is the entry state. `None` (the
        // common case) is a single predictable branch; capture is observation-
        // only and never changes any search decision below.
        #[cfg(feature = "datagen")]
        let mut obs_node_capture = self.obs_begin_node(depth, alpha, beta);

        let mut root_idx = 0;
        loop {
            let m = if NT::ROOT {
                match root_list.get(root_idx) {
                    Some(&m) => {
                        root_idx += 1;
                        m
                    }
                    None => break,
                }
            } else {
                match mp.next_move(&self.board, &self.histories) {
                    Some(m) => m,
                    None => break,
                }
            };
            // Skip the singular-exclusion move (design/10 §3.5, SF `if (move
            // == ss->excludedMove) continue;`): during an exclusion search the
            // MovePicker is still built from `tt_data.mv` and emits it first,
            // so without this skip the search would include the very move it is
            // meant to exclude — silently defeating the singular test and
            // firing spurious multicuts. `excluded_move` is NULL at every
            // normal node (and no real move is NULL), so this only bites inside
            // a verification; when the ttMove is the ONLY legal move the loop
            // then ends with `move_count == 0`, taking the §3.10 fail-low.
            if m == self.stack[cur].excluded_move {
                continue;
            }
            // Interior nodes emit pseudo-legal moves — skip the illegal
            // ones before they count toward move_count (mate detection).
            if !NT::ROOT && !self.board.legal(m) {
                continue;
            }
            let is_capture = self.board.capture_stage(m);
            // The mover, resolved ONCE for the whole iteration (perf study 23
            // arm 6). It used to be read from the mailbox three times — in the
            // capture arm's SEE gate, in the quiet arm's continuation-history
            // gate, and again for `stat_score` — plus a fourth time inside
            // `do_move`, each with its own `Option` unwrap and panic edge. Every
            // path through the body needs it (the two pruning arms partition the
            // block, and everything that survives reaches `stat_score`), so
            // hoisting wastes nothing: the read is not merely deduplicated, it
            // was never conditional to begin with.
            let moved_piece = self
                .board
                .piece_on(m.from())
                .expect("a legal move starts on a piece of the side to move");
            // Perf study 42 arm: the SPECULATIVE TT prefetch (doc 08 §3.15
            // call site 2 / doc 04 §3.10 hook B). `Board::prefetch_key` has
            // existed, tested and documented as "search uses it to start the
            // TT prefetch even earlier than hook A", with NO CALLER, since it
            // landed. Hook A fires inside `make_move` once the child key is
            // final; from here to `do_move` is the whole §3.5 pruning cascade
            // and the singular verification, i.e. hundreds of cycles more
            // overlap. The cost is that a pruned move's hint is wasted, which
            // is why this is measured rather than assumed.
            #[cfg(feature = "ttspec42")]
            tt.prefetch(self.board.prefetch_key(m));
            move_count += 1;
            self.stack[cur].move_count = move_count;
            let nodes_before = self.nodes; // root effort accounting (§3.4)

            // Pre-move facts the §3.5 pruning block and the §3.7.2 PVS
            // three-call structure both read: whether the move gives check,
            // the child depth (`new_depth = depth - 1` — no extensions until
            // Wave D; `mut` because the §3.8 deeper/shallower rule adjusts it
            // in place), and the base per-move reduction `r` (design/10 §3.3;
            // `delta = beta - alpha` is the local window — 1 at NonPV nodes,
            // the aspiration width at the root). `r` is finalized after the
            // move is made (§3.7), but the pruning block reads only its BASE
            // value for `lmr_depth`, so it is computed here, ahead of both.
            // Perf study 53 arm `gc53`: `moved_piece` above is exactly the
            // piece `gives_check` would read back out of the mailbox, and the
            // intervening `stack[cur].move_count` store blocks the CSE. See
            // `Board::gives_check_by`.
            // Perf study 54 arm `nogc54` reverts JUST this line to the
            // pre-study-53 spelling, so `gc53`'s marginal effect can be
            // measured on the tree that actually shipped it — study 53 §0.2
            // left that open because its own control was not inert and
            // `magic53` alone beat the combination on loads by 11.7/node.
            #[cfg(feature = "nogc54")]
            let gives_check = self.board.gives_check(m);
            #[cfg(not(feature = "nogc54"))]
            let gives_check = self.board.gives_check_by(m, moved_piece);
            let mut new_depth = depth - 1;
            let delta = beta - alpha;
            let mut r = self.reduction(improving, depth, move_count, delta);
            // §3.5 ttPv entry bump (design/10 §3.5): provisional — mostly
            // unwound by the §3.7 rebate after the move is made. Its lasting
            // effect is on the pruning block below, whose `lmr_depth` reads
            // this base `r`.
            if self.stack[cur].tt_pv {
                r += params::RED_TTPV_ENTRY();
            }
            #[cfg(feature = "search-control")]
            let trace_node_depth = depth;
            #[cfg(feature = "search-control")]
            let trace_base_reduction = r;

            // §3.5 — shallow-depth pruning, decided BEFORE the move is made.
            // The whole block is fenced by the mate-safety guard `!is_loss(
            // best_value)` (never prune while we might still be getting
            // mated — the losing side keeps searching every move until one
            // escapes), never runs at the root, and never when the side to
            // move has only pawns left (the zugzwang/stalemate-trap guard
            // shared with NMP: sacrificing into a bare-pawn ending must stay
            // searched). Wave-B item 8's capture/check futility+SEE prunings
            // join the `capture || gives_check` arm below.
            if !NT::ROOT
                && !is_loss(best_value)
                && self.board.non_pawn_material(self.board.side_to_move()) != 0
            {
                // Late move pruning (movecount pruning): once enough moves
                // have been tried at this shallow node, stop the picker from
                // emitting any more quiets. The ttMove (always emitted first)
                // and every capture — good or bad — still come; in check the
                // evasion pipeline ignores the skip, so no response to check
                // is ever pruned. `improving` is false in check, halving the
                // threshold there, but the `!is_loss` guard keeps the block
                // dormant until an evasion has produced a non-loss value.
                if move_count >= lmp_move_count(depth, improving) {
                    mp.skip_quiet_moves();
                }

                // The predicted post-LMR depth (design/10 §3.5): the reduced
                // scout depth this move WOULD get, the depth input every
                // shallow-depth margin below takes. The BASE reduction `r`
                // (pre-§3.7 finalization) is the right one here — SF reads
                // `reduction()` output before its post-make adjustments — and
                // it can go negative at shallow nodes.
                //
                // The "history buys effective depth" half of Wave-C item 11
                // (SF `lmr_depth += history / LMR_DIVISOR[...]`, where a
                // good-history quiet earns extra effective depth and is spared
                // from parent-futility / quiet-SEE pruning) is DEFERRED — see
                // design/10 §4 item 11. It is a new tree-relaxing heuristic that
                // must clear its own fixed-node proxy SPRT (doc 14 §9.4), so it
                // is not implemented here: item 11 ships with only its
                // continuation-history pruning half. This site uses the plain
                // `new_depth - r/1024`.
                let mut lmr_depth = new_depth - r / params::RED_UNIT();

                // Captures and checking moves (Wave-B item 8, design/10 §3.5)
                // take their own futility/SEE gates; quiets — spared only on
                // the last iteration's PV path at a PV node (`!follow_pv ||
                // !PV`; `follow_pv` is written at node entry iff
                // FOLLOW_PV_MODE=1, else always false and this OR is always
                // true) — take parent futility then negative-SEE pruning. The `else` already implies a quiet
                // non-check, so its condition drops the `!is_capture &&
                // !gives_check` the capture arm now consumes.
                if is_capture || gives_check {
                    // The capture history `capt_hist = capture[moved][to]
                    // [captured]` (doc 09; the NO_VICTIM slot for en passant
                    // and quiet queen promotions, where no piece sits on `to`)
                    // feeds both capture gates.
                    let capt_hist = self.histories.capture_value(&self.board, m);

                    // Capture futility (design/10 §3.5): a non-checking capture
                    // whose victim value plus a depth-scaled slack and the
                    // capture-history nudge still leaves the static eval below
                    // alpha is futile and skipped. `piece_value(captured)` is 0
                    // for en passant (no piece on `to`) — SF parity. Unlike
                    // parent futility there is NO `!in_check` guard: the block's
                    // outer `!is_loss(best_value)` fence already keeps every
                    // evasion searched until one produces a real value.
                    if !gives_check && lmr_depth < params::CFUT_MAX_LMRDEPTH() {
                        let captured_value = self
                            .board
                            .piece_on(m.to())
                            .map_or(0, |p| p.piece_type.value());
                        if capture_futility_value(
                            self.stack[cur].static_eval,
                            lmr_depth,
                            captured_value,
                            capt_hist,
                        ) <= alpha
                        {
                            continue;
                        }
                    }

                    // Capture/check SEE pruning (design/10 §3.5): a capture or
                    // checking move that cannot win back `-margin` in a static
                    // exchange is pruned — UNLESS `alpha < VALUE_DRAW` and the
                    // moved piece is our LAST non-pawn material (sacrificing it
                    // may be a stalemate resource, which is never pruned).
                    let us = self.board.side_to_move();
                    let moved_value = moved_piece.piece_type.value();
                    if (alpha >= VALUE_DRAW || self.board.non_pawn_material(us) != moved_value)
                        && !self.board.see_ge(m, -capture_see_margin(depth, capt_hist))
                    {
                        continue;
                    }
                } else if !self.stack[cur].follow_pv || !NT::PV {
                    // Continuation-history pruning (design/10 §3.5, Wave-C
                    // item 11): the FIRST quiet gate, ahead of parent futility
                    // and quiet SEE. A quiet whose combined recent history
                    // `contHist(1)[pc][to] + contHist(2)[pc][to] +
                    // pawnHistory[pc][to]` — the offset-1/offset-2 predecessor
                    // slabs `stack[cur-1]/[cur-2]` key (the same contHist(1/2)
                    // move ordering and `stat_score` read) plus the pawn slot,
                    // all PRE-move — is deeply negative is skipped: the
                    // position's own ordering memory says the move has been
                    // failing here. `pc` is the piece on `m.from()`. No
                    // `!in_check` guard (unlike parent futility): the block's
                    // outer `!is_loss(best_value)` fence already keeps every
                    // evasion searched until one produces a real value. Fresh
                    // histories sum to −2366 (above the bar at every depth), so
                    // this is dormant until the search's own updates drive a
                    // quiet's score below `-QHIST_PRUNE_MULT·depth`.
                    let pci = moved_piece.index();
                    let to = m.to().index() as usize;
                    let cont_history =
                        self.histories.cont_ref(self.stack[cur - 1].cont_hist)[pci][to].get()
                            + self.histories.cont_ref(self.stack[cur - 2].cont_hist)[pci][to].get()
                            + self.histories.pawn_slot(self.board.pawn_key())[pci][to].get();
                    if continuation_history_prunes(cont_history, depth) {
                        continue;
                    }

                    // History buys lmr_depth (design/10 §3.5, item 11's
                    // deferred second half; S0, landed alongside the §3.7
                    // reduction terms per the doc's revisit-as-a-unit note):
                    // a quiet the combined history likes earns predicted
                    // depth, sparing it from the parent-futility and SEE
                    // prunings below; one it distrusts loses depth and is
                    // pruned earlier.
                    // SF writes this weight as `64·h/32` — 1/32 granularity
                    // the tuner can use; the product is formed before the
                    // divide (identical to `2·h` at the 64/32 defaults).
                    let history = cont_history
                        + params::LMRDEPTH_MAIN_NUM()
                            * self.histories.main[self.board.side_to_move() as usize]
                                [m.raw16() as usize]
                                .get()
                            / params::LMRDEPTH_MAIN_DEN();
                    lmr_depth += history / params::LMR_DIVISOR[depth.min(16) as usize - 1];

                    // Parent futility (design/10 §3.5): the most optimistic
                    // value this quiet could yield still cannot reach alpha at
                    // a shallow predicted depth. Lift `best_value` to that
                    // bound first — the fail-low value a pruned move plausibly
                    // delivers — guarded so no decisive score is corrupted
                    // (§8.11), then skip. `!in_check` keeps evasions searched.
                    let futility_value = parent_futility_value(
                        self.stack[cur].static_eval,
                        best_move == Move::NULL,
                        lmr_depth,
                        self.stack[cur].static_eval > alpha,
                    );
                    if !in_check
                        && lmr_depth < params::PFUT_MAX_LMRDEPTH()
                        && futility_value <= alpha
                    {
                        if best_value <= futility_value
                            && !is_decisive(best_value)
                            && !is_win(futility_value)
                        {
                            best_value = futility_value;
                        }
                        continue;
                    }

                    // Quiet SEE pruning (design/10 §3.5): a quiet that loses
                    // more than the quadratic margin `-25·lmr_depth²` in a
                    // static exchange is pruned (`lmr_depth` floored at 0, so
                    // the bar is 0 at shallow nodes — any material-losing
                    // quiet is dropped).
                    lmr_depth = lmr_depth.max(0);
                    if !self
                        .board
                        .see_ge(m, -params::QSEE_MULT() * lmr_depth * lmr_depth)
                    {
                        continue;
                    }
                }
            }

            // Perf study 42 arm 2: the speculative TT prefetch moved BELOW the
            // §3.5 pruning cascade. Arm 1 (`ttspec42`, above the cascade) buys
            // the most overlap and pays for it on every move that is then
            // pruned -- +95.6 instructions/node for ~1.07 probes/node, i.e.
            // roughly three quarters of its hints thrown away. From here the
            // move is committed, so the hint count collapses to the probe
            // count, and the overlap on offer is still the whole singular
            // verification search plus the reduction stack below it.
            #[cfg(feature = "ttspec2_42")]
            tt.prefetch(self.board.prefetch_key(m));

            // §3.6 — singular extensions and multicut (Wave-D items 13/14).
            // Only the ttMove is a candidate: an exclusion search below the
            // stored value (with the ttMove excluded, at the SAME ply) decides
            // whether it is singularly best (extend the whole node) or one of
            // several beta-beaters (multicut the subtree). The trigger disables
            // itself inside an exclusion search (`excluded_move` set), so it
            // never recurses (design/10 §5). `is_shuffling` suppresses
            // extensions of back-and-forth shuffles in 50-move territory.
            let mut extension: Depth = 0;
            if !NT::ROOT
                && m == tt_data.mv
                && self.stack[cur].excluded_move == Move::NULL
                && depth
                    >= params::SE_MIN_DEPTH()
                        + params::SE_TTPV_DEPTH_BONUS() * Depth::from(self.stack[cur].tt_pv)
                && is_valid(tt_value)
                && !is_decisive(tt_value)
                && tt_data.bound.includes_lower()
                && tt_data.depth >= depth - params::SE_TT_DEPTH_SLACK()
                && !self.is_shuffling(m, cur, ply)
            {
                let singular_beta =
                    singular_verification_beta(tt_value, depth, self.stack[cur].tt_pv && !NT::PV);
                let singular_depth =
                    new_depth * params::SE_VERIFY_DEPTH_NUM() / params::SE_VERIFY_DEPTH_DEN();
                self.stack[cur].excluded_move = m;
                let value = self.search::<NonPv>(
                    tt,
                    stop,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                    ply,
                );
                self.stack[cur].excluded_move = Move::NULL;

                if value < singular_beta {
                    // Singular: extend. `corr_adj` (Wave-F item 21, now live)
                    // tightens the double/triple margins under eval
                    // uncertainty — a large |correction_value| makes the extra
                    // extensions harder to earn.
                    let corr_adj = correction_value.abs() / params::SE_CORR_DIV();
                    let tmh = self.histories.tt_move.get();
                    let deep = ply > self.root_depth as usize;
                    let double_margin =
                        singular_double_margin(NT::PV, tt_capture, corr_adj, tmh, deep);
                    let triple_margin = singular_triple_margin(
                        NT::PV,
                        tt_capture,
                        self.stack[cur].tt_pv,
                        corr_adj,
                        deep,
                    );
                    extension = params::SE_EXT_BASE()
                        + Depth::from(value < singular_beta - double_margin)
                        + Depth::from(value < singular_beta - triple_margin);
                    // The WHOLE node deepens, not just the ttMove.
                    depth += params::SE_NODE_DEEPEN();
                } else if value >= beta && !is_decisive(value) {
                    // Multicut: multiple moves beat beta without the ttMove —
                    // the node is not singular; prune the whole subtree.
                    self.histories
                        .tt_move
                        .update(params::MULTICUT_TMH_BASE() + params::MULTICUT_TMH_DEPTH() * depth);
                    return value;
                } else if tt_value >= beta {
                    // The ttMove is expected to fail high anyway — search it
                    // shallower in favor of alternatives.
                    extension = params::SE_NEG_EXT_TT_FAILHIGH();
                } else if cut_node {
                    extension = params::SE_NEG_EXT_CUTNODE();
                }
            }
            new_depth += extension;
            self.st_ext += i64::from(extension);

            // §3.7 — the per-node combined history statistic `ss.stat_score`
            // (design/10 Wave-C item 10), computed from PRE-move facts:
            // `moved_piece` (resolved once at the top of the iteration — perf
            // study 23 arm 6) also keys the post-LMR conthist bonus below.
            // `stat_score` lives on the stack because the finalization block
            // docks `r` by it AND the child reads `(ss-1).stat_score` for its
            // fail-low parent bonus (a later wave). Must precede `do_move` —
            // every input (piece_on(from/to), the moving side) is a pre-move
            // fact.
            self.stack[cur].stat_score = self.stat_score(m, is_capture, moved_piece, cur);

            // Observational only: copy the already-maintained protected
            // accumulator for a deterministic, bounded subset of real LMR
            // actions. This occurs before make-move, so the latent and all
            // context describe the action's parent position.
            // WHY `trace_selected` is separate from `trace_latent`: the row used
            // to be pushed only when `classic_policy_accumulators` returned
            // Some, i.e. only when the D41 POLICY latent existed. Policy was
            // ditched (pol_lambda 0), so on every sota-v2 net the accumulators
            // are absent and the trace silently recorded ZERO rows — the file
            // still got its header, which is what made it look armed. The
            // latent matters only to the allocation datagen; walking a failing
            // search needs the decision fields, so selection and latent are now
            // independent.
            #[cfg(feature = "search-control")]
            let trace_selected = self.search_trace.as_ref().is_some_and(|trace| {
                trace.should_sample(key, m.raw16(), trace_node_depth, ply)
                    && depth >= 2
                    && (move_count > 1 || trace.all_moves())
            });
            #[cfg(feature = "search-control")]
            let trace_latent = if trace_selected {
                self.eval_ctx
                    .classic_policy_accumulators(&self.board, ply)
                    .map(|(stm, ntm, _)| {
                        let mut latent = Box::new([0i16; crate::eval::nnue::L1 * 2]);
                        latent[..crate::eval::nnue::L1].copy_from_slice(stm);
                        latent[crate::eval::nnue::L1..].copy_from_slice(ntm);
                        latent
                    })
            } else {
                None
            };
            #[cfg(feature = "search-control")]
            let trace_nominal_child_depth = new_depth;

            // Step 10 — make + recurse: the PVS three-call structure
            // (§3.7.2). Only the move ordered first (the ttMove when the
            // P6 probe supplied one — its likely-best status is what makes
            // the scouts below usually right) gets a full window up front;
            // every later move must first prove itself against a
            // zero-width window before it earns a PV re-search.
            self.do_move(tt, m, gives_check, is_capture, Some(moved_piece), ply);
            // §3.7 — reduction finalization (design/10 §3.7). The ttPv
            // rebate + cutoffCnt + all_node terms below were deferred through
            // the PSQT era (the surcharges over-reduced deep mate lines
            // without NNUE-grade eval); un-deferred with S0 now that the eval
            // upgrade landed. The rebate and the surcharges land TOGETHER by
            // design — the rebate is the counterweight that keeps PV-history
            // (mate-bearing) lines searched full (verified: surcharges alone
            // re-broke the §5 mate suite in unit tests). Still deferred to
            // the next S0 sub-commit: tt_capture (§3.7), sibling depth-2
            // (§3.9).
            //
            // ttPv rebate — first in the pinned adjustment list: a node on
            // (or once on) the PV with a trustworthy TT entry searches much
            // fuller, unwinding the §3.5 entry bump and more. The tt.value
            // term demands a REAL value: VALUE_NONE (32002) would spuriously
            // pass `> alpha` on a TT miss (same guard idiom as ProbCut).
            if self.stack[cur].tt_pv {
                // RED_TTPV_TTMISS_REBATE=1 restores SF parity: SF's miss
                // sentinel VALUE_NONE passes `ttData.value > alpha`
                // unconditionally (search.cpp:1292), so SF grants this term
                // at every PV-flagged TT-miss node; our `is_valid` guard
                // (ProbCut idiom) withholds it there.
                r -= params::RED_TTPV_BASE()
                    + params::RED_TTPV_PV() * i32::from(NT::PV)
                    + params::RED_TTPV_TTVALUE()
                        * i32::from(if is_valid(tt_value) {
                            tt_value > alpha
                        } else {
                            params::RED_TTPV_TTMISS_REBATE() != 0
                        })
                    + i32::from(tt_data.depth >= depth)
                        * (params::RED_TTPV_TTDEPTH_BASE()
                            + params::RED_TTPV_TTDEPTH_CUT() * i32::from(cut_node));
            }
            r += params::RED_FLAT();
            // Checking-move reduction rebate — ARCHON experiment, SF has no
            // such term. DEFAULT 0 ⇒ bit-identical; only a tune build moves it.
            // UNVALIDATED: the mechanism originally claimed for it was refuted
            // (see the doc comment on params::RED_GIVES_CHECK).
            r -= params::RED_GIVES_CHECK() * i32::from(gives_check);
            r -= params::RED_MOVECOUNT() * move_count;
            // Correction slackening (design/10 §3.7, Wave-F item 21): an
            // uncertain eval (large |correction_value|) reduces a touch LESS.
            r -= correction_value.abs() / params::RED_CORR_DIV();
            if cut_node {
                r += params::RED_CUTNODE()
                    + params::RED_CUTNODE_NO_TTMOVE() * i32::from(tt_data.mv == Move::NULL);
            }
            // tt_capture surcharge (design/10 §3.7): a capturing ttMove
            // means the position is tactical — quiet siblings are searched
            // shallower still.
            if tt_capture {
                r += params::RED_TTCAPTURE();
            }
            // cutoffCnt surcharge: the child slot has been failing high under
            // earlier siblings, so this node's late moves are distrusted. The
            // ttMove rebate is its ELSE arm — design/10 §3.7 pins them as
            // mutually exclusive (both firing would double-adjust `r`).
            let all_node = !NT::PV && !cut_node;
            let child_cutoffs = self.stack[cur + 1].cutoff_cnt;
            if child_cutoffs > params::RED_CUTOFFCNT_GATE() {
                r += params::RED_CUTOFFCNT_BASE()
                    + params::RED_CUTOFFCNT_MANY()
                        * i32::from(child_cutoffs > params::RED_CUTOFFCNT_MANY_GATE())
                    + params::RED_CUTOFFCNT_ALLNODE() * i32::from(all_node);
            } else if m == tt_data.mv {
                // The TT's own best move is searched close to full depth.
                r = (r - params::RED_TTMOVE_SUB()).max(0);
            }
            // Good/bad history docks the reduction (design/10 §3.7, Wave-C
            // item 10): a move the combined history likes searches fuller, one
            // it distrusts is reduced harder. Truncation toward zero on a
            // negative `stat_score` is intended (design/10 §3.1).
            r -= self.stack[cur].stat_score * params::RED_STATSCORE_NUM()
                / params::RED_STATSCORE_DEN();
            // All-node scaling — last in the §3.7 adjustment list: a node
            // where neither bound is expected to move reduces proportionally
            // harder, tapering with depth.
            if all_node {
                r += r * params::RED_ALLNODE_NUM()
                    / (params::RED_ALLNODE_DEPTH_MULT() * depth + params::RED_ALLNODE_DEPTH_BASE());
            }
            #[cfg(feature = "search-control")]
            let trace_final_reduction = r;
            #[cfg(feature = "search-control")]
            let mut trace_scout_depth = 0;
            #[cfg(feature = "search-control")]
            let mut trace_scout_score = -VALUE_INFINITE;
            #[cfg(feature = "search-control")]
            let mut trace_scout_nodes = 0u64;
            #[cfg(feature = "search-control")]
            let mut trace_research_nodes = 0u64;
            #[cfg(feature = "search-control")]
            let mut trace_lmr_research = false;
            #[cfg(feature = "search-control")]
            let mut trace_pv_research_nodes = 0u64;
            #[cfg(feature = "search-control")]
            let mut trace_pv_research = false;
            #[cfg(feature = "search-control")]
            let mut intervention_applied = false;
            #[cfg(feature = "search-control")]
            let mut intervention_classical_scout_depth = 0;
            // Overwritten by whichever call below runs last. A PV node's
            // first move skips (a)/(b), and (c)'s `move_count == 1` arm
            // never reads the sentinel.
            let mut value = -VALUE_INFINITE;
            // In-search capture (design/16 §3.5.5): did this move take an LMR
            // full-depth zero-window re-search? Cheap unconditional local (no
            // branch); compiled out of the default build.
            #[cfg(feature = "datagen")]
            let mut obs_researched = false;

            // (a) Reduced zero-window scout — LMR proper (design/10 §3.8). The
            // net reduction `r / RED_UNIT` (1024ths → plies) shrinks
            // `new_depth` to the scout depth `d`, clamped to
            // `[1, new_depth + RED_MAX_DEEPER]` and lifted one ply at PV nodes;
            // a negative net `r` may search a move DEEPER than `new_depth`.
            // LMR scouts always start as cut nodes (§3.7.1). `ss.reduction`
            // records the ply delta for the child's (deferred) hindsight read
            // (N6) and is cleared after — the child clears it at its own entry
            // too, but resetting by hand keeps the protocol robust if a future
            // `d <= 0` were ever to drop the scout straight to qsearch.
            if depth >= params::LMR_MIN_DEPTH() && move_count > params::LMR_MIN_MOVECOUNT() {
                let classical_d = (new_depth - r / params::RED_UNIT())
                    .min(new_depth + params::RED_MAX_DEEPER())
                    .max(params::LMR_MIN_SCOUT_DEPTH())
                    + params::LMR_PV_DEPTH_BONUS() * Depth::from(NT::PV);
                self.st_lmr_n += 1;
                self.st_lmr_plies += i64::from(new_depth - classical_d);
                #[cfg(feature = "search-control")]
                let d = {
                    intervention_classical_scout_depth = classical_d;
                    if let Some(requested) = self.lmr_intervention.as_mut().and_then(|oracle| {
                        oracle.request(key, m, trace_node_depth, ply, move_count)
                    }) {
                        intervention_applied = true;
                        requested.clamp(
                            1,
                            new_depth
                                + params::RED_MAX_DEEPER()
                                + params::LMR_PV_DEPTH_BONUS() * Depth::from(NT::PV),
                        )
                    } else {
                        classical_d
                    }
                };
                #[cfg(not(feature = "search-control"))]
                let d = classical_d;
                self.stack[cur].reduction = new_depth - d;
                #[cfg(feature = "search-control")]
                let trace_nodes_before_scout = self.nodes;
                value = -self.search::<NonPv>(tt, stop, -(alpha + 1), -alpha, d, true, ply + 1);
                #[cfg(feature = "search-control")]
                {
                    trace_scout_depth = d;
                    trace_scout_score = value;
                    trace_scout_nodes = self.nodes - trace_nodes_before_scout;
                }
                self.stack[cur].reduction = 0;

                // (b) The reduced scout beat alpha: apply the deeper/shallower
                // rule to `new_depth`, then re-search full-depth zero-window
                // with the flipped cut-node discipline (`!cut_node`, §3.7.1)
                // whenever the adjusted depth still exceeds the reduced `d`.
                // The deeper arm is itself gated on `d < new_depth` (a scout
                // that already ran at/above `new_depth` never deepens); the
                // shallower arm may still fire on a deepened scout (§3.8 / SF
                // parity). Whichever way it lands, a scout that beat alpha is
                // "worth continuing from" and earns the flat post-LMR
                // continuation-history bonus (design/10 §3.8, Wave-C item 10).
                if value > alpha {
                    let do_deeper =
                        d < new_depth && value > best_value + params::LMR_DEEPER_MARGIN();
                    let do_shallower = value < best_value + params::LMR_SHALLOWER_MARGIN();
                    new_depth += params::LMR_DEEPER_STEP() * Depth::from(do_deeper)
                        - params::LMR_SHALLOWER_STEP() * Depth::from(do_shallower);
                    if new_depth > d {
                        #[cfg(feature = "search-control")]
                        let trace_nodes_before_research = self.nodes;
                        value = -self.search::<NonPv>(
                            tt,
                            stop,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                            ply + 1,
                        );
                        #[cfg(feature = "search-control")]
                        {
                            trace_lmr_research = true;
                            trace_research_nodes = self.nodes - trace_nodes_before_research;
                        }
                        #[cfg(feature = "datagen")]
                        {
                            obs_researched = true;
                        }
                    }
                    self.histories.update_continuation_histories(
                        &self.stack[..],
                        cur,
                        moved_piece,
                        m.to(),
                        params::LMR_CONTHIST_BONUS(),
                    );
                }
            } else if !NT::PV || move_count > 1 {
                // Non-LMR zero-window path (§3.8): depth 1, or a NonPV node's
                // first move (where `(-(alpha+1), -alpha)` IS the node's own
                // zero-width window). A node with no TT move reduces a little
                // more, and a very high nominal `r` still docks 1–2 plies even
                // without LMR proper.
                if tt_data.mv == Move::NULL {
                    r += params::RED_FULLDEPTH_NO_TTMOVE();
                }
                let d = new_depth
                    - params::LMR_FULLDEPTH_DOCK_STEP()
                        * Depth::from(r > params::RED_FULLDEPTH_DOCK1())
                    - params::LMR_FULLDEPTH_DOCK_STEP()
                        * Depth::from(
                            r > params::RED_FULLDEPTH_DOCK2()
                                && new_depth > params::LMR_DOCK2_MIN_DEPTH(),
                        );
                value =
                    -self.search::<NonPv>(tt, stop, -(alpha + 1), -alpha, d, !cut_node, ply + 1);
            }

            // ttMove min-depth clamp (design/10 §3.8, Wave-D item 15): a
            // negative extension must never dive the PV's ttMove straight into
            // qsearch when the TT entry is deep or promises a decisive value.
            if m == tt_data.mv
                && ((is_valid(tt_value) && is_decisive(tt_value) && tt_data.depth > 0)
                    || tt_data.depth > 1)
            {
                new_depth = new_depth.max(1);
            }

            // (c) PV re-search, full window: PV/Root nodes only, for the
            // first move and for any scout that beat alpha. PV children
            // are never cut nodes — and only THIS call writes
            // `pv_table[ply + 1]`, which is §3.3's freshness invariant:
            // a step-13 PV update must never stitch a child row a
            // zero-window scout left stale (see `child_pv_fresh` below).
            let mut child_pv_fresh = false;
            if NT::PV && (move_count == 1 || value > alpha) {
                #[cfg(feature = "search-control")]
                let trace_nodes_before_pv = self.nodes;
                value = -self.search::<Pv>(tt, stop, -beta, -alpha, new_depth, false, ply + 1);
                #[cfg(feature = "search-control")]
                {
                    trace_pv_research = true;
                    trace_pv_research_nodes = self.nodes - trace_nodes_before_pv;
                }
                child_pv_fresh = true;
            }
            self.undo_move(m);

            // In-search capture (design/16 §3.5.5): one Layer-4 row per move
            // actually searched. `alpha`/`beta` here are still the values this
            // move was searched against (step 13 raises alpha below). `r` is the
            // finalized LMR reduction (1024ths); `self.nodes - nodes_before` is
            // this move's subtree cost. Read-only — no search state touched.
            #[cfg(feature = "datagen")]
            if let Some(capture) = obs_node_capture.as_mut() {
                capture.push_move(crate::obs::recorder::move_row(
                    m.raw16(),
                    move_count as u16,
                    new_depth,
                    r / params::RED_UNIT(),
                    extension,
                    value,
                    self.nodes - nodes_before,
                    value > alpha,
                    value >= beta,
                    obs_researched,
                ));
            }

            #[cfg(feature = "search-control")]
            if intervention_applied {
                if let Some(intervention) = self.lmr_intervention.as_mut() {
                    intervention.finish(crate::search_intervention::InterventionOutcome {
                        alpha,
                        beta,
                        classical_scout_depth: intervention_classical_scout_depth,
                        applied_scout_depth: trace_scout_depth,
                        scout_score: trace_scout_score,
                        final_score: value,
                        scout_nodes: trace_scout_nodes,
                        research_nodes: trace_research_nodes,
                        pv_research_nodes: trace_pv_research_nodes,
                        total_nodes: self.nodes - nodes_before,
                        lmr_research: trace_lmr_research,
                        pv_research: trace_pv_research,
                    });
                }
            }

            #[cfg(feature = "search-control")]
            if trace_selected {
                let latent =
                    trace_latent.unwrap_or_else(|| Box::new([0i16; crate::eval::nnue::L1 * 2]));
                let row = crate::search_trace::LmrObservation {
                    position_key: key,
                    fen: self.board.to_fen(),
                    latent,
                    raw_move: m.raw16(),
                    side_to_move: self.board.side_to_move() as u8,
                    moving_piece: moved_piece.piece_type as u8,
                    captured_piece: self
                        .board
                        .piece_on(m.to())
                        .map_or(PieceType::ALL.len() as u8, |piece| piece.piece_type as u8),
                    promotion_piece: m
                        .promotion_piece()
                        .map_or(PieceType::ALL.len() as u8, |piece| piece as u8),
                    depth: trace_node_depth,
                    ply: ply as u16,
                    alpha,
                    beta,
                    static_eval: self.stack[cur].static_eval,
                    best_value_before: best_value,
                    node_kind: if NT::ROOT {
                        0
                    } else if NT::PV {
                        1
                    } else if cut_node {
                        2
                    } else {
                        3
                    },
                    in_check,
                    improving,
                    cut_node,
                    tt_hit,
                    tt_depth: tt_data.depth,
                    tt_bound: tt_data.bound as u8,
                    tt_move: m == tt_data.mv,
                    move_index: move_count as u16,
                    capture: is_capture,
                    gives_check,
                    stat_score: self.stack[cur].stat_score,
                    base_reduction: trace_base_reduction,
                    final_reduction: trace_final_reduction,
                    extension,
                    nominal_child_depth: trace_nominal_child_depth,
                    scout_depth: trace_scout_depth,
                    scout_score: trace_scout_score,
                    final_score: value,
                    scout_nodes: trace_scout_nodes,
                    research_nodes: trace_research_nodes,
                    pv_research_nodes: trace_pv_research_nodes,
                    total_nodes: self.nodes - nodes_before,
                    lmr_research: trace_lmr_research,
                    pv_research: trace_pv_research,
                    alpha_raise: value > alpha,
                    beta_cutoff: value >= beta,
                };
                if let Some(trace) = self.search_trace.as_mut() {
                    trace.push(row);
                }
            }

            // Step 11 — stop check: the subtree result is untrusted
            // (§3.9). Return immediately — no best-move update, no PV
            // write; only `!stop`-guarded ID-loop sites consume results.
            if stop.load(Ordering::Relaxed) {
                return VALUE_ZERO;
            }

            // Step 12 — root bookkeeping (§3.4 per-move updates).
            if NT::ROOT {
                self.update_root_move(m, value, alpha, beta, move_count, self.nodes - nodes_before);
            }

            // Step 12 — the equal-value dither: near the leaves an
            // equal-to-best value is occasionally promoted — deliberate
            // diversity so equal (typically repetition-flavored) lines do
            // not all resolve to the first one found. The `is_win` guard
            // keeps the whole decisive band out (§8.11: mate distances
            // are never dithered), and short-circuiting on `value ==
            // best_value` first keeps `abs()` off the -INFINITE sentinel.
            // Only the comparisons below see `inc` — stored values and
            // window updates stay raw.
            let inc = Value::from(
                value == best_value
                    && ply + params::DITHER_PLY_MARGIN() >= self.root_depth as usize
                    && (self.nodes & params::DITHER_NODE_MASK()) == 0
                    && !is_win(value.abs() + 1),
            );

            // Step 13 — best/alpha/beta updates (dither-biased compares).
            if value + inc > best_value {
                best_value = value;
                if value + inc > alpha {
                    best_move = m;
                    if NT::PV {
                        // §3.3 freshness: the equal-value dither can
                        // promote a move whose LAST child call was a
                        // zero-window scout (`value == alpha`, `inc ==
                        // 1`) — `pv_table[ply + 1]` then still holds an
                        // earlier sibling's line, and stitching it would
                        // publish an illegal PV tail (caught by fastchess
                        // as "Illegal PV move" in the P7 clock games).
                        // Truncate instead: clear the stale row so the
                        // PV honestly ends at `m`.
                        if !child_pv_fresh {
                            self.pv_table[ply + 1].clear();
                        }
                        update_pv(&mut self.pv_table[..], ply, m);
                    }
                    if value >= beta {
                        self.st_fh += 1;
                        self.st_fh_first += u64::from(move_count == 1);
                        self.st_fh_moves += move_count as u64;
                        // Perf study 34: the ordering-quality instrument — the
                        // index of the move that actually caused the cutoff.
                        #[cfg(feature = "census34")]
                        {
                            use crate::census34 as cx;
                            cx::bump(&cx::FH_NODES, 1);
                            cx::bump(&cx::FH_MOVES, move_count as u64);
                            cx::hist(&cx::HIST_FH, move_count as usize);
                        }
                        // A cutoff by a heavily-extended move is weaker evidence
                        // of a fail-high-prone child, so it does not bump the
                        // count (design/10 §3.9, Wave-D item 15).
                        self.stack[cur].cutoff_cnt +=
                            i32::from(extension < params::CUTOFFCNT_EXT_GATE() || NT::PV);
                        break; // fail high
                    }
                    // Sibling depth reduction (design/10 §3.9, Wave-D item
                    // 15, un-deferred with S0): a quiet alpha raise without
                    // a cutoff is evidence the node is already well-resolved
                    // — the remaining siblings get a cheaper search.
                    // Decisive scores are exempt (mate distances stay exact
                    // at pinned depth). Root is exempt (ARCHON deviation,
                    // §3.9 note): the ID driver's iteration bookkeeping
                    // (doc 07) owns root depth, and mutating it mid-
                    // iteration corrupts the aspiration trace invariants.
                    // NB this makes non-root fixed-depth values ORDER-
                    // DEPENDENT (an early raise cheapens a later, better
                    // sibling) — doc 07 §4 step 6's TT-on/off equality bar
                    // is re-scoped accordingly.
                    if !NT::ROOT
                        && depth > params::SIBLING_DEPTH_MIN()
                        && depth < params::SIBLING_DEPTH_MAX()
                        && !is_decisive(value)
                    {
                        depth -= params::SIBLING_DEPTH_STEP();
                    }
                    alpha = value;
                }
            }

            // Remember a tried non-best move for the node-result history
            // burst (doc 09 §3.9). The fail-high `break` above already
            // skipped the cutoff move; a move still equal to best_move (it
            // raised alpha and stayed best) is likewise not recorded.
            if m != best_move {
                if is_capture {
                    if n_captures < captures_searched.len() {
                        captures_searched[n_captures] = m;
                        n_captures += 1;
                    }
                } else if n_quiets < quiets_searched.len() {
                    quiets_searched[n_quiets] = m;
                    n_quiets += 1;
                }
            }
        }

        // Step 14 — fail-high softening (search-instability damping, kept
        // with the frame): pull a fail-high best value partway back
        // toward beta, trusting deeper results more — `(v·depth + beta) /
        // (depth + 1)` is pure shape, no tunable constants. Guarded on
        // best_value AND alpha (alpha, not beta — SF parity, doc 10 §3.10
        // step 1): no decisive score is ever blended (§8.11), and a
        // decisive alpha means the window sits in the mate band, where
        // damping would corrupt mate distances.
        if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
            best_value = (best_value * depth + params::FH_SOFTEN_BETA_W() * beta)
                / (depth + params::FH_SOFTEN_BETA_W());
        }

        // Perf study 34: the move loop exits either by the fail-high `break`
        // (counted at the cutoff site) or by exhausting the picker; the two
        // together are the denominator of the ordering-quality rate.
        #[cfg(feature = "census34")]
        if best_value < beta {
            crate::census34::bump(&crate::census34::NOFH_NODES, 1);
        }

        // Step 14 — node result: ply-relative mate/stalemate terminals
        // (D5): a longer mate scores closer to zero, so the search
        // naturally prefers the shortest mate and delays being mated.
        if move_count == 0 {
            // Inside a singular verification the real best move was excluded, so
            // "no moves" is not mate/stalemate — fail low with alpha (design/10
            // §3.10 step 2, Wave-D item 13).
            best_value = if self.stack[cur].excluded_move != Move::NULL {
                alpha
            } else if in_check {
                mated_in(ply as i32)
            } else {
                VALUE_DRAW
            };
        }
        // Node-result history burst (doc 09 §3.9): a node that found a best
        // move raising alpha rewards it (and the agreeing quiet/capture/
        // continuation tables) and punishes the tried-but-worse moves.
        // `prev_sq` is the parent move's destination — `None` at the root
        // and after a null move, so the refutation-malus branch stays off
        // there. A terminal node (`move_count == 0`) leaves best_move NULL,
        // so this is skipped. SF's paired correction-history write is
        // deferred to its doc-10/11 pruning consumer (doc 09 §3.10 / D22).
        else if best_move != Move::NULL {
            self.histories.update_all_stats(
                &self.board,
                &self.stack[..],
                cur,
                best_move,
                prev_sq,
                &quiets_searched[..n_quiets],
                &captures_searched[..n_captures],
                depth,
                tt_data.mv,
                NT::PV,
            );
            // tt_move_history scalar (design/10 §3.10 step 3, Wave-D item 14):
            // at non-PV nodes, grade whether the ttMove turned out best.
            if !NT::PV {
                self.histories.tt_move.update(if best_move == tt_data.mv {
                    params::TMH_HIT()
                } else {
                    params::TMH_MISS()
                });
            }
        }
        // Fail low with a real move loop (design/10 §3.10 steps 4/5, Wave-E
        // item 19): no move raised alpha, so the parent's move "worked" —
        // reward it. A quiet parent gets the full `bonus_scale` distributed to
        // its conthist / main / pawn histories; a capturing parent gets a flat
        // capture-history bonus.
        // Reached only when the two arms above are false: `move_count == 0` is
        // excluded by the first arm and `best_move != NULL` by the second, so
        // `move_count != 0` here is implied — this is the fail-low case.
        else if let Some(sq) = prev_sq {
            if !prior_capture {
                let bonus_scale = (params::FL_SCALE_BASE()
                    - self.stack[cur - 1].stat_score / params::FL_STATSCORE_DIV()
                    + (params::FL_DEPTH_MULT() * depth).min(params::FL_DEPTH_CAP())
                    + params::FL_LATE_PARENT() * i32::from(self.stack[cur - 1].move_count > 8)
                    + params::FL_BELOW_EVAL()
                        * i32::from(
                            !in_check
                                && best_value
                                    <= self.stack[cur].static_eval - params::FL_BELOW_EVAL_MARGIN(),
                        )
                    + params::FL_BELOW_PARENT_EVAL()
                        * i32::from(
                            !self.stack[cur - 1].in_check
                                && best_value
                                    <= -self.stack[cur - 1].static_eval
                                        - params::FL_BELOW_PARENT_EVAL_MARGIN(),
                        ))
                .max(0);
                let scaled_bonus = (params::FL_BONUS_MULT() * depth - params::FL_BONUS_SUB())
                    .min(params::FL_BONUS_CAP())
                    * bonus_scale;
                if let Some(pc) = self.board.piece_on(sq) {
                    self.histories.update_continuation_histories(
                        &self.stack[..],
                        cur - 1,
                        pc,
                        sq,
                        scaled_bonus * params::FL_CONTHIST_NUM() / params::FL_CONTHIST_DEN(),
                    );
                    let them = self.board.side_to_move().flip();
                    self.histories.main[them as usize][prev_move.raw16() as usize]
                        .update(scaled_bonus * params::FL_MAIN_NUM() / params::FL_MAIN_DEN());
                    if pc.piece_type != PieceType::Pawn && prev_move.kind() != MoveKind::Promotion {
                        let pawn_key = self.board.pawn_key();
                        self.histories.pawn_slot_mut(pawn_key)[pc.index()][sq.index() as usize]
                            .update(scaled_bonus * params::FL_PAWN_NUM() / params::FL_PAWN_DEN());
                    }
                }
            } else if let Some(moved) = self.board.piece_on(sq) {
                // The victim slot mirrors the in-search capture convention:
                // en passant uses NO_VICTIM (None), every other capture its
                // captured piece type.
                let victim = if prev_move.kind() == MoveKind::EnPassant {
                    None
                } else {
                    self.board.captured_piece().map(|p| p.piece_type)
                };
                self.histories
                    .reward_prior_capture(moved, sq, victim, params::FL_CAPTURE_BONUS());
            }
        }

        // ttPv back-propagation on fail low (step 14): a node with no good
        // move below a still-PV parent stays flagged as search-tree-worthy.
        if best_value <= alpha {
            self.stack[cur].tt_pv = self.stack[cur].tt_pv || self.stack[cur - 1].tt_pv;
        }

        // TT store (step 14), through the write-side conversion boundary
        // `value_to_tt` (§3.12): LOWER on a fail high, EXACT only for a PV
        // node that found a best move, else UPPER; the RAW uncorrected
        // eval (D22); the sticky tt_pv flag. Skipped for excluded-move
        // searches and root nodes with pv_idx != 0; stopped searches never
        // reach here (step 11 returned). Terminal nodes store with a depth
        // bonus — the result is exact at any depth.
        if self.stack[cur].excluded_move == Move::NULL && !(NT::ROOT && self.pv_idx != 0) {
            let bound = if best_value >= beta {
                Bound::Lower
            } else if NT::PV && best_move != Move::NULL {
                Bound::Exact
            } else {
                Bound::Upper
            };
            let store_depth = if move_count != 0 {
                depth
            } else {
                // Clamp like SF (search.cpp): the terminal depth bonus must not
                // push the stored depth past MAX_PLY-1, where `TtEntry::save`'s
                // `depth - DEPTH_NONE` byte would overflow the u8 (debug assert)
                // and wrap to the empty-entry marker in release. The N0 clamp
                // (line ~1037) plus the hindsight +1 leaves only one unit of
                // slack; this makes the ceiling explicit rather than incidental.
                (depth + params::TT_NO_MOVES_DEPTH_BONUS()).min(MAX_PLY as Depth - 1)
            };
            probe.writer.write(
                key,
                value_to_tt(best_value, ply as i32),
                self.stack[cur].tt_pv,
                bound,
                store_depth,
                best_move,
                raw_eval,
                generation,
            );
        }

        // Correction-history update (design/10 §3.10 step 9, Wave-F item 21):
        // direction-consistency gated inside `correction_bonus` (not in check,
        // best move not a capture, and the eval-error direction matches whether
        // a best move was found). Skipped inside a singular verification (the
        // real best move was excluded, so the error signal is polluted).
        if self.stack[cur].excluded_move == Move::NULL
            && move_count != 0
            && let Some(bonus) = crate::history::correction_bonus(
                best_value,
                self.stack[cur].static_eval,
                in_check,
                best_move,
                best_move != Move::NULL && self.board.capture_stage(best_move),
                depth,
            )
        {
            self.histories
                .update_correction_history(&self.board, &self.stack[..], cur, bonus);
        }

        // In-search capture (design/16 §3.5.5): finish and buffer this node's
        // record. `self.board` is back at the node position (every move undone),
        // so Layer 1 is the node itself. A search stopped mid-loop returned at
        // step 11 above, so a finished node is always `Completion::Complete`.
        // Node-outcome bound mirrors the TT-store bound exactly. A node that
        // searched no move (terminal / all-pruned) carries no v0 Layer-4 data
        // and is skipped.
        #[cfg(feature = "datagen")]
        if let Some(capture) = obs_node_capture.take().filter(|c| c.searched_any()) {
            let obs_bound = if best_value >= beta {
                crate::obs::Bound::Lower
            } else if NT::PV && best_move != Move::NULL {
                crate::obs::Bound::Exact
            } else {
                crate::obs::Bound::Upper
            };
            let node_type = if NT::ROOT {
                crate::obs::NodeType::Root
            } else if NT::PV {
                crate::obs::NodeType::Pv
            } else if cut_node {
                crate::obs::NodeType::Cut
            } else {
                crate::obs::NodeType::All
            };
            let exit = crate::obs::recorder::NodeExit {
                best_value,
                best_move: best_move.raw16(),
                bound: obs_bound,
                node_type,
                static_eval: self.stack[cur].static_eval,
                ply: ply as u16,
                tt_hit,
                tt_depth: tt_data.depth,
                tt_bound: obs_bound_from_tt(tt_data.bound),
                tt_value: if tt_hit { tt_value } else { 0 },
                tt_move: tt_data.mv.raw16(),
                correction_value,
                node_budget: self.limits.nodes,
            };
            if let Some(recorder) = self.obs_recorder.as_mut() {
                crate::obs::recorder::finish_node(&self.board, self.nodes, recorder, capture, exit);
            }
        }

        debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);
        best_value
    }

    /// The base per-move reduction (design/10 §3.3), in 1024ths of a ply —
    /// divide by 1024 only at the moment of application (the suite's unit
    /// convention). Combines the log table [`Worker::reductions`] (indexed
    /// by `depth` and by `move_number`) with the aspiration window width
    /// `root_delta` (`beta - alpha` at the root, never zero — §3.6):
    ///
    /// ```text
    /// scale = reductions[depth] * reductions[move_number]
    /// r     = scale
    ///       - delta * RED_DELTA_MULT / root_delta         // wide window → less
    ///       + !improving * scale * RED_NON_IMPROVING / 512 // not improving → ~38% more
    ///       + RED_BASE
    /// ```
    ///
    /// Integer division truncates toward zero exactly like the Stockfish
    /// source this re-expresses (D27), so the term order matters and is
    /// preserved. Consumed by the move loop's LMR (design/10 §3.7/§3.8,
    /// Wave-A item 4): the returned `r` is finalized with the Wave-A accretion
    /// subset and applied as `new_depth - r / RED_UNIT`. Its base algebra is
    /// additionally pinned by the golden-value tests below (§5).
    #[inline]
    fn reduction(&self, improving: bool, depth: Depth, move_number: i32, delta: i32) -> i32 {
        debug_assert!(
            self.root_delta > 0,
            "root_delta is beta - alpha, never zero"
        );
        let scale = self.reductions[depth as usize] * self.reductions[move_number as usize];
        scale - delta * params::RED_DELTA_MULT() / self.root_delta
            + i32::from(!improving) * scale * params::RED_NON_IMPROVING()
                / params::RED_NON_IMPROVING_DEN()
            + params::RED_BASE()
    }

    /// The per-node combined history statistic `ss.stat_score` (design/10
    /// §3.7, Wave-C item 10), read from PRE-move state (so it must be called
    /// before `do_move`): `moved_piece` is the piece on `m.from()`, `us` the
    /// side to move. A capture is graded by victim value plus its capture
    /// history — `STATSCORE_CAPT_NUM·piece_value(captured)/STATSCORE_CAPT_DEN +
    /// capt_hist`, with `piece_value(captured) == 0` and the NO_VICTIM
    /// capture-history slot for en passant / quiet queen promotions (no piece
    /// on `to`), consistent with our capture-history victim convention (doc 09
    /// §3.9). A quiet is graded by twice its butterfly history plus the
    /// offset-1 and offset-2 continuation histories, indexed
    /// `[moved_piece][to]` off the predecessor frames `stack[cur-1]` /
    /// `stack[cur-2]` (the same slabs move ordering reads as `contHist(1/2)`,
    /// doc 09 §3.7). Feeds the `RED_STATSCORE_NUM` reduction dock in the
    /// finalization block below.
    #[inline]
    fn stat_score(&self, m: Move, is_capture: bool, moved_piece: Piece, cur: usize) -> i32 {
        if is_capture {
            let captured_value = self
                .board
                .piece_on(m.to())
                .map_or(0, |p| p.piece_type.value());
            params::STATSCORE_CAPT_NUM() * captured_value / params::STATSCORE_CAPT_DEN()
                + self.histories.capture_value(&self.board, m)
        } else {
            let us = self.board.side_to_move();
            // Two planes, one cell — flat-indexed (`history::piece_to`).
            let q = crate::history::piece_to(moved_piece.index(), m.to().index() as usize);
            params::STATSCORE_MAIN_WEIGHT()
                * self.histories.main[us as usize][m.raw16() as usize].get()
                + crate::history::at_piece_to(
                    self.histories.cont_ref(self.stack[cur - 1].cont_hist),
                    q,
                )
                + crate::history::at_piece_to(
                    self.histories.cont_ref(self.stack[cur - 2].cont_hist),
                    q,
                )
        }
    }

    /// The singular anti-shuffle guard (design/10 §3.6, Wave-D item 14): a
    /// quiet move that merely oscillates a piece back and forth in
    /// 50-move-rule territory is not worth extending. True iff the move is
    /// quiet, the halfmove clock and plies-from-null are high enough, the ply
    /// is deep, and the piece is retracing its own path —
    /// `m.from() == (ss-2).to() && (ss-2).from() == (ss-4).to()`. The `ply >=
    /// 20` guard keeps `cur - 4` well in bounds.
    #[inline]
    fn is_shuffling(&self, m: Move, cur: usize, ply: usize) -> bool {
        if self.board.capture_stage(m)
            || (self.board.rule50_count() as i32) < params::SHUFFLE_RULE50()
            || self.board.plies_from_null() < params::SHUFFLE_PLIES_FROM_NULL()
            || ply < params::SHUFFLE_MIN_PLY()
        {
            return false;
        }
        let m2 = self.stack[cur - 2].current_move;
        let m4 = self.stack[cur - 4].current_move;
        m2.is_ok() && m4.is_ok() && m.from() == m2.to() && m2.from() == m4.to()
    }

    /// Step-12 root bookkeeping (design/07 §3.4): update `m`'s [`RootMove`]
    /// after its child search returned `value` against the pre-step-13 root
    /// window `(alpha, beta)`. The root is cold — the linear lookup and the
    /// `Vec` PV rebuild are off the hot path (§6).
    fn update_root_move(
        &mut self,
        m: Move,
        value: Value,
        alpha: Value,
        beta: Value,
        move_count: i32,
        effort: u64,
    ) {
        let idx = self.root_moves[self.pv_idx..]
            .iter()
            .position(|rm| rm.pv[0] == m)
            .map(|i| self.pv_idx + i)
            .expect("the root move loop iterates root_moves");
        let sel_depth = self.sel_depth;
        let rm = &mut self.root_moves[idx];
        rm.effort += effort;
        rm.update_average_score(value);
        rm.update_mean_squared_score(value);
        if move_count == 1 || value > alpha {
            rm.score = value;
            rm.uci_score = value;
            rm.sel_depth = sel_depth;
            rm.score_lowerbound = false;
            rm.score_upperbound = false;
            if value >= beta {
                rm.score_lowerbound = true;
                rm.uci_score = beta; // clamp what we print to the window
            } else if value <= alpha {
                rm.score_upperbound = true;
                rm.uci_score = alpha;
            }
            // Rebuild the PV from the fresh child line — §3.3's "the root
            // copies pv_table[1]" invariant. This branch only runs when
            // the (c) full-window PV re-search ran last (§3.7.2 step 10:
            // `move_count == 1`, or a final value above alpha — which a
            // skipped re-search cannot produce), so the row is always
            // fresh; zero-window scouts are NonPV and never write it.
            rm.pv.truncate(1);
            rm.pv.extend_from_slice(self.pv_table[1].as_slice());
            // Best-move flip mid-iteration: the TM instability signal
            // (§3.4; consumed by doc 12 §3.9's factor stack). Counted in the
            // per-thread atomic (design/13 §3.10) so the main worker can
            // aggregate it across all threads at iteration end.
            if move_count > 1 && self.pv_idx == 0 {
                self.counters.bump_best_move_changes();
            }
        } else {
            // Not the new best: mark unsearched-this-iteration and rely on
            // the STABLE descending sort to keep last iteration's order
            // (§3.4 — an unstable sort here is a silent Elo-losing bug).
            rm.score = -VALUE_INFINITE;
        }
    }

    /// Quiescence search, the full design/07 §3.8 spec (plan step 9):
    /// stand pat on the corrected static eval with the tt-value
    /// substitution and the fail-high softening blend (step 5), captures
    /// only (queen promotions ride the CAPTURES stage) via `capture_stage`
    /// semantics, ALL evasions while in check, the step-7 per-move pruning
    /// block — every rule gated on `!is_loss(best_value)`, the evasion
    /// guard that keeps checkmates from being pruned into existence — with
    /// the late-move cutoff, futility against `futility_base`, SEE against
    /// the window, the non-capture skip and the hard SEE floor, ply-relative
    /// mate when checkmated plus the narrow stalemate test (step 10), the
    /// node-exit fail-high blend (step 11), and the `DEPTH_QS` TT
    /// discipline (steps 4/5/12): probe at entry, eval reuse, the stand-pat
    /// fail-high eval-cache write, and a store at exit that is NEVER
    /// `Exact`. Constants live in [`params`] (D20).
    ///
    /// Stop poll at entry (the Reckless re-poll, D31): `check_time` is
    /// re-checked here so the hard wall is observed even in the middle of a
    /// long forcing-capture dive, bounding worst-case stop latency. qsearch
    /// itself does not abort — it relies on the enclosing main-search node's
    /// step-11 stop check to discard the untrusted subtree — so a stop set
    /// here surfaces the moment this dive returns. Under fixed-depth /
    /// fixed-node search `check_time` can never fire (bench-neutral, D24).
    fn qsearch<NT: NodeType>(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        mut alpha: Value,
        beta: Value,
        ply: usize,
    ) -> Value {
        debug_assert!(!NT::ROOT, "qsearch is never Root-typed (§3.7.1)");
        debug_assert!(NT::PV || alpha == beta - 1, "NonPV requires a zero window");
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);

        // Upcoming repetition (SF search.cpp:1605-1611, toggled by QS_UPREP):
        // SF runs the same alpha-raise that opens search() as the FIRST
        // statement of qsearch too; ours historically covered only the entry
        // node above the dive, so recursive qsearch nodes below the horizon
        // forfeited the cutoff. Same `rule50 >= 3` short-circuit as the
        // search() copy (a cycle needs at least 3 reversible plies).
        if params::QS_UPREP() != 0
            && alpha < VALUE_DRAW
            && self.board.rule50_count() >= 3
            && self.board.upcoming_repetition(ply as i32)
        {
            alpha = value_draw_contempt(self.nodes, ply);
            if alpha >= beta {
                return alpha;
            }
        }

        // Step 2 — init.
        let cur = ss(ply);
        let in_check = !self.board.checkers().is_empty();
        self.stack[cur].in_check = in_check;
        if NT::PV {
            self.pv_table[ply].clear();
            self.sel_depth = self.sel_depth.max(ply as i32 + 1);
        }
        // D31 re-poll: bound stop latency under a long qsearch dive.
        self.check_time(stop);

        // Step 3 — draw/max-ply: plain VALUE_DRAW, deliberately NOT
        // dithered (SF parity; see [`value_draw`]).
        if self.board.is_draw(ply as i32) || ply >= MAX_PLY {
            return if ply >= MAX_PLY && !in_check {
                evaluate(
                    &self.board,
                    &mut self.eval_ctx,
                    self.optimism[self.board.side_to_move() as usize],
                    ply,
                )
            } else {
                VALUE_DRAW
            };
        }

        // Step 4 — TT probe: everything probes and stores at DEPTH_QS.
        // Same read-side conversion boundary as the main search (§3.12).
        let key = self.board.key();
        let generation = tt.generation();
        let probe = tt.probe(key);
        let (tt_hit, tt_data) = (probe.hit, probe.data);
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply as i32, self.board.rule50_count() as i32)
        } else {
            VALUE_NONE
        };
        // The PV flag the step-12 store writes back: the PROBED flag only,
        // deliberately NOT OR'd with this node's own PV-ness (§3.8 step 12
        // — OR-ing would systematically inflate the sticky ttPv signal).
        let pv_hit = tt_hit && tt_data.is_pv;
        self.stack[cur].tt_hit = tt_hit;

        // The NonPV TT cutoff (§3.8 step 4): any stored depth qualifies
        // (`>= DEPTH_QS`). Like the main search's step 7, live since the
        // PVS zero-window (plan step 7) creates NonPV qsearch nodes.
        if !NT::PV
            && is_valid(tt_value)
            && tt_data.depth >= DEPTH_QS
            && (if tt_value >= beta {
                tt_data.bound.includes_lower()
            } else {
                tt_data.bound.includes_upper()
            })
        {
            return tt_value;
        }

        // Step 5 — stand pat (not in check), reusing a valid cached RAW
        // eval (D22). In check there is no stand pat: best_value stays at
        // -VALUE_INFINITE (a loss) so nothing can cut before at least one
        // evasion is searched — the evasion guard that keeps checkmates
        // from being pruned into existence — futility_base stays a loss so
        // the futility sub-block never fires, and raw_eval stays VALUE_NONE
        // for the store.
        let mut best_value = -VALUE_INFINITE;
        let mut futility_base = -VALUE_INFINITE;
        let mut raw_eval = VALUE_NONE;
        if !in_check {
            raw_eval = if tt_hit && is_valid(tt_data.eval) {
                tt_data.eval
            } else {
                evaluate(
                    &self.board,
                    &mut self.eval_ctx,
                    self.optimism[self.board.side_to_move() as usize],
                    ply,
                )
            };
            // S1 (doc 07 §3.8 step 5, D22): stand pat on the CORRECTED eval;
            // the TT stores (fail-high cache above the loop, step-12 store
            // below) keep the RAW eval, same discipline as the main search.
            let correction_value =
                self.histories
                    .correction_value(&self.board, &self.stack[..], cur);
            self.stack[cur].static_eval =
                crate::history::to_corrected_static_eval(raw_eval, correction_value);
            best_value = self.stack[cur].static_eval;

            // A valid, NON-decisive tt_value whose bound points the right
            // way substitutes as a better stand pat (§3.8 step 5). Unlike
            // the main search's step-6 twin this one requires the
            // non-decisive guard (SF parity, asymmetric by design).
            if is_valid(tt_value)
                && !is_decisive(tt_value)
                && (if tt_value > best_value {
                    tt_data.bound.includes_lower()
                } else {
                    tt_data.bound.includes_upper()
                })
            {
                best_value = tt_value;
            }

            if best_value >= beta {
                qtrace(ply, alpha, beta, best_value, "-", "STANDPAT_CUT");
                // Search-instability damping (§3.8 step 5): pull a
                // non-decisive stand-pat cutoff partway back toward beta.
                // Decisive scores are never blended (§8.11).
                if !is_decisive(best_value) {
                    best_value = (params::QS_STANDPAT_BEST_W() * best_value
                        + params::QS_STANDPAT_BETA_W() * beta)
                        / params::QS_STANDPAT_DEN();
                }
                // A stand-pat fail high on a TT miss caches the raw eval
                // as a Bound::Lower eval-only entry (§3.11).
                if !tt_hit {
                    probe.writer.write(
                        key,
                        VALUE_NONE,
                        false,
                        Bound::Lower,
                        DEPTH_UNSEARCHED,
                        Move::NULL,
                        raw_eval,
                        generation,
                    );
                }
                return best_value;
            }
            alpha = alpha.max(best_value);
            // The bar every capture's victim must clear (§3.8 step 5):
            // from the CORRECTED static eval, not the substituted
            // best_value.
            futility_base = self.stack[cur].static_eval + params::QS_FUTILITY_MARGIN();
        }

        // Steps 6–7 — the staged MovePicker (doc 09). Not in check it runs
        // the qsearch pipeline (TT move → captures + queen promotions, the
        // CAPTURES stage of doc 05 §3.2); in check it runs the evasion
        // pipeline (every evasion, captures first). Both emit pseudo-legal
        // moves; legality is tested on the move we are about to play, and
        // the TT move's usability is validated inside the picker (§3.5). A
        // quiet TT move can be emitted here, but the step-7 non-capture skip
        // drops it once best_value is real — the qsearch move set is
        // effectively captures/evasions, as before.
        let cont_hist = [
            self.stack[cur - 1].cont_hist,
            self.stack[cur - 2].cont_hist,
            self.stack[cur - 3].cont_hist,
            self.stack[cur - 4].cont_hist,
            self.stack[cur - 5].cont_hist,
            self.stack[cur - 6].cont_hist,
        ];
        let mut mp = MovePicker::new(&self.board, tt_data.mv, DEPTH_QS, ply, cont_hist);

        // The previous move's destination (§3.8 step 6): recaptures on
        // that square are exempt from the step-7 futility block. NULL
        // means "no real move" (root entry, later null-move search).
        let prev_move = self.stack[cur - 1].current_move;
        let prev_sq = (prev_move != Move::NULL).then(|| prev_move.to());

        qtrace(ply, alpha, beta, best_value, "-", "LOOP_ENTERED");
        let mut best_move = Move::NULL;
        let mut move_count = 0;
        while let Some(m) = mp.next_move(&self.board, &self.histories) {
            if !self.board.legal(m) {
                continue;
            }
            let gives_check = self.board.gives_check(m);
            move_count += 1;

            // Step 7 — the pruning block, gated in FULL on
            // `!is_loss(best_value)`: while in check and no legal evasion
            // has produced a real value yet, best_value is -VALUE_INFINITE
            // (a loss) and nothing may be pruned — a checkmate can never
            // be pruned into existence.
            if !is_loss(best_value) {
                // Non-checking moves not aimed at the previous move's
                // square, promotions excluded; inert while futility_base
                // is a loss (in-check nodes).
                if !gives_check
                    && Some(m.to()) != prev_sq
                    && !is_loss(futility_base)
                    && m.kind() != MoveKind::Promotion
                {
                    // Late-move cutoff: after two such moves, skip them
                    // all (§3.8 step 7).
                    if move_count > params::QS_LATE_MOVE_LIMIT() {
                        qtrace(
                            ply,
                            alpha,
                            beta,
                            best_value,
                            &m.to_string(),
                            "prune:latemove",
                        );
                        continue;
                    }

                    // Futility: even winning the victim for free cannot
                    // lift the eval-based bar above alpha. The pruned
                    // move's optimistic value still raises best_value
                    // (D11 SEE-scale piece values; empty destination —
                    // en passant — counts 0, SF parity).
                    // Class 4: `piece_type.value()` is MATERIAL, `futility_base`
                    // is EVAL, and the two frames differ by a different factor
                    // here than in Stockfish. Identity at 1024.
                    let futility_value = futility_base
                        + params::scale_1024(
                            self.board
                                .piece_on(m.to())
                                .map_or(0, |victim| victim.piece_type.value()),
                            params::QS_PIECE_MULT(),
                        );
                    if futility_value <= alpha {
                        qtrace(
                            ply,
                            alpha,
                            beta,
                            best_value,
                            &m.to_string(),
                            "prune:futility",
                        );
                        best_value = best_value.max(futility_value);
                        continue;
                    }

                    // SEE vs the window: the exchange cannot recover the
                    // gap between futility_base and alpha.
                    // Class 4, as at the ProbCut filter: an eval-frame gap
                    // used as a material threshold. Identity at 1024.
                    if !self.board.see_ge(
                        m,
                        params::scale_1024(alpha - futility_base, params::SEE_THRESH_MULT()),
                    ) {
                        best_value = best_value.max(alpha.min(futility_base));
                        continue;
                    }
                }

                // Skip all remaining non-captures — in practice the quiet
                // evasions once one escape has a real value (queen
                // promotions ride the capture stage).
                if !self.board.capture_stage(m) {
                    continue;
                }

                // The hard SEE floor for everything left (§3.8 step 7).
                if !self.board.see_ge(m, params::QS_SEE_FLOOR()) {
                    qtrace(
                        ply,
                        alpha,
                        beta,
                        best_value,
                        &m.to_string(),
                        "prune:seefloor",
                    );
                    continue;
                }
            }

            // Step 8 — recurse: full window, same node type.
            qtrace(ply, alpha, beta, best_value, &m.to_string(), "SEARCH");
            // Same facts `do_move` used to re-derive; qsearch has no earlier
            // use for either, so this is a move, not a saving (perf study 23
            // arm 6 keeps the main loop's win without taxing qsearch).
            let capture = self.board.capture_stage(m);
            let moved_piece = self.board.piece_on(m.from());
            self.do_move(tt, m, gives_check, capture, moved_piece, ply);
            self.qnodes += 1;
            let value = -self.qsearch::<NT>(tt, stop, -beta, -alpha, ply + 1);
            self.undo_move(m);

            // Step 9 — updates.
            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = m;
                    if NT::PV {
                        update_pv(&mut self.pv_table[..], ply, m);
                    }
                    if value >= beta {
                        break; // fail high
                    }
                    alpha = value;
                }
            }
        }

        // Step 10 — mate and the narrow stalemate test. Every evasion was
        // generated, so zero legal moves while in check is mate,
        // ply-relative (D5), returned WITHOUT a store (SF parity).
        if move_count == 0 {
            if in_check {
                return mated_in(ply as i32);
            }
            // The narrow stalemate test (§3.8 step 10): a full legality
            // scan at every quiet qsearch leaf would be ruinous, so pay
            // for one only when stalemate is cheap to suspect — the side
            // to move has no non-pawn material, every pawn single-push is
            // blocked, and the previous move captured a piece >= knight
            // (the classic "qsearch just won material into a stalemate
            // trap" pattern). Kings are never captured, so ">= knight" is
            // exactly "not a pawn".
            let us = self.board.side_to_move();
            let pawn_pushes = self.board.pieces(us, PieceType::Pawn).pawn_pushes(us);
            if self.board.non_pawn_material(us) == 0
                && (pawn_pushes & !self.board.all_occupancy()).is_empty()
                && self
                    .board
                    .captured_piece()
                    .is_some_and(|victim| victim.piece_type != PieceType::Pawn)
            {
                let mut legal = MoveList::new();
                generate_legal(&self.board, &mut legal);
                if legal.is_empty() {
                    best_value = VALUE_DRAW;
                }
            }
        }

        // Step 11 — fail-high softening: pull a non-decisive best value
        // strictly above beta partway back toward it (search-instability
        // damping; the blended value is what gets stored below). Decisive
        // scores are never blended (§8.11).
        if !is_decisive(best_value) && best_value > beta {
            best_value = (params::QS_FAILHIGH_BEST_W() * best_value
                + params::QS_FAILHIGH_BETA_W() * beta)
                / params::QS_FAILHIGH_DEN();
        }

        // Step 12 — TT store at DEPTH_QS: LOWER on a fail high, else
        // UPPER — NEVER Exact in qsearch — value through `value_to_tt`,
        // the RAW eval, and the probed PV flag only.
        probe.writer.write(
            key,
            value_to_tt(best_value, ply as i32),
            pv_hit,
            if best_value >= beta {
                Bound::Lower
            } else {
                Bound::Upper
            },
            DEPTH_QS,
            best_move,
            raw_eval,
            generation,
        );

        debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);
        best_value
    }

    /// Make `m`, counting the node (§3.9 — `nodes` counts MADE moves,
    /// matching SF and the `bench` fingerprint) and installing the child's
    /// stack slots. The [`TtPrefetch`] hook rides the [`MakeMoveHooks`]
    /// seam (doc 04 §3.10): the child's TT cluster is prefetched the
    /// moment its key is final, mid-make.
    ///
    /// The continuation-history key (doc 09 §3.2) is stamped here, computed
    /// from the PRE-move position: `[node in_check][capture stage][moved
    /// piece][to]`. `stack[cur].cont_hist` becomes the child's offset-1
    /// predecessor slab; the SENTINEL is only for null moves / pre-root
    /// padding (no null moves until doc 10). `in_check` was set at this
    /// node's step-2 entry.
    ///
    /// `capture` and `moved_piece` are the caller's — perf study 23 arm 6. Both
    /// used to be re-derived here, and the main move loop had ALREADY computed
    /// both from the same unchanged board: `capture_stage` at the top of the
    /// loop and the mover for `stat_score`. Two mailbox reads, an `Option`
    /// unwrap with its panic edge, and a `capture_stage` `match` per made move,
    /// for facts that were sitting in registers one line up. The two cold call
    /// sites (ProbCut, qsearch) pass exactly what this function used to compute,
    /// so nothing moved onto a path that did not already pay for it.
    #[inline]
    fn do_move(
        &mut self,
        tt: &TranspositionTable,
        m: Move,
        gives_check: bool,
        capture: bool,
        moved_piece: Option<Piece>,
        ply: usize,
    ) {
        debug_assert_eq!(capture, self.board.capture_stage(m));
        debug_assert_eq!(moved_piece, self.board.piece_on(m.from()));
        self.nodes += 1;
        // Mirror the plain per-worker counter (the deterministic driver of
        // every search decision, D24) into this thread's atomic so the pool
        // can aggregate live `nodes`/`nps` and the node limit across threads
        // (design/13 §3.10). Single-writer load+store — no `fetch_add`; a
        // relaxed store to an owned cache line. Bench reads `self.nodes`, so
        // the count is unchanged.
        self.counters.nodes.store(self.nodes, Ordering::Relaxed);
        let cur = ss(ply);
        let in_check = self.stack[cur].in_check;
        self.stack[cur].current_move = m;
        self.stack[cur].cont_hist = ContHistId::new(in_check, capture, moved_piece, m.to());
        #[cfg(not(feature = "hookc54"))]
        self.board
            .make_move_with(m, gives_check, &mut TtPrefetch { tt });
        // Perf study 54 arm `hookc54` — study 35 §4's fusion, rebuilt on the
        // tree that landed `after_move` instead of it. The record is built
        // inside the make from step 3's registers and handed back through the
        // hook struct, so neither `state.moved` nor `state.captured` is read.
        #[cfg(feature = "hookc54")]
        let mut hooks = TtPrefetchDirty {
            tt,
            dp: DirtyPiece::EMPTY,
        };
        #[cfg(feature = "hookc54")]
        self.board.make_move_with(m, gives_check, &mut hooks);
        // NNUE accumulator maintenance (design/11 §3.4.4): record this move's
        // placement deltas into the child entry at `ply + 1` and mark it
        // uncomputed (the invalidation rule), so a sibling subtree can never
        // reuse a stale accumulator across LMR/PVS re-searches. A no-op — and no
        // derivation cost at all — when NNUE is off, keeping the PSQT hot path
        // and the depth-6 bench bit-identical.
        //
        // Perf study 35: derived from the board AFTER the make, not before it.
        // `DirtyPiece::from_move` re-derived the side to move, the mover
        // (`piece_on(from)` + a panic edge) and the victim (`piece_on(to)`) —
        // three facts `make_move_with` derives for its own Zobrist work and then
        // STORES, two of them into the child `StateInfo` (`moved`/`captured`,
        // put there by study 14 so `unmake_move` need not re-read them).
        // `after_move` reads that store instead of the mailbox, which is why
        // this is a load deletion rather than a code move.
        //
        // This is NOT perf study 13 §5.1, which removed the same duplication the
        // other way — building the record inside `make_move_with` behind Hook C
        // — and rejected it at +41.5 i/n. Study 35 §4 rebuilt that arm and
        // measured **−11.36 i/n**: 13's rejection does NOT reproduce, because
        // study 14 replaced the spilling `mem::take`/224-byte-temp shape that
        // was its stated mechanism. **Do not re-reject the fusion on 13's
        // number.** It is simply worth less than this: it deletes instructions
        // but moves L1d loads by −0.04%, indistinguishable from a byte-identical
        // null, while this shape moves them −0.30…−0.49%. It also has to be a
        // `const WANTS_DIRTY` on the hook type, so the PSQT and ProbCut-probe
        // paths build a record they never read; the `nnue_active()` guard below
        // is what keeps the classical path free.
        //
        // Neither push reads the pre-move board: `push_dirty` takes no board at
        // all, and `push_sota_dirty` deliberately wants the POST-move one (its
        // threat x-rays read the child), so the whole block moves down together.
        if self.eval_ctx.nnue_active() {
            #[cfg(not(feature = "hookc54"))]
            let dp = DirtyPiece::after_move(&self.board, m);
            #[cfg(feature = "hookc54")]
            let dp = hooks.dp;
            self.eval_ctx.push_dirty(dp, ply);
            #[cfg(feature = "sota")]
            self.eval_ctx.push_sota_dirty(dp, ply, &self.board);
        }
    }

    #[inline]
    fn undo_move(&mut self, m: Move) {
        self.board.unmake_move(m);
    }

    /// The §3.8 time-check seam: called at every main-search node entry AND
    /// at qsearch entry (the Reckless re-poll, D31 — bounding stop latency
    /// under a long forcing-capture dive). The poll is gated by a bitmask
    /// on the node counter (`nodes & mask`, D31) rather than a decrementing
    /// field, so both call sites share one cadence for free; the mask
    /// tightens under `go nodes` so tiny budgets are honored.
    ///
    /// Real work — the hard limits a playable engine needs (design/12 §3.8):
    /// `movetime` (wall clock from `Limits::start_time`), `nodes` (single
    /// worker, so `self.nodes` IS the total until SMP aggregation lands,
    /// D23), and the wtime/btime hard wall `elapsed > maximum` from the
    /// worker's [`TimeManager`] — all behind the §3.8 ponder guard. `go
    /// depth` is enforced by the ID loop, `go infinite` never self-stops.
    ///
    /// Two D31 hardening rules gate the real work: the `completed_depth ==
    /// 0` guard (never hard-stop before iteration 1 finishes, guaranteeing
    /// a real bestmove under a tiny budget), and a flat epsilon shaved off
    /// the wall-clock bounds (`maximum`/`movetime`) so we stop a few ms
    /// early to cover the unwind-and-print tail. The `nodes` budget is left
    /// unshaved — an exactness bound, not a clock forfeit risk.
    #[inline]
    fn check_time(&self, stop: &AtomicBool) {
        // Helpers never enforce time or limits (design/13 §3.7): they stop
        // only when they observe the shared `stop`, which the main worker
        // sets against the AGGREGATE node count. One perfectly-predicted
        // branch; at Threads=1 the main worker always passes it (D24).
        if !self.is_main() {
            return;
        }
        if self.nodes & self.check_time_mask != self.check_time_mask {
            return;
        }
        // D31: guarantee a real (if shallow) bestmove — no hard stop of any
        // kind fires before the first iteration has fully completed.
        if self.completed_depth == 0 {
            return;
        }
        // The GUI owns the clock while pondering: NEVER stop here
        // (design/12 §3.8) — `ponderhit`/`stop` are the only ways out of a
        // ponder search, which keeps `bestmove` behind the UCI layer's
        // busy-wait until one of them arrives.
        if self.pool.ponder.load(Ordering::Relaxed) {
            return;
        }
        // Ponderhit already happened AND the soft budget elapsed during the
        // ponder (design/12 step 9): stop now instead of burning to `maximum`
        // on the now-live clock. `stop_on_ponderhit` is only ever set while
        // pondering (soft_stop) and is false for a non-ponder search, so this
        // is inert outside the ponder path (bench-neutral, D24).
        if self.manager.as_ref().is_some_and(|m| m.stop_on_ponderhit) {
            stop.store(true, Ordering::Relaxed);
            return;
        }
        let eps = params::CHECK_TIME_EPSILON_MS();
        // Hard clock wall. `use_time_management` gates the comparison, so
        // depth/nodes/movetime/infinite modes never see it; `tm.elapsed`
        // counts nodes under nodestime, keeping the units consistent with
        // `maximum()`. When the GUI sent only the opponent's clock, `init`
        // left `maximum() == 0`, so the first real poll here stops (§3.7).
        let nodes = self.nodes;
        let tm_stop = self.limits.use_time_management()
            && self.tm.elapsed(|| nodes) > self.tm.maximum() - eps;
        let movetime_stop = self
            .limits
            .movetime
            .is_some_and(|t| self.limits.start_time.elapsed().as_millis() as i64 >= t - eps);
        // The node limit is enforced against the CROSS-THREAD aggregate
        // (design/13 §3.10): with N threads a `go nodes` budget is a total.
        // At Threads=1 the aggregate equals `self.nodes` (the per-node mirror
        // in `do_move`), so this is bit-identical for single-thread `go nodes`
        // / nodes-mode bench; and `go depth` (the fingerprint) never sets a
        // node limit, so the depth-6 bench is unaffected either way (D24).
        let nodes_stop = self
            .limits
            .nodes
            .is_some_and(|n| self.pool.nodes_searched() >= n);
        if tm_stop || movetime_stop || nodes_stop {
            stop.store(true, Ordering::Relaxed);
        }
    }

    // Soft-stop factors (design/12 §3.9). Pure functions of their inputs so
    // every §3.9 constant is pinned by a golden unit test at its clamp
    // edges; the decision that multiplies them lives in [`Self::soft_stop`].

    /// Factor 1 — `fallingEval`: think longer when the score is dropping
    /// (design/12 §3.9). Clamped to `[FALLING_EVAL_MIN, FALLING_EVAL_MAX]`.
    fn falling_eval(prev_avg: Value, best_value: Value, iter_value_old: Value) -> f64 {
        ((params::FALLING_EVAL_BASE()
            + params::FALLING_EVAL_PREV_AVG_COEFF() * f64::from(prev_avg - best_value)
            + params::FALLING_EVAL_ITER_COEFF() * f64::from(iter_value_old - best_value))
            / params::FALLING_EVAL_DIV())
        .clamp(params::FALLING_EVAL_MIN(), params::FALLING_EVAL_MAX())
    }

    /// Factor 2 — best-move-stability `reduction` (design/12 §3.9). Returns
    /// `(time_reduction, reduction)`: `time_reduction` is stored back as
    /// next move's `previous_time_reduction`; `reduction` is the budget
    /// multiplier. Stable-for-many-iterations → higher `time_reduction` →
    /// lower `reduction` → less time (partially giving savings back).
    fn stability_reduction(stability_iters: i32, previous_time_reduction: f64) -> (f64, f64) {
        let time_reduction = lerp(
            f64::from(stability_iters),
            params::TIME_REDUCTION_STABILITY_LO(),
            params::TIME_REDUCTION_STABILITY_HI(),
            params::TIME_REDUCTION_MIN(),
            params::TIME_REDUCTION_MAX(),
        )
        .clamp(params::TIME_REDUCTION_MIN(), params::TIME_REDUCTION_MAX());
        let reduction = (params::REDUCTION_BASE() + previous_time_reduction)
            / (params::REDUCTION_DEN_COEFF() * time_reduction);
        (time_reduction, reduction)
    }

    /// Factor 3 — `bestMoveInstability`: think longer when the best move
    /// keeps flipping (design/12 §3.9). `threads` is 1 until SMP (P11).
    fn best_move_instability(tot_best_move_changes: f64, threads: usize) -> f64 {
        params::INSTABILITY_BASE()
            + params::INSTABILITY_CHANGES_COEFF() * tot_best_move_changes / threads as f64
    }

    /// Factor 4 — `highBestMoveEffort`: stop sooner once the lion's share
    /// of nodes already went into the best move (design/12 §3.9). Clamped
    /// to `[EFFORT_SCALE_AT_HI, EFFORT_SCALE_AT_LO]`.
    fn high_best_move_effort(effort: u64, total_nodes: u64) -> f64 {
        let nodes_effort =
            (effort as i64 * params::EFFORT_UNIT_SCALE() / total_nodes.max(1) as i64) as f64;
        lerp(
            nodes_effort,
            params::EFFORT_LERP_LO(),
            params::EFFORT_LERP_HI(),
            params::EFFORT_SCALE_AT_LO(),
            params::EFFORT_SCALE_AT_HI(),
        )
        .clamp(params::EFFORT_SCALE_AT_HI(), params::EFFORT_SCALE_AT_LO())
    }

    /// The between-iterations soft stop (design/12 §3.9, full four-factor
    /// form — doc 12 step 8): once the soft budget is spent, starting
    /// another iteration is pointless — it would almost certainly be cut
    /// off by the `maximum` hard wall before completing, wasting clock a
    /// later move could have used.
    ///
    /// `total_time = optimum · fallingEval · reduction · bestMoveInstability
    /// · highBestMoveEffort`, then capped at
    /// [`params::ONE_REPLY_TOTAL_TIME_CAP()`] with exactly one legal move
    /// (the result cannot change the move played). The search stops when
    /// `elapsed > min(total_time, maximum)`, or on the two mate early-stops
    /// (every PV line mates in `<=` [`params::SOFT_STOP_MATE_PLIES()`], or we
    /// are mated in [`params::SOFT_STOP_MATED_PLIES()`]). Otherwise it sets
    /// `increase_depth` for the next iteration: honest deepening only while
    /// at most [`params::INCREASE_DEPTH_FRACTION()`] of the budget is spent.
    ///
    /// Guards mirror the doc's `use_time_management() && !stop` gate plus
    /// §3.8's ponder rule: while pondering the GUI owns the clock, so
    /// nothing may self-stop (the doc's "remember the verdict in
    /// `stop_on_ponderhit` instead" conversion is the ponder state machine,
    /// doc 12 step 9 / P11 — until then a pondering search simply never
    /// self-stops, and `increase_depth` stays `true`). `tm.elapsed` counts
    /// nodes under `nodestime`, keeping the units consistent with the
    /// budgets (§3.11) — the fake-clock seam the unit tests use.
    ///
    /// Maintains the per-iteration TM state (`iter_value` ring,
    /// `tm_previous_time_reduction`) as a side effect, hence `&mut self`.
    ///
    /// `best_value` is SF's `bestValue` — the value the LAST MultiPV line's
    /// root search settled on (`root_moves[0].score` at MultiPV 1, the
    /// weakest line at MultiPV > 1), NOT `root_moves[0].score` unconditionally;
    /// it drives `fallingEval` and seeds the `iter_value` ring.
    fn soft_stop(
        &mut self,
        stop: &AtomicBool,
        multi_pv: usize,
        tot_best_move_changes: f64,
        best_value: Value,
    ) {
        // Only the main worker manages the clock and writes `increase_depth`
        // (design/13 §3.7); helpers keep deepening until they see `stop`. At
        // Threads=1 the worker is always main (D24).
        if !self.is_main() || !self.limits.use_time_management() || stop.load(Ordering::Relaxed) {
            return;
        }
        let pondering = self.pool.ponder.load(Ordering::Relaxed);

        // The four §3.9 factors (constants pinned by golden tests).
        let falling_eval = Self::falling_eval(
            self.tm_best_previous_average_score,
            best_value,
            self.iter_value[self.iter_idx],
        );
        let (time_reduction, reduction) = Self::stability_reduction(
            self.root_depth - self.last_best_move_depth,
            self.tm_previous_time_reduction,
        );
        // `bestMoveInstability` averages the cross-thread change count over
        // the thread count (design/13 §3.10). At Threads=1 the divisor is 1
        // (bench-neutral, D24).
        let instability =
            Self::best_move_instability(tot_best_move_changes, self.pool.shells.len());
        let effort = Self::high_best_move_effort(self.root_moves[0].effort, self.nodes);

        let mut total_time =
            self.tm.optimum() as f64 * falling_eval * reduction * instability * effort;
        // Head-free decisiveness scale (searchopt 2026-07-28). Unlike the two
        // factors below it needs no net head, so it is NOT `sota`-gated and is
        // live on every build. Exactly 1.0 at the default `TM_SCORE_C = 0`, so
        // `total_time` is bit-identical and the four-factor golden tests hold.
        total_time *= score_tm_factor(best_value, params::TM_SCORE_C(), params::TM_SCORE_KNEE());
        // EXP-UNC-1 consumer 2: position-condition the time budget by the root
        // value uncertainty. unc_tm_factor is exactly 1.0 when disabled or before
        // a root σ̂ exists, so total_time is unchanged and the four-factor golden
        // tests hold; it only bites with a variance net + a nonzero coeff.
        #[cfg(feature = "sota")]
        {
            total_time *= unc_tm_factor(self.root_sigma, self.sigma_bar, self.unc_tm_c2);
            // Multi-head WDL consumer: bank clock on roots the WDL head calls
            // decided. Exactly 1.0 when disabled / headless — golden tests hold.
            total_time *= wdl_tm_factor(self.root_wdl_d, self.wdl_tm_c3);
        }
        if self.root_moves.len() == 1 {
            total_time = total_time.min(params::ONE_REPLY_TOTAL_TIME_CAP());
        }

        let nodes = self.nodes;
        let elapsed = self.tm.elapsed(|| nodes) as f64;
        // Mate early-stops (design/12 §3.9): every reported PV line already
        // mates in <= SOFT_STOP_MATE_PLIES, or we are mated in exactly
        // SOFT_STOP_MATED_PLIES — either way deeper search cannot help.
        let mate_stop = self.root_moves[multi_pv - 1].score
            >= mate_in(params::SOFT_STOP_MATE_PLIES())
            || self.root_moves[0].score == mated_in(params::SOFT_STOP_MATED_PLIES());

        // TM TRACE (env-gated, off by default => bench-neutral, no effect on
        // search): dumps the five soft-stop factors at every decision so a
        // long think can be attributed to a specific term instead of guessed
        // at. Enable with ARCHON_TM_TRACE=1 and read stderr.
        if std::env::var_os("ARCHON_TM_TRACE").is_some() {
            eprintln!(
                "TMTRACE d={} opt={} max={} fe={:.3} red={:.3} inst={:.3} eff={:.3} \
                 total={:.0} elapsed={:.0} chg={:.2} \
                 bv={} prevavg={} iterold={} dprev={} diter={} stab={}",
                self.root_depth,
                self.tm.optimum(),
                self.tm.maximum(),
                falling_eval,
                reduction,
                instability,
                effort,
                total_time,
                elapsed,
                tot_best_move_changes,
                // Raw fallingEval inputs. The factor is a clamped affine form
                // in these two deltas, and both are in OUR internal Value
                // units — the coefficients were taken from SF, whose units
                // differ, so the deltas are the thing to look at directly
                // rather than re-deriving a scale ratio (which has produced
                // two contradictory answers).
                best_value,
                self.tm_best_previous_average_score,
                self.iter_value[self.iter_idx],
                self.tm_best_previous_average_score - best_value,
                self.iter_value[self.iter_idx] - best_value,
                self.root_depth - self.last_best_move_depth
            );
        }

        if elapsed > total_time.min(self.tm.maximum() as f64) || mate_stop {
            if pondering {
                // While pondering the GUI owns the clock — never self-stop.
                // Instead remember the verdict (design/12 step 9): a later
                // `ponderhit` then stops at once via `check_time` rather than
                // burning to `maximum` on our now-live clock.
                if let Some(m) = &mut self.manager {
                    m.stop_on_ponderhit = true;
                }
            } else {
                stop.store(true, Ordering::Relaxed);
            }
        } else {
            // Shared across the pool (design/13 §3.7): the main worker writes
            // it, every worker reads it at the top of its ID loop.
            let deepen = pondering || elapsed <= total_time * params::INCREASE_DEPTH_FRACTION();
            self.pool.increase_depth.store(deepen, Ordering::Relaxed);
        }

        // Maintain the fallingEval ring and the reduction memory for the
        // next iteration / next `go` (design/12 §3.9). Record this
        // iteration's `time_reduction` in the RUNNING field only — the
        // numerator `tm_previous_time_reduction` stays the per-move constant
        // seeded at `go` start (SF: the local `timeReduction` is recomputed,
        // `previousTimeReduction` is not touched until search end).
        self.iter_value[self.iter_idx] = best_value;
        self.iter_idx = (self.iter_idx + 1) & (ITER_VALUE_RING - 1);
        self.tm_time_reduction = time_reduction;
    }

    /// Poll-cadence bitmask for [`Self::check_time`] (design/12 §3.8;
    /// Reckless bundle D31). Default cadence is [`params::CHECK_TIME_CALLS`]
    /// node-entries (`mask = CALLS - 1`, a power of two). Under `go nodes`
    /// the cadence tightens to the largest power of two `<= min(CALLS,
    /// nodes / 1024)` so a tiny node budget is polled often enough to be
    /// honored (a mask of 0 polls every node).
    fn compute_check_time_mask(&self) -> u64 {
        match self.limits.nodes {
            Some(n) => {
                let cadence = (n / 1024).clamp(1, params::CHECK_TIME_CALLS as u64);
                // Round DOWN to a power of two, then subtract 1 for the mask.
                let pow2 = cadence.next_power_of_two();
                let pow2 = if pow2 > cadence { pow2 >> 1 } else { pow2 };
                pow2 - 1
            }
            None => params::CHECK_TIME_CALLS as u64 - 1,
        }
    }
}

/// Linear interpolation with clamping left to the caller: maps `x` from the
/// segment `[x0, x1]` onto `[y0, y1]` (design/12 §3.9's `lerp`). Extrapolates
/// outside the segment — every soft-stop caller clamps the result.
/// Cached env probe for the NMP node-gate A/B (read once; a per-node
/// getenv would distort the very trees being measured).
pub(crate) fn nmp_allnode_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARCHON_NMP_ALLNODE").is_some())
}

fn lerp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

#[cfg(test)]
mod tests;
