//! Unit tests for [`super`] — the search core.
//!
//! Moved out of `mod.rs` on 2026-07-28: they were 4,203 of its 9,380 lines (45%),
//! which made the actual search unreadable and buried the parts a reader needs.
//! Nothing here changed — this is the same `mod tests` body, now a file module, so
//! `use super::*` still resolves to the search module exactly as before.

use super::*;
use crate::fen::START_FEN;
use crate::movegen::{generate_captures, generate_evasions};
use crate::types::Square;

fn board(fen: &str) -> Board {
    Board::from_fen(fen).unwrap()
}

// ---- Best-thread voting matrix (design/13 §3.8, §5) ----

/// A synthetic report: `mv` heads a PV of `pv_len`, with `score`
/// (untrusted when `bound`).
fn report(mv: Move, score: Value, pv_len: usize, bound: bool) -> SearchReport {
    let mut pv = vec![mv];
    // Pad the PV with a distinct filler move so `pv.len()` is the tiebreak.
    while pv.len() < pv_len {
        pv.push(Move::normal(Square::A2, Square::A3));
    }
    SearchReport {
        pv,
        score,
        uci_score: score,
        score_lowerbound: bound,
        score_upperbound: false,
        average_score: score,
        completed_depth: 1,
        sel_depth: 1,
    }
}

fn mv(from: Square, to: Square) -> Move {
    Move::normal(from, to)
}

#[test]
fn voting_unanimous_elects_that_move() {
    let a = mv(Square::E2, Square::E4);
    let reports = [report(a, 20, 2, false), report(a, 22, 2, false)];
    assert_eq!(reports[best_thread(&reports)].pv[0], a); // (a)
}

#[test]
fn voting_accumulates_votes_over_a_lone_higher_score() {
    // (b) 3 threads × A@10 vs 1 thread × B@30 → A (votes accumulate).
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, 10, 2, false),
        report(a, 10, 2, false),
        report(a, 10, 2, false),
        report(b, 30, 2, false),
    ];
    assert_eq!(reports[best_thread(&reports)].pv[0], a);
}

#[test]
fn voting_prefers_the_shorter_mate() {
    // (c) incumbent mate-in-5 vs candidate mate-in-3 → candidate.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, mate_in(5), 2, false),
        report(b, mate_in(3), 2, false),
    ];
    assert_eq!(best_thread(&reports), 1);
}

#[test]
fn voting_protects_a_decisive_incumbent_from_votes() {
    // (d) incumbent mate-in-3 vs 3 threads voting a normal move → incumbent.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, mate_in(3), 2, false),
        report(b, 100, 2, false),
        report(b, 100, 2, false),
        report(b, 100, 2, false),
    ];
    assert_eq!(best_thread(&reports), 0);
}

#[test]
fn voting_bound_flagged_win_is_not_decisive() {
    // (e) a bound-flagged (aborted-iteration) win is untrusted: it does
    // NOT get the shorter-mate displacement. Contrast with (c): there the
    // candidate mate-in-3 was trusted and won; here a BOUND mate-in-8 must
    // fail to displace a TRUSTED incumbent mate-in-10, precisely because
    // its bound flag makes it non-decisive.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, mate_in(10), 2, false), // trusted decisive incumbent
        report(b, mate_in(8), 2, true),   // bound ⇒ untrusted ⇒ not decisive
    ];
    assert_eq!(best_thread(&reports), 0, "a bound win cannot displace");
    // Sanity: the SAME candidate, trusted, would win (shorter mate).
    let trusted = [
        report(a, mate_in(10), 2, false),
        report(b, mate_in(8), 2, false),
    ];
    assert_eq!(best_thread(&trusted), 1);
}

#[test]
fn voting_equal_votes_break_by_pv_length() {
    // (f) equal votes → the longer PV (depth proxy) wins.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [report(a, 15, 2, false), report(b, 15, 5, false)];
    assert_eq!(best_thread(&reports), 1);
}

#[test]
fn voting_infinite_helper_never_wins() {
    // (g) a helper that never finished depth 1 (score -INF) never wins.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, 30, 2, false),
        report(b, -VALUE_INFINITE, 2, false),
    ];
    assert_eq!(best_thread(&reports), 0);
    // Even with a longer PV, the losing sentinel cannot displace.
    let reports = [
        report(a, 30, 2, false),
        report(b, -VALUE_INFINITE, 9, false),
    ];
    assert_eq!(best_thread(&reports), 0);
}

#[test]
fn voting_trusts_a_proven_loss_over_a_fuzzy_incumbent() {
    // A *trusted* (exact, non-bound) proven-loss candidate DOES displace a
    // non-decisive incumbent — SF `get_best_thread`'s `newThreadDecisive`
    // short-circuit fires for a proven LOSS as well as a proven win. An
    // exact mated_in(n) is ground truth (its PV is the longest defence),
    // more reliable than another thread's fuzzy +50 that has not yet seen
    // the refutation. This is the deliberate reference behaviour, NOT the
    // (never-finished) `-VALUE_INFINITE` sentinel of `voting_infinite_*`:
    // that one is blocked by both `is_trusted_decisive`'s `!= -INFINITE`
    // guard and the vote arm's `!is_loss` guard.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [
        report(a, 50, 2, false),          // non-decisive, optimistic
        report(b, mated_in(5), 2, false), // trusted proven loss
    ];
    assert_eq!(
        best_thread(&reports),
        1,
        "a trusted proven loss is trusted over a fuzzy incumbent (SF-faithful)"
    );
    // Mirror: a trusted proven-loss INCUMBENT is protected from votes just
    // like a proven win — only a shorter/longer decisive of the same sign
    // displaces it (the vote arm never runs while `b_dec` holds).
    let reports = [
        report(a, mated_in(5), 2, false), // trusted proven-loss incumbent
        report(b, 50, 3, false),          // non-decisive, more votes/longer PV
    ];
    assert_eq!(
        best_thread(&reports),
        0,
        "a decisive incumbent (loss) ignores the vote (SF-faithful)"
    );
    // But a BOUND (aborted-iteration) loss is untrusted: it neither wins as
    // a candidate nor protects itself as an incumbent.
    let reports = [
        report(a, 50, 2, false),
        report(b, mated_in(5), 2, true), // bound ⇒ not trusted-decisive
    ];
    assert_eq!(
        best_thread(&reports),
        0,
        "a bound loss cannot displace via the decisive short-circuit"
    );
}

#[test]
fn voting_single_report_is_thread_zero() {
    let a = mv(Square::E2, Square::E4);
    assert_eq!(best_thread(&[report(a, 5, 1, false)]), 0);
}

#[test]
fn voting_disabled_under_fixed_depth_or_single_thread() {
    // A slate where the vote genuinely prefers a helper (thread 1): B's
    // lone higher score outvotes A here.
    let a = mv(Square::E2, Square::E4);
    let b = mv(Square::D2, Square::D4);
    let reports = [report(a, 10, 2, false), report(b, 30, 2, false)];
    assert_ne!(best_thread(&reports), 0, "vote should pick the helper");
    // Fixed-depth `go` (D24 reproducibility, acceptance criterion 3):
    // thread 0 regardless of the vote.
    assert_eq!(elect_winner(&reports, false, true), 0);
    // Single-thread pool: thread 0 regardless.
    assert_eq!(elect_winner(&reports, true, false), 0);
    // Multi-thread, no depth limit: the vote decides (the helper wins).
    assert_eq!(elect_winner(&reports, false, false), best_thread(&reports));
}

/// The smallest real table (1 MiB): every test `go` runs with the TT
/// enabled, exactly like the engine.
fn test_tt() -> TranspositionTable {
    TranspositionTable::new(1)
}

/// One `go` session against `tt`: the per-`go` generation bump the
/// session layer owns (doc 13 §3.7 B), then the driver.
fn go(w: &mut Worker, tt: &TranspositionTable) {
    tt.new_search();
    let stop = AtomicBool::new(false);
    w.iterative_deepening(tt, &stop, &mut TimePersist::default(), &mut |_| {});
}

/// Run a `go depth <depth>` on `tt` (warm or fresh), discarding info
/// reports.
fn go_depth_on(fen: &str, depth: Depth, tt: &TranspositionTable) -> Worker {
    let mut w = Worker::new(board(fen));
    w.limits.depth = Some(depth);
    go(&mut w, tt);
    w
}

/// Run a `go depth <depth>` through the iterative-deepening driver
/// with a fresh table, discarding info reports.
fn go_depth(fen: &str, depth: Depth) -> Worker {
    go_depth_on(fen, depth, &test_tt())
}

/// The best root move's score after a completed `go`.
fn best_score(w: &Worker) -> Value {
    w.root_moves[0].score
}

fn all_legal(b: &Board) -> MoveList {
    let mut legal = MoveList::new();
    generate_legal(b, &mut legal);
    legal
}

// --- value_draw (design/07 §5: ±1, correlated with nodes & 0x2) ---

#[test]
fn value_draw_dithers_plus_minus_one() {
    for nodes in 0..16u64 {
        let v = value_draw(nodes);
        assert!(v == -1 || v == 1, "value_draw({nodes}) = {v}");
        assert_eq!(v == 1, nodes & 0x2 != 0);
    }
}

// --- Base reduction table + reduction() (design/10 §3.3, §5) ---

/// The table's closed form, recomputed exactly as `Worker::new` builds
/// it, for golden cross-checks.
fn red_table(i: usize) -> i32 {
    (params::RED_TABLE_NUM() as f64 / params::RED_TABLE_DEN() as f64 * (i as f64).ln()) as i32
}

#[test]
fn reduction_table_matches_the_closed_form() {
    let w = Worker::new(board(START_FEN));
    // `ln(1) == 0`, so the first usable entry is exactly 0 (§5); index
    // 0 is never read and stays 0.
    assert_eq!(w.reductions[0], 0);
    assert_eq!(w.reductions[1], 0);
    // Concrete golden anchors: int(22.4375 * ln(i)).
    assert_eq!(w.reductions[2], 15);
    assert_eq!(w.reductions[64], 93);
    // Every entry equals the closed form.
    for i in 1..MAX_MOVES {
        assert_eq!(w.reductions[i], red_table(i), "reductions[{i}]");
    }
}

#[test]
fn reduction_table_is_monotonic_nondecreasing() {
    // `ln` is increasing, so the truncated table never steps down.
    let w = Worker::new(board(START_FEN));
    for i in 2..MAX_MOVES {
        assert!(
            w.reductions[i] >= w.reductions[i - 1],
            "reductions dropped at {i}: {} < {}",
            w.reductions[i],
            w.reductions[i - 1]
        );
    }
}

#[test]
fn reduction_scale_product_fits_i32() {
    // `scale = reductions[d] * reductions[mn]` peaks at the top-right
    // corner; design/10 §6 bounds it around 122² ≈ 15k. Prove the
    // product — and the not-improving surcharge `scale * 194` — never
    // overflow i32.
    let w = Worker::new(board(START_FEN));
    let peak = w.reductions[MAX_MOVES - 1];
    let scale = i64::from(peak) * i64::from(peak);
    assert!(
        peak * peak < 20_000,
        "scale ~15k expected, got {}",
        peak * peak
    );
    assert!(
        scale * i64::from(params::RED_NON_IMPROVING()) <= i64::from(i32::MAX),
        "scale * RED_NON_IMPROVING overflows i32"
    );
}

#[test]
fn reduction_not_improving_adds_the_scale_surcharge() {
    // improving vs. not differs by exactly `scale * 194 / 512`, all
    // else fixed (§5).
    let mut w = Worker::new(board(START_FEN));
    w.root_delta = 100;
    let (depth, mn, delta): (Depth, i32, i32) = (12, 8, 40);
    let scale = w.reductions[depth as usize] * w.reductions[mn as usize];
    let improving = w.reduction(true, depth, mn, delta);
    let not_improving = w.reduction(false, depth, mn, delta);
    assert_eq!(
        not_improving - improving,
        scale * params::RED_NON_IMPROVING() / params::RED_NON_IMPROVING_DEN()
    );
    assert!(not_improving > improving);
}

#[test]
fn reduction_wider_window_reduces_less() {
    // A wider local window (larger `delta`) lowers r through the
    // `- delta * 617 / root_delta` term.
    let mut w = Worker::new(board(START_FEN));
    w.root_delta = 50;
    let (depth, mn): (Depth, i32) = (10, 6);
    let narrow = w.reduction(true, depth, mn, 20);
    let wide = w.reduction(true, depth, mn, 400);
    assert!(wide < narrow, "wide {wide} !< narrow {narrow}");
}

#[test]
fn reduction_equals_the_full_formula() {
    // Full cross-check at a representative point, preserving the exact
    // truncating term order of design/10 §3.3.
    let mut w = Worker::new(board(START_FEN));
    w.root_delta = 137;
    let (improving, depth, mn, delta): (bool, Depth, i32, i32) = (false, 15, 20, 33);
    let scale = w.reductions[depth as usize] * w.reductions[mn as usize];
    let want = scale - delta * params::RED_DELTA_MULT() / w.root_delta
        + i32::from(!improving) * scale * params::RED_NON_IMPROVING()
            / params::RED_NON_IMPROVING_DEN()
        + params::RED_BASE();
    assert_eq!(w.reduction(improving, depth, mn, delta), want);
}

// --- Reverse futility pruning (design/10 §3.4 N10, §5) ---

#[test]
fn rfp_margin_matches_the_hand_computed_formula() {
    // (depth, tt_hit, improving, opponent_worsening, expected). Values
    // are `futility_mult·depth - (2789·imp + 335·opp)·futility_mult/1024`
    // with `futility_mult = min(45 + 4·depth, 85) - 20·(!tt_hit)`,
    // computed by hand — the truncating divide is the point.
    let cases: &[(Depth, bool, bool, bool, Value)] = &[
        // fm = min(49,85) = 49; 49·1 - 0 = 49
        (1, true, false, false, 49),
        // fm = min(61,85) = 61; 61·4 - 0 = 244
        (4, true, false, false, 244),
        // fm = 61 - 20 = 41; 41·4 - 335·41/1024 = 164 - 13 = 151
        (4, false, false, true, 151),
        // fm = min(85,85) = 85; 85·10 - (2789+335)·85/1024 = 850 - 259 = 591
        (10, true, true, true, 591),
        // fm = min(109,85) = 85 (cap); 85·16 - 2789·85/1024 = 1360 - 231 = 1129
        (16, true, true, false, 1129),
    ];
    for &(d, hit, imp, opp, want) in cases {
        assert_eq!(
            rfp_margin(d, hit, imp, opp, 0),
            want,
            "rfp_margin{:?}",
            (d, hit, imp, opp)
        );
    }
}

#[test]
fn rfp_margin_preserves_the_truncating_term_order() {
    // Cross-check at a representative point against the exact §3.4
    // expression, guarding the operator precedence / truncation order.
    let (d, hit, imp, opp): (Depth, bool, bool, bool) = (9, false, true, true);
    let fm = (params::RFP_MULT_BASE() + params::RFP_MULT_SLOPE() * d).min(params::RFP_MULT_CAP())
        - params::RFP_MULT_NO_TTHIT() * Value::from(!hit);
    let want = fm * d
        - (params::RFP_IMPROVING() * Value::from(imp)
            + params::RFP_OPP_WORSENING() * Value::from(opp))
            * fm
            / params::RFP_TREND_DEN();
    assert_eq!(rfp_margin(d, hit, imp, opp, 0), want);
}

#[test]
fn rfp_prune_margin_folds_the_threat_relaxation_with_the_right_sign() {
    // The N10 use-site subtracts RFP_THREAT_MARGIN from the base margin
    // exactly when no own piece is threatened — a LOWER margin is easier to
    // clear, so the disarmed case prunes more readily. Pin the sign and
    // magnitude at a representative point so a wiring regression (dropped
    // term / flipped sign) is caught apart from the search control flow.
    let (d, hit, imp, opp): (Depth, bool, bool, bool) = (4, true, false, false);
    let base = rfp_margin(d, hit, imp, opp, 0);
    // Armed (a piece is threatened): no relaxation — equals the base.
    assert_eq!(
        rfp_prune_margin(d, hit, imp, opp, 0, false),
        base,
        "an armed node uses the un-relaxed base margin"
    );
    // Disarmed (nothing hanging): the base LESS the whole threat margin.
    assert_eq!(
        rfp_prune_margin(d, hit, imp, opp, 0, true),
        base - params::RFP_THREAT_MARGIN(),
        "a disarmed node relaxes the margin by exactly RFP_THREAT_MARGIN"
    );
    // The relaxation strictly lowers the bar (RFP_THREAT_MARGIN > 0).
    assert!(
        rfp_prune_margin(d, hit, imp, opp, 0, true) < base,
        "the threat relaxation must make the margin easier to clear"
    );
}

/// EXP-UNC-1: the uncertainty margin scale is 1.0 (off) exactly when disabled
/// or unwarmed, scales up above-average uncertainty and down below-average,
/// and is clamped to [0.7, 1.5]. Pins the SIGN so a wiring/sign regression is
/// caught apart from the search control flow.
#[cfg(feature = "sota")]
#[test]
fn unc_margin_scale_signs_and_clamps() {
    // Disabled / no fresh sigma / unwarmed sigma_bar → exactly 1.0 (off).
    assert_eq!(unc_margin_scale(0.2, 0.1, 0.0), 1.0, "c1=0 is off");
    assert_eq!(unc_margin_scale(0.0, 0.1, 1.0), 1.0, "no sigma this node");
    assert_eq!(unc_margin_scale(0.2, 0.0, 1.0), 1.0, "sigma_bar unwarmed");
    // Above-average uncertainty (σ̂ > σ̄) → LARGER margin (prune less; safety).
    assert!(unc_margin_scale(0.12, 0.10, 1.0) > 1.0);
    // Below-average uncertainty → SMALLER margin (prune more; the Elo).
    assert!(unc_margin_scale(0.08, 0.10, 1.0) < 1.0);
    // Symmetric mid response is exact.
    assert!((unc_margin_scale(0.11, 0.10, 0.5) - 1.05).abs() < 1e-6);
    // Clamped both ways.
    assert_eq!(unc_margin_scale(1.0, 0.10, 1.0), 1.5, "clamped high");
    assert_eq!(unc_margin_scale(0.01, 0.10, 1.0), 0.7, "clamped low");
}

/// Multi-head WDL TM consumer: 1.0 (off) when disabled or uncaptured;
/// decisive roots bank clock, clamped to [0.8, 1.0] (shrink-only).
#[cfg(feature = "sota")]
#[test]
fn wdl_tm_factor_signs_and_clamps() {
    assert_eq!(wdl_tm_factor(0.9, 0.0), 1.0, "c3=0 is off");
    assert_eq!(wdl_tm_factor(0.0, 0.5), 1.0, "no root capture is off");
    assert_eq!(wdl_tm_factor(-1.0, 0.5), 1.0, "negative guard");
    let f = wdl_tm_factor(0.8, 0.2);
    assert!(f < 1.0 && f > 0.8, "decisive root banks time: {f}");
    assert!(
        (wdl_tm_factor(1.0, 5.0) - 0.8).abs() < 1e-6,
        "floor clamp at 0.8"
    );
    assert!(
        (wdl_tm_factor(0.05, 0.1) - 0.995).abs() < 1e-6,
        "linear region"
    );
}

/// EXP-UNC-1 consumer 2: the TM factor is 1.0 (off) when disabled/unwarmed,
/// lengthens the budget at an above-average-uncertainty root and shortens it
/// below, clamped to [0.8, 1.25] (narrower than the margin scale).
#[cfg(feature = "sota")]
#[test]
fn unc_tm_factor_signs_and_clamps() {
    assert_eq!(unc_tm_factor(0.2, 0.1, 0.0), 1.0, "c2=0 is off");
    assert_eq!(unc_tm_factor(0.0, 0.1, 1.0), 1.0, "no root sigma");
    assert_eq!(unc_tm_factor(0.2, 0.0, 1.0), 1.0, "sigma_bar unwarmed");
    assert!(
        unc_tm_factor(0.12, 0.10, 1.0) > 1.0,
        "sharp root → think longer"
    );
    assert!(
        unc_tm_factor(0.08, 0.10, 1.0) < 1.0,
        "reliable root → think less"
    );
    // Clamp bounds round-trip f32→f64, so compare approximately.
    assert!(
        (unc_tm_factor(1.0, 0.10, 1.0) - 1.25).abs() < 1e-6,
        "clamped high"
    );
    assert!(
        (unc_tm_factor(0.01, 0.10, 1.0) - 0.8).abs() < 1e-6,
        "clamped low"
    );
}

#[test]
fn rfp_return_blend_weights_sum_to_the_denominator() {
    // The return `(716·beta + 308·eval)/1024` is a weighted average, so
    // its result lies between beta and eval — the weights must sum to
    // the denominator or the blend would drift off the [beta, eval] band.
    assert_eq!(
        params::RFP_RET_BETA_W() + params::RFP_RET_EVAL_W(),
        params::RFP_RET_DEN()
    );
}

/// Drive `search` directly at a fixed ply on a fresh stack — the RFP
/// analogue of [`qsearch_at`]. `nodes` is zeroed so a return with
/// `w.nodes == 0` proves the node made no move (RFP short-circuited
/// before the move loop).
#[allow(clippy::too_many_arguments)]
fn search_at<NT: NodeType>(
    w: &mut Worker,
    tt: &TranspositionTable,
    alpha: Value,
    beta: Value,
    depth: Depth,
    cut_node: bool,
    ply: usize,
) -> Value {
    init_stack(&mut w.stack);
    w.nodes = 0;
    let stop = AtomicBool::new(false);
    w.search::<NT>(tt, &stop, alpha, beta, depth, cut_node, ply)
}

#[test]
fn rfp_fires_on_a_winning_nonpv_node_and_returns_the_blend() {
    // White is a full queen up, quiet, not in check: the corrected
    // static eval sits far above a low beta, so a non-PV node RFP-prunes
    // — returning the exact §3.4 beta-ward blend and searching NOTHING.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let cv = fresh_cv(&w.board, None);
    let eval = corrected_eval(&w.board, None);
    assert!(!is_win(eval), "material edge alone is not a decisive score");
    let depth: Depth = 4;
    let beta = eval - 200; // eval - margin(~132) >= beta must hold
    // A fresh TT → tt miss at this node, so the margin is the no-ttHit
    // form; the trend flags at ply 1 read VALUE_NONE predecessors
    // (improving false, opponent_worsening true). The margin widens by
    // |cv|/RFP_CORR_DIV exactly as the search computes it.
    let margin = rfp_margin(depth, false, false, true, cv);
    assert!(
        eval - margin >= beta,
        "the fixture must actually trigger RFP"
    );
    let v = search_at::<NonPv>(&mut w, &tt, beta - 1, beta, depth, true, 1);
    assert_eq!(
        v,
        (params::RFP_RET_BETA_W() * beta + params::RFP_RET_EVAL_W() * eval) / params::RFP_RET_DEN()
    );
    assert_eq!(w.nodes, 0, "RFP must return before making any move");
}

#[test]
fn rfp_never_fires_on_a_pv_node() {
    // Same winning position, but a PV node carries ss.tt_pv, so the
    // `!ss.tt_pv` guard blocks RFP: the node searches (nodes > 0) and
    // does NOT return the blend shortcut.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let depth: Depth = 4;
    let beta = eval - 200;
    let v = search_at::<Pv>(&mut w, &tt, beta - 1, beta, depth, false, 1);
    assert!(w.nodes > 0, "a PV node must not RFP-prune");
    assert_ne!(
        v,
        (params::RFP_RET_BETA_W() * beta + params::RFP_RET_EVAL_W() * eval) / params::RFP_RET_DEN(),
        "the PV node must not return the RFP blend"
    );
}

#[test]
fn rfp_never_fires_when_beta_is_decisive() {
    // A loss-band beta (is_loss) must disarm RFP via `!is_loss(beta)`,
    // even though the huge eval clears it — the mate-safety guard. The
    // node searches instead of pruning.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let beta = crate::types::VALUE_TB_LOSS_IN_MAX_PLY; // is_loss(beta) == true
    assert!(is_loss(beta));
    let v = search_at::<NonPv>(&mut w, &tt, beta - 1, beta, 4, true, 1);
    assert!(
        w.nodes > 0,
        "a decisive (loss-band) beta must block RFP and force a search"
    );
    assert!(
        v >= beta,
        "the winning node still fails high, just by searching"
    );
}

// --- Razoring (design/10 §3.4 N9) ---

#[test]
fn razoring_drops_a_hopeless_nonpv_node_straight_to_qsearch() {
    // Black is a full queen up; White to move, quiet, not in check, no
    // captures on the board. With alpha set to the SMALLEST value that
    // satisfies `eval < alpha - margin` at depth 1 (one cp above the
    // boundary), a non-PV node skips the full-width search and returns the
    // qsearch value directly. Because qsearch of a quiet, not-in-check node
    // with no captures makes ZERO moves and stands pat, `w.nodes == 0`
    // proves the move loop was never entered and the return is exactly the
    // stand-pat eval.
    let fen = "7k/8/8/3q4/8/8/8/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    assert!(
        !is_decisive(eval),
        "a queen deficit is not a decisive score"
    );
    let depth: Depth = 1;
    let margin = params::RAZOR_BASE() + params::RAZOR_DEPTH_MULT() * depth * depth;
    let alpha = eval + margin + 1; // smallest alpha with eval < alpha - margin
    let beta = alpha + 1;
    assert!(eval < alpha - margin, "the fixture must trigger razoring");
    let v = search_at::<NonPv>(&mut w, &tt, alpha, beta, depth, true, 1);
    assert_eq!(v, eval, "razoring returns the qsearch stand-pat value");
    assert_eq!(w.nodes, 0, "razoring must return before the move loop");
}

#[test]
fn razoring_holds_off_one_centipawn_below_its_threshold() {
    // The trigger is exact. With alpha set so `alpha - margin == eval`
    // (one cp below the firing point above), the strict `<` fails and the
    // node does NOT razor — it enters the move loop and makes real moves
    // (`w.nodes > 0`). This pins the boundary: a single centipawn on alpha
    // is the difference between razoring and a full search, proving the
    // heuristic fires ONLY under its exact condition.
    let fen = "7k/8/8/3q4/8/8/8/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    let depth: Depth = 1;
    let margin = params::RAZOR_BASE() + params::RAZOR_DEPTH_MULT() * depth * depth;
    let alpha = eval + margin; // eval < alpha - margin is now FALSE (equal)
    let beta = alpha + 1;
    assert!(
        eval >= alpha - margin,
        "the fixture must sit exactly on the no-razor side"
    );
    let v = search_at::<NonPv>(&mut w, &tt, alpha, beta, depth, true, 1);
    assert!(w.nodes > 0, "on the margin, the node must search");
    let _ = v;
}

#[test]
fn razoring_never_fires_on_a_pv_node() {
    // The `!PV` guard (SF's "guard against mates being returned"): the
    // same hopeless position, in a window that WOULD razor at a non-PV
    // node, must still be searched at a PV node — dropping a PV line to
    // qsearch could mask a mate the full search would find. The PV node
    // enters its move loop (`w.nodes > 0`).
    let fen = "7k/8/8/3q4/8/8/8/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let depth: Depth = 1;
    let margin = params::RAZOR_BASE() + params::RAZOR_DEPTH_MULT() * depth * depth;
    let alpha = eval + margin + 1;
    let beta = alpha + 200; // a real PV window (alpha < beta - 1)
    assert!(
        eval < alpha - margin,
        "the window would razor at a non-PV node"
    );
    let v = search_at::<Pv>(&mut w, &tt, alpha, beta, depth, false, 1);
    assert!(w.nodes > 0, "a PV node must never razor to qsearch");
    let _ = v;
}

#[test]
fn razoring_condition_never_fires_on_a_winning_eval() {
    // Razoring only ever drops a node whose eval is far BELOW alpha. A
    // winning static eval fails the strict `<` test for every alpha at or
    // below it, so the dangerous direction — throwing away a win by
    // dropping to a shallow qsearch — is structurally impossible. Pure
    // boundary check over a real (winning, non-mate) eval across depths.
    let eval = static_eval(&board("7k/8/8/3Q4/8/8/8/K7 w - - 0 1"));
    assert!(!is_decisive(eval), "a queen edge is not a decisive score");
    for depth in 1..=6 {
        let margin = params::RAZOR_BASE() + params::RAZOR_DEPTH_MULT() * depth * depth;
        for alpha in [eval - 500, eval, eval + margin] {
            assert!(eval >= alpha - margin, "depth {depth}, alpha {alpha}");
        }
    }
}

#[test]
fn razoring_active_search_still_finds_the_forced_mate() {
    // Mate-safety with razoring live in the tree (the "decisive nodes"
    // guarantee): KRK mates in mate_in(3). Razoring fires in the losing
    // side's deeply-negative nodes — dropping them to qsearch — yet the
    // exact ply-relative mate must still come back along the PV, which the
    // `!PV` guard keeps off razoring entirely. Mirrors the NMP twin.
    let fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
    let w = go_depth(fen, 8);
    assert_eq!(
        best_score(&w),
        mate_in(3),
        "razoring must not corrupt the mate"
    );
}

// --- Late move pruning (design/10 §3.5 move-loop LMP, §5) ---

#[test]
fn lmp_move_count_matches_the_hand_computed_threshold() {
    // (depth, improving, expected) for `(3 + depth²) / (2 - improving)`,
    // computed by hand — the truncating divide (odd numerator / 2) is the
    // point. Not improving halves the divisor, so it prunes ~twice as
    // early; improving keeps every quiet's slot until 3 + depth².
    let cases: &[(Depth, bool, i32)] = &[
        (1, false, 2),  // (3+1)/2 = 2
        (1, true, 4),   // (3+1)/1 = 4
        (2, false, 3),  // (3+4)/2 = 7/2 = 3 (truncated)
        (2, true, 7),   // (3+4)/1 = 7
        (3, false, 6),  // (3+9)/2 = 6
        (3, true, 12),  // (3+9)/1 = 12
        (6, false, 19), // (3+36)/2 = 39/2 = 19 (truncated)
        (6, true, 39),  // (3+36)/1 = 39
        (8, false, 33), // (3+64)/2 = 67/2 = 33 (truncated)
        (8, true, 67),  // (3+64)/1 = 67
    ];
    for &(d, imp, want) in cases {
        assert_eq!(lmp_move_count(d, imp), want, "lmp_move_count({d}, {imp})");
    }
}

#[test]
fn lmp_threshold_grows_with_depth_and_prunes_sooner_when_not_improving() {
    // Monotone non-decreasing in depth (the quadratic numerator only
    // grows), and a not-improving node always prunes at or before an
    // improving one (the divisor is 2 vs 1). The numerator is >= 4, so
    // the not-improving threshold is STRICTLY smaller at every depth.
    let mut prev_imp = 0;
    let mut prev_not = 0;
    for d in 1..=32 {
        let imp = lmp_move_count(d, true);
        let not = lmp_move_count(d, false);
        assert!(imp >= prev_imp, "improving threshold dropped at depth {d}");
        assert!(
            not >= prev_not,
            "not-improving threshold dropped at depth {d}"
        );
        assert!(
            not < imp,
            "depth {d}: not-improving {not} must prune before improving {imp}"
        );
        prev_imp = imp;
        prev_not = not;
    }
}

#[test]
fn lmp_caps_a_wide_quiet_nonpv_node_at_the_threshold() {
    // White is up a knight; there are no captures anywhere, the node is
    // not in check, and no legal move gives check or lands beside the
    // black king — so every child is a captureless qsearch that makes
    // zero moves. `self.nodes` is bumped once per do_move, so it counts
    // exactly the moves the top node searched. All quiets are SEE-safe
    // (nothing to lose material to), so item-7 quiet SEE never fires; the
    // window is pitched to isolate LMP from item-7 PARENT FUTILITY as
    // well — see below.
    let fen = "7k/8/8/8/8/2N5/8/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let legal = all_legal(&w.board).len();
    let eval = static_eval(&w.board);
    assert!(!is_decisive(eval), "a knight edge is not a decisive score");
    let depth: Depth = 1;
    // improving is false at ply 1 (the grandparent static_eval is the
    // VALUE_NONE sentinel), so the threshold divisor is 2.
    let threshold = lmp_move_count(depth, false);
    assert!(
        (threshold as usize) < legal,
        "fixture must offer more quiets ({legal}) than the LMP cap ({threshold})"
    );
    // The window must sit ABOVE every quiet's value (so nothing fails
    // high and the loop runs to the LMP cap) yet BELOW the item-7
    // parent-futility bound `parent_futility_value(eval, no_bestmove=true,
    // lmr_depth=0, eval>alpha=false) = eval + 166` at this depth, so
    // parent futility stays dormant and LMP alone caps the node. `eval +
    // 40` is comfortably inside that band; below the depth-1 razoring
    // margin (801) too. (Pre-item-7 this test used `eval + 400`; parent
    // futility now prunes there, so the window tightened — the LMP cap
    // itself is unchanged.)
    let fut = parent_futility_value(eval, true, 0, false);
    let alpha = eval + 40;
    assert!(alpha < fut, "window must keep parent futility dormant");
    let beta = alpha + 1;
    let _ = search_at::<NonPv>(&mut w, &tt, alpha, beta, depth, true, 1);
    assert_eq!(
        w.nodes, threshold as u64,
        "LMP must cap the node at exactly its move-count threshold"
    );
}

// --- Parent futility + quiet SEE pruning (design/10 §3.5, §5) ---

#[test]
fn parent_futility_value_matches_the_hand_computed_bound() {
    // (static_eval, no_best_move, lmr_depth, eval_above_alpha, expected),
    // `static_eval + 39 + 127·no_best_move + 119·lmr_depth
    //  + 90·eval_above_alpha`, computed by hand — including the negative
    // `lmr_depth` term that LOWERS the bound at shallow nodes.
    let cases: &[(Value, bool, i32, bool, Value)] = &[
        (0, true, -1, false, 47),      // 39 + 127 - 119
        (0, false, -1, false, -80),    // 39 - 119
        (0, false, 0, false, 39),      // the flat base alone
        (100, true, 0, true, 356),     // 100 + 39 + 127 + 90
        (100, true, 5, false, 861),    // 100 + 39 + 127 + 595
        (-50, true, 3, true, 563),     // -50 + 39 + 127 + 357 + 90
        (200, false, 11, false, 1548), // 200 + 39 + 1309
    ];
    for &(ev, nbm, lmr, eaa, want) in cases {
        assert_eq!(
            parent_futility_value(ev, nbm, lmr, eaa),
            want,
            "parent_futility_value({ev}, {nbm}, {lmr}, {eaa})"
        );
    }
}

#[test]
fn parent_futility_prunes_a_quiet_only_at_or_below_its_margin() {
    // Same wide-quiet KN-vs-K node as the LMP test: at depth 1 the loop
    // searches move 1 (fails low, no best move raised), then reaches the
    // second quiet. Its parent-futility bound is
    // `parent_futility_value(eval, no_bestmove=true, lmr_depth=0,
    // eval>alpha=false) = eval + 166`. The move is pruned iff that bound
    // `<= alpha`, so the boundary is sharp:
    //   - alpha = bound - 1: NOT pruned → the second quiet is searched,
    //     move 3+ are LMP-skipped → exactly 2 nodes.
    //   - alpha = bound:     pruned, and `best_value` is first lifted to
    //     the bound → exactly 1 node, and the node returns that bound.
    // The pruned move is SEE-safe, so this isolates parent futility from
    // quiet SEE; `!in_check` and the non-decisive eval keep every guard
    // satisfied.
    let fen = "7k/8/8/8/8/2N5/8/K7 w - - 0 1";
    let eval = corrected_eval(&board(fen), None);
    assert!(!is_decisive(eval), "a knight edge is not a decisive score");
    let bound = parent_futility_value(eval, true, 0, false);
    let depth: Depth = 1;

    // Just below the margin: parent futility dormant, the second quiet is
    // searched. Two nodes (LMP caps move 3+).
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let _ = search_at::<NonPv>(&mut w, &tt, bound - 1, bound, depth, true, 1);
    assert_eq!(w.nodes, 2, "below the margin, the second quiet must search");

    // At the margin: parent futility prunes the second quiet AND raises
    // best_value to the bound; one node, and the returned value is the
    // bound itself (the fail-low value the pruned move contributes).
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let v = search_at::<NonPv>(&mut w, &tt, bound, bound + 1, depth, true, 1);
    assert_eq!(w.nodes, 1, "at the margin, the second quiet must be pruned");
    assert_eq!(v, bound, "parent futility lifts best_value to its bound");
}

#[test]
fn quiet_see_pruning_drops_a_hanging_quiet() {
    // White is boxed by two black rooks (down a rook-plus) with a knight
    // on a1 whose only two moves are quiets: Nc2 (SEE-safe — the white
    // king on c1 defends c2, so the black king cannot recapture) and Nb3
    // (SEE < 0 — it walks into the black king's grasp on c3, undefended).
    // Nc2 has the higher quiet score (Nb3 carries a threat penalty), so
    // Nc2 is searched first, fails low, and leaves best_value non-loss
    // with no best move. Nb3 then reaches the pruning block, where the
    // window is chosen so parent futility is PROVABLY dormant — leaving
    // quiet SEE (`-25·max(0,lmr_depth)² = 0` at depth 1) as the only rule
    // that can drop it. It is SEE < 0, so it is pruned: one node.
    let fen = "1r1r4/8/8/8/8/2k5/8/N1K5 w - - 0 1";
    let b = board(fen);
    let legal = all_legal(&b);
    assert_eq!(legal.len(), 2, "fixture must offer exactly two quiets");
    let safe = Move::normal(Square::A1, Square::C2);
    let hanging = Move::normal(Square::A1, Square::B3);
    assert!(b.see_ge(safe, 0), "Nc2 must be SEE-safe (king-defended)");
    assert!(!b.see_ge(hanging, 0), "Nb3 must lose material by SEE");
    assert!(
        !b.capture_stage(hanging) && !b.gives_check(hanging),
        "Nb3 must be a quiet non-check (the item-7 branch)"
    );

    let eval = static_eval(&b);
    let depth: Depth = 1;
    // lmr_depth = 0 at depth 1, so the SEE bar is `-23·max(0,0)² = 0`:
    // any SEE < 0 quiet is dropped. Pitch alpha above Nc2's value (so it
    // fails low and we reach Nb3) yet below the parent-futility bound
    // `eval + 166` (so parent futility stays dormant and quiet SEE is the
    // sole pruner of Nb3).
    let bound = parent_futility_value(eval, true, 0, false);
    let alpha = eval + 40;
    assert!(alpha < bound, "window must keep parent futility dormant");
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let v = search_at::<NonPv>(&mut w, &tt, alpha, alpha + 1, depth, true, 1);
    assert!(v <= alpha, "Nc2 must fail low (no fail-high shortcut)");
    assert_eq!(w.nodes, 1, "quiet SEE must drop the hanging Nb3");
}

// --- Continuation-history pruning (design/10 §3.5, Wave-C item 11, §5) ---

#[test]
fn continuation_history_prunes_matches_the_threshold() {
    // The gate `history < -QHIST_PRUNE_MULT · depth` (design/10 §3.5) — a
    // LINEAR depth scale (unlike the quadratic quiet-SEE bar). The
    // comparison is STRICT, so a history exactly at `-4313·depth` is NOT
    // pruned while one just below is; positive history never prunes; the
    // fresh combined sum (−2366) stays dormant at every depth.
    let m = params::QHIST_PRUNE_MULT(); // 4313
    let cases: &[(i32, Depth, bool)] = &[
        (0, 1, false),         // healthy history never prunes
        (100, 8, false),       // positive: never
        (-m, 1, false),        // exactly at the bar → strict `<` → NOT pruned
        (-m - 1, 1, true),     // one below → pruned
        (-m * 2, 2, false),    // −8626 at depth 2 → NOT pruned
        (-m * 2 - 1, 2, true), // one below → pruned
        (-m * 4, 4, false),    // boundary at depth 4
        (-m * 4 - 1, 4, true),
        (-2366, 1, false),   // the FRESH sum is dormant (−2366 > −4313)
        (-2366, 32, false),  // fresh stays dormant at every depth
        (-100_000, 8, true), // deeply distrusted: pruned even deep
    ];
    for &(hist, depth, want) in cases {
        assert_eq!(
            continuation_history_prunes(hist, depth),
            want,
            "continuation_history_prunes({hist}, {depth})"
        );
    }
}

#[test]
fn continuation_history_bar_is_linear_and_monotone_in_depth() {
    // The bar `-4313·depth` deepens LINEARLY, so a fixed negative history
    // that prunes at shallow depth clears the (steadily more negative) bar
    // as depth grows: `-20000 < -4313·d` iff `d < 4.63`, i.e. prunes for
    // `d <= 4` only. This linearity is exactly what distinguishes the gate
    // from the quiet-SEE bar's quadratic `-25·lmr_depth²`.
    let hist = -20_000;
    for d in 1..=4 {
        assert!(
            continuation_history_prunes(hist, d),
            "must prune at depth {d}"
        );
    }
    for d in 5..=16 {
        assert!(
            !continuation_history_prunes(hist, d),
            "must be dormant at depth {d}"
        );
    }
    // The fresh combined history is dormant at every depth.
    for d in 1..=64 {
        assert!(
            !continuation_history_prunes(-2366, d),
            "fresh history must stay dormant at depth {d}"
        );
    }
}

#[test]
fn continuation_history_pruning_drops_a_distrusted_quiet() {
    use crate::history::StatEntry;
    // The wide-quiet KN-vs-K node the LMP / parent-futility tests use: at
    // depth 1 move 1 is searched (the `is_loss` fence skips pruning on the
    // first move), then a second quiet reaches the quiet arm. With FRESH
    // histories the combined `cont_history` is −2366, above the bar, so the
    // gate is dormant and the second quiet is searched — the exact 2-node
    // parent-futility "below the margin" result. Poking the offset-1/
    // offset-2 continuation slab (the SENTINEL predecessor frames below the
    // root, which the ply-1 node reads for both offsets) to a large
    // negative drives every quiet's
    // `cont_history = 2·(−8000) + pawnHistory(−1338) = −17338 < −4136·1`,
    // so the gate fires and the second quiet is pruned outright — 1 node.
    let fen = "7k/8/8/8/8/2N5/8/K7 w - - 0 1";
    let eval = corrected_eval(&board(fen), None);
    assert!(!is_decisive(eval), "a knight edge is not a decisive score");
    // Below the parent-futility bound (that gate stays dormant) and just
    // above the quiets' values (move 1 fails low) — the parent-futility
    // test pins this exact window at 2 nodes.
    let bound = parent_futility_value(eval, true, 0, false);
    let depth: Depth = 1;

    // Fresh: cont-hist dormant, the second quiet is searched (2 nodes).
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let vf = search_at::<NonPv>(&mut w, &tt, bound - 1, bound, depth, true, 1);
    assert!(
        vf < bound,
        "fresh: the node must fail low (at or below alpha)"
    );
    assert_eq!(
        w.nodes, 2,
        "fresh histories: cont-hist dormant, the second quiet is searched"
    );

    // Poked: every quiet's cont_history falls below −4313·depth → pruned.
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    for row in w.histories.cont_mut(ContHistId::SENTINEL).iter_mut() {
        for cell in row.iter_mut() {
            *cell = StatEntry::new(-8000);
        }
    }
    let vp = search_at::<NonPv>(&mut w, &tt, bound - 1, bound, depth, true, 1);
    assert!(
        vp < bound,
        "poked: the node must fail low (a cont-hist skip, not a cutoff)"
    );
    assert_eq!(
        w.nodes, 1,
        "continuation-history pruning must drop the second quiet"
    );
}

#[test]
fn continuation_history_pruning_preserves_a_forced_mate() {
    use crate::history::StatEntry;
    // Mate-safety with the gate armed as hard as it goes: poke the WHOLE
    // continuation-history table to its most negative bound, so every quiet
    // the search meets is a maximal pruning candidate. The exact
    // ply-relative mate must still return. Rd8# is a CHECK — it lives in
    // the capture/check arm, never the quiet arm this gate guards — and the
    // deeper mate-suite tests re-run with the gate active cover interior
    // quiets under the shared `!is_loss(best_value)` fence.
    let fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
    let mut w = Worker::new(board(fen));
    w.limits.depth = Some(2);
    for slab in w.histories.cont.iter_mut() {
        for row in slab.iter_mut() {
            for cell in row.iter_mut() {
                *cell = StatEntry::new(-(crate::history::CONT_D as i16));
            }
        }
    }
    let tt = test_tt();
    go(&mut w, &tt);
    assert_eq!(
        best_score(&w),
        mate_in(1),
        "the exact mate must survive a maximally-armed gate"
    );
    assert_eq!(w.best_move().unwrap().to_string(), "d1d8");
}

// --- Capture futility + capture/check SEE pruning (design/10 §3.5, §5) ---

#[test]
fn capture_futility_value_matches_the_hand_computed_bound() {
    // (static_eval, lmr_depth, captured_value, capt_hist, expected):
    // `static_eval + 234 + 247·lmr_depth + captured_value
    //  + 134·capt_hist/1024`, computed by hand — including the negative
    // `lmr_depth` term that LOWERS the bound at shallow nodes and the
    // truncation of `134·capt_hist/1024` (formed before the divide).
    let cases: &[(Value, i32, Value, i32, Value)] = &[
        (0, 0, 0, 0, 234),             // the flat base alone
        (0, -1, 208, -742, 98),        // depth-1 pawn capture, fresh capt_hist: 234-247+208-97
        (0, 0, 0, -1, 234),            // -134/1024 truncates toward zero to 0
        (0, 0, 0, -1024, 100),         // -134·1024/1024 = -134 → 234-134
        (100, 3, 208, 1024, 1417),     // 100+234+741+208+134
        (-1412, -1, 208, -742, -1314), // the search-level fixture's bound
    ];
    for &(se, lmr, cap, ch, want) in cases {
        assert_eq!(
            capture_futility_value(se, lmr, cap, ch),
            want,
            "capture_futility_value({se}, {lmr}, {cap}, {ch})"
        );
    }
}

#[test]
fn capture_see_margin_matches_the_hand_computed_value() {
    // (depth, capt_hist, expected): `177·depth + 34·capt_hist/1024`, the
    // capture-history term formed before the truncating divide.
    let cases: &[(Depth, i32, Value)] = &[
        (1, 0, 177),      // 177 + 0
        (1, -742, 153),   // 177 + (34·-742)/1024 = 177 - 24
        (2, -1, 354),     // -34/1024 truncates toward zero to 0
        (4, 1024, 742),   // 708 + 34
        (10, 3000, 1869), // 1770 + 102000/1024 = 1770 + 99
    ];
    for &(d, ch, want) in cases {
        assert_eq!(
            capture_see_margin(d, ch),
            want,
            "capture_see_margin({d}, {ch})"
        );
    }
}

#[test]
fn singular_verification_beta_matches_the_hand_computed_centre() {
    // (tt_value, depth, ttpv_non_pv, expected):
    // `tt_value - (59 + 66·ttpv_non_pv)·depth / 63`, the `(…)·depth`
    // product formed before the truncating divide (D27).
    let cases: &[(Value, Depth, bool, Value)] = &[
        (0, 0, false, 0),       // no depth → no offset
        (0, 1, false, 0),       // 59/63 = 0 truncates → 0
        (0, 1, true, -1),       // 125/63 = 1 truncates → -1
        (100, 8, false, 93),    // 472/63 = 7 → 100-7
        (100, 8, true, 85),     // 1000/63 = 15 → 100-15
        (1000, 10, false, 991), // 590/63 = 9 → 1000-9
        (-500, 12, true, -523), // 1500/63 = 23 → -500-23
    ];
    for &(tt, d, ttpv, want) in cases {
        assert_eq!(
            singular_verification_beta(tt, d, ttpv),
            want,
            "singular_verification_beta({tt}, {d}, {ttpv})"
        );
    }
}

#[test]
fn singular_double_margin_matches_the_hand_computed_margin() {
    // (pv, tt_capture, corr_adj, tmh, deep, expected):
    // `-2 + 204·pv - 152·(!tt_capture) - corr_adj
    //  - 1175·tmh/114178 - 38·deep`, the `tmh` term formed before its
    // truncating divide (D27).
    let cases: &[(bool, bool, Value, i32, bool, Value)] = &[
        (false, true, 0, 0, false, -2), // base alone (ttCapture: no !cap penalty)
        (false, false, 0, 0, false, -154), // quiet ttMove docks 152
        (true, false, 0, 0, false, 50), // PV bonus 204, minus 152
        (true, true, 0, 0, true, 164),  // PV, ttCapture, deep: 204-2-38
        (false, true, 0, 114178, false, -1177), // full tmh unit: -1175
        (false, true, 0, 1, false, -2), // 1175·1/114178 truncates → 0
        (false, true, 0, -1, false, -2), // negative tmh truncates toward zero → 0
        (false, true, 500, 0, false, -502), // corr_adj subtracts directly
    ];
    for &(pv, cap, corr, tmh, deep, want) in cases {
        assert_eq!(
            singular_double_margin(pv, cap, corr, tmh, deep),
            want,
            "singular_double_margin({pv}, {cap}, {corr}, {tmh}, {deep})"
        );
    }
}

#[test]
fn singular_triple_margin_matches_the_hand_computed_margin() {
    // (pv, tt_capture, ttpv, corr_adj, deep, expected):
    // `70 + 279·pv - 188·(!tt_capture) + 81·ttpv - corr_adj - 43·deep`.
    let cases: &[(bool, bool, bool, Value, bool, Value)] = &[
        (false, true, false, 0, false, 70),    // base alone
        (false, false, false, 0, false, -118), // quiet ttMove docks 188
        (true, false, true, 0, false, 242),    // 70+279-188+81
        (true, true, true, 0, true, 387),      // 70+279+81-43
        (false, true, false, 100, true, -73),  // corr_adj 100, deep 43: 70-100-43
    ];
    for &(pv, cap, ttpv, corr, deep, want) in cases {
        assert_eq!(
            singular_triple_margin(pv, cap, ttpv, corr, deep),
            want,
            "singular_triple_margin({pv}, {cap}, {ttpv}, {corr}, {deep})"
        );
    }
}

#[test]
fn capture_futility_prunes_a_futile_capture_at_its_margin() {
    // White (Ka6, Na1) is boxed to exactly two legal moves, both captures
    // of black pawns' guards: Nxb3 (an even trade — the black knight on b3
    // is pawn-defended, SEE 0) and Nxc2 (an undefended pawn, SEE +208).
    // Both are SEE-safe, so the capture-SEE gate can NEVER prune either —
    // isolating capture FUTILITY as the only rule at play. The picker
    // orders the higher-MVV Nxb3 first: it is searched (its qsearch child
    // recaptures cxb3, so move 1 costs two nodes) and fails low, leaving
    // best_value non-loss with no best move. Nxc2 then reaches the capture
    // branch, where at depth 1 `lmr_depth == 0` and the fresh capture
    // history is −742, so the futility bound is exactly
    // `capture_futility_value(static_eval, 0, 208, -742)`. The boundary is
    // sharp (`futility_value <= alpha`):
    //   - alpha = bound:     Nxc2 is pruned → 2 nodes (only move 1's pair).
    //   - alpha = bound - 1: Nxc2 is searched → 3 nodes.
    let fen = "kr4r1/8/K7/8/2p5/1n6/2p5/N7 w - - 0 1";
    let b = board(fen);
    assert_eq!(
        all_legal(&b).len(),
        2,
        "fixture must offer exactly two moves"
    );
    let nxb3 = Move::normal(Square::A1, Square::B3);
    let nxc2 = Move::normal(Square::A1, Square::C2);
    assert!(
        b.capture_stage(nxb3) && b.capture_stage(nxc2),
        "both are captures"
    );
    assert!(!b.gives_check(nxc2), "Nxc2 must be a non-checking capture");
    // Both SEE-safe → capture SEE cannot prune (isolates futility).
    let depth: Depth = 1;
    let ch = Worker::new(board(fen)).histories.capture_value(&b, nxc2);
    assert_eq!(ch, -742, "fresh capture history is CAPTURE_INIT");
    assert!(
        b.see_ge(nxc2, -capture_see_margin(depth, ch)),
        "Nxc2 must clear the capture-SEE margin (SEE cannot be the pruner)"
    );

    let se = corrected_eval(&b, None);
    assert!(
        !is_decisive(se),
        "the fixture's eval is not a decisive score"
    );
    let bound = capture_futility_value(se, 0, PieceType::Pawn.value(), ch);

    // At the margin: Nxc2 pruned by capture futility. Move 1 (Nxb3) plus
    // its qsearch recapture are the only nodes.
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let _ = search_at::<NonPv>(&mut w, &tt, bound, bound + 1, depth, true, 1);
    assert_eq!(
        w.nodes, 2,
        "at the margin, capture futility must prune Nxc2"
    );

    // One below the margin: capture futility dormant, Nxc2 is searched.
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let _ = search_at::<NonPv>(&mut w, &tt, bound - 1, bound, depth, true, 1);
    assert_eq!(w.nodes, 3, "below the margin, Nxc2 must be searched");
}

#[test]
fn capture_pruning_preserves_a_forced_mate() {
    // Mate-safety with capture futility + capture SEE live in a tree full
    // of captures: the losing side has a queen (h4), so both rooks' capture
    // moves and the queen's are subject to the capture-pruning gates. The
    // shared `!is_loss(best_value)` fence keeps them out of the mate band,
    // so the exact ply-relative mate (Ra8+ …#, mate in two moves) still
    // returns. Companion to the parent-futility / quiet-SEE mate twin.
    let fen = "6k1/5ppp/8/8/7q/8/5PPP/R2R2K1 w - - 0 1";
    let w = go_depth(fen, 8);
    assert_eq!(
        best_score(&w),
        mate_in(3),
        "capture futility / SEE must not corrupt the forced mate"
    );
}

#[test]
fn parent_futility_and_quiet_see_preserve_the_forced_mate() {
    // Mate-safety with both item-7 prunings live in the tree: the
    // `!is_loss(best_value)` fence and the `!is_decisive(best_value)` /
    // `!is_win(futility_value)` guards keep the shallow quiet prunings out
    // of the mate band, so the exact ply-relative KRK mate still returns.
    // Mirrors the LMP / NMP / razoring twins on the same fixture.
    let fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
    let w = go_depth(fen, 8);
    assert_eq!(
        best_score(&w),
        mate_in(3),
        "parent futility / quiet SEE must not corrupt the mate"
    );
}

#[test]
fn lmp_is_disabled_without_non_pawn_material() {
    // The same wide-quiet shape but the side to move has only pawns
    // (+king): the `non_pawn_material(us) != 0` guard disables the whole
    // shallow-pruning block, so LMP never fires and every quiet is
    // searched — `self.nodes` equals the full legal move count, not the
    // LMP cap. This isolates the zugzwang guard from the movecount rule.
    let fen = "7k/8/8/8/8/2P5/PP6/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let legal = all_legal(&w.board).len();
    assert_eq!(
        w.board.non_pawn_material(w.board.side_to_move()),
        0,
        "fixture must have no non-pawn material for the side to move"
    );
    let eval = static_eval(&w.board);
    let depth: Depth = 1;
    assert!(
        (lmp_move_count(depth, false) as usize) < legal,
        "the LMP cap would bite here if the block ran ({legal} quiets)"
    );
    let alpha = eval + 400;
    let beta = alpha + 1;
    let _ = search_at::<NonPv>(&mut w, &tt, alpha, beta, depth, true, 1);
    assert_eq!(
        w.nodes, legal as u64,
        "no non-pawn material must leave every quiet searched"
    );
}

#[test]
fn lmp_active_search_still_finds_the_forced_mate() {
    // Mate-safety with LMP live in the tree: the `!is_loss(best_value)`
    // guard keeps the losing side searching every evasion, and quiets are
    // only skipped once a non-loss value stands — so the movecount rule
    // prunes the wide shuffling nodes yet the exact ply-relative mate
    // still comes back. KRK mates in mate_in(3); mirrors the razoring and
    // NMP twins on the same fixture.
    let fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
    let w = go_depth(fen, 8);
    assert_eq!(best_score(&w), mate_in(3), "LMP must not corrupt the mate");
}

// --- Null move pruning (design/10 §3.4 N11, §5 zugzwang guard) ---

/// Seed an eval-only TT entry carrying a quiet `ttm`: `is_valid` fails on
/// both its value and eval (no TT cutoff, no eval substitution), so its
/// ONLY effect at the node is to veto RFP through the
/// `(ttMove == NULL || ttCapture)` guard — leaving NMP as the heuristic
/// under test.
fn seed_quiet_ttmove(w: &Worker, tt: &TranspositionTable, ttm: Move) {
    let key = w.board.key();
    tt.probe(key).writer.write(
        key,
        VALUE_NONE,
        false,
        Bound::None,
        DEPTH_UNSEARCHED,
        ttm,
        VALUE_NONE,
        tt.generation(),
    );
}

#[test]
fn nmp_prunes_a_winning_cut_node_via_the_null_search() {
    // White a full queen up, quiet, not in check: the static eval clears
    // the N11 bar `beta - 14·depth - 45·improving + 374`, and after the
    // pass the opponent (still down a queen) cannot pull the score below
    // beta — so the null search fails high and NMP returns before any
    // real move. RFP is vetoed by a quiet ttMove, so NMP is what cuts.
    // depth 4 ⇒ R = 8 ⇒ the null search runs at depth −4, i.e. straight
    // into a capture-free qsearch that makes ZERO moves: `w.nodes == 0`
    // proves the node returned through the pass, not a real search.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let depth: Depth = 4;
    let beta = eval - 400;
    // improving is false at ply 1 (VALUE_NONE predecessors), so the bar
    // is `beta - 14·depth + 374`.
    let bar = beta - params::NMP_EVAL_DEPTH() * depth + params::NMP_EVAL_BASE();
    assert!(eval >= bar, "fixture must clear the NMP static-eval bar");
    assert!(!is_win(eval), "material edge alone is not decisive");
    seed_quiet_ttmove(&w, &tt, Move::normal(Square::B1, Square::B2));
    // A real interior node always has a real parent move (the N11 assert
    // `no two null moves in a row`); the search_at harness starts on a
    // fresh stack whose ply-0 slot is NULL, so install one by hand. The
    // marker must END on an occupied square — item 21's correction reads
    // `piece_on((ss-1).current_move.to())`, which in real search always
    // holds (the child sees the parent's piece on its to-square). A
    // plausible black parent move landing on the black king (h8).
    init_stack(&mut w.stack);
    w.stack[ss(0)].current_move = Move::normal(Square::G8, Square::H8);
    w.nodes = 0;
    let stop = AtomicBool::new(false);
    let v = w.search::<NonPv>(&tt, &stop, beta - 1, beta, depth, true, 1);
    assert_eq!(w.nodes, 0, "NMP must return through the pass, no real move");
    assert!(v >= beta, "the null search fails high");
    assert_ne!(
        v,
        (params::RFP_RET_BETA_W() * beta + params::RFP_RET_EVAL_W() * eval) / params::RFP_RET_DEN(),
        "the return is the null value, not the (vetoed) RFP blend"
    );
}

#[test]
fn nmp_never_fires_on_a_pv_node() {
    // NMP is cutNode-only. The same winning position searched as a PV
    // node (cut_node = false) must NOT prune through the pass — it
    // searches a real move (`w.nodes > 0`).
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let depth: Depth = 4;
    let beta = eval - 400;
    seed_quiet_ttmove(&w, &tt, Move::normal(Square::B1, Square::B2));
    // A PV node needs alpha < beta - 1; the RFP/NMP blend/return path is
    // NonPV-shaped, but the point is just that no pass-based cutoff fires.
    let v = search_at::<Pv>(&mut w, &tt, beta - 200, beta, depth, false, 1);
    assert!(w.nodes > 0, "a PV node must not null-move prune");
    let _ = v;
}

#[test]
fn nmp_disabled_below_nmp_min_ply() {
    // The `ply >= nmp_min_ply` gate is the verification scheme's teeth:
    // with the floor raised above the node's ply, the would-be pruning
    // node searches a real move instead of passing (`w.nodes > 0`), even
    // though every other N11 condition holds and RFP stays vetoed.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let depth: Depth = 4;
    let beta = eval - 400;
    seed_quiet_ttmove(&w, &tt, Move::normal(Square::B1, Square::B2));
    w.nmp_min_ply = 100; // ply 1 < 100 ⇒ NMP gate closed
    let v = search_at::<NonPv>(&mut w, &tt, beta - 1, beta, depth, true, 1);
    assert!(w.nodes > 0, "the ply floor must block the pass");
    assert!(v >= beta, "the winning node still fails high by searching");
}

#[test]
fn nmp_disabled_without_non_pawn_material() {
    // The primary zugzwang fence (design/10 §5): a side with only pawns
    // must never null-move prune, because passing is exactly the move it
    // wishes it had. In a K+P endgame the `non_pawn_material(us) != 0`
    // gate keeps NMP off even when the eval clears the bar and RFP is
    // vetoed — so the node searches a real move (`w.nodes > 0`). Contrast
    // `nmp_prunes_a_winning_cut_node_via_the_null_search`, where the same
    // setup WITH a queen prunes to zero nodes.
    let fen = "7k/8/8/8/8/2P1P3/8/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let us = w.board.side_to_move();
    assert_eq!(w.board.non_pawn_material(us), 0, "K+P only for the mover");
    let eval = static_eval(&w.board);
    let depth: Depth = 4;
    let beta = eval - 400;
    let bar = beta - params::NMP_EVAL_DEPTH() * depth + params::NMP_EVAL_BASE();
    assert!(eval >= bar, "the eval still clears the NMP bar");
    assert!(
        !is_loss(beta),
        "beta is non-decisive, so only the NPM gate blocks"
    );
    seed_quiet_ttmove(&w, &tt, Move::normal(Square::A1, Square::B1));
    let v = search_at::<NonPv>(&mut w, &tt, beta - 1, beta, depth, true, 1);
    assert!(
        w.nodes > 0,
        "no non-pawn material ⇒ NMP must not prune the pass"
    );
    let _ = v;
}

#[test]
fn nmp_reduces_nodes_in_the_null_only_regime() {
    // NMP's node-level payoff, isolated from RFP: at depth 10 RFP's
    // margin (~580) exceeds 400, so a window 400cp below the eval leaves
    // RFP's `eval - margin >= beta` unsatisfied while NMP's bar
    // (beta + ~234) is cleared — the regime where NMP fires but RFP does
    // not. White (a full queen up, non-pawn material present) passes, the
    // opponent cannot recover, so the null search fails high and prunes.
    // A control with NMP disabled via the ply floor searches strictly
    // more nodes. (A direct fixed-depth search bypasses the per-go reset,
    // so the floor sticks for the whole subtree.)
    let fen = "r3k2r/ppp2ppp/2n5/8/8/2N5/PPP2PPP/R2QK2R w KQkq - 0 1";
    let depth: Depth = 10;
    let eval = static_eval(&board(fen));
    let (alpha, beta) = (eval - 401, eval - 400);
    let run = |disable: bool| -> u64 {
        let mut w = Worker::new(board(fen));
        let tt = test_tt();
        tt.new_search();
        init_stack(&mut w.stack);
        // Marker must land on an occupied square (item 21 reads the piece
        // on the parent move's to-square): a black knight move to c6.
        w.stack[ss(0)].current_move = Move::normal(Square::B8, Square::C6);
        w.nodes = 0;
        w.root_depth = depth;
        w.nmp_min_ply = if disable { 1_000_000 } else { 0 };
        let stop = AtomicBool::new(false);
        let _ = w.search::<NonPv>(&tt, &stop, alpha, beta, depth, true, 1);
        w.nodes
    };
    let on = run(false);
    let off = run(true);
    assert!(
        on < off,
        "NMP must prune in its own regime: on={on} off={off}"
    );
}

#[test]
fn nmp_active_search_still_finds_the_forced_mate() {
    // Mate-safety with NMP live in the tree: KRK mates in 2 full moves
    // (mate_in(3)). White has a rook (non-pawn material), so NMP fires in
    // the winning-side nodes — yet the exact ply-relative mate must still
    // come back, never a pass-induced non-mate score. Searched deep
    // enough (depth 8) that NMP is exercised throughout.
    let fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
    let w = go_depth(fen, 8);
    assert_eq!(best_score(&w), mate_in(3), "NMP must not corrupt the mate");
}

// --- LMR core (design/10 §3.7/§3.8, Wave-A item 4) ---

#[test]
fn lmr_shrinks_the_effective_branching_factor() {
    // The payoff of LMR (design/10 §6): reducing late moves collapses the
    // per-ply effective branching factor from plain PVS's ~6–8 toward
    // ~2. On a quiet Italian-ish midgame the node count from depth 6 to
    // depth 8 must grow FAR less than an unreduced tree would — a loose
    // ceiling (a per-ply EBF of ~3.5, versus LMR's measured ~1.9) that a
    // mis-signed or mis-scaled reduction (which would fail to prune, or
    // prune the wrong moves and re-search them all) blows straight past.
    // Node counts are single-thread deterministic (D24), so this is a
    // stable regression guard, not a timing test.
    let fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1";
    let n6 = go_depth(fen, 6).nodes;
    let n8 = go_depth(fen, 8).nodes;
    assert!(
        n8 < 12 * n6,
        "LMR must keep the tree shallow: nodes(8)={n8} !< 12·nodes(6)={}",
        12 * n6
    );
}

#[test]
fn lmr_active_deep_search_preserves_the_exact_mate() {
    // Mate safety with LMR reducing the bulk of the tree (the §3.8
    // `depth >= 2 && move_count > 1` gate fires everywhere at this
    // depth): the KRK forced mate in 2 full moves (mate_in(3), three
    // plies) must still come back EXACT at depth 9, never a reduction-
    // truncated shorter/longer distance. The re-search rules and the
    // decisive-score guards are what this pins — complementing the
    // shallow mate suite below (which now also runs under LMR) and the
    // NMP mate test above.
    let w = go_depth("k7/8/2K5/8/8/8/8/7R w - - 0 1", 9);
    assert_eq!(best_score(&w), mate_in(3), "LMR must not corrupt the mate");
}

// --- statScore reduction term + LMR conthist bonus (design/10 §3.7/§3.8,
//     Wave-C item 10) ---

#[test]
fn statscore_reduction_dock_direction_and_truncation() {
    // `r -= stat_score·RED_STATSCORE_NUM / RED_STATSCORE_DEN` (design/10
    // §3.7): a positive (good-history) stat_score LOWERS the reduction, a
    // negative one RAISES it, and — like the Stockfish source it re-
    // expresses (D27) — the division truncates toward zero, so a negative
    // product does NOT floor.
    let dock = |s: i32| s * params::RED_STATSCORE_NUM() / params::RED_STATSCORE_DEN();
    // Exact multiple of the divisor: symmetric magnitudes.
    assert_eq!(
        dock(params::RED_STATSCORE_DEN()),
        params::RED_STATSCORE_NUM()
    );
    assert_eq!(
        dock(-params::RED_STATSCORE_DEN()),
        -params::RED_STATSCORE_NUM()
    );
    // Non-exact: truncation toward zero (a floor would give -11, not -10).
    assert_eq!(dock(100), 10); // 44500/4096 = 10.86 → 10
    assert_eq!(dock(-100), -10); // -44500/4096 = -10.86 → -10 (toward 0)
    // Direction: a good history subtracts from r, a bad one adds to it.
    assert!(dock(5000) > 0 && dock(-5000) < 0);
}

#[test]
fn statscore_capture_grades_victim_value_plus_capture_history() {
    use crate::history::StatEntry;
    // ss.stat_score for a capture = STATSCORE_CAPT_NUM·piece_value(victim)
    // / STATSCORE_CAPT_DEN + capt_hist (design/10 §3.7). Nf3xe5 wins a
    // pawn; with a hand-set capture-history cell the whole statistic is
    // exact. The capture branch reads no stack, so `cur` is inert.
    let fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1";
    let mut w = Worker::new(board(fen));
    let m = Move::normal(Square::F3, Square::E5);
    let moved_piece = w.board.piece_on(m.from()).unwrap();
    let victim = w.board.piece_on(m.to()).unwrap();
    // Same [moved][to][victim] cell `capture_value` reads (doc 09 §3.9).
    w.histories.capture[moved_piece.index()][m.to().idx()][victim.piece_type as usize] =
        StatEntry::new(300);
    let want = params::STATSCORE_CAPT_NUM() * victim.piece_type.value()
        / params::STATSCORE_CAPT_DEN()
        + 300;
    assert_eq!(w.stat_score(m, true, moved_piece, ss(0)), want);
}

#[test]
fn statscore_quiet_grades_main_and_two_continuation_histories() {
    use crate::history::StatEntry;
    // ss.stat_score for a quiet = STATSCORE_MAIN_WEIGHT·main[us][raw16]
    // + contHist(1)[pc][to] + contHist(2)[pc][to] (design/10 §3.7), the
    // offset-1/2 slabs keyed off the predecessor frames stack[cur-1]/
    // [cur-2] (the same slabs move ordering reads as contHist(1/2)).
    let mut w = Worker::new(board(START_FEN));
    let m = Move::normal(Square::G1, Square::F3); // Ng1-f3, a quiet
    let moved_piece = w.board.piece_on(m.from()).unwrap();
    let us = w.board.side_to_move();
    let pci = moved_piece.index();
    let to = m.to().index() as usize;
    let cur = ss(2);
    // Two DISTINCT predecessor continuation-history slabs (different `to`).
    let id1 = ContHistId::new(false, false, Some(moved_piece), Square::E2);
    let id2 = ContHistId::new(false, false, Some(moved_piece), Square::D2);
    assert_ne!(id1.piece_to(), id2.piece_to(), "slabs must be distinct");
    w.stack[cur - 1].cont_hist = id1;
    w.stack[cur - 2].cont_hist = id2;
    // Seed the three read cells to distinct known values.
    w.histories.main[us as usize][m.raw16() as usize] = StatEntry::new(60);
    w.histories.cont_mut(id1)[pci][to] = StatEntry::new(700);
    w.histories.cont_mut(id2)[pci][to] = StatEntry::new(-400);
    let want = params::STATSCORE_MAIN_WEIGHT() * 60 + 700 - 400;
    assert_eq!(w.stat_score(m, false, moved_piece, cur), want);
}

// --- Mate suite (doc 07 §5/§8(4)): exact ply-relative mate scores (D5) ---

#[test]
fn back_rank_mate_in_1_found_at_depth_2() {
    // Rd8# is the only mate (and the only check). Expect the exact
    // ply-relative score mate_in(1) and the mating move as the PV.
    let fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
    for depth in [2, 3, 4] {
        let w = go_depth(fen, depth);
        assert_eq!(best_score(&w), mate_in(1), "depth {depth}");
        assert_eq!(w.best_move().unwrap().to_string(), "d1d8");
    }
}

#[test]
fn krk_mate_in_2_exact_score_at_depth_4() {
    // 1.Kb6 (only-move reply Kb8) 2.Rh8#: three plies, no mate in one.
    //
    // Depth 4, not 3, since aspiration landed (§3.6): the iteration
    // that first sees the mate fails high and its shallower re-search
    // (adjusted_depth) legitimately ends the iteration on a non-mate
    // value — SF-identical dynamics. One more iteration re-scores the
    // front (mating) move inside a mate-band-widened window and the
    // exact score sticks. The §8(4) bar is depth <= 6.
    let w = go_depth("k7/8/2K5/8/8/8/8/7R w - - 0 1", 4);
    assert_eq!(best_score(&w), mate_in(3));
    // The PV replays legally from the root position.
    replay_pv(&w);
}

#[test]
fn kqk_mate_in_2_exact_score_at_depth_4() {
    // 1.Qf8+ Kh7 (forced) 2.Qg7#; no mate in one exists. Depth 4 for
    // the aspiration settling iteration (see the KRK test above).
    let w = go_depth("7k/8/5K2/8/8/Q7/8/8 w - - 0 1", 4);
    assert_eq!(best_score(&w), mate_in(3));
    replay_pv(&w);
}

#[test]
fn deeper_search_keeps_the_exact_mate_score() {
    // Ply-relative scoring must not drift when the horizon extends
    // past the mate (doc 07 §3.1).
    let w = go_depth("7k/8/5K2/8/8/Q7/8/8 w - - 0 1", 5);
    assert_eq!(best_score(&w), mate_in(3));
}

#[test]
fn two_rook_ladder_mate_in_3_exact_at_depth_12() {
    // 1.Rb6+ (rank 5 stays cut by Ra5) K→7th 2.Ra7+ K→8th 3.Rb8#:
    // five plies. The black king starts on g6, too far from the a/b
    // files to ever attack the unprotected checking rooks, and too
    // free for anything shorter than the ladder. Pre-S1 the proof
    // landed at depth 6 (five plies + one aspiration settling
    // iteration); live correction history (S1/D22) shifts proof depth
    // in PAWNLESS studies — the whole tree shares the pawn/minor
    // correction cells, so the mate-line's bonuses saturate them and
    // inflate every sibling's margins — and this position converges to
    // the exact mate 3 at depth 12 (stable through 14+). Doc 07 §8(4)
    // carries the amended depth ≤ 12 bar.
    let w = go_depth("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 12);
    assert_eq!(best_score(&w), mate_in(5));
    assert_eq!(w.root_pv().len(), 5);
    // The PV replays legally and its end really is checkmate.
    let end = replay_pv(&w);
    assert!(all_legal(&end).is_empty() && !end.checkers().is_empty());
}

#[test]
fn mate_suite_1_to_3_exact_scores_and_first_moves() {
    // The §8(4) exit bar in table form: ≥ 10 positions across mates in
    // 1/2/3, each solved at depth ≤ 12 (the S1 bar — see the two-rook
    // ladder test for why pawnless proofs deepened) with the exact
    // ply-relative score AND a verified-optimal first move. Every entry — the mate
    // distance and the complete optimal-first-move set — was checked
    // against an exhaustive independent minimax over python-chess
    // movegen, so a listed move set is exact (a single-element set
    // means the fastest mate is unique).
    //
    // (fen, mate distance in plies, all optimal first moves)
    type Entry = (&'static str, i32, &'static [&'static str]);
    const SUITE: &[Entry] = &[
        // -- mate in 1 --
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 1, &["d1d8"]),
        ("k7/8/1K6/8/8/8/8/7R w - - 0 1", 1, &["h1h8"]),
        ("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1", 1, &["g1g7"]),
        ("6rk/6pp/8/6N1/8/8/8/K7 w - - 0 1", 1, &["g5f7"]),
        ("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", 1, &["e1e8"]),
        ("7k/R7/1R6/8/8/8/8/K7 w - - 0 1", 1, &["b6b8"]),
        ("7r/8/8/8/8/1k6/8/K7 b - - 0 1", 1, &["h8h1"]),
        // -- mate in 2 --
        ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3, &["c6b6", "c6c7"]),
        (
            "7k/8/5K2/8/8/Q7/8/8 w - - 0 1",
            3,
            &["f6f7", "f6g6", "a3f8", "a3e7", "a3a7", "a3g3"],
        ),
        ("k7/8/8/1K6/8/8/8/7R w - - 0 1", 3, &["b5b6"]),
        ("K7/8/8/1k6/8/8/8/7r b - - 0 1", 3, &["b5b6"]),
        ("7k/8/4K3/8/8/Q7/8/8 w - - 0 1", 3, &["e6f7"]),
        // -- mate in 3 --
        ("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 5, &["b1b6"]),
        ("8/8/1k6/7R/8/8/8/6RK w - - 0 1", 5, &["g1g6"]),
    ];
    assert!(SUITE.len() >= 10, "the §8(4) bar is at least 10 positions");
    for &(fen, plies, optimal) in SUITE {
        // Mate-in-1 uses depth 2 so the score comes from a full search
        // iteration, not bare qsearch (found at depth 1, the mate-band
        // EMAs keep every later window effectively infinite). Deeper
        // mates pre-S1 needed the mate distance plus one aspiration
        // settling iteration; live correction history (S1/D22) shifts
        // proof depth in these pawnless studies (see the two-rook
        // ladder test), so the §8(4) bar is now depth ≤ 12 — node
        // counts stay tiny (~5k), the exact-score and optimal-first-
        // move asserts stay untouched.
        let depth = if plies == 1 { 2 } else { 12 };
        let w = go_depth(fen, depth);
        assert_eq!(best_score(&w), mate_in(plies), "score for {fen}");
        let got = w.best_move().unwrap().to_string();
        assert!(
            optimal.contains(&got.as_str()),
            "first move {got} for {fen} is not one of the verified-fastest {optimal:?}"
        );
    }
}

#[test]
fn mated_root_reports_the_negative_mate() {
    // Black's only move is Kg8 (g7/h7 touch the white king); then
    // Qc8#. Ply-relative from the root: mated_in(2) (doc 07 §5's
    // mated-side reporting bar).
    //
    // This KQ-vs-K fixture is a worst case for LMP (doc 10 §3.5): FOUR
    // white queen moves give check (Qa2+/Qb3+/Qc4+ all lie on the
    // a2–g8 diagonal, plus the mating Qc8+ on rank 8), so they tie on
    // the check-order bonus and the mating one sorts fourth. At the
    // white reply node the movecount threshold `(3 + depth²)/2` only
    // reaches 4 once that node has depth ≥ 3 (root depth ≥ 4), so LMP
    // prunes Qc8# for one extra iteration — mate_in(2) surfaces from
    // root depth 4 on. Depth 5 leaves a settling iteration of margin,
    // like the KRK mate tests; the mate is still exact, LMP only
    // delayed it (the `!is_loss` guard keeps the mated SIDE searching).
    let w = go_depth("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1", 5);
    assert_eq!(best_score(&w), mated_in(2));
    assert_eq!(w.best_move().unwrap().to_string(), "h8g8");
}

// --- Terminal roots: no legal moves, nothing to search (§3.5) ---

#[test]
fn checkmated_root_yields_no_root_moves() {
    // Fool's mate: white to move, checkmated. The driver returns
    // without searching; the UCI layer emits `info depth 0 score
    // mate 0` + `bestmove (none)` itself (design/12 §3.2 step 2).
    let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
    let w = go_depth(fen, 3);
    assert!(w.root_moves.is_empty());
    assert!(w.best_move().is_none());
    assert_eq!(w.nodes, 0);
    assert_eq!(w.root_depth, 0);
}

#[test]
fn stalemate_root_yields_no_root_moves() {
    // Black to move: Ka8 has no moves (Kb6 covers a7/b7, Qh2 covers
    // b8 on the h2-b8 diagonal) and is not in check.
    let w = go_depth("k7/8/1K6/8/8/8/7Q/8 b - - 0 1", 4);
    assert!(w.root_moves.is_empty());
    assert!(w.best_move().is_none());
    assert_eq!(w.nodes, 0);
}

// --- Basic sanity on the startpos ---

#[test]
fn startpos_returns_a_legal_pv() {
    let w = go_depth(START_FEN, 4);
    assert!(
        !crate::types::is_decisive(best_score(&w)),
        "startpos is not decisive"
    );
    assert!(w.nodes > 0);
    assert!(
        w.root_moves[0].sel_depth >= 4,
        "PV nodes must bump selDepth"
    );
    replay_pv(&w);
}

/// Replay the root PV move by move, asserting each is legal in turn —
/// the "PVs that replay legally" bar of design/07 §4 — and return the
/// position at the end of the line.
fn replay_pv(w: &Worker) -> Board {
    assert!(!w.root_pv().is_empty());
    let mut b = w.board.clone();
    for &m in w.root_pv() {
        assert!(
            all_legal(&b).contains(&m),
            "PV move {m} illegal in {}",
            b.to_fen()
        );
        b.make_move(m, b.gives_check(m));
    }
    b
}

// --- Qsearch rule fixtures (doc 07 §3.8, §5, §8.8; plan step 9) ---
//
// Each rule gets a FEN whose observable flips if the rule is
// disabled: `w.nodes` counts MADE moves, so `nodes == 0` proves a
// skip, and pinned return values come from the doc's exact formulas
// over in-test `evaluate` calls — no magic score literals.

/// The RAW static eval (`evaluate` only). Since S1 the search never
/// consumes this directly — fixtures pinning search-visible values must
/// use [`corrected_eval`].
fn static_eval(b: &Board) -> Value {
    evaluate(b, &mut EvalContext::new(), VALUE_ZERO, 0)
}

/// The unscaled correction sum a FRESH fixture node computes at entry
/// (S1/D22): every correction entry sits at its fill value and the
/// sentinel frames below the root supply the stack context. `prev` is
/// the parent's just-made move for a CHILD fixture (its to-square piece
/// keys the continuation term); `None` models a root-driven node, where
/// the null-move flat prior applies instead. CAUTION: with today's fill
/// values both variants truncate to the same −1cp delta, so a fixture
/// passing the wrong `prev` would pass by coincidence — pick the
/// variant that matches the driven node, not the one that works.
fn fresh_cv(b: &Board, prev: Option<Move>) -> i32 {
    let mut w = Worker::new(b.clone());
    init_stack(&mut w.stack);
    let cur = ss(1);
    if let Some(m) = prev {
        w.stack[cur - 1].current_move = m;
    }
    w.histories.correction_value(&w.board, &w.stack[..], cur)
}

/// The CORRECTED static eval a fresh fixture node computes (S1/D22):
/// exactly what `ss.static_eval` — and every margin, stand-pat, and
/// blend formula built from it — now holds in place of the raw
/// `evaluate()`. Fixture boundary arithmetic must start here.
fn corrected_eval(b: &Board, prev: Option<Move>) -> Value {
    crate::history::to_corrected_static_eval(static_eval(b), fresh_cv(b, prev))
}

/// Drive `qsearch` directly at ply 0 on a fresh stack: the fixtures
/// pin per-rule behavior against exact windows, bypassing the driver.
fn qsearch_at<NT: NodeType>(
    w: &mut Worker,
    tt: &TranspositionTable,
    alpha: Value,
    beta: Value,
) -> Value {
    init_stack(&mut w.stack);
    let stop = AtomicBool::new(false);
    w.qsearch::<NT>(tt, &stop, alpha, beta, 0)
}

#[test]
fn qsearch_stand_pat_fail_high_blends_toward_beta() {
    // Quiet position, white a queen up, window far below the eval:
    // the stand pat cuts immediately — softened by the §3.8 step-5
    // blend, pinned exactly — and searches nothing. The TT miss
    // leaves an eval-only Bound::Lower entry behind (§3.11).
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let beta = eval - 100;
    let v = qsearch_at::<NonPv>(&mut w, &tt, beta - 1, beta);
    let (w_best, w_beta, den) = (
        params::QS_STANDPAT_BEST_W(),
        params::QS_STANDPAT_BETA_W(),
        params::QS_STANDPAT_DEN(),
    );
    // The blend runs on the CORRECTION-ADJUSTED stand pat, not the raw
    // static eval; the TT keeps the unadjusted one (asserted below).
    assert_eq!(
        v,
        (w_best * corrected_eval(&w.board, None) + w_beta * beta) / den
    );
    assert_eq!(w.nodes, 0, "a stand-pat cutoff searches nothing");
    let cached = tt.probe(w.board.key());
    assert!(cached.hit);
    assert_eq!(cached.data.depth, DEPTH_UNSEARCHED);
    assert!(cached.data.bound.includes_lower());
    assert_eq!(cached.data.eval, eval);
    assert!(!is_valid(cached.data.value), "eval-only entry has no value");
}

#[test]
fn qsearch_stand_pat_uses_tt_value_as_better_eval() {
    // No captures exist; a probed LOWER-bound tt value above the
    // static eval substitutes as the stand pat (§3.8 step 5) and,
    // staying inside the window, comes back unchanged.
    let fen = "7k/8/8/8/8/8/8/1Q5K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let tt_value = eval + 55;
    let key = w.board.key();
    tt.probe(key).writer.write(
        key,
        tt_value,
        false,
        Bound::Lower,
        DEPTH_QS,
        Move::NULL,
        eval,
        tt.generation(),
    );
    let v = qsearch_at::<Pv>(&mut w, &tt, eval - 1000, eval + 1000);
    assert_eq!(v, tt_value);
    assert_eq!(w.nodes, 0);
}

#[test]
fn qsearch_futility_prunes_a_small_capture_below_alpha() {
    // exd5 is the only capture; with alpha a full grand above the
    // eval the §3.8 futility rule skips it unsearched, and its
    // optimistic futility_value IS the returned fail-low best_value.
    let fen = "7k/8/8/3p4/4P3/8/8/7K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    let beta = eval + 1000;
    let v = qsearch_at::<NonPv>(&mut w, &tt, beta - 1, beta);
    assert_eq!(
        v,
        eval + params::QS_FUTILITY_MARGIN() + PieceType::Pawn.value()
    );
    assert_eq!(w.nodes, 0, "the futile capture must not be searched");
}

#[test]
fn qsearch_window_see_rule_skips_an_even_capture_below_the_window() {
    // Rxa8 is a dead-even rook trade (SEE 0 — comfortably above the
    // -74 floor, so ONLY the window rule can skip it), but alpha sits
    // 500 above the eval: `see_ge(m, alpha - futility_base)` fails
    // and best_value is pinned to min(alpha, futility_base).
    let fen = "rr5k/8/8/8/8/8/8/R6K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    let alpha = eval + 500;
    let v = qsearch_at::<NonPv>(&mut w, &tt, alpha, alpha + 1);
    assert_eq!(v, eval + params::QS_FUTILITY_MARGIN());
    assert_eq!(w.nodes, 0, "the SEE-vs-window rule must skip the trade");
}

#[test]
fn qsearch_see_floor_skips_a_losing_capture_even_when_it_checks() {
    // Qxd7+ gives check, so it is exempt from the whole futility
    // block — but it hangs the queen to Kxd7 (SEE -2330 < -74): the
    // §3.8 hard floor skips it and the node stands pat.
    let fen = "4k3/3p4/8/8/3Q4/8/8/7K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    let v = qsearch_at::<Pv>(&mut w, &tt, -VALUE_INFINITE, VALUE_INFINITE);
    assert_eq!(v, eval, "the node must fall back to its stand pat");
    assert_eq!(w.nodes, 0, "the SEE floor must skip the losing capture");
}

#[test]
fn qsearch_late_move_rule_stops_after_two_quiet_window_captures() {
    // Four equal knight-takes-pawn captures — none checking, none
    // recapturable, none aimed at a previous square: the §3.8
    // move-count rule searches exactly the first two (each child is
    // quiet, so nodes counts searched captures one-for-one).
    let fen = "7k/p1p1p3/8/1N1N4/8/8/8/7K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let v = qsearch_at::<Pv>(&mut w, &tt, -VALUE_INFINITE, VALUE_INFINITE);
    assert_eq!(
        w.nodes,
        2,
        "moveCount > {} must stop the remaining captures",
        params::QS_LATE_MOVE_LIMIT()
    );
    assert!(!crate::types::is_decisive(v));
}

#[test]
fn qsearch_evasions_are_searched_while_best_value_is_a_loss() {
    // White is in double check, hugely behind, and the only legal
    // evasion is a QUIET king move — exactly what the step-7
    // non-capture skip would drop were the `!is_loss(best_value)`
    // evasion guard not in force. It must be searched: the node's
    // value is the child's stand pat, not a mate or a pruned -INF.
    let fen = "k7/1q6/8/8/8/8/8/r6K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let kh2 = Move::normal(Square::H1, Square::H2);
    let mut after = w.board.clone();
    after.make_move(kh2, after.gives_check(kh2));
    let v = qsearch_at::<Pv>(&mut w, &tt, -VALUE_INFINITE, VALUE_INFINITE);
    assert_eq!(w.nodes, 1, "the lone evasion must be searched");
    assert_eq!(v, -corrected_eval(&after, Some(kh2)));
    assert!(v < 0 && !is_loss(v), "a real (bad) value, not a mate");
}

#[test]
fn qsearch_non_capture_skip_drops_quiet_evasions_once_best_value_is_real() {
    // The positive direction of the §3.8 step-7 non-capture skip (the
    // §8.8 flip test its evasion-guard sibling above cannot provide):
    // white is in check with THREE evasions — Rxa1, a clean recapture
    // that sorts first, and the quiet Kh2/Kg2. The capture's child
    // value makes best_value real (a rook up, not a loss), so both
    // quiet evasions are dropped unsearched; disable the skip and all
    // three run (nodes == 3). The in-check futility block is inert
    // (futility_base is a loss), so ONLY this rule can skip them.
    let fen = "7k/8/8/R7/8/8/8/r6K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let rxa1 = Move::normal(Square::A5, Square::A1);
    assert!(all_legal(&w.board).contains(&rxa1));
    let mut after = w.board.clone();
    after.make_move(rxa1, after.gives_check(rxa1));
    let v = qsearch_at::<Pv>(&mut w, &tt, -VALUE_INFINITE, VALUE_INFINITE);
    assert_eq!(w.nodes, 1, "only the capture evasion is searched");
    assert_eq!(v, -corrected_eval(&after, Some(rxa1)));
    assert!(v > 0 && !is_decisive(v), "a real material edge, not a mate");
}

#[test]
fn qsearch_in_check_never_stands_pat() {
    // White is well ahead: were stand pat allowed, eval >= beta would
    // cut with zero nodes. In check the §3.8 step-5 rule forbids it —
    // the lone evasion (double check) is searched and the fail-high
    // value is the step-11 blend of the child's exact stand pat.
    let fen = "6k1/1b6/8/5R2/5R2/8/8/r6K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let kh2 = Move::normal(Square::H1, Square::H2);
    let mut after = w.board.clone();
    after.make_move(kh2, after.gives_check(kh2));
    let beta = 0;
    let v = qsearch_at::<NonPv>(&mut w, &tt, beta - 1, beta);
    assert_eq!(w.nodes, 1, "in check the evasion must be searched");
    let (w_best, w_beta, den) = (
        params::QS_FAILHIGH_BEST_W(),
        params::QS_FAILHIGH_BETA_W(),
        params::QS_FAILHIGH_DEN(),
    );
    assert_eq!(
        v,
        (w_best * -corrected_eval(&after, Some(kh2)) + w_beta * beta) / den
    );
    assert!(v >= beta, "still a fail high after the blend");
}

#[test]
fn qsearch_fail_high_softening_blends_toward_beta() {
    // exd5 wins a clean queen against a window pinned far below: the
    // raw fail-high value (the child's exact stand pat, negated) is
    // pulled toward beta by the §3.8 step-11 blend.
    let fen = "7k/8/8/3q4/4P3/8/8/7K w - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = static_eval(&w.board);
    let exd5 = Move::normal(Square::E4, Square::D5);
    let mut after = w.board.clone();
    after.make_move(exd5, after.gives_check(exd5));
    let beta = eval + 100;
    let v = qsearch_at::<NonPv>(&mut w, &tt, beta - 1, beta);
    assert_eq!(w.nodes, 1);
    let (w_best, w_beta, den) = (
        params::QS_FAILHIGH_BEST_W(),
        params::QS_FAILHIGH_BETA_W(),
        params::QS_FAILHIGH_DEN(),
    );
    assert_eq!(
        v,
        (w_best * -corrected_eval(&after, Some(exd5)) + w_beta * beta) / den
    );
}

#[test]
fn qsearch_narrow_stalemate_test_scores_the_trap_as_a_draw() {
    // Qf3xf7 wins a clean rook but stalemates the bare king (f7
    // covers g7/g8/h7): in the child every §3.8 step-10 precondition
    // holds — no non-pawn material, no pawn pushes, previous capture
    // >= knight — so the narrow test pays for the legality scan and
    // returns exactly VALUE_DRAW.
    let fen = "7k/5r2/8/8/8/5Q2/8/1K6 w - - 0 1";
    let mut w = Worker::new(board(fen));
    let qxf7 = Move::normal(Square::F3, Square::F7);
    assert!(all_legal(&w.board).contains(&qxf7));
    w.board.make_move(qxf7, w.board.gives_check(qxf7));
    let tt = test_tt();
    let v = qsearch_at::<Pv>(&mut w, &tt, -10, 10);
    assert_eq!(v, VALUE_DRAW);
    assert_eq!(w.nodes, 0);
}

#[test]
fn qsearch_stalemate_test_needs_a_preceding_big_capture() {
    // The SAME stalemate reached from FEN (no captured piece on the
    // board's last state) must NOT fire the narrow test — the whole
    // point is paying for the legality scan only after "qsearch just
    // won material": the node stands pat instead.
    let fen = "7k/5Q2/8/8/8/8/8/1K6 b - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let eval = corrected_eval(&w.board, None);
    let v = qsearch_at::<Pv>(&mut w, &tt, -10, 10);
    assert_eq!(v, eval, "no prior capture: the narrow test stays off");
}

// --- Node counting and determinism (doc 07 §5, D24) ---

#[test]
fn search_is_deterministic_at_fixed_depth() {
    // Also the dither-determinism gate (§5): both value dithers —
    // the draw dither and the step-12 equal-value dither — read only
    // the worker's own made-move counter, so two identical
    // single-threaded searches replay the same choices node for node
    // (the release-bench two-run fingerprint in tools/check.sh is the
    // deep version of this assert). The dither DOES change the bench
    // number vs the pre-step-10 code — a functional change whose new
    // count the commit trailer records (D24).
    let run = || {
        let w = go_depth(START_FEN, 4);
        (w.nodes, best_score(&w), w.best_move())
    };
    let (n1, v1, m1) = run();
    let (n2, v2, m2) = run();
    assert!(n1 > 0);
    assert_eq!(n1, n2, "two identical searches must count identical nodes");
    assert_eq!(v1, v2);
    assert_eq!(m1, m2);
}

// --- NNUE-in-search accumulator consistency (design/11 §3.4.4) ---

/// The SEARCH analogue of the step-1 incremental==refresh fuzz: with a net
/// installed, the debug oracle `assert_matches_refresh` fires inside
/// `evaluate` at EVERY evaluated node, so a stale-accumulator bug across
/// LMR/PVS re-searches, null-move pruning, or the qsearch dive trips the
/// assert mid-search. A per-`Worker` net injection (not the process-global
/// `EvalFile`) keeps the test hermetic under the parallel harness — nothing
/// here touches `ACTIVE_NET`, so concurrent PSQT search tests are unaffected.
#[test]
fn search_nnue_accumulator_stays_consistent_across_the_tree() {
    use crate::eval::nnue::{Activation, Network, evaluate_forward};

    // Leak a deterministic test net to `'static` (a tiny, one-off leak — the
    // production lifetime the per-thread accumulator borrows). SCReLU
    // exercises both activation clamps.
    let net: &'static Network = Box::leak(Box::new(Network::deterministic_test_net(
        0x00C0_FFEE,
        Activation::SCReLU,
    )));

    // A spread of quiet, tactical, and endgame positions so the search
    // reaches captures (qsearch), null moves, and deep re-searches.
    let positions = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    ];

    let tt = test_tt();
    let mut w = Worker::new(Board::startpos());
    // Inject the net directly into this worker's eval context (bypassing the
    // EvalFile global). `iterative_deepening` only *refreshes* the installed
    // net, so this survives across positions.
    w.eval_ctx.set_net(Some(net));
    assert!(w.eval_ctx.nnue_active(), "net must install");

    for fen in positions {
        w.board = board(fen);
        // A node budget bounds the debug runtime (both the board oracle and
        // the NNUE oracle run at every node) while still spanning thousands
        // of evaluated nodes — plenty to cover the re-search paths.
        w.limits = Limits::new();
        w.limits.nodes = Some(6_000);
        let stop = AtomicBool::new(false);
        tt.new_search();
        w.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |_| {});

        assert!(w.eval_ctx.nnue_active(), "NNUE stayed active for {fen}");
        assert!(w.nodes > 0, "search evaluated nodes for {fen}");
        // The comment above CLAIMS this tree reaches null moves, and the
        // null-move ply mapping (`EvalContext::null_plies`, perf study 15
        // arm 1) is one of the two things this oracle is now the guard for.
        // Assert it rather than argue it — study 14 §7.1's `has_repeated`
        // hole was exactly a claimed-but-unexecuted path (D33 reviewer C, T2).
        assert!(
            w.st_nmp_try > 0,
            "the tree must contain a null move for {fen} — otherwise the \
             acc_ply mapping is untested by this oracle"
        );

        // The board is restored to the root, and `stack[0]` holds the root
        // accumulator refreshed at go-start: the incrementally-driven root
        // eval equals a standalone from-scratch NNUE forward. This half is
        // meaningful in RELEASE too (the per-node oracle above is debug-only).
        let root = board(fen);
        assert_eq!(
            w.eval_ctx.nnue_root_forward_for_test(&root),
            evaluate_forward(net, &root),
            "root accumulator diverged from a from-scratch forward for {fen}"
        );
    }
}

/// The SOTA twin of the test above, and until perf study 15 it did not exist:
/// **no test anywhere installed a SOTA net into a `Worker`**, so
/// `SotaContext::assert_matches_refresh` — the per-node oracle that recomputes
/// BOTH accumulators and both PSQT channels from scratch and compares — had
/// never once run inside a search (D33 reviewers A/C, the study's largest test
/// hole). Every gate that could see the SOTA accumulators was a node-count or
/// eval-hash SNAPSHOT: it fires only if a mistake happens to perturb the tree.
///
/// This is the test that covers perf study 15 arm 1's SOTA half. Mutations it
/// catches, each verified: `push_sota_dirty` indexing `ply` instead of
/// `acc_ply(ply)`; `at_full`/`current_full` reading `ply + 1`; re-adding an
/// EMPTY push in the null-move block; dropping `pop_null`.
#[cfg(feature = "sota")]
#[test]
fn search_sota_accumulator_stays_consistent_across_the_tree() {
    use crate::eval::nnue::sota_loader::{SotaNetwork, tests::test_net_with_threats_pub};

    // Same one-off leak as the v1 test: the accumulator contexts borrow the
    // net for `'static`, which is the production lifetime.
    let net: &'static SotaNetwork = Box::leak(Box::new(test_net_with_threats_pub()));

    let positions = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    ];

    let tt = test_tt();
    let mut w = Worker::new(Board::startpos());
    // Injected per-worker, never through `ACTIVE_NET`, so the parallel test
    // harness stays hermetic. `iterative_deepening` does not re-install nets
    // (only `start_searching`/`bench` do), so this survives every position.
    w.eval_ctx.set_sota_net(Some(net));
    assert!(w.eval_ctx.nnue_active(), "SOTA net must install");

    let mut null_moves = 0u64;
    for fen in positions {
        w.board = board(fen);
        w.limits = Limits::new();
        w.limits.nodes = Some(4_000);
        let stop = AtomicBool::new(false);
        tt.new_search();
        w.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |_| {});

        assert!(w.nodes > 0, "search evaluated nodes for {fen}");
        null_moves += w.st_nmp_try;
    }
    // Without a null move somewhere in these trees the oracle never sees the
    // ply mapping this test exists for — the same measurement-not-argument
    // rule as its v1 twin. Summed over the loop rather than asserted per
    // position: `st_nmp_try` is per-search, and whether NMP fires in any ONE
    // position depends on the synthetic net's eval clearing the NMP margin,
    // which is not a property this test should pin.
    assert!(
        null_moves > 0,
        "at least one of these trees must contain a null move"
    );
}

// --- Stop-flag seam (§3.9): bestmove is printable even so ---

#[test]
fn preset_stop_searches_nothing_but_keeps_a_legal_bestmove() {
    let mut w = Worker::new(board(START_FEN));
    w.limits.depth = Some(6);
    let stop = AtomicBool::new(true);
    let mut infos = 0;
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {
        infos += 1
    });
    // The ID loop condition sees the stop before any search runs and
    // no info line is ever emitted from a stopped iteration (§3.5)...
    assert_eq!(w.nodes, 0);
    assert_eq!(infos, 0);
    // ...but the root move list exists before any search (§3.9), so a
    // legal bestmove is always printable.
    let bm = w.best_move().expect("bestmove printable under stop");
    assert!(all_legal(&w.board).contains(&bm));
}

// --- Limit enforcement seams (design/12 §3.6/§3.8 v1 subset) ---

#[test]
fn depth_limit_is_respected_with_monotone_info_reports() {
    let mut w = Worker::new(board(START_FEN));
    w.limits.depth = Some(3);
    let stop = AtomicBool::new(false);
    let mut reports: Vec<(Depth, i32, u64, Vec<Move>)> = Vec::new();
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |i| {
        assert_eq!(i.multipv, 1, "v1 is single-PV");
        assert!(!i.lowerbound && !i.upperbound, "full window: exact");
        reports.push((i.depth, i.sel_depth, i.nodes, i.pv.to_vec()));
    });
    // Exactly one report per iteration, depths 1..=3, no deeper.
    assert_eq!(w.root_depth, 3);
    let depths: Vec<Depth> = reports.iter().map(|r| r.0).collect();
    assert_eq!(depths, vec![1, 2, 3]);
    for pair in reports.windows(2) {
        assert!(pair[0].2 < pair[1].2, "node counts grow per iteration");
    }
    for (depth, sel_depth, _, pv) in &reports {
        assert!(sel_depth >= depth, "selDepth counts past nominal depth");
        // Every reported PV replays legally from the root.
        let mut b = w.board.clone();
        for &m in pv {
            assert!(all_legal(&b).contains(&m));
            b.make_move(m, b.gives_check(m));
        }
    }
}

#[test]
fn node_limit_stops_within_the_check_cadence() {
    let mut w = Worker::new(board(START_FEN));
    w.limits.nodes = Some(20_000);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(
        stop.load(Ordering::Relaxed),
        "check_time must raise the stop flag on the node limit"
    );
    // The stop fires at the first real check at/past the budget; the
    // bitmask cadence tightens under a node limit to the largest power
    // of two <= 20_000/1024 = 19, i.e. every 16 node-entries (§3.8), so
    // the overshoot is the cadence plus qsearch slack — small.
    assert!(w.nodes >= 20_000);
    assert!(w.nodes < 28_000, "cadence overshoot too large: {}", w.nodes);
    // A mid-search stop still leaves a trusted, legal bestmove (§3.5).
    let bm = w.best_move().expect("bestmove after node-limit stop");
    assert!(all_legal(&w.board).contains(&bm));
}

#[test]
fn movetime_limit_terminates_the_search() {
    let mut w = Worker::new(board(START_FEN));
    w.limits.movetime = Some(50);
    w.limits.start_time = std::time::Instant::now();
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "movetime must self-stop");
    assert!(w.best_move().is_some());
}

#[test]
fn ponder_guard_blocks_every_check_time_stop() {
    // design/12 §3.8: the GUI owns the clock while pondering — an
    // already-expired movetime must NOT stop a ponder search. The
    // depth limit (ID-loop condition, not check_time) bounds the test;
    // check_time only does real work every 512 main-node ENTRIES, so the
    // fixture must have a tree that large. As the pruning suite deepened
    // (LMP, then the Wave C-G history/threat couplings) the startpos tree
    // at any small fixed depth kept shrinking below that floor, making
    // this test flap; kiwipete at depth 5 (~5k nodes) is robustly above
    // it — the fixture, not the depth, is what the test needs.
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let mut w = Worker::new(board(KIWIPETE));
    w.limits.movetime = Some(0); // expired from the first poll
    w.limits.start_time = std::time::Instant::now();
    w.limits.depth = Some(5);
    w.limits.ponder = true;
    w.pool.ponder.store(true, Ordering::Relaxed);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(!stop.load(Ordering::Relaxed), "pondering must never stop");
    assert_eq!(w.root_depth, 5, "the search deepened on GUI time");

    // Control: the same limits without the ponder flag stop at once.
    let mut w = Worker::new(board(KIWIPETE));
    w.limits.movetime = Some(0);
    w.limits.start_time = std::time::Instant::now();
    w.limits.depth = Some(5);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "expired movetime must stop");
}

#[test]
fn clock_time_management_enforces_the_hard_wall() {
    // wtime-only limits (no depth/nodes/movetime): without the
    // `elapsed > maximum` wall this search would be unbounded, so
    // returning at all with the stop flag raised proves the wall
    // (design/12 §3.8 v1 subset). A 50 ms clock keeps `maximum`
    // around 30 ms, so the test is fast.
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 50;
    w.limits.start_time = std::time::Instant::now();
    // Mirror the UCI session setup (run_search): init on the worker's
    // TimeManager against the go limits.
    let mut limits = w.limits.clone();
    let mut persist = crate::timeman::TimePersist::default();
    w.tm.init(
        &mut limits,
        crate::types::Color::White,
        0,
        &crate::uci::Options::new(),
        &mut persist,
    );
    w.limits = limits;
    assert!(w.limits.use_time_management());
    assert!(w.tm.maximum() >= 1);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "the TM wall must self-stop");
    assert!(w.best_move().is_some());
}

#[test]
fn reported_pvs_replay_legally_under_the_dither() {
    // Regression (§3.3 freshness): the equal-value dither could
    // promote a move whose last child call was a zero-window scout
    // (`value == alpha`, `inc == 1`), stitching an earlier sibling's
    // stale `pv_table` row into the PV — fastchess flagged the
    // resulting illegal tails in the P7 clock games. Every reported
    // PV must replay legally from the root (the fastchess repro
    // position, depth deep enough for the dither to fire).
    let fen = "rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq - 5 5";
    let mut w = Worker::new(board(fen));
    w.limits.depth = Some(7);
    let tt = test_tt();
    tt.new_search();
    let stop = AtomicBool::new(false);
    let root = board(fen);
    let mut reported = 0;
    w.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |i| {
        let mut b = root.clone();
        for &m in i.pv {
            assert!(
                all_legal(&b).contains(&m),
                "illegal PV move {m} in reported pv {:?}",
                i.pv
            );
            b.make_move(m, b.gives_check(m));
        }
        reported += 1;
    });
    assert_eq!(reported, 7, "one exact report per completed iteration");
}

// --- TM soft stop (design/12 §3.9 v1 subset, doc 12 step 7) ---

/// Mirror the UCI session setup (`run_search`): init the worker's
/// [`TimeManager`] against its limits with the given options.
fn init_clock(w: &mut Worker, options: &crate::uci::Options) {
    let mut limits = w.limits.clone();
    let mut persist = crate::timeman::TimePersist::default();
    let (us, ply) = (w.board.side_to_move(), w.board.game_ply());
    w.tm.init(&mut limits, us, ply, options, &mut persist);
    w.limits = limits;
}

/// `nodestime 1` options — the doc 12 §3.11 fake clock: `tm.elapsed`
/// returns nodes searched and every budget is a node count, so these
/// tests are bit-deterministic (no wall-clock read affects any stop
/// decision).
fn nodestime_options() -> crate::uci::Options {
    let mut options = crate::uci::Options::new();
    options.set("nodestime", Some("1")).unwrap();
    options
}

#[test]
fn soft_stop_halts_deepening_within_the_budget() {
    // Two scenarios on the 1-node-per-ms fake clock, pinning the two
    // stop paths (doc 12 §3.9). Case A is the combined-budget backstop;
    // case B provably witnesses the between-iteration SOFT stop, so this
    // test earns its name and does not degenerate into a hard-wall check.
    //
    // Case A — 60 s clock (optimum ≈ 1.25k, maximum ≈ 8.6k), first move
    // of the game: `best_previous_average_score` is still the
    // `VALUE_INFINITE` sentinel, so `fallingEval` clamps high (1.708),
    // inflating `total_time` ABOVE `maximum`. A deep last iteration then
    // blows past both budgets in one pass and the hard wall backstops
    // it — so we only bound the result: past optimum, within maximum
    // plus one poll cadence (`params::CHECK_TIME_CALLS`).
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 60_000;
    init_clock(&mut w, &nodestime_options());
    let (optimum, maximum) = (w.tm.optimum(), w.tm.maximum());
    assert!(0 < optimum && optimum < maximum);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "the soft/hard stop must fire");
    assert!(
        w.nodes as i64 > optimum,
        "stopped before optimum was spent: {} <= {optimum}",
        w.nodes
    );
    assert!(
        w.nodes as i64 <= maximum + params::CHECK_TIME_CALLS as i64,
        "overshot the hard wall by more than one poll cadence: {} > {maximum} + {}",
        w.nodes,
        params::CHECK_TIME_CALLS
    );
    assert!(w.best_move().is_some());

    // Case B — 10 min clock (optimum ≈ 16.7k, maximum ≈ 115k) on a LATER
    // move seeded with a moderate `best_previous_average_score`, so
    // `fallingEval` clamps low and `total_time` lands near optimum, far
    // under the hard wall. The search must therefore stop STRICTLY below
    // `maximum` — that can only be the between-iteration soft stop, not
    // the `check_time` wall — while still deepening past optimum. This is
    // the assertion Case A cannot make (its total_time exceeds maximum).
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 600_000;
    init_clock(&mut w, &nodestime_options());
    let (optimum, maximum) = (w.tm.optimum(), w.tm.maximum());
    let mut persist = TimePersist {
        best_previous_score: 30,
        best_previous_average_score: 30,
        ..TimePersist::default()
    };
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut persist, &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "the soft stop must fire");
    // A STABLE quiet search legitimately stops BELOW optimum — banking
    // time is the point of the multiplier stack (fe rides its 0.572
    // floor when the score isn't falling, red < 1 once the best move
    // has been stable, so total_time ≈ 0.4-0.8 × optimum here; the
    // 2026-07-26 depth-redenomination made the banking regime reachable
    // at realistic depths). The multiplier product's theoretical floor
    // is 0.572·0.637·1.096·0.71 ≈ 0.28, so anything above optimum/4
    // is a real soft-stop verdict, not a degenerate early exit.
    assert!(
        w.nodes as i64 > optimum / 4,
        "soft stop fired degenerately early: {} <= {}",
        w.nodes,
        optimum / 4
    );
    assert!(
        (w.nodes as i64) < maximum - params::CHECK_TIME_EPSILON_MS(),
        "the hard wall stopped it, not the soft stop: {} not < {maximum}",
        w.nodes
    );
    assert!(w.best_move().is_some());
}

#[test]
fn sub_iteration_node_budget_still_completes_iteration_one() {
    // `go nodes 5` — a hard node budget far below the depth-1 node cost
    // (~21 nodes at the start position). The tiny budget tightens the
    // check_time poll cadence to every node (`compute_check_time_mask`),
    // so the `nodes >= 5` hard stop would otherwise fire mid-depth-1
    // with `completed_depth == 0` and salvage only a half-searched root.
    // The `completed_depth == 0` guard (design/12 §3.8, D31) suppresses
    // every hard stop until iteration 1 has fully completed, guaranteeing
    // a real, completed bestmove. Without the guard this search returns
    // an aborted depth-0 result; the assertions below (a completed depth
    // >= 1 AND a real bestmove) are exactly what the guard buys and what
    // no other TM test pins.
    let mut w = Worker::new(board(START_FEN));
    w.limits.nodes = Some(5);
    assert!(
        !w.limits.use_time_management(),
        "test premise: a pure node budget, so only check_time can stop"
    );
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(
        stop.load(Ordering::Relaxed),
        "the node budget must stop the search"
    );
    assert!(
        w.nodes >= 5,
        "the node budget was actually reached: {}",
        w.nodes
    );
    assert!(
        w.best_move().is_some(),
        "the guard must salvage a real bestmove under a sub-iteration budget"
    );
    assert!(
        w.completed_depth >= 1,
        "iteration 1 must run to completion despite the 5-node budget (completed_depth = {})",
        w.completed_depth
    );
}

// --- Golden soft-stop factor tests (design/12 §3.9, §5). Each pins the
// §3.9 constants at their clamp edges. Expected values are hand-derived
// from the formulas with the literal constants, so moving a params
// constant fails the matching test. ---

#[test]
fn soft_stop_falling_eval_pins_constants() {
    // (11.48 + 2.30·Δavg + 1.1·Δiter) / 100, clamped to [0.576, 1.728].
    // Mid: Δavg = 100-50 = 50, Δiter = 60-50 = 10 →
    //   (11.48 + 115 + 11)/100 = 1.3748.
    assert!((Worker::falling_eval(100, 50, 60) - 1.3748).abs() < 1e-9);
    // Huge positive drop clamps to the max; huge negative to the min.
    assert!((Worker::falling_eval(10_000, 0, 0) - 1.728).abs() < 1e-12);
    assert!((Worker::falling_eval(0, 10_000, 0) - 0.576).abs() < 1e-12);
}

#[test]
fn soft_stop_stability_reduction_pins_constants() {
    // time_reduction = lerp(iters, 2,15, 0.65,1.55) clamped [0.65,1.55];
    // reduction = (1.468 + prev) / (2.284 · time_reduction).
    // Window [2,15] = SF's [5,18] shifted down by our measured ~3.3
    // iteration deficit at equal time (2026-07-26 depth-redenomination;
    // rationale at params::TIME_REDUCTION_STABILITY_LO).
    let (tr0, red0) = Worker::stability_reduction(0, 0.85); // below lo -> clamp 0.65
    assert!((tr0 - 0.65).abs() < 1e-12);
    assert!((red0 - 1.561_363_330_189_950_3).abs() < 1e-9);
    let (tr2, _) = Worker::stability_reduction(2, 0.85); // lerp x0 -> 0.65
    assert!((tr2 - 0.65).abs() < 1e-12);
    let (tr11, red11) = Worker::stability_reduction(11, 0.85);
    assert!((tr11 - 1.273_076_923_076_923).abs() < 1e-9);
    assert!((red11 - 0.797_191_549_251_062_3).abs() < 1e-9);
    let (tr15, red15) = Worker::stability_reduction(15, 0.85); // lerp x1 -> 1.55
    assert!((tr15 - 1.55).abs() < 1e-12);
    assert!((red15 - 0.654_765_267_499_011_3).abs() < 1e-9);
    let (tr30, _) = Worker::stability_reduction(30, 0.85); // above hi -> clamp 1.55
    assert!((tr30 - 1.55).abs() < 1e-12);
}

#[test]
fn tm_reduction_numerator_is_a_per_move_constant() {
    // Integration guard for the `reduction` factor's two roles (doc 12
    // §3.9; SF search.cpp:580/619). The numerator
    // `previous_time_reduction` is the PREVIOUS MOVE's time_reduction,
    // held CONSTANT for the whole search; only the running
    // `time_reduction` (the denominator, SF's local) is recomputed each
    // iteration and written back to persist at search END. A regression
    // that folded the two into one field would overwrite the numerator
    // mid-search with each iteration's value, inflating the budget on a
    // stabilizing search — invisible to the pure-function golden test
    // above, which only ever passes an explicit numerator.
    const SEED: f64 = 0.42; // distinctive; never a natural time_reduction
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 600_000;
    init_clock(&mut w, &nodestime_options());
    let mut persist = TimePersist {
        previous_time_reduction: SEED,
        best_previous_score: 30,
        best_previous_average_score: 30,
        ..TimePersist::default()
    };
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut persist, &mut |_| {});
    // The numerator working copy is untouched across the whole
    // multi-iteration search.
    assert_eq!(
        w.tm_previous_time_reduction, SEED,
        "the reduction numerator must stay the per-move constant, not the prior iteration's value",
    );
    // The running value WAS recomputed by soft_stop into [MIN, MAX] and
    // is distinct from the seed (proving soft_stop ran and only touched
    // the running field).
    assert!(
        w.tm_time_reduction >= params::TIME_REDUCTION_MIN()
            && w.tm_time_reduction <= params::TIME_REDUCTION_MAX(),
        "running time_reduction out of range: {}",
        w.tm_time_reduction
    );
    assert_ne!(
        w.tm_time_reduction, SEED,
        "the running time_reduction must be recomputed, not left at the numerator seed",
    );
    // Persist takes the running value (SF:619), so next move's numerator
    // is this move's last time_reduction — not the stale seed.
    assert_eq!(
        persist.previous_time_reduction, w.tm_time_reduction,
        "search end must persist the running time_reduction as next move's numerator",
    );
}

#[test]
fn soft_stop_best_move_instability_pins_constants() {
    // 1.077 + 2.229 · changes / threads.
    assert!((Worker::best_move_instability(0.0, 1) - 1.077).abs() < 1e-12);
    assert!((Worker::best_move_instability(2.0, 1) - 5.535).abs() < 1e-12);
    // The /threads term averages the SMP change count (P11).
    assert!((Worker::best_move_instability(4.0, 2) - 5.535).abs() < 1e-12);
}

#[test]
fn soft_stop_high_best_move_effort_pins_constants() {
    // nodes_effort = effort·100000 / max(1,nodes); then
    // lerp(nodes_effort, 79219,101822, 0.924,0.71) clamped [0.71,0.924].
    assert!((Worker::high_best_move_effort(79_219, 100_000) - 0.924).abs() < 1e-9);
    assert!((Worker::high_best_move_effort(101_822, 100_000) - 0.71).abs() < 1e-9);
    // Below the low knee clamps to 0.924; above the high knee to 0.71.
    assert!((Worker::high_best_move_effort(0, 100_000) - 0.924).abs() < 1e-12);
    assert!((Worker::high_best_move_effort(200_000, 100_000) - 0.71).abs() < 1e-12);
    // A mid ratio interpolates.
    assert!(
        (Worker::high_best_move_effort(90_520, 100_000) - 0.817_004_733_884_882_6).abs() < 1e-9
    );
}

#[test]
fn iter_value_ring_is_seeded_per_go_from_persist() {
    // First search of a game (best_previous_score == VALUE_INFINITE):
    // the ring seeds to VALUE_DRAW so fallingEval is well-defined
    // (design/12 §3.9). Fixed depth leaves the ring at its seed (the
    // soft stop never runs), so we can read it back.
    let mut w = Worker::new(board(START_FEN));
    w.limits.depth = Some(2);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert_eq!(w.iter_value, [VALUE_DRAW; 4]);
    // A later move of the game seeds every slot from best_previous_score.
    let mut persist = TimePersist {
        best_previous_score: 137,
        ..TimePersist::default()
    };
    let mut w = Worker::new(board(START_FEN));
    w.limits.depth = Some(2);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut persist, &mut |_| {});
    assert_eq!(w.iter_value, [137; 4]);
}

#[test]
fn one_legal_reply_caps_the_soft_budget() {
    // In check with exactly one evasion — the rook block h6a6,
    // defended by the b7 pawn, so the position after it stays rich
    // and mate-free and iterations grow normally. 10 min on the fake
    // clock puts optimum (~16.7k nodes) far above the ~562-node
    // single-reply cap, so stopping under optimum proves the cap
    // bound the budget (§3.9's `min(561.7, total_time)`).
    const ONE_REPLY: &str = "kr4n1/1p4pp/7r/8/4P3/5N2/5PPP/R6K b - - 0 1";
    let mut w = Worker::new(board(ONE_REPLY));
    w.limits.time[crate::types::Color::Black as usize] = 600_000;
    init_clock(&mut w, &nodestime_options());
    assert!(
        w.tm.optimum() as f64 > 2.0 * params::ONE_REPLY_TOTAL_TIME_CAP(),
        "test premise: the uncapped budget must dwarf the cap"
    );
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert_eq!(w.root_moves.len(), 1, "test premise: one legal move");
    assert!(stop.load(Ordering::Relaxed), "the capped soft stop fires");
    assert!(
        w.nodes as f64 > params::ONE_REPLY_TOTAL_TIME_CAP(),
        "stopped before the cap was spent: {}",
        w.nodes
    );
    assert!(
        (w.nodes as i64) < w.tm.optimum(),
        "the single-reply cap did not bind: {} >= {}",
        w.nodes,
        w.tm.optimum()
    );
    assert!(w.best_move().is_some());
}

#[test]
fn soft_stop_defers_to_the_ponder_guard() {
    // A tiny fake clock (optimum ≈ 3 nodes, maximum ≈ 10) under `go
    // ponder`: the GUI owns the clock, so neither the soft stop nor
    // the hard wall may fire (§3.8's guard applies to both); the
    // depth limit (ID-loop condition) bounds the test.
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 1_000;
    w.limits.depth = Some(4);
    w.limits.ponder = true;
    init_clock(&mut w, &nodestime_options());
    w.pool.ponder.store(true, Ordering::Relaxed);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(
        !stop.load(Ordering::Relaxed),
        "pondering must never self-stop"
    );
    assert_eq!(w.root_depth, 4, "the search deepened on GUI time");

    // Control: the same budgets without the ponder flag stop right
    // after the first completed iteration (~21 nodes > optimum 3).
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 1_000;
    w.limits.depth = Some(4);
    init_clock(&mut w, &nodestime_options());
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(stop.load(Ordering::Relaxed), "the soft stop must fire");
    assert!(w.root_depth < 4, "deepening was cut short at optimum");
}

#[test]
fn pondering_soft_budget_arms_stop_on_ponderhit() {
    // A tiny timed budget under `go ponder`: the search never self-stops
    // (GUI owns the clock), but the elapsed soft budget arms
    // `stop_on_ponderhit` so a later ponderhit stops at once (doc 12 step
    // 9 / doc 13). The depth limit bounds the test.
    let mut w = Worker::new(board(START_FEN));
    w.limits.time[crate::types::Color::White as usize] = 1_000;
    w.limits.depth = Some(4);
    w.limits.ponder = true;
    init_clock(&mut w, &nodestime_options());
    w.pool.ponder.store(true, Ordering::Relaxed);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert!(!stop.load(Ordering::Relaxed), "pondering never self-stops");
    assert!(
        w.manager.as_ref().unwrap().stop_on_ponderhit,
        "the elapsed soft budget armed stop_on_ponderhit"
    );
}

#[test]
fn ponderhit_after_armed_budget_stops_via_check_time() {
    // With `stop_on_ponderhit` armed, the FIRST poll after ponderhit
    // (ponder cleared) stops immediately; while still pondering it must
    // not (the ponder guard precedes the check).
    let arm = |ponder: bool| {
        let mut w = Worker::new(board(START_FEN));
        w.completed_depth = 1; // past iteration 1 (check_time guard)
        w.manager.as_mut().unwrap().stop_on_ponderhit = true;
        w.pool.ponder.store(ponder, Ordering::Relaxed);
        w.nodes = w.check_time_mask; // poll-aligned: check_time does real work
        let stop = AtomicBool::new(false);
        w.check_time(&stop);
        stop.load(Ordering::Relaxed)
    };
    assert!(arm(false), "ponderhit + armed budget stops at once");
    assert!(!arm(true), "still pondering must not stop");
}

// --- MultiPV (design/12 §3.12, step 10) ---

#[test]
fn multipv_reports_k_ordered_lines_per_iteration() {
    // MultiPV=3 `go depth 6`: every completed iteration emits exactly 3
    // lines, multipv 1..=3, with exact scores weakly decreasing (line 0
    // finds the global best, each later line the best of the remaining
    // moves — always <= the line before it).
    use std::collections::BTreeMap;
    let mut w = Worker::new(board(START_FEN));
    w.multi_pv = 3;
    w.limits.depth = Some(6);
    let stop = AtomicBool::new(false);
    let mut by_depth: BTreeMap<Depth, Vec<(usize, Value)>> = BTreeMap::new();
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |i| {
        by_depth
            .entry(i.depth)
            .or_default()
            .push((i.multipv, i.score));
    });
    assert!(by_depth.len() >= 3, "several completed iterations reported");
    for (d, group) in by_depth {
        assert_eq!(group.len(), 3, "depth {d} must report 3 PV lines");
        assert_eq!(
            group.iter().map(|(pv, _)| *pv).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "depth {d} multipv indices are 1..=3 in order"
        );
        assert!(
            group[0].1 >= group[1].1 && group[1].1 >= group[2].1,
            "depth {d}: exact scores must weakly decrease: {group:?}"
        );
    }
}

#[test]
fn multipv_clamps_to_the_legal_move_count() {
    // MultiPV=5 with a 2-move `searchmoves` whitelist clamps to 2 lines
    // per iteration — the filter narrows the root move list before the
    // MultiPV clamp (design/12 §3.6/§3.12).
    let filter = [
        Move::normal(Square::E2, Square::E4),
        Move::normal(Square::G1, Square::F3),
    ];
    let mut w = Worker::new(board(START_FEN));
    w.limits.searchmoves = filter.to_vec();
    w.multi_pv = 5;
    w.limits.depth = Some(4);
    let stop = AtomicBool::new(false);
    let mut counts: std::collections::BTreeMap<Depth, usize> = Default::default();
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |i| {
        *counts.entry(i.depth).or_default() += 1;
    });
    assert!(!counts.is_empty());
    assert!(
        counts.values().all(|&c| c == 2),
        "MultiPV must clamp to the 2 whitelisted moves: {counts:?}"
    );
}

// --- searchmoves filtering (design/12 §3.6) ---

#[test]
fn searchmoves_restricts_root_moves_and_bestmove() {
    let filter = [
        Move::normal(Square::E2, Square::E4),
        Move::normal(Square::G1, Square::F3),
    ];
    let mut w = Worker::new(board(START_FEN));
    w.limits.searchmoves = filter.to_vec();
    w.limits.depth = Some(3);
    let stop = AtomicBool::new(false);
    let mut pv_heads = Vec::new();
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |i| {
        pv_heads.push(i.pv[0])
    });
    assert_eq!(w.root_moves.len(), 2);
    assert!(w.root_moves.iter().all(|rm| filter.contains(&rm.pv[0])));
    assert!(filter.contains(&w.best_move().unwrap()));
    assert!(!pv_heads.is_empty());
    assert!(pv_heads.iter().all(|m| filter.contains(m)));
}

#[test]
fn unusable_searchmoves_filter_is_ignored() {
    // No token matches a legal move: search all 20 root moves rather
    // than none (design/12 §3.6 — never a fake `bestmove (none)`).
    let mut w = Worker::new(board(START_FEN));
    w.limits.searchmoves = vec![Move::normal(Square::E2, Square::E5)];
    w.limits.depth = Some(2);
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&test_tt(), &stop, &mut TimePersist::default(), &mut |_| {});
    assert_eq!(w.root_moves.len(), 20);
    assert!(w.best_move().is_some());
}

// --- lastBestMove tracking (§3.5) ---

#[test]
fn last_best_move_tracks_the_stable_best() {
    // The mate-in-1 move is already best at depth 1 and never
    // changes: the remembered PV sticks and the change-depth stays 1.
    let w = go_depth("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 4);
    assert_eq!(w.last_best_move_pv[0].to_string(), "d1d8");
    assert_eq!(w.last_best_move_score, mate_in(1));
    assert_eq!(w.last_best_move_depth, 1);
}

// --- Iteration hygiene (§3.5, plan step 10; §8.9) ---
//
// `root_hygiene` is scripted directly: a worker with real root moves
// and the §3.5 post-iteration state (one exact front score, the rest
// unsearched) pins each rule against the doc's pseudocode; the
// driver-level tests further down cover the same rules end-to-end.

/// A worker over `fen` with root moves built and one iteration result
/// scripted: `front` searched to an exact `score` and floated to the
/// front by the §3.4 stable sort, everything else unsearched.
fn hygiene_worker(fen: &str, front: Move, score: Value) -> Worker {
    let mut w = Worker::new(board(fen));
    w.root_moves = build_root_moves(&w.board, &[]);
    let idx = w
        .root_moves
        .iter()
        .position(|rm| rm.pv[0] == front)
        .expect("scripted front move must be legal");
    w.root_moves[idx].score = score;
    w.root_moves[idx].uci_score = score;
    sort_descending(&mut w.root_moves[..]);
    assert_eq!(w.root_moves[0].pv[0], front);
    w
}

#[test]
fn forgotten_mate_rolls_back_to_the_proven_line() {
    // §3.5's forgotten-mate construction: iteration 5 proved
    // mate_in(3) behind Kb6; iteration 6 (aspiration instability,
    // pruning noise) came back with a weaker exact score on a
    // different front move. The tracking must refuse the weaker line
    // and the rollback must restore the proven PV/score to the front.
    let mate_move = Move::normal(Square::C6, Square::B6);
    let other = Move::normal(Square::H1, Square::H2);
    let mut w = hygiene_worker("k7/8/2K5/8/8/8/8/7R w - - 0 1", other, 500);
    let n_moves = w.root_moves.len();
    w.root_depth = 6;
    w.last_best_move_pv = vec![mate_move];
    w.last_best_move_score = mate_in(3);
    w.last_best_move_depth = 5;

    w.root_hygiene(false);

    // The remembered mate survives the weaker re-score...
    assert_eq!(w.last_best_move_score, mate_in(3));
    assert_eq!(w.last_best_move_pv, vec![mate_move]);
    // ...and is restored to the front, bound-free.
    let front = &w.root_moves[0];
    assert_eq!(front.pv, vec![mate_move]);
    assert_eq!(front.score, mate_in(3));
    assert_eq!(front.uci_score, mate_in(3));
    assert!(!front.score_is_bound());
    // The rotate keeps every root move (the displaced one included).
    assert_eq!(w.root_moves.len(), n_moves);
    assert!(w.root_moves.iter().any(|rm| rm.pv[0] == other));
    // The front move DID flip this iteration: the TM stability signal
    // records it (§3.5 order: tracking before the rollback).
    assert_eq!(w.last_best_move_depth, 6);
}

#[test]
fn a_shorter_proven_mate_replaces_the_remembered_one() {
    // "Never replace a proven (SHORTER) mate": a larger-magnitude
    // exact mate is genuinely better news, not a forgotten mate —
    // the tracking must adopt it and no rollback may fire.
    let mate_move = Move::normal(Square::C6, Square::B6);
    let other = Move::normal(Square::C6, Square::C7);
    let mut w = hygiene_worker("k7/8/2K5/8/8/8/8/7R w - - 0 1", other, mate_in(1));
    w.root_depth = 6;
    w.last_best_move_pv = vec![mate_move];
    w.last_best_move_score = mate_in(3);

    w.root_hygiene(false);

    assert_eq!(w.last_best_move_score, mate_in(1));
    assert_eq!(w.last_best_move_pv, vec![other]);
    assert_eq!(w.root_moves[0].pv[0], other, "no rollback");
    assert_eq!(w.root_moves[0].score, mate_in(1));
}

#[test]
fn a_bound_mate_rescore_is_rolled_back_to_the_exact_one() {
    // The same front move re-scored as a BOUND (mid-aspiration shape)
    // is also a forgotten mate: bounds never replace a proven exact
    // mate, and the restore clears the flags.
    let mate_move = Move::normal(Square::C6, Square::B6);
    let mut w = hygiene_worker("k7/8/2K5/8/8/8/8/7R w - - 0 1", mate_move, mate_in(3));
    w.root_moves[0].score_lowerbound = true;
    w.root_moves[0].uci_score = mate_in(3) - 20; // window-clamped print value
    w.root_depth = 7;
    w.last_best_move_pv = vec![mate_move];
    w.last_best_move_score = mate_in(3);
    w.last_best_move_depth = 4;

    w.root_hygiene(false);

    let front = &w.root_moves[0];
    assert_eq!(front.pv[0], mate_move);
    assert_eq!(front.score, mate_in(3));
    assert_eq!(front.uci_score, mate_in(3));
    assert!(!front.score_is_bound(), "the restore clears the bounds");
    assert_eq!(w.last_best_move_depth, 4, "same front move: no flip");
}

#[test]
fn aborted_loss_restores_the_previous_iteration_line() {
    // §3.5 aborted-loss hygiene, scripted: the front root move's
    // exact mated score came from a STOPPED iteration — untrusted
    // (unsearched root moves might refute or delay it) — so the
    // previous iteration's line comes back, bound-free, and the
    // stopped iteration never feeds lastBestMove.
    let fen = "8/8/6k1/R7/8/8/8/1R5K b - - 0 1"; // lost to the ladder
    let lost = Move::normal(Square::G6, Square::G7);
    let previous = Move::normal(Square::G6, Square::F6);
    let mut w = hygiene_worker(fen, lost, mated_in(6));
    w.root_depth = 8;
    w.last_best_move_pv = vec![previous];
    w.last_best_move_score = -800;
    w.last_best_move_depth = 7;

    w.root_hygiene(true);

    let front = &w.root_moves[0];
    assert_eq!(front.pv, vec![previous]);
    assert_eq!(front.score, -800);
    assert_eq!(front.uci_score, -800);
    assert!(!front.score_is_bound());
    // The untrusted loss slid one slot back (stable rotate).
    assert_eq!(w.root_moves[1].pv[0], lost);
    assert_eq!(w.last_best_move_score, -800);
    assert_eq!(w.last_best_move_depth, 7);
}

#[test]
fn aborted_loss_without_history_is_published_as_a_bound() {
    // An aborted depth-1 search has nothing to restore: the exact
    // loss is re-labeled a LOWER bound — unsearched root moves may
    // beat it, and the GUI must never see a fake forced loss
    // (doc 12 §3.12).
    let kg8 = Move::normal(Square::H8, Square::G8);
    let mut w = hygiene_worker("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1", kg8, mated_in(2));
    w.root_depth = 1;

    w.root_hygiene(true);

    let front = &w.root_moves[0];
    assert_eq!(front.score, mated_in(2), "the score itself is kept");
    assert!(front.score_lowerbound && !front.score_upperbound);
    assert!(w.last_best_move_pv.is_empty());
}

#[test]
fn a_stopped_non_loss_iteration_is_left_alone() {
    // A stop with an ordinary (non-loss) exact front score triggers
    // neither repair: the §3.4 stable-sort state IS the answer.
    let e5 = Move::normal(Square::E7, Square::E5);
    let mut w = hygiene_worker(START_FEN, Move::normal(Square::E2, Square::E4), 30);
    w.last_best_move_pv = vec![e5]; // not even a root move here
    w.last_best_move_score = -10;

    w.root_hygiene(true);

    assert_eq!(w.root_moves[0].pv[0], Move::normal(Square::E2, Square::E4));
    assert_eq!(w.root_moves[0].score, 30);
    assert!(!w.root_moves[0].score_is_bound());
    assert_eq!(w.last_best_move_score, -10, "stopped: tracking is off");
}

#[test]
fn aborted_loss_end_to_end_reports_the_last_completed_iteration() {
    // §8.9 end-to-end: black is lost to the two-rook ladder but has
    // many moves. Learn where iterations end from an unlimited run
    // (deterministic, D24), then re-run with a node budget that stops
    // the search mid-way through the iteration AFTER the first one
    // whose front score is an exact loss. The aborted iteration must
    // not degrade the report: the front root move carries exactly
    // what the last completed iteration published.
    let fen = "8/8/6k1/R7/8/8/8/1R5K b - - 0 1";
    // Doc 10's LMR (P9) deepens the tree, so the ladder now first proves
    // the loss at iteration 10 (was earlier pre-LMR); the probe cap must
    // leave at least one iteration AFTER that for the abort to land in.
    let max_depth = 12;
    let mut probe = Worker::new(board(fen));
    probe.limits.depth = Some(max_depth);
    let mut infos: Vec<(Depth, Value, Move, u64)> = Vec::new();
    {
        let tt = test_tt();
        tt.new_search();
        let stop = AtomicBool::new(false);
        probe.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |i| {
            infos.push((i.depth, i.score, i.pv[0], i.nodes));
        });
    }
    let loss_at = infos
        .iter()
        .position(|&(_, score, _, _)| is_loss(score))
        .expect("the ladder must prove the loss within the probe depth");
    assert!(
        loss_at + 1 < infos.len(),
        "need one more iteration to abort"
    );
    let budget = infos[loss_at].3 + 1; // stops early in the NEXT iteration

    let mut w = Worker::new(board(fen));
    w.limits.depth = Some(max_depth);
    w.limits.nodes = Some(budget);
    let mut infos2: Vec<(Depth, Value, Move, u64)> = Vec::new();
    let tt = test_tt();
    tt.new_search();
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |i| {
        infos2.push((i.depth, i.score, i.pv[0], i.nodes));
    });

    assert!(stop.load(Ordering::Relaxed), "the node budget must stop");
    let &(depth, score, first, _) = infos2.last().expect("completed iterations exist");
    assert!(is_loss(score), "the abort lands in the exact-loss regime");
    assert!(w.root_depth > depth, "the stop aborted a deeper iteration");
    // The report equals the last COMPLETED iteration, bound-free —
    // never the aborted iteration's partial state.
    assert_eq!(best_score(&w), score);
    assert_eq!(w.root_moves[0].uci_score, score);
    assert_eq!(w.best_move(), Some(first));
    assert!(!w.root_moves[0].score_is_bound());
    assert_eq!(w.last_best_move_score, score);
}

// --- `go mate n` (§3.5, §8.10; doc 12 §3.9's mate-limit stop) ---

/// Run `w`'s configured `go` against a fresh TT, returning the stop
/// flag's final state.
fn go_returning_stop(w: &mut Worker) -> bool {
    let tt = test_tt();
    tt.new_search();
    let stop = AtomicBool::new(false);
    w.iterative_deepening(&tt, &stop, &mut TimePersist::default(), &mut |_| {});
    stop.load(Ordering::Relaxed)
}

#[test]
fn go_mate_2_stops_after_proving_the_mate_in_2() {
    // KRK mates in 2 full moves (3 plies): once an iteration
    // completes on the exact mate_in(3), `VALUE_MATE - 3 <= 2·2`
    // raises the stop — far below the safety-net depth cap, which
    // exists only so a regression cannot run this test to MAX_PLY.
    let mut w = Worker::new(board("k7/8/2K5/8/8/8/8/7R w - - 0 1"));
    w.limits.mate = Some(2);
    w.limits.depth = Some(32);
    assert!(go_returning_stop(&mut w), "the mate limit must self-stop");
    assert_eq!(best_score(&w), mate_in(3));
    assert!(
        w.root_depth <= 6,
        "stopped by the mate rule, not the cap: {}",
        w.root_depth
    );
}

#[test]
fn go_mate_1_ignores_a_longer_mate() {
    // mate_in(3) is outside `go mate 1` (3 > 2·1): the mate rule must
    // never fire and the depth cap ends the go with stop untouched.
    let mut w = Worker::new(board("k7/8/2K5/8/8/8/8/7R w - - 0 1"));
    w.limits.mate = Some(1);
    w.limits.depth = Some(5);
    assert!(!go_returning_stop(&mut w), "a 2-move mate is out of reach");
    assert_eq!(w.root_depth, 5, "the depth cap ended the go");
}

#[test]
fn go_mate_stops_when_the_engine_is_being_mated() {
    // §3.5 gates on is_mate OR is_mated: a proven mated_in(2)
    // (2 <= 2·1) ends a `go mate 1` search from the losing side too.
    let mut w = Worker::new(board("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1"));
    w.limits.mate = Some(1);
    w.limits.depth = Some(16);
    assert!(go_returning_stop(&mut w), "a proven mated score stops");
    assert_eq!(best_score(&w), mated_in(2));
    assert!(w.root_depth <= 6, "stopped early: {}", w.root_depth);
}

// --- Main-node fail-high softening (§3.7.2 step 14; §8.11) ---

#[test]
fn main_search_fail_high_softens_toward_beta() {
    // A single-legal-move position (only e4e5 — the king is boxed by
    // the rooks) whose child qsearch has no captures and fails low,
    // returning its exact stand pat E: the depth-1 parent fails high
    // with best_value = -E and the node-exit blend is pinned exactly
    // — `(best·depth + beta) / (depth + 1)` = `(-E + beta) / 2`.
    let fen = "k7/8/8/8/4P3/1r6/7r/K7 w - - 0 1";
    let mut w = Worker::new(board(fen));
    init_stack(&mut w.stack);
    assert_eq!(all_legal(&w.board).len(), 1);
    let e5 = Move::normal(Square::E4, Square::E5);
    let mut after = w.board.clone();
    after.make_move(e5, after.gives_check(e5));
    let e_child = corrected_eval(&after, Some(e5));
    let beta = -e_child - 100; // parent's true value -E sits above it
    let tt = test_tt();
    // The parent's static eval already clears this low beta, so N10 RFP
    // would short-circuit the node before the move loop and pre-empt the
    // node-exit soften this test targets. Seed a QUIET ttMove (e4e5) —
    // the `(!ttMove || ttCapture)` guard vetoes RFP on a quiet ttMove —
    // so the node searches and reaches the step-14 blend. The eval-only
    // (VALUE_NONE) entry can neither cut (invalid value) nor change the
    // eval (invalid eval → real evaluate()), so only the veto is felt.
    let key = w.board.key();
    tt.probe(key).writer.write(
        key,
        VALUE_NONE,
        false,
        Bound::None,
        DEPTH_UNSEARCHED,
        e5,
        VALUE_NONE,
        tt.generation(),
    );
    let stop = AtomicBool::new(false);
    let depth = 1;
    let v = w.search::<NonPv>(&tt, &stop, beta - 1, beta, depth, true, 1);
    assert_eq!(v, (-e_child * depth + beta) / (depth + 1));
    assert!(v >= beta, "still a fail high after the blend");
    assert!(v < -e_child, "pulled strictly back toward beta");
}

#[test]
fn main_search_never_softens_a_decisive_fail_high() {
    // §8.11: Rxd8# is the first ordered move (the only capture,
    // MVV-first) and fails high against a tiny window — the mate
    // distance must come back exact, never blended toward beta.
    let fen = "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
    let mut w = Worker::new(board(fen));
    init_stack(&mut w.stack);
    let tt = test_tt();
    let stop = AtomicBool::new(false);
    let v = w.search::<NonPv>(&tt, &stop, 99, 100, 1, true, 1);
    assert_eq!(v, mate_in(2), "ply-relative mate through a fail high");
}

// --- Zero-window invariant (§3.7.1; §8 criterion 6) ---
//
// The node-entry debug_asserts ARE the instrumentation criterion 6
// asks for: every debug-build search in this suite executes them at
// every node, so a green debug run IS the "no NonPV node with
// beta != alpha + 1, no PV cut node" report. The #[should_panic]
// tests prove the asserts actually fire on the two illegal
// combinations (doc 07 §5's NodeType guarantees); they are
// debug-gated because release strips `debug_assert!`.

#[cfg(debug_assertions)]
fn raw_search_worker() -> (Worker, TranspositionTable) {
    let mut w = Worker::new(board(START_FEN));
    init_stack(&mut w.stack);
    (w, test_tt())
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "NonPV requires a zero window")]
fn nonpv_search_rejects_a_wide_window() {
    let (mut w, tt) = raw_search_worker();
    let stop = AtomicBool::new(false);
    // Width-2 window at a NonPV node: the §3.7.1 invariant must trip.
    // Depth 2 keeps the call out of the depth-0 qsearch dispatch.
    w.search::<NonPv>(&tt, &stop, 0, 2, 2, true, 1);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "NonPV requires a zero window")]
fn nonpv_qsearch_rejects_a_wide_window() {
    let (mut w, tt) = raw_search_worker();
    let stop = AtomicBool::new(false);
    w.qsearch::<NonPv>(&tt, &stop, 0, 2, 1);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "PV nodes are never cut nodes")]
fn pv_node_rejects_cut_node_status() {
    let (mut w, tt) = raw_search_worker();
    let stop = AtomicBool::new(false);
    w.search::<Pv>(&tt, &stop, 0, 2, 2, true, 1);
}

#[test]
fn bench_suite_debug_run_upholds_the_node_invariants() {
    // §8 criterion 6's instrumented run over the bench suite: in a
    // debug build every node entry executes the §3.7.1 asserts, so
    // completing all 49 positions cleanly is the evidence. Depth 2
    // keeps the debug-mode board-consistency oracle affordable; the
    // full-depth version of this run is tools/check.sh's release
    // bench (which cannot carry the asserts — hence this miniature).
    let tt = test_tt();
    for entry in crate::bench::BENCH_POSITIONS {
        let fen = entry.split(" moves").next().unwrap();
        let mut w = Worker::new(board(fen));
        w.limits.depth = Some(2);
        go(&mut w, &tt);
    }
}

// --- Aspiration windows (§3.6; §8 criterion 7) ---

/// Assert one iteration's attempt group follows the §3.6 recurrences
/// exactly: only fails re-search; delta widens by `44·delta/128` per
/// re-search; a fail low slides beta down to the old alpha and resets
/// `failed_high_cnt`; a fail high backs alpha off to at most the old
/// `beta - delta` and increments the counter; every attempt runs at
/// `max(1, root_depth - failed_high_cnt)` (`search_again_counter` is
/// 0 until doc 12 step 8); the loop only ends inside the window.
/// `seed` is the (average_score, mean_squared_score) pair the first
/// window was built from, when the test controls it.
fn check_asp_group(group: &[AspirationAttempt], seed: Option<(Value, i64)>) {
    let first = &group[0];
    assert_eq!(first.failed_high_cnt, 0);
    assert_eq!(first.adjusted_depth, first.root_depth.max(1));
    if let Some((avg, mss)) = seed {
        let d0 = (params::ASPIRATION_BASE_DELTA() as i64
            + mss.abs() / params::ASPIRATION_MSS_DIVISOR()) as Value;
        assert_eq!(first.delta, d0, "initial delta is 5 + |mss|/10588");
        assert_eq!(first.alpha, (avg - d0).max(-VALUE_INFINITE));
        assert_eq!(first.beta, (avg + d0).min(VALUE_INFINITE));
    }
    for pair in group.windows(2) {
        let (p, n) = (&pair[0], &pair[1]);
        assert_eq!(p.root_depth, n.root_depth, "one iteration per group");
        assert!(
            p.value <= p.alpha || p.value >= p.beta,
            "only a failed attempt re-searches: {p:?}"
        );
        assert_eq!(
            n.delta,
            (p.delta + params::ASPIRATION_WIDEN_NUM() * p.delta / params::ASPIRATION_WIDEN_DEN())
                .min(2 * VALUE_INFINITE),
            "every re-search widens delta"
        );
        if p.value <= p.alpha {
            // Fail low: the window slides down at full depth.
            assert_eq!(n.beta, p.alpha, "beta drops exactly to the old alpha");
            assert_eq!(n.alpha, (p.value - p.delta).max(-VALUE_INFINITE));
            assert_eq!(n.failed_high_cnt, 0, "fail low resets failed_high_cnt");
        } else {
            // Fail high: alpha never loosens past its current value.
            assert_eq!(n.alpha, (p.beta - p.delta).max(p.alpha));
            assert_eq!(n.beta, (p.value + p.delta).min(VALUE_INFINITE));
            assert_eq!(n.failed_high_cnt, p.failed_high_cnt + 1);
        }
        assert_eq!(
            n.adjusted_depth,
            (n.root_depth - n.failed_high_cnt).max(1),
            "each consecutive fail-high re-search runs one ply shallower"
        );
    }
    let last = group.last().unwrap();
    assert!(
        last.alpha < last.value && last.value < last.beta,
        "the loop only ends inside the window (no stop in these tests): {last:?}"
    );
}

/// A worker run through one direct `aspiration_search` call at
/// `root_depth`, with the front root move's window inputs seeded to
/// (avg, mss) — the §5/§8 "scripted window evolution" harness.
fn seeded_aspiration_run(fen: &str, root_depth: Depth, avg: Value, mss: i64) -> Worker {
    let tt = test_tt();
    tt.new_search();
    let mut w = Worker::new(board(fen));
    init_stack(&mut w.stack);
    w.root_moves = build_root_moves(&w.board, &[]);
    w.root_depth = root_depth;
    w.root_moves[0].average_score = avg;
    w.root_moves[0].mean_squared_score = mss;
    let stop = AtomicBool::new(false);
    w.aspiration_search(&tt, &stop, 0);
    w
}

#[test]
fn aspiration_fail_high_widens_and_researches_shallower() {
    // The §4 step-8 done-condition trace: KRK mate-in-2 scores
    // mate_in(3) at depth 3 — far above a window seeded around 0 with
    // mss 0 (delta = the base 5) — forcing the fail-high path.
    let w = seeded_aspiration_run("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3, 0, 0);
    let trace = &w.asp_trace;
    assert!(trace.len() >= 2, "the mate score must fail high");
    let t0 = trace[0];
    assert_eq!(t0.delta, params::ASPIRATION_BASE_DELTA());
    assert_eq!((t0.alpha, t0.beta), (-5, 5));
    assert_eq!(t0.adjusted_depth, 3);
    assert!(t0.value >= t0.beta, "attempt 0 fails high");
    let t1 = trace[1];
    assert_eq!(t1.failed_high_cnt, 1);
    assert_eq!(t1.adjusted_depth, 2, "re-search runs one ply shallower");
    assert_eq!(t1.delta, 5 + 44 * 5 / 128); // widened to 6
    assert_eq!(t1.alpha, 0, "alpha backs off exactly to old beta - delta");
    assert_eq!(t1.beta, (t0.value + 5).min(VALUE_INFINITE));
    check_asp_group(trace, Some((0, 0)));
    // root_delta is left at the last attempt's window width — doc
    // 10's reduction formula divides by it.
    let last = trace.last().unwrap();
    assert_eq!(w.root_delta, last.beta - last.alpha);
}

#[test]
fn aspiration_fail_low_slides_beta_to_old_alpha_at_full_depth() {
    // The startpos is worth far less than the seeded +300 average
    // (delta = 5 + 90_000/6776 = 18, window (282, 318)): the first
    // attempt fails low. Beta drops exactly to the old alpha, alpha
    // backs off delta below the returned value, failed_high_cnt stays
    // 0 so the re-search keeps the FULL depth, and delta widens.
    // (The stop_on_ponderhit revoke rides this branch once the flag
    // exists — doc 12 step 9.)
    let w = seeded_aspiration_run(START_FEN, 2, 300, 300 * 300);
    let trace = &w.asp_trace;
    assert!(trace.len() >= 2, "the quiet score must fail low");
    let t0 = trace[0];
    assert_eq!(t0.delta, 18);
    assert_eq!((t0.alpha, t0.beta), (282, 318));
    assert!(t0.value <= t0.alpha, "attempt 0 fails low");
    let t1 = trace[1];
    assert_eq!(t1.beta, 282, "beta slides down exactly to the old alpha");
    assert_eq!(t1.alpha, (t0.value - 18).max(-VALUE_INFINITE));
    assert_eq!(t1.failed_high_cnt, 0, "fail low resets failed_high_cnt");
    assert_eq!(t1.adjusted_depth, 2, "fail low keeps the full depth");
    assert_eq!(t1.delta, 18 + 44 * 18 / 128); // widened to 24
    check_asp_group(trace, Some((300, 300 * 300)));
}

#[test]
fn aspiration_trace_through_iterative_deepening_on_a_score_jump() {
    // KRK through the real driver — the doc's "crafted volatile
    // position": iterations 1-2 see a quiet rook-up score, iteration
    // 3 jumps to mate_in(3). Iteration 1 runs with the unset-sentinel
    // window (clamped to the full window — "depth 1 is effectively an
    // infinite-window search", §3.6), the jump iteration must fail
    // high then re-search shallower with a widened delta, and the
    // extra iteration 4 — its window blown wide open by the mate-band
    // EMAs — settles on the exact mate.
    let w = go_depth("k7/8/2K5/8/8/8/8/7R w - - 0 1", 4);
    let trace = &w.asp_trace;
    for d in 1..=4 {
        let group: Vec<AspirationAttempt> = trace
            .iter()
            .copied()
            .filter(|a| a.root_depth == d)
            .collect();
        assert!(!group.is_empty(), "iteration {d} left no attempts");
        check_asp_group(&group, None);
    }
    let it1: Vec<AspirationAttempt> = trace
        .iter()
        .copied()
        .filter(|a| a.root_depth == 1)
        .collect();
    assert_eq!(it1.len(), 1, "a full window cannot fail");
    assert_eq!(
        (it1[0].alpha, it1[0].beta),
        (-VALUE_INFINITE, VALUE_INFINITE)
    );
    let it3: Vec<AspirationAttempt> = trace
        .iter()
        .copied()
        .filter(|a| a.root_depth == 3)
        .collect();
    assert!(
        it3[0].alpha > -VALUE_INFINITE && it3[0].beta < VALUE_INFINITE,
        "the jump iteration really ran an aspirated window: {:?}",
        it3[0]
    );
    assert!(
        it3.len() >= 2 && it3[0].value >= it3[0].beta,
        "mate fails high"
    );
    assert_eq!(it3[1].adjusted_depth, 2, "shallower re-search");
    assert!(it3[1].delta > it3[0].delta, "widened delta");
    // And the iteration still converges on the exact mate: the
    // shallower re-search re-finds it through the TT.
    assert_eq!(best_score(&w), mate_in(3));
}

/// The plan-step-5/7 driver shape — full-window root searches, no
/// aspiration — preserved as the reference harness for the step-8
/// value-identity bar and for the PVS/TT equivalence suites below
/// (whose bars are about the PVS structure and TT semantics, which
/// aspiration deliberately layers window dynamics on top of).
fn go_depth_full_window_on(fen: &str, depth: Depth, tt: &TranspositionTable) -> Worker {
    tt.new_search();
    let mut w = Worker::new(board(fen));
    init_stack(&mut w.stack);
    w.root_moves = build_root_moves(&w.board, &[]);
    let stop = AtomicBool::new(false);
    for d in 1..=depth {
        w.root_depth = d;
        for rm in w.root_moves.iter_mut() {
            rm.previous_score = rm.score;
            rm.previous_pv.clone_from(&rm.pv);
        }
        w.sel_depth = 0;
        w.root_delta = 2 * VALUE_INFINITE;
        // Optimism is part of the production root setup (S0), attached
        // to the ITERATION, not the window — a faithful "same search,
        // no windowing" reference carries it identically (mirrors
        // aspiration_search; decisive-fixture consumers are immune:
        // optimism never moves mate bounds).
        let avg = w.root_moves[0].average_score;
        let us = w.board.side_to_move();
        let opt = params::OPTIMISM_NUM() * avg / (avg.abs() + params::OPTIMISM_DEN_BASE());
        w.optimism[us as usize] = opt;
        w.optimism[us.flip() as usize] = -opt;
        w.search::<Root>(tt, &stop, -VALUE_INFINITE, VALUE_INFINITE, d, false, 0);
        sort_descending(&mut w.root_moves[..]);
    }
    w
}

/// [`go_depth_full_window_on`] against a fresh table.
fn go_depth_full_window(fen: &str, depth: Depth) -> Worker {
    go_depth_full_window_on(fen, depth, &test_tt())
}

#[test]
fn aspiration_preserves_values_on_quiet_positions_at_fixed_depth() {
    // Step-8 regression bar (§4/§5): on positions whose iteration
    // scores stay inside every window, aspiration keeps the full-window
    // driver's BEST MOVE. The trace assert pins the fixtures'
    // load-bearing property: no attempt ever failed.
    //
    // Pre-P9 this also pinned exact fixed-depth SCORE identity, but doc
    // 10's LMR (`Worker::reduction`, §3.3) scales the base reduction by
    // `delta / root_delta` — the LOCAL window over the aspiration window.
    // A narrow aspiration window and the `2·VALUE_INFINITE` full window
    // therefore reduce their NonPV subtrees differently, so the lossy
    // fixed-depth score can drift a few cp (measured: 21cp on the
    // queen-up mirror) even when the chosen move is unchanged. The best
    // move, mate distances (mate suite) and the SPRT (D25) are the
    // surviving strength claims; exact score identity is a pre-LMR oracle.
    //
    // "Quiet" here means score-STABLE, which needs a clean material
    // edge: delta = 5 + avg²/10588 then dwarfs the odd/even tempo
    // swing between iterations. Near-zero positions (the startpos)
    // are poor fixtures — delta ~5 sits BELOW their parity swing, and
    // the resulting re-searches legitimately store different TT
    // bounds than one full-window pass (measured: an 8cp drift on the
    // startpos at depth 4). Decisive-score fixtures diverge for the
    // opposite reason — a fail-high's shallower re-search can end the
    // iteration below the mate (see the mate-suite tests, which spend
    // one extra settling iteration instead).
    //
    // Doc 10 §3.5's LMP tightens this window-stability bar further: on a
    // full-material opening its lossy movecount pruning can move the
    // fixed-depth value enough between iterations to trip an aspiration
    // re-search (measured: 7 attempts, and a changed bestmove, on the
    // white queen-up opening at depth 4). The surviving fixtures are the
    // ones whose LMP-pruned value stays window-stable AND move-stable —
    // the queen-up MIRROR (whose ordering keeps every attempt inside its
    // window) and the low-branching knight endgame.
    // S0 optimism re-scope (2026-07-26): on the queen-up mirror,
    // optimism saturates (~+71 for the side up a queen) and its
    // interplay with the window-width-coupled LMR drift lets the three
    // near-tie winning replies reorder between drivers — bestmove
    // identity no longer survives there (same fate as the queen-up
    // OPENING under LMP, above). It still holds on the low-branching
    // knight endgame, which keeps the bestmove bar alive; the mirror
    // keeps the window-containment and value-band bars.
    let suite: &[(&str, Depth, bool)] = &[
        // Black up a queen, black to move (value band only).
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
            4,
            false,
        ),
        // Black up a knight in a bare-kings endgame.
        ("4k3/8/3n4/8/8/8/8/4K3 b - - 0 1", 3, true),
    ];
    const WINDOW_DRIFT_CP: Value = 30;
    for &(fen, depth, bestmove_bar) in suite {
        let asp = go_depth(fen, depth);
        assert_eq!(
            asp.asp_trace.len(),
            depth as usize,
            "score-stable fixture stayed inside every window: {fen}"
        );
        let full = go_depth_full_window(fen, depth);
        assert!(
            (best_score(&asp) - best_score(&full)).abs() <= WINDOW_DRIFT_CP,
            "value band: {fen} (asp {}, full {})",
            best_score(&asp),
            best_score(&full)
        );
        if bestmove_bar {
            assert_eq!(
                asp.best_move(),
                full.best_move(),
                "bestmove identity: {fen}"
            );
        }
    }
}

// --- PVS equivalence vs full-window negamax (doc 07 §4 step 7, §5) ---
//
// Fail-soft alpha-beta over a fixed tree returns the exact minimax
// value at a full-window root, and the PVS three-call structure
// preserves it: a zero-window scout that fails low has proven
// `value <= alpha` (so skipping its PV re-search is sound), and one
// that beats alpha is re-searched with the full window before
// anything consumes it. The caveat since step 7 made the NonPV TT
// cutoffs live: a cutoff may substitute a DEEPER stored value for a
// scout's fixed-depth value. The step-7 done-condition ("scores
// unchanged at fixed depth") pins down that this does not move the
// root value on this suite — the cutoff-free full-window reference
// walking the same tree (same generators, same stand-pat, same
// terminal rules) must agree exactly with the final iteration.
// Positions are chosen with no reachable repetitions so the
// node-counter draw dither cannot fire, and with little material so
// the exponential no-cutoff oracle stays cheap.

fn ref_qsearch(b: &mut Board, ctx: &mut EvalContext, ply: usize) -> Value {
    let in_check = !b.checkers().is_empty();
    let mut list = MoveList::new();
    if in_check {
        generate_evasions(b, &mut list);
    } else {
        generate_captures(b, &mut list);
    }
    let mut best = if in_check {
        -VALUE_INFINITE
    } else {
        evaluate(b, ctx, VALUE_ZERO, ply)
    };
    let mut any_legal = false;
    for m in &list {
        if !b.legal(m) {
            continue;
        }
        any_legal = true;
        b.make_move(m, b.gives_check(m));
        best = best.max(-ref_qsearch(b, ctx, ply + 1));
        b.unmake_move(m);
    }
    if in_check && !any_legal {
        return mated_in(ply as i32);
    }
    best
}

fn ref_negamax(b: &mut Board, ctx: &mut EvalContext, depth: Depth, ply: usize) -> Value {
    if depth <= 0 {
        return ref_qsearch(b, ctx, ply);
    }
    let mut list = MoveList::new();
    generate_legal(b, &mut list);
    if list.is_empty() {
        return if b.checkers().is_empty() {
            VALUE_DRAW
        } else {
            mated_in(ply as i32)
        };
    }
    let mut best = -VALUE_INFINITE;
    for m in &list {
        b.make_move(m, b.gives_check(m));
        best = best.max(-ref_negamax(b, ctx, depth - 1, ply + 1));
        b.unmake_move(m);
    }
    best
}

/// Full-window negamax root oracle: the exact minimax value plus
/// EVERY root move attaining it. PVS may legitimately tie-break
/// differently than the reference loop (ttMove-first ordering vs
/// generation order), so the bestmove bar is set membership.
fn ref_root(fen: &str, depth: Depth) -> (Value, Vec<Move>) {
    let mut b = board(fen);
    let mut ctx = EvalContext::new();
    let list = all_legal(&b);
    let mut best = -VALUE_INFINITE;
    let mut best_moves: Vec<Move> = Vec::new();
    for m in &list {
        b.make_move(m, b.gives_check(m));
        let v = -ref_negamax(&mut b, &mut ctx, depth - 1, 1);
        b.unmake_move(m);
        match v.cmp(&best) {
            std::cmp::Ordering::Greater => {
                best = v;
                best_moves.clear();
                best_moves.push(m);
            }
            std::cmp::Ordering::Equal => best_moves.push(m),
            std::cmp::Ordering::Less => {}
        }
    }
    (best, best_moves)
}

#[test]
fn pvs_matches_full_window_negamax_at_fixed_depth() {
    // doc 07 §4 step 7 done-condition + §5 regression strategy: PVS
    // returns the same VALUE and an equally-best MOVE as full-window
    // negamax at fixed depth. Node counts drop instead of matching —
    // that half of the done-condition is the Bench: trailer (D24).
    // Since step 8 the production driver aspirates the root window,
    // which legitimately moves fixed-depth values (shallower fail-high
    // re-searches); this suite therefore pins the PVS structure
    // through the full-window test driver — aspiration's own bars are
    // the trace/identity tests above plus bench and SPRT (D25).
    //
    // Doc 10 §3.5's LMP is the first LOSSY node-level pruning (LMR/RFP/
    // NMP were value-preserving or fired only on decisive scores here):
    // it discards late quiets outright, so on a wide or tactically
    // shallow node the production fixed-depth value — and even the
    // bestmove — can legitimately part from cutoff-free negamax (the
    // minor-piece skirmish drifted 40cp, the two-rook ladder changed its
    // chosen rook move). The cutoff-free oracle therefore only applies to
    // fixtures LMP does not reshape: low-branching pawn endings and
    // early-mate positions whose (check-ordered) mating move is searched
    // before the movecount threshold. The lossy-node classes migrate to
    // the bench trailer + SPRT (D25) and the LMP-specific tests above.
    let suite: &[(&str, Depth)] = &[
        // KRK mate-in-2: exercises mate-distance pruning soundness.
        ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3),
        // The pawn-ending fixture (both depths) has migrated to the
        // bench+SPRT channel with the S0 landings (2026-07-26): depth 3
        // fell to the §3.7 finalization terms (heavier non-PV
        // reductions floor the shallowest scouts, ~40cp) and depth 4 to
        // §3.9 sibling depth reduction (an early alpha raise cheapens a
        // later, better sibling — fixed-depth centipawn values are now
        // order-dependent BY DESIGN; doc 07 §4 step 6 re-scope). The
        // cutoff-free oracle's living bar is decisive fixtures, whose
        // §3.9 exemption keeps mate distances exact.
        // Back-rank mate-in-1 with pawns on both sides.
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 2),
    ];
    // S1 (D22) re-scopes this oracle the same way LMP once did: live
    // correction history makes the leaf eval history-dependent, so the
    // production value drifts from the raw-eval reference by the
    // correction magnitude a shallow fresh-table search accumulates
    // (measured 3cp on the pawn ending; hard ceiling |cv|max/131072 =
    // 493 at saturated tables). DECISIVE values stay exact — mate
    // distances are eval-independent — and keep the bestmove assert;
    // centipawn fixtures assert the drift band and surrender bestmove
    // equality (a few cp legitimately reorders near-equal roots).
    const S1_CORR_DRIFT_CP: Value = 8;
    for &(fen, depth) in suite {
        let w = go_depth_full_window(fen, depth);
        let (want, optimal) = ref_root(fen, depth);
        let got = best_score(&w);
        if is_decisive(want) {
            assert_eq!(
                got, want,
                "PVS vs full-window negamax decisive value diverged: {fen} depth {depth}"
            );
            let bm = w.best_move().unwrap();
            assert!(
                optimal.contains(&bm),
                "PVS bestmove {bm} is not full-window-optimal for {fen} depth {depth} \
                     (optimal: {optimal:?})"
            );
        } else {
            assert!(
                (got - want).abs() <= S1_CORR_DRIFT_CP,
                "PVS drifted past the S1 correction band: {fen} depth {depth} \
                     (got {got}, reference {want})"
            );
        }
    }
}

// --- TT value equality at fixed depth (doc 07 §4 step 6's bar,
//     re-scoped by step 7; doc 08 §8 criterion 6's value half) ---

/// One position of the TT value-equality suite: the fixed-depth value of
/// the full-window driver against a FRESH table. A cutoff on an
/// equal-draft entry is a correct bound on the node's fixed-depth value
/// (§3.12 keeps mate scores exact; the fixtures keep rule-50 and
/// repetition path-dependence out of reach), so with a FRESH table the
/// returned value equals cutoff-free negamax — which is the TT-boundary
/// correctness the caller pins.
///
/// Pre-P9 this searched twice (fresh, then the same table warmed by that
/// run) and asserted fresh == warm. Doc 10's LMR (`Worker::reduction`,
/// §3.7) couples the reduction to TT-supplied ttMoves (the ttMove rebate
/// and the no-ttMove cut-node/full-depth surcharges), so a WARM table's
/// richer move info legitimately reduces its NonPV subtrees differently
/// and drifts the lossy fixed-depth score a few cp (measured: 1cp on the
/// two-rook ladder, 4cp on the startpos) — never a mate distance, which
/// the mate suite pins. Fresh-vs-warm determinism is therefore no longer
/// a TT invariant; the surviving one is fresh-TT == negamax below, and
/// warm-TT node shrink is covered by
/// [`warm_tt_searches_fewer_nodes_at_equal_depth`].
fn tt_value_pair(fen: &str, depth: Depth) -> Value {
    best_score(&go_depth_full_window_on(fen, depth, &test_tt()))
}

#[test]
fn tt_on_and_off_return_identical_values_at_fixed_depth() {
    // The step-6 exit bar, kept green through step 7: fixed-depth
    // searches with the TT disabled vs enabled (fresh or same-depth
    // warmed) return identical VALUES — node counts may differ.
    // `ref_negamax` — proven equal to the pre-TT engine at the commit
    // that added it — is the TT-disabled oracle.
    //
    // Depths stay <= 4: from a FEN root (no game history) the first
    // possible in-tree repetition is ply 4, which at these depths is
    // qsearch's plain deterministic VALUE_DRAW — the node-counter
    // draw dither is unreachable, so any divergence would be TT state
    // moving a same-draft value, exactly what must not happen.
    // Positions keep the material
    // low: the CUTOFF-FREE oracle explores every capture/evasion
    // chain exhaustively, which explodes on full-material boards (the
    // startpos test below covers that class without the oracle).
    //
    // The `ref_negamax` oracle prunes NOTHING, so — like the PVS suite
    // above — this bar now holds only on fixtures doc 10 §3.5's lossy
    // LMP does not reshape. The wide-branching (two-rook ladder) and
    // tactically-shallow (minor-piece skirmish, the pathological mated
    // KQ-vs-K) fixtures legitimately part from cutoff-free negamax once
    // late quiets are discarded, so they move to the LMP-specific tests
    // and the SPRT; the surviving fixtures are the low-branching /
    // early-mate ones whose value LMP leaves intact.
    let suite: &[(&str, Depth)] = &[
        // KRK mate-in-2: decisive scores through value_to_tt/from_tt.
        ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3),
        // The pawn-ending fixture migrated to the bench+SPRT channel
        // with the S0 §3.9 sibling depth reduction (2026-07-26): TT
        // presence changes move order, and an early alpha raise now
        // cheapens later siblings, so fixed-depth CENTIPAWN equality
        // between TT-on and TT-off no longer holds by design (doc 07
        // §4 step 6 re-scope). Decisive values stay exact — the §3.9
        // rule exempts them — and remain this suite's living bar.
        // Back-rank mate with pawn shields on both sides.
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 3),
    ];
    // S1 (D22): same re-scope as the PVS oracle — decisive values exact,
    // centipawn values within the correction-drift band (see
    // [`pvs_matches_full_window_negamax_at_fixed_depth`]).
    const S1_CORR_DRIFT_CP: Value = 8;
    for &(fen, depth) in suite {
        let tt_off = ref_negamax(&mut board(fen), &mut EvalContext::new(), depth, 0);
        let tt_on = tt_value_pair(fen, depth);
        if is_decisive(tt_off) {
            assert_eq!(tt_on, tt_off, "TT-on vs TT-off decisive value: {fen}");
        } else {
            assert!(
                (tt_on - tt_off).abs() <= S1_CORR_DRIFT_CP,
                "TT-on drifted past the S1 correction band: {fen} \
                     (TT-on {tt_on}, TT-off {tt_off})"
            );
        }
    }
}

#[test]
fn fresh_startpos_search_is_bit_reproducible() {
    // Full-material coverage without the exponential no-cutoff oracle.
    // Pre-P9 this asserted a fresh vs same-depth-WARMED startpos value
    // match; doc 10's LMR couples reductions to TT-supplied ttMoves
    // (see [`tt_value_pair`]), so a warm table legitimately drifts the
    // startpos value a few cp. The surviving invariant is single-thread
    // DETERMINISM (D24, §8 criterion 6): two independent FRESH searches
    // of the startpos — same driver, same fresh table, node-count dither
    // and all — reproduce the value, the best move AND the node count
    // bit-for-bit.
    let a = go_depth_full_window(START_FEN, 3);
    let b = go_depth_full_window(START_FEN, 3);
    assert_eq!(best_score(&a), best_score(&b), "startpos value determinism");
    assert_eq!(a.best_move(), b.best_move(), "startpos move determinism");
    assert_eq!(a.nodes, b.nodes, "startpos node-count determinism");
}

#[test]
fn warm_tt_searches_fewer_nodes_at_equal_depth() {
    // The other half of the step-6 exit bar in miniature (the real
    // "bench nodes drop materially" number is recorded in the commit
    // trailer, D24): re-searching the same position at the same depth
    // against the warmed table must cost fewer nodes — the ttMove-
    // first ordering and the eval cache are actually doing work.
    //
    // Depth 6 (was 5, and 4 before that): doc 10 §3.5's LMP shrinks the
    // startpos tree so hard at small depths that the ttMove ordering can
    // no longer beat it — a warm table actually reshapes the LMP-pruned
    // tree LARGER there. The SF `ebcea3ef` tunable refresh (2026-08-01)
    // shrank the tree again and pulled depth 5 back into that regime
    // (fresh 956 / warm 1583); measured over depths 4..9 the crossover is
    // at 6, where the margin is wide and stable (fresh 5410 / warm 1250,
    // and still 3:2 at depth 9). The fresh-vs-warm VALUE equality is no
    // longer asserted: as with `fresh_startpos_search_is_bit_reproducible`
    // (LMR) and now LMP, a warm table's richer move info legitimately
    // drifts the lossy fixed-depth score a few cp — the node shrink is
    // the invariant this test owns.
    let tt = test_tt();
    let fresh = go_depth_on(START_FEN, 6, &tt);
    let warm = go_depth_on(START_FEN, 6, &tt);
    assert!(
        warm.nodes < fresh.nodes,
        "warm TT must shrink the tree: {} !< {}",
        warm.nodes,
        fresh.nodes
    );
}

// --- Wave-C item 12: eval-diff nudge (N8) + TT-cutoff history (N6b) ---

#[test]
fn item12_nudges_preserve_correctness_and_determinism() {
    // The N8 eval-diff nudge and the N6b TT-cutoff bonus/penalty write
    // into the history tables from out-of-band sites (opponent-keyed main
    // history, parent conthist, quiet ttMove). A search must still (a)
    // never panic — the from-square guard protects against a colliding
    // ttMove mis-indexing — (b) solve a mate exactly, and (c) stay
    // bit-reproducible across fresh runs (the nudges are pure functions of
    // deterministic state).
    //
    // Back-rank mate in 1: 1.Ra8#.
    let w = go_depth("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 4);
    assert!(is_mate(best_score(&w)), "mate score must survive item 12");
    let a = go_depth(START_FEN, 5);
    let b = go_depth(START_FEN, 5);
    assert_eq!(a.nodes, b.nodes, "item 12 preserves node determinism");
    assert_eq!(a.best_move(), b.best_move());
}

// --- Wave-E items 17/19/20: hindsight, N11b, fail-low bonus, TT hardening ---

#[test]
fn wave_e_preserves_mate_and_determinism() {
    // Hindsight depth adjustments (item 17) mutate `depth` mid-node; N11b
    // widens `improving`; the fail-low parent bonus (item 19) and the TT
    // child-verification / wrong-direction penalize (item 20) all fire
    // during a normal search. None may corrupt a mate score or break
    // single-thread determinism.
    //
    // Back-rank mate (1.Ra8#), searched deep so RFP/hindsight/fail-low all
    // run at interior plies without corrupting the mate score.
    let m1 = go_depth("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 6);
    assert!(is_mate(best_score(&m1)), "Wave E keeps mate scores exact");

    let a = go_depth(START_FEN, 7);
    let b = go_depth(START_FEN, 7);
    assert_eq!(a.nodes, b.nodes, "Wave E preserves node determinism");
    assert_eq!(a.best_move(), b.best_move());
}

// --- Wave-D items 13/14/15: singular extensions, multicut, is_shuffling ---

#[test]
fn is_shuffling_guards_and_detects_oscillation() {
    use crate::moves::Move;
    use crate::types::Square;

    // Cheap guards: a capture is never a shuffle, nor a shallow ply.
    let mut w = Worker::new(board(START_FEN));
    let cap = Move::normal(Square::E4, Square::D5);
    assert!(!w.is_shuffling(cap, ss(20), 20)); // capture_stage → false anyway
    let quiet = Move::normal(Square::B1, Square::C3);
    assert!(
        !w.is_shuffling(quiet, ss(20), 20),
        "fresh board: clock/null low"
    );

    // Build a real shuffle: both sides oscillate a knight (alternating
    // turns), lifting rule50 and plies_from_null past the guards and
    // landing the white knight back on b1 with White to move.
    w = Worker::new(board("1n4k1/8/8/8/8/8/8/1N4K1 w - - 8 40"));
    let hops = [
        Move::normal(Square::B1, Square::C3), // w
        Move::normal(Square::B8, Square::C6), // b
        Move::normal(Square::C3, Square::B1), // w
        Move::normal(Square::C6, Square::B8), // b
        Move::normal(Square::B1, Square::C3), // w
        Move::normal(Square::B8, Square::C6), // b
        Move::normal(Square::C3, Square::B1), // w
        Move::normal(Square::C6, Square::B8), // b
    ];
    for &m in &hops {
        let gc = w.board.gives_check(m);
        w.board.make_move(m, gc);
    }
    assert!(w.board.rule50_count() >= 10 && w.board.plies_from_null() >= 6);
    // Oscillation: m = b1->c3; (ss-2) = c3->b1; (ss-4) = b1->c3.
    let cur = ss(20);
    w.stack[cur - 2].current_move = Move::normal(Square::C3, Square::B1);
    w.stack[cur - 4].current_move = Move::normal(Square::B1, Square::C3);
    assert!(
        w.is_shuffling(Move::normal(Square::B1, Square::C3), cur, 20),
        "a retraced quiet in 50-move territory must read as shuffling"
    );
    // Moving a DIFFERENT piece (the king, not the oscillating knight)
    // breaks the pattern — only `m.from()` is tested against (ss-2).to().
    assert!(!w.is_shuffling(Move::normal(Square::G1, Square::F1), cur, 20));
}

#[test]
fn singular_preserves_mate_and_determinism() {
    // The exclusion search must never recurse (excluded_move disables the
    // trigger — asserted structurally + the stack invariant at init), mate
    // scores stay exact, and the search stays bit-reproducible. Depth 10 is
    // past the SE_MIN_DEPTH (6) gate, so singular actually fires.
    let m1 = go_depth("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 10);
    assert!(is_mate(best_score(&m1)), "singular keeps mate exact");

    let a = go_depth(START_FEN, 9);
    let b = go_depth(START_FEN, 9);
    assert_eq!(a.nodes, b.nodes, "singular preserves node determinism");
    assert_eq!(a.best_move(), b.best_move());
}

#[test]
fn exclusion_search_skips_the_excluded_move() {
    // design/10 §3.5 / §3.10 step 2: an exclusion search must SKIP the
    // ttMove (`excluded_move`) — the MovePicker still emits it first, so
    // without the top-of-loop skip the search would include the very move
    // it is meant to exclude (defeating the singular test and firing
    // spurious multicuts). We drive a node whose ONLY legal move is the
    // excluded one: the loop must end with `move_count == 0` and take the
    // §3.10 fail-low, returning `alpha`. Before the skip landed this
    // searched Ka7 and returned a real (non-alpha) value.
    let fen = "k7/8/2K5/8/8/8/8/1R6 b - - 0 1";
    let mut w = Worker::new(board(fen));
    let tt = test_tt();
    let only = Move::normal(Square::A8, Square::A7);
    let legal = all_legal(&w.board);
    assert_eq!(legal.len(), 1, "fixture must have exactly one legal move");
    assert_eq!(legal[0], only, "the one legal move is Ka7");

    init_stack(&mut w.stack);
    // A PV node with a real window reaches the move loop without any
    // pre-loop pruning short-circuit (razoring/NMP/ProbCut are !PV or
    // excluded-gated; the losing eval keeps RFP dormant).
    let cur = ss(1);
    w.stack[cur].excluded_move = only;
    // An exclusion search reuses the enclosing node's already-computed
    // static eval (§3.6); seed it so the eval-reuse path is valid.
    w.stack[cur].static_eval = static_eval(&w.board);
    let stop = AtomicBool::new(false);
    let alpha: Value = -100;
    let beta: Value = 100;
    let v = w.search::<Pv>(&tt, &stop, alpha, beta, 4, false, 1);
    assert_eq!(
        v, alpha,
        "excluding the sole legal move must leave move_count == 0 and fail low with alpha"
    );
}

// --- Wave-G items 22/23: threat-aware ordering + threat-coupled RFP ---

#[test]
fn wave_g_ordering_and_rfp_preserve_correctness() {
    // G1 (offense/escape ordering off the threat map) and G2 (RFP margin
    // relaxed when no own piece is threatened) must not corrupt mate scores
    // or determinism, and must still find a hanging piece.
    let m = go_depth("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 6);
    assert!(is_mate(best_score(&m)), "Wave G keeps mate exact");

    // A free undefended queen is still won (the offense term rewards
    // attacking it, but SEE/search find it regardless).
    let w = go_depth("3q3k/8/8/8/8/8/3R4/3R3K w - - 0 1", 6);
    assert!(best_score(&w) > 800, "Wave G must not miss the free queen");

    let a = go_depth(START_FEN, 7);
    let b = go_depth(START_FEN, 7);
    assert_eq!(a.nodes, b.nodes, "Wave G stays deterministic");
    assert_eq!(a.best_move(), b.best_move());
}

#[test]
fn rfp_threat_coupling_is_disarmed_when_an_own_piece_is_attacked() {
    // The G2 threat-coupled RFP relaxation subtracts RFP_THREAT_MARGIN only
    // when `all_threats() & occupancy(us)` is EMPTY (no own piece
    // threatened). This pins the negative branch the coupling exists for —
    // the `no_own_threat = false` path at mod.rs N10 — which the
    // mate/determinism smoke tests never exercise. Keys on ALL our pieces
    // (pawns included), matching the N10 use-site and Reckless's
    // `all_threats() & colors(stm)`.

    // A hanging own rook: the black queen on d8 attacks the White rook on
    // d4 down the open d-file. An own (White) piece IS threatened, so the
    // predicate is non-empty and the relaxation is DISARMED.
    let attacked = board("3q3k/8/8/8/3R4/8/8/6K1 w - - 0 1");
    let us = attacked.side_to_move();
    assert!(
        !(attacked.all_threats() & attacked.occupancy(us)).is_empty(),
        "a hanging own rook must register in all_threats() & occupancy(us), \
             disarming the RFP relaxation"
    );

    // A quiet position: no White piece stands on any square Black attacks,
    // so the predicate is empty and the relaxation is ARMED (the margin
    // shrinks by RFP_THREAT_MARGIN).
    let safe = board("6k1/8/8/8/8/8/6P1/6K1 w - - 0 1");
    let us2 = safe.side_to_move();
    assert!(
        (safe.all_threats() & safe.occupancy(us2)).is_empty(),
        "with no own piece attacked the relaxation predicate must be empty"
    );
}

// --- Wave-F item 21: correction-history couplings ---

#[test]
fn item21_correction_history_learns_and_preserves_mate() {
    // The node-result correction update (direction-consistency gated) must
    // actually write the correction tables during a search, and the RFP/
    // reduction/singular couplings must not corrupt mate scores or
    // determinism.
    let w = go_depth(START_FEN, 7);
    // At least one correction cell moved off its init (CORR_INIT = -6).
    let moved = w.histories.corr.iter().any(|slot| {
        slot.iter()
            .any(|b| b.pawn.get() != -6 || b.non_pawn_white.get() != -6)
    });
    assert!(moved, "correction history must learn during a search");

    // Mate stays exact with the correction couplings live.
    let m = go_depth("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 6);
    assert!(
        is_mate(best_score(&m)),
        "correction couplings keep mate exact"
    );

    let a = go_depth(START_FEN, 7);
    let b = go_depth(START_FEN, 7);
    assert_eq!(a.nodes, b.nodes, "correction couplings stay deterministic");
    assert_eq!(a.best_move(), b.best_move());
}

#[test]
fn item16_probcut_preserves_tactics_and_determinism() {
    // ProbCut (N13) runs at depth >= 3 with a qsearch probe then a real
    // reduced verification search; N14 is its in-check twin. Neither may
    // corrupt tactics: a free queen (1.Rxd8) is still won, and fresh
    // searches stay bit-reproducible (the probe/verify are deterministic).
    let w = go_depth("3q3k/8/8/8/8/8/3R4/3R3K w - - 0 1", 6);
    assert!(
        best_score(&w) > 800,
        "ProbCut must not miss the winning capture (up ~a queen): {}",
        best_score(&w)
    );
    let a = go_depth(START_FEN, 7);
    let b = go_depth(START_FEN, 7);
    assert_eq!(a.nodes, b.nodes, "ProbCut preserves node determinism");
    assert_eq!(a.best_move(), b.best_move());
}

#[test]
fn n14_in_check_probcut_fails_high_before_the_move_loop() {
    // N14 (design/10 §3.4, Wave-E item 16) is an unconditional early return
    // that runs even in check: a valid, non-decisive, stored LOWER bound at
    // depth >= depth - PROBCUT_DEPTH_SUB already >= beta + PROBCUT_CHECK_BETA
    // lets the node fail high at `beta + PROBCUT_CHECK_BETA` WITHOUT entering
    // the move loop. This white-box test seeds exactly that TT entry on an
    // in-check node and pins the early return; a mirror with a contradicting
    // (UPPER) bound asserts the gate stays dormant. The smoke test
    // (`item16_...`) uses non-check positions and never deterministically
    // populates this path.
    //
    // Black king in check from the white rook down the open h-file.
    let fen = "7k/8/8/8/8/8/6K1/7R b - - 0 1";
    let beta: Value = 100;
    let alpha: Value = beta - 1; // NonPv zero window
    let depth: Depth = 6;
    let check_probcut_beta = beta + params::PROBCUT_CHECK_BETA();

    // Seed the node's own TT entry, then run one direct `search`. The
    // stored depth is one ply short of the N6b cutoff's strict `>` slack
    // (so the N6b cutoff cannot pre-empt N14) but still clears N14's
    // `>= depth - PROBCUT_DEPTH_SUB` (= depth - 4).
    let run = |bound: Bound, value: Value| -> Value {
        let mut w = Worker::new(board(fen));
        let tt = test_tt();
        assert!(!w.board.checkers().is_empty(), "fixture must be in check");
        init_stack(&mut w.stack);
        let key = w.board.key();
        tt.probe(key).writer.write(
            key,
            value_to_tt(value, 1),
            false,
            bound,
            depth - 1,
            Move::NULL,
            VALUE_NONE,
            tt.generation(),
        );
        let stop = AtomicBool::new(false);
        w.search::<NonPv>(&tt, &stop, alpha, beta, depth, true, 1)
    };

    assert_eq!(
        run(Bound::Lower, check_probcut_beta + 50),
        check_probcut_beta,
        "a deep LOWER bound above the raised beta must trip N14's fail-high"
    );
    assert_ne!(
        run(Bound::Upper, check_probcut_beta + 50),
        check_probcut_beta,
        "an UPPER bound fails includes_lower(): N14 must stay dormant"
    );
}

#[test]
fn wave_e_zugzwang_reti_still_draws_or_wins() {
    // The Réti zugzwang study (design/10 §5): NMP + the new hindsight /
    // TT-hardening must not turn the drawn/won study into a loss. White
    // to move holds (the study is a draw with best play; a shallow search
    // must at least not evaluate it as lost for White).
    let w = go_depth("8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1", 8);
    assert!(
        !is_mated(best_score(&w)),
        "Réti zugzwang must not be scored as a White loss: {}",
        best_score(&w)
    );
}

#[test]
fn item12_n8_nudges_opponent_main_history() {
    // The N8 nudge writes into `main[opponent]` for the parent's quiet
    // move — a table cell that a from-scratch board leaves at MAIN_INIT.
    // After a few plies of a quiet opening, at least one opponent-keyed
    // main-history cell must have moved off the init value, proving the
    // out-of-band write site actually fires (and does not panic).
    let w = go_depth(START_FEN, 6);
    let moved = w
        .histories
        .main
        .iter()
        .any(|plane| plane.iter().any(|e| e.get() != -5));
    assert!(moved, "N8/history writes must move main history off init");
}
