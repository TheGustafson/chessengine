//! Tuned search parameters — every SPSA-movable constant, named and
//! greppable in one file (D20; conventions: design/01 §3.8).
//!
//! Values are SF-current starting points from Stockfish commit `eca43a97`
//! (2026-07-06) unless noted. Structural constants (bit widths,
//! `VALUE_MATE`, array sizes) live with their types; piece values are
//! `types::PIECE_VALUES`; history gravity bounds are const generics in
//! `history.rs`.
//!
//! Naming: `TECHNIQUE_ROLE` screaming-snake, grouped by technique in
//! design/07 / design/10 section order. Each constant documents only its
//! role; the formula *shape* is documented at the use site in `search`.
//! Any bare numeric literal in `search/`, `movepick.rs`, or `see.rs` that
//! a tuner could move is a defect (design/01 §3.8).

use crate::types::{Depth, Value};

/// The per-thread vote tempo (design/13 §3.8): each thread votes for its own
/// best move with weight `score - min_score + VOTE_TEMPO`, so even the
/// weakest thread's opinion is counted. Value units.
pub const VOTE_TEMPO: crate::types::Value = 14;
/// Per-offset weights of the continuation-history update, offsets 1..=6
/// (÷[`CONT_UPDATE_DIVISOR`] with the bonus).
pub const CONT_OFFSET_WEIGHTS: [i32; 6] = [1040, 780, 300, 537, 129, 423];
/// Consistency multipliers indexed by the running count of already-positive
/// entries along the chain: an agreeing chain amplifies the update.
pub const CONT_CONSISTENCY_MULT: [i32; 7] = [96, 113, 101, 105, 127, 121, 126];

/// History-buys-lmr_depth divisor by depth (design/10 §3.5):
/// `lmr_depth += history / LMR_DIVISOR[min(depth,16)-1]` where `history` is
/// the quiet's combined continuation/pawn score plus `2·main_history`.
pub const LMR_DIVISOR: [i32; 16] = [
    3307, 2930, 2874, 2818, 3215, 3225, 3224, 2782, 2858, 2919, 3088, 3275, 3180, 2868, 3006, 3599,
];

/// Cadence of the search's clock poll: every `CHECK_TIME_CALLS` calls.
/// A plain const (not a tunable): the poll mask `x & (CALLS - 1)` needs a
/// compile-time power-of-two guarantee, and cadence is not a strength knob.
pub const CHECK_TIME_CALLS: i32 = 512;
const _: () = assert!(
    CHECK_TIME_CALLS > 0 && (CHECK_TIME_CALLS as u64).is_power_of_two(),
    "CHECK_TIME_CALLS must be a positive power of two (the poll mask assumes it)"
);

crate::tunables! {

    // --- Aspiration windows (design/07 §3.6) ---

    /// Base half-width of the window around the root move's average score.
    pub ASPIRATION_BASE_DELTA: Value = 5;
    /// Divisor of `mean_squared_score`: unstable scores widen the base delta.
    /// NOT Stockfish's 10588 — a frame correction: mss is a SQUARED eval
    /// quantity and our frame is 0.64x SF's, so the divisor scales by `r`,
    /// not `r²`: `DIV = r · 10588 = 6776`. NOTE: lower values (~3000) win
    /// corpus-mean node screens but have a catastrophic root fail-high tail;
    /// never lower this off a corpus mean.
    pub ASPIRATION_MSS_DIVISOR: i64 = 6776;
    /// Optimism numerator: `optimism = OPTIMISM_NUM·avg / (|avg| +
    /// OPTIMISM_DEN_BASE)` at aspiration setup (design/11 §3.6); SF's 137/81
    /// scaled into our compressed eval frame.
    pub OPTIMISM_NUM: Value = 75;
    /// Optimism half-saturation denominator base (see OPTIMISM_NUM).
    pub OPTIMISM_DEN_BASE: Value = 45;
    /// Numerator of the per-re-search widening `delta += delta * 44 / 128`.
    pub ASPIRATION_WIDEN_NUM: Value = 44;
    /// Numerator of the search-again depth damping term
    /// `3 * (search_again_counter + 1) / 4` in the aspiration loop's
    /// `adjusted_depth` (design/07 §3.6, design/12 §3.9).
    pub ASPIRATION_SEARCH_AGAIN_NUM: Depth = 3;

    // --- Move ordering: capture scoring (design/09 §3.7) ---

    /// MVV weight on the victim value in the capture ordering score
    /// `CAPTURE_MVV_FACTOR * victim_value + capture_history`: no attacker
    /// (LVA) term — SEE handles attacker risk at the good/bad split gate.
    pub CAPTURE_MVV_FACTOR: Value = 7;

    /// Recapture discount in the QUIESCENCE/ProbCut capture score, in 16ths
    /// of the attacker's value, applied when the destination is defended.
    /// Only where `depth <= DEPTH_QS` — the main pipeline's good/bad SEE
    /// split already prices attacker risk. NOTE: 56 = half a recapture, not
    /// the principled 112: a full discount pushes defended captures past
    /// `QS_LATE_MOVE_LIMIT`, which DELETES them rather than reordering them.
    pub QS_CAPTURE_DEFENDED_NUM: Value = 56;

    /// Quiet SEE penalty: subtracted from a quiet's ordering score when the
    /// move loses material by static exchange (`!see_ge(m, 0)`); guarded on
    /// the destination being enemy-attacked and on
    /// `value > GOOD_QUIET_THRESHOLD` (both skip pointless `see_ge` calls).
    ///
    /// CLAMPED at `GOOD_QUIET_THRESHOLD + 1` so it can only reorder — the
    /// unclamped form wins every cheap proxy (nodes, instructions) and LOSES
    /// games: it duplicates the move loop's depth-scaled quiet SEE prune as
    /// a depth-blind copy. An ordering change that can cross
    /// `GOOD_QUIET_THRESHOLD` is a pruning change. Value unresolved inside
    /// 24576..98304; SPSA owns it.
    pub QUIET_SEE_PENALTY: Value = 49152;

    // --- Move ordering: good/bad capture split (design/09 §3.6 stage 3) ---

    /// Divisor of the score-dependent SEE gate `see_ge(m, -score / 18)` that
    /// splits captures into good and bad (design/09 §3.6).
    pub GOOD_CAPTURE_SEE_DIVISOR: Value = 18;

    // --- Move ordering: quiet scoring (design/09 §3.7) ---

    /// Weight on butterfly (main) history in the quiet ordering score.
    pub QUIET_MAIN_HISTORY_WEIGHT: Value = 2;
    /// Weight on pawn-structure history in the quiet ordering score.
    pub QUIET_PAWN_HISTORY_WEIGHT: Value = 2;
    /// Flat bonus for a quiet that gives check, gated by the SEE margin below
    /// so a check that hangs material earns nothing.
    pub QUIET_CHECK_BONUS: Value = 16384;
    /// SEE margin gating the check bonus (`see_ge(m, -75)`).
    pub QUIET_CHECK_SEE_MARGIN: Value = -75;
    /// Per-piece-value multiplier of the threat escape/entry term: fleeing a
    /// square attacked by a strictly cheaper enemy piece earns
    /// `QUIET_THREAT_FACTOR * piece_value`, stepping onto one costs the same
    /// (`Board::piece_threats`).
    pub QUIET_THREAT_FACTOR: Value = 20;
    /// Weight on low-ply history in the quiet ordering score (plies 0..4 only,
    /// scaled by `/ (1 + ply)`).
    pub QUIET_LOW_PLY_WEIGHT: Value = 8;

    // --- Move ordering: quiet good/bad split + sort (design/09 §3.6) ---

    /// Good/bad quiet partition threshold (design/09 §3.6): quiets scoring
    /// strictly above this are "good" (emitted before bad captures); the
    /// exact complement are "bad" (emitted last).
    pub GOOD_QUIET_THRESHOLD: Value = -14000;
    /// Depth multiplier of the quiet partial-sort limit `-3560 * depth`
    /// (design/09 §3.6): deeper nodes sort more of the list.
    pub QUIET_SORT_DEPTH_FACTOR: Value = -3560;

    // --- Move ordering: evasion scoring (design/09 §3.7) ---

    /// Constant lifting every capturing evasion above every quiet evasion:
    /// `2^28` dwarfs any history-based quiet-evasion score.
    pub EVASION_CAPTURE_BONUS: Value = 1 << 28;

    // --- Quiescence search (design/07 §3.8) ---

    /// Futility headroom over the corrected static eval (§3.8 step 5): a
    /// capture whose victim cannot lift `static_eval + 335` above alpha is
    /// skipped.
    pub QS_FUTILITY_MARGIN: Value = 335;
    /// The hard SEE floor: any move whose static exchange value cannot reach
    /// this bar is never searched in qsearch (§3.8 step 7).
    pub QS_SEE_FLOOR: Value = -74;
    /// Late-move cutoff: after this many non-checking moves not aimed at the
    /// previous move's destination, stop trying such moves (§3.8 step 7).
    pub QS_LATE_MOVE_LIMIT: i32 = 2;

    // --- Razoring (design/10 §3.1; node step N9) ---

    /// Flat margin below alpha at which shallow nodes drop to qsearch.
    pub RAZOR_BASE: Value = 465;
    /// Depth-quadratic term of the razoring margin (`* depth^2`).
    pub RAZOR_DEPTH_MULT: Value = 300;

    // --- Reverse futility pruning (design/10 §3.4; node step N10) ---
    //
    // A non-PV node whose corrected static eval clears beta by a depth-scaled
    // trend-adjusted margin returns a beta-ward blend without searching.

    /// Exclusive depth ceiling: RFP only runs while `depth < RFP_MAX_DEPTH`
    /// (the depth condition matters for mate finding — deep nodes never RFP).
    pub RFP_MAX_DEPTH: Depth = 17;
    /// Base of the per-ply futility multiplier `min(RFP_MULT_BASE +
    /// RFP_MULT_SLOPE·depth, RFP_MULT_CAP)`.
    pub RFP_MULT_BASE: Value = 40;
    /// Per-ply slope of the futility multiplier.
    pub RFP_MULT_SLOPE: Value = 4;
    /// Cap of the futility multiplier.
    pub RFP_MULT_CAP: Value = 80;
    /// Subtracted from the futility multiplier on a TT miss (no TT data =
    /// prune more cautiously).
    pub RFP_MULT_NO_TTHIT: Value = 20;
    /// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which an *improving*
    /// node shrinks the margin.
    pub RFP_IMPROVING: Value = 2934;
    /// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which a *worsening*
    /// opponent shrinks the margin.
    pub RFP_OPP_WORSENING: Value = 343;
    /// Beta weight (in 1024ths) of the RFP return blend
    /// `(W·beta + (1024−W)·eval) / 1024` — a value ≥ beta, pulled toward beta
    /// to damp static-eval noise (same idiom as the fail-high softens).
    pub RFP_RET_BETA_W: Value = 716;

    // --- Null move pruning (design/10 §3.4; node step N11) ---
    //
    // Zugzwang is fenced three ways: NMP runs only with non-pawn material on
    // the side to move, a proven-mate null value is never trusted, and at
    // high depth a verification search re-confirms with NMP disabled up to
    // `nmp_min_ply`. The gate reads `ss.static_eval` (NOT the TT-substituted
    // `eval` RFP uses).

    /// Per-ply slope of the NMP static-eval bar `beta - NMP_EVAL_DEPTH·depth -
    /// NMP_EVAL_IMPROVING·improving + NMP_EVAL_BASE`: deeper nodes need less
    /// headroom.
    pub NMP_EVAL_DEPTH: Value = 14;
    /// Improving discount of the NMP static-eval bar.
    pub NMP_EVAL_IMPROVING: Value = 45;
    /// Flat base of the NMP static-eval bar.
    pub NMP_EVAL_BASE: Value = 374;
    /// Base of the null-move dynamic reduction `R = NMP_R_BASE + depth /
    /// NMP_R_DEPTH_DIV` (whole plies, applied directly to depth).
    pub NMP_R_BASE: Depth = 7;
    /// Depth divisor of the null-move reduction.
    pub NMP_R_DEPTH_DIV: Depth = 3;
    /// Depth at/above which a fail-high null search is re-confirmed by the
    /// verification search; below it the null value is returned directly.
    pub NMP_VERIFY_DEPTH: Depth = 16;

    // --- Base reduction table + `reduction()` (design/10 §3.3) ---
    //
    // The per-worker log table `reductions[i] = int(RED_TABLE_NUM /
    // RED_TABLE_DEN * ln(i))`; reductions are carried in 1024ths of a ply,
    // divided by 1024 only at the moment of application.

    /// Numerator of the log reduction table `int(2834 / 128 * ln(i))`.
    pub RED_TABLE_NUM: i32 = 2834;
    /// Wide-window damping: a wider local window `delta` reduces less, via
    /// `- delta * RED_DELTA_MULT / root_delta`.
    pub RED_DELTA_MULT: i32 = 617;
    /// Not-improving surcharge numerator: a not-improving node reduces more, by
    /// `+ scale * RED_NON_IMPROVING / RED_NON_IMPROVING_DEN` (~38% extra).
    pub RED_NON_IMPROVING: i32 = 194;
    /// Flat base added to every base reduction (in 1024ths of a ply).
    pub RED_BASE: i32 = 1027;

    // --- LMR reduction finalization + search mechanics (design/10 §3.7/§3.8) ---
    //
    // The base [`Worker::reduction`] output `r` accretes the §3.7
    // finalization terms; §3.8 turns it into the reduced zero-window scout
    // depth, with the deeper/shallower re-search rule and the non-LMR dock
    // path.

    /// Flat rebase added to every finalized reduction (§3.7).
    pub RED_FLAT: i32 = 714;

    /// Draw contempt: how much worse than equality a draw is judged, from the
    /// ROOT side's point of view (internal units) — a draw taken at −1 sets
    /// alpha and starves a winning sibling of nodes.
    /// DEFAULT 0 — bit-identical default build, `Bench:` pin untouched.
    pub DRAW_CONTEMPT: Value = 0;
    /// Per-move-count reduction discount `r -= RED_MOVECOUNT * move_count`:
    /// later moves reduce a little less than the raw log table implies (§3.7).
    pub RED_MOVECOUNT: i32 = 62;
    /// Cut-node reduction surcharge: an expected-fail-high node reduces its
    /// late moves harder (§3.7).
    pub RED_CUTNODE: i32 = 3995;
    /// Extra cut-node surcharge when the node has no TT move (§3.7).
    pub RED_CUTNODE_NO_TTMOVE: i32 = 1059;
    /// TT-move reduction rebate `r = max(0, r - RED_TTMOVE_SUB)`: the move the TT
    /// records as best is searched close to full depth (§3.7).
    pub RED_TTMOVE_SUB: i32 = 2016;
    /// Cap on how many plies a NEGATIVE net `r` may DEEPEN the reduced scout
    /// past `new_depth` (§3.8); PV nodes get a further unconditional +1.
    pub RED_MAX_DEEPER: Depth = 2;
    /// Deeper-re-search margin: after a reduced scout beats alpha, re-search one
    /// ply DEEPER when it beat the running best by more than this (§3.8).
    pub LMR_DEEPER_MARGIN: Value = 52;
    /// Shallower-re-search margin: re-search one ply SHALLOWER when the scout only
    /// barely cleared the running best (`value < best_value + this`, §3.8).
    pub LMR_SHALLOWER_MARGIN: Value = 9;
    /// Non-LMR path: a node with no TT move adds this to `r` before the
    /// depth docks below (§3.8).
    pub RED_FULLDEPTH_NO_TTMOVE: i32 = 1085;
    /// Non-LMR path first depth dock: an `r` above this costs one ply (§3.8).
    pub RED_FULLDEPTH_DOCK1: i32 = 5039;
    /// Non-LMR path second depth dock: an `r` above this (and
    /// `new_depth > 2`) costs a second ply (§3.8).
    pub RED_FULLDEPTH_DOCK2: i32 = 5223;

    // --- LMR statScore: combined-history reduction + post-LMR bonus
    //     (design/10 §3.7/§3.8) ---
    //
    // `ss.stat_score` is computed from PRE-move state (the child reads
    // `(ss-1).stat_score`) and docks the reduction. Rust integer division
    // truncates toward zero exactly like C++: a negative score does NOT floor.

    /// Butterfly-history weight in the quiet `stat_score`; a dedicated
    /// tunable, independent of [`QUIET_MAIN_HISTORY_WEIGHT`].
    pub STATSCORE_MAIN_WEIGHT: i32 = 2;
    /// Victim-value multiplier of the capture `stat_score`
    /// (`STATSCORE_CAPT_NUM·piece_value(captured) / STATSCORE_CAPT_DEN`).
    pub STATSCORE_CAPT_NUM: i32 = 809;
    /// Numerator of the reduction dock `r -= stat_score·RED_STATSCORE_NUM /
    /// RED_STATSCORE_DEN`: positive `stat_score` LOWERS the reduction (search
    /// fuller), negative RAISES it.
    pub RED_STATSCORE_NUM: i32 = 445;
    /// Flat continuation-history bonus a move earns when its reduced LMR scout
    /// beats alpha (design/10 §3.8): applied along the offset-1..6 chain.
    pub LMR_CONTHIST_BONUS: i32 = 1415;

    // --- LMR reduction finalization: ttPv rebate / tt_capture / all-node
    //     (design/10 §3.5/§3.7) ---
    //
    // The §3.7 accretion terms that BALANCE the reduction: a PV-flagged node
    // with a trustworthy TT entry searches much fuller.

    /// §3.5 provisional ttPv reduction bump (mostly cancelled by the rebate).
    pub RED_TTPV_ENTRY: i32 = 1006;
    /// ttPv rebate: base.
    pub RED_TTPV_BASE: i32 = 2766;
    /// ttPv rebate: PV bonus.
    pub RED_TTPV_PV: i32 = 1017;
    /// ttPv rebate: `tt.value > alpha` bonus.
    pub RED_TTPV_TTVALUE: i32 = 838;
    /// ttPv rebate: `tt.depth >= depth` base bonus.
    pub RED_TTPV_TTDEPTH_BASE: i32 = 923;
    /// ttPv rebate: extra `tt.depth >= depth` bonus at a cut node.
    pub RED_TTPV_TTDEPTH_CUT: i32 = 955;
    /// tt_capture reduction surcharge (a capturing ttMove distrusts quiet siblings).
    pub RED_TTCAPTURE: i32 = 1039;
    /// All-node reduction scaling numerator `r += r·RED_ALLNODE_NUM /
    /// (RED_ALLNODE_DEPTH_MULT·depth + RED_ALLNODE_DEPTH_BASE)`.
    pub RED_ALLNODE_NUM: i32 = 272;
    /// All-node scaling depth multiplier.
    pub RED_ALLNODE_DEPTH_MULT: i32 = 256;
    /// All-node scaling depth base.
    pub RED_ALLNODE_DEPTH_BASE: i32 = 285;

    // --- Internal iterative reduction (N12, design/10 §3.4; S0) ---

    /// Minimum depth for IIR: a PV/cut node with no ttMove at `depth >=
    /// IIR_MIN_DEPTH` searches one ply shallower.
    pub IIR_MIN_DEPTH: Depth = 6;

    /// Fail-low best-move store: `1` = store the node's argmax move on a
    /// main-search fail low (default stores `Move::NULL`). MEASURED DEAD —
    /// leave at 0; kept only as the reproduction path for that claim (the
    /// argmax at an all-node is over reduced upper bounds of moves ordering
    /// already ranked first, so it carries almost nothing; SF and Reckless
    /// store nothing there). See [`TT_UPPER_PRUNE_BLIND`].
    pub TT_FAILLOW_MOVE: i32 = 0;

    /// `1` = the four sites that read `tt_data.mv == Move::NULL` as "this
    /// node is unresolved" (IIR, RFP's ttMove gate, `RED_CUTNODE_NO_TTMOVE`,
    /// `RED_FULLDEPTH_NO_TTMOVE`) ignore a move riding an UPPER-bound entry.
    /// REJECTED IN GAMES — never put this in an SPSA campaign: the IIR bit
    /// alone shrinks the fixed-depth tree dramatically and loses games badly
    /// (SF: making IIR more aggressive scales poorly). Deliberately separate
    /// from [`TT_FAILLOW_MOVE`]: the two must be screened as a 2x2.
    pub TT_UPPER_PRUNE_BLIND: i32 = 0;

    // --- Move-loop pruning: late move pruning (design/10 §3.5) ---
    //
    // Once `move_count` clears the threshold, the picker stops emitting
    // quiets. The whole move-loop pruning block is fenced by the mate-safety
    // guard `!is_loss(best_value)` and the zugzwang guard
    // `non_pawn_material(us) != 0`, never at the root.

    // --- Implicit-coefficient tunables: names and knobs for terms that
    // carried an invisible x1 or a bare literal; all default to exact
    // identity. ---

    /// Base term of the LMP move-count threshold `(LMP_BASE + depth²) /
    /// (2 - improving)` (design/10 §3.5).
    pub LMP_BASE: i32 = 3;

    /// Numerator of the singular VERIFICATION depth, `new_depth *
    /// SE_VERIFY_DEPTH_NUM / SE_VERIFY_DEPTH_DEN` (was a bare `/ 2`): the
    /// depth of an entire extra search run at every singular candidate, so it
    /// sets both the cost of singular extensions and how trustworthy their
    /// verdict is.
    pub SE_VERIFY_DEPTH_NUM: Depth = 1;

    // --- Move-loop pruning: capture futility + capture/check SEE (design/10 §3.5) ---
    //
    // Two gates keyed by the capture history. `piece_value(captured)` is 0
    // for en passant — SF parity; no `!in_check` guard (the outer `!is_loss`
    // fence keeps evasions searched); the SEE gate never prunes our LAST
    // non-pawn material when `alpha < VALUE_DRAW` — a possible stalemate
    // resource.

    /// Exclusive lmr-depth ceiling of capture futility.
    pub CFUT_MAX_LMRDEPTH: Depth = 7;
    /// Flat base of the capture-futility optimistic value over the static eval.
    pub CFUT_BASE: Value = 231;
    /// Per-lmr-depth slope of the capture-futility value (can be negative at
    /// shallow nodes, LOWERING the bound).
    pub CFUT_LMRDEPTH_MULT: Value = 232;
    /// Capture-history nudge numerator of the capture-futility value
    /// (`CFUT_HIST_MULT·capt_hist / CAPT_HIST_DEN`).
    pub CFUT_HIST_MULT: i32 = 131;
    /// Depth multiplier of the capture/check SEE margin
    /// (`CAP_SEE_DEPTH_MULT·depth + CAP_SEE_HIST_MULT·capt_hist/CAPT_HIST_DEN`).
    pub CAP_SEE_DEPTH_MULT: Value = 175;
    /// Capture-history numerator of the capture/check SEE margin.
    pub CAP_SEE_HIST_MULT: i32 = 34;

    // --- Move-loop pruning: continuation-history pruning (design/10 §3.5) ---
    //
    // First gate in the quiet arm: a quiet whose combined recent history
    // (contHist(1) + contHist(2) + pawn slot, PRE-move state) falls below a
    // depth-scaled bar is skipped. Fresh histories sum to −2366, above the
    // bar at every depth, so the gate is dormant until updates drive a quiet
    // below it.

    /// Linear depth multiplier of the continuation-history pruning threshold
    /// `history < -QHIST_PRUNE_MULT·depth` (design/10 §3.5).
    pub QHIST_PRUNE_MULT: i32 = 4313;

    // --- Move-loop pruning: parent futility (design/10 §3.5) ---
    //
    // A shallow, not-in-check node writes off a quiet whose most optimistic
    // outcome still fails to reach alpha — after first lifting `best_value`
    // to the bound, gated by `!is_decisive(best_value) && !is_win(bound)` so
    // no mate score is corrupted (§8.11). `lmr_depth = new_depth - r/1024`.

    /// Flat base of the parent-futility bound over the static eval.
    pub PFUT_BASE: Value = 40;
    /// Bound surcharge while no best move has been found yet.
    pub PFUT_NO_BESTMOVE: Value = 138;
    /// Per-lmr-depth slope of the bound.
    pub PFUT_LMRDEPTH_MULT: Value = 117;
    /// Bound surcharge when the static eval already clears alpha.
    pub PFUT_EVAL_ABOVE_ALPHA: Value = 90;
    /// Exclusive lmr-depth ceiling of parent futility.
    pub PFUT_MAX_LMRDEPTH: Depth = 12;

    // --- Move-loop pruning: quiet SEE (design/10 §3.5) ---
    //
    // After parent futility, a quiet whose static-exchange value cannot reach
    // `-QSEE_MULT·lmr_depth²` (floored at 0) is pruned.

    /// Quadratic coefficient of the quiet SEE margin `-QSEE_MULT·lmr_depth²`
    /// (`lmr_depth` clamped to `>= 0` first).
    pub QSEE_MULT: Value = 25;

    // --- Transposition-table interaction (design/07 §3.7.2 steps 7/14) ---

    /// TT cutoffs are refused once the halfmove clock reaches this (the
    /// graph-history guard of design/07 §3.7.2 step 7 / design/08 §3.12).
    pub TT_CUTOFF_RULE50_LIMIT: i32 = 96;
    /// Extra depth credited to the TT store of a no-legal-moves node: a
    /// mate/stalemate terminal is exact far beyond its nominal depth.
    pub TT_NO_MOVES_DEPTH_BONUS: Depth = 6;

    // --- TT-cutoff history nudges (N6b extras, design/10 §3.4) ---
    //
    // When the non-PV TT cutoff fires with `tt.value >= beta`: reward a
    // quiet ttMove, penalize the parent's early quiet along its chain.

    /// Depth multiplier of the TT-cutoff quiet-ttMove bonus.
    pub TTCUT_BONUS_MULT: i32 = 114;
    /// Cap of the TT-cutoff quiet-ttMove bonus.
    pub TTCUT_BONUS_CAP: i32 = 724;
    /// Continuation-history malus for the parent's early quiet move on a
    /// beta-side TT cutoff.
    pub TTCUT_PARENT_MALUS: i32 = -2187;
    /// The parent's `move_count` must be below this for the TT-cutoff parent
    /// penalty to fire.
    pub TTCUT_PARENT_MOVECOUNT: i32 = 4;

    // --- Singular extensions + multicut (design/10 §3.6) ---
    //
    // For the ttMove, an exclusion search (`excluded_move = m`) at a margin
    // below `tt.value` decides: singular → extend (1 + double/triple
    // margins) and the WHOLE node deepens; value >= beta → multicut, prune
    // the subtree; else negative extensions (-3 tt-fail-high, -2 cut node).
    // `is_shuffling` suppresses extensions of shuffles in 50-move territory.

    /// Minimum depth for the singular trigger (`depth >= SE_MIN_DEPTH + tt_pv`).
    pub SE_MIN_DEPTH: Depth = 6;
    /// The ttMove's stored depth must be within this of the node depth.
    pub SE_TT_DEPTH_SLACK: Depth = 3;
    /// Base per-ply multiplier of the singular-beta margin.
    pub SE_BETA_MULT: Value = 60;
    /// Extra singular-beta multiplier when `tt_pv && !PV`.
    pub SE_BETA_TTPV_MULT: Value = 70;
    /// Divisor of the singular-beta margin.
    pub SE_BETA_DIV: Value = 59;
    /// Divisor of the correction adjustment folded into the extension
    /// margins (`corr_adj = |correction_value| / SE_CORR_DIV`).
    pub SE_CORR_DIV: i32 = 194_822;
    /// Double-extension margin: base.
    pub SE_DOUBLE_BASE: Value = -3;
    /// Double-extension margin: PV bonus.
    pub SE_DOUBLE_PV: Value = 201;
    /// Double-extension margin: non-tt-capture discount.
    pub SE_DOUBLE_NON_TTCAP: Value = 157;
    /// Double-extension margin: tt_move_history numerator.
    pub SE_DOUBLE_TMH_NUM: i32 = 1081;
    /// Double-extension margin: deep-node (`ply > root_depth`) discount.
    pub SE_DOUBLE_DEEP: Value = 41;
    /// Triple-extension margin: base.
    pub SE_TRIPLE_BASE: Value = 72;
    /// Triple-extension margin: PV bonus.
    pub SE_TRIPLE_PV: Value = 306;
    /// Triple-extension margin: non-tt-capture discount.
    pub SE_TRIPLE_NON_TTCAP: Value = 188;
    /// Triple-extension margin: tt_pv bonus.
    pub SE_TRIPLE_TTPV: Value = 84;
    /// Triple-extension margin: deep-node discount.
    pub SE_TRIPLE_DEEP: Value = 45;
    /// Multicut tt_move_history dock: base (`<< MULTICUT_TMH_BASE + MULTICUT_TMH_DEPTH·depth`).
    pub MULTICUT_TMH_BASE: i32 = -442;
    /// Multicut tt_move_history dock: depth multiplier.
    pub MULTICUT_TMH_DEPTH: i32 = -108;

    // --- tt_move_history node-result update (design/10 §3.10; Wave-D item 14) ---

    /// tt_move_history bonus when the best move was the ttMove (non-PV nodes).
    pub TMH_HIT: i32 = 792;
    /// tt_move_history malus when the best move was NOT the ttMove (non-PV nodes).
    pub TMH_MISS: i32 = -779;

    // --- Sibling depth reduction on alpha raise (design/10 §3.9; Wave-D item 15) ---

    /// A quiet alpha-raise (no cutoff) at `SIBLING_DEPTH_MIN < depth <
    /// SIBLING_DEPTH_MAX` searches remaining siblings two plies shallower.
    pub SIBLING_DEPTH_MIN: Depth = 2;
    /// Upper (exclusive) depth of the sibling depth reduction.
    pub SIBLING_DEPTH_MAX: Depth = 13;

    // --- cutoffCnt reduction gates (design/10 §3.7; Wave-D item 15) ---

    /// The child cutoff count `(ss+1).cutoff_cnt` must exceed this to add the
    /// cutoffCnt reduction surcharge.
    pub RED_CUTOFFCNT_GATE: i32 = 1;
    /// Above this child cutoff count, the surcharge adds RED_CUTOFFCNT_MANY.
    pub RED_CUTOFFCNT_MANY_GATE: i32 = 2;
    /// Base cutoffCnt reduction surcharge (child slot has been failing high).
    pub RED_CUTOFFCNT_BASE: i32 = 236;
    /// Extra surcharge when the child cutoff count exceeds RED_CUTOFFCNT_MANY_GATE.
    pub RED_CUTOFFCNT_MANY: i32 = 1079;
    /// Extra surcharge when the current node is an all-node (`!PV && !cut_node`).
    pub RED_CUTOFFCNT_ALLNODE: i32 = 1143;

    // --- ProbCut (N13, design/10 §3.4) ---
    //
    // A shallow, cheap search proving a *capture* beats `probcut_beta` lets
    // the node fail high early; a confirmed cutoff stores BOUND_LOWER and
    // returns `value - (probcut_beta - beta)`.

    /// Flat base of the ProbCut beta margin.
    pub PROBCUT_BETA_BASE: Value = 214;
    /// Improving discount of the ProbCut beta margin.
    pub PROBCUT_BETA_IMPROVING: Value = 59;
    /// Minimum depth at which ProbCut runs (`depth >= PROBCUT_MIN_DEPTH`).
    pub PROBCUT_MIN_DEPTH: Depth = 3;
    /// Depth subtracted (with `improving`) for the ProbCut re-verification search;
    /// also the child-depth slack of the in-check small ProbCut (N14).
    pub PROBCUT_DEPTH_SUB: Depth = 4;
    /// The in-check small-ProbCut beta margin (N14): `probcut_beta = beta + 428`.
    pub PROBCUT_CHECK_BETA: Value = 428;

    // --- Correction-history couplings (design/10 §4.6) ---
    //
    // `correction_value` (the UNSCALED weighted correction sum, design/09
    // §3.10) widens the RFP margin under eval unreliability, reduces a touch
    // less, and tightens the singular margins.

    /// Divisor of the RFP margin correction-widening term.
    pub RFP_CORR_DIV: i32 = 182_069;
    /// Divisor of the reduction correction-slackening term.
    pub RED_CORR_DIV: i32 = 26_131;

    // --- Hindsight depth adjustments (N6, design/10 §3.4) ---
    //
    // If the parent reduced us hard yet the position did not turn their way,
    // search one ply deeper; a moderate parent reduction plus a healthy
    // combined eval sum searches one ply shallower.

    /// `prior_reduction >= HINDSIGHT_RED_UP && !opponent_worsening` → `depth += 1`.
    pub HINDSIGHT_RED_UP: i32 = 3;
    /// `prior_reduction >= HINDSIGHT_RED_DOWN && depth >= 2 && eval-sum big` →
    /// `depth -= 1`.
    pub HINDSIGHT_RED_DOWN: i32 = 2;
    /// The `ss.static_eval + (ss-1).static_eval` threshold of the depth-down arm.
    pub HINDSIGHT_EVAL_SUM: Value = 173;

    // --- TT-cutoff hardening (N6b, design/10 §3.4) ---
    //
    // Beyond the base non-PV cutoff: require the stored bound direction to
    // AGREE with the expected node type at shallow depth, re-verify a deep
    // cutoff against the ttMove's child entry, and age out an entry whose
    // bound points the wrong way.

    /// Shallow cutoffs (`depth <= TTCUT_AGREE_DEPTH`) require the stored
    /// direction to match the node type.
    pub TTCUT_AGREE_DEPTH: Depth = 4;
    /// At/above this depth, a candidate cutoff is re-verified against the
    /// ttMove's child TT entry (one make/probe/unmake).
    pub TTCUT_VERIFY_DEPTH: Depth = 7;
    /// A wrong-direction bound at `depth > TTCUT_PENALIZE_DEPTH` costs the
    /// stored entry one ply of depth (`penalize(1)`).
    pub TTCUT_PENALIZE_DEPTH: Depth = 5;

    // --- Fail-low parent history bonus (N-result, design/10 §3.10) ---
    //
    // A fail low with no best move under a parent quiet rewards the parent's
    // move: `bonus_scale` accumulates signals, `scaled_bonus` is distributed
    // to the parent conthist / main / pawn histories. Under a parent capture,
    // flat-reward capture history instead.

    /// Flat base (negative) of `bonus_scale`.
    pub FL_SCALE_BASE: i32 = -245;
    /// Divisor of the `(ss-1).stat_score` term subtracted from `bonus_scale`.
    pub FL_STATSCORE_DIV: i32 = 98;
    /// Depth multiplier of the `min(FL_DEPTH_MULT·depth, FL_DEPTH_CAP)` term.
    pub FL_DEPTH_MULT: i32 = 59;
    /// Cap of the depth term.
    pub FL_DEPTH_CAP: i32 = 430;
    /// `bonus_scale` bump when the parent tried many moves (`(ss-1).move_count > 8`).
    pub FL_LATE_PARENT: i32 = 191;
    /// `bonus_scale` bump when `!ss.in_check && best_value <= ss.static_eval - 103`.
    pub FL_BELOW_EVAL: i32 = 143;
    /// Margin of the below-own-eval bump.
    pub FL_BELOW_EVAL_MARGIN: Value = 103;
    /// `bonus_scale` bump when `!(ss-1).in_check && best_value <= -(ss-1).static_eval - 78`.
    pub FL_BELOW_PARENT_EVAL: i32 = 151;
    /// Margin of the below-parent-eval bump.
    pub FL_BELOW_PARENT_EVAL_MARGIN: Value = 78;
    /// Depth multiplier of `scaled_bonus`.
    pub FL_BONUS_MULT: i32 = 141;
    /// Depth-linear subtrahend of `scaled_bonus`.
    pub FL_BONUS_SUB: i32 = 82;
    /// Cap of the `scaled_bonus` depth factor.
    pub FL_BONUS_CAP: i32 = 1472;
    /// Parent-conthist weight of the fail-low bonus (`scaled_bonus·236/16384`).
    pub FL_CONTHIST_NUM: i32 = 236;
    /// Main-history weight of the fail-low bonus (`scaled_bonus·234/32768`).
    pub FL_MAIN_NUM: i32 = 234;
    /// Pawn-history weight of the fail-low bonus (`scaled_bonus·322/8192`).
    pub FL_PAWN_NUM: i32 = 322;
    /// Flat capture-history bonus on a fail low under a parent capture.
    pub FL_CAPTURE_BONUS: i32 = 901;

    // --- Eval-diff parent nudge (N8, design/10 §3.4) ---
    //
    // Grade the opponent's last quiet by the static-eval swing it produced
    // (history nudges indexed by the *opponent* color).

    /// Low clamp of the eval swing feeding the eval-diff nudge.
    pub EVALDIFF_CLAMP_LO: Value = -183;
    /// High clamp of the eval swing.
    pub EVALDIFF_CLAMP_HI: Value = 180;
    /// Flat offset added to the clamped eval swing.
    pub EVALDIFF_OFFSET: Value = 62;
    /// Main-history multiplier of the eval-diff nudge.
    pub EVALDIFF_MAIN_MULT: i32 = 10;
    /// Pawn-history multiplier of the eval-diff nudge (TT-miss, non-pawn mover).
    pub EVALDIFF_PAWN_MULT: i32 = 13;

    // --- Time management: soft stop (design/12 §3.9) ---
    //
    // The soft-stop factors multiply into the plain `optimum` budget; the
    // formula SHAPES are documented at the use site in `search::soft_stop`.

    /// reduction factor 2, stability lerp x0: best-move-stability in iterations
    /// `root_depth − last_best_move_depth`, mapped LO→[TIME_REDUCTION_MIN].
    /// NOTE: window shifted down from SF's [5, 18] — we complete ~3.5 fewer
    /// iterations at equal time, so SF's endpoints read us as "unstable";
    /// the y-range is untouched.
    pub TIME_REDUCTION_STABILITY_LO: f64 = 2.0;
    /// reduction: stability lerp x1, mapped HI→[TIME_REDUCTION_MAX].
    pub TIME_REDUCTION_STABILITY_HI: f64 = 15.0;
    /// reduction: constant term of `(1.48 + prev_time_reduction) / (2.157 ·
    /// time_reduction)` — partially gives back last move's stability savings.
    pub REDUCTION_BASE: f64 = 1.48;
    /// reduction: divisor coefficient scaling `time_reduction`.
    pub REDUCTION_DEN_COEFF: f64 = 2.157;

    /// bestMoveInstability: per-iteration aging divisor of the accumulated
    /// root best-move-change count (SF: `totBestMoveChanges /= 2`).
    /// NOTE: 2.5, not SF's 2 — we complete ~3.3 fewer iterations at equal
    /// time, so the same wobble reads ~10x hotter through /2; 2.5 matches
    /// SF's effective weight at our measured stability.
    pub INSTABILITY_CHANGES_DECAY: f64 = 2.5;

    // (Stop cadence: CHECK_TIME_CALLS is a plain const above the macro — its
    // poll mask needs a compile-time power-of-two guarantee, and cadence is
    // not a strength knob.)

    // --- History updates: node-result bonus/malus (design/09 §3.9) ---
    //
    // `update_all_stats` computes one `bonus` (for the best move and agreeing
    // tables) and one `malus` (for tried non-best moves) per fail-high node,
    // then distributes them through the per-table weights below. The shared
    // `/1024` denominator is [`STAT_WEIGHT_DEN`].

    /// Depth multiplier of the stat bonus `min(STAT_BONUS_DEPTH_MULT·depth −
    /// STAT_BONUS_DEPTH_SUB, STAT_BONUS_MAX)`.
    pub STAT_BONUS_DEPTH_MULT: i32 = 134;
    /// Depth-linear subtrahend of the stat bonus.
    pub STAT_BONUS_DEPTH_SUB: i32 = 79;
    /// Ceiling of the stat bonus.
    pub STAT_BONUS_MAX: i32 = 1572;
    /// Extra bonus when the best move equals the TT move (it was predicted).
    pub STAT_BONUS_TTMOVE: i32 = 382;
    /// Divisor of the parent `stat_score` contribution to the bonus.
    pub STAT_BONUS_STATSCORE_DIV: i32 = 30;
    /// Depth multiplier of the stat malus `min(STAT_MALUS_DEPTH_MULT·depth −
    /// STAT_MALUS_DEPTH_SUB, STAT_MALUS_MAX)`.
    pub STAT_MALUS_DEPTH_MULT: i32 = 1005;
    /// Depth-linear subtrahend of the stat malus.
    pub STAT_MALUS_DEPTH_SUB: i32 = 205;
    /// Ceiling of the stat malus.
    pub STAT_MALUS_MAX: i32 = 2218;
    /// Divisor of the non-PV breadth multiplier `bonus += bonus·tried /
    /// STAT_BREADTH_DIVISOR`. `i64`: the product is widened to avoid 32-bit
    /// overflow (plain i64 truncation).
    pub STAT_BREADTH_DIVISOR: i64 = 256;
    /// Best-quiet bonus weight (`bonus·824/1024`).
    pub QUIET_BEST_BONUS_NUM: i32 = 824;
    /// Geometric quiet-malus start weight (`malus·1136/1024`), decayed by
    /// [`QUIET_MALUS_DECAY_NUM`] per non-best quiet — earliest punished hardest.
    pub QUIET_MALUS_START_NUM: i32 = 1136;
    /// Per-move decay of the geometric quiet malus (`m·956/1024` each step).
    pub QUIET_MALUS_DECAY_NUM: i32 = 956;
    /// Best-capture bonus weight (`bonus·1366/1024`).
    pub CAPTURE_BEST_BONUS_NUM: i32 = 1366;
    /// Non-best-capture malus weight (`−malus·1518/1024`).
    pub CAPTURE_MALUS_NUM: i32 = 1518;
    /// Refutation malus weight for the parent's refuted first-try quiet
    /// (`−malus·683/1024`).
    pub REFUTATION_MALUS_NUM: i32 = 683;

    // --- History updates: per-quiet-table weights (design/09 §3.9) ---

    /// Low-ply history update weight (`bonus·663/1024`, plies < 5 only).
    pub QUIET_LOW_PLY_UPDATE_NUM: i32 = 663;
    /// Continuation-history update weight (`bonus·820/1024`).
    pub QUIET_CONT_UPDATE_NUM: i32 = 820;
    /// Pawn-history update weight for bonuses (`bonus·1038/1024`).
    pub PAWN_HIST_BONUS_NUM: i32 = 1038;
    /// Pawn-history update weight for maluses (`bonus·525/1024`) — maluses are
    /// roughly halved relative to bonuses.
    pub PAWN_HIST_MALUS_NUM: i32 = 525;
    /// The pawn-history bonus/malus split point: `> -7` takes the bonus weight,
    /// `<= -7` the malus weight (deliberately not `> 0`).
    pub PAWN_HIST_ASYMMETRY_THRESHOLD: i32 = -7;

    // --- History updates: continuation chain (design/09 §3.9) ---

    /// Flat bonus added to the offset-1 continuation update (even for maluses).
    pub CONT_OFFSET1_FLAT_BONUS: i32 = 71;
    /// In check, only offsets `<= 2` update (the position before the check bears
    /// little relation to the current one).
    pub CONT_IN_CHECK_MAX_OFFSET: usize = 2;

    // --- Correction history: read combiner (design/09 §3.10) ---
    //
    // `correction_value` blends the learned static-eval corrections in
    // 1/131072-pawn units; [`CORR_VALUE_DIVISOR`] converts back to centipawns.

    /// Weight on the pawn-structure correction.
    pub CORR_PAWN_WEIGHT: i32 = 13345;
    /// Weight on the minor-piece correction.
    pub CORR_MINOR_WEIGHT: i32 = 9280;
    /// Weight on each non-pawn (white/black) correction.
    pub CORR_NONPAWN_WEIGHT: i32 = 11840;
    /// Weight on the summed continuation corrections (offsets 2 and 4).
    pub CORR_CONT_WEIGHT: i32 = 8363;
    /// Flat continuation-correction prior used after a null previous move.
    pub CORR_CONT_NULL_PRIOR: i32 = 64549;

    // --- Correction history: update weights (design/09 §3.10) ---

    /// Minor-piece correction update weight (`bonus·152/128`).
    pub CORR_MINOR_UPDATE_NUM: i32 = 152;
    /// Non-pawn (white and black) correction update weight (`bonus·186/128`).
    pub CORR_NONPAWN_UPDATE_NUM: i32 = 186;
    /// Offset-2 continuation-correction update weight (`bonus·136/128`).
    pub CORR_CONT2_UPDATE_NUM: i32 = 136;
    /// Offset-4 continuation-correction update weight (`bonus·68/128`).
    pub CORR_CONT4_UPDATE_NUM: i32 = 68;

    // --- Correction history: bonus computation (call site, design/10 §4.6) ---
    //
    // The gated, clamped bonus fed to `update_correction_history`.

    /// Depth multiplier when a best move exists (`err·depth·12/128`).
    pub CORR_BONUS_DEPTH_MULT_BESTMOVE: i32 = 12;
    /// Depth multiplier on a fail-low with no best move (`err·depth·18/128`).
    pub CORR_BONUS_DEPTH_MULT_NOMOVE: i32 = 18;
    /// Symmetric clamp on the raw correction bonus (`CORRECTION_HISTORY_LIMIT/4`).
    pub CORR_BONUS_CLAMP: i32 = 256;
    /// Numerator of the outer correction-bonus scaling (`·1114/1024`).
    pub CORR_BONUS_SCALE_NUM: i32 = 1114;

    // --- SF-parity structural toggles: RESOLVED. QS_UPREP and
    // RED_TTPV_TTMISS_REBATE measured worse ON and were deleted (divergence
    // notes at the use sites); FOLLOW_PV_MODE went with the followPV
    // apparatus. PROBCUT_IN_VERIF is a de facto no-op (the stored-value
    // clause blocks the ProbCut block in both engines) — do not re-propose.

    // --- Implicit-coefficient batch 2: terms that carried an invisible
    // coefficient on high-traffic paths, wired at identity. ---

    /// Beta weight (in 1024ths) of the stand-pat fail-high softening
    /// `((1024−W)·best + W·beta) / 1024` — applied to non-decisive stand
    /// pats at or above beta (§3.8 step 5).
    pub QS_STANDPAT_BETA_W: Value = 557;
    /// Beta weight (in 1024ths) of the node-exit fail-high softening
    /// (§3.8 step 11); same folded shape as the stand-pat blend.
    pub QS_FAILHIGH_BETA_W: Value = 543;
    /// Numerator of the verification's NMP-disable window `nmp_min_ply =
    /// ply + NUM·(depth - R) / DEN`: the verification cannot recursively
    /// null-move its way back to the same false cutoff.
    pub NMP_VERIFY_NUM: Depth = 3;
    /// The singular block's negative extension when the non-excluded search
    /// already proved `tt_value >= beta` (the ttMove is expected to fail
    /// high anyway; search it SHALLOWER). SF's bare `-3`.
    pub SE_NEG_EXT_TT_FAILHIGH: Depth = -3;
    /// The singular block's negative extension at a cut node that is neither
    /// singular nor tt-fail-high. SF's bare `-2`.
    pub SE_NEG_EXT_CUTNODE: Depth = -2;
    /// LMR entry gate: minimum depth (`depth >= 2 && move_count > 1`).
    pub LMR_MIN_DEPTH: Depth = 2;
    /// LMR entry gate: minimum move count (strictly greater-than).
    pub LMR_MIN_MOVECOUNT: i32 = 1;
    /// Numerator of the main-history term in the history-buys-lmr_depth
    /// formula `history += NUM·main / DEN` (SF writes `64·h/32`, deliberately
    /// leaving 1/32 granularity for the tuner).
    pub LMRDEPTH_MAIN_NUM: i32 = 64;

}

// ─── D50 constified handles ───
//
// Same `NAME()` spelling as the tunables above, but compile-time constant in
// EVERY build: none is on the SPSA surface (D50: a constant becomes a
// tunable when a campaign wants it, not by default; a num/den pair is ONE
// degree of freedom). Promote one back by moving its line into `tunables!`.
crate::constants! {

    /// Period of the lazy-SMP aspiration jitter (design/13 §3.1): helper `t`
    /// adds `t % ASPIRATION_JITTER_PERIOD` to the base delta. Vacuous at
    /// `thread_idx == 0`, keeping Threads=1 bit-identical (D24).
    pub ASPIRATION_JITTER_PERIOD: usize = 8;
    /// Denominator of the per-re-search widening (~x1.34 per fail).
    pub ASPIRATION_WIDEN_DEN: Value = 128;
    /// Denominator of the search-again depth damping.
    pub ASPIRATION_SEARCH_AGAIN_DEN: Depth = 4;
    /// Denominator of the improving/opponent-worsening margin discounts.
    pub RFP_TREND_DEN: Value = 1024;
    /// Denominator of the log reduction table (a power of two — exactly
    /// representable, so the built table is bit-reproducible).
    pub RED_TABLE_DEN: i32 = 128;
    /// Not-improving surcharge denominator (the `/512` of the term above).
    pub RED_NON_IMPROVING_DEN: i32 = 512;
    /// The application unit of the reduction accumulator: reductions are
    /// carried in 1024ths of a ply, divided only when they become a depth.
    pub RED_UNIT: i32 = 1024;
    /// Divisor of the capture `stat_score` victim term.
    pub STATSCORE_CAPT_DEN: i32 = 128;
    /// Divisor of the reduction dock (a power of two — the truncation toward zero
    /// on a negative `stat_score` is intended, not to be "fixed" to floor).
    pub RED_STATSCORE_DEN: i32 = 4096;
    /// Plies the internal iterative reduction removes (was a bare
    /// `depth -= 1`).
    pub IIR_REDUCTION: Depth = 1;
    /// Divisor base of the LMP threshold `(…) / (LMP_IMPROVING_DEN -
    /// improving)`: at 2 a not-improving node prunes at half the move count.
    pub LMP_IMPROVING_DEN: i32 = 2;
    /// Denominator of the singular verification depth (see
    /// [`SE_VERIFY_DEPTH_NUM`]).
    pub SE_VERIFY_DEPTH_DEN: Depth = 2;
    /// Base extension granted to a confirmed-singular move, before the
    /// double/triple bonuses: `extension = SE_EXT_BASE + (double) + (triple)`.
    pub SE_EXT_BASE: Depth = 1;
    /// Extra depth the WHOLE node gains when its ttMove proves singular;
    /// distinct from [`SE_EXT_BASE`], which extends only the singular move.
    pub SE_NODE_DEEPEN: Depth = 1;
    /// Depth surcharge for entering the singular test at a ttPv node
    /// (`depth >= SE_MIN_DEPTH + SE_TTPV_DEPTH_BONUS * tt_pv`).
    pub SE_TTPV_DEPTH_BONUS: Depth = 1;
    /// Depth subtracted from the ProbCut verification search when improving
    /// (`depth - PROBCUT_DEPTH_SUB - PROBCUT_IMPROVING_SUB * improving`).
    pub PROBCUT_IMPROVING_SUB: Depth = 1;
    /// Depth bonus given to the LMR scout at a PV node, applied to both the
    /// reduced depth and its deeper cap.
    pub LMR_PV_DEPTH_BONUS: Depth = 1;
    /// Floor of the LMR scout depth (`.max(LMR_MIN_SCOUT_DEPTH)`), which
    /// keeps a heavily-reduced scout from collapsing into qsearch.
    pub LMR_MIN_SCOUT_DEPTH: Depth = 1;
    /// Depth step added to the re-search when the LMR scout came back well
    /// above `best_value` (`new_depth += LMR_DEEPER_STEP * do_deeper`).
    pub LMR_DEEPER_STEP: Depth = 1;
    /// Depth step removed from the re-search when the LMR scout came back
    /// barely above `best_value`.
    pub LMR_SHALLOWER_STEP: Depth = 1;
    /// Depth docked from the skipped-LMR full-depth search at each of the two
    /// reduction thresholds (`- LMR_FULLDEPTH_DOCK_STEP` per threshold met).
    pub LMR_FULLDEPTH_DOCK_STEP: Depth = 1;
    /// Minimum `new_depth` for the SECOND full-depth dock to apply.
    pub LMR_DOCK2_MIN_DEPTH: Depth = 2;
    /// Depth slack allowed on a TT cutoff when the stored value is a lower
    /// bound: `tt_depth > depth - TTCUT_BOUND_SLACK * (tt_value <= beta)`.
    pub TTCUT_BOUND_SLACK: Depth = 1;
    /// Extension level at or above which a node stops counting child cutoffs
    /// (`cutoff_cnt += (extension < CUTOFFCNT_EXT_GATE || PV)`).
    pub CUTOFFCNT_EXT_GATE: Depth = 2;
    /// Shared `/1024` divisor of the capture-history reads in both capture
    /// gates (the pruning read weight only).
    pub CAPT_HIST_DEN: i32 = 1024;
    /// Double-extension margin: tt_move_history divisor.
    pub SE_DOUBLE_TMH_DEN: i32 = 117_824;
    /// `is_shuffling`: minimum halfmove clock.
    pub SHUFFLE_RULE50: i32 = 10;
    /// `is_shuffling`: minimum plies since the last null move.
    pub SHUFFLE_PLIES_FROM_NULL: i32 = 6;
    /// `is_shuffling`: minimum ply.
    pub SHUFFLE_MIN_PLY: usize = 20;
    /// Parent-conthist divisor.
    pub FL_CONTHIST_DEN: i32 = 16384;
    /// Main-history divisor.
    pub FL_MAIN_DEN: i32 = 32768;
    /// Pawn-history divisor.
    pub FL_PAWN_DEN: i32 = 8192;
    /// Single-reply cap on the soft time budget (`min(561.7, total_time)`,
    /// design/12 §3.9): with one legal move the result cannot change the move
    /// played — a viewer-experience cap. TM units (ms, or nodes under
    /// `nodestime`).
    pub ONE_REPLY_TOTAL_TIME_CAP: f64 = 561.7;
    /// DIS-TM: the bar, in internal cp, on REMAINING predicted movement (the
    /// head's root prediction minus what the search has already moved the
    /// eval); the arm buys time only above this. Inert unless
    /// `ARCHON_DIS_TM_C` is set.
    pub DIS_TM_T0: f64 = 40.0;
    /// DIS-TM pivot: the REMAINING-movement value treated as neutral; the
    /// MEASURED median of REMAINING itself, so the arm is mean-neutral by
    /// construction. NOTE: pivot on the quantity the comparison actually
    /// uses — pivoting on the parent distribution `dismag` made the arm a
    /// net time-saver.
    pub DIS_TM_PIVOT: f64 = 127.0;
    /// fallingEval factor 1: constant term of `(11.87 + 2.21·Δavg + 1.0·Δiter)
    /// / 100`, clamped to `[0.572, 1.708]` — think longer when the score drops.
    pub FALLING_EVAL_BASE: f64 = 11.87;
    /// fallingEval: weight on `best_previous_average_score − best_value`.
    pub FALLING_EVAL_PREV_AVG_COEFF: f64 = 2.21;
    /// fallingEval: weight on `iter_value[iter_idx] − best_value`.
    pub FALLING_EVAL_ITER_COEFF: f64 = 1.0;
    /// fallingEval: the `/100` scale.
    pub FALLING_EVAL_DIV: f64 = 100.0;
    /// fallingEval lower clamp.
    pub FALLING_EVAL_MIN: f64 = 0.572;
    /// fallingEval upper clamp.
    pub FALLING_EVAL_MAX: f64 = 1.708;
    /// reduction: `time_reduction` lower clamp (and lerp y at low stability).
    pub TIME_REDUCTION_MIN: f64 = 0.65;
    /// reduction: `time_reduction` upper clamp (and lerp y at high stability).
    pub TIME_REDUCTION_MAX: f64 = 1.55;
    /// bestMoveInstability factor 3: constant term of `1.096 + 2.29 ·
    /// tot_best_move_changes / threads` — think longer when the best move
    /// keeps flipping.
    pub INSTABILITY_BASE: f64 = 1.096;
    /// bestMoveInstability: weight on the per-thread-averaged change count.
    pub INSTABILITY_CHANGES_COEFF: f64 = 2.29;
    /// highBestMoveEffort factor 4, effort lerp x0 (per-100k node units):
    /// `nodes_effort = effort · 100000 / max(1, nodes)` mapped 79219→0.924.
    pub EFFORT_LERP_LO: f64 = 79_219.0;
    /// highBestMoveEffort: effort lerp x1, mapped 101822→0.71.
    pub EFFORT_LERP_HI: f64 = 101_822.0;
    /// highBestMoveEffort: scale at low effort (also the upper clamp).
    pub EFFORT_SCALE_AT_LO: f64 = 0.924;
    /// highBestMoveEffort: scale at high effort (also the lower clamp).
    pub EFFORT_SCALE_AT_HI: f64 = 0.71;
    /// highBestMoveEffort: the per-100k fixed-point scale of the effort ratio.
    pub EFFORT_UNIT_SCALE: i64 = 100_000;
    /// Soft-stop `increase_depth`/`search_again_counter` threshold: the next
    /// iteration deepens honestly only while at most this fraction of the soft
    /// budget is spent (design/12 §3.9).
    pub INCREASE_DEPTH_FRACTION: f64 = 0.50;
    /// Soft-stop mate early-stop: stop once every PV line mates in `<=` this
    /// many plies (`rootMoves[multiPV-1].score >= mate_in(3)`, design/12 §3.9).
    pub SOFT_STOP_MATE_PLIES: Depth = 3;
    /// Soft-stop mate early-stop: stop once we are mated in exactly this many
    /// plies (`rootMoves[0].score == mated_in(2)`, design/12 §3.9).
    pub SOFT_STOP_MATED_PLIES: Depth = 2;
    /// Flat epsilon shaved off every enforced *wall-clock* bound
    /// (`maximum()`, `movetime`) to cover the unwind-and-print tail after
    /// `stop`; deliberately NOT applied to the `nodes` budget (an exactness
    /// bound, not a forfeit risk). TM units.
    pub CHECK_TIME_EPSILON_MS: i64 = 15;
    /// `info` throttle (design/12 §3.13): below this many nodes the MultiPV
    /// block is published only when the iteration's LAST line finishes;
    /// above it, after every line.
    pub INFO_THROTTLE_NODES: u64 = 10_000_000;
    /// Shared `/1024` denominator of every history-update weight below.
    pub STAT_WEIGHT_DEN: i32 = 1024;
    /// Divisor of the continuation-history update `bonus·weight·mult / 131072`.
    /// `i64`: the triple product is widened before the divide.
    pub CONT_UPDATE_DIVISOR: i64 = 131072;
    /// Divisor turning the correction sum back into a centipawn adjustment.
    pub CORR_VALUE_DIVISOR: i32 = 131072;
    /// Shared `/128` denominator of the correction-update weights below.
    pub CORR_UPDATE_DEN: i32 = 128;
    /// Divisor of the raw correction bonus.
    pub CORR_BONUS_DIVISOR: i32 = 128;
    /// Denominator of the outer correction-bonus scaling.
    pub CORR_BONUS_SCALE_DEN: i32 = 1024;
    /// Denominator of the verification window fraction.
    pub NMP_VERIFY_DEN: Depth = 4;
    /// Denominator of the main-history lmr_depth term.
    pub LMRDEPTH_MAIN_DEN: i32 = 32;
    /// Additive offset of the log reduction table:
    /// `reductions[i] = (NUM/DEN)·ln(i) + OFFSET/DEN`. The ONLY new degree
    /// of freedom the product form `f(d)·f(mn)` admits (per-axis multipliers
    /// are redundant with RED_TABLE_NUM); an additive constant weighs depth
    /// against move count independently. 0 = identity.
    pub RED_TABLE_OFFSET: i32 = 0;
    /// Multiplier on `failed_high_cnt` in the aspiration loop: each
    /// consecutive root fail-high re-searches this many plies shallower.
    pub ASPIRATION_FAILHIGH_DEPTH_MULT: Depth = 1;
    /// Beta weight of the MAIN-search fail-high softening `(best·depth +
    /// W·beta) / (depth + W)` — the search() analogue of the qsearch blend.
    pub FH_SOFTEN_BETA_W: i32 = 1;
    /// The sibling depth-reduction step (`depth -= 2` on a non-cutoff alpha
    /// raise inside the SIBLING_DEPTH_MIN..MAX window).
    pub SIBLING_DEPTH_STEP: Depth = 2;
}
