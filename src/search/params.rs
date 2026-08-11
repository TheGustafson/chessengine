//! Tuned search parameters — every SPSA-movable constant, named and
//! greppable in one file (D20; conventions: design/01 §3.8).
//!
//! Values are SF-current starting points from Stockfish commit `eca43a97`
//! unless noted; structural constants live with their types.
//!
//! Naming: `TECHNIQUE_ROLE`, grouped by technique; a `*_DEN` / `*_DIV*` handle
//! is the denominator of the like-named numerator. A constant documents only
//! what its name and section heading do not. Any bare numeric literal in
//! `search/`, `movepick.rs` or `see.rs` a tuner could move is a defect.

use crate::types::{Depth, Value};

/// Per-thread vote tempo (design/13 §3.8): each thread votes for its own best
/// move with weight `score - min_score + VOTE_TEMPO`. Value units.
pub const VOTE_TEMPO: crate::types::Value = 14;
/// Per-offset weights of the continuation-history update, offsets 1..=6
/// (÷[`CONT_UPDATE_DIVISOR`] with the bonus).
pub const CONT_OFFSET_WEIGHTS: [i32; 6] = [1040, 780, 300, 537, 129, 423];
/// Indexed by the count of already-positive entries: agreement amplifies.
pub const CONT_CONSISTENCY_MULT: [i32; 7] = [96, 113, 101, 105, 127, 121, 126];

/// `lmr_depth += history / LMR_DIVISOR[min(depth,16)-1]`, where `history` is
/// the quiet's combined continuation/pawn score plus `2·main_history`.
pub const LMR_DIVISOR: [i32; 16] = [
    3307, 2930, 2874, 2818, 3215, 3225, 3224, 2782, 2858, 2919, 3088, 3275, 3180, 2868, 3006, 3599,
];

/// Clock-poll cadence, in search calls. Not a tunable: the poll mask
/// `x & (CALLS - 1)` needs a compile-time power-of-two guarantee.
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
    /// NOT SF's 10588 — mss is a SQUARED eval quantity in our 0.64x frame, so
    /// it scales by `r`, not `r²`. NOTE: ~3000 has a catastrophic root
    /// fail-high tail; never lower this off a corpus mean.
    pub ASPIRATION_MSS_DIVISOR: i64 = 6776;
    /// `optimism = NUM·avg / (|avg| + DEN_BASE)` at aspiration setup.
    pub OPTIMISM_NUM: Value = 75;
    pub OPTIMISM_DEN_BASE: Value = 45;
    /// Per-re-search widening `delta += delta·NUM/128` (~x1.34 per fail).
    pub ASPIRATION_WIDEN_NUM: Value = 44;
    /// Search-again depth damping `NUM·(search_again_counter + 1) / 4`.
    pub ASPIRATION_SEARCH_AGAIN_NUM: Depth = 3;

    // --- Move ordering: capture scoring (design/09 §3.7) ---

    /// `MVV_FACTOR·victim + capt_hist`; no LVA term (SEE prices the attacker).
    pub CAPTURE_MVV_FACTOR: Value = 7;

    /// Recapture discount in the QUIESCENCE/ProbCut capture score, in 16ths of
    /// the attacker's value, when the destination is defended and only where
    /// `depth <= DEPTH_QS`. NOTE: 56 = half a recapture, not the principled
    /// 112 — a full discount pushes defended captures past
    /// `QS_LATE_MOVE_LIMIT`, which DELETES them rather than reordering them.
    pub QS_CAPTURE_DEFENDED_NUM: Value = 56;

    /// Penalty on a quiet that loses material by static exchange
    /// (`!see_ge(m, 0)`), guarded on the destination being enemy-attacked and
    /// on `value > GOOD_QUIET_THRESHOLD`. MUST stay CLAMPED at
    /// `GOOD_QUIET_THRESHOLD + 1` so it can only reorder: unclamped it becomes
    /// a second, depth-blind copy of the move loop's quiet SEE prune and loses
    /// games. Value unresolved inside 24576..98304; SPSA owns it.
    pub QUIET_SEE_PENALTY: Value = 49152;

    // --- Move ordering: good/bad capture split (design/09 §3.6 stage 3) ---

    /// Divisor of the split gate `see_ge(m, -score / DIVISOR)`.
    pub GOOD_CAPTURE_SEE_DIVISOR: Value = 18;

    // --- Move ordering: quiet scoring (design/09 §3.7) ---

    pub QUIET_MAIN_HISTORY_WEIGHT: Value = 2;
    pub QUIET_PAWN_HISTORY_WEIGHT: Value = 2;
    /// Quiet-check bonus, gated by `see_ge(m, CHECK_SEE_MARGIN)`.
    pub QUIET_CHECK_BONUS: Value = 16384;
    pub QUIET_CHECK_SEE_MARGIN: Value = -75;
    /// Fleeing a square attacked by a cheaper enemy earns `FACTOR·piece_value`;
    /// entering one costs the same.
    pub QUIET_THREAT_FACTOR: Value = 20;
    /// Low-ply history weight (plies 0..4 only, scaled by `/ (1 + ply)`).
    pub QUIET_LOW_PLY_WEIGHT: Value = 8;

    // --- Move ordering: quiet good/bad split + sort (design/09 §3.6) ---

    /// Quiets strictly above this are "good" (before bad captures); rest last.
    pub GOOD_QUIET_THRESHOLD: Value = -14000;
    /// Quiet partial-sort limit `FACTOR·depth`.
    pub QUIET_SORT_DEPTH_FACTOR: Value = -3560;

    // --- Move ordering: evasion scoring (design/09 §3.7) ---

    /// Lifts capturing evasions above quiet ones; `2^28` dwarfs any history.
    pub EVASION_CAPTURE_BONUS: Value = 1 << 28;

    // --- Quiescence search (design/07 §3.8) ---

    /// Step 5: skip a capture that cannot lift `static_eval + MARGIN` > alpha.
    pub QS_FUTILITY_MARGIN: Value = 335;
    /// Step 7 hard floor: below this SEE bar a move is never searched.
    pub QS_SEE_FLOOR: Value = -74;
    /// Step 7: stop trying non-checking, non-recapture moves after this many.
    pub QS_LATE_MOVE_LIMIT: i32 = 2;

    // --- Razoring (design/10 §3.1; node step N9) ---

    /// Drop to qsearch below `alpha − RAZOR_BASE − RAZOR_DEPTH_MULT·depth²`.
    pub RAZOR_BASE: Value = 465;
    pub RAZOR_DEPTH_MULT: Value = 300;

    // --- Reverse futility pruning (design/10 §3.4; node step N10) ---
    // A non-PV node whose corrected static eval clears beta by a depth-scaled
    // trend-adjusted margin returns a beta-ward blend without searching.

    /// Exclusive ceiling — deep nodes never RFP, which matters for mate finding.
    pub RFP_MAX_DEPTH: Depth = 17;
    /// Multiplier `min(BASE + SLOPE·depth, CAP)`, less NO_TTHIT on a TT miss.
    pub RFP_MULT_BASE: Value = 40;
    pub RFP_MULT_SLOPE: Value = 4;
    pub RFP_MULT_CAP: Value = 80;
    pub RFP_MULT_NO_TTHIT: Value = 20;
    /// Margin shrink (÷[`RFP_TREND_DEN`] of `futility_mult`), improving / opp.
    pub RFP_IMPROVING: Value = 2934;
    pub RFP_OPP_WORSENING: Value = 343;
    /// Return blend `(W·beta + (1024−W)·eval)/1024`: ≥ beta, damps eval noise.
    pub RFP_RET_BETA_W: Value = 716;

    // --- Null move pruning (design/10 §3.4; node step N11) ---
    // Zugzwang is fenced three ways: non-pawn material on the side to move, a
    // proven-mate null value is never trusted, and at high depth a verification
    // search re-confirms with NMP disabled up to `nmp_min_ply`. The gate reads
    // `ss.static_eval`, NOT the TT-substituted `eval` that RFP uses.

    /// Bar `beta - EVAL_DEPTH·depth - EVAL_IMPROVING·improving + EVAL_BASE`.
    pub NMP_EVAL_DEPTH: Value = 14;
    pub NMP_EVAL_IMPROVING: Value = 45;
    pub NMP_EVAL_BASE: Value = 374;
    /// Dynamic reduction `R = R_BASE + depth / R_DEPTH_DIV`, in whole plies.
    pub NMP_R_BASE: Depth = 7;
    pub NMP_R_DEPTH_DIV: Depth = 3;
    /// At/above this depth a fail-high null is re-confirmed by verification.
    pub NMP_VERIFY_DEPTH: Depth = 16;

    // --- Base reduction table + `reduction()` (design/10 §3.3) ---
    // Per-worker log table `reductions[i] = int(TABLE_NUM / TABLE_DEN·ln(i))`,
    // carried in 1024ths of a ply and divided only on application.

    pub RED_TABLE_NUM: i32 = 2834;
    /// Wide-window damping `- delta·DELTA_MULT / root_delta`.
    pub RED_DELTA_MULT: i32 = 617;
    /// Not-improving surcharge `+ scale·NON_IMPROVING / NON_IMPROVING_DEN`.
    pub RED_NON_IMPROVING: i32 = 194;
    /// Flat base added to every base reduction.
    pub RED_BASE: i32 = 1027;

    // --- LMR reduction finalization + search mechanics (design/10 §3.7/§3.8) ---
    // The base [`Worker::reduction`] output `r` accretes the terms below, then
    // becomes the reduced zero-window scout depth.

    /// Flat rebase added to every finalized reduction.
    pub RED_FLAT: i32 = 714;

    /// How much worse than equality a draw is judged, from the ROOT side's view
    /// (internal units): a draw at −1 sets alpha and starves a winning sibling.
    /// Default 0 = identity.
    pub DRAW_CONTEMPT: Value = 0;
    /// `r -= MOVECOUNT·move_count`: later moves reduce slightly less.
    pub RED_MOVECOUNT: i32 = 62;
    /// Cut-node surcharge, plus NO_TTMOVE when the node has no TT move.
    pub RED_CUTNODE: i32 = 3995;
    pub RED_CUTNODE_NO_TTMOVE: i32 = 1059;
    /// Rebate `r = max(0, r - TTMOVE_SUB)`: search the ttMove near full depth.
    pub RED_TTMOVE_SUB: i32 = 2016;
    /// Cap in PLIES on how far a NEGATIVE `r` deepens past `new_depth`; PV +1.
    pub RED_MAX_DEEPER: Depth = 2;
    /// Re-search a ply DEEPER above DEEPER_MARGIN, SHALLOWER below the other.
    pub LMR_DEEPER_MARGIN: Value = 52;
    pub LMR_SHALLOWER_MARGIN: Value = 9;
    /// Non-LMR path: no TT move adds this; `r` above DOCK1 costs a ply, above
    /// DOCK2 (with `new_depth > 2`) a second.
    pub RED_FULLDEPTH_NO_TTMOVE: i32 = 1085;
    pub RED_FULLDEPTH_DOCK1: i32 = 5039;
    pub RED_FULLDEPTH_DOCK2: i32 = 5223;

    // --- LMR statScore: combined-history reduction + post-LMR bonus
    //     (design/10 §3.7/§3.8) ---
    // `ss.stat_score` is computed from PRE-move state (the child reads
    // `(ss-1).stat_score`). Integer division truncates toward zero, as in C++:
    // a negative score does NOT floor.

    /// Butterfly weight in `stat_score`; independent of the ordering weight.
    pub STATSCORE_MAIN_WEIGHT: i32 = 2;
    /// Capture `stat_score` = `CAPT_NUM·piece_value(captured) / CAPT_DEN`.
    pub STATSCORE_CAPT_NUM: i32 = 809;
    /// Dock `r -= stat_score·NUM / DEN`: positive `stat_score` searches fuller.
    pub RED_STATSCORE_NUM: i32 = 445;
    /// Flat conthist bonus (offsets 1..6) when a reduced LMR scout beats alpha.
    pub LMR_CONTHIST_BONUS: i32 = 1415;

    // --- LMR finalization: ttPv rebate / tt_capture / all-node (design/10 §3.5/§3.7) ---
    // These BALANCE the reduction. The rebate sums BASE + PV + TTVALUE
    // (`tt.value > alpha`) + TTDEPTH_BASE (`tt.depth >= depth`, plus
    // TTDEPTH_CUT again at a cut node).

    /// Provisional ttPv bump, mostly cancelled by the rebate.
    pub RED_TTPV_ENTRY: i32 = 1006;
    pub RED_TTPV_BASE: i32 = 2766;
    pub RED_TTPV_PV: i32 = 1017;
    pub RED_TTPV_TTVALUE: i32 = 838;
    pub RED_TTPV_TTDEPTH_BASE: i32 = 923;
    pub RED_TTPV_TTDEPTH_CUT: i32 = 955;
    /// Surcharge when a capturing ttMove distrusts quiet siblings.
    pub RED_TTCAPTURE: i32 = 1039;
    /// All-node scaling `r += r·NUM / (DEPTH_MULT·depth + DEPTH_BASE)`.
    pub RED_ALLNODE_NUM: i32 = 272;
    pub RED_ALLNODE_DEPTH_MULT: i32 = 256;
    pub RED_ALLNODE_DEPTH_BASE: i32 = 285;

    // --- Internal iterative reduction (N12, design/10 §3.4) ---

    /// A PV/cut node with no ttMove at this depth searches one ply shallower.
    pub IIR_MIN_DEPTH: Depth = 6;

    /// `1` = store the argmax move on a main-search fail low (default
    /// `Move::NULL`). MEASURED DEAD — leave at 0; kept as its repro path.
    pub TT_FAILLOW_MOVE: i32 = 0;

    /// `1` = the four sites reading `tt_data.mv == Move::NULL` as "unresolved"
    /// (IIR, RFP's ttMove gate, `RED_CUTNODE_NO_TTMOVE`,
    /// `RED_FULLDEPTH_NO_TTMOVE`) ignore a move on an UPPER-bound entry.
    /// REJECTED IN GAMES — never put it in an SPSA campaign; the IIR bit alone
    /// loses badly. Separate from [`TT_FAILLOW_MOVE`]: screen the two as a 2x2.
    pub TT_UPPER_PRUNE_BLIND: i32 = 0;

    // --- Move-loop pruning: late move pruning (design/10 §3.5) ---
    // Once `move_count` clears the threshold the picker stops emitting quiets.
    // The whole move-loop pruning block is fenced by `!is_loss(best_value)`
    // (mate safety) and `non_pawn_material(us) != 0`, and never runs at root.

    /// LMP threshold `(LMP_BASE + depth²) / (2 - improving)`.
    pub LMP_BASE: i32 = 3;

    /// Singular VERIFICATION depth `new_depth·NUM / DEN` — a whole extra search
    /// per candidate, so it sets both cost and verdict trustworthiness.
    pub SE_VERIFY_DEPTH_NUM: Depth = 1;

    // --- Move-loop pruning: capture futility + capture/check SEE (design/10 §3.5) ---
    // `piece_value(captured)` is 0 for en passant (SF parity); there is no
    // `!in_check` guard (the outer `!is_loss` fence keeps evasions searched);
    // and the SEE gate never prunes our LAST non-pawn material while
    // `alpha < VALUE_DRAW` — a possible stalemate resource.

    /// Exclusive lmr-depth ceiling of capture futility.
    pub CFUT_MAX_LMRDEPTH: Depth = 7;
    /// Optimistic value over the static eval: `BASE + LMRDEPTH_MULT·lmr_depth +
    /// HIST_MULT·capt_hist / CAPT_HIST_DEN`; the slope may go negative at
    /// shallow nodes, LOWERING the bound.
    pub CFUT_BASE: Value = 231;
    pub CFUT_LMRDEPTH_MULT: Value = 232;
    pub CFUT_HIST_MULT: i32 = 131;
    /// SEE margin `DEPTH_MULT·depth + HIST_MULT·capt_hist / CAPT_HIST_DEN`.
    pub CAP_SEE_DEPTH_MULT: Value = 175;
    pub CAP_SEE_HIST_MULT: i32 = 34;

    // --- Move-loop pruning: continuation-history pruning (design/10 §3.5) ---
    // Skip a quiet whose combined recent history (contHist(1) + contHist(2) +
    // pawn slot, PRE-move state) is below `-QHIST_PRUNE_MULT·depth`. Fresh
    // histories sum to −2366 — above the bar at every depth, so the gate stays
    // dormant until updates drive a quiet below it.

    pub QHIST_PRUNE_MULT: i32 = 4313;

    // --- Move-loop pruning: parent futility (design/10 §3.5) ---
    // A shallow, not-in-check node writes off a quiet whose most optimistic
    // outcome still fails to reach alpha. Gated by `!is_decisive(best_value)
    // && !is_win(bound)` so no mate score is corrupted (§8.11). `lmr_depth =
    // new_depth - r/1024`; bound = BASE + NO_BESTMOVE +
    // LMRDEPTH_MULT·lmr_depth + EVAL_ABOVE_ALPHA, each when its cond holds.

    pub PFUT_BASE: Value = 40;
    pub PFUT_NO_BESTMOVE: Value = 138;
    pub PFUT_LMRDEPTH_MULT: Value = 117;
    pub PFUT_EVAL_ABOVE_ALPHA: Value = 90;
    /// Exclusive lmr-depth ceiling of parent futility.
    pub PFUT_MAX_LMRDEPTH: Depth = 12;

    // --- Move-loop pruning: quiet SEE (design/10 §3.5) ---

    /// Margin `-QSEE_MULT·lmr_depth²`, `lmr_depth` clamped to `>= 0` first.
    pub QSEE_MULT: Value = 25;

    // --- Transposition-table interaction (design/07 §3.7.2 steps 7/14) ---

    /// TT cutoffs are refused at/above this halfmove clock (graph-history).
    pub TT_CUTOFF_RULE50_LIMIT: i32 = 96;
    /// Extra depth for a no-legal-moves store: a terminal is exact far deeper.
    pub TT_NO_MOVES_DEPTH_BONUS: Depth = 6;

    // --- TT-cutoff history nudges (N6b extras, design/10 §3.4) ---
    // On a non-PV TT cutoff with `tt.value >= beta`: reward a quiet ttMove by
    // `min(BONUS_MULT·depth, CAP)`, penalize the parent's early quiet.

    pub TTCUT_BONUS_MULT: i32 = 114;
    pub TTCUT_BONUS_CAP: i32 = 724;
    pub TTCUT_PARENT_MALUS: i32 = -2187;
    /// The parent's `move_count` must be below this for the parent penalty.
    pub TTCUT_PARENT_MOVECOUNT: i32 = 4;

    // --- Singular extensions + multicut (design/10 §3.6) ---
    // An exclusion search (`excluded_move = ttMove`) at a margin below
    // `tt.value` decides: singular → extend and the WHOLE node deepens;
    // `value >= beta` → multicut; else negative extensions. `is_shuffling`
    // suppresses extensions of shuffles in 50-move territory. Each
    // SE_DOUBLE_*/SE_TRIPLE_* is an additive term of its margin: BASE, PV
    // bonus, non-tt-capture discount, TMH NUM/DEN, tt_pv, deep-node discount.

    /// Needs `depth >= SE_MIN_DEPTH + tt_pv` and the ttMove's stored depth
    /// within SE_TT_DEPTH_SLACK.
    pub SE_MIN_DEPTH: Depth = 6;
    pub SE_TT_DEPTH_SLACK: Depth = 3;
    /// Singular beta `tt.value - (MULT + TTPV_MULT·(tt_pv && !PV))·depth/DIV`.
    pub SE_BETA_MULT: Value = 60;
    pub SE_BETA_TTPV_MULT: Value = 70;
    pub SE_BETA_DIV: Value = 59;
    /// `corr_adj = |correction_value| / SE_CORR_DIV`, folded into the margins.
    pub SE_CORR_DIV: i32 = 194_822;
    pub SE_DOUBLE_BASE: Value = -3;
    pub SE_DOUBLE_PV: Value = 201;
    pub SE_DOUBLE_NON_TTCAP: Value = 157;
    pub SE_DOUBLE_TMH_NUM: i32 = 1081;
    pub SE_DOUBLE_DEEP: Value = 41;
    pub SE_TRIPLE_BASE: Value = 72;
    pub SE_TRIPLE_PV: Value = 306;
    pub SE_TRIPLE_NON_TTCAP: Value = 188;
    pub SE_TRIPLE_TTPV: Value = 84;
    pub SE_TRIPLE_DEEP: Value = 45;
    /// Multicut tt_move_history dock `<< TMH_BASE + TMH_DEPTH·depth`.
    pub MULTICUT_TMH_BASE: i32 = -442;
    pub MULTICUT_TMH_DEPTH: i32 = -108;

    // --- tt_move_history node-result update (design/10 §3.10) ---
    // Non-PV nodes only: hit when the best move was the ttMove, miss when not.

    pub TMH_HIT: i32 = 792;
    pub TMH_MISS: i32 = -779;

    // --- Sibling depth reduction on alpha raise (design/10 §3.9) ---

    /// A quiet alpha-raise in this window searches siblings 2 plies shallower.
    pub SIBLING_DEPTH_MIN: Depth = 2;
    pub SIBLING_DEPTH_MAX: Depth = 13;

    // --- cutoffCnt reduction gates (design/10 §3.7) ---
    // `(ss+1).cutoff_cnt` must exceed GATE for the BASE surcharge (the child
    // slot keeps failing high); +MANY above MANY_GATE, +ALLNODE if an all-node.

    pub RED_CUTOFFCNT_GATE: i32 = 1;
    pub RED_CUTOFFCNT_MANY_GATE: i32 = 2;
    pub RED_CUTOFFCNT_BASE: i32 = 236;
    pub RED_CUTOFFCNT_MANY: i32 = 1079;
    pub RED_CUTOFFCNT_ALLNODE: i32 = 1143;

    // --- ProbCut (N13, design/10 §3.4) ---
    // A shallow search proving a *capture* beats `probcut_beta` fails the node
    // high early, storing BOUND_LOWER and returning `value - (pc_beta - beta)`.

    /// `probcut_beta = beta + BETA_BASE - BETA_IMPROVING·improving`.
    pub PROBCUT_BETA_BASE: Value = 214;
    pub PROBCUT_BETA_IMPROVING: Value = 59;
    pub PROBCUT_MIN_DEPTH: Depth = 3;
    /// Subtracted (with `improving`) for re-verification; also the child-depth
    /// slack of the in-check small ProbCut (N14).
    pub PROBCUT_DEPTH_SUB: Depth = 4;
    /// The in-check small-ProbCut margin (N14): `probcut_beta = beta + this`.
    pub PROBCUT_CHECK_BETA: Value = 428;

    // --- Correction-history couplings (design/10 §4.6) ---
    // `correction_value` (the UNSCALED weighted sum) widens the RFP margin
    // under eval unreliability and reduces a touch less.

    pub RFP_CORR_DIV: i32 = 182_069;
    pub RED_CORR_DIV: i32 = 26_131;

    // --- Hindsight depth adjustments (N6, design/10 §3.4) ---

    /// `prior_reduction >= RED_UP && !opponent_worsening` → `depth += 1`.
    pub HINDSIGHT_RED_UP: i32 = 3;
    /// `prior_reduction >= RED_DOWN && depth >= 2 &&
    /// ss.static_eval + (ss-1).static_eval > EVAL_SUM` → `depth -= 1`.
    pub HINDSIGHT_RED_DOWN: i32 = 2;
    pub HINDSIGHT_EVAL_SUM: Value = 173;

    // --- TT-cutoff hardening (N6b, design/10 §3.4) ---

    /// At/below this depth a cutoff needs bound direction == node type.
    pub TTCUT_AGREE_DEPTH: Depth = 4;
    /// At/above this depth a cutoff is re-verified against the ttMove's child.
    pub TTCUT_VERIFY_DEPTH: Depth = 7;
    /// A wrong-direction bound above this depth costs the entry one ply.
    pub TTCUT_PENALIZE_DEPTH: Depth = 5;

    // --- Fail-low parent history bonus (N-result, design/10 §3.10) ---
    // A fail low with no best move under a parent quiet rewards the parent's
    // move: `bonus_scale` sums the FL_* signals; `scaled_bonus` (depth factor
    // `min(BONUS_MULT·depth - BONUS_SUB, BONUS_CAP)`) is split
    // `·CONTHIST_NUM/16384`, `·MAIN_NUM/32768`, `·PAWN_NUM/8192` to the parent
    // conthist / main / pawn tables. Parent CAPTURE: FL_CAPTURE_BONUS flat.

    /// `bonus_scale` = SCALE_BASE − `(ss-1).stat_score`/STATSCORE_DIV +
    /// `min(DEPTH_MULT·depth, DEPTH_CAP)` + the bumps below.
    pub FL_SCALE_BASE: i32 = -245;
    pub FL_STATSCORE_DIV: i32 = 98;
    pub FL_DEPTH_MULT: i32 = 59;
    pub FL_DEPTH_CAP: i32 = 430;
    /// Bump when the parent tried many moves (`(ss-1).move_count > 8`).
    pub FL_LATE_PARENT: i32 = 191;
    /// Bumps when `best_value` falls a margin below our own / the parent's
    /// static eval (each gated on that node not being in check).
    pub FL_BELOW_EVAL: i32 = 143;
    pub FL_BELOW_EVAL_MARGIN: Value = 103;
    pub FL_BELOW_PARENT_EVAL: i32 = 151;
    pub FL_BELOW_PARENT_EVAL_MARGIN: Value = 78;
    pub FL_BONUS_MULT: i32 = 141;
    pub FL_BONUS_SUB: i32 = 82;
    pub FL_BONUS_CAP: i32 = 1472;
    pub FL_CONTHIST_NUM: i32 = 236;
    pub FL_MAIN_NUM: i32 = 234;
    pub FL_PAWN_NUM: i32 = 322;
    pub FL_CAPTURE_BONUS: i32 = 901;

    // --- Eval-diff parent nudge (N8, design/10 §3.4) ---
    // Grade the opponent's last quiet by the static-eval swing: clamp to
    // [CLAMP_LO, CLAMP_HI], add OFFSET, multiply into main history and (TT
    // miss, non-pawn mover) pawn history. Indexed by the *opponent* color.

    pub EVALDIFF_CLAMP_LO: Value = -183;
    pub EVALDIFF_CLAMP_HI: Value = 180;
    pub EVALDIFF_OFFSET: Value = 62;
    pub EVALDIFF_MAIN_MULT: i32 = 10;
    pub EVALDIFF_PAWN_MULT: i32 = 13;

    // --- Time management: soft stop (design/12 §3.9) ---
    // These multiply into the plain `optimum` budget; formula shapes live at
    // the use site in `search::soft_stop`.

    /// Stability lerp over `root_depth − last_best_move_depth`, LO→
    /// [TIME_REDUCTION_MIN] and HI→[TIME_REDUCTION_MAX]. Deliberately below
    /// SF's [5, 18]: we complete ~3.5 fewer iterations at equal time.
    pub TIME_REDUCTION_STABILITY_LO: f64 = 2.0;
    pub TIME_REDUCTION_STABILITY_HI: f64 = 15.0;
    /// `(BASE + prev_time_reduction) / (DEN_COEFF·time_reduction)`.
    pub REDUCTION_BASE: f64 = 1.48;
    pub REDUCTION_DEN_COEFF: f64 = 2.157;

    /// Per-iteration aging divisor of the root best-move-change count.
    /// Deliberately 2.5, not SF's 2 — we complete ~3.3 fewer iterations, so the
    /// same wobble reads ~10x hotter through /2.
    pub INSTABILITY_CHANGES_DECAY: f64 = 2.5;

    // --- History updates: node-result bonus/malus (design/09 §3.9) ---
    // `update_all_stats` computes one `bonus` (best move + agreeing tables) and
    // one `malus` (tried non-best moves) per fail-high node, then distributes
    // them through the weights below (denominator [`STAT_WEIGHT_DEN`]).

    /// `min(DEPTH_MULT·depth − DEPTH_SUB, MAX)`, + TTMOVE if the best move was
    /// the predicted ttMove, − parent `stat_score / STATSCORE_DIV`.
    pub STAT_BONUS_DEPTH_MULT: i32 = 134;
    pub STAT_BONUS_DEPTH_SUB: i32 = 79;
    pub STAT_BONUS_MAX: i32 = 1572;
    pub STAT_BONUS_TTMOVE: i32 = 382;
    pub STAT_BONUS_STATSCORE_DIV: i32 = 30;
    /// Malus `min(DEPTH_MULT·depth − DEPTH_SUB, MAX)`.
    pub STAT_MALUS_DEPTH_MULT: i32 = 1005;
    pub STAT_MALUS_DEPTH_SUB: i32 = 205;
    pub STAT_MALUS_MAX: i32 = 2218;
    /// Non-PV breadth `bonus += bonus·tried / DIVISOR`; `i64` avoids overflow.
    pub STAT_BREADTH_DIVISOR: i64 = 256;
    pub QUIET_BEST_BONUS_NUM: i32 = 824;
    /// Geometric: `malus·START/1024` decaying `·DECAY/1024` per non-best quiet.
    pub QUIET_MALUS_START_NUM: i32 = 1136;
    pub QUIET_MALUS_DECAY_NUM: i32 = 956;
    pub CAPTURE_BEST_BONUS_NUM: i32 = 1366;
    pub CAPTURE_MALUS_NUM: i32 = 1518;
    /// Applied to the parent's refuted first-try quiet.
    pub REFUTATION_MALUS_NUM: i32 = 683;

    // --- History updates: per-quiet-table weights, all /1024 (design/09 §3.9) ---

    /// Plies < 5 only.
    pub QUIET_LOW_PLY_UPDATE_NUM: i32 = 663;
    pub QUIET_CONT_UPDATE_NUM: i32 = 820;
    /// Pawn history: maluses are roughly halved relative to bonuses.
    pub PAWN_HIST_BONUS_NUM: i32 = 1038;
    pub PAWN_HIST_MALUS_NUM: i32 = 525;
    /// Bonus/malus split: `> -7` bonus weight, `<= -7` malus (not `> 0`).
    pub PAWN_HIST_ASYMMETRY_THRESHOLD: i32 = -7;

    // --- History updates: continuation chain (design/09 §3.9) ---

    /// Added to the offset-1 continuation update, even for maluses.
    pub CONT_OFFSET1_FLAT_BONUS: i32 = 71;
    /// In check, only offsets `<= 2` update (the pre-check position differs).
    pub CONT_IN_CHECK_MAX_OFFSET: usize = 2;

    // --- Correction history: read combiner (design/09 §3.10) ---
    // Blends the per-source corrections below in 1/131072-pawn units;
    // [`CORR_VALUE_DIVISOR`] converts back to centipawns.

    pub CORR_PAWN_WEIGHT: i32 = 13345;
    pub CORR_MINOR_WEIGHT: i32 = 9280;
    /// Applied to each of the white and black non-pawn corrections.
    pub CORR_NONPAWN_WEIGHT: i32 = 11840;
    /// Weight on the summed continuation corrections (offsets 2 and 4).
    pub CORR_CONT_WEIGHT: i32 = 8363;
    /// Flat continuation-correction prior used after a null previous move.
    pub CORR_CONT_NULL_PRIOR: i32 = 64549;

    // --- Correction history: update weights, all /128 (design/09 §3.10) ---

    pub CORR_MINOR_UPDATE_NUM: i32 = 152;
    /// Applied to both the white and black non-pawn corrections.
    pub CORR_NONPAWN_UPDATE_NUM: i32 = 186;
    pub CORR_CONT2_UPDATE_NUM: i32 = 136;
    pub CORR_CONT4_UPDATE_NUM: i32 = 68;

    // --- Correction history: bonus computation (call site, design/10 §4.6) ---

    /// `err·depth·MULT/128`: BESTMOVE with a best move, NOMOVE on a fail low.
    pub CORR_BONUS_DEPTH_MULT_BESTMOVE: i32 = 12;
    pub CORR_BONUS_DEPTH_MULT_NOMOVE: i32 = 18;
    /// Symmetric clamp on the raw bonus (`CORRECTION_HISTORY_LIMIT/4`).
    pub CORR_BONUS_CLAMP: i32 = 256;
    /// Outer scaling `·SCALE_NUM/1024`.
    pub CORR_BONUS_SCALE_NUM: i32 = 1114;

    // --- SF-parity structural toggles: RESOLVED, do not re-propose. QS_UPREP
    // and RED_TTPV_TTMISS_REBATE measured worse ON and were deleted;
    // PROBCUT_IN_VERIF is a de facto no-op in both engines. ---

    /// Beta weight (1024ths) of the stand-pat softening `((1024−W)·best +
    /// W·beta)/1024`, on non-decisive stand pats ≥ beta (step 5);
    /// FAILHIGH_BETA_W is the same shape at step 11.
    pub QS_STANDPAT_BETA_W: Value = 557;
    pub QS_FAILHIGH_BETA_W: Value = 543;
    /// Window `nmp_min_ply = ply + NUM·(depth - R)/DEN`, so the verification
    /// cannot recursively null-move back to the same false cutoff.
    pub NMP_VERIFY_NUM: Depth = 3;
    /// Negative extensions: TT_FAILHIGH when the search proved `tt_value >=
    /// beta`, CUTNODE at a cut node that is neither.
    pub SE_NEG_EXT_TT_FAILHIGH: Depth = -3;
    pub SE_NEG_EXT_CUTNODE: Depth = -2;
    /// LMR entry gate `depth >= MIN_DEPTH && move_count > MIN_MOVECOUNT`.
    pub LMR_MIN_DEPTH: Depth = 2;
    pub LMR_MIN_MOVECOUNT: i32 = 1;
    /// `history += NUM·main/DEN` (SF's `64·h/32` leaves 1/32 for the tuner).
    pub LMRDEPTH_MAIN_NUM: i32 = 64;

}

// ─── D50 constified handles ───
//
// Same `NAME()` spelling as the tunables, but compile-time constant in EVERY
// build; none is on the SPSA surface (D50 — a num/den pair is ONE degree of
// freedom). Promote one back by moving its line into `tunables!`.
crate::constants! {

    /// Lazy-SMP jitter: helper `t` adds `t % PERIOD` to the base delta; vacuous
    /// at `thread_idx == 0`, keeping Threads=1 bit-identical (D24).
    pub ASPIRATION_JITTER_PERIOD: usize = 8;
    pub ASPIRATION_WIDEN_DEN: Value = 128;
    pub ASPIRATION_SEARCH_AGAIN_DEN: Depth = 4;
    pub RFP_TREND_DEN: Value = 1024;
    /// A power of two — exactly representable, so the table is reproducible.
    pub RED_TABLE_DEN: i32 = 128;
    pub RED_NON_IMPROVING_DEN: i32 = 512;
    /// `r` is carried in 1024ths of a ply, divided only when it becomes depth.
    pub RED_UNIT: i32 = 1024;
    pub STATSCORE_CAPT_DEN: i32 = 128;
    /// A power of two; truncation toward zero on a negative score is intended.
    pub RED_STATSCORE_DEN: i32 = 4096;
    /// Plies the internal iterative reduction removes.
    pub IIR_REDUCTION: Depth = 1;
    /// `(…) / (DEN - improving)`: at 2 a not-improving node prunes at half.
    pub LMP_IMPROVING_DEN: i32 = 2;
    pub SE_VERIFY_DEPTH_DEN: Depth = 2;
    /// `extension = SE_EXT_BASE + (double) + (triple)`.
    pub SE_EXT_BASE: Depth = 1;
    /// Extra depth the WHOLE node gains when its ttMove proves singular.
    pub SE_NODE_DEEPEN: Depth = 1;
    /// `depth >= SE_MIN_DEPTH + SE_TTPV_DEPTH_BONUS * tt_pv`.
    pub SE_TTPV_DEPTH_BONUS: Depth = 1;
    /// `depth - PROBCUT_DEPTH_SUB - PROBCUT_IMPROVING_SUB * improving`.
    pub PROBCUT_IMPROVING_SUB: Depth = 1;
    /// PV-node scout bonus, applied to the reduced depth and its deeper cap.
    pub LMR_PV_DEPTH_BONUS: Depth = 1;
    /// Scout-depth floor, which keeps a heavily-reduced scout out of qsearch.
    pub LMR_MIN_SCOUT_DEPTH: Depth = 1;
    /// Re-search steps for a scout well above / barely above `best_value`.
    pub LMR_DEEPER_STEP: Depth = 1;
    pub LMR_SHALLOWER_STEP: Depth = 1;
    /// Docked per threshold met; the SECOND needs `new_depth >= DOCK2_MIN`.
    pub LMR_FULLDEPTH_DOCK_STEP: Depth = 1;
    pub LMR_DOCK2_MIN_DEPTH: Depth = 2;
    /// `tt_depth > depth - BOUND_SLACK * (tt_value <= beta)`.
    pub TTCUT_BOUND_SLACK: Depth = 1;
    /// `cutoff_cnt += (extension < EXT_GATE || PV)`.
    pub CUTOFFCNT_EXT_GATE: Depth = 2;
    /// Divisor of the capture-history reads in both capture pruning gates.
    pub CAPT_HIST_DEN: i32 = 1024;
    pub SE_DOUBLE_TMH_DEN: i32 = 117_824;
    /// `is_shuffling` floors: halfmove clock, plies since null, ply.
    pub SHUFFLE_RULE50: i32 = 10;
    pub SHUFFLE_PLIES_FROM_NULL: i32 = 6;
    pub SHUFFLE_MIN_PLY: usize = 20;
    pub FL_CONTHIST_DEN: i32 = 16384;
    pub FL_MAIN_DEN: i32 = 32768;
    pub FL_PAWN_DEN: i32 = 8192;
    /// Single-reply cap on the soft budget: with one legal move the result
    /// cannot change the move. TM units (ms, or `nodestime` nodes).
    pub ONE_REPLY_TOTAL_TIME_CAP: f64 = 561.7;
    /// DIS-TM bar, in internal cp, on REMAINING predicted movement (root
    /// prediction minus movement already realised); the arm buys time only
    /// above it. Inert unless `ARCHON_DIS_TM_C` is set.
    pub DIS_TM_T0: f64 = 40.0;
    /// DIS-TM pivot: the REMAINING value treated as neutral — the measured
    /// median of REMAINING itself, so the arm is mean-neutral. Must pivot on
    /// the compared quantity; pivoting on `dismag` makes it a net time-saver.
    pub DIS_TM_PIVOT: f64 = 127.0;
    /// `(BASE + PREV_AVG_COEFF·Δavg + ITER_COEFF·Δiter) / DIV`, clamped to
    /// `[MIN, MAX]` — think longer when the score drops. Δavg =
    /// `best_previous_average_score − best_value`, Δiter = `iter_value[i] −
    /// best_value`.
    pub FALLING_EVAL_BASE: f64 = 11.87;
    pub FALLING_EVAL_PREV_AVG_COEFF: f64 = 2.21;
    pub FALLING_EVAL_ITER_COEFF: f64 = 1.0;
    pub FALLING_EVAL_DIV: f64 = 100.0;
    pub FALLING_EVAL_MIN: f64 = 0.572;
    pub FALLING_EVAL_MAX: f64 = 1.708;
    /// `time_reduction` clamps, and the lerp y-endpoints at low/high stability.
    pub TIME_REDUCTION_MIN: f64 = 0.65;
    pub TIME_REDUCTION_MAX: f64 = 1.55;
    /// `BASE + COEFF·tot_best_move_changes / threads` (best move keeps flipping).
    pub INSTABILITY_BASE: f64 = 1.096;
    pub INSTABILITY_CHANGES_COEFF: f64 = 2.29;
    /// `effort·UNIT_SCALE / max(1, nodes)`, lerped LERP_LO→SCALE_AT_LO …
    /// LERP_HI→SCALE_AT_HI (also the clamps).
    pub EFFORT_LERP_LO: f64 = 79_219.0;
    pub EFFORT_LERP_HI: f64 = 101_822.0;
    pub EFFORT_SCALE_AT_LO: f64 = 0.924;
    pub EFFORT_SCALE_AT_HI: f64 = 0.71;
    pub EFFORT_UNIT_SCALE: i64 = 100_000;
    /// The next iteration deepens only below this fraction of the soft budget.
    pub INCREASE_DEPTH_FRACTION: f64 = 0.50;
    /// Early-stop: every PV mates in `<=` MATE_PLIES, or we are mated in
    /// exactly MATED_PLIES.
    pub SOFT_STOP_MATE_PLIES: Depth = 3;
    pub SOFT_STOP_MATED_PLIES: Depth = 2;
    /// Shaved off every enforced *wall-clock* bound (`maximum()`, `movetime`)
    /// to cover the unwind-and-print tail after `stop`. Deliberately NOT
    /// applied to the `nodes` budget — an exactness bound, not a forfeit risk.
    pub CHECK_TIME_EPSILON_MS: i64 = 15;
    /// Below this many nodes the MultiPV block is published only when the
    /// iteration's LAST line finishes; above it, after every line.
    pub INFO_THROTTLE_NODES: u64 = 10_000_000;
    pub STAT_WEIGHT_DEN: i32 = 1024;
    /// `bonus·weight·mult / this`; `i64` widens the product before dividing.
    pub CONT_UPDATE_DIVISOR: i64 = 131072;
    /// Turns the correction sum back into a centipawn adjustment.
    pub CORR_VALUE_DIVISOR: i32 = 131072;
    pub CORR_UPDATE_DEN: i32 = 128;
    pub CORR_BONUS_DIVISOR: i32 = 128;
    pub CORR_BONUS_SCALE_DEN: i32 = 1024;
    pub NMP_VERIFY_DEN: Depth = 4;
    pub LMRDEPTH_MAIN_DEN: i32 = 32;
    /// `reductions[i] = (NUM/DEN)·ln(i) + OFFSET/DEN`. The ONLY new degree of
    /// freedom the product form `f(d)·f(mn)` admits; 0 = identity.
    pub RED_TABLE_OFFSET: i32 = 0;
    /// Each consecutive root fail-high re-searches this many plies shallower.
    pub ASPIRATION_FAILHIGH_DEPTH_MULT: Depth = 1;
    /// MAIN-search fail-high softening `(best·depth + W·beta) / (depth + W)`.
    pub FH_SOFTEN_BETA_W: i32 = 1;
    /// `depth -= 2` on a non-cutoff alpha raise inside the SIBLING_DEPTH window.
    pub SIBLING_DEPTH_STEP: Depth = 2;
}
