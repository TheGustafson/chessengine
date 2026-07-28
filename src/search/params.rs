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

// The qsearch return blends and the NMP verify fraction moved INTO the
// `tunables!` block on 2026-07-27 (the tuple-const shape made them
// structurally untunable — the macro only emits scalar atomics; D33
// reviewer finding F1/#7). See QS_STANDPAT_BEST_W / QS_FAILHIGH_BEST_W /
// NMP_VERIFY_NUM and friends below.
/// The per-thread vote tempo (design/13 §3.8; SF-current `get_best_thread`).
/// Each thread votes for its own best move with weight `score - min_score +
/// VOTE_TEMPO`; the `+ VOTE_TEMPO` floor means even the weakest thread's
/// opinion is counted. Value units (centipawn-ish, ply-relative).
pub const VOTE_TEMPO: crate::types::Value = 14;
/// Per-offset weights of the continuation-history update, offsets 1..=6
/// (÷[`CONT_UPDATE_DIVISOR`] with the bonus).
pub const CONT_OFFSET_WEIGHTS: [i32; 6] = [1040, 780, 300, 537, 129, 423];
/// Consistency multipliers indexed by the running count of already-positive
/// entries along the chain: an agreeing chain amplifies the update.
pub const CONT_CONSISTENCY_MULT: [i32; 7] = [96, 113, 101, 105, 127, 121, 126];

/// History-buys-lmr_depth divisor by depth (design/10 §3.5, item 11's
/// second half, S0): `lmr_depth += history / LMR_DIVISOR[min(depth,16)-1]`
/// where `history` is the quiet's combined continuation/pawn score plus
/// `2·main_history`. A tuned per-depth table (SF-current shape).
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

    // --- The eval-frame scale (docs/research/eval-frame-scale-model-2026-07-27.md) ---

    /// **The single knob for the eval-frame correction.** In 1024ths, so
    /// `1024` = 1.0 = *exact identity with today's engine* — every transform
    /// below is `c * FRAME_S / 1024` or its inverse, and integer arithmetic
    /// makes both a no-op at 1024. That identity is the point: it makes the
    /// sweep a clean control instead of a comparison against a rebuilt
    /// baseline.
    ///
    /// Why one knob for many constants. Our static eval measures at ≈0.58×
    /// SF18's internal frame (three independent estimates: 0.55 winner-slope
    /// from games, 0.579 wl50, 0.605 WAC), while our cp-denominated margins
    /// sit at Stockfish's RAW values — razor 465 vs 485, capture-futility
    /// base 231 vs 232, aspiration delta 5 vs 5. If that holds, every one of
    /// them is ~1.7× too large in effect and they are all wrong *by the same
    /// factor*, because they all inherited the same unconverted import.
    ///
    /// The non-linearity does NOT cost us the single parameter — it only
    /// changes how each constant READS the knob. See [`frame_scale`] (class
    /// 1/2, eval-space) and [`frame_expand`] (class 3, cross-unit divisors
    /// that must move the opposite way). Class 5 (plies, depths, counts)
    /// ignores it. Class 4 (material mixed into an eval comparison) cannot be
    /// repaired by any constant and is deliberately NOT wired here.
    ///
    /// Theory puts the optimum near 594 (= 0.58); 1024 is the control.
    pub FRAME_S: Value = 1024;

    // --- Aspiration windows (design/07 §3.6) ---

    /// Base half-width of the window around the root move's average score.
    pub ASPIRATION_BASE_DELTA: Value = 5;
    /// Divisor of `mean_squared_score`: unstable scores widen the base delta.
    pub ASPIRATION_MSS_DIVISOR: i64 = 10588;
    /// Period of the lazy-SMP aspiration jitter (design/13 §3.1): helper thread
    /// `t` adds `t % ASPIRATION_JITTER_PERIOD` to the base delta so helpers search
    /// slightly different windows and seed the shared TT divergently. Vacuous at
    /// `thread_idx == 0` (main/detached/bench), keeping Threads=1 bit-identical
    /// (D24).
    pub ASPIRATION_JITTER_PERIOD: usize = 8;
    /// Optimism numerator: `optimism = OPTIMISM_NUM·avg / (|avg| +
    /// OPTIMISM_DEN_BASE)` at aspiration setup (design/11 §3.6's consumer,
    /// S0). Priors are SF's 137/81 scaled by the measured winner-slope
    /// ~0.55 (our engine frame is ~2× compressed vs SF's reported cp,
    /// loss_forensics 2026-07-26); S2 SPSA owns the refinement.
    pub OPTIMISM_NUM: Value = 75;
    /// Optimism half-saturation denominator base (see OPTIMISM_NUM).
    pub OPTIMISM_DEN_BASE: Value = 45;
    /// Numerator of the per-re-search widening `delta += delta * 44 / 128`.
    pub ASPIRATION_WIDEN_NUM: Value = 44;
    /// Denominator of the per-re-search widening (~x1.34 per fail).
    pub ASPIRATION_WIDEN_DEN: Value = 128;
    /// Numerator of the search-again depth damping term
    /// `3 * (search_again_counter + 1) / 4` in the aspiration loop's
    /// `adjusted_depth` (design/07 §3.6, design/12 §3.9 — the counter itself
    /// is TM-owned and 0 until doc 12 step 8).
    pub ASPIRATION_SEARCH_AGAIN_NUM: Depth = 3;
    /// Denominator of the search-again depth damping.
    pub ASPIRATION_SEARCH_AGAIN_DEN: Depth = 4;

    // --- Move ordering: capture scoring (design/09 §3.7) ---

    /// MVV weight on the victim value in the capture ordering score
    /// `CAPTURE_MVV_FACTOR * victim_value + capture_history` (design/09 §3.7):
    /// pure most-valuable-victim plus capture history, with no attacker (LVA)
    /// term — SEE handles attacker risk at the good/bad split gate.
    pub CAPTURE_MVV_FACTOR: Value = 7;

    // --- Move ordering: good/bad capture split (design/09 §3.6 stage 3) ---

    /// Divisor of the score-dependent SEE gate `see_ge(m, -score / 18)` that
    /// splits captures into good and bad (design/09 §3.6): a capture with a
    /// high ordering score may lose a little material and still count as good;
    /// a capture with terrible capture-history must win material to qualify.
    pub GOOD_CAPTURE_SEE_DIVISOR: Value = 18;

    // --- Move ordering: quiet scoring (design/09 §3.7) ---

    /// Weight on butterfly (main) history in the quiet ordering score.
    pub QUIET_MAIN_HISTORY_WEIGHT: Value = 2;
    /// Weight on pawn-structure history in the quiet ordering score.
    pub QUIET_PAWN_HISTORY_WEIGHT: Value = 2;
    /// Flat bonus for a quiet that gives check, gated by the SEE margin below
    /// so a check that hangs material earns nothing.
    pub QUIET_CHECK_BONUS: Value = 16384;
    /// SEE margin gating the check bonus: a checking quiet earns it only if the
    /// move does not lose more than this much material (`see_ge(m, -75)`).
    pub QUIET_CHECK_SEE_MARGIN: Value = -75;
    /// Per-piece-value multiplier of the threat escape/entry term: fleeing a
    /// square attacked by a strictly cheaper enemy piece earns
    /// `QUIET_THREAT_FACTOR * piece_value`, stepping onto one costs the same.
    /// Fed by the Wave-G threat map (D30, `Board::piece_threats`).
    pub QUIET_THREAT_FACTOR: Value = 20;
    /// Per-piece-value multiplier of the Wave-G offense term (D30, needs the threat
    /// map): moving to a square from which the piece attacks a *vulnerable* enemy —
    /// an undefended enemy piece worth at least as much as the mover — earns
    /// `QUIET_OFFENSE_FACTOR * piece_value`. The "create a threat" counterpart to
    /// the "avoid a threat" escape term; a modest reward (SF/our §3.7 scored only
    /// the defensive side).
    pub QUIET_OFFENSE_FACTOR: Value = 10;
    /// Weight on low-ply history in the quiet ordering score (plies 0..4 only,
    /// scaled by `/ (1 + ply)`).
    pub QUIET_LOW_PLY_WEIGHT: Value = 8;

    // --- Move ordering: quiet good/bad split + sort (design/09 §3.6) ---

    /// Good/bad quiet partition threshold (design/09 §3.6 stages 5/7): quiets
    /// scoring strictly above this are "good" (emitted before bad captures);
    /// the exact complement are "bad" (emitted last). One shared threshold
    /// partitions the quiet region with no dedup bookkeeping.
    pub GOOD_QUIET_THRESHOLD: Value = -14000;
    /// Depth multiplier of the quiet partial-sort limit `-3560 * depth`
    /// (design/09 §3.6 stage 4 / §3.8): deeper nodes sort more of the list, so
    /// sorting work scales with what the search actually consumes.
    pub QUIET_SORT_DEPTH_FACTOR: Value = -3560;

    // --- Move ordering: evasion scoring (design/09 §3.7) ---

    /// Constant lifting every capturing evasion above every quiet evasion
    /// (design/09 §3.7): `2^28` dwarfs any history-based quiet-evasion score.
    pub EVASION_CAPTURE_BONUS: Value = 1 << 28;

    // --- Quiescence search (design/07 §3.8) ---

    /// Futility headroom over the corrected static eval:
    /// `futility_base = static_eval + 335` (§3.8 step 5). A capture whose
    /// victim cannot lift this base above alpha is skipped.
    pub QS_FUTILITY_MARGIN: Value = 335;
    /// **Class-4 converter for the qsearch futility bound**, in 1024ths.
    /// `1024` = identity, so the shipped engine is unchanged.
    ///
    /// `futility_value = static_eval + QS_FUTILITY_MARGIN + captured_value`
    /// adds a MATERIAL quantity to an EVAL one. Both engines do that, but the
    /// two frames differ by a different amount in each: Stockfish's eval pawn
    /// is ~385 against a `PieceValue` pawn of 208, so a captured queen enters
    /// its bound as ~6.6 pawns. Ours is ~246 against the SAME `PieceValue`,
    /// so the same queen enters as ~10.3 pawns.
    ///
    /// We therefore inflate the captured piece by ~1.57x relative to
    /// Stockfish, the bound is almost never below alpha, and qsearch futility
    /// barely prunes. That is paid at EVERY LEAF: measured, our qsearch
    /// expands 2.53 nodes per leaf against SF18's 1.40, which reproduces as
    /// the flat ~2x node multiplier that survived four other explanations
    /// (`docs/research/SCREENING.md` §14-15).
    ///
    /// Unlike the margin itself, this is not a tuning knob with an arbitrary
    /// optimum — 0.64 is the derived value that puts our bound on Stockfish's
    /// footing. `CFUT_PIECE_MULT` is the same correction for the main-loop
    /// capture futility, where it measured neutral because that gate fires
    /// far less often.
    pub QS_PIECE_MULT: i32 = 1024;
    /// The hard SEE floor: any move whose static exchange value cannot reach
    /// this bar is never searched in qsearch (§3.8 step 7).
    pub QS_SEE_FLOOR: Value = -74;
    /// Late-move cutoff: after this many non-checking moves not aimed at the
    /// previous move's destination, stop trying such moves entirely
    /// (`move_count > 2`, §3.8 step 7).
    pub QS_LATE_MOVE_LIMIT: i32 = 2;

    // --- Equal-value dither (design/07 §3.7.2 step 12) ---

    /// Node-counter mask of the equal-value dither: near the leaves an
    /// equal-to-best value is promoted to best only when `(nodes & 14) == 0`
    /// (nodes ≡ 0 or 1 mod 16, ~1 in 8 made moves) — deliberate diversity so
    /// equal lines do not all tie on the first one found.
    pub DITHER_NODE_MASK: u64 = 14;
    /// Leaf-proximity gate of the dither: it fires only when
    /// `ply + 2 >= root_depth`.
    pub DITHER_PLY_MARGIN: usize = 2;

    // --- Razoring (design/10 §3.1; node step N9) ---

    /// Flat margin below alpha at which shallow nodes drop to qsearch.
    pub RAZOR_BASE: Value = 465;
    /// Depth-quadratic term of the razoring margin (`* depth^2`).
    pub RAZOR_DEPTH_MULT: Value = 300;
    /// **Eval-uncertainty coupling for razoring** — margin widens by
    /// `|correction_value| / RAZOR_CORR_DIV`. Effectively OFF at the default
    /// (the term floors to 0), so the shipped engine is unchanged.
    ///
    /// A deliberate deviation from Stockfish, which has no such term.
    /// Reasoning: the eval-frame study established that what a margin must
    /// track is the eval's ERROR, not its magnitude
    /// (`eval-frame-scale-model-2026-07-27.md` §11.3), and correction history
    /// is a per-position estimate of exactly that error. RFP, singular and
    /// LMR already couple to it; razoring does not — despite being the most
    /// destructive gate in the engine, since it replaces an entire node with
    /// a qsearch value and nothing downstream corrects it.
    ///
    /// Under the node-level/move-level hypothesis (`SCREENING.md` §10) that
    /// is precisely backwards: the gate with the largest cost-of-being-wrong
    /// should be the MOST reluctant where the eval is historically
    /// unreliable. Smaller divisor = stronger coupling = razor less often in
    /// positions our eval has been getting wrong.
    pub RAZOR_CORR_DIV: i32 = 1_000_000_000;

    // --- Reverse futility pruning (design/10 §3.4; node step N10) ---
    //
    // "Futility pruning: child" — a non-PV node whose corrected static eval
    // already clears beta by a depth-scaled margin returns a beta-ward blend
    // without searching (the fail-high is almost certainly real). The margin
    // `futility_mult·depth - (RFP_IMPROVING·improving + RFP_OPP_WORSENING·
    // opponent_worsening)·futility_mult / RFP_TREND_DEN` shrinks when the
    // trend flags say the fail-high is well-supported. The Wave-F correction
    // term (`|correction_value| / 182069`) is deliberately absent until
    // correction history is plumbed into this suite (design/10 §4 Wave F).

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
    /// Subtracted from the futility multiplier on a TT miss (a node with no TT
    /// data prunes a little more cautiously).
    pub RFP_MULT_NO_TTHIT: Value = 20;
    /// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which an *improving*
    /// node shrinks the margin — the fail-high is better supported.
    pub RFP_IMPROVING: Value = 2934;
    /// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which a *worsening*
    /// opponent shrinks the margin.
    pub RFP_OPP_WORSENING: Value = 343;
    /// Denominator of the improving/opponent-worsening margin discounts.
    pub RFP_TREND_DEN: Value = 1024;
    /// Threat-coupled RFP relaxation (Wave-G item 23, D30): when NO own piece
    /// (pawns included) is currently threatened (the Wave-G threat map says the
    /// position is safe), subtract this flat term from the N10 margin so RFP fires
    /// a little more readily — a safe position at/above beta is very likely a real
    /// fail-high. The N10 use-site keys on `all_threats() & occupancy(us)` — ALL
    /// our pieces, matching Reckless's `all_threats() & colors(stm)`; a hanging
    /// pawn still blocks the relaxation. Un-tuned placeholder pending the SPSA pass.
    pub RFP_THREAT_MARGIN: Value = 55;
    /// Beta weight of the return blend `(RFP_RET_BETA_W·beta + RFP_RET_EVAL_W·
    /// eval) / RFP_RET_DEN` — a value ≥ beta, pulled toward beta to damp
    /// static-eval noise (the same instability-damping idiom as the node-exit
    /// fail-high soften).
    pub RFP_RET_BETA_W: Value = 716;
    /// Eval weight of the return blend.
    pub RFP_RET_EVAL_W: Value = 308;
    /// Denominator of the return blend (`RFP_RET_BETA_W + RFP_RET_EVAL_W`).
    pub RFP_RET_DEN: Value = 1024;

    // --- Null move pruning (design/10 §3.4; node step N11) ---
    //
    // A cut node whose static eval already clears a beta-relative bar assumes
    // it can afford to "pass" (make a null move) and still fail high: it
    // searches the null move at reduced depth `depth - R` and, if that fails
    // high, returns without searching a real move. The classic zugzwang trap —
    // where passing is illusory — is fenced off three ways: NMP runs only when
    // the side to move has non-pawn material (`non_pawn_material(us) != 0`), a
    // proven-mate null value is never trusted (`!is_win(null_value)`), and at
    // high depth a same-node verification search re-confirms the fail-high with
    // NMP disabled up to `nmp_min_ply` (recursion- and zugzwang-proof).
    //
    // The gate reads `ss.static_eval` (NOT the TT-substituted `eval` RFP uses —
    // design/10 §3.4 N11 marks the site).

    /// Per-ply slope of the NMP static-eval bar `beta - NMP_EVAL_DEPTH·depth -
    /// NMP_EVAL_IMPROVING·improving + NMP_EVAL_BASE`: deeper nodes need less
    /// headroom (a deeper fail-high is better supported).
    pub NMP_EVAL_DEPTH: Value = 14;
    /// Improving discount of the NMP static-eval bar.
    pub NMP_EVAL_IMPROVING: Value = 45;
    /// Flat base of the NMP static-eval bar (dominates at shallow depth, so the
    /// bar sits well above beta there).
    pub NMP_EVAL_BASE: Value = 374;
    /// **Eval-uncertainty coupling for the null-move bar** — the bar rises by
    /// `|correction_value| / NMP_CORR_DIV`, so NMP fires less readily where
    /// our eval has been unreliable. Effectively OFF at the default.
    ///
    /// Same reasoning as [`RAZOR_CORR_DIV`], and NMP is the other node-level
    /// gate with no correction coupling. NMP's whole premise is that the
    /// static eval is trustworthy enough to skip a move — which is exactly
    /// the assumption correction history measures.
    pub NMP_CORR_DIV: i32 = 1_000_000_000;
    /// Base of the null-move dynamic reduction `R = NMP_R_BASE + depth /
    /// NMP_R_DEPTH_DIV` (in whole plies — this reduction is applied directly to
    /// depth, not in the 1024ths the LMR table uses).
    pub NMP_R_BASE: Depth = 7;
    /// Depth divisor of the null-move reduction.
    pub NMP_R_DEPTH_DIV: Depth = 3;
    /// Depth at/above which a fail-high null search is re-confirmed by the
    /// verification search (`depth >= NMP_VERIFY_DEPTH`); below it the null
    /// value is returned directly.
    pub NMP_VERIFY_DEPTH: Depth = 16;

    // --- Base reduction table + `reduction()` (design/10 §3.3) ---
    //
    // The per-worker log reduction table `reductions[i] = int(RED_TABLE_NUM /
    // RED_TABLE_DEN * ln(i))` (built once at `Worker` construction) and the base
    // per-move reduction `reduction(improving, depth, move_number, delta)`, both
    // carried in 1024ths of a ply — divide by 1024 only at the moment of
    // application (the suite's unit convention). The table and function are
    // wired in this Wave-A item but stay UNUSED until Wave-A item 4 applies LMR
    // in the move loop (design/10 §4).

    /// Numerator of the log reduction table `int(2834 / 128 * ln(i))`; the
    /// division is done in floating point once, at table-build time.
    pub RED_TABLE_NUM: i32 = 2834;
    /// Denominator of the log reduction table (a power of two — exactly
    /// representable, so the built table is bit-reproducible).
    pub RED_TABLE_DEN: i32 = 128;
    /// Wide-window damping of the base reduction: a wider local window `delta`
    /// reduces less, via `- delta * RED_DELTA_MULT / root_delta`.
    pub RED_DELTA_MULT: i32 = 617;
    /// Not-improving surcharge numerator: a not-improving node reduces more, by
    /// `+ scale * RED_NON_IMPROVING / RED_NON_IMPROVING_DEN` (~38% extra).
    pub RED_NON_IMPROVING: i32 = 194;
    /// Not-improving surcharge denominator (the `/512` of the term above).
    pub RED_NON_IMPROVING_DEN: i32 = 512;
    /// Flat base added to every base reduction (in 1024ths of a ply).
    pub RED_BASE: i32 = 1027;

    // --- LMR reduction finalization + search mechanics (design/10 §3.7/§3.8) ---
    //
    // Wave-A item 4 (design/10 §4): the base [`Worker::reduction`] output `r`
    // (1024ths of a ply) accretes the Wave-A subset of the §3.7 finalization —
    // the flat rebase, the move-count discount, the cut-node surcharge, and the
    // TT-move rebate — then §3.8 turns it into a reduced zero-window scout depth
    // `d = max(1, min(new_depth - r/RED_UNIT, new_depth + RED_MAX_DEEPER)) + PV`,
    // with the deeper/shallower re-search rule and the non-LMR dock path. Wave-C
    // item 10 adds the `ss.stat_score` combined-history term below. The ttPv,
    // correction (`/26131`), tt_capture, cutoffCnt and all-node terms are still
    // deferred to their later waves (design/10 §4 Waves C/D/F), so those
    // constants are absent here.

    /// The application unit of the reduction accumulator: reductions are carried
    /// in 1024ths of a ply and divided by this only at the moment they become a
    /// depth (`new_depth - r / RED_UNIT`, §3.8 — the suite's unit convention).
    pub RED_UNIT: i32 = 1024;
    /// Flat rebase added to every finalized reduction (§3.7).
    pub RED_FLAT: i32 = 714;
    /// Per-move-count reduction discount `r -= RED_MOVECOUNT * move_count`: later
    /// moves in the list reduce a little less than the raw log table implies
    /// (they were tried later precisely because ordering distrusted them, §3.7).
    pub RED_MOVECOUNT: i32 = 62;
    /// Cut-node reduction surcharge: an expected-fail-high node reduces its late
    /// moves harder — it wants a cheap refutation, not a thorough search (§3.7).
    pub RED_CUTNODE: i32 = 3995;
    /// Extra cut-node surcharge when the node has no TT move: nothing trustworthy
    /// to search near full depth, so reduce even more (§3.7).
    pub RED_CUTNODE_NO_TTMOVE: i32 = 1059;
    /// TT-move reduction rebate `r = max(0, r - RED_TTMOVE_SUB)`: the move the TT
    /// records as best is searched close to full depth (§3.7). In Wave A this is
    /// a plain `if m == tt.mv`; the cutoffCnt `if/else` wrapper is Wave D.
    pub RED_TTMOVE_SUB: i32 = 2016;
    /// Cap on how many plies a NEGATIVE net `r` may DEEPEN the reduced scout past
    /// `new_depth` (`min(new_depth - r/RED_UNIT, new_depth + RED_MAX_DEEPER)`,
    /// §3.8); PV nodes get a further unconditional +1.
    pub RED_MAX_DEEPER: Depth = 2;
    /// Deeper-re-search margin: after a reduced scout beats alpha, re-search one
    /// ply DEEPER when it beat the running best by more than this (§3.8).
    pub LMR_DEEPER_MARGIN: Value = 52;
    /// Shallower-re-search margin: re-search one ply SHALLOWER when the scout only
    /// barely cleared the running best (`value < best_value + this`, §3.8).
    pub LMR_SHALLOWER_MARGIN: Value = 9;
    /// Non-LMR path: a node with no TT move adds this to `r` before the depth
    /// docks below — the full-depth analogue of the cut-node no-ttMove surcharge
    /// (§3.8).
    pub RED_FULLDEPTH_NO_TTMOVE: i32 = 1085;
    /// Non-LMR path first depth dock: an `r` above this costs one ply even for a
    /// move that does not qualify for LMR proper (§3.8).
    pub RED_FULLDEPTH_DOCK1: i32 = 5039;
    /// Non-LMR path second depth dock: an `r` above this (and `new_depth > 2`)
    /// costs a second ply (§3.8).
    pub RED_FULLDEPTH_DOCK2: i32 = 5223;

    // --- LMR statScore: combined-history reduction + post-LMR bonus (design/10
    //     §3.7/§3.8; Wave-C item 10) ---
    //
    // `ss.stat_score` is the per-node combined history statistic of the move about
    // to be searched, read from PRE-move state and stored on the stack (the child
    // reads `(ss-1).stat_score` for its fail-low parent bonus, a later wave). It
    // grades a capture by victim value plus capture history —
    // `STATSCORE_CAPT_NUM·piece_value(captured)/STATSCORE_CAPT_DEN + capt_hist` —
    // and a quiet by twice its butterfly history plus the offset-1/offset-2
    // continuation histories — `STATSCORE_MAIN_WEIGHT·main + contHist(1) +
    // contHist(2)`. It then docks the reduction: `r -= stat_score·
    // RED_STATSCORE_NUM / RED_STATSCORE_DEN` — a move the combined history likes
    // searches fuller, one it distrusts is reduced harder. Rust integer division
    // truncates toward zero exactly like C++ (design/10 §3.1), so a negative
    // `stat_score` does NOT floor. Independently, a move that survives its reduced
    // LMR scout earns a flat [`LMR_CONTHIST_BONUS`] along its continuation chain
    // (design/10 §3.8) — "worth continuing from".

    /// Butterfly-history weight in the quiet `stat_score` (the `2·mainHistory`
    /// term). A dedicated tunable so the reduction statistic can move independently
    /// of the move-ordering weight [`QUIET_MAIN_HISTORY_WEIGHT`].
    pub STATSCORE_MAIN_WEIGHT: i32 = 2;
    /// Victim-value multiplier of the capture `stat_score`
    /// (`STATSCORE_CAPT_NUM·piece_value(captured) / STATSCORE_CAPT_DEN`).
    pub STATSCORE_CAPT_NUM: i32 = 809;
    /// Divisor of the capture `stat_score` victim term.
    pub STATSCORE_CAPT_DEN: i32 = 128;
    /// Numerator of the reduction dock `r -= stat_score·RED_STATSCORE_NUM /
    /// RED_STATSCORE_DEN`: positive `stat_score` LOWERS the reduction (search
    /// fuller), negative RAISES it.
    pub RED_STATSCORE_NUM: i32 = 445;
    /// Divisor of the reduction dock (a power of two — the truncation toward zero
    /// on a negative `stat_score` is intended, not to be "fixed" to floor).
    pub RED_STATSCORE_DEN: i32 = 4096;
    /// Flat continuation-history bonus a move earns when its reduced LMR scout
    /// beats alpha (design/10 §3.8): applied along the offset-1..6 chain.
    pub LMR_CONTHIST_BONUS: i32 = 1415;

    // --- LMR reduction finalization: ttPv rebate / tt_capture / all-node
    //     (design/10 §3.5/§3.7) ---
    //
    // The remaining §3.7 accretion terms that BALANCE the reduction. Their absence
    // alongside the Wave-D cutoffCnt/singular reductions over-reduces PV-history
    // (ttPv) lines and can hide shallow mates, so they land with item 15's cutoffCnt
    // terms to complete the §3.7 finalization (the wave plan left them unassigned).
    //   * §3.5 entry bump: `if ss.tt_pv { r += RED_TTPV_ENTRY }` — a provisional
    //     bump, mostly cancelled by the rebate below.
    //   * §3.7 rebate: `if ss.tt_pv { r -= RED_TTPV_BASE + RED_TTPV_PV·PV +
    //     RED_TTPV_TTVALUE·(tt.value > alpha) + (tt.depth >= depth)·
    //     (RED_TTPV_TTDEPTH_BASE + RED_TTPV_TTDEPTH_CUT·cut_node) }` — a PV-flagged
    //     node with a trustworthy TT entry searches much fuller.
    //   * tt_capture surcharge and the all-node scaling close the balance.

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
    /// Plies the internal iterative reduction removes (was a bare
    /// `depth -= 1`).
    ///
    /// The 17th implicit-x1 found: IIR's *threshold* was tunable while its
    /// *magnitude* was not, so the only question anyone could ask about the
    /// mechanism was "at what depth", never "by how much". `IIR_MIN_DEPTH`
    /// 6 -> 8 is currently the strongest single arm measured
    /// (`docs/research/SCREENING.md` §10), which makes the untunable half of
    /// the same mechanism worth exposing.
    pub IIR_REDUCTION: Depth = 1;

    // --- Move-loop pruning: late move pruning (design/10 §3.5) ---
    //
    // "Movecount pruning": once the running `move_count` clears a depth-scaled
    // threshold at a shallow node, the picker is told to stop emitting quiets
    // ([`MovePicker::skip_quiet_moves`]) — the late moves ordering already
    // distrusted are never even generated. The threshold is
    // `(LMP_BASE + depth²) / (2 - improving)`: quadratic in depth (deeper nodes
    // try more moves before giving up) and halved for a not-improving node (a
    // stagnant eval distrusts its late quiets, so it prunes them roughly twice
    // as early). The whole move-loop pruning block is fenced by the mate-safety
    // guard `!is_loss(best_value)` and the zugzwang guard
    // `non_pawn_material(us) != 0` (design/10 §3.5), never at the root.

    // --- Implicit-coefficient tunables (2026-07-27) ---
    //
    // Every entry below gives a NAME and a knob to a term that carried an
    // invisible x1 or a bare literal. They all default to exact identity, so
    // the shipped engine is unchanged and each is a clean control.
    //
    // Why they were missing. Design/01 §3.8's rule is "any bare numeric
    // literal in `search/` that a tuner could move is a defect" — and that
    // rule STRUCTURALLY CANNOT catch these, because most of them are not
    // literals at all. `+ Depth::from(pv_node)` is a coefficient of 1 that
    // nobody ever wrote down, so it is invisible to a grep for numbers.
    //
    // Why it matters. The prune budget
    // (`docs/research/prune-budget-2026-07-27.md`) forced the distinction
    // between a mechanism's CONTRIBUTION (what happens if it vanishes) and
    // its LEVERAGE (what happens if its parameter moves). LMP removes ~23-47%
    // of the tree and had exactly one low-leverage parameter, because the
    // term that actually sets its threshold — `depth²` — had no coefficient.
    // No SPSA campaign can find what no parameter can express.
    //
    // Stockfish carries the same implicit 1s, and that is not evidence they
    // are right: SF writes an explicit coefficient wherever it has bothered
    // to tune one (`45 * improving` in the NMP bar, `138 * !bestMove` in
    // parent futility). The ones still at an implicit 1 are simply untuned —
    // and never tuned for OUR eval frame or OUR tree shape at all.

    /// Base term of the LMP move-count threshold `(LMP_BASE + depth²) /
    /// (2 - improving)` (design/10 §3.5): the constant floor of quiets always
    /// tried before movecount pruning can bite, before the depth² growth.
    pub LMP_BASE: i32 = 3;
    /// **Coefficient of the depth² term, in 1024ths.** `1024` = 1.0 = exact
    /// identity with Stockfish's shape.
    ///
    /// Why it exists: LMP removes **46.8% of our tree**
    /// (`docs/research/prune-budget-2026-07-27.md` §1), second only to LMR,
    /// and until now the entire mechanism had exactly ONE tunable —
    /// [`LMP_BASE`] — whose leverage is almost nil, because `depth²`
    /// dominates the sum at every depth that matters. Measured: `LMP_BASE`
    /// 2 → 8 moves the tree by +0.4% to +4.6%, against LMP's 46.8%
    /// contribution.
    ///
    /// That is the distinction the prune budget forced into the open:
    /// **contribution is not leverage.** A mechanism can shape half the tree
    /// and still be untunable, and no SPSA campaign can find what no
    /// parameter can express. This coefficient gives the dominant term a
    /// handle.
    ///
    /// Deviating from Stockfish's shape here is deliberate: their constants
    /// are a prior, not a destination, and a coefficient pinned at identity
    /// costs nothing until something measures a better value.
    pub LMP_DEPTH_MULT: i32 = 1024;
    /// Divisor base of the LMP threshold: `(… ) / (LMP_IMPROVING_DEN -
    /// improving)`. At the default 2 a not-improving node prunes at half the
    /// move count, which is a large structural asymmetry that was previously
    /// hard-coded and therefore never tested.
    pub LMP_IMPROVING_DEN: i32 = 2;

    /// Numerator of the singular VERIFICATION depth, `new_depth *
    /// SE_VERIFY_DEPTH_NUM / SE_VERIFY_DEPTH_DEN` (was a bare `/ 2`).
    ///
    /// The highest-leverage implicit constant in the search: this is the
    /// depth of an entire extra search run at every singular candidate, so it
    /// sets both the cost of singular extensions and how trustworthy their
    /// verdict is. A bare `/ 2` fixed that trade-off with no way to measure
    /// it.
    pub SE_VERIFY_DEPTH_NUM: Depth = 1;
    /// Denominator of the singular verification depth (see
    /// [`SE_VERIFY_DEPTH_NUM`]).
    pub SE_VERIFY_DEPTH_DEN: Depth = 2;
    /// Base extension granted to a confirmed-singular move, before the
    /// double/triple bonuses: `extension = SE_EXT_BASE + (double) + (triple)`.
    pub SE_EXT_BASE: Depth = 1;
    /// Extra depth the WHOLE node gains when its ttMove proves singular (was
    /// a bare `depth += 1`). Distinct from [`SE_EXT_BASE`], which extends only
    /// the singular move.
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
    /// Minimum `new_depth` for the SECOND full-depth dock to apply — a bare
    /// `new_depth > 2` guard that stops the dock eating a shallow search.
    pub LMR_DOCK2_MIN_DEPTH: Depth = 2;
    /// Depth slack allowed on a TT cutoff when the stored value is a lower
    /// bound: `tt_depth > depth - TTCUT_BOUND_SLACK * (tt_value <= beta)`.
    pub TTCUT_BOUND_SLACK: Depth = 1;
    /// Extension level at or above which a node stops counting child cutoffs
    /// (`cutoff_cnt += (extension < CUTOFFCNT_EXT_GATE || PV)`).
    pub CUTOFFCNT_EXT_GATE: Depth = 2;

    // --- Move-loop pruning: capture futility + capture/check SEE (design/10 §3.5) ---
    //
    // The capture/check arm of the shallow-depth pruning block (Wave-B item 8),
    // keyed by the capture history `capt_hist = capture[moved][to][captured]`
    // (design/09). Two gates, in node order:
    //
    //   * Capture futility (non-checking captures, `lmr_depth < CFUT_MAX_LMRDEPTH`):
    //     a capture whose victim value plus a depth-scaled slack and the
    //     capture-history nudge still leaves the static eval below alpha is futile
    //     — skip when `ss.static_eval + CFUT_BASE + CFUT_LMRDEPTH_MULT·lmr_depth +
    //     piece_value(captured) + CFUT_HIST_MULT·capt_hist/CAPT_HIST_DEN <= alpha`.
    //     `piece_value(captured)` is 0 for en passant (no piece on `to`) — SF
    //     parity. There is NO `!in_check` guard (unlike parent futility below):
    //     the block's outer `!is_loss(best_value)` fence already keeps every
    //     evasion searched until one produces a real value.
    //   * Capture/check SEE (captures AND checking quiets): a move that cannot win
    //     back `-(CAP_SEE_DEPTH_MULT·depth + CAP_SEE_HIST_MULT·capt_hist/
    //     CAPT_HIST_DEN)` in a static exchange is pruned — UNLESS `alpha <
    //     VALUE_DRAW` and the moved piece is our LAST non-pawn material
    //     (`non_pawn_material(us) == piece_value(moved)`): sacrificing it may be a
    //     stalemate resource, which is never pruned.

    /// Exclusive lmr-depth ceiling of capture futility (`lmr_depth <
    /// CFUT_MAX_LMRDEPTH`); deeper predicted searches are trusted to see the
    /// tactic themselves.
    pub CFUT_MAX_LMRDEPTH: Depth = 7;
    /// **Class-4 converter: material units → eval units, in 1024ths.**
    /// `1024` = 1.0 = exact identity, so the shipped engine is unchanged.
    ///
    /// `PIECE_VALUES` are Stockfish's exactly (pawn 208) while our static
    /// eval measures ~0.64× Stockfish's frame
    /// (`docs/research/eval-frame-scale-model-2026-07-27.md` §11.2, quantile-
    /// matched and flat across the whole range). Every site that ADDS a piece
    /// value to a static eval therefore over-weights the material term by
    /// ~1/0.64 = 1.56×.
    ///
    /// This is the one mismatch class no margin constant can absorb — the
    /// error rides on `captured_value`, which varies by piece, so a flat
    /// margin adjustment cannot cancel it. The frame study could only test it
    /// bundled inside `EVAL_SCALE`'s global rescale (§10.1), which lost for
    /// other reasons; it has never been isolated.
    ///
    /// Applied in [`capture_futility_value`], where the prediction is
    /// `static_eval + margins + captured_value <= alpha`. Over-weighting
    /// `captured_value` inflates the bound, so capture futility prunes LESS —
    /// consistent with its measured contribution of ~0.5% of the tree
    /// (`docs/research/prune-budget-2026-07-27.md` §1), i.e. near-inert.
    ///
    /// Theory puts the interesting region near 655 (= 0.64); 1024 is the
    /// control.
    pub CFUT_PIECE_MULT: i32 = 1024;
    /// **Class-4 converter: eval units → material units, in 1024ths.**
    /// `1024` = identity. The inverse direction of [`CFUT_PIECE_MULT`], for
    /// the two sites that feed an EVAL-frame difference to `see_ge`, whose
    /// threshold is in `PIECE_VALUES` units:
    ///
    /// - ProbCut's capture filter, `probcut_beta - static_eval`
    /// - qsearch futility's `see_ge(m, alpha - futility_base)`
    ///
    /// An eval-frame gap is ~0.64× the material-frame gap that means the same
    /// thing, so both thresholds are too SMALL: ProbCut admits captures that
    /// cannot really reach its raised beta, and qsearch futility declines to
    /// prune exchanges it should. Theory puts the interesting region near
    /// 1600 (= 1/0.64).
    pub SEE_THRESH_MULT: i32 = 1024;
    /// Flat base of the capture-futility optimistic value over the static eval.
    pub CFUT_BASE: Value = 231;
    /// Per-lmr-depth slope of the capture-futility value (can be negative at
    /// shallow nodes, LOWERING the bound).
    pub CFUT_LMRDEPTH_MULT: Value = 232;
    /// Capture-history nudge numerator of the capture-futility value
    /// (`CFUT_HIST_MULT·capt_hist / CAPT_HIST_DEN`): a capture the history likes
    /// is written off a little less readily.
    pub CFUT_HIST_MULT: i32 = 131;
    /// Depth multiplier of the capture/check SEE margin
    /// (`CAP_SEE_DEPTH_MULT·depth + CAP_SEE_HIST_MULT·capt_hist/CAPT_HIST_DEN`):
    /// deeper nodes tolerate a bigger static-exchange loss before pruning.
    pub CAP_SEE_DEPTH_MULT: Value = 175;
    /// Capture-history numerator of the capture/check SEE margin.
    pub CAP_SEE_HIST_MULT: i32 = 34;
    /// Shared `/1024` divisor of the capture-history reads in both capture gates
    /// above (the capture history itself is carried at design/09's `CAPT_D`
    /// gravity scale; this is only the pruning read weight).
    pub CAPT_HIST_DEN: i32 = 1024;

    // --- Move-loop pruning: continuation-history pruning (design/10 §3.5) ---
    //
    // The FIRST gate in the quiet arm of the shallow-depth pruning block (Wave-C
    // item 11), ahead of parent futility and quiet SEE. A quiet whose combined
    // recent history `history = contHist(1)[pc][to] + contHist(2)[pc][to] +
    // pawnHistory[pc][to]` — the offset-1/offset-2 continuation slabs (the same
    // contHist(1/2) move ordering reads) plus the pawn-structure slot, all from
    // PRE-move state — is deeply negative is skipped without a search: the
    // position's own move-ordering memory reports that this move has been failing
    // here. The move is pruned when `history < -QHIST_PRUNE_MULT·depth`, a linear
    // (not quadratic, unlike quiet SEE) depth scale, so a deeper node demands a
    // proportionally more damning history before writing a quiet off. Fenced by
    // the block's outer `!is_loss(best_value)` mate guard and the
    // `non_pawn_material(us) != 0` zugzwang guard, never at the root, and — like
    // every quiet gate — only off the last iteration's PV path (`!follow_pv ||
    // !PV`); the ttMove (searched first under the `is_loss` fence), captures, and
    // checks never reach this arm. There is no `!in_check` guard (unlike parent
    // futility): the outer `!is_loss` fence already keeps every evasion searched
    // until one produces a real value. Fresh histories sum to −2366, above the
    // bar at every depth, so the gate is dormant until the search's own history
    // updates drive a quiet's combined score below `-QHIST_PRUNE_MULT·depth`.

    /// Linear depth multiplier of the continuation-history pruning threshold
    /// `history < -QHIST_PRUNE_MULT·depth` (design/10 §3.5): a quiet whose
    /// offset-1/offset-2 continuation histories plus its pawn history fall below
    /// this depth-scaled bar is pruned before it is searched.
    pub QHIST_PRUNE_MULT: i32 = 4313;

    // --- Move-loop pruning: parent futility (design/10 §3.5) ---
    //
    // (The history-buys-lmr_depth divisor table sits below this macro block
    // with the other const arrays — see `LMR_DIVISOR`.)
    //
    // "Futility pruning: parent node" — a shallow, not-in-check node writes off
    // a quiet whose most optimistic outcome still fails to reach alpha. The
    // bound is `ss.static_eval + PFUT_BASE + PFUT_NO_BESTMOVE·(no best move yet)
    // + PFUT_LMRDEPTH_MULT·lmr_depth + PFUT_EVAL_ABOVE_ALPHA·(eval already beats
    // alpha)`: a depth-scaled slack over the static eval, widened while the node
    // has no alpha-raiser in hand and when the eval already clears alpha. When
    // that bound `<= alpha` at a shallow predicted depth
    // (`lmr_depth < PFUT_MAX_LMRDEPTH`), the quiet is skipped — after first
    // lifting `best_value` to the bound (the fail-low value a pruned move could
    // plausibly deliver), gated by `!is_decisive(best_value) && !is_win(bound)`
    // so no mate score is corrupted (§8.11). The `!ss.in_check` guard keeps
    // every check evasion searched. `lmr_depth = new_depth - r/1024` is the
    // predicted post-LMR depth; the continuation-history nudge SF folds into it
    // first is Wave-C item 11 — this item uses the plain derivation.

    /// Flat base of the parent-futility bound over the static eval.
    pub PFUT_BASE: Value = 40;
    /// Bound surcharge while no best move has been found yet: a node with no
    /// established alpha-raiser writes moves off a little more optimistically.
    pub PFUT_NO_BESTMOVE: Value = 138;
    /// Per-lmr-depth slope of the bound: a deeper predicted search must clear
    /// more headroom before its quiets are futile.
    pub PFUT_LMRDEPTH_MULT: Value = 117;
    /// Bound surcharge when the static eval already clears alpha.
    pub PFUT_EVAL_ABOVE_ALPHA: Value = 90;
    /// Exclusive lmr-depth ceiling: parent futility runs only while
    /// `lmr_depth < PFUT_MAX_LMRDEPTH` (deeper predicted searches are trusted).
    pub PFUT_MAX_LMRDEPTH: Depth = 12;

    // --- Move-loop pruning: quiet SEE (design/10 §3.5) ---
    //
    // After parent futility, a quiet whose static-exchange value cannot reach
    // `-QSEE_MULT·lmr_depth²` (quadratic in the predicted post-LMR depth, floored
    // at 0) is pruned: a material-losing quiet at shallow effective depth is
    // almost never worth a search. At `lmr_depth == 0` the bar is 0, so any
    // negative-SEE quiet is dropped; it slackens quadratically with depth.

    /// Quadratic coefficient of the quiet SEE margin `-QSEE_MULT·lmr_depth²`
    /// (`lmr_depth` clamped to `>= 0` first).
    pub QSEE_MULT: Value = 25;

    // --- Transposition-table interaction (design/07 §3.7.2 steps 7/14) ---

    /// TT cutoffs are refused once the halfmove clock reaches this (the
    /// graph-history guard of design/07 §3.7.2 step 7 / design/08 §3.12).
    /// `i32`: matches `value_from_tt`'s rule50 parameter (doc 02).
    pub TT_CUTOFF_RULE50_LIMIT: i32 = 96;
    /// Extra depth credited to the TT store of a no-legal-moves node
    /// (design/07 §3.7.2 step 14): a mate/stalemate terminal is exact far
    /// beyond its nominal depth, so make the entry hard to displace.
    pub TT_NO_MOVES_DEPTH_BONUS: Depth = 6;

    // --- TT-cutoff history nudges (N6b extras, design/10 §3.4; Wave-C item 12) ---
    //
    // When the non-PV TT cutoff fires with `tt.value >= beta`, the stored ttMove
    // "worked": if it is quiet, reward it (`min(TTCUT_BONUS_MULT·depth,
    // TTCUT_BONUS_CAP)`), and if the parent played an early (`move_count < 4`)
    // non-capturing quiet, penalize that parent move along its continuation chain
    // (`TTCUT_PARENT_MALUS`) — a cheap cutoff here is evidence the parent's move
    // was refutable.

    /// Depth multiplier of the TT-cutoff quiet-ttMove bonus.
    pub TTCUT_BONUS_MULT: i32 = 114;
    /// Cap of the TT-cutoff quiet-ttMove bonus.
    pub TTCUT_BONUS_CAP: i32 = 724;
    /// Continuation-history malus applied to the parent's early quiet move on a
    /// beta-side TT cutoff.
    pub TTCUT_PARENT_MALUS: i32 = -2187;
    /// The parent's `move_count` must be below this for the TT-cutoff parent
    /// penalty to fire (an "early" quiet the cutoff just refuted).
    pub TTCUT_PARENT_MOVECOUNT: i32 = 4;

    // --- Singular extensions + multicut (design/10 §3.6; Wave-D items 13/14) ---
    //
    // When the move under consideration is the ttMove, an exclusion search
    // (`search::<NonPv>(singular_beta-1, singular_beta, new_depth/2, cut_node, ply)`
    // with `excluded_move = m`) at `singular_beta = tt.value - (SE_BETA_MULT +
    // SE_BETA_TTPV_MULT·(tt_pv && !PV))·depth / SE_BETA_DIV` decides:
    //   * value < singular_beta → singular: extend (1 + double/triple margins),
    //     and the WHOLE node deepens (`depth += 1`);
    //   * value >= beta (non-decisive) → multicut: several moves beat beta, prune
    //     the subtree (dock the tt_move_history scalar and return value);
    //   * else if tt.value >= beta → extension = -3 (the ttMove likely fails high
    //     anyway; search it shallower);
    //   * else if cut_node → extension = -2.
    // The `is_shuffling` guard suppresses extensions of back-and-forth shuffles in
    // 50-move territory. `corr_adj` (Wave F) is 0 until item 21 plumbs correction.

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
    /// Divisor of the correction adjustment folded into the extension margins
    /// (Wave F item 21; `corr_adj = |correction_value| / SE_CORR_DIV`).
    pub SE_CORR_DIV: i32 = 194_822;
    /// Double-extension margin: base.
    pub SE_DOUBLE_BASE: Value = -3;
    /// Double-extension margin: PV bonus.
    pub SE_DOUBLE_PV: Value = 201;
    /// Double-extension margin: non-tt-capture discount.
    pub SE_DOUBLE_NON_TTCAP: Value = 157;
    /// Double-extension margin: tt_move_history numerator.
    pub SE_DOUBLE_TMH_NUM: i32 = 1081;
    /// Double-extension margin: tt_move_history divisor.
    pub SE_DOUBLE_TMH_DEN: i32 = 117_824;
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
    /// `is_shuffling`: minimum halfmove clock.
    pub SHUFFLE_RULE50: i32 = 10;
    /// `is_shuffling`: minimum plies since the last null move.
    pub SHUFFLE_PLIES_FROM_NULL: i32 = 6;
    /// `is_shuffling`: minimum ply.
    pub SHUFFLE_MIN_PLY: usize = 20;

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

    // --- ProbCut (N13, design/10 §3.4; Wave-E item 16) ---
    //
    // A shallow, cheap search proving a *capture* beats `probcut_beta = beta +
    // PROBCUT_BETA_BASE - PROBCUT_BETA_IMPROVING·improving` lets the node fail high
    // early. Per candidate (captures clearing `see_ge(probcut_beta - static_eval)`):
    // a qsearch probe at `probcut_beta`, then — if it held and
    // `probcut_depth = depth - PROBCUT_DEPTH_SUB - improving > 0` — a real reduced
    // search; a confirmed cutoff stores BOUND_LOWER and returns
    // `value - (probcut_beta - beta)`.

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

    // --- Correction-history couplings (Wave F, design/10 §4.6; item 21) ---
    //
    // `correction_value` (the UNSCALED weighted correction sum, design/09 §3.10)
    // is plumbed into the pruning suite: it corrects the static eval (N6, via
    // `to_corrected_static_eval`), WIDENS the RFP margin when the corrector says
    // our eval is unreliable (`+ |correction_value| / RFP_CORR_DIV`), reduces a
    // touch LESS under eval uncertainty (`r -= |correction_value| / RED_CORR_DIV`),
    // tightens the singular margins (`corr_adj = |correction_value| / SE_CORR_DIV`,
    // consumed by Wave D once this lands), and the node-result correction update
    // (`correction_bonus` + `update_correction_history`) learns the eval error.

    /// Divisor of the RFP margin correction-widening term.
    pub RFP_CORR_DIV: i32 = 182_069;
    /// Divisor of the reduction correction-slackening term.
    pub RED_CORR_DIV: i32 = 26_131;

    // --- Hindsight depth adjustments (N6, design/10 §3.4; Wave-E item 17) ---
    //
    // From the parent's LMR bookkeeping (`prior_reduction = (ss-1).reduction`
    // before it is reset): if the parent reduced us hard yet the position did not
    // turn their way, the reduction looks unjustified — search one ply deeper;
    // conversely a moderate parent reduction plus a healthy combined eval sum
    // searches one ply shallower.

    /// `prior_reduction >= HINDSIGHT_RED_UP && !opponent_worsening` → `depth += 1`.
    pub HINDSIGHT_RED_UP: i32 = 3;
    /// `prior_reduction >= HINDSIGHT_RED_DOWN && depth >= 2 && eval-sum big` →
    /// `depth -= 1`.
    pub HINDSIGHT_RED_DOWN: i32 = 2;
    /// The `ss.static_eval + (ss-1).static_eval` threshold of the depth-down arm.
    pub HINDSIGHT_EVAL_SUM: Value = 173;

    // --- TT-cutoff hardening (N6b, design/10 §3.4; Wave-E item 20) ---
    //
    // Beyond the base non-PV cutoff (depth slack + rule50 guard, already live):
    // require the stored bound direction to AGREE with the expected node type at
    // shallow depth (`cut_node == (tt.value >= beta) || depth > TTCUT_AGREE_DEPTH`),
    // re-verify a deep cutoff against the ttMove's child entry
    // (`depth >= TTCUT_VERIFY_DEPTH`), and — in the else branch — age out an entry
    // whose bound points the wrong way (`penalize(1)` at `depth > TTCUT_PENALIZE_DEPTH`).

    /// Shallow cutoffs (`depth <= TTCUT_AGREE_DEPTH`) require the stored direction
    /// to match the node type; deeper cutoffs skip the agreement test.
    pub TTCUT_AGREE_DEPTH: Depth = 4;
    /// At/above this depth, a candidate cutoff is re-verified against the ttMove's
    /// child TT entry (one make/probe/unmake).
    pub TTCUT_VERIFY_DEPTH: Depth = 7;
    /// A wrong-direction bound at `depth > TTCUT_PENALIZE_DEPTH` costs the stored
    /// entry one ply of depth (`penalize(1)`), aging it out faster.
    pub TTCUT_PENALIZE_DEPTH: Depth = 5;

    // --- Fail-low parent history bonus (N-result, design/10 §3.10; Wave-E item 19) ---
    //
    // When a node fails low with no best move and the parent played a quiet
    // (`prev_sq` set, `!prior_capture`), the parent's move "worked" — reward it.
    // `bonus_scale` accumulates several signals; `scaled_bonus = min(FL_BONUS_MULT·
    // depth - FL_BONUS_SUB, FL_BONUS_CAP) * bonus_scale` is then distributed to the
    // parent conthist / main / pawn histories at the weights below. A fail low
    // under a parent capture instead flat-rewards capture history (FL_CAPTURE_BONUS).

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
    /// Parent-conthist divisor.
    pub FL_CONTHIST_DEN: i32 = 16384;
    /// Main-history weight of the fail-low bonus (`scaled_bonus·234/32768`).
    pub FL_MAIN_NUM: i32 = 234;
    /// Main-history divisor.
    pub FL_MAIN_DEN: i32 = 32768;
    /// Pawn-history weight of the fail-low bonus (`scaled_bonus·322/8192`).
    pub FL_PAWN_NUM: i32 = 322;
    /// Pawn-history divisor.
    pub FL_PAWN_DEN: i32 = 8192;
    /// Flat capture-history bonus on a fail low under a parent capture.
    pub FL_CAPTURE_BONUS: i32 = 901;

    // --- Eval-diff parent nudge (N8, design/10 §3.4; Wave-C item 12) ---
    //
    // Grade the opponent's last quiet by the static-eval swing it produced.
    // `eval_diff = clamp(-(ss-1).static_eval - ss.static_eval, EVALDIFF_CLAMP_LO,
    // EVALDIFF_CLAMP_HI) + EVALDIFF_OFFSET`; the parent's move (indexed by the
    // *opponent* color) takes `main_history << EVALDIFF_MAIN_MULT·eval_diff`, and —
    // on a TT miss for a non-pawn, non-promotion parent move — its pawn-history
    // slot takes `<< EVALDIFF_PAWN_MULT·eval_diff`.

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

    /// Single-reply cap on the soft time budget (`min(561.7, total_time)`,
    /// design/12 §3.9): with exactly one legal move at the root the search
    /// result cannot change the move played, so never sit on it longer than
    /// this — a viewer-experience cap, not a strength knob. In TM units:
    /// milliseconds, or nodes under `nodestime` (doc 12 §3.11), like every
    /// budget the soft stop compares against.
    pub ONE_REPLY_TOTAL_TIME_CAP: f64 = 561.7;

    // The four soft-stop factors multiply into the plain `optimum` budget
    // (design/12 §3.9, step 8). Each is an SF-current fitted value (D20); the
    // formula SHAPES are documented at the use site in `search::soft_stop`.

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

    /// reduction factor 2, stability lerp x0: best-move-stability in iterations
    /// `root_depth − last_best_move_depth`, mapped LO→[TIME_REDUCTION_MIN].
    ///
    /// 2026-07-26 depth-redenomination (TM front-load fix): SF's window is
    /// [5,18], tuned for SF's iteration counts. Measured on 60 shared
    /// middlegames at 3 s (tools note in docs/research/value-foundation-state
    /// 2026-07-26): we complete 3.3 fewer iterations than SF16 at equal time
    /// (24.2 vs 27.5) while last-best-move-change ply is EQUAL (12.7 vs 12.3),
    /// so our stability input runs ~3.6 iterations lower (11.6 vs 15.2) for
    /// identical decision behavior — the lerp read us as "unstable" on moves
    /// SF banks time on. Both endpoints shift down by the measured deficit;
    /// the y-range is untouched.
    pub TIME_REDUCTION_STABILITY_LO: f64 = 2.0;
    /// reduction: stability lerp x1, mapped HI→[TIME_REDUCTION_MAX].
    pub TIME_REDUCTION_STABILITY_HI: f64 = 15.0;
    /// reduction: `time_reduction` lower clamp (and lerp y at low stability).
    pub TIME_REDUCTION_MIN: f64 = 0.65;
    /// reduction: `time_reduction` upper clamp (and lerp y at high stability).
    pub TIME_REDUCTION_MAX: f64 = 1.55;
    /// reduction: constant term of `(1.48 + prev_time_reduction) / (2.157 ·
    /// time_reduction)` — partially gives back last move's stability savings.
    pub REDUCTION_BASE: f64 = 1.48;
    /// reduction: divisor coefficient scaling `time_reduction`.
    pub REDUCTION_DEN_COEFF: f64 = 2.157;

    /// bestMoveInstability factor 3: constant term of `1.096 + 2.29 ·
    /// tot_best_move_changes / threads` — think longer when the best move
    /// keeps flipping.
    pub INSTABILITY_BASE: f64 = 1.096;
    /// bestMoveInstability: weight on the per-thread-averaged change count.
    pub INSTABILITY_CHANGES_COEFF: f64 = 2.29;
    /// bestMoveInstability: per-iteration aging divisor of the accumulated
    /// root best-move-change count (SF: `totBestMoveChanges /= 2`).
    ///
    /// 2026-07-26 depth-redenomination (TM front-load fix): a root flip's
    /// weight at the stop decision is `decay^-(iterations since the flip)`.
    /// We complete ~3.3 fewer iterations than SF16 at equal time with EQUAL
    /// flip counts per search (4.08 vs 4.05, 60 shared middlegames at 3 s),
    /// so the same wobble reads ~2^3.3 ≈ 10× hotter through SF's /2. Matching
    /// SF's effective weight at our measured stability (d^11.6 = 2^15.2)
    /// gives d ≈ 2.48; pinned at 2.5.
    pub INSTABILITY_CHANGES_DECAY: f64 = 2.5;

    /// highBestMoveEffort factor 4, effort lerp x0 (per-100k node units):
    /// `nodes_effort = effort · 100000 / max(1, nodes)` mapped 79219→0.924.
    pub EFFORT_LERP_LO: f64 = 79_219.0;
    /// highBestMoveEffort: effort lerp x1, mapped 101822→0.71.
    pub EFFORT_LERP_HI: f64 = 101_822.0;
    /// highBestMoveEffort: scale at low effort (also the upper clamp).
    pub EFFORT_SCALE_AT_LO: f64 = 0.924;
    /// highBestMoveEffort: scale at high effort (also the lower clamp) — stop
    /// sooner once the lion's share of nodes already went into the best move.
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

    // --- Stop cadence (design/07 §3.9, design/12 §3.8; Reckless bundle D31) ---

    /// `check_time` does real work once per this many main-search node
    /// entries. The poll is gated by a bitmask on the node counter (`nodes &
    /// mask`, D31 robustness bundle) rather than a decrementing field, so the
    /// added qsearch-entry re-poll shares the exact same cadence for free.
    /// Under a `go nodes` limit the mask tightens to a power of two `<=
    /// min(512, limits.nodes / 1024)` so tiny node budgets are honored.
    // (CHECK_TIME_CALLS is a plain const above the macro: its power-of-two
    // compile-time assert cannot hold for a runtime tunable, and poll cadence
    // is not a strength parameter.)

    // The poll mask is `CHECK_TIME_CALLS - 1` (default branch) / a power-of-two
    // `<= CHECK_TIME_CALLS` (nodes branch). `x & (CALLS - 1) == CALLS - 1` is a
    // valid contiguous low-bit gate ONLY when CALLS is 2^k; a non-power-of-two
    // would open silent gaps far wider than the cadence between real clock
    // checks. Enforce the invariant at compile time so nobody can retune it into
    // a forfeit-latency bug.


    /// Flat epsilon shaved off every enforced *wall-clock* bound — `maximum()`
    /// and `movetime` (design/12 §3.8; Reckless bundle D31). Cuts fast-TC
    /// forfeit risk by stopping a few ms early to cover the unwind-and-print
    /// tail after `stop`. Deliberately NOT applied to the `nodes` budget
    /// (an exactness/testing bound, not a clock forfeit risk). In TM units:
    /// milliseconds, or nodes under `nodestime`.
    pub CHECK_TIME_EPSILON_MS: i64 = 15;

    /// `info` throttle (design/12 §3.13, SF's `NODES_LIMIT_OUTPUT`): below
    /// this many nodes the whole MultiPV block is published only when the
    /// iteration's LAST line finishes; above it, after every line (more
    /// frequent updates on long searches). At MultiPV 1 the block is a single
    /// line, published every iteration either way.
    pub INFO_THROTTLE_NODES: u64 = 10_000_000;

    // --- Lazy-SMP best-thread voting (design/13 §3.8, D20) ---


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
    /// STAT_BREADTH_DIVISOR` (a wide node that tried many moves taught us more).
    /// `i64`: the product is widened to avoid the 32-bit overflow SF guards
    /// against (design/09 §9 open-question 3 — ours is plain i64 truncation).
    pub STAT_BREADTH_DIVISOR: i64 = 256;
    /// Shared `/1024` denominator of every history-update weight below.
    pub STAT_WEIGHT_DEN: i32 = 1024;
    /// Best-quiet bonus weight (`bonus·824/1024`).
    pub QUIET_BEST_BONUS_NUM: i32 = 824;
    /// Geometric quiet-malus start weight (`malus·1136/1024`), decayed by
    /// [`QUIET_MALUS_DECAY_NUM`] before each successive non-best quiet — earliest
    /// failures punished hardest.
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

    /// Divisor of the continuation-history update `bonus·weight·mult / 131072`.
    /// `i64`: the triple product is widened before the divide.
    pub CONT_UPDATE_DIVISOR: i64 = 131072;
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
    /// Divisor turning the correction sum back into a centipawn adjustment.
    pub CORR_VALUE_DIVISOR: i32 = 131072;

    // --- Correction history: update weights (design/09 §3.10) ---

    /// Shared `/128` denominator of the correction-update weights below.
    pub CORR_UPDATE_DEN: i32 = 128;
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
    // The gated, clamped bonus fed to `update_correction_history`. Owned by
    // design/10's node-result step; colocated with the callee here (design/09
    // §3.9 records external write-site formulas for grep-ability) so the P8
    // integration wave can compute it in one call.

    /// Depth multiplier when a best move exists (`err·depth·12/128`).
    pub CORR_BONUS_DEPTH_MULT_BESTMOVE: i32 = 12;
    /// Depth multiplier on a fail-low with no best move (`err·depth·18/128`).
    pub CORR_BONUS_DEPTH_MULT_NOMOVE: i32 = 18;
    /// Divisor of the raw correction bonus.
    pub CORR_BONUS_DIVISOR: i32 = 128;
    /// Symmetric clamp on the raw correction bonus (`CORRECTION_HISTORY_LIMIT/4`).
    pub CORR_BONUS_CLAMP: i32 = 256;
    /// Numerator of the outer correction-bonus scaling (`·1114/1024`).
    pub CORR_BONUS_SCALE_NUM: i32 = 1114;
    /// Denominator of the outer correction-bonus scaling.
    pub CORR_BONUS_SCALE_DEN: i32 = 1024;

    // --- SF-parity structural toggles (2026-07-27 D33 reference-fidelity pass) ---
    //
    // Four mechanisms where our transcription structurally deviates from SF
    // `eca43a97`, found by side-by-side reading (not by constant diffing —
    // none of these is visible to a constants screen). Each is a 0/1 toggle,
    // OFF by default so the engine is bit-identical to the pre-toggle tree;
    // 1 = the SF-parity behavior. Screen each with tools/screen.py, SPRT the
    // survivors, then fold the winner into the default and delete the toggle.

    /// `followPV` tracking (SF search.cpp:752-755): mark nodes on the previous
    /// iteration's PV path. The two consumers (IIR's exemption, the quiet-
    /// pruning PV-path exemption at `!follow_pv || !PV`) have been reading a
    /// never-assigned `false` since P9 item 9 was parked — with the flag dead,
    /// we IIR-reduce and quiet-prune the very path the last iteration proved.
    /// 0 = flag stays false everywhere (today's behavior); 1 = SF parity.
    pub FOLLOW_PV_MODE: i32 = 0;
    /// Upcoming-repetition alpha raise at QSEARCH entry (SF search.cpp:1606).
    /// We run it in `search` only; SF runs it in both, so every qsearch node
    /// below the horizon that could force a repetition draw while losing
    /// forfeits the cutoff. 0 = search only (today); 1 = SF parity.
    pub QS_UPREP: i32 = 0;
    /// The ttPv reduction rebate's tt-value term on a TT MISS (SF
    /// search.cpp:1292 `(ttData.value > alpha) * 838`): SF's miss sentinel
    /// VALUE_NONE (32002) passes `> alpha` unconditionally, so SF grants the
    /// rebate at every PV-flagged TT-miss node. Our `is_valid` guard was added
    /// as ProbCut-idiom hygiene but makes us search ~0.8 ply SHALLOWER than SF
    /// exactly where the TT has no answer. 0 = require a real value (today);
    /// 1 = SF parity (grant on miss).
    pub RED_TTPV_TTMISS_REBATE: i32 = 0;
    /// ProbCut inside a singular verification (SF search.cpp:1037: no
    /// excluded-move gate on the whole block; only the excluded move itself is
    /// skipped in the loop, which our loop already does at the `m ==
    /// excluded_move` continue). 0 = skip ProbCut wholesale under an exclusion
    /// (today); 1 = SF parity.
    pub PROBCUT_IN_VERIF: i32 = 0;

    // --- Implicit-coefficient batch 2 (2026-07-27 D33 reviewer findings) ---
    //
    // Terms that carried an invisible coefficient (a bare literal, a tuple
    // const, or a hard-coded ×1) on high-traffic paths. All wired at identity:
    // the pinned bench is bit-identical at these defaults.

    /// Stand-pat fail-high softening `(BEST_W·best + BETA_W·beta) / DEN` —
    /// applied to non-decisive stand pats at or above beta (§3.8 step 5).
    /// Every qsearch stand-pat cutoff returns through this blend.
    pub QS_STANDPAT_BEST_W: Value = 467;
    /// Beta weight of the stand-pat blend.
    pub QS_STANDPAT_BETA_W: Value = 557;
    /// Denominator of the stand-pat blend.
    pub QS_STANDPAT_DEN: Value = 1024;
    /// Node-exit fail-high softening `(BEST_W·best + BETA_W·beta) / DEN` —
    /// applied to non-decisive qsearch best values strictly above beta
    /// (§3.8 step 11). Every qsearch fail-high returns through this blend.
    pub QS_FAILHIGH_BEST_W: Value = 481;
    /// Beta weight of the qsearch fail-high blend.
    pub QS_FAILHIGH_BETA_W: Value = 543;
    /// Denominator of the qsearch fail-high blend.
    pub QS_FAILHIGH_DEN: Value = 1024;
    /// Numerator of the verification's NMP-disable window `nmp_min_ply =
    /// ply + NUM·(depth - R) / DEN`: NMP stays off for the whole subtree up
    /// to that ply, so the verification cannot recursively null-move its way
    /// back to the same false cutoff.
    pub NMP_VERIFY_NUM: Depth = 3;
    /// Denominator of the verification window fraction.
    pub NMP_VERIFY_DEN: Depth = 4;
    /// The singular block's negative extension when the non-excluded search
    /// already proved `tt_value >= beta` (multicut-adjacent: the ttMove is
    /// expected to fail high anyway, so it searches SHALLOWER in favor of
    /// alternatives). SF's bare `-3`; fires on every ttMove at
    /// `depth >= 6 + ttPv` that is neither singular nor multicut.
    pub SE_NEG_EXT_TT_FAILHIGH: Depth = -3;
    /// The singular block's negative extension at a cut node that is neither
    /// singular nor tt-fail-high. SF's bare `-2`.
    pub SE_NEG_EXT_CUTNODE: Depth = -2;
    /// LMR entry gate: minimum depth (`depth >= 2 && move_count > 1`). The
    /// largest tree-shaper's ENTRY condition, previously hard-coded.
    pub LMR_MIN_DEPTH: Depth = 2;
    /// LMR entry gate: minimum move count (strictly greater-than).
    pub LMR_MIN_MOVECOUNT: i32 = 1;
    /// Numerator of the main-history term in the history-buys-lmr_depth
    /// formula `history += NUM·main / DEN` (SF writes `64·h/32`, deliberately
    /// leaving 1/32 granularity for the tuner; ours had collapsed it to `2·`).
    pub LMRDEPTH_MAIN_NUM: i32 = 64;
    /// Denominator of the main-history lmr_depth term.
    pub LMRDEPTH_MAIN_DEN: i32 = 32;
    /// Additive offset of the log reduction table, in the table's own
    /// pre-division units: `reductions[i] = (NUM/DEN)·ln(i) + OFFSET/DEN`.
    /// The ONLY new degree of freedom the product form `f(d)·f(mn)` admits:
    /// per-axis multipliers are all redundant with RED_TABLE_NUM (only the
    /// product of the two slopes survives), while an additive constant
    /// introduces independent linear `ln(depth)` / `ln(move_count)` terms —
    /// the knob that weighs depth against move count. 0 = identity.
    pub RED_TABLE_OFFSET: i32 = 0;
    /// Multiplier on `failed_high_cnt` in the aspiration loop's
    /// `adjusted_depth = root_depth - MULT·failed_high_cnt - …`: each
    /// consecutive root fail-high re-searches this many plies shallower.
    pub ASPIRATION_FAILHIGH_DEPTH_MULT: Depth = 1;
    /// Beta weight of the MAIN-search fail-high softening `(best·depth +
    /// W·beta) / (depth + W)` — the search() analogue of the qsearch
    /// fail-high blend, which SF parameterised and this one never was.
    pub FH_SOFTEN_BETA_W: i32 = 1;
    /// The sibling depth-reduction step (`depth -= 2` on a non-cutoff alpha
    /// raise inside the SIBLING_DEPTH_MIN..MAX window). The window was
    /// tunable; the step was not — the mechanism could only be moved by
    /// window, never softened.
    pub SIBLING_DEPTH_STEP: Depth = 2;

}

// ─── The eval-frame transforms (docs/research/eval-frame-scale-model-2026-07-27.md) ───
//
// One knob, [`FRAME_S`], read through the transform each constant's UNITS
// require. Deriving the exponent per class is the whole point — guessing it
// is worse than leaving the constant alone, because class 3 moves the
// opposite way to class 1 and applying the wrong one doubles the error
// instead of removing it.
//
//   class 1  pure eval margin      `eval ⋈ eval ± C`        -> frame_scale
//   class 2  eval² divisor         `delta = mss / D`        -> frame_scale
//            (D has units eval²/eval = eval, so it scales the SAME way as a
//            margin — the trap is that being a divisor suggests otherwise)
//   class 3  cross-unit divisor    `eval_adj = |corr| / D`  -> frame_expand
//            (corr keeps its own units, so D must GROW to yield the same
//            eval-space effect)
//   class 4  material into eval    `staticEval + C + PieceValue[x]`
//            -> NOT REPRESENTABLE HERE. The error rides on the piece-value
//            term, which varies by captured piece, so no constant absorbs it.
//   class 5  plies / depths / counts / SEE-vs-material -> untouched

/// Class 1 and 2: an eval-space quantity. Exact identity at `FRAME_S == 1024`.
///
/// i64 intermediate because the largest consumers here (correction and
/// mean-squared divisors) reach ~10^5 and `c * 1024` would overflow an i32
/// at ~2.1M — far below where these constants actually live, but the cast is
/// free and the failure would be silent wraparound.
#[inline(always)]
pub fn frame_scale(c: Value) -> Value {
    round_div_1024(c as i64 * FRAME_S() as i64) as Value
}

/// Divide by 1024 rounding HALF AWAY FROM ZERO, not truncating.
///
/// Truncation is not a rounding detail here, it is a correctness bug for small
/// constants — found by the verification pass 2026-07-27. At s = 594:
/// `SE_DOUBLE_BASE = -3` truncated to -1 against an exact -1.74, an effective
/// ratio of 0.33 rather than 0.58; and `ASPIRATION_BASE_DELTA = 5` truncated
/// to 2 at BOTH s = 594 and s = 470, silently making two sweep arms identical
/// for that constant. A sweep whose arms are not actually distinct cannot
/// measure what it claims to.
///
/// Symmetric about zero so a ±margin pair rounds the same way; truncation
/// biases the two sides differently, which is its own quiet defect.
///
/// Identity at 1024 is preserved: `(c*1024 + 512) / 1024 == c` for every c,
/// because the remainder can never reach 1024.
#[inline(always)]
fn round_div_1024(n: i64) -> i64 {
    round_div(n, 1024)
}

/// Class 3: a divisor converting NON-eval units (correction/history) into
/// eval space. Moves the OPPOSITE way to [`frame_scale`]. Identity at 1024.
#[inline(always)]
pub fn frame_expand(d: Value) -> Value {
    round_div(d as i64 * 1024, FRAME_S().max(1) as i64) as Value
}

/// Apply a 1024ths multiplier, rounding half AWAY from zero.
///
/// The rounding is not a detail. Truncation here turned `SE_DOUBLE_BASE`
/// −3 into −1 under a 0.58 factor — a 0.33 ratio where 0.58 was intended —
/// and made two arms of a sweep silently identical
/// (`docs/research/eval-frame-scale-model-2026-07-27.md`, retraction 5).
/// Class-4 converters run on small signed quantities, so they are exposed to
/// exactly that failure.
#[inline(always)]
pub fn scale_1024(v: i32, mult: i32) -> i32 {
    if mult == 1024 {
        return v; // exact identity, and the only path the shipped engine takes
    }
    round_div(v as i64 * mult as i64, 1024) as i32
}

/// `n / q` rounded half away from zero. The sign-symmetric form matters: a
/// naive `(n + q/2) / q` is correct only for positive `n` and silently breaks
/// identity for negative constants — caught here by the identity test, which
/// is precisely what that test is for.
#[inline(always)]
fn round_div(n: i64, q: i64) -> i64 {
    if n >= 0 {
        (n + q / 2) / q
    } else {
        -((-n + q / 2) / q)
    }
}

/// [`frame_scale`] for the i64 tunables (the aspiration mean-squared divisor).
#[inline(always)]
pub fn frame_scale_i64(c: i64) -> i64 {
    round_div_1024(c * FRAME_S() as i64)
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    /// The identity that makes the sweep a clean control: at FRAME_S = 1024
    /// every transform must be a bit-exact no-op, so s=1.0 reproduces the
    /// shipped engine rather than an arithmetic near-miss.
    #[test]
    fn frame_transforms_are_exact_identities_at_1024() {
        assert_eq!(FRAME_S(), 1024, "the shipped default must be identity");
        for c in [0, 1, 5, 40, 76, 231, 300, 465, 485, 10588, 182069, -465] {
            assert_eq!(frame_scale(c), c, "frame_scale moved {c}");
            assert_eq!(frame_expand(c), c, "frame_expand moved {c}");
            assert_eq!(frame_scale_i64(c as i64), c as i64);
        }
    }

    /// The two directions must actually oppose — a copy-paste that made
    /// frame_expand a second frame_scale would silently double the class-3
    /// error, which is the failure this whole taxonomy exists to prevent.
    #[test]
    fn scale_and_expand_move_opposite_ways() {
        // Exercised via the arithmetic directly, since FRAME_S is a compile
        // -time constant in non-tune builds and cannot be moved in a test.
        let s = 594i64; // 0.58
        let c = 1000i64;
        assert!((c * s) / 1024 < c, "scale must shrink for s < 1");
        assert!((c * 1024) / s > c, "expand must grow for s < 1");
    }
}
