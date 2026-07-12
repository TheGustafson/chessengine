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
pub const RFP_MAX_DEPTH: Depth = 17;
/// Base of the per-ply futility multiplier `min(RFP_MULT_BASE +
/// RFP_MULT_SLOPE·depth, RFP_MULT_CAP)`.
pub const RFP_MULT_BASE: Value = 40;
/// Per-ply slope of the futility multiplier.
pub const RFP_MULT_SLOPE: Value = 4;
/// Cap of the futility multiplier.
pub const RFP_MULT_CAP: Value = 80;
/// Subtracted from the futility multiplier on a TT miss (a node with no TT
/// data prunes a little more cautiously).
pub const RFP_MULT_NO_TTHIT: Value = 20;
/// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which an *improving*
/// node shrinks the margin — the fail-high is better supported.
pub const RFP_IMPROVING: Value = 2934;
/// Weight (÷[`RFP_TREND_DEN`] of `futility_mult`) by which a *worsening*
/// opponent shrinks the margin.
pub const RFP_OPP_WORSENING: Value = 343;
/// Denominator of the improving/opponent-worsening margin discounts.
pub const RFP_TREND_DEN: Value = 1024;
/// Beta weight of the return blend `(RFP_RET_BETA_W·beta + RFP_RET_EVAL_W·
/// eval) / RFP_RET_DEN` — a value ≥ beta, pulled toward beta to damp
/// static-eval noise (the same instability-damping idiom as the node-exit
/// fail-high soften).
pub const RFP_RET_BETA_W: Value = 716;
/// Eval weight of the return blend.
pub const RFP_RET_EVAL_W: Value = 308;
/// Denominator of the return blend (`RFP_RET_BETA_W + RFP_RET_EVAL_W`).
pub const RFP_RET_DEN: Value = 1024;

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
pub const NMP_EVAL_DEPTH: Value = 14;
/// Improving discount of the NMP static-eval bar.
pub const NMP_EVAL_IMPROVING: Value = 45;
/// Flat base of the NMP static-eval bar (dominates at shallow depth, so the
/// bar sits well above beta there).
pub const NMP_EVAL_BASE: Value = 374;
/// Base of the null-move dynamic reduction `R = NMP_R_BASE + depth /
/// NMP_R_DEPTH_DIV` (in whole plies — this reduction is applied directly to
/// depth, not in the 1024ths the LMR table uses).
pub const NMP_R_BASE: Depth = 7;
/// Depth divisor of the null-move reduction.
pub const NMP_R_DEPTH_DIV: Depth = 3;
/// Depth at/above which a fail-high null search is re-confirmed by the
/// verification search (`depth >= NMP_VERIFY_DEPTH`); below it the null
/// value is returned directly.
pub const NMP_VERIFY_DEPTH: Depth = 16;
/// The verification's NMP-disable window `nmp_min_ply = ply +
/// NMP_VERIFY_FRAC.0·(depth - R) / NMP_VERIFY_FRAC.1`: NMP stays off for the
/// whole subtree up to that ply, so the verification (and its children)
/// cannot recursively null-move their way back to the same false cutoff.
pub const NMP_VERIFY_FRAC: (Depth, Depth) = (3, 4);

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
pub const RED_TABLE_NUM: i32 = 2834;
/// Denominator of the log reduction table (a power of two — exactly
/// representable, so the built table is bit-reproducible).
pub const RED_TABLE_DEN: i32 = 128;
/// Wide-window damping of the base reduction: a wider local window `delta`
/// reduces less, via `- delta * RED_DELTA_MULT / root_delta`.
pub const RED_DELTA_MULT: i32 = 617;
/// Not-improving surcharge numerator: a not-improving node reduces more, by
/// `+ scale * RED_NON_IMPROVING / RED_NON_IMPROVING_DEN` (~38% extra).
pub const RED_NON_IMPROVING: i32 = 194;
/// Not-improving surcharge denominator (the `/512` of the term above).
pub const RED_NON_IMPROVING_DEN: i32 = 512;
/// Flat base added to every base reduction (in 1024ths of a ply).
pub const RED_BASE: i32 = 1027;

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
pub const RED_UNIT: i32 = 1024;
/// Flat rebase added to every finalized reduction (§3.7).
pub const RED_FLAT: i32 = 714;
/// Per-move-count reduction discount `r -= RED_MOVECOUNT * move_count`: later
/// moves in the list reduce a little less than the raw log table implies
/// (they were tried later precisely because ordering distrusted them, §3.7).
pub const RED_MOVECOUNT: i32 = 62;
/// Cut-node reduction surcharge: an expected-fail-high node reduces its late
/// moves harder — it wants a cheap refutation, not a thorough search (§3.7).
pub const RED_CUTNODE: i32 = 3995;
/// Extra cut-node surcharge when the node has no TT move: nothing trustworthy
/// to search near full depth, so reduce even more (§3.7).
pub const RED_CUTNODE_NO_TTMOVE: i32 = 1059;
/// TT-move reduction rebate `r = max(0, r - RED_TTMOVE_SUB)`: the move the TT
/// records as best is searched close to full depth (§3.7). In Wave A this is
/// a plain `if m == tt.mv`; the cutoffCnt `if/else` wrapper is Wave D.
pub const RED_TTMOVE_SUB: i32 = 2016;
/// Cap on how many plies a NEGATIVE net `r` may DEEPEN the reduced scout past
/// `new_depth` (`min(new_depth - r/RED_UNIT, new_depth + RED_MAX_DEEPER)`,
/// §3.8); PV nodes get a further unconditional +1.
pub const RED_MAX_DEEPER: Depth = 2;
/// Deeper-re-search margin: after a reduced scout beats alpha, re-search one
/// ply DEEPER when it beat the running best by more than this (§3.8).
pub const LMR_DEEPER_MARGIN: Value = 52;
/// Shallower-re-search margin: re-search one ply SHALLOWER when the scout only
/// barely cleared the running best (`value < best_value + this`, §3.8).
pub const LMR_SHALLOWER_MARGIN: Value = 9;
/// Non-LMR path: a node with no TT move adds this to `r` before the depth
/// docks below — the full-depth analogue of the cut-node no-ttMove surcharge
/// (§3.8).
pub const RED_FULLDEPTH_NO_TTMOVE: i32 = 1085;
/// Non-LMR path first depth dock: an `r` above this costs one ply even for a
/// move that does not qualify for LMR proper (§3.8).
pub const RED_FULLDEPTH_DOCK1: i32 = 5039;
/// Non-LMR path second depth dock: an `r` above this (and `new_depth > 2`)
/// costs a second ply (§3.8).
pub const RED_FULLDEPTH_DOCK2: i32 = 5223;

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
pub const STATSCORE_MAIN_WEIGHT: i32 = 2;
/// Victim-value multiplier of the capture `stat_score`
/// (`STATSCORE_CAPT_NUM·piece_value(captured) / STATSCORE_CAPT_DEN`).
pub const STATSCORE_CAPT_NUM: i32 = 809;
/// Divisor of the capture `stat_score` victim term.
pub const STATSCORE_CAPT_DEN: i32 = 128;
/// Numerator of the reduction dock `r -= stat_score·RED_STATSCORE_NUM /
/// RED_STATSCORE_DEN`: positive `stat_score` LOWERS the reduction (search
/// fuller), negative RAISES it.
pub const RED_STATSCORE_NUM: i32 = 445;
/// Divisor of the reduction dock (a power of two — the truncation toward zero
/// on a negative `stat_score` is intended, not to be "fixed" to floor).
pub const RED_STATSCORE_DEN: i32 = 4096;
/// Flat continuation-history bonus a move earns when its reduced LMR scout
/// beats alpha (design/10 §3.8): applied along the offset-1..6 chain.
pub const LMR_CONTHIST_BONUS: i32 = 1415;

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

/// Base term of the LMP move-count threshold `(LMP_BASE + depth²) /
/// (2 - improving)` (design/10 §3.5): the constant floor of quiets always
/// tried before movecount pruning can bite, before the depth² growth.
pub const LMP_BASE: i32 = 3;

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
pub const CFUT_MAX_LMRDEPTH: Depth = 7;
/// Flat base of the capture-futility optimistic value over the static eval.
pub const CFUT_BASE: Value = 231;
/// Per-lmr-depth slope of the capture-futility value (can be negative at
/// shallow nodes, LOWERING the bound).
pub const CFUT_LMRDEPTH_MULT: Value = 232;
/// Capture-history nudge numerator of the capture-futility value
/// (`CFUT_HIST_MULT·capt_hist / CAPT_HIST_DEN`): a capture the history likes
/// is written off a little less readily.
pub const CFUT_HIST_MULT: i32 = 131;
/// Depth multiplier of the capture/check SEE margin
/// (`CAP_SEE_DEPTH_MULT·depth + CAP_SEE_HIST_MULT·capt_hist/CAPT_HIST_DEN`):
/// deeper nodes tolerate a bigger static-exchange loss before pruning.
pub const CAP_SEE_DEPTH_MULT: Value = 175;
/// Capture-history numerator of the capture/check SEE margin.
pub const CAP_SEE_HIST_MULT: i32 = 34;
/// Shared `/1024` divisor of the capture-history reads in both capture gates
/// above (the capture history itself is carried at design/09's `CAPT_D`
/// gravity scale; this is only the pruning read weight).
pub const CAPT_HIST_DEN: i32 = 1024;

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
pub const QHIST_PRUNE_MULT: i32 = 4313;

// --- Move-loop pruning: parent futility (design/10 §3.5) ---
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
pub const PFUT_BASE: Value = 40;
/// Bound surcharge while no best move has been found yet: a node with no
/// established alpha-raiser writes moves off a little more optimistically.
pub const PFUT_NO_BESTMOVE: Value = 138;
/// Per-lmr-depth slope of the bound: a deeper predicted search must clear
/// more headroom before its quiets are futile.
pub const PFUT_LMRDEPTH_MULT: Value = 117;
/// Bound surcharge when the static eval already clears alpha.
pub const PFUT_EVAL_ABOVE_ALPHA: Value = 90;
/// Exclusive lmr-depth ceiling: parent futility runs only while
/// `lmr_depth < PFUT_MAX_LMRDEPTH` (deeper predicted searches are trusted).
pub const PFUT_MAX_LMRDEPTH: Depth = 12;

// --- Move-loop pruning: quiet SEE (design/10 §3.5) ---
//
// After parent futility, a quiet whose static-exchange value cannot reach
// `-QSEE_MULT·lmr_depth²` (quadratic in the predicted post-LMR depth, floored
// at 0) is pruned: a material-losing quiet at shallow effective depth is
// almost never worth a search. At `lmr_depth == 0` the bar is 0, so any
// negative-SEE quiet is dropped; it slackens quadratically with depth.

/// Quadratic coefficient of the quiet SEE margin `-QSEE_MULT·lmr_depth²`
/// (`lmr_depth` clamped to `>= 0` first).
pub const QSEE_MULT: Value = 25;

// --- Transposition-table interaction (design/07 §3.7.2 steps 7/14) ---

/// TT cutoffs are refused once the halfmove clock reaches this (the
/// graph-history guard of design/07 §3.7.2 step 7 / design/08 §3.12).
/// `i32`: matches `value_from_tt`'s rule50 parameter (doc 02).
pub const TT_CUTOFF_RULE50_LIMIT: i32 = 96;
/// Extra depth credited to the TT store of a no-legal-moves node
/// (design/07 §3.7.2 step 14): a mate/stalemate terminal is exact far
/// beyond its nominal depth, so make the entry hard to displace.
pub const TT_NO_MOVES_DEPTH_BONUS: Depth = 6;

// --- TT-cutoff history nudges (N6b extras, design/10 §3.4; Wave-C item 12) ---
//
// When the non-PV TT cutoff fires with `tt.value >= beta`, the stored ttMove
// "worked": if it is quiet, reward it (`min(TTCUT_BONUS_MULT·depth,
// TTCUT_BONUS_CAP)`), and if the parent played an early (`move_count < 4`)
// non-capturing quiet, penalize that parent move along its continuation chain
// (`TTCUT_PARENT_MALUS`) — a cheap cutoff here is evidence the parent's move
// was refutable.

/// Depth multiplier of the TT-cutoff quiet-ttMove bonus.
pub const TTCUT_BONUS_MULT: i32 = 114;
/// Cap of the TT-cutoff quiet-ttMove bonus.
pub const TTCUT_BONUS_CAP: i32 = 724;
/// Continuation-history malus applied to the parent's early quiet move on a
/// beta-side TT cutoff.
pub const TTCUT_PARENT_MALUS: i32 = -2187;
/// The parent's `move_count` must be below this for the TT-cutoff parent
/// penalty to fire (an "early" quiet the cutoff just refuted).
pub const TTCUT_PARENT_MOVECOUNT: i32 = 4;

// --- Eval-diff parent nudge (N8, design/10 §3.4; Wave-C item 12) ---
//
// Grade the opponent's last quiet by the static-eval swing it produced.
// `eval_diff = clamp(-(ss-1).static_eval - ss.static_eval, EVALDIFF_CLAMP_LO,
// EVALDIFF_CLAMP_HI) + EVALDIFF_OFFSET`; the parent's move (indexed by the
// *opponent* color) takes `main_history << EVALDIFF_MAIN_MULT·eval_diff`, and —
// on a TT miss for a non-pawn, non-promotion parent move — its pawn-history
// slot takes `<< EVALDIFF_PAWN_MULT·eval_diff`.

/// Low clamp of the eval swing feeding the eval-diff nudge.
pub const EVALDIFF_CLAMP_LO: Value = -183;
/// High clamp of the eval swing.
pub const EVALDIFF_CLAMP_HI: Value = 180;
/// Flat offset added to the clamped eval swing.
pub const EVALDIFF_OFFSET: Value = 62;
/// Main-history multiplier of the eval-diff nudge.
pub const EVALDIFF_MAIN_MULT: i32 = 10;
/// Pawn-history multiplier of the eval-diff nudge (TT-miss, non-pawn mover).
pub const EVALDIFF_PAWN_MULT: i32 = 13;

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

// --- History updates: node-result bonus/malus (design/09 §3.9) ---
//
// `update_all_stats` computes one `bonus` (for the best move and agreeing
// tables) and one `malus` (for tried non-best moves) per fail-high node,
// then distributes them through the per-table weights below. The shared
// `/1024` denominator is [`STAT_WEIGHT_DEN`].

/// Depth multiplier of the stat bonus `min(STAT_BONUS_DEPTH_MULT·depth −
/// STAT_BONUS_DEPTH_SUB, STAT_BONUS_MAX)`.
pub const STAT_BONUS_DEPTH_MULT: i32 = 134;
/// Depth-linear subtrahend of the stat bonus.
pub const STAT_BONUS_DEPTH_SUB: i32 = 79;
/// Ceiling of the stat bonus.
pub const STAT_BONUS_MAX: i32 = 1572;
/// Extra bonus when the best move equals the TT move (it was predicted).
pub const STAT_BONUS_TTMOVE: i32 = 382;
/// Divisor of the parent `stat_score` contribution to the bonus.
pub const STAT_BONUS_STATSCORE_DIV: i32 = 30;
/// Depth multiplier of the stat malus `min(STAT_MALUS_DEPTH_MULT·depth −
/// STAT_MALUS_DEPTH_SUB, STAT_MALUS_MAX)`.
pub const STAT_MALUS_DEPTH_MULT: i32 = 1005;
/// Depth-linear subtrahend of the stat malus.
pub const STAT_MALUS_DEPTH_SUB: i32 = 205;
/// Ceiling of the stat malus.
pub const STAT_MALUS_MAX: i32 = 2218;
/// Divisor of the non-PV breadth multiplier `bonus += bonus·tried /
/// STAT_BREADTH_DIVISOR` (a wide node that tried many moves taught us more).
/// `i64`: the product is widened to avoid the 32-bit overflow SF guards
/// against (design/09 §9 open-question 3 — ours is plain i64 truncation).
pub const STAT_BREADTH_DIVISOR: i64 = 256;
/// Shared `/1024` denominator of every history-update weight below.
pub const STAT_WEIGHT_DEN: i32 = 1024;
/// Best-quiet bonus weight (`bonus·824/1024`).
pub const QUIET_BEST_BONUS_NUM: i32 = 824;
/// Geometric quiet-malus start weight (`malus·1136/1024`), decayed by
/// [`QUIET_MALUS_DECAY_NUM`] before each successive non-best quiet — earliest
/// failures punished hardest.
pub const QUIET_MALUS_START_NUM: i32 = 1136;
/// Per-move decay of the geometric quiet malus (`m·956/1024` each step).
pub const QUIET_MALUS_DECAY_NUM: i32 = 956;
/// Best-capture bonus weight (`bonus·1366/1024`).
pub const CAPTURE_BEST_BONUS_NUM: i32 = 1366;
/// Non-best-capture malus weight (`−malus·1518/1024`).
pub const CAPTURE_MALUS_NUM: i32 = 1518;
/// Refutation malus weight for the parent's refuted first-try quiet
/// (`−malus·683/1024`).
pub const REFUTATION_MALUS_NUM: i32 = 683;

// --- History updates: per-quiet-table weights (design/09 §3.9) ---

/// Low-ply history update weight (`bonus·663/1024`, plies < 5 only).
pub const QUIET_LOW_PLY_UPDATE_NUM: i32 = 663;
/// Continuation-history update weight (`bonus·820/1024`).
pub const QUIET_CONT_UPDATE_NUM: i32 = 820;
/// Pawn-history update weight for bonuses (`bonus·1038/1024`).
pub const PAWN_HIST_BONUS_NUM: i32 = 1038;
/// Pawn-history update weight for maluses (`bonus·525/1024`) — maluses are
/// roughly halved relative to bonuses.
pub const PAWN_HIST_MALUS_NUM: i32 = 525;
/// The pawn-history bonus/malus split point: `> -7` takes the bonus weight,
/// `<= -7` the malus weight (deliberately not `> 0`).
pub const PAWN_HIST_ASYMMETRY_THRESHOLD: i32 = -7;

// --- History updates: continuation chain (design/09 §3.9) ---

/// Per-offset weights of the continuation-history update, offsets 1..=6
/// (÷[`CONT_UPDATE_DIVISOR`] with the bonus).
pub const CONT_OFFSET_WEIGHTS: [i32; 6] = [1040, 780, 300, 537, 129, 423];
/// Consistency multipliers indexed by the running count of already-positive
/// entries along the chain: an agreeing chain amplifies the update.
pub const CONT_CONSISTENCY_MULT: [i32; 7] = [96, 113, 101, 105, 127, 121, 126];
/// Divisor of the continuation-history update `bonus·weight·mult / 131072`.
/// `i64`: the triple product is widened before the divide.
pub const CONT_UPDATE_DIVISOR: i64 = 131072;
/// Flat bonus added to the offset-1 continuation update (even for maluses).
pub const CONT_OFFSET1_FLAT_BONUS: i32 = 71;
/// In check, only offsets `<= 2` update (the position before the check bears
/// little relation to the current one).
pub const CONT_IN_CHECK_MAX_OFFSET: usize = 2;

// --- Correction history: read combiner (design/09 §3.10) ---
//
// `correction_value` blends the learned static-eval corrections in
// 1/131072-pawn units; [`CORR_VALUE_DIVISOR`] converts back to centipawns.

/// Weight on the pawn-structure correction.
pub const CORR_PAWN_WEIGHT: i32 = 13345;
/// Weight on the minor-piece correction.
pub const CORR_MINOR_WEIGHT: i32 = 9280;
/// Weight on each non-pawn (white/black) correction.
pub const CORR_NONPAWN_WEIGHT: i32 = 11840;
/// Weight on the summed continuation corrections (offsets 2 and 4).
pub const CORR_CONT_WEIGHT: i32 = 8363;
/// Flat continuation-correction prior used after a null previous move.
pub const CORR_CONT_NULL_PRIOR: i32 = 64549;
/// Divisor turning the correction sum back into a centipawn adjustment.
pub const CORR_VALUE_DIVISOR: i32 = 131072;

// --- Correction history: update weights (design/09 §3.10) ---

/// Shared `/128` denominator of the correction-update weights below.
pub const CORR_UPDATE_DEN: i32 = 128;
/// Minor-piece correction update weight (`bonus·152/128`).
pub const CORR_MINOR_UPDATE_NUM: i32 = 152;
/// Non-pawn (white and black) correction update weight (`bonus·186/128`).
pub const CORR_NONPAWN_UPDATE_NUM: i32 = 186;
/// Offset-2 continuation-correction update weight (`bonus·136/128`).
pub const CORR_CONT2_UPDATE_NUM: i32 = 136;
/// Offset-4 continuation-correction update weight (`bonus·68/128`).
pub const CORR_CONT4_UPDATE_NUM: i32 = 68;

// --- Correction history: bonus computation (call site, design/10 §4.6) ---
//
// The gated, clamped bonus fed to `update_correction_history`. Owned by
// design/10's node-result step; colocated with the callee here (design/09
// §3.9 records external write-site formulas for grep-ability) so the P8
// integration wave can compute it in one call.

/// Depth multiplier when a best move exists (`err·depth·12/128`).
pub const CORR_BONUS_DEPTH_MULT_BESTMOVE: i32 = 12;
/// Depth multiplier on a fail-low with no best move (`err·depth·18/128`).
pub const CORR_BONUS_DEPTH_MULT_NOMOVE: i32 = 18;
/// Divisor of the raw correction bonus.
pub const CORR_BONUS_DIVISOR: i32 = 128;
/// Symmetric clamp on the raw correction bonus (`CORRECTION_HISTORY_LIMIT/4`).
pub const CORR_BONUS_CLAMP: i32 = 256;
/// Numerator of the outer correction-bonus scaling (`·1114/1024`).
pub const CORR_BONUS_SCALE_NUM: i32 = 1114;
/// Denominator of the outer correction-bonus scaling.
pub const CORR_BONUS_SCALE_DEN: i32 = 1024;
