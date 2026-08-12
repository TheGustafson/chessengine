//! Search — the iterative-deepening driver, PVS nodes, and quiescence.
//!
//! Contracts: `design/07-search-core.md` (structure) and
//! `design/10-pruning-extensions.md` (the heuristics).
//!
//! Reading order: [`Worker::iterative_deepening`] (the ID loop),
//! [`Worker::search`] (one PVS node in the Stockfish step order),
//! [`Worker::qsearch`], then [`params`].
//!
//! # Invariants worth knowing before editing
//!
//! * **`Bench:` (D24).** Any change that can affect the tree must state the new
//!   node count; an intended non-functional change must reproduce the old one
//!   bit-for-bit. `bash tools/check.sh` prints the line to paste.
//! * **The bench pin is BLIND to NNUE work** (default-feature build, no `sota`
//!   path); `src/eval/nnue/` needs `CHECK_NNUE_GATE=1 bash tools/check.sh`.
//! * **Scores are OUR internal frame**, not Stockfish printed cp. Never
//!   hand-compare; use `searchopt/tools/uciprobe.py`'s frame helpers.
//! * **Tunables are compile-time constants** unless `--features tune`; without
//!   it every "arm" you set is silently ignored.

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

/// The draw dither (design/07 §3.1): draw-ish MAIN-search returns are never
/// exactly [`VALUE_DRAW`] but carry a ±1 pseudo-random component from the node
/// counter, so repetition lines are not all mutually equal.
///
/// Qsearch's ordinary draw/max-ply return uses plain `VALUE_DRAW` (SF parity);
/// only its upcoming-repetition path dithers.
#[inline]
pub fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// [`value_draw`] with CONTEMPT, from the ROOT side's point of view: a draw
/// scored below equality cannot set alpha as cheaply, so a winning sibling
/// keeps a usable window. Negamax flips perspective per ply, so the sign
/// alternates with ply parity. Default 0 ⇒ inert.
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

// The DIS-* env knobs below plug the disagreement head into RFP / eval / LMR /
// TM. Each is read ONCE through a `OnceLock` (never `env::var` per node); 0.0
// or unset means off and leaves the shipped build bit-identical. A negative
// coefficient is a falsification arm. NOTE: any TM arm must be tested at REAL
// time control — nodestime has produced the opposite sign here.

/// DIS-TM falsification control (`ARCHON_DIS_TM_RANDOM=1`): matched
/// magnitude, zero information.
#[cfg(feature = "sota")]
#[inline]
fn dis_tm_random() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARCHON_DIS_TM_RANDOM").is_some())
}

/// Diagnostic: disable RFP pruning so census nodes are actually searched and
/// produce a ground-truth value. `ARCHON_RFP_OFF=1`. Never set in a match.
#[inline]
fn rfp_off() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARCHON_RFP_OFF").is_some())
}

#[cfg(feature = "sota")]
#[inline]
fn dis_census_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARCHON_DIS_CENSUS").is_some())
}

/// DIS-CENSUS binary feature dump (`ARCHON_DIS_DUMP=<path>`, append mode).
/// One fixed 792-byte record per census row: id u64 | depth i32 | eval i16 |
/// beta i16 | margin i16 | flags u8 (imp|ow<<1|tt<<2) | pad | corr i32 |
/// ft [u8; 768]. Unbuffered on purpose, so records survive any exit path.
#[cfg(feature = "sota")]
#[allow(clippy::too_many_arguments)]
fn dis_dump_write(
    id: u64,
    depth: Depth,
    eval: Value,
    beta: Value,
    margin: Value,
    imp: bool,
    ow: bool,
    tt: bool,
    corr: i32,
    ft: &[u8],
) {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static F: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let slot = F.get_or_init(|| {
        std::env::var_os("ARCHON_DIS_DUMP").map(|path| {
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("ARCHON_DIS_DUMP path"),
            )
        })
    });
    let Some(m) = slot else { return };
    let mut rec = [0u8; 792];
    rec[0..8].copy_from_slice(&id.to_le_bytes());
    rec[8..12].copy_from_slice(&depth.to_le_bytes());
    rec[12..14].copy_from_slice(&(eval.clamp(-32000, 32000) as i16).to_le_bytes());
    rec[14..16].copy_from_slice(&(beta.clamp(-32000, 32000) as i16).to_le_bytes());
    rec[16..18].copy_from_slice(&(margin.clamp(-32000, 32000) as i16).to_le_bytes());
    rec[18] = u8::from(imp) | (u8::from(ow) << 1) | (u8::from(tt) << 2);
    rec[20..24].copy_from_slice(&corr.to_le_bytes());
    let n = ft.len().min(768);
    rec[24..24 + n].copy_from_slice(&ft[..n]);
    let mut f = m.lock().unwrap();
    let _ = f.write_all(&rec);
}

#[cfg(feature = "sota")]
#[inline]
fn dis_tm_c() -> f32 {
    use std::sync::OnceLock;
    static C: OnceLock<f32> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("ARCHON_DIS_TM_C")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

#[cfg(feature = "sota")]
#[inline]
fn dis_lmr_c() -> f32 {
    use std::sync::OnceLock;
    static C: OnceLock<f32> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("ARCHON_DIS_LMR_C")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

#[cfg(feature = "sota")]
#[inline]
fn dis_eval_c() -> f32 {
    use std::sync::OnceLock;
    static C: OnceLock<f32> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("ARCHON_DIS_EVAL_C")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

#[cfg(feature = "sota")]
#[inline]
fn dis_rfp_c1() -> f32 {
    use std::sync::OnceLock;
    static C1: OnceLock<f32> = OnceLock::new();
    *C1.get_or_init(|| {
        std::env::var("ARCHON_DIS_RFP_C1")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

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

/// Feature-gated qsearch decision trace (`--features search-control`, enabled
/// at runtime by `ARCHON_QTRACE=1`); compiled out of every shipped build.
///
/// NOTE: takes the `Move` BY VALUE. With a `&str` parameter the call sites
/// formatted the move before the (empty) call, so the disabled arm paid the
/// whole Display + allocation chain. Keep all formatting inside the arm.
#[cfg(feature = "search-control")]
#[inline]
fn qtrace(ply: usize, alpha: Value, beta: Value, stand: Value, mv: Move, verdict: &str) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("ARCHON_QTRACE").is_some()) {
        // `Move::NULL` carries what the old `"-"` literal did: "no move at this
        // decision point".
        let mv = if mv == Move::NULL {
            "-".to_string()
        } else {
            mv.to_string()
        };
        eprintln!("QT ply={ply} a={alpha} b={beta} sp={stand} {mv:<6} {verdict}");
    }
}

#[cfg(not(feature = "search-control"))]
#[inline(always)]
fn qtrace(_: usize, _: Value, _: Value, _: Value, _: Move, _: &str) {}

/// The late-move-pruning move-count threshold (design/10 §3.5): once
/// `move_count` reaches it the picker stops emitting quiets. Quadratic in
/// depth; a non-improving node prunes its late quiets roughly twice as early.
#[inline]
fn lmp_move_count(depth: Depth, improving: bool) -> i32 {
    (params::LMP_BASE() + depth * depth) / (params::LMP_IMPROVING_DEN() - i32::from(improving))
}

/// The parent-futility bound (design/10 §3.5, the move loop's quiet branch):
/// the most optimistic value a quiet could deliver; a shallow non-check node
/// writes the move off when this cannot reach alpha. `lmr_depth` is the
/// predicted post-LMR depth and can be negative. Pure `i32`; term order and
/// truncation match the Stockfish source (D27).
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

/// The capture-futility optimistic value (design/10 §3.5): the most a
/// non-checking capture could yield — static eval, depth-scaled slack, victim
/// value, capture-history nudge — pruned when it cannot reach alpha.
/// `captured_value` is 0 for en passant (SF parity); the `lmr_depth` term can
/// be negative. The history term is formed before the divide (D27).
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
        + captured_value
        + params::CFUT_HIST_MULT() * capt_hist / params::CAPT_HIST_DEN()
}

/// The capture/check SEE margin (design/10 §3.5): a capture or checking move
/// whose static exchange cannot reach `-margin` is pruned. The history term is
/// formed before the divide (SF truncation, D27).
#[inline]
fn capture_see_margin(depth: Depth, capt_hist: i32) -> Value {
    params::CAP_SEE_DEPTH_MULT() * depth
        + params::CAP_SEE_HIST_MULT() * capt_hist / params::CAPT_HIST_DEN()
}

/// The singular-verification window centre (design/10 §3.6): the exclusion
/// search runs at `[singular_beta - 1, singular_beta]` with the ttMove
/// excluded. `ttpv_non_pv` is `ss.tt_pv && !PV`. The depth product is formed
/// before the divide (SF truncation, D27).
#[inline]
fn singular_verification_beta(tt_value: Value, depth: Depth, ttpv_non_pv: bool) -> Value {
    tt_value
        - (params::SE_BETA_MULT() + params::SE_BETA_TTPV_MULT() * Value::from(ttpv_non_pv)) * depth
            / params::SE_BETA_DIV()
}

/// The singular DOUBLE-extension margin (design/10 §3.6): an exclusion value
/// below `singular_beta - double_margin` earns the ttMove a second extension
/// ply. `corr_adj` tightens the margin under eval uncertainty; `tmh` is the
/// tt_move_history scalar; `deep` is `ply > root_depth`. The `tmh` term is
/// formed before its divide (D27).
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

/// The singular TRIPLE-extension margin (design/10 §3.6): a third extension
/// ply below `singular_beta - triple_margin`. Same `corr_adj`/`deep` as
/// [`singular_double_margin`]; `ttpv` is `ss.tt_pv`.
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

/// The continuation-history pruning predicate (design/10 §3.5): a quiet whose
/// `contHist(1) + contHist(2) + pawnHistory` falls below `-QHIST_PRUNE_MULT ·
/// depth` is written off without a search. LINEAR in depth, unlike the
/// quadratic quiet-SEE bar.
#[inline]
fn continuation_history_prunes(cont_history: i32, depth: Depth) -> bool {
    cont_history < -params::QHIST_PRUNE_MULT() * depth
}

/// The TT prefetch hook (doc 04 §3.10 hook A, doc 08 §3.15 call site 1):
/// wired into [`Worker::do_move`] so every made move prefetches the child's
/// cluster the moment the child key is final, overlapping the DRAM fetch with
/// the rest of `make_move`. Monomorphized to a single `prefetcht0`.
struct TtPrefetch<'a> {
    tt: &'a TranspositionTable,
}

impl MakeMoveHooks for TtPrefetch<'_> {
    #[inline]
    fn on_key(&mut self, key: Key) {
        self.tt.prefetch(key);
    }
}

/// [`TtPrefetch`] plus the dirty-piece record. `WANTS_DIRTY` is `true` only on
/// THIS type, so the ProbCut verification probe (plain [`TtPrefetch`]) pays
/// nothing and `NoHooks` monomorphizes the record away. The const cannot see
/// `nnue_active()`, which is why `do_move`'s `nnue_active()` guard cannot be
/// pushed down into the make.
struct TtPrefetchDirty<'a> {
    tt: &'a TranspositionTable,
    dp: DirtyPiece,
}

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

/// Compile-time node class (design/07 §3.7.1, D17): sealed marker types whose
/// associated consts monomorphize `search` into Root/Pv/NonPv instances, so
/// PV-only paths (PV-line writes, selDepth) vanish from NonPV.
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

/// One completed PV line's report (design/07 §3.5): what the UCI layer needs
/// to format an `info depth ... pv ...` line, borrowed from the worker's
/// root-move bookkeeping. The UCI layer adds `time`/`nps` and formats `score`.
#[derive(Clone, Copy)]
pub struct SearchInfo<'a> {
    /// Root iteration depth.
    pub depth: Depth,
    /// Max ply this line reached, counted from 1 (§3.9).
    pub sel_depth: i32,
    /// 1-based MultiPV line index (always 1 in v1, §3.5).
    pub multipv: usize,
    /// The line's `uci_score`, clamped to the aspiration window on a bound
    /// (§3.4). One info line per COMPLETED aspiration loop, whose front move is
    /// exact by construction, so the flags below stay false for now.
    pub score: Value,
    pub lowerbound: bool,
    pub upperbound: bool,
    /// Nodes searched so far this `go` (single worker until SMP, D23).
    pub nodes: u64,
    /// TT occupancy in permille, sampled once per completed line, never per node.
    pub hashfull: u32,
    /// The line itself; `pv[0]` is the root move.
    pub pv: &'a [Move],
}

/// The type-erased UCI output sink (design/13 §3.3/§3.7 G), held as
/// `Arc<dyn UciSink>` so the pool stays free of the output stream's `W: Write`
/// parameter; `uci.rs` implements it. Only the MAIN worker writes through it,
/// but every method must still be internally line-atomic.
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
/// Returns the winning report's index.
///
/// Each thread votes for its own best move with weight
/// `score - min_score + VOTE_TEMPO`, so identical moves accumulate. A *trusted
/// decisive* incumbent is displaced only by a trusted decisive candidate of
/// strictly larger magnitude. Against a NON-decisive incumbent any trusted
/// decisive candidate — win OR loss — wins outright. Only among non-decisive
/// candidates does the vote decide (more votes, then longer PV), and there a
/// losing or never-finished (`-VALUE_INFINITE`) candidate can never win.
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
/// scratch board and probe the TT; offer the stored move only if the position
/// is not an immediate draw and the move is legal (legal-list membership also
/// discharges the doc 08 §3.5 validation). Cold path, once per `go`.
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

/// One aspiration attempt's window snapshot (test-only, design/07 §8
/// criterion 7). `delta`/`alpha`/`beta`/`failed_high_cnt` are what the attempt
/// SEARCHED with; `value` is what the root search returned.
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

/// Length of the soft-stop `fallingEval` ring (design/12 §3.9): the scores of
/// the previous four completed iterations. A power of two so the cursor
/// advance is one mask; named once so size and mask cannot disagree.
const ITER_VALUE_RING: usize = 4;
const _: () = assert!(ITER_VALUE_RING.is_power_of_two());

/// Main-worker-only state (design/13 §3.3): the null-object payload behind the
/// main worker's `manager: Some(..)` (helpers hold `None`). Per-GAME time state
/// that must survive across `go`s, plus the ponder-hit verdict. Never touched
/// cross-thread — the UCI thread flips only `PoolShared.ponder`.
pub struct SearchManager {
    /// Per-game TM state, seeded into the ID loop each `go` and updated at
    /// search end; reset by `ucinewgame` ([`Worker::clear`]).
    pub persist: TimePersist,
    /// Set when the soft budget elapses *while pondering*, so a later
    /// `ponderhit` stops at once instead of burning the clock (design/12 §3.3).
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

/// Search-anatomy diagnostics — the `ARCHON_SEARCH_STATS` report's counters.
///
/// **Zero-sized unless `cfg(test)` or `--features search-stats` asks for it**,
/// with empty bump methods. A shipped diagnostic must be zero-SIZED when off,
/// not merely unread: as live fields these were per-node stores the compiler
/// could not fold.
#[cfg(any(test, feature = "search-stats"))]
pub struct SearchStats {
    /// Qsearch slice of `nodes` (bumped at the qsearch `do_move` site only).
    pub qnodes: u64,
    /// Main-search beta cutoffs; cutoffs on the FIRST move tried; total moves
    /// tried at cutoff nodes.
    pub fh: u64,
    pub fh_first: u64,
    pub fh_moves: u64,
    /// Main-search TT-cutoff returns.
    pub ttcut: u64,
    /// Net extension plies granted.
    pub ext: i64,
    /// Aspiration re-searches.
    pub research: u64,
    /// LMR-reduced moves, and the plies of reduction they were given.
    pub lmr_n: u64,
    pub lmr_plies: i64,
    /// Null-move-pruning attempts and cutoffs.
    pub nmp_try: u64,
    pub nmp_cut: u64,
    /// Per-gate fire counters, indexed by [`Gate`]; the disabled twin has no
    /// array and an empty `gate()`.
    pub gates: [u64; Gate::COUNT],
}

/// One countable search-gate event; the *type* is free in every build, only
/// the enabled [`SearchStats`] carries the array. Names mirror the node-step
/// labels in `search()`/`qsearch()`.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Gate {
    /// Main-search node entries (post depth<=0 boundary) — the denominator.
    MainEntry,
    /// Main-search node entries while in check.
    MainInCheck,
    /// N9 razoring returns (drop to qsearch).
    Razor,
    /// N10 RFP returns.
    Rfp,
    /// N11 NMP verification searches run.
    NmpVerify,
    /// N12 IIR depth reductions applied.
    Iir,
    /// N6 hindsight depth += 1.
    HindUp,
    /// N6 hindsight depth -= 1.
    HindDown,
    /// N13 ProbCut candidate moves actually searched (qsearch probe made).
    ProbcutMove,
    /// N13 ProbCut fail-high returns.
    ProbcutCut,
    /// N14 in-check small-ProbCut returns.
    CheckCut,
    /// N6b wrong-direction-bound TT penalize writes.
    TtPenalize,
    /// N6b quiet-ttMove history bonus fired on a TT cutoff.
    TtcutBonus,
    /// N8 eval-diff parent history nudge fired.
    EvalDiff,
    /// Move-loop LMP threshold reached (skip_quiet_moves called).
    LmpSkip,
    /// Move-loop capture-futility prunes.
    CfutPrune,
    /// Move-loop capture/check SEE prunes.
    CseePrune,
    /// Move-loop continuation-history prunes.
    ChistPrune,
    /// Move-loop parent-futility prunes.
    PfutPrune,
    /// Move-loop quiet-SEE prunes.
    QseePrune,
    /// Singular exclusion searches run.
    SeTry,
    /// Singular verdict: extend (value < singular_beta).
    SeSingular,
    /// Singular double extension granted.
    SeDouble,
    /// Singular triple extension granted.
    SeTriple,
    /// Singular verdict: multicut return.
    SeMulticut,
    /// Singular verdict: negative extension (either arm).
    SeNegExt,
    /// Step-12 equal-value dither promotions offered (inc == 1).
    Dither,
    /// Step-14 fail-low parent bonus, quiet-parent arm.
    FlQuiet,
    /// Step-14 fail-low parent bonus, capture-parent arm.
    FlCapture,
    /// Node-exit correction-history update fired.
    CorrUpd,
    /// Qsearch node entries (the qsearch denominator; distinct from
    /// `qnodes`, which counts made moves).
    QsEntry,
    /// Qsearch TT-cutoff returns.
    QsTtcut,
    /// Qsearch stand-pat cutoff returns.
    QsStandpat,
    /// Qsearch late-move prunes.
    QsLatemove,
    /// Qsearch futility prunes.
    QsFut,
    /// Qsearch SEE-vs-window prunes.
    QsSeeWindow,
    /// Qsearch hard-SEE-floor prunes.
    QsSeeFloor,
    /// Qsearch narrow stalemate scans (generate_legal paid).
    QsStaleScan,
}

impl Gate {
    /// Number of gates — the census array length.
    pub const COUNT: usize = Gate::QsStaleScan as usize + 1;
    /// Display names, index-aligned with the enum.
    pub const NAMES: [&'static str; Gate::COUNT] = [
        "main_entry",
        "main_in_check",
        "razor",
        "rfp",
        "nmp_verify",
        "iir",
        "hind_up",
        "hind_down",
        "probcut_move",
        "probcut_cut",
        "check_cut",
        "tt_penalize",
        "ttcut_bonus",
        "evaldiff",
        "lmp_skip",
        "cfut_prune",
        "csee_prune",
        "chist_prune",
        "pfut_prune",
        "qsee_prune",
        "se_try",
        "se_singular",
        "se_double",
        "se_triple",
        "se_multicut",
        "se_negext",
        "dither",
        "fl_quiet",
        "fl_capture",
        "corr_upd",
        "qs_entry",
        "qs_ttcut",
        "qs_standpat",
        "qs_latemove",
        "qs_fut",
        "qs_see_window",
        "qs_see_floor",
        "qs_stale_scan",
    ];
}

/// The shipped shape: no fields, therefore no stores. See the enabled twin.
#[cfg(not(any(test, feature = "search-stats")))]
pub struct SearchStats;

/// The zeroed value, spelled once per shape. **Not `Default`:** the disabled
/// shape is a unit struct, and `default()` on one is
/// `clippy::default_constructed_unit_structs`, which this crate denies.
#[cfg(any(test, feature = "search-stats"))]
impl SearchStats {
    pub const NEW: Self = Self {
        qnodes: 0,
        fh: 0,
        fh_first: 0,
        fh_moves: 0,
        ttcut: 0,
        ext: 0,
        research: 0,
        lmr_n: 0,
        lmr_plies: 0,
        nmp_try: 0,
        nmp_cut: 0,
        gates: [0; Gate::COUNT],
    };
}

#[cfg(not(any(test, feature = "search-stats")))]
impl SearchStats {
    pub const NEW: Self = Self;
}

impl SearchStats {
    /// One census gate event. Empty body when disabled.
    #[inline(always)]
    pub fn gate(&mut self, g: Gate) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.gates[g as usize] += 1;
        }
        #[cfg(not(any(test, feature = "search-stats")))]
        let _ = g;
    }

    /// One qsearch node.
    #[inline(always)]
    pub fn qnode(&mut self) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.qnodes += 1;
        }
    }

    /// A beta cutoff after `move_count` moves.
    #[inline(always)]
    pub fn fail_high(&mut self, move_count: i32) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.fh += 1;
            self.fh_first += u64::from(move_count == 1);
            self.fh_moves += move_count as u64;
        }
        #[cfg(not(any(test, feature = "search-stats")))]
        let _ = move_count;
    }

    /// A TT cutoff return from the main search.
    #[inline(always)]
    pub fn tt_cut(&mut self) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.ttcut += 1;
        }
    }

    /// `plies` of extension granted to one move (may be negative).
    #[inline(always)]
    pub fn extension(&mut self, plies: i32) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.ext += i64::from(plies);
        }
        #[cfg(not(any(test, feature = "search-stats")))]
        let _ = plies;
    }

    /// One aspiration re-search.
    #[inline(always)]
    pub fn research(&mut self) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.research += 1;
        }
    }

    /// One LMR-reduced move, reduced by `plies`.
    #[inline(always)]
    pub fn lmr(&mut self, plies: i32) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.lmr_n += 1;
            self.lmr_plies += i64::from(plies);
        }
        #[cfg(not(any(test, feature = "search-stats")))]
        let _ = plies;
    }

    /// One null-move-pruning attempt.
    #[inline(always)]
    pub fn nmp_attempt(&mut self) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.nmp_try += 1;
        }
    }

    /// One null-move-pruning cutoff.
    #[inline(always)]
    pub fn nmp_cutoff(&mut self) {
        #[cfg(any(test, feature = "search-stats"))]
        {
            self.nmp_cut += 1;
        }
    }
}

/// Per-thread search state (design/07 §3.10, design/13 §3.2): one worker owns
/// its board copy, stack, PV table, histories and reduction table, none
/// observable from another thread. The only cross-thread-visible handles are
/// the atomic [`Counters`] and the `Arc<PoolShared>`.
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
    /// Made-move counter (§3.9): bumped in [`Worker::do_move`], matching SF and
    /// the `bench` fingerprint (D24).
    pub nodes: u64,
    /// The `ARCHON_SEARCH_STATS` counters; zero-sized in the shipped build.
    pub stats: SearchStats,
    /// Max ply reached on the current PV line, counted from 1 (§3.9).
    pub sel_depth: i32,
    /// Depth of the current iteration (§3.5).
    pub root_depth: Depth,
    /// Root aspiration window width `beta - alpha`, stored before every root
    /// search including re-searches (§3.6); feeds doc 10's reduction formula.
    pub root_delta: Value,
    /// The per-worker log reduction table (design/10 §3.3), in 1024ths of a
    /// ply, indexed by *either* depth or move number (`MAX_MOVES = 256` and
    /// `MAX_PLY < 256`, so depth always fits). Built deterministically once.
    reductions: [i32; MAX_MOVES],
    /// Null-move pruning floor (design/10 §3.4 N11): NMP is disabled while
    /// `ply < nmp_min_ply`. A high-depth verification search raises it for the
    /// duration of that re-search so the subtree cannot recursively null-move
    /// back to the same (possibly zugzwang) cutoff. Reset to 0 each `go`.
    nmp_min_ply: i32,
    /// Per-color eval optimism (design/11 §3.6): set at aspiration setup from
    /// the root move's average score for the root side to move, negated for the
    /// opponent, then passed to every in-search `evaluate()` call. Zero until
    /// the first aspiration setup.
    optimism: [Value; 2],
    /// MultiPV line currently searched (§3.5; the outer loop index).
    pub pv_idx: usize,
    /// Number of PV lines to search this `go`: the `MultiPV` option, clamped in
    /// [`Worker::iterative_deepening`] to `root_moves.len()` (min 1).
    pub multi_pv: usize,
    /// Depth of the last FULLY completed ID iteration (design/12 §3.8). 0 until
    /// iteration 1 finishes — [`Worker::check_time`] refuses any hard stop while
    /// it is 0, so a real bestmove exists even under a tiny budget. Also the
    /// best-thread vote weight.
    pub completed_depth: Depth,
    /// What `go` asked for (design/12 §3.6).
    pub limits: Limits,
    /// Per-`go` time budgets (design/12 §3.7), main-worker state: `tm.init` runs
    /// at the top of each session, [`Worker::check_time`] enforces `maximum()`
    /// as the hard clock wall, and [`Worker::soft_stop`] compares `optimum()`
    /// between iterations.
    pub tm: TimeManager,
    /// DIS-TM: the head's predicted |search - static| for the ROOT, internal cp;
    /// 0.0 when the net has no DIS1 section or the arm is off. Polled ONCE per
    /// `go` — `disagreement_cached` belongs to whichever node last evaluated,
    /// so reading it at soft-stop time would be stale.
    #[cfg_attr(not(feature = "sota"), allow(dead_code))]
    root_dis_mag: f32,
    /// DIS-CENSUS: per-ply node id stamped on the way in so the node's searched
    /// value can be joined to it on the way out. Diagnostic only.
    #[cfg_attr(not(feature = "sota"), allow(dead_code))]
    dis_census_id: [u64; MAX_PLY + 8],
    /// DIS-TM: the ROOT static eval at search start, paired with `root_dis_mag`
    /// to measure how much predicted movement the search has already found.
    #[cfg_attr(not(feature = "sota"), allow(dead_code))]
    root_static_eval: f32,
    /// Best PV of the last iteration whose result was trusted (`!stop`) and not
    /// a forgotten mate (§3.5), so a `bestmove` is always printable.
    /// [`Worker::root_hygiene`] both maintains it and rolls back to it.
    pub last_best_move_pv: Vec<Move>,
    /// Score belonging to [`Self::last_best_move_pv`] (§3.5).
    pub last_best_move_score: Value,
    /// Iteration at which the best root move last changed — the TM stability
    /// signal consumed by doc 12 §3.9's four-factor soft stop.
    pub last_best_move_depth: Depth,
    /// Soft-stop `fallingEval` ring (design/12 §3.9): the previous four
    /// iterations' scores. Re-seeded per `go` from `persist.best_previous_score`
    /// (or `VALUE_DRAW` on a game's first search), never carried over.
    iter_value: [Value; ITER_VALUE_RING],
    /// Ring cursor into [`Self::iter_value`]; `iter_value[iter_idx]` is the
    /// score from four iterations ago at read time.
    iter_idx: usize,
    /// The *numerator* of the soft-stop `reduction` factor: seeded from persist
    /// at `go` start and held CONSTANT for the whole search. The running value
    /// that replaces it next move is [`Worker::tm_time_reduction`].
    tm_previous_time_reduction: f64,
    /// The running `time_reduction`: seeded to 1.0, recomputed each iteration,
    /// written back to persist at search end. Distinct from the field above so
    /// the numerator stays a per-move constant.
    tm_time_reduction: f64,
    /// Working copy of `persist.best_previous_average_score` for the
    /// soft-stop `fallingEval` factor (design/12 §3.9): seeded from persist
    /// at `go` start.
    tm_best_previous_average_score: Value,
    /// Per-thread eval state (design/11 §3.1); empty in Phase A.
    eval_ctx: EvalContext,
    /// Per-thread move-ordering statistics (design/09 §3.4): butterfly, low-ply,
    /// capture, continuation, pawn and correction history. `new_search()` decays
    /// it each `go`; `ucinewgame` clears it. ~20 MiB behind a `Box` so it never
    /// transits the stack.
    pub histories: Box<Histories>,
    /// Per-`go` snapshot of the `ARCHON_NMP_ALLNODE` A/B gate (a
    /// `OnceLock<bool>` read at every node that reaches the null-move block).
    nmp_allnode: bool,
    /// Poll-cadence bitmask for [`Worker::check_time`]: the real clock check runs
    /// when `nodes & mask == mask`. Recomputed per `go` — tighter under a
    /// `go nodes` limit so tiny budgets are honored.
    check_time_mask: u64,

    // ---- Cross-thread-visible handles (design/13 §3.2) ----
    /// 0 = main; feeds the aspiration jitter and gates helper-only behavior.
    pub thread_idx: usize,
    /// This thread's atomic counters, shared with its shell so the pool can read
    /// them mid-search (§3.10). Written owner-only; the plain `self.nodes` mirror
    /// stays the deterministic driver of every search decision (D24).
    pub counters: Arc<Counters>,
    /// The shared pool block: control flags and every sibling shell. Main uses it
    /// to wake helpers, aggregate counters and read reports; helpers only read
    /// the flags (§3.5).
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

    /// Main-worker-only, bounded observational LMR telemetry. It is compiled
    /// only for search-control experiments and has no authority over search.
    #[cfg(feature = "search-control")]
    search_trace: Option<crate::search_trace::SearchTrace>,
    /// One explicit deterministic replay intervention, never enabled in play.
    #[cfg(feature = "search-control")]
    lmr_intervention: Option<crate::search_intervention::LmrIntervention>,

    /// In-search observation capture (design/16 §3.5.5): buffer + shard writer.
    /// `None` unless `ObsCapturePath` is set on the main worker at
    /// `Threads == 1`. The sample gate is a pure function of the position key,
    /// so capture never perturbs search.
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
        // The log reduction table (design/10 §3.3). Index 0 is never read
        // (depth/move_number are >= 1 everywhere); `ln(1) == 0` makes
        // `reductions[1]` exactly 0. `as i32` truncates toward zero like SF's
        // `int(...)`, and the divisor is a power of two, so it is reproducible.
        let mut reductions = [0i32; MAX_MOVES];
        for (i, r) in reductions.iter_mut().enumerate().skip(1) {
            // The additive OFFSET defaults to 0 and adding literal 0.0 is
            // exact, so the table stays bit-identical at identity.
            *r = (params::RED_TABLE_NUM() as f64 / params::RED_TABLE_DEN() as f64 * (i as f64).ln()
                + params::RED_TABLE_OFFSET() as f64 / params::RED_TABLE_DEN() as f64)
                as i32;
        }
        // A detached worker (bench, unit tests) is its own 1-thread pool: it is
        // `is_main()`, shares its counters with its shell, never emits, and at
        // thread_idx 0 takes no aspiration jitter — bit-identical to Threads=1.
        let counters = Arc::new(Counters::new());
        let pool = PoolShared::detached(Arc::clone(&counters));
        Worker {
            board,
            root_moves: RootMoves::new(),
            stack: Box::new([Stack::default(); STACK_SIZE]),
            pv_table: Box::new([EMPTY_PV; MAX_PLY + 1]),
            nodes: 0,
            stats: SearchStats::NEW,
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
            root_dis_mag: 0.0,
            dis_census_id: [0; MAX_PLY + 8],
            root_static_eval: 0.0,
            last_best_move_pv: Vec::new(),
            last_best_move_score: -VALUE_INFINITE,
            last_best_move_depth: 0,
            iter_value: [VALUE_DRAW; ITER_VALUE_RING],
            iter_idx: 0,
            tm_previous_time_reduction: crate::timeman::PREVIOUS_TIME_REDUCTION_INIT,
            tm_time_reduction: crate::timeman::TIME_REDUCTION_INIT,
            tm_best_previous_average_score: VALUE_INFINITE,
            eval_ctx: EvalContext::new(),
            histories: Histories::new(),
            nmp_allnode: false,
            check_time_mask: params::CHECK_TIME_CALLS as u64 - 1,
            thread_idx: 0,
            counters,
            pool,
            manager: Some(Box::new(SearchManager::new())),
            output: None,
            options: Options::new(),
            minimal: false,
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

    /// Construct a pooled worker on its own thread (design/13 §3.4). The main
    /// worker gets a [`SearchManager`]; helpers get `None`. All heavy tables are
    /// allocated here so their pages are first-touch-local.
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

    /// True iff this is the main worker (thread 0) — the null-object gate: all
    /// main-only bookkeeping is `if self.is_main()`-guarded. Keyed on the
    /// manager's presence (helpers hold `None`).
    #[inline]
    pub fn is_main(&self) -> bool {
        self.manager.is_some()
    }

    /// `Setup` job (design/13 §3.6 step 4): install this `go`'s per-thread state
    /// on the worker's own thread and zero the per-`go` counters and report slot.
    /// [`Worker::iterative_deepening`] resets the rest itself.
    pub fn setup(&mut self, job: SetupJob) {
        self.board = job.board;
        self.limits = job.limits;
        self.options = job.options;
        self.output = Some(job.output);
        self.multi_pv = self.options.multipv().max(1) as usize;
        self.minimal = self.options.minimal();
        self.nodes = 0;
        self.stats = SearchStats::NEW;
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
        // `ucinewgame` is the game boundary: flush the buffered episode before
        // the histories reset. Best-effort — an I/O error must not crash us.
        #[cfg(feature = "datagen")]
        if let Some(rec) = self.obs_recorder.as_mut()
            && let Err(error) = rec.flush_game()
        {
            eprintln!("TEMPO obs capture: game flush failed: {error}");
        }
    }

    /// A `Search` job entry (design/13 §3.4): the main worker orchestrates
    /// (TM, helper wake, output, voting); a helper just deepens until `stop`.
    pub fn start_searching(&mut self, tt: &Arc<TranspositionTable>) {
        // Install this thread's eval from the process-wide active net, once per
        // `go`, before ID refreshes the root accumulator. `None` ⇒ PSQT.
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
                        eprintln!("TEMPO LMR intervention disabled: {error}");
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
                eprintln!("TEMPO search trace write failed: {error}");
            }
        }
        #[cfg(feature = "search-control")]
        if let Some(intervention) = self.lmr_intervention.as_ref() {
            if let Err(error) = intervention.flush() {
                eprintln!("TEMPO LMR intervention result write failed: {error}");
            }
        }
        // Release this `go`'s sink so the UCI layer regains sole ownership of
        // the stream; the next `go`'s `Setup` reinstalls it.
        self.output = None;
    }

    /// Install this thread's eval from the process-wide active NNUE net, read
    /// once at the search boundary. `None` ⇒ PSQT. Unit tests inject a net into
    /// `eval_ctx` directly to stay independent of the global.
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
    }

    /// Lazily create the in-search observation recorder when `ObsCapturePath` is
    /// set. Idempotent, once per `go` on the main worker; refuses (with an
    /// `info string`) at `Threads != 1` to keep capture deterministic. Silent on
    /// success so fastchess sees no unexpected between-game stdout.
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
        // Concurrent matches hand every process the same option value and the
        // shard writer truncates, so suffix the pid to give each its own shard.
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
            engine_version: format!("TEMPO {} (in-search-capture v0)", env!("CARGO_PKG_VERSION")),
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

    /// Node-entry gate for in-search capture. `Some` only when the recorder is
    /// open, the node is deep enough, and its key passes the deterministic sample
    /// gate with buffer room. Disabled, it is one never-taken branch per node.
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

        // B. TM init + the single per-`go` TT generation bump. `persist` moves
        // into a local so the driver can mutate it; the manager box stays
        // present, so `is_main()` remains true throughout.
        let mut persist = std::mem::take(&mut self.manager.as_mut().expect("main").persist);
        let mut limits = self.limits.clone();
        let (us, game_ply) = (self.board.side_to_move(), self.board.game_ply());
        let options = self.options.clone();
        self.tm
            .init(&mut limits, us, game_ply, &options, &mut persist);
        self.limits = limits;
        let start_time = self.limits.start_time;
        // After TM init, before helpers can observe the table (`submit`
        // publishes it).
        tt.new_search();

        // C. Wake the helpers; each runs the identical `iterative_deepening`.
        for shell in pool.shells.iter().skip(1) {
            shell.submit(Job::Search { tt: Arc::clone(tt) });
        }

        // The closure captures owned handles only (no `self` borrow), so it
        // coexists with the `&mut self` the driver needs.
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
        // The per-game persist is restored AFTER voting, so its winner-derived
        // seeds can come from the voted thread.

        // §3.11 post-search `nodestime` accounting: book the nodes consumed
        // minus the increment credit, only when `init` converted our clock.
        if self.limits.npmsec > 0 {
            let inc_nodes = self.limits.inc[us as usize];
            self.tm.advance_nodes_time(self.nodes as i64 - inc_nodes);
        }

        // Publish our final node count for the aggregate before the epilogue.
        self.counters.nodes.store(self.nodes, Ordering::Relaxed);

        // D. Never answer early: while pondering or infinite, spin until
        // `stop`/`ponderhit`. Burns one core for the window, as SF does.
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
        // §3.7 G). `iterative_deepening` already filled `persist` from main's
        // last iteration; overwrite the two winner-derived seeds only when a
        // helper won. Under fixed-depth `go` the winner is forced to 0, so this
        // never fires there. The seeds feed only next move's TM inputs.
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

    /// Build this worker's end-of-search [`SearchReport`] from `root_moves[0]`.
    /// An empty root move list reports an empty PV and `-VALUE_INFINITE`, which
    /// voting never elects.
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
    /// `info` line when the GUI's last-seen PV differs from the played move, then
    /// the exactly-once `bestmove [ponder]` (or `bestmove (none)`).
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
                    // The winner's own deepest completed iteration, so depth,
                    // sel_depth, score and PV all describe the SAME thread.
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

    /// The iterative-deepening driver (design/07 §3.5): the per-`go` entry point
    /// for every search mode. Builds the root move list (with the `searchmoves`
    /// filter), then deepens one aspirated root search at a time.
    ///
    /// The session layer has already bumped `tt.new_search()`; the driver never
    /// does. A search that observes `stop` returns untrusted values, so every
    /// consumer here is `!stop`-guarded. `on_info` fires once per completed PV
    /// line, never after a stop.
    ///
    /// On return `root_moves[0]` holds the best move/PV/score — trusted even
    /// after a mid-iteration stop, because root bookkeeping only publishes
    /// completed subtree results. With no legal moves it returns immediately.
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
        self.stats = SearchStats::NEW;
        self.sel_depth = 0;
        self.root_depth = 0;
        self.completed_depth = 0;
        self.pv_idx = 0;
        // Reset here too so a reused worker on the DIRECT path (bench/tests,
        // which never call `setup`) starts clean, and so `nodes_searched` is
        // accurate from the first `info` line.
        self.counters.nodes.store(0, Ordering::Relaxed);
        self.counters.best_move_changes.store(0, Ordering::Relaxed);
        self.nmp_min_ply = 0;
        // The process-constant env probe, read ONCE per `go` (never per node).
        self.nmp_allnode = crate::search::nmp_allnode_probe();
        self.check_time_mask = self.compute_check_time_mask();
        self.last_best_move_pv.clear();
        self.last_best_move_score = -VALUE_INFINITE;
        self.last_best_move_depth = 0;
        // TUNE-BUILD ONLY: rebuild the table from the live tunables — Worker
        // construction predates `setoption` in SPSA drivers, so a
        // constructed-only table would silently ignore the option.
        #[cfg(feature = "tune")]
        {
            for (i, r) in self.reductions.iter_mut().enumerate().skip(1) {
                *r = (params::RED_TABLE_NUM() as f64 / params::RED_TABLE_DEN() as f64
                    * (i as f64).ln()
                    + params::RED_TABLE_OFFSET() as f64 / params::RED_TABLE_DEN() as f64)
                    as i32;
            }
        }
        // Soft-stop factor state (doc 12 §3.9). `increase_depth` starts `true`
        // so fixed-depth/fixed-node search never damps its depth. Only the MAIN
        // worker writes it: the pool already stored `true` in `start_thinking`,
        // so a helper re-storing here could clobber main's soft-stop verdict.
        // Detached/bench workers are `is_main()`, keeping the direct path's
        // per-`go` reset.
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
        self.eval_ctx.refresh_root(&self.board);
        // DIS-TM: one root poll per search. Uses the FULL `disagreement` (it
        // re-runs `activate_ft`) because the cached form needs a fresh eval at
        // THIS position and none has run.
        #[cfg(feature = "sota")]
        {
            // FALSIFICATION CONTROL (`ARCHON_DIS_TM_RANDOM=1`): same arm and
            // magnitude, no INFORMATION — a draw from the same empirical
            // distribution keyed on the root hash. Separates "the head picks
            // the right moments" from "redistributing time at all helps".
            self.root_dis_mag = if dis_tm_c() != 0.0 && dis_tm_random() {
                let h = self.board.key().wrapping_mul(0x9E37_79B9_7F4A_7C15);
                110.0 + ((h >> 33) % 141) as f32
            } else if dis_tm_c() != 0.0 {
                self.eval_ctx
                    .disagreement_root(&self.board)
                    .map_or(0.0, |d| (f32::from(d[1]) * 600.0).max(0.0))
            } else {
                0.0
            };
            self.root_static_eval = if self.root_dis_mag > 0.0 && self.board.checkers().is_empty() {
                // `evaluate` must never run in check (design/11 §3.1), and a
                // root in check has no static eval to budget against.
                crate::eval::evaluate(&self.board, &mut self.eval_ctx, 0, 0) as f32
            } else {
                self.root_dis_mag = 0.0;
                0.0
            };
        }
        // Per-`go` history refresh (design/09 §3.4): decay butterfly history
        // toward zero and refill low-ply history. Correction/capture/
        // continuation tables persist across the game.
        self.histories.new_search();
        // NNUE root accumulator (design/11 §3.4.4): refresh `stack[0]` from the
        // root, both perspectives, once per search, so it stays valid however
        // much game history UCI replayed. Only the already-installed net is
        // refreshed; a no-op when NNUE is off.
        // MultiPV: the option clamped to the legal-move count (min 1). At
        // MultiPV 1 the loop below is exactly single-PV.
        let multi_pv = self.multi_pv.clamp(1, self.root_moves.len());
        // Time-management instability input (doc 12 §3.9): halved every
        // iteration, harvested from `best_move_changes`, consumed by the
        // four-factor soft stop.
        let mut tot_best_move_changes = 0.0f64;
        // TM-driven re-search damping (doc 12 §3.9): bumped when the previous
        // soft stop said "don't deepen". Stays 0 when the soft stop never runs
        // (fixed depth/nodes), leaving §3.6's adjusted depth untouched.
        let mut search_again_counter: i32 = 0;

        while self.root_depth + 1 < MAX_PLY as Depth
            && !stop.load(Ordering::Relaxed)
            // `go depth d` (design/12 §3.6): the cap applies on the MAIN worker
            // only. Helpers keep deepening to enrich the shared TT and are
            // halted by main's step-E `stop`; voting is disabled under
            // `go depth`, so thread 0's move is chosen regardless.
            && (!self.is_main() || self.limits.depth.is_none_or(|d| self.root_depth < d))
        {
            self.root_depth += 1;
            // §3.9 search-again: a soft budget more than half spent leaves
            // `increase_depth == false`, so this iteration re-searches near the
            // same effective depth instead of starting a deeper pass that would
            // abort mid-way. The flag is shared; the counter is per-worker.
            if !self.pool.increase_depth.load(Ordering::Relaxed) {
                search_again_counter += 1;
            }
            // Age the instability metric; our divisor deviates from SF's /2 —
            // see params::INSTABILITY_CHANGES_DECAY.
            tot_best_move_changes /= params::INSTABILITY_CHANGES_DECAY();

            // Snapshot last iteration (§3.5).
            for rm in self.root_moves.iter_mut() {
                rm.previous_score = rm.score;
            }

            // SF's `bestValue`: what the LAST MultiPV line's root search
            // settled on. Feeds the soft-stop `fallingEval` + `iter_value` ring.
            // Identical to `root_moves[0].score` at MultiPV 1.
            let mut root_best_value = -VALUE_INFINITE;
            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;
                self.sel_depth = 0; // per PV line (§3.9)
                root_best_value = self.aspiration_search(tt, stop, search_again_counter);
                // Order the completed lines among themselves (§3.5); a no-op at
                // MultiPV 1, load-bearing above it.
                sort_descending(&mut self.root_moves[..=pv_idx]);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // §3.13 emission: below the node throttle publish only when the
                // iteration's LAST line finishes; above it, after every line.
                // Republish exactly the lines searched THIS iteration
                // (`0..=pv_idx`) — a higher-index line still holds last
                // iteration's score/pv, and surfacing it under the current depth
                // label would make the block non-monotonic. Helpers skip it:
                // building a `SearchInfo` has side-effecting reads the compiler
                // cannot elide behind their no-op callback.
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
                            // Aggregate across all threads (design/13 §3.10).
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

            // Mark the iteration fully completed: unblocks `check_time`'s hard
            // stops now that a real bestmove exists at this depth.
            if !stop.load(Ordering::Relaxed) {
                self.completed_depth = self.root_depth;
            }

            // Harvest the per-iteration instability signal across ALL threads:
            // only main aggregates, `swap(0)`-ing every counter so none is lost.
            // Helpers never reach `soft_stop`, so their local copy stays 0.
            if self.is_main() {
                let sum: u64 = self
                    .pool
                    .shells
                    .iter()
                    .map(|s| s.counters.best_move_changes.swap(0, Ordering::Relaxed))
                    .sum();
                tot_best_move_changes += sum as f64;
            }

            // `go mate n` (§3.5): a COMPLETED iteration whose front score
            // proves a mate — ours or theirs — within n full moves ends the
            // search. `2` converts full moves to plies (structural).
            if !stop.load(Ordering::Relaxed)
                && self.limits.mate.is_some_and(|n| {
                    let s = self.root_moves[0].score;
                    (is_mate(s) || is_mated(s)) && VALUE_MATE - s.abs() <= 2 * n
                })
            {
                stop.store(true, Ordering::Relaxed);
            }

            // TM soft stop between completed iterations: sets `increase_depth`
            // for the next one and maintains the `iter_value` ring.
            self.soft_stop(stop, multi_pv, tot_best_move_changes, root_best_value);
        }
        // Persist the per-game TM state for this game's next `go`: the front
        // move's score and running average seed next move's `iter_value` and
        // `fallingEval`. Only a truly completed iteration may overwrite them —
        // an aborted depth-0 search leaves them alone.
        if let Some(front) = self.root_moves.first().filter(|_| self.completed_depth > 0) {
            persist.best_previous_score = front.score;
            persist.best_previous_average_score = front.average_score;
        }
        // The running value (this move's last iteration, or the 1.0 seed if the
        // factor block never ran) becomes next move's numerator.
        persist.previous_time_reduction = self.tm_time_reduction;

        // Diagnostic node anatomy: one stderr line per completed search. Gated
        // at COMPILE time, not only by the env var — the counters cost per-node
        // stores that a runtime `if` cannot delete.
        #[cfg(any(test, feature = "search-stats"))]
        if std::env::var_os("ARCHON_SEARCH_STATS").is_some() {
            let st = &self.stats;
            eprintln!(
                "SSTATS nodes={} qnodes={} qshare={:.3} depth={} fh={} fhf={} \
                 fhf_rate={:.3} avg_moves_at_cut={:.2} ttcut={} ext={} research={}",
                self.nodes,
                st.qnodes,
                st.qnodes as f64 / self.nodes.max(1) as f64,
                self.completed_depth,
                st.fh,
                st.fh_first,
                st.fh_first as f64 / st.fh.max(1) as f64,
                st.fh_moves as f64 / st.fh.max(1) as f64,
                st.ttcut,
                st.ext,
                st.research
            );
            eprintln!(
                "SSTATS2 lmr_n={} lmr_avg_plies={:.3} nmp_try={} nmp_cut={} nmp_rate={:.3}",
                st.lmr_n,
                st.lmr_plies as f64 / st.lmr_n.max(1) as f64,
                st.nmp_try,
                st.nmp_cut,
                st.nmp_cut as f64 / st.nmp_try.max(1) as f64
            );
            // The per-gate census.
            let mut line = String::from("SSTATS3");
            for (name, count) in Gate::NAMES.iter().zip(st.gates.iter()) {
                line.push_str(&format!(" {name}={count}"));
            }
            eprintln!("{line}");
        }
    }

    /// End-of-iteration root hygiene (design/07 §3.5): lastBestMove tracking
    /// with the forgotten-mate guard, then the rollback/repair pass. Requires a
    /// non-empty `root_moves`.
    ///
    /// - **Forgotten mate**: once an iteration proves `mate n`, a later weaker
    ///   or bound score must not replace it — roll the front move back.
    /// - **Aborted loss**: an *exact* mated/TB-loss score from a mid-iteration
    ///   stop is untrusted (unsearched root moves might refute it). Restore the
    ///   previous iteration's move/score, or mark the score a lower bound so the
    ///   GUI never sees a fake forced loss. The `pv_idx == 0` guard keeps an
    ///   aborted LATER MultiPV line from leapfrogging a proven loss.
    fn root_hygiene(&mut self, stopped: bool) {
        let front = &self.root_moves[0];
        let forgotten_mate = self.last_best_move_score != -VALUE_INFINITE
            && (is_mate(self.last_best_move_score) || is_mated(self.last_best_move_score))
            && (front.score.abs() < self.last_best_move_score.abs() || front.score_is_bound());

        if !stopped {
            // The TM stability signal: the iteration the best move changed.
            if self.last_best_move_pv.first() != front.pv.first() {
                self.last_best_move_depth = self.root_depth;
            }
            // Never replace a proven mate with a weaker or bound re-score.
            if !forgotten_mate {
                self.last_best_move_pv.clone_from(&self.root_moves[0].pv);
                self.last_best_move_score = self.root_moves[0].score;
            }
        }

        let aborted_loss = stopped && self.pv_idx == 0 && self.root_moves[0].score_is_exact_loss();
        if aborted_loss || (self.root_moves[0].score != -VALUE_INFINITE && forgotten_mate) {
            if let Some(&remembered) = self.last_best_move_pv.first() {
                // Move the remembered line's root move to the front; the rotate
                // preserves the others' order (next iteration's ordering).
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
                // Aborted depth-1 search: nothing to restore, so label the loss
                // a lower bound (unsearched moves may beat it).
                self.root_moves[0].score_lowerbound = true;
            }
        }
    }

    /// The aspiration loop (design/07 §3.6): one iteration's root search for the
    /// current PV line, in a window centred on that line's `average_score` and
    /// re-searched with a widening window until the value lands strictly inside
    /// it (or the search stops).
    ///
    /// - Initial half-width `delta` grows with the line's mean-squared-score
    ///   EMA; the first iterations start from the sentinel, so the window clamps
    ///   to `(-INF, INF)` and depth 1 is effectively full-window.
    /// - `adjusted_depth` shrinks per CONSECUTIVE fail-high and by the TM's
    ///   search-again damping.
    /// - FAIL LOW slides the window down and resets `failed_high_cnt`, revoking
    ///   any pending ponderhit stop (the pondered move may be refuted).
    /// - FAIL HIGH backs alpha off at most to `beta - delta`, never below its
    ///   current value; beta chases the returned value. Each re-search widens
    ///   delta by ~1/3.
    ///
    /// `self.root_delta` is stored before EVERY root search, re-searches
    /// included — doc 10's reduction formula divides by it.
    ///
    /// Returns the value the root search settled on (SF's `bestValue`):
    /// `root_moves[0].score` at MultiPV 1, the weakest line above it.
    fn aspiration_search(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        search_again_counter: i32,
    ) -> Value {
        let rm = &self.root_moves[self.pv_idx];
        // Lazy-SMP aspiration jitter (design/13 §3.1): the `thread_idx % 8` term
        // makes helpers search slightly different windows so they diverge.
        // Vacuous at `thread_idx == 0`, keeping Threads=1 bit-identical.
        let mut delta = (params::ASPIRATION_BASE_DELTA() as i64
            + (self.thread_idx % params::ASPIRATION_JITTER_PERIOD()) as i64
            + rm.mean_squared_score.abs() / params::ASPIRATION_MSS_DIVISOR())
            as Value;
        let avg = rm.average_score;
        // Optimism (design/11 §3.6): lean the eval a few cp toward the side the
        // search already favours. SF-parity shape including the unguarded
        // first-iteration sentinel (iteration 1 is full-window and self-corrects).
        let us = self.board.side_to_move();
        let opt = params::OPTIMISM_NUM() * avg / (avg.abs() + params::OPTIMISM_DEN_BASE());
        self.optimism[us as usize] = opt;
        self.optimism[us.flip() as usize] = -opt;
        let mut alpha = (avg - delta).max(-VALUE_INFINITE);
        let mut beta = (avg + delta).min(VALUE_INFINITE);
        let mut failed_high_cnt: i32 = 0;
        // Assigned before any break, so definite-assignment holds after the loop.
        let mut best;

        loop {
            let adjusted_depth = (self.root_depth
                - params::ASPIRATION_FAILHIGH_DEPTH_MULT() * failed_high_cnt
                - params::ASPIRATION_SEARCH_AGAIN_NUM() * (search_again_counter + 1)
                    / params::ASPIRATION_SEARCH_AGAIN_DEN())
            .max(1);
            self.root_delta = beta - alpha;
            best = self.search::<Root>(tt, stop, alpha, beta, adjusted_depth, false, 0);

            // Stable re-sort per attempt (§3.4): the searched PV move floats to
            // the front, everything else keeps last iteration's order.
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
                self.stats.research();
                // A root fail-low may mean the pondered move is refuted, so
                // revoke any pending ponderhit stop and keep thinking.
                if let Some(m) = &mut self.manager {
                    m.stop_on_ponderhit = false;
                }
            } else if best >= beta {
                // FAIL HIGH: alpha never loosens past its current value.
                alpha = (beta - delta).max(alpha);
                beta = (best + delta).min(VALUE_INFINITE);
                failed_high_cnt += 1;
                self.stats.research();
            } else {
                break; // inside the window: done
            }

            // Widen ~×1.34 per re-search. The cap is semantics-preserving: once
            // delta >= 2·VALUE_INFINITE both edges already clamp to ±INF, and
            // unbounded growth would eventually overflow the i32.
            delta = (delta
                + params::ASPIRATION_WIDEN_NUM() * delta / params::ASPIRATION_WIDEN_DEN())
            .min(2 * VALUE_INFINITE);
        }
        best
    }

    /// The best root line of the current `go`; empty if the root has no legal
    /// moves. Trusted even after a stop (§3.5).
    pub fn root_pv(&self) -> &[Move] {
        self.root_moves.first().map_or(&[][..], |rm| &rm.pv)
    }

    /// The best root move of the current `go`, if any. Always printable once
    /// `iterative_deepening` returns with legal moves present.
    pub fn best_move(&self) -> Option<Move> {
        self.root_pv().first().copied()
    }

    // `search`'s parameter list is the doc 07 §3.7.1 node signature plus the
    // two plumbed-through borrows; bundling would only rename it.
    #[allow(clippy::too_many_arguments)]
    /// DIS-CENSUS: emit this node's searched value for the row stamped on the
    /// way in, and clear the stamp. Must be called at EVERY post-census return
    /// path — labelling only the final store drops the early fail-high returns,
    /// which are exactly the RIGHT-prune examples.
    #[cfg(feature = "sota")]
    #[inline]
    fn dis_census_result(&mut self, cur: usize, value: Value, red: u32, rer: u32) {
        if self.dis_census_id[cur] != 0 {
            eprintln!(
                "DISRESULT id={} value={} red={} rer={}",
                self.dis_census_id[cur], value, red, rer
            );
            self.dis_census_id[cur] = 0;
        }
    }

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
        // Step 1 — upcoming repetition (design/10 item 19 "N1"). If a draw is
        // available in one move and we are worse than a draw, the draw is a
        // LOWER BOUND here: raise alpha to it, and cut if that beats beta. Runs
        // ABOVE the depth-0 boundary (two table probes, no movegen) so shallow
        // nodes get the cutoff too.
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
        // stack, so bound the child's ply below MAX_PLY.
        depth = depth.min(MAX_PLY as Depth - 1);

        // Node-entry invariants (§3.7.1).
        debug_assert!(!(NT::PV && cut_node), "PV nodes are never cut nodes");
        debug_assert!(NT::PV || alpha == beta - 1, "NonPV requires a zero window");
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(NT::ROOT == (ply == 0));

        // Step 2 — init. `cur ± k` stays in bounds for all ply <= MAX_PLY.
        let cur = ss(ply);
        // DIS-CENSUS hygiene: a stamp left by a node that returned through an
        // uninstrumented path must not attach its id to the next node at this
        // ply. A lost label is acceptable; a wrong one is not.
        #[cfg(feature = "sota")]
        if dis_census_on() {
            self.dis_census_id[cur] = 0;
        }
        // DIS-CENSUS (LMR-veto probe): per-node reduced-scout / re-search
        // counters, emitted with DISRESULT.
        #[cfg(feature = "sota")]
        let mut census_lmr_red: u32 = 0;
        #[cfg(feature = "sota")]
        let mut census_lmr_rer: u32 = 0;
        let in_check = !self.board.checkers().is_empty();
        self.stack[cur].in_check = in_check;
        self.stack[cur].move_count = 0;
        self.stack[cur].stat_score = 0;
        self.stats.gate(Gate::MainEntry);
        if in_check {
            self.stats.gate(Gate::MainInCheck);
        }
        // Deliberate SF divergence: no followPV tracking.
        // Parent-move facts (design/10 §3.4 N2): `prev_sq` is the parent's
        // destination (None after a null move / at the root), `prior_capture`
        // whether the parent captured.
        let prev_move = self.stack[cur - 1].current_move;
        let prev_sq = prev_move.is_ok().then(|| prev_move.to());
        let prior_capture = self.board.captured_piece().is_some();
        // The UNSCALED correction sum (design/09 §3.10): its `/131072` is the
        // eval delta; the pruning couplings below use their own divisors.
        let correction_value = self
            .histories
            .correction_value(&self.board, &self.stack[..], cur);
        // Consumed-and-reset (design/10 §3.4 N6): take the parent's LMR
        // reduction, then clear it so it cannot go stale.
        let prior_reduction = self.stack[cur - 1].reduction;
        self.stack[cur - 1].reduction = 0;
        self.stack[cur + 2].cutoff_cnt = 0;
        if NT::PV {
            self.pv_table[ply].clear();
            self.sel_depth = self.sel_depth.max(ply as i32 + 1);
        }
        // §3.9 cadence: every main-search node entry; qsearch relies on these.
        self.check_time(stop);

        if !NT::ROOT {
            // Step 3 — abort/draw/max-ply. A stop here returns the (untrusted)
            // dithered draw; the parent's step-11 check discards it.
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

            // Step 4 — mate-distance pruning: an immediate mate here cannot
            // beat a shorter mate already found higher up.
            alpha = alpha.max(mated_in(ply as i32));
            beta = beta.min(mate_in(ply as i32 + 1));
            if alpha >= beta {
                return alpha;
            }
        }

        // Step 5 — TT probe. The single read-side conversion boundary: a hit's
        // value passes through `value_from_tt` (with the 50-move downgrade)
        // before any use, and a tag-matched-but-unoccupied entry normalizes to
        // VALUE_NONE. The ttMove needs no pseudo_legal/legal check here: it is
        // only ever matched against `generate_legal`'s output.
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
        // singular verification keep the enclosing node's flag — same ply.
        if self.stack[cur].excluded_move == Move::NULL {
            self.stack[cur].tt_pv = NT::PV || (tt_hit && tt_data.is_pv);
        }
        // Both knobs are resolved once per node so the arm bench's override
        // costs one branch, not one per use site.
        #[cfg(not(feature = "arm70"))]
        let fl_mode = params::TT_FAILLOW_MOVE();
        #[cfg(feature = "arm70")]
        let fl_mode = match crate::arm70::arm().fl_move {
            -1 => params::TT_FAILLOW_MOVE(),
            v => v,
        };
        #[cfg(not(feature = "arm70"))]
        let up_blind = params::TT_UPPER_PRUNE_BLIND();
        #[cfg(feature = "arm70")]
        let up_blind = match crate::arm70::arm().tt_up_blind {
            -1 => params::TT_UPPER_PRUNE_BLIND(),
            v => v,
        };
        // A move on an UPPER-bound entry never raised alpha, so it carries none
        // of the "this node is resolved" information IIR, RFP's ttMove gate and
        // the two LMR no-ttMove terms read `tt_data.mv` for.
        // `TT_UPPER_PRUNE_BLIND` is a BITMASK over those four sites — 1 IIR,
        // 2 RFP, 4 LMR-cutnode, 8 LMR-fulldepth, 15 all. `tt_capture` is
        // deliberately NOT masked: it also shapes the singular margins, so
        // including it would make the arm two changes under one name.
        let tt_upper_hidden =
            up_blind != 0 && tt_data.bound == Bound::Upper && tt_data.mv != Move::NULL;
        let pruning_tt_move = |bit: i32| {
            if tt_upper_hidden && (up_blind & bit) != 0 {
                Move::NULL
            } else {
                tt_data.mv
            }
        };
        let tt_capture = tt_data.mv != Move::NULL && self.board.capture_stage(tt_data.mv);

        // Steps 6+7 — the static-eval slot, then the TT cutoff (deliberately
        // AFTER the eval slot so the hindsight depth adjustments can feed it).
        // In check both are skipped: the node borrows the grandparent's eval
        // and jumps to the move loop, since no pruning is sound in check.
        let raw_eval;
        // DIS-LMR: this node's predicted |search - static| in cp, 0.0 when
        // unavailable. Captured at the FRESH-eval site — the only point where
        // the head's cached FT vector belongs to THIS position — and carried
        // down, since by reduction time the scratch holds a child's activation.
        #[cfg(feature = "sota")]
        let mut dis_mag_cp: f32 = 0.0;
        // `improving` is hoisted out of the not-in-check branch: the LMR
        // reduction reads it even in check, where SF pins it `false`.
        let mut improving;
        if in_check {
            raw_eval = VALUE_NONE;
            self.stack[cur].static_eval = self.stack[cur - 2].static_eval;
            improving = false;
        } else {
            // Step 6 — eval. A singular verification (excluded_move set) REUSES
            // the enclosing node's corrected static eval — same ply — and stores
            // no eval-cache entry. Otherwise reuse a valid cached RAW eval (D22)
            // if present, else `evaluate()`; a TT miss caches the raw eval at
            // DEPTH_UNSEARCHED so transpositions skip the eval.
            if self.stack[cur].excluded_move != Move::NULL {
                // `raw_eval` is DEAD here (every TT store of it is
                // `excluded_move == NULL`-gated) and `ss.static_eval` holds the
                // enclosing node's CORRECTED eval, which must never be
                // mislabeled as raw (D22). The sentinel keeps that loud.
                raw_eval = VALUE_NONE;
            } else {
                let tt_served = tt_hit && is_valid(tt_data.eval);
                raw_eval = if tt_served {
                    tt_data.eval
                } else {
                    let fresh = evaluate(
                        &self.board,
                        &mut self.eval_ctx,
                        self.optimism[self.board.side_to_move() as usize],
                        ply,
                    );
                    // DIS-EVAL: correct the RAW eval by the head's predicted
                    // (search - static). Sited INSIDE the fresh branch on
                    // purpose — the poll reads the scratch the last head
                    // forward filled, and a TT-served eval never ran the head.
                    // The DIS-LMR magnitude is captured here once; a second
                    // poll would cost another head forward.
                    #[cfg(feature = "sota")]
                    if dis_lmr_c() != 0.0 {
                        if let Some(d) = self.eval_ctx.disagreement_cached() {
                            dis_mag_cp = (f32::from(d[1]) * 600.0).max(0.0);
                        }
                    }
                    #[cfg(feature = "sota")]
                    let fresh = {
                        let ce = dis_eval_c();
                        if ce != 0.0 && is_valid(fresh) {
                            match self.eval_ctx.disagreement_cached() {
                                // col 0 = signed, network_scale units. Clamped:
                                // the head is an unbounded regressor and one
                                // wild prediction must not invent a winning
                                // score out of a drawn position.
                                Some(d) => {
                                    // ORTHOGONALISED: the signed output is
                                    // dominated by a global rescale (`pred_cp ~
                                    // SLOPE*static + INTERCEPT`), and adding it
                                    // whole rescales the eval — a settled
                                    // interior optimum here, so the naive form
                                    // loses. Subtracting the fitted line leaves
                                    // only the per-position component.
                                    const DIS_SLOPE: f32 = -0.4361;
                                    const DIS_INTERCEPT: f32 = 23.7;
                                    let pred = f32::from(d[0]) * 600.0;
                                    let rescale_part = DIS_SLOPE * fresh as f32 + DIS_INTERCEPT;
                                    let adj =
                                        ((pred - rescale_part) * ce).clamp(-300.0, 300.0) as Value;
                                    fresh + adj
                                }
                                None => fresh,
                            }
                        } else {
                            fresh
                        }
                    };
                    fresh
                };
                // S1 (design/10 N6, D22): `ss.static_eval` is the CORRECTED eval
                // and every downstream consumer reads it. The TT keeps the RAW
                // eval: corrections are per-path (continuation keys), so a
                // cached corrected value would smuggle one path's into another.
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

            // `improving`: our eval beats our eval two plies ago. Uses
            // `ss.static_eval`, never the substituted `eval`; an unset
            // predecessor is the large `VALUE_NONE` sentinel.
            improving = self.stack[cur].static_eval > self.stack[cur - 2].static_eval;
        }

        // N6 trend flag + hindsight depth adjustments (design/10 §3.11), ordered
        // BEFORE the in-check jump so they run for ALL nodes.
        // `opponent_worsening` means the position is turning our way. A hard
        // parent reduction that did NOT turn it the opponent's way looks
        // unjustified — deepen the node; extending a hard-reduced check evasion
        // is exactly where this pays, so it must not be gated out of check. A
        // moderate reduction with a healthy combined eval sum shallows it.
        let opponent_worsening = self.stack[cur].static_eval > -self.stack[cur - 1].static_eval;
        if prior_reduction >= params::HINDSIGHT_RED_UP() && !opponent_worsening {
            depth += 1;
            self.stats.gate(Gate::HindUp);
        }
        if prior_reduction >= params::HINDSIGHT_RED_DOWN()
            && depth >= 2
            && self.stack[cur].static_eval + self.stack[cur - 1].static_eval
                > params::HINDSIGHT_EVAL_SUM()
        {
            depth -= 1;
            self.stats.gate(Gate::HindDown);
        }

        // N6b — the NonPV TT cutoff runs for ALL nodes, in check or not: it keys
        // only on the stored value/depth/bound, never on the static eval. The
        // child-TT verification's make/unmake re-checks `pseudo_legal && legal`,
        // so it is safe in check too.
        {
            // Cutoff shape: enough stored depth (one ply of slack for upper
            // bounds), a valid value, the bound pointing the cutoff's way, the
            // cutNode-agreement guard at shallow depth, the rule-50 guard, and
            // — deep — a child-TT re-verification. The rule-50 guard wraps only
            // the cutoff RETURN, not the move-ordering updates. `tt_cut_base` is
            // shared with the wrong-direction penalize so they cannot drift.
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
                // N6b extras: a beta-side cutoff means the stored ttMove
                // "worked" here. These run regardless of rule50.
                let ttm = tt_data.mv;
                if ttm != Move::NULL && tt_value >= beta {
                    let us = self.board.side_to_move();
                    // The from-square guard keeps a rare colliding TT move from
                    // mis-indexing the histories.
                    if !self.board.capture_stage(ttm)
                        && self
                            .board
                            .piece_on(ttm.from())
                            .is_some_and(|p| p.color == us)
                    {
                        self.stats.gate(Gate::TtcutBonus);
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
                // At rule50 >= limit the node falls through to a real search
                // even though the bonus/malus above already fired.
                if (self.board.rule50_count() as i32) < params::TT_CUTOFF_RULE50_LIMIT() {
                    // N6b child-TT verification: a deep cutoff on a non-decisive
                    // value re-probes the ttMove's child entry; only an invalid
                    // child value or one AGREEING with the cutoff direction
                    // confirms it.
                    if depth >= params::TTCUT_VERIFY_DEPTH()
                        && !is_decisive(tt_value)
                        && tt_data.mv != Move::NULL
                        && self.board.pseudo_legal(tt_data.mv)
                        && self.board.legal(tt_data.mv)
                    {
                        // NOTE: keep the probe body out of line — inlined it
                        // duplicates the whole make/unmake for a rare path.
                        let child_value = self.tt_verify_child(tt, tt_data.mv, ply);
                        if !is_valid(child_value) || (tt_value >= beta) == (-child_value >= beta) {
                            self.stats.tt_cut();
                            return tt_value;
                        }
                    } else {
                        self.stats.tt_cut();
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
                // Wrong-direction bound: the cutoff failed only because the
                // bound points the wrong way — age that entry out faster.
                self.stats.gate(Gate::TtPenalize);
                probe.writer.penalize(1);
            }
        }

        // Steps N8–N13 are skipped in check: no eval-based pruning is sound
        // there, so the node jumps straight to N14 / the move loop.
        if !in_check {
            // The pruning value `eval`: the corrected static eval, then the
            // TT-value-as-better-eval substitution. Fenced to
            // `excluded_move == NULL` — a singular verification must razor/RFP
            // on the enclosing node's static eval, not on the stored value of
            // the very move it excludes, which would corrupt the verdict.
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

            // N8 — grade the opponent's last quiet by the static-eval swing it
            // produced. `prev_sq` is Some only for a real parent move.
            if let Some(sq) = prev_sq
                && !self.stack[cur - 1].in_check
                && !prior_capture
            {
                self.stats.gate(Gate::EvalDiff);
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
                    self.histories.pawn_slot_mut(pawn_key)[pc.index()][sq.idx()]
                        .update(params::EVALDIFF_PAWN_MULT() * eval_diff);
                }
            }

            // N9 — razoring: a non-PV node whose `eval` sits below alpha by more
            // than the quadratic margin drops straight to qsearch, which IS the
            // verification. Uses the TT-substituted `eval`, not
            // `ss.static_eval`. No depth cap (the depth² margin makes deep
            // razoring vanishingly rare); `!PV` keeps a mate score on a PV line
            // from being masked behind a qsearch value.
            if !NT::PV
                && eval < alpha - params::RAZOR_BASE() - params::RAZOR_DEPTH_MULT() * depth * depth
            {
                self.stats.gate(Gate::Razor);
                return self.qsearch::<NonPv>(tt, stop, alpha, beta, ply);
            }

            // N10 — reverse futility pruning. Gated on `!ss.tt_pv` (subsumes
            // `!PV`), a shallow depth (what keeps mates findable), an eval
            // at/above beta, the ttMove/ttCapture guard, and two decisive-score
            // guards that keep the mate bands out. Returns a beta-ward blend
            // rather than the raw eval, damping static-eval instability.
            if !self.stack[cur].tt_pv
                && depth < params::RFP_MAX_DEPTH()
                && eval >= beta
                && (pruning_tt_move(2) == Move::NULL || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
            {
                // DIS-RFP: scale the margin by the head's predicted
                // |search - static| — high predicted movement means trust the
                // eval less and prune less. `disagreement_cached` is valid only
                // straight after a FRESH eval, so the gate re-derives freshness.
                //
                // DIS-CENSUS: one row per RFP-eligible node — the prediction
                // alongside every input the margin uses — so the head can be
                // regressed against the searched value controlling for them.
                #[cfg(feature = "sota")]
                if dis_census_on() {
                    if let Some(d) = self.eval_ctx.disagreement_cached() {
                        self.dis_census_id[cur] = self.nodes;
                        eprintln!(
                            "DISCENSUS id={} d={} eval={} dis={:.1} imp={} ow={} tt={} corr={} beta={} margin={}",
                            self.nodes,
                            depth,
                            eval,
                            f32::from(d[1]) * 600.0,
                            u8::from(improving),
                            u8::from(opponent_worsening),
                            u8::from(tt_hit),
                            correction_value,
                            beta,
                            // The margin THIS node would have used: with
                            // `ARCHON_RFP_OFF=1` the node is searched anyway, so
                            // `eval - margin >= beta` reconstructs "RFP would
                            // have fired".
                            rfp_margin(
                                depth,
                                tt_hit,
                                improving,
                                opponent_worsening,
                                correction_value,
                            ),
                        );
                        // Binary feature row for off-line retraining; the u8 FT
                        // vector is what DisHead::forward consumes.
                        if let Some(ft) = self.eval_ctx.last_ft_slice() {
                            dis_dump_write(
                                self.nodes,
                                depth,
                                eval,
                                beta,
                                rfp_margin(
                                    depth,
                                    tt_hit,
                                    improving,
                                    opponent_worsening,
                                    correction_value,
                                ),
                                improving,
                                opponent_worsening,
                                tt_hit,
                                correction_value,
                                ft,
                            );
                        }
                    }
                }
                #[cfg_attr(not(feature = "sota"), allow(unused_mut))]
                let mut margin = rfp_margin(
                    depth,
                    tt_hit,
                    improving,
                    opponent_worsening,
                    correction_value,
                );
                #[cfg(feature = "sota")]
                {
                    let c1 = dis_rfp_c1();
                    if c1 != 0.0 && !(tt_hit && is_valid(tt_data.eval)) {
                        if let Some(d) = self.eval_ctx.disagreement_cached() {
                            // col 1 = magnitude in `network_scale` units, so
                            // 600x is internal cp. Clamped both ends because the
                            // head is an unbounded regressor; `max(0.0)` drops
                            // the rare negative outputs (prediction error).
                            let mag_cp = (f32::from(d[1]) * 600.0).max(0.0);
                            let scale = (1.0 + c1 * (mag_cp / 100.0)).clamp(0.7, 1.5);
                            // ONLY SCALE A POSITIVE MARGIN: `rfp_margin` can be
                            // negative, and scaling that INVERTS the arm — a
                            // "wider" margin would then prune MORE.
                            if margin > 0 {
                                margin = (margin as f32 * scale) as Value;
                            }
                        }
                    }
                }
                if eval - margin >= beta && !rfp_off() {
                    self.stats.gate(Gate::Rfp);
                    // One degree of freedom: the eval weight is the beta
                    // weight's complement to 1024.
                    return (params::RFP_RET_BETA_W() * beta
                        + (1024 - params::RFP_RET_BETA_W()) * eval)
                        / 1024;
                }
            }

            // N11 — null move pruning with verification (cutNode-only). A cut
            // node whose *static* eval clears a beta-relative bar searches a
            // null move at `depth - R` and, on a fail high, returns without
            // searching a real move. The gate reads `ss.static_eval`, NOT the
            // TT-substituted `eval` RFP uses. Zugzwang is fenced three ways:
            // non-pawn material required, an unproven mate never trusted, and
            // at high depth a verification search that disables NMP up to
            // `nmp_min_ply`.
            let us = self.board.side_to_move();
            // Diagnostic A/B: ARCHON_NMP_ALLNODE=1 gates NMP on !PvNode instead
            // of cutNode, for measurement only.
            let nmp_node_ok = if self.nmp_allnode { !NT::PV } else { cut_node };
            if nmp_node_ok
                && self.stack[cur].static_eval
                    >= beta
                        - params::NMP_EVAL_DEPTH() * depth
                        - params::NMP_EVAL_IMPROVING() * Value::from(improving)
                        + params::NMP_EVAL_BASE()
                && self.stack[cur].excluded_move == Move::NULL
                && self.board.non_pawn_material(us) != 0
                && ply as i32 >= self.nmp_min_ply
                && !is_loss(beta)
            {
                // A null move's child is searched as a non-cut node, so NMP
                // cannot fire two plies deep off a single pass.
                debug_assert!(
                    self.stack[cur - 1].current_move != Move::NULL,
                    "no two null moves in a row"
                );
                let r = params::NMP_R_BASE() + depth / params::NMP_R_DEPTH_DIV();

                // Made by hand, not via `do_move`: a pass searches no node, so
                // it must NOT bump the node counter, and it stamps the markers
                // the child reads as its parent — `Move::NULL` (⇒ no `prev_sq`)
                // and the inert continuation-history SENTINEL.
                self.stack[cur].current_move = Move::NULL;
                self.stack[cur].cont_hist = ContHistId::SENTINEL;
                // NNUE: a null move leaves every piece in place, so the child
                // accumulator IS the parent's — push no entry and let the child
                // index the same one. No-op when NNUE is off.
                self.eval_ctx.push_null();
                self.stats.nmp_attempt();
                self.board.make_null_move();
                let null_value =
                    -self.search::<NonPv>(tt, stop, -beta, -beta + 1, depth - r, false, ply + 1);
                self.board.unmake_null_move();
                self.eval_ctx.pop_null();

                // A null-move "mate" is an artifact of the illegal pass.
                if null_value >= beta && !is_win(null_value) {
                    self.stats.nmp_cutoff();
                    // Below the verification depth, or already inside one
                    // (`nmp_min_ply != 0`), trust the null value.
                    if self.nmp_min_ply != 0 || depth < params::NMP_VERIFY_DEPTH() {
                        #[cfg(feature = "sota")]
                        self.dis_census_result(cur, null_value, 0, 0);
                        return null_value;
                    }
                    // Verification at THIS node (same ply, real moves) with NMP
                    // disabled up to `nmp_min_ply`; only a fail-high here
                    // confirms the cutoff. The disable window spans the whole
                    // subtree, so it cannot null-move back to the same cutoff.
                    self.stats.gate(Gate::NmpVerify);
                    self.nmp_min_ply = ply as i32
                        + params::NMP_VERIFY_NUM() * (depth - r) / params::NMP_VERIFY_DEN();
                    let v = self.search::<NonPv>(tt, stop, beta - 1, beta, depth - r, false, ply);
                    self.nmp_min_ply = 0;
                    if v >= beta {
                        #[cfg(feature = "sota")]
                        self.dis_census_result(cur, null_value, 0, 0);
                        return null_value;
                    }
                }
            }

            // N11b — from here on "improving" also means "eval already beats
            // beta". Feeds ProbCut and the move-loop LMP/reduction.
            improving |= self.stack[cur].static_eval >= beta;

            // N12 — internal iterative reduction: a PV or cut node with no
            // ttMove searches one ply shallower rather than paying full price
            // for a tree the TT cannot seed.
            if (NT::PV || cut_node)
                && depth >= params::IIR_MIN_DEPTH()
                && pruning_tt_move(1) == Move::NULL
            {
                depth -= params::IIR_REDUCTION();
                self.stats.gate(Gate::Iir);
            }

            // N13 — ProbCut: a cheap search proving a capture beats a raised
            // beta fails the node high early. Skipped inside a singular
            // verification; a stored value below the threshold says "don't".
            let probcut_beta = beta + params::PROBCUT_BETA_BASE()
                - params::PROBCUT_BETA_IMPROVING() * Value::from(improving);
            // The `excluded_move == NULL` fence is a de facto no-op: inside a
            // verification the `tt_value < probcut_beta` clause already blocks
            // the block in both engines.
            if depth >= params::PROBCUT_MIN_DEPTH()
                && !is_decisive(beta)
                && self.stack[cur].excluded_move == Move::NULL
                && !(is_valid(tt_value) && tt_value < probcut_beta)
            {
                let probcut_depth = depth
                    - params::PROBCUT_DEPTH_SUB()
                    - params::PROBCUT_IMPROVING_SUB() * Depth::from(improving);
                let threshold = probcut_beta - self.stack[cur].static_eval;
                let mut mp = MovePicker::new_probcut(&self.board, tt_data.mv, threshold);
                while let Some(m) = mp.next_move(&self.board, &self.histories) {
                    if m == self.stack[cur].excluded_move || !self.board.legal(m) {
                        continue;
                    }
                    let gives_check = self.board.gives_check(m);
                    // ProbCut only emits captures and queen promotions, but ask
                    // `capture_stage` anyway: assuming it would be a silent
                    // continuation-history mis-key if the picker ever changed.
                    let capture = self.board.capture_stage(m);
                    let moved_piece = self.board.piece_on(m.from());
                    self.stats.gate(Gate::ProbcutMove);
                    self.do_move(tt, m, gives_check, capture, moved_piece, ply);
                    // Cheap qsearch verification first, then a reduced search.
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
                            self.stats.gate(Gate::ProbcutCut);
                            #[cfg(feature = "sota")]
                            self.dis_census_result(cur, value - (probcut_beta - beta), 0, 0);
                            return value - (probcut_beta - beta);
                        }
                    }
                }
            }
        }

        // N14 — small in-check ProbCut: sits after the in-check jump target, so
        // it runs in check too. A deep stored LOWER bound already above a large
        // raised beta fails the node high without entering the move loop.
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
            self.stats.gate(Gate::CheckCut);
            #[cfg(feature = "sota")]
            self.dis_census_result(cur, check_probcut_beta, 0, 0);
            return check_probcut_beta;
        }

        // Step 9 — move loop. Interior nodes draw from the staged MovePicker:
        // pseudo-legal moves best-first, legality checked per move. At the root
        // the loop iterates `root_moves[pv_idx..]` instead — everything there is
        // legal by construction and already sorted, so the picker sits inert.
        let mut root_list = MoveList::new();
        if NT::ROOT {
            for rm in &self.root_moves[self.pv_idx..] {
                root_list.push(rm.pv[0]);
            }
        }
        // The six predecessor continuation-history keys; in-bounds because
        // `cur >= SS_OFFSET`.
        let cont_hist = [
            self.stack[cur - 1].cont_hist,
            self.stack[cur - 2].cont_hist,
            self.stack[cur - 3].cont_hist,
            self.stack[cur - 4].cont_hist,
            self.stack[cur - 5].cont_hist,
            self.stack[cur - 6].cont_hist,
        ];
        // arm70 `hidett`: order the node as if the TT offered nothing while
        // every pruning gate still reads `tt_data.mv` (the ordering-value null).
        #[cfg(not(feature = "arm70"))]
        let picker_tt_move = tt_data.mv;
        #[cfg(feature = "arm70")]
        let picker_tt_move = if crate::arm70::arm().hide_tt {
            Move::NULL
        } else {
            tt_data.mv
        };
        // `MovePicker::set_quiet_prior` is the seam a policy head would use.
        let mut mp = MovePicker::new(&self.board, picker_tt_move, depth, ply, cont_hist);

        // census70: node kind, and whether the picker got a usable TT move.
        #[cfg(feature = "census70")]
        let c70_kind = match (NT::ROOT, NT::PV, in_check) {
            (true, _, _) => crate::census70::KIND_ROOT,
            (_, true, false) => crate::census70::KIND_PV,
            (_, true, true) => crate::census70::KIND_PV_CHK,
            (_, false, false) => crate::census70::KIND_NONPV,
            (_, false, true) => crate::census70::KIND_NONPV_CHK,
        };
        // Asked of the TT, not the picker's stage, so the `hidett` null still
        // splits the census by the population it nulls.
        #[cfg(feature = "census70")]
        let c70_had_tt = tt_data.mv != Move::NULL && self.board.pseudo_legal(tt_data.mv);
        #[cfg(feature = "census70")]
        crate::census70::node_entry(
            c70_kind,
            depth,
            c70_had_tt,
            {
                // Which of the three TT-less failure kinds this was.
                if !tt_hit {
                    crate::census70::WHY_MISS
                } else if tt_data.mv == Move::NULL {
                    crate::census70::WHY_MOVELESS
                } else {
                    crate::census70::WHY_UNUSABLE
                }
            },
            tt_data.depth,
        );
        // Index of the move that is currently `best_move` (census).
        #[cfg(feature = "census70")]
        let mut c70_best_idx = 0i32;
        // Capture-stage moves emitted so far; running rank among the quiets.
        #[cfg(feature = "census70")]
        let (mut c70_caps_before, mut c70_quiet_rank) = (0i32, 0i32);

        let mut best_value = -VALUE_INFINITE;
        let mut best_move = Move::NULL;
        // The loop's argmax, tracked even when it never raises alpha, so a
        // fail-low node can leave the TT an ordering hint. Folds away entirely
        // at the default `TT_FAILLOW_MOVE == 0`.
        let mut fl_move = Move::NULL;
        let mut move_count = 0;
        // Tried non-best moves, by kind, for the node-result history burst.
        // Fixed 32-slot buffers (no heap on the hot path); anything past 32 is
        // simply not recorded.
        let mut quiets_searched = [Move::NULL; 32];
        let mut captures_searched = [Move::NULL; 32];
        let mut n_quiets = 0usize;
        let mut n_captures = 0usize;

        // In-search observation capture: decide ONCE, at move-loop entry,
        // whether this node is captured. `alpha`/`beta`/`depth` are snapshotted
        // here because all three mutate during the loop and the recorded context
        // is the ENTRY state. Observation-only; never changes a search decision.
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
            // Skip the singular-exclusion move: the picker still emits the
            // ttMove first during an exclusion search, and searching it would
            // defeat the singular test. If it is the only legal move the loop
            // ends with `move_count == 0`, taking the §3.10 fail-low.
            if m == self.stack[cur].excluded_move {
                continue;
            }
            // Interior nodes emit pseudo-legal moves; skip illegal ones before
            // they count toward move_count (mate detection).
            if !NT::ROOT && !self.board.legal(m) {
                continue;
            }
            let is_capture = self.board.capture_stage(m);
            // Census: which stage put the cutting move late.
            #[cfg(feature = "census70")]
            {
                if is_capture {
                    c70_caps_before += 1;
                } else {
                    c70_quiet_rank += 1;
                }
            }
            // The mover, resolved once: every path through the body needs it.
            let moved_piece = self
                .board
                .piece_on(m.from())
                .expect("a legal move starts on a piece of the side to move");
            move_count += 1;
            self.stack[cur].move_count = move_count;
            let nodes_before = self.nodes; // root effort accounting (§3.4)

            // Pre-move facts the pruning block and the PVS structure share:
            // gives-check, the child depth (`mut` — §3.8 adjusts it), and the
            // base reduction `r`. `r` is finalized after the move is made; the
            // pruning block reads only its BASE value for `lmr_depth`.
            let gives_check = self.board.gives_check_by(m, moved_piece);
            let mut new_depth = depth - 1;
            let delta = beta - alpha;
            let mut r = self.reduction(improving, depth, move_count, delta);
            // §3.5 ttPv entry bump: provisional, mostly unwound by the §3.7
            // rebate. Its lasting effect is on the pruning block below.
            if self.stack[cur].tt_pv {
                r += params::RED_TTPV_ENTRY();
            }
            #[cfg(feature = "search-control")]
            let trace_node_depth = depth;
            #[cfg(feature = "search-control")]
            let trace_base_reduction = r;

            // §3.5 — shallow-depth pruning, decided BEFORE the move is made.
            // Fenced by `!is_loss(best_value)` (never prune while we might be
            // getting mated), never at the root, and never when the side to move
            // has only pawns left (the zugzwang/stalemate-trap guard shared with
            // NMP: sacrificing into a bare-pawn ending must stay searched).
            if !NT::ROOT
                && !is_loss(best_value)
                && self.board.non_pawn_material(self.board.side_to_move()) != 0
            {
                // Late move pruning: past the threshold the picker stops
                // emitting quiets. The ttMove and every capture still come, and
                // in check the evasion pipeline ignores the skip, so no response
                // to check is ever pruned.
                if move_count >= lmp_move_count(depth, improving) {
                    self.stats.gate(Gate::LmpSkip);
                    mp.skip_quiet_moves();
                }

                // The predicted post-LMR depth: the scout depth this move WOULD
                // get, and the depth input every margin below takes. The BASE
                // `r` is the right one here, and it can go negative.
                let mut lmr_depth = new_depth - r / params::RED_UNIT();

                // Captures and checking moves take their own futility/SEE gates;
                // quiets take parent futility then negative-SEE pruning.
                if is_capture || gives_check {
                    // `capt_hist = capture[moved][to][captured]` (the NO_VICTIM
                    // slot for en passant and quiet queen promotions) feeds both
                    // capture gates.
                    let capt_hist = self.histories.capture_value(&self.board, m);

                    // Capture futility: a non-checking capture whose victim plus
                    // depth-scaled slack and history nudge still leaves the eval
                    // below alpha is skipped. `piece_value(captured)` is 0 for
                    // en passant. No `!in_check` guard — the outer `!is_loss`
                    // fence already keeps every evasion searched.
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
                            self.stats.gate(Gate::CfutPrune);
                            continue;
                        }
                    }

                    // Capture/check SEE pruning: a move that cannot win back
                    // `-margin` statically is pruned — UNLESS `alpha <
                    // VALUE_DRAW` and the mover is our LAST non-pawn material,
                    // which may be a stalemate resource.
                    let us = self.board.side_to_move();
                    let moved_value = moved_piece.piece_type.value();
                    if (alpha >= VALUE_DRAW || self.board.non_pawn_material(us) != moved_value)
                        && !self.board.see_ge(m, -capture_see_margin(depth, capt_hist))
                    {
                        self.stats.gate(Gate::CseePrune);
                        continue;
                    }
                } else {
                    // Continuation-history pruning: the FIRST quiet gate, ahead
                    // of parent futility and quiet SEE. No `!in_check` guard —
                    // the outer `!is_loss` fence covers evasions. Fresh
                    // histories sit above the bar at every depth, so the gate is
                    // dormant until updates drive a quiet below it.
                    let pci = moved_piece.index();
                    let to = m.to().idx();
                    let cont_history =
                        self.histories.cont_ref(self.stack[cur - 1].cont_hist)[pci][to].get()
                            + self.histories.cont_ref(self.stack[cur - 2].cont_hist)[pci][to].get()
                            + self.histories.pawn_slot(self.board.pawn_key())[pci][to].get();
                    if continuation_history_prunes(cont_history, depth) {
                        self.stats.gate(Gate::ChistPrune);
                        continue;
                    }

                    // History buys lmr_depth: a quiet the combined history likes
                    // earns predicted depth, sparing it from the futility and
                    // SEE gates below; one it distrusts is pruned earlier. The
                    // product is formed before the divide (SF parity).
                    let history = cont_history
                        + params::LMRDEPTH_MAIN_NUM()
                            * self.histories.main[self.board.side_to_move() as usize]
                                [m.raw16() as usize]
                                .get()
                            / params::LMRDEPTH_MAIN_DEN();
                    // `clamp(1, 16) - 1` is a provable 0..=15, so LLVM folds the
                    // bounds check away (`min(16) - 1` would not).
                    let divisor = params::LMR_DIVISOR[(depth.clamp(1, 16) - 1) as usize];
                    lmr_depth += history / divisor;

                    // Parent futility: the most optimistic value this quiet
                    // could yield cannot reach alpha. Lift `best_value` to that
                    // bound first (guarded so no decisive score is corrupted),
                    // then skip. `!in_check` keeps evasions searched.
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
                        self.stats.gate(Gate::PfutPrune);
                        continue;
                    }

                    // Quiet SEE pruning: a quiet losing more than
                    // `-QSEE_MULT·lmr_depth²` statically is pruned; `lmr_depth`
                    // is floored at 0, so shallow nodes drop any losing quiet.
                    lmr_depth = lmr_depth.max(0);
                    if !self
                        .board
                        .see_ge(m, -params::QSEE_MULT() * lmr_depth * lmr_depth)
                    {
                        self.stats.gate(Gate::QseePrune);
                        continue;
                    }
                }
            }

            // §3.6 — singular extensions and multicut. Only the ttMove is a
            // candidate: an exclusion search below the stored value, at the SAME
            // ply, decides whether it is singularly best (extend the node) or
            // one of several beta-beaters (multicut). The trigger disables
            // itself inside an exclusion search, so it never recurses.
            // `is_shuffling` suppresses extensions of 50-move shuffles.
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
                #[cfg(feature = "census70")]
                let c70_sing_before = self.nodes;
                self.stats.gate(Gate::SeTry);
                let value = self.search::<NonPv>(
                    tt,
                    stop,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                    ply,
                );
                #[cfg(feature = "census70")]
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    crate::census70::SING_TRY.fetch_add(1, Relaxed);
                    crate::census70::SING_NODES.fetch_add(self.nodes - c70_sing_before, Relaxed);
                }
                self.stack[cur].excluded_move = Move::NULL;

                if value < singular_beta {
                    self.stats.gate(Gate::SeSingular);
                    // Singular: extend. `corr_adj` tightens the double/triple
                    // margins under eval uncertainty.
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
                    if value < singular_beta - double_margin {
                        self.stats.gate(Gate::SeDouble);
                    }
                    if value < singular_beta - triple_margin {
                        self.stats.gate(Gate::SeTriple);
                    }
                    // The WHOLE node deepens, not just the ttMove.
                    depth += params::SE_NODE_DEEPEN();
                } else if value >= beta && !is_decisive(value) {
                    self.stats.gate(Gate::SeMulticut);
                    // Multicut: several moves beat beta without the ttMove, so
                    // the node is not singular — prune the subtree.
                    self.histories
                        .tt_move
                        .update(params::MULTICUT_TMH_BASE() + params::MULTICUT_TMH_DEPTH() * depth);
                    return value;
                } else if tt_value >= beta {
                    // The ttMove will fail high anyway — search it shallower.
                    extension = params::SE_NEG_EXT_TT_FAILHIGH();
                    self.stats.gate(Gate::SeNegExt);
                } else if cut_node {
                    extension = params::SE_NEG_EXT_CUTNODE();
                    self.stats.gate(Gate::SeNegExt);
                }
            }
            new_depth += extension;
            self.stats.extension(extension);

            // §3.7 — `ss.stat_score`, computed from PRE-move facts, so it must
            // precede `do_move`. On the stack because the finalization docks `r`
            // by it AND the child reads `(ss-1).stat_score`.
            self.stack[cur].stat_score = self.stat_score(m, is_capture, moved_piece, cur);

            // Observational only, before make-move so the context describes the
            // parent. `trace_selected` must stay independent of `trace_latent`:
            // tying the row to the usually-absent policy latent made the trace
            // silently record zero rows while looking armed.
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

            // Step 10 — make + recurse: the PVS three-call structure. Only the
            // move ordered first gets a full window up front; every later move
            // must beat a zero-width window before earning a PV re-search.
            self.do_move(tt, m, gives_check, is_capture, Some(moved_piece), ply);
            // §3.7 — reduction finalization. The ttPv rebate and the surcharges
            // land TOGETHER by design: the rebate is the counterweight keeping
            // PV-history (mate-bearing) lines searched full. It comes first,
            // unwinding the §3.5 entry bump and more. The tt.value term demands
            // a REAL value — VALUE_NONE would spuriously pass `> alpha`.
            if self.stack[cur].tt_pv {
                // Deliberate SF divergence: SF grants the ttValue term at every
                // PV-flagged TT-miss node; that parity measured worse here, so
                // the `is_valid` guard withholds it on a miss.
                r -= params::RED_TTPV_BASE()
                    + params::RED_TTPV_PV() * i32::from(NT::PV)
                    + params::RED_TTPV_TTVALUE()
                        * i32::from(is_valid(tt_value) && tt_value > alpha)
                    + i32::from(tt_data.depth >= depth)
                        * (params::RED_TTPV_TTDEPTH_BASE()
                            + params::RED_TTPV_TTDEPTH_CUT() * i32::from(cut_node));
            }
            r += params::RED_FLAT();
            r -= params::RED_MOVECOUNT() * move_count;
            // Correction slackening: an uncertain eval reduces a touch LESS.
            r -= correction_value.abs() / params::RED_CORR_DIV();
            if cut_node {
                r += params::RED_CUTNODE()
                    + params::RED_CUTNODE_NO_TTMOVE() * i32::from(pruning_tt_move(4) == Move::NULL);
            }
            // A capturing ttMove means a tactical position, so quiet siblings
            // are searched shallower still.
            if tt_capture {
                r += params::RED_TTCAPTURE();
            }
            // cutoffCnt surcharge: the child slot has been failing high under
            // earlier siblings, so late moves here are distrusted. The ttMove
            // rebate is its ELSE arm — the two are mutually exclusive by
            // design; both firing would double-adjust `r`.
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
            // History docks the reduction: a liked move searches fuller.
            // Truncation toward zero on a negative `stat_score` is intended.
            r -= self.stack[cur].stat_score * params::RED_STATSCORE_NUM()
                / params::RED_STATSCORE_DEN();
            // All-node scaling, last in the list: a node where neither bound is
            // expected to move reduces proportionally harder, tapering.
            if all_node {
                r += r * params::RED_ALLNODE_NUM()
                    / (params::RED_ALLNODE_DEPTH_MULT() * depth + params::RED_ALLNODE_DEPTH_BASE());
            }
            // DIS-LMR: reduce LESS where the head predicts a real search would
            // move the eval a lot. The one channel touching NEITHER the eval NOR
            // a margin — those are a jointly calibrated system and every arm
            // perturbing them lost. Normalised by 100 cp; a ply is 1024.
            #[cfg(feature = "sota")]
            {
                let cl = dis_lmr_c();
                if cl != 0.0 && dis_mag_cp > 0.0 {
                    r -= (cl * (dis_mag_cp / 100.0)) as i32;
                }
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
            // Overwritten by whichever call below runs last; a PV node's first
            // move skips (a)/(b) and (c) never reads the sentinel.
            let mut value = -VALUE_INFINITE;
            // Did this move take an LMR full-depth zero-window re-search?
            #[cfg(feature = "datagen")]
            let mut obs_researched = false;

            // (a) Reduced zero-window scout — LMR proper (design/10 §3.8). The
            // net reduction (1024ths → plies) shrinks `new_depth` to the scout
            // depth `d`, clamped to `[1, new_depth + RED_MAX_DEEPER]` and lifted
            // a ply at PV nodes; a negative net `r` may search DEEPER than
            // `new_depth`. LMR scouts always start as cut nodes. `ss.reduction`
            // records the ply delta for the child's hindsight read and is
            // cleared after (the child also clears it at its own entry).
            if depth >= params::LMR_MIN_DEPTH() && move_count > params::LMR_MIN_MOVECOUNT() {
                let classical_d = (new_depth - r / params::RED_UNIT())
                    .min(new_depth + params::RED_MAX_DEEPER())
                    .max(params::LMR_MIN_SCOUT_DEPTH())
                    + params::LMR_PV_DEPTH_BONUS() * Depth::from(NT::PV);
                self.stats.lmr(new_depth - classical_d);
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
                // Both halves are node-accounted: the split is the price of
                // putting a good move late.
                #[cfg(feature = "census70")]
                let c70_scout_before = self.nodes;
                #[cfg(feature = "census70")]
                let mut c70_research_nodes = 0u64;
                #[cfg(feature = "census70")]
                let mut c70_researched = false;
                #[cfg(feature = "sota")]
                {
                    census_lmr_red += u32::from(d < new_depth);
                }
                value = -self.search::<NonPv>(tt, stop, -(alpha + 1), -alpha, d, true, ply + 1);
                #[cfg(feature = "census70")]
                let c70_scout_nodes = self.nodes - c70_scout_before;
                #[cfg(feature = "search-control")]
                {
                    trace_scout_depth = d;
                    trace_scout_score = value;
                    trace_scout_nodes = self.nodes - trace_nodes_before_scout;
                }
                self.stack[cur].reduction = 0;

                // (b) The scout beat alpha: apply the deeper/shallower rule to
                // `new_depth`, then re-search full-depth zero-window with the
                // flipped cut-node discipline whenever the adjusted depth still
                // exceeds `d`. The deeper arm is gated on `d < new_depth`; the
                // shallower arm may still fire on a deepened scout. Either way
                // the move earns the flat post-LMR continuation-history bonus.
                if value > alpha {
                    let do_deeper =
                        d < new_depth && value > best_value + params::LMR_DEEPER_MARGIN();
                    let do_shallower = value < best_value + params::LMR_SHALLOWER_MARGIN();
                    new_depth += params::LMR_DEEPER_STEP() * Depth::from(do_deeper)
                        - params::LMR_SHALLOWER_STEP() * Depth::from(do_shallower);
                    if new_depth > d {
                        #[cfg(feature = "sota")]
                        {
                            census_lmr_rer += 1;
                        }
                        #[cfg(feature = "search-control")]
                        let trace_nodes_before_research = self.nodes;
                        #[cfg(feature = "census70")]
                        let c70_re_before = self.nodes;
                        value = -self.search::<NonPv>(
                            tt,
                            stop,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                            ply + 1,
                        );
                        #[cfg(feature = "census70")]
                        {
                            c70_researched = true;
                            c70_research_nodes = self.nodes - c70_re_before;
                        }
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
                #[cfg(feature = "census70")]
                crate::census70::lmr(
                    c70_kind,
                    move_count,
                    c70_researched,
                    c70_scout_nodes,
                    c70_research_nodes,
                );
            } else if !NT::PV || move_count > 1 {
                // Non-LMR zero-window path: depth 1, or a NonPV node's first
                // move (where `(-(alpha+1), -alpha)` IS its own window). No TT
                // move reduces a little more, and a very high nominal `r` still
                // docks 1–2 plies even without LMR proper.
                if pruning_tt_move(8) == Move::NULL {
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
                #[cfg(feature = "census70")]
                let c70_zw_before = self.nodes;
                value =
                    -self.search::<NonPv>(tt, stop, -(alpha + 1), -alpha, d, !cut_node, ply + 1);
                #[cfg(feature = "census70")]
                crate::census70::ZW_NODES[c70_kind].fetch_add(
                    self.nodes - c70_zw_before,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            // ttMove min-depth clamp: a negative extension must never dive the
            // PV's ttMove into qsearch when the TT entry is deep or decisive.
            if m == tt_data.mv
                && ((is_valid(tt_value) && is_decisive(tt_value) && tt_data.depth > 0)
                    || tt_data.depth > 1)
            {
                new_depth = new_depth.max(1);
            }

            // (c) PV re-search, full window: PV/Root nodes only, for the first
            // move and any scout that beat alpha. PV children are never cut
            // nodes, and only THIS call writes `pv_table[ply + 1]` — §3.3's
            // freshness invariant, so a step-13 PV update can never stitch a
            // child row a zero-window scout left stale.
            if NT::PV && (move_count == 1 || value > alpha) {
                #[cfg(feature = "search-control")]
                let trace_nodes_before_pv = self.nodes;
                #[cfg(feature = "census70")]
                let c70_pv_before = self.nodes;
                value = -self.search::<Pv>(tt, stop, -beta, -alpha, new_depth, false, ply + 1);
                #[cfg(feature = "census70")]
                crate::census70::PVRE_NODES[c70_kind].fetch_add(
                    self.nodes - c70_pv_before,
                    std::sync::atomic::Ordering::Relaxed,
                );
                #[cfg(feature = "search-control")]
                {
                    trace_pv_research = true;
                    trace_pv_research_nodes = self.nodes - trace_nodes_before_pv;
                }
            }
            self.undo_move(m);

            // One Layer-4 row per move actually searched. `alpha`/`beta` are
            // still what this move was searched against (step 13 raises alpha
            // below); `r` is the finalized reduction in 1024ths. Read-only.
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

            // Step 13 — best/alpha/beta updates.
            if value > best_value {
                best_value = value;
                if fl_mode != 0 {
                    fl_move = m;
                }
                if value > alpha {
                    best_move = m;
                    #[cfg(feature = "census70")]
                    {
                        c70_best_idx = move_count;
                    }
                    if NT::PV {
                        // §3.3 freshness holds by construction: promotion
                        // needs a strict `value > alpha`, which under NT::PV
                        // implies the full-window (c) call just ran and wrote
                        // `pv_table[ply + 1]`.
                        update_pv(&mut self.pv_table[..], ply, m);
                    }
                    if value >= beta {
                        self.stats.fail_high(move_count);
                        // Census: the same event split by node kind, depth,
                        // and the picker stage that emitted the cutting move.
                        #[cfg(feature = "census70")]
                        crate::census70::cutoff(
                            c70_kind,
                            depth,
                            move_count,
                            mp.stage_id(),
                            c70_had_tt,
                            c70_caps_before,
                            if is_capture { 0 } else { c70_quiet_rank },
                        );
                        // A cutoff by a heavily-extended move is weaker evidence
                        // of a fail-high-prone child, so it does not bump the
                        // count (design/10 §3.9, Wave-D item 15).
                        self.stack[cur].cutoff_cnt +=
                            i32::from(extension < params::CUTOFFCNT_EXT_GATE() || NT::PV);
                        break; // fail high
                    }
                    // Sibling depth reduction (design/10 §3.9): an alpha raise
                    // without a cutoff says the node is already well-resolved
                    // — remaining siblings get a cheaper search. Decisive
                    // scores are exempt (mate distances stay exact). Root is
                    // exempt (TEMPO deviation): the ID driver owns root
                    // depth, and mutating it mid-iteration corrupts the
                    // aspiration invariants. NB this makes non-root
                    // fixed-depth values order-dependent.
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

        #[cfg(feature = "census70")]
        if best_value < beta {
            crate::census70::allnode(c70_kind, depth, move_count, c70_best_idx, c70_had_tt);
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
                self.stats.gate(Gate::FlQuiet);
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
                        self.histories.pawn_slot_mut(pawn_key)[pc.index()][sq.idx()]
                            .update(scaled_bonus * params::FL_PAWN_NUM() / params::FL_PAWN_DEN());
                    }
                }
            } else if let Some(moved) = self.board.piece_on(sq) {
                self.stats.gate(Gate::FlCapture);
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
            // Perf study 71: on a fail low `best_move` is NULL and the move
            // that actually scored highest is thrown away. Store it. The
            // BOUND is untouched — this is an ordering hint attached to an
            // UPPER entry, not a claim that the move is good.
            let store_move = if best_move != Move::NULL {
                best_move
            } else {
                fl_move
            };
            probe.writer.write(
                key,
                value_to_tt(best_value, ply as i32),
                self.stack[cur].tt_pv,
                bound,
                store_depth,
                store_move,
                raw_eval,
                generation,
            );
            // DIS-CENSUS truth for the normal path.
            #[cfg(feature = "sota")]
            self.dis_census_result(cur, best_value, census_lmr_red, census_lmr_rer);
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
            self.stats.gate(Gate::CorrUpd);
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
    /// divide by 1024 only at the moment of application. Combines the log
    /// table [`Worker::reductions`] (indexed by `depth` and by `move_number`)
    /// with the aspiration window width `root_delta` (`beta - alpha` at the
    /// root, never zero): a wide window reduces less, a non-improving node
    /// ~38% more.
    ///
    /// Integer division truncates toward zero like the Stockfish source
    /// (D27), so the term order is preserved. The returned `r` is finalized
    /// in the move loop and applied as `new_depth - r / RED_UNIT`; its base
    /// algebra is pinned by golden-value tests.
    #[inline]
    fn reduction(&self, improving: bool, depth: Depth, move_number: i32, delta: i32) -> i32 {
        debug_assert!(
            self.root_delta > 0,
            "root_delta is beta - alpha, never zero"
        );
        // The mask is the identity on every reachable value (`depth` is
        // clamped to MAX_PLY-1 at node entry; `move_number` counts legal
        // moves, max 218) and lets LLVM drop the bounds checks. NOTE: keep it
        // a mask — a `clamp` spelling of the same bound has measured worse.
        const RED_M: usize = crate::moves::MAX_MOVES - 1;
        let scale =
            self.reductions[depth as usize & RED_M] * self.reductions[move_number as usize & RED_M];
        scale - delta * params::RED_DELTA_MULT() / self.root_delta
            + i32::from(!improving) * scale * params::RED_NON_IMPROVING()
                / params::RED_NON_IMPROVING_DEN()
            + params::RED_BASE()
    }

    /// The per-node combined history statistic `ss.stat_score` (design/10
    /// §3.7), read from PRE-move state (so it must be called before
    /// `do_move`). A capture is graded by victim value plus its capture
    /// history (victim value 0 and the NO_VICTIM slot for en passant / quiet
    /// queen promotions). A quiet is graded by its weighted butterfly history
    /// plus the offset-1 and offset-2 continuation histories — the same slabs
    /// move ordering reads. Feeds the `RED_STATSCORE_NUM` reduction dock.
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
            let q = crate::history::piece_to(moved_piece.index(), m.to().idx());
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
    // NOTE: `inline(always)` is deliberate — `search::<NT>` is two out-of-line
    // monomorphizations, so search-body helpers never earn LLVM's
    // last-call-to-static inline bonus and a plain `#[inline]` is DECLINED.
    // Forcing it wins only on SMALL bodies (it inverts into spills on large
    // ones); do not extend to a big function without measuring.
    #[cfg_attr(not(feature = "icold67"), inline(always))]
    #[cfg_attr(feature = "icold67", inline(never))]
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

    /// Quiescence search (design/07 §3.8): stand pat on the corrected static
    /// eval with the tt-value substitution and fail-high softening; captures
    /// only (queen promotions ride the CAPTURES stage), ALL evasions while in
    /// check; the per-move pruning block, every rule gated on
    /// `!is_loss(best_value)` — the evasion guard that keeps checkmates from
    /// being pruned into existence; ply-relative mate when checkmated plus
    /// the narrow stalemate test; and the `DEPTH_QS` TT discipline (a store
    /// at exit that is NEVER `Exact`).
    ///
    /// Stop poll at entry: `check_time` is re-checked here so the hard wall
    /// is observed mid-dive. qsearch itself does not abort — a stop set here
    /// surfaces via the enclosing main-search node's step-11 check.
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

        // Deliberate SF divergence: SF re-runs the upcoming-repetition
        // alpha-raise as the first statement of qsearch too; that was measured
        // worse here. The entry node above the dive still runs it in search().

        // Step 2 — init.
        let cur = ss(ply);
        let in_check = !self.board.checkers().is_empty();
        self.stack[cur].in_check = in_check;
        self.stats.gate(Gate::QsEntry);
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
            self.stats.gate(Gate::QsTtcut);
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
            let tt_served = tt_hit && is_valid(tt_data.eval);
            raw_eval = if tt_served {
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
                qtrace(ply, alpha, beta, best_value, Move::NULL, "STANDPAT_CUT");
                // Search-instability damping (§3.8 step 5): pull a
                // non-decisive stand-pat cutoff partway back toward beta.
                // Decisive scores are never blended (§8.11).
                if !is_decisive(best_value) {
                    // One degree of freedom (D50 blend fold): the best-value
                    // weight is the beta weight's complement to 1024.
                    best_value = ((1024 - params::QS_STANDPAT_BETA_W()) * best_value
                        + params::QS_STANDPAT_BETA_W() * beta)
                        / 1024;
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
                self.stats.gate(Gate::QsStandpat);
                return best_value;
            }
            alpha = alpha.max(best_value);
            // The bar every capture's victim must clear (§3.8 step 5):
            // from the CORRECTED static eval, not the substituted
            // best_value.
            futility_base = self.stack[cur].static_eval + params::QS_FUTILITY_MARGIN();
        }

        // Steps 6–7 — the staged MovePicker (doc 09): the qsearch pipeline
        // (TT move → captures + queen promotions) out of check, the evasion
        // pipeline in check. Both emit pseudo-legal moves; legality is tested
        // per move. A quiet TT move can be emitted, but the step-7
        // non-capture skip drops it once best_value is real.
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

        qtrace(ply, alpha, beta, best_value, Move::NULL, "LOOP_ENTERED");
        #[cfg(feature = "census70")]
        let c70_kind = if in_check {
            crate::census70::KIND_QS_CHK
        } else {
            crate::census70::KIND_QS
        };
        #[cfg(feature = "census70")]
        let c70_had_tt = matches!(mp.stage_id(), 7 | 13);
        #[cfg(feature = "census70")]
        crate::census70::node_entry(
            c70_kind,
            0,
            c70_had_tt,
            {
                if !tt_hit {
                    crate::census70::WHY_MISS
                } else if tt_data.mv == Move::NULL {
                    crate::census70::WHY_MOVELESS
                } else {
                    crate::census70::WHY_UNUSABLE
                }
            },
            tt_data.depth,
        );
        #[cfg(feature = "census70")]
        let mut c70_best_idx = 0i32;

        let mut best_move = Move::NULL;
        // Fail-low move store, quiescence twin (see `search`).
        #[cfg(not(feature = "arm70"))]
        let qfl_mode = params::TT_FAILLOW_MOVE();
        #[cfg(feature = "arm70")]
        let qfl_mode = match crate::arm70::arm().qfl_move {
            -1 => params::TT_FAILLOW_MOVE(),
            v => v,
        };
        let mut fl_move = Move::NULL;
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
                        qtrace(ply, alpha, beta, best_value, m, "prune:latemove");
                        self.stats.gate(Gate::QsLatemove);
                        continue;
                    }

                    // Futility: even winning the victim for free cannot
                    // lift the eval-based bar above alpha. The pruned
                    // move's optimistic value still raises best_value
                    // (D11 SEE-scale piece values; empty destination —
                    // en passant — counts 0, SF parity).
                    let futility_value = futility_base
                        + self
                            .board
                            .piece_on(m.to())
                            .map_or(0, |victim| victim.piece_type.value());
                    if futility_value <= alpha {
                        qtrace(ply, alpha, beta, best_value, m, "prune:futility");
                        best_value = best_value.max(futility_value);
                        self.stats.gate(Gate::QsFut);
                        continue;
                    }

                    // SEE vs the window: the exchange cannot recover the
                    // gap between futility_base and alpha.
                    if !self.board.see_ge(m, alpha - futility_base) {
                        best_value = best_value.max(alpha.min(futility_base));
                        self.stats.gate(Gate::QsSeeWindow);
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
                    qtrace(ply, alpha, beta, best_value, m, "prune:seefloor");
                    self.stats.gate(Gate::QsSeeFloor);
                    continue;
                }
            }

            // Step 8 — recurse: full window, same node type.
            qtrace(ply, alpha, beta, best_value, m, "SEARCH");
            let capture = self.board.capture_stage(m);
            let moved_piece = self.board.piece_on(m.from());
            self.do_move(tt, m, gives_check, capture, moved_piece, ply);
            self.stats.qnode();
            let value = -self.qsearch::<NT>(tt, stop, -beta, -alpha, ply + 1);
            self.undo_move(m);

            // Step 9 — updates.
            if value > best_value {
                best_value = value;
                if qfl_mode != 0 {
                    fl_move = m;
                }
                if value > alpha {
                    best_move = m;
                    #[cfg(feature = "census70")]
                    {
                        c70_best_idx = move_count;
                    }
                    if NT::PV {
                        update_pv(&mut self.pv_table[..], ply, m);
                    }
                    if value >= beta {
                        #[cfg(feature = "census70")]
                        crate::census70::cutoff(
                            c70_kind,
                            0,
                            move_count,
                            mp.stage_id(),
                            c70_had_tt,
                            0,
                            0,
                        );
                        break; // fail high
                    }
                    alpha = value;
                }
            }
        }
        #[cfg(feature = "census70")]
        if best_value < beta {
            crate::census70::allnode(c70_kind, 0, move_count, c70_best_idx, c70_had_tt);
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
                self.stats.gate(Gate::QsStaleScan);
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
            // One degree of freedom (D50 blend fold), as at the stand-pat site.
            best_value = ((1024 - params::QS_FAILHIGH_BETA_W()) * best_value
                + params::QS_FAILHIGH_BETA_W() * beta)
                / 1024;
        }

        // Step 12 — TT store at DEPTH_QS: LOWER on a fail high, else
        // UPPER — NEVER Exact in qsearch — value through `value_to_tt`,
        // the RAW eval, and the probed PV flag only.
        let store_move = if best_move != Move::NULL {
            best_move
        } else {
            fl_move
        };
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
            store_move,
            raw_eval,
            generation,
        );

        debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);
        best_value
    }

    /// The N6b child-TT verification probe (design/10 item 20): make the
    /// ttMove on the BOARD ONLY, read the child's stored value, unmake.
    ///
    /// This probe is NOT a search node: it must NOT bump `self.nodes` (which
    /// would inflate the bench fingerprint, the `go nodes` budget, and
    /// node-based TM) nor stamp the search stack. The prefetch hook still
    /// rides along. No NNUE accumulator push: the probe never `evaluate`s the
    /// child and unmakes immediately — the next real `do_move` re-pushes.
    /// Kept a function (not a block) so the rarely-run body stays out of the
    /// hot symbol. Returns [`VALUE_NONE`] on a TT miss.
    #[cfg_attr(feature = "icold67", cold)]
    #[cfg_attr(feature = "icold67", inline(never))]
    fn tt_verify_child(&mut self, tt: &TranspositionTable, m: Move, ply: usize) -> Value {
        let gc = self.board.gives_check(m);
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
        child_value
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
    /// padding. `in_check` was set at this node's step-2 entry.
    ///
    /// `capture` and `moved_piece` are the caller's: the main move loop has
    /// already computed both from the same unchanged board, so re-deriving
    /// them here would be pure waste.
    #[cfg_attr(feature = "icold67b", inline(never))]
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
        // can aggregate live counts across threads. Single-writer load+store,
        // relaxed, owned cache line; bench reads `self.nodes`.
        self.counters.nodes.store(self.nodes, Ordering::Relaxed);
        let cur = ss(ply);
        let in_check = self.stack[cur].in_check;
        self.stack[cur].current_move = m;
        self.stack[cur].cont_hist = ContHistId::new(in_check, capture, moved_piece, m.to());

        // The dirty-piece record is built inside the make from facts it
        // already derives, and handed back through the hook struct.
        let mut hooks = TtPrefetchDirty {
            tt,
            dp: DirtyPiece::EMPTY,
        };
        self.board.make_move_with(m, gives_check, &mut hooks);
        // NNUE accumulator maintenance (design/11 §3.4.4): record this move's
        // placement deltas into the child entry at `ply + 1` and mark it
        // uncomputed (the invalidation rule), so a sibling subtree can never
        // reuse a stale accumulator across LMR/PVS re-searches. A no-op when
        // NNUE is off, keeping the PSQT hot path and the depth-6 bench
        // bit-identical — `WANTS_DIRTY` cannot see `nnue_active()`, so this
        // guard is what keeps the classical path free.
        // Neither push reads the pre-move board: `push_dirty` takes no board,
        // and `push_sota_dirty` deliberately wants the POST-move one (its
        // threat x-rays read the child).
        if self.eval_ctx.nnue_active() {
            let dp = hooks.dp;
            self.eval_ctx.push_dirty(dp, ply);
            #[cfg(feature = "sota")]
            self.eval_ctx.push_sota_dirty(dp, ply, &self.board);
        }
    }

    // icold67b arm, same reason as `do_move`.
    #[cfg_attr(feature = "icold67b", inline(never))]
    #[inline]
    fn undo_move(&mut self, m: Move) {
        self.board.unmake_move(m);
    }

    /// The §3.8 time-check seam: called at every main-search node entry AND
    /// at qsearch entry (bounding stop latency under a long forcing-capture
    /// dive). Polled through a bitmask on the node counter; the mask tightens
    /// under `go nodes` so tiny budgets are honored.
    ///
    /// Enforces the hard limits: `movetime`, `nodes`, and the wtime/btime
    /// hard wall — all behind the ponder guard. Two hardening rules: never
    /// hard-stop before iteration 1 finishes (a real bestmove must exist),
    /// and a flat epsilon is shaved off the wall-clock bounds to cover the
    /// unwind-and-print tail (the `nodes` budget stays exact).
    /// The mask test stays inline; the enforcement body is `check_time_slow`.
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
        self.check_time_slow(stop);
    }

    /// The enforcement body of [`Self::check_time`], split off by perf study 67
    /// so the fetch weight leaves the caller. Reached once every
    /// `check_time_mask + 1` nodes; everything above it is one predicted test.
    #[cfg_attr(feature = "icold67", cold)]
    #[cfg_attr(feature = "icold67", inline(never))]
    fn check_time_slow(&self, stop: &AtomicBool) {
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

    /// The between-iterations soft stop (design/12 §3.9): once the soft
    /// budget is spent, another iteration would almost certainly be cut off
    /// by the `maximum` hard wall before completing.
    ///
    /// `total_time = optimum · fallingEval · reduction · bestMoveInstability
    /// · highBestMoveEffort`, capped with exactly one legal move. The search
    /// stops when `elapsed > min(total_time, maximum)` or on the two mate
    /// early-stops; otherwise it sets `increase_depth` for the next
    /// iteration. While pondering the GUI owns the clock, so nothing may
    /// self-stop — the verdict is remembered in `stop_on_ponderhit` instead.
    /// Maintains the per-iteration TM state as a side effect.
    ///
    /// `best_value` is the value the LAST MultiPV line's root search settled
    /// on (`root_moves[0].score` at MultiPV 1, the weakest line otherwise).
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

        // DIS-TM: keep thinking while the disagreement head says the ROOT
        // eval is still unresolved. The stop decision is the one consumer
        // where the head's prediction is redundant with nothing — everywhere
        // else the search computes `search - static` by searching, and
        // per-node arms that fed it into tuned pruning margins have lost.
        #[cfg_attr(not(feature = "sota"), allow(unused_mut))]
        let mut dis_factor = 1.0f64;
        #[cfg_attr(not(feature = "sota"), allow(unused_mut, unused_variables))]
        let mut dis_remaining = 0.0f32;
        #[cfg(feature = "sota")]
        {
            let c = dis_tm_c();
            if c != 0.0 && self.root_dis_mag > 0.0 {
                // REMAINING predicted movement, not the raw prediction: the
                // head's budget minus what the live search has already found.
                // It falls as the search finds the predicted movement and
                // reaches zero once the prediction is spent (a raw-prediction
                // form went silent exactly at stop-decision depths).
                // TWO-SIDED around a pivot, deliberately: the clock is fixed,
                // so the arm can only MOVE time between moves — it must spend
                // above the pivot and save below it, not ask for more
                // everywhere. NOTE: the pivot must be the median of THIS
                // `remaining` quantity, not of the raw dismag — hence the
                // trace column.
                let found = (best_value as f32 - self.root_static_eval).abs();
                let remaining = (self.root_dis_mag - found).max(0.0);
                let excess = remaining - params::DIS_TM_PIVOT() as f32;
                dis_remaining = remaining;
                // Capped: this multiplies a TIME budget, and an unbounded
                // factor on a bad prediction would burn a whole game's clock
                // on one move. `maximum` is still the hard wall above this.
                dis_factor = (1.0 + f64::from(c * excess / 100.0)).clamp(0.7, 1.5);
            }
        }
        let mut total_time =
            self.tm.optimum() as f64 * falling_eval * reduction * instability * effort * dis_factor;
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
                 dis={:.4} dismag={:.0} disrem={:.0} \
                 total={:.0} elapsed={:.0} chg={:.2} \
                 bv={} prevavg={} iterold={} dprev={} diter={} stab={}",
                self.root_depth,
                self.tm.optimum(),
                self.tm.maximum(),
                falling_eval,
                reduction,
                instability,
                effort,
                // DIS-TM census columns: the factor actually applied and the
                // root prediction that produced it, so a run can be
                // summarised without spending a game.
                dis_factor,
                {
                    #[cfg(feature = "sota")]
                    {
                        self.root_dis_mag
                    }
                    #[cfg(not(feature = "sota"))]
                    {
                        0.0f32
                    }
                },
                dis_remaining,
                total_time,
                elapsed,
                tot_best_move_changes,
                // Raw fallingEval inputs, in OUR internal Value units — the
                // coefficients came from SF, whose units differ, so read the
                // deltas directly rather than re-deriving a scale ratio.
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
