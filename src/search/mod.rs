//! Search — iterative deepening driver, PVS nodes, quiescence
//! (`design/07-search-core.md`) and the pruning/extension heuristics
//! (`design/10-pruning-extensions.md`).
//!
//! Landed so far (design/07 §4 steps 1–5): the [`params`] constants module
//! (P0, design/01 §4 step 4), the per-ply search [`stack`] with PV lines
//! (§3.2–§3.3, D18), [`rootmoves`] bookkeeping (§3.4), the naked
//! negamax + qsearch skeleton in this file — [`Worker`], the [`NodeType`]
//! markers (§3.7.1), `search`/`qsearch` with the step-4 node subset:
//! draw/max-ply handling with the [`value_draw`] dither, mate-distance
//! pruning, full-window move loop over `generate_legal` with MVV-LVA
//! placeholder ordering, PV updates, ply-relative mate/stalemate terminals
//! (D5), node counting and the stop-flag seam (§3.9) — and the
//! iterative-deepening driver with root bookkeeping (§3.5, step 5):
//! [`build_root_moves`] with the `searchmoves` filter, per-iteration
//! full-window root searches, previous-score snapshots, lastBestMove
//! tracking, selDepth, `check_time` enforcement of the v1 hard limits
//! (`depth`/`nodes`/`movetime`, plus the wtime/btime hard wall from the
//! worker-owned [`TimeManager`] — doc 12 §3.8's `elapsed > maximum`), and
//! the [`SearchInfo`] callback seam the UCI layer turns into `info` lines
//! (design/12 §3.13).
//!
//! P6 adds the TT integration (design/07 §4 step 6, at the §3.7.2/§3.8
//! marked points, v1 scope): the probe at node entry with `value_from_tt`
//! at the single conversion boundary (§3.12), ttMove-first move ordering
//! (the stable front placement in [`order_moves`]), raw-eval reuse from
//! probed entries plus the §3.11 eval-cache writes at `DEPTH_UNSEARCHED`,
//! the store at node exit with `value_to_tt`/bound selection, the sticky
//! `ss.tt_pv` round-trip, qsearch's `DEPTH_QS` probe/store discipline
//! (never `Exact` there), the base NonPV TT cutoffs with the rule-50
//! guard, and the TT prefetch
//! hook wired into [`Worker::do_move`] through the [`MakeMoveHooks`] seam
//! (doc 04 §3.10 hook A, doc 08 §3.15). The table itself reaches the
//! search as `&TranspositionTable` through the search entry (doc 07
//! §3.10); the session layer holds the `Arc<TT>` handle for the job's
//! duration and bumps `new_search()` once per `go` (D23, doc 13 §3.4/§3.7
//! — the worker deliberately does NOT store the handle, so the idle UCI
//! thread keeps `Arc::get_mut` for resize/clear).
//!
//! P7 converts the move loop to principal variation search (design/07
//! §4 step 7, §3.7.2 step 10): the three-call structure — (a) a reduced
//! zero-window scout for every move after the first (r = 0 until doc
//! 10's LMR lands in P9, so the (b) full-depth zero-window re-search
//! after a reduced fail-high is structurally present but cannot trigger
//! yet), (b) the plain full-depth zero-window search where (a) does not
//! apply, and (c) the full-window PV re-search for the first move and
//! for any scout that beats alpha — with the §3.7.1 cut-node discipline
//! (scouts start as cut nodes, plain zero-window searches flip
//! `!cut_node`, PV re-searches never are) enforced by the node-entry
//! debug_asserts. This is what turns the P6 NonPV TT cutoffs
//! (main-search step 7 and qsearch's `DEPTH_QS` cutoff) into live code,
//! and it leans on the P6 ttMove-first ordering: the move most likely
//! to be best gets the one full-window search, everything else is
//! proven `<= alpha` as cheaply as possible.
//!
//! Step 8 completes the root driver with aspiration windows
//! ([`Worker::aspiration_search`], design/07 §3.6): each iteration
//! searches a window `average_score ± (5 + |mean_squared_score|/10588)`
//! instead of `(-INF, INF)`, re-searching on fails — a fail-low slides
//! the window down (beta drops exactly to the old alpha, `failed_high_cnt`
//! resets), a fail-high backs alpha off by at most one delta and runs
//! each consecutive re-search one ply shallower (`adjusted_depth`), and
//! every re-search widens delta by ~×1.34 — until the value lands
//! strictly inside the window. `root_delta` is stored before every root
//! search for doc 10's reduction formula, and the §3.4 root bound flags
//! and `uci_score` window clamps become reachable code. The unset EMA
//! sentinels make the first iteration's window clamp to `(-INF, INF)`,
//! so depth 1 needs no special case.
//!
//! Step 9 brings qsearch up to the complete §3.8 spec: the stand-pat
//! tt-value substitution (with its non-decisive guard — deliberately
//! stricter than the main search's step-6 twin), both softening blends
//! (the stand-pat cutoff and node-exit fail-high dampers), and the
//! step-7 per-move pruning block — late-move cutoff, futility against
//! `futility_base = static_eval + 335`, SEE against the window, the
//! non-capture skip and the hard SEE floor, all gated on
//! `!is_loss(best_value)` so in-check nodes prune nothing until one legal
//! evasion has produced a real value — plus the narrow stalemate test
//! that catches "qsearch just won material into a stalemate trap" for
//! ~zero cost. Constants in [`params`] (D20).
//!
//! Step 10 is the hygiene pass that closes out the doc's plan (§3.5,
//! §3.7.2 steps 12/14): the iteration-end bookkeeping grows the
//! forgotten-mate rollback (a proven mate PV/score is never replaced by
//! a weaker or bound re-score) and the aborted-loss repair (an exact
//! mated/TB-loss front score from a STOPPED iteration is untrusted —
//! unsearched root moves might refute or delay it — so the previous
//! iteration's line is restored, or the score marked a lower bound when
//! no previous iteration exists), both in [`Worker::root_hygiene`]; the
//! `go mate n` limit stops the search once a completed iteration proves
//! a mate within `2n` plies (doc 12 §3.9's mate-limit stop, wired
//! through `Limits::mate`); the step-12 equal-value dither occasionally
//! promotes an equal-to-best value near the leaves (with the §3.3
//! PV-freshness guard: a dither-promoted move truncates the PV rather
//! than stitch a child row its zero-window scout left stale); and the
//! main node gains its step-14 fail-high softening blend (qsearch
//! already had its two from step 9). Both value dampers are gated on
//! `is_decisive`/`is_win` — no decisive score is ever dithered or
//! blended (§8.11).
//!
//! Doc 12 step 7 completes P7's basic clock TM: with the `maximum` hard
//! wall already live in [`Worker::check_time`], the ID loop gains the v1
//! between-iterations soft stop ([`Worker::soft_stop`], doc 12 §3.9
//! subset) — stop deepening once elapsed exceeds the plain `optimum`
//! budget (capped by `maximum`), with the single-reply cap
//! ([`params::ONE_REPLY_TOTAL_TIME_CAP`]) as the one §3.9 refinement
//! that ships now. Fixed-depth modes bypass it (`use_time_management`
//! gate), so the bench fingerprint is TM-independent (D24).
//!
//! Deliberately NOT here yet, per the doc's staging: the four-factor
//! soft-stop stack, its mate-score early stops, and the
//! `increase_depth`/`search_again_counter` producer (doc 12 step 8,
//! P10); the ponder state machine with `stop_on_ponderhit` and the
//! fail-low revoke, whose seams the aspiration loop carries (doc 12
//! step 9) — the SMP window jitter `+ thread_idx % 8` (vacuous single-
//! threaded; joins with doc 13 in P11), the speculative `prefetch_key`
//! pre-make prefetch (doc 08 §3.15 call site 2), and every ⟨D10⟩
//! heuristic slot — including the TT cutoff refinements (cutNode
//! agreement, depth ≥ 7 verification, the `penalize(1)` trigger).

pub mod params;
pub mod rootmoves;
pub mod stack;

pub use rootmoves::{RootMove, RootMoves, build_root_moves, sort_descending};
pub use stack::{ContHistId, PvLine, SS_OFFSET, STACK_SIZE, Stack, init_stack, ss, update_pv};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::board::{Board, MakeMoveHooks};
use crate::eval::{EvalContext, evaluate};
use crate::movegen::{generate_captures, generate_evasions, generate_legal};
use crate::moves::{MAX_MOVES, Move, MoveKind, MoveList};
use crate::timeman::{Limits, TimeManager};
use crate::tt::{Bound, TranspositionTable};
use crate::types::{
    DEPTH_QS, DEPTH_UNSEARCHED, Depth, Key, MAX_PLY, PieceType, VALUE_DRAW, VALUE_INFINITE,
    VALUE_MATE, VALUE_NONE, VALUE_ZERO, Value, is_decisive, is_loss, is_mate, is_mated, is_valid,
    is_win, mate_in, mated_in, value_from_tt, value_to_tt,
};

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

/// Per-thread search state (design/07 §3.10): one worker owns its board
/// copy (D8, D23), stack, PV table and counters. Written single-threaded
/// but SMP-shaped — no globals; the stop flag is a shared borrow.
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
    /// Max ply reached on the current PV line, counted from 1 (§3.9).
    pub sel_depth: i32,
    /// Depth of the current iteration (§3.5).
    pub root_depth: Depth,
    /// Current root aspiration window width `beta - alpha`, stored before
    /// every root search — re-searches included (§3.6); feeds doc 10's
    /// reduction formula.
    pub root_delta: Value,
    /// MultiPV line currently searched (§3.5; v1 always 0).
    pub pv_idx: usize,
    /// Time-management instability signal (§3.4/§3.5).
    pub best_move_changes: f64,
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
    /// The `PoolShared.ponder` flag, v1 stand-in (doc 12 §3.2): set by
    /// the UCI layer for a `go ponder` session and cleared by
    /// `ponderhit`, consumed ONLY by [`Worker::check_time`]'s ponder
    /// guard — while pondering the GUI owns the clock, so no limit may
    /// stop the search (§3.8). `Arc` because the UCI thread clears the
    /// same flag the search polls; defaults to an unshared `false`.
    pub ponder: Arc<AtomicBool>,
    /// Best PV of the last iteration whose result was trusted (`!stop`)
    /// AND not a forgotten mate, §3.5. Persists across iterations so a
    /// `bestmove` is always printable; [`Worker::root_hygiene`] both
    /// maintains it and rolls back to it.
    pub last_best_move_pv: Vec<Move>,
    /// Score belonging to [`Self::last_best_move_pv`] (§3.5).
    pub last_best_move_score: Value,
    /// Iteration at which the best root move last changed — the
    /// time-management stability signal (§3.5; consumed by doc 12 §3.9's
    /// four-factor soft stop, step 8/P10).
    pub last_best_move_depth: Depth,
    /// Per-thread eval state (design/11 §3.1); empty in Phase A.
    eval_ctx: EvalContext,
    /// Countdown to the next real `check_time` (§3.9): reloaded from
    /// [`Worker::check_time_reset`] ([`params::CHECK_TIME_CALLS`], tighter
    /// under a node limit).
    check_time_countdown: i32,
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
        Worker {
            board,
            root_moves: RootMoves::new(),
            stack: Box::new([Stack::default(); STACK_SIZE]),
            pv_table: Box::new([EMPTY_PV; MAX_PLY + 1]),
            nodes: 0,
            sel_depth: 0,
            root_depth: 0,
            root_delta: 2 * VALUE_INFINITE,
            pv_idx: 0,
            best_move_changes: 0.0,
            limits: Limits::new(),
            tm: TimeManager::new(),
            ponder: Arc::new(AtomicBool::new(false)),
            last_best_move_pv: Vec::new(),
            last_best_move_score: -VALUE_INFINITE,
            last_best_move_depth: 0,
            eval_ctx: EvalContext::new(),
            check_time_countdown: params::CHECK_TIME_CALLS,
            #[cfg(test)]
            asp_trace: Vec::new(),
        }
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
        on_info: &mut dyn FnMut(&SearchInfo),
    ) {
        // Per-`go` reset (§3.2.4, §3.9): stack, counters, root moves.
        init_stack(&mut self.stack);
        self.nodes = 0;
        self.sel_depth = 0;
        self.root_depth = 0;
        self.pv_idx = 0;
        self.best_move_changes = 0.0;
        self.check_time_countdown = self.check_time_reset();
        self.last_best_move_pv.clear();
        self.last_best_move_score = -VALUE_INFINITE;
        self.last_best_move_depth = 0;
        #[cfg(test)]
        self.asp_trace.clear();
        self.root_moves = build_root_moves(&self.board, &self.limits.searchmoves);
        if self.root_moves.is_empty() {
            return; // mate/stalemate at the root: nothing to search
        }
        // histories.pre_age() hook (§3.5): lands with doc 09's tables.
        // MultiPV is fixed at 1 until the UCI option lands (doc 12 step
        // 10); the loop below is already shaped for it.
        let multi_pv: usize = 1;
        // Time-management instability input (doc 12 §3.9): halved every
        // iteration, harvested from `best_move_changes`. Consumed by the
        // four-factor soft stop when it lands (doc 12 step 8, P10).
        let mut tot_best_move_changes = 0.0f64;
        // TM-driven re-search damping (doc 12 §3.9): bumped at the top of
        // an iteration when the soft stop said "don't deepen"
        // (`increase_depth == false`). The producer lands with doc 12
        // step 8; until then every iteration searches at full depth
        // (`3 * (0 + 1) / 4 == 0` in the §3.6 adjusted-depth formula).
        let search_again_counter: i32 = 0;

        while self.root_depth + 1 < MAX_PLY as Depth
            && !stop.load(Ordering::Relaxed)
            // `go depth d` (design/12 §3.6): stop iterating once reached.
            && self.limits.depth.is_none_or(|d| self.root_depth < d)
        {
            self.root_depth += 1;
            tot_best_move_changes /= 2.0; // age the instability metric

            // Snapshot last iteration (§3.5).
            for (idx, rm) in self.root_moves.iter_mut().enumerate() {
                rm.previous_score = rm.score;
                rm.previous_pv.clone_from(&rm.pv);
                rm.previous_score_exact = idx < multi_pv;
            }

            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;
                self.sel_depth = 0; // per PV line (§3.9)
                self.aspiration_search(tt, stop, search_again_counter);
                // Order the completed lines among themselves (§3.5) — a
                // no-op at MultiPV 1, but the shape is load-bearing for
                // the MultiPV loop.
                sort_descending(&mut self.root_moves[..=pv_idx]);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let rm = &self.root_moves[pv_idx];
                on_info(&SearchInfo {
                    depth: self.root_depth,
                    sel_depth: rm.sel_depth,
                    multipv: pv_idx + 1,
                    score: rm.uci_score,
                    lowerbound: rm.score_lowerbound,
                    upperbound: rm.score_upperbound,
                    nodes: self.nodes,
                    hashfull: tt.hashfull(0),
                    pv: &rm.pv,
                });
            }

            // lastBestMove tracking with the forgotten-mate guard, plus
            // the aborted-loss repair (§3.5, plan step 10).
            self.root_hygiene(stop.load(Ordering::Relaxed));

            // Harvest the per-iteration instability signal (§3.5; doc 13
            // §3.10 owns the SMP aggregation shape).
            tot_best_move_changes += self.best_move_changes;
            self.best_move_changes = 0.0;

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

            // TM soft stop between completed iterations (doc 12 §3.9,
            // v1 subset — doc 12 step 7).
            self.soft_stop(stop);
        }
        // The four-factor soft stop (doc 12 step 8, P10) consumes the
        // accumulator; keep the per-iteration bookkeeping above exercised
        // and warning-free until then.
        let _ = tot_best_move_changes;
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
    ///   `optimism = 137·avg / (|avg| + 81)` attaches here when
    ///   design/11 phase B lands.
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
    fn aspiration_search(
        &mut self,
        tt: &TranspositionTable,
        stop: &AtomicBool,
        search_again_counter: i32,
    ) {
        let rm = &self.root_moves[self.pv_idx];
        let mut delta = (params::ASPIRATION_BASE_DELTA as i64
            + rm.mean_squared_score.abs() / params::ASPIRATION_MSS_DIVISOR)
            as Value;
        let avg = rm.average_score;
        let mut alpha = (avg - delta).max(-VALUE_INFINITE);
        let mut beta = (avg + delta).min(VALUE_INFINITE);
        let mut failed_high_cnt: i32 = 0;

        loop {
            let adjusted_depth = (self.root_depth
                - failed_high_cnt
                - params::ASPIRATION_SEARCH_AGAIN_NUM * (search_again_counter + 1)
                    / params::ASPIRATION_SEARCH_AGAIN_DEN)
                .max(1);
            self.root_delta = beta - alpha;
            let best = self.search::<Root>(tt, stop, alpha, beta, adjusted_depth, false, 0);

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
                // ⟨doc 12 step 9⟩ stop_on_ponderhit revoke goes here.
            } else if best >= beta {
                // FAIL HIGH: alpha never loosens past its current value.
                alpha = (beta - delta).max(alpha);
                beta = (best + delta).min(VALUE_INFINITE);
                failed_high_cnt += 1;
            } else {
                break; // inside the window: done
            }

            // Widen ~×1.34 per re-search. The cap is semantics-preserving:
            // once delta >= 2·VALUE_INFINITE both window edges clamp to
            // ±INF for every representable score and further growth is
            // inert — while unbounded ×1.34 growth would eventually
            // overflow the i32 in a long enough re-search storm.
            delta = (delta + params::ASPIRATION_WIDEN_NUM * delta / params::ASPIRATION_WIDEN_DEN)
                .min(2 * VALUE_INFINITE);
        }
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
    /// steps 2/3/4/5/6/7/9/10/11/12/13/14 — no ⟨D10⟩ heuristics (step 8);
    /// step 12's equal-value dither and step 14's fail-high softening
    /// landed with plan step 10. Step 10 is the
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
        depth: Depth,
        cut_node: bool,
        ply: usize,
    ) -> Value {
        // Depth-0 boundary: drop into qsearch, preserving PV-ness.
        if depth <= 0 {
            return if NT::PV {
                self.qsearch::<Pv>(tt, alpha, beta, ply)
            } else {
                self.qsearch::<NonPv>(tt, alpha, beta, ply)
            };
        }

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
        // Consumed-and-reset slot: doc 10's hindsight logic reads the
        // parent's reduction; resetting here keeps it from going stale.
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
                    evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO)
                } else {
                    value_draw(self.nodes)
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
        // Sticky "was ever on the PV" flag (stored back at node exit).
        self.stack[cur].tt_pv = NT::PV || (tt_hit && tt_data.is_pv);

        // Steps 6+7 — the static-eval slot, then the TT cutoff (doc 07
        // deliberately orders the cutoff AFTER the eval slot so doc 10's
        // hindsight depth adjustments can feed it). In check both are
        // skipped: the node borrows the grandparent's eval and jumps
        // straight to the move loop — no pruning happens in check.
        let raw_eval;
        if in_check {
            raw_eval = VALUE_NONE;
            self.stack[cur].static_eval = self.stack[cur - 2].static_eval;
        } else {
            // Step 6 — eval reuse: a valid cached eval (RAW, uncorrected —
            // D22) saves the evaluate() call. Correction history will
            // adjust it into `ss.static_eval` here (eval doc); in Phase A
            // corrected == raw. The step-6 tt_value-as-better-eval
            // substitution has no consumer until doc 10's pruning lands —
            // deliberately absent.
            raw_eval = if tt_hit && is_valid(tt_data.eval) {
                tt_data.eval
            } else {
                evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO)
            };
            self.stack[cur].static_eval = raw_eval;
            // Eval-cache write (§3.11): a TT miss that just paid for
            // evaluate() caches the raw eval at DEPTH_UNSEARCHED, so
            // transpositions skip the eval even if this node never
            // completes a search result.
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

            // Step 7 — the NonPV TT cutoff, v1 base form (the cutNode
            // agreement, depth ≥ 7 verification and `penalize` refinements
            // are ⟨D10⟩): enough stored depth, a valid value, the bound
            // pointing the cutoff's way, and never when the halfmove clock
            // could stale the entry (rule-50 guard, §3.12). Live since
            // plan step 7's zero-window scouts create NonPV nodes.
            if !NT::PV
                && self.stack[cur].excluded_move == Move::NULL
                && is_valid(tt_value)
                && tt_data.depth > depth - Depth::from(tt_value <= beta)
                && (if tt_value >= beta {
                    tt_data.bound.includes_lower()
                } else {
                    tt_data.bound.includes_upper()
                })
                && (self.board.rule50_count() as i32) < params::TT_CUTOFF_RULE50_LIMIT
            {
                return tt_value;
            }
        }

        // Step 9 — move loop. The staged MovePicker (doc 09) replaces
        // generate_legal + placeholder ordering; legality is then checked
        // per move instead of up front. (`excluded_move` guard arrives
        // with doc 10's singular search — nothing writes it yet.)
        //
        // At the root the loop iterates `root_moves[pv_idx..]` itself
        // (doc 07 §3.7.2 step 9's "skip moves outside the root list",
        // v1 form): the ID loop's stable sort has last iteration's best
        // in front — the root form of the step-5 ttMove override, so the
        // probed move is not consulted here. Everything in `root_moves`
        // is legal by construction, and `pv_last = root_moves.len()`
        // until MultiPV.
        let mut list = MoveList::new();
        if NT::ROOT {
            for rm in &self.root_moves[self.pv_idx..] {
                list.push(rm.pv[0]);
            }
        } else {
            generate_legal(&self.board, &mut list);
            order_moves(&self.board, &mut list, tt_data.mv);
        }

        let mut best_value = -VALUE_INFINITE;
        let mut best_move = Move::NULL;
        let mut move_count = 0;

        for m in &list {
            move_count += 1;
            self.stack[cur].move_count = move_count;
            let nodes_before = self.nodes; // root effort accounting (§3.4)

            // Step 10 — make + recurse: the PVS three-call structure
            // (§3.7.2). Only the move ordered first (the ttMove when the
            // P6 probe supplied one — its likely-best status is what makes
            // the scouts below usually right) gets a full window up front;
            // every later move must first prove itself against a
            // zero-width window before it earns a PV re-search.
            let gives_check = self.board.gives_check(m);
            let new_depth = depth - 1;
            self.do_move(tt, m, gives_check, ply);
            // Overwritten by whichever call below runs last. A PV node's
            // first move skips (a)/(b), and (c)'s `move_count == 1` arm
            // never reads the sentinel.
            let mut value = -VALUE_INFINITE;

            // (a) Reduced zero-window scout, expected to fail low. LMR
            // scouts always START as cut nodes (§3.7.1); the reduction r
            // is doc 10's (P9) — the skeleton runs r = 0, so `d` stays
            // `new_depth` (later: `new_depth - r/1024`, clamped) and the
            // recorded `ss.reduction` is 0 (the child reads it back and
            // resets it at entry, step 2).
            if depth >= 2 && move_count > 1 {
                let d = new_depth;
                self.stack[cur].reduction = new_depth - d;
                value = -self.search::<NonPv>(tt, stop, -(alpha + 1), -alpha, d, true, ply + 1);

                // (b) The scout beat alpha at REDUCED depth: verify at
                // full depth, still zero-window, with the negamax sign
                // discipline on cut-node-ness (`!cut_node`, §3.7.1).
                // Structurally dead while r = 0 keeps d == new_depth.
                if value > alpha && d < new_depth {
                    value = -self.search::<NonPv>(
                        tt,
                        stop,
                        -(alpha + 1),
                        -alpha,
                        new_depth,
                        !cut_node,
                        ply + 1,
                    );
                }
            } else if !NT::PV || move_count > 1 {
                // (b) directly, when (a)'s conditions don't hold: depth 1,
                // or a NonPV node's first move (where `(-(alpha+1), -alpha)`
                // IS the node's own zero-width window).
                value = -self.search::<NonPv>(
                    tt,
                    stop,
                    -(alpha + 1),
                    -alpha,
                    new_depth,
                    !cut_node,
                    ply + 1,
                );
            }

            // (c) PV re-search, full window: PV/Root nodes only, for the
            // first move and for any scout that beat alpha. PV children
            // are never cut nodes — and only THIS call writes
            // `pv_table[ply + 1]`, which is §3.3's freshness invariant:
            // a step-13 PV update must never stitch a child row a
            // zero-window scout left stale (see `child_pv_fresh` below).
            let mut child_pv_fresh = false;
            if NT::PV && (move_count == 1 || value > alpha) {
                value = -self.search::<Pv>(tt, stop, -beta, -alpha, new_depth, false, ply + 1);
                child_pv_fresh = true;
            }
            self.undo_move(m);

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
                    && ply + params::DITHER_PLY_MARGIN >= self.root_depth as usize
                    && (self.nodes & params::DITHER_NODE_MASK) == 0
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
                        // `(extension < 2) || PV` — extensions are doc
                        // 10's; the condition is vacuously true here.
                        self.stack[cur].cutoff_cnt += 1;
                        break; // fail high
                    }
                    alpha = value;
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
            best_value = (best_value * depth + beta) / (depth + 1);
        }

        // Step 14 — node result: ply-relative mate/stalemate terminals
        // (D5): a longer mate scores closer to zero, so the search
        // naturally prefers the shortest mate and delays being mated.
        if move_count == 0 {
            best_value = if in_check {
                mated_in(ply as i32)
            } else {
                VALUE_DRAW
            };
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
                depth + params::TT_NO_MOVES_DEPTH_BONUS
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

        debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);
        best_value
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
            // (§3.4; consumed by doc 12 §3.9's factor stack in P10).
            if move_count > 1 && self.pv_idx == 0 {
                self.best_move_changes += 1.0;
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
    /// No stop poll here (§3.9, SF parity): qsearch is bounded by the
    /// main-search checks above it.
    fn qsearch<NT: NodeType>(
        &mut self,
        tt: &TranspositionTable,
        mut alpha: Value,
        beta: Value,
        ply: usize,
    ) -> Value {
        debug_assert!(!NT::ROOT, "qsearch is never Root-typed (§3.7.1)");
        debug_assert!(NT::PV || alpha == beta - 1, "NonPV requires a zero window");
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);

        // Step 2 — init.
        let cur = ss(ply);
        let in_check = !self.board.checkers().is_empty();
        self.stack[cur].in_check = in_check;
        if NT::PV {
            self.pv_table[ply].clear();
            self.sel_depth = self.sel_depth.max(ply as i32 + 1);
        }

        // Step 3 — draw/max-ply: plain VALUE_DRAW, deliberately NOT
        // dithered (SF parity; see [`value_draw`]).
        if self.board.is_draw(ply as i32) || ply >= MAX_PLY {
            return if ply >= MAX_PLY && !in_check {
                evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO)
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
                evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO)
            };
            self.stack[cur].static_eval = raw_eval; // corrected == raw (Phase A)
            best_value = raw_eval;

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
                // Search-instability damping (§3.8 step 5): pull a
                // non-decisive stand-pat cutoff partway back toward beta.
                // Decisive scores are never blended (§8.11).
                if !is_decisive(best_value) {
                    let (w_best, w_beta, den) = params::QS_STANDPAT_BLEND;
                    best_value = (w_best * best_value + w_beta * beta) / den;
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
            futility_base = self.stack[cur].static_eval + params::QS_FUTILITY_MARGIN;
        }

        // Steps 6–7 — captures (+ queen promotions, the CAPTURES stage of
        // doc 05 §3.2, mirroring `Board::capture_stage`), or every evasion
        // in check. Pseudo-legal generation; legality tested on the move
        // we are about to play. A probed ttMove sorts first when the
        // generator emitted it — list membership subsumes its validation.
        let mut list = MoveList::new();
        if in_check {
            generate_evasions(&self.board, &mut list);
        } else {
            generate_captures(&self.board, &mut list);
        }
        order_moves(&self.board, &mut list, tt_data.mv);

        // The previous move's destination (§3.8 step 6): recaptures on
        // that square are exempt from the step-7 futility block. NULL
        // means "no real move" (root entry, later null-move search).
        let prev_move = self.stack[cur - 1].current_move;
        let prev_sq = (prev_move != Move::NULL).then(|| prev_move.to());

        let mut best_move = Move::NULL;
        let mut move_count = 0;
        for m in &list {
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
                    if move_count > params::QS_LATE_MOVE_LIMIT {
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
                        best_value = best_value.max(futility_value);
                        continue;
                    }

                    // SEE vs the window: the exchange cannot recover the
                    // gap between futility_base and alpha.
                    if !self.board.see_ge(m, alpha - futility_base) {
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
                if !self.board.see_ge(m, params::QS_SEE_FLOOR) {
                    continue;
                }
            }

            // Step 8 — recurse: full window, same node type.
            self.do_move(tt, m, gives_check, ply);
            let value = -self.qsearch::<NT>(tt, -beta, -alpha, ply + 1);
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
            let (w_best, w_beta, den) = params::QS_FAILHIGH_BLEND;
            best_value = (w_best * best_value + w_beta * beta) / den;
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
    /// moment its key is final, mid-make. The continuation-history key
    /// install (keyed by `(in_check, capture, piece, to)`) lands with doc
    /// 09; the sentinel is always safe to read through.
    #[inline]
    fn do_move(&mut self, tt: &TranspositionTable, m: Move, gives_check: bool, ply: usize) {
        self.nodes += 1;
        let cur = ss(ply);
        self.stack[cur].current_move = m;
        self.stack[cur].cont_hist = ContHistId::SENTINEL;
        self.board
            .make_move_with(m, gives_check, &mut TtPrefetch { tt });
    }

    #[inline]
    fn undo_move(&mut self, m: Move) {
        self.board.unmake_move(m);
    }

    /// The §3.9 time-check cadence seam: called at every main-search node
    /// entry, early-outs on a countdown and does real work every
    /// [`params::CHECK_TIME_CALLS`] calls (tightened to ≥ 0.1% of the
    /// budget under `go nodes`, design/12 §3.8, so tiny node budgets are
    /// honored).
    ///
    /// v1 real work — the hard limits a playable engine needs (design/12
    /// step 5): `movetime` (wall clock from `Limits::start_time`, exactly
    /// `TimeManager::elapsed_ms` semantics), `nodes` (single worker, so
    /// `self.nodes` IS the total until SMP aggregation lands, D23), and
    /// the wtime/btime hard wall `elapsed > maximum` from the worker's
    /// [`TimeManager`] — all behind the §3.8 ponder guard. `go depth` is
    /// enforced by the ID loop, `go infinite` never self-stops.
    /// `stop_on_ponderhit` arrives with doc 12 step 9 (P9); the
    /// soft-stop factors with step 8 (P10).
    #[inline]
    fn check_time(&mut self, stop: &AtomicBool) {
        self.check_time_countdown -= 1;
        if self.check_time_countdown > 0 {
            return;
        }
        self.check_time_countdown = self.check_time_reset();
        // The GUI owns the clock while pondering: NEVER stop here
        // (design/12 §3.8) — `ponderhit`/`stop` are the only ways out of
        // a ponder search, which keeps `bestmove` behind the UCI layer's
        // busy-wait until one of them arrives.
        if self.ponder.load(Ordering::Relaxed) {
            return;
        }
        // Hard clock wall (§3.8 v1 subset). `use_time_management` gates
        // the comparison, so depth/nodes/movetime/infinite modes never
        // see it; `tm.elapsed` counts nodes under nodestime, keeping the
        // units consistent with `maximum()`. When the GUI sent only the
        // opponent's clock, `init` left `maximum() == 0` and the first
        // real poll here stops the search (doc 12 §3.7 step 1).
        let nodes = self.nodes;
        let tm_stop =
            self.limits.use_time_management() && self.tm.elapsed(|| nodes) > self.tm.maximum();
        let movetime_stop = self
            .limits
            .movetime
            .is_some_and(|t| self.limits.start_time.elapsed().as_millis() as i64 >= t);
        let nodes_stop = self.limits.nodes.is_some_and(|n| self.nodes >= n);
        if tm_stop || movetime_stop || nodes_stop {
            stop.store(true, Ordering::Relaxed);
        }
    }

    /// The between-iterations soft stop (design/12 §3.9), v1 subset (doc
    /// 12 step 7): once the soft budget is spent, starting another
    /// iteration is pointless — it would almost certainly be cut off by
    /// the `maximum` hard wall before completing, wasting clock a later
    /// move could have used. v1 compares elapsed against the plain
    /// `optimum` allocation; the four multiplicative factors
    /// (fallingEval, reduction, bestMoveInstability, highBestMoveEffort),
    /// the mate-score early stops, and the `increase_depth` /
    /// `search_again_counter` damping land with doc 12 step 8 (P10) —
    /// their inputs (`tot_best_move_changes`, `last_best_move_depth`,
    /// per-root-move `effort`) are already maintained by the ID loop.
    ///
    /// The one §3.9 refinement that IS v1 (this step's brief): the
    /// single-reply cap — with exactly one legal move the search result
    /// cannot change the move played, so the budget is capped at
    /// [`params::ONE_REPLY_TOTAL_TIME_CAP`].
    ///
    /// Guards mirror the doc's `use_time_management() && !stop` gate plus
    /// §3.8's ponder rule: while pondering the GUI owns the clock, so
    /// nothing may self-stop — the doc's "remember the verdict in
    /// `stop_on_ponderhit` instead" conversion arrives with the ponder
    /// state machine (doc 12 step 9). `tm.elapsed` counts nodes under
    /// `nodestime`, keeping the units consistent with the budgets
    /// (§3.11) — which is also the fake-clock seam the unit tests use.
    fn soft_stop(&self, stop: &AtomicBool) {
        if !self.limits.use_time_management()
            || stop.load(Ordering::Relaxed)
            || self.ponder.load(Ordering::Relaxed)
        {
            return;
        }
        // v1 total_time == optimum; the factor stack multiplies in here
        // (doc 12 step 8, P10).
        let mut total_time = self.tm.optimum() as f64;
        if self.root_moves.len() == 1 {
            total_time = total_time.min(params::ONE_REPLY_TOTAL_TIME_CAP);
        }
        let nodes = self.nodes;
        let elapsed = self.tm.elapsed(|| nodes) as f64;
        if elapsed > total_time.min(self.tm.maximum() as f64) {
            stop.store(true, Ordering::Relaxed);
        }
    }

    /// Countdown reload for [`Self::check_time`] (design/12 §3.8): 512
    /// node-entries, or `min(512, nodes_limit / 1024)` under a node limit
    /// (which may be 0 — then every node entry does the real check).
    fn check_time_reset(&self) -> i32 {
        match self.limits.nodes {
            Some(n) => (n / 1024).min(params::CHECK_TIME_CALLS as u64) as i32,
            None => params::CHECK_TIME_CALLS,
        }
    }
}

/// ttMove-first MVV-LVA placeholder ordering until doc 09's staged
/// MovePicker lands (design/07 §4 steps 4/6): the probed TT move — when
/// the generator emitted it, which is also its semantic validation (a
/// collided or garbage `tt_move` simply matches nothing and is dropped,
/// doc 08 §3.5) — sorts in front; then captures by most valuable victim,
/// ties broken by least valuable attacker; promotions add the promoted
/// piece so queen promotions sort with queen captures; quiets keep
/// generation order behind all captures.
///
/// Insertion sort: STABLE and in-place (no allocation on the hot path,
/// §6), so the ttMove's front placement preserves the relative order of
/// everything else and node counts are deterministic (D24). O(n²) worst
/// case is fine for n ≤ 256 in a placeholder the MovePicker deletes.
fn order_moves(board: &Board, list: &mut MoveList, tt_move: Move) {
    let mut scores = [0i32; MAX_MOVES];
    for (s, &m) in scores.iter_mut().zip(list.iter()) {
        // No generated move is ever Move::NULL, so a NULL tt_move (probe
        // miss / move-less entry) matches nothing.
        *s = if m == tt_move {
            i32::MAX
        } else {
            move_score(board, m)
        };
    }
    for i in 1..list.len() {
        let (m, s) = (list[i], scores[i]);
        let mut j = i;
        while j > 0 && scores[j - 1] < s {
            list[j] = list[j - 1];
            scores[j] = scores[j - 1];
            j -= 1;
        }
        list[j] = m;
        scores[j] = s;
    }
}

/// MVV-LVA key for [`order_moves`]. Radix-8 ordinal packing over
/// `PieceType` discriminants — pure ordering structure, not tunable
/// values (the D20 params rule covers tuner-movable constants).
fn move_score(board: &Board, m: Move) -> i32 {
    // Victim: en passant captures a pawn that is not on `to`; castling
    // never captures; otherwise whatever sits on the destination.
    let victim = match m.kind() {
        MoveKind::EnPassant => Some(PieceType::Pawn),
        MoveKind::Castling => None,
        _ => board.piece_on(m.to()).map(|p| p.piece_type),
    };
    let mut score = match victim {
        // Most valuable victim dominates (×8 ranks victims apart);
        // least valuable attacker breaks ties.
        Some(v) => (v as i32 + 1) * 8 - board.piece_on(m.from()).map_or(0, |p| p.piece_type as i32),
        None => 0,
    };
    if let Some(promo) = m.promotion_piece() {
        score += (promo as i32 + 1) * 8;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;
    use crate::types::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
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
        w.iterative_deepening(tt, &stop, &mut |_| {});
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

    // --- Mate suite (doc 07 §5/§8.4): exact ply-relative mate scores (D5) ---

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
        // exact score sticks. The §8.4 bar is depth <= 6.
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
    fn two_rook_ladder_mate_in_3_exact_at_depth_6() {
        // 1.Rb6+ (rank 5 stays cut by Ra5) K→7th 2.Ra7+ K→8th 3.Rb8#:
        // five plies. The black king starts on g6, too far from the a/b
        // files to ever attack the unprotected checking rooks, and too
        // free for anything shorter than the ladder. Depth 6 = the five
        // plies the mate needs plus the aspiration settling iteration
        // (see the KRK test) — exactly the §8.4 gate's "depth ≤ 6".
        let w = go_depth("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 6);
        assert_eq!(best_score(&w), mate_in(5));
        assert_eq!(w.root_pv().len(), 5);
        // The PV replays legally and its end really is checkmate.
        let end = replay_pv(&w);
        assert!(all_legal(&end).is_empty() && !end.checkers().is_empty());
    }

    #[test]
    fn mate_suite_1_to_3_exact_scores_and_first_moves() {
        // The §8.4 exit bar in table form: ≥ 10 positions across mates in
        // 1/2/3, each solved at depth ≤ 6 with the exact ply-relative
        // score AND a verified-optimal first move. Every entry — the mate
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
        assert!(SUITE.len() >= 10, "the §8.4 bar is at least 10 positions");
        for &(fen, plies, optimal) in SUITE {
            // Mate-in-1 uses depth 2 so the score comes from a full search
            // iteration, not bare qsearch (found at depth 1, the mate-band
            // EMAs keep every later window effectively infinite). Deeper
            // mates need the mate distance plus one aspiration settling
            // iteration (see the KRK test) — still within the §8.4
            // "depth ≤ 6" bar.
            let depth = if plies == 1 { 2 } else { plies + 1 };
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
        let w = go_depth("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1", 3);
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

    /// The corrected static eval exactly as qsearch computes it
    /// (Phase A: corrected == raw).
    fn static_eval(b: &Board) -> Value {
        evaluate(b, &mut EvalContext::new(), VALUE_ZERO)
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
        w.qsearch::<NT>(tt, alpha, beta, 0)
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
        let (w_best, w_beta, den) = params::QS_STANDPAT_BLEND;
        assert_eq!(v, (w_best * eval + w_beta * beta) / den);
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
        let eval = static_eval(&w.board);
        let beta = eval + 1000;
        let v = qsearch_at::<NonPv>(&mut w, &tt, beta - 1, beta);
        assert_eq!(
            v,
            eval + params::QS_FUTILITY_MARGIN + PieceType::Pawn.value()
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
        let eval = static_eval(&w.board);
        let alpha = eval + 500;
        let v = qsearch_at::<NonPv>(&mut w, &tt, alpha, alpha + 1);
        assert_eq!(v, eval + params::QS_FUTILITY_MARGIN);
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
        let eval = static_eval(&w.board);
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
            params::QS_LATE_MOVE_LIMIT
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
        assert_eq!(v, -static_eval(&after));
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
        assert_eq!(v, -static_eval(&after));
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
        let (w_best, w_beta, den) = params::QS_FAILHIGH_BLEND;
        assert_eq!(v, (w_best * -static_eval(&after) + w_beta * beta) / den);
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
        let (w_best, w_beta, den) = params::QS_FAILHIGH_BLEND;
        assert_eq!(v, (w_best * -static_eval(&after) + w_beta * beta) / den);
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
        let eval = static_eval(&w.board);
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

    // --- Stop-flag seam (§3.9): bestmove is printable even so ---

    #[test]
    fn preset_stop_searches_nothing_but_keeps_a_legal_bestmove() {
        let mut w = Worker::new(board(START_FEN));
        w.limits.depth = Some(6);
        let stop = AtomicBool::new(true);
        let mut infos = 0;
        w.iterative_deepening(&test_tt(), &stop, &mut |_| infos += 1);
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
        w.iterative_deepening(&test_tt(), &stop, &mut |i| {
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
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert!(
            stop.load(Ordering::Relaxed),
            "check_time must raise the stop flag on the node limit"
        );
        // The stop fires at the first real check at/past the budget; the
        // cadence tightens to 20_000/1024 = 19 node entries (§3.8), so the
        // overshoot is the qsearch slack between checks — small.
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
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert!(stop.load(Ordering::Relaxed), "movetime must self-stop");
        assert!(w.best_move().is_some());
    }

    #[test]
    fn ponder_guard_blocks_every_check_time_stop() {
        // design/12 §3.8: the GUI owns the clock while pondering — an
        // already-expired movetime must NOT stop a ponder search. The
        // depth limit (ID-loop condition, not check_time) bounds the
        // test; depth 4 guarantees the 512-entry cadence fires real
        // checks (depth <= 3 has too few main-search node entries).
        let mut w = Worker::new(board(START_FEN));
        w.limits.movetime = Some(0); // expired from the first poll
        w.limits.start_time = std::time::Instant::now();
        w.limits.depth = Some(4);
        w.limits.ponder = true;
        w.ponder = Arc::new(AtomicBool::new(true));
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert!(!stop.load(Ordering::Relaxed), "pondering must never stop");
        assert_eq!(w.root_depth, 4, "the search deepened on GUI time");

        // Control: the same limits without the ponder flag stop at once.
        let mut w = Worker::new(board(START_FEN));
        w.limits.movetime = Some(0);
        w.limits.start_time = std::time::Instant::now();
        w.limits.depth = Some(4);
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
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
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
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
        w.iterative_deepening(&tt, &stop, &mut |i| {
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
    fn soft_stop_halts_deepening_once_optimum_is_spent() {
        // 60 s on the 1-node-per-ms fake clock: optimum ≈ 1.25k nodes,
        // maximum ≈ 8.6k. The run must end via the between-iterations
        // soft stop — final nodes past optimum (it never stops sooner)
        // but within maximum (the hard wall never fired).
        let mut w = Worker::new(board(START_FEN));
        w.limits.time[crate::types::Color::White as usize] = 60_000;
        init_clock(&mut w, &nodestime_options());
        let (optimum, maximum) = (w.tm.optimum(), w.tm.maximum());
        assert!(0 < optimum && optimum < maximum);
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert!(stop.load(Ordering::Relaxed), "the soft stop must self-stop");
        assert!(
            w.nodes as i64 > optimum,
            "stopped before optimum was spent: {} <= {optimum}",
            w.nodes
        );
        assert!(
            w.nodes as i64 <= maximum,
            "the hard wall fired, not the soft stop: {} > {maximum}",
            w.nodes
        );
        assert!(w.best_move().is_some());
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
            w.tm.optimum() as f64 > 2.0 * params::ONE_REPLY_TOTAL_TIME_CAP,
            "test premise: the uncapped budget must dwarf the cap"
        );
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert_eq!(w.root_moves.len(), 1, "test premise: one legal move");
        assert!(stop.load(Ordering::Relaxed), "the capped soft stop fires");
        assert!(
            w.nodes as f64 > params::ONE_REPLY_TOTAL_TIME_CAP,
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
        w.ponder = Arc::new(AtomicBool::new(true));
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
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
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
        assert!(stop.load(Ordering::Relaxed), "the soft stop must fire");
        assert!(w.root_depth < 4, "deepening was cut short at optimum");
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
        w.iterative_deepening(&test_tt(), &stop, &mut |i| pv_heads.push(i.pv[0]));
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
        w.iterative_deepening(&test_tt(), &stop, &mut |_| {});
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
        let max_depth = 10;
        let mut probe = Worker::new(board(fen));
        probe.limits.depth = Some(max_depth);
        let mut infos: Vec<(Depth, Value, Move, u64)> = Vec::new();
        {
            let tt = test_tt();
            tt.new_search();
            let stop = AtomicBool::new(false);
            probe.iterative_deepening(&tt, &stop, &mut |i| {
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
        w.iterative_deepening(&tt, &stop, &mut |i| {
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
        w.iterative_deepening(&tt, &stop, &mut |_| {});
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
        let e_child = static_eval(&after);
        let beta = -e_child - 100; // parent's true value -E sits above it
        let tt = test_tt();
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
        w.qsearch::<NonPv>(&tt, 0, 2, 1);
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
            let d0 = (params::ASPIRATION_BASE_DELTA as i64
                + mss.abs() / params::ASPIRATION_MSS_DIVISOR) as Value;
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
                (p.delta + params::ASPIRATION_WIDEN_NUM * p.delta / params::ASPIRATION_WIDEN_DEN)
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
        assert_eq!(t0.delta, params::ASPIRATION_BASE_DELTA);
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
        // (delta = 5 + 90_000/10588 = 13, window (287, 313)): the first
        // attempt fails low. Beta drops exactly to the old alpha, alpha
        // backs off delta below the returned value, failed_high_cnt stays
        // 0 so the re-search keeps the FULL depth, and delta widens.
        // (The stop_on_ponderhit revoke rides this branch once the flag
        // exists — doc 12 step 9.)
        let w = seeded_aspiration_run(START_FEN, 2, 300, 300 * 300);
        let trace = &w.asp_trace;
        assert!(trace.len() >= 2, "the quiet score must fail low");
        let t0 = trace[0];
        assert_eq!(t0.delta, 13);
        assert_eq!((t0.alpha, t0.beta), (287, 313));
        assert!(t0.value <= t0.alpha, "attempt 0 fails low");
        let t1 = trace[1];
        assert_eq!(t1.beta, 287, "beta slides down exactly to the old alpha");
        assert_eq!(t1.alpha, (t0.value - 13).max(-VALUE_INFINITE));
        assert_eq!(t1.failed_high_cnt, 0, "fail low resets failed_high_cnt");
        assert_eq!(t1.adjusted_depth, 2, "fail low keeps the full depth");
        assert_eq!(t1.delta, 13 + 44 * 13 / 128); // widened to 17
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
        // scores stay inside every window, aspiration is a pure
        // node-count optimization — same fixed-depth value and best move
        // as the full-window driver. The trace assert pins the fixtures'
        // load-bearing property: no attempt ever failed.
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
        let suite: &[(&str, Depth)] = &[
            // White up a queen in an otherwise symmetric opening.
            (
                "r1b1kbnr/pppppppp/2n5/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
                4,
            ),
            // The mirror image: black up a queen, black to move.
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
                4,
            ),
            // Black up a knight in a bare-kings endgame.
            ("4k3/8/3n4/8/8/8/8/4K3 b - - 0 1", 3),
        ];
        for &(fen, depth) in suite {
            let asp = go_depth(fen, depth);
            assert_eq!(
                asp.asp_trace.len(),
                depth as usize,
                "score-stable fixture stayed inside every window: {fen}"
            );
            let full = go_depth_full_window(fen, depth);
            assert_eq!(best_score(&asp), best_score(&full), "value identity: {fen}");
            assert_eq!(
                asp.best_move(),
                full.best_move(),
                "bestmove identity: {fen}"
            );
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
            evaluate(b, ctx, VALUE_ZERO)
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
        let suite: &[(&str, Depth)] = &[
            // KRK mate-in-2: exercises mate-distance pruning soundness.
            ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3),
            // Tiny pawn ending with captures in the qsearch tree.
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 3),
            // The same ending one ply deeper: scouts inside scouts.
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 4),
            // Back-rank mate-in-1 with pawns on both sides.
            ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 2),
            // Minor-piece skirmish: black to move, hanging knight.
            ("4k3/8/3n4/8/4B3/8/8/4K3 b - - 0 1", 2),
            ("4k3/8/3n4/8/4B3/8/8/4K3 b - - 0 1", 3),
            // Two-rook ladder: wide branching, decisive scores in-tree.
            ("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 3),
            // Mated side: fail-lows all the way down at the root.
            ("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1", 3),
        ];
        for &(fen, depth) in suite {
            let w = go_depth_full_window(fen, depth);
            let (want, optimal) = ref_root(fen, depth);
            assert_eq!(
                best_score(&w),
                want,
                "PVS vs full-window negamax value diverged: {fen} depth {depth}"
            );
            let bm = w.best_move().unwrap();
            assert!(
                optimal.contains(&bm),
                "PVS bestmove {bm} is not full-window-optimal for {fen} depth {depth} \
                 (optimal: {optimal:?})"
            );
        }
    }

    // --- TT value equality at fixed depth (doc 07 §4 step 6's bar,
    //     re-scoped by step 7; doc 08 §8 criterion 6's value half) ---

    /// One position of the TT value-equality suite: search it twice with
    /// the TT enabled — a fresh table, then the same table warmed by that
    /// run — assert both agree, and return the shared value. Same-depth
    /// TT state must never move a fixed-depth value: an equal-draft
    /// entry's bound is a correct bound on the node's fixed-depth value
    /// (§3.12 keeps mate scores exact; the fixtures keep rule-50 and
    /// repetition path-dependence out of reach), so a cutoff only
    /// short-circuits work alpha-beta would have done anyway.
    ///
    /// A table warmed by a DEEPER run is deliberately NOT an equality
    /// leg anymore: plan step 7 made the NonPV TT cutoffs live, and
    /// `tt_depth > depth - (tt_value <= beta)` accepts deeper drafts —
    /// deeper information legitimately substitutes into zero-window
    /// subtrees and may move a fixed-depth value (measured on the
    /// two-rook ladder at depth 3 after a depth-4 warm-up). That is the
    /// cutoff doing its job, not a leak; the P6 version of this helper
    /// asserted the stricter triple only because the cutoffs were
    /// unreachable before the zero-window existed.
    ///
    /// Runs through the full-window test driver since plan step 8: the
    /// aspirated driver's decisive-score dynamics (shallower fail-high
    /// re-searches) move fixed-depth values on the mate fixtures below
    /// for reasons that have nothing to do with TT semantics — which is
    /// what this suite pins.
    fn tt_value_pair(fen: &str, depth: Depth) -> Value {
        let tt = test_tt();
        let fresh = go_depth_full_window_on(fen, depth, &tt);
        let warm = go_depth_full_window_on(fen, depth, &tt);
        assert_eq!(
            best_score(&fresh),
            best_score(&warm),
            "fresh vs warm TT: {fen}"
        );
        best_score(&fresh)
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
        let suite: &[(&str, Depth)] = &[
            // Two-rook ladder: wide branching, decisive scores in the tree.
            ("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 3),
            // KRK mate-in-2: decisive scores through value_to_tt/from_tt.
            ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3),
            // Mated-side reporting through the TT boundary.
            ("7k/8/6K1/8/8/8/2Q5/8 b - - 0 1", 3),
            // Tiny pawn ending: eval-cache traffic plus qsearch captures.
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 4),
            // Back-rank mate with pawn shields on both sides.
            ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 3),
            // Minor-piece skirmish: black to move, hanging knight.
            ("4k3/8/3n4/8/4B3/8/8/4K3 b - - 0 1", 3),
        ];
        for &(fen, depth) in suite {
            let tt_off = ref_negamax(&mut board(fen), &mut EvalContext::new(), depth, 0);
            assert_eq!(tt_value_pair(fen, depth), tt_off, "TT-on vs TT-off: {fen}");
        }
    }

    #[test]
    fn tt_state_never_moves_the_startpos_value() {
        // Full-material coverage for the suite above, without the
        // exponential no-cutoff oracle: fresh and same-depth-warmed
        // searches must agree on the startpos value — any inequality
        // would be same-draft TT state moving a returned value.
        tt_value_pair(START_FEN, 3);
    }

    #[test]
    fn warm_tt_searches_fewer_nodes_at_equal_depth() {
        // The other half of the step-6 exit bar in miniature (the real
        // "bench nodes drop materially" number is recorded in the commit
        // trailer, D24): re-searching the same position at the same depth
        // against the warmed table must cost fewer nodes — the ttMove-
        // first ordering and the eval cache are actually doing work.
        // Depth 4 keeps the draw dither unreachable (see the test above),
        // so the value-equality assert can ride along.
        let tt = test_tt();
        let fresh = go_depth_on(START_FEN, 4, &tt);
        let warm = go_depth_on(START_FEN, 4, &tt);
        assert_eq!(best_score(&fresh), best_score(&warm));
        assert!(
            warm.nodes < fresh.nodes,
            "warm TT must shrink the tree: {} !< {}",
            warm.nodes,
            fresh.nodes
        );
    }
}
