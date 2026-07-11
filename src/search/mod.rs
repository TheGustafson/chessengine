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
//! guard (compiled but unreachable until the PVS zero-window creates
//! NonPV nodes — which is exactly what keeps step 6's TT-on/off VALUE
//! equality bit-exact, proven by the §5 suite below), and the TT prefetch
//! hook wired into [`Worker::do_move`] through the [`MakeMoveHooks`] seam
//! (doc 04 §3.10 hook A, doc 08 §3.15). The table itself reaches the
//! search as `&TranspositionTable` through the search entry (doc 07
//! §3.10); the session layer holds the `Arc<TT>` handle for the job's
//! duration and bumps `new_search()` once per `go` (D23, doc 13 §3.4/§3.7
//! — the worker deliberately does NOT store the handle, so the idle UCI
//! thread keeps `Arc::get_mut` for resize/clear).
//!
//! Deliberately NOT here yet, per the doc's staging: the PVS zero-window
//! three-call structure (step 7), aspiration windows (§3.6, step 8), the
//! full qsearch rules including the stand-pat tt-value substitution and
//! softening blends (§3.8, step 9), the hygiene pass (forgotten mate,
//! aborted loss, `go mate n`, equal-value dither — step 10), the
//! soft-stop factors and ponder guard of the clock TM (doc 12 steps 8–9,
//! P9/P10), the speculative `prefetch_key` pre-make prefetch (doc 08
//! §3.15 call site 2), and every ⟨D10⟩ heuristic slot — including the TT
//! cutoff refinements (cutNode agreement, depth ≥ 7 verification, the
//! `penalize(1)` trigger).

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
    VALUE_NONE, VALUE_ZERO, Value, is_valid, mate_in, mated_in, value_from_tt, value_to_tt,
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
    /// The line's `uci_score`: clamped to the aspiration window on bounds
    /// (never actually bound-flagged before plan step 8 lands).
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
    /// Current root window width `beta - alpha`; feeds doc 10's reduction
    /// formula once aspiration windows land (§3.6).
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
    /// thread), and [`Worker::check_time`] enforces `maximum()` as the
    /// hard clock wall. `stop_on_ponderhit` and the soft-stop factors
    /// are doc 12 steps 8–9 (P9/P10).
    pub tm: TimeManager,
    /// The `PoolShared.ponder` flag, v1 stand-in (doc 12 §3.2): set by
    /// the UCI layer for a `go ponder` session and cleared by
    /// `ponderhit`, consumed ONLY by [`Worker::check_time`]'s ponder
    /// guard — while pondering the GUI owns the clock, so no limit may
    /// stop the search (§3.8). `Arc` because the UCI thread clears the
    /// same flag the search polls; defaults to an unshared `false`.
    pub ponder: Arc<AtomicBool>,
    /// Best PV of the last iteration whose result was trusted (`!stop`),
    /// §3.5. Persists across iterations so a `bestmove` is always
    /// printable; the forgotten-mate / aborted-loss hygiene (plan step 10)
    /// rolls back to it.
    pub last_best_move_pv: Vec<Move>,
    /// Score belonging to [`Self::last_best_move_pv`] (§3.5).
    pub last_best_move_score: Value,
    /// Iteration at which the best root move last changed — the
    /// time-management stability signal (§3.5, doc 12 §3.9, P7).
    pub last_best_move_depth: Depth,
    /// Per-thread eval state (design/11 §3.1); empty in Phase A.
    eval_ctx: EvalContext,
    /// Countdown to the next real `check_time` (§3.9): reloaded from
    /// [`Worker::check_time_reset`] ([`params::CHECK_TIME_CALLS`], tighter
    /// under a node limit).
    check_time_countdown: i32,
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
        }
    }

    /// The iterative-deepening driver (design/07 §3.5), v1 scope: the
    /// per-`go` entry point for every search mode. Builds the root move
    /// list (with the `searchmoves` filter, design/12 §3.6), then deepens
    /// one full-window root search at a time — the aspiration loop (§3.6)
    /// replaces the inner search in plan step 8 — with the per-iteration
    /// bookkeeping this phase owns: previous-score snapshots, stable
    /// re-sorting, lastBestMove tracking and the instability accumulator.
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
        // soft-stop decision when clock TM lands (P7).
        let mut tot_best_move_changes = 0.0f64;

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
                // v1: one full-window root search per iteration; the
                // aspiration loop (§3.6, plan step 8) replaces this.
                self.root_delta = 2 * VALUE_INFINITE;
                self.search::<Root>(
                    tt,
                    stop,
                    -VALUE_INFINITE,
                    VALUE_INFINITE,
                    self.root_depth,
                    false,
                    0,
                );
                // Stable: unsearched moves keep last iteration's order
                // (§3.4) — even on stop, results still sort safely.
                sort_descending(&mut self.root_moves[pv_idx..]);
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

            // lastBestMove tracking (§3.5); the forgotten-mate guard on
            // the update joins with the hygiene pass (plan step 10).
            if !stop.load(Ordering::Relaxed) {
                if self.last_best_move_pv.first() != self.root_moves[0].pv.first() {
                    self.last_best_move_depth = self.root_depth;
                }
                self.last_best_move_pv.clone_from(&self.root_moves[0].pv);
                self.last_best_move_score = self.root_moves[0].score;
            }

            // Harvest the per-iteration instability signal (§3.5; doc 13
            // §3.10 owns the SMP aggregation shape).
            tot_best_move_changes += self.best_move_changes;
            self.best_move_changes = 0.0;

            // ⟨plan step 10⟩ forgotten-mate / aborted-loss hygiene and the
            // `go mate n` stop live here.
            // ⟨P7, doc 12 §3.9⟩ the TM soft-stop decision consumes
            // tot_best_move_changes / last_best_move_depth / effort here.
        }
        // Until the P7 soft stop consumes it, read the accumulator
        // explicitly so the per-iteration bookkeeping above stays
        // exercised and warning-free.
        let _ = tot_best_move_changes;
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
    /// steps 2/3/4/5/6/7/9/10/11/12/13/14 — no ⟨D10⟩ heuristics (step 8),
    /// no equal-value dither in step 12 (plan step 10). Step 10 is plain
    /// full-window negamax until the PVS three-call structure lands (plan
    /// step 7), so PV nodes only ever spawn PV children here — which also
    /// keeps `pv_table[1]` fresh for every root move's step-12 PV rebuild,
    /// and makes the step-7 NonPV TT cutoff below dead code for now:
    /// nothing the TT stores can reach a returned value yet, so TT-on and
    /// TT-off searches return identical values at fixed depth (the step-6
    /// done-condition, pinned by the §5 equality suite).
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
            // could stale the entry (rule-50 guard, §3.12). Unreachable
            // until plan step 7's zero-window creates NonPV nodes.
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

            // Step 10 — make + recurse, full-window negamax (see fn docs).
            let gives_check = self.board.gives_check(m);
            self.do_move(tt, m, gives_check, ply);
            let value = if NT::PV {
                -self.search::<Pv>(tt, stop, -beta, -alpha, depth - 1, false, ply + 1)
            } else {
                -self.search::<NonPv>(tt, stop, -beta, -alpha, depth - 1, !cut_node, ply + 1)
            };
            self.undo_move(m);

            // Step 11 — stop check: the subtree result is untrusted
            // (§3.9). Return immediately — no best-move update, no PV
            // write; only `!stop`-guarded ID-loop sites consume results.
            if stop.load(Ordering::Relaxed) {
                return VALUE_ZERO;
            }

            // Step 12 — root bookkeeping (§3.4 per-move updates; the
            // equal-value dither joins with plan step 10's hygiene pass).
            if NT::ROOT {
                self.update_root_move(m, value, alpha, beta, move_count, self.nodes - nodes_before);
            }

            // Step 13 — best/alpha/beta updates.
            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = m;
                    if NT::PV {
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

        // Step 14 — node result: ply-relative mate/stalemate terminals
        // (D5): a longer mate scores closer to zero, so the search
        // naturally prefers the shortest mate and delays being mated.
        // (Fail-high softening joins with plan step 10's hygiene pass.)
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
            // copies pv_table[1]" invariant. In v1 every root child is a
            // full-window PV search, so the row is always fresh.
            rm.pv.truncate(1);
            rm.pv.extend_from_slice(self.pv_table[1].as_slice());
            // Best-move flip mid-iteration: the TM instability signal
            // (§3.4; consumed by doc 12 §3.9 in P7).
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

    /// Quiescence search (design/07 §3.8): stand pat on the static eval,
    /// captures only (queen promotions ride the CAPTURES stage) via
    /// `capture_stage` semantics, ALL evasions while in check, ply-relative
    /// mate when checkmated, and the `DEPTH_QS` TT discipline (§3.8 steps
    /// 4/5/12): probe at entry, eval reuse, the stand-pat fail-high
    /// eval-cache write, and a store at exit that is NEVER `Exact`. Still
    /// missing (plan step 9): the step-7 per-move rules (late-move cutoff,
    /// futility, SEE gates), the softening blends, the stand-pat tt-value
    /// substitution (deliberately deferred — at v1's all-PV nodes it would
    /// leak TT values into returned scores and break the step-6 TT-on/off
    /// value equality), and the narrow stalemate test.
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
        // (`>= DEPTH_QS`). Like the main search's step 7, unreachable
        // until the PVS zero-window creates NonPV qsearch nodes.
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
        // from being pruned into existence — and raw_eval stays VALUE_NONE
        // for the store.
        let mut best_value = -VALUE_INFINITE;
        let mut raw_eval = VALUE_NONE;
        if !in_check {
            raw_eval = if tt_hit && is_valid(tt_data.eval) {
                tt_data.eval
            } else {
                evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO)
            };
            self.stack[cur].static_eval = raw_eval; // corrected == raw (Phase A)
            best_value = raw_eval;
            if best_value >= beta {
                // Softening blend arrives with plan step 9. A stand-pat
                // fail high on a TT miss caches the raw eval as a
                // Bound::Lower eval-only entry (§3.11).
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

        let mut best_move = Move::NULL;
        let mut move_count = 0;
        for m in &list {
            if !self.board.legal(m) {
                continue;
            }
            move_count += 1;

            // Step 8 — recurse: full window, same node type.
            let gives_check = self.board.gives_check(m);
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

        // Step 10 — checkmate: every evasion was generated, so zero legal
        // moves while in check is mate, ply-relative (D5), returned
        // WITHOUT a store (SF parity). (The narrow stalemate test is plan
        // step 9.)
        if in_check && move_count == 0 {
            return mated_in(ply as i32);
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
    fn krk_mate_in_2_exact_score_at_depth_3() {
        // 1.Kb6 (only-move reply Kb8) 2.Rh8#: three plies, no mate in one.
        let w = go_depth("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3);
        assert_eq!(best_score(&w), mate_in(3));
        // The PV replays legally from the root position.
        replay_pv(&w);
    }

    #[test]
    fn kqk_mate_in_2_exact_score_at_depth_3() {
        // 1.Qf8+ Kh7 (forced) 2.Qg7#; no mate in one exists.
        let w = go_depth("7k/8/5K2/8/8/Q7/8/8 w - - 0 1", 3);
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
    fn two_rook_ladder_mate_in_3_exact_at_depth_5() {
        // 1.Rb6+ (rank 5 stays cut by Ra5) K→7th 2.Ra7+ K→8th 3.Rb8#:
        // five plies. The black king starts on g6, too far from the a/b
        // files to ever attack the unprotected checking rooks, and too
        // free for anything shorter than the ladder. Depth 5 is the §8.4
        // gate's "depth ≤ 6" for mate-in-3.
        let w = go_depth("8/8/6k1/R7/8/8/8/1R5K w - - 0 1", 5);
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
            // Depth = mate distance suffices (mate-in-1 uses 2 so the
            // score comes from a full search iteration, not bare qsearch).
            let w = go_depth(fen, plies.max(2));
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

    // --- Node counting and determinism (doc 07 §5, D24) ---

    #[test]
    fn search_is_deterministic_at_fixed_depth() {
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

    // --- Alpha-beta equivalence vs pure negamax (doc 07 §5) ---
    //
    // Fail-soft alpha-beta over a FIXED tree returns the exact minimax
    // value at a full-window root. The v1 tree IS fixed (no
    // window-dependent pruning anywhere: qsearch has no futility/SEE
    // rules yet, and the TT only reorders moves and caches raw evals —
    // never a returned value, since the sanctioned NonPV cutoffs are
    // unreachable without zero-window nodes), so a cutoff-free reference
    // walking the same tree — same generators, same stand-pat, same
    // terminal rules — must agree exactly with the final iteration
    // (earlier iterations only reorder the root, which cannot move an
    // exact full-window value). Positions are chosen with no reachable
    // repetitions so the node-counter draw dither cannot fire.

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

    #[test]
    fn alpha_beta_equals_pure_negamax_on_tiny_positions() {
        let cases = [
            // KRK mate-in-2: exercises mate-distance pruning soundness.
            ("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3),
            // Tiny pawn ending with captures in the qsearch tree.
            ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", 3),
            // Back-rank mate-in-1 with pawns on both sides.
            ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 2),
            // Minor-piece skirmish: black to move, hanging knight.
            ("4k3/8/3n4/8/4B3/8/8/4K3 b - - 0 1", 2),
        ];
        for (fen, depth) in cases {
            let w = go_depth(fen, depth);
            let want = ref_negamax(&mut board(fen), &mut EvalContext::new(), depth, 0);
            assert_eq!(
                best_score(&w),
                want,
                "alpha-beta vs pure negamax diverged: {fen}"
            );
        }
    }

    // --- TT-on/off value equality (doc 07 §4 step 6 done-condition; doc
    //     08 §8 criterion 6's value half) ---

    /// One position of the TT value-equality suite: search it three ways
    /// with the TT enabled — a fresh table, the same table warmed by that
    /// run, and a table warmed by a DEEPER run first (deeper entries are
    /// the classic source of value-changing TT cutoffs) — assert all
    /// three agree, and return the shared value.
    fn tt_value_triple(fen: &str, depth: Depth) -> Value {
        let tt = test_tt();
        let fresh = go_depth_on(fen, depth, &tt);
        let warm = go_depth_on(fen, depth, &tt);
        assert_eq!(
            best_score(&fresh),
            best_score(&warm),
            "fresh vs warm TT: {fen}"
        );
        let deep_tt = test_tt();
        go_depth_on(fen, depth + 1, &deep_tt);
        let after_deeper = go_depth_on(fen, depth, &deep_tt);
        assert_eq!(
            best_score(&fresh),
            best_score(&after_deeper),
            "fresh vs deeper-warmed TT: {fen}"
        );
        best_score(&fresh)
    }

    #[test]
    fn tt_on_and_off_return_identical_values_at_fixed_depth() {
        // The suite the step-6 exit bar demands: fixed-depth searches with
        // the TT disabled vs enabled return identical VALUES (node counts
        // may differ — ttMove ordering changes cutoff order, never the
        // exact full-window root value). `ref_negamax` — proven equal to
        // the pre-TT engine by the test above at the commit that added it
        // — is the TT-disabled oracle, compared against the fresh/warm/
        // deeper-warmed triple: equality proves the v1 bounds/cutoff
        // discipline — nothing the TT stores reaches a returned value.
        //
        // Depths stay <= 4: from a FEN root (no game history) the first
        // possible in-tree repetition is ply 4, which at these depths is
        // qsearch's plain deterministic VALUE_DRAW — the node-counter
        // draw dither is unreachable, so any divergence would be TT value
        // flow, exactly what must not exist. Positions keep the material
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
            assert_eq!(
                tt_value_triple(fen, depth),
                tt_off,
                "TT-on vs TT-off: {fen}"
            );
        }
    }

    #[test]
    fn tt_state_never_moves_the_startpos_value() {
        // Full-material coverage for the suite above, without the
        // exponential no-cutoff oracle: the fresh/warm/deeper-warmed
        // triple must agree on the startpos value — any inequality would
        // be TT state leaking into returned values.
        tt_value_triple(START_FEN, 3);
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
