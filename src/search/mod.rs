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
//! (`depth`/`nodes`/`movetime`), and the [`SearchInfo`] callback seam the
//! UCI layer turns into `info` lines (design/12 §3.13).
//!
//! Deliberately NOT here yet, per the doc's staging: the TT (§3.7.2 steps
//! 5–7, plan step 6), the PVS zero-window three-call structure (step 7),
//! aspiration windows (§3.6, step 8), the full qsearch rules (§3.8, step
//! 9), the hygiene pass (forgotten mate, aborted loss, `go mate n`,
//! equal-value dither — step 10), clock time management (doc 12 step 7,
//! P7), and every ⟨D10⟩ heuristic slot.

pub mod params;
pub mod rootmoves;
pub mod stack;

pub use rootmoves::{RootMove, RootMoves, build_root_moves, sort_descending};
pub use stack::{ContHistId, PvLine, SS_OFFSET, STACK_SIZE, Stack, init_stack, ss, update_pv};

use std::sync::atomic::{AtomicBool, Ordering};

use crate::board::Board;
use crate::eval::{EvalContext, evaluate};
use crate::movegen::{generate_captures, generate_evasions, generate_legal};
use crate::moves::{MAX_MOVES, Move, MoveKind, MoveList};
use crate::timeman::Limits;
use crate::types::{
    Depth, MAX_PLY, PieceType, VALUE_DRAW, VALUE_INFINITE, VALUE_ZERO, Value, mate_in, mated_in,
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
    pub fn iterative_deepening(&mut self, stop: &AtomicBool, on_info: &mut dyn FnMut(&SearchInfo)) {
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
    /// steps 2/3/4/9/10/11/12/13/14 — no TT (steps 5–7, plan step 6), no
    /// ⟨D10⟩ heuristics (step 8), no equal-value dither in step 12 (plan
    /// step 10). Step 10 is plain full-window negamax until the PVS
    /// three-call structure lands (plan step 7), so PV nodes only ever
    /// spawn PV children here — which also keeps `pv_table[1]` fresh for
    /// every root move's step-12 PV rebuild.
    fn search<NT: NodeType>(
        &mut self,
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
                self.qsearch::<Pv>(alpha, beta, ply)
            } else {
                self.qsearch::<NonPv>(alpha, beta, ply)
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

        // Step 9 — move loop. The staged MovePicker (doc 09) replaces
        // generate_legal + placeholder ordering; legality is then checked
        // per move instead of up front. (`excluded_move` guard arrives
        // with doc 10's singular search — nothing writes it yet.)
        //
        // At the root the loop iterates `root_moves[pv_idx..]` itself
        // (doc 07 §3.7.2 step 9's "skip moves outside the root list",
        // v1 form): the ID loop's stable sort has last iteration's best
        // in front, which is this phase's stand-in for the step-5 ttMove
        // override — the MovePicker restores picker order + skip when
        // doc 09 lands. Everything in `root_moves` is legal by
        // construction, and `pv_last = root_moves.len()` until MultiPV.
        let mut list = MoveList::new();
        if NT::ROOT {
            for rm in &self.root_moves[self.pv_idx..] {
                list.push(rm.pv[0]);
            }
        } else {
            generate_legal(&self.board, &mut list);
            order_moves(&self.board, &mut list);
        }

        let mut best_value = -VALUE_INFINITE;
        let mut move_count = 0;

        for m in &list {
            move_count += 1;
            self.stack[cur].move_count = move_count;
            let nodes_before = self.nodes; // root effort accounting (§3.4)

            // Step 10 — make + recurse, full-window negamax (see fn docs).
            let gives_check = self.board.gives_check(m);
            self.do_move(m, gives_check, ply);
            let value = if NT::PV {
                -self.search::<Pv>(stop, -beta, -alpha, depth - 1, false, ply + 1)
            } else {
                -self.search::<NonPv>(stop, -beta, -alpha, depth - 1, !cut_node, ply + 1)
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

            // Step 13 — best/alpha/beta updates. (Best-move tracking
            // lands with the TT store, plan step 6.)
            if value > best_value {
                best_value = value;
                if value > alpha {
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
        // Fail-high softening, ttPv back-propagation and the TT store are
        // plan steps 10/6.
        if move_count == 0 {
            best_value = if in_check {
                mated_in(ply as i32)
            } else {
                VALUE_DRAW
            };
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

    /// Quiescence search (design/07 §3.8), step-4 skeleton: stand pat on
    /// the static eval, captures only (queen promotions ride the CAPTURES
    /// stage) via `capture_stage` semantics, ALL evasions while in check,
    /// ply-relative mate when checkmated. No TT (step 4 of §3.8), none of
    /// the step-7 per-move rules (late-move cutoff, futility, SEE gates),
    /// no softening blends, no narrow stalemate test — all plan step 9/6.
    ///
    /// No stop poll here (§3.9, SF parity): qsearch is bounded by the
    /// main-search checks above it.
    fn qsearch<NT: NodeType>(&mut self, mut alpha: Value, beta: Value, ply: usize) -> Value {
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

        // Step 5 — stand pat (not in check). In check there is no stand
        // pat: best_value stays at -VALUE_INFINITE (a loss) so nothing
        // can cut before at least one evasion is searched — the evasion
        // guard that keeps checkmates from being pruned into existence.
        let mut best_value = -VALUE_INFINITE;
        if !in_check {
            let static_eval = evaluate(&self.board, &mut self.eval_ctx, VALUE_ZERO);
            self.stack[cur].static_eval = static_eval;
            best_value = static_eval;
            if best_value >= beta {
                return best_value; // softening blend arrives with plan step 9
            }
            alpha = alpha.max(best_value);
        }

        // Steps 6–7 — captures (+ queen promotions, the CAPTURES stage of
        // doc 05 §3.2, mirroring `Board::capture_stage`), or every evasion
        // in check. Pseudo-legal generation; legality tested on the move
        // we are about to play.
        let mut list = MoveList::new();
        if in_check {
            generate_evasions(&self.board, &mut list);
        } else {
            generate_captures(&self.board, &mut list);
        }
        order_moves(&self.board, &mut list);

        let mut move_count = 0;
        for m in &list {
            if !self.board.legal(m) {
                continue;
            }
            move_count += 1;

            // Step 8 — recurse: full window, same node type.
            let gives_check = self.board.gives_check(m);
            self.do_move(m, gives_check, ply);
            let value = -self.qsearch::<NT>(-beta, -alpha, ply + 1);
            self.undo_move(m);

            // Step 9 — updates.
            if value > best_value {
                best_value = value;
                if value > alpha {
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
        // moves while in check is mate, ply-relative (D5). (The narrow
        // stalemate test is plan step 9.)
        if in_check && move_count == 0 {
            return mated_in(ply as i32);
        }

        debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE);
        best_value
    }

    /// Make `m`, counting the node (§3.9 — `nodes` counts MADE moves,
    /// matching SF and the future `bench` fingerprint) and installing the
    /// child's stack slots. The continuation-history key install (keyed by
    /// `(in_check, capture, piece, to)`) and the TT prefetch hook land
    /// with docs 09/08; the sentinel is always safe to read through.
    #[inline]
    fn do_move(&mut self, m: Move, gives_check: bool, ply: usize) {
        self.nodes += 1;
        let cur = ss(ply);
        self.stack[cur].current_move = m;
        self.stack[cur].cont_hist = ContHistId::SENTINEL;
        self.board.make_move(m, gives_check);
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
    /// `TimeManager::elapsed_ms` semantics) and `nodes` (single worker, so
    /// `self.nodes` IS the total until SMP aggregation lands, D23).
    /// `go depth` is enforced by the ID loop, `go infinite` never
    /// self-stops. The clock-TM wall (`elapsed > maximum`) and the ponder
    /// guard arrive with doc 12 steps 7/9 (P7+).
    #[inline]
    fn check_time(&mut self, stop: &AtomicBool) {
        self.check_time_countdown -= 1;
        if self.check_time_countdown > 0 {
            return;
        }
        self.check_time_countdown = self.check_time_reset();
        let movetime_stop = self
            .limits
            .movetime
            .is_some_and(|t| self.limits.start_time.elapsed().as_millis() as i64 >= t);
        let nodes_stop = self.limits.nodes.is_some_and(|n| self.nodes >= n);
        if movetime_stop || nodes_stop {
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

/// MVV-LVA placeholder ordering until doc 09's staged MovePicker lands
/// (design/07 §4 step 4): captures sort first by most valuable victim,
/// ties broken by least valuable attacker; promotions add the promoted
/// piece so queen promotions sort with queen captures; quiets keep
/// generation order behind all captures.
///
/// Insertion sort: stable and in-place (no allocation on the hot path,
/// §6), so node counts are deterministic (D24). O(n²) worst case is fine
/// for n ≤ 256 in a placeholder the MovePicker deletes.
fn order_moves(board: &Board, list: &mut MoveList) {
    let mut scores = [0i32; MAX_MOVES];
    for (s, &m) in scores.iter_mut().zip(list.iter()) {
        *s = move_score(board, m);
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

    /// Run a `go depth <depth>` through the iterative-deepening driver,
    /// discarding info reports.
    fn go_depth(fen: &str, depth: Depth) -> Worker {
        let mut w = Worker::new(board(fen));
        w.limits.depth = Some(depth);
        let stop = AtomicBool::new(false);
        w.iterative_deepening(&stop, &mut |_| {});
        w
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
        w.iterative_deepening(&stop, &mut |_| infos += 1);
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
        w.iterative_deepening(&stop, &mut |i| {
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
        w.iterative_deepening(&stop, &mut |_| {});
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
        w.iterative_deepening(&stop, &mut |_| {});
        assert!(stop.load(Ordering::Relaxed), "movetime must self-stop");
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
        w.iterative_deepening(&stop, &mut |i| pv_heads.push(i.pv[0]));
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
        w.iterative_deepening(&stop, &mut |_| {});
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
    // rules yet), so a cutoff-free reference walking the same tree —
    // same generators, same stand-pat, same terminal rules — must agree
    // exactly with the final iteration (earlier iterations only reorder
    // the root, which cannot move an exact full-window value). Positions
    // are chosen with no reachable repetitions so the node-counter draw
    // dither cannot fire.

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
}
