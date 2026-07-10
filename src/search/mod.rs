//! Search — iterative deepening driver, PVS nodes, quiescence
//! (`design/07-search-core.md`) and the pruning/extension heuristics
//! (`design/10-pruning-extensions.md`).
//!
//! Landed so far (design/07 §4 steps 1–4): the [`params`] constants module
//! (P0, design/01 §4 step 4), the per-ply search [`stack`] with PV lines
//! (§3.2–§3.3, D18), [`rootmoves`] bookkeeping (§3.4), and the naked
//! negamax + qsearch skeleton in this file: [`Worker`], the [`NodeType`]
//! markers (§3.7.1), `search`/`qsearch` with the step-4 node subset —
//! draw/max-ply handling with the [`value_draw`] dither, mate-distance
//! pruning, full-window move loop over `generate_legal` with MVV-LVA
//! placeholder ordering, PV updates, ply-relative mate/stalemate terminals
//! (D5), node counting and the stop-flag seam (§3.9).
//!
//! Deliberately NOT here yet, per the doc's staging: the TT (§3.7.2 steps
//! 5–7, plan step 6), the PVS zero-window three-call structure (step 7),
//! aspiration windows (§3.6, step 8), the full qsearch rules (§3.8, step
//! 9), the iterative-deepening driver + root bookkeeping (§3.5, step 5),
//! and every ⟨D10⟩ heuristic slot.

pub mod params;
pub mod rootmoves;
pub mod stack;

pub use rootmoves::{RootMove, RootMoves, sort_descending};
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
    /// Per-thread eval state (design/11 §3.1); empty in Phase A.
    eval_ctx: EvalContext,
    /// Countdown to the next real `check_time` (§3.9): reset to
    /// [`params::CHECK_TIME_CALLS`].
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
            eval_ctx: EvalContext::new(),
            check_time_countdown: params::CHECK_TIME_CALLS,
        }
    }

    /// One fixed-depth, full-window search from the root — the pre-ID
    /// skeleton entry (design/07 §4 step 4). The iterative-deepening
    /// driver (§3.5, plan step 5) subsumes this: it will own the stack
    /// init and per-iteration bookkeeping and call `search::<Root>` per
    /// aspiration attempt. Until then, tests and `go depth n` drive this.
    ///
    /// `stop` is the §3.9 stop seam: the UCI layer flips it from another
    /// thread; a search that observes it returns untrusted values (the
    /// consumers must be `!stop`-guarded, §3.5). The flag rides the call
    /// stack as a shared borrow — the Rust-safe form of SF's global.
    ///
    /// `nodes` accumulates across calls (per-`go` reset is the driver's
    /// job); the returned value is root-relative, mate scores ply-relative
    /// from the root (D5).
    pub fn search_root(&mut self, depth: Depth, stop: &AtomicBool) -> Value {
        init_stack(&mut self.stack);
        self.sel_depth = 0;
        self.root_depth = depth.max(1);
        // Full window until aspiration lands (§3.6, plan step 8).
        self.root_delta = 2 * VALUE_INFINITE;
        self.check_time_countdown = params::CHECK_TIME_CALLS;
        self.search::<Root>(
            stop,
            -VALUE_INFINITE,
            VALUE_INFINITE,
            self.root_depth,
            false,
            0,
        )
    }

    /// The principal variation of the last completed root search
    /// (`pv_table[0]`; empty if the root had no legal moves).
    pub fn root_pv(&self) -> &[Move] {
        self.pv_table[0].as_slice()
    }

    /// The best root move of the last completed search, if any.
    pub fn best_move(&self) -> Option<Move> {
        self.root_pv().first().copied()
    }

    /// The main search node (design/07 §3.7.2), step-4 subset: node order
    /// steps 2/3/4/9/10/13/14 — no TT (steps 5–7, plan step 6), no ⟨D10⟩
    /// heuristics (step 8), no root bookkeeping (step 12, plan step 5).
    /// Step 10 is plain full-window negamax until the PVS three-call
    /// structure lands (plan step 7), so PV nodes only ever spawn PV
    /// children here.
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
        self.check_time();

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
        let mut list = MoveList::new();
        generate_legal(&self.board, &mut list);
        order_moves(&self.board, &mut list);

        let mut best_value = -VALUE_INFINITE;
        let mut move_count = 0;

        for m in &list {
            move_count += 1;
            self.stack[cur].move_count = move_count;

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

            // Step 13 — best/alpha/beta updates. (Step 12's root
            // bookkeeping and equal-value dither land with plan step 5;
            // best-move tracking lands with the TT store, plan step 6.)
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
    /// [`params::CHECK_TIME_CALLS`] calls. The real work — elapsed-time
    /// and limit checks that raise the stop flag — belongs to the
    /// time-management layer (design/12) and lands with the UCI wiring
    /// (plan step 5); the call site and cadence are fixed here so that
    /// wiring cannot move them.
    #[inline]
    fn check_time(&mut self) {
        self.check_time_countdown -= 1;
        if self.check_time_countdown > 0 {
            return;
        }
        self.check_time_countdown = params::CHECK_TIME_CALLS;
        // TM real work goes here (design/12): elapsed vs limits → stop.
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
    use crate::types::VALUE_MATE;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    fn go(fen: &str, depth: Depth) -> (Worker, Value) {
        let mut w = Worker::new(board(fen));
        let stop = AtomicBool::new(false);
        let v = w.search_root(depth, &stop);
        (w, v)
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

    // --- Mate suite (doc 07 §5): exact ply-relative mate scores (D5) ---

    #[test]
    fn back_rank_mate_in_1_found_at_depth_2() {
        // Rd8# is the only mate (and the only check). Expect the exact
        // ply-relative score mate_in(1) and the mating move as the PV.
        let fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
        for depth in [2, 3, 4] {
            let (w, v) = go(fen, depth);
            assert_eq!(v, mate_in(1), "depth {depth}");
            assert_eq!(w.best_move().unwrap().to_string(), "d1d8");
        }
    }

    #[test]
    fn krk_mate_in_2_exact_score_at_depth_3() {
        // 1.Kb6 (only-move reply Kb8) 2.Rh8#: three plies, no mate in one.
        let (w, v) = go("k7/8/2K5/8/8/8/8/7R w - - 0 1", 3);
        assert_eq!(v, mate_in(3));
        // The PV replays legally from the root position.
        assert_pv_legal(&w);
    }

    #[test]
    fn kqk_mate_in_2_exact_score_at_depth_3() {
        // 1.Qf8+ Kh7 (forced) 2.Qg7#; no mate in one exists.
        let (w, v) = go("7k/8/5K2/8/8/Q7/8/8 w - - 0 1", 3);
        assert_eq!(v, mate_in(3));
        assert_pv_legal(&w);
    }

    #[test]
    fn deeper_search_keeps_the_exact_mate_score() {
        // Ply-relative scoring must not drift when the horizon extends
        // past the mate (doc 07 §3.1).
        let (_, v) = go("7k/8/5K2/8/8/Q7/8/8 w - - 0 1", 5);
        assert_eq!(v, mate_in(3));
    }

    // --- Terminal nodes at the root ---

    #[test]
    fn checkmated_root_reports_mated_in_0() {
        // Fool's mate: white to move, checkmated.
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        let (w, v) = go(fen, 3);
        assert_eq!(v, mated_in(0));
        assert_eq!(v, -VALUE_MATE);
        assert!(w.root_pv().is_empty());
    }

    #[test]
    fn stalemate_returns_exact_draw() {
        // Black to move: Ka8 has no moves (Kb6 covers a7/b7, Qh2 covers
        // b8 on the h2-b8 diagonal) and is not in check.
        let fen = "k7/8/1K6/8/8/8/7Q/8 b - - 0 1";
        for depth in [1, 4] {
            let (w, v) = go(fen, depth);
            assert_eq!(v, VALUE_DRAW, "depth {depth}");
            assert!(w.root_pv().is_empty());
        }
    }

    // --- Basic sanity on the startpos ---

    #[test]
    fn startpos_returns_a_legal_pv() {
        let (w, v) = go(START_FEN, 4);
        assert!(!crate::types::is_decisive(v), "startpos is not decisive");
        assert!(w.nodes > 0);
        assert!(w.sel_depth >= 4, "PV nodes must bump selDepth");
        assert_pv_legal(&w);
    }

    /// Replay the root PV move by move, asserting each is legal in turn —
    /// the "PVs that replay legally" bar of design/07 §4.
    fn assert_pv_legal(w: &Worker) {
        assert!(!w.root_pv().is_empty());
        let mut b = w.board.clone();
        for &m in w.root_pv() {
            let mut legal = MoveList::new();
            generate_legal(&b, &mut legal);
            assert!(legal.contains(&m), "PV move {m} illegal in {}", b.to_fen());
            b.make_move(m, b.gives_check(m));
        }
    }

    // --- Node counting and determinism (doc 07 §5, D24) ---

    #[test]
    fn node_counts_are_deterministic_at_fixed_depth() {
        let run = || {
            let (w, v) = go(START_FEN, 4);
            (w.nodes, v)
        };
        let (n1, v1) = run();
        let (n2, v2) = run();
        assert!(n1 > 0);
        assert_eq!(n1, n2, "two identical searches must count identical nodes");
        assert_eq!(v1, v2);
    }

    // --- Stop-flag seam (§3.9): untrusted return, no tree walked ---

    #[test]
    fn preset_stop_returns_untrusted_zero_after_one_node() {
        let mut w = Worker::new(board(START_FEN));
        let stop = AtomicBool::new(true);
        let v = w.search_root(6, &stop);
        assert_eq!(v, VALUE_ZERO, "stopped searches return VALUE_ZERO");
        // The root made exactly one move; its child aborted at entry.
        assert_eq!(w.nodes, 1);
        assert!(w.root_pv().is_empty(), "no PV write after a stop");
    }

    // --- Alpha-beta equivalence vs pure negamax (doc 07 §5) ---
    //
    // Fail-soft alpha-beta over a FIXED tree returns the exact minimax
    // value at a full-window root. The step-4 tree IS fixed (no
    // window-dependent pruning anywhere: qsearch has no futility/SEE
    // rules yet), so a cutoff-free reference walking the same tree —
    // same generators, same stand-pat, same terminal rules — must agree
    // exactly. Positions are chosen with no reachable repetitions so the
    // node-counter draw dither cannot fire.

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
            let (_, v) = go(fen, depth);
            let want = ref_negamax(&mut board(fen), &mut EvalContext::new(), depth, 0);
            assert_eq!(v, want, "alpha-beta vs pure negamax diverged: {fen}");
        }
    }
}
