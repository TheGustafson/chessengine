//! The SOTA net's **incremental FullThreats accumulator** — the ≤80-events/move
//! dirty-diff observer (design/11 §12.9.1, D40).
//!
//! FullThreats is a *relational* feature set: a feature fires for the directed
//! relation "attacker on `from` threatens victim on `to`". A single move can
//! ripple into dozens of such relations (every slider x-rayed through a vacated
//! or newly-occupied square, every short-range piece adjacent to either
//! endpoint), so unlike the ≤2-add/≤2-sub HalfKA update this observer emits a
//! *raw event list* per move, resolved to feature indices per perspective at
//! consumption time via the very same [`super::sota_features::threat_index`] the
//! from-scratch [`super::sota_features::refresh_threat`] uses.
//!
//! # Structure (Reckless) / content (Stockfish)
//! The event-generation *shape* — the `record_square_touch` primitive with its
//! three parts (outgoing / x-ray ripple / incoming) and the three drivers
//! (`on_move`/`on_change`/`on_mutate`) — is adapted from Reckless
//! (`reckless_reference/`, AGPL, D31: ideas only, never copied). The *content* —
//! which relations exist — is Stockfish's, and is pinned to be **bit-identical**
//! to our from-scratch enumeration: kings are excluded as attacker and victim (a
//! king move contributes only the x-ray ripple, its blocker role), and the
//! pawn-push relation is included but only against a pawn directly in front
//! (matching `for_each_threat`). We may freely *over-generate* raw events that
//! [`threat_index`](super::sota_features::threat_index) drops to `None`
//! (king edges, pawn-diagonal onto a bishop/queen, …); the one place
//! over-generation would be *unsafe* — a pawn push onto a knight/rook, which
//! `threat_index` would keep but the from-scratch path never emits — is handled
//! explicitly (pawn pushes emit only against pawns).
//!
//! # This increment: the first-cut, standalone
//! Per §12.9.1 this lands the **simpler first-cut** driver ("`on_change` every
//! `DirtyPiece` sub — removes first — then every add; auto-cancels, costs extra
//! events") rather than the optimized 3-primitive `on_move` form. It is a
//! `ThreatContext` the tests drive directly; it is **not** wired into
//! `Worker::do_move` / the search hot path yet (a later increment), so the
//! default build is untouched and the depth-6 bench stays 548978.
//!
//! ## Why the first-cut is correct on a post-move board (the telescoping)
//! Reckless calls its hooks *during* make-move, at each intermediate
//! single-square board state. We instead reconstruct those intermediate states
//! from the **post-move board plus the [`DirtyPiece`]**: we rebuild the pre-move
//! occupancy, then replay the edits (remove every sub, then add every add),
//! carrying a running occupancy bitboard and a tiny per-touched-square identity
//! override so each `record_square_touch` sees exactly the occupancy and piece
//! identities of the state *after its own edit* — precisely what Reckless's
//! in-flight board would show. Each single-square edit's touch emits exactly the
//! raw relations that change between `O_{i-1}` and `O_i` (which differ only at
//! that square). Summed over an edit path pre → post, the signed events
//! telescope to `raw_edges(after) − raw_edges(before)`, and resolving each event
//! through `threat_index` telescopes the *feature indices* to
//! `active(after) − active(before)` — i.e. the incremental accumulator equals a
//! from-scratch refresh. The correctness gate (`threats_replay_*`) asserts
//! exactly this at every node of thousands of random games.
//!
//! # Mirror refresh
//! A threat index depends on the perspective's king only through its **mirror
//! half** (files a–d vs e–h; [`orient_threats`](super::sota_features)), never the
//! exact square or a bucket. So a perspective's threat accumulator only needs a
//! full rebuild when *its own* king crosses the d/e boundary — a strict subset
//! of the HalfKA refresh trigger. We detect that per perspective and
//! [`refresh`](ThreatContext::refresh)-rebuild only the crossing side; the other
//! side (and every non-king move) stays incremental.

use super::sota_features::{ACC, refresh_threat, threat_index};
use super::sota_simd::apply_i8_deltas;
use crate::attacks::{
    attacks, bishop_attacks, knight_attacks, pawn_attacks, ray_pass, rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, Piece, PieceType, Square};

/// One perspective-agnostic threat event: "`attacker` on `from` threatens
/// `victim` on `to`", appearing (`add`) or disappearing (`!add`). Resolved to a
/// feature index per perspective by [`threat_index`]; excluded pairs drop out.
#[derive(Debug, Clone, Copy)]
pub struct ThreatDelta {
    pub attacker: Piece,
    pub from: Square,
    pub victim: Piece,
    pub to: Square,
    pub add: bool,
}

/// Raw-event buffer capacity. The optimized 3-primitive form fits the
/// `(8+16)·3+8 = 80` bound; the first-cut "costs extra events" (up to four
/// `on_change` touches for castling, each ≤ ~36 events), so we size generously
/// and never rely on the tight bound holding here.
const DELTA_CAP: usize = 256;

/// A fixed, heap-free scratch buffer of raw threat events (design/11 §12.9.1).
struct DeltaBuf {
    data: [ThreatDelta; DELTA_CAP],
    len: usize,
}

impl DeltaBuf {
    fn new() -> Self {
        let placeholder = ThreatDelta {
            attacker: Piece::new(Color::White, PieceType::Pawn),
            from: Square::A1,
            victim: Piece::new(Color::White, PieceType::Pawn),
            to: Square::A1,
            add: false,
        };
        DeltaBuf {
            data: [placeholder; DELTA_CAP],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, d: ThreatDelta) {
        debug_assert!(self.len < DELTA_CAP, "threat delta buffer overflow");
        self.data[self.len] = d;
        self.len += 1;
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn events(&self) -> &[ThreatDelta] {
        &self.data[..self.len]
    }
}

/// The intermediate replay state: the running occupancy plus a small
/// override map for the few squares the move touches, so identity lookups match
/// the state *after the current edit* even though the underlying [`Board`] is
/// already fully post-move.
struct Replay<'a> {
    board: &'a Board,
    /// Current (intermediate) occupancy — starts at the pre-move occupancy and
    /// evolves one edit at a time.
    occ: Bitboard,
    /// Touched squares and their *current* occupant (`None` == empty). At most a
    /// handful per move (castling: 4); padded.
    over_sq: [Square; 8],
    over_pc: [Option<Piece>; 8],
    n_over: usize,
}

impl<'a> Replay<'a> {
    fn find(&self, sq: Square) -> Option<usize> {
        self.over_sq[..self.n_over].iter().position(|&s| s == sq)
    }

    /// Set the current occupant of a touched square, inserting a slot on first
    /// touch. Keeps `occ` in lockstep.
    fn set(&mut self, sq: Square, pc: Option<Piece>) {
        let i = self.find(sq).unwrap_or_else(|| {
            let i = self.n_over;
            self.over_sq[i] = sq;
            self.n_over += 1;
            i
        });
        self.over_pc[i] = pc;
        self.occ = match pc {
            Some(_) => self.occ | Bitboard::from_square(sq),
            None => self.occ & !Bitboard::from_square(sq),
        };
    }

    /// The piece currently on `sq`. `sq` must be occupied in [`Self::occ`]
    /// (callers only resolve occupied squares).
    #[inline]
    fn resolve(&self, sq: Square) -> Piece {
        match self.find(sq) {
            Some(i) => self.over_pc[i].expect("resolve on an empty touched square"),
            None => self
                .board
                .piece_on(sq)
                .expect("resolve on an empty untouched square"),
        }
    }
}

/// Emit every threat relation a piece `p` conceptually appearing (`add`) or
/// disappearing (`!add`) at `s` participates in, against the replay's current
/// occupancy (design/11 §12.9.1). Over-generates freely; [`threat_index`] drops
/// the excluded pairs at resolution time. The one non-over-generatable case —
/// pawn pushes onto a non-pawn — is guarded here.
fn record_square_touch(buf: &mut DeltaBuf, r: &Replay, p: Piece, s: Square, add: bool) {
    let occ = r.occ;

    // --- 1. Outgoing: what `p` threatens from `s`. Kings are never attackers.
    match p.piece_type {
        PieceType::King => {}
        PieceType::Pawn => {
            // Diagonal captures (any occupant; threat_index filters to P/N/R).
            for to in pawn_attacks(p.color, s) & occ {
                buf.push(ThreatDelta {
                    attacker: p,
                    from: s,
                    victim: r.resolve(to),
                    to,
                    add,
                });
            }
            // Push relation: only a pawn directly in front is threatened
            // (matching for_each_threat's `all_pawns` guard — MAP would
            // otherwise wrongly keep pawn-push onto a knight/rook).
            if let Some(push) = s.shifted(p.color.pawn_push())
                && occ.contains(push)
            {
                let victim = r.resolve(push);
                if victim.piece_type == PieceType::Pawn {
                    buf.push(ThreatDelta {
                        attacker: p,
                        from: s,
                        victim,
                        to: push,
                        add,
                    });
                }
            }
        }
        pt => {
            for to in attacks(pt, s, occ) & occ {
                buf.push(ThreatDelta {
                    attacker: p,
                    from: s,
                    victim: r.resolve(to),
                    to,
                    add,
                });
            }
        }
    }

    // --- 2. X-ray ripple: sliders whose ray reaches `s`. Each such slider `sl`
    // gains/loses its direct relation to `p@s`, and its relation to the first
    // occupant strictly beyond `s` flips the *other* way (removing p unblocks
    // sl; adding p blocks it). Slider identity comes from `resolve`, not the
    // board's post-move bitboards, so a moved slider is never miscounted.
    let ratt = rook_attacks(s, occ);
    let batt = bishop_attacks(s, occ);
    let qatt = ratt | batt;
    for sl_sq in ratt & occ {
        let sl = r.resolve(sl_sq);
        if matches!(sl.piece_type, PieceType::Rook | PieceType::Queen) {
            ripple(buf, r, qatt, sl, sl_sq, p, s, add);
        }
    }
    for sl_sq in batt & occ {
        let sl = r.resolve(sl_sq);
        if matches!(sl.piece_type, PieceType::Bishop | PieceType::Queen) {
            ripple(buf, r, qatt, sl, sl_sq, p, s, add);
        }
    }

    // --- 3. Incoming: short-range attackers of `s` (pawns, knights). Kings are
    // never attackers, so no king incoming. Only meaningful when `p` can be a
    // victim, i.e. not a king.
    if p.piece_type != PieceType::King {
        for a in knight_attacks(s) & occ {
            if r.resolve(a).piece_type == PieceType::Knight {
                buf.push(ThreatDelta {
                    attacker: r.resolve(a),
                    from: a,
                    victim: p,
                    to: s,
                    add,
                });
            }
        }
        // A white pawn attacking `s` sits on a down-diagonal of `s`
        // (`pawn_attacks(Black, s)`), a black pawn on an up-diagonal.
        for a in pawn_attacks(Color::Black, s) & occ {
            let ap = r.resolve(a);
            if ap.color == Color::White && ap.piece_type == PieceType::Pawn {
                buf.push(ThreatDelta {
                    attacker: ap,
                    from: a,
                    victim: p,
                    to: s,
                    add,
                });
            }
        }
        for a in pawn_attacks(Color::White, s) & occ {
            let ap = r.resolve(a);
            if ap.color == Color::Black && ap.piece_type == PieceType::Pawn {
                buf.push(ThreatDelta {
                    attacker: ap,
                    from: a,
                    victim: p,
                    to: s,
                    add,
                });
            }
        }
        // Incoming pawn push: a pawn directly behind `s` threatens it — but only
        // if `p@s` is itself a pawn (the push relation is pawn→pawn only).
        if p.piece_type == PieceType::Pawn {
            for c in Color::ALL {
                if let Some(back) = s.shifted(c.flip().pawn_push())
                    && occ.contains(back)
                {
                    let ap = r.resolve(back);
                    if ap.color == c && ap.piece_type == PieceType::Pawn {
                        buf.push(ThreatDelta {
                            attacker: ap,
                            from: back,
                            victim: p,
                            to: s,
                            add,
                        });
                    }
                }
            }
        }
    }
}

/// One slider's ripple through `s`: its direct relation to `p@s` (sign `add`),
/// and its relation to the discovered piece just beyond `s` on the `sl → s` ray
/// (sign `!add`). At most one discovered piece — `ray_pass(sl, s) & occ & qatt`
/// is the classic "one blocker beyond" query (design/11 §12.9.1; `ray_pass` at
/// src/attacks.rs).
#[inline]
#[allow(clippy::too_many_arguments)]
fn ripple(
    buf: &mut DeltaBuf,
    r: &Replay,
    qatt: Bitboard,
    sl: Piece,
    sl_sq: Square,
    p: Piece,
    s: Square,
    add: bool,
) {
    let beyond = ray_pass(sl_sq, s) & r.occ & qatt;
    if let Some(d) = beyond.lsb() {
        buf.push(ThreatDelta {
            attacker: sl,
            from: sl_sq,
            victim: r.resolve(d),
            to: d,
            add: !add,
        });
    }
    buf.push(ThreatDelta {
        attacker: sl,
        from: sl_sq,
        victim: p,
        to: s,
        add,
    });
}

/// Occupancy-preserving identity replacement at `square` (capture placement or
/// promotion). Only relations whose attacker or victim is that square change;
/// x-rays beyond it do not. This is the compact counterpart to a remove+add
/// pair, which needlessly emits and later cancels those unchanged x-rays.
fn record_mutate(buf: &mut DeltaBuf, r: &Replay, old: Piece, new: Piece, square: Square) {
    record_outgoing(buf, r, old, square, false);
    record_outgoing(buf, r, new, square, true);
    record_incoming_mutate(buf, r, old, new, square);
}

fn record_outgoing(buf: &mut DeltaBuf, r: &Replay, piece: Piece, from: Square, add: bool) {
    if piece.piece_type == PieceType::King {
        return;
    }
    let occ = r.occ;
    if piece.piece_type == PieceType::Pawn {
        for to in pawn_attacks(piece.color, from) & occ {
            buf.push(ThreatDelta {
                attacker: piece,
                from,
                victim: r.resolve(to),
                to,
                add,
            });
        }
        if let Some(to) = from.shifted(piece.color.pawn_push())
            && occ.contains(to)
        {
            let victim = r.resolve(to);
            if victim.piece_type == PieceType::Pawn {
                buf.push(ThreatDelta {
                    attacker: piece,
                    from,
                    victim,
                    to,
                    add,
                });
            }
        }
    } else {
        for to in attacks(piece.piece_type, from, occ) & occ {
            buf.push(ThreatDelta {
                attacker: piece,
                from,
                victim: r.resolve(to),
                to,
                add,
            });
        }
    }
}

fn record_incoming_mutate(buf: &mut DeltaBuf, r: &Replay, old: Piece, new: Piece, to: Square) {
    let occ = r.occ;
    let rookers = rook_attacks(to, occ) & occ;
    let bishops = bishop_attacks(to, occ) & occ;
    let knights = knight_attacks(to) & occ;
    let white_pawns = pawn_attacks(Color::Black, to) & occ;
    let black_pawns = pawn_attacks(Color::White, to) & occ;
    for from in rookers | bishops | knights | white_pawns | black_pawns {
        let attacker = r.resolve(from);
        let attacks_to = match attacker.piece_type {
            PieceType::Rook => rookers.contains(from),
            PieceType::Bishop => bishops.contains(from),
            PieceType::Queen => rookers.contains(from) || bishops.contains(from),
            PieceType::Knight => knights.contains(from),
            PieceType::Pawn => {
                (attacker.color == Color::White && white_pawns.contains(from))
                    || (attacker.color == Color::Black && black_pawns.contains(from))
            }
            PieceType::King => false,
        };
        if attacks_to {
            buf.push(ThreatDelta {
                attacker,
                from,
                victim: old,
                to,
                add: false,
            });
            buf.push(ThreatDelta {
                attacker,
                from,
                victim: new,
                to,
                add: true,
            });
        }
    }
    // The non-capture pawn-push relation exists only when the victim is a pawn.
    for color in Color::ALL {
        if let Some(from) = to.shifted(color.flip().pawn_push())
            && occ.contains(from)
        {
            let attacker = r.resolve(from);
            if attacker.color == color && attacker.piece_type == PieceType::Pawn {
                if old.piece_type == PieceType::Pawn {
                    buf.push(ThreatDelta {
                        attacker,
                        from,
                        victim: old,
                        to,
                        add: false,
                    });
                }
                if new.piece_type == PieceType::Pawn {
                    buf.push(ThreatDelta {
                        attacker,
                        from,
                        victim: new,
                        to,
                        add: true,
                    });
                }
            }
        }
    }
}

/// Does `color`'s king cross the d/e mirror boundary in this move? (If so,
/// `color`'s threat accumulator must be rebuilt from scratch.)
fn king_crossed(dp: &DirtyPiece, color: Color) -> bool {
    let king = Piece::new(color, PieceType::King);
    let mut from = None;
    let mut to = None;
    for &(pc, sq) in dp.subs() {
        if pc == king {
            from = Some(sq);
        }
    }
    for &(pc, sq) in dp.adds() {
        if pc == king {
            to = Some(sq);
        }
    }
    match (from, to) {
        // orient_threats splits on file <= 3 (a–d) vs >= 4 (e–h).
        (Some(f), Some(t)) => (f.file() <= 3) != (t.file() <= 3),
        _ => false,
    }
}

/// A per-ply stack of FullThreats accumulators, maintained incrementally
/// (design/11 §12.9.1). Format-agnostic: it borrows the `i8` threat-weight slice
/// (feature-major, `THREAT_DIMS · ACC`), exactly like
/// [`refresh_threat`]. Standalone this increment — tests drive it directly; it
/// is not yet attached to the search's make/unmake seam.
pub struct ThreatContext<'a> {
    weights: &'a [i8],
    /// One `[White, Black]` accumulator pair per ply; `stack[0]` is the root.
    stack: Vec<[[i16; ACC]; 2]>,
    buf: DeltaBuf,
    /// Persistent scratch for the resolved add/sub feature indices in
    /// [`apply`](Self::apply). Only the `..len` prefix is ever read, so the
    /// storage must never be re-zeroed — declaring these inside the per-move
    /// loop cost an ~8 KB stack memset per made move (LLVM cannot elide it
    /// against a runtime-length read).
    adds: [usize; DELTA_CAP],
    subs: [usize; DELTA_CAP],
}

impl<'a> ThreatContext<'a> {
    /// A fresh context over `weights` (the `i8` FullThreats weight slice). Call
    /// [`refresh`](Self::refresh) to seed the root before any [`apply`](Self::apply).
    pub fn new(weights: &'a [i8]) -> Self {
        ThreatContext {
            weights,
            stack: Vec::with_capacity(64),
            buf: DeltaBuf::new(),
            adds: [0; DELTA_CAP],
            subs: [0; DELTA_CAP],
        }
    }

    /// Rebuild the root accumulator from scratch for `board`, both perspectives,
    /// and reset the stack to that single entry. The from-scratch build reuses
    /// [`refresh_threat`] so the incremental and refresh paths share one
    /// definition of the feature set.
    pub fn refresh(&mut self, board: &Board) {
        let mut root = [[0i16; ACC]; 2];
        // CENN-v2 explicitly supports HalfKA-only networks.  Their threat
        // transformer is represented by an empty slice and contributes the
        // all-zero accumulator; never index that empty slice through the
        // ordinary FullThreats refresh path.
        if !self.weights.is_empty() {
            for (i, &persp) in Color::ALL.iter().enumerate() {
                root[i] = refresh_threat(self.weights, board, persp);
            }
        }
        self.stack.clear();
        self.stack.push(root);
    }

    /// The current (top-of-stack) accumulator pair, `[White, Black]` — the input
    /// [`super::sota_head::SotaHead::forward`] consumes as `threat`.
    pub fn current(&self) -> &[[i16; ACC]; 2] {
        self.stack
            .last()
            .expect("ThreatContext: refresh before current")
    }

    /// Accumulator at a search ply. The search overwrites sibling plies rather
    /// than calling `undo`, so callers use this indexed form.
    pub fn at(&self, ply: usize) -> &[[i16; ACC]; 2] {
        &self.stack[ply]
    }

    /// Fold the move described by `dp` (made to reach `board`) forward: push a
    /// new stack entry = parent + resolved deltas. A perspective whose own king
    /// crossed the d/e boundary is rebuilt from scratch instead; the other is
    /// incremental. `board` is the fully post-move position.
    pub fn apply(&mut self, dp: &DirtyPiece, board: &Board) {
        if self.weights.is_empty() {
            let parent = *self
                .stack
                .last()
                .expect("ThreatContext: refresh before apply");
            self.stack.push(parent);
            return;
        }
        // Build the perspective-agnostic raw event list from dp + post-move board.
        self.buf.clear();
        build_deltas(&mut self.buf, dp, board);

        let parent = *self
            .stack
            .last()
            .expect("ThreatContext: refresh before apply");
        let mut child = parent;
        for (i, &persp) in Color::ALL.iter().enumerate() {
            if king_crossed(dp, persp) {
                child[i] = refresh_threat(self.weights, board, persp);
                continue;
            }
            let ksq = board.king_square(persp);
            let acc = &mut child[i];
            let mut add_len = 0;
            let mut sub_len = 0;
            for d in self.buf.events() {
                if let Some(idx) = threat_index(persp, d.attacker, d.from, d.to, d.victim, ksq) {
                    if d.add {
                        // The simple intermediate-state replay deliberately
                        // over-emits telescoping events. Cancel an opposite
                        // occurrence before purchasing a 768-lane update.
                        if let Some(pos) = self.subs[..sub_len].iter().position(|&old| old == idx) {
                            sub_len -= 1;
                            self.subs[pos] = self.subs[sub_len];
                        } else {
                            self.adds[add_len] = idx;
                            add_len += 1;
                        }
                    } else {
                        if let Some(pos) = self.adds[..add_len].iter().position(|&old| old == idx) {
                            add_len -= 1;
                            self.adds[pos] = self.adds[add_len];
                        } else {
                            self.subs[sub_len] = idx;
                            sub_len += 1;
                        }
                    }
                }
            }
            apply_i8_deltas(
                acc,
                self.weights,
                &self.adds[..add_len],
                &self.subs[..sub_len],
            );
        }
        self.stack.push(child);
    }

    /// Apply a post-move dirty record from `ply` to `ply + 1`, truncating any
    /// stale sibling/deeper entries first. This is the search-stack form of
    /// [`apply`](Self::apply).
    pub fn apply_at(&mut self, ply: usize, dp: &DirtyPiece, board: &Board) {
        assert!(ply < self.stack.len(), "ThreatContext parent ply missing");
        self.stack.truncate(ply + 1);
        self.apply(dp, board);
    }

    /// Pop the top stack entry, returning to the parent — the exact inverse of
    /// the matching [`apply`](Self::apply) (the full parent pair is retained, so
    /// undo is the delta walk run backward with added↔removed swapped, by
    /// construction). Panics if only the root remains.
    pub fn undo(&mut self) {
        assert!(
            self.stack.len() > 1,
            "ThreatContext::undo underflow (only the root remains)"
        );
        self.stack.pop();
    }

    /// Depth of the incremental stack above the root (test/debug introspection).
    #[cfg(any(test, debug_assertions))]
    pub fn depth(&self) -> usize {
        self.stack.len() - 1
    }
}

/// Reconstruct the pre-move occupancy from `board` (post-move) + `dp`, then
/// replay the edits — every sub removed (in order), then every add — recording a
/// `record_square_touch` per edit against the state *after* that edit. See the
/// module docs for the telescoping argument.
fn build_deltas(buf: &mut DeltaBuf, dp: &DirtyPiece, board: &Board) {
    // A non-capture (including a quiet promotion) changes occupancy only by
    // moving the blocker from `from` to `to`. Evaluate both endpoint changes
    // against the shared occupancy with both endpoints empty; this is the
    // compact move primitive and avoids intermediate endpoint interactions.
    if let ([(mover, from)], [(placed, to)]) = (dp.subs(), dp.adds())
        && from != to
    {
        let between = Replay {
            board,
            occ: board.all_occupancy() & !Bitboard::from_square(*to),
            over_sq: [Square::A1; 8],
            over_pc: [None; 8],
            n_over: 0,
        };
        record_square_touch(buf, &between, *mover, *from, false);
        record_square_touch(buf, &between, *placed, *to, true);
        return;
    }

    // Ordinary and promotion captures are the common occupancy-preserving
    // replacement case. Model them as mover removal plus victim→mover identity
    // mutation, avoiding the remove/add x-ray churn at the destination.
    if let ([(mover, from), (victim, victim_sq)], [(placed, to)]) = (dp.subs(), dp.adds())
        && victim_sq == to
        && from != to
    {
        let mut after_remove = Replay {
            board,
            occ: board.all_occupancy(),
            over_sq: [Square::A1; 8],
            over_pc: [None; 8],
            n_over: 0,
        };
        after_remove.set(*to, Some(*victim));
        record_square_touch(buf, &after_remove, *mover, *from, false);

        let final_state = Replay {
            board,
            occ: board.all_occupancy(),
            over_sq: [Square::A1; 8],
            over_pc: [None; 8],
            n_over: 0,
        };
        record_mutate(buf, &final_state, *victim, *placed, *to);
        return;
    }

    // Pre-move occupancy = post-move, with adds cleared then subs set. A square
    // that is both a sub and an add (a capture's destination) ends up SET
    // (occupied pre-move by the victim), because subs are applied last.
    let mut pre_occ = board.all_occupancy();
    for &(_, sq) in dp.adds() {
        pre_occ &= !Bitboard::from_square(sq);
    }
    for &(_, sq) in dp.subs() {
        pre_occ |= Bitboard::from_square(sq);
    }

    let mut r = Replay {
        board,
        occ: pre_occ,
        over_sq: [Square::A1; 8],
        over_pc: [None; 8],
        n_over: 0,
    };

    // Seed touched-square identities to their pre-move occupants: a sub square
    // held the sub piece; an add-only square was empty. (Subs seed last so a
    // sub+add square keeps the victim.)
    for &(_, sq) in dp.adds() {
        if r.find(sq).is_none() {
            r.over_sq[r.n_over] = sq;
            r.over_pc[r.n_over] = None;
            r.n_over += 1;
        }
    }
    for &(pc, sq) in dp.subs() {
        let i = r.find(sq).unwrap_or_else(|| {
            let i = r.n_over;
            r.over_sq[i] = sq;
            r.n_over += 1;
            i
        });
        r.over_pc[i] = Some(pc);
    }

    // Removes first, then adds — each touch runs against the state after its own
    // edit (`set` updates occ + identity before we record).
    for &(pc, sq) in dp.subs() {
        r.set(sq, None);
        record_square_touch(buf, &r, pc, sq, false);
    }
    for &(pc, sq) in dp.adds() {
        r.set(sq, Some(pc));
        record_square_touch(buf, &r, pc, sq, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;
    use crate::movegen::generate_legal;
    use crate::moves::{Move, MoveList};

    // A tiny SplitMix64 so the replay games are deterministic without pulling in
    // a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Distinct pseudo-random `i8` weight columns per feature, so an accumulator
    /// match is a strong proxy for an active-feature-set match: a single wrong /
    /// missing / swapped feature index perturbs a pseudo-random 768-wide column
    /// and the sums diverge. ~44 MiB, allocated once per test.
    fn signature_weights() -> Vec<i8> {
        use super::super::sota_features::THREAT_DIMS;
        let mut w = vec![0i8; THREAT_DIMS * ACC];
        for (i, slot) in w.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z ^= z >> 29;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 32;
            *slot = z as i8;
        }
        w
    }

    fn refresh_pair(weights: &[i8], board: &Board) -> [[i16; ACC]; 2] {
        [
            refresh_threat(weights, board, Color::White),
            refresh_threat(weights, board, Color::Black),
        ]
    }

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    /// A single quiet move's incremental accumulator matches a from-scratch
    /// refresh — the smallest slice of the §12.9.1 invariant.
    #[test]
    fn one_quiet_move_matches_refresh() {
        let w = signature_weights();
        let mut b = board(START_FEN);
        let mut ctx = ThreatContext::new(&w);
        ctx.refresh(&b);

        let m = Move::normal(Square::E2, Square::E4);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);

        assert_eq!(*ctx.current(), refresh_pair(&w, &b));
    }

    /// A capture: the mover's from-square relations vanish, the victim→mover
    /// swap on the destination is captured, and any x-ray through either square
    /// ripples. (1.e4 d5 2.exd5.)
    #[test]
    fn capture_matches_refresh() {
        let w = signature_weights();
        let fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        let mut b = board(fen);
        let mut ctx = ThreatContext::new(&w);
        ctx.refresh(&b);

        let m = Move::normal(Square::E4, Square::D5);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);

        assert_eq!(*ctx.current(), refresh_pair(&w, &b));
    }

    /// Each of the trickier move kinds, one at a time, against a fresh refresh:
    /// en passant, quiet + capture promotion, and both castlings (queenside
    /// crosses the mirror → forces a refresh of that perspective).
    #[test]
    fn special_move_kinds_match_refresh() {
        let w = signature_weights();
        let cases: &[(&str, Move)] = &[
            // En passant: white pawn e5 takes f6 ep.
            (
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                Move::en_passant(Square::E5, Square::F6),
            ),
            // Quiet promotion to queen.
            (
                "8/P6k/8/8/8/8/7K/8 w - - 0 1",
                Move::promotion(Square::A7, Square::A8, PieceType::Queen),
            ),
            // Capture-promotion to knight.
            (
                "1n5k/P7/8/8/8/8/7K/8 w - - 0 1",
                Move::promotion(Square::A7, Square::B8, PieceType::Knight),
            ),
            // Kingside castle (no mirror cross).
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                Move::castling(Square::E1, Square::G1),
            ),
            // Queenside castle (king e1 → c1 crosses d/e).
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                Move::castling(Square::E1, Square::C1),
            ),
        ];
        for &(fen, m) in cases {
            let mut b = board(fen);
            let mut ctx = ThreatContext::new(&w);
            ctx.refresh(&b);
            let dp = DirtyPiece::from_move(&b, m);
            b.make_move(m, b.gives_check(m));
            ctx.apply(&dp, &b);
            assert_eq!(
                *ctx.current(),
                refresh_pair(&w, &b),
                "mismatch after {m} from {fen}"
            );
        }
    }

    /// A king move that crosses the d/e boundary refreshes only the crossing
    /// perspective; the other stays incremental. (King e4 → d4.)
    #[test]
    fn king_mirror_cross_matches_refresh() {
        let w = signature_weights();
        // White is in check from Bd5 (d5–e4); Ke4–d4 escapes and crosses the
        // d/e mirror, with sliders (Bd5, Rc4, Qc7, Rf7) rippling through e4/d4.
        let fen = "k7/2q2r2/8/3b4/2R1Kn2/8/8/8 w - - 0 1";
        let mut b = board(fen);
        let mut ctx = ThreatContext::new(&w);
        ctx.refresh(&b);
        let m = Move::normal(Square::E4, Square::D4);
        assert!(king_crossed(&DirtyPiece::from_move(&b, m), Color::White));
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);
        assert_eq!(*ctx.current(), refresh_pair(&w, &b));
    }

    // Play `games` random legal games of up to `max_plies`, asserting at every
    // node that the incrementally-maintained accumulator equals a from-scratch
    // refresh for both perspectives, and that apply→undo restores the parent
    // exactly (make→apply→unmake→undo == identity).
    fn replay(weights: &[i8], games: usize, max_plies: usize, seed: u64) -> usize {
        let mut rng = Rng(seed);
        let mut nodes = 0usize;
        for _ in 0..games {
            let mut b = board(START_FEN);
            let mut ctx = ThreatContext::new(weights);
            ctx.refresh(&b);

            for _ in 0..max_plies {
                let mut list = MoveList::new();
                generate_legal(&b, &mut list);
                if list.is_empty() {
                    break;
                }
                let m = list[(rng.next() % list.len() as u64) as usize];

                // Snapshot the pre-move accumulator so undo can be checked.
                let before = *ctx.current();

                let dp = DirtyPiece::from_move(&b, m);
                let gc = b.gives_check(m);
                b.make_move(m, gc);
                ctx.apply(&dp, &b);
                nodes += 1;

                // (a) incremental == from-scratch refresh at this node.
                assert_eq!(
                    *ctx.current(),
                    refresh_pair(weights, &b),
                    "incremental != refresh after {m}\nfen (post): {}",
                    b.to_fen(),
                );

                // (b) unmake the board and undo the accumulator: identity.
                b.unmake_move(m);
                ctx.undo();
                assert_eq!(*ctx.current(), before, "apply→undo not identity for {m}");

                // Re-apply to continue the game down this line.
                b.make_move(m, gc);
                ctx.apply(&dp, &b);
            }
        }
        nodes
    }

    /// The correctness gate (default tier): a moderate random-game corpus with
    /// the full incremental == refresh + undo-identity checks at every node.
    #[test]
    fn threats_replay_matches_refresh() {
        let w = signature_weights();
        let nodes = replay(&w, 60, 60, 0x00C0_FFEE_1234);
        assert!(
            nodes > 1500,
            "expected a substantial corpus, got {nodes} nodes"
        );
    }

    /// The large-corpus stress gate — thousands of games. Slow in debug (the
    /// board's per-node `assert_consistent` oracle dominates); run in release:
    /// `cargo test --release --features sota -- --ignored threats_replay_large`.
    #[test]
    #[ignore = "large corpus; run in release"]
    fn threats_replay_large() {
        let w = signature_weights();
        let nodes = replay(&w, 4000, 120, 0x5EED_D00D_F00D);
        assert!(
            nodes > 100_000,
            "expected a large corpus, got {nodes} nodes"
        );
    }
}
