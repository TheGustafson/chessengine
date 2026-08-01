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
//! from-scratch [`super::sota_features::refresh_threat_into`] uses.
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
//! # The driver: the first-cut
//! Per §12.9.1 this lands the **simpler first-cut** driver ("`on_change` every
//! `DirtyPiece` sub — removes first — then every add; auto-cancels, costs extra
//! events") rather than the optimized 3-primitive `on_move` form. It is wired
//! into the SOTA search stack via [`super::sota_context::SotaContext`], behind
//! the `sota` cargo feature, so the default build is untouched.
//!
//! # Fully-lazy folding (perf study 3, §2)
//! The observer is *fully* deferred to evaluate time. The round-1/2 claim that
//! resolution "must run at push time because it reads the post-move board" was
//! false: event generation reads only occupancy + the mailbox, and resolution
//! reads the king square only through a 1-bit mirror ([`orient_threats`]). So
//! [`ThreatContext::apply_at`] snapshots exactly those inputs (the type/colour bitboards +
//! occupancy + both king squares, 74 B — a [`ThreatPosition`]) plus the
//! [`DirtyPiece`], and defers *everything else* — generation ([`build_deltas`]),
//! per-perspective resolution ([`threat_index`]), and the 768-lane fold
//! ([`apply_threat_deltas_into`]) — to [`ThreatContext::compute`], which runs them against
//! the stored snapshot only for the plies that actually reach an [`evaluate`],
//! folding forward from the nearest computed ancestor. Children that cut off
//! (TT hits, immediate fail-lows) pay nothing. Generation is shared across
//! perspectives per folded ply; the fold runs on **0.4151 plies per node**
//! (perf study 18, exact), so this deletes most of the observer block at a
//! **bit-identical** eval — same events, same resolution, same fold, run later
//! and less often. A king cross (the sole from-scratch case) is **also**
//! deferred: [`apply_at`](ThreatContext::apply_at) records it as a
//! [`Fold::Refresh`] obligation and [`compute`](ThreatContext::compute)
//! discharges it from the ply's own snapshot (perf study 18 §3). It used to
//! rebuild eagerly at push because the rebuild took a `&Board` and the live
//! board is gone by fold time; it now takes the snapshot the entry already
//! stored.
//!
//! [`orient_threats`]: super::sota_features
//! [`evaluate`]: super::sota_context::SotaContext::evaluate
//! [`apply_threat_deltas_into`]: super::sota_simd::apply_threat_deltas_into
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
//! of the HalfKA refresh trigger, and one Stockfish itself cannot exploit
//! because its accumulator fuses HalfKA and threats into one vector
//! (`eca43a97:src/nnue/nnue_accumulator.cpp:112-113` concedes exactly this).
//! We detect that per perspective, RECORD it as a [`Fold::Refresh`] obligation,
//! and rebuild only the crossing side, at fold time; the other side (and every
//! non-king move) stays incremental.

use super::sota_accumulator::{psqt_add, psqt_sub};
use super::sota_features::{
    ACC, KEPT_VICTIMS, RowOffset, THREAT_DIMS, ThreatPosition, ThreatRemap, ThreatResolver,
    for_each_threat_at, refresh_threat_into_at, sf_code, threat_pair_kept,
};
#[cfg(test)]
use super::sota_features::{for_each_threat, refresh_threat};
use super::sota_head::BUCKETS;
use super::sota_simd::apply_threat_deltas_into;
use crate::attacks::{AttackTables, knight_attacks, pawn_attacks, ray_pass, tables};
use crate::bitboard::Bitboard;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, PieceType, Square};

/// One perspective-agnostic threat event: "`attacker` on `from` threatens
/// `victim` on `to`", appearing (`add`) or disappearing (`!add`). Resolved to a
/// feature index per perspective by [`threat_index`]; excluded pairs drop out.
///
/// # Why `align(8)` (perf study 24 arm d1)
/// The natural layout is five `u8`-sized fields — **size 7, align 1**. At a
/// 7-byte stride roughly one event in nine straddles a 64-byte cache line,
/// every field access is unaligned, the five stores in
/// [`DeltaBuf::push_kept`] cannot be merged, and the resolver's per-event
/// address arithmetic is a `lea`-unfriendly `add $0x7`. Rounding the SIZE up to
/// 8 (align 8 implies size 8) costs 256 bytes on a 2 KB scratch buffer that
/// never leaves L1, and measured **−37.73 i/n, −0.67 branches/n and
/// −19.34 L1 loads/n** — node- and eval-hash-identical, one attribute.
#[derive(Debug, Clone, Copy)]
#[repr(align(8))]
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
        // §6-T2 (perf study): drop pairs `threat_index` would fold to None at
        // the single choke point every recording site flows through — they
        // were being buffered, resolved twice, then discarded (king victims
        // on every touch, pawn-diagonal onto B/Q, bishop/rook onto Q).
        // `threat_pair_kept` is derived from the SAME `MAP` table the fold
        // consults, so the drop sets cannot diverge; add/sub use the same
        // pure type predicate, so telescoping cancellation is preserved.
        // The pawn-push-onto-N/R asymmetry stays guarded at the recorder.
        if !super::sota_features::threat_pair_kept(d.attacker.piece_type, d.victim.piece_type) {
            return;
        }
        self.push_kept(d);
    }

    /// [`push`](Self::push) for a site that has **already** established the
    /// attacker→victim type pair is one `MAP` keeps — perf study 19 arm g2.
    ///
    /// Most recording sites cannot emit a dropped pair at all, because the pair
    /// is a structural property of the site rather than of the event:
    /// [`record_targets`] iterates `KEPT_VICTIMS[attacker]`, so every victim
    /// type it visits is kept by construction; the pawn-push emitters are
    /// pawn→pawn, and `MAP[Pawn][Pawn] = 0`; and the two incoming scans in
    /// [`record_square_touch`] are already inside hoisted
    /// `threat_pair_kept(Knight | Pawn, p.piece_type)` guards — the same test,
    /// evaluated once per touch instead of once per emitted event.
    ///
    /// Re-testing at the choke point was therefore a `MAP[a][v]` load plus a
    /// sign test on the majority of pushes, to re-derive a fact the caller had
    /// just proved. The four sites that CAN drop — [`ripple`]'s two emissions,
    /// and [`record_incoming_mutate`]'s fan-loop pair, whose rook/bishop/queen
    /// attackers meet an `old`/`new` victim that may itself be a queen — keep
    /// calling [`push`](Self::push). The shipped tree is 8 `push_kept`, 4
    /// `push`.
    ///
    /// **This does not duplicate the drop set.** There is still exactly one
    /// definition (`MAP`, via `threat_pair_kept`); this is an assertion that
    /// the caller has consulted it, and the `debug_assert` below re-checks that
    /// claim on every event in every debug build — which is what the whole
    /// `sota` test suite, including the per-node `assert_matches_refresh`
    /// oracle, runs under.
    #[inline]
    fn push_kept(&mut self, d: ThreatDelta) {
        debug_assert!(
            super::sota_features::threat_pair_kept(d.attacker.piece_type, d.victim.piece_type),
            "push_kept on a pair MAP drops: {:?} -> {:?}",
            d.attacker.piece_type,
            d.victim.piece_type
        );
        // Perf study 55 mechanism B: `DELTA_CAP` is a power of two and the
        // debug_assert above is the invariant, but `self.len` is a struct field
        // LLVM cannot bound, so this store carried a `cmp`/`ja` and a panic edge
        // per threat event. `resolve_both` masks the READ side of this very
        // buffer for exactly this reason (see its `M` there); the write side was
        // never given the same treatment.
        debug_assert!(self.len < DELTA_CAP, "threat delta buffer overflow");
        self.data[self.len & (DELTA_CAP - 1)] = d;
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

/// The replay's working position: a plain by-value copy of a
/// [`ThreatPosition`] that the edit replay mutates with [`ThreatPosition::toggle`].
///
/// The mailbox form needed a parallel override map (`over_sq`/`over_pc` plus a
/// linear `find`) because a mailbox cannot represent "the state after edit *i*"
/// without being copied wholesale. Bitboards can: one XOR per board.
///
/// The struct itself lives in [`super::sota_features`] — it is the position
/// view the FullThreats feature space reads, and perf study 18 made the
/// from-scratch enumeration run off it too, so there is exactly one definition
/// of "what the threat features see".
type Replay = ThreatPosition;

/// Emit every kept relation from `attacker` on `from` to a piece in `targets`,
/// **iterating one victim type at a time** so the victim's type is a loop
/// constant and the excluded pairs never materialise.
///
/// This is the incremental twin of [`for_each_threat`]'s victim-mask loops. The
/// mask comes from [`KEPT_VICTIMS`], which is derived from the same `MAP` the
/// resolver consults, so the two drop sets are the same by construction.
///
/// `KEPT` is that mask as a **const parameter** rather than a runtime load
/// (perf study 19 arm g3). Every caller but one already knows the attacker's
/// type at compile time — `record_square_touch`'s outgoing section reaches this
/// from inside a `match p.piece_type` arm, and `record_outgoing`'s pawn arm is
/// literally `PieceType::Pawn` — so `KEPT_VICTIMS[attacker.piece_type]` was a
/// table load feeding six runtime bit tests to re-derive a compile-time
/// constant. With the mask as a const parameter the `PieceType::ALL` loop
/// disappears entirely: the body is emitted once per victim type actually in
/// the set (3 for a pawn, 4 for a bishop/rook, 5 for a knight/queen) with no
/// test at all.
///
/// The idiom is the same const-generic specialisation `movegen::generate::<GT>`
/// uses, and it is why `KEPT` is `u8` rather than a `PieceType`: const generics
/// cannot yet be user enums, so callers pass `KEPT_VICTIMS[…]` — still the one
/// `MAP`-derived table, just consulted by the compiler instead of the CPU.
#[inline]
fn record_targets<const KEPT: u8>(
    buf: &mut DeltaBuf,
    pos: &Replay,
    attacker: Piece,
    from: Square,
    targets: Bitboard,
    add: bool,
) {
    debug_assert_eq!(
        KEPT, KEPT_VICTIMS[attacker.piece_type as usize],
        "record_targets const mask disagrees with the attacker's KEPT_VICTIMS row"
    );
    for vt in PieceType::ALL {
        if KEPT & (1 << vt as u8) == 0 {
            continue;
        }
        for to in targets & pos.typed(vt) {
            // `vt` came out of `KEPT_VICTIMS[attacker]`, so the pair is kept by
            // construction — see `DeltaBuf::push_kept`.
            buf.push_kept(ThreatDelta {
                attacker,
                from,
                victim: Piece::new(pos.color_on(to), vt),
                to,
                add,
            });
        }
    }
}

/// Emit every threat relation a piece `p` conceptually appearing (`add`) or
/// disappearing (`!add`) at `s` participates in, against the replay's current
/// occupancy (design/11 §12.9.1). Over-generates freely; [`threat_index`] drops
/// the excluded pairs at resolution time. The one non-over-generatable case —
/// pawn pushes onto a non-pawn — is guarded here.
fn record_square_touch(
    buf: &mut DeltaBuf,
    tab: &AttackTables,
    r: &Replay,
    p: Piece,
    s: Square,
    add: bool,
) {
    let occ = r.occ;

    // The two ray fans from `s`. Section 2 needs them unconditionally, and
    // section 1's outgoing set for a SLIDER is exactly one of them — perf
    // study 12 §8: computing the outgoing set with `tab.attacks(pt, s, occ)`
    // re-ran the identical magic lookup a few lines before `ratt`/`batt` did
    // (twice over, for a queen). Hoisting them above the match and reusing
    // them is bit-identical: `AttackTables::attacks` IS `rook_attacks` /
    // `bishop_attacks` / their union for R/B/Q (`src/attacks.rs`).
    let ratt = tab.rook_attacks(s, occ);
    let batt = tab.bishop_attacks(s, occ);
    let qatt = ratt | batt;

    // --- 1. Outgoing: what `p` threatens from `s`. Kings are never attackers.
    match p.piece_type {
        PieceType::King => {}
        PieceType::Rook => {
            record_targets::<{ KEPT_VICTIMS[PieceType::Rook as usize] }>(buf, r, p, s, ratt, add)
        }
        PieceType::Bishop => {
            record_targets::<{ KEPT_VICTIMS[PieceType::Bishop as usize] }>(buf, r, p, s, batt, add)
        }
        PieceType::Queen => {
            record_targets::<{ KEPT_VICTIMS[PieceType::Queen as usize] }>(buf, r, p, s, qatt, add)
        }
        PieceType::Knight => record_targets::<{ KEPT_VICTIMS[PieceType::Knight as usize] }>(
            buf,
            r,
            p,
            s,
            knight_attacks(s),
            add,
        ),
        PieceType::Pawn => {
            record_targets::<{ KEPT_VICTIMS[PieceType::Pawn as usize] }>(
                buf,
                r,
                p,
                s,
                pawn_attacks(p.color, s),
                add,
            );
            // Push relation: only a pawn directly in front is threatened
            // (matching for_each_threat's `all_pawns` guard — MAP would
            // otherwise wrongly keep pawn-push onto a knight/rook).
            if let Some(push) = s.shifted(p.color.pawn_push())
                && r.typed(PieceType::Pawn).contains(push)
            {
                // pawn -> pawn: `MAP[Pawn][Pawn] = 0`, always kept.
                buf.push_kept(ThreatDelta {
                    attacker: p,
                    from: s,
                    victim: Piece::new(r.color_on(push), PieceType::Pawn),
                    to: push,
                    add,
                });
            }
        }
    }

    // --- 2. X-ray ripple: sliders whose ray reaches `s`. Each such slider `sl`
    // gains/loses its direct relation to `p@s`, and its relation to the first
    // occupant strictly beyond `s` flips the *other* way (removing p unblocks
    // sl; adding p blocks it). The ray fans are pre-masked with the slider
    // boards, so only real sliders are ever iterated — the mailbox form walked
    // every occupied square on both fans and resolved each one to find out.
    let queens = r.typed(PieceType::Queen);
    for sl_sq in ratt & r.rook_movers() {
        let pt = if queens.contains(sl_sq) {
            PieceType::Queen
        } else {
            PieceType::Rook
        };
        let sl = Piece::new(r.color_on(sl_sq), pt);
        ripple(buf, r, qatt, sl, sl_sq, p, s, add);
    }
    for sl_sq in batt & r.bishop_movers() {
        let pt = if queens.contains(sl_sq) {
            PieceType::Queen
        } else {
            PieceType::Bishop
        };
        let sl = Piece::new(r.color_on(sl_sq), pt);
        ripple(buf, r, qatt, sl, sl_sq, p, s, add);
    }

    // --- 3. Incoming: short-range attackers of `s` (pawns, knights). Kings are
    // never attackers, so no king incoming. Only meaningful when `p` can be a
    // victim, i.e. not a king. The victim is fixed here, so the
    // attacker-type predicate is a loop-invariant and hoists out entirely: a
    // bishop or queen victim skips the whole pawn scan.
    if p.piece_type != PieceType::King {
        // The knight scan needs NO victim-type guard (perf study 19 arm g4).
        // `MAP`'s knight row is `[0, 1, 2, 3, 4, -1]` — a knight threatens every
        // victim type EXCEPT a king — and this whole block is already inside
        // `p.piece_type != King`, so `threat_pair_kept(Knight, p.piece_type)`
        // could only ever be true here. It was a `MAP` load and a branch per
        // touch to re-derive the condition of the enclosing `if`.
        //
        // The pawn scan below is NOT the same case and keeps its guard: `MAP`'s
        // pawn row is `[0, 1, -1, 2, -1, -1]`, so a pawn threatens only
        // {pawn, knight, rook} and a bishop or queen victim really does skip it.
        {
            debug_assert!(threat_pair_kept(PieceType::Knight, p.piece_type));
            for a in knight_attacks(s) & r.typed(PieceType::Knight) {
                // Knight -> non-king: kept by the argument above.
                buf.push_kept(ThreatDelta {
                    attacker: Piece::new(r.color_on(a), PieceType::Knight),
                    from: a,
                    victim: p,
                    to: s,
                    add,
                });
            }
        }
        if threat_pair_kept(PieceType::Pawn, p.piece_type) {
            let pawns = r.typed(PieceType::Pawn);
            // A white pawn attacking `s` sits on a down-diagonal of `s`
            // (`pawn_attacks(Black, s)`), a black pawn on an up-diagonal.
            for (c, fan) in [
                (Color::White, pawn_attacks(Color::Black, s)),
                (Color::Black, pawn_attacks(Color::White, s)),
            ] {
                let attacker = Piece::new(c, PieceType::Pawn);
                for a in fan & pawns & r.colored(c) {
                    // Inside the hoisted `threat_pair_kept(Pawn, p)` guard.
                    buf.push_kept(ThreatDelta {
                        attacker,
                        from: a,
                        victim: p,
                        to: s,
                        add,
                    });
                }
            }
            // Incoming pawn push: a pawn directly behind `s` threatens it — but
            // only if `p@s` is itself a pawn (push is a pawn→pawn relation).
            if p.piece_type == PieceType::Pawn {
                for c in Color::ALL {
                    if let Some(back) = s.shifted(c.flip().pawn_push())
                        && (pawns & r.colored(c)).contains(back)
                    {
                        // pawn -> pawn, always kept.
                        buf.push_kept(ThreatDelta {
                            attacker: Piece::new(c, PieceType::Pawn),
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
#[inline(always)]
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
            victim: r.piece_on(d),
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
fn record_mutate(
    buf: &mut DeltaBuf,
    tab: &AttackTables,
    r: &Replay,
    old: Piece,
    new: Piece,
    square: Square,
) {
    record_outgoing(buf, tab, r, old, square, false);
    record_outgoing(buf, tab, r, new, square, true);
    record_incoming_mutate(buf, tab, r, old, new, square);
}

fn record_outgoing(
    buf: &mut DeltaBuf,
    tab: &AttackTables,
    r: &Replay,
    piece: Piece,
    from: Square,
    add: bool,
) {
    match piece.piece_type {
        PieceType::King => {}
        PieceType::Pawn => {
            record_targets::<{ KEPT_VICTIMS[PieceType::Pawn as usize] }>(
                buf,
                r,
                piece,
                from,
                pawn_attacks(piece.color, from),
                add,
            );
            if let Some(to) = from.shifted(piece.color.pawn_push())
                && r.typed(PieceType::Pawn).contains(to)
            {
                // pawn -> pawn, always kept.
                buf.push_kept(ThreatDelta {
                    attacker: piece,
                    from,
                    victim: Piece::new(r.color_on(to), PieceType::Pawn),
                    to,
                    add,
                });
            }
        }
        // The remaining four types, spelled out so each gets its own const
        // `KEPT` mask. Knight and Queen share a row of `KEPT_VICTIMS`
        // ({P,N,B,R,Q}) and bishop and rook share the other ({P,N,B,R}), so this
        // is two specialisations, not four.
        pt => {
            let targets = tab.attacks(pt, from, r.occ);
            match pt {
                PieceType::Knight | PieceType::Queen => {
                    record_targets::<{ KEPT_VICTIMS[PieceType::Queen as usize] }>(
                        buf, r, piece, from, targets, add,
                    );
                }
                _ => {
                    debug_assert!(matches!(pt, PieceType::Bishop | PieceType::Rook));
                    record_targets::<{ KEPT_VICTIMS[PieceType::Rook as usize] }>(
                        buf, r, piece, from, targets, add,
                    );
                }
            }
        }
    }
}

fn record_incoming_mutate(
    buf: &mut DeltaBuf,
    tab: &AttackTables,
    r: &Replay,
    old: Piece,
    new: Piece,
    to: Square,
) {
    let occ = r.occ;
    let queens = r.typed(PieceType::Queen);
    let pawns = r.typed(PieceType::Pawn);
    // Pre-masked attacker fans. They are pairwise disjoint — a square is either
    // rank/file-aligned with `to` or diagonally aligned, never both, and a
    // knight/pawn is neither a slider nor the other — so iterating them
    // separately visits every attacker exactly once, like the old union walk,
    // without resolving any identity or re-testing any geometry.
    // (fan, the type its occupants have, whether a queen can also appear).
    let fans = [
        (
            tab.rook_attacks(to, occ) & r.rook_movers(),
            PieceType::Rook,
            true,
        ),
        (
            tab.bishop_attacks(to, occ) & r.bishop_movers(),
            PieceType::Bishop,
            true,
        ),
        (
            knight_attacks(to) & r.typed(PieceType::Knight),
            PieceType::Knight,
            false,
        ),
        (
            pawn_attacks(Color::Black, to) & pawns & r.colored(Color::White),
            PieceType::Pawn,
            false,
        ),
        (
            pawn_attacks(Color::White, to) & pawns & r.colored(Color::Black),
            PieceType::Pawn,
            false,
        ),
    ];
    for (fan, default_pt, slider) in fans {
        for from in fan {
            let pt = if slider && queens.contains(from) {
                PieceType::Queen
            } else {
                default_pt
            };
            let attacker = Piece::new(r.color_on(from), pt);
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
            && (pawns & r.colored(color)).contains(from)
        {
            let attacker = Piece::new(color, PieceType::Pawn);
            // Both emissions are pawn -> pawn under their own guards, and
            // `MAP[Pawn][Pawn] = 0`, so the choke-point filter cannot drop them.
            if old.piece_type == PieceType::Pawn {
                buf.push_kept(ThreatDelta {
                    attacker,
                    from,
                    victim: old,
                    to,
                    add: false,
                });
            }
            if new.piece_type == PieceType::Pawn {
                buf.push_kept(ThreatDelta {
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

/// From-scratch FullThreats PSQT bucket sums for `perspective` (T6c skip):
/// the sum of the 8-wide PSQ1 threat row of every active threat feature —
/// the oracle the incremental fold must match. Zero when `table` is empty
/// (legacy net, or a HalfKA-only psqt net).
pub(crate) fn refresh_psqt_threat(
    table: &[i32],
    board: &Board,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) -> [i32; BUCKETS] {
    refresh_psqt_threat_at(
        table,
        &ThreatPosition::from_board(board),
        perspective,
        remap,
    )
}

/// [`refresh_psqt_threat`] over a [`ThreatPosition`] snapshot — the deferred
/// king-cross rebuild's half of the skip channel (perf study 18 §3).
fn refresh_psqt_threat_at(
    table: &[i32],
    pos: &ThreatPosition,
    perspective: Color,
    remap: Option<&ThreatRemap>,
) -> [i32; BUCKETS] {
    let mut out = [0i32; BUCKETS];
    if table.is_empty() {
        return out;
    }
    for_each_threat_at(pos, perspective, remap, |idx| {
        psqt_add(&mut out, table, idx)
    });
    out
}

/// Which perspectives' kings cross the d/e mirror boundary in this move? (Such
/// a perspective's whole threat accumulator must be rebuilt from scratch —
/// every one of its feature indices is re-oriented.)
///
/// **One pass per LIST, not one pass per COLOUR.** The predecessor
/// `king_crossed(dp, color)` was called once per perspective from `apply_at`,
/// i.e. at 2.0 calls per made move, and each call re-scanned both the sub and
/// the add list for a fact the other call already had in hand. A move moves one
/// piece, so at most one colour's king can appear at all; scanning per colour
/// bought nothing but a second walk.
fn king_cross(dp: &DirtyPiece) -> [bool; 2] {
    // NO INTERMEDIATE `Option` ARRAYS (perf study 24 arm k1). The predecessor
    // materialised `from: [Option<Square>; 2]` and `to: [Option<Square>; 2]`,
    // filled them in two passes and then ran a THIRD pass pairing them: eight
    // bytes of stack initialised to `None`, up to four conditional stores and
    // two `Option` destructurings, on EVERY made move, to answer a question
    // that is false for the ~95% of moves that do not move a king. It is
    // visible in `apply_at`'s disassembly as a 2-byte-strided stack array
    // (`mov %cl,0x1(%rsp,%r8,2)` / `movzbl 0x5(%rsp),%r8d`).
    //
    // A move relocates ONE piece, so a king can appear at most once per list
    // and, if it appears in both, it is the same king — the pairing needs no
    // storage at all, only a nested scan whose outer body is skipped outright
    // for a non-king mover. Same rule, same answer: `out[c]` is set only when
    // colour `c`'s king is in BOTH lists, so `Fold::Refresh` is raised on
    // exactly the same moves.
    //
    // Measured −20.51 i/n and −3.34 branches/n against a live interleaved null
    // of −0.0006 i/n / −0.0005 b/n; node- and eval-hash-identical.
    let mut out = [false; 2];
    for &(pc, from) in dp.subs() {
        if pc.piece_type != PieceType::King {
            continue;
        }
        for &(pc2, to) in dp.adds() {
            if pc2.piece_type == PieceType::King && pc2.color == pc.color {
                // orient_threats splits on file <= 3 (a–d) vs >= 4 (e–h).
                out[pc.color as usize] = (from.file() <= 3) != (to.file() <= 3);
                // Tier-0 T0c census for the richer-representation study: which
                // CANDIDATE context boundaries did this in-tree king move
                // cross? Counter-only, feature-gated, off in every default and
                // release build.
                #[cfg(feature = "foldcount")]
                {
                    super::foldcount::note_king_move(pc.color as usize, from.index(), to.index());
                    // Learned-representation study: the full from→to transition
                    // matrix, so any FUTURE partition (including learned ones)
                    // is priceable from one run.
                    super::foldcount::note_king_trans(pc.color as usize, from.index(), to.index());
                }
                break;
            }
        }
    }
    out
}

/// Does `color`'s king cross the mirror? The single-colour form the tests read
/// against, kept out of the hot path so `apply_at` does one scan, not two.
#[cfg(test)]
fn king_crossed(dp: &DirtyPiece, color: Color) -> bool {
    king_cross(dp)[color as usize]
}

/// Fixed per-thread stack length: `MAX_PLY` search plies plus headroom, matching
/// the HalfKA accumulator stack ([`super::sota_accumulator`]).
const STACK_LEN: usize = MAX_PLY + 8;

/// One per-ply FullThreats entry, folded **fully lazily** — the FullThreats
/// mirror of [`super::sota_accumulator::HalfkaContext`]'s per-ply `Entry`. At
/// push time we store only the move ([`dp`](Self::dp)) and a
/// [`ThreatPosition`] of the post-move board; generation, resolution, and the
/// 768-lane fold are all deferred to [`ThreatContext::compute`], which runs
/// them against the snapshot only for the plies that are actually evaluated.
// Cache-line aligned with the accumulator entries (perf study 3 §5.3).
#[derive(Clone)]
#[repr(align(64))]
struct ThreatEntry {
    /// `[White, Black]` accumulator pair — valid only where `fold[p]` is
    /// [`Fold::Done`].
    acc: [[i16; ACC]; 2],
    /// T6c PSQT bucket sums over the ACTIVE THREAT features, maintained by the
    /// same fold events as `acc` (all-zero when the net carries no threat
    /// PSQ1 rows). Valid where `fold[p]` is [`Fold::Done`], like `acc`.
    psqt: [[i32; BUCKETS]; 2],
    /// Perf study 51 arm `cong51`: 64 bytes of pure padding. It changes nothing
    /// an evaluation can observe and everything about which L1 SET this entry's
    /// accumulator lands in.
    ///
    /// `ThreatEntry` and HalfKA's `Entry` have byte-identical layouts (5120 B
    /// of `acc` + 64 B of `psqt`, `align(64)`, stride **5184**) and both stacks
    /// are `Box<[_; 254]>` of exactly 1,316,736 B, so the allocator returns
    /// both at the same page offset and they come out EXACTLY 4 KB-congruent —
    /// study 50 measured the separation at `0x197000 = 407 x 4096`, and study
    /// 51 reproduced it on an independently ASLR'd run. L1d indexes on
    /// `addr[11:6]`, so `halfka.stack[p].acc[c]` and `threats.stack[p].acc[c]`
    /// contend for the same 64 sets at EVERY ply — and `activate` reads all
    /// four of those arrays interleaved.
    ///
    /// Study 50 recorded this as its own unfinished business: *"the clean
    /// experiment — offset one stack by a non-multiple of 4096 and re-run the
    /// region census — is two lines and I did not build it"*. Padding the
    /// stride to 5248 makes the same-ply separation `D + 64p`, which is
    /// 4 KB-congruent for only one ply in 64.
    ///
    /// Bit-identical by construction: no code reads it.
    #[cfg(feature = "cong51")]
    _cong: [u8; 64],
}

/// The per-ply PUSH RECORD — the move that produced the entry plus the
/// post-move board state its events regenerate against — in its OWN dense
/// array (perf study 31 arm p1), taking the shape studies 22 and 24 landed on
/// `Entry.dirty`/`computed` and `ThreatEntry.fold`.
///
/// # ⚠ THE LOCALITY HYPOTHESIS THIS SPLIT WAS BUILT TO TEST IS **REFUTED**
///
/// Read this before proposing anything else about the snapshot's placement.
///
/// **The hypothesis.** Perf study 28 §7 measured `search::<NonPv>` — into which
/// [`ThreatContext::apply_at`] is inlined — stalling **3.03× Stockfish's search
/// symbol** (+303.4 cycles/node, **26.7% of the whole 1.37× per-node gap**) and
/// named *this snapshot* as the mechanism: "a `DirtyPiece` and an 80-byte
/// `ThreatPosition` written every node into a stack whose per-ply stride is
/// ~10.5 KB … a write into a cold line at every single node, evaluated or not".
/// Study 24 had declined the split on a READ argument (`dp`/`position` "are read
/// exactly ONCE per folded child, never scanned"), which is true and is not the
/// same claim: 100% of nodes WRITE this record and only ~41.5% ever fold it.
///
/// **The measurement** (perf study 31, `tools/perf31/stalls31.sh`, 12
/// randomised-order round-robin rounds against a byte-identical null,
/// node-identical at 5,001,562 and eval-hash-identical at `5ca978e0…`):
///
/// | counter | base | NULL | ARM (this split) |
/// |---|---:|---:|---:|
/// | L1-dcache-load-misses/node | 618.08 | −0.160% | **+0.254% (+1.57)** |
/// | `ex_no_retire.load_not_complete`/node | 2836.01 | +2.100% | **+3.583%** |
/// | instructions/node | 5653.84 | +0.000% | −0.014% (−0.78) |
///
/// **Moving the record from a 5,312-byte stride to an 88-byte one changes the
/// engine's L1 miss count by nothing** — +1.57/node against a null of −1.0/node,
/// on the one counter here whose null is ±0.16%. The lines were already hot:
/// only ~15 plies are live at a time and `compute` walks them repeatedly, so the
/// top of a 5.3 KB-strided stack is L1/L2-resident, not cold.
///
/// **What that means for study 28's item C:** the +303.4 cycles/node is a
/// SITE localisation, not a mechanism. The named mechanism is now excluded;
/// whatever `search::<NonPv>` is stalling on (TT probes into 16 MB, the
/// butterfly/continuation history planes, movegen) has not been identified.
/// Do not re-price item C off this snapshot.
///
/// **Why the split is kept anyway:** −0.78 instructions/node and −0.44
/// branches/node against an exact 0.000% null, 12/12 sign-consistent, and it
/// returns [`ThreatEntry`] to 5,248 bytes — the same size and shape as
/// `sota_accumulator`'s `Entry`. It is kept for the encoded negative result
/// above at least as much as for the instructions.
#[derive(Clone, Copy, Default)]
struct Push {
    /// Read only by the [`Fold::FromParent`] path ([`build_deltas`]).
    dp: DirtyPiece,
    /// Read by BOTH deferred paths: [`Fold::FromParent`] runs [`build_deltas`]
    /// against it, and [`Fold::Refresh`] rebuilds the whole accumulator out of
    /// it.
    position: ThreatPosition,
}

/// The per-ply fold state, in its OWN dense array — perf study 24 arm h1,
/// taking the shape study 22 landed on `sota_accumulator`'s `Entry`.
///
/// `fold` used to be the last two bytes of [`ThreatEntry`], a **5,312-byte**
/// struct (`imul $0x14c0` on every index). [`ThreatContext::compute`] opens by
/// scanning it backwards, per perspective, looking for the nearest ancestor
/// that is not [`Fold::FromParent`] — so a walk of `d` plies touched `d`
/// guaranteed-distinct cache lines, 5 KB apart, to read two bytes from each,
/// and it does that walk **twice per evaluated node**. As its own
/// `[[Fold; 2]; STACK_LEN]` the whole walk is 2 bytes per ply: **32 plies to a
/// 64-byte line**, and the deepest realistic walk fits in one or two.
///
/// `dp`/`position` stay together in [`ThreatEntry`] rather than moving to a
/// third array: unlike `fold` they are read exactly ONCE per folded child,
/// never scanned, so their locality does not matter and splitting them would
/// only add another base pointer to an already register-starved `compute`.
type FoldState = [Fold; 2];

/// How [`ThreatContext::compute`] must materialise one perspective of one
/// [`ThreatEntry`].
///
/// The third state is perf study 18's arm: a king-crossed ply *used* to be
/// rebuilt eagerly inside [`ThreatContext::apply_at`], i.e. on every made move,
/// even though only ~41.5% of made moves are ever evaluated. Recording the
/// obligation instead of discharging it makes the rebuild obey the same laziness
/// as everything else in this module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fold {
    /// `acc[p]`/`psqt[p]` already hold this ply's value.
    Done,
    /// Fold the parent's value forward through this ply's `dp`/`position`.
    FromParent,
    /// The perspective's own king crossed the d/e mirror on this move, so every
    /// one of its feature indices is re-oriented and the parent's accumulator is
    /// worthless: rebuild from scratch out of `position`. Cannot be folded, and
    /// terminates `compute`'s ancestor walk exactly as [`Fold::Done`] does.
    Refresh,
}

impl Default for ThreatEntry {
    fn default() -> Self {
        ThreatEntry {
            acc: [[0i16; ACC]; 2],
            psqt: [[0i32; BUCKETS]; 2],
            #[cfg(feature = "cong51")]
            _cong: [0u8; 64],
        }
    }
}

/// A per-ply stack of FullThreats accumulators, maintained **lazily** — the
/// FullThreats counterpart of [`super::sota_accumulator::HalfkaContext`]
/// (design/11 §12.9.1, R4). Format-agnostic: it borrows the `i8` threat-weight
/// slice (feature-major, `THREAT_DIMS · ACC`), exactly like [`refresh_threat_into`].
///
/// Like HalfKA — whose dirty record is re-resolved to feature indices at fold
/// time — FullThreats now defers *everything*: [`apply_at`](Self::apply_at)
/// snapshots the post-move board ([`ThreatPosition`]) + the [`DirtyPiece`], and
/// [`compute`](Self::compute) regenerates the events, resolves the
/// per-perspective indices, and folds forward from the nearest computed ancestor
/// only for plies that reach an [`evaluate`]. Children that cut off (TT hits,
/// immediate fail-lows) never pay generation, resolution, *or* the fold. The
/// snapshot's x-ray geometry is the *post-move* board's, captured before a
/// sibling can mutate it in place, so the deferred work is bit-identical to the
/// old eager path.
///
/// [`evaluate`]: super::sota_context::SotaContext::evaluate
pub struct ThreatContext<'a> {
    weights: &'a [i8],
    /// Perf study 65 — the per-feature "this row is stored narrow" bitmap, or
    /// empty when the loader left the table in its 8-bit form. Set by
    /// [`with_narrow`](Self::with_narrow) so the five constructors and every
    /// test keep their signatures.
    narrow: &'a [u64],
    /// T6c PSQ1 rows for the THREAT feature space (`threat_rows · BUCKETS`
    /// i32, row-major, indexed by the resolved threat feature index), or
    /// empty ⇒ the skip channel is inert.
    psqt_weights: &'a [i32],
    /// The net's threat re-coding (`None` = identity) — see
    /// [`ThreatRemap`]. Applied inside [`ThreatResolver::index_decoded`], so
    /// every resolved index below is already in the net's row space.
    remap: Option<&'a ThreatRemap>,
    /// One lazily-folded entry per ply; `stack[0]` is the root. Indexed by
    /// absolute search ply and never truncated — a sibling simply overwrites its
    /// ply's entry — exactly like the HalfKA stack. Heap-allocated once.
    /// **Sized** `Box<[T; N]>`, not `Box<[T]>` — study 22 paid for this one and
    /// handed it over: doing exactly this hot/cold split against fat-pointer
    /// arrays measured **+58.7 i/n**, a regression, purely in the length loads
    /// and dynamic bounds checks a slice box forces at every index; the same
    /// split with all arrays sized measured **−9.3 i/n**. A `Box<[T; N]>` is a
    /// thin pointer with a compile-time stride and statically discharged
    /// bounds. The boxes are built through `vec![..].into_boxed_slice()` and
    /// `try_into`, which is a pointer cast — never a 1.3 MB array on the stack.
    stack: Box<[ThreatEntry; STACK_LEN]>,
    /// Per-ply move + post-move board snapshot, in its own dense array so the
    /// every-node WRITE is a sequential 88-byte-stride stream instead of two
    /// cold lines 5.3 KB apart — see [`Push`].
    push: Box<[Push; STACK_LEN]>,
    /// Per-ply, per-perspective fold state. Its own dense array so `compute`'s
    /// ancestor walk is a linear scan over 2 bytes per ply — see [`FoldState`].
    fold: Box<[FoldState; STACK_LEN]>,
    /// Per-ply **value holder**: the ply whose [`ThreatEntry`] actually contains
    /// ply `k`'s accumulator pair. `alias[k] == k` for a materialised entry;
    /// `alias[k] < k` when ply `k`'s fold was a pure parent copy for BOTH
    /// perspectives and the entry was therefore never written at all
    /// (perf study 38).
    ///
    /// The invariant, and every write site maintains it:
    /// **if `fold[k][p] == Fold::Done` then perspective `p`'s value is in
    /// `stack[alias[k]].acc[p]`** — and `alias[alias[k]] == alias[k]`, so a
    /// read is always ONE hop (the alias case copies the parent's holder rather
    /// than pointing at the parent).
    ///
    /// It is a whole-ENTRY alias, not a per-perspective one, because
    /// [`at_full`](Self::at_full) hands out `&[[i16; ACC]; 2]` — both
    /// perspectives from one entry — and a per-perspective alias could not be
    /// returned without changing the head merge kernel's parameter shape. The
    /// measurement that says that is enough: of 1,694,891 pure-copy folds on the
    /// study frame, **1,584,928 (93.5%) come from plies where BOTH perspectives
    /// resolve empty together** (`foldcount` `thr_joint.n11e11`), so the cheap
    /// form captures almost all of it.
    ///
    /// **INLINE, not `Box`ed**, and that is a measured choice rather than a
    /// stylistic one. `stack`, `push` and `fold` are each a boxed array, so each
    /// costs `compute` a base pointer; a fourth one is exactly what `Push`'s own
    /// doc comment declines a further split over ("another base pointer to an
    /// already register-starved `compute`"), and what study 22 priced at
    /// +58.7 i/n in this same walk. 508 bytes at a fixed offset from `self` —
    /// which is live anyway — needs no register of its own, and it is small
    /// beside the 8,192 bytes of `add_scratch`/`sub_scratch` already inline here.
    ///
    /// Behind [`AliasTable`] rather than a bare array so perf study 40 can
    /// delete the whole mechanism at one site (`--features noalias40`).
    alias: AliasTable,
    /// Scratch raw-event buffer for [`build_deltas`], reused every fold.
    buf: DeltaBuf,
    /// Fold-time resolved-index scratch for [`apply_threat_deltas_into`], written by
    /// [`resolve_into`] and reused every [`compute`](Self::compute) — never a
    /// per-node allocation, and never re-zeroed (only the `..len` prefix is
    /// read). One pair per perspective so BOTH perspectives resolve (and
    /// prefetch their weight rows) before either folds — the perf study #4
    /// item #3 resolve→prefetch→fold pipeline.
    add_scratch: [[RowOffset; DELTA_CAP]; 2],
    sub_scratch: [[RowOffset; DELTA_CAP]; 2],
    /// Top-of-stack ply for the cursor form of the API
    /// ([`apply`](Self::apply)/[`current`](Self::current)/[`undo`](Self::undo),
    /// which the standalone tests drive). The search stack uses the explicit
    /// [`apply_at`](Self::apply_at)/[`at`](Self::at) ply form instead.
    top: usize,
}

/// The per-ply value-holder table of [`ThreatContext::alias`], behind two
/// methods so the whole mechanism has exactly ONE definition site.
///
/// Perf study 40 built this to make `--features noalias40` possible: that
/// feature swaps the table for a zero-sized stand-in whose `get(k)` is `k`,
/// which is bit-for-bit the pre-study-38 behaviour — the alias reconstructed
/// away in place, rather than checked out of git and re-benchmarked on a
/// different tree. It exists because study 38's "+19.7 i/n of machinery" was
/// measured against a binary whose copy ALSO compiled differently (a `memcpy`
/// call, against 40 inline zmm moves here), so that number is the sum of two
/// effects and not the machinery. See `docs/research/perf-study-40-*.md` §2.
///
/// `get`'s `.min(k)` is the range fact `ThreatContext::at_full` used to spell
/// at its call site: a no-op on the value (the invariant is `alias[k] <= k`)
/// and the only reason the load carries no `panic_bounds_check` edge.
#[cfg(not(feature = "noalias40"))]
struct AliasTable([u16; STACK_LEN]);

#[cfg(not(feature = "noalias40"))]
impl AliasTable {
    fn new() -> Self {
        const { assert!(STACK_LEN <= u16::MAX as usize) };
        // Identity: every entry starts as its own holder, which is the correct
        // reading of a stack whose `fold` is all `FromParent`. `from_fn`
        // because the array is inline and const-sized — no allocation.
        Self(std::array::from_fn(|i| i as u16))
    }

    /// The ply that actually holds ply `k`'s value. Always ONE hop.
    #[inline(always)]
    fn get(&self, k: usize) -> usize {
        usize::from(self.0[k]).min(k)
    }

    #[inline(always)]
    fn set(&mut self, k: usize, holder: usize) {
        self.0[k] = holder as u16;
    }
}

/// The pure-copy fold's 2560-byte accumulator copy, behind a call boundary
/// (perf study 40 §2.5).
///
/// **`#[inline(never)]` is the entire point of this function and it is a
/// MEASURED choice, not a style one.** Expanded inline, this copy is an
/// 83-instruction straight-line zmm block sitting inside `compute`'s fold loop
/// — and `compute` is inlined into `at_full`, the module's hottest symbol.
/// Moving the block behind a call boundary measured, over 8 paired
/// within-round rounds with **zero spread** and a byte-identical null at
/// 0.000%:
///
/// | counter | vs the inline form |
/// |---|---:|
/// | `instructions:u` / node | **−0.270%** (−16.01) |
/// | `branches:u` / node | **−0.195%** (−1.21) |
/// | `ex_ret_brn_misp:u` / node | −0.578%, 7/12 rounds — no effect |
///
/// The copy runs **0.022 times per node** in the shipped engine (study 38's
/// parent alias elides 93.5% of them), so under 2 of those 16 instructions can
/// be the copy itself. The rest is the surrounding function compiling better
/// without the block in it. Verified node-identical (5,001,562) and
/// eval-hash-identical (`2e1ede9521624dd8`).
///
/// Study 14 recorded "45.4 i/n between two spellings of this copy" and called
/// it *spelling, not semantics*. This is the same effect with a sign and a
/// size — and unlike study 14's comment, it is checkable: `tools/perf40/
/// copycheck40.sh` reads the codegen off the binary, and a build where this
/// helper has been inlined away will show `vmovups=80` inside `at_full`.
///
/// `--features callcopy40` replaces it with the libc call, which measures the
/// same −15.92; it exists only as the instrument that separated this effect
/// from the copy's own cost.
///
/// **Do NOT generalise this into "outline the big kernels".** Perf study 26
/// §6.1 outlined the AVX-512 `l2_tail` for a similarly good codegen reason and
/// **lost 4.26 i/n**, because its caller then spilled its live state around the
/// call — on *every* evaluation. The difference is the CALL RATE: outlining
/// charges a spill/reload per call and pays for it once in the body, so it wins
/// where the block is large and the rate is low. `l2_tail` runs ~0.43 times per
/// node; this copy runs 0.022, and only because study 38's parent alias made it
/// rare. The two results together are the rule; either alone is a bad
/// generalisation.
#[cfg(not(feature = "callcopy40"))]
#[inline(never)]
fn copy_acc_outlined(dst: &mut [i16; ACC], src: &[i16; ACC]) {
    // `chunks_exact` DROPS a remainder, so a width that is not a multiple of 32
    // would silently leave the tail of `dst` holding a stale sibling's lanes.
    const { assert!(ACC.is_multiple_of(32)) };
    for (d, s) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
        d.copy_from_slice(s);
    }
}

/// Perf study 40's null: no table, no reads, no stores — every ply is its own
/// holder, which is what the module did before study 38.
#[cfg(feature = "noalias40")]
struct AliasTable;

#[cfg(feature = "noalias40")]
impl AliasTable {
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn get(&self, k: usize) -> usize {
        k
    }

    #[inline(always)]
    fn set(&mut self, _k: usize, _holder: usize) {}
}

impl<'a> ThreatContext<'a> {
    /// A fresh context over `weights` (the `i8` FullThreats weight slice) plus
    /// the net's threat PSQ1 rows (`&[]` when it has none). Call
    /// [`refresh`](Self::refresh) to seed the root before any [`apply`](Self::apply).
    /// Install the loader's narrow-plane bitmap (perf study 65). Empty — the
    /// default — means every row is stored 8-bit and the fold takes exactly the
    /// path it took before this study.
    pub fn with_narrow(mut self, narrow: &'a [u64]) -> Self {
        self.narrow = narrow;
        self
    }

    pub fn new(weights: &'a [i8], psqt_weights: &'a [i32], remap: Option<&'a ThreatRemap>) -> Self {
        #[cfg(feature = "foldcount")]
        {
            use super::foldcount::region;
            region(
                "thr_weights",
                weights.as_ptr().cast(),
                std::mem::size_of_val(weights),
            );
            region(
                "thr_psqt",
                psqt_weights.as_ptr().cast(),
                std::mem::size_of_val(psqt_weights),
            );
            // Perf study 51: see the twin call in `HalfkaContext::new`. The
            // threat table is 77.7 MB and 54.7% of all folded weight rows, so
            // it is the one that decides whether a narrower weight is possible.
            static THR_VH: std::sync::Once = std::sync::Once::new();
            THR_VH.call_once(|| {
                super::foldcount::valhist(
                    "thr_weights_p128",
                    weights.iter().map(|&w| i64::from(w) + 128),
                );
            });
            static THR_WEIGHTS_RS: std::sync::Once = std::sync::Once::new();
            THR_WEIGHTS_RS.call_once(|| {
                super::foldcount::rowstats("thr_weights", weights, super::sota_head::L1);
                super::foldcount::rowwidth_init(1, weights, super::sota_head::L1);
                super::foldcount::lanestats("thr_weights", weights, super::sota_head::L1);
            });
        }
        #[allow(unused_mut)]
        let mut me = ThreatContext {
            weights,
            narrow: &[],
            psqt_weights,
            remap,
            // `try_into` rather than `expect`: `ThreatEntry` is not `Debug`
            // (it is 5 KB of accumulator) and `Result::expect` demands it.
            stack: match vec![ThreatEntry::default(); STACK_LEN]
                .into_boxed_slice()
                .try_into()
            {
                Ok(b) => b,
                Err(_) => unreachable!("vec! built exactly STACK_LEN entries"),
            },
            // `try_into` + `unreachable!`, not `expect`: `Push` embeds a
            // `ThreatPosition`, which is not `Debug`, and `Result::expect`
            // demands it (same reason as `stack` above).
            push: match vec![Push::default(); STACK_LEN]
                .into_boxed_slice()
                .try_into()
            {
                Ok(b) => b,
                Err(_) => unreachable!("vec! built exactly STACK_LEN push records"),
            },
            fold: vec![[Fold::FromParent; 2]; STACK_LEN]
                .into_boxed_slice()
                .try_into()
                .expect("STACK_LEN fold states"),
            alias: AliasTable::new(),
            buf: DeltaBuf::new(),
            add_scratch: [[RowOffset::default(); DELTA_CAP]; 2],
            sub_scratch: [[RowOffset::default(); DELTA_CAP]; 2],
            top: 0,
        };
        #[cfg(feature = "foldcount")]
        {
            use super::foldcount::region;
            region(
                "thr_stack",
                std::ptr::from_ref(&*me.stack).cast(),
                std::mem::size_of_val(&*me.stack),
            );
        }
        me
    }

    /// Rebuild the root accumulator from scratch for `board`, both perspectives,
    /// and reset the cursor to that single entry. The from-scratch build reuses
    /// [`refresh_threat_into_at`] so the incremental and refresh paths share one
    /// definition of the feature set.
    pub fn refresh(&mut self, board: &Board) {
        self.top = 0;
        // CENN-v2 explicitly supports HalfKA-only networks.  Their threat
        // transformer is represented by an empty slice and contributes the
        // all-zero accumulator; never index that empty slice through the
        // ordinary FullThreats refresh path.
        if self.weights.is_empty() {
            self.stack[0].acc = [[0i16; ACC]; 2];
            self.stack[0].psqt = [[0i32; BUCKETS]; 2];
        } else {
            let weights = self.weights;
            let narrow = self.narrow;
            let psqt_weights = self.psqt_weights;
            let remap = self.remap;
            // One snapshot for all four rebuilds, not four.
            let pos = ThreatPosition::from_board(board);
            for (i, &persp) in Color::ALL.iter().enumerate() {
                refresh_threat_into_at(
                    &mut self.stack[0].acc[i],
                    weights,
                    narrow,
                    &pos,
                    persp,
                    remap,
                );
                self.stack[0].psqt[i] = refresh_psqt_threat_at(psqt_weights, &pos, persp, remap);
            }
        }
        self.fold[0] = [Fold::Done; 2];
        self.alias.set(0, 0);
    }

    /// The current (cursor top) accumulator pair, `[White, Black]` — the input
    /// [`super::sota_head::SotaHead::forward`] consumes as `threat`. Folds the
    /// entry lazily first, so it now takes `&mut self`.
    pub fn current(&mut self) -> &[[i16; ACC]; 2] {
        let top = self.top;
        self.at(top)
    }

    /// [`current`](Self::current) plus the T6c PSQT bucket sums (cursor form
    /// of [`at_full`](Self::at_full)).
    pub fn current_full(&mut self) -> (&[[i16; ACC]; 2], &[[i32; BUCKETS]; 2]) {
        let top = self.top;
        self.at_full(top)
    }

    /// Accumulator at a search ply, folded lazily from the nearest computed
    /// ancestor. The search overwrites sibling plies rather than calling `undo`,
    /// so callers use this indexed form. Now `&mut self` (it may trigger a fold).
    pub fn at(&mut self, ply: usize) -> &[[i16; ACC]; 2] {
        self.at_full(ply).0
    }

    /// [`at`](Self::at) plus the per-perspective T6c PSQT bucket sums of the
    /// SAME entry (all-zero for a net without threat PSQ1 rows).
    pub fn at_full(&mut self, ply: usize) -> (&[[i16; ACC]; 2], &[[i32; BUCKETS]; 2]) {
        // There USED to be a null-move alias walk here (perf study #4 item #3):
        // an EMPTY dirty record's "fold" is a pure 2 x 2.5 KB parent copy, so
        // the walk handed back the parent instead of materialising the child.
        // Perf study 15 arm 1 removed the entry itself — the search pushes
        // nothing for a null move (`EvalContext::null_plies`), which is
        // Stockfish's shape (`do_null_move` at `eca43a97:src/search.cpp:657-662`
        // pushes no accumulator entry, `undo_null_move` at `:669` pops none) —
        // so no EMPTY record ever reaches this stack from a search and the walk
        // could never move. It cost ~4 tests per evaluated node to prove that.
        //
        // The cursor API (`apply`) can still be handed an empty record by a
        // caller outside the search; that entry is now MATERIALISED by the fold
        // below (`build_deltas` yields no events, so the empty-list branch
        // copies it), which is the same value the alias returned.
        self.compute(ply);
        // Perf study 38: `alias[ply]` is `ply` for a materialised entry and an
        // ancestor for a ply whose fold was a pure parent copy — see the
        // field's own doc.
        //
        // `.min(ply)` is a NO-OP on the value and a RANGE FACT for the
        // optimizer — the same move as the threat resolver's `& 63`. The
        // invariant is `alias[k] <= k`, but LLVM only sees `u16 → 0..=65535`
        // and cannot prove the index is inside a `[ThreatEntry; STACK_LEN]`,
        // so without it this load carries a compare and a
        // `panic_bounds_check` edge in the hottest symbol in the module.
        // `ply` is already known in range: `compute(ply)` indexed
        // `self.fold[ply]` one line above. It is also the same defensive shape
        // as `EvalContext::acc_ply`'s `saturating_sub` — if the invariant were
        // ever broken this reads a stale but IN-RANGE entry instead of walking
        // off the stack.
        let entry = &self.stack[self.alias.get(ply)];
        (&entry.acc, &entry.psqt)
    }

    /// Record (but do NOT fold) the move described by `dp` (made to reach
    /// `board`) into the child entry at `ply + 1`, advancing the cursor. Every
    /// perspective gets the post-move [`ThreatPosition`] stored for
    /// [`compute`](Self::compute) to (re)generate and fold later; a perspective
    /// whose own king crossed the d/e boundary is additionally marked
    /// [`Fold::Refresh`], so `compute` rebuilds it from that snapshot instead of
    /// folding it from its parent. `board` is the
    /// fully post-move position. This is the cursor form of
    /// [`apply_at`](Self::apply_at).
    pub fn apply(&mut self, dp: &DirtyPiece, board: &Board) {
        let ply = self.top;
        self.apply_at(ply, dp, board);
    }

    /// Record a post-move dirty record from `ply` to `ply + 1` (search-stack
    /// form of [`apply`](Self::apply)). A stale sibling entry at `ply + 1` is
    /// fully overwritten — every perspective's `fold` state plus the move and
    /// snapshot are reset here, so no truncation is needed and deeper stale
    /// entries are never read (`compute` only ever walks toward the root).
    #[inline(always)]
    pub fn apply_at(&mut self, ply: usize, dp: &DirtyPiece, board: &Board) {
        let child = ply + 1;
        debug_assert!(child < STACK_LEN, "ThreatContext ply {child} exceeds stack");
        self.top = child;

        // HalfKA-only net: the threat channel is a constant all-zero accumulator
        // (and all-zero PSQT sums), so there is nothing to inherit and nothing
        // to fold — just mark the child computed.
        //
        // This used to copy the parent's accumulator into the child, which is a
        // provable no-op: every entry starts zero (`ThreatEntry::default`),
        // `refresh` re-zeroes the root explicitly on this branch, and with empty
        // weights no other site ever writes an accumulator (`discharge_refresh`
        // is reachable only from a `Fold::Refresh`, which only the code below
        // this return ever sets, and `compute` exits immediately because every
        // entry is already `Fold::Done`). The copy was therefore 2·ACC·2 bytes of pure memory
        // traffic on every pushed node — 8 KB/node at the 2048 width the
        // HalfKA-only probe exists to buy, i.e. a tax that grows with exactly
        // the dimension the experiment is spending its speed budget on. It was
        // also present in perf study 7's stripped arms, so the measured
        // 1.25–1.39× threat cost UNDERSTATES what a properly-built
        // HalfKA-only engine reaches.
        if self.weights.is_empty() {
            self.fold[child] = [Fold::Done; 2];
            // The entry is never written on this branch and every entry is
            // all-zero, so the ply is its own holder by construction.
            self.alias.set(child, child);
            return;
        }

        // Snapshot the post-move inputs the observer needs at fold time
        // (occupancy by type/colour + king squares, 74 B used of 80 padded)
        // plus the move itself.
        // Every fold-time (re)generation reads THIS, never the live board a
        // sibling will overwrite. Generation, resolution, and the fold are all
        // deferred.
        //
        // Rebuilt from `board`, NOT derived from the parent's by XOR. Perf
        // study 15 arm 3 measured that derivation — study 14 §8 item 5, which
        // predicted 15-25 i/n of savings from "<=4 `toggle`s" instead of six
        // `pieces_of_type` unions — and it cost **+106 i/n**, node- and
        // eval-identical (perf-artifacts-15/probes-fast.txt,
        // `a3-threatpos-from-parent`, with `a3r-revert-control` reproducing the
        // baseline to 0.005 i/n). The arithmetic favoured it, ~18 ops against
        // ~25, so the arithmetic is not what decides here.
        //
        // WHY it lost is NOT established (D33 reviewer B, m2). Two candidates,
        // neither measured: the parent and child snapshots are two elements of
        // the SAME `Box<[ThreatEntry]>`, so LLVM has no disjointness fact for
        // the 80-byte read against the 80-byte write; and the edit list is a
        // runtime-length loop where `from_board` is straight-line. The
        // codebase's own aliasing idiom for exactly this shape —
        // `split_at_mut`, as `HalfkaContext::compute` uses — was NOT tried.
        // Do not re-derive this without re-measuring.
        self.push[child] = Push {
            dp: *dp,
            position: ThreatPosition::from_board(board),
        };

        // Learned-representation study Tier-0: unconditional make-time threat
        // EVENT census. The production path defers `build_deltas` to fold time
        // and so only ever counts the plies that survive deferral; replaying
        // it here (instrument-only) counts what a non-deferring engine would
        // pay on EVERY made move. `compute` clears `self.buf` before its own
        // use, so this leaves no state behind; the search tree, node count and
        // eval hash are untouched (verified by the neutrality gate).
        #[cfg(feature = "foldcount")]
        {
            self.buf.clear();
            build_deltas(&mut self.buf, tables(), dp, &self.push[child].position);
            super::foldcount::note_thr_make(self.buf.len);
        }

        // A perspective whose king crossed the d/e mirror cannot fold from its
        // parent — every index is re-oriented — so it must be rebuilt from
        // scratch out of the snapshot above. RECORD that obligation; do not
        // discharge it (perf study 18 §3).
        //
        // This used to call `refresh_threat_into` + `refresh_psqt_threat`
        // RIGHT HERE, on the live post-move board. Study 17 §3.3 priced that:
        // **222.7 i/n, 25% of the entire make side**, 6,240 instructions per
        // call at 0.0357 calls/node.
        //
        // How much of it is saved: **41.0%**, measured — the call rate falls
        // 0.0363 → 0.0214 per node. NOT the ~58% study 17 predicted from "only
        // 41.5% of nodes evaluate". **Laziness here is a SUBTREE property, not
        // a node property**: a crossing ply that is never evaluated ITSELF is
        // still refreshed if any DESCENDANT evaluates, because `compute` walks
        // back to it from that descendant. Only a crossing whose entire subtree
        // cuts off is saved. Any future "defer it and the nodes that never
        // evaluate stop paying" argument on this stack has to be written
        // against subtrees. The reason it was eager was
        // that the rebuild took a `&Board` and the live board is gone by fold
        // time; the fix is that the snapshot beside it already carries every
        // input the enumeration reads (`ThreatPosition` is now the from-scratch
        // path's own argument type — `for_each_threat_at`), so `compute` can do
        // the rebuild at its leisure.
        // `[T; N]::map` rather than an indexed loop (D33 reviewer C): one
        // address computation and one 2-byte store instead of two of each
        // through a 5,248-byte-strided `self.stack[child]`.
        self.fold[child] = king_cross(dp).map(|crossed| {
            if crossed {
                Fold::Refresh
            } else {
                Fold::FromParent
            }
        });
    }

    /// Fold `stack[ply]`'s accumulator for **both** perspectives, memoizing them
    /// — the FullThreats mirror of
    /// [`super::sota_accumulator::HalfkaContext::compute`], but fully lazy:
    /// generation ([`build_deltas`]) and resolution ([`resolve_into`]) happen
    /// here too, against each folded ply's stored [`ThreatPosition`]. Walks back
    /// to the nearest **foldable-from** ancestor per perspective — one that is
    /// either already [`Fold::Done`] or a [`Fold::Refresh`] ply this call
    /// discharges on the spot (the root is always `Done`, so the walk always
    /// terminates) — then, for each child across the union span, regenerates the
    /// perspective-agnostic events ONCE and resolves + folds each perspective
    /// that still needs it.
    fn compute(&mut self, ply: usize) {
        if self.fold[ply][0] == Fold::Done && self.fold[ply][1] == Fold::Done {
            return;
        }
        let weights = self.weights;
        let narrow = self.narrow;
        let psqt_weights = self.psqt_weights;
        // Nearest ancestor per perspective that is not a fold-from-parent ply.
        // `Refresh` stops the walk exactly like `Done` does — it is a ply whose
        // value does not come from its parent — and is then materialised here,
        // out of its own snapshot, so the child loop below sees a `Done`
        // ancestor either way.
        let mut anc = [ply; 2];
        for (p, a) in anc.iter_mut().enumerate() {
            // Carry the terminal `fold` in a local (D33 reviewer C, P-1): the
            // loop has already loaded it, and re-reading `self.stack[*a].fold[p]`
            // afterwards costs a second `ply * size_of::<ThreatEntry>()` address
            // computation plus a load, twice per `compute`.
            let mut f = self.fold[*a][p];
            while *a > 0 && f == Fold::FromParent {
                *a -= 1;
                f = self.fold[*a][p];
            }
            // The 3-state marker can finally tell "unseeded root" from
            // "uncomputed root", which the old `bool` could not (D33 reviewer
            // A, A7). Falling off ply 0 still `FromParent` means `refresh()`
            // was never called and every child is about to fold from an
            // all-zero accumulator — a silently wrong eval, not a crash.
            debug_assert!(
                *a > 0 || f != Fold::FromParent,
                "ThreatContext ancestor walk fell off an unseeded root at ply {ply}: \
                 refresh() was never called"
            );
            if f == Fold::Refresh {
                discharge_refresh(
                    &mut self.stack[*a],
                    &self.push[*a].position,
                    weights,
                    narrow,
                    psqt_weights,
                    p,
                    self.remap,
                );
                // `fold` no longer lives on the entry, so the discharge's last
                // act happens here (perf study 24 arm h1). It also keeps
                // `discharge_refresh` at exactly six integer argument
                // registers -- see its doc comment.
                self.fold[*a][p] = Fold::Done;
                // A rebuild WRITES this entry, so the ply is now its own value
                // holder. (It cannot have been aliased: the alias below is only
                // taken when both perspectives are `FromParent`.)
                self.alias.set(*a, *a);
            }
        }
        // Hoist the magic-table LazyLock deref out of the per-touch geometry
        // queries (CLAUDE.md's documented hot-loop pattern for `attacks`): the
        // guard was being re-checked several times per touched square.
        let tab = tables();
        for child in (anc[0].min(anc[1]) + 1)..=ply {
            let need = [
                child > anc[0] && self.fold[child][0] == Fold::FromParent,
                child > anc[1] && self.fold[child][1] == Fold::FromParent,
            ];
            if !need[0] && !need[1] {
                continue;
            }
            // Regenerate the perspective-agnostic raw events once, from this
            // ply's stored snapshot — shared across both perspectives.
            self.buf.clear();
            build_deltas(
                &mut self.buf,
                tab,
                &self.push[child].dp,
                &self.push[child].position,
            );
            // Phase 1 — resolve EVERY needed perspective before ANY fold runs
            // (perf study #4 item #3). The [`ThreatResolver`] hoists the
            // `THREAT_TABLES` LazyLock deref + orientation/color-swap out of
            // the per-event loop. NOTE: the study's software row-prefetch was
            // implemented here and MEASURED A LOSS (−2.5% NPS full-row, ~−0.4%
            // first-two-lines, bench-with-net A/B on the Zen 4 box) — the OoO
            // window + hardware stride prefetchers already cover the fold's
            // row streams — so the pipeline keeps only the resolve/fold split
            // and the constant-factor hoists.
            let mut lens = [(0usize, 0usize); 2];
            // T6c: the psqt bucket-sum DELTA (Σ add rows − Σ sub rows) is
            // accumulated during resolve — the row loads (one random 32 B line
            // per event in the 2.7 MB PSQ1 table) overlap the per-event
            // resolution compute in the OoO window instead of serializing as
            // a second walk after the big fold. Phase 2 then applies
            // `child = parent + delta` (8 adds). Zero when the net has no
            // threat PSQ1 rows.
            let mut psqt_deltas = [[0i32; BUCKETS]; 2];
            if need[0] && need[1] {
                // Both perspectives wanted — the common case. One pass over
                // the buffer, sharing the per-event decode (perf study 9
                // item 1).
                let kings = self.push[child].position.kings;
                let resolvers = [
                    ThreatResolver::with_remap(Color::White, kings[0], self.remap),
                    ThreatResolver::with_remap(Color::Black, kings[1], self.remap),
                ];
                lens = resolve_both(
                    self.buf.events(),
                    &mut self.add_scratch,
                    &mut self.sub_scratch,
                    &resolvers,
                    psqt_weights,
                    &mut psqt_deltas,
                );
            } else {
                for (p, &needs_fold) in need.iter().enumerate() {
                    if !needs_fold {
                        continue;
                    }
                    let ksq = self.push[child].position.kings[p];
                    let resolver = ThreatResolver::with_remap(Color::ALL[p], ksq, self.remap);
                    lens[p] = resolve_into(
                        self.buf.events(),
                        &mut self.add_scratch[p],
                        &mut self.sub_scratch[p],
                        &resolver,
                        psqt_weights,
                        &mut psqt_deltas[p],
                    );
                }
            }
            // Phase 2 — fused parent→child folds (opt #2): read parent lanes,
            // apply deltas, write child lanes in one pass. Paid only for
            // evaluated plies. The T6c PSQT sums fold over the SAME resolved
            // index lists, so the skip channel cannot drift from the
            // accumulator.
            //
            // Perf study 14 §3: the EMPTY-delta branch below used to be
            // `entry.acc[p] = parent.acc[p]`. `[i16; 1280]` is 2560 bytes, far
            // past any inline-store threshold, so that assignment lowered to a
            // **libc `memcpy` call** — an LD_PRELOAD interposer measured 46,903
            // of them, 2560 bytes each, over 150,059 nodes, and they are the
            // whole of the unsymbolised libc block studies 12 and 13 saw in
            // their flat profiles and could not attribute.
            //
            // TWO fixes were measured, and only one works. Deleting this branch
            // so the kernel does the copy (which with empty lists is
            // bit-identically `load 20 zmm, store 20 zmm` per 640-lane tile,
            // and is what Stockfish's `apply_combined` does —
            // `eca43a97:src/nnue/nnue_accumulator.cpp:183-244` has no
            // empty-list special case on the SIMD path at all) measured
            // **+22.7 i/n**. Keeping the branch and chunking the copy by hand
            // measured **−22.7**. Both are bit-identical to the old code and to
            // each other; the 45.4 i/n between them is spelling, not
            // semantics. Do not "simplify" this back into the kernel call
            // without re-measuring — that experiment has been run.
            //
            // **This branch is NOT dead, and it is not the null-move branch.**
            // Study 14's `entry.acc[p] = parent.acc[p]` reached it 0.313 times
            // per node, almost all of them null moves; study 15 removed the
            // null-move entry entirely (`EvalContext::null_plies`), so what
            // reaches it now is the rare REAL case — a move whose resolved
            // threat events cancel to nothing for THIS perspective, which
            // happens because the threat features are relational and a mover
            // that neither attacks nor is attacked changes none of them.
            // Measure the surviving rate before touching it; the two spellings
            // of this copy are 45.4 i/n apart (D33 reviewer C, M6/N4).
            #[cfg(feature = "foldcount")]
            super::foldcount::note_thr_joint(need, lens);
            // Perf study 48: how much of this ply's weight-row traffic is
            // provably inert (an index in BOTH lists, or twice in one)?
            #[cfg(feature = "foldcount")]
            super::foldcount::note_thr_cancel(
                [
                    &self.add_scratch[0][..lens[0].0],
                    &self.add_scratch[1][..lens[1].0],
                ],
                [
                    &self.sub_scratch[0][..lens[0].1],
                    &self.sub_scratch[1][..lens[1].1],
                ],
            );
            // The parent's VALUE HOLDER, which is the parent ply itself unless
            // the parent was a pure-copy ply the alias below elided.
            // `.min(child - 1)` for the reason `at_full`'s `.min(ply)` is
            // there: `src` indexes `before`, whose length is exactly `child`,
            // and the range fact is what keeps the fold's parent load free of
            // a bounds compare and a panic edge.
            let src = self.alias.get(child - 1);
            // ── PERF STUDY 38: THE PURE-COPY PLY IS ALIASED, NOT COPIED ──────
            //
            // A ply whose resolved threat delta is empty for BOTH perspectives
            // has, by definition, exactly its parent's accumulator pair. Rather
            // than materialise that by copying 2 x 2560 bytes, point the ply at
            // its parent's holder and write nothing.
            //
            // It is 25.2% of all folded plies and 93.5% of all pure copies
            // (`foldcount` `thr_joint`, study frame): 771,479 of 3,063,994.
            // The remaining 6.5% are plies where ONE perspective is empty and
            // the other folds or refreshes — those still copy, because the
            // entry must be materialised for the other perspective and
            // `at_full` hands out both perspectives from ONE entry.
            // Perf study 38's NULL ARM (`--features null38`): the alias
            // MACHINERY -- the extra array, its read in `at_full`, its read of
            // the parent's holder here, and its per-fold store -- with the
            // alias itself never taken, so every copy still happens.
            //
            // ⚠ PERF STUDY 40 CORRECTION: on its own it does NOT price the
            // machinery. Differencing it against a no-machinery build moves two
            // things at once, because carrying this array is what makes LLVM
            // stop calling `memcpy` at the copy below and emit 40 inline zmm
            // moves instead. Unconfounded with `noalias40` + `callcopy40`, the
            // machinery is **+4.35 i/n** and the codegen flip **+13.98**; study
            // 38's "+19.7 of machinery" was their sum, and the ~26 i/n it could
            // not explain was pricing the ELIDED copies as library calls
            // (187.9 each) when the binary they are elided from has them inline
            // (111.8 each). Study 40 §2.
            // Perf study 40's `noalias40` deletes the table itself, so the
            // branch must go with it — an alias into a stand-in that answers
            // `get(k) == k` would point the child at ITSELF.
            const ALIAS_ENABLED: bool = !cfg!(feature = "null38") && !cfg!(feature = "noalias40");
            if ALIAS_ENABLED && need[0] && need[1] && lens[0] == (0, 0) && lens[1] == (0, 0) {
                #[cfg(feature = "foldcount")]
                {
                    super::foldcount::THR_ALIAS.hit(0);
                }
                self.alias.set(child, src);
                self.fold[child] = [Fold::Done; 2];
                continue;
            }
            // Anything below WRITES `stack[child]`, so the ply becomes its own
            // holder. Both perspectives are materialised on every path that
            // gets here — one of them possibly by `discharge_refresh` above —
            // which is what makes a whole-entry alias sound.
            self.alias.set(child, child);
            for (p, &needs_fold) in need.iter().enumerate() {
                if !needs_fold {
                    continue;
                }
                let (add_len, sub_len) = lens[p];
                let (before, after) = self.stack.split_at_mut(child);
                let parent = &before[src];
                let entry = &mut after[0];
                if add_len == 0 && sub_len == 0 {
                    // ⚠ THE CHUNKING IS A HINT, NOT A GUARANTEE — CHECK IT,
                    // DO NOT TRUST THIS COMMENT.
                    //
                    // Perf study 14 wrote the loop this way and recorded that
                    // "LLVM emits 40 inline 64-byte moves instead of one
                    // 2560-byte libc `memcpy` call". Perf study 38 read the
                    // shipped binary and found the opposite: at `859a442` this
                    // site was `mov $0xa00,%edx; call *memcpy@GLIBC`, twice,
                    // because LLVM's loop-idiom pass re-forms the call. Same
                    // source text, both outcomes, in binaries built the same
                    // day — the pre-study-38 build called `memcpy`, the build
                    // with the parent alias above unrolled it inline. The
                    // decision depends on surrounding context and LLVM is free
                    // to re-take it.
                    //
                    // It matters: study 38 priced one such call by a doubled-
                    // copy probe at **187.9 instructions, 19.5 branches and
                    // 194.7 data-cache accesses**, against ~80 vector moves for
                    // the inline form.
                    //
                    // So do not assert an outcome here — MEASURE it:
                    //     objdump -d <binary> > /tmp/a.asm
                    //     python3 tools/perf38/memcpy38.py /tmp/a.asm | grep at_full
                    // A hit means the idiom re-formed and this site is paying
                    // a library call again.
                    //
                    // `chunks_exact` DROPS a remainder, so a width that is not
                    // a multiple of 32 would silently leave the tail of
                    // `entry.acc[p]` holding a stale sibling's lanes. The
                    // AVX-512 module carries the same invariant, but behind
                    // `cfg(target_arch = "x86_64")` — this one is local and
                    // holds on every target (D33 reviewer A, m1).
                    const { assert!(ACC.is_multiple_of(32)) };
                    #[cfg(feature = "foldcount")]
                    super::foldcount::THR_COPY.hit(0);
                    // Perf study 29's accumulator-stream probe: same 40 loads
                    // and 40 stores, one hot source line instead of a cold
                    // parent entry 10.5 KB up the stack.
                    #[cfg(not(feature = "probe-accpin"))]
                    let src_acc: &[i16; ACC] = &parent.acc[p];
                    #[cfg(feature = "probe-accpin")]
                    let src_acc: &[i16; ACC] = super::sota_simd::probe::pin_one();
                    #[cfg(not(feature = "probe-dstpin"))]
                    let dst_acc: &mut [i16; ACC] = &mut entry.acc[p];
                    #[cfg(feature = "probe-dstpin")]
                    let dst_acc: &mut [i16; ACC] = super::sota_simd::probe::pin_dst();
                    // Perf study 40, `--features callcopy40`: PIN the codegen to
                    // the library call. An opaque length is what stops LLVM
                    // expanding the copy inline again, and it is the whole point
                    // — study 38 compared a binary whose copy was a `memcpy`
                    // call against one whose copy was 40 inline zmm moves, and
                    // attributed the difference to machinery. With this on, both
                    // sides of that comparison call, and the machinery is the
                    // only thing left moving. Measurement only, never shipped.
                    #[cfg(feature = "callcopy40")]
                    {
                        let n = std::hint::black_box(ACC);
                        dst_acc[..n].copy_from_slice(&src_acc[..n]);
                    }
                    // PERF STUDY 40: the copy is OUTLINED, and that is the whole
                    // change. Expanded inline it is an 83-instruction
                    // straight-line zmm block sitting inside the fold loop, and
                    // moving it behind a call boundary measured **−16.01
                    // instructions and −1.21 branches per node** — while the
                    // copy itself runs 0.022 times per node, so under 2 of that
                    // can be the copy. See [`copy_acc_outlined`].
                    #[cfg(not(feature = "callcopy40"))]
                    copy_acc_outlined(dst_acc, src_acc);
                    entry.psqt[p] = parent.psqt[p];
                    self.fold[child][p] = Fold::Done;
                    continue;
                }
                apply_threat_deltas_into(
                    &mut entry.acc[p],
                    &parent.acc[p],
                    weights,
                    &self.add_scratch[p][..add_len],
                    &self.sub_scratch[p][..sub_len],
                    narrow,
                );
                // T6c: apply the resolve-time psqt delta (zero for a
                // psqt-less net, so this is `parent + 0` — the copy).
                for (dst, (&src, &d)) in entry.psqt[p]
                    .iter_mut()
                    .zip(parent.psqt[p].iter().zip(psqt_deltas[p].iter()))
                {
                    *dst = src + d;
                }
                self.fold[child][p] = Fold::Done;
            }
        }
    }

    /// Return the cursor to the parent — the inverse of the matching
    /// [`apply`](Self::apply). Cheap now: it only steps the cursor back; the
    /// parent entry (already resolved/computed) is untouched, so this is correct
    /// even when the child was pushed but never folded. Panics if only the root
    /// remains.
    pub fn undo(&mut self) {
        assert!(
            self.top > 0,
            "ThreatContext::undo underflow (only the root remains)"
        );
        self.top -= 1;
    }

    /// Depth of the incremental stack above the root (test/debug introspection).
    #[cfg(any(test, debug_assertions))]
    pub fn depth(&self) -> usize {
        self.top
    }
}

/// Rebuild ONE perspective of ONE entry from its own snapshot — the discharge
/// half of [`Fold::Refresh`], out of line.
///
/// `#[cold]` because it fires on **0.0208 of `compute`'s 0.4151 calls per node**
/// (measured; the 0.0214 total call rate of `refresh_threat_into_at` includes
/// 0.0007 of root refreshes, which never route through here).
///
/// **What it actually buys, corrected** (D33 reviewer C): NOT "the 60,720-dim
/// enumeration's locals would otherwise set `compute`'s register pressure" —
/// `refresh_threat_into_at` and `refresh_psqt_threat_at` are already
/// out-of-line symbols, so their locals were never in `compute`'s allocator.
/// What moves out of line is two four-argument call sequences and a store, and
/// with them the layout pressure they put on `compute`, which is inlined into
/// `at_full` — the single largest symbol in this block. That is consistent with
/// the measured −1.2 i/n; the register-pressure story is not. Same correction
/// class as study 17 §5.1d.
///
/// Study 17 §5.2's rule says an arm may be outlined when the caller already
/// holds every argument it needs, and it does: `weights`, `psqt_weights` and
/// `p` are already live, and the entry is the one the walk just stopped at.
///
/// Four arguments, but `&[i8]` and `&[i32]` are FAT pointers, so the call is
/// `rdi`, `rsi`+`rdx`, `rcx`+`r8`, `r9` — **exactly six** integer argument
/// registers, with zero headroom. Nothing spills today; adding a fifth argument
/// of any kind would put a stack write at the call site, which is precisely the
/// −14.2 study 17 §5.1c measured (D33 reviewer A, A10).
#[cold]
#[inline(never)]
fn discharge_refresh(
    entry: &mut ThreatEntry,
    position: &ThreatPosition,
    weights: &[i8],
    narrow: &[u64],
    psqt_weights: &[i32],
    p: usize,
    remap: Option<&ThreatRemap>,
) {
    // `position` now arrives as its own argument: perf study 31 arm p1 moved
    // the snapshot into a parallel dense array, so the two borrows are of
    // different arrays rather than disjoint fields of one struct. The seventh
    // integer argument register does not exist, so `p` (and now the Recode
    // `remap` pointer) go on the stack at this call site -- 0.0208 calls/node,
    // i.e. ~0.1 i/n, against the every-node write this split is buying (study
    // 17 5.2's six-register rule costs what it costs only at a hot call rate;
    // this one is cold, which is exactly why it carries `#[cold]`).
    let persp = Color::ALL[p];
    refresh_threat_into_at(&mut entry.acc[p], weights, narrow, position, persp, remap);
    entry.psqt[p] = refresh_psqt_threat_at(psqt_weights, position, persp, remap);
}

/// Resolve the perspective-agnostic raw `events` to one perspective's
/// FullThreats feature-index lists, written as `usize` indices into the
/// caller's `adds`/`subs` scratch and returned as `(add_len, sub_len)`. The
/// [`ThreatResolver`] maps each raw event (dropping excluded pairs) with the
/// per-fold table/orientation state hoisted out of the loop (perf study #4
/// item #3).
///
/// The replay's telescoping over-emission (a relation removed and re-added by
/// the same move) is NOT cancelled here anymore: the fold's wrapping i16
/// adds/subs cancel such ±pairs arithmetically, so keeping both sides is
/// bit-identical at the accumulator (SF folds the same way). The old
/// per-event O(n²) scan bought its extra row-folds back at a measured **0.48%
/// of events** (bench-with-net, 106.5M events → 0.51M cancels) — ~1 pair per
/// 12 perspective-folds — while its opposite-list walk taxed all 100%.
/// T6c: when `psqt_table` is non-empty, the psqt bucket-sum delta
/// (Σ add rows − Σ sub rows) is accumulated into `psqt_delta` as events
/// resolve — see the phase-1 comment in [`ThreatContext::compute`]. The
/// legacy path (empty table) keeps its original loop untouched.
/// [`resolve_into`] for BOTH perspectives in ONE pass over `events`.
///
/// The two perspectives resolve the same event list to different indices, so
/// the old shape walked the buffer twice. Everything before the perspective
/// enters is shared, though: the five event-field loads and the two `sf_code`
/// decodes are perspective-independent (the perspective is only the `^ swap`
/// colour flip inside [`ThreatResolver::index_decoded`]), so hoisting them out
/// of a fused loop pays them once instead of twice.
///
/// Bit-identical to two [`resolve_into`] calls: same arithmetic per
/// perspective, and the index lists and PSQT accumulate still see events in
/// buffer order, which is what fixes the i32 sum.
///
/// The one-sided case — a ply where one perspective is `Fold::Refresh` (its
/// king crossed the mirror, so it is rebuilt by [`discharge_refresh`] rather
/// than folded) and the other still folds — keeps calling [`resolve_into`];
/// folding a `need` test into this loop would tax the common both-sided path
/// with a branch per event.
#[allow(clippy::too_many_arguments)]
fn resolve_both(
    events: &[ThreatDelta],
    adds: &mut [[RowOffset; DELTA_CAP]; 2],
    subs: &mut [[RowOffset; DELTA_CAP]; 2],
    resolvers: &[ThreatResolver; 2],
    psqt_table: &[i32],
    psqt_deltas: &mut [[i32; BUCKETS]; 2],
) -> [(usize, usize); 2] {
    let mut len = [(0usize, 0usize); 2];

    // `& (DELTA_CAP - 1)` below is a NO-OP on the value and a range fact for
    // the optimizer, the same move as `index_decoded`'s `& 63`. `adds`/`subs`
    // are `[RowOffset; DELTA_CAP]` (perf study 57 retyped the element; the range
    // argument is untouched) and each event pushes at most ONE index into ONE
    // of them, so a cursor can only reach `events.len() <= DELTA_CAP` — and the
    // last *write* is therefore at `DELTA_CAP - 1`. LLVM cannot see that (the
    // cursor is carried across a loop whose trip count it does not relate to
    // the array length), so the store carried a `cmp $0xff` / `ja panic` pair
    // per event per perspective. DELTA_CAP is a power of two, which is what
    // makes the mask free.
    const M: usize = DELTA_CAP - 1;

    // LEGACY (psqt-less) NET — hand-unswitched, exactly as `resolve_into`
    // already was. Perf study 24 arm r1: `resolve_both` never got that
    // treatment, so `has_psqt` was a loop-invariant `&[i32]` emptiness test
    // re-evaluated per resolved index per perspective — and register pressure
    // in this loop had SPILLED it, so the asm re-read it from the stack
    // (`cmpq $0x0,0xd8(%rsp)`, four copies inside the loop body).
    if psqt_table.is_empty() {
        for d in events {
            let att = sf_code(d.attacker);
            let atd = sf_code(d.victim);
            for p in 0..2 {
                let Some(idx) = resolvers[p].index_decoded(att, atd, d.from, d.to) else {
                    continue;
                };
                debug_assert!(idx < THREAT_DIMS, "threat index {idx} out of range");
                if d.add {
                    adds[p][len[p].0 & M] = RowOffset::of(idx);
                    len[p].0 += 1;
                } else {
                    subs[p][len[p].1 & M] = RowOffset::of(idx);
                    len[p].1 += 1;
                }
            }
        }
        return len;
    }

    // Register-resident PSQT accumulators, the §4.5 pairing that only pays
    // alongside `psqt_row`'s fixed-size row.
    let mut acc = *psqt_deltas;
    // Row-major view of the PSQ1 table, hoisted out of the event loop: the
    // per-call `as_chunks` inside `psqt_add`/`psqt_sub` re-derived the row
    // count by a shift on every resolved index.
    let (rows, _tail) = psqt_table.as_chunks::<BUCKETS>();
    for d in events {
        let att = sf_code(d.attacker);
        let atd = sf_code(d.victim);
        // BOTH perspectives resolved before EITHER PSQ1 row is touched (perf
        // study 24 arm r2). Each row is a random 32-byte load out of a 2.7 MB
        // table — an L2-miss/L3-hit on essentially every event — and the
        // address depends on a THREE-DEEP dependent load chain inside
        // `index_decoded` (`lut1` → `offsets` → `lut2`). In the interleaved
        // shape the second chain did not even start until the first row's
        // `vpaddd` had retired, so the two long latencies serialised; a
        // `perf annotate` of the split build put **41% of `resolve_both`** on
        // the four instructions immediately after the `vpaddd`/`vpsubd` pairs.
        // Resolving both first makes the two loads independent and adjacent,
        // so the memory system overlaps them.
        for p in 0..2 {
            let Some(idx) = resolvers[p].index_decoded(att, atd, d.from, d.to) else {
                continue;
            };
            debug_assert!(idx < THREAT_DIMS, "threat index {idx} out of range");
            debug_assert!(
                len[p].0 < DELTA_CAP && len[p].1 < DELTA_CAP,
                "resolve cursor overflow: {len:?} events={}",
                events.len()
            );
            // PERF STUDY 29 REJECTED A SOFTWARE PREFETCH HERE. This is exactly
            // Stockfish's `append_changed_indices` site — it issues one
            // `prefetch<READ, LOW>` per index at `eca43a97:src/nnue/features/
            // full_threats.cpp:295-299` — and it is the right site by the
            // stall profile (`at_full` is 22.78% of load-stall cycles, and a
            // pinning probe showed the weight rows carry 46.2% of them). It
            // still does not measure: -0.50% misses / -0.97% cycles at n = 24
            // against nulls of +0.14% / -0.37%, and nothing survives the
            // paired within-round form. The mechanism is capped — one hint of
            // a 20-line row cannot touch the ~11 demand misses the row takes,
            // and a whole-row hint (`pf-full`, +2.0% instructions) bought
            // -0.69% misses. Numbers in that study's §5.
            if d.add {
                adds[p][len[p].0 & M] = RowOffset::of(idx);
                len[p].0 += 1;
                for (a, &w) in acc[p].iter_mut().zip(&rows[idx]) {
                    *a += w;
                }
            } else {
                subs[p][len[p].1 & M] = RowOffset::of(idx);
                len[p].1 += 1;
                for (a, &w) in acc[p].iter_mut().zip(&rows[idx]) {
                    *a -= w;
                }
            }
        }
    }
    *psqt_deltas = acc;
    len
}

fn resolve_into(
    events: &[ThreatDelta],
    // Perf study 56 mechanism B: these two were `&mut [usize]` — RUNTIME-LENGTH
    // slices — while the only production caller passes `&mut self.add_scratch[p]`,
    // which is already a fixed-size array. The signature threw the length
    // away, so all four cursor stores below carried a `cmp`/`ja panic` pair per
    // event. `resolve_both` — the fused two-perspective twin, twenty lines up,
    // doing the identical job — takes a fixed-size array pair and masks
    // with `& M` for exactly this reason (perf study 24). `resolve_into` never
    // got the same treatment, and it is the branch taken whenever only ONE
    // perspective needs folding. Same argument as there: each event pushes at
    // most one index into one buffer, so a cursor cannot pass `events.len() <=
    // DELTA_CAP`, and `DELTA_CAP` is a power of two, so the mask is free and is
    // the identity on every reachable value.
    //
    // Perf study 57 changed the ELEMENT TYPE from `usize` to [`RowOffset`] and
    // nothing else about this argument. The two mechanisms are orthogonal and
    // both are wanted: study 56's fixed size is what discharges the bounds
    // check on the WRITE, study 57's pre-scaling is what stops the fold kernel
    // recomputing `feature * ACC` once per tile on the READ.
    adds: &mut [RowOffset; DELTA_CAP],
    subs: &mut [RowOffset; DELTA_CAP],
    resolver: &ThreatResolver,
    psqt_table: &[i32],
    psqt_delta: &mut [i32; BUCKETS],
) -> (usize, usize) {
    // The mask that discharges the store's bounds check, per the argument on
    // `adds` above. `DELTA_CAP` is a power of two, so `& M` is the identity on
    // every reachable cursor value and costs one `and` that the optimizer folds
    // into the addressing mode.
    //
    // (This comment replaces a fragment left on `main` by study 56's merge:
    // it opened "OFF, `M` is all-ones" — a sentence about the `bcmisc55`
    // feature gate that the same commit deleted, so it described a `#[cfg]`
    // that no longer exists. Flagged rather than silently reflowed.)
    const M: usize = DELTA_CAP - 1;
    let mut add_len = 0usize;
    let mut sub_len = 0usize;
    if psqt_table.is_empty() {
        for d in events {
            if let Some(idx) = resolver.index(d.attacker, d.from, d.to, d.victim) {
                debug_assert!(idx < THREAT_DIMS, "threat index {idx} out of range");
                if d.add {
                    adds[add_len & M] = RowOffset::of(idx);
                    add_len += 1;
                } else {
                    subs[sub_len & M] = RowOffset::of(idx);
                    sub_len += 1;
                }
            }
        }
    } else {
        // Accumulate the T6c bucket delta in a LOCAL, written back once. The
        // pairing is load-bearing and neither half works alone (perf study 8
        // §4.5): with the row reached as a runtime-length slice the accumulate
        // is 12 instructions split across xmm / vmovq / scalar, and hoisting
        // the destination into a local then only adds copies (measured
        // +1.31%); with the row FIXED-size the body is one ymm load-add-store,
        // and only then does keeping the accumulator in registers across the
        // event loop pay. Bit-identical either way: same i32 adds, same order.
        let mut acc = *psqt_delta;
        for d in events {
            if let Some(idx) = resolver.index(d.attacker, d.from, d.to, d.victim) {
                debug_assert!(idx < THREAT_DIMS, "threat index {idx} out of range");
                if d.add {
                    adds[add_len & M] = RowOffset::of(idx);
                    add_len += 1;
                    psqt_add(&mut acc, psqt_table, idx);
                } else {
                    subs[sub_len & M] = RowOffset::of(idx);
                    sub_len += 1;
                    psqt_sub(&mut acc, psqt_table, idx);
                }
            }
        }
        *psqt_delta = acc;
    }
    (add_len, sub_len)
}

/// Reconstruct the pre-move occupancy from `snap` (post-move) + `dp`, then
/// replay the edits — every sub removed (in order), then every add — recording a
/// `record_square_touch` per edit against the state *after* that edit. See the
/// module docs for the telescoping argument.
fn build_deltas(buf: &mut DeltaBuf, tab: &AttackTables, dp: &DirtyPiece, post: &ThreatPosition) {
    // A non-capture (including a quiet promotion) changes occupancy only by
    // moving the blocker from `from` to `to`. Evaluate both endpoint changes
    // against the shared position with both endpoints empty; this is the
    // compact move primitive and avoids intermediate endpoint interactions.
    if let ([(mover, from)], [(placed, to)]) = (dp.subs(), dp.adds())
        && from != to
    {
        let mut between = *post;
        between.toggle(*placed, *to);
        record_square_touch(buf, tab, &between, *mover, *from, false);
        record_square_touch(buf, tab, &between, *placed, *to, true);
        return;
    }

    // Ordinary and promotion captures are the common occupancy-preserving
    // replacement case. Model them as mover removal plus victim→mover identity
    // mutation, avoiding the remove/add x-ray churn at the destination.
    if let ([(mover, from), (victim, victim_sq)], [(placed, to)]) = (dp.subs(), dp.adds())
        && victim_sq == to
        && from != to
    {
        // The state the mover leaves: post-move occupancy (so `from` is already
        // empty) but with the victim still standing on `to`.
        let mut after_remove = *post;
        after_remove.toggle(*placed, *to);
        after_remove.toggle(*victim, *to);
        record_square_touch(buf, tab, &after_remove, *mover, *from, false);

        record_mutate(buf, tab, post, *victim, *placed, *to);
        return;
    }

    // Generic path (castling, en passant, anything else). Undo every edit to
    // recover the pre-move position — `toggle` is XOR, so "undo" and "redo" are
    // the same primitive — then replay the edits one at a time, recording each
    // touch against the state *after its own edit*.
    //
    // Adds are undone before subs, so a square that is both (a capture
    // destination) correctly ends up holding the victim.
    let mut r = *post;
    for &(pc, sq) in dp.adds() {
        r.toggle(pc, sq);
    }
    for &(pc, sq) in dp.subs() {
        r.toggle(pc, sq);
    }

    for &(pc, sq) in dp.subs() {
        r.toggle(pc, sq);
        record_square_touch(buf, tab, &r, pc, sq, false);
    }
    for &(pc, sq) in dp.adds() {
        r.toggle(pc, sq);
        record_square_touch(buf, tab, &r, pc, sq, true);
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

    /// Distinct pseudo-random PSQ1 threat rows — same signature rationale as
    /// [`signature_weights`]: one wrong feature event perturbs a distinct
    /// 8-wide row and the bucket sums diverge.
    fn signature_psqt() -> Vec<i32> {
        (0..THREAT_DIMS * BUCKETS)
            .map(|i| {
                let mut z = (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                z ^= z >> 32;
                (z % 4001) as i32 - 2000
            })
            .collect()
    }

    fn refresh_psqt_pair(table: &[i32], board: &Board) -> [[i32; BUCKETS]; 2] {
        [
            refresh_psqt_threat(table, board, Color::White, None),
            refresh_psqt_threat(table, board, Color::Black, None),
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
        let q = signature_psqt();
        let mut b = board(START_FEN);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);

        let m = Move::normal(Square::E2, Square::E4);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);

        let (acc, psqt) = ctx.current_full();
        assert_eq!(*acc, refresh_pair(&w, &b));
        assert_eq!(*psqt, refresh_psqt_pair(&q, &b));
    }

    /// A capture: the mover's from-square relations vanish, the victim→mover
    /// swap on the destination is captured, and any x-ray through either square
    /// ripples. (1.e4 d5 2.exd5.)
    #[test]
    fn capture_matches_refresh() {
        let w = signature_weights();
        let q = signature_psqt();
        let fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        let mut b = board(fen);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);

        let m = Move::normal(Square::E4, Square::D5);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);

        let (acc, psqt) = ctx.current_full();
        assert_eq!(*acc, refresh_pair(&w, &b));
        assert_eq!(*psqt, refresh_psqt_pair(&q, &b));
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
        let q = signature_psqt();
        for &(fen, m) in cases {
            let mut b = board(fen);
            let mut ctx = ThreatContext::new(&w, &q, None);
            ctx.refresh(&b);
            let dp = DirtyPiece::from_move(&b, m);
            b.make_move(m, b.gives_check(m));
            ctx.apply(&dp, &b);
            let (acc, psqt) = ctx.current_full();
            assert_eq!(*acc, refresh_pair(&w, &b), "mismatch after {m} from {fen}");
            assert_eq!(
                *psqt,
                refresh_psqt_pair(&q, &b),
                "psqt mismatch after {m} from {fen}"
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
        let q = signature_psqt();
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);
        let m = Move::normal(Square::E4, Square::D4);
        assert!(king_crossed(&DirtyPiece::from_move(&b, m), Color::White));
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b);
        let (acc, psqt) = ctx.current_full();
        assert_eq!(*acc, refresh_pair(&w, &b));
        // The king-crossed eager rebuild must refresh the PSQT sums too.
        assert_eq!(*psqt, refresh_psqt_pair(&q, &b));
    }

    /// A king-crossed ply that is **never evaluated itself**, with ordinary
    /// plies folded on top of it — the shape perf study 18's deferred refresh
    /// created and the eager one could not produce (the eager form marked the
    /// crossing ply computed at push, so a fold could never reach it in the
    /// `Refresh` state).
    ///
    /// `compute` must stop its ancestor walk AT the crossing ply, rebuild that
    /// ply from its own stored [`ThreatPosition`], and only then fold the
    /// children forward. Walking past it, or folding it from its parent, mixes
    /// two mirror orientations of a king-oriented feature space and the
    /// accumulator diverges silently — there is no assertion inside the fold
    /// that could notice.
    #[test]
    fn deferred_king_cross_as_an_unevaluated_ancestor_matches_refresh() {
        let w = signature_weights();
        let q = signature_psqt();
        // Ke4–d4 crosses the d/e mirror for White, and the sliders (Bd5, Rc4,
        // Qc7, Rf7) ripple through both e4 and d4, so the plies stacked on top
        // of the crossing are not trivial either.
        let fen = "k7/2q2r2/8/3b4/2R1Kn2/8/8/8 w - - 0 1";
        let mut b = board(fen);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);

        let cross = Move::normal(Square::E4, Square::D4);
        let dp = DirtyPiece::from_move(&b, cross);
        assert!(
            king_crossed(&dp, Color::White),
            "fixture no longer crosses the mirror"
        );
        b.make_move(cross, b.gives_check(cross));
        // Pushed, NOT read: the crossing ply's rebuild is now an obligation.
        ctx.apply_at(0, &dp, &b);

        // Stack ordinary plies on top of it, still without ever evaluating.
        //
        // Moves that cross the mirror AGAIN for White are excluded, and that is
        // load-bearing rather than tidy: the king on d4 can step to e3/e4/e5,
        // each of which re-crosses. The first draft of this test did not
        // exclude them, the seeded RNG re-crossed at ply 3, `compute`'s walk
        // stopped THERE instead of at ply 1 — and the test still PASSED,
        // because the value is right either way. It was asserting nothing about
        // the multi-ply deferred-ancestor fold it exists for (D33 reviewer A,
        // A5; caught by the shape assertions below, which is why they are
        // here). The filter is expressed as the exact predicate `apply_at`
        // uses, so the shape holds by construction rather than by luck and the
        // test survives any change to `Rng` or to `generate_legal`'s ordering.
        let mut rng = Rng(0x0BAD_F00D_5EED);
        let mut ply = 1usize;
        for _ in 0..5 {
            let mut list = MoveList::new();
            generate_legal(&b, &mut list);
            let cands: Vec<Move> = list
                .iter()
                .copied()
                .filter(|&m| !king_cross(&DirtyPiece::from_move(&b, m))[Color::White as usize])
                .collect();
            if cands.is_empty() {
                break;
            }
            let m = cands[(rng.next() % cands.len() as u64) as usize];
            let dp = DirtyPiece::from_move(&b, m);
            b.make_move(m, b.gives_check(m));
            ctx.apply_at(ply, &dp, &b);
            ply += 1;
        }
        assert!(
            ply >= 4,
            "expected plies stacked above the cross, got {ply}"
        );

        // PIN THE SHAPE, not just the value (D33 reviewer A, A5). The stacked
        // moves are drawn from a seeded RNG, and White's king on d4 can step
        // back to e3/e4/e5 — any of which crosses the mirror AGAIN. If the last
        // such re-cross landed above ply 1, `compute`'s walk would stop there
        // instead, ply 1's obligation would never be discharged, and this test
        // would silently degrade into `king_mirror_cross_matches_refresh` while
        // still passing. Any change to `Rng`, to `generate_legal`'s ordering, or
        // to the fixture could do that; these two assertions say so out loud.
        assert_eq!(
            ctx.fold[1][Color::White as usize],
            Fold::Refresh,
            "the crossing ply must still be an undischarged obligation"
        );
        for k in 2..=ply {
            assert_ne!(
                ctx.fold[k][Color::White as usize],
                Fold::Refresh,
                "a later re-cross at ply {k} short-circuits the walk this test exists to exercise"
            );
        }

        // The FIRST read of the whole line: it must discharge the deferred
        // refresh at ply 1 and then fold plies 2..=ply on top of it.
        let (acc, psqt) = ctx.at_full(ply);
        assert_eq!(*acc, refresh_pair(&w, &b), "deferred king-cross diverged");
        assert_eq!(
            *psqt,
            refresh_psqt_pair(&q, &b),
            "deferred king-cross psqt diverged"
        );

        // AND THE MEMOIZATION, which no value oracle in this repo can see
        // (D33 reviewer A, A4). Deleting `entry.fold[p] = Fold::Done` from
        // `discharge_refresh` leaves EVERY value correct — the rebuild is
        // idempotent, and `need[p]` never re-folds the ancestor ply — so the
        // whole suite passes, the bench node count is unchanged, the eval hash
        // is unchanged, and a 6,240-instruction rebuild fires once per
        // descendant instead of once. The entire −40.4 i/n of perf study 18
        // rests on this one store. This is the only assertion that holds it.
        assert_eq!(
            ctx.fold[1][Color::White as usize],
            Fold::Done,
            "a discharged refresh must memoize, or every descendant re-pays it"
        );
    }

    /// A **non**-crossing ply, folded and memoized, then overwritten at the same
    /// ply by a sibling that DOES cross. The dangerous direction: if the
    /// overwrite left the stale `Fold::Done`, the read would hand back the first
    /// sibling's accumulator. (The other direction — a stale `Fold::Refresh`
    /// surviving onto a non-crossing sibling — is undetectable by value, since
    /// rebuilding from the overwritten snapshot happens to give the right
    /// answer; it only costs speed.)
    #[test]
    fn a_crossing_sibling_overwrites_a_folded_ply() {
        let w = signature_weights();
        let q = signature_psqt();
        let fen = "k7/2q2r2/8/3b4/2R1Kn2/8/8/8 w - - 0 1";
        let start = board(fen);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&start);

        // Sibling A: Rc4–c5, no cross — folded here, which marks ply 1 done.
        let quiet = Move::normal(Square::C4, Square::C5);
        let dp_a = DirtyPiece::from_move(&start, quiet);
        let mut b = start.clone();
        b.make_move(quiet, b.gives_check(quiet));
        ctx.apply_at(0, &dp_a, &b);
        let _ = ctx.at_full(1);

        // Sibling B at the SAME ply: Ke4–d4, which crosses.
        let cross = Move::normal(Square::E4, Square::D4);
        let mut b = start.clone();
        let dp_b = DirtyPiece::from_move(&b, cross);
        assert!(king_crossed(&dp_b, Color::White));
        b.make_move(cross, b.gives_check(cross));
        ctx.apply_at(0, &dp_b, &b);

        let (acc, psqt) = ctx.at_full(1);
        assert_eq!(*acc, refresh_pair(&w, &b), "stale sibling accumulator");
        assert_eq!(*psqt, refresh_psqt_pair(&q, &b), "stale sibling psqt");
    }

    // ---- Lazy-fold path (R4) ----

    /// Push a run of plies WITHOUT ever reading the accumulator, so nothing folds
    /// until the final `at`, then assert the deep lazy fold equals a from-scratch
    /// refresh. Exercises the deferred multi-ply fold from the root that a search
    /// with TT cutoffs (no evaluate at the intermediate plies) produces.
    #[test]
    fn lazy_fold_of_unevaluated_plies_matches_refresh() {
        let w = signature_weights();
        let q = signature_psqt();
        let mut b = board(START_FEN);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);

        let mut rng = Rng(0x1A2F_F01D_C0DE);
        let mut ply = 0usize;
        for _ in 0..24 {
            let mut list = MoveList::new();
            generate_legal(&b, &mut list);
            if list.is_empty() {
                break;
            }
            let m = list[(rng.next() % list.len() as u64) as usize];
            let dp = DirtyPiece::from_move(&b, m);
            b.make_move(m, b.gives_check(m));
            // Push the child at `ply + 1` but never fold it (no `at`/`current`).
            ctx.apply_at(ply, &dp, &b);
            ply += 1;
        }
        assert!(ply > 4, "expected a non-trivial unevaluated run, got {ply}");
        // First fold happens here, spanning every deferred ply from the root.
        let (acc, psqt) = ctx.at_full(ply);
        assert_eq!(*acc, refresh_pair(&w, &b));
        assert_eq!(*psqt, refresh_psqt_pair(&q, &b));
    }

    /// Overwrite the same child ply with two different moves (the sibling-reuse
    /// pattern the search relies on) — first after the child was folded, then
    /// again while it is still unfolded — and assert each `at` reflects the LAST
    /// move applied. Catches any failure to reset `computed` / the delta lists on
    /// overwrite.
    #[test]
    fn sibling_overwrite_at_same_ply_matches_refresh() {
        let w = signature_weights();
        let q = signature_psqt();
        let start = board(START_FEN);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&start);

        // Sibling A: 1.e4, folded (marks the ply-1 entry computed).
        let mut ba = start.clone();
        let ma = Move::normal(Square::E2, Square::E4);
        let dpa = DirtyPiece::from_move(&ba, ma);
        ba.make_move(ma, ba.gives_check(ma));
        ctx.apply_at(0, &dpa, &ba);
        assert_eq!(*ctx.at(1), refresh_pair(&w, &ba));

        // Sibling B: 1.d4 at the SAME ply overwrites the (computed) entry.
        let mut bb = start.clone();
        let mb = Move::normal(Square::D2, Square::D4);
        let dpb = DirtyPiece::from_move(&bb, mb);
        bb.make_move(mb, bb.gives_check(mb));
        ctx.apply_at(0, &dpb, &bb);
        assert_eq!(
            *ctx.at(1),
            refresh_pair(&w, &bb),
            "overwrite after a folded sibling must reflect move B"
        );

        // Sibling C: 1.Nf3 overwrites again while B is already folded above —
        // then a fourth, D, WITHOUT folding C first (pure unfolded overwrite).
        let mut bc = start.clone();
        let mc = Move::normal(Square::G1, Square::F3);
        let dpc = DirtyPiece::from_move(&bc, mc);
        bc.make_move(mc, bc.gives_check(mc));
        ctx.apply_at(0, &dpc, &bc);

        let mut bd = start.clone();
        let md = Move::normal(Square::B1, Square::C3);
        let dpd = DirtyPiece::from_move(&bd, md);
        bd.make_move(md, bd.gives_check(md));
        ctx.apply_at(0, &dpd, &bd);
        let (acc, psqt) = ctx.at_full(1);
        assert_eq!(
            *acc,
            refresh_pair(&w, &bd),
            "overwrite of an unfolded sibling must reflect move D"
        );
        assert_eq!(
            *psqt,
            refresh_psqt_pair(&q, &bd),
            "overwrite of an unfolded sibling must reflect move D's psqt"
        );
    }

    /// Undo after a push that was never folded must restore the (untouched)
    /// parent — the lazy analogue of the eager undo-identity check. No `at`/
    /// `current` is called on the child before the undo.
    #[test]
    fn undo_after_unevaluated_push_restores_parent() {
        let w = signature_weights();
        let q = signature_psqt();
        let mut b = board(START_FEN);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);
        let root = *ctx.current();

        let m = Move::normal(Square::E2, Square::E4);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, b.gives_check(m));
        ctx.apply(&dp, &b); // push at the cursor, but never fold the child
        assert_eq!(ctx.depth(), 1, "the cursor advanced on push");

        ctx.undo();
        b.unmake_move(m);
        assert_eq!(ctx.depth(), 0, "the cursor returned to the root");
        assert_eq!(
            *ctx.current(),
            root,
            "undo of an unevaluated push must leave the root intact"
        );
    }

    /// The alias walk perf study 15 arm 2 deleted from [`ThreatContext::at_full`]
    /// existed to hand back the parent for an EMPTY dirty record. The search no
    /// longer produces one, but the CURSOR API still can, and the deletion rests
    /// on a specific claim: such an entry is now MATERIALISED by the fold to the
    /// same value the alias returned. That claim had nothing behind it (D33
    /// reviewer C, T4); this is the something.
    ///
    /// Mutations it catches: `build_deltas` emitting an event for an empty
    /// record; the empty-list branch copying the wrong slot; reinstating an
    /// alias walk that returns an ANCESTOR rather than this entry.
    #[test]
    fn an_empty_record_materialises_to_the_parent() {
        let w = signature_weights();
        let q = signature_psqt();
        let b = board(START_FEN);
        let mut ctx = ThreatContext::new(&w, &q, None);
        ctx.refresh(&b);
        let (root_acc, root_psqt) = {
            let (a, p) = ctx.current_full();
            (*a, *p)
        };

        // A null move changes no placement, so the post-move board IS this
        // board — exactly the shape of the search's old null-move push.
        ctx.apply(&DirtyPiece::EMPTY, &b);
        assert_eq!(ctx.depth(), 1, "the cursor advanced on an empty push");

        let (child_acc, child_psqt) = {
            let (a, p) = ctx.current_full();
            (*a, *p)
        };
        assert_eq!(
            child_acc, root_acc,
            "empty-record fold must equal the parent"
        );
        assert_eq!(child_psqt, root_psqt, "and so must its PSQT bucket sums");

        // ...and it agrees with a from-scratch refresh, which is what makes this
        // a correctness statement rather than a self-consistency one.
        for (i, &persp) in Color::ALL.iter().enumerate() {
            assert_eq!(
                child_acc[i],
                refresh_threat(&w, &b, persp),
                "empty-record fold diverged from a from-scratch refresh"
            );
        }
    }

    // Play `games` random legal games of up to `max_plies`, asserting at every
    // node that the incrementally-maintained accumulator AND its T6c PSQT
    // bucket sums equal a from-scratch refresh for both perspectives, and that
    // apply→undo restores the parent exactly (make→apply→unmake→undo ==
    // identity).
    fn replay(weights: &[i8], psqt: &[i32], games: usize, max_plies: usize, seed: u64) -> usize {
        let mut rng = Rng(seed);
        let mut nodes = 0usize;
        for _ in 0..games {
            let mut b = board(START_FEN);
            let mut ctx = ThreatContext::new(weights, psqt, None);
            ctx.refresh(&b);

            for _ in 0..max_plies {
                let mut list = MoveList::new();
                generate_legal(&b, &mut list);
                if list.is_empty() {
                    break;
                }
                let m = list[(rng.next() % list.len() as u64) as usize];

                // Snapshot the pre-move accumulator so undo can be checked.
                let (before_acc, before_psqt) = ctx.current_full();
                let (before_acc, before_psqt) = (*before_acc, *before_psqt);

                let dp = DirtyPiece::from_move(&b, m);
                let gc = b.gives_check(m);
                b.make_move(m, gc);
                ctx.apply(&dp, &b);
                nodes += 1;

                // (a) incremental == from-scratch refresh at this node.
                let (acc, got_psqt) = ctx.current_full();
                assert_eq!(
                    *acc,
                    refresh_pair(weights, &b),
                    "incremental != refresh after {m}\nfen (post): {}",
                    b.to_fen(),
                );
                assert_eq!(
                    *got_psqt,
                    refresh_psqt_pair(psqt, &b),
                    "psqt incremental != refresh after {m}\nfen (post): {}",
                    b.to_fen(),
                );

                // (b) unmake the board and undo the accumulator: identity.
                b.unmake_move(m);
                ctx.undo();
                let (acc, got_psqt) = ctx.current_full();
                assert_eq!(*acc, before_acc, "apply→undo not identity for {m}");
                assert_eq!(
                    *got_psqt, before_psqt,
                    "psqt apply→undo not identity for {m}"
                );

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
        let q = signature_psqt();
        let nodes = replay(&w, &q, 60, 60, 0x00C0_FFEE_1234);
        assert!(
            nodes > 1500,
            "expected a substantial corpus, got {nodes} nodes"
        );
    }

    /// TEMPORARY MEASUREMENT (perf-impl-1): how many weight rows does the fold
    /// touch per perspective-ply, and how does that compare with the *minimal*
    /// delta — the symmetric difference of the from-scratch active feature sets
    /// before and after the move? Anything above the minimum is a row we fold
    /// and then cancel arithmetically.
    #[test]
    #[ignore = "measurement"]
    fn measure_fold_row_counts() {
        use std::collections::BTreeSet;
        let q: Vec<i32> = Vec::new();
        let tab = tables();
        let base_seed = 0x1234_5678_9ABC_DEF0u64;
        let mut tot_rows = 0usize;
        let mut tot_min = 0usize;
        let mut tot_raw = 0usize;
        let mut n = 0usize;
        let mut hist = [0usize; 80];
        let mut minhist = [0usize; 80];
        let mut buf = DeltaBuf::new();
        let mut adds = [RowOffset::default(); DELTA_CAP];
        let mut subs = [RowOffset::default(); DELTA_CAP];
        let mut pd;
        for game in 0..400u32 {
            let mut board = Board::startpos();
            let mut rng = Rng(base_seed
                .wrapping_add(u64::from(game))
                .wrapping_mul(0x2545_F491_4F6C_DD1D));
            for _ in 0..100 {
                let mut moves = MoveList::new();
                generate_legal(&board, &mut moves);
                if moves.is_empty() {
                    break;
                }
                let m = moves[(rng.next() % moves.len() as u64) as usize];
                let dp = DirtyPiece::from_move(&board, m);
                // active sets BEFORE
                let mut before = [BTreeSet::new(), BTreeSet::new()];
                for (i, &p) in Color::ALL.iter().enumerate() {
                    for_each_threat(&board, p, |idx| {
                        before[i].insert(idx);
                    });
                }
                let gc = board.gives_check(m);
                board.make_move(m, gc);
                let post = ThreatPosition::from_board(&board);
                buf.clear();
                build_deltas(&mut buf, tab, &dp, &post);
                tot_raw += buf.len;
                for (i, &p) in Color::ALL.iter().enumerate() {
                    if king_crossed(&dp, p) {
                        continue;
                    }
                    let mut after = BTreeSet::new();
                    for_each_threat(&board, p, |idx| {
                        after.insert(idx);
                    });
                    let minimal = before[i].symmetric_difference(&after).count();
                    let resolver = ThreatResolver::new(p, post.kings[i]);
                    pd = [0i32; BUCKETS];
                    let (al, sl) =
                        resolve_into(buf.events(), &mut adds, &mut subs, &resolver, &q, &mut pd);
                    tot_rows += al + sl;
                    tot_min += minimal;
                    n += 1;
                    hist[(al + sl).min(79)] += 1;
                    minhist[minimal.min(79)] += 1;
                }
            }
        }
        eprintln!("perspective-plies      : {n}");
        eprintln!(
            "raw events / ply       : {:.2}",
            tot_raw as f64 / (n as f64 / 2.0)
        );
        eprintln!("resolved rows / p-ply  : {:.2}", tot_rows as f64 / n as f64);
        eprintln!("MINIMAL rows / p-ply   : {:.2}", tot_min as f64 / n as f64);
        eprintln!(
            "overfold factor        : {:.3}",
            tot_rows as f64 / tot_min as f64
        );
        eprint!("rows hist  :");
        for (i, &c) in hist.iter().enumerate() {
            if c > 0 {
                eprint!(" {i}:{c}");
            }
        }
        eprintln!();
        eprint!("minrows hist:");
        for (i, &c) in minhist.iter().enumerate() {
            if c > 0 {
                eprint!(" {i}:{c}");
            }
        }
        eprintln!();
    }

    /// The large-corpus stress gate — thousands of games. Slow in debug (the
    /// board's per-node `assert_consistent` oracle dominates); run in release:
    /// `cargo test --release --features sota -- --ignored threats_replay_large`.
    #[test]
    #[ignore = "large corpus; run in release"]
    fn threats_replay_large() {
        let w = signature_weights();
        let q = signature_psqt();
        let nodes = replay(&w, &q, 4000, 120, 0x5EED_D00D_F00D);
        assert!(
            nodes > 100_000,
            "expected a large corpus, got {nodes} nodes"
        );
    }
}
