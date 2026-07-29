//! Incremental HalfKAv2_hm accumulator with a per-king-state Finny cache.
//!
//! T6c FT PSQT skip: alongside every `[i16; ACC]` accumulator lane set, this
//! module maintains the per-perspective `[i32; BUCKETS]` PSQT bucket sums —
//! the same add/sub feature events applied to the net's 8-wide PSQ1 rows
//! (`sota_loader` module docs). The Finny cache stores/restores the sums with
//! the same keys as the accumulator, so a cached refresh cannot desync the
//! skip channel. A net without a PSQ1 table (`psqt_weights` empty) skips
//! every psqt update and the sums stay zero — bit-identical to pre-T6c.

use super::sota_features::{ACC, halfka_index, halfka_king_key, halfka_oriented, halfka_row_base};
use super::sota_head::BUCKETS;
use super::sota_simd::{apply_i16_deltas, apply_i16_deltas_dual, apply_i16_deltas_into};
use crate::bitboard::Bitboard;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, PieceType, Square};

const STACK_LEN: usize = MAX_PLY + 8;
const FINNY_KEYS: usize = 64;

/// Feature `feature`'s PSQ1 row as a FIXED-size array reference. `table` is
/// row-major (`feature · BUCKETS + bucket`) and must be non-empty.
///
/// The fixed size is the whole point, and it is worth a paragraph because the
/// slice form looked identical and was not. `BUCKETS` i32 is exactly 32 bytes —
/// one `ymm` — but handing the accumulate a `&[i32]` of merely *runtime* length
/// left LLVM's SLP vectoriser unable to see a trip count of 8, so it found a
/// 4-lane group, then a 2-lane group, then gave up:
///
/// ```text
///     vmovdqu (%rsi),%xmm0            ; lanes 0..3
///     vpaddd  (%r8,%rdx,4),%xmm0,%xmm0
///     vmovdqu %xmm0,(%rsi)
///     vmovq   0x10(%r8,%rdx,4),%xmm0  ; lanes 4..5
///     vmovq   0x10(%rsi),%xmm1
///     vpaddd  %xmm0,%xmm1,%xmm0
///     vmovq   %xmm0,0x10(%rsi)
///     mov     0x18(%r8,%rdx,4),%ecx   ; lane 6, SCALAR
///     add     %ecx,0x18(%rsi)
///     mov     0x1c(%rsi),%ecx         ; lane 7, SCALAR
///     add     0x1c(%r8,%rdx,4),%ecx
///     mov     %ecx,0x1c(%rsi)
/// ```
///
/// Twelve instructions for a load / add / store. `psqt_add` + `psqt_sub` are
/// ~3.0% of ALL engine instructions (perf study 8 §4.5) because they run once
/// per resolved feature in *both* the threat fold and the HalfKA fold.
/// # Safety-free form
/// `BUCKETS` i32 is exactly 32 bytes — one `ymm` — but a `&[i32]` of merely
/// RUNTIME length left LLVM's SLP vectoriser unable to see a trip count of 8,
/// so it emitted a 4-lane group, a 2-lane `vmovq` group, and two SCALAR lanes:
/// twelve instructions for a load / add / store. A fixed-size array reference
/// fixes the shape. (A raw-pointer form removes the length check too, but
/// measured +2.2% — it costs LLVM the aliasing facts the slice index gives it.)
///
/// # One bounds check, not three
/// The `table[f*BUCKETS..][..BUCKETS].try_into()` spelling reaches the same
/// `&[i32; BUCKETS]` but pays for it three times: the open range checks
/// `f*BUCKETS <= len`, the closed range checks `BUCKETS <= len - f*BUCKETS`,
/// and `try_into` checks the length again (it folds, but the first two do
/// not). In the threat resolver that is a `sub`/`jb`/`cmp $7`/`jbe` quartet
/// per resolved event per perspective — measured in perf study 10 §2.1 as
/// two of the five panic edges in that ~40-instruction loop.
///
/// `as_chunks` states the shape once: the table IS a slice of BUCKETS-wide
/// rows (`THREAT_DIMS · BUCKETS` / `HALFKA_DIMS · BUCKETS` by construction),
/// so the row index is a single `cmp`/`jae` against a length LLVM derives by
/// a shift. Same row, same arithmetic, bit-identical output.
#[inline]
fn psqt_row(table: &[i32], feature: usize) -> &[i32; BUCKETS] {
    let (rows, _tail) = table.as_chunks::<BUCKETS>();
    &rows[feature]
}

/// Add feature `feature`'s 8-wide PSQ1 row into `psqt`.
#[inline]
pub(crate) fn psqt_add(psqt: &mut [i32; BUCKETS], table: &[i32], feature: usize) {
    for (a, &w) in psqt.iter_mut().zip(psqt_row(table, feature)) {
        *a += w;
    }
}

/// Subtract feature `feature`'s PSQ1 row from `psqt`.
#[inline]
pub(crate) fn psqt_sub(psqt: &mut [i32; BUCKETS], table: &[i32], feature: usize) {
    for (a, &w) in psqt.iter_mut().zip(psqt_row(table, feature)) {
        *a -= w;
    }
}

/// Fused parent→child PSQT fold: `dst = src ± rows`, mirroring the
/// accumulators' `apply_*_deltas_into` shape. No-op guard belongs to callers
/// (`table` must be non-empty here).
#[inline]
pub(crate) fn psqt_fold(
    dst: &mut [i32; BUCKETS],
    src: &[i32; BUCKETS],
    table: &[i32],
    adds: &[usize],
    subs: &[usize],
) {
    // Accumulate in a LOCAL and store once. Paired with `psqt_row`'s fixed-size
    // row this keeps the eight buckets in one ymm register across the whole
    // fold instead of round-tripping them to the parent entry on every feature
    // (perf study 8 §4.5 — the pairing is what pays; neither half does alone).
    let mut acc = *src;
    for &f in adds {
        psqt_add(&mut acc, table, f);
    }
    for &f in subs {
        psqt_sub(&mut acc, table, f);
    }
    *dst = acc;
}

/// From-scratch HalfKA PSQT bucket sums for `perspective` — the test/oracle
/// reference the incremental path must match. Zero when `table` is empty.
/// Consumed only by tests and the debug accumulator oracle, so release builds
/// compile it out (unlike the threat twin, which the eager refresh paths use).
#[cfg(any(test, debug_assertions))]
pub(crate) fn refresh_psqt_halfka(
    table: &[i32],
    board: &Board,
    perspective: Color,
) -> [i32; BUCKETS] {
    let mut out = [0i32; BUCKETS];
    if table.is_empty() {
        return out;
    }
    super::sota_features::for_each_halfka(board, perspective, |idx| psqt_add(&mut out, table, idx));
    out
}

// Cache-line aligned (perf study 3 §5.3): under arch-d47v this struct is
// 3152 B / align 4 — 3152 mod 64 = 16, so every ply's accumulator started
// 16 B further out of phase and 3 plies in 4 split cache lines in the fold
// kernels. One attribute, identical instruction stream, −1.29% L1d loads.
#[derive(Clone)]
#[repr(align(64))]
struct Entry {
    acc: [[i16; ACC]; 2],
    psqt: [[i32; BUCKETS]; 2],
    dirty: DirtyPiece,
    computed: [bool; 2],
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            acc: [[0; ACC]; 2],
            psqt: [[0; BUCKETS]; 2],
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}

#[derive(Clone)]
#[repr(align(64))]
struct FinnyEntry {
    acc: [i16; ACC],
    psqt: [i32; BUCKETS],
    // Perf study #2 option A: the cached POSITION as per-(color,type)
    // bitboards — the refresh diff is 12 XOR/AND-NOTs + index computation
    // for CHANGED squares only, replacing two binary-search passes over
    // sorted feature lists (was ~13% of all cycles; SF's shape).
    by_piece: [[Bitboard; 6]; 2],
}

pub(crate) struct HalfkaContext<'a> {
    weights: &'a [i16],
    bias: &'a [i16],
    /// T6c PSQ1 rows for the HalfKA feature space (`HALFKA_DIMS · BUCKETS`
    /// i32, row-major), or empty ⇒ the skip channel is inert.
    psqt_weights: &'a [i32],
    stack: Box<[Entry]>,
    cache: Box<[Option<FinnyEntry>]>,
}

impl<'a> HalfkaContext<'a> {
    pub(crate) fn new(weights: &'a [i16], bias: &'a [i16], psqt_weights: &'a [i32]) -> Self {
        Self {
            weights,
            bias,
            psqt_weights,
            stack: vec![Entry::default(); STACK_LEN].into_boxed_slice(),
            cache: vec![None; 2 * FINNY_KEYS].into_boxed_slice(),
        }
    }

    pub(crate) fn refresh_root(&mut self, board: &Board) {
        for perspective in Color::ALL {
            self.finny_refresh_into(board, perspective, 0);
        }
        self.stack[0].dirty = DirtyPiece::EMPTY;
        self.stack[0].computed = [true; 2];
    }

    pub(crate) fn push(&mut self, ply: usize, dirty: DirtyPiece) {
        self.stack[ply + 1].dirty = dirty;
        self.stack[ply + 1].computed = [false; 2];
    }

    /// The current accumulator pair plus the per-perspective T6c PSQT bucket
    /// sums of the SAME entry (all-zero for a net without a PSQ1 table).
    pub(crate) fn current_full(
        &mut self,
        board: &Board,
        ply: usize,
    ) -> (&[[i16; ACC]; 2], &[[i32; BUCKETS]; 2]) {
        // There USED to be a null-move alias walk here (perf study #4 item #3,
        // mirrored from the threat stack): an EMPTY dirty record's "fold" is a
        // pure 2 x 2.5 KB parent copy, so the walk handed back the parent
        // instead. Perf study 15 arm 1 removed the entry itself — the search
        // pushes nothing for a null move (`EvalContext::null_plies`), which is
        // Stockfish's shape (`do_null_move` at `eca43a97:src/search.cpp:657-662`
        // pushes no accumulator entry, `undo_null_move` at `:669` pops none;
        // SF's stack is a cursor, `nnue_accumulator.h:135`) — so no EMPTY record
        // ever reaches this stack from a search and the walk could never move.
        // SF has no aliasing walk of its own: the only walk in its equivalent
        // region, `find_last_usable_accumulator`
        // (`eca43a97:src/nnue/nnue_accumulator.cpp:105-119`), is our `compute`
        // ancestor walk, not this one. Deleting it is convergence.
        for perspective in Color::ALL {
            self.compute(board, ply, perspective);
        }
        let entry = &self.stack[ply];
        (&entry.acc, &entry.psqt)
    }

    fn compute(&mut self, board: &Board, ply: usize, perspective: Color) {
        let p = perspective as usize;
        if self.stack[ply].computed[p] {
            return;
        }
        let mut ancestor = ply;
        while ancestor > 0 && !self.stack[ancestor].computed[p] {
            ancestor -= 1;
        }
        let king = Piece::new(perspective, crate::types::PieceType::King);
        // The LAST king move in the span, if any. Every ply from there to `ply`
        // shares the post-move king square, so all of them are reachable from
        // the refreshed leaf by plain reverse folds.
        let last_king_move = ((ancestor + 1)..=ply).rev().find(|&child| {
            self.stack[child]
                .dirty
                .subs()
                .iter()
                .any(|&(piece, _)| piece == king)
        });
        if let Some(king_ply) = last_king_move {
            self.finny_refresh_into(board, perspective, ply);
            self.stack[ply].computed[p] = true;
            // Perf study 12 §6 — BACKWARD FILL, the thing Stockfish does and we
            // did not: `AccumulatorStack::evaluate_side` /
            // `backward_update_incremental`,
            // `eca43a97:src/nnue/nnue_accumulator.cpp:86-101, 138-154`.
            // Refreshing only the
            // leaf leaves every ply between the king move and this node
            // uncomputed, so the NEXT sibling below any of them walks its
            // ancestor search straight past the king move and refreshes from
            // scratch again.
            //
            // Measured: 0.3167 finny refreshes per node against `eca43a97`'s
            // 0.1673 — 1.89x — on a feature set whose refresh TRIGGER is
            // identical (`HalfKAv2_hm::requires_refresh` is "this perspective's
            // king moved", exactly ours). After this change: **0.1611
            // refreshes and 0.9064 refresh rows per node, against Stockfish's
            // 0.1673 and 0.9268** — parity. (The 3.7% is measured on two
            // different search trees, so parity is the claim, not an edge.)
            //
            // Reconstructing the ancestors from the leaf costs one reverse fold
            // each (measured 0.0532 per node) and makes them computed, so the
            // siblings take the incremental path. Bit-identical: a reverse fold
            // subtracts exactly the rows the forward fold added, and wrapping
            // i16 addition is exactly invertible.
            let weights = self.weights;
            let psqt_weights = self.psqt_weights;
            let king_square = board.king_square(perspective);
            for child in (king_ply..ply).rev() {
                // `child` is reconstructed from `child + 1` by undoing the move
                // that LED to `child + 1`.
                let dirty = self.stack[child + 1].dirty;
                // The whole loop bucket-indexes on ONE king square, which is
                // sound only because `king_ply` is the LAST king move in the
                // span. Pin it: the natural next optimisation — refreshing on a
                // king BUCKET change rather than any king move — would silently
                // break this and compute wrong HalfKA indices.
                debug_assert!(
                    !dirty.subs().iter().any(|&(piece, _)| piece == king),
                    "backward span must not contain a king move"
                );
                let (before, after) = self.stack.split_at_mut(child + 1);
                let entry = &mut before[child];
                let source = &after[0];
                // The backward loop's own bound, which the forward loop's
                // argument below does NOT cover (D33 reviewer B): this reads
                // `stack[child + 1]`, and `child + 1 >= king_ply + 1 >=
                // ancestor + 2 >= 2`, so `stack[0]`'s permanently-empty record
                // is out of reach here for a different reason than there.
                debug_assert!(
                    !(dirty.adds().is_empty() && dirty.subs().is_empty()),
                    "empty HalfKA dirty record on the backward fold path"
                );
                apply_dirty_directed::<false>(
                    weights,
                    psqt_weights,
                    entry,
                    source,
                    &dirty,
                    perspective,
                    king_square,
                );
                entry.computed[p] = true;
            }
            return;
        }
        let king_square = board.king_square(perspective);
        let weights = self.weights;
        let psqt_weights = self.psqt_weights;
        for child in (ancestor + 1)..=ply {
            let dirty = self.stack[child].dirty;
            // Fused parent→child fold: read parent lanes, apply deltas, write
            // child lanes in one pass — no separate ~1.5 KB memmove (opt #2).
            //
            // There is NO empty-record special case any more (perf study 16
            // §4, from study 15 §8 item 10). For HalfKA a dirty record is
            // empty only for a null move — every real move emits at least one
            // add and one sub, which is pinned by `DirtyPiece::from_move`'s
            // construction (`board.rs`, every `m.is_ok()` arm) and by
            // `dirty_piece_covers_every_move_kind` in `accumulator.rs`, NOT by
            // this module's own tests (D33 reviewers B and C) — and since
            // study 15 arm 1 a null move pushes no
            // entry at all. `stack[0]`'s record is never read: this loop
            // starts at `ancestor + 1 >= 1`. So the branch was taken 0x per
            // node, and its `else` arm is unchanged — which is why this
            // cannot re-materialise the `memcpy` study 14 §3.1 fought. That
            // arm deleted a special case on a path TAKEN 31% of the time.
            //
            // WHAT THE TRIPWIRE PROTECTS IS THE PERF CLAIM, NOT CORRECTNESS
            // (D33 reviewers A and C, independently). The deleted branch was
            // bit-identical to its `else` arm even when taken:
            // `apply_i16_deltas_into(dst, src, w, &[], &[])` is `dst = src`
            // lane for lane, and `psqt_fold` with empty lists is `*dst = *src`.
            // So this deletion cannot be WRONG, only slower.
            //
            // The corollary is a real future hazard, and it is the reason this
            // paragraph exists: **the empty-list case of the delta kernels
            // must remain a copy.** Adding an `if adds.is_empty() &&
            // subs.is_empty() { return; }` fast path to `apply_i16_deltas_into`
            // looks like a free win and would, at that moment, turn this into
            // a stale-child bug — in release, silently.
            let (before, after) = self.stack.split_at_mut(child);
            let parent = &before[child - 1];
            let entry = &mut after[0];
            debug_assert!(
                !(dirty.adds().is_empty() && dirty.subs().is_empty()),
                "empty HalfKA dirty record on the forward fold path"
            );
            apply_dirty_into(
                weights,
                psqt_weights,
                entry,
                parent,
                &dirty,
                perspective,
                king_square,
            );
            entry.computed[p] = true;
        }
    }

    /// Finny-cached from-scratch refresh, written **in place** (perf study #4
    /// item #5): the cache entry's accumulator is updated where it lives and
    /// then copied ONCE into `stack[ply].acc[perspective]` — the old by-value
    /// form round-tripped the ~2.5 KB accumulator through a local plus a fresh
    /// `FinnyEntry` (~4 copies per refresh). Same deltas, same kernel, same
    /// arithmetic ⇒ bit-identical cache and stack contents.
    fn finny_refresh_into(&mut self, board: &Board, perspective: Color, ply: usize) {
        let king = board.king_square(perspective);
        let key = halfka_king_key(perspective, king);
        let weights = self.weights;
        let psqt_weights = self.psqt_weights;
        // Split the borrow so the fold can write the cache entry AND the stack
        // slot in one pass (perf study 9 item 4). `cache` and `stack` are
        // distinct fields, so destructuring is what makes two `&mut` legal
        // here — the idiomatic Rust answer to "I need two disjoint pieces of
        // self at once", and no `unsafe`.
        let Self {
            cache, stack, bias, ..
        } = self;
        let slot = &mut cache[perspective as usize * FINNY_KEYS + key];
        if let Some(cached) = slot.as_mut() {
            // Option A (perf study #2): bitboard set-diff against the cached
            // position. Same add/sub SETS as the old sorted-list derivation
            // (order may differ; wrapping i16/i32 adds commute => the
            // accumulator and PSQT sums stay bit-identical).
            // The diff walk is 6.8x Stockfish's on its SCALAR half (perf study
            // 10 §2.2: 184 vs 27 instructions/node, against a fold that is only
            // 1.8x on the accumulator traffic), so it is written for codegen,
            // not for prose. Three things it must NOT do:
            //
            //  * iterate `PieceType::ALL` — LLVM materialises that 6-byte array
            //    on the stack every call and reloads the piece type from it per
            //    iteration (`movw $0x504,0xbc(%rsp)` / `movzbl 0xb8(%rsp,%r13,1)`).
            //    Zipping the two `[Bitboard; 6]` ROWS gives a compile-time trip
            //    count and no memory for the iteration variable.
            //  * recompute the row base per direction — hoisted (see
            //    `halfka_row_base`).
            //  * bounds-check the cursor writes — at most one index is pushed
            //    per occupied square, and there are at most 32 pieces, so a
            //    cursor cannot exceed `CAP`; `& (CAP - 1)` is free because CAP
            //    is a power of two, and the `debug_assert` carries the real
            //    invariant. (Same move as the threat resolver's, §2.1.)
            const CAP: usize = 32;
            let mut adds = [0usize; CAP];
            let mut add_len = 0;
            let mut subs = [0usize; CAP];
            let mut sub_len = 0;
            for color in [Color::White, Color::Black] {
                for pt in PieceType::ALL {
                    let new_bb = board.pieces(color, pt);
                    let old_bb = cached.by_piece[color as usize][pt as usize];
                    if new_bb == old_bb {
                        continue;
                    }
                    let piece = Piece::new(color, pt);
                    let base = halfka_row_base(perspective, piece, king);
                    for sq in new_bb & !old_bb {
                        debug_assert!(add_len < CAP, "finny add cursor overflow");
                        adds[add_len & (CAP - 1)] = base + halfka_oriented(perspective, sq, king);
                        add_len += 1;
                    }
                    for sq in old_bb & !new_bb {
                        debug_assert!(sub_len < CAP, "finny sub cursor overflow");
                        subs[sub_len & (CAP - 1)] = base + halfka_oriented(perspective, sq, king);
                        sub_len += 1;
                    }
                    cached.by_piece[color as usize][pt as usize] = new_bb;
                }
            }
            // Two-destination fold (perf study 9 item 4): the cache entry is
            // its own memo for the next refresh with this king key, and the
            // stack slot is what the head reads — the same 2560 bytes in two
            // places. Folding in place and then assigning the stack slot spent
            // a full `memmove` (measured at **102.9 instructions/node**, i.e.
            // 1.45% of ALL engine instructions, by a black_box'd doubled-copy
            // probe; the naive doubled assignment measured +0.006% because
            // dead-store elimination removed it). Writing both homes from the
            // registers the fold already holds costs one store per 32 lanes.
            apply_i16_deltas_dual(
                &mut cached.acc,
                &mut stack[ply].acc[perspective as usize],
                weights,
                &adds[..add_len],
                &subs[..sub_len],
            );
            // T6c: the cache's PSQT sums see the SAME add/sub set, so a cached
            // refresh restores the skip channel exactly (the corruption trap
            // the deploy contract calls out).
            if !psqt_weights.is_empty() {
                // Same register-resident accumulate as `psqt_fold`.
                let mut acc = cached.psqt;
                for &f in &adds[..add_len] {
                    psqt_add(&mut acc, psqt_weights, f);
                }
                for &f in &subs[..sub_len] {
                    psqt_sub(&mut acc, psqt_weights, f);
                }
                cached.psqt = acc;
            }
            // The accumulator is already in the stack slot (the dual fold put
            // it there); only the 32-byte PSQT pair is left to publish.
            stack[ply].psqt[perspective as usize] = cached.psqt;
            return;
        } else {
            let mut acc = [0i16; ACC];
            acc.copy_from_slice(&bias[..ACC]);
            let mut adds = [0usize; 32];
            let mut len = 0usize;
            let mut by_piece = [[Bitboard::EMPTY; 6]; 2];
            for square in board.all_occupancy() {
                let piece = board.piece_on(square).expect("occupied square");
                adds[len] = halfka_index(perspective, piece, square, king);
                len += 1;
                by_piece[piece.color as usize][piece.piece_type as usize] |=
                    Bitboard::from_square(square);
            }
            apply_i16_deltas(&mut acc, weights, &adds[..len], &[]);
            // T6c: seed the PSQT sums from zero (the skip has no bias term).
            let mut psqt = [0i32; BUCKETS];
            if !psqt_weights.is_empty() {
                for &f in &adds[..len] {
                    psqt_add(&mut psqt, psqt_weights, f);
                }
            }
            *slot = Some(FinnyEntry {
                acc,
                psqt,
                by_piece,
            });
        }
        // COLD path only (the slot was empty): one from-scratch build, then the
        // single copy into the stack. Leaving it as a copy is deliberate — the
        // cold branch runs once per king key per search, so a second kernel
        // variant here would buy nothing.
        let cached = slot.as_ref().expect("slot filled above");
        stack[ply].acc[perspective as usize] = cached.acc;
        stack[ply].psqt[perspective as usize] = cached.psqt;
    }
}

/// Fused fold: `dst = src ± dirty deltas` in one pass — the memmove-free form
/// of "copy parent then [`apply_dirty`] in place" (profiling-study opt #2).
/// The resolved indices drive both the accumulator kernel and (when the net
/// carries one) the T6c PSQT bucket-sum fold, so the two channels can never
/// see different feature events.
fn apply_dirty_into(
    weights: &[i16],
    psqt_weights: &[i32],
    entry: &mut Entry,
    parent: &Entry,
    dirty: &DirtyPiece,
    perspective: Color,
    king: Square,
) {
    apply_dirty_directed::<true>(
        weights,
        psqt_weights,
        entry,
        parent,
        dirty,
        perspective,
        king,
    )
}

/// [`apply_dirty_into`] in either direction. `FORWARD` is the ordinary
/// parent → child fold; `!FORWARD` reconstructs a **parent** from its already
/// computed child by swapping the add and sub lists — the inverse of the same
/// move, which is exact because the fold is a sum of whole rows and wrapping
/// i16 addition is invertible.
///
/// The backward direction is what lets [`HalfkaContext::compute`] fill the
/// plies BETWEEN a king move and the refreshed leaf (perf study 12 §6, after
/// `AccumulatorStack::backward_update_incremental` at `eca43a97`
/// `src/nnue/nnue_accumulator.cpp`). `king` is the CURRENT king square in both
/// directions, which is sound precisely because the backward span starts at the
/// king move: no ply inside it moves the king again, so every HalfKA index in
/// the span is bucketed on the same square.
fn apply_dirty_directed<const FORWARD: bool>(
    weights: &[i16],
    psqt_weights: &[i32],
    entry: &mut Entry,
    parent: &Entry,
    dirty: &DirtyPiece,
    perspective: Color,
    king: Square,
) {
    let p = perspective as usize;
    let mut adds = [0usize; 2];
    let mut subs = [0usize; 2];
    for (i, &(piece, square)) in dirty.adds().iter().enumerate() {
        adds[i] = halfka_index(perspective, piece, square, king);
    }
    for (i, &(piece, square)) in dirty.subs().iter().enumerate() {
        subs[i] = halfka_index(perspective, piece, square, king);
    }
    let (a, s) = if FORWARD {
        (&adds[..dirty.adds().len()], &subs[..dirty.subs().len()])
    } else {
        (&subs[..dirty.subs().len()], &adds[..dirty.adds().len()])
    };
    apply_i16_deltas_into(&mut entry.acc[p], &parent.acc[p], weights, a, s);
    if !psqt_weights.is_empty() {
        psqt_fold(&mut entry.psqt[p], &parent.psqt[p], psqt_weights, a, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::sota_features::{HALFKA_DIMS, refresh_pst};
    use crate::movegen::generate_legal;
    use crate::moves::MoveList;

    fn weights() -> Vec<i16> {
        (0..HALFKA_DIMS * ACC)
            .map(|i| ((i.wrapping_mul(17) % 251) as i16) - 125)
            .collect()
    }

    /// Distinct pseudo-random PSQ1 rows per feature (same signature-weights
    /// rationale as the threat tests: one wrong/missing/swapped feature event
    /// perturbs a distinct 8-wide row and the sums diverge).
    fn psqt_table() -> Vec<i32> {
        (0..HALFKA_DIMS * BUCKETS)
            .map(|i| {
                let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z ^= z >> 31;
                (z % 4001) as i32 - 2000
            })
            .collect()
    }

    #[test]
    fn incremental_and_finny_match_refresh() {
        let weights = weights();
        let bias: Vec<i16> = (0..ACC).map(|i| (i % 19) as i16 - 9).collect();
        let psqt = psqt_table();
        let mut context = HalfkaContext::new(&weights, &bias, &psqt);
        let mut board = Board::startpos();
        context.refresh_root(&board);
        let mut seed = 0xD40F_1A11_5EEDu64;
        for ply in 0..120usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            context.push(ply, dirty);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            let (got, got_psqt) = context.current_full(&board, ply + 1);
            let (got, got_psqt) = (*got, *got_psqt);
            for perspective in Color::ALL {
                let fresh = refresh_pst(&weights, &bias, &board, perspective);
                assert_eq!(got[perspective as usize], fresh, "ply {ply} {m}");
                // T6c: the incremental + Finny-cached PSQT sums equal the
                // from-scratch recompute at every node (gate 2b).
                let fresh_psqt = refresh_psqt_halfka(&psqt, &board, perspective);
                assert_eq!(
                    got_psqt[perspective as usize], fresh_psqt,
                    "psqt drift at ply {ply} {m}"
                );
            }
        }
    }

    /// The BACKWARD FILL (perf study 12 §6) reconstructs every ply between a
    /// king move and the refreshed leaf. **`incremental_and_finny_match_refresh`
    /// cannot reach that code**: it calls `current_full` at every ply, so the
    /// span is always length 1, `king_ply == ply`, and the fill loop is empty.
    /// This test walks a line WITHOUT evaluating, so the span is long and
    /// contains a king move, then checks every reconstructed ancestor against a
    /// from-scratch refresh of the position at that ply.
    ///
    /// It fails if the reverse fold does not swap adds/subs, if the fill stops
    /// at the wrong ply, or if the king square used to bucket the reconstructed
    /// indices is wrong.
    #[test]
    fn backward_fill_reconstructs_the_span_exactly() {
        let weights = weights();
        let bias: Vec<i16> = (0..ACC).map(|i| (i % 19) as i16 - 9).collect();
        let psqt = psqt_table();

        // Kings are free to move on both sides, and there is enough material
        // that the plies after the king move still change the accumulator.
        let start = "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1";

        // Play a line where SOME ply moves a king, evaluating only at the end.
        // Try several seeds so at least one line actually contains a king move
        // in the unevaluated span (asserted below).
        let mut exercised = 0usize;
        for seed0 in [0xA11C_E5EEu64, 0xB0B0_CAFE, 0xF00D_BEEF, 0x1234_5678] {
            let mut context = HalfkaContext::new(&weights, &bias, &psqt);
            let mut board = Board::from_fen(start).expect("valid fen");
            context.refresh_root(&board);

            let mut boards = vec![board.clone()];
            let mut seed = seed0;
            let mut saw_king_move = false;
            const SPAN: usize = 6;
            for ply in 0..SPAN {
                let mut moves = MoveList::new();
                generate_legal(&board, &mut moves);
                if moves.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let m = moves[seed as usize % moves.len()];
                if board.piece_on(m.from()).map(|p| p.piece_type) == Some(PieceType::King) {
                    saw_king_move = true;
                }
                let dirty = DirtyPiece::from_move(&board, m);
                // PUSH ONLY — no `current_full`, so nothing in the span is
                // computed and the leaf's ancestor walk runs all the way back.
                context.push(ply, dirty);
                let gives_check = board.gives_check(m);
                board.make_move(m, gives_check);
                boards.push(board.clone());
            }
            if !saw_king_move || boards.len() <= SPAN {
                continue;
            }
            exercised += 1;

            // Evaluate ONLY at the leaf. This is what triggers the refresh and
            // the backward fill.
            let leaf = boards.len() - 1;
            let _ = context.current_full(&board, leaf);

            // Every ply in the span must now agree with a from-scratch refresh
            // of ITS OWN position. `current_full` at an already-computed ply is
            // a pure read, so this reads back exactly what the fill wrote.
            for (ply, b) in boards.iter().enumerate() {
                let (got, got_psqt) = context.current_full(b, ply);
                let (got, got_psqt) = (*got, *got_psqt);
                for perspective in Color::ALL {
                    assert_eq!(
                        got[perspective as usize],
                        refresh_pst(&weights, &bias, b, perspective),
                        "backward-filled accumulator wrong at ply {ply} (seed {seed0:#x})"
                    );
                    assert_eq!(
                        got_psqt[perspective as usize],
                        refresh_psqt_halfka(&psqt, b, perspective),
                        "backward-filled psqt wrong at ply {ply} (seed {seed0:#x})"
                    );
                }
            }
        }
        assert!(
            exercised > 0,
            "no seed produced a king move inside the unevaluated span — the test \
             would silently cover nothing"
        );
    }

    /// A net without a PSQ1 table keeps all-zero sums forever (the inert
    /// channel legacy bit-identity leans on).
    #[test]
    fn empty_psqt_table_stays_zero() {
        let weights = weights();
        let bias: Vec<i16> = (0..ACC).map(|i| (i % 19) as i16 - 9).collect();
        let mut context = HalfkaContext::new(&weights, &bias, &[]);
        let mut board = Board::startpos();
        context.refresh_root(&board);
        let mut seed = 0x0E11_D00D_5EEDu64;
        for ply in 0..40usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            context.push(ply, dirty);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            let (_, got_psqt) = context.current_full(&board, ply + 1);
            assert_eq!(*got_psqt, [[0i32; BUCKETS]; 2], "ply {ply} {m}");
        }
    }
}
