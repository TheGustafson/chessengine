//! Incremental HalfKAv2_hm accumulator with a per-king-state Finny cache.
//!
//! T6c FT PSQT skip: alongside every `[i16; ACC]` accumulator lane set this
//! module maintains the per-perspective `[i32; BUCKETS]` PSQT bucket sums — the
//! same add/sub feature events applied to the net's 8-wide PSQ1 rows. The Finny
//! cache stores/restores the sums under the same keys as the accumulator, so a
//! cached refresh cannot desync the skip channel; a net without a PSQ1 table
//! (`psqt_weights` empty) leaves the sums zero, bit-identical to pre-T6c.

use super::sota_features::{
    ACC, RowOffset, halfka_index, halfka_king_key, halfka_oriented, halfka_row_base,
};
use super::sota_head::BUCKETS;
use super::sota_simd::{apply_halfka_deltas, apply_halfka_deltas_dual, apply_halfka_deltas_into};
use crate::bitboard::Bitboard;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, PieceType, Square};

const STACK_LEN: usize = MAX_PLY + 8;
const FINNY_KEYS: usize = 64;

/// Software-prefetch arm for the Finny warm refresh — diagnostic only; one
/// binary, modes selected by `P43_PF` at run time so control and arm share the
/// same bytes. 0 off; 1 every line of every fold row at `T0` after the diff
/// walk; 2 as 1 but `T2`; 3 first line of each row only; 4 the Finny slot's own
/// payload before the diff walk; 5 = 1 + 4.
#[cfg(feature = "pf43")]
pub(crate) mod pf43 {
    use super::{ACC, RowOffset};
    use std::sync::LazyLock;

    static MODE: LazyLock<u32> = LazyLock::new(|| {
        std::env::var("P43_PF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    });

    #[inline]
    fn hint(p: *const i8, t0: bool) {
        #[cfg(target_arch = "x86_64")]
        {
            #[allow(unsafe_code)]
            // SAFETY: `_mm_prefetch` is a pure cache hint, defined for ANY
            // address — it cannot fault and has no architectural effect beyond
            // cache state, so no validity or provenance claim about `p` is
            // required. Same justification as `TranspositionTable::prefetch`.
            unsafe {
                use std::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T2, _mm_prefetch};
                if t0 {
                    _mm_prefetch(p, _MM_HINT_T0);
                } else {
                    _mm_prefetch(p, _MM_HINT_T2);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (p, t0);
    }

    /// Hint the weight rows this refresh is about to fold. `ACC` i8 per row.
    #[inline]
    pub(crate) fn rows(weights: &[i8], adds: &[RowOffset], subs: &[RowOffset]) {
        let mode = *MODE;
        if mode == 0 || mode == 4 {
            return;
        }
        let full = mode != 3;
        let t0 = mode != 2;
        for &f in adds.iter().chain(subs) {
            // `wrapping_add`: a hint needs no in-bounds proof.
            let base = weights.as_ptr().wrapping_add(f.0);
            if full {
                for off in (0..ACC).step_by(64) {
                    hint(base.wrapping_add(off), t0);
                }
            } else {
                hint(base, t0);
            }
        }
    }

    /// Hint the Finny slot's own payload, before the set-difference walk.
    #[inline]
    pub(crate) fn slot(acc: &[i16; ACC]) {
        if !matches!(*MODE, 4 | 5) {
            return;
        }
        let base = acc.as_ptr().cast::<i8>();
        for off in (0..ACC * 2).step_by(64) {
            hint(base.wrapping_add(off), true);
        }
    }
}

/// `N` clones of `value` on the heap as a fixed-size array: a THIN
/// `Box<[T; N]>` with a compile-time length, not a fat `Box<[T]>` whose every
/// index reloads it. Routed through `Vec` because `Box::new([…; N])` would
/// build the 1.3 MB array as a STACK temporary first.
fn boxed_array<T: Clone, const N: usize>(value: T, n: usize) -> Box<[T; N]> {
    debug_assert_eq!(n, N, "boxed_array length must equal the array size");
    vec![value; n]
        .into_boxed_slice()
        .try_into()
        .ok()
        .expect("length checked above")
}

/// Feature `feature`'s PSQ1 row as a FIXED-size array reference. `table` is
/// row-major (`feature · BUCKETS + bucket`) and must be non-empty.
///
/// The fixed size is the point: `BUCKETS` i32 is one `ymm`, but a RUNTIME-length
/// `&[i32]` hides the trip count from LLVM's SLP vectoriser, which then emits a
/// 4-lane group, a 2-lane group and two SCALAR lanes. `as_chunks` also states
/// the shape once, so the row index is ONE bounds check.
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
///
/// **Two tempting rewrites were MEASURED NEGATIVE here — do not redo them.**
/// Hoisting `as_chunks` out of the two loops is a null (this fold is
/// `#[inline(always)]` with at most two iterations per list, so LLVM already
/// CSEs the shift), and compile-time row counts (`psqt_lanes::<NA, NS>`) cost
/// **+7.12 i/node** — the straight-line form buys nothing over two iterations
/// and pays for copying the indices into `[usize; NA]`.
#[inline(always)]
pub(crate) fn psqt_fold(
    dst: &mut [i32; BUCKETS],
    src: &[i32; BUCKETS],
    table: &[i32],
    adds: &[usize],
    subs: &[usize],
) {
    // Accumulate in a LOCAL and store once: paired with `psqt_row`'s fixed-size
    // row this keeps the eight buckets in one ymm across the whole fold instead
    // of round-tripping to the entry per feature. The pairing is what pays.
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
/// reference the incremental path must match; zero when `table` is empty.
/// Tests and the debug oracle only, so release builds compile it out.
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

// CACHE-LINE ALIGNED, load-bearing: without `align(64)` a per-ply accumulator
// starts further out of phase every ply and splits cache lines in the fold
// kernels. HOT/COLD SPLIT: the per-ply BOOKKEEPING (`dirty`, `computed`) lives
// in [`Meta`], not here, so `compute`'s metadata walk is dense instead of a
// 5 KB-stride pointer chase. The payload here is streamed once per ply per
// perspective and gains nothing from having the flags nearby.
#[derive(Clone)]
#[repr(align(64))]
struct Entry {
    acc: [[i16; ACC]; 2],
    psqt: [[i32; BUCKETS]; 2],
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            acc: [[0; ACC]; 2],
            psqt: [[0; BUCKETS]; 2],
        }
    }
}

/// The per-ply bookkeeping [`HalfkaContext::compute`] walks, split out of
/// [`Entry`] so the walk is dense: **16 bytes per ply**, four plies to a cache
/// line, against one line per ply 5184 bytes apart inside `Entry`. `Entry`
/// itself is 5184 B (`acc` 2 × ACC × 2 + `psqt` 2 × BUCKETS × 4), an exact 81
/// lines, so every ply's `acc[p]` stays 64-byte aligned and the fold's `zmm`
/// stores stay line-aligned.
#[derive(Clone, Copy)]
struct Meta {
    dirty: DirtyPiece,
    computed: [bool; 2],
}

/// `Meta` must stay 16 bytes wide: `compute` indexes `meta` thousands of times
/// per node, which LLVM emits as a `shl $0x4` only while the stride is 16 —
/// widening the struct by one byte turns every index into an `imul`, silently.
/// A `const` block fails the BUILD, so it cannot be skipped the way a test can.
const _: () = assert!(
    std::mem::size_of::<Meta>() == 16,
    "Meta must stay 16 bytes: compute's meta indexing is a shl $4"
);

impl Default for Meta {
    fn default() -> Self {
        Self {
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}

/// A Finny-cache slot: the accumulator for one (perspective, king-bucket) key
/// plus the POSITION it was computed from. Validity lives in
/// [`HalfkaContext::cache_valid`], not in a tag here.
///
/// **PROBE-FIRST FIELD ORDER, pinned by `repr(C)`** because Rust's default
/// layout is unspecified and put the hot head at the END. The set-difference
/// refresh reads `by_piece` and updates `psqt` before the fold streams `acc`,
/// and the sizes make the intent exact: `by_piece` 96 B + `psqt` 32 B = 128 B =
/// two whole cache lines, then `acc` starts 64-B aligned at offset 128 and the
/// slot is 2688 B — exactly 42 lines, no tail padding.
#[derive(Clone)]
#[repr(C, align(64))]
struct FinnyEntry {
    // The cached POSITION as per-(color,type) bitboards, SF's shape: the
    // refresh diff is 12 XOR/AND-NOTs + index computation for CHANGED squares
    // only, instead of two binary-search passes over sorted feature lists.
    by_piece: [[Bitboard; 6]; 2],
    psqt: [i32; BUCKETS],
    /// Keeps `acc` 64-B aligned for ANY bucket count. With the original 8
    /// buckets `by_piece` + `psqt` came to exactly 128 B and this is
    /// zero-sized; a queen-split arch (18 buckets) makes `psqt` 72 B, which
    /// would otherwise land `acc` at offset 168 and trip the alignment
    /// assertion in the fold.
    _pad: [u8; FINNY_PAD],
    acc: [i16; ACC],
}

/// Bytes needed after `by_piece` + `psqt` to round up to a 64-B boundary.
const FINNY_PAD: usize = (64 - ((96 + BUCKETS * 4) % 64)) % 64;

impl Default for FinnyEntry {
    fn default() -> Self {
        Self {
            by_piece: [[Bitboard::EMPTY; 6]; 2],
            psqt: [0; BUCKETS],
            _pad: [0; FINNY_PAD],
            acc: [0; ACC],
        }
    }
}

pub(crate) struct HalfkaContext<'a> {
    /// The i8 HalfKA table. The accumulator stays i16 — only the streamed
    /// WEIGHT ROWS shrink, which is where the bytes were.
    weights: &'a [i8],
    bias: &'a [i16],
    /// T6c PSQ1 rows for the HalfKA feature space (`HALFKA_DIMS · BUCKETS`
    /// i32, row-major), or empty ⇒ the skip channel is inert.
    psqt_weights: &'a [i32],
    /// The per-ply accumulator stack.
    ///
    /// `Box<[Entry; STACK_LEN]>`, not `Box<[Entry]>`: a boxed SLICE is a fat
    /// pointer, so every `stack[ply]` reloads the length and LLVM cannot fold
    /// the check away across the outlined fold calls. A boxed ARRAY is a thin
    /// pointer with a compile-time length, so the index is one `cmp` against an
    /// immediate and the panic edge is provably dead where `ply` is bounded.
    stack: Box<[Entry; STACK_LEN]>,
    /// Per-ply bookkeeping, parallel to `stack`.
    meta: Box<[Meta; STACK_LEN]>,
    cache: Box<[FinnyEntry; 2 * FINNY_KEYS]>,
    /// Which Finny slots have been built, one bit per king key, per perspective.
    /// Not `Box<[Option<FinnyEntry>]>`: `FinnyEntry` is all integers, so the
    /// `Option` had no niche and cost a whole 64-byte cache line per slot that
    /// the probe had to read first. Slots are only ever FILLED, never
    /// invalidated, so the bitmap is monotone — once set, `cache[i]`'s
    /// `by_piece` IS the position its `acc`/`psqt` were computed from.
    cache_valid: [u64; 2],
}

impl<'a> HalfkaContext<'a> {
    pub(crate) fn new(weights: &'a [i8], bias: &'a [i16], psqt_weights: &'a [i32]) -> Self {
        #[allow(unused_mut)]
        let mut me = Self {
            weights,
            bias,
            psqt_weights,
            // Boxing the array directly would build it on the STACK first, and
            // 1.3 MB of `Entry` overflows the thread stack; `boxed_array` goes
            // through `vec!` instead.
            stack: boxed_array(Entry::default(), STACK_LEN),
            meta: boxed_array(Meta::default(), STACK_LEN),
            cache: boxed_array(FinnyEntry::default(), 2 * FINNY_KEYS),
            cache_valid: [0; 2],
        };
        me
    }

    pub(crate) fn refresh_root(&mut self, board: &Board) {
        for perspective in Color::ALL {
            self.finny_refresh_into(board, perspective, 0);
        }
        self.meta[0].dirty = DirtyPiece::EMPTY;
        self.meta[0].computed = [true; 2];
    }

    pub(crate) fn push(&mut self, ply: usize, dirty: DirtyPiece) {
        self.meta[ply + 1] = Meta {
            dirty,
            computed: [false; 2],
        };
    }

    /// The current accumulator pair plus the per-perspective T6c PSQT bucket
    /// sums of the SAME entry (all-zero for a net without a PSQ1 table).
    pub(crate) fn current_full(
        &mut self,
        board: &Board,
        ply: usize,
    ) -> (&[[i16; ACC]; 2], &[[i32; BUCKETS]; 2]) {
        // KNOWN REGRESSION: iterate a `static` BY REFERENCE, never
        // `for perspective in Color::ALL` — consuming the array builds a
        // `core::array::IntoIter` whose `alloca` LLVM declines to promote, so
        // the constant is stored to the stack and reloaded per iteration.
        static PERSPECTIVES: [Color; 2] = Color::ALL;
        for &perspective in &PERSPECTIVES {
            self.compute(board, ply, perspective);
        }
        let entry = &self.stack[ply];
        (&entry.acc, &entry.psqt)
    }

    fn compute(&mut self, board: &Board, ply: usize, perspective: Color) {
        let p = perspective as usize;
        if self.meta[ply].computed[p] {
            return;
        }
        let mut ancestor = ply;
        while ancestor > 0 && !self.meta[ancestor].computed[p] {
            ancestor -= 1;
        }
        let king = Piece::new(perspective, crate::types::PieceType::King);
        // The LAST king move in the span: every ply from there to `ply` shares
        // the post-move king square, so all are reachable from the refreshed
        // leaf by plain reverse folds.
        let last_king_move = ((ancestor + 1)..=ply).rev().find(|&child| {
            self.meta[child]
                .dirty
                .subs()
                .iter()
                .any(|&(piece, _)| piece == king)
        });
        if let Some(king_ply) = last_king_move {
            self.finny_refresh_into(board, perspective, ply);
            self.meta[ply].computed[p] = true;
            // BACKWARD FILL (SF's `backward_update_incremental`). Refreshing
            // only the leaf leaves every ply between the king move and this
            // node uncomputed, so the next sibling below any of them walks
            // straight past the king move and refreshes from scratch again —
            // measured at 1.89x SF's refresh rate before this, parity after.
            // Reconstructing the ancestors costs one reverse fold each and is
            // bit-identical: a reverse fold subtracts exactly the rows the
            // forward fold added, and wrapping i16 addition is invertible.
            let king_square = board.king_square(perspective);
            // Disjoint field borrows, same as the forward loop below.
            let Self {
                weights,
                psqt_weights,
                stack,
                meta,
                ..
            } = self;
            let (weights, psqt_weights) = (*weights, *psqt_weights);
            for child in (king_ply..ply).rev() {
                // `child` is reconstructed from `child + 1` by undoing the move
                // that LED to `child + 1`.
                let dirty = &meta[child + 1].dirty;
                // The whole loop bucket-indexes on ONE king square, sound only
                // because `king_ply` is the LAST king move in the span. Pinned
                // because the natural next optimisation — refreshing on a king
                // BUCKET change rather than any king move — breaks it silently.
                debug_assert!(
                    !dirty.subs().iter().any(|&(piece, _)| piece == king),
                    "backward span must not contain a king move"
                );
                let (before, after) = stack.split_at_mut(child + 1);
                let entry = &mut before[child];
                let source = &after[0];
                // This loop's own bound, not the forward loop's: it reads
                // `stack[child + 1]`, and `child + 1 >= king_ply + 1 >=
                // ancestor + 2 >= 2`, so `stack[0]`'s empty record is unreachable.
                debug_assert!(
                    !(dirty.adds().is_empty() && dirty.subs().is_empty()),
                    "empty HalfKA dirty record on the backward fold path"
                );
                apply_dirty_directed::<false>(
                    weights,
                    psqt_weights,
                    entry,
                    source,
                    dirty,
                    perspective,
                    king_square,
                );
                meta[child].computed[p] = true;
            }
            return;
        }
        let king_square = board.king_square(perspective);
        // Read the dirty record WHERE IT LIVES. `let dirty = self.meta[child]
        // .dirty;` copies 14 bytes to the stack and reads the length and rows
        // back off it — a store-to-load forward on the index loop's critical
        // path. The copy exists only because `apply_dirty_into` wants
        // `&DirtyPiece` while `self.stack` is borrowed mutably; destructuring
        // `self` into DISJOINT FIELD BORROWS is the idiomatic answer.
        let Self {
            weights,
            psqt_weights,
            stack,
            meta,
            ..
        } = self;
        let (weights, psqt_weights) = (*weights, *psqt_weights);
        for child in (ancestor + 1)..=ply {
            let dirty = &meta[child].dirty;
            // Fused parent→child fold: read parent lanes, apply deltas, write
            // child lanes in one pass — no separate ~1.5 KB memmove.
            //
            // There is deliberately NO empty-record special case: for HalfKA a
            // dirty record is empty only for a null move, and a null move
            // pushes no entry at all (`stack[0]`'s record is never read — this
            // loop starts at `ancestor + 1 >= 1`). The `debug_assert` below is
            // the tripwire for that.
            //
            // **HAZARD: the empty-list case of the delta kernels must remain a
            // copy.** Adding an `if adds.is_empty() && subs.is_empty()
            // { return; }` fast path to `apply_halfka_deltas_into` looks free
            // and would turn this into a silent stale-child bug in release.
            let (before, after) = stack.split_at_mut(child);
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
                dirty,
                perspective,
                king_square,
            );
            meta[child].computed[p] = true;
        }
    }

    /// Finny-cached from-scratch refresh, written **in place**: the cache
    /// entry's accumulator is updated where it lives rather than round-tripping
    /// ~2.5 KB through a local and a fresh `FinnyEntry`. Same deltas, same
    /// kernel, same arithmetic ⇒ bit-identical cache and stack contents.
    fn finny_refresh_into(&mut self, board: &Board, perspective: Color, ply: usize) {
        let king = board.king_square(perspective);
        let key = halfka_king_key(perspective, king);
        let weights = self.weights;
        let psqt_weights = self.psqt_weights;
        // Split the borrow so the fold can write the cache entry AND the stack
        // slot in one pass: `cache` and `stack` are distinct fields, so
        // destructuring is what makes two `&mut` legal without `unsafe`.
        let Self {
            cache,
            cache_valid,
            stack,
            bias,
            ..
        } = self;
        let p = perspective as usize;
        let cached = &mut cache[p * FINNY_KEYS + key];
        let valid_bit = 1u64 << key;
        if cache_valid[p] & valid_bit != 0 {
            #[cfg(feature = "pf43")]
            pf43::slot(&cached.acc);
            // Bitboard set-diff against the cached position. Same add/sub SETS
            // as a sorted-list derivation — order may differ, but wrapping
            // i16/i32 adds commute, so the accumulator and PSQT sums stay
            // bit-identical. This walk is written for codegen; three things it
            // must NOT do:
            //
            //  * iterate `PieceType::ALL` BY VALUE — LLVM materialises the
            //    6-byte array on the stack per call and reloads the piece type
            //    per iteration. Zipping the two `[Bitboard; 6]` ROWS against a
            //    `static` gives a compile-time trip count and no such memory.
            //  * recompute the row base per direction — hoisted (see
            //    `halfka_row_base`).
            //  * bounds-check the cursor writes — at most one index is pushed
            //    per occupied square and there are at most 32 pieces, so a
            //    cursor cannot exceed `CAP`; `& (CAP - 1)` is free because CAP
            //    is a power of two, and the `debug_assert` carries the real
            //    invariant.
            const CAP: usize = 32;
            static PIECE_TYPES: [PieceType; 6] = PieceType::ALL;
            // TWO parallel lists per direction, because the consumers want
            // different units: the fold kernel walks its list once per TILE, so
            // it takes pre-scaled [`RowOffset`]s and never multiplies, while
            // PSQ1 rows are `BUCKETS` wide, not `ACC`, so `psqt_add`/`psqt_sub`
            // keep the plain feature index.
            let mut adds = [RowOffset::default(); CAP];
            let mut add_feat = [0usize; CAP];
            let mut add_len = 0;
            let mut subs = [RowOffset::default(); CAP];
            let mut sub_feat = [0usize; CAP];
            let mut sub_len = 0;
            for color in [Color::White, Color::Black] {
                // `board.pieces` is the `pub(crate)` FIELD, not the accessor:
                // the row, not one bitboard, is the unit the diff walks.
                let new_row = &board.pieces[color as usize];
                let old_row = &mut cached.by_piece[color as usize];
                for ((&new_bb, old_bb), &pt) in new_row.iter().zip(old_row).zip(&PIECE_TYPES) {
                    if new_bb == *old_bb {
                        continue;
                    }
                    let piece = Piece::new(color, pt);
                    let base = halfka_row_base(perspective, piece, king);
                    // A software prefetch here (SF's `append_changed_indices`
                    // shape) was measured NOT RESOLVABLE — the arm is
                    // indistinguishable from a byte-different null — because
                    // one hint can convert at most one of the ~22 demand misses
                    // a 40-line row takes. Don't add one back without a new
                    // mechanism; the `pf43` module above is the harness.
                    for sq in new_bb & !*old_bb {
                        debug_assert!(add_len < CAP, "finny add cursor overflow");
                        let f = base + halfka_oriented(perspective, sq, king);
                        add_feat[add_len & (CAP - 1)] = f;
                        adds[add_len & (CAP - 1)] = RowOffset::of(f);
                        add_len += 1;
                    }
                    for sq in *old_bb & !new_bb {
                        debug_assert!(sub_len < CAP, "finny sub cursor overflow");
                        let f = base + halfka_oriented(perspective, sq, king);
                        sub_feat[sub_len & (CAP - 1)] = f;
                        subs[sub_len & (CAP - 1)] = RowOffset::of(f);
                        sub_len += 1;
                    }
                    *old_bb = new_bb;
                }
            }
            // Two-destination fold: the cache entry is its own memo for the
            // next refresh with this king key and the stack slot is what the
            // head reads — the same bytes in two places. Folding in place and
            // then assigning the stack slot costs a full `memmove` (~103
            // instructions/node); writing both homes from the registers the
            // fold already holds costs one store per 32 lanes.
            #[cfg(feature = "pf43")]
            pf43::rows(weights, &adds[..add_len], &subs[..sub_len]);
            apply_halfka_deltas_dual(
                &mut cached.acc,
                &mut stack[ply].acc[p],
                weights,
                &adds[..add_len],
                &subs[..sub_len],
            );
            // T6c: the cache's PSQT sums see the SAME add/sub set, so a cached
            // refresh restores the skip channel exactly.
            if !psqt_weights.is_empty() {
                // Same register-resident accumulate as `psqt_fold`.
                let mut acc = cached.psqt;
                for &f in &add_feat[..add_len] {
                    psqt_add(&mut acc, psqt_weights, f);
                }
                for &f in &sub_feat[..sub_len] {
                    psqt_sub(&mut acc, psqt_weights, f);
                }
                cached.psqt = acc;
            }
            // The accumulator is already in the stack slot (the dual fold put
            // it there); only the 32-byte PSQT pair is left to publish.
            stack[ply].psqt[p] = cached.psqt;
            return;
        }
        // COLD path only (this key has never been built): one from-scratch
        // build, then a single copy into the stack. The copy is deliberate —
        // this branch runs once per king key per search, so a second kernel
        // variant would buy nothing.
        cached.acc.copy_from_slice(&bias[..ACC]);
        let mut adds = [RowOffset::default(); 32];
        let mut add_feat = [0usize; 32];
        let mut len = 0usize;
        cached.by_piece = [[Bitboard::EMPTY; 6]; 2];
        for square in board.all_occupancy() {
            let piece = board.piece_on(square).expect("occupied square");
            let f = halfka_index(perspective, piece, square, king);
            add_feat[len] = f;
            adds[len] = RowOffset::of(f);
            len += 1;
            cached.by_piece[piece.color as usize][piece.piece_type as usize] |=
                Bitboard::from_square(square);
        }
        apply_halfka_deltas(&mut cached.acc, weights, &adds[..len], &[]);
        // T6c: seed the PSQT sums from zero (the skip has no bias term).
        cached.psqt = [0i32; BUCKETS];
        if !psqt_weights.is_empty() {
            let mut psqt = [0i32; BUCKETS];
            for &f in &add_feat[..len] {
                psqt_add(&mut psqt, psqt_weights, f);
            }
            cached.psqt = psqt;
        }
        cache_valid[p] |= valid_bit;
        stack[ply].acc[p] = cached.acc;
        stack[ply].psqt[p] = cached.psqt;
    }
}

/// Fused fold: `dst = src ± dirty deltas` in one pass — the memmove-free form
/// of "copy parent then apply deltas in place". The resolved indices drive both
/// the accumulator kernel and (when the net carries one) the T6c PSQT
/// bucket-sum fold, so the two channels cannot see different feature events.
fn apply_dirty_into(
    weights: &[i8],
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
/// plies BETWEEN a king move and the refreshed leaf. `king` is the CURRENT king
/// square in both directions, sound precisely because the backward span starts
/// at the king move: no ply inside it moves the king again, so every HalfKA
/// index in the span buckets on the same square.
fn apply_dirty_directed<const FORWARD: bool>(
    weights: &[i8],
    psqt_weights: &[i32],
    entry: &mut Entry,
    parent: &Entry,
    dirty: &DirtyPiece,
    perspective: Color,
    king: Square,
) {
    let p = perspective as usize;
    // Scaling to [`RowOffset`] here rather than in the kernel: the kernel's
    // row-list unit is global (the same `i8_deltas_into` serves both channels),
    // and the threat channel pays the multiply once per tile.
    let mut adds = [RowOffset::default(); 2];
    let mut add_feat = [0usize; 2];
    let mut subs = [RowOffset::default(); 2];
    let mut sub_feat = [0usize; 2];
    // Zip the DESTINATIONS, don't index them by a counter: with
    // `.enumerate()` the index range is a runtime fact, so LLVM emits a bounds
    // check per row for the two `[_; 2]` stores and cannot unroll. Zipping
    // `adds`/`add_feat`, whose length is 2 at compile time, makes the trip
    // count `min(2, 2, n)` and the checks vanish by construction. The multiset
    // of rows and the order within each list are unchanged ⇒ bit-identical.
    //
    // `zip().zip()` nests, so the item is `((row, feat), &(piece, square))`;
    // that is the price of pairing three iterators with no itertools
    // dependency (D15 keeps [dependencies] empty), and slice iterators
    // specialise it away via `TrustedRandomAccess`.
    for ((row, feat), &(piece, square)) in adds.iter_mut().zip(&mut add_feat).zip(dirty.adds()) {
        let f = halfka_index(perspective, piece, square, king);
        *feat = f;
        *row = RowOffset::of(f);
    }
    for ((row, feat), &(piece, square)) in subs.iter_mut().zip(&mut sub_feat).zip(dirty.subs()) {
        let f = halfka_index(perspective, piece, square, king);
        *feat = f;
        *row = RowOffset::of(f);
    }
    let (na, ns) = (dirty.adds().len(), dirty.subs().len());
    let (a, s) = if FORWARD {
        (&adds[..na], &subs[..ns])
    } else {
        (&subs[..ns], &adds[..na])
    };
    apply_halfka_deltas_into(&mut entry.acc[p], &parent.acc[p], weights, a, s);
    if !psqt_weights.is_empty() {
        let (af, sf) = if FORWARD {
            (&add_feat[..na], &sub_feat[..ns])
        } else {
            (&sub_feat[..ns], &add_feat[..na])
        };
        psqt_fold(&mut entry.psqt[p], &parent.psqt[p], psqt_weights, af, sf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::sota_features::{HALFKA_DIMS, refresh_pst};
    use crate::movegen::generate_legal;
    use crate::moves::MoveList;

    fn weights() -> Vec<i8> {
        (0..HALFKA_DIMS * ACC)
            .map(|i| ((i.wrapping_mul(17) % 251) as i16 - 125) as i8)
            .collect()
    }

    /// Distinct pseudo-random PSQ1 rows per feature, so one wrong, missing or
    /// swapped feature event perturbs a distinct 8-wide row and the sums diverge.
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

    /// The BACKWARD FILL reconstructs every ply between a king move and the
    /// refreshed leaf. **`incremental_and_finny_match_refresh` cannot reach that
    /// code** — it evaluates at every ply, so the span is always length 1. This
    /// test walks a line WITHOUT evaluating, then checks every reconstructed
    /// ancestor against a from-scratch refresh. It fails if the reverse fold
    /// does not swap adds/subs, if the fill stops at the wrong ply, or if the
    /// king square used to bucket the reconstructed indices is wrong.
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
