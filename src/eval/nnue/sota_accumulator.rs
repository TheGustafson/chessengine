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
use super::sota_simd::{apply_halfka_deltas, apply_halfka_deltas_dual, apply_halfka_deltas_into};
use crate::bitboard::Bitboard;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, PieceType, Square};

const STACK_LEN: usize = MAX_PLY + 8;
const FINNY_KEYS: usize = 64;

/// Perf study 43's software-prefetch arm for the Finny warm refresh.
///
/// **One binary, four modes, selected by `P43_PF` at run time.** Study 29
/// killed a prefetch here on a *separate* build, which confounds the arm with
/// code layout — and its own §5.3 records that a byte-different null moved the
/// cycles median by more than the arm did. Reading the mode from the
/// environment makes control and arm the same bytes, so layout differences out
/// by construction (perf study 39's idiom). The mode check costs one relaxed
/// load and one branch per refresh, i.e. **0.16 per node**.
///
/// Modes, chosen so each names a distinct reason study 29's arm could have
/// failed rather than repeating it:
///
/// | mode | what |
/// |---|---|
/// | 0 | off — the control |
/// | 1 | every line of every fold row, `T0`, issued after the diff walk |
/// | 2 | as 1 but `T2` — study 29's hint LEVEL at study 43's placement |
/// | 3 | first line of each row only, `T0` — study 29's COVERAGE at `T0` |
/// | 4 | the Finny slot's own 2560-byte payload, `T0`, before the diff walk |
/// | 5 | 1 + 4 |
#[cfg(feature = "pf43")]
pub(crate) mod pf43 {
    use super::ACC;
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
            // SAFETY: `_mm_prefetch` is a pure cache hint. It is defined for
            // ANY address value — it cannot fault, cannot trap, and has no
            // architectural effect beyond cache state — so no validity or
            // provenance claim about `p` is required. Same sanctioned use and
            // same justification as `TranspositionTable::prefetch` (doc 08
            // §3.15).
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
    pub(crate) fn rows(weights: &[i8], adds: &[usize], subs: &[usize]) {
        let mode = *MODE;
        if mode == 0 || mode == 4 {
            return;
        }
        let full = mode != 3;
        let t0 = mode != 2;
        for &f in adds.iter().chain(subs) {
            // `wrapping_add`, not `add`: a hint needs no in-bounds proof, and
            // the wrapping form keeps this free of a provenance argument.
            let base = weights.as_ptr().wrapping_add(f * ACC);
            if full {
                for off in (0..ACC).step_by(64) {
                    hint(base.wrapping_add(off), t0);
                }
            } else {
                hint(base, t0);
            }
        }
    }

    /// Hint the Finny slot's own payload, before the set-difference walk gives
    /// the fold ~200 instructions of cover.
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

/// `N` clones of `value` on the heap as a fixed-size array (perf study 22 arm
/// 1b).
///
/// The array size is a const generic so callers get `Box<[T; N]>` — a THIN
/// pointer with a compile-time length — rather than `Box<[T]>`, whose fat
/// pointer makes every index reload the length from memory. The route through
/// `Vec` is what keeps the 1.3 MB `Entry` array off the thread stack:
/// `Box::new([…; N])` would build the array as a temporary first.
///
/// `debug_assert_eq!` rather than a runtime check on the length: `n` is only
/// ever the same constant as `N` at every call site, and `TryFrom` for
/// `Box<[T]> -> Box<[T; N]>` already re-checks it (that check folds away).
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
//
// HOT/COLD SPLIT (perf study 22 arm 1). The 12 bytes of per-ply BOOKKEEPING
// that used to live here — `dirty` and `computed` — moved to [`Meta`]. They
// were the last two fields of a 5248-byte struct, so `stack[ply].computed[p]`
// and `stack[ply].dirty` sat in cache line 81 of 82 and consecutive plies were
// **5248 bytes apart**. Every one of `compute`'s three metadata touches (the
// early-out, the ancestor walk, the last-king-move scan) and every `push` was
// therefore a separate cache line, and the walk was a 5 KB-stride pointer
// chase. The payload here (`acc` + `psqt`) is read ONCE per ply per
// perspective by the fold, which is a streaming access that gains nothing from
// having the flags nearby.
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

/// The per-ply bookkeeping that [`HalfkaContext::compute`] walks, split out of
/// [`Entry`] so the walk is dense (perf study 22 arm 1).
///
/// 12 bytes per ply against `Entry`'s 5248, so the whole `STACK_LEN`-deep array
/// is ~3 KB — 48 cache lines for the entire search stack, versus one line per
/// ply 5248 bytes apart. A parent/child pair now shares a line instead of
/// guaranteeing two misses.
#[derive(Clone, Copy)]
struct Meta {
    dirty: DirtyPiece,
    computed: [bool; 2],
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}

/// A Finny-cache slot: the accumulator for one (perspective, king-bucket) key
/// plus the POSITION it was computed from.
///
/// # PROBE-FIRST FIELD ORDER (perf study 22 arm 2), and why `repr(C)`
///
/// The set-difference refresh reads `by_piece` and updates `psqt` *before* the
/// fold streams `acc`, so those 128 bytes are the slot's hot head and belong at
/// offset 0. Rust's default layout put them at the **end**: measured in the
/// shipped binary as `mov r11,[r14+rdx*1+0xa60]` — `by_piece` at byte 2656 of a
/// 2752-byte slot, i.e. cache lines 41-42, with the `Option` discriminant alone
/// on line 0 and 63 bytes of padding beside it. One probe therefore touched
/// **three** cache lines spread 2.6 KB apart to read 96 bytes and a tag.
///
/// `repr(C)` pins the order (Rust's default layout is explicitly unspecified,
/// so "it happens to be right today" is not a guarantee to build a perf claim
/// on), and the sizes make it exact: `by_piece` 96 B + `psqt` 32 B = 128 B =
/// two whole cache lines, then `acc` starts 64-B aligned at offset 128 and the
/// slot is 2688 B — exactly 42 lines, no tail padding.
///
/// The validity tag is gone: see [`HalfkaContext::cache_valid`].
#[derive(Clone)]
#[repr(C, align(64))]
struct FinnyEntry {
    // Perf study #2 option A: the cached POSITION as per-(color,type)
    // bitboards — the refresh diff is 12 XOR/AND-NOTs + index computation
    // for CHANGED squares only, replacing two binary-search passes over
    // sorted feature lists (was ~13% of all cycles; SF's shape).
    by_piece: [[Bitboard; 6]; 2],
    psqt: [i32; BUCKETS],
    acc: [i16; ACC],
}

impl Default for FinnyEntry {
    fn default() -> Self {
        Self {
            by_piece: [[Bitboard::EMPTY; 6]; 2],
            psqt: [0; BUCKETS],
            acc: [0; ACC],
        }
    }
}

pub(crate) struct HalfkaContext<'a> {
    /// The i8 HalfKA table (perf study 32). The accumulator stays i16 — only
    /// the streamed WEIGHT ROWS shrink, which is where the bytes were.
    weights: &'a [i8],
    bias: &'a [i16],
    /// T6c PSQ1 rows for the HalfKA feature space (`HALFKA_DIMS · BUCKETS`
    /// i32, row-major), or empty ⇒ the skip channel is inert.
    psqt_weights: &'a [i32],
    /// The per-ply accumulator stack.
    ///
    /// `Box<[Entry; STACK_LEN]>`, not `Box<[Entry]>` (perf study 22 arm 1b): a
    /// boxed SLICE is a fat pointer, so every `stack[ply]` reloads the length
    /// from memory and compares against it, and LLVM cannot fold the check away
    /// across the outlined fold calls. A boxed ARRAY is a thin pointer whose
    /// length is a compile-time constant, so the same index is one `cmp $0xfe`
    /// against an immediate — and, more importantly, the panic edge becomes
    /// provably dead wherever `ply` is already bounded. Measured as +58.7 i/n
    /// when arm 1 first moved the metadata to a second boxed SLICE.
    stack: Box<[Entry; STACK_LEN]>,
    /// Per-ply bookkeeping, parallel to `stack` (perf study 22 arm 1).
    meta: Box<[Meta; STACK_LEN]>,
    cache: Box<[FinnyEntry; 2 * FINNY_KEYS]>,
    /// Which Finny slots have been built, one bit per king key, per
    /// perspective (perf study 22 arm 2).
    ///
    /// This replaces `Box<[Option<FinnyEntry>]>`. The `Option` had no niche to
    /// hide in — `FinnyEntry` is all integers — so it cost a **whole 64-byte
    /// cache line per slot** (tag at offset 0, payload pushed to 64 by the
    /// align-64 requirement, 2752 B per slot instead of 2688) and the probe had
    /// to read that line before it could read anything useful. 128 keys fit in
    /// two `u64`s that live in the same line as the rest of the context and are
    /// never out of L1. Slots are only ever FILLED, never invalidated, so the
    /// bitmap is monotone: once set, `cache[i]`'s `by_piece` is the position its
    /// `acc`/`psqt` were computed from, which is the whole cache invariant.
    cache_valid: [u64; 2],
}

impl<'a> HalfkaContext<'a> {
    pub(crate) fn new(weights: &'a [i8], bias: &'a [i16], psqt_weights: &'a [i32]) -> Self {
        #[cfg(feature = "foldcount")]
        {
            use super::foldcount::region;
            // `size_of_val` reports the i8 table, so study 29's census sees
            // the HALVED region (28.9 MB, not 57.7) — which is the point of
            // study 32 and exactly what the census should attribute.
            region(
                "hka_weights",
                weights.as_ptr().cast(),
                std::mem::size_of_val(weights),
            );
            region(
                "hka_psqt",
                psqt_weights.as_ptr().cast(),
                std::mem::size_of_val(psqt_weights),
            );
        }
        #[allow(unused_mut)]
        let mut me = Self {
            weights,
            bias,
            psqt_weights,
            // `vec![…].into_boxed_slice().try_into()` is the allocation-free
            // route from a heap slice to a heap ARRAY — the conversion is a
            // pointer cast, the length check is the `TryFrom` impl's, and it
            // cannot fail because the `vec!` literally states `STACK_LEN`.
            // (Boxing the array directly would build it on the STACK first:
            // 1.3 MB of `Entry` would overflow the thread stack.)
            stack: boxed_array(Entry::default(), STACK_LEN),
            meta: boxed_array(Meta::default(), STACK_LEN),
            cache: boxed_array(FinnyEntry::default(), 2 * FINNY_KEYS),
            cache_valid: [0; 2],
        };
        #[cfg(feature = "foldcount")]
        {
            use super::foldcount::region;
            region(
                "hka_stack",
                std::ptr::from_ref(&*me.stack).cast(),
                std::mem::size_of_val(&*me.stack),
            );
            region(
                "finny_cache",
                std::ptr::from_ref(&*me.cache).cast(),
                std::mem::size_of_val(&*me.cache),
            );
        }
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
        if self.meta[ply].computed[p] {
            return;
        }
        let mut ancestor = ply;
        while ancestor > 0 && !self.meta[ancestor].computed[p] {
            ancestor -= 1;
        }
        let king = Piece::new(perspective, crate::types::PieceType::King);
        // The LAST king move in the span, if any. Every ply from there to `ply`
        // shares the post-move king square, so all of them are reachable from
        // the refreshed leaf by plain reverse folds.
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
                let dirty = self.meta[child + 1].dirty;
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
                self.meta[child].computed[p] = true;
            }
            return;
        }
        let king_square = board.king_square(perspective);
        let weights = self.weights;
        let psqt_weights = self.psqt_weights;
        for child in (ancestor + 1)..=ply {
            let dirty = self.meta[child].dirty;
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
            // `apply_halfka_deltas_into(dst, src, w, &[], &[])` is `dst = src`
            // lane for lane, and `psqt_fold` with empty lists is `*dst = *src`.
            // So this deletion cannot be WRONG, only slower.
            //
            // The corollary is a real future hazard, and it is the reason this
            // paragraph exists: **the empty-list case of the delta kernels
            // must remain a copy.** Adding an `if adds.is_empty() &&
            // subs.is_empty() { return; }` fast path to `apply_halfka_deltas_into`
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
            self.meta[child].computed[p] = true;
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
            cache,
            cache_valid,
            stack,
            bias,
            ..
        } = self;
        let p = perspective as usize;
        #[cfg(feature = "foldcount")]
        super::foldcount::note_finny_key(p, key);
        let cached = &mut cache[p * FINNY_KEYS + key];
        let valid_bit = 1u64 << key;
        if cache_valid[p] & valid_bit != 0 {
            #[cfg(feature = "pf43")]
            pf43::slot(&cached.acc);
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
            //
            // PERF STUDY 22 ARM 3 — THE FIRST OF THOSE THREE WAS NOT ACTUALLY
            // BEING DONE, and had not been for as long as the comment has
            // existed. The loop was `for pt in PieceType::ALL`, and the shipped
            // binary contained the exact anti-pattern forbidden above, opcode
            // for opcode:
            //
            //     mov   WORD  PTR [rsp+0x154],0x504      ; PieceType::ALL, on
            //     mov   DWORD PTR [rsp+0x150],0x3020100  ;   the STACK, per call
            //     movzx r8d,BYTE PTR [rsp+rdi*1+0x150]   ; reloaded per iteration
            //     mov   r9, QWORD PTR [r12+rdx*1+0x140]  ; board.pieces[c][pt]
            //     mov   r11,QWORD PTR [r14+rdx*1+0xa60]  ; cached.by_piece[c][pt]
            //
            // Iterating an array BY VALUE hands LLVM an `alloca` it declined to
            // promote to `.rodata`, so a 6-byte constant was written to the
            // stack on every call and read back once per iteration.
            //
            // THIS IS A MEASURED NULL, and it is landed as documentation
            // repair, not as a speedup: +0.8 i/n on the 5,001,562-node frame,
            // which is inside the ~5 i/n practical floor. The eight
            // instructions per call it removes are worth ~1.3 i/n at 0.16
            // refreshes/node. It is here because a comment that forbids what
            // the code does is worse than no comment — it cost this study time
            // twice (perf study 22 §1.3, §5.3, both the same failure mode).
            const CAP: usize = 32;
            static PIECE_TYPES: [PieceType; 6] = PieceType::ALL;
            let mut adds = [0usize; CAP];
            let mut add_len = 0;
            let mut subs = [0usize; CAP];
            let mut sub_len = 0;
            for color in [Color::White, Color::Black] {
                // `board.pieces` is the `pub(crate)` FIELD, not the accessor:
                // the accessor hands back one bitboard and would put the
                // indexed load straight back. The row is the unit the diff
                // walks. `PIECE_TYPES` is a `static`, so the piece type comes
                // from `.rodata` at a fixed address rather than a stack copy.
                let new_row = &board.pieces[color as usize];
                let old_row = &mut cached.by_piece[color as usize];
                for ((&new_bb, old_bb), &pt) in new_row.iter().zip(old_row).zip(&PIECE_TYPES) {
                    if new_bb == *old_bb {
                        continue;
                    }
                    let piece = Piece::new(color, pt);
                    let base = halfka_row_base(perspective, piece, king);
                    // PERF STUDY 29 REJECTED A SOFTWARE PREFETCH HERE, and it
                    // is worth knowing why before adding one back. This loop
                    // resolves up to 32 indices before `apply_halfka_deltas_dual`
                    // folds any of them — the one site in the engine with a
                    // real gap between "the row address is known" and "the row
                    // is read" — and it is the third-largest load-stall symbol
                    // (13.74%), 81.2% of whose loads are HalfKA weight rows of
                    // which only 7.2% hit L1. Hinting each row at index time,
                    // Stockfish's `append_changed_indices` shape, measured
                    // **not resolvable**: -0.16% misses against a null of
                    // +0.14%, and in the drift-cancelling paired form the arm
                    // and a byte-different null are indistinguishable. See
                    // that study's §5 for the numbers and the ceiling argument
                    // (one hint can convert at most one of the ~22 demand
                    // misses a 40-line row takes).
                    for sq in new_bb & !*old_bb {
                        debug_assert!(add_len < CAP, "finny add cursor overflow");
                        adds[add_len & (CAP - 1)] = base + halfka_oriented(perspective, sq, king);
                        add_len += 1;
                    }
                    for sq in *old_bb & !new_bb {
                        debug_assert!(sub_len < CAP, "finny sub cursor overflow");
                        subs[sub_len & (CAP - 1)] = base + halfka_oriented(perspective, sq, king);
                        sub_len += 1;
                    }
                    *old_bb = new_bb;
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
            stack[ply].psqt[p] = cached.psqt;
            return;
        }
        // COLD path only (this key has never been built): one from-scratch
        // build, then the single copy into the stack. Leaving it as a copy is
        // deliberate — the cold branch runs once per king key per search, so a
        // second kernel variant here would buy nothing. The slot is written in
        // place (it is a real `FinnyEntry`, not an `Option` to be filled), so
        // the 2.5 KB stack temp the old form round-tripped through is gone too.
        cached.acc.copy_from_slice(&bias[..ACC]);
        let mut adds = [0usize; 32];
        let mut len = 0usize;
        cached.by_piece = [[Bitboard::EMPTY; 6]; 2];
        for square in board.all_occupancy() {
            let piece = board.piece_on(square).expect("occupied square");
            adds[len] = halfka_index(perspective, piece, square, king);
            len += 1;
            cached.by_piece[piece.color as usize][piece.piece_type as usize] |=
                Bitboard::from_square(square);
        }
        apply_halfka_deltas(&mut cached.acc, weights, &adds[..len], &[]);
        // T6c: seed the PSQT sums from zero (the skip has no bias term).
        cached.psqt = [0i32; BUCKETS];
        if !psqt_weights.is_empty() {
            let mut psqt = [0i32; BUCKETS];
            for &f in &adds[..len] {
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
/// of "copy parent then [`apply_dirty`] in place" (profiling-study opt #2).
/// The resolved indices drive both the accumulator kernel and (when the net
/// carries one) the T6c PSQT bucket-sum fold, so the two channels can never
/// see different feature events.
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
/// plies BETWEEN a king move and the refreshed leaf (perf study 12 §6, after
/// `AccumulatorStack::backward_update_incremental` at `eca43a97`
/// `src/nnue/nnue_accumulator.cpp`). `king` is the CURRENT king square in both
/// directions, which is sound precisely because the backward span starts at the
/// king move: no ply inside it moves the king again, so every HalfKA index in
/// the span is bucketed on the same square.
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
    apply_halfka_deltas_into(&mut entry.acc[p], &parent.acc[p], weights, a, s);
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

    fn weights() -> Vec<i8> {
        (0..HALFKA_DIMS * ACC)
            .map(|i| ((i.wrapping_mul(17) % 251) as i16 - 125) as i8)
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
