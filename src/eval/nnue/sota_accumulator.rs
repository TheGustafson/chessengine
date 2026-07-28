//! Incremental HalfKAv2_hm accumulator with a per-king-state Finny cache.
//!
//! T6c FT PSQT skip: alongside every `[i16; ACC]` accumulator lane set, this
//! module maintains the per-perspective `[i32; BUCKETS]` PSQT bucket sums —
//! the same add/sub feature events applied to the net's 8-wide PSQ1 rows
//! (`sota_loader` module docs). The Finny cache stores/restores the sums with
//! the same keys as the accumulator, so a cached refresh cannot desync the
//! skip channel. A net without a PSQ1 table (`psqt_weights` empty) skips
//! every psqt update and the sums stay zero — bit-identical to pre-T6c.

use super::sota_features::{ACC, halfka_index, halfka_king_key};
use super::sota_head::BUCKETS;
use super::sota_simd::{apply_i16_deltas, apply_i16_deltas_into};
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
#[inline]
fn psqt_row(table: &[i32], feature: usize) -> &[i32; BUCKETS] {
    table[feature * BUCKETS..][..BUCKETS]
        .try_into()
        .expect("psqt rows are BUCKETS wide by construction")
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
        // Null-move alias (perf study #4 item #3, mirrored from the threat
        // stack): an EMPTY dirty record's "fold" is a pure 2×2.5 KB parent
        // copy — hand back the parent's accumulator instead. Sound because a
        // null move changes no placement (identical features, identical king
        // squares), so the parent entry IS this ply's accumulator. Only
        // fully-unfolded EMPTY entries alias; partially-computed ones (a
        // one-sided span fold passed through) are materialized and used as-is.
        let mut target = ply;
        while target > 0
            && !self.stack[target].computed[0]
            && !self.stack[target].computed[1]
            && self.stack[target].dirty.adds().is_empty()
            && self.stack[target].dirty.subs().is_empty()
        {
            target -= 1;
        }
        for perspective in Color::ALL {
            self.compute(board, target, perspective);
        }
        let entry = &self.stack[target];
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
        if ((ancestor + 1)..=ply).any(|child| {
            self.stack[child]
                .dirty
                .subs()
                .iter()
                .any(|&(piece, _)| piece == king)
        }) {
            self.finny_refresh_into(board, perspective, ply);
            self.stack[ply].computed[p] = true;
            return;
        }
        let king_square = board.king_square(perspective);
        let weights = self.weights;
        let psqt_weights = self.psqt_weights;
        for child in (ancestor + 1)..=ply {
            let dirty = self.stack[child].dirty;
            // Fused parent→child fold: read parent lanes, apply deltas, write
            // child lanes in one pass — no separate ~1.5 KB memmove (opt #2).
            // An EMPTY record inside a span (a null move on the fold path)
            // degenerates to the plain parent copy without the kernel call.
            let (before, after) = self.stack.split_at_mut(child);
            let parent = &before[child - 1];
            let entry = &mut after[0];
            if dirty.adds().is_empty() && dirty.subs().is_empty() {
                entry.acc[p] = parent.acc[p];
                entry.psqt[p] = parent.psqt[p];
            } else {
                apply_dirty_into(
                    weights,
                    psqt_weights,
                    entry,
                    parent,
                    &dirty,
                    perspective,
                    king_square,
                );
            }
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
        let slot = &mut self.cache[perspective as usize * FINNY_KEYS + key];
        if let Some(cached) = slot.as_mut() {
            // Option A (perf study #2): bitboard set-diff against the cached
            // position. Same add/sub SETS as the old sorted-list derivation
            // (order may differ; wrapping i16/i32 adds commute => the
            // accumulator and PSQT sums stay bit-identical).
            let mut adds = [0usize; 32];
            let mut add_len = 0;
            let mut subs = [0usize; 32];
            let mut sub_len = 0;
            for color in [Color::White, Color::Black] {
                for pt in PieceType::ALL {
                    let new_bb = board.pieces(color, pt);
                    let old_bb = cached.by_piece[color as usize][pt as usize];
                    if new_bb == old_bb {
                        continue;
                    }
                    let piece = Piece::new(color, pt);
                    for sq in new_bb & !old_bb {
                        adds[add_len] = halfka_index(perspective, piece, sq, king);
                        add_len += 1;
                    }
                    for sq in old_bb & !new_bb {
                        subs[sub_len] = halfka_index(perspective, piece, sq, king);
                        sub_len += 1;
                    }
                    cached.by_piece[color as usize][pt as usize] = new_bb;
                }
            }
            apply_i16_deltas(&mut cached.acc, weights, &adds[..add_len], &subs[..sub_len]);
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
        } else {
            let mut acc = [0i16; ACC];
            acc.copy_from_slice(&self.bias[..ACC]);
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
        let cached = slot.as_ref().expect("slot filled above");
        self.stack[ply].acc[perspective as usize] = cached.acc;
        self.stack[ply].psqt[perspective as usize] = cached.psqt;
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
    let p = perspective as usize;
    let mut adds = [0usize; 2];
    let mut subs = [0usize; 2];
    for (i, &(piece, square)) in dirty.adds().iter().enumerate() {
        adds[i] = halfka_index(perspective, piece, square, king);
    }
    for (i, &(piece, square)) in dirty.subs().iter().enumerate() {
        subs[i] = halfka_index(perspective, piece, square, king);
    }
    apply_i16_deltas_into(
        &mut entry.acc[p],
        &parent.acc[p],
        weights,
        &adds[..dirty.adds().len()],
        &subs[..dirty.subs().len()],
    );
    if !psqt_weights.is_empty() {
        psqt_fold(
            &mut entry.psqt[p],
            &parent.psqt[p],
            psqt_weights,
            &adds[..dirty.adds().len()],
            &subs[..dirty.subs().len()],
        );
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
