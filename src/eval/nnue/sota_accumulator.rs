//! Incremental HalfKAv2_hm accumulator with a per-king-state Finny cache.

use super::sota_features::{ACC, halfka_index, halfka_king_key};
use super::sota_simd::{apply_i16_deltas, apply_i16_deltas_into};
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, Square};

const STACK_LEN: usize = MAX_PLY + 8;
const FINNY_KEYS: usize = 64;

#[derive(Clone)]
struct Entry {
    acc: [[i16; ACC]; 2],
    dirty: DirtyPiece,
    computed: [bool; 2],
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            acc: [[0; ACC]; 2],
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}

#[derive(Clone)]
struct FinnyEntry {
    acc: [i16; ACC],
    features: [u16; 32],
    len: u8,
}

pub(crate) struct HalfkaContext<'a> {
    weights: &'a [i16],
    bias: &'a [i16],
    stack: Box<[Entry]>,
    cache: Box<[Option<FinnyEntry>]>,
}

impl<'a> HalfkaContext<'a> {
    pub(crate) fn new(weights: &'a [i16], bias: &'a [i16]) -> Self {
        Self {
            weights,
            bias,
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

    pub(crate) fn current(&mut self, board: &Board, ply: usize) -> &[[i16; ACC]; 2] {
        for perspective in Color::ALL {
            self.compute(board, ply, perspective);
        }
        &self.stack[ply].acc
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
        for child in (ancestor + 1)..=ply {
            let dirty = self.stack[child].dirty;
            // Fused parent→child fold: read parent lanes, apply deltas, write
            // child lanes in one pass — no separate ~1.5 KB memmove (opt #2).
            let (before, after) = self.stack.split_at_mut(child);
            apply_dirty_into(
                weights,
                &mut after[0].acc[p],
                &before[child - 1].acc[p],
                &dirty,
                perspective,
                king_square,
            );
            after[0].computed[p] = true;
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
        let mut features = [0u16; 32];
        let mut len = 0usize;
        for square in board.all_occupancy() {
            let piece = board.piece_on(square).expect("occupied square");
            features[len] = halfka_index(perspective, piece, square, king) as u16;
            len += 1;
        }
        features[..len].sort_unstable();

        let weights = self.weights;
        let slot = &mut self.cache[perspective as usize * FINNY_KEYS + key];
        if let Some(cached) = slot.as_mut() {
            let mut adds = [0usize; 32];
            let mut add_len = 0;
            let mut subs = [0usize; 32];
            let mut sub_len = 0;
            for &old in &cached.features[..cached.len as usize] {
                if features[..len].binary_search(&old).is_err() {
                    subs[sub_len] = old as usize;
                    sub_len += 1;
                }
            }
            for &new in &features[..len] {
                if cached.features[..cached.len as usize]
                    .binary_search(&new)
                    .is_err()
                {
                    adds[add_len] = new as usize;
                    add_len += 1;
                }
            }
            apply_i16_deltas(&mut cached.acc, weights, &adds[..add_len], &subs[..sub_len]);
            // Keep the cache's feature list bit-identical even if an exotic
            // illegal position somehow contains duplicate feature ids.
            cached.features = features;
            cached.len = len as u8;
        } else {
            let mut acc = [0i16; ACC];
            acc.copy_from_slice(&self.bias[..ACC]);
            let mut adds = [0usize; 32];
            for (dst, &feature) in adds.iter_mut().zip(&features[..len]) {
                *dst = feature as usize;
            }
            apply_i16_deltas(&mut acc, weights, &adds[..len], &[]);
            *slot = Some(FinnyEntry {
                acc,
                features,
                len: len as u8,
            });
        }
        let cached = slot.as_ref().expect("slot filled above");
        self.stack[ply].acc[perspective as usize] = cached.acc;
    }
}

/// Fused fold: `dst = src ± dirty deltas` in one pass — the memmove-free form
/// of "copy parent then [`apply_dirty`] in place" (profiling-study opt #2).
fn apply_dirty_into(
    weights: &[i16],
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    dirty: &DirtyPiece,
    perspective: Color,
    king: Square,
) {
    let mut adds = [0usize; 2];
    let mut subs = [0usize; 2];
    for (i, &(piece, square)) in dirty.adds().iter().enumerate() {
        adds[i] = halfka_index(perspective, piece, square, king);
    }
    for (i, &(piece, square)) in dirty.subs().iter().enumerate() {
        subs[i] = halfka_index(perspective, piece, square, king);
    }
    apply_i16_deltas_into(
        dst,
        src,
        weights,
        &adds[..dirty.adds().len()],
        &subs[..dirty.subs().len()],
    );
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

    #[test]
    fn incremental_and_finny_match_refresh() {
        let weights = weights();
        let bias: Vec<i16> = (0..ACC).map(|i| (i % 19) as i16 - 9).collect();
        let mut context = HalfkaContext::new(&weights, &bias);
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
            let got = *context.current(&board, ply + 1);
            for perspective in Color::ALL {
                let fresh = refresh_pst(&weights, &bias, &board, perspective);
                assert_eq!(got[perspective as usize], fresh, "ply {ply} {m}");
            }
        }
    }
}
