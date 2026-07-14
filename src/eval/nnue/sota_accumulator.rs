//! Incremental HalfKAv2_hm accumulator with a per-king-state Finny cache.

use super::sota_features::{ACC, halfka_index, halfka_king_key};
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
            let acc = self.finny_refresh(board, perspective);
            self.stack[0].acc[perspective as usize] = acc;
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
            self.stack[ply].acc[p] = self.finny_refresh(board, perspective);
            self.stack[ply].computed[p] = true;
            return;
        }
        let king_square = board.king_square(perspective);
        for child in (ancestor + 1)..=ply {
            let dirty = self.stack[child].dirty;
            let (before, after) = self.stack.split_at_mut(child);
            let mut acc = before[child - 1].acc[p];
            apply_dirty(self.weights, &mut acc, &dirty, perspective, king_square);
            after[0].acc[p] = acc;
            self.stack[child].computed[p] = true;
        }
    }

    fn finny_refresh(&mut self, board: &Board, perspective: Color) -> [i16; ACC] {
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

        let slot = &mut self.cache[perspective as usize * FINNY_KEYS + key];
        let acc = if let Some(cached) = slot.as_ref() {
            let mut acc = cached.acc;
            for &old in &cached.features[..cached.len as usize] {
                if features[..len].binary_search(&old).is_err() {
                    add_column(self.weights, &mut acc, old as usize, -1);
                }
            }
            for &new in &features[..len] {
                if cached.features[..cached.len as usize]
                    .binary_search(&new)
                    .is_err()
                {
                    add_column(self.weights, &mut acc, new as usize, 1);
                }
            }
            acc
        } else {
            let mut acc = [0i16; ACC];
            acc.copy_from_slice(&self.bias[..ACC]);
            for &feature in &features[..len] {
                add_column(self.weights, &mut acc, feature as usize, 1);
            }
            acc
        };
        // Keep the cache's accumulator bit-identical even if an exotic illegal
        // position somehow contains duplicate feature ids.
        let entry = FinnyEntry {
            acc,
            features,
            len: len as u8,
        };
        *slot = Some(entry);
        acc
    }
}

fn apply_dirty(
    weights: &[i16],
    acc: &mut [i16; ACC],
    dirty: &DirtyPiece,
    perspective: Color,
    king: Square,
) {
    for &(piece, square) in dirty.subs() {
        add_column(
            weights,
            acc,
            halfka_index(perspective, piece, square, king),
            -1,
        );
    }
    for &(piece, square) in dirty.adds() {
        add_column(
            weights,
            acc,
            halfka_index(perspective, piece, square, king),
            1,
        );
    }
}

fn add_column(weights: &[i16], acc: &mut [i16; ACC], feature: usize, sign: i16) {
    let column = &weights[feature * ACC..(feature + 1) * ACC];
    for (dst, &weight) in acc.iter_mut().zip(column) {
        *dst = dst.wrapping_add(weight.wrapping_mul(sign));
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
