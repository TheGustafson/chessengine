//! Incremental perspective accumulators with lazy, dirty-piece-driven updates
//! (design/11 §3.4.4).
//!
//! Each search node owns an [`AccumulatorEntry`] holding both perspectives'
//! `L1`-wide feature-transformer outputs, the placement deltas of the move
//! that led to it ([`DirtyPiece`]), and a per-perspective `computed` flag.
//! Accumulators update *lazily* — most made moves are pruned before ever
//! being evaluated, so the columns are folded forward only when
//! [`NnueContext::evaluate`] actually needs them (SF's approach).
//!
//! **The invalidation rule is the correctness linchpin.** `stack[0]` is
//! refreshed from scratch and always trusted; every move made from ply `k`
//! records its deltas into `stack[k+1]` and marks it *uncomputed*, so a sibling
//! subtree can never reuse a stale accumulator. At evaluate time we walk back
//! to the nearest computed ancestor and fold each dirty record forward. The
//! [`NnueContext::assert_matches_refresh`] oracle (the NNUE analogue of
//! `Board::assert_consistent`) checks that the incrementally-maintained
//! columns equal a from-scratch refresh at every computed node.

use super::L1;
use super::network::{Network, feature_index};
use super::simd::{Aligned64, KERNELS};
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Value};

/// Accumulator-stack length: one entry per search ply (root = 0) plus margin.
/// Indexed by SEARCH ply, never by `board.states.len()` (which includes
/// UCI-replayed game history and can exceed `MAX_PLY`) — design/11 §3.4.4.
pub const NNUE_STACK_LEN: usize = MAX_PLY + 8;

/// One node's accumulators (design/11 §3.4.4).
#[derive(Debug, Clone)]
pub struct AccumulatorEntry {
    /// Feature-transformer output per perspective, indexed by [`Color`].
    /// 64-byte aligned for the step-2 AVX2 loads.
    pub acc: Aligned64<[[i16; L1]; 2]>,
    /// The placement deltas of the move that LED TO this entry.
    pub dirty: DirtyPiece,
    /// Whether `acc[perspective]` is valid, per perspective.
    pub computed: [bool; 2],
}

impl Default for AccumulatorEntry {
    fn default() -> Self {
        AccumulatorEntry {
            acc: Aligned64([[0i16; L1]; 2]),
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}

/// A per-thread accumulator stack over a borrowed network (design/11 §3.4.4).
///
/// The design's end state stores `net: &'static Network` (the loaded
/// `EvalFile`); step 1 borrows for a lifetime `'a` so tests can drive a
/// locally-built net without leaking a `'static`. It is NOT yet placed in the
/// search `Worker`'s `EvalContext` — that wiring is step 4.
pub struct NnueContext<'a> {
    stack: Box<[AccumulatorEntry]>,
    net: &'a Network,
}

impl<'a> NnueContext<'a> {
    /// A fresh context over `net`, sized for `MAX_PLY` search. Heap-allocates
    /// the stack once (this is the setup path, not the hot path).
    pub fn new(net: &'a Network) -> Self {
        NnueContext {
            stack: vec![AccumulatorEntry::default(); NNUE_STACK_LEN].into_boxed_slice(),
            net,
        }
    }

    /// The net this context evaluates against.
    pub fn net(&self) -> &Network {
        self.net
    }

    /// Refresh the root entry (`stack[0]`) from scratch for `board`, both
    /// perspectives, and mark it computed. Called at the start of every `go`
    /// so the root is valid regardless of how much game history UCI replayed
    /// (design/11 §3.4.4).
    pub fn refresh_root(&mut self, board: &Board) {
        for persp in Color::ALL {
            let dst = &mut self.stack[0].acc.0[persp as usize];
            refresh_into(self.net, board, persp, dst);
        }
        self.stack[0].dirty = DirtyPiece::EMPTY;
        self.stack[0].computed = [true, true];
    }

    /// Record a move made from search ply `ply`: its deltas land in the child
    /// entry `stack[ply + 1]`, which is marked uncomputed so no sibling
    /// subtree reuses a stale accumulator (the §3.4.4 invalidation rule). Call
    /// this AFTER deriving `dp` from the pre-move board and BEFORE the child
    /// node evaluates.
    pub fn push(&mut self, ply: usize, dp: DirtyPiece) {
        let child = &mut self.stack[ply + 1];
        child.dirty = dp;
        child.computed = [false, false];
    }

    /// Evaluate `board` at search ply `ply` through the NNUE forward pass,
    /// folding the accumulators forward lazily as needed. `board` must be the
    /// position at `ply` (its side to move selects the stm half).
    pub fn evaluate(&mut self, board: &Board, ply: usize) -> Value {
        let stm = board.side_to_move();
        self.compute(ply, stm);
        self.compute(ply, stm.flip());
        let us = &self.stack[ply].acc.0[stm as usize];
        let them = &self.stack[ply].acc.0[stm.flip() as usize];
        (KERNELS.forward)(
            us,
            them,
            self.net.out_weights_arr(),
            self.net.out_bias,
            self.net.activation,
        )
    }

    /// Ensure `stack[ply].acc[perspective]` is valid: if already computed,
    /// done; otherwise walk back to the nearest computed ancestor and fold
    /// each intervening dirty record forward (design/11 §3.4.4). `stack[0]` is
    /// always computed, so the walk terminates.
    fn compute(&mut self, ply: usize, perspective: Color) {
        let p = perspective as usize;
        if self.stack[ply].computed[p] {
            return;
        }
        let mut k = ply;
        while !self.stack[k].computed[p] {
            debug_assert!(k > 0, "stack[0] must always be computed (design/11 §3.4.4)");
            k -= 1;
        }
        let net = self.net;
        for i in (k + 1)..=ply {
            let src = self.stack[i - 1].acc.0[p]; // copy: ends the shared borrow
            let dp = self.stack[i].dirty;
            let mut dst = [0i16; L1];
            apply_dirty(net, &mut dst, &src, &dp, perspective);
            self.stack[i].acc.0[p] = dst;
            self.stack[i].computed[p] = true;
        }
    }

    /// The NNUE analogue of `Board::assert_consistent` (design/11 §3.4.4): the
    /// incrementally-maintained accumulator at `ply` must equal a from-scratch
    /// refresh of `board` for both perspectives. Drives the incremental ==
    /// refresh fuzz gate. Compiled only in test/debug builds.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_matches_refresh(&mut self, board: &Board, ply: usize) {
        for persp in Color::ALL {
            self.compute(ply, persp);
            let mut fresh = [0i16; L1];
            refresh_into(self.net, board, persp, &mut fresh);
            assert!(
                self.stack[ply].acc.0[persp as usize] == fresh,
                "incremental accumulator != from-scratch refresh at ply {ply}, {persp:?}"
            );
        }
    }
}

/// From-scratch accumulator for one perspective: `bias + Σ_active W0[feature]`
/// (design/11 §3.4.4). Resolves each occupied square's feature row and hands
/// the batch to the refresh kernel. No heap: at most 64 pieces, so a
/// fixed-size row buffer suffices.
fn refresh_into(net: &Network, board: &Board, perspective: Color, dst: &mut [i16; L1]) {
    let placeholder = net.ft_row(0);
    let mut rows: [&[i16; L1]; 64] = [placeholder; 64];
    let mut n = 0;
    for sq in board.all_occupancy() {
        let piece = board
            .piece_on(sq)
            .expect("occupied square has a piece (mailbox/bitboard invariant)");
        rows[n] = net.ft_row(feature_index(perspective, piece, sq));
        n += 1;
    }
    (KERNELS.refresh)(dst, net.ft_bias_arr(), &rows[..n]);
}

/// Fold one move's deltas into `dst` from base `src` for `perspective`
/// (design/11 §3.4.4): `dst = src + Σ W0[add] − Σ W0[sub]`. The dominant
/// 1-add/1-sub case (a quiet move) uses the fused kernel in one pass.
fn apply_dirty(
    net: &Network,
    dst: &mut [i16; L1],
    src: &[i16; L1],
    dp: &DirtyPiece,
    perspective: Color,
) {
    let adds = dp.adds();
    let subs = dp.subs();
    if adds.len() == 1 && subs.len() == 1 {
        let add_row = net.ft_row(feature_index(perspective, adds[0].0, adds[0].1));
        let sub_row = net.ft_row(feature_index(perspective, subs[0].0, subs[0].1));
        (KERNELS.add_sub)(dst, src, add_row, sub_row);
        return;
    }
    // General path (captures, promotions, castling, null): start from `src`,
    // then apply each add/sub column-wise.
    *dst = *src;
    for &(piece, sq) in adds {
        let row = net.ft_row(feature_index(perspective, piece, sq));
        for i in 0..L1 {
            dst[i] = dst[i].wrapping_add(row[i]);
        }
    }
    for &(piece, sq) in subs {
        let row = net.ft_row(feature_index(perspective, piece, sq));
        for i in 0..L1 {
            dst[i] = dst[i].wrapping_sub(row[i]);
        }
    }
}

/// From-scratch NNUE forward-pass evaluation of `board` (design/11 §3.4.3):
/// refresh both perspectives and run the output layer, stm half first. This is
/// the forward-pass oracle the `nnue-eval` dev subcommand and the symmetry
/// test call — no incremental state, no in-check precondition (unlike the
/// eval seam, this is the raw net value). Returns the raw integer centipawn
/// value.
pub fn evaluate_forward(net: &Network, board: &Board) -> Value {
    let mut acc = [[0i16; L1]; 2];
    for persp in Color::ALL {
        refresh_into(net, board, persp, &mut acc[persp as usize]);
    }
    let stm = board.side_to_move();
    (KERNELS.forward)(
        &acc[stm as usize],
        &acc[stm.flip() as usize],
        net.out_weights_arr(),
        net.out_bias,
        net.activation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::network::Activation;
    use crate::fen::START_FEN;
    use crate::moves::Move;
    use crate::types::{Piece, PieceType, Square};

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    // A one-move incremental update must match a from-scratch refresh of the
    // resulting position — the smallest slice of the §3.4.4 invariant.
    #[test]
    fn one_move_incremental_equals_refresh() {
        let net = Network::deterministic_test_net(42, Activation::SCReLU);
        let mut b = board(START_FEN);
        let mut ctx = NnueContext::new(&net);
        ctx.refresh_root(&b);

        let m = Move::normal(Square::E2, Square::E4);
        let dp = DirtyPiece::from_move(&b, m);
        b.make_move(m, false);
        ctx.push(0, dp);
        ctx.assert_matches_refresh(&b, 1);

        // And its forward value equals the standalone from-scratch forward.
        assert_eq!(ctx.evaluate(&b, 1), evaluate_forward(&net, &b));
    }

    // DirtyPiece::from_move for every move kind, checked against the pieces
    // that actually move (design/11 §3.4.4).
    #[test]
    fn dirty_piece_covers_every_move_kind() {
        // Quiet: 1 sub + 1 add (the knight).
        let b = board(START_FEN);
        let dp = DirtyPiece::from_move(&b, Move::normal(Square::G1, Square::F3));
        let wn = Piece::new(Color::White, PieceType::Knight);
        assert_eq!(dp.subs(), &[(wn, Square::G1)]);
        assert_eq!(dp.adds(), &[(wn, Square::F3)]);

        // Ordinary capture: 2 subs (mover + victim) + 1 add.
        let b = board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let dp = DirtyPiece::from_move(&b, Move::normal(Square::E4, Square::D5));
        let wp = Piece::new(Color::White, PieceType::Pawn);
        let bp = Piece::new(Color::Black, PieceType::Pawn);
        assert_eq!(dp.subs(), &[(wp, Square::E4), (bp, Square::D5)]);
        assert_eq!(dp.adds(), &[(wp, Square::D5)]);

        // En passant: mover pawn + captured pawn (on e5) subbed, mover added.
        let b = board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        let dp = DirtyPiece::from_move(&b, Move::en_passant(Square::E5, Square::F6));
        assert_eq!(dp.subs(), &[(wp, Square::E5), (bp, Square::F5)]);
        assert_eq!(dp.adds(), &[(wp, Square::F6)]);

        // Quiet promotion: sub pawn, add queen.
        let b = board("8/P6k/8/8/8/8/7K/8 w - - 0 1");
        let dp = DirtyPiece::from_move(
            &b,
            Move::promotion(Square::A7, Square::A8, PieceType::Queen),
        );
        let wq = Piece::new(Color::White, PieceType::Queen);
        assert_eq!(dp.subs(), &[(wp, Square::A7)]);
        assert_eq!(dp.adds(), &[(wq, Square::A8)]);

        // Capture-promotion: sub pawn + victim, add knight.
        let b = board("1n5k/P7/8/8/8/8/7K/8 w - - 0 1");
        let dp = DirtyPiece::from_move(
            &b,
            Move::promotion(Square::A7, Square::B8, PieceType::Knight),
        );
        let wn = Piece::new(Color::White, PieceType::Knight);
        let bn = Piece::new(Color::Black, PieceType::Knight);
        assert_eq!(dp.subs(), &[(wp, Square::A7), (bn, Square::B8)]);
        assert_eq!(dp.adds(), &[(wn, Square::B8)]);

        // Castling (kingside): 2 subs (king + rook) + 2 adds.
        let b = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let dp = DirtyPiece::from_move(&b, Move::castling(Square::E1, Square::G1));
        let wk = Piece::new(Color::White, PieceType::King);
        let wr = Piece::new(Color::White, PieceType::Rook);
        assert_eq!(dp.subs(), &[(wk, Square::E1), (wr, Square::H1)]);
        assert_eq!(dp.adds(), &[(wk, Square::G1), (wr, Square::F1)]);

        // Null / same-square sentinel: empty record.
        let dp = DirtyPiece::from_move(&b, Move::NULL);
        assert!(dp.adds().is_empty() && dp.subs().is_empty());
    }

    // The §3.4.2 symmetry guarantee at the forward-pass level: a color-flipped
    // position evaluates EQUAL (both are stm-relative). Guards the feature
    // indexer + perspective construction.
    #[test]
    fn forward_is_color_flip_symmetric() {
        let net = Network::deterministic_test_net(7, Activation::SCReLU);
        // A handful of asymmetric positions and their manual color-flips.
        let pairs = [
            (
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3",
            ),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "8/4p1p1/8/1r3P1K/kp5R/3P4/2P5/8 b - - 0 1",
            ),
        ];
        for (a, b) in pairs {
            assert_eq!(
                evaluate_forward(&net, &board(a)),
                evaluate_forward(&net, &board(b)),
                "color-flip symmetry broken:\n  {a}\n  {b}"
            );
        }
    }
}
