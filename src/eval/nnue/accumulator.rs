//! Incremental perspective accumulators with lazy, dirty-piece-driven updates
//! (design/11 §3.4.4). Each search node owns an [`AccumulatorEntry`]; updates
//! are *lazy* — most made moves are pruned before ever being evaluated
//! (SF's approach).
//!
//! **The invalidation rule is the correctness linchpin.** `stack[0]` is
//! refreshed from scratch and always trusted; every move made from ply `k`
//! records its deltas into `stack[k+1]` and marks it *uncomputed*, so a
//! sibling subtree can never reuse a stale accumulator. At evaluate time we
//! walk back to the nearest computed ancestor and fold forward. The
//! [`NnueContext::assert_matches_refresh`] oracle pins incremental ==
//! from-scratch at every computed node.

use super::L1;
use super::network::{Network, feature_index};
use super::simd::{Aligned64, KERNELS};
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, Square, Value};

/// Accumulator-stack length: one entry per search ply (root = 0) plus margin.
/// Indexed by SEARCH ply, never by `board.history.len()` (which includes
/// UCI-replayed game history and can exceed `MAX_PLY`) — design/11 §3.4.4.
pub const NNUE_STACK_LEN: usize = MAX_PLY + 8;

/// One node's accumulators (design/11 §3.4.4).
#[derive(Debug, Clone)]
pub struct AccumulatorEntry {
    /// Feature-transformer output per perspective, indexed by [`Color`];
    /// 64-byte aligned for the SIMD loads.
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
/// Borrows for `'a` (not `&'static`) so tests can drive a locally-built net
/// without leaking.
#[derive(Debug)]
pub struct NnueContext<'a> {
    stack: Box<[AccumulatorEntry]>,
    net: &'a Network,
}

impl<'a> NnueContext<'a> {
    /// A fresh context over `net`; heap-allocates the stack once (setup path).
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

    /// Refresh the root entry (`stack[0]`) from scratch, both perspectives,
    /// and mark it computed. Called at the start of every `go` so the root is
    /// valid regardless of how much game history UCI replayed.
    pub fn refresh_root(&mut self, board: &Board) {
        for persp in Color::ALL {
            let dst = &mut self.stack[0].acc.0[persp as usize];
            refresh_into(self.net, board, persp, dst);
        }
        self.stack[0].dirty = DirtyPiece::EMPTY;
        self.stack[0].computed = [true, true];
    }

    /// Record a move made from search ply `ply`: its deltas land in
    /// `stack[ply + 1]`, marked uncomputed (the invalidation rule). Call
    /// AFTER deriving `dp` from the pre-move board and BEFORE the child
    /// evaluates.
    pub fn push(&mut self, ply: usize, dp: DirtyPiece) {
        let child = &mut self.stack[ply + 1];
        child.dirty = dp;
        child.computed = [false, false];
    }

    /// Evaluate `board` at search ply `ply` through the NNUE forward pass,
    /// folding the accumulators forward lazily as needed. `board` must be the
    /// position at `ply` (its side to move selects the stm half).
    pub fn evaluate(&mut self, board: &Board, ply: usize) -> Value {
        // Precondition: the root was refreshed. `compute` is release-safe
        // regardless (it saturates at ply 0), but a missing `refresh_root`
        // yields a silently-wrong zero accumulator — catch it loudly in debug.
        debug_assert!(
            self.stack[0].computed[0] && self.stack[0].computed[1],
            "evaluate at ply {ply} before refresh_root — stack[0] uncomputed (design/11 §3.4.4)"
        );
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

    /// The already-incremental accumulator pair in side-to-move order, for a
    /// frozen classic policy head; never refreshes the transformer.
    #[cfg(feature = "search-control")]
    pub fn policy_accumulators(
        &mut self,
        board: &Board,
        ply: usize,
    ) -> (&[i16; L1], &[i16; L1], super::network::Activation) {
        let stm = board.side_to_move();
        self.compute(ply, stm);
        self.compute(ply, stm.flip());
        let entry = &self.stack[ply].acc.0;
        (
            &entry[stm as usize],
            &entry[stm.flip() as usize],
            self.net.activation,
        )
    }

    /// Ensure `stack[ply].acc[perspective]` is valid: walk back to the
    /// nearest computed ancestor and fold each intervening dirty record
    /// forward. `stack[0]` is always computed, so the walk terminates.
    fn compute(&mut self, ply: usize, perspective: Color) {
        let p = perspective as usize;
        if self.stack[ply].computed[p] {
            return;
        }
        let mut k = ply;
        // The `k > 0` bound keeps this memory-safe in RELEASE even if
        // `refresh_root` was never called: `k` saturates at 0 rather than
        // underflowing `usize` to a wild index.
        while k > 0 && !self.stack[k].computed[p] {
            k -= 1;
        }
        debug_assert!(
            self.stack[k].computed[p],
            "compute reached ply {ply} with no computed ancestor — refresh_root not called? (design/11 §3.4.4)"
        );
        let net = self.net;
        for i in (k + 1)..=ply {
            // Disjoint mutable borrow: src = stack[i-1], dst = stack[i]; the
            // kernel writes straight into `dst` (no per-ply accumulator
            // memcpy). `split_at_mut(i)` proves lo[i-1] and hi[0] don't alias.
            let (lo, hi) = self.stack.split_at_mut(i);
            let dp = hi[0].dirty;
            let src = &lo[i - 1].acc.0[p];
            let dst = &mut hi[0].acc.0[p];
            apply_dirty(net, dst, src, &dp, perspective);
            hi[0].computed[p] = true;
        }
    }

    /// The NNUE analogue of `Board::assert_consistent`: the incremental
    /// accumulator at `ply` must equal a from-scratch refresh of `board`,
    /// both perspectives. Compiled only in test/debug builds.
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

/// From-scratch accumulator for one perspective: `bias + Σ_active W0[feature]`.
/// No heap: at most 64 pieces, so a fixed-size row buffer suffices.
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

/// Fold one move's deltas into `dst` from base `src` for `perspective`:
/// `dst = src + Σ W0[add] − Σ W0[sub]`. Routes by delta shape so the SIMD
/// backend covers every update kind through the [`KERNELS`] seam:
///   - quiet / quiet-promotion → `(1 add, 1 sub)` → `add_sub`;
///   - capture / en passant / capture-promotion → `(1 add, 2 sub)` → `add_sub_sub`;
///   - castling → `(2 add, 2 sub)` → `add2_sub2`;
///   - null (`0, 0`) or any unforeseen shape → the element-wise fallback.
fn apply_dirty(
    net: &Network,
    dst: &mut [i16; L1],
    src: &[i16; L1],
    dp: &DirtyPiece,
    perspective: Color,
) {
    let adds = dp.adds();
    let subs = dp.subs();
    let row = |(piece, sq): (Piece, Square)| net.ft_row(feature_index(perspective, piece, sq));
    match (adds.len(), subs.len()) {
        (1, 1) => (KERNELS.add_sub)(dst, src, row(adds[0]), row(subs[0])),
        (1, 2) => (KERNELS.add_sub_sub)(dst, src, row(adds[0]), row(subs[0]), row(subs[1])),
        (2, 2) => (KERNELS.add2_sub2)(
            dst,
            src,
            row(adds[0]),
            row(adds[1]),
            row(subs[0]),
            row(subs[1]),
        ),
        // General fallback (null = (0, 0); any unforeseen shape).
        _ => {
            *dst = *src;
            for &(piece, sq) in adds {
                let r = row((piece, sq));
                for i in 0..L1 {
                    dst[i] = dst[i].wrapping_add(r[i]);
                }
            }
            for &(piece, sq) in subs {
                let r = row((piece, sq));
                for i in 0..L1 {
                    dst[i] = dst[i].wrapping_sub(r[i]);
                }
            }
        }
    }
}

/// From-scratch NNUE forward-pass evaluation of `board`: refresh both
/// perspectives and run the output layer, stm half first. The forward-pass
/// oracle for tools and tests — no incremental state, no in-check
/// precondition; returns the raw integer centipawn value.
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

    // A one-move incremental update must match a from-scratch refresh.
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

    // DirtyPiece::from_move for every move kind.
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

    // A color-flipped position evaluates EQUAL (both stm-relative) — guards
    // the feature indexer + perspective construction (§3.4.2).
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
