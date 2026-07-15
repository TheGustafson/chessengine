//! Per-thread SOTA eval context: the accumulator stack + head forward for D40
//! nets (design/11 §12, D40).
//!
//! Analogous to [`super::accumulator::NnueContext`] but for the D40 architecture:
//! two feature transformers (HalfKA + FullThreats), each producing a
//! per-perspective [`ACC`]-wide accumulator, fed into the multi-layer bucketed
//! [`SotaHead`].
//!
//! Both transformers are incremental on the search stack. HalfKA folds ordinary
//! dirty-piece columns lazily and uses a 64-state-per-perspective Finny cache on
//! king moves; FullThreats applies its bounded x-ray delta list eagerly after
//! board mutation. Root refresh and standalone `nnue-eval` remain from-scratch.
//!
//! Behind the `sota` cargo feature.

use super::sota_accumulator::HalfkaContext;
#[cfg(debug_assertions)]
use super::sota_features::refresh_both;
use super::sota_loader::SotaNetwork;
use super::sota_threats::ThreatContext;
use crate::board::Board;
use crate::board::DirtyPiece;
use crate::types::Value;

/// Per-thread SOTA evaluation state, owned by a search `Worker` via
/// [`super::super::EvalContext`].
pub struct SotaContext {
    /// The loaded net. Leaked to `'static` by `set_eval_file` so multiple
    /// threads can borrow it concurrently without lifetime gymnastics.
    net: &'static SotaNetwork,
    halfka: HalfkaContext<'static>,
    threats: ThreatContext<'static>,
}

impl std::fmt::Debug for SotaContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SotaContext")
            .field("net", &(self.net as *const SotaNetwork))
            .finish_non_exhaustive()
    }
}

impl SotaContext {
    /// Create a new context over the given net.
    pub fn new(net: &'static SotaNetwork) -> Self {
        SotaContext {
            net,
            halfka: HalfkaContext::new(&net.halfka_weights, &net.ft_bias),
            threats: ThreatContext::new(&net.threat_weights),
        }
    }

    /// The net this context evaluates against.
    pub fn net(&self) -> &SotaNetwork {
        self.net
    }

    /// Seed both accumulator stacks from `board` at ply zero.
    pub fn refresh_root(&mut self, board: &Board) {
        self.halfka.refresh_root(board);
        self.threats.refresh(board);
    }

    /// Record one post-move position at `ply + 1`.
    pub fn push(&mut self, dirty: DirtyPiece, ply: usize, post_move_board: &Board) {
        self.halfka.push(ply, dirty);
        self.threats.apply_at(ply, &dirty, post_move_board);
    }

    /// Evaluate `board` through the incrementally maintained SOTA accumulators.
    pub fn evaluate(&mut self, board: &Board, ply: usize) -> Value {
        // The production search-stack oracle: catch stale sibling entries,
        // missed king refreshes, and FullThreats delta drift at the first node
        // where they could affect search. This mirrors the classic NNUE
        // accumulator oracle and compiles out of release builds.
        #[cfg(debug_assertions)]
        self.assert_matches_refresh(board, ply);

        let stm = board.side_to_move();
        let pst = self.halfka.current(board, ply);
        let threat = self.threats.at(ply);

        // Count total pieces for the output bucket.
        let piece_count = board.all_occupancy().count() as usize;

        self.net.head.forward(pst, threat, stm, piece_count)
    }

    #[cfg(debug_assertions)]
    fn assert_matches_refresh(&mut self, board: &Board, ply: usize) {
        let got_pst = *self.halfka.current(board, ply);
        let got_threat = *self.threats.at(ply);
        let expected = refresh_both(
            &self.net.halfka_weights,
            &self.net.ft_bias,
            &self.net.threat_weights,
            board,
        );
        assert_eq!(
            got_pst, expected.0,
            "SOTA HalfKA accumulator drift at ply {ply}"
        );
        assert_eq!(
            got_threat, expected.1,
            "SOTA FullThreats accumulator drift at ply {ply}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;
    use crate::movegen::generate_legal;
    use crate::moves::MoveList;

    /// Build a leaked static SotaNetwork from the loader's test fixture.
    fn leaked_test_net() -> &'static SotaNetwork {
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        Box::leak(Box::new(test_net_with_threats_pub()))
    }

    fn leaked_halfka_only_test_net() -> &'static SotaNetwork {
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        let mut net = test_net_with_threats_pub();
        net.threat_weights = Vec::new().into_boxed_slice();
        Box::leak(Box::new(net))
    }

    #[test]
    fn evaluate_returns_finite_value() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        ctx.refresh_root(&board);
        let val = ctx.evaluate(&board, 0);
        assert!(
            val.abs() < 100_000,
            "eval should be finite and sane, got {val}"
        );
    }

    #[test]
    fn evaluate_stm_relative() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        // Same position, different side to move.
        let w =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let b =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        ctx.refresh_root(&w);
        let val_w = ctx.evaluate(&w, 0);
        ctx.refresh_root(&b);
        let val_b = ctx.evaluate(&b, 0);
        // With a random test net, these won't be exact negatives, but they
        // should be finite and (with asymmetric accumulators) likely different.
        assert!(val_w.abs() < 100_000);
        assert!(val_b.abs() < 100_000);
    }

    #[test]
    fn evaluate_deterministic() {
        let net = leaked_test_net();
        let mut ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        ctx.refresh_root(&board);
        let a = ctx.evaluate(&board, 0);
        let b = ctx.evaluate(&board, 0);
        assert_eq!(a, b, "eval must be deterministic");
    }

    #[test]
    fn incremental_context_matches_fresh_context() {
        let net = leaked_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let mut seed = 0x51A7_A40Du64;
        for ply in 0..48usize {
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
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, &board);
            let got = incremental.evaluate(&board, ply + 1);

            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(&board);
            let expected = fresh.evaluate(&board, 0);
            assert_eq!(got, expected, "ply {ply} {m}");
        }
    }

    #[test]
    fn halfka_only_context_has_zero_threat_channel() {
        let net = leaked_halfka_only_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let mut seed = 0x0A11_FA1Fu64;
        for ply in 0..24usize {
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
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, &board);
            let got = incremental.evaluate(&board, ply + 1);

            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(&board);
            assert_eq!(got, fresh.evaluate(&board, 0), "ply {ply} {m}");
        }
    }

    fn explore_siblings(
        net: &'static SotaNetwork,
        incremental: &mut SotaContext,
        board: &mut Board,
        ply: usize,
        depth: usize,
    ) -> usize {
        if depth == 0 {
            return 0;
        }
        let mut moves = MoveList::new();
        generate_legal(board, &mut moves);
        let mut visited = 0;
        for index in 0..moves.len().min(3) {
            let m = moves[index];
            let dirty = DirtyPiece::from_move(board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply, board);
            let got = incremental.evaluate(board, ply + 1);
            let mut fresh = SotaContext::new(net);
            fresh.refresh_root(board);
            assert_eq!(got, fresh.evaluate(board, 0), "sibling ply {ply} {m}");
            visited += 1 + explore_siblings(net, incremental, board, ply + 1, depth - 1);
            board.unmake_move(m);
        }
        visited
    }

    #[test]
    fn sibling_reuse_matches_fresh_context() {
        let net = leaked_test_net();
        let mut incremental = SotaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let visited = explore_siblings(net, &mut incremental, &mut board, 0, 3);
        assert!(
            visited >= 30,
            "expected a broad sibling tree, got {visited}"
        );
    }
}
