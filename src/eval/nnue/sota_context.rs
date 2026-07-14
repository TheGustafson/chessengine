//! Per-thread SOTA eval context: the accumulator stack + head forward for D40
//! nets (design/11 §12, D40).
//!
//! Analogous to [`super::accumulator::NnueContext`] but for the D40 architecture:
//! two feature transformers (HalfKA + FullThreats), each producing a
//! per-perspective [`ACC`]-wide accumulator, fed into the multi-layer bucketed
//! [`SotaHead`].
//!
//! **MVP strategy (from-scratch refresh per eval).** This first cut rebuilds
//! both accumulators from scratch at every `evaluate` call. This is correct but
//! O(~200 features × 768 neurons) per perspective per eval — about 100× slower
//! than incremental. It is sufficient for:
//! - Load/sanity testing: does the packed net produce sane centipawns?
//! - Fixed-node SPRT: node-limited games don't care about wall time.
//!
//! Incremental HalfKA (dirty-piece add/sub, Finny cache for king moves) and
//! incremental FullThreats (via [`super::sota_threats::ThreatContext`]) are
//! deferred optimizations — they will be wired into the `MakeMoveHooks` seam
//! once the from-scratch path proves correct end-to-end.
//!
//! Behind the `sota` cargo feature.

use super::sota_features;
use super::sota_loader::SotaNetwork;
use crate::board::Board;
use crate::types::Value;

/// Per-thread SOTA evaluation state, owned by a search `Worker` via
/// [`super::super::EvalContext`].
#[derive(Debug)]
pub struct SotaContext {
    /// The loaded net. Leaked to `'static` by `set_eval_file` so multiple
    /// threads can borrow it concurrently without lifetime gymnastics.
    net: &'static SotaNetwork,
}

impl SotaContext {
    /// Create a new context over the given net.
    pub fn new(net: &'static SotaNetwork) -> Self {
        SotaContext { net }
    }

    /// The net this context evaluates against.
    pub fn net(&self) -> &SotaNetwork {
        self.net
    }

    /// Evaluate `board` through the SOTA forward pass. From-scratch refresh
    /// (MVP): rebuilds both FT accumulators from scratch, then runs the head.
    /// `ply` is accepted for API compatibility but ignored (no lazy stack yet).
    pub fn evaluate(&self, board: &Board, _ply: usize) -> Value {
        let stm = board.side_to_move();

        // From-scratch refresh: build the (pst, threat) pair for both
        // perspectives.
        let (pst, threat) = sota_features::refresh_both(
            &self.net.halfka_weights,
            &self.net.ft_bias,
            &self.net.threat_weights,
            board,
        );

        // Count total pieces for the output bucket.
        let piece_count = board.all_occupancy().count() as usize;

        self.net.head.forward(&pst, &threat, stm, piece_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    /// Build a leaked static SotaNetwork from the loader's test fixture.
    fn leaked_test_net() -> &'static SotaNetwork {
        use super::super::sota_loader::tests::test_net_with_threats_pub;
        Box::leak(Box::new(test_net_with_threats_pub()))
    }

    #[test]
    fn evaluate_returns_finite_value() {
        let net = leaked_test_net();
        let ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        let val = ctx.evaluate(&board, 0);
        assert!(
            val.abs() < 100_000,
            "eval should be finite and sane, got {val}"
        );
    }

    #[test]
    fn evaluate_stm_relative() {
        let net = leaked_test_net();
        let ctx = SotaContext::new(net);
        // Same position, different side to move.
        let w = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let b = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let val_w = ctx.evaluate(&w, 0);
        let val_b = ctx.evaluate(&b, 0);
        // With a random test net, these won't be exact negatives, but they
        // should be finite and (with asymmetric accumulators) likely different.
        assert!(val_w.abs() < 100_000);
        assert!(val_b.abs() < 100_000);
    }

    #[test]
    fn evaluate_deterministic() {
        let net = leaked_test_net();
        let ctx = SotaContext::new(net);
        let board = Board::from_fen(START_FEN).unwrap();
        let a = ctx.evaluate(&board, 0);
        let b = ctx.evaluate(&board, 0);
        assert_eq!(a, b, "eval must be deterministic");
    }
}
