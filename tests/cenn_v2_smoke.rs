//! Tiny integration test: generate a CENN-v2 test net, load it via the
//! SotaContext, and verify it produces finite values.

#[cfg(feature = "sota")]
#[test]
fn cenn_v2_round_trip_and_evaluate() {
    use chessengine::eval::nnue::sota_context::SotaContext;
    use chessengine::eval::nnue::sota_head::{SotaHead, BUCKETS, L1, L2, L3};
    use chessengine::eval::nnue::sota_features::{ACC, HALFKA_DIMS, THREAT_DIMS};
    use chessengine::eval::nnue::sota_loader::{self, SotaNetwork};

    // Build a deterministic synthetic net (same as the unit tests, but
    // constructed directly since the test fixture is pub(crate)).
    let halfka_weights: Box<[i16]> = (0..HALFKA_DIMS * ACC)
        .map(|i| (i % 11) as i16 - 5)
        .collect();
    let ft_bias: Box<[i16]> = (0..ACC).map(|i| (i % 5) as i16 - 2).collect();
    let threat_weights: Box<[i8]> = (0..THREAT_DIMS * ACC)
        .map(|i| (i % 9) as i8 - 4)
        .collect();
    let l1_weights: Box<[i8]> = (0..BUCKETS * L2 * L1)
        .map(|i| ((i as i8).wrapping_mul(5).wrapping_add(7)) % 61 - 30)
        .collect();
    let l1_biases: Box<[f32]> = (0..BUCKETS * L2)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let l2_weights: Box<[f32]> = (0..BUCKETS * L2 * L3)
        .map(|i| (i as f32) * 0.003 - 0.4)
        .collect();
    let l2_biases: Box<[f32]> = (0..BUCKETS * L3)
        .map(|i| (i as f32) * 0.005 - 0.3)
        .collect();
    let l3_weights: Box<[f32]> = (0..BUCKETS * L3)
        .map(|i| (i as f32) * 0.007 - 0.2)
        .collect();
    let l3_biases: Box<[f32]> = (0..BUCKETS)
        .map(|i| (i as f32) * 0.01 - 0.04)
        .collect();

    let net = SotaNetwork {
        halfka_weights,
        ft_bias,
        threat_weights,
        head: SotaHead {
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_biases,
            network_scale: 380,
        },
    };

    // Serialize → load round-trip.
    let bytes = sota_loader::serialize_v2(&net);
    assert_eq!(&bytes[0..4], b"CEN2");
    let loaded = sota_loader::load_v2_bytes(&bytes).expect("valid net loads");
    assert_eq!(net, loaded);

    // Evaluate startpos — must produce a finite value.
    let leaked: &'static SotaNetwork = Box::leak(Box::new(loaded));
    let ctx = SotaContext::new(leaked);
    let board = chessengine::Board::from_fen(chessengine::fen::START_FEN).unwrap();
    let val = ctx.evaluate(&board, 0);
    assert!(
        val.abs() < 100_000,
        "eval should be finite and sane, got {val}"
    );
}
