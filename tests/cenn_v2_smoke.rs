//! Tiny integration test: generate a CENN-v2 test net, load it via the
//! SotaContext, and verify it produces finite values.

#[cfg(feature = "sota")]
#[test]
fn cenn_v2_round_trip_and_evaluate() {
    use chessengine::eval::nnue::aligned::Aligned;
    use chessengine::eval::nnue::sota_context::SotaContext;
    use chessengine::eval::nnue::sota_features::{ACC, HALFKA_DIMS, THREAT_DIMS};
    use chessengine::eval::nnue::sota_head::{
        BUCKETS, FtScale, L1, L1Biases, L1Weights, L2, L2Biases, L2Weights, L3, L3Weights, SotaHead,
    };
    use chessengine::eval::nnue::sota_loader::{self, SotaNetwork};

    // Build a deterministic synthetic net (same as the unit tests, but
    // constructed directly since the test fixture is pub(crate)).
    let halfka_weights: Aligned<i8> = (0..HALFKA_DIMS * ACC).map(|i| (i % 11) as i8 - 5).collect();
    let ft_bias: Aligned<i16> = (0..ACC).map(|i| (i % 5) as i16 - 2).collect();
    let threat_weights: Aligned<i8> = (0..THREAT_DIMS * ACC).map(|i| (i % 9) as i8 - 4).collect();
    let l1_weights: L1Weights = (0..BUCKETS * L2 * L1)
        .map(|i| ((i as i8).wrapping_mul(5).wrapping_add(7)) % 61 - 30)
        .collect();
    let l1_biases: L1Biases = (0..BUCKETS * L2).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let l2_weights: L2Weights = (0..BUCKETS * L2 * L3)
        .map(|i| (i as f32) * 0.003 - 0.4)
        .collect();
    let l2_biases: L2Biases = (0..BUCKETS * L3)
        .map(|i| (i as f32) * 0.005 - 0.3)
        .collect();
    let l3_weights: L3Weights = (0..BUCKETS * L3)
        .map(|i| (i as f32) * 0.007 - 0.2)
        .collect();
    let l3_biases: Box<[f32]> = (0..BUCKETS).map(|i| (i as f32) * 0.01 - 0.04).collect();

    let net = SotaNetwork {
        halfka_weights,
        ft_bias,
        threat_weights,
        threat_rowmap: Vec::new(),
        threat_rows: THREAT_DIMS,
        threat_remap: None,
        psqt_weights: Vec::new().into(),
        head: SotaHead {
            bucket_layout: chessengine::eval::nnue::sota_head::LEGACY_BUCKET_LAYOUT,
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_biases,
            l3_var_weights: None,
            l3_var_biases: None,
            l3_wdl_weights: None,
            l3_wdl_biases: None,
            skip_weights: None,
            dis_head: None,
            ft_scale: FtScale::from_flat_iter(std::iter::repeat_n(1u8, ACC)),
            network_scale: 380,
            threat_shift: 2,
        },
    };

    // Serialize → load round-trip.
    let bytes = sota_loader::serialize(&net);
    assert_eq!(&bytes[0..4], b"CEN2");
    let loaded = sota_loader::load_v2_bytes(&bytes).expect("valid net loads");
    // Perf study 65: under `narrow65` the LOADED threat table is packed in
    // place, so it is deliberately not byte-identical to the source one. The
    // invariant that still has to hold — and the one every net-surgery tool
    // depends on — is that serialising the loaded net reproduces the FILE.
    assert_eq!(
        bytes,
        sota_loader::serialize(&loaded),
        "serialize o load must be the identity on bytes"
    );
    #[cfg(not(feature = "narrow65"))]
    assert_eq!(net, loaded);

    // Evaluate startpos — must produce a finite value.
    let leaked: &'static SotaNetwork = Box::leak(Box::new(loaded));
    let mut ctx = SotaContext::new(leaked);
    let board = chessengine::Board::from_fen(chessengine::fen::START_FEN).unwrap();
    ctx.refresh_root(&board);
    let val = ctx.evaluate(&board, 0);
    assert!(
        val.abs() < 100_000,
        "eval should be finite and sane, got {val}"
    );
}
