//! `set_eval_file` idempotency — the D40 per-game leak regression test.
//!
//! Match runners (fastchess) re-send every configured option, `EvalFile`
//! included, before every game; each install leaks its parsed net to
//! `'static`, so a same-file re-send must be a no-op or a match leaks one
//! net per game. This lives in its own integration-test binary (= its own
//! process) because it installs process-global nets, which the unit-test
//! suite deliberately never does.

#![cfg(feature = "sota")]

use chessengine::eval::nnue::{
    self,
    sota_features::{ACC, HALFKA_DIMS},
    sota_head::{BUCKETS, FtScale, L1, L2, L3, SotaHead},
    sota_loader::{SotaNetwork, serialize},
};

/// A minimal valid no-threats net; `scale` differentiates file contents.
fn tiny_net(scale: i32) -> SotaNetwork {
    SotaNetwork {
        halfka_weights: vec![0i8; HALFKA_DIMS * ACC].into(),
        ft_bias: vec![0i16; ACC].into(),
        threat_weights: Vec::new().into(),
        threat_rows: 0,
        threat_remap: None,
        psqt_weights: Vec::new().into(),
        head: SotaHead {
            l1_weights: vec![0i8; BUCKETS * L2 * L1].into(),
            l1_biases: vec![0.0; BUCKETS * L2].into(),
            l2_weights: vec![0.0; BUCKETS * L2 * L3].into(),
            l2_biases: vec![0.0; BUCKETS * L3].into(),
            l3_weights: vec![0.0; BUCKETS * L3].into(),
            l3_biases: vec![0.0; BUCKETS].into_boxed_slice(),
            l3_var_weights: None,
            l3_var_biases: None,
            l3_wdl_weights: None,
            l3_wdl_biases: None,
            skip_weights: None,
            ft_scale: FtScale::from_flat_iter(std::iter::repeat_n(1u8, ACC)),
            network_scale: scale,
            threat_shift: 2,
        },
    }
}

#[test]
fn same_file_resend_is_a_noop_and_changed_file_reloads() {
    let path = std::env::temp_dir().join(format!(
        "archon-eval-reload-test-{}.cenn",
        std::process::id()
    ));
    let path_str = path.to_str().unwrap();

    std::fs::write(&path, serialize(&tiny_net(380))).unwrap();
    nnue::set_eval_file(path_str).unwrap();
    let first = nnue::active_sota_net().expect("net installed") as *const SotaNetwork;

    // The fastchess pattern: the identical setoption again. The active net
    // must be the same allocation — anything else is a per-game leak.
    nnue::set_eval_file(path_str).unwrap();
    let second = nnue::active_sota_net().expect("net still installed") as *const SotaNetwork;
    assert_eq!(first, second, "same-file re-send must not reload");

    // Overwriting the file at the same path must still reload. The byte
    // length is identical, so this leans on mtime; sleep past coarse
    // filesystem timestamp granularity.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, serialize(&tiny_net(381))).unwrap();
    nnue::set_eval_file(path_str).unwrap();
    let third = nnue::active_sota_net().expect("net reinstalled");
    assert_eq!(third.head.network_scale, 381, "changed file must reload");

    // Empty path turns NNUE off; a missing file is still a clean error.
    nnue::set_eval_file("").unwrap();
    assert!(nnue::active_sota_net().is_none());
    assert!(nnue::set_eval_file("/nonexistent-net.cenn").is_err());

    std::fs::remove_file(&path).ok();
}
