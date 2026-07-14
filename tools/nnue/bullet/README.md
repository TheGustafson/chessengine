# D40 bullet training examples

Our own `bullet` trainer examples for the D40 SOTA NNUE (design/11 §12), preserved
here (tracked) because the `bullet` checkout itself is gitignored + lives in an
ephemeral scratchpad. To use: copy into the bullet checkout's `examples/` dir and
register each as a `[[example]]` in `bullet_lib/Cargo.toml`, then
`cargo run --release --features cuda --example d40_eval` (CUDA_PATH=/usr).

- `d40_eval.rs` — the D40 eval trunk, ALL-NATIVE bullet: `ChessBucketsMirrored`
  → affine → `.crelu().pairwise_mul()` → concat → custom non-uniform 8 output
  buckets → per-bucket L1(i8)/L2/L3(f32). Save order (CENN-v2 transcode target):
  `l0w/l0b i16(255), l1w i8(64) TRANSPOSED, l1b/l2w/l2b/l3w/l3b f32`. Trained a
  real net (loss 0.0244→0.0112, 30 SB on the 49M mix corpus). NOTE the required
  `l0.init_with_effective_input_size(32)` for king-bucketed sparse inputs.
- `d40_features.rs` — FullThreats+HalfKA as a custom `SparseInputType` over the
  concatenated 0..83248 space, as TWO summed affine tensors (HalfKA i16 +
  threats i8, threats bias DROPPED — CSE compile fix + matches §12.2). It calls
  the engine's canonical index functions through a path dependency and checks
  its exact active-index sets against the engine enumerators on real records
  before training. The production graph has the custom 8-bucket 768→16→32→1
  tail and emits the exact mixed-quant tensor order consumed by
  `netpack.py ... d40`. Runtime knobs are `D40_DATA`, `D40_OUT`, `D40_ID`,
  `D40_SUPERBATCHES`, `D40_BPS`, `D40_BATCH`, `D40_THREADS`, `D40_SAVE_RATE`,
  and `D40_LR`.
- `d40_policy.rs` — streams the real 100M-row dual-target NumPy corpus, rebuilds
  each normalized placement, and uses the engine's canonical HalfKA+FullThreats
  enumerators. `D40_POLICY_HEAD=bilinear|full` selects the rank-8 or 4096-logit
  head off the shared pairwise trunk; both optimize `eval + λ·policy-CE`.
