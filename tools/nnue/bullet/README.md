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
  `D40_LR`, `D40_SEED`, `D40_WDL`, `D40_EVAL_TARGET_SCALE`, and
  `D40_SCORE_MULTIPLIER`. `D40_INIT_RAW`/`D40_INIT_QUANT` resume or fine-tune
  from a named checkpoint. `D40_VERIFY_ROWS=0 D40_VERIFY_ONLY=1` performs the
  full corpus feature/capacity parity scan without training. The final
  superbatch also writes a named raw checkpoint. With
  `D40_PROBE_FENS`, either `D40_PROBE_RAW` or `D40_PROBE_QUANT` evaluates the
  identical graph for the orchestrator's cross-boundary parity gate.
- `d40_policy.rs` — streams the real 100M-row dual-target NumPy corpus, rebuilds
  each normalized placement, and uses the engine's canonical HalfKA+FullThreats
  enumerators. `D40_POLICY_HEAD=eval|bilinear|full` selects the controlled
  eval-only baseline, configurable-rank (`D40_POLICY_RANK`, default 8) partial
  policy, or 4096-logit full policy head off
  the shared pairwise trunk. `D40_EVAL_TARGET_SCALE` defaults to the empirically
  stronger soft target 400; scale 200 is the exact `(1+Q)/2` inverse but lost
  the classic strength A/B. `D40_WDL` defaults to 0.3. Seeds and campaign
  sizing are environment-driven.
