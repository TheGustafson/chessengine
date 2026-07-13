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
  threats i8, threats bias DROPPED — CSE compile fix + matches §12.2). Index
  math is a PLACEHOLDER — port the canonical version from
  `src/eval/nnue/sota_features.rs` (task #29/#28).
- `d40_policy.rs` — the low-level `ModelDefinition`+`run::train` policy trainer:
  from×to (4096) head off the shared trunk + combined `eval + λ·policy-CE` loss,
  named `policy_target` input. Ran on a synthetic target (loss 8.37→7.39).
