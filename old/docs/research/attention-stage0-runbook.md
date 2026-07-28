# Attention head — Stage-0 offline runbook (Arm A)

Status: SET UP, NOT YET RUN (2026-07-18). This is the launch recipe for the
Stage-0 offline gate in `docs/research/transformer-head-exploration.md` §5:

> On the SAME inputs and held-out value data, does an attention head fit the
> value target BETTER than our D40 MLP head? If none beats the MLP -> STOP.

It trains two matched-data controls, both **warm-started + FT-frozen from the
720sb incumbent**, so the ONLY thing that differs is the head (MLP vs
attention):

1. **Baseline** — the existing `d40_features` MLP head (`768->16->32->1`/bucket).
2. **Arm A** — `d40_attn`, the diagonal self-gating KQV attention head.

Then compare their **held-out validation loss** on the same disjoint slice.

> BOX-BUSY CAVEAT: a bullet CUDA job is running. Do NOT launch either command
> until the GPU is free (a competing CUDA build/run can OOM the training job).
> Check with `nvidia-smi` / `pgrep -f 'examples/d40'` first.
>
> FIRST-COMPILE CAVEAT: `d40_attn.rs` was written against bullet's API without
> compiling (box-busy). Expect one compile-iteration pass. See the
> "compile-readiness" section at the bottom of this file and the header comment
> in `tools/nnue/bullet/d40_attn.rs` for the ops to sanity-check first.

---

## 0. One-time setup (register the example)

Same mechanism as every other bullet trainer in `tools/nnue/bullet/` (see that
dir's `README.md`): the `.rs` lives here (git-tracked) because the bullet
checkout is gitignored/ephemeral. Copy it in and register it.

```bash
BULLET=.archon-state/toolchains/bullet

# 1. Ensure the frozen-weights patch is applied (train_frozen; needed for FT freeze).
bash tools/nnue/apply_bullet_patches.sh "$BULLET"

# 2. Copy the trainer into the bullet examples dir.
cp tools/nnue/bullet/d40_attn.rs "$BULLET/examples/d40_attn.rs"

# 3. Register it as an example in bullet_lib/Cargo.toml (once), alongside
#    d40_features/d41_alloc11. Add:
#        [[example]]
#        name = "d40_attn"
#        path = "examples/d40_attn.rs"
#    (The chessengine path-dep + cuda feature are already set up for d40_features.)
```

Fast sanity gate BEFORE any real run (parity scan, no training, seconds):

```bash
cd "$BULLET"
CUDA_PATH=/usr D40_DATA=<small.bin> D40_VERIFY_ONLY=1 D40_VERIFY_ROWS=256 \
  cargo run --release --features cuda --example d40_attn
```

This compiles the trainer and runs ONLY the HalfKA+threats feature-parity
self-check against the engine enumerators (shared front end with d40_features).
If this passes, the front end + build are good; only the head remains to smoke.

Then a ~1-superbatch smoke run (proves the attention graph forward/backward
compiles + runs on GPU) before committing to a long run — per the
"smoke-test before long jobs" memory.

```bash
cd "$BULLET"
CUDA_PATH=/usr D40_DATA=<small.bin> \
  D40_SUPERBATCHES=1 D40_BPS=32 D40_BATCH=4096 D40_SAVE_RATE=1 \
  D40_ATTN_TOKENS=12 D40_ATTN_QKDIM=32 \
  cargo run --release --features cuda --example d40_attn
```

---

## 1. Shared run parameters (matched-data control)

Pick a SMALL disjoint train/val split so the A/B is cheap and leakage-free.
Use the SAME `TRAIN`, `VAL`, target knobs, superbatch count, and the SAME
frozen 720sb FT for BOTH arms — that is the whole point (isolate the head).

```bash
export BULLET=.archon-state/toolchains/bullet
export CUDA_PATH=/usr

# Data (bulletformat ChessBoard .bin, or set D40_FORMAT=sfbinpack + a .binpack).
export TRAIN=<path/to/train_slice.bin>          # small matched slice
export VAL=<path/to/val_slice.bin>              # DISJOINT held-out slice

# The frozen FT: the 720sb incumbent raw checkpoint.
export FT_RAW=data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin

# Matched training/target knobs for BOTH arms.
export COMMON="\
  D40_DATA=$TRAIN \
  D40_ATTN_VAL_DATA=$VAL D40_ATTN_VAL_MAX=4096 \
  D40_INIT_RAW=$FT_RAW D40_ATTN_FREEZE_FT=1 \
  D40_SUPERBATCHES=20 D40_BPS=512 D40_BATCH=16384 D40_SAVE_RATE=5 \
  D40_LR=0.001 D40_WDL=0.3 D40_EVAL_TARGET_SCALE=400 D40_SEED=42 \
  D40_THREADS=6"
```

Notes:
- `D40_INIT_RAW=$FT_RAW D40_ATTN_FREEZE_FT=1` = warm-start the FT from 720sb and
  train ONLY the head. `d40_attn` loads ONLY the 3 FT tensors
  (`l0_pstw/l0_pstb/l0_threatw`) from the raw file (it ignores the MLP tensors
  that don't exist in the attention graph).
- `d40_features` (the baseline) ALSO supports `D40_INIT_RAW` (full warm-start),
  but has NO freeze flag. To match "frozen FT" for the baseline, use the
  frozen-trunk path via the existing infrastructure OR accept that the baseline
  fine-tunes the FT slightly; for the cleanest isolation, run the baseline with
  `D40_INIT_RAW=$FT_RAW` and a tiny LR so its FT barely moves, OR (preferred)
  compare against the KNOWN 720sb incumbent held-out loss recomputed on `$VAL`
  with `D40_PROBE_*` (see step 4). The scientific question is only "does
  attention BEAT the MLP head on the same target"; the incumbent MLP's held-out
  loss on `$VAL` is the number to beat.

---

## 2. Run the MLP-head baseline (existing d40_features)

Short matched-data control. (d40_features writes both a quant and a raw
checkpoint; for the offline loss comparison we only need its training/held-out
behavior on the same slice.)

```bash
cd "$BULLET"
env $COMMON \
  D40_OUT=d40_attn_stage0/mlp D40_ID=mlp-720frozen \
  cargo run --release --features cuda --example d40_features \
  2>&1 | tee /tmp/stage0_mlp.log
```

d40_features prints per-batch `loss` (the running training loss). For a held-out
number on `$VAL`, evaluate its final checkpoint with its probe path (step 4).

## 3. Run Arm A (d40_attn)

```bash
cd "$BULLET"
env $COMMON \
  D40_OUT=d40_attn_stage0/attn D40_ID=attnA-720frozen \
  D40_ATTN_TOKENS=12 D40_ATTN_QKDIM=32 D40_ATTN_OUT_HID=16 \
  cargo run --release --features cuda --example d40_attn \
  2>&1 | tee /tmp/stage0_attn.log
```

`d40_attn` prints, at every save point (`D40_SAVE_RATE` superbatches and the
final one):

```
  [sb N] held-out val loss <MSE> over <K> positions
```

This is **mean `(sigmoid(out) - target)^2`** over `$VAL`, computed with the
IDENTICAL target formula and loss as training — so it is directly comparable to
the MLP head's held-out loss on the same slice.

Sweep suggestions once the smoke passes (cheap, one knob at a time):
`D40_ATTN_TOKENS` ∈ {8, 12, 16, 24}, `D40_ATTN_QKDIM` ∈ {16, 32, 64},
`D40_ATTN_VDIM` (default = d), `D40_ATTN_OUT_HID` ∈ {16, 32}.

## 4. Compare held-out loss (the gate)

The decision is a direct number comparison on the SAME `$VAL` slice:

- **Arm A held-out loss** = the final `[sb N] held-out val loss` from
  `/tmp/stage0_attn.log`.
- **MLP baseline held-out loss** = the incumbent MLP head's mean
  `(sigmoid(out)-target)^2` on `$VAL`. Get it from either:
  - the matched d40_features run's final held-out behavior, or
  - the 720sb incumbent directly: run d40_features' probe path over `$VAL`'s
    FENs (`D40_PROBE_FENS` + `D40_PROBE_RAW=$FT_RAW`) and compute MSE vs the same
    `target = wdl*result + (1-wdl)*sigmoid(score/scale)` externally.

Gate (design doc §5 Stage 0):
- **Arm A loss meaningfully < MLP loss** -> eval headroom exists; proceed to
  the subsumption probe (`D40_HALFKA_ONLY=1`: does attention over HalfKA-only
  match HalfKA+threats+MLP?) and then Stage 1 (Rust scalar forward + fixed-node
  screen). The pragmatic payoff is **distillation** into the fast MLP head.
- **Arm A loss ≈ or > MLP loss** -> accumulator-chunk attention does not help;
  STOP Arm A. (Arm C — attention over per-square FT embeddings — is the stronger
  candidate but needs an FT restructure; it is a separate trainer, not this
  file. See the header of `tools/nnue/bullet/d40_attn.rs`.)

Record the result (loss numbers, arm, knobs, verdict) in
`docs/research/value-foundation-state.md` and add a ledger row per D38.

---

## Compile-readiness (what to verify on the first compile pass)

`d40_attn.rs` was composed by reading the bullet API, NOT compiled (box-busy).
The FT front end, mapper, target/loss path, save machinery, warm-start, and
freeze are byte-clones of `d40_features.rs` / `d41_alloc11.rs` (already known to
compile). The NEW surface is the attention head graph. Confidence:

- HIGH (standard ModelNode ops, used in d40): `crelu`, `reshape` (metadata
  no-op), `concat`, `select(buckets)`, `screlu`, affine `forward`, `sigmoid`,
  `squared_error`/`power_error`, `reduce_sum_batch`, `new_weights`/`new_affine`,
  `matmul` weight@activation with k=1.
- MEDIUM-HIGH — softmax composed by hand from `exp` + `reduce_sum_rows` +
  `reduce_sum_cols` + `repeat` + reciprocal (`1.0/node`). Each op exists on
  `ModelNode` (builder.rs:195/200/337/421/516) but the composition + the
  per-token `reduce_sum_rows` on a batched `[dk, T]` activation is not exercised
  by d40. Verify the softmax normalizes over tokens (cols), not rows.
- MEDIUM — `wq.matmul(x)` with `x` a batched `[d, T]` activation (weight @
  activation, k=T>1). Same lowering as affine.forward (linear.rs:141) but d40
  only ever uses k=1; k>1 folds to one `[m,n]@[n, T*batch]` gemm.
- MEDIUM (verify FIRST) — `v.matmul(w_col)`: a **batched activation @ batched
  activation** matmul (both operands batched, the (true,true) branch,
  linear.rs:137). d40/d41 never do act@act. It IS supported end to end: the
  model op lowers to `operation::Matmul` with `batch=batch_size`
  (linear.rs:137-139), which dispatches to `cublasSgemmStridedBatched` with full
  per-matrix strides (strideA=m*k, strideB=k*n — cuda.rs:294, function.rs:271),
  correct for act@act. If it does NOT compile/run, the fallback is to
  reshape/pool differently, but this path should work.

Ops confirmed NOT expressible at the model-builder level (hence the diagonal
self-gating design, not full `Q.K^T`):
- No `ModelNode::transpose` — transpose lives only on the tensor-IR
  `MatrixLayout::transpose` (matmul.rs:16); the model `Matmul` hardcodes
  `col_mjr: true` (linear.rs:130/134/138).
- No standalone `ModelNode::softmax` — only the fused
  `softmax_crossentropy_loss` (builder.rs:360); the real `softmax(inner_size)`
  is on the tensor-IR `TNode` (tensor/builder.rs:159), unreachable from
  `ModelDefinition::build`.
- No model-level reduce-MAX -> the composed softmax omits max-subtraction (safe:
  `crelu` bounds the accumulator to [0,1], QKDIM is small, so `exp` can't
  overflow).
```
