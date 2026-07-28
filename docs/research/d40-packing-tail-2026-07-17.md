# D40 quantization "packing tail" — root cause + fix spec (2026-07-17)

> **Note (2026-07-28 archive sweep):** several documents and scripts cited below were
> moved to `old/`. Prefix a broken path with `old/` to find it. Nothing in `old/` is
> loaded or trusted by the current engine; see `searchopt/LEDGER.md` for what survived.

Task #26 / D45a. Investigate why `d40-uho-720sb` (HalfKAv2_hm+FullThreats
CENN-v2, 720 SB / 18B presentations) fails the runtime-parity gate:
runtime-vs-trainer eval error **mean 0.0121** (bar 0.03, OK) but
**max 0.2517** (bar 0.20, FAIL). Worst position is an OOD pawnless-ish
endgame `1k6/4R3/2K5/8/8/8/P4b2/8 w - - 27 82`.

## TL;DR

**The recorded hypothesis is wrong.** There is no pack-time clamping. No weight
in any tensor overflows its integer quantization range — bullet's quantiser
*errors* (not saturates) on overflow, and it did not error. The failing metric
is the **runtime error = |engine_integer_eval/scale − bullet_float_eval|** on
*identical* quantized weights, i.e. a pure **integer-vs-float arithmetic-path
mismatch**, dominated by the `(left*right) >> FT_SHIFT` **floor** in the pairwise
FT activation kernel. It is an inherent property of the int8 inference kernel,
amplified on saturated (rook-up) positions by the longer schedule's larger L1
weight tail. **No pack-side transform can fix it** — the quant scales are
compile-time-pinned and header-validated, and any change to a stored weight
moves the integer and float paths *together*, so it cannot shrink their
*difference*. The correct fixes are engine-side (round the shift) or
training-side (per-tensor i8 clip / QAT); or recalibrate the gate (a saturated
100cp logit undershoot has ~zero Elo cost — mean 0.0121 and p99 0.063 prove the
packing is correct). **No `-rs1.cenn` was produced; a repack cannot pass.**

## 1. Diagnosis — which tensors clip, and by how much

The D40 quant scheme is `ft255/threat64/l164/shift9/scale400`:

| tensor       | dtype | scale | represents \|w\| ≤ | quantiser |
|--------------|-------|-------|--------------------|-----------|
| `l0_pstw` (HalfKA FT) | i16 | 255 | 128.498 | round |
| `l0_pstb` (FT bias)   | i16 | 255 | 128.498 | round |
| `l0_threatw` (threat FT) | i8 | 64 | 1.9844 | round |
| `l1w` (L1)            | i8  | 64  | 1.9844 | round |
| `l1b,l2w,l2b,l3w,l3b` | f32 | —   | —      | none  |

Numeric analysis of the raw f32 checkpoints (`tools/nnue/audit_d40_quant_clip.py`,
parses bullet's `ModelWeights::write_into` format directly):

```
720sb (FAILED, runtime max 0.2517):
  l0_pstw   max|round(w*255)| = 477   OVERFLOW(|q|>32767): 0  (68x headroom)
  l0_threatw max|round(w*64)| = 125   OVERFLOW(|q|>127):   0
  l1w        max|round(w*64)| = 127   OVERFLOW(|q|>127):   0

240sb (PASSED, runtime max 0.174):
  l0_pstw   max|round(w*255)| = 503   OVERFLOW: 0
  l0_threatw max|round(w*64)| = 127   OVERFLOW: 0   <- passing net is AT the boundary
  l1w        max|round(w*64)| = 118   OVERFLOW: 0
```

**Zero clipping in either net.** Decisively, the *passing* 240sb net's threat
weights reach the i8 boundary (127) while the *failing* 720sb net's do not
(125). So the failure does **not** correlate with weights approaching the clip
range — the "longer training pushes weights past range" story is false.

Why bullet cannot clip: `QuantTarget::quantise` (bullet
`crates/trainer/src/model/weights.rs:298`) computes `qf = round(scale*w)`, does
`qf as i8` (Rust saturates the cast), then **returns `Err`** if
`qf != f64::from(x)`. So an overflowing weight aborts the save with
`.expect("quantise D40 checkpoint")`. The checkpoint exists ⇒ no overflow.
`AdamWParams::default()` (bullet `optimiser/adam.rs:31`) additionally clips every
weight to `[-1.98, 1.98]` each step; at scale 64 that is `±126.72 → ±127`, so i8
overflow is structurally impossible under the current recipe.

### The real failing metric

`tools/nnue/train_run.py:442-444` computes two errors:

- `quant_error = |quant_logit − raw_logit|` — bullet float eval on
  quantized-then-dequantized weights vs on raw f32 weights. **Pure rounding.**
- `runtime_error = |engine_cp/network_scale − quant_logit|` — the engine's
  **integer** forward pass vs bullet's **float** forward pass, both on the
  **same quantized weights**. **Pure arithmetic-path mismatch.**

Reproduced against the existing ledgered net (`--only parity`, no repack; net
sha256 unchanged):

```
parity rows=4096  quant mean/max=0.032953/0.267620   runtime mean/max=0.012124/0.251670
parity quant p95/p99=0.097884/0.153219               runtime p95/p99=0.037510/0.063481
worst quant_logit=1.264170 runtime=1.012500 fen=1k6/4R3/2K5/8/8/8/P4b2/8 w - - 27 82
```

- **quant** (rounding) errors **pass** (max 0.268 « 1.0 bar). Rounding is fine.
- **runtime** max 0.2517 fails; mean 0.0121 and **p99 0.063** are excellent — a
  thin tail (<1% of positions above 0.063) drives the single-metric failure.
- At the worst FEN the integer path **undershoots** the float path by 0.2517
  (logit 1.0125 vs 1.264 ≈ **405cp vs 506cp**) on a *saturated rook-up won*
  position — irrelevant to move choice, but it trips the absolute-max bound.

### Mechanism (the integer kernel, `src/eval/nnue/sota_simd.rs:148-156`)

```rust
let merge = |p, t| {                         // rescale threat acc to pst(255) scale
    let scaled = i32::from(t) * FT_QUANT;    //   round(t * 255/64)  — one rounding/lane
    let rounded = scaled.signum()*((scaled.abs()+THREAT_QUANT/2)/THREAT_QUANT);
    (i32::from(p)+rounded).clamp(0, FT_QUANT) // clamp [0,255]
};
let left = merge(...); let right = merge(...);
out[..] = ((left * right) >> FT_SHIFT) as u8; // FLOOR of the pairwise product
```

vs bullet's exact float `acc.crelu().pairwise_mul()`. The two integer-only lossy
ops are (1) the per-lane threat-rescale rounding and (2) the `>>9` **floor** of
`left*right ∈ [0,65025]`. The floor systematically undershoots the normalized
pairwise activation by up to `511/255² ≈ 0.0079` per lane, biased negative
(mean ≈ −0.004). That per-lane residual is then multiplied by the i8 L1 weight
and summed, so a **larger L1 weight tail amplifies it**. The 720sb `l1w` tail
grew vs 240sb (`p99.9` 0.81→**1.28**, max 1.85→**1.98**), which is the plausible
reason 720 fails where 240 passed — the extra presentations sharpened L1, not
the FT weights.

## 2. What is adjustable WITHOUT engine-code changes

**Almost nothing.** The CENN-v2 loader (`src/eval/nnue/sota_loader.rs:320-330`)
hard-validates the quant constants against the compiled-in engine values:

- `ft_quant` must == `sota_head::FT_QUANT` (255)  — rejected otherwise
- `l1_quant` must == `sota_head::L1_QUANT` (64)    — rejected otherwise
- `ft_shift` must == `sota_head::FT_SHIFT` (9)     — rejected otherwise
- **`threat_quant` (64) is not in the header at all** — always compile-pinned
  (`sota_head::THREAT_QUANT`, used by the SIMD `merge`).
- `network_scale` (header offset 44) is the **only** loader-flexible field
  (accepts `1..=10000`).

So the header carries the quant constants only as a *cross-check*, not as
honored per-tensor scales. There is **no per-tensor scale mechanism** the packer
could exploit for headroom, and no threat scale to widen.

`network_scale` is **demonstrably impotent** for this failure. The FNV checksum
covers only the payload (bytes 64+), so `network_scale` can be flipped in-header
without a repack. Patching 400→800 leaves the worst-position error **identical**:

```
scale=400: cp=405  cp/400 = 1.01250   runtime_err = 0.25167
scale=800: cp=810  cp/800 = 1.01250   runtime_err = 0.25167
```

because `runtime_error = |l3_int − quant_logit|` and `l3_int` (the pre-scale
integer output) is independent of `network_scale`; the field only sets the final
`(l3*scale) as i32` truncation granularity (≤ 1/scale ≈ 0.0025), far below the
0.05 needed to clear the bar.

## 3. Fix attempts

1. **`network_scale` rebalance (only pack-side lever)** — tried; **no effect**
   (identical 0.25167 at 400 and 800, shown above).
2. **Weight rescale / layer rebalance** — *impossible in principle*. The runtime
   error is the *difference* between the integer and float paths on identical
   weights. Both paths read the same stored tensors, so any weight edit (FT, L1,
   even the un-quantized f32 `l1b/l2/l3` biases) shifts *both* outputs by the
   same amount and cannot reduce their difference. And the runtime dequant scales
   are pinned, so a rescale is not a no-op — it changes the actual eval, which
   then breaks the (currently passing) `quant` parity. No legal repack helps.

**No `data/nnue/nets/d40-uho-720sb-rs1.cenn` was written** — a repack cannot pass
parity, and emitting a byte-different "fixed" net would be misleading.
`d40-uho-720sb.cenn` is untouched (sha256
`68f070a2791a0df47a45044f72b2f50b1965e42ec24dd01a57554cc0c56245ac`).

## 4. Recommendation

Ranked. **R0 is the recommended immediate action.**

- **R0 (recalibrate the gate + verify by Elo — do this now).** The parity gate
  exists to catch packing *bugs* (wrong layout/scale/order/checksum), which
  manifest as *broad* error (mean 0.1–1.0). This net's mean is **0.0121** and
  p99 **0.063** — proof the packing is correct. One saturated won position
  exceeds the absolute-max 0.20. Change the D40 runtime gate from an absolute
  `max` to a **p99.9 bound (or max ≈ 0.30)**, and make the real ship criterion a
  **fixed-node + equal-time Elo** check that the packed net matches its float
  parent (per the Elo-decomposition protocol). Promote `d40-uho-720sb` to Elo
  testing; it is correctly packed. (Config lever:
  `parity.max_abs_runtime_logit_error`, or add
  `p99_abs_runtime_logit_error` — `train_run.py:472-474` already supports a p99
  gate hook.)

- **R1 (engine-side, highest fidelity leverage).** Round the pairwise shift:
  `((left*right + (1 << (FT_SHIFT-1))) >> FT_SHIFT)` in all three
  `activate` backends — scalar `sota_simd.rs:156`, avx2 `~:319`
  (`_mm256_srai_epi32` → add rounding before shift), avx512 `~:547`
  (`_mm512_srai_epi32`). This centers the floor bias at 0 (±0.5 LSB), removing
  the systematic undershoot and roughly halving the max runtime error for
  *every* sota net. Update the unit test `sota_head.rs::activate_ft_hand_computed`
  (`(150*255)>>9 = 74` → rounded `(38250+256)>>9 = 75`). **Blast radius:** changes
  sota eval semantics for all sota nets → re-parity + retune + SPRT; the default
  build's bench 548978 is unaffected (sota is feature-gated). This is the
  cleanest true fix; not pack-side and not a training change.

- **R2 (training-side, "clip-aware", the one-knob change).** Tighten the i8
  weight clip so the L1 tail cannot amplify the floor residual. Currently
  `tools/nnue/bullet/d40_features.rs:685` uses `AdamWParams::default()` →
  `min_weight/max_weight = ±1.98` (bullet `optimiser/adam.rs:25-26,31`), applied
  **globally**. The offending tensors are `l1w` and `l0_threatw`; clip them to
  ~**±1.5**. Caveat: a *global* ±1.5 also binds the i16 FT tensors (`l0_pstw`
  reaches ±1.87 and is safe to ±128 at scale 255), costing FT capacity — so the
  correct form is **per-tensor clipping of only `l1w`+`l0_threatw`**, which the
  single-`AdamWParams` API does not expose; use bullet's per-weight clip
  optimiser wrapper (`crates/trainer/src/optimiser/clip.rs`) or construct the L1
  affine with its own clipped AdamW. This is a *partial* mitigation (bounds the
  amplifier, not the floor itself) and is not guaranteed to clear 0.20.

- **R3 (training-side, proper).** QAT: make the trainer forward simulate the
  integer inference (floor `>>9` + threat-rescale rounding + `[0,255]` clamp) so
  learned weights are inference-robust. Largest change; use only if R0/R1 prove
  insufficient across the net zoo.

## Reproduce

```
# clip / overflow audit (numpy via uv):
uv run python tools/nnue/audit_d40_quant_clip.py

# full runtime parity against the ledgered net, no repack (needs CUDA_PATH=/usr,
# a sota-feature target/release/chessengine, and the GPU):
RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --features sota
CUDA_PATH=/usr uv run python tools/nnue/train_run.py \
    --config data/nnue/experiments/d40-long-v1/config.json --only parity
```

## Artifacts

- Raw f32 checkpoint: `data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin`
  (sha256 `d9e18500…70ee5b`)
- Quantised bin: `…/ckpt/d40-uho-720sb-sb720-quant.bin` (sha256 `d321a08b…b3b014`)
- Ledgered net (untouched): `data/nnue/nets/d40-uho-720sb.cenn`
  (sha256 `68f070a2791a0df47a45044f72b2f50b1965e42ec24dd01a57554cc0c56245ac`,
  network_scale 400)
