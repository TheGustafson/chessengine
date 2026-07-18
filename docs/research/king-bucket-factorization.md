# King-bucket FACTORIZATION for the D40 HalfKA trainer (task #27)

Date: 2026-07-18. Owner doc for the `D40_FACTORIZE=1` trainer path. Motivation:
`what-holds-archon-back-2026-07-18.md` finding #2 — we train the 32-bucket
HalfKA tensor UNfactorized, so every `(piece, square)` weight is learned 32×
with no gradient sharing across king buckets. SF nnue-pytorch / Integral /
Viridithas all factorize. Expected **+10–30 fixed-node**, **zero inference
cost** (folded away at export), one-box-verifiable. The single most
under-exploited eval lever.

## What was implemented (env-gated, default byte-identical)

All changes live in `tools/nnue/bullet/d40_features.rs` (the bullet example,
registered at `crates/bullet_lib/Cargo.toml` → the repo source directly). Gated
on `D40_FACTORIZE` (default `0`). When unset, the graph/dims/RNG/save format are
byte-identical to before — the existing 240sb/720sb recipes reproduce exactly.

When `D40_FACTORIZE=1`:

1. **Widened HalfKA FT.** `l0_pst = new_affine("l0_pst", HALFKA_INPUTS + 704,
   HL)`. The extra 704 columns are the bucket-agnostic factoriser (piece-square
   only, king-bucket-free). `HALFKA_INPUTS = 22528 = 32 buckets × 704`.
2. **Doubled active count.** The `halfka_stm`/`halfka_ntm` sparse streams get
   `2 × HALFKA_MAX_ACTIVE = 64` capacity; the FT init effective size becomes 64
   (`init_with_effective_input_size(64)`) so the initial accumulator variance
   AND the merged-weight distribution at t=0 match the non-factorized net (each
   column ~N(0, 2/64); the merged real+factor sum has the same variance as one
   non-factorized column ~N(0, 2/32)).
3. **Mapper emits real + factor.** For each real HalfKA feature `f`
   (`0 ≤ f < 22528`), the mapper writes `f` into the halfka stream AND its
   factor `HALFKA_INPUTS + (f % 704)` into the SAME stream, so the accumulator
   sums `real_weight[f] + factor_weight[f % 704]` — exact factorization. The
   `-1` sentinel fill is extended to the doubled capacity. Threat features are
   king-agnostic and are NOT factorized (untouched).
4. **Merge at export (`merge_factoriser`).** A `SavedFormat` transform on
   `l0_pstw`, run BEFORE `round()/quantise::<i16>(255)`, folds the factoriser
   into the real FT:

   ```
   merged[f*HL + n] = trained[f*HL + n] + trained[(HALFKA_INPUTS + f%704)*HL + n]
       for f in 0..22528, n in 0..768
   ```

   output length `22528 × 768` (the 704 factoriser rows dropped). The emitted
   quant checkpoint + `.cenn` are shape-identical to a non-factorized net, so
   **`netpack.py` needs no change** and inference cost is unchanged.

### Tensor-orientation evidence (why the merge indexes `f*HL + n`)

`new_affine(id, input, output)` stores `l0_pstw` with logical shape
`(rows=HL, cols=input)` **column-major** (`SavedFormat::transpose_impl` reads
`weights[rows*j + i]`), which linearizes to **feature-major**:
`l0_pstw[feature*HL + neuron]`. Three independent confirmations:

- `netpack.py pack_d40`: `pst_w = np.frombuffer(...).reshape(HALFKA_DIMS, acc)`
  and copies it straight through — feature-major `[feature][neuron]`, no
  rearrange, no transpose flag on `l0_pstw`.
- The engine's `sota_features.rs::add_column` reads `weights[index*ACC ..
  index*ACC + ACC]` — feature-major, stride `ACC = HL = 768`.
- `d40_features.rs::load_quantised_weights` sets `l0_pstw` directly (no
  `transpose_saved`, unlike `l1w`), i.e. bullet's INTERNAL `l0_pstw` == the
  on-disk feature-major layout.

The `SavedFormat` merge runs on that internal feature-major buffer, so the merge
indexes `f*HL + n` (real) and `(HALFKA_INPUTS + f%704)*HL + n` (factor). This is
the two-FT manual reimplementation of bullet's
`Factorised::merge_factoriser` (`bullet_lib/.../inputs/factorised.rs`), whose
own layout is `unmerged[layer_size*feat + idx]` with `layer_size = HL`.

### Warm-start / init interaction

- A factorized graph expects the WIDENED `l0_pstw` (`(22528+704) × 768`). It can
  resume only from a **factorized** raw (`D40_INIT_RAW`, matching size). Warming
  from a NON-factorized net (or a merged quant) size-mismatches and is rejected
  by `ModelWeights` — wiring a "load real columns + zero-init the 704 factoriser
  columns" path is possible but NOT implemented; **the factorized recipe trains
  FRESH.** `D40_INIT_QUANT` is unusable with `D40_FACTORIZE` (its `l0_pstw`
  reader is the merged, non-widened shape).
- Same `D40_SEED` does NOT give a bit-identical init to the twin (the wider
  `l0_pstw` consumes more RNG, shifting all later tensors' draws). The A/B is
  method-fair (same seed/data/LR/WDL/hyperparams), not bit-identical init —
  inherent to reparametrization, not a defect.

## Smoke verification (2026-07-18, RTX 4060 Ti, this box)

- Build: `cd .archon-state/toolchains/bullet && CUDA_PATH=/usr cargo build
  --release --features cuda --example d40_features` — clean.
- Factorized smoke (`D40_FACTORIZE=1 D40_SUPERBATCHES=2 D40_BPS=20` on the UHO
  binpack): trains without shape errors; log shows `king-bucket FACTORIZATION:
  ON (halfka FT 22528+704 inputs, 64 active/persp; folded to 22528 at export)`;
  loss decreased 0.081 → 0.058. Quant checkpoint **81,354,816 bytes**.
- Non-factorized smoke (same params, `D40_FACTORIZE` unset): NO factorization
  line; quant checkpoint **81,354,816 bytes** (identical) — default path
  unchanged.
- `netpack.py ... d40` on the factorized quant → `.cenn` **81,354,848 bytes**,
  identical to the non-factorized net's `.cenn` (independently computed:
  payload 81,354,784 + 64 header). Merge produced the engine format.
- Engine load (sota build, `setoption name EvalFile value <fac.cenn>` +
  `eval`): loads via the CENN-v2 path ("NNUE"), evaluates finitely and sanely
  (startpos −148 cp for a 2-superbatch undertrained net; a broken merge would
  give garbage or a load rejection).

## Launch recipe — factorized 240sb twin (fixed-node A/B vs `d40-uho-v1-240sb`)

Exact twin of `d40-uho-v1-240sb` (`tools/nnue/runs/d40-uho-v1-240sb/run.json`):
same data, LR, WDL, scale, SCORE_DIV, skip75, seed, 240 SB — the ONLY change is
`D40_FACTORIZE=1` and distinct output IDs.

```bash
cd /root/projects/learnrust/chessengine
CKPT=data/nnue/experiments/d40-uho-fac/ckpt
mkdir -p "$CKPT"

D40_DATA=data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack \
D40_FORMAT=sfbinpack \
D40_FACTORIZE=1 \
D40_ID=d40-uho-fac-240sb \
D40_OUT="$CKPT" \
D40_SEED=198273612 \
D40_SUPERBATCHES=240 D40_LR_END_SB=240 D40_SAVE_RATE=240 \
D40_BATCH=16384 D40_BPS=1520 \
D40_LR=0.001 D40_WDL=0.3 \
D40_EVAL_TARGET_SCALE=400 D40_SCORE_MULTIPLIER=1.0 \
D40_HALFKA_ONLY=0 D40_SF_BUFFER_MB=256 D40_VERIFY_ROWS=4096 \
SFBINPACK_SCORE_DIV=1.9 D40_RANDOM_SKIP_PCT=75 \
CUDA_PATH=/usr \
./.archon-state/toolchains/bullet/target/release/examples/d40_features

# Pack the (already-merged) quant checkpoint → drop-in .cenn (no netpack change):
uv run python tools/nnue/netpack.py \
  "$CKPT/d40-uho-fac-240sb-sb240-quant.bin" \
  data/nnue/nets/d40-uho-fac-240sb.cenn d40
```

(Or via the orchestrator: clone `d40-uho-v1-240sb`'s config, add
`"D40_FACTORIZE": "1"` to `train.env`, set new `net_id`/`D40_ID`/`D40_OUT`, then
`uv run python tools/nnue/train_run.py --config <config.json>
--only preflight,train,verify,pack,sanity`. Note: the `verify`/parity stage
compares the trainer graph — which is factorized — against the MERGED `.cenn`;
they agree only up to i16 quant rounding, so relax `max_abs_quantization_logit_error`
if it trips, or run parity against a factorized `D40_PROBE_RAW`.)

### Data-efficiency-signature run (60sb, full anneal)

```bash
# Same block, changed to a standalone 60 SB cosine:
D40_ID=d40-uho-fac-60sb  D40_OUT=data/nnue/experiments/d40-uho-fac/ckpt60 \
D40_SUPERBATCHES=60 D40_LR_END_SB=60 D40_SAVE_RATE=60  ... (else identical)
```

## The gate

- **Primary (data-efficiency device):** `d40-uho-fac-240sb` vs `d40-uho-v1-240sb`,
  **300g @5k fixed nodes** (fixed-node, since factorization is zero-cost so
  fixed-node == equal-time here). Promote if **≥ +15** (target of the +10–30
  expected band; note the ±13 one-box resolution floor and 2–3× external
  deflation, `what-holds-archon-back` finding 3).
- **Data-efficiency signature:** `d40-uho-fac-60sb` ≈ `d40-uho-v1-240sb`
  (fixed-node A/B straddling zero) would demonstrate the ~4× data-efficiency
  win — the mechanism's headline claim (gradient sharing across the 32 buckets
  reaches 240sb quality in ~60sb).
- On a win, the factorized net is a **drop-in** replacement (same `.cenn`
  format/size, unchanged runtime); make it the new stage-1 base and re-run the
  longer-stage-1 banking lever (720sb) factorized.
