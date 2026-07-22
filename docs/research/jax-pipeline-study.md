# JAX training-pipeline study + working prototype (2026-07-22)

Commissioned question: *"something that works exactly how we have it now but
using JAX and jupyter/.py scripts and our still fast af loaders — it will
allow more easy exploration of model arch."*

Deliverable: `jax/` — a working, parity-proven prototype of the full D47c
training pipeline in JAX, plus this study. All work was CPU-only (the GPU
belonged to the live d47c trainer throughout) and no Rust was compiled.

**TL;DR verdict:** the migration is *technically de-risked* — the prototype
reproduces bullet's checkpoint bytes exactly, its forward pass agrees with the
shipping engine to 3 cp median through netpack, the numpy loader beats the
Rust loader's throughput on pre-converted data, and the CPU smoke trains both
heads end-to-end into netpack-parseable artifacts. The **one unproven claim is
GPU training throughput** (untestable while the box's only GPU runs
production training). Recommended posture: **adopt as the experimentation
sidecar now, promote to the production trainer only after a GPU benchmark
window** (go/no-go framework in §8).

---

## 1. What "exactly how we have it now" turned out to mean

Every semantic below was read from source and is cited where it is
implemented. The landmines called out in the commission are all pinned by
tests or parity tools:

| Semantic | Source of truth | Where replicated |
|---|---|---|
| HalfKAv2_hm index math (704-block, palindromic king buckets, file-mirror orient) | `src/eval/nnue/sota_features.rs:60-125` | `jax/archon_jax/features.py` (`halfka_batch`), pinned by `jax/tests/test_features.py` (Rust hand-computed vectors) |
| FullThreats tables + index (opposite orient polarity, semi-exclusion dedup) | `sota_features.rs:130-348` | `features.py` threat tables + `threat_batch`; slow-scalar cross-check on random boards in tests |
| Threat enumeration victim masks | `sota_features.rs:376-434` == trainer `d40_features.rs:340-413` | `features.py:enumerate_threat_relations` |
| Two-summed-FT graph, bias on the HalfKA tensor ONLY, zero-init threats | `d40_features.rs:948-983` | `model.py:_ft_accumulate` + `init_params` |
| pairwise_mul = first_half × second_half | `bullet compiler builder.rs:390-393` | `model.py:_pairwise` |
| crelu/screlu = clamp(x,0,1) / clamp²  | `compiler pointwise.rs:76-118` | `model.py` |
| select = contiguous chunk of size rows/divisor | `compiler index.rs:85-115` | `model.py:_select` |
| policy head off the bucket-selected POST-screlu L2 bottleneck | `d40_features.rs:1106-1107` | `model.py:head_forward` |
| −30.0 additive legal mask (CE-underflow-safe), zero-mask rows have zero CE | `d40_features.rs:830, 891-931` | `losses.py`, `data.py:legal_move_masks` (frame tripwire included) |
| value target = `wdl·result + (1−wdl)·sigmoid(score·mult/400)` | `d40_features.rs:877-879` | `features.py:map_batch` |
| summed loss × 1/batch adj == mean-loss gradients | `trainer run.rs:163` | `losses.py` (means directly) |
| AdamW: decay `p*=1−d·lr` BEFORE step, **no bias correction**, eps outside sqrt, ±1.98 post-clip on EVERY tensor | `trainer optimiser/adam.rs:31-52` | `optim.py:bullet_adamw` (stock `optax.adamw` is wrong on all four counts — documented in-module) |
| cosine LR keyed on 1-based superbatch, final = 0.3³·initial | `d40_features.rs:1251-1259` | `spec.cosine_lr` / `train.py` |
| raw ckpt = `name\n + u64le count + f32le[]` in BTreeMap order; internal col-major == (fan_in, fan_out) row-major | `trainer weights.rs:95-110, 404-431` | `checkpoint.py` |
| quant fmt order (policy head BETWEEN threats and l1), transposed saves, i16(255)/i8(64), round-half-AWAY-from-zero, ≤63-byte "bullet" pad | `d40_features.rs:1295-1314`, `weights.rs:158-334` | `checkpoint.py:quant_bytes_iter` (`_round_half_away`; numpy's banker rounding would silently differ) |
| output buckets = piece-count layout [8×,4×,4×,3×,3×,3×,3×,4×] | `d40_features.rs:288-298` | `spec.OUTPUT_BUCKETS_LAYOUT` |
| bulletformat 32-byte record, stm-canonicalized | `bulletformat-1.8.0 src/chess.rs` | `spec.RECORD_DTYPE` |
| 1858-class policy vocab + canonical frame (hmirror by stored king file) | `src/eval/nnue/policy_vocab.rs` | `features.py` (GEOM_TABLE, underpromos), tests |
| D46 data contract: filter at the layer with raw scores; bulletformat path expects PRE-FILTERED corpora | `docs/research/data-recipe-contract.md`, `d40_features.rs:663-673` | `data.py` module docstring; the sfbinpack filter stays Rust-side (§4) |

Architecture confirmed from the live run (consts at `d40_features.rs:227-229`
+ checkpoint byte-size arithmetic + `/tmp/d47c_train.sh`): HL=1280, L2=128,
L3=32, 8 buckets, policy head ON (λ=0.005), batch 8192, bps 3040, 240 SB,
WDL 0.3, MULT 1.0, skip 75%.

## 2. The prototype

```
jax/
  pyproject.toml           uv-compatible; deps: numpy, jax, optax (+chess, jupyter optional)
  archon_jax/
    spec.py        constants + ArchSpec (parameterized shapes) + quant layout + citations
    features.py    batch-vectorized numpy: decode, HalfKA, FullThreats, policy vocab, map_batch
    data.py        memmapped bulletformat loader, threaded prefetch, legal-mask (python-chess)
    model.py       the exact D47c graph, pure-functional JAX (choice justified in-module)
    losses.py      sigmoid-MSE + masked softmax-CE, bullet-mean semantics
    optim.py       bullet_adamw as an optax GradientTransformation (exact kernel)
    checkpoint.py  raw read/write + streaming quant write + dequant (byte-proven)
    train.py       the training loop (superbatches, cosine LR, save cadence)
    export.py      CLI: raw <-> quant
  tools/
    quant_byte_parity.py   parity proof 1 (byte-identical export)
    forward_parity.py      parity proof 2 (full-graph forward on real weights)
    engine_parity.py       parity proof 3 (engine int8 oracle, outer shape replicated)
    fen_probe.py           FEN-level probe (the D40_PROBE_FENS analogue)
    bench_loader.py        loader benchmarks
    train_smoke.py         the CPU training smoke
  tests/test_features.py   Rust test-vector pins + vectorized-vs-scalar cross-checks
  rust-loader-design/      loader candidate (b): PyO3 crate, DESIGN ONLY (no build)
  exploration.ipynb        the arch-exploration demo notebook
```

**Model style decision** (commission asked flax.linen vs equinox): neither —
pure functions over a flat dict whose keys are the bullet tensor names.
Rationale (full version in `model.py`): checkpoint IO becomes an identity map
(no name-translation layer to drift), optax works on any pytree, and an
arch experiment is an edit to one visible pure function. flax/equinox add
module-tree ceremony that actively hurts the parity story at this model size;
wrapping later is trivial if ever wanted.

## 3. Parity evidence (the go/no-go core)

**P1 — byte-identical quant export (strongest single result).**
`tools/quant_byte_parity.py` streamed the REAL final checkpoint of the live
run (`d47c-1280-240sb-sb240-raw.bin`, 432,573,893 bytes = 11 name-tagged f32
tensors + 157 header bytes, exactly as predicted from `weights.rs`) through
the JAX exporter and compared against `...-sb240-quant.bin` written by bullet
itself from the same weights: **all 137,089,088 bytes identical.** This
simultaneously proves the raw parse, the storage convention, the fmt order
(incl. the mid-stream policy head), all transposes, the rounding mode, and
the padding. A single wrong convention anywhere flips millions of bytes.

**P2 — full-graph forward on the real weights.**
`tools/forward_parity.py` (memory-safe compact row-gather; peak ~200 MB) ran
the REAL sb240 weights through `model.forward` on 256 trainer-parity rows
(hardneg corpus):
- raw-weights vs dequantised-quant-weights through the same graph:
  |Δsigmoid| mean 0.0048 / max 0.025 (int8 quant noise ≈ 8 cp median — the
  known cost of the deployed quantisation, now measurable per-change);
  policy argmax agreement 81.6% (argmax flips only where top-2 logits sit
  inside quant noise; unmasked over all 1858 rows).
- value MSE vs trainer targets: 0.0130 on hardneg vs 0.00624 final training
  loss on the UHO train distribution (`/tmp/d47c_train.log` sb240) — the
  expected ~2× on a deliberately-hard off-distribution corpus; sign agreement
  vs labels 97%.

**P3 — deploy-level engine oracle.**
The OFFICIAL sb240 quant was packed with unmodified `tools/nnue/netpack.py`
(D40_ACC=1280 D40_L2=128 D40_L3=32 D40_POLICY=1) and loaded by the prebuilt
sota engine binary; `tools/engine_parity.py` reconstructed 96 corpus
positions as FENs, replicated the engine's outer shape exactly
(`apply_outer_shape`, `src/eval/mod.rs:419-436`: v·(77871+material)/77871,
material = 534·pawns + npm over {781,825,1276,2538}, i64 truncating division),
and compared `eval` output vs the JAX forward ×400:

```
|Δ| p50 = 3 cp, p90 = 9 cp, max = 47 cp; mean bias −1.1 cp; sign agreement 98%
```

Every outlier is a sparse, extreme-eval endgame (|eval| ≈ 1000 cp) — the
signature of the engine's fixed-point pipeline (u8 FT >>9, i8 L1) vs f32,
NOT a graph discrepancy. The JAX graph is the faithful float reference; the
int8 gap is a property of deployment shared by bullet-trained nets.

**P3b — the heldout shadow population (appeared mid-session).**
`data/nnue/heldout/uho-band0-5.bin` (1 M trainer-parity rows the net never
trained on, WITH real packed teacher moves) — the in-distribution numbers:
- value MSE **0.005878** vs the trainer's final training loss 0.006236
  (within 6%, on held-out data — end-to-end target+graph parity);
  sign agreement 98.1%.
- policy (`tools/policy_heldout_check.py`, 512 rows): the frame tripwire
  passed for **every** real packed teacher move (the strongest possible check
  of the canonical-frame math — any orient/vocab bug asserts); masked
  softmax-CE **2.343** vs the trainer's reported 2.424 on-train; masked
  top-1 vs the teacher move 30.7% (random = 3.9% at 25.8 avg legal moves) —
  the λ=0.005 co-trained head's genuine quality. (Unmasked argmax gives a
  misleading 2.4% — it lands on illegal rows; always evaluate policy masked.)

**P4 — training smoke (CPU, scaled arch).**
`tools/train_smoke.py`: 10,240 real rows, ArchSpec(hl=256, l2=32, l3=16,
policy on), synthesized teacher moves + real legal masks (every mask passed
the trainer's frame tripwire — validating the canonical-frame math
end-to-end), 400 jit'd bullet-AdamW steps:
value MSE 0.0602 → 0.0009, policy CE 3.211 → 0.338 (clean co-trained
overfit), raw checkpoint round-trips byte-faithfully, quant round-trip error
≤ 1/128 as bound, and **unmodified netpack.py parsed the JAX-trained quant**
into .cenn + .cpol.

## 4. The loader question ("still fast af")

Reference: the Rust stack streams the 40 GB sfbinpack at ~200k pos/s with 6
threads (decode + filter + feature mapping), and the live GPU consumed ~210k
pos/s (8192×3040 per ~2-minute superbatch).

Measured on this box (`tools/bench_loader.py`, hardneg corpus, batch 8192):

| Stage / candidate | pos/s |
|---|---|
| (a) numpy: read+decode (occ/pcs → piece lists) | 1,105,000 (1 thread) |
| (a) numpy: HalfKA indices (both perspectives) | 2,045,000 (1 thread) |
| (a) numpy: FullThreats indices (both perspectives) | 260,000 (1 thread) — the bottleneck |
| (a) numpy: FULL map_batch (features + targets) | **206,000 (1 thread)** |
| (a) numpy: halfka-only map_batch | 1,111,000 (1 thread) |
| (a) threaded loader end-to-end | 346k (2 w) / 546k (4 w) / **640k (6 w)** |
| (b) PyO3 bindings | designed, not built (GPU guardrail) — see `jax/rust-loader-design/` |
| (c) subprocess batch streaming | design sketch below; not prototyped |

**Recommendation: (a) numpy structured-dtype reader over bulletformat .bin**,
with the sfbinpack→bulletformat conversion remaining a one-time Rust
preprocessing step. Grounds:
- Single-thread full extraction already equals the entire 6-thread Rust
  pipeline; 6 workers give 3× headroom over current GPU consumption, before
  numba or fancier scheduling.
- Zero new build artifacts; pure `uv run`.
- Feature extraction placement: **in the loader (numpy), not in JAX** — the
  gather-based alternative (feeding raw records into jit) would put
  bit-twiddling into XLA where it is worst, and sparse-index tensors are the
  natural loader/model interface (they are exactly bullet's mapper output).
- Candidate (b) remains the right *second* step, for two things numpy cannot
  give: direct sfbinpack STREAMING (no 32-byte-record conversion pass, no
  duplicated 40 GB on disk) and the engine's real `generate_legal` for the
  policy mask (the python-chess prototype manages only ~5-15k pos/s/thread —
  fine for smokes, not for full policy runs; it is the one component below
  reference speed). The crate skeleton + API is written
  (`jax/rust-loader-design/`); building it is a build-window item.
- Candidate (c) (trainer grows `--dump-mapped-batches` piping ready sparse
  batches over stdout/shm) is the least attractive: it keeps the full Rust
  trainer in the loop as a sidecar process, couples JAX batch layout to a
  private protocol, and saves nothing over (b) once (b) exists. Design
  sketch: length-prefixed frames of `{hk[i32;B×32], th[i32;B×160]×2, bucket,
  target, pol_target, pol_mask-bitset}` over a pipe; ~1.3 MB/frame at B=8192;
  trivially fast enough, just architecturally inferior.

**The conversion step** (needed before (a) can see official binpacks): the
trainer ALREADY contains it — the `D40_HELDOUT_DUMP` path
(`d40_features.rs:727-758`) streams the filtered, stm-canonicalized reader to
a raw bulletformat .bin. A conversion mode is that code with the training
keep-test instead of the band filter (~5-line env-gated addition, e.g.
`D40_CONVERT_DUMP=out.bin` using `HELDOUT_BAND=None` semantics + the packed
policy move that the sfbinpack loader already fills in `extra`). Runs at
loader speed (~200k pos/s ⇒ ~90 min for one pass of the 1.1 G-row binpack;
+35 GB disk). Build-window item; no new tool needed.

## 5. What gets easier (the migration's upside, now demonstrated)

- **Arch exploration** (`exploration.ipynb`): widening the bottleneck is a
  1-line ArchSpec change; a co-trained WDL head is ~10 lines (params dict +
  loss term) with the optimizer/exporter untouched. The bullet equivalent is
  a Rust const edit + full rebuild + hand-reviewing every quant-layout
  consumer. The d40_features.rs history (WDL head, EXP-UNC-1's three
  variance-head variants, factorization, frozen-tensor training) is exactly
  this class of experiment, each paid for in Rust.
- **Optimizers/schedules**: optax drops in against the same pytree
  (`bullet_adamw` proves the interface); schedule-free AdamW, warmup-cosine,
  per-tensor LR groups, EMA weights are each a few lines.
- **Held-out evals**: `forward` + a memmap; no trainer rebuild, no GPU
  tenancy (P2 ran the full 108 M-param forward on CPU in ~200 MB via compact
  gather).
- **Introspection**: activations, gradient norms, per-bucket stats are
  ordinary jnp values in a notebook, not println-debugging inside a CUDA
  kernel graph.
- **Multihead/multitask**: the shared-bottleneck branch point (`hl2`) is a
  returned value; new heads do not touch the trainer core.

## 6. What gets harder / riskier

- **GPU training throughput is UNPROVEN** — the deliberate gap. bullet's CUDA
  path is a hand-tuned sparse-gather trainer; XLA must match it with
  `jnp.take`-based gathers + XLA:GPU fusion at batch 8192 over 83k-dim sparse
  inputs. Plausible (embedding-lookup shapes are XLA bread and butter) but
  unmeasured on the 4060 Ti. **This is the promotion gate** (§8). CPU
  evidence for scale only: full map+step at hl=256 ran 4.3k pos/s on CPU —
  irrelevant to the GPU question except as a correctness vehicle.
- **Export fidelity is now a python responsibility.** An NPS-critical quant
  bug (wrong transpose, banker's rounding, layout drift) would ship a subtly
  wrong net. Mitigation exists and is cheap: `quant_byte_parity.py` (against
  a bullet-written reference while bullet remains available) + the engine
  oracle (`engine_parity.py`, p50 3 cp band) must run in any gate script that
  packs a JAX-trained net.
- **Two trainers during transition** = recipe drift risk (the exact failure
  class D46 exists to kill). Mitigation: spec.py is the single constants
  file, cited line-by-line; any bullet recipe change must be mirrored there
  consciously.
- **Ecosystem drift**: jax/jaxlib move fast; uv pinning (pyproject) +
  a frozen lockfile at promotion time is mandatory. bullet, by contrast, is
  a vendored checkout the project controls.
- **The policy legal mask** needs Rust movegen for full-speed policy runs
  (python-chess ~10k pos/s). Covered by loader phase 2 (PyO3) or a
  precomputed mask sidecar during conversion.
- **Init is distribution-parity, not bit-parity** (bullet's Xoroshiro128Plus
  draw order not replicated). Irrelevant for warm starts/resumes (raw
  round-trip is exact); a from-scratch A/B of JAX-vs-bullet will never be
  seed-identical, so comparisons must be strength-based (as they already
  are).

## 7. Phased adoption plan

- **Phase 0 (done, this branch):** prototype + parity tools + this study.
- **Phase 1 — sidecar (no GPU needed):** use the JAX stack for what it is
  already better at: held-out evals of bullet checkpoints, arch screens on
  CPU/small scale, loss-surface probes, notebook analytics. Bullet remains
  the production trainer. Convert one mid-size corpus via the D40_CONVERT
  dump at the next build window.
- **Phase 2 — GPU benchmark window (the gate):** when a GPU window opens
  (between production runs), run the full-arch training-step benchmark at
  batch 8192 (loader already 3× ahead). Measure pos/s vs bullet's ~210k.
  Also build the PyO3 loader crate in the same window.
- **Phase 3 — shadow training:** one full 240 SB recipe on JAX-GPU; gate the
  product through netpack + net_scale_check.sh + the engine oracle + an SPRT
  vs the bullet twin of the same recipe. Ledger row per D38.
- **Phase 4 — promotion:** JAX becomes the trainer of record; bullet is kept
  pinned for reference-parity checks (its role inverts: the oracle).

## 8. Go/no-go framework

Promote to production trainer iff ALL of:
1. **Throughput** ≥ 0.8× bullet on the same GPU for the D47c recipe shape
   (below that, wall-clock cost eats the exploration dividend; between
   0.8-1.0×, judgement call weighted by how active arch exploration is).
2. **Byte gate**: quant export byte-identical for a warm-started checkpoint
   round-trip (already true) AND engine-oracle band ≤ 5 cp p50 on every new
   arch variant.
3. **Strength gate**: JAX-trained twin of a known recipe within noise of the
   bullet twin at equal time (SPRT, the standing methodology).
4. **Ops**: lockfile pinned; parity tools wired into the gate scripts.

Abort/park signals: XLA sparse-gather throughput < 0.5× after reasonable
tuning (custom-call territory — that price cancels the simplicity win), or
recurring jax/jaxlib breakage under the pinned lockfile.

## 9. Session facts for the ledger / reproduction

- Parity target: `data/nnue/experiments/d47c-1280-240sb/ckpt/`
  `d47c-1280-240sb-sb240-{raw,quant}.bin` (read-only), the FINAL superbatch
  of the live D47c run (training completed during this session).
- Engine oracle: prebuilt `target/release/chessengine` (sota build,
  2026-07-21 19:47) + unmodified `tools/nnue/netpack.py`.
- Bench corpus: `data/nnue/hardneg/corpus/corpus.filtered.shuf.bin`
  (1.75 M rows, trainer-parity filtered, no packed moves); heldout:
  `data/nnue/heldout/uho-band0-5.bin` (1 M rows, packed moves present).
- All commands ran under `uv run --no-project --with jax --with optax`
  (+ `--with chess` for masks) with `JAX_PLATFORMS=cpu
  XLA_PYTHON_CLIENT_PREALLOCATE=false`; jax 0.11.0.
- No cargo build was performed; the PyO3 crate and the sfbinpack converter
  are design-only build-window items.
