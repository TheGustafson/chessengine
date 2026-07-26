# nnue-pytorch trainer study — what SF's training stack does that ours doesn't

**Date:** 2026-07-25. **Source:** `/home/nick/projects/nnue-pytorch` (user-cloned),
read-only comparative study vs `jax/archon_jax/` + `tools/nnue/` + `src/eval/nnue/`.
**Ideas only — never copy code (GPLv3).**

The checkout is a *modernized* nnue-pytorch: Lightning removed in favour of a
hand-rolled `SimpleTrainer`, `tyro` dataclass config, `RangerLite` default
optimizer, `docs/nnue.md` documents SFNNv16. It already ships
**HalfKAv2_hm + Full_Threats** — the same feature stack we run — so the
comparison is unusually apples-to-apples.

## 1. Recipe deltas

| Knob | SF | ARCHON |
|---|---|---|
| Optimizer | RangerLite (AdamW core, eps 1e-7) + PNM + Lookahead(α .5, k 5) | bullet_adamw, eps 1e-8, ±1.98 clip every tensor |
| LR / schedule | 8.75e-4, **StepLR γ 0.992/epoch** (≈0.040× over 400) | 1e-3 / 5e-4, cosine to 0.027× at `lr_end_superbatch` |
| Weight decay | **0.0 by default; biases in separate groups, never decayed** | 0.01 on everything incl. biases + FT |
| Grad clip | **global norm 2.0**, hard-coded | none |
| Batch / length | 16384; 400–800 "epochs" of 100M ⇒ **40–80 B presentations** | 8192; 240 sb ≈ 6 B per leg |
| Averaging | Lookahead slow weights are the inference weights (+ optional SWA) | EMA 0.99984 (convergent — not a gap) |
| NaN guard | `TerminateOnNaN`, non-zero exit | none |

**How SF decides a net is done — not by loss.** `run_games.py` runs
*concurrently with training*: converts every checkpoint, plays it vs a fixed
`master` anchor (UHO, 20k nodes, 200 g/round, c-chess-cli), ranks with
`ordo --anchor=master`, and picks the next nets to play by **UCB:
`rating + 1.5 × error`**. `delete_bad_nets.py` keeps top-16 and deletes the
rest. **The shipped net is the ordo-argmax checkpoint, essentially never the
last one.** `easy_train.py` also runs 2 seed replicas per GPU into one pool.

## 2. The loss (`model/nnue.py:39`)

```
score = remap_tablebase_score(score, base=4000, scale=4000, decay=0.85)
q  = (scorenet - 270)/340 ;  qm = (-scorenet - 270)/340
qf = 0.5*(1 + σ(q) - σ(qm))            # draw-aware expected score
pf = same transform on the label with out_offset 270, out_scaling 380
pt = pf*λ + t*(1-λ)                    # t = (result+1)/2
loss = |pt - qf|**2.5   (× optional qp_asymmetry, × optional weights)
```

Deltas that matter:
- **Mate/TB scores are REMAPPED, not dropped** — `|score|≥32000−245` becomes
  `±(4000 + 4000·0.85^plies)`. **We hard-drop `|raw| > 10000` in the D46
  filter**, discarding every mate/TB position: a whole class of "very winning"
  supervision.
- **Double sigmoid**, not plain: the ±270 offset creates a draw band and
  saturates properly. Ours is plain `σ(score/400)`.
- **in_scaling 340 ≠ out_scaling 380**, and net output ×`nnue2score=600` first:
  net-unit↔cp gain and label sharpness are *independent* knobs. Ours is one 400.
- **`pow_exp = 2.5`** (docs: 2.6 also good). Ours is 2.0.
- **λ is RAMPED linearly** (`--start-lambda 1.0 --end-lambda 0.75`): start on
  pure eval, fold in the game result over training. **Ours is constant
  `wdl=0.3` (≡ λ=0.7) from step 0.** (Naming inversion: our `wdl` = `1−λ`.)
- Optional λ jitter (per-sample Gaussian + per-batch AR(1)), asymmetry, and
  sample weighting — all default-off knobs we lack.

## 3. Data pipeline — the skip predicate we don't have

`data_loader/cpp/training_data_loader.cpp:764–1012`, in order:
1. `score == VALUE_NONE` → drop.
2. **Placeholder-zero heuristic** (thread-local look-back: decisive game, last
   score >100, this score exactly 0 → drop). Guards packer bugs.
3. Hard early-ply cut (default off).
4. `random_fen_skipping` (canonical recipe **3** ⇒ keep 1 in 4) — decorrelates
   positions within a game.
5. `filtered` (**ON**): drop if best move is a capture, or stm in check.
6. `wld_filtered` (**ON**): **probabilistic** drop with prob
   `1 − P(result | score, ply)` from the fishtest win-rate model (with a
   deliberate `b *= 1.5` tweak, *"deviating from fishtest results, but yielding
   improved training results"*). Thins positions whose label disagrees with the
   game outcome.
7. `simple_eval_skipping` (off): drop near-balanced.
8. **Soft early-ply ramp** (default 20): accept prob 0.10 @ply0 → 0.25 @10 →
   0.75 @18 → 1.0 @20. Openings *thinned*, not deleted. (Ours: hard ply≥16 cut.)
9. **Adaptive piece-count balancing**: Hermite spline target
   `y=[0,0.4,1,1,0.75]` at pc 0/8/16/24/32, thread-local histograms re-estimated
   with exponential forgetting.

**Shuffling:** each worker fills a **1,048,576-entry post-filter buffer**,
`std::shuffle`s it, pushes to a ring buffer; files interleaved at chunk level.
**Ours:** corpora pre-filtered + pre-shuffled on disk, loader yields
**contiguous 8192-record windows at shuffled offsets** ⇒ *our batch composition
is frozen for the lifetime of the corpus* — record i and i+1 share a batch on
every epoch of every leg, forever.

**Epochs:** infinite cyclic stream; 40 B presentations over a few-B corpus =
each position seen 10–40× with different batch partners each time.

## 4. Quantization + architecture

SF: `L1=1024`/persp (+8 PSQT columns per feature row), `L2=32`, `L3=32`, 8+8
buckets by `(pc−1)//4`. FT clamp `[0,255/256]` (**255 not 256, "for safe
squaring within i16"**) → pairwise multiply of halves. At **both** l1 and l2 the
output is concatenated `[clip(x²·c), clip(x)]`, doubling width; output layer
takes `2·L2+2·L3=128`, **plus a linear bypass** (last two l1 units subtracted,
added to the output) **plus** the PSQT term.
Constants: `nnue2score=600`, ft one 256 / max 255, hidden one 128, weight scales
l1 128 / l2 64 / out 128, out-bias 16, **threats at 256** ⇒ training clip
±127/256≈±0.496. Correction factors are engineered to equal exactly 1.0.

**Per-layer weight clips derived from each layer's scale** — `l1 ±127/128`,
`l2 ±127/64`, `out ±127/128`, `threats ±127/256`, **no clip on the i16 FT** —
with `virtual_params` handling so the clip applies to the *merged* weight.
**QAT with straight-through estimators is ON by default** (activations `floor`
matching the inference bitshift; weights `round`; output layer's integer
arithmetic emulated exactly).

## 5. ★ Measured: our uniform ±1.98 clip is binding almost everywhere

On a live net (`d47p-vrepair2-sb80-raw.bin`):

```
l0_pstw    max 1.9800 (saturated)   0.09% > 0.496
l0_threatw max 1.9800 (saturated)   0.10% > 0.496
l1w        max 1.9747               0.12% > 0.496
l2w        max 1.9783               1.5%  > 0.496
polmovew   max 1.9803 (saturated)  31%    > 0.496
polmoveb   max 1.9803 (saturated)   6.8%  > 0.496
l3w        max 1.7512              27%    > 0.496
```

The inherited bullet ±1.98 is active on nearly every tensor — **including
tensors exported as f32** (`l2w`, `l3w`, `l1b`, `polmoveb`) where there is no
quantization reason to clip at all, and on the **i16 FT** whose real ceiling is
~128. Only the i8-at-64 tensors genuinely need it. It has become a silent
capacity constraint.

Also: **our threat table is quantized 4× coarser than SF's** (64 vs 256) while
the engine rescales `×255/64` with rounding (`sota_head.rs:148`). With ~100–160
active threat features, rounding noise accumulates to ≈`√130·0.5/64 ≈ 0.09` FT
units (~20 quanta of a 255-quantum range). Only 0.095% of our threat weights
exceed SF's ±0.496 — tightening the clip and raising the scale is free precision.

## 6. Tooling we lack outright

`ftperm.py` (FT permutation, see below) · `cross_check_eval.py` (engine↔trainer
per-position eval parity, error percentiles 50/75/90/95/99 + worst-offender FEN
dump) · `perf_sigmoid_fitter.py` (**fit your own** eval→WDL `k` from (eval,
result) data instead of inheriting 400) · continuous UCB gauntlet + ordo +
auto-prune · `do_plots.py` (tfevents loss ∪ ordo Elo on one axis) ·
`visualize*.py` (weight/activation histograms vs a reference net) · device/kernel
parity tests (CPU/CUDA/MPS FT parity, fused-vs-reference, sparse-linear, DDP
loader, NaN abort, e2e smoke) · `serialize.py --out_sha` + embedded description
+ leb128 FT compression.

**`ftperm.py` in detail** (directly portable): gathers the post-pairwise-multiply
FT activation matrix over ~1M positions (`fake_quantize_acts=True`), greedily
searches a permutation maximizing **all-zero 4-element blocks**
(`ZERO_BLOCK_SIZE=4`) via independent 2-swaps then 3-cycles until no
improvement; `ft_permute_impl` applies it to FT weight columns *and* L1 weight
rows, **preserving the pairwise pairing by extending the permutation with
`x + l1_size//2`**. Exactly loss-preserving. Our `find_nnz` skips exactly
4-byte groups (`sota_simd.rs:188`) — same cost model, and we have no equivalent.

## 7. `docs/nnue.md` quotes worth keeping

- Factorizer: *"it seems to only be relevant in the early stages… quickly
  becomes unimportant and in later stages can be removed to gain some training
  speed."* (We carry `l0_factw` through every warm leg.)
- Loss exponent: *"the exponent can be >2. Higher exponents give more weight
  towards precision at a cost of accuracy. Stockfish networks had good training
  results with an exponent of 2.6."*
- Clipping: *"the training can diverge from the quantized representation by more
  than just rounding… it is necessary to somehow limit the range for these
  parameters inside the trainer."*
- Threats: *"In a typical midgame position, there might be 3-4x as many changing
  threat features as piece features, and memory bandwidth becomes a
  bottleneck… Because threat feature weights are typically not high in absolute
  value, we can clip them during training with negligible loss."*
- Threat dedup: drop pairs where the attacker's attack set is a subset of the
  attacked piece's (`pb, pq, pk, bq, rq, kq`), drop king-king, keep one ordering
  for same-type pairs. *"Since evaluation is not called in check, attacks to a
  king are also redundant, though accounting for this has not gained."*
- PSQT init: *"initialize it to something that resembles good PSQT values (but
  remember to scale it according to quantization!)"* — SF inits PSQT from HCE
  piece values / `nnue2score`; ours is zero-init.
- Layer stacks: *"It might be worth it to initialize the layers in each layer
  stack identically."* SF copies bucket 0 into all 8; our `init_params` draws 8
  independent randoms.

## 8. Ranked recommendations

1. **Build an `ftperm` equivalent.** Pure NPS, exactly loss-preserving (verify
   bit-identical eval on a FEN suite). ~1 day. **Highest EV — and it is what SF
   uses *instead of* our λ penalty.** Already scheduled post-SPSA, pre-baseline.
2. **Re-shape the FT-sparsity penalty as GROUP LASSO over the deploy groups**
   (L1 of per-4-group L2 norm / max) instead of flat activation L1. Two defects
   in the current form: it is misaligned with the cost model (engine pays per
   nonzero *group*, not per nonzero activation), and it penalizes *magnitude*,
   shrinking large useful activations — a pure capacity tax. Few lines. Do after
   #1 and re-price λ.
3. **Per-tensor weight clips matched to export encoding.** Drop entirely on
   f32-exported tensors, widen on the i16 FT, keep ±1.98 only for i8-at-64. See
   §5 measurements. Trivial effort; medium risk (the clip is also an unintended
   regularizer — needs a clean A/B, breaks strict bullet parity).
4. **Global grad-norm clip 2.0 + NaN/Inf abort**, and stop decaying biases.
   Free stability insurance.
5. **Threat quantization 64 → 128/255** with a training-time clip on
   `l0_threatw`. 2–4× finer resolution. Deploy-format change.
6. **Stop hard-dropping mate/TB labels — remap them** (`4000 + 4000·0.85^ply`).
   Recovers exactly the "very winning" supervision our nets are weakest in.
7. **WLD-coherence probabilistic thinning** + **soft ply ramp** replacing our
   hard ply≥16 cut. Note it is a *probabilistic thin*, which respects the
   standing "don't filter away good data" rule.
8. **Position-level shuffle buffer** (~1M records) instead of frozen contiguous
   windows. Our batch composition is currently fixed forever across all legs.
9. **Exponential LR decay for chained legs** instead of cosine-to-floor —
   schedule-length-agnostic, composes across warm starts, removes the
   `lr_end_superbatch` foot-gun that has repeatedly bitten the pump chain.
10. **Continuous UCB gauntlet over all checkpoints** (`rating + 1.5·error`,
    ordo-anchored), ship the argmax. We already own `gauntlet.sh` + ordo.
11. **λ ramp (0 → 0.3), `pow_exp` 2.5, and one screen of the double-sigmoid WDL
    transform** with separate in/out scaling. Three canonical settings we differ
    from without evidence. Changes the loss scale ⇒ run paired.
12. Identical init across layer-stack buckets; HCE-value PSQT init. Trivial,
    early-training only — irrelevant while we warm-start.
13. **QAT with STE.** Uncertain benefit; SF defaults it on. Exactly one
    experiment, don't build a program around it.
14. `cross_check_eval`-style engine↔trainer parity report as a standing gate.

**Not worth doing:** porting RangerLite/PNM/Lookahead (EMA already fills the
role; breaks bullet parity + resume machinery) · SWA on top of EMA (redundant) ·
L1-layer factorization (their docs: only matters early; we warm-start) ·
adaptive piece-count rebalancing (we control mix weights explicitly) · copying
their head topology (`[x²,x]` concat + linear bypass — genuinely interesting but
needs new SIMD, new deploy format, full retrain; queue behind everything) ·
PP_3Wide SFNNv16 features.

**Confidence:** §6 is a survey (headers + greps). §7 is grep-sampled from a
3154-line doc. RangerLite internals grepped, not line-verified. §1–§5 and all
measured weight statistics are read directly from source.
