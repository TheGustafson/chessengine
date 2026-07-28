# D41 Phase-1 allocation-head trainer — v0 smoke (2026-07-17)

Status: v0 landed on branch `d41-trainer-v0`. Deliverable is the trainer
`tools/nnue/bullet/d41_alloc.rs` + this note. Not promoted; this is the
fail-fast steps 1 (tiny deterministic smoke) and 5 (small model with
calibration/ranking metrics) of design/19 §7's Phase-1 order.

## What was built

`tools/nnue/bullet/d41_alloc.rs` — a bullet example (copied into the toolchain
like `d40_features.rs`/`d40_policy.rs`) that trains the design/19 §4.4 Phase-1
legal-move consequence head, v0:

- **Target (design/19 §3, M4 audit):** per-arm VALUE-MARGIN vs the best legal
  move, squashed as `sigmoid(margin_cp / MARGIN_SCALE)` (default scale 200).
  Best arm (margin 0) → 0.5; worse arms decay toward 0. This is per-arm value,
  NOT a hard best-move class — the M4 audit found the median best-vs-2nd gap is
  only 16cp (confirmed below), so argmax labels are soft while per-arm scores
  are reproducible.
- **Architecture (the proven gen-0 shape, not a new invention):** the shared
  D40 trunk (HalfKAv2_hm i16 tensor + FullThreats i8 tensor summed per
  perspective → crelu → pairwise_mul → concat stm/ntm = 768), feeding the same
  rank-8 bilinear from×to head as `d40_policy`'s `bilinear` head
  (`s(m|x)=h(x)^T e(m)`). The 4096 from×to logits are gathered at the observed
  move's cell with bullet's `select`, sigmoid'd, and MSE-regressed onto the
  squashed margin. Everything is mover-relative (trunk from the side-to-move
  perspective; move squares vertically flipped for black), matching the
  `d40_policy` convention. v0 trains trunk+head from scratch; freezing a
  protected value trunk (design/19 §4.2) is a v1 concern.
- **Data path:** streams the flat SoA corpus written by `obs-tool
  export-train` (schema `archon.obs-train.rootmove-value-v1`), one row per
  (position, Complete arm). Uses columns `feats[N,32]`, `stm`, `move_idx`,
  `margin`, `position_id`. Corpus is small (~34k rows) so it loads fully into
  RAM and row-shuffles each epoch.
- **Held-out check:** splits 80/20 **by `position_id`** (all arms of a position
  stay on one side — leakage-safe), trains on 80%, and after training scores
  the held-out 20% at batch-1 through a `ModelEvaluator`, reporting eval MSE +
  ranking metrics vs baselines computed explicitly from the data.
- **Env-configurable** like the other trainers: `D41_DATA`, `D41_MARGIN_SCALE`,
  `D41_RANK`, `D41_LR`, `D41_LR_FINAL_FACTOR`, `D41_SUPERBATCHES`, `D41_BPS`,
  `D41_BATCH`, `D41_THREADS`, `D41_HOLDOUT_PCT`, `D41_SPLIT_SEED`, `D41_SEED`,
  `D41_EVAL_MAX`, `D41_LOG_RATE`, `D41_SAVE_RATE`, `D41_OUT`, `D41_ID`. Saves a
  quantised bullet checkpoint + a raw resume source at the end.

### How to run

```
# 1. build the corpus from .ceobs shards (obs-tool needs --features datagen):
cargo build --features datagen --bin obs-tool          # (into a private target dir)
obs-tool export-train data/obs/train/sample-v1-reckless data/obs/sample-v1-reckless.ceobs

# 2. register + run the trainer in the bullet checkout:
cd .archon-state/toolchains/bullet   # [[example]] d41_alloc registered in crates/bullet_lib/Cargo.toml
D41_DATA=$REPO/data/obs/train/sample-v1-reckless \
D41_SUPERBATCHES=1 D41_BPS=300 D41_BATCH=4096 D41_HOLDOUT_PCT=20 \
cargo run --release --features cuda --example d41_alloc
```

## Corpus stats (sample-v1, 1000 positions / 33,909 arm rows each teacher)

- margin: mean −300cp, median −157cp, p5 −603, p95 −11, min −31117 (mate).
- **median best-vs-2nd-arm gap = 16cp** — reproduces the M4 audit's 13–16cp
  finding; the top of the ranking is intrinsically soft.
- ~35.5 Complete arms per position (min 2, max 57); only ~3% of rows are the
  best arm.

## Smoke results (300 batches × 4096, LR 1e-3, 80/20 position split)

Train loss dropped monotonically far below the trivial `predict-zero-margin`
baseline (0.5 everywhere), i.e. the model LEARNS:

```
train zero-margin baseline MSE = 0.0734
batch    0:  0.0959   (≈ init, ≈ baseline)
batch   25:  0.0079
batch   50:  0.0030
batch  100:  0.00080
batch  200:  0.00021
batch  275:  0.000095   (≈ 770× below baseline; heavy overfit on 789 positions)
```

Held-out (20% of positions, ~211 pos / ~6990 arms), two teachers:

| teacher  | model MSE | zero-margin base | vs base | optimal-const base | vs base | top-1 (rand) | mean Pearson(pred,margin) |
|----------|-----------|------------------|---------|--------------------|---------|--------------|---------------------------|
| reckless | 0.0209    | 0.0708           | −70.4%  | 0.0228             | −8.4%   | 0.100 (0.042)| **0.485**                 |
| sf18     | 0.0215    | 0.0660           | −67.5%  | 0.0237             | −9.5%   | 0.128 (0.042)| **0.493**                 |

Reading the numbers:

- **MSE beats the trivial zero-margin baseline by ~70% on held-out data** — the
  headline the task asked for. It also beats the *optimal-constant* baseline
  (the mean target, the best a model that learned nothing position-specific
  could do) by 8–9%, so the gain is genuine position-specific generalization,
  not just fitting the marginal target distribution.
- The 8–9% MSE edge understates the signal: held-out MSE is dominated by the
  many clearly-bad arms (heavy negative margins) that the constant already
  fits. The **per-position Pearson(pred, margin) ≈ 0.49 on unseen positions**
  is the honest headline — the model's predicted per-move values track the
  teacher's true per-arm margins at ~0.49 correlation for positions it never
  saw. That is a strong ordering signal for a from-scratch v0 on 789 positions.
- **top-1 best-move accuracy 0.10–0.13 vs 0.042 random** (2.4–3.0× random) is
  the weakest cell — exactly as expected, because THE best vs the near-best is
  the 16cp-soft regime the M4 audit flagged. The model doubles+ random even
  there; the useful allocation signal lives in the full ranking, not the
  razor-thin top.
- **Both teachers behave near-identically** (MSE within 0.6%, Pearson within
  0.01) — good evidence the data path is teacher-agnostic for move quality
  (design/19 §7); reckless and sf18 rows are interchangeable at this granularity.

## Data-path issues found and fixed

1. **The `data/obs/train/sample-v1-*` NumPy corpora did not exist** — only the
   `.ceobs` shards were present, and the prebuilt `target/debug/obs-tool` was
   stale (pre-`export-train`). Built a fresh `obs-tool --features datagen` into
   a private target dir (the shared `target/` is owned by the running labeling
   job — never touched) and ran `export-train` to produce both corpora
   (33,909 rows / 1000 positions each, exactly as advertised).
2. **`stm`/perspective handling.** `d40_policy` fed White-always because its
   Lc0 data was pre-normalized to white-mover. The rootmove corpus stores
   ABSOLUTE features + an explicit `stm` column, and per-arm scores are
   STM-relative. So the trunk must be built from the side-to-move perspective
   (STM slot = `for_each_halfka(board, stm_color)`) and the move squares
   vertically flipped (`sq ^ 56`) for black — otherwise the trunk frame and the
   target sign convention disagree. Implemented mover-relative throughout.
3. **Bilinear cell layout.** Reused `d40_policy`'s discovered fact that bullet's
   64×64 matmul reshape is linear-`to*64+from`, so the `select` index for a
   move (from,to) is `to*64+from` (not `from*64+to`). Verified consistent with
   the corpus `move_idx = from*64+to` convention.
4. No bugs in `export-train` itself: dedup, tier fallback, ref-best, and the
   NumPy columns round-tripped exactly (feats reconstruct through
   `board_from_features`; probabilities/skip stats all zero on the clean
   sample shards).

## Assessment: is the corpus signal learnable?

**Yes.** On held-out positions the v0 model (a) beats the trivial
zero-margin baseline by ~70% MSE, (b) beats the optimal-constant baseline
(so the gain generalizes and is position-specific), and (c) reaches ~0.49
mean per-position rank correlation with the teacher's per-arm margins — a clear
ordering signal from a from-scratch rank-8 head on only 789 training positions.
The per-arm value-margin target (M4 mandate) is the right call: the ordering is
learnable even though the top-1 is soft (16cp median gap). Reckless and sf18 are
interchangeable at this granularity.

## What v1 needs

- **More data.** 789 train positions overfit hard (train MSE 1e-4 vs held-out
  2e-2). The 10k `big-v1` corpus arriving tonight is the first lever; expect the
  MSE-over-constant edge and the Pearson to widen materially. This mirrors the
  B4 "data volume was the constraint" finding on the value side.
- **Freeze a protected value trunk (design/19 §4.2).** v0 trains the trunk from
  scratch; v1 should init the trunk from the selected binpack-trained evaluator
  and freeze it, training only the bilinear head (or a small adapter), so the
  head learns allocation on top of a strong, bit-identical value representation
  and the eval-degradation risk (D40 lesson) is avoided.
- **Ranking-first objective / metrics.** MSE under-weights the strategically
  important near-best regime. Consider a per-position listwise/pairwise ranking
  loss (or a margin-weighted MSE) and promote on top-k + pairwise-order metrics
  (design/19 §8.1), with MSE as a secondary calibration check.
- **hm-mirror consistency.** v0 applies only the vertical flip to move squares
  while the HalfKA trunk also applies the horizontal king-mirror (hm), same
  approximation `d40_policy` shipped. Worth measuring whether hm-consistent move
  encoding helps once data is larger.
- **Context features (design/19 §5.2).** v0 uses (position, move) only. The real
  Phase-1 head consumes search context (remaining depth, window, node type, TT);
  those columns are not yet in this corpus schema — a schema extension, tracked
  in design/19 §10 item 4.
- **Multi-teacher discipline.** Keep reckless/sf18 rows separated by
  `producer_engine` and run teacher-domain ablations (design/19 §7); do not pool
  blindly even though they look interchangeable here.
```
