# NNUE training gap study — ARCHON vs the strongest engines (2026-07-28)

**Question asked:** *what are we doing wrong in NNUE training?* The belief under test is
"we train according to the SF NNUE recipe on billions of positions". The motivating
evidence: our net is **not capacity-limited** (768→1536 width is Elo-neutral at equal
time); a **768-wide Reckless net matches or beats our 1280-wide net inside our own search
at fixed nodes** (~+27 Elo at n=268, provisional); and on frame-free move-ranking quality
**we ≈ Reckless < SF18-static** (Spearman vs SF18 multipv-8: 0.376 / 0.388 / 0.424,
random null 0.05).

---

## Verdict

**The belief is half true, and the wrong half is load-bearing.**

- The **data** really is Stockfish's own public binpacks — 10.19 B positions, every byte
  from `official-stockfish/master-binpacks`. That part is true.
- The **recipe** is not Stockfish's. It is **bullet's 2024-era default recipe**, faithfully
  re-implemented in JAX. Verified line by line: our optimiser is bullet's AdamW kernel down
  to its missing bias correction and its ±1.98 clip; our loss is bullet's
  `(sigmoid(out) − target)²` family; our target is bullet's
  `wdl·result + (1−wdl)·sigmoid(score/400)`; our data filter is *verbatim* the predicate in
  `bullet/examples/simple.rs:96-102`. Where we deviate from bullet we deviate *toward*
  Stockfish (`pow_exp 2.5`, per-tensor clips, no bias decay, grad clip 2.0).
- Stockfish's recipe has moved a very long way since that snapshot, and **four of its most
  consequential current mechanisms have no counterpart in ARCHON at all**: a WDL-calibrated
  two-sigmoid target on *both* sides of the loss; straight-through quantisation-aware
  training; active shaping of the ply and piece-count marginals; and Leela-BT4 label
  distillation. The first is **already implemented in our own trainer and switched off in
  the shipped run**.

**The comparison that matters most is much tighter than expected.** Reckless's net is not a
simpler architecture that beats a fancier one. Read from `reckless_reference/Reckless`
(`src/nnue.rs`, `origin/main`): their net is
`(66,864 threats + 768×10 king buckets → 768)×2 → 16 → 32 → 1` with **8 output buckets in
exactly our bucket layout**, CReLU-into-pairwise-multiply, `L1_QUANT 64`. Ours is
`(60,720 threats + 22,528 HalfKA → 1280)×2 → 16 → 32 → 1`, 8 output buckets, same layout,
same activation chain, same L1 quantisation. **Same head, same activation, same bucket
scheme, comparable input space (74.5 k vs 83.2 k).** The differences that remain are:
accumulator width (768 vs 1280 — which our own standing finding says is Elo-neutral), the
threat-FT quantisation scale (**255 vs our 64**), and the training recipe and data.

So evidence #2 is not an architecture result. **It is a recipe-and-data result**, and that
is where this study points.

**The most surprising measured result here is not a recipe delta at all:** our training
distribution is skewed roughly three times too far toward endgames. Mean piece count 15.6
against Stockfish's target of 19.5; the 31-32-piece output bucket receives 4.25 % of our
data where Stockfish targets 13.95 %. Alexandria measured **+5.08 Elo STC / +2.28 LTC** for
fixing exactly this, as a data-loader change alone. See §4/G2.

**And there is a meta-gap that makes everything else hard to settle:** our net screens run
100-400 games (±15-25 Elo). Reckless screens **every** net at 25,000 fixed nodes over
~40,000 games (±2 Elo) before it ever reaches a time-control test. That is a ~100× power
difference on the exact instrument this document's experiments need, and on our hardware it
is affordable. See §4/G0.

Method note: this follows the standing bar in `CLAUDE.md` — every ARCHON claim cited to
code, every external claim to a URL, and every hypothesis of mine that **died on
measurement recorded as a null** (§5) rather than deleted. Four of my hypotheses died,
including two that would have looked good in a summary.

---

## 1. What ARCHON actually does — from code, not docs

The shipped D48X run: `tools/nnue/runs/d48x_1280.json` driven by `jax/tools/pump_train.py`.
This produced the current canonical net (`engines/canonical/MANIFEST.md`). Where doc and
code disagree, the code wins and the disagreement is flagged.

### 1.1 Loss and target

`jax/archon_jax/losses.py:27-40`; target built at `jax/archon_jax/features.py:542-556`:

```
target = wdl·result + (1 − wdl)·sigmoid(score · mult / scale)
loss   = mean( |sigmoid(value_logit) − target| ^ pow_exp )
```

`wdl = 0.3`, `scale = 400.0`, `mult = 1.0`, `pow_exp = 2.5`.

- `wdl` and `scale` are **not settable from the run config** — `pump_train.py` never reads a
  `wdl` key, so both fall through to `TrainConfig` defaults (`spec.py:300`, `61-62`).
  **λ is 0.3 constant for the whole run: no ramp, no schedule, no jitter.**
- `pow_exp = 2.5` **is** set (`d48x_1280.json:95`) — a deliberate move toward Stockfish.
- `target_offset = 0.0` (absent ⇒ default) ⇒ the **plain sigmoid** branch,
  `features.py:555`. The draw-aware double-sigmoid branch (`features.py:545-553`) exists, is
  MLE-fitted against our own corpora, and is **not used**.
- No asymmetry term, no per-sample importance weight, no mate/TB score remapping.
- `ft_sparsity_lambda = 0.0` — the accumulator L1 penalty is off.

### 1.2 Optimiser, LR, regularisation

`jax/archon_jax/optim.py:1-101` is a deliberate re-implementation of **bullet's AdamW
kernel** (not `optax.adamw`), and it matches bullet's
`crates/trainer/src/optimiser/adam.rs` exactly:

```
p  *= 1 − decay·rate            # decoupled, lr-scaled, BEFORE the step
m   = β1·m + (1−β1)·g           # 0.9 / 0.999, eps 1e-8 OUTSIDE the sqrt
v   = β2·v + (1−β2)·g²
p  -= rate · m / (sqrt(v) + eps)        # NO bias correction
p   = clip(p, −c, +c)
```

ARCHON deviations, all toward Stockfish: weight decay 0.01 **not applied to biases**
(`no_bias_decay: true`, `spec.py:150-155`); global grad-norm clip **2.0**; **per-tensor**
clips derived from each tensor's export encoding (`spec.py:115-147`) rather than one global
±1.98 — f32 tensors unclipped, i16 FT at `32000/255 = 125.5`, i8 tensors at `127/64 = 1.984`.

LR: piecewise-linear on absolute superbatch (`d48x_1280.json:100-113`) — 1e-3 peak →
**hold 5.5625e-4 from sb200 to sb400** → linear anneal to 0 at sb520, 0.25-sb warm-up.
EMA shadow weights (decay 0.99984); **the shipped net is the EMA**.

### 1.3 Batch, superbatch, positions seen

`spec.py:295-297`: batch **8,192**; 3,040 batches/superbatch ⇒ **24.90 M positions per
superbatch**; 520 superbatches.

| | |
|---|---|
| optimiser steps | 1,580,800 |
| **presentations, D48X leg** | **12.95 B** |
| corpus | **10.19 B** |
| **epochs over the data** | **1.27** |
| cumulative over the whole warm-restart chain | **≈ 87 B** (≈3,500 sb across ~20 legs) |

> **Doc-vs-code disagreement.** `d48x_1280.json:114` says "~28.5 B presentations" for a
> 380-sb run. The real figure is 380 × 24.9 M = **9.46 B** — a 3.0× overstatement, because
> the note assumed a 75 M superbatch. Nothing depends on it, but it is wrong where written.

### 1.4 The warm-restart chain, and the optimiser state we throw away

D48X is one link in a ~20-leg chain (58 rows in `ledger/nets.jsonl`), warm-started from the
annealed A3 trunk of the D47S line, itself out of union → anneal → repair → head-rebuild →
v15 legs, back to the original bullet stage-1.

**Every one of those restarts destroys the optimiser state.** `pump_train.py` calls
`opt_state = opt.init(params)` unconditionally on both the fresh and the resume path; only
weights and the EMA are checkpointed (`train.py:333-341`). The code says so
(`spec.py:355-358`: "fresh optimizer state = the accepted ~2sb transient"), and the D48X run
itself was restarted mid-flight at sb200 (`d48x_1280.json:117`).

For contrast: bullet writes an `optimiser_state/` directory into every checkpoint;
Stockfish's `--resume-from-checkpoint` carries optimiser state, scheduler state *and* the
λ-jitter buffer inside the model state dict, specifically so resumption is exact
(`nnue-pytorch model/nnue.py:242-258`, `model/config.py:72-75`).

### 1.5 Quantisation-aware training

**There is none.** The only quantisation-related mechanism in training is the per-tensor
weight *clip*. The forward pass is pure f32 (`model.py:193-253`); nothing simulates the
deployed integer path. Quantisation happens once at export (`spec.py:269-288`): HalfKA FT
i16@255, **threat FT i8@64**, L1 i8@64, everything else f32.

That matches bullet (which has straight-through `faux_quantise` and `clip_pass_through_grad`
primitives in `crates/compiler/.../qat.rs` that **no example uses**). Stockfish does real
QAT — §2.5.

Recorded deployed cost (`docs/research/jax-pipeline-study.md:107-124`, D47c net, same
scheme): raw-f32 vs dequantised-quant `|Δsigmoid|` **mean 0.0048, max 0.025**; engine-int8
vs JAX-f32 `|Δ|` p50 **3 cp**, p90 9 cp, max 47 cp, sign agreement 98 %. Against a heldout
value loss of 0.006350 (RMS 0.0797) that is ~6 % of the residual — real, but second-order.

**New measurement (this study)** — weight-level statistics of the shipped net
(`d48x-1280-sb520-ema-raw.bin`):

| tensor | params | quant | sd(w) | step/sd | frac at \|level\| ≤ 1 | SNR |
|---|---|---|---|---|---|---|
| `l0_pstw` (HalfKA FT) | 28.8 M | i16 @255 | 0.0748 | 0.052 | 20.9 % | 36.9 dB |
| **`l0_threatw`** | **77.7 M** | **i8 @64** | 0.0589 | **0.265** | **54.2 %** (34.8 % → 0) | **23.3 dB** |
| `l1w` | 0.16 M | i8 @64 | 0.0928 | 0.168 | 37.3 % | 26.5 dB |

**72 % of the network's parameters are deployed at roughly 1.5 effective bits per weight.**
Both comparison engines quantise their threat inputs ~4× finer: **Stockfish at scale 256
with a training-time clip of ±0.496** (`nnue-pytorch model/quantize.py:153-173`),
**Reckless at `FT_QUANT 255`** (`reckless_reference/Reckless/src/nnue.rs`). We are the
outlier. (My "just re-export at scale 256" idea was measured and died: §5.2.)

### 1.6 Data — provenance, filtering, mixing

The twelve corpora of `d48x_1280.json:18-67`, built by `tools/nnue/runs/stage5_data_chain.sh`,
registered in `ledger/data.jsonl`:

| corpus | positions | weight | origin |
|---|---|---|---|
| `uho-band75-100` | 2.560 B | 25.12 % | `nodes5000pv2_UHO` — SF self-play, 5000 nodes, pv-2, UHO book |
| `fishpack32` | 1.612 B | 15.83 % | SF self-play binpack |
| `farseerT74-band75-100` | 1.233 B | 12.10 % | Leela T74, SF-rescored |
| `datapv2-band75-88` | 0.990 B | 9.71 % | `data_pv-2_diff-100_nodes-5000` — SF self-play |
| `farseerT75-band75-83` | 0.922 B | 9.05 % | Leela T75, SF-rescored |
| `multinet-band75-88` | 0.913 B | 8.96 % | `multinet_pv-2_diff-100_nodes-5000` |
| `t60t70-band75-85` | 0.838 B | 8.23 % | Leela T60/T70 + wrongIsRight + farseer |
| `pylon-band75-88` | 0.474 B | 4.65 % | `training_data_pylon` — **units never audited** |
| `wisr-band75-95` | 0.355 B | 3.48 % | `wrongIsRight_nodes5000pv2` |
| `wrongnnue-band75-95` | 0.263 B | 2.58 % | `wrongNNUE_02_d9` — **units never audited** |
| `canon-tight-frame` | 27.5 M | 0.270 % | **house**: SF18@50k relabel of UHO-binpack positions |
| `hardneg-frame` | 1.31 M | 0.013 % | **house**: SF18@50k relabel of ARCHON-loss positions |
| **total** | **10.188 B** | | |

All fetched from HuggingFace `official-stockfish/master-binpacks`
(`tools/nnue/fetch_official_stockfish_binpack.py:22-24`). Four consequences:

1. **99.72 % of the signal is Stockfish's public binpacks; 0.28 % is house data; 0.00 % is
   ARCHON-generated.** Even the house corpora are SF18-labelled — `hardneg` is ARCHON
   *positions* with SF18 *labels*. `generator_net` is `null` on every corpus row.
2. **The mix is a plain union.** Measured: every weight equals that corpus's share of the
   total *to four decimal places* (25.1234 % vs 25.1234 %, …, 0.0129 % vs 0.0129 %). Source
   quality plays no part; disk size is the only input. The SF18-relabelled house data — the
   best labels we own — sits at 0.27 % **because it is small**.
3. **84 % of the available UHO data was discarded for disk.** `D40_CONVERT_BAND=75:100`
   keeps a 25 % hash band, `75:88` keeps 13 %; the full UHO binpack is 16 B positions of
   which 2.56 B survive. `stage5_data_chain.sh:11-12` is explicit that the band is "the disk
   knob".
4. **Labels are old-Stockfish shallow search scores** — 5000-node pv-2 and depth-9 sets from
   2021-2024-era SF nets, plus Leela runs rescored by the same. Stockfish abandoned this
   label generation in June 2026 (§2.2). **Note this is not, by itself, a criticism**: the
   nnue-pytorch wiki calls depth 9 "the sweet spot" and warns that "better evaluations don't
   always give better results, as it's a tradeoff between quality and learnability".
   The criticism in G3 is about *which generation*, not about label depth.

**Filtering** (`docs/research/data-recipe-contract.md` §1, applied by `sfbinpack_filter` at
convert time): `ply ≥ 16`, side-to-move **not in check**, `|raw score| ≤ 10000` (pre-clamp),
best move `Normal`, best-move destination **empty** (non-capture). Drop rate 27-37 %.
Filter first, clamp last; only `Exact`/`Complete` bounds may become labels.
**This is bullet's `examples/simple.rs:96-102` predicate verbatim**, and it is also what
Reckless uses (`!in_check && !is_capture && !is_promotion`, `src/tools/rescore.rs`).

**Augmentation:** none — no mirroring, no DFRC/960, no synthetic positions.
**Shuffling:** corpora pre-shuffled on disk plus a 128-batch (1.05 M-record) in-loader pool
(`data.py:149-165`), added 2026-07-26 after the discovery that batch composition had been
*frozen for the life of a corpus*.
**Score frame:** per-pack `SFBINPACK_SCORE_DIV` (1.49-2.25) at convert time onto a common
"/1.9-equivalent" frame. §5.1 measures how well that worked (answer: it worked).

### 1.7 Architecture, buckets, final scale

`spec.py:26-55`, `d48x_1280.json:68-93`:

- inputs **HalfKA 22,528** (32 king buckets × 704) **+ FullThreats 60,720** = **83,248**;
  32 + 160 active per position.
- accumulator 1280/perspective → CReLU → pairwise-multiply → 640 → concat 1280 into L1.
- L1 1280 → **L2 16** → L3 32 → 1 with SCReLU; **8 output buckets** by piece count
  (`spec.py:47-55`); plus output-skip and PSQT lanes.
- king-bucket factorizer during training, folded into the FT at export.
- **~107 M parameters, of which 77.7 M (72 %) are the threat FT.**
- our threat set already excludes king→piece and piece→king pairs
  (`src/eval/nnue/sota_features.rs:169` — the king row of the threat map is all `−1`),
  i.e. we match Stockfish's post-SFNNv11/v12 shape. We do **not** have SFNNv14's
  opposed-pawn threats or SFNNv16's `PP_3Wide` pawn-pair features.
- dead weight shipped: `pol_projw`, `polmovew`, `polmoveb`, `dis_*` are all exactly 0.0
  (both heads disabled mid-design) — ~103 k params.

**Final scale.** `tools/nnue/calibrate_network_scale.sh` iterates the `.cenn` header's
`network_scale` until `net_scale_check.sh` reports `median(cand/ref) ≈ 1.000` against the
previous net, and the check is a hard pre-game gate (band 0.85-1.18). **This is the correct
practice and matches the field** — see the null in §5.4. The known caveat is recorded in
`net_scale_check.sh:26-48`: the check is self-referential, so it pins net-0's arbitrary
scale into every descendant, and our eval sits at 0.538× Stockfish's internal frame while
our cp constants are Stockfish's raw values. Correcting *that* has been tried and refuted
(§5.3); it matters here only as a constraint on how any target-transform change is deployed.

---

## 2. Stockfish's *current* recipe (dated, primary sources)

Read directly: `official-stockfish/nnue-pytorch` @ `9f729465` (2026-07-26);
**`vondele/nettest` @ `ba46c2ac` (2026-07-27)** — since ~2026-03 *this* is where the
executable recipe lives, not nnue-pytorch; Stockfish master @ `23cf5d82` (2026-07-26);
current net `nn-ab28990d4ea3.nnue` (`c8b91a2b`, 2026-07-26); architecture **SFNNv16**
(`f4bcd404`, 2026-07-20).

Two structural notes that invalidate most secondary write-ups: `model.py` no longer exists
at the repo root (the loss is in `model/nnue.py`), and net commits no longer paste a command
line — they point at a nettest PR.

### 2.1 The loss

`nnue-pytorch/model/nnue.py:39-70`:

```python
q  = (scorenet - in_offset) / in_scaling ;  qm = (-scorenet - in_offset) / in_scaling
qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())        # network side → expected score
s  = (score - out_offset) / out_scaling ;   sm = (-score - out_offset) / out_scaling
pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())        # label   side → expected score
pt = pf * actual_lambda + t * (1.0 - actual_lambda)  # t = game outcome
loss = |pt - qf| ** pow_exp
if qp_asymmetry: loss *= ((qf > pt) * qp_asymmetry + 1)
loss = (loss * weights).sum() / weights.sum()
```

Shipped constants (`nettest/threats.yaml:78-102`, nevergrad-HPO-tuned) vs trainer defaults
(`model/config.py:86-111`):

| | default | **shipped** |
|---|---|---|
| `in_scaling` / `in_offset` | 340 / 270 | **295.65 / 285.27** |
| `out_scaling` / `out_offset` | 380 / 270 | **379.99 / 289.13** |
| `pow_exp` | 2.5 | **2.434** |
| `qp_asymmetry` | 0.0 | **0.2287** |
| `nnue2score` | 600.0 | 600.0 |

Three mechanisms with **no ARCHON counterpart**:

- **An offset two-sigmoid W/D/L transform on BOTH sides.** The offset creates the draw band;
  `qf` and `pf` are both *expected scores*, the same quantity as `t`, so the λ blend mixes
  like with like. Because the *network* side has its own scale and offset, the loss absorbs
  a systematic frame mismatch between the net's output units and the labelling engine's —
  precisely the class of problem `CLAUDE.md` records as our 2.07× frame issue.
  **bullet ships the same idea** as `use_win_rate_model()`
  (`crates/bullet_lib/src/value.rs:104-107`), with the same 270/380 constants.
- **`qp_asymmetry = 0.2287`** — overestimation penalised ~23 % harder than underestimation.
- **`remap_tablebase_score`** (`model/nnue.py:19-36`): `|score| ≥ 32000−245` is remapped to
  `4000 + 0.85^plies · 4000` rather than fed to a sigmoid. *(ARCHON drops those rows.)*

### 2.2 Data — the biggest change, and it postdates our corpus

Since **2026-06-25** (`nn-f8a759c05f9f`, commit `9fcd47a7`) Stockfish trains on data
**relabelled by the value head of Leela's BT4 network** (`BT4-tf13tune.pb.gz`) via a forked
lc0 — knowledge distillation, not search scores. `4488343c` (2026-06-29) adds **74 B
relabelled positions**; `9d2ba2be` adds 3.5 B more. In the current `threats.yaml`, **all 44
training binpacks are `.relabel-BT4-tf13tune` versions.**

Those 44 binpacks are *the same families ARCHON uses* — `leela96`, T60…T80,
`nodes5000pv2_UHO`, `wrongIsRight_nodes5000pv2`, `multinet_pv-2_diff-100_nodes-5000`,
`dfrc_n5000`, plus `tb5dtm` — **with every label replaced.**
**ARCHON is training on the pre-relabel generation of the very same files.**

Dataset: `https://huggingface.co/datasets/xushawn/test80-bt4-relabel`; relabeller fork:
`https://github.com/xu-shawn/lc0/tree/relabel_dual_stream_test`.

Context from the nnue-pytorch wiki that keeps this honest: Stockfish's *own* generated data
is d9/n5000 and the wiki calls that the sweet spot; Leela-derived data is described as
"higher quality"; and the documented order is **train on SF-generated data first, then
retrain on Lc0-derived data** — "training solely on the Lc0-derived datasets doesn't produce
as good results". So the lever is a *retrain stage on relabelled data*, not a from-scratch
swap.

### 2.3 Filtering and distribution shaping

`data_loader/cpp/training_data_loader.cpp`, `make_skip_predicate`, in order:

1. `score == VALUE_NONE` → skip.
2. placeholder-zero heuristic.
3. `early_fen_skipping: 18` — hard ply floor. *(ARCHON: 16 ✓)*
4. `random_fen_skipping: 10` → skip probability 10/11.
5. `filtered`: skip if **in check OR best move is a capture**. *(ARCHON: ✓ identical)*
6. **WDL filtering** — probabilistic skip on `1 − score_result_prob()`: drop positions whose
   eval disagrees with the game result. *(ARCHON: absent)*
7. `simple_eval_skipping` (off).
8. **Soft early-ply spline** (new 2026): for `ply < 32`, accept with piecewise-linear
   probability through `(0, 0.025), (22, 0.05), (25.5, 0.20), (29.5, 0.80), (32, 1.0)`.
   *(ARCHON: absent)*
9. **Piece-count rebalancing spline** (new 2026): a 5-point Hermite spline over piece counts
   0/8/16/24/32 with targets `−0.20, 0.45, 1.0, 0.95, 0.75`, positions accepted to match it.
   Motivation, from SF PR #4942: *"an attempt to reduce the skewed dataset piece count
   distribution where there are much more positions with less than 16 pieces."*
   *(ARCHON: absent — and §4/G2 measures that we have exactly that skew.)*

Items 6, 8, 9 are **reweighting, not discarding**. ARCHON's filter is a pure accept/reject
predicate with no shaping at all.

### 2.4 Optimiser and schedule

- optimiser **`RangerLite`** — a 2026 refactor of Ranger21 keeping Positive-Negative
  Momentum and Lookahead (α 0.5, 5 steps).
- **OneCycleLR stepped per optimiser step**, `one-cycle-steps: 4096000`, warm-up 5 %,
  `final-div 1e3`, `lr 0.9e-3`. The `StepLR(gamma=0.992)` older write-ups quote is dead code
  in this configuration.
- **batch 131,072** pretrain / 65,536 finetune; "epoch" = a fixed 134,217,728-position chunk
  of an infinite stream; 3500 + 3750 epochs ⇒ order **10¹² presentations**.
- weight decay per parameter group, **weights only, never biases** *(ARCHON: ✓)*.
- **λ convention is inverted from bullet's and ours**: `λ = 1` is *pure eval*
  (`model/lambda_utils.py:11-14`). Shipped `start = end = 1.0` with a **cosine dip**
  (`lambda-cycle-delta −0.3` over 3.584 M steps) ⇒ **λ: 1.0 → 0.7 → 1.0** — in ARCHON's
  units, 0 % game outcome dipping once to 30 % and back. Plus per-sample Gaussian λ jitter
  and a per-batch AR(1) λ process.
- two stages only (nettest PR #409): pretrain, then finetune on the **same data** at half
  batch with λ pinned to pure-eval and jitter off — *anneal the regularisers away*.
- resume carries optimiser + scheduler + jitter state.
- SPRT gate in-recipe at `nElo_interval_midpoint −0.75` — new nets accepted at a slightly
  **negative** midpoint. (A direct precedent for our D48X promotion, recorded at
  −9.1 [−34.4, +16.1] n=382 in `CLAUDE.md` and at −16.6 [−43.5, +10.2] n=336 in
  `engines/canonical/MANIFEST.md` — two different snapshots of the same match, both
  quoted where they appear below.)

### 2.5 Quantisation-aware training — real QAT, default on

`model/config.py:115-118`: `use_fake_act_quantization = True`,
`use_fake_weight_quantization = True`. `model/quantize.py:36-56` is a **straight-through
estimator** with activations *floored* to match the inference bitshift, and the output layer
emulating the exact integer division path. Clipping is a callback
(`trainer/callbacks.py:67-79`): layer stacks clipped **after every batch**, the feature
transformer **once per epoch**. Ranges: `l1` ±0.9922 @128, `l2` ±1.9844 @64, `output`
±0.9922 @128, **threat FT ±0.4961 @256**, FT activation clamped to [0, 255/256].
Export is hard-gated (`_safe_convert` raises on any clipping); `--ft_optimize` permutes FT
neurons for NNZ block sparsity.

`docs/nnue.md` on the threat clip: *"Because threat feature weights are typically not high
in absolute value…, we can clip them during training with negligible loss in evaluation
quality"* — with a measured **"+10 % on ARM, +5 % on x86"** from the reduced accumulation
bandwidth. So the i8 choice is right; the *scale* is the question.

### 2.6 Architecture — and the direction of travel

`src/nnue/nnue_architecture.h` @ `23cf5d82`: `FullThreats + PP_3Wide + HalfKAv2_hm^`,
**L1 = 1024**, L2 = 32, L3 = 32, 8 output buckets, 8 PSQT buckets.

| arch | date | change |
|---|---|---|
| SFNNv16 | 2026-07-20 | added `PP_3Wide` pawn-pair features |
| SFNNv14 | 2026-04-02 | opposed-pawn features in FullThreats |
| SFNNv13 | 2026-02-18 | L2 → 32 |
| **SFNNv10** | 2025-11-12 | **added FullThreats; L1 3072 → 1024**; FT quant scale 255 |
| SFNNv9 | ←2025-11-12 | L1 = 3072 |

**When threat inputs landed, Stockfish cut L1 by 3×.** Our 1280 accumulator is already wider
than current Stockfish's on the same feature family. The small net was removed entirely on
2026-05-04 (`ab8f901d`), passing as a failed non-regression in both directions despite
−2.5 % NPS. SFNNv16 itself **failed STC** (LLR −2.94) and was accepted on LTC/VLTC/VVLTC.

---

## 3. bullet, Reckless, and the wider ecosystem

`bullet` (jw1912) is the trainer for Reckless, Stormphrax, Viridithas, Obsidian, Alexandria,
akimbo, Integral, PlentyChess, Renegade, Arasan, Leorik and others. Read at `main` @
**`cebc78a0`** (2026-06-22), `bullet_lib` 1.0.0.

### 3.1 bullet's defaults, and what ARCHON inherited

| knob | bullet `examples/simple.rs` | ARCHON D48X |
|---|---|---|
| optimiser | AdamW, decay 0.01, β 0.9/0.999, eps 1e-8, clip ±1.98, **no bias correction** | **identical kernel**, + per-tensor clips, no bias decay, grad-clip 2.0 |
| loss | `(sigmoid(out) − target)²` (power **2**) | `\|·\|^2.5` — *closer to SF* |
| target | `wdl·result + (1−wdl)·sigmoid(score/400)` | **identical form** |
| eval_scale | 400 | 400 |
| batch × batches/sb | 16,384 × 6,104 = **100 M/sb** | 8,192 × 3,040 = **24.9 M/sb** |
| superbatches | 40 (simple) → 800 (`4_multi_layer`) | 520 (this leg), ≈3,500 cumulative |
| λ (**weight on game result**) | `ConstantWDL 0.75` | **0.30 constant** |
| QAT | none — clip + round at save; STE primitives exist, **zero examples use them** | none — same |
| data filter | `ply≥16 && !in_check && \|score\|≤10000 && Normal && dest empty` | **verbatim the same** |
| validation | none — `test_set` prints "not currently implemented" | heldout loss only; games arbitrate |

**λ conventions are inverted between the ecosystems and must never be copied across.**
bullet: λ multiplies the **game result** (λ=1 → pure WDL). nnue-pytorch: λ multiplies the
**eval** (λ=1 → pure eval). SF's `end-lambda 0.7` and bullet's `wdl 0.3` are the same thing.
ARCHON follows bullet's convention.

### 3.2 Reckless — the engine our evidence #2 is about

Read from `reckless_reference/Reckless` plus the public repos.

**Architecture (`src/nnue.rs`, `origin/main`), net v54-v60, frozen since 2026-02-13:**
`(66,864 threats + 768×10 mirrored king buckets → 768)×2 → 16 → 32 → 1`, 8 output buckets
with the layout `[0×9, 1×4, 2×4, 3×3, 4×3, 5×3, 6×3, 7×4]`, CReLU→pairwise-multiply,
`NETWORK_SCALE 380`, `FT_QUANT 255`, `L1_QUANT 64`, `FT_SHIFT 9`.

**That output-bucket layout is byte-identical to ours** (`spec.py:47-55`), as is the
L2=16/L3=32 head and the CReLU-pairwise FT activation. The comparison is therefore far
cleaner than "their simple net beats our complex one":

| | Reckless v54-v60 | ARCHON D48X |
|---|---|---|
| positional inputs | 768 × 10 mirrored king buckets = 7,680 | HalfKA 22,528 (32 king buckets × 704) |
| threat inputs | 66,864 | 60,720 |
| **total inputs** | **74,544** | **83,248** |
| accumulator/perspective | **768** | **1280** |
| FT activation | CReLU → pairwise-mul | **same** |
| head | 16 → 32 → 1 | **same** |
| output buckets | 8, layout X | **8, same layout** |
| L1 quantisation | i8 @64 | **same** |
| **threat-FT quantisation** | **255** | **64** |
| FT parameters | ≈ 57.2 M | ≈ 106.6 M |
| network scale | 380, **re-pinned per net** | 400, re-pinned per net |

**Their data pipeline is the structural idea in this report.** Era 1 (2024-02 → 2025-05,
`src/tools/datagen.rs` at `bd5daf3^`): in-engine self-play at **7,500 nodes**, 4 random plies
off a book, depth-10 validation rejecting openings beyond ±400, write window ply 16-400,
filter `!in_check && !capture && !promotion`, records in bulletformat's 32-byte shape.
Era 2 (2025-08 → now, branches `origin/nnue-tools*`): **datagen was deleted from `main` and
replaced by harvesting the PGN archive of their own OpenBench SPRT tests.**
`src/tools/pgn.rs` reads bz2 PGN directories, parses the `[FEN]` header and the per-move
`{score/depth}` comment into a binpack; `src/tools/rescore.rs` re-labels an existing binpack
with the current net's static eval and reports MSE/MAE against the old labels.
Their OpenBench instance reports **931 M games played** across 22 contributors.
Their testing traffic *is* their datagen.

**Their net-testing protocol** (extracted from the net commit bodies): every net gets a
**fixed-node screen at N = 25,000 over ~40,000 games (±2 Elo)**, then STC 8+0.08, then LTC
40+0.40. And **net v54 is the D48X situation exactly**: L1 512 → 768 measured
**+19.57 ± 2.08 at fixed nodes** and **−2.62 ± 2.46 STC / +0.47 ± 1.35 LTC with a failing
LLR −0.80 over 60 k games** — merged anyway on an endgames-book pass and an SMP pass.
Six architecturally-frozen nets followed in three months for roughly **+17.5 STC / +19.6 LTC
cumulative, from recipe and data alone**. That is a direct, favourable precedent for the
ruling recorded in `engines/canonical/MANIFEST.md`.

No public λ, LR schedule, superbatch count, or corpus size exists for Reckless. The only
official statement is the v0.8.0 release note: bullet replaced their own trainer, "over 30
iterations of stronger networks", "trained on billions of positions". **Do not infer their
recipe from bullet's defaults** — §3.3 shows how far individual engines stray.

### 3.3 The wider field — where there is consensus and where there is not

| engine | data source | nodes | positions | λ (weight on result) | superbatches | scale |
|---|---|---|---|---|---|---|
| Viridithas | **100 % own self-play, never foreign labels** | 25 k soft | 1.35 B → ~8 B | **LinearWDL 0.4 → 1.0** | 800 | train 400, **infer 240** |
| Stormphrax | **100 % own self-play, stated policy** | 24 k soft | not published | **0.2 → 0.5 → 0.8 → 1.0** | 600 + 500 | 400 |
| Caissa | own self-play, progressive purge of old games | 21 k soft | **20 B+** | n/f | n/f | n/f |
| Berserk | own `fengen` self-play | 20 k | 1.2 B → ~8 B | **0.0-0.10** | 1000 × 100 M | 160 |
| Ethereal | 67 % self-play + 33 % SF-vs-Ethereal **with SF evals discarded** | 5 k→40 k | 2-4 B/net | fixed 0.50 | ≤25000 × 134 M | ≈173 |
| Obsidian | **Leela data only** | n/a | 0.5 B → 2.8 B | n/f | n/f | 400 |
| Alexandria | **Leela data** (ODbL-attributed) | n/a | ~100 B samples/run | **0.0 → 0.15** | 800 + 200 | 362 |
| akimbo | own self-play ≤ v1.0.0, Leela after | 5 k soft | 1.5 B | n/f | n/f | 400 |
| Arasan v8 | n/f | n/f | n/f | **0.0** | 800 | n/f |
| Leorik v3.2 | n/f | 5 k | n/f | **1.0** | 1200 | n/f |
| **ARCHON** | **SF public binpacks, 0 % own** | 5 k (theirs) | 10.19 B | **0.30 constant** | 520 (≈3500 cum.) | 400 |

Four things this table settles, and one it does not:

- **Data provenance has no consensus.** Viridithas and Stormphrax are 100 % own self-play as
  a stated policy; Obsidian, Alexandria and current akimbo are 100 % Leela; Ethereal is
  adversarial with the opponent's evals discarded; Reckless harvests its own test traffic.
  **All are top-15.** But **none of them uses Stockfish's evals as its primary label
  source**, and none of them has *zero* engine-generated data. ARCHON is the only one in
  this table with no data of its own at all.
- **λ is unsettled and the whole range is occupied by strong engines** — 0.0 (Arasan) to
  1.0 (Leorik). What *is* near-universal is that λ is **scheduled**, usually ramping toward
  more WDL. ARCHON's constant 0.3 is not an outlier in value; it is unusual in being
  constant. See the null at §5.5 — Viridithas ran a proper λ sweep (six arms, 200 sb each)
  in which **every arm failed SPRT**.
- **Width is not the lever, confirmed from four directions.** Viridithas's 2048- and
  2560-wide nets lost and were cut *back* to 1024 while adding threat inputs (+27 Elo);
  Stormphrax dropped L1 1792 → 640 to afford threats; Reckless cut L1 to 384 at v47 for the
  same reason; Stockfish cut 3072 → 1024. **This corroborates our standing 768-1536
  Elo-neutrality finding exactly** — and it means evidence #1 is not a defect, it is the
  expected result.
- **The accumulator L1-norm penalty is now standard** — Viridithas 0.005, Alexandria 0.004,
  both to make sparse L1 matmul pay. ARCHON has `ft_sparsity_lambda` implemented and set to
  **0.0**.
- **What it does not settle:** superbatch counts and LR schedules are mostly unpublished
  outside bullet's examples, so "how long should a run be" has no external answer beyond
  "800-1200 superbatches = 80-120 B presentations", which we already match cumulatively.

### 3.4 Alexandria's data filters — the most directly transferable thing found

Alexandria publishes its bullet config (`examples/alexnet-master.rs` on `PGG106/bullet`,
branch `wdlnew-rewrite`). Three stacked filters, all pure data-loader code:

1. **Quiet filter**: `ply ≥ 28 && !in_check && |score| ≤ 20000 && Normal move && dest empty`.
2. **WDL-model rejection sampling**: compute (w, d, l) from a Stockfish-style WDL model, then
   keep the position **with probability equal to the modelled probability of the outcome
   that actually occurred** — discarding eval/result disagreements in proportion to how bad
   they are. (This is the soft form of Stockfish's item 6.)
3. **Online piece-count rebalancing**: a hardcoded 33-entry `DESIRED_DISTRIBUTION` plus
   running counters, accepting each position with
   `clamp(0.5 · desired[pc] / observed_freq[pc], 0, 1)`.
   **Measured +5.08 Elo STC / +2.28 LTC on its own** ([Alexandria PR #599]).

### 3.5 The screening-power gap

| | games per net screen | resulting bar |
|---|---|---|
| Reckless | **~40,000** at N = 25,000 fixed nodes | **±2 Elo** |
| Stockfish | in-recipe SPRT, up to 80,000 rounds | SPRT |
| ARCHON | 100-400 | **±15-25 Elo** |

`engines/canonical/MANIFEST.md` records D48X's promotion at **−16.6 [−43.5, +10.2] over
n=336**. At Reckless's protocol that interval would be ~±2 and the decision would not have
needed a judgment call. See G0.

---

## 4. Ranked gaps

Ranked by **tractability × expected Elo** per `CLAUDE.md`, not by expected Elo alone.
"Cheapest decisive test" means the test that can **kill** the hypothesis.

---

### G0 — Our net-screening instrument is ~100× underpowered, and the fix is cheap
**Tractability HIGH · enables everything below · rank 0**

Reckless screens every net at **25,000 fixed nodes over ~40,000 games, ±2 Elo**, before any
time-control test. We screen at 100-400 games, ±15-25 Elo. Every experiment proposed in this
document produces an effect that our current instrument cannot resolve, which is how D48X
came to be promoted on a judgment call rather than a measurement.

**This is affordable here.** At 25 k nodes and ~900 knps, a move costs ~28 ms; an 80-move
game with both engines is ~4.5 s of core time. 40,000 games on 8 concurrent workers is
**≈6 hours**, and fixed-node games are immune to the load sensitivity that makes our
time-control matches need an idle box. `CLAUDE.md` rule 5 says games arbitrate and start
early; rule 3 says prefer discrete outcomes. A 40 k-game fixed-node screen is both.

**Cheapest decisive test:** re-run the existing `engines/runs/d48x-vs-v` pairing at
N = 25,000 for 40 k games and see whether the interval actually lands near ±2 on this box.
If it does, this becomes the standard net gate and every item below becomes answerable.
It also settles the D48X promotion's falsifiable prediction on its own terms.

---

### G1 — The eval→target transform is uncalibrated; we measured that it is and shipped it anyway
**Tractability HIGH · expected size MED-HIGH · rank 1**

**What we do.** `target = 0.3·result + 0.7·sigmoid(s/400)` (`features.py:555`).
**What they do.** *Both* reference trainers ship the calibrated form, and Stockfish's recipe
uses it with HPO-tuned constants on **both** sides (`model/nnue.py:44-52`;
bullet `value.rs:104-107`, same 270/380 constants).

**Why it costs function.** Measured on our own corpora (§5.1 table): the score at which the
true expected score is 0.75 is **s ≈ 174**, but `sigmoid(174/400) = 0.607`. At `s = 400` the
truth is ≈0.96 and our target says 0.73. Two consequences:
(a) the two halves of our own blend disagree systematically — by up to ~0.23 in target units,
against a heldout RMS of 0.080 — so the 30 % outcome term and the 70 % eval term are not
measuring the same quantity;
(b) **52-74 % of all training positions land in target ∈ [0.4, 0.6]**, where the sigmoid is
essentially linear, so our loss behaves close to plain MSE-on-centipawns and spreads
resolution uniformly across the eval range instead of concentrating it where win probability
actually moves.

**We already fitted the fix and did not ship it.** `tools/nnue/fit_target_transform.py`
MLE-fits the draw-aware double sigmoid: uho `a = 175.5, scale = 71.9`, **Brier 0.0354 vs
0.0532** for the current `sigmoid(s/400)` — a **33 % Brier improvement**; fishpack32
`a = 160.2, scale = 33.2`, 0.0233 vs 0.0499. The code path exists
(`features.py:545-553`, `TrainConfig.target_offset`), is enabled in exactly five run configs
— the **D47S "S" line** — and is **absent from D48X**.

**And the one comparison we have points the right way.** Per `CLAUDE.md`, the S line
(`target_offset 163 / scale 76.9`) was **+35 Elo in *function*** over V (plain sigmoid) and
lost only on **speed** (128-wide head: 724 k vs 1051 k nps). D48X then took V's *speed* fix
(thin head) and **left S's *target* fix behind**. Those are independent axes and no
measurement said to drop the target calibration.

**Expected size.** Confounded (S vs V also differed in width and head width), so no clean
estimate exists — that is precisely what the test supplies.

**Cheapest decisive test.** A **short warm finetune leg**, not a retrain:
1. **Null panel first** — a 20-superbatch warm restart from the D48X sb520 EMA with *no
   change at all*, scored on the fixed-node screen. This bounds what a warm restart alone
   moves (`CLAUDE.md` rule 1).
2. Same leg with `target_offset`/`scale` at the MLE fit, then
   `tools/nnue/calibrate_network_scale.sh` (**mandatory** — see the box below), then the same
   screen against the same pinned opponent at the same node budget.
3. Verdict from `tools/nnue/inframe_heldout.py` + the screen, **never** from the default
   heldout (see §6).

> **Two traps, both already paid for elsewhere.**
> **(a) This is NOT the refuted "rescale the eval frame" experiment.**
> `docs/research/eval-frame-scale-model-2026-07-27.md` §11.4 killed *blanket rescaling of the
> deployed eval magnitude*, in both directions (`EVAL_SCALE` measured −27.8 ± 16.8 over 188
> games). Here the deployed magnitude is held **constant** by `calibrate_network_scale.sh`;
> what changes is *which errors the loss prioritises* — the shape of the function, not its
> scale.
> **(b) Any change to the target transform silently changes the required inference scale.**
> cosmobobak measured eleven Viridithas nets across λ 0→1: the optimal inference SCALE fell
> from ~480 to ~230, and re-pinning it recovered **+28 Elo with no retraining**. Skipping
> `calibrate_network_scale.sh` would measure scale mismatch against SPSA-tuned search margins,
> not the target.

---

### G2 — Our training distribution is skewed ~3× toward endgames — MEASURED, and someone else has already priced the fix
**Tractability HIGH · expected size MED · rank 2**

This was not on my list when I started; it came out of running Stockfish's own target spline
against our corpus.

**Measured** (weighted mixture over the ten large corpora, 1 M rows each, piece count =
`popcount(occ)`), against Stockfish's piece-count target
(`pc_y` = `−0.20, 0.45, 1.0, 0.95, 0.75` at counts 0/8/16/24/32, negatives clamped):

| ARCHON output bucket | piece counts | **ours** | **SF target** | ratio |
|---|---|---|---|---|
| 0 | ≤ 10 | **21.49 %** | 8.97 % | **2.4×** |
| 1 | 11-14 | 19.62 % | 11.02 % | 1.8× |
| 2 | 15-18 | 15.99 % | 15.89 % | 1.0× |
| 3 | 19-21 | 10.11 % | 13.12 % | 0.77× |
| 4 | 22-24 | 10.36 % | 12.87 % | 0.80× |
| 5 | 25-27 | 8.70 % | 12.54 % | 0.69× |
| 6 | 28-30 | 9.48 % | 11.63 % | 0.82× |
| **7** | **31-32** | **4.25 %** | **13.95 %** | **0.30×** |

**Mean piece count: ours 15.61, SF target 19.46.**

Two separate defects:

1. **A 3.3× deficit in the full-board bucket.** Bucket 7 covers the opening and early
   middlegame and receives 4.25 % of our training signal.
2. **A strong even/odd sawtooth at high piece counts** (pc 26: 4.30 %, pc 27: 1.30 %,
   pc 28: 3.89 %, pc 29: 0.61 %, pc 31: 0.15 %). Odd piece counts are mid-exchange positions
   where the best move is usually a recapture — and our filter drops "best move is a
   capture". Stockfish and Reckless apply the *same* filter, but Stockfish then **rebalances
   the marginal afterwards** and we do not, so the filter's side effect on the piece-count
   distribution survives into training untouched.

**This is exactly what SF PR #4942 was written for** ("an attempt to reduce the skewed
dataset piece count distribution where there are much more positions with less than 16
pieces"), and **Alexandria measured its own version at +5.08 Elo STC / +2.28 LTC** as a
standalone data-loader change (PR #599). That is the best external price tag in this
document.

**Tractability is why this ranks high.** The fix is an acceptance probability in the loader;
`features.py:535-537` already computes `popcount(occ)` for the output bucket. No new data, no
corpus rebuild, no engine change.

**The missing rung, stated rather than guessed.** I measured the *training* marginal and
compared it to *Stockfish's chosen target*. I did **not** measure ARCHON's own **deployment**
marginal — the piece-count distribution of positions our search actually evaluates. That is
one inference deep and must be measured before a reshaping target is chosen (`CLAUDE.md`
rule 4). It costs one PGN replay and no engine time; 74 MB of PGNs are already in
`engines/runs/`.

**Cheapest decisive test.** (1) measure the deployment marginal from existing PGNs;
(2) if it confirms the direction, one short warm leg with a loader acceptance probability
targeting it, against the G0 screen and the G1 null panel.

---

### G3 — Our labels are the generation of Stockfish data that Stockfish itself discarded
**Tractability of the *test* HIGH; of the *fix* LOW-MED · expected size HIGH · rank 3**

**What we do.** 99.7 % of the mix is SF search scores at 5000 nodes / depth 9 from
2021-2024-era SF nets, plus Leela runs rescored the same way.
**What they do.** Since 2026-06-25 every one of the 44 binpacks — *the same files* — is
relabelled by Leela BT4's value head.

**Why it costs function.** A pointwise regression cannot exceed its teacher. This is the
cleanest available explanation for evidence #3: our net and Reckless's sit at essentially the
same ranking quality (0.376 / 0.388) *below* SF18-static (0.424), because both are fitted to
shallow-search labels while SF18-static is not.

**Honesty check.** The nnue-pytorch wiki explicitly warns that "better evaluations don't
always give better results, as it's a tradeoff between quality and learnability", calls d9
the sweet spot for SF-generated data, and documents that training *solely* on Lc0-derived
data is worse than using it as a **retrain stage**. So the claim is not "deeper labels are
better" — it is "a *different kind* of evaluator, applied as a retrain stage, is what
Stockfish found worth 74 B positions of relabelling effort".

**Cheapest decisive test — ~1 GB of download and zero GPU.** Do **not** start by retraining.
Fetch a slice of `xushawn/test80-bt4-relabel`, take positions that also exist in our corpus,
and score *both label sets* against an SF18 multipv-8 oracle using **the same Spearman
instrument that produced evidence #3**. If BT4 labels do not rank better than ours on our own
metric, the hypothesis dies for the price of a download.

---

### G4 — We generate zero data of our own, and we are throwing away a corpus we already have
**Tractability HIGH · expected size MED · rank 4**

Every engine in §3.3 has data of its own; ARCHON has none. But the interesting version of
this is not "start a datagen campaign" — it is Reckless's:

**Reckless deleted its datagen and now harvests the PGN archive of its own SPRT tests.**
`src/tools/pgn.rs` converts bz2 PGN directories into binpacks by parsing the `[FEN]` header
and the per-move `{score/depth}` comment; `src/tools/rescore.rs` re-labels an existing
binpack with the current net at zero game cost. Their OpenBench instance has logged 931 M
games.

**Verified: our PGNs are already in that format.** `engines/runs/*/games.pgn` — 74 MB today —
carry `[FEN]` headers and per-move `{+3.24/16 0.021s}` comments, i.e. score and depth, which
is exactly what the Reckless converter consumes. Every match we run adds more, for free.
Two caveats that must be handled at the boundary: fastchess prints score in *pawns of our
printed frame* (our internal units / 100, and our internal is 2.07× SF printed cp), and the
D46 bound quarantine means only completed-iteration scores may become labels.

**And the same tool gives us the flywheel.** The largest single wins in Viridithas's public
net ledger are not new games — they are **re-labelling the existing corpus with the new net**
(+50, +25, +24 Elo at `viri12`, `viri14`, `viri18`). Elo per generation decays from ~+50 for
the first few to ~+5 after about ten.

**Related, and free: the mix is weighted by disk size.** Measured: every mix weight equals
that corpus's share of the total to four decimal places. The project's own "quality-first
flywheel rule" is not implemented anywhere in the mixing code, and the 27.5 M SF18@50k house
positions sit at 0.27 % because they are small.

**Cheapest decisive test — a one-line JSON edit.** Re-run a short warm leg with
`canon-tight-frame` + `hardneg-frame` up-weighted from 0.28 % to 5-10 % and compare against
the G1 null panel. This is the cheapest test in the document and it separates "we need better
labels" from "we need more labels", which are currently entangled. Then, separately, build
the PGN→binpack converter and measure whether ARCHON-labelled on-distribution data helps at
all before committing to a datagen campaign.

---

### G5 — Our threat FT is quantised 4× coarser than *both* comparison engines, and nothing in training knows about it
**Tractability MED · expected size LOW-MED (bounded by measurement) · rank 5**

Measured (§1.5): **54.2 % of our 77.7 M threat weights quantise to `|level| ≤ 1`, 34.8 % to
exactly zero; weight-space SNR 23.3 dB vs 36.9 dB for the HalfKA FT. 72 % of the network's
parameters are deployed at ~1.5 effective bits.**

- Stockfish: threat FT **i8 @ 256**, training-time clip **±0.496**, clipped every epoch,
  plus STE fake-quantisation of weights *and* activations by default.
- Reckless: threat FT at **`FT_QUANT 255`**.
- ARCHON: threat FT **i8 @ 64**, clip ±1.984, no QAT.

This is the sharpest single-number architecture-adjacent difference between our net and the
net that beats it in our own search, with everything else (head, activation, buckets, L1
quantisation) matched.

**But the prize is bounded and I measured the bound.** Recorded deploy parity is
`|Δsigmoid|` mean 0.0048 against a heldout RMS of 0.0797 — ~6 % of the residual. And a *pure
re-export* at the finer scale gains nothing (§5.2): the tail weights' clipping error cancels
the resolution gain exactly. Any gain requires *training* under the tighter clip so the net
adapts, which is what Stockfish does — and Stockfish's own note says the i8 choice is worth
+5 % NPS on x86, so the goal is finer resolution **at the same i8 cost**, not i16.

**Cheapest decisive test — offline, no engine change, no retrain.** Run the existing ranking
instrument (Spearman vs SF18 multipv-8) on (a) the f32 net and (b) the dequantised deployed
net. If deployed ranking quality is statistically indistinguishable from f32's, the family is
dead and G5 leaves the list. Only if there is a real gap should a tighter-clip finetune leg
be run.

---

### G6 — No optimiser-state checkpointing; the net is a 20-leg patchwork of warm restarts
**Tractability HIGH · expected size LOW-MED (unmeasured) · rank 6**

`pump_train.py` calls `opt.init(params)` unconditionally: **every resume starts Adam from
zero moments**, and the LR schedule is re-anchored by hand. The current net is the product of
~20 such legs with different objectives (policy on/off, FT sparsity on/off, distillation
on/off, frozen heads, head rebuilds) rather than one schedule. bullet checkpoints
`optimiser_state/`; Stockfish's resume carries optimiser + scheduler + λ-jitter state.

Our cumulative ≈87 B presentations is in the same range as a strong bullet engine's 800-1200
superbatches (80-120 B) — so this is **not** a "we train too little" gap (§5.6). It is a
"we train in twenty pieces where they train in one" gap.

**Cheapest decisive test.** Add optimiser state to `checkpoint.py` (mechanical), then run the
*same* two-leg sequence twice — once carrying state, once not — and compare the heldout
trajectory across the seam. If carrying state does not visibly remove the documented "~2sb
transient", the mechanism costs nothing.

---

### G7 — Missing soft filters: WDL-disagreement rejection and loss asymmetry
**Tractability VERY HIGH · expected size LOW · rank 7**

Two small, near-free items aimed squarely at the "NO ACCEPTABLE BLUNDERS" bar:

- **WDL-disagreement rejection sampling** (Stockfish item 6; Alexandria's `shouldkeep`):
  keep a position with probability equal to the modelled probability of the outcome that
  actually occurred, discarding eval/result contradictions in proportion to how bad they are.
  This targets the same population the oracle audit found produces ≥100 cp errors. ~40 lines
  in the loader. **Caveat, stated because Stockfish states it**: their own docs note this
  filter "might not be a positive" for some datasets, which is why it has an off switch.
- **`qp_asymmetry`** — penalise overestimation ~23 % harder than underestimation. Three lines
  in `losses.py:27-40` plus a config key.

---

### G8 — `ft_sparsity_lambda = 0.0` where the field standard is 0.004-0.005
**Tractability HIGH · expected size LOW (speed, not function) · rank 8**

Viridithas trains with an accumulator L1 penalty at 0.005 and Alexandria at 0.004, both to
make the sparse L1 matmul pay off. ARCHON has the mechanism (`train.py:202-221`) and ships it
off. This is an NPS lever, not a function lever — but at ~120 Elo/speed-doubling
(`exp-0154`), NPS levers are function levers in disguise. Note the D47M lesson recorded in
`spec.py:371-379`: the λ needs a warm-up ramp or it overshoots the density equilibrium.

---

### G9 — Feature-set generation lag
**Tractability LOW (retrain) · expected size UNKNOWN · rank 9 — recorded, not recommended**

We are at roughly Stockfish's SFNNv12/v13 feature set. We lack SFNNv14's opposed-pawn threats
and SFNNv16's `PP_3Wide` pawn-pair features, both of which were Elo gains for them
(`PP_3Wide` failed STC and passed LTC). `CLAUDE.md` reserves architecture decisions to the
owner; this is recorded so it is on the map, not proposed.

---

### Not gaps — recorded so they are not re-litigated

- **Accumulator width.** 768-1536 Elo-neutral at equal time is *corroborated* by four
  external engines all cutting width to afford threat inputs. Evidence #1 is the expected
  result, not a defect.
- **Batch size 8,192** vs Stockfish's 131,072 — bounded by the 8 GB card, not a decision.
- **Output-scale calibration.** Ours matches Reckless's practice exactly (§5.4).
- **The 5-criterion data filter.** Identical to bullet's and to Reckless's.
- **`pow_exp 2.5`.** Matches Stockfish's default; ahead of bullet's 2.

---

## 5. Nulls — hypotheses that died on measurement

Recorded per `CLAUDE.md` rules 1 and 6.

### 5.1 "Our corpora are on inconsistent score frames" — REFUTED

The per-pack `SFBINPACK_SCORE_DIV` values span 1.49-2.25, raw score dispersions differ by
2.6× (fishpack32 `|s|` p90 = 327 vs datapv2 = 865), and `features.py:522-527` records
disagreeing MLE fits (uho scale 71.9 vs fishpack 33.2). That looked like a frame mismatch
feeding irreducible label noise into 100 % of the mix.

**Measured directly and frame-free** — bin each corpus's own score against its own game
result, 2 M rows each, and read where the empirical expected score crosses 0.75 / 0.90:

| corpus | s at ES = 0.75 | s at ES = 0.90 | draw % |
|---|---|---|---|
| uho-band75-100 | 174.2 | 300.6 | 47.6 |
| fishpack32 | 167.7 | 219.2 | 61.6 |
| farseerT74 | 203.0 | 327.6 | 44.5 |
| datapv2 | 157.9 | 289.3 | 47.3 |
| farseerT75 | 170.4 | 265.1 | 48.4 |
| multinet | 162.2 | 289.9 | 47.5 |
| t60t70 | 167.2 | 267.6 | 54.2 |
| pylon | 173.5 | 279.0 | 49.7 |
| wisr | 161.5 | 265.8 | 63.9 |
| wrongnnue | 190.6 | 408.4 | 49.5 |
| canon-tight | 192.4 | 313.3 | 47.9 |

The 0.75 crossing spans 158-203 — **±13 % across eleven independently-converted corpora.**
The DIV normalisation worked; the dispersion differences are *distributional* (fishpack is
drawish and quiet), not unit errors. **Frame-mixing is dead.** What survives from this table
is G1 — the *global* transform is wrong for all of them at once, in the same direction.

### 5.2 "Re-export the threat FT at Stockfish's scale 256" — REFUTED as an export change

Only **0.030 %** of the shipped net's 77.7 M threat weights exceed Stockfish's ±0.4961 clip,
which made a pure re-export at scale 256 look free. But those are the *largest* weights and
their clipping error dominates: weight-space SNR goes **23.3 dB → 23.8 dB**, i.e. nothing.
A finer export scale recovers nothing on weights trained under a loose clip. Any gain needs
*training* under the tighter clip. G5 is written accordingly.

### 5.3 "Fix the 0.538× eval frame" — already refuted in-repo, do not re-run

`docs/research/eval-frame-scale-model-2026-07-27.md` §11.4: the frame measurement stands, but
correcting it by rescaling — **in either direction** — loses Elo across three independent
analytic routes; `EVAL_SCALE` measured −27.8 ± 16.8 over 188 games. The recorded conclusion
is that the constants sit at a *joint* optimum of the whole search program.

### 5.4 "Our output-scale calibration is a self-referential mistake" — NOT A GAP

`net_scale_check.sh`'s own header criticises itself for being self-referential, which read
like a defect. It is not: **re-deriving the inference scale so a new net reproduces the
incumbent's output magnitude is exactly what the field does.** Reckless moved
`NETWORK_SCALE` 400 → 388 → 380 with its nets and built `src/tools/scale.rs` whose entire job
is `380 · reference_abs_mean / new_abs_mean`; Viridithas trains at 400 and infers at 240;
cosmobobak measured **+28 Elo from re-pinning alone**. Our `calibrate_network_scale.sh` +
`net_scale_check.sh` (hard gate, band 0.85-1.18) is the same mechanism. The *separate*
question of the lineage's absolute 0.538× frame is §5.3, and it is closed.

### 5.5 "λ = 0.3 is wrong" — NOT ESTABLISHED, and the field disagrees with itself

Stockfish ~0 % game outcome; bullet's default 75 %; Arasan 0 %; Leorik 100 %; Berserk
0-10 %; Alexandria 0→15 %; Viridithas 40→100 %; Stormphrax 20→100 %; Ethereal 50 %.
**The entire range is occupied by strong engines**, so "0.3 is wrong" cannot be asserted —
and Viridithas ran a proper six-arm λ sweep (200 sb each, WDL 0.4…0.9) in which **every arm
failed SPRT** (LLR −2.95 to −3.10). What *can* be said is that our λ is unusual in being
**constant**, and that it is blended with a miscalibrated eval term (G1) so the two halves
disagree about what 0.75 means. λ should be re-examined only *after* G1, and only with the
inference scale re-pinned (the +28 Elo artefact in G1's box) and the `SCORE_MULT`↔WDL
coupling rule (`data-recipe-contract.md` §2) applied.

### 5.6 "We train on far too few positions" — REFUTED

12.95 B presentations for the D48X leg looked small against Stockfish's ~10¹². But the
cumulative chain is ≈87 B, inside the 80-120 B that strong bullet-trained engines run
(Arasan 800 sb, Renegade 800 sb, Leorik 1200 sb, Alexandria 800+200, Viridithas 800, all at
bullet's 100 M superbatch). **Volume is not the gap** — the gap is that theirs is one
continuous schedule and ours is twenty (→ G6).

### 5.7 "Our feature space is far larger than Reckless's" — REFUTED

I assumed a 768-input engine meant a ~10 M-parameter FT. Reckless's actual input space is
**74,544** (66,864 threats + 7,680 king-bucketed) against our 83,248, and their FT is
**≈57 M** parameters against our ≈107 M. The 1.9× difference is almost entirely accumulator
**width** — which our own standing finding says is Elo-neutral. **So evidence #2 cannot be
explained by architecture**, and bullet's "you won't have enough data for HalfKA-scale
feature sets" warning does not discriminate between us and them either. That is what forces
the conclusion in the Verdict: evidence #2 is a recipe-and-data result.

---

## 6. What I could NOT determine

Stated rather than guessed.

- **Whether the S-line's +35 function Elo was the target calibration or the 128-wide head.**
  The two changed together; no ablation exists. This is the single most valuable missing
  measurement, and G1's test supplies it.
- **Reckless's actual training recipe** — λ, LR schedule, superbatch count, corpus size.
  None of it is public; their `CHANGELOG.md` stops at v10 and the bullet config lives in a
  private example file. Their *data pipeline* and *architecture* are fully established from
  source; the recipe is not. **Do not infer it from bullet's defaults** (§3.3).
- **ARCHON's deployment piece-count marginal** — the missing rung of G2.
- **The deployed quantisation cost on *ranking* rather than on |Δeval|** — G5's test.
- **What `training_data_pylon` and `wrongNNUE_02_d9` actually are** (label depth, source
  net). Their ledger rows say "UNITS NOT AUDITED — D46 gate before training use" and they
  were trained on anyway: **7.2 % of the mix has unaudited provenance.**
- **Whether SFNNv14's opposed-pawn or SFNNv16's `PP_3Wide` features would pay for us.**
- **Whether the D48X `heldout 0.006350` is comparable across target transforms. It is not.**
  It is computed with `TrainConfig()` defaults (`pump_train.py:52-56`) — always
  `wdl 0.3, scale 400, target_offset 0` — so a net trained on a *calibrated* target scores
  badly on it **by construction**. **Any G1 or λ experiment that reads this number as its
  verdict will get the wrong answer.** `tools/nnue/inframe_heldout.py` exists for this.

---

## 7. What I would change first — three items

Each is cheap, each has a kill test that runs before the arm, and the first one makes the
other two readable.

### First: build the instrument (G0)

Re-run `engines/runs/d48x-vs-v` at **N = 25,000 fixed nodes for 40,000 games** — Reckless's
standard net screen, ≈6 hours on 8 workers, load-insensitive because it is fixed-node. Two
returns for one cost: it tells us whether ±2 Elo is reachable on this box, and it settles
D48X's own falsifiable prediction. **Everything below is unreadable at ±20 Elo**, which is
the real reason this study cannot rank G1-G5 by expected size.

### Second: calibrate the target transform (G1)

*The evidence is already ours.* A 33 % Brier improvement measured on our own corpora in July;
the code path exists and is tested; **both** reference trainers ship the same mechanism; the
one comparison we have (S vs V, function) points the right way; and it was dropped for a
reason (head speed) with nothing to do with the target.

**Decisive test:** the null panel (20-sb warm restart, no change), then the same leg with
`target_offset`/`scale` at the MLE fit → `calibrate_network_scale.sh` → the G0 screen.
Verdict from `inframe_heldout.py` + the screen, never from the default heldout. ~2 GPU-hours
on top of the null. If the arm does not clear the null, G1 dies.

### Third: measure the deployment piece-count marginal, then reshape (G2)

The training marginal is measured and far from Stockfish's target (mean 15.6 vs 19.5; the
31-32-piece bucket at 0.30× of target). The missing half is *our own* deployment marginal,
which costs one replay of PGNs we already have and no engine time. If it confirms the
direction, the fix is an acceptance probability in `data.py` — no new data, no corpus
rebuild, no engine change — and Alexandria has already priced the equivalent change at
**+5.08 Elo STC / +2.28 LTC**.

**Runner-up, nearly free:** G4's one-line mix edit — up-weight the two SF18-relabelled house
corpora from 0.28 % to 5-10 %. It separates "we need better labels" from "we need more
labels", which are currently entangled, and it costs one JSON key.

---

## Appendix — reproducing the measurements

All measurements were made read-only on memmapped corpora and the shipped raw checkpoint;
none touched a running process or the engine, and nothing under `src/` or `jax/` was
modified.

**Line-number caveat.** Citations into `jax/archon_jax/*.py` are against the **working tree
as of 2026-07-28**, which carries uncommitted `dual_tail` changes from the concurrent
`nnue-architecture-study-2026-07-28` work (+26 lines in `spec.py`, +45 in `model.py`).
Symbol names are stable; line numbers below those edits are not.

| measurement | method |
|---|---|
| corpus totals; mix-weight ≡ corpus-share identity | `ls -l` byte counts ÷ 32 vs `d48x_1280.json` weights |
| per-corpus score→result calibration (§5.1) | 2 M-row prefix of each `*.shuf.bin`, 25-unit score bins, linear interpolation of the ES crossings |
| target-distribution statistics (§4/G1) | 300 k-row prefix per corpus; `sigmoid(s/400)` percentiles |
| weight quantisation statistics (§1.5, §5.2) | streaming read of `d48x-1280-sb520-ema-raw.bin` (`name\n` + u64 count + f32 records, `checkpoint.py` format) |
| piece-count marginal (§4/G2) | 1 M-row prefix per corpus, `popcount(occ)`, weighted by the live mix weights; SF target from the `pc_y` control points linearly interpolated and renormalised |
| presentations | `520 × 3040 × 8192` from `spec.py:295-297` and `d48x_1280.json:6` |
| cumulative lineage | sum of `superbatches` over `tools/nnue/runs/*.json` on the D48X ancestral chain |
| PGN format check (§4/G4) | `head` of `engines/runs/agg2-both-dc50/games.pgn`; `du -shc engines/runs/*/games.pgn` |

**External sources.**
*Stockfish:* [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) @ `9f729465`,
[vondele/nettest](https://github.com/vondele/nettest) @ `ba46c2ac`
([threats.yaml](https://github.com/vondele/nettest/blob/master/threats.yaml)),
[Stockfish](https://github.com/official-stockfish/Stockfish) @ `23cf5d82`,
[docs/nnue.md](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md),
[Training-datasets wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets),
[PR #4942](https://github.com/official-stockfish/Stockfish/pull/4942),
[PR #5149](https://github.com/official-stockfish/Stockfish/pull/5149),
[xushawn/test80-bt4-relabel](https://huggingface.co/datasets/xushawn/test80-bt4-relabel),
[xu-shawn/lc0 relabel fork](https://github.com/xu-shawn/lc0/tree/relabel_dual_stream_test).
*bullet:* [repo](https://github.com/jw1912/bullet) @ `cebc78a0`,
[examples/simple.rs](https://github.com/jw1912/bullet/blob/main/examples/simple.rs),
[progression](https://github.com/jw1912/bullet/tree/main/examples/progression),
[docs/1-basics.md](https://github.com/jw1912/bullet/blob/main/docs/1-basics.md),
[wdl.rs](https://github.com/jw1912/bullet/blob/main/crates/bullet_lib/src/trainer/schedule/wdl.rs),
[adam.rs](https://github.com/jw1912/bullet/blob/main/crates/trainer/src/optimiser/adam.rs),
[value.rs](https://github.com/jw1912/bullet/blob/main/crates/bullet_lib/src/value.rs),
[qat.rs](https://github.com/jw1912/bullet/blob/main/crates/compiler/src/tensor/operation/autograd/qat.rs),
[issue #111](https://github.com/jw1912/bullet/issues/111).
*Reckless:* [engine](https://github.com/codedeliveryservice/Reckless),
[RecklessNetworks](https://github.com/codedeliveryservice/RecklessNetworks),
[bullet-fork](https://github.com/codedeliveryservice/bullet-fork),
[recklesschess.space](https://recklesschess.space/),
local clone `reckless_reference/Reckless` (`src/nnue.rs`, `src/tools/{datagen,pgn,rescore,scale}.rs`),
[OpenBench](https://github.com/AndyGrant/OpenBench).
*Others:* [Viridithas](https://github.com/cosmobobak/viridithas)
+ [networkhistory.txt](https://github.com/cosmobobak/viridithas/blob/master/notes/networkhistory.txt),
[cosmobobak on target weighting / eval scale](https://web.archive.org/web/20260125032441/https://cosmo.tardis.ac/files/2025-12-12-nnue-eval-scale.html),
[Alexandria](https://github.com/PGG106/Alexandria) + [PR #599](https://github.com/PGG106/Alexandria/pull/599)
+ [PGG106/bullet alexnet-master.rs](https://github.com/PGG106/bullet/blob/wdlnew-rewrite/examples/alexnet-master.rs),
[Stormphrax](https://github.com/Ciekce/Stormphrax), [Obsidian](https://github.com/gab8192/Obsidian),
[Berserk](https://github.com/jhonnold/berserk), [Ethereal](https://github.com/AndyGrant/Ethereal)
+ [TCEC questionnaire](https://wiki.chessdom.org/Ethereal_questionnaire_20251109),
[Caissa](https://github.com/Witek902/Caissa), [akimbo](https://github.com/jw1912/akimbo),
[PlentyChess](https://github.com/Yoshie2000/PlentyChess),
[Arasan v8 config](https://github.com/jdart1/arasan-chess/blob/master/network/arasanv8.rs),
[Renegade config](https://github.com/pkrisz99/Renegade/blob/main/renegade-net.rs),
[Leorik v3.2 config](https://github.com/lithander/Leorik/blob/master/Resources/leorik-v3.2.rs).
