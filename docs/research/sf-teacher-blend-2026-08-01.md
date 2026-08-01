# Stockfish's net as a LIVE teacher: the build, the gates, the probe

**2026-08-01.** Ships the 60/40 distillation blend — 40 % SF-net teacher,
60 % existing labels — as a live model inside the JAX trainer. Companion to
`sf-teacher-calibration-2026-08-01.md`, which measured the frame this consumes.

Artifacts:

| what | where |
|---|---|
| converter + exact integer reference | `tools/nnue/sf_to_archon_warm.py` |
| gate 1, converter fidelity | `jax/tools/sf_teacher_parity.py` |
| gate 2, frame conversion | `jax/tools/sf_teacher_frame_check.py` |
| gate 3, alpha=0 bit-identity null | `jax/tools/sf_teacher_null.py` |
| probe spec generator | `tools/nnue/make_sfblend_probe.py` |
| the teacher checkpoint (gitignored) | `data/nnue/teachers/sf-eca43a97-teacher-raw.bin` |

---

## 0. The findings, in the order that matters

1. **No relabelling pass, and no second feature front-end.** The two nets share
   their ENTIRE input space, so SF's weights load into ARCHON's own parameter
   dict and run through ARCHON's own `model.forward`, sharing the student's
   sparse-index gather batch for batch. §1.
2. **The topology maps EXACTLY** — not approximately — onto three flags that
   were already in the tree (`dual_tail`, `out_skip`, `psqt`) plus widths. One
   genuinely new flag was needed, `sf_act`, and its null says it earns its
   place: the same weights through our tail are **27× worse**. §2.
3. **Gate 1: 9550/9550 EXACT** against the `sf-eval` oracle on an exact-integer
   reference fed ARCHON's own feature indices. The float teacher the trainer
   actually runs sits at median 6.4 internal units over a ±4700 range, slope
   0.9938. §3.
4. **A null I ran and lost, reported.** Folding SF's truncation bias out looks
   free and exact; measured, it made the gap **4.7× worse**, because a CReLU'd
   accumulator is mostly zeros and a truncation that discards nothing has mean
   0, not ½. §3.2.
5. **The per-corpus `(k, p)` frame is WIRED, not approximated by a scalar.**
   `teacher_frame_scale` is a scalar and no scalar can do this job. §4.
6. **Gate 2 caught my own wrong instrument before it caught anything else.**
   My first statistic — a level-space through-origin slope — read 0.94–1.67 and
   said FAIL. The table is fitted in LOG space on the decisive subpopulation,
   deliberately, with a stated reason. On the statistic it was actually fitted
   to, the pipeline reproduces it to **2.8 % worst case, weighted mean 1.0005**.
   §5.
7. **One thing the blend changes that nobody asked it to**: at α=0.4 the
   ground-truth `result` term's weight falls from 0.30 to 0.18. §6.

---

## 1. Why this is a converter and not a data pass

`sf-eval` is a subprocess that prints integers. The tempting design is to run
it over the corpus once and store a column. That would have been wrong for the
owner's stated goal ("net-only approach at inference time") and it is also
unnecessary, because the two nets are the same shape where it counts:

| | ARCHON (D49) | Stockfish `eca43a97` |
|---|---|---|
| HalfKAv2_hm | 22,528 | 22,528 |
| FullThreats | 60,720 | 60,720 |
| **total inputs** | **83,248** | **83,248** |
| accumulator / perspective | 1280 | 1024 |
| neck / tail | 16 / 32 | 32 / 32 |

The feature identity is proven, not assumed — `sf_swap.rs`'s header records
ARCHON's own `for_each_halfka` / `for_each_threat` index lists matching a
bit-exact SF oracle 40/40 × 2 perspectives. So the teacher can be handed the
STUDENT'S OWN sparse index columns, which is what makes a live teacher cheap:
one extra FT gather per step, no second feature pipeline, no storage.

## 2. The topology maps exactly

SF's head differs from ours in four ways, and three of them were already
implemented for other reasons:

| SF | ARCHON flag |
|---|---|
| SqrCReLU **and** CReLU forward at both layers; `fc_2` reads the 128-wide concat of BOTH layers' pairs | `dual_tail=True` |
| `skip_0 = fc0[30] - fc0[31]`, added outside every nonlinearity | `out_skip=True` |
| PSQT buckets as a linear channel, `(psqt[stm][b] - psqt[ntm][b])/2` | `psqt=True` — `psqt_combine` is literally `0.5*(stm-ntm)[b]` |
| the exact activation | **`sf_act=True`, new** |

`skip_0` deserves a note because it looks unmappable: it reads two RAW `fc0`
units, before any activation. But `fc0` is affine in the pairwise features, so
`fc0[30] - fc0[31]` is affine in them too — and `out_skip` is exactly "a
per-bucket learned linear read of `hl1`". The linear half goes into `skipw`,
the constant half folds into `l3b`. Nothing is approximated.

### 2.1 Why `sf_act` had to be new

`dual_tail`'s docstring says **shape only**, and for borrowing a topology that
is the right call: SF's `WeightScaleBits` shifts are int8 bookkeeping. For
REPRODUCING their net it is not enough, on two counts:

* SF squares the **unclipped** pre-activation (`(v*v) >> 21` has no lower
  clamp), so a negative unit contributes **positively**, where our SqrCReLU
  sends it to 0. That is a different function, not a rounding difference.
* the square lane's gain is 127/128, not 1.

Both fall out of one substitution. Their clipped lane is `(v >> (WSB+1))`
clamped to [0,127]; write `x = v / (2^(WSB+1)·127)` and it becomes exactly
`clamp(x, 0, 1)`. The same substitution turns the square lane into
`min(x² · 127/128, 1)` — at BOTH layers, because the `WSB+1` / `WSB` pair
cancels against the `2*(WSB+1)+7` / `2*WSB+7` shifts.

**Null:** the same converted weights read through our own tail (`sf_act=False`)
give median |diff| **173.2** against the oracle where `sf_act=True` gives
**6.4**, and resid_sd 290 against 16.6. The flag is doing what it claims.

### 2.2 The one thing that does NOT map: the output bucket

SF's layer-stack index is `(pieces-1)/4`. Ours is
`OUTPUT_BUCKETS_LAYOUT[pieces]`. **They are not the same partition and no
relabelling makes them one** — our bucket 0 spans popcounts 0–8, which is their
buckets 0, 1 *and* 2. Handing the teacher our index evaluates most positions on
the wrong stack, silently.

So the teacher gets its own `t_bucket` column
(`features.map_batch(sf_bucket=True)`), and `train._teacher_targets` swaps it
in for the teacher's forward only.

## 3. GATE 1 — converter fidelity

Three quantities on identical positions: **A** the `sf-eval` oracle, **B** an
exact numpy transcription of `evaluate_raw` fed ARCHON's own sparse indices,
**C** the float teacher the trainer runs.

**B is the layout proof** — integer in, integer out, no float slack for an
error to hide in.

| panel | n | B vs A | C vs A median / max | C slope | resid_sd |
|---|---|---|---|---|---|
| wl50 FENs | 50 | **50/50 exact** | 6.19 / 58.7 | 0.9892 | 15.8 |
| uho-band75-100 | 2000 | **2000/2000 exact** | 6.43 / 103.1 | 0.9915 | 16.6 |
| fishpack32 | 1500 | **1500/1500 exact** | 5.97 / 86.1 | 1.0005 | 11.7 |
| farseerT74 | 1500 | **1500/1500 exact** | 5.10 / 134.2 | 0.9940 | 12.9 |
| datapv2 | 1500 | **1500/1500 exact** | 6.69 / 187.3 | 0.9938 | 18.5 |
| t60t70 | 1500 | **1500/1500 exact** | 5.12 / 107.1 | 0.9943 | 13.4 |
| wrongnnue | 1500 | **1500/1500 exact** | 7.27 / 142.2 | 0.9934 | 20.8 |
| **total** | **9550** | **9550/9550 exact** | | | |

Free control on the way past: canonicalizing a black-to-move FEN (rank mirror
+ colour swap, castling/ep dropped) moves `sf-eval` by **0 over 50**, so the
stm-canonical corpus convention is sound for this measurement.

### 3.1 What the float residual costs, stated

C is not exact and cannot be — SF's pipeline truncates in three places and ours
is f32. The decision-relevant question is not the absolute gap but whether it
is systematic, because `(k, p)` were regressed against `sf-eval` itself:

* **slope 0.9938** (range 0.989–1.001 over seven panels): the float teacher
  runs ~0.6 % low. Every corpus's own `k` CI is 0.3–5.9 %, so this sits inside
  the calibration's own uncertainty for 10 of 12 corpora.
* **intercept +0.9 to +2.8** internal units — about 1 cp once converted.
* **resid_sd 11.7–20.8** internal units ⇒ ~5–9 label cp after ×0.42, against
  label scales `b` of 33–120.

**I did not fold a 1/0.9938 correction in.** The deviation's spread across
corpora (0.989–1.001) is the same size as its mean, so correcting by the mean
would help some corpora and hurt others, on a constant measured from 6 of 12.
Reported instead.

### 3.2 A null I ran and lost

Three of SF's steps floor a fractional part. The obvious model is "mean ½, and
½ is a constant, so subtract `0.5·Σw` into the next layer's bias" — exact,
free, and it fixes a genuine systematic.

**Measured, it made the gap 4.7× worse:** wl50 median 6.19 → 27.28, max 58.7 →
277.5. The arithmetic was fine; the **premise** was wrong. A CReLU'd
accumulator is mostly zeros, `lo*hi` is then exactly 0, and a truncation that
discards nothing has mean 0, not ½. The true mean loss is a data-dependent
fraction of ½ that cannot be derived from the weights alone. Kept behind
`--debias`, off, with the receipt in the docstring.

## 4. The frame: per-corpus `(k, p)`, not a scalar

`teacher_frame_offset` / `teacher_frame_scale` are scalars, and they encode a
different transform anyway (the inverse of a draw-aware double sigmoid). The
measured conversion is

```
label_frame_teacher = k_i * (1 + material/77871)^(p_i) * teacher_raw
material = 534*count<PAWN>() + non_pawn_material()
```

per corpus **and** per position, with `k` spanning 0.2976–0.6556 and `p`
spanning −1.57 to +2.25.

**A scalar cannot be made to work**, and this is measured rather than argued:
over the (corpus × material) grid the multiplier runs 0.319–0.604, the
weight-averaged constant 0.4182 is wrong by up to **+45 %**, and `uho` — 25 % of
the mix on its own — is mispriced **−20 % at low material and +19 % at high by
the same constant**. There is no better scalar; the error needs the exponent.

### What shipped

`TrainConfig.teacher_kp`, a list of `(k, p)` indexed **positionally against
`mix`**, exactly like `per_corpus_target` — the contract the calibration study
verified directly against the data (11/12 corpora reproduce their own slot to
≤1.6 %). `make_sfblend_probe.py` additionally **order-checks it by corpus
name** before writing the spec, so a positional slip cannot reach a run.

Per record, `data.py` computes `k·(1+material/77871)^p` host-side (material is a
property of a record we have already decoded) and ships only the finished
multiplier. `train._teacher_targets` then applies the SAME per-corpus
draw-aware transform the label went through:

```
cp  = teacher_raw * teacher_mult
tv  = 0.5 * (1 + sigmoid((cp - a_i)/b_i) - sigmoid((-cp - a_i)/b_i))
```

which reduces to the plain sigmoid at `a_i = 0`, so one expression covers both
regimes. Blending a teacher through a *different* transform from the label is
the D48Y bug — the two halves of `target` become different quantities.

## 5. GATE 2 — and the instrument that failed first

**My first statistic was wrong, and it said FAIL.** I fitted a level-space
through-origin OLS of label on converted teacher and got slopes 0.94–1.67,
weighted mean 1.22, worst 67 % off.

The table is not fitted that way. `(k, p)` come from the study's model MP: a
**log-space** fit of `log(|label|/|teacher|)` on `log(blend)`, restricted to
the decisive sign-agreeing subpopulation (`|teacher| > 150`) — a deliberate
choice with a stated reason, that a level-space LS slope is dominated by the
|eval| tail while the trainer's target is a saturating sigmoid, so the TYPICAL
position is what the loss sees.

So the gate must test the quantity the table was fitted to. On n=6000 iid
records per corpus:

| # | corpus | k | p | **ratio** | null | terc | ols0 | R² |
|---|---|---|---|---|---|---|---|---|
| 0 | uho-band75-100 | 0.3045 | +2.14 | **0.991** | 4.04 | 1.013 | 1.035 | 0.874 |
| 1 | fishpack32 | 0.3436 | +1.71 | **0.999** | 2.52 | 0.871 | 1.020 | 0.741 |
| 2 | farseerT74-band75-100 | 0.6556 | −1.57 | **0.988** | 4.39 | 0.962 | 1.448 | 0.596 |
| 3 | datapv2-band75-88 | 0.4434 | −0.61 | **0.995** | 3.58 | 0.951 | 1.054 | 0.846 |
| 4 | farseerT75-band75-83 | 0.5489 | −1.40 | **1.015** | 4.50 | 0.968 | 1.562 | 0.619 |
| 5 | multinet-band75-88 | 0.3836 | +0.11 | **1.012** | 3.98 | 0.865 | 1.181 | 0.830 |
| 6 | t60t70-band75-85 | 0.4139 | −0.17 | **1.027** | 5.54 | 0.989 | 1.570 | 0.558 |
| 7 | pylon-band75-88 | 0.5336 | −1.32 | **1.012** | 4.62 | 0.979 | 1.613 | 0.593 |
| 8 | wisr-band75-95 | 0.2976 | +1.87 | **0.972** | 9.05 | 1.057 | 1.107 | 0.865 |
| 9 | wrongnnue-band75-95 | 0.3704 | +1.01 | **1.026** | 5.26 | 1.125 | 1.556 | 0.505 |
| 10 | canon-tight-frame | 0.3630 | +1.52 | **0.989** | 4.16 | 0.966 | 0.934 | 0.894 |
| 11 | hardneg-frame | 0.3053 | +2.25 | **0.988** | 2.51 | 0.959 | 0.914 | 0.860 |

**Weighted mean 1.0005, range 0.972–1.027, worst 2.8 % — PASS.** The null
(teacher column permuted) reads 2.5–9.0, so the statistic discriminates rather
than being satisfied by any smooth column.

Two independent cross-checks ride along:

* `terc` (material spread after correction) lands 0.865–1.125 here against the
  study's own 0.876–1.127 — so this pipeline's `material` decode agrees with
  the study's, computed from different code.
* `ols0` reproduces my failed first statistic, and its pattern is real
  information rather than noise: it tracks R² almost monotonically. Where the
  teacher explains the labels least (`wrongnnue` 0.505, `t60t70` 0.558), the
  tail-weighted slope is highest (1.56, 1.57). **The teacher systematically
  under-predicts the biggest labels on exactly the corpora whose labels it
  agrees with least.** At α=0.4 that pulls extreme targets toward the middle on
  those corpora. Not a defect of the conversion — a property of the teacher —
  but it is the thing to look at first if the probe disappoints.

## 6. What the blend changes that nobody asked for

`distill_alpha` blends against the FINISHED label, and the finished label is
already `wdl*result + (1-wdl)*eval_t` with `wdl = 0.3`. So at α=0.4:

```
target = 0.4*teacher_eval + 0.6*(0.3*result + 0.7*eval_t)
       = 0.4*teacher_eval + 0.18*result + 0.42*eval_t
```

**The ground-truth game result's weight falls from 0.30 to 0.18.** That is the
literal reading of "40 % teacher, 60 % existing labels" and it is what the
existing `distill_alpha` knob has always meant, so no code path forks — but it
is a second change riding along with the one under test, and it is stated here
rather than left in the code. If the probe disappoints, the alternative
(teacher replaces only the `eval_t` half, holding the result weight at 0.30) is
a one-line variant worth an arm.

## 7. GATE 3 — the alpha=0 bit-identity null

Three arms from the same warm checkpoint over the same batches: **A** no
teacher (today's code), **B** teacher wired at α=0, **C** teacher wired at
α=0.4. B must be bit-identical to A; **C must differ**, because a null that
passes through a disconnected harness proves nothing.

Result: see `jax/tools/sf_teacher_null.py` output, recorded in §9 below.

The config for all three comes from `pump_train.build_cfg` — extracted in this
change for exactly this reason. A null that tests a configuration the trainer
would never construct is worth nothing, and `TrainConfig` is built from an
explicit field list where a missing line silently drops a spec key (the D48Y2
failure mode).

## 8. The probe

12 superbatches, α=0.4, warm from the sb1630 EMA checkpoint, generated by
`tools/nnue/make_sfblend_probe.py` from the plain run's own spec so that
"same everything except the teacher" is enforced by construction.

`resume_sb=1630` additionally sets `skip_batches = bps*1630`, so the probe
consumes the **same batches in the same order** the plain continuation
consumed over the same superbatches — paired at the batch level.

**One caveat, stated because it weakens the comparison.** The probe warms from
`d49lrc3-sb1630-ema-raw.bin` (the EMA checkpoint, as instructed). The plain
continuation resumed at 12:26 from `d49-1280-sb1630-raw.bin`, which no longer
exists on disk; the surviving `d49lrc3-sb1630-raw.bin` and its EMA sibling
**differ**. So the plain run's `cyc.log` over sb1631–1642 is an *indicative*
reference, not a matched control. The matched control is a second 12-sb arm at
α=0 from the identical warm, and it is the companion this probe needs before
any Elo claim.

## 9. Log

**15:31:53** `archon-d49sfb.service` launched, resume_sb=1630, α=0.4.
Teacher loaded and reported at the right shape:
`hl=1024 l2=32 l3=32 dual_tail=True sf_act=True`.

**sb1631** — `value 0.002589`, **155,268 pos/s**.

Two things to read off that line, and one not to.

* **Throughput −14 %**, not −50 %. The teacher costs one extra FT gather
  (1024-wide against the student's 1280) with no backward pass, so a live
  teacher is much cheaper here than the "total step ~2×" the D47V comment
  budgeted for a same-width ARCHON teacher.
* **The train loss is NOT comparable to the plain run's 0.006782** — it is
  measured against a different target. What it does say is directional and
  useful: blending in 40 % of a *deterministic* function of the position
  removes irreducible noise from the target, so the blended loss SHOULD fall,
  and it did (0.00259). Had the frame been mis-scaled the way the retracted
  ≈1.023 prediction would have been — 2.45× too large — the blended target
  would have been harder to fit and this number would have gone UP. It is a
  weak check but it is the right sign, at zero cost, one superbatch in.
* **Heldout is the comparable metric** (fixed set, run frame a=175.1 b=71.6,
  no teacher involved), every 2 sb. Plain run reference: sb1640
  `value 0.010495 | ema 0.010455`.

### Two harness bugs this cost, both mine, both worth recording

1. **`archon_mask.so` was missing from the worktree** — `pol_mask: "ids"` needs
   the PyO3 masker and there is deliberately no python fallback. This would
   have killed the launch identically; the null run found it first, which is
   the entire argument for running a dry pass on the real config.
2. **The null harness donated one device buffer twice.** `params` and the EMA
   shadow must be SEPARATE buffers because `step` donates both — and
   `jnp.asarray` on an already-device array is a no-op, so building the shadow
   from `params` shares them. `pump_train` carries a comment warning about
   exactly this and I reproduced it anyway.
