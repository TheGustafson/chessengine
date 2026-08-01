# SF-net teacher → label frame: the per-corpus calibration table

**2026-08-01.** Gates the 60 % label / 40 % SF-net-teacher distillation blend.
Machine-readable companion: `docs/research/sf-teacher-calibration-2026-08-01.json`.
Tools: `tools/nnue/sf_teacher_calibration.py`, `tools/nnue/verify_per_corpus_target.py`.

Reproduce (each headline number is the mean of three independent 20 000-record
iid samples, seeds 20260801 / 777001 / 313370):

```
.venv/bin/python tools/nnue/sf_teacher_calibration.py --n 20000 --reps 300 \
    --out docs/research/sf-teacher-calibration-2026-08-01.json
.venv/bin/python tools/nnue/sf_teacher_calibration.py --n 20000 --reps 0 --null   # null panel
.venv/bin/python tools/nnue/verify_per_corpus_target.py --sample 400000
```

---

## 0. The findings, in the order that matters

1. **The premise this study was commissioned with was wrong, and wrong in the
   expensive direction.** `sf-eval` does **not** print ARCHON-frame values; it
   prints Stockfish's **raw internal NNUE units**. The predicted teacher→label
   slope of **≈ 1.023 does not hold for any corpus** — the correct conversion is
   **0.32–0.60**, so shipping 1.023 would have scaled every teacher target
   **~2.45× too large**. §2.
2. **The conversion varies 1.90× over the (corpus × material) grid**, 0.319 →
   0.604. A single pooled constant is wrong by up to **+45 %**, and its error
   spans 0.762× to 1.445×. §4.
3. **Material must be handled, and the obvious theory is only half right.** SF's
   `to_cp` divisor `p_a` is a red herring — it varies ±0.9 % over the entire
   material range. SF's `evaluate()` blend `(77871+material)/77871` is the real
   law, but its **exponent is not 1 for any corpus** (fitted −1.57 to +2.25), and
   the exponent's sign splits the corpora by labelling provenance. §5.
4. **`per_corpus_target[i]` maps to `mix[i]` positionally — verified against the
   data, not assumed.** 11 of 12 corpora reproduce their own stored `(a, b)` to
   within 1.6 % on an independent MLE refit; the 12th is a documented deliberate
   duplicate, not a mapping error. §3.
5. **Nulls and controls pass.** Permuted-teacher null collapses every slope to
   |≤ 0.026| against a real 0.32–0.83, R² to 0.000, sign agreement to 0.46–0.52.
   The teacher matches Stockfish's own `eval` **36/36 exactly** on the precise
   canonicalized FENs used here. The decoder round-trips byte-exactly 4800/4800.
   Sign agreement is **0.989–0.997** at |eval| > 100 for every corpus. §6.
6. **One of my own mid-study claims is retracted here.** I reported that
   contiguous-window sampling had broken the error bars. Re-tested correctly, the
   window-vs-iid difference is inside noise. §6.

---

## 1. What is being calibrated, and in which direction

Fit direction is **teacher → OUR frame**, per the standing ruling (the eval frame
has a measured interior optimum we already sit on; moving toward SF's frame costs
~18 Elo). Nothing here converts our labels toward SF.

The regression is **label on teacher**, not the reverse. The teacher is
deterministic — the same FEN gives the same integer every time — while the label
is a noisy search score. Regressing the noisy quantity on the exact one is the
direction that gives an unbiased slope, and it is the direction we need.

### The three frames

| quantity | what it actually is |
|---|---|
| **teacher** | `engines/bin/archon-sfswapbin sf-eval <net>` → `sf_swap::Net::evaluate_raw` — Stockfish's **raw NNUE output in their internal units**, side-to-move relative |
| **label** | `recs["score"]` — a Stockfish **search** score, mover-relative, divided at convert time by that pack's own `SFBINPACK_SCORE_DIV` |
| **material** | SF's `evaluate()` multiplies the raw net by `(77871 + material)/77871` *before the search ever sees it* (`eca43a97:src/evaluate.cpp:80-81`), where `material = 534*count<PAWN>() + non_pawn_material()`. Every label carries that factor; `evaluate_raw` does not. |

### Conventions, checked not assumed

Records are stm-canonicalized (the mover is White on a vertically mirrored
board), so the decoded FEN is white-to-move and `sf-eval`'s stm-relative output
is already in the label's point of view. Castling rights and the ep square are
not stored — irrelevant, because HalfKAv2_hm + threats are blind to both
(measured 2026-07-30, 800/800 bit-identical twins), and confirmed directly by the
36/36 Stockfish control in §6.

---

## 2. The ≈ 1.023 prediction: built on a wrong premise, and it fails

The brief derived it as: labels are SF-internal ÷ 1.9 = 0.526 × SF-internal,
`sf-eval` output is 0.538 × SF-internal, therefore 0.538/0.526 ≈ **1.023**.

**The second half is false.** `src/eval/sf_swap.rs` (branch
`experiment/reckless-net-2`, `9b9d5aa1`) has two entry points:

- `evaluate_raw()` — *"Their evaluation, in **THEIR internal units**, side-to-move
  relative"*
- `evaluate()` — `evaluate_raw() as f32 * FRAME`, `const FRAME: f32 = 0.5126`,
  documented as *"SF internal units → ARCHON's frame. **Measured, not assumed**"*

and `src/uci.rs:100` wires the `sf-eval` subcommand to **`evaluate_raw`**. The
0.538 figure belongs to `evaluate()`'s conversion, not to what `sf-eval` prints.
Stockfish's own binary settles it independently — its `eval` command prints the
same integers under the header *"(side to move, internal units)"*, 36/36 (§6).

(0.5126 was measured against the *pre*-`network_scale`-600 canonical net; the
brief's separate observation that our current canonical is 0.7604 × teacher is
0.5126 × 600/400 = 0.769 — the same fact after the 2026-07-30 promotion.)

**With the units read correctly, the prediction is per corpus, not global.** A
label is a Stockfish search value divided by that pack's own divisor, so

> predicted slope of `label` on `teacher × blend` = **1 / SCORE_DIV**

and the divisors are not all 1.9. From `ledger/data.jsonl` and
`tools/nnue/runs/stage5_data_chain.sh`:

| corpus | DIV | source |
|---|---|---|
| uho-band75-100 | 1.90 | ledger, `rescale_records.py --div 1.9` |
| fishpack32 | 1.90 | ledger, `rescale_records.py --div 1.9` |
| farseerT74-band75-100 | 1.66 | ledger |
| datapv2-band75-88 | 2.25 | chain `do_pack` |
| farseerT75-band75-83 | 1.94 | ledger |
| multinet-band75-88 | 1.98 | chain |
| t60t70-band75-85 | 1.97 | ledger |
| pylon-band75-88 | 1.95 | chain |
| wisr-band75-95 | 1.85 | chain |
| wrongnnue-band75-95 | 1.49 | chain |
| canon-tight-frame | 0.674 | chain — applied to **SF18 UCI cp**, a different frame family |
| hardneg-frame | 0.699 | chain — applied to **SF18 UCI cp** |

### Verdict, per corpus

`MB` is the model `label = b0 + b1·(teacher × blend)`, so `b1` is directly
comparable to `1/DIV`. The two house corpora are compared on the `to_cp` path
(`MA`) instead, since their labels went through `to_cp`.

| # | corpus | measured | predicted `1/DIV` | ratio | holds? |
|---|---|---|---|---|---|
| 0 | uho-band75-100 | 0.3545 | 0.5263 | **0.674×** | ✗ 33 % low |
| 1 | fishpack32 | 0.3851 | 0.5263 | **0.732×** | ✗ 27 % low |
| 2 | farseerT74-band75-100 | 0.7381 | 0.6024 | **1.225×** | ✗ 23 % high |
| 3 | datapv2-band75-88 | 0.4077 | 0.4444 | 0.917× | ~ 8 % low |
| 4 | farseerT75-band75-83 | 0.6336 | 0.5155 | **1.229×** | ✗ 23 % high |
| 5 | multinet-band75-88 | 0.4132 | 0.5051 | **0.818×** | ✗ 18 % low |
| 6 | t60t70-band75-85 | 0.5428 | 0.5076 | 1.069× | ✓ 7 % high |
| 7 | pylon-band75-88 | 0.6712 | 0.5128 | **1.309×** | ✗ 31 % high |
| 8 | wisr-band75-95 | 0.3586 | 0.5405 | **0.663×** | ✗ 34 % low |
| 9 | wrongnnue-band75-95 | 0.5319 | 0.6711 | **0.793×** | ✗ 21 % low |
| 10 | canon-tight-frame (MA) | 1.3834 | 1.4837 | 0.932× | ~ 7 % low |
| 11 | hardneg-frame (MA) | 1.1825 | 1.4306 | **0.827×** | ✗ 17 % low |

Measured run-to-run uncertainty on the scalar family is 0.2–1.6 % (§6), so every
one of these deviations is real by a wide margin. **The recorded divisors are a
good ordering prior and a bad magnitude prior:** the derivation survives within
10 % for `datapv2`, `t60t70` and `canon-tight`, and misses by 17–34 % on the
other nine.

**Stated plainly, as the brief asked: the ≈ 1.023 does not hold anywhere.** Under
the corrected units it was never the right number, and its per-corpus replacement
`1/DIV` also fails for 9 of 12. Ship §4's measured table instead.

---

## 3. The corpus ↔ `per_corpus_target` mapping — verified

**Claim: `per_corpus_target[i]` belongs to `mix[i]`, positionally.**

*From the consumer code* — `jax/archon_jax/data.py:151` builds
`bins = [b for b, _ in mix]`; `:176-182` hard-checks `len(_pct) == len(bins)` and
unpacks the pairs into `_off` / `_scl` **in list order**; `:187-189` indexes
`_off[src]` / `_scl[src]` where `src` is the mix index carried per record. There
is no name lookup anywhere in the path.

*From the data* — `tools/nnue/verify_per_corpus_target.py` refits each corpus's
own draw model by MLE on 400 k records (numpy-only coarse-to-fine grid; scipy is
not in the venv) and compares against the stored pair:

| # | corpus | fitted `(a, b)` | stored `(a, b)` | Δ |
|---|---|---|---|---|
| 0 | uho-band75-100 | 175.7, 71.8 | 175.5, 71.9 | 0.1 % / 0.2 % |
| 1 | fishpack32 | 160.5, 33.3 | 160.2, 33.1 | 0.2 % / 0.6 % |
| 2 | farseerT74-band75-100 | 166.0, 118.6 | 164.5, 117.7 | 0.9 % / 0.8 % |
| 3 | datapv2-band75-88 | 166.2, 70.0 | 166.9, 70.3 | 0.4 % / 0.4 % |
| 4 | farseerT75-band75-83 | 143.1, 91.5 | 142.8, 91.2 | 0.2 % / 0.4 % |
| 5 | multinet-band75-88 | 168.6, 69.5 | 168.6, 69.5 | 0.0 % / 0.0 % |
| 6 | t60t70-band75-85 | 147.9, 80.9 | 148.4, 81.2 | 0.3 % / 0.4 % |
| 7 | pylon-band75-88 | 146.4, 94.4 | 146.5, 95.0 | 0.1 % / 0.7 % |
| 8 | wisr-band75-95 | 167.5, 52.5 | 167.8, 52.5 | 0.2 % / 0.1 % |
| 9 | wrongnnue-band75-95 | 244.7, 120.4 | 247.0, 122.3 | 0.9 % / 1.6 % |
| 10 | canon-tight-frame | 183.1, 72.5 | 182.5, 72.5 | 0.3 % / 0.0 % |
| 11 | hardneg-frame | **141.9, 79.5** | **175.5, 71.9** | **19.1 % / 10.6 %** |

**11/12 land on their own slot to ≤ 1.6 %, most under 0.5 %. The mapping is
confirmed.**

Slot 11 is **not** a mapping error. Its stored pair is byte-identical to slot 0's,
deliberately: `docs/research/d48y-wdl-fix-plan.md:234` — *"hardneg-frame | (no
fit) | | | 0.013 % of the mix; uho params used, immaterial"*. hardneg-frame is
trained with uho's draw model. Its own refit is **a = 141.9, b = 79.5**, should
anyone want to close it; at 0.013 % of the mix, not closing it costs nothing.

*Method note:* a nearest-neighbour permutation test alone reports "identity
mapping NOT confirmed" here — correctly and misleadingly, because a duplicated
stored pair makes one of the two slots lose itself to a tie. The verifier now
tests the positional claim directly and names duplicates as duplicates.

---

## 4. THE TABLE — per-corpus teacher → label conversion

Recommended shippable law, per corpus **i**:

```
label_frame_teacher = k_i * (1 + material/77871)^(p_i) * teacher_raw
material = 534*count<PAWN>() + non_pawn_material()      # both colours
```

Chosen over the alternatives because it is the only form that is both (a)
positive everywhere — a free *linear* material coefficient fits −4.0 on
wrongnnue, which flips sign in the endgame and is unusable as a conversion — and
(b) actually flattens the material dependence (right-hand column).

| # | corpus | wt % | **k** | 95 % (measured) | **p** | 95 % (measured) | material spread after |
|---|---|---|---|---|---|---|---|
| 0 | uho-band75-100 | 25.12 | **0.3045** | ± 0.0059 (1.9 %) | **+2.14** | ± 0.14 | 1.011 |
| 1 | fishpack32 | 15.83 | **0.3436** | ± 0.0068 (2.0 %) | **+1.71** | ± 0.12 | 0.876 |
| 2 | farseerT74-band75-100 | 12.10 | **0.6556** | ± 0.0159 (2.4 %) | **−1.57** | ± 0.10 | 1.012 |
| 3 | datapv2-band75-88 | 9.71 | **0.4434** | ± 0.0015 (0.3 %) | **−0.61** | ± 0.02 | 0.946 |
| 4 | farseerT75-band75-83 | 9.05 | **0.5489** | ± 0.0167 (3.0 %) | **−1.40** | ± 0.15 | 0.998 |
| 5 | multinet-band75-88 | 8.96 | **0.3836** | ± 0.0061 (1.6 %) | **+0.11** | ± 0.08 | 0.897 |
| 6 | t60t70-band75-85 | 8.23 | **0.4139** | ± 0.0110 (2.7 %) | **−0.17** | ± 0.11 | 1.013 |
| 7 | pylon-band75-88 | 4.65 | **0.5336** | ± 0.0208 (3.9 %) | **−1.32** | ± 0.22 | 0.970 |
| 8 | wisr-band75-95 | 3.48 | **0.2976** | ± 0.0055 (1.8 %) | **+1.87** | ± 0.17 | 1.055 |
| 9 | wrongnnue-band75-95 | 2.58 | **0.3704** | ± 0.0219 (5.9 %) | **+1.01** | ± 0.50 | 1.127 |
| 10 | canon-tight-frame | 0.27 | **0.3630** | ± 0.0020 (0.5 %) | **+1.52** | ± 0.08 | 0.986 |
| 11 | hardneg-frame | 0.01 | **0.3053** | ± 0.0002 (0.1 %) | **+2.25** | ± 0.02 | 0.987 |

Bars are **measured**, not modelled: 1.96 × the standard error of the mean over
three independent 20 000-record iid samples. They agree with the bootstrap
intervals in the JSON to within ~1.2× (§6).

"material spread after" is the ratio of the median label:converted-teacher in the
**high**-material tercile to the **low**-material tercile. 1.000 means the
correction removed the material dependence; it lands in [0.97, 1.06] for 8 of 12,
worst 1.127 (wrongnnue) and 0.876 (fishpack).

### Same fit, the plain scalar, for comparison

| # | corpus | scalar slope | 95 % | R² | pearson | material spread (uncorrected) |
|---|---|---|---|---|---|---|
| 0 | uho-band75-100 | 0.3877 | ± 0.0034 | 0.857 | 0.926 | 1.449 |
| 1 | fishpack32 | 0.4424 | ± 0.0044 | 0.738 | 0.859 | 1.228 |
| 2 | farseerT74-band75-100 | 0.8305 | ± 0.0256 | 0.619 | 0.787 | 0.825 |
| 3 | datapv2-band75-88 | 0.4446 | ± 0.0028 | 0.846 | 0.920 | 0.940 |
| 4 | farseerT75-band75-83 | 0.7191 | ± 0.0214 | 0.614 | 0.784 | 0.843 |
| 5 | multinet-band75-88 | 0.4483 | ± 0.0024 | 0.832 | 0.912 | 0.980 |
| 6 | t60t70-band75-85 | 0.6033 | ± 0.0119 | 0.575 | 0.759 | 1.010 |
| 7 | pylon-band75-88 | 0.7612 | ± 0.0056 | 0.592 | 0.769 | 0.841 |
| 8 | wisr-band75-95 | 0.3862 | ± 0.0045 | 0.847 | 0.920 | 1.346 |
| 9 | wrongnnue-band75-95 | 0.5728 | ± 0.0133 | 0.525 | 0.725 | 1.271 |
| 10 | canon-tight-frame | 0.3925 | ± 0.0016 | 0.900 | 0.949 | 1.315 |
| 11 | hardneg-frame | 0.3238 | ± 0.0032 | 0.846 | 0.920 | 1.605 |

Intercepts are small relative to the label scale and are in the JSON; the
recommended law drops them, since a distillation target should be odd in the
score. Note the R² column separately: the teacher explains 53–90 % of label
variance depending on corpus, so a 40 % distillation weight is a materially
different proposition on `wrongnnue` (0.525) than on `canon-tight` (0.900).

### What a single constant actually costs

The **conversion multiplier** `k·blend^p` evaluated at each corpus's own material
p10 / p50 / p90. This is the honest quantity: `k` and `p` trade off, so `k`'s own
spread (2.20×) overstates the real one.

| # | corpus | p10 | p50 | p90 | within-corpus |
|---|---|---|---|---|---|
| 0 | uho-band75-100 | 0.333 | 0.379 | 0.497 | 1.49× |
| 1 | fishpack32 | 0.376 | 0.430 | 0.515 | 1.37× |
| 2 | farseerT74-band75-100 | 0.604 | 0.518 | 0.437 | 1.38× |
| 3 | datapv2-band75-88 | 0.433 | 0.416 | 0.385 | 1.12× |
| 4 | farseerT75-band75-83 | 0.510 | 0.442 | 0.382 | 1.33× |
| 5 | multinet-band75-88 | 0.385 | 0.388 | 0.393 | 1.02× |
| 6 | t60t70-band75-85 | 0.411 | 0.405 | 0.397 | 1.04× |
| 7 | pylon-band75-88 | 0.498 | 0.435 | 0.379 | 1.31× |
| 8 | wisr-band75-95 | 0.319 | 0.344 | 0.421 | 1.32× |
| 9 | wrongnnue-band75-95 | 0.381 | 0.398 | 0.449 | 1.18× |
| 10 | canon-tight-frame | 0.386 | 0.421 | 0.516 | 1.34× |
| 11 | hardneg-frame | 0.333 | 0.390 | 0.540 | 1.62× |

**Over the whole (corpus × material) grid the multiplier runs 0.319 → 0.604, a
1.90× spread.** The weight-averaged multiplier at median material is **0.4182**.
Using that one number everywhere costs, per corpus at its worst material decile:

```
farseerT74 +45%   hardneg +29%   canon +23%   fishpack +23%   farseerT75 +22%
pylon +19%        uho −20%       wisr −24%    wrongnnue −9%   multinet −8%
datapv2 −8%       t60t70 −5%
```

**Worst single-point error: +45 %; the full range is 0.762× to 1.445×.** Note that
uho — 25 % of the mix on its own — is mispriced −20 % at low material and +19 % at
high material *by the same constant*, so the error cannot be tuned away with a
better global scalar: it needs the per-corpus material exponent.

For scale, the `per_corpus_target` `a`-term spread that motivated going per corpus
in the first place is 1.73×. The teacher-frame spread measured here is **1.90×**,
slightly larger than the effect that already justified the per-corpus treatment.

---

## 5. The material term

### `to_cp` is a red herring

SF's UCI conversion divides by `a = p_a(material)` (`uci.cpp:501-517`), which
sounds material-dependent and is not:

| WDL material count | 17 | 25 | 35 | 45 | 58 | 68 | 78 |
|---|---|---|---|---|---|---|---|
| `p_a` | 388.2 | 382.9 | 381.0 | 382.4 | 385.5 | 386.0 | 382.4 |

±0.9 % over the entire legal range — the cubic is anchored at material 58 and is
nearly flat. **Treat `p_a` as the constant 385.** The `MA` model in the JSON
differs from `MB` by almost exactly the scalar 3.85 and adds no shape.

### The blend is the whole material law — but not at exponent 1

SF's `evaluate()` blend runs 1.05 (bare endgame) to 1.32 (startpos) and is
genuinely present. Applying it at exponent 1 (`MB`) moves the material tercile
spread 1.449 → 1.243 on uho and 0.825 → 0.713 on farseerT74 — it **under-corrects
one family and over-corrects the other**. Fitting the exponent fixes both.

The fitted exponents split the corpora into two families, and the split tracks
labelling provenance:

| p > 0 — label grows *faster* than the teacher with material | p ≤ 0 — label grows *slower* |
|---|---|
| hardneg +2.25, uho +2.14, wisr +1.87, fishpack +1.71, canon +1.52, wrongnnue +1.01 | multinet +0.11, t60t70 −0.17, datapv2 −0.61, pylon −1.32, farseerT75 −1.40, farseerT74 −1.57 |
| pearson 0.92–0.95 on uho / canon / hardneg | pearson 0.76–0.79 on farseerT74/T75 / pylon / t60t70 |

The p ≤ 0 family is exactly the Leela-derived and never-audited packs
(farseerT74/T75, t60t70, pylon) plus `datapv2`; the p > 0 family is the SF
self-play and SF18-relabelled packs. **The mechanism is not established and this
study does not claim one** — it is a measurement, and the exponent is a property
of the labelling process, which is precisely what per-corpus calibration exists
to absorb. Candidate mechanisms worth one experiment each: the labelling net
differed from the eca43a97 teacher and differed most in endgames; a search score
resolves to leaves with different material than its root; the Leela packs were
rescored rather than searched natively.

### Honest limit on the material correction

`p` is a **mean-shift** parameter, not a variance-reducer. Its bars exclude 0 for
10 of 12 corpora and it demonstrably flattens the tercile medians — but the
log-ratio R² it achieves is **0.000–0.103**. The per-position scatter of
label ÷ teacher is enormous (a 5 000-node search against a static eval) and
material explains almost none of it. **The correction removes a systematic bias;
it does not make the teacher agree with the label position by position, and
nothing here should be read as claiming it does.**

---

## 6. Nulls, gates and controls

**NULL PANEL (run before the arm, per the standing rule).** Permute the teacher
against the labels within each corpus:

| statistic | real | permuted null |
|---|---|---|
| scalar slope | 0.32 … 0.83 | −0.014 … +0.026 |
| R² | 0.53 … 0.90 | 0.000 … 0.002 |
| sign agreement (\|eval\| > 100) | 0.989 … 0.997 | 0.459 … 0.523 |

The instrument produces no signal on inert input. Note that `MP`'s `k` and `p`
are **not** null-safe by construction — `MP` conditions on sign agreement, so
under permutation it still returns numbers (garbage ones). `MP` is interpretable
only once the scalar null has passed, which it has.

**TEACHER CONTROL — against real Stockfish, on the exact inputs used here.**
Three canonicalized FENs from each of the twelve corpora (36 total; white to
move, no castling, no ep — exactly what this study feeds the teacher) run through
`engines/bin/stockfish-eca43a97`'s own `eval` with the same net: **36/36 exact
integer agreement** with `sf-eval`. Stockfish labels that line

```
NNUE evaluation          +3 (side to move, internal units)
```

which settles §2's units question from Stockfish's own mouth rather than from our
source reading, and simultaneously disposes of the worry that discarding castling
rights and the ep square might change the evaluation. It does not.

**SIGN AGREEMENT — the convention gate.** No corpus is anywhere near 50 %:

| # | corpus | > 0 | > 50 | > 100 | > 200 |
|---|---|---|---|---|---|
| 0 | uho-band75-100 | 0.875 | 0.988 | 0.996 | 0.999 |
| 1 | fishpack32 | 0.957 | 0.989 | 0.994 | 0.997 |
| 2 | farseerT74-band75-100 | 0.869 | 0.980 | 0.992 | 0.997 |
| 3 | datapv2-band75-88 | 0.925 | 0.990 | 0.997 | 0.999 |
| 4 | farseerT75-band75-83 | 0.861 | 0.979 | 0.991 | 0.997 |
| 5 | multinet-band75-88 | 0.873 | 0.989 | 0.995 | 0.999 |
| 6 | t60t70-band75-85 | 0.841 | 0.980 | 0.992 | 0.997 |
| 7 | pylon-band75-88 | 0.855 | 0.978 | 0.989 | 0.995 |
| 8 | wisr-band75-95 | 0.800 | 0.985 | 0.995 | 0.998 |
| 9 | wrongnnue-band75-95 | 0.854 | 0.987 | 0.995 | 0.999 |
| 10 | canon-tight-frame | 0.893 | 0.987 | 0.995 | 0.998 |
| 11 | hardneg-frame | 0.924 | 0.986 | 0.992 | 0.997 |

The pooled `> 0` column is 0.80–0.96 purely because the near-zero band is
coin-flippy by construction — a quiet-filtered corpus is full of |label| < 25
positions where the sign carries no information. Quoting only the pooled number
would have looked like a problem and was not one; quoting only `> 200` would have
hidden a real one. Both are reported.

**DECODER CONTROL.** Record → FEN → record reproduces `occ` and `pcs` byte-exactly
for **4800/4800** records sampled across all twelve corpora, with the re-encoder
written independently from the format spec. Independently, the decoded white-king
square matches the record's separately stored `ksq` field — which the decoder
never reads — **4800/4800**. A horizontal mirror or a nibble-order slip survives a
sign-agreement check and cannot survive either of these.

### RETRACTION: "the window sampler broke the error bars"

Mid-study I found that an independent-seed replication landed outside the
bootstrap 95 % interval for 3–5 of 12 corpora, concluded that the contiguous
500-record window sampler was the cause (same-game clustering inflating the
effective n), switched to iid sampling, and said so in a commit message.

**The first half was a wrong test and the conclusion does not survive it.**
Comparing one noisy estimate against *another noisy estimate's* interval is not a
coverage test: the difference of two independent estimates has √2 times the SE of
either, so a replicate sits inside a ±1.96 SE interval only ~84 % of the time even
when the interval is perfectly calibrated. Expected misses at n = 12 are ~1.9, not
~0.6 — so 3–5 was mildly high, not damning.

Re-tested correctly, as pairwise z-scores against the bootstrap SE inflated by √2:

| sampler | statistic | n comparisons | sd(z), expect 1.00 | \|z\| > 1.96, expect 5 % |
|---|---|---|---|---|
| window (chunk 500) | `k` | 12 | 1.30 | 17 % |
| window (chunk 500) | scalar slope | 12 | 1.12 | 8 % |
| **iid (chunk 1)** | `k` | 36 | **1.19** | **6 %** |
| **iid (chunk 1)** | scalar slope | 36 | **1.12** | **6 %** |

**The window-vs-iid difference is inside noise at these n.** What is real, and
present in both regimes, is a mild ~1.1–1.2× bootstrap under-statement — the
ordinary heavy-tail behaviour of a percentile bootstrap on an OLS slope whose
leverage is concentrated in a few extreme-|teacher| positions.

The iid sampler is kept because a record-level bootstrap *assumes* iid draws and
3.5 s per 20 000 records is not worth preserving an assumption for — but it fixed
a principle, not a measured defect, and the earlier commit message overstated it.
Point estimates moved < 2 % across the change.

**The bars in §4 are measured, not bootstrapped**: 1.96 × SE of the mean over the
three independent runs, which sidesteps the question entirely. They come to
0.1–5.9 % on `k` and 0.2–1.6 % on the scalar slope — far below every effect
claimed here (smallest: the 17 % `1/DIV` miss; largest: the 45 % pooled-constant
error).

**SAMPLING.** 20 000 iid records per corpus per run, three runs. Zero records
were dropped by the |label| ≤ 12 000 mate/TB guard in any corpus.

---

## 7. What to do with this

1. **Ship the §4 table as twelve `(k, p)` pairs indexed the same way as
   `per_corpus_target`** — positionally against `mix`, which §3 verified. The
   loader already carries per-record corpus identity (`src`) for exactly this
   reason (`data.py:187-189`), so a second per-corpus table rides the same index
   with no new plumbing.
2. **Do not use 1.023.** It is ~2.45× too large. **And do not use `1/SCORE_DIV`
   either** — it misses by 17–34 % on nine corpora. §2.
3. **Do not use a single pooled constant**, with or without a material term.
   Worst-case error +45 %, and uho alone (25 % of the mix) is mispriced in
   *opposite directions* at low and high material. §4.
4. `material` must be computed **per position** in the loader with SF's own
   weights (534 × pawns + non-pawn material, both colours, values N 781 / B 825 /
   R 1276 / Q 2538). It is one `bitwise_count` plus a table lookup over the
   already-decoded pieces.
5. **Re-measure if the teacher net changes.** Every number here is against
   `nn-0ee0657fb25e` via `archon-sfswapbin`. The table is a property of the
   *pair* (teacher net, corpus), not of the corpus alone.
6. **If tighter bars are ever wanted**, average more independent seeds rather than
   trusting a narrower bootstrap interval at the current `n` (§6).
7. **Open, and not closed by this study:** why `p` splits into two families by
   labelling provenance (§5). It does not block the blend — the calibration
   absorbs it either way — but it is the kind of thing that turns out to be a data
   defect often enough to be worth one experiment.
