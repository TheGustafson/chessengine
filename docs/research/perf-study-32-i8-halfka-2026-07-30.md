# ARCHON perf study #32 — halving the HalfKA feature-transformer row to i8

Date: 2026-07-30. Instruments: `tools/perf32/`. Artifacts:
`docs/research/perf-artifacts-32/`.

**LANDED.** The HalfKA feature transformer is now an i8 table with a per-column
integer scale, quantised at load. Measured **−6.835% cycles/node** (8/8 rounds
negative, null −1.382%) and **−17.42% load-stall cycles/node** (null −1.675%),
with strength **+3.6 ± 7.7 Elo over 2932 fixed-node games** across two
independent screens — i.e. no detectable fidelity cost.

At the project's own calibration constant (122.9 Elo per doubling of speed,
exp-0154) the speed is worth **+12.6 Elo**.

---

## 0. HEADLINE

Study 31 §4.3 named this the largest single unclaimed item in the engine:
52% of load-stall cycles are FT weight rows and 61% of those are HalfKA, so
halving the row was priced at **−463 cycles/node ≈ +17 Elo**. Study 29 confirmed
the target independently at 46.2% by a different instrument (traffic-pinning
probes rather than counter-census × geometry) and added the economics that make
it the right lever: weight rows miss at **38.3%** and cost **17.0 stall cycles
per miss**, against the accumulator stream's 14.6% and ≤2.

Six results, in the order they changed what got built:

1. **The fidelity question is answerable offline with zero engine work, and
   exactly.** Quantising a weight to i8 with scale `s` and multiplying by `s`
   into an i16 accumulator is *numerically identical* to storing `s·round(w/s)`
   in today's i16 table. So every scheme was written out as an ordinary CENN-v2
   net and scored by the shipped binary (§1). No model, no approximation.
2. **The pre-registered offline bar FAILED for every post-training scheme**
   (§2). Best of family was p95 4.35 / p99 7.25 SF cp against bars of 3.0 and
   5.0. Reported as a null, not smoothed over.
3. **The weight-MSE proxy MISRANKS the schemes** (§2.3). A per-column scale
   tuned to minimise weight MSE is better on MSE (1.78% vs 1.98%) and **worse
   on eval error** (p95 4.83 vs 4.35). The scale rule that shipped is the one
   the eval battery chose, not the one the L2 norm chose.
4. **Games settled it, and the offline bar was simply miscalibrated** (§3).
   Two independent 1466-game fixed-node screens — the quantised net against the
   reference net on one binary (isolating the SCHEME), and the landed binary
   against the pre-change binary on one net (isolating the IMPLEMENTATION) —
   pool to **+3.6 ± 7.7 Elo over 2932 games**.
5. **The format did not have to change** (§4). Quantising at LOAD, the way the
   D47V skip lane already does, keeps the CENN-v2 payload, `netpack.py`, the jax
   checkpoint contract and the trainer untouched, and every existing net —
   including D49's, still training — loads and gets the win for free.
6. **The i16 fold family is gone.** Both feature transformers now fold i8 rows,
   so there is one fold family instead of two, and it keeps study 25's
   shape-specialised const-generic dispatch (§4.3).

---

## 0.1 THE PASS BAR, PRE-REGISTERED — and it failed

*Committed in `2a8c021`, before the first eval delta was measured.*

| statistic (800-FEN battery, SF-frame cp) | bar | best scheme achieved | verdict |
|---|---|---|---|
| mean signed Δ | ±0.5 | −0.077 | **pass** |
| median \|Δ\| | ≤ 1.0 | 0.966 | **pass** |
| p95 \|Δ\| | ≤ 3.0 | 4.348 | **FAIL** |
| p99 \|Δ\| | ≤ 5.0 | 7.246 | **FAIL** |
| Spearman ρ | ≥ 0.9990 | 0.99980 | **pass** |

Three of five. I could not close the p95/p99 gap: the whole post-training scale
family bottoms out at ~1.58% relative weight error (§2.2), and the measured
mapping p99 ≈ 3.7 × rel% puts the bar at ≤1.35%, which no scale rule reaches.

**The honest reading is that the bar was the wrong instrument, and the way I
know that is a number, not an argument.** It was set from a precedent
(study 8's threat-frame change, "mean 0.14 SF cp") whose *mean* I beat by 3×,
and I guessed at the tail. Games — the standing arbiter — say the tail I was
worried about is worth nothing: **+3.6 ± 7.7 Elo over 2932 games** (§3). A bar
that rejects a change the arbiter cannot distinguish from zero is a bar that was
too tight.

I am recording the failure rather than restating the bar, because the next
person to quantise something in this engine should know that **p99 ≈ 7 SF cp of
static-eval perturbation is not measurably worth anything at 2932 games**, and
should set their screen accordingly.

---

## 1. THE INSTRUMENT THAT MADE PHASE 1 FREE

`tools/perf32/cennq.py` reads and writes CENN-v2. The identity it exploits:

```
    acc = Σ_f  s · q[f]        where q[f] = round(w[f] / s)
        ≡ the accumulator of an i16 table whose entries are s·round(w/s)
```

so an i8 scheme with any per-column integer scale can be *simulated exactly* by
an ordinary net, and scored with `chessengine nnue-eval`. The only fiddly part
is the 135 MB FNV-1a-64 payload checksum, which a Python loop cannot do in
reasonable time — it is compiled to a C shared object on first use.

This is what let the entire fidelity question, and the whole 1466-game strength
verdict, be answered **before a single line of Rust changed**.

**It was verified end to end after the fact.** The shipped engine's
`nnue-eval` output over all 800 battery positions is **byte-identical** to the
offline NumPy simulation of the same scheme (`d48x-i8-rust.cenn`). That is the
strongest correctness evidence in this study: two independent implementations
of the quantiser and the reduced-frame merge, agreeing on every position.

---

## 2. WHAT THE WEIGHTS ACTUALLY LOOK LIKE

### 2.1 The census (`tools/perf32/census32.py`)

D48X HalfKA, 22528 × 1280 = 28.8 M weights, in 255-frame integer units:

```
  mean -1.53   std 49.63   mean|w| 33.52   rms 49.66   zeros 5.89%
  p50 23   p90 76   p99 173   p99.9 307   p99.99 451   max 935  (= 3.667 in float)
  per-column max|w|:  p10 213   p50 334   p90 511   p100 935
  per-column no-clip integer scale: s=2 x273, s=3 x560, s=4 x312, s=5 x91, s>=6 x40
```

A single global scale would have to be 8 to avoid clipping (935/127), and that
is set by ONE outlier — the same pathology `tools/nnue/requant_threats.py`
found on the threat channel, where one weight out of 77.7 M pinned the scale
for all of them. The per-column rule spends each column's i8 range on its own
range and never clips.

### 2.2 The scale family, and where it bottoms out

Relative weight error (rms error / rms weight):

| scheme | rel err | note |
|---|---:|---|
| global s = 8 (no clip) | 4.609% | the naive choice |
| global s = 4 | 2.768% | the global optimum; clips 0.0044% |
| **per-column integer, no clip** | **1.982%** | **shipped** |
| per-column integer, MSE-optimal | 1.777% | clips 0.008% |
| per-column fractional, no clip | 1.661% | needs 32-bit lanes at the merge |
| per-column fractional, MSE-optimal | 1.587% | the family's floor |
| + per-row power-of-two shift | 1.584% | **null** — rows are homogeneous |

**A null worth recording:** per-ROW scaling buys nothing. Row maxima have a
p90/p10 spread of 2.09 against columns' 2.40, and the no-clip row shift chose
`k = 0` for 21,386 of 22,528 rows. The dynamic range of this table is a
column property, not a feature property.

For context, the threat channel already runs at **5.58%** relative error and
86% of all accumulator quantisation noise. This change adds *less* noise than
the engine has been carrying.

### 2.3 The proxy misranks — measure the eval, not the weights

800-FEN battery, SF-frame cp, reference = the canonical net:

| scheme | rel err | mean | median\|Δ\| | p95 | p99 | max | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| global s=8 | 4.609% | +0.300 | 2.415 | 11.111 | 17.879 | 42.5 | 0.99932 |
| global s=4 | 2.768% | −0.007 | 1.449 | 6.763 | 10.145 | 19.3 | 0.99977 |
| global s=4, bias on grid | 2.768% | +0.053 | 1.449 | 6.763 | 9.662 | 20.8 | 0.99977 |
| **per-column no-clip** | **1.982%** | **−0.077** | **0.966** | **4.348** | **7.246** | 11.6 | 0.99980 |
| per-column no-clip, bias on grid | 1.982% | −0.062 | 0.966 | 4.831 | 7.246 | 14.0 | 0.99976 |
| per-column MSE-optimal | **1.777%** | −0.118 | 0.966 | **4.831** | **8.213** | 10.6 | 0.99982 |
| shipped (round-half-away ties) | 1.982% | +0.045 | 0.966 | 5.314 | 7.729 | 10.6 | 0.99979 |

The MSE-optimal row is the finding: **11.5% better on the L2 norm of the weight
error and 11% worse at p95 of the eval error.** Its extra fidelity comes from
clipping 2,311 large weights, and large HalfKA weights are exactly the ones the
head has learned to lean on. Rule 3 of the standing method bar — prefer the
outcome you care about over the statistic that is easy to compute — earned its
place here.

So the shipped rule is the SIMPLEST one: `s_j = max(1, ceil(colmax_j / 127))`,
no clipping, ties rounded half-away-from-zero (the project's convention, and it
is load-bearing: at `s = 2` half the weights land exactly on a tie).

---

## 3. GAMES — THE ARBITER

**Fidelity, isolated from implementation.** Same binary, same everything,
`EvalFile` the only variable: the i8-simulating net against the canonical net,
fixed 25,000 nodes, UHO book, 1466 games (bar ±10 Elo).

```
Elo: +9.48 +/- 11.08,  nElo: +15.23 +/- 17.79,  LOS: 95.34%
Games: 1466  W 453  L 413  D 600   DrawRatio 46.4%  PairsRatio 1.12
Ptnml(0-2): [21, 164, 340, 170, 38]
```

Zero is inside the interval, so the honest statement is **"no detectable
strength cost"**, not "+9.5 Elo" — and the second screen below, which came in
at −1.90, is the reason to say so plainly rather than to enjoy the sign.

**The landed binary against the pre-change binary** (`tools/perf32/binscreen32.sh`),
same net file, fixed nodes — the gate on the *implementation* rather than on the
scheme, so it covers the kernels, the Finny paths and the reduced-frame merge:

```
Elo: -1.90 +/- 10.72,  nElo: -3.15 +/- 17.79,  LOS: 36.43%
Games: 1466  W 438  L 446  D 582   DrawRatio 47.1%  PairsRatio 0.97
Ptnml(0-2): [25, 172, 345, 168, 23]
```

**Pooled over both screens — 2932 games, inverse-variance weighted:
+3.6 ± 7.7 Elo.** Zero is inside, at a tighter bound than either screen alone.

The two point estimates differ by 11.4 Elo, which is **1.45 σ** on their
combined error — consistent with noise, and worth stating rather than picking
the flattering one. Neither screen alone excludes a ±10 Elo effect; together
they exclude ±8.

Fixed nodes hides the throughput win on purpose. The case is the two halves
together: **cycles say it is faster, 2932 games say it is not weaker.**

---

## 4. THE IMPLEMENTATION

### 4.1 Quantise at LOAD, not in the format

`SotaNetwork::halfka_weights` is now `Aligned<i8>`; the CENN-v2 payload is
**unchanged** and still carries i16. `sota_loader.rs` computes the per-column
scales in one pass over the payload bytes and quantises in a second, straight
into the i8 table — the i16 form is never materialised, so the loader's peak
footprint goes DOWN (which matters: a screen runs 2 × concurrency of these
processes and `tools/screen.sh` derives its concurrency cap from net size).

This is the `SkipLane` pattern (perf study 8 §4.7), and following it buys:

* every existing net gets the win with no re-export, including the D49 net that
  is still training;
* `netpack.py`, `rescale_net.py`, `jax/archon_jax/{spec,checkpoint}.py`,
  `strip_threats.py` and the bullet `SavedFormat` contracts are all untouched;
* nothing to keep in sync between two sides of a format change.

The serializer reconstructs `q · scale`, which is lossy against the original
file but **idempotent** — reloading derives the same scales and the same
quotients — so `load(serialize(load(x))) == load(x)` and the round-trip tests
hold.

### 4.2 The reduced frame, and where the scale is applied

The HalfKA accumulator now lives in the frame `FT_QUANT / scale[j]`, and
`activate` puts it back before the clamp:

```
    clamp(pst[j] * scale[j] + (threat[j] << 2), 0, 255)
```

one `vpmullw` per 32 lanes, on top of a shift the merge already did. **Applying
the scale at the merge rather than at the fold is a 3.6× cheaper place to put
it**: the fold runs 3.13 HalfKA rows per node (2.218 fold + 0.913 refresh,
study 31 §2.1) at 40 zmm ops each = 125 ops/node, while the merge runs at
0.4307 evals/node × 80 = 34.5 ops/node. The FT bias is rebased onto the same
per-column grid at load, so the accumulator seed and the rows share one frame
and the merge needs no extra add.

`pst * scale` is a **wrapping i16 multiply in all three backends** — that is
defined semantics, not an accident. The product is the value the old
FT_QUANT-frame accumulator held, so it wraps exactly where that one wrapped;
pinning the width is what keeps scalar (which would otherwise widen to i32),
avx2 (which multiplies in i16 *before* its widen) and avx512 bit-identical.

### 4.3 One fold family instead of two

Both transformers now fold i8 rows, so `i16_deltas`, `i16_deltas_into`,
`i16_deltas_dual`, `i16_into_shaped` and `i16_deltas_into_generic` are deleted
across all three backends, and the i8 family gains what they had:

* `i8_deltas` and `i8_deltas_dual` (new; the Finny cold build and warm refresh);
* `i8_into_shaped::<NA, NS>` — **study 25 arm 1's const-generic shape dispatch,
  ported**. This mattered: routing HalfKA onto the plain `i8_deltas_into` would
  have silently given back study 25's win (it measured the unshaped form at
  11.16% of all engine cycles for 6.93% of instructions). The four
  instantiations `(1,1) (1,2) (2,1) (2,2)` are the shapes
  `apply_dirty_directed` actually emits, and the threat channel now gets them
  for free.

The kernel-oracle tests move with them: every backend is still cross-checked
against the scalar reference at wrapping extremes, on long feature lists, on
all four shapes including the empty-list pure-copy contract, and on both homes
of the dual kernel.

---

## 5. THE MEASUREMENT

### 5.1 Frame and instrument

`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 50 positions × 100 k nodes ≈ 5.0 M nodes, `--features sota,arch-d48x`, fat
LTO, CGU = 1, `target-cpu=native`, **non-PGO**, `taskset -c 6`, `nice -n 19`.
Baseline binary built from `main` (`24b655e`) in the same frame.

`tools/perf32/panel32.sh` is **study 29's paired within-round form**, and that
is not a detail: study 29's prefetch arm read −1.67% cycles at n = 13 on a
merely *rotated* round-robin and nearly landed, and the paired form then showed
a byte-different null moving +1.5% while the arms moved +1.1%. So every round
runs base, a **byte-identical copy of base**, and the arm back to back in a
randomised order, and the figure is the median of the per-round ratio.

### 5.2 The panel — 8 rounds

| counter | base | NULL | ARM | rounds negative |
|---|---:|---:|---:|---:|
| **cycles/node** | 4786.41 | −1.382% | **−6.835%** | **8/8** |
| **load-stall cycles/node** | 2754.07 | −1.675% | **−17.421%** | **8/8** |
| **L1d misses/node** | 617.54 | −0.072% | **−6.542%** | **8/8** |
| L1d loads/node | 3049.04 | −0.096% | −0.626% | 8/8 |
| instructions/node | 5659.75 | +0.000% | +5.519% | 0/8 |

**The gate arms on cycles** (study 20 landed −5.99% cycles with instructions UP
0.19%; an instruction gate would have rejected it). Cycles clear their null by
**4.9×** and are sign-consistent 8/8. The mechanism row is better still:
load-stall cycles clear their null by **10.4×**, and L1d misses — the counter
study 31 measured at a ±0.16% null — clear theirs by **91×**.

**−6.835% of 4786.41 = −327 cycles/node**, i.e. **1.0734×**, i.e. **+12.6 Elo**
at 122.9 Elo/doubling.

### 5.3 Reconciling with the −463 forecast

Study 31 predicted −463 cycles/node from 125.2 HalfKA weight-row lines/node at
a residual 7.4 cycles/line. What actually happened:

* **The line-traffic prediction is essentially exact.** Load-stall cycles fell
  **480/node** (17.421% of 2754.07) against a forecast of 463 — 3.7% high. For
  a number study 31 called "a residual of a residual… the weakest number in the
  document", that is a much better showing than its own §5 discount rate
  (block-excess items realising 6.8%) would have predicted. **This one was an
  edit-priced item in disguise** — "this change removes N bytes at M sites" —
  and it realised like one.
* **The forecast omitted the cost side.** Instructions rose 5.519% (+312/node):
  the fold pays one `vpmovsxbw` per 32 lanes it did not pay before, and the
  merge pays one `vpmullw`. At the engine's 0.78 cycles/instruction that is
  ≈ +243 cycles, and −480 + 243 ≈ −237 against the −327 measured (the balance
  being the L1-miss reduction's own second-order effect on the front end).
* So the honest split is: **the bytes-moved model was right and the study that
  built it under-quoted its own accuracy; the Elo forecast was optimistic
  because it priced only the saving.** +17 predicted, +12.6 delivered — a 74%
  realisation, against a historic 76.5% for edit-priced items. Study 31's own
  discount table called this one correctly.

⚠ **Two cautions on the counters, stated rather than buried.** (a) The cycles
null is −1.382%, larger than an idle box would give — the D49 trainer was live
throughout — so the cycles figure is quoted with its null and rests on two
independent corroborating counters rather than standing alone. (b) On Zen 4 a
512-bit load raises **2.0015** `L1-dcache-loads` events per instruction and a
256-bit load raises 1.0016 (study 29, measured over 10⁸ issued loads). This
change converts 512-bit weight-row loads into 256-bit ones, so the
`L1d loads/node` row above **understates** the reduction in load
*instructions*, and no claim in this study is built on it.

---

## 6. GATES

This change alters eval output, so it is **neither node-identical nor
eval-bit-identical** and two of the three pins move — intentionally, in the
same commit, with the new numbers stated (CLAUDE.md's rule).

| gate | old pin | new pin | note |
|---|---|---|---|
| `Bench:` depth-6, default features, `x86-64-v3` | 587683 | **587683** | **unchanged** — the default build has no NNUE, exactly the blindness the NNUE gate exists to cover |
| NNUE gate, wl50 eval hash | `5ca978e0…8b38` | **`eb981b11…4f4f37`** | moves: the eval changed |
| NNUE gate, d13 node total | 1922947 | **1985398** | moves: +3.25%, the tree reshapes |
| `cargo test --features sota,arch-d48x` | — | **707 unit + 97 integration pass, 0 fail** | |
| `cargo clippy --all-targets` (default features, `-D warnings`) | — | **clean** | |
| `cargo fmt --check` | — | **clean** | |

`tools/check.sh`'s per-feature-combination loop caught one real defect this
study would otherwise have shipped: `src/policy.rs`'s `policy_trunk` call also
needs the scale, and `policy` is off in every shipped build so neither the
default clippy pass nor the sota one reaches that line.

Four warnings remain under `--features sota,arch-d48x` (`read_i8s`,
`NNZ_IOTA`, an `is_multiple_of` lint in `aligned.rs`, a `drop` lint in
`protected_halfka.rs`). **All four pre-date this study** — none of those four
sites was touched — and `tools/check.sh` clippies default features, which is
clean.

Reproduce: `bash tools/perf32/gates32.sh`.

---

## 7. THE TWO SCREENS, SIDE BY SIDE

| screen | what it isolates | n | Elo |
|---|---|---:|---:|
| net vs net, one binary | the quantisation SCHEME | 1466 | +9.48 ± 11.08 |
| binary vs binary, one net | the IMPLEMENTATION | 1466 | −1.90 ± 10.72 |
| **pooled** | | **2932** | **+3.6 ± 7.7** |

Both were run at 25,000 fixed nodes on the UHO book with `-repeat -games 2`, so
they are paired and independent of each other. The verdict is the pooled one:
**no detectable strength cost at ±8 Elo**, against a speed win worth +12.6.

---

## 8. WHAT I AM LEAST SURE ABOUT

Ranked by how much it would change the conclusion.

1. **The −6.835% cycles figure is a non-PGO probe-frame number on a contended
   box.** The probe→PGO translation has gone wrong in both directions in this
   project's history (studies 12–15 over-stated by 1.12–2.0×, study 17 by
   2.17×, study 18 *under*-stated by 2.94×). The mechanism rows (load-stalls,
   L1 misses) are much better anchored than the cycles row, and the cycles null
   at −1.382% is the reason I quote the ratio to its own null rather than as an
   absolute. **Nobody should quote +12.6 Elo as measured strength — it is a
   speed measurement times a calibration constant.**
2. **The strength verdict is "indistinguishable at ±8 Elo", and a −4 Elo
   fidelity cost would be invisible to it.** That is the honest resolution of
   this instrument even pooled over 2932 games. It is enough to land a change
   whose speed is worth +12.6; it is not enough to claim the quantisation is
   free. A 5,880-game screen (±5 Elo) would tighten it and costs about 80
   minutes. The 1.45 σ disagreement between the two screens is the visible
   reminder of how much room is left at this `n`.
3. **The per-column scales are derived from the D48X net's weight
   distribution, and they are derived at load, so they follow whatever net is
   loaded.** That is the right behaviour, but it means a future net with a much
   heavier tail gets coarser scales automatically and silently. There is no
   tripwire on the scale histogram. If a net ever needs `s ≥ 16` on many
   columns, this study's fidelity numbers do not transfer and nothing will say
   so.
4. **`pst * scale` can wrap.** It wraps exactly where today's FT_QUANT-frame
   accumulator wraps, all three backends wrap identically, and 800 battery
   positions plus 1466 games show no sign of it — but I did not instrument the
   accumulator's dynamic range to prove the margin, and there is no
   `debug_assert` on it the way there is on the threat channel's ±8191 bound.
   That is the cheapest remaining piece of hardening.
5. **I did not re-verify the Zen 4 double-pump ratio on my own build**, as
   study 29 advised. I inherited it, and the only row it would affect is the
   `L1d loads/node` row, which carries no claim. Every load-bearing number here
   is cycles, stall cycles, L1 *misses*, instructions, or games.
6. **The offline battery is 800 positions from `scale-check-800.fens`** and
   the p95/p99 tails are estimated from it. A different position mix would move
   the tail figures; it would not move the game result, which is what decided
   this.

---

## 9. WHAT THIS OPENS

* **The HalfKA table is now 28.9 MB, inside the 32 MB L3 for the first time.**
  Study 31 predicted the residual cost-per-cold-line would itself fall once the
  table fit; the −480 stall cycles/node came in 3.7% ABOVE its linear
  prediction, which is weak evidence that it did. A direct L3/DRAM fill-depth
  measurement (`ls_dmnd_fills_from_sys.*`) would price it.
* **Study 31's items 1 and 2 are not exhausted.** The threat table is 77.7 MB
  and already i8; its remaining lever is the co-access permutation (§4.1
  there), which is bit-identical and needs no retrain.
* **This composes with width.** At `ACC = 1024` with i8 rows the HalfKA table
  is 23.1 MB. The two are independent.
* **The fold now widens.** `vpmovsxbw` per 32 lanes is +312 instructions/node
  of pure format conversion. A layout that lets the accumulate consume i8
  directly — or a 4-bit second tier for the low-magnitude columns — would take
  that back, and it is now the largest identified cost inside this change.
