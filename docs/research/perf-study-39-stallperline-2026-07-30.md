# Perf study 39 — the stall-per-line disagreement, closed

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1d,
1 MB L2/core, 32 MB L3 · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Base:** `main` at `ad6d459` ·
**Artifacts and instruments:** `docs/research/perf-artifacts-39/`,
`tools/perf39/`

**Commission:** studies 29 and 31 priced the same quantity — one 64-byte line of
accumulator streaming — in opposite directions on the same night, agreed on
misses per line to 8%, and disagreed **5.2×** on stall cycles per line (29:
**0.45**, subtractive; 31: **2.36**, additive). Both deferred it, both said
"close this first", and nobody had. It is the entire width of accumulator
fusion's bracket: **+48 cycles/node (a loss) to −136 (+4.7 Elo)**.

---

## ★ HEADLINE

### 1. Which number is right: **neither**. The right value is ~0.8–1.1 stall / ~0.9–1.2 total cycles per line, and it is a THIRD measurement.

Both studies measured competently and I reproduced both on today's engine, in
one panel, to within a few percent. They measured **different things**, and the
thing that prices fusion is neither of them:

| | study 29 | study 31 | **study 39** |
|---|---:|---:|---:|
| direction | subtractive | additive | **subtractive** |
| tree held identical? | **no** | yes | **yes** |
| instructions held identical? | **no** (−7.6%) | n/a (they *are* the traffic) | **yes (+0.019%)** |
| misses/line, vs the real stream's **0.861** | n/a | 0.477 (**warmer**) | **0.861** |
| stall cycles per line | 0.45 | 2.36 | **1.119** (panel 1) / **0.803** (panel 3) |
| reproduced here | **0.350** (its own statistic: −4.503% vs its −4.63%) | **2.22** fitted, **3.18** at dose 1 | — |

The measurement that carries no confound is **`dstpin` differenced against
`actpin0`**, and it exists only as a difference of two individually broken arms:
both pin `activate` to the same all-zero buffer, so both compute the same wrong
evaluation, search the same tree, and retire the same instructions (**+1.05 i/n
out of 5474**). Their difference relocates **154.92 lines/node** of *real* fold
accumulator traffic from cold stack memory to a hot 2.5 KB buffer and changes
nothing else: **−133.46 misses, −173.29 stall cycles, −187.98 cycles per node,
24/24 sign-consistent on misses.**

**The same arm run twice does not repeat to better than 1.4×.** Panel 3 re-ran
it on a busier box (load average 3.5 → 6.9, another agent building) and got
−124.36 stall and −139.28 cycles for the identical 154.9 lines — 0.803 and
0.899 per line against panel 1's 1.119 and 1.213. Both are 24/24
sign-consistent on misses and agree on the miss count to 2%; it is the
cycles-per-miss conversion that moves with box load, which is the same axis the
whole study is about. **§6 prices fusion at the panel-1 (higher) value on
purpose — it is the optimistic end for fusion, and the verdict survives it.**

### 2. Why the two differed — factorised exactly, no residual

Between my own two clean arms the gap is **2.85×**, and it multiplies out:

```
stall per LINE, additive / subtractive           2.85x
  of which MISS COUNT (1.272 vs 0.861 miss/line) 1.48x
  of which MISS DEPTH (2.50 vs 1.30 stall/miss)  1.93x
  product                                        2.85x
```

**Both factors are properties of the PROBE, not of the direction it points.**

* **Miss count (1.48×).** An additive probe built out of fresh buffers necessarily
  touches colder memory than the engine's own recycled per-ply stack: my ring
  lines miss at 1.272 per line where the real fold stream misses at 0.861 (a
  fully cold line costs ~1.21 events on this counter — §1).
* **Miss DEPTH (1.93×).** An added miss costs 2.50 stall cycles, a removed one
  saves 1.30 — and this is *not* an additive/subtractive asymmetry, it is where
  the miss is served from. Sweeping the probe's own footprint 10 KB → 160 KB →
  2.6 MB at **byte-identical instructions** moves stall cycles per miss
  **1.759 → 2.503 → 3.203** while the miss *count* barely changes (§4.4). The
  real accumulator stack sits at **1.298**, below the smallest probe — an
  L2-resident ~10-ply hot window. An additive probe built from fresh buffers
  misses further away than the engine's own recycled stack, by construction.
* The remaining factor from 2.85× to the published 5.2× is **study 29's
  per-instruction normalisation**, which I replicated: raw **0.913** →
  normalised **0.350** on the same data, a 2.6× move from a modelling choice.

**A correction to my own first draft of this section.** Study 31's probe was not
colder than the real stream — it was **warmer**, at 0.477 misses/line against
the stream's 0.861, and `P39_SRC=live` reproduces that (§7) by having the copy
read the accumulator `activate` is about to consume. What its number carries is
therefore the **miss-depth** factor, not the miss-count factor: its stall *per line*
(2.36) replicates my additive arm's (2.22 fitted, 3.18 at dose 1) almost exactly,
even though we disagree 2.7× on how many misses that line took. **Study 31's
additive measurement was never the thing that was wrong.**

### 3. Study 29's proposed reconciliation is FALSIFIED

Study 29 §4.1 proposed that both numbers are correct in their own regime because
"marginal cost at a capacity cliff exceeds average value below it". That
requires the cost function to be **convex** at our operating point. It is not:

* the **dose series** (0 → 1 → 2 → 4 copies per evaluated node, 24 rounds,
  tree bit-identical, 24/24 sign-consistent at every dose) has marginal slopes
  **3.184 → 1.422 → 1.980** stall cycles/line. Not monotone, not rising —
  if anything mildly *saturating*, which is the opposite sign;
* changing the probe's own footprint **260×** (10 KB → 160 KB → 2.6 MB) at
  byte-identical instructions moves stall by −31% / 0 / −5% and misses barely at
  all. A capacity cliff is precisely a working-set effect; this working set does
  not find one.

### 4. A NEW fact that hurts fusion, and neither study had it

**`activate`'s four accumulator arrays do not miss.** Relocating all 68.98
lines/node of them to a hot buffer changes L1 misses by **−0.22/node, i.e.
0.0032 misses per line** — they were written by the fold at this ply moments
before and they are still resident. Study 28 and study 31 both counted that
10.24 KB/eval merge as roughly a third of fusion's prize. In stall cycles it is
worth **zero**; halving it buys issue work and nothing else.

### 5. THE FUSION DECISION: **do not build it** — and now for measured reasons on both sides

Priced on today's rates in **one currency** (study 31 kept an instruction ledger
and a cycles ledger and quoted only the second; instructions cost cycles):

| | cycles/node |
|---|---:|
| saving — threat fold pass, 68.30 lines × 1.213 | −82.9 |
| saving — `activate` merge, 34.49 lines, **measured not to miss** | **−0.0** |
| saving — issue, −118 to −226 i/n at IPC 1.453 | −80.9 … −155.6 |
| **total saving** | **−163.8 … −238.4** |
| cost — 12.06 cold threat weight-row lines at 6.5–13 cycles | +78.4 … +156.8 |
| cost — `dst.fill(0)`, 5.80 lines | +7.0 |
| cost — rebuild issue, 189–407 i/n at IPC 1.453 | +129.8 … +280.4 |
| **total cost** | **+215.3 … +444.3** |
| **NET** | **−23.1 (best) … +280.5 (worst)** |
| **⇒ Elo at 120/doubling** | **+1.0 … −11.5** |

The saving column uses **panel 1's** 1.213 cycles/line, the higher of the two
measurements of that arm. On panel 3's 0.899 the fold saving falls from −82.9 to
−61.4 and the best case becomes **−1.6 cycles/node ⇒ +0.1 Elo**. The verdict is
the same either way, which is the point of quoting the optimistic one.

**At the most favourable end of every input, fusion is worth one Elo.** It is
not a lever; it is a large architectural change whose *entire* expected value is
inside the noise of a 400-game match, and it additionally trades away a measured
**8.4× refresh advantage** over Stockfish (our threat channel rebuilds
0.0196/node; fused it inherits HalfKA's 0.1647). Study 29's verdict stands. Its
reasoning does not.

### Gates, reproduced on the landing tree

`tools/perf39/gates39.sh`, which runs `CHECK_NNUE_GATE=1 bash tools/check.sh`
because `src/eval/nnue/` was touched and **the `Bench:` pin is blind to NNUE
work**:

```
  eval hash  : 2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4
  d13 nodes  : 1739588
Bench: 587683
All checks passed.
```

All three pins are the current ones, unmoved. The probe features are
deliberately **not** in the gate — they compute wrong evaluations by
construction, so a green suite under `probe-dstpin` would mean the probe was
inert — but `foldcount` **is** gated, because its entire claim is that it
changes nothing.

### What landed

**Instruments and a decision, no strength change.** `probe-addcopy` (study 31's
additive probe rebuilt, with `P39_DOSE`/`P39_RING`/`P39_SRC`/`P39_SITE` as
*runtime* knobs so one binary serves the whole panel and code layout is
differenced away), `probe-actpin` (+`P39_PINFILL=live`), a `thr_refresh`
foldcount site, and `tools/perf39/` — build, identity gate, counter calibration,
workload census, two panels, dose regression, the study-29 replication, the gap
decomposition and the fusion ledger. All off by default; the shipped build is
byte-untouched and all three pins reproduce.

---
## 0. METHOD — and the one thing that made this tractable

### 0.1 Frame

`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 50 FENs × 100 k nodes, **5,001,562 nodes**, one thread, 16 MB TT, D48X @
`network_scale=600`. Shipped release profile (fat LTO, CGU = 1),
`-C target-cpu=native`, non-PGO, pinned to one core with `taskset`. One build
path (`tools/perf39/build39.sh`), one panel runner
(`tools/perf39/panel39.sh`), order **randomised inside every round** and every
delta taken **paired within the round** — study 29 §5.3 retracted a four-round
result that turned out to be measuring position-in-round, and study 25 §0.2
showed a fixed rotation cancels monotone drift but not curvature.

### 0.2 The instrument change that mattered: runtime knobs, one binary

Study 31's additive probe was never committed, so it had to be rebuilt. Rather
than rebuild it as a feature, **every knob is an environment variable and the
whole additive panel is ONE binary**: `P39_DOSE` (copies per evaluated node),
`P39_RING` (how many 5120 B slots they rotate through), `P39_SRC`
(`ring`/`live`), `P39_SITE` (`activate`/`fold`).

That is not a convenience. Study 29 needed *two* nulls — a byte-identical copy
and a different binary with the same behaviour — because its arms were separate
builds and a 1.2 MB binary's code layout is its own effect. Here every additive
arm is the same bytes, so **code layout is differenced away by construction**,
and the byte-identical null is left in as the error bar.

### 0.3 Null first

Two nulls, both carried through all 24 rounds of the main panel, both against
the control `d0` (the probe binary with `P39_DOSE=0`):

| null | Δ `stall/n` | Δ `cyc/n` | Δ `miss/n` | Δ `ins/n` | sign |
|---|---:|---:|---:|---:|---|
| `d0null` — **byte-identical** copy of the probe binary | **+2.709%** | +1.799% | +0.338% | +0.000% | 8/24, coin flip |
| `base` — a **different binary**, same behaviour | +0.714% | +0.909% | +0.024% | −0.080% | 9/24, coin flip |

**Read the consequence as a gate.** The floor on `stall/n` is ±2.7%. The
smallest additive arm moves it **+11.183%** — 4.1× the floor — and is negative
in **0 of 24** rounds. Every arm quoted below either clears that floor several
times over with a perfect sign count, or is reported as no effect.

---

## 1. THE CALIBRATION NOBODY HAD DONE — and a correction to study 29's census

Study 29 §2.1 calibrated `L1-dcache-loads` against issued zmm loads (2.0015
events per load: Zen 4 double-pumps a 512-bit access on a 256-bit datapath) and
built its whole line census on that. **It never calibrated the counters against
traffic that also STORES**, and both studies quote a "misses per line" off
`L1-dcache-load-misses`.

`probe-addcopy` moves exactly `DOSE × 5120 B` read and the same written per
evaluated node, so the dose difference is a ruler with no modelling in it.
`tools/perf39/calib39.sh`, dose 4 − dose 0, against **137.95 lines read +
137.95 written per node**:

| counter | Δ/node | per line read | per line R+W |
|---|---:|---:|---:|
| `ls_dispatch.ld_dispatch` | 141.08 | **1.023** | 0.511 |
| `ls_dispatch.store_dispatch` | 281.86 | 2.043 | **1.022** |
| `L1-dcache-loads` | 566.73 | 4.108 | **2.054** |
| `L1-dcache-load-misses` | 334.57 | 2.425 | **1.213** |
| `ls_mab_alloc.load_store_allocations` | 322.61 | 2.339 | **1.169** |

Three facts fall out, and the first is a correction:

* **`L1-dcache-loads` counts STORES.** 141.08 load dispatches double-pumped to
  282.16 events, plus 281.86 store dispatches, is 564.0 — the measured 566.73.
  A load-only microbenchmark cannot see this, which is why study 29's
  calibration missed it. **Its per-site "loads/node" census therefore mixes
  loads and stores**, and any future comparison of a hand-counted kernel
  against that counter must count both.
* The copy issues **one 64-byte load per line** (zmm) and **two 32-byte stores**
  — Zen 4's store datapath is half the load datapath.
* **A fully cold 64-byte line costs ~1.2 `L1-dcache-load-misses` events**, not
  1.0. So a "misses per line" of 0.477 or 0.516 is **~40–43% of lines cold**,
  not 48–52%. This does not move the ratio between the two studies — they
  agreed on this column — but it is the scale that lets an arm's coldness be
  read off the counter, which §4 needs.

---

## 2. THE WORKLOAD CENSUS — today's rates, re-derived

`tools/perf39/census39.sh`, `foldcount` build, same 5,001,562-node frame. Rule
8: studies 29 and 31 both priced fusion against a census taken **before** perf
study 32 (the i8 HalfKA transformer) and **before** study 38 (the threat fold's
parent alias), and one of those rates has moved by 15×.

| site | calls/node | rows/node | study 29 (2026-07-30, earlier) |
|---|---:|---:|---|
| `hka_into` — HalfKA parent→child fold | 0.9784 | 2.2295 | 0.9749 / 2.2176 |
| `hka_dual` — Finny **warm** refresh | 0.1647 | 0.9422 | 0.1610 / 0.9132 |
| `thr_into` — FullThreats fold | 0.8538 | 3.9222 | 0.8498 / 3.9123 |
| `thr_copy` — threat empty-delta copy | **0.0220** | — | **0.3339** |
| `activate` / `eval` | 0.4311 | — | 0.4258 |

**The threat empty-delta copy is gone** — study 38's parent alias took it from
0.334 calls/node to 0.022, i.e. study 29's ranked item #3 has already been
harvested. Everything else is within 2%.

`activate` at **0.4311/node** is the denominator every "per line" number in
this study divides by, and it is measured, not assumed: one dose of the probe
is `160 lines × 0.4311 = **68.98 lines/node**`, which is study 31's 68.9 to
0.1%. **The two probes are dose-matched by measurement, not by hope.**

**The threat refresh, re-derived** (`thr_refresh`, a foldcount site this study
added because study 29 §8.1 says outright "the threat-refresh row count was not
measured" and fusion's cost side depends on it): **98,024 calls / 407,362 rows
over 5,001,562 nodes = 0.01960 rebuilds/node at 4.156 rows each.** Study 31
measured 0.01955 and 4.18 with its own counter. **Independently reproduced to
0.5%** — that number is solid and can be built on.

---

## 3. THE DOSE SERIES — the additive answer, and the falsification

`probe-addcopy` adds `DOSE` redundant accumulator-pair copies per evaluated
node. One copy = 5120 B read + 5120 B written = **160 lines**, and at the
measured 0.4311 evals/node one dose is **68.98 lines/node** — study 31's 68.9
to 0.1%. **The node count is bit-identical to `base` at every dose
(5,001,562)**: the probe does not touch the evaluation, so the tree cannot
move.

24 rounds, randomised order, every delta paired inside its round, against `d0`
(the same binary at dose 0):

| arm | dose | lines/n | Δ stall/n | Δ miss/n | Δ ins/n | Δ cyc/n | stall/line | sign |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `d0` | 0 | 0 | — | — | — | — | — | — |
| `d1` | 1 | 68.98 | **+219.63** | +87.74 | +78.90 | +294.75 | 3.184 | **24/24** |
| `d2` | 2 | 137.95 | **+317.68** | +170.56 | +155.64 | +442.50 | 2.303 | **24/24** |
| `d4` | 4 | 275.90 | **+590.84** | +322.15 | +309.13 | +875.93 | 2.141 | **24/24** |

Least-squares slope through the origin: **2.222** stall cycles per line on the
medians, **3.032** on the quietest round (the round whose control stalled least
— the coordinator's "median AND min" rule). Study 31's 2.36 sits inside that.
**The additive number replicates.**

### 3.1 The capacity cliff is falsified, twice

Study 29 §4.1's reconciliation — "marginal cost at a capacity cliff exceeds
average value below it" — requires the cost function to be **convex** at our
operating point. Two independent tests say it is not.

**Test A, the marginals.** Between consecutive doses:

```
    0.0 ->  69.0 lines/n :  3.184 stall cycles/line
   69.0 -> 138.0 lines/n :  1.422
  138.0 -> 275.9 lines/n :  1.980
```

Not monotone and not rising. If anything the first dose is the dearest, which
is mild *saturation* — convexity with the sign reversed.

**Test B, the footprint sweep.** `P39_RING` changes how much memory the probe's
own copies rotate through, at **byte-identical instructions** (`ins/n` differs
by 0.14 out of 6007 across the three arms):

| arm | probe footprint | Δ stall/n | Δ miss/n | Δ loads/n |
|---|---:|---:|---:|---:|
| `d1r1` | 10 KB | +150.77 | +85.71 | +139.27 |
| `d1` | 160 KB | +219.63 | +87.74 | +143.39 |
| `d1r256` | 2.6 MB | +208.22 | +65.00 | +179.63 |

**260× of working set moves the cost by ~30% and misses by ~25%, without a
knee.** A capacity cliff is by definition a working-set effect; sweeping the
working set across L1, L2 and L3 does not find one.

*(A confound in my own first series, found and fixed: with a fixed ring, raising
the dose shortens the probe's own reuse distance — 16 evaluations at dose 1, 4
at dose 4 — which is visible above as misses per dose falling 87.7 → 85.3 →
80.5. `tools/perf39/panel2.sh` re-runs the series with `RING` scaled to `DOSE`
so the reuse distance is constant; §7.)*

---

## 4. THE MEASUREMENT THAT CARRIES NO CONFOUND

### 4.1 It is a difference of two broken arms

`probe-dstpin` and `probe-actpin` both redirect `activate`'s four accumulator
arrays to the same all-zero buffer. They therefore compute the **same wrong
evaluation**, search the **same tree**, and — measured, not assumed — retire the
**same instructions**:

| arm vs `d0` | Δ ins/n | Δ miss/n | Δ stall/n | Δ cyc/n |
|---|---:|---:|---:|---:|
| `actpin0` — pins `activate`'s reads only | −454.80 | −0.22 | −47.80 | −121.58 |
| `dstpin` — pins those **and** all fold reads and writes | −453.75 | −133.07 | −201.45 | −297.27 |

`dstpin` **− `actpin0`**, paired within round, n = 24:

| counter | Δ | per line (154.92 lines/node) | sign |
|---|---:|---:|---|
| instructions | **+1.05** (+0.019%) | — | — |
| L1 misses | **−133.46** | 0.861 | **24/24** |
| load-stall cycles | **−173.29** | **1.119** | 23/24 |
| cycles | **−187.98** | **1.213** | 22/24 |

The 154.92 lines are not an estimate — they are the `pin_src!` / `pin_dst!`
sites multiplied by the measured call rates: HalfKA fold src+dst 78.27, threat
fold src+dst 68.30, Finny warm refresh's pinned destination 6.59, threat
empty-delta copy 1.76.

**This is a subtractive measurement of the engine's real fold accumulator
stream with the tree, the instruction count and the head's sparsity all held
fixed.** It is what neither study had, and it exists only because the two dirty
arms are dirty in exactly the same way.

### 4.2 `activate`'s accumulator reads do not miss

Look again at the `actpin0` row: relocating **68.98 lines/node** of `activate`'s
accumulator reads to a hot buffer changes L1 misses by **−0.22/node** —
0.0032 misses per line, against 0.861 for the fold's traffic and 1.272 for the
probe's cold ring. They are read moments after the fold at this ply wrote them
and they are still resident.

The 10.24 KB/evaluation that Stockfish carries as 4.10 KB — the single most
visible symptom of not fusing, and roughly a third of the prize in both study
28's and study 31's ledgers — **costs nothing in stall cycles.** Study 29's §3.2
saw the shadow of this from the sampled side (`forward_into` is 12.25% of loads
and 2.18% of load-stall cycles) and did not draw the consequence.

### 4.3 The 5.2×, accounted for

```
stall per LINE, additive / subtractive           2.85x
  of which MISS COUNT (1.272 vs 0.861 miss/line) 1.48x
  of which MISS DEPTH (2.50 vs 1.30 stall/miss)  1.93x
  product                                        2.85x
```

and the remaining factor to the published 5.2× is study 29's normalisation
(§5.3), which moves its own data from 0.913 to 0.350. Both factors are
properties of the PROBE, not of the direction it points — §4.4 measures the
second one directly.

### 4.4 The "direction" factor is mostly a MISS-DEPTH factor, and it is measured

I first wrote §4.3's second factor up as an unexplained asymmetry. It is not
unexplained — the footprint sweep already in the panel resolves most of it. All
three arms are the **same binary with the same instruction count** (6007.5 i/n,
differing by 0.14) and differ only in how much memory the copies rotate through:

| arm | probe footprint | miss/line | **stall cycles per MISS** |
|---|---:|---:|---:|
| `d1r1` | 10 KB | 1.243 | **1.759** |
| `d1` | 160 KB | 1.272 | **2.503** |
| `d1r256` | 2.6 MB | 0.942 | **3.203** |
| *the real fold stream* (`dstpin − actpin0`) | ~50–100 KB hot window | 0.861 | **1.298** |

**The miss COUNT barely moves and the price per miss nearly doubles.** These are
the same L1 misses being served from progressively further away: a 10 KB ring is
evicted from L1 between evaluations but stays in L2; a 2.6 MB one is gone to L3
or memory. And the real accumulator stack sits at the shallow end — 1.298, just
below the smallest probe — exactly where an L2-resident ~10-ply hot window
should sit.

So the additive/subtractive gap is not really *additive versus subtractive*. It
is **cold-and-far versus warm-and-near**, and an additive probe built from fresh
buffers lands on the wrong side of that by construction unless its footprint is
tuned to match — which nobody, including me at first, thought to do.

**What survives as genuinely unexplained** is the last ~1.35× between the real
stream's 1.298 and the nearest probe's 1.759. I suspected `dstpin`'s
single-buffer store funnelling was depressing the subtractive number; §7.3
splits the read and write halves and finds it is **not needed as an
explanation** — the read half, where no funnelling exists, carries 79% of the
cycles saving by itself. Funnelling is not excluded; nothing requires it.

---
## 5. WHY THE SUBTRACTIVE PROBE ANSWERS A DIFFERENT QUESTION

The additive and subtractive probes are not two views of one quantity. The
subtractive one changes the engine's **workload**, twice over, and both changes
are measured here rather than argued.

### 5.1 A zero-filled pin turns off the second half of the network

`probe-accpin` / `probe-dstpin` redirect the accumulator reads to a buffer of
**zeroes**. `activate` then emits an all-zero `ft`, so `find_nnz` returns almost
nothing and `l1_dot` — the sparse L1 layer, which has its own weight rows and
its own misses — does almost no work. Study 29 saw the resulting instruction
drop and called it *compiler CSE* (§3, "`accpin` deletes 6.8% of the engine's
instructions to compiler CSE").

It is not CSE, and the panel shows it three ways:

* **All three pinning arms lose the same instructions regardless of what they
  pin.** `actpin` redirects only `activate`'s four arrays and leaves the fold
  `src` stream completely alone; `dstpin` redirects the folds' reads *and*
  writes as well. If the drop were CSE of the pinned loads they could not be
  within 1 i/n of each other. They are.
* **`P39_PINFILL=live` restores it.** Same binary, same code layout, one
  runtime branch: fill the pin from the first REAL accumulator instead of
  zeroes, so magnitudes and group sparsity are realistic and the head does its
  normal work. Instructions go from −7.7% to **+4.7%**.
* **Study 29's own table already said so.** Its `accpin` row reports
  `stall/instr` **+8.21%** — a probe that *removes* accumulator read traffic
  cannot raise stall per instruction by 8% by removing traffic. It raised it by
  shrinking the denominator.

### 5.2 And it changes the search tree

`foldcount` on the same binary (`tools/perf39/census39.sh`) — the node TOTAL is
fixed by the `nodes` budget, but the node MIX is not:

| build | evals/node | `hka_into`/node | `thr_into` rows/node |
|---|---:|---:|---:|
| `fc-base` | 0.4311 | 0.9784 | 3.9222 |
| `fc-actpin`, zero fill | 0.4635 (**+7.5%**) | 0.9988 | 3.8511 |
| `fc-actpin`, live fill | 0.5191 (**+20.4%**) | 1.0959 | 4.1645 |

A pin that computes wrong evaluations changes what the search does. It cannot
not: the evaluation drives the tree. **So there is no tree-clean subtractive
probe for accumulator streaming short of building fusion itself** — and that,
not a capacity cliff, is why the two directions disagree.

### 5.3 The normalisation is where most of the 5.2× comes from

Study 29 handled its instruction drop by normalising: report **stall cycles per
instruction**, on the assumption that the deleted instructions carried their
proportional share of stalls. `tools/perf39/replicate29.py` recomputes exactly
that statistic on today's panel — and the raw and normalised numbers are a
factor apart.

The normalisation is not obviously wrong; it is **unmeasured**. The deleted work
is `l1_dot` walking sparse weight rows, a memory-bound layer whose stall density
is not the engine average and was never measured. Rule 4 — one inference deep,
then measure — is exactly the rule it skips, and the choice moves the answer by
more than a factor of two.

The replication, `tools/perf39/replicate29.py`, n = 24, `dstpin` against `base`
over study 29's 253.7 lines:

| | study 29 | study 39, today's engine |
|---|---:|---:|
| Δ instructions | −6.5% | **−7.58%** |
| Δ `stall/instr` (**its published statistic**) | **−4.63%** | **−4.503%** |
| ⇒ normalised stall cycles/node | −114.6 | **−88.83** |
| ⇒ stall cycles per line | **0.45** | **0.350** |
| Δ stall cycles/node, **RAW** | *(not published)* | **−231.62** |
| ⇒ stall cycles per line, **RAW** | *(not published)* | **0.913** |

**Study 29 is reproduced to 2.7% on the exact statistic it published.** Its
number is not a measurement error, a box difference, or an engine-version
difference. It is a correct measurement of a quantity that is not the cost of
accumulator streaming, pushed through a normalisation that moves it 2.6×.

---

## 6. THE FUSION LEDGER, RE-DERIVED

Rule 8. Studies 29 and 31 priced fusion against a census taken **before** perf
study 32's i8 HalfKA transformer and **before** study 38's threat-fold parent
alias. `tools/perf39/fusion39.py` runs it on today's rates, in one currency.

**Saving.** A fused accumulator does one parent→child fold pass where we do two,
so one pass's accumulator stream disappears, and `activate` reads two arrays
instead of four.

| | lines/node | cycles/line | cycles/node |
|---|---:|---:|---:|
| threat fold pass, parent read + child write | 68.30 | 1.213 *(§4.1, measured)* | **−82.9** |
| `activate` merge, half its read | 34.49 | **0.000** *(§4.2, measured)* | **−0.0** |
| issue: 118 i/n (probe-derived) … 226 i/n (studies 22 + 12's kernel counts) | | ÷ IPC 1.453 | **−80.9 … −155.6** |
| **total** | | | **−163.8 … −238.4** |

**Cost.** Fusion makes the threat channel inherit HalfKA's refresh trigger.
Measured here: HalfKA refreshes **0.1647/node**, the threat channel rebuilds
**0.0196/node**, so fusion forces **+0.1451 from-scratch threat rebuilds per
node — 8.4× more** — each folding **4.156 rows** out of the 77.7 MB threat
table, the engine's most expensive load class.

| | value | cycles/node |
|---|---:|---:|
| 0.1451 × 4.156 rows × 20 lines = 12.06 cold weight-row lines, at 6.5–13 cycles/line | | **+78.4 … +156.8** |
| `dst.fill(0)`, 0.1451 × 40 = 5.80 lines | | **+7.0** |
| rebuild issue: 1,300 i/rebuild (study 31, derived) … 2,808 (studies 18/19, measured symbol) | 189–407 i/n | **+129.8 … +280.4** |
| **total** | | **+215.3 … +444.3** |

| | cycles/node | throughput | Elo @ 120/doubling |
|---|---:|---:|---:|
| **best case for fusion** | **−23.1** | 1.0057× | **+1.0** |
| **worst case** | **+280.5** | 0.9357× | **−11.5** |

### 6.1 Where this differs from study 31's −136 (+4.7 Elo)

Three corrections, in order of size:

1. **`activate`'s merge is worth zero in stall cycles**, not a third of the
   prize (§4.2). Study 31 counted 34.5 lines at 2.36 = −81 cycles that are not
   there.
2. **The saving per line is 1.213, not 2.36** — matched on both miss count and miss depth
   (§4.1, §4.3).
3. **The rebuild's instruction cost was never converted to cycles.** Study 31
   ran an instruction ledger (its own range: −42 … **+199** i/n) beside a cycles
   ledger holding only the memory side, called the first "instruction-neutral",
   and quoted the second as the answer. +199 i/n is **+137 cycles** at this
   engine's IPC, and 0.1451 rebuilds at the *measured* symbol size is +280.

Study 31's own measured inputs were fine — I reproduced its rebuild rate and row
count to 0.5% — and its arithmetic was the problem.

### 6.2 The advantage being traded away, quantified

Our threat channel rebuilds from scratch **0.0196 times per node**. Stockfish,
because it fused, rebuilds its whole active threat set on every own-king move —
`eca43a97:src/nnue/nnue_accumulator.cpp:577` calls
`ThreatFeatureSet::append_active_indices` on the refresh path and
`find_last_usable_accumulator` says so in a comment: *"Threat feature set
refreshes require a king move across the center, i.e., a subset of halfka
refreshes"*. Fusion would take us from 0.0196 to 0.1647 rebuilds/node. **That
is not an incidental cost of fusion; it is the same 8.4× that dominates the cost
column above**, and it is a structural advantage over Stockfish that we
currently hold and would be giving up.

---
---

## 7. TWO MORE ARMS: does the number TRANSFER, and what was study 31 actually copying?

Panel 2, `tools/perf39/panel2.sh`, n = 24, same binary as its own control
(`d0null`, byte-identical, moves `stall/n` +0.884%, 8/24 negative).

### 7.1 Placement: the number transfers

The recommendation in §6 applies a price measured on traffic issued in one place
to traffic fusion removes from another. Fusion removes accumulator traffic from
the **fold**; the probe adds it in **`activate`**. `P39_SITE=fold` moves the
copies. The fold fires 0.8538/node against `activate`'s 0.4311, so fold dose 1
and activate dose 2 are the **same volume in different places**, at the same
reuse distance in nodes:

| arm | site | lines/node | Δ ins/n | Δ miss/n | Δ stall/n | stall/line | Δ cyc/n |
|---|---|---:|---:|---:|---:|---:|---:|
| `f1` | fold | 136.60 | +156.65 | +117.55 | +313.24 | **2.293** | +490.67 |
| `d2` | `activate` | 137.95 | +157.37 | +121.57 | +318.73 | **2.310** | +548.27 |
| `f2` | fold | 273.20 | +308.63 | +224.49 | +640.69 | **2.345** | +1068.80 |
| `d4` | `activate` | 275.90 | +312.58 | +226.05 | +593.59 | **2.152** | +1014.60 |

**Within 1.7% and 7.9% on stall cycles, within 0.4% and 0.7% on misses.** Stall
per line is a property of the traffic, not of where in the node it is issued.
The price in §6 transfers.

### 7.2 `P39_SRC=live` reproduces study 31's miss count

My ring copy misses 1.272 times per line; study 31's copy missed 0.477. The
obvious reconstruction is that its copy's READ half touched memory the engine
was about to touch anyway — prefetching `activate` rather than adding traffic.
`P39_SRC=live` does exactly that, reading the accumulator pair `activate` is
about to consume:

| arm | source | Δ miss/n | miss/line | Δ stall/n | stall/line |
|---|---|---:|---:|---:|---:|
| `d1` | cold ring | +86.56 | 1.255 | +207.06 | 3.002 |
| `d1L` | **live accumulator** | **+42.95** | **0.623** | +135.91 | **1.970** |
| *study 31* | *(unknown)* | *+32.88* | *0.477* | *+162.7* | *2.36* |

**The misses halve, and both columns land on study 31's.** Its probe was a
live-source copy — 0.623 against its 0.477 on the count, 1.970 against its 2.36
on the price. That closes the last open question about its instrument: it was
not measuring something exotic, it was measuring a copy whose read half was
free.

*(A caveat on panel 2's dose-matched series, which is why panel 1's fixed-ring
series stays the primary one: scaling `RING` with `DOSE` inflates
`L1-dcache-loads` faster than instructions — loads go 1 : 2.47 : 4.96 across
doses 1/2/4 where instructions go 1 : 1.97 : 3.92 — almost certainly page-walk
traffic as the ring outgrows the L1 dTLB. It does not touch the placement or
source comparisons above, which are ring-matched pairs.)*

### 7.3 Read half versus write half — and a retraction of study 29's "the stores are the expensive part"

Panel 3, `tools/perf39/panel3.sh`, n = 24. `accpin` pins the fold `src` **reads**
and leaves every store in the real per-ply stack; `dstpin` pins the stores too.
All three pin arms compute the same all-zero evaluation and retire the same
instructions (+1.83 and +1.05 i/n against `actpin0`), so both differences are
clean:

| half | lines/node | Δ miss/n | miss/line | Δ stall/n | stall/line | Δ cyc/n | **cyc/line** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **reads** (`accpin − actpin0`) | 74.17 | −63.28 | 0.853 | −77.30 | 1.042 | −108.15 | **1.458** |
| **writes** (`dstpin − accpin`) | 80.76 | −67.50 | 0.836 | −47.06 | 0.583 | −31.13 | **0.385** |
| both (`dstpin − actpin0`) | 154.93 | −130.78 | 0.844 | −124.36 | 0.803 | −139.28 | 0.899 |

**The two halves miss at the same rate and cost 3.8× differently.** Study 29 §3
wrote its correction as *"the accumulator STORES are ~64% of accumulator-stream
misses, and they cost by capacity"*. The first half survives — writes are 52% of
the misses here — and **the second half does not**: the writes are the *cheap*
half, worth 22% of the cycles.

The mechanism is not subtle once seen. `ex_no_retire.load_not_complete` counts
cycles in which **nothing retired because LOAD data had not arrived**. A store
that misses does not block retirement — the store buffer absorbs it and the line
is fetched behind the retiring instruction. So a store miss is nearly invisible
to that counter *and* cheap in cycles, exactly as measured.

**This also disposes of §4.4's residual worry.** I had flagged `dstpin`'s
single-buffer store funnelling as a possible depressant on the subtractive
number. It is not needed as an explanation: the write half is cheap for a reason
that follows from what the counter counts, and the read half — where no
funnelling exists, because `accpin` leaves the stores alone — carries 79% of the
cycles saving on its own. Funnelling is not excluded, but nothing requires it.

**And it leaves the §6 ledger where it was**, because fusion removes a fold
pass's reads and its writes in equal measure. Splitting panel 1's −187.98
cycles/node by panel 3's 3.79:1 read:write ratio gives 1.968 cycles per read
line and 0.519 per write line; fusion's 34.15 + 34.15 lines then price at
**−84.9 cycles/node** against the −82.9 the uniform 1.213 gave. A 2.4%
difference, inside everything else in that table.

*(Panel 3 carries no byte-identical null — an oversight. Its combined
0.803 cycles/line sits below panel 1's 1.119 on a box whose load average had
risen from 3.5 to 6.9 by then, which is why it is used for the read:write RATIO
and panel 1 for the LEVEL.)*

---

## 8. RANKED REMAINDER

| # | item | prize | evidence | tractability |
|---|---|---|---|---|
| 1 | **The weight rows, still.** The whole accumulator stream is 235 lines/node at ~1.2 cycles/line ≈ **280 cycles/node of 4081, i.e. 6.9%**. Everything else memory-shaped in the NNUE is weight rows. Study 29 ranked "fewer threat rows" #1 and study 31 ranked "the HalfKA weight rows, never the subject of an arm" #1; nothing in this study moves either down | the residual, ~93% of the NNUE's memory cost | §2, §4 | medium |
| 2 | **Measure a cold threat weight-row line directly.** Fusion's cost column swings +78 … +157 on a number (6.5–13 cycles/line) that is a *residual* in study 31's model and has never been measured. A `wpin`-style probe aimed at the threat refresh path alone would pin it, and the same number prices item 1 | decides half the width of every FT memory estimate |  §6, §9.2 | **high — the probes exist** |
| 3 | **Nothing points at `activate` any more.** Its four-array merge is 12.25% of loads, 2.18% of load-stall cycles (study 29) and **0.0032 misses per line** (§4.2). Three studies have now aimed at it from different angles. Leave it | — | §4.2 | **closed** |
| 4 | **Accumulator-channel fusion** | **−23 (best) … +281 (worst) cycles/node ⇒ +1.0 … −11.5 Elo**, and it trades away a measured 8.4× refresh advantage | §6 | **DO NOT BUILD** |
| 5 | **The capacity-cliff reconciliation** | falsified two ways (§3.1); do not re-open it to explain a future probe disagreement — check miss depth first | §3.1, §4.4 | **closed** |

### 8.1 What this study did NOT do

* **No strength change landed, and none was available.** The commissioned
  deliverable was a decision on a 34.9%-of-cycles architectural question, and
  the decision is "no". The one thing that would have been a strength change —
  building fusion — is what the measurement says not to build.
* **No games were run.** Nothing here reached a state worth arbitrating, and by
  the standing rule that means nothing here is arbitrated.
* **`P39_SITE=fold` was added late** and the arms that use it are in panel 2
  only, so the fold-site placement result rests on one panel rather than two.
* **A `cargo fmt` at the end reflowed two `eprintln!` calls** in the probe
  module, shifting line numbers after every panel had been measured. It cannot
  change codegen, but it can change panic `Location` strings and therefore the
  binaries' hashes (study 35's finding), so the measured binaries are not
  reproducible from the landing tree byte-for-byte. Their md5s are recorded;
  the claim is that their *behaviour* is reproducible, not their bytes.
* **The three pin binaries were built from three different source states**
  (`accpin`/`dstpin` from `b8d6e13`'s predecessor, `actpin` from `4320db4`,
  `addcopy` twice). Every code path they exercise is `#[cfg]`-identical across
  those states, and the headline arm's own instruction counts agree to
  +1.05 i/n out of 5474 — but the binaries are not bit-comparable and their
  md5s are recorded in `docs/research/perf-artifacts-39/binaries.md5` rather
  than claimed to match.

## 9. WHAT I AM LEAST SURE ABOUT

In order, most doubtful first.

1. **The last 1.35× between the real stream's 1.298 stall cycles per miss and
   the shallowest probe's 1.759.** §4.4 explains most of the additive/
   subtractive gap as miss depth, with a measured monotone curve; §7.3 removes
   store funnelling as the explanation for the rest. What is left is the
   residual, and it is unexplained. If the true price were 1.759 rather than
   1.213, fusion's saving grows ~45% and the best case moves from +1.0 Elo to
   roughly +2.5 — still not a lever, but the verdict would then rest on the
   cost side alone.
2. **Panel 3 has no byte-identical null.** Panels 1 and 2 both carry one and
   both are quoted against it; panel 3 was queued in a hurry and is not, so its
   absolute level (0.803 cycles/line against panel 1's 1.119, on a box whose
   load had doubled) is not trustworthy on its own. It is used only for the
   read:write RATIO, and §7.3 says so — but a ratio taken without a null is
   still a ratio taken without a null.
3. **The refresh cost side is a model, not a measurement.** The rebuild ROW
   count is now measured, and the extra rebuild RATE is measured, but the price
   of a cold threat weight-row line is still study 31's residual — a number
   that absorbs every modelling error in its construction, quoted as a 5–13
   band. Nobody has ever measured it directly. A `wpin`-style probe aimed at
   the *threat refresh path only* would close it, and it is the cheapest
   remaining thing on this board.
4. **That any of this converts to Elo at the assumed rate.** Everything here is
   cycles on a non-PGO build. Studies 12–15 measured the non-PGO frame
   overstating the PGO delta by 1.12–2.0× on data-layout changes, and no game
   has been played. Nothing in this study has been arbitrated by games, which
   by the standing rule means it has not been arbitrated.
5. **The dose series measures adding traffic to a working set that already
   contains the real accumulator stack.** Fusion removes traffic *and* shrinks
   the stack, so the post-fusion engine is at a different operating point than
   any arm I ran. §3 shows the response is close enough to linear over a 4×
   span that extrapolating one dose backwards is defensible, but it is still an
   extrapolation past the last measured point.
6. **`ex_no_retire.load_not_complete` is a count of stall cycles, not of time.**
   It has behaved well here — the byte-identical null is small and every
   additive arm clears it several-fold with a perfect sign count — but the
   engine's absolute level of it has moved a long way between studies (2934 in
   study 31, ~1960 here) as the engine got faster, so ratios travel between
   studies and absolute levels do not.

---

## 10. HANDOVER

1. **Runtime knobs beat cargo features for a probe panel.** Every additive arm
   in this study is the same 1.2 MB binary driven by environment variables, so
   code layout — which study 29 had to carry a second null for — is differenced
   away by construction. It also collapsed what would have been eight fat-LTO
   builds into one. If a probe's parameter can be read from the environment
   without a branch in the inner loop, read it from the environment.
2. **`L1-dcache-loads` on Zen 4 counts STORES.** Measured in §1 against traffic
   of known geometry. Study 29's per-site load census is a load+store census.
   And a fully cold 64-byte line costs ~1.2 `L1-dcache-load-misses` events, not
   1.0, so a "miss rate" off that counter is ~1.2× the fraction of lines that
   are actually cold.
3. **A probe that computes wrong evaluations changes the search tree, and no
   normalisation repairs that.** Measured: +7.5% evaluations per node with a
   zero-filled pin, +20.4% with a realistic one. The node TOTAL being fixed by
   a `nodes` budget is necessary but nowhere near sufficient — check the node
   MIX with `foldcount` before believing a per-node delta from an
   eval-breaking arm.
4. **A zero-filled pin turns off the sparse L1 layer.** All-zero accumulator ⇒
   all-zero `ft` ⇒ `find_nnz` finds nothing ⇒ `l1_dot` does nothing. That is
   where `accpin`'s "6.8% of instructions to compiler CSE" actually went, and
   `P39_PINFILL=live` is the one-line control that proves it.
5. **Price a change in the currency you will spend it in.** The commissioned
   question was stall cycles per line, and that is answered; but fusion is
   bought with *cycles*, and the probe's `Δcycles/line` is materially larger
   than its `Δstall/line` because the copy also has to be issued. Study 31
   measured in stall cycles and converted to Elo through a cycles ratio.
6. **Additive probes are the tractable kind for accumulator work.** They leave
   the tree bit-identical (node counts matched exactly here), and their extra
   instructions ARE the thing being priced rather than a confound to normalise
   away. The cost is that their lines are colder than the real stream's, which
   is a bounded, measurable correction — §4 — rather than an unmeasured one.
