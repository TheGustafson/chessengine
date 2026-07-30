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

### 1. Which number is right: **neither**. The right value is 1.12 stall / 1.21 total cycles per line, and it is a THIRD measurement.

Both studies measured competently and I reproduced both on today's engine, in
one panel, to within a few percent. They measured **different things**, and the
thing that prices fusion is neither of them:

| | study 29 | study 31 | **study 39** |
|---|---:|---:|---:|
| direction | subtractive | additive | **subtractive** |
| tree held identical? | **no** | yes | **yes** |
| instructions held identical? | **no** (−7.6%) | n/a (they *are* the traffic) | **yes (+0.019%)** |
| line coldness vs the real stack | n/a | **~1.5× colder** | **the real stack itself** |
| stall cycles per line | 0.45 | 2.36 | **1.119** |
| reproduced here | **0.350** (its own statistic, −4.503% vs its −4.63%) | **3.184** at dose 1, 2.22 fitted | — |

The measurement that carries no confound is **`dstpin` differenced against
`actpin0`**, and it exists only as a difference of two individually broken arms:
both pin `activate` to the same all-zero buffer, so both compute the same wrong
evaluation, search the same tree, and retire the same instructions (**+1.05 i/n
out of 5474**). Their difference relocates **154.92 lines/node** of *real* fold
accumulator traffic from cold stack memory to a hot 2.5 KB buffer and changes
nothing else: **−133.46 misses, −173.29 stall cycles, −187.98 cycles per node,
24/24 sign-consistent on misses.**

### 2. Why the two differed — factorised exactly, no residual

Between my own two clean arms the gap is **2.85×**, and it multiplies out:

```
stall per LINE, additive / subtractive          2.85x
  of which COLDNESS  (1.272 vs 0.861 miss/line) 1.48x
  of which DIRECTION (2.50 vs 1.30 stall/miss)  1.93x
  product                                       2.85x
```

* **Coldness (1.48×).** An additive probe's ring lines are ~100% cold; the real
  accumulator stack's are ~71%. Study 31 priced traffic colder than the traffic
  fusion removes.
* **Direction (1.93×).** An *added* miss costs about twice what a *removed* miss
  saves. This is the asymmetry study 29 guessed at — but it is **not a capacity
  cliff** (see 3), and it means the direction-matched measurement is the one
  that prices a change which *removes* traffic.
* The remaining factor from 2.85× to the published 5.2× is **study 29's
  per-instruction normalisation**, which I replicated: raw **0.913** →
  normalised **0.350** on the same data, a 2.6× move from a modelling choice.

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

**At the most favourable end of every input, fusion is worth one Elo.** It is
not a lever; it is a large architectural change whose *entire* expected value is
inside the noise of a 400-game match, and it additionally trades away a measured
**8.4× refresh advantage** over Stockfish (our threat channel rebuilds
0.0196/node; fused it inherits HalfKA's 0.1647). Study 29's verdict stands. Its
reasoning does not.

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
stall per LINE, additive / subtractive          2.85x
  of which COLDNESS  (1.272 vs 0.861 miss/line) 1.48x
  of which DIRECTION (2.50 vs 1.30 stall/miss)  1.93x
  product                                       2.85x
```

and the remaining factor to the published 5.2× is study 29's normalisation
(§5.3), which moves its own data from 0.913 to 0.350.

**On the direction factor I am reporting an asymmetry I cannot fully explain.**
An added miss costs 2.50 stall cycles, a removed one saves 1.30. Two mechanisms
are plausible and I did not separate them: added lines are pure pollution with
no reuse and consume miss-address buffers the weight rows are also queueing
for; and `dstpin` funnels every fold's stores into **one** 2560 B buffer, so its
saving may be partly eaten by store-port and store-buffer contention that the
real engine would not have. The second would make the subtractive number a
**lower** bound, which is the direction that matters for a "do not build"
verdict — it is the conservative side for the recommendation, and the
optimistic side for fusion. The fusion ledger in §6 is run at both ends anyway.

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
2. **The saving per line is 1.213, not 2.36** — direction- and coldness-matched
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
