# Perf study 62 — the dispatch table was the cheap route, and the kernel study 61 exempted was the bigger half

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`69922ca4`**
(post-study-61), rebased to **`f0d8f405`** and re-measured there · **Net/arch:**
D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 · **Instruments:**
`tools/perf61/` reused verbatim + `tools/perf62/` + three new `foldcount`
censuses · **Artifacts:** `docs/research/perf-artifacts-62/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, non-PGO, pinned to physical core 2. **A GPU training run
shared the box for the whole study** (95 % GPU utilisation, load average 1.8–2.5
throughout), so every verdict rests on `instructions:u` / `branches:u` — null
0.000, load-robust — and `cycles:u` is quoted for its sign only, exactly as
study 61 had to.

---

## ★ 0. WHAT LANDED

**Two shape dispatches. Together the largest instruction win of the 52–62 chain
except study 59's −176.8 i/n, and 1.64× study 61's.**

| counter | median | se | sign | p | its null |
|---|---:|---:|---:|---:|---|
| `instructions:u` | **−2.024 %** (−112.98/node) | **0.000** | **24/24** | **0.0000** | −0.000 %, se 0.000 |
| `branches:u` | **−3.649 %** (−20.26/node) | **0.000** | **24/24** | **0.0000** | 0.000 %, se 0.000 |
| `branch-misses:u` | **−1.004 %** (−0.213/node) | 0.170 | **20/24** | **0.0015** | −0.086 %, 15/24 |
| `ic_tag_hit_miss.instruction_cache_miss` | **+6.199 %** (+9.25/node) | 0.187 | 0/24 | 0.0000 | +0.134 %, 9/24 |
| `cycles:u` | +0.186 % | 0.334 | 11/24 | 0.8388 | **+0.500 %, p = 0.023 — its null FAILS** |

Node-identical (**5,001,562**), eval hash exact (**`2e1ede95…79c4`**),
`.text` 722,773 → 760,085 (**+37,312 B, +5.16 %**).

1. **The threat fold's dispatch table finished to a 6 × 6 grid** — the rest of
   study 61's open item. Coverage of threat folds **68.40 % → 92.45 %**;
   of threat *rows*, the unit the generic kernel's control is charged in,
   **59.66 % → 19.31 %** left generic. Measured alone: **−61.80 i/n**.
2. **`i8_deltas_dual_inner` — the Finny warm refresh — got a shape dispatch,
   having had none at all.** Study 61 ranked its caller at 9.64 % of engine IBS
   latency, called it *"the highest-ranked symbol nobody is currently working
   on"*, and screened it out. Measured alone: **−42.54 i/n**, and it is the best
   instructions-per-byte rung in the whole ladder after the first.
3. A third, unplanned **−8.64 i/n** appeared on promotion: the three measured
   rings, written as three chained `match` statements, cost that much more than
   the same 33 arms in **one** `match`. Recorded because it is free and nobody
   would look for it.

**Also landed: three instruments.** `foldcount::THR_LENS` (the threat fold's
per-list lengths, **uncapped** where `THR_PAIR` saturates at 7),
`foldcount::DUAL_PAIR` (the warm refresh's `(adds, subs)` pair — the shape
statistic that site has never had), and `tools/perf62/{ladder62,tail62}.py`.

**And the cost side, which the licensed counter is structurally blind to:**
`.text` grows 5.16 % and **L1i misses rise 6.199 % against a null of +0.134 %,
0/24, p < 0.0001**. §4 measures what that costs (nothing resolvable, because the
engine is backend-bound), states the landing rule that was pre-registered before
the candidate existed, and lists the **three further rungs that win on
instructions and were dropped anyway**.

**Two claims of study 61's are refuted with numbers** (§1.3, §6.3), and the
wall-clock arbiter **refused to license** (§7.1). Both are reported as such.

---

## ★ 1. THE COMMISSION'S QUESTION, ANSWERED BEFORE ANY KERNEL CODE

Study 61 closed with two routes and a judgement:

> **31.60 % of threat folds are still generic**, and the win per converted fold
> **grows** with the shape (**−115 i/fold** one-sided vs **−322 i/fold**
> three-row) — so either continue the table, or fix the generic kernel's loop
> control, **which now looks the cheaper of the two.**

The commission asked me to verify that judgement rather than inherit it, and to
get the tail's shape from `foldcount::THR_PAIR` first. Both were done, in that
order.

### 1.1 The tail, re-counted on the shipped tree

`bin61/fcount` (`--features foldcount`), 4,270,334 threat folds, node total
5,001,562 — study 61's census reproduced exactly, including its 31.60 %:

| grid | arms | coverage of FOLDS | share of all threat ROWS left generic |
|---|---:|---:|---:|
| `0..=3` — what study 61 shipped | 15 | **68.40 %** | **59.66 %** |
| `0..=4` | 24 | 79.42 % | 44.31 % |
| `0..=5` | 35 | 87.04 % | 30.84 % |
| `0..=6` | 48 | 92.45 % | 19.31 % |

**The two columns disagree sharply, and that is the point.** The generic kernel
charges loop control **per row per tile**, so ROWS is the unit the prize is
denominated in — and after study 61 landed, **59.66 % of all threat rows were
still on the runtime-length kernel** while only 31.60 % of folds were.

### 1.2 Why the table is the cheap route, and Route B's ceiling is the same number

"Fix the generic kernel's loop control" and "continue the table" collect **the
same prize**: the control the generic kernel pays and the shaped kernel does
not. So Route B's ceiling is exactly the sum, over all folds, of the per-fold
win Route A collects one shape at a time — and **Route A measures that sum
incrementally, for free, as a side effect of being built.** The table is
simultaneously the arm and the instrument that prices its own alternative.

Route A is also risk-free in a way Route B is not: it is a `match` arm calling
an existing kernel, bit-identical by construction (same tiles in the same order,
same rows in the same order, same per-lane wrapping i16 arithmetic), with no new
kernel logic and no aliasing question.

Route B carries **three measured failures**, all in this same kernel, all quoted
in its own banner:

| arm | study | result |
|---|---|---:|
| paired row loops (`split_at(min)`) | 37, `TILES = 2` | **+0.147 % instructions** |
| the same, re-priced | 58, `TILES = 8` | **+16.96 i/n** |
| hand-split `as_chunks::<4>` unroll groups | 58 | **+63.35 i/n** |

Study 58's banner already generalised it: *"in this kernel a loop-control
instruction count is not a prize, it is an estimate of what LLVM would charge
you for taking the loop away from it."*

> ### 1.3 Verdict on the inherited judgement: REFUTED
>
> Continuing the table is **not** the expensive route. On this tree the rest of
> the square grid is **−61.80 i/n**, as a pure dispatch change; the second site
> §3 found is a further **−42.54 i/n** by the same mechanism.
>
> This study makes **no claim that Route B cannot work** — it was priced and
> deprioritised, not attempted. What is refuted is the *ordering*: Route B was
> not the cheaper of the two, and the three receipts above were already on the
> record when the judgement was written.

---

## ★★ 2. THE TAIL IS ASYMMETRIC — and `THR_PAIR` could not show it

`THR_PAIR` saturates each side at 7. That is enough to ask "does a 6 × 6 grid
cover this fold" and useless for asking how far the tail runs, which is what
decides whether a square grid can ever close. **`THR_LENS`** (new here) records
the two list lengths uncapped:

| n | adds ≤ n | subs ≤ n |
|---:|---:|---:|
| 3 | 83.44 % | 72.25 % |
| 4 | 91.74 % | 81.57 % |
| 5 | 96.29 % | 88.27 % |
| **6** | **98.64 %** | **92.99 %** |
| 8 | 99.90 % | 97.64 % |
| 10 | 100.00 % | 99.47 % |
| longest observed | **12** | **16** |

**The residual tail is almost entirely LONG SUB LISTS against short add lists.**
`adds ≥ 7` is **1.36 %** of the channel; `subs ≥ 7` is **7.01 %**. A square ring
past `0..=6` would spend nine of its arms on cells holding 1.36 % between them.

Rectangular coverage, so the choice is visible rather than asserted (arm count
in brackets):

| | subs ≤ 4 | subs ≤ 5 | subs ≤ 6 | subs ≤ 7 |
|---|---:|---:|---:|---:|
| **adds ≤ 4** | 79.42 % (24) | 84.76 % (29) | 88.17 % (34) | 91.74 % (39) |
| **adds ≤ 5** | 80.94 % (29) | 87.04 % (35) | 91.12 % (41) | 96.29 % (47) |
| **adds ≤ 6** | 81.44 % (34) | 87.97 % (41) | **92.45 % (48)** | 98.64 % (55) |
| adds ≤ 7 | 81.57 % (39) | 88.27 % (47) | 92.99 % (55) | 100.00 % (63) |

(The `subs ≤ 7` column is a lower bound — those are the saturated cells.)

So the rungs built past `0..=6` here are **rectangular**: the `subs = 7` and
`subs = 8` columns for every add count already covered, 7 arms each.

---

## ★★ 3. THE SECOND SITE — the kernel with NO dispatch, which study 61 exempted

Study 61 §10 item 5 named it in one line: *"`i8_deltas_inner` and
`i8_deltas_dual_inner` have **no shape dispatch at all**."* Its §8 then ranked
`HalfkaContext::finny_refresh_into` — which is `i8_deltas_dual`'s only caller —
at **9.64 % of the engine's IBS tag-to-retire latency**, *"the highest-ranked
symbol nobody is currently working on"*, and screened it out:

> **It is not an opportunity**: study 12 §6 measured this engine at 0.1611
> refreshes and 0.9064 refresh rows per node against `eca43a97`'s 0.1673 and
> 0.9268 — Stockfish parity — so the refresh RATE is already at the reference
> and the rows it folds are the same capacity-bound rows as §3.

**Both halves of that are true and the conclusion does not follow.** The rate is
at parity and the rows are capacity-bound; neither statement is about the
kernel's *loop control*, which is what a shape dispatch removes and which
Stockfish parity says nothing about at all.

> **A site screened on the axes that happen to have instruments is not
> screened.** Study 61 had a refresh-rate instrument (study 12) and a locality
> instrument (its own §2–3), and it had no shape instrument for this kernel —
> `HKA_DUAL_SHAPE` records the SUM of the two list lengths, which is the exact
> defect study 61 itself found in `THR_SHAPE`, one kernel over.

### 3.1 The stale assumption, stated precisely

Study 25 gave `i8_deltas_into_inner` a dispatch because a HalfKA *make-move*
fold is always one of `(1,1) (1,2) (2,1) (2,2)`. The Finny **warm refresh** folds
the set difference between two king buckets, which is a different and much wider
distribution, so those four arms genuinely would not have covered it — and it
was left on the runtime-length body. **That was a correct decision about the
four arms and it was never re-read as a decision about dispatch.** Nobody
counted the distribution.

### 3.2 `DUAL_PAIR` — counted here for the first time

823,842 warm refreshes in the study frame. **It is the most dispatch-friendly
fold in the engine**, concentrated on the diagonal:

| shape | calls | share |
|---|---:|---:|
| `(2,2)` | 157,018 | **19.06 %** |
| `(1,1)` | 104,454 | **12.68 %** |
| `(3,3)` | 100,891 | **12.25 %** |
| `(4,4)` | 44,307 | 5.38 % |
| `(3,2)` | 41,112 | 4.99 % |
| `(2,3)` | 34,940 | 4.24 % |
| `(4,3)` | 31,564 | 3.83 % |
| `(3,4)` | 31,051 | 3.77 % |
| `(0,0)` (a pure 2560 B copy) | 3,145 | 0.38 % |

| grid | arms | coverage of warm refreshes |
|---|---:|---:|
| `0..=2` | 8 | 38.64 % |
| `0..=3` | 15 | 62.82 % |
| **`0..=4` — landed** | 24 | **79.95 %** |
| `0..=5` | 35 | 89.93 % |
| `0..=6` (`dual62b`, not landed) | 48 | 95.01 % |

**Three of the four shapes study 25 chose for HalfKA — `(1,1)`, `(2,2)` and the
`(1,2)`/`(2,1)` pair — are 38.6 % of this site on their own.** The dispatch it
was exempted from would have covered nearly two fifths of it from day one.

---

## ★★ 4. THE COST SIDE `instructions:u` CANNOT SEE — and the landing rule, pre-registered

A dispatch table buys instructions with **`.text`**, and `instructions:u` is
structurally blind to what `.text` costs: an instruction-cache miss adds cycles,
never retired instructions. Study 61 landed +8,048 B and did not measure the
front end. This study's ladder reaches **+70,336 B (+9.7 %)** at its top, which
is well past the point where "the counter says it wins" is an adequate answer.

So every panel carries **`ic_tag_hit_miss.instruction_cache_miss`** (and, in
`panel-ladder.txt`, `op_cache_hit_miss.op_cache_miss`) alongside the exact
counters, with the byte-identical null in the same rotation. They move, hard:

`panel-ladder.txt`, 30 rounds, realised mean position 2.50 for all six labels:

| label | Δ`.text` | `instructions:u` | `branches:u` | **L1i misses** | op-cache misses | `cycles:u` |
|---|---:|---:|---:|---:|---:|---:|
| **null** | 0 | −0.000 % | −0.000 % | +0.104 % (13/30) | −0.067 % (18/30) | +0.379 % (13/30) |
| `s62c2` (threat `0..=6`) | +23,648 | **−1.107 %** | **−2.387 %** | **+2.450 %** | +0.699 % | +0.035 % |
| `d62b` (dual `0..=6`) | +33,008 | **−1.008 %** | **−1.732 %** | **+5.466 %** | +2.050 % | +0.116 % |
| `comb` | +56,656 | **−2.115 %** | **−4.119 %** | **+5.708 %** | +3.405 % | +0.333 % |
| `comb2` (everything) | +70,336 | **−2.282 %** | **−4.690 %** | **+9.397 %** | +3.413 % | +0.452 % |

Every `instructions:u` and `branches:u` figure is 30/30 at se 0.000. Every L1i
and op-cache figure is 0/30 or 3/30 at p = 0.0000, against a null that passes
its own test. **`cycles:u` is not resolvable and its own null moves as much as
any arm** — study 60 §6's defect, reproduced.

### 4.1 The rule, written down before the candidate was built

With the arbiter unavailable, the landing decision cannot be "the biggest
instruction number". The rule fixed here **before building or measuring the
candidate set** is:

> Land a rung only if its **marginal** `instructions:u` win is at least
> **2.0 i/node per KB of `.text`** it adds. Drop any rung below 1.0.

Measured marginal rates over the whole ladder (single-run whole-frame totals;
`base` was re-measured four times across the study and spans 27,911,704,200 to
27,911,705,726, i.e. **5.5 parts in 10⁸**, so a rung's Δ of 10⁷–10⁸ is far
outside run-to-run variation):

| rung | Δ`.text` | Δ i/node | **i/node per KB** | verdict |
|---|---:|---:|---:|---|
| threat `0..=4` (a) | +4,912 B | −19.90 | **4.05** | keep |
| threat `0..=5` (b) | +7,408 B | −19.01 | **2.57** | keep |
| threat `0..=6` (c) | +11,328 B | −22.89 | **2.02** | keep |
| threat `subs = 7` (d) | +6,496 B | −3.91 | **0.60** | **drop** |
| threat `subs = 8` (e) | +7,184 B | −5.37 | **0.75** | **drop** |
| dual `0..=4` | +13,296 B | −42.54 | **3.20** | keep |
| dual `0..=6` | +19,712 B | −13.71 | **0.70** | **drop** |

**The cut is not a borderline call: there is a 2.7× gap between the worst kept
rung (2.02) and the best dropped one (0.75), so every threshold in
(0.75, 2.02) selects the same set.** The level of the threshold is therefore
immaterial, which is the only reason a rule with an invented constant is
admissible here.

The three dropped rungs are **real, measured instruction wins** —
`shape62d` −3.91, `shape62e` −5.37, `dual62b` −13.71 i/n, every one of them
node-identical and eval-hash exact — and they are dropped anyway. They remain
switchable so one panel on a quiet box can settle them.

### 4.2 What the `.text` actually costs — nothing resolvable, and here is why

`panel-frontend.txt`, 24 rounds, adds `de_no_dispatch_per_slot.*`, which counts
**dispatch slots lost** rather than misses. Reference per node: 3514 cycles
(21,084 slots at 6-wide), 9223 slots lost to the **backend**, 4531 to the
**front end**, 5581 retired.

| label | L1i misses | front-end slots lost | net of null |
|---|---:|---:|---:|
| null | +0.357 % | +0.743 % (p = 0.0066 — **null fails**) | — |
| `comb3` | **+5.772 %** (0/24) | +0.442 % (9/24, p = 0.31) | **−0.30 %** |
| `comb2` | **+9.740 %** (0/24) | +1.082 % (2/24) | +0.34 % |

> **A 5.8 % rise in L1i misses produces no resolvable rise in front-end
> starvation.** The machine is **backend-bound** — 9223 slots/node lost to the
> backend against 4531 to the front end — and the decoupled fetch queue absorbs
> the extra misses. The `.text` cost is real as a miss count and is not, on this
> engine, a dispatch cost.

**This panel's null failed its own test on the front-end counters** (the box got
busier mid-panel), so the two front-end columns are quoted as a **bound**, not a
measurement: whatever the cost is, it is smaller than a null that itself moved
+0.743 %. The L1i column's null passes (p = 0.15) and that column is a
measurement.

This result *weakens* the case for the landing rule that was already
pre-registered. It was not used to move the cut, because moving a
pre-registered rule after seeing the data is the failure the rule exists to
prevent. §8 hands the question to the next study instead.

---

## ★ 5. THE ARMS, RUNG BY RUNG

### 5.1 The threat fold's ladder

`tools/perf62/ladder62.py` over `perf-artifacts-62/foldcount-census.txt`:

| rung | folds converted | rows converted | Δ instructions | i/node | **i/fold** | i/row |
|---|---:|---:|---:|---:|---:|---:|
| `0..=4` | 470,940 | 2,952,523 | −99,549,970 | **−19.90** | 211.4 | 33.7 |
| `0..=5` | 325,183 | 2,590,849 | −95,050,817 | **−19.01** | 292.3 | 36.7 |
| `0..=6` | 230,871 | 2,216,297 | −114,519,299 | **−22.90** | 496.0 | 51.7 |
| cumulative | 1,026,994 (24.05 %) | 7,759,669 | −309,120,086 | **−61.80** | | |

### 5.2 The Finny warm refresh

| rung | coverage | Δ instructions | i/node |
|---|---:|---:|---:|
| `0..=4` (landed) | 79.95 % of 823,842 calls | −212,748,876 | **−42.54** |
| `0..=6` (`dual62b`) | 95.01 % | −281,313,228 | −56.25 |

**−42.54 i/n from 24 `match` arms on a kernel nobody had counted.** Per
converted call that is **323 instructions**.

### 5.3 The mechanism probe, and its REFUTATION

Study 61's generalisation — *"the win per converted fold grows with the shape"*
— does not survive the ladder. Per-fold win against mean rows per fold:

| family | mean rows/fold | i/fold |
|---|---:|---:|
| one-sided (study 61) | 1.24 | −115 |
| three-row (study 61) | 4.61 | −322 |
| **max = 4 ring** | **6.27** | **−211** ← more rows, cheaper fold |
| max = 5 ring | 7.97 | −292 |
| max = 6 ring | 9.60 | −496 |
| `subs = 7` column | ~9.5 | **−177** |
| `subs = 8` column | ~10.5 | **−414** |

Not monotone in rows, and not monotone in either list length.

**A mechanism was pre-registered and then killed by its own probe.** The
disassembly of `i8_deltas_into_generic` shows LLVM unrolling the row loops **×4**
with a one-row-at-a-time remainder loop, re-entered once per list per TILE
(8 tiles). If deleting that remainder is what the shaped arms buy, cost should
track `len mod 4`, and `subs = 7` (3 remainder rows) should dominate `subs = 8`
(none) — which is why the two columns were built as separate features rather
than as one rung. Measured: **`subs = 8` is worth −5.37 i/n against `subs = 7`'s
−3.91**, i.e. ~414 against ~177 instructions per converted fold, *the opposite
of the prediction and by 2.3×*.

> **Refuted. The per-fold win is not predictable from the shape — the only way
> to price a ring is to build it.** This is worth more than the mechanism would
> have been: it says the next rung's value cannot be extrapolated from this
> study's numbers either.

### 5.4 The free −8.64 i/n nobody would look for

The three rings were measured as three `#[cfg]`-gated `shaped_dispatch!` calls,
i.e. three chained `match` statements. Promoted, they became **one** `match` with
33 arms — and the promoted binary is **−8.64 i/n better than the sum of the
measured rings** (`comb3` 27,389,836,137 → `land` 27,346,635,302 for +368 B of
`.text`). LLVM builds a better decision tree from one 33-arm match than from
three chained ones. It is reported because the landed binary is therefore **not
bit-identical to the arm that was measured**, it is better, and a study that
quietly quoted the arm's number would be understating by 8 %.

---

## ★ 6. THE LANDING PANEL

`docs/research/perf-artifacts-62/panel-land.txt` — 24 rounds, blocked Latin
square, four labels, realised mean position **1.50 for every label**, node total
**5,001,562 on every run**.

```
REFERENCE per node: instructions 5580.64  branches 555.22
                    branch-misses 21.24  L1i-misses 149.17  cycles 3519.33

label   event                       median %     se    neg/n      p
null    instructions:u                -0.000   0.000    13/24   0.8388
null    branches:u                    +0.000   0.000    10/24   0.5413
null    branch-misses:u               -0.086   0.136    15/24   0.3075
null    ic_..instruction_cache_miss   +0.134   0.115     9/24   0.3075
null    cycles:u                      +0.500   0.329     6/24   0.0227  <- null FAILS

revert  instructions:u                +0.000   0.000     9/24   0.3075
revert  branches:u                    +0.000   0.000    10/24   0.5413
revert  ic_..instruction_cache_miss   +0.213   0.148    10/24   0.5413

land    instructions:u                -2.024   0.000    24/24   0.0000
land    branches:u                    -3.649   0.000    24/24   0.0000
land    branch-misses:u               -1.004   0.170    20/24   0.0015
land    ic_..instruction_cache_miss   +6.199   0.187     0/24   0.0000
land    cycles:u                      +0.186   0.334    11/24   0.8388
```

### 6.1 The revert is an exact control

`--features noshape62` restores study 61's dispatch at **both** sites.
`revert` reads `instructions:u` **+0.000 % at se 0.000** and `branches:u`
**+0.000 %** against `base`, and its `.text` is byte-for-byte the same size
(722,773). Built from the study-62 source, it reproduces the pre-study-62 tree.

> Its `.text` **md5 differs** (`c4ca4c89` vs `242abfdd`) and that is expected,
> not a discrepancy: study 61's trap — a Rust symbol name embeds the crate
> metadata hash, which moves with the Cargo feature set. Within a fixed feature
> set the md5 was used and held: after `cargo fmt` and after the rebase, `land`
> rebuilt to `75c9955e` both times, and the pre-edit control `base2` reproduced
> `base`'s `242abfdd` exactly across a macro refactor plus a `cfg_attr` change.

### 6.2 `target-cpu`, both quoted

The commission's rule (an arm's value is `target-cpu` dependent and the
direction varies) checks out but is mild here:

| build | base i/node | landed i/node | Δ | Δ % |
|---|---:|---:|---:|---:|
| `native` | 5580.64 | 5467.66 | **−112.98** | **−2.024 %** |
| `x86-64-v3` | 5752.66 | 5651.49 | **−101.16** | **−1.758 %** |

Same direction, 0.90× the magnitude — unlike study 57's arms (−30.8 native
vs −187.6 v3) or study 59's `qrow` (14.1× larger on native). Both builds are
node-identical at 5,001,562 with the exact eval hash.

### 6.3 Study 61's numbers, checked

| study 61 claim | check |
|---|---|
| 31.60 % of threat folds generic; the `THR_PAIR` histogram | **Reproduced exactly**, every cell |
| the fold shape dispatch is bit-identical by construction | **Held** — 11 new binaries, all eval-hash exact, all node-identical |
| `.text` md5 is not valid across a feature-set change; is valid within one | **Confirmed in both directions** (§6.1) |
| "the win per converted fold GROWS with the shape" | **REFUTED** (§5.3): the max-4 ring folds more rows per fold than the three-row family and is 35 % cheaper per fold |
| "fix the generic kernel's loop control … the cheaper of the two" | **REFUTED as an ordering** (§1.3) |
| `finny_refresh_into` "is not an opportunity" | **REFUTED**: −42.54 i/n = −0.762 % of frame from 24 `match` arms (§3) |
| `cycles:u` is not load-robust | **Confirmed again**: its null fails at p = 0.023–0.0227 in three of four panels while `instructions:u`'s null is exactly 0.000 with se 0.000 in all four |

---

## ★ 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The wall-clock value. The instrument REFUSED, and the refusal is the
   result.** `tools/wallclock_panel.py --base bin61/base --arm bin61/comb3
   --rounds 40`:

   ```
   NULL (byte-identical): median +0.134%  sd 0.977  sign 16/40 neg  p=0.2682
   ARM                  : median +0.026%  sd 0.876  sign 19/40 neg  p=0.8746
   NOT LICENSED: null median +0.134% vs arm +0.026%.
   ```

   The arm's median is indistinguishable from zero **and** from the null, so
   nothing may be quoted — including "it is zero". The tool's own docstring puts
   the non-PGO-instructions → PGO-wall-clock ratio at **2.63–4.08×** on studies
   52–58, which would put −2.024 % instructions somewhere near −0.5 % to −0.8 %
   of wall clock before the `.text` cost; the panel cannot resolve anything that
   size against a null with sd 0.98. **A quiet box should re-run exactly this
   command.** It is the single most valuable follow-up in this document.
2. **No PGO pair.** Everything here is non-PGO, the frame studies 51, 60 and 61
   used. PGO absorbed 38 % of the instruction win on studies 52–56 and would
   plausibly do so again; the shipped engine is PGO.
3. **No games.** The arm is node-identical, so a fixed-node match is a dead heat
   by construction and only a real-clock match could speak — which needs item 1's
   quiet box.
4. **The front-end cost is bounded, not measured** (§4.2). The panel that
   carries the dispatch-slot counters had its own null fail on exactly those
   counters. The L1i column is a measurement; the slot columns are a bound.
5. **The three dropped rungs may be wrong to drop.** §4.2 says the `.text` cost
   does not convert into dispatch starvation, which is an argument for
   reinstating them; the rule that dropped them was pre-registered and was not
   moved after the fact. One panel on a quiet box decides it.
6. **`i8_deltas_inner`** (the in-place cold Finny build) still has no dispatch.
   It is 128 calls per 5 M nodes and was left alone deliberately — recorded so
   the next reader does not have to re-derive that it is negligible.

---

## ★ 8. WHAT THE NEXT STUDY SHOULD KNOW

1. **A site screened on the axes that have instruments is not screened.** §3 is
   a −42.54 i/n arm on the symbol the previous study explicitly ranked, named,
   and dismissed — on two true facts, neither of which was about the thing that
   was wrong. When a screen concludes "not an opportunity", check what the screen
   *measured*, not what it concluded.
2. **The per-fold win of a dispatch arm is NOT predictable from its shape**
   (§5.3). Two mechanisms have now been proposed for it — study 61's "grows with
   the shape" and this study's `len mod 4` — and both are refuted by
   measurement. Build the ring, then price it.
3. **Where the remaining prize is, exactly.** After this study: **7.55 % of
   threat folds** (≥ 19.31 % of threat rows, a lower bound) and **20.05 % of
   warm refreshes** are still generic. The threat residual is almost all long
   SUB lists (§2). `shape62d`, `shape62e` and `dual62b` are built, measured
   (−3.91, −5.37, −13.71 i/n) and switchable; reinstating them needs a `cycles`
   verdict, not more instruction counting.
4. **The `.text` question is now open and instrumented, not hand-waved.** L1i
   misses rise measurably with every dispatch arm (+6.2 % for the landed set,
   0/24, p < 0.0001) and dispatch-slot starvation does not follow (§4.2). The
   engine is backend-bound: 9223 slots/node lost to the backend against 4531 to
   the front end. **If that holds on a quiet box, the `.text` budget for
   dispatch tables is much larger than this study assumed** and rungs at
   0.6–0.8 i/node/KB become landable.
5. **`de_no_dispatch_per_slot.{no_ops_from_frontend,backend_stalls}` is the
   counter pair this series was missing.** It prices a front-end change in the
   same unit as a backend one, and it says in one line that this engine is
   backend-bound by 2:1. No prior study in the 40–61 range used it.
6. **One `match` beats three chained ones by 8.64 i/n** (§5.4). Any study that
   measures arms behind separate `#[cfg]`s and then promotes them must
   re-measure the promoted binary; the difference here was 8 % of the arm.
7. **Still open, inherited and untouched:** study 43 §9 item 2's subtractive
   Finny-line arm; study 57 §6's FNV-1a-64 payload checksum (~45 % of startup,
   needs a format ruling); study 48 §9 item 3's AVX2 `activate` hoist. Study 61
   §3's capacity closure stands — nothing here touched weight-row locality, and
   the L1i movement is `.text`, not weight rows.

---

## ★ 9. GATES

| gate | result |
|---|---|
| node identity, study frame, every binary (15 built) | **5,001,562** — exact |
| wl50 eval hash, every binary | **`2e1ede95…79c4`** — exact |
| byte-identical null through the harness | `instructions:u` **−0.000 %**, `branches:u` **−0.000 %**, both se 0.000 |
| `noshape62` revert vs `base` | `instructions:u` **+0.000 % se 0.000**, `.text` **722,773 both** |
| realised mean panel position | **1.50 / 1.50 / 1.00 / 2.50** — ideal in all four panels |
| `base` reproducibility across the study (n = 4) | 27,911,704,200 … 27,911,705,726 = **5.5 parts in 10⁸** |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | **exit 0 — `All checks passed.`, `Bench: 587683`, eval hash `2e1ede95…79c4`, d13 nodes `1739588`; all three pins unmoved. Run TWICE: on `69922ca4` and again after the rebase to `f0d8f405`** |
| `cargo test --release -- --ignored perft_full_suite` | **ok** (1 passed), both times |
| `cargo clippy --all-targets` | **clean** |
| `cargo fmt` | no change (rebuild after it reproduced `.text` md5 `75c9955e`) |
| rebase re-measure (study 56's `bcmisc55` rule) | `main` moved `69922ca4 → f0d8f405`; `git diff --stat` over the span touches **no `src/` and no `Cargo.toml`**. Re-measured anyway: **−565,070,306 instructions = −112.98/node**, against **−565,069,756** pre-rebase — the same number to 1 part in 10⁶ |

D24 perft ritual not strictly required (nothing under `movegen`, `board`
mutation, `zobrist` or `attacks` is touched) — run anyway, both times.

---

## ★ 10. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net;
                                              # engines/canonical/net.cenn dangles otherwise

# The census that decided the routes
bash tools/perf61/build61.sh fcount foldcount
ARCHON_FOLDCOUNT=/var/tmp/fc.txt taskset -c 2 ./bin61/fcount \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
python3 tools/perf62/ladder62.py /var/tmp/fc.txt      # the rung pricing
python3 tools/perf62/tail62.py   /var/tmp/fc.txt      # THR_LENS + DUAL_PAIR

# The landed pair, and the ladder rungs that were not landed
bash tools/perf61/build61.sh land                     # default = landed
bash tools/perf61/build61.sh revert  noshape62        # the control
cp bin61/revert bin61/null                            # byte-identical null
bash tools/perf61/build61.sh s62d7   shape62d         # subs = 7 column
bash tools/perf61/build61.sh s62e78  shape62e         # + subs = 8 column
bash tools/perf61/build61.sh d62b    dual62b          # dual grid 0..=6
P61_CPU=2 bash tools/perf61/ident61.sh revert land s62d7 s62e78 d62b

# The landing panel
P61_CPU=2 P61_EV='instructions:u,branches:u,branch-misses:u,ic_tag_hit_miss.instruction_cache_miss,cycles:u' \
  P61_PANEL=docs/research/perf-artifacts-62/panel-land.txt \
  bash tools/perf61/panel61.sh 24 base null revert land
python3 tools/perf61/med61.py docs/research/perf-artifacts-62/panel-land.txt base null revert land

# The front-end panel (§4.2)
P61_CPU=2 P61_EV='instructions:u,de_no_dispatch_per_slot.no_ops_from_frontend,de_no_dispatch_per_slot.backend_stalls,ic_tag_hit_miss.instruction_cache_miss,cycles:u' \
  P61_PANEL=docs/research/perf-artifacts-62/panel-frontend.txt \
  bash tools/perf61/panel61.sh 24 base null comb3 comb2

# THE ONE TO RUN ON A QUIET BOX (§7 item 1)
python3 tools/wallclock_panel.py --base bin61/revert --arm bin61/land --rounds 40

CHECK_NNUE_GATE=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

**Do not run a panel while anything else builds** (study 60 §7 rule 5), and do
not park perf text on `/tmp` — it is tmpfs.
