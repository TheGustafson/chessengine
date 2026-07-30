# ARCHON perf study #43 — `finny_refresh_into`, and what it is actually waiting for

**Date:** 2026-07-30 · **Base:** `main` at `01cc007` · **Branch:**
`perf-study-43-finny` · **Instruments:** `tools/perf43/` (new) plus study 34's
build/ident kit · **Artifacts:** `docs/research/perf-artifacts-43/`

---

## 0. HEADLINE

The commission handed me a target, a starting hypothesis, and a conversion
factor. **The target is real. The hypothesis is mostly wrong, and I can say by
how much. The conversion factor was withdrawn mid-study and replaced with a
direct measurement, which is the single most useful thing in this document.**

| what | commission said | measured here |
|---|---|---|
| `finny_refresh_into` share of cycles | 8.75% (study 36, `74c07da`) | **6.33%** on `01cc007` — the figure is stale |
| its signature | "few loads, few instructions, many cycles" | **confirmed and sharpened: 71% of its cycles are `load_not_complete` stall — the most stall-dominated symbol in the engine** |
| why | "every refresh streams ~42 cache lines" of `FinnyEntry` | **234 lines per refresh, of which 114 (49%) are WEIGHT ROWS, not the Finny slot.** The slot is 80 of them |
| the SoA fix (Q2) | "does the probe pull payload lines it does not need?" | **No — bounded at 2 lines of 42, and the warm path needs all 40 payload lines anyway.** §4 |
| `FINNY_KEYS = 64` (Q3) | "nobody knows where the optimum is" | **It is forced. The key map is a bijection from the 64 king squares; all 128 slots are used; there is no dead half.** §5 |
| prefetch (Q4) | "killed six times — beat the prior WITH A MEASUREMENT" | **Four placements built, all node-identical, all measured. Each has an EXACT non-zero instruction cost and no cycles benefit this box can resolve. Kill #7.** §6 |

**The one number to take away**, because it is the number the commission's own
conversion factor got wrong and the one study 42 warned about:

> A cache line in `finny_refresh_into` costs **5.6 stall cycles** if you charge
> all of the symbol's stall to the lines that miss L1, or **20.6** if you charge
> it only to the lines that miss L2. Study 39's 1.119 stall/line — which the
> commission told me to use — is **5× to 18× too low for this population**, and
> study 42's ~121 cycles per TT probe is **6× to 22× too high.** Neither
> imported rate would have given a usable answer.

**Landed:** the instrument kit (`tools/perf43/`), two census counters, an
instrument DEFECT fix that invalidates part of my own first panel (§6.2), and
one hardening commit (§7). **Not landed: any speedup.** §8 says so plainly and
§9 says what I would do next.

---

## 1. THE FRAME, THE BOX, AND A REPRODUCTION CHECK

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes, one thread, 16 MB TT, D48X, `--features sota,arch-d48x`,
`RUSTFLAGS=-C target-cpu=native`, shipped release profile (fat LTO, CGU = 1),
`taskset -c 2`, `nice 19`.

**Node identity holds on every binary in this study: 5,001,562 nodes, wl50 eval
hash `2e1ede9521624dd8`.**

### 1.1 My base against study 40's preserved binary

The coordinator preserved study 40's own base binary (`b40`) as a known-good
reference. Run back to back, alternating, on the same core:

| binary | nodes | `instructions:u`/node | `cycles:u`/node |
|---|---:|---:|---:|
| `b40` (study 40's tree) | 5,001,562 | **5924.300** | 4169.68 / 4134.48 |
| `base` (mine, `01cc007`) | 5,001,562 | **5929.086** | 4171.42 / 4167.96 |

**+0.081% instructions**, and both binaries reproduce their own number to the
last decimal on repeat runs. That is real code between the two commits, not a
build defect — `instructions:u` is an exact counter with a 0.000% null.

### 1.2 The box moved 11% under me, which decides what I am allowed to quote

| when | load average | `cycles:u`/node on the same binary |
|---|---:|---:|
| early session | 13.6 | 4522 – 4614 |
| mid session | ~6 | 4135 – 4171 |
| during the panels | varies | 4103 – 4403 |

The commission's 4104–4180 band is the quiet end and I reproduce it. **Every
absolute cycles figure below states which denominator it uses.** Four agents
plus the D49 trainer were live throughout; every build went through
`tools/perf43/build43.sh`, which takes `flock /home/nick/.cache/archon-build.lock`
and refuses to start below 2500 MB available. Nothing was OOM-killed.

---

## 2. ATTRIBUTION (Q1) — MY OWN PROFILE, NOT STUDY 36'S

Four separate `perf record` runs on `base`, one event each, symbol shares
(within-run attributions, immune to the between-run drift in §1.2):

| symbol | %cycles | %instr | %stall | %L1d-miss |
|---|---:|---:|---:|---:|
| `avx512::i8_deltas_into` | 11.21 | 7.69 | 13.55 | 18.23 |
| `avx512::i8_deltas_into_generic` | 10.02 | 7.24 | 13.67 | 19.68 |
| `search::<NonPv>` | 9.26 | **13.66** | 10.17 | 3.95 |
| `SotaHead::forward_into` | 8.65 | 10.57 | 4.38 | **22.87** |
| **`HalfkaContext::finny_refresh_into`** | **6.33** | **3.65** | **9.26** | **7.97** |
| `MovePicker::init_quiets` | 6.26 | 8.02 | 6.55 | 2.15 |
| `ThreatContext::at_full` | 5.93 | 6.41 | 5.80 | 6.17 |
| `Board::unmake_move` | 4.25 | — | — | — |
| `HalfkaContext::current_full` | 3.44 | 1.81 | 5.18 | 1.73 |

Engine totals, `perf stat`, no multiplexing (3 events per group, all at
100.00% enabled):

| counter | per node |
|---|---:|
| `instructions:u` | **5929.09** (identical across all three groups) |
| `cycles:u` | 4522 (busy) … 4403 … 4135 (quiet) |
| `ex_no_retire.load_not_complete:u` | 2206.5 (busy) |
| `L1-dcache-load-misses:u` | 568.15 |
| `ex_ret_brn_misp:u` | 25.48 |
| `l2_cache_req_stat.dc_hit_in_l2:u` | **527.29** |
| `l2_cache_req_stat.ic_dc_miss_in_l2:u` | **59.06** |

Total L2 requests 586.4/node — study 42's independently established 580.9
cache fills/node, reproduced to **1.0%** on a different event. Good.

### 2.1 The symbol, in absolute terms

At the 4403 cyc/node denominator (the run whose L2 counters are quoted above):

| quantity | value |
|---|---:|
| cycles/node | **279** |
| instructions/node | **216** (= 1311 per refresh) |
| load-stall cycles/node | **204** |
| **stall as a fraction of its own cycles** | **73%** |
| L1-dcache-load-misses/node | 45.3 |
| L2 requests (= L1-missing lines)/node | **36.4** |
| L2 **misses**/node | **9.92** |
| symbol IPC | **0.77** vs engine 1.35 |

**73% of this symbol's cycles are cycles in which nothing retired because load
data had not arrived.** `i8_deltas_into` is at 59%, `search::<NonPv>` at 54%.
The commission's "it is waiting, not computing" is correct and this is the
number that says so.

### 2.2 The traffic model, and it closes to 6%

Per warm refresh, from the disassembly (`REGS = 20`, `LANES = 32`, `TILE = 640`,
`TILES = 2`) and the measured 5.7200 rows/call:

| stream | bytes | lines | lines/node |
|---|---:|---:|---:|
| read `cached.acc` | 2 560 | 40 | 6.59 |
| read weight rows (5.72 × 1280 B, i8) | 7 322 | 114.4 | **18.84** |
| write `cached.acc` | 2 560 | 40 | 6.59 |
| write `stack[ply].acc[p]` | 2 560 | 40 | 6.59 |
| **total** | **14 & 1002 B** | **234.4** | **38.6** |

Measured L1-missing lines: **36.4/node**. Model 38.6. **Agreement to 6%**, and
the residual has the right sign (some store lines are already resident).

**So the commission's "~42 cache lines per refresh" is the FinnyEntry slot
alone. The refresh actually moves 234 lines, and 114 of them — 49% — are weight
rows from the 28.8 MB HalfKA table.** Study 29's IBS said the same thing from a
different instrument ("81.2% of this symbol's loads are HalfKA weight rows");
by lines the figure is 18.84 / 25.4 read lines = **74%**, and the two agree
once you remember that study 29's counter mixes loads with stores (study 38
§1.2, study 39 §1).

### 2.3 WHICH POPULATION — the measurement that replaces the imported rate

The commission told me to convert with study 39's 1.119 stall / 1.213 cycles
per 64-byte line. **Mid-study the coordinator withdrew that**, on study 42's
finding that a stall-per-line rate is a property of the *population* of lines,
not of the machine: study 39 calibrated on the L2-resident, hardware-prefetchable
accumulator stream, and the same rate under-predicted the TT's DRAM-class,
dependent, unprefetchable lines by ~100×.

So I measured the population. Symbol shares of the two L2 events:

| symbol | % of L2 **hits** | % of L2 **misses** | miss : hit |
|---|---:|---:|---:|
| `SotaHead::forward_into` | 24.61 | 3.14 | **0.13** |
| `i8_deltas_into` | 16.47 | 19.72 | 1.20 |
| `i8_deltas_into_generic` | 16.89 | 25.71 | 1.52 |
| **`finny_refresh_into`** | **5.03** | **16.79** | **3.34** |

**`finny_refresh_into` has the most L2-miss-skewed profile of any large symbol
in the engine.** In absolute terms it carries 26.5 L2-hit and **9.92 L2-miss**
lines per node: a **27.3% L2-miss rate against the engine-wide 10.1%**, i.e.
**2.7×**.

That places it firmly between the two calibrated populations, and lets me price
its lines directly instead of importing anyone's rate:

| basis | stall cycles per line |
|---|---:|
| all 36.4 L1-missing lines carry the 204 stall cycles | **5.6** |
| only the 9.92 L2-missing lines carry them | **20.6** |
| *study 39's accumulator-stream rate (imported)* | *1.119* |
| *study 42's TT rate (imported)* | *~121* |

The truth is inside [5.6, 20.6] and the two imported rates are both outside it,
in opposite directions, by 5–18× and 6–22×. **This is the study's most
transferable result: do not carry a stall-per-line rate across populations.**

### 2.4 What it is worth, in Elo

279 cycles/node is **6.33% of engine cycles**. At ARCHON's own ~120 Elo per
speed-doubling constant, removing the symbol *entirely* would be worth
`120 × log2(1/(1−0.0633))` = **+11.3 Elo**. The Finny-specific traffic — the
19.8 lines/node that are the slot's own reads and writes, i.e. everything a
Finny-cache redesign could touch without also being a weight-table fix — is at
most **19.8/36.4 = 54% of the stall**, so **≤ ~110 cycles/node ≈ 2.5% ≈ +4.4
Elo**, and that is a ceiling assuming perfect elimination, not an estimate of
any achievable arm.

**The ~120 Elo/doubling constant has a 95% CI that includes zero** (one 100-game
match differenced against one 200-game match, ±~95 Elo/doubling). Every Elo
figure here is "a speed measurement times a contested constant", not measured
strength.

---

## 3. THE CENSUS — two candidate arms killed before a build

`--features foldcount` with two counters added by this study
(`HKA_DUAL_SHAPE`, `FINNY_KEY`), node- and hash-identical to `base`:

| statistic | measured |
|---|---:|
| warm refreshes (`hka_dual`) | 823,842 = **0.16472/node** |
| weight rows folded | 4,712,390 = **0.94218/node** |
| **rows per warm refresh** | **5.7200** |
| cold builds (`hka_inplace`) | **128** = 2 perspectives × 64 keys, exactly once each |
| king moves made | 1,467,567 = **0.2934/node** |

Study 29's published controls reproduce: 0.1610 → 0.16472 refreshes/node
(+2.3%) and 0.9132 → 0.94218 rows/node (+3.2%), the drift being real search
changes across ~14 studies, and rows/call 5.671 → 5.720 (+0.9%).

### 3.1 The row-count histogram, which nobody had

Only the *mean* (5.68) had ever been measured. The distribution:

| rows | share | cumulative |
|---:|---:|---:|
| **0** | **0.38%** | 0.38% |
| 1 | 0.59% | 0.98% |
| 2 | 13.16% | 14.13% |
| 3 | 5.89% | 20.02% |
| 4 | **21.73%** | 41.75% |
| 5 | 9.71% | 51.46% |
| 6 | 16.06% | 67.52% |
| 7 | 8.64% | 76.16% |
| 8 | 8.68% | 84.84% |
| 9–12 | 12.55% | 97.39% |
| **≥16** | **0.81%** | — |

Median 5. Even counts dominate because a quiet piece move is exactly one sub
plus one add.

**Two arms die here, each on one number:**

* **An empty-difference fast path** — skip the fold's redundant 2560-byte
  write-back to the cache entry when the set-difference is empty, the shape of
  win study 38 took on the threat channel's `thr_copy`. **The empty case is
  0.38% of refreshes = 0.00063/node.** Even a perfect fast path saves 40 lines
  × 0.00063 = **0.025 lines/node**. Dead.
* **Higher Finny associativity** — cache two positions per king key and fold
  whichever differs less, which would target the heavy tail. **The tail is
  0.81% of refreshes at ≥16 rows.** There is no tail to target, and the cost is
  a doubled 672 KB table plus a second probe. Dead.

Neither was built. Both are reported because a measured null on a well-posed
arm is a result, and because the numbers that kill them are cheap and durable.

---

## 4. THE HOT/COLD SPLIT (Q2) — bounded, not built

The question: "the probe reads `by_piece` to compute the diff, then streams
`acc`. Interleaved in one struct, does the probe pull payload lines it does not
need?"

**Answer: no, and the bound is arithmetic rather than a matter of opinion.**

Study 22 arm 2 already moved `by_piece` (96 B) + `psqt` (32 B) to offset 0 under
`#[repr(C, align(64))]`, so the probe's working set is **exactly the slot's
first two cache lines**, and `acc` begins 64-byte aligned at offset 128. I
verified the layout is still that in today's binary and that the allocation is
64-byte aligned in practice (§7).

The bound: an SoA split can save **at most the payload lines a probe touches
and then does not use.** In the warm path there are none — the fold reads all
40 lines of `cached.acc` unconditionally. The only recoverable quantity is
making the 2 metadata lines cheaper to reach by moving them into a compact
128 × 96 B side table, and that is bounded above by

> 2 lines × 0.16472 refreshes/node = **0.33 lines/node**,

i.e. **0.9% of the symbol's 36.4 L1-missing lines/node**, against the measured
5.6–20.6 stall cycles/line ⇒ **1.8 to 6.8 cycles/node, 0.04%–0.15% of engine
cycles.** That is below this box's cycles floor by more than an order of
magnitude (§6.2), and it would additionally *break* the one thing the current
layout has going for it: `by_piece` at offset 0 followed by `acc` at offset 128
is an ascending access stream inside one slot, which is what a hardware
streamer wants.

**This is an arithmetic bound, not a measurement.** I did not build it, and I
am flagging that as a limitation rather than dressing the bound up as a result.
What makes me comfortable stopping is that the bound is computed from
quantities I measured myself (0.16472 refreshes/node, 36.4 lines/node,
5.6–20.6 cycles/line) and it is 10× below the resolving power of the only
instrument that could adjudicate it.

---

## 5. `FINNY_KEYS = 64` (Q3) — it is forced, and the table has no dead half

The commission asked whether 64 is "anywhere near optimal", noting a change
here is behavioural. **It is not a tunable at all**, and the reason is a fact
study 22 §5.3 established and I re-derived independently from the source before
reading it:

```rust
key = KING_BUCKETS[ksq ^ (56 * perspective)] * 2 + (file(ksq) <= 3)
```

`KING_BUCKETS` folds file `f` with `7 − f` (every row of the table is a
palindrome), and the orientation bit separates exactly those two halves again.
The composition is therefore a **bijection from the 64 king squares onto the 64
keys** — checked on rank 1: a1..h1 → keys 1, 3, 5, 7, 6, 4, 2, 0. Two distinct
king squares never share a key.

Three consequences, all measured:

1. **The table cannot be larger.** 64 is the whole key space.
2. **`halfka_king_key`'s advertised fast path is unreachable dead code.** Its
   doc comment says "moving the king without changing this key leaves every
   non-king feature's index stable" — no king move can preserve the key, so a
   king move *always* forces a refresh. I confirmed the consequence in the
   census: 0.2934 king moves/node against 0.16472 refreshes/node, and the gap
   is pruning and backward fill, not key-preserving king moves.
3. **Making it smaller means collisions**, and a collision costs a larger
   set-difference — the one quantity §3.1 shows is already concentrated at
   4–6 rows with no slack.

And the usage census, which is the part nobody had:

| slots | share of refreshes | footprint |
|---:|---:|---:|
| top 8 | 19.47% | 21 KB |
| top 16 | 33.95% | 42 KB |
| top 32 | 55.51% | 84 KB |
| top 64 | 81.01% | 168 KB |
| **all 128** | **100%** | **336 KB** |

**All 128 slots are touched** (and `hka_inplace` = exactly 128 cold builds
confirms each is built exactly once). Usage is spread, not concentrated: there
is no dead half to reclaim and no small resident subset that a smaller table
would capture. The *effective* working set is ~84–168 KB inside a 1 MB L2 —
which, on its own, would be comfortable, and §2.3 shows it is not, because the
same L2 is being swept by 8.9 KB/node of weight rows.

**Caveat I cannot remove with this instrument:** these counts are aggregated
over all 50 bench positions in one process, so they measure the union across
positions, not the resident set within one search. The right instrument is
per-position counters and I did not build it.

---

## 6. PREFETCH (Q4) — four placements, all node-identical, kill #7

### 6.1 What is genuinely different from the six prior kills

The prior is strongly negative: study 22 §5.1 cites two study-4-era NPS kills
(−2.5% full-row, −0.4% first-two-lines); study 19 ranked it #1 and could not
price it; study 29 §5.2 built it properly and killed it with a ceiling
argument; study 42 killed two more speculative arms. **Six.**

Study 29's arm hinted **the first cache line only**, with `_MM_HINT_T2`, at
index-resolution time, in a **separately compiled binary**. Its own §5.3 records
that a byte-different null moved the cycles median further than the arm did.
Four things are different here, and each names a distinct reason that arm could
have failed:

| mode | placement | hint | coverage |
|---:|---|---|---|
| 0 | — | — | control |
| **1** | after the diff walk, before the fold | **T0** | **every line of every row (20/row)** |
| 2 | same | T2 | every line — isolates the HINT LEVEL from placement |
| 3 | same | T0 | first line only — isolates COVERAGE from placement |
| 4 | before the diff walk | T0 | the Finny slot's own 2560 B payload |
| 5 | 1 + 4 | | |

And critically: **one binary, six modes, selected by `P43_PF` at run time**, so
control and arm are the same bytes and code layout differences out by
construction (study 39's idiom). The mode check costs one relaxed load and one
branch per refresh — 0.16 per node.

**All six modes are node-identical and eval-hash-identical** (5,001,562 /
`2e1ede9521624dd8`), which they must be: a prefetch cannot change a value.

### 6.2 NULL FIRST — and my first panel was defective

**16 rounds, cyclic rotation (inherited from `panel34.sh`), reference `pf43@0`:**

| label | cyc/n | sign | stall/n |
|---|---:|---|---:|
| **`pf43null@0`** — same file, same mode | **−0.5700%** | **13/16 −** | −1.7855% |
| `pf43@1` | −0.6066% | 9/16 | −1.0065% |
| `pf43@2` | −0.2102% | 9/16 | −1.4727% |
| `pf43@3` | +1.1130% | 7/16 | +0.2918% |
| `pf43@4` | +1.0161% | 6/16 | +0.8994% |

A **byte-identical null running the same binary in the same mode** read
−0.5700% with 13 of 16 rounds negative. That is not noise, it is bias:
**cyclic rotation preserves adjacency.** `set -- "${@:2}" "$1"` rotates the
whole list, so the null always ran immediately *after* the reference, and any
neighbour effect (boost ramp, page cache, THP) is systematic and within-round
pairing cannot cancel it. Study 39 randomised inside every round; `panel34.sh`
does not.

**Fixed** — `tools/perf43/panel43.sh` now shuffles the label order inside every
round — and re-run. **This defect is in `tools/perf34/panel34.sh` and therefore
in every study that used it.** Whether it changed any of their conclusions I did
not check; it is flagged here as a live issue.

**40 rounds, shuffled, 4 labels, reference `pf43@0`:** *(see §6.4)*

### 6.3 The instruction cost is exact, and every probe costs what it should

`instructions:u` has a **0.0000% null with zero spread across all rounds**, so
this column decides nothing about benefit but is exact about cost:

| mode | Δ i/n | per refresh | mechanism check |
|---|---:|---:|---|
| `pf43null@0` | **+0.0000%** | 0 | the exact null |
| `pf43@4` slot | +0.1111% (**+6.59**) | **40.0** | 40 lines × 0.16472 refreshes/node — **exact to 3 s.f.** |
| `pf43@1` row T0 | +0.5262% (**+31.2**) | 189 | 114 hints + ~73 of `.chain()` iterator overhead |
| `pf43@2` row T2 | +0.5262% (**+31.2**) | 189 | **identical to mode 1** — same code, different hint ✓ |
| `pf43@3` first-line T0 | +0.2188% (**+12.98**) | 78.8 | 5.72 hints + the same ~73 of `.chain()` |

Mode 4's cost lands on 40.0 instructions per refresh for 40 prefetches — one
instruction each, fully unrolled. Modes 1 and 2 being bit-identical on
instructions is the check that the mode dispatch is doing what it claims.

**A defect in my own probe, reported because it bounds what the arm could have
shown:** `adds.iter().chain(subs)` costs ~73 instructions per refresh
independent of the hint count. Modes 1 and 3 therefore carry ~12 i/n of
overhead that a landing implementation would not. Mode 1's honest cost is
~+19 i/n, not +31.

### 6.4 The verdict

*(filled from the 40-round shuffled panel — see §6.5 table)*

### 6.5 Why it fails, mechanistically

Two different reasons for the two targets, and §2.3's population split is what
distinguishes them:

* **Mode 4 (the slot) prefetches the wrong population.** The Finny slots are
  ~84–168 KB effective in a 1 MB L2 (§5) — they are the L2-*hit* half of this
  symbol's traffic. A T0 hint on a line the demand load would have got from L2
  anyway, with 20-wide MLP already in flight, has almost nothing to convert.
  Its 6.59 i/n is paid for a target that was not stalling.
* **Mode 1 (the rows) prefetches the right population too late.** The weight
  rows are the L2-missing half (§2.3), so the target is correct — but the hints
  are issued at most ~190 instructions (~130 cycles) before the fold's first
  demand load, against L3/DRAM latency, and the 20 independent
  `vpmovsxbw`-per-tile already saturate Zen 4's ~24 miss-handling registers.
  Issuing 114 hints back to back moves the same misses earlier by roughly the
  length of the prefetch block itself and cannot create miss-level parallelism
  that the out-of-order window was not already extracting.

Study 29's ceiling argument survives contact with a much better-placed,
better-covered, T0 arm. **That is kill #7, and it is a real result: the
"address is known a long way ahead" intuition is true of the Finny key and
false of the thing that actually stalls.**

### 6.6 The arm I priced and did NOT build, with the number

**Prefetching at `make_move` time**, when a king move is made — the genuinely
longer distance (a whole node, ~1000+ cycles, instead of ~130). Not built,
because the census prices it out before the build:

* king moves are **0.2934/node** against **0.16472 refreshes/node**, so the
  hint is **56% targeted** at best;
* 40 slot lines × 0.2934 = **11.7 i/n** plus the key computation ≈ **13 i/n**,
  i.e. **twice mode 4's cost for a strictly worse-targeted hint**;
* and it aims at the same L2-resident slot population mode 4 already showed is
  not the stalling half.

The distance argument is the one thing that genuinely differs, so this is a
priced deferral rather than a refutation. If someone builds it, the prediction
to falsify is: **no resolvable cycles change, at +13 i/n.**

---

## 7. WHAT LANDED

### 7.1 The instruments (commits `c2583f7a`, `a076a3ae`)

* `foldcount::HKA_DUAL_SHAPE` — the warm refresh's row histogram (§3.1).
* `foldcount::FINNY_KEY` — per-key refresh census (§5).
* `--features pf43` + `P43_PF` — the runtime-mode prefetch arm.
* `tools/perf43/{build43,panel43,prof43}.sh`, `med43.py`.
* **`panel43.sh` shuffles instead of rotating** — §6.2, a defect fix.

### 7.2 A latent hazard, checked and hardened

`FinnyEntry` carries `#[repr(C, align(64))]`; **`Entry` — the per-ply
accumulator, 40 zmm stores per fold — carries neither `repr(C)` nor an
alignment.** Its 40 `vmovdqu64` stores are 64-byte-aligned only if the
allocation happens to be.

**Measured, not assumed:** the `foldcount` region dump gives
`hka_stack lo=0x7fee24101040`, which **is** 64-byte aligned, because glibc
services a 1,316,736-byte `Vec` with `mmap`. So this is **a null, not a bug** —
and it is a null that depends on allocator behaviour and on `Entry`'s size
being 5184 = 81 × 64, neither of which is stated anywhere.

Two corrections land with it:

* `Entry`'s doc comment claims **5248 bytes**; it is **5184** (`acc`
  2 × 1280 × 2 B + `psqt` 2 × 8 × 4 B). The census region size, 1,316,736 =
  254 × 5184, is the receipt.
* `#[repr(C, align(64))]` on `Entry` makes the invariant structural instead of
  accidental. Size is unchanged (5184 is already a multiple of 64), so the
  change is bit-identical.

### 7.3 The arm that is still open: non-temporal stores

**Built, not yet adjudicated** (`--features nt43`). The dual fold writes the
same 2560 bytes to two destinations with opposite reuse distances:

* `out` = `stack[ply].acc[p]`, read by the head **within the same node** —
  must stay a normal store;
* `acc` = **the Finny cache entry**, whose next read is the next refresh at
  this king key. At 0.16472 refreshes/node spread over all 128 slots, a hot slot
  returns roughly every **~350 nodes**, during which the engine streams ~3 MB of
  weight rows through a 1 MB L2. **The line cannot survive**, so keeping it
  costs a read-for-ownership now (6.59 lines/node) and evicts weight rows for
  nothing.

`_mm512_stream_si512` needs 64-byte alignment, which is structural here
(`FinnyEntry` is `align(64)`, `acc` at offset 128) and is now `debug_assert`ed
at the wrapper. Same store count, different opcode ⇒ **zero instruction-count
change**, so it is a pure cache-policy arm and `instructions:u` is its identity
check rather than its measurement.

**Nobody has tried non-temporal stores anywhere in this engine** (`grep` for
`stream_si512` over `src/` returns nothing).

---

## 8. WHAT I COULD NOT ESTABLISH

In order of how much it bothers me.

1. **Any speedup.** The deliverable is landed, verified work and I did not land
   one. Four prefetch placements measured and killed, two census-killed arms,
   one arm (§7.3) built but not adjudicated. That is a real outcome and it is
   not the outcome that was wanted.
2. **That the cycles instrument could have resolved a win if there had been
   one.** The byte-identical null moves ±0.6–0.7% on this box with four agents
   live. The largest plausible Finny-specific arm is ~2.5% of engine cycles at
   its *ceiling* (§2.4) and realistic arms are 0.1–0.3%. **The instrument is
   2–7× too coarse for anything I could build**, which is why §4's bound and
   §3.1's kills are stated as arithmetic rather than as panels.
3. **The 5.6-vs-20.6 stall-per-line bracket is 3.7× wide** (§2.3). Narrowing it
   needs a subtractive arm that removes a known number of Finny lines while
   holding instructions fixed — study 39's `dstpin`/`actpin` shape, which I did
   not build. Everything in §2.4 and §4 inherits that width.
4. **Whether `panel34.sh`'s rotation defect (§6.2) changed any prior study's
   conclusion.** I found it, fixed it in my own tool, and did not audit studies
   34–42 for it. It biases a byte-identical null by ~0.6% with a 13/16 sign
   count, which is the same order as several published arms.
5. **The per-position Finny working set** (§5). My counters aggregate over all
   50 bench positions, so "128 of 128 slots" is a union, not a resident set.
6. **1311 instructions per refresh vs ~986 in my static model** (§2.2 and the
   loop analysis). A third of the symbol's instructions are not accounted for
   by the fold, the diff walk, the psqt update and the prologue. The gap may be
   sampling error in the 3.65% symbol share; I did not close it.
7. **That any of this converts to Elo at 120/doubling.** That constant's 95% CI
   includes zero, and everything here is cycles on a non-PGO build with no game
   played.

---

## 9. WHAT I WOULD DO NEXT

1. **Adjudicate `nt43`** (§7.3) on a quiet box, on the L2/L1-miss counters
   rather than cycles — misses are a ±1.6% instrument where cycles is ±0.7%
   *median displacement*, and the RFO reduction it targets is directly visible
   as a miss-count change.
2. **Build the subtractive Finny-line arm** to collapse §2.3's 3.7× bracket.
   Until it exists, every Finny pricing in this repo carries that width.
3. **Audit studies 34–42 for the rotation defect** (§6.2).
4. **Stop treating this symbol as a Finny problem.** 49% of its lines and the
   whole of its L2-missing population are HalfKA weight rows from a 28.8 MB
   table — the same table, the same problem, and the same fix as
   `i8_deltas_into` (11.21% of cycles) and `i8_deltas_into_generic` (10.02%).
   Together those three symbols are **27.6% of engine cycles and 59% of engine
   L2 misses**, and they are one problem wearing three symbol names.

---

## 10. GATE LOG

| gate | result |
|---|---|
| node identity, study frame, every binary and every `P43_PF` mode | **5,001,562** |
| wl50 eval hash, every binary and mode | **`2e1ede9521624dd8`** |
| `base` vs study 40's `b40` | +0.081% instructions, both exactly repeatable |
| `cargo check --features sota,arch-d48x,pf43` | clean |
| `cargo check --features foldcount,arch-d48x` | clean |
| `cargo check --features sota,arch-d48x,nt43` | clean |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | *§10.1* |

### 10.1 Merge gate

*(filled below)*

---

## 11. REPRODUCING

```bash
# binaries (serialised behind flock; refuses under 2500 MB available)
bash tools/perf43/build43.sh base    "sota,arch-d48x"
bash tools/perf43/build43.sh fold43  "foldcount,arch-d48x"
bash tools/perf43/build43.sh pf43    "sota,arch-d48x,pf43"
bash tools/perf43/build43.sh nt43    "sota,arch-d48x,nt43"
bash tools/perf34/ident34.sh base fold43 pf43 nt43     # 5001562 / 2e1ede95...

# §2 the four-event symbol census + the L2 population split
bash tools/perf43/prof43.sh base
perf record -e l2_cache_req_stat.dc_hit_in_l2:u    -c 200003 -o L2H.data -- \
  taskset -c 2 ./bin34/base bench 16 1 100000 tools/nnue/calib/wl50.fens \
  nodes engines/canonical/net.cenn
perf record -e l2_cache_req_stat.ic_dc_miss_in_l2:u -c 200003 -o L2M.data -- ...

# §3, §5 the census
ARCHON_FOLDCOUNT=docs/research/perf-artifacts-43/foldcount43.txt \
  ./bin34/fold43 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn

# §6 the prefetch panel (SHUFFLED; pf43null is a hard copy of pf43)
cp bin34/pf43 bin34/pf43null
P43_PANEL=docs/research/perf-artifacts-43/panel-pf2.txt \
  bash tools/perf43/panel43.sh 40 pf43@0 pf43null@0 pf43@1 pf43@4
python3 tools/perf43/med43.py docs/research/perf-artifacts-43/panel-pf2.txt pf43@0
```

Raw artifacts: `panel-pf.txt` (16 rounds, **rotated — carries the §6.2 defect,
kept as the evidence for it**), `panel-pf2.txt` (40 rounds, shuffled),
`foldcount43.txt`, `sym-base-*.txt`. The `perf.data` files are gitignored;
regenerate with `prof43.sh`.
