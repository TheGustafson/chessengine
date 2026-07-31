# Perf study 49 — `current_full`'s 3.04 c/load is an ARTIFACT, and it takes the census's 12.5× spread with it

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `e33e2f09` (study 48
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf49/` · **Artifacts:**
`docs/research/perf-artifacts-49/` · **Branch:** `perf-study-49-currentfull`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
physical core 2 (siblings 2,10 — the quietest pair, 21.8 % combined, measured
before starting).

---

## ★ THE ANSWER, FIRST

**The 3.04 stall cycles per load is not real.** It reproduces as a
*measurement* — I read 2.80 on my own binary against study 48's 3.04 — but it is
not a property of `HalfkaContext::current_full`'s loads.

| `HalfkaContext::current_full` | value | instrument |
|---|---:|---|
| share of engine load dispatches | **2.20 %** | sampled `ls_dispatch.ld_dispatch` (study 48: 2.27 %) |
| share of engine loads | **2.98 %** | AMD IBS, precise |
| **share of engine L1 misses** | **0.592 %** | AMD IBS, precise |
| **share of engine load-miss LATENCY** | **0.478 %** | AMD IBS, precise |
| its own loads' L1 hit rate | **93.6 %** | AMD IBS, precise |
| share of engine stall attributed to it | **7.96 %** | sampled `ex_no_retire.load_not_complete` |

**The stall charged to it is 16.7× the miss latency its own loads produce — and
at or above 100 % of what they could possibly produce.** A cycle counted by
`ex_no_retire.load_not_complete` for a symbol requires the oldest un-retired op
to be an incomplete load *belonging to that symbol*, so every such cycle lies
inside one of that symbol's own miss intervals. The ratio therefore has a hard
physical ceiling of 100 %. Engine-wide it is **7.9 %**, because misses overlap
~6-way. For `current_full` it is **131 %**.

**It is a caller absorbing its callees' stall.** `current_full` sits directly
above `i8_deltas_into` (33.4 % of the engine's true miss latency) and
`finny_refresh_into` (8.3 %), and its own body is independent scalar index
arithmetic that the out-of-order machine runs ahead into while those misses are
outstanding. Three other scalar callers show the same signature and two of them
are worse: **`Worker::do_move` 461 %**, **`Board::see_ge` 273 %**,
`search::<NonPv>` 56 %. Every SIMD leaf is far *below* the engine average:
`forward_into` **2.0 %**, `i8_deltas_into` **3.1 %**, `i8_deltas_into_generic`
3.9 %.

**So study 48's assumption A2/A5 — "per-symbol attribution of a sampled
non-precise stall counter is sound" — is FALSE for scalar callers of missing
leaves, which is exactly the case study 48 said it doubted.** It is *sound* for
leaves: a known-answer injected stall was attributed **91 %** correctly (§3.3).

### What that does to study 48's census

Study 48's headline was that stall-per-load spans 12.5× and that the lever is
"fewer loads *that miss*, and those are weight rows". **The direction survives;
the ranking does not.** Re-ranked by true miss latency:

| symbol | study 48's c/load | study 48's %stall | **true %miss-latency (IBS)** |
|---|---:|---:|---:|
| `i8_deltas_into` | 1.41 | 12.85 | **33.4** |
| `i8_deltas_into_generic` | 1.16 | 11.92 | **22.8** |
| **`SotaHead::forward_into`** | **0.29** | **4.23** | **17.5** |
| `finny_refresh_into` | 1.96 | 8.31 | **8.3** |
| `TranspositionTable::probe` | — | <1 | **5.2** |
| **`HalfkaContext::current_full`** | **3.04** | **7.61** | **0.478** |

**`forward_into` is the third-largest true stall source in the engine.** Study
48 priced its loads at 0.29 c/load, called them "the *cheapest* loads in the
engine … just-written 2560-byte arrays, L1-hot", and used that to conclude its
own landed win could only be worth 0.24 % of the stall. **IBS measures 40.5 % of
`forward_into`'s loads MISSING L1** — they are not L1-hot. §6 has the mechanism,
and it is not the one anyone predicted.

---

## ★ WHAT LANDED

1. **`tools/perf49/` — the IBS instrument.** `ibs49.py` builds a per-symbol
   census from AMD Instruction-Based Sampling: precise rip plus measured
   per-load DC-miss latency, so there is no skid by construction. **This repo
   has never used a precise instrument.** Zen 4 has no PEBS; IBS is the only
   one, and it was sitting enabled on the box the whole time. Plus `where49.sh`,
   `census49.sh`, `exact49.sh` (blocked Latin square), `med49.py`,
   `attrib49.py` (what *kind* of instruction a stall counter lands on),
   `join49.py`, `master49.py` (the three-instrument table), `build49.sh`.
2. **`skidprobe` (`src/eval/nnue/skidprobe.rs`, diagnostic feature)** — a
   known-answer stall injector: a strictly serial cache-missing chase in its own
   `#[inline(never)]` leaf, called from `current_full`, so a profiler's
   attribution of an unambiguously-owned stall can be *measured*. `black_box`ed;
   node count unchanged.
3. **`current_full` stops iterating `Color::ALL` by value** (§5.2) — the
   anti-pattern this same file forbids in prose. Three stores of a compile-time
   constant per call plus a per-iteration stack reload; **stack frame 264 → 232
   bytes**.
4. **The `Meta` size claim is corrected and pinned** (§5.3). The doc said "12
   bytes per ply … ~3 KB … 48 cache lines"; `Meta` is **16 bytes**, the array is
   **4064 B / 64 lines**, and nothing checked it. Now a `const` block assertion
   fails the build if it moves — which matters beyond the doc, because
   `compute`'s `meta` indexing is a `shl $0x4` only while the stride is 16.
5. **One arm built, gated, measured and REFUTED with exact counters** (§5.1).

### ★ And the honest headline on the third question: NO SPEEDUP LANDED, and the reason is the first result

The commission's part 3 was *"reduce it — 158.4 stall cycles/node is ~3.8 % of
the frame; landing even half is worth more than study 48's landed win."*
**Part 3's premise is void at this site**: §2 puts `current_full`'s own true
stall at ~8 cycles/node, not 158, so there is no 3.8 % here to halve. Two arms
were built against it anyway — one refuted (+0.054 % instructions, 0/15), one
null — and together with study 48 §5's outlining and §4's psqt probe that makes
**four measured failures at this address**. That is now a closed question rather
than an open one.

**Where the 3.8 % actually is, is §4**, and I priced three kernel-level attacks
on it *with arithmetic* rather than building them:

| candidate | why not built |
|---|---|
| split `activate_inner` into two half-passes to halve its 10 concurrent streams | **arithmetically impossible**: `prod = mulhi(left << 7, right)` multiplies lane against lane + 640, so the halves are pairwise-multiplied, not independent |
| fuse `find_nnz` into `activate` to delete a 1280 B re-read of `ft` | **prices negative**: saves 20 zmm loads/eval (8.6 loads/node) but needs a nonzero mask accumulated across 40 scattered 32-byte stores in non-monotone output order — ~160 extra scalar ops/eval ≈ **+69 i/node** |
| `prefetchnta` on the fold's weight rows to stop them evicting the accumulator | untried mechanism (studies 29 and 43 both tried T0/T2 for *latency*, not for *pollution*), but a demand load still allocates normally after an NTA hint, and study 43 measured 114 hints/refresh at +31.2 i/node |

The one attack §4 does endorse is **channel fusion** — the accumulator pair is
10.24 KB per evaluation where Stockfish carries 4.10 KB, and `sota_simd.rs:726`
already says so. That is a net-architecture decision, not a kernel one, and it
is out of scope for a perf study.

---

## 0. THE FRAME, RE-DERIVED (rule 8)

`tools/perf49/census49.sh`, 3 unmultiplexed groups × 3 reps, median.

| | study 48 (`base48`) | **this study (`base49`)** | |
|---|---:|---:|---|
| `instructions:u` | 6170.77 | **6153.06** | **−0.287 %** |
| `ls_dispatch.ld_dispatch:u` | 2297.01 | **2275.86** | −0.92 % |
| `ls_any_fills_from_sys.all` | 576.20 | **575.74** | **−0.08 %** |
| `ex_no_retire.load_not_complete` | 2081.43 | 1764.05 | |
| `ls_alloc_mab_count` / cycles (MLP) | 5.768 | **5.955** | |
| `cycles:u` | 4162.8 (derived) | 3753.76 | quieter box |
| DRAM fills | 7.27 | 5.53 | |

**The −0.287 % on `instructions:u` is study 48's own landed `activate` arm**,
which measured −0.286 % and is now in `main`. Two independent measurements of
the same change agreeing to 1 part in 300 is the cleanest confirmation of that
result available, and it also says my build is the right one. Fills reproduce to
**0.08 %**.

`cycles:u` is 10 % lower than study 48's inherited denominator because the box
was quieter, so **every percentage-of-cycles figure below is against my own
measured 3753.76**, never an inherited one.

---

## 1. THE INSTRUMENT — why IBS, and what it actually gives

`ex_no_retire.load_not_complete` is not a precise event. The PMU interrupt is
delivered tens of cycles after the counter overflows, and the rip recorded is
wherever the machine had got to by then. Two hypotheses predict identical
observational profiles:

* **REAL, low memory-level parallelism** — the symbol's own few misses are
  serial, so each blocks retirement for its whole latency.
* **MISATTRIBUTION** — the stall belongs to a callee, and the rip lands in
  whatever independent work the caller ran ahead into.

No amount of re-sampling the same counter separates those. **IBS does.** It tags
a micro-op at issue and records that op's own rip (`IbsOpRip`) plus, for a load
that misses the data cache, the measured miss latency (`IbsDcMissLat`). Linux
exposes the latter as `PERF_SAMPLE_WEIGHT` and the memory level as `data_src`.

```
perf record -e ibs_op/cnt_ctl=1,swfilt=1/ -c 8000 -d -W ...
```

`swfilt=1` is what makes it work at `perf_event_paranoid=2` (kernel-side sample
filtering). **6 runs, 1,514,793 tagged ops, 457,179 loads, 138,486 DC-missing
loads, 6,997,327 summed miss cycles.**

**Two traps, both of which I fell into first.** (a) My validation
microbenchmark was a pointer chase whose "next" map was a multiplicative hash;
that has short rho cycles, so the whole chase sat in L1 and IBS correctly
reported `L1 hit`/latency 0 — I nearly concluded the kernel wasn't populating
the field. A true single-cycle permutation fixed it. (b) The same defect then
reappeared in the `skidprobe` arm (§3.3), seeded from `ply`, which takes ~60
distinct values. **A cache-miss probe must be checked for having enough distinct
addresses before its null is believed.**

### 1.1 IBS is not under-sampling the symbol under test

The obvious way this analysis could be wrong is if IBS simply saw fewer of
`current_full`'s loads. It does not — if anything it sees *more*, which makes
every bound below more conservative:

| `current_full`, share of engine loads | |
|---|---:|
| sampled `ls_dispatch.ld_dispatch`, this study | 2.20 % |
| sampled `ls_dispatch.ld_dispatch`, study 48 | 2.27 % |
| **IBS** | **2.98 %** |
| what `load_not_complete` charges it in stall | **7.96 %** |

---

## ★ 2. THE MASTER TABLE — three instruments, one row per symbol

`tools/perf49/master49.py`. `misslat c/n` is the symbol's IBS share of miss
latency applied to `ls_alloc_mab_count` = **22354.3 cycles/node**, which is the
hardware's own count of total outstanding-miss cycles. `stall/lat` is the
fraction of its own loads' miss latency the symbol converts into retirement
stall: **physical maximum 100 %, engine-wide value 7.9 %.**

| symbol | loads/n | stall c/n | misslat c/n | **stall/lat** | study-48 c/load |
|---|---:|---:|---:|---:|---:|
| `i8_deltas_into` | 188.2 | 228.6 | 7449.8 | 3.1 % | 1.21 |
| `search::<NonPv>` | 280.6 | 202.7 | 364.0 | **55.7 %** | 0.72 |
| `i8_deltas_into_generic` | 211.4 | 197.6 | 5090.6 | 3.9 % | 0.93 |
| **`HalfkaContext::current_full`** | 50.1 | 140.4 | 107.0 | **131.3 %** | **2.80** |
| `finny_refresh_into` | 82.4 | 131.2 | 1852.4 | 7.1 % | 1.59 |
| `MovePicker::init_quiets` | 188.0 | 113.8 | 246.1 | **46.2 %** | 0.61 |
| `ThreatContext::at_full` | 141.6 | 111.3 | 349.4 | **31.9 %** | 0.79 |
| `SotaHead::forward_into` | 281.3 | 79.7 | **3907.0** | **2.0 %** | 0.28 |
| **`Worker::do_move`** | 116.3 | 76.7 | 16.7 | **460.7 %** | 0.66 |
| `make_move_with::<TtPrefetch>` | 82.6 | 41.1 | 87.3 | 47.1 % | 0.50 |
| `qsearch::<NonPv>` | 21.8 | 36.9 | 67.9 | 54.3 % | 1.69 |
| **`Board::see_ge`** | 43.5 | 22.4 | 8.2 | **273.0 %** | 0.52 |
| — ENGINE | 2275.9 | 1764.0 | 22354.3 | **7.9 %** | 0.78 |

**Read the `stall/lat` column and the whole thing falls out.** Every symbol above
100 % is a scalar function that *calls* something; every symbol far below the
7.9 % average is a SIMD leaf that *misses*. The counter is moving stall from the
leaves to their callers, and the amount it moves scales with how much
independent scalar work the caller has for the machine to run ahead into.

**The three instruments share the same symbol address ranges**, so inlining is
not a confounder for the ratio: whatever `apply_dirty_into` contributes to
"`current_full`" contributes identically to all three columns.

---

## 3. THE FOUR TESTS, INCLUDING THE ONE THAT FAILED

### 3.1 The bound — `stall ≤ own miss latency`, and `current_full` breaks it

Two independent anchors for the denominator, because the margin is not large and
I did not want it resting on one. `current_full`'s attributed stall is
**140.4 c/node** (7.96 % of 1764.05).

| anchor for the symbol's OWN total miss latency | derivation | value | ratio to 140.4 |
|---|---|---:|---:|
| **`ls_alloc_mab_count`** (hardware MAB-occupancy cycles) | 0.478 % of 22354.3 | **107.0 c/n** | **131 %** |
| IBS fills × latency | (0.592 % × 575.74 fills/n) × 40.8 c | **139.0 c/n** | **101 %** |

**The two disagree by 1.30×, and the MAB anchor is the correct one.** A single
L1 miss can be charged several IBS `DcMissLat` samples when several loads hit the
same in-flight line; MAB counts each missing line's occupancy once. That is
exactly the 29 097 / 22 354 = 1.30 gap between the two engine totals. So the
fills route systematically *over*-states a symbol's own latency, and 101 % is a
generous upper bound rather than the estimate.

**Either way the conclusion is the same, and it does not depend on which anchor
you prefer: `current_full` would have to convert 101–131 % of its own loads'
miss latency into retirement stall, when the physical maximum is 100 %, the
engine-wide value is 7.9 %, and every SIMD leaf is at 2–4 %.** Even the most
generous reading requires this one scalar function to be perfectly serialised
with zero overlap while the machine around it runs 6-way.

Statistical weight: the 0.478 % rests on **820 DC-miss samples** (Poisson
se ≈ 3.5 %), so the MAB-anchored ceiling is 107.0 ± 3.7 c/node.

**This is the study's strongest-in-principle test and its narrowest in margin**,
which is exactly why §3.2 and §3.3 exist and why the conclusion is not rested on
it alone.

### 3.2 What kind of instruction does the stall land on?

`tools/perf49/attrib49.py` classifies every annotated instruction by whether it
carries a memory *source* operand. A faithful load-stall sample should land on a
load.

| symbol | on a **load** | store-only | no memory at all |
|---|---:|---:|---:|
| `i8_deltas_into` (leaf) | **50.3 %** | 18.8 % | 31.0 % |
| `forward_into` (leaf) | **49.7 %** | 2.3 % | 47.8 % |
| **`current_full`** | **12.2 %** | 23.4 % | **64.6 %** |

**The single hottest "load stall" instruction in `current_full` — 17.5 % of the
symbol's entire stall — is `mov %rax,0x20(%rsp)`, a stack STORE**, in the middle
of HalfKA index arithmetic. Only **1.5 %** of its stall lands within three
instructions of a `call` return, so this is *not* simple across-the-return skid:
it is the machine running far ahead into the caller's independent scalar work.

### 3.3 ★ The controlled experiment — a stall whose owner is known by construction

Observation cannot separate the two hypotheses, so I injected a stall that has
no ambiguity: `skidprobe::stall`, a strictly serial dependent chase through a
32 MB single-cycle permutation, in its own `#[inline(never)]` leaf that does
nothing else, called from the top of `current_full`. **Node count stayed
5,001,562**, so the probe is inert to the search.

| | base49 | skid49 | Δ |
|---|---:|---:|---:|
| `ex_no_retire.load_not_complete`, engine | 1764.1 c/n | 2567.1 c/n | **+803.0** |
| attributed to `skidprobe::stall` | — | 732.6 c/n | **91 % of the injection** |
| attributed to `current_full` | 140.4 c/n | **31.1 c/n** | **−109.3** |
| attributed to `i8_deltas_into` | 228.6 c/n | 370.2 c/n | +141.6 |

**Two results, and they point in opposite directions — both are reported.**

1. **The counter attributed 91 % of a known, unambiguously-owned stall to the
   right symbol.** So it is *not* uniformly broken, and the naive "this counter
   can't attribute" story is wrong. It attributes a *tight serial leaf loop*
   correctly, because there is nothing for the machine to run ahead into.
2. **`current_full`'s own attributed stall collapsed 4.5×, from 140.4 to 31.1
   c/node, under a change that added work and removed none.** A quantity that
   moves like that under an upstream perturbation is not a property of the
   symbol's own loads.

### 3.4 ★ THE TEST THAT DID NOT WORK — reported because it was going to be the headline

`join49.py` joins, per 16-byte window inside one symbol, the stall the counter
attributes there against the miss latency IBS measures there. The prediction was
clean: under "real, serial", the two should correlate; under misattribution,
they should not. `current_full` reads **r = +0.132**, and **60.7 % of its
attributed stall sits in windows with zero measured miss latency**. That looked
decisive.

**It is not, because the controls fail the same way:**

| symbol | Pearson r | stall in zero-latency windows |
|---|---:|---:|
| `current_full` | +0.132 | 60.7 % |
| `i8_deltas_into` (pure leaf, owns its stall) | **+0.147** | 14.2 % |
| `forward_into` | +0.426 | 38.2 % |
| `finny_refresh_into` | +0.543 | 46.0 % |
| `at_full` | +0.651 | 37.2 % |

`i8_deltas_into` is a leaf with no calls at all and 50.3 % of its stall on real
loads, and it reads r = 0.147 — as low as the symbol under test. **Within-symbol
correlation does not discriminate on this box**, which is study 36 §2.3's
finding arriving from a new direction. The zero-latency-window share does rank
`current_full` highest, but not outside the controls' range. **No weight is
placed on §3.4 anywhere in this study.**

---

## 4. WHERE THE ENGINE'S STALL ACTUALLY IS

Ranked by IBS miss latency, the frame is **22354 cycles/node of outstanding-miss
time, converted to 1764 cycles/node of retirement stall (47.0 % of all cycles)**
at an engine-wide overlap of 5.955.

| symbol | %miss-latency | miss/load | L1/L2/L3/RAM of its sampled loads | lat/miss |
|---|---:|---:|---|---:|
| `i8_deltas_into` | 33.4 % | 82.5 % | 7134 / 28378 / 4645 / 556 | 69.4 |
| `i8_deltas_into_generic` | 22.8 % | 63.5 % | 18069 / 28008 / 2921 / 495 | 50.7 |
| `forward_into` | **17.5 %** | **40.5 %** | 40612 / 27310 / 342 / **6** | 44.2 |
| `finny_refresh_into` | 8.3 % | 52.9 % | 7866 / 6533 / 1987 / 327 | 65.5 |
| `TranspositionTable::probe` | 5.2 % | 34.3 % | 2033 / 143 / 14 / **905** | **344.7** |

**Two things nobody has written down.**

**(a) `forward_into`'s loads miss 40.5 % of the time and are essentially pure
L2** — 342 L3 and *six* DRAM samples out of 68 k. It is not reading a big table;
it is re-reading the accumulators that were just written, and losing them from
L1 in between. The mechanism is structural: `activate_inner` streams **12,800 B
per evaluation** out of **four separate allocations** — `pst` from
`HalfkaContext.stack` (a 1.3 MB array), `threat` from `ThreatContext.stack` (a
*different* 1.3 MB array), `ft_scale` from the leaked net blob, and `ft` on the
worker stack — touching each at `lane` and `lane + 640` for two perspectives,
i.e. **ten concurrent streams**. Between the fold writing `stack[ply].acc` and
the head reading that same memory, the fold's own 7.18 weight rows (9190 B)
stream through L1 and evict it.

**I checked the obvious fix and it is not available:** splitting the loop into
two half-passes to halve the stream count is impossible, because
`prod = mulhi(left << 7, right)` multiplies lane against lane+640 — the halves
are pairwise-multiplied, not independent. The real fix is the one
`sota_simd.rs:726` already names: **fuse the two channels so the pair is 4.10 KB
as Stockfish carries it, not 10.24 KB.** That is a net-architecture change, not
a kernel one.

**(b) `TranspositionTable::probe` is the densest true site in the engine at
344.7 cycles per miss, 29 % of its missing loads served from DRAM** — 7× the
engine's mean miss latency. It is 5.2 % of all miss latency on 0.71 % of loads.
Study 42 already measured two extra prefetch placements as nulls and the three
live probe sites already carry a long-distance hint, so the cheap levers are
spent; but no study has yet priced *this* site in the true frame.

---

## 5. ARMS

### 5.1 ★ REFUTED — precomputing the "last king move" index (`lk49`)

**The hypothesis.** `compute` walks the meta stack **twice** on every
uncomputed call, once per perspective: a downward walk for `ancestor`, then a
second reversed pass over `(ancestor+1)..=ply` iterating each dirty record's sub
list to find the last king move. That fact is a running maximum along the path,
so `push` can carry it forward in O(1). Instrumented region cost: ~17 i/node.

**Built, gated bit-identical (5 001 562 / 1 739 588 / `2e1ede95…79c4`), and it
LOST on the exact counter.** 15 rounds, blocked Latin square, byte-identical
null riding in the rotation:

| `lk49` vs `base49` | median | trimmed | se | neg/n | p |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | **+0.054 %** | +0.054 | **0.000** | **0/15** | 0.0001 |
| `ls_dispatch.ld_dispatch:u` | +0.386 % | +0.364 | 0.055 | 1/15 | 0.0010 |
| **`ls_dispatch.store_dispatch:u`** | **+1.187 %** | +1.242 | 0.055 | 0/15 | 0.0001 |
| `cycles:u` | +0.217 % | +0.049 | 0.271 | 7/15 | 1.00 |
| `ex_no_retire.load_not_complete` | +0.224 % | −0.314 | 0.408 | 7/15 | 1.00 |

**The null, published as required:** `instructions:u` **0.000 %**,
`ld_dispatch` −0.039 %, `store_dispatch` +0.003 %, `cycles:u` −0.510 %,
`load_not_complete` −0.777 %. Realised mean position **1.00 for all three
labels**.

**Why it lost, stated plainly: the scan was never worth 17 i/node.** It exits
almost immediately in the common case (span 1, `n_sub` test then two compares),
and the memo's maintenance in `push` — which runs *more* often than `compute`,
once per made move against 0.86 — cost more than the scan it removed, +1.19 %
of all stores. **This is the third failed attack on `current_full`** after study
48 §5's outlining and its psqt probe. It is the last one that should be
attempted on instruction-count grounds: §2 says the site's own true stall is
~8 c/node, so there is nothing there to win.

### 5.2 LANDED — `current_full` stops iterating `Color::ALL` by value

`for perspective in Color::ALL` consumes a two-byte array, which builds a
`core::array::IntoIter` (pointer plus alive range) that LLVM declined to
promote. Every call opened with:

```
movq $0x0,0xf0(%rsp)              ; iterator alive.start
movq $0x2,0xf8(%rsp)              ; iterator alive.end
movw $0x100,0x100(%rsp)           ; Color::ALL = [White, Black]
  ...
movzbl 0x100(%rsp,%rax,1),%r8d    ; and the colour reloaded per iteration
```

**This is study 22 arm 3's anti-pattern, and `finny_refresh_into` forbids it in
prose three hundred lines below in the same file** — the comment survived, the
same mistake reappeared in the neighbouring function. Iterating a `static` by
reference removes the iterator object entirely; **the stack frame drops 264 →
232 bytes**, which is the 264-byte frame study 48 §1.3 flagged and its outlining
arm failed to shrink.

Gates: **5 001 562 / 1 739 588 / `2e1ede95…79c4`**, all exact.

**Measured — and it is a NULL, reported as one.** 15 rounds, blocked Latin
square, byte-identical null riding in the rotation:

| `col49` vs `base49` | median | trimmed | se | neg/n | p |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.000 %** | −0.000 | **0.000** | **15/15** | 0.0001 |
| `ls_dispatch.ld_dispatch:u` | −0.267 % | −0.220 | 0.077 | 13/15 | 0.0074 |
| `ls_dispatch.store_dispatch:u` | −0.400 % | −0.362 | 0.090 | 13/15 | 0.0074 |
| `cycles:u` | +0.016 % | −0.225 | 0.343 | 7/15 | 1.00 |
| `ex_no_retire.load_not_complete` | +0.153 % | −0.140 | 0.584 | 7/15 | 1.00 |

**The null in THIS panel is not clean, and that is what decides it:**
`instructions:u` 0.000 %, but `ld_dispatch` **−0.206 %** (12/15, p = 0.035) and
`store_dispatch` **−0.166 %** (12/15). The arm's −0.267 % / −0.400 % are barely
past their own null, so no load or store claim is made.

The exact counter settles the size of the effect directly, since
`instructions:u` is deterministic here (se 0.000, and the byte-identical null is
0.000 % at every `n`): **30 774 901 100 → 30 774 892 240, i.e. −8 860
instructions in 3.08 × 10¹⁰, or −0.0018 per node.** LLVM did not delete work; it
traded three stores and a stack reload for the same number of ALU ops.

**So this is landed as documentation repair, not as a speedup** — the same
disposition, and for the same reason, as study 22 arm 3, which measured +0.8 i/n
and landed anyway: *a comment that forbids what the code does is worse than no
comment*, and this file now contains the forbidden pattern's own removal next to
the paragraph forbidding it. The one hard fact is the frame: **264 → 232
bytes**.

### 5.3 LANDED — the `Meta` size claim, corrected and pinned

The doc comment said **"12 bytes per ply … the whole `STACK_LEN`-deep array is
~3 KB — 48 cache lines"**. `DirtyPiece` is 14 bytes, not 10 — `(Piece, Square)`
is 3 bytes, because `Piece` is a two-field struct rather than a packed index —
so `Meta` is **16 bytes** and the array is **4064 B, 64 cache lines**.
Independently confirmed from the shipped disassembly: `compute` indexes `meta`
with `shl $0x4,%rbp` and reads `computed[p]` at `0xe(%r8,%rax,1)`.

Nothing checked it, in contrast to the `Entry` size in the very next paragraph,
which study 43 verified from the `foldcount` region dump. A `const` block
assertion now fails the **build** if it moves — and this is not merely
documentation hygiene: `compute`'s `meta` indexing is a `shl $0x4` *only* while
the stride is 16, so one extra byte in `DirtyPiece` silently turns thousands of
indexings per node into an `imul`.

---

## 6. GATES

| gate | result |
|---|---|
| node identity, study frame, every binary | **5 001 562** — exact on `base49`, `lk49`, `col49`, `skid49`, `null49` |
| d13 node identity | **1 739 588** — exact |
| wl50 eval hash | **`2e1ede95…79c4`** — exact |
| `instructions:u`/node vs study 48 minus its landed arm | 6153.06 vs 6153.1 predicted (**0.001 %**) |
| `ls_any_fills_from_sys.all` vs study 48 | 575.74 vs 576.20 (**−0.08 %**) |
| byte-identical null through THIS harness | `instructions:u` **0.000 %**, `ld_dispatch` −0.039 %, `store_dispatch` +0.003 % |
| realised mean panel position | **1.00** for every label |

**`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh`** — see the run
recorded alongside this document. The library suite is the real check on §5.1's
memo idea and on §5.2: the `debug_assert_eq!` written for `lk49` recomputed the
scan it replaced at every node of every test, and `Meta`'s new `const` block
assertion is evaluated during compilation, so a size change fails the build
rather than a test.

D24 perft ritual not required: nothing under `movegen`, `board` mutation,
`zobrist` or `attacks` is touched.

**No pin moves.** All three landed changes are bit-identical, so a pin update
would have been evidence one of them was not what it claims to be.

---

## 7. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | IBS `IbsOpRip` is precise — the sample IS the tagged op | Architectural: IBS tags at issue and records the op's own rip, which is why it exists. Cross-check: IBS's load share for `current_full` (2.98 %) sits beside two independent `ld_dispatch` samplings (2.20 %, 2.27 %) while the stall counter says 7.96 % | §2 and the whole answer |
| A2 | `ls_alloc_mab_count` is total outstanding-miss cycles | It is what study 44 divides by cycles to get MLP 5.77; my 5.955 reproduces that. Independent agreement with the IBS-derived total (22 354 vs 29 074, same order) | The absolute form of §3.1's bound; not the 18.5× share ratio |
| A3 | A `load_not_complete` cycle charged to a symbol must lie inside one of THAT symbol's miss intervals | It is the counter's definition (no retire + oldest op is an incomplete load) | §3.1 only — §3.2 and §3.3 are independent of it |
| A4 | IBS sampling is not biased against scalar symbols | A1's cross-check; and a bias would have to be ~18× to explain the gap | §2 |
| A5 | The landed arms are bit-identical | Verified three ways, not argued: node count at 5 M, d13, wl50 eval hash | A behaviour change wearing a speedup's clothes |
| A6 | ~120 Elo per doubling of speed | Inherited (exp-0154); its 95 % CI includes zero | Only the Elo translation |

---

## 8. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The exact split of `current_full`'s 140.4 c/node between "its own" and
   "absorbed".** §3.1 bounds its own at ≤ 96–126 c/node and the engine's overlap
   factor suggests ~8; §3.3 says 31.1 survives an upstream injection. **The
   honest statement is "somewhere between ~8 and ~31, not 140"**, and I did not
   narrow it further.
2. **§3.4's within-symbol join is inconclusive** and is reported as such. It was
   going to be the headline.
3. **`do_move` at 461 % and `see_ge` at 273 % are not explained**, only
   observed. They are worse violations than the symbol I was sent to study, and
   `do_move` has 260.3 i/node and 116.3 loads/node of real work that nobody has
   censused.
4. **No cycles claim is made for `col49`.** `cycles:u`'s null in this harness is
   −0.510 % at n = 15, which is larger than the arm could be.
5. **`forward_into`'s 17.5 % is priced but not attacked.** The one kernel-level
   idea I had (split the loop to halve the concurrent stream count) is
   arithmetically impossible — the halves are pairwise-multiplied. The available
   fix is channel fusion, which is a net-architecture decision.
6. **Nothing was arbitrated by games**, and nothing could be: every binary is
   node-identical, so a fixed-node match is a dead heat by construction.
7. **Study 48's §1 census table is not retracted** — its load column is right and
   its stall column is right as a *measurement*. Only the per-symbol *causal*
   reading of the stall column is wrong, and only for callers. I did not re-audit
   the earlier studies that used the same column.

---

## 9. REPRODUCTION

```bash
# 0. the worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. build, and gate before any timing
bash tools/perf49/build49.sh base49
cp bin49/base49 bin49/null49                      # the byte-identical null
./bin49/base49 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn          # must print 5001562

# 2. the frame
bash tools/perf49/census49.sh 3 base49

# 3. THE INSTRUMENT — IBS, precise rip + per-load DC-miss latency.
#    `swfilt=1` is what makes this work at perf_event_paranoid=2.
for i in 1 2 3 4 5 6; do
  taskset -c 2 perf record -q -e ibs_op/cnt_ctl=1,swfilt=1/ -c 8000 -d -W \
    -o ibs-$i.data -- ./bin49/base49 bench 16 1 100000 \
    tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn >/dev/null 2>&1
  perf script -i ibs-$i.data -F ip,sym,symoff,weight,data_src >> ibs-big.txt
done
python3 tools/perf49/ibs49.py < ibs-big.txt        # the per-symbol census

# 4. the three-instrument master table (§2)
bash tools/perf49/where49.sh ls_dispatch.ld_dispatch:u        200003 base49 ld.data
bash tools/perf49/where49.sh ex_no_retire.load_not_complete:u 200003 base49 lnc.data
python3 tools/perf49/master49.py ld.data lnc.data ibs-big.txt

# 5. the controlled experiment (§3.3)
bash tools/perf49/build49.sh skid49 skidprobe
bash tools/perf49/where49.sh ex_no_retire.load_not_complete:u 200003 skid49 lnc-skid.data
perf report -i lnc-skid.data --stdio -F overhead,symbol | head -20

# 6. the exact-counter panel (blocked Latin square)
P49_CPU=2 \
P49_EV='instructions:u,cycles:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u,ex_no_retire.load_not_complete:u' \
P49_PANEL=docs/research/perf-artifacts-49/panel-col.txt \
  bash tools/perf49/exact49.sh 15 base49 col49 null49
python3 tools/perf49/med49.py docs/research/perf-artifacts-49/panel-col.txt \
  base49 col49 null49
```

Pick a quiet **physical** core first — sum each SMT sibling pair (`N` and `N+8`
on this box) before choosing; core 2 was the quietest at 21.8 % combined.
**Do not park the IBS text on `/tmp`**: it is tmpfs, and a 250 MB script dump
vanished mid-study.
