# Perf study 29 — the load census, and the death of the fusion item

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1d,
1 MB L2/core, 32 MB L3 · **Net/arch:** D48X, `sota,arch-d48x`, ACC 1280 ·
**Base:** `main` at `d696685`, merged into this worktree · **Artifacts and
instruments:** `docs/research/perf-artifacts-29/`, `tools/perf29/`

**Commission:** study 28 attributed 82.8% of ARCHON's per-node deficit against
Stockfish to load stalls, and 1.32× the load volume at the same miss rate. It
named accumulator-channel fusion as the largest addressable item, bracketed it
at **−150 to −375 cycles/node**, and **refused to narrow it** because nobody
had split the fold kernels' loads into accumulator streaming versus weight
rows. Get that number, then act on it.

---

## ★ HEADLINE

### The census, exact

**Inside the accumulator layer, accumulator streaming is 44.2% of the loads
and weight rows are 55.8%. Excluding the head's merge, accumulator streaming
is 31.3% and weight rows are 68.7%.** In engine terms: 321.8 accumulator-stream
loads/node and 407.0 weight-row loads/node, out of 3033 total.

### The number that actually decides it — and it is not the census

Splitting the loads was the commissioned measurement and it is delivered, but
it turned out **not to be the deciding number**, and saying so is the main
result of this study. Two traffic-pinning probes — each holding the load
*count* fixed and collapsing one stream's working set to a single hot line —
give the split that pays:

| probe | loads/node | L1 misses/node | **load-stall cycles/node** |
|---|---:|---:|---:|
| control (`base`, median of 6) | 3033.0 | 633.3 | 5743 |
| **`wpin`** — every weight row becomes row 0 | 3060.8 (+0.9%) | 477.3 (**−24.6%**) | 3090 (**−46.2%**) |
| **`accpin`** — every accumulator read becomes one hot buffer | 2854.3 (−5.9%) | 586.2 (−7.4%) | 5651 (**−1.6%**) |
| null (byte-identical copy) | −0.07% | +0.34% | **+3.70%** |

**The accumulator stream is not what stalls this engine. The weight rows are.**
`accpin` deletes the *entire* locality cost of every accumulator read in the
engine — folds, the empty-delta copy, and the head's four-array merge — and
additionally loses 178 loads/node outright to compiler CSE, and it still moves
load-stall cycles by **less than the noise floor**. `wpin` moves them by
**46.2%**, twelve times the null.

### Therefore: fusion is dead as a stall lever

Fusing HalfKA and FullThreats into one accumulator (Stockfish's shape,
verified at `eca43a97:src/nnue/nnue_accumulator.h:47` — *"combined HalfKA +
Threats"*, one `std::array<i16, L1>` per colour) removes **162.8 loads/node**:
94.7 from the folds and 68.1 from `activate`. Weight-row traffic is untouched
by construction — the same rows are still fetched, merely in one pass.

162.8 load events is ~81 zmm load *instructions*/node. At our IPC that is
**~65 cycles/node, ~1.4%, ~+1.7 Elo** — a front-end/issue win, not a stall
win. **Study 28's −150 to −375 cycles/node bracket is high by 2.3× to 5.8×,
and the mechanism it named is refuted:** the 10.24 KB/ply of accumulator that
Stockfish carries as 4.10 KB is real, and it is nearly free, because it is
sequential, 64-byte aligned, and covered by the hardware stride prefetcher.
That prize does not pay for the refresh-trigger union (studies 22 and 26's
blocker) and it does not pay for the Finny-key problem. **Do not build it.**

### What landed

**Instruments, not a strength change.** `foldcount` (the exact census),
`probe-accpin` / `probe-wpin` (the two traffic probes that killed the fusion
item), `tools/perf29/`, and `gates29.sh` which clippies and tests all four
feature sets. All off by default; the shipped build is byte-untouched.

**No engine change landed, and §5 is the reason.** The one arm this study's own
measurement pointed at — prefetch the weight row when its index is born,
Stockfish's `append_changed_indices` shape — was built at three strengths,
gated node- and eval-hash-identical, and **measured as not resolvable**. It is
recorded as rejected, in the code comments at both sites, with the ceiling
argument that explains why. §5.3 also retracts a reading of my own data that
looked like a clean win for four rounds and was positional drift.

---

## 0. METHOD, AND WHAT EACH NUMBER IS

### 0.1 Frame

One frame throughout, quoted with every number:
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 50 FENs × 100 k nodes, **5,001,562 nodes**, one thread, 16 MB TT, D48X.
Builds are the **shipped release profile — fat LTO, CGU = 1** (`probe26`'s
rule; at 4 CGUs `instructions:u` is itself unreliable, study 23), non-PGO,
`-C target-cpu=native`. `tools/perf29/build29.sh` is the one build path.

### 0.2 Which numbers are exact and which are sampled

This matters more than usual here, so it is stated up front:

| number | status |
|---|---|
| kernel calls/node, weight rows/node, shape histograms | **EXACT** — integer counters in an instrumented build |
| loads per call (the static shape) | **EXACT** — from the source, confirmed against the disassembly (§2.2) |
| L1d load events per 512-bit / 256-bit load | **MEASURED**, 2.0015 / 1.0016 (§2.1) |
| engine-wide loads, misses, stall cycles, instructions | **EXACT TOTALS** — `perf stat`, whole run, no sampling |
| per-symbol shares | **SAMPLED** — `perf record`, used only to cross-check |
| the probe deltas (§3) | exact totals, **differenced across builds**, against a stated null |

### 0.3 Null first

Two nulls were run before any arm.

**Null A — the counting instrument must not change what it counts.** The
`foldcount` build against `base`: **node count identical (5,001,562) and eval
hash identical (`5ca978e012da1bb0`, the pinned prefix)**. It costs +19.5 i/n
and ~+5 loads/node, all of it the atomics themselves.

**Null B — a byte-identical copy of the base binary, interleaved into the same
rotated rounds.** On the traffic panel (§3): loads −0.07%, misses +0.34%,
**load-stall cycles +3.70%**, cycles ±8%. On the arm panel (§5), taken later
on a quieter box: cycles −0.20%, misses −0.12%, stall −0.81%.

**Read the consequence, because it is a gate and not a footnote:** with a
±3.7% floor on load-stall cycles, the `accpin` probe's −1.6% **is not
resolvable** and is reported as *no effect*, while `wpin`'s −46.2% is twelve
times the floor. And on the arm panel, a −0.66% miss delta against a −0.12%
null is the load-bearing evidence, not the −1.67% cycles delta against a
−0.20% null on an instrument whose per-round control spread is ±8%.

---

## 1. THE INSTRUMENT

`src/eval/nnue/foldcount.rs`, behind `--features foldcount`: one relaxed
atomic pair per accumulator-kernel call, recording calls and the number of
weight ROWS folded, plus shape histograms and the address ranges of every
weight table and accumulator stack. Counters sit in the dispatch wrappers in
`sota_simd.rs`, which is where the add/sub slice lengths are visible, plus the
threat fold's empty-delta copy and the head's `forward_into`.

The tally is dumped on **stderr** at the end of `bench`, so stdout — the
fingerprint — stays byte-identical to a normal build's.

This is the same move study 21 used to kill the event-cancellation lead before
anyone wrote code, and study 25 used to justify a shape dispatch. It cost one
build.

### 1.1 The raw tally

`docs/research/perf-artifacts-29/census-census-100000.txt`, 5,001,562 nodes:

| site | calls | weight rows | calls/node | rows/node | rows/call |
|---|---:|---:|---:|---:|---:|
| `apply_i16_deltas_into` — HalfKA parent→child fold | 4,875,878 | 11,091,677 | 0.9749 | 2.2176 | 2.275 |
| `apply_i16_deltas_dual` — Finny **warm** refresh | 805,331 | 4,567,372 | 0.1610 | 0.9132 | 5.671 |
| `apply_i16_deltas` — Finny **cold** refresh | 128 | 644 | 0.0000 | 0.0001 | 5.031 |
| `apply_i8_deltas_into` — FullThreats fold | 4,250,394 | 19,567,569 | 0.8498 | 3.9123 | 4.604 |
| threat empty-delta copy | 1,669,923 | — | 0.3339 | — | — |
| `activate` — the head's four-array merge | 2,129,416 | — | 0.4258 | — | — |
| evaluations | 2,129,416 | — | 0.4258 | — | — |

Three facts fall straight out and none was known before:

* **The HalfKA fold is one of three shapes in this bench** — 1a1s
  (3,535,957 = 72.5%), 1a2s (1,291,367 = 26.5%), 2a1s (48,554 = 1.0%) — and
  **2a2s occurs zero times in 4.88 M folds**. The 2a1s bucket is the *backward*
  fill of a capture: `apply_dirty_directed::<false>` swaps the two lists, so a
  1a2s forward fold is a 2a1s backward one, and 48,554 is the capture subset of
  study 12's 0.0532 backward folds/node.
  **DO NOT delete the 2a2s instantiation on this evidence.** I nearly wrote
  that recommendation. 2a2s is castling (`DirtyPiece::castling_record`,
  `board.rs:451` — sub king, sub rook, add king, add rook), which folds for the
  non-moving side while the mover refreshes. It is zero here because `wl50` is
  50 calibration positions whose trees rarely contain a castle, not because it
  is unreachable. A bench-frequency zero is not a proof of deadness.
* **The threat fold folds 4.604 rows per call against HalfKA's 2.275** — 2.02×
  — and its row count has a long tail: 803,916 folds of 1 row, but also 25,683
  of 15 and a maximum of 26.
* **The threat channel's empty-delta copy is alive and frequent**: 0.334 per
  node, 28% of all threat folds. Study 14's note asked for this rate to be
  measured before anyone touched that branch; here it is.

---

## 2. THE UNIT — AND THE 2× THAT WOULD HAVE WRECKED THE CENSUS

### 2.1 Zen 4 counts a 512-bit load TWICE

The census multiplies exact call counts by loads per call. The unit of "load"
has to match the counter, and on Zen 4 it does not match the instruction
count: AVX-512 executes on a 256-bit datapath and a 512-bit load is
**double-pumped into two load ops**.

Measured, not recalled — `tools/perf29`'s calibration binary issues exactly
1.0 × 10⁸ vector loads of each width over an L1-resident 4 KB buffer:

| load width | `L1-dcache-loads` per issued load |
|---|---:|
| 512-bit (`vmovdqu64`) | **2.0015** |
| 256-bit (`vmovdqu`) | **1.0016** |

So a 64-byte accumulator or HalfKA-row block costs **2** events and a 32-byte
FullThreats-row block costs **1**. Taking a zmm load as one event halves the
whole census and puts it 2× below the sampled per-symbol profile — which is
how the discrepancy was caught.

### 2.2 The static shape, confirmed against the disassembly

At ACC = 1280 the AVX-512 backend holds `REGS = 20` blocks of `LANES = 32`
lanes in registers and walks `TILES = 2` tiles, i.e. **40 blocks of 64 bytes
per call**. Per call each kernel therefore issues:

* **one** read of each accumulator block — 40 × 64 B = 2560 B = **80 events**;
* **one** read of each weight-row block per row — 80 events for an i16 HalfKA
  row, **40** for an i8 FullThreats row (32-byte `vmovdqu`, sign-extended);
* and `activate` reads **four** `[i16; ACC]` arrays — 10,240 B = **320
  events** — which is precisely the 10.24 KB/ply study 28 contrasted with
  Stockfish's 4.10.

Confirmed on `bin29/base`:

* **zero spills** — no `vmovdqu64 %zmm*,(%rsp)` anywhere in the fold symbol,
  so no accumulator block is ever re-loaded (21 live zmm out of 32);
* every store in the shaped fold targets `dst` (96 of 100 matched by base
  register `%rdi`, the rest the no-displacement form);
* LLVM folds most weight-row loads into the arithmetic as memory operands
  (`vpaddw (%rsi,%r9,1),%zmm6,%zmm6`) rather than emitting a separate
  `vmovdqu64`. That changes which mnemonic carries the load, not how many
  loads there are.

**Alignment is not a hidden factor.** The registered regions show every weight
table and both accumulator stacks based at `…040` (64-byte aligned), and the
stack strides are `size_of::<Entry>() = 5184` and
`size_of::<ThreatEntry>() = 5312` — both exact multiples of 64 — so every
accumulator array in every ply is line-aligned and no 64-byte load straddles
two lines.

### 2.3 The census

| site | calls/n | rows/n | **ACC STREAM** | **WEIGHT ROWS** | total |
|---|---:|---:|---:|---:|---:|
| HalfKA fold | 0.9749 | 2.2176 | 77.99 | 177.41 | 255.40 |
| Finny warm refresh | 0.1610 | 0.9132 | 12.88 | 73.06 | 85.94 |
| Finny cold refresh | ~0 | ~0 | 0.00 | 0.01 | 0.01 |
| FullThreats fold | 0.8498 | 3.9123 | 67.99 | 156.49 | 224.48 |
| threat empty-delta copy | 0.3339 | — | 26.71 | — | 26.71 |
| `activate` | 0.4258 | — | 136.24 | — | 136.24 |
| **TOTAL** | | | **321.81** | **406.97** | **728.78** |

* **accumulator streaming = 44.2%** of the accumulator layer's loads;
* **excluding `activate`, i.e. the fold kernels proper: 31.3%** (185.6 of
  592.5) — so **inside the fold kernels the answer to the commissioned
  question is that accumulator streaming is the MINORITY**, at slightly under
  a third;
* against 3033.0 engine loads/node: accumulator stream **10.6%**, weight rows
  **13.4%**, everything else 76.0%.

### 2.4 Cross-check against the sampled profile

Independent instrument, same run family (`perf record -e
ls_dispatch.ld_dispatch:u`, share × exact total):

| symbol | sampled loads/node | census model for what is inside it |
|---|---:|---|
| `avx512::i16_deltas_into` | 242 | 255.4 (HalfKA fold) — **−5%** |
| `finny_refresh_into` | 100 | 85.9 (dual fold) + the bitboard diff |
| `ThreatContext::at_full` | 498 | 224.5 (threat fold) + 26.7 (copy) + resolve/build/psqt |
| `SotaHead::forward_into` | 372 | 136.2 (`activate`) + `find_nnz` + `l1_dot` + tails |

The one directly comparable pair agrees to 5%, which is sampling error on
~6,100 samples. The census is not an estimate dressed up; it is an exact count
that a second instrument corroborates.

---

## 3. THE PROBES — WHICH STREAM ACTUALLY STALLS

The census answers "how many loads". It cannot answer "which loads hurt", and
after study 28 that is the only question that pays. Two probes settle it.

Both are compiled behind their own feature and **hold the load count and width
of every kernel fixed**, changing only where the loads point:

* **`probe-wpin`** — every fold folds weight row 0, whatever the feature index
  says. Same rows, same lengths, same dispatch shapes; the entire 57.7 MB
  HalfKA table and 77.7 MB threat table collapse to one hot row.
* **`probe-accpin`** — every accumulator read (fold `src`, the empty-delta
  copy's source, and `activate`'s four arrays) comes from a static 2.5 KB
  buffer instead of the per-ply stack.

They compute wrong evaluations on purpose. `bench … nodes` fixes the node
budget, so both arms search exactly 5,001,5xx nodes and the totals stay
comparable.

**`accpin` needed a second pass to stay honest.** The naive form handed the
same pointer to `activate`'s `pst` and `threat`, and LLVM CSE'd 80 of its 160
loads into 80; with a constant pointer it also CSE'd the src stream across the
HalfKA and threat folds. That version dropped **249 loads/node and 360
instructions/node** — a different experiment. Two distinct buffers plus
`black_box` on every handout cut the leak to 178 loads/node, which is stated
rather than hidden: it makes `accpin` an *upper* bound on the accumulator
stream's cost, which is the direction that matters here.

### 3.1 Result — 6 rotated rounds, medians

| arm | i/n | cycles/n | loads/n | misses/n | miss rate | **load-stall/n** |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 5654.2 | 8045 | 3033.0 | 632.2 | 20.87% | 5639 |
| `basecopy` (**null**) | 5654.2 | 8237 | 3030.8 | 634.4 | 20.90% | 5848 |
| `wpin` | 5849.8 | 5515 | 3060.8 | **477.3** | **15.55%** | **3090** |
| `accpin` | 5268.7 | 7898 | 2854.3 | 586.2 | 20.53% | 5651 |

Against the mean of the two controls:

| | Δ loads | Δ misses | Δ **load-stall cycles** | vs the null (+3.70%) |
|---|---:|---:|---:|---|
| **`wpin`** | +0.9% | **−24.6%** (−156.0/node) | **−46.2%** (−2653/node) | **12.5× the null** |
| **`accpin`** | −5.9% | −7.4% (−47.1/node) | −1.6% (−92/node) | **inside the null** |

### 3.2 What the two probes say together

Divide the census by the probes:

| stream | loads/node | misses/node | **miss rate** | stall cycles per miss |
|---|---:|---:|---:|---:|
| weight rows | 407.0 | 156.0 | **38.3%** | **17.0** |
| accumulator stream | 321.8 | 47.1 | **14.6%** | ≲ 2 (unresolvable) |
| engine average | 3033.0 | 633 | 20.9% | — |

The weight rows are a random 2560-byte (HalfKA) or 1280-byte (threats) column
out of a 57.7 MB / 77.7 MB table. The accumulator stream is a sequential,
line-aligned 2560-byte walk over a 1.3 MB stack. **They miss at 2.6× different
rates and each miss costs an order of magnitude more.**

The IBS data-source profile agrees independently: of `finny_refresh_into`'s
loads, 81.2% are HalfKA weight rows and only **7.2% of those hit L1** (the
rest L2/L3/DRAM); of `at_full`'s, 54.2% are threat weight rows at 43.4% L1
hit, while the 12.3% that touch the threat stack behave like ordinary warm
data.

And the stall profile makes the same point about `activate` from the other
side: `forward_into` is **12.25% of the engine's loads but only 2.18% of its
load-stall cycles**. The single biggest accumulator-stream reader in the
engine — the 10.24 KB/ply merge that is the entire visible symptom of not
fusing — is already almost completely overlapped.

---

## 4. WHAT THIS DOES TO STUDY 28's ITEM #1

Stockfish does fuse, and at our pinned revision — verified directly, not from
a checkout:

```
eca43a97:src/nnue/nnue_accumulator.h
    // Class that holds the result of affine transformation of input features,
    // combined HalfKA + Threats
    struct alignas(CacheLineSize) Accumulator {
        std::array<std::array<i16, L1>, COLOR_NB>  accumulation;
```

One `i16[1024]` per colour carrying both feature sets, and the Finny cache
entry (`AccumulatorCaches::Entry`) holds that same fused array. So the design
is possible; the question was always what it buys us.

**It buys 162.8 loads/node and no stall relief.** At the same width 1280:

| | now | fused | saved |
|---|---:|---:|---:|
| fold accumulator stream | 185.6 | 90.9 | 94.7 |
| `activate` | 136.2 | 68.1 | 68.1 |
| weight rows | 407.0 | 407.0 | **0** |
| **total** | | | **162.8 loads/node = 5.37% of engine loads** |

162.8 load events is ~81 zmm load instructions/node. Removing 81 instructions
at IPC ≈ 1.25 is **~65 cycles/node (~1.4%, ~+1.7 Elo)** — and by §3 essentially
none of it comes back as recovered stall time, because the stream being halved
carries ≲ 1.6% of load-stall cycles in total.

**Against that, the costs are on the expensive side of the ledger:**

* the refresh-trigger union — HalfKA refreshes at **0.1610/node** (measured
  here, the `hka_dual` call rate) against the threat channel's 0.0214, so
  fusion forces **+0.139 threat refreshes/node**, and a threat refresh folds
  rows out of the 77.7 MB table, i.e. the *most* expensive stream in the
  engine at 17.0 stall cycles per miss;
* study 22 priced the same union at ≤157 i/n saved against +872 i/n cost, from
  the other direction;
* study 26 hit the Finny-key wall independently.

**Verdict: do not build it.** Study 28's bracket was 2.3–5.8× high, and it was
high because it priced a load-VOLUME reduction at the engine's average
stall-per-load when the volume being removed is the engine's *cheapest* load
class. That is the same error class as pricing a change in instructions when
instructions are not the binding constraint — study 28's own central finding,
applied one level down.

*(Width is the same story from a different angle: narrowing 1280 → 1024 shrinks
the accumulator stream by 20% and the weight rows by 20% as well, so unlike
fusion it does touch the expensive stream. It still needs a retrain and the
standing finding is that width is Elo-neutral in 768–1536.)*

---

## 5. THE ARM THIS STUDY BUILT AND REJECTED — THE WEIGHT-ROW PREFETCH

Study 19 ranked this #1 and could not price it; it had been rejected once on a
study-4-era NPS instrument at "−2.5% NPS full-row, ~−0.4% first-two-lines",
which study 19 correctly refused to treat as settled because −0.4% was inside
that instrument's noise. §3 makes it the only item aimed at the stream that
actually stalls.

**Stockfish's shape, exactly** — one hint per index, first cache line only,
`PrefetchLoc::LOW`:

```
eca43a97:src/nnue/features/full_threats.cpp:295-299
    const IndexType index = make_index(perspective, attacker, from, to, attacked, ksq);
    if (prefetchBase)
        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(... prefetchBase + index * prefetchStride);
```

`PrefetchLoc::LOW` is `_MM_HINT_T2` (`eca43a97:src/misc.h:112-113`). **First
line only is the whole point:** a row is 40 cache lines at ACC = 1280, and
hinting all of them is 40 instructions per row — which is what the study-4-era
"full-row" attempt measured at −2.5%. One hint starts the TLB walk and the
first fill; the fold's own sequential walk down the row is then picked up by
the hardware stride prefetcher.

Two sites, because two sites have a real gap between "the index is known" and
"the row is read":

* **`pf-finny`** — `HalfkaContext::finny_refresh_into`'s bitboard diff loop
  resolves up to 32 indices before `apply_i16_deltas_dual` folds any of them.
  13.74% of the engine's load-stall cycles; 81.2% of its loads are HalfKA rows
  at 7.2% L1 hit.
* **`pf-threat`** — `resolve_both`, the phase-1 half of the resolve/fold split.
  This is literally SF's `append_changed_indices`. `at_full` is 22.78% of load-
  stall cycles.

The HalfKA *incremental* fold gets nothing: `apply_dirty_directed` computes its
indices and calls the kernel on the next line, so there is no latency to hide.

### 5.1 Result — 24 interleaved rotated rounds, medians

`bin29/pfnone` is the arms-off build **of the same source** — the honest
control, since the arm added a (dead, LTO-eliminated) parameter to
`resolve_both`. `base` is `main`, i.e. a second, byte-different null.

| arm | i/n | cycles/n | loads/n | miss/n | miss rate | load-stall/n |
|---|---:|---:|---:|---:|---:|---:|
| `pfnone` (control) | 5656.59 | 4593.93 | 3050.62 | 614.09 | 20.145% | 2595.48 |
| `base` (**null**) | −0.042% | −0.374% | −0.052% | **+0.135%** | +0.122% | **−0.696%** |
| `pf-finny` | +0.024% | +0.348% | −0.202% | −0.159% | −0.072% | +1.096% |
| `pf-threat` | +0.197% | −0.971% | −0.038% | −0.498% | −0.603% | −0.854% |
| both | +0.220% | −0.006% | −0.209% | −0.312% | −0.283% | −0.023% |
| `pf-full` (whole row, n=6) | **+2.046%** | — | −0.347% | −0.688% | −0.344% | — |

Control spread over the 24 rounds: **cycles ±8.3%, load-stall ±12.5%**, misses
±1.6%, instructions ±0.000%.

**Identity:** every arm reproduces **5,001,562 nodes** and eval hash
**`5ca978e012da1bb0`**.

### 5.2 Verdict: NOT RESOLVABLE, and the mechanism says why

Nothing here clears its null by enough to keep. `pf-finny` is *positive* on
load-stall cycles. `pf-threat`'s −0.50% on misses against a +0.14% null is the
best signal in the table and it does not carry into cycles by more than the
control's own −0.37% wander. And every arm costs instructions: +11 to +12.5
i/n for the first-line form, **+116 i/n for the whole-row form**, against a
0.000 i/n null.

**The ceiling explains it, and it was computable in advance from §3.** `wpin`
removes 156.0 misses/node across 7.04 weight rows/node — **22 demand misses
per row.** A row is 40 lines (HalfKA) or 20 (threats), so the hardware stride
prefetcher is *not* covering these walks: each row is a cold single-visit
burst, and the streamer never trains because the next row is somewhere else in
a 57.7 / 77.7 MB table. One software hint can therefore convert at most **1 of
22**. Seven hints per node against a 156-miss target is a 4.5% ceiling; we
measured ~3 misses/node, i.e. the arm worked about as well as its mechanism
permits, and its mechanism is capped an order of magnitude below the prize.

The whole-row form removes the cap and still buys only −0.69% of misses,
because `_MM_HINT_T2` (SF's `PrefetchLoc::LOW`) lands the line in **L2, not
L1** — so a prefetched row still counts as an L1 miss on the demand load. The
IBS data says the weight rows *already* mostly hit L2 (finny's HalfKA rows:
7.2% L1, 72.7% L2), so T2 has little left to add. Hinting to L1 (`T0`) instead
would evict the accumulator for 2.5 KB of single-use weights — which is
presumably why Stockfish chose LOW.

### 5.3 RETRACTION — I read a win off this table and it was drift

At n = 13 rounds the medians were monotone and beautiful: `pfboth` −1.67%
cycles, −2.48% load-stall, −0.66% misses, with `pft` > `pff` in every column.
I had written that into the headline of this document as the landed arm.

It did not survive n = 24, and the **paired within-round form** — delta taken
against the control *inside the same round*, which cancels monotone drift —
kills it outright:

| arm, paired vs `pfnone`, n=16 | Δ cycles | Δ misses | Δ load-stall | rounds negative |
|---|---:|---:|---:|---|
| `base` (**null**) | **+1.518%** | +0.360% | **+1.643%** | 7/16, 6/16 |
| `pf-finny` | +1.078% | +0.204% | +2.089% | 7/16, 5/16 |
| `pf-threat` | +1.110% | −0.123% | +1.102% | 7/16, 9/16 |
| both | +1.113% | −0.204% | +1.149% | 5/16, 11/16 |

**The byte-different null moves +1.5% in the same form the arms move +1.1%.**
Whichever binary runs second in a round eats the ramp; rotating the order
spreads that across arms but does not remove it, and the sign counts are coin
flips. Rule 6 — *when the story gets beautiful, attack it hardest* — earned its
keep here: the beautiful table was measuring position-in-round.

**What I would have shipped if I had stopped at 13 rounds:** a +12.5 i/n
regression sold as a −1.7% cycles win.

---

## 6. WHAT DID NOT WORK, AND WHAT IT COST

* **IBS with data addresses was the right instrument and I could not fully
  land it.** `perf mem record` reports only `mem-ldst` on this kernel;
  `perf record -e ibs_op/swfilt=1/ -d` works and gives data address + data
  source, but `perf script` resolves symbols to `[unknown]` where `perf
  report` resolves them fine, and hand-mapping IP → disassembly resolved only
  21% of load samples. The partial data (§3.2) is quoted only where it
  corroborates the probes, never on its own. `tools/perf29/ibs29.sh` +
  `ibssplit29.py` are left working to that degree for whoever wants to finish
  it.
* **The first `accpin` was not load-count-neutral** and would have overstated
  the accumulator stream's cost by attributing 249 deleted loads/node to
  locality. Caught by watching `loads/n`, which is why that column is in the
  panel at all.
* **`perf annotate` on `L1-dcache-load-misses` was abandoned** — the event is
  not precise on Zen 4, so per-instruction attribution skids, and the
  accumulator-read and weight-row instructions sit adjacent inside the tile
  body.

---

## 7. RANKED REMAINDER

**The whole ranking now points at ONE thing: the weight rows carry 46.2% of the
engine's load-stall cycles, they miss at 38.3%, and each miss costs 17.0 stall
cycles. Nothing else on the board is within an order of magnitude.** §5 shows
prefetching cannot get at it. So the items below are about *reducing or
relocating the rows themselves*.

| # | item | prize | evidence | tractability |
|---|---|---|---|---|
| 1 | **Fewer threat rows.** The threat fold is **3.912 rows/node** — 55% of all weight rows and the single biggest consumer at 156.5 loads/node — against HalfKA's 3.13. It folds 4.604 rows per call where HalfKA folds 2.275. Every row not emitted is a random column out of 77.7 MB not fetched. Study 19's g2/g3/g4 attacked the *cost per event*; nobody has attacked the *count* | up to ~1500 stall cycles/node at the measured 17.0/miss, if the count can be halved | §1.1, §3.2 | medium — it is a feature-set question, and it interacts with net quality |
| 2 | **Re-price the per-node `ThreatPosition` snapshot on cycles** (study 28 item #2, untouched here). 80 bytes at every node into a 10.5 KB-stride stack; 100% of nodes paying for something 43% use. Rejected once at "+106 i/n", and §5 is a fresh demonstration that instructions are the wrong currency | `search::<NonPv>` is 7.95% of load-stall cycles | study 28 §9 | medium — the arm existed once |
| 3 | **The threat empty-delta copy, 0.334/node.** 28% of threat folds are a pure 2560-byte parent copy. Study 14 measured the two spellings 45.4 i/n apart and asked for the surviving rate before anyone touched it — **the rate is now measured**. A fold-state "same as parent" alias would remove 26.7 loads/node and the matching stores | 26.7 loads/node + 0.334 × 2560 B of stores | §1.1, §2.3 | medium — touches the fold-state machine, needs the full NNUE gate |
| 4 | **Weight-row layout.** 22 of a row's 40 lines take a demand miss because each row is a single-visit cold burst the stride prefetcher never trains on. Interleaving the rows a fold uses together, or splitting the table so a fold's rows share pages, attacks the miss *rate* rather than the miss count | the same 46.2% pool | §5.2 | low — a loader + trainer format change |
| 5 | **Accumulator-channel fusion** | ~65 cycles/node of issue work, ~+1.7 Elo, against a refresh-union cost on the expensive stream | §4 | **DO NOT BUILD** — this study's main result |
| 6 | Prefetching the weight rows | measured not resolvable at three strengths; ceiling is 1 miss of 22 per row, and `T2` lands in L2 where they already mostly hit | §5 | **closed** |
| 7 | `activate`'s four-array merge as a standalone target | 136.2 loads/node, but `forward_into` is 12.25% of loads and only 2.18% of stalls | §3.2 | already overlapped; leave it |
| 8 | **Do NOT delete the 2a2s fold instantiation** despite its zero count | it is castling; `wl50` simply has none | §1.1 | recorded so the next study does not "optimise" it |

---

## 8. HANDOVER

1. **The unit is not the instruction.** On Zen 4 a `vmovdqu64` is two
   `L1-dcache-loads`. Anything comparing a hand-counted SIMD kernel against
   that counter must double the 512-bit loads. Measured in
   `tools/perf29`; the calibration takes ten seconds and it caught a 2× error
   in this study's own first draft.
2. **Load volume is not load cost.** Two streams in the same three symbols
   differ by 2.6× in miss rate and ~10× in stall cycles per miss. Ranking a
   memory optimisation by how many loads it removes is the same mistake as
   ranking by instructions, one level down — and study 28 made it while
   correctly diagnosing the level above.
3. **A traffic-pinning probe is cheap and decisive.** Hold the load count
   fixed, collapse one stream's working set, and read the difference off exact
   hardware totals. Two builds and no sampling. Watch `loads/n` to prove the
   probe stayed count-neutral — the first `accpin` did not, and would have
   inverted this study's conclusion in the cautious direction.
4. **`ex_no_retire.load_not_complete` is noisy on a shared box** — ±12.5%
   round to round here. It still resolved a −46% effect without difficulty.
   Pick effects that clear the floor by 10×, or change instrument.
5. **Rotating the arm order is not enough; take the delta INSIDE the round.**
   The rotated median said `pfboth` was −1.7% cycles at n = 13. The paired
   within-round delta said a byte-different null moved +1.5% in the same
   direction. §5.3. The panel script now prints the control's spread next to
   every median for exactly this reason.
6. **Two nulls, not one.** `base` (a different binary from a different source
   with the same behaviour) and `basecopy` (byte-identical) answer different
   questions, and this study needed both: the byte-identical one bounds the
   instrument, the different-binary one bounds *code layout*, which on a
   1.2 MB binary is its own effect.

### 8.1 What this study did NOT do

* **No strength change landed**, and §5 is the whole reason. The one arm the
  measurement pointed at was built at three strengths and rejected on its own
  data.
* **No games were run.** Nothing here reached a state worth arbitrating.
* **The IBS symbolisation is half-finished** (§6). Its numbers are quoted only
  as corroboration of the probes, never alone.
* **The threat-refresh row count was not measured**, so the refresh-union cost
  in §4 is quoted from studies 22 and 28 rather than re-derived. It is not
  load-bearing: the *benefit* side of fusion is already ≈ 0 in stall cycles
  before any cost is counted, which is what decides it.
