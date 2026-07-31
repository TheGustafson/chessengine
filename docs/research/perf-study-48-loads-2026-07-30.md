# Perf study 48 — the loads, censused by site and by purpose

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `f9b09bf5` (studies
43/44/45/46/47 merged) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Instruments:** `tools/perf48/` · **Artifacts:**
`docs/research/perf-artifacts-48/` · **Branch:** `perf-study-48-loads`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO. Percentages
against the settled quiet-box **4179.7 cycles/node**; my own reads were
4081.5 – 4101.3.

**`base48`'s `.text` md5 is `f411ea70`** — byte-identical to study 47's
`base47`, which is itself `cmp`-identical to study 46's `base46`. This study's
reference is the same code three studies measured, established before anything
was built on it.

---

## ★ WHAT LANDED, AND WHAT FRACTION OF THE STALL IT REMOVES

**`activate`'s reused operand now lives in a register instead of memory.**
`sota_simd.rs`, AVX-512 `activate_inner`: the lane loop moved outside the
perspective loop and the `scale` array — the one of the merge's four arrays
that does **not** depend on the perspective — is hoisted into two `zmm`, with
the per-perspective accumulator load folded into the multiply in its place.

| per 32-lane group (4 merges) | ops | 512-bit loads |
|---|---:|---:|
| before: 4 × (`vmovdqu64` pst + `vpsllw`[threat] + `vpmullw`[scale]) | 12 | 12 |
| **after: 2 × `vmovdqu64` scale + 4 × (`vpmullw`[pst] + `vpsllw`[threat])** | **10** | **10** |

**Measured — 15 paired rounds, blocked Latin square, byte-identical null riding
in the rotation:**

| `act48` vs `base48` | median | trimmed | se | neg/n | p | per node |
|---|---:|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.286 %** | −0.286 | **0.000** | **15/15** | 0.0001 | **−17.65** |
| **`L1-dcache-loads:u`** | **−0.950 %** | −0.949 | 0.117 | **15/15** | 0.0001 | **−28.9** |
| `ls_dispatch.ld_dispatch:u` | −0.608 % | −0.513 | 0.345 | 12/15 | 0.0352 | −13.9 |
| `ex_no_retire.load_not_complete` | −1.841 % | −0.166 | 1.158 | 10/15 | 0.30 | *unresolvable* |
| `cycles:u` | −1.635 % | −0.352 | 1.074 | 9/15 | 0.61 | *unresolvable* |

**The null, published as required:** `instructions:u` **0.000 %** (exact, every
round), `L1-dcache-loads` **+0.038 %** (6/15), `ld_dispatch` **+0.138 %**
(5/15) — and `cycles:u` **−1.089 %**, which is why no cycles claim is made.

**The static model predicted −17.2 instructions/node and the measurement is
−17.65 — it closes to 2.6 %.** Priced at study 47's *measured* 0.586 – 0.735
cycles/op for this exact kernel, the **40 removed 512-bit memory ops per
evaluation** are **−10.1 … −12.7 cycles/node = −0.24 % … −0.30 % of the 4179.7
frame** (≈ +0.3 … +0.4 Elo at 120 Elo/doubling).

### ★ And the honest answer to the commissioned question: it removes ~0.24 % of the stall, not more

**The commission asked what fraction of the 2555 stall cycles/node the arm
removes. It is about 5 cycles — 0.24 %.** The loads this arm deletes are
`forward_into`'s, and §2 measures those at **0.29 stall cycles per load against
an engine average of 0.906**. They are the *cheapest* loads in the engine.

**That is not a disappointing footnote; it is the study's central result**, and
§2 is the reason: **the "2281 dispatches × 1.12 stall cycles each" model in the
commission is an average over a population that varies 12.5× in cost.** The
lever is not "fewer loads". It is "fewer loads *that miss*", and those are
somewhere else entirely.

### Everything else that landed

* **`tools/perf48/`** — seven tools: `census48.sh`/`reduce48.py` (the load
  census, within-run shares), `where48.sh` (per-symbol attribution by sampling
  the load counter itself), **`static48.py` (the first STATIC per-symbol load +
  spill census this repo has had)**, `exact48.sh`/`med48.py` (blocked Latin
  square on the exact counters), `build48.sh`, `ident48.sh`.
* **`foldcount` `THR_CANCEL`** — the redundancy census of the resolved threat
  index lists. **NULL RESULT (§3).**
* **`nopsqt48`** — a subtractive diagnostic probe that drops the PSQ1 trailer at
  load, pricing the whole psqt channel (§4).

---

## ★ 0. THE FRAME, RE-DERIVED — AND THE COMMISSION'S HEADLINE NUMBER IS SUPERSEDED

Rule 8. The first thing measured here is study 44's own denominator, on this
study's binary: `tools/perf48/census48.sh`, 5 unmultiplexed Zen 4 groups × 4
reps, every group carrying `cycles:u` **and** `instructions:u` so shares are
computed *within* a run (study 44 §0.3).

| | study 44 (`base44`) | **this study (`base48`)** | Δ |
|---|---:|---:|---:|
| `instructions:u` | 5929.09 | **6170.77** | +4.08 % |
| `ls_dispatch.ld_dispatch:u` | 2280.76 | **2297.01** | +0.71 % |
| `ls_dispatch.store_dispatch:u` | 907.31 | **897.55** | −1.08 % |
| memory ops as a share of instructions | 53.8 % | **51.77 %** | |
| `ls_any_fills_from_sys.all` | 585.86 | **576.20** | −1.65 % |
| DRAM fills | 7.90 | **7.27** | |
| MLP (`ls_alloc_mab_count` / cycles) | 5.74 | **5.768** | +0.5 % |
| L1-miss residency | 53.99 c | 48.83 c | |
| **`ex_no_retire.load_not_complete`** | **2558.77 (54.80 %)** | **2081.43 (50.00 %)** | **−18.7 %** |
| stall cycles per load dispatch | 1.122 | **0.906** | −19.3 % |

**The load frame reproduces to 0.71 %. The stall frame does not, and the reason
is a change study 44 itself landed.**

`5929.09 × 1.04076 = 6170.7` — and **+4.076 % is exactly what study 44 §4.5
measured `REGS_CAP = 5` costing on `instructions:u`.** So study 44's published
denominator was taken on the **pre-`REGS_CAP`** binary, i.e. on the tile that
study's own arm replaced. The engine has been the post-landing shape ever since.

**Consequence, stated plainly: the commission's target — "`load_not_complete` at
2558.77 cycles/node = 54.80 % of every cycle" — is not the number the shipping
engine carries. It is 2081.4 cycles/node = 50.00 %.** The 4.8-point drop is
partly `r5` (study 44 measured −1.937 % on this counter) and partly denominator:
study 44's census ran on a loaded box at 4668.94 cycles/node against my
4081.5 – 4101.3. Everything below is priced against **2081.4**, and every claim
of "fraction of the stall removed" uses that.

**What does NOT change:** memory is still the answer. `load_not_complete` is
still **half of every cycle the engine spends**, still 90 %+ L2-sourced
(DRAM 7.27 fills/node = 1.26 % of fills), still 5.8-way overlapped. Study 44's
shape survives; only its scalar moved.

---

## ★ 1. THE LOAD CENSUS, BY SITE

`tools/perf48/where48.sh` samples `ls_dispatch.ld_dispatch` **itself** (period
200003), so a symbol's share reads as "this code issues N % of the engine's load
dispatches". The companion column samples `ex_no_retire.load_not_complete` on
the same binary. Per study 36 §2.3 per-*instruction* attribution of these
counters is an artifact on Zen 4; per-**symbol** shares are what this asks.

**Nobody had this table.** Study 29's per-symbol census is in a different
counter frame (`L1-dcache-loads`, 3033/node) on a binary three landed changes
ago.

| symbol | % loads | **loads/node** | % stall | **stall c/node** | **stall per load** |
|---|---:|---:|---:|---:|---:|
| `SotaHead::forward_into` | 13.29 | **305.3** | 4.23 | 88.0 | **0.29** |
| `search::<NonPv>` | 11.75 | **269.9** | 11.52 | 239.8 | 0.89 |
| `avx512::i8_deltas_into_generic` (threat fold) | 9.33 | 214.3 | 11.92 | 248.1 | 1.16 |
| `avx512::i8_deltas_into` (HalfKA fold) | 8.24 | 189.3 | 12.85 | 267.5 | **1.41** |
| `MovePicker::init_quiets` | 8.20 | 188.4 | 6.75 | 140.5 | 0.75 |
| `ThreatContext::at_full` | 6.08 | 139.7 | 6.09 | 126.8 | 0.91 |
| `Worker::do_move` | 4.88 | 112.1 | 3.91 | 81.4 | 0.73 |
| `HalfkaContext::finny_refresh_into` | 3.84 | 88.2 | 8.31 | 173.0 | **1.96** |
| `Board::make_move_with::<TtPrefetch>` | 3.72 | 85.4 | 2.05 | 42.7 | 0.50 |
| **`HalfkaContext::current_full`** | **2.27** | **52.1** | **7.61** | **158.4** | **3.04** |
| `Board::refresh_check_info` | 2.17 | 49.8 | 1.46 | 30.4 | 0.61 |
| `SotaContext::evaluate` | 1.89 | 43.4 | 2.02 | 42.0 | 0.97 |
| `Board::see_ge` | 1.69 | 38.8 | 1.21 | 25.2 | 0.65 |
| `record_square_touch` | 1.58 | 36.3 | 1.60 | 33.3 | 0.92 |
| `Board::attackers_to` | 1.51 | 34.7 | 1.08 | 22.5 | 0.65 |
| `ThreatContext::apply_at` | 1.36 | 31.2 | 0.92 | 19.1 | 0.61 |
| — engine | 100 | 2297.0 | 100 | 2081.4 | **0.906** |

**⚠ Three rows in the raw profile are STARTUP, not per-node, and are excluded
above:** `quantise_halfka` 1.44 %, `load_v2_bytes` 1.19 %, `init_magics` 0.35 %
— **2.98 % of the profile is loading the 138 MB net once.** A per-node reading
of the raw `perf report` over-counts by that much. This has not been flagged in
any prior study's per-symbol table.

### ★ 1.1 THE FINDING — the stall density spans 12.5×, so "fewer loads" is the wrong instruction

**`forward_into` issues 305.3 loads/node and pays 88.0 stall cycles.
`current_full` issues 52.1 and pays 158.4.** One-sixth the loads, nearly twice
the stall. The ratio between the extremes is **3.04 / 0.29 = 10.5×**, and 12.5×
against the whole table.

The commission's arithmetic — 2281 dispatches × 1.12 stall cycles = 2555 — is
true as an average and **misleading as a lever**, because deleting a load buys
between 0.29 and 3.04 cycles depending entirely on *which* load. Ranked by what
a removed load is actually worth:

| class | loads/node | stall c/node | c/load | what they are |
|---|---:|---:|---:|---|
| **weight rows** (2 folds + finny) | 491.8 | **688.6** | **1.40** | 7.1754 rows/node × 40 × 32 B out of a 28.8 MB + 12.6 MB table |
| scalar search / board / movepick | ~750 | ~505 | ~0.67 | TT, history, bitboards, spill slots |
| **accumulator stream + `activate`** | ~367 | ~106 | **~0.29** | just-written 2560-byte arrays, L1-hot |

**33.1 % of every stall cycle in the engine sits on 21.4 % of its loads, and
they are all the same thing: weight rows.** Study 43's handover said this in
different words — *"49 % of its lines … are HalfKA weight rows … the same table,
the same problem, and the same fix as `i8_deltas_into` and
`i8_deltas_into_generic` … one problem wearing three symbol names."* The
density table is the independent confirmation, from a different counter.

**And the weight-row count is not a perf parameter.** It is 7.1754 folded rows
per node, fixed by the feature set: `hka_into` 2.2295 + `hka_dual` 0.94218 +
`thr_into` 3.92216 + `thr_refresh` 0.08145 (exact `foldcount` integers, §3).
Each row is 1280 i8 = 20 cache lines, read as 40 × 32-byte `vpmovsxbw` —
**287.0 load dispatches/node, and study 47 §5.1 already proved that count is a
hard architectural floor at this data layout** (a 64-byte load plus two register
converts is 3 ops for 2, and double-pumping makes the *dispatch* count
identical either way).

**So the honest statement of where the prize is: it is in folding fewer ROWS,
which is a feature-set decision, not a kernel one.** §7 records what that would
be worth.

### 1.2 The census by PURPOSE — exact shape × exact call counts

Study 47's method (static disassembly shape × exact `foldcount` integers), no
sampling:

| block | loads/node | share | derivation |
|---|---:|---:|---|
| weight rows | **287.0** | 12.5 % | 7.1754 rows × 40 × 32 B |
| fold accumulator stream (read parent) | **159.8** | 7.0 % | (0.9784 + 0.8538 + 0.16472) calls × 40 zmm × 2 |
| `activate` (4-array merge) | **206.9** | 9.0 % | 0.43115 evals × 240 zmm × 2 |
| **modelled NNUE data subtotal** | **653.7** | **28.5 %** | |
| measured NNUE *symbols* total | 1146.9 | 49.9 % | §1's NNUE rows summed |
| scalar search / board / movepick / history | ~1150 | ~50 % | the remainder |

Zen 4 double-pumps a 512-bit load into **2.0015** `L1-dcache-loads` (study 29
§2.1, measured); the 32-byte `vpmovsxbw` weight-row load is **1**. Both are used
above and neither is double-counted.

**Two things this table says that nothing before it did.** First, the NNUE data
streams that everyone models are **28.5 %** of the engine's loads — the NNUE
*symbols* are 49.9 %, so **~490 loads/node inside NNUE symbols are not weight
rows, not accumulator, and not `activate`**: they are index resolution, ancestor
walks, and the threat event machinery. Second, **half the engine's loads are
outside NNUE altogether and have never been censused per-site by anybody.** They
are also diffuse: the largest single non-NNUE symbol is 11.75 %.

### 1.3 The static census — and a trap it sets

`tools/perf48/static48.py` counts, per symbol, instructions carrying a memory
*source* operand, split by width and by base register. Only `%rsp` is reliably
the stack (release builds have no frame pointer, so `%rbp` is a data register).

| symbol | insns | loads | **`%rsp` loads** | stores | `%rsp` stores |
|---|---:|---:|---:|---:|---:|
| `search::<NonPv>` | 3390 | 713 | **458 (64.2 %)** | 388 | 295 |
| `search::<Pv>` | 3332 | 749 | **494 (66.0 %)** | 352 | 267 |
| `finny_refresh_into` | 1824 | 543 | 51 | 199 | — |
| `at_full` | 1636 | 392 | **190 (48.5 %)** | 222 | 133 |
| `init_quiets` | 1136 | 244 | 92 (37.7 %) | 144 | 110 |
| `current_full` | 541 | 106 | **59 (55.7 %)** | 64 | 51 |
| `do_move` | 217 | 36 | 14 | 49 | 16 |

Study 29 checked the SIMD kernels for spills, found **zero**, and concluded
there is no spill lever. **It never looked at the scalar symbols, and they are
two-thirds stack traffic.** §5 is what happened when I aimed at that.

---

## 2. THE THREE SITES THE COMMISSION NAMED, PRICED

| site | loads/node | stall c/node | c/load | verdict |
|---|---:|---:|---:|---|
| `ThreatContext::at_full` | 139.7 | 126.8 | 0.91 | **exactly average.** Not a load problem; it is a lot of ordinary loads |
| `HalfkaContext::current_full` | 52.1 | 158.4 | **3.04** | **the densest site in the engine** — see below |
| `finny_refresh_into` | 88.2 | 173.0 | 1.96 | weight rows, as study 43 said |

Study 47's puzzle — *"`at_full` and `current_full` together carry 11.35 % of
engine stalls while containing 21 and ZERO 512-bit ops"* — resolves differently
for the two. **`at_full` is unremarkable**: it issues 6.08 % of loads for 6.09 %
of stalls. **`current_full` is the anomaly**: 2.27 % of loads for 7.61 % of
stalls.

**⚠ And I could not close `current_full`, which is the study's largest open
item.** Two candidate mechanisms were tested and both failed:

1. **Spills** (55.7 % of its static loads are `%rsp`-relative, and it carries a
   264-byte frame spilling ~20 values across the `finny_refresh_into` call).
   **Arm built, LOST — §5.**
2. **The 2.7 MB PSQ1 table**, read randomly once per folded row. **Probe built,
   priced at 18.7 loads/node for the WHOLE channel — far too small. §4.**

A third candidate is *stated and not tested*: `ex_no_retire.load_not_complete`
is not a precise event, and `current_full` is the immediate **caller** of
`i8_deltas_into`. Sample skid at the call boundary would attribute the kernel's
stalls to its caller, and `current_full` is exactly the symbol shape where that
does most damage. **Study 44's assumption A5 (symbol attribution is sound even
though instruction attribution is not) has never been validated for a caller
adjacent to a stall-heavy callee**, and this is the case that would expose it.

---

## 3. NULL — the threat delta lists are already non-redundant

**The hypothesis.** `resolve_both` pushes each event's resolved feature index
into an add or a sub list with no de-duplication (`sota_threats.rs:1843-1921`).
An index appearing in **both** lists contributes `+row` and `−row` to the same
lanes, which under wrapping `i16` is **exactly zero** — so both rows are pure
weight-table traffic that provably changes nothing, and dropping them would be
bit-identical. An index appearing **twice in one list** is a row read twice
(`x + w + w ≡ x + 2w`, also exact). Given that §1.1 makes weight rows the whole
prize, this was the first thing to check.

**The census** (`foldcount::THR_CANCEL`, exact atomics, study frame):

| | count | |
|---|---:|---|
| folded plies | 3,077,341 | |
| resolved adds | 8,339,951 | |
| resolved subs | 11,276,945 | **sum = 19,616,896 = `thr_into.rows` exactly** |
| **cancelling add/sub pairs** | **194,875** | = 389,750 rows = **1.99 % of threat rows** |
| **repeated indices within a list** | **0** | add **and** sub |

**Verdict: DEAD, with a number.** 1.99 % of threat rows = **0.0779 rows/node =
3.1 load dispatches/node = 0.14 % of engine loads**, against the cost of an
O(n²) scan over ~4.6 indices on every folded ply. Zero duplicates is the
stronger half of the result: the event generator and the resolver together
already emit a clean multiset. The counter is kept in `foldcount` so nobody
re-derives it.

---

## 4. THE PSQT CHANNEL, PRICED AND CLOSED

The canonical net carries a **PSQ1 section of 2,704,920 bytes** (magic at file
offset 135,578,720 of 138,283,640) and every resolved row triggers a random
32-byte read into it — 7.1754 per node, into a 2.7 MB table that cannot sit in a
1 MB L2 beside 20 MB of history and 41 MB of FT weights. `sota_threats.rs:1874`
calls it *"an L2-miss/L3-hit on essentially every event"*. If true, the arm is
obvious: **interleave each feature's 32-byte PSQ1 row into its own 1280-byte FT
row at load time**, converting a random access into a hit on a stream the fold
is already pulling.

**Before building a loader-plus-both-kernels rewrite, measure the ceiling.**
`nopsqt48` is a subtractive probe (study 29's `dstpin` shape): it drops the PSQ1
trailer at load, and both channels already hand-unswitch on
`psqt_weights.is_empty()`, so the difference is the **entire** psqt channel.

| `psq48` vs `base48`, n = 6 | median | se | neg/n | p |
|---|---:|---:|---:|---:|
| `instructions:u` | **−0.391 %** | **0.000** | **6/6** | 0.031 |
| `ls_dispatch.ld_dispatch:u` | **−0.820 %** | 0.389 | **6/6** | 0.031 |
| `ex_no_retire.load_not_complete` | −2.793 % | 3.506 | 5/6 | 0.22 |
| `ls_any_fills_from_sys.all` | **+1.589 %** | 0.480 | 0/6 | 0.031 |
| null | 0.000 % / +0.253 % | | 3/6 | 1.00 |

**The whole psqt channel is 24.1 instructions/node and 18.7 load dispatches/node
(0.82 %).** An interleaving arm can win at most a fraction of that, against a
+5 % inflation of every FT row (1280 → 1344 bytes to keep 64-byte alignment,
i.e. +14 weight-row loads/node) and a rewrite of the loader and both fold
kernels. **Closed. Do not build it.**

The probe changes the eval, so the tree differs — `bench … nodes` pins the node
count but not the positions, which is why fills read **+1.589 %** (wrong sign
for the mechanism, right sign for "different tree"). The two exact counters are
read at 6/6 with zero spread and are the ones quoted.

---

## 5. ARM THAT LOST — outlining the cold branch out of `HalfkaContext::compute`

**The hypothesis, from §1.3.** `current_full` carries a 264-byte frame and 59 of
its 106 static loads are `%rsp`-relative. `compute`'s king-move branch — Finny
refresh plus backward fill — runs on **19.1 %** of calls (823,842 refreshes
against 4,312,784 calls) but the `finny_refresh_into` **call** inside it clobbers
every caller-saved register, forcing the *hot* forward-fold path's values onto
the stack. Outline it behind `#[inline(never)]` and the hot path keeps its
induction variables in registers.

**It did exactly what it was supposed to do inside the symbol, and lost
engine-wide.**

| static, `current_full` | insns | loads | `%rsp` loads | `%rsp` stores |
|---|---:|---:|---:|---:|
| `base48` | 541 | 106 | 59 | 51 |
| `ctl48` (refactor, no `inline(never)`) | 541 | 106 | 59 | 51 |
| **`out48`** | **337** | **54** | **29** | **31** |

| `out48` vs `ctl48`, n = 4 | median | se | neg/n |
|---|---:|---:|---:|
| `instructions:u` | **+0.075 %** | **0.000** | **0/4** |
| `ls_dispatch.ld_dispatch:u` | +0.327 % | 0.243 | 1/4 |
| `ls_dispatch.store_dispatch:u` | +0.852 % | 0.288 | 1/4 |

**★ THE METHODOLOGICAL FINDING, and it is worth more than the arm: a static
per-symbol spill count is not a valid optimisation target.** Outlining halved
`current_full`'s static loads and its spills — and the engine issued *more*
loads, because the spills were **relocated into the callee's own frame plus a
call sequence**, not removed. Anyone reading §1.3's table as a to-do list should
read this paragraph first.

**A control worth recording:** `ctl48` is the same refactor with the attribute
removed, and it is **0.000 % on `instructions:u`** against `base48` and
identical on every static count. So the extraction itself is codegen-neutral and
`inline(never)` is the entire effect. The arm was reverted; `main` is unchanged
here.

---

## ★ 6. THE INSTRUMENT — three counters are not what they were advertised to be

The commission states that `instructions:u` / `branches:u` / `L1-dcache-loads:u`
are "exact, ~0 null". **Measured on byte-identical nulls in this study's own
harness, that is true of one of them.**

| counter, byte-identical null | n = 15 | n = 6 | n = 4 |
|---|---:|---:|---:|
| **`instructions:u`** | **0.000 %** | **0.000 %** | **0.000 %** |
| `L1-dcache-loads:u` | +0.038 % (6/15) | — | +0.059 % (1/4) |
| `ls_dispatch.ld_dispatch:u` | +0.138 % (5/15) | +0.253 % (3/6) | +0.157 % (0/4) |
| `ls_dispatch.store_dispatch:u` | — | — | +0.210 % (0/4) |
| **`ex_no_retire.load_not_complete`** | **−0.494 % (9/15)** | **+3.478 % (2/6)** | — |
| `cycles:u` | −1.089 % (10/15) | +1.509 % (2/6) | −0.763 % (2/4) |

* **`instructions:u` is the only exact counter**, at every `n`, in every panel.
* **`ls_dispatch.*` and `L1-dcache-loads` are NOT exact.** They count
  *dispatched* and *speculative* accesses, so branch-predictor state — which
  differs run to run even on an identical tree — moves them. Their real null is
  **0.04 – 0.25 %**. Still 5–20× better than `cycles:u`, and the reason this
  study could adjudicate at n = 15 where a cycles panel needs 40.
* **★ `ex_no_retire.load_not_complete` is not a resolvable adjudicating counter
  at achievable `n` on this box.** Its byte-identical null read **+3.478 % at
  n = 6** and −0.494 % at n = 15. **Study 44 adjudicated `r5` on this counter at
  −1.937 % (27/40)** — smaller than my n = 6 null, and 27/40 is p = 0.039. That
  reading is not *wrong*, but it is much thinner than it looks, and the
  store-queue counter (−19.474 %, 40/40) is what actually carried that arm.
  Nobody should aim a new arm at `load_not_complete` as its adjudicating
  statistic.

**Ordering** is the **blocked Latin square** throughout (`exact48.sh`): rotate
within a block of `NLAB`, reshuffle the base between blocks. Realised mean
position on the landing panel was **1.00 for all three labels** against an ideal
of 1.00.

**One operational trap, found the hard way and worth a line:** `GROUPS` is a
**bash read-only special array** (the caller's group IDs). `census48.sh`
assigned to it, bash silently kept `1000`, and `perf` reported
`event syntax error: '1000'`. Renamed to `EVGROUPS`.

---

## 7. GATES

| gate | result |
|---|---|
| node identity, every binary, study frame | **5 001 562** — exact on `base48`, `act48`, `out48`, `ctl48`, `null48` |
| d13 node identity | **1 739 588** — exact |
| wl50 eval hash | **`2e1ede95…79c4`** — exact |
| `base48` `.text` md5 vs study 47's `base47` | **`f411ea70` both** |
| `instructions:u`/node vs studies 46/47 | 6170.77 vs 6170.77 / 6170.97 (**0.000 % / −0.003 %**) |
| `ls_any_fills_from_sys.all` vs studies 42/44 | 576.20 vs 580.9 / 585.86 (−0.8 % / −1.7 %) |
| `foldcount` rows/node vs study 47 | **7.1753** vs 7.1754 |
| `cargo fmt --check` / `cargo clippy --all-targets` | clean |
| release perft suite | pass |
| `cargo test --features sota,arch-d48x` | **726 passed, 0 failed** — includes the kernel-vs-scalar equivalence tests, which are the real check on a retiled `activate` |
| **`Bench:`** | **587683** — unchanged |
| **NNUE eval hash** (from source) | **`2e1ede95…79c4`** — unchanged |
| **NNUE d13 nodes** (from source) | **1739588** — unchanged |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh` | **All checks passed.** |

**No pin moves, and that is the point:** the landed change is bit-identical, so
a pin update would have been evidence it was not what it claims to be.

D24 perft ritual not required: nothing under `movegen`, `board`, `zobrist` or
`attacks` is touched.

---

## 8. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | `ls_dispatch.ld_dispatch` counts one event per load op, not per double-pumped half | Weight-row model (287.0) matches study 47's independently-derived 287 `vpmovsxbw`/node exactly; and `L1-dcache-loads` / `ld_dispatch` = 1.32, which is what study 29's measured 2.0015-per-`zmm` predicts for this mix | §1's absolute loads/node. **Not** the contrasts, which are ratios |
| A2 | Per-symbol attribution of a sampled non-precise counter is sound (study 44 A5) | Two independent profiles agree symbol-for-symbol; shares reproduce study 44's to ~1 point | §1 and §2 — and §2 states the one case (a caller adjacent to a stall-heavy callee) where I actively doubt it |
| A3 | The landed `activate` change is bit-identical | Verified three ways rather than argued: identical node count at 5 M, identical d13, identical wl50 eval hash, plus 726 sota tests including kernel-vs-scalar equivalence | A behaviour change wearing a speedup's clothes |
| A4 | Study 47's 0.586 – 0.735 c/op for `activate` transfers to the ops removed here | Measured by study 47 on `activate`'s **own emitted sequence** with distinct load addresses; the removed ops are that sequence's own | Only the cycles *translation* of §0's result, never the exact-counter measurement |
| A5 | ~120 Elo per doubling of speed | Inherited (exp-0154); its 95 % CI includes zero (study 43) | Only the Elo translation |
| A6 | The blocked Latin square plus within-round differencing removes shared-box load | Nulls in §6, realised mean position 1.00 for every label | Every contrast in the study |

---

## 9. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **`HalfkaContext::current_full`'s 3.04 stall cycles per load is unexplained,
   and it is the largest single anomaly in the engine** (158.4 cycles/node,
   7.61 % of all stall, on 2.27 % of loads). Two mechanisms were tested and both
   died with numbers (§4, §5). The third — **sample skid from
   `i8_deltas_into` into its caller** — is untested and would, if true, mean the
   symbol has no stall of its own and study 44's A5 needs a caveat. **The next
   session should test A5 directly** before building anything aimed here.
2. **The cycles value of the landed arm is modelled, not measured.** −10.1 …
   −12.7 c/node comes from study 47's measured per-op cost × an exactly counted
   op delta. The panel's own `cycles:u` read −1.635 % against a −1.089 % null:
   consistent, useless. **A 40-round `cycclean` panel would settle it and I did
   not spend the wall clock**, because the two exact counters already decide
   whether to land.
3. **The same transform was NOT applied to the AVX2 backend.** Its
   `activate_inner` is a different i32/8-lane shape (`sota_simd.rs:1129`), so
   the trick has to be re-derived rather than copied. The `x86-64-v3` build is a
   real target and this is free Elo left on the floor for whoever wants it.
4. **Half the engine's loads have still never been censused per-site.**
   `search::<NonPv>` (269.9/node), `init_quiets` (188.4), `do_move` (112.1) and
   `make_move_with` (85.4) are 28.5 % of loads and 24 % of stall, and §1.3 shows
   they are two-thirds stack traffic. §5 kills the naive attack on that; it does
   not kill the target.
5. **`load_not_complete`'s null (§6) is bad enough that some prior conclusions
   should be re-read**, not retracted. I checked the one that matters —
   study 44's `r5` — and it is carried by `store_queue_rsrc_stall` at 40/40,
   not by this counter. I did not audit the others.
6. **Nothing here was arbitrated by games, and nothing could be**: every binary
   except the `nopsqt48` probe is node-identical, so a fixed-node match is a
   dead heat by construction, and at −0.3 % of cycles a real-time-control match
   needs thousands of games.
7. **The 4179.7 quiet-box denominator is inherited.** My own reads were
   4081.5 – 4101.3 with the D49 trainer and two other studies live. Percentages
   scale inversely with it; cycles/node figures do not.

---

## 10. REPRODUCTION

```bash
# 0. the worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. build. NOTE: after this study landed, the register-scale form is the
# DEFAULT, so `base48` here is plain main and there is no revert feature --
# reproduce the contrast by checking out the parent commit for the reference.
bash tools/perf48/build48.sh base48
cp bin48/base48 bin48/null48                      # the byte-identical null

# 2. THE GATE, before any timing
P48_CPU=2 bash tools/perf48/ident48.sh base48

# 3. the load census + the re-derived frame
P48_CPU=2 bash tools/perf48/census48.sh 4 base48
python3 tools/perf48/reduce48.py

# 4. where the loads and the stalls are, by symbol
P48_CPU=2 bash tools/perf48/where48.sh ls_dispatch.ld_dispatch:u       200003 base48
P48_CPU=2 bash tools/perf48/where48.sh ex_no_retire.load_not_complete:u 200003 base48

# 5. the STATIC per-symbol load + spill census
python3 tools/perf48/static48.py bin48/base48 \
    '6Worker6search' '10MovePicker11init_quiets' '13HalfkaContext12current_full'

# 6. the exact-counter panel (blocked Latin square)
P48_CPU=2 \
P48_EV='instructions:u,cycles:u,ls_dispatch.ld_dispatch:u,ex_no_retire.load_not_complete:u,L1-dcache-loads:u' \
P48_PANEL=docs/research/perf-artifacts-48/panel-act.txt \
  bash tools/perf48/exact48.sh 15 base48 act48 null48
python3 tools/perf48/med48.py docs/research/perf-artifacts-48/panel-act.txt \
  base48 act48 null48

# 7. the threat-redundancy census (NULL) and the psqt probe (CLOSED)
bash tools/perf48/build48.sh fc48   foldcount
bash tools/perf48/build48.sh psq48  nopsqt48
ARCHON_FOLDCOUNT=/tmp/fc.txt ./bin48/fc48 \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep thr_cancel /tmp/fc.txt
```

Pick a quiet **physical** core first (study 44 §7 — `taskset -c 2` is the house
default and gets contended; sum each SMT sibling pair before choosing).
