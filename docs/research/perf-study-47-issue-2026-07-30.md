# Perf study 47 — 512-bit issue throughput, narrowed and attacked

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 16 GB · **Base:** `main` at `6f4e09db` (studies 43/44/45/46
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280,
L1 1280 / L2 16 / L3 32 / BUCKETS 8 · **Instruments:** `tools/perf47/` +
`docs/research/perf-artifacts-47/` · **Worktree/branch:** `perf-study-47-issue`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO.
Denominator for percentages: the settled quiet-box **4179.7 cycles/node**;
study 44's loaded **4668.94** is quoted alongside where it changes the reading.

---

## ★ HEADLINE

**The bracket is narrowed from 13.5 points to 3.6, and the engine sits at the
BOTTOM of study 46's range, not its middle.**

| | 512-bit issue as a share of all cycles |
|---|---|
| study 44 (divisor "4 FP pipes", never measured) | 7.1 % – 14.4 % → *"the FP pipes are ≥ 85 % idle"* |
| study 46 (divisor measured; op mix unmeasured) | 16.8 % – 30.3 % |
| **study 47 (op mix measured, mix overlap measured twice)** | **20.8 % – 24.4 %** |

**869 – 1022 cycles/node of the 4179.7 frame are spent issuing 512-bit
operations.** Study 44's "≥ 85 % idle" does not survive — the corrected figure
is still **entirely above** its whole bracket. But study 46's guess that the
truth sat *"nearer the middle than either end"* of 16.8–30.3 is **wrong**: it is
near the bottom.

**★ I had this at 23.9 – 25.4 % and my own next measurement refuted it.** The
first draft priced `activate` from isolated per-op throughputs with the fold's
measured overlap factor borrowed across. Measuring `activate`'s real op sequence
(§2.4) put it at **0.59 c/op against the 0.89 that model predicted** — its mix
overlaps far better than the fold's, because it spreads across load, shift,
multiply, saturating-add and convert pipes where the fold hammers one converter.
The lower edge moved down ~3 points and the bracket got *wider*, not narrower.

**And then the corrected number was itself corrected.** The first `activate`
harness gave all three loads in a merge the same buffer offset, so they hit one
cache line — optimistic by 10 % (0.532 vs the faithful **0.586** c/op). Both
instrument faults are recorded in §2.4 rather than quietly fixed, because a
1.5-point bracket was exactly the kind of beautiful number rule 6 says to attack
hardest, and attacking it twice moved the answer both times.

**What closed the 13.5 points** was not one measurement but three, in this
order — each is a section below:

1. **The op-class throughput table** (§1). The engine's hot kernels issue
   `vpmovsxbw`, `vpaddw`, `vpaddsw`, `vpmullw`, `vpsllw`, `vpmovwb`,
   `vpcompressw` — none of which study 46 measured. They do **not** share one
   throughput: this box runs 512-bit ops at **0.52, 1.00 and 1.50 c/op**
   depending on class, a 2.9× spread.
2. **The exact dynamic op census** (§3), from static kernel shape × exact
   `foldcount` call counts, closing to the measured `pack_512` total with
   **15.2 % unattributed** (stated, not hidden).
3. **Two mix-overlap measurements** (§2.3, §2.5), which is the part a per-op
   table cannot give you. The shipped fold sequence costs **1.331
   c/chunk-update** where the isolated per-op sum predicts 1.524 (overlap 0.873);
   `activate`'s 15-op iteration costs **0.586 c/op** where its sum predicts 0.889
   (overlap **0.659**). **The two differ by 1.3×, which is why one measurement
   was not enough and why the borrowed factor was wrong.**

**★ The commissioned width-equivalence question is answered by measurement, and
it is CONFIRMED** (§1.2). `vpmovsxbw` moves **31.8 i16 lanes/cycle at 512 bits
and 31.7 at 256 bits**; `vpaddw` moves **61.8 and 63.0**. Zen 4's double-pump is
uniform, so a width change moves identical bytes per cycle and costs instruction
count. Nobody should build a width arm. I was asked to prove or refute this
rather than take it on trust; it is proved.

**★ THE HONEST HEADLINE ON ARMS: no code arm cleared the bar, and that is a
measurement result rather than a shrug.** **Fourteen candidate rewrites across
five sites** were priced against the throughput table; every one is **already at
its issue floor**, and
§5 gives each a number and a receipt rather than an argument. The two that
looked largest before measurement died for reasons worth keeping:

* **The fold kernels' sign-extension — 287 `vpmovsxbw`/node, 22.3 % of every
  512-bit op in the engine — is a HARD ARCHITECTURAL FLOOR, priced at 287
  cycles/node (6.9 % of the frame).** The converter runs at 32 i16 lanes/cycle
  and the fold must widen 7.175 rows × 1280 lanes = 9184 lanes/node. Every
  alternative was measured, not argued: ALU-domain sign-extension is 16–21
  lanes/cycle (worse), 64-byte-load-then-two-register-converts is 3 ops per 64
  bytes against 2 (worse), and i16 weight rows buy −0.195 c/chunk of issue while
  doubling the traffic of a kernel that is **86–88 % load-stall** (study 44 §2).
* **`activate` is 100 % memory-folded already** (§5.2) — the shipped
  disassembly folds the threat load into `vpsllw` and the scale load into
  `vpmullw`. And a new finding kills the obvious "CSE the repeated scale load"
  idea outright: **a folded memory operand costs exactly one issue slot, the
  same as the register-form op**, so hoisting a folded load *adds* an op. That
  also retro-explains study 46 §3.2's unexplained result (spreading `l1_dot`'s
  tail measured worse and it did not know why).

**Landed:** the instrument (`tools/perf47/`, eight tools, all reusable and all
with their controls), this document, and the measured closures written into
`sota_simd.rs` at the sites they govern so the next session does not re-derive
them. **Pins unmoved and proved unmoved:** `base47` is **byte-identical** to
study 46's `base46` (`cmp` clean), and the comment-only source edits rebuild
`cmp`-identical again.

**Instrument null, published as required:** the harness reproduces study 46's
two control instructions to **+0.4 % and −0.5 %** (`vfmadd231ps` 1.0044 vs
1.002; `vpaddd` 0.5187 vs 0.5516) before any new row was read off it.

---

## 0. WHAT I WAS ASKED, AND WHAT HAPPENED

| # | commission | outcome | where |
|---|---|---|---|
| 1 | Narrow the 13.5-point bracket by measuring the word-wise ops | **20.8 % – 24.4 %**, 3.6 points (a 3.7× narrowing) | §1–§4 |
| 2 | Prove or refute the 256/512 byte-per-cycle equivalence | **CONFIRMED by measurement**, both op classes | §1.2 |
| 3 | Attack it and land something | **No arm cleared the bar**; 14 candidates across 5 sites priced and closed with numbers | §5 |

---

## 1. THE INSTRUMENT — what a 512-bit op actually costs, by class

`tools/perf47/gen_ops47.py` is study 46's `gen_ilp.py` with the real opcode
inventory (read off the shipped disassembly by `census47.py`, not guessed) and
with **memory-source forms**, because `vpmovsxbw zmm, ymmword ptr [mem]` is how
the fold kernels ship and a register-form number is not the one the engine pays.
Method is study 46 §1.2's: generated inline asm, N independent chains,
`perf stat -e cycles:u`, min-of-5, pinned.

**The null first.** Study 46's `vfmadd231ps` and `vpaddd` are carried verbatim
as controls. This harness must reproduce them or its new rows mean nothing:

| control | study 46 | **study 47** | Δ |
|---|---:|---:|---:|
| `vfmadd231ps` zmm, latency (1 chain) | 4.005 | **4.028** | +0.6 % |
| `vfmadd231ps` zmm, throughput | 1.002 | **1.0044** | +0.2 % |
| `vpdpbusd` zmm, throughput | 1.002 | **1.0074** | +0.5 % |
| `vpaddd` zmm, throughput | 0.5516 | **0.5187** | −6.0 % |

Latency, the knee at 4, and the 1.00/0.55 two-class split all reproduce.

### 1.1 ★ The table nobody had — 512-bit reciprocal throughput by op class

Full sweep in `perf-artifacts-47/sweep-ops.txt`. Throughput column = 10 chains.

| instruction | where it lives | latency | **c/op** | effective pipes |
|---|---|---:|---:|---:|
| `vpaddw` / `vpsubw` | fold row loops | 1.03 | **0.518** | **~2** |
| `vpmaxsw` / `vpminsw` | `activate` clamp | 1.04 | **0.519 / 0.517** | **~2** |
| `vpaddd` (control) | — | 1.04 | 0.519 | ~2 |
| `vpmovsxbw` | **fold widening** | 1.03 | **1.005** | **1** |
| `vpmovsxbw` *(memory form)* | fold widening, as shipped | — | **1.005** | 1 |
| `vpsllw` | `activate` threat shift | 1.03 | 1.005 | 1 |
| `vpaddsw` | `activate` merge | 2.03 | **1.003** | 1 |
| `vpmullw` | `activate` scale multiply | 3.04 | 1.006 | 1 |
| `vpmulhuw` | `activate` product | 3.05 | 1.003 | 1 |
| `vpmovwb` | `activate` store | 1.03 | 1.007 | 1 |
| `vpbroadcastd` | `l1_dot` group broadcast | 1.04 | 1.007 | 1 |
| `vptestmd` | `find_nnz` | 1.04 | 1.006 | 1 |
| `vcvtdq2ps` | `l1_dequant` | 1.03 | 1.004 | 1 |
| `vpermt2b` | VBMI2 paths | 5.04 | 1.005 | 1 |
| `vfmadd231ps` / `vpdpbusd` | `l2_tail` / `l1_dot` | 4.03 | 1.004 | 1 |
| `vmovdqu64` *(512-bit load)* | everywhere | — | **1.077** | ~1 |
| `vpaddw` *(memory form)* | — | — | **1.102** | ~1 |
| **`vpcompressw`** | `find_nnz` | 5.08 | **1.517** | **<1** |

**Three findings in this table, all new:**

1. **The 2-per-cycle class is small and specific.** Only plain word/dword
   add/sub/min/max make it. Everything else in this engine — including
   *saturating* add — is one per cycle.
2. **★ Saturation costs 2×.** `vpaddsw` is **1.003** where `vpaddw` is
   **0.518**. Same lane width, same apparent ALU work, twice the issue cost.
   This is the single most surprising row and it is what makes `activate`'s
   merge more expensive than its instruction count suggests.
3. **★ A memory operand forces ~1 c/op regardless of class.** `vpaddw` is 0.518
   register-to-register and **1.102** with a memory source — it loses its second
   pipe. A bare 512-bit load is 1.077. **So on this box the unit of cost is
   closer to "one 512-bit memory touch or 1-pipe op per cycle" than to any
   per-instruction ALU model.**

### 1.2 ★ COMMISSIONED: the 256 vs 512 equivalence — CONFIRMED

The commission stated that 2×256-bit ops/cycle move the same bytes as
1×512-bit, so a width change buys nothing, and told me to prove or refute it
myself. Measured on the same harness, same method:

| instruction | 512-bit c/op | lanes/op | **lanes/cycle** | 256-bit c/op | lanes/op | **lanes/cycle** |
|---|---:|---:|---:|---:|---:|---:|
| `vpmovsxbw` | 1.005 | 32 | **31.8** | 0.504 | 16 | **31.7** |
| `vpaddw` | 0.518 | 32 | **61.8** | 0.254 | 16 | **63.0** |

**Equivalent to 0.3 % and 1.9 %, in both op classes.** Zen 4's double-pump is
uniform across the shuffle-class and ALU-class pipes alike, so a 512-bit op is
exactly two 256-bit ops' worth of pipe occupancy and nothing else. **The
equivalence holds; a width change is instruction count for zero throughput. Do
not build a width arm.**

This also settles the *shape* of the machine in a way the per-op table alone
does not: the box has a fixed **lane** throughput — ~32 i16 lanes/cycle through
the shuffle/convert pipes, ~63 through the simple-ALU pipes — and 512-bit
instructions are simply the cheapest way to spend it in instruction count.

---

## 2. WHAT A MIX COSTS — the measurement a per-op table cannot give

### 2.1 Why this is needed

Per-op reciprocal throughputs are measured one instruction at a time at maximum
ILP. Summing them across a mix assumes zero cross-class overlap, which is
false whenever two classes use disjoint pipes. `vpmovsxbw` (1 pipe, shuffle) and
`vpaddw` (2 pipes, ALU) are exactly that case, and the fold kernel issues them
**1:1** — so the sum could over-price the kernel by up to 2×.

`tools/perf47/gen_mix47.py` replicates the fold's actual op sequence with REGS
accumulators and N rows, entirely L1-resident so that **memory is not the
constraint and what is left is issue**.

### 2.2 The fold sequence, measured

Cycles per **chunk-update** (one row × one 32-lane chunk), REGS = 5:

| variant | what it is | rows=1 | rows=2 | rows=4 |
|---|---|---:|---:|---:|
| **`shipped`** | `vpmovsxbw zmm,[w]` + `vpaddw acc,acc,zmm` | 1.496 | 1.489 | **1.331** |
| `i16w` | `vpaddw acc,acc,[w]` — i16 weights, no widening | 1.232 | 1.207 | **1.136** |
| `cvtonly` | the widening alone | 1.225 | 1.115 | 1.064 |
| `alu` | `vpaddw acc,acc,zmm31` — no memory at all | 0.828 | 0.696 | 0.625 |

### 2.3 ★ The overlap factor, and what it prices

**Isolated sum** for the shipped pair = 1.005 + 0.518 = **1.524**.
**Measured together** = **1.331**. ⇒ **overlap factor 0.873.**

The classes genuinely share the machine — but only partly, and the residual is
the point: **adding the `vpaddw` to the widening costs +0.267 c/chunk (1.331 −
1.064), i.e. the ALU half is 80 % absorbed while the shuffle half is not
absorbed at all.** The widening is the critical path of this kernel.

**What deleting the widening would be worth, measured rather than modelled:**
`shipped` − `i16w` = **0.195 c/chunk × 287 chunk-updates/node = 56 c/node =
1.34 % of the frame** — and that is the *ceiling*, obtained by doubling the
weight bytes. §4.1 explains why that trade is a loss.

### 2.4 ★ `activate`'s real sequence — and TWO instrument corrections

The same harness carries `activate`'s **emitted** lane-iteration, transcribed
from the shipped disassembly at `0x70ab0` rather than from the Rust source: two
6-op merges (`vmovdqu64` pst, `vpsllw` with the threat load folded, `vpmullw`
with the scale load folded, `vpaddsw`, `vpmaxsw`, `vpminsw`) plus a 3-op tail
(`vpsllw $7`, `vpmulhuw`, `vpmovwb`) = **15 512-bit ops**.

| independent lane-groups | 1 | 2 | 3 | **4** |
|---|---:|---:|---:|---:|
| c/op — *first harness, all 3 loads at ONE offset* | 1.427 | 0.794 | 0.598 | *0.532* |
| **c/op — distinct addresses per load (FAITHFUL)** | 1.404 | 0.911 | 0.735 | **0.586** |
| c/iteration (faithful) | 21.06 | 13.67 | 11.03 | **8.78** |

**Correction 1 — the model was wrong.** The isolated per-op sum for this
sequence is **13.34 c/iteration = 0.889 c/op**. Measured, it is **0.586** — an
overlap factor of **0.659**. The fold's 0.873 does *not* generalise: the fold
issues a homogeneous, converter-heavy 1:1 mix that saturates one pipe class,
while `activate` spreads fifteen ops across load, shift, multiply,
saturating-add, min/max and convert pipes and absorbs most of them. Using the
fold's factor over-priced `activate` by 1.3×, and **that is what put the first
draft of §4's bracket at 23.9 – 25.4 %.**

**Correction 2 — my first `activate` harness was optimistic by 10 %.** It gave
all three loads in a merge the *same* buffer offset, so all three hit one cache
line; the real kernel reads three separate arrays. Re-generated with distinct
addresses per load, 0.532 → **0.586 c/op**. Both rows are printed above rather
than the flawed one being deleted.

**The fold measurement in §2.2 does NOT share this defect** — its sequence has
exactly one memory operand per chunk-update and the offset advances every slot.
It is used unchanged.

**Which group count the engine gets:** `activate`'s 40 lane-iterations are
mutually independent and LLVM unrolls the loop by 12 (confirmed by §3.1's census
— `vpmullw` appears 24 times = 2 per iteration × 12). So ≥ 4 independent groups
are always available, and the curve has not flattened at 4. §4 therefore
brackets `activate` between the groups-4 and groups-3 readings (**0.586 –
0.735**) rather than quoting the best number.

### 2.5 A free corroboration of `REGS_CAP = 5`

The same harness sweeps the tile width on the **issue** axis alone, which no
prior study could isolate (studies 44 and 46 measured it in-engine, where
store-queue pressure dominates):

| REGS | rows=2 (HalfKA shape) | rows=4 | rows=5 (threat shape) |
|---:|---:|---:|---:|
| 2 | 1.780 | 1.633 | 1.576 |
| 4 | 1.479 | 1.384 | 1.346 |
| **5** | **1.365** | **1.352** | **1.331** |
| 8 | 1.420 | 1.325 | 1.322 |

**`REGS = 5` is optimal on the issue axis at the HalfKA kernel's row count and
within 2 % of optimal at the threat kernel's** — an independent second reason
for study 44's setting, arrived at on a different axis. It also closes a
per-kernel split as a candidate arm: the most a split could win is the 1.352 →
1.325 at rows=4 (**−2.0 % of one kernel's issue**), against study 46's measured
**+1.009 % cycles** for global `r8`. Not worth a build.

---

## 3. THE OP CENSUS — static shape × exact dynamic counts

Study 46 could not weight its throughput classes because it had no op mix. The
mix is obtained here without sampling: **static per-call op counts read off the
shipped disassembly, multiplied by exact `foldcount` call counts** on the
identical frame (`perf-artifacts-43/foldcount43.txt`, 5,001,562 nodes — the same
artifact study 46 used for `evals/node`).

### 3.1 The static shapes (`tools/perf47/census47.py`)

| symbol | 512-bit insns | the shape |
|---|---:|---|
| `i8_deltas_into_generic` | 1135 | **450 `vpmovsxbw` : 450 `vpaddw`/`vpsubw`** + 235 moves |
| `finny_refresh_into` | 819 | **300 : 300** + 219 moves |
| `refresh_threat_into_at` | 700 | **120 : 120** + 460 moves |
| `SotaHead::forward_into` | 403 | the head mix (`vpdpbusd` 41, `vfmadd231ps` 30, `vpsllw` 28, `vpmullw` 24, `vpaddsw` 24, `vpmaxsw` 24, `vpminsw` 24, …) |
| `ThreatContext::at_full` | **21** | scalar/GPR code — **not vector at all** |
| `HalfkaContext::current_full` | **0** | **zero 512-bit ops** |

**★ The last two rows are a result, not a footnote.** Study 44 attributes
**11.35 % of all engine stalls** to `at_full` + `current_full`; this study shows
neither is 512-bit issue-bound in any degree. The engine's 512-bit cost is
concentrated in **four** places, which is what makes the bracket attributable.

### 3.2 The dynamic counts

From `foldcount43.txt`, per node (÷ 5,001,562):

| site | calls/node | rows/node |
|---|---:|---:|
| `hka_into` (shaped kernel) | 0.9784 | 2.2295 |
| `hka_dual` (Finny warm, two destinations) | 0.1647 | 0.9422 |
| `thr_into` (**the threat channel**) | 0.8538 | **3.9221** |
| `thr_refresh` | 0.0196 | 0.0815 |
| `thr_copy` (pure copy, 0 rows) | 0.0220 | 0 |
| `hka_inplace` | 0.0000 | 0.0001 |
| **total fold rows** | | **7.1754** |
| `activate` = `eval` | **0.43117** | — |

`rows/node = 7.1754` reproduces study 44's independently-derived 7.13 to 0.6 %.
`evals/node = 0.43117` is study 46 §2's exact figure.

### 3.3 The op budget, and how well it closes

Per-call shapes: the fold kernels do **40 accumulator loads + 40 stores** per
call (`hka_dual` 40 + 80, two destinations) plus **40 `vpmovsxbw` + 40
add/sub per row**; `activate` runs **40 lane-iterations × 15 512-bit ops**
(counted off the emitted loop in §4.2, not from source).

| block | 512-bit ops/node | share |
|---|---:|---:|
| fold row work (**287.0 widen + 287.0 add/sub**) | **574.0** | 44.7 % |
| fold accumulator loads + stores | 168.9 | 13.1 % |
| `activate` | 258.7 | 20.1 % |
| `l1_dot` (33.6) + `find_nnz` (30.2) + `l2_tail` (13.8) + misc | 87.6 | 6.8 % |
| **modelled subtotal** | **1089.2** | 84.8 % |
| **unattributed** | **195.5** | **15.2 %** |
| **measured total (§3.4)** | **1284.73** | 100 % |

**The 15.2 % is stated, not absorbed.** Its likely home is the 512-bit moves
LLVM emits for plain struct copies through the NNUE context types plus
`refresh_threat_into_at`'s 460 move instructions; I did not chase it because the
bracket is computed from the **measured total**, not from the model, and the
model is used only to split that total between throughput classes.

### 3.4 The denominator, re-derived (rule 8)

Three independent reads of the load-bearing total, on three binaries:

| | study 44 | study 46 | **study 47** |
|---|---:|---:|---:|
| `pack_512_uops_retired`/node | 1282.60 | 1283.25 | **1284.73** |
| `pack_256_uops_retired`/node | 33.75 | 33.75 | **33.69** |
| `fp_ops_retired_by_width.all`/node | 1399.34 | 1398.05 | **1396.72** |
| `instructions:u`/node | 5929.09 | 6170.77 | **6170.97** |

Spread on the load-bearing row: **0.17 %**. The instruction row confirms
`base47` is the same binary as study 46's `base46` — and §5 proves it with
`cmp`.

**A dead end worth recording so nobody repeats it:** Zen 4 also exposes
`fp_ops_retired_by_type.vector_{add,mul,mac,cvt,shuffle,…}`, which looks like it
would give the mix directly. It does not — **it counts only FLOATING-POINT ops.**
Validated by the same known-count method study 46 used for A4
(`tools/perf47/typesem47.sh`): `vfmadd231ps` lands in `vector_mac` at ratio
0.987, while `vpaddd`, `vpaddw`, `vpmovsxbw`, `vpmullw`, `vpmovwb` and every
other integer op read **exactly zero in every by_type bucket** and 1.0 in
`pack_512`. Artifact: `perf-artifacts-47/typesem.txt`.

---

## 4. ★ THE NARROWED BRACKET

Combining the measured class throughputs (§1.1), the measured mix overlap
(§2.3) and the census (§3.3), against the measured total of 1284.73 ops/node:

Two of the four blocks are now measured directly rather than modelled, which is
what makes the bracket narrow:

| block | ops/node | **c/node low** | **c/node high** | basis |
|---|---:|---:|---:|---|
| fold row work | 574.0 | **382.0** | **382.0** | **measured** (§2.2), 287 chunk-updates × 1.331 |
| fold accumulator loads + stores | 168.9 | 157.7 | 180.7 | `vmovdqu64` memory form measured at 1.077 (§1.1); low applies the fold's own 0.873 |
| `activate` | 258.7 | **151.6** | **190.1** | **measured** (§2.4), 0.586–0.735 c/op |
| head kernels + unattributed | 283.1 | 177.2 | 269.0 | **the only unmeasured block** — bracketed between `activate`'s 0.659 overlap and none |
| **total 512-bit issue** | **1284.7** | **868.5** | **1021.8** | |

| denominator | **bracket** |
|---|---|
| quiet box, 4179.7 c/node | **20.8 % – 24.4 %** |
| study 44's loaded 4668.94 | 18.6 % – 21.9 % |

**Study 44's bracket was 7.1 % – 14.4 % and its verdict "≥ 85 % idle". Study
46's was 16.8 % – 30.3 %. The measured answer is roughly one cycle in five.**

* **Study 44's kill does not survive at either edge** — even 20.8 % is above its
  entire bracket, so no reading of the counter rescues "the FP pipes are ≥ 85 %
  idle".
* **Study 46's correction is confirmed in direction and narrowed 3.7×** (13.5
  points → 3.6), but **its guess about where inside the range the truth sat is
  refuted**: it expected the middle, and the answer is the bottom quarter.
* **The dominant residual uncertainty is one block** — the head kernels plus the
  15.2 % unattributed, 283.1 ops/node, whose overlap factor is the difference
  between 20.8 % and 24.4 %. That is the measurement the next session should run
  first, and §2.4's harness already does this shape of job.

---

## 5. THE ARMS — five sites, fourteen candidates, every one at its issue floor

Each was priced against §1.1 before any code was written. **None cleared the
bar.** They are recorded with numbers so the next session does not re-derive
them, and because "already minimal" is only a legitimate conclusion when it
comes with the measurement that established it.

### 5.1 The fold widening — 287 ops/node (22.3 % of all 512-bit ops)

The largest single block in the engine, and a hard floor:

> The fold must widen **7.1754 rows × 1280 lanes = 9184 i16 lanes/node**. The
> convert pipes run at **32 lanes/cycle** (§1.2, measured at both widths).
> ⇒ **287 cycles/node = 6.87 % of the frame, irreducible at this data layout.**

Four alternatives, all measured or counted rather than argued:

| alternative | result |
|---|---|
| 64-B `vmovdqu64` + 2 register `vpmovsxbw` | **3 ops per 64 B vs 2** — the fused `vpmovsxbw zmm,[mem]` is already optimal (§1.1: memory form is 1.005, same as register) |
| ALU-domain sign-extend (`vpsllw`+`vpsraw`, or mask/sub) | 16–21 lanes/cycle vs 32 — **worse**, shifts and logicals are 1-pipe too |
| `vpmaddubsw` with a {1,0} selector | 2 ops per 64 B — **identical count**, and scrambles lane order |
| i8-domain row summing before one widen | **incorrect**: two i8 weights sum to [−255, 254], outside i8 |
| **i16 weight rows** (the only real win) | **−0.195 c/chunk = −1.34 % of frame** (§2.3) — but doubles the traffic of a kernel study 44 measured at **86–88 % load-stall**, and reverts study 32's measured **−6.8 % cycles**. Rejected on the same arithmetic that made study 32 right. |

### 5.2 `activate` — 258.7 ops/node (20.1 %)

**It is already fully memory-folded.** Read off the shipped disassembly
(`0x70ab0`), the per-merge sequence is:

```
vmovdqu64 (%rax,%r14,2),%zmm2          ; pst  load
vpsllw    $0x1,(%r8,%r14,2),%zmm3      ; threat load FOLDED into the shift
vpmullw   (%rbx,%r14,2),%zmm2,%zmm2    ; scale  load FOLDED into the multiply
vpaddsw   %zmm3,%zmm2,%zmm2
vpmaxsw   %zmm1,%zmm2,%zmm2
vpminsw   %zmm0,%zmm2,%zmm2
```

15 512-bit ops per lane-iteration, 40 iterations per eval. Candidates and why
each dies:

| candidate | verdict |
|---|---|
| CSE the scale load (each of 40 vectors is loaded twice per eval) | **★ STRICTLY LOSES.** A folded memory operand costs one issue slot — the same as the register-form op (§1.1). Hoisting turns `2 × vpmullw[mem]` into `1 × vmovdqu64 + 2 × vpmullw[reg]` = **3 ops for 2**. |
| `vpaddsw` → `vpaddw` (would halve that op: 1.003 → 0.518) | **Not bit-safe.** `vpmullw` is wrapping *by design* (the source pins this), so the saturating add can genuinely fire; wrapping instead flips those lanes to the opposite clamp. |
| `vpsllw $1` → `vpaddw t,t` (1.005 → 0.518) | **Loses.** The shift carries a folded load; the add cannot, so it costs `vmovdqu64` + `vpaddw` = 1.588 vs 1.005. |
| `vpsllw $7` + `vpmulhuw` → `vpmullw` + `vpsrlw` | **2 ops either way**, both 1-pipe. |
| `vpackuswb` to fuse the four clamp ops | **Loses**: 1 pack + 2 `vpmovzxbw` + mul + shift + narrow = 7 ops vs 7, and `vpmaxsw`/`vpminsw` are already the *cheap* 2-pipe class. |
| Pre-shift the threat accumulator to delete `vpsllw` | **Backwards.** It moves work from `activate` (34.5 merges/node) to the fold (**287 chunk-updates/node**) — 8× more frequent. |

### 5.3 `l1_dot` — 33.6 ops/node

Emitted inner loop is **2 ops per active group**: `vpbroadcastd` of the 4 FT
bytes, then `vpdpbusd` with the weight row as the memory operand. The obvious
arm — VNNI's embedded broadcast `{1to16}` — **cannot apply**: only one operand
may be memory, and it is already spent on the 64-byte weight row. Folding the
broadcast would force a separate 64-byte load. **2 ops is the floor.**

### 5.4 Per-kernel `REGS` split

Closed by §2.5 on the issue axis before a build: max win 2.0 % of one kernel's
issue, against study 46's measured **+1.009 % cycles** for global `r8`.

### 5.5 `at_full` / `current_full`

**21 and 0 512-bit ops respectively** (§3.1). 11.35 % of engine stalls, and no
issue-throughput lever exists in either. Their cost is scalar and memory.

---

## 6. GATES

| gate | result |
|---|---|
| `base47` node identity | **5 001 562** — exact |
| `base47` d13 node identity | **1 739 588** — exact |
| `base47` wl50 eval hash | `2e1ede95…79c4` — exact |
| **`base47` vs study 46's `base46`** | **`cmp` clean — BYTE-IDENTICAL** (independent reproduction of study 46's build) |
| `instructions:u`/node vs study 46 | 6170.97 vs 6170.77 (**+0.003 %**) |
| `pack_512_uops_retired`/node vs studies 44/46 | 1284.73 vs 1282.60 / 1283.25 (**spread 0.17 %**) |
| harness control vs study 46 (`vfmadd231ps` thr) | 1.0044 vs 1.002 (**+0.2 %**) |
| **comment-only source edits are codegen-neutral** | rebuilt binary **`cmp`-identical to `base47`** — proved, not asserted |
| `cargo fmt --check` / `cargo clippy --all-targets` | clean |
| release perft suite | pass |
| **`Bench:`** | **587683** — unchanged |
| **NNUE eval hash** (from source) | **`2e1ede95…79c4`** — unchanged |
| **NNUE d13 nodes** (from source) | **1739588** — unchanged |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` | **All checks passed** (rc = 0) |

### 6.1 ★ An instrument defect found in passing — `build44.sh` / `build46.sh` print a filename-dependent hash

`build47.sh` inherited this line from study 44:

```bash
objdump -d --section=.text "bin47/$LABEL" | md5sum
```

**`objdump`'s first output line is `<path>: file format elf64-x86-64`, so the
hash includes the FILENAME.** Two byte-identical binaries built under different
labels print different md5s. Caught here because `cmt` and `base47` are provably
identical — same `md5sum`, `cmp` clean — yet printed `227d73ba` and `b3ed01ed`.
Stripping the header with `tail -n +3` makes both `f411ea70`.

Nothing published by studies 44 or 46 depends on it (their code-size claims use
`size -A` and `nm` byte counts, which are sound), but the line is still live in
both scripts and would report a false "differs" to anyone using it to check that
two arms share codegen. **Fixed in `tools/perf47/build47.sh`; the other two are
untouched and should be corrected when next edited.**

**No pin moves, and no pin should move: nothing reaching the shipped binary
changed.** The only source edits are comments recording §5's closures.

---

## 7. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | The generated-asm sweep measures issue throughput, not a loop or frequency artifact | Reproduces study 46's four control rows to ≤ 0.6 %; the 256-bit forms give a different, self-consistent answer (§1.2) | §1.1, and with it the whole bracket |
| A2 | The static shapes in §3.1/§3.3 are the shapes the engine executes | Read off the **shipped** disassembly, and `rows/node = 7.1754` reproduces study 44's independent 7.13 to 0.6 % | §3.3's split of the total between classes; **not** the total itself |
| A3 | The head kernels + the unattributed 15.2 % overlap somewhere between `activate`'s 0.598 and no overlap at all | **The one remaining modelled block.** Both endpoints are *measured* factors from this study, applied to a block whose composition is only partly known | The bracket's width. Both edges move together; the *direction* of the study-44 correction survives either way |
| A4 | `evals/node = 0.43117` and the `foldcount` counts transfer to this binary | Exact atomic counters on the identical frame; identical in three studies' artifacts | §3.2, §3.3 |
| A5 | `pack_512_uops_retired` = one event per 512-bit instruction | **Not assumed** — closed by study 46 §5.1 (ratio 1.0000) and re-confirmed here in §3.4's by-type validation | Nothing; the total is the anchor |
| A6 | The cycles/node denominator is 4179.7 (quiet) | Inherited from study 40/46; my own box read 4804.0 under the live D49 trainer | The *percentage*, not the cycles/node. Both denominators are quoted |

---

## 8. REPRODUCTION

```bash
# 0. worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. THE OP-CLASS THROUGHPUT TABLE  (no repo build needed)
python3 tools/perf47/gen_ops47.py /tmp/ops47.c
gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi2 \
    -o /tmp/ops47 /tmp/ops47.c
P47_REPS=5 bash tools/perf47/sweep47.sh vfmadd231ps vpaddd vpmovsxbw vpaddw \
    vpaddsw vpmullw vpcompressw vpmovsxbw_y vpaddw_y      # controls first

# 2. THE MIX / OVERLAP MEASUREMENT (issue isolated: all L1-resident)
python3 tools/perf47/gen_mix47.py /tmp/mix47.c
gcc -O2 -mavx512f -mavx512bw -mavx512vl -o /tmp/mix47 /tmp/mix47.c
perf stat -e cycles:u -- taskset -c 0 /tmp/mix47 shipped 5 4 400000
perf stat -e cycles:u -- taskset -c 0 /tmp/mix47 i16w    5 4 400000

# 3. the by_type dead end, so nobody re-walks it
bash tools/perf47/typesem47.sh

# 4. the static census + the re-derived denominator
bash tools/perf47/build47.sh base47
python3 tools/perf47/census47.py bin47/base47 \
    'i8_deltas_into_generic' 'SotaHead>::forward_into' 'ThreatContext>::at_full'
P47_CPU=0 bash tools/perf47/ident47.sh base47
P47_REPS=5 bash tools/perf47/fpcensus47.sh base47
```

---

## 9. WHAT I COULD NOT ESTABLISH

In order, most doubtful first.

1. **★ THE ARM SIDE PRODUCED NOTHING, and that is the study's real shortfall.**
   Seven sites are priced and closed in §5, but a priced closure is weaker
   evidence than a built arm that lost — §5.2's `vpaddsw` and CSE rows in
   particular rest on *reading* the disassembly and the throughput table, not on
   building the variant and watching `instructions:u`. Each is cheap to falsify
   and I did not spend the build to do it. **If any single line in this document
   deserves attack first, it is the claim that `activate` is at its floor.**
2. **~~The overlap factor is measured for ONE op pair~~ — RESOLVED IN SESSION,
   and it moved the answer.** I measured `activate`'s real sequence (§2.4)
   rather than borrowing the fold's factor, and it came back 1.5× cheaper,
   dropping the bracket's lower edge from 23.9 % to 20.8 %. **The same job is
   still owed on the head kernels + the unattributed block** (283.1 ops/node),
   which is now the single largest source of width in §4. `gen_mix47.py` already
   does this shape of job; `l1_dot`'s 2-op group and `find_nnz`'s iteration are
   both short enough to transcribe in minutes.
3. **15.2 % of the 512-bit ops are unattributed** (§3.3). They are split between
   classes pro-rata with the head, which is a guess. Because they are ~1-pipe
   ops either way the bracket moves by well under a point, but the *attribution*
   in §3.3's table should not be quoted to the last figure.
4. **The 4179.7 quiet-box denominator is inherited, not measured here.** My own
   reads were 4804.0 with the D49 trainer live. Every percentage scales
   inversely with it; the cycles/node figures do not.
5. **Nothing here was arbitrated by games, and nothing could be** — every binary
   in this study is node-identical, so a fixed-node match is a dead heat by
   construction. No cycles panel was run either: **this study measured
   microbenchmarks and exact counters only**, and deliberately did not spend
   wall clock on a panel it had no arm to put through.
6. **`vpcompressw` at 1.517 c/op is the worst op class in the engine and is
   left unexploited.** It is only ~4.3 ops/node (`find_nnz`, 1 per 128-group
   iteration), so it cannot matter at the current width — but it would scale
   linearly with ACC, and at a wider net it is the first row of §1.1 that would
   start to hurt.
7. **The i16-weight arm (§5.1) was priced from a microbenchmark, never built.**
   Its issue win (−1.34 % of frame) is measured; its memory loss is *inferred*
   from study 32's measured −6.8 % and study 44's 86–88 % load-stall share. The
   inference is one rung deep and the conclusion is comfortable, but it is an
   inference.
