# Perf study 44 — the back end, measured

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 16 GB · **Base:** `main` at `7dd617df` (study 42
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf44/` · **Artifacts:**
`docs/research/perf-artifacts-44/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO.
**Every binary in this study is node-identical at 5,001,562, d13-identical at
1,739,588, and eval-hash-identical at
`2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4`** — gated
per binary by `tools/perf44/ident44.sh`, never assumed.

---

## ★ HEADLINE

**The 49 % back end is 83 % MEMORY and 17 % EXECUTION.**
`ex_no_retire.load_not_complete` is **2558.77 cycles/node — 54.80 % of every
cycle the engine spends**, which is **12.4× the entire front-end bucket study 41
spent a session on**. The execution residual is 531.31 (11.38 %). The partition
closes on cycles to **−0.161 %**, and study 41's slot accounting reproduces
(back end **48.61 %** vs 49.0 %, front end **20.25 %** vs 19.9 %).

**Of the four candidates the commission named, one is dead and one is new:**

* **Execution port pressure — DEAD.** Zen 4's four FP pipes are **≥ 85 % idle**
  and `fp_sch_rsrc_stall` blocks dispatch for only **4.35 %** of cycles.
* **Store-buffer pressure — REAL, and the largest single dispatch blocker in
  the engine** at **509.30 cycles/node (10.91 %)**, ahead of the load queue's
  450.12. Nobody had looked.
* **Dependency chains — 11.38 %**, and localised to exactly **one** symbol:
  `forward_into` takes 8.20 % of `not_complete` on 3.87 % of `load_not_complete`.
* **Memory — the answer, but NOT the memory people assume.** Only **1.61 % of
  demand fills reach DRAM** (7.90/node). 90.76 % come from **L2**. The engine
  is not waiting on catastrophic misses; it pays **1.12 stall cycles on every
  one of 2281 load dispatches**, because **53.8 % of all instructions are memory
  operations**, at MLP 5.74 and 54-cycle average miss residency. **The lever is
  the NUMBER of memory operations, not their latency.**

**Two negatives reported as results:**

1. **Arm 1 (history prefetch) LOST** — +2.44 % instructions for a **4/10 coin
   flip on the counter it was built to move**. The six history planes are
   reused across the quiets of one `init_quiets` call, so there is no exposed
   latency to hide. The census had already said the problem is memory-op
   *count*; a prefetch adds count.
2. **The panel instrument was BROKEN and the null caught it.** A byte-identical
   binary read **+2.36 % cycles**. Cause: a +3.9 % within-round drift combined
   with a free shuffle that seated the *reference* at mean position 0.77 of 1.50
   — faking a loss and masking a win simultaneously. Replaced with a **balanced
   Latin square** (§4.2).

---

## 0. THE COMMISSION, AND THE FIRST THING I DID WITH IT

Study 41 closed the front end (19.9 % of dispatch slots, ~8.5 Elo if the whole
`ex_no_retire.empty` bucket were abolished) and handed over one number nobody
had decomposed: **the engine is back-end bound, 49.0 % of dispatch slots.**

Rule 8 says re-derive any number whose `n` you did not set. So the first panel
here is not an arm, it is study 41's own slot accounting rebuilt from scratch on
this study's own binary.

### 0.1 Study 41's slot accounting REPRODUCES

`tools/perf44/census44.sh`, 12 unmultiplexed groups × 6 reps, `bin44/base44`.

| dispatch slots | study 41 (n = 6) | **this study (n = 6)** |
|---|---:|---:|
| ops dispatched | 29.6 % | **27.66 %** |
| **back-end bound** | **49.0 %** | **48.61 %** |
| **front-end bound** | **19.9 %** | **20.25 %** |
| SMT contention | 1.5 % | 3.49 % |
| closure (implied vs measured cycles) | −0.03 % | **−0.396 %** |

**The two load-bearing figures reproduce to 0.4 and 0.35 points.** The SMT row is
the one that moved, and it moved for a stated reason: this study ran on a box
carrying more concurrent agents than study 41's (see §7), and
`ex_no_retire.thread_not_selected` is the counter study 40 measured swinging
−56 % to +203 % between rounds. My closure is −0.396 % against study 41's
−0.03 %; §0.3 explains what closed most of that gap.

**So the 49 % stands, and this study is a decomposition of it, not a re-audit.**

### 0.2 The denominator, stated, and which binary it belongs to

| `bin44/base44` — `main` @ `7dd617df`, non-PGO, `-C target-cpu=native` | per node |
|---|---:|
| `instructions:u` | **5929.09** |
| `cycles:u` (census, box load ~5–9) | **4668.94** |
| `ex_no_retire.load_not_complete:u` (raw level — see §0.3) | **2596.93** |
| `ex_no_retire.not_complete:u` (raw level — see §0.3) | **3133.67** |
| `ls_dispatch.ld_dispatch:u` | **2280.76** |
| `ls_dispatch.store_dispatch:u` | **907.31** |
| `ls_any_fills_from_sys.all:u` | **585.86** |

Two independent frame checks, both passed before anything was built on them:

* **`instructions:u` = 5929.09** against study 41's 5928.40 (+0.012 %) and the
  coordinator's settled 5923.909 (+0.087 %). Same frame.
* **585.86 cache fills/node** against study 42's independently measured
  **580.9** (+0.85 %). Same memory frame.

`cycles:u` is the one that is not comparable across studies, and this study
never compares it across studies. **4668.94 is a loaded-box figure and can only
be an over-estimate of the quiet-box level** (contention adds cycles, never
removes them); the settled quiet figure is 4179.7 and study 42 bracketed
3992.4 … 4390.8 on its own panels. **Every cycle percentage below therefore
UNDER-states the share slightly**, which is the safe direction. Absolute
cycle conversions are quoted as a range across 4179.7 … 4668.94.

**Perf study 36's 8262.9 cycles/node is used nowhere in this study**
(`grep -rn 8262 tools/perf44 docs/research/perf-study-44-*` is empty).

### 0.3 ★ Shares are computed WITHIN each run, never level-over-pooled-level

Mid-study the coordinator circulated a harness defect found in study 43: five
panel scripts reordered labels by *cyclic rotation*, which changes only the
starting point and leaves relative order invariant, so per-position bias never
averages out. **None of this study's panels ever rotated** (`panel44.sh` used a
free shuffle and now uses a Latin square, §4.2). But the warning landed on
something else of mine that was genuinely wrong:

**`census44.sh` walks its 12 event groups in a FIXED order**, so each group
always occupies the same slot of a rep — and `reduce44.py` originally divided
each bucket's per-node LEVEL by a *global* median of `cycles:u` pooled across
all 72 runs. That imports one slot's bias into another slot's numerator, which
is exactly the failure mode the note describes.

**The fix, applied to every share in this document:** every group carries
`cycles:u`, so each run yields a self-contained ratio `event / cycles` measured
in the *same run*, at the *same slot*, under the *same* instantaneous load. Any
common multiplicative drift cancels in the division. Shares are the median of
those per-run ratios; absolute cycles/node are that share × the reference
denominator.

**It tightened both closures and moved no conclusion:**

| | pooled-level (wrong) | **within-run (used)** |
|---|---:|---:|
| `ex_no_retire` partition residual | −1.183 % | **−0.161 %** |
| dispatch-slot closure | −0.997 % | **−0.396 %** |
| `load_not_complete` share of cycles | 55.62 % | **54.80 %** |
| back-end slot share | 48.38 % | **48.61 %** |

A 7.3× improvement in the partition residual is a real check that the
correction was warranted rather than cosmetic — and the headline moved by
0.8 points, which is why the conclusions stand.

---

## ★ 1. THE DECOMPOSITION

### 1.1 The partition, which closes on cycles rather than on slots

Zen 4's `ex_no_retire.*` umasks tile every cycle in which nothing retired, and
— this is the event that makes the study possible —
**`ex_no_retire.load_not_complete` is a hardware-provided SUBSET of
`ex_no_retire.not_complete`**: the cycles where the oldest op was blocked
specifically because *load data had not arrived*. The complement is everything
else that can keep the oldest op incomplete: execution latency, dependency
chains, divides, long-latency shuffles.

| bucket | cycles/node | **% of all cycles** |
|---|---:|---:|
| **`load_not_complete` — MEMORY** | **2558.77** | **54.80 %** |
| `not_complete` − `load_not_complete` — **EXECUTION** | **531.31** | **11.38 %** |
| `other` | 221.94 | 4.75 % |
| `empty` — front end (study 41's whole subject) | 206.88 | 4.43 % |
| `thread_not_selected` — SMT | 59.63 | 1.28 % |
| **— all no-retire cycles** | **3578.53** | **76.65 %** |
| **RETIRING cycles** | **1090.41** | **23.35 %** |

Closure: `ex_no_retire.all` measured **3584.30** against the parts' sum
**3578.53** — **a −0.161 % residual**.

**The answer to the commissioned question, in one line: the 49 % back-end
bucket is 83 % memory and 17 % execution.** Not "memory-bound" as a mood —
`load_not_complete` alone is **more than half of every cycle the engine
spends**, and it is 12.4× the entire front-end bucket study 41 spent a session
on.

### 1.2 The four candidates, resolved with numbers

The commission named four. Three of them are now priced and two are dead.

| candidate | the measurement | verdict |
|---|---|---|
| **Memory / load stalls** | `load_not_complete` = **2558.77 c/node, 54.80 % of cycles** | **THE ANSWER.** §2 says which memory. |
| **Execution port pressure** | `fp_ops_retired_by_width.all` = **1399.34 uops/node** (1282.60 of them 512-bit). Over Zen 4's 4 FP pipes that is **7.1 % – 14.4 % occupancy** depending on how the counter names a double-pumped op (see below). Corroborated by a counter that needs no such assumption: `fp_sch_rsrc_stall` = **203.01 c/node (4.35 % of cycles)** | **DEAD at either end of the bracket. The FP pipes are ≥ 85 % idle.** Do not build a port-pressure arm. |
| **Dependency chains / latency** | the execution residual, **531.31 c/node = 11.38 %** | **REAL BUT SECONDARY**, and §2.2 localises it to ONE symbol. |
| **Store-buffer / write pressure** | `store_queue_rsrc_stall` = **509.30 c/node = 10.91 % of cycles** — the **largest single dispatch-blocking resource in the engine**, ahead of the load queue at 450.12 | **REAL.** It is what §4's arm targets. |

**Why the port-pressure figure is a BRACKET and not a number, stated because
the honest version is less quotable than the tidy one.** Whether
`pack_512_uops_retired` counts one event per 512-bit instruction or two (Zen 4
executes 512-bit ops on a 256-bit datapath) decides whether the pipe demand is
2605 or 1399 pipe-cycles per node — 14.4 % or 7.1 % of a 4-pipe machine. I did
not validate that semantics on this box, and the arithmetic points at the
second reading: a static count of the fold kernels' FP work
(7.13 weight rows/node × 40 blocks × 2 ops = 570, plus `activate` ~171, plus
`l1_dot` and the rest) lands near **790 instructions/node** against a measured
**1282.60**, a ratio of 1.62 that is much easier to explain by double-counting
than by 493 missing FP instructions.

**The verdict does not depend on which reading is right**, which is why the
bracket is quoted instead of a resolution: the pipes are between 85.6 % and
92.9 % idle, and `fp_sch_rsrc_stall` — a cycle counter that involves no uop
accounting at all — independently says dispatch is blocked on the FP scheduler
for only 4.35 % of cycles.

The full dispatch-blocking table, since it is the first time anyone has looked
(these are cycles in which dispatch was blocked on that structure; they overlap
with each other and so do not partition):

| resource | cycles/node | % of cycles |
|---|---:|---:|
| **store queue** | **509.30** | **10.91 %** |
| load queue | 450.12 | 9.64 % |
| FP register file | 333.50 | 7.14 % |
| integer physical register file | 256.57 | 5.50 % |
| **FP scheduler** (the port-pressure signature) | **203.01** | **4.35 %** |
| retire tokens (ROB full) | 196.57 | 4.21 % |
| integer scheduler 1 | 149.82 | 3.21 % |
| integer schedulers 0 / 3 / 2 | 78.99 / 69.90 / 37.02 | 1.69 / 1.50 / 0.79 % |
| FP flush recovery | 5.95 | 0.13 % |
| taken-branch buffer | 0.01 | 0.00 % |

### 1.3 WHICH memory — and it is not DRAM

Study 42's trap says a stall rate is a property of the line population. So here
is this engine's population, measured rather than assumed:

| where a DEMAND load is satisfied | fills/node | share |
|---|---:|---:|
| local **L2** | 444.15 | **90.76 %** |
| local CCX (**L3**) | 37.29 | 7.62 % |
| **DRAM** | 7.90 | **1.61 %** |
| far / remote / alternate | 0.00 | 0 % |

Plus the hardware prefetcher's own traffic: `ls_mab_alloc` splits
**496.03 load/store allocations** against **95.10 hardware-prefetcher
allocations** per node, and L2 data traffic is **90.57 % hits**
(526.94 of 581.81), of which **198.78 requests/node are the L2 HW prefetcher**.

Three derived quantities that between them describe the whole memory system:

| | value | how |
|---|---:|---|
| memory ops per node | **3188.1** (2280.8 loads + 907.3 stores) = **53.8 % of instructions** | `ls_dispatch.*` |
| average in-flight L1 misses (**MLP**) | **5.74** | `ls_alloc_mab_count` / `cycles` |
| average L1-miss residency | **53.99 cycles** | `ls_alloc_mab_count` / `ls_mab_alloc.load_store_allocations` |
| load-stall cycles per load dispatched | **1.122** | `load_not_complete` / `ld_dispatch` |
| load-stall cycles per demand fill | **5.23** | `load_not_complete` / demand fills |

**This is the shape of the problem, and it is not the shape people assume.**
The engine is not waiting on a handful of catastrophic DRAM misses — there are
only **7.9 DRAM fills per node**, and the average missing line comes back in
**54 cycles** with **5.7 of them in flight at once**. It is waiting **1.12
cycles on every one of 2281 load dispatches**, because 53.8 % of all
instructions are memory operations and the L2 is doing 444 fills a node to keep
up.

**Consequence for anyone planning a lever: the prize is in reducing the NUMBER
of memory operations, or in raising MLP. It is not in reducing miss LATENCY,
because the latency is already L2-class and already 5.7-way overlapped.**
§3 is what happens when you ignore that sentence.

---

## 2. WHERE — the stalls, attributed to symbols

`tools/perf44/where44.sh` samples the stall counter *itself* (period 200003)
rather than `cycles`, so a symbol's share reads as "this code owns N % of the
engine's load-stall cycles". Per study 36 §2.3, per-*instruction* attribution of
a counter like this is an artifact on Zen 4; per-**symbol** shares survive,
which is all this table asks.

| symbol | % of `load_not_complete` | % of `not_complete` | **implied: fraction of its own stalls that is LOAD** |
|---|---:|---:|---:|
| `avx512::i8_deltas_into` (HalfKA fold) | **13.56** | 12.97 | 86 % |
| `avx512::i8_deltas_into_generic` (threat fold) | **12.87** | 12.01 | 88 % |
| `search::<NonPv>` | 11.06 | 10.75 | 84 % |
| `HalfkaContext::finny_refresh_into` | 9.43 | 8.26 | **94 %** |
| `MovePicker::init_quiets` | 6.57 | 6.03 | 89 % |
| `HalfkaContext::current_full` | 6.01 | 5.70 | 86 % |
| `ThreatContext::at_full` | 5.94 | 5.65 | 86 % |
| **`SotaHead::forward_into`** | **3.87** | **8.20** | **39 %** |
| `Worker::do_move` | 3.45 | 3.11 | 91 % |
| `Board::make_move_with::<TtPrefetch>` | 2.33 | 2.24 | 85 % |
| `qsearch::<NonPv>` | 2.14 | 1.96 | 89 % |

(The last column is `0.8288 × sL/sN`, where 0.8288 is the engine-wide
`load_not_complete / not_complete` ratio. It is a ratio of two sampled shares
and should be read as "clearly above / clearly below average", not to two
figures.)

### 2.1 NNUE owns ~62 % of every load-stall cycle in the engine

Summing the NNUE symbols (both fold kernels, `finny_refresh_into`,
`current_full`, `at_full`, `forward_into`, `record_square_touch`,
`SotaContext::evaluate`, `apply_at`, `refresh_threat_into_at`,
`ensure_threats`) gives **~62.3 %** of `load_not_complete`. The two feature-
transformer fold kernels alone are **26.43 %** — worth **676 … 756 cycles/node**
across the denominator range.

### 2.2 One symbol is execution-bound, and only one

**`SotaHead::forward_into` takes 8.20 % of `not_complete` while taking only
3.87 % of `load_not_complete`** — its share of total stalls is **2.1× its share
of load stalls**, the only inversion in the table. That is the signature of a
dependency chain rather than a memory wait, and it is consistent with study 29's
independent IBS observation that `forward_into` is **12.25 % of the engine's
loads but only 2.18 % of its load-stall cycles**: this symbol issues a lot of
memory traffic that does *not* stall, and stalls on something else.

**The obvious mechanism is NOT the mechanism, and I checked before writing it
down.** `l1_dot_inner` is a `vpdpbusd` chain (Zen 4 ~4-cycle latency at ~1/cycle
throughput) and perf study 21 added `CHAINS` independent accumulator sets for
exactly that — gated on `BLOCKS = L2 / L2_BLK < 4`. My first draft of this
section asserted that D48X has `L2 = 128`, hence `BLOCKS = 8`, hence
`CHAINS = 1`, hence study 21's fix is switched off. **That is false.**
`sota_head.rs:100-101` gives `L2 = 16` for `arch-d48x`, so `BLOCKS = 1` and
`CHAINS = 4` — **study 21's fix is already active at the shipping arch.**

So the chain is somewhere else in `forward_into` — `l3_dot` is a true reduction,
`propagate_l2` is a 16×32 f32 matmul, and the WDL softmax is serial — and this
study did not localise it further. §8 records it as the best-identified
UNBUILT lever, with that correction attached, rather than as a finding.

---

## 3. ARM 1 — the history prefetch. It LOST, and the reason is worth more than the arm.

### 3.1 The hypothesis, and why it looked good

The history catalogue is **~20 MiB** (`pawn` 12.6 MB, `cont` 5.1 MB, `low_ply`
640 KB, `main` 256 KB, `cont_corr` 1.28 MB, `corr` 1 MB), and it shares a 32 MB
L3 with a 16 MB TT and ~29 MB of NNUE i8 weights — a **~65 MB working set
against 32 MB of L3**. `init_quiets` reads **seven randomly addressed history
cells per scored quiet** and nothing prefetches any of them, while study 42 had
just priced the ONE cache hint the TT ships at **3.5 % of engine cycles**.

`histpf44` software-pipelines an `_MM_HINT_T0` hint `PF_DIST = 4` moves ahead of
the read, with a prologue for the first four. Node-identical by construction and
gated as such.

### 3.2 It lost on every counter that matters, including its own

`tools/perf44/panel44.sh`, paired within round, Fisher-Yates order per round,
byte-identical null (`null44 = cp base44`) riding in the rotation, **n = 10**.

| `hpf44` vs `base44` | median | trimmed mean | se | rounds negative | p |
|---|---:|---:|---:|---:|---:|
| `instructions:u` | **+2.435 %** | +2.435 % | **0.000** | **0/10** | 0.0020 |
| `cycles:u` | **+1.513 %** | +1.591 % | 0.237 | 1/10 | 0.0215 |
| `ex_no_retire.not_complete` | +0.820 % | +0.978 % | 0.321 | 1/10 | 0.0215 |
| **`ex_no_retire.load_not_complete`** | **+0.763 %** | +0.917 % | 0.380 | **4/10** | **0.7539** |

**The null is a coin flip on every counter** — 5/10, 5/10, 5/10, p = 1.0000 —
and exactly 0.000 % on `instructions:u`, which is the check that the panel is
sound.

**Read the last row first. The arm moved the counter it was built to move by
4/10 — a coin flip.** It bought **nothing**, and it paid **+2.435 %
instructions = +144.4 per node**, which at the frame's marginal issue rate is
most of the +1.5 % of cycles.

### 3.3 ★ Why, and this is the finding

**All the quiets scored in one `init_quiets` call share the SAME six
1536-byte plane blocks.** `pawn_slot(pawn_key)` and the five `cont_ref(...)`
bases are resolved *once*, above the loop; only `q = piece_to(pci, to)` varies
per move, and `q < 768` indexes *within* those same six blocks. A block is 24
cache lines, a real quiet list is 20–40 moves — so the lines are **reused
across the loop**, the first touch of each pays, and there is no per-move
exposed latency for a 4-move lookahead to hide.

Which is exactly what §1.3 predicted and I did not listen to: **the engine's
problem is the NUMBER of memory operations, not their latency.** A prefetch
adds memory operations. It was structurally the wrong instrument for this
machine's failure mode, and the census said so one section before I built it.

**I am recording this as a lever killed with a number, not as a null to be
retried at another distance.** A larger `PF_DIST`, a cheaper subset of the seven
planes, or an early bulk pre-pass would all reduce the *cost* — but the benefit
term is measured at zero (4/10 on its own counter), and reducing the cost of
something worth zero is worth zero. Two things WOULD justify revisiting it, and
neither is a distance change: evidence that the first touch of each plane is
itself expensive (it is not — these are L2-class lines, §1.3), or a caller whose
planes are *not* reused across a loop.

---

## 4. ARM 2 — the fold-tile width, aimed by the store-queue counter

*(populated below — see §4.2 for the panel)*

### 4.1 Why this arm exists, and why nobody had built it

`REGS` is the number of live `zmm` accumulators a fold kernel's tile holds; at
ACC = 1280 it is **20**, chosen as the largest exact divisor of `ACC_BLOCKS`.
Everything ever measured about it was measured on **`instructions:u`** — a
counter whose noise floor on this box is exactly 0.0, which is precisely why the
question was never asked properly: **more tiles is unambiguously more loop
control, so a narrower tile loses on instructions by construction.** The source
comment records `REGS = 40` as a disaster (+1007.7 i/n, spills) and concludes
"20 is the optimum"; the sweep below 20 does not appear anywhere.

Two things say it should be re-asked on cycles:

* **Study 20 is the standing counterexample** — −5.99 % cycles with
  `instructions:u` UP 0.19 %. `instructions:u` would have rejected a real win.
* **§1.2 supplies a mechanism.** `store_queue_rsrc_stall` is the engine's
  largest single dispatch blocker at **502.63 cycles/node**. A `REGS = 20` tile
  ends with 20 `zmm` stores, and Zen 4 splits each into two 32-byte store ops
  (study 39 §1 measured 2.043 store dispatches per line written) — so **every
  tile closes with a 40-store burst** into a 64-entry store queue, twice per
  fold, ~2.0 folds per node.

Narrowing the cap interleaves those stores with the next tile's loads instead of
bunching them. It is **bit-identical at every setting**: the same blocks in the
same order with the same arithmetic, and register `k` consumes its own distinct
32 bytes of each weight row either way, so no row is read more or fewer times.
`REGS ∈ {20, 10, 8, 5, 4}` are the exact divisors of `ACC_BLOCKS = 40` at or
below 20.

**The code-size cost, read off the binaries** (`size -A`, and `nm` summed over
`i8_deltas_into*`, `i8_into_shaped*`, `i8_deltas_dual*`, `i8_deltas_inner`) —
this matters because study 41 established that `ex_no_retire.empty` follows
instruction FETCHES at ≥ 0.79, and fetches ≈ executed code bytes ÷ 64:

| binary | `REGS` | `TILES` | `.text` | vs base | fold kernels | vs base |
|---|---:|---:|---:|---:|---:|---:|
| `base44` | 20 | 2 | 694 325 | — | 8 176 | — |
| `r10` | 10 | 4 | 697 221 | **+0.42 %** | 9 408 | +15.1 % |
| `r8` | 8 | 5 | 702 197 | +1.13 % | 12 426 | +52.0 % |
| `r5` | 5 | 8 | 704 981 | +1.54 % | 13 388 | +63.7 % |

`base44`'s `.text` is **694 325 bytes — identical to study 41's reference
binary**, an independent confirmation that this study's base is the same code
the front-end study measured. At study 41's slope, `r10`'s +0.42 % of code
buys back only ~0.016 % of cycles in extra front-end bubble: **not a material
cost at this size**, and not the explanation for anything below.

---

### 4.2 ★ THE INSTRUMENT FAILED, AND THE NULL IS WHAT CAUGHT IT

**A byte-identical copy of the base binary read `+2.363 % cycles`,
`+5.535 % load_not_complete` and `+6.067 % store_queue_rsrc_stall`, at 2/11
rounds on each.** A byte-identical binary cannot be 2.4 % slower. By rule 1 the
panel could not answer the question, and everything measured on it had to be
set aside rather than interpreted.

**The cause, found by bucketing every run by its POSITION IN ITS ROUND:**

| position in round | mean `cycles:u`/node | n |
|---|---:|---:|
| 0 (first) | **4282.4** | 13 |
| 1 | 4353.7 | 13 |
| 2 | 4329.5 | 12 |
| 3 (last) | **4449.4** | 12 |

**+3.9 % from the first slot of a round to the last** — a real within-round
drift, on a box where three other workloads come and go on the same timescale
as one round (4 labels × ~20 s).

And the free shuffle did not balance against it. Realised mean positions over
13 rounds:

| label | mean position | ideal |
|---|---:|---:|
| **`base44` (the REFERENCE)** | **0.77** | 1.50 |
| `r5` | 1.50 | 1.50 |
| `null44` | 1.62 | 1.50 |
| `r10` | 2.00 | 1.50 |

**The reference was systematically running early.** Every label is differenced
against it, so a reference in a cheap slot inflates *everything* — and it
penalises whichever arm drew the late slots hardest. At ~1.3 % per position,
`null44`'s 0.85-position deficit accounts for most of its impossible +2.4 %,
and `r10`'s 1.23-position deficit was **hiding a real effect at the same time
as it was faking a fake one**. That is the part worth remembering: a broken
randomisation does not just add noise in one direction.

**The fix, and why it is not "shuffle harder".** A shuffle cancels drift only
IN EXPECTATION; at n = 13 the realised imbalance was 0.73 positions on the one
label that matters most. `panel44.sh` now uses a **balanced Latin square**:
within each block of `NLAB` rounds every label occupies every position exactly
once, so any monotone within-round trend contributes *identically* to every
label — by construction, not by luck. The base order is re-shuffled per block,
which keeps study 25 §0.2's requirement that the design vary *which* orderings
appear (a single fixed rotation cancels monotone drift but not curvature).
Verified: mean position **1.50 for all four labels**.

**What this does and does not invalidate.**

* **Arm 1's verdict STANDS, and I checked rather than assumed.** Its panel ran
  3 labels with realised mean positions **0.91 / 1.18 / 0.91** against an ideal
  of 1.00, over a gradient that was only **+0.5 % and non-monotone**
  (4308.7 / 4272.7 / 4330.9). `hpf44`'s 0.27-position deficit is worth ~3
  cycles/node — 0.07 % against a measured +1.51 %. Its null also behaved (5/10,
  p = 1.00). The defect was specific to the LONGER 4- and 5-label rounds.
* **Every `cycles` figure from the arm-2 sweep is provisional** and is re-run
  in §4.4 on the fixed instrument.
* **The mechanism counters survive on margin.** Store-queue reductions of
  13–23 % at 10/10 and 11/11 are one to two orders of magnitude outside what a
  ±1.3 %-per-position gradient can manufacture, and `instructions:u` has a
  literally zero noise floor.

The broken panel is kept as `perf-artifacts-44/panel-shuffled-BROKEN.txt`
rather than deleted.

### 4.3 ★ A PRE-REGISTERED BOUND, written down BEFORE the panel finished

At n = 8 (2 complete Latin-square blocks) `r5` read **−5.44 % cycles
(trimmed) / −6.48 % (median)**. That is a beautiful number, so rule 6 says
attack it, and the cheapest attack is to ask whether the arm's own mechanism
counters can pay for it.

They cannot, on the same panel's numbers:

| term | cycles/node |
|---|---:|
| store-queue-blocked cycles removed (−26.4 % of 509.30) | **−134.3** |
| load-stall cycles removed (−7.33 % of 2558.77) | **−187.6** |
| — the two OVERLAP (a full store queue blocks dispatch of loads too), so the removal is **at most** −187.6, not their sum | |
| instructions added (+4.076 % of 5929.09 = +242) at a marginal 1.5 – 2.45 issue/cycle | **+99 … +161** |
| **predicted net** | **−29 … −92 c/node = −0.63 % … −1.96 %** |

**So the mechanism supports roughly −1 % to −2 %, and the panel at n = 8 is
claiming −5.4 %. One of them is wrong, and small `n` is the likelier
suspect** — the arm-2 sweep already demonstrated this exact failure mode,
shrinking from −2.49 % at n = 5 to −0.77 % at n = 12.

**Pre-registered, before the remaining rounds landed:** if `r5`'s contrast
against the null converges into the **−0.6 % … −2.0 %** band, the arm is real
and the mechanism is understood. If it stays near −5 %, then something other
than store-queue relief is doing the work and the mechanism story in §4.1 is
wrong regardless of whether the number is. §4.5 reports which happened.

### 4.4 The sweep — and the honest history of how the number moved

**This arm was nearly published off a lucky `n`, and the record of that is the
most useful part of the section.** The sweep panel's `cycles:u` estimate for
`r10` read **−2.49 % (5/5 rounds negative)** at n = 5, **−1.50 % (7/10)** at
n = 10, and **−0.77 % (7/12)** at n = 12 — shrinking monotonically as rounds
accumulated, with the sign count stuck at 7/12 (p = 0.77). That is precisely
the shape of study 29 §5.3's retracted result, which looked like a −1.67 %
cycles win at n = 13 and evaporated at n = 24.

So the sweep was **stopped and replaced by a longer, narrower panel** —
`base44`, `r10`, `r5`, byte-identical null, 45 rounds — carrying
`ex_no_retire.thread_not_selected` so the reduction can also report
`cycclean = cycles − thread_not_selected` (study 40 §3.2 measured that
subtraction cutting a byte-identical null's trimmed sd from 0.888 % to
0.624 %, i.e. 2.0× fewer rounds for the same resolution).

**The mechanism counters never wavered, and they are what the arm was aimed
at.** From the sweep panel (`panel-regs-sweep.txt`, n = 9–12, contrasts against
the same-round reference):

| | `r10` | `r8` | `r5` | **null** |
|---|---:|---:|---:|---:|
| `store_queue_rsrc_stall` | **−13.05 %** (10/10, p = 0.002) | −19.10 % (9/9, p = 0.004) | **−23.33 %** (10/10, p = 0.002) | −0.06 % (5/10, p = 1.00) |
| `load_queue_rsrc_stall` | **−8.40 %** (10/10, p = 0.002) | −9.84 % (9/9, p = 0.004) | **−14.40 %** (10/10, p = 0.002) | −0.18 % (5/10, p = 1.00) |
| `ex_no_retire.load_not_complete` | −2.11 % (8/10) | +0.67 % (2/9) | −4.17 % (8/10) | +0.62 % (4/10) |
| `instructions:u` | +1.257 % | +2.305 % | +4.087 % | +0.007 % (5/10) |

**Store-queue pressure falls monotonically with tile width — 13 %, 19 %, 23 %
— at 10/10 and 9/9, against a null that is a coin flip at 5/10.** The predicted
mechanism is real and large. `instructions:u` rises monotonically and with
exactly zero spread, as designed: this arm can only ever lose on that counter,
which is the whole reason it had never been tried.

**`r8` is anomalous and I cannot explain it.** It has the second-largest
store-queue reduction and the *worst* cycles reading (+0.67 %, 4/9 — a null),
breaking the monotone ordering that `r10` and `r5` both fit. Its fold kernels
are 12 426 B against `r10`'s 9 408 and `r5`'s 13 388, so code size does not
order it either. `TILES = 5` is the only odd trip count in the sweep, which is
a hypothesis and not a finding. It is recorded here as an unexplained
non-monotonicity, not smoothed away.

---

## 6. GATES

| gate | result |
|---|---|
| node identity, all 6 binaries, study frame | **5 001 562** — exact on `base44`, `hpf44`, `null44`, `r10`, `r8`, `r5` |
| d13 node identity, all 6 | **1 739 588** — exact |
| wl50 eval hash, all 6 | `2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4` — exact |
| `instructions:u` vs study 41 / settled frame | 5929.09 vs 5928.40 / 5923.909 (+0.012 % / +0.087 %) |
| `ls_any_fills_from_sys.all` vs study 42 | 585.86 vs 580.9 (+0.85 %) |
| `.text` of `base44` vs study 41's `b41` | **694 325 both** |
| byte-identical null, `instructions:u` | **0.000 %**, every round, every panel |
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets` | *(see §5)* |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | *(see §5)* |

D24 perft ritual **not** required: nothing under `movegen`, `board`, `zobrist`
or `attacks` is touched by any arm here.

---

## 7. BOX LOAD, STATED — AND ONE OPERATIONAL FINDING WORTH KEEPING

Every panel here ran with the D49 trainer live (~156 % CPU, ~5.8 GB RSS) and
two other measuring agents on the same 8-core box; 1-minute load average ranged
**3.6 – 8.8**, and available memory **7.0 – 9.3 GB** with swap effectively
exhausted (~40 MB free of 4095). No panel was started while a build of mine was
running. The protocol's defence is structural, not circumstantial: every label
is differenced against the byte-identical null **inside the same round**, with
the order Fisher-Yates shuffled per round.

**The operational finding: `taskset -c 2` is now actively harmful, because it is
the house default and every concurrent agent uses it.** Mid-study, runs that
take 9 s were taking ~40 s; `/proc/stat` sampling showed **cpu2 at 100.0 %
busy** while the box as a whole was at ~35 %, because another agent's binary was
pinned to the same logical CPU. The first ~5 rounds of the arm-2 panel were
discarded and the panel restarted on **cpu4**.

**And cpu10 is not the escape hatch it looks like** — it read 8.1 % busy, the
quietest logical CPU on the box, and
`/sys/devices/system/cpu/cpu2/topology/thread_siblings_list` says `2,10`. It is
core 2's SMT sibling; pinning there would have shared a physical core with the
100 %-busy contender. Physical-core busy (summing each sibling pair) is the
figure to pick on: core 0 = 24.1 %, core 4 = 28.3 %, core 6 = 34.3 %,
**core 2 = 108.1 %**.

---

## 8. WHAT I COULD NOT ESTABLISH

In order, most doubtful first.

1. **The per-symbol attribution in §2 was sampled on `bin44/b44`, which is
   study 41's reference binary (`main` @ `56fc4c3`), not on `base44`
   (`main` @ `7dd617df`).** The two retire 5928.40 and 5929.09 instructions per
   node — 0.012 % apart, i.e. the same code for this purpose — and study 42
   landed no shipped-path change between them, so I am confident the shares
   transfer. But it is a different binary and the `.text` md5 differs, so the
   §2 table is quoted as "the engine", not "this exact binary".

2. **The last column of §2 ("fraction of its own stalls that is load") is a
   ratio of two SAMPLED shares from two different `perf record` runs.** Each
   share carries sampling error, and dividing them squares it. It is sound for
   the one thing this study uses it for — spotting `forward_into` as the single
   inversion, a 2.1× effect against neighbours all sitting within ±10 % of the
   engine average — and it should not be read to two figures.

3. **`ex_no_retire.load_not_complete` is the memory subset of `not_complete`
   by AMD's definition, and I did not independently validate that definition
   on this box.** The commission asked for counters validated against a case
   whose answer is known; the validation I have is indirect but real —
   study 42's `ttnopf42` arm removed the TT prefetch and moved this counter
   +5.571 % (17/18) while moving `instructions:u` −0.363 %, which is a pure
   memory perturbation producing a pure response on this counter. That is
   consistent with the definition; it is not a proof of it.

4. **The engine-wide 0.8288 `load_not_complete / not_complete` ratio used to
   normalise §2 comes from the census (n = 6 per group, cross-group), not from
   a paired panel.** It is a scaling constant for one derived column and
   nothing else in the study depends on it.

5. **`r8`'s non-monotonicity is unexplained** (§4.2). Store-queue pressure says
   it should sit between `r10` and `r5`; cycles says it is the worst of the
   three. `TILES = 5` being the only odd trip count is a hypothesis I did not
   test.

6. **The execution residual (536.75 c/node, 11.50 %) is localised to
   `forward_into` but NOT to a mechanism inside it.** §2.2 rules out the
   obvious candidate with a source reference and stops there. This is the
   best-identified unbuilt lever in the study: a symbol taking 8.20 % of all
   retirement-blocked cycles while demonstrably not waiting on memory, whose
   dependency chain has not been found. `l3_dot` (a true reduction),
   `propagate_l2` (16×32 f32) and the WDL softmax are the places to look.

7. **Nothing here has been arbitrated by games.** Arm 2 is node-identical, so a
   fixed-node match is a dead heat by construction and only a real-time-control
   match could see it; at ~120 Elo per doubling of speed the effect size is a
   few Elo, which is thousands of games. The cycles panel is the arbiter this
   study can actually run, and §4.2 records how hard it fought back.

8. **Only ACC = 1280 was measured.** `REGS_CAP` is a global cap and changes the
   tile at every other architecture too (768 → 8, 1536 → 8, 2048 → 8 at a cap
   of 10). The `const _` asserts still machine-check that the tile divides the
   accumulator exactly at every width, so nothing can silently break — but the
   *benefit* is measured at one width only.

---

## 9. REPRODUCTION

```bash
# 0. the worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. build the arms (each serialises behind the shared build lock)
bash tools/perf44/build44.sh base44                 # REGS = 20, today's main
bash tools/perf44/build44.sh r10  regs10_44
bash tools/perf44/build44.sh r5   regs5_44
cp bin44/base44 bin44/null44                        # the byte-identical null

# 2. THE GATE, before any timing: node count, d13, wl50 eval hash
bash tools/perf44/ident44.sh base44 r10 r5 null44

# 3. the back-end census (12 unmultiplexed groups x 6 reps)
bash tools/perf44/census44.sh 6 base44
python3 tools/perf44/reduce44.py docs/research/perf-artifacts-44/census.txt

# 4. where the stalls are, by symbol
P44_CPU=4 bash tools/perf44/where44.sh ex_no_retire.load_not_complete 200003 base44
P44_CPU=4 bash tools/perf44/where44.sh ex_no_retire.not_complete      200003 base44

# 5. the arbitrating panel -- pick a QUIET PHYSICAL core first (see §7)
P44_CPU=4 \
P44_EV='cycles:u,instructions:u,ex_no_retire.thread_not_selected:u,ex_no_retire.load_not_complete:u,de_dis_dispatch_token_stalls1.store_queue_rsrc_stall:u' \
P44_PANEL=docs/research/perf-artifacts-44/panel-final.txt \
  bash tools/perf44/panel44.sh 45 base44 r10 r5 null44
python3 tools/perf44/med44.py docs/research/perf-artifacts-44/panel-final.txt \
  base44 r10 r5 null44
```

Picking the core (§7) — physical-core busy, summing each SMT sibling pair:

```bash
for c in $(seq 0 15); do
  echo "cpu$c $(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"
done
```
