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
| ops dispatched | 29.6 % | **27.87 %** |
| **back-end bound** | **49.0 %** | **48.38 %** |
| **front-end bound** | **19.9 %** | **20.29 %** |
| SMT contention | 1.5 % | 3.46 % |
| closure (implied vs measured cycles) | −0.03 % | **−1.00 %** |

**The two load-bearing figures reproduce to 0.6 and 0.4 points.** The SMT row is
the one that moved, and it moved for a stated reason: this study ran on a box
carrying more concurrent agents than study 41's (see §7), and
`ex_no_retire.thread_not_selected` is the counter study 40 measured swinging
−56 % to +203 % between rounds. My closure is −1.00 % against study 41's
−0.03 % for the same reason — my twelve groups were collected over ~25 minutes
of a *changing* load, and each group carries its own `cycles:u`.

**So the 49 % stands, and this study is a decomposition of it, not a re-audit.**

### 0.2 The denominator, stated, and which binary it belongs to

| `bin44/base44` — `main` @ `7dd617df`, non-PGO, `-C target-cpu=native` | per node |
|---|---:|
| `instructions:u` | **5929.09** |
| `cycles:u` (census, box load ~5–9) | **4668.94** |
| `ex_no_retire.load_not_complete:u` | **2596.93** |
| `ex_no_retire.not_complete:u` | **3133.67** |
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
| **`load_not_complete` — MEMORY** | **2596.93** | **55.62 %** |
| `not_complete` − `load_not_complete` — **EXECUTION** | **536.75** | **11.50 %** |
| `other` | 225.51 | 4.83 % |
| `empty` — front end (study 41's whole subject) | 209.66 | 4.49 % |
| `thread_not_selected` — SMT | 60.97 | 1.31 % |
| **— all no-retire cycles** | **3629.81** | **77.74 %** |
| **RETIRING cycles** | **1039.14** | **22.26 %** |

Closure: `ex_no_retire.all` measured **3673.27** against the parts' sum
**3629.81** — **a −1.18 % residual**, from the same cross-group drift as §0.1.

**The answer to the commissioned question, in one line: the 49 % back-end
bucket is 83 % memory and 17 % execution.** Not "memory-bound" as a mood —
`load_not_complete` alone is **more than half of every cycle the engine
spends**, and it is 4.8× the entire front-end bucket study 41 spent a session
on.

### 1.2 The four candidates, resolved with numbers

The commission named four. Three of them are now priced and two are dead.

| candidate | the measurement | verdict |
|---|---|---|
| **Memory / load stalls** | `load_not_complete` = **2596.93 c/node, 55.62 % of cycles** | **THE ANSWER.** §2 says which memory. |
| **Execution port pressure** | 1282.60 512-bit + 33.75 256-bit + 6.06 scalar FP uops/node. On Zen 4 a 512-bit op occupies an FP pipe for 2 cycles ⇒ **2605 pipe-cycles/node over 4 pipes = 651 c/node = 13.9 % occupancy.** Corroborated directly: `fp_sch_rsrc_stall` = **196.41 c/node (4.2 %)** | **DEAD. The FP pipes are 86 % idle.** Do not build a port-pressure arm. |
| **Dependency chains / latency** | the execution residual, **536.75 c/node = 11.50 %** | **REAL BUT SECONDARY**, and §2.2 localises it to ONE symbol. |
| **Store-buffer / write pressure** | `store_queue_rsrc_stall` = **502.63 c/node = 10.76 % of cycles** — the **largest single dispatch-blocking resource in the engine**, ahead of the load queue at 442.11 | **REAL.** It is what §4's arm targets. |

The full dispatch-blocking table, since it is the first time anyone has looked
(these are cycles in which dispatch was blocked on that structure; they overlap
with each other and so do not partition):

| resource | cycles/node | % of cycles |
|---|---:|---:|
| **store queue** | **502.63** | **10.76 %** |
| load queue | 442.11 | 9.47 % |
| FP register file | 329.07 | 7.05 % |
| integer physical register file | 259.55 | 5.56 % |
| **FP scheduler** (the port-pressure signature) | **196.41** | **4.21 %** |
| retire tokens (ROB full) | 194.91 | 4.17 % |
| integer scheduler 1 | 149.73 | 3.21 % |
| integer schedulers 0 / 3 / 2 | 78.35 / 68.31 / 35.96 | 1.68 / 1.46 / 0.77 % |
| FP flush recovery | 5.83 | 0.12 % |
| taken-branch buffer | 0.013 | 0.000 % |

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
| load-stall cycles per load dispatched | **1.139** | `load_not_complete` / `ld_dispatch` |
| load-stall cycles per demand fill | **5.31** | `load_not_complete` / demand fills |

**This is the shape of the problem, and it is not the shape people assume.**
The engine is not waiting on a handful of catastrophic DRAM misses — there are
only **7.9 DRAM fills per node**, and the average missing line comes back in
**54 cycles** with **5.7 of them in flight at once**. It is waiting **1.14
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
transformer fold kernels alone are **26.43 %** — worth **686 … 767 cycles/node**
across the denominator range.

### 2.2 One symbol is execution-bound, and only one

**`SotaHead::forward_into` takes 8.20 % of `not_complete` while taking only
3.87 % of `load_not_complete`** — its share of total stalls is **2.1× its share
of load stalls**, the only inversion in the table. That is the signature of a
dependency chain, and it has a known mechanism: `l1_dot_inner` is a chain of
`vpdpbusd` (Zen 4: ~4-cycle latency, ~1/cycle throughput) and perf study 21
already added `CHAINS` independent accumulator sets for exactly this reason —
but only for widths with `BLOCKS < 4`. At D48X (`L2 = 128`, `L2_BLK = 16`)
`BLOCKS = 8`, so `CHAINS = 1` and study 21's fix is switched off by its own
condition.

**This is the single best-identified lever this study did not have time to
build**, and §8 records it as such rather than as a finding.

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

*(populated at the end)*
