# Perf study 41 — the front end, measured

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 16 GB ·
**Base:** `main` at `56fc4c3` (study 40 merged) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf41/` · **Artifacts:** `docs/research/perf-artifacts-41/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
core 2 with `taskset`. **Every binary here is node-identical at 5,001,562 and
eval-hash-identical at `2e1ede9521624dd8`**, with `instructions:u` matching to
**0.019 % or better** — gated per binary, not assumed
(`tools/perf41/identalign41.sh`).

**The reference reproduces study 40 on the two levels this study is built on**,
which is the frame check that matters most:

| per node | study 40 (n = 159) | this study |
|---|---:|---:|
| `ex_ret_brn_misp:u` | 24.092 | **24.336** (panel, n = 110) |
| `ex_no_retire.empty:u` | 192.907 | **191.757** (census, n = 6, quiet box) |

---

## ★ THE DECOMPOSITION

### The headline correction: mispredicts do not explain "a small fraction". They over-explain the whole thing.

The commission's premise was that branch mispredicts account for only a sliver
of the 192.91. **That is falsified by study 40's own two numbers multiplied
together**, and this study reproduces both of them independently:

> **24.336 mispredicts/node × 9.13 cycles each** (study 40 §3.3 — measured on a
> different arm family, and flagged *there* as a **lower** bound)
> = **222.2 cycles/node**, against an `ex_no_retire.empty` of **191.8–205.3**.

That is **≥ 108 % of the bucket before anything else is charged to it.** No
search for a missing cause was ever needed; the cause was already in the ledger,
unmultiplied.

### And the second cause is the same cause: `empty` is INSTRUCTION-FETCH-bound

| channel | how it was measured | share of `ex_no_retire.empty` | verdict |
|---|---|---:|---|
| **branch mispredicts** | 24.336/node × 9.13 c, study 40's arm | **≥ 108 %** | the bucket |
| **instruction fetch count** | dose response, §3 — local elasticity **0.79**, dose at 35 se | **0.79 % of the bucket per 1 % of fetches** | the *same* currency |
| **L1i misses specifically** | **REFUTED as the driver**, §3.2 | **not the mechanism** | the miss counter is the wrong instrument |
| **instruction TLB** | 0.244 L2-ITLB hits + 0.019 walks per node, direct | **≤ 1.5 %** | **DEAD — do not build the hugepage arm** |
| **decode / op cache bandwidth** | loop buffer delivers **exactly 0** ops/node; op cache 66.8 % | **~ 0** | wrong bucket by construction |
| **unattributed residual** | — | **none — the terms OVERLAP** | §2.4 |

**The residual is negative, and that is the finding.** Mispredicts and fetch
pressure are not two causes to be split. A mispredict's cost *is* a refetch.
Charge the marginal rate of either channel across the whole bucket and you get
~100 % twice, because the same cycles carry both labels.

### The one number the next study should take

**`ex_no_retire.empty` moves about four-fifths as fast, in percent, as
`ic_tag_hit_miss.all_instruction_cache_accesses`** — the count of 64-byte
instruction fetches, whatever causes them.

| dose | Δ fetch accesses | Δ `empty` | ratio |
|---|---:|---:|---:|
| `k416` (64 B blocks) | **+2.922 % ± 0.083 %** (35.4 se) | **+2.313 % ± 0.405 %** (5.7 se) | **0.79** |
| `k417` (128 B) | +6.972 % ± 0.084 % (82.7 se) | +3.578 % ± 0.379 % (9.4 se) | 0.51 |
| `k418` (256 B) | +16.363 % ± 0.101 % (161.5 se) | +6.683 % ± 0.420 % (15.9 se) | 0.41 |

The ratio *falls* as the dose grows, so **0.79 is a floor for the slope at the
shipping operating point**, and all three doses only push one way. In absolute
terms that is **0.30 cycles of `empty` per instruction fetch**
(0.79 × 205.333 / 538.42).

The lever this names is **fewer instruction fetches** — by mispredicting less,
or by executing fewer code *bytes*. It is emphatically **not** by moving code
around, and §3.2 is why.

---

## 0. THE DENOMINATOR, STATED — AND WHICH BINARY IT BELONGS TO

Every percentage below divides by a number measured **on this study's own
binary, in this study's own panel**, never inherited:

| `bin34/b41` — `main` @ `56fc4c3`, non-PGO, `-C target-cpu=native` | per node |
|---|---:|
| `cycles:u` | **4304.53** |
| `instructions:u` | **5928.40** |
| IPC | **1.377** |
| `ex_ret_brn_misp:u` | **24.336** |
| `ex_no_retire.empty:u` | **205.333** |

Source: `docs/research/perf-artifacts-41/dose3.txt`, median over **110 paired
rounds**, box carrying two perf studies and the D49 trainer. `cycles:u` can only
be inflated by contention, so 4304.53 is an upper bound on the quiet-box figure
and study 40's implied ~4104 is the lower one; **every cycle percentage here is
therefore a slight UNDER-statement**, the safe direction for a study whose job
is to bound a prize. The quieter 6-rep census (§1) reads
`ex_no_retire.empty = 191.757`, within 0.6 % of study 40's 192.907.

**Perf study 36's 8262.9 cycles/node is not used anywhere in this study.** No
result here was derived from it and none needs re-deriving —
`grep -rn 8262 tools/perf41 docs/research/perf-study-41-*` is empty.

**A binary-identity note, because an independent measurement of "the same
frame" reads differently and both numbers are right.** Measured on
`engines/canonical/archon` (the shipped PGO binary `engines/bin/archon-s25`,
built 2026-07-29 22:53, `.text` 699 893 B): 4489.8 cycles, 5659.0 instructions,
22.35 mispredicts per node over **5,001,566** nodes. That is a *different
binary*, and the node count proves it — PGO cannot change a fixed-node search's
node count, so 5,001,566 ≠ 5,001,562 means different source, not different
optimisation.

It does not move this study's conclusion, and the reason is worth stating:
study 40's 9.13 cycles-per-mispredict was measured on the **same non-PGO frame
as `b41`**, so `24.336 × 9.13` is like-for-like. Run the same arithmetic on the
shipped binary instead — `22.35 × 9.13 = 204.0` — and mispredicts still account
for essentially the whole bucket. **The headline survives either binary.**

---

## 1. THE CENSUS — the whole Zen 4 front end, one frame, six unmultiplexed groups

`tools/perf41/census41.sh`, 6 reps, `bin34/b41`. Zen 4 has 6 general PMCs and
every group is ≤ 6 events, so nothing is multiplexed (the reducer checks the
enabled/running ratio and refuses a scaled row).

| per node | median | spread |
|---|---:|---:|
| `cycles:u` | 4364.55 | 11.9 % |
| `instructions:u` | 5928.54 | 0.51 % |
| `branches:u` | 622.78 | 0.24 % |
| `ex_ret_brn_misp:u` | 25.97 | 3.9 % |
| **`ex_no_retire.empty:u`** | **191.76** | 14.7 % |
| `ex_no_retire.not_complete:u` | 2815.41 | 4.6 % |
| `ex_no_retire.other:u` | 220.46 | 4.0 % |
| `ex_no_retire.thread_not_selected:u` | 24.03 | 138 % |
| `ic_tag_hit_miss.all_instruction_cache_accesses:u` | **538.42** | 1.7 % |
| `ic_tag_hit_miss.instruction_cache_miss:u` | **164.64** | 2.8 % |
| `ic_cache_fill_l2:u` | 129.08 | 4.1 % |
| `ic_cache_fill_sys:u` | 3.26 | 12.7 % |
| `bp_l1_tlb_fetch_hit.if4k:u` | 1509.58 | 0.9 % |
| `bp_l1_tlb_fetch_hit.if2m:u` | 3.68 | 81 % |
| **`bp_l1_tlb_miss_l2_tlb_hit:u`** | **0.244** | 98 % |
| **`bp_l1_tlb_miss_l2_tlb_miss.all:u`** | **0.019** | 20 % |
| `de_src_op_disp.op_cache:u` | 5173.48 | 0.95 % |
| `de_src_op_disp.decoder:u` | 2566.24 | 1.3 % |
| **`de_src_op_disp.loop_buffer:u`** | **0.000** | — |
| `op_cache_hit_miss.op_cache_miss:u` | 372.47 | 1.5 % |
| `de_no_dispatch_per_slot.no_ops_from_frontend:u` | 5202.11 | 4.4 % |
| `de_no_dispatch_per_slot.backend_stalls:u` | 12840.21 | 5.1 % |
| `de_no_dispatch_per_slot.smt_contention:u` | 396.36 | 49 % |

### 1.1 The slot accounting closes to 0.03 %, which is what makes it quotable

Zen 4 dispatches 6 ops/cycle, so the four slot buckets plus the ops actually
dispatched must sum to 6 × cycles. They do:

```
dispatched      5173.48 + 2566.24 =  7739.72
front-end empty                      5202.11
back-end stalls                     12840.21
SMT contention                        396.36
                                    --------
total slots                         26178.40   / 6 = 4363.07 cycles/node
measured cycles:u                              =   4364.55  (-0.03%)
```

**The engine's top-down profile, per node:**

| | slots | share |
|---|---:|---:|
| ops dispatched | 7739.72 | **29.6 %** |
| **back-end bound** | 12840.21 | **49.0 %** |
| **front-end bound** | 5202.11 | **19.9 %** |
| SMT contention | 396.36 | 1.5 % |

Two things follow immediately. **The engine is back-end bound, not front-end
bound** — 49 % against 20 %. And the front-end slot figure (19.9 %) is **4.5×
larger than `ex_no_retire.empty`** (4.4 % of cycles), because `empty` counts
only the tail in which the ROB fully drains; the rest is partial starvation the
back end absorbs. §3.4 shows those two respond *differently* to the same dose,
which is the evidence that they are measuring different things rather than the
same thing at two scales.

### 1.2 Two channels are dead on the census alone, before any arm

* **Instruction TLB.** 1513.26 L1-ITLB fetch hits per node against **0.244**
  misses — a **99.984 %** hit rate — and only 0.019 full page walks. Pricing
  every L2-ITLB hit at 8 cycles and every walk at 50 gives **2.9 cycles/node**,
  **1.5 % of the bucket**, as a hard upper bound. **The `.text`-in-hugepages arm
  is dead and does not need building.** (`.text` is 694 KB = 170 pages, which
  already sits inside a 512-entry L2 ITLB — the geometry was never against us.)
* **The loop buffer delivers exactly 0 ops**, and the op cache covers 66.8 % of
  dispatch. A decode *bandwidth* limit cannot produce `ex_no_retire.empty` in
  the first place — a front end running at 4 ops/cycle instead of 6 starves
  slots but never empties the ROB — so it lands in the 19.9 % slot bucket, not
  in the 4.4 % one.

---

## 2. WHY MISPREDICTS ALREADY CLOSE THE BUCKET

### 2.1 Both inputs re-derived, not inherited (rule 8)

| | study 40 | here | agreement |
|---|---:|---:|---|
| `ex_ret_brn_misp:u`/node | 24.092 | **24.336** (panel n = 110) / 25.97 (census n = 6) | +1.0 % |
| `ex_no_retire.empty:u`/node | 192.907 | **191.757** (census n = 6) | −0.6 % |

### 2.2 The multiplication

`24.336 × 9.13 = 222.2` cycles/node of front-end bubble attributable to
mispredicts, against a bucket of 191.8–205.3. **108–116 %.**

### 2.3 Why 9.13 is the right rate to use, and why it is conservative

Study 40 measured it as a *marginal* rate: its `insertwin34` arm removed 1.088
mispredicts/node and bought 9.93 cycles of `ex_no_retire.empty` (14.4 se). It
then flagged, in its own §7.2, that `empty` cannot see recovery work that
overlaps with other retirement, so **9.13 is a lower bound on the true cost per
mispredict**. Using a lower bound and still landing above 100 % is the
conservative direction.

The one thing it is not is an *average* over all mispredict sites — it is the
marginal rate at one partial-insertion-sort inner loop. §3.3 gives an
independent handle on that from a completely different perturbation family.

### 2.4 So the residual is negative, and the two channels are one channel

Mispredicts ≥ 108 %, fetch pressure ~79 % per 1 % of fetches, iTLB ≤ 1.5 %,
decode ~0. Nothing is missing; the terms **double-count**. The physical picture
that fits every measurement here:

> A mispredict redirects the fetcher. The redirect is an instruction fetch. The
> engine issues **538 instruction fetches per node** and misses on **164** of
> them because the hot set is bigger than the L1i. Every marginal analysis of
> either channel therefore prices the same cycles.

This is not a hedge — it makes a falsifiable prediction, and §3 tested it: **an
arm that raises instruction fetches while leaving mispredicts alone must raise
`empty`.** It does, at 9.4 se, with mispredicts flat at 1.9 se in the *opposite*
direction.

---

## 3. THE DOSE — three gated layout arms, and the one that killed the pretty story

### 3.1 The instrument, and the null

The dose knob is LLVM's `-align-all-nofallthru-blocks=k`: every basic block
with no fall-through predecessor is forced to a 2^k boundary. LLVM names the
flag for exactly the property this needs — a block nobody falls into is entered
only by a taken branch, so **the padding is jumped over and never executed**.

| binary | flag | `.text` | hot set | nodes | eval hash | `instructions:u` |
|---|---|---:|---:|---|---|---:|
| `b41` | — | 694 325 | 2.88 × L1i | 5 001 562 | `2e1ede95…` | 29 654 692 261 |
| `a410` | `-align-all-functions=0` | **694 325** | 2.88 × | 5 001 562 | `2e1ede95…` | 29 654 691 828 |
| `k416` | `…nofallthru-blocks=6` | 967 781 | 3.46 × | 5 001 562 | `2e1ede95…` | 29 650 528 242 |
| `k417` | `…=7` | 1 325 157 | 4.20 × | 5 001 562 | `2e1ede95…` | 29 650 528 222 |
| `k418` | `…=8` | 2 166 373 | 5.79 × | 5 001 562 | `2e1ede95…` | 29 650 527 976 |

`-align-all-functions=0` is a **no-op in LLVM**, so `a410` comes out with a
`.text` md5 byte-identical to the reference — this study's free byte-identical
null. Instruction totals span **0.019 %**, inside the coordinator's 0.02 % frame
tolerance, and node counts and eval hashes are exact.

**The null is a coin flip on every counter it needs to be**, n = 110:

| counter | `a410` sign count | p |
|---|---|---|
| `ex_no_retire.empty:u` | 58/110 | 0.63 |
| `ic_tag_hit_miss.instruction_cache_miss:u` | 59/110 | 0.50 |
| `ex_ret_brn_misp:u` | 50/110 | 0.39 |

### 3.2 ★ `k416` refutes the icache-capacity story — and it was my story

I had a coherent, mechanism-rich narrative: the hot set is **94 438 bytes,
2.88 × the 32 KB L1i** (`tools/perf41/hotsize41.py`), the icache miss rate is
**30.6 %**, therefore `empty` is capacity-driven and the lever is footprint.
Rule 6 says attack the beautiful story hardest. `k416` did it for me
(n = 129, contrasts against the byte-identical null):

| `k416` − `a410` | contrast | se |
|---|---:|---:|
| `ic_tag_hit_miss.all_instruction_cache_accesses:u` | **+2.922 %** | 0.083 % (**35.4 se**) |
| `ic_tag_hit_miss.instruction_cache_miss:u` | **−0.968 %** | 0.140 % (**6.9 se**) |
| `ex_no_retire.empty:u` | **+2.313 %** | 0.405 % (**5.7 se**) |

**`empty` rose 2.3 % while the L1i miss count FELL 1.0 %, both at ≥ 5.7 se.** A
model in which `empty` is driven by icache misses predicts the opposite sign.
`k416` aligns every non-fall-through block to exactly one cache line, which
*reduces* line-spanning (hence fewer misses) while *increasing* the number of
distinct 64-byte fetches the same executed code needs — and `empty` follows the
fetches, not the misses.

**Consequence: the L1i miss counter is the wrong instrument for this bucket**,
and the "hot set is 2.9 × L1i so shrink it" program does not follow from the
data. Had I stopped at `k417` alone — which moves misses and fetches together
and would have given a tidy β = 0.45 cycles/miss at 9.4 se — I would have
published exactly that, and it would have been wrong.

### 3.3 The full dose table, n = 129, contrasts against the byte-identical null

| | `k416` (64 B) | `k417` (128 B) | `k418` (256 B) |
|---|---:|---:|---:|
| `ic…all_instruction_cache_accesses` | +2.922 % (35.4 se) | +6.972 % (82.7 se) | +16.363 % (161.5 se) |
| `ic…instruction_cache_miss` | −0.968 % (6.9 se) | +7.836 % (57.4 se) | +17.510 % (118.1 se) |
| **`ex_no_retire.empty`** | **+2.313 % (5.7 se)** | **+3.578 % (9.4 se)** | **+6.683 % (15.9 se)** |
| `de_no_dispatch…no_ops_from_frontend` | +0.337 % (1.2 se) | +2.751 % (10.7 se) | +9.523 % (32.6 se) |
| Δ`empty` / Δfetches | **0.79** | 0.51 | 0.41 |

And from the n = 110 panel that carried the mispredict counter, `k417` moves
`ex_ret_brn_misp` by **−0.289 % ± 0.151 %** — null at 1.9 se, and *negative*,
i.e. the wrong sign to explain a +3.6 % rise in `empty`. **The fetch channel is
identified independently of the mispredict channel.**

**The response saturates**, which is the second reason no extrapolation to
"zero fetch pressure" is defensible: 0.79 → 0.51 → 0.41 as the dose grows. All
three doses push the same way, so the honest statement is a *floor on the local
slope* (0.79), not a total.

### 3.4 `empty` and the front-end slot counter are not the same measurement

`k416` moves `empty` at 5.7 se and `no_ops_from_frontend` at **1.2 se — null**.
So the ROB-drain tail responds to a perturbation that the steady-state slot
count does not notice. Anyone treating the 19.9 % front-end slot figure and the
4.4 % `empty` figure as the same quantity at two scales will mis-price both.

### 3.5 Two more levers, killed with numbers

* **LLVM hot-cold splitting goes the WRONG WAY.** `hcs41`
  (`-hot-cold-split -enable-cold-section`) buys a 2.6 % smaller `.text` and a
  real `__llvm_cold` section, and costs **+1.485 % ± 0.229 % (6.5 se) more L1i
  misses**, with `empty` unmoved (+0.961 % ± 0.506 %, 1.9 se). It is also not
  instruction-inert (−0.090 %, 4.5 × the frame tolerance), so it was carried as
  a flagged point, not a clean layout arm.
* **Function alignment is a mispredict lever, not a fetch lever.** `a417`
  (`-align-all-functions=7`) moves `ex_ret_brn_misp` by **−0.853 % ± 0.147 %
  (5.8 se)** with L1i misses null (+0.219 %, 1.0 se) and `empty` at −0.557 % ±
  0.488 % (1.1 se). Real, consistent in sign, and far too small to land: −0.208
  mispredicts/node × 9.13 = **−1.9 cycles/node = −0.04 % of cycles**, for +7.5 %
  of `.text`.

---

## 4. WHY THERE IS NO COLD-CODE PRIZE EITHER

`tools/perf41/lines41.py` samples the frame, buckets every sample's IP into its
64-byte cache line, and counts the **distinct lines** each symbol contributes.
Lines × 64 is the executed footprint; allocated − executed is cold code sitting
inside hot functions — the only footprint an outlining arm can remove.

| symbol | allocated | executed | cold | exec % |
|---|---:|---:|---:|---:|
| `search::<NonPv>` | 16 089 | 13 504 | 2 585 | **83.9 %** |
| `qsearch::<NonPv>` | 6 193 | 4 544 | 1 649 | 73.4 % |
| `at_full` | 7 911 | 5 696 | 2 215 | 72.0 % |
| `refresh_threat_into_at` | 7 918 | 7 424 | 494 | **93.8 %** |
| `init_quiets` | 5 767 | 4 928 | 839 | 85.5 % |
| `i8_deltas_into` | 4 572 | 4 572 | 0 | **100 %** |
| top-26 total | 82 810 | **65 218** | 17 592 | 78.8 % |

**The executed hot footprint is 65 218 bytes = 1.99 × L1i**, and the 21 % that
is cold is *by definition already not fetched*. Moving it cannot reduce a fetch,
which is the currency §3 identified. `search::<NonPv>` — the obvious candidate,
16 KB — is 83.9 % executed.

(Read `EXEC` as a lower bound: sampling cannot see a line that runs a handful of
times in a whole frame. That makes `COLD` an **upper** bound on the prize, which
is the right direction for deciding not to attempt the refactor.)

### 4.1 Where the fetches actually come from: code is 5.32 bytes per instruction

`tools/perf41/density41.py` — symbol ranges from `nm`, instructions counted by
address containment:

| symbol | bytes | insns | B/insn |
|---|---:|---:|---:|
| `search::<NonPv>` | 16 089 | 3 383 | 4.76 |
| `at_full` | 7 911 | 1 627 | 4.86 |
| `refresh_threat_into_at` | 7 918 | 1 120 | **7.07** |
| `i8_deltas_into` (AVX-512) | 4 572 | 647 | **7.07** |
| **hot total** | **75 886** | **14 251** | **5.32** |

At 5.32 B/instruction a 64-byte fetch block holds **12.02** instructions at
best. The engine measures 5928.54 instructions against 538.42 fetches =
**11.01 per fetch, 91.6 % of ideal.** The fetcher is efficient; there is simply
a lot of long AVX-512 code.

**That is the whole mechanism in one line: fetches ≈ executed code bytes ÷ 64,
and `empty` follows fetches at ≥ 0.79.**

---

## 5. WHAT THIS PRICES, AND WHY IT IS SMALL

Using the floor slope and this study's own denominator (4304.53 cycles/node):

| move | Δ fetches | Δ `empty` | Δ cycles | Elo @ ~120/doubling |
|---|---:|---:|---:|---:|
| −1 % executed code bytes | −1 % | −0.79 % (−1.62 c/node) | **−0.038 %** | ~0.07 |
| −5 % executed code bytes | −5 % | −3.95 % (−8.1 c/node) | −0.19 % | ~0.33 |
| **every** L1i fetch free (impossible) | −100 % | −100 % (−205 c/node) | −4.77 % | **~8.5** |

**The entire front-end starvation bucket is worth ~8.5 Elo if it were abolished,
and every realistic lever inside it is worth well under half an Elo.** Against
that, the back end holds **49 % of dispatch slots** to the front end's 19.9 %.

**This is the close.** The 4.7 % question is answered, its causes are named, its
levers are individually killed with numbers rather than arguments, and its
ceiling is small enough that the next study should not be a front-end study.

---

## 6. WHAT LANDED

1. **`tools/perf41/` — the front-end kit** (9 tools): `census41.sh` (the whole
   Zen 4 front-end event set, six unmultiplexed groups), `reduce41.py`,
   `identalign41.sh` (frame gate: nodes + eval hash + exact instructions +
   `.text`), `buildalign41.sh` / `buildblk41.sh` / `buildhcs41.sh` (the layout
   dose), `slope41.py` / `attrib41.py` (dose-slope and multivariate
   attribution), `hotsize41.py` (allocated hot footprint), `lines41.sh` +
   `lines41.py` (**executed** footprint by cache line), `density41.py`
   (bytes/instruction), `where41.sh`, `live41.sh`, `waitpanel41.sh`.
2. **The decomposition**, above, with every channel measured and the residual
   named as overlap rather than left blank.
3. **Four levers killed with numbers**, each of which would otherwise have cost
   a future session a day: hugepage `.text` (iTLB hit rate 99.984 %), LLVM
   hot-cold splitting (6.5 se **worse**), cold-code outlining (executed
   footprint measured), and layout work in general (the dose saturates and
   `k416` inverts the sign of the assumed mechanism).
4. **`CLAUDE.md`'s NNUE gate pins, corrected and re-verified from source.**
   The doc named `eb981b11…4f4f37` / d13 `1985398`; the enforced gate in
   `tools/check.sh` is `2e1ede95…79c4` / `1739588`, and a from-source non-PGO
   `sota,arch-d48x` build reproduces check.sh **exactly** (both values run in
   §7 below). Cause: `859a442` moved both pins in `check.sh` correctly but
   landed its one-line CLAUDE.md edit on the *preceding*-pin line, which both
   left the current slot naming a superseded gate and overwrote perf study 8's
   real hash `5ca978e0…8b38`, leaving it paired with study 8's node count as a
   combination that never existed. Full chain restored.

**No `src/` change was made**, so `Bench: 587683` is unaffected — and that is
stated as a *verified* fact in §7, not an assumption.

---

## 7. GATES

| gate | result |
|---|---|
| `bin34/b41` `.text` md5 vs the landing tree | `8c9d05de3e9022e8b1b37f7875a7fe70`, **identical** |
| `cargo test` | pass |
| `cargo clippy --all-targets` | clean |
| `Bench:` | **587683** |
| NNUE eval hash (from source) | `2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4` |
| NNUE d13 nodes (from source) | **1739588** |

D24 perft ritual not required: nothing under `movegen`, `board`, `zobrist` or
`attacks` was touched (no `src/` file was touched at all).

---

## 8. WHAT I AM LEAST SURE ABOUT

In order, most doubtful first.

1. **The 0.79 slope is a floor from ONE dose family, all pushing one way.**
   Every dose here inflates fetches; none reduces them. The saturation
   (0.79 → 0.51 → 0.41) is exactly the shape that says the slope keeps rising as
   the dose shrinks, so 0.79 is conservative — but the true slope at zero dose
   is unmeasured, and §5's Elo table scales linearly with it. An arm that
   genuinely *reduced* executed code bytes would settle this in one panel, and I
   did not have one.
2. **`9.13 cycles per mispredict` is inherited from study 40**, and it is a
   marginal rate at one call site, not an average over all 24.336 mispredicts.
   I re-derived both *inputs* (§2.1) but not the rate itself. My own attempt to
   re-derive it from the dose family failed for a good reason — the layout doses
   move mispredicts by ≤ 0.3 %, so the coefficient came back ±13 on 14 (§3.3's
   panel) and is not identified. The headline needs only "≥ 100 %", which
   survives a rate anywhere above 8.4.
3. **`ex_no_retire.empty`'s absolute level moved 7 % between my own panels**
   (205.333 at n = 110 under load, 191.757 at n = 6 quiet) — the *contrasts* are
   noise-immune because they pair within round, but every absolute conversion in
   §5 inherits that 7 %. I quoted the higher (loaded) denominator, which
   understates the prize.
4. **The overlap between the mispredict and fetch channels is argued, not
   partitioned.** I show they cannot be independent (their marginal rates sum
   past 100 %) and I show the fetch channel is real with mispredicts held flat.
   What I cannot give is "X % of fetches are mispredict-driven refetches" — that
   needs an arm that changes mispredicts a lot at constant code size, which is
   precisely what study 40 spent 159 rounds discovering is hard.
5. **The census is n = 6.** The count events are stable (`instructions` 0.51 %,
   `ic` accesses 1.7 %, op-cache 0.95 %) and the slot accounting closing to
   0.03 % is strong corroboration, but `bp_l1_tlb_miss_l2_tlb_hit` at 0.244 has
   a 98 % spread. It would need to be wrong by **12×** to reach even 5 % of the
   bucket, so the iTLB verdict is safe; nothing else rests on the noisy rows.
6. **None of this has been arbitrated by games**, and by §5's own arithmetic
   nothing here is large enough that it could be.
