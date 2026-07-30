# ARCHON perf study #37 — the threat stream, and the `#[cold]` that was true once

**Date:** 2026-07-30 · **Base:** `main` at `5e42622` (study 36 merged) ·
**Instruments:** `tools/perf37/` on top of study 34's panel kit ·
**Artifacts:** `docs/research/perf-artifacts-37/`

---

## 0. HEADLINE

### 0.1 The `#[cold]` was not a mistake. It was TRUE, and study 32 falsified it.

`git log -S` settles it in two commits, and the answer is more interesting than
"someone got it wrong".

* **`2703512` — perf study 25 arm 1** created the shape dispatch and wrote the
  attribute. At that commit the fold family was still split **by weight type**:
  HalfKA folded `i16` rows through `apply_i16_deltas_into`, and the threat
  channel had its own separate `i8` kernel (`sota_threats.rs` imported
  `apply_i8_deltas_into`). The `#[cold]` generic was the fallback of the *i16*
  dispatch, so its only callers really were "the test suite's long feature lists
  and the EMPTY case". **The label was correct when it was written.**
* **`aed3016` — perf study 32** quantised HalfKA to `i8`. Its own commit message
  states the merge: *"Both feature transformers now fold i8 rows, so the i16
  fold family is deleted and the i8 one gains study 25's shape-specialised
  const-generic dispatch."* From that commit on, **every threat fold enters the
  shared kernel, misses all four `(≤2, ≤2)` shapes (threat lists average 4.658
  rows) and lands in the function marked cold.** The attribute came across with
  the code; the doc comment was even updated to say "for the threat channel's
  longer lists", and the `#[cold]` above it was not revisited.

Study 36 then measured the consequence: **12.38% of engine cycles in a function
labelled cold.**

### 0.2 What the two attributes cost, measured SEPARATELY

The commission's instinct to split them was right, and the split produced a
result no panel was needed for:

| arm | what changed | verdict |
|---|---|---|
| `noinl` — drop `#[inline(never)]`, keep `#[cold]` | **nothing at all** | the binary is **byte-identical to `base` (same md5)**. `#[cold]` already forbids inlining, so `#[inline(never)]` on this function is dead text. |
| `nocold` — drop `#[cold]`, keep `#[inline(never)]` | block placement in the CALLER | the generic's own 3616 bytes are bit-identical and at the same address; the only diff in the whole binary is the ORDER of the shape-dispatch blocks inside `i8_deltas_into`. |
| `noboth` — drop both | the generic is **inlined** | the symbol disappears; `i8_deltas_into` grows 4576 → 8160 bytes. |

Measured on the study frame, 12 paired within-round rounds against a
byte-identical null (`docs/research/perf-artifacts-37/panel-core.txt`):

| label | `instructions:u`/node | `branches:u`/node | `L1-dcache-loads:u`/node |
|---|---:|---:|---:|
| `null` (byte-identical) | **−0.000%** [0.00, 0.00] | **+0.000%** [0.00, 0.00] | +0.071% [−0.32, +0.24] |
| **`nocold`** | **−0.013%** [−0.01, −0.01] | **−0.120%** [−0.12, −0.12] | −0.015% |
| `noboth` | **+0.137%** [+0.14, +0.14] | −0.092% [−0.09, −0.09] | **+0.408%** [−0.04, +1.12] |
| `thp` (§4) | +0.000% | +0.000% | −0.050% |

Reference levels: 5905.510 instructions, 619.318 branches, 3004.77 L1d loads
per node. Every arm is node-identical at **5,001,624** and eval-hash-identical
at `eb981b119bd66d6e`.

**So the answer to the headline question is:**

* **`#[cold]` costs 0.744 branches and 0.744 instructions per node** — one
  branch per threat fold, near enough (0.744 / 0.857 folds = 0.87), which is
  exactly the taken jump into the sunk block that removing the hint turns into a
  fallthrough. −0.120% of all engine branches, zero spread, against a null with
  zero spread.
* **`#[inline(never)]` costs nothing and does nothing — while `#[cold]` is
  present.** The md5 is the measurement.
* **But it stops being inert the moment `#[cold]` goes**, and then it is
  *earning its keep*: `noboth` lets LLVM inline the 3616-byte generic into
  `i8_deltas_into` (4576 → 8160 bytes) and that costs **+0.137% instructions and
  +0.408% L1d loads** — the only arm in this study to clear the load
  instrument's ±0.3% floor, and in the wrong direction — to buy a branch delta
  *smaller* than `nocold`'s.

The landed change is therefore **`nocold`: remove `#[cold]`, keep
`#[inline(never)]`**. That is not the "both attributes are wrong" answer the
shape of the problem suggests, and measuring them separately is the only reason
it is visible.

### 0.3 The threat channel's row width, layout and access pattern

The commission asked what study 32's i8 lever's equivalent is for the threat
channel. **There is none, because the lever was already spent before study 32
existed**: FullThreats weights are `i8` **on disk**, in the CENN v2 format, and
always have been (`sota_loader.rs`: `threat_w: i8[threat_dims · acc_width]`,
read straight through `read_i8s_aligned`). Study 32's win was pulling HalfKA
*down* to the width threats were already at. Establishing that first is what
kept this study off a dead arm.

What the channel actually is:

| property | value |
|---|---|
| features | **60,720** (2.7× HalfKA's 22,528) |
| row | `ACC` = 1280 **bytes** (i8), 20 cache lines, contiguous |
| table | **77.7 MB**, feature-major |
| index layout | `lut1[att][atd][from<to] + offsets[att][from] + lut2[att][from][to]` — **attacker-block major**, then victim-relation, then from-square, then to-ordinal |
| rows/node | **3.993** (0.857 folds × 4.658 rows), plus 0.330 empty-delta 2560 B copies |
| kernel | `TILES = 2` × `REGS = 20` zmm; the row list is walked once per tile |

The layout matters for the access pattern and it is the pessimal one for
locality: the features that fire together come from **one move**, so they share
a from- or to-square and differ in attacker/victim type — and attacker/victim is
the OUTERMOST axis, separated by `piece_offset[attacker]` rows. A queen
attacker's single victim-relation sub-block is already ~1.86 MB. The ~4.7 rows
of one fold are therefore scattered over tens of megabytes, every time.

---

## 1. METHOD, AND THE BOX

### 1.1 The box, and it was genuinely quiet — for once

This study is the first in the 33-37 run that got an idle machine, and the
difference is visible in the instrument itself. All measurements in §2 and §3
were taken in **one** window with the D49 trainer as the only other load
(load average 4.3–4.8 against 20–26 for studies 34/36).

The signature: `cycles:u` on the reference binary reads **4289.1 per node here
against study 36's 8262.9** on the same frame and the same tree — a factor of
**1.93**. That is study 33's "absolutes inflate ~1.9× under contention"
reproduced almost exactly, from the other side, and it is a warning attached to
every cycles *level* any of these studies has printed. (Instructions reproduce
study 36's landed tree to seven digits over the same interval: **5905.510** here
against its 5905.511.)

Source reading, `git log -S` archaeology and arm design were all done *before*
the window opened, on an explicitly loaded box, exactly so the window could be
spent on measurement.

### 1.2 Instruments

| tool | used for |
|---|---|
| `tools/perf34/build34.sh` | every labelled binary (MemoryMax=6G scope, `-j2`, ≥3000 MB gate) |
| `tools/perf34/ident34.sh` | the identity gate: node count + wl50 eval hash |
| `tools/perf37/patch37.py` | the four attribute variants, so they differ by the attribute lines and nothing else |
| `tools/perf37/panel37.sh` + `med37.py` | the paired within-round panel, event set selectable |
| `tools/perf37/symcount37.sh` | STATIC per-symbol instruction census out of the disassembly |
| `tools/perf37/vmflags37.sh` | what the kernel did with the huge-page hint |

### 1.3 Null first

Every panel carries `null` — a byte-identical `cp` of the reference binary — as
its own label in the same rotation, and `med37.py` differences inside the round
before medianing.

| counter | null median | null spread | verdict |
|---|---:|---|---|
| `instructions:u`/node | **−0.000%** | **[0.00, 0.00]** | exact — quotable |
| `branches:u`/node | **+0.000%** | **[0.00, 0.00]** | exact — quotable |
| `L1-dcache-loads:u`/node | +0.071% | [−0.32, +0.24] | ±0.3% floor |
| `ex_ret_brn_misp:u`/node | −0.253% | [−2.05, +4.87] | cannot adjudicate |
| `cycles:u`/node | **+0.784%** | **[−5.39, +5.29]** | **dead** |

**A fifth independent confirmation that cycles cannot resolve a sub-1% change on
this box — and the first one taken on an IDLE machine.** Studies 33, 34, 35 and
36 could each be answered with "your box was loaded". This null ran with load
average 4.3, and a byte-identical binary still moved +0.784% over a 10.7-point
range. Contention is not the cause; it is the counter.

This study therefore quotes instructions and branches, treats loads as a ±0.3%
instrument, and makes no cycles claim anywhere.


---

## 2. ARM 1 — the attribute pair, and why it had to be split

### 2.1 `#[inline(never)]` is inert, and the md5 is the whole measurement

`bin34/noinl` — `#[cold]` kept, `#[inline(never)]` removed — is
**byte-for-byte the same file as `bin34/base`**:

```
10edaf9df249f3e707784ea17cf09345  bin34/base
19798abf79732d92c9fbcb2c90af699f  bin34/nocold
10edaf9df249f3e707784ea17cf09345  bin34/noinl     <- identical to base
cedb1651fdab59c987135bf0f370211f  bin34/noboth
```

LLVM's inliner refuses to inline a function carrying the `cold` attribute, so
the second hint had nothing left to say. **This is the strongest possible form
of a null** — not "below the noise floor", but *the same bytes* — and no panel
can improve on it. It is also the reason the commission's instruction to measure
them separately mattered: lumped together, the pair would have been reported as
one change worth `noboth`'s numbers, which are the *wrong sign* on instructions.

### 2.2 `#[cold]` is pure block placement — but in the CALLER, not the text layout

The obvious story for `#[cold]` is a hot/cold text split: the function is
banished to `.text.unlikely` and its caller pays an icache/iTLB hop. **That is
not what happens here**, and the binaries say so:

* `i8_deltas_into_generic` is at the **same address `0x75420`** in `base` and
  `nocold`, with the **same 3616 bytes and the same 714 instructions** and an
  identical mnemonic histogram. No section migration, no `optsize` effect on the
  loop — the unroller made the same choices either way. (I predicted the
  opposite: LLVM's `ProfileSummaryInfo` reports a `cold` function as
  "optimize for size", and this kernel's tile walk is documented as depending on
  LLVM choosing to unroll it. **The prediction was wrong**; the disassembly is
  bit-identical.)
* The *entire* difference between the two binaries is inside the caller
  `i8_deltas_into`, whose symbol is the **same 4576 bytes** — the four shaped
  blocks and the generic-call block are laid out in a different order.

What `cold` actually does on this dispatch:

```asm
cmp  $0x1,%r8                 ; subs.len() == 1 ?
je   <shape (?,1)>
cmp  $0x2,%r8                 ; subs.len() == 2 ?
jne  <trampoline> ; --------> jmp i8_deltas_into_generic
```

A threat fold averages 4.658 rows, so it fails both tests and takes that last
branch — **every time**. `#[cold]` is the instruction telling LLVM to lay the
block out as the one that is *not* taken. Removing it makes the generic path the
fallthrough and deletes one taken branch per threat fold.

The arithmetic closes: **0.744 branches/node removed ÷ 0.857 threat folds/node =
0.87 branches per fold.** One branch, near enough, and the residual is the
handful of folds that do land on a shape.

### 2.3 Removing `#[inline(never)]` as well is a LOSS

`noboth` inlines the generic into `i8_deltas_into` (4576 → 8160 bytes, the
separate symbol gone) and measures **+0.137% instructions, +0.408% L1d loads**
for a branch win *smaller* than `nocold`'s. The loads figure is the only number
in this study that clears the ±0.3% load floor and it points the wrong way.

So the pair is not "one wrong hint and one redundant hint". It is:
**`#[cold]` is wrong; `#[inline(never)]` is inert while `#[cold]` stands and
becomes load-bearing the moment it is removed.** Landing `nocold` — remove one,
keep the other — is the only one of the four combinations that is a win, and it
is not the one the phrase "a function carrying 12.38% of cycles is marked cold"
points at.

---

## 3. ARM 3 — the paired row loop, BUILT AND REJECTED

The threat fold's per-row loop control, read off the shipped disassembly rather
than estimated:

```asm
75510:  mov  (%rcx,%r15,1),%r12     ; the feature index
        add  $0x8,%r15              ; cursor
        lea  (%r12,%r12,4),%r12     ; x5   |
        shl  $0x8,%r12              ; x256 |  = feature * 1280
        ... 20 x vpmovsxbw + 20 x vpaddw ...
75637:  cmp  %r15,%r8
        jne  75510
```

**Six instructions per row per tile**, not the four perf study 36 estimated from
a static reading — so at 3.993 rows/node over `TILES = 2` the control alone is
**~47.9 instructions and ~8.0 branches per node**, i.e. 0.81% of engine
instructions and 1.29% of engine branches. A real target.

The arm: walk the add and sub lists together over their common prefix
(`split_at(min(na, ns))`), dropping the unequal tail into the existing
single-sided loops. Two cursor bumps, two compares and two back-edges per pair
become one of each. It is bit-identical — wrapping `i16` add/sub is a
commutative group operation per lane, so only the multiset of signed rows
matters — and it gated clean: node-identical at 5,001,624, hash
`eb981b119bd66d6e`.

**Measured, 12 paired within-round rounds, reference `nocold`:**

| counter | `paired` | null | spread |
|---|---:|---:|---|
| `instructions:u`/node | **+0.147%** | +0.000% | zero, both |
| `branches:u`/node | **+0.243%** | +0.000% | zero, both |
| `L1-dcache-loads:u`/node | **+0.302%** | +0.022% | clears the floor |

**A loss on all three exact counters.** Reverted.

### 3.1 Why it lost — and the general lesson

The static census is the explanation, and it inverts the intuition:

| kernel | bytes | instructions | `vmovdqu64` | `vpaddw` |
|---|---:|---:|---:|---:|
| two loops (shipped) | 3616 | 714 | 220 | 60 |
| paired (rejected) | **1744** | **405** | **40** | 40 |

The two-loop form is twice the size because **LLVM clones the whole tile body
across the empty-list cases and unrolls both tiles**, and that duplication is
worth more than the loop control it duplicates. The paired form is tidier,
smaller, and slower: one tile body, a real tile loop, and — because two loops
became three — two empty-tail entry tests that the common `na == ns` case did
not previously pay.

**In this kernel the duplicated, specialised shape IS the optimisation.** Any
future restructuring argued from a control-instruction count has to clear that
bar, and the ⚠ banner now on `i8_deltas_into_generic` says so with the numbers.

This is a failed prediction of mine, recorded at full size: I priced the arm at
−0.23% instructions and −0.74% branches and it delivered +0.147% and +0.243%.

---

## 4. ARM 2 — huge pages for the weight tables: BUILT, VERIFIED CORRECT, MEASURED NULL, NOT LANDED

### 4.1 The hypothesis, and why it looked like the right lever

Study 36's anomaly is that `i8_deltas_into_generic` runs at **2.70 cycles per
instruction** against an engine-wide 1.0, from the **largest** table, while its
*accumulator* amortisation is the better of the two channels (1099 B/row against
HalfKA's 2249). The obvious remaining difference is the table: **77.7 MB against
28.8 MB**, and Zen 4's L2 STLB is 3072 entries — **12 MB of reach at 4 KiB**
against a ~106 MB weight working set. A 1280-byte row straddles a 4 KiB page
31% of the time, so ~7.1 rows/node cost ~9.3 page touches, essentially all
distinct. At 2 MiB the same 106 MB is 53 entries.

This was also *open work*, not a new idea: perf study 4 item #8 left it
**RE-OPENED** ("if `madvise`, add `madvise(MADV_HUGEPAGE)` on the big
allocations"), and `tt.rs` still carries `// later: huge-page-backed buffer
(doc 08 §3.13 phase 2)`.

### 4.2 What was built

A `src/hugepage.rs` with one `extern "C"` `madvise` prototype (no new crate
dependency — `std` already links libc, and the crate's zero-dependency rule
stands), plus a hook in `Aligned::from_iter_len` that aligns any payload of at
least one huge page to 2 MiB and advises the range **before** the fill, so faults
allocate huge pages directly instead of waiting on `khugepaged`.

It is preserved as `docs/research/perf-artifacts-37/thp-arm.patch` — it applies
to `5e42622` and rebuilds in one command.

### 4.3 It works. The kernel refuses it.

`tools/perf37/vmflags37.sh` on the built binary, at the UCI prompt with the net
loaded:

```
7f16ad400000-7f16b1e1f000  size=  75900 kB  AnonHuge= 0 kB  VmFlags: … ac sd hg
7f16b2000000-7f16b3b80000  size=  28160 kB  AnonHuge= 0 kB  VmFlags: … ac sd hg
```

Both VMAs carry **`hg`** — the advice was accepted and applied to exactly the
right ranges (75,900 kB = `60720 × 1280` and 28,160 kB = `22528 × 1280`, to the
byte), and the mapping is 2 MiB-aligned. **`AnonHugePages: 0`.** The reason is
one line of `/proc/buddyinfo`:

```
Node 0, zone Normal   516 29025 13011 4442 1506 544 240 52 6 0 0
                                                          ^order-9 (2 MiB): 6
```

Six free 2 MiB blocks in the whole Normal zone after eleven hours of trainer and
match load, `defrag` at `[madvise]`, and `/proc/vmstat` showing
`thp_fault_fallback 261114`. The kernel tries direct compaction on every fault
in these VMAs and loses.

### 4.4 The measurement, and a number worth more than the arm

TLB panel, 8 paired within-round rounds
(`docs/research/perf-artifacts-37/panel-tlb.txt`):

| counter | base level/node | `thp` | null | verdict |
|---|---:|---:|---:|---|
| `ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit` | **0.0000** | **0.0000** | 0.0000 | discrete, unambiguous: **no huge page, ever** |
| `ls_l1_d_tlb_miss.all_l2_miss` (page walks) | **2.2465** | −3.043% [−14.1, +9.0] | +2.303% [−15.1, +7.5] | inside the null |
| `ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit` | 46.245 | −2.840% | +0.258% | inside the null |
| `instructions:u` | 5905.51 | **+0.000%** | +0.000% | bit-identical work, as designed |

**And the base level is the finding.** I predicted ~9–11 page walks per node
from STLB-reach arithmetic. The engine takes **2.25**. Of 46.2 L1-DTLB misses
per node, 95% hit in the L2 STLB; only 2.25 walk the page tables. Even a
*perfect* 4 KiB→2 MiB conversion could have removed only those 2.25, which at
~20–30 cycles is well under 1% — a **third or less** of what I priced.

**So the threat channel's 2.70 cycles/instruction is not a TLB problem, and
this closes perf study 4 item #8 as a lever rather than leaving it open.** The
STLB is coping with 106 MB far better than its 12 MB nominal reach suggests
(page-walk caches, plus the fact that only ~7 rows/node are drawn at all).

### 4.5 Why it is NOT landed

The crate is `#![deny(unsafe_code)]` and admits **"measured, documented"**
exceptions. An `unsafe extern "C"` FFI block whose measured effect is exactly
zero — on an instrument whose own null moves ±15 points, against a mechanism
counter that reads a flat 0.0000 — does not meet that bar. Landing it would put
unsafe code in the engine on the strength of an argument rather than a
measurement, which is precisely the trade this project has spent five studies
learning not to make.

**What would change the verdict**, in order: a box with free order-9 pages
(`grep -c . /proc/buddyinfo`, column 10 of zone Normal ≫ 6), and then a re-run
of §4.4's panel. §4.4's own base level says the ceiling is under 1% even then,
so it should be re-opened only if something else makes page walks expensive.

---

## 5. PRICED, AND NOT TAKEN

| candidate | the number that decided it |
|---|---|
| **i8 rows for the threat channel** (study 32's lever) | **Already spent.** FullThreats weights are i8 *on disk* in CENN v2 and always were; study 32 pulled HalfKA down to the width threats already had. §0.3. |
| **Shape-specialising the threat fold** | Study 36 priced the loop control at 32 i/n; the real figure is **47.9 i/n and 8.0 br/n** (§3, 6 instructions per row per tile, read off the disassembly). But §3.1 measured that restructuring this kernel *loses*, because LLVM's cloning of the tile body is worth more than the control it duplicates, and a `(adds, subs)` matrix up to 27 rows cannot be enumerated anyway. |
| **Pre-scaled row offsets** (store `feature * ACC` at resolve time, kill the `lea`+`shl`) | Arithmetically −8 i/n (−0.135%): the producer would pay 2 instructions per row once where the kernel pays them once per tile. Not built — §3.1 says a restructuring of this loop must be measured, not argued, and the same cloning risk applies. Left as the best-priced remaining candidate. |
| **The `thr_copy` empty-delta 2560 B copy**, 0.330/node | The largest exact-counter prize left in the channel: 40 zmm loads + 40 stores per copy = **~26 i/n (0.45%)** and ~26 load events (0.9%). Needs a per-perspective alias through `ThreatContext::at_full`, whose signature hands out `&[[i16; ACC]; 2]` — both perspectives from one entry — so an alias that differs per perspective cannot be returned without changing the merge kernel's parameter shape. Real, and a whole arm on its own. |
| **A locality permutation of the threat table** (§0.3: attacker/victim is the outermost axis, so one move's ~4.7 rows scatter over tens of MB; a load-time row permutation with a matching `threat_index` is bit-identical) | Superseded by §4.4 before it was built: the access pattern's TLB cost is 2.25 walks/node total, and §3.1 says the kernel is not control-bound either. The permutation's remaining upside would be DRAM row-buffer locality, which no instrument on this box can resolve. |
| **Row prefetch** | Settled dead three times — study 21's wrong-address control (+0.01%), study 29 §5 at the `append_changed_indices` site (−0.50% misses against a +0.14% null), and the in-tree note at `resolve_both`. Not re-run. |

---

## 6. WHAT LANDED

One change: **`#[cold]` comes off `i8_deltas_into_generic`; `#[inline(never)]`
stays.**

* **−0.120% of all engine branches** (619.318 → 618.573/node) and **−0.013% of
  all engine instructions**, zero spread on both, against a null of exactly zero
  on both, 12 paired within-round rounds on an idle box.
* Node-identical (**5,001,624**) and eval-hash-identical (`eb981b119bd66d6e`)
  — it is a code-placement change and cannot be anything else.
* The mechanism is closed, not asserted: 0.744 branches/node removed ÷ 0.857
  threat folds/node = **0.87 branches per fold**, i.e. the one taken jump into
  the sunk block, and the residual is the folds that do land on a shape.

Two things go with it, and they are the durable part:

1. **The doc comment now records the archaeology** — that the attribute was
   *true* at study 25 and was falsified by study 32's channel merge — so the
   next person who reads `#[inline(never)]` on this function knows why it is
   there and what the pair actually costs.
2. **A ⚠ banner on the two row loops** carrying §3's rejected arm with its
   numbers, so the next "these two loops should obviously be one" is answered
   before it is built.
3. **`the_generic_fold_matches_scalar_at_every_short_length_pair`** — a 15×15
   grid of `(adds.len(), subs.len())` against the scalar reference. Written for
   the rejected arm and kept, because it fills a real hole: every other kernel
   test in the module runs either ≤2 rows or 200+, so **nothing covered the
   short unequal lengths that are the threat channel's entire working range**
   (18.66% of threat folds are one row; 59.10% are four or fewer).

---

## 7. GATES

`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` — **exit 0, "All
checks passed"**. (`--test-threads=2` because the D49 trainer was live and the
debug test binary is ~2.4 GB at the default 4.)

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| feature-matrix `cargo check` | clean |
| `cargo test` (debug: unit + fast perft tier) | pass |
| release perft suite (`--ignored perft_full_suite`) | pass |
| `cargo test --features sota,arch-d48x` (the sota tests no other gate runs) | pass, including the new 15×15 grid test |
| `bench 16 1 6 default depth` | **587683** — the pin, bit-for-bit |
| wl50 eval hash | `eb981b119bd66d6eade8e314aa32d254e7fc4821c29e33ac20b5ccc4c84f4f37` — the pin |
| d13 node total | **1985398** — the pin |
| node identity, study frame | **5,001,624** on every binary in every panel |

**And one gate this study added for itself:** the landing tree was rebuilt as
`bin34/land` and md5'd against the binary the panel actually measured.

```
19798abf79732d92c9fbcb2c90af699f  bin34/nocold   <- what §0.2 measured
19798abf79732d92c9fbcb2c90af699f  bin34/land     <- what is being committed
```

Byte-identical, so the doc comments and the new `#[cfg(test)]` grid test that
went in alongside the attribute change provably did not perturb the codegen the
numbers came from. Worth doing on any arm whose write-up happens after the
measurement.

---

## 8. REPRODUCING

```bash
# the four attribute variants — patch, build, restore, in one step each
for L in base nocold noinl noboth; do bash tools/perf37/build37.sh "$L"; done
cp bin34/base bin34/null
md5sum bin34/base bin34/noinl          # <- the §2.1 result, no panel needed
bash tools/perf34/ident34.sh base nocold noinl noboth

# the static census that shows `cold` changed only the caller's block order
bash tools/perf37/symcount37.sh bin34/base   i8_deltas_into_generic
bash tools/perf37/symcount37.sh bin34/nocold i8_deltas_into_generic

# the core panel (§0.2)
P37_PANEL=docs/research/perf-artifacts-37/panel-core.txt \
  bash tools/perf37/panel37.sh 12 base null nocold noboth thp
python3 tools/perf37/med37.py docs/research/perf-artifacts-37/panel-core.txt base

# the TLB panel (§4.4)
P37_PANEL=docs/research/perf-artifacts-37/panel-tlb.txt \
P37_EV=ls_l1_d_tlb_miss.all_l2_miss:u,ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit:u,ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit:u,instructions:u \
  bash tools/perf37/panel37.sh 8 base null thp

# the huge-page arm, if a future box has free order-9 pages
git apply docs/research/perf-artifacts-37/thp-arm.patch
bash tools/perf37/vmflags37.sh ./bin34/thp
```

Raw panels: `panel-core.txt`, `panel-paired.txt`, `panel-tlb.txt` — every row,
in order.

---

## 9. WHAT I AM LEAST SURE ABOUT

In order.

1. **That `nocold`'s −0.744 branches/node is worth landing on its own.** It is
   exact, it is free, and it removes a hint that is factually false about 12.38%
   of engine cycles — but it is 0.12% of branches, and this study cannot say
   what that is in cycles because §1.3 disqualifies the only counter that could.
   The mechanism arithmetic (0.87 branches per threat fold) is tight enough that
   I believe the *count*; what I cannot price is what one predicted-not-taken
   branch per fold costs on Zen 4, and I have not pretended to.
2. **That my `#[cold]` mechanism is complete.** I predicted an `optsize` effect
   (LLVM reports a `cold` function as cold to `ProfileSummaryInfo`, and this
   kernel's tile walk is documented as depending on the unroller's choice) and
   the disassembly says it did not happen — same address, same 3616 bytes, same
   histogram. Block placement in the caller is what I can *see*; whether there
   is a second effect I did not look for, I do not know.
3. **§4.4's 2.25 page walks per node**, which retires my own hypothesis. It is
   an 8-round median on a counter whose null moved +2.303% over 22.6 points, so
   the *level* is solid (it is a level, not a delta) but I would not defend the
   third digit. The discrete part — `tlb_reload_2m_l2_hit` reading a flat
   0.0000 on every label — is the part I would defend, and it is what actually
   decides the arm.
4. **Whether the paired arm would still lose at a different `REGS`/`TILES`.**
   §3.1's explanation is that LLVM's cloning outweighs the control, which is a
   statement about *this* tile shape. At `TILES = 3` (ACC 1536) the control cost
   rises 50% and the clone count rises with it; I did not test another width and
   the banner does not claim I did.
5. **The `thr_copy` price in §5** (~26 i/n, 0.45%). It is derived from the
   foldcount rate (0.330/node, study 36's census) times a static count of 40
   zmm loads + 40 stores, not measured by removing it. The count is solid; that
   the removal would *land* at that size is not — §3 is a fresh reminder that
   this codebase's kernels do not always pay what their instruction counts say.
6. **That the box was quiet enough.** Load average 4.3–4.8 with the D49 trainer
   live is the best window this study run has had, and the exact counters do not
   care regardless — but the cycles column in every panel here is still
   unusable, and I want to be clear that "quiet" did not fix it.
