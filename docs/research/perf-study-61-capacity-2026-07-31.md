# Perf study 61 — the weight rows are a CAPACITY problem, and the cache is already at the bound

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`5fdee111`**
(post-study-60) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`,
ACC 1280 · **Instruments:** `tools/perf61/` · **Artifacts:**
`docs/research/perf-artifacts-61/` · **Branch:**
`worktree-agent-a122e2d810839db7a`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, non-PGO, pinned to physical core 2. **A GPU training run
shared the box for the whole study**, so `cycles:u` is quoted for its sign only
and every verdict rests on `instructions:u` (null 0.000, load-robust) or on a
counter with its own null in the same rotation — study 60 §6's rule, applied.

**Rebase check (study 60's `bcmisc55` rule).** The study opened on `5fdee111`;
`main` moved to **`2da9c5d8`** and then **`0f319802`** during the write-up.
`git diff --stat 5fdee111 0f319802` is **three docs files and no `src/`**, so by the rule that a number
expires when its *substrate* moves, nothing here expired. It was re-verified
anyway: rebuilt from source on the rebased tree, the arm-off and arm-on binaries
reproduce their pre-rebase `.text` md5s exactly (`7403cc4a` / `242abfdd`) and
the whole-frame contrast is **28,256,163,127 → 27,911,703,999 = −344,459,128
instructions = −68.87/node = −1.219 %**, identical to the panel.

---

## ★ 0. WHAT LANDED

**A speed change, landed unconditionally — the first since study 59, and the
second-largest single instruction win of the 52–61 chain** (study 59's −176.8
i/n is larger; studies 56 at −65.8 and 55 at −83.8 are the nearest neighbours).

The fold's shape dispatch, extended from the four pairs study 25 chose for a
HalfKA-only kernel to the **one-sided** and **three-row** families. Straight-line
coverage of the threat channel **27.27 % → 68.40 %**. Node-identical
(**5,001,562**), eval hash exact (**`2e1ede95…79c4`**), `.text` +8,048 B.

| counter | median | se | sign | p | its null |
|---|---:|---:|---:|---:|---|
| `instructions:u` | **−1.219 %** (−68.87/node) | **0.000** | **32/32** | **0.0000** | −0.000 %, se 0.000 |
| `branches:u` | **−2.274 %** (−12.92/node) | **0.000** | **32/32** | **0.0000** | −0.000 %, se 0.000 |
| `branch-misses:u` | **−1.336 %** (−0.297/node) | 0.129 | **31/32** | **0.0000** | +0.043 %, 14/32 |

The mechanism arithmetic closes: **−115 instructions per converted one-sided
fold and −322 per converted three-row fold** — 8 tiles × two runtime loops with
an odd/even unroll remainder each, which is exactly what a straight-line
instantiation deletes, and it scales with the row count because the generic
kernel pays that control *per row per tile*.

**`cycles:u` was NOT resolvable** — a GPU training run shared the box and in
panel 1 the byte-identical null moved further than the arm (−0.497 % against
−0.135 %). The arms are landed on the exact counters and **their wall-clock
value is not established**; §9 item 3 says what to run on a quiet box.

**Also landed: five instruments, all new, none of which existed anywhere in the
series.**

1. `tools/perf61/fills61.sh` — the **fill-SOURCE census** (`ls_any_fills_from_sys`
   split L2 / L3 / DRAM). Nobody had ever asked which *level* served this
   engine's misses.
2. `foldcount::note_row_reuse` — the **reuse-distance census** of the fold's
   weight rows, per channel.
3. `probe-wpin-hka` / `probe-wpin-thr` — study 51's ceiling probe **split by
   channel**.
4. `foldcount::THR_PAIR` / `NNZ` — the threat fold's `(adds, subs)` pair
   histogram (which is what found the landed arm) and the sparse L1 matmul's
   group density.
5. `tools/perf61/ibstop61.py` — the engine-wide IBS census ranked by
   **tag-to-retire latency** beside op count, which is study 29 §8's rule
   applied to cycles rather than to misses (§8).

**And five measured negatives, each with a number** — §3 (the capacity model
closes the tenth route), §4 (`shape61x`), §5 (REGS vs L2), §6 (dense L1 at
31.3 % group density), §7 (the channel split).

---

## ★ 1. THE FILL-SOURCE CENSUS — 85 % of this engine's L2 misses are weight rows

Every prior study in this line measured *L1* misses. `L1-dcache-load-misses`
counts a line that was not in L1; it does not say whether that line came from
L2 at ~14 cycles or from DRAM at ~300. Those two have completely different
levers, and the whole closed-route catalogue was built without the distinction.

`ls_any_fills_from_sys` on Zen 4 splits exactly that way. Medians per node,
`base` n = 6, probes n = 3, all on core 2:

| | `base` | `wpin` (all rows pinned) | **weight rows** | share of the engine |
|---|---:|---:|---:|---:|
| L1 fills, total | 565.53 | 449.48 | **116.05** | 20.5 % |
| …served by **L2** | 522.24 | 442.19 | **80.05** | 15.3 % |
| …served by **L3** | 38.58 | 5.669 | **32.91** | **85.3 %** |
| …served by **DRAM** | 5.154 | 1.566 | **3.588** | **69.6 %** |
| `instructions:u` | 5650.76 | 5611.18 | | |
| `cycles:u` (sign only) | 3529.3 | 3032.4 | **−14.08 %** | |

Two facts nobody had:

* **The weight rows are 20.5 % of L1 fills but 85.3 % of L3 fills and 69.6 % of
  DRAM fills.** Everything else in this engine — the accumulator stacks, the
  head's tables, the TT, the history planes, movegen — is essentially
  **L2-resident**: 449.5 fills/node of non-weight traffic produce only 5.7 L3
  fills and 1.6 DRAM fills.
* **Latency-weighted, the expensive part of the fold's memory cost is the L2
  misses, not the L1 misses.** At Zen 4's ~14 / ~50 / ~300 cycle service times
  the 116.05 fills decompose as 1121 / 1646 / 1076 unoverlapped cycles: **the
  29 % of fills that miss L2 carry 71 % of the cost.**

> **This retires the reading that the fold's problem is "arrival time in L1".**
> Nine dead arms attacked L1 arrival. The money is one level down.

The `cycles:u` column reproduces study 51's `probe-wpin` result (−13.19 %) at
−14.08 % on a different base with a different instruction count, which is the
cross-check that this frame is the same experiment.

---

## ★ 2. THE REUSE-DISTANCE CENSUS — the instrument the catalogue never had

`foldcount::note_row_reuse` records, for every folded weight row, **the number
of row-touches in that channel since the same row was last folded**, bucketed by
`floor(log2 d)`. One `Vec<u32>` load/compare/store per row; it never touches the
weight bytes, so it cannot perturb the locality it measures.

Reading it: a fully-associative LRU cache holding the last `W` DISTINCT rows
gets **at least** `CDF(W)` hits, because the number of distinct rows in a window
of `W` touches is at most `W`. So `CDF` evaluated at `cache bytes ÷ 1280` is a
(slightly optimistic, because it assumes the cache holds nothing else) bound on
the achievable hit rate.

```
FOLDROWREUSE hka touches=15,863,960  distinct=20,022 of 22,528  first-touch 0.129 %
FOLDROWREUSE thr touches=19,616,896  distinct=44,277 of 60,720  first-touch 0.226 %
```

The two channels' distributions are the same shape, and the blend uses the
measured 0.4471 / 0.5529 touch split:

| cache capacity | rows | hka | thr | **blended CDF** |
|---|---:|---:|---:|---:|
| **32 KB (L1d)** | 25 | 20.3 % | 17.1 % | **18.5 %** |
| 256 KB | 205 | 52.8 % | 60.8 % | 57.3 % |
| 512 KB | 410 | 63.0 % | 71.1 % | 67.5 % |
| **1 MB (L2)** | 819 | 71.7 % | 78.8 % | **75.6 %** |
| 2 MB | 1638 | 78.9 % | 84.3 % | 81.9 % |
| **32 MB (L3)** | 26214 | 95.2 % | 95.8 % | **95.6 %** |

---

## ★★ 3. THE MODEL IS VALIDATED AGAINST THE HARDWARE, AND THAT CLOSES THE ROUTE

The census and the fill-source census are independent instruments. Put them
together and the model **predicts the measured hit rates at both levels**:

| level | model | measured | derivation of the measurement |
|---|---:|---:|---|
| **L1 hit rate on weight-row lines** | **18.5 %** | **19.1 %** | 7.1754 rows × 20 lines = 143.51 lines/node demanded; 116.05 fills ⇒ 1 − 116.05/143.51 |
| **P(L2 hit \| L1 miss)** | **70.1 %** | **69.0 %** | 80.05 / 116.05 |
| P(L3 hit \| L2 miss) | 81.8 % | 90.2 % | 32.91 / 36.50 |

`(0.756 − 0.185)/(1 − 0.185) = 70.1 %` is the conditional the L2 actually faces.
**The prediction lands within 0.6 points at L1 and 1.1 points at L2**, from a
model with no fitted parameter — the only inputs are the reuse histogram and the
cache sizes.

> ### The tenth route, stated as a result rather than as a failure
>
> **The fold's weight-row misses are pure CAPACITY, and the cache hierarchy is
> already extracting essentially all of the locality this access stream
> contains.** There is no ordering headroom (an ideal, exclusive-use, fully-associative LRU
> would do 1.1 points better at L2 than the real cache does), no conflict headroom, and no
> arrival headroom (which is why nine prefetch arms died). The reuse distance of
> a weight row is a property of *chess* — of how often a search re-touches the
> same (king bucket, piece, square) or the same threat relation — and the only
> free variable left in `bytes ÷ 1280` is the 1280.

**This is a stronger closure than "we tried and failed", and it is the one thing
that could have been established at any point in the last ten studies and was
not.** It also explains the whole catalogue at once: prefetching cannot help a
capacity miss, packing 0.03 % of bytes cannot help a capacity miss, huge pages
and 4 KB congruence are not capacity, and pollution control is not capacity
either (study 51 measured that at 10.4 % and the hardware does not offer it).

### What it hands to the width decision, quantified

Study 51 §6 concluded that "what is left is the net architecture" and priced a
1280 → 1024 step at ~2.6 % of frame *on the byte term alone*. The capacity model
says the byte term is not the whole story, because **narrowing the row also
raises the hit rate** — capacity in rows is `1 MB ÷ width`, and the CDF is
steep there. L2-missing weight-row bytes per node, model-extrapolated
(the log2 bucketing makes the hit-rate column coarse — it steps rather than
glides — so read the `rel` column as a bracket, not a decimal):

| ACC | L2 capacity (rows) | model L1 hit | model L2\|L1miss | L2-miss B/node | rel |
|---:|---:|---:|---:|---:|---:|
| 1536 | 682 | 18.5 % | 70.1 % | 2686 | 1.200 |
| **1280 (shipped)** | 819 | 18.5 % | 70.1 % | **2239** | **1.000** |
| 1024 | 1024 | 27.5 % | 70.6 % | 1567 | 0.700 |
| 896 | 1170 | 32.1 % | 73.3 % | 1165 | 0.521 |
| 768 | 1365 | 32.1 % | 73.3 % | 999 | 0.446 |
| 640 | 1638 | 32.1 % | 73.3 % | 832 | 0.372 |

**1280 → 1024 is a 20 % byte cut but a 30 % cut in L2-missing bytes.** The speed
side of the standing "width is ~Elo-neutral across 768–1536" finding is
therefore *convex in width*, not linear, and 1536 is worse than its byte count
says (1.200×, not 1.20× of a smaller term). This is a number the width decision
did not have.

---

## ★ 4. THE LANDED ARM — a dispatch table written for one channel, inherited by another

### 4.1 The gap, counted for the first time

`i8_deltas_into_inner` dispatches on `(adds.len(), subs.len())` and has exactly
four arms: `(1,1) (1,2) (2,1) (2,2)`. Those were chosen in **study 25, when this
kernel was HalfKA-only** — `apply_dirty_directed` builds both lists in
`[usize; 2]` and every real move emits at least one add and one sub, so the four
arms are a *complete cover* of HalfKA and the fallthrough was genuinely dead.
**Study 32 quantised HalfKA to i8, deleted the i16 family, and routed the entire
FullThreats channel through the same dispatch** — without revisiting the arm
list, and without anyone counting what the new traffic's shapes are.

`foldcount::THR_PAIR`, 4,270,334 threat folds:

| `(adds, subs)` | folds | share | path |
|---|---:|---:|---|
| the four study-25 arms | 1,164,466 | **27.27 %** | shaped |
| **`(0,1)`** | 444,368 | **10.41 %** | generic |
| **`(1,0)`** | 371,467 | **8.70 %** | generic |
| `(0,2)` | 168,840 | 3.95 % | generic |
| `(2,0)` | 84,674 | 1.98 % | generic |
| the 3-row family (`(3,0) (0,3) (1,3) (3,1) (2,3) (3,2) (3,3)`) | 686,947 | 16.09 % | generic |
| everything else | 1,349,572 | 31.60 % | generic |

**19.11 % of every threat fold is a SINGLE weight row, and every one of them
runs the runtime-length kernel** — two loops whose trip counts LLVM cannot see,
an ×2 unroll with an odd/even remainder branch, and an empty-list test, eight
times over (once per tile), to apply one row. A one-sided list is precisely the
case the generic form was never for.

`shape61` adds `(1,0) (0,1) (2,0) (0,2)`: straight-line coverage 27.27 % →
**52.31 %**. `i8_into_shaped` already handles `NS = 0` — `srow` is `[usize; 0]`,
the `zip` fills nothing, the loop compiles away — so this is a dispatch change,
not a kernel change, and it is bit-identical by construction (same tiles, same
order, same rows, same order, same per-lane wrapping i16 arithmetic).

### 4.2 The panel

`docs/research/perf-artifacts-61/panel-shape.txt` — 40 rounds (39
adjudicated), blocked Latin square, `null` a byte-identical copy of `base`
riding in the rotation, realised mean position 1.00 for all three labels, node
total 5,001,562 on every run.

```
REFERENCE per node: instructions 5649.51  branches 568.14
                    branch-misses 21.54  L1-dcache-load-misses 537.37  cycles 3537.04

label   event                       median %     se    neg/n      p
null    instructions:u                0.000   0.000    17/39   0.5224
null    branches:u                    0.000   0.000    15/39   0.1996
null    branch-misses:u              -0.022   0.222    20/39   1.0000
null    L1-dcache-load-misses:u      -0.083   0.065    26/39   0.0533   ← borderline
null    cycles:u                     -0.497   0.251    24/39   0.1996

shape   instructions:u               -0.438   0.000    39/39   0.0000
shape   branches:u                   -0.509   0.000    39/39   0.0000
shape   branch-misses:u              -0.296   0.126    25/39   0.1081
shape   L1-dcache-load-misses:u      -0.048   0.063    23/39   0.3368
shape   cycles:u                     -0.135   0.270    21/39   0.7493
```

**The two exact counters are unanimous with zero spread and their null is
exactly 0.000 %: −24.74 instructions and −2.891 branches per node.**

The other three are reported and none of them carries the verdict:

* `branch-misses` −0.296 % against a null of −0.022 % is **−0.059/node net** at
  25/39 (p = 0.108) — the right sign for deleting a data-dependent trip count,
  but not resolvable, and far too small to matter either way.
* `L1-dcache-load-misses` moves +0.035 % net and **its own null fails at
  p = 0.0533**. Correct: the arm changes code shape, not which bytes are read.
* **`cycles:u` is not resolvable and the NULL MOVES MORE THAN THE ARM**
  (−0.497 % against −0.135 %). That is study 60 §6's defect reproduced exactly
  on this box, under this study's GPU-training-run contention, and it is why §9
  item 3 says the arm's wall-clock value is not established.

### 4.3 The three-row family — **it more than doubles the win, and it also lands**

The natural worry about extending a dispatch table is `.text`: seven more
instantiations of a body up to six rows wide. **Measured rather than estimated:
`.text` goes 714,725 → 719,685 (one-sided) → 722,773 (three-row), so the second
step costs only +3,088 B** because LLVM shares the tile prologue and epilogue
across instantiations. Coverage 52.31 % → **68.40 %**.

`docs/research/perf-artifacts-61/panel-shapex.txt` — 32 rounds, four labels in
one blocked rotation, realised mean position 1.50 for all four, node total
5,001,562 on every run:

```
REFERENCE per node: instructions 5649.51  branches 568.14
                    branch-misses 21.50  L1-dcache-load-misses 536.28  cycles 3540.50

label    event                       median %     se    neg/n      p
null     instructions:u               -0.000   0.000    17/32   0.8601
null     branches:u                   -0.000   0.000    17/32   0.8601
null     branch-misses:u              +0.043   0.140    14/32   0.5966
null     L1-dcache-load-misses:u      +0.228   0.063    10/32   0.0501   ← borderline
null     cycles:u                     +0.151   0.363    15/32   0.8601

shape    instructions:u               -0.438   0.000    32/32   0.0000
shape    branches:u                   -0.509   0.000    32/32   0.0000
shape    branch-misses:u              -0.094   0.147    19/32   0.3771
shape    L1-dcache-load-misses:u      +0.240   0.058     6/32   0.0005
shape    cycles:u                     -0.154   0.261    17/32   0.8601

shapex   instructions:u               -1.219   0.000    32/32   0.0000
shapex   branches:u                   -2.274   0.000    32/32   0.0000
shapex   branch-misses:u              -1.336   0.129    31/32   0.0000
shapex   L1-dcache-load-misses:u      +0.344   0.080     4/32   0.0000
shapex   cycles:u                     +0.222   0.262    14/32   0.5966
```

**`shapex` is −1.219 % instructions and −2.274 % branches, 32/32, zero spread —
−68.87/node and −12.92/node.** It also converts branch misses: **−1.336 %
against a null of +0.043 %, i.e. −1.379 % net = −0.297/node, 31/32,
p < 0.0001**, which is the data-dependent trip counts disappearing.

**The generalisation this arm bought, and it is the useful part:** the win per
converted fold **grows with the shape** — **−115 instructions per one-sided
fold against −322 per three-row fold** — because the generic kernel pays its
loop control *per row per tile*. So the table does not saturate where I expected
it to, and **31.60 % of threat folds are still generic**.

`L1-dcache-load-misses` is quoted and not used: **its own null fails at
p = 0.0501 in this panel**, the net moves are +0.012 % and +0.116 %, and panel
1 (n = 39) gave the same `shape` contrast as +0.035 % net. Two panels of the
same binaries disagreeing on this counter by a factor of three is the counter's
known behaviour (study 60 §6), not a property of the arm — which changes code
shape and not which bytes are read.

**`cycles:u` is not resolvable in either panel and in panel 1 the NULL MOVED
FURTHER THAN THE ARM** (−0.497 % against −0.135 %). A GPU training run shared
this box throughout. **Both arms are landed on the exact counters; their wall
clock value is not established** (§9 item 3).

### 4.4 What landed, and how to un-land it

Both families are **unconditional** in the shipped build. `--features
noshape61` restores the study-25 dispatch, so the contrast stays measurable on
the shipped tree — the same falsifiability `null34` and `nogc54` keep for their
studies.

| binary | nodes | eval hash | `.text` | `instructions:u` (whole frame) |
|---|---:|---|---:|---:|
| `base` (`5fdee111`) | 5,001,562 | `2e1ede95…79c4` | 714,725 | 28,256,162,803 |
| `revert` (`noshape61`) | 5,001,562 | `2e1ede95…79c4` | 714,725 | 28,256,163,755 |
| `shape` | 5,001,562 | `2e1ede95…79c4` | 719,685 | 28,132,465,167 |
| `shapex` | 5,001,562 | `2e1ede95…79c4` | 722,773 | 27,911,704,117 |
| **`landed` (default)** | **5,001,562** | **`2e1ede95…79c4`** | **722,773** | **27,911,704,350** |

`revert` reproduces `base` to **3.4 parts per 10⁸** and `landed` reproduces
`shapex` to **8 parts per 10⁹**, with identical `.text` sizes — the promotion
from `#[cfg(feature = …)]` to unconditional changed nothing.

> **A trap worth recording: `.text` md5 is NOT a valid identity check across a
> Cargo feature-set change.** Study 55's md5 strips objdump's filename banner,
> but the disassembly listing also carries symbol names in its branch-target
> annotations, and a Rust symbol name embeds the crate **metadata hash** — which
> moves when the feature set moves, for byte-identical code. `revert` and `base`
> have different md5s (`7403cc4a` vs `440f2232`) and identical `.text` sizes and
> instruction counts. `tools/perf61/ident61.sh` is the check that survives it.
> The md5 remains exactly right for the case study 55 and study 60 used it for —
> the same feature set, an edit inside a `#[cfg]` — and it was used that way
> here (`base3` = `base` = `440f2232` before the promotion).

---

## ★ 5. REGS SWEPT ON A COUNTER NOBODY LOOKED AT — and my own hypothesis dies

Before building anything I ran the null-first mechanism test the brief asks for.
**Hypothesis:** the fold is tile-major, so a weight row's 1280 contiguous bytes
are read as 8 chunks of 160 B separated by other rows' chunks; a *wider* tile
(REGS = 20 ⇒ 640 B chunks) restores a longer sequential run, which the Zen 4 L2
stream prefetcher can train on. Prediction: **fewer L3 fills at higher REGS.**

Measured, 2 runs each, medians per node:

| REGS | row chunk per tile | **L3 fills/node** | vs base | DRAM fills/node | `instructions:u` |
|---:|---:|---:|---:|---:|---:|
| 4 | 128 B | 38.01 | −1.0 % | 5.283 | 5736.47 |
| **5 (shipped)** | 160 B | **38.40** | — | 5.154 | 5647.29 |
| 10 | 320 B | 40.40 | **+5.2 %** | 5.346 | 5514.32 |
| 20 | 640 B | **47.72** | **+24.3 %** | 5.918 | 5458.17 |

**The sign is the opposite of the hypothesis, and it is large and monotone.**
Wider tiles are strictly *worse* for L2 residency. `local_l2` falls (522.2 →
510.6) while `local_ccx` rises (38.4 → 47.7) at near-constant total fills, i.e.
the same lines are being fetched and more of them have been evicted from L2 —
consistent with study 44's mechanism (a REGS = 20 tile closes with a 40-store
burst) reaching one level further out than study 44 could see.

Two things follow. **(a) My prefetcher hypothesis is refuted before a line of
arm code was written** — which is what null-first is for. **(b) `REGS_CAP = 5`
is now confirmed from a third independent direction** (study 44's store-queue
stalls, study 60's instruction count and dead miss statistic, and this L2
census), and study 44's cycles loss at REGS = 20 now has a *memory* mechanism
attached to it, not only a store-queue one. Do not re-open the sweep.

---

## ★ 6. THE DENSE L1 MATMUL — priced at 31.3 % group density and dead

`l1_dot`'s inner block is **6.9 % of the engine's IBS tag-to-retire latency** at
41–48 cycles/op — the single most concentrated stall site outside the fold:

```
movzwl (%r12),%r11d                 ; g = nnz[i]
vpbroadcastd (%rbx,%r11,4),%zmm4    ; the group's four FT bytes
shl    $0x6,%r11d
vpdpbusd (%rax,%r11,1),%zmm4,%zmm1  ; one 64-byte GATHERED line of l1_weights
```

One randomly-addressed 64-byte line per nonzero four-lane group, over a 20 KB
per-bucket table that study 51 measures at a **91.44 % L1 miss rate**. A group
is zero only if all FOUR of its clipped-ReLU lanes are zero, so the *group*
density is far above the element density — and if it were high enough, the
**dense** form would read the same 20 KB *sequentially* (a stream the hardware
prefetcher can follow) for a modest amount of extra arithmetic, and be
bit-exact, because a zero group contributes nothing. That is squarely the
brief's "change what is fetched, not when", and it is not prefetching.

**Measured before building** (`foldcount::NNZ`): 216,200,386 nonzero groups over
2,156,392 evaluations = **100.26 of 320 groups, 31.3 % density**. The dense form
is **3.19× the `vpdpbusd` work and 3.19× the bytes** (20,480 B/eval against
6,417 B). Dead, decisively, at the cost of one counter.

Recorded because the idea is attractive from the disassembly and the 91.44 %
miss rate, and the number that kills it is not derivable from anything already
published: `l1_weights` traffic is **43.2 lines/node**, not the 320/eval the
table size suggests.

---

## ★ 7. THE CEILING, SPLIT BY CHANNEL — and one honest confound

Study 51's `probe-wpin` pins *every* fold's row index to 0 and prices the whole
weight-row stream at −13.19 % cycles. It is one number covering two tables of
very different shape (`hka_weights` 28.8 MB / 22,528 rows folded 2.279 per fold;
`thr_weights` 77.7 MB / 60,720 rows folded 4.594 per fold), and a prize that is
80/20 is a different engineering problem from one that is 50/50.

`probe-wpin-hka` and `probe-wpin-thr` are strict subsets of `probe-wpin`, so
`Δhka + Δthr ≈ Δboth` is a built-in consistency check. **The refactor is
byte-identical in both directions — `bin61/base3` reproduces `base`'s `.text`
md5 `440f2232` and `bin61/wpin2` reproduces `wpin`'s `df350961`** (verified by
building, not by mtime — study 60 §7 item 6).

Deltas from `base`, per node:

| | Δ L1 fills | Δ **L3 fills** | Δ **DRAM fills** | Δ `cycles:u` |
|---|---:|---:|---:|---:|
| `wpin-hka` | −10.74 | **−20.02** | −1.951 | −4.11 % |
| `wpin-thr` | −62.26 | **−16.16** | −1.859 | −6.90 % |
| **sum** | −73.00 | **−36.18** | **−3.810** | −11.01 % |
| `wpin` (both) | **−116.05** | **−32.91** | **−3.588** | **−14.08 %** |

**The L3 and DRAM columns are additive to within 10 % and 6 %.** Summing them
(L3 + DRAM = the L2 misses), the prize splits **54.9 % HalfKA / 45.1 %
threats** — against a ROW split of 44.2 % / 55.8 % (3.1718 and 3.9221 rows/node
under the respective probes). Per row that is **6.93 L2 misses per HalfKA row
against 4.59 per threat row, a 1.51× asymmetry**, despite `hka_weights` being
the smaller table by 2.7×. The reuse census (§2) says the two channels' distance
distributions are *nearly the same shape*, so the asymmetry is in how the two
sets interleave, and it is not explained here.

(`probe-wpin-thr` pins `apply_threat_deltas_into` but not
`refresh_threat_into_at`, which is a further 0.0815 rows/node — 2.0 % of the
channel — so the threat side is very slightly under-counted.)

**The confound, stated rather than buried:** each probe computes a wrong
evaluation and therefore searches a different tree. `wpin-hka` runs at
5857.80 i/n against `base`'s 5650.76 — **+3.7 %, a materially more expensive
tree** — which is why the L1-fill column is *not* additive (−73.00 against
−116.05) and why the `cycles` column under-adds. The L3/DRAM columns survive it
because weight rows are 85 % of that traffic, so the confound is proportionally
small there; the L1-fill and cycles columns do not, and are quoted for their
sign only.

**Retracted before it was published:** an earlier read of the IBS symbol census
attributed 34.2 % of engine latency to `i8_deltas_into` (2.23 rows/node) and
22.8 % to `i8_deltas_into_generic` (3.92 rows/node), i.e. "HalfKA rows cost
2.65× more latency per row". **That attribution is wrong.** `THR_PAIR` shows
27.27 % of *threat* folds also take the shaped path, so the two symbols are not
the two channels, and IBS tag-to-retire additionally over-charges a fully
unrolled straight-line block relative to a looped one. The 1.6× figure above is
from the pin probes, which do not have either problem.

---

## ★ 8. THE IBS LATENCY CENSUS — the ranking study 29 §8 asked for, engine-wide

`tools/perf61/ibstop61.py` over 725,233 sampled ops (3 runs). Ranking by
**tag-to-retire latency**, with the op-count share beside it, because the
*difference* is the signal:

| ops % | **lat %** | cy/op | symbol |
|---:|---:|---:|---|
| 6.01 | **34.23** | 27.9 | `avx512::i8_deltas_into` |
| 8.88 | **22.75** | 12.5 | `avx512::i8_deltas_into_generic` |
| 11.17 | **15.55** | 6.8 | `SotaHead::forward_into` |
| 4.03 | **9.64** | 11.7 | `HalfkaContext::finny_refresh_into` |
| **16.84** | 4.90 | **1.4** | `search::<NonPv>` |
| 1.70 | 2.30 | 6.6 | `qsearch::<NonPv>` |
| **7.01** | 1.57 | **1.1** | `ThreatContext::at_full` |
| **8.37** | 1.33 | **0.8** | `MovePicker::init_quiets` |

**The four accumulator symbols are 82.2 % of tag-to-retire latency on 30.1 % of
ops.** The search layer is the mirror image — 16.8 % of ops at 1.4 cy/op — which
is the quantitative form of "instruction reductions have stopped converting":
the instructions that remain in the search are the cheap ones.

`finny_refresh_into` at 9.64 % is the highest-ranked symbol nobody is currently
working on. **It is not an opportunity**: study 12 §6 measured this engine at
**0.1611 refreshes and 0.9064 refresh rows per node against `eca43a97`'s 0.1673
and 0.9268** — Stockfish parity — so the refresh RATE is already at the
reference and the rows it folds are the same capacity-bound rows as §3.

---

## ★ 8.5 ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | A weight row is exactly **20 aligned cache lines** | 1280 = 20 × 64, and both tables are 64-byte aligned in the runtime region dump (`hka_weights lo=…040`, `thr_weights lo=…040`), so no row straddles a 21st line | §3's "143.5 lines/node demanded", hence the L1 hit rate |
| A2 | `ls_any_fills_from_sys.<umask>` partitions L1 fills by source level | The umasks sum to `.all` in every run (565.97 vs 565.53 across medians of different runs); `near_cache` reads exactly 0 on a single-CCX part, which is the correct answer and a free sanity check | §1 and §7 entirely |
| A3 | `probe-wpin` changes only weight-row locality | Study 51 A2, re-verified: `instructions:u` moves −0.68 % (5649.5 → 5611.0/node) and the kernels are untouched | §1's attribution |
| A4 | Zen 4 service times ~14 / ~50 / ~300 cycles for L2 / L3 / DRAM | Vendor-published ranges, not measured here | Only the "71 % of the cost" split in §1 — the FILL COUNTS do not depend on it |
| A5 | `CDF(W)` bounds a `W`-row cache's hit rate | Distinct rows in a window of `W` touches ≤ `W` | §3's model would be neither a bound nor a prediction |
| A6 | The shape dispatch is bit-identical | Same tiles, same order, same rows, same order, same wrapping i16 arithmetic; and the eval hash + node total are exact on every binary | The landed arm |

---

## ★ 8.6 THE COMMISSION'S CLAIMS, CHECKED

The brief asked to be told plainly where it fails a check. It mostly does not.

| commission claim | check |
|---|---|
| `probe-wpin` = −13.19 % cycles | **Reproduced** at −14.08 % on this base (§1) |
| IPC 1.595 | **1.613** measured here (28.256 G instructions / 17.52 G cycles, `base`) |
| 543 L1d misses/node, 18.33 % miss rate | **537.4/node** on the non-PGO build (the brief's figure is PGO) |
| branch mispredicts ~9.4 % of cycles | **21.52/node against 3529 cycles/node**, i.e. ~11 % at an 18-cycle penalty. Slightly higher than the brief, and **study 27 already attacked this block and got a NULL on branch misses** while landing 4 instruction arms — which is why this study did not go there |
| `cycles:u` is not load-robust | **Confirmed hard.** The byte-identical null read se 0.251–0.363 and, in panel 1, moved **further than the arm** (−0.497 % vs −0.135 %) |
| `instructions:u` is a guard, and a memory arm may RAISE it and still win | **The licence was available and not needed.** The one candidate that would have used it — the dense L1 matmul, which trades 3.19× the arithmetic for a sequential stream — was killed by a density counter (§6) before it was built |
| "the fold's weight rows are 55.05 % of miss LATENCY" | Not contradicted, but **refined**: that figure is IBS *unoverlapped* latency, which study 51 §9 item 1 itself warns "is not cycles". §1 gives the same fact in a unit that is not an inference — 20.5 % of L1 fills, **85.3 % of L3 fills, 69.6 % of DRAM fills** |
| "do not re-propose prefetching" | **Honoured.** The one prefetch-adjacent hypothesis here is about the *hardware* L2 stream prefetcher and tile width; it was tested with features that already existed and **refuted before any arm was written** (§5) |
| `build_pgo.sh` and the dangling-net trap | Not exercised: this study is **entirely non-PGO**, the same frame studies 51 and 60 §3 used, so its numbers compare directly to theirs |

---

## ★ 9. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The capacity model's L3 rung does not fit.** It predicts 81.8 % for
   `P(L3 hit | L2 miss)` and the hardware does 90.2 %. The L1 and L2 rungs fit
   to 0.6 and 1.1 points, so I believe the model where it is checked; the L3 rung
   involves a 32 MB cache shared with a 16 MB TT and the whole 106.5 MB weight
   image, and `distinct ≤ window` is a loose bound at that scale. **Do not build
   on the L3 row.**
2. **The reuse census counts touches, not distinct rows.** `CDF(W)` is a lower
   bound on a `W`-row cache's hit rate, so the model is *optimistic* by an
   unmeasured amount, and the width extrapolation in §3 inherits that. The exact
   statistic (Mattson stack distance) needs a Fenwick tree over 35.5 M
   timestamps ≈ 143 MB; it was not run because a GPU trainer owns this box's
   memory. The agreement at two levels is the evidence that the bound is tight
   here, not a proof that it is.
3. **No cycles verdict, and no wall-clock panel.** The box carried a GPU
   training run throughout. `cycles:u` is not load-robust (study 60 §6, this
   study's own null reads se 0.5–0.6 against a quiet-box 0.25–0.30) and
   `tools/wallclock_panel.py` would have been reading the same contaminated
   signal one level up. **`shape61` is landed on `instructions:u` and
   `branches:u`, both exact at 40/40 with zero spread, and its value in wall
   clock is NOT established.** Whoever next gets a quiet box should run
   `wallclock_panel.py --base bin61/base --arm bin61/shape` before quoting it as
   Elo.
4. **The channel-split probes carry a tree confound** (§7) and only their
   L3/DRAM columns are quoted as measurements.
5. **Nothing was arbitrated by games.** `shape61` is node-identical, so a
   fixed-node match is a dead heat by construction and only a real-clock match
   could speak — which needs the quiet box of item 3.

---

## ★ 10. WHAT THE NEXT STUDY SHOULD KNOW

1. **The fold's weight rows are closed, and now for a reason rather than for
   nine reasons.** §3 is a validated capacity model: reuse distance + cache size
   predicts the measured L1 and L2 hit rates to within 0.6 and 1.1 points. Any
   future proposal at this site must say which of *reuse distance* or *bytes per
   row* it changes. Nothing else is a lever.
2. **Rank by LATENCY, then by LEVEL.** Study 29 §8 said rank by miss latency,
   not miss count; §1 adds the next rung — of the latency, ask which cache
   served it. 85 % of this engine's L2 misses are weight rows and everything
   else is L2-resident.
3. **`REGS_CAP = 5` is confirmed a third time, and wider is worse for L2** (§5).
   Closed at 20/10/8/5/4 on three independent counters.
4. **The dense L1 matmul is dead at 31.3 % group density** (§6). Do not
   re-derive it from the 91.44 % miss rate.
5. **A dispatch table written for one channel and inherited by another is worth
   re-counting.** §4 is a −24.7 i/n arm that existed because study 32 changed
   who calls `i8_deltas_into_inner` and nobody re-read its `match`. The same
   question is open elsewhere: `i8_deltas_inner` and `i8_deltas_dual_inner` have
   **no shape dispatch at all**, and `HKA_DUAL_SHAPE` says the warm Finny
   refresh averages 5.72 rows over a wide distribution.
6. **Still open, inherited:** study 43 §9 item 2's subtractive Finny-line arm;
   study 57 §6's FNV-1a-64 payload checksum (~45 % of all startup, needs a
   format ruling); study 48 §9 item 3's AVX2 `activate` hoist; and study 51
   §3's "what evicts the head's tables" — for which §1 now supplies half an
   answer, since the head's tables are among the 449.5 fills/node that are
   L2-resident and therefore cheap.

---

## ★ 11. GATES

| gate | result |
|---|---|
| node identity, study frame, every binary | **5,001,562** — exact |
| wl50 eval hash, `base` / `shape` / `shapex` | **`2e1ede95…79c4`** — exact |
| control `.text` md5 after every source edit | `base3` = `base` = **`440f2232`**; `wpin2` = `wpin` = **`df350961`** |
| `instructions:u`/node vs study 60 | 5649.51 vs 5649.51 — exact |
| `foldcount` rows/node vs studies 47/48/50/51 | **7.17540** — exact |
| byte-identical null through this harness | `instructions:u` **−0.000 %**, `branches:u` **−0.000 %**, both se 0.000 |
| realised mean panel position | **1.00** for every label |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | **exit 0 — `All checks passed.`, `Bench: 587683`, eval hash `2e1ede95…79c4`, d13 nodes `1739588`; all three pins unmoved. Run TWICE: on `5fdee111` and again after the rebase** |
| `cargo test --release -- --ignored perft_full_suite` | **ok** (1 passed), both before and after the rebase |
| `cargo clippy --all-targets` | **clean** (run by `check.sh`) |

D24 perft ritual not strictly required (nothing under `movegen`, `board`
mutation, `zobrist` or `attacks` is touched) — run anyway.

---

## ★ 12. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net

bash tools/perf61/build61.sh base             # the control
cp bin61/base bin61/null                      # the byte-identical null
bash tools/perf61/build61.sh shape  shape61
bash tools/perf61/build61.sh shapex shape61x
bash tools/perf61/build61.sh wpin    probe-wpin
bash tools/perf61/build61.sh wpinhka probe-wpin-hka
bash tools/perf61/build61.sh wpinthr probe-wpin-thr
bash tools/perf61/build61.sh regs20  regs20_44
bash tools/perf61/build61.sh regs10  regs10_44
bash tools/perf61/build61.sh regs4   regs4_44
bash tools/perf61/build61.sh rowreuse rowprobe    # §2's census
bash tools/perf61/build61.sh fcount   foldcount   # §4.1's THR_PAIR, §6's NNZ

# §1 and §7 — the fill-SOURCE census
bash tools/perf61/fills61.sh base wpin wpinhka wpinthr
# §5 — REGS on the L2 counter
P61_REPS=2 bash tools/perf61/fills61.sh regs4 base regs10 regs20

# §2 — the reuse distance
ARCHON_FOLDCOUNT=/tmp/reuse.txt taskset -c 2 ./bin61/rowreuse \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep FOLDROWREUSE /tmp/reuse.txt

# §4.1 and §6 — the shape pair histogram and the nnz density
ARCHON_FOLDCOUNT=/tmp/fc.txt taskset -c 2 ./bin61/fcount \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep -E 'thr_pair|nnz\.' /tmp/fc.txt

# §8 — the IBS latency census
P59_CPU=2 bash tools/perf59/ibs59.sh base 3 /tmp/ibs.txt   # needs bin59/base
python3 tools/perf61/ibstop61.py /tmp/ibs.txt 24

# §4.2 / §4.3 — the panels
P61_CPU=2 P61_EV='instructions:u,branches:u,branch-misses:u,L1-dcache-load-misses:u,cycles:u' \
  P61_PANEL=docs/research/perf-artifacts-61/panel-shape.txt \
  bash tools/perf61/panel61.sh 40 base null shape
python3 tools/perf61/med61.py docs/research/perf-artifacts-61/panel-shape.txt base null shape

CHECK_NNUE_GATE=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

**Do not run a panel while anything else builds** (study 60 §7 rule 5), and do
not park perf text on `/tmp` — it is tmpfs.
