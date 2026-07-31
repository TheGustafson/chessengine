# Perf study 51 — the fold is NOT what evicts the head, and Zen 4 has no pollution control

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `45ca8862` (study 50
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf51/` · **Artifacts:**
`docs/research/perf-artifacts-51/` · **Branch:** `perf-study-51-pollution`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
physical core 2 (siblings 2,10 — measured quietest at 7.6 %, same pair study 50
chose).

---

## ★ 0. WHAT LANDED, STATED HONESTLY

**No speedup landed. Four things did:**

1. **A hardware fact that closes two of the commission's four mechanisms**, with
   a positive control — `tools/perf51/ntprobe51.c`.
2. **The ceiling on this whole line of work: ≥ 13.19 % of cycles**, measured,
   nobody had it.
3. **The refutation of the commission's premise**, from the repo's first
   *engine-wide* per-region IBS census.
4. **A study-50 instrument bug fixed** (`ft_scale`'s region was registered at
   twice its real width, so its published miss rate is wrong).

Plus five nulls, three of which cost a build each and one of which would have
been a multi-day rewrite aimed at 0.03 % of the bytes.

**The commissioned question — "how do you stop single-use streaming data from
evicting a small hot working set?" — turns out to be the wrong question at this
site, and §3 is the measurement that says so.** Pollution is 10.4 % of the
prize. The other 89.6 % is the fold fetching its own bytes, and those bytes are
measured irreducible.

### The before/after the commission asked for, on the head's five weight tables

The commission asked me to lead with this table. It leads with it because it is
the study's central result, and it is a **null**:

| region | base miss rate | with weight-row streaming made **FREE** | Δ |
|---|---:|---:|---:|
| `l1_weights` | 91.44 % | **91.88 %** | **+0.44 pp** (worse) |
| `l2_weights` | 98.84 % | **98.38 %** | −0.46 pp |
| `skip_q` | 96.27 % | **95.40 %** | −0.87 pp |
| `l3_weights` | 100.00 % | **100.00 %** | 0.00 pp |
| `l1_biases` | 78.26 % | 90.91 % | +12.65 pp (n = 22) |

**Deleting 100 % of the fold's cold weight-row streaming leaves the head's five
tables missing at 91–100 %.** The fold is not what evicts them.

---

## ★ 1. MECHANISMS 1 AND 2 ARE DEAD ON THIS HARDWARE — measured first, before any arm

The commission required the hardware be verified before anything was built.
`tools/perf51/ntprobe51.c`: a 24 KB HOT array re-read every iteration, a 24 KB
COLD chunk streamed past it from a 64 MB array, 20 000 iterations. L1d is 32 KB,
so the pair deliberately overflows it. The counter is
`ls_any_fills_from_sys.all` — lines **entering** L1.

### Test A — does the loader put the line in L1 at all? (cold stream only)

Expected demand lines = 384 × 20 000 = **7,680,000**.

| loader | `ls_any_fills` | `ls_dmnd_fills` | `ls_hw_pf_dc_fills` | cycles |
|---|---:|---:|---:|---:|
| normal 512-bit load | 8,043,953 | 7,684,296 | 425,411 | 97,860,518 |
| **`VMOVNTDQA`** | **7,977,888** | **7,810,343** | 343,451 | 83,356,366 |
| `PREFETCHNTA` + load | 8,043,451 | 5,533,725 | 196,753 | 96,358,162 |
| `PREFETCHT0` + load | 7,780,893 | 5,567,468 | 272,663 | 117,112,744 |

**`VMOVNTDQA` generates 7.81 M DEMAND L1 fills for 7.68 M lines streamed.** It
allocates in L1 exactly like an ordinary load. The hardware prefetcher
contributed only 343 k, so this is the NT loads themselves and not a prefetcher
masking the effect — which was the obvious way this test could have lied.

### Test B — does either hint protect a hot working set? (hot + cold)

Floor if the cold loader does not pollute ≈ 7.70 M. Ceiling if it thrashes ≈ 15.36 M.

| loader for the cold chunk | `ls_any_fills` | verdict |
|---|---:|---|
| normal | 15,735,407 | hot array evicted |
| **`VMOVNTDQA`** | **15,795,484** | **hot array evicted** |
| **`PREFETCHNTA`** | **15,772,530** | **hot array evicted** |
| `PREFETCHT0` | 15,786,907 | hot array evicted |
| **CONTROL: hot only, no cold stream** | **23,954** | **fully resident** |

**The control is what makes this airtight.** The hot array costs 23,954 fills
when nothing streams past it — 0.003 fills per line, i.e. it lives in L1 across
all 20 000 iterations. Under every cold loader the total is 15.74–15.80 M, and
the excess over (cold-only + hot-only) is ≈ 7.7 M = **exactly the hot array's
7.68 M lines**. It misses on every single access, under all four.

> **Zen 4 offers no L1 cache-pollution control on WB memory.** `VMOVNTDQA` is
> WC-only, as the ISA says and as is now measured here rather than assumed in
> either direction; `PREFETCHNTA` fills L1 and does not de-prioritise the line,
> and it is additionally the most expensive of the four in cycles (166.8 M in
> test B). **Commission mechanisms 1 and 2: closed.**

This also retires study 49's open item, which listed `prefetchnta` on the fold's
weight rows as an untried mechanism. It is now tried, at the hardware level,
and it is inert. It is the **eighth** prefetch to die in this engine and the
first killed by a microbenchmark rather than a panel.

---

## ★ 2. THE CEILING: ≥ 13.19 % OF CYCLES, AND THE CONFOUND HELPS

`probe-wpin` (in the tree since study 29) sends every folded row index to row 0,
so the fold issues exactly the same 287 load ops per node against a permanently
L1-resident row. It isolates *the cost of the weight rows being COLD* — the
fold's own miss cost plus every bit of collateral damage it does.

15 rounds, blocked Latin square, byte-identical null riding in the rotation
(`tools/perf51/exact51.sh`, realised mean position 1.00 for all three labels):

| `wpin51` vs `base51` | median | trimmed | se | neg/n | p |
|---|---:|---:|---:|---:|---:|
| **`cycles:u`** | **−13.186 %** | −12.918 | 0.212 | **15/15** | **0.0001** |
| `l1-dcache-load-misses:u` | −20.456 % | −20.509 | 0.082 | 15/15 | 0.0001 |
| `ls_any_fills_from_sys.all:u` | −19.741 % | −19.742 | 0.085 | 15/15 | 0.0001 |
| `ls_dispatch.ld_dispatch:u` | −2.761 % | −2.752 | 0.030 | 15/15 | 0.0001 |
| `instructions:u` | −0.330 % | −0.330 | 0.000 | 15/15 | 0.0001 |

**The null, published as required:** `cycles:u` **+0.028 %** (7/15, p = 1.00),
`instructions:u` −0.000 %, `l1-dcache-load-misses` −0.068 % (10/15),
`ls_any_fills` −0.223 %, `ld_dispatch` +0.204 %. The arm is **470× the null's
magnitude on cycles and perfectly signed at 15/15**, which is the first time in
this line of studies that `cycles:u` has been allowed to carry a claim — it is
allowed here only because the effect is an order of magnitude clear of the
instrument's known floor.

### ★ The confound exists and its sign is favourable

`wpin` computes a wrong evaluation, so it searches a different tree. I measured
what that costs instead of arguing about it — `foldcount` on the `wpin` binary:

| | `base51` | `wpin51` | ratio |
|---|---:|---:|---:|
| evaluations / node | 0.43114 | **0.48415** | **1.1229** |
| weight rows / node | 7.17540 | **7.75405** | **1.0806** |
| `thr_into` rows / node | 3.92215 | 4.31934 | 1.1013 |

**`wpin` does 8.1 % MORE weight rows and 12.3 % MORE evaluations per node than
the base, and is still 13.19 % faster.** So 13.19 % is a **lower bound** on the
prize, not an upper one. Had the tree been held fixed the number would be
larger.

### Cross-check, independent of the panel

Weight rows are 7.1754 rows/node × 1280 B = **9,184 B/node = 143.5 lines/node**,
against a measured engine fill rate of **566.41 fills/node** — i.e. **25.3 % of
every line that enters this engine's L1 is a weight row.** `wpin` removed
19.741 % of fills while *adding* 8.1 % more rows. The two numbers agree.

**Frame re-derived (rule 8):** `instructions:u` = **6153.10/node**, reproducing
study 50's 6153.10 exactly and study 49's 6153.06 to **7 parts per million**.
`cycles:u` 3675.33, `ld_dispatch` 2282.64, fills 566.41.

---

## ★ 3. THE REFUTATION — pollution is 10.4 % of it, and the head's tables do not care

The commission's model: *"The fold's weight rows are read ONCE per eval and
never reused. The head's tables are read EVERY eval. The single-use data is
evicting the reused data."*

Study 50's per-region census only ever ran inside `forward_into`. Running it
**engine-wide** (`tools/perf51/ibsaddr51.py` with an empty symbol filter), base
vs `wpin`, per-run region dumps so study 50's ASLR trap cannot recur:

| region | base miss% | wpin miss% | Δ pp | base miss-latency | Δ latency |
|---|---:|---:|---:|---:|---:|
| `hka_weights` | 84.17 % | **31.35 %** | −52.82 | 306,708 (28.40 %) | **−288,148** |
| `thr_weights` | 73.64 % | **13.18 %** | −60.46 | 287,743 (26.65 %) | **−258,663** |
| `(other)` | 25.59 % | 23.84 % | −1.75 | 132,708 | −10,159 |
| `hka_stack` | 56.60 % | 53.82 % | −2.78 | 84,933 | −15,052 |
| **`l1_weights`** | **91.44 %** | **91.88 %** | **+0.44** | 73,774 | **+10,430** |
| `thr_stack` | 45.21 % | 33.25 % | −11.96 | 66,703 | −29,283 |
| `(stack/mmap)` | 4.55 % | 4.68 % | +0.13 | 49,989 | −7,306 |
| **`l2_weights`** | **98.84 %** | **98.38 %** | **−0.46** | 28,286 | **+684** |
| **`skip_q`** | **96.27 %** | **95.40 %** | **−0.87** | 15,550 | −1,414 |
| `finny_cache` | 69.97 % | 50.13 % | −19.84 | 13,142 | −6,438 |
| `ft_scale` | 37.23 % | 32.95 % | −4.28 | 7,981 | −2,098 |
| **`l3_weights`** | **100.00 %** | **100.00 %** | **0.00** | 1,002 | +616 |
| **`l1_biases`** | 78.26 % | 90.91 % | +12.65 | 752 | +282 |
| `head_ft` | 0.21 % | 0.15 % | −0.06 | 266 | −194 |
| **ENGINE** | **25.06 %** | **17.09 %** | | **1,079,771** | **−610,570 (−56.5 %)** |

**The decomposition:**

| | Δ miss-latency | share of the improvement |
|---|---:|---:|
| the fold's OWN rows (`hka_weights` + `thr_weights`) | −546,811 | **89.6 %** |
| COLLATERAL (every other region combined) | −63,759 | **10.4 %** |

And in the base, **the two weight tables are 55.05 % of ALL engine miss
latency.**

**So the answer to the commissioned question is that it is the wrong question
here.** The fold does pollute — `thr_stack` −11.96 pp and `finny_cache`
−19.84 pp are real and they are the accumulator arrays sitting nearest the fold
in time — but the five tables the commission named, the ones at 89–100 % miss
that motivated the whole study, **do not move at all**. Perfect pollution
control, if the hardware offered it, would buy 10.4 % of 13.19 % ≈ **1.4 % of
frame**, and the hardware does not offer it.

### What DOES evict the head's tables, then?

Not measured, but the arithmetic is not subtle and it is the right place for a
successor to start: **evaluations are 0.43115/node, so ~2.3 nodes of search run
between one evaluation and the next** — TT probes into a 16 MB table, six
1536-byte history planes, movegen, movepick. The head's warm set is
`l1_weights` 20 KB/bucket + `l2_weights` 2 KB + `skip_q` 1.28 KB ≈ 24 KB in a
32 KB L1. It cannot survive that, and no amount of discipline in the fold will
make it.

One asymmetry supports this directly and is worth keeping: **`ft_scale`
(1280 B, read early in `forward_into`) misses 37.23 %, while `l2_weights`
(2048 B, read late, after the sparse L1 matmul has streamed ~6 KB of
`l1_weights`) misses 98.84 %.** Within a single evaluation the head evicts its
own tail.

---

## ★ 4. FIVE NULLS

Reported prominently because four of them are levers a successor would
otherwise re-derive, and one of them nearly took me with it.

### 4.1 Skip all-zero weight rows — 30 of 19,616,896

Adding zero is the group identity, so a row of zeros could be skipped
bit-identically. **Every static statistic screams that this will work:**

| static statistic | `thr_weights` |
|---|---|
| Shannon entropy | **3.5486 bits** stored in 8 |
| values exactly zero | **34.79 %** |
| rows entirely zero | **9,976 of 60,720 = 16.43 %** |

Fold-weighted, which is the only population that can pay:

| | folded rows | of which all-zero |
|---|---:|---:|
| HalfKA | 11,150,925 | **0** |
| FullThreats | 19,616,896 | **30** |

**The zeros are entirely in features that never occur.** 0.00015 % of folded
rows.

### 4.2 Nibble-pack narrow rows — 0.03 % of the bytes

Fold-weighted distribution of the signed bit width each folded row actually
needs (`FOLDROWBITS`):

| signed bits needed | % of folded HalfKA rows | % of folded threat rows |
|---:|---:|---:|
| ≤ 4 | 0.000 | 0.068 |
| 5 | 0.000 | 4.580 |
| 6 | 0.000 | 48.416 |
| 7 | 2.701 | 38.708 |
| **8** | **97.299** | 8.228 |

A per-row width tag storing ≤ 4-bit rows as nibbles saves **0.03 % of
weight-row bytes**.

> **★ A static histogram of a feature-indexed weight table is not evidence about
> the fold's byte traffic.** Study 50's `ft_scale` win generalised badly here and
> I nearly followed it: `ft_scale` is read **in full on every evaluation**, so
> every element is in the touched population and its 15-of-16-bits-are-padding
> result was sound. A feature table's touched population is a *biased sample*,
> and it is biased hard toward the dense, high-magnitude rows. The static read
> said 3.55 bits; the folded read says 8.

### 4.3 Break the 4 KB congruence between the two accumulator stacks

Study 50 measured that `hka_stack` and `thr_stack` are exactly 4 KB-congruent
and left the experiment explicitly unbuilt: *"the clean experiment — offset one
stack by a non-multiple of 4096 and re-run the region census — is two lines and
I did not build it."* I built it (`cong51`: 64 B of padding in `ThreatEntry`,
stride 5184 → 5248, so the same-ply separation becomes `D + 64p` and is
4 KB-congruent for only one ply in 64). Verified from the runtime dump:
`thr_stack` 1,316,736 B → 1,332,992 B = 254 × 5248.

**Bit-identical: 5,001,562 nodes and eval hash `2e1ede95…79c4`, both exact.**

Its own mechanism statistic reads null — the two arrays the arm exists to
separate move in **opposite directions**:

| | base | `cong51` | Δ |
|---|---:|---:|---:|
| `hka_stack` miss rate | 56.60 % | 55.43 % | −1.17 pp |
| `thr_stack` miss rate | 45.21 % | 45.95 % | **+0.74 pp** |

The adjudicated panel (§4.6) agrees: the arm lands **on top of** the
byte-identical null on every counter.

### 4.4 Cancelling add/sub pairs in the threat fold — 1.99 % of threat rows

Study 48 added `THR_CANCEL` and called it a null without publishing the number.
The number is: over 3,077,341 folds carrying 8,339,951 adds and 11,276,945 subs,
there are **194,875 cancelling pairs and zero duplicates** — 389,750 elidable
rows = **1.99 % of threat rows = 1.09 % of all weight rows**, before paying for
the ~10 comparisons per fold that would find them.

### 4.5 Make the fold lazy — it already is, in Stockfish's exact shape

The tempting reading of the census is "0.978 folds/node against 0.431
evals/node, so 55 % of folds are waste". **It is a coincidence and the framing
is wrong.** `do_move` does record-only work (`Meta { dirty, computed: [false;
2] }`); the fold happens inside `evaluate`, walking back to the nearest computed
ancestor — `find_last_usable_accumulator` plus forward apply, plus study 12's
backward fill. The identity is

```
0.43114 evals/node × 2 perspectives × 1.1346 average chain length = 0.978
```

and **1.1346 is already near its floor of 1.0.** The threats channel defers
strictly more than Stockfish does (generation and index resolution too), and
`sota_threats.rs` carries a pre-written rebuttal of exactly this idea:
*"**Laziness here is a SUBTREE property, not a node property** … Any future
'defer it and the nodes that never evaluate stop paying' argument on this stack
has to be written against subtrees."* Closed without a build.

### 4.6 The `cong51` verdict panel — the arm lands ON TOP OF the null

15 rounds, blocked Latin square, realised mean position 1.00 for all three
labels. Reference `cycles:u` 3689.82/node, `instructions:u` 6153.10/node.

| counter | **`cong51`** | **byte-identical null** | verdict |
|---|---:|---:|---|
| `instructions:u` | **−0.000 %** (8/15) | −0.000 % (8/15) | arm confirmed instruction-identical |
| `cycles:u` | **−0.535 %** (9/15, p = 0.61) | **−0.509 %** (9/15, p = 0.61) | indistinguishable |
| `l1-dcache-load-misses:u` | +0.049 % (5/15) | +0.078 % (5/15) | indistinguishable |
| `ls_any_fills_from_sys.all:u` | −0.030 % (9/15) | +0.085 % (6/15) | indistinguishable |
| `ls_dispatch.ld_dispatch:u` | −0.069 % (9/15) | −0.032 % (8/15) | indistinguishable |

**The arm sits on top of the null on every counter, at the same n, with the same
sign counts.** `instructions:u` at exactly 0.000 for both proves the panel is
measuring memory behaviour and nothing else — and there is none to measure. Note
also that the null itself reads **−0.509 % cycles**, which is the instrument
floor on this box and exactly why §2's −13.19 % (470× that, 15/15) is the only
cycles claim this study makes.

**Study 50's open item §7.5 is now closed with a number: the exact 4 KB
congruence between the two 1.3 MB accumulator stacks costs nothing measurable.**
Study 39's `probe-actpin` bound it at −0.22 misses/node by testing address
*reuse*; this tests *conflict*, which is a different mechanism, and it agrees.

---

## ★ 5. CORRECTIONS TO PRIOR STUDIES

1. **Study 50's `ft_scale` region was registered at twice its real width.** It
   halved `FtScale` to `u8` but left `region("ft_scale", …, L1 * 2)`, so the
   registered range covered **1280 B of neighbouring heap**. Its published
   post-arm miss rate of **52.50 %** is therefore contaminated. Fixed here by
   deriving the width from the array (`size_of_val`), and the real figure is
   **37.23 %**. Note this is *below* study 50's own "two ops per line ⇒ 50 %
   floor" model, which is itself informative: a 1280 B array **does** partially
   survive between evaluations, where the head's later tables do not.
2. **Study 48's weight-table sizes are wrong.** It says "a 28.8 MB + 12.6 MB
   table". Measured from the engine's own region registry: `hka_weights`
   **28,835,840 B** ✓ but `thr_weights` **77,721,600 B**, not 12.6 MB — 6.2×
   larger, and 60,720 rows of 1280 B.
3. **Study 49 listed `prefetchnta` on the fold's weight rows as untried.** Tried
   (§1), inert at the hardware level.

---

## ★ 6. THE FORWARD POINTER — and it is not a code change

Every code-side lever at this site is now measured closed:

| lever | status |
|---|---|
| non-temporal loads / `PREFETCHNTA` | dead — **hardware** (§1) |
| fewer bytes per row (zeros, nibbles) | dead — **data** (§4.1, §4.2) |
| fewer rows (lazy folding) | dead — **already implemented**, chain 1.1346 (§4.5) |
| fewer rows (cancellation) | 1.99 % of threat rows, priced (§4.4) |
| load shape (40 × 32 B per row) | at its floor — study 47 |
| layout / congruence | null, arm built and adjudicated (§4.3, §4.6) |
| protecting the victims | 10.4 % of the prize (§3) |

**What is left is the only thing that moves 9,184 B/node: the net architecture.**
Weight-row bytes per node are exactly

```
rows/node (7.1754) × accumulator width (1280 B)
```

and both factors are training decisions, not code. **This study contributes a
number that the width decision did not previously have.** `engines/canonical/`
records the standing finding that width is ~Elo-neutral across 768–1536 (+44 fn
per doubling against −45 speed per doubling). The speed side of that trade is
measured here to be **larger than it was credited**: weight-row streaming is
**25.3 % of all L1 fills, 55.05 % of all engine miss latency, and ≥ 13.19 % of
cycles**, and it scales **linearly** in accumulator width. A 1280 → 1024 step is
worth ~2.6 % of frame on this term alone, *before* the accumulator-stack and
`activate` traffic that also scale with width.

The threat channel is where the rows are: **3.92216 of 7.17540 rows/node =
54.7 %**, at 4.594 rows per fold against HalfKA's 2.279. Halving the *threat*
feature turnover is worth about as much as halving the width, and it costs no
accumulator capacity.

---

## ★ 7. GATES

| gate | result |
|---|---|
| node identity, study frame, `base51` | **5,001,562** — exact |
| node identity, `cong51` (bit-identical arm) | **5,001,562** — exact |
| wl50 eval hash, `base51` and `cong51` | **`2e1ede95…79c4`** — exact, pin unmoved |
| `instructions:u`/node vs studies 49 and 50 | 6153.10 vs 6153.06 / 6153.10 (**+7 ppm**) |
| `foldcount` rows/node vs studies 47/48/50 | **7.17540** vs 7.1754 |
| byte-identical null through THIS harness | `instructions:u` **−0.000 %**, `cycles:u` **+0.028 %** (7/15), fills −0.223 % |
| realised mean panel position | **1.00** for every label |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh` | **exit 0** — `Bench: 587683`, eval hash `2e1ede95…79c4`, d13 nodes `1739588`, all three pins unmoved |
| `cargo clippy --all-targets` on `foldcount` / `rowprobe` / `cong51` | clean, all three |

D24 perft ritual not required: nothing under `movegen`, `board` mutation,
`zobrist` or `attacks` is touched. **No pin moves**, and for the shipped build
nothing can move — every source change in this branch is behind `foldcount`,
`rowprobe` or `cong51`, none of which is in any default or release build, plus
one diagnostic-only region-size fix.

---

## ★ 8. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | `ls_any_fills_from_sys.all` counts lines entering L1d | Cross-checked against the known line count: 8.04 M fills for 7.68 M demand lines streamed, ratio 1.047 | §1 entirely |
| A2 | `probe-wpin` changes only weight-row locality | It substitutes the index, not the kernel; `instructions:u` moves only −0.330 % and `ld_dispatch` −2.761 %, both consistent with a tree change, and `foldcount` confirms the row COUNT rises | §2's ceiling |
| A3 | `wpin`'s tree change biases the ceiling DOWNWARD | Measured: +8.1 % rows, +12.3 % evals — strictly more work | 13.19 % would become an over-estimate rather than an under-estimate |
| A4 | IBS `PERF_SAMPLE_ADDR` is the tagged op's real data address | Architectural; cross-check: region load shares reproduce the kernels' static shapes | §3 |
| A5 | Region dumps are per-run | Enforced — one dump per `perf record`, study 50 §6's trap | §3 |
| A6 | ~120 Elo per doubling of speed | Inherited (exp-0154); its 95 % CI includes zero | Only the Elo translation |

---

## ★ 9. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The IBS miss RATES are inflated by overlap, and I did not correct for it.**
   A cold 1280-byte row is 20 lines read by 40 loads, so the floor for a fully
   cold row is 50 % — yet `hka_weights` reads **84.17 %**. The explanation is
   almost certainly that both loads of a line are in flight before the fill
   lands, so the second is tagged L2 rather than L1, and study 49 already
   measured these loads as ~50-way overlapped. **The consequence is that §3's
   latency column is not cycles** and its shares should be read as shares of
   *unoverlapped* latency. §2's −13.19 % is a real cycles measurement and does
   not depend on this.
2. **The single-run region censuses in §3 and §4.3 carry no null and no
   repetition.** Study 50 ran three and got ±0.2 pp agreement; I ran one per
   arm. The 52-pp and 60-pp moves on the weight tables are far outside any
   plausible run-to-run noise, and so is the flatness of `l1_weights`/
   `l2_weights`/`l3_weights` — but the small rows (`l1_biases` at n = 22,
   `thr_psqt` at +41.7 % latency) are noise and I have not separated them.
3. **What actually evicts the head's five tables is inferred, not measured.**
   §3's closing argument (2.3 nodes of search between evaluations) is arithmetic
   plus one supporting asymmetry, not an experiment. The experiment exists and
   is cheap: a probe that pins the TT probe and the history planes, then re-runs
   the region census.
4. **`cong51` breaks the congruence for 63 of 64 plies, not all 64.** Padding
   changes the stride; the surgical version shifts the base. Given the mechanism
   statistic read null on the 63/64 version, I did not build the other.
5. **Nothing was arbitrated by games, and nothing could be.** `cong51` is
   node-identical, so a fixed-node match is a dead heat by construction.
6. **The backward fill's read rate is unmeasured.** ~266 k folds per frame
   (5.4 % of `hka_into.calls`) reconstruct ancestors speculatively for siblings
   that may never evaluate. There is no counter separating forward from backward
   folds — `foldcount` merges them by design. It is a ≤ 5.4 % pool, which is why
   I left it.

---

## ★ 10. REPRODUCTION

```bash
# 0. a fresh worktree lacks every gitignored path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. THE HARDWARE QUESTION, FIRST, before building any arm
gcc -O2 -march=native -o /tmp/ntprobe tools/perf51/ntprobe51.c
for m in 0 1 2 3 10 11 12 13 99; do
  taskset -c 2 perf stat -e ls_any_fills_from_sys.all,ls_dmnd_fills_from_sys.all,\
ls_hw_pf_dc_fills.all,l1-dcache-load-misses,cycles /tmp/ntprobe $m 20000
done
# mode 99 is the control: hot array only.  If it does not come out near zero
# fills, the test is measuring something else and every other row is void.

# 2. builds + gates
bash tools/perf51/build51.sh base51                      # the control
cp bin51/base51 bin51/null51                             # byte-identical null
bash tools/perf51/build51.sh wpin51    probe-wpin        # the ceiling probe
bash tools/perf51/build51.sh cong51    cong51            # the layout arm
bash tools/perf51/build51.sh fc51      foldcount         # region census
bash tools/perf51/build51.sh fcwpin51  foldcount,probe-wpin
bash tools/perf51/build51.sh fccong51  foldcount,cong51
bash tools/perf51/build51.sh fcrow51   rowprobe          # row sparsity / bit width
./bin51/base51 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn                           # must print 5001562
./bin51/base51 nnue-eval engines/canonical/net.cenn \
    < tools/nnue/calib/wl50.fens | sha256sum             # must be 2e1ede95...

# 3. THE CEILING
P51_CPU=2 P51_BIN=bin51 \
 P51_EV='instructions:u,cycles:u,ls_any_fills_from_sys.all:u,ls_dispatch.ld_dispatch:u,l1-dcache-load-misses:u' \
 P51_PANEL=docs/research/perf-artifacts-51/panel-ceiling.txt \
 bash tools/perf51/exact51.sh 15 base51 wpin51 null51
python3 tools/perf51/med51.py docs/research/perf-artifacts-51/panel-ceiling.txt \
    base51 wpin51 null51

# and the confound control -- wpin does MORE work per node, not less
for b in fc51 fcwpin51; do
  ARCHON_FOLDCOUNT=/home/nick/.cache/perf51/$b.txt taskset -c 2 ./bin51/$b \
    bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
done
grep -E 'eval.calls|\.rows' /home/nick/.cache/perf51/fc{51,wpin51}.txt

# 4. THE ENGINE-WIDE REGION CENSUS.  The "" argument is the whole point: study
#    50's censuses were all filtered to forward_into, which is why the fold's
#    own 55% share of engine miss latency had never been seen.
#    The dump MUST come from the same process as the samples (ASLR).
for arm in fc51 fcwpin51 fccong51; do
  ARCHON_FOLDCOUNT=/home/nick/.cache/perf51/ibs-$arm.fc.txt \
   taskset -c 2 perf record -q -e ibs_op/cnt_ctl=1,swfilt=1/ -c 10000 -d -W \
     -o /home/nick/.cache/perf51/ibs-$arm.data -- ./bin51/$arm \
     bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
  perf script -i /home/nick/.cache/perf51/ibs-$arm.data \
      -F ip,sym,symoff,addr,weight,data_src \
    | python3 tools/perf51/ibsaddr51.py /home/nick/.cache/perf51/ibs-$arm.fc.txt ""
done

# 5. the two data nulls -- fold-weighted, which is the only population that pays
ARCHON_FOLDCOUNT=/home/nick/.cache/perf51/row.txt ./bin51/fcrow51 \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep -E 'FOLDZEROROW|FOLDROWBITS|FOLDROWSTATS|FOLDVALHIST' /home/nick/.cache/perf51/row.txt
```

**Do not park the IBS text on `/tmp`** — it is tmpfs. This study pipes
`perf script` straight into the classifier so the dump is never materialised.

Pick a quiet **physical** core first: sum each SMT sibling pair (`N` and `N+8`
on this box). Pair 2,10 measured 7.6 % against 54.3 % for pair 3,11.
