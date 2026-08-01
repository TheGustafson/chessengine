# Perf study 65 — the narrow row is BUILT and BIT-EXACT and it LOSES; the 45 % elision prize has no free predictor

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB
L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`bc2b446a`**
(post-study-64 handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Branch:** `worktree-agent-ab1cf725d9c6150b4`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, non-PGO, pinned to core 2. A GPU training run shared the
box throughout. **Every number below is an exact hardware counter or an exact
integer census**, and the speed claim is made against the arm's **own null in
the same binary** (`ARCHON_NARROW=0`), not against a separate build — so the
box's load cancels in the comparison. No `cycles:u` and no wall-clock panel
appears in this document (studies 63/64's standing refusal).

---

## ★ 0. WHAT LANDED, AND THE TWO ONE-LINE ANSWERS

**Landed and gate-green:**

1. **`narrow65` — the narrow weight row. Built, BIT-EXACT, and measured
   NEGATIVE.** Node-identical (**5,001,562**) and eval-hash exact
   (**`2e1ede95…79c4`**) with the packing live, and against its own null it is
   **+14.32 % `instructions:u`, +3.53 % L1d load misses and +13.23 % L2
   misses** (+15.21 % instructions against the pre-study control, which is the
   same story with the plumbing's −0.218 % folded in). Every memory counter
   moved the *wrong way* despite the fold
   demonstrably reading 3.4 GB fewer bytes. §2 has the mechanism, which is the
   useful part: **the byte cut and the offset-arithmetic saving are the same
   design choice and they cancel.**
2. **Two censuses and their tools** (`foldcount::FW_*`, `LANEMAX`,
   `foldcount::PRED_*`, `tools/perf65/pred65.py`), landed in `77cc4813` before
   any arm was written. Every hook is `#[cfg(feature = "foldcount")]`, so the
   shipped build carries none of it — but note that this study did **not**
   reproduce study 64's `.text`-md5 invisibility proof for that commit (the
   narrow-row plumbing landed on top before a control was rebuilt), and §2.4
   states what was verified instead.

**The narrow-row answer:**

> The route is worth **7.40 % of all weight-row bytes** and the arm collects
> them. It still loses, because keeping each narrow row inside its **unshrunk
> 1280-byte slot** — the trick that made `feature * ACC` offset arithmetic
> survive and saved the 243 KB permutation table study 64 §6.1 priced the route
> with — leaves a **dead 320-byte tail in every narrow row**, and the L2
> streaming prefetcher fetches it anyway. Measured: L2 prefetch hits **+18.90 %**
> and L2 misses **+13.23 %**. **A byte you do not read is only saved if nothing
> else fetches it.**

**The elision answer (deliverable 2, measured, nothing landed):**

> Over the **1,332,313 RFP** and **808,369 qsearch** decision nodes that
> study 64's 61.41 % is made of, **no cheap signal answers the threshold
> comparison at any useful accuracy.** Not one of ten candidates converts a
> single elision at a **zero** false-elide rate. The best FREE predictor
> (`pardelta`) converts **7.36 %** of RFP elisions at 2 % wrong and **65.86 %**
> at 5 % wrong, and **zero** of the qsearch channel at any rate. The two
> signals the brief named specifically are worse than that: **the TT remnant is
> available at six nodes in 2.14 million**, and **the PSQT-only score converts
> nothing a constant does not.** §5.

---

## ★★★ 1. THE NUMBER THE BRIEF GOT WRONG, AND THE ONE THAT DECIDES THE LAYOUT

The brief opened on study 64 §6.1 and read it as a per-fold rate:

> *"86.4 % of threat **rows** fit 6 bits, but only 53.06 % of **folds** do …
> That gap is the whole design problem — a fold is only cheap if every row in it
> is narrow, so the per-fold hit rate is what pays."*

**The 53.06 % is already a per-ROW figure** — study 64's own table header says
`threat, by FOLD (19,616,896 rows)`, and 19,616,896 is `thr_into.rows`, i.e.
folded ROWS. 86.4 % vs 53.06 % is *static rows* vs *fold-weighted rows*: the
1.63× bias study 64 §6.1 correctly flags. The per-FOLD number did not exist, so
this study measured it (`foldcount::FW_*`, `FOLDWIDTH`):

| FullThreats, whole frame | folds | rows |
|---|---:|---:|
| every row ≤ 6 bits (**all-narrow**) | 891,571 (20.88 %) | **1,960,463 (9.99 %)** |
| widest row = 7 bits | 2,387,036 (55.90 %) | 11,943,722 (60.89 %) |
| contains an 8-bit row | 991,727 (23.22 %) | 5,712,711 (29.12 %) |
| **total** | **4,270,334** | **19,616,896** |

Both columns reproduce `thr_into.calls` and `thr_into.rows` to the unit, and the
row-class marginal reproduces study 51 §4.2 / study 64 §6.1 exactly
(10,409,593 / 7,593,311 / 1,613,992 = 53.06 / 38.71 / 8.23 %).

> ### **A whole-fold "all narrow?" predicate converts 9.99 % of folded rows. A per-ROW plane converts 53.06 %. The brief's framing costs 5.31× of the prize.**

The mechanism is visible in `FOLDWIDTHSHAPE`: all-narrow folds are *short*
folds. At one row **52.4 %** of folds are all-narrow — which reproduces the
per-row rate of 53.06 %, as it must — then 30.0 % at two rows, 12.5 % at four,
2.4 % at ten. Narrowness is close to independent across the rows of a fold, so
the all-narrow rate decays geometrically while the row rate does not. **The
per-fold predicate is not merely worse, it is worst exactly where the rows are.**

So the arm dispatches **per row**, hoisted out of the tile walk beside study
57's offset hoist. HalfKA is excluded outright: **0** of its 11,150,925 folded
rows fit six bits, 2.70 % fit seven, and only 36,368 of its 4,893,314 folds
contain even a 7-bit row.

### 1.1 The no-dispatch layout, measured and rejected

There is one layout that would need no per-row dispatch at all: split the
accumulator's **lanes**. If lane `j` satisfies `|w[r][j]| ≤ 31` for *every* row,
that lane can be six bits for every row at once, and a fixed lane permutation is
free (the loader can apply it to the FT, the FT bias and the head's L1/skip
weights once). `LANEMAX` measures it:

| table | lanes | nonzero | max ≤ 31 | max ≤ 63 |
|---|---:|---:|---:|---:|
| `thr_weights` | 1280 | 1277 | **91** | **891** |
| `hka_weights` | 1280 | 1277 | 3 | 4 |

91 lanes at 6 bits and a further 800 at 7 gives 1158 B, which pads to **1216 B =
19 cache lines against 20** — **one line in twenty, for a whole-table
permutation and two decode paths.** Dead before a build. (The `sans8` column
this census also prints is tautological — excluding rows whose peak exceeds 63
trivially leaves every lane ≤ 63 — and is reported here only so nobody quotes it
as a result.)

---

## ★★★ 2. THE ARM: BIT-EXACT, AND EVERY MEMORY COUNTER MOVES THE WRONG WAY

### 2.1 What it does

A FullThreats row whose peak magnitude is ≤ 31 fits six signed bits losslessly.
`narrow65` rewrites such a row, at load, into **960 B = exactly 15 cache lines**
(against 20), **inside its own unshrunk 1280-byte slot**. That last clause is
the whole design: the slot is kept, so `RowOffset::of(f) = f * ACC` is untouched,
every producer is untouched, and the route costs **none** of the 60,720-entry /
243 KB permutation table study 64 §6.1 costed it with. Cache lines are fetched
on demand, so "not read" and "not present" *ought* to cost the same.

The decode is four instructions per 32 lanes against one:

```
vpermb          spread 24 packed bytes so each group of 8 lanes owns a qword
vpmultishiftqb  lift all 8 unaligned 6-bit fields of every qword at once
vpternlogd      (m & 0x3F) ^ 0x20   ⇒  the byte is exactly  w + 32
vpmovzxbw       widen to 32 i16
```

Eight 6-bit fields are 48 bits, so **no field ever straddles a qword** — that is
what makes `vpmultishiftqb` a single instruction here, and it is why the packer
groups by eight. The `+32` is lane-uniform and is undone **once per fold** (and
is exactly zero when a fold has as many narrow subs as narrow adds, which the
modal `(1,1)` shape does), so it costs one `vpsubw` per tile register instead of
one per lane. The narrow dispatch reuses the **same 48-shape grid** the wide path
uses, so a fold does not lose studies 61/62's straight-line coverage merely
because one of its rows is narrow.

### 2.2 It is bit-exact

| gate | result |
|---|---|
| node total, `narrow65` on | **5,001,562** — exact |
| wl50 eval hash, `narrow65` on | **`2e1ede95…79c4`** — exact |
| `cargo test --features sota,arch-d48x,narrow65` | **724 passed, 0 failed** — including the debug accumulator oracle (`assert_matches_refresh`, a from-scratch recompute of both channels inside a real search) and a new `narrow_rows_fold_bit_exactly` that folds the same rows from an 8-bit and a packed table with a **mixed** bitmap and demands the same accumulator |
| `serialize ∘ load` | now the identity on BYTES; the round-trip test caught that a packed table would otherwise have been written back to disk as if it were i8 — **a live net-corruption bug this arm would have shipped** |

### 2.3 And it loses — measured against its own null

`ARCHON_NARROW=0` is the null: **the same binary**, the same `.text`, the same
dispatch, and no packed rows. Everything below is therefore attributable to the
DATA change alone.

| exact counter, whole frame | null | arm | Δ |
|---|---:|---:|---:|
| `instructions:u` | 27,500,885,785 | 31,437,813,914 | **+14.32 %** |
| `L1-dcache-load-misses` | 2,733,915,749 | 2,830,372,632 | **+3.53 %** |
| `l2_cache_req_stat.dc_access_in_l2` | 2,850,922,213 | 2,944,534,095 | **+3.28 %** |
| `l2_cache_req_stat.ic_dc_miss_in_l2` | 205,398,747 | 232,567,251 | **+13.23 %** |
| `l2_pf_hit_l2.all` | 1,102,988,150 | 1,311,372,564 | **+18.90 %** |
| IBS tag-to-retire, 2 reps | 1,203,617 | 1,434,421 | **+19.18 %** |

Node total 5,001,562 and eval hash exact on both, so this is the same tree, the
same rows and the same arithmetic — only the storage differs.

> ### The mechanism, and why it is not fixable inside this layout
>
> The arm's demand traffic really does fall: 10,409,593 narrow rows × 320 B =
> **3.400 GB of the 45.94 GB** the frame folds, **7.40 % of all weight-row
> bytes**. What it buys is nothing, because keeping the 1280-byte slot leaves a
> **dead 320-byte tail** at the end of every narrow row, and the L2 streaming
> prefetcher — which had a perfect 20-line sequential run to work with, and now
> gets 15 lines and a hole — **fetches the tail anyway**. `l2_pf_hit_l2.all`
> **+18.90 %** is the prefetcher doing more work, and `ic_dc_miss_in_l2`
> **+13.23 %** is what that costs: the speculative lines evict weight rows that
> were going to be used.
>
> **The offset-arithmetic saving and the byte saving are the same design choice,
> and it can only be had once.** Compacting the table (narrow plane at 960 B
> stride, wide plane at 1280, a per-feature offset table) removes the hole — and
> re-introduces exactly the 243 KB permutation table read 3.92×/node that
> keeping the slot avoided.

A second, smaller contribution is on the instruction side and is genuine: the
decode is +3 instructions per 32 lanes, i.e. **+120 per narrow row**, which at
2.081 narrow rows/node predicts +250 i/node. Measured is **+787 i/node**, so
roughly two thirds of the instruction cost is *not* the decode — it is the
per-row branch (≈ 63 i/node), the per-fold bias correction (≈ 20), the
narrowness prologue (≈ 24) and whatever the extra live range costs the wide rows
sharing the loop. **The decode is the cheap part of a narrow row; the dispatch
around it is not.** Anyone re-attempting this should cost the dispatch first.

Two smaller facts, both measured, both worth having:

* **The 8-byte tail overread makes it 16 lines, not 15.** The last group of a
  tile loads 32 bytes at byte 936, reaching byte 967 — inside line 15 (960–1023).
  So a narrow row touches 16 lines and saves 4 of 20, not 5. A masked 24-byte
  load on the last group would recover the fifth line; it was not built.
* **The load stream becomes split.** Wide rows load 32 bytes at 32-byte offsets
  off a 64-byte-aligned base and therefore *never* cross a line. Narrow rows load
  at 24-byte offsets, and 3 of every 8 loads cross a line boundary — 55 L1
  accesses per row against 40.

### 2.4 The off-path is functionally identical and slightly cheaper

The plumbing (`apply_threat_deltas_into` gained a `&[u64]`; the from-scratch
rebuild gained a constant-false test) is NOT `#[cfg]`-gated, so the shipped
`sota` `.text` moved: **760,085 → 758,101 B**. It is node-identical, eval-hash
exact, and **−0.218 % `instructions:u`** (27,346,636,578 → 27,286,909,689).
Reported rather than smoothed over: study 64's `.text`-md5 invisibility proof
does **not** hold for this study's landed diff, and the honest gate is the one
`ident61.sh` was written for — node total, eval hash, size and an exact
instruction comparison.

---

## ★★★ 3. THE 45 % ELISION PRIZE: THE CEILING IS NOT FREE

Study 64 §5 measured that **61.41 % of every fresh forward pass** answers one
one-sided threshold comparison and nothing else, and priced removing them at
**≈ 45 % of the engine's L1-miss latency at the free-predictor ceiling.** The
word doing the work is *free*. Nobody had measured whether a free predictor
exists, because every counter in `foldcount` is charged at fold or eval time —
after the value exists.

`foldcount::PRED_*` is charged **at the comparison**. For every decision node it
records the true slack `value − threshold` and each of ten candidates' answer to
the same comparison, cross-tabbed, so sweeping a safety band `b` over
`pred − threshold ≥ b` reads the whole ROC off one table. `const0` is the null
(a constant predictor) and `truth` is the self-test — **`truth` reads
100.00 % converted / 0 wrong at band 0 with a single-bucket residual, so the
census is wired correctly**, which is the failure a null alone cannot catch.

### 3.1 The population

| channel | decision nodes | of which elide | study 64's count |
|---|---:|---:|---:|
| `rfp` — the reverse-futility comparison | 1,332,313 | **1,003,825 (75.34 %)** | `main_fresh_rfp` 977,093 |
| `qssp` — the qsearch stand pat | 808,369 | **351,508 (43.48 %)** | `qs_fresh_standpat` 351,508 |

`qssp` reproduces study 64 **to the unit**. `rfp` is 26,732 higher because this
census's gate is the RFP shape *minus* its two eval-dependent terms and
therefore includes the mate-band nodes `!is_win(eval)` blocks — 2.7 % of the
channel, which is the one reconciliation item and is not load-bearing for
anything below.

### 3.2 The distribution that decides it

True slack quantiles (engine units; 1 SF cp ≈ 2.07 of these):

| channel | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| `rfp` | −400 | 0 | **+144** | +368 | ≥ +1024 |
| `qssp` | ≤ −1024 | −448 | −48 | +112 | +848 |

**27.95 % of RFP elisions and 43.09 % of qsearch elisions clear the bar by under
128 units (≈ 62 SF cp).** Nearly half the prize is decided inside a band that is
smaller than every candidate's own error, and that — not the candidates — is why
the ceiling is not reachable.

### 3.3 The ROC, at six operating points

Coverage = % of that channel's TRUE elisions converted, at the tightest safety
band whose wrong-elision rate is at most the column header. `errpN` are
quantiles of `pred − value`; `-inf` is the under-range bucket.

**Channel `rfp`** (1,332,313 nodes):

| candidate | avail | errp5 | errp50 | errp95 | 0 % | 1 % | 2 % | 5 % | 10 % | 20 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hkaonly` | 99.7 % | −352 | 0 | 608 | — | — | **15.45 %** | 36.96 % | 91.65 % | 98.43 % |
| `psqtonly` | 99.7 % | −inf | −112 | ≥1024 | — | — | — | — | — | 63.49 % |
| `hcpsqt` | 100 % | −inf | −144 | 912 | — | — | — | — | — | 73.03 % |
| `hcpsqt2` | 100 % | −inf | −144 | 624 | — | — | — | 9.06 % | 17.74 % | 87.59 % |
| `parent` | 100 % | −688 | −96 | 304 | — | — | — | 12.17 % | 80.85 % | 97.35 % |
| `gparent` | 99.9 % | −inf | −128 | 240 | — | — | — | 18.77 % | 63.37 % | 84.79 % |
| **`pardelta`** | 100 % | −704 | −96 | **224** | — | **5.05 %** | **7.36 %** | **65.86 %** | 85.99 % | 98.39 % |
| `pargp` | 99.9 % | −1024 | −112 | 240 | — | 3.33 % | 4.83 % | 55.17 % | 77.20 % | 94.05 % |
| `ttval` | **0.0 %** | — | — | — | — | — | — | — | — | — |
| `const0` (NULL) | 100 % | −inf | −112 | ≥1024 | — | — | — | — | — | **62.10 %** |
| `truth` (self-test) | 100 % | 0 | 0 | 0 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

**Channel `qssp`** (808,369 nodes): `hkaonly` reaches **12.84 %** at 5 % wrong
and 21.26 % at 10 %. **Every free candidate is at or below the null at every
operating point, including 20 %.**

Candidates, for the record: `psqtonly` is the head's PSQT term with the net
output zeroed — what a fold touching only the 32-byte PSQ1 rows (1/40th of a
weight row) could produce; `hcpsqt` / `hcpsqt2` the shipped hand-crafted tapered
PSQT eval, raw and ×2 (its units are classic centipawns, ~half this engine's
frame, and a band sweep cannot repair a SCALE error — carrying both is how the
census answers "signal or units?" in one run); `parent` the parent's corrected
static eval negated; `gparent` the same side's eval two plies ago; `pardelta`
`−parent + (hc_now + hc_parent)`, i.e. the parent's eval corrected by the move's
own delta; `hkaonly` the same head over the HalfKA channel alone.

### 3.4 Four measured negatives, three of them on signals the brief named

1. **The TT remnant is empty: `ttval` was available at 4 nodes of 1,332,313 and
   2 of 808,369.** Not 0.0 % as a rounding — six nodes in 2.14 million. It is
   empty *by construction*: a fresh evaluation happens precisely when the TT
   eval field is invalid, and an entry with a valid stored value essentially
   always carries a valid eval too. There is nothing there to read.
2. **`psqtonly` is the null.** Its residual quantiles (−inf / −112 / ≥1024) are
   `const0`'s to the bucket; it converts **zero** at every operating point from
   0 % to 10 % wrong, exactly as the constant does, and beats it by **1.39
   points** (63.49 % vs 62.10 %) at a 20 % error rate. **PSQ1 carries no usable signal in D48X.** This
   kills the one candidate that would have been genuinely free in the ceiling
   sense — a PSQT-only fold reads 32 B per row against 1280, i.e. 2.5 % of the
   traffic — so it is the most expensive negative in this section.
3. **Nothing reaches a zero-wrong band.** Not one candidate converts a single
   elision without also converting one that should not have been. The right tail
   of every residual is heavier than the slack distribution's left edge.
4. **The qsearch channel is refractory.** It is 351,508 of the 1,328,601 (26.5 %)
   and no free signal touches it.

### 3.5 The cost of being wrong

At each candidate's 5 %-wrong operating point, how far *below* the bar were the
nodes it elided anyway (share of wrong elisions by true slack):

| candidate | band | wrong | ≤ −992 | −992..−480 | −480..−224 | −224..−96 | −96..−32 | −32..0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pardelta` | 32 | 31,650 | 2 % | 11 % | 18 % | 23 % | 25 % | 21 % |
| `parent` | 416 | 5,300 | 10 % | 26 % | 27 % | 20 % | 12 % | 6 % |
| `hkaonly` | 432 | 16,196 | 0 % | 3 % | 19 % | 31 % | 28 % | 18 % |

**A wrong elision is not usually marginal.** For `pardelta` at its 5 % point,
**31 % of the wrong elisions are more than 224 units (≈ 108 SF cp) below the
bar** and 13 % are more than 480. These are not "we pruned a node that was one
point short"; they are nodes the search wanted.

### 3.6 Break-even, stated plainly

Write `f` for the share of the 1,355,333-node elidable population converted and
`c` for the predictor's own cost as a fraction of one full evaluation, paid at
**every** fresh-eval node (2,163,490 of them — the predictor cannot know in
advance which nodes will elide). Using study 64's shares (fold 67.21 % of engine
L1-miss latency, head 14.44 %, total 81.65 %):

```
net = 45 % · f  −  81.65 % · c
```

* **Free predictors** (`parent`, `gparent`, `pardelta`; `c ≈ 0.005`).
  `pardelta` at 5 % wrong converts 65.86 % of `rfp` and 0 % of `qssp` ⇒
  `f = 0.488` ⇒ **net ≈ +22 % of engine latency, at 31,650 wrong elisions**. At
  2 % wrong it is `f = 0.055` ⇒ **+2.5 %**. At 1 %, **+1.7 %**. **The entire
  trade lives between a 1 % and a 5 % functional error rate**, and which of
  those is worth anything is a question only games can answer.
* **`hkaonly` is a TIER, not a predictor**, and its arithmetic is different
  because the HalfKA fold is shared rather than duplicated:
  `saving = f · 36.8 % − (1 − f) · 14.44 %`, so **break-even is f = 28.2 %**.
  (The 36.8 % is study 64's measured 55.25 % + 11.96 % apportioned by the threat
  channel's 55.80 % share of folded rows — an apportionment, not a separate
  measurement, and it is the softest number in this section.)
  At 5 % wrong it converts 30.7 % ⇒ **+1.3 % of engine, i.e. nothing**. At 10 %
  wrong it converts 73.4 % ⇒ **+23.2 %**. Note this candidate is the *big* net
  run with the threat channel zeroed — an out-of-distribution input, so it is a
  **lower bound** on what a net actually trained HalfKA-only would do at the
  same cost. That is the shape of the `smallnet_*` machinery this engine already
  carries, fully wired and inert, and it is the shape Stockfish removed
  (`ab8f901d`).

### 3.7 Pre-registered strength-test protocol (NOT run — games are the owner's call)

Nothing is landed behind this section. If it is ever tried, this is the
protocol, registered before any result exists:

* **Arm.** `pardelta` at the **2 %-wrong band (+688)**, applied to the RFP site
  only, returning `(RFP_RET_BETA_W·beta + RFP_RET_EVAL_W·pred)/RFP_RET_DEN`.
  2 % and not 5 %: §3.5 shows the wrong elisions at 5 % are large, and a
  first arm should be the one whose failure mode is bounded.
* **Why the RFP site only.** The qsearch channel converts nothing (§3.3) and
  razoring — the one consumer that needs the predicate and not the value — is
  18,387 evaluations, 1.36 %.
* **Instrument.** Games, not a speed panel. The change is functional by
  construction: it moves node counts, so `bench` node totals and the eval hash
  are both meaningless as gates here.
* **Pair.** Both PGO, both `-C target-cpu=native`, canonical net, on an idle box.
* **Decision rule.** SPRT `H0: 0, H1: +5`, α = β = 0.05, at the standard TC.
  **Pre-registered abort:** if the arm's node count per second does not rise by
  at least 5 % on the fixed-node frame, the mechanism did not fire and the match
  is void rather than negative.
* **Null.** The same binary with the band set to `+∞` (no elision), interleaved.

---

## ★★ 4. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **No wall-clock number, so no Elo, for either deliverable.** Study 63's
   refusal stands and this study did not re-litigate it. The narrow-row verdict
   rests on six exact counters all agreeing in sign against an in-binary null;
   that is strong evidence of direction and says nothing precise about seconds.
2. **The prefetch mechanism in §2.3 is an inference from two counters**
   (`l2_pf_hit_l2.all` +18.90 %, `ic_dc_miss_in_l2` +13.23 %), not a controlled
   experiment. The clean test — disable the L2 prefetcher, or build the
   compacted two-plane layout and see the sign flip — was not run. It is the
   first thing a successor should do, because it decides whether the route is
   dead or merely mis-implemented.
3. **Two thirds of the arm's instruction cost is unattributed** (§2.3: +787
   i/node measured against +250 predicted from the decode). I did not
   disassemble the narrow kernel against the wide one to find where.
4. **The IBS symbol aggregation in this study does not reproduce study 64's
   shape** (`forward_into` 32.4 % here against 14.4 % there, on 2 reps against
   3). I did not reconcile it and have therefore quoted **no** IBS share except
   the total, which is a within-run ratio against the arm's own null.
5. **`hkaonly` is not a small net.** It is the big net with an input channel
   zeroed. Its 91.65 %-at-10 % coverage is a lower bound on a trained
   HalfKA-only tier and an upper bound on nothing.
6. **The `rfp` census population is 2.7 % larger than study 64's**
   `main_fresh_rfp` and the attribution to `!is_win(eval)` is an argument, not a
   counter. It would have cost one more counter to settle.
7. **No games.** The one arm worth playing (§3.7) is not built, and the arm that
   is built is bit-exact and lost on counters.

---

## ★ 5. GATES

| gate | result |
|---|---|
| node identity, study frame, all three binaries | **5,001,562** — exact |
| wl50 eval hash, all three binaries **including `narrow65` on** | **`2e1ede95…79c4`** — exact |
| `cargo test --features sota,arch-d48x,narrow65` | **724 passed, 0 failed, 8 ignored** |
| `cargo clippy --all-targets`, default / `sota,arch-d48x` / `+narrow65` | clean (the two surviving dead-code notes, `read_i8s` and `NNZ_IOTA`, pre-date this study) |
| `cargo fmt` | clean |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | see the appended line below |
| `cargo test --release --test perft_suite -- --ignored perft_full_suite` | see below |
| shipped `sota` `.text` | **760,085 → 758,101 B**, node-identical, eval-exact, **−0.218 % `instructions:u`** — §2.4 |

D24 perft ritual not strictly required (nothing under `movegen`, `board`
mutation, `zobrist` or `attacks` is touched) — run anyway.

---

## ★ 6. WHAT THE NEXT STUDY SHOULD KNOW

1. **The narrow row is not dead, but the slot-preserving layout is.** The byte
   cut is real and collected (3.400 GB, 7.40 % of weight-row bytes). The only
   version that can keep it is the **compacted** one: narrow plane at 960 B
   stride, wide plane at 1280, and the 243 KB per-feature offset table study 64
   §6.1 costed. That table is now the cheaper of two evils, not the expensive
   one — measure it against the +13.23 % L2 misses this study paid to avoid it.
   Everything else — the packer, the scalar decode, the VBMI kernel, the
   dispatch, the bit-exactness test, the `serialize` unpack — is landed and
   reusable; only the addressing changes.
2. **A byte you do not read is only saved if nothing else fetches it.** This is
   the generalisation, and it applies to every future "read less of a hot table"
   proposal in this engine, not just this one.
3. **`ttval` and `psqtonly` are closed as elision signals** — 0.0 % availability
   and indistinguishable-from-constant respectively. Do not re-propose either.
   The PSQT one is the expensive closure: it was the only candidate that was
   free in the ceiling sense.
4. **The 45 % prize is a NET decision** (study 64 §9 item 2 said so and this
   study did not overturn it). The one live route is a **trained HalfKA-only
   tier** with break-even at 28.2 % coverage, and the `smallnet_*` machinery for
   it already ships, inert.
5. **Charge a candidate to the COMPARISON, not to the eval.** `foldcount::PRED_*`
   is one array and two hook sites, and it turned "is there a cheap predictor?"
   into a ten-candidate ROC with a null and a self-test inside it, from one
   engine run. Extend it rather than rebuilding it.
6. **Still open, inherited and untouched:** study 43 §9 item 2's subtractive
   Finny-line arm; study 57 §6's FNV-1a-64 payload checksum; study 48 §9 item
   3's AVX2 `activate` hoist; study 63 §7 item 1's re-run of the +3.14 Elo
   receipt on the repaired wall-clock tool.

---

## ★ 7. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net

bash tools/perf65/build65.sh base                 # the pre-study control
bash tools/perf65/build65.sh fc65 foldcount,rowprobe
bash tools/perf65/build65.sh n65  narrow65

# §1 — the per-FOLD width census and the per-LANE magnitude census
ARCHON_FOLDCOUNT=/var/tmp/s65-fc65.txt taskset -c 2 ./bin65/fc65 \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep -E 'FOLDWIDTH|LANEMAX' /var/tmp/s65-fc65.txt

# §3 — the eval-elision predictor ROC (same dump)
python3 tools/perf65/pred65.py /var/tmp/s65-fc65.txt

# §2.3 — the arm against its own null, in ONE binary
taskset -c 2 perf stat -e instructions:u,L1-dcache-load-misses,\
l2_cache_req_stat.dc_access_in_l2,l2_cache_req_stat.ic_dc_miss_in_l2,l2_pf_hit_l2.all \
  ./bin65/n65 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
ARCHON_NARROW=0 taskset -c 2 perf stat -e ... ./bin65/n65 bench ...   # the null

# §5 — identity and the merge gate
mkdir -p bin61 && cp bin65/base bin65/base2 bin65/n65 bin61/
P61_CPU=2 bash tools/perf61/ident61.sh base base2 n65
bash tools/perf65/test65.sh sota,arch-d48x,narrow65
CHECK_NNUE_GATE=1 bash tools/check.sh
```

**Do not run a panel while anything else builds**, and do not park perf text on
`/tmp` — it is tmpfs.
