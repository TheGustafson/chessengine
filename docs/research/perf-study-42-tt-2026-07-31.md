# Perf study 42 — the transposition table, measured at last

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1d,
1 MB L2/core, 32 MB L3, THP `always` / defrag `madvise` · **Net/arch:** D48X,
`sota,arch-d48x`, ACC 1280 · **Base:** `main` at `b7f4e55` ·
**Artifacts and instruments:** `docs/research/perf-artifacts-42/`,
`tools/perf42/`, `perf42data/`

**Commission:** thirty-one perf studies, seven of which mention the
transposition table in passing, and not one measured it. It is the engine's
other big random-access structure, probed once per node, and the Recode POC
explicitly left it in the unquantified residual.

---

## 0. DENOMINATORS — stated up front, because one was just found wrong

Mid-study the coordinator retracted **perf study 36's "8262.9 cycles/node"** as
~1.84x too big and asked every study to name its own denominator.
**Nothing in this study ever used it**, and nothing needed to: every percentage
here is differenced inside a single panel and converted with **that panel's own
measured reference median**, never with an imported constant.

For the record, on **this study's binary** — from-source
`cargo build --release --features sota,arch-d48x`, `target-cpu=native`,
**non-PGO** (all five arms built by the same script, which is what makes them
comparable) — at the pinned frame `bench 16 1 100000 wl50.fens nodes <net>`,
5 001 562 nodes:

| | quiet-ish box (load ~4.6) | inside the 30-round panel (load ~5.5) |
|---|---:|---:|
| `cycles:u` /node | **3992.4** | **4390.8** |
| `instructions:u` /node | **5929.1** | 5929.1 |
| `ex_no_retire.load_not_complete` /node | **1965.9** | 2322.7 |
| IPC | 1.485 | 1.350 |

Two notes on comparability, both mine to own:

* The coordinator measured **5659.0 instructions/node** on the **canonical PGO
  binary**. Mine is 5929.1 on a **non-PGO** build — a 4.6% gap in the direction
  PGO moves instruction counts. Neither is wrong; they are different binaries,
  and an A/B panel must be built one way, so this study is non-PGO throughout.
* `cycles:u` is inflatable by contention and never deflatable, so the quiet
  column is the lower bound. **Every absolute cycle figure below is quoted as a
  range across both columns**, and the percentage — which is what the panel
  actually measures — is quoted first.

---

## ★ HEADLINE

### H1. The TT touches **1.067 cache lines per node** — and that is a hard bound, not an estimate

Three independent facts compose into it, each measured here:

| fact | value | instrument |
|---|---:|---|
| probes per node | **1.066734** | `src/ttcensus42.rs`, exact integer count |
| lines per probe | **exactly 1** | `align42.py`: zero traffic at the pad offsets |
| sibling-cluster reuse | **0** | `same_line` counter, exact |

A `Cluster` is 32 bytes and `align(32)`, so a probe reads one 32-byte half of
one 64-byte line and **never straddles** — verified from the outside, on the
shipped binary, by histogramming the low 6 bits of every observed TT data
address (`tools/perf42/align42.py`, on the Hash = 256 MB capture where the
window is exact):

* **zero** accesses at offsets {30, 31, 62, 63} — the two pad bytes of each
  cluster — out of 4289;
* a **48.7% / 51.3%** split between the two halves of the 64-byte line, which is
  exactly what two 32-byte clusters per line must produce;
* and every non-zero offset lands on a `TtEntry` field: `key16` at cluster
  +0/+10/+20, then `depth8` +2, `gen_bound8` +3, `move16` +4, `value16` +6,
  `eval16` +8 within each entry. **Traffic at a non-field offset: 0 of 4289.**

**Straddle rate: 0.000%.** The module doc said "`align(32)` guarantees a cluster
never straddles a line boundary" and nothing had ever checked it against the
allocator; `no_cluster_straddles_a_64_byte_cache_line` now does, permanently.

Against the engine's **580.9 cache fills per node**, 1.067 lines is **0.18%**.
The TT is a rounding error in cache *traffic*.

### H2. It is not a rounding error in TIME, because every one of those lines is a DRAM-class miss — and the prefetch we already ship is worth 3.5% of the engine

The node-identical arm `ttnopf42` turns `TranspositionTable::prefetch` into a
no-op and changes nothing else. Node counts reproduce bit-for-bit
(5001562 at the study frame, 1739588 at d13), so the counter is the only free
variable.

| | median | sign count | p | per node (quiet — panel) |
|---|---:|---:|---:|---:|
| `instructions:u` | **−0.363%** | 18/18 | <0.0001 | **−21.50** (exact) |
| `cycles:u` | **+3.504%** | 18/18 | <0.0001 | **+139.9 — +153.8** |
| `ex_no_retire.load_not_complete` | **+5.571%** | 17/18 | 0.0001 | **+109.5 — +129.4** |
| `ex_no_retire.empty` | +4.708% | 18/18 | <0.0001 | +8.4 — +9.1 |

The **null** (`base` against a byte-identical copy of itself, same panel, same
rounds, shuffled order) is a coin flip on every one of them: 10/18, 10/18,
9/18, p = 0.81, 0.81, 1.00.

Divide by the exactly-counted 1.067 probes: **an unprefetched TT probe costs
103–121 stall cycles and 131–144 total cycles.** That is memory-latency scale,
and it says what the cache hierarchy already implied — **a TT probe misses
essentially every time**. Priced correctly, the prefetch buys back rather more
than the cycle delta shows: it also *spends* 21.5 instructions/node, worth
~14.5 cycles at IPC 1.485, so the stall it removes is **~154–168 cycles/node**.

The percentage is the measured quantity and the honest headline: **the TT
prefetch is worth 3.5% of engine cycles**, i.e. ~6 Elo at the project's
~120 Elo/doubling constant.

The commission warned that prefetching has been proposed and killed **six
times** in this engine, most recently by study 29. That prior is about *new*
prefetches. The hint measured here is the **incumbent** — it shipped with the
TT in P6, its only justification was a 2026-07-10 standalone microbenchmark,
and doc 08 §4 step 7's "re-measure the search-integrated delta when the call
sites land" had been outstanding ever since. **It is not a seventh proposal; it
is the one we already run, and this is the first time it was measured in the
search.** The seventh and eighth proposals are `spec` and `spec2` below, and
both die (§4.3) — so the prior survives this study intact at 6-for-8.

### H3. ★ A METHODOLOGY CORRECTION: study 39's stall-per-line rate must NOT be applied here, and the commission's own conversion would have been wrong by ~100×

The commission asked me to convert misses to stall cycles with study 39's
confound-free figure of **1.119 stall cycles per 64-byte line**. Doing that:

```
1.067 lines/node  x  1.119 stall/line  =  1.19 stall cycles/node
```

The measured value is **109.5–129.4**. The conversion is off by **92–109×**,
and no choice of denominator rescues it — this is a two-orders-of-magnitude
error, not a framing one.

It is not study 39's number that is wrong — it is the population. Study 39
calibrated on **accumulator streaming** lines: a per-ply stack that the fold
wrote moments earlier, sitting at **1.298 stall cycles per miss** and
0.861 misses per line, i.e. an L2-resident hot window. The TT's lines are the
opposite population by construction: `mul_hi64` spreads a Zobrist key
uniformly over 16–256 MB, so a TT line is cold, is served from L3 or DRAM, and
costs two orders of magnitude more than a line of accumulator.

**A stall-per-line rate is a property of the population, not of the machine.**
Carrying one across studies is exactly the "chained inference" the project's
own method bar forbids, and this study is the receipt.

---

## 2. ✗ RETRACTED IN DRAFT — "IBS shows the prefetch converting the population"

My first draft of this section said: with the hint in place IBS reports
**87–95% of TT demand loads as L1 hits**, on a 16 MB table and on a 256 MB one,
which "is impossible for a uniformly-indexed table 8x the size of L3 unless the
prefetch is converting nearly the whole population."

**That is wrong, and I caught it by running the falsification.** I captured IBS
again on `nopf` — the arm with **no prefetch at all** — and the same block came
back at **90.5% L1 hits**, against `base`'s 89.6%. The statistic does not move,
so it was never evidence about the prefetch.

Two mechanisms, both of which had to be checked rather than assumed:

1. **One probe issues ~8 memory accesses into ONE line.** The first allocates
   the miss; the other seven find the line already resident (or merge into the
   in-flight fill) and are reported as L1 hits. So ~87% L1 is the *layout's*
   signature, not the prefetch's — it is what 8 accesses per 64-byte line must
   produce whether or not anything prefetched it.
2. **IBS reports where the data came FROM, not when it arrived.** A demand load
   that waits on a software prefetch's fill still reports the fill's source. The
   prefetch changes *when* the line lands, which is exactly the axis this
   counter cannot see.

**Consequence, and it is the reason §H2 is built the way it is:** the IBS level
distribution is structurally blind to the question "is the prefetch working".
Only the node-identical differencing arm can answer it, and it does (§H2:
+3.504% cycles, 12–18 of 12–18 rounds, against a clean null). Nothing in this
study's headline rests on the retracted claim.

---

## 3. Where the TT is, and how it was found without instrumenting anything

The attribution trick is worth keeping. `glibc` puts the transposition table
and the NNUE weight tables in the same 200 MB+ heap mapping, so a mapping-level
census returns one blob. But **the TT is the region whose width follows
`Hash`**, and nothing else in the process does:

| `Hash` | uniform block found | width |
|---:|---|---:|
| 16 MB | span offset 198–213 MB | **16 MB** |
| 64 MB | span offset 148–211 MB | **64 MB** |
| 256 MB | span offset 0–255 MB | **256 MB** |

Its share of address-carrying IBS samples is stable across all three
(2.38%, 2.36%, 2.35%) — a table whose *footprint* changes 16× while its
*access rate* does not, which is the signature of a structure probed a fixed
number of times per node.

**And the 256 MB block carries an exact `TtEntry` fingerprint**, which settles
identity on its own without the size argument. Every one of its 4289 sampled
accesses falls on a field boundary of a 10-byte `TtEntry` at cluster offset
0, 10 or 20:

```
non-field offsets: 0 of 4289 (0.00%)      pad bytes {30,31}: 0 (0.00%)
```

No other structure in this engine is 10-byte records in 32-byte groups.
(At 16 and 64 MB my block *edges* are eyeballed off a 1 MB histogram rather
than exact, so those windows carry 2.7–3.6% foreign traffic from whatever
abuts them; the 256 MB window starts at the mapping's own offset 0 and is
exact. Every layout claim above is quoted from that one.)

**Caveat on IBS shares — read this before quoting one.** IBS op sampling here
is cycle-periodic (`cnt_ctl=0`), so a sample share is closer to a share of
*time* than a share of *ops*. Reconstructing the TT's op count from the census
instead:

```
probes/node                                 1.0667
  tag-matched (54.5%): key16 loads by slot  1x1267527 + 2x974061 + 3x665559
                       + is_occupied + read()'s five fields
  missed      (45.5%): 3 key16 + 3x2 for the replacement scan
  ---------------------------------------------------------------> ~8.9 loads/node
saves/node 0.802: ~3 loads + ~6 stores (LLVM merges some)  ------->  ~7.1 ops/node
                                                            total   ~16 ops/node
```

~16 of the engine's ~3030 memory ops per node is **0.53%**, against IBS's
**2.35%** sample share — a ~4.4x over-representation, which is exactly what a
time-weighted sample does to the longest-latency ops in the program. **So the
IBS share is evidence that the TT is slow per access, not that it is frequent.**
Every absolute number in this study comes from the exact census or from a
node-identical differencing arm; IBS is used for attribution and profile,
never for a rate.

---

## 3.1 The census — reference numbers the project did not have

`src/ttcensus42.rs`, exact integers, `Hash=16`, 5 001 562 nodes, node-identical
to `base` at both gate frames:

| | total | per node |
|---|---:|---:|
| probes | 5 335 338 | **1.0667** |
| tag matches | 2 907 147 | 0.5812 |
| **hits** (tag match AND occupied) | 2 907 123 | **0.5812** |
| — matched in slot 0 / 1 / 2 | 1 267 527 / 974 061 / 665 559 | 43.6% / 33.5% / 22.9% of matches |
| saves | 4 013 058 | 0.8024 |
| — accepted by the §3.7 rules | 3 899 672 | 0.7797 (**97.2%**) |
| penalize | 35 512 | 0.0071 |
| prefetch calls | 5 031 939 | 1.0061 |
| **consecutive probes in the same 64 B line** | **0** | **0.0000** |
| consecutive probes in the same cluster | 195 646 | 0.0391 |

Three things worth carrying:

* **hit rate 54.5%**, on a saturated table — the 50-position bench writes 3.90 M
  entries into 1 572 864 slots, turning it over ~2.5x, and the table is never
  cleared between positions (only `new_search` bumps the 5-bit generation).
* **the tag scan is not front-loaded**: 22.9% of matches are in slot 2, so a
  probe usually reads more than one entry — which is why one probe issues ~8
  memory accesses into its line, and why the §2 retraction was necessary.
* **97.2% of saves are accepted.** The replacement rules reject almost nothing:
  the depth/age arithmetic in `TtEntry::save` is doing far less work than its
  five tunables suggest. Those five (`TT_REPLACE_DEPTH_WEIGHT`,
  `TT_REPLACE_AGE_WEIGHT`, `TT_REFRESH_DEPTH_MARGIN`, `TT_REFRESH_PV_CREDIT`,
  `TT_SECONDARY_AGING_*`) were made tunable on 2026-07-27 and — verified, not
  assumed: `grep` over `tools/spsa/` finds them in **no campaign**. If one is
  ever built, it should know first that it is steering a decision that already
  goes one way 97 times in 100, and that the rejected 2.8% is where all of its
  leverage has to come from.

---

## 4. The arms

All five binaries are node-identical (`tools/perf42/ident42.sh`): a cache hint
decides nothing, so both gate frames reproduce bit-for-bit (5001562 / 1739588)
on every one of them.

| label | what it changes |
|---|---|
| `base` | the shipped engine (`.text` identical to `bin34/b42`) |
| `null` | a byte-identical copy of `base` under another name — the noise floor |
| `nopf` | `prefetch` becomes a no-op |
| `pft2` | the hint is demoted `_MM_HINT_T0` → `_MM_HINT_T2` (L2/L3, not L1) |
| `spec` | `Board::prefetch_key` finally gets its caller: a second, speculative hint in the main move loop, the whole pruning cascade earlier than hook A |

Three panels were run, all shuffled-order and within-round paired
(`tools/perf42/panel42.sh`, reduced by `tools/perf40/med40.py`):

| | file | arms | rounds | null's cycles median / sign |
|---|---|---:|---:|---|
| **A** | `panel-h16.txt` | 3 | 19 | −0.506%, 10/18, p=0.81 — **cleanest** |
| **B** | `panel-all-h16.txt` | 5 | 14 | −0.068%, 8/14, p=0.79 |
| **C** | `panel-six-h16.txt` | 6 | 14 | +0.980%, 5/14, p=0.42 |

**More arms made the panel worse, and the null says so.** A six-label round
takes ~30 s, long enough for the box's own load to drift inside one round, and
panel C's null drifts to a +0.98% median even though its *sign count* stays a
coin flip. This is the one place where reading a median without its sign count
would have produced a fake +1% effect. Panel A is quoted wherever it has the
arm; C is quoted only for arms A and B do not carry.

### 4.1 Replicate panel (14 rounds, 5 arms), `base` as reference

| arm | instructions | cycles | load-stall | verdict |
|---|---:|---:|---:|---|
| `null` | +0.000% (6/14, p=0.79) | −0.068% (8/14, p=0.79) | −0.338% (8/14, p=0.79) | **coin flip — panel is readable** |
| `nopf` | −0.363% (14/14) | **+4.051% (12/14, p=0.013)** | **+6.052% (12/14, p=0.013)** | prefetch is real and large |
| `pft2` | +0.000% (6/13, p=1.00) | **+0.558% (11/13, p=0.023)** | +1.217% (10/13, p=0.092) | **T2 is WORSE than T0** |
| `spec` | +1.612% (14/14) | −0.481% (8/14, p=0.79) | −1.075% (8/14, p=0.79) | **null — same sign count as the null** |

### 4.2 A PRE-REGISTERED prediction for `spec2`, written before the data

`spec` costs **+95.58 instructions/node** and returns a cycles/load-stall
result **numerically identical in sign count to the byte-identical null**
(8/14 both). Two readings are possible and they differ in what `spec2` must do:

* **(a) the hint's placement is worth something, and `spec` merely spent it on
  its own overhead.** ~72% of `spec`'s hints land on moves the §3.5 cascade
  then prunes. `spec2` issues the same hint below the cascade, so its hint
  count collapses to roughly the probe count and its instruction cost should
  fall to ~1/4. Prediction: **`spec2` lands negative on cycles with a sign
  count clearly outside the null's**.
* **(b) hook A already saturates the recoverable stall,** and no earlier hint
  can help because the line is already in flight. Prediction: **`spec2` is also
  a null**, and the seventh prefetch proposal in this engine dies like the
  previous six — with the difference that this one is refuted by measurement
  rather than by argument.

**I am explicitly NOT pricing `spec`'s instructions at IPC to infer "recovered
stall".** The machine is stalled on loads 53% of cycles here, so added
instructions partly fill existing bubbles and cost far less than 1/IPC; the
arithmetic "spec should have been +1.61%, it was −0.48%, therefore it recovered
92 cycles" is exactly the unmeasured second inference the project's rule 4
forbids. The measured facts are the cycles and the stall, and both say null.

### 4.3 Verdicts, every arm across every panel that carried it

`cycles:u`, median (sign count, p):

| arm | panel A (3, 19r) | panel B (5, 14r) | panel C (6, 14r) | verdict |
|---|---|---|---|---|
| `null` | −0.506% (10/18, .81) | −0.068% (8/14, .79) | +0.980% (5/14, .42) | coin flip everywhere |
| `nopf` | **+3.504% (18/18, <.001)** | **+4.051% (12/14, .013)** | **+2.825% (14/14, <.001)** | **REAL, ~+3.5%** |
| `pft2` | — | +0.558% (11/13, **.023**) | +1.761% (9/14, .42) | worse than T0 |
| `spec` | — | −0.481% (8/14, .79) | +0.276% (7/14, **1.00**) | **NULL** |
| `spec2` | — | — | +0.405% (8/13, .58) | **NULL** |

**`nopf` is the only arm that repeats.** Three panels, three different label
sets, 44 of 46 rounds on the same side. Everything in §H2 rests on it.

**`spec` and `spec2` are nulls, and that resolves the pre-registration in
favour of branch (b).** Two independent panels put `spec` at 8/14 and 7/14 —
chance — and `spec2`, which was built precisely to remove `spec`'s instruction
overhead, is 8/13. The load-stall counter agrees (7/14 and 9/14). **Hook A
already recovers what there is to recover; issuing the hint earlier buys
nothing, at either price.** `Board::prefetch_key` therefore stays without a
caller — but now for a measured reason rather than by neglect, and this is the
seventh prefetch proposal in this engine to die. It is also the first to die
with a number attached.

**`pft2` (T0 → T2) is worse**, and it is the cleanest frame in the study:
instruction-identical to `base` to six decimal places (5929.087 both), so the
only free variable is where the line lands. It is significant in panel B
(11/13, p = 0.023) and directionally the same but not significant in the noisier
panel C (9/14, p = 0.42) — I report it as **directionally consistent, decisive
in one panel of two**, not as a settled number. Study 29's "T2 lands in L2 where
the lines already hit" does not transfer to the TT, where the lines hit nowhere:
the hint has to go all the way to L1.

---

## 5. Three levers that are DEAD, each with the measurement that kills it

### 5.1 Huge pages for the TT: dead

THP is `always` on this box and the engine gets **zero** transparent huge
pages — `AnonHugePages: 0 kB` across every mapping, at Hash 16 MB and at
256 MB (`tools/perf42/smaps42.sh`). With `defrag = madvise` and the box's free
memory fragmented, a fault that cannot get a 2 MB page instantly falls back to
4 KB, and khugepaged never catches up inside a 4-second bench. So every TT
probe walks 4 KB pages: 4096 of them at 16 MB, 65536 at 256 MB, against a
3072-entry L2 dTLB.

That is a real capacity argument and it is **wrong**:

| | TT L1-dTLB hit | rest of engine | TT share of page walks |
|---|---:|---:|---:|
| Hash 16 MB | 94.78% | 95.33% | **1.63%** |
| Hash 256 MB | 95.38% | 95.26% | **1.99%** |

The TT's TLB behaviour is **indistinguishable from the engine average**, even
at 256 MB.

*A caveat I have to apply here after §2 bit me on the same shape:* the
**per-access hit rate is diluted** — ~8 of a probe's accesses go to one line and
so to one page, so at most 1 in 8 can be a TLB miss, and a ~87.5% hit rate is
the floor for any structure with this access shape. **The share column is not
diluted** (both sides are absolute page-walk counts) and it is the one the
conclusion rests on.

Sized end to end: engine-wide page walks are **1.89/node at 16 MB and
2.72/node at 1 GB** (§6.1), and the TT's share of them is ~2%, i.e. **~0.04–0.05
walks/node**. Even granting the whole +0.83 walks/node the sweep adds between
16 MB and 1 GB to the table, and pricing a walk generously, that is tens of
cycles per node against the **+397.7 stall cycles** the same size change costs.
**A `MADV_HUGEPAGE` allocator (doc 08 §3.13 phase 2, still unimplemented)
would be buying ~5% of the problem, and the raw-syscall/unsafe cost of building
one in a crate with zero runtime dependencies is not worth that.** Dead lever,
priced.

### 5.2 The 32-byte cluster is NOT costing us a straddle — but it is halving the associativity per line fetched

`same_line = 0`, exactly: over 5.34 million probes, **no** probe ever landed in
the 64-byte line the previous probe had just pulled. The sibling cluster that
shares each line is never reused, which is what uniform indexing over 524288
clusters must give.

So the layout is straddle-free (§H1) but **each 64-byte line the TT fetches
carries 3 usable entries and 3 unusable ones**. A 64-byte, 6-entry cluster
would carry 6 usable entries for exactly the same traffic — same lines
fetched, same table size, double the associativity. This is a *quality* lever,
not a cost lever: it cannot reduce misses per probe (already exactly one line),
only improve what the table remembers. It is written up in §7 as the next
session's work, not claimed here — it changes the tree, so it moves `Bench:`
and it needs games.

---

### 5.3 `_MM_HINT_T2` is worse than `_MM_HINT_T0`, measured

Study 29 killed a prefetch proposal partly on the argument that a `T2` hint
"lands in L2 where the lines already hit". The TT is the case where that can be
tested instead of asserted, because a TT line does *not* already hit anywhere.
`ttpft242` demotes the hint T0 → T2 and is **instruction-identical to `base` to
six decimal places** (5929.087 both, median −0.000%) — the same instruction with
a different immediate, which is as clean a frame validation as this study gets.
It **loses** (§4.3): +0.558% cycles, 11/13, p = 0.023. The hint has to go to L1.

---

## 6. THE FRAME — is 16 MB hiding anything?

**The perf frame and the strength frame are the same frame.** Every match
harness in the project defaults to `Hash=16`: `tools/sprt.sh:97`,
`tools/screen.sh:49`, `tools/gauntlet.sh:56`, `tools/spsa/spsa_tune.py:77`,
`tools/spsa/night_gates_v1.sh`, `tools/search_knob_gate.sh:38`. So a TT number
measured at `bench 16 …` describes the engine our Elo actually comes from —
which is the good case, and worth stating explicitly since the coordinator has
just been burned by a frame error.

**The bench does not measure an empty table.** 16 MB is 524288 clusters =
1 572 864 entries; the 50-position bench writes 0.780 accepted saves/node x
5 001 562 nodes = **3.90 million entries**, i.e. it turns the table over ~2.5x,
and the table is not cleared between positions (only `new_search` bumps the
5-bit generation). The measured hit rate of **54.5%** is therefore a saturated
table's hit rate, not a warm-up artifact.

### 6.1 ★ But 16 MB is the CHEAPEST point on the curve, and the curve is steep

Fixed node budget, one core (4), four sizes back to back, `bin42/base`:

| `Hash` | instr/node | cycles/node | load-stall/node | DRAM fills/node | L3 fills/node | page walks/node |
|---:|---:|---:|---:|---:|---:|---:|
| **16 MB** | 5929.09 | **4342.2** | **2200.5** | 9.386 | 52.03 | 1.889 |
| 64 MB | 5918.40 | 4601.6 | 2390.1 | 9.958 | 52.08 | 2.255 |
| 256 MB | 5920.69 | 4666.6 | 2486.5 | 9.697 | 52.48 | 2.160 |
| **1024 MB** | 5910.30 | **4719.9** | **2598.2** | 10.325 | 52.47 | 2.723 |
| | **−0.3%** | **+8.7%** | **+18.1%** | **+10.0%** | +0.8% | +44% |

**Instructions are flat to −0.3% across a 64x change in table size** — the
trees differ, the work per node does not, which is what makes the four columns
comparable at all. Everything else is **monotone in `Hash`** across all four
points: cycles, load-stall, DRAM fills and page walks.

The arithmetic that makes this concrete: 16 MB → 1 GB adds **+397.7 stall
cycles/node** and **+0.94 DRAM fills/node**. That is ~423 stall cycles per
added DRAM fill — a full memory latency — and ~0.94 fills is, to within the
measurement, **the TT's entire line population** (1.067 lines/node, §H1).

**Interpretation, one inference deep and flagged as such:** at 16 MB the table
mostly lives in the 32 MB L3 and a probe is an L3 hit; at 1 GB every probe is a
DRAM miss, and the hook-A prefetch's lead time — the remainder of `make_move` —
is not long enough to hide a full DRAM latency. A confound I cannot separate
with this instrument: a bigger TT also evicts the NNUE weight tables from L3,
so some of the +0.94 fills are the net's, not the table's.

**What this means for the frame.** Our perf frame and our strength frame agree
(both 16 MB), so nothing measured here is wrong *for the engine we test*. But
16 MB is the cheapest point on a steep curve, and a tournament configuration at
256 MB–1 GB pays 7.5–8.7% more cycles per node. **Any TT-related lever
evaluated at `Hash=16` is being evaluated where the TT costs least.**

### 6.2 …but the PREFETCH's share does not grow with `Hash` — and that is the more useful half

I expected the prefetch to be worth much more on a big table and ran the arm
there to check — 15 rounds at `Hash=256`, `panel-h256.txt`, null clean at
8/15 on cycles (p = 1.00) and 9/15 on load-stall (p = 0.61):

| | `Hash` = 16 MB | `Hash` = 256 MB |
|---|---:|---:|
| `nopf` cycles | +3.504% (18/18, p<0.001) | **+3.943%** (12/15, p=0.035) |
| `nopf` load-stall | +5.571% (17/18, p=0.0001) | **+6.112%** (13/15, p=0.0074) |
| `nopf` instructions | −0.363% (18/18) | −0.363% (15/15) |
| panel `base` cycles/node | 4390.8 | 4516.8 |
| ⇒ prefetch worth, absolute | **~154 cycles/node** | **~178 cycles/node** |

**The percentage barely moves: 3.50% → 3.94%.** And that is the finding, because
of what it is set against. Between the same two sizes the engine's cost rises by
**~325 cycles/node** (§6.1, 4342.2 → 4666.6) while the prefetch's absolute value
rises by only **~24** — **the prefetch recovers about 7% of the increment a
bigger table brings.** Its lead time is the remainder of `make_move`, a fixed
budget; a deeper miss simply overflows it, and no amount of extra table makes
the hint cover more.

So the frame answer has two halves and the second is the actionable one:

* **Yes, the TT costs much more at a tournament `Hash`** — +7.5% cycles/node at
  256 MB, +8.7% at 1 GB, entirely un-hidden.
* **No, the mechanism we already ship does not scale to cover it**, and §4.3
  killed both obvious ways of extending it (earlier hint, cheaper earlier hint).

Whatever recovers that increment is therefore **not** about the timing of the
fetch. It has to be about what the fetched line CONTAINS — which is §5.2's
64-byte cluster, and which is why that is the top item in §7.

---

## 7. What to do next, in the order I would do it

1. **The 64-byte, 6-entry cluster (§5.2).** The one lever this study found and
   did not pull. `same_line = 0` says every 64-byte line the TT fetches carries
   three entries it can never use; a 64-byte cluster makes all six reachable for
   **identical** memory traffic and identical table size. It is a hit-rate
   change, so the discrete instrument is right there and is immune to this
   box's timing noise: **`bench 16 1 13 <fens> depth` node totals — fewer nodes
   to the same depth is a better table.** It changes the tree, so it moves
   `Bench:` and it needs games. Costs: the probe scans 6 entries instead of 3,
   and half a dozen replacement-policy tests hard-code a 3-slot cluster and
   would need rewriting against `CLUSTER_SIZE`.
2. **Ask whether `Hash=16` is the right default for our own matches.** Not a
   perf question — a strength one, and this study only supplies the input. The
   bench turns a 16 MB table over ~2.5x in 5 M nodes (§3.1); a 10+0.1 game is
   several times that, so our SPRT/screen/gauntlet harnesses may all be playing
   with a table that is cycling hard. Both engines in a match share the setting,
   so the comparison stays fair — but the *absolute* level does not, and any
   lever that interacts with TT pressure is being judged in an unusual regime.
   The instrument is games, not counters.
3. **Print `clusters.as_ptr()` from a two-line instrument arm** and retire the
   last approximation in §3 (uncertainty 1).
4. **Nothing about huge pages, `_MM_HINT_T2`, or an earlier prefetch.** All
   three are measured dead above; re-opening any of them needs a new mechanism,
   not a new argument.

---

## 8. Gates, and the kit

**Gates.** `bash tools/check.sh` green: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, the feature matrix, `cargo test`
(including the new `no_cluster_straddles_a_64_byte_cache_line`), the release
perft tier, and the pinned depth-6 fingerprint. Every arm is node-identical at
both frames (`tools/perf42/ident42.sh`: 5001562 / 1739588). Every feature this
study adds is **OFF by default**, so the shipped binary is untouched.

**One clippy lesson worth keeping.** The census's first wiring used
`for (_slot, entry) in cluster.entries.iter().enumerate()`, which
`-D clippy::unused-enumerate-index` rejects when the feature is off. The fix
was not an `#[allow]` — it was to derive the slot from the entry's address so
that **the hot loop keeps exactly the shape it had before the instrument
existed**. An instrument that changes the codegen of the path it measures is
not an instrument. (The rebuilt binary reproduces the census file byte for
byte, so nothing above needed re-deriving.)

**The kit** (`tools/perf42/`), for whoever picks this up:

| script | what it does |
|---|---|
| `build42.sh` | one labelled arm into `bin42/`, cgroup-capped, `-j2`, cores 0-1 |
| `panel42.sh` | shuffled within-round paired panel; `P42_MB` makes Hash a panel variable |
| `census42.sh` | per-node engine census, three non-multiplexing event groups |
| `runcensus42.sh` | the exact TT census at one or more Hash sizes |
| `ident42.sh` | node identity at BOTH gate frames — the gate for every arm here |
| `textid42.sh` | `.text` identity between two binaries by path |
| `ibs42.sh` | IBS op capture with data addresses, Hash as a parameter |
| `regions42.py` / `histo42.py` | attribution by DATA REGION and per-bin histogram |
| `align42.py` | the straddle check, from data addresses, on the shipped binary |
| `tlb42.py` | TT TLB behaviour against the engine's |
| `smaps42.sh` | what actually backs the table (THP) |

Reduce any panel with `tools/perf40/med40.py <panel> base`. **Read the null row
first**; §4 is the worked example of a panel whose median lied and whose sign
count did not.

---

## 9. What I am least sure about

Ordered by how much of the study would move if I am wrong.

1. **The BLOCK EDGES at 16 and 64 MB, not the identification.** Identification
   I now consider settled: the 256 MB window carries a perfect `TtEntry`
   fingerprint (0 of 4289 accesses off a field boundary, 0 in the pad bytes)
   *and* the block width tracks `Hash` across three sizes. What is sloppy is
   that at 16 and 64 MB I read the block's edges off a 1 MB histogram by eye,
   so those windows carry **2.7–3.6% foreign traffic**. Every layout number is
   therefore quoted from the 256 MB capture. The two-line arm that prints
   `clusters.as_ptr()` would remove even this, and I did not build it — it is
   the cheapest thing left on the table.

2. **The Hash-sweep confound I could not separate.** A bigger TT evicts the
   NNUE weight tables from L3, so some of the +0.94 DRAM fills/node between
   16 MB and 1 GB belongs to the net, not the table. The *frame* conclusion
   (16 MB is the cheapest point, and it is the point every TT lever has been
   judged at) does not depend on the split; the attribution "that is the TT's
   line population migrating L3 → DRAM" does.

3. **Sweep points are single runs, not panels.** The four Hash points are one
   measurement each, taken back to back on one core while a panel ran on
   another. Monotonicity across four points on three separate counters, at flat
   instructions, is what makes me believe them; a paired panel per point would
   be better and I did not have the box time.

4. **`same_line = 0` only counts CONSECUTIVE probes.** A probe could still hit
   the sibling cluster of a line pulled several probes ago and still resident.
   Uniform indexing over 524288 clusters makes that negligible by construction
   — but "negligible by construction" is an argument, and the counter is a
   measurement of something slightly narrower than the claim it supports.

5. **Everything here is single-threaded.** The TT is the one structure in the
   engine that is *shared* under SMP, and every number in this study is from a
   one-thread bench. Cluster layout, replacement pressure and false sharing all
   change when workers contend; a 32-byte cluster puts two clusters in one line
   and therefore two writers in one coherence unit, which is a cost that
   **cannot appear anywhere in this study's data**. Nothing here should be
   carried into an SMP decision without re-measuring.

6. **`cycles:u` on this box.** The commission's own null panel spans +0.383% to
   +5.266% on a byte-identical binary. My nulls came back coin flips (4/9,
   6/14, 9/18 across three panels) and the arms I call real are 12/14–18/18, so
   I believe the *ranking*. I am much less sure of the absolute cycle figures,
   which is why §0 quotes them as a range and why the percentage leads every
   claim.
