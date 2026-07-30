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

### 1. The TT touches **1.067 cache lines per node** — and that is a hard bound, not an estimate

Three independent facts compose into it, each measured here:

| fact | value | instrument |
|---|---:|---|
| probes per node | **1.066734** | `src/ttcensus42.rs`, exact integer count |
| lines per probe | **exactly 1** | `align42.py`: zero traffic at the pad offsets |
| sibling-cluster reuse | **0** | `same_line` counter, exact |

A `Cluster` is 32 bytes and `align(32)`, so a probe reads one 32-byte half of
one 64-byte line and **never straddles** — verified from the outside, on the
shipped binary, by histogramming the low 6 bits of every observed TT data
address: **zero** accesses at offsets {30, 31, 62, 63} (the two pad bytes of
each cluster), and a 48.7% / 51.3% split between the two halves of the line,
which is exactly what two 32-byte clusters per line must produce.
**Straddle rate: 0.000%.** The design comment was right and had never been
checked against the allocator.

Against the engine's **580.9 cache fills per node**, 1.067 lines is **0.18%**.
The TT is a rounding error in cache *traffic*.

### 2. It is not a rounding error in TIME, because every one of those lines is a DRAM-class miss — and the prefetch we already ship is worth 3.5% of the engine

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
~120 Elo/doubling constant. It is the sixth prefetch proposal in this engine's
history and the first one anybody measured.

### 3. ★ A METHODOLOGY CORRECTION: study 39's stall-per-line rate must NOT be applied here, and the commission's own conversion would have been wrong by ~100×

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

## 2. What the TT costs *today*, with the prefetch on

Section 1 prices the TT **unprefetched**. With the hint in place, IBS reports
that **87–95% of TT demand loads are L1 hits**, on a 16 MB table *and on a
256 MB one* — impossible for a uniformly-indexed table 8× the size of L3
unless the line is already there when the load retires. The prefetch is not
merely helping; it is converting nearly the whole population.

The residual is therefore small, and §4 measures it directly rather than
inferring it.

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

### 4.3 Authoritative panel (6 arms)

*(panel results below)*

---

## 5. Two levers that are DEAD, with the measurement that kills each

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
at 256 MB, because `prefetcht0` walks the page table too — the same hint that
hides the cache miss hides the TLB miss. Engine-wide page walks are
2.67/node; the TT's share of them is ~0.05/node. A `MADV_HUGEPAGE` allocator
(doc 08 §3.13 phase 2, still unimplemented) would be buying that.

### 5.2 The 32-byte cluster is NOT costing us a straddle — but it is halving the associativity per line fetched

`same_line = 0`, exactly: over 5.34 million probes, **no** probe ever landed in
the 64-byte line the previous probe had just pulled. The sibling cluster that
shares each line is never reused, which is what uniform indexing over 524288
clusters must give.

So the layout is straddle-free (§1) but **each 64-byte line the TT fetches
carries 3 usable entries and 3 unusable ones**. A 64-byte, 6-entry cluster
would carry 6 usable entries for exactly the same traffic — same lines
fetched, same table size, double the associativity. This is a *quality* lever,
not a cost lever: it cannot reduce misses per probe (already exactly one line),
only improve what the table remembers. It is written up in §7 as the next
study's work, not claimed here — it changes the tree and needs games.

---

## 5.3 `_MM_HINT_T2` is worse than `_MM_HINT_T0`, measured

Study 29 killed a prefetch proposal partly on the argument that a `T2` hint
"lands in L2 where the lines already hit". The TT is the case where that can be
tested instead of asserted, because a TT line does *not* already hit anywhere.
`ttpft242` demotes the hint T0 → T2 and is **instruction-identical to `base` to
six decimal places** (5929.087 both, median −0.000%) — the same instruction with
a different immediate, which is as clean a frame validation as this study gets.

*(result below)*

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

*(sweep + the Hash-256 arm below)*

---

## 7. Gates

- `cargo test`, `cargo clippy --all-targets` clean.
- All arms node-identical at both frames.
- Default build untouched: every feature added is OFF by default.

---

## 7. What I am least sure about

*(filled in at the end)*
