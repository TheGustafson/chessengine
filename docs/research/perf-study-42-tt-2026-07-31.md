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

| | median | sign count | p | per node |
|---|---:|---:|---:|---:|
| `instructions:u` | **−0.363%** | 18/18 | <0.0001 | **−21.50** |
| `cycles:u` | **+3.504%** | 18/18 | <0.0001 | **+153.8** |
| `ex_no_retire.load_not_complete` | **+5.571%** | 17/18 | 0.0001 | **+129.4** |
| `ex_no_retire.empty` | +4.708% | 18/18 | <0.0001 | +9.1 |

The **null** (`base` against a byte-identical copy of itself, same panel, same
rounds, shuffled order) is a coin flip on every one of them: 10/18, 10/18,
9/18, p = 0.81, 0.81, 1.00.

Divide by the 1.067 probes: **each TT probe costs ~121 stall cycles and
~144 total cycles when it is not prefetched.** That is memory-latency scale,
and it says the thing the cache hierarchy already implied — **a TT probe
misses essentially every time**. The prefetch buys back ~168 cycles/node once
the 21.5 instructions it costs are priced back in (at the engine's IPC of
1.485, those instructions are ~14.5 cycles).

At ~120 Elo per speed doubling, the prefetch we already ship is worth
**~6 Elo**. It is the sixth prefetch proposal in this engine's history and the
first one anybody measured.

### 3. ★ A METHODOLOGY CORRECTION: study 39's stall-per-line rate must NOT be applied here, and the commission's own conversion would have been wrong by ~100×

The commission asked me to convert misses to stall cycles with study 39's
confound-free figure of **1.119 stall cycles per 64-byte line**. Doing that:

```
1.067 lines/node  x  1.119 stall/line  =  1.19 stall cycles/node
```

The measured value is **129.4**. The conversion is off by **109×**.

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

**Caveat on IBS shares.** IBS op sampling here is cycle-periodic
(`cnt_ctl=0`), so a sample share is closer to a share of *time* than a share
of *ops*, and long-latency ops are over-represented. The TT's share of samples
(2.35%) is therefore **not** its share of memory operations — the census says
it issues ~11 memory ops per node out of ~3030, i.e. 0.4%. Every absolute
number in this study comes from the exact census or from a node-identical
differencing arm; IBS is used for *attribution and profile*, never for a rate.

---

## 4. The arms

All five binaries are node-identical (`tools/perf42/ident42.sh`): a cache hint
decides nothing, so `Bench:` is unmoved on every one.

*(panel numbers filled in below from `docs/research/perf-artifacts-42/panel-all-h16.txt`)*

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

## 6. Gates

- `cargo test`, `cargo clippy --all-targets` clean.
- All arms node-identical at both frames.
- Default build untouched: every feature added is OFF by default.

---

## 7. What I am least sure about

*(filled in at the end)*
