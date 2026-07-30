# Perf study 21 — stall-class optimisation, on ARCHON's own terms

**Date:** 2026-07-29 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1d,
1 MB L2/core, 32 MB L3, THP `always` · **Net/arch:** D48X, `sota,arch-d48x`,
ACC 1280 · **Base commit:** `8b851b8`

**Brief:** land four *stall-class* wins — cache misses, alignment, dead memory
traffic, layout, branch mispredicts. Not instruction-count wins; studies 12–19
already took −3.17% there. Gate on **cycles**, median of ≥6 interleaved
rounds; report instructions alongside. Do not rank work by diffing against
Stockfish.

---

## 0. Read this first: three corrections to the brief

### 0.1 The target table is an INCLUSIVE profile, and two rows are stale

The commissioning brief ranked six symbols by "share of our own cycles (d16
bench)". Measured on today's `main` with an **exclusive** (`--no-children`)
`cycles:u` profile on that exact frame (`bench 16 1 16 default depth`,
9,444,932 nodes):

| symbol | brief | measured (d16, exclusive) | measured (wl50 frame) |
|---|---|---|---|
| `search::<NonPv>` | 19.90% | **6.81%** | 6.75% |
| `i16_deltas_into` | 13.14% | 10.46% | 11.08% |
| `finny_refresh_into` | 9.15% | **1.17%** | 1.57% |
| `forward_into` | 8.77% | 9.49% | 9.14% |
| `init_quiets` | 6.35% | 5.19% | 5.56% |
| `current_full` | 3.28% | 3.31% | 3.61% |
| **`i8_deltas_into`** | *not listed* | **14.88%** | 10.86% |

Four of six match closely. Two do not, and both matter: `search::<NonPv>` at
19.90% is an *inclusive* figure (it contains everything the node calls), and
`finny_refresh_into` at 9.15% predates perf study 12's backward fill and study
18's lazy king-cross rebuild, which between them cut refresh work ~8×.

**The single largest exclusive symbol in ARCHON is `i8_deltas_into`, the
FullThreats fold, at 14.88% of d16 cycles — and it was not on the list.**

### 0.2 Symbol shares are frame-dependent; quote the frame

Same binary, same box, two frames (columns above). `i8_deltas_into` moves
10.86% → 14.88% between them. Every number in this document names its frame.

- **Screening frame:** `bench 16 1 100000 tools/nnue/calib/wl50.fens nodes
  engines/canonical/net.cenn` — a fixed **5,001,562-node** tree.
- **Ranking frame:** `bench 16 1 16 default depth engines/canonical/net.cenn`
  — **9,444,932** nodes.

### 0.3 The measurement floor is ~1%, and it is not improvable today

A 400-superbatch trainer holds the box at ~150% CPU and shares the 32 MB L3
with a benchmark whose weight tables are 135 MB. Interleaved paired A/B at
n=15 gives a per-round spread of ±150 to ±600 cycles/node on a mean of ~4160.
**Effects below ~1% are not resolvable here**, and this document says so rather
than reporting them as wins. `instructions:u` per node remains exact (0.0
noise) and is used as the identity gate.

---

## 1. The pre-registered prediction

Written before the first PGO build, as required:

> The named lead is that Stockfish prefetches each feature index the instant it
> is computed, and our phase split leaves that latency exposed. **Prediction:**
> issuing the FT weight-row prefetches early — both HalfKA perspectives before
> either folds, and the FullThreats rows between resolve and fold — will cut
> cycles by ≥1.5% with `instructions:u` **rising** ~2%.
>
> **What kills this model:** a control build that issues the *identical*
> prefetch instructions at a *resident* address measuring the same cycles as
> the real arm. That would mean the address is irrelevant and any apparent win
> is instruction-mix or code layout, not latency.

**The falsifier fired.** See §2.

---

## 2. REJECTED — the prefetch lead is dead, and now properly so

Implemented at both sites: `HalfkaContext::prefetch_pending` (both
perspectives' rows before either fold, ~1000 uops of cover for the second) and
a Phase-1.5 pass in `ThreatContext::at_full` (both perspectives' resolved rows
before either fold). Two lines per row, `_MM_HINT_T0`.

The control (`n1-prefetch-null`) is the same source with one difference: the
row offset is masked to `& 0x3FF`, pinning every hint inside the first 2 KB of
the table — resident from node one. The mask *keeps* `halfka_index` and every
prefetch instruction alive (writing `0` would let LLVM delete them and the
control would no longer be the same shape), and it costs one `and` **more**
than the arm, so the control is conservative.

Screening frame, interleaved paired A/B:

| panel | n | Δcyc_n | Δins_n | Δl1m_n |
|---|---|---|---|---|
| baseline → null (wrong address) | 9 | −5.47% | +2.22% | +3.12 |
| **null → arm (right address)** | **15** | **+0.01%** | +0.54% | −2.92 |
| baseline → arm | 15 | −1.36% | +2.78% | +1.05 |

**Prefetching the row the fold is about to read is worth +0.01%.** The address
does not matter. Whatever the residual −1.36% vs baseline is, it is code layout
or instruction mix — not latency hiding — and I will not bank a 170 i/n patch
on a mechanism its own control refutes. Reverted; the patch is kept at
`docs/research/perf-artifacts-21/rejected-a1-prefetch.patch`.

This supersedes the older, weaker rejections in the source
(`sota_threats.rs`: "−2.5% NPS full-row, ~−0.4% first-two-lines"). Those were
measured on the NPS instrument, whose noise floor is 24%; the −0.4% figure was
never evidence. It is now settled on cycles with a wrong-address control.

**Note the null panel itself:** an inert address change moved the
median-of-medians by −5.47% at n=9. That is rule 1 firing — at n=9, on this
box, the instrument cannot resolve a 1% arm. Every subsequent panel ran at
n≥15 and is read off the *paired* median as well as the medians.

---

## 3. REJECTED before building — FullThreats event cancellation

`build_deltas` regenerates all threat relations at each touched square, so a
relation unchanged by the move would be emitted as both an add and a sub —
a cancelling pair, two full weight rows of pure waste, removable
bit-identically (`+row −row = 0` in wrapping i16).

Instrumented the real fold and measured it instead of building the fix:

```
threat folds=8,205,019   adds/fold=3.23  subs/fold=4.52
cancel/fold=0.075        folds/node=0.8687          (d16 frame)
```

**0.075 cancelling pairs per fold — 0.9% of events.** Deduplication would save
0.15 rows out of 7.75 and cost a comparison loop. Dead. Cost: one instrumented
build, no wasted arm.

The same probe corrected my own estimate: I had reasoned from cycle share to
"~21 features per fold". The truth is 7.75. Measuring beat the inference by 3×.

---

## 4. LANDED

### B1 — the 20 MiB of history tables all started 16 bytes off a cache line

`src/history.rs`, commit `67e88fa`.

Measured directly by printing `ptr % 64` for a real `Histories::new()`:

```
main=16  low_ply=16  capture=48  cont=16  pawn=16  corr=16  cont_corr=16
```

Every history row is already a whole number of cache lines wide (`PieceToHist`
1536 B, a butterfly plane 131072 B, a capture plane 896 B), so row alignment is
decided entirely by the table's base pointer — and `boxed_array`'s `vec![row;
N]` requests `align_of::<Row>()` = **2**, because the leaf is an `i16`. Rust's
`Global` forwards any align ≤ 16 to `malloc`; glibc serves these sizes with
`mmap`; an mmap'd chunk returns `page + 16`. Every 128-byte `[piece][to]` row
therefore spanned **three** cache lines instead of two.

This is exactly the defect perf study 20 found in the NNUE weight tables
(−4.99% cycles there). Nobody had looked at the history tables.

`Line64<T>` raises the outer element's alignment to 64, routing the same `vec!`
through `aligned_alloc`. Zero extra bytes; `Deref`/`DerefMut` keep all ~84
indexing sites compiling unchanged. `corr` is deliberately not wrapped (16-byte
element — aligning it would quadruple a 1 MiB table).

| screening frame, n=15 | baseline | B1 | Δ |
|---|---|---|---|
| cyc_n | 4253.1 | 4229.2 | **−0.56%** (paired median −77.9, 10/15 neg) |
| ins_n | 6086.5 | 6105.9 | +0.32% |
| l1m_n | 623.2 | 620.1 | −3.13/node |

Node-identical (5,001,562), eval-hash-identical (`5ca978e012da1bb0`).

**Honest size:** ~0.5%, at the edge of resolution, and the L1-miss win is
−3.1/node against ~36 predicted from row geometry — most history rows were
already resident, so the third line was usually an L1 hit. The *misalignment*
is measured fact; the *cycle* figure is one standard error from zero. Kept
because it is free, node-identical, gated by two new tests
(`every_history_table_starts_on_a_cache_line`, `line64_adds_no_padding`), and
closes a real latent defect — not because the cycle number alone carries it.


### B4 — study 20's alignment sweep stopped one field short

`src/eval/nnue/{sota_head,sota_loader,sota_features}.rs`, commit `74df5af`.

Perf study 20 arm 5 gave `l1_weights` and `l2_weights` the `Aligned` treatment
and left three siblings as plain `Box<[f32]>`: `l1_biases`, `l2_biases`,
`l3_weights`.

Verified before fixing, by writing the gate test first and watching it fail:

```
head_f32_blocks_start_on_a_cache_line ... FAILED
  assertion failed: l1_biases base is 32 mod 64
```

Each is read per eval as a **whole block** — `[f32; L2]` is 64 bytes, `[f32;
L3]` is 128 — at a bucket-strided offset, and both strides are multiples of 64.
So the base pointer alone decides whether every one of those `_mm512_loadu_ps`
crosses a cache line, and at 32 mod 64 every single one did. Fix:
`read_f32s` → `read_f32s_aligned` at the three load sites (the helper already
existed) and `Box<[f32]>` → `Aligned<f32>` on the three fields.

### B2 — three head kernel reshapes: instructions down, **cycles a null**

`src/eval/nnue/sota_simd.rs`, commit `41f6f80`. Landed, but **not counted as a
stall win**, and the commit message says so.

`find_nnz_vbmi2_inner` 16 → 32 groups per iteration (`vpcompressw` is a
full-`zmm` instruction; the kernel used only the 256-bit form); `l1_dot_inner`
and `skip_dot_i8_inner` split from one `vpdpbusd` accumulator to four (~4-cycle
latency, ~1/cycle throughput ⇒ a 39-deep chain costs 156 cycles for 39 cycles
of work).

| b1 → b2, screening frame | Δcyc_n (medians) | paired median | negative | Δins_n |
|---|---|---|---|---|
| n=15 | +0.34% | −16.7 | 11/15 | −0.46% |
| n=25 | +1.05% | −15.1 | 13/25 | −0.46% |

**The two statistics disagree in sign across both panels — the signature of no
effect.** The latency model behind the chain splits is therefore *not*
supported: the head appears throughput-bound at this width, not
dpbusd-latency-bound. Kept because it is bit-identical, strictly cheaper in
instructions, and gives both dot loops constant trip counts — landed on the
instruction axis, excluded from the win count.

---

## 5. REJECTED — `Stack` 36 → 32 bytes with a 64-aligned array

Removed the write-only `Stack::ply` field (written in `init_stack`, read by
nothing but this module's own assertion), taking `Stack` from 36 to 32 bytes so
two entries share a cache line, and wrapped the array in a 64-aligned
`StackArray` so the shrink was not cancelled by a `page + 16` base. The target
was the six-deep `stack[cur-6 ..= cur]` continuation-history read on every
node: 216 unaligned bytes → 192 bytes on three aligned lines.

| b2 → b3, screening frame, n=21 | value |
|---|---|
| Δcyc_n (medians) | **+0.81%** |
| paired median | **+24.8** |
| negative rounds | 8/21 |
| Δins_n | −0.39% |

**Both statistics agree: a regression**, with a tight spread (−140 to +162).
Fewer instructions, more cycles. Reverted. I do not have a mechanism for why —
the honest statement is that shrinking and re-aligning this struct made the
generated code slower and I did not chase it further.

---

## 6. The instrument, and why you can believe §7

Run last, after every arm: **the same binary against itself**, two names.

| baseline vs baseline-copy, n=25 | value |
|---|---|
| Δcyc_n (medians) | **+0.01%** |
| paired median | +15.3 |
| negative rounds | **12/25** |
| spread | −93.7 … +76.3 |

The harness has no positional bias, and on a quiet box the paired spread is
±90, not the ±550 seen mid-session. So: the paired median is the statistic to
trust, the medians-of-arms drift with external load, and 12/25 is the coin-flip
reference against which the sign counts below should be read.

---

## 7. Cumulative result

`baseline` → `b4-final` (B1 + B2 + B4), screening frame, two independent
25-round panels:

| panel | Δcyc_n (medians) | paired median | negative | Δins_n | Δl1m_n |
|---|---|---|---|---|---|
| first | −3.70% | −36.2 | 18/25 | −0.07% | −5.72 |
| replicate (quiet box) | **−1.35%** | **−61.3** | 18/25 | −0.07% | −6.82 |
| *self-null reference* | *+0.01%* | *+15.3* | *12/25* | *0.00%* | *−1.18* |

**Headline: −1.35% cycles at −0.07% instructions** — a pure stall win, which is
the class asked for. 36/50 rounds negative across two panels against a 12/25
null. L1 misses −6.8/node out of 623.

Identity, every arm including the rejects: node count **5,001,562** and eval
hash **`5ca978e012da1bb0`** unchanged throughout. Full gate green, all three
pins reproduced: **`Bench: 587683`**, eval hash `5ca978e0…8b38`, d13 nodes
**1922947**.

---

## 8. Scorecard, honestly

The brief asked for four landed stall wins. **Two landed** (B1, B4 — both
alignment, both verified by direct measurement of the defect before the fix).
One landed on the instruction axis with cycles unresolved (B2). **Three
rejected**, each with the measurement that killed it: the prefetch lead (by its
own wrong-address control), FullThreats event cancellation (by instrumenting
the fold instead of building the fix), and the `Stack` shrink (a measured
regression).

Why not four: at a ~1% resolution floor set by a trainer I must not disturb,
every remaining candidate I could identify is worth well under 1% individually.
Three of the six symbols in the brief turned out to be either mis-ranked (§0.1)
or structurally immovable without changing the eval hash (§9.1). I would rather
report two verified wins and three controlled rejections than four arms at the
noise floor.

**What actually moved: alignment, twice, both times in a table nobody had
looked at.** That is now three instances across two studies. The generalisable
rule is in §9.2.

---

## 9. Ranked remainder

### 9.1 The one big item: ARCHON maintains TWO 1280-wide accumulators

Independently reached by my own traffic analysis and by the head review. Per
node ARCHON reads+writes four `[i16; 1280]` accumulators (HalfKA × 2, threat ×
2) = ~20 KB of pure copy traffic, and `activate_inner` issues **160 zmm loads =
10,240 bytes per eval** because every output byte needs a lane from two
separate accumulators. Stockfish folds both feature sets into **one**
accumulator and merges with a single `vec_add_16`.

Measured context: the FT tables are 57.7 MB (HalfKA) + 77.7 MB (threat) =
**135 MB against a 32 MB L3**; the fold kernels plus the head own **65% of all
L1 misses**; total 496 L2 fills, 61 L3 fills and 6.4 DRAM fills per node.

This is the largest single number available and it is **not** a perf-study
item: the merge is `vpaddsw(p, 4t)` with saturation, so folding the threat
deltas into the HalfKA accumulator changes saturating to wrapping arithmetic
and **moves the eval hash**. It needs the fidelity battery and a pin move, the
process perf study 8 §4.6 followed. Next study's item 1.

### 9.2 `corr` — the same alignment defect, still open

`Histories::corr` was measured at **16 mod 64** and deliberately left unwrapped
in B1: its element is a 32-byte `[CorrBundle; 2]`, so `Line64` would pad it to
64 and double a 1 MiB table. But at base 16 mod 64 with a 32-byte stride,
**every other element straddles a cache line** (element k spans
`[16+32k, 47+32k]`; odd k crosses the boundary). The zero-cost fix is to wrap
*pairs* — `Box<[Line64<[[CorrBundle; 2]; 2]>; CORR_SIZE/2]>` is exactly 64
bytes per wrapper and the same total memory — at the cost of `corr[k>>1][k&1]`
at the four use sites. Untested; `corr` is touched only ~2-4× per evaluated
node, so expect it to be small.

**The generalisable rule this study earned:** in this codebase, *any*
`Box<[T]>`/`Vec<T>` whose element alignment is ≤ 16 and whose rows are
cache-line multiples is sitting at 16 (or 32, or 48) mod 64. Three separate
tables were found this way across two studies. A crate-wide alignment sweep
test — one assertion per large heap array — is cheap and would close the class
rather than one instance at a time.

### 9.3 Candidates identified but not built

Ranked by my estimate, all below the resolution floor individually:

1. **`init_quiets`'s butterfly index is to-major.** `main[us][raw16()]` has
   byte offset `2·from + 128·to`, so the cache line is decided by `to`; movegen
   emits quiets grouped by *from*-square, so every quiet lands on its own line
   (~28 lines/node from a 128 KiB plane). From-major `(from<<6)|to` would halve
   that. Costs ~4 ALU ops per scored quiet (+2% instructions) to save ~14 line
   touches — the two axes fight and I do not know which wins. Bit-identical
   (a bijection over the 16-bit space, every cell initialised to one value).
2. **Hoist `see_ge` out of the quiet scoring loop.** The `call` at
   `movepick.rs:745` forces six loop-invariant row pointers onto the stack and
   they are reloaded every iteration. Record check candidates, apply the bonus
   after the loop.
3. **`propagate_l3` is 56 narrow instructions for a 32-element dot** — the last
   hot head path still on autovectorisation (16 `vmovsd` + xmm FMAs + a
   15-instruction extract tree). Wants an `l3_dot` kernel beside `l2_tail`.
   Bit-identical if it reproduces the pinned 16-lane + halving-tree shape.
4. **`ThreatPosition` snapshots on every made move** — 80 bytes written per
   push, but only ~43% of pushes are ever evaluated (folds/node 0.8687 over two
   perspectives). ~46 bytes/node of dead write traffic. Hard to make lazy: the
   fold needs ancestor positions the board no longer has.
5. **`l2_tail_inner`'s two 16-deep FMA chains** — 64 cycles of latency for 32
   of work. **Not bit-identical** (f32 addition is not associative and the
   i-sequential order is a pinned contract); needs the fidelity battery and a
   pin move.

### 9.4 Do not re-propose

- **Software prefetch of FT weight rows**, at any site, in any width. Settled
  in §2 with a wrong-address control: **+0.01%**.
- **FullThreats add/sub cancellation**: 0.075 pairs per fold (§3).
- **Huge pages for the weight tables**: THP is already `always` on this box.
- **`Entry`/`ThreatEntry`/TT-cluster alignment**: all already correct.
  `Cluster` is `align(32)` at 32 bytes, which genuinely suffices — a cluster
  never straddles a line regardless of its base.

---

## 10. Artifacts

`docs/research/perf-artifacts-21/`

- `scripts/probe21.sh` — build + identity probe (nodes, eval hash, `instructions:u`)
- `scripts/ab21.sh` — the accept gate: interleaved paired A/B on cycles
- `scripts/flat21.sh`, `scripts/flatd16.sh` — flat attribution, one per frame
- `probes-fast.txt`, `ab.txt` — every number in this document
- `flat-base-cycles.txt`, `flat-d16-cycles.txt`, `flat-base-l1miss.txt`,
  `flat-base-brmiss.txt` — the four profiles §0.1 and §9.1 rest on
- `rejected-a1-prefetch.patch` — the prefetch arm, kept so it is not rewritten
