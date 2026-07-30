# Perf study 22 — the dual accumulator, and what is actually in `sota_accumulator.rs`

**Date:** 2026-07-29
**Scope:** `src/eval/nnue/sota_accumulator.rs` (+ `sota_context.rs` where the
stack is driven). `sota_simd.rs`, `sota_head.rs`, `sota_loader.rs`,
`sota_features.rs`, `history.rs` belong to the concurrent study 21 and were
read but never edited.
**Frame, quoted with every number in this document:**
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
= 50 FENs × 100k nodes, one thread, 16 MB TT, D48X net, `sota,arch-d48x`,
`-C target-cpu=native`, thin LTO, 4 CGUs — **a fixed 5,001,562-node tree**.
Every arm below is node-identical at 5,001,562 and eval-hash-identical at
`5ca978e012da1bb0` unless it says otherwise.

---

## 0. The headline, in one paragraph

The premise this study was redirected onto — *"ARCHON maintains **two** 1280-wide
accumulators where Stockfish maintains **one**"* — is **false**, and the pinned
reference says so directly. At `eca43a97`, `AccumulatorStack` declares
`psq_accumulators` **and** `threat_accumulators` as two separate `MaxSize`-deep
arrays, each holding a full-width `int16` accumulator per colour, refreshed by
two different strategies and combined only at `transform()` time. That is
ARCHON's structure, feature for feature. §1 gives the citations. The three
things this study then landed in `sota_accumulator.rs` are real but ordinary:
a hot/cold split of the per-ply stack entry, a boxed-array conversion that was
worth more than the split it was written to rescue, and a probe-first
relayout of the Finny cache slot. Together: **−1.06% cycles against a
−0.07% null**, later refined.

---

## 1. RETRACTION OF THE PREMISE: Stockfish keeps two accumulators too

### 1.1 What was claimed

The study-21 remainder that redirected this study reads:

> ARCHON maintains **TWO** 1280-wide accumulators where Stockfish maintains
> **one** — ~20 KB/node of copy traffic, **65% of all L1 misses**, 135 MB of
> weight tables against a 32 MB L3. It's a pin-moving refactor
> (saturating→wrapping merge), not a perf-study arm.

### 1.2 What the reference actually contains

`stockfish_reference/Stockfish/src/nnue/nnue_accumulator.h`, at the pinned
constants revision `eca43a97`:

```cpp
    std::array<AccumulatorState<PSQFeatureSet>,    MaxSize> psq_accumulators;
    std::array<AccumulatorState<ThreatFeatureSet>, MaxSize> threat_accumulators;
    std::size_t                                             size = 1;
```

and `nnue_architecture.h`:

```cpp
using ThreatFeatureSet = Features::FullThreats;
using PSQFeatureSet    = Features::HalfKAv2_hm;
constexpr IndexType TransformedFeatureDimensionsBig = 1024;
```

Each `AccumulatorState` holds an `Accumulator<Big>` **and** an
`Accumulator<Small>`, and `Accumulator<Size>` is
`std::array<std::array<std::int16_t, Size>, COLOR_NB>` plus its own
`psqtAccumulation` and `computed` flags. So per ply Stockfish carries:

| | ARCHON (D48X) | Stockfish `eca43a97` |
|---|---|---|
| PSQ/HalfKA accumulator | `[[i16; 1280]; 2]` = 5120 B | `[[i16; 1024]; 2]` + `[[i16; 128]; 2]` = 2304 B |
| Threat accumulator | `[[i16; 1280]; 2]` = 5120 B | `[[i16; 1024]; 2]` + `[[i16; 128]; 2]` = 2304 B |
| **accumulator stacks** | **2** | **2** |
| PSQT bucket sums | 2 × `[[i32; 8]; 2]` | 2 × `[[i32; 8]; 2]` (per size) |
| refresh strategy | Finny set-diff (HalfKA) / from-scratch (threats) | `update_accumulator_refresh_cache` (PSQ) / `update_threats_accumulator_full` (threats) |
| where they combine | `activate(out, pst, threat, stm)` | `FeatureTransformer::transform`, `accumulation` + `threatAccumulation` |

`AccumulatorStack::evaluate` makes the shape explicit — four independent folds,
two per feature set:

```cpp
    evaluate_side<PSQFeatureSet>(WHITE, ...);
    if (UseThreats) evaluate_side<ThreatFeatureSet>(WHITE, ...);
    evaluate_side<PSQFeatureSet>(BLACK, ...);
    if (UseThreats) evaluate_side<ThreatFeatureSet>(BLACK, ...);
```

`evaluate_side` then branches on the feature set for the refresh:
`update_accumulator_refresh_cache` for PSQ (the Finny cache), and
`update_threats_accumulator_full` — a **from-scratch rebuild** — for threats.
That is exactly ARCHON's `HalfkaContext::finny_refresh_into` /
`ThreatContext::discharge_refresh` split.

### 1.3 Where the "one accumulator" reading came from

From ARCHON's own comment in `sota_simd.rs`'s `activate_inner`:

> Stockfish — whose two feature sets fold into ONE 1024-wide accumulator, not
> two — merges the threat channel with a single
> `vec_add_16(acc, vec_convert_8_16(column))`.

That sentence is about the **quantisation frame**, not the number of
accumulators: SF's two feature sets share one quantisation scale, so its
threat column is added with no rescale, whereas ours needs the `<<2`. The
"ONE 1024-wide accumulator" phrasing is wrong on its face — SF's threat fold
in `update_threats_accumulator_full` writes into `threatAccumulation`, a
distinct array — and it is the origin of the claim that redirected this study.
**It should be corrected in place.** (Not done here: `sota_simd.rs` is study
21's file this session.)

### 1.4 Why a merge is not free even in principle

Suppose we merged anyway, keeping one `M = pst + 4·threat` per ply. The
blocker is that the two feature sets have **different refresh triggers**, and
the merged accumulator must refresh on the *union*:

* HalfKA refreshes whenever `halfka_king_key(perspective, king)` changes.
  That key is `bucket*2 + (file ≤ 3)`, and `KING_BUCKETS` is
  rank-major/file-mirrored, so the map square → key is a **bijection on all 64
  squares** — every own-king move changes it. (Checked by hand on rank 1:
  keys 1,3,5,7,6,4,2,0 for a1..h1.) Rate: **0.1611 refreshes/node**
  (`compute`'s own comment, perf study 12 §6).
* FullThreats refreshes only when that perspective's king crosses the d/e
  mirror. Rate: **0.0214 refreshes/node**, and one costs **6,240
  instructions** (`apply_at`'s comment, perf study 17 §3.3 + study 18).

A merged accumulator cannot reconstruct either channel separately, so an
own-king move would force a full threat rebuild:

```
  Δ instructions/node  ≈ (0.1611 − 0.0214) × 6240  ≈  +872 i/n   (+14%)
```

against a saving of one accumulator's parent-read + child-write, i.e.
5120 B + 5120 B per **evaluated** node. Even valuing that at one cycle per
64-byte line it is ~160 cycles per evaluated node, and the instruction cost
alone is ~620 cycles/node at the measured IPC of 1.45. Routing the merged
refresh through the Finny cache is *worse*, not better: the cache's set-diff is
between two arbitrary positions, and FullThreats has no set-difference form, so
it would need both positions enumerated — 2 × 6,240.

**Conclusion: the dual accumulator is a faithful reproduction of Stockfish's
own structure, and the merge is not an available win.** The 20 KB/node figure
is arithmetically right (2 stacks × 2 perspectives × 2560 B, read + written,
per evaluated node) and is not addressable by merging.

---

## 2. The instrument, and the null that changed it

The brief quoted a cycles/node noise floor of **0.6–1.1% (median of 6)**. The
first thing this study ran was the null: the same binary, copied to two names,
through the interleaved A/B panel at n=6, pinned to core 3 as study 20 did.

```
A base    n=6 cyc_n=4727.8   B base2   n=6 cyc_n=4567.3
  paired dcyc_n (B-A): median=-282.1 min=-1528.5 max=+181.2
  medians: dcyc_n=-160.5 (-3.40%)  dins_n=-0.00
```

**−3.40% cycles between a binary and its own copy.** Cause, found by reading
`/proc/stat`: core 3 was at **100% busy** — a sibling study had its own bench
pinned there, and my harness process was getting 2.2% CPU. Instructions were
still exact (0.000%), which is why an instruction-gated study would never have
noticed.

Moving to core 2 (9.5% busy) fixed it. The null on core 2, n=8:

```
A base    n=8 cycmin=4227.7 cycmed=4445.2 ins=6085.0 l1m=624.46 l2m=74.52 dram=8.39 dtlb=2.19
B base2   n=8 cycmin=4140.3 cycmed=4441.9 ins=6084.6 l1m=623.08 l2m=74.08 dram=8.21 dtlb=2.18
  D cycmin=-87.5 (-2.07%)  cycmed=-3.2 (-0.07%)  ins=-0.45 (-0.007%)
  D l1m=-0.22%  l2m=-0.59%  dram=-2.17%  dtlb=-0.38%
```

Two things this fixes for the rest of the study:

1. **The gate is `cycmed`, n ≥ 8, on core 2.** Floor **0.07%**.
2. **`cycmin` is NOT the robust estimator here.** The intuition that
   interference can only add cycles, so min-of-N is safest, is wrong on this
   box: min-of-8 read −2.07% on two identical binaries, thirty times the
   median's error. One lucky round dominates a min; a median needs half the
   rounds to move. Reported alongside, never gated on.
3. Secondary counters and their nulls: `l1m` 0.22%, `l2m` 0.59%, `dram` 2.17%,
   `dtlb` 0.38%, `instructions` 0.007%.

Harness: `docs/research/perf-artifacts-22/scripts/{build22,probe22,panel22,cores22}.sh`.
Raw output: `docs/research/perf-artifacts-22/{probes-fast,panel}.txt`.

---

## 3. Landed arms

*(filled in below as each is gated)*

---

## 4. Rejected without building, and why

### 4.1 Software prefetch of the fold's weight rows — REJECTED, pre-refuted

The obvious stall arm on a fold that streams ~11 KB/node of random rows out of
a 57.7 MB `i16` table. It is already refuted twice on this box, by adjacent
measurements on *larger* tables:

* `sota_threats.rs:1101` — "the study's software row-prefetch was implemented
  here and MEASURED A LOSS (−2.5% NPS full-row, ~−0.4% first-two-lines,
  bench-with-net A/B on the Zen 4 box) — the OoO window and hardware stride
  prefetchers already cover the fold's row streams".
* `sota_simd.rs:1380` — "prefetch the next group's rows … measured
  noise-to-negative (−0.3% over 4 interleaved pairs)".

The HalfKA fold's rows are 2560 B of *contiguous* `i16` walked at a 64-byte
stride — the single most prefetcher-friendly shape there is, and strictly
easier for the hardware than the threat fold's, which already lost. Building a
third instance of an arm that lost twice is not a good use of a 12-minute
panel.

### 4.2 The 512 dead memset bytes in the Finny set-diff — REJECTED, priced below the floor

Confirmed present in the shipped binary. `finny_refresh_into`'s prologue
contains eight `vmovdqa64 ZMMWORD PTR [rsp+…],zmm0` — 512 bytes of zeroing for
`adds: [usize; 32]` and `subs: [usize; 32]`, of which only the first
`add_len`/`sub_len` entries are ever read. This is exactly the class study 20
arm 4 closed in `MoveList`.

It is rejected on study 20's own measurement of the identical fix: at **0.5
calls per node** it was worth −6.53 i/n and **+0.25% cycles, i.e. cycle-neutral
inside the null floor** ("the store buffer was absorbing them"). This site runs
at ~0.16 calls/node — roughly a third of the rate — so the predicted effect is
~−2 i/n and no cycles at all. Against that it would cost the first
`#[allow(unsafe_code)]` in a file that currently has none, which the crate's
`#![deny(unsafe_code)]` policy admits "only with a measured, documented win".
There is no measured win to document.

### 4.3 "Refresh on a king *bucket* change, not on any king move" — REJECTED, unreachable

`halfka_king_key`'s own doc comment advertises it:

> Cache key for a HalfKAv2_hm king state: 32 bucket ids × two horizontal
> orientations. **Moving the king without changing this key leaves every
> non-king feature's index stable, so a dirty-piece update is sufficient.**

The condition can never hold. `key = KING_BUCKETS[ksq ^ flip] * 2 +
(file(ksq) ≤ 3)`, `KING_BUCKETS` folds file f with file 7−f, and the
orientation bit separates exactly those two halves again — so the composition
is a **bijection from the 64 squares onto the 64 keys**. Two distinct king
squares never share a key, and the promised fast path is dead code that was
never written. The comment should be corrected to say the key is a
relabelling of the king square (which is also why `FINNY_KEYS == 64`).

---
