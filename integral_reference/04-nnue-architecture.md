# Integral study — NNUE architecture & input features

Source studied: `integral_reference/integral` (aronpetko/integral), checked out
at `3f15ba6b` (2026-04-02), plus its full git history (`git log --all`, 669
commits on the mainline graph reachable from this HEAD, 947 branches total
including experiment branches never merged). Primary files read in full:
`shared/nnue/definitions.h`, `src/engine/evaluation/nnue/{nnue.h,nnue.cc,
accumulator.h,sparse.h}`, `shared/multi_array.h`, `shared/simd.h` (constants
only), `preprocess/net_processing.cc`, `README.md`. Historical states read via
`git show <rev>:<path>` for `abbd1390`, `62618129`, `581cebc1`, `126b4d57`,
`9fe45daa`, `cd3f7acb`, `b6c951cc` (see §8 for what each is).

## ⚠ Provenance & licensing warning (read before touching this material)

- **Repo:** `aronpetko/integral`, no `LICENSE` file present anywhere in the
  tree → **all rights reserved by default**, stricter than Stockfish's GPLv3
  or Reckless's AGPLv3. Per the task brief (stricter than our normal D27
  rule): **nothing below is copied code.** Every array, formula, and pseudocode
  block is my own restatement from reading the source; only bare numeric
  constants (facts, not expression) and file:line / git-rev citations are
  reproduced, exactly as the Reckless notes already do for an AGPLv3 source,
  just with a harder line — **do not paste, transliterate, or lightly rename
  Integral source into our tree,** and never ship or embed an Integral-trained
  net file.
- **Strength context:** top-6 engine (per the task brief), solo-developer
  (`aronpetko`), C++. Took a game off Stockfish in TCEC. This note's sibling
  docs (09/10/11 in this same `integral_reference/` set) own the "how did one
  person get here" question; this note stays on architecture, citing process
  facts only where they directly explain an architecture number (§8).

## 0. One-paragraph summary

Integral ships a **large, deep, heavily-bucketed** perspective network —
`(768×12 factorized → 1536)×2 → (16 → 32 → 1)×8` per the README — that looks,
at the constant level, almost identical to Reckless's net (same `FT_QUANT=255`,
same `FT_SHIFT=9`, same int8-sparse-FT-then-float-tail hybrid, same
king-bucket-map shape, same output-bucket-by-material-count idea) but is
roughly **2× wider** (L1 1536 vs Reckless's 768) and **20% more finely
king-bucketed** (12 vs 10). The single most interesting *distinct* technical
idea is the **activation**: Integral (like Reckless, unlike Stockfish) gets its
nonlinearity by **splitting the accumulator vector in half and multiplying the
two halves together elementwise**, rather than literally squaring a clipped
value — two independent top engines converging on the same "bilinear form
instead of a square" trick, which our own design/11 v1 plan does *not* yet use
(it plans a literal `clamp(x,0,QA)²`). Git-history archaeology (§8) shows this
shape was reached incrementally over ~8 months (Aug 2024 → Apr 2025) through a
sequence of individually-landed, individually-bench-tagged commits — the same
SPRT-gated-stepwise-scaling discipline our own design/11 "upgrade path" already
prescribes, now with a second worked example of *exactly which steps, in what
order, at what constants* a strong engine actually took.

## 1. Architecture at a glance

Constants (`shared/nnue/definitions.h:13-22`):

| name | value | role |
|---|---|---|
| `kL1Size` | 1536 | feature-transformer / accumulator width, per perspective |
| `kL2Size` | 16 | first affine layer's output width |
| `kL3Size` | 32 | second affine layer's output width |
| `kInputBucketCount` | 12 | king-relative input buckets (one table per perspective) |
| `kOutputBucketCount` | 8 | material-count-selected output "expert" (own full L1/L2/L3 weight set each) |
| `kFtQuantization` (`QA`) | 255 | feature-transformer int8/int16 quant scale |
| `kL1Quantization` (`QB`) | 128 | first-affine-layer int8 quant scale |
| `kEvalScale` | 200 | final float→centipawn multiplier |

Shape, restated as a pipeline (own pseudocode, not source):

```
input: 12 buckets × 2(own/enemy) × 6 piece-types × 64 squares  →  accumulator[1536]  (×2 perspectives)
accumulator  --split-halves pairwise multiply (§3)-->  feature_output[1536]  (int8, ~85%+ zero)
feature_output  --sparse int8 dot, NNZ-skipped (§5)-->  l1_sums[16]  (int32 → dequant to f32)
l1_sums  --dense f32 affine + clamp[0,1] (§5)-->  l2_output[32]
l2_output  --dense f32 affine + clamp[0,1] (§5)-->  l3_output  (scalar f32)
l3_output * kEvalScale  →  centipawn score
```

`RawNetwork`/`Network` (`shared/nnue/definitions.h:27-50`) hold this as one
struct: `feature_weights[12][2][6][64][1536]` (i16), `feature_biases[1536]`
(i16), `l1_weights[8][16][1536]` (i8, transposed to `[8][1536][16]` by the
offline preprocessor — §2.3), `l1_biases[8][16]` (f32), `l2_weights[8][16][32]`
(f32), `l2_biases[8][32]` (f32), `l3_weights[8][32]` (f32), `l3_biases[8]`
(f32). The feature-transformer table alone is `12·2·6·64·1536 ≈ 14.2 M` `i16`
weights (~28 MB) — the overwhelming majority of the net's parameters, same as
every NNUE design: the input layer is enormous and sparse, everything past it
is small and dense.

## 2. Input features: factorized king buckets + horizontal mirroring

Feature indexing (my own restatement of `accumulator.h:44-62`, the
`GetFeatureTable` function):

```
feature_index(square, king_square, piece, piece_color, perspective):
    if king_square.file >= E:                     # kingside half
        square = square XOR 0b111                  # flip file of every piece square
    rel_king = king_square XOR (56 * perspective)  # flip rank if perspective is Black
    bucket   = KING_BUCKET_MAP[rel_king]           # 64-entry table, 12 distinct values (below)
    sq_idx   = square XOR (56 * perspective)
    color_idx= (perspective != piece_color) as int # 0 = own piece, 1 = enemy piece
    return feature_weights[bucket][color_idx][piece][sq_idx]   # a length-1536 row
```

`KING_BUCKET_MAP` (current shape, `accumulator.h:16-25`, reproduced as a
formula description — rank-major, a1=0):

- rank 1 (own back rank): 4 distinct buckets by king file, mirrored
  `a/h→0, b/g→1, c/f→2, d/e→3`
- rank 2: 4 more buckets `4..7`, same file-mirrored split
- rank 3: collapses to 2 buckets `8,8,9,9,9,9,8,8` (center vs edge files)
- ranks 4-5: one bucket, `10`
- ranks 6-8: one bucket, `11`

So granularity is finest exactly on the two ranks a castled king actually
occupies, and coarsens sharply the further the king strays from home — the
same design intuition Reckless's asymmetric 10-bucket table encodes
(`reckless_reference/12-nnue-accumulator-psq.md` §4), independently arrived at
with different exact boundaries (Integral: 4+4+2+1+1 = 12; Reckless: 4+4+1+1 =
10). Horizontal mirroring folds the kingside half onto the queenside table by
flipping every *piece* square's file (`square XOR 0b111`) whenever the king is
on a kingside file — the king-bucket table is itself already file-symmetric
(each mirrored file pair maps to the same bucket number), so one XOR handles
both "which bucket" and "which weight row within it" consistently. This is the
same combined-XOR trick Reckless uses (`pst_index`, cited in
`12-nnue-accumulator-psq.md:265-272`) and the same technique our own design/11
§3.4.1 names as the upgrade-path destination ("horizontally-mirrored
king-bucketed inputs (HalfKAv2_hm-style)").

**"Factorized"**: the README's `(768×12 (Factorized) -> 1536)` refers to a
*training-time* technique, not something visible in the shipped `Network`
struct. The standard factorizer trick (own words): train each of the 12
per-bucket weight rows *alongside* one shared, bucket-independent "virtual"
weight row that every bucket's gradient also flows into; this gives
rarely-visited buckets (e.g. king-on-h1-with-queenside-attack positions) a
useful gradient signal from all the *other* buckets' training data too, fixing
the "some buckets see 100× less data than others" problem. Before quantizing
and shipping, the shared row is folded (added) into each bucket's own row and
discarded — so the 12-bucket `feature_weights` table is already
"pre-factorized" by the time it reaches this repo. Evidence for this reading:
`RawNetwork` and `Network` (`shared/nnue/definitions.h:27-50`) both declare
`feature_weights` with *only* the 12-bucket dimension, no extra "virtual
factor" axis, and the offline `preprocess/net_processing.cc:6-51` step (which
transposes/permutes weight layouts for SIMD — §2.3 below) never touches a
factorizer dimension either — confirming the merge happens upstream, inside
the (unvendored, external) bullet training config, before `RawNetwork` is ever
written to disk. This matches the classic Stockfish-popularized factorizer
technique from NNUE's early days; our own design/11 doesn't currently plan a
factorizer (v1 has no king buckets, so there's nothing to factorize yet — the
technique only earns its keep once bucket count > 1, see §10 item 3).

### 2.3 What the offline preprocessor does (`preprocess/net_processing.cc`)

A standalone binary, `preprocess <in.nnue> <out.nnue>`, run once per trained
net before it's embedded (own restatement of `net_processing.cc:6-81`):
transposes `l1_weights` from bullet's natural `[bucket][l2][l1]` layout to the
inference-friendly `[bucket][l1][l2]` (and `l2_weights` similarly
`[bucket][l3][l2] → [bucket][l2][l3]`), lane-permutes the feature-transformer
weights/biases for the `packus` instruction's cross-lane quirk
(`simd::kPackusOrder`), and builds a second `int8`-dot-product-friendly
permutation (`l1_weights_alt`) for the DPBUSD kernel. None of this changes any
number — it is a pure memory-layout transform so the hot loop (§5) can stream
weights in the exact order its SIMD kernel wants, with zero runtime
transposition cost. Good evidence for a general lesson: **treat "quantize +
train" and "lay out for SIMD" as two separate offline stages** rather than one
combined export step — worth doing the same in our own `netpack` helper
(design/11 §3.4.7 step 3) once a fast SIMD kernel wants a non-obvious weight
order.

## 3. The activation: split-halves pairwise multiply, not literal SCReLU-square

This is the single most load-bearing algorithmic idea in the file. Own
restatement of the scalar reference path (`nnue.cc:239-250`; the SIMD path,
`nnue.cc:60-96`, computes the identical arithmetic via a `mulhi`-based
shift-multiply trick for speed, not a different formula):

```
kFtShift = 9
for them in [0, 1]:                                  # own perspective, then flipped
    acc = accumulator[side_to_move XOR them]          # length-1536 i16 vector
    for i in 0 .. 768:
        a = clamp(acc[i],       0, QA)                # QA = 255
        b = clamp(acc[i + 768], 0, QA)
        feature_output[i + them*768] = (a * b) >> kFtShift        # → u8
```

So the 1536-wide accumulator is treated as **two disjoint 768-lane halves**,
each half clipped independently, and the halves are multiplied together
lane-by-lane rather than one value being squared against itself. Because the
two halves are driven by two *independently-trained* weight sub-matrices
(they're just two different slices of the same `feature_weights` row), this is
a genuine bilinear form — strictly more expressive than `clamp(x,0,QA)²` at
**identical compute cost** (one clip-pair, one multiply, one shift, same as
squaring), since training gets to shape what each half represents rather than
being forced to use the same value twice.

**This is not literal SCReLU** (`clamp(x,0,QA)²`, what Stockfish's public
architecture uses at every stage, per Reckless's cross-engine table:
`reckless_reference/13-nnue-inference-simd.md:436-452`, computed *and*
concatenated with plain `ClippedReLU` at each layer, doubling effective
width). It is, however, **exactly the same trick Reckless's `activate_ft`
uses** (`reckless_reference/12-nnue-accumulator-psq.md` §4,
`13-nnue-inference-simd.md` §2.1: `lhs = clamp(pst+threat, 0, QA)`,
`rhs = clamp(pst+threat @ i+384, 0, QA)`, `out = (lhs*rhs) >> FT_SHIFT`),
right down to `FT_SHIFT = 9` being the identical constant in both engines. The
one structural difference: Reckless first **sums two separate accumulators**
(piece-square + threats, §9) element-wise before the halves-split-multiply;
Integral has only one accumulator (no threat features, §9), so there's no
summing step — the halves-multiply applies directly to the single
piece-square accumulator's own two halves.

**A second dequantization use of the same shift constant** (`nnue.cc:171-174`,
`256-259`): once `feature_output` is dotted against the int8 `l1_weights` (§5)
to produce an `i32` sum, that sum is converted to `f32` by multiplying by
`kL1Normalization = (1 << kFtShift) / (QA² · QB)` — i.e. `kFtShift` (9) does
double duty as both the FT activation's post-multiply shift *and* the exponent
in the later int32→float dequantization constant. This is the same formula
shape as Reckless's `DEQUANT_MULTIPLIER = 2^FT_SHIFT / (FT_QUANT² · L1_QUANT)`
(`reckless_reference/13-nnue-inference-simd.md:222`) — Integral's version
differs only in `L1_QUANT` (128 here vs Reckless's 64), otherwise the same
"bullet-standard" constant family both engines independently reach.

### 3.1 An abandoned/parallel experiment worth flagging: literal square-and-concat

One branch in the git graph, `b6c951cc` "Dual activation" (2025-06-09,
`Bench: 2185479`) — confirmed **not an ancestor of the checked-out HEAD**
(`git merge-base --is-ancestor b6c951cc HEAD` fails) — tries exactly the
Stockfish-style move: alongside the existing `clamp(x,0,1)` CReLU output, also
compute `clamp(x·x, 0, 1)` (literal square) and **concatenate** it, doubling
`kL2Size` for that experiment from 16 to 32 columns feeding L2. This shows
Integral's author *did* try the "square-and-concat" idea SF uses, as a
parallel branch to the mainline halves-multiply design studied above — but
since it isn't merged into the tree the README/current source describes, its
outcome (adopted, regressed, or simply superseded) can't be determined from
this clone alone. Flagged as **uncertain provenance** — do not treat "dual
activation" as Integral's shipped architecture; treat the halves-multiply
scheme in §3's main body as what's actually running.

## 4. Output buckets by material count

`Accumulator::GetOutputBucket` (`accumulator.h:395-398`), own restatement:

```
kBucketDivisor = ceil(32 / kOutputBucketCount) = 4
bucket(state) = min( (popcount(all_occupied) - 2) / kBucketDivisor,  kOutputBucketCount - 1 )
```

Uniform-width buckets of 4 piece-count each (2-5 pieces → bucket 0, ...,
29-32 → bucket 7), each owning a fully independent `l1/l2/l3` weight set — 8
small MLPs sharing one feature transformer, selected once per `Evaluate` call
(`nnue.cc:43`). **This is the exact formula our own design/11 §3.4.1 already
pins** for the v1 upgrade path (`idx = (popcount(occ) - 2) / 4`, 8 buckets) —
independent confirmation, not a new idea to adopt, that our planned formula is
what a top engine ships verbatim. Reckless instead hand-shapes a 33-entry
lookup table with non-uniform boundaries, finer in the piece-rich
opening/middlegame range (`reckless_reference/13-nnue-inference-simd.md:80-94`)
— a refinement Integral did *not* take (its bucketing stays a clean uniform
division), worth remembering as a "cheap thing to try later, not obviously
necessary" data point: two strong engines differ on this exact axis, so it's
evidently not make-or-break.

## 5. Sparse int8 FT/L1, float L2/L3 tail

Forward pass (own restatement, `nnue.cc:60-236`, both the SIMD and scalar
branches compute this):

```
feature_output[1536]  (u8, from §3, typically 85%+ zero after clipping)
nnz_indices           = positions of nonzero 4-byte groups in feature_output   # via a precomputed
                                                                                # 256-entry popcount/index
                                                                                # lookup table (sparse.h)
l1_sums[16] (i32)     = Σ over nnz groups only: dpbusd(feature_output[group], l1_weights[bucket][group])
l1_output[16] (f32)   = clamp( l1_sums * kL1Normalization + l1_biases[bucket], 0, 1 )
l2_sums[32] (f32)     = l2_biases[bucket] + Σ_i l1_output[i] · l2_weights[bucket][i]
l2_output[32] (f32)   = clamp(l2_sums, 0, 1)
l3_output (f32)       = l3_biases[bucket] + Σ_i l2_output[i] · l3_weights[bucket][i]
score = l3_output * kEvalScale
```

The **NNZ (number-of-nonzero) table** (`sparse.h:14-32`) is a precomputed
`array<NnzEntry, 256>`, one entry per possible 8-bit nonzero-lane bitmask,
each listing which bit positions were set — the classic "movemask + LUT
expand" trick, identical in spirit to Reckless's `SparseEntry` table
(`reckless_reference/13-nnue-inference-simd.md:175-185`) and to Stockfish's
`NNZInfo`/`nnz_helper.h` path both those notes already cross-reference. Only
the nonzero 4-byte groups of `feature_output` are ever dotted against
`l1_weights` (`nnue.cc:98-168`, `DpbusdEpi32`/`DpbusdEpi32x2` — the `x2`
variant processes two nonzero groups per iteration purely for
instruction-level parallelism, mirroring Reckless's `double_dpbusd`). Past
that first int8 sparse layer, **everything is plain `f32`** with ordinary
clamp (not squared) activation — no further quantization tax, because the
16-and-32-wide layers are cheap enough that integer quantization would save
work that doesn't matter. This int8-sparse-FT/L1-then-float-tail shape is
*exactly* the hybrid Reckless's note already identified as the target
architecture family (as opposed to Stockfish's fully-integer, skip-connected,
dual-activation-concatenated cascade) — Integral is a second independent top
engine confirming that specific "hybrid, not fully-integer" choice.

**Sparse neuron permutation** (`sparse.h:35-106`, landed `6223b7fb`
2025-04-07, gated behind a `SPARSE_PERMUTE` build flag): an offline tool that
counts, over a bench run, how often each of the 768 activation-producing
neuron *pairs* fires nonzero, then permutes both the feature-transformer
output ordering and the `l1_weights` columns so frequently-active neurons are
grouped together. This doesn't change the math (verified by re-running bench
and comparing), only the *memory order* — the point is that grouping
often-nonzero lanes together makes the NNZ-driven sparse loop's memory access
pattern more predictable/cache-friendly. A real but second-order optimization,
worth knowing about only once a sparse L1 already exists in our own engine
(post-v1, design/11 upgrade path).

## 6. Accumulator maintenance: lazy ply-stack + Finny tables + fused updates

Own restatement, `accumulator.h:202-419`. Structurally this is the *same*
"ply-indexed flat stack, walk back to nearest computed ancestor, apply deltas
forward" design our own design/11 §3.4.4 already plans and that Reckless
independently confirmed (`reckless_reference/12-nnue-accumulator-psq.md` §2) —
Integral is a **third** independent instance of the same shape, so I'll note
only what's distinctive rather than re-deriving the whole mechanism:

- **Cache keyed by (mirror-half, king-bucket), 2×12 = 24 physical slots**
  (`input_bucket_cache_: MultiArray<BucketCacheEntry, 2, 12>`,
  `accumulator.h:418`), each slot's `BucketCacheEntry` internally holding
  **both perspectives'** cached accumulator + piece/color bitboards
  (`piece_bbs`/`side_bbs` indexed `[perspective][piece]`,
  `accumulator.h:187,243-246`) — so the "24 slots" figure understates the
  real cached-state count (48, since each slot is really two independent
  per-perspective sub-caches). This is a Finny table exactly like Reckless's
  (`reckless_reference/12-nnue-accumulator-psq.md` §3) but organized as
  "(mirror, bucket) outer, perspective nested" rather than Reckless's flat
  "(pov, mirror, bucket) = 40 peers" — different data layout, same idea and
  same payoff (diff against a recent same-bucket snapshot instead of
  recomputing from bias).
- **Invalidation trigger** (`NeedRefresh`, `accumulator.h:371-382`): a full
  refresh (not incremental patch) is forced exactly when the king crosses the
  kingside/queenside mirror boundary *or* changes king-bucket — the same
  precise trigger condition Reckless uses
  (`reckless_reference/12-nnue-accumulator-psq.md:150-156`).
- **Variadic fused-operation update** (`ApplyChange<ops...>`,
  `accumulator.h:98-134`): a template that takes a compile-time list of
  `kAdd`/`kSub` operations and a matching list of `FeatureData` structs, then
  folds all of them into one pass over the 1536-wide vector
  (`Fused<ops...>(previous[i], changes[i]...)`). This is the C++-generic-code
  equivalent of what our design/11 §3.4.4 calls "fused add+sub kernels" for
  the dominant 1-add/1-sub case — Integral generalizes it to *any* fixed
  combination (2-add/2-sub for castling, 1-add/2-sub for captures) via one
  template rather than hand-writing each case, landed as two dedicated
  commits (`82e8bad7` "Template fused updates", `45e0e036` "Finny table fused
  updates", both 2025). In idiomatic Rust the same generalization is more
  natural still: a small `const`-generic array of `(AddOrSub, FeatureData)`
  pairs (or just distinct fixed-size arrays for adds/subs, since Rust doesn't
  need C++'s variadic-template dance to express "loop over 1-4 known-size
  deltas in one pass") — this is confirmation of the *shape* to build, not a
  C++ idiom to transplant.
- **Lazy update walk** (`ApplyChanges`, `accumulator.h:329-369`): walks
  backward from the current head to the nearest `updated[perspective]`
  ancestor, then forward re-applying each ply's `AccumulatorChange` (or
  triggering a bucket-cache refresh if a king-bucket/mirror crossing is
  detected mid-walk) — same algorithm as Reckless's `can_update_pst`/
  `update_pst_accumulator` and our own doc 11 §3.4.4 plan, a third convergent
  data point for "ply-indexed lazy stack" being the right shape.

## 7. Quantization scheme, summary table

| constant | Integral | Reckless | our design/11 v1 plan | our design/11 upgrade-path plan |
|---|---|---|---|---|
| `QA` (FT quant) | 255 | 255 | 255 | 255 (unchanged) |
| `QB` / L1 quant | 128 | 64 | 64 (`QB`, output-layer scale) | not yet pinned — Integral/Reckless disagree (128 vs 64), a value to tune not copy |
| FT shift | 9 | 9 | n/a (v1 has no sparse int8 L1 stage) | 9 is the emerging "bullet-standard" default across two engines |
| activation | halves-multiply (§3) | halves-multiply (sums 2 accs first) | literal `clamp(x,0,QA)²` (SCReLU) | **candidate change** — see §10 item 1 |
| L1/L2/L3 depth | 1536 → 16 → 32 → 1 | 768 → 16 → 32 → 1 | 256 (single hidden layer) → 1 | named destination already: "deep 15/32-neuron tail with sparse propagation" |
| input buckets | 12, hand-tuned rank/file layout | 10, hand-tuned rank/file layout | none (v1) | HalfKAv2_hm-style, count TBD |
| output buckets | 8, uniform `(pc-2)/4` | 8, hand-shaped 33-entry table | none (v1) | 8, `(pc-2)/4` — **already matches Integral exactly** |
| threat features | **none** | yes, separate 66,864-row accumulator | none | flagged near-term item (design/17 §3.5), Reckless-sourced |

## 8. Architecture evolution: an 8-month scaling timeline

Every number in §1 was reached incrementally, one bench-tagged commit at a
time (`git log --date=short --format="%ad %h %s" -- src/engine/evaluation/nnue
shared/nnue`, full list below is the complete architecture-relevant subset;
dates are commit dates, all times in the author's local timezone as recorded
by git):

| date | commit | change |
|---|---|---|
| 2024-08-27 | `cd3f7acb` | first NNUE at all: `768 → 16 → 1`, single perspective pair, no buckets, `kEvalScale=400` |
| 2024-09-06 | `99b7b22e` | NNUE merged to main (`#52`) |
| 2024-09-13 | `9fe45daa` | hidden layer 16 → **1024**; **8 output buckets already present here** — output bucketing predates king bucketing and predates mirroring by 5 days |
| 2024-09-18 | `796c4dca` | horizontal mirroring added (`#69`) |
| 2024-09-21 | `abbd1390` | hidden layer → **1280**; **first king input buckets: 4** (coarse: rank1 center-vs-edge split, ranks2-8 uniform) |
| 2024-09-25 | `bb56cb85` | Finny-table refresh cache added (`#81`) |
| 2024-10-22 | `62618129` | hidden layer → **1536**; buckets 4 → **9** (rank1: 4, rank2-3 merged into 2, ranks4-8: 1) — first commit to use the word "factorized" in a commit message |
| 2025-01-01 | `581cebc1` | **"Multi-layer network" (`#153`)**: introduces the L2(16)/L3(32) tail for the first time (previously a single hidden→output affine); hidden layer *dropped back* to 1280 (cost of the new tail momentarily paid for by shrinking L1); buckets 9 → **12** (today's rank1:4/rank2:4/rank3:2/rank4-5:1/rank6-8:1 layout); quantization renamed/changed (`kHiddenLayerQuantization=255,kOutputQuantization=64` → `kFtQuantization=255, kL1Quantization=128`) |
| 2025-02-04 | `d2ce6fc7` | "Sparsity table speedup" |
| 2025-03-27 | `83347d29` | "Fix NNUE inference & speedups" — the halves-multiply activation and int8 DPBUSD sparse matmul (§3, §5) date to at least here |
| 2025-03-31 | `126b4d57` | hidden layer 1280 → **1536** again (recovering, now with the L2/L3 tail already in place) |
| 2025-04-07 | `6223b7fb` | sparse neuron permutation tool (§5) |
| 2025-04-24 | `576bef0b` | "Pre-process the network and use it directly from the binary" — the `preprocess/net_processing.cc` split (§2.3) |
| 2025-04-29 | `45e0e036`, `223b80a6` | Finny-table + generic fused-update templates, bench-speed pass |
| 2025-05-13 | `82e8bad7` | "Template fused updates" — the `ApplyChange<ops...>` generalization (§6) |
| 2025-06-09 | `b6c951cc` | "Dual activation" experiment (§3.1) — **not merged into the studied HEAD** |

Reading this top-to-bottom: **output buckets shipped before king buckets**,
**king buckets shipped before the L2/L3 tail**, and **the L2/L3 tail's
addition briefly *shrank* L1** (1536→1280) before growing it back to 1536
three months later once the new tail justified the cost — every step SPRT/bench-gated
individually (each commit message ends `Bench: <n>`, the same discipline our
own `Bench:` trailer convention formalizes, design/14 §3.2), never a single
big-bang net redesign. This is a second, more granular worked example of
exactly the sequencing our own design/11 §3.4.1 "upgrade path" paragraph
already prescribes in the abstract ("L1 256 → 512 → 1024; output buckets...;
horizontally-mirrored king-bucketed inputs... a separate PSQT output head") —
Integral shows the *order* one real engine actually used (output buckets →
mirroring → coarse king buckets → Finny cache → wider L1 → deeper tail →
finer king buckets → sparse tricks → layout preprocessing), which is useful
sequencing evidence, not just a destination shape.

**On the sheer volume of architecture experimentation**: `git branch -a` on
this clone lists **947 branches**, a meaningful fraction visibly
architecture-named — `7buckets`, `8bucket_factorized`, `9kb_net`
(`more_layers_9kb`), `12kb_net`, `13kb_net`, `1536l1`, `1280hl_500sb`,
`128hl`/`128hl_300`, `dual_activation`, `screlu2`, `output_buckets`,
`bucketed_psqts`/`double_bucketed_psqts{,2}`, `nnue_256hl`, `subnet`,
`simple_net`, `leela_net`, `vine_net`, `yoshie_net`, `quiet_hist_factorizer`
(the factorizer *idea* applied to a history table, not NNUE — interesting
cross-pollination of one architectural trick into an unrelated subsystem).
I did not check out or read each branch's diff (out of scope for this note —
that volume of individually-tested architecture variants is itself a §9-11
success-factor question, not a §04 architecture one) but the *existence* of
this many named, presumably-SPRT-tested net-shape experiments is strong
circumstantial evidence that every number in §1/§7 was arrived at by
measurement, not by reading a paper and copying its hyperparameters — worth
flagging up to whichever note in this set owns the "process" question.

## 9. Threat features: the one thing Integral does *not* have

Grepping the whole source tree for "threat" turns up only **move-ordering**
uses — `TUNABLE` history-bonus constants and a `BitBoard threats` field
consumed by `MovePicker` (`src/engine/search/move_picker.cc:27-32,221-245`,
`src/chess/board.h:274,334-336`) — **nothing in the NNUE input feature set**.
This is the one clear point of divergence from both Stockfish's dev tree
(`Features::FullThreats`) and Reckless's dedicated 66,864-row threats
accumulator (`reckless_reference/11-nnue-features-threats.md`,
`13-nnue-inference-simd.md` §1,§4). Read generously, this is useful calibration
for prioritization: **a top-6 engine (and one that has beaten Stockfish in a
TCEC game) reaches that strength on base-architecture depth/width/bucketing
alone**, without the threat-feature lever at all — suggesting the "scale up
L1/L2/L3, add king buckets, add output buckets, sparse-propagate" program is
sufficient to get very far, and threat features are a *later, additional*
lever rather than a prerequisite for competitiveness. This is consistent with
(not a reason to change) our own design/17 §3.5 sequencing, which already
lists threat/relational features as a "near-term, buildable-now" item for
P13's *first strong net* rather than the very first net — Integral is a
second data point that base-architecture-first is a defensible order, on top
of Reckless already being one.

## Adapt to us

Ordered roughly by how directly actionable each item is against our current
plan (design/11 §3.4, phase P13; design/17 for the research layer).

1. **Prefer the split-halves pairwise-multiply activation over literal
   SCReLU-square for any post-v1 net.** *Doc/phase*: design/11 §3.4.3
   (currently pins `clamp(x,0,QA)²`); revisit when the v1 net's SPRT baseline
   is in and a first upgrade is considered. *What*: split the accumulator into
   two equal halves at build time (i.e. train it as two independently-weighted
   768-or-wider sub-blocks per perspective instead of one block used twice),
   clip both halves to `[0,QA]`, multiply elementwise, shift down — same
   instruction count as squaring, strictly more expressive since the two
   halves carry independent weights. **Now confirmed by two independent top
   engines** (Reckless *and* Integral, identical `FT_SHIFT=9` in both) rather
   than one — the strongest single piece of new cross-checking evidence in
   this note. *Effort*: small once a net is already being retrained (an
   activation-function swap plus doubling the accumulator's effective training
   parameterization); zero engine-side compute cost. *Impact*: likely small-
   to-moderate Elo for free compute, per the "identical cost, more expressive"
   argument both engines' authors evidently found worth shipping.

2. **Treat "quantize the trained net" and "lay out weights for SIMD" as two
   separate offline stages.** *Doc/phase*: design/11 §3.4.7 step 3 (`netpack`
   helper). *What*: once our own inference kernel wants weights in a
   non-training-natural order (transposed matrices, lane-permuted rows for a
   `packus`/`pshufb`-style instruction), do that reordering in a dedicated
   second pass over the already-quantized net, the way Integral's
   `preprocess/net_processing.cc` does — keeps the "trained → quantized"
   step simple and the "quantized → SIMD-laid-out" step independently
   testable (diff the reordered output against the pre-reorder one by
   permuting back). *Effort*: small, and only relevant once our SIMD kernel
   design (design/11 §3.4.6) settles on a non-trivial memory order. *Impact*:
   organizational clarity more than Elo — avoids a monolithic "train script
   also does memory-layout arithmetic" script that's hard to unit-test.

3. **Confirms (no change): our planned `(popcount(occ)-2)/4`, 8-bucket output
   formula is exactly what Integral ships.** *Doc/phase*: design/11 §3.4.1.
   *What*: nothing to change; cite Integral (`accumulator.h:12-13,395-398`)
   alongside Reckless as a second independent confirmation when this design
   choice comes up in review — Reckless's non-uniform 33-entry table is a
   *documented alternative* two strong engines disagree on, so ours (uniform,
   simpler) is a reasonable default, not an under-baked shortcut.
   *Effort*: none. *Impact*: confidence, not code.

4. **When king buckets land, budget for a genuine bucket-count *growth
   path*, not a single number chosen up front.** *Doc/phase*: design/11
   §3.4.1 upgrade-path bullet ("horizontally-mirrored king-bucketed inputs").
   *What*: Integral's own history (§8) went 4 → 9 → 12 buckets over ~5 weeks,
   each transition SPRT/bench-gated and each transition adding granularity
   specifically near the back rank first — a concrete, low-risk way to
   introduce king buckets ourselves: ship a coarse 4-ish-bucket layout first
   (small retraining cost, small correctness surface), then re-split the
   fine-grained ranks once the coarse version's SPRT is in hand, rather than
   hand-designing a final N-bucket table before any data exists to validate
   it. *Effort*: none extra vs. the existing plan — this is a sequencing note,
   not new work. *Impact*: lower risk per step; each step independently
   SPRT-attributable per our own D25 discipline.

5. **NNZ sparse-propagation table**, once a second dense layer exists past the
   feature transformer. *Doc/phase*: design/11 §3.4 upgrade path (the L1/L2/L3
   deepening step). *What*: precompute once (build time or net-load time) a
   256-entry table mapping every possible 8-bit nonzero-lane bitmask to the
   list of set-bit positions (`sparse.h:14-32`); use it to skip zero 4-byte
   groups of the post-activation feature vector when dotting against the next
   layer's int8 weights. Already independently adopted by Reckless
   (`reckless_reference/13-nnue-inference-simd.md` §2.2) — Integral is a third
   confirming instance, so this is now "convergent across every modern NNUE
   engine we've studied," not a novel idea. *Effort*: small once the
   dense-layer-past-FT step exists; zero correctness risk (pure work-skipping
   over an identical result, testable by comparing dense vs. sparse paths
   bit-for-bit). *Impact*: throughput at that future point, no Elo change by
   itself.

6. **Generalize the accumulator's fused add/sub kernel to an arbitrary
   compile-time-known op list, Rust-idiomatically.** *Doc/phase*: design/11
   §3.4.4 (currently names only the dominant 1-add/1-sub fused kernel).
   *What*: Integral's C++ `ApplyChange<ops...>` variadic template
   (`accumulator.h:98-134`) is solving a problem Rust doesn't need templates
   for: a small `const N: usize` array of `(is_add: bool, feature_row: &[i16;
   L1])` pairs (or, simpler, our own already-planned fixed `adds`/`subs`
   slices in `DirtyPiece`) folded in one pass over the accumulator vector.
   Nothing to import from the C++ shape itself, but it's confirmation that
   "one generic pass handling 1, 2, or 4 deltas uniformly" (rather than
   separate hand-written 1-add/1-sub, 2-add/2-sub, 1-add/2-sub kernels) is
   the right level of generality to aim for. *Effort*: trivial — this is
   close to what `DirtyPiece{adds:[..;2], subs:[..;2], n_add, n_sub}` already
   supports; just loop `0..n_add`/`0..n_sub` in one pass rather than writing
   per-move-kind specialized kernels. *Impact*: code-simplicity, not Elo.

7. **Threat/relational input features remain a *later*, not a *blocking*,
   lever.** *Doc/phase*: design/17 §3.5 (already lists this as the near-term
   supporting thrust, sourced from the Reckless study). *What*: Integral's
   absence of any threat-feature input, while still reaching top-6 and beating
   Stockfish in at least one TCEC game, is a second independent data point
   (after Reckless's presence of one) that a strong base architecture (wide
   L1, king+output buckets, deep sparse tail) carries most of the way there
   *without* threat features — supports keeping our own sequencing (base
   value/WDL net first, threat features as a named-but-deferred stretch, not
   a P13-v1 requirement). *Effort*: none (a sequencing confirmation).
   *Impact*: risk reduction — avoids over-indexing on one engine's (Reckless's)
   single example as if threat features were mandatory for top-tier strength.

8. **When retraining a deeper net, expect the "add a new stage temporarily
   shrinks an existing dimension" pattern, and don't be alarmed by it.**
   *Doc/phase*: design/11 §3.4.1 upgrade path, general expectation-setting for
   whoever executes it. *What*: Integral's own "Multi-layer network" commit
   (`581cebc1`, §8) *dropped* L1 from 1536 to 1280 in the same commit that
   added the L2/L3 tail, only regrowing L1 back to 1536 three months later once
   the new tail's cost/benefit was separately validated — i.e. "add depth" and
   "add width" were deliberately decoupled into separate SPRT-gated steps
   rather than done simultaneously (which would confound which change caused
   which Elo delta). *Effort*: none — a testing-discipline note. *Impact*:
   keeps our own SPRT attribution clean when we eventually add an L2/L3 tail
   to design/11's `768→256×2→1` v1 net; change one axis (depth *or* width) per
   gated step, per our existing D25 discipline, not both at once.
