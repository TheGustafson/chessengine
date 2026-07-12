# Integral — NNUE inference, accumulator, SIMD

Source studied: `integral_reference/integral` (aronpetko/integral, no explicit
license → all-rights-reserved; **ideas and formulas only, no code
reproduced**, stricter than our GPL/AGPL handling of Stockfish/Reckless).
Files read in full: `src/engine/evaluation/nnue/{nnue.h,nnue.cc,accumulator.h,
sparse.h}`, `shared/nnue/definitions.h`, `shared/simd.h`, `src/utils/fused.h`,
`src/chess/board.cc` (the `MakeMove`/`UndoMove`/`MakeNullMove` accumulator
wiring), `preprocess/net_processing.cc` + `preprocess/CMakeLists.txt`, the
top-level `CMakeLists.txt` and `Makefile`, plus `git log`/`git show` on the
vendored history for the timeline in §9. All line numbers are into the
vendored tree under `integral_reference/integral/`; every number/constant
below is a bare fact (dimension, offset, formula), not expression, and is
cited so a future reader can re-find and re-verify it rather than trust this
prose.

## 0. One-paragraph summary

Integral's NNUE is a wider, shallower-cascade cousin of Reckless's: a single
1536-wide dual-perspective feature-transformer accumulator (`arch::kL1Size`,
`shared/nnue/definitions.h:13`) — exactly 2× Reckless's 768 — feeding an
int8-quantized, NNZ-sparse `1536→16` affine layer via VNNI-style `dpbusd`,
then two small dense `f32` layers (`16→32→1`) with plain ClippedReLU, gated
by 8 output buckets (by piece count) and 12 king-relative input buckets (by
king square, mirrored). The accumulator stack is a `ply`-indexed,
lazily-updated array with a Finny-table refresh cache — structurally the
same shape our own design/11 §3.4.4 already plans and Reckless already
implements, a third independent confirmation of that design. What's
genuinely distinctive here vs. both prior studies: (1) a **separate,
per-SIMD-tier build-time preprocessing binary** that transposes and
byte-permutes the trained network into the exact memory layout the target
ISA's kernels want, baked once into the shipped binary rather than fixed up
at load time; (2) a **5-tier compile-time build matrix**
(`vnni512/avx512/avx2_bmi2/avx2/sse41_popcnt`, `CMakeLists.txt:86-91`) with
zero runtime CPU dispatch, matching Stockfish's official-binary convention
more than Reckless's single-portable-binary-with-runtime-detection default;
(3) a **profile-guided neuron-permutation research mode**
(`SPARSE_PERMUTE`) that reorders accumulator neurons by measured activation
frequency to improve sparse-matvec locality — not present in either prior
study; and (4) a git history showing the entire lazy-update → Finny-table →
SIMD pipeline landing inside about **one month** of first shipping NNUE at
all (§9) — a concrete data point for the "how did a solo dev get here"
question that the success-factor notes (09-11) will go deeper on.

## 1. Architecture at a glance

Constants (`shared/nnue/definitions.h:13-22`):

| name | value | role |
|---|---|---|
| `kL1Size` | 1536 | feature-transformer / accumulator width (per perspective) |
| `kL2Size` | 16 | first affine layer's output width |
| `kL3Size` | 32 | second affine layer's output width |
| `kInputBucketCount` | 12 | king-relative input buckets (mirrored) |
| `kOutputBucketCount` | 8 | material-count (piece-count) output buckets |
| `kFtQuantization` (QA) | 255 | feature-transformer integer scale |
| `kL1Quantization` (QB) | 128 | first-affine-layer int8 weight scale (Reckless: 64 — see §7) |
| `kEvalScale` | 200 | final float→centipawn multiplier (Reckless: 380; our plan: 400 — each engine's own training-time calibration, not a comparable/portable constant) |

Real shape, in the same "name after the consuming layer" convention Reckless
uses (see the naming-trap note in `reckless_reference/13-nnue-inference-simd.md`
§1 — Integral has the identical trap: `kL1Size` is the *accumulator* width,
not layer 1's *output*):

```text
1536 (FT, dual accumulator, int8-activated via pairwise mulhi-SCReLU)
  -> sparse int8 affine (NNZ-gated, dpbusd)         -> 16  (f32, ClippedReLU)
  -> dense f32 affine                                -> 32  (f32, ClippedReLU)
  -> dense f32 affine (output)                       -> 1   (f32) * kEvalScale -> Score
```

`RawNetwork` vs `Network` (`shared/nnue/definitions.h:27-50`): two distinct
`struct` layouts for the *same* logical parameters — `RawNetwork` is the
as-trained, row-major format (`l1_weights: [bucket][L2][L1]` i8,
`l2_weights: [bucket][L3][L2]` f32, straightforward "output-major" matrices a
trainer would naturally emit); `Network` is the runtime, SIMD-ready format
(`l1_weights` transposed to `[bucket][L1][L2]`, *plus* a second
`l1_weights_alt: [bucket][L1*L2]` union view permuted for the
two-groups-at-once dot product kernel, §7; `l2_weights` transposed to
`[bucket][L2][L3]`; every array `alignas(simd::kAlignment)`). A trained net
is never used directly — it is transformed **once, at build time** by a
standalone tool (§8) into `Network`'s layout before being embedded in the
binary. Rough size: `feature_weights` alone is
`12 × 2 × 6 × 64 × 1536` `i16` ≈ 27 MiB (`kInputBucketCount ×
2(color) × 6(piece types) × 64(squares) × kL1Size`,
`shared/nnue/definitions.h:28,39`) — the net file is dominated by the
12-bucket feature table, not the tiny L1/L2/L3 tail.

## 2. Input features and king-relative bucket indexing

`GetFeatureTable` (`accumulator.h:44-62`) computes the feature row for a
`(square, king_square, piece, piece_color, perspective)` tuple:

```text
if king_square.file >= E: square = square XOR 0b111        // file-mirror all pieces
relative_king = king_square XOR (56 * perspective)          // rank-flip for the pov
king_bucket   = kKingBucketMap[relative_king]
square_idx    = square XOR (56 * perspective)
color_idx     = (perspective != piece_color) ? 1 : 0
row = feature_weights[king_bucket][color_idx][piece][square_idx]   // i16[1536]
```

This is the same "one XOR mask serves both bucket lookup and every piece
square" idea Reckless uses (`reckless_reference/12-nnue-accumulator-psq.md`
§4's `pst_index`), just written as two sequential XORs (file-mirror keyed
off the *absolute* king file, then a separate perspective rank-flip) instead
of one combined mask — algebraically equivalent, a style difference only.

`kKingBucketMap` (`accumulator.h:16-25`, reproduced as a description, not the
array — 64 entries, 12 distinct values): rank 1 (the back rank) gets 4
distinct buckets by king file, mirrored (a/h→0, b/g→1, c/f→2, d/e→3); rank 2
gets 4 more (4-7) the same way; rank 3 splits into two buckets — edge files
(a,b,g,h)→8, middle files (c,d,e,f)→9; rank 4 collapses **entirely** into one
bucket (10); ranks 5-8 collapse **entirely** into one bucket (11). So
granularity is finest on the back two ranks (where a castled king actually
lives) and coarsest — a single bucket for four whole ranks — everywhere a
king rarely goes outside the endgame. This is a different concrete asymmetric
scheme than Reckless's 10-bucket layout (which keeps per-file granularity on
ranks 1-2 only, collapsing ranks 3-8 into just 2 big buckets rather than
Integral's 4: 8, 9, 10, 11) — both hand-designed, both "spend bucket capacity
where the king actually goes," different granularity budgets. Worth noting
as a second real data point for "how many king buckets, split how" when we
design our own (design/11 §3.4.1 upgrade path), not a value to copy verbatim.

## 3. The accumulator: per-perspective storage and fused updates

`PerspectiveAccumulator` (`accumulator.h:64-175`) wraps one
`alignas(simd::kAlignment) std::array<I16, 1536>`. `Reset()` sets it to
`feature_biases`; `Refresh(state, perspective)` resets then adds every piece
currently on the board (from-scratch, used only as a helper — the real
refresh path is the Finny-table one in §5, not this method directly).

The interesting piece is `ApplyChange`'s templated fused-op design
(`accumulator.h:98-163`, `src/utils/fused.h:1-35`): a variadic
`Fused<ops...>(in, operand...)` recursively folds a whole chain of `+`/`-`
operations into **one pass** over the 1536-wide array, so "add 1, sub 1"
(normal move), "add 1, sub 2" (capture/en-passant), and "add 2, sub 2"
(castling) each become a single loop with the right number of terms baked in
at compile time via a `switch` on an `enum FusedOperation { kAdd, kSub }`
parameter pack — no intermediate accumulator materialized between the add
and the sub. `AccumulatorChange` (`accumulator.h:34-42`) is a small
`{sub_0, add_0, sub_1, add_1, type}` struct where `type ∈ {kNormal, kCapture,
kCastle}` selects which fused-op combination `ApplyChange` dispatches to
(`accumulator.h:136-163`).

**This change record is built eagerly, with concrete squares/pieces, at
move-make time** — `Board::MakeMove` (`board.cc:204-281`) fills in
`accum_change.sub_0/add_0/(sub_1/add_1)` inline as it processes the move
(capture, en passant, castling, promotion each get their own branch,
`board.cc:214-262`), then calls `accumulator_->PushChanges(state_,
accum_change)` once at the end (`board.cc:280`). This is **not** Reckless's
approach (which stores only `(mv, piece, captured)` and re-derives the
actual add/sub squares later, lazily, inside `PstAccumulator::update`,
`reckless_reference/12-nnue-accumulator-psq.md` §2.2) — it is, instead,
exactly our own design/11 §3.4.4 plan (`DirtyPiece{adds,subs}` populated
while `make_move_with`'s locals are already in scope). A second independent
top-tier engine doing it our way is worth citing alongside the Reckless
citation next time that design choice needs justifying in review.

## 4. The accumulator stack and the lazy update walk

`Accumulator` (`accumulator.h:202-419`) owns `stack_: std::vector<
AccumulatorEntry>`, preallocated to 512 entries and grown via
`stack_.emplace_back()` on overflow (`IncrementHead`, `accumulator.h:384-389`)
— a **dynamically growable** vector, not a fixed `MAX_PLY`-sized boxed slice
the way Reckless (`Box<[PstAccumulator]>`, size 240) and our own design/11
plan (`Box<[AccumulatorEntry]>`, "MAX_PLY + margin") both use. `head_idx_`
is a plain cursor; `UndoMove()` is just `--head_idx_` (`accumulator.h:391-393`)
— zero recomputation on unmake, same "pop is free" property as Reckless.

`AccumulatorEntry` (`accumulator.h:177-183`) holds both perspectives'
`PerspectiveAccumulator`, the `AccumulatorChange` that produced this entry,
both sides' king squares, an `updated: [bool; 2]` flag (Reckless calls this
`accurate`; our design/11 plan calls it `computed`), and a full `BoardState`
copy (used only if a forced refresh is needed, `accumulator.h:353-354`).

`ApplyChanges()` (`accumulator.h:329-369`) is the lazy evaluate-time walk,
run independently per perspective: if the head entry is already `updated`,
done. Otherwise walk **backward** to the nearest ancestor entry with
`updated[perspective] == true` (guaranteed to terminate — the root is always
marked updated by `SetFromState`, `accumulator.h:208-216`), then walk
**forward** re-applying each ply's stored change — either the fused
`ApplyChange` (§3) if `NeedRefresh` says no, or a full Finny-table refresh
(§5) if it says yes — marking each entry `updated` as it goes.
`NeedRefresh` (`accumulator.h:371-382`) triggers on exactly two conditions:
the king crossed the kingside/queenside mirror boundary (file ≥ E flips),
or it moved to a different `kKingBucketMap` bucket. This is, point for
point, the same invalidation/walk-back-to-nearest-computed-ancestor design
as Reckless's `can_update_pst`/`update_pst_accumulator`
(`reckless_reference/12-nnue-accumulator-psq.md` §2.3) and our own design/11
§3.4.4 — a **third** independent implementation of the identical shape,
reinforcing that this is a converged-upon, not merely convenient, design.

**Null moves never touch the accumulator stack at all.**
`Board::MakeNullMove`/`UndoNullMove` (`board.cc:283-302`) call
`history_.Push`/`PopBack` for the board state but make **no** call into
`accumulator_` in either direction — correct, because a null move changes
only side-to-move, not piece placement, so the current head entry remains
valid for both perspectives with zero work. Compare our own design/11
§3.4.4, which describes "null moves directly in the `do_null_move` wrapper"
storing an *empty* dirty record — i.e. our current plan still pushes a
(trivial) stack entry on a null move where Integral shows that push isn't
needed at all. A small, free simplification worth folding into the plan
(§10 item 5 below).

## 5. The Finny-table refresh cache

`input_bucket_cache_: MultiArray<BucketCacheEntry, 2, kInputBucketCount>`
(`accumulator.h:418`) — 2 (mirrored) × 12 (king bucket) = 24 top-level slots,
but each `BucketCacheEntry` (`accumulator.h:185-200`) internally holds
**both perspectives'** accumulator plus their own piece/side bitboard
snapshots (`piece_bbs`/`side_bbs` indexed `[2][...]` by perspective), so the
structure covers 2(perspective) × 2(mirror) × 12(bucket) = 48 distinct
cached (accumulator, board-snapshot) pairs in total — comparable in spirit
to, though organized differently from, Reckless's fully flat
2(pov)×2(mirror)×10(bucket) = 40-slot cache
(`reckless_reference/12-nnue-accumulator-psq.md` §3).

`RefreshPerspective` (`accumulator.h:218-305`) is the one function that both
`SetFromState` (root refresh, `reset=true`) and `ApplyChanges`'s
forced-refresh branch (mid-search, `reset=false`) call through — a nice
unification: "totally from scratch" and "diff against a cached snapshot" are
the *same* code path, parameterized only by whether the cache slot's
`Reset()` is called first (which empties its stored bitboards, so the
subsequent diff against "nothing" degenerates into "add every piece").
Given the target cache slot, it diffs the *current* board's per-piece-type
bitboards against the slot's *stored* ones (`to_remove = ~new & old`,
`to_add = new & ~old`, `accumulator.h:249-260`), collects up to 32 add/sub
feature-row pointers, then applies them in **groups of 4** (unrolled,
`accumulator.h:273-298`: `perspective_accumulator[i] += adds[n-4][i] +
adds[n-3][i] + adds[n-2][i] + adds[n-1][i]` for every `i` in one pass) with
a remainder loop for the final 0-3 — a different micro-strategy than
Reckless's register-blocked SIMD `apply_changes` (`psq.rs:170-202`) for the
same goal (amortize the 1536-wide sweep over however many simultaneous
add/sub features apply). Finally the cache slot's stored bitboards/values
are overwritten to the current board, so it's always "the last position
this bucket was computed for," ready for the next cheap diff. This is not
part of the ply-indexed search stack — it's one long-lived structure per
`Accumulator` (i.e. per board/thread), persisting across the whole search
tree and across moves in a game, which is the actual payoff over a plain
"recompute from bias every refresh."

## 6. Output-bucket selection

`GetOutputBucket` (`accumulator.h:395-398`): `bucket = min((popcount(occupied)
- 2) / kBucketDivisor, kOutputBucketCount - 1)`, where `kBucketDivisor =
(32 + 8 - 1) / 8 = 4` (integer division, a `constexpr`, `accumulator.h:12-13`).
A plain arithmetic bucket, not a hand-tuned lookup table — contrast
Reckless's explicit 33-entry `OUTPUT_BUCKETS_LAYOUT` table
(`reckless_reference/13-nnue-inference-simd.md` §1), which is *not*
perfectly linear in piece count (it front-loads resolution toward the
"many pieces" end). Same idea (bucket by material/piece count), simpler
implementation here because a straight division happens to be an adequate
approximation for a 12-vs-33-granularity tradeoff Integral apparently
accepted.

## 7. The quantized forward pass, step by step

Entry point `nnue::Evaluate` (`nnue.cc:38-309`), SIMD branch
(`BUILD_HAS_SIMD && !SPARSE_PERMUTE`, `nnue.cc:47-236`):

1. **`accumulator.ApplyChanges()`** brings the head accumulator entry up to
   date (§4); `bucket = accumulator.GetOutputBucket(state)` (§6).
2. **FT activation** (`nnue.cc:60-126`): for `them ∈ {0,1}` (own perspective,
   opponent's — own half first, which is what makes the net side-to-move
   relative), split the 1536-wide accumulator into two 768-wide halves and,
   per SIMD-width chunk: clip the low half to `[0, QA]`
   (`simd::Clip`/`_mm512_min_epi16(max(x,0),QA)`), clip the high half only
   from above (`simd::Min(x, QA)` — no lower clamp), left-shift the low half
   by `16 - kFtShift` and take the high 16 bits of the product with the high
   half via `MulhiEpi16` (a mul-then-keep-top-half trick avoiding a 32-bit
   widen), then pack the two resulting `i16` halves down to `u8` via
   **unsigned-saturating** `PackusEpi16` — which floors any residual
   negative to 0 for free, exactly why the high half only needed an upper
   clamp. This is the *identical* pairwise-mulhi-SCReLU trick Reckless's
   vectorized `activate_ft` uses (`reckless_reference/13-nnue-inference-simd.md`
   §2.1), same `FT_SHIFT`-named constant (here `kFtShift = 9`,
   `nnue.cc:45`), applied to an accumulator exactly twice as wide.
3. **NNZ sparse index building** (`nnue.cc:98-125`): the 1536 packed `u8`
   feature outputs are grouped into 4-byte lanes and checked for nonzero via
   `GetNnzMask` (a `_mm512_cmpgt_epi32_mask`/AVX2-movemask over the same
   packed bytes reinterpreted as `i32` lanes); the resulting bitmask is
   consumed 8 bits at a time through a **precomputed 256-entry lookup
   table** (`sparse::nnz_table`, `sparse.h:14-33` — one `NnzEntry{indices:
   [u16;8]}` per possible byte pattern, built via `constexpr` at compile
   time) that expands each set bit to its relative group index in one
   `_mm_loadu_si128`/`_mm_storeu_si128` pair, with a running `nnz_base`
   offset converting relative→absolute indices. Same "movemask + LUT
   expand" recipe as Reckless's `find_nnz` and Stockfish's `NNZInfo` (see
   `reckless_reference/13-nnue-inference-simd.md` §2.2) — convergent, not
   novel here.
4. **Sparse int8 L1 matvec** (`nnue.cc:132-168`): walks `nnz_indices` two
   groups at a time, broadcasting each group's 4 packed bytes to every SIMD
   lane (`SetEpi32`) and calling `DpbusdEpi32x2` — a dual `dpbusd` that
   folds two nonzero groups' contributions into one accumulation chain for
   instruction-level parallelism (mirrors Reckless's `double_dpbusd`) —
   against the (build-time-transposed, §8) `l1_weights[bucket]` table,
   accumulating into a 16-wide `i32` array (all of `kL2Size` fits in one or
   two SIMD registers). A remainder loop handles the odd leftover group with
   single-group `DpbusdEpi32`. **Both `DpbusdEpi32` variants
   (`shared/simd.h:89-101` for AVX-512) are implemented as the manual
   `maddubs+madd+set1_epi16(1)` emulation ladder, never the native
   single-instruction `_mm512_dpbusd_epi32` VNNI intrinsic** — see the
   caution in §8.
5. **Dequantize + L1 activation** (`nnue.cc:170-190`): `kL1Normalization =
   2^kFtShift / (QA² · QB)` — same formula shape as Reckless's
   `DEQUANT_MULTIPLIER`, but with `QB = kL1Quantization = 128` here vs
   Reckless's 64 (half the multiplier, double the L1 weight resolution) —
   converts the `i32` sums to `f32` via one FMA (`MultiplyAddPs`, adding the
   bias in the same instruction), then clamps to `[0, 1]` (plain
   ClippedReLU, no squaring past the FT stage — same choice Reckless makes).
6. **L2 dense affine, `16→32`** (`nnue.cc:192-205`): output accumulator is
   **initialized from the bias vector** (`std::memcpy` from
   `l2_biases[bucket]`, not zero-then-add-bias-at-the-end — the same
   bias-as-initial-accumulator ordering Reckless's notes flag as a
   bit-exactness requirement for cross-backend agreement, §7 of that note),
   then for each of the 16 L1 outputs, broadcast it and FMA against its
   32-wide weight row (`l2_weights[bucket]`, transposed to L2-major at build
   time, §8) — plain dense, no sparsity exploited (too small/cheap to
   bother, same judgment Reckless makes for its equivalent layer).
7. **L2 activation** (`nnue.cc:207-213`): clamp to `[0,1]`, no squaring.
8. **L3/output, `32→1`** (`nnue.cc:215-234`): accumulates into
   `kResultChunks = 64 / sizeof(Vepf32)` independent partial-sum SIMD
   registers in parallel (more ILP, same idea as step 4's dual-dpbusd,
   generalized to floats), FMA-ing each chunk of `l3_weights[bucket]`
   against the corresponding `l2_output` slice, then a single
   `simd::ReduceAddPs` horizontal reduction across all the partial-sum
   registers, `+ l3_biases[bucket]`, `× kEvalScale (200)`, cast to `Score`.
9. **Scalar fallback** (`nnue.cc:238-308`) mirrors the same arithmetic shape
   step for step (same clamp/mulhi-shift formula, same bias-as-initial-sum,
   same recursive-halving final reduce via `ReduceAddPsRecursive`,
   `shared/simd.h:402-407`) — the reference the SIMD path is presumably
   meant to match, though (unlike Reckless) no inline comment in this
   vendored snapshot explicitly calls out an `f32::mul_add`-vs-`a*b+c`
   bit-exactness requirement; worth treating the same discipline as implied
   rather than confirmed absent.

`eval::Evaluate` (`src/engine/evaluation/evaluation.cc:7-9`) is a bare
passthrough to `nnue::Evaluate` — no visible outer material-scaled optimism
blend, 50-move damping, or correction-history term at this layer (contrast
Reckless's `evaluation.rs::correct_eval`, `reckless_reference/13-...md` §2.5,
and our own design/11 §3.6 outer-shape plan); if Integral applies any such
correction it must live in the search layer, out of scope for this note.

## 8. The SIMD abstraction layer

`shared/simd.h` (`1-410`) picks **exactly one** of two backends
(`BUILD_HAS_AVX512` or `BUILD_HAS_AVX2`, mutually exclusive `#elif` chain,
`simd.h:79,222`) at compile time based on which `BUILD_*` macro CMake
defined (`simd.h:6-71` — `BUILD_NATIVE` probes real `__AVX512F__` etc.;
the five named tiers `BUILD_VNNI512/AVX512/AVX2_BMI2/AVX2/SSE41_POPCNT`
hard-code the flag set for that tier regardless of the *actual* compiling
machine, `simd.h:20-68`). Each backend exports the same ~20 function names
(`ZeroEpi16`, `LoadEpi16`, `Clip`, `MulhiEpi16`, `PackusEpi16`,
`DpbusdEpi32`/`DpbusdEpi32x2`, `ReduceAddPs`, `GetNnzMask`, ...) at
namespace scope — call sites elsewhere (`nnue.cc`) never see which backend
is active, resolved purely by which `#elif` branch compiled. This is the
same "same free-function names, `cfg`-selected implementation, ordinary
calls, zero indirection" pattern already documented as the standout Rust
idiom in the Reckless SIMD note (`reckless_reference/13-nnue-inference-simd.md`
§3.1) — here done in C++ with the preprocessor rather than Rust's
`#[cfg]`+module system, so it's less clean syntactically (a long `#elif`
ladder duplicating each function's full body per backend rather than
nested-module `pub use` re-export) but the *shape* of the idea — "one
logical interface, N backend modules picked at compile time, no runtime
branch" — is identical across all three codebases we've now studied
(Stockfish's macro cascade, Reckless's `cfg`-module tree, Integral's
`#elif`-guarded namespace).

`kAlignment` (64 for AVX-512, `max(8, sizeof(Vepi16))` for AVX2,
`simd.h:87,230`) and `kPackusOrder` (`{0,2,4,6,1,3,5,7}` for AVX-512,
`{0,2,1,3}` for AVX2, `simd.h:86,229`) are backend-specific constants
consumed by the build-time preprocessor (§9), not by inference itself —
they describe how each ISA's native `packus`/lane-pack instruction
interleaves 128-bit sub-lanes, so weight rows can be pre-permuted to match
before the binary ever runs.

**A caution worth flagging explicitly**: `BUILD_HAS_AVX512VNNI` is defined
(`simd.h:12,22,32,42,52,62`) and threaded all the way through the
`BUILD_VNNI512` tier's flag set (`-march=znver4 -mtune=znver4`,
`CMakeLists.txt:87`, targeting AMD Zen 4's real AVX-512 VNNI hardware), but
**no code anywhere in this vendored snapshot actually branches on it** — a
repo-wide grep for `BUILD_HAS_AVX512VNNI`/`dpbusd_epi32`/`_mm512_dpbusd`
outside its own six definition sites returns nothing. `DpbusdEpi32` in the
AVX-512 branch (`simd.h:89-93`) always uses the manual
`maddubs+madd+set1_epi16(1)` emulation ladder — the same sequence the
AVX2 backend uses — never the single-cycle native
`_mm512_dpbusd_epi32` instruction VNNI hardware provides. So, at least as
captured here, the `BUILD_VNNI512` tier buys znver4 code tuning + AVX-512 +
fast BMI2 PEXT, but *not* an actually-VNNI-accelerated dot product — the
feature flag is plumbed through but not wired to a fast path. (This may
have been fixed in a later upstream revision not present in this vendored
snapshot; treat as "true of this snapshot, re-check upstream before relying
on it.") Reckless, by contrast, does nest a real `avx512vnni`-vs-not branch
one level inside its AVX-512 backend specifically for `dpbusd`
(`reckless_reference/13-nnue-inference-simd.md` §3.3, "tiered fallback
within one backend"). Concrete lesson: a `BUILD_HAS_X` flag existing and
being threaded through a build matrix is not evidence that the hot kernel
actually uses `X` — always grep for the actual intrinsic, not just the
feature macro.

## 9. Build-time network preprocessing

`preprocess/net_processing.cc` (`1-113`) is a **separate CMake target**
(`preprocess/CMakeLists.txt`, built with the *same* architecture flags as
the main engine target, options threaded through from the top-level
`CMakeLists.txt:58-65`) whose only job is: read a raw trained `.nnue` file
into `RawNetwork`, transform it into `Network`'s runtime layout, write it
back out. The transform (`ProcessNetwork`, `net_processing.cc:6-81`):

- `feature_weights`/`feature_biases` copied as-is *except*, when SIMD is
  enabled, every 128-bit block is re-ordered by `kPackusOrder`
  (`net_processing.cc:14-37`) — undoing, ahead of time, the lane-interleave
  that `PackusEpi16` will apply at runtime, so the packed byte order
  produced during inference lines up with the feature index each byte is
  supposed to represent.
- `l1_weights` transposed from `[bucket][L2][L1]` (trainer-natural) to
  `[bucket][L1][L2]` (`net_processing.cc:44-51`), **then** (SIMD builds
  only) a second pass builds `l1_weights_alt` by re-grouping every 4
  consecutive `L1` features' weight columns contiguously
  (`net_processing.cc:53-68`) — this is exactly the layout `DpbusdEpi32x2`
  (§7 step 4) needs to read one contiguous 4-byte-times-16-lane chunk per
  dot product without re-gathering at runtime.
- `l2_weights` transposed from `[bucket][L3][L2]` to `[bucket][L2][L3]`
  (`net_processing.cc:71-78`) for the broadcast-then-FMA loop shape (§7
  step 6).

The top-level `CMakeLists.txt` wires this in as a build dependency
(`add_custom_command`/`add_custom_target run_preprocess`, `CMakeLists.txt:
70-83`), so `EVALFILE` (whether user-supplied or auto-downloaded, see below)
is *always* run through `preprocess` before being handed to `INCBIN`
(`nnue.cc:21`, a third-party public-domain single-header library that
embeds a file into the binary via inline `.incbin`/`objcopy`-equivalent
tooling — Rust's `include_bytes!` is the direct analogue) — the engine
binary never reads or transposes a raw net at runtime; the shipped bytes
already match `Network`'s exact expected memory layout for whichever SIMD
tier that particular binary was built for. This is a clean separation our
own plan doesn't currently have an equivalent step for (§10 item 1).

Net provenance: `CMakeLists.txt:24-48` defines `NETWORK_NAME = zekrom-v7`,
constructs a GitHub Releases URL
(`aronpetko/integral-networks/releases/download/<name>/<name>.nnue`), and
`file(DOWNLOAD ...)`s it automatically if the local file is missing —
i.e. a **separate, versioned "networks" repository**, fetched by URL at
configure time, matching Reckless's equivalent `build.rs`-fetch pattern
(`reckless_reference/13-nnue-inference-simd.md` §4 table, "net provenance"
row) and our own D15 ("nets are build artifacts, not source... release
attachments").

## 10. The compile-time build matrix (no runtime CPU dispatch)

Five mutually exclusive tiers, each a distinct `-march`/tuning string
(`CMakeLists.txt:86-91`):

| tier | flags |
|---|---|
| `BUILD_VNNI512` | `-march=znver4 -mtune=znver4` (+ `USE_PEXT`) |
| `BUILD_AVX512` | `-march=x86-64-v4 -mtune=skylake-avx512` (+ `USE_PEXT`) |
| `BUILD_AVX2_BMI2` | `-march=haswell -mtune=haswell -mavx2 -mbmi2` (+ `USE_PEXT`) |
| `BUILD_AVX2` | `-march=bdver4 -mno-tbm -mno-sse4a -mno-bmi2 -mtune=znver2` |
| `BUILD_SSE41_POPCNT` | `-march=nehalem -mtune=sandybridge` |

plus `BUILD_NATIVE` (`-march=native`, the default, with a further runtime
`lscpu`/compiler-macro probe at *configure* time, not player-machine runtime,
to decide whether to also pass `-DUSE_PEXT` — `CMakeLists.txt:110-158`,
including a specific carve-out disabling PEXT on Zen 1/2 where hardware PEXT
is known to be slow, and a further special case re-enabling it only for AMD
EPYC 7B on Zen 2). The `Makefile` (`1-101`) exposes each tier as a `make
<tier>` target that reconfigures and rebuilds from scratch, copying the
result to a tier-suffixed executable name. **There is no runtime
`cpuid`/feature-detection dispatch anywhere in this codebase** — every
binary is compiled for exactly one fixed target and will `SIGILL` on a CPU
lacking what it was built for. This is the same tradeoff already discussed
in the Reckless SIMD note §3.1/§5 (compile-time `cfg`-module dispatch,
"expects to be rebuilt per deployment target") and directly matches our own
D16 "fast builds" note (`RUSTFLAGS="-C target-cpu=x86-64-v3"`, portable
default kept separate) — a third data point that top engines ship a
build-matrix of pinned targets as the *tested/distributed* configuration,
reserving runtime dispatch (if used at all) for a portable fallback tier.

## 11. Sparse neuron permutation (`SPARSE_PERMUTE`, a research tool)

`sparse.h:35-105`, gated behind a *separate*, mutually-exclusive-with-SIMD
CMake option (`option(SPARSE_PERMUTE)`, `CMakeLists.txt:51-54` — note the
comment there: "requires turning off AVX"). The idea: instrument a bench
run to count, per accumulator neuron pair, how often that neuron is
non-zero after activation (`sparse::CountActivations`, `sparse.h:41-46`,
called from the non-SIMD `Evaluate` path when this mode is on, `nnue.cc:
252-254`), then offline (`SavePermutedNetwork`, `sparse.h:48-105`) stable-sort
neuron indices by descending activation frequency and rewrite
`feature_biases`, every input bucket's `feature_weights` column, and every
output bucket's `l1_weights` row into that new neuron order, producing a
new `.nnue` file with frequently-active neurons clustered at low indices.
This is a **profile-guided reordering of the accumulator's own neuron
axis** — distinct from (and complementary to) the NNZ sparse *group-skip*
optimization (§7 step 3), which skips whole zero groups at inference time
regardless of neuron order; permutation instead aims to make the *surviving*
nonzero groups denser/more contiguous, improving locality for whatever
inference-time sparse-skip logic already exists. This is not present in
either Stockfish or Reckless per our prior studies, and no bench numbers
for its effect are visible in the commit message read for it (§12, PR #185)
— flag as a genuinely novel idea worth trying, not a proven win.

## 12. Evolution timeline (from `git log`)

The project's first commit is 2023-10-25; movegen/board work occupies
roughly the first ten months. NNUE-relevant milestones, all timestamped via
`git show -s --format=%ad`:

| date | commit | milestone |
|---|---|---|
| 2024-08-27 | `cd3f7acb` | "first nnue" |
| 2024-09-06 | `99b7b22e` (#52) | NNUE lands on `main` |
| 2024-09-06 | `128b64d2` (#53) | "Accumulator speedups" |
| 2024-09-09 | `5c030fca` (#58) | SIMD for inference |
| 2024-09-18 | `796c4dca` (#69) | horizontal mirroring |
| 2024-09-21 | `abbd1390` (#75) | hidden layer → 1280, +4 input buckets |
| 2024-09-23 | `51f29eed` (#77) | lazy accumulator updates |
| 2024-09-25 | `bb56cb85` (#81) | Finny tables |
| 2024-10-22 | `62618129` (#99) | hidden layer → 1536, 9 king buckets (factorized) |
| 2024-11-13 | `287d22ed` (#130) | SIMD for the accumulator refresh path |
| 2025-01-01 | `581cebc1` (#153) | multi-layer network (the L2/L3 tail) |
| 2025-04-07 | `6223b7fb` (#185) | sparse neuron permutation (§11) |
| 2025-04-24 | `576bef0b` (#190) | build-time network preprocessing (§9) |
| 2025-04-29 | `45e0e036`/`223b80a6` (#192/#193) | Finny-table fused updates, "massively improve bench speed" |

The headline data point: **first NNUE at all (Aug 27) to lazy updates +
Finny tables (Sep 25) is exactly 29 days** — SIMD inference, horizontal
mirroring, a bigger net, lazy accumulator updates, *and* the Finny-table
refresh cache all landed within one month of shipping the first working
NNUE. That cadence — and what infrastructure made confidently merging that
many search-behavior-changing PRs in a month possible — is exactly the
question the success-factor notes (09/10/11) are for; this note only
establishes the dates and commit hashes as evidence to build on there.

## 13. Comparison summary

| axis | Stockfish (per design/11 + Reckless note §4) | Reckless | Integral | our plan (design/11 §3.4) |
|---|---|---|---|---|
| FT/accumulator width | 1024 (current dev tree) | 768 | **1536** | 256 (v1), upgrade path to 512/1024 |
| post-FT depth | `→32→32→1`, all-integer, dual activation + skip connection | `→16→32→1`, int8 first layer then f32 tail | `→16→32→1`, int8 first layer then f32 tail (same shape as Reckless) | `→1` single output layer (v1; deeper tail is upgrade path) |
| input buckets | king-bucketed (`HalfKAv2_hm`) | 10 | **12** | none in v1 (upgrade path) |
| output buckets | 8, by material | 8, by piece-count (33-entry table) | 8, by piece-count (`/4` formula) | none in v1 (upgrade path) |
| QA / QB | — | 255 / 64 | **255 / 128** | 255 / 64 (matches Reckless, not Integral) |
| accumulator stack | `StateInfo`-embedded (structural freshness) | fixed `MAX_PLY` boxed slice, ply-indexed | **growable `std::vector`**, ply-indexed | fixed `MAX_PLY`+margin boxed slice, ply-indexed |
| dirty-piece record | — | minimal `(mv,piece,captured)`, lazily re-derived | **eager, fully materialized at make-time** (matches our plan) | eager, fully materialized at make-time |
| null move vs accumulator | — | not confirmed in prior note | **no stack push/pop at all** | plan implies an empty-record push (§4 above) |
| refresh cache (Finny) | `AccumulatorCaches` (adopted from the community) | 40 slots, flat `pov×mirror×bucket` | 24 top-level slots × 2 perspectives = 48 effective | none in v1; planned once king buckets land |
| build/dispatch | multiple official prebuilt binaries + macro cascade | `cfg`-module tree, expects per-target rebuild | **5-tier CMake/Makefile matrix, zero runtime dispatch**, plus a separate per-tier net-preprocessing binary | `LazyLock<Kernels>` runtime dispatch (portable default) + D16 pinned-target fast build |
| net provenance | ships GPL'd SF-trained nets | own repo, `build.rs` fetch, no validation | own repo, CMake `file(DOWNLOAD)` fetch, no validation | own container format, magic+version+checksum, real error type |

## Adapt to us

Ordered roughly by actionability against design/11 and the P13 phase target.

1. **A dedicated build-time net-preprocessing step**, separating "the format
   `bullet` (or our own trainer) naturally emits" from "the exact byte
   layout our SIMD kernels want to read," transformed once and baked into
   the embedded bytes rather than fixed up on every engine start.
   *Doc/phase*: design/11 §3.4.5 (net format) and §3.4.7 item 3 (`netpack`
   helper) — extend `netpack` to also perform any SIMD-lane-order
   permutation the chosen kernel layout needs, rather than assuming the
   container's payload layout is already 1:1 with what the AVX2 kernels
   read. *Effort*: small now (our v1 net is a single small `256`-wide
   layer with no transpose/permute need yet); becomes directly relevant the
   moment we add a deeper int8 L1/L2 cascade (item 4 below). *Impact*: keeps
   the runtime loader's already-planned validation (magic/version/checksum)
   free of doing layout math on every load; the loader validates, a
   separate offline step (or `netpack` itself) does the one-time transform.

2. **Treat our `LazyLock<Kernels>` runtime dispatch as the portable-build
   fallback, not the only dispatch mechanism** — Integral (like Reckless)
   confirms that top engines ship a **compile-time-pinned build matrix**
   (5 tiers here) as the tested/distributed configuration, with runtime
   detection either absent (Integral) or reserved for a portable tier. Our
   existing D16 fast-build note already anticipates this; this is a second
   confirming data point, not a new idea. *Doc/phase*: design/11 §3.4.6,
   design/01 §3.7 (D16). *Effort*: small (a CI/release matrix, not new
   runtime code). *Impact*: nps on the pinned SPRT/bench configuration; no
   behavior risk since both dispatch paths must already agree bit-exactly.

3. **Skip the accumulator-stack push entirely on null moves.** Integral's
   `MakeNullMove` never calls into the accumulator at all — nothing about
   either perspective's active features changes, so no stack entry is
   needed. *Doc/phase*: design/11 §3.4.4, the `do_null_move` wrapper (which
   currently implies pushing an empty dirty record). *Effort*: trivial —
   have `do_null_move` simply not touch the `NnueContext` stack, and make
   sure the *next* real move after a null move still walks back correctly
   (it will: `ApplyChanges`'s backward search just skips over the ply that
   was never pushed, exactly as if it never existed, since there IS no
   entry at that ply to skip — this requires the null-move ply to not
   consume a stack slot number either, i.e. ply bookkeeping and accumulator
   stack indexing must agree that a null move ply has no entry). *Impact*:
   tiny (fewer stack slots touched under NMP-heavy trees, marginally
   simpler invalidation reasoning) — a free, correctness-preserving
   simplification worth taking now rather than after P13 ships.

4. **When we build the post-v1 deeper cascade (int8 sparse `L1→L2` +
   NNZ + `dpbusd`), budget for the same three build-time weight-layout
   transforms Integral needs**: transpose the first affine layer's weights
   to input-major, build a "process 2 groups at once" permuted variant if
   we adopt a `DpbusdEpi32x2`-style dual-dot kernel, and transpose the next
   dense layer similarly. *Doc/phase*: design/11 §3.4.1 "upgrade path" (the
   `L1 256→512→1024`, deeper tail step), P13 sub-phase. *Effort*: medium,
   but only once we actually add a second dense layer — not v1 scope.
   *Impact*: throughput of the forward pass at that future point.

5. **If/when we add an AVX-512/VNNI kernel tier, verify the `dpbusd`
   kernel actually emits the native VNNI instruction, don't just gate a
   feature flag.** Integral's own vendored snapshot defines
   `BUILD_HAS_AVX512VNNI` and ships a `BUILD_VNNI512` compile tier, but the
   `DpbusdEpi32` kernel in that same tier still uses the manual
   `maddubs+madd` emulation ladder — the flag exists but nothing branches
   on it. *Doc/phase*: design/11 §3.4.6 ("AVX-512/VNNI: later, same
   dispatch seam, only after AVX2 is measured") — add an explicit
   verification step to that future work item: a test or bench comparing
   instruction counts (or at minimum nps) between the VNNI-enabled and
   VNNI-disabled build of the identical kernel, so we would notice this
   class of bug rather than ship a "VNNI build" that doesn't use VNNI.
   *Effort*: trivial (one checklist line now; one small bench/asm-inspection
   step when we get there). *Impact*: avoids a real, now-documented bug
   class from a top-6 engine's own code.

6. **Confidence, not a change**: our planned eager `DirtyPiece{adds,subs}`
   record (populated at `make_move_with` hook time with concrete
   squares/pieces) matches Integral's `AccumulatorChange` construction in
   `Board::MakeMove` almost exactly, and differs from Reckless's
   lazier-reconstruct-from-`(mv,piece,captured)` approach. Cite both
   Integral (`board.cc:214-262`) and Reckless as independent precedent when
   this design choice needs justifying in review — two out of two studied
   top engines that build the record eagerly did it our way; the one that
   didn't (Reckless) made a legitimate but different tradeoff (§3 above).

7. **A second data point for king-bucket granularity when we design our
   own** (12 buckets, asymmetric rank/file layout weighted toward the back
   two ranks) alongside Reckless's 10-bucket layout — two real, different,
   hand-tuned schemes to compare against when design/11's upgrade path
   actually specs out king buckets, rather than inventing one from
   first principles. *Doc/phase*: design/11 §3.4.1 upgrade path.
   *Effort*: design-only (a paragraph noting both shapes exist as precedent).
   *Impact*: informs a future net-architecture decision; not urgent.

8. **Not worth adopting yet, flagged for later**: the `SPARSE_PERMUTE`
   profile-guided neuron-reordering research mode (§11) is a genuinely
   novel idea with no visible before/after bench numbers even in Integral's
   own history — worth a one-line mention in design/11's upgrade-path list
   as "a research idea to try once NNZ sparsity is already in place and
   measured," not something to build now. *Effort*: large (needs an
   instrumented bench mode plus an offline permutation tool) for unproven
   payoff. *Impact*: unknown; explicitly speculative.
