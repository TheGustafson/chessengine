# Reckless — NNUE quantized forward pass & SIMD kernels

Source studied: `reckless_reference/Reckless` (v0.10.0-dev, Rust edition 2024,
AGPLv3), commit as vendored in this repo. Files read in full: `src/nnue.rs`
(module root, constants, `Network`, `Parameters`), `src/nnue/forward/{scalar,
vectorized}.rs`, `src/nnue/simd/{avx2,avx512,neon,wasm,scalar}.rs`,
`src/evaluation.rs`, plus enough of `src/nnue/accumulator{.rs,/psq.rs,
/threats.rs}` to understand what the forward pass consumes, and Stockfish's
`src/nnue/{nnue_architecture.h, layers/sqr_clipped_relu.h,
layers/affine_transform_sparse_input.h, nnz_helper.h, simd.h}` for the
cross-engine comparison. All line numbers below are into the Reckless tree
under `reckless_reference/Reckless/`. **No Reckless code is reproduced** —
everything is restated in prose/original pseudocode; the few quoted items are
bare numeric constants and short arithmetic identities (facts, not
expression).

## 0. One-paragraph summary

Reckless runs a small, shallow, **hybrid-quantized** network: a big sparse
input stage (768-wide dual accumulator, produced by two *separate* feature
sets — piece-square and "threats" — summed and squared-clipped down to `u8`
via a mul-hi trick) feeds one `int8` sparse affine layer (`768 → 16`,
VNNI-style `dpbusd`), and from there **everything is plain `f32`**: two more
small dense affine layers (`16 → 32 → 1`) with ordinary clamp activation, a
fixed-order horizontal reduction, and a scalar multiply into centipawns. Only
the first, largest, sparsest layer is integer; the entire tail is float. That
one architectural choice is the single biggest thing worth taking away, and
it is discussed first in §4. The other half of the note — the SIMD
abstraction — is a genuinely clean piece of Rust: a ~15-function free-function
module (not a trait, not a macro) reimplemented once per ISA and selected
purely by nested `#[cfg(target_feature = …)] mod X; pub use X::*;`, so the
numeric algorithm is written exactly once and the backends are swapped in at
zero runtime cost.

## 1. Architecture at a glance

Constants (`src/nnue.rs:82-99`):

| name | value | role |
|---|---|---|
| `NETWORK_SCALE` | 380 | final float→centipawn multiplier |
| `INPUT_BUCKETS` | 10 | king-position input buckets (one per perspective) |
| `OUTPUT_BUCKETS` | 8 | material-count output buckets |
| `L1_SIZE` | 768 | feature-transformer / accumulator width (**not** an SF-style "layer 1 output" — see naming note below) |
| `L2_SIZE` | 16 | output width of the first affine layer |
| `L3_SIZE` | 32 | output width of the second affine layer |
| `FT_QUANT` | 255 | feature-transformer quant scale (`QA`) |
| `L1_QUANT` | 64 | first-affine-layer quant scale (`QB`) |
| `FT_SHIFT` | 9 | right-shift baked into the FT activation |

**Naming trap for a reader used to Stockfish/bullet conventions**: Reckless
names each buffer after the layer that *consumes* it, not the layer that
*produces* it. `L1_SIZE` (768) is the accumulator/feature-transformer width,
fed into the function `propagate_l1`; `L2_SIZE` (16) is `propagate_l1`'s
*output* and `propagate_l2`'s input; `L3_SIZE` (32) is `propagate_l2`'s output
and `propagate_l3`'s input. So the real shape is:

```
768 (FT, dual accumulator, int8 activation) → propagate_l1 → 16 (f32)
   → propagate_l2 → 32 (f32) → propagate_l3 → 1 (f32) → × NETWORK_SCALE → i32 cp
```

Two independent input feature sets are summed before activation: a
piece-square-style set (`PstAccumulator`, one weight table `ft_piece_weights`
of shape `[i16; 768]` per `(input_bucket, side, piece_type, square)` combo —
`INPUT_BUCKETS * 768 = 7680` rows, `nnue.rs:359`) and a much larger "threats"
set (`ThreatAccumulator`, weight table `ft_threat_weights: [i8; 768]` with
**66 864** rows, `nnue.rs:358` — one row per `(attacking piece, from-square,
attacked piece, to-square)` combination that can occur on a board, mirrored
for king side). Both accumulators are `[i16; 768]` per perspective
(`accumulator/psq.rs:19`, `accumulator/threats.rs:60`) and are maintained
incrementally (add/sub on `make`/`unmake`, refreshed wholesale on king moves
that cross an input-bucket boundary — `nnue.rs:242-284`). This "threats" input
set is the same idea Stockfish's dev tree has recently added
(`Features::FullThreats`, `stockfish_reference/Stockfish/src/nnue/features/
full_threats.h`) — convergent evolution, not something Reckless copied from
or Stockfish copied from Reckless; both independently found that a
threats-aware auxiliary feature set is worth the extra accumulator.

Output-bucket selection (`OUTPUT_BUCKETS_LAYOUT`, `nnue.rs:113-123`) is a
33-entry table indexed by total piece count (`popcount` of all occupancy,
0..32) that maps piece count monotonically down to one of 8 buckets (bucket 0
covers 0-8 pieces, bucket 7 covers 29-32 — heavier on the "many pieces"
buckets since that's the opening/middlegame range, roughly 3-4 piece-counts
per bucket everywhere else). Each output bucket owns its own full copy of the
`l1_weights/l1_biases/l2_weights/l2_biases/l3_weights/l3_biases` — i.e. 8
completely independent small MLPs sharing one feature transformer, selected
once per `evaluate()` call (`nnue.rs:287`, `output_transformer`). Input-bucket
selection (`INPUT_BUCKETS_LAYOUT`, `nnue.rs:101-111`) is the usual
king-square-driven table: the two center files split into 4+4 buckets each
(mirrored left/right), and the entire lower half of the board (ranks 3-8,
i.e. "king not near its own back rank/corner") collapses into just 2 giant
buckets (8 and 9) — very coarse away from the king's own camp, fine-grained
only for back-rank king placement.

## 2. The forward pipeline, step by step

Entry point is `Network::output_transformer` (`nnue.rs:286-301`), called
from `evaluate()` after the two accumulators are confirmed up to date. It
runs five steps, all inside one `unsafe` block, dispatched to whichever of
`forward::{scalar, vectorized}` was selected at compile time:

```text
ft_out            = activate_ft(pst_acc, threat_acc, side_to_move)   // [u8; 768]
(nnz_idx, nnz_n)  = find_nnz(ft_out)                                 // sparse index list into 768/4 groups
l1_out            = propagate_l1(ft_out, nnz_idx[..nnz_n], bucket)   // [f32; 16]
l2_out            = propagate_l2(l1_out, bucket)                    // [f32; 32]
l3_out            = propagate_l3(l2_out, bucket)                    // f32 scalar
cp                = (l3_out * NETWORK_SCALE) as i32
```

### 2.1 Feature-transformer activation (`activate_ft`)

Each perspective's accumulator (own and flipped) is 768 `i16` values, split
into two halves of 384. For each half-pair, in my own restatement of
`forward/scalar.rs:9-25`:

```text
for half in [own_perspective, other_perspective]:
    lhs = clamp(pst[half][i]     + threat[half][i],     0, FT_QUANT)   // i in 0..384
    rhs = clamp(pst[half][i+384] + threat[half][i+384], 0, FT_QUANT)
    out[half*384 + i] = ((lhs as i32 * rhs as i32) >> FT_SHIFT) as u8
```

This is the "pairwise-multiply SCReLU" trick: instead of squaring one
clipped-ReLU value (`clamp(x,0,QA)^2`), the 768-wide accumulator is treated as
two interleaved halves and the halves are clipped and multiplied against each
other, elementwise, producing 384 outputs per perspective and 768 total
(own-perspective half first — that's what makes the whole net
side-to-move-relative). `FT_SHIFT = 9` rescales the `QA²`-scaled product back
down into a `u8` (`0..=255`) range in one shift.

The **vectorized** version (`forward/vectorized.rs:10-53`) computes the same
thing with a different, cheaper instruction sequence: rather than a 32-bit
multiply, it left-shifts the clamped `lhs` by `16 - FT_SHIFT - MUL_HI_SHIFT`
and takes the *high 16 bits* of `lhs_shifted * rhs` (`_mm256/512_mulhi_epi16`
— a 16-bit "multiply and keep the top half" instruction, one cycle,
vs. widening to 32-bit lanes and shifting). `MUL_HI_SHIFT` is an
architecture-specific fixup constant (0 on AVX2/AVX-512, **1 on NEON** —
`simd/neon.rs:5,35-38` — because ARM's `vqdmulhq_s16` is a *doubling*
multiply-high, so the pre-shift must be one bit smaller to compensate). A
second asymmetry: the vectorized `rhs` is only upper-clamped
(`min(x, QA)`, no lower `max(x, 0)` — `forward/vectorized.rs:36-37`), while
the scalar path clamps both bounds. This is not a bug — the final step packs
the 16-bit product down to `u8` with an *unsigned-saturating* pack
instruction (`packus`), which floors any negative product to 0 for free, so
the lower clamp is redundant in SIMD but **not** in scalar Rust: a plain
`as u8` cast in Rust wraps/truncates rather than saturating, so the scalar
reference implementation has to do the equivalent clamp explicitly or it
would silently produce wrong (wrapped) results. That is a nice concrete
"Rust `as`-cast gotcha" worth remembering when writing a scalar reference path
that must match a saturating-arithmetic SIMD path.

`packus` on AVX2/AVX-512 also needs a lane-fixup permute afterward
(`_mm256_permute4x64_epi64`/`_mm512_permutexvar_epi64`,
`forward/vectorized.rs:45-48`) because those pack instructions operate
within 128-bit sub-lanes and interleave the two 256-bit halves; NEON and wasm
don't need it (`permute` is the identity there, `simd/neon.rs:50-52`,
`simd/wasm.rs:50-52`) since their pack instructions don't cross lanes the same
way.

### 2.2 Sparse nonzero detection (`find_nnz`)

The 768 `u8` FT outputs are reinterpreted as 192 groups of 4 bytes (a `u32`
each). `find_nnz` produces the list of group indices that contain at least
one nonzero byte — this list drives the sparse layer below, since a whole
`u32` group can be skipped in the dot product if all 4 of its bytes are zero
(ReLU-style sparsity: with a squared-clip activation, ~85-95% of the 768
inputs are typically zero). Four independent implementations exist, one per
architecture family, because this step needs a fast "compact the indices of
set bits" primitive that differs the most across ISAs:

- **Scalar** (`forward/scalar.rs:110-128`): a plain OR-reduce over each
  4-byte group, no lookup table.
- **Generic x86 (SSE/AVX2 tier, no AVX-512VBMI2)**
  (`forward/vectorized.rs:159-187`): compute an 8-bit-per-group nonzero
  bitmask via compare+`movemask`, then split it into two 4-bit nibbles and
  use each nibble to index a **precomputed 256-entry lookup table**
  (`SparseEntry { indexes: [u16;8], count }`, built once in
  `Network::new`, `nnue.rs:144-157`) that already lists, for every possible
  8-bit pattern, which of the 8 bit positions were set. This is the classic
  "movemask + LUT expand" trick used across the NNUE-engine community (also
  present in Stockfish's non-AVX-512 `vec_nnz`/`NNZInfo` path,
  `stockfish_reference/Stockfish/src/nnue/nnz_helper.h:107-164`) — not
  something unique to either engine.
- **AVX-512VBMI2** (`forward/vectorized.rs:189-226`): uses the hardware
  `_mm512_maskz_compress_epi16` instruction to compact indices directly from
  a 32-bit mask, no lookup table needed — the ISA does the compaction in
  hardware. Stockfish's AVX-512 path does the equivalent thing
  (`nnz_helper.h:73-100`).
- **NEON** (`forward/vectorized.rs:228-257`) and **wasm**
  (`forward/vectorized.rs:259-300`): both fall back to the LUT-expand
  approach (no compress instruction available), computing the bitmask by
  ISA-appropriate means (a bit-weighted horizontal add on NEON since NEON has
  no `movemask`; narrow+bitmask on wasm).

### 2.3 The sparse `int8` affine layer (`propagate_l1`)

For every nonzero group index `g` (0..192), the four packed `u8` FT outputs
at that group are dotted against the corresponding 4 `int8` weight columns
for all 16 outputs, accumulating into `i32`. The core primitive is a
"dot-product of bytes" (`dpbusd`: unsigned×signed byte pairs, summed into an
`i32` — VNNI's native op, emulated elsewhere via
`maddubs` (`u8×i8 → i16` pairwise) then `madd` (`i16×1 → i32` pairwise widen)
then add). Reckless processes nonzero groups **two at a time**
(`double_dpbusd`) when possible, purely to get more independent accumulation
chains in flight (instruction-level parallelism), falling back to a
single-group `dpbusd` for the odd remainder
(`forward/vectorized.rs:64-95`). This mirrors Stockfish's own
`AffineTransformSparseInput` layer, which uses the identical `ChunkSize = 4`
byte-grouping and the identical dpbusd-emulation ladder
(`stockfish_reference/Stockfish/src/nnue/layers/
affine_transform_sparse_input.h:62-67`); both engines converge on the same
"pack 4 int8 lanes, dpbusd, accumulate i32" recipe — it is close to the only
sane way to do a VNNI-less int8 sparse dot product.

After accumulation, the `i32` sums are **dequantized to `f32` in this one
step and never re-quantized** — this is the architectural fork point vs.
Stockfish (§4):

```text
dequant_multiplier = 2^FT_SHIFT / (FT_QUANT^2 * L1_QUANT)   // = 512 / (255*255*64) ≈ 1.231e-4
l1_out[j] = clamp(pre_activation[j] * dequant_multiplier + l1_bias[j], 0.0, 1.0)
```

(`nnue.rs:99`, `forward/scalar.rs:60-61`, `forward/vectorized.rs:99-107`).
The comment at `forward/scalar.rs:58-59` is explicit that from here on
"order of operations is critical for staying semantically identical to the
vectorized inference. In particular, using FMA instead of separate mul+add
matters" — i.e. the scalar reference path deliberately uses `f32::mul_add`
(fused multiply-add) rather than `a*b+c`, specifically so its rounding
matches the hardware FMA the vectorized path gets from `_mm256/512_fmadd_ps`.
Getting this wrong (using separate mul+add in the "reference" path) would
make the scalar and SIMD backends disagree in the last bit — a portability
trap worth remembering verbatim if we ever write our own dual scalar/SIMD
reference pair.

### 2.4 The two dense `f32` layers (`propagate_l2`, `propagate_l3`)

`propagate_l2` (16→32) is a plain outer-product accumulation: the output
buffer is **initialized to the bias vector** (not zero, with the bias added
at the end — the comment at `forward/scalar.rs:70-71` flags this ordering
too, again for scalar/SIMD parity), then for each of the 16 inputs, its
scalar value is broadcast and FMA'd against that input's 32-wide weight row
into the running output, followed by a final `clamp(0,1)` (plain
ClippedReLU, no squaring at this depth — squaring/SCReLU is a feature-
transformer-only trick here). `propagate_l3` (32→1) is the same FMA-broadcast
pattern but reduces to a single scalar: to guarantee the *same* answer on
every ISA regardless of native vector width, Reckless always sum-reduces into
a fixed 16-lane buffer first (splitting 32 inputs into two rounds of 16 on
narrower ISAs) and then does a **fixed recursive-halving horizontal add**
(`forward/scalar.rs:87-105`) explicitly chosen to match what
`_mm512_reduce_add_ps` does in hardware — so a 4-lane NEON reduction
(`simd/neon.rs:116-126`) and an 8-lane AVX2 reduction
(`simd/avx2.rs:89-103`) are written to produce bit-identical sums to the
native AVX-512 `_mm512_reduce_add_ps` intrinsic
(`simd/avx512.rs:102-104`), rather than each ISA reducing in its "natural"
order. Floating-point addition is not associative, so this is a real
correctness requirement, not a style choice, whenever a test asserts
scalar/vectorized/cross-ISA equality.

### 2.5 Output-bucket / final scale, and the eval outer shape

`output_transformer` multiplies the final `f32` by `NETWORK_SCALE = 380` and
truncates to `i32` centipawns (`nnue.rs:299`). That raw value then passes
through `evaluation.rs::correct_eval` (13 lines total) before reaching search:

```text
eval = (raw * (21032 + material) + optimism[stm] * (1548 + material)) / 27015
eval = eval * (200 - fifty_move_clock) / 200
eval = eval + correction_value
eval = clamp(eval, -TB_WIN_IN_MAX+1, TB_WIN_IN_MAX-1)
```

(`evaluation.rs:4-13`). This is the same *shape* of outer eval blend our own
design doc already specifies (design/11 §3.6: material-scaled optimism blend,
50-move damping, TB clamp) — material-weighted optimism instead of a separate
learned "complexity" signal. Reckless has no PSQT/positional output-head split
(unlike current Stockfish, see §4), so it approximates "complexity" purely by
scaling optimism's contribution with the position's non-pawn material — a
simpler proxy than Stockfish's dedicated NNUE complexity term, but structurally
the same family of correction.

## 3. The SIMD abstraction layer — the Rust-idiom lesson

This is the part of Reckless most directly reusable as a **pattern**,
independent of the chess-specific numerics.

### 3.1 Shape of the abstraction

Two separate nested-module trees inside `nnue.rs`, each following the exact
same recipe (`nnue.rs:17-80`):

```text
module `forward`:
    if any(avx2 | neon | wasm+simd128) at compile time -> submodule
    `vectorized`, re-exported wholesale; else -> submodule `scalar`,
    re-exported wholesale.
module `simd` (same recipe, five-way priority ladder):
    avx512f -> avx512;  else avx2 -> avx2;  else neon -> neon;
    else wasm32+simd128 -> wasm;  else -> scalar
    (each arm: a cfg-gated `mod` + a cfg-gated glob re-export)
```

Every backend module exports the **same set of free-function names** with
the same signatures (`add_i16`, `sub_i16`, `clamp_i16`, `mul_high_i16`,
`splat_i32`, `mul_add_f32`, `clamp_f32`, `dpbusd`, `double_dpbusd`,
`horizontal_sum`, `nnz_bitmask`, plus the three `_LANES` and `MUL_HI_SHIFT`
consts). Call sites elsewhere in the crate (`forward/vectorized.rs`, and
independently `accumulator/psq.rs`/`accumulator/threats.rs` for accumulator
maintenance) write `simd::add_i16(a, b)` etc. and never see which backend is
active — the module-path re-export makes the choice at compile time, with
**zero indirection**: it's not a trait object, not a generic parameter, not a
macro-generated match — just name resolution picking one of five modules.
This is a textbook example of Rust's "same interface, `cfg`-selected
implementation, ordinary function calls" pattern, and it is simpler than the
alternatives:

- vs. a `trait SimdBackend { fn add_i16(...); ... }` + generic
  `fn kernel<B: SimdBackend>()`: no trait, no generic instantiation
  boilerplate, no need to name the backend type anywhere outside `nnue.rs`.
- vs. macros generating one function body per ISA: the *algorithm* code
  (`forward/vectorized.rs`) is written **once**, in plain Rust, calling
  `simd::` functions — no macro-expanded near-duplicates to keep in sync.
- vs. our own design's `LazyLock<Kernels>` struct-of-`unsafe fn`-pointers
  (design/11 §3.4.6): Reckless pays no runtime dispatch/indirect-call cost
  and needs no `is_x86_feature_detected!` check at all, at the price of
  requiring the *compiling* target to already declare the feature
  (`-C target-feature=+avx2` / `target-cpu=x86-64-v3`/`native`) — there is no
  fallback within one binary if the deployed CPU lacks what the binary was
  compiled for (it would `SIGILL`). Reckless's own `Makefile` defaults
  `RUSTFLAGS ?= -C target-cpu=native` and ships an `x64-check` target that
  only `cargo check`s (not builds) x86-64-v1..v4 — i.e. Reckless expects to be
  *rebuilt* per deployment target (matching common practice for
  competition-tested engines built fresh per OpenBench worker), not to ship
  one portable binary that runtime-detects. Compare our D16, which documents
  exactly this same fast-but-target-pinned build as the *supported*
  configuration for strength testing, while keeping the *default* portable
  build runtime-safe — see §5 for how the two approaches can coexist.

### 3.2 The minimal primitive surface, and why it's small

Counting distinct primitives needed across all of §2's algorithm: 3 integer
ops (`add_i16`/`sub_i16`/`clamp_i16`/`min_i16`, `shift_left_i16`,
`mul_high_i16`), 2 packing ops (`packus`, `permute`), 5 float ops
(`splat_f32`/`zero_f32`, `mul_add_f32`, `convert_to_f32`, `clamp_f32`), 2 dot
ops (`dpbusd`, `double_dpbusd`), and 2 reduction ops (`horizontal_sum`,
`nnz_bitmask`) — about 15 functions total, each 1-4 lines of intrinsic code
per backend (`simd/avx2.rs` and `simd/avx512.rs` are each only 108 lines
covering the *entire* interface). Keeping this surface small is what makes
adding a 5th backend (wasm) or a 6th (LoongArch/RISC-V, which neither engine
has yet) tractable — each new backend is "implement 15 small functions,"
not "port the whole forward pass."

Notably, the `simd::scalar` backend (`simd/scalar.rs`, 13 lines total) only
implements 3 of the 15 (`add_i16`, `sub_i16`, `convert_i8_i16`) — because
those 3 are the *only* ones the **accumulator maintenance code**
(`accumulator/psq.rs`, `accumulator/threats.rs`) needs, and that code is
written against `simd::` unconditionally (even with no SIMD ISA available,
"vector width 1" scalar add/sub is a legitimate implementation of the same
interface). The **forward pass's** own scalar fallback
(`forward/scalar.rs`) does *not* reuse `simd::scalar` — it's a completely
independent, hand-written plain-Rust reimplementation of the FT/L1/L2/L3
math, because primitives like `mul_high_i16` or `dpbusd` don't reduce to
anything a width-1 scalar type benefits from expressing generically; scalar
`forward.rs` is deliberately the simplest possible readable version and
doubles as the correctness oracle the vectorized backends must match. So
there are two different scalar strategies in one codebase, chosen per call
site by whether "vector width 1" is a meaningful specialization (accumulator
add/sub: yes) or would just be indirection for no benefit (the FT/L1/L2/L3
math: no) — a judgment call worth remembering rather than mechanically
genericizing everything through one interface.

### 3.3 Backend-specific wrinkles absorbed cleanly

- **NEON's doubling multiply.** `vqdmulhq_s16` computes `(2·a·b) >> 16`
  instead of plain `(a·b) >> 16`; rather than special-casing the algorithm,
  Reckless exposes a single per-backend constant `MUL_HI_SHIFT` (0 on
  x86, 1 on NEON, `simd/neon.rs:5`) that the *shared* `activate_ft` algorithm
  folds into its pre-shift amount — the ISA quirk is absorbed entirely inside
  the constant, the algorithm text never branches on architecture.
- **Missing stable intrinsic, raw `asm!` as the escape hatch.** ARM's `sdot`
  dot-product instruction has a nightly-only `std::arch` builtin
  (`vdotq_s32`); to get it on **stable** Rust, `simd/neon.rs:91-103` drops to
  an inline `std::arch::asm!("sdot ...")` block with `options(pure, nomem,
  nostack)`, falling back to a manually-widened multiply-and-horizontal-add
  sequence (`dot_bytes`, `simd/neon.rs:79-89,105-108`) when the `dotprod`
  target feature isn't present at all. This is a concrete, reusable pattern
  for "the instruction exists in hardware and even in nightly Rust, but not
  in stable `std::arch` yet": wrap it in a tiny `unsafe fn` behind
  `#[cfg(target_feature = "…")]` with an explicit safety-option inline-asm
  call, and give it a plain-Rust fallback for when the feature is absent.
- **Tiered fallback *within* one backend.** `simd/avx512.rs:77-100` itself
  branches on `avx512vnni` (a *further*, finer-grained feature than
  `avx512f`): if VNNI is present, `dpbusd` is one hardware instruction
  (`_mm512_dpbusd_epi32`); otherwise it's emulated with the same
  maddubs+madd ladder the AVX2 backend always uses. So even inside a single
  `#[cfg(target_feature = "avx512f")]` module there's a second layer of
  `#[cfg]` for an optional sub-feature — the abstraction nests cleanly rather
  than requiring a 6th top-level backend module just for "AVX-512 without
  VNNI."
- **wasm's `relaxed-simd` as an optional accelerant.** Every float op in
  `simd/wasm.rs` (`mul_add_f32`, `clamp_f32`, `dpbusd`, `double_dpbusd`) has a
  `#[cfg(target_feature = "relaxed-simd")]` fast path (real FMA / real dot
  instructions) and a portable fallback built from plain wasm SIMD ops when
  that proposal isn't enabled by the host — the same "prefer the fused/native
  op when available, degrade gracefully" idea applied to a *web* target
  rather than a native CPU feature.

### 3.4 Bit-exactness discipline

Multiple inline comments (`forward/scalar.rs:58-59,70-71,87-88`,
`simd/neon.rs:117-118`) explicitly call out that the *order of operations*
(FMA vs. separate mul+add, bias-as-initial-accumulator vs.
bias-added-at-the-end, the specific recursive-halving horizontal-reduction
order) is chosen **so that the scalar path is bit-identical to every
vectorized path**, not merely "close." This strongly implies (though the test
file wasn't in scope for this note) a test asserting exact equality between
backends — which is exactly the promise our own design doc already makes for
the AVX2/scalar pair (design/11 §3.4.6: "the reference the AVX2 path must
match bit-exactly in tests"). Reckless is a working demonstration of *how
hard-won* that promise actually is in practice: it requires deliberately
choosing FMA over mul+add, deliberately fixing the accumulator-init order,
and deliberately fixing the horizontal-reduction shape across 2/4/8/16-wide
backends — none of which is "the natural way to write it," all of which had
to be discovered and pinned down by someone hitting a mismatch.

## 4. What's distinctive vs Stockfish

| axis | Stockfish (current dev tree) | Reckless |
|---|---|---|
| depth after FT | `1024 → 32 → 32 → 1`, **all integer** (`u8`/`i32`) through every stage, with `SqrClippedReLU` *and* `ClippedReLU` computed and concatenated at each stage (doubling effective width), plus a 2-neuron skip connection from the first layer straight to the output (`nnue_architecture.h:56-136`) | `768 → 16 → 32 → 1`; **only the FT+first affine layer are int8**; dequantized to `f32` once and the rest is plain float, single activation (clamp, no squaring past the FT), no skip connections |
| input features | `HalfKAv2_hm` (king-bucketed piece-square) **+** `FullThreats` (added recently) | Piece-square-style set **+** a much larger dedicated Threats set (66 864 rows) — same two-feature-family idea, independently arrived at |
| sparse layer | `AffineTransformSparseInput`, `ChunkSize = 4`, dpbusd-emulation ladder identical in spirit | Same `ChunkSize = 4` / dpbusd recipe — convergent, standard approach |
| nonzero detection | movemask+LUT (non-AVX512) / hardware compress (AVX512), `NNZInfo` | Same two-tier strategy, same instructions, independently implemented |
| output buckets | 8 `LayerStacks`, each a full parameter set (like Reckless) | Same idea (8 `OUTPUT_BUCKETS`, popcount-indexed) |
| dispatch mechanism | C preprocessor: `#ifdef USE_AVX512/AVX2/…` cascades and function-like macros (`vec_add_16(a,b)` etc.) throughout `simd.h`, resolved by the chosen translation unit / multiple prebuilt binaries selected by a launcher script | Rust `#[cfg]`-gated modules + `pub use` re-export, resolved by the compiler at the same call sites — no macros, no preprocessor, ordinary function calls end-to-end |
| net provenance | ships and depends on GPL'd Stockfish-trained nets | own `RecklessNetworks` repo, `build.rs` fetches by URL at build time, `include_bytes!` + raw `transmute` — **no format validation at all** (no magic/version/checksum check; the embedded bytes are trusted to already match `Parameters`'s exact memory layout) |
| eval outer shape | NNUE PSQT/positional split feeds a "complexity" term separately from the positional score | No separate complexity signal; approximates it by scaling optimism's weight with material (`evaluation.rs`) — simpler, one less learned quantity |

The single biggest takeaway: **Reckless is shallower and does far less
integer work than Stockfish for a comparable ELO tier**, because it only
pays the quantization tax where sparsity makes it worth it (the huge 768-wide
FT/L1 stage) and gets the accuracy and simplicity of `f32` for the tiny
16/32-wide tail where the flop count is negligible anyway. Stockfish's
fully-integer, skip-connected, dual-activation tail buys it a deeper/wider net
at the same speed, but at real implementation and quantization-training cost.

## 5. What's distinctive vs our (planned) design (design/11 §3.4)

- Our v1 architecture (`768 → 256×2 → 1`, single shared hidden layer,
  concatenated stm/nstm halves dotted with one output vector) is **shallower
  than both** Reckless and Stockfish — appropriate for a first NNUE and
  exactly what the design doc's "upgrade path" flags as the eventual
  destination (deeper L1→L2→L3 tail, output buckets, king buckets). Reckless
  is direct evidence for *which* shape of "deeper" to build toward first: not
  Stockfish's fully-integer multi-stage design, but the much simpler
  int8-sparse-then-float-tail hybrid (§4) — far less quantization-training
  surface for a comparable strength gain, and a completely natural
  Rust/`f32` implementation for the tail with no SIMD work needed there at
  all beyond ordinary autovectorization.
- Our planned `QA = 255`, `QB = 64` (design/11 §3.4.3) are **exactly**
  Reckless's `FT_QUANT = 255` / `L1_QUANT = 64` (`nnue.rs:91-92`) — both
  independently landed on the same "bullet-standard" defaults. This is a
  reassuring cross-check, not a new idea: our chosen constants are exactly
  what a top-2 engine ships.
- Our dispatch plan (`LazyLock<Kernels>` of `unsafe fn` pointers,
  runtime `is_x86_feature_detected!`) is heavier than Reckless's compile-time
  `cfg`-module approach but is the right choice **for our portable default
  build** (one binary must run on whatever CPU it's launched on). Reckless
  sidesteps this because it expects to be rebuilt per target. See "Adapt to
  us" below for a concrete way to get both.
- We do not yet have a "threats" auxiliary feature set anywhere in design/11;
  both Reckless and current Stockfish now ship one. Worth a note as a future
  upgrade-path candidate once the base architecture is proven, but it's a
  meaningfully larger accumulator-maintenance project (a second incremental
  accumulator, a much bigger feature-index space) and shouldn't gate P13's
  first landing.
- Our planned net loader (design/11 §3.4.5: magic/version/dims/activation-id/
  checksum, real error type, never panics) is **more defensive** than
  Reckless's `include_bytes!` + `transmute` (no validation at all, trusted
  entirely by being produced in lockstep with the code by the same build).
  Keep ours — the extra safety is nearly free and we've already promised an
  `EvalFile` UCI option to load externally-supplied nets, which Reckless's
  approach could not safely support (or does the build.rs-fetch-and-transmute
  under the assumption the file is a trusted git-tagged release asset).

## Adapt to us

1. **Adopt the int8-sparse-FT + `f32`-tail hybrid as the target shape for the
   post-v1 "deeper" architecture**, instead of chasing Stockfish's fully
   integer multi-stage pipeline. *Touches*: design/11 §3.4.1's "upgrade path"
   paragraph (the `L1 256→512→1024` / deeper-tail step) and P13. *Effort*:
   design-only now (rewrite the upgrade-path prose + pin a `QA=255/QB=64`-style
   quantization note referencing this), real implementation effort is a
   later P13 sub-phase. *Impact*: potentially large — meaningfully less
   quantization-training complexity than a full Stockfish-style integer tail,
   for most of the strength benefit of going deeper than `768→256×2→1`.

2. **Structure our SIMD kernel dispatch as nested `cfg`-gated modules with
   `pub use` re-export (Reckless's pattern), reserved for the pinned-target
   fast build (D16's `-C target-cpu=x86-64-v3`/`native`), with the existing
   `LazyLock<Kernels>` fn-pointer struct kept only as the portable-build
   fallback.** Concretely: give `src/eval/nnue/simd.rs` a
   `#[cfg(target_feature = "avx2")]` module exactly like Reckless's, and make
   the default `KERNELS` static pick the *compile-time* module when
   `cfg(target_feature = "avx2")` already holds (skipping
   `is_x86_feature_detected!` and the indirect call entirely), falling back to
   runtime detection only in the portable default build. Our own design doc
   already anticipated exactly this escape hatch in one sentence (design/11
   §3.4.6, "the compile-time `cfg(target_feature = "avx2")` path can skip
   runtime detection") — Reckless is a full worked example of what that
   sentence looks like built out. *Touches*: design/11 §3.4.6, P13.
   *Effort*: small once the runtime-dispatch version already exists (it's an
   additive fast path, not a rewrite). *Impact*: removes one indirect call
   per forward pass in the SPRT/strength-test build; likely a small but real
   nps win, free of behavior risk since both paths must already agree
   bit-exactly.
3. **Borrow the exact bit-exactness gotchas as a pre-flight checklist when we
   write our own scalar/SIMD pair**: use `f32::mul_add` (not `a*b+c`)
   everywhere a SIMD backend will use a native FMA; initialize float
   accumulators to the bias vector rather than adding the bias at the end;
   fix the horizontal-reduction shape (recursive halving into a
   architecture-independent lane count) once and reuse it verbatim across
   backends; remember that a scalar path relying on saturating pack behavior
   (e.g. clamp-via-cast) must clamp explicitly since Rust's `as` truncates
   rather than saturates. *Touches*: design/11 §3.4.6 (add this as an
   explicit sub-bullet under "Scalar backend … is the reference the AVX2 path
   must match bit-exactly in tests"), P13. *Effort*: tiny (a paragraph in the
   design doc now; a few lines of care during implementation). *Impact*:
   avoids a class of "works on my machine, diverges by one ULP on CI"
   failures that would otherwise cost real debugging time.
4. **Keep our small-and-shared primitive-surface instinct**: when P13 lands,
   inventory exactly which vector primitives the forward pass *and* the
   accumulator-update code need (Reckless needed ~15 across both), and expose
   only those from `src/eval/nnue/simd.rs` — resist the temptation to grow a
   general-purpose SIMD wrapper library. *Touches*: design/11 §3.4.6.
   *Effort*: free (it's a design discipline, not new code). *Impact*: keeps
   adding a new backend (e.g. AVX-512 after AVX2, or wasm if we ever target
   it) a bounded, small task instead of a full port.
5. **Consider a dedicated "threats" auxiliary input accumulator** as a
   post-NNUE-v1 upgrade-path candidate, now that both Reckless and current
   Stockfish independently validate the idea. *Touches*: design/11 §3.4.1
   upgrade-path list (add a bullet next to king buckets / output buckets).
   *Effort*: large (a second incremental accumulator with its own dirty-piece
   tracking, a large feature-index space, and its own SIMD update kernels —
   comparable in scope to the whole existing accumulator subsystem).
   *Impact*: unclear without an SPRT, but it is what two independent top
   engines chose to add, which is a meaningful prior.
6. **Do not change our net-loader validation plan.** Reckless's
   `include_bytes!`+`transmute`-with-no-validation is fine *for them* because
   the net and the code are always built together from one pinned repository;
   our own magic/version/checksum loader (design/11 §3.4.5) is strictly safer
   and costs nothing at load time (once, not per-node) — no change needed,
   just confirms the existing plan is the right call given we've committed to
   supporting an external `EvalFile`.
