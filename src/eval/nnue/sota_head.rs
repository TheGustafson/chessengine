//! The SOTA net's **head** forward pass (design/11 §12.3, D40) — the
//! Reckless-shaped lean hybrid: pairwise-multiply FT activation → NNZ-sparse
//! int8 L1 → f32 L2/L3 tail → 8 output buckets, no PSQT, no output-skip.
//!
//! This is a faithful transcription of our own bit-exact reimplementation on
//! `experiment/reckless-net` (`src/eval/reckless_net/forward.rs`, matched to
//! Reckless's scalar reference over 42 positions) — our code (D27/D31; never
//! copied), which is why it moves cleanly to main. Reckless's *weights* stay
//! firewalled; nothing here loads them.
//!
//! **Feature-set agnostic.** The head consumes two per-perspective accumulators
//! (`pst` and `threat`) and knows nothing about how they were filled. Reckless
//! filled them from king-bucketed HalfKA + relational threats; our SOTA net
//! fills them from **SF's** HalfKAv2_hm + FullThreats (the front-end, task #29
//! part 2) — the head math is identical. So this module is the reusable head
//! for the whole triple-net grid (the policy heads attach elsewhere, off the
//! `pst`/`threat` trunk).
//!
//! Behind the `sota` cargo feature; the default build never compiles it, so the
//! depth-6 bench stays 548978.

use super::aligned::Blocks;
use super::sota_simd::{activate, l1_dequant, l1_dot, l2_tail, l3_dot};
use crate::types::Color;

/// Accumulator width per perspective — D47 (user-directed, 2026-07-22):
/// the shared-bottleneck multihead arch. 1536/perspective (threats retained),
/// L2=256 = the shared bottleneck where heads branch (value tail continues
/// 256→32→1×buckets; the 1858-row policy head dots the 256 — see
/// `policy_vocab`). "Go big first — learnability before NPS" is the pinned
/// research posture; the 768/L2=16 incumbent remains the fallback ship
/// (gate old nets with engines/bin/archon-sota-native.prev-era binaries).
// The arch-* features are MUTUALLY EXCLUSIVE: each pins a different (L1, L2,
// L3) triple and a CENN is only loadable by a build whose dims match it.
// Combining two used to fail with `E0428: the name L1 is defined multiple
// times`, which is true but says nothing about why. Note the L2/L3 cfg arms
// below deliberately SHARE (d47v with hka2048, d48w with d48x), so a guard
// keyed on those alone would not catch every pair.
#[cfg(any(
    all(feature = "arch-d47v", feature = "arch-d48w"),
    all(feature = "arch-d47v", feature = "arch-d48x"),
    all(feature = "arch-d47v", feature = "arch-hka2048"),
    all(feature = "arch-d48w", feature = "arch-d48x"),
    all(feature = "arch-d48w", feature = "arch-hka2048"),
    all(feature = "arch-d48x", feature = "arch-hka2048"),
))]
compile_error!(
    "the arch-* cargo features are mutually exclusive: each sets a different \
     L1/L2/L3 triple and a .cenn only loads into a build whose dims match. \
     Pick exactly one (or none, for the default 1280/128/32)."
);

#[cfg(not(any(
    feature = "arch-d47v",
    feature = "arch-d48w",
    feature = "arch-d48x",
    feature = "arch-hka2048"
)))]
pub const L1: usize = 1280;
/// D47V (2026-07-26): the distilled 768 deploy arch — see the doctrine
/// entry in the live log (train big → distill to competitive size).
#[cfg(feature = "arch-d47v")]
pub const L1: usize = 768;
/// D48W (2026-07-27): the gen-2 candidate — wide substrate, thin head
/// (the SF18 shape: accumulator capacity is amortized, head capacity is
/// per-eval). 1536/16/32 + skip; pol16/dis64 heads are TRAINER-side
/// until their engine deploy clears the <1%-cycle rule.
#[cfg(feature = "arch-d48w")]
pub const L1: usize = 1536;
/// D48X (2026-07-27): the same gen-2 recipe carried at the A3 trunk's own
/// width. The pre-launch NPS smoke measured 1536/16/32 at 804k nps — only
/// 1.11× the 1280/128 net and 0.77× the 768 deploy net, i.e. a ~45 Elo
/// equal-time speed handicap at the measured ~120 Elo/doubling. 1280 keeps
/// the trunk transfer exact (no net2net expansion) and buys the speed back.
#[cfg(feature = "arch-d48x")]
pub const L1: usize = 1280;
/// HalfKA-only probe (2026-07-27, UNRATIFIED — not a pinned decision).
/// Dropping the threat rows buys **+25% instructions (768/wl50) to +39%
/// (1280/middlegame)**, i.e. roughly 45–70 Elo of speed at the house
/// 120/doubling — perf study 7 §5, deterministic `instructions:u`, 3/3 rounds.
/// The earlier "1.78× at 1280 / 1.40× at 768" figure this comment used to
/// quote was REFUTED by that study the same morning (a 2-position
/// movetime-adjacent instrument on a loaded box, reading ~0.1–0.15 high on
/// ratio, against a stripped net that also searched a different tree); the
/// derived "width becomes 5.7× cheaper" multiplier came from the same session
/// and does not survive either, so the choice of 2048 needs re-deriving from
/// the study's grid before this arch is launched. 2048/32/32 + skip.
#[cfg(feature = "arch-hka2048")]
pub const L1: usize = 2048;
/// L1→L2 output width (the D47 shared bottleneck).
#[cfg(not(any(
    feature = "arch-d47v",
    feature = "arch-d48w",
    feature = "arch-d48x",
    feature = "arch-hka2048"
)))]
pub const L2: usize = 128;
#[cfg(any(feature = "arch-d47v", feature = "arch-hka2048"))]
pub const L2: usize = 32;
#[cfg(any(feature = "arch-d48w", feature = "arch-d48x"))]
pub const L2: usize = 16;
/// L2→L3 output width.
pub const L3: usize = 32;
/// Material/output buckets.
pub const BUCKETS: usize = 8;

/// A per-perspective pair of T6c PSQT-skip bucket sums (`[White, Black]`, like
/// the accumulators): the i32 sums of the active features' PSQ1 rows. The
/// stored rows fold in `0.5 · 400 · 256` (contract:
/// `jax/archon_jax/checkpoint.py`), so the head adds
/// `(psqt[stm][b] − psqt[ntm][b]) >> 8` centipawns — an UNCLAMPED linear
/// channel that bypasses every nonlinearity.
pub type PsqtPair = [[i32; BUCKETS]; 2];

/// The all-zero skip pair — what a net without a PSQ1 section contributes.
/// `(0 − 0) >> 8 == 0`, so passing this is bit-identical to the pre-T6c head.
pub const ZERO_PSQT: PsqtPair = [[0; BUCKETS]; 2];

pub(crate) const FT_QUANT: i32 = 255; // FT activation clamp ceiling / QA
/// The quantisation scale of the separately exported FullThreats transformer
/// that a **`CEN2` container implies**, because that header has no field for it.
///
/// **The scale is a property of the NET, not of the engine** (perf study 30).
/// A net trained WITHOUT quantisation-aware training has threat weights
/// spanning ±1.98, and 64 is simply the largest scale that does not clip them
/// (127/1.9778 = 64.2). A net trained WITH QAT at `qat_threat_scale = 256` has
/// them clipped to ±127/256 = ±0.496, and exporting THOSE at 64 throws away two
/// bits: measured on D49-sb100, `max|w| = 32/127` and 65 of 256 i8 levels used,
/// against a healthy net's 127/127 and 247/256. Hence [`CENN-v5`], which
/// carries `threat_quant` in the header; this constant is only what a
/// headerless `CEN2` file means.
///
/// [`CENN-v5`]: super::sota_loader
pub(crate) const THREAT_QUANT_CEN2: i32 = 64;
pub(crate) const L1_QUANT: i32 = 64; // L1 weight scale / QB
pub(crate) const FT_SHIFT: i32 = 9; // pairwise-multiply right shift

/// The threat→HalfKA frame shift a container's `threat_quant` implies, or
/// `None` if the value is not one this runtime can carry as a shift.
///
/// The merge must rescale the threat accumulator by `FT_QUANT / threat_quant`
/// before summing it into the HalfKA frame. Since perf study 8 §4.6 that ratio
/// is carried as the nearest POWER OF TWO — `256 / threat_quant`, i.e. a single
/// `slli` instead of a six-op `round(t · 255/q)` — which is a 0.39% gain change
/// on one eval channel, measured at mean 0.14 SF cp over the 800-FEN battery
/// and well inside the net's own quantisation noise. The approximation is the
/// same size in either direction: 255/64 = 3.984 → 4, and 255/256 = 0.996 → 1.
///
/// So `threat_quant` must be a power of two, and the shift is
/// `log2(256 / threat_quant)`. `256 → 0`, `128 → 1`, `64 → 2` (every CEN2 net).
///
/// **The accepted set is exactly `{64, 128, 256}`** ([`MAX_THREAT_SHIFT`]), and
/// that is a PERFORMANCE contract, not a taste. The merge shift has to be an
/// IMMEDIATE: `vpsllw $0x2,(%rcx,%r8,2),%zmm3` folds the accumulator load into
/// the shift, while the register-count form `vpsllw %xmm0,%zmm4,%zmm4` cannot
/// — `VPSLLW`'s memory operand is the COUNT, not the data — so it needs a
/// separate `vmovdqu64` per merge, measured at **+34.48 instructions per node,
/// +0.61%** (perf study 30 §5.3). The backends therefore monomorphise the
/// kernel over the shift and dispatch once per eval, which only stays cheap
/// while the set is small. Widening it is a one-line change here plus one more
/// `match` arm per backend — but nothing coarser than 64 is ever wanted (it
/// would quantise the threat table *worse* than today) and nothing finer than
/// 256 fits an i8, so three is the whole real domain: 64 legacy, 128 the
/// measured non-QAT optimum (`tools/nnue/requant_threats.py`), 256 the QAT
/// scale.
pub(crate) const fn threat_shift_for(threat_quant: i32) -> Option<u32> {
    if threat_quant <= 0 || threat_quant > 256 || !(threat_quant as u32).is_power_of_two() {
        return None;
    }
    let shift = 8 - (threat_quant as u32).trailing_zeros();
    if shift > MAX_THREAT_SHIFT {
        return None;
    }
    Some(shift)
}

/// The largest merge shift the SIMD backends are monomorphised for — the
/// invariant `sota_simd`'s `activate` dispatch relies on to stay exhaustive
/// without a panic arm. See [`threat_shift_for`].
pub(crate) const MAX_THREAT_SHIFT: u32 = 2;

/// The `threat_quant` a merge shift came from — the exact inverse of
/// [`threat_shift_for`] over its whole domain, so the loader and the serializer
/// cannot disagree about which scale a net is in.
pub(crate) const fn threat_quant_for(shift: u32) -> i32 {
    256 >> shift
}
/// `(1 << FT_SHIFT) / (FT_QUANT² · L1_QUANT)` — the i32→f32 bridge at L1
/// (design/11 §12.6).
const DEQUANT: f32 = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT * L1_QUANT) as f32;

/// Piece-count → output bucket (non-uniform: fine at mid-material, coarse at the
/// extremes — copy Reckless's *shape*, retune boundaries on our data later).
const OUTPUT_BUCKETS_LAYOUT: [usize; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 0..8 pieces
    1, 1, 1, 1, // 9..12
    2, 2, 2, 2, // 13..16
    3, 3, 3, // 17..19
    4, 4, 4, // 20..22
    5, 5, 5, // 23..25
    6, 6, 6, // 26..28
    7, 7, 7, 7, // 29..32
];

/// The output bucket the head selects for a total piece count — exposed so
/// tests, tools, and the debug oracle can pin the T6c skip against the SAME
/// bucket index the forward uses.
#[inline]
pub fn output_bucket(piece_count: usize) -> usize {
    OUTPUT_BUCKETS_LAYOUT[piece_count]
}

// The head's five bucket-selected weight tables, and the skip lane, named as
// what they are: `BUCKETS` fixed-size blocks. Perf study 26 §3 — as
// `Aligned<T>` each acquisition re-derived a fat pointer, range-checked it and
// then `try_into`-length-checked it, five times per eval, to recover a shape
// that has been fixed since load. `Blocks` states it in the type, so the only
// runtime check left is `bucket < BUCKETS` against a constant, CSE'd across the
// whole forward pass.
//
// Every `N · size_of::<T>()` below is a multiple of 64 at every shipped arch,
// which `Blocks::LAYOUT_OK` asserts at compile time — so the reshape adds no
// padding and the on-disk block-major layout is unchanged.
/// `[BUCKETS][L2 · L1]` i8 — the sparse L1 matmul's weights.
pub type L1Weights = Blocks<i8, { L2 * L1 }, BUCKETS>;
/// `[BUCKETS][L2]` f32.
pub type L1Biases = Blocks<f32, L2, BUCKETS>;
/// `[BUCKETS][L2 · L3]` f32.
pub type L2Weights = Blocks<f32, { L2 * L3 }, BUCKETS>;
/// `[BUCKETS][L3]` f32.
pub type L2Biases = Blocks<f32, L3, BUCKETS>;
/// `[BUCKETS][L3]` f32.
pub type L3Weights = Blocks<f32, L3, BUCKETS>;
/// `[BUCKETS][L1]` i8 — the D47V output-skip lane.
pub type SkipQ = Blocks<i8, L1, BUCKETS>;

/// The D47V output-skip lane, quantised to i8 at load (perf study 8 §4.7).
///
/// `w[b][i] ≈ q[b · L1 + i] · scale[b]`, with a per-bucket scale so a bucket
/// whose weights are small keeps its full i8 resolution. Quantising costs
/// accuracy — measured, not assumed, on the 800-FEN battery — and buys a
/// 6x cheaper lane: `vpdpbusd` consumes 64 u8 x i8 pairs per instruction where
/// the f32 form needed a widen, a convert and an FMA per 16.
#[derive(Debug)]
pub struct SkipLane {
    /// `[BUCKETS][L1]` i8, bucket-major contiguous.
    pub q: SkipQ,
    /// Per-bucket dequantisation scale, `[BUCKETS]`.
    pub scale: Box<[f32]>,
}

/// The head's per-bucket weights + the net's output scale. Format-agnostic: the
/// CENN-v2 loader (task #29) fills these from disk; tests fill them synthetically.
/// Layouts are bucket-major, flattened.
#[derive(Debug)]
pub struct SotaHead {
    /// `[BUCKETS][L2 · L1]` i8.
    pub l1_weights: L1Weights,
    /// `[BUCKETS][L2]` f32.
    pub l1_biases: L1Biases,
    /// `[BUCKETS][L2 · L3]` f32.
    pub l2_weights: L2Weights,
    /// `[BUCKETS][L3]` f32.
    pub l2_biases: L2Biases,
    /// `[BUCKETS][L3]` f32.
    pub l3_weights: L3Weights,
    /// `[BUCKETS]` f32.
    pub l3_biases: Box<[f32]>,
    /// f32→centipawn scale (Reckless 380; ours is baked at training, per-net).
    pub network_scale: i32,
    /// Left shift applied to the threat accumulator before it merges into the
    /// HalfKA frame — [`threat_shift_for`] of the container's `threat_quant`.
    ///
    /// Lives here, beside `network_scale`, for the same reason that one does:
    /// both are per-net SCALARS the forward pass reads, not weights, and
    /// `forward_into` is the only caller that has to reach them. A `CEN2` net
    /// yields 2 — the literal every SIMD backend used to hardcode — so a net
    /// loaded from a v2 container evaluates bit-identically to before
    /// (perf study 30).
    pub threat_shift: u32,
    /// EXP-UNC-1 (search study bet 1): the OPTIONAL heteroscedastic log-variance
    /// head — a parallel `[BUCKETS][L3]` weight block + `[BUCKETS]` bias trained
    /// under Gaussian NLL alongside the value mean. `None` for every net without
    /// the head (all existing nets), in which case [`forward`](Self::forward)
    /// returns `sigma = None` and every uncertainty consumer falls back to its
    /// constant — so a variance-less net is bit-identical to today. Loaded from a
    /// small sidecar (`D40_UNC_HEAD`), not the CENN, so the CENN format is
    /// unchanged.
    pub l3_var_weights: Option<Box<[f32]>>,
    /// `[BUCKETS]` f32, paired with [`l3_var_weights`](Self::l3_var_weights).
    pub l3_var_biases: Option<Box<[f32]>>,
    /// D41 rich-net: OPTIONAL co-trained WDL head — a bucket-agnostic `[3·L3]` weight
    /// block + `[3]` biases (W/D/L logits off the same bucket-selected L3 activation
    /// the value head reads), trained additively (softmax-CE on the game result).
    /// `None` for nets without it (⇒ [`forward`](Self::forward) returns `wdl = None`,
    /// consumers fall back to their constants — bit-identical to today). Loaded from a
    /// `D40_WDL_HEAD` sidecar, not the CENN, so the CENN + a WDL-less net stay
    /// bit-identical. Layout: row-major `[o·L3 + j]`, o ∈ {L,D,W}.
    pub l3_wdl_weights: Option<Box<[f32]>>,
    /// `[3]` f32 biases paired with [`l3_wdl_weights`](Self::l3_wdl_weights).
    pub l3_wdl_biases: Option<Box<[f32]>>,
    /// D47V output-skip lane, QUANTISED TO i8 at load ([`SkipLane`]).
    /// `None` = pre-skip net: the eval path is bit-identical to before.
    pub skip_weights: Option<SkipLane>,
}

/// Pairwise-multiply FT activation (design/11 §12.3 step 1): sum the two
/// accumulators, clamp each half to `[0, FT_QUANT]`, multiply first×second half
/// and `>> FT_SHIFT` → u8 ∈ [0,127]. STM half occupies `[0, L1/2)`. The
/// `wrapping_add` reproduces the reference's wrapping i16 accumulator math.
/// Out-param form (perf study #4 item #5): the kernel writes the caller's
/// buffer directly instead of returning the 1280 B array by value.
fn activate_ft(
    pst: &[[i16; L1]; 2],
    threat: &[[i16; L1]; 2],
    stm: Color,
    threat_shift: u32,
    out: &mut [u8; L1],
) {
    activate(out, pst, threat, stm as usize, threat_shift);
}

/// Float view of the quantized shared D40 trunk for the optional policy head.
/// The trainer's pairwise CReLU produces values in `[0, 1]`; the runtime FT
/// stores the same product after `>> FT_SHIFT`, so undo that scale here.  This
/// deliberately shares [`activate_ft`] with eval to prevent the two consumers
/// from drifting on perspective order, threat scaling, or clamp semantics.
#[cfg(feature = "policy")]
pub fn policy_trunk(
    pst: &[[i16; L1]; 2],
    threat: &[[i16; L1]; 2],
    stm: Color,
    threat_shift: u32,
) -> [f32; L1] {
    let mut quantized = [0u8; L1];
    activate_ft(pst, threat, stm, threat_shift, &mut quantized);
    let scale = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32;
    quantized.map(|value| f32::from(value) * scale)
}

/// Convert the independently quantized threat accumulator to the HalfKA scale
/// before summing. This is `round(pst + threat · FT_QUANT/threat_quant)` in
/// quantized space. Adding the raw integers would underweight every threat
/// feature by the frame ratio (4x on a CEN2 net).
#[cfg(test)]
#[inline]
fn merge_transformers(pst: i16, threat: i16, threat_shift: u32) -> i16 {
    // One shift: the frame ratio is carried as a power of two (a 0.39% gain
    // change on the threat channel), which is what every SIMD backend emits
    // and what the 800-FEN fidelity check measured at mean 0.14 SF cp
    // (study 8 §4.6; see `threat_shift_for`).
    let scaled = i32::from(threat) << threat_shift;
    (i32::from(pst) + scaled).clamp(0, FT_QUANT) as i16
}

/// Indices of the 4-byte groups (of `L1/4`) holding any nonzero activation
/// (design/11 §12.3 step 2) — ~85–95% are zero after the clamp. Dispatched:
/// the scalar scan's 192 data-dependent branches mispredict on the sparse
/// pattern, so the SIMD backends compact via movemask instead (R4 #2).
/// Out-param form (perf study #4 item #5): no 640 B by-value return.
fn find_nnz(ft: &[u8; L1], idx: &mut [u16; L1 / 4]) -> usize {
    super::sota_simd::find_nnz(ft, idx)
}

/// Reusable per-thread scratch for [`SotaHead::forward_into`] — the activated
/// feature vector and the non-zero group list its sparse L1 matmul walks.
///
/// **This is a caller-owned buffer, not two locals, for a measured reason**
/// (perf study 15 arm 4). Declared inside `forward_into` as
/// `let mut ft = [0u8; L1]`, the array's zero-initialiser compiled to a
/// **1280-byte libc `memset` call at every evaluation** — `mov $0x500,%edx;
/// xor %esi,%esi; call memset` as the third instruction of the function — even
/// though the very next statement (`activate_ft`) overwrites every lane. LLVM
/// cannot dead-store-eliminate it, because `activate_ft` is an indirect
/// `kernel_dispatch!` call and it therefore cannot prove the buffer is fully
/// written. `nnz` had the same shape at 640 bytes, and it genuinely IS only
/// partly written (only `..count` is ever read) — which is why the fix is a
/// long-lived buffer rather than a `MaybeUninit` + `unsafe` dance: a buffer
/// that is never re-initialised has nothing left to eliminate, and it needs no
/// safety argument. Same pattern, and the same reason, as
/// `ThreatContext::add_scratch`.
///
/// Stale bytes are never read: `activate_ft` writes all `L1` lanes of `ft`
/// before anything reads it, and `nnz` is consumed as `&nnz[..count]` with the
/// `count` `find_nnz` just returned.
/// Cache-line aligned like `Entry`/`FinnyEntry`/`ThreatEntry`, and like both
/// references' equivalents — Stockfish's `alignas(CacheLineSize)
/// transformedFeatures` (`eca43a97:src/nnue/network.cpp:144-148`) and
/// Reckless's `Aligned<T>` wrapper. Measured −0.6 i/n, which is 300x this
/// study's rebuild null but small; it is kept on the reference convergence and
/// on `sota_simd.rs:232`'s `read_unaligned` (whose comment says "`ft` is only
/// byte-aligned" and no longer has to be), not on its instruction number.
/// Alignment is a cycles lever, and cycles are not this study's metric.
#[repr(align(64))]
pub struct HeadScratch {
    ft: [u8; L1],
    nnz: [u16; L1 / 4],
}

impl HeadScratch {
    pub fn new() -> Self {
        HeadScratch {
            ft: [0u8; L1],
            nnz: [0u16; L1 / 4],
        }
    }

    /// A scratch filled with values no forward pass can legitimately leave
    /// behind — the test-only stand-in for "the previous position's buffer".
    ///
    /// The whole safety argument for reusing this buffer is that `activate_ft`
    /// writes every one of `L1` lanes before anything reads them. While `ft`
    /// was a fresh `[0u8; L1]` local, a lane a kernel failed to write was
    /// zero — and ~85-95% of activations ARE zero after the clamp, so the bug
    /// was invisible. Reused, that lane is the previous position's activation
    /// and the eval becomes silently history-dependent.
    #[cfg(test)]
    fn poisoned() -> Self {
        HeadScratch {
            ft: [0xABu8; L1],
            nnz: [0xFFFFu16; L1 / 4],
        }
    }
}

impl Default for HeadScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl SotaHead {
    /// int8 sparse matmul over the nonzero groups → dequantized SCReLU f32[L2]
    /// (design/11 §12.3 step 3). Products accumulate in i32, matching Bullet's
    /// affine layer and the `dpbusd` semantics used by the reference design.
    /// Out-param form (perf study #4 item #5): writes the caller's L2 buffer —
    /// which for the policy stash is the context's inline slot — instead of
    /// returning 512 B by value.
    fn propagate_l1(&self, ft: &[u8; L1], nnz: &[u16], bucket: usize, out: &mut [f32; L2]) {
        let mut pre = [0i32; L2];
        let wb = self.l1_weights.block(bucket);
        l1_dot(&mut pre, ft, nnz, wb);
        let bias = self.l1_biases.block(bucket);
        // Explicit dequant+SCReLU kernel (perf study #4 item #2a); the scalar
        // backend is the loop that used to live here — bit-identical.
        l1_dequant(out, &pre, bias, DEQUANT);
    }

    /// Dense f32[L2]→f32[L3], bias-seeded FMA then SCReLU (step 4). The
    /// i-outer/j-inner order is load-bearing: it pins the accumulation order.
    /// Now an explicit dispatched kernel (perf study #4 item #2a — at D47
    /// dims this is 4096 FMAs/eval, the last autovectorized hot path; the
    /// AVX-512 backend holds the whole L3 row in two zmm accumulators with
    /// one broadcast + two FMAs per i). The scalar backend is the loop that
    /// used to live here — j-lane order and per-lane i-sequential
    /// accumulation preserved ⇒ bit-identical.
    fn propagate_l2(&self, l1: &[f32; L2], bucket: usize) -> [f32; L3] {
        let mut out = [0.0f32; L3];
        let bias = self.l2_biases.block(bucket);
        let wblock = self.l2_weights.block(bucket);
        l2_tail(&mut out, l1, wblock, bias);
        out
    }

    /// Dense f32[L3]→f32, 16-lane FMA + recursive-halving reduction + bias
    /// (step 5). Reduction order mirrors a 16-lane horizontal add.
    fn propagate_l3(&self, l2: &[f32; L3], bucket: usize) -> f32 {
        // Dispatched since perf study 26 arm 3 — the loop that used to live
        // here is now `sota_simd::scalar::l3_dot`, unchanged, and is the
        // bit-exact reference the SIMD backends match. It was the last
        // autovectorized f32 path in the head, and LLVM would not widen it:
        // for a 32-element dot it emitted sixteen 2-lane `vfmadd213ps` fed by
        // sixteen `vmovsd` pair-loads — 4.56% of `forward_into`'s executed
        // instructions.
        l3_dot(self.l3_weights.block(bucket), l2) + self.l3_biases[bucket]
    }

    /// EXP-UNC-1: the log-variance head's dense f32[L3]→f32, mirroring
    /// [`propagate_l3`](Self::propagate_l3) exactly (same lane layout + reduction
    /// order) but reading the variance weights. Returns raw `s = log(σ²)`, the
    /// log-variance of the SIGMOID (WDL) value — the NLL trains it on the
    /// `(sigmoid(out) − target)` residual, so σ̂ = exp(½·s) is a WDL-space std,
    /// NOT a cp-space one (D33 review, reviewer B: mind this when it scales the
    /// cp-space RFP margin). Caller guarantees the variance head is present.
    // `#[cold] #[inline(never)]`: NO SHIPPED NET HAS EITHER OF THESE HEADS.
    // `l3_var_weights` and `l3_wdl_weights` are `None` for every net in
    // `engines/`, so both bodies are unreachable at run time — but inlined into
    // `forward_into` they contributed ~140 static instructions to the hottest
    // eval function, including a 31-deep chain of SCALAR `vfmadd231ss` and 32
    // `lea`/`cmp`/`jae` bounds-check triples on the runtime-length
    // `Box<[f32]>` (perf study 26 §5, measured on a line-attributed profile).
    // Same treatment, and the same reason, as `i16_deltas_into_generic`.
    #[cold]
    #[inline(never)]
    fn propagate_l3_var(&self, l2: &[f32; L3], bucket: usize) -> f32 {
        const LANES: usize = 16;
        let mut sums = [0.0f32; LANES];
        let wbase = bucket * L3;
        let vw = self.l3_var_weights.as_ref().expect("variance head present");
        let vb = self.l3_var_biases.as_ref().expect("variance head present");
        let weights: &[f32; L3] = vw[wbase..wbase + L3]
            .try_into()
            .expect("l3 var weight block");
        let mut i = 0;
        while i < L3 {
            let wl: &[f32; LANES] = weights[i..i + LANES].try_into().expect("l3 var lane");
            let ll: &[f32; LANES] = l2[i..i + LANES].try_into().expect("l2 lane");
            for j in 0..LANES {
                sums[j] = wl[j].mul_add(ll[j], sums[j]);
            }
            i += LANES;
        }
        let mut stride = LANES / 2;
        while stride > 0 {
            for k in 0..stride {
                sums[k] += sums[k + stride];
            }
            stride /= 2;
        }
        sums[0] + vb[bucket]
    }

    /// D41 rich-net WDL head: 3 raw logits (L,D,W) off the same bucket-selected L3
    /// activation the value head reads. Bucket-agnostic weights `[3·L3]` row-major
    /// (`w[o·L3 + j]`) + `[3]` biases — mirrors the trainer's `l3_wdl` affine
    /// (`L3 → 3`). Softmax is left to the consumer. Caller guarantees the head is
    /// present.
    #[cold]
    #[inline(never)]
    fn propagate_l3_wdl(&self, l2: &[f32; L3]) -> [f32; 3] {
        let ww = self.l3_wdl_weights.as_ref().expect("wdl head present");
        let wb = self.l3_wdl_biases.as_ref().expect("wdl head present");
        let mut out = [0.0f32; 3];
        for (o, slot) in out.iter_mut().enumerate() {
            let base = o * L3;
            let mut s = 0.0f32;
            for j in 0..L3 {
                s = ww[base + j].mul_add(l2[j], s);
            }
            *slot = s + wb[o];
        }
        out
    }

    /// The head forward: two per-perspective accumulators + side to move + total
    /// piece count → (stm-relative centipawns, optional value-uncertainty σ̂).
    /// The front-end (task #29 part 2) produces `pst`/`threat`; this is the
    /// shared trunk read for both eval and the policy heads. `psqt` is the T6c
    /// skip pair the front-end accumulates alongside ([`ZERO_PSQT`] for a net
    /// without one — bit-identical). `sigma` is `Some`
    /// only when the net carries the EXP-UNC-1 log-variance head; it is the
    /// per-position value std in sigmoid (WDL) space, `σ̂ = exp(½·log σ²)`,
    /// computed from the SAME `l2` activations at ~32 extra FLOPs (T ≈ 1).
    pub fn forward(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
        psqt: &PsqtPair,
    ) -> (i32, Option<f32>, Option<[f32; 3]>) {
        // Convenience wrapper: builds its own scratch, so it pays the 1920 B of
        // zero-init that [`HeadScratch`] exists to remove. That is fine for the
        // tests and for the smallnet/largenet tiers that call it, which cannot
        // borrow `SotaContext::head_scratch` anyway — their `current_full` holds
        // a `&mut` to the side stack across the call (D33 reviewer B, M3). The
        // hot path calls `forward_into` directly.
        let mut l1 = [0.0f32; L2];
        let mut scratch = HeadScratch::new();
        self.forward_into(pst, threat, stm, piece_count, psqt, &mut scratch, &mut l1)
    }

    /// [`forward`](Self::forward), writing the shared-bottleneck activation
    /// (the L2-wide output the D47 policy head dots against) into `l1_out` —
    /// an out-param so the policy stash lands directly in the caller's slot
    /// with no by-value copy and no discarded return on the `forward` path
    /// (perf study #4 item #5; the pre-restructure `forward` called
    /// `forward_full(...).0` and threw the 512 B array away).
    ///
    /// `scratch` is the caller's long-lived [`HeadScratch`] — see that type for
    /// why it is a parameter and not two locals.
    // Eight arguments, one over clippy's default: they are the forward pass's
    // genuinely independent inputs (two accumulators, three position facts, and
    // two caller-owned output buffers). Bundling them into a struct would move
    // the same words without adding a fact. NOTE: `tools/check.sh` clippies
    // DEFAULT features only, so no gate sees this lint either way — see the
    // study-15 handoff.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
        psqt: &PsqtPair,
        scratch: &mut HeadScratch,
        l1_out: &mut [f32; L2],
    ) -> (i32, Option<f32>, Option<[f32; 3]>) {
        // Destructure rather than `scratch.ft` / `scratch.nnz` at each use:
        // `find_nnz(&scratch.ft, &mut scratch.nnz)` needs a shared and a unique
        // borrow of two DIFFERENT fields at once, and a field-by-field
        // destructuring pattern is what makes that legal without `unsafe` —
        // the same idiom `HalfkaContext::finny_refresh_into` uses on `self`.
        #[cfg(feature = "foldcount")]
        super::foldcount::EVAL.hit(0);
        let HeadScratch { ft, nnz } = scratch;
        activate_ft(pst, threat, stm, self.threat_shift, ft);
        let count = find_nnz(ft, nnz);
        let bucket = OUTPUT_BUCKETS_LAYOUT[piece_count];
        self.propagate_l1(ft, &nnz[..count], bucket, l1_out);
        let l1 = l1_out;
        let l2 = self.propagate_l2(l1, bucket);
        let l3 = self.propagate_l3(&l2, bucket);
        // T6c PSQT skip: the unclamped linear channel joins AFTER the clamped
        // tail, straight in centipawns — `(stm − ntm) >> 8` (arithmetic shift;
        // the rows fold in 0.5·400·256, see `PsqtPair`). Zero pair ⇒ +0 ⇒
        // bit-identical for legacy nets.
        // D47V output-skip: a learned linear read of the SAME activated
        // pairwise features the L1 matmul consumes, added in LOGIT units
        // before the cp scale — mirroring SF's fc_0 +1 lane. Dequant reuses
        // policy_trunk's proven factor: ft_u8 = hl1_f · FT_QUANT²/2^FT_SHIFT.
        let l3 = if let Some(sk) = &self.skip_weights {
            // §6-S/S2a/S2 (perf study): branchless, bucket-major contiguous
            // lane (the loader transposes at parse).
            //
            // INTEGER since perf study 8 §4.7. The f32 form was a dense
            // 1280-lane `vpmovzxbd`/`vcvtdq2ps`/`vfmadd` chain — three
            // instructions per 16 elements, ~240 per eval — for the one lane
            // Stockfish gets free as the `FC_0_OUTPUTS + 1`-th row of the
            // sparse int8 matmul it already runs. We cannot fold it into
            // `l1_dot` without a 17th output row (a second, near-empty `zmm`
            // accumulator, which doubles that kernel), but we CAN pay for it
            // in the same currency: `vpdpbusd` over u8 x i8 does 64 lanes per
            // instruction, so 1280 lanes cost 20 loads and 20 multiply-adds.
            let lane = sk.q.block(bucket);
            let acc = super::sota_simd::skip_dot_i8(&ft[..], lane) as f32 * sk.scale[bucket];
            l3 + acc * ((1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32)
        } else {
            l3
        };
        let stm_i = stm as usize;
        let value = (l3 * self.network_scale as f32) as i32
            + ((psqt[stm_i][bucket] - psqt[stm_i ^ 1][bucket]) >> 8);
        // σ̂ = exp(½·logσ²). Guard against a non-finite result (only reachable
        // via a garbage/mismatched D40_UNC_HEAD sidecar, which is size- but not
        // content-validated): return None so every consumer falls back to its
        // factor-1.0 no-op instead of NaN-poisoning the RFP margin / TM budget
        // (D33 review, reviewer A).
        let sigma = self.l3_var_weights.as_ref().and_then(|_| {
            let s = (0.5 * self.propagate_l3_var(&l2, bucket)).exp();
            s.is_finite().then_some(s)
        });
        // D41 rich-net: WDL probabilities [L,D,W] (softmax of the 3 logits). `None`
        // if no head or a non-finite result (garbage sidecar) → consumers fall back
        // to their no-op, same guard rationale as `sigma`.
        let wdl = self.l3_wdl_weights.as_ref().and_then(|_| {
            let logits = self.propagate_l3_wdl(&l2);
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = [0.0f32; 3];
            let mut sum = 0.0f32;
            for (p, &z) in probs.iter_mut().zip(logits.iter()) {
                *p = (z - m).exp();
                sum += *p;
            }
            probs.iter_mut().for_each(|p| *p /= sum);
            probs.iter().all(|p| p.is_finite()).then_some(probs)
        });
        (value, sigma, wdl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perf study 15 arm 4 made `ft`/`nnz` a REUSED buffer, so nothing in the
    /// forward pass may depend on their prior contents. A poisoned scratch must
    /// produce the same value, sigma and WDL as a fresh one — for both sides to
    /// move, and for a bucket at each end of the piece-count range.
    ///
    /// Mutation it catches: any `activate` backend that writes fewer than `L1`
    /// lanes (a dropped `chunks_exact` remainder, a width assumption), or a
    /// `find_nnz` whose returned count under-covers the groups it wrote.
    /// Nothing else in the tree can catch that — the cross-kernel test in
    /// `sota_simd` compares two buffers that are also both poisoned, but only
    /// this one closes the loop through the whole head.
    #[test]
    fn a_poisoned_scratch_gives_the_same_forward_as_a_fresh_one() {
        let head = synth_head();
        let (pst, threat) = synth_accs();

        for stm in [Color::White, Color::Black] {
            for piece_count in [4usize, 20, 32] {
                let mut fresh = HeadScratch::new();
                let mut dirty = HeadScratch::poisoned();
                let mut l1_a = [0.0f32; L2];
                let mut l1_b = [0.0f32; L2];
                let a = head.forward_into(
                    &pst,
                    &threat,
                    stm,
                    piece_count,
                    &ZERO_PSQT,
                    &mut fresh,
                    &mut l1_a,
                );
                let b = head.forward_into(
                    &pst,
                    &threat,
                    stm,
                    piece_count,
                    &ZERO_PSQT,
                    &mut dirty,
                    &mut l1_b,
                );
                assert_eq!(a.0, b.0, "value depends on scratch contents");
                assert_eq!(a.1, b.1, "sigma depends on scratch contents");
                assert_eq!(a.2, b.2, "wdl depends on scratch contents");
                assert_eq!(l1_a, l1_b, "the L2 trunk depends on scratch contents");
            }
        }

        // …and the buffer really was dirty on entry, so the assertions above
        // are not vacuous.
        assert_eq!(HeadScratch::poisoned().ft[0], 0xAB);
        assert_ne!(HeadScratch::poisoned().ft, HeadScratch::new().ft);
    }

    #[test]
    fn mixed_transformer_scales_are_normalized_before_sum() {
        // A CEN2 net: threat_quant 64 ⇒ shift 2.
        assert_eq!(merge_transformers(0, 64, 2), 255);
        assert_eq!(merge_transformers(255, -64, 2), 0);
        assert_eq!(merge_transformers(100, 16, 2), 164);
        assert_eq!(merge_transformers(100, -16, 2), 36);
        assert_eq!(merge_transformers(250, 2, 2), 255);
        assert_eq!(merge_transformers(5, -2, 2), 0);
    }

    /// The shift is the net's, not the engine's: a QAT net at threat_quant 256
    /// merges the threat accumulator UNSCALED (shift 0), and 128 halves it.
    #[test]
    fn threat_shift_tracks_the_container_quant() {
        assert_eq!(threat_shift_for(64), Some(2));
        assert_eq!(threat_shift_for(128), Some(1));
        assert_eq!(threat_shift_for(256), Some(0));
        // Not representable as a power-of-two frame ratio.
        assert_eq!(threat_shift_for(255), None);
        assert_eq!(threat_shift_for(512), None);
        assert_eq!(threat_shift_for(0), None);
        assert_eq!(threat_shift_for(-64), None);
        // Representable, but coarser than 64 and therefore outside the set the
        // backends monomorphise — the `_` arm of their `activate` dispatch is
        // only sound because this is rejected here.
        assert_eq!(threat_shift_for(32), None);
        assert_eq!(threat_shift_for(1), None);
        // …and the inverse round-trips over the whole accepted domain.
        for q in [64, 128, 256] {
            assert_eq!(threat_quant_for(threat_shift_for(q).unwrap()), q);
        }
        assert_eq!(
            threat_shift_for(threat_quant_for(MAX_THREAT_SHIFT)),
            Some(MAX_THREAT_SHIFT)
        );

        assert_eq!(merge_transformers(100, 16, 0), 116);
        assert_eq!(merge_transformers(100, 16, 1), 132);
        assert_eq!(merge_transformers(100, 16, 2), 164);
    }

    #[test]
    fn dense_tail_uses_squared_clipped_relu() {
        let mut head = synth_head();
        head.l1_weights.fill(0);
        head.l1_biases.fill(0.5);
        let mut l1 = [0.0f32; L2];
        head.propagate_l1(&[0; L1], &[], 0, &mut l1);
        assert_eq!(l1, [0.25; L2]);

        head.l2_weights.fill(0.0);
        head.l2_biases.fill(0.5);
        let l2 = head.propagate_l2(&l1, 0);
        assert_eq!(l2, [0.25; L3]);
    }

    /// A deterministic synthetic head + inputs (no Reckless weights — this is a
    /// transcription regression guard, not a strength net).
    fn synth_head() -> SotaHead {
        let l1w: Vec<i8> = (0..BUCKETS * L2 * L1)
            .map(|i| ((i as i32 * 7 % 61) - 30) as i8)
            .collect();
        let l1b: Vec<f32> = (0..BUCKETS * L2).map(|i| (i % 5) as f32 * 0.01).collect();
        let l2w: Vec<f32> = (0..BUCKETS * L2 * L3)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let l2b: Vec<f32> = (0..BUCKETS * L3).map(|i| (i % 7) as f32 * 0.01).collect();
        let l3w: Vec<f32> = (0..BUCKETS * L3)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let l3b: Vec<f32> = (0..BUCKETS).map(|i| i as f32 * 0.1).collect();
        SotaHead {
            l1_weights: l1w.into(),
            l1_biases: l1b.into(),
            l2_weights: l2w.into(),
            l2_biases: l2b.into(),
            l3_weights: l3w.into(),
            l3_biases: l3b.into_boxed_slice(),
            network_scale: 380,
            threat_shift: 2,
            l3_var_weights: None,
            l3_var_biases: None,
            l3_wdl_weights: None,
            l3_wdl_biases: None,
            skip_weights: None,
        }
    }

    fn synth_accs() -> ([[i16; L1]; 2], [[i16; L1]; 2]) {
        let mut pst = [[0i16; L1]; 2];
        let mut threat = [[0i16; L1]; 2];
        // Asymmetric perspectives (the `+ p*k` terms) so an stm flip actually
        // swaps distinct halves — otherwise stm-relativity is untestable.
        for p in 0..2 {
            for i in 0..L1 {
                pst[p][i] = (((i as i32 * 3 + p as i32 * 17) % 200) - 40) as i16;
                threat[p][i] = (((i as i32 * 5 + p as i32 * 11) % 120) - 30) as i16;
            }
        }
        (pst, threat)
    }

    /// `activate_ft` matches the hand-computed pairwise-multiply on known values,
    /// covering the clamp and the shift.
    #[test]
    fn activate_ft_hand_computed() {
        let mut pst = [[0i16; L1]; 2];
        let mut threat = [[0i16; L1]; 2];
        // index 0 (STM half low): left = 150, right clamps to 255.
        pst[0][0] = 150;
        const H: usize = L1 / 2;
        pst[0][H] = 200;
        threat[0][H] = 16; // rescaled to 64, so right sum = 264 → clamp 255
        let mut ft = [0u8; L1];
        activate_ft(&pst, &threat, Color::White, 2, &mut ft);
        // (150 * 255) >> 9 = 38250 >> 9 = 74
        assert_eq!(ft[0], 74, "pairwise clamp+shift");
        // a negative sum clamps to 0 → product 0
        let mut pst2 = [[0i16; L1]; 2];
        pst2[0][1] = -500;
        let mut ft2 = [0u8; L1];
        activate_ft(&pst2, &threat, Color::White, 2, &mut ft2);
        assert_eq!(ft2[1], 0, "negative clamps to 0");
    }

    /// The sparse L1 path equals a dense recomputation over ALL groups (zeros
    /// contribute nothing) — catches any `find_nnz` / sparse-indexing slip.
    #[test]
    fn sparse_l1_equals_dense() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let mut ft = [0u8; L1];
        activate_ft(&pst, &threat, Color::White, 2, &mut ft);
        let mut nnz = [0u16; L1 / 4];
        let count = find_nnz(&ft, &mut nnz);
        let bucket = 3;
        let mut sparse = [0.0f32; L2];
        head.propagate_l1(&ft, &nnz[..count], bucket, &mut sparse);
        // Dense: iterate every group with the same mathematical i32 dot.
        let mut pre = [0i32; L2];
        let wb = head.l1_weights.block(bucket);
        for g in 0..L1 / 4 {
            for j in 0..L2 {
                let mut v = 0i32;
                for k in 0..4 {
                    v += i32::from(ft[g * 4 + k]) * i32::from(wb[g * (L2 * 4) + j * 4 + k]);
                }
                pre[j] += v;
            }
        }
        let mut dense = [0.0f32; L2];
        for j in 0..L2 {
            let clipped = (pre[j] as f32)
                .mul_add(DEQUANT, head.l1_biases.block(bucket)[j])
                .clamp(0.0, 1.0);
            dense[j] = clipped * clipped;
        }
        assert_eq!(sparse, dense);
        assert!(count > 0, "the synthetic input must produce some nonzeros");
    }

    /// The forward is deterministic, finite, and the piece count actually
    /// selects a bucket (different-material positions can differ).
    #[test]
    fn forward_deterministic_and_bucketed() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let (a, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (b, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        assert_eq!(a, b, "deterministic");
        assert!(a.abs() < 100_000, "finite, sane cp");
        // piece counts 20 and 32 map to different buckets (4 vs 7).
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[20], 4);
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[32], 7);
        // stm flip changes which half leads → generally a different value.
        let (w, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (bl, _, _) = head.forward(&pst, &threat, Color::Black, 20, &ZERO_PSQT);
        assert!(w != bl, "side-to-move relativity is wired");
    }

    /// T6c: the head adds EXACTLY `(psqt[stm][b] − psqt[ntm][b]) >> 8` cp on
    /// top of the psqt-less value — pinned per bucket, per side to move, and
    /// for negative differences (arithmetic shift = floor division by 256,
    /// NOT truncation toward zero).
    #[test]
    fn psqt_skip_adds_exactly_the_shifted_difference() {
        let head = synth_head();
        let (pst, threat) = synth_accs();

        // A distinct, sign-mixed pair per (perspective, bucket).
        let mut psqt = ZERO_PSQT;
        for (p, side) in psqt.iter_mut().enumerate() {
            for (b, slot) in side.iter_mut().enumerate() {
                *slot = (p as i32 * 2 - 1) * (1000 * b as i32 + 137) + b as i32 * 71;
            }
        }

        for stm in Color::ALL {
            for piece_count in [2usize, 9, 20, 32] {
                let bucket = OUTPUT_BUCKETS_LAYOUT[piece_count];
                let (base, _, _) = head.forward(&pst, &threat, stm, piece_count, &ZERO_PSQT);
                let (skip, _, _) = head.forward(&pst, &threat, stm, piece_count, &psqt);
                let s = stm as usize;
                let expected = (psqt[s][bucket] - psqt[s ^ 1][bucket]) >> 8;
                assert_eq!(
                    skip - base,
                    expected,
                    "stm {stm:?} pieces {piece_count} bucket {bucket}"
                );
            }
        }

        // Arithmetic-shift semantics on a negative difference: −1 >> 8 == −1
        // (floor), never 0 (truncation).
        let mut neg = ZERO_PSQT;
        neg[0][OUTPUT_BUCKETS_LAYOUT[20]] = -1;
        let (base, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (skip, _, _) = head.forward(&pst, &threat, Color::White, 20, &neg);
        assert_eq!(skip - base, -1, "arithmetic shift floors toward −∞");
    }

    /// EXP-UNC-1: the optional variance head yields a finite, positive σ̂ and
    /// leaves the value (mean) path byte-identical — the head-level form of the
    /// "bit-identical when the uncertainty consumers are off" guarantee. Without
    /// the head, σ̂ is `None`.
    #[test]
    fn variance_head_yields_finite_sigma_without_touching_value() {
        let (pst, threat) = synth_accs();

        let plain = synth_head();
        let (v0, s0, _) = plain.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        assert!(s0.is_none(), "no variance head => no sigma");

        let mut head = synth_head();
        let vw: Vec<f32> = (0..BUCKETS * L3)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let vb: Vec<f32> = (0..BUCKETS).map(|i| i as f32 * 0.01 - 0.02).collect();
        head.l3_var_weights = Some(vw.into_boxed_slice());
        head.l3_var_biases = Some(vb.into_boxed_slice());
        let (v1, s1, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);

        assert_eq!(
            v0, v1,
            "the value (mean) path is unchanged by the variance head"
        );
        let sigma = s1.expect("variance head => Some(sigma)");
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma finite & positive, got {sigma}"
        );
    }

    /// EXP-UNC-1 (D33 review, reviewer C): propagate_l3_var must be a faithful
    /// mirror of the trusted propagate_l3 — given identical weights/biases it
    /// returns identical values for every bucket. Pins the lane layout, reduction
    /// order, and bias so the σ̂ dot-product cannot silently diverge from the mean
    /// head (the two are hand-kept duplicates until deduped).
    #[test]
    fn propagate_l3_var_mirrors_propagate_l3() {
        let mut head = synth_head();
        head.l3_var_weights = Some(head.l3_weights.iter().copied().collect());
        head.l3_var_biases = Some(head.l3_biases.clone());
        let l2: [f32; L3] = std::array::from_fn(|i| (i as f32 * 0.031 - 0.5).sin());
        for bucket in 0..BUCKETS {
            assert_eq!(
                head.propagate_l3_var(&l2, bucket),
                head.propagate_l3(&l2, bucket),
                "variance head must mirror the mean head exactly at bucket {bucket}"
            );
        }
    }
}
