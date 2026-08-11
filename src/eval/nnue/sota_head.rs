//! The SOTA net's **head** forward pass (design/11 §12.3, D40): pairwise-multiply
//! FT activation → NNZ-sparse int8 L1 → f32 L2/L3 tail → 8 output buckets.
//! Our own reimplementation (D27/D31; never copied) — Reckless's *weights* stay
//! firewalled and nothing here loads them.
//!
//! **Feature-set agnostic**: it consumes two per-perspective accumulators
//! (`pst` and `threat`) and knows nothing about how they were filled.
//!
//! Behind the `sota` cargo feature; the default build never compiles it.

use super::aligned::Blocks;
use super::sota_simd::{activate, l1_dequant, l1_dot, l2_tail, l3_dot};
use crate::types::Color;

/// Accumulator width per perspective. L2 is the D47 shared bottleneck where
/// heads branch: the value tail continues L2→L3→1×buckets.
// The arch-* features are MUTUALLY EXCLUSIVE: each pins a different (L1, L2,
// L3) triple and a CENN only loads into a build whose dims match it. The L2/L3
// cfg arms below deliberately SHARE, so a guard keyed on those alone would not
// catch every pair.
#[cfg(any(
    all(feature = "arch-d47v", feature = "arch-d48w"),
    all(feature = "arch-d47v", feature = "arch-d48x"),
    all(feature = "arch-d47v", feature = "arch-hka2048"),
    all(feature = "arch-d48w", feature = "arch-d48x"),
    all(feature = "arch-d48w", feature = "arch-hka2048"),
    all(feature = "arch-d48x", feature = "arch-hka2048"),
    all(feature = "arch-d47v", feature = "arch-hka768"),
    all(feature = "arch-d48w", feature = "arch-hka768"),
    all(feature = "arch-d48x", feature = "arch-hka768"),
    all(feature = "arch-hka2048", feature = "arch-hka768"),
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
    feature = "arch-hka2048",
    feature = "arch-hka768"
)))]
pub const L1: usize = 1280;
/// D47V: the distilled 768 deploy arch (768/32/32).
#[cfg(feature = "arch-d47v")]
pub const L1: usize = 768;
/// D48W: gen-2 wide substrate, thin head — 1536/16/32 + skip.
#[cfg(feature = "arch-d48w")]
pub const L1: usize = 1536;
/// D48X: the gen-2 recipe at the A3 trunk's own width, 1280/16/32 + skip, so
/// the trunk transfer stays exact (no net2net expansion).
#[cfg(feature = "arch-d48x")]
pub const L1: usize = 1280;
/// HalfKA-only probe, UNRATIFIED: 2048/32/32 + skip, 2048 needs re-deriving.
#[cfg(feature = "arch-hka2048")]
pub const L1: usize = 2048;
/// HalfKA-ONLY 768 with the d48x head shape (16/32) — the fastest net we can
/// build. The threat rows are dropped ENTIRELY: HalfKA is 22,528 of the 83,248
/// input rows, so the FT working set falls 101.6 MB -> 16.5 MB. This feature
/// only fixes the DIMS — the CENN decides whether threat sections exist — and
/// the quality cost of dropping 72.9 % of the input rows is UNMEASURED.
#[cfg(feature = "arch-hka768")]
pub const L1: usize = 768;
/// L1→L2 output width (the D47 shared bottleneck).
#[cfg(not(any(
    feature = "arch-d47v",
    feature = "arch-d48w",
    feature = "arch-d48x",
    feature = "arch-hka2048",
    feature = "arch-hka768"
)))]
pub const L2: usize = 128;
#[cfg(any(feature = "arch-d47v", feature = "arch-hka2048"))]
pub const L2: usize = 32;
#[cfg(any(feature = "arch-d48w", feature = "arch-d48x", feature = "arch-hka768"))]
pub const L2: usize = 16;
/// L2→L3 output width.
pub const L3: usize = 32;
/// Material/output buckets.
pub const BUCKETS: usize = 8;

/// A per-perspective pair of T6c PSQT-skip bucket sums (`[White, Black]`, like
/// the accumulators): the i32 sums of the active features' PSQ1 rows. The stored
/// rows fold in `0.5 · 400 · 256` (contract: `jax/archon_jax/checkpoint.py`), so
/// the head adds `(psqt[stm][b] − psqt[ntm][b]) >> 8` centipawns — an UNCLAMPED
/// linear channel that bypasses every nonlinearity.
pub type PsqtPair = [[i32; BUCKETS]; 2];

/// The all-zero skip pair — what a net without a PSQ1 section contributes.
/// `(0 − 0) >> 8 == 0`, so passing this is bit-identical to the pre-T6c head.
pub const ZERO_PSQT: PsqtPair = [[0; BUCKETS]; 2];

pub(crate) const FT_QUANT: i32 = 255; // FT activation clamp ceiling / QA
/// The quantisation scale of the separately exported FullThreats transformer
/// that a **`CEN2` container implies**, because that header has no field for it.
/// **The scale is a property of the NET, not of the engine**: without QAT the
/// threat weights span ±1.98 and 64 is the largest non-clipping scale, while QAT
/// at `qat_threat_scale = 256` clips them to ±127/256 and exporting THOSE at 64
/// throws away two bits. Hence [`CENN-v5`], which carries `threat_quant` in the
/// header; this constant is only what a headerless `CEN2` file means.
///
/// [`CENN-v5`]: super::sota_loader
pub(crate) const THREAT_QUANT_CEN2: i32 = 64;
pub(crate) const L1_QUANT: i32 = 64; // L1 weight scale / QB
pub(crate) const FT_SHIFT: i32 = 9; // pairwise-multiply right shift

/// The threat→HalfKA frame shift a container's `threat_quant` implies, or
/// `None` if the value is not one this runtime can carry as a shift.
///
/// The merge rescales the threat accumulator by `FT_QUANT / threat_quant`
/// before summing it into the HalfKA frame, carrying that ratio as the nearest
/// POWER OF TWO — `256 / threat_quant`, one `slli` instead of a six-op
/// `round(t · 255/q)` (symmetric: 255/64 → 4, 255/256 → 1; mean 0.14 SF cp over
/// the 800-FEN battery). So `threat_quant` must be a power of two and the shift
/// is `log2(256 / threat_quant)`: `256 → 0`, `128 → 1`, `64 → 2` (CEN2).
///
/// **The accepted set is exactly `{64, 128, 256}`** ([`MAX_THREAT_SHIFT`]), and
/// that is a PERFORMANCE contract: the shift has to be an IMMEDIATE for the
/// accumulator load to fold into it, so the backends monomorphise the kernel
/// over it. Nothing coarser than 64 is wanted and nothing finer than 256 fits
/// an i8, so widening the set is not expected.
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

/// The largest merge shift the SIMD backends are monomorphised for; what lets
/// `sota_simd`'s `activate` dispatch stay exhaustive without a panic arm.
pub(crate) const MAX_THREAT_SHIFT: u32 = 2;

/// The `threat_quant` a merge shift came from — the exact inverse of
/// [`threat_shift_for`], so loader and serializer cannot disagree on the scale.
pub(crate) const fn threat_quant_for(shift: u32) -> i32 {
    256 >> shift
}
/// `(1 << FT_SHIFT) / (FT_QUANT² · L1_QUANT)` — the i32→f32 bridge at L1
/// (design/11 §12.6).
const DEQUANT: f32 = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT * L1_QUANT) as f32;
/// `(1 << FT_SHIFT) / FT_QUANT²` — the u8 pairwise activation back to the
/// trainer's float `hl1 ∈ [0,1]` (`ft_u8 = hl1 · FT_QUANT²/2^FT_SHIFT`). Any
/// head that reads the FT vector as floats reads it through this.
pub(crate) const FT_DEQUANT: f32 = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32;

/// Domain of the output-bucket map: `popcount(occupancy)` ∈ `0..=32`.
pub const PIECE_COUNT_BINS: usize = 33;

/// A piece-count → output-bucket map. **Per-net data** (the OBL1 container
/// section), because a net is trained against one specific partition of the
/// piece-count axis. Containers without an OBL1 section imply
/// [`LEGACY_BUCKET_LAYOUT`], keeping every existing net bit-identical.
pub type BucketLayout = [u8; PIECE_COUNT_BINS];

/// Expand `BUCKETS` inclusive piece-count ceilings into the full map: a piece
/// count lands in the first bucket whose ceiling covers it. Ceilings must be
/// strictly increasing and end at 32, or const evaluation fails below.
const fn layout_from_ceilings(ceilings: [u8; BUCKETS]) -> BucketLayout {
    let mut map = [0u8; PIECE_COUNT_BINS];
    let mut count = 0;
    while count < PIECE_COUNT_BINS {
        let mut bucket = 0;
        while (ceilings[bucket] as usize) < count {
            bucket += 1;
        }
        map[count] = bucket as u8;
        count += 1;
    }
    map
}

/// The implied layout of every container WITHOUT an OBL1 section.
///
/// ATTRIBUTION: adopted from Reckless (AGPLv3), retained as computed ranges
/// rather than copied table text, and ONLY because every already-trained net's
/// per-bucket weights were fitted against this exact partition — changing the
/// default degrades them and moves the NNUE gate's eval-hash pin. **Delete
/// when the pre-OBL1 nets retire.**
pub const LEGACY_BUCKET_LAYOUT: BucketLayout =
    layout_from_ceilings([8, 12, 16, 19, 22, 25, 28, 32]);

/// ARCHON's own output-bucket layout: equal-mass buckets over the piece-count
/// distribution of our training data — derived, not designed, by
/// `tools/nnue/derive_output_buckets.py` (ceiling k = smallest piece count whose
/// cumulative mass reaches k/8; receipt in
/// `docs/audit/2026-08-originality/output-buckets-derivation.txt`). Nothing
/// selects it at runtime until a net ships an OBL1 section naming it.
pub const ARCHON_BUCKET_LAYOUT: BucketLayout = layout_from_ceilings(ARCHON_BUCKET_CEILINGS);

/// The derived inclusive ceilings behind [`ARCHON_BUCKET_LAYOUT`], kept as the
/// 8 numbers the tool produces so its output diffs against the shipped value.
pub const ARCHON_BUCKET_CEILINGS: [u8; BUCKETS] = [8, 10, 13, 15, 18, 21, 25, 32];

// The head's weight tables as `BUCKETS` fixed-size blocks: `Blocks` states the
// shape in the type, so the only runtime check per acquisition is
// `bucket < BUCKETS`. Every `N · size_of::<T>()` below is a multiple of 64 at
// every shipped arch (`Blocks::LAYOUT_OK` asserts it), so the reshape adds no
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
/// `[L1]` u8 — the i8 HalfKA transformer's per-column dequantisation scale.
pub type FtScale = Blocks<u8, L1, 1>;

/// The D47V output-skip lane, quantised to i8 at load:
/// `w[b][i] ≈ q[b · L1 + i] · scale[b]`, per-bucket scale so a bucket with small
/// weights keeps its full i8 resolution. Costs accuracy, buys `vpdpbusd`.
#[derive(Debug)]
pub struct SkipLane {
    /// `[BUCKETS][L1]` i8, bucket-major contiguous.
    pub q: SkipQ,
    /// Per-bucket dequantisation scale, `[BUCKETS]`.
    pub scale: Box<[f32]>,
}

/// Columns the disagreement head emits: 0 = the SIGNED residual
/// `E[target − sigmoid(value) | features]`, 1 = the MAGNITUDE
/// `E[|residual| | features]`. Both live in PROBABILITY space, not centipawns,
/// so a consumer that wants cp must map through the same sigmoid the value
/// head's `network_scale` implies. The container's `outs` is validated here.
pub const DIS_OUTS: usize = 2;

/// The OPTIONAL disagreement head — a two-layer probe of the SAME pairwise
/// activation `hl1` the value path's L1 consumes,
/// `crelu(hl1 · proj_w + proj_b) · out_w + out_b`
/// (`jax/archon_jax/model.py::head_forward`, the `arch.dis_proj` branch).
///
/// **Deliberately NOT on the per-node eval path** — [`SotaHead::disagreement`]
/// is the only entry, so the cost is bounded by the caller's poll rate. Stored
/// f32 rather than i8 because off the hot path there is no cycle argument for
/// an uncalibrated quantisation frame, and f32 keeps the deployed function the
/// one the trainer fitted.
#[derive(Debug)]
pub struct DisHead {
    /// Projection width (`dis_proj`) — a per-net width, not a build constant.
    pub proj: usize,
    /// `[L1][proj]` f32, row-major: `proj_w[i · proj + j]`.
    pub proj_w: Box<[f32]>,
    /// `[proj]` f32.
    pub proj_b: Box<[f32]>,
    /// `[proj][DIS_OUTS]` f32, row-major: `out_w[j · DIS_OUTS + k]`.
    pub out_w: Box<[f32]>,
    /// `[DIS_OUTS]` f32.
    pub out_b: [f32; DIS_OUTS],
}

impl DisHead {
    /// The head's forward pass over an activated FT vector: `ft` is the u8
    /// pairwise activation in the `FT_QUANT²/2^FT_SHIFT` frame ([`FT_DEQUANT`]
    /// undoes it). The zero skip is bit-exact — dropping a term worth exactly
    /// `0.0 · w` leaves the ORDER of the rest.
    pub fn forward(&self, ft: &[u8; L1]) -> [f32; DIS_OUTS] {
        let mut hidden = self.proj_b.to_vec();
        for (i, &a) in ft.iter().enumerate() {
            if a == 0 {
                continue;
            }
            let x = f32::from(a) * FT_DEQUANT;
            let row = &self.proj_w[i * self.proj..(i + 1) * self.proj];
            for (h, &w) in hidden.iter_mut().zip(row) {
                *h = x.mul_add(w, *h);
            }
        }
        let mut out = self.out_b;
        for (j, &h) in hidden.iter().enumerate() {
            // CReLU — clamp(h, 0, 1), the trainer's `_crelu`. A NaN activation
            // propagates, which is what the caller's finite check catches.
            let x = h.clamp(0.0, 1.0);
            let row = &self.out_w[j * DIS_OUTS..(j + 1) * DIS_OUTS];
            for (o, &w) in out.iter_mut().zip(row) {
                *o = x.mul_add(w, *o);
            }
        }
        out
    }
}

/// The head's per-bucket weights + the net's output scale. Format-agnostic: the
/// CENN loader fills these from disk; tests fill them synthetically. Layouts are
/// bucket-major, flattened.
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
    /// HalfKA frame — [`threat_shift_for`] of the container's `threat_quant`,
    /// a per-net SCALAR like `network_scale`. A `CEN2` net yields 2, the literal
    /// every SIMD backend used to hardcode.
    pub threat_shift: u32,
    /// EXP-UNC-1: the OPTIONAL heteroscedastic log-variance head — a parallel
    /// `[BUCKETS][L3]` weight block + `[BUCKETS]` bias trained under Gaussian
    /// NLL. `None` ⇒ [`forward`](Self::forward) returns `sigma = None` and
    /// consumers fall back to their constants. `D40_UNC_HEAD` sidecar, not CENN.
    pub l3_var_weights: Option<Box<[f32]>>,
    /// `[BUCKETS]` f32, paired with [`l3_var_weights`](Self::l3_var_weights).
    pub l3_var_biases: Option<Box<[f32]>>,
    /// D41 rich-net: OPTIONAL co-trained WDL head — a bucket-agnostic `[3·L3]`
    /// weight block + `[3]` biases (W/D/L logits off the same bucket-selected L3
    /// activation the value head reads). `None` ⇒ `wdl = None`. `D40_WDL_HEAD`
    /// sidecar, not CENN. Layout: row-major `[o·L3 + j]`, o ∈ {L,D,W}.
    pub l3_wdl_weights: Option<Box<[f32]>>,
    /// `[3]` f32 biases paired with [`l3_wdl_weights`](Self::l3_wdl_weights).
    pub l3_wdl_biases: Option<Box<[f32]>>,
    /// D47V output-skip lane, QUANTISED TO i8 at load ([`SkipLane`]).
    /// `None` = pre-skip net: the eval path is bit-identical to before.
    pub skip_weights: Option<SkipLane>,
    /// The OPTIONAL disagreement head ([`DisHead`]), from the container's DIS1
    /// section. Even with one present the value path is bit-identical, because
    /// [`SotaHead::disagreement`] is the only accessor.
    pub dis_head: Option<DisHead>,
    /// Per-accumulator-lane dequantisation multiplier for the **i8 HalfKA
    /// feature transformer**, `[L1]`.
    ///
    /// The HalfKA table is stored on disk as i16 in the `FT_QUANT = 255` frame
    /// and quantised to i8 AT LOAD, one integer scale per accumulator column, so
    /// each column spends its full i8 range on its own dynamic range. The
    /// accumulator therefore lives in the reduced frame `255 / ft_scale[j]` and
    /// [`activate_ft`] multiplies it back before the clamp; all-ones makes that
    /// frame the identity and the eval bit-identical to the pre-i8 path.
    pub ft_scale: FtScale,
    /// This net's piece-count → output-bucket map: the container's OBL1
    /// section, or [`LEGACY_BUCKET_LAYOUT`] when it has none. The map is
    /// trained INTO the per-bucket weights, so it travels with them.
    pub bucket_layout: BucketLayout,
}

/// Pairwise-multiply FT activation (design/11 §12.3 step 1): sum the two
/// accumulators, clamp each half to `[0, FT_QUANT]`, multiply first×second half
/// and `>> FT_SHIFT` → u8 ∈ [0,127]. STM half occupies `[0, L1/2)`, and the
/// wrapping i16 accumulator math is part of the frame, not an accident.
fn activate_ft(
    pst: &[[i16; L1]; 2],
    threat: &[[i16; L1]; 2],
    scale: &[u8; L1],
    stm: Color,
    threat_shift: u32,
    out: &mut [u8; L1],
) {
    activate(out, pst, threat, scale, stm as usize, threat_shift);
}

/// Convert the independently quantized threat accumulator to the HalfKA scale
/// before summing: `round(pst + threat · FT_QUANT/threat_quant)` in quantized
/// space. Adding the raw integers would underweight every threat feature by the
/// frame ratio (4x on a CEN2 net).
#[cfg(test)]
#[inline]
fn merge_transformers(pst: i16, threat: i16, scale: i16, threat_shift: u32) -> i16 {
    // Two INDEPENDENT rescales: threat by a power-of-two shift, HalfKA by
    // `scale[j]` (the i8 transformer stores rows at `FT_QUANT / scale[j]`). The
    // HalfKA multiply WRAPS in i16 by definition — the vector backends multiply
    // in i16 lanes, so widening here would disagree with them.
    let scaled = i32::from(threat) << threat_shift;
    (i32::from(pst.wrapping_mul(scale)) + scaled).clamp(0, FT_QUANT) as i16
}

/// Indices of the 4-byte groups (of `L1/4`) holding any nonzero activation
/// (design/11 §12.3 step 2) — ~85–95% are zero after the clamp. Dispatched
/// because the scalar scan's data-dependent branches mispredict on the sparse
/// pattern; the SIMD backends compact via movemask instead.
fn find_nnz(ft: &[u8; L1], idx: &mut [u16; L1 / 4]) -> usize {
    super::sota_simd::find_nnz(ft, idx)
}

/// Reusable per-thread scratch for [`SotaHead::forward_into`] — the activated
/// feature vector and the non-zero group list its sparse L1 matmul walks.
///
/// **KNOWN REGRESSION — keep this caller-owned, not two locals in
/// `forward_into`.** As `let mut ft = [0u8; L1]` the zero-initialiser compiles
/// to a libc `memset` at EVERY evaluation, because LLVM cannot dead-store-
/// eliminate it across the indirect `activate_ft` dispatch. A long-lived buffer
/// has nothing to eliminate and needs no `MaybeUninit` safety argument. Stale
/// bytes are never read: `activate_ft` writes all `L1` lanes of `ft` before
/// anything reads it, and `nnz` is consumed as `&nnz[..count]`. Cache-line
/// aligned, like SF's `alignas(CacheLineSize) transformedFeatures`.
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

    /// The activated feature-transformer vector left behind by the last
    /// [`SotaHead::forward_into`] on this scratch — what makes a disagreement
    /// poll affordable, since re-running `activate_ft` is the hottest thing in
    /// the binary.
    ///
    /// **Validity is the caller's contract**, and it is narrow: read it only
    /// immediately after evaluating the node in question and before any child.
    /// Nothing enforces that — a stale flag would cost the hot path.
    #[inline]
    pub fn last_ft(&self) -> &[u8; L1] {
        &self.ft
    }

    /// A scratch filled with values no forward pass can legitimately leave
    /// behind. A lane a kernel fails to write is invisible against a ZEROED
    /// buffer (most activations are zero after the clamp); against a reused one
    /// it is the previous position's, and the eval goes history-dependent.
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
    /// The output bucket THIS NET selects for a total piece count — the index
    /// the forward pass uses, exposed so tests and the debug oracle pin the T6c
    /// skip against it. A method because the map is per-net data.
    #[inline]
    pub fn output_bucket(&self, piece_count: usize) -> usize {
        usize::from(self.bucket_layout[piece_count])
    }

    /// int8 sparse matmul over the nonzero groups → dequantized SCReLU f32[L2]
    /// (design/11 §12.3 step 3). Products accumulate in i32, `dpbusd` semantics.
    fn propagate_l1(&self, ft: &[u8; L1], nnz: &[u16], bucket: usize, out: &mut [f32; L2]) {
        let mut pre = [0i32; L2];
        let wb = self.l1_weights.block(bucket);
        l1_dot(&mut pre, ft, nnz, wb);
        let bias = self.l1_biases.block(bucket);
        l1_dequant(out, &pre, bias, DEQUANT);
    }

    /// Dense f32[L2]→f32[L3], bias-seeded FMA then SCReLU (step 4). The
    /// i-outer/j-inner order is load-bearing: it pins the accumulation order,
    /// and the scalar backend is the bit-exact reference the SIMD backends must
    /// match (j-lane order, per-lane i-sequential accumulation).
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
        // `sota_simd::scalar::l3_dot` is the bit-exact reference the SIMD
        // backends match.
        l3_dot(self.l3_weights.block(bucket), l2) + self.l3_biases[bucket]
    }

    /// EXP-UNC-1: the log-variance head's dense f32[L3]→f32, mirroring
    /// [`propagate_l3`](Self::propagate_l3) exactly (same lane layout +
    /// reduction order) but on the variance weights. Returns raw `s = log(σ²)`
    /// of the SIGMOID (WDL) value, so σ̂ = exp(½·s) is a WDL-space std, NOT a
    /// cp-space one. Caller guarantees the head is present.
    // `#[cold] #[inline(never)]` because no shipped net has either of these
    // heads: both bodies are unreachable at run time, and inlining them cost
    // ~140 static instructions in the hottest eval function.
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

    /// D41 rich-net WDL head: 3 raw logits (L,D,W) off the same bucket-selected
    /// L3 activation the value head reads. Bucket-agnostic weights `[3·L3]`
    /// row-major (`w[o·L3 + j]`) + `[3]` biases, mirroring the trainer's
    /// `l3_wdl` affine. Softmax is the consumer's; caller guarantees the head.
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
    /// piece count → (stm-relative centipawns, optional σ̂, optional WDL).
    /// `psqt` is the T6c skip pair the front-end accumulates alongside
    /// ([`ZERO_PSQT`] for a net without one — bit-identical); `sigma` is `Some`
    /// only for an EXP-UNC-1 net, in sigmoid (WDL) space off the same `l2`.
    pub fn forward(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
        psqt: &PsqtPair,
    ) -> (i32, Option<f32>, Option<[f32; 3]>) {
        // Convenience wrapper: builds its own scratch, so it pays the zero-init
        // `HeadScratch` exists to remove. The hot path calls `forward_into`.
        let mut l1 = [0.0f32; L2];
        let mut scratch = HeadScratch::new();
        self.forward_into(pst, threat, stm, piece_count, psqt, &mut scratch, &mut l1)
    }

    /// The disagreement head's prediction for a position, `None` when the net
    /// carries no DIS1 section or when the head produced a non-finite value.
    ///
    /// **This is an on-demand poll and is NOT called by [`forward`] or
    /// [`forward_into`].** It re-runs `activate_ft` rather than borrowing the
    /// value pass's FT vector, so the per-node eval path does not change at all
    /// whether or not a net has this head; poll many nodes through
    /// [`disagreement_from_ft`](Self::disagreement_from_ft) instead. The
    /// `[0u8; L1]` local is the `memset` [`HeadScratch`] keeps off the hot path
    /// — free at this head's poll rates.
    ///
    /// [`forward`]: Self::forward
    /// [`forward_into`]: Self::forward_into
    #[cold]
    #[inline(never)]
    pub fn disagreement(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
    ) -> Option<[f32; DIS_OUTS]> {
        // Early-out BEFORE the activation: on a net with no DIS1 section this
        // whole call is one null check, so a caller may poll unconditionally.
        self.dis_head.as_ref()?;
        let mut ft = [0u8; L1];
        activate_ft(
            pst,
            threat,
            self.ft_scale.block(0),
            stm,
            self.threat_shift,
            &mut ft,
        );
        self.disagreement_from_ft(&ft)
    }

    /// [`disagreement`](Self::disagreement) on an ALREADY-ACTIVATED feature
    /// vector — the affordable form of the poll. Pair it with
    /// [`HeadScratch::last_ft`], whose doc carries the validity contract: the
    /// vector must belong to the position being asked about. Same `None`
    /// semantics as its sibling.
    #[inline]
    pub fn disagreement_from_ft(&self, ft: &[u8; L1]) -> Option<[f32; DIS_OUTS]> {
        let dis = self.dis_head.as_ref()?;
        let out = dis.forward(ft);
        // Same guard as `sigma`: a consumer must get `None` rather than a NaN
        // that silently poisons a search margin.
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// [`forward`](Self::forward), writing the shared-bottleneck activation
    /// (the L2-wide output an attached head dots against) into `l1_out`.
    /// `scratch` is the caller's long-lived [`HeadScratch`] — see that type for
    /// why it is a parameter and not two locals.
    // One argument over clippy's default; they are all genuinely independent.
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
        // Destructured because `find_nnz(&scratch.ft, &mut scratch.nnz)` needs
        // a shared and a unique borrow of two DIFFERENT fields at once, which a
        // field-by-field pattern makes legal without `unsafe`.
        let HeadScratch { ft, nnz } = scratch;
        activate_ft(
            pst,
            threat,
            self.ft_scale.block(0),
            stm,
            self.threat_shift,
            ft,
        );
        // `bucket` is DUPLICATED under the two `cfg`s rather than hoisted, so
        // the arm-OFF build keeps its statement order byte-for-byte.
        #[cfg(any(feature = "hskip60", feature = "htail60"))]
        let bucket = usize::from(self.bucket_layout[piece_count]);
        // Experimental arm `htail60` — prefetch the head's TAIL tables (~2.4 KB,
        // read in full every evaluation at a ~99 % L1 miss rate) at entry, ~800
        // instructions before their first use. `T0`, not `T2`: the miss being
        // converted is an L1 miss. Costs ~36 hints per evaluation.
        #[cfg(feature = "htail60")]
        {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                let mut hint = |p: *const u8| {
                    #[allow(unsafe_code)]
                    // SAFETY: `_mm_prefetch` is a pure cache hint, defined for
                    // ANY address value -- it cannot fault, cannot trap and has
                    // no architectural effect beyond cache state. Same
                    // sanctioned use as `TranspositionTable::prefetch`.
                    unsafe {
                        _mm_prefetch(p.cast(), _MM_HINT_T0);
                    }
                };
                let l2w = self.l2_weights.block(bucket).as_ptr().cast::<u8>();
                for off in (0..L2 * L3 * 4).step_by(64) {
                    hint(l2w.wrapping_add(off));
                }
                hint(self.l2_biases.block(bucket).as_ptr().cast());
                hint(self.l3_weights.block(bucket).as_ptr().cast());
                hint(
                    self.l3_weights
                        .block(bucket)
                        .as_ptr()
                        .cast::<u8>()
                        .wrapping_add(64),
                );
                hint(self.l1_biases.block(bucket).as_ptr().cast());
            }
        }
        // Experimental arm `hskip60` — issue the output-skip lane's cold reads
        // HERE rather than after `propagate_l3`, putting them in flight across
        // the whole matmul. Bit-exact by construction: the f32 ADD into `l3`
        // stays where it was, so no floating-point operation is reordered.
        #[cfg(feature = "hskip60")]
        let skip_early = self.skip_weights.as_ref().map(|sk| {
            super::sota_simd::skip_dot_i8(&ft[..], sk.q.block(bucket)) as f32 * sk.scale[bucket]
        });
        let count = find_nnz(ft, nnz);
        #[cfg(not(any(feature = "hskip60", feature = "htail60")))]
        let bucket = usize::from(self.bucket_layout[piece_count]);
        self.propagate_l1(ft, &nnz[..count], bucket, l1_out);
        let l1 = l1_out;
        let l2 = self.propagate_l2(l1, bucket);
        let l3 = self.propagate_l3(&l2, bucket);
        // T6c PSQT skip: the unclamped linear channel joins AFTER the clamped
        // tail, straight in centipawns — `(stm − ntm) >> 8` (arithmetic shift;
        // the rows fold in 0.5·400·256, see `PsqtPair`). Zero pair ⇒ +0.
        // D47V output-skip: a learned linear read of the SAME activated pairwise
        // features the L1 matmul consumes, added in LOGIT units before the cp
        // scale (SF's fc_0 +1 lane). Frame: ft_u8 = hl1_f · FT_QUANT²/2^FT_SHIFT.
        let l3 = if let Some(sk) = &self.skip_weights {
            // Branchless, bucket-major contiguous lane (the loader transposes
            // at parse), dotted as INTEGER u8 x i8 `vpdpbusd`. It cannot fold
            // into `l1_dot` without a 17th output row, which doubles that kernel.
            #[cfg(not(feature = "hskip60"))]
            let lane = sk.q.block(bucket);
            #[cfg(not(feature = "hskip60"))]
            let acc = super::sota_simd::skip_dot_i8(&ft[..], lane) as f32 * sk.scale[bucket];
            // `skip_early` is `Some` exactly when `skip_weights` is; `expect`
            // so an edit that breaks the pairing fails loudly, not silently.
            #[cfg(feature = "hskip60")]
            let acc = skip_early.expect("skip_early is Some whenever skip_weights is");
            l3 + acc * ((1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32)
        } else {
            l3
        };
        let stm_i = stm as usize;
        let value = (l3 * self.network_scale as f32) as i32
            + ((psqt[stm_i][bucket] - psqt[stm_i ^ 1][bucket]) >> 8);
        // σ̂ = exp(½·logσ²). The D40_UNC_HEAD sidecar is size- but not
        // content-validated, so guard: `None` makes consumers fall back to their
        // factor-1.0 no-op instead of NaN-poisoning a margin.
        let sigma = self.l3_var_weights.as_ref().and_then(|_| {
            let s = (0.5 * self.propagate_l3_var(&l2, bucket)).exp();
            s.is_finite().then_some(s)
        });
        // WDL probabilities [L,D,W] (softmax of the 3 logits). `None` if no head
        // or a non-finite result — same guard rationale as `sigma`.
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

    /// `ft`/`nnz` are a REUSED buffer, so nothing in the forward pass may depend
    /// on their prior contents. Catches an `activate` backend that writes fewer
    /// than `L1` lanes, or a `find_nnz` count that under-covers its groups.
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

        // …and the buffer really was dirty on entry (not a vacuous assertion).
        assert_eq!(HeadScratch::poisoned().ft[0], 0xAB);
        assert_ne!(HeadScratch::poisoned().ft, HeadScratch::new().ft);
    }

    #[test]
    fn mixed_transformer_scales_are_normalized_before_sum() {
        // A CEN2 net at HalfKA scale 1 — the pre-i8-transformer frame.
        assert_eq!(merge_transformers(0, 64, 1, 2), 255);
        assert_eq!(merge_transformers(255, -64, 1, 2), 0);
        assert_eq!(merge_transformers(100, 16, 1, 2), 164);
        assert_eq!(merge_transformers(100, -16, 1, 2), 36);
        assert_eq!(merge_transformers(250, 2, 1, 2), 255);
        assert_eq!(merge_transformers(5, -2, 1, 2), 0);
        // A reduced-frame lane of 60 at HalfKA scale 4 is 240 in the FT_QUANT
        // frame, so the merge clamps the SCALED value.
        assert_eq!(merge_transformers(60, 0, 4, 2), 240);
        assert_eq!(merge_transformers(60, 8, 4, 2), 255);
        assert_eq!(merge_transformers(-60, 8, 4, 2), 0);
        // INDEPENDENT: one HalfKA lane, three threat frames, HalfKA = 240 in all.
        assert_eq!(merge_transformers(60, 4, 4, 0), 244);
        assert_eq!(merge_transformers(60, 4, 4, 1), 248);
        assert_eq!(merge_transformers(60, 4, 4, 2), 255);
    }

    /// The shift is the net's: threat_quant 256 merges UNSCALED, 128 halves.
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
        // Outside the monomorphised set — the backends' `activate` dispatch's
        // `_` arm is only sound because these are rejected here.
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

        assert_eq!(merge_transformers(100, 16, 1, 0), 116);
        assert_eq!(merge_transformers(100, 16, 1, 1), 132);
        assert_eq!(merge_transformers(100, 16, 1, 2), 164);
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

    /// The identity HalfKA scale a synthetic net loads with (weights fit i8
    /// unscaled), so these tests exercise the unreduced frame.
    const UNIT_SCALE: [u8; L1] = [1; L1];

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
            dis_head: None,
            ft_scale: FtScale::from_flat_iter(std::iter::repeat_n(1u8, L1)),
            bucket_layout: LEGACY_BUCKET_LAYOUT,
        }
    }

    fn synth_accs() -> ([[i16; L1]; 2], [[i16; L1]; 2]) {
        let mut pst = [[0i16; L1]; 2];
        let mut threat = [[0i16; L1]; 2];
        // Asymmetric perspectives, so an stm flip swaps distinct halves.
        for p in 0..2 {
            for i in 0..L1 {
                pst[p][i] = (((i as i32 * 3 + p as i32 * 17) % 200) - 40) as i16;
                threat[p][i] = (((i as i32 * 5 + p as i32 * 11) % 120) - 30) as i16;
            }
        }
        (pst, threat)
    }

    /// `activate_ft` matches a hand-computed pairwise-multiply: clamp + shift.
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
        activate_ft(&pst, &threat, &UNIT_SCALE, Color::White, 2, &mut ft);
        // (150 * 255) >> 9 = 38250 >> 9 = 74
        assert_eq!(ft[0], 74, "pairwise clamp+shift");
        // a negative sum clamps to 0 → product 0
        let mut pst2 = [[0i16; L1]; 2];
        pst2[0][1] = -500;
        let mut ft2 = [0u8; L1];
        activate_ft(&pst2, &threat, &UNIT_SCALE, Color::White, 2, &mut ft2);
        assert_eq!(ft2[1], 0, "negative clamps to 0");
    }

    /// The sparse L1 path equals a dense recomputation over ALL groups, which
    /// catches any `find_nnz` / sparse-indexing slip.
    #[test]
    fn sparse_l1_equals_dense() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let mut ft = [0u8; L1];
        activate_ft(&pst, &threat, &UNIT_SCALE, Color::White, 2, &mut ft);
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

    /// The forward is deterministic and finite, and the piece count selects.
    #[test]
    fn forward_deterministic_and_bucketed() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let (a, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (b, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        assert_eq!(a, b, "deterministic");
        assert!(a.abs() < 100_000, "finite, sane cp");
        // 20 and 32 land in different buckets on the legacy layout.
        assert_eq!(head.output_bucket(20), 4);
        assert_eq!(head.output_bucket(32), 7);
        // stm flip changes which half leads → generally a different value.
        let (w, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (bl, _, _) = head.forward(&pst, &threat, Color::Black, 20, &ZERO_PSQT);
        assert!(w != bl, "side-to-move relativity is wired");
    }

    /// The ceiling expansion is the ONLY definition of both layouts: pin legacy
    /// against the EDGES every pre-OBL1 net was trained under, ARCHON against
    /// its derivation receipt.
    #[test]
    fn bucket_layouts_expand_their_ceilings_exactly() {
        // Legacy edges: 8 | 12 | 16 | 19 | 22 | 25 | 28 | 32, both ends each.
        for (count, bucket) in [
            (0usize, 0u8),
            (8, 0),
            (9, 1),
            (12, 1),
            (13, 2),
            (16, 2),
            (17, 3),
            (19, 3),
            (20, 4),
            (22, 4),
            (23, 5),
            (25, 5),
            (26, 6),
            (28, 6),
            (29, 7),
            (32, 7),
        ] {
            assert_eq!(LEGACY_BUCKET_LAYOUT[count], bucket, "legacy count {count}");
        }
        // ARCHON edges: 8 | 10 | 13 | 15 | 18 | 21 | 25 | 32.
        for (count, bucket) in [
            (0usize, 0u8),
            (8, 0),
            (9, 1),
            (10, 1),
            (11, 2),
            (13, 2),
            (14, 3),
            (15, 3),
            (16, 4),
            (18, 4),
            (19, 5),
            (21, 5),
            (22, 6),
            (25, 6),
            (26, 7),
            (32, 7),
        ] {
            assert_eq!(ARCHON_BUCKET_LAYOUT[count], bucket, "archon count {count}");
        }
        // Structural sanity: monotone, no skipped bucket, covers all BUCKETS.
        for layout in [&LEGACY_BUCKET_LAYOUT, &ARCHON_BUCKET_LAYOUT] {
            assert_eq!(layout[0], 0);
            assert_eq!(usize::from(layout[PIECE_COUNT_BINS - 1]), BUCKETS - 1);
            for pair in layout.windows(2) {
                assert!(
                    pair[1] == pair[0] || pair[1] == pair[0] + 1,
                    "monotone, no skipped bucket: {pair:?}"
                );
            }
        }
    }

    /// A head with a different layout selects different buckets through the REAL
    /// forward pass; counts the layouts agree on stay bit-identical.
    #[test]
    fn forward_reads_the_per_net_bucket_layout() {
        let legacy = synth_head();
        let mut archon = synth_head();
        archon.bucket_layout = ARCHON_BUCKET_LAYOUT;
        let (pst, threat) = synth_accs();
        // Count 11: legacy bucket 1, ARCHON bucket 2, and the head's per-bucket
        // tables differ, so the values must differ.
        assert_eq!(legacy.output_bucket(11), 1);
        assert_eq!(archon.output_bucket(11), 2);
        let (a, _, _) = legacy.forward(&pst, &threat, Color::White, 11, &ZERO_PSQT);
        let (b, _, _) = archon.forward(&pst, &threat, Color::White, 11, &ZERO_PSQT);
        assert_ne!(a, b, "the forward pass must read the net's own layout");
        // Piece count 8 is bucket 0 in BOTH layouts → identical output.
        let (a0, _, _) = legacy.forward(&pst, &threat, Color::White, 8, &ZERO_PSQT);
        let (b0, _, _) = archon.forward(&pst, &threat, Color::White, 8, &ZERO_PSQT);
        assert_eq!(a0, b0, "a shared bucket evaluates bit-identically");
    }

    /// T6c: the head adds EXACTLY `(psqt[stm][b] − psqt[ntm][b]) >> 8` cp on top
    /// of the psqt-less value, per bucket, per side, and for negative
    /// differences (arithmetic shift = floor, not truncation).
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
                let bucket = head.output_bucket(piece_count);
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

        // Arithmetic shift on a negative difference: −1 >> 8 == −1, never 0.
        let mut neg = ZERO_PSQT;
        neg[0][head.output_bucket(20)] = -1;
        let (base, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (skip, _, _) = head.forward(&pst, &threat, Color::White, 20, &neg);
        assert_eq!(skip - base, -1, "arithmetic shift floors toward −∞");
    }

    /// The optional variance head yields a finite, positive σ̂ and leaves the
    /// value (mean) path byte-identical. Without the head, σ̂ is `None`.
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

    /// The disagreement head reproduces a HAND-WRITTEN dense reference of
    /// `crelu(hl1 · W + b) · Wo + bo`, `hl1 = ft · FT_DEQUANT`. This pins the
    /// layout (`proj_w[i·proj + j]`, `out_w[j·DIS_OUTS + k]`), which a
    /// transposed read reproduces plausibly; the zero skip must be EXACT, hence
    /// the bit-equality assert at the end.
    #[test]
    fn dis_head_matches_a_dense_hand_reference() {
        let proj = 6usize;
        let mut head = synth_head();
        head.dis_head = Some(DisHead {
            proj,
            proj_w: (0..L1 * proj)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.011)
                .collect(),
            proj_b: (0..proj).map(|j| (j as f32 - 2.0) * 0.05).collect(),
            out_w: (0..proj * DIS_OUTS)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
                .collect(),
            out_b: [0.125, -0.25],
        });
        let (pst, threat) = synth_accs();

        // The activation the head will see, recomputed here independently.
        let mut ft = [0u8; L1];
        activate_ft(
            &pst,
            &threat,
            head.ft_scale.block(0),
            Color::White,
            head.threat_shift,
            &mut ft,
        );
        assert!(ft.iter().any(|&a| a != 0), "synthetic ft is all zero");

        let d = head.dis_head.as_ref().unwrap();
        let mut hidden = vec![0.0f32; proj];
        for (j, h) in hidden.iter_mut().enumerate() {
            let mut acc = d.proj_b[j];
            for (i, &a) in ft.iter().enumerate() {
                // DENSE: every lane, including the zeros the shipped loop skips.
                acc = (f32::from(a) * FT_DEQUANT).mul_add(d.proj_w[i * proj + j], acc);
            }
            *h = acc;
        }
        let mut want = [0.0f32; DIS_OUTS];
        for (k, w) in want.iter_mut().enumerate() {
            let mut acc = d.out_b[k];
            for (j, &h) in hidden.iter().enumerate() {
                acc = h.clamp(0.0, 1.0).mul_add(d.out_w[j * DIS_OUTS + k], acc);
            }
            *w = acc;
        }

        let got = head
            .disagreement(&pst, &threat, Color::White)
            .expect("head present and finite");
        // Dense j-outer vs shipped i-outer: equal to float rounding, not bits.
        for k in 0..DIS_OUTS {
            assert!(
                (got[k] - want[k]).abs() <= 1e-5 * (1.0 + want[k].abs()),
                "column {k}: got {} want {}",
                got[k],
                want[k]
            );
        }

        // The zero SKIP must be exact: same order, zero terms simply not added.
        let dense_order = {
            let mut hidden = d.proj_b.to_vec();
            for (i, &a) in ft.iter().enumerate() {
                let x = f32::from(a) * FT_DEQUANT;
                for (j, h) in hidden.iter_mut().enumerate() {
                    *h = x.mul_add(d.proj_w[i * proj + j], *h);
                }
            }
            let mut out = d.out_b;
            for (j, &h) in hidden.iter().enumerate() {
                let x = h.clamp(0.0, 1.0);
                for (k, o) in out.iter_mut().enumerate() {
                    *o = x.mul_add(d.out_w[j * DIS_OUTS + k], *o);
                }
            }
            out
        };
        assert_eq!(
            got, dense_order,
            "skipping zero lanes must be bit-exact, not approximate"
        );
    }

    /// The head is OFF the eval path: installing one must not move `forward`'s
    /// value, sigma or WDL by a bit. A net without one returns `None`.
    #[test]
    fn dis_head_does_not_touch_the_value_forward() {
        let (pst, threat) = synth_accs();
        let plain = synth_head();
        assert!(plain.disagreement(&pst, &threat, Color::White).is_none());
        let base = plain.forward(&pst, &threat, Color::White, 24, &ZERO_PSQT);

        let mut head = synth_head();
        head.dis_head = Some(DisHead {
            proj: 4,
            proj_w: (0..L1 * 4).map(|i| (i % 9) as f32 * 0.07 - 0.3).collect(),
            proj_b: vec![0.1, -0.2, 0.3, 0.0].into_boxed_slice(),
            out_w: (0..8).map(|i| (i as f32) * 0.11 - 0.4).collect(),
            out_b: [1.5, -1.5],
        });
        let with = head.forward(&pst, &threat, Color::White, 24, &ZERO_PSQT);
        assert_eq!(base.0, with.0);
        assert_eq!(base.1, with.1);
        assert_eq!(base.2, with.2);
        assert!(head.disagreement(&pst, &threat, Color::White).is_some());
    }

    /// A NaN weight must yield `None`, not a NaN a consumer folds into a margin.
    #[test]
    fn dis_head_rejects_a_non_finite_result() {
        let (pst, threat) = synth_accs();
        let mut head = synth_head();
        head.dis_head = Some(DisHead {
            proj: 2,
            proj_w: (0..L1 * 2).map(|_| 0.001f32).collect(),
            proj_b: vec![0.0, 0.0].into_boxed_slice(),
            out_w: vec![f32::NAN, 1.0, 1.0, 1.0].into_boxed_slice(),
            out_b: [0.0, 0.0],
        });
        assert!(head.disagreement(&pst, &threat, Color::White).is_none());
    }

    /// `propagate_l3_var` must mirror `propagate_l3` bit for bit given the same
    /// weights — the two are hand-kept duplicates, so this pins the lane layout,
    /// reduction order and bias.
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
