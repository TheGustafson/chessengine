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
/// HalfKA-ONLY 768 (2026-08-01, owner-directed): the fastest net we can build.
/// Same 768 width the fold kernels were originally written for (this module's
/// header still says "768-wide"), with the d48x head shape (16/32) rather than
/// D47V's 32.
///
/// The threat rows are dropped ENTIRELY, not narrowed: HalfKA is 22,528 of the
/// 83,248 input rows (27.1 %), so the FT working set falls 101.6 MB -> 16.5 MB
/// (0.162x). Perf study 7 §5 measured dropping threat rows at **+25 % (768/wl50)
/// to +39 % (1280/middlegame) instructions:u, 3/3 rounds** — roughly 45-70 Elo
/// at the house 120/doubling, which exceeds the whole architecture-matched speed
/// deficit vs eca43a97 (1.185x / ~29 Elo, measured 2026-08-01).
///
/// Nothing here forces a threatless net: `largenet_threats` is an `Option` and
/// the loader accepts a net with no threat sections, so the CENN decides. This
/// feature only fixes the DIMS.
///
/// **The quality cost is UNMEASURED** — this discards 72.9 % of the input rows,
/// and `eca43a97`, which beats us by +90.7 in our own search, keeps its threats.
/// Equal-time games decide it.
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
/// `(1 << FT_SHIFT) / FT_QUANT²` — the u8 pairwise activation back to the
/// trainer's float `hl1 ∈ [0,1]` (`ft_u8 = hl1 · FT_QUANT²/2^FT_SHIFT`, the
/// factor the D47V skip lane already dequantises by). Any head that reads the
/// FT vector as floats reads it through this.
pub(crate) const FT_DEQUANT: f32 = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32;

/// Domain of the output-bucket map: `popcount(occupancy)` ∈ `0..=32`.
pub const PIECE_COUNT_BINS: usize = 33;

/// A piece-count → output-bucket map. **Per-net data since 2026-08** (the
/// OBL1 container section, [`super::sota_loader`]): a net is trained against
/// one specific partition of the piece-count axis, so the partition belongs
/// to the net, not to the binary. Containers without an OBL1 section imply
/// [`LEGACY_BUCKET_LAYOUT`], which keeps every existing net bit-identical.
pub type BucketLayout = [u8; PIECE_COUNT_BINS];

/// Expand `BUCKETS` inclusive piece-count ceilings into the full map: a piece
/// count lands in the first bucket whose ceiling covers it. Ceilings must be
/// strictly increasing and end at 32 — a shorter last ceiling fails const
/// evaluation on the out-of-bounds index below.
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

/// The implied layout of every container WITHOUT an OBL1 section — i.e. every
/// net trained before 2026-08.
///
/// ATTRIBUTION / REMEDIATION (2026-08 originality audit, item 1 —
/// `docs/audit/2026-08-originality/00-synthesis.md` §2): these boundaries were
/// adopted from Reckless (AGPLv3) by the pre-audit hand-copied table this
/// expression replaces. They are retained — as computed ranges, not as the
/// copied table text — ONLY because they are baked into every already-trained
/// net: the head's per-bucket weights were fitted against this exact
/// partition, so changing the default silently degrades every existing net
/// and moves the NNUE gate's eval-hash pin. **Delete this constant when the
/// pre-OBL1 nets retire.** New nets carry [`ARCHON_BUCKET_LAYOUT`] (or their
/// own) in the container.
pub const LEGACY_BUCKET_LAYOUT: BucketLayout =
    layout_from_ceilings([8, 12, 16, 19, 22, 25, 28, 32]);

/// ARCHON's own output-bucket layout: equal-mass buckets over the piece-count
/// distribution of ARCHON's training data — derived, not designed, so the
/// boundaries are a measured property of our corpus rather than anyone's
/// hand-tuned table.
///
/// Derivation (reproducible): `tools/nnue/derive_output_buckets.py` over the
/// first 100M records of `fishpack32-sf5k.shuf.bin` (globally shuffled, so a
/// prefix is a uniform sample; 50,135,273,696 B = 1,566,727,303 records,
/// first-MiB sha256 `f403e94e…010101`). Rule: ceiling k = smallest piece
/// count whose cumulative mass reaches k/8 of the sample, last ceiling forced
/// to 32. Full receipt (histogram, per-bucket mass 9.0–16.5%):
/// `docs/audit/2026-08-originality/output-buckets-derivation.txt`.
///
/// The derived map agrees with the legacy map on only 16/33 piece counts —
/// our corpus is materially lighter than Reckless's boundaries assume, so
/// equal mass pulls every interior ceiling down. The FIRST ceiling (8) does
/// coincide with the legacy one; that is where 15.0% of our sample's
/// cumulative mass happens to land, and the coincidence is flagged here per
/// the audit's honesty rule.
///
/// Ships with the FIRST net trained under it (the OBL1 section); until then
/// nothing selects it at runtime, so today's eval is untouched.
pub const ARCHON_BUCKET_LAYOUT: BucketLayout = layout_from_ceilings(ARCHON_BUCKET_CEILINGS);

/// The derived inclusive ceilings behind [`ARCHON_BUCKET_LAYOUT`] — kept as
/// the 8 numbers the derivation actually produces so the tool's output can be
/// diffed against the shipped value directly.
pub const ARCHON_BUCKET_CEILINGS: [u8; BUCKETS] = [8, 10, 13, 15, 18, 21, 25, 32];

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
/// `[L1]` u8 — the i8 HalfKA transformer's per-column dequantisation scale.
pub type FtScale = Blocks<u8, L1, 1>;

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

/// Columns the disagreement head emits. Column 0 is the SIGNED residual
/// `E[target − sigmoid(value) | features]` (which way to correct the eval);
/// column 1 is the MAGNITUDE `E[|residual| | features]` (whether to trust it at
/// all). Both live in PROBABILITY space, not centipawns — the trainer fits them
/// against `target − sigmoid(value_logit)` (`jax/archon_jax/losses.py`), so a
/// consumer that wants cp must map through the same sigmoid the value head's
/// `network_scale` implies. Pinned here because the container's `outs` field is
/// validated against it.
pub const DIS_OUTS: usize = 2;

/// The OPTIONAL disagreement head — a two-layer probe of the SAME pairwise
/// activation `hl1` the value path's L1 consumes:
/// `crelu(hl1 · proj_w + proj_b) · out_w + out_b`
/// (`jax/archon_jax/model.py::head_forward`, the `arch.dis_proj` branch).
///
/// **Deliberately NOT on the per-node eval path.** `forward_into` never touches
/// this; the only entry is [`SotaHead::disagreement`], which re-runs
/// `activate_ft` itself. That is the whole design premise — the head is polled
/// SELECTIVELY (root, a candidate node, a diagnostic dump), so its cost is
/// bounded by the caller rather than multiplied by the node count, and adding
/// it changed nothing about the value pipeline's instruction count.
///
/// Stored f32 rather than i8 for the reason the jax-side contract gives: off
/// the hot path there is no cycle argument for a quantisation frame nobody has
/// calibrated, and f32 makes the deployed function the one the trainer fitted,
/// which is what the JAX↔Rust parity check needs to be able to see plumbing
/// bugs rather than rounding.
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
    /// The head's forward pass over an activated FT vector.
    ///
    /// `ft` is the u8 pairwise activation in the `FT_QUANT²/2^FT_SHIFT` frame
    /// ([`FT_DEQUANT`] takes it back to the trainer's `hl1 ∈ [0,1]`).
    ///
    /// The zero skip is not an optimisation with a correctness caveat: 85–95 %
    /// of lanes are zero after the clamp, and dropping a term whose value is
    /// exactly `0.0 · w` leaves the ORDER of every surviving term untouched, so
    /// the result is bit-identical to the dense loop for all finite weights.
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
            // CReLU — clamp(h, 0, 1), the trainer's `_crelu`. `f32::clamp`
            // panics on a NaN bound, never on a NaN value; a NaN activation
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
    /// The OPTIONAL disagreement head ([`DisHead`]), from the container's DIS1
    /// section. `None` for every net without one — and for a net WITH one the
    /// value path is still bit-identical, because nothing on it reads this
    /// field: [`SotaHead::disagreement`] is the only accessor.
    pub dis_head: Option<DisHead>,
    /// Per-accumulator-lane dequantisation multiplier for the **i8 HalfKA
    /// feature transformer** (perf study 32), `[L1]`.
    ///
    /// The HalfKA table is stored on disk as i16 in the `FT_QUANT = 255` frame
    /// and quantised to i8 AT LOAD, one integer scale per accumulator column —
    /// the same shape as [`SkipLane`]'s per-bucket scale, for the same reason:
    /// a column spends its full i8 range on its own dynamic range. The HalfKA
    /// accumulator therefore lives in the reduced frame `255 / ft_scale[j]`,
    /// and [`activate_ft`] multiplies it back before the clamp — one `vpmullw`
    /// per 32 lanes per eval, against a HALVING of the 57.7 MB weight table
    /// that the fold streams cold on every node.
    ///
    /// All-ones for a synthetic head, which makes the reduced frame the
    /// identity and the eval bit-identical to the pre-study-32 path.
    ///
    /// `Blocks<_, L1, 1>` rather than `Aligned<i16>` for the reason study 26 §3
    /// gives: the kernel wants a `&[i16; L1]`, and a `Blocks` states that shape
    /// in the type, so acquiring it is `0 < 1` against a constant instead of a
    /// fat-pointer range check plus a `try_into` length check with a panic edge
    /// in the eval's hot path.
    pub ft_scale: FtScale,
    /// This net's piece-count → output-bucket map: the container's OBL1
    /// section, or [`LEGACY_BUCKET_LAYOUT`] when the container has none —
    /// which is every net trained before 2026-08, kept bit-identical by that
    /// default. The map is trained INTO the head's per-bucket weights, so it
    /// travels with them.
    pub bucket_layout: BucketLayout,
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
    scale: &[u8; L1],
    stm: Color,
    threat_shift: u32,
    out: &mut [u8; L1],
) {
    activate(out, pst, threat, scale, stm as usize, threat_shift);
}

// `policy_trunk` (the float view of the quantized trunk for the D41 policy
// head) was deleted 2026-08-01 with the `policy` feature (simplification plan
// tier 0, item 0.10); a future head that needs the trunk view re-derives it
// through `activate_ft`, which is the shared, drift-proof entry.

/// Convert the independently quantized threat accumulator to the HalfKA scale
/// before summing. This is `round(pst + threat · FT_QUANT/threat_quant)` in
/// quantized space. Adding the raw integers would underweight every threat
/// feature by the frame ratio (4x on a CEN2 net).
#[cfg(test)]
#[inline]
fn merge_transformers(pst: i16, threat: i16, scale: i16, threat_shift: u32) -> i16 {
    // BOTH channels carry a net-dependent rescale into the merge, and they are
    // independent of each other:
    //
    //   threat — the frame ratio is a power of two (a 0.39% gain change),
    //     which is what every SIMD backend emits and what the 800-FEN fidelity
    //     check measured at mean 0.14 SF cp (study 8 §4.6; `threat_shift_for`).
    //   HalfKA — the i8 transformer stores rows at `FT_QUANT / scale[j]`, so
    //     the lane is multiplied back before the clamp (study 32). WRAPPING in
    //     i16 by definition: the vector backends multiply in i16 lanes, so
    //     widening here would disagree with them on the (never observed)
    //     overflow, exactly where the old FT_QUANT-frame accumulator wrapped.
    let scaled = i32::from(threat) << threat_shift;
    (i32::from(pst.wrapping_mul(scale)) + scaled).clamp(0, FT_QUANT) as i16
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

    /// The activated feature-transformer vector left behind by the last
    /// [`SotaHead::forward_into`] on this scratch.
    ///
    /// **This is what makes a disagreement poll affordable.**
    /// [`SotaHead::disagreement`] re-runs `activate_ft` from the accumulators,
    /// which is `i8_into_shaped` — measured at **53.61 % of this engine's
    /// L1-miss latency**, the single hottest thing in the binary. Paying it a
    /// second time roughly doubles eval cost at every polled node, which is
    /// why that method's own doc says a caller polling on a large fraction of
    /// nodes has misunderstood it.
    ///
    /// The value pass already produced this exact vector for this exact
    /// position, so a poll that reuses it costs only the head's dense layers
    /// (768x32 + 32x2 = 24,640 MACs, ~1.9x the value head's own tail) and no
    /// extra memory traffic at all.
    ///
    /// **Validity is the caller's contract**, and it is narrow: the contents
    /// belong to whichever position last ran `forward_into` on this scratch.
    /// Read it only immediately after evaluating the node you are asking
    /// about, before any child is evaluated. There is deliberately no
    /// bookkeeping here to enforce that — a stale-flag would cost the hot path
    /// something to protect a cold caller.
    #[inline]
    pub fn last_ft(&self) -> &[u8; L1] {
        &self.ft
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
    /// The output bucket THIS NET selects for a total piece count — the same
    /// index the forward pass uses, exposed so tests, tools, and the debug
    /// oracle pin the T6c skip against it. A method, not a free function,
    /// because the map is per-net data (see [`BucketLayout`]).
    #[inline]
    pub fn output_bucket(&self, piece_count: usize) -> usize {
        usize::from(self.bucket_layout[piece_count])
    }

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

    /// The disagreement head's prediction for a position, `None` when the net
    /// carries no DIS1 section (every net before 2026-08-03) or when the head
    /// produced a non-finite value.
    ///
    /// **This is an on-demand poll and is NOT called by [`forward`] or
    /// [`forward_into`].** It re-runs `activate_ft` from the caller's
    /// accumulators instead of borrowing the value pass's FT vector, which
    /// costs one extra activation per poll and buys the property the design
    /// wants: the per-node eval path does not change, at all, whether or not a
    /// net has this head. A caller polling it on a large fraction of nodes has
    /// misunderstood it — the accumulators it needs are already there, so the
    /// FT vector could be shared if we ever wanted that, and the deliberate
    /// choice is not to.
    ///
    /// The `[0u8; L1]` local is exactly the memset perf study 15 arm 4 removed
    /// from the hot path. It stays here on purpose: at poll rates this head is
    /// meant for, 768 bytes of `memset` is free, and NOT threading a second
    /// scratch through every caller keeps the hot-path types unchanged.
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
        // Early-out BEFORE the activation, not after: on a net with no DIS1
        // section this whole call is one null check, so a caller may poll
        // unconditionally without branching on the net's shape.
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
    /// vector — the affordable form of the poll.
    ///
    /// Pair it with [`HeadScratch::last_ft`], whose doc carries the cost
    /// argument and the validity contract: the vector must belong to the
    /// position being asked about. Same `None` semantics as its sibling — a
    /// net without a DIS1 section, or a non-finite result, yields `None`
    /// rather than a NaN that would silently poison a search margin.
    #[inline]
    pub fn disagreement_from_ft(&self, ft: &[u8; L1]) -> Option<[f32; DIS_OUTS]> {
        let dis = self.dis_head.as_ref()?;
        let out = dis.forward(ft);
        // Same guard, and the same rationale, as `sigma`: a consumer must get
        // `None` rather than a NaN that silently poisons a search margin.
        out.iter().all(|v| v.is_finite()).then_some(out)
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
        let HeadScratch { ft, nnz } = scratch;
        activate_ft(
            pst,
            threat,
            self.ft_scale.block(0),
            stm,
            self.threat_shift,
            ft,
        );
        // The `bucket` binding is DUPLICATED under the two `cfg`s rather than
        // hoisted once, so that the arm-OFF build keeps its statement order
        // byte-for-byte: hoisting it unconditionally moved one index load
        // across `find_nnz` and changed the control build's `.text` md5
        // (d0dff381 -> caab9574), which would have made every number below a
        // property of the edit rather than of the arm.
        #[cfg(any(feature = "hskip60", feature = "htail60"))]
        let bucket = usize::from(self.bucket_layout[piece_count]);
        // PERF STUDY 60 arm `htail` — hint the head's TAIL tables at the top of
        // the forward pass.
        //
        // This is the ninth software prefetch in this engine and the first at
        // this site. The eight dead ones (studies 4, 21, 29 x3, 43 x4, 44, 51)
        // are all on the FOLD's weight rows, where study 29 sec 5.2 computed the
        // ceiling: a row is a cold single-visit burst taking 22 demand misses,
        // one hint converts at most 1 of 22, and `_MM_HINT_T2` lands in L2 where
        // those rows already hit 72.7 % of the time. **None of that is true
        // here.** The tail tables are 2,368 B at this arch (`l2_weights`
        // 16*32*4 = 2048 + `l2_biases` 128 + `l3_weights` 128 + `l1_biases`
        // 64 -- the same arithmetic that reproduces study 51's "l1_weights
        // 20 KB/bucket"), they are read IN FULL on every
        // evaluation (so a hint cannot be wasted on data that is not touched),
        // study 51's region census measures them at 98.84 % / 100.00 % L1 miss,
        // and their addresses are known at entry while their first use is ~800
        // instructions away at the end of the serial `l1 -> l2 -> l3` chain.
        // `T0`, not `T2`, for the same reason: the miss being converted is an
        // L1 miss.
        //
        // 36 hints per evaluation ~= 15.5 instructions/node, which is why this
        // is an ARM and not an obvious win -- see the panel.
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
                    // sanctioned use and justification as
                    // `TranspositionTable::prefetch` (doc 08 sec 3.15) and
                    // `sota_accumulator::pf43`.
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
        // PERF STUDY 60 arm `hskip` — issue the output-skip lane's 20 cold
        // lines HERE, not after `propagate_l3`.
        //
        // `skip_dot_i8` depends on `ft` and on `sk.q.block(bucket)` only, and
        // both are ready the instant `activate_ft` returns — but the value is
        // consumed at the very END of a strictly serial chain
        // (`l1_dot -> l1_dequant -> l2_tail -> l3_dot`). Left where it was, its
        // reads of a 1280-byte lane that study 51 measured at a **96.27 % L1
        // miss rate** are issued when nothing else is left to overlap them
        // with. Hoisted, they are in flight across the whole matmul.
        //
        // This is memory-level parallelism, not prefetching: no hint
        // instruction, no extra load, the same loads issued ~800 instructions
        // earlier. Bit-exact by construction — the integer dot and the two f32
        // multiplies are unchanged and the f32 ADD into `l3` stays where it
        // was, so no floating-point operation is reordered.
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
            #[cfg(not(feature = "hskip60"))]
            let lane = sk.q.block(bucket);
            #[cfg(not(feature = "hskip60"))]
            let acc = super::sota_simd::skip_dot_i8(&ft[..], lane) as f32 * sk.scale[bucket];
            // `skip_early` is `Some` exactly when `skip_weights` is `Some`, so
            // this arm is unreachable-with-`None` by construction; `expect`
            // rather than `unwrap_or` so a future edit that breaks the pairing
            // fails loudly instead of silently evaluating to zero.
            #[cfg(feature = "hskip60")]
            let acc = skip_early.expect("skip_early is Some whenever skip_weights is");
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
        // A CEN2 net at HalfKA scale 1 (weights that already fit i8): the
        // pre-study-32 frame, so these are the original expectations.
        assert_eq!(merge_transformers(0, 64, 1, 2), 255);
        assert_eq!(merge_transformers(255, -64, 1, 2), 0);
        assert_eq!(merge_transformers(100, 16, 1, 2), 164);
        assert_eq!(merge_transformers(100, -16, 1, 2), 36);
        assert_eq!(merge_transformers(250, 2, 1, 2), 255);
        assert_eq!(merge_transformers(5, -2, 1, 2), 0);
        // The i8 HalfKA scale (study 32): a reduced-frame lane of 60 at scale 4
        // is 240 in the FT_QUANT frame, so the merge must clamp the SCALED
        // value, not the stored one.
        assert_eq!(merge_transformers(60, 0, 4, 2), 240);
        assert_eq!(merge_transformers(60, 8, 4, 2), 255);
        assert_eq!(merge_transformers(-60, 8, 4, 2), 0);
        // The two rescales are INDEPENDENT: same HalfKA lane, three threat
        // frames, and the HalfKA contribution is 240 in every one.
        assert_eq!(merge_transformers(60, 4, 4, 0), 244);
        assert_eq!(merge_transformers(60, 4, 4, 1), 248);
        assert_eq!(merge_transformers(60, 4, 4, 2), 255);
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

    /// A deterministic synthetic head + inputs (no Reckless weights — this is a
    /// transcription regression guard, not a strength net).
    /// The identity HalfKA scale: what a synthetic net (whose weights all fit
    /// i8 unscaled) loads with, so these tests exercise the pre-study-32 frame.
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

    /// The sparse L1 path equals a dense recomputation over ALL groups (zeros
    /// contribute nothing) — catches any `find_nnz` / sparse-indexing slip.
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
        // piece counts 20 and 32 map to different buckets (4 vs 7 on the
        // legacy layout the synthetic head carries).
        assert_eq!(head.output_bucket(20), 4);
        assert_eq!(head.output_bucket(32), 7);
        // stm flip changes which half leads → generally a different value.
        let (w, _, _) = head.forward(&pst, &threat, Color::White, 20, &ZERO_PSQT);
        let (bl, _, _) = head.forward(&pst, &threat, Color::Black, 20, &ZERO_PSQT);
        assert!(w != bl, "side-to-move relativity is wired");
    }

    /// 2026-08 originality audit, item 1: the ceiling expansion is now the
    /// ONLY definition of both layouts. Pin the legacy expansion against the
    /// historical bucket EDGES (the behavior every pre-OBL1 net was trained
    /// under — this is the legacy-default-preservation test), and the ARCHON
    /// expansion against its committed derivation receipt
    /// (`docs/audit/2026-08-originality/output-buckets-derivation.txt`).
    #[test]
    fn bucket_layouts_expand_their_ceilings_exactly() {
        // Legacy partition: 0..=8 | 9..=12 | 13..=16 | 17..=19 | 20..=22 |
        // 23..=25 | 26..=28 | 29..=32. Both edges of every bucket.
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
        // ARCHON partition per the derivation receipt: 0..=8 | 9..=10 |
        // 11..=13 | 14..=15 | 16..=18 | 19..=21 | 22..=25 | 26..=32.
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
        // Structural sanity for any layout the expansion can build: total,
        // monotone with no skipped bucket, covering all BUCKETS.
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

    /// New-layout selection (audit item 1): a head carrying a different
    /// layout selects different buckets through the REAL forward pass, and
    /// piece counts the two layouts agree on stay bit-identical.
    #[test]
    fn forward_reads_the_per_net_bucket_layout() {
        let legacy = synth_head();
        let mut archon = synth_head();
        archon.bucket_layout = ARCHON_BUCKET_LAYOUT;
        let (pst, threat) = synth_accs();
        // Piece count 11: legacy bucket 1, ARCHON bucket 2. The synthetic
        // head's per-bucket tables differ, so the values must differ.
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

        // Arithmetic-shift semantics on a negative difference: −1 >> 8 == −1
        // (floor), never 0 (truncation).
        let mut neg = ZERO_PSQT;
        neg[0][head.output_bucket(20)] = -1;
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

    /// The disagreement head reproduces a HAND-WRITTEN dense reference of the
    /// trainer's expression — `crelu(hl1 · W + b) · Wo + bo` with
    /// `hl1 = ft · FT_DEQUANT` — over the real activation of a synthetic
    /// position. This is what pins the layout (`proj_w[i·proj + j]`,
    /// `out_w[j·DIS_OUTS + k]`): a transposed read still produces plausible
    /// numbers, and only an independent index expression catches it.
    ///
    /// The zero skip must be EXACT, not approximate, which is why this asserts
    /// bit equality rather than a tolerance.
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
        // The hidden-lane accumulation order differs (dense j-outer vs the
        // shipped i-outer), so the two agree to float rounding, not to the bit.
        for k in 0..DIS_OUTS {
            assert!(
                (got[k] - want[k]).abs() <= 1e-5 * (1.0 + want[k].abs()),
                "column {k}: got {} want {}",
                got[k],
                want[k]
            );
        }

        // The zero SKIP, on the other hand, must be exact: same order, terms
        // that are identically zero simply not added.
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

    /// The head is OFF the eval path, and this is the assertion that says so:
    /// installing one must not move `forward`'s value, sigma or WDL by a bit.
    /// A net without a head returns `None` rather than a default.
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

    /// A NaN weight must yield `None`, not a NaN a consumer folds into a search
    /// margin — the same guard, and the same reason, as `sigma`.
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
