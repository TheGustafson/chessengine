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

use super::sota_simd::{activate, l1_dequant, l1_dot, l2_tail};
use crate::types::Color;

/// Accumulator width per perspective — D47 (user-directed, 2026-07-22):
/// the shared-bottleneck multihead arch. 1536/perspective (threats retained),
/// L2=256 = the shared bottleneck where heads branch (value tail continues
/// 256→32→1×buckets; the 1858-row policy head dots the 256 — see
/// `policy_vocab`). "Go big first — learnability before NPS" is the pinned
/// research posture; the 768/L2=16 incumbent remains the fallback ship
/// (gate old nets with engines/bin/archon-sota-native.prev-era binaries).
pub const L1: usize = 1280;
/// L1→L2 output width (the D47 shared bottleneck).
pub const L2: usize = 128;
/// L2→L3 output width.
pub const L3: usize = 32;
/// Material/output buckets.
pub const BUCKETS: usize = 8;

pub(crate) const FT_QUANT: i32 = 255; // FT activation clamp ceiling / QA
/// Quantization scale of the separately exported FullThreats transformer.
/// CENN-v2 fixes this as an architecture constant so its weights fit in i8.
pub(crate) const THREAT_QUANT: i32 = 64;
pub(crate) const L1_QUANT: i32 = 64; // L1 weight scale / QB
pub(crate) const FT_SHIFT: i32 = 9; // pairwise-multiply right shift
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

/// The head's per-bucket weights + the net's output scale. Format-agnostic: the
/// CENN-v2 loader (task #29) fills these from disk; tests fill them synthetically.
/// Layouts are bucket-major, flattened.
#[derive(Debug)]
pub struct SotaHead {
    /// `[BUCKETS][L2 · L1]` i8.
    pub l1_weights: Box<[i8]>,
    /// `[BUCKETS][L2]` f32.
    pub l1_biases: Box<[f32]>,
    /// `[BUCKETS][L2 · L3]` f32.
    pub l2_weights: Box<[f32]>,
    /// `[BUCKETS][L3]` f32.
    pub l2_biases: Box<[f32]>,
    /// `[BUCKETS][L3]` f32.
    pub l3_weights: Box<[f32]>,
    /// `[BUCKETS]` f32.
    pub l3_biases: Box<[f32]>,
    /// f32→centipawn scale (Reckless 380; ours is baked at training, per-net).
    pub network_scale: i32,
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
}

/// Pairwise-multiply FT activation (design/11 §12.3 step 1): sum the two
/// accumulators, clamp each half to `[0, FT_QUANT]`, multiply first×second half
/// and `>> FT_SHIFT` → u8 ∈ [0,127]. STM half occupies `[0, L1/2)`. The
/// `wrapping_add` reproduces the reference's wrapping i16 accumulator math.
/// Out-param form (perf study #4 item #5): the kernel writes the caller's
/// buffer directly instead of returning the 1280 B array by value.
fn activate_ft(pst: &[[i16; L1]; 2], threat: &[[i16; L1]; 2], stm: Color, out: &mut [u8; L1]) {
    activate(out, pst, threat, stm as usize);
}

/// Float view of the quantized shared D40 trunk for the optional policy head.
/// The trainer's pairwise CReLU produces values in `[0, 1]`; the runtime FT
/// stores the same product after `>> FT_SHIFT`, so undo that scale here.  This
/// deliberately shares [`activate_ft`] with eval to prevent the two consumers
/// from drifting on perspective order, threat scaling, or clamp semantics.
#[cfg(feature = "policy")]
pub fn policy_trunk(pst: &[[i16; L1]; 2], threat: &[[i16; L1]; 2], stm: Color) -> [f32; L1] {
    let mut quantized = [0u8; L1];
    activate_ft(pst, threat, stm, &mut quantized);
    let scale = (1i32 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT) as f32;
    quantized.map(|value| f32::from(value) * scale)
}

/// Convert the independently quantized threat accumulator to the HalfKA scale
/// before summing. This is `round(pst + threat * 255/64)` in quantized space.
/// Adding the raw integers would underweight every threat feature by ~4x.
#[cfg(test)]
#[inline]
fn merge_transformers(pst: i16, threat: i16) -> i16 {
    let scaled = i32::from(threat) * FT_QUANT;
    let rounded = if scaled >= 0 {
        (scaled + THREAT_QUANT / 2) / THREAT_QUANT
    } else {
        (scaled - THREAT_QUANT / 2) / THREAT_QUANT
    };
    (i32::from(pst) + rounded).clamp(0, FT_QUANT) as i16
}

/// Indices of the 4-byte groups (of `L1/4`) holding any nonzero activation
/// (design/11 §12.3 step 2) — ~85–95% are zero after the clamp. Dispatched:
/// the scalar scan's 192 data-dependent branches mispredict on the sparse
/// pattern, so the SIMD backends compact via movemask instead (R4 #2).
/// Out-param form (perf study #4 item #5): no 640 B by-value return.
fn find_nnz(ft: &[u8; L1], idx: &mut [u16; L1 / 4]) -> usize {
    super::sota_simd::find_nnz(ft, idx)
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
        let wbase = bucket * L2 * L1;
        let wb = &self.l1_weights[wbase..wbase + L2 * L1];
        l1_dot(&mut pre, ft, nnz, wb);
        let bb = bucket * L2;
        let bias: &[f32; L2] = self.l1_biases[bb..bb + L2]
            .try_into()
            .expect("l1 bias block");
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
        let bb = bucket * L3;
        let bias: &[f32; L3] = self.l2_biases[bb..bb + L3]
            .try_into()
            .expect("l2 bias block");
        let wbase = bucket * L2 * L3;
        let wblock: &[f32; L2 * L3] = self.l2_weights[wbase..wbase + L2 * L3]
            .try_into()
            .expect("l2 weight block");
        l2_tail(&mut out, l1, wblock, bias);
        out
    }

    /// Dense f32[L3]→f32, 16-lane FMA + recursive-halving reduction + bias
    /// (step 5). Reduction order mirrors a 16-lane horizontal add.
    fn propagate_l3(&self, l2: &[f32; L3], bucket: usize) -> f32 {
        const LANES: usize = 16;
        let mut sums = [0.0f32; LANES];
        let wbase = bucket * L3;
        // Fixed-size bucket + per-lane slices hoist the bounds checks so the
        // 16-wide FMA vectorizes; same lane assignment ⇒ bit-identical.
        let weights: &[f32; L3] = self.l3_weights[wbase..wbase + L3]
            .try_into()
            .expect("l3 weight block");
        let mut i = 0;
        while i < L3 {
            let wl: &[f32; LANES] = weights[i..i + LANES].try_into().expect("l3 weight lane");
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
        sums[0] + self.l3_biases[bucket]
    }

    /// EXP-UNC-1: the log-variance head's dense f32[L3]→f32, mirroring
    /// [`propagate_l3`](Self::propagate_l3) exactly (same lane layout + reduction
    /// order) but reading the variance weights. Returns raw `s = log(σ²)`, the
    /// log-variance of the SIGMOID (WDL) value — the NLL trains it on the
    /// `(sigmoid(out) − target)` residual, so σ̂ = exp(½·s) is a WDL-space std,
    /// NOT a cp-space one (D33 review, reviewer B: mind this when it scales the
    /// cp-space RFP margin). Caller guarantees the variance head is present.
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
    /// shared trunk read for both eval and the policy heads. `sigma` is `Some`
    /// only when the net carries the EXP-UNC-1 log-variance head; it is the
    /// per-position value std in sigmoid (WDL) space, `σ̂ = exp(½·log σ²)`,
    /// computed from the SAME `l2` activations at ~32 extra FLOPs (T ≈ 1).
    pub fn forward(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
    ) -> (i32, Option<f32>, Option<[f32; 3]>) {
        let mut l1 = [0.0f32; L2];
        self.forward_into(pst, threat, stm, piece_count, &mut l1)
    }

    /// [`forward`](Self::forward), writing the shared-bottleneck activation
    /// (the L2-wide output the D47 policy head dots against) into `l1_out` —
    /// an out-param so the policy stash lands directly in the caller's slot
    /// with no by-value copy and no discarded return on the `forward` path
    /// (perf study #4 item #5; the pre-restructure `forward` called
    /// `forward_full(...).0` and threw the 512 B array away).
    pub fn forward_into(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
        l1_out: &mut [f32; L2],
    ) -> (i32, Option<f32>, Option<[f32; 3]>) {
        let mut ft = [0u8; L1];
        activate_ft(pst, threat, stm, &mut ft);
        let mut nnz = [0u16; L1 / 4];
        let count = find_nnz(&ft, &mut nnz);
        let bucket = OUTPUT_BUCKETS_LAYOUT[piece_count];
        self.propagate_l1(&ft, &nnz[..count], bucket, l1_out);
        let l1 = l1_out;
        let l2 = self.propagate_l2(l1, bucket);
        let l3 = self.propagate_l3(&l2, bucket);
        let value = (l3 * self.network_scale as f32) as i32;
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

    #[test]
    fn mixed_transformer_scales_are_normalized_before_sum() {
        assert_eq!(merge_transformers(0, 64), 255);
        assert_eq!(merge_transformers(255, -64), 0);
        assert_eq!(merge_transformers(100, 16), 164);
        assert_eq!(merge_transformers(100, -16), 36);
        assert_eq!(merge_transformers(250, 2), 255);
        assert_eq!(merge_transformers(5, -2), 0);
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
            l1_weights: l1w.into_boxed_slice(),
            l1_biases: l1b.into_boxed_slice(),
            l2_weights: l2w.into_boxed_slice(),
            l2_biases: l2b.into_boxed_slice(),
            l3_weights: l3w.into_boxed_slice(),
            l3_biases: l3b.into_boxed_slice(),
            network_scale: 380,
            l3_var_weights: None,
            l3_var_biases: None,
            l3_wdl_weights: None,
            l3_wdl_biases: None,
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
        activate_ft(&pst, &threat, Color::White, &mut ft);
        // (150 * 255) >> 9 = 38250 >> 9 = 74
        assert_eq!(ft[0], 74, "pairwise clamp+shift");
        // a negative sum clamps to 0 → product 0
        let mut pst2 = [[0i16; L1]; 2];
        pst2[0][1] = -500;
        let mut ft2 = [0u8; L1];
        activate_ft(&pst2, &threat, Color::White, &mut ft2);
        assert_eq!(ft2[1], 0, "negative clamps to 0");
    }

    /// The sparse L1 path equals a dense recomputation over ALL groups (zeros
    /// contribute nothing) — catches any `find_nnz` / sparse-indexing slip.
    #[test]
    fn sparse_l1_equals_dense() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let mut ft = [0u8; L1];
        activate_ft(&pst, &threat, Color::White, &mut ft);
        let mut nnz = [0u16; L1 / 4];
        let count = find_nnz(&ft, &mut nnz);
        let bucket = 3;
        let mut sparse = [0.0f32; L2];
        head.propagate_l1(&ft, &nnz[..count], bucket, &mut sparse);
        // Dense: iterate every group with the same mathematical i32 dot.
        let mut pre = [0i32; L2];
        let wb = &head.l1_weights[bucket * L2 * L1..bucket * L2 * L1 + L2 * L1];
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
                .mul_add(DEQUANT, head.l1_biases[bucket * L2 + j])
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
        let (a, _, _) = head.forward(&pst, &threat, Color::White, 20);
        let (b, _, _) = head.forward(&pst, &threat, Color::White, 20);
        assert_eq!(a, b, "deterministic");
        assert!(a.abs() < 100_000, "finite, sane cp");
        // piece counts 20 and 32 map to different buckets (4 vs 7).
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[20], 4);
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[32], 7);
        // stm flip changes which half leads → generally a different value.
        let (w, _, _) = head.forward(&pst, &threat, Color::White, 20);
        let (bl, _, _) = head.forward(&pst, &threat, Color::Black, 20);
        assert!(w != bl, "side-to-move relativity is wired");
    }

    /// EXP-UNC-1: the optional variance head yields a finite, positive σ̂ and
    /// leaves the value (mean) path byte-identical — the head-level form of the
    /// "bit-identical when the uncertainty consumers are off" guarantee. Without
    /// the head, σ̂ is `None`.
    #[test]
    fn variance_head_yields_finite_sigma_without_touching_value() {
        let (pst, threat) = synth_accs();

        let plain = synth_head();
        let (v0, s0, _) = plain.forward(&pst, &threat, Color::White, 20);
        assert!(s0.is_none(), "no variance head => no sigma");

        let mut head = synth_head();
        let vw: Vec<f32> = (0..BUCKETS * L3)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let vb: Vec<f32> = (0..BUCKETS).map(|i| i as f32 * 0.01 - 0.02).collect();
        head.l3_var_weights = Some(vw.into_boxed_slice());
        head.l3_var_biases = Some(vb.into_boxed_slice());
        let (v1, s1, _) = head.forward(&pst, &threat, Color::White, 20);

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
        head.l3_var_weights = Some(head.l3_weights.clone());
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
