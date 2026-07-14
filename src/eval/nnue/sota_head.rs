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

use crate::types::Color;

/// Accumulator width per perspective (Reckless-proven; an SPRT knob, D40).
pub const L1: usize = 768;
/// L1→L2 output width.
pub const L2: usize = 16;
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
}

/// Pairwise-multiply FT activation (design/11 §12.3 step 1): sum the two
/// accumulators, clamp each half to `[0, FT_QUANT]`, multiply first×second half
/// and `>> FT_SHIFT` → u8 ∈ [0,127]. STM half occupies `[0, L1/2)`. The
/// `wrapping_add` reproduces the reference's wrapping i16 accumulator math.
fn activate_ft(pst: &[[i16; L1]; 2], threat: &[[i16; L1]; 2], stm: Color) -> [u8; L1] {
    const H: usize = L1 / 2; // 384
    let mut out = [0u8; L1];
    for flip in 0..2usize {
        let persp = (stm as usize) ^ flip;
        let p = &pst[persp];
        let t = &threat[persp];
        for i in 0..H {
            let left = merge_transformers(p[i], t[i]);
            let right = merge_transformers(p[i + H], t[i + H]);
            out[i + flip * H] = ((left as i32 * right as i32) >> FT_SHIFT) as u8;
        }
    }
    out
}

/// Convert the independently quantized threat accumulator to the HalfKA scale
/// before summing. This is `round(pst + threat * 255/64)` in quantized space.
/// Adding the raw integers would underweight every threat feature by ~4x.
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
/// (design/11 §12.3 step 2) — ~85–95% are zero after the clamp.
fn find_nnz(ft: &[u8; L1]) -> ([u16; L1 / 4], usize) {
    let mut idx = [0u16; L1 / 4];
    let mut count = 0;
    for g in 0..L1 / 4 {
        let mut nz = 0u8;
        for k in 0..4 {
            nz |= ft[g * 4 + k];
        }
        if nz != 0 {
            idx[count] = g as u16;
            count += 1;
        }
    }
    (idx, count)
}

impl SotaHead {
    /// int8 sparse matmul over the nonzero groups → dequantized SCReLU f32[L2]
    /// (design/11 §12.3 step 3). The inner 4-product sum runs in wrapping i16.
    fn propagate_l1(&self, ft: &[u8; L1], nnz: &[u16], bucket: usize) -> [f32; L2] {
        let mut pre = [0i32; L2];
        let wbase = bucket * L2 * L1;
        let wb = &self.l1_weights[wbase..wbase + L2 * L1];
        for &g in nnz {
            let g = g as usize;
            for j in 0..L2 {
                let mut vector: i16 = 0;
                for k in 0..4 {
                    let byte = ft[g * 4 + k] as i16;
                    let weight = wb[g * (L2 * 4) + j * 4 + k] as i16;
                    vector = vector.wrapping_add(byte.wrapping_mul(weight));
                }
                pre[j] += vector as i32;
            }
        }
        let mut out = [0.0f32; L2];
        let bb = bucket * L2;
        for j in 0..L2 {
            let clipped = (pre[j] as f32)
                .mul_add(DEQUANT, self.l1_biases[bb + j])
                .clamp(0.0, 1.0);
            out[j] = clipped * clipped;
        }
        out
    }

    /// Dense f32[L2]→f32[L3], bias-seeded FMA then SCReLU (step 4). The
    /// i-outer/j-inner order is load-bearing: it pins the accumulation order.
    #[allow(clippy::needless_range_loop)]
    fn propagate_l2(&self, l1: &[f32; L2], bucket: usize) -> [f32; L3] {
        let mut out = [0.0f32; L3];
        let bb = bucket * L3;
        out.copy_from_slice(&self.l2_biases[bb..bb + L3]);
        let wbase = bucket * L2 * L3;
        for i in 0..L2 {
            for j in 0..L3 {
                let w = self.l2_weights[wbase + i * L3 + j];
                out[j] = w.mul_add(l1[i], out[j]);
            }
        }
        for v in &mut out {
            let clipped = v.clamp(0.0, 1.0);
            *v = clipped * clipped;
        }
        out
    }

    /// Dense f32[L3]→f32, 16-lane FMA + recursive-halving reduction + bias
    /// (step 5). Reduction order mirrors a 16-lane horizontal add.
    fn propagate_l3(&self, l2: &[f32; L3], bucket: usize) -> f32 {
        const LANES: usize = 16;
        let mut sums = [0.0f32; LANES];
        let wbase = bucket * L3;
        let mut i = 0;
        while i < L3 {
            for j in 0..LANES {
                let w = self.l3_weights[wbase + i + j];
                sums[j] = w.mul_add(l2[i + j], sums[j]);
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

    /// The head forward: two per-perspective accumulators + side to move + total
    /// piece count → stm-relative centipawns. The front-end (task #29 part 2)
    /// produces `pst`/`threat`; this is the shared trunk read for both eval and
    /// the policy heads.
    pub fn forward(
        &self,
        pst: &[[i16; L1]; 2],
        threat: &[[i16; L1]; 2],
        stm: Color,
        piece_count: usize,
    ) -> i32 {
        let ft = activate_ft(pst, threat, stm);
        let (nnz, count) = find_nnz(&ft);
        let bucket = OUTPUT_BUCKETS_LAYOUT[piece_count];
        let l1 = self.propagate_l1(&ft, &nnz[..count], bucket);
        let l2 = self.propagate_l2(&l1, bucket);
        let l3 = self.propagate_l3(&l2, bucket);
        (l3 * self.network_scale as f32) as i32
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
        let l1 = head.propagate_l1(&[0; L1], &[], 0);
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
        let l3w: Vec<f32> = (0..BUCKETS * L3).map(|i| ((i % 11) as f32 - 5.0) * 0.03).collect();
        let l3b: Vec<f32> = (0..BUCKETS).map(|i| i as f32 * 0.1).collect();
        SotaHead {
            l1_weights: l1w.into_boxed_slice(),
            l1_biases: l1b.into_boxed_slice(),
            l2_weights: l2w.into_boxed_slice(),
            l2_biases: l2b.into_boxed_slice(),
            l3_weights: l3w.into_boxed_slice(),
            l3_biases: l3b.into_boxed_slice(),
            network_scale: 380,
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
        let ft = activate_ft(&pst, &threat, Color::White);
        // (150 * 255) >> 9 = 38250 >> 9 = 74
        assert_eq!(ft[0], 74, "pairwise clamp+shift");
        // a negative sum clamps to 0 → product 0
        let mut pst2 = [[0i16; L1]; 2];
        pst2[0][1] = -500;
        let ft2 = activate_ft(&pst2, &threat, Color::White);
        assert_eq!(ft2[1], 0, "negative clamps to 0");
    }

    /// The sparse L1 path equals a dense recomputation over ALL groups (zeros
    /// contribute nothing) — catches any `find_nnz` / sparse-indexing slip.
    #[test]
    fn sparse_l1_equals_dense() {
        let head = synth_head();
        let (pst, threat) = synth_accs();
        let ft = activate_ft(&pst, &threat, Color::White);
        let (nnz, count) = find_nnz(&ft);
        let bucket = 3;
        let sparse = head.propagate_l1(&ft, &nnz[..count], bucket);
        // dense: iterate every group, same wrapping-i16 inner math.
        let mut pre = [0i32; L2];
        let wb = &head.l1_weights[bucket * L2 * L1..bucket * L2 * L1 + L2 * L1];
        for g in 0..L1 / 4 {
            for j in 0..L2 {
                let mut v: i16 = 0;
                for k in 0..4 {
                    v = v.wrapping_add((ft[g * 4 + k] as i16).wrapping_mul(wb[g * (L2 * 4) + j * 4 + k] as i16));
                }
                pre[j] += v as i32;
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
        let a = head.forward(&pst, &threat, Color::White, 20);
        let b = head.forward(&pst, &threat, Color::White, 20);
        assert_eq!(a, b, "deterministic");
        assert!(a.abs() < 100_000, "finite, sane cp");
        // piece counts 20 and 32 map to different buckets (4 vs 7).
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[20], 4);
        assert_eq!(OUTPUT_BUCKETS_LAYOUT[32], 7);
        // stm flip changes which half leads → generally a different value.
        let w = head.forward(&pst, &threat, Color::White, 20);
        let bl = head.forward(&pst, &threat, Color::Black, 20);
        assert!(w != bl, "side-to-move relativity is wired");
    }
}
