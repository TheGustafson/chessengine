//! Fused accumulator kernels for the 768-wide SOTA feature transformers.
//!
//! Keeping the accumulator in registers while applying all feature deltas is
//! substantially cheaper than reading and writing all 768 lanes once per
//! changed feature. The scalar backend is the bit-exact reference; AVX2 is
//! selected once at startup and preserves wrapping i16 arithmetic.

use super::sota_features::ACC;
use super::sota_head::{FT_QUANT, FT_SHIFT, THREAT_QUANT};
use std::sync::LazyLock;

type ApplyI16Fn = fn(&mut [i16; ACC], &[i16], &[usize], &[usize]);
type ApplyI8Fn = fn(&mut [i16; ACC], &[i8], &[usize], &[usize]);
type L1DotFn = fn(&mut [i32; 16], &[u8; ACC], &[u16], &[i8]);
type ActivateFn = fn(&mut [u8; ACC], &[[i16; ACC]; 2], &[[i16; ACC]; 2], usize);

struct Kernels {
    i16_deltas: ApplyI16Fn,
    i8_deltas: ApplyI8Fn,
    l1_dot: L1DotFn,
    activate: ActivateFn,
}

static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return avx2::KERNELS;
    }
    scalar::KERNELS
});

#[inline]
pub(crate) fn apply_i16_deltas(
    acc: &mut [i16; ACC],
    weights: &[i16],
    adds: &[usize],
    subs: &[usize],
) {
    (KERNELS.i16_deltas)(acc, weights, adds, subs);
}

#[inline]
pub(crate) fn apply_i8_deltas(
    acc: &mut [i16; ACC],
    weights: &[i8],
    adds: &[usize],
    subs: &[usize],
) {
    (KERNELS.i8_deltas)(acc, weights, adds, subs);
}

/// Sparse `[u8; 768] × [i8; 16]` dot product. `weights` is laid out as
/// `[group][output][4]`, and `nnz` contains active four-byte group indices.
#[inline]
pub(crate) fn l1_dot(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
    (KERNELS.l1_dot)(pre, ft, nnz, weights);
}

#[inline]
pub(crate) fn activate(
    out: &mut [u8; ACC],
    pst: &[[i16; ACC]; 2],
    threat: &[[i16; ACC]; 2],
    stm: usize,
) {
    (KERNELS.activate)(out, pst, threat, stm);
}

mod scalar {
    use super::*;

    pub(super) fn i16_deltas(
        acc: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        for lane in 0..ACC {
            let mut value = acc[lane];
            for &feature in adds {
                value = value.wrapping_add(weights[feature * ACC + lane]);
            }
            for &feature in subs {
                value = value.wrapping_sub(weights[feature * ACC + lane]);
            }
            acc[lane] = value;
        }
    }

    pub(super) fn i8_deltas(acc: &mut [i16; ACC], weights: &[i8], adds: &[usize], subs: &[usize]) {
        for lane in 0..ACC {
            let mut value = acc[lane];
            for &feature in adds {
                value = value.wrapping_add(i16::from(weights[feature * ACC + lane]));
            }
            for &feature in subs {
                value = value.wrapping_sub(i16::from(weights[feature * ACC + lane]));
            }
            acc[lane] = value;
        }
    }

    pub(super) fn l1_dot(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        for &group in nnz {
            let group = usize::from(group);
            for output in 0..16 {
                let mut dot = 0i32;
                for lane in 0..4 {
                    dot += i32::from(ft[group * 4 + lane])
                        * i32::from(weights[group * 64 + output * 4 + lane]);
                }
                pre[output] += dot;
            }
        }
    }

    pub(super) fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        stm: usize,
    ) {
        const H: usize = ACC / 2;
        for flip in 0..2 {
            let perspective = stm ^ flip;
            for lane in 0..H {
                let merge = |p: i16, t: i16| {
                    let scaled = i32::from(t) * FT_QUANT;
                    let rounded =
                        scaled.signum() * ((scaled.abs() + THREAT_QUANT / 2) / THREAT_QUANT);
                    (i32::from(p) + rounded).clamp(0, FT_QUANT)
                };
                let left = merge(pst[perspective][lane], threat[perspective][lane]);
                let right = merge(pst[perspective][lane + H], threat[perspective][lane + H]);
                out[lane + flip * H] = ((left * right) >> FT_SHIFT) as u8;
            }
        }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i8_deltas,
        l1_dot,
        activate,
    };
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    const LANES: usize = 16;
    const _: () = assert!(ACC.is_multiple_of(LANES));

    #[target_feature(enable = "avx2")]
    unsafe fn i16_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: `off` covers one in-bounds 16-lane accumulator block.
        // Feature indices originate from validated feature maps and each
        // transformer is loader-validated to contain complete ACC-wide rows.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm256_loadu_si256(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm256_add_epi16(value, _mm256_loadu_si256(row.cast()));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm256_sub_epi16(value, _mm256_loadu_si256(row.cast()));
                }
                _mm256_storeu_si256(acc.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn i8_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as above. Each 128-bit row load contains 16 in-bounds i8
        // weights and is sign-extended to the accumulator's 16 i16 lanes.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm256_loadu_si256(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    let bytes = _mm_loadu_si128(row.cast());
                    value = _mm256_add_epi16(value, _mm256_cvtepi8_epi16(bytes));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    let bytes = _mm_loadu_si128(row.cast());
                    value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(bytes));
                }
                _mm256_storeu_si256(acc.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l1_dot_inner(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // maddubs forms adjacent two-product sums. With ft<=127 and an i8
        // weight those sums fit i16 at both extrema; madd_epi16(..., 1) then
        // widens each four-product dot to i32, matching Bullet exactly.
        // SAFETY: active groups address four FT bytes and sixteen complete
        // four-byte output rows in loader-validated storage.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut lo = _mm256_loadu_si256(pre.as_ptr().cast());
            let mut hi = _mm256_loadu_si256(pre.as_ptr().add(8).cast());
            for &group in nnz {
                let group = usize::from(group);
                let inputs = _mm256_set1_epi32(i32::from_le_bytes([
                    ft[group * 4],
                    ft[group * 4 + 1],
                    ft[group * 4 + 2],
                    ft[group * 4 + 3],
                ]));
                let base = weights.as_ptr().add(group * 64);
                let wlo = _mm256_loadu_si256(base.cast());
                let whi = _mm256_loadu_si256(base.add(32).cast());
                let plo = _mm256_maddubs_epi16(inputs, wlo);
                let phi = _mm256_maddubs_epi16(inputs, whi);
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(plo, ones));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(phi, ones));
            }
            _mm256_storeu_si256(pre.as_mut_ptr().cast(), lo);
            _mm256_storeu_si256(pre.as_mut_ptr().add(8).cast(), hi);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn activate_inner(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        stm: usize,
    ) {
        const H: usize = ACC / 2;
        // SAFETY: each iteration consumes eight in-bounds i16 lanes from each
        // half. The i32 arithmetic exactly implements signed nearest rounding,
        // clamp, pairwise multiply, and shift from the scalar reference.
        unsafe {
            debug_assert_eq!(THREAT_QUANT, 64);
            let scale = _mm256_set1_epi32(FT_QUANT);
            let rounding = _mm256_set1_epi32(THREAT_QUANT / 2);
            let zero = _mm256_setzero_si256();
            let ceiling = _mm256_set1_epi32(FT_QUANT);
            for flip in 0..2 {
                let perspective = stm ^ flip;
                let p = &pst[perspective];
                let t = &threat[perspective];
                for lane in (0..H).step_by(8) {
                    let merge = |offset: usize| {
                        let pv =
                            _mm256_cvtepi16_epi32(_mm_loadu_si128(p.as_ptr().add(offset).cast()));
                        let tv =
                            _mm256_cvtepi16_epi32(_mm_loadu_si128(t.as_ptr().add(offset).cast()));
                        let scaled = _mm256_mullo_epi32(tv, scale);
                        let magnitude = _mm256_srai_epi32::<6>(_mm256_add_epi32(
                            _mm256_abs_epi32(scaled),
                            rounding,
                        ));
                        let signed = _mm256_sign_epi32(magnitude, scaled);
                        _mm256_min_epi32(
                            _mm256_max_epi32(_mm256_add_epi32(pv, signed), zero),
                            ceiling,
                        )
                    };
                    let left = merge(lane);
                    let right = merge(lane + H);
                    let product = _mm256_srai_epi32::<FT_SHIFT>(_mm256_mullo_epi32(left, right));
                    let mut lanes = [0i32; 8];
                    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), product);
                    for (offset, value) in lanes.into_iter().enumerate() {
                        out[flip * H + lane + offset] = value as u8;
                    }
                }
            }
        }
    }

    fn i16_deltas(acc: &mut [i16; ACC], weights: &[i16], adds: &[usize], subs: &[usize]) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i16_deltas_inner(acc, weights, adds, subs) }
    }

    fn i8_deltas(acc: &mut [i16; ACC], weights: &[i8], adds: &[usize], subs: &[usize]) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i8_deltas_inner(acc, weights, adds, subs) }
    }

    fn l1_dot(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { l1_dot_inner(pre, ft, nnz, weights) }
    }

    fn activate(out: &mut [u8; ACC], pst: &[[i16; ACC]; 2], threat: &[[i16; ACC]; 2], stm: usize) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { activate_inner(out, pst, threat, stm) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i8_deltas,
        l1_dot,
        activate,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatched_i16_and_i8_match_scalar_at_wrapping_extremes() {
        let mut i16_weights = vec![0i16; 7 * ACC];
        let mut i8_weights = vec![0i8; 7 * ACC];
        for (i, value) in i16_weights.iter_mut().enumerate() {
            *value = (i as u16).wrapping_mul(40503) as i16;
        }
        for (i, value) in i8_weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let seed = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds = [0, 2, 2, 6, 1];
        let subs = [3, 5, 0, 3];

        let mut reference = seed;
        scalar::i16_deltas(&mut reference, &i16_weights, &adds, &subs);
        let mut dispatched = seed;
        apply_i16_deltas(&mut dispatched, &i16_weights, &adds, &subs);
        assert_eq!(dispatched, reference);

        let mut reference = seed;
        scalar::i8_deltas(&mut reference, &i8_weights, &adds, &subs);
        let mut dispatched = seed;
        apply_i8_deltas(&mut dispatched, &i8_weights, &adds, &subs);
        assert_eq!(dispatched, reference);
    }

    #[test]
    fn dispatched_l1_dot_matches_i32_scalar_at_extremes() {
        let mut ft = [0u8; ACC];
        for (i, value) in ft.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(113) & 127;
        }
        let mut weights = vec![0i8; (ACC / 4) * 64];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(197) as i8;
        }
        let nnz: Vec<u16> = (0..ACC / 4)
            .filter(|group| group % 3 != 0)
            .map(|group| group as u16)
            .collect();
        let seed = std::array::from_fn(|i| i as i32 * 123_457 - 700_000);
        let mut reference = seed;
        scalar::l1_dot(&mut reference, &ft, &nnz, &weights);
        let mut dispatched = seed;
        l1_dot(&mut dispatched, &ft, &nnz, &weights);
        assert_eq!(dispatched, reference);
    }

    #[test]
    fn dispatched_activation_matches_scalar_for_full_i16_domain_sample() {
        let mut pst = [[0i16; ACC]; 2];
        let mut threat = [[0i16; ACC]; 2];
        for perspective in 0..2 {
            for lane in 0..ACC {
                pst[perspective][lane] = (lane as u16)
                    .wrapping_mul(40503)
                    .wrapping_add(perspective as u16 * 173)
                    as i16;
                threat[perspective][lane] = (lane as u16)
                    .wrapping_mul(49711)
                    .wrapping_add(perspective as u16 * 251)
                    as i16;
            }
        }
        for stm in 0..2 {
            let mut reference = [0u8; ACC];
            scalar::activate(&mut reference, &pst, &threat, stm);
            let mut dispatched = [0u8; ACC];
            activate(&mut dispatched, &pst, &threat, stm);
            assert_eq!(dispatched, reference);
        }
    }
}
