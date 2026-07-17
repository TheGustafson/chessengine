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
type FindNnzFn = fn(&[u8; ACC], &mut [u16; ACC / 4]) -> usize;

struct Kernels {
    i16_deltas: ApplyI16Fn,
    i8_deltas: ApplyI8Fn,
    l1_dot: L1DotFn,
    activate: ActivateFn,
    find_nnz: FindNnzFn,
}

static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        // avx512f+bw cover the 512-bit i16/i32 ops; avx512vnni supplies
        // dpbusd (the fused u8×i8 dot-accumulate the L1 layer is shaped
        // for). Every VNNI-era CPU has all three (e.g. Zen 4, Ice Lake+).
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            return avx512::KERNELS;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return avx2::KERNELS;
        }
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

/// Compact the indices of the nonzero four-byte groups of `ft` into `out`
/// (ascending), returning the count — the sparse-group set [`l1_dot`]
/// consumes. Every backend produces the identical index set.
#[inline]
pub(crate) fn find_nnz(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
    (KERNELS.find_nnz)(ft, out)
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

    pub(super) fn find_nnz(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        let mut count = 0;
        for g in 0..ACC / 4 {
            let mut nz = 0u8;
            for k in 0..4 {
                nz |= ft[g * 4 + k];
            }
            if nz != 0 {
                out[count] = g as u16;
                count += 1;
            }
        }
        count
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i8_deltas,
        l1_dot,
        activate,
        find_nnz,
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

    /// `NNZ_COMPACT[m]` = the positions of the set bits of `m`, ascending,
    /// zero-padded to 8 — one 128-bit row of u16 lane offsets per possible
    /// 8-group nonzero mask.
    static NNZ_COMPACT: [[u16; 8]; 256] = {
        let mut table = [[0u16; 8]; 256];
        let mut mask = 0usize;
        while mask < 256 {
            let mut bit = 0;
            let mut n = 0;
            while bit < 8 {
                if mask & (1 << bit) != 0 {
                    table[mask][n] = bit as u16;
                    n += 1;
                }
                bit += 1;
            }
            mask += 1;
        }
        table
    };

    #[target_feature(enable = "avx2")]
    unsafe fn find_nnz_inner(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // Eight 4-byte groups per iteration: compare the u32 view against
        // zero, movemask the eight sign bits into a nonzero bitmask, then
        // append the mask's compacted bit positions (one table row) plus the
        // running group base. Identical index set to the scalar scan, without
        // its 192 data-dependent branches.
        // SAFETY: each load reads 32 in-bounds FT bytes. Each store writes a
        // full 8-lane row at `out[count]`: entering chunk `c`, count <= 8c <=
        // ACC/4 - 8, so the row fits even when every group so far was
        // nonzero; the lanes past the popcount are dead padding overwritten
        // by the next store or ignored via the returned count.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut count = 0usize;
            for chunk in 0..ACC / 4 / 8 {
                let groups = _mm256_loadu_si256(ft.as_ptr().add(chunk * 32).cast());
                let is_zero = _mm256_cmpeq_epi32(groups, zero);
                let mask = !(_mm256_movemask_ps(_mm256_castsi256_ps(is_zero)) as u32) & 0xff;
                let row = _mm_loadu_si128(NNZ_COMPACT[mask as usize].as_ptr().cast());
                let base = _mm_set1_epi16((chunk * 8) as i16);
                _mm_storeu_si128(out.as_mut_ptr().add(count).cast(), _mm_add_epi16(row, base));
                count += mask.count_ones() as usize;
            }
            count
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

    pub(super) fn find_nnz(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // SAFETY: installed only after runtime AVX2 detection (the avx512
        // kernel set reuses this too — every avx512 CPU has AVX2).
        unsafe { find_nnz_inner(ft, out) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i8_deltas,
        l1_dot,
        activate,
        find_nnz,
    };
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    const LANES: usize = 32;
    const _: () = assert!(ACC.is_multiple_of(LANES));

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i16_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: `off` covers one in-bounds 32-lane accumulator block.
        // Feature indices originate from validated feature maps and each
        // transformer is loader-validated to contain complete ACC-wide rows.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm512_loadu_si512(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm512_add_epi16(value, _mm512_loadu_si512(row.cast()));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm512_sub_epi16(value, _mm512_loadu_si512(row.cast()));
                }
                _mm512_storeu_si512(acc.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as above. Each 256-bit row load contains 32 in-bounds i8
        // weights and is sign-extended to the accumulator's 32 i16 lanes.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm512_loadu_si512(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    let bytes = _mm256_loadu_si256(row.cast());
                    value = _mm512_add_epi16(value, _mm512_cvtepi8_epi16(bytes));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    let bytes = _mm256_loadu_si256(row.cast());
                    value = _mm512_sub_epi16(value, _mm512_cvtepi8_epi16(bytes));
                }
                _mm512_storeu_si512(acc.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn l1_dot_inner(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // The 16 i32 outputs are exactly one 512-bit register, and dpbusd is
        // the whole kernel: for each dword lane it accumulates the u8×i8
        // four-product dot directly into i32 — the maddubs+madd pair (and its
        // i16 saturation reasoning) collapses into one instruction per group.
        // SAFETY: active groups address four FT bytes and sixteen complete
        // four-byte output rows in loader-validated storage.
        unsafe {
            let mut sums = _mm512_loadu_si512(pre.as_ptr().cast());
            for &group in nnz {
                let group = usize::from(group);
                let inputs = _mm512_set1_epi32(i32::from_le_bytes([
                    ft[group * 4],
                    ft[group * 4 + 1],
                    ft[group * 4 + 2],
                    ft[group * 4 + 3],
                ]));
                let rows = _mm512_loadu_si512(weights.as_ptr().add(group * 64).cast());
                sums = _mm512_dpbusd_epi32(sums, inputs, rows);
            }
            _mm512_storeu_si512(pre.as_mut_ptr().cast(), sums);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn activate_inner(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        stm: usize,
    ) {
        const H: usize = ACC / 2;
        // SAFETY: each iteration consumes sixteen in-bounds i16 lanes from
        // each half. The i32 arithmetic exactly implements signed nearest
        // rounding, clamp, pairwise multiply, and shift from the scalar
        // reference; there is no 512-bit `sign` instruction, so the sign
        // reapplication uses a compare mask + masked subtract-from-zero.
        unsafe {
            debug_assert_eq!(THREAT_QUANT, 64);
            let scale = _mm512_set1_epi32(FT_QUANT);
            let rounding = _mm512_set1_epi32(THREAT_QUANT / 2);
            let zero = _mm512_setzero_si512();
            let ceiling = _mm512_set1_epi32(FT_QUANT);
            for flip in 0..2 {
                let perspective = stm ^ flip;
                let p = &pst[perspective];
                let t = &threat[perspective];
                for lane in (0..H).step_by(16) {
                    let merge = |offset: usize| {
                        let pv = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                            p.as_ptr().add(offset).cast(),
                        ));
                        let tv = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                            t.as_ptr().add(offset).cast(),
                        ));
                        let scaled = _mm512_mullo_epi32(tv, scale);
                        let magnitude = _mm512_srai_epi32::<6>(_mm512_add_epi32(
                            _mm512_abs_epi32(scaled),
                            rounding,
                        ));
                        let negative = _mm512_cmplt_epi32_mask(scaled, zero);
                        let signed = _mm512_mask_sub_epi32(magnitude, negative, zero, magnitude);
                        _mm512_min_epi32(
                            _mm512_max_epi32(_mm512_add_epi32(pv, signed), zero),
                            ceiling,
                        )
                    };
                    let left = merge(lane);
                    let right = merge(lane + H);
                    let product =
                        _mm512_srai_epi32::<{ FT_SHIFT as u32 }>(_mm512_mullo_epi32(left, right));
                    // vpmovdb truncates each i32 to its low byte — the scalar
                    // reference's `as u8`.
                    let bytes = _mm512_cvtepi32_epi8(product);
                    _mm_storeu_si128(out.as_mut_ptr().add(flip * H + lane).cast(), bytes);
                }
            }
        }
    }

    fn i16_deltas(acc: &mut [i16; ACC], weights: &[i16], adds: &[usize], subs: &[usize]) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i16_deltas_inner(acc, weights, adds, subs) }
    }

    fn i8_deltas(acc: &mut [i16; ACC], weights: &[i8], adds: &[usize], subs: &[usize]) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i8_deltas_inner(acc, weights, adds, subs) }
    }

    fn l1_dot(pre: &mut [i32; 16], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { l1_dot_inner(pre, ft, nnz, weights) }
    }

    fn activate(out: &mut [u8; ACC], pst: &[[i16; ACC]; 2], threat: &[[i16; ACC]; 2], stm: usize) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { activate_inner(out, pst, threat, stm) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i8_deltas,
        l1_dot,
        activate,
        // The AVX2 movemask+compact form; a 512-bit variant needs
        // vpcompressw (avx512-vbmi2, outside our f+bw+vnni gate) to beat it.
        find_nnz: avx2::find_nnz,
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

    /// Every backend the CPU supports must match scalar on all four kernels —
    /// the dispatched-vs-scalar tests above only cover whichever backend the
    /// dispatcher picked (on an AVX-512 box, AVX2 would otherwise go dark).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn every_available_backend_matches_scalar() {
        let mut backends: Vec<(&str, &Kernels)> = Vec::new();
        if std::arch::is_x86_feature_detected!("avx2") {
            backends.push(("avx2", &avx2::KERNELS));
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            backends.push(("avx512", &avx512::KERNELS));
        }

        // Shared fixtures spanning the wrapping / saturation corners.
        let mut i16_weights = vec![0i16; 7 * ACC];
        let mut i8_weights = vec![0i8; 7 * ACC];
        for (i, value) in i16_weights.iter_mut().enumerate() {
            *value = (i as u16).wrapping_mul(40503) as i16;
        }
        for (i, value) in i8_weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds = [0, 2, 2, 6, 1];
        let subs = [3, 5, 0, 3];

        let mut ft = [0u8; ACC];
        for (i, value) in ft.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(113) & 127;
        }
        let mut l1_weights = vec![0i8; (ACC / 4) * 64];
        for (i, value) in l1_weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(197) as i8;
        }
        let nnz: Vec<u16> = (0..ACC / 4)
            .filter(|group| group % 3 != 0)
            .map(|group| group as u16)
            .collect();
        let pre_seed: [i32; 16] = std::array::from_fn(|i| i as i32 * 123_457 - 700_000);

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

        for (name, kernels) in backends {
            let mut reference = seed;
            scalar::i16_deltas(&mut reference, &i16_weights, &adds, &subs);
            let mut got = seed;
            (kernels.i16_deltas)(&mut got, &i16_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i16_deltas");

            for fixture in nnz_fixtures() {
                let mut reference = [0u16; ACC / 4];
                let ref_count = scalar::find_nnz(&fixture, &mut reference);
                let mut got = [0u16; ACC / 4];
                let got_count = (kernels.find_nnz)(&fixture, &mut got);
                assert_eq!(got_count, ref_count, "{name} find_nnz count");
                assert_eq!(
                    got[..got_count],
                    reference[..ref_count],
                    "{name} find_nnz indices"
                );
            }

            let mut reference = seed;
            scalar::i8_deltas(&mut reference, &i8_weights, &adds, &subs);
            let mut got = seed;
            (kernels.i8_deltas)(&mut got, &i8_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i8_deltas");

            let mut reference = pre_seed;
            scalar::l1_dot(&mut reference, &ft, &nnz, &l1_weights);
            let mut got = pre_seed;
            (kernels.l1_dot)(&mut got, &ft, &nnz, &l1_weights);
            assert_eq!(got, reference, "{name} l1_dot");

            for stm in 0..2 {
                let mut reference = [0u8; ACC];
                scalar::activate(&mut reference, &pst, &threat, stm);
                let mut got = [0u8; ACC];
                (kernels.activate)(&mut got, &pst, &threat, stm);
                assert_eq!(got, reference, "{name} activate stm {stm}");
            }
        }
    }

    /// FT fixtures spanning the nnz corners: empty, dense, lone groups at the
    /// array edges, a nonzero byte in each of a group's four positions, and
    /// deterministic pseudo-random patterns at realistic (~10%) and adversarial
    /// (~50%) densities.
    fn nnz_fixtures() -> Vec<[u8; ACC]> {
        let mut fixtures = vec![[0u8; ACC], [1u8; ACC]];
        for group in [0, 1, ACC / 4 / 2, ACC / 4 - 2, ACC / 4 - 1] {
            for lane in 0..4 {
                let mut ft = [0u8; ACC];
                ft[group * 4 + lane] = 200;
                fixtures.push(ft);
            }
        }
        for (keep_mod, salt) in [(11, 3u32), (2, 7u32)] {
            let mut ft = [0u8; ACC];
            let mut rng = salt;
            for group in 0..ACC / 4 {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if (rng >> 16) % keep_mod == 0 {
                    ft[group * 4 + ((rng >> 8) % 4) as usize] = (rng % 255 + 1) as u8;
                }
            }
            fixtures.push(ft);
        }
        fixtures
    }

    #[test]
    fn dispatched_find_nnz_matches_scalar_across_sparsity() {
        for fixture in nnz_fixtures() {
            let mut reference = [0u16; ACC / 4];
            let ref_count = scalar::find_nnz(&fixture, &mut reference);
            let mut got = [0u16; ACC / 4];
            let got_count = find_nnz(&fixture, &mut got);
            assert_eq!(got_count, ref_count);
            assert_eq!(got[..got_count], reference[..ref_count]);
        }
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
