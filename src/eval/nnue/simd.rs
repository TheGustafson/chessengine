//! The compute kernels behind the accumulator and forward pass (design/11
//! §3.4.6): [`scalar`] (plain safe Rust, i64 forward — the bit-exact
//! reference) and, on x86_64, [`avx2`]. [`KERNELS`] runtime-detects AVX2
//! once at startup; the AVX2 backend must reproduce the scalar reference
//! lane-for-lane (pinned by the `avx2_bit_exact` tests).
//!
//! **The AVX2 backend is the crate's sanctioned SIMD `unsafe` (D14b).** All
//! `unsafe` is contained inside [`avx2`] with `// SAFETY:` on every block;
//! the [`Kernels`] table holds plain **safe `fn` pointers** — the AVX2
//! kernels are `#[target_feature] unsafe fn` reached only through thin safe
//! wrappers installed after AVX2 detection, so no `unsafe` leaks past this
//! module.

use super::network::Activation;
use super::{L1, QA, QB, SCALE};
use crate::types::Value;
use std::sync::LazyLock;

/// 64-byte-aligned wrapper for accumulator storage: the *stored* accumulator
/// never straddles a cache line (the kernels still use alignment-agnostic
/// `loadu`, since temporaries and FT rows are only 2-aligned).
///
/// `Deref`/`DerefMut` expose the inner value (`acc[color]`, not `acc.0[color]`).
#[repr(align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aligned64<T>(pub T);

impl<T> std::ops::Deref for Aligned64<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Aligned64<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// `dst = src + W0[add_row] − W0[sub_row]` (quiet / quiet-promotion).
pub type AddSubFn =
    fn(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]);
/// `dst = src + W0[add] − W0[sub0] − W0[sub1]` (capture / ep / capture-promo).
pub type AddSubSubFn =
    fn(dst: &mut [i16; L1], src: &[i16; L1], add: &[i16; L1], sub0: &[i16; L1], sub1: &[i16; L1]);
/// `dst = src + W0[add0] + W0[add1] − W0[sub0] − W0[sub1]` (castling).
pub type Add2Sub2Fn = fn(
    dst: &mut [i16; L1],
    src: &[i16; L1],
    add0: &[i16; L1],
    add1: &[i16; L1],
    sub0: &[i16; L1],
    sub1: &[i16; L1],
);
/// `dst = bias + Σ_f W0[f]`: a from-scratch accumulator refresh.
pub type RefreshFn = fn(dst: &mut [i16; L1], bias: &[i16; L1], rows: &[&[i16; L1]]);
/// The quantized output forward pass (design/11 §3.4.3): `us`/`them` are the
/// stm/nstm accumulators, `w1` the `2·L1` output weights (stm half first).
pub type ForwardFn = fn(
    us: &[i16; L1],
    them: &[i16; L1],
    w1: &[i16; 2 * L1],
    out_bias: i32,
    act: Activation,
) -> Value;

/// The compute kernels the accumulator and forward pass dispatch through
/// (design/11 §3.4.6). Safe `fn` pointers in both backends (module docs).
pub struct Kernels {
    pub add_sub: AddSubFn,
    pub add_sub_sub: AddSubSubFn,
    pub add2_sub2: Add2Sub2Fn,
    pub refresh: RefreshFn,
    pub forward: ForwardFn,
}

/// The active kernel backend, selected once: AVX2 where available.
pub static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: avx2 verified at startup via is_x86_feature_detected!
        // (D14b). Every `avx2::KERNELS` wrapper's precondition is exactly
        // "this CPU has AVX2"; this single site establishes it process-wide.
        if std::arch::is_x86_feature_detected!("avx2") {
            return avx2::KERNELS;
        }
    }
    scalar::KERNELS
});

/// The scalar reference backend (plain safe Rust); the forward accumulates
/// in `i64`.
pub mod scalar {
    use super::*;

    /// `dst = src + add_row − sub_row`, column-wise. `wrapping_add`/`_sub`
    /// match the i16 lane wrap of the AVX2 `_mm256_add/sub_epi16`, so the two
    /// backends agree bit-for-bit even on a synthetic net that overflows.
    pub fn add_sub(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]) {
        for i in 0..L1 {
            dst[i] = src[i].wrapping_add(add_row[i]).wrapping_sub(sub_row[i]);
        }
    }

    /// `dst = src + add − sub0 − sub1`, column-wise; same lane-wrap
    /// discipline as [`add_sub`].
    pub fn add_sub_sub(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        for i in 0..L1 {
            dst[i] = src[i]
                .wrapping_add(add[i])
                .wrapping_sub(sub0[i])
                .wrapping_sub(sub1[i]);
        }
    }

    /// `dst = src + add0 + add1 − sub0 − sub1`, column-wise; same lane-wrap
    /// discipline as [`add_sub`].
    pub fn add2_sub2(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add0: &[i16; L1],
        add1: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        for i in 0..L1 {
            dst[i] = src[i]
                .wrapping_add(add0[i])
                .wrapping_add(add1[i])
                .wrapping_sub(sub0[i])
                .wrapping_sub(sub1[i]);
        }
    }

    /// `dst = bias + Σ rows`, column-wise (a from-scratch refresh).
    pub fn refresh(dst: &mut [i16; L1], bias: &[i16; L1], rows: &[&[i16; L1]]) {
        *dst = *bias;
        for row in rows {
            for i in 0..L1 {
                dst[i] = dst[i].wrapping_add(row[i]);
            }
        }
    }

    /// The raw pre-scaling dot-product sum, accumulated in `i64` — the
    /// **bit-exact reference** the AVX2 path must reproduce: SCReLU
    /// `Σ clamp(a,0,QA)²·w`; CReLU drops the `²`; stm half first. Exposed so
    /// the bit-exactness tests can compare backends at the pre-narrowing
    /// width, where an AVX2 i32-lane overflow cannot hide behind the final
    /// `as Value` truncation.
    pub fn forward_raw_sum(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        act: Activation,
    ) -> i64 {
        let mut sum: i64 = 0;
        match act {
            Activation::SCReLU => {
                for i in 0..L1 {
                    let a = (us[i] as i64).clamp(0, QA as i64);
                    sum += a * a * w1[i] as i64;
                }
                for i in 0..L1 {
                    let a = (them[i] as i64).clamp(0, QA as i64);
                    sum += a * a * w1[L1 + i] as i64;
                }
            }
            Activation::CReLU => {
                for i in 0..L1 {
                    let a = (us[i] as i64).clamp(0, QA as i64);
                    sum += a * w1[i] as i64;
                }
                for i in 0..L1 {
                    let a = (them[i] as i64).clamp(0, QA as i64);
                    sum += a * w1[L1 + i] as i64;
                }
            }
        }
        sum
    }

    /// The exact §3.4.3 integer forward pass, accumulating in `i64`.
    /// SCReLU: `v = (sum/QA + bias)·SCALE/(QA·QB)`; CReLU has no `/QA`.
    /// All divisions truncate toward zero; the Python oracle mirrors this.
    pub fn forward(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        out_bias: i32,
        act: Activation,
    ) -> Value {
        let mut sum = forward_raw_sum(us, them, w1, act);
        if let Activation::SCReLU = act {
            sum /= QA as i64; // per-term scale QA²·QB → QA·QB
        }
        let v = (sum + out_bias as i64) * SCALE as i64 / (QA as i64 * QB as i64);
        v as Value
    }

    /// The scalar kernel table.
    pub const KERNELS: Kernels = Kernels {
        add_sub,
        add_sub_sub,
        add2_sub2,
        refresh,
        forward,
    };
}

/// The AVX2 backend (design/11 §3.4.6, D14b): 16×i16 lanes per register.
///
/// **This module is the crate's sanctioned SIMD `unsafe`** — the
/// `#[allow(unsafe_code)]` re-opens the crate-wide `#![deny(unsafe_code)]`
/// for exactly this backend. Every kernel is a private `#[target_feature]
/// unsafe fn`; the release surface is exactly the safe wrappers in
/// [`KERNELS`], reachable only after the [`super::KERNELS`] dispatch has
/// confirmed AVX2 at runtime.
///
/// **Loads/stores are unaligned** (`loadu`/`storeu`): stack temporaries and
/// the 2-aligned FT weight rows would fault an aligned load.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// i16 lanes per 256-bit AVX2 register.
    const LANES: usize = 16;
    // No remainder handling: `L1` must divide evenly into LANES-wide chunks.
    const _: () = assert!(
        L1.is_multiple_of(LANES),
        "L1 must be a multiple of 16 for AVX2"
    );

    /// `dst = src + add_row − sub_row`, 16 i16 lanes at a time
    /// (`_mm256_add/sub_epi16` wrap i16 exactly like the scalar `wrapping_*`).
    #[target_feature(enable = "avx2")]
    unsafe fn add_sub(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add_row: &[i16; L1],
        sub_row: &[i16; L1],
    ) {
        // SAFETY: every ref is exactly L1 i16; `off` steps 0..L1 by 16, so
        // each alignment-agnostic loadu/storeu covers 16 in-bounds lanes.
        unsafe {
            for off in (0..L1).step_by(LANES) {
                let s = _mm256_loadu_si256(src.as_ptr().add(off).cast());
                let a = _mm256_loadu_si256(add_row.as_ptr().add(off).cast());
                let b = _mm256_loadu_si256(sub_row.as_ptr().add(off).cast());
                let r = _mm256_sub_epi16(_mm256_add_epi16(s, a), b);
                _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), r);
            }
        }
    }

    /// `dst = src + add − sub0 − sub1` (capture / ep / capture-promotion).
    #[target_feature(enable = "avx2")]
    unsafe fn add_sub_sub(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        // SAFETY: all five refs are exactly L1 i16; `off` steps 0..L1 by 16.
        unsafe {
            for off in (0..L1).step_by(LANES) {
                let s = _mm256_loadu_si256(src.as_ptr().add(off).cast());
                let a = _mm256_loadu_si256(add.as_ptr().add(off).cast());
                let s0 = _mm256_loadu_si256(sub0.as_ptr().add(off).cast());
                let s1 = _mm256_loadu_si256(sub1.as_ptr().add(off).cast());
                let r = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(s, a), s0), s1);
                _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), r);
            }
        }
    }

    /// `dst = src + add0 + add1 − sub0 − sub1` (castling — king + rook).
    #[target_feature(enable = "avx2")]
    unsafe fn add2_sub2(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add0: &[i16; L1],
        add1: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        // SAFETY: all six refs are exactly L1 i16; `off` steps 0..L1 by 16.
        unsafe {
            for off in (0..L1).step_by(LANES) {
                let s = _mm256_loadu_si256(src.as_ptr().add(off).cast());
                let a0 = _mm256_loadu_si256(add0.as_ptr().add(off).cast());
                let a1 = _mm256_loadu_si256(add1.as_ptr().add(off).cast());
                let s0 = _mm256_loadu_si256(sub0.as_ptr().add(off).cast());
                let s1 = _mm256_loadu_si256(sub1.as_ptr().add(off).cast());
                let acc = _mm256_add_epi16(_mm256_add_epi16(s, a0), a1);
                let r = _mm256_sub_epi16(_mm256_sub_epi16(acc, s0), s1);
                _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), r);
            }
        }
    }

    /// `dst = bias + Σ rows`; the chunk accumulator stays in a register
    /// across all rows before the single store.
    #[target_feature(enable = "avx2")]
    unsafe fn refresh(dst: &mut [i16; L1], bias: &[i16; L1], rows: &[&[i16; L1]]) {
        // SAFETY: dst/bias and every `row` are exactly L1 i16; `off` steps
        // 0..L1 by 16. loadu/storeu are alignment-agnostic.
        unsafe {
            for off in (0..L1).step_by(LANES) {
                let mut acc = _mm256_loadu_si256(bias.as_ptr().add(off).cast());
                for row in rows {
                    let r = _mm256_loadu_si256(row.as_ptr().add(off).cast());
                    acc = _mm256_add_epi16(acc, r);
                }
                _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), acc);
            }
        }
    }

    /// The raw pre-scaling sum (AVX2 analogue of [`scalar::forward_raw_sum`]).
    ///
    /// SCReLU per chunk: clamp `a` to `[0, QA]`, `p = mullo(a, w)` (fits i16
    /// by the §3.4.3 clip `|w1| ≤ 127`: `|a·w| ≤ 255·127 < 2^15`), then
    /// `madd(p, a)` → i32 pairwise `a²·w`; CReLU drops the square. Under the
    /// clip no i32 lane overflows (≤ 64·255²·127 ≈ 5.3e8), but the *total*
    /// can exceed i32, so the horizontal reduction **widens each lane to
    /// i64** — that makes this bit-identical to the scalar i64 reference.
    #[target_feature(enable = "avx2")]
    unsafe fn forward_sum(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        act: Activation,
    ) -> i64 {
        // SAFETY: us/them are L1 i16, w1 is 2·L1 i16; `off` steps 0..L1 by
        // 16, and the nstm half reads w1[L1 + off ..] (in-bounds: off < L1).
        // The final storeu targets a local [i32; 8] — one register wide.
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);
            let mut acc = _mm256_setzero_si256(); // 8 × i32
            // (accumulator base pointer, output-weight offset): stm half first.
            let halves = [(us.as_ptr(), 0usize), (them.as_ptr(), L1)];
            match act {
                Activation::SCReLU => {
                    for (base, woff) in halves {
                        for off in (0..L1).step_by(LANES) {
                            let raw = _mm256_loadu_si256(base.add(off).cast());
                            let a = _mm256_min_epi16(_mm256_max_epi16(raw, zero), qa);
                            let w = _mm256_loadu_si256(w1.as_ptr().add(woff + off).cast());
                            let p = _mm256_mullo_epi16(a, w);
                            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, a));
                        }
                    }
                }
                Activation::CReLU => {
                    for (base, woff) in halves {
                        for off in (0..L1).step_by(LANES) {
                            let raw = _mm256_loadu_si256(base.add(off).cast());
                            let a = _mm256_min_epi16(_mm256_max_epi16(raw, zero), qa);
                            let w = _mm256_loadu_si256(w1.as_ptr().add(woff + off).cast());
                            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, w));
                        }
                    }
                }
            }
            // Horizontal reduce the 8 i32 lanes, widening to i64 first.
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
            lanes.iter().map(|&x| x as i64).sum()
        }
    }

    /// The AVX2 forward pass: the raw sum via SIMD, then the same scalar i64
    /// tail as [`scalar::forward`].
    fn forward(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        out_bias: i32,
        act: Activation,
    ) -> Value {
        // SAFETY: installed into KERNELS only after is_x86_feature_detected!(
        // "avx2") (super::KERNELS dispatch, D14b).
        let mut sum = unsafe { forward_sum(us, them, w1, act) };
        if let Activation::SCReLU = act {
            sum /= QA as i64;
        }
        let v = (sum + out_bias as i64) * SCALE as i64 / (QA as i64 * QB as i64);
        v as Value
    }

    // --- Safe wrappers: the KERNELS entry points. super::KERNELS installs
    //     them only after runtime AVX2 detection (D14b). ---

    fn add_sub_w(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]) {
        // SAFETY: reached only via super::KERNELS after AVX2 detection.
        unsafe { add_sub(dst, src, add_row, sub_row) }
    }
    fn add_sub_sub_w(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        // SAFETY: reached only via super::KERNELS after AVX2 detection.
        unsafe { add_sub_sub(dst, src, add, sub0, sub1) }
    }
    fn add2_sub2_w(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add0: &[i16; L1],
        add1: &[i16; L1],
        sub0: &[i16; L1],
        sub1: &[i16; L1],
    ) {
        // SAFETY: reached only via super::KERNELS after AVX2 detection.
        unsafe { add2_sub2(dst, src, add0, add1, sub0, sub1) }
    }
    fn refresh_w(dst: &mut [i16; L1], bias: &[i16; L1], rows: &[&[i16; L1]]) {
        // SAFETY: reached only via super::KERNELS after AVX2 detection.
        unsafe { refresh(dst, bias, rows) }
    }

    /// Test-only: the raw pre-scaling sum through the AVX2 path, for the
    /// pre-narrowing bit-exactness comparison. Callers MUST have confirmed
    /// AVX2 (the tests do). `#[cfg(test)] pub(super)` keeps the shipped
    /// surface exactly the safe `KERNELS` wrappers.
    #[cfg(test)]
    pub(super) fn forward_raw_sum(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        act: Activation,
    ) -> i64 {
        // SAFETY: the bit-exactness tests gate on AVX2 detection before calling.
        unsafe { forward_sum(us, them, w1, act) }
    }

    /// The AVX2 kernel table (safe-wrapper fn pointers), installed by
    /// [`super::KERNELS`] once AVX2 is detected. `pub(super)` so safe code
    /// OUTSIDE `simd` cannot install AVX2 fn pointers on a non-AVX2 CPU.
    pub(super) const KERNELS: Kernels = Kernels {
        add_sub: add_sub_w,
        add_sub_sub: add_sub_sub_w,
        add2_sub2: add2_sub2_w,
        refresh: refresh_w,
        forward,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned64_is_cache_line_aligned() {
        assert_eq!(std::mem::align_of::<Aligned64<[[i16; L1]; 2]>>(), 64);
        // Deref reaches the inner value.
        let mut a = Aligned64([1i16, 2, 3]);
        assert_eq!(a[0], 1);
        a[2] = 9;
        assert_eq!(a.0[2], 9);
    }

    #[test]
    fn add_sub_is_src_plus_add_minus_sub() {
        let src = [10i16; L1];
        let add = [3i16; L1];
        let sub = [5i16; L1];
        let mut dst = [0i16; L1];
        scalar::add_sub(&mut dst, &src, &add, &sub);
        assert!(dst.iter().all(|&x| x == 8)); // 10 + 3 − 5
    }

    #[test]
    fn add_sub_sub_is_src_plus_add_minus_two_subs() {
        let src = [20i16; L1];
        let add = [3i16; L1];
        let sub0 = [5i16; L1];
        let sub1 = [4i16; L1];
        let mut dst = [0i16; L1];
        scalar::add_sub_sub(&mut dst, &src, &add, &sub0, &sub1);
        assert!(dst.iter().all(|&x| x == 14)); // 20 + 3 − 5 − 4
    }

    #[test]
    fn add2_sub2_is_src_plus_two_adds_minus_two_subs() {
        let src = [20i16; L1];
        let add0 = [3i16; L1];
        let add1 = [2i16; L1];
        let sub0 = [5i16; L1];
        let sub1 = [4i16; L1];
        let mut dst = [0i16; L1];
        scalar::add2_sub2(&mut dst, &src, &add0, &add1, &sub0, &sub1);
        assert!(dst.iter().all(|&x| x == 16)); // 20 + 3 + 2 − 5 − 4
    }

    #[test]
    fn refresh_is_bias_plus_rows() {
        let bias = [1i16; L1];
        let r0 = [2i16; L1];
        let r1 = [3i16; L1];
        let mut dst = [0i16; L1];
        scalar::refresh(&mut dst, &bias, &[&r0, &r1]);
        assert!(dst.iter().all(|&x| x == 6)); // 1 + 2 + 3
    }

    // A hand-verifiable forward pass: L1 columns all equal.
    #[test]
    fn forward_matches_hand_computed_screlu_and_crelu() {
        // us clamps to 10 (in [0,QA]); them is negative → clamps to 0.
        let us = [10i16; L1];
        let them = [-4i16; L1];
        let mut w1 = [0i16; 2 * L1];
        for w in w1.iter_mut().take(L1) {
            *w = 2; // stm-half weights
        }
        // nstm-half weights are 7 but "them" clamps to 0, so they contribute 0.
        for w in w1.iter_mut().skip(L1) {
            *w = 7;
        }
        let out_bias = 100;

        // SCReLU: sum = Σ_us 10²·2 = L1·200; +them 0. /QA. then (…+bias)·SCALE/(QA·QB).
        let screlu_sum = (L1 as i64) * 100 * 2 / QA as i64;
        let expected_screlu =
            ((screlu_sum + out_bias) * SCALE as i64 / (QA as i64 * QB as i64)) as Value;
        assert_eq!(
            scalar::forward(&us, &them, &w1, out_bias as i32, Activation::SCReLU),
            expected_screlu
        );

        // CReLU: sum = Σ_us 10·2 = L1·20; +them 0. No /QA.
        let crelu_sum = (L1 as i64) * 20;
        let expected_crelu =
            ((crelu_sum + out_bias) * SCALE as i64 / (QA as i64 * QB as i64)) as Value;
        assert_eq!(
            scalar::forward(&us, &them, &w1, out_bias as i32, Activation::CReLU),
            expected_crelu
        );
    }

    #[test]
    fn kernels_lazylock_dispatch_matches_scalar_reference() {
        // Smoke: whatever backend the dispatch installed reproduces scalar.
        let us = [5i16; L1];
        let them = [5i16; L1];
        let w1 = [1i16; 2 * L1];
        let via_lock = (KERNELS.forward)(&us, &them, &w1, 0, Activation::SCReLU);
        let direct = scalar::forward(&us, &them, &w1, 0, Activation::SCReLU);
        assert_eq!(via_lock, direct);
    }

    // --- AVX2 == scalar, bit-exact (design/11 §3.4.6 GATE): all five
    // kernels, many random inputs, both backends reached explicitly.
    // They no-op (pass) on a machine without AVX2.
    #[cfg(target_arch = "x86_64")]
    mod avx2_bit_exact {
        use super::*;
        use crate::attacks::Prng;

        /// A full-range random `[i16; L1]` (stresses clamp ends + i16 wrap).
        fn rand_arr(rng: &mut Prng) -> [i16; L1] {
            let mut a = [0i16; L1];
            for x in a.iter_mut() {
                *x = rng.rand64() as i16;
            }
            a
        }

        #[test]
        fn add_sub_family_matches_scalar_over_full_range_i16() {
            if !std::arch::is_x86_feature_detected!("avx2") {
                // Marker: a green run here is not AVX2 coverage.
                eprintln!(
                    "SKIP add_sub_family_matches_scalar: host lacks AVX2 — \
                     scalar dispatch only, AVX2 kernels NOT verified"
                );
                return;
            }
            let s = scalar::KERNELS;
            let v = avx2::KERNELS;
            let mut rng = Prng::new(0x0000_ACC2_BEEF_1357);
            for _ in 0..3000 {
                let src = rand_arr(&mut rng);
                let r = [
                    rand_arr(&mut rng),
                    rand_arr(&mut rng),
                    rand_arr(&mut rng),
                    rand_arr(&mut rng),
                    rand_arr(&mut rng),
                ];
                let (mut ds, mut dv) = ([0i16; L1], [0i16; L1]);

                // add_sub (1+/1−) — exact for any i16 (add/sub_epi16 wrap = wrapping_*).
                (s.add_sub)(&mut ds, &src, &r[0], &r[1]);
                (v.add_sub)(&mut dv, &src, &r[0], &r[1]);
                assert_eq!(ds, dv, "add_sub");

                // add_sub_sub (1+/2−).
                (s.add_sub_sub)(&mut ds, &src, &r[0], &r[1], &r[2]);
                (v.add_sub_sub)(&mut dv, &src, &r[0], &r[1], &r[2]);
                assert_eq!(ds, dv, "add_sub_sub");

                // add2_sub2 (2+/2−).
                (s.add2_sub2)(&mut ds, &src, &r[0], &r[1], &r[2], &r[3]);
                (v.add2_sub2)(&mut dv, &src, &r[0], &r[1], &r[2], &r[3]);
                assert_eq!(ds, dv, "add2_sub2");

                // refresh with a random row count 0..=5 (bias = src).
                let rows: [&[i16; L1]; 5] = [&r[0], &r[1], &r[2], &r[3], &r[4]];
                let n = (rng.rand64() % 6) as usize;
                (s.refresh)(&mut ds, &src, &rows[..n]);
                (v.refresh)(&mut dv, &src, &rows[..n]);
                assert_eq!(ds, dv, "refresh (n={n})");
            }
        }

        #[test]
        fn forward_matches_scalar_pre_narrowing_and_final() {
            if !std::arch::is_x86_feature_detected!("avx2") {
                eprintln!(
                    "SKIP forward_matches_scalar_pre_narrowing_and_final: host \
                     lacks AVX2 — AVX2 forward NOT verified"
                );
                return;
            }
            let mut rng = Prng::new(0xF0_2E_1A_7C_5B_39_11_23);
            // Weights bounded to the §3.4.3 clip |w1| ≤ 127 — the condition
            // for AVX2/scalar bit-exactness; real nets satisfy it too.
            let w1_row = |rng: &mut Prng| -> [i16; 2 * L1] {
                let mut w = [0i16; 2 * L1];
                for x in w.iter_mut() {
                    *x = (rng.rand64() % 255) as i16 - 127; // [-127, 127]
                }
                w
            };
            for act in [Activation::SCReLU, Activation::CReLU] {
                for _ in 0..3000 {
                    let us = rand_arr(&mut rng);
                    let them = rand_arr(&mut rng);
                    let w1 = w1_row(&mut rng);
                    let out_bias = (rng.rand64() % 60_001) as i32 - 30_000;

                    // (1) pre-narrowing raw i64 sum.
                    let ss = scalar::forward_raw_sum(&us, &them, &w1, act);
                    let vs = avx2::forward_raw_sum(&us, &them, &w1, act);
                    assert_eq!(ss, vs, "{act:?} raw sum diverged");

                    // (2) final Value.
                    let sf = (scalar::KERNELS.forward)(&us, &them, &w1, out_bias, act);
                    let vf = (avx2::KERNELS.forward)(&us, &them, &w1, out_bias, act);
                    assert_eq!(sf, vf, "{act:?} final Value diverged");
                }
            }
        }

        /// Cross-check scalar vs AVX2 at BOTH the raw i64 sum and the final
        /// `Value`, returning the raw sum. AVX2 must be present (callers gate).
        fn agree(
            us: &[i16; L1],
            them: &[i16; L1],
            w1: &[i16; 2 * L1],
            out_bias: i32,
            act: Activation,
            label: &str,
        ) -> i64 {
            let ss = scalar::forward_raw_sum(us, them, w1, act);
            let vs = avx2::forward_raw_sum(us, them, w1, act);
            assert_eq!(ss, vs, "{label}: {act:?} raw sum diverged");
            let sf = (scalar::KERNELS.forward)(us, them, w1, out_bias, act);
            let vf = (avx2::KERNELS.forward)(us, them, w1, out_bias, act);
            assert_eq!(sf, vf, "{label}: {act:?} final Value diverged");
            ss
        }

        // Constructed (non-RNG) boundary cases: the RNG sweep above rarely
        // makes the SCReLU total exceed i32, so on its own it would NOT catch
        // a regression from the i64 horizontal reduction back to i32. These
        // force the overflow deterministically and pin the clamp/clip corners.
        #[test]
        fn forward_constructed_cases_exercise_i64_reduction_and_boundaries() {
            if !std::arch::is_x86_feature_detected!("avx2") {
                eprintln!(
                    "SKIP forward_constructed_cases: host lacks AVX2 — the i64 \
                     reduction / boundary corners are NOT verified"
                );
                return;
            }
            let qa = QA as i16; // 255

            // (A) i64-REDUCTION PROOF, positive: every lane clamps to QA and
            // every weight sits at +127, so the SCReLU raw sum is
            // 2·L1·QA²·127 = 4,228,185,600 > i32::MAX — an i32 reduction
            // would WRAP and diverge. Also drives every i32 madd lane to its
            // ceiling under the §3.4.3 clip.
            let us_max = [qa; L1];
            let them_max = [qa; L1];
            let w_pos = [127i16; 2 * L1];
            let raw_pos = agree(
                &us_max,
                &them_max,
                &w_pos,
                12_345,
                Activation::SCReLU,
                "all-QA×+127",
            );
            assert_eq!(
                raw_pos,
                2 * L1 as i64 * QA as i64 * QA as i64 * 127,
                "SCReLU max positive raw sum"
            );
            assert!(
                raw_pos > i32::MAX as i64,
                "constructed case must exceed i32::MAX to exercise the i64 \
                 reduction (got {raw_pos}, i32::MAX = {})",
                i32::MAX
            );

            // (B) i64-REDUCTION PROOF, negative: all weights at −127, so the
            // raw sum is −4,228,185,600 < i32::MIN.
            let w_neg = [-127i16; 2 * L1];
            let raw_neg = agree(
                &us_max,
                &them_max,
                &w_neg,
                -12_345,
                Activation::SCReLU,
                "all-QA×−127",
            );
            assert_eq!(
                raw_neg,
                -(2 * L1 as i64 * QA as i64 * QA as i64 * 127),
                "SCReLU max negative raw sum"
            );
            assert!(
                (raw_neg) < i32::MIN as i64,
                "constructed case must underflow i32::MIN (got {raw_neg})"
            );

            // CReLU counterpart (no square): raw = 2·L1·QA·(+127) = 16,581,120.
            let craw = agree(
                &us_max,
                &them_max,
                &w_pos,
                0,
                Activation::CReLU,
                "all-QA×+127 CReLU",
            );
            assert_eq!(craw, 2 * L1 as i64 * QA as i64 * 127);

            // (C) Mid-range mix: total well inside i32 with every lane
            // populated, so a lane-vs-total confusion still cannot hide.
            let us_mid = [qa / 2; L1];
            let w_alt = {
                let mut w = [0i16; 2 * L1];
                for (i, x) in w.iter_mut().enumerate() {
                    *x = if i % 2 == 0 { 127 } else { -127 };
                }
                w
            };
            agree(
                &us_mid,
                &them_max,
                &w_alt,
                7,
                Activation::SCReLU,
                "mid×alt-±127",
            );

            // (D) BOUNDARY PINS: accumulator lanes cycle the SCReLU-clamp
            // corners {−1, 0, 1, QA−1, QA, QA+1}; weights cycle the clip
            // corners {−127, 0, 127}. Both activations, both bias signs.
            let acc_corners = [-1i16, 0, 1, qa - 1, qa, qa + 1];
            let w_corners = [-127i16, 0, 127];
            let mut us_b = [0i16; L1];
            let mut them_b = [0i16; L1];
            let mut w_b = [0i16; 2 * L1];
            for i in 0..L1 {
                us_b[i] = acc_corners[i % acc_corners.len()];
                them_b[i] = acc_corners[(i + 3) % acc_corners.len()];
            }
            for (i, x) in w_b.iter_mut().enumerate() {
                *x = w_corners[i % w_corners.len()];
            }
            for act in [Activation::SCReLU, Activation::CReLU] {
                agree(&us_b, &them_b, &w_b, -30_000, act, "boundary corners −bias");
                agree(&us_b, &them_b, &w_b, 30_000, act, "boundary corners +bias");
            }
        }

        // Non-gating micro-measurement of AVX2-vs-scalar forward throughput;
        // run with:
        //   cargo test --release -- --ignored avx2_forward_speedup --nocapture
        #[test]
        #[ignore = "timing microbenchmark; run manually with --nocapture in release"]
        fn avx2_forward_speedup() {
            use std::hint::black_box;
            use std::time::Instant;
            if !std::arch::is_x86_feature_detected!("avx2") {
                println!("no AVX2 on this host");
                return;
            }
            let mut rng = Prng::new(0x5EED_DEAD_BEEF_0001); // any nonzero seed
            let us = rand_arr(&mut rng);
            let them = rand_arr(&mut rng);
            let mut w1 = [0i16; 2 * L1];
            for x in w1.iter_mut() {
                *x = (rng.rand64() % 255) as i16 - 127;
            }
            const N: usize = 2_000_000;
            let time = |k: &Kernels| -> f64 {
                let mut sink = 0i64;
                let start = Instant::now();
                for _ in 0..N {
                    sink = sink.wrapping_add((k.forward)(
                        black_box(&us),
                        black_box(&them),
                        &w1,
                        0,
                        Activation::SCReLU,
                    ) as i64);
                }
                black_box(sink);
                start.elapsed().as_secs_f64() * 1e9 / N as f64
            };
            let (mut sc, mut av) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..3 {
                sc = sc.min(time(&scalar::KERNELS));
                av = av.min(time(&avx2::KERNELS));
            }
            println!("scalar forward: {sc:.1} ns/call");
            println!("avx2   forward: {av:.1} ns/call  ({:.2}x)", sc / av);
        }
    }
}
