//! The compute kernels behind the accumulator and forward pass (design/11
//! §3.4.6).
//!
//! **Step 1 is SCALAR ONLY** (no `unsafe`, no AVX2). The design's eventual
//! seam uses `#[target_feature(enable = "avx2")] unsafe fn` pointers selected
//! once at startup; here the [`Kernels`] table holds plain safe `fn`
//! pointers and [`KERNELS`] always returns the scalar backend. The scalar
//! forward pass accumulates in `i64` and IS the bit-exact reference the step-2
//! AVX2 backend must reproduce. When the AVX2 backend lands, the pointer types
//! become `unsafe fn` and the dispatch below gains the one `is_x86_feature_
//! detected!("avx2")` branch (its single `// SAFETY:` call site).

use super::network::Activation;
use super::{L1, QA, QB, SCALE};
use crate::types::Value;
use std::sync::LazyLock;

/// 64-byte-aligned wrapper for accumulator storage. Harmless for the scalar
/// backend; step 2's AVX2 loads want the accumulator cache-line aligned.
///
/// `Deref`/`DerefMut` expose the inner value so callers write `acc[color]`
/// rather than `acc.0[color]`.
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

/// `dst = src + W0[add_row] − W0[sub_row]`: the fused 1-add/1-sub column
/// update (the dominant case, a quiet move).
pub type AddSubFn =
    fn(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]);
/// `dst = bias + Σ_f W0[f]` over already-resolved `L1`-wide feature rows: a
/// from-scratch accumulator refresh.
pub type RefreshFn = fn(dst: &mut [i16; L1], bias: &[i16; L1], rows: &[&[i16; L1]]);
/// The quantized output forward pass (design/11 §3.4.3): `us`/`them` are the
/// stm/nstm accumulators, `w1` the `2·L1` output weights (stm half first).
/// Returns the centipawn value.
pub type ForwardFn = fn(
    us: &[i16; L1],
    them: &[i16; L1],
    w1: &[i16; 2 * L1],
    out_bias: i32,
    act: Activation,
) -> Value;

/// The compute kernels the accumulator and forward pass dispatch through
/// (design/11 §3.4.6). Safe `fn` pointers in step 1; `unsafe fn` in step 2.
pub struct Kernels {
    pub add_sub: AddSubFn,
    pub refresh: RefreshFn,
    pub forward: ForwardFn,
}

/// The active kernel backend, selected once. Step 1: always scalar.
pub static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    // step 2: AVX2 backend selected here when is_x86_feature_detected!("avx2")
    // — e.g. `#[cfg(target_arch = "x86_64")] if is_x86_feature_detected!("avx2")
    // { return avx2::KERNELS; }` — with the one `// SAFETY:` comment on the
    // dispatch. Not implemented this step (scalar-only, no unsafe).
    scalar::KERNELS
});

/// The scalar reference backend (plain safe Rust; the autovectorizer handles
/// the column loops respectably). The forward pass accumulates in `i64`.
pub mod scalar {
    use super::*;

    /// `dst = src + add_row − sub_row`, column-wise. `wrapping_add`/`_sub`
    /// match the i16 lane wrap the AVX2 `_mm256_add/sub_epi16` will do — the
    /// accumulator columns are i16 by design (design/11 §3.4.3) and real nets
    /// never overflow them, but the two backends must agree bit-for-bit even
    /// on a synthetic net that would.
    pub fn add_sub(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]) {
        for i in 0..L1 {
            dst[i] = src[i].wrapping_add(add_row[i]).wrapping_sub(sub_row[i]);
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

    /// The exact §3.4.3 integer forward pass, accumulating in `i64`.
    ///
    /// SCReLU: `sum = Σ clamp(a,0,QA)² · w`; `v = (sum/QA + bias)·SCALE/(QA·QB)`.
    /// CReLU:  `sum = Σ clamp(a,0,QA)  · w`; `v = (sum     + bias)·SCALE/(QA·QB)`
    /// (the `²` and the `/QA` drop out). The stm half (`w1[0..L1]`) is summed
    /// first. All divisions truncate toward zero (Rust `/`); the Python oracle
    /// mirrors this exactly.
    pub fn forward(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        out_bias: i32,
        act: Activation,
    ) -> Value {
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
                sum /= QA as i64; // per-term scale QA²·QB → QA·QB
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
        let v = (sum + out_bias as i64) * SCALE as i64 / (QA as i64 * QB as i64);
        v as Value
    }

    /// The scalar kernel table.
    pub const KERNELS: Kernels = Kernels {
        add_sub,
        refresh,
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
    fn refresh_is_bias_plus_rows() {
        let bias = [1i16; L1];
        let r0 = [2i16; L1];
        let r1 = [3i16; L1];
        let mut dst = [0i16; L1];
        scalar::refresh(&mut dst, &bias, &[&r0, &r1]);
        assert!(dst.iter().all(|&x| x == 6)); // 1 + 2 + 3
    }

    // A tiny hand-verifiable forward pass. With L1 columns all equal we can
    // compute the exact integer result by hand for both activations.
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
    fn kernels_lazylock_returns_scalar_backend() {
        // Smoke: the dispatched forward equals the scalar forward directly.
        let us = [5i16; L1];
        let them = [5i16; L1];
        let w1 = [1i16; 2 * L1];
        let via_lock = (KERNELS.forward)(&us, &them, &w1, 0, Activation::SCReLU);
        let direct = scalar::forward(&us, &them, &w1, 0, Activation::SCReLU);
        assert_eq!(via_lock, direct);
    }
}
