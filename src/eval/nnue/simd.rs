//! The compute kernels behind the accumulator and forward pass (design/11
//! §3.4.6).
//!
//! **Two backends, selected once at startup** (design/11 §3.4.6):
//! [`scalar`] (plain safe Rust, forward pass accumulated in `i64`) and, on
//! x86_64 with AVX2, [`avx2`]. [`KERNELS`] runtime-detects AVX2 via
//! `is_x86_feature_detected!("avx2")` and installs the faster backend; on any
//! other CPU it falls back to scalar. The scalar forward IS the bit-exact
//! reference — the `avx2` backend reproduces it lane-for-lane, pinned by
//! `avx2_matches_scalar_*` below.
//!
//! **The AVX2 backend is the crate's sanctioned SIMD `unsafe` (D14b).** The
//! crate is `#![deny(unsafe_code)]`; all `unsafe` here is contained inside the
//! [`avx2`] module (a localized `#[allow(unsafe_code)]`, `// SAFETY:` on every
//! block — the same discipline as `tt`'s prefetch intrinsic, D14a). The
//! [`Kernels`] table therefore keeps plain **safe `fn` pointers**: the AVX2
//! kernels are `#[target_feature(enable = "avx2")] unsafe fn`, reached only
//! through thin *safe wrappers* that are installed into [`KERNELS`] once AVX2
//! detection has succeeded. So the accumulator/forward callers stay safe and
//! no `unsafe` leaks past this module (a deliberate deviation from the doc's
//! `unsafe fn`-pointer sketch, recorded in design/11 §3.4.6).

use super::network::Activation;
use super::{L1, QA, QB, SCALE};
use crate::types::Value;
use std::sync::LazyLock;

/// 64-byte-aligned wrapper for accumulator storage. Harmless for the scalar
/// backend; keeps the *stored* accumulator cache-line aligned so the AVX2
/// kernels' 16-lane loads never straddle a cache line (they still use
/// alignment-agnostic `loadu`, since the temporaries and FT rows they also
/// touch are only 2-aligned — design/11 §3.4.6).
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
/// update (the dominant case, a quiet or quiet-promotion move).
pub type AddSubFn =
    fn(dst: &mut [i16; L1], src: &[i16; L1], add_row: &[i16; L1], sub_row: &[i16; L1]);
/// `dst = src + W0[add] − W0[sub0] − W0[sub1]`: the fused 1-add/2-sub update
/// (an ordinary capture, en passant, or capture-promotion).
pub type AddSubSubFn =
    fn(dst: &mut [i16; L1], src: &[i16; L1], add: &[i16; L1], sub0: &[i16; L1], sub1: &[i16; L1]);
/// `dst = src + W0[add0] + W0[add1] − W0[sub0] − W0[sub1]`: the fused
/// 2-add/2-sub update (castling — the king and rook both move).
pub type Add2Sub2Fn = fn(
    dst: &mut [i16; L1],
    src: &[i16; L1],
    add0: &[i16; L1],
    add1: &[i16; L1],
    sub0: &[i16; L1],
    sub1: &[i16; L1],
);
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
/// (design/11 §3.4.6). **Safe `fn` pointers** in both backends: the AVX2
/// kernels are `#[target_feature] unsafe fn` but are reached through safe
/// wrappers, so all `unsafe` stays inside [`avx2`] and callers never write it.
pub struct Kernels {
    pub add_sub: AddSubFn,
    pub add_sub_sub: AddSubSubFn,
    pub add2_sub2: Add2Sub2Fn,
    pub refresh: RefreshFn,
    pub forward: ForwardFn,
}

/// The active kernel backend, selected once (design/11 §3.4.6). AVX2 where
/// available, scalar otherwise.
pub static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: avx2 verified at startup via is_x86_feature_detected! (D14b).
        // Every `avx2::KERNELS` wrapper's precondition is exactly "this CPU has
        // AVX2"; this runtime check discharges it for the whole process, so it
        // is sound to install the AVX2 backend below. (The `unsafe` itself is
        // contained in `avx2`; this is the single site that establishes the
        // invariant those blocks rely on.)
        if std::arch::is_x86_feature_detected!("avx2") {
            return avx2::KERNELS;
        }
    }
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

    /// `dst = src + add − sub0 − sub1`, column-wise (a capture / ep /
    /// capture-promotion). Same i16 lane-wrap discipline as [`add_sub`].
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

    /// `dst = src + add0 + add1 − sub0 − sub1`, column-wise (castling — king +
    /// rook). Same i16 lane-wrap discipline as [`add_sub`].
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
    /// **bit-exact reference** the AVX2 path must reproduce (design/11 §3.4.3,
    /// §3.4.6). This is the accumulator *before* the SCReLU `/QA` fold, the
    /// output-bias add, and the `SCALE / (QA·QB)` narrowing:
    ///
    /// - SCReLU: `Σ clamp(a,0,QA)² · w`
    /// - CReLU:  `Σ clamp(a,0,QA)  · w`  (drop the `²`)
    ///
    /// The stm half (`w1[0..L1]`) is summed first. Exposed so the
    /// bit-exactness tests can compare backends at the pre-narrowing width,
    /// where an AVX2 i32-lane overflow would otherwise be hidden by the final
    /// `as Value` truncation (reviewer B's F3 landmine).
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
    ///
    /// SCReLU: `sum = Σ clamp(a,0,QA)² · w`; `v = (sum/QA + bias)·SCALE/(QA·QB)`.
    /// CReLU:  `sum = Σ clamp(a,0,QA)  · w`; `v = (sum     + bias)·SCALE/(QA·QB)`
    /// (the `²` and the `/QA` drop out). All divisions truncate toward zero
    /// (Rust `/`); the Python oracle mirrors this exactly. The accumulation is
    /// factored into [`forward_raw_sum`]; the SCReLU `/QA` and the final
    /// narrowing stay here — arithmetically identical to folding `/QA` into the
    /// SCReLU branch (division-before-add of a common divisor), so the byte
    /// output is unchanged.
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

/// The AVX2 backend (design/11 §3.4.6, D14b): 16×i16 lanes per 256-bit
/// register, `L1 = 256` ⇒ 16 chunks per accumulator.
///
/// **This module is the crate's sanctioned SIMD `unsafe`.** The
/// `#[allow(unsafe_code)]` below re-opens the crate-wide `#![deny(unsafe_code)]`
/// for exactly this backend (the same localization `tt` uses for its prefetch
/// intrinsic, D14a). Every kernel is a private `#[target_feature(enable =
/// "avx2")] unsafe fn`; the release surface is exactly the safe wrappers in
/// [`KERNELS`] (`pub(super)`, so the dispatch and tests reach them but nothing
/// outside `simd` can) — plus, under `#[cfg(test)]` only, `forward_raw_sum` for
/// the bit-exactness cross-check — all reachable only after the
/// [`super::KERNELS`] dispatch has confirmed AVX2 at runtime. Each `unsafe`
/// block carries its `// SAFETY:` note.
///
/// **Loads/stores are unaligned** (`loadu`/`storeu`): [`Aligned64`] covers the
/// *stored* accumulator, but not the stack temporaries the kernels receive nor
/// the 2-aligned FT weight rows, so an aligned load off those would fault
/// (design/11 §3.4.6 step-2 note).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// i16 lanes per 256-bit AVX2 register.
    const LANES: usize = 16;
    // The kernels stride the accumulator in `LANES`-wide chunks with no
    // remainder handling, so `L1` must divide evenly.
    const _: () = assert!(
        L1.is_multiple_of(LANES),
        "L1 must be a multiple of 16 for AVX2"
    );

    /// `dst = src + add_row − sub_row`, 16 i16 lanes at a time. Exact vs the
    /// scalar `wrapping_add`/`wrapping_sub`: `_mm256_add/sub_epi16` wrap i16
    /// identically.
    #[target_feature(enable = "avx2")]
    unsafe fn add_sub(
        dst: &mut [i16; L1],
        src: &[i16; L1],
        add_row: &[i16; L1],
        sub_row: &[i16; L1],
    ) {
        // SAFETY: every ref is exactly L1 i16; `off` steps 0..L1 by 16, so each
        // loadu/storeu covers 16 in-bounds lanes. loadu/storeu are
        // alignment-agnostic (module note).
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

    /// `dst = bias + Σ rows` (a from-scratch refresh). The chunk accumulator
    /// stays in a register across all rows before the single store.
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

    /// The raw pre-scaling dot-product sum (the AVX2 analogue of
    /// [`scalar::forward_raw_sum`]).
    ///
    /// SCReLU per 16-lane chunk: clamp `a` to `[0, QA]`
    /// (`_mm256_max_epi16` then `_mm256_min_epi16`), `p = _mm256_mullo_epi16(a,
    /// w)` (fits i16 by the §3.4.3 weight clip `|w1| ≤ 127`, so `|a·w| ≤
    /// 255·127 = 32385 < 32768`), then `_mm256_madd_epi16(p, a)` → i32 pairwise
    /// `a²·w`. CReLU drops the square: `_mm256_madd_epi16(a, w)`. The 8 i32
    /// lanes accumulate across all 32 chunks (2·L1 terms) with `add_epi32`;
    /// under the clip each lane holds ≤ 64·255²·127 ≈ 5.3e8 < i32::MAX, so no
    /// lane overflows. The final horizontal reduction **widens each lane to
    /// i64** before summing, so even a total that exceeds i32 (worst case
    /// ≈ 4.2e9 > i32::MAX) is exact — that is what makes this bit-identical to
    /// the scalar i64 reference (reviewer B's F3: i32-lane accumulation is safe,
    /// the i32 *total* is not, so the reduction is i64).
    #[target_feature(enable = "avx2")]
    unsafe fn forward_sum(
        us: &[i16; L1],
        them: &[i16; L1],
        w1: &[i16; 2 * L1],
        act: Activation,
    ) -> i64 {
        // SAFETY: us/them are L1 i16, w1 is 2·L1 i16; `off` steps 0..L1 by 16
        // and the nstm half reads w1[L1 + off .. +16] (in-bounds since off < L1).
        // loadu/storeu are alignment-agnostic. The final storeu targets a local
        // [i32; 8] (32 bytes) — the exact width of one register.
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

    /// The AVX2 forward pass. The AVX2 part is just the raw sum; the SCReLU
    /// `/QA` fold and the final `SCALE / (QA·QB)` narrowing are plain i64 scalar
    /// arithmetic, byte-identical to [`scalar::forward`]'s tail.
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

    // --- Safe wrappers: the KERNELS entry points. Each discharges its callee's
    //     "AVX2 is present" precondition against the super::KERNELS dispatch,
    //     which only installs these after runtime detection (D14b). ---

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
    /// pre-narrowing bit-exactness comparison against
    /// [`scalar::forward_raw_sum`]. Callers MUST have confirmed
    /// `is_x86_feature_detected!("avx2")` (the tests do).
    ///
    /// `#[cfg(test)] pub(super)`: its sole caller is the bit-exactness test, so
    /// it is compiled out of every release binary — the AVX2 module's shipped
    /// surface is exactly the safe `KERNELS` wrappers (A-F1).
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

    /// The AVX2 kernel table (safe-wrapper fn pointers). Installed by
    /// [`super::KERNELS`] once AVX2 is detected. `pub(super)` (not `pub`): only
    /// the `simd`-module dispatch and the bit-exactness tests may reach it, so
    /// safe code OUTSIDE `simd` cannot install AVX2 fn pointers on a non-AVX2
    /// CPU — that keeps the crate-wide `#![deny(unsafe_code)]` an honest UB
    /// guarantee (A-F1).
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
    fn kernels_lazylock_dispatch_matches_scalar_reference() {
        // Smoke: whichever backend the runtime dispatch installed (AVX2 on this
        // box, scalar elsewhere) reproduces the scalar reference forward.
        let us = [5i16; L1];
        let them = [5i16; L1];
        let w1 = [1i16; 2 * L1];
        let via_lock = (KERNELS.forward)(&us, &them, &w1, 0, Activation::SCReLU);
        let direct = scalar::forward(&us, &them, &w1, 0, Activation::SCReLU);
        assert_eq!(via_lock, direct);
    }

    // --- AVX2 == scalar, bit-exact (design/11 §3.4.6 step-2 GATE) -----------
    //
    // The key step-2 gate: the AVX2 backend must reproduce the scalar i64
    // reference lane-for-lane, for ALL FIVE kernels, over many random inputs.
    // The dispatch auto-selects AVX2 on this box, so these tests reach both
    // backends *explicitly* (`scalar::KERNELS` vs `avx2::KERNELS`) and compare
    // them directly. They no-op (pass) on a machine without AVX2.
    #[cfg(target_arch = "x86_64")]
    mod avx2_bit_exact {
        use super::*;
        use crate::attacks::Prng;

        /// Draw a full-range random `[i16; L1]` (stresses the SCReLU clamp on
        /// both ends and the i16 add/sub wrap).
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
                // no AVX2: dispatch uses scalar, nothing to cross-check. Emit a
                // marker so a green run here is not mistaken for AVX2 coverage.
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
            // Weights bounded to the §3.4.3 training clip |w1| ≤ round(1.98·QB)
            // = 127. This is the exact condition that keeps the AVX2 path
            // bit-exact vs the scalar i64 reference (reviewer B's F3):
            //   * `mullo_epi16(clamp(a), w)` fits i16: |a·w| ≤ 255·127 < 2^15;
            //   * each i32 lane sums ≤ 64·255²·127 ≈ 5.3e8 < i32::MAX (no lane
            //     overflow), and the horizontal reduction is done in i64 so the
            //     ≈4.2e9 worst-case *total* (which exceeds i32) is still exact.
            // Real bullet nets satisfy the same clip, so this matches inference.
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

                    // (1) pre-narrowing raw i64 sum — an i32-lane overflow would
                    // diverge here even when the final `as Value` truncation
                    // happens to collapse two different sums to the same i32.
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

        /// Cross-check the scalar reference and the AVX2 backend at BOTH the raw
        /// i64 sum and the final `Value`, returning the (identical) raw sum so a
        /// caller can pin its magnitude. AVX2 must be present (callers gate).
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

        // Constructed (non-RNG) boundary cases. The RNG sweep above rarely — if
        // ever — makes the SCReLU total exceed i32, so on its own it would NOT
        // catch a regression from the i64 horizontal reduction back to i32.
        // These cases force the overflow deterministically and pin the exact
        // saturated corners of the clamp and the weight clip (B-MAJOR-1 /
        // B-MINOR-1).
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

            // (A) i64-REDUCTION PROOF, positive. Every accumulator lane clamps to
            // QA; every output weight is at the +127 clip. Under SCReLU the raw
            // sum is 2·L1 · QA² · 127 = 512 · 65025 · 127 = 4,228,185,600, which
            // is > i32::MAX (2,147,483,647): a reduction that summed the eight i32
            // madd lanes in i32 (instead of widening to i64) would WRAP here and
            // diverge from the scalar reference — this case would then fail. It
            // also drives every i32 madd lane to its ceiling 64·QA²·127 =
            // 528,523,200 (≈5.3e8), the max a lane can hold under the §3.4.3 clip.
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
            assert_eq!(raw_pos, 4_228_185_600, "SCReLU max positive raw sum");
            assert!(
                raw_pos > i32::MAX as i64,
                "constructed case must exceed i32::MAX to exercise the i64 \
                 reduction (got {raw_pos}, i32::MAX = {})",
                i32::MAX
            );

            // (B) i64-REDUCTION PROOF, negative. Same accumulators, all weights at
            // the −127 clip: raw sum = −4,228,185,600 < i32::MIN.
            let w_neg = [-127i16; 2 * L1];
            let raw_neg = agree(
                &us_max,
                &them_max,
                &w_neg,
                -12_345,
                Activation::SCReLU,
                "all-QA×−127",
            );
            assert_eq!(raw_neg, -4_228_185_600, "SCReLU max negative raw sum");
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

            // (C) A mid-range mix (accumulators at QA/2, weights alternating
            // ±127): keeps the total well inside i32 while every lane stays
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

            // (D) BOUNDARY PINS. Accumulator lanes cycle the SCReLU-clamp corners
            // {−1, 0, 1, QA−1, QA, QA+1}; output weights cycle the clip corners
            // {−127, 0, 127}. This isolates the clamp-at-0 / clamp-at-QA edges and
            // the weight-at-±127 extremes at known, staggered positions (the two
            // cycle lengths, 6 and 3, stagger the us/them halves). Both
            // activations, both signs of the out-bias.
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

        // A quick, non-gating micro-measurement (design/11 §3.4.6 optional):
        // AVX2-vs-scalar forward-pass throughput. Ignored by default (timing is
        // machine-dependent); run with:
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
