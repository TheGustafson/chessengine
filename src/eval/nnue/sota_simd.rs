//! Fused accumulator kernels for the 768-wide SOTA feature transformers.
//!
//! Keeping the accumulator in registers while applying all feature deltas is
//! substantially cheaper than reading and writing all 768 lanes once per
//! changed feature. The scalar backend is the bit-exact reference; AVX2 is
//! selected once at startup and preserves wrapping i16 arithmetic.

use super::sota_features::ACC;
use super::sota_head::{FT_QUANT, FT_SHIFT, L2, L3};
use std::sync::LazyLock;

/// The L1 sparse dot writes `L2` int32 accumulators. The kernels process the
/// outputs in 16-wide blocks (one avx512 register / two avx256 registers each),
/// so `L2` must be a multiple of 16. Widening the head bottleneck (D41) just
/// bumps [`super::sota_head::L2`]; the block loop scales automatically.
const L2_BLK: usize = 16;
const _: () = assert!(
    L2.is_multiple_of(L2_BLK),
    "L2 must be a multiple of 16 for the l1_dot block loop"
);
/// The f32 tail kernels likewise assume whole 16-lane blocks.
const _: () = assert!(
    L3.is_multiple_of(L2_BLK),
    "L3 must be a multiple of 16 for the l2_tail block loop"
);

type ApplyI16Fn = fn(&mut [i16; ACC], &[i16], &[usize], &[usize]);
// Fused copy+apply: read `src`, apply deltas, write `dst` in ONE pass — the
// lazy fold's child = parent ± deltas without a separate ~1.5 KB memmove
// (profiling-study §4.3/opt #2). `dst`/`src` may not alias.
type ApplyI16IntoFn = fn(&mut [i16; ACC], &[i16; ACC], &[i16], &[usize], &[usize]);
type ApplyI8IntoFn = fn(&mut [i16; ACC], &[i16; ACC], &[i8], &[usize], &[usize]);
// TWO-DESTINATION in-place apply (perf study 9 item 4): `acc ±= deltas`, and
// the result is ALSO written to `out`. The Finny cached refresh needs the same
// bytes in two places — the cache entry (its memo for the next refresh with
// this king key) and the search stack's accumulator slot — and used to get
// there by folding in place and then `memmove`ing 2×1280 i16 to the stack.
// Both destinations are written from the SAME registers the fold already
// holds, so the second home costs one store per block instead of a whole
// read+write pass. `acc` and `out` may not alias.
type ApplyI16DualFn = fn(&mut [i16; ACC], &mut [i16; ACC], &[i16], &[usize], &[usize]);
type L1DotFn = fn(&mut [i32; L2], &[u8; ACC], &[u16], &[i8]);
type ActivateFn = fn(&mut [u8; ACC], &[[i16; ACC]; 2], &[[i16; ACC]; 2], usize);
type FindNnzFn = fn(&[u8; ACC], &mut [u16; ACC / 4]) -> usize;
// The explicit f32 head-tail kernels (perf study #4 item #2a — the one hot
// path that still lived on autovectorization after the ×8 D47 head growth).
// Per-lane op sequences match the scalar reference exactly (same fused
// `mul_add`, same clamp order, same i-sequential accumulation per j lane),
// so every backend is bit-identical on finite inputs.
type L1DequantFn = fn(&mut [f32; L2], &[i32; L2], &[f32; L2], f32);
type L2TailFn = fn(&mut [f32; L3], &[f32; L2], &[f32; L2 * L3], &[f32; L3]);
// The policy selected-row dot (perf study #4 item #4): i8 row × f32
// bottleneck. Reduction shape is DEFINED as 16 partial lanes (lane k sums
// elements k, k+16, …) + a halving tree — the same fixed-shape pattern as
// `propagate_l3` — so every backend is bit-identical to the scalar
// reference. NOTE this deliberately replaced the old serial element-order
// sum in `PolicyHead::row_score` (a ~128-deep add dependency chain that
// could never vectorize); the new shape's f32 rounding differs from the old
// serial loop's, which was verified non-functional by the package's
// 100-game fixed-node PGN-identity oracle before landing.
type PolicyDotFn = fn(&[i8; L2], &[f32; L2]) -> f32;
// The SKP1 skip-lane dot (perf study §6-S2): u8 activations × bucket-major
// f32 lane. Reduction shape is DEFINED as 16 partial lanes + the halving
// tree (the policy_dot discipline) so every backend is bit-identical to the
// scalar reference. NOTE this replaced the head's serial element-order sum;
// the rounding change was gated by integer-cp identity on the 800-FEN
// battery (S2 landing commit).
type SkipDotFn = fn(&[u8], &[i8]) -> i32;

struct Kernels {
    i16_deltas: ApplyI16Fn,
    i16_deltas_into: ApplyI16IntoFn,
    i16_deltas_dual: ApplyI16DualFn,
    i8_deltas_into: ApplyI8IntoFn,
    l1_dot: L1DotFn,
    activate: ActivateFn,
    find_nnz: FindNnzFn,
    l1_dequant: L1DequantFn,
    l2_tail: L2TailFn,
    policy_dot: PolicyDotFn,
    skip_dot_i8: SkipDotFn,
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
            // vbmi2+vl unlock the vpcompressw find_nnz (item #2d); both are
            // present on every Zen 4 / Ice Lake+ part, but the plain table
            // keeps pre-VBMI2 AVX-512 CPUs (e.g. Skylake-X never gets here —
            // no VNNI — so this is belt-and-braces).
            if std::arch::is_x86_feature_detected!("avx512vbmi2")
                && std::arch::is_x86_feature_detected!("avx512vl")
            {
                return avx512::KERNELS_VBMI2;
            }
            return avx512::KERNELS;
        }
        // The f32 tail kernels use fused-multiply-add; every AVX2 chess-era
        // CPU has FMA, and an exotic AVX2-without-FMA part falls to scalar.
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return avx2::KERNELS;
        }
    }
    scalar::KERNELS
});

/// Compile-time backend selection (perf study #4 item #2c; the design/11
/// §3.4.6 escape hatch, Reckless pattern): when the build target statically
/// carries the same AVX-512 features the runtime dispatcher would detect
/// (`-C target-cpu=native` on a VNNI-era CPU), route straight to the avx512
/// backend — no LazyLock initialized-check, no indirect call, and the kernel
/// bodies can inline into their call sites. Portable and x86-64-v3 builds
/// keep the runtime table: v3 has no AVX-512, and statically pinning AVX2
/// there would LOSE the runtime-detected AVX-512 kernels.
macro_rules! kernel_dispatch {
    ($field:ident($($arg:expr),*)) => {{
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni"
        ))]
        {
            avx512::$field($($arg),*)
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni"
        )))]
        {
            (KERNELS.$field)($($arg),*)
        }
    }};
}

#[inline]
pub(crate) fn apply_i16_deltas(
    acc: &mut [i16; ACC],
    weights: &[i16],
    adds: &[usize],
    subs: &[usize],
) {
    kernel_dispatch!(i16_deltas(acc, weights, adds, subs))
}

/// Fused copy+apply for the HalfKA (i16) fold: `dst = src ± deltas` in one
/// pass, avoiding the separate parent→child memmove.
#[inline]
pub(crate) fn apply_i16_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i16],
    adds: &[usize],
    subs: &[usize],
) {
    kernel_dispatch!(i16_deltas_into(dst, src, weights, adds, subs))
}

/// Two-destination in-place apply: `acc ±= deltas`, and the same result is
/// written to `out` as well (perf study 9 item 4). See [`ApplyI16DualFn`].
/// `acc` and `out` must not alias.
#[inline]
pub(crate) fn apply_i16_deltas_dual(
    acc: &mut [i16; ACC],
    out: &mut [i16; ACC],
    weights: &[i16],
    adds: &[usize],
    subs: &[usize],
) {
    kernel_dispatch!(i16_deltas_dual(acc, out, weights, adds, subs))
}

/// Fused copy+apply for the FullThreats (i8-weight) fold.
#[inline]
pub(crate) fn apply_i8_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i8],
    adds: &[usize],
    subs: &[usize],
) {
    kernel_dispatch!(i8_deltas_into(dst, src, weights, adds, subs))
}

/// Sparse `[u8; ACC] × [i8; L2]` dot product. `weights` is laid out as
/// `[group][output][4]` (output ∈ 0..L2), and `nnz` contains active four-byte
/// group indices.
#[inline]
pub(crate) fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
    // The SIMD backends read each group's four FT bytes as ONE unaligned
    // 32-bit load (see [`ft_group`]), which is what lets `vpbroadcastd` fold
    // the load into the broadcast with no range check. `find_nnz` is the only
    // producer of `nnz` and only ever emits group indices in `0..ACC/4`; this
    // is the debug-time proof of that invariant — checked once per group over
    // the whole test suite instead of once per group in the shipped kernel.
    debug_assert!(
        nnz.iter().all(|&g| usize::from(g) < ACC / 4),
        "l1_dot: nnz group index out of range (find_nnz contract broken)"
    );
    kernel_dispatch!(l1_dot(pre, ft, nnz, weights))
}

/// The four FT bytes of `group`, as the little-endian `i32` the `dpbusd` /
/// `maddubs` kernels broadcast.
///
/// Written as one unaligned 32-bit read rather than
/// `i32::from_le_bytes([ft[g * 4], …])` **for the codegen, not the arithmetic**
/// — the two are bit-identical on any little-endian target. Indexing the array
/// made LLVM emit a `cmp $0x13f,%ecx` / `ja <panic>` pair it could not hoist
/// out of the group loop (it cannot see `find_nnz`'s range), and that loop is
/// only ~11 instructions long and runs ~39 times per NODE. Removing the pair
/// measured **−1.75% of all engine instructions and −5.25% of all engine
/// branches**, node-identical and eval bit-exact (perf study 8 §4.1).
///
/// # Safety
/// `group < ACC / 4`, i.e. `group` came from [`find_nnz`]. The `debug_assert`
/// in [`l1_dot`] is the checked form of this contract.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn ft_group(ft: &[u8; ACC], group: usize) -> i32 {
    // SAFETY: `group < ACC/4` ⇒ `group * 4 + 3 < ACC`, so all four bytes lie
    // inside `ft`. `read_unaligned` is required — `ft` is only byte-aligned.
    unsafe { ft.as_ptr().add(group * 4).cast::<i32>().read_unaligned() }
}

#[inline]
pub(crate) fn activate(
    out: &mut [u8; ACC],
    pst: &[[i16; ACC]; 2],
    threat: &[[i16; ACC]; 2],
    stm: usize,
) {
    kernel_dispatch!(activate(out, pst, threat, stm))
}

/// Compact the indices of the nonzero four-byte groups of `ft` into `out`
/// (ascending), returning the count — the sparse-group set [`l1_dot`]
/// consumes. Every backend produces the identical index set. Statically
/// dispatched only when vbmi2+vl are ALSO compile-time present (the vbmi2
/// variant is strictly better than the avx2 form the plain avx512 table
/// reuses); otherwise the runtime table decides, which detects vbmi2 itself.
#[inline]
pub(crate) fn find_nnz(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi2"
    ))]
    {
        avx512::find_nnz_vbmi2(ft, out)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi2"
    )))]
    {
        (KERNELS.find_nnz)(ft, out)
    }
}

/// Dequantize + SCReLU the L1 pre-activations: per lane,
/// `out[j] = clamp(pre[j]·dequant + bias[j], 0, 1)²` with a fused mul-add —
/// exactly the scalar head loop it replaces (perf study #4 item #2a).
#[inline]
pub(crate) fn l1_dequant(out: &mut [f32; L2], pre: &[i32; L2], bias: &[f32; L2], dequant: f32) {
    kernel_dispatch!(l1_dequant(out, pre, bias, dequant))
}

/// The dense f32 L2→L3 tail: `out = SCReLU(bias + Σ_i weights[i]·l1[i])`,
/// j-lane order preserved and i-sequential per lane — bit-identical to the
/// scalar mul_add loop (perf study #4 item #2a: at D47 dims this is 4096
/// FMAs/eval and was the last autovectorized hot path).
#[inline]
pub(crate) fn l2_tail(
    out: &mut [f32; L3],
    l1: &[f32; L2],
    weights: &[f32; L2 * L3],
    bias: &[f32; L3],
) {
    kernel_dispatch!(l2_tail(out, l1, weights, bias))
}

/// Policy selected-row dot: `Σ row[i]·h[i]` with the fixed 16-partial-lane +
/// halving-tree reduction shape (see the [`PolicyDotFn`] note).
#[inline]
pub(crate) fn policy_dot(row: &[i8; L2], h: &[f32; L2]) -> f32 {
    kernel_dispatch!(policy_dot(row, h))
}

/// SKP1 skip-lane dot, INTEGER: `Σ ft[i]·lane[i]` with `ft` u8 and `lane` the
/// i8-quantised bucket-major slice ([`super::sota_head::SkipLane`]).
///
/// i32 accumulation, so the sum is exact and the reduction ORDER is free —
/// unlike the f32 form this replaces, which had to pin a 16-lane + halving-tree
/// shape across every backend to keep the backends agreeing with each other.
#[inline]
pub(crate) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
    debug_assert_eq!(ft.len(), lane.len());
    debug_assert_eq!(ft.len() % 64, 0);
    kernel_dispatch!(skip_dot_i8(ft, lane))
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

    pub(super) fn i16_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        for lane in 0..ACC {
            let mut value = src[lane];
            for &feature in adds {
                value = value.wrapping_add(weights[feature * ACC + lane]);
            }
            for &feature in subs {
                value = value.wrapping_sub(weights[feature * ACC + lane]);
            }
            dst[lane] = value;
        }
    }

    pub(super) fn i16_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
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
            out[lane] = value;
        }
    }

    pub(super) fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        for lane in 0..ACC {
            let mut value = src[lane];
            for &feature in adds {
                value = value.wrapping_add(i16::from(weights[feature * ACC + lane]));
            }
            for &feature in subs {
                value = value.wrapping_sub(i16::from(weights[feature * ACC + lane]));
            }
            dst[lane] = value;
        }
    }

    pub(super) fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        for &group in nnz {
            let group = usize::from(group);
            for output in 0..L2 {
                let mut dot = 0i32;
                for lane in 0..4 {
                    dot += i32::from(ft[group * 4 + lane])
                        * i32::from(weights[group * (L2 * 4) + output * 4 + lane]);
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
                // 255/64 carried as 4 (see the avx512 twin).
                let merge =
                    |p: i16, t: i16| (i32::from(p) + (i32::from(t) << 2)).clamp(0, FT_QUANT);
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

    /// The bit-exact reference for the L1 dequant+SCReLU (the loop moved here
    /// verbatim from `SotaHead::propagate_l1`).
    pub(super) fn l1_dequant(out: &mut [f32; L2], pre: &[i32; L2], bias: &[f32; L2], dequant: f32) {
        for j in 0..L2 {
            let clipped = (pre[j] as f32).mul_add(dequant, bias[j]).clamp(0.0, 1.0);
            out[j] = clipped * clipped;
        }
    }

    /// The bit-exact reference for the dense L2→L3 tail (moved verbatim from
    /// `SotaHead::propagate_l2`): bias-seeded, i-outer/j-inner fused mul-add —
    /// the i-sequential accumulation per j lane is the pinned order — then
    /// SCReLU.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn l2_tail(
        out: &mut [f32; L3],
        l1: &[f32; L2],
        weights: &[f32; L2 * L3],
        bias: &[f32; L3],
    ) {
        out.copy_from_slice(bias);
        for i in 0..L2 {
            let row: &[f32; L3] = weights[i * L3..i * L3 + L3]
                .try_into()
                .expect("l2 weight row");
            let li = l1[i];
            for j in 0..L3 {
                out[j] = row[j].mul_add(li, out[j]);
            }
        }
        for v in out.iter_mut() {
            let clipped = v.clamp(0.0, 1.0);
            *v = clipped * clipped;
        }
    }

    /// The bit-exact reference for the policy row dot: 16 partial lanes
    /// (lane k accumulates elements k, k+16, …, each a fused mul-add) then
    /// the same halving tree as `propagate_l3`.
    pub(super) fn policy_dot(row: &[i8; L2], h: &[f32; L2]) -> f32 {
        const LANES: usize = 16;
        let mut sums = [0.0f32; LANES];
        let mut i = 0;
        while i < L2 {
            for j in 0..LANES {
                sums[j] = f32::from(row[i + j]).mul_add(h[i + j], sums[j]);
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
        sums[0]
    }

    /// The reference for the skip-lane dot: a plain i32 sum. Integer addition
    /// is associative, so unlike the f32 form this replaced there is no lane
    /// shape to pin — every backend is free to reduce in whatever order is
    /// fastest and they still agree bit-for-bit.
    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        let mut acc = 0i32;
        for (&a, &w) in ft.iter().zip(lane) {
            acc += i32::from(a) * i32::from(w);
        }
        acc
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i16_deltas_into,
        i16_deltas_dual,
        i8_deltas_into,
        l1_dot,
        activate,
        find_nnz,
        l1_dequant,
        l2_tail,
        policy_dot,
        skip_dot_i8,
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
    unsafe fn i16_deltas_dual_inner(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // `i16_deltas_inner` with a second store per block — same loads, same
        // per-lane wrapping adds in the same order, so bit-identical.
        // SAFETY: as `i16_deltas_inner`; `out` is a disjoint ACC-wide array.
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
                _mm256_storeu_si256(out.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    pub(super) fn i16_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime avx2 detection.
        unsafe { i16_deltas_dual_inner(acc, out, weights, adds, subs) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l1_dot_inner(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // maddubs forms adjacent two-product sums. With ft<=127 and an i8
        // weight those sums fit i16 at both extrema; madd_epi16(..., 1) then
        // widens each four-product dot to i32, matching Bullet exactly.
        // Each 16-output block is two 8-lane i32 registers; the outer loop
        // walks the L2/16 blocks so a wider head bottleneck scales for free.
        // SAFETY: active groups address four FT bytes and, per block, sixteen
        // complete four-byte output rows in loader-validated storage.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            for blk in 0..(L2 / L2_BLK) {
                let pbase = blk * L2_BLK; // 16 outputs per block
                let wblk = blk * 64; // 16 outputs × 4 bytes within each group
                let mut lo = _mm256_loadu_si256(pre.as_ptr().add(pbase).cast());
                let mut hi = _mm256_loadu_si256(pre.as_ptr().add(pbase + 8).cast());
                for &group in nnz {
                    let group = usize::from(group);
                    let inputs = _mm256_set1_epi32(ft_group(ft, group));
                    let base = weights.as_ptr().add(group * (L2 * 4) + wblk);
                    let wlo = _mm256_loadu_si256(base.cast());
                    let whi = _mm256_loadu_si256(base.add(32).cast());
                    let plo = _mm256_maddubs_epi16(inputs, wlo);
                    let phi = _mm256_maddubs_epi16(inputs, whi);
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(plo, ones));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(phi, ones));
                }
                _mm256_storeu_si256(pre.as_mut_ptr().add(pbase).cast(), lo);
                _mm256_storeu_si256(pre.as_mut_ptr().add(pbase + 8).cast(), hi);
            }
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
                        // 255/64 carried as 4 (see the avx512 twin).
                        let scaled = _mm256_slli_epi32::<2>(tv);
                        _mm256_min_epi32(
                            _mm256_max_epi32(_mm256_add_epi32(pv, scaled), zero),
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

    pub(super) fn i16_deltas(
        acc: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i16_deltas_inner(acc, weights, adds, subs) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn i16_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as i16_deltas_inner; reads `src`, writes the disjoint `dst`.
        // §6-F (perf study): pre-scale each feature's row offset ONCE — the
        // block-outer nest re-derived `feature * ACC` per (block, feature)
        // pair (the kernels' dominant scalar/GPR share, §2.4). Chunked so
        // arbitrarily long event lists stay allocation-free; later passes
        // fold onto dst (wrapping adds commute ⇒ bit-identical).
        //
        // KEEP THE HOIST HERE. Perf study 12 §4 removed it from the AVX-512
        // twins and measured −114 i/n, but the two backends have different
        // nests and the hoist's value scales with the OUTER trip count: this
        // one is block-outer, `ACC / LANES` = 40 iterations, so the hoist saves
        // 39 x rows multiplies against ~91 instructions of prologue; the
        // AVX-512 form holds a 20-register tile and runs its outer loop TWICE,
        // where it saved only 1 x rows and the prologue dominated.
        unsafe {
            const CAP: usize = 96;
            // MaybeUninit: only the written prefix is ever read — the zeroing
            // memset of 2x96 usizes per call cost more than the hoist saved
            // on the <=2-event HalfKA folds (measured flat on first cut).
            let mut a_off = [core::mem::MaybeUninit::<usize>::uninit(); CAP];
            let mut s_off = [core::mem::MaybeUninit::<usize>::uninit(); CAP];
            let (mut ai, mut si) = (0usize, 0usize);
            let mut source: *const i16 = src.as_ptr();
            loop {
                let an = (adds.len() - ai).min(CAP);
                let sn = (subs.len() - si).min(CAP);
                for k in 0..an {
                    a_off[k].write(adds[ai + k] * ACC);
                }
                for k in 0..sn {
                    s_off[k].write(subs[si + k] * ACC);
                }
                for off in (0..ACC).step_by(LANES) {
                    let mut value = _mm256_loadu_si256(source.add(off).cast());
                    for base in a_off[..an].iter().map(|m| m.assume_init()) {
                        let row = weights.as_ptr().add(base + off);
                        value = _mm256_add_epi16(value, _mm256_loadu_si256(row.cast()));
                    }
                    for base in s_off[..sn].iter().map(|m| m.assume_init()) {
                        let row = weights.as_ptr().add(base + off);
                        value = _mm256_sub_epi16(value, _mm256_loadu_si256(row.cast()));
                    }
                    _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), value);
                }
                ai += an;
                si += sn;
                if ai >= adds.len() && si >= subs.len() {
                    break;
                }
                source = dst.as_ptr();
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn i8_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as i8_deltas_inner; reads `src`, writes the disjoint `dst`.
        // §6-F hoist + chunking — see i16_deltas_into_inner above.
        unsafe {
            const CAP: usize = 96;
            // MaybeUninit: only the written prefix is ever read — the zeroing
            // memset of 2x96 usizes per call cost more than the hoist saved
            // on the <=2-event HalfKA folds (measured flat on first cut).
            let mut a_off = [core::mem::MaybeUninit::<usize>::uninit(); CAP];
            let mut s_off = [core::mem::MaybeUninit::<usize>::uninit(); CAP];
            let (mut ai, mut si) = (0usize, 0usize);
            let mut source: *const i16 = src.as_ptr();
            loop {
                let an = (adds.len() - ai).min(CAP);
                let sn = (subs.len() - si).min(CAP);
                for k in 0..an {
                    a_off[k].write(adds[ai + k] * ACC);
                }
                for k in 0..sn {
                    s_off[k].write(subs[si + k] * ACC);
                }
                for off in (0..ACC).step_by(LANES) {
                    let mut value = _mm256_loadu_si256(source.add(off).cast());
                    for base in a_off[..an].iter().map(|m| m.assume_init()) {
                        let bytes = _mm_loadu_si128(weights.as_ptr().add(base + off).cast());
                        value = _mm256_add_epi16(value, _mm256_cvtepi8_epi16(bytes));
                    }
                    for base in s_off[..sn].iter().map(|m| m.assume_init()) {
                        let bytes = _mm_loadu_si128(weights.as_ptr().add(base + off).cast());
                        value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(bytes));
                    }
                    _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), value);
                }
                ai += an;
                si += sn;
                if ai >= adds.len() && si >= subs.len() {
                    break;
                }
                source = dst.as_ptr();
            }
        }
    }

    fn i16_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i16_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i8_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
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

    #[target_feature(enable = "avx2,fma")]
    unsafe fn l1_dequant_inner(
        out: &mut [f32; L2],
        pre: &[i32; L2],
        bias: &[f32; L2],
        dequant: f32,
    ) {
        // SAFETY: each iteration touches one in-bounds 8-lane block. Per lane
        // this is exactly the scalar reference: fused mul-add, then
        // clamp(0,1) as max-then-min, then square — bit-identical for the
        // finite values a loader-validated net produces.
        unsafe {
            let dq = _mm256_set1_ps(dequant);
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            for off in (0..L2).step_by(8) {
                let p = _mm256_cvtepi32_ps(_mm256_loadu_si256(pre.as_ptr().add(off).cast()));
                let b = _mm256_loadu_ps(bias.as_ptr().add(off));
                let v = _mm256_fmadd_ps(p, dq, b);
                let clipped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
                _mm256_storeu_ps(out.as_mut_ptr().add(off), _mm256_mul_ps(clipped, clipped));
            }
        }
    }

    fn l1_dequant(out: &mut [f32; L2], pre: &[i32; L2], bias: &[f32; L2], dequant: f32) {
        // SAFETY: installed only after runtime AVX2+FMA detection.
        unsafe { l1_dequant_inner(out, pre, bias, dequant) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn l2_tail_inner(
        out: &mut [f32; L3],
        l1: &[f32; L2],
        weights: &[f32; L2 * L3],
        bias: &[f32; L3],
    ) {
        // Four 8-lane accumulators = the L3=32 output row, seeded from the
        // bias; per i one broadcast + four fused mul-adds — j-lane order
        // preserved and i-sequential per lane, so the accumulation is
        // bit-identical to the scalar reference. SAFETY: all loads/stores
        // cover in-bounds 8-lane blocks of loader-validated storage.
        unsafe {
            let mut acc0 = _mm256_loadu_ps(bias.as_ptr());
            let mut acc1 = _mm256_loadu_ps(bias.as_ptr().add(8));
            let mut acc2 = _mm256_loadu_ps(bias.as_ptr().add(16));
            let mut acc3 = _mm256_loadu_ps(bias.as_ptr().add(24));
            for (i, &input) in l1.iter().enumerate() {
                let li = _mm256_set1_ps(input);
                let row = weights.as_ptr().add(i * L3);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row), li, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(8)), li, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(16)), li, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(24)), li, acc3);
            }
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let screlu = |acc| {
                let clipped = _mm256_min_ps(_mm256_max_ps(acc, zero), one);
                _mm256_mul_ps(clipped, clipped)
            };
            _mm256_storeu_ps(out.as_mut_ptr(), screlu(acc0));
            _mm256_storeu_ps(out.as_mut_ptr().add(8), screlu(acc1));
            _mm256_storeu_ps(out.as_mut_ptr().add(16), screlu(acc2));
            _mm256_storeu_ps(out.as_mut_ptr().add(24), screlu(acc3));
        }
    }

    fn l2_tail(out: &mut [f32; L3], l1: &[f32; L2], weights: &[f32; L2 * L3], bias: &[f32; L3]) {
        // SAFETY: installed only after runtime AVX2+FMA detection.
        unsafe { l2_tail_inner(out, l1, weights, bias) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn policy_dot_inner(row: &[i8; L2], h: &[f32; L2]) -> f32 {
        // Two 8-lane accumulators = the 16 partial lanes of the scalar
        // reference (acc_lo = lanes 0-7, acc_hi = lanes 8-15); the halving
        // tree runs scalar over the spilled lanes, matching the reference
        // reduction order exactly. SAFETY: loads cover in-bounds 8-byte /
        // 8-lane blocks of the fixed-size inputs.
        unsafe {
            let mut acc_lo = _mm256_setzero_ps();
            let mut acc_hi = _mm256_setzero_ps();
            for i in (0..L2).step_by(16) {
                let bytes_lo = _mm_loadl_epi64(row.as_ptr().add(i).cast());
                let bytes_hi = _mm_loadl_epi64(row.as_ptr().add(i + 8).cast());
                let w_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_lo));
                let w_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes_hi));
                acc_lo = _mm256_fmadd_ps(w_lo, _mm256_loadu_ps(h.as_ptr().add(i)), acc_lo);
                acc_hi = _mm256_fmadd_ps(w_hi, _mm256_loadu_ps(h.as_ptr().add(i + 8)), acc_hi);
            }
            let mut sums = [0.0f32; 16];
            _mm256_storeu_ps(sums.as_mut_ptr(), acc_lo);
            _mm256_storeu_ps(sums.as_mut_ptr().add(8), acc_hi);
            let mut stride = 8;
            while stride > 0 {
                for k in 0..stride {
                    sums[k] += sums[k + stride];
                }
                stride /= 2;
            }
            sums[0]
        }
    }

    fn policy_dot(row: &[i8; L2], h: &[f32; L2]) -> f32 {
        // SAFETY: installed only after runtime AVX2+FMA detection.
        unsafe { policy_dot_inner(row, h) }
    }

    #[target_feature(enable = "avx2,fma")]
    #[target_feature(enable = "avx2")]
    unsafe fn skip_dot_i8_inner(ft: &[u8], lane: &[i8]) -> i32 {
        // maddubs forms adjacent u8 x i8 two-product sums; with ft <= 127 and
        // |w| <= 127 each pair is at most 2*127*127 = 32258 < i16::MAX, so the
        // saturating step never fires. madd_epi16 then widens to i32.
        // SAFETY: caller guarantees equal lengths, a multiple of 64.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            for i in (0..ft.len()).step_by(32) {
                let a = _mm256_loadu_si256(ft.as_ptr().add(i).cast());
                let w = _mm256_loadu_si256(lane.as_ptr().add(i).cast());
                let prod = _mm256_maddubs_epi16(a, w);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
            }
            let mut out = [0i32; 8];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), acc);
            out.iter().sum()
        }
    }

    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        // SAFETY: installed only after runtime avx2+fma detection.
        unsafe { skip_dot_i8_inner(ft, lane) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i16_deltas_into,
        i16_deltas_dual,
        i8_deltas_into,
        l1_dot,
        activate,
        find_nnz,
        l1_dequant,
        l2_tail,
        policy_dot,
        skip_dot_i8,
    };
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    const LANES: usize = 32;
    const _: () = assert!(ACC.is_multiple_of(LANES));

    /// zmm blocks in one accumulator (40 at the shipping ACC = 1280).
    const BLOCKS: usize = ACC / LANES;

    /// Accumulator blocks held in registers across the feature loop.
    ///
    /// Perf study 12 §7 — **−314 i/n, the study's largest single win.** The
    /// fold's arithmetic is fixed (`rows x BLOCKS` load/add pairs) but its LOOP
    /// CONTROL is `tiles x rows x ~6` — load the feature, `lea`+`shl` for
    /// `feature * ACC`, bump, compare, branch — so it scales with the number of
    /// tiles and not with the work. Study 2 chose `REGS = 8`, i.e. **five**
    /// tiles at ACC = 1280, spending ~220 i/n of control on ~7.1 rows per node.
    /// AVX-512 has 32 zmm and the body needs only one live register beyond the
    /// accumulators, so the tile can be far wider.
    ///
    /// Exact divisors of `BLOCKS` only: a non-dividing tile pushes the
    /// remainder into the one-block tail loop, which costs more control than it
    /// saves. 1280 -> 20 (2 tiles), 768 -> 12 (2), 1536 -> 16 (3), 2048 -> 16 (4).
    /// Measured: 20 is the optimum at 1280 and the largest divisor that fits.
    /// `REGS = 40` (one tile, no tile loop at all) was measured too and is a
    /// **disaster** — 6581.5 -> 7589.2 i/n — because 40 accumulators spill.
    const fn pick_regs() -> usize {
        let mut r = 20;
        while r > 8 {
            if BLOCKS.is_multiple_of(r) {
                return r;
            }
            r -= 1;
        }
        8
    }
    const REGS: usize = pick_regs();
    // The win depends on the tile dividing the accumulator EXACTLY: a remainder
    // falls into the one-block `while off < ACC` tail, whose control cost per
    // block is what the wide tile exists to remove. True for every shipped
    // width — 1280/20, 768/12, 1536/16, 2048/16 all give remainder 0.
    const _: () = assert!(
        BLOCKS.is_multiple_of(REGS),
        "fold tile must divide the accumulator exactly"
    );
    // One zmm is live beyond the accumulators (the row being folded in).
    const _: () = assert!(REGS < 32, "fold tile must leave a spare zmm");

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
            // Study-2 option B: REGS-wide tile — hold 8 blocks in registers
            // across the feature loop (8x less loop control + feature-base
            // traffic). Per-lane wrapping adds, same per-lane order =>
            // bit-identical. ACC/LANES is a multiple of 8 for every arch we
            // build (24/40/48/80); the remainder loop keeps it general.
            let tile = LANES * REGS;
            let mut off = 0;
            while off + tile <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(acc.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_add_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_sub_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += tile;
            }
            while off < ACC {
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
                off += LANES;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i16_deltas_dual_inner(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // `i16_deltas_inner` with a second store per block. Same loads, same
        // 8-register tile, same per-lane wrapping adds in the same order ⇒
        // bit-identical to fold-then-copy, at one extra store per 32 lanes
        // instead of a 2560 B memmove (perf study 9 item 4).
        // SAFETY: as `i16_deltas_inner`; `out` is a disjoint ACC-wide array.
        unsafe {
            let tile = LANES * REGS;
            let mut off = 0;
            while off + tile <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(acc.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_add_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_sub_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                    _mm512_storeu_si512(out.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += tile;
            }
            while off < ACC {
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
                _mm512_storeu_si512(out.as_mut_ptr().add(off).cast(), value);
                off += LANES;
            }
        }
    }

    pub(super) fn i16_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection (or
        // compiled with the features statically enabled).
        unsafe { i16_deltas_dual_inner(acc, out, weights, adds, subs) }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i16_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as i16_deltas_inner; reads `src`, writes the disjoint `dst`.
        //
        // Perf study 12 §4: this used to hoist `feature * ACC` into a
        // `[MaybeUninit<usize>; 96]` stack array, chunking the feature lists to
        // fit it. The hoist saves `(tiles - 1) * rows` multiplies — 4 x 2.1 for
        // a HalfKA fold — and cost **91 executed instructions of prologue** per
        // call: a 1656-byte stack frame, six callee-saved pushes, the CAP
        // min/cmov pair per list, and (because the trip count is a runtime
        // value) LLVM's fully generic vectorized offset loop, a `vpmullq`
        // 8-wide body plus its scalar remainder, emitted twice — for a loop
        // whose measured trip count is ONE. `feature * ACC` with ACC = 1280 is
        // `lea (f,f,4)` + `shl $8`, and it is now paid once per tile. The
        // in-place twins (`i16_deltas_inner`, `i16_deltas_dual_inner`) never
        // had the hoist, which is why only the `_into` pair carried the cost.
        //
        // Bit-identical: same rows, same per-lane wrapping i16 adds, same
        // order. Dropping the chunking is also bit-identical — mod-2^16
        // addition is associative, so applying all rows in one pass equals
        // applying them 96 at a time through `dst`.
        unsafe {
            let tile = LANES * REGS;
            let mut off = 0;
            while off + tile <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_add_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        *vk = _mm512_sub_epi16(*vk, _mm512_loadu_si512(row.add(k * LANES).cast()));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += tile;
            }
            while off < ACC {
                let mut value = _mm512_loadu_si512(src.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm512_add_epi16(value, _mm512_loadu_si512(row.cast()));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    value = _mm512_sub_epi16(value, _mm512_loadu_si512(row.cast()));
                }
                _mm512_storeu_si512(dst.as_mut_ptr().add(off).cast(), value);
                off += LANES;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: as i8_deltas_inner; reads `src`, writes the disjoint `dst`.
        // Perf study 12 §4 — the offset hoist + CAP chunking is gone; see the
        // i16 twin above for the full rationale and the bit-identity argument.
        unsafe {
            let tile = LANES * REGS;
            let mut off = 0;
            while off + tile <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature * ACC + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += tile;
            }
            while off < ACC {
                let mut value = _mm512_loadu_si512(src.as_ptr().add(off).cast());
                for &feature in adds {
                    let bytes =
                        _mm256_loadu_si256(weights.as_ptr().add(feature * ACC + off).cast());
                    value = _mm512_add_epi16(value, _mm512_cvtepi8_epi16(bytes));
                }
                for &feature in subs {
                    let bytes =
                        _mm256_loadu_si256(weights.as_ptr().add(feature * ACC + off).cast());
                    value = _mm512_sub_epi16(value, _mm512_cvtepi8_epi16(bytes));
                }
                _mm512_storeu_si512(dst.as_mut_ptr().add(off).cast(), value);
                off += LANES;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn l1_dot_inner(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // dpbusd is the whole kernel: for each dword lane it accumulates the
        // u8×i8 four-product dot directly into i32 — the maddubs+madd pair
        // (and its i16 saturation reasoning) collapses into one instruction
        // per (group, block). Loop order is GROUP-outer with all L2/16 output
        // blocks held in registers (perf study #4 item #2b): the blk-outer
        // form re-walked the nnz list and re-broadcast each 4-byte FT group
        // once per block — free at L2=16 (1 block), 7 redundant passes at the
        // D47 L2=128 (8 blocks, 8 zmm accumulators — well inside the 32-reg
        // file). i32 wrapping adds are order-independent ⇒ bit-identical.
        // SAFETY: active groups address four FT bytes and L2 complete
        // four-byte output rows in loader-validated storage.
        unsafe {
            const BLOCKS: usize = L2 / L2_BLK;
            let mut sums = [_mm512_setzero_si512(); BLOCKS];
            for (blk, sum) in sums.iter_mut().enumerate() {
                *sum = _mm512_loadu_si512(pre.as_ptr().add(blk * L2_BLK).cast());
            }
            // NOTE: the study's "prefetch the next group's rows" was tried
            // here and measured noise-to-negative (−0.3% over 4 interleaved
            // pairs): the per-bucket L1 table is ~160 KB and L2-resident, so
            // the loads never miss far enough to hide.
            for &group in nnz {
                let group = usize::from(group);
                let inputs = _mm512_set1_epi32(ft_group(ft, group));
                let base = weights.as_ptr().add(group * (L2 * 4));
                for (blk, sum) in sums.iter_mut().enumerate() {
                    let rows = _mm512_loadu_si512(base.add(blk * 64).cast());
                    *sum = _mm512_dpbusd_epi32(*sum, inputs, rows);
                }
            }
            for (blk, sum) in sums.iter().enumerate() {
                _mm512_storeu_si512(pre.as_mut_ptr().add(blk * L2_BLK).cast(), *sum);
            }
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
        // Study-2 option D (D1+D2 as one i16 pipeline, 32 lanes/zmm — the
        // old path widened to i32 at 16 lanes/zmm, 2.54x SF's cost):
        //   4t          via 2x vpaddsw   (saturates only for |t| > 8191 —
        //                                 measured max |t| = 431 over the
        //                                 800-FEN battery + 6k corpus
        //                                 positions; debug builds trip the
        //                                 assert below if a future net
        //                                 breaks the bound)
        //   |4t|        via vpabsw
        //   round(|t|·255/64) = (|4t|·32640 + 2^14) >> 15  via vpmulhrsw —
        //     EXACT: (4|t|·32640+16384)>>15 = 512·(|t|·255+32)>>15
        //     = (|t|·255+32)>>6, the scalar reference's magnitude rounding.
        //     NOTE the study's original recipe applied vpmulhrsw to the
        //     SIGNED value; that rounds negative half-cases toward +inf and
        //     diverges from the reference by 1 for every t ≡ 32 (mod 64),
        //     t < 0 — hence abs/mulhrsw/sign, mirroring the i32 path's own
        //     sign-reapplication shape.
        //   sign        via vpsignw (t == 0 ⇒ 0, matching signum()·…)
        //   p + r       via vpaddsw — saturation only occurs beyond the
        //               [0,255] clamp, so the clamped result is exact
        //   clamp       via vpmaxsw/vpminsw
        //   (l·r)>>9    via vpsllw<7> + vpmulhw (l ≤ 255 ⇒ l·128 < 2^15)
        //   store       via vpmovwb — products ≤ 127, truncation == value.
        // SAFETY: each iteration consumes 32 in-bounds i16 lanes per half.
        unsafe {
            #[cfg(debug_assertions)]
            for side in threat {
                for &t in side.iter() {
                    debug_assert!(
                        t.unsigned_abs() <= 8191,
                        "threat lane {t} exceeds the i16 activate bound"
                    );
                }
            }
            let zero = _mm512_setzero_si512();
            let ceiling = _mm512_set1_epi16(FT_QUANT as i16);
            for flip in 0..2 {
                let perspective = stm ^ flip;
                let p = &pst[perspective];
                let t = &threat[perspective];
                for lane in (0..H).step_by(32) {
                    // The threat rescale, as ONE SHIFT.
                    //
                    // The two FT transformers are quantised differently —
                    // HalfKA at FT_QUANT = 255, FullThreats at
                    // THREAT_QUANT = 64 — so the merge used to carry a
                    // six-op `round(t · 255/64)`: abs / shift / mulhrs /
                    // compare-mask / masked-sub / add. That was 24 of this
                    // loop body's 48 instructions and ~2.2% of ALL engine
                    // instructions, where Stockfish — whose two 1024-wide
                    // accumulators share one frame — merges its own with a
                    // single `vec_add_16` (`nnue_feature_transformer.h:337`).
                    //
                    // 255/64 = 3.984375, and the ratio does not have to be
                    // exact: it is a GAIN on one eval channel, and moving it to
                    // 4 is a 0.39% systematic change, well inside the net's own
                    // quantisation noise. So the merge is `t << 2` — i.e.
                    // THREAT_QUANT is effectively 63.75 rather than 64.
                    //
                    // Measured on the 800-FEN battery against the exact form:
                    // **74.0% of positions identical, mean |delta| 0.14 SF cp,
                    // p99 0.97, max 2.42, Pearson r = 0.999999.** Two rejected
                    // alternatives, both measured (perf study 8 §4.6): rebasing
                    // the stored weights into the FT_QUANT frame at load — the
                    // obvious fix — saturates 0.031% of them at i8 and measured
                    // **max 205 SF cp**; and the exact `4t - t/64`
                    // (`slli`/`srai`/`sub`) costs an extra op and is FOUR TIMES
                    // LESS faithful (mean 0.55 SF cp), because `vpsraw` floors
                    // and that asymmetry is noisier than the 0.39% of gain it
                    // corrects.
                    let merge = |offset: usize| {
                        let pv = _mm512_loadu_si512(p.as_ptr().add(offset).cast());
                        let tv = _mm512_loadu_si512(t.as_ptr().add(offset).cast());
                        let scaled = _mm512_slli_epi16::<2>(tv);
                        let sum = _mm512_adds_epi16(pv, scaled);
                        _mm512_min_epi16(_mm512_max_epi16(sum, zero), ceiling)
                    };
                    let left = merge(lane);
                    let right = merge(lane + H);
                    let prod = _mm512_mulhi_epi16(_mm512_slli_epi16::<7>(left), right);
                    _mm256_storeu_si256(
                        out.as_mut_ptr().add(flip * H + lane).cast(),
                        _mm512_cvtepi16_epi8(prod),
                    );
                }
            }
        }
    }

    pub(super) fn i16_deltas(
        acc: &mut [i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i16_deltas_inner(acc, weights, adds, subs) }
    }

    pub(super) fn i16_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i16],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i16_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    pub(super) fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[usize],
        subs: &[usize],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i8_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    pub(super) fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { l1_dot_inner(pre, ft, nnz, weights) }
    }

    pub(super) fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        stm: usize,
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { activate_inner(out, pst, threat, stm) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn l1_dequant_inner(
        out: &mut [f32; L2],
        pre: &[i32; L2],
        bias: &[f32; L2],
        dequant: f32,
    ) {
        // SAFETY: each iteration touches one in-bounds 16-lane block. Per lane
        // this is exactly the scalar reference: fused mul-add, then clamp(0,1)
        // as max-then-min, then square — bit-identical for the finite values a
        // loader-validated net produces.
        unsafe {
            let dq = _mm512_set1_ps(dequant);
            let zero = _mm512_setzero_ps();
            let one = _mm512_set1_ps(1.0);
            for off in (0..L2).step_by(16) {
                let p = _mm512_cvtepi32_ps(_mm512_loadu_si512(pre.as_ptr().add(off).cast()));
                let b = _mm512_loadu_ps(bias.as_ptr().add(off));
                let v = _mm512_fmadd_ps(p, dq, b);
                let clipped = _mm512_min_ps(_mm512_max_ps(v, zero), one);
                _mm512_storeu_ps(out.as_mut_ptr().add(off), _mm512_mul_ps(clipped, clipped));
            }
        }
    }

    pub(super) fn l1_dequant(out: &mut [f32; L2], pre: &[i32; L2], bias: &[f32; L2], dequant: f32) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection (or
        // compiled with the features statically enabled).
        unsafe { l1_dequant_inner(out, pre, bias, dequant) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn l2_tail_inner(
        out: &mut [f32; L3],
        l1: &[f32; L2],
        weights: &[f32; L2 * L3],
        bias: &[f32; L3],
    ) {
        // Two 16-lane accumulators = the L3=32 output row, seeded from the
        // bias; per i one broadcast + two fused mul-adds (perf study #4 item
        // #2a: 4096 FMAs/eval at D47 dims, previously autovectorized).
        // j-lane order preserved and i-sequential per lane ⇒ bit-identical
        // to the scalar reference. SAFETY: all loads/stores cover in-bounds
        // 16-lane blocks of loader-validated storage.
        unsafe {
            let mut acc0 = _mm512_loadu_ps(bias.as_ptr());
            let mut acc1 = _mm512_loadu_ps(bias.as_ptr().add(16));
            for (i, &input) in l1.iter().enumerate() {
                let li = _mm512_set1_ps(input);
                let row = weights.as_ptr().add(i * L3);
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row), li, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16)), li, acc1);
            }
            let zero = _mm512_setzero_ps();
            let one = _mm512_set1_ps(1.0);
            let screlu = |acc| {
                let clipped = _mm512_min_ps(_mm512_max_ps(acc, zero), one);
                _mm512_mul_ps(clipped, clipped)
            };
            _mm512_storeu_ps(out.as_mut_ptr(), screlu(acc0));
            _mm512_storeu_ps(out.as_mut_ptr().add(16), screlu(acc1));
        }
    }

    pub(super) fn l2_tail(
        out: &mut [f32; L3],
        l1: &[f32; L2],
        weights: &[f32; L2 * L3],
        bias: &[f32; L3],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection (or
        // compiled with the features statically enabled).
        unsafe { l2_tail_inner(out, l1, weights, bias) }
    }

    /// `NNZ_IOTA` — the 16 u16 lane indices `0..16`, the vpcompressw source.
    const NNZ_IOTA: [u16; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn find_nnz_vbmi2_inner(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // Sixteen 4-byte groups per iteration: one 64 B load, test the dwords
        // against zero into a 16-bit mask, vpcompressw the constant iota to
        // the mask's set positions (ascending), add the running group base —
        // the same index set as the scalar scan and the AVX2 movemask+LUT
        // form, at 20 iterations instead of 40 and no 4 KiB table
        // (perf study #4 item #2d; Zen 4 / Ice Lake+ have vbmi2).
        // SAFETY: each load reads 64 in-bounds FT bytes. Each store writes a
        // full 16-lane row at `out[count]`: entering chunk c, count <= 16c <=
        // ACC/4 - 16, so the row fits even when every group so far was
        // nonzero; lanes past the popcount are dead padding overwritten by
        // the next store or ignored via the returned count.
        unsafe {
            let iota = _mm256_loadu_si256(NNZ_IOTA.as_ptr().cast());
            let mut count = 0usize;
            for chunk in 0..ACC / 4 / 16 {
                let groups = _mm512_loadu_si512(ft.as_ptr().add(chunk * 64).cast());
                let mask = _mm512_test_epi32_mask(groups, groups);
                let row = _mm256_maskz_compress_epi16(mask, iota);
                let base = _mm256_set1_epi16((chunk * 16) as i16);
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(count).cast(),
                    _mm256_add_epi16(row, base),
                );
                count += mask.count_ones() as usize;
            }
            count
        }
    }

    pub(super) fn find_nnz_vbmi2(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // SAFETY: installed only after runtime avx512 vl+vbmi2 detection (or
        // compiled with the features statically enabled).
        unsafe { find_nnz_vbmi2_inner(ft, out) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn policy_dot_inner(row: &[i8; L2], h: &[f32; L2]) -> f32 {
        // One zmm accumulator = the 16 partial lanes of the scalar reference
        // (lane k sums elements k, k+16, …); the halving tree runs scalar
        // over the spilled lanes, matching the reference reduction exactly.
        // SAFETY: loads cover in-bounds 16-byte / 16-lane blocks of the
        // fixed-size inputs.
        unsafe {
            let mut acc = _mm512_setzero_ps();
            for i in (0..L2).step_by(16) {
                let bytes = _mm_loadu_si128(row.as_ptr().add(i).cast());
                let w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
                acc = _mm512_fmadd_ps(w, _mm512_loadu_ps(h.as_ptr().add(i)), acc);
            }
            let mut sums = [0.0f32; 16];
            _mm512_storeu_ps(sums.as_mut_ptr(), acc);
            let mut stride = 8;
            while stride > 0 {
                for k in 0..stride {
                    sums[k] += sums[k + stride];
                }
                stride /= 2;
            }
            sums[0]
        }
    }

    pub(super) fn policy_dot(row: &[i8; L2], h: &[f32; L2]) -> f32 {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection (or
        // compiled with the features statically enabled).
        unsafe { policy_dot_inner(row, h) }
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn skip_dot_i8_inner(ft: &[u8], lane: &[i8]) -> i32 {
        // `vpdpbusd` is the whole kernel: 64 u8 x i8 products accumulated into
        // 16 i32 lanes per instruction. 1280 lanes = 20 loads + 20 multiply-adds
        // where the f32 form needed 80 x (widen, convert, fma).
        // SAFETY: caller guarantees equal lengths, a multiple of 64.
        unsafe {
            let mut acc = _mm512_setzero_si512();
            for i in (0..ft.len()).step_by(64) {
                let a = _mm512_loadu_si512(ft.as_ptr().add(i).cast());
                let w = _mm512_loadu_si512(lane.as_ptr().add(i).cast());
                acc = _mm512_dpbusd_epi32(acc, a, w);
            }
            _mm512_reduce_add_epi32(acc)
        }
    }

    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { skip_dot_i8_inner(ft, lane) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i16_deltas,
        i16_deltas_into,
        i16_deltas_dual,
        i8_deltas_into,
        l1_dot,
        activate,
        // The AVX2 movemask+compact form; the 512-bit variant needs
        // vpcompressw (avx512-vbmi2+vl — see KERNELS_VBMI2).
        find_nnz: avx2::find_nnz,
        l1_dequant,
        l2_tail,
        policy_dot,
        skip_dot_i8,
    };

    /// [`KERNELS`] plus the vpcompressw `find_nnz` — selected when the CPU
    /// also reports avx512vl+vbmi2 (every Zen 4 / Ice Lake+ part).
    pub(super) const KERNELS_VBMI2: Kernels = Kernels {
        find_nnz: find_nnz_vbmi2,
        ..KERNELS
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatched_activate_matches_scalar_bitwise() {
        // Option D gate: the i16 pipeline must equal the scalar reference
        // on every lane, INCLUDING the negative half-cases (t ≡ 32 mod 64,
        // t < 0) where the study's original recipe diverged, and values up
        // to 4x the measured |t| bound.
        let mut pst = [[0i16; ACC]; 2];
        let mut threat = [[0i16; ACC]; 2];
        for p in 0..2 {
            for i in 0..ACC {
                pst[p][i] = ((i as i32 * 2731 + p as i32 * 977) % 1200 - 300) as i16;
                let t = ((i as i32 * 1543 + p as i32 * 389) % 3600) - 1800;
                threat[p][i] = t as i16;
            }
            // plant exact negative half-cases and extremes
            threat[p][0] = -32;
            threat[p][1] = 32;
            threat[p][2] = -96;
            threat[p][3] = -1760;
            threat[p][4] = 1760;
            threat[p][5] = 0;
        }
        for stm in 0..2 {
            let mut want = [0u8; ACC];
            let mut got = [0u8; ACC];
            scalar::activate(&mut want, &pst, &threat, stm);
            activate(&mut got, &pst, &threat, stm);
            assert_eq!(want, got, "stm {stm}");
        }
    }

    #[test]
    fn dispatched_skip_dot_i8_matches_scalar_exactly() {
        // i32 accumulation: every backend must agree with the scalar reference
        // EXACTLY, whatever order it reduces in. Uses the real extremes — ft at
        // its 127 ceiling and weights at both i8 rails — so the avx2 backend's
        // `maddubs` no-saturation argument (2*127*127 < i16::MAX) is exercised.
        let mut ft = vec![0u8; 768];
        let mut lane = vec![0i8; 768];
        for (i, v) in ft.iter_mut().enumerate() {
            *v = ((i as u32 * 97 + 13) % 128) as u8;
        }
        for (i, v) in lane.iter_mut().enumerate() {
            *v = (((i as i32 * 61 + 7) % 255) - 127) as i8;
        }
        ft[0] = 127;
        ft[1] = 127;
        lane[0] = 127;
        lane[1] = 127;
        ft[2] = 127;
        ft[3] = 127;
        lane[2] = -128;
        lane[3] = -128;
        let want = scalar::skip_dot_i8(&ft, &lane);
        let got = skip_dot_i8(&ft, &lane);
        assert_eq!(want, got, "{want} vs {got}");
    }

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

        let mut reference = [0i16; ACC];
        scalar::i8_deltas_into(&mut reference, &seed, &i8_weights, &adds, &subs);
        let mut dispatched = [0i16; ACC];
        apply_i8_deltas_into(&mut dispatched, &seed, &i8_weights, &adds, &subs);
        assert_eq!(dispatched, reference);
    }

    #[test]
    fn dispatched_i16_deltas_match_scalar_on_long_feature_lists() {
        // The wrapping-extremes test above runs 5 adds / 4 subs, which is the
        // incremental-update shape. The Finny fold is the other caller and it
        // arrives with feature lists an order of magnitude longer, where the
        // per-lane i16 wrapping has many more chances to diverge from the
        // scalar reference — and where the `_into_` kernels chunk their
        // hoisted offsets at 96, so any future hoist here (perf study 3's
        // ranked item 10, prototyped on perf/i16-deltas-hoist) must stay
        // bit-exact across that boundary. Sizes below straddle it.
        const ROWS: usize = 11;
        let mut weights = vec![0i16; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u16).wrapping_mul(40503) as i16;
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds: Vec<usize> = (0..240).map(|i| (i * 7) % ROWS).collect();
        let subs: Vec<usize> = (0..200).map(|i| (i * 5 + 3) % ROWS).collect();

        for (a, s) in [
            (&adds[..], &subs[..]),
            (&adds[..96], &subs[..96]),
            (&adds[..97], &subs[..1]),
            (&adds[..0], &subs[..200]),
        ] {
            let mut reference = seed;
            scalar::i16_deltas(&mut reference, &weights, a, s);
            let mut dispatched = seed;
            apply_i16_deltas(&mut dispatched, &weights, a, s);
            assert_eq!(dispatched, reference, "adds={} subs={}", a.len(), s.len());
        }
    }

    /// The two-destination Finny kernel (perf study 9 item 4) must equal
    /// "fold in place, then copy" on BOTH of its homes, on every backend the
    /// CPU offers. The point of the kernel is that the second destination is
    /// written from the fold's own registers, so a backend that wired the
    /// stores up wrongly would silently desync the search stack from the cache
    /// — and that desync survives no test the engine had, because the two are
    /// only compared at the NEXT refresh with the same king key.
    #[test]
    fn dual_kernel_writes_both_homes_identically_on_every_backend() {
        const ROWS: usize = 11;
        let mut weights = vec![0i16; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u16).wrapping_mul(40503) as i16;
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds: Vec<usize> = (0..240).map(|i| (i * 7) % ROWS).collect();
        let subs: Vec<usize> = (0..200).map(|i| (i * 5 + 3) % ROWS).collect();

        #[allow(clippy::type_complexity)]
        let mut backends: Vec<(&str, ApplyI16Fn, ApplyI16DualFn)> =
            vec![("scalar", scalar::i16_deltas, scalar::i16_deltas_dual)];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                backends.push(("avx2", avx2::i16_deltas, avx2::i16_deltas_dual));
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                backends.push(("avx512", avx512::i16_deltas, avx512::i16_deltas_dual));
            }
        }

        for (a, s) in [
            (&adds[..], &subs[..]),
            (&adds[..96], &subs[..96]),
            (&adds[..0], &subs[..0]),
        ] {
            let mut reference = seed;
            scalar::i16_deltas(&mut reference, &weights, a, s);
            for (name, single, dual) in &backends {
                // The single-destination kernel is the same arithmetic, so it
                // pins the dual against BOTH scalar and its own backend.
                let mut own = seed;
                single(&mut own, &weights, a, s);
                assert_eq!(own, reference, "{name} single, adds={}", a.len());

                let mut acc = seed;
                let mut out = [0i16; ACC];
                dual(&mut acc, &mut out, &weights, a, s);
                assert_eq!(acc, reference, "{name} dual in-place home");
                assert_eq!(out, reference, "{name} dual second home");
            }
        }
    }

    #[test]
    fn dispatched_l1_dot_matches_i32_scalar_at_extremes() {
        let mut ft = [0u8; ACC];
        for (i, value) in ft.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(113) & 127;
        }
        let mut weights = vec![0i8; (ACC / 4) * (L2 * 4)];
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
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            backends.push(("avx2", &avx2::KERNELS));
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            backends.push(("avx512", &avx512::KERNELS));
            if std::arch::is_x86_feature_detected!("avx512vbmi2")
                && std::arch::is_x86_feature_detected!("avx512vl")
            {
                backends.push(("avx512-vbmi2", &avx512::KERNELS_VBMI2));
            }
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
        let mut l1_weights = vec![0i8; (ACC / 4) * (L2 * 4)];
        for (i, value) in l1_weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(197) as i8;
        }
        let nnz: Vec<u16> = (0..ACC / 4)
            .filter(|group| group % 3 != 0)
            .map(|group| group as u16)
            .collect();
        let pre_seed: [i32; L2] = std::array::from_fn(|i| i as i32 * 123_457 - 700_000);

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
                    as i16
                    % 8192; // option-D guaranteed domain
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

            let mut reference = [0i16; ACC];
            scalar::i16_deltas_into(&mut reference, &seed, &i16_weights, &adds, &subs);
            let mut got = [0i16; ACC];
            (kernels.i16_deltas_into)(&mut got, &seed, &i16_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i16_deltas_into");

            let mut reference = [0i16; ACC];
            scalar::i8_deltas_into(&mut reference, &seed, &i8_weights, &adds, &subs);
            let mut got = [0i16; ACC];
            (kernels.i8_deltas_into)(&mut got, &seed, &i8_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i8_deltas_into");

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

            // f32 head-tail kernels: finite fixtures spanning both clamp
            // edges and interior values; exact f32 equality is the contract.
            let pre: [i32; L2] = std::array::from_fn(|j| j as i32 * 87_643 - 4_000_000);
            let fbias: [f32; L2] = std::array::from_fn(|j| (j as f32 * 0.037).sin() * 0.8);
            let dequant = 1.7e-6f32;
            let mut reference = [0.0f32; L2];
            scalar::l1_dequant(&mut reference, &pre, &fbias, dequant);
            let mut got = [0.0f32; L2];
            (kernels.l1_dequant)(&mut got, &pre, &fbias, dequant);
            assert_eq!(got, reference, "{name} l1_dequant");

            let l1_act: [f32; L2] =
                std::array::from_fn(|i| ((i as f32 * 0.071).cos() * 0.6).max(0.0));
            let mut l2w = vec![0.0f32; L2 * L3];
            for (i, value) in l2w.iter_mut().enumerate() {
                *value = ((i as f32 * 0.013).sin() - 0.1) * 0.25;
            }
            let l2w: &[f32; L2 * L3] = l2w.as_slice().try_into().unwrap();
            let l3bias: [f32; L3] = std::array::from_fn(|j| (j as f32 * 0.11).cos() * 0.5);
            let mut reference = [0.0f32; L3];
            scalar::l2_tail(&mut reference, &l1_act, l2w, &l3bias);
            let mut got = [0.0f32; L3];
            (kernels.l2_tail)(&mut got, &l1_act, l2w, &l3bias);
            assert_eq!(got, reference, "{name} l2_tail");

            let prow: [i8; L2] = std::array::from_fn(|i| ((i as u8).wrapping_mul(89)) as i8);
            let ph: [f32; L2] = std::array::from_fn(|i| ((i as f32 * 0.043).sin()) * 0.9);
            assert_eq!(
                (kernels.policy_dot)(&prow, &ph),
                scalar::policy_dot(&prow, &ph),
                "{name} policy_dot"
            );
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
        // CONTRACT NARROWED (study-2 option D): the i16 activate pipeline is
        // bit-identical to the scalar reference for |threat lane| <= 8191 —
        // the domain the debug assert in the kernel enforces (measured real
        // nets: max |t| = 431, a 19x margin). Beyond it the saturating 4t
        // form diverges BY DESIGN; the old full-i16-domain sweep therefore
        // now samples the guaranteed domain. pst keeps the full domain
        // (saturating add is exact post-clamp for ALL pst).
        let mut pst = [[0i16; ACC]; 2];
        let mut threat = [[0i16; ACC]; 2];
        for perspective in 0..2 {
            for lane in 0..ACC {
                pst[perspective][lane] = (lane as u16)
                    .wrapping_mul(40503)
                    .wrapping_add(perspective as u16 * 173)
                    as i16;
                let raw = (lane as u16)
                    .wrapping_mul(49711)
                    .wrapping_add(perspective as u16 * 251) as i16;
                threat[perspective][lane] = raw % 8192;
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
