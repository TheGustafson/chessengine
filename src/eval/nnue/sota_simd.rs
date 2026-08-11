//! Fused accumulator kernels for the 768-wide SOTA feature transformers.
//!
//! Keeping the accumulator in registers while applying all feature deltas is
//! substantially cheaper than reading and writing all 768 lanes once per
//! changed feature. The scalar backend is the bit-exact reference; AVX2 is
//! selected once at startup and preserves wrapping i16 arithmetic.

use super::sota_features::{ACC, RowOffset};
use super::sota_head::{FT_QUANT, FT_SHIFT, L2, L3, MAX_THREAT_SHIFT};
use std::sync::LazyLock;

/// The L1 sparse dot writes `L2` int32 accumulators in 16-wide blocks (one
/// avx512 register / two avx256 registers each), so `L2` must be a multiple
/// of 16; widening `L2` scales the block loop automatically.
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

// BOTH feature transformers fold i8 weight rows into an i16 accumulator (the
// HalfKA table is quantised to i8 with a per-column scale the merge applies),
// so there is one fold family shared by the threat and HalfKA channels.
//
// In-place fold: the Finny cache's cold from-scratch build.
type ApplyI8Fn = fn(&mut [i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
// Fused copy+apply: read `src`, apply deltas, write `dst` in ONE pass — the
// lazy fold's child = parent ± deltas without a separate memmove. `dst`/`src`
// may not alias.
type ApplyI8IntoFn = fn(&mut [i16; ACC], &[i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
// Two-destination in-place apply: `acc ±= deltas`, and the result is ALSO
// written to `out`. The Finny warm refresh needs the same bytes in the cache
// entry and the search stack slot; both are written from the registers the
// fold already holds. `acc` and `out` may not alias.
type ApplyI8DualFn = fn(&mut [i16; ACC], &mut [i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
type L1DotFn = fn(&mut [i32; L2], &[u8; ACC], &[u16], &[i8]);
// The FT merge + pairwise activation. BOTH channels carry a net-dependent
// rescale into it, and they are independent of one another:
//
//   `scale` — the HalfKA accumulator's per-lane dequantisation multiplier.
//     The i8 rows are stored at `FT_QUANT / scale[j]`, so the merge puts lane
//     `j` back into the FT_QUANT frame before it clamps. A WRAPPING i16
//     multiply in every backend — pinning the width is what keeps the three
//     backends bit-identical.
//   `threat_shift` — the NET's threat frame shift
//     (`sota_head::threat_shift_for`), not a compile-time constant: a QAT net
//     exports its threat table at scale 256 (shift 0) where a legacy CEN2 net
//     needs 2. Each backend monomorphises over its three admissible values.
type ActivateFn = fn(&mut [u8; ACC], &[[i16; ACC]; 2], &[[i16; ACC]; 2], &[u8; ACC], usize, u32);
type FindNnzFn = fn(&[u8; ACC], &mut [u16; ACC / 4]) -> usize;
// The explicit f32 head-tail kernels. Per-lane op sequences match the scalar
// reference exactly (same fused `mul_add`, same clamp order, same
// i-sequential accumulation per j lane), so every backend is bit-identical
// on finite inputs.
type L1DequantFn = fn(&mut [f32; L2], &[i32; L2], &[f32; L2], f32);
type L2TailFn = fn(&mut [f32; L3], &[f32; L2], &[f32; L2 * L3], &[f32; L3]);
// The policy selected-row dot: i8 row × f32 bottleneck. Reduction shape is
// DEFINED as 16 partial lanes (lane k sums elements k, k+16, …) + a halving
// tree, so every backend is bit-identical to the scalar reference.
type PolicyDotFn = fn(&[i8; L2], &[f32; L2]) -> f32;
// The head's final f32 L3 -> scalar dot. Same fixed reduction SHAPE as
// `policy_dot`, bit-identical to the scalar reference (the loop that used to
// live in `SotaHead::propagate_l3`).
type L3DotFn = fn(&[f32; L3], &[f32; L3]) -> f32;
// The SKP1 skip-lane dot: u8 activations × the i8-quantised bucket-major
// lane. i32 accumulation is exact, so no reduction shape needs pinning.
type SkipDotFn = fn(&[u8], &[i8]) -> i32;

// ---------------------------------------------------------------------------
// The NARROW (6-bit) FullThreats weight row.
// ---------------------------------------------------------------------------

/// Signed bits a "narrow" FullThreats weight row is stored in.
///
/// Design constraints, all measured before pinning:
/// * **Per ROW, not per fold** — most narrow rows do not live in an all-narrow
///   fold, so the narrow/wide dispatch is per row, hoisted out of the tile
///   loop.
/// * **Threats only** — HalfKA has essentially no narrow rows.
/// * 6 bits × 1280 lanes = 960 B = exactly 15 cache lines (vs 20 wide), with
///   the row start still 64-byte aligned and no padding; 7 bits would need
///   padding, which is why there is no second plane.
pub(crate) const NARROW_BITS: usize = 6;
/// Stored bytes of a narrow row. The row keeps its `ACC`-byte SLOT — the
/// layout, the `feature * ACC` offset arithmetic and every producer are
/// untouched — and simply stops reading the tail; unread cache lines are never
/// fetched, so "not read" costs the same as "not present".
pub(crate) const NARROW_BYTES: usize = ACC * NARROW_BITS / 8;

#[cfg(feature = "narrow65")]
/// The magnitude a row's peak must not exceed to be stored narrow: 6 signed
/// bits hold `-32..=31`.
pub(crate) const NARROW_MAX: i16 = 31;

#[cfg(feature = "narrow65")]
/// Whether the kernel that will actually run can decode a narrow row.
///
/// The packed form is read with `vpermb` + `vpmultishiftqb`, which are
/// **AVX512_VBMI** — not implied by the VNNI the compile-time direct path keys
/// on (Cascade Lake has VNNI and no VBMI). Decided once, at LOAD time, by the
/// same predicate the dispatcher uses: if false the loader leaves the table in
/// its 8-bit form, the map stays empty, and every row takes the existing path.
pub(crate) fn narrow_supported() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    {
        // The compile-time direct path calls `avx512::*` with no runtime check,
        // so VBMI must be statically present too.
        cfg!(target_feature = "avx512vbmi")
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vbmi")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

/// The FullThreats **row map**, which decouples "which rows are narrow" from
/// "where rows live" and is what makes the compacted layout possible.
///
/// `map[f]` is the BYTE offset of feature `f`'s row in the threat table, with
/// **bit 0 set when the row is stored narrow**. Both strides are even and
/// every plane base is a multiple of 64, so bit 0 is structurally free.
/// Empty ⇒ the table is flat i8 at `f·ACC` and every path takes the wide
/// route.
///
/// Called once per row per FOLD (hoisted out of the tile walk), not once per
/// row per tile.
#[inline(always)]
pub(crate) fn row_entry(map: &[u32], off: usize) -> (usize, bool) {
    let m = map.get(off / ACC).copied().unwrap_or(off as u32);
    ((m & !1) as usize, m & 1 != 0)
}

/// Decode lane `j` of a narrow row whose packed bytes start at `row`.
///
/// The scalar reference for [`pack_narrow_row`], used by the from-scratch
/// rebuild path and by the tests that prove the SIMD decode agrees with it.
#[inline]
pub(crate) fn unpack_narrow_lane(row: &[i8], j: usize) -> i16 {
    let g = j / 8;
    let mut q = 0u64;
    for (i, &b) in row[g * 6..g * 6 + 6].iter().enumerate() {
        q |= u64::from(b as u8) << (8 * i);
    }
    let v = (q >> (6 * (j % 8))) & 0x3F;
    // Sign-extend from 6 bits: `(v ^ 32) - 32`.
    (v as i16 ^ 32) - 32
}

#[cfg(feature = "narrow65")]
/// Pack one row's `ACC` i8 weights into [`NARROW_BYTES`] bytes, little-endian,
/// 8 lanes per 6 bytes.
///
/// The grouping is what makes the decode one shuffle: eight 6-bit fields are 48
/// bits, so **no field ever straddles a 64-bit boundary**, and
/// `vpmultishiftqb` — which extracts eight unaligned byte-wide fields from each
/// qword — can lift all eight in one instruction once `vpermb` has put each
/// group's 6 bytes into its own qword.
///
/// Values are stored as the raw two's-complement 6-bit pattern; the kernel
/// recovers `w + 32` with a single `vpternlogd` (`(m & 0x3F) ^ 0x20`) and pays
/// the `-32` bias once per fold instead of once per lane.
pub(crate) fn pack_narrow_row(row: &[i8], out: &mut [u8]) {
    debug_assert_eq!(row.len(), ACC);
    debug_assert_eq!(out.len(), NARROW_BYTES);
    out.fill(0);
    for (g, group) in row.chunks_exact(8).enumerate() {
        let mut q: u64 = 0;
        for (k, &w) in group.iter().enumerate() {
            q |= u64::from((w as u8) & 0x3F) << (6 * k);
        }
        out[g * 6..g * 6 + 6].copy_from_slice(&q.to_le_bytes()[..6]);
    }
}

struct Kernels {
    i8_deltas: ApplyI8Fn,
    i8_deltas_into: ApplyI8IntoFn,
    i8_deltas_dual: ApplyI8DualFn,
    l1_dot: L1DotFn,
    activate: ActivateFn,
    find_nnz: FindNnzFn,
    l1_dequant: L1DequantFn,
    l2_tail: L2TailFn,
    l3_dot: L3DotFn,
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
            // vbmi2+vl unlock the vpcompressw find_nnz; both are present on
            // every Zen 4 / Ice Lake+ part, but the plain table keeps
            // pre-VBMI2 AVX-512 CPUs as belt-and-braces.
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

/// Compile-time backend selection: when the build target statically carries
/// the same AVX-512 features the runtime dispatcher would detect
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

/// Traffic-pinning probes (`probe-accpin` / `probe-wpin` and friends).
///
/// All keep the number and the WIDTH of every load in the kernels exactly as
/// they are and change only WHERE those loads point, so `L1-dcache-loads` is
/// invariant and the drop in misses isolates one stream's share. They produce
/// wrong evaluations by construction — measurement instruments, never
/// shipped; `bench … nodes` fixes the node budget so arms stay comparable.
#[cfg(any(
    feature = "probe-accpin",
    feature = "probe-wpin",
    feature = "probe-wpin-hka",
    feature = "probe-wpin-thr",
    feature = "probe-actpin",
    feature = "probe-addcopy"
))]
pub(crate) mod probe {
    #[cfg(any(
        feature = "probe-accpin",
        feature = "probe-actpin",
        feature = "probe-addcopy"
    ))]
    use super::ACC;

    /// One accumulator-shaped buffer every pinned read is redirected to.
    /// `Box` so it is heap-resident like the real stack, `LazyLock` so it is
    /// built once; the point is that the whole working set is L1-resident.
    /// TWO distinct buffers, and every handout goes through `black_box` —
    /// with one buffer or a constant pointer LLVM CSEs loads away and the
    /// probe stops being load-count-neutral. `black_box` makes the pointer
    /// opaque, so the same loads are issued against the same few KB of L1.
    #[cfg(feature = "probe-accpin")]
    pub(crate) static PIN_ACC: std::sync::LazyLock<Box<[[i16; ACC]; 2]>> =
        std::sync::LazyLock::new(|| Box::new([[0i16; ACC]; 2]));
    #[cfg(feature = "probe-accpin")]
    pub(crate) static PIN_ACC2: std::sync::LazyLock<Box<[[i16; ACC]; 2]>> =
        std::sync::LazyLock::new(|| Box::new([[0i16; ACC]; 2]));

    #[cfg(feature = "probe-accpin")]
    #[inline]
    pub(crate) fn pin_one() -> &'static [i16; ACC] {
        std::hint::black_box(&PIN_ACC[0])
    }

    /// The WRITE half (`probe-dstpin`). `probe-accpin` alone redirects only
    /// the fold's `src`, so the accumulator stack keeps its write footprint
    /// and keeps evicting weight rows; this buffer takes the stores too.
    ///
    /// A leaked `Box` rather than a `static mut`: one allocation, `'static`
    /// by construction. Every fold writing the same buffer is the traffic
    /// collapse being measured — wrong evaluations by design.
    #[cfg(feature = "probe-dstpin")]
    // D14(a)-style sanctioned relaxation for a measurement-only feature.
    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn pin_dst() -> &'static mut [i16; ACC] {
        use std::sync::OnceLock;
        static SLOT: OnceLock<usize> = OnceLock::new();
        let addr = *SLOT.get_or_init(|| {
            let b: Box<[i16; ACC]> = Box::new([0i16; ACC]);
            Box::into_raw(b) as usize
        });
        // SAFETY: the pointer comes from `Box::into_raw` on a leaked
        // allocation, so it is non-null, aligned, and valid for the whole
        // program. The engine is single-threaded under `bench` (the only
        // thing this feature runs under); the repeated `&mut` handout is the
        // traffic collapse being measured, evaluations known-wrong by design.
        unsafe { &mut *(addr as *mut [i16; ACC]) }
    }

    /// Feature indices, all zero, for the `wpin` arm. The kernels read
    /// `adds.len()`/`subs.len()` for their shape dispatch, so the SLICE
    /// LENGTHS must be preserved exactly — only the values change.
    #[cfg(any(
        feature = "probe-wpin",
        feature = "probe-wpin-hka",
        feature = "probe-wpin-thr"
    ))]
    pub(crate) static ZERO_IDX: [super::RowOffset; 64] = [super::RowOffset(0); 64];

    /// The SUBTRACTIVE pinning arm: `probe-actpin` pins ONLY `activate`'s
    /// four accumulator arrays (`probe-accpin` pins the fold `src` stream
    /// too, so the two are not comparable at equal magnitude).
    ///
    /// **`P39_PINFILL=live` is the point of this module.** A pin buffer full
    /// of ZEROES makes `activate` emit an all-zero `ft`, so `find_nnz`
    /// returns almost nothing and `l1_dot` — with its own weight rows and its
    /// own misses — does almost no work: the probe silently deletes a whole
    /// layer of the network. Under `live` the FIRST `activate` copies the
    /// real accumulators into the pin and every later one reads that
    /// snapshot: realistic magnitudes and group sparsity, so the head does
    /// its normal work. Same binary, same code layout, one runtime branch.
    #[cfg(feature = "probe-actpin")]
    pub(crate) mod livepin {
        use super::ACC;
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

        type Pair = [[i16; ACC]; 2];

        struct Pins {
            a: usize,
            b: usize,
            live: bool,
        }
        static PINS: OnceLock<Pins> = OnceLock::new();
        static FILLED: AtomicBool = AtomicBool::new(false);

        fn pins() -> &'static Pins {
            PINS.get_or_init(|| {
                let live = std::env::var("P39_PINFILL").as_deref() == Ok("live");
                eprintln!(
                    "[p39] actpin: pinfill={}",
                    if live { "live" } else { "zero" }
                );
                let leak = || Box::into_raw(Box::new([[0i16; ACC]; 2])) as usize;
                Pins {
                    a: leak(),
                    b: leak(),
                    live,
                }
            })
        }

        /// Hand out the pinned pair, filling it from the real accumulators on
        /// the first call when `P39_PINFILL=live`.
        ///
        /// Relaxed load/store on `FILLED` rather than a `swap`: an atomic RMW
        /// on the eval path would be pure probe artefact, and `bench` is
        /// single-threaded.
        #[inline]
        #[allow(unsafe_code)]
        pub(crate) fn take(pst: &Pair, threat: &Pair) -> (&'static Pair, &'static Pair) {
            let p = pins();
            // SAFETY: both pointers come from `Box::into_raw` on leaked `Pair`
            // allocations — non-null, aligned, valid for the whole program.
            // The one-shot fill happens before either reference is handed out
            // and `bench` is single-threaded; the contents being shared
            // across plies is the traffic collapse this instrument exists to
            // create (wrong evaluations by design).
            unsafe {
                if p.live && !FILLED.load(Relaxed) {
                    FILLED.store(true, Relaxed);
                    *(p.a as *mut Pair) = *pst;
                    *(p.b as *mut Pair) = *threat;
                }
                (
                    std::hint::black_box(&*(p.a as *const Pair)),
                    std::hint::black_box(&*(p.b as *const Pair)),
                )
            }
        }
    }

    /// The ADDITIVE traffic probe: redundant accumulator-pair copies per
    /// evaluated node, cost read off the stall counters. Both knobs are
    /// **runtime** on purpose — ONE binary serves the whole panel, so no
    /// comparison inside it can be confounded by code layout:
    ///
    /// * `P39_DOSE` — copies per evaluated node (default 1); a dose series
    ///   tests whether stall cycles are LINEAR in added traffic.
    /// * `P39_RING` — distinct accumulator-pair slots the copies rotate
    ///   through, rounded up to a power of two (default 16). RING = 1 keeps
    ///   the copy L1-resident: byte-identical code, identical instruction and
    ///   load counts, essentially no cold lines — so `cold − hot` is the
    ///   probe's LOCALITY cost with its ISSUE cost differenced away.
    #[cfg(feature = "probe-addcopy")]
    pub(crate) mod addcopy {
        use super::ACC;
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        /// i16 lanes in one accumulator PAIR (2 × 1280 × 2 B = 5120 B = 80
        /// 64-byte lines; one copy moves 80 read + 80 write lines).
        const PAIR: usize = 2 * ACC;

        /// `src`/`dst` are leaked `Box<[i16]>` allocations kept as integers
        /// (a `&'static mut` cannot be handed out of a `&'static Rings`);
        /// mirrors the `pin_dst` idiom above.
        struct Rings {
            src: usize,
            dst: usize,
            mask: usize,
            dose: usize,
            /// `P39_SRC=live` — read the REAL accumulator pair `activate` is
            /// about to consume instead of the cold ring, so the copy's read
            /// half touches memory the engine was about to touch anyway.
            live_src: bool,
            /// `P39_SITE=fold` — issue the copies from the FullThreats fold
            /// instead of from `activate`, to test whether stall cost is a
            /// property of the traffic or of where in the node it is issued.
            fold_site: bool,
        }

        static RINGS: OnceLock<Rings> = OnceLock::new();
        /// Plain relaxed load/store, never a `fetch_add`: an atomic RMW is a
        /// locked op and would be a large fraction of what this probe
        /// measures; `bench` is single-threaded.
        static CURSOR: AtomicUsize = AtomicUsize::new(0);

        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }

        fn rings() -> &'static Rings {
            RINGS.get_or_init(|| {
                let slots = env_usize("P39_RING", 16).max(1).next_power_of_two();
                let dose = env_usize("P39_DOSE", 1);
                let live_src = std::env::var("P39_SRC").as_deref() == Ok("live");
                let fold_site = std::env::var("P39_SITE").as_deref() == Ok("fold");
                eprintln!(
                    "[p39] addcopy: ring={slots} slots, footprint={} KB, dose={dose}, \
                     src={}, site={}",
                    2 * slots * PAIR * 2 / 1024,
                    if live_src { "live" } else { "ring" },
                    if fold_site { "fold" } else { "activate" }
                );
                let leak =
                    || Box::leak(vec![0i16; slots * PAIR].into_boxed_slice()).as_mut_ptr() as usize;
                Rings {
                    src: leak(),
                    dst: leak(),
                    mask: slots - 1,
                    dose,
                    live_src,
                    fold_site,
                }
            })
        }

        /// One dose = one accumulator-pair-sized vector copy through the ring,
        /// issued from `activate` (the default site).
        ///
        /// NOTE: the copy is `wrapping_add` of an opaque zero, NOT
        /// `copy_from_slice` — the latter lowers to a libc `memcpy`
        /// (`rep movsb`), a different microarchitectural animal from the
        /// `vmovdqu64` streaming this probe prices; the add form
        /// autovectorises to the fold's own load/`vpaddw`/store shape. Check
        /// the disassembly before trusting this; do not "simplify" it back.
        #[inline]
        pub(crate) fn tick(live: &[[i16; ACC]; 2]) {
            let r = rings();
            if !r.fold_site {
                copies(r, Some(live));
            }
        }

        /// The FullThreats-fold call site (`P39_SITE=fold`).
        #[inline]
        pub(crate) fn tick_fold() {
            let r = rings();
            if r.fold_site {
                copies(r, None);
            }
        }

        #[inline]
        #[allow(unsafe_code)]
        fn copies(r: &Rings, live: Option<&[[i16; ACC]; 2]>) {
            let bias = std::hint::black_box(0i16);
            for _ in 0..r.dose {
                let i = CURSOR.load(Relaxed);
                CURSOR.store(i.wrapping_add(1), Relaxed);
                let off = (i & r.mask) * PAIR;
                // SAFETY: both rings are leaked `Box<[i16]>` of exactly
                // `(mask + 1) * PAIR` lanes, and `off + PAIR` is therefore in
                // bounds for every `i`. The allocations are `'static`, never
                // freed, and `bench` — the only thing this measurement-only
                // feature is ever run under — is single-threaded.
                let (s, d) = unsafe {
                    let ring = std::slice::from_raw_parts((r.src as *const i16).add(off), PAIR);
                    let s = match live {
                        // Same PAIR lanes, from the accumulator `activate` is
                        // about to read rather than from cold ring memory.
                        Some(l) if r.live_src => {
                            std::slice::from_raw_parts(l.as_ptr().cast::<i16>(), PAIR)
                        }
                        _ => ring,
                    };
                    (
                        s,
                        std::slice::from_raw_parts_mut((r.dst as *mut i16).add(off), PAIR),
                    )
                };
                for (dl, &sl) in d.iter_mut().zip(s.iter()) {
                    *dl = sl.wrapping_add(bias);
                }
                std::hint::black_box(d);
            }
        }
    }
}

/// The ceiling probe, split by channel: `probe-wpin` pins all four fold
/// sites' row indices to row 0; `probe-wpin-hka` pins only the three HalfKA
/// sites; `probe-wpin-thr` only `apply_threat_deltas_into`. The split probes
/// are strict subsets of `probe-wpin`, so `Δhka + Δthr ≈ Δboth` is a built-in
/// consistency check on the pair. Measurement only — they compute wrong
/// evaluations by construction.
#[cfg(any(feature = "probe-wpin", feature = "probe-wpin-hka"))]
macro_rules! pin_rows_hka {
    ($adds:ident, $subs:ident) => {
        let $adds: &[RowOffset] = &probe::ZERO_IDX[..$adds.len().min(64)];
        let $subs: &[RowOffset] = &probe::ZERO_IDX[..$subs.len().min(64)];
    };
}
#[cfg(not(any(feature = "probe-wpin", feature = "probe-wpin-hka")))]
macro_rules! pin_rows_hka {
    ($adds:ident, $subs:ident) => {};
}

#[cfg(any(feature = "probe-wpin", feature = "probe-wpin-thr"))]
macro_rules! pin_rows_thr {
    ($adds:ident, $subs:ident) => {
        let $adds: &[RowOffset] = &probe::ZERO_IDX[..$adds.len().min(64)];
        let $subs: &[RowOffset] = &probe::ZERO_IDX[..$subs.len().min(64)];
    };
}
#[cfg(not(any(feature = "probe-wpin", feature = "probe-wpin-thr")))]
macro_rules! pin_rows_thr {
    ($adds:ident, $subs:ident) => {};
}

#[cfg(feature = "probe-accpin")]
macro_rules! pin_src {
    ($src:ident) => {
        // The shadow makes the real parameter unused, which is the point.
        #[allow(unused_variables)]
        let $src: &[i16; ACC] = probe::pin_one();
    };
}
#[cfg(not(feature = "probe-accpin"))]
macro_rules! pin_src {
    ($src:ident) => {};
}

#[cfg(feature = "probe-dstpin")]
macro_rules! pin_dst {
    ($dst:ident) => {
        #[allow(unused_variables)]
        let $dst: &mut [i16; ACC] = probe::pin_dst();
    };
}
#[cfg(not(feature = "probe-dstpin"))]
macro_rules! pin_dst {
    ($dst:ident) => {};
}

/// In-place i8-weight fold — the HalfKA Finny cache's cold from-scratch build.
#[inline]
pub(crate) fn apply_halfka_deltas(
    acc: &mut [i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    pin_rows_hka!(adds, subs);
    kernel_dispatch!(i8_deltas(acc, weights, adds, subs))
}

/// Fused copy+apply for the HalfKA (i8-weight) fold. `dst` and `src` may not
/// alias.
#[inline]
// `probe-accpin` shadows the accumulator parameters with its pinned buffer;
// the originals then go unread. No effect on builds without that feature.
#[cfg_attr(feature = "probe-accpin", allow(unused_variables))]
pub(crate) fn apply_halfka_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    pin_rows_hka!(adds, subs);
    pin_src!(src);
    pin_dst!(dst);
    kernel_dispatch!(i8_deltas_into(dst, src, weights, adds, subs))
}

/// Two-destination in-place apply: `acc ±= deltas`, and the same result is
/// written to `out` as well. See [`ApplyI8DualFn`]. `acc` and `out` must not
/// alias.
#[inline]
#[cfg_attr(feature = "probe-dstpin", allow(unused_variables))]
pub(crate) fn apply_halfka_deltas_dual(
    acc: &mut [i16; ACC],
    out: &mut [i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    // The `nt43` arm stores `acc` non-temporally, which the ISA only defines
    // for a 64-byte-aligned address. The alignment is structural
    // (`FinnyEntry` is `align(64)`, `acc` at offset 128), so this asserts the
    // invariant rather than checking a value that could legitimately vary.
    debug_assert_eq!(
        std::ptr::from_ref(acc) as usize % 64,
        0,
        "the Finny cache entry's accumulator must be 64-byte aligned"
    );
    pin_rows_hka!(adds, subs);
    pin_dst!(out);
    kernel_dispatch!(i8_deltas_dual(acc, out, weights, adds, subs))
}

/// Fused copy+apply for the FullThreats (i8-weight) fold.
#[inline]
// `probe-accpin` shadows the accumulator parameters with its pinned buffer;
// the originals then go unread. No effect on builds without that feature.
#[cfg_attr(feature = "probe-accpin", allow(unused_variables))]
pub(crate) fn apply_threat_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
    row_map: &[u32],
) {
    // `probe-addcopy`'s fold-site arm (`P39_SITE=fold`); inert otherwise.
    #[cfg(feature = "probe-addcopy")]
    probe::addcopy::tick_fold();
    pin_rows_thr!(adds, subs);
    pin_src!(src);
    pin_dst!(dst);
    // One branch per FOLD, not per row. `row_map` is empty unless the loader
    // packed a narrow plane — which it only does behind `narrow_supported()`
    // and the `narrow65` feature — so every other build reaches exactly the
    // call it reached before.
    #[cfg(all(target_arch = "x86_64", feature = "narrow65"))]
    if !row_map.is_empty() {
        return avx512::i8_deltas_into_narrow(dst, src, weights, adds, subs, row_map);
    }
    let _ = row_map;
    kernel_dispatch!(i8_deltas_into(dst, src, weights, adds, subs))
}

/// Sparse `[u8; ACC] × [i8; L2]` dot product. `weights` is laid out as
/// `[group][output][4]` (output ∈ 0..L2), and `nnz` contains active four-byte
/// group indices.
#[inline]
pub(crate) fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
    // The SIMD backends read each group's four FT bytes as ONE unaligned
    // 32-bit load (see [`ft_group`]). `find_nnz` is the only producer of
    // `nnz` and only ever emits group indices in `0..ACC/4`; this is the
    // debug-time proof of that invariant.
    debug_assert!(
        nnz.iter().all(|&g| usize::from(g) < ACC / 4),
        "l1_dot: nnz group index out of range (find_nnz contract broken)"
    );
    kernel_dispatch!(l1_dot(pre, ft, nnz, weights))
}

/// The four FT bytes of `group`, as the little-endian `i32` the `dpbusd` /
/// `maddubs` kernels broadcast.
///
/// One unaligned 32-bit read rather than `i32::from_le_bytes([ft[g * 4], …])`
/// **for the codegen, not the arithmetic** — the two are bit-identical on any
/// little-endian target, but indexing made LLVM emit a bounds-check pair it
/// could not hoist out of the short, hot group loop. Removing that pair
/// mattered; do not rewrite as indexed `from_le_bytes` without measuring.
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
// The pin probes shadow the accumulator parameters with their pinned buffer;
// the originals then go unread. No effect on builds without those features.
#[cfg_attr(
    any(feature = "probe-accpin", feature = "probe-actpin"),
    allow(unused_variables)
)]
pub(crate) fn activate(
    out: &mut [u8; ACC],
    pst: &[[i16; ACC]; 2],
    threat: &[[i16; ACC]; 2],
    scale: &[u8; ACC],
    stm: usize,
    threat_shift: u32,
) {
    // `probe-addcopy`'s default call site.
    #[cfg(feature = "probe-addcopy")]
    probe::addcopy::tick(pst);
    // `probe-actpin` pins ONLY the four accumulator arrays read here;
    // `probe-accpin` additionally pins the fold `src` stream.
    #[cfg(feature = "probe-accpin")]
    let (pst, threat) = (
        std::hint::black_box(&*probe::PIN_ACC),
        std::hint::black_box(&*probe::PIN_ACC2),
    );
    #[cfg(all(feature = "probe-actpin", not(feature = "probe-accpin")))]
    let (pst, threat) = probe::livepin::take(pst, threat);
    // The HalfKA rescale is a WRAPPING i16 multiply in every backend (pinning
    // the width keeps the three bit-identical). A wrap turns a saturating
    // lane into a near-zero one and the clamp hides it, so this tripwire is
    // what fires if a future net's accumulator range grows; the threat
    // channel's shift bound check is its twin. Skipped under the pin probes,
    // which redirect the accumulators and compute wrong evals by design.
    #[cfg(not(any(feature = "probe-accpin", feature = "probe-actpin")))]
    debug_assert!(
        pst.iter().all(
            |side| side
                .iter()
                .zip(scale.iter())
                .all(
                    |(&p, &s)| i32::from(p) * i32::from(s) <= i32::from(i16::MAX)
                        && i32::from(p) * i32::from(s) >= i32::from(i16::MIN)
                )
        ),
        "HalfKA accumulator x ft_scale overflows i16: the reduced-frame merge \
         would wrap and the clamp would hide it"
    );
    kernel_dispatch!(activate(out, pst, threat, scale, stm, threat_shift))
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
/// exactly the scalar head loop it replaces.
#[inline]
pub(crate) fn l1_dequant(out: &mut [f32; L2], pre: &[i32; L2], bias: &[f32; L2], dequant: f32) {
    kernel_dispatch!(l1_dequant(out, pre, bias, dequant))
}

/// The dense f32 L2→L3 tail: `out = SCReLU(bias + Σ_i weights[i]·l1[i])`,
/// j-lane order preserved and i-sequential per lane — bit-identical to the
/// scalar mul_add loop.
#[inline]
pub(crate) fn l2_tail(
    out: &mut [f32; L3],
    l1: &[f32; L2],
    weights: &[f32; L2 * L3],
    bias: &[f32; L3],
) {
    kernel_dispatch!(l2_tail(out, l1, weights, bias))
}

/// The head's final dense `f32[L3] → f32` dot: `Σ weights[j]·l2[j]`, in the
/// fixed 16-partial-lane + halving-tree shape (see the [`L3DotFn`] note). The
/// bias is added by the caller, as in the scalar reference.
#[inline]
pub(crate) fn l3_dot(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
    kernel_dispatch!(l3_dot(weights, l2))
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

    pub(super) fn i8_deltas(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        for lane in 0..ACC {
            let mut value = acc[lane];
            for &feature in adds {
                value = value.wrapping_add(i16::from(weights[feature.0 + lane]));
            }
            for &feature in subs {
                value = value.wrapping_sub(i16::from(weights[feature.0 + lane]));
            }
            acc[lane] = value;
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn i8_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        for lane in 0..ACC {
            let mut value = acc[lane];
            for &feature in adds {
                value = value.wrapping_add(i16::from(weights[feature.0 + lane]));
            }
            for &feature in subs {
                value = value.wrapping_sub(i16::from(weights[feature.0 + lane]));
            }
            acc[lane] = value;
            out[lane] = value;
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        for lane in 0..ACC {
            let mut value = src[lane];
            for &feature in adds {
                value = value.wrapping_add(i16::from(weights[feature.0 + lane]));
            }
            for &feature in subs {
                value = value.wrapping_sub(i16::from(weights[feature.0 + lane]));
            }
            dst[lane] = value;
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
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

    /// The scalar reference for the merged FT activation, stated from the
    /// head's own math.
    ///
    /// Per perspective, accumulator lane `j` carries two channels that must
    /// meet in one `FT_QUANT` frame:
    ///
    ///   * HalfKA, rescaled by its per-column i8 quantisation scale. The
    ///     multiply happens in i16 and WRAPS — defined semantics, not a
    ///     shortcut: the vector backends multiply in i16 lanes (`vpmullw`),
    ///     so a widen-before-multiply here would disagree with them on the
    ///     (never observed) overflow.
    ///   * threats, rescaled by the power-of-two merge shift
    ///     (`FT_QUANT/threat_quant` carried as a shift — 4 for a CEN2 net,
    ///     1 at threat_quant 128, 0 for a QAT net at 256).
    ///
    /// The clamped sum is the lane's pre-activation in `[0, FT_QUANT]`; the
    /// activation multiplies each lane by its partner `ACC/2` above it and
    /// drops the product back into u8 range with the `FT_SHIFT` right shift.
    /// The side to move's pairs fill `out[..ACC/2]`; the opponent's fill
    /// `out[ACC/2..]`.
    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
        stm: usize,
        threat_shift: u32,
    ) {
        const PAIR: usize = ACC / 2;
        let squash = |side: usize, j: usize| -> i32 {
            let halfka = i32::from(pst[side][j].wrapping_mul(i16::from(scale[j])));
            let threats = i32::from(threat[side][j]) << threat_shift;
            (halfka + threats).clamp(0, FT_QUANT)
        };
        let (own_half, opp_half) = out.split_at_mut(PAIR);
        for (side, half) in [(stm, own_half), (stm ^ 1, opp_half)] {
            for (j, cell) in half.iter_mut().enumerate() {
                *cell = ((squash(side, j) * squash(side, j + PAIR)) >> FT_SHIFT) as u8;
            }
        }
    }

    /// The scalar reference for the sparse-group scan, stated from what the
    /// L1 matmul needs: the activated FT vector is consumed as `ACC/4`
    /// four-byte groups (one `vpdpbusd` word each), and a group is worth
    /// visiting exactly when any of its four activations is nonzero. Emit
    /// each live group's index, ascending, into `out`; return the count.
    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn find_nnz(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        let mut live = 0usize;
        for (group, quad) in ft.chunks_exact(4).enumerate() {
            if quad.iter().any(|&b| b != 0) {
                out[live] = group as u16;
                live += 1;
            }
        }
        live
    }

    /// The bit-exact reference for the L1 dequant+SCReLU (the loop moved here
    /// verbatim from `SotaHead::propagate_l1`).
    #[cfg_attr(feature = "split58", inline(never))]
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
    #[cfg_attr(feature = "split58", inline(never))]
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

    /// The bit-exact reference for the head's L3 dot — moved unchanged from
    /// `SotaHead::propagate_l3` so the SIMD backends have something to be
    /// identical to.
    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn l3_dot(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
        const LANES: usize = 16;
        let mut sums = [0.0f32; LANES];
        let mut i = 0;
        while i < L3 {
            for j in 0..LANES {
                sums[j] = weights[i + j].mul_add(l2[i + j], sums[j]);
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
    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        let mut acc = 0i32;
        for (&a, &w) in ft.iter().zip(lane) {
            acc += i32::from(a) * i32::from(w);
        }
        acc
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i8_deltas,
        i8_deltas_into,
        i8_deltas_dual,
        l1_dot,
        activate,
        find_nnz,
        l1_dequant,
        l2_tail,
        l3_dot,
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

    // `SHIFT` is `i32` here and `u32` in the avx512 twin because that is what
    // each intrinsic's own const parameter is (`_mm256_slli_epi32<const IMM8:
    // i32>` vs `_mm512_slli_epi16<const IMM8: u32>`), and converting a const
    // generic in the turbofish needs `generic_const_exprs`.
    #[target_feature(enable = "avx2")]
    unsafe fn activate_inner<const SHIFT: i32>(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
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
                        // The i8 HalfKA rescale happens in i16 lanes, BEFORE
                        // the widen, so this wraps exactly where the avx512
                        // `vpmullw` and the scalar `wrapping_mul` wrap.
                        let raw = _mm_loadu_si128(p.as_ptr().add(offset).cast());
                        // The scale is u8 in memory: a 64-bit byte load plus
                        // a zero-extend, half the bytes at the same op count;
                        // `_mm_cvtepu8_epi16` is exact for the loader's
                        // 1..=255 range.
                        let sv =
                            _mm_cvtepu8_epi16(_mm_loadl_epi64(scale.as_ptr().add(offset).cast()));
                        let pv = _mm256_cvtepi16_epi32(_mm_mullo_epi16(raw, sv));
                        let tv =
                            _mm256_cvtepi16_epi32(_mm_loadu_si128(t.as_ptr().add(offset).cast()));
                        // FT_QUANT/threat_quant carried as a power of two
                        // (see the avx512 twin).
                        let scaled = _mm256_slli_epi32::<SHIFT>(tv);
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

    /// In-place i8 fold — the HalfKA Finny cold
    /// build. Same 16-lane block walk; the row is a 16-byte load widened to 16
    /// i16 lanes, which is the shape the threat fold has always used.
    #[target_feature(enable = "avx2")]
    unsafe fn i8_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_into_inner`; each row is `ACC` in-bounds i8.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm256_loadu_si256(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature.0 + off);
                    value =
                        _mm256_add_epi16(value, _mm256_cvtepi8_epi16(_mm_loadu_si128(row.cast())));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature.0 + off);
                    value =
                        _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(_mm_loadu_si128(row.cast())));
                }
                _mm256_storeu_si256(acc.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    /// Two-destination i8 fold — the HalfKA Finny warm
    /// refresh, writing the cache entry and the stack slot from one register.
    #[target_feature(enable = "avx2")]
    unsafe fn i8_deltas_dual_inner(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_inner`; `out` is a disjoint ACC-wide array.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm256_loadu_si256(acc.as_ptr().add(off).cast());
                for &feature in adds {
                    let row = weights.as_ptr().add(feature.0 + off);
                    value =
                        _mm256_add_epi16(value, _mm256_cvtepi8_epi16(_mm_loadu_si128(row.cast())));
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature.0 + off);
                    value =
                        _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(_mm_loadu_si128(row.cast())));
                }
                _mm256_storeu_si256(acc.as_mut_ptr().add(off).cast(), value);
                _mm256_storeu_si256(out.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn i8_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as i8_deltas_inner; reads `src`, writes the disjoint `dst`.
        // The adds/subs arrive as pre-scaled [`RowOffset`]s, so there is no
        // per-block `feature * ACC` arithmetic left to hoist.
        unsafe {
            for off in (0..ACC).step_by(LANES) {
                let mut value = _mm256_loadu_si256(src.as_ptr().add(off).cast());
                for &base in adds {
                    let bytes = _mm_loadu_si128(weights.as_ptr().add(base.0 + off).cast());
                    value = _mm256_add_epi16(value, _mm256_cvtepi8_epi16(bytes));
                }
                for &base in subs {
                    let bytes = _mm_loadu_si128(weights.as_ptr().add(base.0 + off).cast());
                    value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(bytes));
                }
                _mm256_storeu_si256(dst.as_mut_ptr().add(off).cast(), value);
            }
        }
    }

    fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i8_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    pub(super) fn i8_deltas(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i8_deltas_inner(acc, weights, adds, subs) }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn i8_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { i8_deltas_dual_inner(acc, out, weights, adds, subs) }
    }

    fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe { l1_dot_inner(pre, ft, nnz, weights) }
    }

    fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
        stm: usize,
        threat_shift: u32,
    ) {
        // The `_` arm is `MAX_THREAT_SHIFT` — see the avx512 twin.
        debug_assert!(threat_shift <= MAX_THREAT_SHIFT);
        // SAFETY: installed only after runtime AVX2 detection.
        unsafe {
            match threat_shift {
                0 => activate_inner::<0>(out, pst, threat, scale, stm),
                1 => activate_inner::<1>(out, pst, threat, scale, stm),
                _ => activate_inner::<2>(out, pst, threat, scale, stm),
            }
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
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
            // `L3 / 8` 8-lane accumulators — see the avx512 twin: the fixed
            // four assumed L3 = 32 and overran on a narrower head.
            const BLK: usize = 8;
            const NB: usize = L3 / BLK;
            let mut acc = [_mm256_setzero_ps(); NB];
            for (b, a) in acc.iter_mut().enumerate() {
                *a = _mm256_loadu_ps(bias.as_ptr().add(b * BLK));
            }
            for (i, &input) in l1.iter().enumerate() {
                let li = _mm256_set1_ps(input);
                let row = weights.as_ptr().add(i * L3);
                for (b, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(b * BLK)), li, *a);
                }
            }
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let screlu = |acc| {
                let clipped = _mm256_min_ps(_mm256_max_ps(acc, zero), one);
                _mm256_mul_ps(clipped, clipped)
            };
            for (b, &a) in acc.iter().enumerate() {
                _mm256_storeu_ps(out.as_mut_ptr().add(b * BLK), screlu(a));
            }
        }
    }

    fn l2_tail(out: &mut [f32; L3], l1: &[f32; L2], weights: &[f32; L2 * L3], bias: &[f32; L3]) {
        // SAFETY: installed only after runtime AVX2+FMA detection.
        unsafe { l2_tail_inner(out, l1, weights, bias) }
    }

    /// The last three steps of the reference halving tree, in registers.
    /// Shared by both f32 dots and by both SIMD backends so the four of them
    /// cannot drift from the scalar reference or from each other.
    /// `#[target_feature]` so the caller's feature set propagates and it
    /// inlines.
    #[target_feature(enable = "avx")]
    pub(super) fn halve8(a: __m256) -> f32 {
        let b = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
        let c = _mm_add_ps(b, _mm_movehl_ps(b, b));
        _mm_cvtss_f32(_mm_add_ss(c, _mm_movehdup_ps(c)))
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn l3_dot_inner(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
        // Two 8-lane accumulators = the scalar reference's 16 partial lanes
        // (acc_lo = lanes 0-7, acc_hi = lanes 8-15), then the SAME halving
        // tree, done in vector registers instead of over a spilled array —
        // identical pairs in identical order, so bit-identical, and f32
        // non-associativity is respected rather than assumed away.
        // SAFETY: L3 = 32 is two 8-lane blocks per accumulator, in bounds.
        unsafe {
            let mut acc_lo = _mm256_setzero_ps();
            let mut acc_hi = _mm256_setzero_ps();
            for i in (0..L3).step_by(16) {
                acc_lo = _mm256_fmadd_ps(
                    _mm256_loadu_ps(weights.as_ptr().add(i)),
                    _mm256_loadu_ps(l2.as_ptr().add(i)),
                    acc_lo,
                );
                acc_hi = _mm256_fmadd_ps(
                    _mm256_loadu_ps(weights.as_ptr().add(i + 8)),
                    _mm256_loadu_ps(l2.as_ptr().add(i + 8)),
                    acc_hi,
                );
            }
            // stride 8 of the reference's halving tree; `halve8` finishes it.
            halve8(_mm256_add_ps(acc_lo, acc_hi))
        }
    }

    fn l3_dot(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
        // SAFETY: installed only after runtime AVX2+FMA detection.
        unsafe { l3_dot_inner(weights, l2) }
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

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        // SAFETY: installed only after runtime avx2+fma detection.
        unsafe { skip_dot_i8_inner(ft, lane) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i8_deltas,
        i8_deltas_into,
        i8_deltas_dual,
        l1_dot,
        activate,
        find_nnz,
        l1_dequant,
        l2_tail,
        l3_dot,
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
    /// `activate_inner` walks the HALF-accumulator `0..ACC / 2` in `LANES`-wide
    /// steps with no remainder handling, so it needs `ACC % (2 * LANES) == 0`
    /// — one factor stronger than the assert above, which guards the FOLD
    /// kernels. Load-bearing, not cosmetic: `ft` is a reused buffer, so a
    /// future width that left its tail UNWRITTEN would read the PREVIOUS
    /// position's activations rather than a deterministic zero.
    const _: () = assert!(ACC.is_multiple_of(2 * LANES));

    /// zmm blocks in one accumulator (40 at the shipping ACC = 1280). NOT
    /// `BLOCKS` — `l1_dot_inner` has a local const of that name meaning
    /// `L2 / L2_BLK`, and one module must not spell two things the same way.
    const ACC_BLOCKS: usize = ACC / LANES;

    /// Accumulator blocks held in registers across the feature loop.
    ///
    /// The fold's arithmetic is fixed (`rows × BLOCKS` load/add pairs) but its
    /// LOOP CONTROL scales with the number of tiles, so wider tiles win on
    /// instruction count. Exact divisors of `ACC_BLOCKS` only: a non-dividing
    /// tile pushes the remainder into the one-block tail loop, which costs
    /// more control than it saves; `REGS = 40` (one tile, no tile loop) was
    /// measured and is a disaster — 40 accumulators spill.
    ///
    /// The cap is 5, chosen on CYCLES, not instructions: a wide tile ends
    /// with a burst of `zmm` stores (each split into two store ops on Zen 4)
    /// into the store queue, and narrowing the cap interleaves those stores
    /// with the next tile's loads. Bit-identical at every setting — same
    /// blocks, same order, same arithmetic, no row read more or fewer times.
    /// NOTE: do not re-judge the cap on `instructions:u`; that counter can
    /// only prefer the widest tile (instructions RISE at cap 5 while cycles
    /// fall). Measured at ACC = 1280 only; other widths fall to their largest
    /// divisor ≤ 5 and stay machine-checked by the asserts below.
    const REGS_CAP: usize = if cfg!(feature = "regs20_44") {
        20
    } else if cfg!(feature = "regs4_44") {
        4
    } else if cfg!(feature = "regs8_44") {
        8
    } else if cfg!(feature = "regs10_44") {
        10
    } else {
        5
    };

    const fn pick_regs() -> usize {
        let mut r = REGS_CAP;
        // Down to 1: 1 divides everything, so the `const _` assert below can
        // never fire on a future width whose BLOCKS has no small divisor —
        // that case degrades to the per-block loop rather than breaking the
        // build.
        while r > 1 {
            if ACC_BLOCKS.is_multiple_of(r) {
                return r;
            }
            r -= 1;
        }
        1
    }
    const REGS: usize = pick_regs();
    // The win depends on the tile dividing the accumulator EXACTLY: a
    // remainder would fall into a one-block tail whose per-block control cost
    // is what the wide tile exists to remove.
    const _: () = assert!(
        ACC_BLOCKS.is_multiple_of(REGS),
        "fold tile must divide the accumulator exactly"
    );
    // One zmm is live beyond the accumulators (the row being folded in).
    const _: () = assert!(REGS < 32, "fold tile must leave a spare zmm");

    /// Lanes per tile, and the number of tiles in one accumulator.
    ///
    /// The tile walk has NO remainder: `pick_regs` returns an exact divisor
    /// of `ACC_BLOCKS`, and `TILES * TILE == ACC` below is the machine-checked
    /// statement of that. NOTE: keep the walk as `while off + TILE <= ACC` —
    /// rewriting it as a counted `for` loop was measured slower (LLVM keeps a
    /// real induction variable instead of unrolling); do not restructure it
    /// without measuring.
    const TILE: usize = LANES * REGS;
    const TILES: usize = ACC / TILE;
    const _: () = assert!(TILES * TILE == ACC, "the tile walk must cover ACC exactly");

    /// Expand a list of `(NA, NS)` pairs into const-generic kernel dispatch
    /// arms: `shaped_dispatch!(kern(args…); na, ns; (1,1), (1,2), …)`.
    ///
    /// **Rust idiom note.** A `macro_rules!` repetition is the declarative way
    /// to write a table whose rows differ only in two const-generic arguments:
    /// each row must be its own monomorphisation with compile-time counts.
    /// Taking the callee as `$f:ident` lets one table serve both fold arities;
    /// the argument list is captured as a single `tt` so it need not repeat
    /// per shape. `macro_rules!` is *textually* scoped, hence its position at
    /// the head of the backend.
    macro_rules! shaped_dispatch {
        ($f:ident $args:tt; $na:expr, $ns:expr; $(($a:literal, $s:literal)),* $(,)?) => {
            match ($na, $ns) {
                $(($a, $s) => return $f::<$a, $s> $args,)*
                _ => {}
            }
        };
    }

    // ── The narrow-row decode. ─────────────────────────────────────────────
    //
    // A wide row's 32 lanes are ONE instruction: `vpmovsxbw zmm, [mem]`. A
    // narrow row's 32 lanes are 24 packed bytes and four:
    //
    //   vpermb          spread the 24 bytes so each group of 8 lanes owns a qword
    //   vpmultishiftqb  lift all 8 unaligned 6-bit fields of every qword at once
    //   vpternlogd      (m & 0x3F) ^ 0x20  ⇒  the byte is exactly `w + 32`
    //   vpmovzxbw       widen to 32 i16
    //
    // The `+32` per lane is deliberately NOT undone here: it is lane-uniform,
    // so the whole fold's bias is one `vpsubw` per tile register (and exactly
    // zero when the fold has as many narrow subs as narrow adds). Eight 6-bit
    // fields are 48 bits, so no field ever straddles a qword — the property
    // that makes `vpmultishiftqb` a single instruction here, and why the
    // packer groups by eight.

    #[cfg(feature = "narrow65")]
    /// `vpermb` index: output qword `q` takes source bytes `6q..6q+5`.
    const fn narrow_spread() -> [i8; 32] {
        let mut a = [0i8; 32];
        let mut q = 0;
        while q < 4 {
            let mut i = 0;
            while i < 6 {
                a[q * 8 + i] = (q * 6 + i) as i8;
                i += 1;
            }
            q += 1;
        }
        a
    }

    #[cfg(feature = "narrow65")]
    /// `vpmultishiftqb` control: byte `j` of each qword selects bits
    /// `[6j, 6j+8)` of that qword. The widest is `j = 7` at bit 42, so every
    /// window stays inside its own qword.
    const fn narrow_shift() -> [i8; 32] {
        let mut a = [0i8; 32];
        let mut k = 0;
        while k < 32 {
            a[k] = ((k % 8) * 6) as i8;
            k += 1;
        }
        a
    }

    #[cfg(feature = "narrow65")]
    static NARROW_SPREAD: [i8; 32] = narrow_spread();
    #[cfg(feature = "narrow65")]
    static NARROW_SHIFT: [i8; 32] = narrow_shift();

    /// The four decode constants, materialised ONCE per fold and passed by
    /// value. NOTE: loaded inside `narrow_lanes`, LLVM hoisted them only as
    /// far as the TILE loop — re-broadcast `.rodata` loads competing with the
    /// weight stream for L1, more than the narrow rows save. Passing by value
    /// makes the hoist a property of the source, not the register allocator;
    /// keep it.
    #[cfg(feature = "narrow65")]
    #[derive(Clone, Copy)]
    struct NarrowConsts {
        spread: __m256i,
        shift: __m256i,
        m3f: __m256i,
        m20: __m256i,
    }

    #[cfg(feature = "narrow65")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn narrow_consts() -> NarrowConsts {
        // SAFETY: two 32-byte reads of `'static` 32-byte arrays.
        unsafe {
            NarrowConsts {
                spread: _mm256_loadu_si256(NARROW_SPREAD.as_ptr().cast()),
                shift: _mm256_loadu_si256(NARROW_SHIFT.as_ptr().cast()),
                m3f: _mm256_set1_epi8(0x3F),
                m20: _mm256_set1_epi8(0x20),
            }
        }
    }

    #[cfg(feature = "narrow65")]
    /// The 32 i16 lanes of a narrow row at lane offset `lane`, as `w + 32`.
    ///
    /// # Safety
    /// `row` must point at a live `ACC`-byte row slot and `lane + 32 <= ACC`.
    /// The 32-byte load reads 8 bytes past the packed region on the last group
    /// of a tile; those bytes are inside the row's own (unshrunk) 1280-byte
    /// slot, which is why the layout keeps the slot rather than compacting it.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn narrow_lanes(row: *const i8, lane: usize, c: NarrowConsts) -> __m512i {
        unsafe {
            let packed = _mm256_loadu_si256(row.add(lane * NARROW_BITS / 8).cast());
            let spread = _mm256_permutexvar_epi8(c.spread, packed);
            let fields = _mm256_multishift_epi64_epi8(c.shift, spread);
            // (a & b) ^ c with A=0xF0, B=0xCC, C=0xAA ⇒ imm8 = 0x6A.
            let biased = _mm256_ternarylogic_epi32::<0x6A>(fields, c.m3f, c.m20);
            _mm512_cvtepu8_epi16(biased)
        }
    }

    /// In-place i8 fold — the HalfKA Finny cold
    /// build. Same `REGS`-wide register tile; the row is a 32-byte load
    /// sign-extended to 32 i16 lanes, halving the bytes the tile pulls in.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_inner(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_into_inner`; each row is `ACC` in-bounds i8.
        unsafe {
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(acc.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    /// Two-destination i8 fold — the HalfKA Finny warm refresh, both homes
    /// written from the fold's own registers.
    ///
    /// Shape-dispatched like the into-fold: the warm refresh folds a set
    /// difference between two king buckets whose `(adds, subs)` distribution
    /// concentrates on and near the diagonal, so the `0..=4` grid covers the
    /// large majority of calls. `noshape62` restores the runtime-length body
    /// for the whole site.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_dual_inner(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: the shaped kernel is bit-identical to the generic one — same
        // tiles in the same order, same rows in the same order (adds then
        // subs), same per-lane wrapping i16 arithmetic, same two stores — and
        // each arm is selected only when the list lengths match its const
        // parameters exactly.
        unsafe {
            #[cfg(not(feature = "noshape62"))]
            shaped_dispatch!(i8_dual_shaped(acc, out, weights, adds, subs);
                adds.len(), subs.len();
                (1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (0, 1), (2, 0), (0, 2),
                (3, 0), (0, 3), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3),
                (4, 0), (0, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4), (4, 4));
            // Below the pre-registered `.text` landing rule; not default.
            #[cfg(all(feature = "dual62b", not(feature = "noshape62")))]
            shaped_dispatch!(i8_dual_shaped(acc, out, weights, adds, subs);
                adds.len(), subs.len();
                (5, 0), (0, 5), (5, 1), (1, 5), (5, 2), (2, 5), (5, 3), (3, 5), (5, 4), (4, 5),
                (5, 5),
                (6, 0), (0, 6), (6, 1), (1, 6), (6, 2), (2, 6), (6, 3), (3, 6), (6, 4), (4, 6),
                (6, 5), (5, 6), (6, 6));
            i8_deltas_dual_generic(acc, out, weights, adds, subs)
        }
    }

    /// The Finny warm refresh with **compile-time row counts** — exactly
    /// [`i8_into_shaped`] with the second destination store, and with `acc`
    /// read as well as written (the fold is in-place on the cache entry).
    /// `(0, 0)` is deliberately NOT dispatched: an empty warm refresh must
    /// copy `acc` to `out` verbatim, which the generic body does; keeping it
    /// on one path keeps the empty-list contract in one place.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn i8_dual_shaped<const NA: usize, const NS: usize>(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_dual_generic`. The caller has checked
        // `adds.len() == NA` and `subs.len() == NS`, so every index read below
        // is in bounds; `TILES * TILE == ACC` is a const assert.
        unsafe {
            let mut arow = [0usize; NA];
            for (slot, &feature) in arow.iter_mut().zip(adds) {
                *slot = feature.0;
            }
            let mut srow = [0usize; NS];
            for (slot, &feature) in srow.iter_mut().zip(subs) {
                *slot = feature.0;
            }
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(acc.as_ptr().add(off + k * LANES).cast());
                }
                for &base in &arow {
                    let row = weights.as_ptr().add(base + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &base in &srow {
                    let row = weights.as_ptr().add(base + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    // The `nt43` split is kept verbatim from the generic body —
                    // see its comment for why only `acc` is a candidate for a
                    // non-temporal store.
                    #[cfg(feature = "nt43")]
                    _mm512_stream_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                    #[cfg(not(feature = "nt43"))]
                    _mm512_storeu_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                    _mm512_storeu_si512(out.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    /// The runtime-length Finny warm refresh — the shapes the dispatch above
    /// does not name, plus the empty-list copy contract.
    ///
    /// `#[inline(never)]` mirrors `i8_deltas_into_generic`: once a dispatch
    /// exists, the runtime-length body must not be inlined back into the
    /// dispatcher, or the shaped arms lose their straight-line layout. Dropped
    /// under `noshape62` so the revert build stays byte-identical to the
    /// pre-dispatch tree (a valid control).
    #[target_feature(enable = "avx512f,avx512bw")]
    #[cfg_attr(not(feature = "noshape62"), inline(never))]
    unsafe fn i8_deltas_dual_generic(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_inner`; `out` is a disjoint ACC-wide array.
        unsafe {
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(acc.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    // The `nt43` arm. The two destinations have opposite reuse
                    // distances and only one wants a cache line:
                    //  * `out` is `stack[ply].acc[p]`, read by the head within
                    //    the same node — it must stay a normal store;
                    //  * `acc` is the FINNY CACHE ENTRY, next read at the next
                    //    refresh with this king key — far enough away that the
                    //    line cannot survive, so keeping it costs an RFO now
                    //    and evicts weight rows for nothing.
                    // `_mm512_stream_si512` REQUIRES 64-byte alignment; that
                    // is structural — `FinnyEntry` is `#[repr(C, align(64))]`
                    // and `acc` sits at offset 128 — and the wrapper
                    // `debug_assert`s it.
                    #[cfg(feature = "nt43")]
                    _mm512_stream_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                    #[cfg(not(feature = "nt43"))]
                    _mm512_storeu_si512(acc.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                    _mm512_storeu_si512(out.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    /// The fold with **compile-time row counts**.
    ///
    /// `NA`/`NS` are the exact lengths of `adds`/`subs`, so both row loops
    /// unroll away completely: no loop control, no unroll remainder, no
    /// empty-list clones, and the row offsets are read ONCE for the whole
    /// call instead of once per tile. What is left is straight-line
    /// `vpmovsxbw` / `vpaddw` / `vpsubw` with no branch of any kind.
    ///
    /// Bit-identical to the generic path: same rows, same order (adds then
    /// subs), same per-lane wrapping i16 arithmetic.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn i8_into_shaped<const NA: usize, const NS: usize>(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_into_inner`. The caller has checked
        // `adds.len() == NA` and `subs.len() == NS`, so every index read below
        // is in bounds; `TILES * tile == ACC` is a const assert, so the tile
        // walk covers the accumulator exactly with no tail.
        unsafe {
            let mut arow = [0usize; NA];
            for (slot, &feature) in arow.iter_mut().zip(adds) {
                *slot = feature.0;
            }
            let mut srow = [0usize; NS];
            for (slot, &feature) in srow.iter_mut().zip(subs) {
                *slot = feature.0;
            }
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for &base in &arow {
                    let row = weights.as_ptr().add(base + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &base in &srow {
                    let row = weights.as_ptr().add(base + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_into_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // Dispatch on the SHAPE of the delta set.
        //
        // `apply_dirty_directed` (`sota_accumulator.rs`) builds both lists in
        // `[RowOffset; 2]`, so a HalfKA fold is always one of the four
        // `{1,2}²` shapes; LLVM cannot see that through slices, so the
        // runtime-length body carries loop control the shaped arms compile
        // away. The FullThreats channel routes through this same dispatch
        // with longer lists; the square grid to `(6,6)` covers the large
        // majority of threat folds, and the generic fallback takes the rest
        // plus the EMPTY case, which must remain a plain copy — see the
        // "stale-child bug" note at `sota_accumulator.rs`'s forward fold.
        //
        // Shaped and generic are bit-identical by construction: same tiles in
        // the same order, same rows in the same order (adds then subs), same
        // per-lane wrapping i16 arithmetic. NOTE: the per-fold win of a new
        // arm is NOT predictable from its shape — measure any extension of
        // the grid. `noshape61`/`noshape62` restore the smaller dispatch
        // tables so those contrasts stay measurable; `hot63` (measured and
        // rejected — see `Cargo.toml`) hoists the modal `(1,1)` shape to a
        // leading branch.
        unsafe {
            #[cfg(feature = "hot63")]
            if adds.len() == 1 && subs.len() == 1 {
                return i8_into_shaped::<1, 1>(dst, src, weights, adds, subs);
            }
            match (adds.len(), subs.len()) {
                (1, 1) => return i8_into_shaped::<1, 1>(dst, src, weights, adds, subs),
                (1, 2) => return i8_into_shaped::<1, 2>(dst, src, weights, adds, subs),
                (2, 1) => return i8_into_shaped::<2, 1>(dst, src, weights, adds, subs),
                (2, 2) => return i8_into_shaped::<2, 2>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (1, 0) => return i8_into_shaped::<1, 0>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (0, 1) => return i8_into_shaped::<0, 1>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (2, 0) => return i8_into_shaped::<2, 0>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (0, 2) => return i8_into_shaped::<0, 2>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (3, 0) => return i8_into_shaped::<3, 0>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (0, 3) => return i8_into_shaped::<0, 3>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (1, 3) => return i8_into_shaped::<1, 3>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (3, 1) => return i8_into_shaped::<3, 1>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (2, 3) => return i8_into_shaped::<2, 3>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (3, 2) => return i8_into_shaped::<3, 2>(dst, src, weights, adds, subs),
                #[cfg(not(feature = "noshape61"))]
                (3, 3) => return i8_into_shaped::<3, 3>(dst, src, weights, adds, subs),
                _ => {}
            }
            // The rest of the square grid to `(6,6)`.
            #[cfg(not(feature = "noshape62"))]
            shaped_dispatch!(i8_into_shaped(dst, src, weights, adds, subs);
                adds.len(), subs.len();
                (4, 0), (0, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4), (4, 4),
                (5, 0), (0, 5), (5, 1), (1, 5), (5, 2), (2, 5), (5, 3), (3, 5), (5, 4), (4, 5),
                (5, 5),
                (6, 0), (0, 6), (6, 1), (1, 6), (6, 2), (2, 6), (6, 3), (3, 6), (6, 4), (4, 6),
                (6, 5), (5, 6), (6, 6));
            // The rectangular `subs = 7` / `subs = 8` columns (the residual
            // tail is long sub lists against short add lists): real
            // instruction wins, but below the pre-registered `.text` landing
            // rule — kept behind features, not default.
            #[cfg(all(feature = "shape62d", not(feature = "noshape62")))]
            shaped_dispatch!(i8_into_shaped(dst, src, weights, adds, subs);
                adds.len(), subs.len();
                (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7));
            #[cfg(all(feature = "shape62e", not(feature = "noshape62")))]
            shaped_dispatch!(i8_into_shaped(dst, src, weights, adds, subs);
                adds.len(), subs.len();
                (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8));
            i8_deltas_into_generic(dst, src, weights, adds, subs)
        }
    }

    /// The runtime-length i8 fold — **the whole FullThreats channel** beyond
    /// the shaped table, plus the tests' long lists and the EMPTY-list
    /// contract (both lists empty must copy `src` to `dst` verbatim, which
    /// the in-place twins never have to do). Not `#[cold]`: this is a hot
    /// path, and the label cost real block placement when it was mislabelled.
    ///
    /// NOTE — rewrites of this body that were built and LOST; measure before
    /// believing any control-count argument here:
    /// * merging the add/sub loops into one paired walk (built twice, at two
    ///   tile widths) — LLVM clones and unrolls the two-loop form, and that
    ///   cloning is worth more than the control it duplicates;
    /// * pre-splitting the row lists into compile-time 4-row chunks
    ///   (`as_chunks::<4>`) — LLVM's own unroll scaffolding is cheaper than
    ///   any hand-split.
    /// NOTE — the 1:1 `vpmovsxbw` : `vpaddw` ratio is a hardware floor at
    /// this layout (the convert pipes are the limit, and the fused
    /// `vpmovsxbw zmm,[mem]` load already rides free); i16 weight rows would
    /// double the traffic of a load-stall-bound kernel. Do not re-propose
    /// either.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline(never)]
    unsafe fn i8_deltas_into_generic(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: as `i8_deltas_into_inner`; reads `src`, writes the disjoint
        // `dst`, and each row is a 32-byte load sign-extended to 32 i16 lanes.
        unsafe {
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for &feature in subs {
                    let row = weights.as_ptr().add(feature.0 + off);
                    for (k, vk) in v.iter_mut().enumerate() {
                        let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                        *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    // ── The narrow-plane twins of the two into-kernels. ────────────────────
    //
    // SEPARATE functions rather than a `const NARROW: bool` parameter,
    // deliberately: (a) the decode needs AVX512_VBMI, which the compile-time
    // direct dispatch does not imply, so the narrow bodies must carry a wider
    // `target_feature` than the wide ones may; (b) `row_map` is empty for
    // HalfKA and for every build where the loader declined to pack, in which
    // case the dispatch below never runs — "nothing about the wide path
    // changed" is a property of the diff.

    #[cfg(feature = "narrow65")]
    /// The shaped fold with a per-ROW narrow/wide decision, hoisted: the width
    /// test happens ONCE per row in the prologue, beside the offset hoist, and
    /// the tile walk branches on a register. Per row and not per fold because
    /// most narrow rows do not live in an all-narrow fold.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn i8_into_narrow_shaped<const NA: usize, const NS: usize>(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
        row_map: &[u32],
    ) {
        // SAFETY: as `i8_into_shaped`. Narrow rows read `NARROW_BYTES` of their
        // own 1280-byte slot (plus an 8-byte tail overread that stays inside
        // it); wide rows are byte-for-byte the existing path.
        unsafe {
            let mut arow = [0usize; NA];
            let mut anar = [false; NA];
            for ((slot, flag), &feature) in arow.iter_mut().zip(anar.iter_mut()).zip(adds) {
                (*slot, *flag) = super::row_entry(row_map, feature.0);
            }
            let mut srow = [0usize; NS];
            let mut snar = [false; NS];
            for ((slot, flag), &feature) in srow.iter_mut().zip(snar.iter_mut()).zip(subs) {
                (*slot, *flag) = super::row_entry(row_map, feature.0);
            }
            // Every narrow lane arrives as `w + 32`; the bias is lane-uniform,
            // so the whole fold owes one constant. It is exactly zero whenever
            // the fold has as many narrow subs as narrow adds.
            let bias = 32
                * (anar.iter().filter(|b| **b).count() as i32
                    - snar.iter().filter(|b| **b).count() as i32);
            let biasv = _mm512_set1_epi16(bias as i16);
            let c = narrow_consts();
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for (i, &base) in arow.iter().enumerate() {
                    let row = weights.as_ptr().add(base);
                    if anar[i] {
                        for (k, vk) in v.iter_mut().enumerate() {
                            *vk = _mm512_add_epi16(*vk, narrow_lanes(row, off + k * LANES, c));
                        }
                    } else {
                        let row = row.add(off);
                        for (k, vk) in v.iter_mut().enumerate() {
                            let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                            *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                        }
                    }
                }
                for (i, &base) in srow.iter().enumerate() {
                    let row = weights.as_ptr().add(base);
                    if snar[i] {
                        for (k, vk) in v.iter_mut().enumerate() {
                            *vk = _mm512_sub_epi16(*vk, narrow_lanes(row, off + k * LANES, c));
                        }
                    } else {
                        let row = row.add(off);
                        for (k, vk) in v.iter_mut().enumerate() {
                            let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                            *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                        }
                    }
                }
                if bias != 0 {
                    for vk in v.iter_mut() {
                        *vk = _mm512_sub_epi16(*vk, biasv);
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    #[cfg(feature = "narrow65")]
    /// The runtime-length twin, for the threat folds outside the shaped table.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    #[inline(never)]
    unsafe fn i8_into_narrow_generic(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
        row_map: &[u32],
    ) {
        // SAFETY: as `i8_deltas_into_generic`.
        unsafe {
            let nb = |list: &[RowOffset]| {
                list.iter()
                    .filter(|f| super::row_entry(row_map, f.0).1)
                    .count() as i32
            };
            let bias = 32 * (nb(adds) - nb(subs));
            let biasv = _mm512_set1_epi16(bias as i16);
            let c = narrow_consts();
            let mut off = 0;
            while off + TILE <= ACC {
                let mut v = [_mm512_setzero_si512(); REGS];
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = _mm512_loadu_si512(src.as_ptr().add(off + k * LANES).cast());
                }
                for &feature in adds {
                    let (base, narrow) = super::row_entry(row_map, feature.0);
                    let row = weights.as_ptr().add(base);
                    if narrow {
                        for (k, vk) in v.iter_mut().enumerate() {
                            *vk = _mm512_add_epi16(*vk, narrow_lanes(row, off + k * LANES, c));
                        }
                    } else {
                        let row = row.add(off);
                        for (k, vk) in v.iter_mut().enumerate() {
                            let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                            *vk = _mm512_add_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                        }
                    }
                }
                for &feature in subs {
                    let (base, narrow) = super::row_entry(row_map, feature.0);
                    let row = weights.as_ptr().add(base);
                    if narrow {
                        for (k, vk) in v.iter_mut().enumerate() {
                            *vk = _mm512_sub_epi16(*vk, narrow_lanes(row, off + k * LANES, c));
                        }
                    } else {
                        let row = row.add(off);
                        for (k, vk) in v.iter_mut().enumerate() {
                            let bytes = _mm256_loadu_si256(row.add(k * LANES).cast());
                            *vk = _mm512_sub_epi16(*vk, _mm512_cvtepi8_epi16(bytes));
                        }
                    }
                }
                if bias != 0 {
                    for vk in v.iter_mut() {
                        *vk = _mm512_sub_epi16(*vk, biasv);
                    }
                }
                for (k, vk) in v.iter().enumerate() {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(off + k * LANES).cast(), *vk);
                }
                off += TILE;
            }
        }
    }

    #[cfg(feature = "narrow65")]
    /// Shape dispatch for the narrow path — the same grid the wide path uses,
    /// so a fold does not lose its straight-line coverage merely because one
    /// of its rows is narrow.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    unsafe fn i8_deltas_into_narrow_inner(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
        row_map: &[u32],
    ) {
        // SAFETY: as `i8_deltas_into_inner`.
        unsafe {
            shaped_dispatch!(
                i8_into_narrow_shaped(dst, src, weights, adds, subs, row_map);
                adds.len(), subs.len();
                (1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (0, 1), (2, 0), (0, 2),
                (3, 0), (0, 3), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3),
                (4, 0), (0, 4), (4, 1), (1, 4), (4, 2), (2, 4), (4, 3), (3, 4), (4, 4),
                (5, 0), (0, 5), (5, 1), (1, 5), (5, 2), (2, 5), (5, 3), (3, 5), (5, 4), (4, 5),
                (5, 5),
                (6, 0), (0, 6), (6, 1), (1, 6), (6, 2), (2, 6), (6, 3), (3, 6), (6, 4), (4, 6),
                (6, 5), (5, 6), (6, 6));
            i8_into_narrow_generic(dst, src, weights, adds, subs, row_map)
        }
    }

    #[cfg(feature = "narrow65")]
    /// Public entry: the FullThreats fold when the loader packed a narrow plane.
    ///
    /// # Safety
    /// Installed only behind [`super::narrow_supported`], which is also what
    /// gates the packing — so reaching this function implies the CPU has the
    /// VBMI the decode needs.
    pub(super) fn i8_deltas_into_narrow(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
        row_map: &[u32],
    ) {
        // SAFETY: see the doc comment.
        unsafe { i8_deltas_into_narrow_inner(dst, src, weights, adds, subs, row_map) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn l1_dot_inner(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // dpbusd is the whole kernel: for each dword lane it accumulates the
        // u8×i8 four-product dot directly into i32 — the maddubs+madd pair
        // (and its i16 saturation reasoning) collapses into one instruction
        // per (group, block). Loop order is GROUP-outer with all L2/16 output
        // blocks held in registers; the blk-outer form re-walked the nnz list
        // and re-broadcast each FT group once per block.
        //
        // CHAINS: `vpdpbusd` is ~4-cycle latency at ~1-per-cycle throughput
        // (one effective 512-bit pipe — Zen 4 double-pumps), so at `L2 = 16`
        // (BLOCKS = 1) every active group would accumulate into one zmm and
        // serialise. Four independent accumulator sets cover the latency; the
        // measured knee is exactly 4 with a flat curve beyond, so raising
        // CHAINS buys nothing — do not re-propose 8. Widths with BLOCKS >= 4
        // already have independent chains and keep CHAINS = 1. The remainder
        // tail folds into `sums[0]`; spreading it was measured worse.
        // Bit-identical either way: i32 wrapping adds are order-independent.
        // SAFETY: active groups address four FT bytes and L2 complete
        // four-byte output rows in loader-validated storage.
        unsafe {
            const BLOCKS: usize = L2 / L2_BLK;
            const CHAINS: usize = if BLOCKS >= 4 { 1 } else { 4 / BLOCKS };
            let mut sums = [[_mm512_setzero_si512(); BLOCKS]; CHAINS];
            for (blk, sum) in sums[0].iter_mut().enumerate() {
                *sum = _mm512_loadu_si512(pre.as_ptr().add(blk * L2_BLK).cast());
            }
            // NOTE: prefetching upcoming rows (here and on the FT weight
            // rows) was measured at noise — the table is L2-resident; do not
            // re-propose.
            //
            // `as_chunks` (not `chunks_exact`) so the inner trip count is a
            // CONSTANT: it removes the per-group conditional pointer advance
            // LLVM emits when it cannot bound the loop.
            let (quads, tail) = nnz.as_chunks::<CHAINS>();
            for quad in quads {
                for (chain, &group) in quad.iter().enumerate() {
                    let group = usize::from(group);
                    let inputs = _mm512_set1_epi32(ft_group(ft, group));
                    let base = weights.as_ptr().add(group * (L2 * 4));
                    for (blk, sum) in sums[chain].iter_mut().enumerate() {
                        let rows = _mm512_loadu_si512(base.add(blk * 64).cast());
                        *sum = _mm512_dpbusd_epi32(*sum, inputs, rows);
                    }
                }
            }
            for &group in tail {
                let group = usize::from(group);
                let inputs = _mm512_set1_epi32(ft_group(ft, group));
                let base = weights.as_ptr().add(group * (L2 * 4));
                for (blk, sum) in sums[0].iter_mut().enumerate() {
                    let rows = _mm512_loadu_si512(base.add(blk * 64).cast());
                    *sum = _mm512_dpbusd_epi32(*sum, inputs, rows);
                }
            }
            for blk in 0..BLOCKS {
                let mut total = sums[0][blk];
                for chain in sums.iter().skip(1) {
                    total = _mm512_add_epi32(total, chain[blk]);
                }
                _mm512_storeu_si512(pre.as_mut_ptr().add(blk * L2_BLK).cast(), total);
            }
        }
    }

    /// NOTE — this loop is at its issue floor; the standing guards, each from
    /// measurement:
    /// * On Zen 4 a folded memory operand costs exactly ONE issue slot (the
    ///   same as the register-form op), and both loads that CAN fold here
    ///   already do — so CSE-ing a folded load into a separate `vmovdqu64`
    ///   STRICTLY LOSES (three ops for two).
    /// * `vpaddsw` → `vpaddw` is NOT bit-safe: `vpmullw` is wrapping by
    ///   design, so the saturating add really can fire, and wrapping instead
    ///   flips those lanes to the opposite clamp.
    /// * Never move work from `activate` into the fold — the fold runs ~8×
    ///   more often per node.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn activate_inner<const SHIFT: u32>(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
        stm: usize,
    ) {
        const H: usize = ACC / 2;
        // The merge as one i16 pipeline (32 lanes/zmm; the old path widened
        // to i32 at 16 lanes/zmm). Exactness, per step:
        //   t << SHIFT  stays in i16 — the |t| bound is debug-asserted below
        //   p·scale + t·2^SHIFT  via vpmullw + vpsllw + vpaddsw — the
        //               saturating add can only saturate beyond the
        //               [0, FT_QUANT] clamp, so the clamped result is exact
        //   clamp       via vpmaxsw/vpminsw
        //   (l·r)>>9    via vpsllw<7> + vpmulhw (l ≤ 255 ⇒ l·128 < 2^15)
        //   store       via vpmovwb — products ≤ 127, truncation == value.
        // SAFETY: each iteration consumes 32 in-bounds i16 lanes per half.
        unsafe {
            #[cfg(debug_assertions)]
            {
                // `t << SHIFT` must stay inside i16 (8191 at the CEN2 shift of
                // 2, 32767 at a QAT net's shift of 0).
                let bound = (i16::MAX as u16) >> SHIFT;
                for side in threat {
                    for &t in side.iter() {
                        debug_assert!(
                            t.unsigned_abs() <= bound,
                            "threat lane {t} exceeds the i16 activate bound {bound}"
                        );
                    }
                }
            }
            let zero = _mm512_setzero_si512();
            let ceiling = _mm512_set1_epi16(FT_QUANT as i16);
            // LANE OUTER, PERSPECTIVE INNER, AND THE SCALE IN A REGISTER.
            // `scale` is the one array here that does NOT depend on the
            // perspective; perspective-outer streamed the whole array twice
            // per evaluation. `vpmullw` is commutative, so the REUSED operand
            // (the scale) lives in a register and the per-perspective
            // accumulator load folds into the multiply instead — removing an
            // op AND a load per merge pair, the only strict improvement of
            // the two orders. Bit-identical: every output byte is the same
            // expression over the same inputs.
            for lane in (0..H).step_by(32) {
                // The scale is u8, so this is a 32-byte load zero-extended
                // into 32 i16 lanes instead of a 64-byte load: same op count
                // (`vpmovzxbw` takes its memory operand in one issue slot),
                // half the bytes streamed.
                let sv_left =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256(scale.as_ptr().add(lane).cast()));
                let sv_right =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256(scale.as_ptr().add(lane + H).cast()));
                for flip in 0..2 {
                    let perspective = stm ^ flip;
                    let p = &pst[perspective];
                    let t = &threat[perspective];
                    // The threat rescale, as ONE SHIFT: the two FTs are
                    // quantised differently (HalfKA at FT_QUANT = 255,
                    // FullThreats at the net's own `threat_quant`), and the
                    // ratio is a GAIN on one eval channel, so exactness is
                    // not required — it is carried as the nearest power of
                    // two, `256/threat_quant` (`sota_head::threat_shift_for`).
                    //
                    // NOTE: the shift is a const generic, not a runtime
                    // parameter, so LLVM folds the accumulator load INTO the
                    // shift — `vpsllw`'s register-count form takes its memory
                    // operand as the COUNT, not the data, so a runtime shift
                    // forces a separate load per merge. Do not make it
                    // runtime, and do not rebase the stored weights into the
                    // FT_QUANT frame at load (measured: i8 saturation, large
                    // eval error), without re-measuring.
                    //
                    // The HalfKA rescale, as ONE MULTIPLY: the table is i8
                    // with a per-lane integer scale, so the accumulator lives
                    // in the reduced frame `FT_QUANT / scale[j]` and
                    // `vpmullw` puts it back before the clamp, in i16 lanes —
                    // it wraps exactly where the old FT_QUANT-frame
                    // accumulator wrapped, and the scalar and avx2 backends
                    // pin the same width to agree.
                    let merge = |offset: usize, sv: __m512i| {
                        let pv = _mm512_loadu_si512(p.as_ptr().add(offset).cast());
                        let tv = _mm512_loadu_si512(t.as_ptr().add(offset).cast());
                        let scaled = _mm512_slli_epi16::<SHIFT>(tv);
                        let sum = _mm512_adds_epi16(_mm512_mullo_epi16(pv, sv), scaled);
                        _mm512_min_epi16(_mm512_max_epi16(sum, zero), ceiling)
                    };
                    let left = merge(lane, sv_left);
                    let right = merge(lane + H, sv_right);
                    let prod = _mm512_mulhi_epi16(_mm512_slli_epi16::<7>(left), right);
                    _mm256_storeu_si256(
                        out.as_mut_ptr().add(flip * H + lane).cast(),
                        _mm512_cvtepi16_epi8(prod),
                    );
                }
            }
        }
    }

    pub(super) fn i8_deltas(
        acc: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i8_deltas_inner(acc, weights, adds, subs) }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn i8_deltas_dual(
        acc: &mut [i16; ACC],
        out: &mut [i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i8_deltas_dual_inner(acc, out, weights, adds, subs) }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn i8_deltas_into(
        dst: &mut [i16; ACC],
        src: &[i16; ACC],
        weights: &[i8],
        adds: &[RowOffset],
        subs: &[RowOffset],
    ) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { i8_deltas_into_inner(dst, src, weights, adds, subs) }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn l1_dot(pre: &mut [i32; L2], ft: &[u8; ACC], nnz: &[u16], weights: &[i8]) {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { l1_dot_inner(pre, ft, nnz, weights) }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
        stm: usize,
        threat_shift: u32,
    ) {
        // ONE branch per eval, then a kernel whose shift is an immediate — the
        // form that folds the load (see `activate_inner`). The `_` arm is
        // `MAX_THREAT_SHIFT`, which `sota_head::threat_shift_for` guarantees is
        // the largest value any loaded net can carry; the debug assert catches
        // a hand-built `SotaHead` that violates it, since the field is public.
        debug_assert!(
            threat_shift <= MAX_THREAT_SHIFT,
            "threat_shift {threat_shift} > MAX_THREAT_SHIFT — \
             add a match arm and a monomorphisation"
        );
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe {
            match threat_shift {
                0 => activate_inner::<0>(out, pst, threat, scale, stm),
                1 => activate_inner::<1>(out, pst, threat, scale, stm),
                _ => activate_inner::<2>(out, pst, threat, scale, stm),
            }
        }
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

    #[cfg_attr(feature = "split58", inline(never))]
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
        // bias; per i one broadcast + two fused mul-adds. j-lane order
        // preserved and i-sequential per lane ⇒ bit-identical to the scalar
        // reference. SAFETY: all loads/stores cover in-bounds 16-lane blocks
        // of loader-validated storage.
        //
        // NOTE: each accumulator carries an L2-deep FMA chain — a real ILP
        // deficit — and a 4-accumulator version was priced and REJECTED: the
        // prize is below panel resolution and it is NOT eval-bit-identical
        // (f32 reassociation changes the rounding), so it would move the eval
        // hash and require changing the scalar reference and both SIMD
        // backends. Re-price if `L2` ever grows; the chain deepens with L2.
        unsafe {
            // `L3 / 16` 16-lane accumulators, not two hardcoded ones: at
            // L3 = 32 this is the identical pair in the identical order (so
            // bit-identical), and at any other multiple of 16 it stays in
            // bounds. The fixed pair silently wrote `out[16..32]` on an
            // L3 = 16 build — a 64-byte store off the end of the caller's
            // stack array.
            const BLK: usize = 16;
            const NB: usize = L3 / BLK;
            let mut acc = [_mm512_setzero_ps(); NB];
            for (b, a) in acc.iter_mut().enumerate() {
                *a = _mm512_loadu_ps(bias.as_ptr().add(b * BLK));
            }
            for (i, &input) in l1.iter().enumerate() {
                let li = _mm512_set1_ps(input);
                let row = weights.as_ptr().add(i * L3);
                for (b, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(b * BLK)), li, *a);
                }
            }
            let zero = _mm512_setzero_ps();
            let one = _mm512_set1_ps(1.0);
            let screlu = |acc| {
                let clipped = _mm512_min_ps(_mm512_max_ps(acc, zero), one);
                _mm512_mul_ps(clipped, clipped)
            };
            for (b, &a) in acc.iter().enumerate() {
                _mm512_storeu_ps(out.as_mut_ptr().add(b * BLK), screlu(a));
            }
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
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

    /// The 32-lane iota — the full-`zmm` `vpcompressw` source.
    const NNZ_IOTA32: [u16; 32] = {
        let mut t = [0u16; 32];
        let mut i = 0;
        while i < 32 {
            t[i] = i as u16;
            i += 1;
        }
        t
    };

    /// `find_nnz` writes a whole compaction row at `out[count]`, so the row
    /// width must divide the group count or the last store runs off the end.
    const _: () = assert!(
        (ACC / 4).is_multiple_of(32),
        "find_nnz_vbmi2 stores a full 32-lane row at out[count]; ACC/4 must be a multiple of 32"
    );

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn find_nnz_vbmi2_inner(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // THIRTY-TWO 4-byte groups per iteration: two independent 64 B loads,
        // test the dwords against zero into two 16-bit masks concatenated
        // into a `__mmask32`, `vpcompressw` the constant iota to the mask's
        // set positions (ascending), add the running group base — the same
        // index set as the scalar scan and the AVX2 movemask+LUT form, with
        // no 4 KiB table. The full-zmm form also halves the serial `count`
        // chain, the loop's only loop-carried dependency (each store's
        // address depends on every previous popcount).
        // SAFETY: each iteration reads 128 in-bounds FT bytes. Each store
        // writes a full 32-lane row at `out[count]`: entering chunk c,
        // count <= 32c <= ACC/4 - 32, so the row fits even when every group so
        // far was nonzero; lanes past the popcount are dead padding
        // overwritten by the next store or ignored via the returned count.
        unsafe {
            let iota = _mm512_loadu_si512(NNZ_IOTA32.as_ptr().cast());
            let mut count = 0usize;
            for chunk in 0..ACC / 4 / 32 {
                let lo = _mm512_loadu_si512(ft.as_ptr().add(chunk * 128).cast());
                let hi = _mm512_loadu_si512(ft.as_ptr().add(chunk * 128 + 64).cast());
                let mask: u32 = u32::from(_mm512_test_epi32_mask(lo, lo))
                    | (u32::from(_mm512_test_epi32_mask(hi, hi)) << 16);
                let row = _mm512_maskz_compress_epi16(mask, iota);
                let base = _mm512_set1_epi16((chunk * 32) as i16);
                _mm512_storeu_si512(
                    out.as_mut_ptr().add(count).cast(),
                    _mm512_add_epi16(row, base),
                );
                count += mask.count_ones() as usize;
            }
            count
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn find_nnz_vbmi2(ft: &[u8; ACC], out: &mut [u16; ACC / 4]) -> usize {
        // SAFETY: installed only after runtime avx512 vl+vbmi2 detection (or
        // compiled with the features statically enabled).
        unsafe { find_nnz_vbmi2_inner(ft, out) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn l3_dot_inner(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
        // One 16-lane accumulator IS the scalar reference's 16 partial lanes,
        // so the whole L3 = 32 dot is two `vfmadd231ps` and the halving tree.
        // SAFETY: L3 = 32 is exactly two in-bounds 16-lane blocks.
        unsafe {
            let mut acc = _mm512_setzero_ps();
            for i in (0..L3).step_by(16) {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(weights.as_ptr().add(i)),
                    _mm512_loadu_ps(l2.as_ptr().add(i)),
                    acc,
                );
            }
            // stride 8 of the reference's halving tree; `halve8` finishes it.
            super::avx2::halve8(_mm256_add_ps(
                _mm512_castps512_ps256(acc),
                _mm512_extractf32x8_ps::<1>(acc),
            ))
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn l3_dot(weights: &[f32; L3], l2: &[f32; L3]) -> f32 {
        // SAFETY: installed only after runtime avx512f detection (or compiled
        // with the feature statically enabled).
        unsafe { l3_dot_inner(weights, l2) }
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
        // `vpdpbusd` is the whole kernel: 64 u8 × i8 products accumulated into
        // 16 i32 lanes per instruction. FOUR independent accumulator chains
        // cover its ~4-cycle latency at ~1/cycle throughput; a single chain
        // would serialise. Bit-identical by contract: i32 addition is
        // associative, so each backend may reduce in any order. Every shipped
        // ACC is a multiple of 256, so the 4-wide body covers the whole
        // vector; the tail loop is kept for a future width, not because it
        // runs today.
        // SAFETY: caller guarantees equal lengths, a multiple of 64.
        unsafe {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let n = ft.len();
            let mut i = 0;
            while i + 256 <= n {
                let l0 = _mm512_loadu_si512(ft.as_ptr().add(i).cast());
                let w0 = _mm512_loadu_si512(lane.as_ptr().add(i).cast());
                let l1 = _mm512_loadu_si512(ft.as_ptr().add(i + 64).cast());
                let w1 = _mm512_loadu_si512(lane.as_ptr().add(i + 64).cast());
                let l2 = _mm512_loadu_si512(ft.as_ptr().add(i + 128).cast());
                let w2 = _mm512_loadu_si512(lane.as_ptr().add(i + 128).cast());
                let l3 = _mm512_loadu_si512(ft.as_ptr().add(i + 192).cast());
                let w3 = _mm512_loadu_si512(lane.as_ptr().add(i + 192).cast());
                a0 = _mm512_dpbusd_epi32(a0, l0, w0);
                a1 = _mm512_dpbusd_epi32(a1, l1, w1);
                a2 = _mm512_dpbusd_epi32(a2, l2, w2);
                a3 = _mm512_dpbusd_epi32(a3, l3, w3);
                i += 256;
            }
            while i < n {
                let a = _mm512_loadu_si512(ft.as_ptr().add(i).cast());
                let w = _mm512_loadu_si512(lane.as_ptr().add(i).cast());
                a0 = _mm512_dpbusd_epi32(a0, a, w);
                i += 64;
            }
            _mm512_reduce_add_epi32(_mm512_add_epi32(
                _mm512_add_epi32(a0, a1),
                _mm512_add_epi32(a2, a3),
            ))
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn skip_dot_i8(ft: &[u8], lane: &[i8]) -> i32 {
        // SAFETY: installed only after runtime avx512f+bw+vnni detection.
        unsafe { skip_dot_i8_inner(ft, lane) }
    }

    pub(super) const KERNELS: Kernels = Kernels {
        i8_deltas,
        i8_deltas_into,
        i8_deltas_dual,
        l1_dot,
        activate,
        // The AVX2 movemask+compact form; the 512-bit variant needs
        // vpcompressw (avx512-vbmi2+vl — see KERNELS_VBMI2).
        find_nnz: avx2::find_nnz,
        l1_dequant,
        l2_tail,
        l3_dot,
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

    /// The identity HalfKA scale: the kernel tests drive the merge with
    /// weights that already sit in the FT_QUANT frame.
    const UNIT_SCALE: [u8; ACC] = [1; ACC];

    /// A non-unit HalfKA scale, so the backends' `vpmullw` is actually
    /// exercised against the scalar `wrapping_mul` rather than multiplied by
    /// one. 4 is the modal column scale in the canonical net.
    const FOUR_SCALE: [u8; ACC] = [4; ACC];

    #[test]
    fn dispatched_activate_matches_scalar_bitwise() {
        // The i16 pipeline must equal the scalar reference on every lane,
        // INCLUDING the negative half-cases (t ≡ 32 mod 64, t < 0) and
        // values well past the |t| range real nets produce.
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
        // Every shift a container can ask for: 2 = CEN2, 0 = QAT at
        // threat_quant 256, 1 = the 128 midpoint. The planted extremes keep
        // `t << 2` inside i16. Crossed with the HalfKA scale, so the two
        // rescales are exercised together rather than one at a time.
        for threat_shift in [0u32, 1, 2] {
            for scale in [&UNIT_SCALE, &FOUR_SCALE] {
                for stm in 0..2 {
                    // POISONED with two DIFFERENT fills, not zeroed: most
                    // activations are zero after the clamp, so zeroed buffers
                    // let a lane NEITHER kernel wrote compare equal. `ft` is a
                    // reused buffer in the engine, so an unwritten lane is a
                    // stale activation, not a harmless zero — this test is
                    // what says every backend writes all of it.
                    let mut want = [0xABu8; ACC];
                    let mut got = [0x5Cu8; ACC];
                    scalar::activate(&mut want, &pst, &threat, scale, stm, threat_shift);
                    activate(&mut got, &pst, &threat, scale, stm, threat_shift);
                    assert_eq!(want, got, "stm {stm} shift {threat_shift}");
                }
            }
        }
    }

    #[test]
    fn dispatched_skip_dot_i8_matches_scalar_exactly() {
        // i32 accumulation: every backend must agree with the scalar reference
        // EXACTLY, whatever order it reduces in. Uses the real extremes — ft at
        // its 127 ceiling and weights at both i8 rails — so the avx2 backend's
        // `maddubs` no-saturation argument (2*127*127 < i16::MAX) is exercised.
        // Full-ACC vectors, so the avx512 four-chain body (not just its tail
        // loop) is the path driven.
        let mut ft = vec![0u8; ACC];
        let mut lane = vec![0i8; ACC];
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
    fn dispatched_i8_folds_match_scalar_at_wrapping_extremes() {
        let mut i8_weights = vec![0i8; 7 * ACC];
        for (i, value) in i8_weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let seed = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds = [0, 2, 2, 6, 1].map(RowOffset::of);
        let subs = [3, 5, 0, 3].map(RowOffset::of);

        let mut reference = seed;
        scalar::i8_deltas(&mut reference, &i8_weights, &adds, &subs);
        let mut dispatched = seed;
        apply_halfka_deltas(&mut dispatched, &i8_weights, &adds, &subs);
        assert_eq!(dispatched, reference);

        let mut reference = [0i16; ACC];
        scalar::i8_deltas_into(&mut reference, &seed, &i8_weights, &adds, &subs);
        let mut dispatched = [0i16; ACC];
        apply_threat_deltas_into(&mut dispatched, &seed, &i8_weights, &adds, &subs, &[]);
        assert_eq!(dispatched, reference);
    }

    /// The narrow plane is BIT-EXACT, checked the only way that proves it:
    /// fold the same rows twice, once from the 8-bit table and once from a
    /// packed table, and demand the same accumulator. A mixed narrow/wide map
    /// is deliberate — the per-ROW dispatch is the branch that matters.
    #[test]
    #[cfg(feature = "narrow65")]
    fn narrow_rows_fold_bit_exactly() {
        if !super::narrow_supported() {
            return;
        }
        const ROWS: usize = 8;
        // In range for six signed bits, so every row is packable; a couple of
        // rows are left wide by the map anyway to force the mixed path.
        let wide: Vec<i8> = (0..ROWS * ACC)
            .map(|i| ((i * 7 + i / ACC) % 64) as i8 - 32)
            .collect();
        const NARROW_ROWS: [usize; 5] = [0, 1, 3, 4, 6];
        let is_narrow = |r: usize| NARROW_ROWS.contains(&r);
        let mut buf = [0u8; super::NARROW_BYTES];

        // Layout A: packed IN PLACE, wide stride kept, a dead tail behind
        // every narrow row.
        let mut slot = wide.clone();
        let mut slot_map = vec![0u32; ROWS];
        for r in 0..ROWS {
            slot_map[r] = (r * ACC) as u32;
            if !is_narrow(r) {
                continue;
            }
            super::pack_narrow_row(&wide[r * ACC..(r + 1) * ACC], &mut buf);
            for (dst, &src) in slot[r * ACC..r * ACC + super::NARROW_BYTES]
                .iter_mut()
                .zip(buf.iter())
            {
                *dst = src as i8;
            }
            slot_map[r] |= 1;
        }

        // Layout B: COMPACTED, wide plane then narrow plane, no hole. The
        // trailing pad absorbs the last narrow row's 8-byte tail overread,
        // exactly as the loader's does.
        let n_wide = (0..ROWS).filter(|&r| !is_narrow(r)).count();
        let mut pack = vec![0i8; n_wide * ACC + (ROWS - n_wide) * super::NARROW_BYTES + 64];
        let mut pack_map = vec![0u32; ROWS];
        let (mut w_cur, mut n_cur) = (0usize, n_wide * ACC);
        for r in 0..ROWS {
            if is_narrow(r) {
                super::pack_narrow_row(&wide[r * ACC..(r + 1) * ACC], &mut buf);
                for (dst, &src) in pack[n_cur..n_cur + super::NARROW_BYTES]
                    .iter_mut()
                    .zip(buf.iter())
                {
                    *dst = src as i8;
                }
                pack_map[r] = n_cur as u32 | 1;
                n_cur += super::NARROW_BYTES;
            } else {
                pack[w_cur..w_cur + ACC].copy_from_slice(&wide[r * ACC..(r + 1) * ACC]);
                pack_map[r] = w_cur as u32;
                w_cur += ACC;
            }
        }

        // The scalar decode must agree with the source, lane for lane, in both
        // layouts — this is what `add_threat_column`'s from-scratch path reads.
        for r in NARROW_ROWS {
            for (table, map) in [(&slot, &slot_map), (&pack, &pack_map)] {
                let base = (map[r] & !1) as usize;
                for j in (0..ACC).step_by(37) {
                    assert_eq!(
                        super::unpack_narrow_lane(&table[base..], j),
                        i16::from(wide[r * ACC + j]),
                        "scalar unpack disagrees at row {r} lane {j}"
                    );
                }
            }
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as i16).wrapping_mul(11) - 300);
        for (adds, subs) in [
            (&[0usize][..], &[1usize][..]),
            (&[0, 2][..], &[1, 3][..]),
            (&[2, 5, 7][..], &[0][..]),
            (&[0, 1, 2, 3, 4, 5, 6, 7][..], &[7, 6, 5, 4, 3, 2, 1, 0][..]),
            (&[][..], &[4][..]),
        ] {
            let a: Vec<RowOffset> = adds.iter().map(|&f| RowOffset::of(f)).collect();
            let s: Vec<RowOffset> = subs.iter().map(|&f| RowOffset::of(f)).collect();
            let mut want = [0i16; ACC];
            apply_threat_deltas_into(&mut want, &seed, &wide, &a, &s, &[]);
            for (name, table, map) in [
                ("slot-preserving", &slot, &slot_map),
                ("compacted", &pack, &pack_map),
            ] {
                let mut got = [0i16; ACC];
                apply_threat_deltas_into(&mut got, &seed, table, &a, &s, map);
                assert_eq!(got, want, "{name} fold differs for {adds:?} / {subs:?}");
            }
        }
    }

    #[test]
    fn dispatched_i8_deltas_match_scalar_on_long_feature_lists() {
        // The wrapping-extremes test above runs 5 adds / 4 subs, the
        // incremental-update shape. The Finny fold is the other caller and it
        // arrives with feature lists an order of magnitude longer, where the
        // per-lane i16 wrapping has many more chances to diverge; sizes below
        // straddle the retired 96-row chunk boundary in both directions.
        const ROWS: usize = 11;
        let mut weights = vec![0i8; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds: Vec<RowOffset> = (0..240).map(|i| RowOffset::of((i * 7) % ROWS)).collect();
        let subs: Vec<RowOffset> = (0..200)
            .map(|i| RowOffset::of((i * 5 + 3) % ROWS))
            .collect();

        for (a, s) in [
            (&adds[..], &subs[..]),
            (&adds[..96], &subs[..96]),
            (&adds[..97], &subs[..1]),
            (&adds[..0], &subs[..200]),
        ] {
            let mut reference = seed;
            scalar::i8_deltas(&mut reference, &weights, a, s);
            let mut dispatched = seed;
            apply_halfka_deltas(&mut dispatched, &weights, a, s);
            assert_eq!(dispatched, reference, "adds={} subs={}", a.len(), s.len());
        }
    }

    /// The shape-specialised dispatch arms — `(adds.len(), subs.len())` ∈
    /// {0..=2}² — are the shapes the engine actually executes; the module's
    /// other kernel tests run 4-5 rows or 200+, which land in the generic
    /// fallback.
    ///
    /// Mutations it catches: a transposed `(NA, NS)` arm (which would apply a
    /// sub as an add — invisible to a node count), a shaped body that tiles
    /// short, or a dropped `return` that ran the generic pass on top of the
    /// shaped one.
    ///
    /// The empty shape is in the table on purpose: the accumulator's forward
    /// fold relies on `apply_threat_deltas_into(dst, src, w, &[], &[])` being
    /// `dst = src` lane for lane; a fast-path `return` on empty lists would
    /// be a silent stale-child bug in release.
    #[test]
    fn every_dispatched_i8_fold_shape_matches_scalar() {
        const ROWS: usize = 7;
        let mut weights = vec![0i8; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let src: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        // Distinct row indices per slot so a transposed arm cannot cancel.
        let adds = [1usize, 4].map(RowOffset::of);
        let subs = [2usize, 6].map(RowOffset::of);
        for na in 0..=2 {
            for ns in 0..=2 {
                let (a, s) = (&adds[..na], &subs[..ns]);
                let mut reference = src;
                scalar::i8_deltas(&mut reference, &weights, a, s);
                let mut dispatched = [0i16; ACC];
                apply_threat_deltas_into(&mut dispatched, &src, &weights, a, s, &[]);
                assert_eq!(dispatched, reference, "shape ({na}, {ns})");
                if na == 0 && ns == 0 {
                    assert_eq!(dispatched, src, "the empty shape must be a pure copy");
                }
            }
        }
    }

    /// The GENERIC fold — the threat channel's kernel — over the full 15×15
    /// grid of `(adds.len(), subs.len())`, against the scalar reference: the
    /// short, UNEQUAL lengths that are the threat channel's actual working
    /// range, which no other test walks.
    ///
    /// Mutations it catches: any restructuring of the two row loops that
    /// mis-slices the lists, drops the longer side's tail, or applies a sub as
    /// an add. The index pools are disjoint and the weight table is dense, so
    /// a dropped or mis-signed row cannot cancel coincidentally.
    #[test]
    fn the_generic_fold_matches_scalar_at_every_short_length_pair() {
        const ROWS: usize = 29;
        let mut weights = vec![0i8; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let src: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        // Disjoint index pools, so an add applied as a sub cannot cancel.
        let adds: Vec<RowOffset> = (0..14).map(|i| RowOffset::of(i * 2)).collect();
        let subs: Vec<RowOffset> = (0..14).map(|i| RowOffset::of(i * 2 + 1)).collect();
        for na in 0..=14 {
            for ns in 0..=14 {
                let (a, s) = (&adds[..na], &subs[..ns]);
                let mut reference = src;
                scalar::i8_deltas(&mut reference, &weights, a, s);
                let mut folded = [0i16; ACC];
                apply_threat_deltas_into(&mut folded, &src, &weights, a, s, &[]);
                assert_eq!(folded, reference, "lengths ({na}, {ns})");
            }
        }
    }

    /// The two-destination Finny kernel must equal "fold in place, then copy"
    /// on BOTH of its homes, on every backend the
    /// CPU offers. The point of the kernel is that the second destination is
    /// written from the fold's own registers, so a backend that wired the
    /// stores up wrongly would silently desync the search stack from the cache
    /// — and that desync survives no test the engine had, because the two are
    /// only compared at the NEXT refresh with the same king key.
    #[test]
    fn dual_kernel_writes_both_homes_identically_on_every_backend() {
        const ROWS: usize = 11;
        let mut weights = vec![0i8; ROWS * ACC];
        for (i, value) in weights.iter_mut().enumerate() {
            *value = (i as u8).wrapping_mul(173) as i8;
        }
        let seed: [i16; ACC] = std::array::from_fn(|i| (i as u16).wrapping_mul(49711) as i16);
        let adds: Vec<RowOffset> = (0..240).map(|i| RowOffset::of((i * 7) % ROWS)).collect();
        let subs: Vec<RowOffset> = (0..200)
            .map(|i| RowOffset::of((i * 5 + 3) % ROWS))
            .collect();

        #[allow(clippy::type_complexity)]
        let mut backends: Vec<(&str, ApplyI8Fn, ApplyI8DualFn)> =
            vec![("scalar", scalar::i8_deltas, scalar::i8_deltas_dual)];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                backends.push(("avx2", avx2::i8_deltas, avx2::i8_deltas_dual));
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                backends.push(("avx512", avx512::i8_deltas, avx512::i8_deltas_dual));
            }
        }

        for (a, s) in [
            (&adds[..], &subs[..]),
            (&adds[..96], &subs[..96]),
            (&adds[..0], &subs[..0]),
        ] {
            let mut reference = seed;
            scalar::i8_deltas(&mut reference, &weights, a, s);
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
        let adds = [0, 2, 2, 6, 1].map(RowOffset::of);
        let subs = [3, 5, 0, 3].map(RowOffset::of);
        // LONG lists, run through EVERY backend: the `_into` kernels used to
        // chunk their feature lists at 96 and re-read `dst` per chunk; the
        // one-pass form must stay equivalent, which is exactly what a >96-row
        // list tests. These straddle the old boundary in both directions.
        let long_adds: Vec<RowOffset> = (0..240).map(|i| RowOffset::of((i * 7 + 3) % 7)).collect();
        let long_subs: Vec<RowOffset> = (0..200).map(|i| RowOffset::of((i * 5 + 1) % 7)).collect();

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
            scalar::i8_deltas(&mut reference, &i8_weights, &adds, &subs);
            let mut got = seed;
            (kernels.i8_deltas)(&mut got, &i8_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i8_deltas");

            let mut reference = seed;
            let mut ref_out = [0i16; ACC];
            scalar::i8_deltas_dual(&mut reference, &mut ref_out, &i8_weights, &adds, &subs);
            let mut got = seed;
            let mut got_out = [0i16; ACC];
            (kernels.i8_deltas_dual)(&mut got, &mut got_out, &i8_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i8_deltas_dual (acc)");
            assert_eq!(got_out, ref_out, "{name} i8_deltas_dual (out)");

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
            scalar::i8_deltas_into(&mut reference, &seed, &i8_weights, &adds, &subs);
            let mut got = [0i16; ACC];
            (kernels.i8_deltas_into)(&mut got, &seed, &i8_weights, &adds, &subs);
            assert_eq!(got, reference, "{name} i8_deltas_into");

            // The same kernel on lists that straddle the retired CAP = 96.
            let mut reference = [0i16; ACC];
            scalar::i8_deltas_into(&mut reference, &seed, &i8_weights, &long_adds, &long_subs);
            let mut got = [0i16; ACC];
            (kernels.i8_deltas_into)(&mut got, &seed, &i8_weights, &long_adds, &long_subs);
            assert_eq!(got, reference, "{name} i8_deltas_into (long lists)");

            let mut reference = pre_seed;
            scalar::l1_dot(&mut reference, &ft, &nnz, &l1_weights);
            let mut got = pre_seed;
            (kernels.l1_dot)(&mut got, &ft, &nnz, &l1_weights);
            assert_eq!(got, reference, "{name} l1_dot");

            for threat_shift in [0u32, 1, 2] {
                for scale in [&UNIT_SCALE, &FOUR_SCALE] {
                    for stm in 0..2 {
                        let mut reference = [0u8; ACC];
                        scalar::activate(&mut reference, &pst, &threat, scale, stm, threat_shift);
                        let mut got = [0u8; ACC];
                        (kernels.activate)(&mut got, &pst, &threat, scale, stm, threat_shift);
                        assert_eq!(
                            got, reference,
                            "{name} activate stm {stm} shift {threat_shift}"
                        );
                    }
                }
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

            // `l3_dot`: EXACT f32 equality, on fixtures whose partial sums
            // differ per lane so a reduction tree that pairs lanes in the
            // wrong ORDER is caught; magnitudes spread over four decades so a
            // re-association actually changes the rounding.
            for scale in [1.0f32, 1e-3, 1e3, 1e6] {
                let l3w: [f32; L3] =
                    std::array::from_fn(|j| ((j as f32 * 0.29).sin() - 0.2) * scale);
                let l3in: [f32; L3] =
                    std::array::from_fn(|j| ((j as f32 * 0.17).cos() + 0.05) / scale);
                assert_eq!(
                    (kernels.l3_dot)(&l3w, &l3in),
                    scalar::l3_dot(&l3w, &l3in),
                    "{name} l3_dot scale {scale}"
                );
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
        // CONTRACT NARROWED: the i16 activate pipeline is bit-identical to
        // the scalar reference for |threat lane| <= 32767 >> shift — the
        // domain the kernel's debug assert enforces (real nets sit far inside
        // it). Beyond it the saturating form diverges BY DESIGN, so the sweep
        // samples the guaranteed domain. pst keeps the full domain
        // (saturating add is exact post-clamp for ALL pst). A QAT net's
        // shift 0 covers the whole i16 range; the CEN2 shift 2 a quarter.
        for threat_shift in [0u32, 1, 2] {
            let bound = i32::from(i16::MAX >> threat_shift) + 1;
            for scale in [&UNIT_SCALE, &FOUR_SCALE] {
                // The HalfKA half of the same narrowing: the reduced-frame
                // accumulator represents `pst · scale` in the FT_QUANT frame,
                // so the full pst domain is the domain REPRESENTABLE AT THAT
                // SCALE — `32767 / scale`, exactly as the threat bound is
                // `32767 >> shift`. Beyond it the i16 multiply wraps, which
                // every backend does identically.
                let pst_bound = i32::from(i16::MAX) / i32::from(scale[0]) + 1;
                let mut pst = [[0i16; ACC]; 2];
                let mut threat = [[0i16; ACC]; 2];
                for perspective in 0..2 {
                    for lane in 0..ACC {
                        let raw = (lane as u16)
                            .wrapping_mul(40503)
                            .wrapping_add(perspective as u16 * 173)
                            as i16;
                        pst[perspective][lane] = (i32::from(raw) % pst_bound) as i16;
                        let raw = (lane as u16)
                            .wrapping_mul(49711)
                            .wrapping_add(perspective as u16 * 251)
                            as i16;
                        threat[perspective][lane] = (i32::from(raw) % bound) as i16;
                    }
                }
                for stm in 0..2 {
                    let mut reference = [0u8; ACC];
                    scalar::activate(&mut reference, &pst, &threat, scale, stm, threat_shift);
                    let mut dispatched = [0u8; ACC];
                    activate(&mut dispatched, &pst, &threat, scale, stm, threat_shift);
                    assert_eq!(dispatched, reference, "shift {threat_shift}");
                }
            }
        }
    }

    /// The scalar `activate` and `find_nnz` bodies were restated from the
    /// math while the SIMD backends were left byte-for-byte untouched;
    /// bitwise agreement between the scalar and EVERY available backend on
    /// randomized inputs is the proof the restatement changed nothing. (The
    /// dispatched tests above only exercise the backend the running CPU
    /// selects; this one enumerates them.)
    ///
    /// Inputs are drawn inside the pipeline's contract domain, which the
    /// kernels' own debug tripwires enforce: per-lane `|pst| <= 32767/scale`,
    /// `|threat| <= 32767 >> shift`, `scale` in the loader's 1..=255. The
    /// bound edges are planted explicitly each round.
    #[test]
    fn restated_scalar_kernels_match_every_backend_on_random_inputs() {
        // xorshift64* — deterministic.
        let mut state: u64 = 0x2026_08_09_D40_u64 | 1;
        let mut rng = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        let mut backends: Vec<(&str, &Kernels)> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
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
        }

        for round in 0..64u32 {
            let threat_shift = round % 3;
            let stm = (round as usize / 3) % 2;
            let t_bound = i32::from(i16::MAX >> threat_shift);

            let mut scale = [1u8; ACC];
            for s in scale.iter_mut() {
                *s = (rng() % 255 + 1) as u8; // loader domain 1..=255
            }
            let mut pst = [[0i16; ACC]; 2];
            let mut threat = [[0i16; ACC]; 2];
            for side in 0..2 {
                for j in 0..ACC {
                    let p_bound = 32767 / i32::from(scale[j]);
                    let p = (rng() % (2 * p_bound as u64 + 1)) as i32 - p_bound;
                    pst[side][j] = p as i16;
                    let t = (rng() % (2 * t_bound as u64 + 1)) as i32 - t_bound;
                    threat[side][j] = t as i16;
                }
                // Plant the exact contract edges on a few lanes each round.
                pst[side][0] = (32767 / i32::from(scale[0])) as i16;
                pst[side][1] = -pst[side][1].abs();
                threat[side][0] = (i16::MAX >> threat_shift) as i16;
                threat[side][1] = -((i16::MAX >> threat_shift) as i16);
                threat[side][2] = 0;
            }

            // activate: poison the two outputs differently so an unwritten
            // lane cannot compare equal.
            let mut want = [0xA7u8; ACC];
            scalar::activate(&mut want, &pst, &threat, &scale, stm, threat_shift);
            for (name, k) in &backends {
                let mut got = [0x58u8; ACC];
                (k.activate)(&mut got, &pst, &threat, &scale, stm, threat_shift);
                assert_eq!(
                    want, got,
                    "{name} activate diverges from the restated scalar \
                     (round {round}, stm {stm}, shift {threat_shift})"
                );
            }

            // find_nnz: drive it with the real activation output (its actual
            // input distribution), then with a sparsified copy so the empty /
            // near-empty group paths are exercised at every density.
            let mut sparse = want;
            for cell in sparse.iter_mut() {
                if rng() % 8 != 0 {
                    *cell = 0;
                }
            }
            for ft in [&want, &sparse] {
                let mut ref_idx = [0u16; ACC / 4];
                let ref_n = scalar::find_nnz(ft, &mut ref_idx);
                for (name, k) in &backends {
                    let mut got_idx = [0u16; ACC / 4];
                    let got_n = (k.find_nnz)(ft, &mut got_idx);
                    assert_eq!(got_n, ref_n, "{name} find_nnz count (round {round})");
                    assert_eq!(
                        got_idx[..got_n],
                        ref_idx[..ref_n],
                        "{name} find_nnz indices (round {round})"
                    );
                }
            }
        }
    }
}
