//! Fused accumulator kernels for the 768-wide SOTA feature transformers.
//!
//! Keeping the accumulator in registers while applying all feature deltas is
//! substantially cheaper than reading and writing all 768 lanes once per
//! changed feature. The scalar backend is the bit-exact reference; AVX2 is
//! selected once at startup and preserves wrapping i16 arithmetic.

use super::sota_features::{ACC, RowOffset};
use super::sota_head::{FT_QUANT, FT_SHIFT, L2, L3, MAX_THREAT_SHIFT};
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

// BOTH feature transformers fold i8 weight rows into an i16 accumulator since
// perf study 32 — HalfKA's 57.7 MB i16 table became a 28.9 MB i8 one plus a
// per-column scale the merge applies — so there is exactly one fold family and
// the threat channel and the HalfKA channel share every kernel below.
//
// In-place fold: the Finny cache's cold from-scratch build.
type ApplyI8Fn = fn(&mut [i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
// Fused copy+apply: read `src`, apply deltas, write `dst` in ONE pass — the
// lazy fold's child = parent ± deltas without a separate ~1.5 KB memmove
// (profiling-study §4.3/opt #2). `dst`/`src` may not alias.
type ApplyI8IntoFn = fn(&mut [i16; ACC], &[i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
// TWO-DESTINATION in-place apply (perf study 9 item 4): `acc ±= deltas`, and
// the result is ALSO written to `out`. The Finny cached refresh needs the same
// bytes in two places — the cache entry (its memo for the next refresh with
// this king key) and the search stack's accumulator slot — and used to get
// there by folding in place and then `memmove`ing 2×1280 i16 to the stack.
// Both destinations are written from the SAME registers the fold already
// holds, so the second home costs one store per block instead of a whole
// read+write pass. `acc` and `out` may not alias.
type ApplyI8DualFn = fn(&mut [i16; ACC], &mut [i16; ACC], &[i8], &[RowOffset], &[RowOffset]);
type L1DotFn = fn(&mut [i32; L2], &[u8; ACC], &[u16], &[i8]);
// The FT merge + pairwise activation. BOTH channels carry a net-dependent
// rescale into it, and they are independent of one another:
//
//   `scale` — the HalfKA accumulator's per-lane dequantisation multiplier
//     (perf study 32). The i8 rows are stored at `FT_QUANT / scale[j]`, so the
//     merge puts lane `j` back into the FT_QUANT frame before it clamps. A
//     WRAPPING i16 multiply in every backend: the reduced accumulator wraps
//     exactly where the old FT_QUANT-frame one did, and pinning the width is
//     what keeps the three backends bit-identical. A vector, so it stays a
//     parameter rather than becoming a const generic.
//   `threat_shift` — the NET's threat frame shift
//     (`sota_head::threat_shift_for`), not a compile-time constant: a QAT net
//     exports its threat table at scale 256 and needs shift 0 where a legacy
//     CEN2 net needs 2 (perf study 30). Each backend monomorphises over its
//     three admissible values, so the inner loop keeps exactly the instruction
//     it had — measured, §5 of that study.
type ActivateFn = fn(&mut [u8; ACC], &[[i16; ACC]; 2], &[[i16; ACC]; 2], &[u8; ACC], usize, u32);
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
// The head's final f32 L3 -> scalar dot (perf study 26 arm 3). Same fixed
// reduction SHAPE as `policy_dot` -- 16 partial lanes (lane k sums elements
// k, k+16) then a halving tree -- so every backend is bit-identical to the
// scalar reference, which is the loop that used to live in
// `SotaHead::propagate_l3`. It is here rather than autovectorized because
// LLVM would not widen it: with L3 = 32 it emitted **2-lane** `vfmadd213ps`
// over sixteen `vmovsd` pair-loads, 4.56% of `forward_into`'s instructions
// for a 32-element dot product.
type L3DotFn = fn(&[f32; L3], &[f32; L3]) -> f32;
// The SKP1 skip-lane dot (perf study §6-S2): u8 activations × bucket-major
// f32 lane. Reduction shape is DEFINED as 16 partial lanes + the halving
// tree (the policy_dot discipline) so every backend is bit-identical to the
// scalar reference. NOTE this replaced the head's serial element-order sum;
// the rounding change was gated by integer-cp identity on the 800-FEN
// battery (S2 landing commit).
type SkipDotFn = fn(&[u8], &[i8]) -> i32;

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

/// Perf study 29's TRAFFIC-PINNING probes (`probe-accpin` / `probe-wpin`).
///
/// Both keep the number and the WIDTH of every load in the kernels exactly as
/// they are and change only WHERE those loads point, so `L1-dcache-loads` is
/// invariant and the drop in misses / load-stall cycles isolates one stream's
/// share. They produce wrong evaluations by construction — these features are
/// measurement instruments, never shipped, and `bench … nodes` fixes the node
/// budget so the two arms remain comparable at the same tree size.
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
    /// built once. Cache-line alignment does not matter here: the point is
    /// that the whole working set is 2.5 KB and therefore L1-resident.
    /// TWO distinct buffers, and every handout goes through `black_box`.
    /// Without both, the probe stops being load-count-neutral: with one buffer
    /// `activate`'s `pv`/`tv` pair becomes one address and LLVM CSEs 80 of its
    /// 160 loads away, and with a constant pointer it CSEs the src stream
    /// across the HalfKA and threat folds in the same `compute`. Measured: the
    /// naive form dropped 249 loads/node and 360 instructions/node, which is a
    /// different experiment. `black_box` makes the pointer opaque, so the same
    /// loads are issued — they simply all hit the same 2.5 KB of L1.
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
    /// the fold's `src`, so the per-ply accumulator stack keeps its full write
    /// volume and its full footprint — which means it keeps evicting weight
    /// rows, and the probe under-reports whatever share of fusion's prize is
    /// capacity rather than read latency. This buffer takes the stores too.
    ///
    /// A leaked `Box` rather than a `static mut`: one allocation, `'static`
    /// by construction, and no aliasing question about the *address*. The
    /// aliasing question about the CONTENTS is real and deliberate — every
    /// fold now writes the same 2.5 KB — which is exactly why this feature
    /// computes wrong evaluations and is a measurement instrument only.
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
        // program. The engine is single-threaded under `bench`, and this
        // feature exists only to run `bench` — handing out repeated `&mut` to
        // one buffer is precisely the traffic collapse being measured, and the
        // resulting evaluations are known-wrong by design.
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

    /// Perf study 39's **SUBTRACTIVE** arm at study 31's magnitude, plus the
    /// control that says whether a pinning probe measures locality at all.
    ///
    /// `probe-actpin` pins ONLY `activate`'s four accumulator arrays — 68.9
    /// lines/node, the exact dose of study 31's additive copy — so the two
    /// directions are finally comparable at equal magnitude. `probe-accpin`
    /// pins the fold `src` stream too and therefore cannot be.
    ///
    /// **`P39_PINFILL=live` is the point of this module.** Study 29 read a
    /// −6.8% instruction drop off `accpin` and attributed it to *compiler CSE*.
    /// It is not CSE: a pin buffer full of ZEROES makes `activate` emit an
    /// all-zero `ft`, so `find_nnz` returns almost nothing and `l1_dot` — the
    /// sparse L1 layer, with its own weight rows and its own misses — does
    /// almost no work. The probe silently deletes a whole layer of the
    /// network, and per-instruction normalisation cannot repair that because
    /// the deleted instructions carry loads and stalls of their own.
    ///
    /// Under `P39_PINFILL=live` the FIRST `activate` copies the real
    /// accumulators into the pin and every later one reads that snapshot: a
    /// realistic accumulator, realistic magnitudes, realistic group sparsity,
    /// so the head does its normal work. Same binary, same code layout, one
    /// runtime branch — the difference between the two modes is the size of
    /// the confound.
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
        /// on the eval path would be ~20 cycles of pure probe artefact, and
        /// `bench` is single-threaded.
        #[inline]
        #[allow(unsafe_code)]
        pub(crate) fn take(pst: &Pair, threat: &Pair) -> (&'static Pair, &'static Pair) {
            let p = pins();
            // SAFETY: both pointers come from `Box::into_raw` on leaked `Pair`
            // allocations — non-null, aligned, valid for the whole program.
            // Handing out `&'static Pair` aliased against a transient `&mut`
            // during the one-shot fill is sound here because the fill happens
            // before either reference is handed out and `bench` is
            // single-threaded; the *contents* being shared across plies is the
            // traffic collapse this instrument exists to create, which is why
            // it computes wrong evaluations by design.
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

    /// Perf study 39's **ADDITIVE** traffic probe — study 31's instrument,
    /// rebuilt with the two controls it never had.
    ///
    /// Study 31 priced accumulator streaming by ADDING one redundant
    /// accumulator-pair copy per evaluated node and reading the cost off
    /// `ex_no_retire.load_not_complete`: **2.36 stall cycles per 64-byte
    /// line**. Study 29 priced the same quantity SUBTRACTIVELY
    /// (`probe-dstpin`) and got **0.45**. Neither ran the arms that separate
    /// the candidate explanations, so both knobs here are **runtime**:
    ///
    /// * `P39_DOSE` — copies per evaluated node (default 1). A dose series
    ///   measures whether stall cycles are LINEAR in added traffic. Study
    ///   29's "capacity cliff" reconciliation requires the cost function to be
    ///   CONVEX at our operating point; a linear dose response falsifies it.
    /// * `P39_RING` — how many distinct accumulator-pair slots the copies
    ///   rotate through, rounded up to a power of two (default 16). The
    ///   probe's own footprint is `2 × RING × 5120` bytes, so RING sweeps the
    ///   copy from L1-resident (RING = 1, 10 KB) through the real accumulator
    ///   stack's hot window (RING = 16, 160 KB) and out into L3.
    ///   **RING = 1 is the control study 31 lacked**: byte-identical code,
    ///   identical instruction and load counts, essentially no cold lines —
    ///   so `cold − hot` is the probe's LOCALITY cost with its ISSUE cost
    ///   differenced away.
    ///
    /// Runtime knobs rather than cargo features on purpose: ONE binary serves
    /// the whole additive panel, so no comparison inside it can be confounded
    /// by code layout (study 29 §8.6 needed two nulls precisely because its
    /// arms were different binaries; here they are the same bytes).
    #[cfg(feature = "probe-addcopy")]
    pub(crate) mod addcopy {
        use super::ACC;
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        /// i16 lanes in one accumulator PAIR: 2 × 1280 × 2 B = 5120 B = 80
        /// 64-byte lines. One copy therefore moves 80 read + 80 write lines =
        /// **160 lines**, which is study 31's per-evaluated-node figure
        /// exactly.
        const PAIR: usize = 2 * ACC;

        /// `src`/`dst` are leaked `Box<[i16]>` allocations kept as integers.
        /// A `&'static mut` cannot be handed out of a `&'static Rings`, and
        /// this mirrors the existing `pin_dst` idiom two screens up.
        struct Rings {
            src: usize,
            dst: usize,
            mask: usize,
            dose: usize,
            /// `P39_SRC=live` — read the REAL accumulator pair `activate` is
            /// about to consume instead of the cold ring.
            ///
            /// This exists because my ring copy generates **2.7× the L1 misses
            /// study 31's copy did** for the same geometric line count (1.30
            /// per line against 0.477), and a calibration against known
            /// traffic (`tools/perf39/calib39.sh`) puts a fully-cold line at
            /// ~1.2 counter events — i.e. my ring lines are ~100% cold and
            /// study 31's were ~40%. The obvious way to be 40% cold is for the
            /// copy's READ half to touch memory the engine was about to touch
            /// anyway, which prefetches `activate` rather than adding traffic.
            /// This arm tests that reconstruction directly rather than
            /// asserting it.
            live_src: bool,
            /// `P39_SITE=fold` — issue the copies from the FullThreats fold
            /// instead of from `activate`.
            ///
            /// The recommendation this study makes turns on whether "stall
            /// cycles per line" is a property of the TRAFFIC or of where in
            /// the node it happens to be issued. Fusion removes accumulator
            /// traffic from the fold; the probe adds it in `activate`. The
            /// fold fires 0.8538 times per node against `activate`'s 0.4311,
            /// so **fold dose 1 (136.6 lines/node) and activate dose 2 (137.95)
            /// are the same traffic volume in two different places** — and if
            /// they price the same, the number transfers.
            fold_site: bool,
        }

        static RINGS: OnceLock<Rings> = OnceLock::new();
        /// Plain relaxed load/store, never a `fetch_add`: an atomic RMW is a
        /// locked op worth ~20 cycles and would be a large fraction of what
        /// this probe is trying to measure. `bench` is single-threaded, so a
        /// non-atomic increment is all the ordering this needs.
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

        /// One dose = one accumulator-pair-sized vector copy through the ring.
        ///
        /// `wrapping_add(bias)` with an opaque zero rather than
        /// `copy_from_slice`: the latter lowers to `memcpy`, which for 5 KB on
        /// Zen 4 is `rep movsb` — a different microarchitectural animal from
        /// the `vmovdqu64` streaming this probe is pricing (see study 38 on
        /// `rep movsb`). The add form autovectorises to the same
        /// load/`vpaddw`/store shape the fold kernels use.
        /// The `activate` call site — the default, and study 31's placement.
        #[inline]
        pub(crate) fn tick(live: &[[i16; ACC]; 2]) {
            let r = rings();
            if !r.fold_site {
                copies(r, Some(live));
            }
        }

        /// The FullThreats-fold call site (`P39_SITE=fold`), which is where the
        /// traffic fusion actually removes lives.
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

/// Perf study 61 — the ceiling probe, **split by channel**.
///
/// Study 51 measured `probe-wpin` (all four fold sites pinned to row 0) at
/// −13.19 % cycles and called it "the whole prize for the fold's weight rows".
/// It is one number for two tables with completely different shapes:
/// `hka_weights` is 28.8 MB / 22,528 rows and `thr_weights` is 77.7 MB /
/// 60,720 rows, and the threat channel folds 4.594 rows per fold against
/// HalfKA's 2.279. A prize that is 80 % one channel and 20 % the other is a
/// different engineering problem from a 50/50 split, and nothing in studies
/// 29–60 says which it is.
///
/// `probe-wpin-hka` pins only the three HalfKA sites; `probe-wpin-thr` pins
/// only `apply_threat_deltas_into`. Both are strict subsets of `probe-wpin`,
/// so `Δhka + Δthr ≈ Δboth` is a built-in consistency check on the pair.
/// Measurement only — they compute wrong evaluations by construction.
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
    #[cfg(feature = "foldcount")]
    {
        super::foldcount::HKA_INPLACE.hit(adds.len() + subs.len());
        #[cfg(feature = "rowprobe")]
        {
            super::foldcount::note_row_reuse(0, adds);
            super::foldcount::note_row_reuse(0, subs);
        }
    }
    pin_rows_hka!(adds, subs);
    kernel_dispatch!(i8_deltas(acc, weights, adds, subs))
}

/// Two-destination i8-weight fold — the HalfKA Finny cache's warm refresh.
/// `acc` and `out` must not alias. See [`ApplyI8DualFn`] for why both homes
/// are written from the fold's own registers.
#[inline]
// `probe-accpin` shadows the accumulator parameters with its pinned
// buffer, which is exactly what the probe is for; the originals then
// go unread. No effect on any build without that feature.
#[cfg_attr(feature = "probe-accpin", allow(unused_variables))]
pub(crate) fn apply_halfka_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    #[cfg(feature = "foldcount")]
    {
        super::foldcount::HKA_INTO.hit(adds.len() + subs.len());
        super::foldcount::note_hka_shape(adds.len(), subs.len());
        #[cfg(feature = "rowprobe")]
        {
            let z = &super::foldcount::ZERO_ROW_HKA;
            super::foldcount::note_zero_rows(z, weights, adds, ACC);
            super::foldcount::note_zero_rows(z, weights, subs, ACC);
            super::foldcount::note_row_fold(0, adds);
            super::foldcount::note_row_fold(0, subs);
            super::foldcount::note_row_reuse(0, adds);
            super::foldcount::note_row_reuse(0, subs);
        }
    }
    pin_rows_hka!(adds, subs);
    pin_src!(src);
    pin_dst!(dst);
    kernel_dispatch!(i8_deltas_into(dst, src, weights, adds, subs))
}

/// Two-destination in-place apply: `acc ±= deltas`, and the same result is
/// written to `out` as well (perf study 9 item 4). See [`ApplyI8DualFn`].
/// `acc` and `out` must not alias.
#[inline]
#[cfg_attr(feature = "probe-dstpin", allow(unused_variables))]
pub(crate) fn apply_halfka_deltas_dual(
    acc: &mut [i16; ACC],
    out: &mut [i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    #[cfg(feature = "foldcount")]
    {
        super::foldcount::HKA_DUAL.hit(adds.len() + subs.len());
        super::foldcount::note_hka_dual_shape(adds.len() + subs.len());
        #[cfg(feature = "rowprobe")]
        {
            super::foldcount::note_row_reuse(0, adds);
            super::foldcount::note_row_reuse(0, subs);
        }
    }
    // Perf study 43 arm NT stores `acc` non-temporally, which the ISA only
    // defines for a 64-byte-aligned address. The alignment is structural
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
// `probe-accpin` shadows the accumulator parameters with its pinned
// buffer, which is exactly what the probe is for; the originals then
// go unread. No effect on any build without that feature.
#[cfg_attr(feature = "probe-accpin", allow(unused_variables))]
pub(crate) fn apply_threat_deltas_into(
    dst: &mut [i16; ACC],
    src: &[i16; ACC],
    weights: &[i8],
    adds: &[RowOffset],
    subs: &[RowOffset],
) {
    #[cfg(feature = "foldcount")]
    {
        super::foldcount::THR_INTO.hit(adds.len() + subs.len());
        super::foldcount::note_thr_shape(adds.len() + subs.len());
        super::foldcount::note_thr_pair(adds.len(), subs.len());
        #[cfg(feature = "rowprobe")]
        {
            let z = &super::foldcount::ZERO_ROW_THR;
            super::foldcount::note_zero_rows(z, weights, adds, ACC);
            super::foldcount::note_zero_rows(z, weights, subs, ACC);
            super::foldcount::note_row_fold(1, adds);
            super::foldcount::note_row_fold(1, subs);
            super::foldcount::note_row_reuse(1, adds);
            super::foldcount::note_row_reuse(1, subs);
        }
    }
    // Perf study 39's placement arm: the same copies, issued from the fold
    // rather than from `activate` (`P39_SITE=fold`). Inert otherwise.
    #[cfg(feature = "probe-addcopy")]
    probe::addcopy::tick_fold();
    pin_rows_thr!(adds, subs);
    pin_src!(src);
    pin_dst!(dst);
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
// `probe-accpin` shadows the accumulator parameters with its pinned
// buffer, which is exactly what the probe is for; the originals then
// go unread. No effect on any build without that feature.
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
    #[cfg(feature = "foldcount")]
    super::foldcount::ACTIVATE.hit(0);
    // Perf study 39's ADDITIVE arm rides here — one evaluated node, one
    // redundant accumulator-pair copy, which is study 31's instrument.
    #[cfg(feature = "probe-addcopy")]
    probe::addcopy::tick(pst);
    // `activate` is the single largest accumulator-STREAM reader in the
    // engine: four `[i16; ACC]` arrays, 10.24 KB per evaluation, which is the
    // exact quantity Stockfish carries as 4.10 KB by fusing the two channels.
    //
    // `probe-actpin` (study 39) pins ONLY these four arrays — 68.9 lines/node,
    // the exact magnitude of study 31's additive copy, in the subtractive
    // direction. `probe-accpin` additionally pins the fold `src` stream.
    #[cfg(feature = "probe-accpin")]
    let (pst, threat) = (
        std::hint::black_box(&*probe::PIN_ACC),
        std::hint::black_box(&*probe::PIN_ACC2),
    );
    #[cfg(all(feature = "probe-actpin", not(feature = "probe-accpin")))]
    let (pst, threat) = probe::livepin::take(pst, threat);
    // The HalfKA rescale is a WRAPPING i16 multiply in every backend, which is
    // what keeps the three bit-identical (see the avx512 merge). It wraps
    // exactly where the pre-study-32 FT_QUANT-frame accumulator wrapped, so it
    // is no *new* hazard — but unlike that one it is a hazard nobody would
    // notice, because a wrap turns a saturating lane into a near-zero one and
    // the clamp hides it. The threat channel carries the same style of bound
    // check on its own shift (`|t| <= 8191`); this is its twin, and it is the
    // tripwire that would fire if a future net's accumulator range grew.
    //
    // Skipped under `probe-accpin`/`probe-actpin`: those features deliberately
    // redirect the accumulators to a pinned buffer and compute wrong evals by
    // design, so the bound is not their contract.
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

    #[cfg_attr(feature = "split58", inline(never))]
    pub(super) fn activate(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
        stm: usize,
        threat_shift: u32,
    ) {
        const H: usize = ACC / 2;
        for flip in 0..2 {
            let perspective = stm ^ flip;
            for lane in 0..H {
                // Both rescales, and they are independent. THREAT:
                // FT_QUANT/threat_quant as a power of two — 4 for a CEN2 net,
                // 1 for a QAT net at scale 256. HALFKA: the lane put back into
                // the FT_QUANT frame by its own per-column i8 scale.
                // `wrapping_mul` in i16 is the DEFINED semantics, not a
                // convenience: the vector backends multiply in i16 lanes, so
                // widening first here would disagree with them on the (never
                // observed) overflow — exactly where the old FT_QUANT-frame
                // accumulator wrapped too. (See the avx512 twin.)
                let merge = |p: i16, t: i16, s: u8| {
                    (i32::from(p.wrapping_mul(i16::from(s))) + (i32::from(t) << threat_shift))
                        .clamp(0, FT_QUANT)
                };
                let left = merge(
                    pst[perspective][lane],
                    threat[perspective][lane],
                    scale[lane],
                );
                let right = merge(
                    pst[perspective][lane + H],
                    threat[perspective][lane + H],
                    scale[lane + H],
                );
                out[lane + flip * H] = ((left * right) >> FT_SHIFT) as u8;
            }
        }
    }

    #[cfg_attr(feature = "split58", inline(never))]
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

    /// The bit-exact reference for the head's L3 dot — this IS the loop that
    /// lived in `SotaHead::propagate_l3` until perf study 26 arm 3, moved here
    /// unchanged so the SIMD backends have something to be identical to.
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
                        // Perf study 50: the scale is u8 on disk and in
                        // memory, so the 128-bit i16 load becomes a 64-bit
                        // byte load plus a zero-extend. Same op count, half
                        // the bytes; `_mm_cvtepu8_epi16` is exact for the
                        // 1..=255 range the loader guarantees.
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
        //
        // PERF STUDY 57 DELETED THE §6-F HOIST HERE, and the deletion is the
        // point rather than a tidy-up. This kernel used to carry a
        // `[MaybeUninit<usize>; 96]` pair plus a chunking loop whose ONLY job
        // was to compute `feature * ACC` once instead of once per 16-lane
        // block. Now that the row list arrives pre-scaled as [`RowOffset`],
        // there is nothing left to hoist: the array, the two `assume_init`
        // reads, the chunk cursor and the `source` re-point are all dead, and
        // removing them takes the last `MaybeUninit` out of this backend.
        // Bit-identical: same rows, same order (adds then subs), same wrapping
        // i16 lane arithmetic — the chunked form folded the same multiset in
        // the same sequence, it just re-entered the block loop every 96 rows.
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

    /// The last three steps of the reference halving tree, in registers:
    /// `sums[k] += sums[k+4]`, then `+= sums[k+2]`, then `+= sums[1]`.
    /// Shared by both f32 dots and by both SIMD backends so the four of them
    /// cannot drift from the scalar reference or from each other.
    ///
    /// SAFETY-free: `#[target_feature]` is not needed on a pure AVX shuffle
    /// chain that is only ever reached from an AVX2/AVX-512 caller, but it is
    /// marked anyway so the caller's feature set propagates and it inlines.
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
    /// kernels. Every defined `L1` (768/1280/1536/2048) satisfies it; the point
    /// is that a future width cannot silently leave the tail of `ft` UNWRITTEN.
    /// That went from cosmetic to load-bearing with perf study 15 arm 4: `ft` is
    /// now a reused buffer, so an unwritten lane is the PREVIOUS position's
    /// activation rather than a deterministic zero (D33 reviewer A, A4).
    const _: () = assert!(ACC.is_multiple_of(2 * LANES));

    /// zmm blocks in one accumulator (40 at the shipping ACC = 1280). NOT
    /// `BLOCKS` — `l1_dot_inner` has a local const of that name meaning
    /// `L2 / L2_BLK`, and one module must not spell two things the same way.
    const ACC_BLOCKS: usize = ACC / LANES;

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
    /// Exact divisors of `ACC_BLOCKS` only: a non-dividing tile pushes the
    /// remainder into the one-block tail loop, which costs more control than it
    /// saves. 1280 -> 20 (2 tiles), 768 -> 12 (2), 1536 -> 16 (3), 2048 -> 16 (4).
    /// Measured: 20 is the optimum at 1280 and the largest divisor that fits.
    /// `REGS = 40` (one tile, no tile loop at all) was measured too and is a
    /// **disaster** — 6581.5 -> 7589.2 i/n — because 40 accumulators spill.
    ///
    /// # Perf study 44 — the cap is an ARM, because 20 was chosen on the wrong counter
    ///
    /// Everything above was adjudicated on `instructions:u`, whose noise floor
    /// on this box is exactly 0.0 — which is precisely why a narrower tile was
    /// never tried: more tiles is unambiguously more loop control, so it loses
    /// on instructions by construction. Study 20 is the standing counterexample
    /// (−5.99% CYCLES with instructions UP 0.19%), and study 44's back-end
    /// census supplies the mechanism that would make this one of those cases:
    /// **`de_dis_dispatch_token_stalls1.store_queue_rsrc_stall` is 502.6
    /// cycles/node, the largest single dispatch-blocking resource in the
    /// engine** — larger than the load queue (442.1). A `REGS = 20` tile ends
    /// with 20 `zmm` stores, and Zen 4 splits each into two 32-byte store ops
    /// (study 39 §1: 2.043 store dispatches per line written), so every tile
    /// closes with a **40-store burst** into a 64-entry store queue.
    ///
    /// Narrowing the cap interleaves those stores with the next tile's loads
    /// instead of bunching them. It is bit-identical at every setting: the tile
    /// walk covers the same blocks in the same order with the same arithmetic,
    /// and no row is read more or fewer times (register `k` consumes its own
    /// distinct 32 bytes of each row either way).
    ///
    /// ## MEASURED, and the cap is now 5 (perf study 44 §4.5)
    ///
    /// 32 paired rounds, balanced Latin square, byte-identical null riding in
    /// the rotation, node- and eval-hash-identical at every setting. Contrasts
    /// are arm-minus-null on `cycclean` (`cycles` − SMT theft):
    ///
    /// | cap | `store_queue_rsrc_stall` | cycles contrast | `instructions:u` |
    /// |---|---:|---:|---:|
    /// | 20 (was) | — | — | — |
    /// | 10 | −9.24% (30/32) | −0.33% (0.8 se) — **null** | +1.261% |
    /// | 8 | −19.74% (9/9) | +0.18% — **null**, unexplained | +2.305% |
    /// | **5 (now)** | **−19.70% (32/32, p < 0.0001)** | **−1.10% (2.2 se)** | **+4.076%** |
    ///
    /// The null read 16/32 on cycles — a coin flip — and exactly 0.000% on
    /// instructions, every round.
    ///
    /// **`instructions:u` rises 4.08% and cycles still fall.** That is the
    /// whole reason this was never found: eight prior studies of this kernel
    /// adjudicated on a counter that can only ever prefer the widest tile.
    ///
    /// **Measured at ACC = 1280 ONLY.** The cap applies at every width (768 and
    /// 1536 and 2048 all fall to `REGS = 4`, their largest divisor ≤ 5), and the
    /// `const _` asserts below still machine-check that the tile divides the
    /// accumulator exactly — but the *benefit* is measured at one width.
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
        // Down to 1, not to 8: 1 divides everything, so the `const _` assert below
        // can never fire on a future width whose BLOCKS has no divisor in 9..=20
        // (e.g. ACC 800 -> 25). That case degrades to the per-block loop rather
        // than breaking the build (review A finding 3).
        while r > 1 {
            if ACC_BLOCKS.is_multiple_of(r) {
                return r;
            }
            r -= 1;
        }
        1
    }
    const REGS: usize = pick_regs();
    // The win depends on the tile dividing the accumulator EXACTLY: a remainder
    // falls into the one-block `while off < ACC` tail, whose control cost per
    // block is what the wide tile exists to remove. True for every shipped
    // width — 1280/20, 768/12, 1536/16, 2048/16 all give remainder 0.
    const _: () = assert!(
        ACC_BLOCKS.is_multiple_of(REGS),
        "fold tile must divide the accumulator exactly"
    );
    // One zmm is live beyond the accumulators (the row being folded in).
    const _: () = assert!(REGS < 32, "fold tile must leave a spare zmm");

    /// Lanes per tile, and the number of tiles in one accumulator.
    ///
    /// Perf study 25. Every fold kernel used to end its tile walk with a
    /// one-block `while off < ACC` remainder, under a comment saying that
    /// remainder is DEAD — `pick_regs` returns an exact divisor of
    /// `ACC_BLOCKS`, pinned by the `const _` assert above. It really is dead,
    /// at every width, and `TILES * TILE == ACC` below is the machine-checked
    /// statement of that. Deleting the four of them is bit-identical: same
    /// blocks, same order, same arithmetic.
    ///
    /// **Measured, because the obvious form was the wrong one.** Rewriting the
    /// walk as `for t in 0..TILES { let off = t * TILE; … }` — trip count
    /// compile-time, which reads like the strictly better shape — cost **+9.6
    /// instructions/node**, and `for _ in 0..TILES` with an accumulated offset
    /// still cost **+7.9** with branches up 3.6/node: LLVM declines to unroll
    /// the counted loop and keeps a real induction variable. Leaving the walk
    /// as `while off + TILE <= ACC` and removing ONLY the remainder measured
    /// **−3.2 i/n**, because with nothing after the loop LLVM unrolls both
    /// tiles outright. All three are node- and eval-hash-identical; the
    /// counter has a 0.0 noise floor, so those signs are real.
    const TILE: usize = LANES * REGS;
    const TILES: usize = ACC / TILE;
    const _: () = assert!(TILES * TILE == ACC, "the tile walk must cover ACC exactly");

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

    /// Two-destination i8 fold — the HalfKA Finny warm
    /// refresh, both homes written from the fold's own registers.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn i8_deltas_dual_inner(
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
                    // Perf study 43 arm NT. The two destinations have opposite
                    // reuse distances and only one of them wants a cache line:
                    //
                    //  * `out` is `stack[ply].acc[p]`, which the head reads
                    //    within the same node -- it must stay a normal store;
                    //  * `acc` is the FINNY CACHE ENTRY, whose next read is the
                    //    next refresh at this king key. Measured: 0.16472
                    //    refreshes/node spread over all 128 slots, so a hot slot
                    //    comes round roughly every ~350 nodes, and in 350 nodes
                    //    the engine streams ~3 MB of weight rows through a 1 MB
                    //    L2. The line cannot survive, so keeping it costs an RFO
                    //    now (6.6 lines/node) and evicts weight rows for nothing.
                    //
                    // `_mm512_stream_si512` REQUIRES 64-byte alignment; that is
                    // guaranteed structurally, not by luck -- `FinnyEntry` is
                    // `#[repr(C, align(64))]` and `acc` sits at offset 128
                    // (`by_piece` 96 + `psqt` 32) -- and the wrapper
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

    /// The fold with **compile-time row counts** — perf study 25 arm 1, moved
    /// onto i8 rows by perf study 32.
    ///
    /// `NA`/`NS` are the exact lengths of `adds`/`subs`, so both row loops
    /// unroll away completely: no loop control, no unroll remainder, no
    /// empty-list clones, and `feature * ACC` is computed ONCE for the whole
    /// call instead of once per tile. What is left is straight-line
    /// `vpmovsxbw` / `vpaddw` / `vpsubw` with no branch of any kind.
    ///
    /// This is NOT perf study 12 §4's rejected offset hoist. That one built a
    /// `[MaybeUninit<usize>; 96]` for a *runtime*-length list and paid 91
    /// instructions of prologue to save 4 multiplies. Here the array is
    /// `[usize; NA]` with `NA <= 2` known at compile time: it lives in two
    /// registers, the "loop" filling it is two `lea`+`shl` pairs, and there is
    /// no prologue at all.
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
        // Perf study 25 arm 1 — dispatch on the SHAPE of the delta set.
        //
        // `apply_dirty_directed` (`sota_accumulator.rs`) builds both lists in
        // `[usize; 2]` arrays, so a HalfKA fold is ALWAYS one of four shapes,
        // and the comment there pins the lower bound: every real move emits at
        // least one add and one sub (a null move pushes no entry at all). LLVM
        // cannot see any of that through `&[usize]`, so it emitted a 6231-byte
        // function — the row loops unrolled x2 with an odd/even remainder
        // branch, cloned across the empty-list cases, 480 `vpaddw`/`vpsubw` in
        // 24 copies of the tile body. Measured on that code: 11.16% of all
        // engine cycles for 6.93% of instructions (fast build, d16 frame).
        //
        // The four shaped instantiations are branch-free straight lines. The
        // generic fallback stays for the threat channel's longer feature lists
        // and for the EMPTY case, which must remain a plain copy — see the
        // "stale-child bug" note at `sota_accumulator.rs`'s forward fold.
        // ## Perf study 61 — the ONE-SIDED and THREE-ROW shapes. **LANDED.**
        //
        // The four arms above were chosen in study 25, when this kernel was
        // HalfKA-ONLY: `apply_dirty_directed` builds both lists in `[usize; 2]`
        // and every real move emits at least one add and one sub, so `(1,1)`,
        // `(1,2)`, `(2,1)`, `(2,2)` is a COMPLETE cover of HalfKA and the
        // fallthrough was genuinely dead. Study 32 quantised HalfKA to i8,
        // deleted the i16 family, and routed **the entire FullThreats channel**
        // through this same dispatch — without revisiting the arm list, and
        // without anyone counting what the new traffic's shapes are.
        //
        // Counted here for the first time (`foldcount::THR_PAIR`, 4,270,334
        // threat folds in the study frame):
        //
        // | `(adds, subs)` | folds | share | path |
        // |---|---:|---:|---|
        // | `(0,1)` | 444,368 | **10.41 %** | generic |
        // | `(1,0)` | 371,467 | **8.70 %** | generic |
        // | `(0,2)` | 168,840 | 3.95 % | generic |
        // | `(2,0)` | 84,674 | 1.98 % | generic |
        // | the four study-25 arms | 1,164,466 | 27.27 % | shaped |
        // | everything else | 2,036,519 | 47.69 % | generic |
        //
        // **19.11 % of every threat fold is a SINGLE weight row, and it runs the
        // runtime-length kernel** — two loops whose trip counts LLVM cannot see,
        // an x2 unroll with an odd/even remainder branch, and an empty-list test,
        // eight times over (once per tile), to apply one row.
        //
        // A one-sided list is exactly the case the generic form was never for.
        // Adding the four one-sided shapes takes straight-line coverage from
        // 27.27 % to **52.31 %** of threat folds. `i8_into_shaped` already
        // handles `NS = 0` correctly — `srow` is `[usize; 0]`, the `zip` fills
        // nothing and the `for &base in &srow` loop compiles away entirely — so
        // this is a dispatch change and not a kernel change.
        //
        // **Bit-identical by construction, at the strongest level:** the shaped
        // and generic kernels walk the same tiles in the same order, read the
        // same rows in the same order (adds then subs), and perform the same
        // per-lane wrapping i16 add/sub. The gate checks it anyway.
        //
        // The cost is `.text`: eleven more instantiations of a straight-line
        // tile body, measured at **+8,048 B total** (714,725 → 722,773), which
        // is far less than the naive estimate because LLVM shares the tile
        // prologue and epilogue across instantiations.
        //
        // **Measured together, 32 rounds, blocked Latin square, byte-identical
        // null in the rotation, node total 5,001,562 on every run:**
        //
        // | arm | coverage of threat folds | `instructions:u` | `branches:u` |
        // |---|---:|---:|---:|
        // | study 25 (was) | 27.27 % | — | — |
        // | + one-sided | 52.31 % | **−0.438 % (−24.7/node)** | **−0.509 %** |
        // | + three-row (**shipped**) | **68.40 %** | **−1.219 % (−68.9/node)** | **−2.274 %** |
        //
        // Both counters read 32/32 at **se 0.000** against a null of exactly
        // −0.000 %. `cycles:u` was NOT resolvable — a GPU training run shared
        // the box and the null moved further than the arm — so this is landed
        // on the exact counters and its wall-clock value is not established.
        //
        // **The win per converted fold GROWS with the shape**, which is the
        // useful generalisation: −115 instructions per one-sided fold against
        // −322 per three-row fold, because the generic kernel pays its loop
        // control per row per tile. **31.60 % of threat folds are still
        // generic** (both lists ≥ 4, or one ≥ 5) and the next study should
        // either continue the table or fix the generic kernel's loop control,
        // which is now the cheaper-looking of the two.
        //
        // `noshape61` restores the study-25 dispatch so this contrast stays
        // measurable on the shipped tree.
        unsafe {
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
                // The THREE-row family. `(3,0) (0,3) (1,3) (3,1) (2,3) (3,2)
                // (3,3)` is a further 686,947 folds = 16.09 % of the channel,
                // taking coverage 52.31 % → 68.40 %. It was measured as its own
                // arm (`shape61x`) precisely because a body up to six rows wide
                // is where `.text` cost could plausibly overtake the win — and
                // it does not: it MORE than doubles the win (−24.7 → −68.9 i/n)
                // for +3,088 B.
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
            i8_deltas_into_generic(dst, src, weights, adds, subs)
        }
    }

    /// The runtime-length i8 fold — **the whole FullThreats channel**, plus the
    /// tests' long lists and the EMPTY-list contract (both lists empty must copy
    /// `src` to `dst` verbatim, which the in-place twins never have to do).
    ///
    /// # It is not cold, and it stopped being cold in perf study 32
    ///
    /// This carried `#[cold]` + `#[inline(never)]` from perf study 25 arm 1,
    /// where it was **true**: the fold family was still split by weight type,
    /// HalfKA folded `i16` rows through the shaped dispatch above and the threat
    /// channel had its own `i8` kernel, so nothing but tests and the empty case
    /// reached this function. Study 32 quantised HalfKA to `i8` and deleted the
    /// `i16` family — "both feature transformers now fold i8 rows" — which
    /// routed every threat fold into the *shared* kernel, out through the shape
    /// dispatch (threat lists average 4.658 rows, so they miss all four shapes)
    /// and into here. The attribute came across with the code. Perf study 36
    /// then measured the result: **12.38% of engine cycles in a function
    /// labelled cold.**
    ///
    /// Perf study 37 measured what the label cost. `#[inline(never)]` was
    /// **provably inert** — the binary is byte-identical with it removed,
    /// because `cold` already forbids inlining — and `#[cold]` was worth
    /// −0.744 instructions and −0.744 branches per node, entirely from block
    /// placement in the caller: it laid the generic call out as the unlikely
    /// arm of the shape dispatch, so the one path that is actually taken took a
    /// jump the fallthrough now covers.
    ///
    /// # ⚠ DO NOT "TIDY" THE TWO ROW LOOPS INTO ONE — that arm was built and it LOST
    ///
    /// Perf study 37 arm 3 walked the add and sub lists TOGETHER for their
    /// common prefix (`split_at(adds.len().min(subs.len()))`, then one paired
    /// loop plus whichever tail survives), on the reasoning that two cursor
    /// bumps, two compares and two back-edges per pair become one of each. The
    /// per-row control here really is **6 instructions per row per tile**
    /// (`mov`, `add`, `lea`, `shl`, `cmp`, `jne` — read off the shipped
    /// disassembly, not estimated), i.e. ~47.9 i/n and ~8.0 br/n at 3.993
    /// rows/node over 2 tiles, so the target was real.
    ///
    /// It is bit-identical — wrapping `i16` add/sub is a commutative group
    /// operation per lane, so only the multiset of signed rows matters — and it
    /// measured **+0.147% instructions, +0.243% branches and +0.331% L1d
    /// loads**, node- and hash-identical, 12 paired rounds, against a null of
    /// exactly zero on the first two. A LOSS, three times over.
    ///
    /// **Why**, and this is the part worth keeping: the two-loop form is
    /// 3616 bytes because **LLVM clones the whole tile body across the
    /// empty-list cases and unrolls both tiles**, and that cloning is worth more
    /// than the control it duplicates. The paired form collapsed to 1744 bytes
    /// (405 instructions from 714, one tile body instead of the clones) —
    /// tidier, smaller, and slower. Turning two loops into three also gives the
    /// common `na == ns` case two empty-tail entry tests it did not have.
    ///
    /// The lesson generalises to anything else proposed here: in this kernel the
    /// specialised, duplicated shape IS the optimisation, and a restructuring
    /// that lets LLVM re-merge the paths pays for it. Measure before believing
    /// a control-count argument.
    ///
    /// ## ⚠ PERF STUDY 58 RE-PRICED IT AT `TILES = 8` AND IT LOST HARDER
    ///
    /// Study 37 §9 item 4 left this open in writing — *"whether the paired arm
    /// would still lose at a different `REGS`/`TILES`… I did not test another
    /// width and the banner does not claim I did"* — and study 44 then took
    /// `REGS_CAP` to 5, so the walk this arm attacks became **four times
    /// wider**. That is the exact shape that made study 57 arm A worth −41 i/n
    /// after study 37 had priced it at −8, so it was rebuilt and re-measured
    /// against post-57 main on the 5,001,562-node frame:
    ///
    /// | arm | `instructions:u` | `branches:u` |
    /// |---|---:|---:|
    /// | study 37, `TILES = 2` | +0.147% | +0.243% |
    /// | **study 58, `TILES = 8`** | **+0.290% (+16.96 i/n)** | **+0.566% (+3.35 br/n)** |
    ///
    /// Node-identical, 3 paired rounds, every round the same sign, against a
    /// counter whose null on this box is 0.000. **The width was not the
    /// reason.** At eight tiles there are no unrolled tile bodies left to
    /// clone, so study 37's stated mechanism does not even apply here — and the
    /// arm is twice as bad. What actually costs is turning two row loops into
    /// three: each gets its own entry test and its own unroll prologue, per
    /// tile, and at `TILES = 8` that multiplies by four as well.
    ///
    /// The same study built the neighbouring idea — split the row lists ONCE
    /// outside the tile walk into compile-time 4-row groups plus a ≤3 tail
    /// (`as_chunks::<4>`), to stop LLVM re-deciding its unroll prologue inside
    /// the tile loop (~5 instructions per list per tile, ~80 per call, read off
    /// the disassembly). That measured **+1.083% instructions (+63.35 i/n) and
    /// +16.24 br/n** — the largest single loser of the study. Two loops became
    /// four, and LLVM's own scaffolding is cheaper than any hand-split.
    ///
    /// **The rule, now paid for three times: in this kernel a loop-control
    /// instruction count is not a prize, it is an estimate of what LLVM would
    /// charge you for taking the loop away from it.**
    ///
    /// # ⚠ THE `vpmovsxbw` : `vpaddw` 1:1 RATIO IS A HARDWARE FLOOR (perf study 47)
    ///
    /// The disassembly of this function is **450 `vpmovsxbw` against 450
    /// `vpaddw`/`vpsubw`** — the i8→i16 widening is half of every vector op it
    /// issues, and at 7.1754 fold rows/node it is **287 `vpmovsxbw`/node, 22.3%
    /// of every 512-bit op in the engine**. Study 47 measured the box to find
    /// out whether that is reducible. It is not, and the number is:
    ///
    /// > the convert pipes run at **32 i16 lanes/cycle** (measured, and measured
    /// > to be the SAME at 256 bits — Zen 4's double-pump is uniform), and the
    /// > fold must widen 7.1754 rows × 1280 lanes = **9184 lanes/node**
    /// > ⇒ **287 cycles/node = 6.87% of the frame, irreducible at this layout.**
    ///
    /// Four alternatives were measured or counted, none of them argued:
    ///
    /// * **64-B `vmovdqu64` + two register `vpmovsxbw`**: 3 ops per 64 B against
    ///   2. The fused `vpmovsxbw zmm,[mem]` form already costs exactly what the
    ///   register form does (both 1.005 c/op) — the load rides free.
    /// * **ALU-domain sign extension** (`vpsllw`+`vpsraw`, or mask/shift/sub):
    ///   16–21 lanes/cycle against 32. Shifts and logicals are 1-pipe here too.
    /// * **`vpmaddubsw` with a {1,0} byte selector**: identical op count, and it
    ///   scrambles lane order for nothing.
    /// * **Summing rows in the i8 domain before one widen**: incorrect. Two i8
    ///   weights sum to [−255, 254], outside i8.
    ///
    /// The only real win is **i16 weight rows**, measured at −0.195 c/chunk-update
    /// (−1.34% of frame) — and it doubles the traffic of a kernel that study 44
    /// measured at **86–88% load-stall**, i.e. it re-pays study 32's measured
    /// −6.8% cycles to buy back a third of it. **Do not re-propose it.**
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
        // Perf study 12 §4 — the offset hoist + CAP chunking is gone; see the
        // shaped kernel above for the full rationale.
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
        // PERF STUDY 21: `CHAINS` independent accumulator sets.
        //
        // The group-outer order above is right, but at the SHIPPING width it
        // leaves one dependency chain. `L2 = 16` ⇒ `BLOCKS = 1`, so every one
        // of the ~39 active groups accumulated into the same `zmm`, and
        // `vpdpbusd` on Zen 4 is ~4-cycle latency at ~1-per-cycle throughput:
        // 39 x 4 = 156 cycles of dependency for 39 cycles of work. Widths that
        // already carry >= 4 blocks have enough independent chains of their
        // own and are left byte-for-byte as they were.
        //
        // ## PERF STUDY 46: 4 IS THE RIGHT NUMBER, AND IT IS NOW MEASURED
        //
        // Study 21 DERIVED 4 from a cited latency x throughput. Study 46's
        // commission proposed that 4 was only half the truth -- "2 issue ports
        // against a 4-5 cycle latency needs 8-10 chains" -- so the constant was
        // measured directly on the box (generated inline asm, chain-count
        // sweep, min-of-5, `tools/perf46/gen_ilp.py`). Cycles per op:
        //
        //   chains        1      2      3      4      5      8     10
        //   vpdpbusd zmm  4.007  2.002  1.338  1.004  1.002  1.002  1.002
        //   vpdpbusd ymm  4.005  2.003  1.335  1.002  0.802  0.538  0.527
        //
        // 512-bit latency is 4 cycles and 512-bit throughput is 1.00 op/cycle
        // -- ONE effective FMA/VNNI pipe, because Zen 4 double-pumps 512-bit
        // ops over a 256-bit datapath. The knee is exactly 4 and the curve is
        // FLAT from 4 to 10, so raising `CHAINS` above 4 buys provably zero.
        // The ymm row is the control that proves the mechanism rather than
        // asserting it: the same instruction at 256 bits runs 2/cycle and its
        // knee moves to 8, at the same 4-cycle latency.
        //
        // DO NOT re-propose 8 chains here. If a future arch raises `L2` so
        // that `BLOCKS >= 4`, the gate above already switches `CHAINS` to 1 for
        // the same reason -- the blocks are themselves independent chains.
        //
        // Study 46 also tested spreading the `as_chunks::<CHAINS>()` remainder
        // across chains instead of folding all of it into `sums[0]` below
        // (bit-identical, since i32 wrapping adds are order-independent). It
        // measured WORSE -- 96.81 vs 92.95 cycles/call at the typical count of
        // 39 -- so the shipped shape stays. The tail is not a defect.
        //
        // Bit-identical, for the reason the paragraph above already states:
        // i32 wrapping adds are order-independent, so partitioning the groups
        // across chains and summing the partials at the end is the same value.
        // SAFETY: active groups address four FT bytes and L2 complete
        // four-byte output rows in loader-validated storage.
        unsafe {
            const BLOCKS: usize = L2 / L2_BLK;
            const CHAINS: usize = if BLOCKS >= 4 { 1 } else { 4 / BLOCKS };
            let mut sums = [[_mm512_setzero_si512(); BLOCKS]; CHAINS];
            for (blk, sum) in sums[0].iter_mut().enumerate() {
                *sum = _mm512_loadu_si512(pre.as_ptr().add(blk * L2_BLK).cast());
            }
            // NOTE: the study's "prefetch the next group's rows" was tried
            // here and measured noise-to-negative (−0.3% over 4 interleaved
            // pairs): the per-bucket L1 table is ~160 KB and L2-resident, so
            // the loads never miss far enough to hide. Perf study 21 re-ran
            // the same idea on the FT weight rows with a proper
            // wrong-address control and measured +0.01% — the address does
            // not matter, so do not re-propose row prefetching here.
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

    /// # ⚠ THIS LOOP IS AT ITS ISSUE FLOOR — and the reason generalises (study 47)
    ///
    /// The emitted lane-iteration is **15 512-bit ops** (two 6-op merges plus a
    /// 3-op tail), 40 iterations per eval, **258.7 ops/node = 20.1% of every
    /// 512-bit op in the engine**. Measured against its own emitted shape it
    /// costs **0.586 c/op** — the isolated per-op sum predicts 0.889, so this
    /// mix already overlaps the pipes at a factor of 0.659, far better than the
    /// fold kernel's 0.873. There is no arrangement win left here.
    ///
    /// **★ THE RULE THAT KILLS THE OBVIOUS IDEAS: on Zen 4 a folded memory
    /// operand costs exactly ONE issue slot — the same as the register-form op**
    /// (`vpmullw` reg 1.006, `vpmullw` mem 1.006; a bare `vmovdqu64` is 1.077).
    /// Both loads that CAN fold here already do, read off the shipped
    /// disassembly:
    ///
    /// ```text
    /// vmovdqu64 (%rax,%r14,2),%zmm2       ; pst   load
    /// vpsllw    $0x1,(%r8,%r14,2),%zmm3   ; threat load FOLDED into the shift
    /// vpmullw   (%rbx,%r14,2),%zmm2,%zmm2 ; scale  load FOLDED into the multiply
    /// ```
    ///
    /// So **CSE-ing a folded load STRICTLY LOSES**: the `scale` vectors are each
    /// re-loaded once per perspective, and hoisting them turns
    /// `2 × vpmullw[mem]` into `1 × vmovdqu64 + 2 × vpmullw[reg]` — three ops for
    /// two. (This also retro-explains perf study 46 §3.2, which measured
    /// spreading `l1_dot`'s tail as WORSE and could not say why.)
    ///
    /// Four more candidates, all closed by measurement rather than by argument:
    ///
    /// * **`vpaddsw` → `vpaddw`** would genuinely halve that op (1.003 → 0.518 —
    ///   saturation costs 2× on this box, which nobody knew). It is **NOT
    ///   bit-safe**: `vpmullw` is wrapping *by design* (see the study-32 note
    ///   below), so the saturating add really can fire, and wrapping instead
    ///   flips those lanes to the opposite clamp.
    /// * **`vpsllw $1` → `vpaddw t,t`** (1.005 → 0.518) loses: the shift carries
    ///   a folded load and the add cannot, so it costs 1.588 against 1.005.
    /// * **`vpsllw $7` + `vpmulhuw` → `vpmullw` + `vpsrlw`**: two 1-pipe ops
    ///   either way.
    /// * **Pre-shifting the threat accumulator to delete `vpsllw`** is
    ///   *backwards*: it moves work from here (34.5 merges/node) into the fold
    ///   (287 chunk-updates/node), which runs 8× more often. **Never move work
    ///   from `activate` to the fold.**
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn activate_inner<const SHIFT: u32>(
        out: &mut [u8; ACC],
        pst: &[[i16; ACC]; 2],
        threat: &[[i16; ACC]; 2],
        scale: &[u8; ACC],
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
            // ★ PERF STUDY 48 — LANE OUTER, PERSPECTIVE INNER, AND THE SCALE
            // IN A REGISTER.
            //
            // `scale` is the one array here that does NOT depend on the
            // perspective: both merges read the SAME 2560 bytes at the SAME
            // offsets. With the perspective loop outside, the whole array was
            // therefore streamed TWICE per evaluation — 40 redundant `zmm`
            // loads, and on Zen 4 a 512-bit load is double-pumped into two
            // dispatches (study 29 §2.1, measured 2.0015), so **80 redundant
            // load dispatches per eval = 34.5 per node = 1.5 % of every load
            // the engine issues**.
            //
            // Interchanging the loops alone would only trade those loads for
            // an equal number of `vmovdqu64`s, which is the trade perf study 47
            // §5.2 priced and (correctly, for the shape it considered)
            // rejected: hoisting a FOLDED memory operand costs an issue slot,
            // so `2 × vpmullw[mem]` → `1 × vmovdqu64 + 2 × vpmullw[reg]` is
            // three ops for two.
            //
            // **`vpmullw` is commutative, and that is what makes this strictly
            // better rather than a wash.** Exactly one of the multiply's two
            // operands can be memory. Study 47 assumed it had to be the scale;
            // it does not. Put the REUSED operand (the scale) in a register and
            // fold the per-perspective accumulator load into the multiply
            // instead, and the separate `vmovdqu64` for `pst` disappears:
            //
            // | per 32-lane group (4 merges) | ops | 512-bit loads |
            // |---|---:|---:|
            // | before: 4 × (`vmovdqu64` pst + `vpsllw`[threat] + `vpmullw`[scale]) | 12 | 12 |
            // | after: 2 × `vmovdqu64` scale + 4 × (`vpmullw`[pst] + `vpsllw`[threat]) | 10 | 10 |
            //
            // So it removes an op AND a load per merge pair — the only strict
            // improvement of the two, and the reason it is not the arm study 47
            // closed. Bit-identical: every output byte is the same expression
            // over the same inputs; only the loop order and the operand that
            // happens to live in a register change.
            for lane in (0..H).step_by(32) {
                // ★ PERF STUDY 50 — the scale is u8, so this is a 32-byte
                // load zero-extended into 32 i16 lanes instead of a 64-byte
                // load. `vpmovzxbw` takes its 256-bit memory operand in the
                // SAME single issue slot a `vmovdqu64` takes (study 47's
                // table: every memory-operand op costs ~1 c), so the op count
                // is unchanged and only the BYTES move: 2560 -> 1280 per
                // evaluation, and the whole array drops from 40 cache lines
                // to 20. On Zen 4 a 512-bit load is double-pumped into two
                // dispatches and a 256-bit load into one (study 29 §2.1,
                // measured 2.0015), so it is also -40 load dispatches/eval.
                let sv_left =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256(scale.as_ptr().add(lane).cast()));
                let sv_right =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256(scale.as_ptr().add(lane + H).cast()));
                for flip in 0..2 {
                    let perspective = stm ^ flip;
                    let p = &pst[perspective];
                    let t = &threat[perspective];
                    // The threat rescale, as ONE SHIFT.
                    //
                    // The two FT transformers are quantised differently —
                    // HalfKA at FT_QUANT = 255, FullThreats at the NET's own
                    // `threat_quant` — so the merge used to carry a six-op
                    // `round(t · 255/64)`: abs / shift / mulhrs /
                    // compare-mask / masked-sub / add. That was 24 of this
                    // loop body's 48 instructions and ~2.2% of ALL engine
                    // instructions, where Stockfish — whose two feature sets
                    // fold into ONE 1024-wide accumulator, not two — merges the
                    // threat channel with a single
                    // `vec_add_16(acc, vec_convert_8_16(column))`
                    // (`eca43a97:src/nnue/nnue_accumulator.cpp:237-238`).
                    //
                    // The ratio does not have to be exact: it is a GAIN on one
                    // eval channel, so it is carried as the nearest power of
                    // two, `256/threat_quant` (`sota_head::threat_shift_for`).
                    // 255/64 = 3.984375 → 4 is a 0.39% systematic change; at a
                    // QAT net's threat_quant 256, 255/256 = 0.996 → 1 is the
                    // SAME 0.39% in the other direction, i.e. shift 0.
                    //
                    // **The shift is a const generic and not a parameter for a
                    // measured reason** (perf study 30 §5.3). With an immediate,
                    // LLVM folds the accumulator load INTO the shift —
                    // `vpsllw $0x2,(%rcx,%r8,2),%zmm3`. `vpsllw`'s
                    // register-count form takes its memory operand as the
                    // COUNT, not the data, so a runtime shift forces a separate
                    // `vmovdqu64` per merge and costs the unroll: measured
                    // **+34.48 instructions per node, +0.61%**, branches
                    // unchanged. Monomorphising over the three legal shifts and
                    // branching once per eval gives it back (+2.77 i/n).
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
                    // The HalfKA rescale, as ONE MULTIPLY (perf study 32).
                    //
                    // The HalfKA table is stored i8 with a per-accumulator-lane
                    // integer scale, so the accumulator lives in the reduced
                    // frame `FT_QUANT / scale[j]` and has to be put back before
                    // the clamp. `vpmullw` keeps the product in i16 lanes: the
                    // result is the value the old FT_QUANT-frame accumulator
                    // held, so it wraps exactly where that one wrapped, and the
                    // scalar and avx2 backends pin the same width to agree.
                    // The threat channel's `<< 2` is unchanged.
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
        // bias; per i one broadcast + two fused mul-adds (perf study #4 item
        // #2a: 4096 FMAs/eval at D47 dims, previously autovectorized).
        // j-lane order preserved and i-sequential per lane ⇒ bit-identical
        // to the scalar reference. SAFETY: all loads/stores cover in-bounds
        // 16-lane blocks of loader-validated storage.
        //
        // ## PERF STUDY 46: this IS an ILP deficit, and it was PRICED and
        // ## REJECTED -- do not rebuild it without re-reading the number.
        //
        // Each accumulator carries a 16-deep FMA chain, and study 46 measured
        // Zen 4's 512-bit FMA at 4-cycle latency / 1.00 op-per-cycle:
        //
        //   throughput floor (32 FMAs)        32 cycles
        //   latency bound at 2 chains         64 cycles   <-- what this is
        //   latency bound at 4 chains         32 cycles
        //
        // So this kernel runs at 2x its own throughput floor, exactly as study
        // 21 §9.3 item 5 suspected. A 4-accumulator version measured -17.9
        // cycles/call in isolation (`tools/perf46/kern.rs`; 8 accumulators
        // measured WORSE than 4, re-confirming the knee).
        //
        // It is rejected on ARITHMETIC, not on difficulty. At the measured
        // 0.43117 evals/node the prize is -17.9 x 0.43117 = -7.7 cycles/node =
        // -0.185% of engine cycles -- below the paired panel's own resolution
        // (se ~0.26% at n = 40), and it is NOT eval-bit-identical: a
        // left-to-right float sum IS a serial dependency chain, so any
        // reassociation changes the rounding. Landing it would move the eval
        // hash and d13 pins and would have to change the scalar reference (the
        // bit-exact contract), this AVX-512 backend AND the AVX2 backend
        // below -- which carries the identical 2x deficit for the same reason
        // (4 ymm accumulators over 4 DISJOINT output slices, each still 16
        // deep, against a ymm knee of 8).
        //
        // Re-price it if `L2` ever grows: the chain deepens linearly with L2,
        // so at the D47 `L2 = 128` this deficit would be ~8x larger.
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

    /// The 32-lane iota — the full-`zmm` `vpcompressw` source (perf study 21).
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
        // PERF STUDY 21: THIRTY-TWO groups per iteration, not sixteen.
        //
        // `vpcompressw` is a full-`zmm` instruction — it compacts 32 u16 lanes
        // under a `__mmask32`. The kernel above used only the 256-bit form, so
        // it paid 20 iterations of test / kmov / popcnt / broadcast / add /
        // store where 10 do the same work. The mask for a 32-group iteration
        // is just the two 16-bit dword-test masks concatenated, and the two
        // 64 B loads are independent.
        //
        // This also halves the length of the serial `count` chain, which is
        // the loop's only loop-carried dependency (each store's address
        // depends on every previous popcount).
        //
        // Bit-identical: same ascending index set, same count. The old
        // 16-wide iota and the AVX2 movemask+LUT form remain the reference.
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
        // so the whole L3 = 32 dot is two `vfmadd231ps` and a four-step
        // halving tree — against the sixteen 2-lane `vfmadd213ps` and sixteen
        // `vmovsd` pair-loads LLVM produced for the autovectorized source
        // (perf study 26 arm 3).
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
        // `vpdpbusd` is the whole kernel: 64 u8 x i8 products accumulated into
        // 16 i32 lanes per instruction. 1280 lanes = 20 loads + 20 multiply-adds
        // where the f32 form needed 80 x (widen, convert, fma).
        //
        // PERF STUDY 21: FOUR accumulators, not one. `vpdpbusd` on Zen 4 is
        // ~4-cycle latency at ~1-per-cycle throughput, so twenty of them
        // chained into a single register cost 20 x 4 = 80 cycles of pure
        // dependency for 20 cycles of work. Four independent chains put four
        // in flight and bring that to ~20 + 4. The loads were never the
        // problem; the accumulator was.
        //
        // Bit-identical, and this is a stated contract rather than an
        // accident: the scalar reference's own doc fixes the reduction SHAPE
        // (16 partial lanes + a halving tree) and leaves each backend free to
        // reach it in any order, because i32 addition is associative. Every
        // shipped ACC (768/1280/1536/2048) is a multiple of 256, so the 4-wide
        // body covers the whole vector; the tail loop is kept for a future
        // width, not because it runs today.
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

    /// The identity HalfKA scale (perf study 32): the kernel tests drive the
    /// merge with weights that already sit in the FT_QUANT frame, so a scale of
    /// 1 keeps them testing exactly what they tested before.
    const UNIT_SCALE: [u8; ACC] = [1; ACC];

    /// A non-unit HalfKA scale, so the backends' `vpmullw` is actually
    /// exercised against the scalar `wrapping_mul` rather than multiplied by
    /// one. 4 is the modal column scale in the canonical net.
    const FOUR_SCALE: [u8; ACC] = [4; ACC];

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
        // Every shift a container can ask for (perf study 30): 2 is what every
        // CEN2 net means and what the backends used to hardcode, 0 is a QAT net
        // at threat_quant 256, 1 the 128 midpoint. The extremes planted above
        // are |t| = 1760, so `t << 2` = 7040 stays inside the i16 bound.
        // Crossed with study 32's HalfKA scale, so the two rescales are
        // exercised together rather than one at a time.
        for threat_shift in [0u32, 1, 2] {
            for scale in [&UNIT_SCALE, &FOUR_SCALE] {
                for stm in 0..2 {
                    // POISONED with two DIFFERENT fills, not zeroed (D33 reviewer
                    // C, T3). ~85-95% of activations are zero after the clamp, so
                    // with both buffers starting at 0 a lane that NEITHER kernel
                    // wrote compared equal and the test passed. Perf study 15 arm
                    // 4 made `ft` a reused buffer, so an unwritten lane is now a
                    // stale activation from the previous position rather than a
                    // harmless zero — this test is what says every backend writes
                    // all of it.
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
        //
        // STALE FIXTURE, fixed by perf study 25: these were 768-lane vectors,
        // hardcoded from the D47V era. At the shipping ACC = 1280 that
        // exercised a width the engine never passes — and, worse, 768 is not a
        // multiple of 256, so the fixture drove the avx512 backend's 64-wide
        // TAIL loop while the real call drives only its 256-wide four-chain
        // body. The four-accumulator split (study 21 arm B2) was therefore
        // going untested against scalar on the one path that ships.
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
        apply_threat_deltas_into(&mut dispatched, &seed, &i8_weights, &adds, &subs);
        assert_eq!(dispatched, reference);
    }

    #[test]
    fn dispatched_i8_deltas_match_scalar_on_long_feature_lists() {
        // The wrapping-extremes test above runs 5 adds / 4 subs, which is the
        // incremental-update shape. The Finny fold is the other caller and it
        // arrives with feature lists an order of magnitude longer, where the
        // per-lane i16 wrapping has many more chances to diverge from the
        // scalar reference — and where the `_into_` kernels chunk their
        // hoisted offsets at 96, so any future hoist here (perf study 3's
        // ranked item 10, prototyped on perf/i16-deltas-hoist) must stay
        // bit-exact across that boundary. Sizes below straddle it.
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

    /// Perf study 25 arm 1 gave `i8_deltas_into` four **shape-specialised**
    /// instantiations — `(adds.len(), subs.len())` ∈ {1,2}² — because
    /// `apply_dirty_directed` builds both lists in `[usize; 2]`. Those are the
    /// shapes the engine actually executes, and until this test NOTHING covered
    /// them: every other kernel test in this module runs 4-5 rows or 200+, all
    /// of which land in the generic fallback.
    ///
    /// Mutation it catches: a transposed `(NA, NS)` arm in the dispatch (which
    /// would apply a sub as an add — an eval error of exactly two weight rows,
    /// invisible to a node count and to every shape-agnostic test), a shaped
    /// body that tiles short, or a dropped `return` that ran the generic pass a
    /// second time on top of the shaped one.
    ///
    /// The empty shape is in the table on purpose: `sota_accumulator.rs`'s
    /// forward fold deletes its own copy special-case and relies on
    /// `apply_threat_deltas_into(dst, src, w, &[], &[])` being `dst = src` lane for
    /// lane. A fast-path `return` on empty lists would be a silent stale-child
    /// bug in release, and this asserts the copy behaviour directly.
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
                apply_threat_deltas_into(&mut dispatched, &src, &weights, a, s);
                assert_eq!(dispatched, reference, "shape ({na}, {ns})");
                if na == 0 && ns == 0 {
                    assert_eq!(dispatched, src, "the empty shape must be a pure copy");
                }
            }
        }
    }

    /// The GENERIC fold — the threat channel's kernel — over the full 15×15
    /// grid of `(adds.len(), subs.len())`, against the scalar reference.
    ///
    /// Written for perf study 37's rejected pairing arm (see the ⚠ banner on
    /// `i8_deltas_into_generic`) and kept after it was reverted, because the
    /// coverage it adds is real and was missing: the module's other kernel tests
    /// run either ≤2 rows (the shaped dispatch) or 200+ (one long list), so
    /// **no test walked the short, UNEQUAL lengths that are the threat
    /// channel's actual working range** — 18.66% of threat folds are exactly one
    /// row, 59.10% are four or fewer, and the mean is 4.658.
    ///
    /// Mutation it catches: any future restructuring of the two row loops that
    /// mis-slices the lists, drops the longer side's tail, or applies a sub as
    /// an add. The index pools are disjoint and the weight table is dense, so a
    /// dropped or mis-signed row cannot cancel coincidentally.
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
                apply_threat_deltas_into(&mut folded, &src, &weights, a, s);
                assert_eq!(folded, reference, "lengths ({na}, {ns})");
            }
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
        // LONG lists, run through EVERY backend (perf study 12, review C
        // finding 9). The `_into` kernels used to chunk their feature lists at
        // CAP = 96 and re-read `dst` as the source for each later chunk; the
        // AVX-512 pair now applies every row in one pass. That equivalence is
        // exactly what a >96-row list tests, and until now the only long-list
        // test drove the DISPATCHED backend, so on any given box one of the two
        // shapes went unexercised. These straddle the old boundary in both
        // directions.
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

            // `l3_dot` (perf study 26 arm 3). EXACT f32 equality, on fixtures
            // whose partial sums differ per lane so a reduction tree that
            // pairs the lanes in the wrong ORDER is caught — associativity is
            // the whole risk when a scalar halving loop becomes a shuffle
            // chain, and magnitudes are spread over four decades so a
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
        // CONTRACT NARROWED (study-2 option D): the i16 activate pipeline is
        // bit-identical to the scalar reference for |threat lane| <= 8191 —
        // the domain the debug assert in the kernel enforces (measured real
        // nets: max |t| = 431, a 19x margin). Beyond it the saturating 4t
        // form diverges BY DESIGN; the old full-i16-domain sweep therefore
        // now samples the guaranteed domain. pst keeps the full domain
        // (saturating add is exact post-clamp for ALL pst).
        //
        // The domain is `32767 >> threat_shift`, so a QAT net's shift 0 covers
        // the WHOLE i16 range and the CEN2 shift 2 covers a quarter of it
        // (perf study 30).
        for threat_shift in [0u32, 1, 2] {
            let bound = i32::from(i16::MAX >> threat_shift) + 1;
            for scale in [&UNIT_SCALE, &FOUR_SCALE] {
                // The HalfKA half of the same narrowing (perf study 32). The
                // reduced-frame accumulator represents `pst · scale` in the
                // FT_QUANT frame, so "the full pst domain" is now the full
                // domain REPRESENTABLE AT THAT SCALE — `32767 / scale`,
                // exactly as the threat bound above is `32767 >> shift`.
                // Beyond it the i16 multiply wraps, which every backend does
                // identically and the kernel's debug tripwire flags.
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
}
