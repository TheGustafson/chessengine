//! Transparent-huge-page hinting for the big, randomly-addressed tables.
//!
//! # Why this exists
//!
//! The two feature-transformer weight tables are **28.8 MB (HalfKA) and
//! 77.7 MB (FullThreats)**, and the engine reads them the worst possible way:
//! perf study 36's exact fold census puts **7.131 weight rows per node**, each
//! row 1280 contiguous bytes at a *data-dependent* feature index, i.e. a
//! uniformly scattered 1280-byte pull out of 106 MB, ~7 times per node.
//!
//! A 4 KiB page holds 3.2 such rows and a row straddles a page boundary 31% of
//! the time, so each row costs ~1.31 page touches and almost every one of them
//! is a page this core has not touched recently. **Zen 4's L2 STLB is 3072
//! entries: 12 MB of reach at 4 KiB, against a 106 MB working set.** The
//! translation therefore misses to a page walk on essentially every row.
//!
//! At 2 MiB the same 106 MB is 53 entries. That is the whole idea.
//!
//! This is `design/08` §3.13 phase 2's plan for the transposition table and
//! perf study 4 item #8 ("if `madvise`, add `madvise(MADV_HUGEPAGE)` on the
//! big allocations"), which was left **RE-OPENED** and never built. It changes
//! no value the engine computes — it is a hint to the kernel about how to back
//! pages that already exist — so every accumulator, every eval and every node
//! count is bit-identical by construction.
//!
//! # Why an `extern "C"` block and not the `libc` crate
//!
//! The crate has **zero runtime dependencies** and that is a standing rule.
//! `madvise(2)` is one libc symbol, and `std` already links libc on every
//! `*-linux-gnu` target, so declaring the one prototype ourselves costs nothing
//! and adds no dependency. This is the ordinary Rust idiom for a single syscall
//! wrapper in a dependency-free crate; the `libc` crate's value is its ~10,000
//! *other* declarations and its cross-platform constant tables, neither of
//! which we need for one advice flag.
//!
//! # Failure is not an error
//!
//! `madvise` returns `EINVAL` when THP is compiled out or the range is not
//! anonymous memory, and the kernel is free to ignore the hint when memory is
//! too fragmented to produce a 2 MiB block. All of that is *fine*: the table
//! still works, just on 4 KiB pages, which is exactly today's behaviour. So
//! [`advise_huge`] reports success as a `bool` for tests and never panics.

/// The x86-64 huge-page size, and the alignment [`advise_huge`] wants its
/// range to have. A THP-backed region must contain whole 2 MiB-aligned,
/// 2 MiB-sized chunks; a range that merely *overlaps* one gets 4 KiB pages for
/// the partial chunks at both ends.
pub const HUGE: usize = 2 * 1024 * 1024;

/// `MADV_HUGEPAGE` from `include/uapi/asm-generic/mman-common.h`. Stable
/// kernel ABI since 2.6.38 and identical on every architecture Linux supports
/// (the arch-specific `MADV_*` values start at 100).
#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

// `madvise` is NOT a safe FFI item and is deliberately not declared `safe`:
// the general syscall takes advice values such as `MADV_DONTNEED`, which
// zeroes anonymous memory and so can invalidate live objects. What is safe is
// this module's *use* of it — one advice value, one range — and that is what
// the `unsafe` block in [`advise_huge`] argues, in the crate's documented
// style for its `deny(unsafe_code)` exceptions.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
unsafe extern "C" {
    /// `int madvise(void *addr, size_t length, int advice)`.
    fn madvise(addr: usize, length: usize, advice: i32) -> i32;
}

/// Ask the kernel to back `[addr, addr + len)` with 2 MiB pages.
///
/// `addr` is taken as an integer rather than a pointer deliberately: nothing
/// on this side dereferences it, the callee is the kernel, and passing an
/// address keeps every byte of pointer arithmetic at the call site in safe
/// code.
///
/// Returns whether the kernel accepted the hint. Call this **before** the range
/// is written: a fault on a fresh madvised range allocates a huge page
/// directly, whereas a range advised after it is populated has to wait for
/// `khugepaged` to collapse it in the background, which may never happen under
/// memory pressure.
///
/// Not a promise: `true` means the advice was recorded, not that any particular
/// page ended up huge. `/proc/self/smaps_rollup`'s `AnonHugePages` is the
/// observation; a page-walk counter is the effect.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
pub fn advise_huge(addr: usize, len: usize) -> bool {
    // SAFETY: `MADV_HUGEPAGE` is a pure backing-store hint — it neither reads,
    // writes, unmaps nor discards the range, so no Rust value can observe a
    // change through it, whatever `addr`/`len` are. A range that is unmapped,
    // misaligned or not anonymous is rejected by the kernel with `EINVAL` and
    // reported here as `false`; there is no address for which this call is
    // undefined behaviour. That is the whole argument, and it is why the
    // wrapper takes `usize`: it has no pointer preconditions to state.
    len >= HUGE && unsafe { madvise(addr, len, MADV_HUGEPAGE) } == 0
}

/// Everywhere else the hint does not exist, and 4 KiB pages are what the
/// tables get — which is what every platform got before this module.
#[cfg(not(target_os = "linux"))]
pub fn advise_huge(_addr: usize, _len: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hint is accepted for a real, page-aligned anonymous range. This is
    /// a live syscall test, not a mock: if the constant is wrong (say a future
    /// `MADV_*` renumbering) `madvise` returns `EINVAL` and this fails, which
    /// is the only way a wrong flag number would ever be noticed — a bad
    /// advice value is silently ignored by every other layer.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_2mib_aligned_range_accepts_the_hint() {
        // Own the memory for the whole call, and align a 2 MiB window inside it.
        let buf = vec![0u8; 4 * HUGE];
        let base = buf.as_ptr() as usize;
        let addr = base.next_multiple_of(HUGE);
        assert!(
            advise_huge(addr, 2 * HUGE),
            "madvise(MADV_HUGEPAGE) rejected a 4 MiB aligned anonymous range"
        );
        // Touch the range so the buffer cannot be optimised away before the
        // syscall, and so a wrong length would fault here rather than silently.
        assert_eq!(buf[buf.len() - 1], 0);
    }

    /// Ranges below one huge page are declined without a syscall — there is no
    /// 2 MiB-aligned chunk inside them to back, so the advice could only cost
    /// `khugepaged` scan time.
    #[test]
    fn a_range_smaller_than_one_huge_page_is_declined() {
        let buf = vec![0u8; 4096];
        assert!(!advise_huge(buf.as_ptr() as usize, buf.len()));
    }
}
