//! Cache-line-aligned heap slices for the NNUE weight tables.
//!
//! # Why this type exists
//!
//! Every SIMD kernel in `sota_simd` reads its weights as full 64-byte `zmm`
//! loads out of a feature-major table: row `f` of an `ACC`-wide table starts at
//! element `f · ACC`, and `ACC · size_of::<T>()` is a multiple of 64 for every
//! shipped width. So the alignment of *every* weight load in the engine is
//! decided by ONE thing — the alignment of the table's first element.
//!
//! A `Box<[i16]>` built by `collect()` has alignment 2. In practice these
//! tables are tens of megabytes, glibc serves them with `mmap`, and an mmap'd
//! chunk hands back `page + 16` — so the base lands at **16 mod 64** and every
//! `zmm` load in the fold and matmul kernels straddles a cache line.
//!
//! Measured on the shipped D48X net (perf study 20 §5, 5,001,562-node frame):
//! **1608 cacheline-crossing loads per node**, of which the
//! `ls_misal_loads.ma64` profile put **81.6% inside four kernels** —
//! `i16_deltas_into` 30.3%, `forward_into` 20.5%, `i16_deltas_dual` 16.1%,
//! `i8_deltas_into` 14.6%. Those are exactly the four that read weight tables.
//!
//! # How the alignment is obtained
//!
//! No allocator API. `Vec::with_capacity` reserves the buffer up front,
//! `<*const T>::align_offset` (a safe method) reports how many elements to skip
//! to reach a 64-byte boundary, and the payload is then written after exactly
//! that many padding elements. Growing a `Vec` never moves it while it stays
//! inside its reserved capacity, so the offset measured before the fill is
//! still valid after it — which the `payload_is_line_aligned` test checks
//! directly rather than by argument.
//!
//! The one trap worth naming: **`into_boxed_slice()` must not be called on the
//! backing vector.** It shrinks-to-fit, and a realloc is free to return a
//! differently-aligned block, silently undoing the whole point. That is why
//! the payload is kept in a `Vec<T>` and handed out through `Deref`.
//!
//! # The one invariant: `OFF_LE_LEN`
//!
//! **`self.off <= self.buf.len()`, always.**
//!
//! `Deref`/`DerefMut` rely on it to hand out `buf[off..]` *without* a range
//! check, which is the difference between "a weight table" and "a bounds-checked
//! range slice re-derived on every single access" — perf study 25 §4 found the
//! latter was what its head-block arm was actually fighting, and perf study 26
//! §2 measured the fix at **−8.30 instructions and −2.98 branches per node,
//! crate-wide**, against an exactly-zero rebuild null.
//!
//! Why it cannot be violated, in three parts, none of which is an argument
//! about the caller:
//!
//! 1. **Both fields are private and this module is the whole surface.**
//!    `Aligned` has no public constructor other than [`Aligned::from_iter_len`]
//!    and the two `From`/`FromIterator` wrappers that call it, so `off` is only
//!    ever written in one place.
//! 2. **That place establishes it by construction.** `buf.resize(off, …)`
//!    makes `buf.len() == off` exactly, and `buf.extend(iter)` can only grow it.
//!    There is no path on which `off` is chosen after the fill, and none on
//!    which `buf` shrinks — `Aligned` exposes no `truncate`, `pop`, `clear` or
//!    `into_boxed_slice`, and `DerefMut` hands out `&mut [T]`, which cannot
//!    change a `Vec`'s length.
//! 3. **Debug builds re-check it on every access.** `Aligned::debug_check` runs
//!    in front of both derefs, so the entire test suite — and every debug run of
//!    the engine, which touches these tables millions of times per search —
//!    asserts the invariant at the exact point the release build assumes it.
//!    A new constructor that got it wrong could not survive one `cargo test`.

use std::fmt;
use std::ops::{Deref, DerefMut};

/// The alignment every weight table is pinned to — one x86-64 cache line, and
/// also the width of one `zmm` load.
pub const LINE: usize = 64;

/// A heap slice of `T` whose first element is [`LINE`]-aligned.
///
/// Derefs to `[T]`, so it is a drop-in for the `Box<[T]>` it replaces:
/// `&table[a..b]`, `table.len()`, `table.is_empty()` and passing `&table` to a
/// `&[T]` parameter all work unchanged through deref coercion.
pub struct Aligned<T> {
    /// `[padding …][payload …]`. Never shrunk — see the module docs.
    buf: Vec<T>,
    /// Index of the first payload element; `buf.as_ptr().add(off)` is
    /// `LINE`-aligned.
    off: usize,
}

impl<T: Default + Clone> Aligned<T> {
    /// Collect `iter` into a cache-line-aligned buffer.
    ///
    /// `len` is the exact element count to expect; it sizes the one allocation
    /// so the pointer measured up front is the pointer the payload ends up on.
    /// A shorter iterator simply yields a shorter payload (the loader's own
    /// slice bounds are what enforce section sizes — this type does not change
    /// the error model).
    pub fn from_iter_len(len: usize, iter: impl Iterator<Item = T>) -> Self {
        // Worst-case padding: one element short of a whole line. `size_of::<T>()`
        // divides LINE for every weight element type we use (i8/i16/i32/f32),
        // so this bound is exact rather than merely sufficient.
        let pad = LINE / size_of::<T>();
        let mut buf: Vec<T> = Vec::with_capacity(len + pad);
        // Safe pointer method: how many ELEMENTS from the start until a LINE
        // boundary. `usize::MAX` means "impossible", which cannot happen for a
        // type whose size divides LINE, but fall back to 0 rather than panic —
        // a mis-aligned table is slow, not wrong.
        let off = match buf.as_ptr().align_offset(LINE) {
            usize::MAX => 0,
            n => n,
        };
        debug_assert!(off < pad, "align_offset exceeded the reserved padding");
        buf.resize(off, T::default());
        buf.extend(iter);
        debug_assert!(
            buf.len() <= buf.capacity(),
            "the fill must not have reallocated"
        );
        Self { buf, off }
    }
}

impl<T> Aligned<T> {
    /// The debug-build half of [`OFF_LE_LEN`](self#the-one-invariant-off_le_len):
    /// every `Deref`/`DerefMut` re-checks the invariant the release build
    /// assumes. Compiled out of release by `debug_assert!`, so the whole test
    /// suite (and every debug run of the engine) exercises the check on every
    /// single weight-table access — which is a far denser test of the invariant
    /// than any unit test could be.
    #[inline]
    fn debug_check(off: usize, len: usize) {
        debug_assert!(
            off <= len,
            "Aligned: OFF_LE_LEN violated ({off} > {len}) — a constructor did \
             not resize-then-extend"
        );
    }
}

impl<T: Default + Clone> FromIterator<T> for Aligned<T> {
    /// Convenience for call sites that already have a sized iterator; uses the
    /// iterator's lower size hint to pick the capacity.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let it = iter.into_iter();
        let len = it.size_hint().0;
        Self::from_iter_len(len, it)
    }
}

impl<T: Default + Clone> From<Vec<T>> for Aligned<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_iter_len(v.len(), v.into_iter())
    }
}

// The crate denies `unsafe_code`; this is the documented, measured exception
// (perf study 26 §2). Both blocks below are the same two lines of pointer
// arithmetic `&self.buf[self.off..]` already performs, minus the range check
// that [`OFF_LE_LEN`] proves can never fire.
#[allow(unsafe_code)]
impl<T> Deref for Aligned<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        Self::debug_check(self.off, self.buf.len());
        // SAFETY: OFF_LE_LEN — `self.off <= self.buf.len()`, so `off..` is a
        // valid range for `buf`. `get_unchecked` on a `Vec` autoderefs to the
        // slice method, i.e. this is `&buf_slice[off..]` with the bounds check
        // elided, not an allocator or provenance trick.
        unsafe { self.buf.get_unchecked(self.off..) }
    }
}

#[allow(unsafe_code)]
impl<T> DerefMut for Aligned<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        Self::debug_check(self.off, self.buf.len());
        // SAFETY: OFF_LE_LEN, as for `deref`. `&mut [T]` cannot change the
        // `Vec`'s length, so no caller can invalidate the invariant either.
        unsafe { self.buf.get_unchecked_mut(self.off..) }
    }
}

/// Payload equality — padding is invisible, so two tables with the same
/// contents compare equal whatever the allocator handed them.
impl<T: PartialEq> PartialEq for Aligned<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

/// Length only. The `Box<[T]>` this replaces printed every element, which for a
/// 138 MB net is not a debug aid but a hang; `SotaNetwork` derives `Debug`, so
/// the compact form is the useful one.
impl<T> fmt::Debug for Aligned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Aligned<{}>[{}]", std::any::type_name::<T>(), self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point, checked on the property rather than on the reasoning:
    /// the payload pointer is line-aligned for every element type the loader
    /// uses, at sizes on both sides of glibc's mmap threshold.
    #[test]
    fn payload_is_line_aligned() {
        fn check<T: Default + Clone + Copy>(name: &str, one: T) {
            for len in [1usize, 7, 64, 1000, 65_536, 1_000_003] {
                let a: Aligned<T> = Aligned::from_iter_len(len, std::iter::repeat_n(one, len));
                assert_eq!(a.len(), len, "{name} len {len}");
                assert_eq!(
                    a.as_ptr().align_offset(LINE),
                    0,
                    "{name} payload not {LINE}-aligned at len {len}"
                );
            }
        }
        check::<i8>("i8", 1);
        check::<i16>("i16", 1);
        check::<i32>("i32", 1);
        check::<f32>("f32", 1.0);
    }

    /// Contents survive the padding trick — an off-by-one in `off` would show
    /// up here as a shifted or truncated payload.
    #[test]
    fn contents_round_trip() {
        let a: Aligned<i16> = (0..1000i16).collect();
        assert_eq!(a.len(), 1000);
        assert!(a.iter().copied().eq(0..1000i16));
        assert_eq!(&a[10..13], &[10, 11, 12]);
    }

    /// `DerefMut` writes reach the payload, not the padding.
    #[test]
    fn mutation_reaches_the_payload() {
        let mut a: Aligned<i32> = Aligned::from_iter_len(8, 0..8);
        a.fill(7);
        assert_eq!(&a[..], &[7i32; 8]);
    }

    /// An empty table (a net with no threats channel / no PSQT section) is
    /// still a valid, empty, aligned slice — the loader leans on `is_empty()`
    /// to switch whole channels off.
    #[test]
    fn empty_is_empty() {
        let a: Aligned<i8> = Aligned::from_iter_len(0, std::iter::empty());
        assert!(a.is_empty());
        assert_eq!(a.as_ptr().align_offset(LINE), 0);
    }
}
