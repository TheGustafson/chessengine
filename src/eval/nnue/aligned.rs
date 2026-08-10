//! Cache-line-aligned heap slices for the NNUE weight tables.
//!
//! Every SIMD kernel in `sota_simd` reads its weights as full 64-byte `zmm`
//! loads out of a feature-major table whose row stride is a multiple of 64
//! bytes — so the alignment of *every* weight load is decided by ONE thing:
//! the alignment of the table's first element. A `Box<[i16]>` built by
//! `collect()` has alignment 2, and glibc's mmap path hands large tables
//! back at `page + 16`, so the base lands at **16 mod 64** and every `zmm`
//! load straddles a cache line.
//!
//! The alignment needs no allocator API: `Vec::with_capacity` reserves up
//! front, the safe `<*const T>::align_offset` reports how many elements to
//! skip, and the payload is written after exactly that many padding
//! elements; a `Vec` never moves while it stays inside its reserved
//! capacity (`payload_is_line_aligned` checks the property directly).
//! NOTE: **never call `into_boxed_slice()` on the backing vector** — its
//! shrink-to-fit realloc may return a differently-aligned block. That is why
//! the payload stays in a `Vec<T>` behind `Deref`.
//!
//! # The one invariant: `OFF_LE_LEN`
//!
//! **`self.off <= self.buf.len()`, always.** `Deref`/`DerefMut` rely on it
//! to hand out `buf[off..]` *without* a range check. It holds by
//! construction: both fields are private and [`Aligned::from_iter_len`] is
//! the only writer of `off`; `buf.resize(off, …)` makes `buf.len() == off`
//! and `extend` can only grow it; nothing shrinks `buf` (no
//! `truncate`/`pop`/`into_boxed_slice`, and `DerefMut`'s `&mut [T]` cannot
//! change a `Vec`'s length). Debug builds re-check it on every access
//! (`debug_check`).

use std::fmt;
use std::ops::{Deref, DerefMut};

/// One x86-64 cache line — also the width of one `zmm` load.
pub const LINE: usize = 64;

/// A heap slice of `T` whose first element is [`LINE`]-aligned. Derefs to
/// `[T]`, so it is a drop-in for the `Box<[T]>` it replaces.
pub struct Aligned<T> {
    /// `[padding …][payload …]`. Never shrunk — see the module docs.
    buf: Vec<T>,
    /// Index of the first payload element; `buf.as_ptr().add(off)` is
    /// `LINE`-aligned.
    off: usize,
}

impl<T: Default + Clone> Aligned<T> {
    /// Collect `iter` into a cache-line-aligned buffer. `len` is the exact
    /// element count, sizing the one allocation so the pointer measured up
    /// front is the pointer the payload ends up on; a shorter iterator
    /// simply yields a shorter payload (the loader enforces section sizes).
    pub fn from_iter_len(len: usize, iter: impl Iterator<Item = T>) -> Self {
        // Worst-case padding: one element short of a whole line (exact, since
        // size_of::<T>() divides LINE for every element type we use).
        let pad = LINE / size_of::<T>();
        let mut buf: Vec<T> = Vec::with_capacity(len + pad);
        // How many ELEMENTS until a LINE boundary. `usize::MAX` means
        // "impossible" (cannot happen when size_of::<T>() divides LINE), but
        // fall back to 0 rather than panic — misaligned is slow, not wrong.
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
    /// assumes, on every weight-table access of every debug run.
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
    /// Convenience for sized iterators; the lower size hint picks capacity.
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

// The crate denies `unsafe_code`; this is a documented exception. Both
// blocks below are the pointer arithmetic `&self.buf[self.off..]` already
// performs, minus the range check that [`OFF_LE_LEN`] proves can never fire.
#[allow(unsafe_code)]
impl<T> Deref for Aligned<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        Self::debug_check(self.off, self.buf.len());
        // SAFETY: OFF_LE_LEN — `self.off <= self.buf.len()`, so `off..` is a
        // valid range for `buf`; this is `&buf_slice[off..]` with the bounds
        // check elided, not an allocator or provenance trick.
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

/// Payload equality — padding is invisible.
impl<T: PartialEq> PartialEq for Aligned<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

/// Length only — a 138 MB net must not print itself (`SotaNetwork` derives
/// `Debug`).
impl<T> fmt::Debug for Aligned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Aligned<{}>[{}]", std::any::type_name::<T>(), self.len())
    }
}

// --- Blocks — the shape the head's weight tables actually have -------------

/// One bucket's weight block: `[T; N]`, starting on its own cache line.
/// `repr(align(64))` makes alignment a property of the *type*, so the
/// allocator honours it for every array element and `into_boxed_slice`'s
/// shrink-to-fit — the trap [`Aligned`] has to dodge — cannot lose it.
#[repr(align(64))]
#[derive(Clone, PartialEq, Eq)]
pub struct Line<T, const N: usize>(pub [T; N]);

/// `B` cache-line-aligned blocks of `N` `T`s.
///
/// Exists next to [`Aligned`] because every head weight table is read
/// exactly one way — *give me bucket `b`'s block* — and as an `Aligned<T>`
/// that acquisition re-derived load-time facts on every eval (fat-pointer
/// materialisation, range check, `try_into` length check). Stated as
/// `Blocks<T, N, B>` it is one pointer load plus one `lea`, and the only
/// surviving check is `b < B` against a *compile-time constant*, which LLVM
/// CSEs across the forward pass. No `unsafe` in this type.
pub struct Blocks<T, const N: usize, const B: usize>(Box<[Line<T, N>; B]>);

impl<T, const N: usize, const B: usize> Blocks<T, N, B> {
    /// Compile-time proof that `[Line<T, N>; B]` is `B · N` contiguous `T`s
    /// with no interior padding (keeps the on-disk layout meaningful).
    /// Referenced from [`block`](Self::block) so it is actually evaluated.
    const LAYOUT_OK: () = assert!(
        N * size_of::<T>() % LINE == 0,
        "Blocks<T, N, B>: N * size_of::<T>() must be a multiple of the cache \
         line, or repr(align(64)) pads between blocks"
    );

    /// Bucket `i`'s block. One bounds check against the constant `B`.
    #[inline]
    pub fn block(&self, i: usize) -> &[T; N] {
        let () = Self::LAYOUT_OK;
        &self.0[i].0
    }

    /// Flat iteration over all `B · N` elements (checksum path, never hot).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter().flat_map(|line| line.0.iter())
    }

    /// Overwrite every element (`[T]::fill` for a block table).
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for line in self.0.iter_mut() {
            line.0.fill(value.clone());
        }
    }

    /// Base pointer, for the alignment gate. Casting a reference to a pointer
    /// dereferences nothing, so this needs no `unsafe`.
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr().cast()
    }

    /// Total element count, `B · N`.
    pub const fn len(&self) -> usize {
        N * B
    }

    /// Never true for a shipped table; clippy requires it alongside `len`.
    pub const fn is_empty(&self) -> bool {
        N * B == 0
    }
}

impl<T: Copy + Default, const N: usize, const B: usize> Blocks<T, N, B> {
    /// Build from exactly `B · N` elements in block-major order.
    ///
    /// **Panics** on a short iterator — the same error model the range
    /// slices had, moved from *every eval* to *load time*, where the CENN
    /// section-size validation has already made it unreachable.
    pub fn from_flat_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let mut it = iter.into_iter();
        let mut v: Vec<Line<T, N>> = Vec::with_capacity(B);
        for b in 0..B {
            let mut line = Line([T::default(); N]);
            for (j, slot) in line.0.iter_mut().enumerate() {
                *slot = it
                    .next()
                    .unwrap_or_else(|| panic!("Blocks: iterator ended at block {b} element {j}"));
            }
            v.push(line);
        }
        // `len == capacity == B`, so `into_boxed_slice` cannot realloc; and
        // even if it did, `Line`'s alignment travels in the layout.
        let boxed: Box<[Line<T, N>]> = v.into_boxed_slice();
        match <Box<[Line<T, N>; B]>>::try_from(boxed) {
            Ok(arr) => Blocks(arr),
            Err(_) => unreachable!("exactly B lines were pushed"),
        }
    }
}

impl<T: Copy + Default, const N: usize, const B: usize> FromIterator<T> for Blocks<T, N, B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_flat_iter(iter)
    }
}

impl<T: Copy + Default, const N: usize, const B: usize> From<Vec<T>> for Blocks<T, N, B> {
    fn from(v: Vec<T>) -> Self {
        Self::from_flat_iter(v)
    }
}

impl<T: PartialEq, const N: usize, const B: usize> PartialEq for Blocks<T, N, B> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Shape only — same reason as [`Aligned`]'s.
impl<T, const N: usize, const B: usize> fmt::Debug for Blocks<T, N, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Blocks<{}>[{B}x{N}]", std::any::type_name::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload pointer is line-aligned for every element type the loader
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

    /// Contents survive the padding trick (an off-by-one in `off` would show).
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

    /// An empty table is still a valid, empty, aligned slice — the loader
    /// leans on `is_empty()` to switch whole channels off.
    #[test]
    fn empty_is_empty() {
        let a: Aligned<i8> = Aligned::from_iter_len(0, std::iter::empty());
        assert!(a.is_empty());
        assert_eq!(a.as_ptr().align_offset(LINE), 0);
    }

    /// EVERY block starts on a cache line, not just the first — the property
    /// `Aligned` could only give for the table base.
    #[test]
    fn every_block_starts_on_a_cache_line() {
        fn check<T: Copy + Default, const N: usize, const B: usize>(name: &str) {
            let b: Blocks<T, N, B> =
                Blocks::from_flat_iter(std::iter::repeat_n(T::default(), N * B));
            for i in 0..B {
                assert_eq!(
                    (b.block(i).as_ptr()).align_offset(LINE),
                    0,
                    "{name}: block {i} not {LINE}-aligned"
                );
            }
            // No interior padding: consecutive blocks are exactly N apart.
            for i in 1..B {
                let step = b.block(i).as_ptr() as usize - b.block(i - 1).as_ptr() as usize;
                assert_eq!(step, N * size_of::<T>(), "{name}: gap between blocks");
            }
        }
        check::<f32, 16, 8>("f32 x16");
        check::<f32, 32, 8>("f32 x32");
        check::<i8, 1280, 8>("i8 x1280");
        check::<i8, 20480, 8>("i8 x20480");
    }

    /// Block-major order survives the reshape; `iter()`/`len()` still see the
    /// flat table.
    #[test]
    fn block_major_order_round_trips() {
        let b: Blocks<f32, 16, 8> = (0..128).map(|i| i as f32).collect();
        assert_eq!(b.len(), 128);
        assert!(b.iter().copied().eq((0..128).map(|i| i as f32)));
        for bucket in 0..8 {
            let blk = b.block(bucket);
            assert_eq!(blk[0], (bucket * 16) as f32);
            assert_eq!(blk[15], (bucket * 16 + 15) as f32);
        }
    }

    /// The load-time panic the range slices used to raise per eval.
    #[test]
    #[should_panic(expected = "iterator ended at block 1 element 0")]
    fn a_short_table_fails_at_construction() {
        let _: Blocks<f32, 16, 8> = (0..16).map(|i| i as f32).collect();
    }
}
