//! Move-ordering statistics: the history tables and their single write
//! primitive (design/09 — move ordering: MovePicker and history tables).
//!
//! This module owns every history-table *type* (butterfly, low-ply, capture,
//! continuation, pawn, correction), the per-thread [`Histories`] container,
//! the borrow-safe `ContHistId` continuation-history key, and the update
//! functions. It lands incrementally over design/09 §4:
//!
//! - **step 2 (here):** the generic gravity cell [`StatEntry`] — the *only*
//!   way any table is ever written after init.
//! - later steps: the table catalogue + `Histories` (§3.4), `ContHistId`
//!   (§3.2), and the `update_*` / correction-history functions (§3.9–§3.10).
//!
//! Everything is per-thread, plain (non-atomic) `i16` (D23); nothing here
//! allocates on a search hot path.

/// A single history cell bounded to `[-D, D]` by the "gravity" update rule
/// (design/09 §3.3).
///
/// The one write primitive for every table: no cell is ever assigned except
/// by `fill` at init time and by [`StatEntry::update`] thereafter. The bound
/// is structural — `D` is a const generic, so each table fixes its own
/// saturation limit at the type level (`StatEntry<7183>`, `StatEntry<30000>`,
/// …) and the compiler monomorphizes the arithmetic per limit.
///
/// Storage is `i16`: every `D` in the catalogue (§3.4) is `<= 30000`, so a
/// bounded value always fits — checked at compile time per instantiation via
/// [`StatEntry::D_FITS`].
///
/// Idiom note: a `const` *generic parameter* (`const D: i32`) lets a single
/// type carry a compile-time-varying bound without a runtime field or a
/// macro-generated family of structs. It is the Rust analogue of a C++
/// non-type template parameter, and the way Stockfish's `StatsEntry<T, D>`
/// is expressed idiomatically here (D27 — reimplementation, not a copy).
#[derive(Clone, Copy)]
pub struct StatEntry<const D: i32>(i16);

impl<const D: i32> StatEntry<D> {
    /// Compile-time guard: every table's bound must be positive and fit the
    /// `i16` backing store. Referenced from [`StatEntry::new`] so the check
    /// is forced for each concrete `D` the crate instantiates (a bare
    /// associated const is only evaluated when used).
    const D_FITS: () = assert!(D > 0 && D <= i16::MAX as i32);

    /// A cell seeded to `v` (the per-table `fill` init value of §3.4). The
    /// caller is responsible for `v` lying in `[-D, D]`; `update` keeps it
    /// there forever after.
    pub const fn new(v: i16) -> Self {
        // Touching D_FITS forces its assert to evaluate at monomorphization.
        let () = Self::D_FITS;
        Self(v)
    }

    /// The current value, widened to `i32` for the scoring arithmetic that
    /// reads it (sums of several history terms overflow `i16`).
    pub const fn get(self) -> i32 {
        self.0 as i32
    }

    /// The gravity update (design/09 §3.3): clamp the bonus to `[-D, D]`,
    /// then move the cell toward it while decaying the old value in
    /// proportion to `|bonus| / D`.
    ///
    /// Two provable consequences (both exercised in the tests below):
    /// - the cell never escapes `[-D, D]` — integer truncation of
    ///   `v * |b| / D` only ever shrinks the correction, so the closed-form
    ///   bound survives;
    /// - repeated same-sign bonuses have diminishing effect and asymptote to
    ///   `±D` (a max-magnitude bonus `|b| == D` lands there in one step,
    ///   since `v + D - v*D/D == D`), while stale signal decays geometrically
    ///   toward zero once the bonus flips or stops.
    pub fn update(&mut self, bonus: i32) {
        let b = bonus.clamp(-D, D);
        let v = self.0 as i32;
        let next = v + b - v * b.abs() / D;
        debug_assert!(next.abs() <= D, "gravity escaped [-{D}, {D}]: {next}");
        self.0 = next as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny LCG (PCG-XSH multiplier + odd increment) — deterministic
    // pseudo-randomness with zero dev-dependencies (design/09 §5 step 2).
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    // Bound invariant: over a long adversarial stream of bonuses (many well
    // outside [-D, D], so the clamp path is hammered) and occasional random
    // re-seeds of the cell across the whole [-D, D] band, the value never
    // escapes the bound. debug_assert inside `update` double-guards this.
    fn assert_bounded<const D: i32>() {
        let mut s = 0x853c49e6748fea9bu64 ^ (D as u64);
        let mut e = StatEntry::<D>::new(0);
        for _ in 0..100_000 {
            let r = lcg_next(&mut s);
            let bonus = ((r >> 11) as i64 % (6 * D as i64 + 1) - 3 * D as i64) as i32;
            if r & 0xF == 0 {
                let init = ((lcg_next(&mut s) >> 11) as i64 % (2 * D as i64 + 1) - D as i64) as i16;
                e = StatEntry::<D>::new(init);
            }
            e.update(bonus);
            assert!(e.get().abs() <= D, "D={D}: escaped to {}", e.get());
        }
    }

    #[test]
    fn gravity_stays_bounded() {
        // Every D in the §3.4 catalogue, plus a small one for good measure.
        assert_bounded::<100>();
        assert_bounded::<1024>(); // CORR_D
        assert_bounded::<7183>(); // MAIN_D
        assert_bounded::<8192>(); // PAWN_D
        assert_bounded::<10692>(); // CAPT_D
        assert_bounded::<30000>(); // CONT_D
    }

    // Decay direction: a fixed sub-clamp bonus drives the cell monotonically
    // toward the matching bound and, given enough repetitions, lands exactly
    // on it (the gravity fixed point is ±D).
    fn assert_saturates<const D: i32>() {
        let step = (D / 10).max(1);

        let mut e = StatEntry::<D>::new(0);
        let mut prev = e.get();
        for _ in 0..10_000 {
            e.update(step);
            let cur = e.get();
            assert!(cur >= prev, "D={D}: not monotone up ({prev} -> {cur})");
            assert!(cur <= D);
            prev = cur;
        }
        assert_eq!(e.get(), D, "D={D}: positive bonus did not saturate to +D");

        let mut e = StatEntry::<D>::new(0);
        let mut prev = e.get();
        for _ in 0..10_000 {
            e.update(-step);
            let cur = e.get();
            assert!(cur <= prev, "D={D}: not monotone down ({prev} -> {cur})");
            assert!(cur >= -D);
            prev = cur;
        }
        assert_eq!(e.get(), -D, "D={D}: negative bonus did not saturate to -D");
    }

    #[test]
    fn gravity_saturates_toward_bound() {
        assert_saturates::<100>();
        assert_saturates::<1024>();
        assert_saturates::<7183>();
        assert_saturates::<8192>();
        assert_saturates::<10692>();
        assert_saturates::<30000>();
    }

    // A zero bonus is a pure identity for any starting value in-band.
    fn assert_zero_noop<const D: i32>() {
        let half = (D / 2) as i16;
        for v in [0i16, 1, -1, half, -half, D as i16, -(D as i16)] {
            let mut e = StatEntry::<D>::new(v);
            e.update(0);
            assert_eq!(e.get(), v as i32, "D={D}: update(0) changed {v}");
        }
    }

    #[test]
    fn gravity_zero_bonus_is_noop() {
        assert_zero_noop::<100>();
        assert_zero_noop::<1024>();
        assert_zero_noop::<7183>();
        assert_zero_noop::<30000>();
    }

    // The formula pinned against hand-computed values, including the integer
    // truncation-toward-zero of `v * |b| / D`, the clamp, and the one-step
    // saturation at |bonus| == D.
    #[test]
    fn gravity_matches_hand_computed_values() {
        // D = 100: easy mental arithmetic.
        let mut e = StatEntry::<100>::new(0);
        e.update(50);
        assert_eq!(e.get(), 50); // 0 + 50 - 0*50/100
        e.update(50);
        assert_eq!(e.get(), 75); // 50 + 50 - 2500/100 (=25)
        e.update(50);
        assert_eq!(e.get(), 88); // 75 + 50 - 3750/100 (=37)

        // Mixed sign, and truncation of a negative product toward zero.
        let mut e = StatEntry::<100>::new(-40);
        e.update(30);
        assert_eq!(e.get(), 2); // -40 + 30 - (-1200/100 = -12)
        let mut e = StatEntry::<100>::new(-33);
        e.update(-30);
        assert_eq!(e.get(), -54); // -33 - 30 - (-990/100 = -9)  [trunc toward 0]

        // Clamp: an out-of-range bonus is pinned to ±D before the update.
        let mut e = StatEntry::<100>::new(0);
        e.update(500);
        assert_eq!(e.get(), 100); // clamp(500)=100; 0 + 100 - 0
        let mut e = StatEntry::<100>::new(80);
        e.update(-500);
        assert_eq!(e.get(), -100); // clamp(-500)=-100; 80 - 100 - 8000/100 (=80)

        // A real bound: D = 7183.
        let mut e = StatEntry::<7183>::new(1000);
        e.update(1000);
        assert_eq!(e.get(), 1861); // 1000 + 1000 - 1_000_000/7183 (=139)

        // |bonus| == D lands on ±D in a single step from any start:
        // v + D - v*D/D == D.
        let mut e = StatEntry::<7183>::new(3000);
        e.update(7183);
        assert_eq!(e.get(), 7183);
        let mut e = StatEntry::<7183>::new(-2000);
        e.update(-7183);
        assert_eq!(e.get(), -7183);
    }

    #[test]
    fn stat_entry_is_two_bytes() {
        assert_eq!(std::mem::size_of::<StatEntry<7183>>(), 2);
    }
}
