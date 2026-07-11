//! Move-ordering statistics: the history tables and their single write
//! primitive (design/09 — move ordering: MovePicker and history tables).
//!
//! This module owns every history-table *type* (butterfly, low-ply, capture,
//! continuation, pawn, correction), the per-thread [`Histories`] container,
//! the borrow-safe `ContHistId` continuation-history key, and the update
//! functions. It lands incrementally over design/09 §4:
//!
//! - **step 2:** the generic gravity cell [`StatEntry`] — the *only* way any
//!   table is ever written after init.
//! - **steps 3–4 (here):** the table catalogue + [`Histories`] container
//!   (§3.4) with its heap-construction / [`clear`](Histories::clear) /
//!   [`new_search`](Histories::new_search) plumbing, and the borrow-safe
//!   [`ContHistId`] continuation-history key (§3.2).
//! - later steps: the `update_*` / correction-history functions (§3.9–§3.10).
//!
//! Everything is per-thread, plain (non-atomic) `i16` (D23); nothing here
//! allocates on a search hot path.

use crate::types::{Key, Piece, Square};

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

// ---------------------------------------------------------------------------
// Table catalogue (design/09 §3.4): shapes, saturation bounds, and the
// init / per-`go` fill & decay values.
// ---------------------------------------------------------------------------

// Structural saturation bounds `D` per table family. These are NOT SPSA
// tunables (params.rs owns those); they are the type-level gravity limits, so
// they live here as the const-generic arguments of the `StatEntry`
// instantiations below (design/09 §3.4; params.rs header).
pub const MAIN_D: i32 = 7183; // butterfly + low-ply
pub const CAPT_D: i32 = 10692;
pub const CONT_D: i32 = 30000;
pub const PAWN_D: i32 = 8192;
pub const CORR_D: i32 = 1024; // unified + continuation correction
pub const TT_MOVE_D: i32 = 8192; // the lone tt-move reliability scalar

// Fixed dimensions (design/09 §3.4). `BUTTERFLY_SIZE` is the whole
// `Move::raw16` space: the raw16 bijection makes indexing by it statistically
// identical to Stockfish's from×to butterfly (design/09 §3.4, D2).
const BUTTERFLY_SIZE: usize = 1 << 16; // 65536
const LOW_PLY_SIZE: usize = 5; // plies 0..=4
const VICTIM_NB: usize = 7; // 6 piece types + a "no victim" slot (ep, quiet promos)
const CONT_NB: usize = 3328; // in_check(2) × capture(2) × piece13 × to(64)
const CONT_CORR_NB: usize = 832; // piece13 × to(64)
const PAWN_HIST_SIZE: usize = 8192; // power of two, masked by the pawn key
const CORR_SIZE: usize = 65536; // power of two, masked by each correction key

// Init (new-game) and per-`go` fill / decay values (design/09 §3.4, D20).
// Like the bounds above, these are the tables' defined reset state rather than
// scoring multipliers, so they stay here with their catalogue.
const MAIN_INIT: i16 = -5;
const CAPTURE_INIT: i16 = -699;
const CONT_INIT: i16 = -552;
const PAWN_INIT: i16 = -1262;
const CORR_INIT: i16 = -6;
const CONT_CORR_INIT: i16 = 5;
const LOW_PLY_REFILL: i16 = 100; // per-`go` fill; low-ply has no new-game init
const MAIN_DECAY_NUM: i32 = 789; // per-`go` butterfly decay: e = e * 789 / 1024
const MAIN_DECAY_DEN: i32 = 1024;

/// One `[piece × to]` slab of the continuation-history tensor — the innermost
/// two axes shared by every [`ContHistId`]-keyed table (design/09 §3.4).
pub type PieceToHist = [[StatEntry<CONT_D>; 64]; 12];

/// The continuation-*correction* counterpart: the same `[piece × to]` shape
/// with the tighter `CORR_D` bound (design/09 §3.10).
pub type PieceToCorr = [[StatEntry<CORR_D>; 64]; 12];

/// The four correction-history fields that share one `corr[slot][stm]` cell.
///
/// Each field is addressed under its *own* Zobrist key into the same array
/// (design/09 §3.10, D9); the bundle only colocates them for cache locality,
/// it does not imply they are ever read at a single shared index.
#[derive(Clone, Copy)]
pub struct CorrBundle {
    pub pawn: StatEntry<CORR_D>,
    pub minor: StatEntry<CORR_D>,
    pub non_pawn_white: StatEntry<CORR_D>,
    pub non_pawn_black: StatEntry<CORR_D>,
}

impl CorrBundle {
    /// A bundle with all four fields seeded to `v` (the §3.4 fill value).
    const fn filled(v: i16) -> Self {
        CorrBundle {
            pawn: StatEntry::new(v),
            minor: StatEntry::new(v),
            non_pawn_white: StatEntry::new(v),
            non_pawn_black: StatEntry::new(v),
        }
    }
}

/// A borrow-safe handle to one inner [`PieceToHist`] slab of the continuation
/// history (design/09 §3.2).
///
/// Stockfish stores a raw `PieceToHistory*` on its search stack; we cannot —
/// a raw pointer would need `unsafe` (D14) and a `&` borrow would freeze the
/// whole tensor while the search still needs to `make_move` and update other
/// histories between picker calls. Instead the stack carries this compact
/// `Copy` flat index, and [`Histories::cont_ref`] / [`Histories::cont_mut`]
/// resolve it in a single indexed load.
///
/// The index flattens `(in_check, capture, prior piece, prior to-square)` —
/// with a 13th "no piece" slot (index 12) for null moves and the pre-root
/// padding — as `((in_check*2 + capture) * 13 + piece13) * 64 + to`, landing
/// in `0..3328`. The `(piece13, to)` sub-index occupies the low `13*64 = 832`
/// positions of each quadrant, which is exactly why
/// [`piece_to`](Self::piece_to) — shared with the continuation *correction*
/// history — is a plain `% 832`.
///
/// Idiom note: this replaces Stockfish's stack-stored pointer with an index,
/// the standard Rust move to keep an aliasing-heavy data structure both
/// `unsafe`-free and borrow-checker-friendly (the search reads *and* writes
/// the tables while these keys sit inertly on the stack).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ContHistId(u16);

impl ContHistId {
    /// The inert `[false][false][no-piece][A1]` slab (flat index 768):
    /// installed at the pre-root padding slots and after null moves. It is
    /// filled like every other slab and *read* by scoring, but never
    /// *written* (updates gate on `current_move.is_ok()`), so it behaves as a
    /// constant baseline — mirroring Stockfish's sentinel continuation table.
    pub const SENTINEL: ContHistId = ContHistId(768); // ((0*2+0)*13 + 12) * 64 + 0

    /// Flatten `(in_check, capture, piece, to)` to the slab index. `piece` is
    /// `None` for the null-move / padding sentinel (the 13th "no piece" slot,
    /// index 12), matching Stockfish's `NO_PIECE` continuation slab.
    #[inline]
    pub fn new(in_check: bool, capture: bool, piece: Option<Piece>, to: Square) -> Self {
        let pc = piece.map_or(12, |p| p.index()); // 12 = "no piece"
        let flat =
            ((in_check as usize * 2 + capture as usize) * 13 + pc) * 64 + to.index() as usize;
        ContHistId(flat as u16)
    }

    /// The `(piece13 * 64 + to)` sub-index (`0..832`), shared verbatim with
    /// the continuation-correction history's outer index (design/09 §3.10).
    /// The `% 832` is a constant modulus (compiles to a multiply-shift).
    #[inline]
    pub fn piece_to(self) -> usize {
        self.0 as usize % CONT_CORR_NB
    }
}

/// The per-thread catalogue of move-ordering statistics (design/09 §3.4).
///
/// All tables are plain (non-atomic) `i16` behind `Box`es: ~20 MiB that lives
/// once per `Worker` (D23), never synchronized. The multi-MB tables are boxed
/// so the arrays are built straight on the heap — see [`Histories::new`].
pub struct Histories {
    /// Butterfly history, `[stm][move.raw16()]`.
    pub main: Box<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; 2]>,
    /// Low-ply history, `[ply][move.raw16()]` for plies 0..=4 — refilled to
    /// `LOW_PLY_REFILL` each `go`; no new-game init (§3.4).
    pub low_ply: Box<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; LOW_PLY_SIZE]>,
    /// Capture history, `[piece][to][victim_slot]` (victim 6 = "no victim").
    pub capture: Box<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>,
    /// Continuation history, outer index [`ContHistId`], inner [`PieceToHist`].
    pub cont: Box<[PieceToHist; CONT_NB]>,
    /// Pawn-structure history, `[pawn_key & mask][piece][to]`.
    pub pawn: Box<[[[StatEntry<PAWN_D>; 64]; 12]; PAWN_HIST_SIZE]>,
    /// Unified correction history, `[key & mask][stm]` of [`CorrBundle`]s.
    pub corr: Box<[[CorrBundle; 2]; CORR_SIZE]>,
    /// Continuation correction, outer [`ContHistId::piece_to`], inner
    /// [`PieceToCorr`].
    pub cont_corr: Box<[PieceToCorr; CONT_CORR_NB]>,
    /// A single scalar tracking the reliability of the TT move.
    pub tt_move: StatEntry<TT_MOVE_D>,
}

/// Heap-allocate a `Box<[T; N]>` with every element cloned from `row`, without
/// the whole array ever transiting the stack.
///
/// Idiom (design/09 §3.4): `vec![row; N]` allocates the `N` copies *directly
/// on the heap*; `into_boxed_slice` drops the length word (the `Vec`'s
/// capacity already equals `N`, so no realloc); and the stable
/// `TryFrom<Box<[T]>> for Box<[T; N]>` is a pointer-metadata cast, not a move.
/// This is the safe, stable replacement for `Box::new([row; N])` (which would
/// build the multi-MB array on the stack first) and for `Box::new_zeroed`
/// (whose `assume_init` is `unsafe`, and zero is no table's init state).
fn boxed_array<T: Copy, const N: usize>(row: T) -> Box<[T; N]> {
    vec![row; N]
        .into_boxed_slice()
        .try_into()
        // The Vec has exactly N elements, so the conversion is infallible; the
        // Err payload is the original `Box<[T]>` (not `Debug`, so no `expect`).
        .unwrap_or_else(|_| unreachable!("vec![row; N] has exactly N elements"))
}

impl Histories {
    /// Allocate every table already filled with its new-game init value
    /// (design/09 §3.4) — the allocation *is* the [`clear`](Self::clear)
    /// state. Large tables go through `boxed_array` so nothing multi-MB
    /// touches the stack; the returned struct is seven pointers plus a scalar,
    /// so a plain `Box::new` finishes the job.
    pub fn new() -> Box<Self> {
        Box::new(Histories {
            main: boxed_array([StatEntry::new(MAIN_INIT); BUTTERFLY_SIZE]),
            // Low-ply has no new-game init (§3.4); 0 is a placeholder that
            // `new_search` overwrites before the first read of any `go`.
            low_ply: boxed_array([StatEntry::new(0); BUTTERFLY_SIZE]),
            capture: boxed_array([[StatEntry::new(CAPTURE_INIT); VICTIM_NB]; 64]),
            cont: boxed_array([[StatEntry::new(CONT_INIT); 64]; 12]),
            pawn: boxed_array([[StatEntry::new(PAWN_INIT); 64]; 12]),
            corr: boxed_array([CorrBundle::filled(CORR_INIT); 2]),
            cont_corr: boxed_array([[StatEntry::new(CONT_CORR_INIT); 64]; 12]),
            tt_move: StatEntry::new(0),
        })
    }

    /// Reset every table with a new-game init to that value (design/09 §3.4,
    /// the "Init" column) — the `ucinewgame` reset, in place, no realloc.
    /// Low-ply is deliberately left untouched (no new-game init; `new_search`
    /// refills it each `go`).
    pub fn clear(&mut self) {
        for row in self.main.iter_mut() {
            row.fill(StatEntry::new(MAIN_INIT));
        }
        for plane in self.capture.iter_mut() {
            for row in plane.iter_mut() {
                row.fill(StatEntry::new(CAPTURE_INIT));
            }
        }
        for slab in self.cont.iter_mut() {
            for row in slab.iter_mut() {
                row.fill(StatEntry::new(CONT_INIT));
            }
        }
        for plane in self.pawn.iter_mut() {
            for row in plane.iter_mut() {
                row.fill(StatEntry::new(PAWN_INIT));
            }
        }
        for slot in self.corr.iter_mut() {
            slot.fill(CorrBundle::filled(CORR_INIT));
        }
        for slab in self.cont_corr.iter_mut() {
            for row in slab.iter_mut() {
                row.fill(StatEntry::new(CONT_CORR_INIT));
            }
        }
        self.tt_move = StatEntry::new(0);
    }

    /// The per-`go` refresh (design/09 §3.4, the "Per-go" column): decay the
    /// butterfly history toward zero (`e = e * 789 / 1024`, so recent searches
    /// dominate) and refill the low-ply history to `LOW_PLY_REFILL`.
    pub fn new_search(&mut self) {
        for row in self.main.iter_mut() {
            for e in row.iter_mut() {
                *e = StatEntry::new((e.get() * MAIN_DECAY_NUM / MAIN_DECAY_DEN) as i16);
            }
        }
        for row in self.low_ply.iter_mut() {
            row.fill(StatEntry::new(LOW_PLY_REFILL));
        }
    }

    /// Resolve a continuation-history key to its slab (read side).
    #[inline]
    pub fn cont_ref(&self, id: ContHistId) -> &PieceToHist {
        &self.cont[id.0 as usize]
    }

    /// Resolve a continuation-history key to its slab (write side).
    #[inline]
    pub fn cont_mut(&mut self, id: ContHistId) -> &mut PieceToHist {
        &mut self.cont[id.0 as usize]
    }

    /// The pawn-history slot for a position, masked by its pawn key.
    #[inline]
    pub fn pawn_slot(&self, key: Key) -> &[[StatEntry<PAWN_D>; 64]; 12] {
        &self.pawn[(key & (PAWN_HIST_SIZE as u64 - 1)) as usize]
    }
}

// Compile-time guards on the catalogue (design/09 §8 criterion 2). Every
// `StatEntry` is 2 bytes, so these double as shape checks against §3.4.
const _: () = assert!(size_of::<StatEntry<MAIN_D>>() == 2);
const _: () = assert!(size_of::<CorrBundle>() == 8);
const _: () = assert!(size_of::<PieceToHist>() == 12 * 64 * 2); // 1.5 KiB
const _: () = assert!(size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; 2]>() == 256 * 1024);
const _: () = assert!(size_of::<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>() == 12 * 64 * 7 * 2);
const _: () = assert!(CONT_NB == 4 * 13 * 64);
const _: () = assert!(CONT_CORR_NB == 13 * 64);
// The sentinel slab must be a real, in-bounds continuation index.
const _: () = assert!(ContHistId::SENTINEL.0 == (12 * 64) as u16);
const _: () = assert!((ContHistId::SENTINEL.0 as usize) < CONT_NB);

/// Total heap footprint of one [`Histories`] (design/09 §8 criterion 2:
/// ≤ 21 MiB). ~20 MiB in practice.
const HISTORIES_BYTES: usize = size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; 2]>()
    + size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; LOW_PLY_SIZE]>()
    + size_of::<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>()
    + size_of::<[PieceToHist; CONT_NB]>()
    + size_of::<[[[StatEntry<PAWN_D>; 64]; 12]; PAWN_HIST_SIZE]>()
    + size_of::<[[CorrBundle; 2]; CORR_SIZE]>()
    + size_of::<[PieceToCorr; CONT_CORR_NB]>()
    + size_of::<StatEntry<TT_MOVE_D>>();
// The catalogue is ~20 MiB: comfortably under the 21 MiB budget (§8), and
// well over 19 MiB (a guard against a table silently shrinking away).
const _: () = assert!(HISTORIES_BYTES <= 21 * 1024 * 1024);
const _: () = assert!(HISTORIES_BYTES > 19 * 1024 * 1024);

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

    // ---- design/09 §5, steps 3–4: ContHistId + the table catalogue ----

    use crate::types::{Color, PieceType};

    // The 13 piece slots: the 12 real pieces (in `Piece::index` order) then
    // the `None` sentinel slot.
    fn piece_for_slot(pc: usize) -> Option<Piece> {
        if pc < 12 {
            let color = if pc < 6 { Color::White } else { Color::Black };
            Some(Piece::new(color, PieceType::ALL[pc % 6]))
        } else {
            None
        }
    }

    // `ContHistId::new` is exactly the §3.2 flat formula: sweeping the four
    // axes in nested order enumerates 0..CONT_NB with no gaps or collisions,
    // and `piece_to()` recovers the (piece, to) sub-index of every id.
    #[test]
    fn cont_hist_id_is_the_flat_index_bijection() {
        let mut expected = 0usize;
        for in_check in [false, true] {
            for capture in [false, true] {
                for pc in 0..13usize {
                    let piece = piece_for_slot(pc);
                    for to in 0..64u8 {
                        let id = ContHistId::new(in_check, capture, piece, Square::new(to));
                        assert_eq!(id.0 as usize, expected, "flat index off");
                        assert_eq!(id.piece_to(), pc * 64 + to as usize, "piece_to off");
                        assert!((id.0 as usize) < CONT_NB);
                        assert!(id.piece_to() < CONT_CORR_NB);
                        expected += 1;
                    }
                }
            }
        }
        assert_eq!(expected, CONT_NB, "sweep did not cover exactly CONT_NB ids");
    }

    // SENTINEL is the real, in-bounds `[false][false][None][A1]` slab (768),
    // distinct from any real move's slab; reading through it is inert.
    #[test]
    fn cont_hist_id_sentinel_is_no_piece_a1() {
        assert_eq!(
            ContHistId::SENTINEL,
            ContHistId::new(false, false, None, Square::A1)
        );
        assert_eq!(ContHistId::SENTINEL.0, 768);
        assert!((ContHistId::SENTINEL.0 as usize) < CONT_NB);
        assert_eq!(ContHistId::SENTINEL.piece_to(), 12 * 64);

        // No real move (piece index 0..12) can produce the sentinel slab.
        for pc in 0..12usize {
            let id = ContHistId::new(false, false, piece_for_slot(pc), Square::A1);
            assert_ne!(id, ContHistId::SENTINEL);
        }
    }

    // Fresh allocation carries each table's §3.4 init value at both extremes.
    #[test]
    fn new_has_init_values() {
        let h = Histories::new();
        assert_eq!(h.main[0][0].get(), MAIN_INIT as i32);
        assert_eq!(h.main[1][BUTTERFLY_SIZE - 1].get(), MAIN_INIT as i32);
        // Low-ply has no new-game init: the placeholder 0.
        assert_eq!(h.low_ply[0][0].get(), 0);
        assert_eq!(h.low_ply[LOW_PLY_SIZE - 1][BUTTERFLY_SIZE - 1].get(), 0);
        assert_eq!(h.capture[0][0][0].get(), CAPTURE_INIT as i32);
        assert_eq!(h.capture[11][63][VICTIM_NB - 1].get(), CAPTURE_INIT as i32);
        assert_eq!(h.cont[0][0][0].get(), CONT_INIT as i32);
        assert_eq!(h.cont[CONT_NB - 1][11][63].get(), CONT_INIT as i32);
        assert_eq!(h.pawn[0][0][0].get(), PAWN_INIT as i32);
        assert_eq!(h.pawn[PAWN_HIST_SIZE - 1][11][63].get(), PAWN_INIT as i32);
        let b = &h.corr[CORR_SIZE - 1][1];
        assert_eq!(b.pawn.get(), CORR_INIT as i32);
        assert_eq!(b.minor.get(), CORR_INIT as i32);
        assert_eq!(b.non_pawn_white.get(), CORR_INIT as i32);
        assert_eq!(b.non_pawn_black.get(), CORR_INIT as i32);
        assert_eq!(h.cont_corr[0][0][0].get(), CONT_CORR_INIT as i32);
        assert_eq!(
            h.cont_corr[CONT_CORR_NB - 1][11][63].get(),
            CONT_CORR_INIT as i32
        );
        assert_eq!(h.tt_move.get(), 0);
    }

    // `clear` restores every new-game-init table to its init value after the
    // cells are poked to arbitrary values.
    #[test]
    fn clear_restores_init_values() {
        let mut h = Histories::new();
        h.main[1][123] = StatEntry::new(4000);
        h.capture[5][20][3] = StatEntry::new(-3000);
        h.cont[100][7][40] = StatEntry::new(9999);
        h.pawn[42][3][30] = StatEntry::new(2000);
        h.corr[9][1].minor = StatEntry::new(500);
        h.cont_corr[7][2][10] = StatEntry::new(-400);
        h.tt_move = StatEntry::new(1234);

        h.clear();

        assert_eq!(h.main[1][123].get(), MAIN_INIT as i32);
        assert_eq!(h.capture[5][20][3].get(), CAPTURE_INIT as i32);
        assert_eq!(h.cont[100][7][40].get(), CONT_INIT as i32);
        assert_eq!(h.pawn[42][3][30].get(), PAWN_INIT as i32);
        assert_eq!(h.corr[9][1].minor.get(), CORR_INIT as i32);
        assert_eq!(h.cont_corr[7][2][10].get(), CONT_CORR_INIT as i32);
        assert_eq!(h.tt_move.get(), 0);
    }

    // `new_search` decays the butterfly history by 789/1024 (1024 -> 789,
    // truncating toward zero on negatives) and refills low-ply to 100, leaving
    // every other table alone.
    #[test]
    fn new_search_decays_main_and_refills_low_ply() {
        let mut h = Histories::new();
        h.main[0][7] = StatEntry::new(1024);
        h.main[1][9] = StatEntry::new(-1024);
        assert_eq!(h.low_ply[0][0].get(), 0);

        h.new_search();

        assert_eq!(h.main[0][7].get(), 789); // 1024 * 789 / 1024
        assert_eq!(h.main[1][9].get(), -789); // truncates toward zero
        for ply in 0..LOW_PLY_SIZE {
            assert_eq!(h.low_ply[ply][0].get(), LOW_PLY_REFILL as i32);
            assert_eq!(
                h.low_ply[ply][BUTTERFLY_SIZE - 1].get(),
                LOW_PLY_REFILL as i32
            );
        }
        // A table with no per-`go` op is untouched.
        assert_eq!(h.capture[0][0][0].get(), CAPTURE_INIT as i32);
    }

    // The resolving accessors are in bounds at both extremes and the sentinel,
    // and `pawn_slot` masks by `PAWN_HIST_SIZE - 1` so aliased keys share a
    // slab.
    #[test]
    fn accessors_resolve_in_bounds() {
        let mut h = Histories::new();
        let id = ContHistId::new(
            true,
            true,
            Some(Piece::new(Color::Black, PieceType::King)),
            Square::H8,
        );
        h.cont_mut(id)[3][20] = StatEntry::new(4242);
        assert_eq!(h.cont_ref(id)[3][20].get(), 4242);
        assert_eq!(
            h.cont_ref(ContHistId::SENTINEL)[0][0].get(),
            CONT_INIT as i32
        );

        assert_eq!(h.pawn_slot(u64::MAX)[0][0].get(), PAWN_INIT as i32);
        assert!(std::ptr::eq(
            h.pawn_slot(5),
            h.pawn_slot(5 + PAWN_HIST_SIZE as u64),
        ));
    }

    // Catalogue shapes match §3.4. The ≤ 21 MiB total budget is enforced at
    // compile time by the `const _: () = assert!(HISTORIES_BYTES <= …)` guard
    // above; here we pin the per-cell/-slab byte sizes those totals rest on.
    #[test]
    fn catalogue_shapes_match_spec() {
        assert_eq!(size_of::<CorrBundle>(), 8);
        assert_eq!(size_of::<PieceToHist>(), 12 * 64 * 2); // 1.5 KiB
        assert_eq!(size_of::<PieceToCorr>(), 12 * 64 * 2);
        assert_eq!(size_of::<StatEntry<MAIN_D>>(), 2);
    }
}
