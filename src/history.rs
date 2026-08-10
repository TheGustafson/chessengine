//! Move-ordering statistics (design/09): every history-table *type*
//! (butterfly, low-ply, capture, continuation, pawn, correction), the
//! per-thread [`Histories`] container, the borrow-safe [`ContHistId`] key,
//! and the update functions. The *call sites* and per-site bonus formulas
//! belong to design/10; this module owns only the internals (weights,
//! offsets, combiners).
//!
//! Everything is per-thread, plain (non-atomic) `i16` (D23); nothing here
//! allocates on a search hot path.

use crate::board::Board;
use crate::moves::Move;
use crate::search::params;
use crate::search::stack::{SS_OFFSET, Stack};
use crate::types::{
    Color, Depth, Key, Piece, PieceType, Square, VALUE_TB_LOSS_IN_MAX_PLY, VALUE_TB_WIN_IN_MAX_PLY,
    Value,
};

/// A single history cell bounded to `[-D, D]` by the "gravity" update rule
/// (design/09 §3.3) — the one write primitive for every table: no cell is
/// ever assigned except by `fill` at init and [`StatEntry::update`] after.
/// The const-generic `D` fixes each table's saturation limit at the type
/// level (the Rust analogue of Stockfish's `StatsEntry<T, D>` non-type
/// template parameter, reimplemented per D27); storage is `i16`, checked per
/// instantiation via [`StatEntry::D_FITS`].
#[derive(Clone, Copy)]
pub struct StatEntry<const D: i32>(i16);

impl<const D: i32> StatEntry<D> {
    /// Compile-time guard: `D` must be positive and fit `i16`. Referenced
    /// from [`StatEntry::new`] so the check fires for each concrete `D`.
    const D_FITS: () = assert!(D > 0 && D <= i16::MAX as i32);

    /// A cell seeded to `v`. The caller is responsible for `v` lying in
    /// `[-D, D]`; `update` keeps it there forever after.
    pub const fn new(v: i16) -> Self {
        // Touching D_FITS forces its assert to evaluate at monomorphization.
        let () = Self::D_FITS;
        Self(v)
    }

    /// The current value, widened to `i32` (sums of terms overflow `i16`).
    pub const fn get(self) -> i32 {
        self.0 as i32
    }

    /// The gravity update (design/09 §3.3): clamp the bonus to `[-D, D]`,
    /// then move the cell toward it while decaying the old value in
    /// proportion to `|bonus| / D`. Provable (and tested below): the cell
    /// never escapes `[-D, D]`; repeated same-sign bonuses have diminishing
    /// effect and asymptote to `±D`; stale signal decays geometrically.
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

// Structural saturation bounds `D` per table family — type-level gravity
// limits, NOT SPSA tunables (params.rs owns those).
pub const MAIN_D: i32 = 7183; // butterfly + low-ply
pub const CAPT_D: i32 = 10692;
pub const CONT_D: i32 = 30000;
pub const PAWN_D: i32 = 8192;
pub const CORR_D: i32 = 1024; // unified + continuation correction
pub const TT_MOVE_D: i32 = 8192; // the lone tt-move reliability scalar

// Fixed dimensions (design/09 §3.4). `BUTTERFLY_SIZE` is the whole
// `Move::raw16` space — statistically identical to SF's from×to butterfly.
const BUTTERFLY_SIZE: usize = 1 << 16; // 65536
/// Low-ply history plane count (plies 0..=4); public for the quiet scorer.
pub const LOW_PLY_SIZE: usize = 5;
const VICTIM_NB: usize = 7; // 6 piece types + a "no victim" slot (ep, quiet promos)
const CONT_NB: usize = 3328; // in_check(2) × capture(2) × piece13 × to(64)
/// Allocated slab count for `cont`. NOTE: padding it to 4096 to fold the
/// bounds check away costs +1.125 MiB per worker — over the design/09 §8
/// budget; measured and declined.
const CONT_ALLOC: usize = CONT_NB;
const CONT_CORR_NB: usize = 832; // piece13 × to(64)
/// Allocated slab count for `cont_corr`: padded to the next power of two so
/// [`CONT_CORR_MASK`] folds the bounds checks away. Unlike `cont`, this
/// padding costs only +288 KiB per worker — inside the §8 budget.
const CONT_CORR_ALLOC: usize = CONT_CORR_NB.next_power_of_two(); // 1024
/// Index mask for [`CONT_CORR_ALLOC`] — see that constant.
const CONT_CORR_MASK: usize = CONT_CORR_ALLOC - 1;
/// The `(in_check, capture)` axis pair — the LOW axis of [`ContHistId`],
/// which is what makes [`ContHistId::piece_to`] a shift.
const QUADRANT_NB: usize = 4;
const PAWN_HIST_SIZE: usize = 8192; // power of two, masked by the pawn key
const CORR_SIZE: usize = 65536; // power of two, masked by each correction key

// Init (new-game) and per-`go` fill / decay values (design/09 §3.4, D20) —
// the tables' defined reset state, not scoring multipliers.
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

/// The `[piece × to]` cell address as ONE index — `pci * 64 + to`. The flat
/// form matters where one `(pci, to)` pair is read against SEVERAL planes
/// (the quiet scorer): the shared term is the whole offset, one indexed load
/// per plane. The bounds check folds away without a mask or `unsafe`:
/// `pci < 12` and `to < 64` are already visible to LLVM's range analysis.
#[inline]
pub const fn piece_to(pci: usize, to: usize) -> usize {
    pci * 64 + to
}

/// Read one `[piece × to]` plane at a [`piece_to`] index. Generic over the
/// bound so the continuation and pawn planes share one spelling.
#[inline]
pub fn at_piece_to<const D: i32>(plane: &[[StatEntry<D>; 64]; 12], q: usize) -> i32 {
    plane.as_flattened()[q].get()
}

/// [`at_piece_to`]'s write side — the cell itself, for `update`.
#[inline]
pub fn at_piece_to_mut<const D: i32>(
    plane: &mut [[StatEntry<D>; 64]; 12],
    q: usize,
) -> &mut StatEntry<D> {
    &mut plane.as_flattened_mut()[q]
}

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

/// A borrow-safe handle to one inner [`PieceToHist`] slab of the
/// continuation history (design/09 §3.2): where Stockfish stores a raw
/// `PieceToHistory*` on its search stack, we store this compact `Copy` flat
/// index — a raw pointer would need `unsafe` (D14), and a `&` borrow would
/// freeze the whole tensor between picker calls.
///
/// The index flattens `(in_check, capture, prior piece, prior to-square)` —
/// with a 13th "no piece" slot for null moves and the pre-root padding — as
/// `(piece13 * 64 + to) * 4 + (in_check*2 + capture)`, in `0..3328`. The
/// `(piece13, to)` sub-index deliberately sits in the HIGH bits so
/// [`piece_to`](Self::piece_to) is a shift, not a `% 832`; nothing outside
/// this impl observes the ordering.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ContHistId(u16);

impl ContHistId {
    /// The inert `[false][false][no-piece][A1]` slab (flat index 3072):
    /// installed at the pre-root padding slots and after null moves. Read by
    /// scoring but never *written* (updates gate on `current_move.is_ok()`),
    /// so it is a constant baseline — SF's sentinel continuation table.
    pub const SENTINEL: ContHistId = ContHistId((12 * 64 * QUADRANT_NB) as u16);

    /// Flatten `(in_check, capture, piece, to)` to the slab index; `piece`
    /// is `None` for the null-move / padding sentinel (SF's `NO_PIECE`).
    #[inline]
    pub fn new(in_check: bool, capture: bool, piece: Option<Piece>, to: Square) -> Self {
        let pc = piece.map_or(12, |p| p.index()); // 12 = "no piece"
        let flat = (pc * 64 + to.idx()) * QUADRANT_NB + (in_check as usize * 2 + capture as usize);
        // INVARIANT (the `cont` indexing relies on it): `pc <= 12` and
        // `to < 64` (asserted in release too), so `flat <= 3327 < CONT_NB`
        // and the `as u16` cannot truncate.
        debug_assert!(flat < CONT_NB, "ContHistId {flat} out of range");
        ContHistId(flat as u16)
    }

    /// The `(piece13 * 64 + to)` sub-index (`0..832`), shared verbatim with
    /// the continuation-correction history's outer index (design/09 §3.10).
    /// A shift, because the axis order puts `(piece13, to)` in the high bits.
    #[inline]
    pub fn piece_to(self) -> usize {
        self.0 as usize / QUADRANT_NB
    }
}

/// A cache-line-aligned wrapper around one history table's outer element.
/// Row alignment is decided solely by the table's base pointer, and plain
/// `vec![row; N]` allocations landed 16 mod 64 — every 128-byte
/// `[piece][to]` row straddled THREE cache lines instead of two (measured).
/// `align(64)` routes the identical `vec!` through `aligned_alloc` at zero
/// size cost (asserted in the tests below). `corr` is deliberately NOT
/// wrapped: its 16-byte element would quadruple a 1 MiB table.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Line64<T>(pub T);

impl<T> std::ops::Deref for Line64<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Line64<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// The per-thread catalogue of move-ordering statistics (design/09 §3.4):
/// ~20 MiB of plain (non-atomic) `i16` behind `Box`es, once per `Worker`
/// (D23), never synchronized. Multi-row tables' outer elements are wrapped
/// in [`Line64`] so each allocation starts on a cache line.
pub struct Histories {
    /// Butterfly history, `[stm][move.raw16()]`.
    pub main: Box<[Line64<[StatEntry<MAIN_D>; BUTTERFLY_SIZE]>; 2]>,
    /// Low-ply history, `[ply][move.raw16()]` for plies 0..=4 — refilled to
    /// `LOW_PLY_REFILL` each `go`; no new-game init (§3.4).
    pub low_ply: Box<[Line64<[StatEntry<MAIN_D>; BUTTERFLY_SIZE]>; LOW_PLY_SIZE]>,
    /// Capture history, `[piece][to][victim_slot]` (victim 6 = "no victim").
    pub capture: Box<[Line64<[[StatEntry<CAPT_D>; VICTIM_NB]; 64]>; 12]>,
    /// Continuation history, outer index [`ContHistId`], inner [`PieceToHist`].
    pub cont: Box<[Line64<PieceToHist>; CONT_ALLOC]>,
    /// Pawn-structure history, `[pawn_key & mask][piece][to]`.
    pub pawn: Box<[Line64<[[StatEntry<PAWN_D>; 64]; 12]>; PAWN_HIST_SIZE]>,
    /// Unified correction history, `[key & mask][stm]` of [`CorrBundle`]s.
    pub corr: Box<[[CorrBundle; 2]; CORR_SIZE]>,
    /// Continuation correction, outer [`ContHistId::piece_to`], inner
    /// [`PieceToCorr`].
    pub cont_corr: Box<[Line64<PieceToCorr>; CONT_CORR_ALLOC]>,
    /// A single scalar tracking the reliability of the TT move.
    pub tt_move: StatEntry<TT_MOVE_D>,
}

/// Heap-allocate a `Box<[T; N]>` with every element cloned from `row`,
/// without the whole array ever transiting the stack. Idiom: `vec![row; N]`
/// allocates directly on the heap; `into_boxed_slice` + the stable
/// `TryFrom<Box<[T]>> for Box<[T; N]>` are pointer-metadata casts, not
/// moves — the safe replacement for `Box::new([row; N])`, which would build
/// the multi-MB array on the stack first.
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
    /// state. Large tables go through `boxed_array`.
    pub fn new() -> Box<Self> {
        Box::new(Histories {
            main: boxed_array(Line64([StatEntry::new(MAIN_INIT); BUTTERFLY_SIZE])),
            // Low-ply has no new-game init (§3.4); 0 is a placeholder that
            // `new_search` overwrites before the first read of any `go`.
            low_ply: boxed_array(Line64([StatEntry::new(0); BUTTERFLY_SIZE])),
            capture: boxed_array(Line64([[StatEntry::new(CAPTURE_INIT); VICTIM_NB]; 64])),
            cont: boxed_array(Line64([[StatEntry::new(CONT_INIT); 64]; 12])),
            pawn: boxed_array(Line64([[StatEntry::new(PAWN_INIT); 64]; 12])),
            corr: boxed_array([CorrBundle::filled(CORR_INIT); 2]),
            cont_corr: boxed_array(Line64([[StatEntry::new(CONT_CORR_INIT); 64]; 12])),
            tt_move: StatEntry::new(0),
        })
    }

    /// The `ucinewgame` reset, in place, no realloc (design/09 §3.4).
    /// Low-ply is deliberately untouched (`new_search` refills it each `go`).
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

    /// The per-`go` refresh (design/09 §3.4): decay the butterfly history
    /// toward zero and refill the low-ply history to `LOW_PLY_REFILL`.
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

    /// Resolve a [`ContHistId::piece_to`] sub-index to its continuation-
    /// correction slab (read side). See [`CONT_CORR_ALLOC`] for why the mask.
    #[inline]
    fn corr_slab(&self, o: usize) -> &PieceToCorr {
        &self.cont_corr[o & CONT_CORR_MASK]
    }

    /// [`Self::corr_slab`]'s write side.
    #[inline]
    fn corr_slab_mut(&mut self, o: usize) -> &mut PieceToCorr {
        &mut self.cont_corr[o & CONT_CORR_MASK]
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

    /// The pawn-history slot for a position (write side). Used by the
    /// search's out-of-band pawn nudges (design/10 §3.4 N8 / §3.10), which
    /// key off a non-side-to-move move and so cannot route through
    /// [`Self::update_quiet_histories`].
    #[inline]
    pub fn pawn_slot_mut(&mut self, key: Key) -> &mut [[StatEntry<PAWN_D>; 64]; 12] {
        &mut self.pawn[(key & (PAWN_HIST_SIZE as u64 - 1)) as usize]
    }
}

// ---------------------------------------------------------------------------
// History updates (design/09 §3.9) and correction history (§3.10).
//
// Every bonus flows through the gravity operator (§3.3). Signatures take the
// full padded stack slice plus the padded index `cur = ply + SS_OFFSET`
// (never the raw ply, D18): the refutation malus walks offsets 1..=6 from
// `cur - 1`, so its deepest read is `stack[cur - 7]`, in-bounds by the
// SS_OFFSET padding. `ContHistId` is `Copy`, so a key is read out of a stack
// slot *before* `hist` is mutated — no aliasing.
// ---------------------------------------------------------------------------

/// The "no victim" capture-history slot (en passant, quiet queen promotions):
/// the last victim axis, mirroring the movepick scorer's `NO_VICTIM`.
const NO_VICTIM: usize = VICTIM_NB - 1;

impl Histories {
    /// Reward or punish "move `pc`→`to` after each predecessor move" along the
    /// continuation chain (design/09 §3.9). `cur` is the padded index of the
    /// node whose predecessors are walked; offsets 1..=6 read `stack[cur - i]`.
    ///
    /// The consistency multiplier grows with the running count of already-
    /// positive entries (an agreeing chain amplifies the update); the offset-1
    /// entry always takes a flat `+71`, even for maluses. In check only offsets
    /// 1..=2 update — the position before the check bears little relation.
    pub fn update_continuation_histories(
        &mut self,
        stack: &[Stack],
        cur: usize,
        pc: Piece,
        to: Square,
        bonus: i32,
    ) {
        debug_assert!(cur >= SS_OFFSET);
        let in_check = stack[cur].in_check;
        // Every iteration hits the SAME `[pci][toi]` cell in a different
        // plane — the shape `piece_to` exists for.
        let q = piece_to(pc.index(), to.idx());
        let mut positive_count = 0usize;

        // A `macro_rules!` body: a `let` binding holding a borrow of `*self`
        // cannot live across `self.cont_mut(id)`.
        macro_rules! cont_body {
            ($idx:expr, $weight:expr, $frame:expr) => {{
                let idx: usize = $idx;
                let i = idx + 1;
                if in_check && i > params::CONT_IN_CHECK_MAX_OFFSET() {
                    break;
                }
                // Null move / pre-root padding: no predecessor move to pair with.
                if $frame.current_move.is_ok() {
                    // Copy the key out first, then borrow `self` — no aliasing (§3.2).
                    let id = $frame.cont_hist;
                    let entry = at_piece_to_mut(self.cont_mut(id), q);
                    if entry.get() > 0 {
                        positive_count += 1;
                    }
                    let mult = params::CONT_CONSISTENCY_MULT[positive_count];
                    let scaled = (bonus as i64 * $weight as i64 * mult as i64
                        / params::CONT_UPDATE_DIVISOR()) as i32
                        + if i == 1 {
                            params::CONT_OFFSET1_FLAT_BONUS()
                        } else {
                            0
                        };
                    entry.update(scaled);
                }
            }};
        }

        for (idx, &weight) in params::CONT_OFFSET_WEIGHTS.iter().enumerate() {
            cont_body!(idx, weight, &stack[cur - (idx + 1)]);
        }
    }

    /// Update every quiet table for one quiet move (design/09 §3.9): main
    /// history ×1, low-ply (plies < 5), the continuation chain (×820/1024), and
    /// pawn history (×1038/1024 for bonuses, ×525/1024 for maluses — the split
    /// point is `> -7`, deliberately not `> 0`). `m` is scored in `board`, so
    /// `piece_on(m.from())` is the moving piece.
    pub fn update_quiet_histories(
        &mut self,
        board: &Board,
        stack: &[Stack],
        cur: usize,
        m: Move,
        bonus: i32,
    ) {
        debug_assert!(cur >= SS_OFFSET);
        let us = board.side_to_move();
        let raw = m.raw16() as usize;
        self.main[us as usize][raw].update(bonus);

        let ply = cur - SS_OFFSET;
        if ply < LOW_PLY_SIZE {
            self.low_ply[ply][raw]
                .update(bonus * params::QUIET_LOW_PLY_UPDATE_NUM() / params::STAT_WEIGHT_DEN());
        }

        let pc = board
            .piece_on(m.from())
            .expect("a quiet move starts on a piece of the side to move");
        let to = m.to();
        self.update_continuation_histories(
            stack,
            cur,
            pc,
            to,
            bonus * params::QUIET_CONT_UPDATE_NUM() / params::STAT_WEIGHT_DEN(),
        );

        let pawn_weight = if bonus > params::PAWN_HIST_ASYMMETRY_THRESHOLD() {
            params::PAWN_HIST_BONUS_NUM()
        } else {
            params::PAWN_HIST_MALUS_NUM()
        };
        let slot = (board.pawn_key() & (PAWN_HIST_SIZE as u64 - 1)) as usize;
        self.pawn[slot][pc.index()][to.idx()]
            .update(bonus * pawn_weight / params::STAT_WEIGHT_DEN());
    }

    /// The node-result write burst on a best move that raised alpha
    /// (design/09 §3.9). `quiets_searched`/`captures_searched` are the tried
    /// non-best moves; `prev_sq` the parent move's destination (`None` at the
    /// root or after a null move). Computes one `bonus`/`malus` pair and
    /// distributes them: the best move is rewarded, every tried non-best move
    /// punished (quiets via a geometrically decaying malus, captures flat), and
    /// — when the parent played a refuted first-try quiet — its continuation
    /// entries take an extra malus.
    // icold67: outlined — this burst runs only on nodes that raise alpha
    // with a non-TT best move; inlined it fragments the search's hot fetch
    // span.
    #[cfg_attr(feature = "icold67", inline(never))]
    #[allow(clippy::too_many_arguments)]
    pub fn update_all_stats(
        &mut self,
        board: &Board,
        stack: &[Stack],
        cur: usize,
        best_move: Move,
        prev_sq: Option<Square>,
        quiets_searched: &[Move],
        captures_searched: &[Move],
        depth: Depth,
        tt_move: Move,
        is_pv: bool,
    ) {
        debug_assert!(cur >= SS_OFFSET);

        #[cfg(not(feature = "arm70"))]
        let tt_bonus = params::STAT_BONUS_TTMOVE();
        #[cfg(feature = "arm70")]
        let tt_bonus = match crate::arm70::arm().tt_bonus {
            -1 => params::STAT_BONUS_TTMOVE(),
            v => v,
        };
        let mut bonus = (params::STAT_BONUS_DEPTH_MULT() * depth - params::STAT_BONUS_DEPTH_SUB())
            .min(params::STAT_BONUS_MAX())
            + tt_bonus * (best_move == tt_move) as i32
            + stack[cur - 1].stat_score / params::STAT_BONUS_STATSCORE_DIV();
        let malus = (params::STAT_MALUS_DEPTH_MULT() * depth - params::STAT_MALUS_DEPTH_SUB())
            .min(params::STAT_MALUS_MAX());

        // A wide node that tried many moves before the cutoff taught us more.
        if !is_pv {
            let tried = (quiets_searched.len() + captures_searched.len()) as i64;
            bonus += (bonus as i64 * tried / params::STAT_BREADTH_DIVISOR()) as i32;
        }

        #[cfg(not(feature = "arm70"))]
        let (best_num, malus_start, malus_decay, cross) = (
            params::QUIET_BEST_BONUS_NUM(),
            params::QUIET_MALUS_START_NUM(),
            params::QUIET_MALUS_DECAY_NUM(),
            false,
        );
        #[cfg(feature = "arm70")]
        let (best_num, malus_start, malus_decay, cross) = {
            let a = crate::arm70::arm();
            let pick = |v: i32, d: i32| if v == 0 { d } else { v };
            (
                pick(a.best_bonus, params::QUIET_BEST_BONUS_NUM()),
                pick(a.malus_start, params::QUIET_MALUS_START_NUM()),
                pick(a.malus_decay, params::QUIET_MALUS_DECAY_NUM()),
                a.cross_malus != 0,
            )
        };
        let quiet_malus = |me: &mut Self, board: &Board, stack: &[Stack]| {
            // Geometrically decaying malus: earliest failures punished hardest.
            let mut m = malus * malus_start / params::STAT_WEIGHT_DEN();
            for &q in quiets_searched {
                m = m * malus_decay / params::STAT_WEIGHT_DEN();
                me.update_quiet_histories(board, stack, cur, q, -m);
            }
        };
        let capture_malus = |me: &mut Self, board: &Board| {
            for &c in captures_searched {
                me.capture_entry_mut(board, c)
                    .update(-malus * params::CAPTURE_MALUS_NUM() / params::STAT_WEIGHT_DEN());
            }
        };
        if !board.capture_stage(best_move) {
            self.update_quiet_histories(
                board,
                stack,
                cur,
                best_move,
                bonus * best_num / params::STAT_WEIGHT_DEN(),
            );
            quiet_malus(self, board, stack);
        } else {
            self.capture_entry_mut(board, best_move)
                .update(bonus * params::CAPTURE_BEST_BONUS_NUM() / params::STAT_WEIGHT_DEN());
            // Study 70 arm: the tried quiets also failed at this node, and the
            // shipped code throws that evidence away whenever the best move
            // happened to be a capture.
            if cross {
                quiet_malus(self, board, stack);
            }
        }

        // Refutation malus: the parent's move was a quiet it tried first and
        // that got refuted here (design/09 §3.9). `cur - 1` walks the parent's
        // predecessors, reaching `stack[cur - 7]` — in-bounds by D18.
        if let Some(prev_sq) = prev_sq {
            let prev = &stack[cur - 1];
            if prev.move_count == 1 + prev.tt_hit as i32 && board.captured_piece().is_none() {
                let pc = board
                    .piece_on(prev_sq)
                    .expect("the parent's moved piece sits on its to-square");
                self.update_continuation_histories(
                    stack,
                    cur - 1,
                    pc,
                    prev_sq,
                    -malus * params::REFUTATION_MALUS_NUM() / params::STAT_WEIGHT_DEN(),
                );
            }
        }

        capture_malus(self, board);
    }

    /// The capture-history value for `m` in `board` (read side): the same
    /// `[moved][to][victim]` cell [`Self::capture_entry_mut`] writes, with the
    /// [`NO_VICTIM`] slot for en passant and quiet queen promotions (no piece
    /// on `to`). Read by design/10 §3.5's capture futility / SEE pruning.
    pub fn capture_value(&self, board: &Board, m: Move) -> i32 {
        let moved = board
            .piece_on(m.from())
            .expect("a capture or checking move starts on a piece of the side to move");
        let victim = board
            .piece_on(m.to())
            .map_or(NO_VICTIM, |p| p.piece_type as usize);
        self.capture[moved.index()][m.to().idx()][victim].get()
    }

    /// Flat capture-history reward for the parent's capturing move that caused
    /// a fail low (design/10 §3.10 step 5). The board is already PAST the
    /// parent move, so the cell is addressed by the reconstructed key: `moved`
    /// is the parent's piece (now on `to`), `victim` its captured piece type
    /// (`None` ⇒ the [`NO_VICTIM`] slot, e.g. en passant — keeping the same
    /// victim convention the in-search capture writes use).
    pub fn reward_prior_capture(
        &mut self,
        moved: Piece,
        to: Square,
        victim: Option<PieceType>,
        bonus: i32,
    ) {
        let vi = victim.map_or(NO_VICTIM, |pt| pt as usize);
        self.capture[moved.index()][to.idx()][vi].update(bonus);
    }

    /// The mutable capture-history cell for `m` in `board`: `[moved][to]
    /// [victim]`, with the [`NO_VICTIM`] slot for en passant and quiet queen
    /// promotions (no piece on `to`).
    fn capture_entry_mut(&mut self, board: &Board, m: Move) -> &mut StatEntry<CAPT_D> {
        let moved = board
            .piece_on(m.from())
            .expect("a capture move starts on a piece of the side to move");
        let victim = board
            .piece_on(m.to())
            .map_or(NO_VICTIM, |p| p.piece_type as usize);
        &mut self.capture[moved.index()][m.to().idx()][victim]
    }

    /// A correction-history bundle for `key`'s slot and side `us` (read side).
    #[inline]
    fn corr_slot(&self, key: Key, us: Color) -> &CorrBundle {
        &self.corr[(key & (CORR_SIZE as u64 - 1)) as usize][us as usize]
    }

    /// A correction-history bundle for `key`'s slot and side `us` (write side).
    #[inline]
    fn corr_slot_mut(&mut self, key: Key, us: Color) -> &mut CorrBundle {
        &mut self.corr[(key & (CORR_SIZE as u64 - 1)) as usize][us as usize]
    }

    /// The weighted correction of the raw static eval, in 1/131072-pawn units
    /// (design/09 §3.10). Each correction field is read under its *own*
    /// specialized key; the continuation term sums offsets 2 and 4 (or falls
    /// back to a flat prior after a null previous move). Combine with
    /// [`to_corrected_static_eval`].
    pub fn correction_value(&self, board: &Board, stack: &[Stack], cur: usize) -> i32 {
        debug_assert!(cur >= SS_OFFSET);
        let us = board.side_to_move();
        let pcv = self.corr_slot(board.pawn_key(), us).pawn.get();
        let micv = self.corr_slot(board.minor_key(), us).minor.get();
        let wnp = self
            .corr_slot(board.non_pawn_key(Color::White), us)
            .non_pawn_white
            .get();
        let bnp = self
            .corr_slot(board.non_pawn_key(Color::Black), us)
            .non_pawn_black
            .get();

        let prev = stack[cur - 1].current_move;
        let cntcv = if prev.is_ok() {
            let to = prev.to();
            let pc = board
                .piece_on(to)
                .expect("the previous move's piece sits on its to-square");
            // Two planes, one cell — flat-indexed for the reason in `piece_to`.
            let q = piece_to(pc.index(), to.idx());
            let o2 = stack[cur - 2].cont_hist.piece_to();
            let o4 = stack[cur - 4].cont_hist.piece_to();
            params::CORR_CONT_WEIGHT()
                * (at_piece_to(self.corr_slab(o2), q) + at_piece_to(self.corr_slab(o4), q))
        } else {
            params::CORR_CONT_NULL_PRIOR()
        };

        params::CORR_PAWN_WEIGHT() * pcv
            + params::CORR_MINOR_WEIGHT() * micv
            + params::CORR_NONPAWN_WEIGHT() * (wnp + bnp)
            + cntcv
    }

    /// Fold the learned correction back into the raw eval (design/09 §3.10):
    /// the caller-computed, gated bonus (see [`correction_bonus`]) is
    /// distributed across the pawn/minor/non-pawn and — when the previous move
    /// is real — the two continuation-correction tables.
    pub fn update_correction_history(
        &mut self,
        board: &Board,
        stack: &[Stack],
        cur: usize,
        bonus: i32,
    ) {
        debug_assert!(cur >= SS_OFFSET);
        let us = board.side_to_move();
        self.corr_slot_mut(board.pawn_key(), us).pawn.update(bonus);
        self.corr_slot_mut(board.minor_key(), us)
            .minor
            .update(bonus * params::CORR_MINOR_UPDATE_NUM() / params::CORR_UPDATE_DEN());
        self.corr_slot_mut(board.non_pawn_key(Color::White), us)
            .non_pawn_white
            .update(bonus * params::CORR_NONPAWN_UPDATE_NUM() / params::CORR_UPDATE_DEN());
        self.corr_slot_mut(board.non_pawn_key(Color::Black), us)
            .non_pawn_black
            .update(bonus * params::CORR_NONPAWN_UPDATE_NUM() / params::CORR_UPDATE_DEN());

        let prev = stack[cur - 1].current_move;
        if prev.is_ok() {
            let to = prev.to();
            let pc = board
                .piece_on(to)
                .expect("the previous move's piece sits on its to-square");
            let (pci, toi) = (pc.index(), to.idx());
            let o2 = stack[cur - 2].cont_hist.piece_to();
            let o4 = stack[cur - 4].cont_hist.piece_to();
            let c2 = bonus * params::CORR_CONT2_UPDATE_NUM() / params::CORR_UPDATE_DEN();
            let c4 = bonus * params::CORR_CONT4_UPDATE_NUM() / params::CORR_UPDATE_DEN();
            self.corr_slab_mut(o2)[pci][toi].update(c2);
            self.corr_slab_mut(o4)[pci][toi].update(c4);
        }
    }
}

/// Fold the correction sum into the raw static eval and keep the result inside
/// the non-tablebase band (design/09 §3.10): `clamp(raw + cv/131072, just
/// inside TB-loss..TB-win)`. The **raw** eval is what the TT stores; the
/// corrected one drives pruning (D22).
pub fn to_corrected_static_eval(raw: Value, cv: i32) -> Value {
    (raw + cv / params::CORR_VALUE_DIVISOR())
        .clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}

/// The gated, clamped correction-history bonus (design/10 §4.6, colocated with
/// its callee per design/09 §3.9). Returns `None` when the update must not
/// fire: in check, when the best move is a real capture, or when the eval-error
/// direction disagrees with whether a best move was found. Otherwise the
/// depth-scaled error is clamped to `±256` and scaled by `1114/1024`, ready for
/// [`Histories::update_correction_history`].
pub fn correction_bonus(
    best_value: Value,
    static_eval: Value,
    in_check: bool,
    best_move: Move,
    best_is_capture: bool,
    depth: Depth,
) -> Option<i32> {
    if in_check || (best_move.is_ok() && best_is_capture) {
        return None;
    }
    // The error direction must match whether we actually found a best move.
    if (best_value > static_eval) != best_move.is_ok() {
        return None;
    }
    let depth_mult = if best_move.is_ok() {
        params::CORR_BONUS_DEPTH_MULT_BESTMOVE()
    } else {
        params::CORR_BONUS_DEPTH_MULT_NOMOVE()
    };
    let raw = ((best_value - static_eval) * depth * depth_mult / params::CORR_BONUS_DIVISOR())
        .clamp(-params::CORR_BONUS_CLAMP(), params::CORR_BONUS_CLAMP());
    Some(raw * params::CORR_BONUS_SCALE_NUM() / params::CORR_BONUS_SCALE_DEN())
}

// Compile-time guards on the catalogue (design/09 §8 criterion 2).
const _: () = assert!(size_of::<StatEntry<MAIN_D>>() == 2);
const _: () = assert!(size_of::<CorrBundle>() == 8);
const _: () = assert!(size_of::<PieceToHist>() == 12 * 64 * 2); // 1.5 KiB
const _: () = assert!(size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; 2]>() == 256 * 1024);
const _: () = assert!(size_of::<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>() == 12 * 64 * 7 * 2);
const _: () = assert!(CONT_NB == 4 * 13 * 64);
const _: () = assert!(CONT_CORR_NB == 13 * 64);
const _: () = assert!(CONT_NB == CONT_CORR_NB * QUADRANT_NB);
// `piece_to()`'s divisor is the LOW axis span, and it must be a power of two
// or the accessor stops being a shift (perf study 16 §5).
const _: () = assert!(QUADRANT_NB.is_power_of_two());
// The sentinel slab must be a real, in-bounds continuation index.
const _: () = assert!(ContHistId::SENTINEL.0 == (12 * 64 * QUADRANT_NB) as u16);
const _: () = assert!((ContHistId::SENTINEL.0 as usize) < CONT_NB);

/// Total heap footprint of one [`Histories`] (design/09 §8 criterion 2:
/// ≤ 21 MiB). ~20 MiB in practice.
const HISTORIES_BYTES: usize = size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; 2]>()
    + size_of::<[[StatEntry<MAIN_D>; BUTTERFLY_SIZE]; LOW_PLY_SIZE]>()
    + size_of::<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>()
    + size_of::<[PieceToHist; CONT_ALLOC]>()
    + size_of::<[[[StatEntry<PAWN_D>; 64]; 12]; PAWN_HIST_SIZE]>()
    + size_of::<[[CorrBundle; 2]; CORR_SIZE]>()
    // `CONT_CORR_ALLOC`, not `CONT_CORR_NB` — the budget must count what is
    // ALLOCATED (that table is padded to a power of two).
    + size_of::<[PieceToCorr; CONT_CORR_ALLOC]>()
    + size_of::<StatEntry<TT_MOVE_D>>();
// The catalogue is ~20 MiB: comfortably under the 21 MiB budget (§8), and
// well over 19 MiB (a guard against a table silently shrinking away).
/// design/09 §8 criterion 2.
const HISTORIES_BUDGET: usize = 21 * 1024 * 1024;
const _: () = assert!(HISTORIES_BYTES <= HISTORIES_BUDGET);
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

    // Bound invariant: over an adversarial stream of bonuses and random
    // re-seeds, the value never escapes [-D, D].
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

    // A fixed sub-clamp bonus drives the cell monotonically toward the
    // matching bound and eventually lands on it (the fixed point is ±D).
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

    // `ContHistId::new` is a BIJECTION onto `0..CONT_NB`, and `piece_to()`
    // recovers the (piece, to) sub-index of every id. Deliberately written
    // without asserting the axis ORDER — that is a free implementation
    // choice; a collision or an out-of-range id is what corrupts the tensor,
    // and `seen` catches both.
    #[test]
    fn cont_hist_id_is_the_flat_index_bijection() {
        let mut seen = [false; CONT_NB];
        let mut n = 0usize;
        for in_check in [false, true] {
            for capture in [false, true] {
                for pc in 0..13usize {
                    let piece = piece_for_slot(pc);
                    for to in 0..64u8 {
                        let id = ContHistId::new(in_check, capture, piece, Square::new(to));
                        let flat = id.0 as usize;
                        assert!(flat < CONT_NB, "id {flat} out of range");
                        assert!(!seen[flat], "flat index {flat} collides");
                        seen[flat] = true;
                        assert_eq!(id.piece_to(), pc * 64 + to as usize, "piece_to off");
                        assert!(id.piece_to() < CONT_CORR_NB);
                        n += 1;
                    }
                }
            }
        }
        assert_eq!(n, CONT_NB, "sweep did not cover exactly CONT_NB ids");
        assert!(seen.iter().all(|&s| s), "sweep left a slab unreachable");
    }

    // SENTINEL is the real, in-bounds `[false][false][None][A1]` slab (3072),
    // distinct from any real move's slab; reading through it is inert.
    #[test]
    fn cont_hist_id_sentinel_is_no_piece_a1() {
        assert_eq!(
            ContHistId::SENTINEL,
            ContHistId::new(false, false, None, Square::A1)
        );
        assert_eq!(ContHistId::SENTINEL.0, 3072);
        assert!((ContHistId::SENTINEL.0 as usize) < CONT_NB);
        assert_eq!(ContHistId::SENTINEL.piece_to(), 12 * 64);

        // No real move (piece index 0..12) can produce the sentinel slab.
        for pc in 0..12usize {
            let id = ContHistId::new(false, false, piece_for_slot(pc), Square::A1);
            assert_ne!(id, ContHistId::SENTINEL);
        }
    }

    /// Every multi-row history table must START on a cache line. A GATE, not
    /// a comment: the property is invisible in any behavioural test and is
    /// re-broken by simply dropping the [`Line64`] wrapper from one field.
    #[test]
    fn every_history_table_starts_on_a_cache_line() {
        let h = Histories::new();
        for (name, base) in [
            ("main", h.main.as_ptr() as usize),
            ("low_ply", h.low_ply.as_ptr() as usize),
            ("capture", h.capture.as_ptr() as usize),
            ("cont", h.cont.as_ptr() as usize),
            ("pawn", h.pawn.as_ptr() as usize),
            ("cont_corr", h.cont_corr.as_ptr() as usize),
        ] {
            assert_eq!(base % 64, 0, "{name} base is {} mod 64", base % 64);
        }
    }

    /// The wrapper must cost nothing: every wrapped row is already a whole
    /// number of cache lines, so `Line64` may not add a single padding byte.
    #[test]
    fn line64_adds_no_padding() {
        assert_eq!(
            size_of::<Line64<PieceToHist>>(),
            size_of::<PieceToHist>(),
            "Line64 padded PieceToHist"
        );
        assert_eq!(
            size_of::<Line64<[StatEntry<MAIN_D>; BUTTERFLY_SIZE]>>(),
            size_of::<[StatEntry<MAIN_D>; BUTTERFLY_SIZE]>(),
            "Line64 padded a butterfly plane"
        );
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

    // ---- update functions + correction history: golden tests ----
    //
    // Hand-computed post-update cells for known bonus/malus sequences.
    // Poking start cells to 0 makes gravity-from-0 the identity, so a cell
    // ends at exactly the (weighted, truncated) bonus argument.

    use crate::board::Board;
    use crate::moves::Move;
    use crate::search::stack::{SS_OFFSET, Stack};

    // A real (is_ok) placeholder predecessor move.
    fn real_move() -> Move {
        Move::normal(Square::A1, Square::B1)
    }

    // Wire six distinct predecessor slabs at `stack[cur-1..=cur-6]`, each with a
    // real move and its target `[pc][to]` cell poked to `start`. Returns the ids
    // in offset order.
    fn wire_cont_chain(
        h: &mut Histories,
        stack: &mut [Stack],
        cur: usize,
        pci: usize,
        toi: usize,
        start: i16,
    ) -> Vec<ContHistId> {
        let filler = Some(Piece::new(Color::White, PieceType::Pawn));
        (1..=6)
            .map(|i| {
                let id = ContHistId::new(false, false, filler, Square::new(i as u8));
                stack[cur - i].current_move = real_move();
                stack[cur - i].cont_hist = id;
                h.cont_mut(id)[pci][toi] = StatEntry::new(start);
                id
            })
            .collect()
    }

    // §3.9: offset weights {1040,780,300,537,129,423}, consistency mult 96 at
    // count 0 (no pre-positive entries), flat +71 at offset 1, ÷131072.
    #[test]
    fn cont_update_offset_weights_and_flat_bonus() {
        let mut h = Histories::new();
        let cur = SS_OFFSET;
        let mut stack = vec![Stack::default(); cur + 1];
        let pc = Piece::new(Color::White, PieceType::Knight);
        let (pci, toi) = (pc.index(), Square::E4.index() as usize);
        let ids = wire_cont_chain(&mut h, &mut stack, cur, pci, toi, 0);

        h.update_continuation_histories(&stack, cur, pc, Square::E4, 1024);

        let want = [851, 585, 225, 402, 96, 317];
        for (idx, id) in ids.iter().enumerate() {
            assert_eq!(
                h.cont_ref(*id)[pci][toi].get(),
                want[idx],
                "offset {}",
                idx + 1
            );
        }
    }

    // §3.9: in check, only offsets 1-2 update; 3..6 keep their poked start.
    #[test]
    fn cont_update_in_check_truncates_to_offsets_one_two() {
        let mut h = Histories::new();
        let cur = SS_OFFSET;
        let mut stack = vec![Stack::default(); cur + 1];
        stack[cur].in_check = true;
        let pc = Piece::new(Color::White, PieceType::Knight);
        let (pci, toi) = (pc.index(), Square::E4.index() as usize);
        let ids = wire_cont_chain(&mut h, &mut stack, cur, pci, toi, 0);

        h.update_continuation_histories(&stack, cur, pc, Square::E4, 1024);

        let want = [851, 585, 0, 0, 0, 0];
        for (idx, id) in ids.iter().enumerate() {
            assert_eq!(
                h.cont_ref(*id)[pci][toi].get(),
                want[idx],
                "offset {}",
                idx + 1
            );
        }
    }

    // §3.9: with every predecessor cell already positive, positive_count reaches
    // `i` at offset `i`, so the consistency multiplier is CMHC[i] — an agreeing
    // chain amplifies the update. Gravity from +1 just adds the arg (the decay
    // term truncates to 0 for these magnitudes).
    #[test]
    fn cont_update_consistency_multiplier_amplifies_agreeing_chain() {
        let mut h = Histories::new();
        let cur = SS_OFFSET;
        let mut stack = vec![Stack::default(); cur + 1];
        let pc = Piece::new(Color::White, PieceType::Knight);
        let (pci, toi) = (pc.index(), Square::E4.index() as usize);
        let ids = wire_cont_chain(&mut h, &mut stack, cur, pci, toi, 1);

        h.update_continuation_histories(&stack, cur, pc, Square::E4, 1024);

        // CMHC = {_,113,101,105,127,121,126} at counts 1..6.
        let want = [990, 616, 247, 533, 122, 417];
        for (idx, id) in ids.iter().enumerate() {
            assert_eq!(
                h.cont_ref(*id)[pci][toi].get(),
                want[idx],
                "offset {}",
                idx + 1
            );
        }
    }

    const PAWN_FEN: &str = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";

    // §3.9 update_quiet_histories: main ×1, low-ply ×663/1024 (ply < 5),
    // pawn ×1038/1024 (bonus). Null predecessors ⇒ the cont chain is skipped.
    #[test]
    fn quiet_update_writes_all_tables() {
        let mut h = Histories::new();
        let b = Board::from_fen(PAWN_FEN).unwrap();
        let m = Move::normal(Square::E2, Square::E3);
        let pc = b.piece_on(Square::E2).unwrap();
        let ply = 2usize;
        let cur = SS_OFFSET + ply;
        let stack = vec![Stack::default(); cur + 1];
        let us = b.side_to_move() as usize;
        let raw = m.raw16() as usize;
        let slot = (b.pawn_key() & (PAWN_HIST_SIZE as u64 - 1)) as usize;
        let (pci, toi) = (pc.index(), Square::E3.index() as usize);

        h.main[us][raw] = StatEntry::new(0);
        h.low_ply[ply][raw] = StatEntry::new(0);
        h.pawn[slot][pci][toi] = StatEntry::new(0);

        h.update_quiet_histories(&b, &stack, cur, m, 2048);

        assert_eq!(h.main[us][raw].get(), 2048); // ×1
        assert_eq!(h.low_ply[ply][raw].get(), 1326); // 2048·663/1024
        assert_eq!(h.pawn[slot][pci][toi].get(), 2076); // 2048·1038/1024
    }

    // §3.9: the pawn-history bonus/malus asymmetry pivots at `bonus > -7`.
    #[test]
    fn quiet_update_pawn_history_bonus_malus_asymmetry() {
        let b = Board::from_fen(PAWN_FEN).unwrap();
        let m = Move::normal(Square::E2, Square::E3);
        let pc = b.piece_on(Square::E2).unwrap();
        let cur = SS_OFFSET + 2;
        let stack = vec![Stack::default(); cur + 1];
        let slot = (b.pawn_key() & (PAWN_HIST_SIZE as u64 - 1)) as usize;
        let (pci, toi) = (pc.index(), Square::E3.index() as usize);

        let pawn_after = |bonus: i32| -> i32 {
            let mut h = Histories::new();
            h.pawn[slot][pci][toi] = StatEntry::new(0);
            h.update_quiet_histories(&b, &stack, cur, m, bonus);
            h.pawn[slot][pci][toi].get()
        };

        assert_eq!(pawn_after(-2048), -1050); // malus weight 525: -2048·525/1024
        assert_eq!(pawn_after(-7), -3); //  -7 is NOT > -7 ⇒ 525: -7·525/1024
        assert_eq!(pawn_after(-6), -6); //  -6 IS  > -7 ⇒ 1038: -6·1038/1024
    }

    // §3.9 update_all_stats: best-quiet bonus (×824/1024) and the geometrically
    // decaying malus over tried quiets (start ×1136/1024, ×956/1024 per move).
    #[test]
    fn all_stats_best_quiet_and_geometric_malus() {
        let mut h = Histories::new();
        let b = Board::from_fen("4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1").unwrap();
        let cur = SS_OFFSET;
        let stack = vec![Stack::default(); cur + 1]; // stat_score 0
        let us = b.side_to_move() as usize;
        let bq = Move::normal(Square::A2, Square::A3);
        let quiets = [
            Move::normal(Square::B2, Square::B3),
            Move::normal(Square::C2, Square::C3),
            Move::normal(Square::D2, Square::D3),
        ];
        for m in std::iter::once(bq).chain(quiets) {
            h.main[us][m.raw16() as usize] = StatEntry::new(0);
        }

        // depth 8, PV node (no breadth), tt_move ≠ best: bonus 993, malus 2218.
        h.update_all_stats(&b, &stack, cur, bq, None, &quiets, &[], 8, Move::NULL, true);

        assert_eq!(h.main[us][bq.raw16() as usize].get(), 799); // 993·824/1024
        // 2218·1136/1024 = 2460, then ×956/1024 before each use.
        assert_eq!(h.main[us][quiets[0].raw16() as usize].get(), -2296);
        assert_eq!(h.main[us][quiets[1].raw16() as usize].get(), -2143);
        assert_eq!(h.main[us][quiets[2].raw16() as usize].get(), -2000);
        // Ratio pinned: each successive malus ≈ 956/1024 of the last.
        assert_eq!(2143 * 1024 / 2296, 955); // ≈ 956/1024 (truncation)
    }

    // §3.9: best-capture bonus (×1366/1024) and non-best-capture malus
    // (×1518/1024), addressed by [moved][to][victim].
    #[test]
    fn all_stats_best_capture_and_nonbest_capture_malus() {
        let mut h = Histories::new();
        let b = Board::from_fen("1n1qk3/8/8/8/8/8/1N1Q4/4K3 w - - 0 1").unwrap();
        let cur = SS_OFFSET;
        let stack = vec![Stack::default(); cur + 1];
        let best = Move::normal(Square::D2, Square::D8); // Qxd8: moved Q, victim Q
        let cap = Move::normal(Square::B2, Square::B8); // Nxb8: moved N, victim N
        assert!(b.capture_stage(best));
        let wq = Piece::new(Color::White, PieceType::Queen).index();
        let wn = Piece::new(Color::White, PieceType::Knight).index();
        let (d8, b8) = (Square::D8.index() as usize, Square::B8.index() as usize);
        h.capture[wq][d8][PieceType::Queen as usize] = StatEntry::new(0);
        h.capture[wn][b8][PieceType::Knight as usize] = StatEntry::new(0);

        h.update_all_stats(
            &b,
            &stack,
            cur,
            best,
            None,
            &[],
            &[cap],
            8,
            Move::NULL,
            true,
        );

        assert_eq!(h.capture[wq][d8][PieceType::Queen as usize].get(), 1324); // 993·1366/1024
        assert_eq!(h.capture[wn][b8][PieceType::Knight as usize].get(), -3288); // -2218·1518/1024
    }

    // §3.9: the refutation malus fires only when the parent played its first
    // move (`move_count == 1 + tt_hit`) as an uncaptured quiet, and never
    // without a parent square. A capture best move keeps the only continuation
    // write to the refutation itself.
    #[test]
    fn all_stats_refutation_malus_gated_on_first_try_quiet() {
        let b = Board::from_fen("3qk3/1n6/8/8/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let best = Move::normal(Square::D2, Square::D8); // capture ⇒ no best cont walk
        assert!(b.capture_stage(best));
        let prev_sq = Square::B7;
        let refuted_pc = b.piece_on(prev_sq).unwrap(); // black knight
        let (rpci, rto) = (refuted_pc.index(), prev_sq.index() as usize);
        let cur = SS_OFFSET + 2; // ply 2

        // Only stack[cur-2] (offset 1 from the parent) carries a real move.
        let refut_id = ContHistId::new(
            true,
            false,
            Some(Piece::new(Color::White, PieceType::Rook)),
            Square::H1,
        );
        let build = |move_count: i32| {
            let mut stack = vec![Stack::default(); cur + 1];
            stack[cur - 1].move_count = move_count;
            stack[cur - 2].current_move = real_move();
            stack[cur - 2].cont_hist = refut_id;
            stack
        };
        let run = |stack: &[Stack], prev: Option<Square>| -> i32 {
            let mut h = Histories::new();
            h.cont_mut(refut_id)[rpci][rto] = StatEntry::new(0);
            h.update_all_stats(&b, stack, cur, best, prev, &[], &[], 8, Move::NULL, true);
            h.cont_ref(refut_id)[rpci][rto].get()
        };

        // Gate satisfied (move_count 1 == 1 + tt_hit 0): -malus·683/1024 = -1479,
        // then the offset-1 cont update = -1479·1040·96/131072 + 71 = -1055.
        assert_eq!(run(&build(1), Some(prev_sq)), -1055);
        // Parent tried more than one move ⇒ no refutation write.
        assert_eq!(run(&build(2), Some(prev_sq)), 0);
        // No parent square ⇒ no refutation write.
        assert_eq!(run(&build(1), None), 0);
    }

    // §3.10 correction_value: the linear combiner over the six inputs, with a
    // real previous move so the continuation term (×8363) is read.
    #[test]
    fn correction_value_combiner_matches_hand_computed() {
        let mut h = Histories::new();
        let b = Board::from_fen(crate::fen::START_FEN).unwrap();
        let us = b.side_to_move();
        let cur = SS_OFFSET + 4;
        let mut stack = vec![Stack::default(); cur + 1];
        let prev_to = Square::B1; // a white knight sits here in the start position
        let prev_pc = b.piece_on(prev_to).unwrap();
        stack[cur - 1].current_move = Move::normal(Square::G1, prev_to);
        let id2 = ContHistId::new(
            false,
            false,
            Some(Piece::new(Color::White, PieceType::Knight)),
            prev_to,
        );
        let id4 = ContHistId::new(
            true,
            false,
            Some(Piece::new(Color::Black, PieceType::Queen)),
            Square::D8,
        );
        stack[cur - 2].cont_hist = id2;
        stack[cur - 4].cont_hist = id4;
        assert_ne!(id2.piece_to(), id4.piece_to());

        h.corr_slot_mut(b.pawn_key(), us).pawn = StatEntry::new(3);
        h.corr_slot_mut(b.minor_key(), us).minor = StatEntry::new(-5);
        h.corr_slot_mut(b.non_pawn_key(Color::White), us)
            .non_pawn_white = StatEntry::new(7);
        h.corr_slot_mut(b.non_pawn_key(Color::Black), us)
            .non_pawn_black = StatEntry::new(-11);
        let (pci, toi) = (prev_pc.index(), prev_to.index() as usize);
        h.cont_corr[id2.piece_to()][pci][toi] = StatEntry::new(4);
        h.cont_corr[id4.piece_to()][pci][toi] = StatEntry::new(6);

        let want = 13345 * 3 + 9280 * -5 + 11840 * (7 + -11) + 8363 * (4 + 6);
        assert_eq!(h.correction_value(&b, &stack, cur), want);
    }

    // §3.10: after a null previous move the continuation term is the flat prior.
    #[test]
    fn correction_value_null_prev_uses_flat_prior() {
        let h = Histories::new();
        let b = Board::from_fen(crate::fen::START_FEN).unwrap();
        let us = b.side_to_move();
        let cur = SS_OFFSET + 4;
        let stack = vec![Stack::default(); cur + 1]; // stack[cur-1].current_move = NULL

        let pcv = h.corr_slot(b.pawn_key(), us).pawn.get();
        let micv = h.corr_slot(b.minor_key(), us).minor.get();
        let wnp = h
            .corr_slot(b.non_pawn_key(Color::White), us)
            .non_pawn_white
            .get();
        let bnp = h
            .corr_slot(b.non_pawn_key(Color::Black), us)
            .non_pawn_black
            .get();
        let want = 13345 * pcv + 9280 * micv + 11840 * (wnp + bnp) + 64549;
        assert_eq!(h.correction_value(&b, &stack, cur), want);
    }

    // §3.10 to_corrected_static_eval: add cv/131072, clamp into the non-TB band.
    #[test]
    fn to_corrected_static_eval_adds_and_clamps() {
        assert_eq!(to_corrected_static_eval(50, 131072), 51); // +1
        assert_eq!(to_corrected_static_eval(50, -262144), 48); // -2
        assert_eq!(to_corrected_static_eval(50, 131071), 50); // truncates to +0
        assert_eq!(
            to_corrected_static_eval(VALUE_TB_WIN_IN_MAX_PLY, 131072 * 1000),
            VALUE_TB_WIN_IN_MAX_PLY - 1
        );
        assert_eq!(
            to_corrected_static_eval(VALUE_TB_LOSS_IN_MAX_PLY, -131072 * 1000),
            VALUE_TB_LOSS_IN_MAX_PLY + 1
        );
    }

    // §3.10 / doc 10 §4.6: the direction-consistency gate and clamped bonus.
    #[test]
    fn correction_bonus_respects_direction_gate() {
        let real = Move::normal(Square::E2, Square::E4);
        // Best move + improved eval: fires. err 40, depth 8 ⇒ 40·8·12/128 = 30,
        // then 1114·30/1024 = 32.
        assert_eq!(correction_bonus(100, 60, false, real, false, 8), Some(32));
        // Best move but eval fell ⇒ direction mismatch.
        assert_eq!(correction_bonus(50, 60, false, real, false, 8), None);
        // Fail-low (no best move) with a fallen eval: directions agree ⇒ fires.
        assert_eq!(
            correction_bonus(40, 60, false, Move::NULL, false, 8),
            Some(-23)
        );
        // Fail-low but eval rose ⇒ mismatch.
        assert_eq!(correction_bonus(80, 60, false, Move::NULL, false, 8), None);
        // In check ⇒ never fires.
        assert_eq!(correction_bonus(100, 60, true, real, false, 8), None);
        // Best move is a real capture ⇒ never fires.
        assert_eq!(correction_bonus(100, 60, false, real, true, 8), None);
        // The raw bonus is clamped to ±256 before scaling: 1114·256/1024 = 278.
        assert_eq!(correction_bonus(5000, 0, false, real, false, 20), Some(278));
    }

    // §3.10 update_correction_history: per-table weights over pawn/minor/
    // non-pawn and the two continuation-correction tables.
    #[test]
    fn correction_update_writes_all_tables() {
        let mut h = Histories::new();
        let b = Board::from_fen(crate::fen::START_FEN).unwrap();
        let us = b.side_to_move();
        let cur = SS_OFFSET + 4;
        let mut stack = vec![Stack::default(); cur + 1];
        let prev_to = Square::B1;
        let prev_pc = b.piece_on(prev_to).unwrap();
        stack[cur - 1].current_move = Move::normal(Square::G1, prev_to);
        let id2 = ContHistId::new(
            false,
            false,
            Some(Piece::new(Color::White, PieceType::Knight)),
            prev_to,
        );
        let id4 = ContHistId::new(
            true,
            false,
            Some(Piece::new(Color::Black, PieceType::Queen)),
            Square::D8,
        );
        stack[cur - 2].cont_hist = id2;
        stack[cur - 4].cont_hist = id4;
        assert_ne!(id2.piece_to(), id4.piece_to());

        h.corr_slot_mut(b.pawn_key(), us).pawn = StatEntry::new(0);
        h.corr_slot_mut(b.minor_key(), us).minor = StatEntry::new(0);
        h.corr_slot_mut(b.non_pawn_key(Color::White), us)
            .non_pawn_white = StatEntry::new(0);
        h.corr_slot_mut(b.non_pawn_key(Color::Black), us)
            .non_pawn_black = StatEntry::new(0);
        let (pci, toi) = (prev_pc.index(), prev_to.index() as usize);
        h.cont_corr[id2.piece_to()][pci][toi] = StatEntry::new(0);
        h.cont_corr[id4.piece_to()][pci][toi] = StatEntry::new(0);

        h.update_correction_history(&b, &stack, cur, 512);

        assert_eq!(h.corr_slot(b.pawn_key(), us).pawn.get(), 512); // ×1
        assert_eq!(h.corr_slot(b.minor_key(), us).minor.get(), 608); // ·152/128
        assert_eq!(
            h.corr_slot(b.non_pawn_key(Color::White), us)
                .non_pawn_white
                .get(),
            744
        ); // ·186/128
        assert_eq!(
            h.corr_slot(b.non_pawn_key(Color::Black), us)
                .non_pawn_black
                .get(),
            744
        );
        assert_eq!(h.cont_corr[id2.piece_to()][pci][toi].get(), 544); // ·136/128
        assert_eq!(h.cont_corr[id4.piece_to()][pci][toi].get(), 272); // ·68/128
    }
}
