//! Precomputed attack tables (design/03, D12/D13).
//!
//! This module answers exactly one question: *which squares does a piece on
//! `sq` attack given occupancy `occ`?* Legality (pins, checks, castling) is
//! movegen's job. Const-built leaper/geometry tables, plus the fancy-magic
//! slider machinery behind `LazyLock<AttackTables>` — hoist a [`tables`]
//! handle in hot loops.
//!
//! All tables here are `static`, not `const` (one copy in `.rodata`, never
//! inlined per use site). Const builders operate on raw `u64` — operator
//! traits are not `const` — and wrap into [`Bitboard`] at the boundary.

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::types::{Color, PieceType, Square};

// --- Const foundations: safe stepping + the naive ray oracle ----------------

/// Bit of the target square, or 0 if the step leaves the board or wraps —
/// a const wrapper over [`Square::try_offset`] for the u64-domain builders.
const fn safe_destination(sq: i32, step: i32) -> u64 {
    debug_assert!(0 <= sq && sq < 64);
    match Square::new(sq as u8).try_offset(step as i8) {
        Some(to) => 1u64 << to.index(),
        None => 0,
    }
}

/// Rook direction step deltas on the LERF mapping (D3): N, S, E, W.
const ROOK_DIRS: [i32; 4] = [8, -8, 1, -1];
/// Bishop direction step deltas: NE, NW, SE, SW.
const BISHOP_DIRS: [i32; 4] = [9, 7, -7, -9];

/// Walk each direction from `sq` until the board edge or the first blocker,
/// which is included (capture semantics).
const fn sliding_attack(dirs: [i32; 4], sq: u8, occupied: u64) -> u64 {
    let mut attacks = 0u64;
    let mut d = 0;
    while d < 4 {
        let mut s = sq as i32;
        loop {
            let dest = safe_destination(s, dirs[d]);
            if dest == 0 {
                break;
            }
            attacks |= dest;
            s += dirs[d];
            if occupied & dest != 0 {
                break;
            }
        }
        d += 1;
    }
    attacks
}

/// Naive rook attacks: the module's correctness oracle — the geometry
/// tables are defined in terms of it and the magic init verifies against it.
pub(crate) const fn rook_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(ROOK_DIRS, sq, occ)
}

/// Naive bishop attacks; see [`rook_attacks_naive`].
pub(crate) const fn bishop_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(BISHOP_DIRS, sq, occ)
}

// --- Leaper tables ----------------------------------------------------------

/// Knight step deltas (same on any LERF board).
const KNIGHT_STEPS: [i32; 8] = [-17, -15, -10, -6, 6, 10, 15, 17];
/// King step deltas.
const KING_STEPS: [i32; 8] = [-9, -8, -7, -1, 1, 7, 8, 9];

/// Union of the (wrap-checked) single steps from `sq`.
const fn leaper_attack(steps: [i32; 8], sq: u8) -> u64 {
    let mut b = 0u64;
    let mut i = 0;
    while i < 8 {
        b |= safe_destination(sq as i32, steps[i]);
        i += 1;
    }
    b
}

/// One `[Bitboard; 64]` table from a leaper step set.
const fn build_leaper_table(steps: [i32; 8]) -> [Bitboard; 64] {
    let mut t = [Bitboard::EMPTY; 64];
    let mut s = 0;
    while s < 64 {
        t[s] = Bitboard::from_u64(leaper_attack(steps, s as u8));
        s += 1;
    }
    t
}

/// Knight attacks from each square.
pub static KNIGHT_ATTACKS: [Bitboard; 64] = build_leaper_table(KNIGHT_STEPS);

/// King attacks from each square.
pub static KING_ATTACKS: [Bitboard; 64] = build_leaper_table(KING_STEPS);

/// Set-wise pawn attacks for all `pawns` of `color` at once (`const`, so
/// other const builders can use it).
pub const fn pawn_attacks_bb(color: Color, pawns: Bitboard) -> Bitboard {
    pawns.pawn_attacks(color)
}

/// Attacks FROM `sq` BY a pawn of the indexing color. Defined for all 64
/// squares (entries on the impossible ranks are simply empty).
pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = {
    let mut t = [[Bitboard::EMPTY; 64]; 2];
    let mut s = 0;
    while s < 64 {
        let b = Bitboard::from_u64(1u64 << s);
        t[Color::White as usize][s] = pawn_attacks_bb(Color::White, b);
        t[Color::Black as usize][s] = pawn_attacks_bb(Color::Black, b);
        s += 1;
    }
    t
};

// --- Pseudo (empty-board) slider tables -------------------------------------

/// Empty-board bishop rays from each square.
pub static PSEUDO_BISHOP: [Bitboard; 64] = {
    let mut t = [Bitboard::EMPTY; 64];
    let mut s = 0;
    while s < 64 {
        t[s] = Bitboard::from_u64(bishop_attacks_naive(s as u8, 0));
        s += 1;
    }
    t
};

/// Empty-board rook rays from each square.
pub static PSEUDO_ROOK: [Bitboard; 64] = {
    let mut t = [Bitboard::EMPTY; 64];
    let mut s = 0;
    while s < 64 {
        t[s] = Bitboard::from_u64(rook_attacks_naive(s as u8, 0));
        s += 1;
    }
    t
};

/// Empty-board queen rays from each square: bishop ∪ rook.
pub static PSEUDO_QUEEN: [Bitboard; 64] = {
    let mut t = [Bitboard::EMPTY; 64];
    let mut s = 0;
    while s < 64 {
        t[s] =
            Bitboard::from_u64(bishop_attacks_naive(s as u8, 0) | rook_attacks_naive(s as u8, 0));
        s += 1;
    }
    t
};

// --- Ray geometry: BETWEEN, LINE, RAY_PASS, DISTANCE ------------------------
//
// All four 64×64 tables are built at compile time from the naive generator;
// they are private — consume them through the accessors below.

/// Which slider (if any) aligns `s1` with `s2`. A square is never aligned
/// with itself (empty-board rays exclude their origin).
#[derive(Clone, Copy)]
enum Ray {
    Unaligned,
    Bishop,
    Rook,
}

const fn ray_alignment(s1: usize, s2: usize) -> Ray {
    let bit2 = 1u64 << s2;
    if PSEUDO_BISHOP[s1].to_u64() & bit2 != 0 {
        Ray::Bishop
    } else if PSEUDO_ROOK[s1].to_u64() & bit2 != 0 {
        Ray::Rook
    } else {
        Ray::Unaligned
    }
}

/// `BETWEEN[s1][s2]`: squares strictly between the pair, OR'd with
/// `bit(s2)` **unconditionally** (SF semantics; rationale on [`between`]).
const fn between_entry(s1: usize, s2: usize) -> u64 {
    let (bit1, bit2) = (1u64 << s1, 1u64 << s2);
    let strict = match ray_alignment(s1, s2) {
        Ray::Bishop => bishop_attacks_naive(s1 as u8, bit2) & bishop_attacks_naive(s2 as u8, bit1),
        Ray::Rook => rook_attacks_naive(s1 as u8, bit2) & rook_attacks_naive(s2 as u8, bit1),
        Ray::Unaligned => 0,
    };
    strict | bit2
}

/// `LINE[s1][s2]`: the full edge-to-edge line through both squares plus
/// both endpoints; 0 if unaligned. Symmetric; `LINE[s][s]` is empty.
const fn line_entry(s1: usize, s2: usize) -> u64 {
    let (bit1, bit2) = (1u64 << s1, 1u64 << s2);
    match ray_alignment(s1, s2) {
        Ray::Bishop => {
            (bishop_attacks_naive(s1 as u8, 0) & bishop_attacks_naive(s2 as u8, 0)) | bit1 | bit2
        }
        Ray::Rook => {
            (rook_attacks_naive(s1 as u8, 0) & rook_attacks_naive(s2 as u8, 0)) | bit1 | bit2
        }
        Ray::Unaligned => 0,
    }
}

/// `RAY_PASS[s1][s2]`: the ray from `s1` (exclusive) through `s2` and past
/// it to the board edge; 0 if unaligned.
const fn ray_pass_entry(s1: usize, s2: usize) -> u64 {
    let (bit1, bit2) = (1u64 << s1, 1u64 << s2);
    match ray_alignment(s1, s2) {
        Ray::Bishop => {
            bishop_attacks_naive(s1 as u8, 0) & (bishop_attacks_naive(s2 as u8, bit1) | bit2)
        }
        Ray::Rook => rook_attacks_naive(s1 as u8, 0) & (rook_attacks_naive(s2 as u8, bit1) | bit2),
        Ray::Unaligned => 0,
    }
}

/// Selector for [`build_pair_table`] — const contexts cannot call through
/// `fn` pointers, so the builder dispatches on an enum.
#[derive(Clone, Copy)]
enum PairTable {
    Between,
    Line,
    RayPass,
}

/// The shared 64×64 builder loop for the three square-pair tables.
const fn build_pair_table(which: PairTable) -> [[Bitboard; 64]; 64] {
    let mut t = [[Bitboard::EMPTY; 64]; 64];
    let mut s1 = 0;
    while s1 < 64 {
        let mut s2 = 0;
        while s2 < 64 {
            let b = match which {
                PairTable::Between => between_entry(s1, s2),
                PairTable::Line => line_entry(s1, s2),
                PairTable::RayPass => ray_pass_entry(s1, s2),
            };
            t[s1][s2] = Bitboard::from_u64(b);
            s2 += 1;
        }
        s1 += 1;
    }
    t
}

static BETWEEN: [[Bitboard; 64]; 64] = build_pair_table(PairTable::Between);
static LINE: [[Bitboard; 64]; 64] = build_pair_table(PairTable::Line);
static RAY_PASS: [[Bitboard; 64]; 64] = build_pair_table(PairTable::RayPass);

/// Chebyshev distances, built from [`Square::distance`]; access via [`distance`].
static DISTANCE: [[u8; 64]; 64] = {
    let mut t = [[0u8; 64]; 64];
    let mut s1 = 0;
    while s1 < 64 {
        let mut s2 = 0;
        while s2 < 64 {
            t[s1][s2] = Square::new(s1 as u8).distance(Square::new(s2 as u8));
            s2 += 1;
        }
        s1 += 1;
    }
    t
};

// --- Accessors (all #[inline] table lookups) --------------------------------

/// Squares a knight on `sq` attacks.
#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.idx()]
}

/// Squares a king on `sq` attacks.
#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.idx()]
}

/// Squares a pawn of `color` on `sq` attacks.
#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color as usize][sq.idx()]
}

/// Empty-board attacks for `pt` on `sq`; `pt != Pawn` (use [`pawn_attacks`]).
#[inline]
pub fn pseudo_attacks(pt: PieceType, sq: Square) -> Bitboard {
    debug_assert!(
        !matches!(pt, PieceType::Pawn),
        "use pawn_attacks(color, sq)"
    );
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::King => king_attacks(sq),
        PieceType::Bishop => PSEUDO_BISHOP[sq.idx()],
        PieceType::Rook => PSEUDO_ROOK[sq.idx()],
        PieceType::Queen => PSEUDO_QUEEN[sq.idx()],
        PieceType::Pawn => Bitboard::EMPTY, // unreachable in debug
    }
}

/// Squares strictly between `s1` and `s2`, OR'd with the **target square
/// `s2`**; for unaligned pairs (and `s1 == s2`) exactly `{s2}`. NOT
/// symmetric. The unconditional `| s2` makes `between(king_sq, checker_sq)`
/// directly the "block-or-capture" evasion mask.
#[inline]
pub fn between(s1: Square, s2: Square) -> Bitboard {
    BETWEEN[s1.idx()][s2.idx()]
}

/// The full line through both squares, extended to the board edges and
/// including both endpoints; `EMPTY` if unaligned (incl. `s1 == s2`).
/// Symmetric.
#[inline]
pub fn line(s1: Square, s2: Square) -> Bitboard {
    LINE[s1.idx()][s2.idx()]
}

/// The ray from `s1` (exclusive) through `s2` and past it to the board
/// edge; `EMPTY` if not aligned. `between(s1, s2)` ⊆ `ray_pass(s1, s2)`.
/// Consumer: NNUE discovered-attack rays.
#[inline]
pub fn ray_pass(s1: Square, s2: Square) -> Bitboard {
    RAY_PASS[s1.idx()][s2.idx()]
}

/// Chebyshev distance: `max(|Δfile|, |Δrank|)` — king moves between the
/// squares. Table-backed.
#[inline]
pub fn distance(s1: Square, s2: Square) -> u8 {
    DISTANCE[s1.idx()][s2.idx()]
}

/// True if `s3` lies on the extended line through `s1` and `s2` — the
/// legality workhorse ("does this move stay on the pin ray").
#[inline]
pub fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    line(s1, s2).contains(s3)
}

// --- Prng: xorshift64* ------------------------------------------------------

/// Xorshift64\* PRNG. `pub(crate)` + `const fn` because the Zobrist table
/// builder reuses it in const context; the magic search calls it at runtime.
pub(crate) struct Prng {
    s: u64,
}

impl Prng {
    pub(crate) const fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64* has an all-zero fixed point");
        Prng { s: seed }
    }

    pub(crate) const fn rand64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// AND of three draws (~8 bits set). Good magics are empirically
    /// sparse; sampling sparse candidates converges orders of magnitude faster.
    pub(crate) const fn sparse_rand(&mut self) -> u64 {
        self.rand64() & self.rand64() & self.rand64()
    }
}

// --- Fancy magic bitboards: data structures ---------------------------------
//
// The magic lookup answers `slider_attacks(sq, occ)` with: mask the occupancy
// to the relevant blocker squares, multiply by a per-square magic constant so
// the scattered mask bits collect into the top bits, shift down, index a
// per-square subtable:
//
//     index   = ((occ & mask) * magic) >> shift     // wrapping 64-bit mul
//     shift   = 64 - popcount(mask)
//     attacks = table[offset + index]
//
// "Fancy": each square's subtable has its own size 2^popcount(mask), packed
// back-to-back in one shared array.

/// Entries in the rook block of the packed shared table (800 KB).
pub const ROOK_TABLE_SIZE: usize = 0x19000; // 102_400
/// Entries in the bishop block of the packed shared table (41 KB).
pub const BISHOP_TABLE_SIZE: usize = 0x1480; // 5_248
/// Rook block occupies `table[0..ROOK_TABLE_SIZE]`, bishop block the rest.
const TABLE_SIZE: usize = ROOK_TABLE_SIZE + BISHOP_TABLE_SIZE;

/// Per-square, per-slider magic metadata. A `u32` offset into the shared
/// table instead of an absolute pointer: no self-referential struct, no
/// `unsafe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct Magic {
    /// Relevant-blocker mask (board edges excluded).
    mask: u64,
    /// Multiplier found by the seeded search at init.
    magic: u64,
    /// This square's block start within [`AttackTables::table`].
    offset: u32,
    /// `64 - popcount(mask)`.
    shift: u32,
}

impl Magic {
    const ZEROED: Magic = Magic {
        mask: 0,
        magic: 0,
        offset: 0,
        shift: 0,
    };

    /// The packed-table slot holding the attack set for `occupied`.
    #[inline]
    fn index(&self, occupied: Bitboard) -> usize {
        self.offset as usize
            + ((((occupied.to_u64() & self.mask).wrapping_mul(self.magic)) >> self.shift) as usize)
    }
}

/// Both magics for one square packed into exactly one cache line: one line
/// fetch serves both slider types, which queen lookups always want.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MagicPair {
    bishop: Magic, // 24 bytes
    rook: Magic,   // 24 bytes, 16 bytes tail padding
}

const _: () = assert!(size_of::<MagicPair>() == 64);
const _: () = assert!(align_of::<MagicPair>() == 64);

impl MagicPair {
    const ZEROED: MagicPair = MagicPair {
        bishop: Magic::ZEROED,
        rook: Magic::ZEROED,
    };
}

/// Allocation length of the shared attack table, rounded up to a power of
/// two so `slot`'s index mask is a provable no-op (`i & (2^k - 1) < 2^k`,
/// and the array's length IS `2^k`), which folds the slice bounds check
/// away entirely. The padding is allocated, zeroed once, and never
/// reachable: `init_magics` step 6 proves every producible index is
/// `< TABLE_SIZE`, exhaustively, at startup.
const TABLE_ALLOC: usize = TABLE_SIZE.next_power_of_two();
/// Index mask for [`TABLE_ALLOC`]: `idx & TABLE_MASK` is always in bounds.
const TABLE_MASK: usize = TABLE_ALLOC - 1;

// The mask in `AttackTables::slot` is sound only while these three hold, so
// they fail the build rather than a test.
const _: () = assert!(TABLE_ALLOC.is_power_of_two());
const _: () = assert!(TABLE_ALLOC >= TABLE_SIZE);
const _: () = assert!(TABLE_MASK == TABLE_ALLOC - 1);

/// The runtime-initialized slider attack tables: per-square magic metadata
/// plus one densely packed shared attack table (rook block then bishop
/// block). Built once behind a `LazyLock` (see [`tables`]); every launch
/// re-derives and thereby re-verifies the magics against the naive oracle.
pub struct AttackTables {
    magics: [MagicPair; 64],
    /// Fixed-size so the length is a compile-time constant; only
    /// `table[..TABLE_SIZE]` is ever reachable.
    table: Box<[Bitboard; TABLE_ALLOC]>,
}

impl AttackTables {
    /// One packed-table slot: the single place the slice is indexed, so the
    /// bounds-folding mask has exactly one home.
    #[inline]
    fn slot(&self, i: usize) -> Bitboard {
        // The mask is provably a no-op (init step 6); what it buys is the
        // bounds check, discharged because the length IS `TABLE_MASK + 1`.
        self.table[i & TABLE_MASK]
    }

    /// Squares a rook on `sq` attacks given `occupied` (blockers included —
    /// capture semantics; off-mask occupancy bits are ignored by design).
    #[inline]
    pub fn rook_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.slot(self.magics[sq.idx()].rook.index(occupied))
    }

    /// Squares a bishop on `sq` attacks given `occupied`.
    #[inline]
    pub fn bishop_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.slot(self.magics[sq.idx()].bishop.index(occupied))
    }

    /// Squares a queen on `sq` attacks given `occupied`: two lookups OR'd —
    /// there is no queen magic table.
    #[inline]
    pub fn queen_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occupied) | self.bishop_attacks(sq, occupied)
    }

    /// General dispatch over every non-pawn piece type; debug-asserts
    /// `pt != Pawn` (pawns need a color, use [`pawn_attacks`]).
    #[inline]
    pub fn attacks(&self, pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
        debug_assert!(
            !matches!(pt, PieceType::Pawn),
            "use pawn_attacks(color, sq)"
        );
        match pt {
            PieceType::Knight => knight_attacks(sq),
            PieceType::King => king_attacks(sq),
            PieceType::Bishop => self.bishop_attacks(sq, occupied),
            PieceType::Rook => self.rook_attacks(sq, occupied),
            PieceType::Queen => self.queen_attacks(sq, occupied),
            PieceType::Pawn => Bitboard::EMPTY, // unreachable in debug
        }
    }

    fn new() -> AttackTables {
        // `vec![..].into_boxed_slice().try_into()` builds the array straight
        // on the heap; `Box::new([_; TABLE_ALLOC])` would materialise 1 MiB
        // on the stack first.
        let mut table: Box<[Bitboard; TABLE_ALLOC]> = vec![Bitboard::EMPTY; TABLE_ALLOC]
            .into_boxed_slice()
            .try_into()
            .expect("the vec is TABLE_ALLOC long by construction");
        let mut magics = [MagicPair::ZEROED; 64];
        init_magics(Slider::Rook, 0, &mut table[..], &mut magics);
        init_magics(Slider::Bishop, ROOK_TABLE_SIZE, &mut table[..], &mut magics);
        AttackTables { magics, table }
    }
}

// --- Magic initialization ---------------------------------------------------

/// Per-rank PRNG seeds, re-seeding per square. Tuned against an exact PRNG
/// consumption order, so the candidate loop below must consume the PRNG
/// identically: prefilter rejections also advance it.
///
/// ARCHON's own row, derived by the `find_own_magic_seeds` harness below
/// (deterministic 4096-candidate sweep, per-rank argmin on attempt count,
/// every winner re-validated against the attack oracle). Seeds only affect
/// init time — the magics they find are exact, so lookups, node counts and
/// the bench fingerprint are unchanged. Cost vs the previous row: 234,888
/// candidate attempts against 204,846, i.e. a few ms once at startup.
const MAGIC_SEEDS: [u64; 8] = [6723, 4627, 5863, 2501, 3959, 2821, 6327, 255];

/// Which of the two magic-backed sliders `init_magics` is building.
#[derive(Clone, Copy)]
enum Slider {
    Bishop,
    Rook,
}

impl Slider {
    /// The naive oracle for this slider type.
    fn attacks_naive(self, sq: u8, occ: u64) -> u64 {
        match self {
            Slider::Bishop => bishop_attacks_naive(sq, occ),
            Slider::Rook => rook_attacks_naive(sq, occ),
        }
    }

    /// Exact number of packed-table entries this slider's subtables consume.
    fn table_entries(self) -> usize {
        match self {
            Slider::Bishop => BISHOP_TABLE_SIZE,
            Slider::Rook => ROOK_TABLE_SIZE,
        }
    }
}

/// Find a magic for every square of `slider` and fill its block of the
/// packed shared `table`, starting at `base_offset`. Runs once per slider
/// type inside the `LazyLock` init; deterministic (fixed seeds).
fn init_magics(
    slider: Slider,
    base_offset: usize,
    table: &mut [Bitboard],
    magics: &mut [MagicPair; 64],
) {
    init_magics_seeded(slider, base_offset, table, magics, &MAGIC_SEEDS);
}

/// [`init_magics`] parameterized over the per-rank seed row, returning the
/// candidate ATTEMPTS spent per rank — the seed-quality metric. Any seed
/// row terminates with oracle-verified magics, so attempts-per-rank is the
/// whole comparison surface. Exists for the `find_own_magic_seeds` harness;
/// the production path passes [`MAGIC_SEEDS`] and ignores the counts.
fn init_magics_seeded(
    slider: Slider,
    base_offset: usize,
    table: &mut [Bitboard],
    magics: &mut [MagicPair; 64],
    seeds: &[u64; 8],
) -> [u32; 8] {
    // Scratch space shared across squares: `occupancy`/`reference` hold the
    // current square's Carry-Rippler enumeration; `epoch` + `attempt` make a
    // slot whose epoch predates the current attempt "empty", so a failed
    // candidate costs no memset.
    let mut occupancy = [0u64; 4096]; // 2^12 = max rook subtable
    let mut reference = [0u64; 4096];
    let mut epoch = [0u32; 4096];
    let mut attempt = 0u32;
    let mut attempts_by_rank = [0u32; 8];

    let mut offset = base_offset;
    for s in Square::all() {
        // Step 1 — relevant-blocker mask: the empty-board ray minus edge
        // squares NOT on s's own rank/file (a far-edge blocker never changes
        // the attack set; an edge square on s's own line IS relevant when s
        // sits on that edge).
        let edges = ((Bitboard::RANK_1 | Bitboard::RANK_8) & !Bitboard::RANKS[s.rank() as usize])
            | ((Bitboard::FILE_A | Bitboard::FILE_H) & !Bitboard::FILES[s.file() as usize]);
        let mask = slider.attacks_naive(s.index(), 0) & !edges.to_u64();
        let shift = 64 - mask.count_ones();

        // Step 3 — Carry-Rippler enumeration: `(b - mask) & mask` steps
        // through every subset of `mask` in a cycle starting and ending at
        // 0; the subtraction intentionally wraps.
        let mut size = 0;
        let mut b = 0u64;
        loop {
            occupancy[size] = b;
            reference[size] = slider.attacks_naive(s.index(), b);
            size += 1;
            b = b.wrapping_sub(mask) & mask;
            if b == 0 {
                break;
            }
        }
        debug_assert_eq!(size, 1usize << mask.count_ones());

        // Step 4 — candidate search: sparse random candidates through a
        // cheap prefilter (a good magic pushes many mask bits into the top
        // byte), then an O(size) collision check. Benign collisions — same
        // attack set at the same index — are allowed; they are why the
        // packed table can be this small.
        let mut rng = Prng::new(seeds[s.rank() as usize]);
        let square_attempts_base = attempt;
        let magic = loop {
            let mut candidate = 0u64;
            while (candidate.wrapping_mul(mask) >> 56).count_ones() < 6 {
                candidate = rng.sparse_rand();
            }
            attempt += 1;
            let collision_free = (0..size).all(|i| {
                let idx = ((occupancy[i] & mask).wrapping_mul(candidate) >> shift) as usize;
                if epoch[idx] < attempt {
                    epoch[idx] = attempt; // first claim this attempt
                    table[offset + idx] = Bitboard::from_u64(reference[i]);
                    true
                } else {
                    // Occupied this attempt: benign iff the attack sets match.
                    table[offset + idx].to_u64() == reference[i]
                }
            });
            if collision_free {
                // The subtable is now fully filled — verified against the
                // naive oracle by construction.
                break candidate;
            }
        };

        let found = Magic {
            mask,
            magic,
            offset: offset as u32,
            shift,
        };
        // Step 6 — REACHABILITY: the safety proof for `slot`'s index mask.
        // Every index this magic can ever produce must land inside its own
        // block, hence below `TABLE_SIZE`, hence unchanged by `& TABLE_MASK`
        // — checked through `Magic::index` itself over the exhaustive
        // enumeration. A real `assert!`, not a `debug_assert!`: without it
        // the mask would turn a would-be panic into a silent padding read.
        for &occ in &occupancy[..size] {
            let idx = found.index(Bitboard::from_u64(occ));
            assert!(
                idx >= offset && idx < offset + size && idx < TABLE_SIZE,
                "magic index {idx} escapes block [{offset}, {}) of {TABLE_SIZE}",
                offset + size
            );
        }
        match slider {
            Slider::Bishop => magics[s.idx()].bishop = found,
            Slider::Rook => magics[s.idx()].rook = found,
        }
        attempts_by_rank[s.rank() as usize] += attempt - square_attempts_base;
        offset += size;
    }

    // Step 5 — packing invariant: catches any mask/enumeration bug at
    // startup rather than as silent table overlap.
    assert_eq!(
        offset - base_offset,
        slider.table_entries(),
        "magic subtables must consume exactly the block reserved for them"
    );
    attempts_by_rank
}

// --- Storage and slider accessors -------------------------------------------

static TABLES: LazyLock<AttackTables> = LazyLock::new(AttackTables::new);

/// Force/fetch the magic tables. Every deref of the `LazyLock` costs an
/// atomic acquire load plus a predicted branch — cheap, but hot loops
/// should call `tables()` once at function entry and use the
/// `&'static AttackTables` methods thereafter. `main` forces it before the
/// UCI loop so init never happens mid-search.
///
/// NOTE: replacing this guard with an `&'static` field on `Board` has been
/// measured and REJECTED — the direct saving is tiny and the shrunken
/// callees trigger PGO inline re-decisions that cost more than the guard.
/// Do not re-propose without new evidence.
#[inline]
pub fn tables() -> &'static AttackTables {
    &TABLES
}

/// Squares a rook on `sq` attacks given `occupied`. Convenience wrapper for
/// cold paths; see [`tables`] for the hot-loop pattern.
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    tables().rook_attacks(sq, occupied)
}

/// Squares a bishop on `sq` attacks given `occupied`.
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    tables().bishop_attacks(sq, occupied)
}

/// Squares a queen on `sq` attacks given `occupied`.
#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    tables().queen_attacks(sq, occupied)
}

/// General dispatch over every non-pawn piece type; debug-asserts
/// `pt != Pawn` (pawns need a color, use [`pawn_attacks`]).
#[inline]
pub fn attacks(pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
    tables().attacks(pt, sq, occupied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Direction;

    /// Readable expected sets: `bb(&[Square::B3, Square::C2])`.
    fn bb(squares: &[Square]) -> Bitboard {
        squares.iter().copied().collect()
    }

    /// Tiny SplitMix64 PRNG: deterministic random occupancies, no dev-dependency.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    // -- Independent shift-based models (never touch safe_destination) ------

    /// Knight attacks as two-step `Bitboard::shift` compositions —
    /// independent of the table builders' stepping primitive.
    fn knight_shift_model(sq: Square) -> Bitboard {
        use Direction::*;
        let b = Bitboard::from_square(sq);
        b.shift(North).shift(NorthEast)         // +17
            | b.shift(North).shift(NorthWest)   // +15
            | b.shift(East).shift(NorthEast)    // +10
            | b.shift(West).shift(NorthWest)    // +6
            | b.shift(East).shift(SouthEast)    // -6
            | b.shift(West).shift(SouthWest)    // -10
            | b.shift(South).shift(SouthEast)   // -15
            | b.shift(South).shift(SouthWest) // -17
    }

    /// King attacks as the 3×3 neighborhood minus the square itself.
    fn king_shift_model(sq: Square) -> Bitboard {
        use Direction::*;
        let b = Bitboard::from_square(sq);
        let horizontal = b | b.shift(East) | b.shift(West);
        (horizontal | horizontal.shift(North) | horizontal.shift(South)) & !b
    }

    /// Pawn attacks from per-square checked stepping (`Square::shifted`).
    fn pawn_step_model(color: Color, sq: Square) -> Bitboard {
        let (east, west) = match color {
            Color::White => (Direction::NorthEast, Direction::NorthWest),
            Color::Black => (Direction::SouthEast, Direction::SouthWest),
        };
        [sq.shifted(east), sq.shifted(west)]
            .into_iter()
            .flatten()
            .collect()
    }

    const ROOK_DIRECTIONS: [Direction; 4] = [
        Direction::North,
        Direction::South,
        Direction::East,
        Direction::West,
    ];
    const BISHOP_DIRECTIONS: [Direction; 4] = [
        Direction::NorthEast,
        Direction::NorthWest,
        Direction::SouthEast,
        Direction::SouthWest,
    ];

    /// Slider attacks by walking `Square::shifted` rays one step at a time —
    /// the independent reference model for the naive generator.
    fn slider_step_model(sq: Square, dirs: [Direction; 4], occupied: Bitboard) -> Bitboard {
        let mut attacks = Bitboard::EMPTY;
        for d in dirs {
            let mut s = sq;
            while let Some(next) = s.shifted(d) {
                attacks |= Bitboard::from_square(next);
                if occupied.contains(next) {
                    break; // blockers are included (capture semantics)
                }
                s = next;
            }
        }
        attacks
    }

    // -- Hand-verified sets --------------------------------------------------

    #[test]
    fn leaper_hand_cases() {
        assert_eq!(knight_attacks(Square::A1), bb(&[Square::B3, Square::C2]));
        assert_eq!(
            knight_attacks(Square::E4),
            bb(&[
                Square::D2,
                Square::F2,
                Square::C3,
                Square::G3,
                Square::C5,
                Square::G5,
                Square::D6,
                Square::F6,
            ])
        );
        assert_eq!(
            king_attacks(Square::H8),
            bb(&[Square::G7, Square::G8, Square::H7])
        );
        assert_eq!(
            pawn_attacks(Color::White, Square::E4),
            bb(&[Square::D5, Square::F5])
        );
        assert_eq!(pawn_attacks(Color::Black, Square::A5), bb(&[Square::B4]));
    }

    // -- Count properties -----------------------------------------------------

    #[test]
    fn king_attack_count_distribution() {
        let mut corners = 0;
        let mut edges = 0;
        let mut interior = 0;
        for sq in Square::all() {
            match king_attacks(sq).count() {
                3 => corners += 1,
                5 => edges += 1,
                8 => interior += 1,
                n => panic!("king on {sq} attacks {n} squares"),
            }
        }
        assert_eq!((corners, edges, interior), (4, 24, 36));
    }

    #[test]
    fn knight_attack_counts() {
        for sq in Square::all() {
            let n = knight_attacks(sq).count();
            assert!(
                matches!(n, 2 | 3 | 4 | 6 | 8),
                "knight on {sq} attacks {n} squares"
            );
        }
        assert_eq!(knight_attacks(Square::A1).count(), 2);
        assert_eq!(knight_attacks(Square::E4).count(), 8);
    }

    #[test]
    fn pawn_attack_counts() {
        for color in Color::ALL {
            let last_rank = match color {
                Color::White => 7,
                Color::Black => 0,
            };
            for sq in Square::all() {
                let n = pawn_attacks(color, sq).count();
                if sq.rank() == last_rank {
                    // No pawn can stand here; the shift leaves the board.
                    assert_eq!(n, 0, "{color:?} pawn on {sq}");
                } else {
                    let expected = if sq.file() == 0 || sq.file() == 7 {
                        1
                    } else {
                        2
                    };
                    assert_eq!(n, expected, "{color:?} pawn on {sq}");
                }
            }
        }
    }

    // -- Edge wrapping ----------------------------------------------------------

    #[test]
    fn no_edge_wrapping() {
        for sq in Square::all() {
            let sets = [
                ("knight", knight_attacks(sq)),
                ("king", king_attacks(sq)),
                ("white pawn", pawn_attacks(Color::White, sq)),
                ("black pawn", pawn_attacks(Color::Black, sq)),
            ];
            for (name, set) in sets {
                if sq.file() == 0 {
                    assert!(
                        (set & Bitboard::FILE_H).is_empty(),
                        "{name} on {sq} wraps onto file H"
                    );
                }
                if sq.file() == 7 {
                    assert!(
                        (set & Bitboard::FILE_A).is_empty(),
                        "{name} on {sq} wraps onto file A"
                    );
                }
            }
        }
    }

    // -- Set-wise pawn attacks -------------------------------------------------

    #[test]
    fn pawn_attacks_bb_matches_per_square_union() {
        for color in Color::ALL {
            // Full-rank sets ...
            for rank in 0..8 {
                let pawns = Bitboard::RANKS[rank];
                let expected = pawns
                    .squares()
                    .map(|sq| pawn_attacks(color, sq))
                    .fold(Bitboard::EMPTY, |acc, a| acc | a);
                assert_eq!(
                    pawn_attacks_bb(color, pawns),
                    expected,
                    "{color:?} pawns on rank {rank}"
                );
            }
            // ... and random sets for good measure.
            let mut rng = SplitMix64(0xA77A_C55E_75A7_7ACC);
            for _ in 0..1000 {
                let pawns = Bitboard::from_u64(rng.next_u64());
                let expected = pawns
                    .squares()
                    .map(|sq| pawn_attacks(color, sq))
                    .fold(Bitboard::EMPTY, |acc, a| acc | a);
                assert_eq!(
                    pawn_attacks_bb(color, pawns),
                    expected,
                    "{color:?} {pawns:?}"
                );
            }
        }
    }

    // -- Symmetry properties -----------------------------------------------------

    #[test]
    fn leaper_attack_symmetry() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                assert_eq!(
                    knight_attacks(s1).contains(s2),
                    knight_attacks(s2).contains(s1),
                    "knight symmetry {s1}/{s2}"
                );
                assert_eq!(
                    king_attacks(s1).contains(s2),
                    king_attacks(s2).contains(s1),
                    "king symmetry {s1}/{s2}"
                );
                // Pawn reciprocity: a white pawn on s1 attacks s2 iff a
                // black pawn on s2 attacks s1 (the attackers-of-a-square
                // trick movegen relies on).
                assert_eq!(
                    pawn_attacks(Color::White, s1).contains(s2),
                    pawn_attacks(Color::Black, s2).contains(s1),
                    "pawn reciprocity {s1}/{s2}"
                );
            }
        }
    }

    #[test]
    fn pawn_attacks_mirror_between_colors() {
        for sq in Square::all() {
            let white = pawn_attacks(Color::White, sq);
            let black_mirrored: Bitboard = pawn_attacks(Color::Black, sq.flip_rank())
                .squares()
                .map(Square::flip_rank)
                .collect();
            assert_eq!(white, black_mirrored, "color mirror on {sq}");
        }
    }

    // -- Exhaustive cross-checks vs shift-based construction ---------------------

    #[test]
    fn leaper_tables_match_shift_models_exhaustively() {
        for sq in Square::all() {
            assert_eq!(knight_attacks(sq), knight_shift_model(sq), "knight on {sq}");
            assert_eq!(king_attacks(sq), king_shift_model(sq), "king on {sq}");
            for color in Color::ALL {
                assert_eq!(
                    pawn_attacks(color, sq),
                    pawn_step_model(color, sq),
                    "{color:?} pawn on {sq}"
                );
            }
        }
    }

    #[test]
    fn pseudo_tables_match_ray_models_exhaustively() {
        for sq in Square::all() {
            let rook = slider_step_model(sq, ROOK_DIRECTIONS, Bitboard::EMPTY);
            let bishop = slider_step_model(sq, BISHOP_DIRECTIONS, Bitboard::EMPTY);
            assert_eq!(pseudo_attacks(PieceType::Rook, sq), rook, "rook on {sq}");
            assert_eq!(
                pseudo_attacks(PieceType::Bishop, sq),
                bishop,
                "bishop on {sq}"
            );
            assert_eq!(
                pseudo_attacks(PieceType::Queen, sq),
                rook | bishop,
                "queen on {sq}"
            );
            // Empty-board rook rays always cover its full rank + file.
            assert_eq!(rook.count(), 14, "rook ray count on {sq}");
            // Non-slider dispatch hits the leaper tables.
            assert_eq!(pseudo_attacks(PieceType::Knight, sq), knight_attacks(sq));
            assert_eq!(pseudo_attacks(PieceType::King, sq), king_attacks(sq));
        }
    }

    // -- The naive oracle itself -------------------------------------------------

    #[test]
    fn naive_slider_blocker_semantics() {
        // Rook on a1, blocker on a4: the ray stops at (and includes) a4;
        // rank 1 is open to h1.
        let occ = Bitboard::from_square(Square::A4);
        let attacks = Bitboard::from_u64(rook_attacks_naive(Square::A1.index(), occ.to_u64()));
        assert_eq!(
            attacks,
            bb(&[
                Square::A2,
                Square::A3,
                Square::A4, // blocker included: capture semantics
                Square::B1,
                Square::C1,
                Square::D1,
                Square::E1,
                Square::F1,
                Square::G1,
                Square::H1,
            ])
        );
        // Bishop on c1, blocker on e3: stops there; the other diagonal runs
        // to a3.
        let occ = Bitboard::from_square(Square::E3);
        let attacks = Bitboard::from_u64(bishop_attacks_naive(Square::C1.index(), occ.to_u64()));
        assert_eq!(
            attacks,
            bb(&[Square::B2, Square::A3, Square::D2, Square::E3])
        );
    }

    #[test]
    fn naive_sliders_match_ray_models_on_random_occupancies() {
        let mut rng = SplitMix64(0x0BAD_5EED_0BAD_5EED);
        for _ in 0..200 {
            let occ = Bitboard::from_u64(rng.next_u64());
            for sq in Square::all() {
                let rook = Bitboard::from_u64(rook_attacks_naive(sq.index(), occ.to_u64()));
                assert_eq!(
                    rook,
                    slider_step_model(sq, ROOK_DIRECTIONS, occ),
                    "rook on {sq}, occ {occ:?}"
                );
                let bishop = Bitboard::from_u64(bishop_attacks_naive(sq.index(), occ.to_u64()));
                assert_eq!(
                    bishop,
                    slider_step_model(sq, BISHOP_DIRECTIONS, occ),
                    "bishop on {sq}, occ {occ:?}"
                );
            }
        }
    }

    // -- Const-evaluability (step 2's geometry builders depend on it) -------------

    #[test]
    fn foundations_are_const_evaluable() {
        // Rook on a1, empty board: file A + rank 1 minus a1 itself.
        const A1_ROOK: u64 = rook_attacks_naive(0, 0);
        const { assert!(A1_ROOK == 0x0101_0101_0101_01FE) }
        // pawn_attacks_bb is const, per the doc 03 §3.1 API.
        const E4_WHITE: Bitboard = pawn_attacks_bb(Color::White, Bitboard::from_square(Square::E4));
        assert_eq!(E4_WHITE, bb(&[Square::D5, Square::F5]));
    }

    // -- Geometry: independent coordinate-walk models -----------------------------
    //
    // Recomputed in (file, rank) coordinates, never touching the stepping
    // primitive, the naive generator, or any table.

    /// The unit (file, rank) step from `s1` toward `s2` if the squares are
    /// aligned on a rank, file, or diagonal; `None` otherwise (incl. equal).
    fn alignment_dir(s1: Square, s2: Square) -> Option<(i8, i8)> {
        let df = s2.file() as i8 - s1.file() as i8;
        let dr = s2.rank() as i8 - s1.rank() as i8;
        if (df, dr) == (0, 0) || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
            None
        } else {
            Some((df.signum(), dr.signum()))
        }
    }

    /// One coordinate step; `None` off the board. Wrap-free by construction
    /// (file/rank arithmetic, not index arithmetic).
    fn coord_step(s: Square, (df, dr): (i8, i8)) -> Option<Square> {
        let f = s.file() as i8 + df;
        let r = s.rank() as i8 + dr;
        ((0..8).contains(&f) && (0..8).contains(&r)).then(|| Square::from_coords(f as u8, r as u8))
    }

    /// `between` model: walk strictly from `s1` to `s2`, then OR in `s2`
    /// (the includes-target semantics under test).
    fn between_model(s1: Square, s2: Square) -> Bitboard {
        let target = Bitboard::from_square(s2);
        let Some(dir) = alignment_dir(s1, s2) else {
            return target;
        };
        let mut b = target;
        let mut s = coord_step(s1, dir).unwrap();
        while s != s2 {
            b |= Bitboard::from_square(s);
            s = coord_step(s, dir).unwrap();
        }
        b
    }

    /// `line` model: from `s1`, walk to the board edge both ways.
    fn line_model(s1: Square, s2: Square) -> Bitboard {
        let Some((df, dr)) = alignment_dir(s1, s2) else {
            return Bitboard::EMPTY;
        };
        let mut b = Bitboard::from_square(s1);
        for dir in [(df, dr), (-df, -dr)] {
            let mut s = s1;
            while let Some(next) = coord_step(s, dir) {
                b |= Bitboard::from_square(next);
                s = next;
            }
        }
        b
    }

    /// `ray_pass` model: from `s1` (exclusive) toward `s2`, to the edge.
    fn ray_pass_model(s1: Square, s2: Square) -> Bitboard {
        let Some(dir) = alignment_dir(s1, s2) else {
            return Bitboard::EMPTY;
        };
        let mut b = Bitboard::EMPTY;
        let mut s = s1;
        while let Some(next) = coord_step(s, dir) {
            b |= Bitboard::from_square(next);
            s = next;
        }
        b
    }

    // -- Hand-verified geometry sets -----------------------------------------------

    #[test]
    fn geometry_hand_cases() {
        // BETWEEN includes the target square (SF semantics) ...
        assert_eq!(
            between(Square::A1, Square::D4),
            bb(&[Square::B2, Square::C3, Square::D4])
        );
        // ... in BOTH directions — only the included endpoint flips.
        assert_eq!(
            between(Square::D4, Square::A1),
            bb(&[Square::C3, Square::B2, Square::A1])
        );
        assert_eq!(
            between(Square::E1, Square::E8),
            bb(&[
                Square::E2,
                Square::E3,
                Square::E4,
                Square::E5,
                Square::E6,
                Square::E7,
                Square::E8,
            ])
        );
        // Unaligned: exactly the target square.
        assert_eq!(between(Square::A1, Square::B3), bb(&[Square::B3]));
        // Adjacent aligned (evasion-mask edge case: checker next to the king
        // leaves only the capture, no blocking square).
        assert_eq!(between(Square::E1, Square::E2), bb(&[Square::E2]));

        // LINE runs edge to edge and includes both endpoints.
        assert_eq!(
            line(Square::B2, Square::C3),
            bb(&[
                Square::A1,
                Square::B2,
                Square::C3,
                Square::D4,
                Square::E5,
                Square::F6,
                Square::G7,
                Square::H8,
            ])
        );
        assert_eq!(line(Square::C4, Square::F4), Bitboard::RANKS[3]);
        assert_eq!(line(Square::A5, Square::A7), Bitboard::FILE_A);
        assert_eq!(line(Square::A1, Square::B3), Bitboard::EMPTY);

        // RAY_PASS: from s1 (exclusive) through s2 to the edge.
        assert_eq!(
            ray_pass(Square::D4, Square::F6),
            bb(&[Square::E5, Square::F6, Square::G7, Square::H8])
        );
        assert_eq!(
            ray_pass(Square::F6, Square::D4),
            bb(&[Square::E5, Square::D4, Square::C3, Square::B2, Square::A1])
        );
        assert_eq!(ray_pass(Square::A1, Square::B3), Bitboard::EMPTY);

        assert_eq!(distance(Square::A1, Square::H8), 7);
        assert_eq!(distance(Square::E4, Square::G5), 2);
    }

    // -- Diagonal edge cases --------------------------------------------------------

    #[test]
    fn geometry_diagonal_edge_cases() {
        for s in Square::all() {
            assert_eq!(
                between(s, s),
                Bitboard::from_square(s),
                "BETWEEN[{s}][{s}] must be {{{s}}}"
            );
            assert!(line(s, s).is_empty(), "LINE[{s}][{s}] must be empty");
            assert!(
                ray_pass(s, s).is_empty(),
                "RAY_PASS[{s}][{s}] must be empty"
            );
        }
    }

    // -- Exhaustive property tests over all 4096 ordered pairs -----------------------

    #[test]
    fn between_includes_target_always() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                let b = between(s1, s2);
                assert!(b.contains(s2), "between({s1}, {s2}) misses {s2}");
                // Exactly {s2} whenever there is nothing strictly between.
                let expect_only_target = match alignment_dir(s1, s2) {
                    None => true,
                    Some(_) => distance(s1, s2) == 1,
                };
                assert_eq!(
                    b == Bitboard::from_square(s2),
                    expect_only_target,
                    "between({s1}, {s2}) = {b:?}"
                );
            }
        }
    }

    #[test]
    fn geometry_tables_match_coordinate_models() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                assert_eq!(
                    between(s1, s2),
                    between_model(s1, s2),
                    "between({s1}, {s2})"
                );
                assert_eq!(line(s1, s2), line_model(s1, s2), "line({s1}, {s2})");
                assert_eq!(
                    ray_pass(s1, s2),
                    ray_pass_model(s1, s2),
                    "ray_pass({s1}, {s2})"
                );
            }
        }
    }

    #[test]
    fn geometry_tables_agree_with_naive_rays() {
        // The defining formulas, recomputed test-side through the
        // independent step-walker.
        for s1 in Square::all() {
            for s2 in Square::all() {
                let dirs = if pseudo_attacks(PieceType::Bishop, s1).contains(s2) {
                    BISHOP_DIRECTIONS
                } else if pseudo_attacks(PieceType::Rook, s1).contains(s2) {
                    ROOK_DIRECTIONS
                } else {
                    // Unaligned pair: emptiness is covered by the model test.
                    continue;
                };
                let (b1, b2) = (Bitboard::from_square(s1), Bitboard::from_square(s2));
                assert_eq!(
                    between(s1, s2),
                    (slider_step_model(s1, dirs, b2) & slider_step_model(s2, dirs, b1)) | b2,
                    "between({s1}, {s2}) vs ray formula"
                );
                assert_eq!(
                    line(s1, s2),
                    (slider_step_model(s1, dirs, Bitboard::EMPTY)
                        & slider_step_model(s2, dirs, Bitboard::EMPTY))
                        | b1
                        | b2,
                    "line({s1}, {s2}) vs ray formula"
                );
                assert_eq!(
                    ray_pass(s1, s2),
                    slider_step_model(s1, dirs, Bitboard::EMPTY)
                        & (slider_step_model(s2, dirs, b1) | b2),
                    "ray_pass({s1}, {s2}) vs ray formula"
                );
            }
        }
    }

    #[test]
    fn line_symmetry_endpoints_and_extension() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                let l = line(s1, s2);
                assert_eq!(l, line(s2, s1), "line symmetry {s1}/{s2}");
                if s1 != s2 && alignment_dir(s1, s2).is_some() {
                    assert!(l.contains(s1), "line({s1}, {s2}) misses {s1}");
                    assert!(l.contains(s2), "line({s1}, {s2}) misses {s2}");
                } else {
                    assert!(l.is_empty(), "line({s1}, {s2}) must be empty");
                }
                // Any point of the line regenerates the whole line — the
                // table stores edge-to-edge lines, never segments.
                for s3 in l.squares() {
                    if s3 != s1 {
                        assert_eq!(line(s1, s3), l, "line({s1}, {s3}) vs line({s1}, {s2})");
                    }
                }
            }
        }
    }

    #[test]
    fn ray_pass_properties() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                let r = ray_pass(s1, s2);
                if s1 != s2 && alignment_dir(s1, s2).is_some() {
                    assert!(!r.contains(s1), "ray_pass({s1}, {s2}) contains {s1}");
                    assert!(r.contains(s2), "ray_pass({s1}, {s2}) misses {s2}");
                    assert_eq!(r & line(s1, s2), r, "ray_pass({s1}, {s2}) ⊄ line");
                    let btw = between(s1, s2);
                    assert_eq!(btw & r, btw, "between({s1}, {s2}) ⊄ ray_pass");
                } else {
                    assert!(r.is_empty(), "ray_pass({s1}, {s2}) must be empty");
                }
            }
        }
    }

    #[test]
    fn distance_properties() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                let d = distance(s1, s2);
                assert_eq!(d, distance(s2, s1), "distance symmetry {s1}/{s2}");
                assert_eq!(d == 0, s1 == s2, "distance({s1}, {s2}) zero-iff-equal");
                // Recomputed inline ...
                let expected = s1
                    .file()
                    .abs_diff(s2.file())
                    .max(s1.rank().abs_diff(s2.rank()));
                assert_eq!(d, expected, "distance({s1}, {s2})");
                // ... and against the reference implementation.
                assert_eq!(d, s1.distance(s2), "table vs Square::distance");
            }
        }
    }

    // -- Collinearity -----------------------------------------------------------------

    #[test]
    fn aligned_collinearity_cases() {
        assert!(aligned(Square::A1, Square::C3, Square::E5));
        assert!(aligned(Square::A1, Square::C3, Square::B2));
        assert!(aligned(Square::E4, Square::E8, Square::E1));
        assert!(aligned(Square::A4, Square::H4, Square::C4));
        assert!(!aligned(Square::A1, Square::C3, Square::D3));
        // Unaligned generators span no line — nothing is collinear with
        // them, not even the endpoints themselves.
        assert!(!aligned(Square::A1, Square::B3, Square::A1));
        assert!(!aligned(Square::A1, Square::B3, Square::C5));

        for s1 in Square::all() {
            for s2 in Square::all() {
                let l = line_model(s1, s2);
                for s3 in Square::all() {
                    assert_eq!(
                        aligned(s1, s2, s3),
                        l.contains(s3),
                        "aligned({s1}, {s2}, {s3})"
                    );
                }
            }
        }
    }

    // -- Slider magic validation (the core gate) -----------------------------------

    fn naive_rook(sq: Square, occ: Bitboard) -> Bitboard {
        Bitboard::from_u64(rook_attacks_naive(sq.index(), occ.to_u64()))
    }

    fn naive_bishop(sq: Square, occ: Bitboard) -> Bitboard {
        Bitboard::from_u64(bishop_attacks_naive(sq.index(), occ.to_u64()))
    }

    #[test]
    fn magic_hand_cases() {
        // The naive-oracle hand cases, now through the magic path.
        let occ = Bitboard::from_square(Square::A4);
        assert_eq!(
            rook_attacks(Square::A1, occ),
            bb(&[
                Square::A2,
                Square::A3,
                Square::A4, // blocker included: capture semantics
                Square::B1,
                Square::C1,
                Square::D1,
                Square::E1,
                Square::F1,
                Square::G1,
                Square::H1,
            ])
        );
        let occ = Bitboard::from_square(Square::E3);
        assert_eq!(
            bishop_attacks(Square::C1, occ),
            bb(&[Square::B2, Square::A3, Square::D2, Square::E3])
        );
    }

    #[test]
    fn magic_lookups_match_naive_on_random_occupancies() {
        // 10,000 seeded occupancies × 64 squares × {rook, bishop, queen} vs
        // the naive oracle. Mixed density: off-mask bits exercise the
        // `& mask` step. Seeded ⇒ reproducible failures.
        let mut rng = SplitMix64(0x3A61_C0DE_5EED_ED01);
        for trial in 0..10_000u32 {
            let occ = Bitboard::from_u64(match trial % 3 {
                0 => rng.next_u64(),                                   // uniform
                1 => rng.next_u64() & rng.next_u64() & rng.next_u64(), // sparse
                _ => rng.next_u64() | rng.next_u64(),                  // dense
            });
            for sq in Square::all() {
                let rook = naive_rook(sq, occ);
                let bishop = naive_bishop(sq, occ);
                assert_eq!(rook_attacks(sq, occ), rook, "rook on {sq}, occ {occ:?}");
                assert_eq!(
                    bishop_attacks(sq, occ),
                    bishop,
                    "bishop on {sq}, occ {occ:?}"
                );
                assert_eq!(
                    queen_attacks(sq, occ),
                    rook | bishop,
                    "queen on {sq}, occ {occ:?}"
                );
            }
        }
    }

    #[test]
    fn magic_edge_occupancies() {
        // Empty board must reproduce the PSEUDO_* rays; full board, the
        // mask itself, and the square's own bit stress the index path.
        let t = tables();
        for sq in Square::all() {
            let s = sq.index() as usize;
            assert_eq!(
                t.rook_attacks(sq, Bitboard::EMPTY),
                PSEUDO_ROOK[s],
                "rook on empty board, {sq}"
            );
            assert_eq!(
                t.bishop_attacks(sq, Bitboard::EMPTY),
                PSEUDO_BISHOP[s],
                "bishop on empty board, {sq}"
            );
            assert_eq!(
                t.queen_attacks(sq, Bitboard::EMPTY),
                PSEUDO_QUEEN[s],
                "queen on empty board, {sq}"
            );
            let pair = &t.magics[s];
            let edge_occs = |mask: u64| {
                [
                    Bitboard::FULL,
                    Bitboard::from_u64(mask),
                    Bitboard::from_square(sq),
                ]
            };
            for occ in edge_occs(pair.rook.mask) {
                assert_eq!(
                    t.rook_attacks(sq, occ),
                    naive_rook(sq, occ),
                    "rook on {sq}, occ {occ:?}"
                );
            }
            for occ in edge_occs(pair.bishop.mask) {
                assert_eq!(
                    t.bishop_attacks(sq, occ),
                    naive_bishop(sq, occ),
                    "bishop on {sq}, occ {occ:?}"
                );
            }
        }
    }

    /// Independently written Carry-Rippler (never touches the init code):
    /// calls `f` on every subset of `mask`, the empty set included.
    fn for_each_subset(mask: u64, mut f: impl FnMut(Bitboard)) {
        let mut b = 0u64;
        loop {
            f(Bitboard::from_u64(b));
            b = b.wrapping_sub(mask) & mask;
            if b == 0 {
                break;
            }
        }
    }

    /// Edge squares irrelevant to a blocker mask on `sq` — the step-1
    /// formula, recomputed test-side (never read from the init code).
    fn irrelevant_edges(sq: Square) -> Bitboard {
        ((Bitboard::RANK_1 | Bitboard::RANK_8) & !Bitboard::RANKS[sq.rank() as usize])
            | ((Bitboard::FILE_A | Bitboard::FILE_H) & !Bitboard::FILES[sq.file() as usize])
    }

    #[test]
    fn all_subsets_exact_on_every_square() {
        // The core gate: every subset of every relevant-blocker mask,
        // through the public API vs the naive oracle. Re-proves what init
        // verified, but catches offset/packing bugs too. Masks recomputed
        // test-side; `magic_masks_exclude_irrelevant_edges` proves they
        // equal the ones init used.
        let t = tables();
        for sq in Square::all() {
            let s = sq.index() as usize;
            let relevant = !irrelevant_edges(sq);
            for_each_subset((PSEUDO_ROOK[s] & relevant).to_u64(), |occ| {
                assert_eq!(
                    t.rook_attacks(sq, occ),
                    naive_rook(sq, occ),
                    "rook on {sq}, subset {occ:?}"
                );
            });
            for_each_subset((PSEUDO_BISHOP[s] & relevant).to_u64(), |occ| {
                assert_eq!(
                    t.bishop_attacks(sq, occ),
                    naive_bishop(sq, occ),
                    "bishop on {sq}, subset {occ:?}"
                );
            });
        }
    }

    #[test]
    fn attacks_dispatch_matches_specialized_fns() {
        // Free-fn and method dispatch agree with the specialized lookups.
        let t = tables();
        let mut rng = SplitMix64(0xD15A_7C4E_D15A_7C4E);
        for _ in 0..100 {
            let occ = Bitboard::from_u64(rng.next_u64());
            for sq in Square::all() {
                assert_eq!(attacks(PieceType::Knight, sq, occ), knight_attacks(sq));
                assert_eq!(attacks(PieceType::King, sq, occ), king_attacks(sq));
                assert_eq!(
                    attacks(PieceType::Bishop, sq, occ),
                    t.bishop_attacks(sq, occ)
                );
                assert_eq!(attacks(PieceType::Rook, sq, occ), t.rook_attacks(sq, occ));
                assert_eq!(attacks(PieceType::Queen, sq, occ), t.queen_attacks(sq, occ));
                assert_eq!(
                    t.attacks(PieceType::Queen, sq, occ),
                    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
                );
            }
        }
    }

    #[test]
    fn packing_structural_invariants() {
        // Walking squares in init order, each block's offsets must tile its
        // reserved range exactly — disjoint, contiguous, exhaustive — and
        // every shift must equal 64 - popcount(mask).
        let t = tables();
        let mut expected = 0usize; // rook block starts the shared table
        for sq in Square::all() {
            let m = &t.magics[sq.idx()].rook;
            assert_eq!(m.offset as usize, expected, "rook offset on {sq}");
            assert_eq!(m.shift, 64 - m.mask.count_ones(), "rook shift on {sq}");
            assert!(
                (10..=12).contains(&m.mask.count_ones()),
                "rook mask popcount on {sq}"
            );
            expected += 1usize << m.mask.count_ones();
        }
        assert_eq!(expected, ROOK_TABLE_SIZE, "rook block size");
        for sq in Square::all() {
            let m = &t.magics[sq.idx()].bishop;
            assert_eq!(m.offset as usize, expected, "bishop offset on {sq}");
            assert_eq!(m.shift, 64 - m.mask.count_ones(), "bishop shift on {sq}");
            assert!(
                (5..=9).contains(&m.mask.count_ones()),
                "bishop mask popcount on {sq}"
            );
            expected += 1usize << m.mask.count_ones();
        }
        assert_eq!(
            expected, TABLE_SIZE,
            "bishop block ends at the packed-table end"
        );

        // The ALLOCATION is the padded power-of-two length; the PACKED
        // CONTENT still ends at `TABLE_SIZE`. Both halves are pinned because
        // the mask is only sound while they differ in exactly this way.
        assert_eq!(
            t.table.len(),
            TABLE_ALLOC,
            "allocation is the padded length"
        );

        // A non-empty slot at or above `TABLE_SIZE` would mean some magic
        // wrote outside its reserved block — precisely the corruption the
        // index mask would otherwise hide.
        assert!(
            t.table[TABLE_SIZE..].iter().all(|b| b.is_empty()),
            "padding slots must stay empty"
        );
    }

    #[test]
    fn magic_masks_exclude_irrelevant_edges() {
        // The mask is the empty-board ray minus edge squares not on the
        // square's own rank/file, recomputed test-side.
        let t = tables();
        for sq in Square::all() {
            let s = sq.index() as usize;
            let edges = irrelevant_edges(sq);
            assert_eq!(
                Bitboard::from_u64(t.magics[s].rook.mask),
                PSEUDO_ROOK[s] & !edges,
                "rook mask on {sq}"
            );
            assert_eq!(
                Bitboard::from_u64(t.magics[s].bishop.mask),
                PSEUDO_BISHOP[s] & !edges,
                "bishop mask on {sq}"
            );
        }
    }

    #[test]
    fn init_is_deterministic() {
        // Two independent runs must find identical magics and tables
        // (guards PRNG-consumption-order changes that would silently
        // de-tune the hardcoded seeds), and match the LazyLock instance.
        let a = AttackTables::new();
        let b = AttackTables::new();
        let t = tables();
        for sq in Square::all() {
            let s = sq.index() as usize;
            assert_eq!(a.magics[s], b.magics[s], "magics differ on {sq}");
            assert_eq!(a.magics[s], t.magics[s], "LazyLock magics differ on {sq}");
        }
        assert_eq!(a.table, b.table, "packed tables differ between runs");
        assert_eq!(a.table, t.table, "LazyLock packed table differs");
    }

    #[test]
    #[should_panic(expected = "all-zero fixed point")]
    fn prng_rejects_zero_seed() {
        let _ = Prng::new(0);
    }

    #[test]
    fn prng_is_deterministic_and_sparse() {
        // Same seed ⇒ same stream; sparse_rand consumes exactly three
        // rand64 draws (the consumption-order contract the seeds are tuned
        // against).
        let mut a = Prng::new(MAGIC_SEEDS[0]);
        let mut b = Prng::new(MAGIC_SEEDS[0]);
        let (x, y, z) = (a.rand64(), a.rand64(), a.rand64());
        assert_eq!(b.sparse_rand(), x & y & z);
        assert_eq!(a.rand64(), b.rand64(), "streams must stay in lockstep");
    }

    /// REMEDIATION-PENDING harness (originality audit item 6a): find and
    /// validate ARCHON's OWN per-rank magic seeds, replacing the SF row.
    /// A rank's cost under a seed is independent of every other rank (the
    /// PRNG re-seeds per square), so the best row is the per-rank argmin
    /// over one shared candidate sweep. `#[ignore]`d (minutes); run with
    /// `cargo test --release -- --ignored find_own_magic_seeds --nocapture`,
    /// paste the printed row over `MAGIC_SEEDS`, and drop that constant's
    /// notice. A bad row cannot land silently: init re-verifies against the
    /// naive oracle at startup, and this test's final section re-checks it.
    #[test]
    #[ignore = "minutes-long seed sweep; audit item 6a — run at remediation time"]
    fn find_own_magic_seeds() {
        // 4096 odd candidates: candidate k is simply 2k+1, so the sweep is
        // reproducible by inspection.
        const CANDIDATES: u64 = 4096;

        let mut table: Box<[Bitboard]> = vec![Bitboard::EMPTY; TABLE_ALLOC].into_boxed_slice();
        let mut magics = [MagicPair::ZEROED; 64];

        let mut best_seed = [0u64; 8];
        let mut best_cost = [u32::MAX; 8];
        for k in 0..CANDIDATES {
            let candidate = 2 * k + 1;
            let row = [candidate; 8];
            let rook = init_magics_seeded(Slider::Rook, 0, &mut table, &mut magics, &row);
            let bishop = init_magics_seeded(
                Slider::Bishop,
                ROOK_TABLE_SIZE,
                &mut table,
                &mut magics,
                &row,
            );
            for rank in 0..8 {
                let cost = rook[rank] + bishop[rank];
                if cost < best_cost[rank] {
                    best_cost[rank] = cost;
                    best_seed[rank] = candidate;
                }
            }
        }

        // Validate the winner row and price it against the shipped row.
        let rook_new = init_magics_seeded(Slider::Rook, 0, &mut table, &mut magics, &best_seed);
        let bishop_new = init_magics_seeded(
            Slider::Bishop,
            ROOK_TABLE_SIZE,
            &mut table,
            &mut magics,
            &best_seed,
        );
        // Oracle re-check on the FINAL tables while they hold the new row's
        // magics, through the production index arithmetic.
        let mut rng = Prng::new(0x20260809);
        for s in Square::all() {
            for _ in 0..64 {
                let occ = rng.rand64();
                let rook_idx = magics[s.idx()].rook.index(Bitboard::from_u64(occ));
                assert_eq!(
                    table[rook_idx].to_u64(),
                    rook_attacks_naive(s.index(), occ),
                    "rook oracle mismatch on {s} under the found seeds"
                );
                let bishop_idx = magics[s.idx()].bishop.index(Bitboard::from_u64(occ));
                assert_eq!(
                    table[bishop_idx].to_u64(),
                    bishop_attacks_naive(s.index(), occ),
                    "bishop oracle mismatch on {s} under the found seeds"
                );
            }
        }
        let rook_old = init_magics_seeded(Slider::Rook, 0, &mut table, &mut magics, &MAGIC_SEEDS);
        let bishop_old = init_magics_seeded(
            Slider::Bishop,
            ROOK_TABLE_SIZE,
            &mut table,
            &mut magics,
            &MAGIC_SEEDS,
        );
        let total = |a: &[u32; 8], b: &[u32; 8]| -> u64 {
            a.iter().chain(b.iter()).map(|&x| u64::from(x)).sum()
        };
        let ours = total(&rook_new, &bishop_new);
        let shipped = total(&rook_old, &bishop_old);
        println!("ARCHON candidate row: const MAGIC_SEEDS: [u64; 8] = {best_seed:?};");
        println!("per-rank attempts (rook+bishop): {best_cost:?}");
        println!("total attempts: found {ours} vs shipped(SF) {shipped}");
        assert!(
            ours <= shipped.saturating_mul(2),
            "found row converges catastrophically slower than the shipped one \
             ({ours} vs {shipped} attempts) — widen CANDIDATES"
        );
    }
}
