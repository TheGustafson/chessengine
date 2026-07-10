//! Precomputed attack tables (design/03-attack-tables.md, D12/D13).
//!
//! This module answers exactly one question: *which squares does a piece on
//! `sq` attack given occupancy `occ`?* Legality (pins, checks, castling) is
//! movegen's job (design/05).
//!
//! Landed so far (doc 03 §4 steps 1–2): the `const fn` foundations
//! ([`safe_destination`] + the naive ray generator that serves as the
//! module's correctness oracle), the compile-time leaper and empty-board
//! slider tables (`KNIGHT_ATTACKS`, `KING_ATTACKS`, `PAWN_ATTACKS`,
//! `PSEUDO_*`) with their accessor functions, and the 64×64 ray-geometry
//! tables behind [`between`]/[`line`]/[`ray_pass`]/[`distance`]/[`aligned`].
//! Still to come, in doc 03's step order: the fancy-magic slider machinery
//! behind `LazyLock<AttackTables>` (steps 3–4).
//!
//! All tables here are `static`, not `const`: a `const` is a value inlined
//! (and potentially duplicated) at every use site, while a `static`
//! guarantees one copy in `.rodata` (doc 03 §3.8). Const builders operate on
//! raw `u64` internally — operator traits like `BitOr` are not `const` — and
//! wrap into [`Bitboard`] at the boundary.

use crate::bitboard::Bitboard;
use crate::types::{Color, PieceType, Square};

// ---------------------------------------------------------------------------
// Const foundations: safe stepping + the naive ray oracle (doc 03 §3.2)
// ---------------------------------------------------------------------------

/// Bit of the target square, or 0 if the step leaves the board or wraps
/// around a board edge.
///
/// A thin const wrapper over [`Square::try_offset`] — P0's single safe
/// stepping primitive — returning the destination as a raw bit for the
/// u64-domain const builders. `try_offset`'s wrap test (Chebyshev distance
/// ≤ 2) and doc 03 §3.2's (file distance ≤ 2 within 0..64) accept exactly
/// the same steps for every step set used here: genuine knight/king/slider
/// steps move at most 2 files and 2 ranks, while a horizontal wrap jumps at
/// least 6 files.
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

/// Walk each direction from `sq` until the board edge or the first blocker;
/// the blocker square is included in the attack set (capture semantics).
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

/// Naive rook attacks: the module's correctness oracle (doc 03 §3.2). The
/// const geometry tables are defined in terms of it, the magic init will
/// verify against it, and the test suite cross-checks everything else
/// against it.
pub(crate) const fn rook_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(ROOK_DIRS, sq, occ)
}

/// Naive bishop attacks; see [`rook_attacks_naive`].
pub(crate) const fn bishop_attacks_naive(sq: u8, occ: u64) -> u64 {
    sliding_attack(BISHOP_DIRS, sq, occ)
}

// ---------------------------------------------------------------------------
// Leaper tables (doc 03 §3.3)
// ---------------------------------------------------------------------------

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

/// Set-wise pawn attacks for all `pawns` of `color` at once; `const` so
/// movegen and eval can use it anywhere, including other const builders.
///
/// Delegates to [`Bitboard::pawn_attacks`] (doc 02 §3.5) — the same masked
/// shifts doc 03 §3.3 spells out: white attacks NW (+7, illegal from file A)
/// and NE (+9, illegal from file H); black mirrors.
pub const fn pawn_attacks_bb(color: Color, pawns: Bitboard) -> Bitboard {
    pawns.pawn_attacks(color)
}

/// Attacks FROM `sq` BY a pawn of the indexing color:
/// `PAWN_ATTACKS[color as usize][square index]`.
///
/// Defined for all 64 squares, including the ranks where no pawn can stand
/// (rank 8 for White, rank 1 for Black — the shift leaves the board, so
/// those entries are geometrically correct and empty; harmless, per SF).
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

// ---------------------------------------------------------------------------
// Pseudo (empty-board) slider tables (doc 03 §3.3)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Ray geometry: BETWEEN, LINE, RAY_PASS, DISTANCE (doc 03 §3.4)
// ---------------------------------------------------------------------------
//
// All four 64×64 tables are built at compile time from the naive generator:
// the defining formulas only need slider attacks with zero-or-one blockers,
// which `*_attacks_naive` computes in const context. Outputs are bit-identical
// to Stockfish's. The tables are private; consume them through `between`/
// `line`/`ray_pass`/`distance` below.

/// Which slider (if any) aligns `s1` with `s2`: the per-pair dispatch of the
/// geometry builders.
///
/// Two distinct squares are aligned on at most one of bishop/rook, and a
/// square is never aligned with itself (empty-board rays exclude their
/// origin, so `s1 == s2` falls out as `Unaligned`). Reading the `PSEUDO_*`
/// statics from const context is fine since Rust 1.83 (`const_refs_to_static`).
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

/// `BETWEEN[s1][s2]`: squares strictly between the pair — each endpoint
/// blocking the other's ray — OR'd with `bit(s2)` **unconditionally**, so
/// every entry contains the target square (SF semantics; rationale on
/// [`between`]). Unaligned pairs and `s1 == s2` yield exactly `{s2}`.
const fn between_entry(s1: usize, s2: usize) -> u64 {
    let (bit1, bit2) = (1u64 << s1, 1u64 << s2);
    let strict = match ray_alignment(s1, s2) {
        Ray::Bishop => bishop_attacks_naive(s1 as u8, bit2) & bishop_attacks_naive(s2 as u8, bit1),
        Ray::Rook => rook_attacks_naive(s1 as u8, bit2) & rook_attacks_naive(s2 as u8, bit1),
        Ray::Unaligned => 0,
    };
    strict | bit2
}

/// `LINE[s1][s2]`: the full edge-to-edge line through both squares (the
/// intersection of their empty-board rays) plus both endpoints; 0 if
/// unaligned. Symmetric; `LINE[s][s]` is empty.
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
/// it to the board edge; 0 if unaligned. `s1`'s empty-board ray masked to
/// the `s2` side: everything `s2` sees looking back until it hits `s1`,
/// plus `s2` itself.
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

/// Selector for [`build_pair_table`]: const contexts cannot call through
/// `fn` pointers, so the shared builder loop dispatches on an enum instead.
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

/// Chebyshev distances, built from the P0 reference implementation
/// [`Square::distance`] (which stays the reference; doc 02 §9.3, doc 03
/// §3.4). Access via [`distance`].
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

// ---------------------------------------------------------------------------
// Accessors (doc 03 §3.1; all #[inline] table lookups)
// ---------------------------------------------------------------------------

/// Squares a knight on `sq` attacks.
#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.index() as usize]
}

/// Squares a king on `sq` attacks.
#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.index() as usize]
}

/// Squares a pawn of `color` on `sq` attacks.
#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color as usize][sq.index() as usize]
}

/// Empty-board attacks for `pt` on `sq`. Debug-asserts `pt != Pawn` — pawns
/// need a color, use [`pawn_attacks`].
#[inline]
pub fn pseudo_attacks(pt: PieceType, sq: Square) -> Bitboard {
    debug_assert!(
        !matches!(pt, PieceType::Pawn),
        "use pawn_attacks(color, sq)"
    );
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::King => king_attacks(sq),
        PieceType::Bishop => PSEUDO_BISHOP[sq.index() as usize],
        PieceType::Rook => PSEUDO_ROOK[sq.index() as usize],
        PieceType::Queen => PSEUDO_QUEEN[sq.index() as usize],
        PieceType::Pawn => Bitboard::EMPTY, // unreachable in debug
    }
}

/// Squares strictly between `s1` and `s2`, OR'd with the **target square
/// `s2`** — SF's exact semantics (doc 03 §3.4). Every result contains `s2`;
/// for unaligned pairs (and `s1 == s2`) it is exactly `{s2}`. NOT symmetric.
///
/// The unconditional `| s2` is why the dominant consumers need no extra
/// work: `between(king_sq, checker_sq)` is directly the "block the check or
/// capture the checker" evasion mask, and `between(king_sq, pinner_sq)`
/// keeps the pinner capturable in pin handling.
#[inline]
pub fn between(s1: Square, s2: Square) -> Bitboard {
    BETWEEN[s1.index() as usize][s2.index() as usize]
}

/// The full line through both squares, extended to the board edges and
/// including both endpoints; `EMPTY` if the squares are not aligned on a
/// rank, file, or diagonal (including `s1 == s2`). Symmetric.
#[inline]
pub fn line(s1: Square, s2: Square) -> Bitboard {
    LINE[s1.index() as usize][s2.index() as usize]
}

/// The ray from `s1` (exclusive) through `s2` and past it to the board
/// edge; `EMPTY` if not aligned. For aligned pairs it contains everything
/// strictly between plus `s2` and beyond, so `between(s1, s2)` ⊆
/// `ray_pass(s1, s2)`. Consumer: NNUE discovered-attack rays (doc 03 §3.4).
#[inline]
pub fn ray_pass(s1: Square, s2: Square) -> Bitboard {
    RAY_PASS[s1.index() as usize][s2.index() as usize]
}

/// Chebyshev distance: `max(|Δfile|, |Δrank|)` — king moves between the
/// squares. Table-backed; [`Square::distance`] is the const reference
/// implementation the table is built from.
#[inline]
pub fn distance(s1: Square, s2: Square) -> u8 {
    DISTANCE[s1.index() as usize][s2.index() as usize]
}

/// True if `s3` lies on the (extended) line through `s1` and `s2` — the
/// movegen legality workhorse ("does this move stay on the pin ray").
#[inline]
pub fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    line(s1, s2).contains(s3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Direction;

    /// Readable expected sets: `bb(&[Square::B3, Square::C2])`.
    fn bb(squares: &[Square]) -> Bitboard {
        squares.iter().copied().collect()
    }

    /// Tiny SplitMix64 PRNG for deterministic random occupancies without a
    /// dev-dependency (same pattern as bitboard.rs, D15).
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

    /// Knight attacks as two-step `Bitboard::shift` compositions; every
    /// shift masks the wrapping file itself, so this path is independent of
    /// the table builders' stepping primitive.
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
    /// the reference model for the naive generator (which is itself the
    /// oracle for everything later, so it gets its own independent check).
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

    // -- §5.1 hand-verified sets ---------------------------------------------

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

    // -- §5.1 count properties ------------------------------------------------

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
        // The extremes pin the distribution's ends.
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
                    // No pawn can stand here; the geometrically correct
                    // shift leaves the board (doc 03 §3.3).
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

    // -- §5.1 edge wrapping -----------------------------------------------------

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

    // -- §5.1 set-wise pawn attacks --------------------------------------------

    #[test]
    fn pawn_attacks_bb_matches_per_square_union() {
        for color in Color::ALL {
            // Full-rank sets (the doc's case) ...
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

    // -- §5.2 geometry: independent coordinate-walk models ------------------------
    //
    // These recompute the geometry in (file, rank) coordinates and never touch
    // `safe_destination`, the naive ray generator, or any table — a fully
    // independent path to the same sets.

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

    // -- §5.2 hand-verified geometry sets ------------------------------------------

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

    // -- §5.2 documented edge cases (doc 03 §4 step 2) ------------------------------

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

    // -- §5.2 exhaustive property tests over all 4096 ordered pairs ------------------

    #[test]
    fn between_includes_target_always() {
        for s1 in Square::all() {
            for s2 in Square::all() {
                let b = between(s1, s2);
                assert!(b.contains(s2), "between({s1}, {s2}) misses {s2}");
                // Exactly {s2} whenever there is nothing strictly between:
                // unaligned, equal, or adjacent-aligned pairs (§5.2).
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
        // The doc 03 §3.4 defining formulas, recomputed test-side through the
        // independent step-walker (`slider_step_model`, which the step-1 tests
        // prove equal to the naive generator).
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
                // Extension to the edges: any point of the line regenerates
                // the whole line — the table stores edge-to-edge lines, never
                // segments (§5.2).
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
                // Recomputed inline (§5.2) ...
                let expected = s1
                    .file()
                    .abs_diff(s2.file())
                    .max(s1.rank().abs_diff(s2.rank()));
                assert_eq!(d, expected, "distance({s1}, {s2})");
                // ... and against the P0 reference implementation the table
                // is built from.
                assert_eq!(d, s1.distance(s2), "table vs Square::distance");
            }
        }
    }

    // -- §5.2 collinearity ------------------------------------------------------------

    #[test]
    fn aligned_collinearity_cases() {
        // Diagonal, file, and rank lines; s3 order doesn't matter.
        assert!(aligned(Square::A1, Square::C3, Square::E5));
        assert!(aligned(Square::A1, Square::C3, Square::B2));
        assert!(aligned(Square::E4, Square::E8, Square::E1));
        assert!(aligned(Square::A4, Square::H4, Square::C4));
        // s3 off the line.
        assert!(!aligned(Square::A1, Square::C3, Square::D3));
        // Unaligned generators span no line at all — nothing is collinear
        // with them, not even the endpoints themselves.
        assert!(!aligned(Square::A1, Square::B3, Square::A1));
        assert!(!aligned(Square::A1, Square::B3, Square::C5));

        // Exhaustive agreement with the coordinate model over all triples.
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
}
