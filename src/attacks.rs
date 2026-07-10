//! Precomputed attack tables (design/03-attack-tables.md, D12/D13).
//!
//! This module answers exactly one question: *which squares does a piece on
//! `sq` attack given occupancy `occ`?* Legality (pins, checks, castling) is
//! movegen's job (design/05).
//!
//! Landed so far (doc 03 §4 step 1): the `const fn` foundations
//! ([`safe_destination`] + the naive ray generator that serves as the
//! module's correctness oracle) and the compile-time leaper and empty-board
//! slider tables (`KNIGHT_ATTACKS`, `KING_ATTACKS`, `PAWN_ATTACKS`,
//! `PSEUDO_*`) with their accessor functions. Still to come, in doc 03's
//! step order: the ray-geometry tables (`BETWEEN`/`LINE`/`RAY_PASS`/
//! `DISTANCE`, step 2) and the fancy-magic slider machinery behind
//! `LazyLock<AttackTables>` (steps 3–4).
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
}
