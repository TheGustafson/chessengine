use crate::types::{Color, Direction, Square};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

/// A set of squares, one bit per square: a1 = bit 0, h8 = bit 63.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const FULL: Bitboard = Bitboard(u64::MAX);

    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    /// FILES[0] is the a-file, FILES[7] the h-file.
    pub const FILES: [Bitboard; 8] = {
        let mut files = [Bitboard::EMPTY; 8];
        let mut i = 0;
        while i < 8 {
            files[i] = Bitboard(Bitboard::FILE_A.0 << i);
            i += 1;
        }
        files
    };

    /// RANKS[0] is rank 1, RANKS[7] rank 8.
    pub const RANKS: [Bitboard; 8] = {
        let mut ranks = [Bitboard::EMPTY; 8];
        let mut i = 0;
        while i < 8 {
            ranks[i] = Bitboard(Bitboard::RANK_1.0 << (8 * i));
            i += 1;
        }
        ranks
    };

    pub const fn from_u64(bits: u64) -> Bitboard {
        Bitboard(bits)
    }

    pub const fn to_u64(self) -> u64 {
        self.0
    }

    pub const fn from_square(square: Square) -> Bitboard {
        Bitboard(1u64 << square.index())
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of set squares (popcount).
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub const fn contains(self, square: Square) -> bool {
        (self.0 & (1u64 << square.index())) != 0
    }

    /// The lowest set square, if any.
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::new(self.0.trailing_zeros() as u8))
        }
    }

    /// Remove and return the lowest set square.
    pub fn pop_lsb(&mut self) -> Option<Square> {
        let sq = self.lsb()?;
        self.0 &= self.0 - 1;
        Some(sq)
    }

    /// Iterate over the set squares in ascending order.
    ///
    /// `Bitboard` also implements `IntoIterator`, so `for sq in bb` works;
    /// this method is for iterator chains where naming the source reads better.
    pub fn squares(self) -> BitboardIter {
        BitboardIter(self)
    }

    /// True iff at least two bits are set. One AND + compare — cheaper and
    /// more explicit than `count() >= 2`. Primary uses: double-check
    /// detection (evasion generation), "lone attacker" tests in SEE
    /// (design/02-core-types.md §3.5).
    ///
    /// `wrapping_sub` keeps this panic-free on `EMPTY` in debug builds
    /// (`0u64 - 1` would be an overflow panic); `u64::MAX & 0 == 0` still
    /// gives the right answer.
    pub const fn more_than_one(self) -> bool {
        (self.0 & self.0.wrapping_sub(1)) != 0
    }

    /// Shift every set square one step in `d`. Squares that would leave the
    /// board vanish; east/west components mask the wrapping file first
    /// (design/02-core-types.md §3.5).
    ///
    /// Hot call sites must pass a literal (or provably constant) direction —
    /// the match then constant-folds to one AND + one shift, while a
    /// runtime-variant direction in a hot loop becomes a jump table
    /// (design/02 §6).
    pub const fn shift(self, d: Direction) -> Bitboard {
        // Operator traits (`BitAnd` etc.) are not `const fn`, so const
        // helpers use the raw u64 (fine inside bitboard.rs).
        const NOT_FILE_A: u64 = !Bitboard::FILE_A.to_u64(); // 0xFEFE_FEFE_FEFE_FEFE
        const NOT_FILE_H: u64 = !Bitboard::FILE_H.to_u64(); // 0x7F7F_7F7F_7F7F_7F7F
        Bitboard::from_u64(match d {
            Direction::North => self.0 << 8,
            Direction::South => self.0 >> 8,
            Direction::East => (self.0 & NOT_FILE_H) << 1,
            Direction::West => (self.0 & NOT_FILE_A) >> 1,
            Direction::NorthEast => (self.0 & NOT_FILE_H) << 9,
            Direction::NorthWest => (self.0 & NOT_FILE_A) << 7,
            Direction::SouthEast => (self.0 & NOT_FILE_H) >> 7,
            Direction::SouthWest => (self.0 & NOT_FILE_A) >> 9,
        })
    }

    /// Single pawn pushes for `color` pawns in `self`.
    ///
    /// Deliberately does NOT mask occupancy: movegen ANDs with `!occupied`
    /// itself, because double pushes need the intermediate emptiness test
    /// anyway (design/02-core-types.md §3.5).
    pub const fn pawn_pushes(self, color: Color) -> Bitboard {
        self.shift(color.pawn_push())
    }

    /// Squares attacked by `color` pawns in `self` (both diagonals, bulk).
    pub const fn pawn_attacks(self, color: Color) -> Bitboard {
        match color {
            Color::White => Bitboard::from_u64(
                self.shift(Direction::NorthEast).0 | self.shift(Direction::NorthWest).0,
            ),
            Color::Black => Bitboard::from_u64(
                self.shift(Direction::SouthEast).0 | self.shift(Direction::SouthWest).0,
            ),
        }
    }
}

/// Iterator over the squares of a [`Bitboard`], lowest square first.
#[derive(Debug, Clone)]
pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        self.0.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitboardIter {}
impl std::iter::FusedIterator for BitboardIter {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    fn into_iter(self) -> BitboardIter {
        BitboardIter(self)
    }
}

impl FromIterator<Square> for Bitboard {
    fn from_iter<I: IntoIterator<Item = Square>>(iter: I) -> Bitboard {
        iter.into_iter()
            .fold(Bitboard::EMPTY, |bb, sq| bb | Bitboard::from_square(sq))
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Shl<u32> for Bitboard {
    type Output = Bitboard;
    fn shl(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<u32> for Bitboard {
    type Output = Bitboard;
    fn shr(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(0x{:016x})", self.0)
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::from_coords(file, rank);
                let c = if self.contains(sq) { 'X' } else { '.' };
                if file > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{c}")?;
            }
            if rank > 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_square() {
        assert_eq!(Bitboard::from_square(Square::A1).to_u64(), 1);
        assert_eq!(Bitboard::from_square(Square::H8).to_u64(), 1 << 63);
    }

    #[test]
    fn contains() {
        let bb = Bitboard::from_square(Square::A1);
        assert!(bb.contains(Square::A1));
        assert!(!bb.contains(Square::B1));
    }

    #[test]
    fn count_and_is_empty() {
        assert!(Bitboard::EMPTY.is_empty());
        assert_eq!(Bitboard::EMPTY.count(), 0);
        assert_eq!(Bitboard::FULL.count(), 64);

        let bb = Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::B1);
        assert_eq!(bb.count(), 2);
        assert!(!bb.is_empty());
    }

    #[test]
    fn iteration_is_ascending_and_exact() {
        let bb: Bitboard = [Square::H8, Square::A1, Square::E4].into_iter().collect();
        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares, vec![Square::A1, Square::E4, Square::H8]);

        let iter = bb.squares();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn from_iterator_roundtrip() {
        let bb: Bitboard = Square::all().collect();
        assert_eq!(bb, Bitboard::FULL);
    }

    #[test]
    fn lsb_and_pop_lsb() {
        assert_eq!(Bitboard::EMPTY.lsb(), None);

        let mut bb: Bitboard = [Square::C3, Square::F7].into_iter().collect();
        assert_eq!(bb.lsb(), Some(Square::C3));
        assert_eq!(bb.pop_lsb(), Some(Square::C3));
        assert_eq!(bb.pop_lsb(), Some(Square::F7));
        assert_eq!(bb.pop_lsb(), None);
    }

    #[test]
    fn file_and_rank_constants() {
        assert_eq!(Bitboard::FILE_A.count(), 8);
        assert_eq!(Bitboard::RANK_1.count(), 8);
        assert!(Bitboard::FILE_A.contains(Square::A1));
        assert!(Bitboard::FILE_A.contains(Square::A8));
        assert!(Bitboard::FILE_H.contains(Square::H4));
        assert!(Bitboard::RANK_1.contains(Square::H1));
        assert!(Bitboard::RANK_8.contains(Square::A8));

        for i in 0..8u8 {
            assert_eq!(Bitboard::FILES[i as usize].count(), 8);
            assert_eq!(Bitboard::RANKS[i as usize].count(), 8);
            assert!(Bitboard::FILES[i as usize].contains(Square::from_coords(i, 3)));
            assert!(Bitboard::RANKS[i as usize].contains(Square::from_coords(3, i)));
        }

        let all_files: Bitboard = Bitboard::FILES
            .into_iter()
            .fold(Bitboard::EMPTY, |acc, f| acc | f);
        assert_eq!(all_files, Bitboard::FULL);
    }

    #[test]
    fn set_operations() {
        let a1 = Bitboard::from_square(Square::A1);
        let b1 = Bitboard::from_square(Square::B1);
        let both = a1 | b1;

        assert_eq!(both & a1, a1);
        assert_eq!(both ^ a1, b1);
        assert_eq!(both & !a1, b1);
        assert_eq!(!Bitboard::EMPTY, Bitboard::FULL);

        let mut bb = a1;
        bb |= b1;
        assert_eq!(bb, both);
        bb &= a1;
        assert_eq!(bb, a1);
        bb ^= a1;
        assert_eq!(bb, Bitboard::EMPTY);
    }

    #[test]
    fn shifts() {
        // Shifting rank 1 up by one rank gives rank 2.
        assert_eq!(Bitboard::RANK_1 << 8, Bitboard::RANKS[1]);
        assert_eq!(Bitboard::RANK_8 >> 8, Bitboard::RANKS[6]);
    }

    /// Tiny SplitMix64 PRNG: deterministic random bitboards for in-module
    /// equivalence tests without a dev-dependency (doc 02 §5 item 9, D15).
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

    // Doc 02 §5 item 9: fixed shift edges — wrapping files vanish, off-board
    // ranks vanish.
    #[test]
    fn shift_by_direction_edges() {
        assert_eq!(Bitboard::RANK_1.shift(Direction::North), Bitboard::RANKS[1]);
        assert_eq!(Bitboard::FILE_H.shift(Direction::East), Bitboard::EMPTY);
        assert_eq!(Bitboard::FULL.shift(Direction::East), !Bitboard::FILE_A);
        assert_eq!(Bitboard::FULL.shift(Direction::West), !Bitboard::FILE_H);
        assert_eq!(Bitboard::RANK_8.shift(Direction::North), Bitboard::EMPTY);
        assert_eq!(Bitboard::RANK_1.shift(Direction::South), Bitboard::EMPTY);
        assert_eq!(
            Bitboard::from_square(Square::E4).shift(Direction::NorthEast),
            Bitboard::from_square(Square::F5)
        );
    }

    // Doc 02 §5 item 9, randomized half: the bulk shift equals the union of
    // per-square checked stepping, for 1000 random boards × all 8 directions.
    #[test]
    fn shift_matches_per_square_shifted_on_random_boards() {
        let mut rng = SplitMix64(0x0BAD_5EED_0BAD_5EED);
        for _ in 0..1000 {
            let bb = Bitboard::from_u64(rng.next_u64());
            for d in Direction::ALL {
                let expected: Bitboard = bb.squares().filter_map(|sq| sq.shifted(d)).collect();
                assert_eq!(bb.shift(d), expected, "shift({d:?}) on {bb:?}");
            }
        }
    }

    // Doc 02 §5 item 10: pawn helpers against per-square models, both colors.
    #[test]
    fn pawn_helpers_match_per_square_models() {
        let mut rng = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..1000 {
            let bb = Bitboard::from_u64(rng.next_u64());
            for color in Color::ALL {
                let pushes: Bitboard = bb
                    .squares()
                    .filter_map(|sq| sq.shifted(color.pawn_push()))
                    .collect();
                assert_eq!(bb.pawn_pushes(color), pushes, "pushes for {color:?}");

                let (east, west) = match color {
                    Color::White => (Direction::NorthEast, Direction::NorthWest),
                    Color::Black => (Direction::SouthEast, Direction::SouthWest),
                };
                let attacks: Bitboard = bb
                    .squares()
                    .flat_map(|sq| [sq.shifted(east), sq.shifted(west)])
                    .flatten()
                    .collect();
                assert_eq!(bb.pawn_attacks(color), attacks, "attacks for {color:?}");
            }
        }
        // Edge case pinned by the doc: an a-file white pawn attacks only NE
        // (and, mirrored, an h-file black pawn only SW).
        assert_eq!(
            Bitboard::from_square(Square::A4).pawn_attacks(Color::White),
            Bitboard::from_square(Square::B5)
        );
        assert_eq!(
            Bitboard::from_square(Square::H5).pawn_attacks(Color::Black),
            Bitboard::from_square(Square::G4)
        );
    }

    // Doc 02 §5 item 10: more_than_one over the boundary cases.
    #[test]
    fn more_than_one_cases() {
        assert!(!Bitboard::EMPTY.more_than_one());
        for sq in Square::all() {
            assert!(!Bitboard::from_square(sq).more_than_one());
        }
        let two = Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::H8);
        assert!(two.more_than_one());
        assert!(Bitboard::FULL.more_than_one());
    }

    // Doc 02 §8 criterion 6 (this step's part): downstream docs can build
    // const tables on Bitboard::shift.
    #[test]
    fn shift_is_const_evaluable() {
        const RANK_2: Bitboard = Bitboard::RANK_1.shift(Direction::North);
        assert_eq!(RANK_2, Bitboard::RANKS[1]);
    }
}
