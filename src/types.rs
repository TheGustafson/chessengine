use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub const COUNT: usize = 6;
    pub const ALL: [PieceType; PieceType::COUNT] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// Lowercase FEN character for this piece type.
    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }

    /// Parse a lowercase FEN character.
    pub const fn from_char(c: char) -> Option<PieceType> {
        match c {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub const COUNT: usize = 2;
    pub const ALL: [Color; Color::COUNT] = [Color::White, Color::Black];

    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    /// FEN character: uppercase for white, lowercase for black.
    pub fn to_char(self) -> char {
        let c = self.piece_type.to_char();
        match self.color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    /// Parse a FEN character: uppercase for white, lowercase for black.
    pub fn from_char(c: char) -> Option<Piece> {
        let piece_type = PieceType::from_char(c.to_ascii_lowercase())?;
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        Some(Piece::new(color, piece_type))
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A square on the board, 0..64 with a1 = 0, b1 = 1, ..., h8 = 63
/// (little-endian rank-file mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square {
    index: u8,
}

macro_rules! define_square_consts {
    ($($name:ident = $index:expr),* $(,)?) => {
        impl Square {
            $(pub const $name: Square = Square { index: $index };)*
        }
    };
}

define_square_consts!(
    A1 = 0,
    B1 = 1,
    C1 = 2,
    D1 = 3,
    E1 = 4,
    F1 = 5,
    G1 = 6,
    H1 = 7,
    A2 = 8,
    B2 = 9,
    C2 = 10,
    D2 = 11,
    E2 = 12,
    F2 = 13,
    G2 = 14,
    H2 = 15,
    A3 = 16,
    B3 = 17,
    C3 = 18,
    D3 = 19,
    E3 = 20,
    F3 = 21,
    G3 = 22,
    H3 = 23,
    A4 = 24,
    B4 = 25,
    C4 = 26,
    D4 = 27,
    E4 = 28,
    F4 = 29,
    G4 = 30,
    H4 = 31,
    A5 = 32,
    B5 = 33,
    C5 = 34,
    D5 = 35,
    E5 = 36,
    F5 = 37,
    G5 = 38,
    H5 = 39,
    A6 = 40,
    B6 = 41,
    C6 = 42,
    D6 = 43,
    E6 = 44,
    F6 = 45,
    G6 = 46,
    H6 = 47,
    A7 = 48,
    B7 = 49,
    C7 = 50,
    D7 = 51,
    E7 = 52,
    F7 = 53,
    G7 = 54,
    H7 = 55,
    A8 = 56,
    B8 = 57,
    C8 = 58,
    D8 = 59,
    E8 = 60,
    F8 = 61,
    G8 = 62,
    H8 = 63,
);

impl Square {
    pub const COUNT: usize = 64;

    pub const fn new(index: u8) -> Self {
        assert!(index < 64, "Square index must be between 0 and 63");
        Square { index }
    }

    pub const fn from_coords(file: u8, rank: u8) -> Self {
        assert!(
            file < 8 && rank < 8,
            "file and rank must be between 0 and 7"
        );
        Square {
            index: rank * 8 + file,
        }
    }

    pub const fn index(self) -> u8 {
        self.index
    }

    /// File 0-7 (a-h).
    pub const fn file(self) -> u8 {
        self.index & 0b111
    }

    /// Rank 0-7 (1-8).
    pub const fn rank(self) -> u8 {
        self.index >> 3
    }

    /// All 64 squares in index order (a1, b1, ..., h8).
    pub fn all() -> impl Iterator<Item = Square> {
        (0..64).map(Square::new)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSquareError;

impl fmt::Display for ParseSquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected a square like \"e4\" (file a-h, rank 1-8)")
    }
}

impl std::error::Error for ParseSquareError {}

impl FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let [file_char, rank_char] = s.as_bytes() else {
            return Err(ParseSquareError);
        };
        let file = file_char.wrapping_sub(b'a');
        let rank = rank_char.wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Ok(Square::from_coords(file, rank))
        } else {
            Err(ParseSquareError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_file_rank() {
        assert_eq!(Square::A1.file(), 0);
        assert_eq!(Square::A1.rank(), 0);
        assert_eq!(Square::H1.file(), 7);
        assert_eq!(Square::H1.rank(), 0);
        assert_eq!(Square::A8.file(), 0);
        assert_eq!(Square::A8.rank(), 7);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::H8.rank(), 7);
        assert_eq!(Square::E4, Square::from_coords(4, 3));
    }

    #[test]
    fn square_display_parse_roundtrip() {
        for sq in Square::all() {
            let parsed: Square = sq.to_string().parse().unwrap();
            assert_eq!(parsed, sq);
        }
        assert_eq!("e4".parse::<Square>().unwrap(), Square::E4);
        assert_eq!(Square::A1.to_string(), "a1");
        assert_eq!(Square::H8.to_string(), "h8");
    }

    #[test]
    fn square_parse_rejects_garbage() {
        assert!("".parse::<Square>().is_err());
        assert!("e".parse::<Square>().is_err());
        assert!("e44".parse::<Square>().is_err());
        assert!("i1".parse::<Square>().is_err());
        assert!("a9".parse::<Square>().is_err());
        assert!("a0".parse::<Square>().is_err());
    }

    #[test]
    #[should_panic]
    fn square_new_rejects_out_of_range() {
        Square::new(64);
    }

    #[test]
    fn piece_char_roundtrip() {
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let piece = Piece::new(color, piece_type);
                assert_eq!(Piece::from_char(piece.to_char()), Some(piece));
            }
        }
        assert_eq!(
            Piece::from_char('K'),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            Piece::from_char('q'),
            Some(Piece::new(Color::Black, PieceType::Queen))
        );
        assert_eq!(Piece::from_char('x'), None);
        assert_eq!(Piece::from_char('1'), None);
    }

    #[test]
    fn color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }
}
