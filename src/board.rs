use crate::bitboard::Bitboard;
use crate::types::{Color, Piece, PieceType, Square};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);
    pub const WHITE_KINGSIDE: CastlingRights = CastlingRights(0b0001);
    pub const WHITE_QUEENSIDE: CastlingRights = CastlingRights(0b0010);
    pub const BLACK_KINGSIDE: CastlingRights = CastlingRights(0b0100);
    pub const BLACK_QUEENSIDE: CastlingRights = CastlingRights(0b1000);
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    pub fn add(&mut self, rights: CastlingRights) {
        self.0 |= rights.0;
    }

    pub fn remove(&mut self, rights: CastlingRights) {
        self.0 &= !rights.0;
    }

    pub const fn contains(self, rights: CastlingRights) -> bool {
        (self.0 & rights.0) == rights.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// FEN castling field: "KQkq", any sub-combination, or "-".
impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "-");
        }
        if self.contains(CastlingRights::WHITE_KINGSIDE) {
            write!(f, "K")?;
        }
        if self.contains(CastlingRights::WHITE_QUEENSIDE) {
            write!(f, "Q")?;
        }
        if self.contains(CastlingRights::BLACK_KINGSIDE) {
            write!(f, "k")?;
        }
        if self.contains(CastlingRights::BLACK_QUEENSIDE) {
            write!(f, "q")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCastlingRightsError;

impl fmt::Display for ParseCastlingRightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected \"-\" or a non-repeating combination of K, Q, k, q"
        )
    }
}

impl std::error::Error for ParseCastlingRightsError {}

impl FromStr for CastlingRights {
    type Err = ParseCastlingRightsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "-" {
            return Ok(CastlingRights::NONE);
        }
        if s.is_empty() {
            return Err(ParseCastlingRightsError);
        }
        let mut rights = CastlingRights::NONE;
        for c in s.chars() {
            let right = match c {
                'K' => CastlingRights::WHITE_KINGSIDE,
                'Q' => CastlingRights::WHITE_QUEENSIDE,
                'k' => CastlingRights::BLACK_KINGSIDE,
                'q' => CastlingRights::BLACK_QUEENSIDE,
                _ => return Err(ParseCastlingRightsError),
            };
            if rights.contains(right) {
                return Err(ParseCastlingRightsError);
            }
            rights.add(right);
        }
        Ok(rights)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    // Primary representation: bitboards.
    pub(crate) pieces: [[Bitboard; PieceType::COUNT]; Color::COUNT],
    pub(crate) color_occupancy: [Bitboard; Color::COUNT],
    pub(crate) all_occupancy: Bitboard,

    // Redundant mailbox for O(1) piece lookup by square.
    pub(crate) mailbox: [Option<Piece>; Square::COUNT],

    // Game state.
    pub(crate) side_to_move: Color,
    pub(crate) castling_rights: CastlingRights,
    pub(crate) en_passant_square: Option<Square>,
    pub(crate) halfmove_clock: u32,
    pub(crate) fullmove_number: u32,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            pieces: [[Bitboard::EMPTY; PieceType::COUNT]; Color::COUNT],
            color_occupancy: [Bitboard::EMPTY; Color::COUNT],
            all_occupancy: Bitboard::EMPTY,
            mailbox: [None; Square::COUNT],
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index() as usize]
    }

    /// The squares holding `color`'s pieces of type `piece_type`.
    pub fn pieces(&self, color: Color, piece_type: PieceType) -> Bitboard {
        self.pieces[color as usize][piece_type as usize]
    }

    /// All squares occupied by `color`.
    pub fn occupancy(&self, color: Color) -> Bitboard {
        self.color_occupancy[color as usize]
    }

    /// All occupied squares.
    pub fn all_occupancy(&self) -> Bitboard {
        self.all_occupancy
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    pub fn en_passant_square(&self) -> Option<Square> {
        self.en_passant_square
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    pub fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    /// Place `piece` on the empty square `sq`, keeping bitboards and mailbox in sync.
    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        debug_assert!(
            self.piece_on(sq).is_none(),
            "put_piece on occupied square {sq}"
        );
        let bb = Bitboard::from_square(sq);
        self.pieces[piece.color as usize][piece.piece_type as usize] |= bb;
        self.color_occupancy[piece.color as usize] |= bb;
        self.all_occupancy |= bb;
        self.mailbox[sq.index() as usize] = Some(piece);
    }

    /// Remove and return the piece on `sq`, keeping bitboards and mailbox in sync.
    pub fn remove_piece(&mut self, sq: Square) -> Option<Piece> {
        let piece = self.mailbox[sq.index() as usize].take()?;
        let bb = Bitboard::from_square(sq);
        self.pieces[piece.color as usize][piece.piece_type as usize] ^= bb;
        self.color_occupancy[piece.color as usize] ^= bb;
        self.all_occupancy ^= bb;
        Some(piece)
    }

    /// Check that the bitboards and mailbox agree. Test-only invariant check.
    #[cfg(test)]
    pub(crate) fn assert_consistent(&self) {
        let mut white = Bitboard::EMPTY;
        let mut black = Bitboard::EMPTY;
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let bb = self.pieces(color, piece_type);
                for sq in bb {
                    assert_eq!(
                        self.piece_on(sq),
                        Some(Piece::new(color, piece_type)),
                        "mailbox disagrees with bitboard on {sq}"
                    );
                }
                match color {
                    Color::White => white |= bb,
                    Color::Black => black |= bb,
                }
            }
        }
        assert_eq!(white, self.occupancy(Color::White));
        assert_eq!(black, self.occupancy(Color::Black));
        assert_eq!(white | black, self.all_occupancy());
        assert!((white & black).is_empty());
        let mailbox_count = self.mailbox.iter().filter(|p| p.is_some()).count();
        assert_eq!(mailbox_count as u32, self.all_occupancy().count());
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::empty()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{}", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_coords(file, rank);
                match self.piece_on(sq) {
                    Some(piece) => write!(f, " {piece}")?,
                    None => write!(f, " .")?,
                }
            }
            writeln!(f)?;
        }
        write!(f, " ")?;
        for file in 0..8u8 {
            write!(f, " {}", (b'a' + file) as char)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_piece_updates_all_layers() {
        let mut board = Board::empty();
        let piece = Piece::new(Color::White, PieceType::Pawn);
        board.put_piece(piece, Square::E2);

        assert_eq!(board.piece_on(Square::E2), Some(piece));
        assert!(
            board
                .pieces(Color::White, PieceType::Pawn)
                .contains(Square::E2)
        );
        assert!(board.occupancy(Color::White).contains(Square::E2));
        assert!(board.all_occupancy().contains(Square::E2));
        assert!(!board.occupancy(Color::Black).contains(Square::E2));
        board.assert_consistent();
    }

    #[test]
    fn remove_piece_reverses_put_piece() {
        let mut board = Board::empty();
        let piece = Piece::new(Color::Black, PieceType::Knight);
        board.put_piece(piece, Square::G8);

        assert_eq!(board.remove_piece(Square::G8), Some(piece));
        assert_eq!(board.piece_on(Square::G8), None);
        assert!(board.all_occupancy().is_empty());
        assert!(board.pieces(Color::Black, PieceType::Knight).is_empty());
        board.assert_consistent();

        // Removing from an empty square is a no-op.
        assert_eq!(board.remove_piece(Square::G8), None);
    }

    #[test]
    fn castling_rights_operations() {
        let mut rights = CastlingRights::NONE;
        assert!(rights.is_empty());

        rights.add(CastlingRights::WHITE_KINGSIDE);
        rights.add(CastlingRights::BLACK_QUEENSIDE);
        assert!(rights.contains(CastlingRights::WHITE_KINGSIDE));
        assert!(rights.contains(CastlingRights::BLACK_QUEENSIDE));
        assert!(!rights.contains(CastlingRights::WHITE_QUEENSIDE));
        assert!(!rights.contains(CastlingRights::ALL));

        rights.remove(CastlingRights::WHITE_KINGSIDE);
        assert!(!rights.contains(CastlingRights::WHITE_KINGSIDE));
        assert!(rights.contains(CastlingRights::BLACK_QUEENSIDE));

        assert!(CastlingRights::ALL.contains(CastlingRights::BLACK_KINGSIDE));
    }

    #[test]
    fn castling_rights_display_parse_roundtrip() {
        for s in ["-", "K", "Qk", "KQkq", "Kq", "kq"] {
            let rights: CastlingRights = s.parse().unwrap();
            assert_eq!(rights.to_string(), s);
        }
        assert!("".parse::<CastlingRights>().is_err());
        assert!("KK".parse::<CastlingRights>().is_err());
        assert!("KQx".parse::<CastlingRights>().is_err());
    }

    #[test]
    fn display_empty_board() {
        let board = Board::empty();
        let expected = "\
8 . . . . . . . .
7 . . . . . . . .
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 . . . . . . . .
1 . . . . . . . .
  a b c d e f g h";
        assert_eq!(board.to_string(), expected);
    }

    #[test]
    fn display_shows_pieces_at_correct_coordinates() {
        let mut board = Board::empty();
        board.put_piece(Piece::new(Color::White, PieceType::King), Square::E1);
        board.put_piece(Piece::new(Color::Black, PieceType::King), Square::E8);
        board.put_piece(Piece::new(Color::White, PieceType::Queen), Square::A4);

        let expected = "\
8 . . . . k . . .
7 . . . . . . . .
6 . . . . . . . .
5 . . . . . . . .
4 Q . . . . . . .
3 . . . . . . . .
2 . . . . . . . .
1 . . . . K . . .
  a b c d e f g h";
        assert_eq!(board.to_string(), expected);
    }
}
