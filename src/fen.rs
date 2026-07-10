use crate::board::Board;
use crate::types::{Color, Piece, Square};
use std::fmt;
use std::fmt::Write as _;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    /// FEN must have 4 to 6 whitespace-separated fields.
    FieldCount(usize),
    /// The placement field must have exactly 8 ranks separated by '/'.
    RankCount(usize),
    /// A character in the placement field is neither a piece nor a digit 1-8.
    InvalidPiece(char),
    /// A rank's pieces and empty-square digits don't add up to 8 files.
    RankWidth {
        rank: u8,
        width: u8,
    },
    SideToMove(String),
    Castling(String),
    EnPassant(String),
    Clock(String),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::FieldCount(n) => {
                write!(f, "expected 4 to 6 fields, found {n}")
            }
            FenError::RankCount(n) => {
                write!(f, "expected 8 ranks in the placement field, found {n}")
            }
            FenError::InvalidPiece(c) => {
                write!(f, "invalid character {c:?} in the placement field")
            }
            FenError::RankWidth { rank, width } => {
                write!(f, "rank {rank} covers {width} files, expected 8")
            }
            FenError::SideToMove(s) => {
                write!(f, "invalid side to move {s:?}, expected \"w\" or \"b\"")
            }
            FenError::Castling(s) => write!(f, "invalid castling field {s:?}"),
            FenError::EnPassant(s) => write!(f, "invalid en passant field {s:?}"),
            FenError::Clock(s) => write!(f, "invalid move clock {s:?}"),
        }
    }
}

impl std::error::Error for FenError {}

impl Board {
    /// The standard chess starting position.
    pub fn startpos() -> Board {
        Board::from_fen(START_FEN).expect("start position FEN is valid")
    }

    /// Parse a position from Forsyth-Edwards Notation.
    ///
    /// The two move-clock fields may be omitted (as in EPD), defaulting to
    /// halfmove 0 and fullmove 1.
    pub fn from_fen(fen: &str) -> Result<Board, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if !(4..=6).contains(&fields.len()) {
            return Err(FenError::FieldCount(fields.len()));
        }

        let mut board = Board::empty();

        // Field 1: piece placement, rank 8 first.
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::RankCount(ranks.len()));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file: u8 = 0;
            for c in rank_str.chars() {
                if let Some(skip) = c.to_digit(10) {
                    if !(1..=8).contains(&skip) {
                        return Err(FenError::InvalidPiece(c));
                    }
                    file += skip as u8;
                } else {
                    let piece = Piece::from_char(c).ok_or(FenError::InvalidPiece(c))?;
                    if file >= 8 {
                        return Err(FenError::RankWidth {
                            rank: rank + 1,
                            width: file + 1,
                        });
                    }
                    board.put_piece(piece, Square::from_coords(file, rank));
                    file += 1;
                }
            }
            if file != 8 {
                return Err(FenError::RankWidth {
                    rank: rank + 1,
                    width: file,
                });
            }
        }

        // Field 2: side to move.
        board.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            s => return Err(FenError::SideToMove(s.to_string())),
        };

        // Field 3: castling rights.
        board.castling_rights = fields[2]
            .parse()
            .map_err(|_| FenError::Castling(fields[2].to_string()))?;

        // Field 4: en passant target square.
        board.en_passant_square = match fields[3] {
            "-" => None,
            s => {
                let sq: Square = s.parse().map_err(|_| FenError::EnPassant(s.to_string()))?;
                if sq.rank() != 2 && sq.rank() != 5 {
                    return Err(FenError::EnPassant(s.to_string()));
                }
                Some(sq)
            }
        };

        // Fields 5 and 6: halfmove clock and fullmove number, optional.
        board.halfmove_clock = match fields.get(4) {
            Some(s) => s.parse().map_err(|_| FenError::Clock(s.to_string()))?,
            None => 0,
        };
        board.fullmove_number = match fields.get(5) {
            Some(s) => s.parse().map_err(|_| FenError::Clock(s.to_string()))?,
            None => 1,
        };

        Ok(board)
    }

    /// Serialize the position to Forsyth-Edwards Notation.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_run = 0;
            for file in 0..8 {
                match self.piece_on(Square::from_coords(file, rank)) {
                    Some(piece) => {
                        if empty_run > 0 {
                            write!(fen, "{empty_run}").unwrap();
                            empty_run = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => empty_run += 1,
                }
            }
            if empty_run > 0 {
                write!(fen, "{empty_run}").unwrap();
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        let side = match self.side_to_move() {
            Color::White => 'w',
            Color::Black => 'b',
        };
        write!(fen, " {side} {}", self.castling_rights()).unwrap();

        match self.en_passant_square() {
            Some(sq) => write!(fen, " {sq}").unwrap(),
            None => fen.push_str(" -"),
        }

        write!(fen, " {} {}", self.halfmove_clock(), self.fullmove_number()).unwrap();

        fen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::CastlingRights;
    use crate::types::PieceType;

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn startpos_has_expected_state() {
        let board = Board::startpos();
        board.assert_consistent();

        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);

        assert_eq!(board.all_occupancy().count(), 32);
        for color in Color::ALL {
            assert_eq!(board.occupancy(color).count(), 16);
            assert_eq!(board.pieces(color, PieceType::Pawn).count(), 8);
            assert_eq!(board.pieces(color, PieceType::King).count(), 1);
            assert_eq!(board.pieces(color, PieceType::Queen).count(), 1);
        }

        assert_eq!(
            board.piece_on(Square::E1),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            board.piece_on(Square::D8),
            Some(Piece::new(Color::Black, PieceType::Queen))
        );
        assert_eq!(
            board.piece_on(Square::A7),
            Some(Piece::new(Color::Black, PieceType::Pawn))
        );
        assert_eq!(board.piece_on(Square::E4), None);
    }

    #[test]
    fn roundtrip_startpos_and_kiwipete() {
        for fen in [START_FEN, KIWIPETE] {
            let board = Board::from_fen(fen).unwrap();
            board.assert_consistent();
            assert_eq!(board.to_fen(), fen);
        }
    }

    #[test]
    fn parses_en_passant_square() {
        // Position after 1. e4.
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.en_passant_square(), Some(Square::E3));
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn parses_partial_state_fields() {
        // EPD-style FEN without the move clocks.
        let board = Board::from_fen("8/8/8/8/8/8/8/4K2k w - -").unwrap();
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);
        assert_eq!(board.castling_rights(), CastlingRights::NONE);

        // Non-default clocks survive a roundtrip.
        let fen = "8/8/8/8/8/8/8/4K2k b - - 42 99";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.halfmove_clock(), 42);
        assert_eq!(board.fullmove_number(), 99);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn rejects_malformed_fens() {
        use FenError::*;

        assert_eq!(Board::from_fen(""), Err(FieldCount(0)));
        assert_eq!(Board::from_fen("8/8/8/8 w - -"), Err(RankCount(4)));
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8/8 w - -"),
            Err(RankCount(9))
        );
        assert_eq!(
            Board::from_fen("x7/8/8/8/8/8/8/8 w - -"),
            Err(InvalidPiece('x'))
        );
        assert_eq!(
            Board::from_fen("9/8/8/8/8/8/8/8 w - -"),
            Err(InvalidPiece('9'))
        );
        assert_eq!(
            Board::from_fen("ppppppppp/8/8/8/8/8/8/8 w - -"),
            Err(RankWidth { rank: 8, width: 9 })
        );
        assert_eq!(
            Board::from_fen("p6/8/8/8/8/8/8/8 w - -"),
            Err(RankWidth { rank: 8, width: 7 })
        );
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 x - -"),
            Err(SideToMove("x".to_string()))
        );
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 w KX -"),
            Err(Castling("KX".to_string()))
        );
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 w - e9"),
            Err(EnPassant("e9".to_string()))
        );
        // En passant target square must be on rank 3 or 6.
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 w - e4"),
            Err(EnPassant("e4".to_string()))
        );
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 w - - x 1"),
            Err(Clock("x".to_string()))
        );
        assert_eq!(
            Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 -3"),
            Err(Clock("-3".to_string()))
        );
    }
}
