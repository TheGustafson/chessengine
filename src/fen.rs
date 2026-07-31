use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::{Board, CastlingRights};
use crate::types::{Color, Piece, PieceType, Square};
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

        // Field 3: castling rights. Each granted right registers its
        // revocation mask, rook square, and path (doc 04 §3.7); the rights
        // themselves land in the root state.
        let rights: CastlingRights = fields[2]
            .parse()
            .map_err(|_| FenError::Castling(fields[2].to_string()))?;
        for (right, color, rook_from) in [
            (CastlingRights::WHITE_KINGSIDE, Color::White, Square::H1),
            (CastlingRights::WHITE_QUEENSIDE, Color::White, Square::A1),
            (CastlingRights::BLACK_KINGSIDE, Color::Black, Square::H8),
            (CastlingRights::BLACK_QUEENSIDE, Color::Black, Square::A8),
        ] {
            if rights.contains(right) {
                board.set_castling_right(color, rook_from);
            }
        }

        // Field 4: en passant target square. A square off ranks 3/6 is a
        // parse error; a well-formed square is then kept only if an ep
        // capture is actually playable (doc 04 §3.7 a–d) — the ep FILE is
        // hashed, so a spurious square would make identical positions hash
        // differently and break repetition detection and the TT.
        let ep_square = match fields[3] {
            "-" => None,
            s => {
                let sq: Square = s.parse().map_err(|_| FenError::EnPassant(s.to_string()))?;
                if sq.rank() != 2 && sq.rank() != 5 {
                    return Err(FenError::EnPassant(s.to_string()));
                }
                Some(sq).filter(|&sq| board.ep_capture_legal(sq))
            }
        };

        // Fields 5 and 6: halfmove clock and fullmove number, optional.
        let rule50 = match fields.get(4) {
            Some(s) => s
                .parse::<u32>()
                .ok()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| FenError::Clock(s.to_string()))?,
            None => 0,
        };
        let fullmove = match fields.get(5) {
            Some(s) => s
                .parse::<u32>()
                .map_err(|_| FenError::Clock(s.to_string()))?,
            None => 1,
        };
        // game_ply mapping with the fullmove-0 clamp (doc 04 §3.3): without
        // the clamp the common incorrect "fullmove 0" FENs would wrap
        // fullmove_number() to garbage in re-emitted FENs.
        let game_ply = (2 * (i64::from(fullmove) - 1)).clamp(0, i64::from(i32::MAX - 2)) as i32
            + (board.side_to_move == Color::Black) as i32;

        board.finish_setup(ep_square, rule50, game_ply);
        Ok(board)
    }

    /// Doc 04 §3.7 conditions a–d: is an en-passant capture of a just
    /// double-pushed pawn actually playable? Placement and side to move
    /// must already be set. Nothing less than full legality suffices: the
    /// ep file is hashed (D9), so keeping a square whose capture is illegal
    /// (a pinned capturer, say) would make identical positions hash
    /// differently.
    fn ep_capture_legal(&self, ep: Square) -> bool {
        let us = self.side_to_move();
        let them = us.flip();
        // The ep target sits on OUR relative rank 6 (rank 6 with White to
        // move, rank 3 with Black to move); anything else is spurious.
        if ep.relative_rank(us) != 5 {
            return false;
        }
        // a. one of our pawns attacks the ep square (attack symmetry: the
        //    squares a `them` pawn ON `ep` would attack are exactly the
        //    squares from which an `us` pawn attacks `ep`).
        let candidates = attacks::pawn_attacks(them, ep) & self.pieces(us, PieceType::Pawn);
        if candidates.is_empty() {
            return false;
        }
        // b. the double-pushed enemy pawn stands in front of the ep square.
        let target = ep.offset(them.pawn_push());
        if !self.pieces(them, PieceType::Pawn).contains(target) {
            return false;
        }
        // c. the ep square and the pawn's origin square are empty.
        let origin = ep.offset(us.pawn_push());
        if !(self.all_occupancy() & (Bitboard::from_square(ep) | Bitboard::from_square(origin)))
            .is_empty()
        {
            return false;
        }
        // d. at least one candidate can play the capture without leaving
        //    its own king in check: simulate the post-capture occupancy
        //    (captured pawn and capturer lifted off, capturer placed on the
        //    ep square) and require that no enemy piece other than the
        //    removed target attacks our king. This covers pins of the
        //    capturer — including the horizontal both-pawns-off-the-rank
        //    kind — and checks the capture fails to address.
        let Some(ksq) = self.pieces(us, PieceType::King).lsb() else {
            return true; // king-less analysis boards: nothing can be in check
        };
        let cleared =
            self.all_occupancy() ^ Bitboard::from_square(target) ^ Bitboard::from_square(ep);
        candidates.into_iter().any(|from| {
            let occupied = cleared ^ Bitboard::from_square(from);
            (self.attackers_to(ksq, occupied)
                & self.occupancy(them)
                & !Bitboard::from_square(target))
            .is_empty()
        })
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
    use crate::types::{BISHOP_VALUE, KNIGHT_VALUE, QUEEN_VALUE, ROOK_VALUE};
    use crate::zobrist::{ZOBRIST, fold_rule50};

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

    // Doc 04 §4 step 3: from-scratch key computation pins. The literals
    // change ONLY if the zobrist seed/fill order changes — which
    // invalidates every stored key and must be deliberate (the pins in
    // zobrist::tests catch the same event at the table level).
    #[test]
    fn startpos_keys_are_stable_and_recomputable() {
        let board = Board::startpos();
        board.assert_state_keys_consistent();

        // rule50 = 0: the exposed key is the raw key, unfolded.
        assert_eq!(board.key(), board.raw_key());
        assert_eq!(board.raw_key(), 0x8F8F01D4562F59FB);
        assert_eq!(board.pawn_key(), 0x916BB95C438C0BA9);
        assert_eq!(board.material_key(), 0xA0710EF633754DD6);
        assert_eq!(board.minor_key(), 0x41AF5A253E95F502);
        assert_eq!(board.non_pawn_key(Color::White), 0x85DC4DDCD53BCFCC);
        assert_eq!(board.non_pawn_key(Color::Black), 0x54B51DF93C8CF372);

        // Stability across independent parses.
        assert_eq!(
            Board::from_fen(START_FEN).unwrap().raw_key(),
            board.raw_key()
        );

        // Independent in-test recompute of the main key straight from the
        // public zobrist tables (different code path from init_state).
        let mut key = 0;
        for sq in Square::all() {
            if let Some(pc) = board.piece_on(sq) {
                key ^= ZOBRIST.psq[pc.index()][sq.idx()];
            }
        }
        key ^= ZOBRIST.castling[board.castling_rights().bits() as usize];
        assert_eq!(board.raw_key(), key);

        // Non-pawn material: 2N + 2B + 2R + Q per side, symmetric.
        let npm = 2 * (KNIGHT_VALUE + BISHOP_VALUE + ROOK_VALUE) + QUEEN_VALUE;
        assert_eq!(board.non_pawn_material(Color::White), npm);
        assert_eq!(board.non_pawn_material(Color::Black), npm);
    }

    // Doc 04 §5 (FEN-loadable subset of transposition invariance): side to
    // move and castling rights each perturb the key by their exact table
    // entries.
    #[test]
    fn state_fields_change_the_key() {
        let base = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let black = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").unwrap();
        assert_eq!(base.raw_key() ^ black.raw_key(), ZOBRIST.side);

        let partial = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1").unwrap();
        assert_eq!(
            base.raw_key() ^ partial.raw_key(),
            ZOBRIST.castling[CastlingRights::ALL.bits() as usize]
                ^ ZOBRIST.castling[partial.castling_rights().bits() as usize]
        );
        // Sanity: "Kq" is WK | BQ = 0b1001.
        assert_eq!(partial.castling_rights().bits(), 0b1001);
    }

    // Doc 04 §5 rule50 folding: raw keys ignore the clock entirely; key()
    // folds in 8-step buckets starting at 14.
    #[test]
    fn key_folds_rule50_from_fen_clocks() {
        let at =
            |clock: u32| Board::from_fen(&format!("4k3/8/8/8/8/8/8/4K3 w - - {clock} 1")).unwrap();
        let raw = at(0).raw_key();
        for clock in [0, 13, 14, 21, 22, 99] {
            assert_eq!(
                at(clock).raw_key(),
                raw,
                "raw key must ignore clock {clock}"
            );
        }
        assert_eq!(at(13).key(), raw);
        assert_eq!(at(14).key(), fold_rule50(raw, 14));
        assert_ne!(at(14).key(), raw);
        assert_eq!(at(14).key(), at(21).key());
        assert_ne!(at(22).key(), at(21).key());
    }

    // Doc 04 §4 step 3: FEN round-trips preserve the ENTIRE state — the
    // reparsed board compares equal (placement, game_ply, all keys).
    #[test]
    fn fen_roundtrip_preserves_state() {
        for fen in [
            START_FEN,
            KIWIPETE,
            "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 4 7",
            "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            board.assert_state_keys_consistent();
            let reparsed = Board::from_fen(&board.to_fen()).unwrap();
            assert_eq!(
                reparsed, board,
                "state must survive a round-trip of {fen:?}"
            );
            assert_eq!(reparsed.key(), board.key());
            assert_eq!(reparsed.game_ply(), board.game_ply());
        }
    }

    // Doc 04 §5: the fullmove-0 clamp — FENs with the (incorrect but
    // common) fullmove counter 0 map to game_ply 0/1, report fullmove 1
    // with no u32 wrap, and re-emit a legal counter.
    #[test]
    fn fullmove_zero_fens_clamp_to_game_ply_zero() {
        let white = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 0").unwrap();
        assert_eq!(white.game_ply(), 0);
        assert_eq!(white.fullmove_number(), 1);
        assert!(white.to_fen().ends_with(" 0 1"), "{}", white.to_fen());

        let black = Board::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 0").unwrap();
        assert_eq!(black.game_ply(), 1);
        assert_eq!(black.fullmove_number(), 1);
        assert!(black.to_fen().ends_with(" 0 1"), "{}", black.to_fen());
    }

    // game_ply mapping (doc 04 §3.3): 2 * (fullmove - 1) + black-to-move.
    #[test]
    fn game_ply_tracks_fullmove_and_side() {
        let ply = |fen: &str| Board::from_fen(fen).unwrap().game_ply();
        assert_eq!(Board::startpos().game_ply(), 0);
        assert_eq!(ply("4k3/8/8/8/8/8/8/4K3 b - - 0 1"), 1);
        assert_eq!(ply("4k3/8/8/8/8/8/8/4K3 w - - 0 3"), 4);
        assert_eq!(ply("4k3/8/8/8/8/8/8/4K3 b - - 0 3"), 5);
    }

    // Doc 04 §3.7: the parsed ep square survives only when a capture is
    // actually playable — (a) one of our pawns attacks it, (b) the double-
    // pushed enemy pawn stands in front of it, (c) the target and origin
    // squares are empty, (d) at least one capturer can play it without
    // leaving its own king in check.
    #[test]
    fn ep_square_kept_when_capture_playable() {
        // After 1. e4 e6 2. e5 d5: exd6 e.p. is available (White to move).
        let fen = "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.en_passant_square(), Some(Square::D6));
        assert_eq!(board.to_fen(), fen);

        // Black-to-move mirror: after 1. e4 e5 2. f4 exf4 3. g4, fxg3 e.p.
        let fen = "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.en_passant_square(), Some(Square::G3));
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn spurious_ep_squares_are_dropped() {
        // (square, FEN): each ep target fails one §3.7 condition.
        let cases = [
            // (a) no capturing pawn: after 1. e4 nothing attacks e3 —
            // Stockfish drops this ep square too.
            (
                "e3",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ),
            // (b) no double-pushed pawn in front of d6 (d5 is empty).
            (
                "d6",
                "rnbqkbnr/ppp2ppp/4p3/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            ),
            // (c) the origin square d7 is occupied (a bishop retreated
            // there), so d7-d5 cannot have just been played.
            (
                "d6",
                "rn1qkbnr/pppb1ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 4",
            ),
            // Wrong side-relative rank: rank 3 belongs to a WHITE double
            // push, so with White to move e3 is spurious.
            (
                "e3",
                "rnbqkbnr/pp1ppppp/8/8/2p1P3/8/PPPPQPPP/RNBQKBN1 w Qkq e3 0 5",
            ),
        ];
        for (ep, fen) in cases {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(board.en_passant_square(), None, "{fen:?} must drop ep");
            assert_eq!(
                board.to_fen(),
                fen.replace(&format!(" {ep} "), " - "),
                "{fen:?} must re-emit '-'"
            );
        }
    }

    // Doc 04 §3.7 condition d (needs attackers_to): the ep square is
    // dropped when every capturer would leave its own king in check.
    #[test]
    fn ep_squares_with_illegal_captures_are_dropped() {
        // (ep, FEN): each capture is fully pseudo-possible but illegal.
        let cases = [
            // Vertical pin: exd6 opens the e-file, Re8 hits Ke1.
            ("d6", "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
            // Horizontal both-pawns-off-the-rank pin (the doc 04 §5
            // Kh5/Pf5/Ra5 case): fxe6 clears BOTH e5 and f5 off rank 5.
            ("e6", "7k/8/8/r3pP1K/8/8/8/8 w - e6 0 2"),
            // The same family, queen flavor (perft classic, black to
            // move): exd3 clears the rank between Qh4 and Ka4.
            ("d3", "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1"),
            // Unrelated check: axb6 does nothing about Re8's check.
            ("b6", "4r2k/8/8/Pp6/8/8/8/4K3 w - b6 0 2"),
        ];
        for (ep, fen) in cases {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(board.en_passant_square(), None, "{fen:?} must drop ep");
            assert_eq!(
                board.to_fen(),
                fen.replace(&format!(" {ep} "), " - "),
                "{fen:?} must re-emit '-'"
            );
        }
    }

    // Condition d keeps the square as soon as ONE capturer is legal: a
    // capture along the pin line stays on it, and a second unpinned pawn
    // rescues a pinned one.
    #[test]
    fn ep_squares_kept_when_some_capture_is_legal() {
        let cases = [
            // dxe6 travels ALONG the c4-f7 pin diagonal and re-blocks it.
            ("e6", "7k/5b2/8/3Pp3/2K5/8/8/8 w - e6 0 2"),
            // dxe6 is illegal (Rd8 pins d5 vertically) but fxe6 is fine.
            ("e6", "3r3k/8/8/3PpP2/8/8/8/3K4 w - e6 0 2"),
            // Black-side mirror of the along-the-pin-line case.
            ("d3", "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1"),
        ];
        for (ep, fen) in cases {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(
                board.en_passant_square().map(|sq| sq.to_string()),
                Some(ep.to_string()),
                "{fen:?} must keep ep"
            );
            assert_eq!(board.to_fen(), fen, "{fen:?} must round-trip");
        }
    }

    // The ep FILE is hashed only when the square is kept (D9): a dropped
    // ep square hashes identically to the same FEN written with "-", and a
    // kept one differs by exactly its file key.
    #[test]
    fn ep_keying_follows_validation() {
        let kept = Board::from_fen("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
            .unwrap();
        let none = Board::from_fen("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
            .unwrap();
        assert_eq!(
            kept.raw_key() ^ none.raw_key(),
            ZOBRIST.en_passant[Square::D6.file() as usize]
        );

        let dropped =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let dash =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_eq!(dropped, dash);
        assert_eq!(dropped.raw_key(), dash.raw_key());
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
