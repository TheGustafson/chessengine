//! Property-based tests (proptest) over the existing core types: `Bitboard`
//! set laws, `Square` round-trips, `Move` encode/decode, `CastlingRights`
//! text round-trips, and FEN round-trips on randomly generated placements.
//!
//! Case counts are pinned per block (instead of relying on proptest
//! defaults) so CI runtime stays deterministic and in the seconds range.
//! On failure proptest persists the seed under `proptest-regressions/`;
//! commit those files so the regression replays everywhere.

use chessengine::{
    Bitboard, Board, CastlingRights, Color, Move, MoveKind, Piece, PieceType, Square,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn square() -> impl Strategy<Value = Square> {
    (0u8..64).prop_map(Square::new)
}

fn bitboard() -> impl Strategy<Value = Bitboard> {
    any::<u64>().prop_map(Bitboard::from_u64)
}

fn color() -> impl Strategy<Value = Color> {
    prop_oneof![Just(Color::White), Just(Color::Black)]
}

/// Any piece type a pawn may promote to.
fn promotion_piece() -> impl Strategy<Value = PieceType> {
    prop_oneof![
        Just(PieceType::Knight),
        Just(PieceType::Bishop),
        Just(PieceType::Rook),
        Just(PieceType::Queen),
    ]
}

/// All 16 possible rights masks, built through the public `add` API.
fn castling_rights() -> impl Strategy<Value = CastlingRights> {
    (0u8..16).prop_map(|mask| {
        let singles = [
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
        ];
        let mut rights = CastlingRights::NONE;
        for (bit, &right) in singles.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                rights.add(right);
            }
        }
        rights
    })
}

/// A non-king piece (kings are placed separately so each side gets exactly one).
fn non_king_piece() -> impl Strategy<Value = Piece> {
    let types = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ];
    (color(), 0..types.len()).prop_map(move |(c, i)| Piece::new(c, types[i]))
}

/// A board built through a random `put_piece` sequence, respecting exactly
/// the invariants FEN round-tripping relies on: one king per side and no
/// pawns on ranks 1/8. Material is otherwise unconstrained — the FEN layer
/// must not care.
fn placement_board() -> impl Strategy<Value = Board> {
    let kings =
        (0u8..64, 0u8..64).prop_filter("kings must not share a square", |(wk, bk)| wk != bk);
    let fillers = proptest::collection::vec(proptest::option::weighted(0.2, non_king_piece()), 64);
    (kings, fillers).prop_map(|((wk, bk), fillers)| {
        let (wk, bk) = (Square::new(wk), Square::new(bk));
        let mut board = Board::empty();
        board.put_piece(Piece::new(Color::White, PieceType::King), wk);
        board.put_piece(Piece::new(Color::Black, PieceType::King), bk);
        for (index, slot) in fillers.into_iter().enumerate() {
            let Some(piece) = slot else { continue };
            let sq = Square::new(index as u8);
            if sq == wk || sq == bk {
                continue; // kings already placed there
            }
            if piece.piece_type == PieceType::Pawn && (sq.rank() == 0 || sq.rank() == 7) {
                continue; // no pawns on the back ranks
            }
            board.put_piece(piece, sq);
        }
        board
    })
}

// ---------------------------------------------------------------------------
// Bitboard set-operation laws, Square and CastlingRights round-trips,
// Move encode/decode
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn bitboard_de_morgan(a in bitboard(), b in bitboard()) {
        prop_assert_eq!(!(a | b), !a & !b);
        prop_assert_eq!(!(a & b), !a | !b);
    }

    #[test]
    fn bitboard_xor_is_symmetric_difference(a in bitboard(), b in bitboard()) {
        prop_assert_eq!(a ^ b, (a | b) & !(a & b));
    }

    #[test]
    fn bitboard_iteration_count_is_popcount(a in bitboard()) {
        prop_assert_eq!(a.into_iter().count(), a.count() as usize);
    }

    #[test]
    fn bitboard_iter_collect_roundtrip(a in bitboard()) {
        let squares: Vec<Square> = a.into_iter().collect();
        // Ascending, member squares only ...
        for pair in squares.windows(2) {
            prop_assert!(pair[0] < pair[1], "iteration must ascend");
        }
        for &sq in &squares {
            prop_assert!(a.contains(sq));
        }
        // ... and FromIterator rebuilds the identical set.
        let rebuilt: Bitboard = squares.into_iter().collect();
        prop_assert_eq!(rebuilt, a);
    }

    #[test]
    fn square_coords_roundtrip(file in 0u8..8, rank in 0u8..8) {
        let sq = Square::from_coords(file, rank);
        prop_assert_eq!(sq.file(), file);
        prop_assert_eq!(sq.rank(), rank);
        prop_assert_eq!(Square::new(sq.index()), sq);
    }

    #[test]
    fn square_from_own_coords_is_identity(sq in square()) {
        prop_assert_eq!(Square::from_coords(sq.file(), sq.rank()), sq);
    }

    #[test]
    fn square_display_parse_roundtrip(sq in square()) {
        prop_assert_eq!(sq.to_string().parse::<Square>(), Ok(sq));
    }

    #[test]
    fn castling_rights_display_parse_roundtrip(rights in castling_rights()) {
        prop_assert_eq!(rights.to_string().parse::<CastlingRights>(), Ok(rights));
    }

    #[test]
    fn move_normal_roundtrip(from in square(), to in square()) {
        let m = Move::normal(from, to);
        prop_assert_eq!(m.from(), from);
        prop_assert_eq!(m.to(), to);
        prop_assert_eq!(m.kind(), MoveKind::Normal);
        prop_assert_eq!(m.promotion_piece(), None);
        prop_assert_eq!(m.to_string(), format!("{from}{to}"));
    }

    #[test]
    fn move_promotion_roundtrip(from in square(), to in square(), piece in promotion_piece()) {
        let m = Move::promotion(from, to, piece);
        prop_assert_eq!(m.from(), from);
        prop_assert_eq!(m.to(), to);
        prop_assert_eq!(m.kind(), MoveKind::Promotion);
        prop_assert_eq!(m.promotion_piece(), Some(piece));
        prop_assert_eq!(m.to_string(), format!("{from}{to}{}", piece.to_char()));
    }

    #[test]
    fn move_en_passant_roundtrip(from in square(), to in square()) {
        let m = Move::en_passant(from, to);
        prop_assert_eq!(m.from(), from);
        prop_assert_eq!(m.to(), to);
        prop_assert_eq!(m.kind(), MoveKind::EnPassant);
        prop_assert_eq!(m.promotion_piece(), None);
    }

    #[test]
    fn move_castling_roundtrip(from in square(), to in square()) {
        let m = Move::castling(from, to);
        prop_assert_eq!(m.from(), from);
        prop_assert_eq!(m.to(), to);
        prop_assert_eq!(m.kind(), MoveKind::Castling);
        prop_assert_eq!(m.promotion_piece(), None);
    }

    /// The four kinds (and all promotion pieces) with identical from/to
    /// squares must encode to distinct 16-bit patterns.
    #[test]
    fn move_kinds_and_promotions_are_distinct(from in square(), to in square()) {
        let all = [
            Move::normal(from, to),
            Move::en_passant(from, to),
            Move::castling(from, to),
            Move::promotion(from, to, PieceType::Knight),
            Move::promotion(from, to, PieceType::Bishop),
            Move::promotion(from, to, PieceType::Rook),
            Move::promotion(from, to, PieceType::Queen),
        ];
        for (i, &a) in all.iter().enumerate() {
            for &b in &all[i + 1..] {
                prop_assert_ne!(a, b);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FEN round-trips on generated positions
// ---------------------------------------------------------------------------

proptest! {
    // Board generation is heavier than the pure-value properties above;
    // 256 cases still finishes in well under a second.
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// FEN round-trips on boards built through random `put_piece`
    /// sequences: the placement layer survives square for square and the
    /// FEN text is a fixed point. (Full `Board` equality against the
    /// put_piece-built original would also compare Zobrist state, which
    /// only the canonical constructors compute — key deltas live in
    /// `make_move`, not in `put_piece`; design/04 §3.3.)
    #[test]
    fn fen_roundtrip_on_generated_placements(board in placement_board()) {
        let fen = board.to_fen();
        let reparsed = Board::from_fen(&fen);
        prop_assert!(reparsed.is_ok(), "generated FEN {:?} rejected: {:?}", fen, reparsed.err());
        let reparsed = reparsed.unwrap();
        for sq in Square::all() {
            prop_assert_eq!(reparsed.piece_on(sq), board.piece_on(sq), "mismatch on {}", sq);
        }
        // from_fen is deterministic: a second parse yields an equal Board,
        // Zobrist state and all.
        prop_assert_eq!(&Board::from_fen(&fen).unwrap(), &reparsed);
        prop_assert_eq!(reparsed.to_fen(), fen);
    }

    /// Full-FEN round-trip with randomized state fields spliced onto a
    /// generated placement: side to move, castling rights, a plausible
    /// en-passant target (rank follows the side to move), and clocks.
    ///
    /// The ep square is subject to FEN validation (design/04 §3.7): it is
    /// either kept exactly or dropped to `-` — never invented, never
    /// replaced — and `to_fen` re-emits whatever survived.
    #[test]
    fn fen_roundtrip_with_random_state_fields(
        board in placement_board(),
        black_to_move in any::<bool>(),
        rights in castling_rights(),
        ep_file in proptest::option::of(0u8..8),
        halfmove in 0u32..=150,
        fullmove in 1u32..=400,
    ) {
        let full = board.to_fen();
        let placement = full.split_whitespace().next().expect("to_fen yields fields");
        let side = if black_to_move { 'b' } else { 'w' };
        // A real ep target sits on rank 3 after a white double push (black
        // to move) and on rank 6 after a black one (white to move).
        let ep_square = ep_file
            .map(|file| Square::from_coords(file, if black_to_move { 2 } else { 5 }));
        let ep = ep_square.map_or("-".to_string(), |sq| sq.to_string());
        let fen = format!("{placement} {side} {rights} {ep} {halfmove} {fullmove}");

        let parsed = Board::from_fen(&fen);
        prop_assert!(parsed.is_ok(), "constructed FEN {:?} rejected: {:?}", fen, parsed.err());
        let parsed = parsed.unwrap();

        let expected_side = if black_to_move { Color::Black } else { Color::White };
        prop_assert_eq!(parsed.side_to_move(), expected_side);
        prop_assert_eq!(parsed.castling_rights(), rights);
        prop_assert_eq!(parsed.halfmove_clock(), halfmove);
        prop_assert_eq!(parsed.fullmove_number(), fullmove);
        if parsed.en_passant_square().is_some() {
            prop_assert_eq!(parsed.en_passant_square(), ep_square, "ep must never be replaced");
        }
        let ep_out = parsed.en_passant_square().map_or("-".to_string(), |sq| sq.to_string());
        let expected = format!("{placement} {side} {rights} {ep_out} {halfmove} {fullmove}");
        prop_assert_eq!(parsed.to_fen(), expected);
    }
}
