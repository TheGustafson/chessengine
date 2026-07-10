//! FEN corpus tests: every position in `tests/data/fens.txt` must parse,
//! round-trip byte-identically, and describe an internally consistent board.
//! These run against existing code (fen/board) — no RED markers needed.

mod common;

use chessengine::{Bitboard, Board, Color, PieceType, Square};

#[test]
fn corpus_has_at_least_60_positions() {
    let fens = common::load_fens();
    assert!(
        fens.len() >= 60,
        "expected a corpus of at least 60 FENs, found {}",
        fens.len()
    );
}

#[test]
fn corpus_round_trips_through_fen() {
    for fen in common::load_fens() {
        let board = common::board(&fen);
        assert_eq!(board.to_fen(), fen, "round-trip mismatch for {fen:?}");
    }
}

/// Cross-check the board layers through public accessors only: the mailbox
/// (`piece_on`) and the bitboards (`pieces`/`occupancy`/`all_occupancy`)
/// must describe the same position.
#[test]
fn corpus_boards_are_internally_consistent() {
    for fen in common::load_fens() {
        let board = common::board(&fen);

        // Per-square agreement between mailbox and bitboards.
        for sq in Square::all() {
            match board.piece_on(sq) {
                Some(piece) => {
                    assert!(
                        board.pieces(piece.color, piece.piece_type).contains(sq),
                        "{fen:?}: mailbox has {piece} on {sq}, bitboard disagrees"
                    );
                    assert!(board.occupancy(piece.color).contains(sq));
                    assert!(board.all_occupancy().contains(sq));
                }
                None => assert!(
                    !board.all_occupancy().contains(sq),
                    "{fen:?}: occupancy claims a piece on empty {sq}"
                ),
            }
        }

        // Occupancy counts match a full mailbox scan.
        for color in Color::ALL {
            let mailbox_count = Square::all()
                .filter(|&sq| board.piece_on(sq).is_some_and(|p| p.color == color))
                .count();
            assert_eq!(
                mailbox_count as u32,
                board.occupancy(color).count(),
                "{fen:?}: {color:?} occupancy count disagrees with mailbox scan"
            );

            // Piece bitboards partition the color occupancy: their union is
            // the occupancy and their counts sum to it (so they are disjoint).
            let mut union = Bitboard::EMPTY;
            let mut count_sum = 0;
            for piece_type in PieceType::ALL {
                union |= board.pieces(color, piece_type);
                count_sum += board.pieces(color, piece_type).count();
            }
            assert_eq!(union, board.occupancy(color));
            assert_eq!(count_sum, board.occupancy(color).count());
        }

        assert_eq!(
            board.occupancy(Color::White) | board.occupancy(Color::Black),
            board.all_occupancy(),
            "{fen:?}: color occupancies do not add up to all_occupancy"
        );
        assert!(
            (board.occupancy(Color::White) & board.occupancy(Color::Black)).is_empty(),
            "{fen:?}: a square is occupied by both colors"
        );
    }
}

#[test]
fn corpus_positions_have_one_king_each_and_no_back_rank_pawns() {
    let back_ranks = Bitboard::RANK_1 | Bitboard::RANK_8;
    for fen in common::load_fens() {
        let board = common::board(&fen);
        for color in Color::ALL {
            assert_eq!(
                board.pieces(color, PieceType::King).count(),
                1,
                "{fen:?}: {color:?} must have exactly one king"
            );
            assert!(
                (board.pieces(color, PieceType::Pawn) & back_ranks).is_empty(),
                "{fen:?}: {color:?} has a pawn on rank 1 or 8"
            );
        }
    }
}

#[test]
fn malformed_fens_are_rejected() {
    let malformed = [
        // structurally empty / wrong field count
        "",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra",
        // placement errors
        "rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // 7 ranks
        "rnbqkbnr/pppppppp/8/9/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // digit 9
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP/RNBQKBNR w KQkq - 0 1", // 7-wide rank
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPPP/RNBQKBNR w KQkq - 0 1", // 9-wide rank
        "rnbqkbnr/pppppppp/8/8/4X3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // bad piece
        // state-field errors
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", // bad side
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkqK - 0 1", // dup right
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e5 0 1", // ep rank
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq i3 0 1", // ep square
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1", // bad clock
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 -1", // bad clock
    ];
    for fen in malformed {
        assert!(
            Board::from_fen(fen).is_err(),
            "expected rejection of malformed FEN {fen:?}"
        );
    }
}
