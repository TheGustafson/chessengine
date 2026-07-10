//! P3 step 3 (design/05-movegen.md §4): equivalence of the bulk-shift pawn
//! generator against a slow per-pawn reference, over the checked-in FEN
//! corpus and randomized placements.
//!
//! The reference recomputes every pawn move with per-square
//! `Square::shifted` arithmetic off the mailbox — none of the bulk-shift,
//! rank-3-relay, or from-square-recovery tricks — and classifies each move
//! per the doc 05 §3.2 table (queen promotions with CAPTURES, push
//! underpromotions with QUIETS). Any disagreement points at the fast path.
//!
//! Scope: out-of-check positions only (CAPTURES/QUIETS/NON_EVASIONS).
//! EVASIONS pawn semantics are P3 step 4's tests.

mod common;

use std::collections::HashSet;

use chessengine::movegen::{gen_type, generate};
use chessengine::{Board, Color, Direction, Move, MoveList, Piece, PieceType, Square};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// The slow reference generator
// ---------------------------------------------------------------------------

/// Reference pawn moves, pre-split per the doc 05 §3.2 table.
struct PawnSets {
    captures: HashSet<Move>,
    quiets: HashSet<Move>,
}

fn reference_pawn_moves(board: &Board) -> PawnSets {
    let us = board.side_to_move();
    let them = us.flip();
    let up = us.pawn_push();
    let diagonals = match us {
        Color::White => [Direction::NorthWest, Direction::NorthEast],
        Color::Black => [Direction::SouthWest, Direction::SouthEast],
    };
    let mut sets = PawnSets {
        captures: HashSet::new(),
        quiets: HashSet::new(),
    };
    let empty = |sq: Square| board.piece_on(sq).is_none();

    for from in Square::all() {
        if board.piece_on(from) != Some(Piece::new(us, PieceType::Pawn)) {
            continue;
        }
        let promoting = from.relative_rank(us) == 6;

        // Pushes. A pawn always has an on-board `up` neighbor (no pawns on
        // the last rank), but go through the checked `shifted` anyway —
        // this is the slow path, it gets to be paranoid.
        if let Some(to) = from.shifted(up)
            && empty(to)
        {
            if promoting {
                sets.captures
                    .insert(Move::promotion(from, to, PieceType::Queen));
                for pt in [PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
                    sets.quiets.insert(Move::promotion(from, to, pt));
                }
            } else {
                sets.quiets.insert(Move::normal(from, to));
            }
            if from.relative_rank(us) == 1
                && let Some(to2) = to.shifted(up)
                && empty(to2)
            {
                sets.quiets.insert(Move::normal(from, to2));
            }
        }

        // Diagonal captures, including capturing promotions and en passant.
        for d in diagonals {
            let Some(to) = from.shifted(d) else { continue };
            if board.piece_on(to).is_some_and(|p| p.color == them) {
                if promoting {
                    for pt in [
                        PieceType::Queen,
                        PieceType::Rook,
                        PieceType::Bishop,
                        PieceType::Knight,
                    ] {
                        sets.captures.insert(Move::promotion(from, to, pt));
                    }
                } else {
                    sets.captures.insert(Move::normal(from, to));
                }
            } else if Some(to) == board.en_passant_square() {
                sets.captures.insert(Move::en_passant(from, to));
            }
        }
    }
    sets
}

// ---------------------------------------------------------------------------
// The fast path under test
// ---------------------------------------------------------------------------

/// The pawn-made subset of `generate::<GT>` as a set (also asserting the
/// generator emitted no duplicate pawn moves).
fn generated_pawn_moves<const GT: u8>(board: &Board) -> HashSet<Move> {
    let mut list = MoveList::new();
    generate::<GT>(board, &mut list);
    let pawn_moves: Vec<Move> = list
        .iter()
        .copied()
        .filter(|m| board.piece_on(m.from()).map(|p| p.piece_type) == Some(PieceType::Pawn))
        .collect();
    let set: HashSet<Move> = pawn_moves.iter().copied().collect();
    assert_eq!(
        set.len(),
        pawn_moves.len(),
        "duplicate pawn moves generated"
    );
    set
}

/// Compare all three out-of-check GenTypes against the reference.
fn assert_matches_reference(board: &Board, context: &str) {
    let expected = reference_pawn_moves(board);
    assert_eq!(
        generated_pawn_moves::<{ gen_type::CAPTURES }>(board),
        expected.captures,
        "CAPTURES pawn set diverges from the reference in {context}"
    );
    assert_eq!(
        generated_pawn_moves::<{ gen_type::QUIETS }>(board),
        expected.quiets,
        "QUIETS pawn set diverges from the reference in {context}"
    );
    let union: HashSet<Move> = expected.captures.union(&expected.quiets).copied().collect();
    assert_eq!(
        generated_pawn_moves::<{ gen_type::NON_EVASIONS }>(board),
        union,
        "NON_EVASIONS pawn set diverges from the reference in {context}"
    );
}

// ---------------------------------------------------------------------------
// Corpus equivalence (deterministic, every usable position)
// ---------------------------------------------------------------------------

#[test]
fn corpus_pawn_moves_match_reference() {
    let mut checked = 0;
    for fen in common::load_fens() {
        let board = common::board(&fen);
        if !board.checkers().is_empty() {
            continue; // EVASIONS is step 4's territory
        }
        assert_matches_reference(&board, &fen);
        checked += 1;
    }
    assert!(
        checked >= 50,
        "FEN corpus unexpectedly thin: only {checked} out-of-check positions"
    );
}

// ---------------------------------------------------------------------------
// Randomized equivalence (proptest placements, both sides to move, with
// randomly proposed ep squares that the FEN layer keeps or drops per D9)
// ---------------------------------------------------------------------------

/// Filler pieces biased toward pawns: pawn interactions (blocks, captures,
/// promotions) are what this suite is about; the sprinkling of other piece
/// types provides rank-8/1 capture targets for promotion fans.
fn pawn_biased_piece() -> impl Strategy<Value = Piece> {
    let color = prop_oneof![Just(Color::White), Just(Color::Black)];
    let piece_type = prop_oneof![
        6 => Just(PieceType::Pawn),
        1 => Just(PieceType::Knight),
        1 => Just(PieceType::Bishop),
        1 => Just(PieceType::Rook),
        1 => Just(PieceType::Queen),
    ];
    (color, piece_type).prop_map(|(c, pt)| Piece::new(c, pt))
}

/// A random placement through `put_piece` (the testing convention): one
/// king per side, no pawns on the back ranks, ~19 filler pieces expected.
fn pawn_heavy_placement() -> impl Strategy<Value = Board> {
    let kings =
        (0u8..64, 0u8..64).prop_filter("kings must not share a square", |(wk, bk)| wk != bk);
    let fillers =
        proptest::collection::vec(proptest::option::weighted(0.3, pawn_biased_piece()), 64);
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

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        // In-check placements are rejected below; leave generous headroom.
        max_global_rejects: 4096,
        ..ProptestConfig::default()
    })]

    #[test]
    fn randomized_pawn_moves_match_reference(
        placement in pawn_heavy_placement(),
        black_to_move in any::<bool>(),
        ep_file in proptest::option::of(0u8..8),
    ) {
        // Splice a side to move and a plausibly-ranked ep target onto the
        // placement; from_fen keeps the ep square only if the capture is
        // actually playable (D9), which is exactly what movegen assumes.
        let full = placement.to_fen();
        let pieces = full.split_whitespace().next().expect("to_fen yields fields");
        let side = if black_to_move { 'b' } else { 'w' };
        let ep = ep_file
            .map(|file| {
                Square::from_coords(file, if black_to_move { 2 } else { 5 }).to_string()
            })
            .unwrap_or_else(|| "-".to_string());
        let fen = format!("{pieces} {side} - {ep} 0 1");
        let board = Board::from_fen(&fen).expect("constructed FEN must parse");
        prop_assume!(board.checkers().is_empty());
        assert_matches_reference(&board, &fen);
    }
}
