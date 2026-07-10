//! Corpus-driven cross-checks for the doc 04 §4 step 4 attack queries.
//!
//! `Board::attackers_to` composes six piece-set intersections in one
//! expression; these tests recompute the same answer the slow, independent
//! way — ask every piece on the board, one per-piece attack query at a
//! time, whether IT attacks the square — over every position in
//! `tests/data/fens.txt`, under both the real occupancy and randomly
//! reduced ones (the SEE x-ray pattern). `checkers` (derived by
//! `set_check_info`'s caller) is cross-checked through the same
//! independent path.

mod common;

use chessengine::{Bitboard, Board, Color, PieceType, Square, attacks};

/// Deterministic SplitMix64 for random occupancy subsets — the same
/// dev-dependency-free pattern as the unit tests (D15).
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

/// Independent `attackers_to`: one per-piece attack query per board piece.
/// Like the real thing, the attacker set comes from the piece placement
/// while `occupied` shapes only the slider rays.
fn attackers_by_piece_queries(board: &Board, sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attackers = Bitboard::EMPTY;
    for from in board.all_occupancy() {
        let piece = board.piece_on(from).expect("occupied square has a piece");
        let attacked = match piece.piece_type {
            PieceType::Pawn => attacks::pawn_attacks(piece.color, from),
            pt => attacks::attacks(pt, from, occupied),
        };
        if attacked.contains(sq) {
            attackers |= Bitboard::from_square(from);
        }
    }
    attackers
}

#[test]
fn corpus_attackers_to_matches_per_piece_queries() {
    for fen in common::load_fens() {
        let board = common::board(&fen);
        let all = board.all_occupancy();
        for sq in Square::all() {
            assert_eq!(
                board.attackers_to(sq, all),
                attackers_by_piece_queries(&board, sq, all),
                "{fen:?}: attackers_to({sq}) disagrees with per-piece queries"
            );
        }
    }
}

#[test]
fn corpus_attackers_to_matches_under_reduced_occupancies() {
    let mut rng = SplitMix64(0x0DDC_0FFE_EBAD_F00D);
    for fen in common::load_fens() {
        let board = common::board(&fen);
        for _ in 0..4 {
            // A random subset of the occupancy: lifted pieces stop blocking
            // rays but keep attacking — exactly how SEE walks batteries.
            let occupied = board.all_occupancy() & Bitboard::from_u64(rng.next_u64());
            for sq in Square::all() {
                assert_eq!(
                    board.attackers_to(sq, occupied),
                    attackers_by_piece_queries(&board, sq, occupied),
                    "{fen:?}: attackers_to({sq}) disagrees under occupancy {occupied:?}"
                );
            }
        }
    }
}

#[test]
fn corpus_checkers_match_brute_force() {
    for fen in common::load_fens() {
        let board = common::board(&fen);
        let stm = board.side_to_move();
        let expected =
            attackers_by_piece_queries(&board, board.king_square(stm), board.all_occupancy())
                & board.occupancy(stm.flip());
        assert_eq!(board.checkers(), expected, "{fen:?}: checkers disagree");
    }
}

/// Structural sanity for the pin bookkeeping over the corpus: blockers are
/// real pieces aligned with their king, pinners are enemy sliders aligned
/// with the enemy king. (Exact-set semantics are pinned by the hand-built
/// fixtures in `src/board.rs` — notably the stacked-sniper convention,
/// which a naive remove-one-piece recompute would NOT reproduce.)
#[test]
fn corpus_blockers_and_pinners_are_aligned_with_the_king() {
    for fen in common::load_fens() {
        let board = common::board(&fen);
        for color in Color::ALL {
            let ksq = board.king_square(color);
            let blockers = board.blockers_for_king(color);
            assert_eq!(
                blockers,
                blockers & board.all_occupancy(),
                "{fen:?}: phantom blocker for {color:?}"
            );
            for b in blockers {
                assert!(
                    !attacks::line(ksq, b).is_empty(),
                    "{fen:?}: blocker {b} not aligned with the {color:?} king"
                );
            }

            // pinners(color) pin against the OTHER king.
            let enemy_ksq = board.king_square(color.flip());
            let sliders = board.pieces(color, PieceType::Bishop)
                | board.pieces(color, PieceType::Rook)
                | board.pieces(color, PieceType::Queen);
            for p in board.pinners(color) {
                assert!(
                    sliders.contains(p),
                    "{fen:?}: pinner {p} is not a {color:?} slider"
                );
                assert!(
                    !attacks::line(enemy_ksq, p).is_empty(),
                    "{fen:?}: pinner {p} not aligned with the enemy king"
                );
                // Its sole blocker is recorded on the enemy side's list.
                assert!(
                    !(attacks::between(enemy_ksq, p) & board.blockers_for_king(color.flip()))
                        .is_empty(),
                    "{fen:?}: pinner {p} has no recorded blocker"
                );
            }
        }
    }
}
