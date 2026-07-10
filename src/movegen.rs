//! Type-parameterized pseudo-legal move generation (design/05-movegen.md).
//!
//! Stockfish monomorphizes one generator per `GenType`; the stable-Rust
//! equivalent is a `u8` const-generic parameter with named constants in
//! [`gen_type`] (doc 05 §3.1, §9 Q3 — `adt_const_params` is not stable, so
//! no enum in const position yet). `generate` dispatches once on the side
//! to move into a color-monomorphized worker, so all 4×2 instantiations
//! compile to straight-line routines with the dead branches removed — the
//! effect of Stockfish's `if constexpr` (never its code, D27).
//!
//! Semantics per type (doc 05 §3.2 — the split the MovePicker's staged
//! generation depends on): `CAPTURES` includes non-capturing *queen*
//! promotions; `QUIETS` includes non-capturing *under*promotions; their
//! disjoint union is `NON_EVASIONS`. One target-mask `Bitboard`
//! parameterizes every non-king generator (doc 05 §3.3).
//!
//! Hot-path rules: no heap allocation ([`MoveList`] is stack-only), no
//! `unsafe` (D14 — crate-wide `deny`).
//!
//! P3 step 2 status (doc 05 §4): the GenType mechanism, target-mask
//! plumbing, and non-pawn generation are live; `pawn_moves` (step 3) and
//! `castling_moves` (step 5) land empty, and the legality layer
//! (`Board::legal`, `generate_legal`, step 6) follows.

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveList};
use crate::types::{Color, PieceType, Square};

/// The `GenType` constants: what `generate::<GT>` emits (doc 05 §3.1).
pub mod gen_type {
    /// Captures (incl. en passant), all capturing promotions (Q/R/B/N),
    /// and non-capturing QUEEN promotions (doc 05 §3.2).
    pub const CAPTURES: u8 = 0;
    /// Non-captures except castling, plus non-capturing UNDERpromotions
    /// (R/B/N). Never emits a capturing promotion.
    pub const QUIETS: u8 = 1;
    /// Check evasions; only valid while the side to move is in check.
    pub const EVASIONS: u8 = 2;
    /// Everything pseudo-legal: `CAPTURES` ⊎ `QUIETS` (disjoint union).
    pub const NON_EVASIONS: u8 = 3;
}

/// Append all pseudo-legal moves of the given type to `list`.
///
/// Does NOT clear `list` first — callers compose sub-ranges by remembering
/// `list.len()` before the call (doc 05 §3.3).
///
/// Debug-asserts that `GT == EVASIONS` if and only if the side to move is
/// in check (doc 05 §3.1).
pub fn generate<const GT: u8>(board: &Board, list: &mut MoveList) {
    debug_assert_eq!(
        GT == gen_type::EVASIONS,
        !board.checkers().is_empty(),
        "generate::<EVASIONS> iff the side to move is in check"
    );
    // Color as a second const parameter: pawn directions and rank masks
    // become compile-time constants, exactly SF's `template<Color Us>`.
    match board.side_to_move() {
        Color::White => generate_all::<GT, true>(board, list),
        Color::Black => generate_all::<GT, false>(board, list),
    }
}

/// Append all pseudo-legal captures (doc 05 §3.2 row `CAPTURES`).
#[inline]
pub fn generate_captures(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::CAPTURES }>(board, list);
}

/// Append all pseudo-legal quiet moves (doc 05 §3.2 row `QUIETS`).
#[inline]
pub fn generate_quiets(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::QUIETS }>(board, list);
}

/// Append all pseudo-legal check evasions (doc 05 §3.6).
#[inline]
pub fn generate_evasions(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::EVASIONS }>(board, list);
}

/// Append every pseudo-legal move (doc 05 §3.2 row `NON_EVASIONS`).
#[inline]
pub fn generate_non_evasions(board: &Board, list: &mut MoveList) {
    generate::<{ gen_type::NON_EVASIONS }>(board, list);
}

/// The color-monomorphized worker: computes the single target mask
/// (doc 05 §3.3) and runs the per-piece generators in the pinned emission
/// order (doc 05 §5.4): pawns, N, B, R, Q, K, castles.
fn generate_all<const GT: u8, const WHITE: bool>(board: &Board, list: &mut MoveList) {
    use gen_type::{CAPTURES, EVASIONS, NON_EVASIONS, QUIETS};
    let us = if WHITE { Color::White } else { Color::Black };
    let ksq = board.king_square(us);
    // One atomic-acquire deref for the whole generation pass; the slider
    // closures below then use the `&'static` tables directly (attacks.rs
    // hot-loop pattern).
    let tables = attacks::tables();

    // Double check: only king moves can be legal — skip all non-king work.
    if GT != EVASIONS || !board.checkers().more_than_one() {
        let target = match GT {
            // `between` includes the checker square by convention
            // (doc 03 §3.4), so this mask is "block OR capture" as-is.
            EVASIONS => attacks::between(ksq, board.checkers().lsb().unwrap()),
            NON_EVASIONS => !board.occupancy(us),
            CAPTURES => board.occupancy(us.flip()),
            _ => !board.all_occupancy(), // QUIETS
        };

        pawn_moves::<GT, WHITE>(board, target, list);

        let occ = board.all_occupancy();
        piece_moves(
            board.pieces(us, PieceType::Knight),
            target,
            attacks::knight_attacks,
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Bishop),
            target,
            |from| tables.bishop_attacks(from, occ),
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Rook),
            target,
            |from| tables.rook_attacks(from, occ),
            list,
        );
        piece_moves(
            board.pieces(us, PieceType::Queen),
            target,
            |from| tables.queen_attacks(from, occ),
            list,
        );
    }

    // King moves: under EVASIONS the king may go anywhere not our own —
    // fleeing is not restricted to the block/capture ray. Safety testing
    // (including the x-ray through the vacated square) is `Board::legal`'s
    // job (doc 05 §3.8), not the generator's.
    let king_target = match GT {
        EVASIONS | NON_EVASIONS => !board.occupancy(us),
        CAPTURES => board.occupancy(us.flip()),
        _ => !board.all_occupancy(), // QUIETS
    };
    for to in attacks::king_attacks(ksq) & king_target {
        list.push(Move::normal(ksq, to));
    }

    // A castle never captures and never evades (the mover is not in check
    // in both remaining contexts) — QUIETS/NON_EVASIONS only (doc 05 §3.7).
    if GT == QUIETS || GT == NON_EVASIONS {
        castling_moves::<WHITE>(board, list);
    }
}

/// Pawn pushes, captures, promotions, en passant (doc 05 §3.4).
///
/// TODO(P3 step 3, doc 05 §3.4): bulk-shift pushes with the rank-3 relay
/// trick, diagonal captures, the §3.2 promotion split via
/// `push_promotions`, and en passant (with the EVASIONS discovered-check
/// skip). Lands empty rather than `todo!` so the non-pawn generators are
/// exercisable on fixtures whose *mover* has no pawns (doc 05 §4 step 2).
fn pawn_moves<const GT: u8, const WHITE: bool>(
    _board: &Board,
    _target: Bitboard,
    _list: &mut MoveList,
) {
}

/// One shared inner loop, monomorphized per call site via `impl Fn` — each
/// call gets its own copy with the attack lookup inlined, the Rust idiom
/// for SF's `template<PieceType Pt>` (doc 05 §3.5). Slider closures capture
/// occupancy by value (`Bitboard` is `Copy`); leapers ignore it.
#[inline(always)]
fn piece_moves(
    pieces: Bitboard,
    target: Bitboard,
    attack: impl Fn(Square) -> Bitboard,
    list: &mut MoveList,
) {
    for from in pieces {
        for to in attack(from) & target {
            list.push(Move::normal(from, to));
        }
    }
}

/// Castling: rights present + path empty; attacked-square testing is
/// deferred to `Board::legal` (doc 05 §3.7).
///
/// TODO(P3 step 5, doc 05 §3.7): emit `Move::castling` per the §3.7 table
/// via `CastlingRights::{kingside, queenside}` and `Board::castling_path`.
/// Lands empty rather than `todo!` — it is reached by every QUIETS /
/// NON_EVASIONS call (doc 05 §4 step 2).
fn castling_moves<const WHITE: bool>(_board: &Board, _list: &mut MoveList) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::MoveKind;
    use std::collections::HashSet;

    /// Parse a FEN, panicking with context on failure.
    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap_or_else(|e| panic!("bad test FEN {fen:?}: {e}"))
    }

    fn generated<const GT: u8>(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        generate::<GT>(board, &mut list);
        list
    }

    /// Moves in `list` made by a `pt` of the side to move.
    fn count_for_piece(board: &Board, list: &MoveList, pt: PieceType) -> usize {
        list.iter()
            .filter(|m| {
                board
                    .piece_on(m.from())
                    .expect("generated move must start on a piece")
                    .piece_type
                    == pt
            })
            .count()
    }

    fn move_set(list: &MoveList) -> HashSet<Move> {
        let set: HashSet<Move> = list.iter().copied().collect();
        assert_eq!(set.len(), list.len(), "duplicate moves within one list");
        set
    }

    /// Hand fixtures with per-piece counts hand-verified against the board
    /// diagram (doc 05 §4 step 2, §5 for this step). No mover-side pawns —
    /// `pawn_moves` is a step-3 stub — so every count below stays valid
    /// once pawn generation lands. Fields: FEN, then expected
    /// (knight, bishop, rook, queen, king) NON_EVASIONS counts, then
    /// expected CAPTURES total.
    const FIXTURES: &[(&str, [usize; 5], usize)] = &[
        // Nc3 vs pawn d5: 8 knight moves (1 capture), 5 king moves.
        ("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1", [8, 0, 0, 0, 5], 1),
        // Be2 vs pawn g4: rays stop at the capture; e2 is own-blocked for
        // the king (4 king moves, not 5).
        ("4k3/8/8/8/6p1/8/4B3/4K3 w - - 0 1", [0, 8, 0, 0, 4], 1),
        // Rd4 between pawns b4/g4: both rank rays end in captures.
        ("k7/8/8/8/1p1R2p1/8/8/4K3 w - - 0 1", [0, 0, 12, 0, 5], 2),
        // Black to move (color-dispatch check): open-board queen on d5.
        ("4k3/8/8/3q4/8/8/8/4K3 b - - 0 1", [0, 0, 0, 27, 5], 0),
        // King capture: e2 is the only enemy piece adjacent to e1.
        ("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1", [0, 0, 0, 0, 5], 1),
        // Boxed rook: own knight (a2) and own king (b1) block both rays —
        // zero rook moves proves the target mask excludes own pieces and
        // occupancy stops sliders.
        ("4k3/8/8/8/8/8/N7/RK6 w - - 0 1", [3, 0, 0, 0, 3], 0),
    ];

    #[test]
    fn per_piece_counts_on_hand_fixtures() {
        for &(fen, per_piece, _) in FIXTURES {
            let b = board(fen);
            let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
            let expected: [(PieceType, usize); 5] = [
                (PieceType::Knight, per_piece[0]),
                (PieceType::Bishop, per_piece[1]),
                (PieceType::Rook, per_piece[2]),
                (PieceType::Queen, per_piece[3]),
                (PieceType::King, per_piece[4]),
            ];
            for (pt, count) in expected {
                assert_eq!(
                    count_for_piece(&b, &all, pt),
                    count,
                    "{pt:?} count in {fen}"
                );
            }
            assert_eq!(
                all.len(),
                per_piece.iter().sum::<usize>(),
                "total NON_EVASIONS count in {fen}"
            );
        }
    }

    #[test]
    fn capture_counts_on_hand_fixtures() {
        for &(fen, _, captures) in FIXTURES {
            let b = board(fen);
            assert_eq!(
                generated::<{ gen_type::CAPTURES }>(&b).len(),
                captures,
                "CAPTURES count in {fen}"
            );
        }
    }

    /// doc 05 §3.3 target masks: CAPTURES lands on enemy pieces only,
    /// QUIETS on empty squares only, NON_EVASIONS never on own pieces —
    /// and every from-square holds a piece of the side to move.
    #[test]
    fn target_mask_filtering() {
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let us = b.side_to_move();

            for m in &generated::<{ gen_type::CAPTURES }>(&b) {
                assert!(
                    b.occupancy(us.flip()).contains(m.to()),
                    "CAPTURES move {m} in {fen} must land on an enemy piece"
                );
            }
            for m in &generated::<{ gen_type::QUIETS }>(&b) {
                assert!(
                    !b.all_occupancy().contains(m.to()),
                    "QUIETS move {m} in {fen} must land on an empty square"
                );
            }
            for m in &generated::<{ gen_type::NON_EVASIONS }>(&b) {
                assert!(
                    !b.occupancy(us).contains(m.to()),
                    "move {m} in {fen} may never land on an own piece"
                );
                assert_eq!(
                    b.piece_on(m.from()).map(|p| p.color),
                    Some(us),
                    "move {m} in {fen} must start on a mover piece"
                );
                assert_eq!(m.kind(), MoveKind::Normal); // no pawns/castles yet
            }
        }
    }

    /// doc 05 §5.1 property 1 on this step's fixtures: no duplicates within
    /// a type, CAPTURES ∩ QUIETS = ∅, and CAPTURES ∪ QUIETS = NON_EVASIONS
    /// (lengths add up).
    #[test]
    fn captures_and_quiets_partition_non_evasions() {
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let captures = move_set(&generated::<{ gen_type::CAPTURES }>(&b));
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));

            assert!(
                captures.is_disjoint(&quiets),
                "CAPTURES ∩ QUIETS must be empty in {fen}"
            );
            let union: HashSet<Move> = captures.union(&quiets).copied().collect();
            assert_eq!(union, all, "CAPTURES ⊎ QUIETS != NON_EVASIONS in {fen}");
            assert_eq!(captures.len() + quiets.len(), all.len());
        }
    }

    /// Exact move-set equality on the knight fixture: every generated move
    /// lands precisely on the hand-listed target squares (doc 05 §4 step 2).
    #[test]
    fn knight_fixture_exact_move_set() {
        let b = board("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1");
        let expected: HashSet<Move> = [
            // Nc3 (d5 is the lone capture):
            (Square::C3, Square::A2),
            (Square::C3, Square::A4),
            (Square::C3, Square::B1),
            (Square::C3, Square::B5),
            (Square::C3, Square::D1),
            (Square::C3, Square::D5),
            (Square::C3, Square::E2),
            (Square::C3, Square::E4),
            // Ke1:
            (Square::E1, Square::D1),
            (Square::E1, Square::D2),
            (Square::E1, Square::E2),
            (Square::E1, Square::F1),
            (Square::E1, Square::F2),
        ]
        .into_iter()
        .map(|(from, to)| Move::normal(from, to))
        .collect();

        assert_eq!(
            move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b)),
            expected
        );
        assert_eq!(
            move_set(&generated::<{ gen_type::CAPTURES }>(&b)),
            HashSet::from([Move::normal(Square::C3, Square::D5)])
        );
    }

    /// Exact move set for the boxed-rook fixture — the rook contributes
    /// nothing, so this pins that blocked sliders emit no moves at all.
    #[test]
    fn boxed_rook_exact_move_set() {
        let b = board("4k3/8/8/8/8/8/N7/RK6 w - - 0 1");
        let expected: HashSet<Move> = [
            (Square::A2, Square::B4),
            (Square::A2, Square::C1),
            (Square::A2, Square::C3),
            (Square::B1, Square::B2),
            (Square::B1, Square::C1),
            (Square::B1, Square::C2),
        ]
        .into_iter()
        .map(|(from, to)| Move::normal(from, to))
        .collect();

        assert_eq!(
            move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b)),
            expected
        );
    }

    /// doc 05 §5.4: emission order is part of bench reproducibility. Within
    /// one generate call the mover piece types appear in the pinned class
    /// order N → B → R → Q → K (pawns first and castles last once they land).
    #[test]
    fn emission_order_class_is_pinned() {
        fn stage(pt: PieceType) -> usize {
            match pt {
                PieceType::Pawn => 0,
                PieceType::Knight => 1,
                PieceType::Bishop => 2,
                PieceType::Rook => 3,
                PieceType::Queen => 4,
                PieceType::King => 5,
            }
        }
        for &(fen, _, _) in FIXTURES {
            let b = board(fen);
            let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
            let stages: Vec<usize> = all
                .iter()
                .map(|m| stage(b.piece_on(m.from()).unwrap().piece_type))
                .collect();
            assert!(
                stages.windows(2).all(|w| w[0] <= w[1]),
                "emission order not monotone in {fen}: {stages:?}"
            );
        }
    }

    /// `generate` appends without clearing (doc 05 §3.3): callers compose
    /// sub-ranges, so CAPTURES followed by QUIETS into the same list must
    /// concatenate.
    #[test]
    fn generate_appends_without_clearing() {
        let b = board("k7/8/8/8/1p1R2p1/8/8/4K3 w - - 0 1");
        let captures = generated::<{ gen_type::CAPTURES }>(&b);
        let quiets = generated::<{ gen_type::QUIETS }>(&b);

        let mut composed = MoveList::new();
        generate_captures(&b, &mut composed);
        assert_eq!(composed.len(), captures.len());
        generate_quiets(&b, &mut composed);
        assert_eq!(composed.len(), captures.len() + quiets.len());
        assert_eq!(&composed[..captures.len()], &captures[..]);
        assert_eq!(&composed[captures.len()..], &quiets[..]);
    }

    /// The EVASIONS↔in-check contract is a debug_assert (doc 05 §3.1);
    /// debug builds only — release strips it by design.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "generate::<EVASIONS> iff the side to move is in check")]
    fn evasions_without_check_panics_in_debug() {
        let b = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let mut list = MoveList::new();
        generate_evasions(&b, &mut list);
    }
}
