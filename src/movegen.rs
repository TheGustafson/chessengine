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
//! P3 step 5 status (doc 05 §4): generation is complete for all four
//! GenTypes — GenType plumbing and non-pawn moves (step 2), pawn moves
//! (step 3), evasions (step 4), and castling (step 5). Still to come:
//! the legality layer (`Board::legal`, `generate_legal`, step 6) and the
//! TT-move validator (`Board::pseudo_legal`, step 7).

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::{Board, CastlingRights};
use crate::moves::{Move, MoveList};
use crate::types::{Color, Direction, PieceType, Square};

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

/// Pawn pushes, promotions, captures, and en passant (doc 05 §3.4).
///
/// Never loops per pawn: the whole pawn set is shifted at once and the
/// from-square recovered as `to − offset` — every pawn in one batch moved
/// by the same offset, and a shifted-and-masked destination has a valid,
/// non-wrapping origin by construction, so `Square::offset` (panics only
/// on a genuine bug) is the right primitive. `WHITE` makes every
/// direction and rank mask below a compile-time constant.
fn pawn_moves<const GT: u8, const WHITE: bool>(
    board: &Board,
    target: Bitboard,
    list: &mut MoveList,
) {
    use gen_type::{CAPTURES, EVASIONS, QUIETS};

    let us = if WHITE { Color::White } else { Color::Black };
    let them = us.flip();
    let up = if WHITE {
        Direction::North
    } else {
        Direction::South
    };
    let up_right = if WHITE {
        Direction::NorthEast
    } else {
        Direction::SouthWest
    };
    let up_left = if WHITE {
        Direction::NorthWest
    } else {
        Direction::SouthEast
    };
    let down = opposite(up);
    // Relative rank 7 (pawns about to promote) and the rank-3 relay rank.
    let rank7 = if WHITE {
        Bitboard::RANKS[6]
    } else {
        Bitboard::RANKS[1]
    };
    let rank3 = if WHITE {
        Bitboard::RANKS[2]
    } else {
        Bitboard::RANKS[5]
    };

    let empty = !board.all_occupancy();
    // EVASIONS: a pawn capture can only resolve the check by taking the
    // checker itself; interpositions are the push loops' target mask.
    let enemies = if GT == EVASIONS {
        board.checkers()
    } else {
        board.occupancy(them)
    };
    let pawns = board.pieces(us, PieceType::Pawn);
    let on7 = pawns & rank7;
    let not7 = pawns & !rank7;

    // --- 1. Single and double pushes (everything except CAPTURES) ---
    if GT != CAPTURES {
        let mut b1 = not7.shift(up) & empty;
        // Rank-3 relay trick: a double push exists iff the single push
        // landed on relative rank 3 AND one more shift is also empty —
        // both-empty tested with two ANDs, zero per-pawn work.
        let mut b2 = (b1 & rank3).shift(up) & empty;
        if GT == EVASIONS {
            // Only interpositions on the block/capture ray count.
            b1 &= target;
            b2 &= target;
        }
        for to in b1 {
            list.push(Move::normal(to.offset(down), to));
        }
        for to in b2 {
            list.push(Move::normal(to.offset(down).offset(down), to));
        }
    }

    // --- 2. Promotions (the §3.2 split, encoded by `push_promotions`) ---
    if !on7.is_empty() {
        if GT != QUIETS {
            // Capturing promotions, both diagonals.
            for to in on7.shift(up_right) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_right)), to);
            }
            for to in on7.shift(up_left) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_left)), to);
            }
        }
        let mut b3 = on7.shift(up) & empty;
        if GT == EVASIONS {
            b3 &= target; // a push promotion must land on the block ray
        }
        for to in b3 {
            push_promotions::<GT, false>(list, to.offset(down), to);
        }
    }

    // --- 3. Ordinary and en-passant captures (not for QUIETS) ---
    if GT != QUIETS {
        for to in not7.shift(up_right) & enemies {
            list.push(Move::normal(to.offset(opposite(up_right)), to));
        }
        for to in not7.shift(up_left) & enemies {
            list.push(Move::normal(to.offset(opposite(up_left)), to));
        }
        if let Some(ep) = board.en_passant_square() {
            // D9 (design/04 §3.7): an ep square is only ever recorded when
            // a legal — hence pseudo-legal — ep capture exists, both on FEN
            // load and in make_move, so a capturer must be present.
            debug_assert!(
                !(not7 & attacks::pawn_attacks(them, ep)).is_empty(),
                "D9 violated: recorded ep square {ep} has no pseudo-legal capturer"
            );
            // EVASIONS: if the double-push origin (one step `up` of the ep
            // square from our view) lies on the checking ray, the check is
            // a discovered slider check through that square — an ep capture
            // can never resolve it. Skip. (Unreachable while D9 holds; kept
            // as defense-in-depth per doc 05 §3.4.)
            if GT == EVASIONS && target.contains(ep.offset(up)) {
                return;
            }
            // Our pawns attacking the ep square = pawn attacks FROM the ep
            // square as if an enemy pawn stood there (reversal trick).
            for from in not7 & attacks::pawn_attacks(them, ep) {
                list.push(Move::en_passant(from, ep));
            }
        }
    }
}

/// Emit the promotions for one from/to pair per the doc 05 §3.2 table:
/// queen first, then R, B, N (pinned emission order, §5.4). `CAPTURE`
/// distinguishes capturing promotions (all four pieces belong to CAPTURES)
/// from push promotions (queen to CAPTURES, underpromotions to QUIETS).
#[inline(always)]
fn push_promotions<const GT: u8, const CAPTURE: bool>(
    list: &mut MoveList,
    from: Square,
    to: Square,
) {
    use gen_type::{CAPTURES, EVASIONS, NON_EVASIONS, QUIETS};
    let all = GT == EVASIONS || GT == NON_EVASIONS;
    if GT == CAPTURES || all {
        list.push(Move::promotion(from, to, PieceType::Queen));
    }
    if (GT == CAPTURES && CAPTURE) || (GT == QUIETS && !CAPTURE) || all {
        list.push(Move::promotion(from, to, PieceType::Rook));
        list.push(Move::promotion(from, to, PieceType::Bishop));
        list.push(Move::promotion(from, to, PieceType::Knight));
    }
}

/// Direction negation. Doc 05 §3.4 keeps this private to movegen until
/// doc 02 grows a `Direction::opposite`.
const fn opposite(d: Direction) -> Direction {
    match d {
        Direction::North => Direction::South,
        Direction::NorthEast => Direction::SouthWest,
        Direction::East => Direction::West,
        Direction::SouthEast => Direction::NorthWest,
        Direction::South => Direction::North,
        Direction::SouthWest => Direction::NorthEast,
        Direction::West => Direction::East,
        Direction::NorthWest => Direction::SouthEast,
    }
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

/// Castling generation (doc 05 §3.7): the right must be present and the
/// between-king-and-rook path empty at generation time — the rights
/// themselves guarantee king/rook placement (make/unmake revokes them on
/// any king/rook move or capture). Attacked-square testing is deliberately
/// deferred to `Board::legal` (step 6), matching SF.
///
/// Only reached under QUIETS / NON_EVASIONS: a castle never captures and
/// never evades, since the mover is not in check in both contexts. The
/// encoding is the king's two-square move (D2): e1g1 / e1c1 relative.
/// Emission order: kingside before queenside (doc 05 §5.4 order pin).
fn castling_moves<const WHITE: bool>(board: &Board, list: &mut MoveList) {
    let us = if WHITE { Color::White } else { Color::Black };
    let from = Square::E1.relative_to(us);
    for (right, to) in [
        (CastlingRights::kingside(us), Square::G1.relative_to(us)),
        (CastlingRights::queenside(us), Square::C1.relative_to(us)),
    ] {
        if board.castling_rights().contains(right)
            && (board.all_occupancy() & board.castling_path(right)).is_empty()
        {
            list.push(Move::castling(from, to));
        }
    }
}

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

    /// The pawn-made subset of one generation pass (promotions and ep
    /// included by construction; piece/king/castle moves filtered out).
    fn pawn_move_set<const GT: u8>(board: &Board) -> HashSet<Move> {
        let list = generated::<GT>(board);
        move_set(&list)
            .into_iter()
            .filter(|m| {
                board
                    .piece_on(m.from())
                    .expect("generated move must start on a piece")
                    .piece_type
                    == PieceType::Pawn
            })
            .collect()
    }

    const ALL_PROMOS: [PieceType; 4] = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    const UNDER_PROMOS: [PieceType; 3] = [PieceType::Rook, PieceType::Bishop, PieceType::Knight];

    /// The promotion fan `from -> to` for the given pieces, as a set.
    fn promo_fan(from: Square, to: Square, pieces: &[PieceType]) -> HashSet<Move> {
        pieces
            .iter()
            .map(|&pt| Move::promotion(from, to, pt))
            .collect()
    }

    /// Hand fixtures with per-piece counts hand-verified against the board
    /// diagram (doc 05 §4 step 2, §5 for this step). No mover-side pawns
    /// (pawn generation contributes nothing) and no castling rights, so
    /// every count below stays valid once `castling_moves` (step 5) lands.
    /// Fields: FEN, then expected (knight, bishop, rook, queen, king)
    /// NON_EVASIONS counts, then expected CAPTURES total.
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
                // These movers have no pawns and no castling rights.
                assert_eq!(m.kind(), MoveKind::Normal);
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

    // -----------------------------------------------------------------------
    // P3 step 3: pawn generation (doc 05 §3.4, §4 step 3, §5.2)
    // -----------------------------------------------------------------------

    #[test]
    fn startpos_pawn_pushes() {
        let b = board(crate::fen::START_FEN);
        // 8 single + 8 double pushes, all quiet; nothing to capture.
        let mut expected = HashSet::new();
        for file in 0..8 {
            let from = Square::from_coords(file, 1);
            expected.insert(Move::normal(from, Square::from_coords(file, 2)));
            expected.insert(Move::normal(from, Square::from_coords(file, 3)));
        }
        assert_eq!(pawn_move_set::<{ gen_type::QUIETS }>(&b), expected);
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
        // Doubles as the perft(1) = 20 pin: the castle paths are occupied,
        // so this count stays exact once step 5 lands.
        assert_eq!(generated::<{ gen_type::NON_EVASIONS }>(&b).len(), 20);
    }

    /// Doc 05 §4 step 3 "double-push blocking": a blocker on the relay
    /// square kills both pushes; a blocker one further kills only the
    /// double. Both colors (the rank-3 relay mask is per-color).
    #[test]
    fn blocked_and_relayed_pushes() {
        // Directly blocked: no single push, hence no double either.
        let b = board("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b).is_empty());

        // Relay square e3 free, e4 occupied: single push only.
        let b = board("4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([Move::normal(Square::E2, Square::E3)])
        );

        // Black mirrors of both shapes.
        let b = board("4k3/4p3/4P3/8/8/8/8/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b).is_empty());
        let b = board("4k3/4p3/8/4P3/8/8/8/4K3 b - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([Move::normal(Square::E7, Square::E6)])
        );
    }

    /// Diagonal capture shifts in both directions for both colors, with the
    /// from-square recovered as `to − offset`.
    #[test]
    fn captures_both_directions_both_colors() {
        let b = board("4k3/8/8/3p1p2/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::normal(Square::E4, Square::D5),
                Move::normal(Square::E4, Square::F5),
            ])
        );
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::E4, Square::E5)])
        );

        let b = board("4k3/8/8/4p3/3P1P2/8/8/4K3 b - - 0 1");
        assert_eq!(
            pawn_move_set::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::normal(Square::E5, Square::D4),
                Move::normal(Square::E5, Square::F4),
            ])
        );
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::E5, Square::E4)])
        );
    }

    /// A/H-file capture shifts must not wrap: each fixture places the only
    /// enemy piece exactly where the unmasked ±9/±7 shift would land.
    #[test]
    fn capture_shifts_do_not_wrap_files() {
        // White h5; unmasked NE (<<9) from h5 would land on a7.
        let b = board("4k3/p7/8/7P/8/8/8/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
        assert_eq!(
            pawn_move_set::<{ gen_type::QUIETS }>(&b),
            HashSet::from([Move::normal(Square::H5, Square::H6)])
        );

        // White a5; unmasked NW (<<7) from a5 would land on h5.
        let b = board("4k3/8/8/P6p/8/8/8/4K3 w - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());

        // Black a4; unmasked SW (>>9) from a4 would land on h2.
        let b = board("4k3/8/8/8/p7/8/7P/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());

        // Black h4; unmasked SE (>>7) from h4 would land on a3.
        let b = board("4k3/8/8/8/7p/P7/8/4K3 b - - 0 1");
        assert!(pawn_move_set::<{ gen_type::CAPTURES }>(&b).is_empty());
    }

    /// Non-capturing promotions (§3.2 table, right column of row 2): the
    /// queen goes to CAPTURES, the underpromotions to QUIETS, all four to
    /// NON_EVASIONS. Both colors.
    #[test]
    fn push_promotion_split() {
        for (fen, from, to) in [
            ("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1", Square::G7, Square::G8),
            ("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1", Square::B2, Square::B1),
        ] {
            let b = board(fen);
            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                promo_fan(from, to, &[PieceType::Queen]),
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                promo_fan(from, to, &UNDER_PROMOS),
                "QUIETS in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
                promo_fan(from, to, &ALL_PROMOS),
                "NON_EVASIONS in {fen}"
            );
        }
    }

    /// Capturing promotions (§3.2 table row 1): all four pieces belong to
    /// CAPTURES in both directions; QUIETS gets only the *push*
    /// underpromotions. Both colors, both diagonals at once.
    #[test]
    fn promotion_captures_full_split() {
        for (fen, from, (left, up, right)) in [
            (
                "1n1nk3/2P5/8/8/8/8/8/4K3 w - - 0 1",
                Square::C7,
                (Square::B8, Square::C8, Square::D8),
            ),
            (
                "4k3/8/8/8/8/8/6p1/4KN1N b - - 0 1",
                Square::G2,
                (Square::F1, Square::G1, Square::H1),
            ),
        ] {
            let b = board(fen);
            let mut captures = promo_fan(from, left, &ALL_PROMOS);
            captures.extend(promo_fan(from, right, &ALL_PROMOS));
            captures.extend(promo_fan(from, up, &[PieceType::Queen]));
            let quiets = promo_fan(from, up, &UNDER_PROMOS);

            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                captures,
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                quiets,
                "QUIETS in {fen}"
            );
            let mut all = captures;
            all.extend(quiets);
            assert_eq!(
                pawn_move_set::<{ gen_type::NON_EVASIONS }>(&b),
                all,
                "NON_EVASIONS in {fen}"
            );
        }
    }

    /// Doc 05 §5.2's classic promotion position, both colors (the position
    /// is its own color mirror): blocked push promotions contribute nothing
    /// (a7/c7 resp. f2/h2), the free pawn gets the full 12-move fan.
    #[test]
    fn classic_promotion_position_partition() {
        for (fen, from, (left, up, right)) in [
            (
                "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
                Square::B7,
                (Square::A8, Square::B8, Square::C8),
            ),
            (
                "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
                Square::G2,
                (Square::F1, Square::G1, Square::H1),
            ),
        ] {
            let b = board(fen);
            let mut captures = promo_fan(from, left, &ALL_PROMOS);
            captures.extend(promo_fan(from, right, &ALL_PROMOS));
            captures.extend(promo_fan(from, up, &[PieceType::Queen]));
            let quiets = promo_fan(from, up, &UNDER_PROMOS);

            assert_eq!(
                pawn_move_set::<{ gen_type::CAPTURES }>(&b),
                captures,
                "CAPTURES in {fen}"
            );
            assert_eq!(
                pawn_move_set::<{ gen_type::QUIETS }>(&b),
                quiets,
                "QUIETS in {fen}"
            );
        }
    }

    /// En passant: emitted (with `MoveKind::EnPassant` and the recovered
    /// from-square) by CAPTURES and NON_EVASIONS, never by QUIETS; both
    /// capturers found when two pawns flank the target; nothing emitted
    /// without a recorded ep square.
    #[test]
    fn en_passant_generation() {
        fn ep_moves<const GT: u8>(b: &Board) -> HashSet<Move> {
            generated::<GT>(b)
                .iter()
                .copied()
                .filter(|m| m.kind() == MoveKind::EnPassant)
                .collect()
        }

        // Single white capturer (corpus opening after 1.e4 e6 2.e5 d5).
        let b = board("rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        let ep = Move::en_passant(Square::E5, Square::D6);
        assert_eq!(ep_moves::<{ gen_type::CAPTURES }>(&b), HashSet::from([ep]));
        assert_eq!(
            ep_moves::<{ gen_type::NON_EVASIONS }>(&b),
            HashSet::from([ep])
        );
        assert!(ep_moves::<{ gen_type::QUIETS }>(&b).is_empty());

        // Two white capturers flanking d5.
        let b = board("4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1");
        assert_eq!(
            ep_moves::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::en_passant(Square::C5, Square::D6),
                Move::en_passant(Square::E5, Square::D6),
            ])
        );

        // Two black capturers flanking e4 (black's shifts and rank masks).
        let b = board("4k3/8/8/8/3pPp2/8/8/4K3 b - e3 0 1");
        assert_eq!(
            ep_moves::<{ gen_type::CAPTURES }>(&b),
            HashSet::from([
                Move::en_passant(Square::D4, Square::E3),
                Move::en_passant(Square::F4, Square::E3),
            ])
        );

        // No ep square recorded: same placement, zero ep moves.
        let b = board("4k3/8/8/2PpP3/8/8/8/4K3 w - - 0 1");
        assert!(ep_moves::<{ gen_type::NON_EVASIONS }>(&b).is_empty());
    }

    /// Doc 05 §5.4 order pins within pawn generation: pushes → promotions
    /// (Q, R, B, N) → captures → ep, then pieces and king — asserted as one
    /// exact emission sequence on a fixture exercising all four phases.
    #[test]
    fn pawn_emission_order_is_pinned() {
        let b = board("4k3/6P1/8/3pP3/2P5/8/P7/4K3 w - d6 0 2");
        let all = generated::<{ gen_type::NON_EVASIONS }>(&b);
        let expected = [
            // Single pushes (destination bitboard order), then doubles.
            Move::normal(Square::A2, Square::A3),
            Move::normal(Square::C4, Square::C5),
            Move::normal(Square::E5, Square::E6),
            Move::normal(Square::A2, Square::A4),
            // Push promotions, Q → R → B → N.
            Move::promotion(Square::G7, Square::G8, PieceType::Queen),
            Move::promotion(Square::G7, Square::G8, PieceType::Rook),
            Move::promotion(Square::G7, Square::G8, PieceType::Bishop),
            Move::promotion(Square::G7, Square::G8, PieceType::Knight),
            // Captures (up-right batch first), then en passant.
            Move::normal(Square::C4, Square::D5),
            Move::en_passant(Square::E5, Square::D6),
            // King last (no other pieces on the board).
            Move::normal(Square::E1, Square::D1),
            Move::normal(Square::E1, Square::F1),
            Move::normal(Square::E1, Square::D2),
            Move::normal(Square::E1, Square::E2),
            Move::normal(Square::E1, Square::F2),
        ];
        assert_eq!(&all[..], &expected[..]);
    }

    /// Pawn-heavy positions for the step-3 "Done" gate (doc 05 §4): the
    /// disjoint-union property and the §3.2 per-type semantics must hold
    /// with promotions and en passant in play. Castling rights present in
    /// some fixtures are inert until step 5 — the properties below are
    /// unaffected either way.
    const PAWN_FIXTURES: &[&str] = &[
        crate::fen::START_FEN,
        // Kiwipete.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // Perft pos 3 (ep-pin geometry), pos 5 (promotion storm), pos 6
        // (quiet middlegame breadth). Pos 4 is excluded here on purpose:
        // its mover is in check, so it belongs to step 4's EVASIONS tests.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        // The classic promotion position, both colors.
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        // Live ep squares, one per color.
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3",
    ];

    #[test]
    fn pawn_fixture_partition_and_semantics() {
        for &fen in PAWN_FIXTURES {
            let b = board(fen);
            let captures = move_set(&generated::<{ gen_type::CAPTURES }>(&b));
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));

            // Doc 05 §5.1 property 1.
            assert!(
                captures.is_disjoint(&quiets),
                "CAPTURES ∩ QUIETS must be empty in {fen}"
            );
            let union: HashSet<Move> = captures.union(&quiets).copied().collect();
            assert_eq!(union, all, "CAPTURES ⊎ QUIETS != NON_EVASIONS in {fen}");

            // §3.2 semantics: everything CAPTURES emits is a real capture
            // or a queen promotion (the rule `capture_stage` will mirror in
            // step 6); QUIETS never captures, never queens, never ep.
            let enemies = b.occupancy(b.side_to_move().flip());
            for &m in &captures {
                let stage_capture = match m.kind() {
                    MoveKind::Normal => enemies.contains(m.to()),
                    MoveKind::Promotion => {
                        enemies.contains(m.to()) || m.promotion_piece() == Some(PieceType::Queen)
                    }
                    MoveKind::EnPassant => Some(m.to()) == b.en_passant_square(),
                    MoveKind::Castling => false,
                };
                assert!(stage_capture, "CAPTURES emitted {m} in {fen}");
            }
            for &m in &quiets {
                assert!(
                    !b.all_occupancy().contains(m.to()),
                    "QUIETS move {m} lands on a piece in {fen}"
                );
                assert_ne!(m.kind(), MoveKind::EnPassant, "QUIETS emitted ep in {fen}");
                if m.kind() == MoveKind::Promotion {
                    assert_ne!(
                        m.promotion_piece(),
                        Some(PieceType::Queen),
                        "QUIETS emitted a queen promotion in {fen}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // P3 step 4: evasion generation (doc 05 §3.6, §4 step 4, §5.2)
    // -----------------------------------------------------------------------

    use crate::types::Piece;

    /// The unrestricted pseudo-legal move set on an IN-CHECK position — the
    /// reference the evasion oracle filters. `generate::<NON_EVASIONS>`
    /// debug-asserts "not in check" (that assert is itself part of the
    /// contract under test), so the reference bypasses the dispatcher and
    /// calls the color worker directly.
    fn all_pseudo_ignoring_check(board: &Board) -> MoveList {
        let mut list = MoveList::new();
        match board.side_to_move() {
            Color::White => generate_all::<{ gen_type::NON_EVASIONS }, true>(board, &mut list),
            Color::Black => generate_all::<{ gen_type::NON_EVASIONS }, false>(board, &mut list),
        }
        list
    }

    /// Make/unmake legality oracle (doc 05 §5.1.5) for in-check positions:
    /// play the move, ask whether our king is attacked, undo. Castling is
    /// resolved by rule instead — castling out of check is illegal, and the
    /// after-state cannot express that (the king may land on a perfectly
    /// safe square).
    fn oracle_legal(board: &mut Board, m: Move) -> bool {
        debug_assert!(
            !board.checkers().is_empty(),
            "this oracle only serves the in-check fixtures"
        );
        if m.kind() == MoveKind::Castling {
            return false;
        }
        let us = board.side_to_move();
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
        let safe = (board.attackers_to(board.king_square(us), board.all_occupancy())
            & board.occupancy(us.flip()))
        .is_empty();
        board.unmake_move(m);
        safe
    }

    /// Doc 05 §4 step 4 completeness gate: EVASIONS, filtered through the
    /// make/unmake oracle, must equal the full pseudo-legal superset under
    /// the same filter — no legal block/capture/flee missing, nothing
    /// extra. Pins the §3.6 shape rules along the way.
    fn assert_evasion_completeness(b: &Board, fen: &str) {
        let us = b.side_to_move();
        let ksq = b.king_square(us);
        assert!(!b.checkers().is_empty(), "fixture must be in check: {fen}");

        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(b));
        let superset = move_set(&all_pseudo_ignoring_check(b));
        assert!(
            evasions.is_subset(&superset),
            "EVASIONS must be a subset of the pseudo-legal superset in {fen}"
        );

        // §3.6 rule 5: castling is never an evasion. Rule 2: double check
        // is king-only. Rule 3: single-check non-king moves land on the
        // block-or-capture mask (en passant resolves on the captured
        // pawn's square, not on `to`, so it is exempt from the to-test).
        for &m in &evasions {
            assert_ne!(m.kind(), MoveKind::Castling, "castle evasion {m} in {fen}");
        }
        if b.checkers().more_than_one() {
            for &m in &evasions {
                assert_eq!(
                    m.from(),
                    ksq,
                    "non-king evasion {m} under double check in {fen}"
                );
            }
        } else {
            let target = attacks::between(ksq, b.checkers().lsb().unwrap());
            for &m in &evasions {
                if m.from() != ksq && m.kind() != MoveKind::EnPassant {
                    assert!(
                        target.contains(m.to()),
                        "evasion {m} off the block/capture ray in {fen}"
                    );
                }
            }
        }

        let mut scratch = b.clone();
        let legal_evasions: HashSet<Move> = evasions
            .into_iter()
            .filter(|&m| oracle_legal(&mut scratch, m))
            .collect();
        let legal_superset: HashSet<Move> = superset
            .into_iter()
            .filter(|&m| oracle_legal(&mut scratch, m))
            .collect();
        assert_eq!(
            legal_evasions, legal_superset,
            "evasion completeness in {fen}"
        );
    }

    /// In-check corpus: single checks with blocks/captures/flights, pawn
    /// and knight checks, ep-as-evasion, promotion evasions, double
    /// checks, castling rights while in check, x-ray flights, and one
    /// checkmate (pseudo evasions exist, none legal).
    const IN_CHECK_FIXTURES: &[&str] = &[
        // Perft pos 4 (white in check by Bb6) and its color mirror.
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        // Rook check: capture (Nxe4), interposition (Be3), flights.
        "4k3/8/8/8/4r3/8/3N4/4K1B1 w - - 0 1",
        // The checker is pawn-capturable; a second enemy piece is not.
        "k7/8/8/8/3r1b2/4P3/7K/8 w - - 0 1",
        // Interposition by a double pawn push through the relay square.
        "8/2p5/8/k6R/8/8/8/4K3 b - - 0 1",
        // Ep capture of the checking double-pushed pawn.
        "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
        // Capturing promotions resolve the check; push promotions cannot.
        "4r3/3P1P2/8/8/8/8/8/4K2k w - - 0 1",
        // A push promotion lands on the checking ray and blocks.
        "K6r/2P5/8/8/8/8/8/7k w - - 0 1",
        // Knight check: no interposition square exists.
        "4k3/8/8/8/8/5n2/8/4K3 w - - 0 1",
        // Pawn check, capturable by knight or king.
        "4k3/8/8/8/8/8/3p4/1N2K3 w - - 0 1",
        // Double checks: king moves only.
        "r3k3/8/8/1B6/8/8/8/4RK2 b - - 0 1",
        "k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1",
        // Castling rights present while in check (no castle evasions).
        "r3k2r/8/8/8/B7/8/8/4K3 b kq - 0 1",
        // King on the checker's ray: the x-ray flight e5 and the on-ray
        // step e3 are pseudo-generated, both oracle-illegal.
        "4k3/8/8/8/4K3/8/8/4r3 w - - 0 1",
        // Fool's mate: the lone pseudo evasion Kf2 is illegal.
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    ];

    #[test]
    fn evasions_complete_on_in_check_fixtures() {
        for &fen in IN_CHECK_FIXTURES {
            assert_evasion_completeness(&board(fen), fen);
        }
    }

    /// Exact block/capture/flee set for a single rook check (doc 05 §5.2):
    /// Nd2 captures the checker, Bg1 interposes on e3, the king has four
    /// flights (d2 is own-occupied). Ke2 stays on the checker's file:
    /// pseudo-generated — flight safety is legal()'s job (§3.6 rule 4) —
    /// and rejected by the oracle.
    #[test]
    fn single_check_block_capture_flee() {
        let b = board("4k3/8/8/8/4r3/8/3N4/4K1B1 w - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::D2, Square::E4),
            Move::normal(Square::G1, Square::E3),
            Move::normal(Square::E1, Square::D1),
            Move::normal(Square::E1, Square::E2),
            Move::normal(Square::E1, Square::F1),
            Move::normal(Square::E1, Square::F2),
        ]
        .into_iter()
        .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        let mut scratch = b.clone();
        assert!(!oracle_legal(
            &mut scratch,
            Move::normal(Square::E1, Square::E2)
        ));
        assert!(oracle_legal(
            &mut scratch,
            Move::normal(Square::D2, Square::E4)
        ));
    }

    /// Double check emits king moves only (doc 05 §3.6 rule 2, via
    /// `more_than_one`) — including flights that legal() will reject.
    #[test]
    fn double_check_generates_king_moves_only() {
        // Re1 + Bb5 both check e8; the a8 rook could otherwise block.
        let b = board("r3k3/8/8/1B6/8/8/8/4RK2 b - - 0 1");
        assert!(b.checkers().more_than_one());
        let expected: HashSet<Move> = [Square::D7, Square::D8, Square::E7, Square::F7, Square::F8]
            .into_iter()
            .map(|to| Move::normal(Square::E8, to))
            .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        // Nf3 + Re8 both check e1 (white side of the color dispatch).
        let b = board("k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1");
        assert!(b.checkers().more_than_one());
        let expected: HashSet<Move> = [Square::D1, Square::D2, Square::E2, Square::F1, Square::F2]
            .into_iter()
            .map(|to| Move::normal(Square::E1, to))
            .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// §3.4 `enemies = checkers()`: the e3 pawn attacks both d4 (a rook
    /// that is NOT the checker) and f4 (the checking bishop); only the
    /// checker capture is emitted.
    #[test]
    fn evasion_pawn_captures_only_the_checker() {
        let b = board("k7/8/8/8/3r1b2/4P3/7K/8 w - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::E3, Square::F4),
            Move::normal(Square::H2, Square::G1),
            Move::normal(Square::H2, Square::G2),
            Move::normal(Square::H2, Square::G3),
            Move::normal(Square::H2, Square::H1),
            Move::normal(Square::H2, Square::H3),
        ]
        .into_iter()
        .collect();
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&b));
        assert!(!evasions.contains(&Move::normal(Square::E3, Square::D4)));
        assert_eq!(evasions, expected);
    }

    /// Rh5 checks a5 along rank 5; c7-c5 — a double push through the empty
    /// relay square — is the only non-king evasion (the single push c6
    /// misses the ray and must NOT appear).
    #[test]
    fn evasion_pawn_interposition_by_double_push() {
        let b = board("8/2p5/8/k6R/8/8/8/4K3 b - - 0 1");
        let expected: HashSet<Move> = [
            Move::normal(Square::C7, Square::C5),
            Move::normal(Square::A5, Square::A4),
            Move::normal(Square::A5, Square::A6),
            Move::normal(Square::A5, Square::B4),
            Move::normal(Square::A5, Square::B5),
            Move::normal(Square::A5, Square::B6),
        ]
        .into_iter()
        .collect();
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// Promotions under EVASIONS (§3.2 table): the full four-piece fan is
    /// emitted, but only where it resolves the check — capturing the
    /// checker on e8, or landing a push promotion on the checking ray.
    #[test]
    fn evasion_promotions() {
        // Re8 checks e1: dxe8 and fxe8 capture the checker (full fans);
        // the non-resolving pushes d8/f8 are masked out.
        let b = board("4r3/3P1P2/8/8/8/8/8/4K2k w - - 0 1");
        let mut expected = promo_fan(Square::D7, Square::E8, &ALL_PROMOS);
        expected.extend(promo_fan(Square::F7, Square::E8, &ALL_PROMOS));
        for to in [Square::D1, Square::D2, Square::E2, Square::F1, Square::F2] {
            expected.insert(Move::normal(Square::E1, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);

        // Rh8 checks a8: the push promotion c8 lands ON the ray and
        // blocks (`b3 &= target` keeps it), again with the full fan.
        let b = board("K6r/2P5/8/8/8/8/8/7k w - - 0 1");
        let mut expected = promo_fan(Square::C7, Square::C8, &ALL_PROMOS);
        for to in [Square::A7, Square::B7, Square::B8] {
            expected.insert(Move::normal(Square::A8, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
    }

    /// Doc 05 §5.2 "ep as evasion": white just played d2-d4 giving check
    /// with the pawn itself; exd3 en passant captures the checker. D9
    /// keeps the ep square (the capture is legal), and the ep move is
    /// exempt from the to-on-ray mask — its resolution square is d4.
    #[test]
    fn ep_capture_of_the_checking_pawn_is_an_evasion() {
        let b = board("8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1");
        assert_eq!(b.en_passant_square(), Some(Square::D3));
        let ep = Move::en_passant(Square::E4, Square::D3);
        let mut expected = HashSet::from([ep]);
        for to in [
            Square::B4,
            Square::B5,
            Square::B6,
            Square::C4,
            Square::C6,
            Square::D4,
            Square::D5,
            Square::D6,
        ] {
            expected.insert(Move::normal(Square::C5, to));
        }
        assert_eq!(move_set(&generated::<{ gen_type::EVASIONS }>(&b)), expected);
        assert!(oracle_legal(&mut b.clone(), ep));
    }

    /// Doc 05 §5.2 "ep discovered-check skip": d2-d4 opened the c1-g5
    /// diagonal, so Bc1 checks THROUGH the vacated push origin and exd3
    /// can never resolve the check (§3.4). Through FEN, D9 already drops
    /// the unusable ep square at load; the generator's own skip branch is
    /// defense-in-depth, exercised by forcing the D9-violating state onto
    /// the board directly.
    #[test]
    fn ep_skipped_when_check_is_discovered_through_the_push_origin() {
        let b = board("8/8/8/6k1/3Pp3/8/8/2B1K3 b - d3 0 1");
        assert_eq!(b.en_passant_square(), None, "D9 must drop the ep square");
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&b));
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::EnPassant));
        // The e4 pawn still blocks on the diagonal with a plain push.
        assert!(evasions.contains(&Move::normal(Square::E4, Square::E3)));

        // Same placement with ep pinned to d3: now the §3.4 skip (not D9)
        // must reject the ep capture, while everything else is unchanged.
        let mut forced = Board::empty();
        for (piece, sq) in [
            (Piece::new(Color::White, PieceType::Bishop), Square::C1),
            (Piece::new(Color::White, PieceType::King), Square::E1),
            (Piece::new(Color::White, PieceType::Pawn), Square::D4),
            (Piece::new(Color::Black, PieceType::Pawn), Square::E4),
            (Piece::new(Color::Black, PieceType::King), Square::G5),
        ] {
            forced.put_piece(piece, sq);
        }
        forced.side_to_move = Color::Black;
        forced.finish_setup(Some(Square::D3), 0, 1);
        assert_eq!(forced.en_passant_square(), Some(Square::D3));
        let evasions = move_set(&generated::<{ gen_type::EVASIONS }>(&forced));
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::EnPassant));
        assert!(evasions.contains(&Move::normal(Square::E4, Square::E3)));
        assert_evasion_completeness(&forced, "forced ep-discovered-check fixture");
    }

    /// §3.6 rule 5: castling is never generated under EVASIONS, even with
    /// both rights present and both paths empty — while the unrestricted
    /// superset on the same position emits both castles, proving the
    /// exclusion is the EVASIONS rule and not a rights/path failure.
    #[test]
    fn no_castling_under_evasions() {
        let b = board("r3k2r/8/8/8/B7/8/8/4K3 b kq - 0 1");
        assert!(!b.checkers().is_empty());
        let evasions = generated::<{ gen_type::EVASIONS }>(&b);
        assert!(evasions.iter().all(|m| m.kind() != MoveKind::Castling));
        let superset = move_set(&all_pseudo_ignoring_check(&b));
        assert!(superset.contains(&Move::castling(Square::E8, Square::G8)));
        assert!(superset.contains(&Move::castling(Square::E8, Square::C8)));
    }

    /// The other direction of the §3.1 debug_assert: requesting
    /// NON_EVASIONS while in check panics in debug builds.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "generate::<EVASIONS> iff the side to move is in check")]
    fn non_evasions_while_in_check_panics_in_debug() {
        let b = board("4k3/8/8/8/8/5n2/8/4K3 w - - 0 1");
        let mut list = MoveList::new();
        generate_non_evasions(&b, &mut list);
    }

    // -----------------------------------------------------------------------
    // P3 step 5: castling generation (doc 05 §3.7, §4 step 5, §5.2)
    // -----------------------------------------------------------------------

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    /// The step-5 "Done" gate (doc 05 §4): Kiwipete emits both white
    /// castles in QUIETS — as the last two moves, kingside first (§5.4
    /// order pin) — and in NON_EVASIONS, never in CAPTURES.
    #[test]
    fn kiwipete_emits_both_white_castles_in_quiets() {
        let b = board(KIWIPETE);
        let oo = Move::castling(Square::E1, Square::G1);
        let ooo = Move::castling(Square::E1, Square::C1);

        let quiets = generated::<{ gen_type::QUIETS }>(&b);
        assert_eq!(&quiets[quiets.len() - 2..], &[oo, ooo]);

        let all = move_set(&generated::<{ gen_type::NON_EVASIONS }>(&b));
        assert!(all.contains(&oo) && all.contains(&ooo));
        assert!(
            generated::<{ gen_type::CAPTURES }>(&b)
                .iter()
                .all(|m| m.kind() != MoveKind::Castling)
        );
    }

    /// Black's rank-8 constants via Kiwipete with the side to move flipped
    /// (black is not in check there).
    #[test]
    fn kiwipete_reversed_emits_both_black_castles() {
        let b = board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1");
        let quiets = generated::<{ gen_type::QUIETS }>(&b);
        assert_eq!(
            &quiets[quiets.len() - 2..],
            &[
                Move::castling(Square::E8, Square::G8),
                Move::castling(Square::E8, Square::C8),
            ]
        );
    }

    /// §3.7 condition 1: each right gates exactly its own castle.
    #[test]
    fn castling_requires_the_right() {
        for (fen, kingside, queenside) in [
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", true, true),
            ("4k3/8/8/8/8/8/8/R3K2R w K - 0 1", true, false),
            ("4k3/8/8/8/8/8/8/R3K2R w Q - 0 1", false, true),
            ("4k3/8/8/8/8/8/8/R3K2R w - - 0 1", false, false),
            ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", true, true),
            ("r3k2r/8/8/8/8/8/8/4K3 b k - 0 1", true, false),
            ("r3k2r/8/8/8/8/8/8/4K3 b q - 0 1", false, true),
        ] {
            let b = board(fen);
            let us = b.side_to_move();
            let from = Square::E1.relative_to(us);
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            assert_eq!(
                quiets.contains(&Move::castling(from, Square::G1.relative_to(us))),
                kingside,
                "kingside in {fen}"
            );
            assert_eq!(
                quiets.contains(&Move::castling(from, Square::C1.relative_to(us))),
                queenside,
                "queenside in {fen}"
            );
        }
    }

    /// §3.7 condition 2, one blocker per path square (doc 05 §4 step 5):
    /// any piece — own or enemy — on f1/g1 kills the kingside castle and
    /// on b1/c1/d1 the queenside one, without disturbing the other wing.
    /// b1 is rook-path-only (the king never crosses it) yet still blocks.
    #[test]
    fn castling_blocked_on_each_path_square() {
        for (fen, kingside, queenside) in [
            ("4k3/8/8/8/8/8/8/R3KN1R w KQ - 0 1", false, true), // f1 own
            ("4k3/8/8/8/8/8/8/R3K1NR w KQ - 0 1", false, true), // g1 own
            ("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1", true, false), // b1 own
            ("4k3/8/8/8/8/8/8/R1N1K2R w KQ - 0 1", true, false), // c1 own
            ("4k3/8/8/8/8/8/8/R2NK2R w KQ - 0 1", true, false), // d1 own
            ("4k3/8/8/8/8/8/8/Rn2K2R w KQ - 0 1", true, false), // b1 enemy
        ] {
            let b = board(fen);
            let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
            assert_eq!(
                quiets.contains(&Move::castling(Square::E1, Square::G1)),
                kingside,
                "kingside in {fen}"
            );
            assert_eq!(
                quiets.contains(&Move::castling(Square::E1, Square::C1)),
                queenside,
                "queenside in {fen}"
            );
        }
    }

    /// Attacked-square testing is deferred to `Board::legal` (§3.7): the
    /// f3 queen attacks f1 (kingside crossing) and d1 (queenside crossing)
    /// without giving check, yet generation still emits both castles —
    /// step 6's legal() is what will drop them.
    #[test]
    fn castling_generated_through_attacked_squares() {
        let b = board("4k3/8/8/8/8/5q2/8/R3K2R w KQ - 0 1");
        assert!(b.checkers().is_empty());
        let quiets = move_set(&generated::<{ gen_type::QUIETS }>(&b));
        assert!(quiets.contains(&Move::castling(Square::E1, Square::G1)));
        assert!(quiets.contains(&Move::castling(Square::E1, Square::C1)));
    }
}
