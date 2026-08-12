//! The canonical 1,858-class complete-move policy vocabulary (2026-07-22).
//!
//! One flat categorical ID per *complete* move, replacing every factored
//! (from/to marginal) policy encoding — squares are categorical, promotions
//! need distinction, and independent from/to prediction loses the joint
//! dependency. The vocabulary is the standard compact chess set:
//!
//! * **1,792 geometric moves** — every (from, to) pair reachable by queen
//!   geometry (8 rays × up to 7 steps) or knight geometry, enumerated on the
//!   board. Covers kings, pawns (pushes, captures, double pushes), sliders,
//!   castling (king two-step, e1g1 style), en passant, and queen promotions
//!   (which ride their normal from→to class).
//! * **66 underpromotions** — the 22 pawn (from, to) promotion pairs
//!   (8 pushes + 14 captures on the last rank) × {knight, bishop, rook}.
//!
//! Total: **1,858**, matching the Lc0/AlphaZero-style flat policy vector.
//!
//! ## Canonical frame (the TEMPO deviation from Lc0 — deliberate)
//!
//! IDs are computed in the **trunk-normalized frame**, not the raw board
//! frame: the mover's perspective (vertical flip `^56` for black to move)
//! PLUS the HalfKAv2_hm horizontal mirror (file flip `^7` when the mover's
//! king, post-vertical-flip, sits on files a–d). Lc0 rank-flips only; our
//! accumulators additionally mirror by king file, and a policy head reading
//! mirrored features against unmirrored move IDs would have to spend
//! capacity learning to conditionally un-mirror (the exp-0121 orientation
//! lesson). The mirror is a symmetry of queen/knight geometry, so the
//! vocabulary is closed under it. Consumers that read the head at inference
//! must apply [`canonical_from_to`] with the same king square they fed the
//! feature transformer — one function, both sides, no drift.
//!
//! This module is the SINGLE implementation of the vocabulary: the bullet
//! trainer imports it through the crate dependency exactly like
//! [`halfka_index`](super::sota_features::halfka_index), which is what has
//! kept feature-index math convention-trap-free (design/16 lesson).

use crate::types::{PieceType, Square};

/// Vocabulary size: 1,792 geometric (from, to) classes + 66 underpromotions.
pub const POLICY_VOCAB: usize = 1858;

/// Number of geometric (queen|knight) from→to classes.
pub const GEOMETRIC_CLASSES: usize = 1792;

/// `NONE` marker for the packed u16 channel (fits any id < 0xFFFF).
pub const POLICY_ID_NONE: u16 = u16::MAX;

/// Dense lookup: `geom_table[from*64 + to]` → class id in `[0, 1792)`, or
/// `POLICY_ID_NONE` where (from, to) is not queen/knight geometry.
/// Built once; ~8 KiB.
static GEOM_TABLE: std::sync::LazyLock<[u16; 64 * 64]> = std::sync::LazyLock::new(|| {
    let mut table = [POLICY_ID_NONE; 64 * 64];
    let mut id: u16 = 0;
    // Enumeration order is part of the format: from-square major, then the
    // to-square in ascending order. Deterministic and easily reproduced.
    for from in 0..64usize {
        let (ff, fr) = ((from % 8) as i32, (from / 8) as i32);
        for to in 0..64usize {
            if to == from {
                continue;
            }
            let (tf, tr) = ((to % 8) as i32, (to / 8) as i32);
            let (df, dr) = (tf - ff, tr - fr);
            let queen_like = df == 0 || dr == 0 || df.abs() == dr.abs();
            let knight_like = (df.abs() == 1 && dr.abs() == 2) || (df.abs() == 2 && dr.abs() == 1);
            if queen_like || knight_like {
                table[from * 64 + to] = id;
                id += 1;
            }
        }
    }
    assert_eq!(
        id as usize, GEOMETRIC_CLASSES,
        "queen+knight geometry must count 1792"
    );
    table
});

/// Underpromotion block lookup: `(from, to)` on the promotion ranks → the
/// 0-based index of the pair in `[0, 22)`, from-major/to-ascending, or NONE.
/// The canonical frame always promotes on rank 8 (index 56..64): after the
/// vertical flip the mover's pawns always advance toward rank 8.
fn underpromo_pair_index(from: u8, to: u8) -> Option<u16> {
    let (ff, fr) = (i32::from(from % 8), i32::from(from / 8));
    let (tf, tr) = (i32::from(to % 8), i32::from(to / 8));
    if fr != 6 || tr != 7 || (tf - ff).abs() > 1 {
        return None;
    }
    // from ∈ {48..56}: pushes have df=0 (8 pairs), captures df=±1 (14 pairs).
    // Enumerate from-major, to-ascending to index the 22 pairs.
    let mut idx: u16 = 0;
    for f in 48..56u8 {
        let file = i32::from(f % 8);
        for t in 56..64u8 {
            let tfile = i32::from(t % 8);
            if (tfile - file).abs() <= 1 {
                if f == from && t == to {
                    return Some(idx);
                }
                idx += 1;
            }
        }
    }
    None
}

/// The trunk-normalized square transform: vertical flip for a black mover,
/// then the HalfKAv2_hm horizontal mirror when the mover's king (already in
/// the mover's frame) sits on files a–d. `king_sq_mover_frame` is the king
/// square AFTER the vertical flip — i.e. exactly the king the feature
/// transformer indexes with. Matches `orient_halfka` polarity.
#[inline]
pub fn canonical_orient(stm_is_black: bool, king_sq_mover_frame: Square) -> u8 {
    let vflip = if stm_is_black { 56u8 } else { 0u8 };
    let hflip = if king_sq_mover_frame.file() <= 3 {
        7u8
    } else {
        0u8
    };
    vflip ^ hflip
}

/// Map raw board squares into the canonical frame. `orient` comes from
/// [`canonical_orient`]; note the vertical component must already be folded
/// into the king used to compute it.
#[inline]
pub fn canonical_from_to(from: Square, to: Square, orient: u8) -> (u8, u8) {
    (from.index() ^ orient, to.index() ^ orient)
}

/// Complete-move class id in the canonical frame. `promotion` is `None` for
/// non-promotions AND queen promotions (queen promos ride the geometric
/// class, per the standard vocabulary); underpromotions get the tail block.
/// Returns `None` for square pairs outside the geometry (impossible for
/// legal chess moves — a `None` on a legal move indicates a frame bug and
/// callers should treat it as such).
pub fn policy_move_id(from_c: u8, to_c: u8, promotion: Option<PieceType>) -> Option<u16> {
    match promotion {
        Some(PieceType::Knight) | Some(PieceType::Bishop) | Some(PieceType::Rook) => {
            let pair = underpromo_pair_index(from_c, to_c)?;
            let piece_off = match promotion {
                Some(PieceType::Knight) => 0u16,
                Some(PieceType::Bishop) => 1,
                Some(PieceType::Rook) => 2,
                _ => unreachable!(),
            };
            Some(GEOMETRIC_CLASSES as u16 + pair * 3 + piece_off)
        }
        // None or queen promotion: the geometric class.
        _ => {
            let id = GEOM_TABLE[from_c as usize * 64 + to_c as usize];
            (id != POLICY_ID_NONE).then_some(id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn vocabulary_counts() {
        // Force table build + count check (the assert inside fires on build).
        // (a1,a1) is skipped by construction; the first real class is a1->b1.
        assert_eq!(GEOM_TABLE[1], 0);
        // 22 underpromo pairs × 3 pieces + 1792 = 1858.
        let mut pairs = 0;
        for f in 48..56u8 {
            for t in 56..64u8 {
                if underpromo_pair_index(f, t).is_some() {
                    pairs += 1;
                }
            }
        }
        assert_eq!(pairs, 22);
        assert_eq!(GEOMETRIC_CLASSES + 22 * 3, POLICY_VOCAB);
    }

    #[test]
    fn geometric_ids_are_a_bijection_onto_0_to_1791() {
        let mut seen = vec![false; GEOMETRIC_CLASSES];
        for from in 0..64usize {
            for to in 0..64usize {
                let id = GEOM_TABLE[from * 64 + to];
                if id != POLICY_ID_NONE {
                    assert!(!seen[id as usize], "duplicate id {id}");
                    seen[id as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every id in [0,1792) must be hit");
    }

    #[test]
    fn knight_and_castle_and_ep_geometries_are_covered() {
        // e1g1 castling = king two-step, queen geometry.
        let e1 = Square::E1.index();
        let g1 = Square::G1.index();
        assert!(policy_move_id(e1, g1, None).is_some());
        // Knight b1c3.
        assert!(policy_move_id(1, 18, None).is_some());
        // A non-geometric pair (a1 -> b4: df=1, dr=3) must be None.
        assert!(policy_move_id(0, 25, None).is_none());
    }

    #[test]
    fn underpromotions_index_the_tail_block() {
        // a7a8 = pair 0: knight/bishop/rook → 1792/1793/1794.
        assert_eq!(policy_move_id(48, 56, Some(PieceType::Knight)), Some(1792));
        assert_eq!(policy_move_id(48, 56, Some(PieceType::Bishop)), Some(1793));
        assert_eq!(policy_move_id(48, 56, Some(PieceType::Rook)), Some(1794));
        // Queen promo rides the geometric class.
        let q = policy_move_id(48, 56, Some(PieceType::Queen)).unwrap();
        assert!(usize::from(q) < GEOMETRIC_CLASSES);
        // The last underpromotion id is exactly POLICY_VOCAB - 1 (h7h8 rook).
        assert_eq!(
            policy_move_id(55, 63, Some(PieceType::Rook)),
            Some(POLICY_VOCAB as u16 - 1)
        );
    }

    #[test]
    fn every_legal_move_in_sample_positions_has_an_id() {
        use crate::Board;
        for fen in [
            crate::fen::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let stm_black = board.side_to_move() == crate::Color::Black;
            let vflip = if stm_black { 56u8 } else { 0 };
            let ksq_mover = Square::new(board.king_square(board.side_to_move()).index() ^ vflip);
            let orient = canonical_orient(stm_black, ksq_mover);
            let mut list = crate::moves::MoveList::new();
            crate::movegen::generate_legal(&board, &mut list);
            for &m in list.iter() {
                let (fc, tc) = canonical_from_to(m.from(), m.to(), orient);
                assert!(
                    policy_move_id(fc, tc, m.promotion_piece()).is_some(),
                    "legal move {m} in {fen} must map to a policy id (orient {orient})"
                );
            }
        }
    }
}
