//! The SOTA net's **SF-faithful feature front-end** (design/11 §12.1-12.2, D40):
//! the two feature-index functions (HalfKAv2_hm + FullThreats) and the
//! from-scratch accumulator refresh that fills the `pst`/`threat` pair the
//! Reckless-shaped [`super::sota_head`] consumes.
//!
//! This is a **reimplementation of the index math** in Stockfish's
//! `half_ka_v2_hm.{h,cpp}` and `full_threats.{h,cpp}` — the arithmetic and the
//! constant tables (OrientTBL, KingBuckets, PieceSquareIndex, the threat LUTs)
//! are transcribed, but no GPLv3 code is copied (D27): the loops, types, and
//! structure are our own idiomatic Rust. Correctness is pinned by the tests at
//! the bottom, which hand-compute the SF `make_index` for known tuples.
//!
//! Behind the `sota` cargo feature; the default build never compiles it, so the
//! depth-6 bench stays 548978.
//!
//! # What this increment covers
//! - [`halfka_index`]: the 22,528-dim HalfKAv2_hm index.
//! - [`threat_index`]: the 60,720-dim FullThreats index.
//! - [`refresh_pst`] / [`refresh_threat`] / [`refresh_both`]: from-scratch
//!   accumulator builds (format-agnostic — they take weight slices, like
//!   [`super::sota_head`]).
//!
//! # TODO — later #29 increments (deliberately NOT here)
//! - Incremental (dirty) HalfKA updates on the `MakeMoveHooks` seam (§3.4.4):
//!   ≤2 sub / ≤2 add per move from `DirtyPiece`.
//! - The HalfKA "Finny table" accumulator cache keyed by `(perspective,
//!   king-bucket, mirror)` (§12.2).
//! - The FullThreats dirty-diff observer (≤80 add/remove threat events per
//!   move, the x-ray ripple) — the one genuinely heavy new machine.
//! - The CENN-v2 net loader (§12.6) that fills real weights from disk.

use crate::attacks::{attacks, king_attacks, knight_attacks, pawn_attacks, pseudo_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, PieceType, Square};
use std::sync::LazyLock;

/// Accumulator width per perspective — must equal the SOTA head's `L1` (768) so
/// the refresh output feeds [`super::sota_head::SotaHead::forward`] directly.
pub const ACC: usize = super::sota_head::L1;
const _: () = assert!(ACC == 768, "refresh output width must match the head");

// ===========================================================================
// HalfKAv2_hm — 22,528 = 64 · PS_NB / 2   (half_ka_v2_hm.{h,cpp})
// ===========================================================================

/// One 64-wide slot per piece "kind": 10 non-king (W/B × P,N,B,R,Q) + one
/// **shared** king slot. `PS_NB = 11 · 64`.
const PS_NB: usize = 704;
/// Shared king piece-slot (`10 · 64`) — both colors' kings land here.
const PS_KING: usize = 640;
/// HalfKAv2_hm feature-space dimension (`64 · PS_NB / 2`).
pub const HALFKA_DIMS: usize = 22528;

/// King-bucket number for each square (SF `KingBuckets / PS_NB`), indexed by the
/// oriented king square (`ksq ^ 56·persp`). 32 distinct buckets; the table is
/// horizontally symmetric (each row is a palindrome), which is why the index
/// does **not** need a horizontal mirror — only the vertical `^56` flip.
/// Reading order is square index 0 = a1 … 63 = h8.
#[rustfmt::skip]
const KING_BUCKETS: [u16; 64] = [
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
];

/// HalfKA square orientation: a **file mirror** so the king ends up on files
/// e-h. `7` (XOR flips the file) for a king on files a-d, else `0`. This is the
/// polarity that normalizes the king to the kingside; FullThreats uses the
/// *opposite* polarity (see [`orient_threats`]) — the documented fidelity trap.
#[inline]
fn orient_halfka(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 7 } else { 0 }
}

/// Cache key for a HalfKAv2_hm king state: 32 bucket ids × two horizontal
/// orientations. Moving the king without changing this key leaves every
/// non-king feature's index stable, so a dirty-piece update is sufficient.
#[inline]
pub(crate) fn halfka_king_key(perspective: Color, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    bucket * 2 + usize::from(orient_halfka(king_sq) != 0)
}

/// The piece-slot offset within a 704-block (SF `PieceSquareIndex`).
///
/// From `persp`'s point of view own pieces occupy the "white" even slots
/// (`0,128,…,512`) and enemy pieces the "black" odd slots (`64,192,…,576`);
/// kings share [`PS_KING`]. So a black-perspective refresh swaps which color's
/// pieces land in the white slots — the §3.4.2 symmetry that makes the two
/// accumulators exchange roles under a color flip.
#[inline]
fn piece_square_index(persp: Color, piece: Piece) -> usize {
    if matches!(piece.piece_type, PieceType::King) {
        return PS_KING;
    }
    // pt ∈ 0..=4 for P,N,B,R,Q → white slot = pt·128, black slot = pt·128 + 64.
    let pt = piece.piece_type as usize;
    if piece.color == persp {
        pt * 128
    } else {
        pt * 128 + 64
    }
}

/// The HalfKAv2_hm feature index of `piece` on `sq`, as seen from `perspective`
/// with the own king on `king_sq` (design/11 §12.1).
///
/// `index = (sq ^ orient(ksq) ^ 56·persp) + piece_square_index + king_bucket`,
/// exactly SF's `make_index`. The oriented square (0-63) plus the piece slot
/// (a multiple of 64, ≤640) stays inside one `PS_NB`-block; the king bucket adds
/// `bucket·704`. Range: `[0, 22528)`.
#[inline]
pub fn halfka_index(perspective: Color, piece: Piece, sq: Square, king_sq: Square) -> usize {
    let flip = 56 * perspective as u8; // 0 or 56 (vertical mirror for black)
    let oriented_sq = (sq.index() ^ orient_halfka(king_sq) ^ flip) as usize;
    let bucket = KING_BUCKETS[(king_sq.index() ^ flip) as usize] as usize;
    oriented_sq + piece_square_index(perspective, piece) + bucket * PS_NB
}

// ===========================================================================
// FullThreats — 60,720   (full_threats.{h,cpp})
// ===========================================================================

/// FullThreats feature-space dimension.
pub const THREAT_DIMS: usize = 60720;

/// SF piece code: white P..K = 1..6, black P..K = 9..14 (color = bit 3). The
/// gap layout is deliberate — XOR-8 flips color, which is the trick FullThreats
/// uses to swap perspective (`attacker ^ 8·persp`).
#[inline]
const fn sf_code(piece: Piece) -> usize {
    ((piece.color as usize) << 3) | (piece.piece_type as usize + 1)
}

/// AllPieces in SF's enumeration order — this order fixes the attacker-block
/// bases (`cumulativeOffset`), so it must not be reordered.
const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];

/// `numValidTargets[code]` = 2 × (count of victim piece-types this attacker may
/// threaten). Pawn 6 (P/N/R), knight 10 (P/N/B/R/Q), bishop 8 (P/N/B/R),
/// rook 8, queen 10, king 0. Indexed by SF piece code (16-wide, gaps unused).
#[rustfmt::skip]
const NUM_VALID_TARGETS: [usize; 16] = [
    0, 6, 10, 8, 8, 10, 0, 0,
    0, 6, 10, 8, 8, 10, 0, 0,
];

/// `map[attacker_type-1][attacked_type-1]` (types PAWN..KING = 1..6): the dense
/// victim-relation ordinal for a legal attacker→victim type pair, or `-1` when
/// the pair is excluded. Kings attack nothing; pawns threaten only P/N/R;
/// bishops/rooks skip Q & K; knights/queens include Q.
#[rustfmt::skip]
const MAP: [[i8; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1], // pawn
    [ 0,  1,  2,  3,  4, -1], // knight
    [ 0,  1,  2,  3, -1, -1], // bishop
    [ 0,  1,  2,  3, -1, -1], // rook
    [ 0,  1,  2,  3,  4, -1], // queen
    [-1, -1, -1, -1, -1, -1], // king
];

/// FullThreats square orientation — the **opposite polarity** to HalfKA:
/// `0` for a king on files a-d, `7` for files e-h (design/11 §12.1, the
/// "a1/h1 polarity" fidelity trap: HalfKA's OrientTBL is a1→SQ_H1, threats' is
/// a1→SQ_A1).
#[inline]
fn orient_threats(ksq: Square) -> u8 {
    if ksq.file() <= 3 { 0 } else { 7 }
}

/// Empty-board attack geometry of an SF piece code on `from` — the same sets SF
/// builds its threat LUTs from. Pawns use *push-or-attacks* (single push |
/// both diagonal captures); everything else is the plain pseudo-attack set.
fn geom_attacks(code: usize, from: Square) -> Bitboard {
    let color = if code < 8 { Color::White } else { Color::Black };
    match code & 7 {
        1 => pawn_push_or_attacks(color, from),
        2 => knight_attacks(from),
        3 => pseudo_attacks(PieceType::Bishop, from),
        4 => pseudo_attacks(PieceType::Rook, from),
        5 => pseudo_attacks(PieceType::Queen, from),
        6 => king_attacks(from),
        _ => Bitboard::EMPTY,
    }
}

/// SF `PawnPushOrAttacks[c][from]` — the pawn's single push square unioned with
/// its two diagonal capture squares (all masked to the board). A pawn's "threat"
/// includes the pawn it is blocked by / about to push into, hence the push bit.
fn pawn_push_or_attacks(color: Color, from: Square) -> Bitboard {
    let mut bb = pawn_attacks(color, from);
    if let Some(push) = from.shifted(color.pawn_push()) {
        bb |= Bitboard::from_square(push);
    }
    bb
}

/// The three threat lookup tables, built once. Sizes: `lut2` 64 KiB, the rest a
/// few KiB — all indexed by SF piece code (16-wide, gap entries unused).
struct ThreatTables {
    /// `offsets[code][from]` = prefix-sum of empty-board attack counts over
    /// from-squares before `from` (the per-attacker from-square base).
    offsets: [[u32; 64]; 16],
    /// `piece_offset[code]` = total empty-board attack edges for the attacker
    /// (SF `cumulativePieceOffset`; the per-attacker geometry size). Folded into
    /// `lut1` for the index, but kept explicitly: the table-invariant test reads
    /// it, and the coming incremental threats observer (§12.2) needs it.
    #[allow(dead_code)]
    piece_offset: [u32; 16],
    /// `cumulative_offset[code]` = the attacker-block base (SF `cumulativeOffset`).
    /// Kept for the same reasons as `piece_offset`.
    #[allow(dead_code)]
    cumulative_offset: [u32; 16],
    /// `lut2[code][from][to]` = popcount of the attacker's attack bits strictly
    /// below `to` (the dense ordinal of `to` among `from`'s targets).
    lut2: Box<[[[u8; 64]; 64]; 16]>,
    /// `lut1[attacker][attacked][from<to]` = the block+sub-block feature base, or
    /// [`THREAT_DIMS`] (an out-of-range sentinel → dropped) for an excluded or
    /// semi-excluded pair. The `from<to` axis dedups symmetric same-type threats.
    lut1: [[[u32; 2]; 16]; 16],
}

fn build_threat_tables() -> ThreatTables {
    let mut offsets = [[0u32; 64]; 16];
    let mut piece_offset = [0u32; 16];
    let mut cumulative_offset = [0u32; 16];
    let mut lut2 = Box::new([[[0u8; 64]; 64]; 16]);

    // Per-attacker geometry: prefix-sum from-square bases + the `to`-ordinal LUT.
    let mut cumulative = 0u32;
    for &code in &ALL_PIECES {
        let is_pawn = code & 7 == 1;
        let mut cum_piece = 0u32;
        for from_i in 0..64u8 {
            offsets[code][from_i as usize] = cum_piece;
            let from = Square::new(from_i);
            let attacks = geom_attacks(code, from);
            // Fill the ordinal LUT for every from-square (SF builds it for all
            // 64, even rank 1/8 pawns that never occur in play).
            let bits = attacks.to_u64();
            for to_i in 0..64u8 {
                let below = bits & ((1u64 << to_i) - 1);
                lut2[code][from_i as usize][to_i as usize] = below.count_ones() as u8;
            }
            // Advance the prefix sum. Pawns only count from squares a2..h7
            // (a pawn never sits on rank 1 or 8), matching SF's guard.
            if !is_pawn || (8..=55).contains(&from_i) {
                cum_piece += attacks.count();
            }
        }
        piece_offset[code] = cum_piece;
        cumulative_offset[code] = cumulative;
        cumulative += NUM_VALID_TARGETS[code] as u32 * cum_piece;
    }
    debug_assert_eq!(
        cumulative, THREAT_DIMS as u32,
        "threat blocks must tile 60720"
    );

    // Per (attacker, attacked) feature base + the exclusion sentinels.
    let mut lut1 = [[[THREAT_DIMS as u32; 2]; 16]; 16];
    for &attacker in &ALL_PIECES {
        for &attacked in &ALL_PIECES {
            let same_type_diff_color = (attacker ^ attacked) == 8;
            let att_t = attacker & 7; // 1..=6
            let atd_t = attacked & 7; // 1..=6
            let map = MAP[att_t - 1][atd_t - 1];
            // Same-type threats are symmetric (A attacks B ⟺ B attacks A) for
            // every non-pawn and for opposite-color pawns, so one direction is
            // dropped to avoid double counting. Same-color pawn→pawn (a push)
            // is directional and kept.
            let semi_excluded = att_t == atd_t && (same_type_diff_color || att_t != 1);
            let excluded = map < 0;

            let feature = if excluded {
                THREAT_DIMS as u32
            } else {
                let atd_color = attacked >> 3; // 0 = white victim, 1 = black
                let num_types = NUM_VALID_TARGETS[attacker] / 2;
                let sub = atd_color * num_types + map as usize;
                cumulative_offset[attacker] + sub as u32 * piece_offset[attacker]
            };
            lut1[attacker][attacked][0] = if excluded {
                THREAT_DIMS as u32
            } else {
                feature
            };
            lut1[attacker][attacked][1] = if excluded || semi_excluded {
                THREAT_DIMS as u32
            } else {
                feature
            };
        }
    }

    ThreatTables {
        offsets,
        piece_offset,
        cumulative_offset,
        lut2,
        lut1,
    }
}

static THREAT_TABLES: LazyLock<ThreatTables> = LazyLock::new(build_threat_tables);

/// The FullThreats feature index of the directed relation "`attacker` on `from`
/// threatens `attacked` on `to`", as seen from `perspective` with own king on
/// `ksq` (design/11 §12.1). Returns `None` when the pair is excluded (kings, or
/// an illegal attacker→victim type) or semi-excluded (the dropped direction of a
/// symmetric same-type threat) — SF's `push_back_if_lt(index, Dimensions)`.
///
/// `index = lut1[att][atd][from<to] + offsets[att][from] + lut2[att][from][to]`,
/// all with the squares file/rank-oriented by `OrientTBL[ksq] ^ 56·persp` and
/// the piece codes color-swapped by `^ 8·persp`. Range when `Some`: `[0, 60720)`.
#[inline]
pub fn threat_index(
    perspective: Color,
    attacker: Piece,
    from: Square,
    to: Square,
    attacked: Piece,
    ksq: Square,
) -> Option<usize> {
    let t = &*THREAT_TABLES;
    let p = perspective as u8;
    let orientation = orient_threats(ksq) ^ (56 * p);
    let from_o = (from.index() ^ orientation) as usize;
    let to_o = (to.index() ^ orientation) as usize;
    let swap = 8 * p as usize; // 0 or 8: flips the piece color bit
    let att = sf_code(attacker) ^ swap;
    let atd = sf_code(attacked) ^ swap;
    let dir = (from_o < to_o) as usize;

    let base = t.lut1[att][atd][dir];
    if base >= THREAT_DIMS as u32 {
        return None; // excluded / semi-excluded
    }
    let idx = base + t.offsets[att][from_o] + t.lut2[att][from_o][to_o] as u32;
    Some(idx as usize)
}

/// Force the threat LUTs to build now (parity with `attacks::init`). Optional —
/// the first [`threat_index`] / refresh triggers the `LazyLock` anyway.
pub fn init() {
    LazyLock::force(&THREAT_TABLES);
}

// ===========================================================================
// Feature enumeration — the active features on a board, per perspective
// ===========================================================================

/// Call `f` with the HalfKA index of every piece on the board, from
/// `perspective`. Exactly one feature per piece (kings included), so the call
/// count equals the piece count.
pub fn for_each_halfka(board: &Board, perspective: Color, mut f: impl FnMut(usize)) {
    let ksq = board.king_square(perspective);
    for sq in board.all_occupancy() {
        let piece = board.piece_on(sq).expect("occupancy ⇒ a piece");
        f(halfka_index(perspective, piece, sq, ksq));
    }
}

/// Call `f` with the FullThreats index of every active directed threat on the
/// board, from `perspective`. Mirrors SF `append_active_indices`: every piece of
/// both colors is an attacker; victim sets are pre-filtered to the legal types;
/// excluded / semi-excluded relations are skipped ([`threat_index`] returns
/// `None`).
pub fn for_each_threat(board: &Board, perspective: Color, mut f: impl FnMut(usize)) {
    let ksq = board.king_square(perspective);
    let occ = board.all_occupancy();

    let bb = |c: Color, pt: PieceType| board.pieces(c, pt);
    let both = |pt: PieceType| bb(Color::White, pt) | bb(Color::Black, pt);
    let all_pawns = both(PieceType::Pawn);
    // Victim masks (both colors — a threat can land on a friendly piece too).
    let pawn_targets = both(PieceType::Pawn) | both(PieceType::Knight) | both(PieceType::Rook);
    let minor_slider_targets = pawn_targets | both(PieceType::Bishop);
    let queen_targets = minor_slider_targets | both(PieceType::Queen);

    let mut emit = |attacker: Piece, from: Square, to: Square, attacked: Piece| {
        if let Some(idx) = threat_index(perspective, attacker, from, to, attacked, ksq) {
            f(idx);
        }
    };

    for c in Color::ALL {
        // --- Pawns: two diagonal captures + a straight push-threat ---------
        let attacker = Piece::new(c, PieceType::Pawn);
        for from in board.pieces(c, PieceType::Pawn) {
            for to in pawn_attacks(c, from) & pawn_targets {
                let attacked = board.piece_on(to).expect("target mask ⇒ a piece");
                emit(attacker, from, to, attacked);
            }
            // Push threat: a pawn directly in front (own or enemy) is threatened.
            if let Some(to) = from.shifted(c.pawn_push())
                && all_pawns.contains(to)
            {
                let attacked = board.piece_on(to).expect("pawn in front");
                emit(attacker, from, to, attacked);
            }
        }

        // --- Knight, bishop, rook, queen -----------------------------------
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let attacker = Piece::new(c, pt);
            let targets = if matches!(pt, PieceType::Knight | PieceType::Queen) {
                queen_targets
            } else {
                minor_slider_targets
            };
            for from in board.pieces(c, pt) {
                for to in attacks(pt, from, occ) & targets {
                    let attacked = board.piece_on(to).expect("target mask ⇒ a piece");
                    emit(attacker, from, to, attacked);
                }
            }
        }
        // Kings are never attackers (whole row of `map` is -1), so they are
        // deliberately not looped.
    }
}

// ===========================================================================
// From-scratch accumulator refresh (format-agnostic weight slices)
// ===========================================================================

/// Add the `ACC`-wide column at feature `index` of a feature-major weight slice
/// into `acc`, using wrapping i16 math (the quantized accumulator wraps, exactly
/// like [`super::sota_head`]'s FT sum). `W` is `i16` for HalfKA, `i8` for threats.
#[inline]
fn add_column<W: Copy + Into<i16>>(acc: &mut [i16; ACC], weights: &[W], index: usize) {
    let col = &weights[index * ACC..index * ACC + ACC];
    for (a, &w) in acc.iter_mut().zip(col.iter()) {
        *a = a.wrapping_add(w.into());
    }
}

/// Build the HalfKA accumulator for `perspective` from scratch: seed with the FT
/// bias, then add the weight column of every active HalfKA feature (§12.2).
/// `halfka_weights` is feature-major (`[HALFKA_DIMS · ACC]` i16), `ft_bias` is
/// `ACC`-wide.
pub fn refresh_pst(
    halfka_weights: &[i16],
    ft_bias: &[i16],
    board: &Board,
    perspective: Color,
) -> [i16; ACC] {
    assert_eq!(
        halfka_weights.len(),
        HALFKA_DIMS * ACC,
        "HalfKA transformer has the wrong size"
    );
    assert_eq!(ft_bias.len(), ACC, "SOTA FT bias has the wrong size");
    let mut acc = [0i16; ACC];
    acc.copy_from_slice(&ft_bias[..ACC]);
    for_each_halfka(board, perspective, |idx| {
        add_column(&mut acc, halfka_weights, idx)
    });
    acc
}

/// Build the threats accumulator for `perspective` from scratch: seed with zero,
/// then add the weight column of every active threat feature (§12.2).
/// `threat_weights` is feature-major (`[THREAT_DIMS · ACC]` i8).
pub fn refresh_threat(threat_weights: &[i8], board: &Board, perspective: Color) -> [i16; ACC] {
    let mut acc = [0i16; ACC];
    if threat_weights.is_empty() {
        return acc;
    }
    assert_eq!(
        threat_weights.len(),
        THREAT_DIMS * ACC,
        "FullThreats transformer has the wrong size"
    );
    for_each_threat(board, perspective, |idx| {
        add_column(&mut acc, threat_weights, idx)
    });
    acc
}

/// Build both accumulators for both perspectives — the `(pst, threat)` pair
/// [`super::sota_head::SotaHead::forward`] wants (index 0 = White, 1 = Black).
pub fn refresh_both(
    halfka_weights: &[i16],
    ft_bias: &[i16],
    threat_weights: &[i8],
    board: &Board,
) -> ([[i16; ACC]; 2], [[i16; ACC]; 2]) {
    let mut pst = [[0i16; ACC]; 2];
    let mut threat = [[0i16; ACC]; 2];
    for (i, &persp) in Color::ALL.iter().enumerate() {
        pst[i] = refresh_pst(halfka_weights, ft_bias, board, persp);
        threat[i] = refresh_threat(threat_weights, board, persp);
    }
    (pst, threat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn wp() -> Piece {
        Piece::new(Color::White, PieceType::Pawn)
    }
    fn wk() -> Piece {
        Piece::new(Color::White, PieceType::King)
    }
    fn wn() -> Piece {
        Piece::new(Color::White, PieceType::Knight)
    }

    // -------- HalfKAv2_hm --------------------------------------------------

    /// The index reproduces SF `make_index` hand-computed from the formula for
    /// several (piece, sq, ksq, perspective) tuples.
    #[test]
    fn halfka_hand_computed() {
        // White king on e1 (file e ⇒ orient 0), the king itself as a feature.
        // (4^0^0) + PS_KING(640) + KingBuckets[4](=31)·704 = 4 + 640 + 21824.
        assert_eq!(
            halfka_index(Color::White, wk(), Square::E1, Square::E1),
            4 + 640 + 31 * 704
        );
        // White pawn on a1, white king on a1 (file a ⇒ orient 7).
        // (0^7^0) + 0 + KingBuckets[0](=28)·704 = 7 + 0 + 19712 = 19719.
        assert_eq!(
            halfka_index(Color::White, wp(), Square::A1, Square::A1),
            19719
        );
        // Black perspective: WHITE pawn on a2, BLACK king on e8.
        // orient(e8)=0, flip=56 ⇒ sq 8^56=48; enemy pawn ⇒ PS_B_PAWN(64);
        // KingBuckets[60^56 = 4](=31)·704. = 48 + 64 + 21824 = 21936.
        assert_eq!(
            halfka_index(Color::Black, wp(), Square::A2, Square::E8),
            21936
        );
    }

    /// Horizontal mirror symmetry: a queenside king setup and its kingside
    /// mirror map to the *same* index (the point of the file-mirror orient +
    /// palindromic KingBuckets).
    #[test]
    fn halfka_mirror_symmetry() {
        // King b1 + knight c1  vs  mirror King g1 + knight f1.
        let qs = halfka_index(Color::White, wn(), Square::C1, Square::B1);
        let ks = halfka_index(Color::White, wn(), Square::F1, Square::G1);
        assert_eq!(qs, ks, "queenside and kingside mirror to one index");
        // Sanity: not a degenerate all-equal function.
        assert_ne!(
            halfka_index(Color::White, wn(), Square::C1, Square::B1),
            halfka_index(Color::White, wn(), Square::D1, Square::B1),
        );
    }

    /// Every (perspective, piece, sq, ksq) yields an in-range index.
    #[test]
    fn halfka_indices_in_range() {
        for persp in Color::ALL {
            for color in Color::ALL {
                for pt in PieceType::ALL {
                    let piece = Piece::new(color, pt);
                    for sq in Square::all() {
                        for ksq in Square::all() {
                            assert!(halfka_index(persp, piece, sq, ksq) < HALFKA_DIMS);
                        }
                    }
                }
            }
        }
    }

    // -------- FullThreats --------------------------------------------------

    /// The empty-board geometry sizes match SF's documented per-attacker edge
    /// counts (P 132, N 336, B 560, R 896, Q 1456, K 420) and the blocks tile
    /// exactly 60,720.
    #[test]
    fn threats_geometry_and_tiling() {
        let t = &*THREAT_TABLES;
        // code → expected piece_offset, for both colors.
        for base in [0usize, 8] {
            assert_eq!(t.piece_offset[base + 1], 132, "pawn edges");
            assert_eq!(t.piece_offset[base + 2], 336, "knight edges");
            assert_eq!(t.piece_offset[base + 3], 560, "bishop edges");
            assert_eq!(t.piece_offset[base + 4], 896, "rook edges");
            assert_eq!(t.piece_offset[base + 5], 1456, "queen edges");
            assert_eq!(t.piece_offset[base + 6], 420, "king edges");
        }
        // Telescoping: last block base + its size == Dimensions.
        let last = 14; // B_QUEEN is the last with nonzero size; B_KING adds 0.
        let end = t.cumulative_offset[last] + NUM_VALID_TARGETS[last] as u32 * t.piece_offset[last];
        assert_eq!(end, THREAT_DIMS as u32);
        // No non-sentinel lut1 base can push an index out of range.
        for &att in &ALL_PIECES {
            for &atd in &ALL_PIECES {
                for &base in &t.lut1[att][atd] {
                    if base < THREAT_DIMS as u32 {
                        assert!(base + t.piece_offset[att] <= THREAT_DIMS as u32);
                    }
                }
            }
        }
    }

    /// A hand-computed threat index for one simple geometry:
    /// White knight b1 → black pawn c3, white king a1 (orient 0), white view.
    /// lut1[Nw][Pb][from<to] = 792 + (1·5 + 0)·336 = 2472;
    /// offsets[Nw][b1] = 2 (knight a1 has 2 edges); lut2[Nw][b1][c3] = 2.
    #[test]
    fn threats_hand_computed() {
        let idx = threat_index(
            Color::White,
            Piece::new(Color::White, PieceType::Knight),
            Square::B1,
            Square::C3,
            Piece::new(Color::Black, PieceType::Pawn),
            Square::A1,
        );
        assert_eq!(idx, Some(2472 + 2 + 2));
    }

    /// The map exclusions and the semi-exclusion dedup hold.
    #[test]
    fn threats_exclusions() {
        let ksq = Square::A1; // orient 0, no flip
        // King is never an attacker.
        assert_eq!(
            threat_index(
                Color::White,
                Piece::new(Color::White, PieceType::King),
                Square::E4,
                Square::E5,
                Piece::new(Color::Black, PieceType::Pawn),
                ksq,
            ),
            None
        );
        // Pawn → bishop is excluded (pawns only threaten P/N/R).
        assert_eq!(
            threat_index(
                Color::White,
                Piece::new(Color::White, PieceType::Pawn),
                Square::B2,
                Square::C3,
                Piece::new(Color::Black, PieceType::Bishop),
                ksq,
            ),
            None
        );
        // Same-type rook→rook: the `from < to` direction is dropped, the other
        // kept (symmetric-threat dedup).
        let rw = Piece::new(Color::White, PieceType::Rook);
        let rb = Piece::new(Color::Black, PieceType::Rook);
        assert_eq!(
            threat_index(Color::White, rw, Square::A1, Square::A8, rb, ksq),
            None,
            "from<to direction of a symmetric threat is semi-excluded"
        );
        let kept = threat_index(Color::White, rw, Square::A8, Square::A1, rb, ksq);
        assert!(
            kept.is_some() && kept.unwrap() < THREAT_DIMS,
            "other direction kept"
        );
    }

    /// On real positions, every active threat index is in range and — crucially
    /// — the active indices are all distinct (the index math is collision-free).
    #[test]
    fn threats_active_in_range_and_distinct() {
        let fens = [
            crate::fen::START_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            for persp in Color::ALL {
                let mut seen = HashSet::new();
                for_each_threat(&board, persp, |idx| {
                    assert!(idx < THREAT_DIMS, "{fen}: threat idx {idx} out of range");
                    assert!(seen.insert(idx), "{fen}: duplicate active threat idx {idx}");
                });
            }
        }
    }

    // -------- Refresh ------------------------------------------------------

    /// A tiny synthetic HalfKA weight set: all-ones columns, so an accumulator
    /// neuron equals `bias + (#active features)` — and #active == piece count.
    fn ones_halfka() -> (Vec<i16>, Vec<i16>) {
        let weights = vec![1i16; HALFKA_DIMS * ACC];
        let bias: Vec<i16> = (0..ACC).map(|j| (j % 5) as i16).collect();
        (weights, bias)
    }

    #[test]
    fn refresh_pst_counts_and_seeds_bias() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let (w, bias) = ones_halfka();
        let n = board.all_occupancy().count() as i16;
        for persp in Color::ALL {
            let acc = refresh_pst(&w, &bias, &board, persp);
            for (j, &v) in acc.iter().enumerate() {
                assert_eq!(v, bias[j] + n, "neuron {j}: bias + active-count");
            }
        }
        assert_eq!(n, 32, "startpos has 32 pieces");
    }

    /// Stride check: with all weights zero except the white king's own feature
    /// column set to a ramp `col[j] = j`, the white-perspective accumulator must
    /// be exactly `bias[j] + j` — proving the `index·ACC + j` layout, that the
    /// king is an active feature, and bias seeding, all at once. A transposed
    /// index would read a zero column and fail.
    #[test]
    fn refresh_pst_stride_and_king_feature() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let mut w = vec![0i16; HALFKA_DIMS * ACC];
        let bias: Vec<i16> = (0..ACC).map(|j| (j % 7) as i16).collect();

        let ksq = board.king_square(Color::White);
        let king_feat = halfka_index(Color::White, wk(), ksq, ksq);
        for j in 0..ACC {
            w[king_feat * ACC + j] = j as i16;
        }
        let acc = refresh_pst(&w, &bias, &board, Color::White);
        for (j, &v) in acc.iter().enumerate() {
            assert_eq!(v, bias[j] + j as i16, "neuron {j}");
        }
    }

    #[test]
    fn refresh_threat_counts() {
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let w = vec![1i8; THREAT_DIMS * ACC];
        for persp in Color::ALL {
            // Count active threats independently.
            let mut n = 0i16;
            for_each_threat(&board, persp, |_| n += 1);
            let acc = refresh_threat(&w, &board, persp);
            for &v in acc.iter() {
                assert_eq!(v, n, "every neuron == active threat count");
            }
            assert!(n > 0, "startpos has threats");
        }
    }

    /// Refresh is deterministic and both perspectives run without panic; the
    /// `refresh_both` pair agrees with the single-perspective builders.
    #[test]
    fn refresh_deterministic_and_consistent() {
        let board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
                .unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![2i8; THREAT_DIMS * ACC];

        let a = refresh_both(&hw, &bias, &tw, &board);
        let b = refresh_both(&hw, &bias, &tw, &board);
        assert_eq!(a, b, "deterministic");

        let (pst, threat) = &a;
        assert_eq!(pst[0], refresh_pst(&hw, &bias, &board, Color::White));
        assert_eq!(pst[1], refresh_pst(&hw, &bias, &board, Color::Black));
        assert_eq!(threat[0], refresh_threat(&tw, &board, Color::White));
        assert_eq!(threat[1], refresh_threat(&tw, &board, Color::Black));
    }

    /// End-to-end wiring: the refresh output feeds `SotaHead::forward` (types
    /// line up, runs, finite result). Trivial synthetic head — not a strength net.
    #[test]
    fn refresh_feeds_sota_head() {
        use super::super::sota_head::{BUCKETS, L1, L2, L3, SotaHead};
        let board = Board::from_fen(crate::fen::START_FEN).unwrap();
        let (hw, bias) = ones_halfka();
        let tw = vec![0i8; THREAT_DIMS * ACC];
        let (pst, threat) = refresh_both(&hw, &bias, &tw, &board);

        let head = SotaHead {
            l1_weights: vec![1i8; BUCKETS * L2 * L1].into_boxed_slice(),
            l1_biases: vec![0.0f32; BUCKETS * L2].into_boxed_slice(),
            l2_weights: vec![0.01f32; BUCKETS * L2 * L3].into_boxed_slice(),
            l2_biases: vec![0.0f32; BUCKETS * L3].into_boxed_slice(),
            l3_weights: vec![0.02f32; BUCKETS * L3].into_boxed_slice(),
            l3_biases: vec![0.0f32; BUCKETS].into_boxed_slice(),
            network_scale: 380,
        };
        let piece_count = board.all_occupancy().count() as usize;
        let cp = head.forward(&pst, &threat, Color::White, piece_count);
        assert!(cp.abs() < 1_000_000, "finite centipawns");
    }
}
