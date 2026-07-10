//! PeSTO piece-square tables and the Phase A inner evaluation
//! (design/11 §3.2.3–3.2.4).
//!
//! Provenance: *PeSTO's Evaluation Function* by Ronald Friederich (PeSTO /
//! rofChade), published on the Chess Programming Wiki
//! (chessprogramming.org/PeSTO%27s_Evaluation_Function). His texel-tuned
//! tables are the standard public starting point for tapered-PSQT engines;
//! we transcribe the **values**, not code (D27). Everything here is
//! tuner-fodder for Phase B.

use super::score::{GAME_PHASE_INC, GAME_PHASE_MAX, Score, game_phase, s, taper};
use crate::board::Board;
use crate::types::{Color, PieceType, Square, Value};

/// PeSTO piece values, indexed by `PieceType as usize` (P N B R Q K).
/// Baked into [`PSQT`] at compile time; kept `pub` as the Phase A
/// material anchor for tests and tools.
pub const MG_VALUE: [i32; PieceType::COUNT] = [82, 337, 365, 477, 1025, 0];
/// Endgame companion of [`MG_VALUE`].
pub const EG_VALUE: [i32; PieceType::COUNT] = [94, 281, 297, 512, 936, 0];

// The raw CPW tables, transcribed verbatim in the order published on the
// wiki: board-top-down, index 0 = a8, index 63 = h1. `build_psqt` flips
// them (`sq ^ 56`) into our little-endian a1 = 0 mapping (D3) — the
// transcription trap design/11 §3.2.3 warns about. The color-flip symmetry
// suite (tests/eval_symmetry.rs) is the tripwire for getting this wrong.

#[rustfmt::skip]
const MG_PESTO: [[i32; Square::COUNT]; PieceType::COUNT] = [
    // Pawn
    [
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    // Knight
    [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    // Bishop
    [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    // Rook
    [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    // Queen
    [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    // King
    [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
];

#[rustfmt::skip]
const EG_PESTO: [[i32; Square::COUNT]; PieceType::COUNT] = [
    // Pawn
    [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    // Knight
    [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    // Bishop
    [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    // Rook
    [
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
         7,  7,  7,  5,  4,  -3,  -5,  -3,
         4,  3, 13,  1,  2,   1,  -1,   2,
         3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    ],
    // Queen
    [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    // King
    [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
];

/// `PSQT[pt][sq]`: white-POV (mg, eg) for a piece of type `pt` on `sq` in
/// OUR a1 = 0 mapping, with the piece's material value baked in so the eval
/// loop is a single lookup-and-add. Black looks up its rank-mirrored square
/// (`Square::relative_to`).
pub static PSQT: [[Score; Square::COUNT]; PieceType::COUNT] = build_psqt();

/// Compile-time bake: vertical flip out of CPW order plus material
/// (design/11 §3.2.3). `while` loops because trait calls aren't const.
const fn build_psqt() -> [[Score; Square::COUNT]; PieceType::COUNT] {
    let mut out = [[s(0, 0); Square::COUNT]; PieceType::COUNT];
    let mut pt = 0;
    while pt < PieceType::COUNT {
        let mut sq = 0;
        while sq < Square::COUNT {
            let cpw = sq ^ 56; // CPW index 0 = a8; ours = a1.
            out[pt][sq] = s(
                MG_VALUE[pt] + MG_PESTO[pt][cpw],
                EG_VALUE[pt] + EG_PESTO[pt][cpw],
            );
            sq += 1;
        }
        pt += 1;
    }
    out
}

/// Phase A inner evaluation: material + PSQT, tapered by game phase,
/// returned from the side to move's perspective (design/11 §3.2.4).
///
/// Computed from scratch on every call — ~32 table adds is noise next to
/// make_move and movegen, and this function stays the test oracle if an
/// incremental version ever earns its keep (measured first, per the §3.2.4
/// escape hatch). No tempo bonus: PeSTO ships without one.
pub fn psqt_eval(board: &Board) -> Value {
    let mut total = Score::default(); // white POV while accumulating
    let mut phase = 0;
    for color in Color::ALL {
        let sign = match color {
            Color::White => 1,
            Color::Black => -1,
        };
        for pt in PieceType::ALL {
            for sq in board.pieces(color, pt) {
                // relative_to: White reads sq as-is, Black mirrors ranks.
                total += PSQT[pt as usize][sq.relative_to(color).index() as usize] * sign;
                phase += GAME_PHASE_INC[pt as usize];
            }
        }
    }
    let phase = phase.min(GAME_PHASE_MAX);
    debug_assert_eq!(phase, game_phase(board), "inline phase drifted");
    let v = taper(total, phase);
    match board.side_to_move() {
        Color::White => v,
        Color::Black => -v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    #[test]
    fn material_is_baked_into_tables() {
        // A PSQT entry is exactly value + table; spot-check via corner
        // squares whose CPW row/col are unambiguous. White pawn a2:
        // a2 = index 8, CPW index 8 ^ 56 = 48 (row 6, col 0) = -35 mg, 13 eg.
        let pawn_a2 = PSQT[PieceType::Pawn as usize][Square::A2.index() as usize];
        assert_eq!(pawn_a2, s(82 - 35, 94 + 13));
        // White knight a1: CPW index 56 (row 7, col 0) = -105 mg, -29 eg.
        let knight_a1 = PSQT[PieceType::Knight as usize][Square::A1.index() as usize];
        assert_eq!(knight_a1, s(337 - 105, 281 - 29));
    }

    #[test]
    fn startpos_is_zero() {
        assert_eq!(psqt_eval(&board(START_FEN)), 0);
    }

    // design/11 §4 step 2: the `sq ^ 56` flip is verified by a white piece
    // scoring exactly what the rank-mirrored black piece scores.
    #[test]
    fn white_pawn_e4_matches_black_pawn_e5() {
        let white = board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1");
        let black = board("4k3/8/8/4p3/8/8/8/4K3 b - - 0 1");
        // Both are side-to-move relative, so the values are EQUAL.
        assert_eq!(psqt_eval(&white), psqt_eval(&black));
    }

    // design/11 §4 step 2: three asymmetric positions against hand-computed
    // values (kings on e1/e8 mirror each other and cancel exactly).
    #[test]
    fn hand_computed_anchors() {
        // White pawn e4: mg 82+17, eg 94-7; phase 0 => pure eg = 87.
        assert_eq!(psqt_eval(&board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")), 87);

        // White knight f3: mg 337+17 = 354, eg 281-3 = 278; phase 1 =>
        // (354*1 + 278*23)/24 = 6748/24 = 281.
        assert_eq!(psqt_eval(&board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")), 281);

        // Black rook a8, white to move: mg -(477-19) = -458,
        // eg -(512-9) = -503; phase 2 => (-916 - 11066)/24 = -499.
        assert_eq!(psqt_eval(&board("r3k3/8/8/8/8/8/8/4K3 w - - 0 1")), -499);
    }

    #[test]
    fn side_to_move_perspective_negates() {
        // Same placement, opposite side to move: exact negation.
        let w = board("r3k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let b = board("r3k3/8/8/8/8/8/8/4K3 b - - 0 1");
        assert_eq!(psqt_eval(&w), -psqt_eval(&b));
    }
}
