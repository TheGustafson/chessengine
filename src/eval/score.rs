//! `Score` and game-phase machinery for the tapered evaluation
//! (design/11 §3.2.1–3.2.2).
//!
//! Stockfish classic packs (mg, eg) into one `i32`; we deliberately don't —
//! the packing saves nothing measurable in a leaf loop this small and
//! invites overflow subtleties. A plain two-field struct with field-wise
//! operators is the idiomatic shape.

use crate::board::Board;
use crate::types::{Color, PieceType, Value};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A middlegame/endgame score pair, blended by [`taper`] at the end of the
/// evaluation. White-positive by convention while accumulating; the
/// side-to-move flip happens once, after tapering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

/// Shorthand constructor, mirroring the `S(mg, eg)` idiom of C engines:
/// `s(-27, -2)` keeps PSQT/param tables readable.
pub const fn s(mg: i32, eg: i32) -> Score {
    Score { mg, eg }
}

impl Add for Score {
    type Output = Score;

    fn add(self, rhs: Score) -> Score {
        s(self.mg + rhs.mg, self.eg + rhs.eg)
    }
}

impl AddAssign for Score {
    fn add_assign(&mut self, rhs: Score) {
        *self = *self + rhs;
    }
}

impl Sub for Score {
    type Output = Score;

    fn sub(self, rhs: Score) -> Score {
        s(self.mg - rhs.mg, self.eg - rhs.eg)
    }
}

impl SubAssign for Score {
    fn sub_assign(&mut self, rhs: Score) {
        *self = *self - rhs;
    }
}

impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        s(-self.mg, -self.eg)
    }
}

impl Mul<i32> for Score {
    type Output = Score;

    fn mul(self, rhs: i32) -> Score {
        s(self.mg * rhs, self.eg * rhs)
    }
}

/// PeSTO's per-piece game-phase increments, indexed by `PieceType as usize`:
/// P = 0, N = 1, B = 1, R = 2, Q = 4, K = 0 (design/11 §3.2.2).
pub const GAME_PHASE_INC: [i32; PieceType::COUNT] = [0, 1, 1, 2, 4, 0];

/// The start position's total phase (2 × (2N + 2B + 2R·2 + Q·4)); the
/// middlegame end of the [`taper`] blend. Sums above this (early
/// promotions) clamp down to it.
pub const GAME_PHASE_MAX: i32 = 24;

/// Game phase of `board`, summed over both colors and clamped to
/// [`GAME_PHASE_MAX`] (early promotions can push the raw sum above it).
pub fn game_phase(board: &Board) -> i32 {
    let mut p = 0;
    for color in Color::ALL {
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            p += GAME_PHASE_INC[pt as usize] * board.pieces(color, pt).count() as i32;
        }
    }
    p.min(GAME_PHASE_MAX)
}

/// Linear mg/eg interpolation at `phase` (already clamped to
/// `0..=GAME_PHASE_MAX`): full phase returns pure middlegame, zero phase
/// pure endgame. Integer division truncates toward zero, matching PeSTO.
pub const fn taper(score: Score, phase: i32) -> Value {
    debug_assert!(0 <= phase && phase <= GAME_PHASE_MAX);
    (score.mg * phase + score.eg * (GAME_PHASE_MAX - phase)) / GAME_PHASE_MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    #[test]
    fn score_ops_are_field_wise() {
        let a = s(3, -4);
        let b = s(-1, 10);
        assert_eq!(a + b, s(2, 6));
        assert_eq!(a - b, s(4, -14));
        assert_eq!(-a, s(-3, 4));
        assert_eq!(a * 2, s(6, -8));
        assert_eq!(a * -1, -a);
        assert_eq!(Score::default(), s(0, 0));

        let mut c = a;
        c += b;
        assert_eq!(c, a + b);
        c -= b;
        assert_eq!(c, a);
    }

    #[test]
    fn taper_boundaries_and_rounding() {
        let score = s(100, -60);
        // Full phase = pure middlegame; zero phase = pure endgame.
        assert_eq!(taper(score, GAME_PHASE_MAX), 100);
        assert_eq!(taper(score, 0), -60);
        // Exact midpoint of a divisible pair.
        assert_eq!(taper(s(24, 0), 12), 12);
        assert_eq!(taper(s(0, 24), 12), 12);
        // Integer division truncates toward zero on both signs.
        assert_eq!(taper(s(1, 0), 12), 0);
        assert_eq!(taper(s(-1, 0), 12), 0);
        // Tapering is linear: interior value sits between the endpoints.
        let mid = taper(score, 12);
        assert!((-60..=100).contains(&mid));
    }

    #[test]
    fn game_phase_anchors() {
        let startpos = Board::from_fen(START_FEN).unwrap();
        assert_eq!(game_phase(&startpos), GAME_PHASE_MAX);

        // Kings and pawns contribute nothing.
        let kings = Board::from_fen("k7/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        assert_eq!(game_phase(&kings), 0);
        let kp = Board::from_fen("k7/8/8/8/8/8/P7/K7 w - - 0 1").unwrap();
        assert_eq!(game_phase(&kp), 0);

        // Per-piece increments: N/B = 1, R = 2, Q = 4.
        let knvk = Board::from_fen("k7/8/8/8/8/8/8/KN6 w - - 0 1").unwrap();
        assert_eq!(game_phase(&knvk), 1);
        let krvk = Board::from_fen("k7/8/8/8/8/8/8/KR6 w - - 0 1").unwrap();
        assert_eq!(game_phase(&krvk), 2);
        let kqvk = Board::from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1").unwrap();
        assert_eq!(game_phase(&kqvk), 4);
    }

    #[test]
    fn game_phase_clamps_above_max() {
        // 6 queens + 2 rooks = 6*4 + 2*2 = 28 raw, clamped to 24
        // (the early-promotion case, design/11 §3.2.2).
        let board = Board::from_fen("2qqkqq1/8/8/8/8/8/8/QQR1K2R w - - 0 1").unwrap();
        assert_eq!(game_phase(&board), GAME_PHASE_MAX);
    }
}
