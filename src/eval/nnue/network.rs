//! The quantized network: weights, biases, header metadata, and the feature
//! indexer (design/11 §3.4.1, §3.4.3).

use super::{INPUT, L1, QA, QB, SCALE};
use crate::attacks::Prng;
use crate::types::{Color, Piece, Square};

/// Output activation function (design/11 §3.4.1). The id is stored in the
/// CENN header (§3.4.5) so the same container can carry either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `clamp(x, 0, QA)`.
    CReLU,
    /// `clamp(x, 0, QA)²` — the default (stronger at equal size).
    SCReLU,
}

impl Activation {
    /// The header byte (design/11 §3.4.5): `0 = CReLU`, `1 = SCReLU`.
    pub const fn id(self) -> u8 {
        match self {
            Activation::CReLU => 0,
            Activation::SCReLU => 1,
        }
    }

    /// Parse a header byte; `None` for any unknown id.
    pub const fn from_id(id: u8) -> Option<Activation> {
        match id {
            0 => Some(Activation::CReLU),
            1 => Some(Activation::SCReLU),
            _ => None,
        }
    }
}

/// The feature index for a piece as seen from `perspective` (design/11
/// §3.4.1):
///
/// ```text
/// idx = ((c != p) as usize · 6 + pt) · 64 + rel_sq(p, sq)
/// ```
///
/// with `rel_sq(White, sq) = sq` and `rel_sq(Black, sq) = sq ^ 56` — i.e.
/// [`Square::relative_to`], the identity for White and a vertical mirror for
/// Black. The color bit is "is this piece the *opponent* from `perspective`'s
/// point of view", which is what makes the two accumulators swap roles under
/// a color flip (the §3.4.2 symmetry guarantee).
#[inline]
pub fn feature_index(perspective: Color, piece: Piece, sq: Square) -> usize {
    let opponent = (piece.color != perspective) as usize;
    let rel = sq.relative_to(perspective).index() as usize;
    (opponent * 6 + piece.piece_type as usize) * 64 + rel
}

/// A quantized `768 → 2×L1 → 1` network (design/11 §3.4.1, §3.4.3).
///
/// The weights are boxed (heap once at load — this is not the hot path; the
/// accumulator is). The feature-transformer weights are **feature-major**
/// (`ft_weights[feature · L1 + neuron]` = `W0[feature][neuron]`) so an
/// accumulator update touches one contiguous `L1`-wide row per feature
/// (design/11 §3.4.5). `qa`/`qb`/`scale` are carried from the header for
/// versioning; the loader pins them to [`QA`]/[`QB`]/[`SCALE`] for v1, and
/// the scalar forward pass uses the constants directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Network {
    /// `INPUT · L1` weights, feature-major: `W0[feature][neuron]`.
    pub ft_weights: Box<[i16]>,
    /// `L1` feature-transformer biases.
    pub ft_bias: Box<[i16]>,
    /// `2 · L1` output weights, **stm half first** (`[0, L1)` = side to move).
    pub out_weights: Box<[i16]>,
    /// The scalar output bias (at scale `QA · QB`).
    pub out_bias: i32,
    pub activation: Activation,
    pub qa: u16,
    pub qb: u16,
    pub scale: u16,
}

impl Network {
    /// The feature-transformer weight row for `feature` (an `L1`-wide slice),
    /// as a fixed-size array reference the accumulator kernels take. Safe:
    /// `try_into` on the exact-length subslice, no `unsafe`.
    #[inline]
    pub fn ft_row(&self, feature: usize) -> &[i16; L1] {
        let base = feature * L1;
        (&self.ft_weights[base..base + L1])
            .try_into()
            .expect("ft row is exactly L1 wide")
    }

    /// The feature-transformer bias as a fixed-size array reference.
    #[inline]
    pub fn ft_bias_arr(&self) -> &[i16; L1] {
        (&self.ft_bias[..])
            .try_into()
            .expect("ft_bias is exactly L1 wide")
    }

    /// The output weights as a fixed-size `2·L1` array reference.
    #[inline]
    pub fn out_weights_arr(&self) -> &[i16; 2 * L1] {
        (&self.out_weights[..])
            .try_into()
            .expect("out_weights is exactly 2·L1 wide")
    }

    /// A deterministic pseudo-random network for tests and the
    /// `nnue-testnet` dev command (design/11 step-1 forward-pass oracle).
    ///
    /// **This is a test/dev fixture, not a trained net** — weights are small
    /// so the forward pass exercises both SCReLU clamps (values that go
    /// negative *and* past `QA`) without any real overflow. Reproducible from
    /// `seed` alone (the crate's `xorshift64*`; no `rand` dep, D15), so the
    /// Rust `nnue-eval` and the Python oracle score byte-identical nets.
    pub fn deterministic_test_net(seed: u64, activation: Activation) -> Network {
        // xorshift64* forbids an all-zero seed; remap only 0 (a plain `| 1`
        // would collapse every even seed onto its odd neighbour).
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        let mut prng = Prng::new(seed);
        // Small signed weights in roughly [-100, 100]: with ~16–32 active
        // features an accumulator neuron lands in ~[-3000, 3000], so the
        // clamp to [0, QA] is meaningfully exercised on both ends.
        let mut small = || -> i16 { (prng.rand64() % 201) as i16 - 100 };

        let ft_weights: Box<[i16]> = (0..INPUT * L1).map(|_| small()).collect();
        let ft_bias: Box<[i16]> = (0..L1).map(|_| small()).collect();
        let out_weights: Box<[i16]> = (0..2 * L1).map(|_| small()).collect();
        // A modest bias at the QA·QB scale.
        let out_bias = (prng.rand64() % 20001) as i32 - 10000;

        // `small()` keeps `out_weights` in [-100, 100], well inside the §3.4.3
        // output-weight clip `|w1_q| ≤ round(1.98·QB) = 127`. That bound is the
        // exact condition under which the step-2 AVX2 forward pass
        // (`_mm256_mullo_epi16` + i32-lane `_mm256_madd_epi16`) is bit-identical
        // to the scalar i64 reference (reviewer B's F3; design/11 §3.4.3,
        // §3.4.6). Every net this fixture feeds — the accumulator fuzz, the
        // `verify_forward.sh` oracle (now AVX2-active in release), and the
        // bit-exactness test — therefore stays exact. Assert it so a future
        // widening of `small()` can't silently break that.
        debug_assert!(
            out_weights.iter().all(|&w| w.unsigned_abs() <= 127),
            "test-net out_weights must stay within the §3.4.3 clip (|w1| ≤ 127) \
             for AVX2/scalar bit-exactness"
        );

        Network {
            ft_weights,
            ft_bias,
            out_weights,
            out_bias,
            activation,
            qa: QA as u16,
            qb: QB as u16,
            scale: SCALE as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PieceType;

    // design/11 §3.4.1: the feature indexer against hand-computed values.
    #[test]
    fn feature_index_hand_computed() {
        let wp = Piece::new(Color::White, PieceType::Pawn);
        let bk = Piece::new(Color::Black, PieceType::King);
        let bp = Piece::new(Color::Black, PieceType::Pawn);

        // Own pawn on a1 from White's view: (0·6+0)·64 + 0 = 0.
        assert_eq!(feature_index(Color::White, wp, Square::A1), 0);
        // Enemy king on h8 from White's view: (1·6+5)·64 + 63 = 767 (max).
        assert_eq!(feature_index(Color::White, bk, Square::H8), 767);
        // Enemy (White) pawn on a1 from Black's view: opponent=1, pt=0,
        // rel = 0^56 = 56 → (1·6+0)·64 + 56 = 440.
        assert_eq!(feature_index(Color::Black, wp, Square::A1), 440);
        // Own (Black) king on h8 from Black's view: opponent=0, pt=5,
        // rel = 63^56 = 7 → (0·6+5)·64 + 7 = 327.
        assert_eq!(feature_index(Color::Black, bk, Square::H8), 327);

        // Every index is in range for every (perspective, piece, square).
        for persp in Color::ALL {
            for color in Color::ALL {
                for pt in PieceType::ALL {
                    let piece = Piece::new(color, pt);
                    for sq in Square::all() {
                        assert!(feature_index(persp, piece, sq) < INPUT);
                    }
                }
            }
        }

        // The color bit is perspective-relative, not absolute: a Black pawn
        // seen from Black is an "own" feature (bit 0); the SAME Black pawn
        // seen from White is an "enemy" feature (bit set).
        let a4 = Square::A4;
        assert!(feature_index(Color::Black, bp, a4) < 6 * 64);
        assert!(feature_index(Color::White, bp, a4) >= 6 * 64);
    }

    #[test]
    fn activation_id_roundtrip() {
        for a in [Activation::CReLU, Activation::SCReLU] {
            assert_eq!(Activation::from_id(a.id()), Some(a));
        }
        assert_eq!(Activation::from_id(2), None);
    }

    #[test]
    fn deterministic_net_is_reproducible_and_well_shaped() {
        let a = Network::deterministic_test_net(0xABCD_1234, Activation::SCReLU);
        let b = Network::deterministic_test_net(0xABCD_1234, Activation::SCReLU);
        assert_eq!(a, b, "same seed → identical net");
        assert_eq!(a.ft_weights.len(), INPUT * L1);
        assert_eq!(a.ft_bias.len(), L1);
        assert_eq!(a.out_weights.len(), 2 * L1);
        // Row/array views have the exact expected widths.
        assert_eq!(a.ft_row(INPUT - 1).len(), L1);
        assert_eq!(a.out_weights_arr().len(), 2 * L1);

        let c = Network::deterministic_test_net(0xABCD_1235, Activation::SCReLU);
        assert_ne!(a.ft_weights, c.ft_weights, "different seed → different net");
    }
}
