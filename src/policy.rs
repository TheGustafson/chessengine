//! Optional D40 policy head used only for quiet-move ordering.
//!
//! `CPOL` v2 companions contain either a rank-K bilinear from/to projection or
//! a full 64x64 move-logit projection.  Both consume the exact quantized shared
//! D40 trunk used by eval; the companion intentionally does not duplicate the
//! 80MB feature transformers.  Experiments may instead set `POLICY_TRUNK` to
//! keep a classic evaluator active while loading a separate D40 policy trunk.

use crate::board::Board;
use crate::eval::nnue::{
    Activation, L1, QA, active_sota_net,
    sota_features::refresh_both,
    sota_head,
    sota_loader::{self, SotaNetwork},
};
use crate::types::{Color, Square};
use std::sync::{Arc, OnceLock};

const MAGIC: &[u8; 4] = b"CPOL";
const VERSION: u16 = 2;
const TRUNK: usize = 768;
const HEADER: usize = 32;

#[derive(Debug)]
enum Head {
    Bilinear {
        rank: usize,
        canonical_layout: bool,
        from_w: Vec<f32>,
        from_b: Vec<f32>,
        to_w: Vec<f32>,
        to_b: Vec<f32>,
    },
    Full {
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    ClassicBilinear {
        rank: usize,
        from_w: Vec<f32>,
        from_b: Vec<f32>,
        to_w: Vec<f32>,
        to_b: Vec<f32>,
    },
    ClassicFactorized {
        rank: usize,
        position_w: Vec<f32>,
        position_b: Vec<f32>,
        move_w: Arc<[f32]>,
        move_b: Arc<[f32]>,
    },
}

#[derive(Debug)]
pub struct PolicyNet {
    head: Head,
    trunk: Option<Arc<SotaNetwork>>,
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("checked header"),
    )
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("checked header"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("checked header"),
    )
}
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}

impl PolicyNet {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let invalid = |message: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, message);
        if bytes.len() < HEADER || &bytes[..4] != MAGIC || u16_at(&bytes, 4) != VERSION {
            return Err(invalid("not a CPOL-v2 policy companion"));
        }
        let kind = bytes[6];
        let layout = bytes[7];
        let trunk = u32_at(&bytes, 8) as usize;
        let rank = u32_at(&bytes, 12) as usize;
        let payload_len = u64_at(&bytes, 16) as usize;
        let checksum = u64_at(&bytes, 24);
        if bytes.len() != HEADER + payload_len {
            return Err(invalid("CPOL payload length does not match its header"));
        }
        let payload = &bytes[HEADER..];
        if fnv1a64(payload) != checksum || payload.len() % 4 != 0 {
            return Err(invalid("CPOL checksum or float alignment is invalid"));
        }
        let values = floats(payload);
        let head = match kind {
            1 if rank > 0 && trunk == TRUNK => {
                if layout > 1 {
                    return Err(invalid("CPOL bilinear layout flag is invalid"));
                }
                let projection = 64 * rank;
                let matrix = projection * TRUNK;
                let expected = 2 * (matrix + projection);
                if values.len() != expected {
                    return Err(invalid("CPOL bilinear payload has the wrong size"));
                }
                let mut at = 0;
                let from_w = values[at..at + matrix].to_vec();
                at += matrix;
                let from_b = values[at..at + projection].to_vec();
                at += projection;
                let to_w = values[at..at + matrix].to_vec();
                at += matrix;
                let to_b = values[at..at + projection].to_vec();
                Head::Bilinear {
                    rank,
                    canonical_layout: layout == 1,
                    from_w,
                    from_b,
                    to_w,
                    to_b,
                }
            }
            2 if rank == 0 && layout == 0 => {
                if trunk != TRUNK {
                    return Err(invalid("CPOL full head requires a D40 trunk"));
                }
                let matrix = 4096 * TRUNK;
                if values.len() != matrix + 4096 {
                    return Err(invalid("CPOL full payload has the wrong size"));
                }
                Head::Full {
                    weights: values[..matrix].to_vec(),
                    biases: values[matrix..].to_vec(),
                }
            }
            3 if rank > 0 && layout == 1 && trunk == 2 * L1 => {
                let projection = 64 * rank;
                let matrix = projection * trunk;
                let expected = 2 * (matrix + projection);
                if values.len() != expected {
                    return Err(invalid("CPOL classic payload has the wrong size"));
                }
                let mut at = 0;
                let from_w = values[at..at + matrix].to_vec();
                at += matrix;
                let from_b = values[at..at + projection].to_vec();
                at += projection;
                let to_w = values[at..at + matrix].to_vec();
                at += matrix;
                let to_b = values[at..at + projection].to_vec();
                Head::ClassicBilinear {
                    rank,
                    from_w,
                    from_b,
                    to_w,
                    to_b,
                }
            }
            4 if rank > 0 && rank <= 64 && layout == 0 && trunk == 2 * L1 => {
                let position_matrix = rank * trunk;
                let move_matrix = 4096 * rank;
                let expected = position_matrix + rank + move_matrix + 4096;
                if values.len() != expected {
                    return Err(invalid(
                        "CPOL classic-factorized payload has the wrong size",
                    ));
                }
                let mut at = 0;
                let position_w = values[at..at + position_matrix].to_vec();
                at += position_matrix;
                let position_b = values[at..at + rank].to_vec();
                at += rank;
                let move_w = Arc::from(values[at..at + move_matrix].to_vec());
                at += move_matrix;
                let move_b = Arc::from(values[at..at + 4096].to_vec());
                Head::ClassicFactorized {
                    rank,
                    position_w,
                    position_b,
                    move_w,
                    move_b,
                }
            }
            _ => return Err(invalid("unknown CPOL head kind")),
        };
        Ok(Self { head, trunk: None })
    }

    /// Attach an independent D40 trunk used only by policy inference. This
    /// permits clean attribution games where the evaluator remains a classic
    /// net such as net-0. Without an attached trunk, policy uses the active D40
    /// `EvalFile` exactly as generation-0 experiments did.
    pub fn with_trunk(mut self, path: &str) -> std::io::Result<Self> {
        if matches!(
            self.head,
            Head::ClassicBilinear { .. } | Head::ClassicFactorized { .. }
        ) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "classic policy reuses EvalFile and does not accept POLICY_TRUNK",
            ));
        }
        let trunk = sota_loader::load_v2(path).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid POLICY_TRUNK: {error}"),
            )
        })?;
        self.trunk = Some(Arc::new(trunk));
        Ok(self)
    }

    /// Compute one node's policy scores. This correctness-first experiment
    /// refreshes the shared trunk from the active eval net; campaign benchmarks
    /// make its cost visible before any incremental optimization is attempted.
    pub fn scores(&self, board: &Board) -> Option<PolicyScores> {
        if matches!(
            self.head,
            Head::ClassicBilinear { .. } | Head::ClassicFactorized { .. }
        ) {
            return None;
        }
        let eval: &SotaNetwork = match self.trunk.as_deref() {
            Some(trunk) => trunk,
            None => active_sota_net()?,
        };
        let (pst, threat) = refresh_both(
            &eval.halfka_weights,
            &eval.ft_bias,
            &eval.threat_weights,
            board,
        );
        // The HalfKA accumulator is in the i8 transformer's reduced frame
        // (perf study 32), so the trunk needs the same per-column scale the
        // eval merge uses — sharing `activate_ft` is what stops the two
        // consumers drifting, and this keeps that true.
        let trunk = sota_head::policy_trunk(
            &pst,
            &threat,
            eval.head.ft_scale.block(0),
            board.side_to_move(),
        );
        let values = match &self.head {
            Head::Bilinear {
                rank,
                canonical_layout,
                from_w,
                from_b,
                to_w,
                to_b,
            } => bilinear_logits(&trunk, *rank, *canonical_layout, from_w, from_b, to_w, to_b),
            Head::Full { weights, biases } => affine(weights, biases, &trunk),
            Head::ClassicBilinear { .. } | Head::ClassicFactorized { .. } => {
                unreachable!("returned above")
            }
        };
        Some(PolicyScores {
            values: ScoreValues::Dense(values),
            stm: board.side_to_move(),
        })
    }

    pub fn uses_classic_accumulator(&self) -> bool {
        matches!(
            self.head,
            Head::ClassicBilinear { .. } | Head::ClassicFactorized { .. }
        )
    }

    pub fn scores_from_classic(
        &self,
        stm_acc: &[i16; L1],
        ntm_acc: &[i16; L1],
        activation: Activation,
        stm: Color,
    ) -> Option<PolicyScores> {
        let mut input = Vec::with_capacity(2 * L1);
        for acc in [stm_acc, ntm_acc] {
            input.extend(acc.iter().map(|&value| {
                let clipped = f32::from(value.clamp(0, QA as i16)) / QA as f32;
                match activation {
                    Activation::CReLU => clipped,
                    Activation::SCReLU => clipped * clipped,
                }
            }));
        }
        let values = match &self.head {
            Head::ClassicBilinear {
                rank,
                from_w,
                from_b,
                to_w,
                to_b,
            } => ScoreValues::Dense(bilinear_logits(
                &input, *rank, true, from_w, from_b, to_w, to_b,
            )),
            Head::ClassicFactorized {
                rank,
                position_w,
                position_b,
                move_w,
                move_b,
            } => {
                let mut position = affine(position_w, position_b, &input);
                for value in &mut position {
                    *value = value.clamp(0.0, 1.0).powi(2);
                }
                ScoreValues::Factorized {
                    rank: *rank,
                    position,
                    move_w: move_w.clone(),
                    move_b: move_b.clone(),
                }
            }
            _ => return None,
        };
        Some(PolicyScores { values, stm })
    }
}

fn affine(weights: &[f32], biases: &[f32], input: &[f32]) -> Vec<f32> {
    let width = input.len();
    biases
        .iter()
        .enumerate()
        .map(|(row, &bias)| {
            weights[row * width..(row + 1) * width]
                .iter()
                .zip(input)
                .fold(bias, |sum, (&weight, &value)| sum + weight * value)
        })
        .collect()
}

fn bilinear_logits(
    input: &[f32],
    rank: usize,
    canonical_layout: bool,
    from_w: &[f32],
    from_b: &[f32],
    to_w: &[f32],
    to_b: &[f32],
) -> Vec<f32> {
    let from = affine(from_w, from_b, input);
    let to = affine(to_w, to_b, input);
    let mut logits = vec![0.0; 4096];
    for f in 0..64 {
        for t in 0..64 {
            let mut score = 0.0;
            for k in 0..rank {
                score += if canonical_layout {
                    from[k * 64 + f] * to[t * rank + k]
                } else {
                    from[k * 64 + t] * to[f * rank + k]
                };
            }
            logits[f * 64 + t] = score;
        }
    }
    logits
}

enum ScoreValues {
    Dense(Vec<f32>),
    Factorized {
        rank: usize,
        position: Vec<f32>,
        move_w: Arc<[f32]>,
        move_b: Arc<[f32]>,
    },
}

pub struct PolicyScores {
    values: ScoreValues,
    stm: Color,
}
impl PolicyScores {
    pub fn move_score(&self, from: Square, to: Square) -> f32 {
        let from = from.relative_to(self.stm).index() as usize;
        let to = to.relative_to(self.stm).index() as usize;
        let index = from * 64 + to;
        match &self.values {
            ScoreValues::Dense(values) => values[index],
            ScoreValues::Factorized {
                rank,
                position,
                move_w,
                move_b,
            } => move_w[index * rank..(index + 1) * rank]
                .iter()
                .zip(position)
                .fold(move_b[index], |sum, (&weight, &value)| sum + weight * value),
        }
    }
}

static SHARED: OnceLock<Option<Arc<PolicyNet>>> = OnceLock::new();
static WEIGHT: OnceLock<i32> = OnceLock::new();
static MIN_DEPTH: OnceLock<i32> = OnceLock::new();
static MAX_PLY: OnceLock<usize> = OnceLock::new();
static ROOT_ORDER: OnceLock<bool> = OnceLock::new();
static CAPTURES: OnceLock<bool> = OnceLock::new();
static CAPTURE_WEIGHT: OnceLock<i32> = OnceLock::new();
pub fn shared_from_env() -> Option<Arc<PolicyNet>> {
    SHARED
        .get_or_init(|| {
            let path = std::env::var("POLICY_NET").ok()?;
            let mut policy = match PolicyNet::load(&path) {
                Ok(policy) => policy,
                Err(error) => {
                    eprintln!("info string failed to load POLICY_NET {path}: {error}");
                    return None;
                }
            };
            if let Ok(trunk) = std::env::var("POLICY_TRUNK") {
                policy = match policy.with_trunk(&trunk) {
                    Ok(policy) => policy,
                    Err(error) => {
                        eprintln!("info string failed to load POLICY_TRUNK {trunk}: {error}");
                        return None;
                    }
                };
            }
            Some(Arc::new(policy))
        })
        .clone()
}
pub fn quiet_weight() -> i32 {
    *WEIGHT.get_or_init(|| {
        std::env::var("POLICY_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000)
    })
}

/// Minimum remaining search depth at which the expensive policy forward is
/// allowed. Generation-0 behavior is the default (all main-search depths),
/// while real-time experiments can focus policy on the much smaller high-depth
/// tree with `POLICY_MIN_DEPTH`.
pub fn min_depth() -> i32 {
    *MIN_DEPTH.get_or_init(|| {
        std::env::var("POLICY_MIN_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
    })
}

/// Deepest ply (root = 0) at which policy inference is allowed. By default the
/// gate is unbounded, preserving generation-0 behavior. A small value focuses
/// the expensive independent-trunk refresh near the root, where ordering has
/// the greatest chance to reduce the downstream tree.
pub fn max_ply() -> usize {
    *MAX_PLY.get_or_init(|| {
        std::env::var("POLICY_MAX_PLY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

#[inline]
pub fn enabled_at(depth: i32, ply: usize) -> bool {
    depth >= min_depth() && ply <= max_ply()
}

/// Opt-in cached policy ordering for the root move list. This performs one
/// forward per `go`, independently of the interior MovePicker gates.
pub fn root_order_enabled() -> bool {
    *ROOT_ORDER.get_or_init(|| {
        std::env::var("POLICY_ROOT")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
    })
}

/// Whether policy also participates in capture ordering. Kept opt-in so the
/// quiet-only generation-0 result remains a controlled baseline.
pub fn captures_enabled() -> bool {
    *CAPTURES.get_or_init(|| {
        std::env::var("POLICY_CAPTURES")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
    })
}

pub fn capture_weight() -> i32 {
    *CAPTURE_WEIGHT.get_or_init(|| {
        std::env::var("POLICY_CAPTURE_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(quiet_weight)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_fixture() {
        assert_eq!(fnv1a64(b"CPOL"), 0x3605_a29b_f843_a485);
    }

    #[test]
    fn bilinear_projection_uses_bullet_column_major_reshapes() {
        let rank = 2;
        let from_square = 5;
        let to_square = 7;
        let mut from = [0.0; 128];
        let mut to = [0.0; 128];
        from[from_square] = 2.0;
        from[64 + from_square] = 3.0;
        to[to_square * rank] = 7.0;
        to[to_square * rank + 1] = 11.0;
        let score = (0..rank)
            .map(|k| from[k * 64 + from_square] * to[to_square * rank + k])
            .sum::<f32>();
        assert_eq!(score, 47.0);
    }

    #[test]
    fn factorized_scores_only_the_requested_canonical_move() {
        let rank = 2;
        let index = Square::E2.index() as usize * 64 + Square::E4.index() as usize;
        let mut weights = vec![0.0; 4096 * rank];
        weights[index * rank] = 2.0;
        weights[index * rank + 1] = 3.0;
        let mut biases = vec![0.0; 4096];
        biases[index] = 5.0;
        let scores = PolicyScores {
            values: ScoreValues::Factorized {
                rank,
                position: vec![7.0, 11.0],
                move_w: Arc::from(weights),
                move_b: Arc::from(biases),
            },
            stm: Color::White,
        };
        assert_eq!(scores.move_score(Square::E2, Square::E4), 52.0);
        assert_eq!(scores.move_score(Square::D2, Square::D4), 0.0);
    }
}
