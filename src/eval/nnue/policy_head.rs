//! D47 deployed policy head — CPL1 sidecar loader + selected-row scorer.
//!
//! `score(m) = rows[move_id] · h256 + bias[move_id]` over the 1,858-class
//! vocabulary ([`policy_vocab`](super::policy_vocab)), evaluated for LEGAL
//! moves only (the 62×-cheaper selected-row form — never the dense head).
//! `h256` is the shared bottleneck activation (the [`SotaHead`] L2 output) of
//! a FRESH evaluation of the same position.
//!
//! Frame safety by construction: the engine iterates legal moves in BOARD
//! frame and maps each *into* the canonical frame for its row id; ids are
//! never decoded back into moves, so no inverse-mirror path exists (this
//! retires convention-trap watchlist #6 structurally).
//!
//! The sidecar loads automatically when `<EvalFile stem>.cpol` sits beside
//! the CENN; absent file ⇒ `None` ⇒ ordering falls back to classical — the
//! engine without a .cpol is bit-identical to pre-D47 behavior.

use super::policy_vocab::{self, POLICY_VOCAB};
use super::sota_head::L2;
use crate::Board;
use crate::movegen::generate_legal;
use crate::moves::{Move, MoveList};
use crate::types::{Color, Square};

/// CPL1 header: magic + u32 vocab + u32 dim.
const CPL1_HEADER: usize = 12;

/// The loaded policy head. Rows are int8, row-major `[POLICY_VOCAB][L2]`
/// in canonical-frame move-id order (netpack writes them straight from the
/// trainer's transposed quant tensor); biases are f32.
#[derive(Debug)]
pub struct PolicyHead {
    rows: Box<[i8]>,
    bias: Box<[f32]>,
}

impl PolicyHead {
    /// Parse a CPL1 sidecar. Errors are strings for the caller to log —
    /// a bad sidecar must never abort the engine (fall back to classical).
    pub fn from_bytes(bytes: &[u8]) -> Result<PolicyHead, String> {
        if bytes.len() < CPL1_HEADER || &bytes[0..4] != b"CPL1" {
            return Err("CPL1: bad magic".into());
        }
        let vocab = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if vocab != POLICY_VOCAB || dim != L2 {
            return Err(format!(
                "CPL1: vocab {vocab} x dim {dim}, this build wants {POLICY_VOCAB} x {L2}"
            ));
        }
        let need = CPL1_HEADER + vocab * dim + vocab * 4;
        if bytes.len() != need {
            return Err(format!("CPL1: {} bytes, expected {need}", bytes.len()));
        }
        let rows: Box<[i8]> = bytes[CPL1_HEADER..CPL1_HEADER + vocab * dim]
            .iter()
            .map(|&b| b as i8)
            .collect();
        let bias_off = CPL1_HEADER + vocab * dim;
        let bias: Box<[f32]> = bytes[bias_off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(PolicyHead { rows, bias })
    }

    /// Load the sidecar sitting beside an EvalFile path (`x.cenn` → `x.cpol`).
    /// `None` (with a stderr note on real read/parse errors) when absent.
    pub fn sibling_of(eval_path: &std::path::Path) -> Option<PolicyHead> {
        let path = eval_path.with_extension("cpol");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => {
                eprintln!("policy: cannot read {path:?}: {error}");
                return None;
            }
        };
        match PolicyHead::from_bytes(&bytes) {
            Ok(head) => Some(head),
            Err(error) => {
                eprintln!("policy: {path:?}: {error} — classical ordering");
                None
            }
        }
    }

    /// One selected row: `rows[id] · h + bias[id]`.
    #[inline]
    fn row_score(&self, id: usize, h: &[f32; L2]) -> f32 {
        let row = &self.rows[id * L2..id * L2 + L2];
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(h.iter()) {
            acc += f32::from(*w) * x;
        }
        acc + self.bias[id]
    }

    /// The policy-argmax LEGAL move for `board`, scored against the fresh
    /// shared-bottleneck activation `h256` of the SAME position. Returns the
    /// move in board frame (no inverse mapping exists by design). `None` on
    /// no legal moves or if every legal move falls outside the vocabulary
    /// (impossible for chess; defensive).
    pub fn best_move(&self, board: &Board, h256: &[f32; L2]) -> Option<Move> {
        let stm_black = board.side_to_move() == Color::Black;
        let vflip = if stm_black { 56u8 } else { 0 };
        let ksq_mover = Square::new(board.king_square(board.side_to_move()).index() ^ vflip);
        let orient = policy_vocab::canonical_orient(stm_black, ksq_mover);

        let mut list = MoveList::new();
        generate_legal(board, &mut list);
        let mut best: Option<(f32, Move)> = None;
        for &m in list.iter() {
            let (fc, tc) = policy_vocab::canonical_from_to(m.from(), m.to(), orient);
            let Some(id) = policy_vocab::policy_move_id(fc, tc, m.promotion_piece()) else {
                continue;
            };
            let s = self.row_score(id as usize, h256);
            if best.is_none_or(|(b, _)| s > b) {
                best = Some((s, m));
            }
        }
        best.map(|(_, m)| m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    fn synth_head(seed: u8) -> PolicyHead {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"CPL1");
        bytes.extend_from_slice(&(POLICY_VOCAB as u32).to_le_bytes());
        bytes.extend_from_slice(&(L2 as u32).to_le_bytes());
        // deterministic pseudo-random rows
        let mut x = seed as u32 | 1;
        for _ in 0..POLICY_VOCAB * L2 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            bytes.push((x >> 16) as u8);
        }
        for _ in 0..POLICY_VOCAB {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        PolicyHead::from_bytes(&bytes).expect("synthetic head parses")
    }

    #[test]
    fn cpl1_rejects_malformed() {
        assert!(PolicyHead::from_bytes(b"XXXX").is_err());
        let mut short = Vec::new();
        short.extend_from_slice(b"CPL1");
        short.extend_from_slice(&(POLICY_VOCAB as u32).to_le_bytes());
        short.extend_from_slice(&(L2 as u32).to_le_bytes());
        assert!(PolicyHead::from_bytes(&short).is_err(), "truncated body");
    }

    #[test]
    fn best_move_is_legal_everywhere() {
        let head = synth_head(7);
        let h = [0.31f32; L2];
        for fen in [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let best = head.best_move(&board, &h).expect("legal moves exist");
            assert!(board.legal(best), "policy-best must be legal in {fen}");
        }
    }

    #[test]
    fn different_activations_change_the_argmax_somewhere() {
        // Sanity that scores actually depend on h (not just bias/row norms):
        // across a few activations at least one position flips its argmax.
        let head = synth_head(7);
        let board = Board::from_fen(START_FEN).unwrap();
        let a = head.best_move(&board, &[0.9f32; L2]).unwrap();
        let mut h = [0.9f32; L2];
        for i in 0..L2 {
            if i % 2 == 0 {
                h[i] = -0.9;
            }
        }
        let b = head.best_move(&board, &h).unwrap();
        // Not guaranteed to differ for every seed, but for this fixed seed it
        // does — the test pins that the dot product is wired to h at all.
        assert_ne!(a, b, "argmax insensitive to the activation");
    }
}
