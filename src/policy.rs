//! Experimental learned **policy** head for move ordering (design/17 Thrust II).
//!
//! A rank-`K` **bilinear** from×to move-prior over the SHARED NNUE stm feature
//! encoding ([`feature_index`]): `768 → trunk (SCReLU) → {from_emb[64×K],
//! to_emb[64×K]}`, and a move's prior is `from_emb[from] · to_emb[to]`. Trained
//! offline to distill Lc0's MCTS policy; used only to bias MovePicker quiet
//! ordering (design/17 §3.4.1) — it can never affect legality or eval.
//!
//! **Why bilinear, rank-8.** The full `768→4096` from×to head is the strongest
//! move predictor but far too expensive per node; a plain from+to *additive*
//! factorization is cheap but throws away the from-to correlation (recovers only
//! ~44% of the ordering benefit). A low-rank bilinear head recovers ~83% of it
//! at ~8× less cost, and plateaus at `K=8` (offline ablation, 2026-07-13).
//!
//! **Shared trunk.** The input features are exactly the eval accumulator's
//! stm-perspective features (`feature_index(side_to_move, piece, sq)`), so a
//! shipped version reads the eval accumulator instead of recomputing the trunk
//! (design/17 §3.2's "free trunk read"). This first cut recomputes a small
//! dedicated trunk to keep the experiment self-contained.
//!
//! Behind the `policy` cargo feature — the default build and the depth-6 bench
//! are byte-identical (this module is not compiled in).

use crate::board::Board;
use crate::eval::nnue::feature_index;
use crate::types::{Color, Square};

/// A loaded bilinear policy net (float; ordering-only, so not held to the
/// eval's quantized-SIMD budget in this first cut).
pub struct PolicyNet {
    inp: usize,
    tr: usize,
    k: usize,
    l1w: Vec<f32>, // tr × inp, row-major: weight(row r, col c) = l1w[r*inp + c]
    l1b: Vec<f32>, // tr
    few: Vec<f32>, // (64·k) × tr, row-major
    feb: Vec<f32>, // 64·k
    tew: Vec<f32>, // (64·k) × tr, row-major
    teb: Vec<f32>, // 64·k
}

fn rd_u32(b: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}
fn rd_f32s(b: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(f32::from_le_bytes(b[*off..*off + 4].try_into().unwrap()));
        *off += 4;
    }
    v
}

impl PolicyNet {
    /// Load a `policy_bilinear.bin` (header `u32 in,tr,k` then the f32 arrays in
    /// `l1w,l1b,fe.w,fe.b,te.w,te.b` order — the layout `policy_bil_export.py`
    /// writes). Panics only on a malformed/truncated file (a build asset, not
    /// user input on the hot path).
    pub fn load(path: &str) -> std::io::Result<PolicyNet> {
        let b = std::fs::read(path)?;
        let mut o = 0usize;
        let inp = rd_u32(&b, &mut o) as usize;
        let tr = rd_u32(&b, &mut o) as usize;
        let k = rd_u32(&b, &mut o) as usize;
        let l1w = rd_f32s(&b, &mut o, tr * inp);
        let l1b = rd_f32s(&b, &mut o, tr);
        let few = rd_f32s(&b, &mut o, 64 * k * tr);
        let feb = rd_f32s(&b, &mut o, 64 * k);
        let tew = rd_f32s(&b, &mut o, 64 * k * tr);
        let teb = rd_f32s(&b, &mut o, 64 * k);
        Ok(PolicyNet { inp, tr, k, l1w, l1b, few, feb, tew, teb })
    }

    /// Compute the side-to-move move-prior embeddings for `board` — one forward
    /// pass, shared by every move at this node.
    pub fn scores(&self, board: &Board) -> PolicyScores {
        let stm = board.side_to_move();
        // trunk = SCReLU(l1w · onehot(features) + l1b); features are sparse (one
        // set bit per piece), so the FT reduces to summing the l1w columns of
        // the active features into the bias — the same trick the eval FT uses.
        let mut h = self.l1b.clone();
        for sq in board.all_occupancy() {
            let piece = board.piece_on(sq).expect("occupied square has a piece");
            let f = feature_index(stm, piece, sq); // 0..768 (== eval's stm feature)
            let mut r = 0;
            while r < self.tr {
                h[r] += self.l1w[r * self.inp + f];
                r += 1;
            }
        }
        for v in h.iter_mut() {
            let c = v.clamp(0.0, 1.0);
            *v = c * c;
        }
        let fe = affine(&self.few, &self.feb, &h, 64 * self.k, self.tr);
        let te = affine(&self.tew, &self.teb, &h, 64 * self.k, self.tr);
        PolicyScores { fe, te, k: self.k, stm }
    }
}

fn affine(w: &[f32], b: &[f32], h: &[f32], out: usize, tr: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; out];
    for r in 0..out {
        let mut s = b[r];
        let base = r * tr;
        for c in 0..tr {
            s += w[base + c] * h[c];
        }
        o[r] = s;
    }
    o
}

/// One position's from/to embeddings — cache once per node, then `move_score`
/// each candidate move cheaply (a `K`-dim dot).
pub struct PolicyScores {
    fe: Vec<f32>, // 64 × k
    te: Vec<f32>, // 64 × k
    k: usize,
    stm: Color,
}

impl PolicyScores {
    /// Bilinear prior for the move `from → to`. Squares are taken stm-relative
    /// (matching training, which normalized every position to white-to-move).
    pub fn move_score(&self, from: Square, to: Square) -> f32 {
        let f = from.relative_to(self.stm).index() as usize;
        let t = to.relative_to(self.stm).index() as usize;
        let mut s = 0.0;
        for j in 0..self.k {
            s += self.fe[f * self.k + j] * self.te[t * self.k + j];
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use std::str::FromStr;

    /// Parity gate: the Rust bilinear forward must reproduce the reference
    /// Python (`policy_parity_bil.py`) to within f32 accumulation noise — this
    /// pins the feature encoding, the SCReLU trunk, and the stm-relative flip.
    ///
    /// The net `.bin` is a research artifact outside the repo, so the test
    /// SKIPS unless `POLICY_NET` points at a compatible net. Run it manually:
    ///   `POLICY_NET=<scratch>/policy_bilinear.bin cargo test --features policy parity`
    #[test]
    fn parity_with_python_bilinear() {
        let Ok(net_path) = std::env::var("POLICY_NET") else {
            return;
        };
        let net = PolicyNet::load(&net_path).expect("load policy net");
        // (fen, [(uci_move, expected_score)]) straight from the Python fixture —
        // white pos 1 and its black mirror (pos 2) must give mirrored scores.
        let cases: &[(&str, &[(&str, f32)])] = &[
            (
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                &[("b1c3", 1.4197), ("d2d3", 1.2533), ("f1b5", 1.1136)],
            ),
            (
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3",
                &[("b8c6", 1.4197), ("d7d6", 1.2533), ("f8b4", 1.1136)],
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                &[("g2g4", 3.032), ("d5e6", 2.5202), ("e5d3", 2.4166)],
            ),
        ];
        for (fen, moves) in cases {
            let board = Board::from_fen(fen).expect("parse fen");
            let sc = net.scores(&board);
            for (uci, want) in *moves {
                let from = Square::from_str(&uci[0..2]).unwrap();
                let to = Square::from_str(&uci[2..4]).unwrap();
                let got = sc.move_score(from, to);
                assert!(
                    (got - want).abs() < 2e-2,
                    "{fen} {uci}: got {got}, want {want}"
                );
            }
        }
    }
}
