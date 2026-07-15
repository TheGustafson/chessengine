//! CENN-v4: a frozen classic evaluator plus an exact HalfKAv2_hm residual.
//!
//! The residual is maintained independently and added to the bit-identical
//! CENN-v1 base before the ordinary evaluation outer shape is applied.

use super::{
    Network, load_bytes,
    sota_features::{HALFKA_DIMS, halfka_index},
};
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, PieceType, Value};
use std::{error::Error, fmt};

pub const MAGIC: [u8; 4] = *b"CEN4";
const VERSION: u16 = 4;
const HEADER: usize = 64;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;
pub const ACC: usize = 256;

#[derive(Debug)]
pub struct ProtectedHalfkaNetwork {
    pub base: Network,
    pub halfka_weights: Box<[i16]>,
    pub ft_bias: Box<[i16]>,
    pub out_weights: Box<[i16]>,
    pub out_bias: i32,
}

#[derive(Debug)]
pub enum ErrorKind {
    TooShort,
    BadMagic,
    Wrong(&'static str, u32, u32),
    BadLength(&'static str, u64, u64),
    BadChecksum(u64, u64),
    BadBase(String),
    OutWeightOutOfClip(i16),
}
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "CENN-v4 is truncated"),
            Self::BadMagic => write!(f, "CENN-v4 bad magic"),
            Self::Wrong(n, e, g) => write!(f, "CENN-v4 {n} {g} != {e}"),
            Self::BadLength(n, e, g) => write!(f, "CENN-v4 {n} length {g} != {e}"),
            Self::BadChecksum(e, g) => write!(f, "CENN-v4 checksum {g:#x} != {e:#x}"),
            Self::BadBase(e) => write!(f, "CENN-v4 frozen base is invalid: {e}"),
            Self::OutWeightOutOfClip(v) => write!(f, "CENN-v4 residual output |weight| {v} > 127"),
        }
    }
}
impl Error for ErrorKind {}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn fnv(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &v in data {
        hash = (hash ^ u64::from(v)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}
fn residual_len() -> usize {
    2 * (HALFKA_DIMS * ACC + ACC + 2 * ACC) + 4
}

pub fn load_bytes_v4(bytes: &[u8]) -> Result<ProtectedHalfkaNetwork, ErrorKind> {
    if bytes.len() < HEADER {
        return Err(ErrorKind::TooShort);
    }
    if bytes[..4] != MAGIC {
        return Err(ErrorKind::BadMagic);
    }
    for (name, expected, got) in [
        ("version", VERSION as u32, u16_at(bytes, 4) as u32),
        ("halfka_dims", HALFKA_DIMS as u32, u32_at(bytes, 8)),
        ("acc_width", ACC as u32, u32_at(bytes, 12)),
        ("qa", QA as u32, u32_at(bytes, 16)),
        ("qb", QB as u32, u32_at(bytes, 20)),
        ("scale", SCALE as u32, u32_at(bytes, 24)),
    ] {
        if expected != got {
            return Err(ErrorKind::Wrong(name, expected, got));
        }
    }
    let base_len = u64_at(bytes, 32) as usize;
    let got_residual_len = u64_at(bytes, 40);
    let expected_residual_len = residual_len() as u64;
    if got_residual_len != expected_residual_len {
        return Err(ErrorKind::BadLength(
            "residual",
            expected_residual_len,
            got_residual_len,
        ));
    }
    let total = HEADER
        .checked_add(base_len)
        .and_then(|x| x.checked_add(residual_len()))
        .ok_or(ErrorKind::TooShort)?;
    if bytes.len() != total {
        return Err(ErrorKind::BadLength(
            "file",
            total as u64,
            bytes.len() as u64,
        ));
    }
    let payload = &bytes[HEADER..];
    let expected_hash = u64_at(bytes, 48);
    let got_hash = fnv(payload);
    if expected_hash != got_hash {
        return Err(ErrorKind::BadChecksum(expected_hash, got_hash));
    }

    let base = load_bytes(&payload[..base_len]).map_err(|e| ErrorKind::BadBase(e.to_string()))?;
    let residual = &payload[base_len..];
    let mut offset = 0;
    let mut take_i16 = |count: usize| {
        let values = residual[offset..offset + count * 2]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        offset += count * 2;
        values.into_boxed_slice()
    };
    let halfka_weights = take_i16(HALFKA_DIMS * ACC);
    let ft_bias = take_i16(ACC);
    let out_weights = take_i16(2 * ACC);
    if let Some(max) = out_weights.iter().map(|x| x.abs()).max()
        && max > 127
    {
        return Err(ErrorKind::OutWeightOutOfClip(max));
    }
    drop(take_i16);
    let out_bias = i32::from_le_bytes(residual[offset..offset + 4].try_into().unwrap());
    Ok(ProtectedHalfkaNetwork {
        base,
        halfka_weights,
        ft_bias,
        out_weights,
        out_bias,
    })
}

pub struct ProtectedHalfkaContext {
    net: &'static ProtectedHalfkaNetwork,
    stack: Box<[Entry]>,
}
#[derive(Clone)]
struct Entry {
    acc: [[i16; ACC]; 2],
    dirty: DirtyPiece,
    computed: [bool; 2],
}
impl Default for Entry {
    fn default() -> Self {
        Self {
            acc: [[0; ACC]; 2],
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}
impl fmt::Debug for ProtectedHalfkaContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProtectedHalfkaContext")
            .finish_non_exhaustive()
    }
}
impl ProtectedHalfkaContext {
    pub fn new(net: &'static ProtectedHalfkaNetwork) -> Self {
        Self {
            net,
            stack: vec![Entry::default(); MAX_PLY + 8].into_boxed_slice(),
        }
    }
    pub fn net(&self) -> &ProtectedHalfkaNetwork {
        self.net
    }
    pub fn refresh_root(&mut self, board: &Board) {
        for perspective in Color::ALL {
            self.stack[0].acc[perspective as usize] = self.refresh(board, perspective);
        }
        self.stack[0].computed = [true; 2];
    }
    pub fn push(&mut self, dirty: DirtyPiece, ply: usize) {
        self.stack[ply + 1].dirty = dirty;
        self.stack[ply + 1].computed = [false; 2];
    }
    fn refresh(&self, board: &Board, perspective: Color) -> [i16; ACC] {
        let mut acc = [0i16; ACC];
        acc.copy_from_slice(&self.net.ft_bias);
        let king = board.king_square(perspective);
        for square in board.all_occupancy() {
            let piece = board.piece_on(square).unwrap();
            let base = halfka_index(perspective, piece, square, king) * ACC;
            for (dst, &weight) in acc
                .iter_mut()
                .zip(&self.net.halfka_weights[base..base + ACC])
            {
                *dst = dst.wrapping_add(weight);
            }
        }
        acc
    }
    fn compute(&mut self, board: &Board, ply: usize, perspective: Color) {
        let p = perspective as usize;
        if self.stack[ply].computed[p] {
            return;
        }
        let mut ancestor = ply;
        while ancestor > 0 && !self.stack[ancestor].computed[p] {
            ancestor -= 1;
        }
        let king_piece = crate::types::Piece::new(perspective, PieceType::King);
        if ((ancestor + 1)..=ply)
            .any(|k| self.stack[k].dirty.subs().iter().any(|x| x.0 == king_piece))
        {
            self.stack[ply].acc[p] = self.refresh(board, perspective);
            self.stack[ply].computed[p] = true;
            return;
        }
        let king = board.king_square(perspective);
        for child in (ancestor + 1)..=ply {
            let dirty = self.stack[child].dirty;
            let mut acc = self.stack[child - 1].acc[p];
            for &(piece, square) in dirty.subs() {
                let base = halfka_index(perspective, piece, square, king) * ACC;
                for (dst, &weight) in acc
                    .iter_mut()
                    .zip(&self.net.halfka_weights[base..base + ACC])
                {
                    *dst = dst.wrapping_sub(weight);
                }
            }
            for &(piece, square) in dirty.adds() {
                let base = halfka_index(perspective, piece, square, king) * ACC;
                for (dst, &weight) in acc
                    .iter_mut()
                    .zip(&self.net.halfka_weights[base..base + ACC])
                {
                    *dst = dst.wrapping_add(weight);
                }
            }
            self.stack[child].acc[p] = acc;
            self.stack[child].computed[p] = true;
        }
    }
    pub fn evaluate_residual(&mut self, board: &Board, ply: usize) -> Value {
        for perspective in Color::ALL {
            self.compute(board, ply, perspective);
        }
        let acc = &self.stack[ply].acc;
        let stm = board.side_to_move() as usize;
        let out: &[i16; 2 * ACC] = self
            .net
            .out_weights
            .as_ref()
            .try_into()
            .expect("CENN-v4 output length validated");
        (super::KERNELS.forward)(
            &acc[stm],
            &acc[stm ^ 1],
            out,
            self.net.out_bias,
            super::Activation::SCReLU,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::Activation;
    use crate::movegen::generate_legal;
    use crate::moves::MoveList;

    fn leaked_test_net() -> &'static ProtectedHalfkaNetwork {
        let halfka_weights = (0..HALFKA_DIMS * ACC)
            .map(|i| (i as i16 % 7) - 3)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let out_weights = (0..2 * ACC)
            .map(|i| (i as i16 % 11) - 5)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Box::leak(Box::new(ProtectedHalfkaNetwork {
            base: Network::deterministic_test_net(7, Activation::SCReLU),
            halfka_weights,
            ft_bias: vec![17; ACC].into_boxed_slice(),
            out_weights,
            out_bias: 1234,
        }))
    }

    #[test]
    fn incremental_residual_matches_fresh_context() {
        let net = leaked_test_net();
        let mut incremental = ProtectedHalfkaContext::new(net);
        let mut board = Board::startpos();
        incremental.refresh_root(&board);
        let mut seed = 0xC3E4_51D4u64;
        for ply in 0..48usize {
            let mut moves = MoveList::new();
            generate_legal(&board, &mut moves);
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let m = moves[seed as usize % moves.len()];
            let dirty = DirtyPiece::from_move(&board, m);
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            incremental.push(dirty, ply);
            let got = incremental.evaluate_residual(&board, ply + 1);

            let mut fresh = ProtectedHalfkaContext::new(net);
            fresh.refresh_root(&board);
            let expected = fresh.evaluate_residual(&board, 0);
            assert_eq!(got, expected, "ply {ply} {m}");
        }
    }
}
