//! CENN-v3: king-conditioned Chess768 transformer with a classic SCReLU head.

use super::feature_index;
use crate::board::{Board, DirtyPiece};
use crate::types::{Color, MAX_PLY, Piece, PieceType, Square, Value};
use std::{error::Error, fmt};

pub const MAGIC: [u8; 4] = *b"CEN3";
const VERSION: u16 = 3;
const HEADER: usize = 48;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;
pub const SIMPLE_ACC: usize = 256;
pub const KING768_DIMS: usize = 64 * 768;

#[inline]
fn king768_index(perspective: Color, piece: Piece, square: Square, king: Square) -> usize {
    king.relative_to(perspective).index() as usize * 768 + feature_index(perspective, piece, square)
}

#[derive(Debug)]
pub struct SimpleHalfkaNetwork {
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
    BadLength(u64, u64),
    BadChecksum(u64, u64),
}
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "CENN-v3 is truncated"),
            Self::BadMagic => write!(f, "CENN-v3 bad magic"),
            Self::Wrong(n, e, g) => write!(f, "CENN-v3 {n} {g} != {e}"),
            Self::BadLength(e, g) => write!(f, "CENN-v3 payload length {g} != {e}"),
            Self::BadChecksum(e, g) => write!(f, "CENN-v3 checksum {g:#x} != {e:#x}"),
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
fn payload_len() -> usize {
    2 * (KING768_DIMS * SIMPLE_ACC + SIMPLE_ACC + 2 * SIMPLE_ACC) + 4
}

pub fn load_bytes(bytes: &[u8]) -> Result<SimpleHalfkaNetwork, ErrorKind> {
    if bytes.len() < HEADER {
        return Err(ErrorKind::TooShort);
    }
    if bytes[..4] != MAGIC {
        return Err(ErrorKind::BadMagic);
    }
    for (name, expected, got) in [
        ("version", VERSION as u32, u16_at(bytes, 4) as u32),
        ("king768_dims", KING768_DIMS as u32, u32_at(bytes, 8)),
        ("acc_width", SIMPLE_ACC as u32, u32_at(bytes, 12)),
        ("qa", QA as u32, u32_at(bytes, 16)),
        ("qb", QB as u32, u32_at(bytes, 20)),
        ("scale", SCALE as u32, u32_at(bytes, 24)),
    ] {
        if expected != got {
            return Err(ErrorKind::Wrong(name, expected, got));
        }
    }
    let expected_len = payload_len() as u64;
    let got_len = u64_at(bytes, 32);
    if expected_len != got_len {
        return Err(ErrorKind::BadLength(expected_len, got_len));
    }
    if bytes.len() != HEADER + expected_len as usize {
        return Err(ErrorKind::TooShort);
    }
    let payload = &bytes[HEADER..];
    let expected_hash = u64_at(bytes, 40);
    let got_hash = fnv(payload);
    if expected_hash != got_hash {
        return Err(ErrorKind::BadChecksum(expected_hash, got_hash));
    }
    let mut offset = 0;
    let mut take_i16 = |count: usize| {
        let values = payload[offset..offset + count * 2]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        offset += count * 2;
        values.into_boxed_slice()
    };
    let halfka_weights = take_i16(KING768_DIMS * SIMPLE_ACC);
    let ft_bias = take_i16(SIMPLE_ACC);
    let out_weights = take_i16(2 * SIMPLE_ACC);
    drop(take_i16);
    let out_bias = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
    Ok(SimpleHalfkaNetwork {
        halfka_weights,
        ft_bias,
        out_weights,
        out_bias,
    })
}

pub struct SimpleHalfkaContext {
    net: &'static SimpleHalfkaNetwork,
    stack: Box<[SimpleEntry]>,
}
#[derive(Clone)]
struct SimpleEntry {
    acc: [[i16; SIMPLE_ACC]; 2],
    dirty: DirtyPiece,
    computed: [bool; 2],
}
impl Default for SimpleEntry {
    fn default() -> Self {
        Self {
            acc: [[0; SIMPLE_ACC]; 2],
            dirty: DirtyPiece::EMPTY,
            computed: [false; 2],
        }
    }
}
impl fmt::Debug for SimpleHalfkaContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleHalfkaContext")
            .finish_non_exhaustive()
    }
}
impl SimpleHalfkaContext {
    pub fn new(net: &'static SimpleHalfkaNetwork) -> Self {
        Self {
            net,
            stack: vec![SimpleEntry::default(); MAX_PLY + 8].into_boxed_slice(),
        }
    }
    pub fn net(&self) -> &SimpleHalfkaNetwork {
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
    fn refresh(&self, board: &Board, perspective: Color) -> [i16; SIMPLE_ACC] {
        let mut acc = [0i16; SIMPLE_ACC];
        acc.copy_from_slice(&self.net.ft_bias);
        let king = board.king_square(perspective);
        for square in board.all_occupancy() {
            let piece = board.piece_on(square).unwrap();
            let base = king768_index(perspective, piece, square, king) * SIMPLE_ACC;
            for i in 0..SIMPLE_ACC {
                acc[i] = acc[i].wrapping_add(self.net.halfka_weights[base + i]);
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
                let base = king768_index(perspective, piece, square, king) * SIMPLE_ACC;
                for i in 0..SIMPLE_ACC {
                    acc[i] = acc[i].wrapping_sub(self.net.halfka_weights[base + i]);
                }
            }
            for &(piece, square) in dirty.adds() {
                let base = king768_index(perspective, piece, square, king) * SIMPLE_ACC;
                for i in 0..SIMPLE_ACC {
                    acc[i] = acc[i].wrapping_add(self.net.halfka_weights[base + i]);
                }
            }
            self.stack[child].acc[p] = acc;
            self.stack[child].computed[p] = true;
        }
    }
    pub fn evaluate(&mut self, board: &Board, ply: usize) -> Value {
        for perspective in Color::ALL {
            self.compute(board, ply, perspective);
        }
        let acc = &self.stack[ply].acc;
        let stm = board.side_to_move() as usize;
        #[cfg(not(any(
            feature = "nnue-l1-512",
            feature = "nnue-l1-1024",
            feature = "nnue-l1-2048"
        )))]
        {
            let out: &[i16; 2 * SIMPLE_ACC] = self
                .net
                .out_weights
                .as_ref()
                .try_into()
                .expect("CENN-v3 output length was validated by the loader");
            return (super::KERNELS.forward)(
                &acc[stm],
                &acc[stm ^ 1],
                out,
                self.net.out_bias,
                super::Activation::SCReLU,
            );
        }
        #[cfg(any(
            feature = "nnue-l1-512",
            feature = "nnue-l1-1024",
            feature = "nnue-l1-2048"
        ))]
        {
            let mut sum = 0i64;
            for (half, perspective) in [stm, stm ^ 1].into_iter().enumerate() {
                for i in 0..SIMPLE_ACC {
                    let a = i64::from(acc[perspective][i]).clamp(0, i64::from(QA));
                    sum += a * a * i64::from(self.net.out_weights[half * SIMPLE_ACC + i]);
                }
            }
            sum /= i64::from(QA);
            ((sum + i64::from(self.net.out_bias)) * i64::from(SCALE) / i64::from(QA * QB)) as Value
        }
    }
}
