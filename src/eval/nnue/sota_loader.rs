//! The CENN-v2 network container: parse, validate, and serialize D40 SOTA nets
//! (design/11 §12.6).
//!
//! Completely separate from the v1 CENN loader ([`super::loader`]) — the arch
//! constants, payload shape, and returned type (`SotaNetwork` vs `Network`) are
//! all different. A v2 file starts with magic `b"CEN2"` so the two never
//! collide: the v1 loader rejects it with `BadMagic`, the v2 loader rejects a
//! v1 file with `BadMagic`.
//!
//! ```text
//! CENN-v2 header (64 bytes)
//! offset  size  field
//! 0       4     magic = b"CEN2"
//! 4       2     version: u16 LE (= 2)
//! 6       2     flags: u16 (bit0 = has_threats)
//! 8       4     halfka_dims: u32 (22528)
//! 12      4     threat_dims: u32 (60720 or 0)
//! 16      4     acc_width: u32 (768)
//! 20      4     l2_width: u32 (16)
//! 24      4     l3_width: u32 (32)
//! 28      4     out_buckets: u32 (8)
//! 32      4     ft_quant: i32 (255)
//! 36      4     l1_quant: i32 (64)
//! 40      4     ft_shift: i32 (9)
//! 44      4     network_scale: i32 (380)
//! 48      8     payload_len: u64 LE
//! 56      8     payload_checksum: u64 LE (FNV-1a-64)
//! ```
//!
//! Payload (all little-endian, in order):
//! ```text
//! halfka_w:  i16[halfka_dims · acc_width]      feature-major
//! ft_bias:   i16[acc_width]
//! threat_w:  i8[threat_dims · acc_width]        feature-major (present iff has_threats)
//! l1_w:      i8[out_buckets · l2_width · acc_width]  bucket-major, group-interleaved
//! l1_b:      f32[out_buckets · l2_width]
//! l2_w:      f32[out_buckets · l2_width · l3_width]
//! l2_b:      f32[out_buckets · l3_width]
//! l3_w:      f32[out_buckets · l3_width]
//! l3_b:      f32[out_buckets]
//! ```
//!
//! Parsing NEVER panics on user-supplied bytes — every anomaly is a
//! [`SotaNnueError`]. The loader validates every dimension against the
//! compile-time architecture constants from [`super::sota_head`] and
//! [`super::sota_features`].

use super::sota_features::{ACC, HALFKA_DIMS, THREAT_DIMS};
use super::sota_head::{self, BUCKETS, L1, L2, L3, SotaHead};
use std::error::Error;
use std::fmt;
use std::path::Path;

/// CENN-v2 magic bytes.
pub const MAGIC_V2: [u8; 4] = *b"CEN2";
/// Container version.
pub const CENN_V2_VERSION: u16 = 2;
/// Fixed header size in bytes.
pub const HEADER_V2_LEN: usize = 64;
/// Flag bit: threats feature transformer is present.
pub const FLAG_HAS_THREATS: u16 = 1;

// Compile-time assertions: the head and feature constants must agree.
const _: () = assert!(L1 == ACC, "SotaHead::L1 must equal ACC (the accumulator width)");

/// A D40 SOTA network: two feature transformers + multi-layer bucketed head.
/// Produced by [`load_v2_bytes`]; consumed by the SOTA eval context.
#[derive(Debug)]
pub struct SotaNetwork {
    /// HalfKAv2_hm FT weights, feature-major: `[HALFKA_DIMS · ACC]` i16.
    pub halfka_weights: Box<[i16]>,
    /// Shared FT bias: `[ACC]` i16 (seeds the HalfKA accumulator; threats seed 0).
    pub ft_bias: Box<[i16]>,
    /// FullThreats FT weights, feature-major: `[THREAT_DIMS · ACC]` i8.
    /// Empty if the net has no threats channel.
    pub threat_weights: Box<[i8]>,
    /// The multi-layer bucketed head (owns its own weight boxes).
    pub head: SotaHead,
}

impl PartialEq for SotaNetwork {
    fn eq(&self, other: &Self) -> bool {
        self.halfka_weights == other.halfka_weights
            && self.ft_bias == other.ft_bias
            && self.threat_weights == other.threat_weights
            && self.head.network_scale == other.head.network_scale
            && self.head.l1_weights == other.head.l1_weights
            && self.head.l1_biases == other.head.l1_biases
            && self.head.l2_weights == other.head.l2_weights
            && self.head.l2_biases == other.head.l2_biases
            && self.head.l3_weights == other.head.l3_weights
            && self.head.l3_biases == other.head.l3_biases
    }
}

impl Eq for SotaNetwork {}

/// An error from parsing a CENN-v2 container. Fallible parsing returns this,
/// never a panic — the loader takes untrusted files.
#[derive(Debug)]
pub enum SotaNnueError {
    /// The file could not be read.
    Io(std::io::Error),
    /// Fewer bytes than needed.
    TooShort { need: usize, got: usize },
    /// The first four bytes were not `b"CEN2"`.
    BadMagic([u8; 4]),
    /// Version mismatch.
    UnsupportedVersion(u16),
    /// A dimension in the header didn't match this build's compile-time constant.
    WrongDim {
        field: &'static str,
        expected: u32,
        found: u32,
    },
    /// A quantization constant didn't match.
    WrongQuant {
        field: &'static str,
        expected: i32,
        found: i32,
    },
    /// Declared payload length didn't match the architecture's computed size.
    BadPayloadLen { expected: u64, found: u64 },
    /// Payload checksum mismatch.
    ChecksumMismatch { expected: u64, found: u64 },
}

impl fmt::Display for SotaNnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SotaNnueError::Io(e) => write!(f, "reading SOTA NNUE file: {e}"),
            SotaNnueError::TooShort { need, got } => {
                write!(f, "CENN-v2 too short: need {need} bytes, got {got}")
            }
            SotaNnueError::BadMagic(m) => {
                write!(f, "CENN-v2 bad magic: {m:?} (expected b\"CEN2\")")
            }
            SotaNnueError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "CENN-v2 unsupported version {v} (this build reads {CENN_V2_VERSION})"
                )
            }
            SotaNnueError::WrongDim {
                field,
                expected,
                found,
            } => write!(
                f,
                "CENN-v2 {field} {found} != {expected} (architecture mismatch)"
            ),
            SotaNnueError::WrongQuant {
                field,
                expected,
                found,
            } => write!(
                f,
                "CENN-v2 {field} {found} != {expected} (quant mismatch)"
            ),
            SotaNnueError::BadPayloadLen { expected, found } => write!(
                f,
                "CENN-v2 payload_len {found} != {expected} (architecture mismatch)"
            ),
            SotaNnueError::ChecksumMismatch { expected, found } => write!(
                f,
                "CENN-v2 checksum mismatch: header {expected:#018x}, computed {found:#018x}"
            ),
        }
    }
}

impl Error for SotaNnueError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SotaNnueError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ─── Payload size helpers ──────────────────────────────────────────────────

/// Payload byte length for a net WITH threats.
fn payload_len_with_threats() -> usize {
    let halfka_bytes = HALFKA_DIMS * ACC * 2; // i16
    let bias_bytes = ACC * 2; // i16
    let threat_bytes = THREAT_DIMS * ACC; // i8
    let l1w_bytes = BUCKETS * L2 * L1; // i8
    let l1b_bytes = BUCKETS * L2 * 4; // f32
    let l2w_bytes = BUCKETS * L2 * L3 * 4; // f32
    let l2b_bytes = BUCKETS * L3 * 4; // f32
    let l3w_bytes = BUCKETS * L3 * 4; // f32
    let l3b_bytes = BUCKETS * 4; // f32
    halfka_bytes
        + bias_bytes
        + threat_bytes
        + l1w_bytes
        + l1b_bytes
        + l2w_bytes
        + l2b_bytes
        + l3w_bytes
        + l3b_bytes
}

/// Payload byte length for a net WITHOUT threats.
fn payload_len_no_threats() -> usize {
    let halfka_bytes = HALFKA_DIMS * ACC * 2;
    let bias_bytes = ACC * 2;
    let l1w_bytes = BUCKETS * L2 * L1;
    let l1b_bytes = BUCKETS * L2 * 4;
    let l2w_bytes = BUCKETS * L2 * L3 * 4;
    let l2b_bytes = BUCKETS * L3 * 4;
    let l3w_bytes = BUCKETS * L3 * 4;
    let l3b_bytes = BUCKETS * 4;
    halfka_bytes + bias_bytes + l1w_bytes + l1b_bytes + l2w_bytes + l2b_bytes + l3w_bytes + l3b_bytes
}

// ─── Low-level readers ─────────────────────────────────────────────────────

#[inline]
fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
#[inline]
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[inline]
fn read_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[inline]
fn read_u64(b: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(buf)
}

/// FNV-1a 64-bit (same as the v1 loader — design/11 §3.4.5).
fn fnv1a64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ─── Parser ────────────────────────────────────────────────────────────────

/// Parse and validate a CENN-v2 container. Rejects every anomaly with a
/// [`SotaNnueError`]; never panics on the input bytes.
pub fn load_v2_bytes(bytes: &[u8]) -> Result<SotaNetwork, SotaNnueError> {
    // ── Header ─────────────────────────────────────────────────────────
    if bytes.len() < HEADER_V2_LEN {
        return Err(SotaNnueError::TooShort {
            need: HEADER_V2_LEN,
            got: bytes.len(),
        });
    }

    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != MAGIC_V2 {
        return Err(SotaNnueError::BadMagic(magic));
    }

    let version = read_u16(bytes, 4);
    if version != CENN_V2_VERSION {
        return Err(SotaNnueError::UnsupportedVersion(version));
    }

    let flags = read_u16(bytes, 6);
    let has_threats = (flags & FLAG_HAS_THREATS) != 0;

    // Validate all architecture dimensions against compile-time constants.
    let check_dim = |off, name, expected: usize| -> Result<(), SotaNnueError> {
        let found = read_u32(bytes, off);
        if found as usize != expected {
            return Err(SotaNnueError::WrongDim {
                field: name,
                expected: expected as u32,
                found,
            });
        }
        Ok(())
    };
    check_dim(8, "halfka_dims", HALFKA_DIMS)?;
    let threat_dim_expected = if has_threats { THREAT_DIMS } else { 0 };
    check_dim(12, "threat_dims", threat_dim_expected)?;
    check_dim(16, "acc_width", ACC)?;
    check_dim(20, "l2_width", L2)?;
    check_dim(24, "l3_width", L3)?;
    check_dim(28, "out_buckets", BUCKETS)?;

    // Validate quantization constants.
    let check_quant = |off, name, expected: i32| -> Result<(), SotaNnueError> {
        let found = read_i32(bytes, off);
        if found != expected {
            return Err(SotaNnueError::WrongQuant {
                field: name,
                expected,
                found,
            });
        }
        Ok(())
    };
    check_quant(32, "ft_quant", sota_head::FT_QUANT)?;
    check_quant(36, "l1_quant", sota_head::L1_QUANT)?;
    check_quant(40, "ft_shift", sota_head::FT_SHIFT)?;
    let network_scale = read_i32(bytes, 44);

    // Payload length + checksum.
    let expected_payload = if has_threats {
        payload_len_with_threats()
    } else {
        payload_len_no_threats()
    };
    let declared_payload = read_u64(bytes, 48);
    if declared_payload != expected_payload as u64 {
        return Err(SotaNnueError::BadPayloadLen {
            expected: expected_payload as u64,
            found: declared_payload,
        });
    }
    let header_checksum = read_u64(bytes, 56);

    // ── Payload ────────────────────────────────────────────────────────
    let total_need = HEADER_V2_LEN + expected_payload;
    if bytes.len() < total_need {
        return Err(SotaNnueError::TooShort {
            need: total_need,
            got: bytes.len(),
        });
    }
    let payload = &bytes[HEADER_V2_LEN..HEADER_V2_LEN + expected_payload];
    let computed_checksum = fnv1a64(payload);
    if computed_checksum != header_checksum {
        return Err(SotaNnueError::ChecksumMismatch {
            expected: header_checksum,
            found: computed_checksum,
        });
    }

    // All reads below are in-bounds by construction (payload length validated).
    let mut cur = 0usize;

    // HalfKA FT weights: i16[HALFKA_DIMS · ACC], feature-major.
    let n_halfka = HALFKA_DIMS * ACC;
    let halfka_weights = read_i16s(payload, &mut cur, n_halfka);

    // FT bias: i16[ACC].
    let ft_bias = read_i16s(payload, &mut cur, ACC);

    // Threats FT weights: i8[THREAT_DIMS · ACC], feature-major (if present).
    let threat_weights = if has_threats {
        read_i8s(payload, &mut cur, THREAT_DIMS * ACC)
    } else {
        Vec::new().into_boxed_slice()
    };

    // L1 weights: i8[BUCKETS · L2 · L1], bucket-major, group-interleaved.
    let l1_weights = read_i8s(payload, &mut cur, BUCKETS * L2 * L1);

    // L1 biases: f32[BUCKETS · L2].
    let l1_biases = read_f32s(payload, &mut cur, BUCKETS * L2);

    // L2 weights: f32[BUCKETS · L2 · L3].
    let l2_weights = read_f32s(payload, &mut cur, BUCKETS * L2 * L3);

    // L2 biases: f32[BUCKETS · L3].
    let l2_biases = read_f32s(payload, &mut cur, BUCKETS * L3);

    // L3 weights: f32[BUCKETS · L3].
    let l3_weights = read_f32s(payload, &mut cur, BUCKETS * L3);

    // L3 biases: f32[BUCKETS].
    let l3_biases = read_f32s(payload, &mut cur, BUCKETS);

    debug_assert_eq!(cur, expected_payload, "payload cursor must consume all bytes");

    let head = SotaHead {
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        l3_weights,
        l3_biases,
        network_scale,
    };

    Ok(SotaNetwork {
        halfka_weights,
        ft_bias,
        threat_weights,
        head,
    })
}

/// Load and validate a CENN-v2 net from a file.
pub fn load_v2(path: impl AsRef<Path>) -> Result<SotaNetwork, SotaNnueError> {
    let bytes = std::fs::read(path).map_err(SotaNnueError::Io)?;
    load_v2_bytes(&bytes)
}

// ─── Serializer (tests + netpack cross-check) ──────────────────────────────

/// Serialize a `SotaNetwork` to CENN-v2 bytes (the inverse of [`load_v2_bytes`]).
/// Used by tests to build deterministic nets and round-trip them; also lets the
/// Rust side produce a reference `.cenn` for cross-checking the Python packer.
pub fn serialize_v2(net: &SotaNetwork) -> Vec<u8> {
    let has_threats = !net.threat_weights.is_empty();
    let flags: u16 = if has_threats { FLAG_HAS_THREATS } else { 0 };

    // ── Payload ────────────────────────────────────────────────────────
    let expected_payload = if has_threats {
        payload_len_with_threats()
    } else {
        payload_len_no_threats()
    };
    let mut payload = Vec::with_capacity(expected_payload);

    // HalfKA FT weights.
    for &w in net.halfka_weights.iter() {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    // FT bias.
    for &b in net.ft_bias.iter() {
        payload.extend_from_slice(&b.to_le_bytes());
    }
    // Threats FT weights (if present).
    if has_threats {
        for &w in net.threat_weights.iter() {
            payload.push(w as u8);
        }
    }
    // L1 weights (i8).
    for &w in net.head.l1_weights.iter() {
        payload.push(w as u8);
    }
    // L1 biases (f32).
    for &b in net.head.l1_biases.iter() {
        payload.extend_from_slice(&b.to_le_bytes());
    }
    // L2 weights (f32).
    for &w in net.head.l2_weights.iter() {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    // L2 biases (f32).
    for &b in net.head.l2_biases.iter() {
        payload.extend_from_slice(&b.to_le_bytes());
    }
    // L3 weights (f32).
    for &w in net.head.l3_weights.iter() {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    // L3 biases (f32).
    for &b in net.head.l3_biases.iter() {
        payload.extend_from_slice(&b.to_le_bytes());
    }
    debug_assert_eq!(payload.len(), expected_payload);
    let checksum = fnv1a64(&payload);

    // ── Header ─────────────────────────────────────────────────────────
    let mut out = Vec::with_capacity(HEADER_V2_LEN + expected_payload);
    out.extend_from_slice(&MAGIC_V2);
    out.extend_from_slice(&CENN_V2_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(HALFKA_DIMS as u32).to_le_bytes());
    let threat_dim = if has_threats { THREAT_DIMS as u32 } else { 0 };
    out.extend_from_slice(&threat_dim.to_le_bytes());
    out.extend_from_slice(&(ACC as u32).to_le_bytes());
    out.extend_from_slice(&(L2 as u32).to_le_bytes());
    out.extend_from_slice(&(L3 as u32).to_le_bytes());
    out.extend_from_slice(&(BUCKETS as u32).to_le_bytes());
    out.extend_from_slice(&sota_head::FT_QUANT.to_le_bytes());
    out.extend_from_slice(&sota_head::L1_QUANT.to_le_bytes());
    out.extend_from_slice(&sota_head::FT_SHIFT.to_le_bytes());
    out.extend_from_slice(&net.head.network_scale.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());
    debug_assert_eq!(out.len(), HEADER_V2_LEN);

    out.extend_from_slice(&payload);
    out
}

// ─── Payload element readers ───────────────────────────────────────────────

fn read_i16s(payload: &[u8], cur: &mut usize, count: usize) -> Box<[i16]> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(i16::from_le_bytes([payload[*cur], payload[*cur + 1]]));
        *cur += 2;
    }
    v.into_boxed_slice()
}

fn read_i8s(payload: &[u8], cur: &mut usize, count: usize) -> Box<[i8]> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(payload[*cur] as i8);
        *cur += 1;
    }
    v.into_boxed_slice()
}

fn read_f32s(payload: &[u8], cur: &mut usize, count: usize) -> Box<[f32]> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let bytes = [payload[*cur], payload[*cur + 1], payload[*cur + 2], payload[*cur + 3]];
        v.push(f32::from_le_bytes(bytes));
        *cur += 4;
    }
    v.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attacks::Prng;

    /// Build a deterministic synthetic SOTA net (with threats) for round-trip
    /// tests. NOT a trained net — weights are small random values.
    fn test_net_with_threats() -> SotaNetwork {
        let mut prng = Prng::new(0xD40C_E420);

        let small_i16 = |prng: &mut Prng| -> i16 { (prng.rand64() % 201) as i16 - 100 };
        let small_i8 = |prng: &mut Prng| -> i8 { (prng.rand64() % 121) as i8 - 60 };
        let small_f32 = |prng: &mut Prng| -> f32 {
            ((prng.rand64() % 201) as f32 - 100.0) * 0.01
        };

        let halfka_weights: Box<[i16]> = (0..HALFKA_DIMS * ACC)
            .map(|_| small_i16(&mut prng))
            .collect();
        let ft_bias: Box<[i16]> = (0..ACC).map(|_| small_i16(&mut prng)).collect();
        let threat_weights: Box<[i8]> = (0..THREAT_DIMS * ACC)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_weights: Box<[i8]> = (0..BUCKETS * L2 * L1)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_biases: Box<[f32]> = (0..BUCKETS * L2)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l2_weights: Box<[f32]> = (0..BUCKETS * L2 * L3)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l2_biases: Box<[f32]> = (0..BUCKETS * L3)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l3_weights: Box<[f32]> = (0..BUCKETS * L3)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l3_biases: Box<[f32]> = (0..BUCKETS)
            .map(|_| small_f32(&mut prng))
            .collect();

        SotaNetwork {
            halfka_weights,
            ft_bias,
            threat_weights,
            head: SotaHead {
                l1_weights,
                l1_biases,
                l2_weights,
                l2_biases,
                l3_weights,
                l3_biases,
                network_scale: 380,
            },
        }
    }

    /// A no-threats variant: tests the `flags=0` path.
    fn test_net_no_threats() -> SotaNetwork {
        let mut full = test_net_with_threats();
        full.threat_weights = Vec::new().into_boxed_slice();
        full
    }

    #[test]
    fn round_trip_with_threats() {
        let net = test_net_with_threats();
        let bytes = serialize_v2(&net);
        assert_eq!(&bytes[0..4], b"CEN2");
        assert_eq!(read_u16(&bytes, 4), CENN_V2_VERSION);
        assert_eq!(read_u16(&bytes, 6) & FLAG_HAS_THREATS, FLAG_HAS_THREATS);

        let back = load_v2_bytes(&bytes).expect("valid net loads");
        assert_eq!(net, back);
    }

    #[test]
    fn round_trip_no_threats() {
        let net = test_net_no_threats();
        let bytes = serialize_v2(&net);
        assert_eq!(read_u16(&bytes, 6) & FLAG_HAS_THREATS, 0);
        assert_eq!(read_u32(&bytes, 12), 0); // threat_dims = 0

        let back = load_v2_bytes(&bytes).expect("valid no-threats net loads");
        assert_eq!(net, back);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        bytes[0] = b'X';
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadMagic(_))
        ));
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn rejects_wrong_dims() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        // Tamper halfka_dims.
        bytes[8..12].copy_from_slice(&9999u32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "halfka_dims",
                ..
            })
        ));
    }

    #[test]
    fn rejects_wrong_quant() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        // Tamper ft_quant at offset 32.
        bytes[32..36].copy_from_slice(&999i32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongQuant {
                field: "ft_quant",
                ..
            })
        ));
    }

    #[test]
    fn rejects_bad_payload_len() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        // Tamper payload_len at offset 48.
        let bad_len = 12345u64;
        bytes[48..56].copy_from_slice(&bad_len.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadPayloadLen { .. })
        ));
    }

    #[test]
    fn rejects_truncated() {
        let bytes = serialize_v2(&test_net_with_threats());
        // Truncate to less than header.
        assert!(matches!(
            load_v2_bytes(&bytes[..10]),
            Err(SotaNnueError::TooShort { .. })
        ));
        // Truncate mid-payload.
        assert!(matches!(
            load_v2_bytes(&bytes[..bytes.len() - 4]),
            Err(SotaNnueError::TooShort { .. })
        ));
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut bytes = serialize_v2(&test_net_with_threats());
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn error_display_and_source() {
        let e = load_v2(Path::new("/nonexistent.cen2")).unwrap_err();
        assert!(matches!(e, SotaNnueError::Io(_)));
        assert!(!e.to_string().is_empty());
        assert!(e.source().is_some());

        let e = SotaNnueError::BadMagic([0, 1, 2, 3]);
        assert!(e.to_string().contains("magic"));
        assert!(e.source().is_none());
    }

    #[test]
    fn payload_len_sanity() {
        // With threats: the expected payload is the sum of all tensor byte sizes.
        let expected = HALFKA_DIMS * ACC * 2  // halfka i16
            + ACC * 2                         // bias i16
            + THREAT_DIMS * ACC               // threats i8
            + BUCKETS * L2 * L1               // l1w i8
            + BUCKETS * L2 * 4                // l1b f32
            + BUCKETS * L2 * L3 * 4           // l2w f32
            + BUCKETS * L3 * 4                // l2b f32
            + BUCKETS * L3 * 4                // l3w f32
            + BUCKETS * 4;                    // l3b f32
        assert_eq!(payload_len_with_threats(), expected);

        // Verify the serialized net has the right total size.
        let net = test_net_with_threats();
        let bytes = serialize_v2(&net);
        assert_eq!(bytes.len(), HEADER_V2_LEN + expected);
    }

    /// FNV-1a-64 matches the v1 loader's implementation (same canonical vectors).
    #[test]
    fn fnv1a64_known_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
