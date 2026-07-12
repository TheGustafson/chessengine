//! The CENN network container: parse, validate, and serialize (design/11
//! §3.4.5).
//!
//! ```text
//! offset  size  field
//! 0       4     magic = b"CENN"
//! 4       4     version: u32 LE (= 1)
//! 8       4     input_dims: u32 (= 768)
//! 12      4     l1: u32 (= 256)
//! 16      1     activation: u8 (0 = CReLU, 1 = SCReLU)
//! 17      3     padding (zero)
//! 20      2     qa: u16 (= 255)
//! 22      2     qb: u16 (= 64)
//! 24      2     scale: u16 (= 400)
//! 26      6     padding (zero)
//! 32      8     payload_len: u64 LE
//! 40      8     payload_checksum: u64 LE (FNV-1a-64 over payload)
//! 48      -     payload, all little-endian, in order:
//!               ft_weights: i16[input_dims · l1]  (feature-major)
//!               ft_bias:    i16[l1]
//!               out_weights:i16[2 · l1]           (stm half first)
//!               out_bias:   i32
//! ```
//!
//! Parsing NEVER panics on user-supplied bytes (CLAUDE.md): every anomaly is a
//! [`NnueError`]. The default net will eventually be embedded via
//! `include_bytes!` (zero runtime deps, D15); this step only provides the
//! parse/validate/serialize primitives (the loader is exercised by the
//! `nnue-eval` dev subcommand and tests). The **vs-`bullet` cross-check** —
//! wrapping bullet's raw quantised.bin in this container and confirming
//! identical inference — folds in at step 3 when a real trained net exists.

use super::network::{Activation, Network};
use super::{INPUT, L1, QA, QB, SCALE};
use std::error::Error;
use std::fmt;
use std::path::Path;

/// CENN magic bytes.
pub const MAGIC: [u8; 4] = *b"CENN";
/// The only container version this build understands.
pub const CENN_VERSION: u32 = 1;
/// Fixed header size in bytes (design/11 §3.4.5).
pub const HEADER_LEN: usize = 48;

/// Payload byte length for the pinned `768 → 2×256 → 1` architecture:
/// `(INPUT·L1 + L1 + 2·L1)` i16 weights + one i32 bias.
const PAYLOAD_LEN: usize = (INPUT * L1 + L1 + 2 * L1) * 2 + 4;

/// A malformed or unsupported CENN container (design/11 §3.4.5). Fallible
/// parsing returns this, never a panic — the loader takes untrusted files.
#[derive(Debug)]
pub enum NnueError {
    /// The file could not be read.
    Io(std::io::Error),
    /// Fewer bytes than the header or the declared payload requires.
    TooShort { need: usize, got: usize },
    /// The first four bytes were not `b"CENN"`.
    BadMagic([u8; 4]),
    /// The container version is not [`CENN_VERSION`].
    UnsupportedVersion(u32),
    /// `input_dims` did not match this build's [`INPUT`].
    WrongInputDims(u32),
    /// `l1` did not match this build's [`L1`].
    WrongL1(u32),
    /// The activation id was neither 0 (CReLU) nor 1 (SCReLU).
    BadActivation(u8),
    /// A quantization constant in the header did not match this build's.
    WrongQuant {
        field: &'static str,
        expected: u16,
        found: u16,
    },
    /// The declared `payload_len` did not match the architecture's fixed size.
    BadPayloadLen { expected: u64, found: u64 },
    /// The recomputed FNV-1a-64 payload checksum did not match the header.
    ChecksumMismatch { expected: u64, found: u64 },
}

impl fmt::Display for NnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NnueError::Io(e) => write!(f, "reading NNUE file: {e}"),
            NnueError::TooShort { need, got } => {
                write!(f, "CENN too short: need {need} bytes, got {got}")
            }
            NnueError::BadMagic(m) => write!(f, "CENN bad magic: {m:?} (expected b\"CENN\")"),
            NnueError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "CENN unsupported version {v} (this build reads {CENN_VERSION})"
                )
            }
            NnueError::WrongInputDims(d) => {
                write!(
                    f,
                    "CENN input_dims {d} != {INPUT} (this build's architecture)"
                )
            }
            NnueError::WrongL1(l) => write!(f, "CENN l1 {l} != {L1} (this build's architecture)"),
            NnueError::BadActivation(a) => {
                write!(
                    f,
                    "CENN bad activation id {a} (expected 0=CReLU or 1=SCReLU)"
                )
            }
            NnueError::WrongQuant {
                field,
                expected,
                found,
            } => write!(
                f,
                "CENN {field} {found} != {expected} (this build's quantization)"
            ),
            NnueError::BadPayloadLen { expected, found } => {
                write!(
                    f,
                    "CENN payload_len {found} != {expected} (architecture mismatch)"
                )
            }
            NnueError::ChecksumMismatch { expected, found } => write!(
                f,
                "CENN checksum mismatch: header {expected:#018x}, computed {found:#018x}"
            ),
        }
    }
}

impl Error for NnueError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NnueError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// FNV-1a 64-bit over `data` (design/11 §3.4.5 payload checksum).
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

/// Serialize `net` to CENN bytes (design/11 §3.4.5). The inverse of
/// [`load_bytes`]; lets tests build deterministic nets and round-trip them.
pub fn serialize(net: &Network) -> Vec<u8> {
    assert_eq!(net.ft_weights.len(), INPUT * L1, "ft_weights width");
    assert_eq!(net.ft_bias.len(), L1, "ft_bias width");
    assert_eq!(net.out_weights.len(), 2 * L1, "out_weights width");

    // Payload first, so the checksum covers exactly what will be stored.
    let mut payload = Vec::with_capacity(PAYLOAD_LEN);
    for &w in net.ft_weights.iter() {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    for &b in net.ft_bias.iter() {
        payload.extend_from_slice(&b.to_le_bytes());
    }
    for &w in net.out_weights.iter() {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    payload.extend_from_slice(&net.out_bias.to_le_bytes());
    debug_assert_eq!(payload.len(), PAYLOAD_LEN);
    let checksum = fnv1a64(&payload);

    let mut out = Vec::with_capacity(HEADER_LEN + PAYLOAD_LEN);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&CENN_VERSION.to_le_bytes());
    out.extend_from_slice(&(INPUT as u32).to_le_bytes());
    out.extend_from_slice(&(L1 as u32).to_le_bytes());
    out.push(net.activation.id());
    out.extend_from_slice(&[0u8; 3]); // padding
    out.extend_from_slice(&net.qa.to_le_bytes());
    out.extend_from_slice(&net.qb.to_le_bytes());
    out.extend_from_slice(&net.scale.to_le_bytes());
    out.extend_from_slice(&[0u8; 6]); // padding
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&payload);
    debug_assert_eq!(out.len(), HEADER_LEN + PAYLOAD_LEN);
    out
}

#[inline]
fn read_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}
#[inline]
fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}
#[inline]
fn read_u64(bytes: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(b)
}

/// Parse and validate a CENN container (design/11 §3.4.5). Rejects every
/// anomaly with an [`NnueError`]; never panics on the input bytes.
pub fn load_bytes(bytes: &[u8]) -> Result<Network, NnueError> {
    if bytes.len() < HEADER_LEN {
        return Err(NnueError::TooShort {
            need: HEADER_LEN,
            got: bytes.len(),
        });
    }

    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != MAGIC {
        return Err(NnueError::BadMagic(magic));
    }
    let version = read_u32(bytes, 4);
    if version != CENN_VERSION {
        return Err(NnueError::UnsupportedVersion(version));
    }
    let input_dims = read_u32(bytes, 8);
    if input_dims as usize != INPUT {
        return Err(NnueError::WrongInputDims(input_dims));
    }
    let l1 = read_u32(bytes, 12);
    if l1 as usize != L1 {
        return Err(NnueError::WrongL1(l1));
    }
    let activation = Activation::from_id(bytes[16]).ok_or(NnueError::BadActivation(bytes[16]))?;
    let qa = read_u16(bytes, 20);
    if qa as i32 != QA {
        return Err(NnueError::WrongQuant {
            field: "qa",
            expected: QA as u16,
            found: qa,
        });
    }
    let qb = read_u16(bytes, 22);
    if qb as i32 != QB {
        return Err(NnueError::WrongQuant {
            field: "qb",
            expected: QB as u16,
            found: qb,
        });
    }
    let scale = read_u16(bytes, 24);
    if scale as i32 != SCALE {
        return Err(NnueError::WrongQuant {
            field: "scale",
            expected: SCALE as u16,
            found: scale,
        });
    }
    let payload_len = read_u64(bytes, 32);
    if payload_len != PAYLOAD_LEN as u64 {
        return Err(NnueError::BadPayloadLen {
            expected: PAYLOAD_LEN as u64,
            found: payload_len,
        });
    }
    let header_checksum = read_u64(bytes, 40);

    if bytes.len() < HEADER_LEN + PAYLOAD_LEN {
        return Err(NnueError::TooShort {
            need: HEADER_LEN + PAYLOAD_LEN,
            got: bytes.len(),
        });
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + PAYLOAD_LEN];
    let computed = fnv1a64(payload);
    if computed != header_checksum {
        return Err(NnueError::ChecksumMismatch {
            expected: header_checksum,
            found: computed,
        });
    }

    // The payload length is exact (validated above), so these reads are in
    // bounds by construction.
    let mut cur = 0usize;
    let read_i16s = |cur: &mut usize, count: usize| -> Box<[i16]> {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(i16::from_le_bytes([payload[*cur], payload[*cur + 1]]));
            *cur += 2;
        }
        v.into_boxed_slice()
    };
    let ft_weights = read_i16s(&mut cur, INPUT * L1);
    let ft_bias = read_i16s(&mut cur, L1);
    let out_weights = read_i16s(&mut cur, 2 * L1);
    let out_bias = i32::from_le_bytes([
        payload[cur],
        payload[cur + 1],
        payload[cur + 2],
        payload[cur + 3],
    ]);

    Ok(Network {
        ft_weights,
        ft_bias,
        out_weights,
        out_bias,
        activation,
        qa,
        qb,
        scale,
    })
}

/// Load and validate a CENN net from a file (design/11 §3.4.5).
pub fn load(path: impl AsRef<Path>) -> Result<Network, NnueError> {
    let bytes = std::fs::read(path).map_err(NnueError::Io)?;
    load_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_net() -> Network {
        Network::deterministic_test_net(0x0051_EED5, Activation::SCReLU)
    }

    // design/11 §3.4.5: serialize → load round-trips exactly, and the bytes
    // have the documented shape.
    #[test]
    fn round_trip_is_identity() {
        let net = test_net();
        let bytes = serialize(&net);
        assert_eq!(bytes.len(), HEADER_LEN + PAYLOAD_LEN);
        assert_eq!(&bytes[0..4], b"CENN");
        assert_eq!(read_u32(&bytes, 4), CENN_VERSION);
        assert_eq!(read_u32(&bytes, 8), INPUT as u32);
        assert_eq!(read_u32(&bytes, 12), L1 as u32);
        assert_eq!(bytes[16], Activation::SCReLU.id());
        let back = load_bytes(&bytes).expect("valid net loads");
        assert_eq!(net, back);

        // CReLU nets round-trip too.
        let crelu = Network::deterministic_test_net(9, Activation::CReLU);
        assert_eq!(load_bytes(&serialize(&crelu)).unwrap(), crelu);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize(&test_net());
        bytes[0] = b'X';
        assert!(matches!(load_bytes(&bytes), Err(NnueError::BadMagic(_))));
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = serialize(&test_net());
        bytes[4] = 2;
        assert!(matches!(
            load_bytes(&bytes),
            Err(NnueError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn rejects_wrong_dims() {
        let mut bytes = serialize(&test_net());
        bytes[8..12].copy_from_slice(&769u32.to_le_bytes());
        assert!(matches!(
            load_bytes(&bytes),
            Err(NnueError::WrongInputDims(769))
        ));

        let mut bytes = serialize(&test_net());
        bytes[12..16].copy_from_slice(&512u32.to_le_bytes());
        assert!(matches!(load_bytes(&bytes), Err(NnueError::WrongL1(512))));
    }

    #[test]
    fn rejects_bad_activation() {
        let mut bytes = serialize(&test_net());
        bytes[16] = 7;
        assert!(matches!(
            load_bytes(&bytes),
            Err(NnueError::BadActivation(7))
        ));
    }

    #[test]
    fn rejects_wrong_quant() {
        let mut bytes = serialize(&test_net());
        bytes[20..22].copy_from_slice(&123u16.to_le_bytes());
        assert!(matches!(
            load_bytes(&bytes),
            Err(NnueError::WrongQuant { field: "qa", .. })
        ));
    }

    #[test]
    fn rejects_truncated_payload() {
        let bytes = serialize(&test_net());
        // Drop the last few payload bytes; the header still declares the full
        // payload_len, so the loader must see it as too short.
        let truncated = &bytes[..bytes.len() - 4];
        assert!(matches!(
            load_bytes(truncated),
            Err(NnueError::TooShort { .. })
        ));
        // A stub shorter than even the header is TooShort too.
        assert!(matches!(
            load_bytes(&bytes[..10]),
            Err(NnueError::TooShort { .. })
        ));
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut bytes = serialize(&test_net());
        // Flip a payload byte but leave the stored checksum stale.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(matches!(
            load_bytes(&bytes),
            Err(NnueError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn error_displays_and_has_io_source() {
        // Display is non-empty and Error::source wires the io error through.
        let e = load(std::path::Path::new("/definitely/not/here.cenn")).unwrap_err();
        assert!(matches!(e, NnueError::Io(_)));
        assert!(!e.to_string().is_empty());
        assert!(e.source().is_some());

        let e = NnueError::BadMagic([0, 1, 2, 3]);
        assert!(e.to_string().contains("magic"));
        assert!(e.source().is_none());
    }

    #[test]
    fn fnv1a64_known_vectors() {
        // Canonical FNV-1a-64 test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
