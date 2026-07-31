//! The SOTA network container: parse, validate, and serialize D40 SOTA nets
//! (design/11 §12.6). Two versions are readable — `CEN2` and `CEN5` — and both
//! produce the same [`SotaNetwork`].
//!
//! Completely separate from the v1 CENN loader ([`super::loader`]) — the arch
//! constants, payload shape, and returned type (`SotaNetwork` vs `Network`) are
//! all different. The magic keeps every container family apart: `CENN` is v1,
//! `CEN3` is [`super::simple_halfka`], `CEN4` is [`super::protected_halfka`],
//! and this module owns `CEN2` and `CEN5`. Each loader rejects the others with
//! `BadMagic`.
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
//! **CENN-v5 (`CEN5`, perf study 30) — the same container plus `threat_quant`.**
//! The v2 header is FULL: every one of its 64 bytes is spoken for, with no
//! reserved slot, which is why adding one field costs a magic. v5 therefore
//! keeps bytes `0..64` byte-for-byte identical in meaning (only the magic and
//! version differ) and APPENDS, so the two versions share one parser for
//! everything above and a v2 net cannot be perturbed by the change:
//! ```text
//! CENN-v5 header (72 bytes) = the v2 header, with
//! 0       4     magic = b"CEN5"
//! 4       2     version: u16 LE (= 5)
//! ...           (8..64 exactly as CENN-v2)
//! 64      4     threat_quant: i32 — power of two in 1..=256
//! 68      4     reserved: u32, MUST be 0
//! ```
//! and the payload starts at 72 instead of 64.
//!
//! `threat_quant` is the scale at which the FullThreats table was exported to
//! i8, and it is a property of the NET, not of the engine: a non-QAT net's
//! weights span ±1.98 and clip catastrophically above 64, while a QAT net
//! trained at `qat_threat_scale = 256` is clipped to ±127/256 and wastes two
//! bits of the i8 at 64 (measured: `max|w| 32/127`, 65 of 256 levels, against
//! 127/127 and 247/256 for a healthy net — and −128.7 ± 9.6 Elo over 1600
//! calibrated games). A `CEN2` file has no field for it, so it means
//! [`sota_head::THREAT_QUANT_CEN2`] = 64 — which is exactly what every engine
//! before this change assumed, so **loading a v2 net is bit-identical**.
//! The runtime consumes it as [`sota_head::threat_shift_for`].
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
//! **Optional PSQ1 trailer (T6c FT PSQT skip; contract pinned in
//! `jax/archon_jax/checkpoint.py`).** After the payload the container MAY carry
//! ```text
//! b"PSQ1" | u32 num_buckets (= out_buckets) | u32 rows | i32[rows · num_buckets]
//! ```
//! row-major, where `rows = halfka_dims + threat_dims` (HalfKA rows FIRST, then
//! threats; a no-threats net carries only the HalfKA rows). Row indexing is the
//! engine's canonical feature index: HalfKA feature `f` → row `f`, threat
//! feature `t` → row `halfka_dims + t`. Each stored value is
//! `round_half_away(w · 0.5 · 400 · 256)` — the stm-canonical 0.5 and the cp
//! scale 400 are folded in, 8 fractional bits — so the head adds
//! `(psqt_stm[b] − psqt_ntm[b]) >> 8` centipawns (arithmetic shift). The
//! section rides OUTSIDE the header checksum (like its quant.bin counterpart);
//! nets without it load and evaluate exactly as before.
//!
//! Parsing NEVER panics on user-supplied bytes — every anomaly is a
//! [`SotaNnueError`]. The loader validates every dimension against the
//! compile-time architecture constants from [`super::sota_head`] and
//! [`super::sota_features`].

use super::aligned::{Aligned, Blocks};
use super::sota_features::{ACC, HALFKA_DIMS, THREAT_DIMS, ThreatRemap};
use super::sota_head::{
    self, BUCKETS, L1, L1Biases, L1Weights, L2, L2Biases, L2Weights, L3, L3Weights, SotaHead,
};
use std::error::Error;
use std::fmt;
use std::path::Path;

/// CENN-v2 magic bytes.
pub const MAGIC_V2: [u8; 4] = *b"CEN2";
/// Container version.
pub const CENN_V2_VERSION: u16 = 2;
/// Fixed header size in bytes.
pub const HEADER_V2_LEN: usize = 64;
/// CENN-v5 magic bytes — the same container plus a `threat_quant` field.
///
/// **Five, not three**: `CEN3` and `CEN4` are already taken by
/// [`super::simple_halfka`] and [`super::protected_halfka`], both routed by
/// `uci.rs`'s magic peek, so reusing `CEN3` would misroute a SOTA net into the
/// HalfKA loader — precisely the collision the magic exists to prevent.
pub const MAGIC_V5: [u8; 4] = *b"CEN5";
/// Container version for [`MAGIC_V5`].
pub const CENN_V5_VERSION: u16 = 5;
/// Fixed header size in bytes for [`MAGIC_V5`]: the v2 header plus
/// `threat_quant: i32` at 64 and a zero `reserved: u32` at 68.
pub const HEADER_V5_LEN: usize = 72;
/// Flag bit: threats feature transformer is present.
pub const FLAG_HAS_THREATS: u16 = 1;
/// Magic bytes of the optional trailing PSQT-skip section (T6c).
pub const PSQT_MAGIC: [u8; 4] = *b"PSQ1";
/// SKP1 magic (D47V output-skip lane; contract: jax checkpoint.py
/// skip_section_bytes — b"SKP1" | u32 num_buckets | u32 hl | f32[hl·buckets],
/// row-major, raw f32; rides AFTER the PSQ1 section).
pub const SKIP_MAGIC: [u8; 4] = *b"SKP1";
/// TRM1 magic (the Recode POC threat re-coding,
/// `docs/research/learned-feature-code-study-2026-07-30.md` §6.1 mode (ii)):
/// `b"TRM1" | u32 dims (= THREAT_DIMS) | u32 rows | u16[dims]` little-endian,
/// riding after the SKP1 section (or after PSQ1 / the payload when the earlier
/// sections are absent). When present, the header's `threat_dims` field
/// carries `rows` — the actual threat weight-table row count, `1..=60720` —
/// and every remap entry must be `< rows`. A net whose header says 60,720 may
/// still carry a TRM1 (the identity/shuffle instrumentation arms); a net whose
/// header says less MUST.
pub const REMAP_MAGIC: [u8; 4] = *b"TRM1";
/// PSQ1 section header length: magic + u32 num_buckets + u32 rows.
const PSQT_HEADER_LEN: usize = 12;

// Compile-time assertions: the head and feature constants must agree.
const _: () = assert!(
    L1 == ACC,
    "SotaHead::L1 must equal ACC (the accumulator width)"
);

/// A D40 SOTA network: two feature transformers + multi-layer bucketed head.
/// Produced by [`load_v2_bytes`]; consumed by the SOTA eval context.
#[derive(Debug)]
pub struct SotaNetwork {
    /// HalfKAv2_hm FT weights, feature-major: `[HALFKA_DIMS · ACC]` **i8**.
    ///
    /// The CENN-v2 payload carries these as i16 in the `FT_QUANT = 255` frame;
    /// [`quantise_halfka`] halves them at load, one integer scale per
    /// accumulator column ([`sota_head::SotaHead::ft_scale`]). The FORMAT is
    /// unchanged — every existing net loads — and the 57.7 MB table the fold
    /// streams cold on every node becomes 28.9 MB, which is the whole point
    /// (perf study 32; study 31 §4.3 priced it at −463 cycles/node).
    pub halfka_weights: Aligned<i8>,
    /// Shared FT bias: `[ACC]` i16 (seeds the HalfKA accumulator; threats seed 0).
    pub ft_bias: Aligned<i16>,
    /// FullThreats FT weights, feature-major: `[threat_rows · ACC]` i8.
    /// Empty if the net has no threats channel.
    pub threat_weights: Aligned<i8>,
    /// Number of rows in [`threat_weights`](Self::threat_weights):
    /// `THREAT_DIMS` for every identity-coded net, the compressed row count
    /// for a TRM1-carrying net, 0 for a threat-less net.
    pub threat_rows: usize,
    /// The threat re-coding table (TRM1 section), or `None` = identity. See
    /// [`super::sota_features::ThreatRemap`]. A sized box, so the hot paths
    /// carry a thin pointer and fixed-size indexing.
    pub threat_remap: Option<Box<ThreatRemap>>,
    /// T6c FT PSQT skip table, row-major `[rows · BUCKETS]` i32 with the HalfKA
    /// rows first, then the threat rows (module docs). Empty when the net has
    /// no PSQ1 section — every psqt path is then inert and the eval is
    /// bit-identical to a pre-T6c build.
    pub psqt_weights: Aligned<i32>,
    /// The multi-layer bucketed head (owns its own weight boxes).
    pub head: SotaHead,
}

impl SotaNetwork {
    /// The HalfKA slice of the PSQT skip table (`[HALFKA_DIMS · BUCKETS]`),
    /// indexed by `halfka_index · BUCKETS + bucket`; empty when the net has no
    /// skip.
    pub fn halfka_psqt(&self) -> &[i32] {
        if self.psqt_weights.is_empty() {
            &[]
        } else {
            &self.psqt_weights[..HALFKA_DIMS * BUCKETS]
        }
    }

    /// The threat slice of the PSQT skip table (`[THREAT_DIMS · BUCKETS]`),
    /// indexed by `threat_index · BUCKETS + bucket`; empty when the net has no
    /// skip or no threats channel.
    pub fn threat_psqt(&self) -> &[i32] {
        if self.psqt_weights.len() > HALFKA_DIMS * BUCKETS {
            &self.psqt_weights[HALFKA_DIMS * BUCKETS..]
        } else {
            &[]
        }
    }
}

/// `PartialEq`/`Eq` are implemented for **bit-exact round-trip testing only**.
/// The f32 head weights are compared with `==`, which is sound for serializer
/// round-trips (bit-identical bytes → bit-identical f32) but would be incorrect
/// in contexts where NaN is possible. Do not use for fuzzy comparison.
impl PartialEq for SotaNetwork {
    fn eq(&self, other: &Self) -> bool {
        self.halfka_weights == other.halfka_weights
            && self.head.ft_scale == other.head.ft_scale
            && self.ft_bias == other.ft_bias
            && self.threat_weights == other.threat_weights
            && self.threat_rows == other.threat_rows
            && self.threat_remap == other.threat_remap
            && self.psqt_weights == other.psqt_weights
            && self.head.network_scale == other.head.network_scale
            && self.head.threat_shift == other.head.threat_shift
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
    /// The first four bytes were neither `b"CEN2"` nor `b"CEN5"`.
    BadMagic([u8; 4]),
    /// Version mismatch.
    UnsupportedVersion(u16),
    /// A `CEN5` header's `threat_quant` is not a power of two in `1..=256`, so
    /// the runtime cannot carry `FT_QUANT/threat_quant` as a shift.
    BadThreatQuant(i32),
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
    /// A TRM1 remap entry points past the net's threat row count, or a
    /// compressed threat table arrived without its TRM1 section.
    BadRemap { detail: &'static str },
    /// Payload checksum mismatch.
    ChecksumMismatch { expected: u64, found: u64 },
}

impl fmt::Display for SotaNnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SotaNnueError::Io(e) => write!(f, "reading SOTA NNUE file: {e}"),
            SotaNnueError::TooShort { need, got } => {
                write!(f, "CENN too short: need {need} bytes, got {got}")
            }
            SotaNnueError::BadMagic(m) => {
                write!(f, "CENN bad magic: {m:?} (expected b\"CEN2\" or b\"CEN5\")")
            }
            SotaNnueError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "CENN unsupported version {v} (this build reads \
                     {CENN_V2_VERSION} and {CENN_V5_VERSION})"
                )
            }
            SotaNnueError::BadThreatQuant(q) => write!(
                f,
                "CENN threat_quant {q} is not a power of two in 1..=256 \
                 (FT_QUANT/threat_quant must be carriable as a shift)"
            ),
            SotaNnueError::WrongDim {
                field,
                expected,
                found,
            } => write!(
                f,
                "CENN {field} {found} != {expected} (architecture mismatch)"
            ),
            SotaNnueError::WrongQuant {
                field,
                expected,
                found,
            } => write!(f, "CENN {field} {found} != {expected} (quant mismatch)"),
            SotaNnueError::BadPayloadLen { expected, found } => write!(
                f,
                "CENN payload_len {found} != {expected} (architecture mismatch)"
            ),
            SotaNnueError::BadRemap { detail } => {
                write!(f, "CENN TRM1 threat remap invalid: {detail}")
            }
            SotaNnueError::ChecksumMismatch { expected, found } => write!(
                f,
                "CENN checksum mismatch: header {expected:#018x}, computed {found:#018x}"
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

/// Payload byte length for a net WITH threats (`threat_rows` weight rows —
/// `THREAT_DIMS` unless a TRM1 re-coding shrinks the table).
fn payload_len_with_threats(threat_rows: usize) -> usize {
    let halfka_bytes = HALFKA_DIMS * ACC * 2; // i16
    let bias_bytes = ACC * 2; // i16
    let threat_bytes = threat_rows * ACC; // i8
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
    halfka_bytes
        + bias_bytes
        + l1w_bytes
        + l1b_bytes
        + l2w_bytes
        + l2b_bytes
        + l3w_bytes
        + l3b_bytes
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

/// Parse and validate a `CEN2` or `CEN5` container. Rejects every anomaly with
/// a [`SotaNnueError`]; never panics on the input bytes.
pub fn load_v2_bytes(bytes: &[u8]) -> Result<SotaNetwork, SotaNnueError> {
    // ── Header ─────────────────────────────────────────────────────────
    // The shorter of the two headers first, so a truncated file still reports
    // TooShort rather than reading past its end for the magic.
    if bytes.len() < HEADER_V2_LEN {
        return Err(SotaNnueError::TooShort {
            need: HEADER_V2_LEN,
            got: bytes.len(),
        });
    }

    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    // Bytes 0..64 mean the same thing in both versions (module docs), so only
    // the header LENGTH and the trailing threat_quant differ here.
    let (expected_version, header_len) = match magic {
        MAGIC_V2 => (CENN_V2_VERSION, HEADER_V2_LEN),
        MAGIC_V5 => (CENN_V5_VERSION, HEADER_V5_LEN),
        _ => return Err(SotaNnueError::BadMagic(magic)),
    };
    if bytes.len() < header_len {
        return Err(SotaNnueError::TooShort {
            need: header_len,
            got: bytes.len(),
        });
    }

    let version = read_u16(bytes, 4);
    if version != expected_version {
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
    // The threat_dims field carries the threat weight-table ROW count. For
    // every net before the TRM1 re-coding that is exactly THREAT_DIMS (or 0
    // without the flag); a compressed net says how many rows it actually
    // carries, and the TRM1 trailer (validated below) maps the 60,720
    // geometric codes onto them.
    let threat_rows = read_u32(bytes, 12) as usize;
    if has_threats {
        if !(1..=THREAT_DIMS).contains(&threat_rows) {
            return Err(SotaNnueError::WrongDim {
                field: "threat_dims",
                expected: THREAT_DIMS as u32,
                found: threat_rows as u32,
            });
        }
    } else {
        check_dim(12, "threat_dims", 0)?;
    }
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
    if !(1..=10_000).contains(&network_scale) {
        return Err(SotaNnueError::WrongDim {
            field: "network_scale",
            expected: 380,
            found: network_scale as u32,
        });
    }

    // threat_quant: read from a v5 header, IMPLIED by a v2 one. This is the
    // whole point of the version bump — see the module docs.
    let threat_quant = if header_len == HEADER_V5_LEN {
        let reserved = read_u32(bytes, 68);
        if reserved != 0 {
            return Err(SotaNnueError::WrongDim {
                field: "reserved",
                expected: 0,
                found: reserved,
            });
        }
        read_i32(bytes, 64)
    } else {
        sota_head::THREAT_QUANT_CEN2
    };
    let threat_shift = sota_head::threat_shift_for(threat_quant)
        .ok_or(SotaNnueError::BadThreatQuant(threat_quant))?;

    // Payload length + checksum.
    let expected_payload = if has_threats {
        payload_len_with_threats(threat_rows)
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
    let total_need = header_len + expected_payload;
    if bytes.len() < total_need {
        return Err(SotaNnueError::TooShort {
            need: total_need,
            got: bytes.len(),
        });
    }
    let payload = &bytes[header_len..header_len + expected_payload];
    let computed_checksum = fnv1a64(payload);
    if computed_checksum != header_checksum {
        return Err(SotaNnueError::ChecksumMismatch {
            expected: header_checksum,
            found: computed_checksum,
        });
    }

    // All reads below are in-bounds by construction (payload length validated).
    let mut cur = 0usize;

    // HalfKA FT weights: i16[HALFKA_DIMS · ACC] on disk, feature-major,
    // QUANTISED TO i8 on the way in (perf study 32 — see `quantise_halfka`).
    // Read straight from the payload bytes rather than through `read_i16s`:
    // materialising the 57.7 MB i16 table first would double the loader's peak
    // footprint, and a screen runs 2 x concurrency of these processes at once.
    let n_halfka = HALFKA_DIMS * ACC;
    let halfka_raw = &payload[cur..cur + n_halfka * 2];
    cur += n_halfka * 2;
    let ft_scale = halfka_column_scales(halfka_raw);
    let halfka_weights = quantise_halfka(halfka_raw, ft_scale.block(0));

    // FT bias: i16[ACC], rebased onto the same per-column grid so the
    // accumulator seed and the rows share one frame.
    let ft_bias_raw = read_i16s(payload, &mut cur, ACC);
    let ft_bias: Aligned<i16> = ft_bias_raw
        .iter()
        .zip(ft_scale.block(0).iter())
        .map(|(&b, &s)| round_half_away(i32::from(b), i32::from(s)) as i16)
        .collect();

    // Threats FT weights: i8[threat_rows · ACC], feature-major (if present).
    let threat_weights = if has_threats {
        read_i8s_aligned(payload, &mut cur, threat_rows * ACC)
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };

    // The head's five bucket-major tables. Read as `Blocks`, not as one flat
    // aligned run: the bucket stride is fixed at load, so it belongs in the
    // type rather than in a range slice re-derived on every eval (perf study
    // 26 §3). The on-disk layout is unchanged — `Blocks` is block-major
    // contiguous with no interior padding, which `Blocks::LAYOUT_OK` asserts
    // at compile time.
    //
    // L1 weights: i8[BUCKETS][L2 · L1], group-interleaved within a bucket.
    let l1_weights: L1Weights = read_i8_blocks(payload, &mut cur);
    // L1 biases: f32[BUCKETS][L2]. Every block starts on a cache line now,
    // not just the table (perf study 21 arm B4 fixed the base; this fixes all
    // BUCKETS of them by construction).
    let l1_biases: L1Biases = read_f32_blocks(payload, &mut cur);
    // L2 weights: f32[BUCKETS][L2 · L3].
    let l2_weights: L2Weights = read_f32_blocks(payload, &mut cur);
    // L2 biases: f32[BUCKETS][L3].
    let l2_biases: L2Biases = read_f32_blocks(payload, &mut cur);
    // L3 weights: f32[BUCKETS][L3].
    let l3_weights: L3Weights = read_f32_blocks(payload, &mut cur);

    // L3 biases: f32[BUCKETS].
    let l3_biases = read_f32s(payload, &mut cur, BUCKETS);

    debug_assert_eq!(
        cur, expected_payload,
        "payload cursor must consume all bytes"
    );

    // ── Optional PSQ1 trailer (T6c FT PSQT skip; module docs) ──────────
    // Only bytes that START with the PSQ1 magic are interpreted; any other
    // trailing bytes are ignored exactly as this loader always has.
    let rest = &bytes[total_need..];
    // ★ PERF STUDY 48 — `nopsqt48` is a SUBTRACTIVE DIAGNOSTIC PROBE, never a
    // shipping configuration. It drops the PSQ1 trailer at load, which switches
    // off every psqt path (both channels hand-unswitch on
    // `psqt_weights.is_empty()`), so differencing it against the base prices
    // the WHOLE psqt channel: 7.18 random 32-byte reads per node out of a
    // 2.7 MB table that does not fit beside everything else in a 1 MB L2.
    //
    // It changes the EVAL, so the search tree differs — but `bench … nodes`
    // pins the node count, so per-node counters stay comparable. Read it as a
    // CEILING on any psqt-layout arm, not as a speedup.
    let psqt_weights = if rest.len() >= 4 && rest[0..4] == PSQT_MAGIC && !cfg!(feature = "nopsqt48")
    {
        parse_psqt_section(rest, threat_rows)?
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };
    // D47V: optional SKP1 trailer after PSQ1 (or directly, on a psqt-less
    // net). Empty slice = no skip lane = the pre-D47V eval, bit-identical.
    let psqt_len = if psqt_weights.is_empty() {
        0
    } else {
        PSQT_HEADER_LEN + psqt_weights.len() * 4
    };
    let tail = &rest[psqt_len.min(rest.len())..];
    let skip_weights = if tail.len() >= 4 && tail[0..4] == SKIP_MAGIC {
        parse_skip_section(tail)?
    } else {
        Vec::new().into_boxed_slice()
    };
    // Recode POC: optional TRM1 trailer after SKP1 (or directly after PSQ1 /
    // the payload). Mandatory when the header's threat row count is
    // compressed; `None` = the identity code = every net before this change.
    let threat_remap = {
        let skip_len = if skip_weights.is_empty() {
            0
        } else {
            12 + skip_weights.len() * 4
        };
        let tail2 = &tail[skip_len.min(tail.len())..];
        if tail2.len() >= 4 && tail2[0..4] == REMAP_MAGIC {
            Some(parse_remap_section(tail2, threat_rows)?)
        } else {
            None
        }
    };
    if has_threats && threat_rows != THREAT_DIMS && threat_remap.is_none() {
        return Err(SotaNnueError::BadRemap {
            detail: "compressed threat table (threat_dims < 60720) without a TRM1 section",
        });
    }

    let head = SotaHead {
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        l3_weights,
        l3_biases,
        network_scale,
        threat_shift,
        l3_var_weights: None,
        l3_var_biases: None,
        l3_wdl_weights: None,
        l3_wdl_biases: None,
        skip_weights: (!skip_weights.is_empty()).then(|| quantise_skip_lane(&skip_weights)),
        ft_scale,
    };

    Ok(SotaNetwork {
        halfka_weights,
        ft_bias,
        threat_weights,
        threat_rows: if has_threats { threat_rows } else { 0 },
        threat_remap,
        psqt_weights,
        head,
    })
}

/// Parse a TRM1 section (starting at its magic): the threat re-coding table.
/// Strict: `dims` must equal [`THREAT_DIMS`], `rows` must equal the header's
/// threat row count, and every entry must be `< rows` — that validation is
/// what lets the fold paths index the weight table without per-event bounds
/// checks. Tolerant of further trailing bytes (future sections).
fn parse_remap_section(sec: &[u8], threat_rows: usize) -> Result<Box<ThreatRemap>, SotaNnueError> {
    const HDR: usize = 12;
    if sec.len() < HDR {
        return Err(SotaNnueError::TooShort {
            need: HDR,
            got: sec.len(),
        });
    }
    let dims = read_u32(sec, 4);
    if dims as usize != THREAT_DIMS {
        return Err(SotaNnueError::WrongDim {
            field: "remap_dims",
            expected: THREAT_DIMS as u32,
            found: dims,
        });
    }
    let rows = read_u32(sec, 8);
    if rows as usize != threat_rows {
        return Err(SotaNnueError::WrongDim {
            field: "remap_rows",
            expected: threat_rows as u32,
            found: rows,
        });
    }
    let need = HDR + THREAT_DIMS * 2;
    if sec.len() < need {
        return Err(SotaNnueError::TooShort {
            need,
            got: sec.len(),
        });
    }
    let mut table = Box::new([0u16; THREAT_DIMS]);
    for (dst, src) in table.iter_mut().zip(sec[HDR..need].chunks_exact(2)) {
        let v = u16::from_le_bytes([src[0], src[1]]);
        if usize::from(v) >= threat_rows {
            return Err(SotaNnueError::BadRemap {
                detail: "remap entry >= threat row count",
            });
        }
        *dst = v;
    }
    Ok(table)
}

/// Parse an SKP1 section (starting at its magic). Strict on this build's
/// [`BUCKETS`] and accumulator width [`crate::eval::nnue::sota_head::L1`];
/// tolerant of further trailing bytes (future sections).
/// Round `n / d` half-away-from-zero in integer arithmetic (`d > 0`).
///
/// The project's quantisation convention (`_round_half_away` in the trainer),
/// and it matters here rather than being cosmetic: at `scale = 2` HALF the
/// weights land exactly on a tie, so a half-to-even rule would move a quarter
/// of that column's table by one step.
#[inline]
fn round_half_away(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    let q = (2 * n.abs() + d) / (2 * d);
    if n < 0 { -q } else { q }
}

/// Per-accumulator-column integer scale for the i8 HalfKA table: the smallest
/// `s` for which every weight in the column fits `[-127, 127]` after division.
///
/// **Why per-column and why no clipping.** Measured offline over the 800-FEN
/// battery (perf study 32 §1): one GLOBAL scale of 4 costs p95 6.76 / p99 10.15
/// SF cp; per-column no-clip costs **p95 4.35 / p99 7.25**; and a per-column
/// scale tuned to minimise weight MSE — which clips 0.008% of the table — is
/// BETTER on weight MSE (1.78% vs 1.98%) and WORSE on eval error (p95 4.83).
/// The weight-MSE proxy misranks the schemes, so the no-clip rule is the one
/// the eval battery chose, not the one the L2 norm chose.
fn halfka_column_scales(raw: &[u8]) -> sota_head::FtScale {
    let mut peak = vec![0u16; ACC];
    for row in raw.chunks_exact(ACC * 2) {
        for (p, w) in peak.iter_mut().zip(row.chunks_exact(2)) {
            let v = i16::from_le_bytes([w[0], w[1]]).unsigned_abs();
            *p = (*p).max(v);
        }
    }
    // `div_ceil` is the no-clip condition; the `max(1)` covers an all-zero
    // column (1024 of the 22528 HalfKA rows are all-zero in the canonical net,
    // and three whole columns are), where any scale is exact and 1 keeps the
    // reduced frame the identity.
    sota_head::FtScale::from_flat_iter(
        peak.iter()
            // Perf study 50: stored as u8. The saturation is one notch
            // tighter than the `i16::MAX` it replaces and is unreachable for
            // any net that can work: a column scale above 255 needs a peak
            // above 32,385 in the FT_QUANT frame, and the reduced frame then
            // requires |acc| <= 127 for `acc * scale` to stay in i16 — which
            // the `activate` debug_assert already enforces and which no
            // accumulator summing dozens of rows satisfies. Canonical net:
            // 8 distinct values, max 8 (`FOLDVALHIST ft_scale`).
            .map(|&p| u32::from(p).div_ceil(127).clamp(1, u8::MAX as u32) as u8),
    )
}

/// Divide the FT_QUANT-frame i16 table by its per-column scale into i8.
fn quantise_halfka(raw: &[u8], scale: &[u8; ACC]) -> Aligned<i8> {
    Aligned::from_iter_len(
        raw.len() / 2,
        raw.chunks_exact(2).enumerate().map(|(i, w)| {
            let v = i32::from(i16::from_le_bytes([w[0], w[1]]));
            round_half_away(v, i32::from(scale[i % ACC])).clamp(-127, 127) as i8
        }),
    )
}

/// Quantise the D47V output-skip lane to i8, one scale per bucket
/// (perf study 8 §4.7).
///
/// `w[b][i] ≈ q[b·L1 + i] · scale[b]`, with `scale[b] = max|w[b]| / 127` so each
/// bucket spends its full i8 range on its own dynamic range. The lane arrives
/// bucket-major and contiguous from [`parse_skip_section`], so a bucket is one
/// slice.
///
/// This is a lossy step and it is measured, not asserted: the study reports the
/// cp deltas over the 800-FEN battery. It buys a ~6x cheaper lane — `vpdpbusd`
/// takes 64 u8 x i8 pairs per instruction, where the f32 form needed a widen, a
/// convert and an FMA per 16 lanes — for the one channel Stockfish gets free as
/// the `FC_0_OUTPUTS + 1`-th row of the sparse matmul it already runs.
fn quantise_skip_lane(w: &[f32]) -> sota_head::SkipLane {
    let hl = w.len() / BUCKETS;
    // Built as a plain `Vec` and converted once: the i8 lane is read by
    // `skip_dot_i8` as 32-byte loads, so it is one of the tables that has to
    // start on a cache line (perf study 20 §5).
    let mut q = vec![0i8; w.len()];
    let mut scale = vec![0.0f32; BUCKETS].into_boxed_slice();
    for b in 0..BUCKETS {
        let row = &w[b * hl..(b + 1) * hl];
        let peak = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // An all-zero bucket keeps scale 0 and q 0 — it contributes nothing
        // either way, and this avoids a 0/0.
        let s = if peak > 0.0 { peak / 127.0 } else { 0.0 };
        scale[b] = s;
        if s > 0.0 {
            for (dst, &v) in q[b * hl..(b + 1) * hl].iter_mut().zip(row) {
                *dst = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    sota_head::SkipLane { q: q.into(), scale }
}

fn parse_skip_section(sec: &[u8]) -> Result<Box<[f32]>, SotaNnueError> {
    const HDR: usize = 12;
    if sec.len() < HDR {
        return Err(SotaNnueError::TooShort {
            need: HDR,
            got: sec.len(),
        });
    }
    let num_buckets = read_u32(sec, 4);
    if num_buckets as usize != BUCKETS {
        return Err(SotaNnueError::WrongDim {
            field: "skip_buckets",
            expected: BUCKETS as u32,
            found: num_buckets,
        });
    }
    let hl = read_u32(sec, 8);
    if hl as usize != crate::eval::nnue::sota_head::L1 {
        return Err(SotaNnueError::WrongDim {
            field: "skip_hl",
            expected: crate::eval::nnue::sota_head::L1 as u32,
            found: hl,
        });
    }
    let count = hl as usize * BUCKETS;
    let need = HDR + count * 4;
    if sec.len() < need {
        return Err(SotaNnueError::TooShort {
            need,
            got: sec.len(),
        });
    }
    // §6-S2a (perf study): transpose the file's row-major [hl][BUCKETS] to
    // bucket-major [BUCKETS][hl] at load, so the per-eval lane is hl
    // contiguous f32 instead of a stride-BUCKETS gather. Values and the
    // head's summation order are unchanged — bit-identical.
    let hl = hl as usize;
    let mut out = vec![0.0f32; count];
    for i in 0..hl {
        for b in 0..BUCKETS {
            let off = HDR + (i * BUCKETS + b) * 4;
            out[b * hl + i] = f32::from_le_bytes(sec[off..off + 4].try_into().unwrap());
        }
    }
    Ok(out.into_boxed_slice())
}

/// Parse a PSQ1 section (starting at its magic). Strict: the bucket count must
/// equal this build's [`BUCKETS`], the row count must equal the net's feature
/// space (`HALFKA_DIMS` plus `THREAT_DIMS` iff threats), and the full
/// `rows · BUCKETS` i32 table must be present.
fn parse_psqt_section(sec: &[u8], threat_rows: usize) -> Result<Aligned<i32>, SotaNnueError> {
    if sec.len() < PSQT_HEADER_LEN {
        return Err(SotaNnueError::TooShort {
            need: PSQT_HEADER_LEN,
            got: sec.len(),
        });
    }
    let num_buckets = read_u32(sec, 4);
    if num_buckets as usize != BUCKETS {
        return Err(SotaNnueError::WrongDim {
            field: "psqt_buckets",
            expected: BUCKETS as u32,
            found: num_buckets,
        });
    }
    // `threat_rows` is 0 for a threat-less net, the (possibly compressed)
    // weight-table row count otherwise — the PSQ1 threat rows live in the same
    // re-coded space as the weight rows.
    let expected_rows = HALFKA_DIMS + threat_rows;
    let rows = read_u32(sec, 8);
    if rows as usize != expected_rows {
        return Err(SotaNnueError::WrongDim {
            field: "psqt_rows",
            expected: expected_rows as u32,
            found: rows,
        });
    }
    let count = expected_rows * BUCKETS;
    let need = PSQT_HEADER_LEN + count * 4;
    if sec.len() < need {
        return Err(SotaNnueError::TooShort {
            need,
            got: sec.len(),
        });
    }
    let mut cur = PSQT_HEADER_LEN;
    Ok(read_i32s(sec, &mut cur, count))
}

/// Load and validate a CENN-v2 net from a file.
pub fn load_v2(path: impl AsRef<Path>) -> Result<SotaNetwork, SotaNnueError> {
    let bytes = std::fs::read(path).map_err(SotaNnueError::Io)?;
    load_v2_bytes(&bytes)
}

/// EXP-UNC-1 (search study bet 1): load the OPTIONAL value-uncertainty head into
/// `head` from the sidecar named by `D40_UNC_HEAD`, if set. Kept out of the CENN
/// (so the net format is unchanged) — the head is small (`BUCKETS·L3` weights +
/// `BUCKETS` biases) and rides a tiny companion file, extracted from the
/// trainer's raw checkpoint. Layout: flat little-endian f32, `l3_var_weights`
/// (bucket-major `[b·L3 + j]`, identical layout to `l3_weights`) then
/// `l3_var_biases` (`[b]`). Absent env or any read/shape error leaves the head
/// variance-less (σ̂ = `None`, every uncertainty consumer off) — so a net without
/// the sidecar is bit-identical to today. Called once at net-load time.
pub(crate) fn load_uncertainty_sidecar(head: &mut SotaHead) {
    let Some(path) = std::env::var_os("D40_UNC_HEAD") else {
        return;
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("D40_UNC_HEAD: cannot read {path:?}: {error}");
            return;
        }
    };
    let n_weights = BUCKETS * L3;
    let expected = (n_weights + BUCKETS) * 4;
    if bytes.len() != expected {
        eprintln!(
            "D40_UNC_HEAD: {path:?} is {} bytes, expected {expected} \
             ({n_weights} weights + {BUCKETS} biases, f32) — ignoring",
            bytes.len()
        );
        return;
    }
    let mut floats = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let weights: Vec<f32> = (&mut floats).take(n_weights).collect();
    let biases: Vec<f32> = floats.take(BUCKETS).collect();
    head.l3_var_weights = Some(weights.into_boxed_slice());
    head.l3_var_biases = Some(biases.into_boxed_slice());
    eprintln!("D40_UNC_HEAD: loaded value-uncertainty head from {path:?}");
}

/// D41 rich-net: load the co-trained WDL head from a `D40_WDL_HEAD` sidecar (flat
/// LE f32: `[3·L3]` weights row-major `[o·L3+j]`, o ∈ {L,D,W}, then `[3]` biases —
/// the layout `tools/nnue/extract_head.py l3_wdl` writes). Absent env ⇒ no head
/// (`wdl = None`) ⇒ bit-identical to today; the head is NOT in the CENN, so the
/// value net is unchanged. Called once at net-load, beside `load_uncertainty_sidecar`.
pub(crate) fn load_wdl_sidecar(head: &mut SotaHead) {
    let Some(path) = std::env::var_os("D40_WDL_HEAD") else {
        return;
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("D40_WDL_HEAD: cannot read {path:?}: {error}");
            return;
        }
    };
    let n_weights = 3 * L3;
    let expected = (n_weights + 3) * 4;
    if bytes.len() != expected {
        eprintln!(
            "D40_WDL_HEAD: {path:?} is {} bytes, expected {expected} \
             ({n_weights} weights + 3 biases, f32) — ignoring",
            bytes.len()
        );
        return;
    }
    let mut floats = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let weights: Vec<f32> = (&mut floats).take(n_weights).collect();
    let biases: Vec<f32> = floats.take(3).collect();
    head.l3_wdl_weights = Some(weights.into_boxed_slice());
    head.l3_wdl_biases = Some(biases.into_boxed_slice());
    eprintln!("D40_WDL_HEAD: loaded WDL head from {path:?}");
}

// ─── Serializer (tests + netpack cross-check) ──────────────────────────────

/// Serialize a `SotaNetwork` (the inverse of [`load_v2_bytes`]). Used by tests
/// to build deterministic nets and round-trip them; also lets the Rust side
/// produce a reference `.cenn` for cross-checking the Python packer.
///
/// **Emits the OLDEST container that can represent the net**: `CEN2` when the
/// threat frame is the one a v2 header implies
/// ([`sota_head::THREAT_QUANT_CEN2`]), `CEN5` otherwise. So every net that
/// round-tripped to v2 bytes before perf study 30 still does, byte for byte,
/// and only a net that genuinely needs the new field pays the version bump.
pub fn serialize(net: &SotaNetwork) -> Vec<u8> {
    let has_threats = !net.threat_weights.is_empty();
    let flags: u16 = if has_threats { FLAG_HAS_THREATS } else { 0 };
    let threat_quant = sota_head::threat_quant_for(net.head.threat_shift);
    let v5 = threat_quant != sota_head::THREAT_QUANT_CEN2;
    let header_len = if v5 { HEADER_V5_LEN } else { HEADER_V2_LEN };

    // ── Payload ────────────────────────────────────────────────────────
    let expected_payload = if has_threats {
        payload_len_with_threats(net.threat_rows)
    } else {
        payload_len_no_threats()
    };
    let mut payload = Vec::with_capacity(expected_payload);

    // HalfKA FT weights. The in-memory table is i8 + a per-column scale
    // (perf study 32), so the FT_QUANT-frame i16 the format carries is
    // reconstructed as `q · scale`. That is lossy against the ORIGINAL file —
    // exactly like the D47V skip lane, which the serializer simply drops — but
    // it is IDEMPOTENT: reloading these bytes derives the same scales (a
    // column's peak is `127 · scale` at most and never below `126 · scale`
    // unless the column is degenerate) and the same quotients, so
    // `load(serialize(load(x))) == load(x)`, which is what the round-trip
    // tests assert.
    for (i, &q) in net.halfka_weights.iter().enumerate() {
        let w = i16::from(q).wrapping_mul(i16::from(net.head.ft_scale.block(0)[i % ACC]));
        payload.extend_from_slice(&w.to_le_bytes());
    }
    // FT bias, back in the FT_QUANT frame.
    for (j, &b) in net.ft_bias.iter().enumerate() {
        let b = b.wrapping_mul(i16::from(net.head.ft_scale.block(0)[j]));
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
    let mut out = Vec::with_capacity(header_len + expected_payload);
    out.extend_from_slice(if v5 { &MAGIC_V5 } else { &MAGIC_V2 });
    out.extend_from_slice(&(if v5 { CENN_V5_VERSION } else { CENN_V2_VERSION }).to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(HALFKA_DIMS as u32).to_le_bytes());
    let threat_dim = if has_threats {
        net.threat_rows as u32
    } else {
        0
    };
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
    if v5 {
        out.extend_from_slice(&threat_quant.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    }
    debug_assert_eq!(out.len(), header_len);

    out.extend_from_slice(&payload);

    // Optional PSQ1 trailer (T6c): present only when the net carries a skip
    // table, so a psqt-less net serializes byte-identically to before.
    if !net.psqt_weights.is_empty() {
        debug_assert_eq!(net.psqt_weights.len() % BUCKETS, 0);
        out.extend_from_slice(&PSQT_MAGIC);
        out.extend_from_slice(&(BUCKETS as u32).to_le_bytes());
        let rows = net.psqt_weights.len() / BUCKETS;
        out.extend_from_slice(&(rows as u32).to_le_bytes());
        for &w in net.psqt_weights.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    // Optional TRM1 trailer (Recode POC): present only when the net carries a
    // threat re-coding, so every other net serializes byte-identically.
    if let Some(remap) = &net.threat_remap {
        out.extend_from_slice(&REMAP_MAGIC);
        out.extend_from_slice(&(THREAT_DIMS as u32).to_le_bytes());
        out.extend_from_slice(&(net.threat_rows as u32).to_le_bytes());
        for &r in remap.iter() {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    out
}

// ─── Payload element readers ───────────────────────────────────────────────
//
// These four run once per net load, but "once" is 84 MB — and the load is not
// the rare event it looks like. Every SPSA/screen game pair spawns two
// engines, so at fn20k the loader was measured (2026-07-27) at **12.2% of an
// SPSA unit's instructions**: 1.68 G to load against 24.1 G to play. Test
// throughput is our binding constraint, so that is worth more than its zero
// effect on NPS at depth suggests.
//
// The old shape was `for _ in 0..count { v.push(payload[*cur] ...) }`, which
// pays a bounds check AND a capacity check per ELEMENT — 84 million of each.
// The rewrite slices the whole run once (one bounds check) and hands `collect`
// a `chunks_exact` iterator. Two properties make that fast: `chunks_exact`
// yields only whole chunks, so the compiler knows each `c` is exactly N bytes
// and elides the inner bounds checks; and it is an `ExactSizeIterator`, so
// `collect` preallocates exactly once instead of growing. The loop then
// vectorises.
//
// Panic behaviour is deliberately unchanged: an over-short payload panicked on
// the element index before and panics on the slice range now. Section sizes are
// validated upstream; this is a speed patch, not an error-model change.

fn read_i16s(payload: &[u8], cur: &mut usize, count: usize) -> Aligned<i16> {
    let end = *cur + 2 * count;
    let out = payload[*cur..end]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    *cur = end;
    out
}

fn read_i8s(payload: &[u8], cur: &mut usize, count: usize) -> Box<[i8]> {
    let end = *cur + count;
    // `i8` and `u8` share a layout, so this is a pure reinterpret that LLVM
    // lowers to a memcpy — no per-byte work at all.
    let out = payload[*cur..end].iter().map(|&b| b as i8).collect();
    *cur = end;
    out
}

/// [`read_i8s`] into a cache-line-aligned buffer — for the FullThreats FT
/// table, whose 1280 B rows `i8_deltas_into` walks as whole `zmm` loads.
/// See `super::aligned` for why the base pointer decides every one of them.
fn read_i8s_aligned(payload: &[u8], cur: &mut usize, count: usize) -> Aligned<i8> {
    let end = *cur + count;
    let out = Aligned::from_iter_len(count, payload[*cur..end].iter().map(|&b| b as i8));
    *cur = end;
    out
}

fn read_i32s(payload: &[u8], cur: &mut usize, count: usize) -> Aligned<i32> {
    let end = *cur + 4 * count;
    let out = payload[*cur..end]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *cur = end;
    out
}

/// [`read_i8s_aligned`]'s block-shaped sibling: `B` blocks of `N` i8, in the
/// same block-major order the flat read produced.
fn read_i8_blocks<const N: usize, const B: usize>(
    payload: &[u8],
    cur: &mut usize,
) -> Blocks<i8, N, B> {
    let end = *cur + N * B;
    let out = Blocks::from_flat_iter(payload[*cur..end].iter().map(|&b| b as i8));
    *cur = end;
    out
}

/// The f32 form of [`read_i8_blocks`]; same block-major order.
fn read_f32_blocks<const N: usize, const B: usize>(
    payload: &[u8],
    cur: &mut usize,
) -> Blocks<f32, N, B> {
    let end = *cur + 4 * N * B;
    let out = Blocks::from_flat_iter(
        payload[*cur..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
    *cur = end;
    out
}

fn read_f32s(payload: &[u8], cur: &mut usize, count: usize) -> Box<[f32]> {
    let end = *cur + 4 * count;
    let out = payload[*cur..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *cur = end;
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::attacks::Prng;

    /// Build a deterministic synthetic SOTA net (with threats) for round-trip
    /// tests. NOT a trained net — weights are small random values.
    /// `pub(crate)` so sibling modules' tests can use it.
    pub(crate) fn test_net_with_threats_pub() -> SotaNetwork {
        test_net_with_threats()
    }

    fn test_net_with_threats() -> SotaNetwork {
        let mut prng = Prng::new(0xD40C_E420);

        let small_i16 = |prng: &mut Prng| -> i16 { (prng.rand64() % 201) as i16 - 100 };
        let small_i8 = |prng: &mut Prng| -> i8 { (prng.rand64() % 121) as i8 - 60 };
        let small_f32 = |prng: &mut Prng| -> f32 { ((prng.rand64() % 201) as f32 - 100.0) * 0.01 };

        // Small enough that every column's scale comes out 1, so the synthetic
        // net's reduced accumulator frame IS the FT_QUANT frame and the round
        // trip is exact (perf study 32).
        let halfka_weights: Aligned<i8> = (0..HALFKA_DIMS * ACC)
            .map(|_| small_i8(&mut prng))
            .collect();
        let ft_bias: Aligned<i16> = (0..ACC).map(|_| small_i16(&mut prng)).collect();
        let threat_weights: Aligned<i8> = (0..THREAT_DIMS * ACC)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_weights: L1Weights = (0..BUCKETS * L2 * L1)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_biases: L1Biases = (0..BUCKETS * L2).map(|_| small_f32(&mut prng)).collect();
        let l2_weights: L2Weights = (0..BUCKETS * L2 * L3)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l2_biases: L2Biases = (0..BUCKETS * L3).map(|_| small_f32(&mut prng)).collect();
        let l3_weights: L3Weights = (0..BUCKETS * L3).map(|_| small_f32(&mut prng)).collect();
        let l3_biases: Box<[f32]> = (0..BUCKETS).map(|_| small_f32(&mut prng)).collect();

        SotaNetwork {
            halfka_weights,
            ft_bias,
            threat_weights,
            threat_rows: THREAT_DIMS,
            threat_remap: None,
            psqt_weights: Vec::new().into(),
            head: SotaHead {
                l1_weights,
                l1_biases,
                l2_weights,
                l2_biases,
                l3_weights,
                l3_biases,
                network_scale: 380,
                threat_shift: 2,
                l3_var_weights: None,
                l3_var_biases: None,
                l3_wdl_weights: None,
                l3_wdl_biases: None,
                skip_weights: None,
                ft_scale: sota_head::FtScale::from_flat_iter(std::iter::repeat_n(1u8, ACC)),
            },
        }
    }

    /// Every f32 block the head streams per eval must START on a cache line.
    ///
    /// A GATE (perf study 21 arm B4). `l1_biases`/`l2_biases`/`l3_weights`
    /// were plain `Box<[f32]>` while their weight siblings were already
    /// `Aligned` -- perf study 20 arm 5 fixed the weights and missed the
    /// biases. Each is read per eval as a whole `[f32; L2]` (64 B) or
    /// `[f32; L3]` (128 B) block at a `bucket`-strided offset, and both
    /// strides are multiples of 64, so the base pointer alone decides whether
    /// every one of those `_mm512_loadu_ps` crosses a cache line.
    #[test]
    fn head_f32_blocks_start_on_a_cache_line() {
        let net = test_net_with_threats();
        for (name, base) in [
            ("l1_biases", net.head.l1_biases.as_ptr() as usize),
            ("l2_biases", net.head.l2_biases.as_ptr() as usize),
            ("l3_weights", net.head.l3_weights.as_ptr() as usize),
        ] {
            assert_eq!(base % 64, 0, "{name} base is {} mod 64", base % 64);
        }
    }

    /// A no-threats variant: tests the `flags=0` path.
    fn test_net_no_threats() -> SotaNetwork {
        let mut full = test_net_with_threats();
        full.threat_weights = Vec::new().into();
        full.threat_rows = 0;
        full
    }

    /// A deterministic T6c PSQT skip table covering the full feature space
    /// (`pub(crate)` so the context tests can arm the same fixture).
    pub(crate) fn test_psqt_table() -> Aligned<i32> {
        let mut prng = Prng::new(0x9541_7AB1_E000);
        ((0..(HALFKA_DIMS + THREAT_DIMS) * BUCKETS).map(|_| (prng.rand64() % 20001) as i32 - 10000))
            .collect()
    }

    /// The threats fixture with a PSQ1 skip table attached.
    pub(crate) fn test_net_with_psqt_pub() -> SotaNetwork {
        let mut net = test_net_with_threats();
        net.psqt_weights = test_psqt_table();
        net
    }

    /// The Recode POC fixture pair: a **compressed** net (`ROWS` threat rows +
    /// a TRM1 surjection) and its **scattered twin** (full 60,720-row tables
    /// with `W'[t] = Wsmall[remap[t]]`, carrying an identity-sized TRM1 so
    /// both nets exercise the remap path). The two must evaluate
    /// bit-identically on every position — that equality is the whole
    /// correctness argument for deploy mode (ii), and `recode_pair_is_eval
    /// _identical_over_a_game` in `sota_context` pins it end-to-end through
    /// the incremental accumulators. Both carry PSQ1 threat rows so the skip
    /// channel is exercised too.
    pub(crate) fn test_recode_pair_pub() -> (SotaNetwork, SotaNetwork) {
        const ROWS: usize = 4096;
        let mut prng = Prng::new(0x5EC0_DE01);

        // A deterministic surjection covering every target row (entry i % ROWS
        // guarantees surjectivity for i < ROWS; the rest scatter pseudo-randomly).
        let mut remap = Box::new([0u16; THREAT_DIMS]);
        for (i, r) in remap.iter_mut().enumerate() {
            *r = if i < ROWS {
                i as u16
            } else {
                (prng.rand64() % ROWS as u64) as u16
            };
        }

        // Small tables: ROWS weight rows + ROWS psqt rows.
        let small_i8 = |prng: &mut Prng| -> i8 { (prng.rand64() % 121) as i8 - 60 };
        let w_small: Aligned<i8> = (0..ROWS * ACC).map(|_| small_i8(&mut prng)).collect();
        let psqt_small: Vec<i32> = (0..ROWS * BUCKETS)
            .map(|_| (prng.rand64() % 20001) as i32 - 10000)
            .collect();

        // The compressed net.
        let mut compressed = test_net_with_threats();
        compressed.threat_weights = w_small.iter().copied().collect();
        compressed.threat_rows = ROWS;
        compressed.threat_remap = Some(remap.clone());
        let halfka_psqt: Vec<i32> = test_psqt_table()[..HALFKA_DIMS * BUCKETS].to_vec();
        compressed.psqt_weights = halfka_psqt
            .iter()
            .copied()
            .chain(psqt_small.iter().copied())
            .collect();

        // The scattered twin: full-size tables that read the SAME values
        // through the identity code.
        let mut scattered = test_net_with_threats();
        scattered.threat_weights = remap
            .iter()
            .flat_map(|&r| {
                let r = usize::from(r);
                w_small[r * ACC..(r + 1) * ACC].iter().copied()
            })
            .collect();
        scattered.threat_rows = THREAT_DIMS;
        scattered.threat_remap = Some(Box::new(std::array::from_fn(|i| i as u16)));
        scattered.psqt_weights = halfka_psqt
            .iter()
            .copied()
            .chain(remap.iter().flat_map(|&r| {
                let r = usize::from(r);
                psqt_small[r * BUCKETS..(r + 1) * BUCKETS].iter().copied()
            }))
            .collect();

        (compressed, scattered)
    }

    /// A compressed TRM1 net round-trips through serialize/load bit-exactly,
    /// and the header's threat_dims field carries the compressed row count.
    #[test]
    fn round_trip_with_remap() {
        let (compressed, scattered) = test_recode_pair_pub();
        for net in [&compressed, &scattered] {
            let bytes = serialize(net);
            assert_eq!(
                read_u32(&bytes, 12) as usize,
                net.threat_rows,
                "header threat_dims must carry the row count"
            );
            let back = load_v2_bytes(&bytes).expect("remap net loads");
            assert_eq!(*net, back);
        }
    }

    /// A compressed threat table without its TRM1 section must be rejected,
    /// and so must a remap entry pointing past the row count.
    #[test]
    fn rejects_compressed_without_trm1_and_bad_entries() {
        let (compressed, _) = test_recode_pair_pub();
        let good = serialize(&compressed);

        // Truncate the TRM1 section (it is the final trailer).
        let trm1_len = 12 + THREAT_DIMS * 2;
        let bytes = &good[..good.len() - trm1_len];
        assert!(matches!(
            load_v2_bytes(bytes),
            Err(SotaNnueError::BadRemap { .. })
        ));

        // Corrupt one remap entry to 0xFFFF (>= ROWS).
        let mut bytes = good.clone();
        let n = bytes.len();
        bytes[n - 2] = 0xFF;
        bytes[n - 1] = 0xFF;
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadRemap { .. })
        ));
    }

    /// The scattered twin and the compressed net produce identical
    /// from-scratch accumulators (the projection identity, one position).
    /// The end-to-end game-walk twin lives in `sota_context`.
    #[test]
    fn recode_pair_matches_on_refresh() {
        use super::super::sota_features::refresh_threat_into;
        let (compressed, scattered) = test_recode_pair_pub();
        let board = crate::board::Board::startpos();
        for persp in crate::types::Color::ALL {
            let mut a = [0i16; ACC];
            let mut b = [0i16; ACC];
            refresh_threat_into(
                &mut a,
                &compressed.threat_weights,
                &board,
                persp,
                compressed.threat_remap.as_deref(),
            );
            refresh_threat_into(
                &mut b,
                &scattered.threat_weights,
                &board,
                persp,
                scattered.threat_remap.as_deref(),
            );
            assert_eq!(a, b, "compressed and scattered accumulators must agree");
        }
    }

    #[test]
    fn round_trip_with_threats() {
        let net = test_net_with_threats();
        let bytes = serialize(&net);
        assert_eq!(&bytes[0..4], b"CEN2");
        assert_eq!(read_u16(&bytes, 4), CENN_V2_VERSION);
        assert_eq!(read_u16(&bytes, 6) & FLAG_HAS_THREATS, FLAG_HAS_THREATS);

        let back = load_v2_bytes(&bytes).expect("valid net loads");
        assert_eq!(net, back);
    }

    /// **The perf-study-30 safety property, in-tree.** A `CEN2` container has
    /// no `threat_quant` field, so it must keep meaning 64 ⇒ shift 2 — the
    /// literal every backend hardcoded before the version bump. If this moves,
    /// every shipped net's evaluation moves with it.
    #[test]
    fn cen2_implies_the_legacy_threat_frame() {
        let bytes = serialize(&test_net_with_threats());
        assert_eq!(&bytes[0..4], b"CEN2");
        assert_eq!(
            bytes.len(),
            HEADER_V2_LEN + payload_len_with_threats(THREAT_DIMS)
        );
        let back = load_v2_bytes(&bytes).expect("v2 loads");
        assert_eq!(back.head.threat_shift, 2);
        assert_eq!(
            sota_head::threat_quant_for(back.head.threat_shift),
            sota_head::THREAT_QUANT_CEN2
        );
    }

    /// A QAT net (threat_quant 256 ⇒ shift 0) serializes as `CEN5` and reloads
    /// with its own frame — and the PAYLOAD is byte-identical to the v2 form,
    /// because v5 only APPENDS to the header.
    #[test]
    fn cen5_carries_threat_quant_and_only_appends() {
        let v2_net = test_net_with_threats();
        let v2_bytes = serialize(&v2_net);

        let mut qat = test_net_with_threats();
        qat.head.threat_shift = 0;
        let v5_bytes = serialize(&qat);

        assert_eq!(&v5_bytes[0..4], b"CEN5");
        assert_eq!(read_u16(&v5_bytes, 4), CENN_V5_VERSION);
        assert_eq!(read_i32(&v5_bytes, 64), 256);
        assert_eq!(read_u32(&v5_bytes, 68), 0, "reserved must be written zero");
        assert_eq!(v5_bytes.len(), v2_bytes.len() + 8);
        // Bytes 4..64 (version aside) and the whole payload are identical.
        assert_eq!(v5_bytes[6..64], v2_bytes[6..64]);
        assert_eq!(v5_bytes[HEADER_V5_LEN..], v2_bytes[HEADER_V2_LEN..]);

        let back = load_v2_bytes(&v5_bytes).expect("v5 loads");
        assert_eq!(back.head.threat_shift, 0);
        assert_eq!(qat, back);
        // …and it is NOT equal to the v2 net, i.e. the frame really is carried.
        assert_ne!(v2_net, back);
    }

    #[test]
    fn rejects_bad_threat_quant_and_dirty_reserved() {
        let mut qat = test_net_with_threats();
        qat.head.threat_shift = 0;
        let good = serialize(&qat);

        // 255 is not a power of two, so the frame ratio is not a shift.
        let mut bytes = good.clone();
        bytes[64..68].copy_from_slice(&255i32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadThreatQuant(255))
        ));

        // 512 would need a NEGATIVE shift.
        let mut bytes = good.clone();
        bytes[64..68].copy_from_slice(&512i32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadThreatQuant(512))
        ));

        // The reserved word is a future field: a non-zero value means the file
        // was written by something newer, so refuse rather than misread it.
        let mut bytes = good.clone();
        bytes[68..72].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "reserved",
                ..
            })
        ));

        // A v5 magic with a v2-length body is short, not silently truncated.
        let mut bytes = good.clone();
        bytes.truncate(HEADER_V2_LEN + 4);
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::TooShort { need: 72, .. })
        ));

        // A v5 magic with the v2 version number is a mismatch, not a v2 file.
        let mut bytes = good;
        bytes[4..6].copy_from_slice(&CENN_V2_VERSION.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn round_trip_no_threats() {
        let net = test_net_no_threats();
        let bytes = serialize(&net);
        assert_eq!(read_u16(&bytes, 6) & FLAG_HAS_THREATS, 0);
        assert_eq!(read_u32(&bytes, 12), 0); // threat_dims = 0

        let back = load_v2_bytes(&bytes).expect("valid no-threats net loads");
        assert_eq!(net, back);
    }

    #[test]
    fn round_trip_with_psqt() {
        let net = test_net_with_psqt_pub();
        let bytes = serialize(&net);
        // The PSQ1 trailer sits after the checksummed payload.
        let base_len = serialize(&test_net_with_threats()).len();
        assert_eq!(&bytes[base_len..base_len + 4], b"PSQ1");
        assert_eq!(
            bytes.len(),
            base_len + 12 + (HALFKA_DIMS + THREAT_DIMS) * BUCKETS * 4
        );

        let back = load_v2_bytes(&bytes).expect("psqt net loads");
        assert_eq!(net, back);
        assert_eq!(back.halfka_psqt().len(), HALFKA_DIMS * BUCKETS);
        assert_eq!(back.threat_psqt().len(), THREAT_DIMS * BUCKETS);
        // Row split: HalfKA rows first, threat rows after.
        assert_eq!(back.halfka_psqt()[5], net.psqt_weights[5]);
        assert_eq!(
            back.threat_psqt()[3],
            net.psqt_weights[HALFKA_DIMS * BUCKETS + 3]
        );
    }

    #[test]
    fn round_trip_psqt_no_threats() {
        let mut net = test_net_no_threats();
        net.psqt_weights = test_psqt_table()[..HALFKA_DIMS * BUCKETS].to_vec().into();
        let bytes = serialize(&net);
        let back = load_v2_bytes(&bytes).expect("halfka-only psqt net loads");
        assert_eq!(net, back);
        assert_eq!(back.halfka_psqt().len(), HALFKA_DIMS * BUCKETS);
        assert!(back.threat_psqt().is_empty());
    }

    #[test]
    fn rejects_bad_psqt_section() {
        let net = test_net_with_psqt_pub();
        let base_len = serialize(&test_net_with_threats()).len();

        // Wrong bucket count.
        let mut bytes = serialize(&net);
        bytes[base_len + 4..base_len + 8].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "psqt_buckets",
                ..
            })
        ));

        // Wrong row count.
        let mut bytes = serialize(&net);
        bytes[base_len + 8..base_len + 12].copy_from_slice(&(HALFKA_DIMS as u32).to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "psqt_rows",
                ..
            })
        ));

        // Truncated table.
        let bytes = serialize(&net);
        assert!(matches!(
            load_v2_bytes(&bytes[..bytes.len() - 4]),
            Err(SotaNnueError::TooShort { .. })
        ));
    }

    /// A legacy net (no PSQ1 trailer) parses with an empty skip table — the
    /// "nets without it behave exactly as today" half of the contract.
    #[test]
    fn legacy_net_has_no_psqt() {
        let bytes = serialize(&test_net_with_threats());
        let back = load_v2_bytes(&bytes).expect("legacy net loads");
        assert!(back.psqt_weights.is_empty());
        assert!(back.halfka_psqt().is_empty());
        assert!(back.threat_psqt().is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize(&test_net_with_threats());
        bytes[0] = b'X';
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::BadMagic(_))
        ));
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = serialize(&test_net_with_threats());
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn rejects_wrong_dims() {
        let mut bytes = serialize(&test_net_with_threats());
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
        let mut bytes = serialize(&test_net_with_threats());
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
        let mut bytes = serialize(&test_net_with_threats());
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
        let bytes = serialize(&test_net_with_threats());
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
        let mut bytes = serialize(&test_net_with_threats());
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
        let expected = HALFKA_DIMS * ACC * 2  // halfka i16 ON DISK (i8 in memory)
            + ACC * 2                         // bias i16
            + THREAT_DIMS * ACC               // threats i8
            + BUCKETS * L2 * L1               // l1w i8
            + BUCKETS * L2 * 4                // l1b f32
            + BUCKETS * L2 * L3 * 4           // l2w f32
            + BUCKETS * L3 * 4                // l2b f32
            + BUCKETS * L3 * 4                // l3w f32
            + BUCKETS * 4; // l3b f32
        assert_eq!(payload_len_with_threats(THREAT_DIMS), expected);

        // Verify the serialized net has the right total size.
        let net = test_net_with_threats();
        let bytes = serialize(&net);
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
