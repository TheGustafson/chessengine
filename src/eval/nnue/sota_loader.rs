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

use super::aligned::Aligned;
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
/// Magic bytes of the optional trailing PSQT-skip section (T6c).
pub const PSQT_MAGIC: [u8; 4] = *b"PSQ1";
/// SKP1 magic (D47V output-skip lane; contract: jax checkpoint.py
/// skip_section_bytes — b"SKP1" | u32 num_buckets | u32 hl | f32[hl·buckets],
/// row-major, raw f32; rides AFTER the PSQ1 section).
pub const SKIP_MAGIC: [u8; 4] = *b"SKP1";
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
    /// HalfKAv2_hm FT weights, feature-major: `[HALFKA_DIMS · ACC]` i16.
    pub halfka_weights: Aligned<i16>,
    /// Shared FT bias: `[ACC]` i16 (seeds the HalfKA accumulator; threats seed 0).
    pub ft_bias: Aligned<i16>,
    /// FullThreats FT weights, feature-major: `[THREAT_DIMS · ACC]` i8.
    /// Empty if the net has no threats channel.
    pub threat_weights: Aligned<i8>,
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
            && self.ft_bias == other.ft_bias
            && self.threat_weights == other.threat_weights
            && self.psqt_weights == other.psqt_weights
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
            } => write!(f, "CENN-v2 {field} {found} != {expected} (quant mismatch)"),
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
    if !(1..=10_000).contains(&network_scale) {
        return Err(SotaNnueError::WrongDim {
            field: "network_scale",
            expected: 380,
            found: network_scale as u32,
        });
    }

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
        read_i8s_aligned(payload, &mut cur, THREAT_DIMS * ACC)
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };

    // L1 weights: i8[BUCKETS · L2 · L1], bucket-major, group-interleaved.
    let l1_weights = read_i8s_aligned(payload, &mut cur, BUCKETS * L2 * L1);

    // L1 biases: f32[BUCKETS · L2]. Cache-line aligned like the weights:
    // the head reads a whole 64-byte `[f32; L2]` block per eval and the
    // bucket stride is a multiple of 64, so the base decides them all.
    let l1_biases = read_f32s_aligned(payload, &mut cur, BUCKETS * L2);

    // L2 weights: f32[BUCKETS · L2 · L3].
    let l2_weights = read_f32s_aligned(payload, &mut cur, BUCKETS * L2 * L3);

    // L2 biases: f32[BUCKETS · L3]. Aligned; see `l1_biases`.
    let l2_biases = read_f32s_aligned(payload, &mut cur, BUCKETS * L3);

    // L3 weights: f32[BUCKETS · L3]. Aligned; see `l1_biases`.
    let l3_weights = read_f32s_aligned(payload, &mut cur, BUCKETS * L3);

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
    let psqt_weights = if rest.len() >= 4 && rest[0..4] == PSQT_MAGIC {
        parse_psqt_section(rest, has_threats)?
    } else {
        Aligned::from_iter_len(0, std::iter::empty())
    };
    // D47V: optional SKP1 trailer after PSQ1 (or directly, on a psqt-less
    // net). Empty slice = no skip lane = the pre-D47V eval, bit-identical.
    let skip_weights = {
        let psqt_len = if psqt_weights.is_empty() {
            0
        } else {
            PSQT_HEADER_LEN + psqt_weights.len() * 4
        };
        let tail = &rest[psqt_len.min(rest.len())..];
        if tail.len() >= 4 && tail[0..4] == SKIP_MAGIC {
            parse_skip_section(tail)?
        } else {
            Vec::new().into_boxed_slice()
        }
    };

    let head = SotaHead {
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        l3_weights,
        l3_biases,
        network_scale,
        l3_var_weights: None,
        l3_var_biases: None,
        l3_wdl_weights: None,
        l3_wdl_biases: None,
        skip_weights: (!skip_weights.is_empty()).then(|| quantise_skip_lane(&skip_weights)),
    };

    Ok(SotaNetwork {
        halfka_weights,
        ft_bias,
        threat_weights,
        psqt_weights,
        head,
    })
}

/// Parse an SKP1 section (starting at its magic). Strict on this build's
/// [`BUCKETS`] and accumulator width [`crate::eval::nnue::sota_head::L1`];
/// tolerant of further trailing bytes (future sections).
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
fn parse_psqt_section(sec: &[u8], has_threats: bool) -> Result<Aligned<i32>, SotaNnueError> {
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
    let expected_rows = HALFKA_DIMS + if has_threats { THREAT_DIMS } else { 0 };
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

/// [`read_f32s`] into a cache-line-aligned buffer — for the L2 weight block,
/// which `l2_tail` reads as whole `zmm` operands (perf study 20 arm 5).
fn read_f32s_aligned(payload: &[u8], cur: &mut usize, count: usize) -> Aligned<f32> {
    let end = *cur + 4 * count;
    let out = Aligned::from_iter_len(
        count,
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

        let halfka_weights: Aligned<i16> = (0..HALFKA_DIMS * ACC)
            .map(|_| small_i16(&mut prng))
            .collect();
        let ft_bias: Aligned<i16> = (0..ACC).map(|_| small_i16(&mut prng)).collect();
        let threat_weights: Aligned<i8> = (0..THREAT_DIMS * ACC)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_weights: Aligned<i8> = (0..BUCKETS * L2 * L1)
            .map(|_| small_i8(&mut prng))
            .collect();
        let l1_biases: Aligned<f32> = (0..BUCKETS * L2).map(|_| small_f32(&mut prng)).collect();
        let l2_weights: Aligned<f32> = (0..BUCKETS * L2 * L3)
            .map(|_| small_f32(&mut prng))
            .collect();
        let l2_biases: Aligned<f32> = (0..BUCKETS * L3).map(|_| small_f32(&mut prng)).collect();
        let l3_weights: Aligned<f32> = (0..BUCKETS * L3).map(|_| small_f32(&mut prng)).collect();
        let l3_biases: Box<[f32]> = (0..BUCKETS).map(|_| small_f32(&mut prng)).collect();

        SotaNetwork {
            halfka_weights,
            ft_bias,
            threat_weights,
            psqt_weights: Vec::new().into(),
            head: SotaHead {
                l1_weights,
                l1_biases,
                l2_weights,
                l2_biases,
                l3_weights,
                l3_biases,
                network_scale: 380,
                l3_var_weights: None,
                l3_var_biases: None,
                l3_wdl_weights: None,
                l3_wdl_biases: None,
                skip_weights: None,
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
    fn round_trip_with_psqt() {
        let net = test_net_with_psqt_pub();
        let bytes = serialize_v2(&net);
        // The PSQ1 trailer sits after the checksummed payload.
        let base_len = serialize_v2(&test_net_with_threats()).len();
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
        let bytes = serialize_v2(&net);
        let back = load_v2_bytes(&bytes).expect("halfka-only psqt net loads");
        assert_eq!(net, back);
        assert_eq!(back.halfka_psqt().len(), HALFKA_DIMS * BUCKETS);
        assert!(back.threat_psqt().is_empty());
    }

    #[test]
    fn rejects_bad_psqt_section() {
        let net = test_net_with_psqt_pub();
        let base_len = serialize_v2(&test_net_with_threats()).len();

        // Wrong bucket count.
        let mut bytes = serialize_v2(&net);
        bytes[base_len + 4..base_len + 8].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "psqt_buckets",
                ..
            })
        ));

        // Wrong row count.
        let mut bytes = serialize_v2(&net);
        bytes[base_len + 8..base_len + 12].copy_from_slice(&(HALFKA_DIMS as u32).to_le_bytes());
        assert!(matches!(
            load_v2_bytes(&bytes),
            Err(SotaNnueError::WrongDim {
                field: "psqt_rows",
                ..
            })
        ));

        // Truncated table.
        let bytes = serialize_v2(&net);
        assert!(matches!(
            load_v2_bytes(&bytes[..bytes.len() - 4]),
            Err(SotaNnueError::TooShort { .. })
        ));
    }

    /// A legacy net (no PSQ1 trailer) parses with an empty skip table — the
    /// "nets without it behave exactly as today" half of the contract.
    #[test]
    fn legacy_net_has_no_psqt() {
        let bytes = serialize_v2(&test_net_with_threats());
        let back = load_v2_bytes(&bytes).expect("legacy net loads");
        assert!(back.psqt_weights.is_empty());
        assert!(back.halfka_psqt().is_empty());
        assert!(back.threat_psqt().is_empty());
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
            + BUCKETS * 4; // l3b f32
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
