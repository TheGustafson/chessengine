//! NNUE inference (design/11 §3.4) — the Phase C evaluation path.
//!
//! **Scope (P13 step 1).** This is the *inference skeleton*: the quantized
//! network container (CENN, [`loader`]), the perspective feature transformer
//! and its incremental accumulator ([`accumulator`]), and a SCALAR quantized
//! SCReLU/CReLU forward pass ([`simd`]). It is purely additive — the active
//! evaluation (`eval::evaluate`) still returns the PSQT value; the NNUE
//! forward pass is exercised only by tests and the `nnue-eval` dev
//! subcommand. Wiring NNUE in as the eval, the AVX2 backend, and a trained
//! net all land in later steps (design/11 §11).
//!
//! **Architecture** (`768 → 2×256 → 1`, single perspective pair, SCReLU):
//! 768 sparse binary inputs (`2 colors × 6 piece types × 64 squares`) feed a
//! shared feature transformer producing two [`L1`]-wide accumulators (one per
//! perspective). The output layer concatenates the two activated
//! accumulators — **side-to-move half first** — and dots with the output
//! weights (design/11 §3.4.1).
//!
//! **Quantization** (bullet-standard, design/11 §3.4.3): feature-transformer
//! weights and bias scaled by [`QA`], output weights by [`QB`], centipawn
//! output by [`SCALE`]. The scalar forward accumulates in `i64` and is the
//! bit-exact reference the future AVX2 path (step 2) must match.

pub mod accumulator;
pub mod loader;
pub mod network;
/// Frozen classic evaluator plus an exact HalfKAv2_hm residual (CENN-v4).
#[cfg(feature = "sota")]
pub mod protected_halfka;
pub mod simd;
/// Exact HalfKAv2_hm plus the protected classic SCReLU head (CENN-v3).
#[cfg(feature = "sota")]
pub mod simple_halfka;
/// Incremental HalfKA stack and Finny cache for the SOTA net.
#[cfg(feature = "sota")]
mod sota_accumulator;
/// Per-thread SOTA eval context: accumulators + head forward (design/11 §12) —
/// behind `sota`.
#[cfg(feature = "sota")]
pub mod sota_context;
/// The SOTA net's SF feature front-end (HalfKAv2_hm + FullThreats index math +
/// from-scratch accumulator refresh, D40, design/11 §12.1-12.2) — behind `sota`.
#[cfg(feature = "sota")]
pub mod sota_features;
/// The SOTA net's head forward (D40, design/11 §12) — behind the `sota` feature.
#[cfg(feature = "sota")]
pub mod sota_head;
/// The CENN-v2 container loader for D40 SOTA nets (design/11 §12.6) — behind
/// `sota`.
#[cfg(feature = "sota")]
pub mod sota_loader;
/// Runtime-dispatched fused accumulator kernels for the SOTA transformers.
#[cfg(feature = "sota")]
mod sota_simd;
/// The SOTA net's incremental FullThreats accumulator (the ≤80/move dirty-diff
/// observer, D40, design/11 §12.9.1) — behind `sota`.
#[cfg(feature = "sota")]
pub mod sota_threats;

// --- Architecture dimensions (design/11 §3.4.1) ----------------------------

/// Number of sparse input features: `2 colors × 6 piece types × 64 squares`.
pub const INPUT: usize = 768;

/// Feature-transformer / accumulator width (the "L1" hidden size). 256 by
/// default (design/11 §11 locked choice); the `nnue-l1-512` feature widens it to
/// 512 for the arch-scale experiment (task #19). A build loads only nets of its
/// own width — the loader checks the CENN header `l1` (`WrongL1`). Everything
/// downstream (accumulator arrays, AVX2 kernel strides, `PAYLOAD_LEN`) derives
/// from this const, so 512 is a one-line widening (512 % 16 == 0, so the AVX2
/// lane assert still holds).
#[cfg(not(any(
    feature = "nnue-l1-512",
    feature = "nnue-l1-1024",
    feature = "nnue-l1-2048"
)))]
pub const L1: usize = 256;
#[cfg(all(
    feature = "nnue-l1-512",
    not(any(feature = "nnue-l1-1024", feature = "nnue-l1-2048"))
))]
pub const L1: usize = 512;
#[cfg(all(
    feature = "nnue-l1-1024",
    not(any(feature = "nnue-l1-512", feature = "nnue-l1-2048"))
))]
pub const L1: usize = 1024;
#[cfg(all(
    feature = "nnue-l1-2048",
    not(any(feature = "nnue-l1-512", feature = "nnue-l1-1024"))
))]
pub const L1: usize = 2048;
#[cfg(any(
    all(feature = "nnue-l1-512", feature = "nnue-l1-1024"),
    all(feature = "nnue-l1-512", feature = "nnue-l1-2048"),
    all(feature = "nnue-l1-1024", feature = "nnue-l1-2048")
))]
compile_error!("nnue-l1-512 / nnue-l1-1024 / nnue-l1-2048 are mutually exclusive");

// --- Quantization constants (design/11 §3.4.3) -----------------------------

/// Feature-transformer scale: `w0_q = round(w0 · QA)`, stored `i16`. Also the
/// SCReLU clamp ceiling (`clamp(acc, 0, QA)`).
pub const QA: i32 = 255;

/// Output-layer weight scale: `w1_q = round(w1 · QB)`, stored `i16`.
pub const QB: i32 = 64;

/// Centipawn output scale.
pub const SCALE: i32 = 400;

// --- Re-exported public surface --------------------------------------------

pub use accumulator::{AccumulatorEntry, NnueContext, evaluate_forward};
pub use loader::{NnueError, load, load_bytes, serialize};
pub use network::{Activation, Network, feature_index};
pub use simd::{Aligned64, KERNELS, Kernels};

// `DirtyPiece` is defined in `board` (its data is a board-mutation fact and
// its constructor reads `Board`'s private castling geometry — design/11
// §3.4.4 layering note). Re-exported here so the design-doc-facing path
// `eval::nnue::DirtyPiece` still resolves.
pub use crate::board::DirtyPiece;

// --- Active-net registry (design/11 §3.4.5: EvalFile, opt-in) ----------------

use std::sync::RwLock;

/// The process-wide active NNUE net (design/11 §3.4.5). `None` ⇒ NNUE is OFF
/// and evaluation is PSQT — the default, so the depth-6 bench fingerprint stays
/// bit-identical. Set via the UCI `EvalFile` option or the `bench` net argument
/// ([`set_eval_file`]); read once per search at the eval boundary
/// (`Worker::refresh_eval_net`), never on the hot per-node path. `RwLock` (not
/// an `AtomicPtr`) keeps the crate `unsafe`-free (D14): a `&'static Network` is
/// `Send + Sync`, so concurrent worker threads read the same net lock-free
/// enough (uncontended reader lock, once per `go`).
static ACTIVE_NET: RwLock<Option<&'static Network>> = RwLock::new(None);

/// Process-wide active SOTA (D40) net. Mirrors [`ACTIVE_NET`] for CENN-v2 files.
/// When a CENN-v2 net is installed the v1 ACTIVE_NET is cleared, and vice versa.
#[cfg(feature = "sota")]
static ACTIVE_SOTA_NET: RwLock<Option<&'static sota_loader::SotaNetwork>> = RwLock::new(None);
#[cfg(feature = "sota")]
static ACTIVE_SIMPLE_HALFKA_NET: RwLock<Option<&'static simple_halfka::SimpleHalfkaNetwork>> =
    RwLock::new(None);
#[cfg(feature = "sota")]
static ACTIVE_PROTECTED_HALFKA_NET: RwLock<
    Option<&'static protected_halfka::ProtectedHalfkaNetwork>,
> = RwLock::new(None);

/// Install (or clear) the `EvalFile` net. An empty/blank path turns NNUE OFF
/// (back to PSQT). Otherwise the file is loaded and fully validated.
///
/// With the `sota` feature: the first 4 bytes are peeked — if `b"CEN2"` the
/// file is loaded as a D40 SOTA net (clearing any v1 net), otherwise as a v1
/// CENN net (clearing any SOTA net). Without the `sota` feature: only v1 CENN
/// is accepted. Loader errors are returned for a clean UCI `info string`; this
/// never panics on a user-supplied file (CLAUDE.md).
pub fn set_eval_file(path: &str) -> Result<(), NnueError> {
    if path.trim().is_empty() {
        *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
        #[cfg(feature = "sota")]
        {
            *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_SIMPLE_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_PROTECTED_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
        }
        return Ok(());
    }

    // Peek magic to route v1 vs v2.
    #[cfg(feature = "sota")]
    {
        let bytes = std::fs::read(path).map_err(NnueError::Io)?;
        if bytes.len() >= 4 && bytes[0..4] == *b"CEN2" {
            let net = sota_loader::load_v2_bytes(&bytes).map_err(|e| {
                NnueError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            let leaked: &'static sota_loader::SotaNetwork = Box::leak(Box::new(net));
            *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = Some(leaked);
            *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_SIMPLE_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_PROTECTED_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            return Ok(());
        }
        if bytes.len() >= 4 && bytes[0..4] == simple_halfka::MAGIC {
            let net = simple_halfka::load_bytes(&bytes).map_err(|e| {
                NnueError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            *ACTIVE_SIMPLE_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = Some(Box::leak(Box::new(net)));
            *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_PROTECTED_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            return Ok(());
        }
        if bytes.len() >= 4 && bytes[0..4] == protected_halfka::MAGIC {
            let net = protected_halfka::load_bytes_v4(&bytes).map_err(|e| {
                NnueError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            let leaked: &'static protected_halfka::ProtectedHalfkaNetwork =
                Box::leak(Box::new(net));
            *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = Some(&leaked.base);
            *ACTIVE_PROTECTED_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = Some(leaked);
            *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
            *ACTIVE_SIMPLE_HALFKA_NET
                .write()
                .unwrap_or_else(|e| e.into_inner()) = None;
            return Ok(());
        }
        // Not CEN2 → fall through to v1 loader. Clear SOTA.
        *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
        *ACTIVE_SIMPLE_HALFKA_NET
            .write()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *ACTIVE_PROTECTED_HALFKA_NET
            .write()
            .unwrap_or_else(|e| e.into_inner()) = None;
        let net: &'static Network = Box::leak(Box::new(load_bytes(&bytes)?));
        *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = Some(net);
        Ok(())
    }

    // Non-sota build: v1 only.
    #[cfg(not(feature = "sota"))]
    {
        let leaked: &'static Network = Box::leak(Box::new(load(path)?));
        *ACTIVE_NET.write().unwrap_or_else(|e| e.into_inner()) = Some(leaked);
        Ok(())
    }
}

/// The active v1 net, or `None` when NNUE is off (⇒ PSQT). Read at each search's
/// eval boundary and handed to the per-thread [`NnueContext`].
///
/// With the `embed-net` feature (distribution builds), an explicit `EvalFile`
/// still wins, but when none is set the compile-time-embedded default net is
/// used instead of falling back to PSQT — so the engine plays NNUE out of the
/// box for GUIs that can't set UCI string options. The default build (no
/// `embed-net`) ignores this entirely and stays PSQT/bench-neutral (548978).
pub fn active_net() -> Option<&'static Network> {
    let from_eval_file = *ACTIVE_NET.read().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "embed-net")]
    {
        return from_eval_file.or_else(|| Some(embedded_net()));
    }
    #[cfg(not(feature = "embed-net"))]
    from_eval_file
}

/// The active D40 SOTA net, or `None`. Read at each search's eval boundary.
#[cfg(feature = "sota")]
pub fn active_sota_net() -> Option<&'static sota_loader::SotaNetwork> {
    *ACTIVE_SOTA_NET.read().unwrap_or_else(|e| e.into_inner())
}

#[cfg(feature = "sota")]
pub fn active_simple_halfka_net() -> Option<&'static simple_halfka::SimpleHalfkaNetwork> {
    *ACTIVE_SIMPLE_HALFKA_NET
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

#[cfg(feature = "sota")]
pub fn active_protected_halfka_net() -> Option<&'static protected_halfka::ProtectedHalfkaNetwork> {
    *ACTIVE_PROTECTED_HALFKA_NET
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

/// The compile-time-embedded default net (distribution builds only, `embed-net`
/// feature). Loaded and leaked once. The bytes are a build asset validated at
/// build time, so a load failure is a broken build — not user input — hence the
/// `expect`. Read once per search (never on the hot path), same as [`active_net`].
#[cfg(feature = "embed-net")]
fn embedded_net() -> &'static Network {
    use std::sync::OnceLock;
    static EMBEDDED: OnceLock<&'static Network> = OnceLock::new();
    *EMBEDDED.get_or_init(|| {
        const BYTES: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/nnue/nets/net-0.cenn"
        ));
        Box::leak(Box::new(
            load_bytes(BYTES).expect("embedded default net is a valid CENN file"),
        )) as &'static Network
    })
}
