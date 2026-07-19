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
    feature = "nnue-l1-2048",
    feature = "nnue-l1-4096"
)))]
pub const L1: usize = 256;
#[cfg(all(
    feature = "nnue-l1-512",
    not(any(
        feature = "nnue-l1-1024",
        feature = "nnue-l1-2048",
        feature = "nnue-l1-4096"
    ))
))]
pub const L1: usize = 512;
#[cfg(all(
    feature = "nnue-l1-1024",
    not(any(
        feature = "nnue-l1-512",
        feature = "nnue-l1-2048",
        feature = "nnue-l1-4096"
    ))
))]
pub const L1: usize = 1024;
#[cfg(all(
    feature = "nnue-l1-2048",
    not(any(
        feature = "nnue-l1-512",
        feature = "nnue-l1-1024",
        feature = "nnue-l1-4096"
    ))
))]
pub const L1: usize = 2048;
#[cfg(all(
    feature = "nnue-l1-4096",
    not(any(
        feature = "nnue-l1-512",
        feature = "nnue-l1-1024",
        feature = "nnue-l1-2048"
    ))
))]
pub const L1: usize = 4096;
#[cfg(any(
    all(feature = "nnue-l1-512", feature = "nnue-l1-1024"),
    all(feature = "nnue-l1-512", feature = "nnue-l1-2048"),
    all(feature = "nnue-l1-512", feature = "nnue-l1-4096"),
    all(feature = "nnue-l1-1024", feature = "nnue-l1-2048"),
    all(feature = "nnue-l1-1024", feature = "nnue-l1-4096"),
    all(feature = "nnue-l1-2048", feature = "nnue-l1-4096")
))]
compile_error!("the nnue-l1-* width features are mutually exclusive");

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

use std::path::PathBuf;
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;

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

/// Identity of the file behind the currently-installed net: canonical path,
/// size, and mtime. Match runners (fastchess) re-send every configured option
/// — `EvalFile` included — before EVERY game of a match, and each install
/// leaks its parsed net to `'static`; without this guard a 300-game match
/// leaked one full net per game (~78MB for D40) until the OOM killer fired.
/// A same-file re-send is a no-op; a changed file at the same path (new
/// size/mtime, e.g. netpack overwrote it) still reloads. `None` mtime means
/// the filesystem can't report one — such files are never treated as
/// already-loaded, trading redundant loads for correctness.
static LOADED_FILE: Mutex<Option<(PathBuf, u64, Option<SystemTime>)>> = Mutex::new(None);

/// The (canonical path, size, mtime) identity of `path` right now.
fn file_identity(path: &str) -> Result<(PathBuf, u64, Option<SystemTime>), NnueError> {
    let canonical = std::fs::canonicalize(path).map_err(NnueError::Io)?;
    let meta = std::fs::metadata(&canonical).map_err(NnueError::Io)?;
    Ok((canonical, meta.len(), meta.modified().ok()))
}

/// Install (or clear) the `EvalFile` net. An empty/blank path turns NNUE OFF
/// (back to PSQT). Otherwise the file is loaded and fully validated —
/// unless the exact same file is already the active net, in which case this
/// is a no-op (see [`LOADED_FILE`]).
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
        *LOADED_FILE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        return Ok(());
    }

    let identity = file_identity(path)?;
    if identity.2.is_some()
        && LOADED_FILE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            == Some(&identity)
    {
        return Ok(());
    }

    let result = install_eval_file(path);
    // A failed install may leave the slots partially cleared (pre-existing
    // contract: the error is reported and play continues) — drop the cached
    // identity so the next call always does a real load.
    *LOADED_FILE.lock().unwrap_or_else(|e| e.into_inner()) = match &result {
        Ok(()) if identity.2.is_some() => Some(identity),
        _ => None,
    };
    result
}

/// The unconditional load path behind [`set_eval_file`]: read, parse,
/// validate, leak, and swap the parsed net into its active slot.
fn install_eval_file(path: &str) -> Result<(), NnueError> {
    // Peek magic to route v1 vs v2.
    #[cfg(feature = "sota")]
    {
        let bytes = std::fs::read(path).map_err(NnueError::Io)?;
        if bytes.len() >= 4 && bytes[0..4] == *b"CEN2" {
            let mut net = sota_loader::load_v2_bytes(&bytes).map_err(|e| {
                NnueError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            // EXP-UNC-1: attach the optional value-uncertainty head (D40_UNC_HEAD
            // sidecar) before leaking. No-op when the env is unset → bit-identical.
            sota_loader::load_uncertainty_sidecar(&mut net.head);
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

/// The smallnet-gate cheap net (perf study #3 item #8), or `None` when the gate
/// is off. Env-driven — `ARCHON_SMALLNET` is a CENN-v2 path, typically a
/// HalfKA-only net — and loaded once, lazily, into a leaked `'static` (a single
/// load, not the per-game leak [`set_eval_file`] guards against, because the
/// `OnceLock` loads exactly once). `None` (env unset, blank, or an unreadable /
/// invalid file) ⇒ the gate is inert and the D40 eval is bit-identical to today
/// (sota bench 147691). Read once per search when a [`sota_context::SotaContext`]
/// is built (the eval boundary), never on the hot per-node path.
///
/// Only the cheap net's HalfKA channel is consumed — its FullThreats channel is
/// never built, which IS the speed win — so ANY CENN-v2 net works here (a
/// HalfKA-only net is simply the cheapest, and a full net doubles as a stand-in
/// for smoke-testing the gate plumbing before a dedicated cheap net is trained).
#[cfg(feature = "sota")]
pub fn active_smallnet() -> Option<&'static sota_loader::SotaNetwork> {
    use std::sync::OnceLock;
    static SMALLNET: OnceLock<Option<&'static sota_loader::SotaNetwork>> = OnceLock::new();
    *SMALLNET.get_or_init(|| {
        let path = std::env::var("ARCHON_SMALLNET").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        let bytes = std::fs::read(&path).ok()?;
        let net = sota_loader::load_v2_bytes(&bytes).ok()?;
        Some(Box::leak(Box::new(net)) as &'static sota_loader::SotaNetwork)
    })
}

/// The t-shirt gate's LARGE-tier net (S/M/L adaptive eval routing). A full-feature
/// CENN-v2 net run ONLY on CRITICAL positions — the ones the MEDIUM net is least
/// sure about (its EXP-UNC-1 σ̂ ≥ `ARCHON_LARGENET_SIGMA`) — spending extra eval
/// accuracy where it matters most. Env-driven (`ARCHON_LARGENET`), loaded once,
/// lazily, leaked. `None` (unset/blank/bad file) ⇒ no large tier ⇒ bit-identical.
///
/// NOTE (honest): a large tier only pays off if this net is genuinely MORE
/// ACCURATE than the medium net. Our one-box results say width is saturated and
/// eval is data-limited (`training-methods-study` / `sf16-gap-diagnosis`), so a
/// same-data wider net will NOT help — the tier's real value is as the deployment
/// vehicle for a beyond-SF16 fleet net or an ensemble. This is the routing
/// mechanism; the accuracy must come from the net itself.
#[cfg(feature = "sota")]
pub fn active_largenet() -> Option<&'static sota_loader::SotaNetwork> {
    use std::sync::OnceLock;
    static LARGENET: OnceLock<Option<&'static sota_loader::SotaNetwork>> = OnceLock::new();
    *LARGENET.get_or_init(|| {
        let path = std::env::var("ARCHON_LARGENET").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        let bytes = std::fs::read(&path).ok()?;
        let net = sota_loader::load_v2_bytes(&bytes).ok()?;
        Some(Box::leak(Box::new(net)) as &'static sota_loader::SotaNetwork)
    })
}

/// The compile-time-embedded default net (distribution builds only, `embed-net`
/// feature). Loaded and leaked once. The bytes are a build asset validated at
/// build time, so a load failure is a broken build — not user input — hence the
/// `expect`. Read once per search (never on the hot path), same as [`active_net`].
///
/// The net to embed is chosen at BUILD time: set `ARCHON_EMBED_NET` to a
/// v1 CENN path relative to the crate root (its width must match the
/// build's `nnue-l1-*` feature, or the expect fires on first use). The
/// variable is required whenever `embed-net` is on — an explicit choice
/// beats a stale hardcoded default now that multiple widths ship.
#[cfg(feature = "embed-net")]
fn embedded_net() -> &'static Network {
    use std::sync::OnceLock;
    static EMBEDDED: OnceLock<&'static Network> = OnceLock::new();
    *EMBEDDED.get_or_init(|| {
        const BYTES: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/",
            env!(
                "ARCHON_EMBED_NET",
                "embed-net builds require ARCHON_EMBED_NET=<path to .cenn, relative to the crate root>"
            )
        ));
        Box::leak(Box::new(
            load_bytes(BYTES).expect("embedded net is a valid CENN file of this build's width"),
        )) as &'static Network
    })
}
