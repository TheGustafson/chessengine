//! NNUE inference (design/11 §3.4): the CENN network container ([`loader`]),
//! the perspective feature transformer and its incremental accumulator
//! ([`accumulator`]), and the quantized SCReLU/CReLU forward pass ([`simd`]).
//!
//! **Architecture** (`768 → 2×L1 → 1`, single perspective pair): 768 sparse
//! binary inputs (`2 colors × 6 piece types × 64 squares`) feed a shared
//! feature transformer producing two [`L1`]-wide accumulators. The output
//! layer concatenates the two activated accumulators — **side-to-move half
//! first** — and dots with the output weights (§3.4.1).
//!
//! **Quantization** (bullet-standard, §3.4.3): FT weights/bias scaled by
//! [`QA`], output weights by [`QB`], centipawn output by [`SCALE`].

pub mod accumulator;
/// Cache-line-aligned heap slices for the weight tables.
#[cfg(feature = "sota")]
pub mod aligned;
pub mod loader;
pub mod network;
/// D47 deployed policy head (CPL1 sidecar + selected-row scorer).
#[cfg(feature = "sota")]
pub mod policy_head;
pub mod policy_vocab;
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
/// Per-thread SOTA eval context: accumulators + head forward (design/11 §12).
#[cfg(feature = "sota")]
pub mod sota_context;
/// The SOTA net's SF feature front-end (HalfKAv2_hm + FullThreats, D40).
#[cfg(feature = "sota")]
pub mod sota_features;
/// The SOTA net's head forward (D40, design/11 §12).
#[cfg(feature = "sota")]
pub mod sota_head;
/// The CENN-v2 container loader for D40 SOTA nets (design/11 §12.6).
#[cfg(feature = "sota")]
pub mod sota_loader;
/// Runtime-dispatched fused accumulator kernels for the SOTA transformers.
#[cfg(feature = "sota")]
mod sota_simd;
/// The SOTA net's incremental FullThreats accumulator (D40, design/11 §12.9.1).
#[cfg(feature = "sota")]
pub mod sota_threats;

// --- Architecture dimensions (design/11 §3.4.1) ----------------------------

/// Number of sparse input features: `2 colors × 6 piece types × 64 squares`.
pub const INPUT: usize = 768;

/// Feature-transformer / accumulator width (the "L1" hidden size), 256 by
/// default. A build loads only nets of its own width (the loader checks the
/// CENN header `l1`); everything downstream derives from this const, so the
/// `nnue-l1-*` features are one-line widenings.
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

// `DirtyPiece` is defined in `board` (its constructor reads `Board`'s
// private castling geometry); re-exported so `eval::nnue::DirtyPiece` resolves.
pub use crate::board::DirtyPiece;

// --- Active-net registry (design/11 §3.4.5: EvalFile, opt-in) ----------------

use std::path::PathBuf;
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;

/// The process-wide active NNUE net (design/11 §3.4.5). `None` ⇒ NNUE OFF ⇒
/// PSQT (the default). Set via UCI `EvalFile` or the `bench` net argument;
/// read once per search at the eval boundary, never on the hot per-node
/// path. `RwLock` (not `AtomicPtr`) keeps the crate `unsafe`-free (D14).
static ACTIVE_NET: RwLock<Option<&'static Network>> = RwLock::new(None);

/// Process-wide active SOTA (D40) net. Mirrors [`ACTIVE_NET`] for CENN-v2 files.
/// When a CENN-v2 net is installed the v1 ACTIVE_NET is cleared, and vice versa.
#[cfg(feature = "sota")]
static ACTIVE_SOTA_NET: RwLock<Option<&'static sota_loader::SotaNetwork>> = RwLock::new(None);
/// D47: the deployed policy head (CPL1 sidecar auto-loaded beside the CENN;
/// `None` ⇒ classical ordering, bit-identical to pre-D47).
#[cfg(feature = "sota")]
static ACTIVE_POLICY_HEAD: RwLock<Option<&'static policy_head::PolicyHead>> = RwLock::new(None);

/// The active policy head, if a CPL1 sidecar accompanied the loaded net.
#[cfg(feature = "sota")]
pub fn active_policy_head() -> Option<&'static policy_head::PolicyHead> {
    *ACTIVE_POLICY_HEAD.read().unwrap_or_else(|e| e.into_inner())
}

/// Install the CPL1 sidecar sitting beside `eval_path` into the active slot
/// (used by direct-load tools like `nnue-eval`; the UCI install path does
/// this automatically). Returns whether a head is now active.
#[cfg(feature = "sota")]
pub fn install_policy_sidecar(eval_path: &std::path::Path) -> bool {
    let head = policy_head::PolicyHead::sibling_of(eval_path).map(|h| &*Box::leak(Box::new(h)));
    let loaded = head.is_some();
    *ACTIVE_POLICY_HEAD
        .write()
        .unwrap_or_else(|e| e.into_inner()) = head;
    loaded
}
#[cfg(feature = "sota")]
static ACTIVE_SIMPLE_HALFKA_NET: RwLock<Option<&'static simple_halfka::SimpleHalfkaNetwork>> =
    RwLock::new(None);
#[cfg(feature = "sota")]
static ACTIVE_PROTECTED_HALFKA_NET: RwLock<
    Option<&'static protected_halfka::ProtectedHalfkaNetwork>,
> = RwLock::new(None);

/// Identity (canonical path, size, mtime) of the file behind the installed
/// net. NOTE: match runners re-send `EvalFile` before EVERY game and each
/// install leaks its parsed net to `'static` — without this guard a long
/// match leaks one full net per game until the OOM killer fires. A same-file
/// re-send is a no-op; a changed file at the same path still reloads. `None`
/// mtime ⇒ never treated as already-loaded (redundant loads over staleness).
static LOADED_FILE: Mutex<Option<(PathBuf, u64, Option<SystemTime>)>> = Mutex::new(None);

/// The (canonical path, size, mtime) identity of `path` right now.
fn file_identity(path: &str) -> Result<(PathBuf, u64, Option<SystemTime>), NnueError> {
    let canonical = std::fs::canonicalize(path).map_err(NnueError::Io)?;
    let meta = std::fs::metadata(&canonical).map_err(NnueError::Io)?;
    Ok((canonical, meta.len(), meta.modified().ok()))
}

/// Install (or clear) the `EvalFile` net. An empty/blank path turns NNUE OFF
/// (back to PSQT); a file already active is a no-op (see [`LOADED_FILE`]).
/// With `sota`, the peeked magic bytes route to the right loader (installing
/// a SOTA net clears the v1 slot and vice versa). Loader errors are returned
/// for a clean UCI `info string`; never panics on a user-supplied file.
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
    // A failed install may leave the slots partially cleared — drop the
    // cached identity so the next call always does a real load.
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
        if bytes.len() >= 4
            && (bytes[0..4] == sota_loader::MAGIC_V2 || bytes[0..4] == sota_loader::MAGIC_V5)
        {
            let mut net = sota_loader::load_v2_bytes(&bytes).map_err(|e| {
                NnueError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            // Attach the optional sidecar heads (uncertainty, WDL) before
            // leaking; no-op when unset → bit-identical.
            sota_loader::load_uncertainty_sidecar(&mut net.head);
            sota_loader::load_wdl_sidecar(&mut net.head);
            let leaked: &'static sota_loader::SotaNetwork = Box::leak(Box::new(net));
            *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = Some(leaked);
            // Auto-load the CPL1 policy sidecar beside the CENN; absent =>
            // None => classical ordering.
            *ACTIVE_POLICY_HEAD
                .write()
                .unwrap_or_else(|e| e.into_inner()) =
                policy_head::PolicyHead::sibling_of(std::path::Path::new(path))
                    .map(|h| &*Box::leak(Box::new(h)));
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
        // Not a SOTA container → fall through to v1 loader. Clear SOTA + policy.
        *ACTIVE_SOTA_NET.write().unwrap_or_else(|e| e.into_inner()) = None;
        *ACTIVE_POLICY_HEAD
            .write()
            .unwrap_or_else(|e| e.into_inner()) = None;
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

/// Which evaluator a `go` will actually use, as a short human-readable tag.
/// Exists because a failed net load is INVISIBLE — a PSQT search produces
/// entirely plausible numbers — so the UCI layer reports this on every
/// `EvalFile` set and at the start of every search.
pub fn active_eval_kind() -> &'static str {
    #[cfg(feature = "sota")]
    {
        if active_sota_net().is_some() {
            return "NNUE(sota-v2)";
        }
        if active_simple_halfka_net().is_some() {
            return "NNUE(halfka)";
        }
        if active_protected_halfka_net().is_some() {
            return "NNUE(halfka-prot)";
        }
    }
    if active_net().is_some() {
        return "NNUE(v1)";
    }
    "PSQT"
}

/// The active v1 net, or `None` when NNUE is off (⇒ PSQT). With `embed-net`,
/// an explicit `EvalFile` still wins, but when none is set the embedded
/// default net is used instead of falling back to PSQT.
pub fn active_net() -> Option<&'static Network> {
    let from_eval_file = *ACTIVE_NET.read().unwrap_or_else(|e| e.into_inner());
    // On a `sota` build the embedded bytes are a sota net and the v1 loader
    // would panic on them; the sota fallback lives in `active_sota_net()` —
    // this one is for v1-only builds.
    #[cfg(all(feature = "embed-net", not(feature = "sota")))]
    {
        return from_eval_file.or_else(|| Some(embedded_net()));
    }
    #[cfg(not(all(feature = "embed-net", not(feature = "sota"))))]
    from_eval_file
}

/// The active D40 SOTA net, or `None`. Read at each search's eval boundary.
/// With `embed-net`, an explicit `EvalFile` still wins, but when none is set
/// the embedded net is used instead of silently falling back to PSQT.
#[cfg(feature = "sota")]
pub fn active_sota_net() -> Option<&'static sota_loader::SotaNetwork> {
    let from_eval_file = *ACTIVE_SOTA_NET.read().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "embed-net")]
    {
        return from_eval_file.or_else(|| Some(embedded_sota_net()));
    }
    #[cfg(not(feature = "embed-net"))]
    from_eval_file
}

/// The compile-time-embedded SOTA net; reads the same `ARCHON_EMBED_NET`
/// path as `embedded_net`, so one env var serves both formats. Panics on an
/// arch mismatch — deliberate: the alternative is a binary that silently
/// evaluates with PSQT.
#[cfg(all(feature = "sota", feature = "embed-net"))]
fn embedded_sota_net() -> &'static sota_loader::SotaNetwork {
    use std::sync::OnceLock;
    static EMBEDDED_SOTA: OnceLock<&'static sota_loader::SotaNetwork> = OnceLock::new();
    *EMBEDDED_SOTA.get_or_init(|| {
        const BYTES: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/",
            env!(
                "ARCHON_EMBED_NET",
                "embed-net builds require ARCHON_EMBED_NET=<path to .cenn, relative to the crate root>"
            )
        ));
        let mut net = sota_loader::load_v2_bytes(BYTES)
            .expect("embedded net is a valid CENN file matching this build's arch features");
        sota_loader::load_uncertainty_sidecar(&mut net.head);
        sota_loader::load_wdl_sidecar(&mut net.head);
        Box::leak(Box::new(net)) as &'static sota_loader::SotaNetwork
    })
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

/// The smallnet-gate cheap net, or `None` when the gate is off. Env-driven
/// (`ARCHON_SMALLNET`, a CENN-v2 path) and loaded once, lazily, into a
/// leaked `'static`. `None` (unset/blank/bad file) ⇒ the gate is inert and
/// the D40 eval is bit-identical. Read once per search, never per node.
/// Only the cheap net's HalfKA channel is consumed — its FullThreats channel
/// is never built, which IS the speed win — so ANY CENN-v2 net works here.
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

/// The t-shirt gate's LARGE-tier net (S/M/L adaptive eval routing): a
/// full-feature CENN-v2 net run ONLY on positions the MEDIUM net is least
/// sure about (σ̂ ≥ `ARCHON_LARGENET_SIGMA`). Env-driven (`ARCHON_LARGENET`),
/// loaded once, lazily, leaked; `None` ⇒ no large tier ⇒ bit-identical.
/// This is the routing mechanism only — the accuracy must come from the net.
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

/// The compile-time-embedded default net (`embed-net` builds), loaded and
/// leaked once. The bytes are a build asset, so a load failure is a broken
/// build — hence the `expect`. `ARCHON_EMBED_NET` (crate-root-relative path,
/// width matching the build's `nnue-l1-*` feature) picks the net at BUILD
/// time and is required whenever `embed-net` is on.
#[cfg(all(feature = "embed-net", not(feature = "sota")))]
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
