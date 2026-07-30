//! Perf study 29 — the **exact** accumulator load census.
//!
//! Study 28 attributed 886.0 L1 loads/node to the fold kernels by SAMPLING
//! (`perf record`, per-symbol) and then refused to split them into
//! accumulator streaming versus weight rows, because a symbol share cannot
//! see inside a symbol. This module is the instrument that can: it counts,
//! with exact integers, every call to every accumulator kernel and the number
//! of weight ROWS that call folded. The load count then follows from the
//! kernel's static shape (verified by disassembly — see the study doc), with
//! no sampling anywhere in the chain.
//!
//! Compiled only under `--features foldcount`. The counters are plain relaxed
//! atomics on a single-threaded bench, so they cost one `lock`-free `add` per
//! kernel call — which perturbs timing and *nothing else*: the search tree,
//! the node count and the eval hash are untouched, and the study verifies
//! that.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// One counted site: how many times it ran, and how many weight rows it folded.
pub struct Site {
    pub calls: AtomicU64,
    pub rows: AtomicU64,
}

impl Site {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
        }
    }
    #[inline]
    pub fn hit(&self, rows: usize) {
        self.calls.fetch_add(1, Relaxed);
        self.rows.fetch_add(rows as u64, Relaxed);
    }
    fn read(&self) -> (u64, u64) {
        (self.calls.load(Relaxed), self.rows.load(Relaxed))
    }
}

/// `apply_halfka_deltas_into` — the HalfKA parent→child fold (forward + backward).
pub static HKA_INTO: Site = Site::new();
/// `apply_halfka_deltas_dual` — the Finny *warm* refresh (two destinations).
pub static HKA_DUAL: Site = Site::new();
/// `apply_halfka_deltas` — the Finny *cold* refresh (in place, once per king key).
pub static HKA_INPLACE: Site = Site::new();
/// `apply_threat_deltas_into` — the FullThreats parent→child fold.
pub static THR_INTO: Site = Site::new();
/// The threat fold's empty-delta branch: a hand-chunked 2560-byte copy.
pub static THR_COPY: Site = Site::new();
/// Perf study 38's parent alias: a ply whose delta was empty for BOTH
/// perspectives, so the entry was never written at all.
pub static THR_ALIAS: Site = Site::new();
/// `activate` — the head's accumulator merge. Reads FOUR `[i16; ACC]` arrays
/// (two perspectives x two channels) and writes one `[u8; ACC]`. This is the
/// 10.24 KB/ply Stockfish carries as 4.10 KB.
pub static ACTIVATE: Site = Site::new();
/// `SotaHead::forward_into` — one full evaluation.
pub static EVAL: Site = Site::new();

/// Shape histogram for the HalfKA fold: index `adds * 4 + subs`, capped at 15.
pub static HKA_SHAPE: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
/// Shape histogram for the threat fold: `min(adds + subs, 31)`.
pub static THR_SHAPE: [AtomicU64; 32] = [const { AtomicU64::new(0) }; 32];

/// Perf study 38 — the JOINT (per-perspective) shape of one ply's threat fold,
/// indexed `need0 | need1<<1 | empty0<<2 | empty1<<3`.
///
/// The question it answers: can a parent→child alias be taken on the WHOLE
/// entry (both perspectives at once, so `at_full`'s `&[[i16; ACC]; 2]` return
/// type survives), or must it be per-perspective (which forces the head's
/// merge kernel to take two separate rows)? Only the `1,1,1,1` cell licenses
/// the cheap form.
pub static THR_JOINT: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

#[inline]
pub fn note_hka_shape(adds: usize, subs: usize) {
    let i = (adds.min(3) * 4 + subs.min(3)).min(15);
    HKA_SHAPE[i].fetch_add(1, Relaxed);
}

#[inline]
pub fn note_thr_shape(rows: usize) {
    THR_SHAPE[rows.min(31)].fetch_add(1, Relaxed);
}

#[inline]
pub fn note_thr_joint(need: [bool; 2], lens: [(usize, usize); 2]) {
    let empty = |p: usize| lens[p].0 == 0 && lens[p].1 == 0;
    let i = usize::from(need[0])
        | usize::from(need[1]) << 1
        | usize::from(empty(0)) << 2
        | usize::from(empty(1)) << 3;
    THR_JOINT[i].fetch_add(1, Relaxed);
}

/// Named address regions, so an IBS data-address sample can be attributed to
/// the accumulator stack or to a weight table. Sixteen slots is plenty; the
/// registry is write-once at construction and read once at dump.
const REGIONS: usize = 16;
static REGION_LO: [AtomicU64; REGIONS] = [const { AtomicU64::new(0) }; REGIONS];
static REGION_HI: [AtomicU64; REGIONS] = [const { AtomicU64::new(0) }; REGIONS];
static REGION_NAME: std::sync::Mutex<Vec<(usize, &'static str)>> =
    std::sync::Mutex::new(Vec::new());
static REGION_N: AtomicU64 = AtomicU64::new(0);

/// Record `[ptr, ptr + bytes)` under `name`.
pub fn region(name: &'static str, ptr: *const u8, bytes: usize) {
    let i = REGION_N.fetch_add(1, Relaxed) as usize;
    if i >= REGIONS {
        return;
    }
    REGION_LO[i].store(ptr as u64, Relaxed);
    REGION_HI[i].store(ptr as u64 + bytes as u64, Relaxed);
    if let Ok(mut g) = REGION_NAME.lock() {
        g.push((i, name));
    }
}

/// Emit the census as `key=value` lines to the path in `ARCHON_FOLDCOUNT`
/// (default `foldcount.txt`). Parsed by `tools/perf29/census29.py`.
///
/// **A FILE, not a stream.** `bench` writes its whole fingerprint to STDERR
/// and keeps stdout a clean UCI stream, and `spawned_bench_*_byte_identical`
/// assert the exact line count of that stderr — so an instrument that appends
/// to either stream breaks the format contract it is supposed to be neutral
/// to. It did, in this study's first draft.
pub fn dump(nodes: u64) {
    use std::io::Write;
    let path = std::env::var("ARCHON_FOLDCOUNT").unwrap_or_else(|_| "foldcount.txt".into());
    let Ok(mut f) = std::fs::File::create(&path) else {
        return;
    };
    macro_rules! emit {
        ($($arg:tt)*) => { let _ = writeln!(f, $($arg)*); };
    }
    let width = super::sota_features::ACC;
    emit!("FOLDCOUNT nodes={nodes} acc_width={width}");
    for (name, site) in [
        ("hka_into", &HKA_INTO),
        ("hka_dual", &HKA_DUAL),
        ("hka_inplace", &HKA_INPLACE),
        ("thr_into", &THR_INTO),
        ("thr_copy", &THR_COPY),
        ("thr_alias", &THR_ALIAS),
        ("activate", &ACTIVATE),
        ("eval", &EVAL),
    ] {
        let (calls, rows) = site.read();
        emit!("FOLDCOUNT {name}.calls={calls} {name}.rows={rows}");
    }
    for a in 0..4 {
        for s in 0..4 {
            let n = HKA_SHAPE[a * 4 + s].load(Relaxed);
            if n != 0 {
                emit!("FOLDCOUNT hka_shape.{a}a{s}s={n}");
            }
        }
    }
    for (r, cell) in THR_SHAPE.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT thr_shape.{r}={n}");
        }
    }
    for (i, cell) in THR_JOINT.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!(
                "FOLDCOUNT thr_joint.n{}{}e{}{}={n}",
                i & 1,
                (i >> 1) & 1,
                (i >> 2) & 1,
                (i >> 3) & 1
            );
        }
    }
    if let Ok(g) = REGION_NAME.lock() {
        for &(i, name) in g.iter() {
            let (lo, hi) = (REGION_LO[i].load(Relaxed), REGION_HI[i].load(Relaxed));
            emit!("FOLDREGION {name} lo={lo:#x} hi={hi:#x} bytes={}", hi - lo);
        }
    }
}
