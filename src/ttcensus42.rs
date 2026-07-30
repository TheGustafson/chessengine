//! Perf study 42 — the **exact** transposition-table census.
//!
//! Thirty-one perf studies had measured everything in this engine except the
//! other big random-access structure. Study 33's DWARF inline census put TT
//! `probe` at 3.17% of the `search::<NonPv>` symbol, but a symbol share says
//! nothing about the misses a random-access table causes, and the IBS
//! address-attribution this study runs answers "what share of DRAM traffic" —
//! not "per probe". Converting one into the other needs the denominator, and
//! nothing in the engine reported it.
//!
//! This module reports it: probes, hits, how deep the tag scan went, saves and
//! their accept/reject split, and — the one that decides a LAYOUT question —
//! how often two consecutive probes land in the **same 64-byte cache line**.
//!
//! Why that last one matters. A [`crate::tt::Cluster`] is 32 bytes and
//! `align(32)`, so a probe touches exactly one line and never straddles
//! (verified independently from data addresses by `tools/perf42/align42.py`).
//! But the hardware still fetches 64 bytes, so **half of every line the TT
//! pulls holds the *neighbouring* cluster**, which is useful only if some
//! other probe asks for it while the line is still resident. `SAME_LINE`
//! counts exactly those. If it is ~0, the TT gets 3 candidate entries per line
//! fetched where a 64-byte cluster would get 6 for the same traffic.
//!
//! Compiled only under `--features ttcensus42`. The counters are relaxed
//! atomics on a single-threaded bench: they perturb TIMING and nothing else —
//! the search tree, the node count and the eval hash are untouched, which
//! `tools/perf42/ident42.sh` checks. Never shipped, never a timing reference.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
    };
}

counters!(
    PROBES,
    TAG_MATCH,
    HITS,
    SLOT0,
    SLOT1,
    SLOT2,
    SAVES,
    SAVE_ACCEPT,
    PENALIZE,
    PREFETCH,
    SAME_LINE,
    SAME_CLUSTER,
);

/// The previous probe's cluster address, so `SAME_LINE` can be counted without
/// a trace. Single-threaded instrument builds only; a race would merely
/// mis-count one comparison.
static LAST_CLUSTER: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn bump(c: &AtomicU64) {
    c.fetch_add(1, Relaxed);
}

/// Record a probe's target cluster address and classify it against the
/// previous probe's: same 32-byte cluster, or merely the same 64-byte line
/// (i.e. the sibling cluster the fetch already paid for).
#[inline]
pub fn note_cluster(addr: usize) {
    let prev = LAST_CLUSTER.swap(addr as u64, Relaxed) as usize;
    if prev == addr {
        bump(&SAME_CLUSTER);
    } else if prev & !63 == addr & !63 {
        bump(&SAME_LINE);
    }
}

/// Emit the census as `key=value` lines to the path in `ARCHON_TTCENSUS`
/// (default `ttcensus42.txt`).
///
/// **A FILE, not a stream** — the same rule study 29's census learned the hard
/// way: `bench` writes its fingerprint to stderr and keeps stdout a clean UCI
/// stream, and `spawned_bench_*_byte_identical` asserts the exact line count of
/// that stderr. An instrument that appends to either breaks the format
/// contract it is supposed to be neutral to.
pub fn dump(nodes: u64) {
    use std::io::Write;
    let path = std::env::var("ARCHON_TTCENSUS").unwrap_or_else(|_| "ttcensus42.txt".into());
    let Ok(mut f) = std::fs::File::create(&path) else {
        return;
    };
    let _ = writeln!(f, "TTCENSUS nodes={nodes}");
    for (name, c) in [
        ("probes", &PROBES),
        ("tag_match", &TAG_MATCH),
        ("hits", &HITS),
        ("slot0", &SLOT0),
        ("slot1", &SLOT1),
        ("slot2", &SLOT2),
        ("saves", &SAVES),
        // Rejected writes are `saves - save_accept`; counting them at the
        // `else if` would have meant wrapping the condition in a block
        // expression, i.e. perturbing the very branch being measured.
        ("save_accept", &SAVE_ACCEPT),
        ("penalize", &PENALIZE),
        ("prefetch", &PREFETCH),
        ("same_line", &SAME_LINE),
        ("same_cluster", &SAME_CLUSTER),
    ] {
        let v = c.load(Relaxed);
        let _ = writeln!(f, "TTCENSUS {name}={v} {name}_per_node={:.6}", v as f64 / nodes as f64);
    }
}
