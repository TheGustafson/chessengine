//! Perf study 34 — the move-ordering census (instrument only, feature `census34`).
//!
//! Study 23 §3 measured how much work `init_quiets` DOES (22.6 quiets scored
//! per call, 12.3 sort insertions, 36.7 shift steps). It never measured how
//! much of that work is ever USED. This module answers that: for every picker
//! that generated quiets, how far into the scored-and-sorted list the search
//! actually walked before the node ended.
//!
//! Everything here is `Relaxed` atomics on a single-threaded bench, bumped on
//! paths that make no search decision — the tree is bit-identical to the
//! shipped build (checked with `ident34.sh`), so the ratios below describe the
//! REAL search, not the instrument's.
//!
//! Not compiled into any shipped configuration.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Histogram bucket count: values `0..=62` are exact, `63` is "63 or more".
pub const HB: usize = 64;

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
    };
}

counters! {
    // --- init_quiets: work done ---
    IQ_CALLS,      // init_quiets calls
    IQ_SCORED,     // quiets generated + scored
    IQ_PREFIX,     // elements the partial sort placed in the sorted prefix
    IQ_INSERTS,    // sort insertions (== IQ_PREFIX - IQ_CALLS, cross-check)
    IQ_SHIFTS,     // sort shift steps
    // --- quiet region: work used ---
    GQ_SCANNED,    // elements `select` inspected in the GoodQuiet stage
    GQ_EMITTED,    // moves emitted from GoodQuiet
    BQ_SCANNED,    // elements inspected in BadQuiet
    BQ_EMITTED,    // moves emitted from BadQuiet
    Q_SKIPPED,     // pickers whose quiets were cut short by skip_quiet_moves()
    Q_EXHAUSTED,   // pickers that walked the entire quiet region
    // --- MovePicker::new / pseudo_legal(tt_move) (study 23 remainder item 6) ---
    MP_NEW,            // MovePicker::new calls
    MP_TT_CAND,        // ... with a candidate TT move (tt_move.is_ok())
    MP_TT_EVASION_GEN, // ... in check: pseudo_legal runs a FULL evasion generation
    MP_TT_NONNORMAL,   // ... non-Normal kind: pseudo_legal runs a FULL generation
    MP_TT_USABLE,      // ... and the TT move survived validation
    IE_CALLS,          // init_evasions calls (the SECOND evasion generation)
    // --- captures, for comparison ---
    IC_CALLS,
    IC_SCORED,
    GC_SCANNED,
    GC_EMITTED,
    // --- search-side ordering quality ---
    FH_NODES,      // main-search beta cutoffs
    FH_MOVES,      // sum of move_count at those cutoffs
    NOFH_NODES,    // main-search move-loop nodes that did NOT fail high
}

/// `HIST_GQ[k]` = pickers that inspected exactly `k` elements in GoodQuiet.
pub static HIST_GQ: [AtomicU64; HB] = [const { AtomicU64::new(0) }; HB];
/// `HIST_NQ[k]` = init_quiets calls that generated exactly `k` quiets.
pub static HIST_NQ: [AtomicU64; HB] = [const { AtomicU64::new(0) }; HB];
/// `HIST_PFX[k]` = init_quiets calls whose sorted prefix was `k` long.
pub static HIST_PFX: [AtomicU64; HB] = [const { AtomicU64::new(0) }; HB];
/// `HIST_FH[k]` = beta cutoffs that happened on move number `k`.
pub static HIST_FH: [AtomicU64; HB] = [const { AtomicU64::new(0) }; HB];

#[inline]
pub fn bump(c: &AtomicU64, by: u64) {
    c.fetch_add(by, Relaxed);
}

#[inline]
pub fn hist(h: &[AtomicU64; HB], k: usize) {
    h[k.min(HB - 1)].fetch_add(1, Relaxed);
}

fn g(c: &AtomicU64) -> f64 {
    c.load(Relaxed) as f64
}

fn dump(name: &str, h: &[AtomicU64; HB]) {
    let total: u64 = h.iter().map(|c| c.load(Relaxed)).sum();
    if total == 0 {
        return;
    }
    // Cumulative share, so "what fraction stops by k" reads straight off.
    let mut acc = 0u64;
    let mut parts = Vec::new();
    for (k, c) in h.iter().enumerate() {
        let v = c.load(Relaxed);
        if v == 0 && acc == total {
            break;
        }
        acc += v;
        parts.push(format!(
            "{k}:{:.2}%/{:.2}%",
            100.0 * v as f64 / total as f64,
            100.0 * acc as f64 / total as f64
        ));
    }
    eprintln!("C34 {name} n={total} {}", parts.join(" "));
}

/// Print the census to stderr. Called once at the end of `bench`.
pub fn report(nodes: u64) {
    let n = nodes.max(1) as f64;
    let calls = g(&IQ_CALLS).max(1.0);
    let scored = g(&IQ_SCORED);
    let prefix = g(&IQ_PREFIX);
    let gq_s = g(&GQ_SCANNED);
    let gq_e = g(&GQ_EMITTED);
    let bq_s = g(&BQ_SCANNED);
    let bq_e = g(&BQ_EMITTED);
    eprintln!("C34 nodes={nodes}");
    eprintln!(
        "C34 init_quiets calls={:.0} /node={:.4} scored/call={:.2} prefix/call={:.2} \
         inserts/call={:.2} shifts/call={:.2}",
        calls,
        calls / n,
        scored / calls,
        prefix / calls,
        g(&IQ_INSERTS) / calls,
        g(&IQ_SHIFTS) / calls
    );
    eprintln!(
        "C34 quiets  scored/node={:.3} gq_scanned/call={:.2} gq_emitted/call={:.2} \
         bq_scanned/call={:.2} bq_emitted/call={:.2}",
        scored / n,
        gq_s / calls,
        gq_e / calls,
        bq_s / calls,
        bq_e / calls
    );
    eprintln!(
        "C34 USE     scanned/scored={:.2}% scanned/prefix={:.2}% emitted/scored={:.2}% \
         skipped={:.2}% exhausted={:.2}%",
        100.0 * (gq_s + bq_s) / scored.max(1.0),
        100.0 * gq_s / prefix.max(1.0),
        100.0 * (gq_e + bq_e) / scored.max(1.0),
        100.0 * g(&Q_SKIPPED) / calls,
        100.0 * g(&Q_EXHAUSTED) / calls
    );
    let ccalls = g(&IC_CALLS).max(1.0);
    eprintln!(
        "C34 captures calls={:.0} /node={:.4} scored/call={:.2} gc_scanned/call={:.2} \
         gc_emitted/call={:.2}",
        ccalls,
        ccalls / n,
        g(&IC_SCORED) / ccalls,
        g(&GC_SCANNED) / ccalls,
        g(&GC_EMITTED) / ccalls
    );
    let mpn = g(&MP_NEW).max(1.0);
    eprintln!(
        "C34 mp_new  calls={:.0} /node={:.4} tt_cand={:.0} usable={:.0} \
         evasion_gen={:.0} (/node={:.4}) nonnormal_gen={:.0} (/node={:.4}) \
         init_evasions={:.0} (/node={:.4})",
        mpn,
        mpn / n,
        // Printed as raw counts too: the interesting question is whether
        // `pseudo_legal` ever REJECTS a TT move, and a percentage hides that.
        g(&MP_TT_CAND),
        g(&MP_TT_USABLE),
        g(&MP_TT_EVASION_GEN),
        g(&MP_TT_EVASION_GEN) / n,
        g(&MP_TT_NONNORMAL),
        g(&MP_TT_NONNORMAL) / n,
        g(&IE_CALLS),
        g(&IE_CALLS) / n
    );
    let fh = g(&FH_NODES).max(1.0);
    eprintln!(
        "C34 ordering fh_nodes={:.0} nofh_nodes={:.0} fh_rate={:.2}% avg_move_at_cut={:.3}",
        fh,
        g(&NOFH_NODES),
        100.0 * fh / (fh + g(&NOFH_NODES)),
        g(&FH_MOVES) / fh
    );
    dump("HIST_GQ", &HIST_GQ);
    dump("HIST_NQ", &HIST_NQ);
    dump("HIST_PFX", &HIST_PFX);
    dump("HIST_FH", &HIST_FH);
}
