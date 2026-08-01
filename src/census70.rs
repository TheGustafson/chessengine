//! Perf study 70 — the move-ordering QUALITY census (instrument only,
//! feature `census70`).
//!
//! Study 34's `census34` measured how much of `init_quiets`' work the search
//! consumes, and printed one flat cutoff-index histogram. That aggregate is
//! not actionable: a mean cutoff index of 1.6 is equally consistent with
//! "94 % cut on move 1, 6 % on move 11" and with "80 % on move 1, 20 % on
//! move 4", and those two worlds want opposite fixes. This module splits the
//! same population three ways at once —
//!
//! - **node kind** (PV / NonPV / qsearch): NonPV cut-nodes are where the
//!   tree's width actually lives,
//! - **remaining depth**: ordering at `d=1` is a leaf-count question, at
//!   `d=10` it is a subtree-count question,
//! - **the `MovePicker` stage that emitted the cutting move**: TT, good
//!   capture, good quiet, bad capture, bad quiet, evasion.
//!
//! and adds the two populations study 34 never counted: the **all-nodes**
//! (no cutoff) and, within them, where the eventual best move sat.
//!
//! Everything is `Relaxed` atomics on paths that make no search decision, so
//! the tree is bit-identical to the shipped build (checked by
//! `tools/perf70/ident70.sh`). Never shipped; not a timing reference.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Node kinds. The root is split out because it never touches `MovePicker`
/// at all (it walks `RootMoves`), and in-check nodes are split out because
/// they run a *different pipeline* — the evasion one, whose scoring shares
/// nothing with the main quiet/capture scoring.
pub const KIND_ROOT: usize = 0;
pub const KIND_PV: usize = 1;
pub const KIND_PV_CHK: usize = 2;
pub const KIND_NONPV: usize = 3;
pub const KIND_NONPV_CHK: usize = 4;
pub const KIND_QS: usize = 5;
pub const KIND_QS_CHK: usize = 6;
pub const KINDS: usize = 7;
pub const KIND_NAMES: [&str; KINDS] = [
    "ROOT", "PV", "PVCHK", "NONPV", "NONPVCHK", "QS", "QSCHK",
];

/// Depth buckets: 0 = qsearch/`<= 0`, 1..=23 verbatim, 24 = "24 or deeper".
pub const DEPTHS: usize = 25;
/// Move-index buckets: 1..=62 verbatim, 63 = "63rd or later". Slot 0 is the
/// "no move" case (terminal nodes) and is meaningful for the all-node table.
pub const IDXS: usize = 64;
/// `MovePicker::Stage` discriminants.
pub const SRCS: usize = 16;
pub const SRC_NAMES: [&str; SRCS] = [
    "MainTt",
    "CaptureInit",
    "GoodCapture",
    "QuietInit",
    "GoodQuiet",
    "BadCapture",
    "BadQuiet",
    "EvasionTt",
    "EvasionInit",
    "Evasion",
    "ProbcutTt",
    "ProbcutInit",
    "Probcut",
    "QsearchTt",
    "QcaptureInit",
    "Qcapture",
];

/// `const {}` block expressions are the stable way to repeat a non-`Copy`
/// value in an array initializer: each element is a fresh evaluation of the
/// block rather than a copy of one value (`AtomicU64` is deliberately not
/// `Copy`). Nesting them builds the 2-D and 3-D tables the same way.
macro_rules! zeros {
    ($n:expr) => {
        [const { AtomicU64::new(0) }; $n]
    };
    ($n:expr, $m:expr) => {
        [const { zeros!($m) }; $n]
    };
    ($n:expr, $m:expr, $k:expr) => {
        [const { zeros!($m, $k) }; $n]
    };
}

/// Beta cutoffs by \[kind\]\[depth\]\[cutoff move index\].
pub static CUT_IDX: [[[AtomicU64; IDXS]; DEPTHS]; KINDS] = zeros!(KINDS, DEPTHS, IDXS);
/// Beta cutoffs by \[kind\]\[emitting `MovePicker` stage\].
pub static CUT_SRC: [[AtomicU64; SRCS]; KINDS] = zeros!(KINDS, SRCS);
/// Nodes whose move loop ran out without a cutoff, by \[kind\]\[depth\],
/// with the moves they searched.
pub static ALL_N: [[AtomicU64; DEPTHS]; KINDS] = zeros!(KINDS, DEPTHS);
pub static ALL_MOVES: [[AtomicU64; DEPTHS]; KINDS] = zeros!(KINDS, DEPTHS);
/// At those nodes, the index of the move that ended up best (0 = none).
pub static ALL_BEST_IDX: [[AtomicU64; IDXS]; KINDS] = zeros!(KINDS, IDXS);
/// Node entries with / without a usable TT move to order on, by
/// \[kind\]\[depth\].
pub static TT_YES: [[AtomicU64; DEPTHS]; KINDS] = zeros!(KINDS, DEPTHS);
pub static TT_NO: [[AtomicU64; DEPTHS]; KINDS] = zeros!(KINDS, DEPTHS);

/// **The cost of mis-ordering.** An LMR scout that beats alpha is the search
/// saying "I reduced a move I should not have"; the full-depth re-search that
/// follows is a whole extra subtree bought by putting that move late. Counted
/// by \[kind\]\[move index\], with the nodes each phase actually spent.
pub static LMR_TRY: [[AtomicU64; IDXS]; KINDS] = zeros!(KINDS, IDXS);
pub static LMR_RESEARCH: [[AtomicU64; IDXS]; KINDS] = zeros!(KINDS, IDXS);
pub static LMR_SCOUT_NODES: [AtomicU64; KINDS] = zeros!(KINDS);
pub static LMR_RESEARCH_NODES: [AtomicU64; KINDS] = zeros!(KINDS);
/// The non-LMR zero-window scout (move 1 at a NonPV node, or shallow depth)
/// and the PV full-window re-search, for the same accounting.
pub static ZW_NODES: [AtomicU64; KINDS] = zeros!(KINDS);
pub static PVRE_NODES: [AtomicU64; KINDS] = zeros!(KINDS);
/// Singular-extension verification searches and the nodes they cost.
pub static SING_TRY: AtomicU64 = AtomicU64::new(0);
pub static SING_NODES: AtomicU64 = AtomicU64::new(0);

/// One LMR scout at move index `idx`; `researched` if it beat alpha and a
/// full-depth re-search followed.
#[inline]
pub fn lmr(kind: usize, idx: i32, researched: bool, scout_nodes: u64, research_nodes: u64) {
    let i = ibucket(idx);
    LMR_TRY[kind][i].fetch_add(1, Relaxed);
    LMR_SCOUT_NODES[kind].fetch_add(scout_nodes, Relaxed);
    if researched {
        LMR_RESEARCH[kind][i].fetch_add(1, Relaxed);
        LMR_RESEARCH_NODES[kind].fetch_add(research_nodes, Relaxed);
    }
}

#[inline]
fn dbucket(depth: i32) -> usize {
    (depth.max(0) as usize).min(DEPTHS - 1)
}

#[inline]
fn ibucket(idx: i32) -> usize {
    (idx.max(0) as usize).min(IDXS - 1)
}

/// A beta cutoff: `kind` node, `depth` plies left, on the `move_count`-th
/// searched move, emitted by `MovePicker` stage `src`.
#[inline]
pub fn cutoff(kind: usize, depth: i32, move_count: i32, src: u8) {
    CUT_IDX[kind][dbucket(depth)][ibucket(move_count)].fetch_add(1, Relaxed);
    CUT_SRC[kind][(src as usize).min(SRCS - 1)].fetch_add(1, Relaxed);
}

/// A node whose move loop ended without a cutoff.
#[inline]
pub fn allnode(kind: usize, depth: i32, moves_searched: i32, best_idx: i32) {
    let d = dbucket(depth);
    ALL_N[kind][d].fetch_add(1, Relaxed);
    ALL_MOVES[kind][d].fetch_add(moves_searched.max(0) as u64, Relaxed);
    ALL_BEST_IDX[kind][ibucket(best_idx)].fetch_add(1, Relaxed);
}

/// Node entry, and whether the picker got a usable TT move.
#[inline]
pub fn node_entry(kind: usize, depth: i32, has_tt_move: bool) {
    let t = if has_tt_move { &TT_YES } else { &TT_NO };
    t[kind][dbucket(depth)].fetch_add(1, Relaxed);
}

fn row(h: &[AtomicU64]) -> Vec<u64> {
    h.iter().map(|c| c.load(Relaxed)).collect()
}

fn csv(v: &[u64]) -> String {
    // Trim the all-zero tail so a 64-wide row does not dominate the log.
    let last = v.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    v[..last]
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Print the census to stderr as machine-readable `C70 ...` lines. Called
/// once at the end of `bench`.
pub fn report(nodes: u64) {
    eprintln!("C70 nodes={nodes}");
    for k in 0..KINDS {
        let kname = KIND_NAMES[k];
        // Per-depth rows.
        let mut tot_fh = 0u64;
        let mut tot_fh1 = 0u64;
        let mut tot_moves = 0u64;
        let mut tot_all = 0u64;
        let mut tot_allmoves = 0u64;
        for d in 0..DEPTHS {
            let h = row(&CUT_IDX[k][d]);
            let fh: u64 = h.iter().sum();
            let alln = ALL_N[k][d].load(Relaxed);
            if fh == 0 && alln == 0 {
                continue;
            }
            let fh1 = h[1];
            let moves: u64 = h.iter().enumerate().map(|(i, &c)| i as u64 * c).sum();
            let allmoves = ALL_MOVES[k][d].load(Relaxed);
            tot_fh += fh;
            tot_fh1 += fh1;
            tot_moves += moves;
            tot_all += alln;
            tot_allmoves += allmoves;
            eprintln!(
                "C70 d kind={kname} d={d} fh={fh} fh1={fh1} fhf={:.4} mean_idx={:.3} \
                 excess={:.3} all_n={alln} all_moves={allmoves} all_mean={:.3} \
                 tt_yes={} tt_no={} hist={}",
                fh1 as f64 / fh.max(1) as f64,
                moves as f64 / fh.max(1) as f64,
                // "excess" = wasted child searches per cutoff: every move
                // before the cutting one was searched for nothing.
                (moves.saturating_sub(fh)) as f64 / fh.max(1) as f64,
                allmoves as f64 / alln.max(1) as f64,
                TT_YES[k][d].load(Relaxed),
                TT_NO[k][d].load(Relaxed),
                csv(&h),
            );
        }
        if tot_fh == 0 && tot_all == 0 {
            continue;
        }
        eprintln!(
            "C70 tot kind={kname} fh={tot_fh} fh1={tot_fh1} fhf={:.4} mean_idx={:.3} \
             excess_moves={} all_n={tot_all} all_moves={tot_allmoves} cut_share={:.4}",
            tot_fh1 as f64 / tot_fh.max(1) as f64,
            tot_moves as f64 / tot_fh.max(1) as f64,
            tot_moves.saturating_sub(tot_fh),
            tot_fh as f64 / (tot_fh + tot_all).max(1) as f64,
        );
        let s = row(&CUT_SRC[k]);
        let st: u64 = s.iter().sum();
        if st > 0 {
            let named: Vec<String> = s
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c != 0)
                .map(|(i, &c)| format!("{}={c}({:.2}%)", SRC_NAMES[i], 100.0 * c as f64 / st as f64))
                .collect();
            eprintln!("C70 src kind={kname} {}", named.join(" "));
        }
        let b = row(&ALL_BEST_IDX[k]);
        if b.iter().sum::<u64>() > 0 {
            eprintln!("C70 bestidx kind={kname} hist={}", csv(&b));
        }
        let lt = row(&LMR_TRY[k]);
        let lr = row(&LMR_RESEARCH[k]);
        let ltn: u64 = lt.iter().sum();
        if ltn > 0 {
            let lrn: u64 = lr.iter().sum();
            eprintln!(
                "C70 lmr kind={kname} try={ltn} research={lrn} rate={:.4}                  scout_nodes={} research_nodes={} research_share={:.4}",
                lrn as f64 / ltn as f64,
                LMR_SCOUT_NODES[k].load(Relaxed),
                LMR_RESEARCH_NODES[k].load(Relaxed),
                LMR_RESEARCH_NODES[k].load(Relaxed) as f64 / nodes.max(1) as f64,
            );
            eprintln!("C70 lmrtry kind={kname} hist={}", csv(&lt));
            eprintln!("C70 lmrres kind={kname} hist={}", csv(&lr));
        }
        let zw = ZW_NODES[k].load(Relaxed);
        let pv = PVRE_NODES[k].load(Relaxed);
        if zw > 0 || pv > 0 {
            eprintln!("C70 phase kind={kname} zw_nodes={zw} pv_research_nodes={pv}");
        }
    }
    let st = SING_TRY.load(Relaxed);
    if st > 0 {
        eprintln!(
            "C70 singular try={st} nodes={} share={:.4}",
            SING_NODES.load(Relaxed),
            SING_NODES.load(Relaxed) as f64 / nodes.max(1) as f64
        );
    }
}
