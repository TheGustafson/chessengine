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

use super::sota_features::RowOffset;
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
/// `refresh_threat_into_at` — the FullThreats FROM-SCRATCH rebuild, both the
/// root refresh and `discharge_refresh`'s mid-search one.
///
/// Perf study 39 added this because studies 29 (§8.1) and 31 both priced
/// accumulator fusion's COST side — a fused accumulator inherits HalfKA's
/// refresh trigger, so every HalfKA refresh drags a threat rebuild with it —
/// from an assumed row count neither had measured. `rows` here is the number
/// of weight COLUMNS folded, i.e. the number of active threat features for one
/// perspective, which is the quantity that decides whether fusion is a loss.
pub static THR_REFRESH: Site = Site::new();
/// Perf study 51 — weight ROWS that are entirely zero, counted where it
/// matters: as a fraction of the rows actually FOLDED, not of the table.
///
/// A row of zeros contributes nothing to the accumulator, so the fold's 40
/// loads and 40 `vpaddw` for it are pure waste, and skipping it is
/// bit-identical (adding zero is the group identity). The static table share
/// and the folded share are different questions — zeros could sit entirely in
/// rare features — so `ZERO_ROW_*` counts the folded population and
/// `FOLDROWSTATS` reports the static one.
pub static ZERO_ROW_HKA: Site = Site::new();
/// The same for the FullThreats channel, which is 54.7% of all folded rows.
pub static ZERO_ROW_THR: Site = Site::new();

/// Count the all-zero rows among `idx`, charged to `site`. `rows` counts every
/// row inspected so `rows − calls` is not needed: `calls` is the zero count.
///
/// `iter().all()` short-circuits on the first nonzero byte, so a dense row
/// costs one comparison and only a genuinely zero row pays the full `acc`
/// scan. Diagnostic build only.
#[inline]
pub fn note_zero_rows(site: &Site, weights: &[i8], idx: &[RowOffset], acc: usize) {
    for &r in idx {
        let base = r.0;
        if let Some(row) = weights.get(base..base + acc) {
            if row.iter().all(|&w| w == 0) {
                site.calls.fetch_add(1, Relaxed);
            }
        }
        site.rows.fetch_add(1, Relaxed);
    }
}

/// Per-channel fold counts and per-row peak magnitude, so the question "how
/// many BITS does a folded weight row actually need?" is answered over the
/// FOLDED population rather than over the table.
///
/// The distinction is not academic: `FOLDROWSTATS` says 16.4% of threat rows
/// are entirely zero, and `FOLDZEROROW` says 30 of 19.6 M FOLDED rows are —
/// the zeros sit in features that never occur. A static histogram of the table
/// answers a question nobody asked.
pub const ROWCH: usize = 2;
pub const ROWCH_NAMES: [&str; ROWCH] = ["hka", "thr"];
static ROWCOUNT: [std::sync::OnceLock<Vec<AtomicU64>>; ROWCH] =
    [const { std::sync::OnceLock::new() }; ROWCH];
static ROWMAXABS: [std::sync::OnceLock<Vec<u8>>; ROWCH] =
    [const { std::sync::OnceLock::new() }; ROWCH];

/// Charge each index in `idx` to its row's fold counter.
#[inline]
pub fn note_row_fold(ch: usize, idx: &[RowOffset]) {
    if let Some(c) = ROWCOUNT[ch].get() {
        for &r in idx {
            if let Some(cell) = c.get(r.feature()) {
                cell.fetch_add(1, Relaxed);
            }
        }
    }
}

/// Perf study 61 — the **reuse-distance census** of the fold's weight rows.
///
/// Studies 29/43/44/51/60 closed nine routes to the −13.19 % weight-row prize,
/// every one of them by changing *when* a row arrives (eight prefetch arms) or
/// *how many bytes* it occupies (zeros, nibbles, huge pages, congruence). None
/// of them ever asked the prior question: **how far apart in time are two
/// touches of the SAME row?** That number decides whether the misses are
/// capacity (nothing but fewer bytes can help) or conflict/ordering (layout
/// can).
///
/// The statistic is the **temporal reuse distance**: for each folded row, the
/// number of row-touches since that same row was last folded, in this channel.
/// Bucketed by `floor(log2 d)`. Reading it: a fully-associative cache holding
/// the last `W` DISTINCT rows gets at least `CDF(W)` hits, because the number
/// of distinct rows in a window of `W` touches is at most `W`. So the CDF at
/// `W = 819` (= 1 MB of L2 / 1280 B per row) is a **lower bound on the
/// achievable L2 hit rate**, and the CDF at `W = 25` (32 KB of L1) the same
/// for L1.
///
/// One `Vec<u32>` load, compare and store per folded row — instrument-only,
/// and it does not touch the weight bytes, so it cannot perturb the very
/// locality it is measuring beyond its own (small, contiguous, per-channel)
/// table.
const RD_BUCKETS: usize = 28;
static ROWLAST: [std::sync::OnceLock<Vec<std::sync::atomic::AtomicU32>>; ROWCH] =
    [const { std::sync::OnceLock::new() }; ROWCH];
static ROWCLOCK: [AtomicU64; ROWCH] = [const { AtomicU64::new(0) }; ROWCH];
static ROWDIST: [[AtomicU64; RD_BUCKETS]; ROWCH] =
    [const { [const { AtomicU64::new(0) }; RD_BUCKETS] }; ROWCH];
/// Touches whose row had never been folded before — the compulsory misses.
static ROWFIRST: [AtomicU64; ROWCH] = [const { AtomicU64::new(0) }; ROWCH];

/// Charge each index in `idx` to the reuse-distance histogram of channel `ch`.
#[inline]
pub fn note_row_reuse(ch: usize, idx: &[RowOffset]) {
    let Some(last) = ROWLAST[ch].get() else {
        return;
    };
    for &r in idx {
        let now = ROWCLOCK[ch].fetch_add(1, Relaxed) + 1;
        let Some(cell) = last.get(r.feature()) else {
            continue;
        };
        let prev = cell.swap(now as u32, Relaxed);
        if prev == 0 {
            ROWFIRST[ch].fetch_add(1, Relaxed);
            continue;
        }
        let d = (now as u32).saturating_sub(prev).max(1);
        // `ilog2` of a `u32` is 0..=31; the top bucket absorbs everything
        // beyond 2^27 touches, which at 7.2 rows/node is ~18.6 M nodes — more
        // than the whole frame, so nothing real lands there.
        let b = (d.ilog2() as usize).min(RD_BUCKETS - 1);
        ROWDIST[ch][b].fetch_add(1, Relaxed);
    }
}

/// Build channel `ch`'s per-row peak-magnitude table and size its counters.
pub fn rowwidth_init(ch: usize, weights: &[i8], acc: usize) {
    let peaks: Vec<u8> = weights
        .chunks_exact(acc)
        .map(|row| {
            row.iter()
                .map(|&w| i16::from(w).unsigned_abs().min(255) as u8)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let n = peaks.len();
    let _ = ROWMAXABS[ch].set(peaks);
    let _ = ROWCOUNT[ch].set((0..n).map(|_| AtomicU64::new(0)).collect());
    // Perf study 61's reuse clock. 0 means "never folded", so the clock starts
    // at 1 and a first touch is distinguishable from a distance-0 one.
    let _ = ROWLAST[ch].set(
        (0..n)
            .map(|_| std::sync::atomic::AtomicU32::new(0))
            .collect(),
    );
}

/// Static row structure of a weight table: how many rows are entirely zero,
/// and the distribution of nonzero lanes per row in 16 buckets of `acc/16`.
pub fn rowstats(name: &'static str, weights: &[i8], acc: usize) {
    let (mut zero, mut total) = (0u64, 0u64);
    let mut hist = [0u64; 17];
    for row in weights.chunks_exact(acc) {
        total += 1;
        let nz = row.iter().filter(|&&w| w != 0).count();
        if nz == 0 {
            zero += 1;
        }
        hist[(nz * 16 / acc).min(16)] += 1;
    }
    if let Ok(mut g) = ROWSTATS.lock() {
        g.push((name, total, zero, hist));
    }
}

#[allow(clippy::type_complexity)]
static ROWSTATS: std::sync::Mutex<Vec<(&'static str, u64, u64, [u64; 17])>> =
    std::sync::Mutex::new(Vec::new());

/// `activate` — the head's accumulator merge. Reads FOUR `[i16; ACC]` arrays
/// (two perspectives x two channels) and writes one `[u8; ACC]`. This is the
/// 10.24 KB/ply Stockfish carries as 4.10 KB.
pub static ACTIVATE: Site = Site::new();
/// `SotaHead::forward_into` — one full evaluation.
pub static EVAL: Site = Site::new();

/// Perf study 61 — the **group density of the sparse L1 matmul**.
///
/// `l1_dot` is a gather: one 64-byte `vpdpbusd` load per NONZERO four-lane
/// group of the activated FT, indexed through `find_nnz`'s output. It is 6.9 %
/// of the engine's IBS tag-to-retire latency and study 51 measures its table
/// (`l1_weights`, 20 KB per output bucket) at a **91.44 % L1 miss rate**.
///
/// Sparsity only pays if the groups it skips are worth the irregularity it
/// buys. A four-lane group is zero only if all FOUR of its clipped-ReLU lanes
/// are zero, so the GROUP density is far above the element density — and if it
/// is high enough, the dense form reads the same 20 KB *sequentially*, which is
/// a different memory access pattern for a small amount of extra arithmetic.
/// `rows` holds the summed nonzero-group count and `calls` the evaluations, so
/// `rows/calls / (ACC/4)` is the density.
pub static NNZ: Site = Site::new();

/// Shape histogram for the HalfKA fold: index `adds * 4 + subs`, capped at 15.
pub static HKA_SHAPE: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Perf study 43 — row-count histogram for the Finny **warm** refresh
/// (`apply_halfka_deltas_dual`), `min(adds + subs, 63)`.
///
/// `hka_dual`'s mean of ~5.68 rows/call is the only shape statistic anyone had
/// for this site, and a mean cannot answer either of the two questions study 43
/// needed: how often the set-difference is EMPTY (in which case the fold's
/// 2560-byte write-back to the cache entry is pure waste and a copy would do),
/// and how heavy the tail is (which is what an associativity change would
/// target). Both are read straight off this histogram.
pub static HKA_DUAL_SHAPE: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

/// Perf study 43 — refreshes per Finny key, indexed `perspective * 64 + key`.
///
/// `FINNY_KEYS == 64` because `halfka_king_key` is a BIJECTION from the 64 king
/// squares onto the 64 keys (perf study 22 §5.3), so the table cannot be made
/// larger and every key is reachable. What nobody had measured is how many of
/// them a search actually *visits*: the resident working set is
/// `distinct keys x 2688 B`, not the 336 KB the array occupies, and that is the
/// number that decides whether the table's SIZE is a locality problem at all.
pub static FINNY_KEY: [AtomicU64; 128] = [const { AtomicU64::new(0) }; 128];

/// Hook: a warm Finny refresh folded `rows` weight rows.
#[inline]
pub fn note_hka_dual_shape(rows: usize) {
    HKA_DUAL_SHAPE[rows.min(63)].fetch_add(1, Relaxed);
}

/// Hook: perspective `p` refreshed Finny key `key`.
#[inline]
pub fn note_finny_key(p: usize, key: usize) {
    FINNY_KEY[(p * 64 + key).min(127)].fetch_add(1, Relaxed);
}
/// Shape histogram for the threat fold: `min(adds + subs, 31)`.
pub static THR_SHAPE: [AtomicU64; 32] = [const { AtomicU64::new(0) }; 32];

/// Perf study 61 — the threat fold's `(adds, subs)` PAIR, capped at 7 each.
///
/// `THR_SHAPE` records only the SUM, and the sum cannot answer the question the
/// kernel dispatch actually asks. `i8_deltas_into_inner` matches on the pair
/// and has exactly four arms — `(1,1) (1,2) (2,1) (2,2)` — chosen in study 25
/// when this kernel was HalfKA-only and the threat channel had its own; study
/// 32 merged the two and routed every threat fold through the same dispatch
/// without revisiting the arm list. So the share of threat folds that reach the
/// branch-free straight-line path, and the share that fall through to the
/// runtime-length generic, have never been counted.
pub static THR_PAIR: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

/// Hook: a threat fold had `adds` add-rows and `subs` sub-rows.
#[inline]
pub fn note_thr_pair(adds: usize, subs: usize) {
    THR_PAIR[adds.min(7) * 8 + subs.min(7)].fetch_add(1, Relaxed);
    THR_LENS[adds.min(31)].fetch_add(1, Relaxed);
    THR_LENS[32 + subs.min(31)].fetch_add(1, Relaxed);
}

/// Perf study 62 — the threat fold's per-list lengths, **uncapped at 7**.
///
/// [`THR_PAIR`] saturates each side at 7, which is fine for asking "does the
/// dispatch table cover this fold" up to a 6×6 grid but useless for asking how
/// far the tail runs — and the tail is what decides whether a square grid can
/// ever close, or whether the generic kernel has to be fixed instead. Index
/// `0..32` is `adds.len()`, `32..64` is `subs.len()`, each capped at 31
/// (`THR_SHAPE` shows the SUM never exceeds 27, so 31 is not a real cap).
pub static THR_LENS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

/// Perf study 62 — the Finny **warm refresh**'s `(adds, subs)` pair, capped at
/// 7 each, exactly as [`THR_PAIR`] does for the threat fold.
///
/// `i8_deltas_dual_inner` is the one hot fold kernel with **no shape dispatch
/// at all** (study 61 §10 item 5), and `HKA_DUAL_SHAPE` records only the SUM —
/// which cannot answer the question a dispatch asks, for the same reason
/// `THR_SHAPE` could not.
pub static DUAL_PAIR: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

/// Hook: a warm Finny refresh had `adds` add-rows and `subs` sub-rows.
#[inline]
pub fn note_dual_pair(adds: usize, subs: usize) {
    DUAL_PAIR[adds.min(7) * 8 + subs.min(7)].fetch_add(1, Relaxed);
}

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

/// **Perf study 48 — the redundancy census of the resolved threat index
/// lists.** A feature index that appears in BOTH the add list and the sub list
/// for the same perspective contributes `+row` and `−row` to the same
/// accumulator lanes. Under wrapping `i16` arithmetic that is EXACTLY zero, so
/// the pair is pure weight-table traffic (2 × 1280 B of the 12.6 MB threat
/// table) that provably changes no lane. A repeated index *within* one list is
/// a second read of a row already read — `x + w + w ≡ x + 2w (mod 2¹⁶)`, also
/// exact.
///
/// This is a COUNT, not a fix: the study measures the population before
/// deciding whether the de-duplication is worth its own scan.
pub static THR_CANCEL: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Row names for [`THR_CANCEL`], in index order.
pub const THR_CANCEL_NAMES: [&str; 6] = [
    "folds",
    "adds",
    "subs",
    "cancel_pairs",
    "dup_add",
    "dup_sub",
];

/// Tally one folded ply's resolved index lists into [`THR_CANCEL`].
///
/// Quadratic in the list length on purpose: this is instrument-only code, the
/// lists are at most `DELTA_CAP` long and in practice ~5, and a hash set here
/// would allocate inside the hot fold.
#[inline]
pub fn note_thr_cancel(adds: [&[RowOffset]; 2], subs: [&[RowOffset]; 2]) {
    THR_CANCEL[0].fetch_add(1, Relaxed);
    for p in 0..2 {
        let (a, s) = (adds[p], subs[p]);
        THR_CANCEL[1].fetch_add(a.len() as u64, Relaxed);
        THR_CANCEL[2].fetch_add(s.len() as u64, Relaxed);
        // Greedy multiset intersection: each sub may cancel at most one add.
        let mut taken = [false; 64];
        let mut cancel = 0u64;
        for &x in a {
            for (j, &y) in s.iter().enumerate().take(64) {
                if !taken[j] && x == y {
                    taken[j] = true;
                    cancel += 1;
                    break;
                }
            }
        }
        THR_CANCEL[3].fetch_add(cancel, Relaxed);
        let dups =
            |v: &[RowOffset]| (1..v.len()).filter(|&i| v[..i].contains(&v[i])).count() as u64;
        THR_CANCEL[4].fetch_add(dups(a), Relaxed);
        THR_CANCEL[5].fetch_add(dups(s), Relaxed);
    }
}

/// King-boundary crossing census (richer-representation study, Tier-0 T0c).
///
/// A threat-context refinement bit b(ksq) forces a from-scratch threat rebuild
/// whenever the relevant king crosses b's boundary — the same trigger shape as
/// the existing d/e mirror. Its runtime price is therefore
/// `crossing_rate x at_full`, and the crossing rate must be measured IN-TREE
/// (search walks kings differently from games). These counters tally, for
/// every made move that relocates a king, which candidate boundaries it
/// crossed; `kmove.mirror` doubles as the calibration column against the
/// measured 0.0214 rebuilds/node (its ratio to `thr_refresh.calls` on the same
/// run gives the deferral survival factor, which the deferral machinery
/// applies identically to any boundary).
///
/// Indexed by the moving king's color; ranks are color-relative (rank 0 = that
/// king's home rank), matching the per-perspective own-frame definitions in
/// `tools/nnue/region_census.py`.
pub static KMOVE: [[AtomicU64; 2]; 6] = [const { [const { AtomicU64::new(0) }; 2] }; 6];
/// Row names for [`KMOVE`], in index order.
pub const KMOVE_NAMES: [&str; 6] = ["total", "mirror", "rank12", "rank01", "fpair", "rankhalf"];

/// Hook: color `c`'s king moved `from` → `to` (absolute squares, 0..64).
#[inline]
pub fn note_king_move(c: usize, from: u8, to: u8) {
    let (ff, tf) = (from & 7, to & 7);
    // Color-relative ranks: home rank = 0 for both colors.
    let flip = (c as u8) * 7;
    let (fr, tr) = ((from >> 3) ^ flip, (to >> 3) ^ flip);
    let hits: [bool; 6] = [
        true,
        (ff <= 3) != (tf <= 3), // d/e mirror (the existing context)
        (fr <= 1) != (tr <= 1), // back-two-ranks vs advanced
        (fr == 0) != (tr == 0), // back rank vs off it
        (ff >> 1) != (tf >> 1), // file pair a-b/c-d/e-f/g-h (finer than mirror)
        (fr <= 3) != (tr <= 3), // own half vs enemy half of the board
    ];
    for (i, hit) in hits.iter().enumerate() {
        if *hit {
            KMOVE[i][c].fetch_add(1, Relaxed);
        }
    }
}

/// Learned-representation concept study, Tier-0: the MAKE-time locality
/// census. Every counter above tallies at FOLD or EVAL time — i.e. after the
/// deferral machinery has already discarded the plies that never needed an
/// accumulator — so none of them can answer "what does one made move change?"
/// unconditionally. These can:
///
/// * [`MAKE`] — one hit per made move recorded into the SOTA stacks
///   (`SotaContext::push`; null moves and the TT-verification probe bypass it,
///   matching `Worker::nodes`). `rows` accumulates the HalfKA delta size
///   (adds + subs, ≤ 4).
/// * [`THR_MAKE`] — same denominator; `rows` accumulates the RAW threat event
///   count from an instrument-only eager [`build_deltas`] replay into the
///   context's scratch buffer at `apply_at` time. The production path stays
///   fully lazy; `compute` clears the buffer before its own use, so the replay
///   leaves no state behind. Comparing `THR_MAKE.rows/calls` against
///   `thr_into.rows/thr_into.calls` measures exactly what deferral saves.
pub static MAKE: Site = Site::new();
pub static THR_MAKE: Site = Site::new();
/// Make-time HalfKA delta shape, `adds * 4 + subs` (same layout as
/// [`HKA_SHAPE`], different conditioning: made moves, not surviving folds).
pub static HKA_MAKE_SHAPE: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
/// Make-time threat event count histogram, capped at 95 (observer max ≤ 80).
pub static THR_MAKE_SHAPE: [AtomicU64; 96] = [const { AtomicU64::new(0) }; 96];

#[inline]
pub fn note_make(adds: usize, subs: usize) {
    MAKE.hit(adds + subs);
    HKA_MAKE_SHAPE[(adds.min(3) * 4 + subs.min(3)).min(15)].fetch_add(1, Relaxed);
}

#[inline]
pub fn note_thr_make(events: usize) {
    THR_MAKE.hit(events);
    THR_MAKE_SHAPE[events.min(95)].fetch_add(1, Relaxed);
}

/// Active-rows census over full evaluations (the positions the engine actually
/// prices): [`ACTIVE_HKA`] histograms the piece count (= active HalfKA rows
/// per perspective) per full eval; [`ACTIVE_THR`] hits once per (eval,
/// perspective) with the enumerated active threat row count, histogrammed in
/// [`ACTIVE_THR_SHAPE`] (cap 127; observed per-perspective max ≤ 128).
pub static ACTIVE_HKA: [AtomicU64; 33] = [const { AtomicU64::new(0) }; 33];
pub static ACTIVE_THR: Site = Site::new();
pub static ACTIVE_THR_SHAPE: [AtomicU64; 128] = [const { AtomicU64::new(0) }; 128];

#[inline]
pub fn note_active(pieces: u32, thr: [usize; 2]) {
    ACTIVE_HKA[(pieces as usize).min(32)].fetch_add(1, Relaxed);
    for n in thr {
        ACTIVE_THR.hit(n);
        ACTIVE_THR_SHAPE[n.min(127)].fetch_add(1, Relaxed);
    }
}

/// The in-tree king-square TRANSITION MATRIX, color-relative (black squares
/// rank-mirrored via `sq ^ 56`, matching the per-perspective own-frame the
/// region tools use), both colors summed. `KTRANS[from][to]` counts made moves
/// that relocated a king `from → to`.
///
/// This is the asset that prices ANY candidate king-context partition without
/// another engine run: for a partition `g`, its in-tree flip rate is
/// `Σ_{g(from) ≠ g(to)} KTRANS[from][to] / nodes` — the quantity the
/// richer-representation study could only measure for six hand-picked bits
/// (`KMOVE`), now available for every partition at once, including learned
/// ones and the soft assignments a training-time locality penalty needs.
pub static KTRANS: [[AtomicU64; 64]; 64] = [const { [const { AtomicU64::new(0) }; 64] }; 64];

/// Hook: color `c`'s king moved `from` → `to` (absolute squares, 0..64).
#[inline]
pub fn note_king_trans(c: usize, from: u8, to: u8) {
    let m = (c as u8) * 56;
    KTRANS[(from ^ m) as usize][(to ^ m) as usize].fetch_add(1, Relaxed);
}

/// Named address regions, so an IBS data-address sample can be attributed to
/// the accumulator stack or to a weight table. Sixteen slots is plenty; the
/// registry is write-once at construction and read once at dump.
const REGIONS: usize = 24;
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
        ("thr_refresh", &THR_REFRESH),
        ("activate", &ACTIVATE),
        ("eval", &EVAL),
        ("nnz", &NNZ),
        ("make", &MAKE),
        ("thr_make", &THR_MAKE),
        ("active_thr", &ACTIVE_THR),
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
    for (r, cell) in HKA_DUAL_SHAPE.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT hka_dual_shape.{r}={n}");
        }
    }
    for (i, cell) in FINNY_KEY.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT finny_key.p{}k{}={n}", i / 64, i % 64);
        }
    }
    for (r, cell) in THR_SHAPE.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT thr_shape.{r}={n}");
        }
    }
    for (i, cell) in THR_PAIR.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT thr_pair.{}a{}s={n}", i / 8, i % 8);
        }
    }
    for (i, cell) in THR_LENS.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!(
                "FOLDCOUNT thr_lens.{}{}={n}",
                if i < 32 { "a" } else { "s" },
                i % 32
            );
        }
    }
    for (i, cell) in DUAL_PAIR.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT dual_pair.{}a{}s={n}", i / 8, i % 8);
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
    for a in 0..4 {
        for s in 0..4 {
            let n = HKA_MAKE_SHAPE[a * 4 + s].load(Relaxed);
            if n != 0 {
                emit!("FOLDCOUNT hka_make_shape.{a}a{s}s={n}");
            }
        }
    }
    for (r, cell) in THR_MAKE_SHAPE.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT thr_make_shape.{r}={n}");
        }
    }
    for (p, cell) in ACTIVE_HKA.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT active_hka.{p}={n}");
        }
    }
    for (r, cell) in ACTIVE_THR_SHAPE.iter().enumerate() {
        let n = cell.load(Relaxed);
        if n != 0 {
            emit!("FOLDCOUNT active_thr_shape.{r}={n}");
        }
    }
    for (f, row) in KTRANS.iter().enumerate() {
        for (t, cell) in row.iter().enumerate() {
            let n = cell.load(Relaxed);
            if n != 0 {
                emit!("FOLDCOUNT ktrans.{f}.{t}={n}");
            }
        }
    }
    for (i, name) in THR_CANCEL_NAMES.iter().enumerate() {
        emit!(
            "FOLDCOUNT thr_cancel.{name}={}",
            THR_CANCEL[i].load(Relaxed)
        );
    }
    for (i, name) in KMOVE_NAMES.iter().enumerate() {
        let (w, b) = (KMOVE[i][0].load(Relaxed), KMOVE[i][1].load(Relaxed));
        emit!(
            "FOLDCOUNT kmove.{name}={} kmove.{name}.w={w} kmove.{name}.b={b}",
            w + b
        );
    }
    if let Ok(g) = REGION_NAME.lock() {
        for &(i, name) in g.iter() {
            let (lo, hi) = (REGION_LO[i].load(Relaxed), REGION_HI[i].load(Relaxed));
            emit!("FOLDREGION {name} lo={lo:#x} hi={hi:#x} bytes={}", hi - lo);
        }
    }
    for (name, site) in [("hka", &ZERO_ROW_HKA), ("thr", &ZERO_ROW_THR)] {
        let (zero, seen) = site.read();
        emit!("FOLDZEROROW {name}.folded={seen} {name}.allzero={zero}");
    }
    // Fold-weighted signed bit width. A row whose peak magnitude is `m` needs
    // `ceil(log2(m+1)) + 1` signed bits; bucket 0 is an all-zero row.
    for ch in 0..ROWCH {
        let (Some(peaks), Some(cnt)) = (ROWMAXABS[ch].get(), ROWCOUNT[ch].get()) else {
            continue;
        };
        let mut by_bits = [0u64; 10];
        let mut rows_by_bits = [0u64; 10];
        for (r, &m) in peaks.iter().enumerate() {
            let bits = if m == 0 {
                0
            } else {
                (16 - u16::from(m).leading_zeros() + 1) as usize
            };
            let b = bits.min(9);
            by_bits[b] += cnt[r].load(Relaxed);
            rows_by_bits[b] += 1;
        }
        let folded: u64 = by_bits.iter().sum();
        let f: Vec<String> = by_bits
            .iter()
            .enumerate()
            .map(|(b, n)| format!("{b}:{n}"))
            .collect();
        let s: Vec<String> = rows_by_bits
            .iter()
            .enumerate()
            .map(|(b, n)| format!("{b}:{n}"))
            .collect();
        emit!(
            "FOLDROWBITS {} folded={folded} byfolds {} byrows {}",
            ROWCH_NAMES[ch],
            f.join(" "),
            s.join(" ")
        );
    }
    // Perf study 61 — the reuse-distance census, plus the distinct-row count
    // the whole frame actually touched (the working set's UPPER bound).
    for ch in 0..ROWCH {
        let Some(cnt) = ROWCOUNT[ch].get() else {
            continue;
        };
        let distinct = cnt.iter().filter(|c| c.load(Relaxed) > 0).count();
        let first = ROWFIRST[ch].load(Relaxed);
        let hist: Vec<String> = ROWDIST[ch]
            .iter()
            .enumerate()
            .map(|(b, n)| format!("{b}:{}", n.load(Relaxed)))
            .collect();
        emit!(
            "FOLDROWREUSE {} touches={} distinct={distinct} first={first} log2dist {}",
            ROWCH_NAMES[ch],
            ROWCLOCK[ch].load(Relaxed),
            hist.join(" ")
        );
    }
    if let Ok(g) = ROWSTATS.lock() {
        for (name, total, zero, hist) in g.iter() {
            let h: Vec<String> = hist
                .iter()
                .enumerate()
                .map(|(i, n)| format!("{i}:{n}"))
                .collect();
            emit!(
                "FOLDROWSTATS {name} rows={total} allzero={zero} nzhist16 {}",
                h.join(" ")
            );
        }
    }
    if let Ok(g) = VALHIST.lock() {
        for (name, hist) in g.iter() {
            let occupied: Vec<String> = hist
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n > 0)
                .map(|(v, n)| format!("{v}:{n}"))
                .collect();
            emit!(
                "FOLDVALHIST {name} distinct={} {}",
                occupied.len(),
                occupied.join(" ")
            );
        }
    }
}

/// A value histogram for a small integer weight array, so a quantisation
/// question ("does every `ft_scale` fit in a `u8`?") is answered by the engine
/// that loads the net rather than by re-deriving the loader's arithmetic in
/// Python. Values outside `0..VALHIST_N` land in the last bucket.
const VALHIST_N: usize = 1024;
static VALHIST: std::sync::Mutex<Vec<(&'static str, Vec<u64>)>> = std::sync::Mutex::new(Vec::new());

/// Record the value distribution of `vals` under `name`.
pub fn valhist(name: &'static str, vals: impl IntoIterator<Item = i64>) {
    let mut hist = vec![0u64; VALHIST_N + 1];
    for v in vals {
        let i = if (0..VALHIST_N as i64).contains(&v) {
            v as usize
        } else {
            VALHIST_N
        };
        hist[i] += 1;
    }
    if let Ok(mut g) = VALHIST.lock() {
        g.push((name, hist));
    }
}
