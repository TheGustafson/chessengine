//! Move ordering: the staged [`MovePicker`] and its scoring machinery
//! (design/09 — move ordering: MovePicker and history tables).
//!
//! The picker is a lazily staged, one-move-at-a-time generator driven by an
//! explicit state machine over a single 256-slot [`ExtMove`] buffer. Most
//! search nodes cut off on the TT move or the first good capture without ever
//! generating quiets, so the staging property matters above all: a TT-move
//! cutoff performs *zero* generation, scoring, and sorting.
//!
//! This module lands incrementally over design/09 §4:
//!
//! - **step 5:** [`ExtMove`] (the move + ordering score, §3.1) and
//!   [`partial_insertion_sort`] (the limited descending sort, §3.8).
//! - **step 6:** the [`MovePicker`] skeleton — the single owned
//!   `[ExtMove; MAX_MOVES]` buffer carved into regions by *indices* (§3.1),
//!   the branchless TT-stage skip (§3.5), the dedup [`select`](MovePicker)
//!   scan (§3.6) — and the two capture-only pipelines it drives:
//!   quiescence (`QsearchTt → QcaptureInit → Qcapture`) and ProbCut
//!   (`ProbcutTt → ProbcutInit → Probcut` with the `see_ge(threshold)` gate).
//! - **step 7 (here):** the main pipeline (`MainTt → CaptureInit →
//!   GoodCapture → QuietInit → GoodQuiet → BadCapture → BadQuiet`): quiet
//!   scoring (butterfly + pawn + continuation history, the check bonus, the
//!   threat escape/entry term via [`threats_by_lesser`], low-ply history),
//!   the score-dependent good/bad capture split with region swaps, the
//!   good/bad quiet split, and [`skip_quiet_moves`](MovePicker::skip_quiet_moves).
//! - **step 8 (here):** the evasion pipeline (`EvasionTt → EvasionInit →
//!   Evasion`) with its capture-first scoring.
//!
//! Hot-path rules (design/09 §6): no heap allocation on any [`next_move`]
//! path (generation uses a stack [`MoveList`], scoring writes into the owned
//! buffer), and the picker holds no borrows — `&Board` and
//! `&Histories` are passed into each call so the search can freely
//! make/unmake and update histories between calls.
//!
//! [`next_move`]: MovePicker::next_move

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::history::{Histories, LOW_PLY_SIZE};
use crate::movegen::{generate_captures, generate_evasions, generate_quiets};
use crate::moves::{MAX_MOVES, Move, MoveList};
use crate::search::params::{
    CAPTURE_MVV_FACTOR, EVASION_CAPTURE_BONUS, GOOD_CAPTURE_SEE_DIVISOR, GOOD_QUIET_THRESHOLD,
    QUIET_CHECK_BONUS, QUIET_CHECK_SEE_MARGIN, QUIET_LOW_PLY_WEIGHT, QUIET_MAIN_HISTORY_WEIGHT,
    QUIET_OFFENSE_FACTOR, QUIET_PAWN_HISTORY_WEIGHT, QUIET_SORT_DEPTH_FACTOR, QUIET_THREAT_FACTOR,
};
use crate::types::{DEPTH_QS, Depth, PieceType, VALUE_ZERO, Value};

/// A move paired with its ordering score (design/09 §3.1, D7).
///
/// Scoring happens only inside the picker's own buffer; [`MoveList`] stays two
/// bytes per move (doc 05). The representation contract is fixed in doc 02
/// §3.8 and asserted below: exactly 8 bytes, `#[repr(C)]`, align 4. Two
/// `ExtMove`s are ordered by `value` only — the move bits are payload, never a
/// tiebreak (the `partial_insertion_sort` below compares `value`).
///
/// **The layout is `value` first and the padding is a FIELD (perf study 27
/// arm 2).** Doc 02 §3.8 justified the 8-byte size with "8-byte elements move
/// as single u64 register loads/stores" — and for four months they did not.
/// The old order was `mv: Move` then an implicit 2-byte hole then `value`, and
/// a hole makes a struct copy *illegal* to widen: the padding bytes are `undef`
/// and LLVM may not read or write them, so `head[q] = head[q - 1]` in the
/// insertion sort compiled to a 2-byte move plus a 4-byte move, two loads and
/// two stores, per shift step. Naming the pad closes the hole; putting `value`
/// at offset 0 means the one 8-byte load that copies the element ALSO carries
/// the score in its low half, so the sort's compare is free of its own load.
///
/// The field order is not observable — nothing serialises an `ExtMove`, and
/// `#[repr(C)]` still pins the size against silent growth, which is what doc
/// 02 asked the attribute for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ExtMove {
    /// The ordering score; sorted and filtered on this alone. At offset 0 so
    /// it shares a load with the whole-element copy.
    pub value: i32,
    /// The move (`u16`).
    pub mv: Move,
    /// Explicit tail padding. Always zero — its only job is to make the type
    /// hole-free so a copy is one 8-byte move instead of two narrow ones.
    pad: u16,
}

impl ExtMove {
    /// The only constructor, so `pad` cannot be forgotten.
    #[inline]
    pub(crate) const fn new(mv: Move, value: i32) -> Self {
        ExtMove { value, mv, pad: 0 }
    }
}

// design/09 §8 criterion 2 / doc 02 §3.8: 8 bytes, align 4. `#[repr(C)]` lays
// out `value` (i32), `mv` (u16) and `pad` (u16) back to back — 8 bytes with NO
// hole, and the alignment is the i32's 4.
const _: () = assert!(size_of::<ExtMove>() == 8);
const _: () = assert!(align_of::<ExtMove>() == 4);

/// Descending, in-place, allocation-free partial sort with a `limit`
/// (design/09 §3.8).
///
/// After the pass the prefix `moves[0..=sorted_end]` is sorted descending and
/// holds the original element 0 **plus every element scoring `>= limit`**; the
/// tail is left unsorted and (bar that element-0 quirk) entirely below the
/// limit. Sorting work therefore scales with what the search actually
/// consumes: a cutoff usually arrives within the first few moves, so deep
/// nodes sort a large `limit`-selected prefix while shallow nodes barely sort
/// at all. A full `sort_unstable_by_key` of a 40-move quiet list at shallow
/// depth is pure waste — do not substitute one.
///
/// Ties keep their relative order (the shift condition is a strict `<`), so
/// the sort is stable among equal scores. Call sites: captures and evasions
/// with `limit = i32::MIN` (a full sort); quiets with a depth-scaled limit.
/// Width of `partial_insertion_sort`'s branchless window (perf study 34 §8).
///
/// Covers ~68% of insertions: the mean sorted prefix is 9.31 elements
/// (`HIST_PFX`), insertions run at `se = 1..P-1`, so the covered fraction is
/// `E[min(P-1, 8)] / E[P-1]` = 5.62/8.31. Widening it costs a store per lane
/// on every insertion, which is why it is not 16.
#[cfg(feature = "insertwin34")]
const INSERT_WINDOW: usize = 8;

pub(crate) fn partial_insertion_sort(moves: &mut [ExtMove], limit: i32) {
    // Perf study 40's DOSE AMPLIFIER (`--features dosesort40`, dose from
    // `P40_DOSE`). The arm this module is waiting on is a ~0.2%-of-cycles
    // trade, an order of magnitude under what `cycles:u` can resolve on this
    // box; the amplifier replays REAL earlier inputs through the same sort D
    // extra times per call so the same trade lands at (D+1)x its size and a
    // dose series can be regressed back to one call. See `dose40`.
    #[cfg(feature = "dosesort40")]
    crate::dose40::replay(moves, limit);
    partial_insertion_sort_inner(moves, limit)
}

/// The sort proper. Split out only so [`crate::dose40`] can re-enter it without
/// recursing through the amplifier; `#[inline(always)]` keeps the shipped
/// codegen byte-identical (verified: `.text` md5 unchanged, study 40 §1.3).
#[inline(always)]
pub(crate) fn partial_insertion_sort_inner(moves: &mut [ExtMove], limit: i32) {
    if moves.is_empty() {
        return;
    }
    // Element 0 anchors the prefix: `sorted_end` is the last sorted index.
    let mut sorted_end = 0;
    for p in 1..moves.len() {
        // One read of `moves[p]`, not two — the old form loaded `.value` for
        // the test and then loaded the whole `ExtMove` again for `tmp`.
        let tmp = moves[p];
        if tmp.value < limit {
            continue;
        }
        sorted_end += 1;
        // INVARIANT: `sorted_end <= p`. Before this iteration `sorted_end` is
        // the number of insertions among `1..p`, which is at most `p - 1`, so
        // after the increment it is at most `p`.
        debug_assert!(sorted_end <= p, "partial sort prefix outran its scan");
        // Perf study 20 arm 7 — spelled so LLVM can DISCHARGE the bounds
        // checks rather than emit them. The old form indexed the full slice
        // with `sorted_end` and `q`, neither of which the compiler can relate
        // to `moves.len()`, so the shipped binary carried two `cmp`/`jae`
        // pairs and two panic edges per INSERTED element (measured: the sort
        // is 36% of `init_quiets`, itself 9.3% of all engine instructions).
        //
        // `head` has length `p + 1` — a length LLVM derives from the loop
        // range — and every index touched below (`p`, `se`, `q <= se`) is
        // provably inside it. `.min(p)` is a no-op by the invariant just
        // asserted, and it is precisely the fact the compiler was missing: one
        // `cmov` in place of a check plus a panic block.
        let se = sorted_end.min(p);
        // Perf study 34 arm 2 — the BRANCHLESS fixed-width insertion, taken
        // when the sorted prefix still fits one window.
        //
        // The walk below is 1.28% of all engine cycles and 6.46% of ALL engine
        // branch mispredicts (study 34 §5), essentially all of it in its own
        // exit test: the displacement averages 2.8 with a wide spread, which is
        // the worst case a predictor can be handed. A fixed trip count has no
        // exit test to mispredict.
        //
        // It is the SAME permutation, not an approximation. Because
        // `moves[0..se]` is sorted descending, `{ j < se : value < tmp.value }`
        // is a contiguous SUFFIX, so a full-window pass with the shift
        // predicated on that comparison performs exactly the walk's writes:
        //   - `j` in `[q, se)`   -> shift, `moves[j+1] = moves[j]`
        //   - `j` in `[0, q)`    -> the compare is false, `moves[j+1]` rewritten
        //                           with its own value
        //   - `j` in `[se, W)`   -> masked off, likewise a self-write
        // and `q` falls out as `se` minus the number of shifts. Descending `j`
        // means every read is of a slot not yet written, which is what makes
        // the self-writes no-ops rather than smears.
        //
        // Guarded on `se <= W` (the window must cover the prefix) and
        // `moves.len() > W` (so `moves[W]` is in bounds — the self-writes may
        // reach past `p`, which is harmless in value but must be in range).
        // Everything else falls through to the walk unchanged.
        //
        // ⚠ **DECIDED, AND THE ANSWER IS NO. DO NOT RE-OPEN THIS BLIND.**
        //
        // Perf study 34 built it, verified it node-identical, and measured the
        // trade exactly — study 40 reproduced every figure on today's tree over
        // 60 paired within-round rounds with zero spread and a byte-identical
        // null at 0.000%: `instructions:u` **+0.719%/node** (+42.58),
        // `branches:u` **-0.525%** (-3.26), `ex_ret_brn_misp:u` **-4.409%**
        // (-1.014), 60/60 rounds on each. The mechanism is exactly as designed.
        //
        // What study 34 could not do was PRICE it, because `cycles:u` cannot
        // see a 0.3% effect on this box. Study 40 priced it on a counter that
        // can: **`ex_no_retire.empty`** — the cycles in which nothing retired
        // because the front end had nothing to deliver, which is where a
        // mispredict spends its cost. It is 192.91 cycles/node here (4.7% of
        // the engine), so the arm's effect is 4.7% OF IT rather than 0.22% of
        // cycles. 159 paired rounds, contrasts against the byte-identical null:
        //
        //     ex_ret_brn_misp     -4.517% +/- 0.102% (44.3 se) -1.088 misp/node
        //     ex_no_retire.empty  -5.150% +/- 0.357% (14.4 se)  -9.93 cyc/node
        //     cycles - SMT        +0.186% +/- 0.137%  (1.4 se)  +7.47 cyc/node
        //
        // **The arm buys 9.93 cycles of mispredict bubble and pays 17.40 cycles
        // of issue for it.** Study 34's model assumed ~18 cycles per mispredict
        // and ~4 IPC on the added instructions; measured on this arm in this
        // engine they are **9.13** and **2.45**, both worse for the arm, and
        // that is the whole gap between its predicted -0.17% and the measured
        // **+0.19%** (about -0.3 Elo). For the arm to break even the 42.58
        // added instructions would have to sustain 42.58/9.93 = 4.29 per cycle.
        //
        // The code stays because it is correct, node-identical, and the exact
        // negative result is worth more in the tree than in a doc. If you widen
        // the window or shorten the dependency chain, re-price it against
        // **9.13 cycles per mispredict** — that is the exchange rate on this
        // engine, and at it the whole of `init_quiets`' mispredict share (3.31
        // per node) is only 30.2 cycles/node, 0.74% of cycles.
        // Details: study 40 §3.
        #[cfg(feature = "insertwin34")]
        if se <= INSERT_WINDOW && moves.len() > INSERT_WINDOW {
            moves[p] = moves[se];
            // Length `INSERT_WINDOW + 1` is a constant LLVM can see, so every
            // index below is discharged without a panic edge (study 20 arm 7's
            // lesson, applied to a constant instead of `p`).
            let win = &mut moves[..=INSERT_WINDOW];
            let mut q = se;
            for j in (0..INSERT_WINDOW).rev() {
                let cur = win[j];
                let nxt = win[j + 1];
                let shift = j < se && cur.value < tmp.value;
                win[j + 1] = if shift { cur } else { nxt };
                q -= shift as usize;
            }
            win[q] = tmp;
            continue;
        }
        let head = &mut moves[..=p];
        // Hole-punch: the element leaving the prefix edge goes to `p`.
        head[p] = head[se];
        // PERF STUDY 23 measured this loop at 2.84% of ALL engine cycles and
        // 10.19% of all branch misses — the largest mispredict source outside
        // the NNUE and `search` itself. Two rewrites were tried and BOTH lost:
        // scan-then-`copy_within` cost +91 i/n (memmove is the wrong tool for a
        // ~3-element shift) and `partition_point` cost +77.8 i/n without moving
        // branch misses at all. The constraint that makes this hard is that the
        // TAIL permutation is observable — `BadQuiet` emits the unsorted
        // remainder in buffer order — so no algorithm that reorders it can pass
        // the node-identity gate. See docs/research/perf-study-23-*.md §6.
        let mut q = se;
        while q > 0 && head[q - 1].value < tmp.value {
            head[q] = head[q - 1];
            q -= 1;
        }
        head[q] = tmp;
    }
}

/// The per-piece-type facts `init_quiets`' scoring loop reads (perf study 23
/// arm 4). One 32-byte row per type, so a quiet's five `pt`-indexed reads share
/// a single scaled address and one cache line instead of four bases in four
/// different objects. Built once per `init_quiets`, six rows, 192 bytes.
#[derive(Clone, Copy)]
struct PtRow {
    /// Squares attacked by an enemy piece strictly cheaper than this type.
    threat: Bitboard,
    /// Reverse attack fan onto the undefended enemies this type may threaten.
    offense: Bitboard,
    /// Squares from which this type would give check.
    check: Bitboard,
    /// `pt.value() * QUIET_THREAT_FACTOR()`, premultiplied.
    threat_w: i32,
    /// `pt.value() * QUIET_OFFENSE_FACTOR()`, premultiplied.
    offense_w: i32,
}

/// The P-ORD quiet prior's per-move bonus, kept OUT of `init_quiets`' scoring
/// loop (perf study 23 arm 3).
///
/// `QuietPrior::bonus` is `#[inline]` and its body is a selected-row AVX-512
/// dot product over `L2` floats plus a horizontal reduction: ~55 instructions
/// of vector code that LLVM was splicing straight into the quiet-scoring loop.
/// The prior is OFF in every shipped configuration (`CPL_ORD_W` defaults to 0,
/// so `quiet_prior` is `None` and the branch is never taken), yet the loop paid
/// for it three ways — 200-odd bytes of the loop's icache footprint, the
/// register pressure of keeping the dot's live values schedulable (the loop
/// carried visible spill/reload traffic around the block), and two
/// `vzeroupper`s forced by the dirty AVX upper state.
///
/// `#[cold]` + `#[inline(never)]` says exactly that: the call is real when the
/// experiment is switched on, and free when it is not.
#[cfg(feature = "sota")]
#[cold]
#[inline(never)]
fn quiet_prior_bonus(qp: &crate::eval::nnue::policy_head::QuietPrior, m: Move) -> i32 {
    qp.bonus(m)
}

/// The "no victim" capture-history slot (design/09 §3.4): en passant and quiet
/// queen promotions score a victim of 0 and share this slot, mirroring
/// Stockfish's `NO_PIECE_TYPE` behaviour. Equals `VICTIM_NB - 1` in
/// `history.rs`.
const NO_VICTIM: usize = 6;

/// The capture ordering score (design/09 §3.7): pure MVV plus capture history,
/// `CAPTURE_MVV_FACTOR * victim_value + capture[pc][to][victim_slot]`.
///
/// Deliberately no attacker (LVA) term — the good/bad split's score-dependent
/// `see_ge` gate (doc 09 §3.6, step 7/8) accounts for attacker risk. En
/// passant and quiet queen promotions have no piece on `to`, so they score a
/// victim of 0 and use the [`NO_VICTIM`] slot.
fn score_capture(board: &Board, hist: &Histories, m: Move) -> i32 {
    let pc = board
        .piece_on(m.from())
        .expect("a capture move starts on a piece of the side to move");
    let victim = board.piece_on(m.to());
    let victim_value = victim.map_or(0, |p| p.piece_type.value());
    let victim_slot = victim.map_or(NO_VICTIM, |p| p.piece_type as usize);
    let cap_hist = hist.capture[pc.index()][m.to().index() as usize][victim_slot].get();
    CAPTURE_MVV_FACTOR() * victim_value + cap_hist
}

/// The picker's explicit state machine (design/09 §3.5). Each pipeline is a
/// contiguous run of variants, so the branchless TT skip is a single `+1` on
/// the discriminant. Steps 05/06 implement the two capture-only pipelines
/// (quiescence, ProbCut) plus the shared TT stages; the main and evasion
/// pipelines land in doc 09 §4 steps 7-8.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Debug)]
#[repr(u8)]
enum Stage {
    // main search
    MainTt,
    CaptureInit,
    GoodCapture,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,
    // evasions
    EvasionTt,
    EvasionInit,
    Evasion,
    // probcut
    ProbcutTt,
    ProbcutInit,
    Probcut,
    // quiescence
    QsearchTt,
    QcaptureInit,
    Qcapture,
}

impl Stage {
    /// Decode a discriminant back to its stage. Total over the 16 valid
    /// values; any other is a construction bug.
    #[inline]
    fn from_u8(v: u8) -> Stage {
        match v {
            0 => Stage::MainTt,
            1 => Stage::CaptureInit,
            2 => Stage::GoodCapture,
            3 => Stage::QuietInit,
            4 => Stage::GoodQuiet,
            5 => Stage::BadCapture,
            6 => Stage::BadQuiet,
            7 => Stage::EvasionTt,
            8 => Stage::EvasionInit,
            9 => Stage::Evasion,
            10 => Stage::ProbcutTt,
            11 => Stage::ProbcutInit,
            12 => Stage::Probcut,
            13 => Stage::QsearchTt,
            14 => Stage::QcaptureInit,
            15 => Stage::Qcapture,
            _ => unreachable!("invalid movepick stage {v}"),
        }
    }

    /// The next stage in the pipeline (design/09 §3.6): the Rust replacement
    /// for Stockfish's `goto`/fallthrough between stages.
    #[inline]
    fn bump(self) -> Stage {
        Stage::from_u8(self as u8 + 1)
    }
}

/// A lazily staged, one-move-at-a-time move generator (design/09 §3.5).
///
/// Holds no borrows: `&Board` and `&Histories` are passed into every
/// [`next_move`](MovePicker::next_move) call, so the search can make/unmake
/// and update histories between calls while the picker sits as plain owned
/// data (~2.1 KiB) on the search stack. Construct it in place per node and
/// never move or copy it (that would be a 2 KiB memcpy — doc 09 §6).
///
/// The single `[ExtMove; MAX_MOVES]` buffer is carved into regions by the
/// index cursors below rather than pointers (borrow-checker-friendly, equally
/// fast — doc 09 §3.1): `[0, end_bad_captures)` bad captures, `[end_bad_captures,
/// end_captures)` captures, `[end_captures, end_generated)` quiets, with
/// `[cur, end_cur)` the live emission window over one of them.
pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    depth: Depth,
    ply: usize,
    /// ProbCut only: the `see_ge` threshold captures must clear.
    threshold: Value,
    skip_quiets: bool,
    /// `stack[cur-1 ..= cur-6].cont_hist`, copied in by the search.
    cont_hist: [crate::history::ContHistId; 6],
    cur: usize,
    end_cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,
    /// Where `BadQuiet` resumes the quiet region (perf study 34 §5).
    ///
    /// `GoodQuiet` and `BadQuiet` partition the quiet region by the SAME
    /// predicate, so between them they used to inspect every quiet TWICE:
    /// `GoodQuiet` walked to the end of the region and threw its position
    /// away, and `BadQuiet` then restarted at `end_captures` and re-rejected
    /// every above-threshold element `GoodQuiet` had just emitted. Once
    /// [`sorted_covers`](Self::sorted_covers) holds, the split point is a
    /// single index — the sorted prefix's first sub-threshold element — so
    /// `GoodQuiet` stops there and `BadQuiet` starts there. Both stages
    /// emit exactly what they emitted before, in the same order.
    bad_quiet_start: usize,
    /// True when the quiet sort's `limit` is at or below
    /// `GOOD_QUIET_THRESHOLD`, i.e. when the sorted prefix is guaranteed to
    /// contain every above-threshold quiet (perf study 34 §5).
    ///
    /// The sort places element 0 plus everything scoring `>= limit` in a
    /// descending prefix and leaves the rest — all strictly below `limit` —
    /// in the tail. So if `limit <= GOOD_QUIET_THRESHOLD`, every tail element
    /// is below the threshold as well, and within the prefix descending order
    /// means the sub-threshold elements are a contiguous suffix. The whole
    /// "good" set is then the prefix `[end_captures, bad_quiet_start)` and the
    /// whole "bad" set is `[bad_quiet_start, end_generated)`.
    ///
    /// `limit = QUIET_SORT_DEPTH_FACTOR * depth` (-3560·depth by default) and
    /// the threshold is -14000, so this holds from depth 4 up. It is computed
    /// rather than assumed because both are SPSA-tunable under `tune`; at
    /// depths 1-3 the flag is false and both stages take the old full scan,
    /// which is what keeps shallow nodes bit-identical.
    sorted_covers: bool,
    /// Experiment (design/17 Thrust II): the shared learned move-prior, blended
    /// into quiet ordering when `POLICY_NET` is set. `None` ⇒ pure history stack
    /// (the default build never compiles this in). Set once per node by the
    /// search via [`set_policy_net`](MovePicker::set_policy_net); the forward is
    /// run lazily in [`init_quiets`](MovePicker::init_quiets), so nodes that cut
    /// off before generating quiets pay nothing.
    #[cfg(feature = "policy")]
    policy_net: Option<std::sync::Arc<crate::policy::PolicyNet>>,
    /// Lazily computed once per node and shared by capture/quiet scoring when
    /// the capture-policy experiment is enabled.
    #[cfg(feature = "policy")]
    policy_scores: Option<crate::policy::PolicyScores>,
    /// P-ORD: the deployed D47 head's quiet-move prior for this node, set by
    /// the search when `CPL_ORD_W != 0` and the node's L2 activation was
    /// available at construction. `None` ⇒ pure history stack (the default —
    /// the env knob ships OFF, so ordering is bit-identical unless opted in).
    #[cfg(feature = "sota")]
    quiet_prior: Option<crate::eval::nnue::policy_head::QuietPrior>,
    /// Perf study #2 option C: UNINITIALISED buffer — the old
    /// `[ExtMove::NULL; MAX_MOVES]` zeroed 2 KB per node (plus the
    /// constructor's return copy) for slots no stage ever reads.
    ///
    /// SAFETY ARGUMENT (the D14a-style exemption this module carries):
    /// every read goes through [`Self::slot`] / [`Self::slots_mut`] /
    /// [`Self::swap_slots`], and the region-index invariant — reads only
    /// ever address `[0, end_generated)` (emission windows, sorts, swaps),
    /// which is exactly the union of the fill loops' written prefixes —
    /// is machine-checked in debug builds by the `written` tracker below
    /// (the `assert_consistent` precedent): every `set_slot` marks, every
    /// read asserts. Release compiles the tracker out.
    buf: [core::mem::MaybeUninit<ExtMove>; MAX_MOVES],
    #[cfg(debug_assertions)]
    written: [bool; MAX_MOVES],
    /// Perf study 34's census (instrument builds only): per-picker tallies,
    /// aggregated in `Drop` because only the drop point knows how far the
    /// search actually walked before the node ended.
    #[cfg(feature = "census34")]
    c: Census,
}

/// Per-picker census tallies (perf study 34, instrument builds only).
#[cfg(feature = "census34")]
#[derive(Default)]
struct Census {
    /// Length of the sorted prefix `partial_insertion_sort` produced for the
    /// quiet region — the elements whose order the sort actually paid for.
    prefix: u32,
    n_quiets: u32,
    gq_scanned: u32,
    gq_emitted: u32,
    bq_scanned: u32,
    bq_emitted: u32,
    gc_scanned: u32,
    gc_emitted: u32,
    /// Set once `init_quiets` has run, so `Drop` can tell a picker that never
    /// reached `QuietInit` from one that generated an empty quiet list.
    quiets_generated: bool,
}

#[cfg(feature = "census34")]
impl Drop for MovePicker {
    fn drop(&mut self) {
        use crate::census34 as cx;
        cx::bump(&cx::GC_SCANNED, self.c.gc_scanned as u64);
        cx::bump(&cx::GC_EMITTED, self.c.gc_emitted as u64);
        if !self.c.quiets_generated {
            return;
        }
        cx::bump(&cx::GQ_SCANNED, self.c.gq_scanned as u64);
        cx::bump(&cx::GQ_EMITTED, self.c.gq_emitted as u64);
        cx::bump(&cx::BQ_SCANNED, self.c.bq_scanned as u64);
        cx::bump(&cx::BQ_EMITTED, self.c.bq_emitted as u64);
        cx::bump(&cx::Q_SKIPPED, self.skip_quiets as u64);
        // "Exhausted" = the search consumed every quiet the picker generated.
        let walked = self.c.gq_scanned + self.c.bq_scanned;
        cx::bump(&cx::Q_EXHAUSTED, u64::from(walked >= self.c.n_quiets));
        cx::hist(&cx::HIST_GQ, self.c.gq_scanned as usize);
        cx::hist(&cx::HIST_NQ, self.c.n_quiets as usize);
        cx::hist(&cx::HIST_PFX, self.c.prefix as usize);
    }
}

impl MovePicker {
    /// Read one buffer slot. Debug builds assert the slot was written.
    #[inline]
    fn slot(&self, i: usize) -> ExtMove {
        #[cfg(debug_assertions)]
        debug_assert!(self.written[i], "movepick read of unwritten slot {i}");
        // SAFETY: the region-index invariant (see `buf`) guarantees `i` was
        // written by a fill loop; the debug tracker machine-checks it.
        #[allow(unsafe_code)]
        unsafe {
            self.buf[i].assume_init()
        }
    }

    /// Write one buffer slot (and mark it in debug builds).
    #[inline]
    fn set_slot(&mut self, i: usize, e: ExtMove) {
        self.buf[i].write(e);
        #[cfg(debug_assertions)]
        {
            self.written[i] = true;
        }
    }

    /// A `&mut [ExtMove]` view of a fully-written region (for the sorts).
    #[inline]
    fn slots_mut(&mut self, r: core::ops::Range<usize>) -> &mut [ExtMove] {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.written[r.clone()].iter().all(|&w| w),
            "movepick slice over unwritten slots {r:?}"
        );
        // SAFETY: every index in `r` was written (region invariant +
        // debug tracker); MaybeUninit<T> has the same layout as T.
        #[allow(unsafe_code)]
        unsafe {
            core::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(r.start).cast(), r.len())
        }
    }

    /// Swap two WRITTEN slots (the good/bad-capture parking swap).
    #[inline]
    fn swap_slots(&mut self, a: usize, b: usize) {
        #[cfg(debug_assertions)]
        debug_assert!(self.written[a] && self.written[b]);
        self.buf.swap(a, b);
    }

    /// Main search (`depth > 0`), quiescence (`depth <= 0`), and evasions
    /// (side to move in check) — one constructor, like Stockfish
    /// (design/09 §3.5). `cont_hist` are the six predecessor continuation-
    /// history keys the quiet scorer reads (step 7).
    ///
    /// The base stage is `EvasionTt` when in check, else `MainTt` for
    /// `depth > 0`, else `QsearchTt`; the branchless TT skip then advances
    /// past the `*Tt` stage exactly when the TT move is unusable.
    pub fn new(
        board: &Board,
        tt_move: Move,
        depth: Depth,
        ply: usize,
        cont_hist: [crate::history::ContHistId; 6],
    ) -> Self {
        let base = if !board.checkers().is_empty() {
            Stage::EvasionTt
        } else if depth > 0 {
            Stage::MainTt
        } else {
            Stage::QsearchTt
        };
        // A TT move is usable iff it is a real move that this position would
        // actually generate; an unusable one skips the `*Tt` stage (§3.5).
        let usable = tt_move.is_ok() && board.pseudo_legal(tt_move);
        // Perf study 34: price the two `pseudo_legal` paths that run a FULL
        // move generation just to validate one move (study 23 remainder 6).
        #[cfg(feature = "census34")]
        {
            use crate::census34 as cx;
            let in_check = !board.checkers().is_empty();
            let cand = tt_move.is_ok();
            cx::bump(&cx::MP_NEW, 1);
            cx::bump(&cx::MP_TT_CAND, cand as u64);
            cx::bump(&cx::MP_TT_USABLE, usable as u64);
            cx::bump(
                &cx::MP_TT_NONNORMAL,
                (cand && tt_move.kind() != crate::moves::MoveKind::Normal) as u64,
            );
            cx::bump(
                &cx::MP_TT_EVASION_GEN,
                (cand && in_check && tt_move.kind() == crate::moves::MoveKind::Normal) as u64,
            );
        }
        MovePicker::with_base(base, tt_move, depth, ply, VALUE_ZERO, cont_hist, usable)
    }

    /// ProbCut (design/09 §3.5): captures clearing `see_ge(threshold)`, where
    /// `threshold = probcut_beta - static_eval` is computed by the caller
    /// (doc 10). The TT move is usable only if it is additionally a capture or
    /// queen promotion (`capture_stage`).
    pub fn new_probcut(board: &Board, tt_move: Move, threshold: Value) -> Self {
        let usable = tt_move.is_ok() && board.pseudo_legal(tt_move) && board.capture_stage(tt_move);
        MovePicker::with_base(
            Stage::ProbcutTt,
            tt_move,
            DEPTH_QS,
            0,
            threshold,
            [crate::history::ContHistId::SENTINEL; 6],
            usable,
        )
    }

    /// Shared field setup + branchless TT skip (design/09 §3.5): an unusable
    /// TT move lands directly on the `*Init` stage via `base + !usable`.
    #[allow(clippy::too_many_arguments)]
    fn with_base(
        base: Stage,
        tt_move: Move,
        depth: Depth,
        ply: usize,
        threshold: Value,
        cont_hist: [crate::history::ContHistId; 6],
        usable: bool,
    ) -> Self {
        MovePicker {
            stage: Stage::from_u8(base as u8 + (!usable) as u8),
            tt_move,
            depth,
            ply,
            threshold,
            skip_quiets: false,
            cont_hist,
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            bad_quiet_start: 0,
            sorted_covers: false,
            #[cfg(feature = "policy")]
            policy_net: None,
            #[cfg(feature = "policy")]
            policy_scores: None,
            #[cfg(feature = "sota")]
            quiet_prior: None,
            buf: [core::mem::MaybeUninit::uninit(); MAX_MOVES],
            #[cfg(debug_assertions)]
            written: [false; MAX_MOVES],
            #[cfg(feature = "census34")]
            c: Census::default(),
        }
    }

    /// Attach the shared policy net for this node (design/17 Thrust II). Called
    /// by the search right after construction; a no-op cost when `None`. The
    /// forward is deferred to [`init_quiets`](MovePicker::init_quiets).
    #[cfg(feature = "policy")]
    #[inline]
    pub fn set_policy_net(&mut self, net: Option<std::sync::Arc<crate::policy::PolicyNet>>) {
        self.policy_net = net;
    }

    #[cfg(feature = "policy")]
    #[inline]
    pub fn set_policy_scores(&mut self, scores: Option<crate::policy::PolicyScores>) {
        self.policy_scores = scores;
    }

    /// Attach the D47 trained-head quiet prior for this node (P-ORD).
    #[cfg(feature = "sota")]
    #[inline]
    pub fn set_quiet_prior(&mut self, prior: crate::eval::nnue::policy_head::QuietPrior) {
        self.quiet_prior = Some(prior);
    }

    /// Emit one pseudo-legal move per call, `None` once the active pipeline is
    /// exhausted (design/09 §3.6). The caller checks legality only on the move
    /// it actually plays. Idempotent after exhaustion: further calls keep
    /// returning `None`.
    ///
    /// **Every fall-through assigns a LITERAL stage, never `self.stage.bump()`
    /// (perf study 27 arm 3).** The two are identical in behaviour, and the
    /// `bump()` spelling reads better, but it hides the destination behind
    /// arithmetic on a value LLVM loaded from memory — so each `continue` had
    /// to go back through the 16-entry jump table (`movslq (%r13,%rcx,4); add
    /// %r13,%rcx; jmp *%rcx`) to reach a block the source already names. With
    /// a literal the back-edge is a direct jump. That is also why the shared
    /// `CaptureInit | ProbcutInit | QcaptureInit` arm is split into three
    /// copies of one line: the bodies are the same, the successors are not.
    // `inline(always)`, not `inline` -- perf study 27 arm 5 measured the
    // difference at -37.6 instructions and -6.2 branches per node, and study 23
    // left the same change on the table for want of an instrument. What made it
    // look expensive was `L1-dcache-load-misses` rising; measuring the
    // DENOMINATOR settles that -- total L1d LOADS fall 4.1/node, so this is less
    // memory work at a marginally worse hit rate, not more work.
    #[inline(always)]
    pub fn next_move(&mut self, board: &Board, hist: &Histories) -> Option<Move> {
        loop {
            match self.stage {
                // §3.6 item 1: the whole point of staging — a TT-move cutoff
                // costs zero generation and scoring. The branchless skip in
                // the constructor guarantees `tt_move` is usable here.
                Stage::MainTt | Stage::EvasionTt | Stage::ProbcutTt | Stage::QsearchTt => {
                    self.stage = self.stage.bump();
                    return Some(self.tt_move);
                }

                // §3.6 item 2: generate + score captures, full sort, advance.
                // Main search, ProbCut, and quiescence share this init verbatim.
                Stage::CaptureInit => {
                    self.init_captures(board, hist);
                    self.stage = Stage::GoodCapture;
                }
                Stage::ProbcutInit => {
                    self.init_captures(board, hist);
                    self.stage = Stage::Probcut;
                }
                Stage::QcaptureInit => {
                    self.init_captures(board, hist);
                    self.stage = Stage::Qcapture;
                }

                // Quiescence: every capture, TT move deduped.
                Stage::Qcapture => return self.select(board, |_, _| true),

                // ProbCut: captures clearing the SEE threshold, TT deduped.
                Stage::Probcut => {
                    let threshold = self.threshold;
                    return self.select(board, move |b, e| b.see_ge(e.mv, threshold));
                }

                // §3.6 stage 3: the score-dependent good/bad split. A capture
                // clears the SEE bar `-value / 18` (a high ordering score buys
                // slack) to be *good*; failures are swapped down into the
                // bad-captures region for later and skipped here.
                Stage::GoodCapture => {
                    while self.cur < self.end_cur {
                        let ext = self.slot(self.cur);
                        #[cfg(feature = "census34")]
                        {
                            self.c.gc_scanned += 1;
                        }
                        // Dedup: the TT move was already emitted at MainTt.
                        if ext.mv == self.tt_move {
                            self.cur += 1;
                            continue;
                        }
                        if board.see_ge(ext.mv, -ext.value / GOOD_CAPTURE_SEE_DIVISOR()) {
                            self.cur += 1;
                            #[cfg(feature = "census34")]
                            {
                                self.c.gc_emitted += 1;
                            }
                            return Some(ext.mv);
                        }
                        // Bad: park it in `[0, end_bad_captures)`. The stale,
                        // already-emitted element it displaces goes to `cur`,
                        // which `cur += 1` then steps past.
                        self.swap_slots(self.end_bad_captures, self.cur);
                        self.end_bad_captures += 1;
                        self.cur += 1;
                    }
                    self.stage = Stage::QuietInit;
                }

                // §3.6 stage 4: generate + score quiets (unless skipping),
                // laying them out just past the captures region.
                Stage::QuietInit => {
                    if !self.skip_quiets {
                        self.init_quiets(board, hist);
                    }
                    self.stage = Stage::GoodQuiet;
                }

                // §3.6 stage 5: quiets scoring above the threshold, then rewind
                // the emission window to the bad-captures region.
                //
                // Spelled as its own loop rather than a `select` closure
                // (perf study 34 §5) for the one thing a closure cannot say:
                // BREAK. Under `sorted_covers` the first sub-threshold element
                // ends the stage — the prefix is descending and the tail is
                // entirely below the limit, so nothing after it can pass — and
                // the index it stops at is exactly where `BadQuiet` must
                // resume. The old form ran on to the end of the region
                // rejecting every remaining element, then discarded the
                // position it had reached.
                Stage::GoodQuiet => {
                    if !self.skip_quiets {
                        while self.cur < self.end_cur {
                            let ext = self.slot(self.cur);
                            #[cfg(feature = "census34")]
                            {
                                self.c.gq_scanned += 1;
                            }
                            if ext.value <= GOOD_QUIET_THRESHOLD() {
                                if self.sorted_covers {
                                    break;
                                }
                                self.cur += 1;
                                continue;
                            }
                            self.cur += 1;
                            // Dedup: the TT move was already emitted at MainTt.
                            if ext.mv != self.tt_move {
                                #[cfg(feature = "census34")]
                                {
                                    self.c.gq_emitted += 1;
                                }
                                return Some(ext.mv);
                            }
                        }
                    }
                    // Without the sorted-prefix guarantee the good and bad sets
                    // interleave, so `BadQuiet` still has to rescan the region
                    // from the top — the depth-1..3 path, bit-identical to the
                    // pre-study behaviour.
                    self.bad_quiet_start = if self.sorted_covers {
                        self.cur
                    } else {
                        self.end_captures
                    };
                    self.cur = 0;
                    self.end_cur = self.end_bad_captures;
                    self.stage = Stage::BadCapture;
                }

                // §3.6 stage 6: the deferred bad captures (in swap order), then
                // rewind to the quiet region for the sub-threshold quiets.
                Stage::BadCapture => {
                    if let Some(m) = self.select(board, |_, _| true) {
                        return Some(m);
                    }
                    self.cur = self.bad_quiet_start;
                    self.end_cur = self.end_generated;
                    self.stage = Stage::BadQuiet;
                }

                // §3.6 stage 7: the exact complement of GoodQuiet (`value <=
                // threshold`), so the two passes partition the quiet region.
                Stage::BadQuiet => {
                    if self.skip_quiets {
                        return None;
                    }
                    return self.select(board, |_, e| e.value <= GOOD_QUIET_THRESHOLD());
                }

                // Evasion pipeline (§3.6): generate + score all evasions
                // (captures lifted above quiets), full sort, emit every one.
                Stage::EvasionInit => {
                    self.init_evasions(board, hist);
                    self.stage = Stage::Evasion;
                }
                Stage::Evasion => return self.select(board, |_, _| true),
            }
        }
    }

    /// Generate captures into the buffer, score them (§3.7), lay out the
    /// captures region, and full-sort it (design/09 §3.6 item 2). Generation
    /// uses a stack [`MoveList`]; the scored moves are copied into the owned
    /// buffer — no heap allocation.
    // OUTLINED (perf-impl-1). These three run at most once per stage per node;
    // `select` and the stage machine around them run per emitted move. Inlined,
    // they fused into one 2,665-instruction `next_move` whose register pressure
    // cost ~1.5 G spill/reload loads over a 5 M-node bench. Splitting them out
    // is pure code layout -- node counts are bit-identical -- and buys 6.2%
    // fewer instructions and 8.4% fewer L1d loads. See
    // docs/research/perf-impl-1-*.md.
    #[inline(never)]
    fn init_captures(&mut self, board: &Board, hist: &Histories) {
        let mut list = MoveList::new();
        generate_captures(board, &mut list);
        let n = list.len();
        #[cfg(feature = "policy")]
        if crate::policy::captures_enabled()
            && crate::policy::enabled_at(self.depth, self.ply)
            && self.policy_scores.is_none()
        {
            self.policy_scores = self.policy_net.as_ref().and_then(|net| net.scores(board));
        }
        #[cfg(feature = "policy")]
        let policy_weight = crate::policy::capture_weight();
        for (i, &m) in list.iter().enumerate() {
            let value = score_capture(board, hist, m);
            // Borrow `policy_scores` INSIDE the loop, not before it: binding it
            // outside holds an immutable borrow of `*self` across the whole
            // body, which collides with the `set_slot` below. Scoped here, NLL
            // ends the borrow at the end of the expression — the same shape the
            // `quiet_prior` read in `score_quiets` uses.
            #[cfg(feature = "policy")]
            let value = value
                + self
                    .policy_scores
                    .as_ref()
                    .map(|scores| {
                        (policy_weight as f32 * scores.move_score(m.from(), m.to())) as i32
                    })
                    .unwrap_or(0);
            self.set_slot(i, ExtMove::new(m, value));
        }
        // Captures occupy `[0, n)`; nothing has been swapped into the
        // bad-captures region and no quiets exist yet.
        self.cur = 0;
        self.end_bad_captures = 0;
        self.end_captures = n;
        self.end_generated = n;
        self.end_cur = n;
        partial_insertion_sort(self.slots_mut(0..n), i32::MIN);
        #[cfg(feature = "census34")]
        {
            use crate::census34 as cx;
            cx::bump(&cx::IC_CALLS, 1);
            cx::bump(&cx::IC_SCORED, n as u64);
        }
    }

    /// Late-move pruning (design/09 §3.5): abandon quiet generation and
    /// emission — bad captures still come. Idempotent and callable
    /// mid-iteration; if set before `QuietInit` no quiets are ever generated,
    /// if set during `GoodQuiet` the remaining quiets are simply not emitted.
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    /// Generate quiets into `buf[end_captures..]`, score them (§3.7), and
    /// partial-sort only the promising prefix (limit `-3560 * depth`; deeper
    /// nodes sort more — design/09 §3.6 stage 4 / §3.8).
    ///
    /// The five continuation-history slabs, the pawn-history slot, the
    /// butterfly/low-ply planes, and the threat sets are resolved **once**
    /// here, not per move (design/09 §6): each move then costs a handful of
    /// array reads.
    #[inline(never)]
    fn init_quiets(&mut self, board: &Board, hist: &Histories) {
        let mut list = MoveList::new();
        generate_quiets(board, &mut list);
        let n = list.len();

        let us = board.side_to_move();
        // Escape/entry tiers from the Wave-G threat map (D30): squares attacked
        // by an enemy piece strictly cheaper than each of our types (cumulative
        // union — the same structure the old local `threats_by_lesser`
        // produced, now read O(1) from the precomputed per-node map).
        // ONE fill check for all five reads (perf study 13 §4.2): the four
        // `piece_threats` calls and the `all_threats` below were five separate
        // `ensure_threats` calls, of which four could only ever find the cache
        // already filled.
        let (all_threats, by_type) = board.threat_map();
        let by_pawn = by_type[PieceType::Pawn as usize];
        let by_minor =
            by_pawn | by_type[PieceType::Knight as usize] | by_type[PieceType::Bishop as usize];
        let by_rook = by_minor | by_type[PieceType::Rook as usize];
        let threat: [Bitboard; 6] = [
            Bitboard::EMPTY, // Pawn: no cheaper attacker
            by_pawn,         // Knight
            by_pawn,         // Bishop
            by_minor,        // Rook
            by_rook,         // Queen
            Bitboard::EMPTY, // King
        ];
        // Offense (D30): the "create a threat" counterpart — for each of our
        // piece types, the undefended enemy pieces (not in the opponent's own
        // attack map) worth at least as much as the mover; a quiet moving to a
        // square attacking one earns the offense bonus.
        let them = us.flip();
        let occ = board.all_occupancy();
        let undefended = board.occupancy(them) & !all_threats;
        let eq = board.pieces(them, PieceType::Queen);
        let er = eq | board.pieces(them, PieceType::Rook);
        let eb = er | board.pieces(them, PieceType::Bishop);
        let en = eb | board.pieces(them, PieceType::Knight);
        let ep = en | board.pieces(them, PieceType::Pawn);
        let vulnerable: [Bitboard; 6] = [
            ep & undefended, // Pawn: any undefended enemy
            en & undefended, // Knight: >= knight value
            eb & undefended, // Bishop
            er & undefended, // Rook
            eq & undefended, // Queen: only an undefended queen
            Bitboard::EMPTY, // King
        ];
        let main_plane = &hist.main[us as usize];
        let pawn_plane = hist.pawn_slot(board.pawn_key());
        // Offsets 1,2,3,4,6 → cont_hist[0,1,2,3,5]; offset 5 is not read for
        // ordering (only written by updates — design/09 §3.7).
        let c1 = hist.cont_ref(self.cont_hist[0]);
        let c2 = hist.cont_ref(self.cont_hist[1]);
        let c3 = hist.cont_ref(self.cont_hist[2]);
        let c4 = hist.cont_ref(self.cont_hist[3]);
        let c6 = hist.cont_ref(self.cont_hist[5]);
        let ply = self.ply;
        let low_ply_plane = (ply < LOW_PLY_SIZE).then(|| &hist.low_ply[ply]);
        // Hoist the slider-table `LazyLock` deref (CLAUDE.md hot-loop rule).
        // Since perf study 12 §5 the offense term no longer calls `attacks()`
        // per quiet — it builds one reverse fan per piece type below — so this
        // is now paid once per `init_quiets` and read only inside that fan.
        let att_tables = crate::attacks::tables();

        // Perf study 12 §5 — the offense term, turned inside out.
        //
        // The test is `attacks(pt, to, occ) & vulnerable[pt] != 0`, which cost
        // ONE magic-bitboard lookup per scored quiet (Stockfish's `score<QUIETS>`
        // has no analogue at all — this is an ARCHON original, D30). The attack
        // relation is SYMMETRIC under a fixed occupancy: `sq ∈ attacks(pt, to,
        // occ)` iff `to ∈ attacks(pt, sq, occ)`, because both say "sq and to lie
        // on a pt-ray and every square strictly between them is empty in occ".
        // So the union of the *reverse* fans over the (few) vulnerable enemy
        // squares is a per-piece-type bitboard, and the per-move test collapses
        // to one `contains`. Lookups drop from ~one per quiet to
        // Σ_pt popcount(vulnerable[pt]) — and those sets are nested and usually
        // near-empty. Bit-identical by the symmetry above; measured −34 i/n,
        // node-identical.
        //
        // Pawns are the one asymmetric case: `target ∈ pawn_attacks(us, to)`
        // iff `to ∈ pawn_attacks(them, target)` — the reverse fan flips colour.
        let offense_from: [Bitboard; 6] = std::array::from_fn(|i| {
            let targets = vulnerable[i];
            if targets.is_empty() {
                return Bitboard::EMPTY;
            }
            let pt = PieceType::ALL[i];
            let mut fan = Bitboard::EMPTY;
            for sq in targets {
                fan |= if pt == PieceType::Pawn {
                    crate::attacks::pawn_attacks(them, sq)
                } else {
                    att_tables.attacks(pt, sq, occ)
                };
            }
            fan
        });

        // Experiment (design/17 Thrust II): one policy forward per node, reused
        // by every quiet below. Owned (borrows `self.policy_net` only for the
        // `map`), so the mutable `self.buf[..]` writes in the loop stay clean.
        #[cfg(feature = "policy")]
        if crate::policy::enabled_at(self.depth, self.ply) && self.policy_scores.is_none() {
            self.policy_scores = self.policy_net.as_ref().and_then(|net| net.scores(board));
        }
        #[cfg(feature = "policy")]
        let policy_weight = crate::policy::quiet_weight();

        // One `st()` for all ~30 scored quiets (perf study 13 §4.4) instead of
        // one per quiet: `pt` varies per move, so the query cannot be hoisted
        // at the call site.
        let check_squares = board.check_squares_all();

        // Perf study 23 arm 4 — ONE per-piece-type row instead of four parallel
        // arrays. The loop below reads five `pt`-indexed facts per quiet, and
        // before this they lived in four different places with four different
        // bases (two stack arrays, the board's `StateInfo`, and the `PIECE_VALUES`
        // table) plus a two-instruction `imul`/`lea` chain to turn the piece
        // value into the threat and offense weights. Interleaved into a 32-byte
        // row they share ONE `pt << 5` address, sit in one cache line, and the
        // weights are premultiplied once per call (6 multiplies) instead of once
        // per quiet (~23). Pure code shape: the arithmetic is unchanged, so the
        // scores — and therefore the tree — are bit-identical.
        // Spelled as a literal, not `array::from_fn`: the closure form left a
        // real six-iteration loop with four bounds checks in it (measured
        // +326 instructions per call), while the literal is six independent
        // 32-byte stores LLVM folds into vector moves.
        let row_of = |i: usize| PtRow {
            threat: threat[i],
            offense: offense_from[i],
            check: check_squares[i],
            threat_w: PieceType::ALL[i].value() * QUIET_THREAT_FACTOR(),
            offense_w: PieceType::ALL[i].value() * QUIET_OFFENSE_FACTOR(),
        };
        let rows: [PtRow; 6] = [
            row_of(0),
            row_of(1),
            row_of(2),
            row_of(3),
            row_of(4),
            row_of(5),
        ];
        let base = self.end_captures;

        // Perf study 44 §6 — SOFTWARE-PIPELINED HISTORY PREFETCH.
        //
        // The scoring loop below reads SEVEN randomly-addressed history cells
        // per quiet, from tables totalling ~20 MiB that share a 32 MB L3 with
        // a 16 MB TT and ~29 MB of NNUE weights. Study 44 sampled
        // `ex_no_retire.load_not_complete` and found `init_quiets` owns
        // **6.57% of every load-stall cycle in the engine**, of which 89% is
        // load-wait rather than execution — i.e. this loop is not computing,
        // it is waiting for `pawn` (12.6 MB) and the five `cont` planes
        // (5.1 MB).
        //
        // The six plane BASES are already hoisted above the loop, but a base
        // is not a line: `q` varies per move, so the line is not known until
        // the iteration that needs it, and the loop body is far too short to
        // cover a DRAM round trip. Looking `PF_DIST` moves ahead turns the
        // per-move dependent miss into an overlapped one, which is the same
        // trade `TranspositionTable::prefetch` already makes (study 42 priced
        // that one hint at 3.5% of engine cycles).
        //
        // NODE-IDENTICAL BY CONSTRUCTION, not by test: `_mm_prefetch` computes
        // no value the score depends on, so the tree cannot move. The gate
        // still checks it, because "by construction" has been wrong before.
        //
        // `PF_DIST` is the one free parameter and it was measured, not picked
        // — see §6.2 for the 2/4/8 sweep.
        #[cfg(feature = "histpf44")]
        const PF_DIST: usize = 4;
        #[cfg(feature = "histpf44")]
        let prefetch_history_for = |m: crate::moves::Move| {
            // A quiet always starts on one of our pieces, but this runs on a
            // move the loop has not reached yet, so it asks rather than
            // asserts — the `expect` below stays the loop's own contract.
            if let Some(pc) = board.piece_on(m.from()) {
                let q = crate::history::piece_to(pc.index(), m.to().index() as usize);
                crate::history::prefetch_piece_to(pawn_plane, q);
                crate::history::prefetch_piece_to(c1, q);
                crate::history::prefetch_piece_to(c2, q);
                crate::history::prefetch_piece_to(c3, q);
                crate::history::prefetch_piece_to(c4, q);
                crate::history::prefetch_piece_to(c6, q);
                crate::history::prefetch_line(
                    main_plane.as_ptr().wrapping_add(m.raw16() as usize),
                );
            }
        };
        // Prologue: the first `PF_DIST` moves have no earlier iteration to
        // issue their hint, so they get it here, before the loop's first body.
        #[cfg(feature = "histpf44")]
        for &m in list.iter().take(PF_DIST) {
            prefetch_history_for(m);
        }

        for (i, &m) in list.iter().enumerate() {
            #[cfg(feature = "histpf44")]
            if let Some(&ahead) = list.get(i + PF_DIST) {
                prefetch_history_for(ahead);
            }
            let pc = board
                .piece_on(m.from())
                .expect("a quiet move starts on a piece of the side to move");
            let pt = pc.piece_type;
            // NOT clamped with `.min(5)` / `.min(11)`, though the bounds are type
            // invariants and LLVM cannot see them (both values arrive through a
            // mailbox byte, so it emits a `cmp`/`jae` and a panic block for each,
            // twice per scored quiet). Perf study 23 measured the clamps: they
            // remove 7.4 branches/node — an exact match for the 2 panic edges x
            // 3.93 quiets/node they target — and cost **+21.7 instructions/node**
            // doing it, because the `cmov` blocks the addressing-mode folding the
            // bounds-checked form gets for free. Predicted branches are cheap;
            // this trade is not worth taking.
            let row = &rows[pt as usize];
            let pci = pc.index();
            let to = m.to().index() as usize;
            let raw = m.raw16() as usize;

            // Perf study 36 — SIX planes, ONE cell. Written two-dimensionally
            // (`p[pci][to]`), LLVM shares `pci * 128` and `to * 2` in two
            // registers and then pays a `lea` per plane to add that plane's
            // base, because `base + pci*128 + to*2` is three variable terms and
            // an x86 addressing mode holds two. `piece_to` makes the shared
            // term the WHOLE offset, so each plane becomes one
            // `movswl (%base,%q,2)`. Same cells, same order, same sum —
            // `[[T; 64]; 12]` is contiguous row-major — so the tree is
            // bit-identical, and `q < 768` is provable (see `history::piece_to`)
            // so no bounds check or mask comes back in exchange.
            let q = crate::history::piece_to(pci, to);

            let mut value = QUIET_MAIN_HISTORY_WEIGHT() * main_plane[raw].get()
                + QUIET_PAWN_HISTORY_WEIGHT() * crate::history::at_piece_to(pawn_plane, q)
                + crate::history::at_piece_to(c1, q)
                + crate::history::at_piece_to(c2, q)
                + crate::history::at_piece_to(c3, q)
                + crate::history::at_piece_to(c4, q)
                + crate::history::at_piece_to(c6, q);

            // Check bonus, gated so a checking quiet that hangs material earns
            // nothing (design/09 §3.7).
            if row.check.contains(m.to()) && board.see_ge(m, QUIET_CHECK_SEE_MARGIN()) {
                value += QUIET_CHECK_BONUS();
            }

            // Threat escape/entry: +factor·value for fleeing a cheaper-piece
            // attack, -factor·value for stepping into one (design/09 §3.7).
            let from_threat = row.threat.contains(m.from()) as i32;
            let to_threat = row.threat.contains(m.to()) as i32;
            value += row.threat_w * (from_threat - to_threat);

            // Offense (D30): moving to a square from which the piece attacks a
            // vulnerable enemy earns a modest reward. `occ` is the pre-move
            // occupancy (a heuristic approximation — the mover still sits on
            // `from`); acceptable for ordering. Precomputed as a reverse fan
            // above, so this is one bit test instead of a magic lookup.
            if row.offense.contains(m.to()) {
                value += row.offense_w;
            }

            // Low-ply history (plies 0..4 only), damped toward the leaves.
            if let Some(lp) = low_ply_plane {
                value += QUIET_LOW_PLY_WEIGHT() * lp[raw].get() / (1 + ply as i32);
            }

            // Learned move-prior (design/17 Thrust II), scaled into history
            // units. Additive, so it only nudges ties the history stack leaves
            // ambiguous rather than overriding a strong history signal.
            // Read through `self` here rather than binding the borrow above
            // the loop: an outer binding holds `*self` immutably across the
            // whole body and collides with `set_slot`. This is the same shape
            // as the `quiet_prior` read just below.
            #[cfg(feature = "policy")]
            if let Some(ps) = &self.policy_scores {
                value += (policy_weight as f32 * ps.move_score(m.from(), m.to())) as i32;
            }

            // P-ORD: the deployed D47 head's prior, quiets only (captures stay
            // on MVV/SEE — the head is near-blind there by training-data
            // construction). One selected-row SIMD dot per quiet.
            //
            // Called through the outlined [`quiet_prior_bonus`], NOT inlined —
            // see that function's note. `CPL_ORD_W` ships 0, so the branch is
            // not taken in any shipped configuration.
            #[cfg(feature = "sota")]
            if let Some(qp) = &self.quiet_prior {
                value += quiet_prior_bonus(qp, m);
            }

            self.set_slot(base + i, ExtMove::new(m, value));
        }

        self.cur = base;
        self.end_cur = base + n;
        self.end_generated = base + n;
        let sort_limit = QUIET_SORT_DEPTH_FACTOR() * self.depth;
        // See `sorted_covers`: with the prefix limit at or below the good/bad
        // threshold, the two emission stages split the region at one index.
        //
        // `null34` is perf study 34's NULL ARM: it forces the flag off, so the
        // binary carries the arm's struct layout, its extra fields and its
        // rewritten `GoodQuiet` loop, and performs the OLD amount of scanning.
        // A null that does not perturb the same machinery the arm perturbs is
        // not a null (study 23 §2.2) — a byte-different build with an inert
        // `let` reordering would not have moved the stage machine at all.
        self.sorted_covers = !cfg!(feature = "null34") && sort_limit <= GOOD_QUIET_THRESHOLD();
        // Census BEFORE the sort: the prefix model is "element 0 plus every
        // LATER element >= limit", and the sort permutes the range, so the
        // count is only well defined on the pre-sort order.
        #[cfg(feature = "census34")]
        let pfx = if n == 0 {
            0
        } else {
            1 + self
                .slots_mut(base + 1..base + n)
                .iter()
                .filter(|e| e.value >= sort_limit)
                .count()
        };
        partial_insertion_sort(self.slots_mut(base..base + n), sort_limit);
        #[cfg(feature = "census34")]
        {
            use crate::census34 as cx;
            cx::bump(&cx::IQ_CALLS, 1);
            cx::bump(&cx::IQ_SCORED, n as u64);
            cx::bump(&cx::IQ_PREFIX, pfx as u64);
            self.c.prefix = pfx as u32;
            self.c.n_quiets = n as u32;
            self.c.quiets_generated = true;
        }
    }

    /// Generate + score check evasions into `buf[0..n]`, full-sort, and set the
    /// emission window over the whole region (design/09 §3.6/§3.7). Capturing
    /// evasions are lifted above every quiet evasion by [`EVASION_CAPTURE_BONUS`].
    #[inline(never)]
    fn init_evasions(&mut self, board: &Board, hist: &Histories) {
        let mut list = MoveList::new();
        generate_evasions(board, &mut list);
        let n = list.len();

        let us = board.side_to_move();
        let main_plane = &hist.main[us as usize];
        let c1 = hist.cont_ref(self.cont_hist[0]);

        for (i, &m) in list.iter().enumerate() {
            let value = if board.capture_stage(m) {
                let victim_value = board.piece_on(m.to()).map_or(0, |p| p.piece_type.value());
                victim_value + EVASION_CAPTURE_BONUS()
            } else {
                let pc = board
                    .piece_on(m.from())
                    .expect("an evasion starts on a piece of the side to move");
                main_plane[m.raw16() as usize].get() + c1[pc.index()][m.to().index() as usize].get()
            };
            self.set_slot(i, ExtMove::new(m, value));
        }

        self.cur = 0;
        self.end_cur = n;
        self.end_captures = 0;
        self.end_bad_captures = 0;
        self.end_generated = n;
        partial_insertion_sort(self.slots_mut(0..n), i32::MIN);
        #[cfg(feature = "census34")]
        crate::census34::bump(&crate::census34::IE_CALLS, 1);
    }

    /// Scan the emission window `buf[cur..end_cur]` for the next element passing
    /// `ok`, skipping any equal to the already-emitted TT move (design/09 §3.6
    /// shared select semantics). Advances `cur` past every element it inspects.
    /// The closure sees the whole [`ExtMove`] so value-filtered stages (good/
    /// bad quiets) can test the ordering score directly.
    fn select(
        &mut self,
        board: &Board,
        mut ok: impl FnMut(&Board, ExtMove) -> bool,
    ) -> Option<Move> {
        while self.cur < self.end_cur {
            let ext = self.slot(self.cur);
            self.cur += 1;
            #[cfg(feature = "census34")]
            self.census_scan();
            // Dedup: the TT move was already emitted at the `*Tt` stage.
            if ext.mv != self.tt_move && ok(board, ext) {
                #[cfg(feature = "census34")]
                self.census_emit();
                return Some(ext.mv);
            }
        }
        None
    }

    /// Census (perf study 34): attribute one inspected element to its stage.
    #[cfg(feature = "census34")]
    #[inline]
    fn census_scan(&mut self) {
        match self.stage {
            Stage::GoodQuiet => self.c.gq_scanned += 1,
            Stage::BadQuiet => self.c.bq_scanned += 1,
            Stage::Qcapture | Stage::Probcut => self.c.gc_scanned += 1,
            _ => {}
        }
    }

    /// Census (perf study 34): attribute one emitted move to its stage.
    #[cfg(feature = "census34")]
    #[inline]
    fn census_emit(&mut self) {
        match self.stage {
            Stage::GoodQuiet => self.c.gq_emitted += 1,
            Stage::BadQuiet => self.c.bq_emitted += 1,
            Stage::Qcapture | Stage::Probcut => self.c.gc_emitted += 1,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Histories;
    use crate::movegen::generate_quiets;
    use crate::types::{Color, Piece, Square};
    use std::collections::HashSet;

    // A tiny LCG (same multiplier/increment as history.rs' tests) —
    // deterministic pseudo-randomness with zero dev-dependencies.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    // A distinctly-tagged ExtMove: `idx` (0..256) uniquely identifies the slot
    // via its move bits, so a multiset of (raw16, value) tracks permutations.
    fn ext(idx: usize, value: i32) -> ExtMove {
        ExtMove::new(Move::from_raw16(idx as u16), value)
    }

    fn multiset(slice: &[ExtMove]) -> Vec<(u16, i32)> {
        let mut v: Vec<(u16, i32)> = slice.iter().map(|e| (e.mv.raw16(), e.value)).collect();
        v.sort_unstable();
        v
    }

    // --- design/09 §5 step 5: ExtMove representation ---

    #[test]
    fn ext_move_is_eight_bytes_align_four() {
        assert_eq!(size_of::<ExtMove>(), 8);
        assert_eq!(align_of::<ExtMove>(), 4);
    }

    // --- design/09 §5 step 5: partial_insertion_sort contract ---

    // The reference model of §3.8: the prefix is element 0 plus every later
    // element scoring `>= limit`; the tail is everything else.
    fn reference_split(input: &[ExtMove], limit: i32) -> (Vec<ExtMove>, Vec<ExtMove>) {
        let mut prefix = vec![input[0]];
        let mut tail = Vec::new();
        for &e in &input[1..] {
            if e.value >= limit {
                prefix.push(e);
            } else {
                tail.push(e);
            }
        }
        (prefix, tail)
    }

    fn assert_sort_contract(input: &[ExtMove], limit: i32) {
        let mut buf = input.to_vec();
        partial_insertion_sort(&mut buf, limit);

        // Whole slice is a permutation of the input.
        assert_eq!(
            multiset(&buf),
            multiset(input),
            "sort must preserve the multiset (limit {limit})"
        );

        let (want_prefix, want_tail) = reference_split(input, limit);
        let plen = want_prefix.len();

        // The prefix is exactly the reference prefix, sorted descending.
        assert!(
            buf[..plen].windows(2).all(|w| w[0].value >= w[1].value),
            "prefix must be descending (limit {limit}): {:?}",
            &buf[..plen]
        );
        assert_eq!(
            multiset(&buf[..plen]),
            multiset(&want_prefix),
            "prefix content wrong (limit {limit})"
        );
        assert_eq!(
            multiset(&buf[plen..]),
            multiset(&want_tail),
            "tail content wrong (limit {limit})"
        );
    }

    #[test]
    fn partial_sort_edge_cases() {
        // Empty and single-element slices are no-ops.
        let mut empty: [ExtMove; 0] = [];
        partial_insertion_sort(&mut empty, 0);
        let mut one = [ext(0, 42)];
        partial_insertion_sort(&mut one, 100);
        assert_eq!(one[0].value, 42);

        // Element 0 below the limit is still anchored into the prefix.
        assert_sort_contract(&[ext(0, 1), ext(1, 5), ext(2, 3)], 4);
        // Tail-perturbing case from the doc walk-through.
        assert_sort_contract(&[ext(0, 5), ext(1, 1), ext(2, 2), ext(3, 8)], 4);
    }

    #[test]
    fn partial_sort_limit_above_max_keeps_only_element_zero() {
        // No later element clears the limit: the prefix is just element 0 and
        // the slice is otherwise untouched.
        let input = [ext(0, 10), ext(1, 50), ext(2, 30), ext(3, 20)];
        let mut buf = input;
        partial_insertion_sort(&mut buf, 1000);
        assert_eq!(buf, input, "limit above max must leave the slice unchanged");
        assert_sort_contract(&input, 1000);
    }

    #[test]
    fn partial_sort_int_min_is_a_full_descending_sort() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in [2usize, 3, 8, 40, 200] {
            let input: Vec<ExtMove> = (0..len)
                .map(|i| {
                    // Values in a narrow band so ties occur (stability check).
                    let v = (lcg_next(&mut state) >> 40) as i32 % 11 - 5;
                    ext(i, v)
                })
                .collect();
            let mut buf = input.clone();
            partial_insertion_sort(&mut buf, i32::MIN);

            // §5: full sort equals a reference descending sort ignoring ties.
            let mut want: Vec<i32> = input.iter().map(|e| e.value).collect();
            want.sort_unstable_by(|a, b| b.cmp(a));
            let got: Vec<i32> = buf.iter().map(|e| e.value).collect();
            assert_eq!(got, want, "i32::MIN must fully sort (len {len})");

            // And the general contract holds for the full-sort limit too.
            assert_sort_contract(&input, i32::MIN);
        }
    }

    #[test]
    fn partial_sort_randomized_contract_sweep() {
        let mut state = 0xdead_beef_cafe_babeu64;
        for _ in 0..2000 {
            let len = 1 + (lcg_next(&mut state) % 60) as usize;
            let input: Vec<ExtMove> = (0..len)
                .map(|i| ext(i, (lcg_next(&mut state) >> 33) as i32 - (1 << 30)))
                .collect();
            // A limit somewhere within the value band, plus the extremes.
            let limit = (lcg_next(&mut state) >> 33) as i32 - (1 << 30);
            assert_sort_contract(&input, limit);
            assert_sort_contract(&input, i32::MIN);
            assert_sort_contract(&input, i32::MAX);
        }
    }

    // --- design/09 §5 step 6: qsearch / probcut emission completeness ---

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap_or_else(|e| panic!("bad test FEN {fen:?}: {e}"))
    }

    // FEN fixtures with captures available and the side to move NOT in check
    // (in-check qsearch takes the evasion pipeline, step 8). Startpos is
    // included as the no-captures degenerate case.
    const FIXTURES: &[&str] = &[
        crate::fen::START_FEN,
        // Kiwipete: many captures, promotions, and en-passant-free tactics.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // A sharp middlegame with hanging pieces both sides.
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        // Position 5 of the perft suite: promotions and captures.
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        // An en-passant-capable position (black just played d7d5).
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        // Endgame with a capturing promotion available.
        "8/2P1P3/3K4/8/8/3k4/2p1p3/8 w - - 0 1",
    ];

    // The captures the generator produces, as a dedup set (the generator emits
    // each once, so a set is faithful and doubles as a duplicate check).
    fn capture_set(b: &Board) -> HashSet<Move> {
        let mut list = MoveList::new();
        generate_captures(b, &mut list);
        let set: HashSet<Move> = list.iter().copied().collect();
        assert_eq!(
            set.len(),
            list.len(),
            "generator emitted a duplicate capture"
        );
        set
    }

    // Drain a picker, asserting it never emits a duplicate.
    fn drain(mut mp: MovePicker, b: &Board, hist: &Histories) -> Vec<Move> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        while let Some(m) = mp.next_move(b, hist) {
            assert!(seen.insert(m), "picker emitted {m} twice");
            out.push(m);
        }
        out
    }

    // Poke pseudo-random values into the capture history — the only table the
    // capture scorer reads — to prove ordering never changes the emitted set.
    fn randomize_captures(hist: &mut Histories, seed: u64) {
        let mut state = seed;
        for plane in hist.capture.iter_mut() {
            for row in plane.iter_mut() {
                for cell in row.iter_mut() {
                    let v = (lcg_next(&mut state) >> 48) as i16 % 9001 - 4500;
                    *cell = crate::history::StatEntry::new(v);
                }
            }
        }
    }

    // A quiet, pseudo-legal move for the side to move, if one exists — used as
    // a non-capture TT move (usable, but absent from the capture set).
    fn some_quiet(b: &Board) -> Option<Move> {
        let mut list = MoveList::new();
        generate_quiets(b, &mut list);
        list.iter().copied().find(|&m| m.is_ok())
    }

    #[test]
    fn qsearch_emits_exactly_the_captures() {
        for &fen in FIXTURES {
            let b = board(fen);
            assert!(
                b.checkers().is_empty(),
                "fixture {fen} must not be in check"
            );
            let captures = capture_set(&b);

            for hist in [Histories::new(), {
                let mut h = Histories::new();
                randomize_captures(&mut h, 0x51ed ^ fen.len() as u64);
                h
            }] {
                // No TT move: the emission set is exactly the captures.
                let mp = MovePicker::new(&b, Move::NULL, DEPTH_QS, 0, sentinels());
                let emitted = drain(mp, &b, &hist);
                assert_eq!(
                    emitted.iter().copied().collect::<HashSet<_>>(),
                    captures,
                    "qsearch emission set != captures in {fen}"
                );

                // A capturing TT move is emitted first and exactly once.
                if let Some(&tt) = captures.iter().next() {
                    let mp = MovePicker::new(&b, tt, DEPTH_QS, 0, sentinels());
                    let emitted = drain(mp, &b, &hist);
                    assert_eq!(emitted[0], tt, "usable TT capture must come first in {fen}");
                    assert_eq!(
                        emitted.iter().copied().collect::<HashSet<_>>(),
                        captures,
                        "qsearch with TT capture must still emit every capture in {fen}"
                    );
                }

                // A quiet (usable, non-capture) TT move is emitted first, then
                // every capture — the set grows by exactly that one move.
                if let Some(tt) = some_quiet(&b) {
                    assert!(!captures.contains(&tt));
                    let mp = MovePicker::new(&b, tt, DEPTH_QS, 0, sentinels());
                    let emitted = drain(mp, &b, &hist);
                    assert_eq!(
                        emitted[0], tt,
                        "usable quiet TT move must come first in {fen}"
                    );
                    let mut want = captures.clone();
                    want.insert(tt);
                    assert_eq!(
                        emitted.iter().copied().collect::<HashSet<_>>(),
                        want,
                        "qsearch with quiet TT move must add exactly it in {fen}"
                    );
                    assert_eq!(emitted.len(), captures.len() + 1);
                }
            }
        }
    }

    #[test]
    fn probcut_neg_infinity_threshold_emits_all_captures() {
        // A threshold below any real SEE — every capture clears it — must
        // reproduce the whole capture set (design/09 §5).
        const NEG_INF: Value = -30_000;
        for &fen in FIXTURES {
            let b = board(fen);
            let captures = capture_set(&b);
            let hist = Histories::new();

            let mp = MovePicker::new_probcut(&b, Move::NULL, NEG_INF);
            let emitted = drain(mp, &b, &hist);
            assert_eq!(
                emitted.iter().copied().collect::<HashSet<_>>(),
                captures,
                "probcut(-inf) emission set != captures in {fen}"
            );

            if let Some(&tt) = captures.iter().next() {
                let mp = MovePicker::new_probcut(&b, tt, NEG_INF);
                let emitted = drain(mp, &b, &hist);
                assert_eq!(
                    emitted[0], tt,
                    "usable probcut TT capture must come first in {fen}"
                );
                assert_eq!(
                    emitted.iter().copied().collect::<HashSet<_>>(),
                    captures,
                    "probcut(-inf) with TT capture must emit every capture in {fen}"
                );
            }
        }
    }

    #[test]
    fn probcut_threshold_matches_see_ge_filter() {
        // With no TT move the probcut emission set is exactly the captures
        // passing `see_ge(threshold)` — verified directly against `see_ge`.
        for &fen in FIXTURES {
            let b = board(fen);
            let captures = capture_set(&b);
            let hist = Histories::new();
            for threshold in [-200, -100, 0, 100, 200, 800] {
                let want: HashSet<Move> = captures
                    .iter()
                    .copied()
                    .filter(|&m| b.see_ge(m, threshold))
                    .collect();
                let mp = MovePicker::new_probcut(&b, Move::NULL, threshold);
                let emitted = drain(mp, &b, &hist);
                assert_eq!(
                    emitted.iter().copied().collect::<HashSet<_>>(),
                    want,
                    "probcut(threshold={threshold}) != see_ge filter in {fen}"
                );
            }
        }
    }

    #[test]
    fn unusable_tt_move_is_never_emitted() {
        // A corrupt / illegal TT move (not pseudo-legal here) must not appear
        // and must not disturb the capture set.
        let b = board(FIXTURES[1]);
        let captures = capture_set(&b);
        let hist = Histories::new();
        // h1h8: no such pseudo-legal move in this position.
        let bogus = Move::normal(Square::H1, Square::H8);
        assert!(!b.pseudo_legal(bogus));
        let mp = MovePicker::new(&b, bogus, DEPTH_QS, 0, sentinels());
        let emitted = drain(mp, &b, &hist);
        assert!(!emitted.contains(&bogus));
        assert_eq!(emitted.iter().copied().collect::<HashSet<_>>(), captures);
    }

    fn sentinels() -> [crate::history::ContHistId; 6] {
        [crate::history::ContHistId::SENTINEL; 6]
    }

    // --- design/09 §5 steps 7-8: main + evasion pipelines ---

    // A value in `[-bound, bound]` from the shared LCG.
    fn rnd(state: &mut u64, bound: i16) -> i16 {
        (((lcg_next(state) >> 40) as i64) % (2 * bound as i64 + 1) - bound as i64) as i16
    }

    // Poke pseudo-random values into every table the quiet + capture scorers
    // read (main, low-ply, capture, pawn, continuation). Ordering must change
    // freely while the *emitted set* stays invariant (design/09 §5).
    fn randomize_ordering(hist: &mut Histories, seed: u64) {
        let mut s = seed;
        for row in hist.main.iter_mut() {
            for c in row.iter_mut() {
                *c = crate::history::StatEntry::new(rnd(&mut s, 7000));
            }
        }
        for row in hist.low_ply.iter_mut() {
            for c in row.iter_mut() {
                *c = crate::history::StatEntry::new(rnd(&mut s, 7000));
            }
        }
        for plane in hist.capture.iter_mut() {
            for row in plane.iter_mut() {
                for c in row.iter_mut() {
                    *c = crate::history::StatEntry::new(rnd(&mut s, 10000));
                }
            }
        }
        for plane in hist.pawn.iter_mut() {
            for row in plane.iter_mut() {
                for c in row.iter_mut() {
                    *c = crate::history::StatEntry::new(rnd(&mut s, 8000));
                }
            }
        }
        for slab in hist.cont.iter_mut() {
            for row in slab.iter_mut() {
                for c in row.iter_mut() {
                    *c = crate::history::StatEntry::new(rnd(&mut s, 29000));
                }
            }
        }
    }

    // The full pseudo-legal set of a not-in-check position (captures ⊎ quiets).
    fn nonevasion_set(b: &Board) -> HashSet<Move> {
        let mut list = MoveList::new();
        crate::movegen::generate_non_evasions(b, &mut list);
        let set: HashSet<Move> = list.iter().copied().collect();
        assert_eq!(set.len(), list.len(), "generator emitted a duplicate move");
        set
    }

    // The pseudo-legal evasion set of an in-check position.
    fn evasion_set(b: &Board) -> HashSet<Move> {
        let mut list = MoveList::new();
        generate_evasions(b, &mut list);
        let set: HashSet<Move> = list.iter().copied().collect();
        assert_eq!(
            set.len(),
            list.len(),
            "generator emitted a duplicate evasion"
        );
        set
    }

    // The set of quiet moves (for stage-order assertions).
    fn quiet_set(b: &Board) -> HashSet<Move> {
        let mut list = MoveList::new();
        generate_quiets(b, &mut list);
        list.iter().copied().collect()
    }

    // The main pipeline must emit exactly the full pseudo-legal set, with the
    // usable TT move (capturing or quiet) emitted first and exactly once.
    fn assert_main_emits_full(b: &Board, hist: &Histories) {
        let full = nonevasion_set(b);

        // No TT move: emission set == the whole pseudo-legal set.
        let emitted = drain(MovePicker::new(b, Move::NULL, 8, 0, sentinels()), b, hist);
        assert_eq!(emitted.iter().copied().collect::<HashSet<_>>(), full);
        assert_eq!(emitted.len(), full.len());

        // A capturing TT move: first, once, set unchanged.
        if let Some(&tt) = capture_set(b).iter().next() {
            let emitted = drain(MovePicker::new(b, tt, 8, 0, sentinels()), b, hist);
            assert_eq!(emitted[0], tt, "usable TT capture must come first");
            assert_eq!(emitted.iter().copied().collect::<HashSet<_>>(), full);
            assert_eq!(emitted.len(), full.len());
        }

        // A quiet TT move: first, once, set unchanged (it is already in `full`).
        if let Some(tt) = some_quiet(b) {
            let emitted = drain(MovePicker::new(b, tt, 8, 0, sentinels()), b, hist);
            assert_eq!(emitted[0], tt, "usable quiet TT move must come first");
            assert_eq!(emitted.iter().copied().collect::<HashSet<_>>(), full);
            assert_eq!(emitted.len(), full.len());
        }
    }

    #[test]
    fn main_pipeline_emits_full_pseudo_legal_set() {
        // One fresh and one fully-randomized history, reused across fixtures:
        // the emission set must be identical (ordering-invariant, design/09 §5).
        let fresh = Histories::new();
        let mut rand = Histories::new();
        randomize_ordering(&mut rand, 0x0DDF_00D5_u64);
        for &fen in FIXTURES {
            let b = board(fen);
            assert!(
                b.checkers().is_empty(),
                "fixture {fen} must not be in check"
            );
            assert_main_emits_full(&b, &fresh);
            assert_main_emits_full(&b, &rand);
        }
    }

    // In-check fixtures with a mix of capturing and quiet evasions.
    const EVASION_FIXTURES: &[&str] = &[
        // Rook checks the king from an adjacent square: Kxe2 is a capturing
        // evasion alongside quiet king steps.
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
        // Rook checks down the c-file: only quiet king moves evade.
        "2r1k3/8/8/8/8/8/8/2K5 w - - 0 1",
        // Queen checks adjacently: both Rxd2 and Kxd2 capture the checker.
        "4k3/8/8/8/8/8/3q4/3RK3 w - - 0 1",
    ];

    #[test]
    fn evasion_pipeline_emits_all_evasions_captures_first() {
        let fresh = Histories::new();
        let mut rand = Histories::new();
        randomize_ordering(&mut rand, 0x0E7A_5104_u64);
        for &fen in EVASION_FIXTURES {
            let b = board(fen);
            assert!(!b.checkers().is_empty(), "fixture {fen} must be in check");
            let all = evasion_set(&b);
            for hist in [&*fresh, &*rand] {
                // Without a TT move the pipeline is EvasionInit → Evasion, fully
                // sorted: every capturing evasion (score ≥ 2^28) precedes every
                // quiet one, and the emitted set is exactly the generator's.
                let emitted = drain(MovePicker::new(&b, Move::NULL, 8, 0, sentinels()), &b, hist);
                assert_eq!(
                    emitted.iter().copied().collect::<HashSet<_>>(),
                    all,
                    "evasion set != generator in {fen}"
                );
                let mut seen_quiet = false;
                for &m in &emitted {
                    if b.capture_stage(m) {
                        assert!(!seen_quiet, "capturing evasion after a quiet one in {fen}");
                    } else {
                        seen_quiet = true;
                    }
                }

                // A usable TT evasion is emitted first and exactly once; the set
                // is unchanged.
                let tt = *all.iter().next().unwrap();
                let emitted = drain(MovePicker::new(&b, tt, 8, 0, sentinels()), &b, hist);
                assert_eq!(emitted[0], tt, "usable TT evasion must come first in {fen}");
                assert_eq!(
                    emitted.iter().copied().collect::<HashSet<_>>(),
                    all,
                    "TT evasion changed the set in {fen}"
                );
            }
        }
    }

    // design/09 §5 / §8 criterion 1: the four-pipeline emission-completeness
    // sweep over the FULL perft FEN suite (design/06), not merely the
    // hand-picked FIXTURES above. For every position and each pipeline, the
    // emitted multiset must equal the corresponding generator's — in check the
    // evasion pipeline; otherwise the main pipeline (all pseudo-legal moves),
    // quiescence (captures), and probcut with a −∞ threshold (captures) — with
    // the usable TT move emitted first and exactly once (criterion 3). Run
    // against both fresh and fully-randomized histories: ordering may change
    // freely, the emitted SET may not.
    const PERFT_SUITE: &str = include_str!("../tests/data/perft.epd");

    // The FEN of each non-comment, non-blank EPD line is everything before the
    // first `;D<depth>` perft field.
    fn perft_suite_fens() -> Vec<&'static str> {
        PERFT_SUITE
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split(" ;").next().unwrap().trim())
            .collect()
    }

    #[test]
    fn emission_completeness_over_full_perft_suite() {
        const NEG_INF: Value = -30_000;
        let fresh = Histories::new();
        let mut rand = Histories::new();
        randomize_ordering(&mut rand, 0xF00D_5EED_u64);

        let fens = perft_suite_fens();
        assert!(fens.len() >= 20, "perft suite unexpectedly small");

        for fen in fens {
            let b = board(fen);
            for hist in [&*fresh, &*rand] {
                if b.checkers().is_empty() {
                    // Main pipeline: full pseudo-legal set, TT (capturing or
                    // quiet) first and once — reuses the fixture-test oracle.
                    assert_main_emits_full(&b, hist);

                    // Quiescence and probcut(−∞) both reproduce the capture set.
                    let caps = capture_set(&b);
                    let q = drain(
                        MovePicker::new(&b, Move::NULL, DEPTH_QS, 0, sentinels()),
                        &b,
                        hist,
                    );
                    assert_eq!(
                        q.iter().copied().collect::<HashSet<_>>(),
                        caps,
                        "qsearch set != captures in {fen}"
                    );
                    let p = drain(MovePicker::new_probcut(&b, Move::NULL, NEG_INF), &b, hist);
                    assert_eq!(
                        p.iter().copied().collect::<HashSet<_>>(),
                        caps,
                        "probcut(-inf) set != captures in {fen}"
                    );
                } else {
                    // Evasion pipeline: every evasion, captures first, TT first.
                    let all = evasion_set(&b);
                    let emitted =
                        drain(MovePicker::new(&b, Move::NULL, 8, 0, sentinels()), &b, hist);
                    assert_eq!(
                        emitted.iter().copied().collect::<HashSet<_>>(),
                        all,
                        "evasion set != generator in {fen}"
                    );
                    let mut seen_quiet = false;
                    for &m in &emitted {
                        if b.capture_stage(m) {
                            assert!(!seen_quiet, "capturing evasion after a quiet one in {fen}");
                        } else {
                            seen_quiet = true;
                        }
                    }
                    // (A suite position is never checkmate, but guard the
                    // empty set rather than unwrap a would-be panic.)
                    if let Some(&tt) = all.iter().next() {
                        let emitted = drain(MovePicker::new(&b, tt, 8, 0, sentinels()), &b, hist);
                        assert_eq!(emitted[0], tt, "usable TT evasion must come first in {fen}");
                        assert_eq!(
                            emitted.iter().copied().collect::<HashSet<_>>(),
                            all,
                            "TT evasion changed the set in {fen}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn capture_history_flips_good_to_bad() {
        // An equal rook trade (SEE exactly 0): the black rook a2 is defended
        // only by its king on b3, so Rxa2 then Kxa2 nets zero.
        let b = board("8/8/8/8/8/1k6/r7/R6K w - - 0 1");
        assert!(b.checkers().is_empty());
        let rxa2 = Move::normal(Square::A1, Square::A2);
        assert!(b.pseudo_legal(rxa2) && b.is_capture(rxa2));
        assert!(
            b.see_ge(rxa2, 0) && !b.see_ge(rxa2, 1),
            "the trade must be exactly even"
        );
        let quiets = quiet_set(&b);
        assert!(!quiets.is_empty());

        // Fresh history: an even capture clears `-value/18` → GOOD, so it is
        // emitted before every quiet.
        let fresh = Histories::new();
        let emitted = drain(
            MovePicker::new(&b, Move::NULL, 8, 0, sentinels()),
            &b,
            &fresh,
        );
        let good_idx = emitted
            .iter()
            .position(|&m| m == rxa2)
            .expect("Rxa2 emitted");
        for &q in &quiets {
            let qi = emitted.iter().position(|&m| m == q).unwrap();
            assert!(
                good_idx < qi,
                "fresh: good capture Rxa2 must precede quiet {q}"
            );
        }

        // Terrible capture history drives `value` negative, so `-value/18`
        // becomes a positive material demand the even trade cannot meet → BAD,
        // emitted after every (good) quiet.
        let mut poked = Histories::new();
        poked.capture[Piece::new(Color::White, PieceType::Rook).index()]
            [Square::A2.index() as usize][PieceType::Rook as usize] =
            crate::history::StatEntry::new(-10692);
        let emitted = drain(
            MovePicker::new(&b, Move::NULL, 8, 0, sentinels()),
            &b,
            &poked,
        );
        let bad_idx = emitted
            .iter()
            .position(|&m| m == rxa2)
            .expect("Rxa2 emitted");
        for &q in &quiets {
            let qi = emitted.iter().position(|&m| m == q).unwrap();
            assert!(
                bad_idx > qi,
                "poked: bad capture Rxa2 must follow quiet {q}"
            );
        }
    }

    // Drive a picker's capture+quiet init directly and read one quiet's score
    // out of the buffer (region order is irrelevant — we search the whole quiet
    // region for the move). Fresh histories make every table cell its init
    // value, so scores are hand-computable.
    fn quiet_score(b: &Board, hist: &Histories, m: Move, ply: usize) -> i32 {
        let mut mp = MovePicker::new(b, Move::NULL, 8, ply, sentinels());
        mp.init_captures(b, hist);
        mp.init_quiets(b, hist);
        mp.slots_mut(mp.end_captures..mp.end_generated)
            .iter()
            .find(|e| e.mv == m)
            .unwrap_or_else(|| panic!("quiet {m} not generated"))
            .value
    }

    #[test]
    fn quiet_scoring_matches_hand_computed_values() {
        // Fresh-history base: 2·main(−5) + 2·pawn(−1262) + 5·cont(−552).
        const BASE: i32 = 2 * -5 + 2 * -1262 + 5 * -552; // −5294
        let fresh = Histories::new();

        // (a) Plain quiet at ply ≥ 5 (no low-ply term): a pawn push gives no
        //     check and a pawn has no cheaper attacker (threat[Pawn] EMPTY).
        let b = board(crate::fen::START_FEN);
        let a3 = Move::normal(Square::A2, Square::A3);
        assert_eq!(quiet_score(&b, &fresh, a3, 7), BASE);

        // (b) Check bonus: Rd1-d4 checks the king on d8 and is SEE-safe.
        let b = board("3k4/8/8/8/8/8/8/3RK3 w - - 0 1");
        let rd4 = Move::normal(Square::D1, Square::D4);
        assert!(b.check_squares(PieceType::Rook).contains(Square::D4));
        assert_eq!(quiet_score(&b, &fresh, rd4, 7), BASE + QUIET_CHECK_BONUS());

        // (c) Threat escape: a queen fleeing a rook-attacked square earns
        //     +factor·QUEEN_VALUE.
        let b = board("3rk3/8/8/8/3Q4/8/8/4K3 w - - 0 1");
        let qf4 = Move::normal(Square::D4, Square::F4);
        assert_eq!(
            quiet_score(&b, &fresh, qf4, 7),
            BASE + PieceType::Queen.value() * QUIET_THREAT_FACTOR()
        );

        // (d) Threat entry: a queen stepping onto a rook-attacked square pays
        //     the same, sinking the move far below the other quiets.
        let b = board("3rk3/8/8/8/8/8/8/Q3K3 w - - 0 1");
        let qd1 = Move::normal(Square::A1, Square::D1);
        assert_eq!(
            quiet_score(&b, &fresh, qd1, 7),
            BASE - PieceType::Queen.value() * QUIET_THREAT_FACTOR()
        );
        assert!(
            quiet_score(&b, &fresh, qd1, 7) < GOOD_QUIET_THRESHOLD(),
            "must be a bad quiet"
        );
    }

    #[test]
    fn wave_g_offense_term_scores_only_an_undefended_target() {
        // G1 offense (D30): a quiet moving to a square from which the piece
        // attacks an UNDEFENDED enemy piece worth >= the mover earns
        // QUIET_OFFENSE_FACTOR · piece_value. This pins BOTH branches of the
        // term precisely (hand-computed), including the disarm branch the
        // determinism/mate smoke tests never touch.
        //
        // Fresh-history base (same as quiet_scoring_matches_hand_computed_values).
        const BASE: i32 = 2 * -5 + 2 * -1262 + 5 * -552; // −5294
        let fresh = Histories::new();
        let nc6 = Move::normal(Square::E5, Square::C6);

        // Armed: the lone Black rook on b8 is undefended (nothing Black attacks
        // b8). White Ne5–c6 attacks b8 (rook ≥ knight), so the offense term
        // fires and is the ONLY non-base term (no check, no threat escape:
        // threat[Knight] is the pawn map, and Black has no pawns).
        let armed = board("1r5k/8/8/4N3/8/8/8/K7 w - - 0 1");
        assert!(
            !armed.all_threats().contains(Square::B8),
            "the lone Black rook on b8 is undefended (absent from Black's attack map)"
        );
        assert_eq!(
            quiet_score(&armed, &fresh, nc6, 7),
            BASE + PieceType::Knight.value() * QUIET_OFFENSE_FACTOR(),
            "attacking an undefended enemy rook must add the offense bonus"
        );

        // Disarmed: a Black knight on a6 defends b8, so b8 joins Black's attack
        // map and is no longer 'undefended' — the offense bonus is withheld and
        // the identical move scores the bare base. (a6 is a Knight attacker, so
        // it feeds piece_threats(Knight)/by_minor, never threat[Knight] — the
        // White knight's escape term stays zero, so nothing else moves.)
        let defended = board("1r5k/8/n7/4N3/8/8/8/K7 w - - 0 1");
        assert!(
            defended.all_threats().contains(Square::B8),
            "with a defender the rook's square joins Black's attack map"
        );
        assert_eq!(
            quiet_score(&defended, &fresh, nc6, 7),
            BASE,
            "a DEFENDED enemy is not a vulnerable target — the offense bonus is disarmed"
        );
    }

    /// Perf study 12 §5 rewrote the offense term as a REVERSE attack fan. The
    /// knight case above cannot catch the two things that rewrite actually
    /// changed, because a knight's fan is both occupancy- and colour-independent:
    ///
    ///  * the **pawn colour flip** — the reverse fan of `pawn_attacks(us, to)`
    ///    is `pawn_attacks(them, target)`, not `pawn_attacks(us, target)`;
    ///  * **slider occupancy symmetry** — the reverse fan must see the same
    ///    blockers the forward lookup did.
    ///
    /// Writing `us` for `them`, or fanning with an empty occupancy, passes every
    /// other test in the suite. This is the only place either is pinned.
    #[test]
    fn offense_reverse_fan_pins_the_pawn_flip_and_slider_blockers() {
        const BASE: i32 = 2 * -5 + 2 * -1262 + 5 * -552; // −5294, as above
        let fresh = Histories::new();

        // PAWN. b3–b4 attacks the undefended rook on c5: b4 ∈ pawn_attacks(us=White, c5)
        // forward, i.e. c5 ∈ pawn_attacks(them=Black, b4) reversed. The reverse fan
        // of c5 for a White pawn mover is pawn_attacks(Black, c5) = {b4, d4}.
        // Fanning with `us` gives pawn_attacks(White, c5) = {b6, d6} and silently
        // withholds the bonus.
        let pawn = board("7k/8/8/2r5/8/1P6/8/K7 w - - 0 1");
        assert!(
            !pawn.all_threats().contains(Square::C5),
            "the lone rook must be undefended for the term to arm"
        );
        assert_eq!(
            quiet_score(&pawn, &fresh, Move::normal(Square::B3, Square::B4), 7),
            BASE + PieceType::Pawn.value() * QUIET_OFFENSE_FACTOR(),
            "a pawn attacking an undefended rook must earn the offense bonus"
        );

        // SLIDER, clear ray. Re1–d1 attacks the undefended queen on d8 down an
        // empty d-file.
        let open = board("3q3k/8/8/8/8/8/8/K3R3 w - - 0 1");
        assert!(!open.all_threats().contains(Square::D8));
        assert_eq!(
            quiet_score(&open, &fresh, Move::normal(Square::E1, Square::D1), 7),
            BASE + PieceType::Rook.value() * QUIET_OFFENSE_FACTOR(),
            "a rook attacking an undefended queen down a clear file earns the bonus"
        );

        // SLIDER, blocked. The same move with a pawn on d5: the reverse fan
        // `rook_attacks(d8, occ)` stops at d5 and never reaches d1, exactly as
        // the forward `rook_attacks(d1, occ)` stopped at d5 and never reached d8.
        // An occupancy-blind fan would wrongly pay the bonus here.
        let blocked = board("3q3k/8/8/3p4/8/8/8/K3R3 w - - 0 1");
        assert_eq!(
            quiet_score(&blocked, &fresh, Move::normal(Square::E1, Square::D1), 7),
            BASE,
            "a blocked ray must NOT earn the offense bonus"
        );
    }

    #[test]
    fn capture_scoring_matches_hand_computed_values() {
        // 7·victim_value + capture_history; a fresh capture cell is −699.
        let b = board("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1");
        let exd5 = Move::normal(Square::E4, Square::D5);
        assert!(b.is_capture(exd5));
        let expect = CAPTURE_MVV_FACTOR() * PieceType::Knight.value() - 699;
        assert_eq!(score_capture(&b, &Histories::new(), exd5), expect);
        assert_eq!(expect, 4768); // 7·781 − 699
    }

    #[test]
    fn skip_quiet_moves_suppresses_quiets() {
        // Kiwipete: many captures and quiets, not in check.
        let b = board(FIXTURES[1]);
        assert!(b.checkers().is_empty());
        let hist = Histories::new();
        let captures = capture_set(&b);
        let quiets = quiet_set(&b);
        assert!(!captures.is_empty() && !quiets.is_empty());

        // Skip before QuietInit: no quiet is ever generated, yet every capture
        // (good and bad) still comes out.
        let mut mp = MovePicker::new(&b, Move::NULL, 8, 0, sentinels());
        mp.skip_quiet_moves();
        let emitted = drain(mp, &b, &hist);
        for &m in &emitted {
            assert!(!quiets.contains(&m), "quiet {m} emitted despite skip");
        }
        assert_eq!(emitted.iter().copied().collect::<HashSet<_>>(), captures);

        // Skip mid-GoodQuiet: quiets already emitted stay, no further quiets
        // appear, and the bad captures still follow.
        let mut mp = MovePicker::new(&b, Move::NULL, 8, 0, sentinels());
        let mut got = Vec::new();
        let mut seen = HashSet::new();
        let mut skipped = false;
        while let Some(m) = mp.next_move(&b, &hist) {
            assert!(seen.insert(m), "duplicate {m} after mid-skip");
            if !skipped && quiets.contains(&m) {
                mp.skip_quiet_moves();
                skipped = true;
            }
            got.push(m);
        }
        assert!(
            skipped,
            "no quiet ever appeared to trigger the mid-iteration skip"
        );
        assert!(
            captures.iter().all(|c| got.contains(c)),
            "a capture went missing after a mid-iteration skip"
        );
    }
}
