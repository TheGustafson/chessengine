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
//! - **step 5 (here):** [`ExtMove`] (the move + ordering score, §3.1) and
//!   [`partial_insertion_sort`] (the limited descending sort, §3.8).
//! - **step 6 (here):** the [`MovePicker`] skeleton — the single owned
//!   `[ExtMove; MAX_MOVES]` buffer carved into regions by *indices* (§3.1),
//!   the branchless TT-stage skip (§3.5), the dedup [`select`](MovePicker)
//!   scan (§3.6) — and the two capture-only pipelines it drives now:
//!   quiescence (`QsearchTt → QcaptureInit → Qcapture`) and ProbCut
//!   (`ProbcutTt → ProbcutInit → Probcut` with the `see_ge(threshold)` gate).
//! - later steps (07/08): the main pipeline (good/bad capture split, quiet
//!   scoring + split, `skip_quiet_moves`) and the evasion pipeline.
//!
//! Hot-path rules (design/09 §6): no heap allocation on any [`next_move`]
//! path (generation uses a stack [`MoveList`], scoring writes into the owned
//! buffer), no `unsafe` (D14), and the picker holds no borrows — `&Board` and
//! `&Histories` are passed into each call so the search can freely
//! make/unmake and update histories between calls.
//!
//! [`next_move`]: MovePicker::next_move

use crate::board::Board;
use crate::history::Histories;
use crate::movegen::generate_captures;
use crate::moves::{MAX_MOVES, Move, MoveList};
use crate::search::params::CAPTURE_MVV_FACTOR;
use crate::types::{DEPTH_QS, Depth, VALUE_ZERO, Value};

/// A move paired with its ordering score (design/09 §3.1, D7).
///
/// Scoring happens only inside the picker's own buffer; [`MoveList`] stays two
/// bytes per move (doc 05). The representation contract is fixed in doc 02
/// §3.8 and asserted below: exactly 8 bytes, `#[repr(C)]`, align 4. Two
/// `ExtMove`s are ordered by `value` only — the move bits are payload, never a
/// tiebreak (the `partial_insertion_sort` below compares `value`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ExtMove {
    /// The move (`u16`).
    pub mv: Move,
    /// The ordering score; sorted and filtered on this alone.
    pub value: i32,
}

impl ExtMove {
    /// The buffer-fill placeholder: an inert slot no emission stage reads
    /// (region indices guarantee only written slots are ever consumed).
    const NULL: ExtMove = ExtMove {
        mv: Move::NULL,
        value: 0,
    };
}

// design/09 §8 criterion 2 / doc 02 §3.8: 8 bytes, align 4. `#[repr(C)]` lays
// out `mv` (u16) then two padding bytes then `value` (i32), so the size is 8
// and the alignment is the i32's 4.
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
pub(crate) fn partial_insertion_sort(moves: &mut [ExtMove], limit: i32) {
    if moves.is_empty() {
        return;
    }
    // Element 0 anchors the prefix: `sorted_end` is the last sorted index.
    let mut sorted_end = 0;
    for p in 1..moves.len() {
        if moves[p].value >= limit {
            let tmp = moves[p];
            sorted_end += 1;
            // Hole-punch: the element leaving the prefix edge goes to `p`.
            moves[p] = moves[sorted_end];
            let mut q = sorted_end;
            while q > 0 && moves[q - 1].value < tmp.value {
                moves[q] = moves[q - 1];
                q -= 1;
            }
            moves[q] = tmp;
        }
    }
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
    CAPTURE_MVV_FACTOR * victim_value + cap_hist
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
/// fast — doc 09 §3.1). Steps 05/06 use only the `[cur .. end_cur)` emission
/// window over a captures region; the bad-capture/quiet regions and the
/// depth/ply/continuation inputs are populated now but consumed by the main
/// and evasion pipelines in doc 09 §4 steps 7-8.
// The struct deliberately carries the full doc-09 §3.5 field set now so the
// constructor signatures match what the search will call; the fields the
// capture-only pipelines don't yet read (`depth`, `ply`, `skip_quiets`,
// `cont_hist`, and the `end_bad_captures`/`end_captures`/`end_generated`
// region cursors) are wired through by steps 7-8.
#[allow(dead_code)]
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
    buf: [ExtMove; MAX_MOVES],
}

impl MovePicker {
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
            buf: [ExtMove::NULL; MAX_MOVES],
        }
    }

    /// Emit one pseudo-legal move per call, `None` once the active pipeline is
    /// exhausted (design/09 §3.6). The caller checks legality only on the move
    /// it actually plays. Idempotent after exhaustion: further calls keep
    /// returning `None`.
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
                // ProbCut and quiescence share this init verbatim.
                Stage::ProbcutInit | Stage::QcaptureInit => {
                    self.init_captures(board, hist);
                    self.stage = self.stage.bump();
                }

                // Quiescence: every capture, TT move deduped.
                Stage::Qcapture => return self.select(board, |_, _| true),

                // ProbCut: captures clearing the SEE threshold, TT deduped.
                Stage::Probcut => {
                    let threshold = self.threshold;
                    return self.select(board, move |b, m| b.see_ge(m, threshold));
                }

                // Main and evasion pipelines land in doc 09 §4 steps 7-8.
                Stage::CaptureInit
                | Stage::GoodCapture
                | Stage::QuietInit
                | Stage::GoodQuiet
                | Stage::BadCapture
                | Stage::BadQuiet
                | Stage::EvasionInit
                | Stage::Evasion => {
                    todo!("main and evasion pipelines land in doc 09 §4 steps 7-8")
                }
            }
        }
    }

    /// Generate captures into the buffer, score them (§3.7), lay out the
    /// captures region, and full-sort it (design/09 §3.6 item 2). Generation
    /// uses a stack [`MoveList`]; the scored moves are copied into the owned
    /// buffer — no heap allocation.
    fn init_captures(&mut self, board: &Board, hist: &Histories) {
        let mut list = MoveList::new();
        generate_captures(board, &mut list);
        let n = list.len();
        for (i, &m) in list.iter().enumerate() {
            self.buf[i] = ExtMove {
                mv: m,
                value: score_capture(board, hist, m),
            };
        }
        // Captures occupy `[0, n)`; nothing has been swapped into the
        // bad-captures region and no quiets exist yet.
        self.cur = 0;
        self.end_bad_captures = 0;
        self.end_captures = n;
        self.end_generated = n;
        self.end_cur = n;
        partial_insertion_sort(&mut self.buf[..n], i32::MIN);
    }

    /// Scan the emission window `buf[cur..end_cur]` for the next move passing
    /// `ok`, skipping any element equal to the already-emitted TT move
    /// (design/09 §3.6 shared select semantics). Advances `cur` past every
    /// element it inspects.
    fn select(&mut self, board: &Board, mut ok: impl FnMut(&Board, Move) -> bool) -> Option<Move> {
        while self.cur < self.end_cur {
            let m = self.buf[self.cur].mv;
            self.cur += 1;
            // Dedup: the TT move was already emitted at the `*Tt` stage.
            if m != self.tt_move && ok(board, m) {
                return Some(m);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Histories;
    use crate::movegen::generate_quiets;
    use crate::types::Square;
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
        ExtMove {
            mv: Move::from_raw16(idx as u16),
            value,
        }
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
}
