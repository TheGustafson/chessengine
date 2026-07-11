# 09 — Move ordering: MovePicker and history tables

Status: Draft v1

This document specifies the two modules that decide *in what order* the search
considers moves: `movepick` (a lazily staged, one-move-at-a-time generator
driven by an explicit state machine over a single 256-slot `ExtMove` buffer)
and `history` (the full catalogue of statistics tables — butterfly, low-ply,
capture, continuation, pawn, correction, ttMove — plus the single "gravity"
bounded-update operator and the `update_*` functions that write them). Move
ordering is the highest-leverage component of an alpha-beta engine: most nodes
must cut off on the TT move or the first good capture without ever generating
quiets. All tuned constants below are Stockfish-current values adopted
verbatim as starting points (D20); everything is expressed as original Rust
sketches and prose, never copied source (D27).

## 1. Purpose and scope

**This doc owns:**

- `src/movepick.rs`: `ExtMove` (behavior; representation contract is in
  `design/02-core-types.md` §3.8), `MovePicker` with all four stage pipelines
  (main, evasion, probcut, qsearch), `partial_insertion_sort`, the scoring
  formulas for captures/quiets/evasions, `skip_quiet_moves`, and the
  threat-by-lesser-piece helper.
- `src/history.rs`: every history table type with its shape, bound `D`, init
  and per-search decay values; the generic gravity operator `StatEntry<D>`;
  the `Histories` per-thread container; `ContHistId` (the borrow-safe
  replacement for Stockfish's stack-stored continuation-history pointers);
  and the update functions `update_all_stats`, `update_quiet_histories`,
  `update_continuation_histories`, `update_correction_history`, plus the read
  side `correction_value` and `to_corrected_static_eval`.

**This doc does NOT own** (it provides the callees; these docs own the call
sites and *when/with what bonus* they run):

- The search node flow, per-move pruning, and every place `update_*` is
  invoked with which bonus formula: `design/10-pruning-extensions.md`.
- The search skeleton, `Stack` array layout (D18) and its field inventory
  (`current_move`, `stat_score`, `move_count`, `tt_hit`, `in_check`,
  `cont_hist`): `design/07-search.md`. We specify only the *representation*
  of the `cont_hist` handle and its lookup.
- Move generation (`generate_captures/quiets/evasions`, `pseudo_legal`,
  `capture_stage`): `design/05-movegen.md`.
- `see_ge`, `check_squares`, `pawn_key`, the specialized Zobrist keys, and
  `captured_piece`: `design/04-position-state.md`.
- `Value`/`Depth` aliases, `PIECE_VALUES`, `Piece::index()`,
  `Move::raw16()`: `design/02-core-types.md`.
- Where `Histories` lives (`Worker`, per-thread, D23) and `Worker::clear()`
  plumbing: `design/13-threading-smp.md`.

## 2. References

- `stockfish_reference/03-movegen-movepick-history.md` — staging overview,
  gravity rule, table survey, Rust implications (index-based regions).
- `stockfish_reference/stockfish_search_notes.md` §8 (stages, scoring), §9
  (tables, gravity, `update_all_stats`), §10.2 (correction history).
- Verified against SF source (read-only, GPL — ideas and constants only):
  - `Stockfish/src/movepick.{h,cpp}` — stage order, region pointers, select
    dedup, partial insertion sort, scoring formulas, branchless TT skip.
  - `Stockfish/src/history.h` — table shapes, `D` bounds, gravity operator,
    shared-history layout, `CORRECTION_HISTORY_LIMIT`, `LOW_PLY_HISTORY_SIZE`.
  - `Stockfish/src/search.cpp` — `Worker::clear()` fill values, per-`go`
    decay (789/1024) and low-ply refill (100), `update_all_stats`,
    `update_quiet_histories`, `update_continuation_histories`,
    `correction_value`, `update_correction_history`, sentinel conthist setup.
  - `Stockfish/src/position.h` — `capture_stage` semantics (captures OR queen
    promotions), `attacks_by` shape.

## 3. Design

### 3.1 ExtMove and the one buffer

`ExtMove` pairs a move with its ordering score (D7). The representation
contract (8 bytes, `#[repr(C)]`, align 4, size-asserted) is set in doc 02;
it lives here:

```rust
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ExtMove {
    pub mv: Move,     // u16
    pub value: i32,   // ordering score; compared by value only
}
const _: () = assert!(std::mem::size_of::<ExtMove>() == 8);
```

`MoveList` stays 2-bytes-per-move (doc 05); scoring happens only inside the
picker's own buffer. The picker owns exactly one `[ExtMove; MAX_MOVES]`
(256 slots, ~2 KiB) carved into regions by **indices, not pointers** —
borrow-checker friendly and equally fast:

```
[0 .. end_bad_captures)          bad captures (deferred, swapped down here)
[end_bad_captures .. end_captures)   captures (sorted; consumed part is stale)
[end_captures .. end_generated)  quiets (partially sorted)
cur, end_cur                     current emission window inside the above
```

### 3.2 ContHistId — borrow-safe continuation-history plumbing

Stockfish stores raw pointers to inner `PieceToHistory` tables in the search
stack. We store a **compact flat index** instead (coordinating with
`design/07-search.md` and doc 10, which names the field
`StackEntry::cont_hist: ContHistId`):

```rust
/// Identifies one inner piece-to table of the continuation history:
/// (in_check, capture, prior piece incl. a "none" sentinel, prior to-square)
/// flattened to ((in_check*2 + capture)*13 + piece_idx)*64 + to  (0..3328).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ContHistId(u16);

impl ContHistId {
    pub const SENTINEL: ContHistId = ContHistId(768); // (false,false,None,A1)

    pub fn new(in_check: bool, capture: bool, piece: Option<Piece>, to: Square) -> Self {
        let pc = piece.map_or(12, |p| p.index());          // 12 = "no piece"
        let flat = ((in_check as usize * 2 + capture as usize) * 13 + pc) * 64
                 + to.index() as usize;
        ContHistId(flat as u16)
    }

    /// Sub-index (piece13 * 64 + to), shared with the continuation
    /// *correction* history. Derivable because the flat layout puts
    /// (piece, to) in the low `13*64 = 832` positions of each quadrant.
    pub fn piece_to(self) -> usize { self.0 as usize % 832 }
}
```

- **Who writes it:** search (doc 07/10) sets `stack[cur].cont_hist =
  ContHistId::new(node_in_check, board.capture_stage(m), Some(moved_piece),
  m.to())` right after `make_move` (`cur = ply + SS_OFFSET`, the padded stack
  index of doc 07 §3.2/D18 — used for all stack indexing in this doc, see
  §3.9); a null move (and the pre-root padding
  entries, D18) sets `ContHistId::SENTINEL`, mirroring SF's sentinel table
  `[0][0][NO_PIECE][A1]`. The sentinel table is initialized like every other
  (fill −552) and is *read* by scoring but never *written* (writes are gated
  on `current_move.is_ok()`), so it behaves as a constant baseline — same as
  Stockfish.
- **Lookup cost:** one `u16` load from the stack entry (already hot), one
  indexed load into a boxed array `tables[id.0 as usize]`, then `[pc][to]`.
  Versus SF's stored pointer this adds a single add+shift to form the
  address; the `% 832` for the correction sub-index compiles to a
  multiply-high — both negligible. There is no separate `ContCorrId` needed
  in the stack: doc 07/10 can derive it via `piece_to()` (see §9).

### 3.3 The gravity operator — one generic bounded update

Every table uses the same self-saturating update rule. Semantics: clamp the
bonus to `[-D, D]`, then move the entry toward the bonus while decaying it in
proportion to `|bonus|/D`. Entries provably stay in `[-D, D]`; repeated
same-sign bonuses have diminishing effect; stale signal decays exponentially.

```rust
/// A history cell bounded to [-D, D] by the gravity update rule.
#[derive(Clone, Copy)]
pub struct StatEntry<const D: i32>(i16);

impl<const D: i32> StatEntry<D> {
    pub const fn new(v: i16) -> Self { Self(v) }
    pub fn get(self) -> i32 { self.0 as i32 }

    pub fn update(&mut self, bonus: i32) {
        let b = bonus.clamp(-D, D);
        let v = self.0 as i32;
        let next = v + b - v * b.abs() / D;
        debug_assert!(next.abs() <= D);
        self.0 = next as i16;
    }
}
```

All `D` values fit i16 storage (max 30000). This is the ONLY write primitive;
no table is ever assigned except by `fill` at init time.

### 3.4 The table catalogue

All tables are **per-thread, plain (non-atomic) i16** inside
`Worker::histories: Box<Histories>` (D23; current SF shares pawn/correction/
continuation histories per NUMA node behind relaxed atomics — deferred, see
§9). Shapes use OUR indexing: `Piece::index()` 0..12 (D4), `Move::raw16()`
as a 16-bit butterfly index (bit layout differs from SF per D2 — it is a
bijection on u16, so statistically equivalent), victim axis 0..6 where 6 =
"no victim" (en-passant and quiet queen promotions, mirroring SF's
`NO_PIECE_TYPE` slot).

| Table | Rust shape | D | Init (new game) | Per-`go` | Size |
|---|---|---|---|---|---|
| `main` (butterfly) | `[[E; 65536]; 2]` by `[stm][move.raw16()]` | 7183 | fill(−5) | decay: `e = e * 789 / 1024` | 256 KiB |
| `low_ply` | `[[E; 65536]; 5]` by `[ply][raw16]`, plies 0–4 | 7183 | — | fill(100) | 640 KiB |
| `capture` | `[[[E; 7]; 64]; 12]` by `[pc][to][victim_type]` | 10692 | fill(−699) | — | 10.5 KiB |
| `cont` (continuation) | `Box<[PieceToHist; 3328]>`, outer = `ContHistId` (`[in_check][capture][piece13][to]`), inner `PieceToHist = [[E; 64]; 12]` | 30000 | fill(−552) | — | 4.88 MiB |
| `pawn` | `[[[E; 64]; 12]; 8192]` by `[pawn_key & 8191][pc][to]` | 8192 | fill(−1262) | — | 12 MiB |
| `corr` (unified correction) | `[[CorrBundle; 2]; 65536]` by `[key & 65535][stm]`; bundle = {pawn, minor, non_pawn_white, non_pawn_black} entries, each field read/written under its own key | 1024 | fill(−6) | — | 1 MiB |
| `cont_corr` | `Box<[PieceToCorr; 832]>`, outer = `ContHistId::piece_to()`, inner `[[E; 64]; 12]` | 1024 | fill(5) | — | 1.22 MiB |
| `tt_move` | one scalar `StatEntry<8192>` | 8192 | 0 | — | 2 B |
| killers / countermoves | **do not exist** (D17) | — | — | — | — |

Total ≈ 20 MiB per thread. All fill/decay values are SF-current (D20).

```rust
pub struct Histories {
    pub main:      Box<[[StatEntry<MAIN_D>; 65536]; 2]>,
    pub low_ply:   Box<[[StatEntry<MAIN_D>; 65536]; LOW_PLY_SIZE]>, // LOW_PLY_SIZE = 5
    pub capture:   Box<[[[StatEntry<CAPT_D>; VICTIM_NB]; 64]; 12]>, // VICTIM_NB = 7
    pub cont:      Box<[PieceToHist; 3328]>,
    pub pawn:      Box<[[[StatEntry<PAWN_D>; 64]; 12]; PAWN_HIST_SIZE]>, // 8192, pow2
    pub corr:      Box<[[CorrBundle; 2]; CORR_SIZE]>,               // 65536, pow2
    pub cont_corr: Box<[PieceToCorr; 832]>,
    pub tt_move:   StatEntry<8192>,
}

pub type PieceToHist = [[StatEntry<CONT_D>; 64]; 12];   // CONT_D = 30000, 1.5 KiB

impl Histories {
    /// Allocate + clear(). NEVER `Box::new([..; ..])` — multi-MB arrays must
    /// not transit the stack. Not `Box::new_zeroed` either: its `assume_init`
    /// is an `unsafe fn` no D14 clause covers, and zero is not any table's
    /// init state (every fill value above is non-zero). The safe, stable
    /// pattern, per table:
    ///   `vec![ROW_INIT; N].into_boxed_slice().try_into().unwrap()`
    /// (the `Vec` allocates directly on the heap; `TryFrom<Box<[T]>> for
    /// Box<[T; N]>` is stable), where ROW_INIT is the fill-valued row from
    /// the Init column. The outer struct is small (seven pointers + one
    /// scalar), so a plain `Box::new(Histories { .. })` finishes the job.
    pub fn new() -> Box<Self>;
    pub fn clear(&mut self);        // ucinewgame: the "Init" column above
    pub fn new_search(&mut self);   // each `go`: main decay 789/1024, low_ply fill(100)
    pub fn cont_ref(&self, id: ContHistId) -> &PieceToHist { &self.cont[id.0 as usize] }
    pub fn cont_mut(&mut self, id: ContHistId) -> &mut PieceToHist { ... }
    pub fn pawn_slot(&self, key: Key) -> &[[StatEntry<PAWN_D>; 64]; 12] {
        &self.pawn[(key & (PAWN_HIST_SIZE as u64 - 1)) as usize]
    }
}
```

Structural bounds (`MAIN_D = 7183`, `CAPT_D = 10692`, `CONT_D = 30000`,
`PAWN_D = 8192`, `CORR_D = 1024`) are const generics in `history.rs`; all
scoring/update *multipliers* below are named constants in `search/params.rs`
(D20).

Small `moves.rs`/`types.rs` extensions this doc needs (additive, D1):
`Move::raw16(self) -> u16` and `Piece::index(self) -> usize` (doc 02), plus
`Move::is_ok(self) -> bool { self.from() != self.to() }` — true for every
real move, false for `Move::NULL` and any future null-move sentinel (D2).

### 3.5 MovePicker: an explicit state machine

```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
#[repr(u8)]
enum Stage {
    // main search
    MainTt, CaptureInit, GoodCapture, QuietInit, GoodQuiet, BadCapture, BadQuiet,
    // evasions
    EvasionTt, EvasionInit, Evasion,
    // probcut
    ProbcutTt, ProbcutInit, Probcut,
    // quiescence
    QsearchTt, QcaptureInit, Qcapture,
}
// Stage::from_u8 via a match; bump() = from_u8(self as u8 + 1).

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    depth: Depth,
    ply: usize,
    threshold: Value,              // probcut only
    skip_quiets: bool,
    cont_hist: [ContHistId; 6],    // stack[cur-1..=cur-6].cont_hist, copied in
                                   // by search (cur = ply + SS_OFFSET, §3.9)
    cur: usize,
    end_cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,
    buf: [ExtMove; MAX_MOVES],
}
```

The picker **holds no borrows**: `&Board` and `&Histories` are passed into
each call. This is the Rust resolution of SF's `const Position&` member —
search can freely `make_move`/`unmake_move` and update histories between
calls, and the picker is plain owned data on the search function's stack
(~2.1 KiB per ply; doc 07 sizes thread stacks accordingly).

```rust
impl MovePicker {
    /// Main search (depth > 0), qsearch (depth <= 0, DEPTH_QS = 0, D6),
    /// and evasions (board in check) — one constructor, like SF.
    pub fn new(board: &Board, tt_move: Move, depth: Depth, ply: usize,
               cont_hist: [ContHistId; 6]) -> Self;

    /// ProbCut: captures with see_ge(threshold); threshold is
    /// probcut_beta - static_eval, computed by the caller (doc 10).
    pub fn new_probcut(board: &Board, tt_move: Move, threshold: Value) -> Self;

    /// Emits one pseudo-legal move per call; None when exhausted.
    /// The caller checks legality on the move it actually plays.
    pub fn next_move(&mut self, board: &Board, hist: &Histories) -> Option<Move>;

    /// Late-move pruning: abandon quiet generation/emission (bad captures
    /// still come). Idempotent; callable mid-iteration.
    pub fn skip_quiet_moves(&mut self) { self.skip_quiets = true; }
}
```

**Constructor stage selection with the branchless TT skip.** The base stage is
`EvasionTt` if the side to move is in check, else `MainTt` when `depth > 0`,
else `QsearchTt`. A TT move is *usable* iff `tt_move.is_ok() &&
board.pseudo_legal(tt_move)`; ProbCut additionally requires
`board.capture_stage(tt_move)` (captures or queen promotions — doc 05). The
skip is arithmetic, not a branch:

```rust
self.stage = Stage::from_u8(base as u8 + (!usable) as u8);
```

so an unusable TT move lands directly on the `*Init` stage.

### 3.6 next_move: the four pipelines

Emission order overall (main search): **TT move → good captures → good
quiets → bad captures → bad quiets**. Bad captures outrank bad quiets. The
body is a `loop { match self.stage { .. } }` — init stages do their work,
`bump()`, and `continue` (replacing SF's `goto`/fallthrough); emission stages
`return` or bump and continue when exhausted.

**Shared select semantics** (every emission stage): scan `buf[cur..end_cur]`;
skip any element equal to `tt_move` (dedup — the TT move was already emitted
at the `*Tt` stage); apply the stage filter; on success `cur += 1` past the
element and return it.

1. `MainTt | EvasionTt | QsearchTt | ProbcutTt`: `bump(); return
   Some(tt_move)` — **no generation happened at all**. This is the whole
   point of staging: a TT-move cutoff costs zero generation and scoring.

2. `CaptureInit | ProbcutInit | QcaptureInit`: run `generate_captures` into a
   local `MoveList`, copy into `buf[0..n]` while scoring (§3.7), set
   `cur = end_bad_captures = 0`, `end_cur = end_captures = n`, then
   **full sort**: `partial_insertion_sort(&mut buf[..n], i32::MIN)`.
   `bump(); continue`.

3. `GoodCapture`: filter = `board.see_ge(m, -value / 18)` — the good/bad
   split is a *score-dependent* SEE threshold: a capture whose ordering score
   (7×victim + capture history) is high may lose a little material and still
   count as good; a capture with terrible history must win material to
   qualify. On filter failure the element is swapped down into the
   bad-captures region and skipped:

   ```rust
   self.buf.swap(self.end_bad_captures, self.cur);
   self.end_bad_captures += 1;
   self.cur += 1; // the stale (already-emitted) element swapped up is skipped
   ```

   When exhausted: `bump()` and fall into `QuietInit`.

4. `QuietInit`: if `!skip_quiets`: `generate_quiets` into a `MoveList`, score
   quiets (§3.7) into `buf[end_captures..]`, `end_cur = end_generated =
   end_captures + n`, then sort **only the promising prefix**:
   `partial_insertion_sort(&mut buf[end_captures..end_generated],
   -3560 * depth)` — deeper nodes sort more of the list; the tail stays
   unsorted because a cutoff usually arrives within the first few moves.
   `bump(); continue`. (If `skip_quiets`, just bump — quiets are never even
   generated.)

5. `GoodQuiet`: if `!skip_quiets`, select with filter
   `value > GOOD_QUIET_THRESHOLD` where `GOOD_QUIET_THRESHOLD = -14000`.
   Note the scan does not stop at the first sub-threshold element: the
   unsorted tail may still contain values above it (only elements
   ≥ `-3560*depth` were sorted forward). When exhausted, rewind to the
   bad-captures region: `cur = 0; end_cur = end_bad_captures; bump()`.

6. `BadCapture`: select with filter `true` (they were already scored and are
   emitted in swap order). When exhausted, rewind to the quiet region:
   `cur = end_captures; end_cur = end_generated; bump()`.

7. `BadQuiet`: if `skip_quiets`, return `None`. Else select with filter
   `value <= GOOD_QUIET_THRESHOLD` — the exact complement of stage 5, so the
   two passes partition the quiet region with no dedup bookkeeping. `None`
   when exhausted.

**Evasion pipeline**: `EvasionTt` → `EvasionInit` (generate_evasions, score
§3.7, full sort with `i32::MIN`, regions `cur = 0`, `end_cur =
end_generated = n`) → `Evasion` (select all). Double-check positions generate
only king moves (doc 05); legality is still the caller's job.

**ProbCut pipeline**: `ProbcutTt` → `ProbcutInit` (as stage 2) → `Probcut`:
select with filter `board.see_ge(m, self.threshold)`.

**Qsearch pipeline**: `QsearchTt` → `QcaptureInit` (as stage 2) → `Qcapture`:
select all. (In-check qsearch nodes take the evasion pipeline via the
constructor — check evasions include quiets, scored as such.)

### 3.7 Scoring formulas (SF-current constants, D20)

Ordering piece values are the SEE values (D11):
`PIECE_VALUES = [208, 781, 825, 1276, 2538, 0]` for P, N, B, R, Q, K
(doc 02 §3.9; `PieceType::value()`). Victim value for a move is
`board.piece_on(m.to()).map_or(0, |p| p.piece_type.value())` — en passant
and quiet queen promotions score a victim of 0 and use the "no victim"
capture-history slot, exactly like SF's `NO_PIECE` behavior.

**Captures** (`CaptureInit`, `ProbcutInit`, `QcaptureInit`):

```
value = 7 * victim_value + capture[pc][to][victim_slot]
```

Pure MVV plus history — deliberately no attacker (LVA) term; SEE handles
attacker risk via the `-value/18` gate.

**Quiets** (`QuietInit`), with `pc = moved piece`, `pt = pc.piece_type`,
`us = side to move`:

```
value = 2 * main[us][m.raw16()]
      + 2 * pawn[pawn_key & 8191][pc][to]
      + cont(1)[pc][to] + cont(2)[pc][to] + cont(3)[pc][to]
      + cont(4)[pc][to] + cont(6)[pc][to]                 // offsets 1,2,3,4,6
      + 16384 * (to ∈ check_squares(pt) && see_ge(m, -75)) // check bonus
      + PIECE_VALUES[pt] * 20 * ((from ∈ threat[pt]) - (to ∈ threat[pt]))
      + if ply < 5 { 8 * low_ply[ply][m.raw16()] / (1 + ply) } else { 0 }
```

where `cont(k)` is `hist.cont_ref(self.cont_hist[k-1])` — resolved **once**
per scoring pass into five locals, then five array reads per move. Offset 5
is *not* read for ordering (it is only written by updates, weight 129).

The **threat escape/entry term** uses `threat[pt]` = squares attacked by
enemy pieces *strictly cheaper* than `pt`, computed once per scoring pass:

```rust
// threats_by_lesser(board, us): [Bitboard; 6] indexed by our PieceType
threat[Pawn]   = EMPTY;                 threat[King] = EMPTY;
threat[Knight] = threat[Bishop] = enemy pawn attacks (bulk shifts);
threat[Rook]   = enemy knight ∪ bishop attacks ∪ threat[Knight];
threat[Queen]  = enemy rook attacks ∪ threat[Rook];
```

Fleeing a square attacked by a cheaper piece earns `+20 * PIECE_VALUES[pt]`
(a queen escaping a rook attack gets +50760, dwarfing history); stepping onto
one costs the same, burying the move below even bad quiets. The helper lives
in `movepick.rs` as a free function over `Board` + the attacks module
(doc 03); each enemy attack union is a pop-lsb loop over that piece type
against `all_occupancy` (pawns via bulk shifts).

**Evasions** (`EvasionInit`):

```
if board.capture_stage(m): value = victim_value + (1 << 28)
else:                      value = main[us][m.raw16()] + cont(1)[pc][to]
```

The `1 << 28` constant lifts every capturing evasion above every quiet
evasion.

### 3.8 partial_insertion_sort

Descending, in place, allocation-free, with a limit: after the pass, the
prefix `[0..=sorted_end]` is sorted descending and contains the original
element 0 **plus every element with `value >= limit`**; the tail is
unsorted and (except for that quirk about element 0) entirely below the
limit. Sorting work scales with what the search actually consumes. Do NOT
substitute `sort_unstable_by_key` — full sorts of 40-move quiet lists at
shallow depth are pure waste.

```rust
pub(crate) fn partial_insertion_sort(moves: &mut [ExtMove], limit: i32) {
    if moves.is_empty() { return; }
    let mut sorted_end = 0;                    // element 0 anchors the prefix
    for p in 1..moves.len() {
        if moves[p].value >= limit {
            let tmp = moves[p];
            sorted_end += 1;
            moves[p] = moves[sorted_end];      // hole-punch: move prefix edge out
            let mut q = sorted_end;
            while q > 0 && moves[q - 1].value < tmp.value {
                moves[q] = moves[q - 1];
                q -= 1;
            }
            moves[q] = tmp;
        }
    }
}
```

Call sites: captures and evasions with `limit = i32::MIN` (full sort); quiets
with `limit = -3560 * depth`. SF's AVX-512 16-lane variant is out of scope
(see §9).

### 3.9 Update functions

All bonuses flow through the gravity operator (§3.3). The *call sites and
bonus formulas at those sites* belong to doc 10; the three functions below —
their internals, weights, and offsets — belong here. Signatures take the
**full padded stack slice** plus the **padded index** `cur = ply + SS_OFFSET`
(doc 07 §3.2, D18) — never the raw search ply: every `stack[cur - i]` below
reaches back up to 7 slots (the refutation malus walks offsets 1..=6 from
`cur - 1`), and only the padded index makes those reads in-bounds by
construction at the root (`cur - 7 >= 0` exactly; with raw ply, `ply - i`
underflows `usize` and panics). Each function opens with
`debug_assert!(cur >= SS_OFFSET)`. Passing keys rather than references works
because `ContHistId` is `Copy`: reading keys from the stack and mutating
`Histories` never conflicts.

**`update_continuation_histories(stack, cur, hist, pc, to, bonus)`** — walks
the predecessor chain and rewards/punishes "move `pc`→`to` after that prior
move". `cur` is the padded index of the node being updated; the deepest read
is `stack[cur - 6]` (or `cur - 7` when called with `cur - 1` for the
refutation malus), which bottoms out at index 0 at the root — safe by D18.
Offsets and weights (divisor 131072):

```
offsets = [(1, 1040), (2, 780), (3, 300), (4, 537), (5, 129), (6, 423)]
consistency multipliers CMHC = [96, 113, 101, 105, 127, 121, 126]
```

Algorithm: `positive_count = 0`; for each `(i, weight)` in order:
- if `stack[cur].in_check && i > 2`: stop (in check, only offsets 1–2 carry
  signal — the position before the check bears little relation).
- if `!stack[cur - i].current_move.is_ok()`: skip (null move / padding).
- else with `e = &mut hist.cont_mut(stack[cur - i].cont_hist)[pc][to]`:
  if `e.get() > 0 { positive_count += 1; }` then
  `e.update(bonus * weight * CMHC[positive_count] / 131072
            + if i == 1 { 71 } else { 0 })`.

The consistency multiplier grows with the count of already-positive entries
along the chain (an agreeing chain amplifies the update); offset 1 always
gets the flat +71 (even for maluses).

**`update_quiet_histories(board, stack, cur, hist, m, bonus)`** — one quiet
move, all quiet tables, with per-table weights (the low-ply table is indexed
by the raw search ply, recovered as `ply = cur - SS_OFFSET`):

```
main[us][m.raw16()]           .update(bonus)                       // ×1
if ply < 5:                                  // ply = cur - SS_OFFSET
    low_ply[ply][m.raw16()]   .update(bonus * 663 / 1024)
update_continuation_histories(stack, cur, hist, pc, to, bonus * 820 / 1024)
pawn[pawn_key & 8191][pc][to] .update(bonus * w / 1024)
    where w = 1038 if bonus > -7 else 525    // bonuses near-full, maluses halved
```

(The pawn-history asymmetry threshold is exactly `bonus > -7`, not `> 0`.)

**`update_all_stats(board, stack, cur, hist, best_move, prev_sq,
quiets_searched, captures_searched, depth, tt_move, is_pv)`** — the main
write burst when a node found a best move that raised alpha (doc 10 step
"node result"). `quiets_searched`/`captures_searched` are the caller's
32-capacity lists of tried non-best moves (`&[Move]`).

```
bonus = min(134 * depth - 79, 1572)
      + 382 * (best_move == tt_move) as i32
      + stack[cur - 1].stat_score / 30
malus = min(1005 * depth - 205, 2218)

if !is_pv:   // breadth multiplier: wide nodes taught us more
    bonus += (bonus as i64 * (quiets_searched.len() + captures_searched.len()) as i64
              / 256) as i32
```

(SF computes the breadth term through a u64 wrap that floor-divides negative
bonuses; we define ours as plain i64 truncating division — see §9.)

- **Best move is quiet** (`!board.capture_stage(best_move)`):
  `update_quiet_histories(best_move, bonus * 824 / 1024)`, then a
  **geometrically decaying malus** over the tried quiets, earliest failures
  punished hardest:

  ```rust
  let mut m = malus * 1136 / 1024;
  for &q in quiets_searched {
      m = m * 956 / 1024;                       // decay BEFORE each use
      update_quiet_histories(.., q, -m);
  }
  ```

- **Best move is a capture**: `capture[pc][to][victim_slot].update(bonus *
  1366 / 1024)`.

- **Refutation malus for the parent's first-try quiet**: if `prev_sq` is
  `Some`, the parent tried only its first candidate
  (`stack[cur - 1].move_count == 1 + stack[cur - 1].tt_hit as i32`), and the
  parent's move was quiet (`board.captured_piece().is_none()`):
  `update_continuation_histories(stack, cur - 1, hist,
  board.piece_on(prev_sq).unwrap(), prev_sq, -malus * 683 / 1024)` — this is
  the `cur - 7` deepest read noted above, in-bounds by D18.

- **All tried non-best captures**: for each,
  `capture[moved][to][victim_slot].update(-malus * 1518 / 1024)`.

**External write sites** (constants recorded here for grep-ability; doc 10
owns when they fire): quiet TT-move fail-high bonus `min(114*depth, 724)`
via `update_quiet_histories`; prior-ply early-quiet penalty `-2187` via
`update_continuation_histories(cur-1, ..)`; post-LMR deeper-research
conthist bonus `+1415`; fail-high/fail-low parent nudges
(`main[them] << scaled * 234/32768`, scaled conthist via `bonus_scale`);
eval-diff nudges (`main << 10 * eval_diff`, `pawn << 13 * eval_diff`);
`tt_move` scalar: `.update(if best == tt { 792 } else { -779 })` at node
result, `.update(-442 - 108 * depth)` on the singular-search path.

### 3.10 Correction history (tables + read/write internals)

Owned here per doc 10 §1; *when* to read/write and the raw-vs-corrected eval
discipline (raw in TT, corrected in pruning) are docs 10/11 (D22). All
entries `StatEntry<1024>`.

- **Unified table**: `corr[slot][stm]` where each slot holds a 4-field
  bundle; each field is addressed under its *own* key into the same array:
  `.pawn` at `pawn_key & 65535`, `.minor` at `minor_key & 65535`,
  `.non_pawn_white` at `non_pawn_key(White) & 65535`, `.non_pawn_black` at
  `non_pawn_key(Black) & 65535` (keys from doc 04, D9). Fill −6.
- **Continuation correction**: `cont_corr[piece_to][pc][to]` — outer index is
  `stack[cur - k].cont_hist.piece_to()` for `k ∈ {2, 4}` (`cur` = padded
  stack index, §3.9), inner indexed by the piece now sitting on the *previous
  move's* to-square. Fill 5.

```rust
/// Weighted correction of the raw static eval, in 1/131072 pawn units.
/// `cur` is the PADDED stack index (`ply + SS_OFFSET`, §3.9/D18): the
/// `cur - 4` read below is why raw ply must never be passed here.
pub fn correction_value(board: &Board, stack: &[StackEntry], cur: usize,
                        hist: &Histories) -> i32 {
    debug_assert!(cur >= SS_OFFSET);
    let us = board.side_to_move();
    let pcv  = corr_field!(pawn,  board.pawn_key());
    let micv = corr_field!(minor, board.minor_key());
    let wnp  = corr_field!(non_pawn_white, board.non_pawn_key(Color::White));
    let bnp  = corr_field!(non_pawn_black, board.non_pawn_key(Color::Black));
    let prev = stack[cur - 1].current_move;
    let cntcv = if prev.is_ok() {
        let to = prev.to();
        let pc = board.piece_on(to).unwrap();   // the piece that just landed
        8363 * (hist.cont_corr[stack[cur - 2].cont_hist.piece_to()][pc][to].get()
              + hist.cont_corr[stack[cur - 4].cont_hist.piece_to()][pc][to].get())
    } else {
        64549                                   // flat prior after a null move
    };
    13345 * pcv + 9280 * micv + 11840 * (wnp + bnp) + cntcv
}

/// corrected = clamp(raw + cv / 131072, just inside the TB-loss..TB-win band)
pub fn to_corrected_static_eval(raw: Value, cv: i32) -> Value;
```

**`update_correction_history(board, stack, cur, hist, bonus)`** (caller
computes `bonus` — doc 10 §3.10 step 9 uses `clamp((best - static_eval)
* depth * (12 if best_move else 18) / 128, -256, 256)` with an outer
`* 1114 / 1024`):

```
corr.pawn            .update(bonus)               // ×1
corr.minor           .update(bonus * 152 / 128)
corr.non_pawn_white  .update(bonus * 186 / 128)
corr.non_pawn_black  .update(bonus * 186 / 128)
if stack[cur - 1].current_move.is_ok():
    cont_corr(cur-2)[pc][to].update(bonus * 136 / 128)
    cont_corr(cur-4)[pc][to].update(bonus *  68 / 128)
```

### 3.11 Memory and threading placement

Everything in this doc is per-thread inside `Worker` (D23): ~20 MiB/thread,
zero synchronization, plain i16. `Worker::clear()` calls `Histories::clear()`
(new game); the iterative-deepening entry calls `new_search()` (per `go`).
Current SF shares pawn/correction/continuation histories per NUMA node with
relaxed-atomic entries and thread-count-scaled sizes; that is an SMP-phase
option we explicitly defer — the type seam (`StatEntry` could grow a
`Shared` atomic variant) is the only forward-compatibility we keep.

## 4. Implementation plan

Each step is one commit; `cargo test` + `cargo clippy --all-targets` clean at
every step.

1. **Core accessors**: `Move::raw16()`, `Move::is_ok()`, `Piece::index()`
   (coordinate with doc 02 — land whichever isn't already in).
   *Done when:* unit tests cover NULL/real moves and all 12 piece indices.
2. **`history.rs`: `StatEntry<D>` + gravity.** *Done when:* bound invariant,
   saturation-toward-±D, zero-bonus no-op, and clamp tests pass (LCG-driven
   random sequences, no dev-deps needed).
3. **`history.rs`: table types + `Histories`** with heap construction helper
   (no stack transit), `clear()`, `new_search()`, size `const` asserts.
   *Done when:* fill values verified by spot reads; `new_search` decay
   verified (e.g. 1024 → 789).
4. **`ContHistId`**: constructor, `SENTINEL`, `piece_to()`. *Done when:*
   round-trip and sentinel tests pass; `piece_to` cross-checked against the
   flat formula for all 3328 ids.
5. **`movepick.rs`: `ExtMove` + `partial_insertion_sort`.** *Done when:*
   randomized tests confirm the exact prefix/limit contract of §3.8 and
   multiset preservation.
6. **MovePicker: qsearch + probcut pipelines** (constructor with branchless
   TT skip, capture generation/scoring, select with dedup). *Done when:*
   emission tests on a FEN suite match `generate_captures` multisets, TT
   dedup verified, probcut SEE filter verified against `see_ge`.
7. **Main pipeline**: quiet scoring (threat helper, check bonus, low-ply),
   good/bad capture split with region swaps, good/bad quiet split,
   `skip_quiet_moves`. *Done when:* full emission-completeness and
   stage-order tests pass (§5).
8. **Evasion pipeline** + evasion scoring. *Done when:* in-check FEN tests
   pass; the four-mode property sweep over the perft suite is green.
9. **Update functions**: `update_continuation_histories`,
   `update_quiet_histories`, `update_all_stats` with scripted-stack unit
   tests (hand-computed gravity results, in-check offset limit, consistency
   multiplier, geometric malus sequence).
10. **Correction history**: tables + `correction_value` +
    `to_corrected_static_eval` + `update_correction_history`, unit-tested
    with synthetic keys. (Consumed later by docs 10/11 wiring.)
11. **Search integration** (joint with docs 07/10): pass `cont_hist` keys,
    call `update_all_stats` at node result, report the new bench signature
    (D24), SPRT the ordering waves (D25).

## 5. Testing and validation

- **Gravity unit tests** (step 2): for 10⁵ random (entry, bonus) pairs the
  post-update value stays in `[-D, D]`; monotone approach to ±D under
  repeated max bonuses; decay property `update(0)` is identity.
- **Partial sort contract** (step 5): for random arrays and limits, the
  prefix is descending, contains element 0 plus exactly the ≥-limit
  elements, and the whole slice is a permutation of the input. Include
  `i32::MIN` (full sort equals a reference descending sort ignoring ties)
  and `limit > max` (only element 0 in prefix).
- **Emission completeness (the key MovePicker oracle):** for every position
  in the perft FEN suite (doc 05) and each pipeline: collect all
  `next_move()` output; assert it equals the corresponding generator's
  multiset (captures+quiets when not in check; evasions in check; captures
  for qsearch/probcut-with-−∞-threshold), with the usable TT move emitted
  exactly once and first. Run once with freshly-initialized histories and
  once with LCG-randomized history values (ordering must never change *what*
  is emitted, only the order).
- **Stage-order assertions** on crafted positions: every good capture
  precedes every quiet; every bad capture precedes every bad quiet; a capture
  with hand-poked terrible capture-history flips from good to bad via the
  `-value/18` gate; the 16384 check bonus reorders a checking quiet to the
  front; a queen stepping onto a rook-attacked square sinks below quiets.
- **skip_quiet_moves**: called before `QuietInit` → no quiets ever generated
  (assert via emission set); called mid-`GoodQuiet` → no further quiets, bad
  captures still emitted.
- **Update-function golden tests** (step 9): scripted stack (fixed
  `cont_hist` ids, `in_check` flags, `current_move`s) + tiny board; assert
  exact post-update i16 values computed by hand for: best-quiet path (824/1024
  weight, pawn 1038 vs 525 asymmetry, low-ply 663 at ply<5 only),
  geometric malus over 3 quiets, in-check truncation to offsets 1–2,
  consistency multipliers with pre-seeded positive entries, refutation-malus
  gating on `move_count == 1 + tt_hit`.
- **Perft is untouched** by this component (no movegen changes), but the
  suite still runs per D24. Once search exists, **every ordering change is
  functional**: it must be SPRT-tested (D25) and must report its new bench
  node count; "non-functional" refactors here (e.g. index-type changes) must
  leave the bench signature bit-identical (D24).

## 6. Performance notes

- **Targets:** picker + history reads should stay in the low single-digit
  percent of search time (SF-class). The staging property to preserve above
  all: a TT-move cutoff performs zero generation, zero scoring, zero sorting.
- **No allocation in `next_move`** — generation uses a stack `MoveList`,
  scoring writes into the owned buffer. Watch that `MovePicker` is
  constructed in place per node and never moved/copied (2 KiB memcpy).
- **Buffer init cost:** `[ExtMove; 256]` safe initialization zeroes 2 KiB per
  node. Ship safe first; if profiling shows it, switch to
  `MaybeUninit`-backed storage with a `// SAFETY:` note and the measurement
  recorded (D14c) — region indices already guarantee only written slots are
  read.
- **Resolve conthist refs once per scoring pass** (5 locals), not per move;
  same for the threat bitboards and the pawn-history slot (one hash slot per
  position, reused for every quiet).
- **Bounds checks:** fixed-size arrays + `u16` ids let LLVM elide most
  checks; only reach for `get_unchecked` after a profile shows a specific
  hot check (D14).
- **Division quirks:** `-value / 18` and `8 * lp / (1 + ply)` are Rust
  truncating divisions — same as C++ — keep them as written; `% 832` is a
  constant modulus (compiles to multiply-shift).
- **What to measure:** bench NPS and node count before/after each wave;
  `perf` attribution of `next_move` + `score_*` + `partial_insertion_sort`;
  first-move-cutoff rate and average moves emitted per node (instrumented
  under a debug feature flag) — the partial-sort limit and staging should
  keep "moves scored but never emitted" low; L1/L2 miss rate delta when the
  12 MiB pawn history lands (hash-indexed tables are miss-prone by design —
  verify they pay for themselves via SPRT, not vibes).

## 7. Interactions

- `design/02-core-types.md` — `ExtMove` representation contract,
  `Piece::index`, `Move::raw16`, `PIECE_VALUES`/`PieceType::value`,
  `Value`/`Depth` (D5/D6/D7).
- `design/03-attack-tables.md` — attack primitives used by
  `threats_by_lesser` and by `check_squares` upstream.
- `design/04-position-state.md` — `see_ge` (D11), `check_squares`,
  `pawn_key`/`minor_key`/`non_pawn_key` (D9), `captured_piece`,
  `piece_on`.
- `design/05-movegen.md` — `generate_captures/quiets/evasions`,
  `pseudo_legal`, `capture_stage`, `MoveList`; the picker emits pseudo-legal
  and search filters legality lazily.
- `design/07-search.md` — `Stack` layout (D18) and fields consumed here
  (`current_move`, `cont_hist`, `in_check`, `move_count`, `tt_hit`,
  `stat_score`); sets `cont_hist` ids in make/unmake flow; passes the
  6-element key array to `MovePicker::new`; thread stack sizing for the
  per-ply 2 KiB picker.
- `design/08-transposition-table.md` — source of `tt_move` (validated
  pseudo-legal before staging, D19).
- `design/10-pruning-extensions.md` — owns every call site and bonus formula
  feeding §3.9/§3.10 (`update_all_stats` at node result, TT-hit bonuses,
  LMR/statScore interplay, `skip_quiet_moves` on LMP, probcut threshold,
  correction bonus computation).
- `design/11-evaluation.md` — correction history boundary (D22): raw eval in
  TT, correction applied in search via `correction_value` defined here.
- `design/13-threading-smp.md` — `Worker.histories: Box<Histories>`
  placement, `Worker::clear()`; per-thread histories (D23).
- `design/14-performance-testing.md` — bench signature discipline (D24) and
  SPRT harness (D25) used by §5/§8.

## 8. Acceptance criteria (D26)

1. `cargo test` green including: gravity bound/saturation suite, partial-sort
   contract suite, the four-pipeline emission-completeness sweep over the
   full perft FEN suite (with and without randomized histories), stage-order
   assertions, skip-quiets semantics, and the golden update-function tests
   with hand-computed values. `cargo clippy --all-targets` clean.
2. Compile-time asserts hold: `ExtMove` is 8 bytes/align 4; `Histories`
   totals ≤ 21 MiB; all `D` bounds and fill values match §3.4's table
   (spot-checked by tests).
3. The TT move, when usable, is always emitted first and never twice;
   verified across the sweep in (1).
4. No heap allocation on any `next_move` path (reviewed; generation uses
   stack `MoveList`).
5. After integration (docs 07/10): the engine with staged MovePicker +
   history ordering beats the same engine with naive MVV-LVA-only ordering
   in an SPRT run at fixed depth-and-time settings (D25), and the bench
   node count (D24) is recorded in the integration commit message.
6. Instrumented ordering quality on the bench suite reports a first-move
   cutoff rate and it is tracked (baseline recorded; no fixed threshold —
   the SPRT in (5) is the strength gate).

## 9. Open questions

1. **Cross-reference filename:** sibling docs refer to this document as
   `design/09-movepick-history.md`; it is written as
   `design/09-move-ordering-history.md` per the commissioning instruction.
   One rename (or a sweep of the ~10 references) is needed at suite
   assembly; content is unaffected.
2. **`ContCorrId` stack field:** RESOLVED (P7 froze `Stack`,
   `src/search/stack.rs`) — no separate `cont_corr_hist`/`ContCorrId` field;
   the continuation-correction outer index is derived from the `cont_hist`
   key via `ContHistId::piece_to()` at its use sites (§3.2, §3.10). `Stack`
   carries the single `cont_hist: ContHistId` and nothing more.
3. **Breadth-multiplier arithmetic:** SF's `bonus * u64(count) / 256` wraps
   negative bonuses through u64, effectively floor-dividing them; we define
   plain i64 truncating division. Off-by-one on negative bonuses versus SF —
   deliberate, ours is the spec (we cannot bit-match SF benches anyway).
4. **AVX-512 partial-sort variant** (16-lane insertion network): skipped;
   scalar is the baseline. Revisit only if profiling shows the sort matters
   at our NPS (unlikely before NNUE).
5. **NUMA/thread-shared histories:** SF shares pawn/correction/continuation
   tables per NUMA node with relaxed atomics and thread-scaled sizing; we
   ship per-thread (D23) and revisit at the SMP-scaling phase
   (`design/13-threading-smp.md`).
6. **`threats_by_lesser` home:** specified as a `movepick.rs` helper; if
   eval (doc 11) later needs the same threat sets, promote it to a `Board`
   method — coordinate then, not now.
