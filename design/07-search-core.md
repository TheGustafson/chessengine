# 07 — Search core: PVS, qsearch, iterative deepening, aspiration

Status: Reviewed v1

This document specifies the naked search skeleton of the engine: the value
semantics every score obeys, the per-ply search stack and its safe-Rust
indexing scheme, PV storage, root-move bookkeeping, the iterative-deepening
driver with aspiration windows, the shape of the PVS negamax node, and the
complete quiescence search. It deliberately contains *no* pruning, reduction,
or extension heuristics beyond mate-distance pruning and the qsearch move
rules — those live in doc 10 and hang on the frame defined here. Everything is
modeled on current Stockfish (dev tree at commit `eca43a97`, 2026-07-06);
constants are SF-current starting points (D20), expressed as original Rust
sketches and prose, never copied source (D27).

## 1. Purpose and scope

**This doc owns:**

- How search *uses* the `Value`/`Depth` semantics: the ply-relative flow of
  mate scores through the tree, the `value_to_tt` / `value_from_tt`
  conversion boundary discipline, the decisive-score gates, and the draw
  dither (the one value function search itself owns).
- The search `Stack`: fixed array, +7 offset (D18), field list, safe indexing.
- PV storage (per-ply fixed lines) and the PV-update invariant.
- `RootMove` / `RootMoves`: scores, EMAs, effort, selDepth, bound flags,
  stable-sort semantics.
- The iterative-deepening loop: per-iteration bookkeeping, history pre-aging
  hook, `lastBestMove` tracking, forgotten-mate and aborted-loss hygiene,
  `go mate n` handling.
- Aspiration windows: initial delta, centering, fail-low/high window updates,
  widening, `failedHighCnt` depth adjustment, `rootDelta`.
- The PVS node shape: `NodeType` as compile-time markers, runtime `cutNode`,
  `allNode` derivation, the zero-window invariant, the ordered node skeleton
  with explicit slots where doc 10 heuristics plug in.
- Qsearch, complete: TT interaction at `DEPTH_QS`, stand pat with softening,
  `futilityBase`, all per-move rules, evasion guard, narrow stalemate test.
- Mate-distance pruning, selDepth, node counting, stop-flag cadence and the
  untrusted-return-on-stop contract.

**This doc does NOT own** (pointers to owners):

- The `Value`/`Depth` layer itself — constants, band predicates,
  `mate_in`/`mated_in`, `value_to_tt`/`value_from_tt` — → doc 02 (core
  types, module `types`), where it is defined and unit-tested with no TT
  dependency. §3.1 restates only what search consumes.
- Pruning/reduction/extension heuristics (razoring, RFP, NMP, IIR, ProbCut,
  LMR/LMP, futility, SEE pruning, singular extensions, hindsight adjustments,
  main-search value softening details) → doc 10.
- MovePicker staging, all history tables, the gravity update, continuation
  history layout → doc 09. This doc only defines the *reference* the stack
  stores (§3.2.3).
- TT entry layout, probe/store API, replacement, aging → doc 08 (per D19).
  This doc defines only what search does with probed data.
- Static evaluation, correction history, optimism → the eval docs (D21/D22).
  Search calls one seam: `evaluate(&Board, ...) -> Value`.
- Time management (optimum/maximum, soft-stop factors, `check_time`
  internals, `searchAgainCounter` policy) → the time-management doc. This doc
  fixes only the *call sites* and cadence.
- Threads/lazy SMP, UCI protocol → their own docs. Everything here is written
  single-threaded but SMP-shaped (D23): per-`Worker` state, `&AtomicBool`
  stop flag, no globals.

## 2. References

- `design/02-core-types.md` §3.2 (owner of the `Value` layer:
  constants, predicates, `mate_in`/`mated_in`, `value_to_tt`/`value_from_tt`).
- `stockfish_reference/stockfish_search_notes.md` §§1–6 (big picture, value
  semantics, iterative deepening + aspiration, node walk-through, reduction
  function, qsearch) and §14 (implementation order).
- `stockfish_reference/04-tt-search-structures.md` (Stack/RootMove layout,
  mate-score TT adjustment, threading data layout).
- Verified against SF source (read-only): `Stockfish/src/search.cpp` (ID loop
  266–628, node 701–1587, qsearch 1596–1828, `value_to_tt`/`value_from_tt`
  1852–1895, `check_time` 2035–2060), `search.h` (PVMoves 60–95, Stack
  104–120, RootMove 126–156), `types.h` (value constants 117–191).
- Decisions: D2, D5, D6, D13, D14, D16, D17, D18, D19, D20, D23, D24, D25,
  D26, D27.

## 3. Design

### 3.1 Value semantics (owned by doc 02, consumed here)

The `Value`/`Depth` layer — constants, band predicates, `mate_in`/`mated_in`,
and the TT conversion pair — is **owned and unit-tested by doc 02**
(core types, §3.2, module `types`), with no TT dependency. This section
restates what search leans on and fixes how search *uses* it; the one
search-owned piece is the draw dither at the end. Constants (SF-current,
restated from doc 02):

```rust
// types (doc 02) — restated for reference
pub type Value = i32;
pub type Depth = i32;

pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_MATE: Value = 32000;
pub const VALUE_INFINITE: Value = 32001;
pub const VALUE_NONE: Value = 32002;

pub const MAX_PLY: usize = 246;

pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value;   // 31754
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;
pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;                    // 31753
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY as Value;   // 31507
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;

pub const DEPTH_QS: Depth = 0;
pub const DEPTH_UNSEARCHED: Depth = -2; // eval-only TT writes
pub const DEPTH_NONE: Depth = -3;
```

Predicates as free `const fn`s (D5), defined in doc 02: `is_valid`, `is_win`,
`is_loss`, `is_decisive`, `is_mate`, `is_mated`. Every heuristic in docs
07/10 gates on these — "never prune/soften/dither a decisive score". Mate
constructors take `ply: i32` per D6 (`ply` is `usize` where it indexes
arrays, `i32` in value arithmetic; search casts at the call site, e.g.
`mate_in(ply as i32)`):

```rust
// types (doc 02) — restated for reference
pub const fn mate_in(ply: i32) -> Value  { VALUE_MATE - ply }
pub const fn mated_in(ply: i32) -> Value { -VALUE_MATE + ply }
```

**Mate scores are ply-relative from day one** (D5): inside the tree a mate
score means "plies from the *root*". The TT must store them relative to the
*node* so an entry transposes correctly. Conversion happens at exactly one
boundary — every TT store goes through `value_to_tt`, every probe through
`value_from_tt` — even in the pre-TT skeleton (the functions exist, in doc
02's `types`, and are unit-tested there before the TT does).

```rust
// types (doc 02 §3.2) — owns the definitions; signatures restated:
pub const fn value_to_tt(v: Value, ply: i32) -> Value;
pub const fn value_from_tt(v: Value, ply: i32, rule50: i32) -> Value;
```

`value_to_tt` shifts win scores up and loss scores down by `ply`
(root-relative → node-relative). `value_from_tt` inverts it and adds
50-move-rule hygiene: a decisive score that cannot be realized before the
halfmove clock expires is downgraded to the edge of the TB band
(`VALUE_TB_WIN_IN_MAX_PLY - 1` / `VALUE_TB_LOSS_IN_MAX_PLY + 1` — the
graph-history interaction guard). Search call sites cast:
`value_from_tt(v, ply as i32, board.rule50() as i32)`.

**Draw dither** (search-owned; doc 02 explicitly defers it here because it
reads search state). Draw-ish returns are never exactly `VALUE_DRAW`; they carry
a ±1 pseudo-random component from the node counter so repetition lines are
not all mutually equal ("3-fold blindness" damping):

```rust
/// VALUE_DRAW - 1 + (nodes & 0x2): evaluates to -1 or +1.
pub fn value_draw(nodes: u64) -> Value { VALUE_DRAW - 1 + (nodes & 0x2) as Value }
```

Used for: draw detection and stop-abort returns in the main search, and the
upcoming-repetition alpha raise in both searches. Note one asymmetry copied
from SF deliberately: qsearch's *ordinary* draw/max-ply return uses plain
`VALUE_DRAW`, not the dither; only its upcoming-repetition path dithers.

### 3.2 The search stack (D18)

#### 3.2.1 Layout and offset

One fixed array per `Worker`, sized so that reads from `ss-7` through `ss+2`
never leave the array, with the search operating at offset +7:

- `ss-7 … ss-1` padding below: continuation-history updates for the move at
  `ss-1` read back to `ss-6`; one more slot of slack matches SF.
- `ss+2` above: each node zeroes its grandchild's `cutoff_cnt` on entry.

```rust
pub const SS_OFFSET: usize = 7;
pub const STACK_SIZE: usize = MAX_PLY + 10;   // 256

/// Index of the stack slot for a given ply.
#[inline]
pub const fn ss(ply: usize) -> usize { ply + SS_OFFSET }
```

`ply` is `usize` where it indexes, `i32` in value arithmetic (D6). Bounds:
`ply <= MAX_PLY` ⇒ `ss(ply)+2 <= 255 < STACK_SIZE` and `ss(ply)-7 >= 0`.

#### 3.2.2 Safe indexing pattern

We do **not** mimic SF's `Stack*` pointer. The search functions receive `ply`
and index the worker-owned array directly; each access is a short-lived
borrow, so the borrow checker is happy and every access is bounds-checked
(cheap; `get_unchecked` only after a profile justifies it, per D14):

```rust
impl Worker {
    // read a relative slot; `off` in -7..=+2
    #[inline]
    fn ss_at(&self, ply: usize, off: isize) -> &Stack {
        debug_assert!((-7..=2).contains(&off));
        &self.stack[ss(ply).checked_add_signed(off).unwrap()]
    }
    #[inline]
    fn ss_at_mut(&mut self, ply: usize, off: isize) -> &mut Stack {
        debug_assert!((-7..=2).contains(&off));
        &mut self.stack[ss(ply).checked_add_signed(off).unwrap()]
    }
}
```

Inside `search`/`qsearch` the common style is direct arithmetic on the local
index, which optimizes identically:

```rust
let cur = ss(ply);
self.stack[cur].in_check = in_check;
let prev_move = self.stack[cur - 1].current_move;
self.stack[cur + 2].cutoff_cnt = 0;
```

No `SsView` struct holding `&mut` slices: a long-lived mutable view would
fight the borrow checker the moment we also touch `self.board`. Plain index
arithmetic on a plain array is the Rust-safe expression of SF's pointer trick.

#### 3.2.3 Stack fields

```rust
#[derive(Clone, Copy)]
pub struct Stack {
    pub ply: i32,
    pub current_move: Move,       // Move::NULL when none (D2)
    pub excluded_move: Move,      // singular-search exclusion (doc 10 writes)
    pub static_eval: Value,       // corrected eval; VALUE_NONE when unset
    pub stat_score: i32,          // move's history score (doc 10 consumes)
    pub move_count: i32,
    pub in_check: bool,
    pub tt_pv: bool,              // sticky "was ever on the PV" (stored to TT)
    pub tt_hit: bool,
    pub cutoff_cnt: i32,          // fail-highs seen at this ply (LMR input, doc 10)
    pub reduction: i32,           // LMR reduction applied to the child, 1024ths
                                  // of a ply (doc 10 writes, child reads back)
    pub follow_pv: bool,          // "on last iteration's PV path" (doc 10 gates)
    pub cont_hist: ContHistId,    // continuation-history key, see below
}
```

There is deliberately **no** separate `cont_corr_hist` field: the
continuation *correction* history's outer index is derived from `cont_hist`
via `ContHistId::piece_to()` at its two use sites (`correction_value`,
`update_correction_history` — eval doc / doc 09 §3.9), settling doc 09's
Open Question 2 in favor of dropping the field.

Notably absent vs SF: a `pv` pointer (§3.3 removes the need) and killers
(D17). `Stack` must be `Copy` and small; target ≤ 40 bytes so the whole stack
is ~10 KB.

**Continuation-history key — index, not pointer.** SF stores a raw
pointer to a `PieceToHistory` slab inside the conthist tensor. A Rust raw
pointer here would poison the whole search with `unsafe`; a `&'a` reference
would freeze the tables. Instead the stack stores a compact **key** that doc
09 resolves in O(1):

```rust
/// Opaque u16 key resolving to one [Piece×Square] slab of the conthist
/// tensor. The encoding (a flat index, NOT bit-packed), the SENTINEL value,
/// and the resolving accessors — including piece_to(), which derives the
/// correction-history sub-index — are doc 09 §3.2's ContHistId.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ContHistId(u16);

impl ContHistId {
    /// The "no piece" slab: installed at ss-7..ss-1 padding slots and after
    /// null moves, so reads through it are valid but inert.
    pub const SENTINEL: ContHistId = /* doc 09 owns the value */;
}
```

The representation and the resolving accessors
(`Histories::cont_ref(&self, id: ContHistId) -> &PieceToHist`,
`ContHistId::piece_to()`) are owned by doc 09 §3.2; this doc fixes the
contract: `Copy`, one-load resolution, a sentinel that is always safe to
read/update, installed by search's `do_move` wrapper keyed by
`(in_check, capture, moved_piece, to)`.

#### 3.2.4 Initialization (per `go`)

At the top of `iterative_deepening`:

- zero the whole array (`Stack` gets a `Default` with `Move::NULL`,
  `VALUE_NONE` static eval where appropriate);
- for the 7 padding slots below the root: `cont_hist = ContHistId::SENTINEL`,
  `static_eval = VALUE_NONE`;
- for `ply in 0..=MAX_PLY + 2`: `stack[ss(ply)].ply = ply as i32`.

### 3.3 PV storage: per-ply fixed lines (decision)

**Options considered.**

1. *SF-style parent-local lines linked by `ss->pv` pointers*: each node owns
   a stack-local `[Move; MAX_PLY+1]` and the child writes through a pointer
   into the parent frame. In Rust this is a lifetime knot (every recursion
   would hold `&mut` into its caller's frame) and it puts ~500 bytes on the
   machine stack per PV ply.
2. *Triangular PV table*: `MAX_PLY(MAX_PLY+1)/2` moves ≈ 61 KB; classic, but
   index arithmetic is fiddly and buys only memory nobody is short of.
3. **Per-ply fixed arrays, worker-owned (chosen)**: `pv_table[ply]` is a
   fixed line of `MAX_PLY+1` moves plus a length. ~122 KB per worker, touched
   only at PV nodes (a rare node class), trivially indexed by the same `ply`
   the stack uses, no aliasing, no lifetimes.

```rust
#[derive(Clone)]
pub struct PvLine {
    moves: [Move; MAX_PLY + 1],
    len: usize,
}

impl PvLine {
    pub fn clear(&mut self) { self.len = 0; }
    pub fn as_slice(&self) -> &[Move] { &self.moves[..self.len] }
    /// self = [first] ++ child
    pub fn load(&mut self, first: Move, child: &PvLine) {
        self.moves[0] = first;
        self.moves[1..=child.len].copy_from_slice(&child.moves[..child.len]);
        self.len = child.len + 1;
    }
}

/// Borrow-splitting update: pv_table[ply] = [m] ++ pv_table[ply + 1].
fn update_pv(pv_table: &mut [PvLine], ply: usize, m: Move) {
    let (head, tail) = pv_table.split_at_mut(ply + 1);
    head[ply].load(m, &tail[0]);
}
```

**Invariants** (replacing SF's `ss->pv` pointer discipline):

- Only PV/Root nodes touch `pv_table` — statically guaranteed because the
  code is gated on `NT::PV`, which monomorphizes away in NonPV instances.
- A PV node clears `pv_table[ply]` on entry (`len = 0`, no memset of the
  array body).
- `update_pv(ply, m)` is called only in the `value > alpha` branch, which in
  PVS always follows a *full-window PV-typed* child search — so
  `pv_table[ply + 1]` is fresh, never stale data from a zero-window child
  (zero-window children are NonPV and never write it).
- The root copies `pv_table[1]` (its child row) into `RootMove::pv`.

### 3.4 RootMove and RootMoves

```rust
pub struct RootMove {
    pub pv: Vec<Move>,                  // pv[0] is the root move; root is cold, Vec is fine
    pub previous_pv: Vec<Move>,
    pub score: Value,                   // -VALUE_INFINITE when unsearched this iteration
    pub previous_score: Value,          // last completed iteration's score
    pub uci_score: Value,               // what we print (clamped to window on bounds)
    pub average_score: Value,           // EMA, init -VALUE_INFINITE
    pub mean_squared_score: i64,        // EMA of value*|value|, init -(VALUE_INFINITE^2)
    pub score_lowerbound: bool,
    pub score_upperbound: bool,
    pub previous_score_exact: bool,
    pub sel_depth: i32,
    pub effort: u64,                    // nodes spent under this root move
    // tb_rank / tb_score: added with Syzygy support later
}

pub type RootMoves = Vec<RootMove>;

impl RootMove {
    pub fn new(m: Move) -> Self { /* pv = vec![m], scores at sentinels */ }
    pub fn score_is_bound(&self) -> bool { self.score_lowerbound || self.score_upperbound }
    pub fn score_is_exact_loss(&self) -> bool {
        self.score != -VALUE_INFINITE && is_loss(self.score) && !self.score_is_bound()
    }
    /// Descending sort key: (score, previous_score).
    pub fn sort_key(&self) -> (Value, Value) { (self.score, self.previous_score) }
}
```

`mean_squared_score` is `i64`: `VALUE_INFINITE²` ≈ 1.024e9 and the EMA's
intermediate sum `value·|value| + prev` reaches ~2.05e9 — inside `i32` range
but within 5% of overflow; SF lives with `int`, we take the free headroom.

**Stable descending sort — semantics matter.** After every aspiration
attempt, only the searched PV move carries a fresh score; all other root
moves were reset to `-VALUE_INFINITE`. A *stable* descending sort therefore
floats the new best to the front while preserving last iteration's order for
everything else — that order is the move ordering for the next iteration.
Rust's `slice::sort_by` is stable; `sort_unstable_by` is a bug here:

```rust
root_moves[pv_idx..pv_last].sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));
```

**Per-move updates inside the root node** (move loop, §3.7), after each child
search returns `value`:

- `effort += nodes_now - nodes_at_move_start`;
- `average_score = if unset { value } else { (value + average_score) / 2 }`;
- `mean_squared_score = if unset { v*|v| } else { (v*|v| + prev) / 2 }` (as i64);
- if `move_count == 1 || value > alpha`: set `score = uci_score = value`,
  `sel_depth`, clear bound flags; if `value >= beta` set `score_lowerbound`
  and `uci_score = beta`; if `value <= alpha` set `score_upperbound` and
  `uci_score = alpha`; rebuild `pv` from move + `pv_table[1]`; if
  `move_count > 1 && pv_idx == 0` increment `best_move_changes` (time-management
  instability signal);
- else: `score = -VALUE_INFINITE` (relies on stable sort as above).

### 3.5 Iterative deepening

Driver: `Worker::iterative_deepening(&mut self)`. Sketch of the loop with
every bookkeeping element this doc owns (time-management policy calls are
marked `TM:` and specified in the time-management doc):

```text
init stack (§3.2.4); pv line for root; root_depth = 0
last_best_move_pv = []; last_best_move_score = -INF; last_best_move_depth = 0
tot_best_move_changes = 0.0; search_again_counter = 0
histories.pre_age()                      # hook, owned by doc 09:
                                         #   low_ply_history.fill(100)
                                         #   main_history *= 789/1024
multi_pv = min(option MultiPV, root_moves.len())

while root_depth + 1 < MAX_PLY
      and !stop
      and !(limits.depth reached          # depth limit: main thread only
            and main_thread):
    root_depth += 1
    tot_best_move_changes /= 2.0          # age the instability metric
    for rm in root_moves:                 # snapshot last iteration
        rm.previous_score = rm.score
        rm.previous_pv    = rm.pv.clone()
        rm.previous_score_exact = (index < multi_pv)

    for pv_idx in 0..multi_pv:            # MultiPV loop (v1 runs it with 1)
        sel_depth = 0
        <aspiration loop, §3.6>
        stable-sort root_moves[..=pv_idx]
        if main_thread and !stop: print "info ... pv" line
        if stop: break

    # --- forgotten-mate + lastBestMove tracking ---
    forgotten_mate =
        last_best_move_score != -INF
        and (is_mate(last_best_move_score) or is_mated(last_best_move_score))
        and (|root_moves[0].score| < |last_best_move_score|
             or root_moves[0].score_is_bound())
    if !stop:
        if last_best_move_pv is empty or last_best_move_pv[0] != root_moves[0].pv[0]:
            last_best_move_depth = root_depth
        if !forgotten_mate:               # never replace a proven (shorter) mate
            last_best_move_pv    = root_moves[0].pv.clone()
            last_best_move_score = root_moves[0].score

    # --- aborted-loss hygiene ---
    aborted_loss = stop and pv_idx == 0 and root_moves[0].score_is_exact_loss()
    if aborted_loss or (root_moves[0].score != -INF and forgotten_mate):
        if !last_best_move_pv.is_empty():
            move root_moves entry matching last_best_move_pv[0] to front
            root_moves[0].score = root_moves[0].uci_score = last_best_move_score
            root_moves[0].pv = last_best_move_pv.clone()
            clear bound flags
        else if aborted_loss:             # aborted depth-1 search: label as bound
            root_moves[0].score_lowerbound = true

    # --- go mate n ---
    if limits.mate != 0 and !stop
       and is_mate_or_mated(root_moves[0].score)
       and VALUE_MATE - |root_moves[0].score| <= 2 * limits.mate:
        stop = true

    if !main_thread: continue
    TM: soft-stop decision, increase_depth / search_again_counter update
```

Rationale for the two hygiene rules:

- **Forgotten mate**: once an iteration proves `mate n`, a later iteration
  returning a weaker or bound score (aspiration instability, pruning noise)
  must not replace the proven line; we roll the front root move back to the
  remembered PV/score.
- **Aborted loss**: an *exact* mated/TB-loss score for the whole root from a
  search that was stopped mid-iteration is untrusted — unsearched root moves
  might refute or delay it. Restore the previous iteration's move/score, or
  (no previous iteration) mark the score as a lower bound. MultiPV adds
  guards so an aborted later line cannot leapfrog a *proven* loss in an
  earlier line (structure the code for it now; full MultiPV behavior lands
  with the UCI doc).

`best_move_changes` accumulates into `tot_best_move_changes` per iteration
and is a time-management input; `last_best_move_depth` (iteration at which
the best move last changed) is the TM stability signal.

### 3.6 Aspiration windows

Per MultiPV line, per iteration (constants SF-current, D20; all named in
`search/params.rs`):

```text
delta = 5 + |root_moves[pv_idx].mean_squared_score| / 10588
        (+ thread_idx % 8 once SMP lands — free lazy-SMP diversity, threads doc)
avg   = root_moves[pv_idx].average_score
alpha = max(avg - delta, -VALUE_INFINITE)
beta  = min(avg + delta,  VALUE_INFINITE)
(eval-doc hook: optimism[us] = 137*avg / (|avg| + 81), optimism[them] = -optimism[us])
failed_high_cnt = 0

loop:
    adjusted_depth = max(1, root_depth - failed_high_cnt
                            - 3 * (search_again_counter + 1) / 4)
    self.root_delta = beta - alpha        # feeds the reduction formula (doc 10)
    best = search::<Root>(alpha, beta, adjusted_depth, cut_node = false, ply = 0)

    stable-sort root_moves[pv_idx..pv_last]
    if stop: break                        # results refer to previous iteration; still sorted safely

    if best <= alpha:                     # FAIL LOW
        beta  = alpha                     # window slides down; old alpha caps it
        alpha = max(best - delta, -VALUE_INFINITE)
        failed_high_cnt = 0
        TM: cancel stop_on_ponderhit      # dropping score revokes early-stop permission
    else if best >= beta:                 # FAIL HIGH
        alpha = max(beta - delta, alpha)  # alpha backs off at most to old beta - delta,
                                          # never below its current value: window stays tight
        beta  = min(best + delta, VALUE_INFINITE)
        failed_high_cnt += 1
    else:
        break                             # score inside the window: done

    delta += 44 * delta / 128             # ~×1.34 per re-search
```

Notes:

- First iterations: `mean_squared_score` initializes to `-VALUE_INFINITE²`,
  so `|mss|/10588 ≈ 96,700` — the window clamps to `(-INF, +INF)` and depth 1
  is effectively an infinite-window search. No special-casing needed.
- `failed_high_cnt` shrinking `adjusted_depth`: each consecutive fail-high
  re-search runs one ply shallower — fail-highs are usually resolved by move
  ordering, not depth, and this caps re-search cost. `search_again_counter`
  is incremented by the TM layer when it said "don't deepen"; default 0 until
  that doc lands.
- `root_delta` (the current root window width) is stored on the `Worker` and
  consumed by doc 10's reduction formula (`- delta * 617 / rootDelta` term).
  It must be set before every root search, including re-searches.

### 3.7 The PVS node

#### 3.7.1 Node types

Compile-time node class + runtime `cutNode`, exactly SF's split (D17). Rust
expression: sealed marker types with associated consts (monomorphizes into
three `search` instances; stable Rust, no `adt_const_params`):

```rust
mod sealed { pub trait Sealed {} }
pub trait NodeType: sealed::Sealed {
    const ROOT: bool;
    const PV: bool;
}
pub enum Root {}   // uninhabited markers
pub enum Pv {}
pub enum NonPv {}
impl NodeType for Root  { const ROOT: bool = true;  const PV: bool = true;  }
impl NodeType for Pv    { const ROOT: bool = false; const PV: bool = true;  }
impl NodeType for NonPv { const ROOT: bool = false; const PV: bool = false; }

impl Worker {
    fn search<NT: NodeType>(&mut self, alpha: Value, beta: Value,
                            depth: Depth, cut_node: bool, ply: usize) -> Value { ... }
    fn qsearch<NT: NodeType>(&mut self, alpha: Value, beta: Value,
                             ply: usize) -> Value { ... }   // never NT = Root
}
```

Invariants, asserted at node entry:

```rust
debug_assert!(!(NT::PV && cut_node));               // PV nodes are never cut nodes
debug_assert!(NT::PV || alpha == beta - 1);         // zero-window invariant for NonPV
debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
let all_node = !NT::PV && !cut_node;                // expected fail-low node
```

`cut_node` flips with each zero-window recursion (`-qsearch`-style negamax
sign discipline: the child of a cut node is an all node and vice versa)
except where doc 10 dictates otherwise (LMR searches are always started with
`cut_node = true`; full-depth zero-window re-searches with `!cut_node`).

#### 3.7.2 Node skeleton, in order

`search::<NT>(alpha, beta, depth, cut_node, ply)`. If `depth <= 0`, dispatch
to qsearch preserving PV-ness:

```rust
if depth <= 0 {
    return if NT::PV { self.qsearch::<Pv>(alpha, beta, ply) }
           else      { self.qsearch::<NonPv>(alpha, beta, ply) };
}
```

Then, with `⟨D10⟩` marking empty slots this doc reserves for doc 10:

1. **Upcoming-repetition early draw** (non-root; needs cuckoo tables — lands
   with that infrastructure, slot reserved): if `alpha < VALUE_DRAW` and the
   position has a provable move-to-repetition, `alpha = value_draw(nodes)`;
   return `alpha` if `alpha >= beta`.
2. **Init**: `ss.in_check` from checkers; `ss.move_count = 0`;
   `best_value = -VALUE_INFINITE`; PV nodes clear `pv_table[ply]`; main
   thread calls `check_time` (§3.9); PV nodes bump `sel_depth`;
   `stack[cur].stat_score = 0`; `stack[cur - 1].reduction` is read into a
   local and reset to 0; `stack[cur + 2].cutoff_cnt = 0`;
   `ss.follow_pv` = root, or parent followed the previous iteration's PV and
   played its move (consumed by doc 10).
3. **Abort/draw/max-ply** (non-root): if `stop` (relaxed load) or
   `board.is_draw(ply)` or `ply >= MAX_PLY`: return `evaluate(board)` if at
   max ply and not in check, else `value_draw(self.nodes)`.
4. **Mate-distance pruning** (non-root): even an immediate mate can't beat a
   shorter mate already found higher up:
   ```rust
   alpha = alpha.max(mated_in(ply as i32));
   beta  = beta.min(mate_in(ply as i32 + 1));
   if alpha >= beta { return alpha; }
   ```
5. **TT probe** (doc 08 API): unpack `(tt_hit, tt_data, tt_writer)`; map
   `tt_data.value` through `value_from_tt(v, ply as i32, board.rule50() as i32)`;
   at root, override the ttMove with `root_moves[pv_idx].pv[0]`. Sticky PV
   flag: `ss.tt_pv = NT::PV || (tt_hit && tt_data.is_pv)`. No cutoff happens
   here — the NonPV TT cutoff runs *after* the static-eval slot (step 7),
   because doc 10's hindsight depth adjustments must feed it.
6. **Static eval slot**: in check → `ss.static_eval` borrows
   `stack[cur - 2].static_eval` and the node jumps to the move loop (no
   pruning happens in check). Otherwise: reuse `tt_data.eval` if valid, else
   `evaluate(board)`; correction history (eval doc) produces
   `ss.static_eval`; a valid `tt_value` whose bound points the right way
   (`bound & (if tt_value > eval { LOWER } else { UPPER })`) substitutes as
   a better `eval` for pruning decisions — deliberately with **no**
   non-decisive guard, unlike qsearch's stand-pat substitution (§3.8 step 5),
   which does require it (SF parity, asymmetric by design). On a TT
   miss, immediately store an eval-only entry (`VALUE_NONE` value,
   `BOUND_NONE`, `DEPTH_UNSEARCHED`, the **raw uncorrected** eval) — the TT
   doubles as an eval cache. `improving` / `opponent_worsening` derivations
   and the hindsight depth adjustments (which modify `depth` before step 7
   reads it): ⟨D10⟩.
7. **TT cutoff** (mirrors doc 10 N6 → N6b ordering; every condition below
   sees the hindsight-adjusted `depth` from step 6): at NonPV with no
   excluded move, needs `tt_data.depth > depth - (tt_value <= beta) as Depth`,
   a valid value, the bound matching the cutoff direction
   (`bound & (if tt_value >= beta { LOWER } else { UPPER })`), and never when
   `board.rule50() >= 96` (graph-history guard). The refinements (cutNode
   agreement, child-entry verification at depth ≥ 7, cutoff history bonuses,
   entry penalization) are ⟨D10⟩/doc 09 concerns layered onto this check.
8. ⟨D10⟩: razoring, reverse futility, null move + verification, IIR,
   ProbCut, in-check ProbCut. (In check: skip straight to the move loop.)
9. **Move loop** (MovePicker, doc 09). Per move: skip `ss.excluded_move`;
   skip illegal moves (picker emits pseudo-legal); at root skip moves outside
   `root_moves[pv_idx..pv_last]`. `ss.move_count += 1`; `new_depth = depth - 1`.
   ⟨D10⟩: LMP/futility/SEE pruning block, singular extensions, reduction
   finalization.
10. **Make + recurse** (PVS shape). `do_move` wrapper: `nodes += 1`, install
    the child's `cont_hist` key
    `(in_check, capture, piece, to)`, TT prefetch hook, set
    `ss.current_move`. Then the three-call PVS structure:
    - **(a) reduced zero-window** (`depth >= 2 && move_count > 1`): depth
      `d = new_depth - r/1024` (clamped per doc 10; skeleton phase runs
      `r = 0`), window `(-alpha-1, -alpha)`, `NonPv`, `cut_node = true`;
      record `stack[cur].reduction = new_depth - d`. If it beats alpha and
      `d < new_depth`, **(b) full-depth zero-window re-search** at
      `new_depth`, `NonPv`, `!cut_node`. When (a)'s conditions don't hold and
      (`!NT::PV || move_count > 1`), go straight to (b).
    - **(c) PV re-search**: PV/Root nodes only, when
      `move_count == 1 || value > alpha`: full window `(-beta, -alpha)` at
      `new_depth`, child type `Pv`, `cut_node = false`.
    Undo the move.
11. **Stop check**: `if stop { return VALUE_ZERO; }` — the subtree result is
    untrusted (§3.9).
12. **Root bookkeeping**: §3.4 per-move updates. **Equal-value dither**: near
    the leaves an equal-to-best value is occasionally promoted —
    `inc = (value == best_value && ply + 2 >= root_depth as usize
    && (nodes & 14) == 0 && !is_win(value.abs() + 1)) as Value` — deliberate
    diversity; then compare `value + inc` against `best_value`/`alpha`.
13. **Best/alpha/beta updates**: on `value > best_value`: update; on
    `value > alpha`: `best_move = move`; PV nodes `update_pv(ply, move)`; if
    `value >= beta`: `stack[cur].cutoff_cnt += ((extension < 2) || NT::PV) as i32`,
    break (fail high); else `alpha = value` (⟨D10⟩: the sibling
    depth-decrease on alpha raise). Non-best moves are recorded (two fixed
    32-slot arrays for quiets/captures) for doc 09's maluses.
14. **Node result**:
    - Fail-high softening (search-instability damping, kept with the frame):
      when `best_value >= beta` and neither `best_value` nor `alpha` is
      decisive (alpha, not beta — matching SF and doc 10 §3.10 step 1),
      `best_value = (best_value * depth + beta) / (depth + 1)`.
    - No legal moves: `if excluded_move != Move::NULL { alpha } else if
      in_check { mated_in(ply as i32) } else { VALUE_DRAW }`.
    - History updates on best move (doc 09's `update_all_stats`), fail-low
      parent bonuses: doc 09/⟨D10⟩.
    - ttPv back-propagation on fail low: `stack[cur].tt_pv |= stack[cur - 1].tt_pv`.
    - **TT store** (skipped for excluded-move searches and for root nodes
      with `pv_idx != 0`): bound `LOWER` if `best_value >= beta`, `EXACT` if
      PV node with a best move, else `UPPER`; value through
      `value_to_tt(best_value, ply as i32)`; move `best_move`; depth `depth` (or
      `depth + 6` when `move_count == 0`); eval = the raw uncorrected static
      eval; `is_pv = ss.tt_pv`.
    - Correction-history update: eval doc.
    - `debug_assert!(best_value > -VALUE_INFINITE && best_value < VALUE_INFINITE)`.

### 3.8 Quiescence search — complete spec

`qsearch::<NT>(alpha, beta, ply)`, `NT ∈ {Pv, NonPv}`. No depth parameter:
everything probes and stores at `DEPTH_QS = 0`. There is **no quiet-checks
stage** — the picker yields captures + queen promotions, or all evasions in
check (doc 09). Steps:

1. **Upcoming-repetition** (same slot as main search): if
   `alpha < VALUE_DRAW` and repetition is forcible, `alpha = value_draw(nodes)`;
   return if `alpha >= beta`.
2. **Init**: PV nodes clear `pv_table[ply]`; `ss.in_check` from checkers;
   `move_count = 0`; PV nodes bump `sel_depth = max(sel_depth, ply + 1)`.
3. **Draw/max-ply**: `board.is_draw(ply) || ply >= MAX_PLY` → return
   `evaluate(board)` if at max ply and not in check, else plain `VALUE_DRAW`
   (deliberately not dithered here; SF parity).
4. **TT probe**: map value through `value_from_tt`. NonPV cutoff needs only
   `tt_depth >= DEPTH_QS`, a valid value, and
   `bound & (if tt_value >= beta { LOWER } else { UPPER })`.
5. **Stand pat** (not in check):
   - `raw_eval` = valid `tt_data.eval`, else `evaluate(board)`; corrected →
     `ss.static_eval`; `best_value = ss.static_eval`;
   - a valid, non-decisive `tt_value` whose bound points the right way
     (`bound & (if tt_value > best_value { LOWER } else { UPPER })`)
     replaces `best_value`;
   - if `best_value >= beta`: **softening blend** when non-decisive —
     `best_value = (467 * best_value + 557 * beta) / 1024` — then on a TT
     miss write an eval-only entry (`VALUE_NONE`, `BOUND_LOWER`,
     `DEPTH_UNSEARCHED`, no move, raw eval) and return `best_value`;
   - `alpha = alpha.max(best_value)`;
   - `futility_base = ss.static_eval + 335`.

   **In check**: `best_value = -VALUE_INFINITE`, `futility_base = -VALUE_INFINITE`
   — no stand pat; all evasions get searched.
6. **MovePicker** at `DEPTH_QS`, passing only the ply-1 continuation history;
   `prev_sq` = destination of `stack[cur - 1].current_move` if it is a real
   move, else none.
7. **Per-move rules**. For each pseudo-legal move: legality check;
   `move_count += 1`. The entire pruning block is gated on
   `!is_loss(best_value)` — the **evasion guard**: while in check and no
   legal evasion has produced a real value yet, `best_value` is
   `-VALUE_INFINITE` (a loss), so nothing is pruned until at least one escape
   has been searched; a checkmate can never be pruned into existence.
   Within the block:
   - For moves that do **not** give check, whose destination is **not**
     `prev_sq`, that are **not** promotions, and while
     `!is_loss(futility_base)`:
     - **late-move cutoff**: if `move_count > 2`, skip (and all later such
       moves);
     - **futility**: `futility_value = futility_base + PIECE_VALUE[captured]`
       (SEE scale, D11: P 208, N 781, B 825, R 1276, Q 2538); if
       `futility_value <= alpha`:
       `best_value = best_value.max(futility_value)`, skip;
     - **SEE vs the window**: if `!board.see_ge(m, alpha - futility_base)`:
       `best_value = best_value.max(alpha.min(futility_base))`, skip.
   - Skip all remaining non-captures (`!capture_stage(m)` → continue). In
     practice this drops the quiet evasions once one escape has a real
     value; queen promotions are `capture_stage` (doc 05 §3.2), so they are
     never dropped here — they proceed to the SEE floor like any capture.
   - **Hard SEE floor** for everything left: if `!board.see_ge(m, -74)`, skip.
8. **Recurse** with the full window and the same node type:
   `value = -self.qsearch::<NT>(-beta, -alpha, ply + 1)` around
   do/undo (node counter bumped, child cont-hist ref installed).
9. **Updates**: on `value > best_value`: update; on `value > alpha`:
   `best_move = m`, PV nodes `update_pv`, `alpha = value` if `value < beta`,
   else break (fail high).
10. **Mate and the narrow stalemate test** (`move_count == 0` after the
    loop): in check → return `mated_in(ply as i32)`. Otherwise run the stalemate
    check only when it is cheap and plausible: side to move has **no
    non-pawn material**, none of its pawns has an empty square in front
    (single-push targets all blocked), and the previous move captured a
    piece ≥ knight (the classic "qsearch just won material into a stalemate
    trap" pattern) — only then pay for a full legal-move-count check; if
    zero legal moves, `best_value = VALUE_DRAW`. Rationale: a full legality
    scan at every quiet qsearch leaf would be ruinous; this narrow test
    catches nearly all real qsearch stalemate blunders for ~zero cost.
11. **Fail-high softening**: if `best_value > beta` and not decisive,
    `best_value = (481 * best_value + 543 * beta) / 1024`.
12. **TT store**: bound `LOWER` if `best_value >= beta` else `UPPER` —
    **never EXACT in qsearch** — depth `DEPTH_QS`, move `best_move`, value
    `value_to_tt(best_value, ply as i32)`, eval = raw uncorrected static eval,
    `is_pv` = the probed pv flag **only** (`tt_hit && tt_data.is_pv`),
    deliberately NOT OR'd with the node's own PV status (SF stores `pvHit`
    here; OR-ing in PV-ness would systematically inflate the sticky ttPv
    flag that feeds doc 10's large reduction adjustments).

### 3.9 Node counting, selDepth, stop cadence, untrusted returns

- **Node counting**: `self.nodes += 1` inside the search's `do_move` wrapper
  (counts made moves, matching SF and our future `bench` numbers). Plain
  `u64` field now; becomes a relaxed `AtomicU64` read by others when SMP
  lands (D23) — same call sites.
- **selDepth**: per worker, reset to 0 at the start of every PV line
  (§3.6); bumped only at PV nodes in both searches:
  `sel_depth = sel_depth.max(ply as i32 + 1)` (selDepth counts from 1, ply
  from 0). Copied into `RootMove::sel_depth` when a root move's score is set.
- **Stop cadence**: two mechanisms.
  1. The main thread calls `check_time()` at every main-search node entry;
     `check_time` early-outs on a countdown counter and does real work every
     **512** calls (or `min(512, limits.nodes / 1024)` under a node limit) —
     TM doc owns what "real work" is; this doc owns the cadence and call site.
  2. Every node (main search step 3, and after every child search returns —
     step 11) does `stop.load(Ordering::Relaxed)`. Qsearch relies on the
     periodic main-search checks above it — it does not poll the flag —
     bounded extra latency, SF parity.
- **Untrusted-return-on-stop contract**: once `stop` is observed after a
  child search, the node returns `VALUE_ZERO` *immediately* — no best-move
  update, no PV write, no TT store, no history update. Consequently any value
  propagated out of a stopped search is garbage by construction, and the only
  consumers of search results are ID-loop sites explicitly guarded by
  `!stop` (plus the aborted-loss hygiene of §3.5). `bestmove` is always
  printable because depth-1 root results complete before any stop can be
  honored (the ID loop finishes the first iteration's move 1 before TM may
  stop) and `last_best_move_pv` persists across iterations.

### 3.10 Worker state and module layout

Modules per (D13): `search/mod.rs` (Worker, ID loop, search/qsearch),
`search/params.rs` (every named constant below), `search/stack.rs` (Stack,
PvLine), `search/rootmoves.rs`. All tuned constants live in
`search/params.rs` with SF-current values (D20) — no inline magic numbers:

```rust
// search/params.rs — SF-current starting points (D20)
pub const ASPIRATION_BASE_DELTA: Value = 5;
pub const ASPIRATION_MSS_DIVISOR: i64 = 10588;
pub const ASPIRATION_WIDEN_NUM: Value = 44;    // delta += 44 * delta / 128
pub const ASPIRATION_WIDEN_DEN: Value = 128;
pub const FAIL_HIGH_SOFTEN: (Value, Value) = (1, 1);      // (v*depth + beta)/(depth+1) — shape, not tunable pair
pub const QS_FUTILITY_MARGIN: Value = 335;     // futility_base = eval + 335
pub const QS_STANDPAT_BLEND: (Value, Value, Value) = (467, 557, 1024);
pub const QS_FAILHIGH_BLEND: (Value, Value, Value) = (481, 543, 1024);
pub const QS_SEE_FLOOR: Value = -74;
pub const QS_LATE_MOVE_LIMIT: i32 = 2;         // moveCount > 2 → stop non-checks
pub const TT_CUTOFF_RULE50_LIMIT: i32 = 96;  // i32: matches value_from_tt's rule50 (doc 02)
pub const CHECK_TIME_CALLS: i32 = 512;
```

```rust
pub struct Worker {
    pub board: Board,                          // own copy (D8, D23)
    pub root_moves: RootMoves,
    pub stack: Box<[Stack; STACK_SIZE]>,
    pub pv_table: Box<[PvLine; MAX_PLY + 1]>,
    pub nodes: u64,                            // relaxed AtomicU64 in per-thread Counters
                                               // when SMP lands (doc 13 §3.10), §3.9
    pub sel_depth: i32,
    pub root_depth: Depth,
    pub root_delta: Value,
    pub pv_idx: usize,
    pub best_move_changes: f64,                // becomes AtomicU64 in Counters when SMP
                                               // lands (doc 13 §3.10 owns aggregation:
                                               // main worker swap(0)-harvests per iteration;
                                               // tot_best_move_changes stays main-worker-only
                                               // manager state)
    pub limits: Limits,                        // UCI doc owns Limits
    // histories: doc 09; tt: &TT via search entry, doc 08; TM state: TM doc
}
```

The stop flag is passed as `&AtomicBool` (later `Arc` plumbing per the
threads doc). `Board::clone()` per worker carries its history Vec (D8).

## 4. Implementation plan

Ordered, commit-sized; each step keeps `cargo test` and `cargo clippy
--all-targets` green. External prerequisites: attack tables, movegen,
make/unmake + Zobrist, perft green (their own docs), a phase-A `evaluate`
(D21). Steps 6+ interleave with docs 08/09 as noted.

1. **Value semantics** — doc 02's `types` layer (constants, predicates,
   `mate_in`/`mated_in`, `value_to_tt`/`value_from_tt`) landed per that doc's
   plan; this step adds only `value_draw` and the `search/params.rs`
   scaffold. *Done when*: doc 02's unit tests (band edges, roundtrips for
   all plies 0..246, rule-50 downgrade cases) are green and `value_draw` is
   covered here.
2. **Stack + PvLine** — `search/stack.rs`: `Stack`, `ContHistId` stub with
   `SENTINEL`, `SS_OFFSET`/`STACK_SIZE`/`ss()`, `PvLine` + `update_pv`,
   init routine. *Done when*: tests exercise `ss(0)-7` and
   `ss(MAX_PLY)+2` in-bounds, PvLine load/clear, size assertion on `Stack`.
3. **RootMove(s)** — struct, EMA update helpers, stable descending sort.
   *Done when*: sort test proves stability (unsearched moves keep prior
   order) and EMA tests match the formulas.
4. **Naked negamax + qsearch skeleton** — `search::<NT>` with steps
   2/3/4/9/10/13/14 (no TT), qsearch with stand pat + captures (MVV placeholder
   ordering until doc 09), mate-distance pruning, mate/stalemate returns,
   fixed-depth `go depth n`. *Done when*: a mate-in-1/2/3 suite is solved at
   depth ≤ 5 with correct `mate n` scores; startpos depth 6 returns a sane
   move; node counts deterministic across runs.
5. **Iterative deepening + root bookkeeping** — ID loop, previousScore
   snapshots, lastBestMove tracking, `bestmove` output path, selDepth, node
   counting, stop flag + untrusted-return, `check_time` cadence stub.
   *Done when*: `go depth N` emits monotone `info depth` lines with PVs that
   replay legally; `stop` mid-search always yields a legal bestmove.
6. **TT integration** (after doc 08 lands): probe/store at the marked
   points, `value_to_tt`/`from_tt` at the boundary, eval-cache writes,
   NonPV TT cutoff with rule-50 guard, qsearch `DEPTH_QS` interaction.
   *Done when*: bench nodes drop materially vs step 5 at equal depth; TT-off
   vs TT-on return identical values on a test suite at fixed depth.
   (Re-scoped by the S0 landing of doc 10 §3.9's sibling depth reduction,
   2026-07-26: fixed-depth centipawn values are order-dependent once an
   alpha raise cheapens later siblings, and the TT changes move order.
   The bar holds exactly for DECISIVE values — the §3.9 rule exempts them
   — and within a drift band for centipawn fixtures; the SPRT channel
   arbitrates beyond that, as it already does for LMP/LMR lossiness.)
7. **PVS zero-window** — three-call structure with `r = 0`, `cut_node`
   propagation, `all_node`, zero-window debug_asserts. *Done when*: fixed
   suite node counts drop vs pure negamax; scores unchanged at fixed depth.
8. **Aspiration windows** — §3.6 complete, `root_delta`, `failedHighCnt`
   depth shrink, widening, bound flags on uciScore. *Done when*: bench nodes
   drop again; a crafted volatile position shows fail-high → shallower
   re-search → widened delta in a trace test.
9. **Qsearch full spec** — needs `Board::see_ge` (D11): futility rules,
   window-SEE, floor `-74`, late-move cutoff, evasion guard, narrow
   stalemate test, softening blends, never-EXACT stores. *Done when*: unit
   FENs cover each rule (a hanging-queen recapture pruned by SEE floor, the
   stalemate trap FEN returns 0, in-check node with one losing evasion still
   searches it) and bench improves or holds strength in self-play.
10. **Hygiene pass** — forgotten mate, aborted loss, `go mate n`, equal-value
    dither, fail-high softening in the main node. *Done when*: regression
    tests for each (a stopped search on a mated position reports the previous
    iteration's move; `go mate 2` stops after proving mate-in-2).

Every functional step reports its new `bench` node count in the commit
message; non-functional refactors must not change it (D24). Strength deltas
claimed only via fastchess/cutechess SPRT (D25).

## 5. Testing and validation

**Unit tests**

- Value (owned by doc 02, gating here): `value_to_tt`/`value_from_tt`
  roundtrip property over `v ∈ {mate band, TB band, normal}` ×
  `ply ∈ 0..=246`; downgrade triggers exactly when
  `VALUE_MATE - v > 100 - rule50` (and TB analog). This doc adds:
  `value_draw` ∈ {−1, +1} and correlates with `nodes & 0x2`.
- Stack: offsets in bounds at both extremes; padding slots carry
  `ContHistId::SENTINEL` and `VALUE_NONE` evals after init; `ply` field
  matches index.
- PvLine: `update_pv` splices `[m] ++ child`; clearing is O(1); root PV
  extraction equals the concatenation of chosen moves in a scripted tree.
- RootMoves: stable sort semantics; EMA init/update; `score_is_exact_loss`.
- NodeType: compile-fail-style guarantees via debug_asserts exercised in a
  `#[should_panic]` (debug) test for PV+cutNode and non-zero NonPV windows.

**Search behavior tests** (fixed positions, fixed depth — deterministic
single-threaded)

- Mate suite: ~20 FENs mate-in-1..4, expected `mate n` and best move.
- Mated-side reporting: a lost position reports negative mate correctly
  (ply-relative conversion both directions through the TT).
- Qsearch: stand-pat cutoff FEN, futility-pruned bad capture FEN, SEE-floor
  FEN, stalemate-trap FEN, in-check single-evasion FEN.
- Draw handling: threefold shuffle position scores in {−1, +1}; 50-move-rule
  draw detected at the boundary.
- Stop robustness: launch `go infinite`, stop at randomized node counts
  (scripted), assert bestmove is always legal and equal to the last
  completed-iteration best.

**Gates (D24)**

- Perft suite must stay green (unchanged by this doc, asserted anyway).
- `bench`: fixed FEN suite → total nodes + nps; the node count is the search
  fingerprint. Two consecutive runs must match exactly (single thread).
  Every commit in §4 records its bench.

**Regression strategy**

- Each §4 step: fixed-depth score-equality suite vs previous step where
  semantics should be identical (PVS, TT) and self-play SPRT where they
  should gain (aspiration, qsearch rules) per (D25).

## 6. Performance notes

- **Targets**: skeleton (steps 4–5, PSQT eval) ≥ 1.5 Mnps single-thread on
  x86-64-v3 (D16 release profile); after TT+PVS+aspiration, effective branching
  factor from `bench` depth/time curves should sit near ~2 at depth 12+.
  Track: bench nodes, nps, aspiration re-search count, TT hit rate.
- **Bounds checks**: stack/pv indexing is bounds-checked; the pattern
  `stack[cur ± k]` with a hoisted `cur` lets LLVM elide most checks. Only if
  a profile shows remaining cost do we consider `get_unchecked` with a
  measurement comment (D14).
- **Do not memset**: `PvLine::clear` is `len = 0`; never zero the 494-byte
  move array. Stack re-init per `go` is a single fill of 10 KB — fine.
- **No allocation in the tree**: `Vec` appears only in `RootMove` (root-only,
  cold). Everything per-node lives in locals or the worker arrays.
- **Monomorphization**: three `search` instances + two `qsearch` — watch
  I-cache bloat when doc 10 lands; if `search` grows huge, factor cold tails
  (`#[cold]` result handling) out of the generic body.
- **Overflow traps**: `mean_squared_score` is i64 (§3.4); mate arithmetic
  stays within i32 comfortably; `debug_assert!` the
  `-VALUE_INFINITE < value < VALUE_INFINITE` invariant at every return.
- **Pitfalls**: unstable sort at root (ordering regression that costs real
  Elo and is invisible to tests that only check the best move); forgetting to
  reset `stack[cur-1].reduction` (poisons hindsight logic later); storing
  corrected instead of raw eval into the TT (poisons the eval cache);
  dithering decisive scores (breaks mate play) — the `is_decisive` gates are
  load-bearing.

## 7. Interactions

- **Depends on**: board/make-unmake + Zobrist/StateInfo doc (D8–D10),
  attack-tables doc (D12), movegen/perft doc, SEE (`Board::see_ge`, D11),
  the phase-A evaluation doc (D21/D22) for `evaluate()`.
- **doc 08 (TT)**: this doc consumes `tt.probe`/`TtWriter::write`, defines
  the `value_to_tt`/`value_from_tt` boundary, `DEPTH_QS`/`DEPTH_UNSEARCHED`
  usage, never-EXACT-in-qsearch, eval-cache writes, `tt_pv` round-trip.
- **doc 09 (move ordering & history)**: MovePicker construction sites (main,
  qsearch-at-`DEPTH_QS`), the `ContHistId` contract (§3.2.3; representation,
  `SENTINEL`, and `piece_to()` owned by doc 09 §3.2), history pre-aging hook
  (`fill(100)`, `×789/1024`), quiet/capture searched-lists for maluses,
  `update_all_stats` call site.
- **doc 10 (pruning/reductions/extensions)**: every ⟨D10⟩ slot in §3.7.2;
  consumes `root_delta`, `ss.reduction`, `ss.cutoff_cnt`, `ss.stat_score`,
  `ss.follow_pv`, `improving`; owns the LMR `r` fed into call (a).
- **Time-management doc**: `check_time` internals, soft-stop factors,
  `search_again_counter`, `stop_on_ponderhit`, consumes
  `best_move_changes`/`last_best_move_depth`/`effort`.
- **Threads/SMP doc (D23)**: Worker ownership, `&AtomicBool` stop, atomic
  node counters, aspiration `thread_idx % 8` jitter, best-thread voting.
- **UCI doc**: `Limits`, `go` parsing, `info`/`bestmove` emission,
  searchmoves filtering into `RootMoves`, MultiPV option.

## 8. Acceptance criteria (D26)

1. `cargo test` and `cargo clippy --all-targets` clean; all §5 unit and
   behavior tests present and passing.
2. `value_to_tt`/`value_from_tt` property tests pass, including 50-move
   downgrade edges; every TT read/write in search demonstrably routes
   through them (grep-able single boundary).
3. Search stack is `[Stack; MAX_PLY + 10]` at offset 7; a debug build
   completes a forced deep search (fine-70 style shuffling position,
   `go depth 40`) with no panic and no out-of-bounds.
4. Mate suite: 100% of mate-in-1..3 solved with exact `mate n` scores at
   depth ≤ 12; a mated root reports the correct negative mate. (Bar was
   depth ≤ 6 until S1 — live correction history, 2026-07-16 — shifted
   proof depth in the pawnless fixtures: the whole tree shares the
   pawn/minor correction cells, so the mate line's own bonuses saturate
   them and inflate every sibling's pruning margins. Exact scores and
   optimal first moves remain mandatory; only the depth budget moved.)
5. `bench` produces bit-identical node counts across two consecutive
   single-threaded runs; the bench number is recorded in every search commit.
6. Zero-window invariant holds: an instrumented debug run over the bench
   suite reports no NonPV node with `beta != alpha + 1` and no PV cutNode.
7. Aspiration mechanics verified by trace test: initial delta matches
   `5 + |mss|/10588`; window widens by `44·delta/128` per re-search;
   fail-high re-searches run at `max(1, root_depth − failedHighCnt)`;
   fail-low resets `failedHighCnt` and lowers beta to old alpha.
8. Qsearch rule coverage: each of the five per-move rules (late-move cutoff,
   futility, window-SEE, non-capture skip, −74 floor) has a FEN test that
   flips behavior when the rule is disabled; stalemate-trap FEN returns 0;
   in-check nodes never stand pat and never prune before one legal evasion.
9. Stop robustness: 100 randomized-stop runs produce a legal bestmove every
   time; an aborted iteration never degrades the reported score/PV below the
   last completed iteration (aborted-loss + forgotten-mate tests).
10. `go mate n` terminates once `VALUE_MATE − |score| ≤ 2n` after a completed
    iteration.
11. Draws (repetition/50-move) return ±1 dithered values in the main search;
    no decisive score is ever dithered, softened, or blended (asserted).
12. Strength: with steps 1–10 complete plus docs 08/09 baselines, fixed-depth
    self-play vs the pure-negamax skeleton shows the expected node reductions
    at equal depth, and SPRT vs the previous milestone is non-regressing
    (D25).

## 9. Open questions

1. **Sibling doc filenames**: docs 09 (move ordering/history) and 10
   (pruning) are fixed by the suite plan; the numbers assumed here for TT
   (08), time management, threads, and UCI must be reconciled against the
   final suite index when those docs land.
2. **`follow_pv` in v1**: the field is in the `Stack` now (cheap, frame-level),
   but SF's `followPV`/`lastIterationIdxPV` semantics are post-SF17 and only
   matter once doc 10's IIR/PV-quiet-pruning gates exist. Decision: field
   now, semantics with doc 10; revisit if it complicates the skeleton.
3. **`mean_squared_score` as `i64`** deviates from SF's `int` (which fits
   with ~5% headroom). Chosen deliberately; if a profile ever shows the
   widening matters (it won't — root-only), revert with a comment.
4. **Qsearch stop polling**: we follow SF (no stop check inside qsearch,
   bounded by main-search cadence). If worst-case qsearch subtrees ever
   exceed acceptable stop latency under `movetime`, add a cheap relaxed load
   at qsearch entry — functional no-op, revisit with time-management testing.
5. **MultiPV depth of implementation in v1**: the loop structure, bound
   flags, and hygiene guards are designed in; v1 ships with `MultiPV = 1`
   and the multi-line aborted-search guards stubbed but untested until the
   UCI doc's MultiPV support lands.
