# 10 — Pruning, reductions, extensions: the full heuristic suite

Status: Draft v1

This document is the authoritative, node-ordered specification of every
pruning, reduction, and extension heuristic in our main search node — the
layer that turns a correct-but-slow PVS negamax into a Stockfish-class
selective search. It walks the node from entry to TT store, gives every
formula with its exact SF-current constants (starting points per D20, facts
per D27), specifies where each constant lives in `search/params.rs`, and lays
out a phased introduction plan in which each wave of heuristics is a separate
SPRT-gated, bench-recorded increment (D24, D25). Nothing here is copied code:
formulas and tuned values are cited as facts from the studied Stockfish tree
(commit `eca43a97`), re-expressed as our own Rust design.

## 1. Purpose and scope

**This doc owns:**

- The complete order of operations inside `search()` at a non-root node:
  upcoming-repetition early draw, init/followPV, draw/max-ply, mate distance
  pruning, TT probe + non-PV TT cutoff logic, static eval (with correction
  history and TT-value substitution), improving/opponentWorsening, hindsight
  depth adjustments, the eval-diff history nudge, razoring, reverse futility
  pruning, null move pruning (+ verification), internal iterative reduction,
  ProbCut (+ the in-check TT shortcut), all move-loop pruning (LMP, capture
  futility/SEE, quiet history pruning, parent futility, quiet SEE), singular
  extensions/multicut/negative extensions, reduction finalization, LMR search
  mechanics and re-search rules, post-move bookkeeping, and the node-result
  block (softening, stats, TT store, correction-history update).
- The base reduction table/function and every reduction adjustment constant.
- `search/params.rs`: the single home for all tuned constants (D20).
- The phased (wave) introduction plan with SPRT gates.

**This doc does NOT own** (it consumes their interfaces):

- The search skeleton — iterative deepening, aspiration windows, root move
  handling, the `Stack` allocation trick (D18), qsearch: the search-core doc
  (`design/07-search-core.md`).
- The transposition table and its probe/store API (D19): the TT doc
  (`design/08-transposition-table.md`).
- MovePicker staging, history table layouts, the gravity update operator, and
  the concrete update helpers (`update_quiet_histories`,
  `update_continuation_histories`, `update_all_stats`,
  `update_correction_history`): the move-ordering/history doc
  (`design/09-move-ordering-history.md`). This doc specifies *when* they are
  called and with *what bonus values*.
- Static evaluation itself (D21/D22) and correction-history table internals;
  we consume `evaluate(&Board, …) -> Value` and a
  `correction_value(...) -> i32`.
- `Board` extensions: `make_move`/`unmake_move` (D8/D10), `see_ge` (D11),
  `gives_check`, `checkers`, `upcoming_repetition` (cuckoo tables),
  `is_draw`, `non_pawn_material` — board/movegen docs.
- Syzygy tablebase probing (step N7 below is plumbing only — see §9).

## 2. References

- `stockfish_reference/stockfish_search_notes.md` §4 (node walkthrough),
  §5 (reduction function), §6 (qsearch, for the shared softening idiom),
  §10 (static eval / correction history), §14 (introduction order).
- `stockfish_reference/Stockfish/src/search.cpp` (verified at commit
  `eca43a97`): node steps at lines ~700–1590, reduction table at 693–694 and
  1830–1833, `lmrDivisor` at 54–55, `is_shuffling` at 149–156,
  `SEARCHEDLIST_CAPACITY = 32` at 71, `value_to_tt`/`value_from_tt` at
  1852–1894.
- `stockfish_reference/Stockfish/src/history.h`
  (`CORRECTION_HISTORY_LIMIT = 1024`).
- Our code: `src/board.rs`, `src/moves.rs`, `src/types.rs` for existing APIs
  (`Board::pieces(color, piece_type)`, `Move::normal(from, to)`,
  `MoveList`, `Square::E4`-style constants).

All constants below were re-verified against the C++ source, not just the
notes. Where the notes and source differed, the source wins (differences are
flagged inline).

## 3. Design

### 3.1 Where constants live: `search/params.rs` (D20)

Every tuned number in this document gets a named `pub const` in
`src/search/params.rs`, grouped by heuristic. No magic numbers inline in
`search.rs` — the reviewer check is literally `grep -nE '[0-9]{3,}' src/search/search.rs`
returning (almost) nothing. Values are SF-current starting points; the
*shape* of each formula is the design, the values are tunables.

Rust integer division truncates toward zero, exactly like C++ — so formulas
like `stat_score * 445 / 4096` with negative `stat_score` behave identically.
This matters: do not "fix" them to floor division.

Excerpt (full list follows the walkthrough it belongs to; this shows the
style):

```rust
//! search/params.rs — every tuned search constant, greppable and SPSA-able.
//! Values are Stockfish-current starting points (see design/10, D20).

use crate::search::types::{Depth, Value};

// --- Razoring (N9) ---
pub const RAZOR_BASE: Value = 465;
pub const RAZOR_DEPTH_MULT: Value = 300; // * depth^2

// --- Reverse futility pruning (N10) ---
pub const RFP_MAX_DEPTH: Depth = 17;            // exclusive: depth < 17
pub const RFP_MULT_BASE: Value = 40;
pub const RFP_MULT_SLOPE: Value = 4;            // per depth
pub const RFP_MULT_CAP: Value = 80;
pub const RFP_MULT_NO_TTHIT: Value = 20;        // subtracted when !tt_hit
pub const RFP_IMPROVING: Value = 2934;          // /1024 of futility_mult
pub const RFP_OPP_WORSENING: Value = 343;       // /1024 of futility_mult
pub const RFP_CORR_DIV: i32 = 182_069;          // |correction_value| divisor
pub const RFP_RET_BETA_W: i32 = 716;            // return blend, /1024
pub const RFP_RET_EVAL_W: i32 = 308;
// ... etc.
```

### 3.2 Data structures

**Search stack entry** — doc 07 §3.2.3 owns this type; the sketch below is a
non-normative mirror so this doc reads standalone. The stack is
`Box<[Stack; STACK_SIZE]>` (`STACK_SIZE = MAX_PLY + 10`) owned by the
`Worker`, with search ply `p` living at index `p + SS_OFFSET`
(`SS_OFFSET = 7`, doc 07 §3.2.1) so that reads from `ss-7` to `ss+2` are
plain in-bounds indexing (D18). `Stack` is `Copy` and small (≤ 40 bytes, the
whole stack ~10 KB). There is **no `pv` field**: PV collection goes through
the worker-owned `pv_table: Box<[PvLine; MAX_PLY + 1]>` and `update_pv` —
doc 07 §3.3 explicitly rejects stack-held PV lines. All fields this doc
reads/writes:

```rust
#[derive(Clone, Copy)]
pub struct Stack {
    pub ply: i32,
    pub current_move: Move,      // Move::NULL when none / null move marker
    pub excluded_move: Move,     // Move::NULL unless inside a singular verification
    pub static_eval: Value,      // corrected static eval (VALUE_NONE when unset)
    pub stat_score: i32,         // history score of current_move, read by the child
    pub move_count: i32,
    pub in_check: bool,
    pub tt_pv: bool,             // sticky "was ever on the PV" flag
    pub tt_hit: bool,
    pub cutoff_cnt: i32,         // fail-highs observed at this ply (child reads parent+1)
    pub reduction: i32,          // plies of reduction the parent's LMR removed before calling us
    pub follow_pv: bool,         // on last iteration's PV path
    pub cont_hist: ContHistId,   // flat conthist index, set by do_move (doc 09 §3.2)
}
```

`ContHistId` is doc 09 §3.2's `Copy` `u16` flat index —
`((in_check*2 + capture)*13 + piece_idx)*64 + to`, range 0..3328, with a
`SENTINEL` installed at the padding slots and after null moves — resolved
against the worker's tables in one load; no borrows across recursion, no
`unsafe`. (Doc 07 §3.2.3 sketches the same handle as a packed-bitfield
`ContHistRef`; doc 09's flat index is the representation of record.) There
is no separate `cont_corr_hist` field: the continuation-correction outer
index is derived on use as `ContHistId::piece_to()` (`id % 832`) — doc 09
§9 Q2; the final call rests with docs 07/09 when `Stack` is frozen.

**Searched-move lists** for end-of-node maluses — fixed capacity 32 each
(SF `SEARCHEDLIST_CAPACITY = 32`), quiets and captures separately:

```rust
pub struct SearchedMoves {
    moves: [Move; 32],
    len: usize,
}
impl SearchedMoves {
    pub fn push(&mut self, m: Move) { /* silently drops past capacity */ }
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ { /* ... */ }
}
```

(The push-site guard in SF is `moveCount <= 32`; we make the list itself
saturating so the call site stays clean.)

**Node type.** PVS needs `Root`, `PV`, `NonPV` at compile time plus a runtime
`cut_node` flag. Doc 07 §3.7.1 owns the mechanism: sealed marker types —
uninhabited enums `Root`/`Pv`/`NonPv` implementing
`trait NodeType { const ROOT: bool; const PV: bool; }` — monomorphize one
generic `search` into three instances on stable Rust; there is no separate
`search_root` function. `all_node = !NT::PV && !cut_node`. Signature this doc
designs against (doc 07's, verbatim):

```rust
impl Worker {
    fn search<NT: NodeType>(
        &mut self,
        alpha: Value,
        beta: Value,
        depth: Depth,
        cut_node: bool,
        ply: usize,           // stack index = ply + SS_OFFSET
    ) -> Value;
}
```

The board is not a parameter — the `Worker` owns its copy (`self.board`,
D8/D23). Shorthand throughout this doc: `PV` in a condition or formula means
`NT::PV`, `root` means `NT::ROOT`, and `search::<NonPv>` / `qsearch::<NonPv>`
name the monomorphized instances. Debug invariants at entry (assert-only,
from SF's asserts):
`-VALUE_INFINITE <= alpha < beta <= VALUE_INFINITE`;
`NT::PV || alpha == beta - 1`; `!(NT::PV && cut_node)`.

**Per-worker fields this suite needs** (beyond histories/TT):
`reductions: [i32; MAX_MOVES]`, `root_delta: i32` (set by aspiration loop),
`root_depth: Depth`, `nmp_min_ply: i32`, `tt_move_history: StatsEntry<8192>`
(one scalar), `last_iteration_pv: Vec<Move>` (for followPV),
`optimism: [Value; 2]`.

### 3.3 The base reduction table and `reduction()`

One 1-D table per worker, indexed by *either* depth or move number
(`MAX_MOVES = 256` from `src/moves.rs`; `MAX_PLY = 246 < 256` so depth always
fits). Built once at `Worker` construction — runtime `f64::ln` is fine here
because it runs once, deterministically, before any search:

```rust
let mut reductions = [0i32; MAX_MOVES];
for i in 1..MAX_MOVES {
    reductions[i] = (2834.0 / 128.0 * (i as f64).ln()) as i32;
}
```

The base per-move reduction, in **1024ths of a ply** (the suite's unit
convention — divide by 1024 only at the moment of application):

```rust
#[inline]
fn reduction(&self, improving: bool, depth: Depth, move_number: i32, delta: i32) -> i32 {
    let scale = self.reductions[depth as usize] * self.reductions[move_number as usize];
    scale
        - delta * RED_DELTA_MULT / self.root_delta          // 617: wide local window → reduce less
        + (!improving) as i32 * scale * RED_NON_IMPROVING / 512 // 194: not improving → ~38% more
        + RED_BASE                                          // 1027
}
```

`root_delta = beta - alpha` at the aspiration window (never zero;
debug-assert it). Params: `RED_TABLE_NUM = 2834`, `RED_TABLE_DEN = 128`,
`RED_DELTA_MULT = 617`, `RED_NON_IMPROVING = 194` (/512), `RED_BASE = 1027`.

### 3.4 Node walkthrough — before the move loop

Steps numbered N0–N14 in execution order. "eval" below means the corrected
static eval *after* possible TT-value substitution; `ss.static_eval` is the
corrected eval *without* substitution. The raw (uncorrected) eval is kept in
a local for TT stores. `is_win/is_loss/is_decisive` are the D5 predicates.

**N0. Depth clamp + qsearch handoff.** `if depth <= 0 { return qsearch... }`
then `depth = depth.min(MAX_PLY - 1)` (extensions can overshoot).

**N1. Upcoming-repetition early draw** (non-root). If `alpha < VALUE_DRAW`
and `board.upcoming_repetition(ply)` (Zobrist cuckoo test, board doc):
`alpha = draw_value(nodes)` — the dithered draw `VALUE_DRAW - 1 + (nodes & 2)`
— and return `alpha` if `alpha >= beta`. Kills forced-repetition subtrees
for one hash lookup.

**N2. Init.** `ss.in_check = !board.checkers().is_empty()`;
`prior_capture` = did the parent's move capture; `ss.move_count = 0`;
`best_value = -VALUE_INFINITE`; `max_value = VALUE_INFINITE` (TB cap).
`ss.follow_pv = root || ((ss-1).follow_pv && (ss-1).current_move == last_iteration_pv[ply-1])`
(guarding the index). Main thread polls time; PV nodes bump `sel_depth`.
Then: `prior_reduction = (ss-1).reduction; (ss-1).reduction = 0;`
`ss.stat_score = 0; (ss+2).cutoff_cnt = 0;`
`prev_sq` = the parent move's destination (None after a null move).
Compute `correction_value` (weighted sum from the correction tables, eval
doc; the *unscaled* sum — its `/131072` is the eval delta, and the pruning
couplings below divide the unscaled value by their own big divisors).

**N3. Abort / draw / max ply** (non-root). Stop flag, `board.is_draw(ply)`,
or `ply >= MAX_PLY` → return dithered draw (or `evaluate(board)` when at
MAX_PLY and not in check).

**N4. Mate distance pruning** (non-root).
`alpha = alpha.max(mated_in(ply)); beta = beta.min(mate_in(ply + 1));`
return `alpha` if `alpha >= beta`.

**N5. TT probe.** `(tt_hit, tt) = tt.probe(key)`. `tt.value` through
`value_from_tt(v, ply, rule50)` (D5 semantics incl. the stale-decisive
downgrade). At root the ttMove is overridden with the current PV move. If
inside a singular search (`excluded_move != Move::NULL`) keep the existing
`ss.tt_pv`, else `ss.tt_pv = PV || (tt_hit && tt.is_pv)`.
`tt_capture = tt.mv != Move::NULL && board.capture_stage(tt.mv)`.

**N6. Static eval, three paths.**

- *In check*: `ss.static_eval = eval = (ss-2).static_eval` (borrowed; no
  pruning will run — after N7 we jump straight to the move loop, skipping
  N8–N13 but NOT N14).
- *Excluded move set*: reuse `ss.static_eval` from the enclosing invocation
  (same ply, already corrected); no TT eval store. `eval = ss.static_eval`
  with **no** TT-value-as-better-bound substitution — SF takes a dedicated
  `else if (excludedMove)` branch that does not fall through to the ttHit
  substitution (search.cpp). This matters: the re-probe inside a singular
  verification hits the excluded move's own entry, so applying the
  substitution would inflate `eval` toward `tt_value` and corrupt the
  razoring/RFP inputs — and hence the exclusion-search return that decides
  singular vs multicut vs negative extension. (Round-2 review fix,
  2026-07-12; previously applied unconditionally.)
- *TT hit*: `raw_eval = tt.eval` if valid else `evaluate(board)`;
  `ss.static_eval = eval = raw_eval + correction_value / 131072` (clamped
  inside the non-decisive range);
  **TT-value-as-better-bound**: if `tt.value` is valid and
  `tt.bound` includes `LOWER` when `tt.value > eval` (else `UPPER`), then
  `eval = tt.value`.
- *TT miss*: `raw_eval = evaluate(board)`; correct as above; then
  **immediately store an eval-only entry**: value `VALUE_NONE`, bound
  `NONE`, depth `DEPTH_UNSEARCHED`, move `Move::NULL`, eval = **raw** eval —
  the TT doubles as an eval cache.

Then the two trend flags:
`improving = ss.static_eval > (ss-2).static_eval;`
`opponent_worsening = ss.static_eval > -(ss-1).static_eval;`

**Hindsight depth adjustments** (from the parent's LMR bookkeeping):
- `prior_reduction >= 3 && !opponent_worsening` → `depth += 1` (the parent
  reduced us hard yet our position didn't improve for them — reduction looks
  unjustified);
- `prior_reduction >= 2 && depth >= 2 && ss.static_eval + (ss-1).static_eval > 173`
  → `depth -= 1`. (Note the `depth >= 2` guard — the distilled notes omit it.)
Params: `HINDSIGHT_RED_UP = 3`, `HINDSIGHT_RED_DOWN = 2`,
`HINDSIGHT_EVAL_SUM = 173`.

**N6b. Non-PV TT cutoff** (requires `!PV && excluded_move == Move::NULL`).
Cut when ALL of:

- `tt.depth > depth - (tt.value <= beta) as Depth` — one ply of depth slack
  for upper-bound (fail-low-ish) values;
- `tt.value` is valid;
- `tt.bound` includes `LOWER` if `tt.value >= beta`, else `UPPER`;
- `cut_node == (tt.value >= beta) || depth > 4` — shallow cutoffs only when
  the stored direction agrees with the expected node type.

Before actually returning:

- If `tt.mv != Move::NULL && tt.value >= beta`: if the ttMove is quiet, give
  it `update_quiet_histories(bonus = min(114 * depth, 724))`
  (`TTCUT_BONUS_MULT = 114`, `TTCUT_BONUS_CAP = 724`); and if
  `prev_sq` exists, `(ss-1).move_count < 4`, and `!prior_capture`, penalize
  the parent's early quiet with conthist `-2187` (`TTCUT_PARENT_MALUS`).
- **rule50 guard**: only cut if `board.rule50_count() < 96`
  (`TTCUT_RULE50_MAX = 96`) — graph-history-interaction mitigation, paired
  with the rule50 key folding (D9).
- **Child-TT verification** at `depth >= 7` (`TTCUT_VERIFY_DEPTH`): if the
  ttMove is pseudo-legal + legal and `tt.value` is not decisive, make the
  ttMove, probe the child's entry, unmake. Cut (return `tt.value`) if the
  child value is invalid, or if
  `(tt.value >= beta) == (-child.value >= beta)` (the child agrees with the
  cutoff direction). If it contradicts, fall through — no cutoff.
- Otherwise (shallower, or no usable ttMove) return `tt.value` directly.

**Wrong-direction-bound penalize** (else-branch of the same probe): if the
*only* reason the cutoff failed is that the stored bound points the wrong
way — same depth-slack condition and valid value, `tt.bound != EXACT`,
bound includes `UPPER` if `tt.value >= beta` else `LOWER` — and `depth > 5`,
then `tt_writer.penalize(1)` (decrement stored depth; TT doc). The entry
keeps mismatching our windows; age it out faster.

**N7. Syzygy TB probe** — plumbing only for now (§9). The shape: within
cardinality, `rule50 == 0`, no castling rights → a win/loss WDL result
becomes `±(VALUE_TB - ply)`; a draw-band result (draw, cursed win, blessed
loss) scores `VALUE_DRAW + 2*wdl*drawScore` with `drawScore = 1` when the
50-move rule is respected (Syzygy50MoveRule) else 0 — i.e. 0, +2, or −2,
NOT the node-counter dither (`search.cpp:921–928`); if the implied bound
already decides the node, store at `depth + 6` and return; in PV nodes tighten
`best_value` (lower) / `max_value` (upper) instead. We keep `max_value` and
the final clamp wired from day one so Syzygy can drop in later.

**If in check: skip N8–N13** (straight to N14 and the move loop).

**N8. Eval-diff history nudge for the parent's quiet move.** If the parent
move exists (not null), parent wasn't in check, and it wasn't a capture:
`eval_diff = clamp(-(ss-1).static_eval - ss.static_eval, -183, 180) + 62;`
give the parent's move `main_history << 10 * eval_diff` (indexed by the
*opponent* color — the parent's mover), and additionally
`pawn_history << 13 * eval_diff` when `!tt_hit` and the moved piece is not a
pawn and the move was not a promotion. Params: `EVALDIFF_CLAMP_LO = -183`,
`EVALDIFF_CLAMP_HI = 180`, `EVALDIFF_OFFSET = 62`, `EVALDIFF_MAIN_MULT = 10`,
`EVALDIFF_PAWN_MULT = 13`. (Grade the opponent's last quiet by the eval
swing it produced.)

**N9. Razoring** (non-PV). If `eval < alpha - 465 - 300 * depth * depth`:
return `qsearch::<NonPv>(alpha, beta, ply)` directly. No depth cap —
the quadratic margin makes it self-limiting.

**N10. Reverse futility pruning** ("futility: child"). Conditions:
`!ss.tt_pv && depth < 17 && eval >= beta && (tt.mv == Move::NULL || tt_capture)
&& !is_loss(beta) && !is_win(eval)`. Then:

```text
futility_mult = min(40 + 4*depth, 80) - 20*(!tt_hit)
margin = futility_mult * depth
       - (2934*improving + 343*opponent_worsening) * futility_mult / 1024
       + |correction_value| / 182069
if eval - margin >= beta: return (716*beta + 308*eval) / 1024
```

The return is a **blend toward beta**, not raw eval (instability damping —
the same idiom recurs at N23). The correction term *widens* the margin when
the corrector says our eval is unreliable here.

**N11. Null move pruning — cutNode-only.** Conditions:
`cut_node && ss.static_eval >= beta - 14*depth - 45*improving + 374
&& excluded_move == Move::NULL && board.non_pawn_material(us) != 0
&& ply as i32 >= self.nmp_min_ply && !is_loss(beta)`.
Note this gate uses `ss.static_eval`, NOT the TT-substituted `eval`.

- `R = 7 + depth / 3` (`NMP_R_BASE = 7`, `NMP_R_DEPTH_DIV = 3`).
- `make_null_move`; `null_value = -search::<NonPv>(-beta, -beta + 1,
  depth - R, false, ply + 1)`; `unmake_null_move`.
- If `null_value >= beta && !is_win(null_value)`:
  - return `null_value` immediately if `self.nmp_min_ply != 0 || depth < 16`
    (`NMP_VERIFY_DEPTH = 16`);
  - else **verification search** at this same node: set
    `self.nmp_min_ply = ply as i32 + 3 * (depth - R) / 4` (NMP disabled for
    the whole subtree up to that ply — recursion-proof, zugzwang-proof),
    run `v = search::<NonPv>(beta - 1, beta, depth - R, false, ply)`,
    reset `nmp_min_ply = 0`, and return `null_value` only if `v >= beta`.

Params: `NMP_EVAL_DEPTH = 14`, `NMP_EVAL_IMPROVING = 45`,
`NMP_EVAL_BASE = 374`, `NMP_VERIFY_FRAC = (3, 4)`.

**N11b. Improving late-OR.** `improving |= ss.static_eval >= beta;` — from
here on, "improving" also means "eval already beats beta" (feeds ProbCut and
the move-loop LMP/reduction). Easy to miss; SF places it between NMP and IIR.

**N12. Internal iterative reduction.** If
`!ss.follow_pv && !all_node && depth >= 6 && tt.mv == Move::NULL`:
`depth -= 1`. (PV/cut nodes without a TT move search one ply shallower
instead of running an internal search; `IIR_MIN_DEPTH = 6`.)

**N13. ProbCut.** `probcut_beta = beta + 214 - 59 * improving as Value`.
Requires `depth >= 3 && !is_decisive(beta)` and NOT
(`tt.value` valid and `tt.value < probcut_beta`) — a TT value below the
threshold says "don't bother". Then a MovePicker in ProbCut mode
(captures/queen-promotions with `see_ge(m, probcut_beta - ss.static_eval)`;
doc 09), `probcut_depth = depth - 4 - improving as Depth`. Per legal
candidate (skipping `excluded_move`):

1. make the move; `value = -qsearch::<NonPv>(-probcut_beta,
   -probcut_beta + 1, ply + 1)` — cheap verification first;
2. if it held and `probcut_depth > 0`: re-verify with a real search
   `-search::<NonPv>(-probcut_beta, -probcut_beta + 1, probcut_depth,
   !cut_node, ply + 1)`;
3. unmake; if `value >= probcut_beta`: store to TT — `BOUND_LOWER`, depth
   `probcut_depth + 1`, this move, raw eval — and (if `value` not decisive)
   return `value - (probcut_beta - beta)` (shift the surplus back onto our
   window).

Params: `PROBCUT_BETA_BASE = 214`, `PROBCUT_BETA_IMPROVING = 59`,
`PROBCUT_MIN_DEPTH = 3`, `PROBCUT_DEPTH_SUB = 4`.

**N14. Small in-check ProbCut** (runs even in check — it sits after the
in-check jump target, immediately before the move loop).
`probcut_beta = beta + 428` (`PROBCUT_CHECK_BETA = 428`); if the TT entry has
`bound & LOWER`, `tt.depth >= depth - 4`, valid non-decisive
`tt.value >= probcut_beta`, and `beta` not decisive → return `probcut_beta`.

### 3.5 The move loop — per-move pruning (before making the move)

MovePicker emits pseudo-legal moves lazily (doc 09). Skip `excluded_move`,
illegal moves, and (root) moves outside the current MultiPV band. Then:
`ss.move_count = move_count + 1`; `capture = board.capture_stage(m)`;
`gives_check = board.gives_check(m)`; `new_depth = depth - 1`;
`delta = beta - alpha`;
`r = self.reduction(improving, depth, move_count, delta)`;
if `ss.tt_pv { r += 1006; }` (`RED_TTPV_ENTRY = 1006` — provisional bump,
mostly cancelled at §3.7's ttPv subtraction).

**Pruning block** — requires `!root && board.non_pawn_material(us) != 0 &&
!is_loss(best_value)` (never prune while we still might be getting mated):

- **Late move pruning / movecount pruning**: once
  `move_count >= (3 + depth * depth) / (2 - improving as i32)` →
  `mp.skip_quiet_moves()` (remaining quiets are never even generated;
  captures still come). Params: `LMP_BASE = 3` (shape:
  `(3 + d²) / (2 - improving)`).
- `lmr_depth = new_depth - r / 1024` — the *predicted* post-LMR depth; all
  shallow-depth margins below take it as their depth input.
- **Captures and checking moves**, with
  `capt_hist = capture_history[moved_piece][to][captured_type]`:
  - *Capture futility* (`!gives_check && lmr_depth < 7`): skip if
    `ss.static_eval + 231 + 232*lmr_depth + piece_value(captured)
    + 131*capt_hist/1024 <= alpha`.
    Params: `CFUT_MAX_LMRDEPTH = 7`, `CFUT_BASE = 231`,
    `CFUT_LMRDEPTH_MULT = 232`, `CFUT_HIST_MULT = 131` (/1024).
    Piece values are the D11 SEE values (P 208, N 781, B 825, R 1276,
    Q 2538).
  - *Capture/check SEE*: `margin = 175*depth + 34*capt_hist/1024`; skip if
    `!board.see_ge(m, -margin)` — UNLESS `alpha < VALUE_DRAW` and the moved
    piece is our last non-pawn material
    (`board.non_pawn_material(us) == piece_value(moved_piece)`): sacrificing
    the last piece may be a stalemate trap, never prune it.
    Params: `CAP_SEE_DEPTH_MULT = 175`, `CAP_SEE_HIST_MULT = 34` (/1024).
- **Quiets** — gated `!ss.follow_pv || !PV` (on the PV path of a PV node,
  quiets are never pruned here):
  - `history = cont_hist(1)[pc][to] + cont_hist(2)[pc][to]
    + pawn_history[pc][to]`;
    **continuation-history pruning**: skip if `history < -4313 * depth`
    (`QHIST_PRUNE_MULT = 4313`).
  - `history += 2 * main_history[us][m.raw()]`; then **history buys
    lmr_depth**: `lmr_depth += history / LMR_DIVISOR[min(depth, 16) - 1]`
    with the 16-entry tuned table
    `LMR_DIVISOR = [3307, 2930, 2874, 2818, 3215, 3225, 3224, 2782,
    2858, 2919, 3088, 3275, 3180, 2868, 3006, 3599]`.
    **DEFERRED (P9 Waves C–G, 2026-07-12):** this history-buys-effective-depth
    half of item 11 is NOT implemented — item 11 ships with only its
    continuation-history-pruning half above. It is a distinct
    tree-*relaxing* heuristic (a good-history quiet earns effective depth and
    is thereby spared from parent-futility / quiet-SEE pruning), so it must
    clear its own fixed-node proxy SPRT (doc 14 §9.4) rather than ride the
    batched Waves-C–G sanity SPRT. There is no `LMR_DIVISOR` const in
    `params.rs` yet; add it (and the `+ 2*main_history` term at the pruning
    site) when this lands. Revisit alongside the deferred §3.7 reduction terms
    (item 15) so the reduction/effective-depth balance is tuned as a unit.
  - **Parent futility**:
    `futility_value = ss.static_eval + 40 + 138*(best_move == Move::NULL)
    + 117*lmr_depth + 90*(ss.static_eval > alpha)`;
    if `!ss.in_check && lmr_depth < 12 && futility_value <= alpha`: skip —
    and first, if `best_value <= futility_value` and neither
    `is_decisive(best_value)` nor `is_win(futility_value)`, raise
    `best_value = futility_value` (the pruned move's optimistic bound feeds
    the fail-low value). Params: `PFUT_BASE = 40`, `PFUT_NO_BESTMOVE = 138`,
    `PFUT_LMRDEPTH_MULT = 117`, `PFUT_EVAL_ABOVE_ALPHA = 90`,
    `PFUT_MAX_LMRDEPTH = 12`.
  - `lmr_depth = lmr_depth.max(0)`; **quiet SEE pruning**: skip if
    `!board.see_ge(m, -25 * lmr_depth * lmr_depth)`
    (`QSEE_MULT = 25`, quadratic in effective depth).

### 3.6 Singular extensions and multicut

Runs when the move under consideration is the ttMove. Trigger — ALL of:
`!root`, `m == tt.mv`, `excluded_move == Move::NULL`,
`depth >= 6 + ss.tt_pv as Depth`, `tt.value` valid and not decisive,
`tt.bound & LOWER`, `tt.depth >= depth - 3`, and `!is_shuffling(m, ss, board)`.

`is_shuffling` (anti-shuffle guard): true iff the move is quiet
(`!capture_stage`), `rule50_count() >= 10`, `plies_from_null >= 6`,
`ply >= 20`, and the piece is oscillating —
`m.from() == (ss-2).current_move.to() &&
(ss-2).current_move.from() == (ss-4).current_move.to()`. Don't extend
back-and-forth shuffles in 50-move-rule territory.

The exclusion search:

```text
singular_beta  = tt.value - (60 + 70*(ss.tt_pv && !PV)) * depth / 59
singular_depth = new_depth / 2
ss.excluded_move = m
value = search::<NonPv>(singular_beta - 1, singular_beta,
                        singular_depth, cut_node, ply)     // same ply!
ss.excluded_move = Move::NULL
```

Outcomes:

- **`value < singular_beta`** — the ttMove is singular. Extension size:
  ```text
  corr_adj      = |correction_value| / 194822
  double_margin = -3 + 201*PV - 157*(!tt_capture) - corr_adj
                  - 1081*tt_move_history/117824 - 41*(ply > root_depth)
  triple_margin = 72 + 306*PV - 188*(!tt_capture) + 84*ss.tt_pv - corr_adj
                  - 45*(ply > root_depth)
  extension = 1 + (value < singular_beta - double_margin) as Depth
                + (value < singular_beta - triple_margin) as Depth
  depth += 1       // the WHOLE node deepens, not just the ttMove
  ```
- **`value >= beta` and not decisive** — **multicut**: multiple moves beat
  beta without the ttMove; the node is not singular, prune the whole subtree:
  dock the scalar `tt_move_history << (-442 - 108 * depth)` and
  `return value`.
- **else if `tt.value >= beta`** → `extension = -3` (negative extension: the
  ttMove is expected to fail high anyway; search it shallower in favor of
  alternatives).
- **else if `cut_node`** → `extension = -2`.

Params: `SE_MIN_DEPTH = 6`, `SE_TT_DEPTH_SLACK = 3`, `SE_BETA_MULT = 60`,
`SE_BETA_TTPV_MULT = 70`, `SE_BETA_DIV = 59`, `SE_CORR_DIV = 194_822`,
`SE_DOUBLE_BASE = -3`, `SE_DOUBLE_PV = 201`, `SE_DOUBLE_NON_TTCAP = 157`,
`SE_DOUBLE_TMH_NUM = 1081`, `SE_DOUBLE_TMH_DEN = 117_824`,
`SE_DOUBLE_DEEP = 41`, `SE_TRIPLE_BASE = 72`, `SE_TRIPLE_PV = 306`,
`SE_TRIPLE_NON_TTCAP = 188`, `SE_TRIPLE_TTPV = 84`, `SE_TRIPLE_DEEP = 45`,
`MULTICUT_TMH_BASE = -442`, `MULTICUT_TMH_DEPTH = -108`,
`SHUFFLE_RULE50 = 10`, `SHUFFLE_PLIES_FROM_NULL = 6`, `SHUFFLE_MIN_PLY = 20`.

There are **no plain check extensions** anywhere (D17 spirit: we follow
current SF, not folklore).

### 3.7 Make move, reduction finalization

`make_move(m, gives_check)` (D10) — bumps the node counter, installs the
child's conthist handles keyed `[in_check][capture][piece][to]`, prefetches
the child's TT line (board/TT docs). Then `new_depth += extension` and the
reduction `r` accretes its adjustment list — all values in 1024ths:

```text
if ss.tt_pv:
    r -= 2766 + 1017*PV + 838*(tt.value > alpha)
         + (tt.depth >= depth) * (923 + 955*cut_node)
r += 714                                   // flat rebase
r -= 62 * move_count
r -= |correction_value| / 26131            // uncertain eval → search fuller
if cut_node:            r += 3995 + 1059*(tt.mv == Move::NULL)
if tt_capture:          r += 1039
if (ss+1).cutoff_cnt > 1:
    r += 236 + 1079*((ss+1).cutoff_cnt > 2) + 1143*all_node
else if m == tt.mv:     r = max(0, r - 2016)
ss.stat_score = if capture {
    809 * piece_value(captured) / 128 + capt_hist
} else {
    2*main_history[us][m.raw()] + cont_hist(1)[pc][to] + cont_hist(2)[pc][to]
}
r -= ss.stat_score * 445 / 4096
if all_node:            r += r * 272 / (256*depth + 285)
```

Params (grouped as `RED_*`): `RED_TTPV_BASE = 2766`, `RED_TTPV_PV = 1017`,
`RED_TTPV_TTVALUE = 838`, `RED_TTPV_TTDEPTH_BASE = 923`,
`RED_TTPV_TTDEPTH_CUT = 955`, `RED_FLAT = 714`, `RED_MOVECOUNT = 62`,
`RED_CORR_DIV = 26_131`, `RED_CUTNODE = 3995`, `RED_CUTNODE_NO_TTMOVE = 1059`,
`RED_TTCAPTURE = 1039`, `RED_CUTOFFCNT_BASE = 236`,
`RED_CUTOFFCNT_MANY = 1079`, `RED_CUTOFFCNT_ALLNODE = 1143`,
`RED_TTMOVE_SUB = 2016`, `RED_STATSCORE_NUM = 445` (/4096),
`RED_ALLNODE_NUM = 272`, `RED_ALLNODE_DEPTH_MULT = 256`,
`RED_ALLNODE_DEPTH_BASE = 285`, `STATSCORE_CAPT_NUM = 809` (/128).

### 3.8 LMR search mechanics

**Reduced zero-window search** — when `depth >= 2 && move_count > 1`:

```text
d = max(1, min(new_depth - r/1024, new_depth + 2)) + NT::PV as Depth
ss.reduction = new_depth - d          // child reads this (hindsight, N6)
value = -search::<NonPv>(-(alpha + 1), -alpha, d, true, ply + 1)
ss.reduction = 0
```

Negative net `r` lets a move search up to 2 plies *deeper* than `new_depth`;
PV nodes get +1 unconditionally. If the reduced search beats alpha:

- **deeper/shallower rule**:
  `new_depth += (d < new_depth && value > best_value + 52) as Depth
              - (value < best_value + 9) as Depth;`
  if `new_depth > d`, re-search full-depth zero-window with `!cut_node`.
- Either way, the move's continuation histories get a flat `+1415` bonus
  (`LMR_CONTHIST_BONUS = 1415`; params `LMR_DEEPER_MARGIN = 52`,
  `LMR_SHALLOWER_MARGIN = 9`).

**Non-LMR zero-window path** — else, when `!PV || move_count > 1`:
if `tt.mv == Move::NULL { r += 1085; }` (`RED_FULLDEPTH_NO_TTMOVE = 1085`);
search zero-window at
`new_depth - (r > 5039) as Depth - (r > 5223 && new_depth > 2) as Depth`
with `!cut_node` — a very high nominal reduction still costs 1–2 plies even
without LMR proper. Params: `RED_FULLDEPTH_DOCK1 = 5039`,
`RED_FULLDEPTH_DOCK2 = 5223`.

**PV re-search** — for PV nodes when `move_count == 1 || value > alpha`:
full window `(-beta, -alpha)` at `new_depth`, `cut_node = false`. Before it,
the **ttMove min-depth clamp**: if `m == tt.mv` and the TT entry is deep or
decisive (`(tt.value valid && decisive && tt.depth > 0) || tt.depth > 1`),
`new_depth = new_depth.max(1)` — don't let negative extensions dive the PV
move straight into qsearch.

### 3.9 Undo, post-move bookkeeping

Unmake; if the stop flag is set, return `VALUE_ZERO` immediately (subtree
untrusted). Root bookkeeping (effort, score EMAs, PV, `best_move_changes`)
belongs to the search-core doc. Then:

- **Equal-value dither near the leaves**: `inc = 1` iff
  `value == best_value && ply as i32 + 2 >= self.root_depth
  && (nodes & 14) == 0 && !is_win(value.abs() + 1)` — occasionally treat an
  equal move as strictly better (deliberate diversity; pairs with the draw
  dither).
- If `value + inc > best_value`: `best_value = value`; if
  `value + inc > alpha`: `best_move = m`, and PV nodes call
  `update_pv(ply, m)` on the worker's `pv_table` (doc 07 §3.3); if
  `value >= beta`: `ss.cutoff_cnt += ((extension < 2) || NT::PV) as i32`,
  break (fail high). Otherwise (alpha raise without cutoff): if
  `depth > 2 && depth < 13 && !is_decisive(value)` → `depth -= 2` (all
  remaining siblings get a cheaper search; `SIBLING_DEPTH_MIN = 2`,
  `SIBLING_DEPTH_MAX = 13`), then `alpha = value`.
- Every searched non-best move is recorded into the 32-capacity quiet or
  capture list for end-of-node maluses.

### 3.10 Node result

In order:

1. **Fail-high softening** — doc 07 §3.7.2 step 13 owns this (it stays with
   the node frame); restated here only for node-order completeness: if
   `best_value >= beta` and neither `best_value` nor `alpha` is decisive
   (source-verified guard, `search.cpp:1507` — the second test is on
   `alpha`, not `beta`):
   `best_value = (best_value * depth + beta) / (depth + 1)` — blend toward
   beta, damping fail-high noise (same idiom as N10's return).
2. **No legal moves** (`move_count == 0`):
   `best_value = if excluded_move != Move::NULL { alpha }
   else if ss.in_check { mated_in(ply) } else { VALUE_DRAW }`.
   (Inside a singular verification, stalemate/mate is meaningless — the real
   best move was excluded; fail low with alpha.)
3. **Best move found** → `update_all_stats(...)` (doc 09 owns the internals);
   at non-PV nodes additionally
   `tt_move_history << if best_move == tt.mv { 792 } else { -779 }`
   (`TMH_HIT = 792`, `TMH_MISS = -779`).
4. **Fail low, parent's move was quiet** (`best_move == Move::NULL`,
   `!prior_capture`, `prev_sq` exists) — the parent's move "worked", reward
   it:
   ```text
   bonus_scale = max(0,
         -245
       - (ss-1).stat_score / 98
       + min(59*depth, 430)
       + 191*((ss-1).move_count > 8)
       + 143*(!ss.in_check && best_value <= ss.static_eval - 103)
       + 151*(!(ss-1).in_check && best_value <= -(ss-1).static_eval - 78))
   scaled_bonus = min(141*depth - 82, 1472) * bonus_scale
   conthist(parent)  << scaled_bonus * 236 / 16384
   main_history      << scaled_bonus * 234 / 32768
   pawn_history      << scaled_bonus * 322 / 8192   // only if mover not a
                                                    // pawn and not a promotion
   ```
   Params: `FL_SCALE_BASE = -245`, `FL_STATSCORE_DIV = 98`,
   `FL_DEPTH_MULT = 59`, `FL_DEPTH_CAP = 430`, `FL_LATE_PARENT = 191`,
   `FL_BELOW_EVAL = 143` (margin 103), `FL_BELOW_PARENT_EVAL = 151`
   (margin 78), `FL_BONUS_MULT = 141`, `FL_BONUS_SUB = 82`,
   `FL_BONUS_CAP = 1472`, weights `236/16384`, `234/32768`, `322/8192`.
5. **Fail low, parent's move was a capture**: flat
   `capture_history << 901` (`FL_CAPTURE_BONUS = 901`).
6. **TB clamp** (PV nodes): `best_value = best_value.min(max_value)`.
7. **ttPv back-propagation**: if `best_value <= alpha`,
   `ss.tt_pv |= (ss-1).tt_pv` — a fail-low under a PV parent keeps the
   sticky flag alive.
8. **TT store** — skipped inside singular verifications
   (`excluded_move != Move::NULL`) and for non-first MultiPV root lines:
   bound = `LOWER` if `best_value >= beta`, `EXACT` if PV node with a best
   move, else `UPPER`; depth = `depth`, or `min(MAX_PLY - 1, depth + 6)`
   when `move_count == 0` (`TT_NOMOVES_DEPTH_BONUS = 6` — a mate/stalemate/
   TB verdict is worth more than its nominal depth); value through
   `value_to_tt(best_value, ply)`; eval = the **raw, uncorrected** static
   eval (D22).
9. **Correction-history update** — direction-consistency gated: only when
   `!ss.in_check`, best move is not a capture, and
   `(best_value > ss.static_eval) == (best_move != Move::NULL)` (we beat our
   eval iff we actually found a move — otherwise the error signal is
   polluted by the bound):
   `bonus = clamp((best_value - ss.static_eval) * depth
   * (if best_move != Move::NULL { 12 } else { 18 }) / 128, -256, 256)`
   (clamp = ±`CORRECTION_HISTORY_LIMIT/4`), then
   `update_correction_history(1114 * bonus / 1024)` (the 1114/1024 outer
   scale is in current SF; the distilled note omits it). Params:
   `CORR_UPDATE_BESTMOVE = 12`, `CORR_UPDATE_NO_BESTMOVE = 18` (/128),
   `CORR_UPDATE_SCALE = 1114` (/1024), `CORR_UPDATE_CLAMP = 256`.

Return `best_value`.

### 3.11 Node-order summary (one screen)

```text
N0  depth<=0 → qsearch; clamp depth
N1  upcoming-repetition early draw           N2  init, followPV, priorReduction
N3  stop/draw/max-ply                        N4  mate distance pruning
N5  TT probe, ttPv                           N6  static eval + correction + TT-bound subst;
                                                 improving/opponentWorsening; hindsight ±depth
N6b non-PV TT cutoff (+bonus/penalty/verify/penalize)
N7  TB probe (plumbing)      — in check? jump to N14 —
N8  eval-diff parent nudge                   N9  razoring
N10 reverse futility (blend return)          N11 null move (cutNode-only, verification)
N11b improving |= eval>=beta                 N12 IIR
N13 ProbCut                                  N14 in-check small ProbCut
loop: LMP → lmrDepth → capture fut/SEE | quiet hist-prune/buy-depth/fut/SEE
      singular/multicut/neg-ext → make → finalize r → LMR d → re-searches → PV
      → undo → dither/alpha/beta/cutoffCnt/depth-2 → 32+32 lists
end:  fail-high soften → no-moves → stats/ttMoveHistory → fail-low parent bonus
      → TB clamp → ttPv backprop → TT store → correction update
```

## 4. Implementation plan — phased waves

Prerequisites (other docs, already sequenced per notes §14 steps 1–4): PVS
negamax + qsearch + iterative deepening + aspiration windows, TT, staged
MovePicker with butterfly/capture history and `see_ge`, mate distance
pruning and value semantics. Steps N0–N5 and the bare move loop land with
the search-core doc; this plan starts from that baseline.

Ground rules for every wave (D24, D25):

- Each commit is one heuristic (or one tightly-coupled cluster) behind
  nothing — no feature flags; the commit either passes its gate or is
  reverted.
- Every functional commit changes the **bench** signature; the new count is
  recorded in the commit footer (`bench: <nodes>`). A "non-functional"
  cleanup must reproduce the bench exactly.
- **SPRT gate**: fastchess, TC 10+0.1s, 16 MB hash, 1 thread, vs. the
  previous accepted commit, alpha = beta = 0.05. Bounds by expected size:
  big-ticket items `[0, 10]` Elo, medium `[0, 5]`, refinements `[0, 3]`;
  pure refactors that should be neutral use non-regression bounds `[-3, 1]`.
  Elo brackets below are experience-based estimates for an engine at this
  maturity, NOT claims — the SPRT is the claim (D25).

**Wave A — the big three (expected +150..300 Elo combined).**
1. `params.rs` skeleton + reduction table + `reduction()` + `ss.reduction`
   plumbing. Done when: unit tests for table values pass; bench unchanged
   (table exists but unused).
2. Reverse futility pruning (N10, without the correction term — `RFP_CORR_DIV`
   term joins in Wave F). Done: SPRT `[0, 10]` pass. (+20..50)
3. Null move pruning with verification scheme (N11) + `nmp_min_ply` worker
   field. Done: SPRT `[0, 10]` pass AND the zugzwang suite (§5) shows no
   new failures. (+50..80)
4. LMR core: base `r`, the `RED_FLAT`/`RED_MOVECOUNT`/cutNode/ttMove-subtract
   adjustments, `d = max(1, min(new_depth - r/1024, new_depth + 2)) + PV`,
   deeper/shallower re-search, non-LMR dock path (r > 5039/5223), PV
   re-search rule. Done: SPRT `[0, 10]` pass. (+80..150)

**Wave B — shallow-depth pruning (expected +30..80 combined).**
5. Razoring (N9). SPRT `[0, 5]`.
6. LMP movecount pruning + `mp.skip_quiet_moves()`. SPRT `[0, 5]`.
7. Parent futility + quiet SEE pruning (`-25·lmrDepth²`), with `lmr_depth`
   derivation. SPRT `[0, 5]`.
8. Capture futility + capture SEE pruning (needs captureHistory from doc 09).
   SPRT `[0, 5]`.
9. IIR (N12) + followPV plumbing (`last_iteration_pv`). SPRT `[0, 3]`.
   *Status (2026-07-12): REVERTED. Wave B's item 9 (IIR) regressed ~−15 Elo
   over ~1036 proxy games and was backed out to branch
   `wip/p9-item9-iir-followpv`; N12 is genuinely absent between N11b and
   ProbCut (an intentional revert, not a silent deviation). The `follow_pv`
   field exists but is inert/forward-compatible. Revisit post-NNUE.*

**Wave C — history-coupled pruning/reduction (needs continuation history
from doc 09; expected +20..60 combined).**
10. `ss.stat_score` + the `RED_STATSCORE_NUM` reduction term + LMR flat
    `+1415` conthist bonus. SPRT `[0, 5]`.
11. Continuation-history pruning (`history < -4313·depth`) +
    history-buys-lmrDepth via `LMR_DIVISOR`. SPRT `[0, 5]`.
    **Partially deferred (P9 Waves C–G, 2026-07-12):** the continuation-history
    pruning landed; the **history-buys-lmrDepth** half (`+2*main_history` then
    `lmr_depth += history / LMR_DIVISOR[...]`) is **DEFERRED** — a distinct
    tree-relaxing heuristic with no `LMR_DIVISOR` const yet, gated on its own
    fixed-node proxy SPRT (doc 14 §9.4). See §3.5's quiet bullet for the full
    note. The `lmr_depth` site in `search/mod.rs` uses the plain
    `new_depth - r/1024` and flags the deferral in-line.
12. Eval-diff parent nudge (N8) + TT-cutoff quiet bonus/parent penalty (N6b
    extras). SPRT `[0, 3]`.

**Wave D — singular extensions and multicut (needs a trustworthy TT;
expected +15..40 combined).**
13. Singular trigger + exclusion search + single extension + node `depth++`;
    `excluded_move` semantics everywhere (static-eval reuse, TT-store skip,
    no-legal-moves → alpha). SPRT `[0, 5]`.
14. Multicut + negative extensions (−3/−2) + `is_shuffling` guard +
    `tt_move_history` scalar (dock on multicut, 792/−779 at node result) +
    double/triple extension margins. SPRT `[0, 5]`.
15. ttMove min-depth clamp at PV re-search + cutoffCnt reduction terms +
    sibling `depth -= 2` on alpha raise. SPRT `[0, 3]`.
    **Partially deferred (P9 Waves C–G, 2026-07-12):** the ttMove min-depth
    clamp landed (it only helps mate finding); the **cutoffCnt reduction terms
    and the §3.9 sibling `depth -= 2`** are **DEFERRED**. Their params
    (`RED_CUTOFFCNT_*`, `RED_ALLNODE_*`, `RED_TTPV_*`, `RED_TTCAPTURE`,
    `SIBLING_DEPTH_*`) are defined in `params.rs` but currently **unconsumed** —
    in this PSQT-only engine they over-reduce deep mate lines past the doc 07
    §8(4) "solved at depth ≤ 6" bar (SF absorbs the same reductions with NNUE-grade
    eval/ordering). Revisit with the P13 eval upgrade.

**Wave E — flow refinements (each small; expected +10..30 aggregate).**
16. ProbCut (N13) + in-check small ProbCut (N14). SPRT `[0, 3]`.
    *Fidelity note (round-2 review, 2026-07-12): both ProbCuts are gated
    `excluded_move == NULL`, so — unlike SF, which runs ProbCut inside an
    exclusion search and merely `continue`s over the excluded move in its own
    loop — they never fire during a singular verification. Intentional: it
    keeps the exclusion-search verdict a clean function of the ordinary move
    search. Consequence: the exclusion search does strictly more work than
    SF's, contributing to the deferred-reductions node-count trade-off. The
    now-dead `m == excluded_move` skip inside the N13 loop is retained as a
    self-documenting mirror of SF's structure.*
17. Hindsight depth adjustments (N6) + `improving |= eval >= beta` (N11b).
    SPRT `[0, 3]`.
    *Fidelity note (2026-07-12): per §3.11 the N6 trend flags and the two
    hindsight `depth ±1` adjustments run for ALL nodes — including in check —
    BEFORE the N7 in-check jump to the move loop. `improving` stays false in
    check; `opponent_worsening` and the hindsight arms use the grandparent's
    eval there.*
18. Fail-high softening + RFP return blend + equal-value dither + dithered
    draws (if not already in core). SPRT `[0, 3]`.
19. Fail-low parent bonus (full `bonus_scale` formula) + fail-low capture
    bonus + ttPv backprop + TT `depth+6` no-moves store + upcoming-repetition
    early draw (needs cuckoo tables from the board doc). SPRT `[0, 3]`.
    **Partially deferred (P9, 2026-07-12):** the fail-low bonuses, ttPv
    backprop, and TT no-moves store landed; the **N1 upcoming-repetition /
    cuckoo early draw** is **DEFERRED** — the board doc's cuckoo tables are not
    yet built.
20. TT-cutoff hardening: depth slack, cutNode agreement, rule50 ≥ 96 guard,
    depth ≥ 7 child verification, wrong-direction penalize. SPRT `[0, 3]`
    (some of these are safety, not Elo; accept on non-regression `[-3, 1]`).
    *Fidelity note (round-2 review, 2026-07-12): the `rule50 < 96` limit is
    folded into `cutoff_ok`, so at rule50 ∈ [96, 100) the whole N6b block —
    including the quiet-ttMove history bonus and the parent early-quiet malus —
    is skipped. SF instead applies those history nudges whenever the bound
    matches and gates only the RETURN / child verification on `rule50 < 90`.
    Accepted deviation: the affected band is rare (rule50 96–99) and the nudges
    are second-order; revisit if a rule50 axis is added to the histories.*

**Wave F — correction-history couplings (needs correction history from the
eval doc; expected +5..15).**
21. `correction_value` plumbed into this suite: RFP margin term
    (`/182069`), reduction term (`/26131`), singular `corr_adj`
    (`/194822`), and the node-result correction update with its
    direction-consistency gate. SPRT `[0, 5]`.
    **Partially deferred (P9, 2026-07-12):** the RFP/reduction/singular
    couplings and the node-result update landed and are live (`corr_adj` now
    tightens the singular double/triple margins); **folding the correction
    INTO `ss.static_eval`** itself (the N6 corrected static eval, via
    `to_corrected_static_eval`) is **DEFERRED** to the eval doc's corrected-eval
    consumer (D22) so the exact-value pruning-margin unit tests stay pinned to
    the raw eval.
    *Fidelity note (round-2 review, 2026-07-12): the node-result correction
    update is gated `excluded_move == NULL && move_count != 0`. SF has NEITHER
    guard. The excluded-move skip is deliberate (an exclusion search's best
    move was forcibly removed, so its eval-error signal is polluted). The
    `move_count != 0` guard is an accepted deviation: it drops the update at
    terminal (stalemate) nodes SF would still learn from — a negligible,
    rarely-reached signal — and is kept for symmetry with the TT no-moves
    store. Also (round-2, §3.7): en-passant contributes 0 victim value to
    `stat_score` (`piece_on(to)` is empty), unlike SF's post-move
    `PieceValue[captured] == PAWN` — an intentional internal convention that
    matches this engine's NO_VICTIM capture-history slot for e.p. (doc 09
    §3.9); it under-docks an e.p. capture's reduction by a pawn-scaled term
    only.*

**Wave G — Reckless-derived threat map + dependents (2026-07-11 study, D30;
scheduled AFTER the existing waves A–F).** Prereq is *infrastructure, not a
heuristic*: **the per-node opponent threat map** (D30) — once per
`make_move`/`make_null_move`, build `all_threats` + `piece_threats[pt]` for the
side-not-to-move via whole-army *setwise* attack fills (own king excluded from
occupancy so sliders see through it). Its own commit, proven **perft-clean and
bench-node-neutral** (full D24 ritual + `Bench:` trailer) before any dependent
lands. Then:
22. Threat-aware quiet ordering: a threat-escape term (moving a piece off an
    attacked square) + a symmetric offense term (moving to attack a vulnerable
    enemy). `design/09` §3.4/§3.7. SPRT `[0, 5]`.
23. Threat-coupled RFP margin: subtract a flat term (`RFP_THREAT_MARGIN`) from
    the N10 margin when no own piece is currently threatened. §3.4 N10.
    SPRT `[0, 3]`. *"Own piece" means ALL our pieces, pawns included —
    `all_threats() & occupancy(us)` — matching Reckless's `colors(stm)`; a
    threatened pawn blocks the relaxation. `RFP_THREAT_MARGIN` (currently 55,
    Reckless uses 54) is a provisional guess pending SPSA. The G1 offense
    ordering term (`QUIET_OFFENSE_FACTOR`) reads the same king-x-rayed
    `piece_threats` map (sliders see through our own king — the king-danger
    build), an intentional D30 behavior change from the deleted full-occupancy
    `threats_by_lesser`.*

**Reckless-derived candidates (study 2026-07-11, D31) — enrich the menu without
restructuring the waves; each is one commit, proxy-gated, reimplemented
idiomatically (AGPLv3 prior art, D27).** Tried at the named gate:
- *Wave C item 11* — history-coupled LMP threshold (couple the move-count
  budget to the move's own ordering history, not just `improving`); +
  history-scaled quiet-SEE margin (add a history term to the `-25·lmr_depth²`
  bar).
- *Wave C / Wave F* — rule50 (`fold_rule50`) bucket axis on the pawn/non-pawn
  correction tables (reuse the bucket we already compute; `design/09` §3.10);
  and qsearch SEE folds `|correction_value|` as an eval-uncertainty term
  (`design/07` §3.8).
- *Wave D items 13–15* — LDSE (low-depth singular extension at shallow cut-nodes
  below the `depth ≥ 6 + ttPv` SE gate); + bound-type-conditioned singular
  margin (`depth/4` when the TT bound is Exact, full depth when only Lower).
- *Wave E item 16* — dynamic-depth/dynamic-beta ProbCut re-verify (scale probe
  depth and a shifted beta by how far the qsearch probe exceeded
  `probcut_beta`, with a full-window fallback re-search).
- **NMP couplings** on the shipped N11 (three isolated SPRTs `[0,3]`): a
  cutoffCnt discount when `cutoff_count[ply+1] < 2`; veto NMP when the TT
  already promises a beta-beating capture of ≥knight material; narrow the
  null-move window to `tt_score` when a fresh TT Lower bound sits below beta.
- **Root-move re-scoring** after each completed sibling subtree (root only), so
  ordering reflects the history the just-finished subtree wrote (`design/09`
  §3.6). A mid-iteration staleness fix neither SF nor we do today.
- **SEE accuracy** (`design/04` §3.11): run promotions/en-passant through the
  real exchange loop with the *promoted* piece's value (not the blanket
  non-Normal→0 shortcut); and exclude a pinned attacker from SEE only when the
  capture square is *off* its king-pin ray (allowing on-ray recaptures).
  Correctness + SPRT; D24 perft applies.
- **nps micro-ops** (verify by bench, not SPRT — must reproduce the node count):
  branchless least-valuable-attacker in `see_ge`; an O(1) `is_direct_check`
  proxy at pruning-exemption sites (accepts losing discovered-check exemptions).
- **Correctness cleanups** (D24 perft applies): enemy-occupancy-only sniper
  detection for `slider_blockers` (`design/04` §3.5); harden `Board::legal`'s
  pin check to directional (`!ray_pass(king,from).contains(to)`, `design/05`
  §3.8) so it is correct standalone, not only when composed after
  `pseudo_legal`.

**Post-Wave-C standalone experiment (D31) — isolated and revert-friendly, never
folded into wave work.** `ilog2(depth)`-only LMR base (§3.3): drop SF's
move-number reduction dimension entirely, letting the continuous per-move
history score substitute (Reckless base `269·ilog2(depth)`, no move-count term).
Plausibly the study's highest-ceiling single idea, but a genuine change to a
shipped component — run as its OWN SPRT after the Wave-C history couplings land.

Done-condition for the whole document: §8 acceptance criteria.

## 5. Testing and validation

**Unit tests** (in `search/params.rs` and `search/reduction.rs`):
- Reduction table spot values (`reductions[1] == 0`,
  `reductions[64] == (2834.0/128.0 * 64f64.ln()) as i32`) and monotonicity.
- `reduction()` algebra: improving vs. not differs by `scale*194/512`; wider
  `delta` reduces `r`; all at fixed `root_delta`.
- Margin helpers (RFP margin, LMP threshold, singular margins) as pure
  functions with table-driven cases, including negative-value truncation
  semantics (`-4313 * depth` comparisons, `stat_score * 445 / 4096` with
  negative inputs).
- `SearchedMoves` saturation at 32.
- `is_shuffling` with hand-built stacks.

**Search-level invariants** (debug_assert, active in `cargo test`):
window sanity at entry; `value` in `(-VALUE_INFINITE, VALUE_INFINITE)`;
`!(PV && cut_node)`; `ss.reduction` reset after use; singular search never
recurses (`excluded_move` set implies trigger disabled).

**Perft** is untouched by this suite but remains the gate for any board API
this doc pulls in (`gives_check`, `upcoming_repetition`) (D24).

**Bench regression** (D24): fixed position suite at fixed depth; every
functional commit records the node count. Expected direction: Waves A–B cut
bench nodes drastically (30–70% at equal depth); Waves C–E shift it less.
An unexplained bench change = bug.

**Tactical safety suites**, run at fixed depth in CI-ish scripts:
- *Zugzwang set* (NMP verification): e.g.
  `8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1` (Réti),
  `1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1`, and a handful of pawn-endgame
  fortress positions — engine must still find the right result post-NMP.
- *Mate suite* (razoring/RFP/LMP guards): a fixed set of mate-in-2..5
  positions must be solved at depth ≤ N; the `is_loss(beta)` /
  `is_win(eval)` / `!is_loss(best_value)` guards are what this validates.
- *Repetition/50-move set* for N1 and the rule50 TT guard.

**SPRT** per wave as specified in §4 (D25). Self-play first; periodic
gauntlets vs. reference engines once past Wave B.

## 6. Performance notes

- **Targets**: the heuristics themselves must be arithmetic-only at the
  node — no allocation, no hashing beyond the TT probe already paid. The
  expensive extras are: N6b child-TT verification (one make/unmake + probe,
  gated `depth >= 7`), the singular exclusion search (a real reduced
  search, gated `depth >= 6+`), ProbCut's qsearch probes (gated
  `depth >= 3` + SEE threshold), and NMP verification (rare: `depth >= 16`).
  All are depth-gated precisely so their cost amortizes; keep the gates.
- **Branch layout**: order conditions cheapest-first and most-likely-false
  first (e.g. `!ss.tt_pv` before the margin computation in RFP). The
  in-check jump over N8–N13 should be one branch, not thirteen.
- **Integer semantics**: all formulas in `i32`; `Value = i32` (D5) makes
  this free. Watch the two i64-ish products: `scaled_bonus` peaks around
  `1472 * ~1100 ≈ 1.6M` (fits i32 comfortably) and
  `reductions[d] * reductions[mn]` peaks ≈ `122² ≈ 15k` — no overflow, but
  add a unit test proving bounds rather than reasoning in comments.
- **`ss.reduction` protocol**: parent writes before the reduced child call,
  zeroes after; child consumes-and-clears `(ss-1).reduction`. Getting this
  wrong silently misfires the hindsight adjustments — assert the reset.
- **What to measure**: nps and bench nodes per wave; time-to-depth on a
  fixed midgame set; effective branching factor
  (`nodes(d)/nodes(d-1)` trend) — the whole suite exists to push EBF from
  ~6–8 (plain PVS) toward ~1.7–2.0.
- **Pitfalls**: (a) applying `r` without the /1024 (unit confusion is THE
  classic bug here — keep a `fn to_plies(r_milli: i32) -> Depth` helper);
  (b) pruning decisive scores (every guard `is_win/is_loss/is_decisive`
  exists for mate correctness — the mate suite catches omissions);
  (c) using `eval` where SF uses `ss.static_eval` (NMP gate, futility
  bases) or vice versa (razoring, RFP) — this doc marks each site;
  (d) forgetting `excluded_move` exemptions (TT store, static-eval path,
  no-legal-moves) — breaks singular search subtly.

## 7. Interactions

Depends on (consumes):
- `design/07-search-core.md`: search skeleton, Stack allocation (D18),
  `pv_table`/`update_pv`, qsearch, aspiration/`root_delta`/`root_depth`,
  followPV source data, root move handling, stop/time plumbing.
- `design/08-transposition-table.md`: probe/store API, `TTWriter::penalize`,
  bound/gen/pv packing, `value_to_tt`/`value_from_tt` (D19).
- `design/09-move-ordering-history.md`: MovePicker (main/evasion/probcut
  modes, `skip_quiet_moves`), all history tables, the gravity operator, the
  `update_*` helpers this doc calls with specific bonuses, `ContHistId`.
- `design/11-evaluation.md`: `evaluate()` seam (D22), correction-history
  tables and `correction_value()`.
- Board-layer docs: `see_ge` (D11), `gives_check`, `checkers`,
  `non_pawn_material`, `is_draw`, `upcoming_repetition` (cuckoo),
  `rule50_count`, `plies_from_null`, make/unmake + null move (D8/D10),
  `capture_stage`.

Feeds:
- `design/12-uci-time-management.md`: `best_move_changes`, effort —
  produced in the root part of this walkthrough.
- `design/13-threading-smp.md`: everything here is per-`Worker` state
  (reductions table, `nmp_min_ply`, `tt_move_history`) by construction
  (D23) — no new shared state is introduced by this suite.
- The bench subcommand (D24) is this doc's primary regression instrument.

## 8. Acceptance criteria (D26)

1. Every constant in §3 exists as a named `pub const` (or const table) in
   `src/search/params.rs`; `grep -nE '\b[0-9]{3,}\b' src/search/search.rs`
   shows no unexplained magic numbers.
2. The node executes in exactly the N0–N14 → move-loop → result order of
   §3.4–§3.10; a code-comment cross-reference (`// N10: RFP`) exists per
   step.
3. All Wave A–E commits merged, each with a recorded `bench:` footer and a
   passed SPRT at the bounds in §4 (Wave F may trail until correction
   history lands).
4. Cumulative strength: full-suite engine beats the pre-Wave-A baseline in a
   1000+ game match at 10+0.1 by ≥ +150 Elo (measured, games only, D25).
5. Zugzwang, mate, and repetition suites (§5) pass at their pinned depths;
   `cargo test` green; `cargo clippy --all-targets` clean.
6. Bench is bit-reproducible single-threaded across runs (dither uses node
   counts, not RNG — determinism preserved).
7. No `unsafe` introduced by this suite (D14 — nothing here needs it).

**Shipped-reality deviations (P9 Waves C–G, D32).** Two criteria above are
relaxed by decisions taken after §8 was written; recorded here so the bar
matches the code:

- *Criterion 2 (exact N0–N14 order)* holds for the **consumed** steps, but the
  §3.7 all-node reduction extras (`cutoffCnt`, the ttPv rebate, `tt_capture`)
  and the §3.9 sibling `depth -= 2` are **deferred** (params defined in
  `params.rs`, currently unconsumed — §4 Wave-D/§842 note): in this PSQT-only
  engine they over-reduce deep mate lines past the doc 07 §8(4) bar. They land
  with the P13 eval upgrade. The N6b non-PV TT cutoff runs for **all** nodes
  (in check or not), matching SF's placement of the early cutoff before the
  in-check jump — see §3.11.
- *Criterion 3 (a passed SPRT per Wave A–E commit at the §4 bounds)* is
  superseded for Waves C–G by **D32**: the whole batch is implemented and
  unit-tested first, then gated by a single batched sanity SPRT plus the
  fixed-node proxy SPRT (doc 14 §9.4), rather than one SPRT per heuristic.
  **RESULT (2026-07-12, commit `80dc33d`):** the C–G batch PASSED. First a D33
  multi-persona review loop (4 rounds + a clean confirming round; it caught a
  silent excluded-move TT-value-substitution bug in singular verification),
  then the D32 tripwire — fixed-node proxy SPRT vs the pre-C baseline
  (`0f664e2`): **400 games, +208.68 ± 28.64 Elo (nElo +313.57), LOS 100 %,
  W-L-D 242-27-131 = 76.88 %, LLR 2.95 → H1 accepted.** Verdict KEEP: decisively
  positive at equal nodes, no bug. The deferred §3.7/§3.9 reductions inflate the
  depth-6 bench (350190 → 548978), a known SPSA item for P13 — real-TC will read
  lower than the fixed-node proxy until they are retuned (D32; see
  `docs/strength-log.md`). Waves A/B used the older per-heuristic proxy rhythm.

## 9. Open questions

1. **Param mutability for tuning**: `pub const` now; SPSA tuning will
   eventually want runtime-settable values. Plan: a `tune` cfg feature that
   swaps consts for statics + UCI options, deferred until we actually tune.
   Deviation risk from D20 is nil either way; the open part is the mechanism.
2. **Syzygy**: N7 is specified as plumbing (`max_value` clamp, store shape)
   with no owning doc yet. Decide during Wave E whether TB support gets its
   own doc or an appendix to search-core.
3. **Elo brackets** in §4 are estimates for sequencing/expectation-setting
   only; if a wave SPRT-fails at its bracket, we shrink the commit (one
   heuristic at a time) rather than tune constants ad hoc.
