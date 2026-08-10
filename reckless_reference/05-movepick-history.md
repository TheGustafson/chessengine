# Move ordering: MovePicker and history tables (Reckless)

Source: `movepick.rs` (214 lines), `history.rs` (266 lines), plus the call
sites in `search.rs`, the threat/legality machinery in `board.rs`, the
vectorized attack generators in `setwise.rs`, and the `ThreadData`/
`SharedContext` plumbing in `thread.rs`. Read in full; this note also follows
into every function those two files call. Ideas and constants only — no
Reckless source is reproduced verbatim (AGPLv3; mirrors our D27 policy).

Compare against: `stockfish_reference/03-movegen-movepick-history.md`,
`stockfish_reference/stockfish_search_notes.md` §8-10, our
`design/09-move-ordering-history.md`, `src/movepick.rs`, `src/history.rs`.

## 1. The big picture: one lean stage machine, not four pipelines

Both Stockfish and our engine (design/09 §3.5) run **four separate
pipelines** — Main, Evasion, ProbCut, Qsearch — each its own run of stages
(16 `Stage` variants total: 7 main + 3 evasion + 3 probcut + 3 qsearch).
Reckless collapses all four into **one 5-variant enum** and a single
constructor:

```text
stage machine (5 states, one enum, one pipeline):
    hash-move -> generate-noisy -> good-noisy -> quiet -> bad-noisy
```
(`movepick.rs:10-16`). There is no `EvasionTt`/`ProbcutInit`/`QcaptureInit` at
all. The differences between "modes" are pushed into two runtime knobs
instead of extra stages:

- `threshold: Option<i32>` — `None` for the main search (good/bad noisy split
  uses the per-move dynamic SEE gate, §2 below); `Some(t)` for ProbCut, where
  the gate becomes the fixed `see(mv, t)` check (`movepick.rs:64`,
  constructed at the ProbCut call site `search.rs:639`).
- `skip_quiets: bool`, threaded through every `next()` call by the caller —
  qsearch passes a closure result that is `true` whenever the side to move is
  not in a "losing" position (`search.rs:1290`, `!in_check || !is_loss(..)`),
  so an in-check qsearch node *reuses the same picker* to emit quiet
  evasions, and a not-in-check qsearch node skips straight past `Quiet` to
  `BadNoisy`. **There is no separate evasion pipeline** — check-evasion
  legality is resolved once, upstream, by `Board::is_legal`/generation
  (§4), not by a dedicated movepick stage.

Constructor: `MovePicker::new(tt_move, threshold)` starts at `HashMove` only
if `tt_move.is_present()`, else jumps straight to `GenerateNoisy`
(`movepick.rs:29-39`) — the same branchless-skip idea as SF's
`stage += !(ttm usable)`, just expressed as an `if` over the initial field
rather than pointer arithmetic on an enum. Note this construction-time check
is weaker than SF/ours: it only tests *presence*, not
`board.pseudo_legal(tt_move)` — the actual legality check happens lazily
inside `next()`'s `HashMove` arm (`td.board.is_legal(self.tt_move)`,
`movepick.rs:49`), one call later than SF's up-front test. Functionally
equivalent, just relocated.

### 1.1 Dispatch style: a waterfall of `if`s, not `loop { match }`

`next()` (`movepick.rs:45-107`) is a flat sequence of
`if self.stage == Stage::X { ...; self.stage = Y; }` blocks that fall through
to the next `if` in the same call when a stage produces nothing to emit yet
(e.g. `HashMove` falling into `GenerateNoisy` falling into `GoodNoisy`).
There's no `bump()`/`continue` dance (our design/09 §3.6) — the whole method
re-enters at the top on every call and just skips the `if`s whose stage has
already passed. Cheap (enum discriminants), and arguably reads flatter than a
`match` with fallthrough, at the cost of re-testing already-passed stages
every call (negligible — `Stage` is a one-byte discriminant, four extra `==`
per call at worst).

### 1.2 Selection is O(n) per pick, not a partial insertion sort

Where SF/we run `partial_insertion_sort` once per stage (captures: full sort;
quiets: sorted prefix down to a depth-scaled `limit`, design/09 §3.8), and
then just walk the sorted region, Reckless's `get_best_entry`
(`movepick.rs:109-120`) does a **linear scan for the maximum, every single
`next()` call**, then removes it via swap-remove
(`MoveList::remove` → `ArrayVec::swap_remove`, `types/movelist.rs:125-127`).
That's O(n) work per emitted move and O(n²) over a full node instead of one
O(n log n)-ish sort — but it needs zero extra bookkeeping (no `sorted_end`,
no `limit` constant, no region-index juggling) and, for the small `n` typical
of a chess position (usually well under 60 moves), is unlikely to be the
bottleneck. It is a genuine simplicity-over-asymptotics trade **worth
measuring**, not assuming, before porting (§6). One side effect worth noting:
ties break toward the *last*-seen equal-scoring entry (`entry.score >=
best_score` keeps advancing `best_index`), the opposite tie-break from a
stable forward scan.

### 1.3 Root-only re-scoring after every completed root subtree

Both the `GoodNoisy` and `Quiet` stages carry a `NODE::ROOT` branch that
re-invokes the *entire* scoring pass over the remaining buffer before
returning a move (`movepick.rs:70-72`, `90-91`):

```text
at a ROOT node (compile-time node-type flag), before yielding each pick:
    re-run the whole noisy (resp. quiet) scoring pass over the unpicked tail
```

At first glance this looks like dead work — nothing in a single `next()` call
changes the board or the histories between two picks. But at the **root**,
each pick is followed by a *complete, recursive search of that move's whole
subtree* before the picker is asked for the next one — and that subtree
search freely updates `td.{noisy,quiet,pawn}_history`/`continuation_history`
millions of times. So by the time the second root move is requested, the
history tables the *first* root move's score was computed from are stale.
Re-running `score_noisy`/`score_quiet` on the *remaining* candidates refreshes
their order using history that just got updated by the sibling subtree that
was searched a moment ago. This is a cheap (`O(remaining moves)`), precise fix
for a real staleness problem that plain "sort root moves once by previous
iteration's score" schemes (the SF/our-design style) never address *within*
one iteration. Neither Stockfish nor our design/09 do this.

### 1.4 A move-count cap fused into the good/bad noisy split

The good/bad capture gate is normally a pure SEE test. Reckless adds one more
condition that has no SF/our-design analogue (`movepick.rs:65`):

```text
relegate a candidate capture to bad-noisy when EITHER
    the SEE test against the picker's threshold fails            (the usual gate)
OR  the hash move is a quiet AND >2 noisies were already accepted (the extra gate)
```

If the *hash move itself is quiet* (a strong signal that captures are not
what's good here) and three or more captures have already been accepted as
"good" this node, every further capture is shoved into `bad_noisy`
regardless of its own SEE value. It's a move-count/LMP-flavored idea folded
directly into the picker rather than into the search loop — capped generation
of "trustworthy" captures when the position's own best-known move says
captures aren't the point.

### 1.5 The dynamic SEE threshold, and the in-check attacker bias

The good/bad split's default threshold (used whenever ProbCut hasn't fixed
one) is **score-dependent**, same idea as SF's `-value/18` but recomputed per
move from its own just-computed ordering score (`movepick.rs:64`):

```
threshold = -entry.score / 47 + 116
```

A capture with a strong MVV+history score tolerates losing a bit more
material before being demoted; the constant (47, vs. SF's 18) and the +116
offset are unique to Reckless's tuning.

`score_noisy` (`movepick.rs:128-141`) adds a large flat, **attacker-type**
bonus whenever the side to move is in check:
```
+ (200000 − 20000·pt) · in_check
```
(`pt` = the *moving* piece's type index, Pawn=0..King=5). Since this term
dwarfs every history/MVV term, it reorders in-check noisy moves purely by
"cheapest attacker first" — a least-valuable-attacker tiebreak that only
matters for check evasions, achieved inside the *same* noisy-scoring function
used everywhere else, instead of a dedicated evasion scorer
(`victim_value + (1<<28)`, our design/09 §3.7) that ignores the attacker.

## 2. Quiet scoring: threat/offense/king-safety terms baked into one formula

`score_quiet` (`movepick.rs:143-213`) computes, **once per scoring pass**
(not once per move), four bitboard families from the live position:

- `threatened[pt]`: squares attacked by an enemy piece *cheaper* than `pt` —
  structurally identical to SF's/our `threats_by_lesser` (cumulative union:
  minors are threatened by pawns, rooks by minors-or-pawns, queens by rooks
  -or-lower). Same idea, different name.
- `offense[pt]` — **squares where moving a piece of type `pt` there would let
  it attack a *vulnerable* enemy piece** (a piece of higher or equal value
  that isn't itself defended by a threat). Computed with the vectorized
  `*_attacks_setwise` generators (`setwise.rs`, §5) run against the enemy's
  vulnerable-piece bitboards, masked off the squares we're already attacked
  on. This is the "create a threat" counterpart to `threatened`'s "avoid a
  threat" — **SF and our design/09 §3.7 only score the escape/entry side;
  Reckless also rewards the offense side**, with its own per-piece-type bonus
  array `[0, 8854, 8170, 14051, 20357, 0]` (P/N/B/R/Q/K) rather than SF's flat
  `20 × PieceValue[pt]` multiplier for both directions.
- `wall_pawns` — pawns still shielding our own king on its home row
  (`Bitboard::HOME_ROWS`) — moving one costs a flat `−4494`. Not present in
  SF or our design at all: a king-safety-flavored quiet-history term.
- the check bonus `+10723` for moving to a `checking_squares(pt)` square, with
  **no SEE gate** — SF/our design require `see_ge(m, −75)` before granting
  the (larger, 16384) check bonus; Reckless grants it unconditionally.

Full formula (constants as found, weights `/1024`):
```
value = 1763·quiet_history[stm][from][to]/1024
      + 1024·pawn_history[pawn_key][piece][to]/1024
      + 1614·cont(1) /1024 + 1066·cont(2) /1024
      + 1086·cont(4) /1024 + 1051·cont(6) /1024
      + escape[pt]      if from ∈ threatened[pt]
      + 10723           if to ∈ checking_squares[pt]
      − 8875            if to ∈ threatened[pt]
      + 3446            if to ∈ offense[pt]
      − 4494            if from ∈ wall_pawns
```
(`movepick.rs:201-211`). Continuation-history offsets read are **{1,2,4,6}**
— SF/our design read {1,2,3,4,6} (offset 3 dropped here too, see §3).

Noisy formula (`movepick.rs:136-139`):
```
value = 14232·victim_value/1024 + noisy_history[piece][to][victim][to_threatened]
      + 4558   if queen promotion
      + (200000 − 20000·pt)   if in_check      // §1.5
```
Piece values used for the MVV term are Reckless's own SEE/eval values — about
half Stockfish's (`Pawn 109, Knight 403, Bishop 435, Rook 679, Queen 1242,
King 0`, `types/piece.rs:138-148`) — so the ratio to history bonuses differs
even where the *shape* of the formula matches SF's `7×victim + history`.

## 3. History tables: context baked into storage, not into the formula

The single biggest structural difference from both SF and us: several tables
carry an extra **threat-context axis in their storage**, rather than
expressing threat-awareness as an additive scoring bonus computed from
separately-maintained bitboards.

| Table | Shape (`history.rs`) | D (gravity bound) | Init | Size |
|---|---|---|---|---|
| `QuietHistory` (the only "butterfly"/main table) | `[stm][from∈threats][to∈threats][from][to]` | 8192 | **0** | 4×64KiB=256 KiB×~ (4 sub-tables) |
| `NoisyHistory` | `[piece][to][captured_type][to∈threats]` | 12800 | **0** | 12×64×7×2×2B ≈ 21 KiB |
| `PawnHistory` | `[pawn_key & 511][piece][to]` | 8192 | **0** | 512×13×64×2B ≈ 852 KiB |
| `ContinuationHistory` | `[in_check][capture][piece13][to] → [piece13][to]` | 15320 | **0** | ~5.5 MiB, `mmap`+hugepages |
| `CorrectionHistory` (pawn / non-pawn-white / non-pawn-black) | `[bucket(16)][stm][key & 65535]`, **atomic i16, NUMA-shared** | 14605 | **0** | 3×2 MiB |
| `ContinuationCorrectionHistory` | same shape as `ContinuationHistory`, per-thread | 16418 | **0** | ~5.5 MiB, `mmap`+hugepages |
| low-ply history | **absent** | — | — | — |
| tt-move reliability scalar | **absent** | — | — | — |
| killers / countermoves | **absent** (matches D17) | — | — | — |

Everything zero-inits (`zeroed_box`/`HugeBox::new_zeroed`, `history.rs:257-266`,
`9-75`) — **no negative-fill priors anywhere**. SF and our design/09 §3.4
deliberately seed tables to small negative values (`MAIN_INIT = −5`,
`CAPTURE_INIT = −699`, `CONT_INIT = −552`, `PAWN_INIT = −1262`,
`CORR_INIT = −6`, …) so an untried move doesn't look artificially competitive
next to one with any real negative signal. Reckless has no such priors; the
scoring/pruning constants (§2, and the LMP/FP/history-pruning gates in
`search.rs:774-832`) are evidently tuned around a zero baseline instead. This
is a load-bearing, easy-to-miss difference: porting any Reckless bonus/malus
constant verbatim onto a negative-initialized table would be wrong.

**Contextual butterfly history.** `QuietHistory` is `[stm][from_threatened]
[to_threatened][from][to]` (`history.rs:82-100`) — four independent 64×64
tables per side instead of one. SF/our design instead compute the escape/entry
signal as a hand-tuned scoring-time bonus (`20 × PieceValue[pt] × (from∈threat
− to∈threat)`, design/09 §3.7) layered on top of a single from-to table. In
Reckless the *learned statistic itself* is threat-context-conditioned — the
system discovers, per side-to-move and per threat context, how good a
from→to move is, rather than being told a fixed multiplier. `NoisyHistory`
does the analogous thing on one axis (`to_threatened`,
`history.rs:135-157`). One plausible reading: this substitutes for SF/our
`low_ply` and `tt_move`-reliability tables entirely (both absent here) —
Reckless is leaner in *table count* (6 vs. our design's 8) but denser in
*context per table*.

**The gravity operator is identical in substance** to ours/SF's — one
generic bounded update, `entry += clamp(bonus,−D,D) − entry·|clamp(bonus,−D,D)|/D`
(`history.rs:77-80`; our `StatEntry::update`, design/09 §3.3, is the same
formula). Confirms our implementation direction is right; Reckless just
free-functions it (`apply_bonus::<const MAX: i32>`) instead of wrapping it in
a newtype the way our `StatEntry<const D: i32>` does — our version is the
more idiomatic (and equally zero-cost) choice, not a thing to change.

### 3.1 Continuation history: fewer offsets, and no consistency multiplier on write

Scoring reads offsets **{1, 2, 4, 6}** (`movepick.rs:203-206`) — SF/our
design read {1, 2, 3, 4, 6} (offset 3 dropped entirely here, not just
unread-but-written as SF's offset 5 is). The *write* side
(`update_continuation_histories`, defined `search.rs:1415-1422` and called
from every node-result/TT-hit/refutation site above) is dramatically
simpler than SF's/our design/09 §3.9: it applies the **same bonus, unweighted,
to all four present offsets** —
```text
for each offset in {1, 2, 4, 6}:
    if a move was actually played at ply - offset:
        gravity-update that ply's conthist entry for (piece, to) by the
        ONE shared bonus (no weight, no scaling by offset)
```
— with no per-offset weight table, no growing "consistency multiplier" for
already-agreeing entries, no in-check truncation to the first two offsets,
and no flat +71 add-on for offset 1. SF's version (and our design/09 §3.9)
carries all four of those refinements. Whether they're worth their
complexity given Reckless's results (top-2 engine) is an open, testable
question — see §6.

### 3.2 Correction history: fifty-move bucketing, no minor-piece key, NUMA-shared

`eval_correction`/`update_correction_histories` (`search.rs:1364-1413`) sum
**three** correction terms (pawn, non-pawn-white, non-pawn-black) plus a
continuation-correction term from stack offsets 2 and 4, then divide by a
flat `64`:
```
correction = (pawn + non_pawn_white + non_pawn_black + cont_corr(2) + cont_corr(4)) / 64
```
(`search.rs:1364-1383`). Notably **absent**: a minor-piece correction key. SF (and our design/09
§3.10, §9 open question) also track a `minor_key`-indexed correction table;
Reckless doesn't. Also distinctive: every correction-history read/write is
indexed not just by `[stm][key]` but by a **fifty-move-clock bucket**,
`(fiftymove_clock.saturating_sub(8) / 8).min(15)` — 16 buckets
(`board.rs:75-77`, `CorrectionHistory` shape `[bucket][stm][key]`,
`history.rs:159-195`). The idea: a position 90 plies from any pawn move or
capture has a different correction profile than a fresh one (fifty-move-rule
proximity correlates with drawishness/simplification), so the correction
table is partitioned by it. Neither SF nor our design bucket by the
fifty-move clock at all — this looks like a genuinely novel, cheap axis
worth testing on its own.

The bonus is asymmetric and clamped, not symmetric like our design's ±256:
```
bonus = (148 * depth * diff / 128).clamp(-4678, 2496)
```
(`search.rs:1389`, where `diff = best_score − eval`; the whole function is
`search.rs:1385-1413`). And architecturally,
`CorrectionHistory` is the **one** table in Reckless's whole stack that is
thread-shared: `AtomicI16` entries behind `Arc<NumaReplicated<
SharedCorrectionHistory>>` (`thread.rs:165-172`, `history.rs:159-195`),
replicated per NUMA node (`numa.rs`). `ContinuationCorrectionHistory`, by
contrast, is per-thread like everything else. So Reckless's answer to "what
should be NUMA-shared" (design/09 §9 open question 5, deferred by us to the
SMP phase) is: *only* the two cheapest, smallest, hash-indexed correction
tables — not pawn history, not continuation history, not the main quiet/noisy
tables.

### 3.3 No per-`go` decay; only a full `ucinewgame` reset

SF and our design/09 §3.4 decay the main/butterfly history by `789/1024`
and refill low-ply history every `go` (`Histories::new_search`). Reckless has
no equivalent at all: `ThreadPool::clear()` (called only on `ucinewgame`,
`uci.rs:75/177`) fully drops and reconstructs every `ThreadData`, which
zero-reinitializes every table via `Default`; `SharedCorrectionHistory` gets
an explicit `.clear()` at the same point (`uci.rs:181-184`). Between
successive `go` commands in the same game, **every history table simply
persists and keeps accumulating** — no decay step exists in `search.rs`,
`thread.rs`, or `threadpool.rs`. Given the gravity operator is
self-saturating, an un-decayed table can't blow up, but it can carry
stale-game-phase signal (e.g. middlegame quiet-history weight lingering into
an endgame) that SF's per-`go` decay is specifically designed to fade. This
is a clean, cheap A/B to run ourselves (§6).

### 3.4 `HugeBox`: raw `mmap` + `MADV_HUGEPAGE` for the two ~5.5 MiB tables

`ContinuationHistory` and `ContinuationCorrectionHistory` (the two
largest, most randomly-indexed tables — indexed by the *previous move's*
`(in_check, capture, piece, to)`, effectively unpredictable from the
allocator's point of view) are backed by a hand-rolled `HugeBox<T>`
(`history.rs:9-75`): `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` followed by
`madvise(..., MADV_HUGEPAGE)` on Linux (falling back to a plain zeroed
`alloc` elsewhere), with matching `munmap` in `Drop`. The stated motivation
(inferable from the choice, not documented in-source) is TLB pressure: a
~5.5 MiB table walked at effectively-random 1536-byte-slab granularity on
every quiet move is a page-miss machine without huge pages. Everything else
(the smaller `QuietHistory`/`NoisyHistory`/`PawnHistory`) uses the ordinary
`zeroed_box` heap path.

## 4. What feeds the picker: threats computed once per node, reused for legality and ordering

Reckless's `Board` maintains `all_threats`/`piece_threats: [Bitboard; 6]`/
`checking_squares: [Bitboard; 6]` as part of its per-node state, refreshed by
`Board::update_threats` (`board.rs:431ff`, also computes pins in the same
pass) using the vectorized `*_attacks_setwise` bulk generators
(`setwise.rs:1-137`; AVX2 shuffle-based sliding-piece variants behind
`target_feature = "avx2"`, scalar pop-lsb fallback otherwise). This isn't
scoped to move-ordering: `Board::is_legal` (`board.rs:349ff`) — Reckless's
**complete** legality oracle, called for
*every* move the search is about to play, not just the "specials" SF/our
design lazily re-check — reads `all_threats()` directly to validate king
moves, and `update_threats` is therefore paid for on every node regardless of
whether move ordering needs it. The picker's `threatened`/`offense` arrays in
`score_quiet` are just a second consumer of an already-computed asset. One
practical consequence: `MovePicker::next()` never needs to call a legality
predicate on the moves it emits from the noisy/quiet stages at all — only the
injected `tt_move` gets an explicit `is_legal` check (`movepick.rs:49`) — because
the generators themselves are trusted to emit only legal moves. This is a
different architectural cut from SF's/our design's "pseudo-legal generation +
lazy legality filter" split (design/09 §7 references `design/05-movegen.md`),
not a move-ordering trick per se, but it's *why* Reckless's picker code has no
legality-filtering logic in its hot loop at all.

## 5. Update call sites (search.rs) worth citing

- **TT-hit quiet bonus** (a quiet TT move that would fail high on re-search):
  `quiet_bonus = min(190·depth − 81, 1691)`, `cont_bonus = min(96·depth − 73,
  1206)`, gated on `stack[ply-1].move_count < 4` (`search.rs:395-401`).
- **Eval-diff quiet nudge** (two plies of quiet moves, static eval trending
  down): `bonus = clamp(812·−(eval+stack[ply-1].eval)/128, −144, 324)` applied
  to the *parent's* move via `quiet_history.update(prior_threats, !stm, ..)`
  (`search.rs:490-502`) — same idea as SF's eval-diff correction-history
  nudge, but here it feeds the *quiet* butterfly table directly rather than a
  separate correction table.
- **Node-result burst** (`search.rs:1062-1119`): six depth-scaled
  bonus/malus constants (`noisy_bonus/malus`, `quiet_bonus/malus`,
  `cont_bonus/malus`), each `min(a·depth, cap) − b − c·(cut_node or
  moves_tried)` — i.e. the "breadth" and "cut-node" adjustments SF applies as
  *multipliers after the fact* (design/09 §3.9's `bonus *= (1 + count/256)`)
  are baked as **linear subtractions into the base formula** here instead.
  The malus applied to each non-best tried quiet/noisy move is scaled by a
  **closed-form quadratic decay** computed independently per list index `i`,
  not SF's per-step multiplicative recurrence:
  ```
  scale(i) = 1024³ / (1024 + 45·i)²
  ```
  (`search.rs:1085-1096`, reused for quiet, pawn, *and* continuation malus).
  No loop-carried decay state — each tried move's scale is a pure function of
  its position in the list, so it could be computed in parallel/out of order
  if that ever mattered.
- **Refutation malus / fail-high-parent nudge / prior-noisy bonus**
  (`search.rs:1110-1150`) — structurally parallel to SF's/our design's
  equivalents (design/09 §3.9 "external write sites"), with Reckless's own
  constants; not re-derived here in full, just flagged as present and at
  those lines.
- **ProbCut** reuses the *same* `update_*` machinery — no ProbCut-specific
  history writes were found (unlike SF, which doesn't write history from
  ProbCut either, so this is a non-difference, noted for completeness).
- **`make_move`** (`search.rs:1424-1435`) stamps both `stack[ply].conthist`
  and `stack[ply].contcorrhist` as raw `*mut PieceToHistory<i16>` pointers
  (wiring at `search.rs:1428-1431`; `subtable_ptr` defined at
  `history.rs:205-209`/`235-239`) — Reckless *does* use
  Stockfish's original raw-pointer-on-the-stack idiom (unsafe, single-threaded
  per `ThreadData`), where our design/09 §3.2 deliberately replaced it with
  the borrow-checker-friendly `ContHistId` flat index specifically to stay
  `unsafe`-free (D14). This is a place where our design is already the more
  idiomatic Rust choice, not a gap to close.

## 6. Distinctive vs Stockfish and vs us — summary

| Aspect | Stockfish / our design/09 | Reckless |
|---|---|---|
| Pipelines | 4 (Main/Evasion/ProbCut/Qsearch), 16 stages total | 1 pipeline, 5 stages, mode = `threshold: Option` + caller-driven `skip_quiets` |
| Selection | partial insertion sort (captures: full; quiets: depth-scaled prefix) | linear max-scan per pick (O(n) per move) |
| Table count | 8 (main, low_ply, capture, cont, pawn, corr-unified, cont_corr, tt_move) | 6 (no low_ply, no tt_move reliability) |
| Threat-awareness | additive scoring bonus (`20·PieceValue·(escape−entry)`) | baked into table *storage* (extra threat-context axes on quiet/noisy tables) + a symmetric "offense" bonus SF lacks |
| Table init | negative-biased fill per table | all zero |
| Per-`go` decay | 789/1024 butterfly decay + low-ply refill | none — persists whole game, only `ucinewgame` resets |
| Correction history keys | pawn, minor, non-pawn(w/b), cont(2,4) | pawn, non-pawn(w/b), cont(2,4) — **no minor key**; extra fifty-move-clock bucket axis |
| Correction history sharing | per-thread (D23) now; SF shares per NUMA node | pawn/non-pawn shared+atomic across NUMA nodes; everything else (incl. cont-correction) per-thread |
| Continuation-history offsets | 1,2,3,4,6 (scoring); weighted + consistency multiplier (write) | 1,2,4,6 (both); flat unweighted bonus (write) |
| Killers/countermoves | absent (D17) | absent (same call) |
| Large-table allocation | plain boxed arrays | `mmap` + `MADV_HUGEPAGE` for the two ~5.5 MiB continuation tables |
| Root move (re)ordering | sorted once per ID iteration by previous score | re-scored *within* one root pass after every completed sibling subtree |
| ContHist stack plumbing | our design: safe `ContHistId` flat index (D14-motivated); SF: raw pointer | raw pointer (like SF) — `unsafe` |

## Adapt to us

Ordered roughly by impact/effort ratio; each is independently testable via
our fixed-node-proxy SPRT (design/09 §5, D25) without disturbing the rest of
the stack.

1. **Root-move re-scoring after each completed sibling subtree** (§1.3).
   *Touches:* `design/09-move-ordering-history.md` §3.6 (root pipeline), our
   `src/movepick.rs` main-pipeline `next_move` root branch; phase **P9/P10**
   (current pruning-wave work, cheap to slot in alongside it). *Effort:*
   small — call the existing `score_captures`/`score_quiets` scoring helpers
   again from the root-branch arms before returning, guarded by a
   `NODE::ROOT`-equivalent const generic we already have. *Impact:* plausible
   free win — root ordering currently only reflects the *previous iteration's*
   score, never mid-iteration history drift from already-searched sibling
   root moves; likely most valuable at fast time controls / early iterations
   before the TT and histories are warm.

2. **A fifty-move-clock (or similar drawishness-proxy) bucket axis on
   correction history** (§3.2). *Touches:* `design/09-move-ordering-history.md`
   §3.4/§3.10, `design/decisions.md` D9 (key definitions) if a new bucket
   dimension is added; phase **P9 tail / correction-history tuning**.
   *Effort:* medium — adds one more index dimension to `corr`
   (`[bucket][key][stm]` instead of `[key][stm]`), a `fiftymove_clock_bucket()`
   accessor on `Board`, and updated `correction_value`/
   `update_correction_history` signatures (doc 09 §3.10). *Impact:* unverified
   but cheap to SPRT in isolation; a plausibly real, currently-untested
   correlation (fifty-move proximity ≈ simplification/drawishness) neither SF
   nor we currently model.

3. **Symmetric "offense" term alongside our existing threat-escape/entry
   term** (§2). *Touches:* `design/09-move-ordering-history.md` §3.7 (quiet
   scoring formula) and the `threats_by_lesser` helper (§9 open question 6:
   "if eval later needs the same threat sets, promote to a `Board` method" —
   this is exactly that trigger); phase **P9 Wave B/C** (ordering
   refinement) or a small dedicated wave. *Effort:* medium — needs the
   "vulnerable enemy piece" bitboards (which pieces aren't already
   defended/threatened) computed once per scoring pass, plus a new tuned bonus
   array; can reuse our existing `attacks` setwise-equivalents (design/03) if
   we build them, or a plain per-piece-type pop-lsb loop first and measure
   before reaching for vectorization. *Impact:* untested but directly
   analogous to a term SF itself lacks and Reckless (top-2 engine) carries —
   worth an isolated SPRT.

4. **Re-examine whether our per-`go` butterfly decay is pulling its weight**
   (§3.3). *Touches:* `design/09-move-ordering-history.md` §3.4
   (`Histories::new_search`); phase **P9/P10** (cheap, standalone experiment).
   *Effort:* tiny — toggle the decay off behind a feature flag or SPSA-tune
   its coefficient toward 1024/1024 (no-op) and measure. *Impact:* unknown
   sign; Reckless's top-2 result with *no* such decay at all suggests it may
   be safe to weaken or drop given our bonus/malus formulas, but this is a
   "verify, don't copy" candidate specifically because removing a
   stabilizing mechanism without re-tuning the rest is exactly the kind of
   change that can look free and not be.

5. **`mmap` + transparent-hugepage backing for our largest, most randomly
   indexed tables** (§3.4). *Touches:* `design/09-move-ordering-history.md`
   §3.4 (`Histories::new`, the "never `Box::new_zeroed`, `unsafe fn`" note)
   and `design/decisions.md` D14 (unsafe budget) — this would need a new,
   explicitly-scoped unsafe carve-out, so it's a decisions-first change, not
   a drop-in. Phase: **P11 (lazy SMP)**-adjacent, since the TLB-pressure
   argument gets stronger with more concurrent threads walking the same
   table shape. *Effort:* medium (a small `HugeBox`-equivalent + `Drop` +
   platform fallback, mirroring the pattern here) but requires the D14
   conversation first. *Impact:* unverified — profile our continuation
   history's TLB/cache-miss rate (design/09 §6 already calls for this
   measurement) before deciding it's worth the unsafe surface.

6. **Consider whether the continuation-history write side needs its
   per-offset weights and consistency multiplier at all** (§3.1). *Touches:*
   `design/09-move-ordering-history.md` §3.9
   (`update_continuation_histories`); phase **P9** (history-stack tuning).
   *Effort:* small — an isolated SPRT with the weight table replaced by a
   single flat bonus (matching Reckless) and the consistency-multiplier
   dropped. *Impact:* unknown; this is a complexity-reduction candidate, not
   an obviously-better one — Reckless's simplicity here could be a
   simplification that works *because of* something else it does
   differently (e.g. the contextual butterfly axis, item below), so it
   should be tested holistically, not assumed portable in isolation.

7. **NOT recommended to port as-is: negative-fill table initialization
   removal.** Flagged explicitly as a *don't*, not an idea: our
   negative-biased init (design/09 §3.4) and Reckless's all-zero init are
   both internally consistent with the rest of each engine's tuned constants
   (§3). Changing our init without re-tuning every bonus/malus/pruning margin
   that assumes it would be a silent, hard-to-attribute regression — recorded
   here so a future reader doesn't "simplify" it away by analogy to this
   note.

8. **Contextual (threat-indexed) quiet/noisy butterfly axes as a replacement
   for the hand-tuned escape/entry multiplier** — the deepest, riskiest, and
   potentially highest-ceiling idea here (§3, table). *Touches:*
   `design/09-move-ordering-history.md` §3.4 (table catalogue — would resize
   `main`/`capture` by a `×4`/`×2` threat-context factor) and §3.7 (would
   *delete* the current `20·PieceValue·(escape−entry)` term, replacing a
   formula with learned context). Phase: **post-P9**, a dedicated ordering
   experiment, not a slot-in — it changes table shapes (bench-signature
   affecting, D24) and needs its own SPRT wave, ideally after Wave B/C
   settle. *Effort:* large (new table shapes, new indexing, full
   re-tune of dependent pruning margins that read `history` as a signal —
   `search.rs:774-832`'s LMP/FP/HP/SEE-pruning gates all consume the *same*
   history value this would reshape). *Impact:* highest-variance entry on
   this list — it's the single structural choice most plausibly responsible
   for Reckless matching SF-class strength with two fewer tables; also the
   most expensive to validate honestly.

## File:line index

- `movepick.rs`: `Stage` 10-16; `MovePicker` fields 18-26; `new` 29-39;
  `next` 45-107 (HashMove 46-52, GenerateNoisy 54-59, GoodNoisy 61-86 incl.
  the tt-quiet count cap at 65 and root re-score at 70-72, Quiet 88-97 incl.
  root re-score at 90-91, BadNoisy 99-106); `get_best_entry` 109-120;
  `remove_tt` 122-126; `score_noisy` 128-141; `score_quiet` 143-213 (threat
  arrays 156-195, main loop 197-212).
- `history.rs`: `HugeBox` 9-75; `apply_bonus` 77-80; `QuietHistory` 82-106;
  `PawnHistory` 108-133; `NoisyHistory` 135-157; `CorrectionHistory` 159-195
  (shape comment + bucket-dimensioned field 160-161, `get`/`update` bucket
  param 170-178); `ContinuationCorrectionHistory` 197-225; `ContinuationHistory`
  227-255; `zeroed_box` 257-266.
- `search.rs`: TT-hit quiet bonus 395-401; eval-diff quiet nudge 490-502;
  ProbCut movepicker use 632-685; singular-extension correction-value use
  710-716; main-loop history read 767-772; LMP/FP/BNFP/HP/SEE pruning gates
  774-832; LMR formula 843-915; FDS formula 916-963; node-result update burst
  1062-1119 (quadratic malus scale 1085-1096); refutation malus 1110-1113;
  post-cutoff conthist bonus 1115-1118; fail-high/low parent nudge
  1121-1151; `eval_correction` 1364-1383; `update_correction_histories`
  1385-1413; `update_continuation_histories` 1415-1422 (free fn, distinct
  from the history-module methods); `make_move` 1424-1435 (conthist/
  contcorrhist pointer wiring at 1428-1431); qsearch movepicker loop
  1285-1348.
- `board.rs`: `attackers_to` 340-347; `is_legal` 349ff; `update_threats`
  431ff (also computes pins in the same function, 463-471+);
  `fiftymove_clock_bucket` 75-77.
- `setwise.rs`: `pawn_attacks_setwise` 14-21; `knight_attacks_setwise` 24-38;
  `bishop_attacks_setwise`/`rook_attacks_setwise` (scalar + AVX2 variants)
  40-137.
- `types/moves.rs`: `is_quiet`/`is_noisy` bit-trick 64-72 (kind low-3-bits
  `> Castling`); `capture_sq` 90-92 (`to ^ (is_en_passant · 8)`).
- `types/movelist.rs`: `MoveEntry`/`MoveList` 7-142; `remove` = swap-remove
  125-127.
- `thread.rs`: `SharedCorrectionHistory` 144-153; `ThreadData` struct 213,
  history fields 223-227; `ThreadData::conthist` helper 294-296.
- `types/piece.rs`: `PieceType::value` 138-148.
