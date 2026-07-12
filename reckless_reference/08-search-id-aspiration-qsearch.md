# Reckless search — iterative deepening, aspiration, root/PV, qsearch, Stack

Source studied: Reckless (Rust, edition 2024, AGPLv3) at the clone under
`reckless_reference/Reckless`, tag/branch `v0.10.0-dev`. Files read in full:
`src/search.rs` (1449 lines), `src/stack.rs` (80 lines), plus supporting reads
of `src/thread.rs` (RootMove, PrincipalVariationTable, PlyArray, SharedContext),
`src/movepick.rs` (whole file, 215 lines), `src/board/movegen.rs` (unified
generator), `src/transposition.rs::write` (replacement/`force` semantics),
`src/time.rs` (TimeManager — only as far as it explains call sites in
`search.rs`), and `src/types/{score,piece}.rs` (value constants). Every
citation below is `reckless_reference/Reckless/src/<file>:<line>` — **no
Reckless source is reproduced**; formulas are re-derived in prose/pseudocode
and, where a single constant matters, quoted inline as a bare number.

Read alongside `stockfish_reference/stockfish_search_notes.md` (SF dev
`eca43a97`) throughout — this note assumes that one and calls out deltas
rather than re-deriving shared ground. Compared against our own
`design/07-search-core.md` (Reviewed v1) and the current
`src/search/mod.rs`/`src/search/stack.rs`.

**Top-line framing**: Reckless's search skeleton is recognizably
Stockfish-shaped (PVS negamax, node-type markers, TT-gated cutoffs, the same
three-call reduced/full/PV move loop), but nearly every subsystem this note
covers has been **simplified, unified, or re-tuned** relative to SF, and a
few pieces are simplified *relative to our own design doc*, which currently
tracks SF closely. The recurring theme: Reckless collapses SF's several
special-cased staging pipelines and blend formulas into one reusable
mechanism used everywhere (one `MovePicker` state machine, one `lerp` helper,
one generic per-ply array type), at the cost of a few accepted, deliberate
inaccuracies (no qsearch stalemate detector, a single-function search node
with everything inlined).

---

## 1. Big picture

```
ThreadPool::search (thread setup, root move list, per-thread clones)
  └─ search::start(td, report, thread_count)     (search.rs:55)
       └─ for depth in 1..MAX_PLY:
            for pv_idx in 0..multi_pv:
                 aspiration loop
                     └─ search::<Root>(td, alpha, beta, depth', cut_node=false, ply=0)
                          └─ search::<PV/NonPV>(...)   (search.rs:300, recursive PVS)
                               └─ qsearch::<PV/NonPV>(...) at depth <= 0   (search.rs:1178)
```

Node types (`search.rs:32-53`): a `NodeType` trait with `const PV: bool` /
`const ROOT: bool`, three zero-sized marker structs `Root`/`PV`/`NonPV` — the
same const-generic-flavored monomorphization idea as SF's template parameter
and as our own design's sealed-trait `Root`/`Pv`/`NonPv` markers (design/07
§3.7.1). No separate `all_node` binding exists as a named local; the
`!NODE::PV && !cut_node` expression is just written inline at each of its two
or three uses.

Unlike SF (and unlike our design's plan, doc 09), **there is exactly one
`search` function and one `qsearch` function** — no helper split into
"probe/cutoff/move-loop" stages. Everything from mate-distance pruning
through the TT store lives in one ~880-line function body
(`search.rs:300-1176`). This is a legitimate, working alternative to our
design's explicit "ordered node skeleton with numbered steps" framing
(design/07 §3.7.2) — Reckless gets the same ordering for free from top-to-
bottom control flow and early returns, at the cost of a much longer function
than idiomatic Rust style would usually recommend split up. We already
diverge from this (our `search_at`/`qsearch_at` helpers), which is arguably
the more maintainable choice at our size.

---

## 2. The iterative-deepening driver — `search::start` (search.rs:55-298)

### 2.1 Loop shape and MultiPV banding

Depth loop: `for depth in 1..MAX_PLY as i32` (search.rs:78), guarded by a
`Limits::Depth` check for the main thread only (`td.id == 0`, search.rs:79-85)
— same idea as SF's main-thread-only depth limit, same idea as our design
§3.5. Per iteration: `sel_depth = 0`, `root_depth = depth`,
`best_move_changes = 0`, snapshot every root move's `previous_score`
(search.rs:87-96) — **no `previous_pv` snapshot at all**: Reckless doesn't
carry a `previous_pv` field on `RootMove` (§3 below), only `previous_score`.

MultiPV banding (search.rs:104-112) groups root moves by `tb_rank` into
`[pv_start, pv_end)` windows exactly like SF's Syzygy-rank grouping, but here
it's unconditional (Syzygy is feature-gated; `tb_rank` defaults to `0` for
every move without the `syzygy` feature, so in the common build this
degenerates to one band covering all root moves — same shape either way).

### 2.2 Aspiration windows (search.rs:98-178)

`delta` and `reduction` are declared **once per depth iteration**, *outside*
the `pv_idx` loop (search.rs:98-99), not freshly per line as in SF
(`delta = 5 + threadIdx%8 + |mss|/10588`, recomputed per `pv_idx`). Then,
per `pv_idx`:

```
delta += average[pv_idx]² / 26394        // grows delta, doesn't reset it
alpha  = max(average[pv_idx] - delta, -INF)
beta   = min(average[pv_idx] + delta,  INF)
```

Two things fall out of this that are worth flagging precisely before
copying:

- **The base constant is `23`** (search.rs:98), not SF's `5`, and the widening
  term uses the **square** of the plain average score (`average² / 26394`),
  not SF's `|meanSquaredScore| / 10588` (an independently-maintained EMA of
  `v·|v|`) — Reckless has no `mean_squared_score` field on `RootMove` at all
  (§3). Same *shape* (bigger average score ⇒ wider initial window), simpler
  state (no extra EMA to maintain), different tuned scale.
- **`delta` (and `reduction`) carry across MultiPV lines within one
  iteration.** After line 0's aspiration loop finishes — possibly having
  widened `delta` several times via re-searches — line 1's window starts from
  that *already-widened* `delta`, plus its own `average²/26394` addition on
  top. Whether this is a deliberate "MultiPV lines get harder to pin down as
  you go" model or an incidental consequence of variable scoping, we can't
  tell from the source alone — flagging it so we make the choice
  *deliberately* if we ever port this shape rather than copying it by
  reflex (see §9).

**Optimism, cross-thread** (search.rs:120-124): before each root search,
Reckless reads a **shared, cross-thread** atomic signal — a packed 32-bit
word is loaded, its low 16 bits (score, offset by `32768`) are averaged
with this thread's own current `average[pv_idx]`, and *that* blended value
(call it `best_avg`) feeds the same shape of optimism formula SF uses:
`optimism[us] = 113·best_avg / (|best_avg| + 201)`, negated for the
opponent. `shared.best_stats: [AtomicU32; MAX_MOVES]` (thread.rs:164) is written after
a thread's own line completes (search.rs:164-167) via
`fetch_max((depth << 16) | (average + 32768) as u32)` — packing depth into
the high bits means a **higher-depth result from any thread always wins**
the max, and only ties in depth break on score. So `best_avg` blends *this
thread's own current average* with *the deepest cross-thread average anyone
has found so far for this pv_idx* — a genuine lazy-SMP information-sharing
channel that predates (and is independent of) the shared TT's propagation.
SF's optimism (`optimism[us] = 137·avg/(|avg|+81)`, stockfish notes §10.3) is
single-thread-local; this cross-thread variant is squarely **P11 (lazy SMP)**
territory for us, not something to backport before threading exists —
noting the mechanism now so it isn't rediscovered from scratch later.

**Fail low / fail high** (search.rs:141-171) — same shape as SF/our design,
different widening constants and a depth-reduction side effect on fail high:

```
fail low  (score <= alpha):  beta = alpha; alpha = max(score - delta, -INF)
                              delta += 26 * delta / 128        (~1.20x)
fail high (score >= beta):   alpha = max(beta - delta, alpha)
                              beta  = min(score + delta, INF)
                              if !is_decisive(score) { reduction += 1 }
                              else                    { reduction = reduction.min(1) }
                              delta += 60 * delta / 128        (~1.47x)
```

Two asymmetries vs SF worth naming: (a) the fail-low and fail-high widening
**rates differ** (1.20x vs 1.47x — SF uses one rate, `44/128 ≈ 1.34x`, for
both); (b) the **depth-shrink counter (`reduction`) only grows on non-decisive
fail-highs** and is explicitly *capped back down to ≤1* the moment a fail-high
resolves to a decisive (mate/TB) score — SF's `failedHighCnt` has no such
decisive-score carve-out (it resets on fail-low, not on a decisive fail-high).
The root search depth actually used is `(depth - reduction).max(1)`
(search.rs:133) — i.e. Reckless folds SF's `adjustedDepth = rootDepth -
failedHighCnt - 3·(searchAgainCounter+1)/4` formula down to a single term,
dropping the `searchAgainCounter` piece entirely (that concern lives
entirely in the TM layer here — see `time.rs`, out of this note's scope).

**Sorting cadence** (search.rs:135, 173): every attempt (even a `stop`ped
one) does `root_moves[pv_index..pv_end].sort_by_key(|rm| Reverse(rm.score))`
— note **`sort_by_key`, not a stable-with-secondary-key sort**: unsearched
moves all carry `score = -Score::INFINITE` (RootMove default, thread.rs:406),
so ties among them are broken by `Vec`'s stable sort preserving prior order —
same *effective* stability property as our design's `sort_by(|a,b|
b.sort_key().cmp(&a.sort_key()))` (design/07 §3.4), just without a compound
sort key since Reckless's `RootMove` has no `previous_score` tiebreak wired
into the comparator (previous_score exists as a field but is only used for
the `eval_stability` signal, §2.3, not as a sort tiebreak). A second, narrower
sort (`root_moves[pv_start..=pv_index]`, search.rs:173) runs on every
*non-final* aspiration attempt too, to keep the current line's slot ordered
mid-search even before the outer loop's full-band sort.

### 2.3 Stability signals and cross-thread soft-stop voting

Beyond SF's `bestMoveChanges`/`totBestMoveChanges`, Reckless tracks two more
per-iteration counters used purely for time management (search.rs:66-67,
185-196):

- `eval_stability`: incremented when `|root_moves[0].score - average[0]| <
  12`, else reset to 0 — "the final score matched its own running average
  closely this iteration."
- `pv_stability`: incremented when the iteration's best move equals the
  previous iteration's best move, else reset — a plain move-level version of
  SF's `lastBestMoveDepth`, kept as a running counter instead of a depth
  stamp.

These feed a five-factor multiplicative soft-stop budget computed once per
iteration (search.rs:245-263: node-fraction, score-trend, `pv_stability`,
`eval_stability`, `best_move_changes` via `ln_1p`) — structurally analogous
to SF's four-factor product (falling eval × reduction × instability ×
effort, stockfish notes §13.3) but a different factor set and, notably, one
Reckless computes as a plain **closure** (`let multiplier = || { ... };`,
search.rs:245) passed into `time_manager.soft_limit(td, multiplier)`
(time.rs:75) rather than SF's inline arithmetic at the call site — a small,
genuinely nice Rust idiom (the TM policy owns *when* to call the formula;
the ID loop owns *what* the formula is, without either side needing the
other's internals). This whole area is design/12's territory for us
(time management), not doc 07's — flagged here only because it lives in the
same function.

**Lazy-SMP soft-stop is a majority vote**, not "any thread decides"
(search.rs:265-280): each thread that independently decides to soft-stop
casts one vote into a shared `AtomicUsize`; only once votes reach `⌈65% ×
thread_count⌉` does any thread actually flip the shared `Status::STOPPED`.
A thread that later stops *wanting* to stop (score improved, instability
dropped) retracts its vote (`fetch_sub`, search.rs:277-278). This is a
genuinely different lazy-SMP time-management shape from SF's "main thread
alone decides, helpers just watch `stop`" — worth remembering for **P11**,
independent of whether we adopt the exact 65% figure.

### 2.4 Forgotten-mate / aborted-loss hygiene, `go mate n`

Same two hygiene rules as SF and our design (§3.5), expressed via
`RootMove::upperbound`/`lowerbound` flags rather than a `score_is_bound()`
method (search.rs:199-222): a `forgotten_mate` check (weaker or now-bound
score where a mate was previously proven) and an `aborted_loss` check (a
stopped search's exact non-bound mated/TB-loss root score is untrusted).
Recovery is inlined as a `Vec::remove` + `Vec::insert(0, ...)` on
`root_moves` (search.rs:212-214) rather than field-by-field restoration —
functionally identical to our design's "move root_moves entry matching
`last_best_move_pv[0]` to front" (design/07 §3.5), just phrased as a full
struct move instead of overwriting a few fields. `go mate n` is the same
`VALUE_MATE - |score| <= 2·n` test (search.rs:237-243).

One thing **not** present here that our design calls for: `last_best_rootmove`
is a single `RootMove` snapshot (search.rs:64, 211-222), so there's no
separate `last_best_move_depth` stability stamp — that information is folded
into the `pv_stability` counter instead (§2.3). Net effect is the same
signal, carried differently.

---

## 3. Root moves and PV storage

### 3.1 `RootMove` (thread.rs:387-416)

The struct carries: the move itself; a current `score`, `previous_score`, and
`display_score` triple; `upperbound`/`lowerbound` bound flags; `sel_depth`;
a `nodes: u64` counter (plays exactly the role our design's `effort` field
does); a `PrincipalVariationTable` (§3.2); and `tb_rank`/`tb_score` for
Syzygy root ranking.

Notably **absent** vs SF and vs our design's `RootMove` (design/07 §3.4):
`average_score` / `mean_squared_score` (those EMAs live only in the *local*
`Vec<Value>` the ID loop threads through `start()`'s own stack frame,
search.rs:63, not on the struct — a real simplification: the per-root-move
average only needs to survive one iteration boundary, not the whole game, so
it doesn't need to be a struct field at all), and `previous_pv` (§2.1).
`display_score` plays the role of our/SF's `uci_score` (clamped to the window
edge on a bound, search.rs:990-1002) — same idea, different name.

### 3.2 `PrincipalVariationTable` — one reused rectangular table (thread.rs:418-455)

The type is a boxed 2-D array of moves, `(MAX_PLY+1)` rows of `(MAX_PLY+1)`
moves each, plus a parallel `[usize; MAX_PLY + 1]` of per-row lengths — a
full rectangle, not a triangle.

This is option 2 from our own design's PV-storage tradeoff analysis (design/07
§3.3: "triangular PV table... classic, but only buys memory nobody is short
of") pushed one step further: Reckless's table isn't even triangular, it's a
full `(MAX_PLY+1) × (MAX_PLY+1)` rectangle (`update()`, thread.rs:434-441,
does the same "child row copies up into parent row" shuffle our `PvLine::load`
does, just against 2-D storage instead of two independent fixed arrays).

The interesting/cautionary part: **`RootMove::pv` is a whole second
`PrincipalVariationTable`**, one per root move (`RootMove { mv, ..Default::default()
}` at root-move construction, `threadpool.rs:107,160`, using `PrincipalVariationTable::default()`,
thread.rs:449-455, which unconditionally allocates the full rectangle). Only
row 0 of that per-root-move copy is ever read (`.line()`, thread.rs:426-428)
— `commit_full_root_pv(&td.pv_table, 1)` (thread.rs:443-447, called from
search.rs:1007) copies exactly one row (`td.pv_table`'s row 1) into it. So
every legal root move pays for a full `(MAX_PLY+1)²`-`Move` allocation
(≈115 KB at `Move = u16`, `MAX_PLY = 240`) to store what is, per move, a
single `MAX_PLY+1`-long line. With, say, 40 legal moves at some root that's
~4.6 MB of PV-table allocation set up once per `go`, almost all of it dead
weight. This reads as "reuse the one struct/API everywhere" prioritized over
memory precision — a real Rust-idiom tradeoff (one type, one set of methods,
zero duplication) but not one we should copy: our own `PvLine` (a flat
`[Move; MAX_PLY+1]` + `len`, design/07 §3.3 option 3) is the tighter design
and we should keep it.

### 3.3 Per-move root bookkeeping (search.rs:981-1014)

Same shape as SF/our design: `nodes += current - initial` (effort), a
three-way match on `score` vs `alpha`/`beta` to set `display_score` and the
bound flags, `pv.commit_full_root_pv(&td.pv_table, 1)` only on a real update,
`best_move_changes += 1` gated on `move_count > 1 && pv_index == 0` — matches
our "increment only for the first MultiPV line, only after move 1" rule
exactly (design/07 §3.4).

---

## 4. The PVS node — what's relevant to search-core (not doc-10 territory)

The full pruning/reduction/extension constant catalogue (razoring, RFP, NMP
+ verification, ProbCut, singular extensions/multicut/negative extensions,
LMP/futility/SEE pruning, the LMR/FDS reduction formulas) is **doc 10's
subject**, not this note's — a future pass should mine `search.rs:536-963` on
its own terms against `design/10-pruning-extensions.md`. What follows is only
the shape and constants that bear on iterative deepening, TT interaction
discipline, and node-exit softening — this note's actual scope.

### 4.1 Static eval, TT substitution, `estimated_score` (search.rs:446-481)

Same three-step shape as SF/our design (correction-history-adjusted
`eval`, then a TT-value substitution for pruning purposes): Reckless names
the substituted value **`estimated_score`** as an explicit local, computed
once (search.rs:470-481) and reused at every pruning site below it (razoring
line 539, RFP line 551, NMP line 570). Our design describes the same
substitution inline at each site rather than factoring it into one named
local (design/07 §3.7.2 step 6). Binding it once, as Reckless does, is a
small, harmless win worth taking regardless of what we do with the
surrounding pruning logic — one substitution computed once, three or four
call sites read the same local instead of re-deriving it.

### 4.2 TT cutoff nuances not in our current spec (search.rs:376-407)

- **Rule-50 threshold is `90`, not `96`** (`if td.board.fiftymove_clock() < 90
  { return tt_score; }`, search.rs:403) — same guard *idea* as SF's "never
  cut when rule50 ≥ 96" and our design's `TT_CUTOFF_RULE50_LIMIT = 96`
  (design/07 §3.10), different tuned edge.
- **The history bonus on a quiet-ttMove cutoff fires even when the cutoff
  itself is suppressed** by the rule-50 guard (search.rs:395-406): the bonus
  block is inside the outer `if` but *before* the `if fiftymove_clock() < 90`
  early return, so a near-the-clock position still gets its history nudge,
  it just doesn't short-circuit the node. Our design's equivalent bullet
  (design/07 §3.7.2 step 7, deferred to doc 09/⟨D10⟩) doesn't yet specify
  this ordering explicitly — worth pinning down when that slot is filled.
- **No depth-≥7 do/undo child-TT-verification refinement.** SF's TT cutoff
  (stockfish notes §4 step 6) does an extra "make the ttMove, probe the
  child's TT entry, only cut if it doesn't contradict" safety check at
  `depth ≥ 7`. Reckless's cutoff condition (search.rs:385-394) has no
  analogue — a straightforward simplification (cheaper, presumably a
  measured Elo-neutral or Elo-positive cut given Reckless's strength), not
  something SF or our design currently omit.

### 4.3 Hindsight depth adjustment and LDSE (search.rs:504-516, 733-736)

The hindsight adjustment matches SF's shape (parent over-reduced by a lot and
the position didn't get worse ⇒ `depth += 1`; parent under-reduced and it did
get worse ⇒ `depth -= 1`) with Reckless's own thresholds
(`reduction >= 2249`, `eval_delta < 0`, `depth -= 1` gated additionally on
`!tt_pv`). One extension mechanism SF's notes don't mention at all: **"Low
Depth Singular Extensions (LDSE)"** (search.rs:733-736, the comment is
verbatim in the source) — an `else` branch alongside the ordinary
singular-extension trigger that fires, roughly, whenever the node is shallow
(`depth <= 7`), not in check, a cut-node, and the TT-substituted eval already
sits at least `25` below alpha (`estimated_score <= alpha - 25`): under those
conditions it directly sets a one-ply extension, with no singular-search
verification at all.

i.e. at shallow cut-nodes where the position already looks like it's failing
low by a real margin, extend by one ply *without* running a singular-search
verification at all — a cheap, unverified extension gated purely on depth/
node-type/static-margin. This is squarely doc 10's subject for us (pruning
& extensions), flagged here only because it sits in the same singular-
extension `if/else if` chain this note already had to read to describe the
ID-loop-adjacent bookkeeping around it.

### 4.4 `lerp` — one blend helper used at every softening site (search.rs:1446-1448)

A private free function, `lerp(a, b, t)`, computes the single-precision lerp
`a + t·(b-a)` via one `f32::mul_add` call and casts back to `i32` — a plain
float lerp, idiomatic and free of the integer-blend fixed-point shift dance
SF and our design use (e.g. `(716·beta + 308·eval)/1024`). Reckless reaches
for `lerp(x, beta, t)` at
**six** different call sites, each with its own tuned `t`, where SF/our
design use a fixed-ratio integer blend at exactly two or three:

| site | `t` (weight toward `beta`/target) | search.rs |
|---|---|---|
| RFP return | 0.6945 | 562 |
| ProbCut return (decisive skip aside) | 0.2695 | 682 |
| Singular multicut return | 0.4027 | 724 |
| qsearch stand-pat fail-high | **0.8256** | 1271 |
| qsearch move-loop fail-high | 0.5072 | 1351 |

Compare SF's two ratios (stockfish notes §4.6, §6): main-search fail-high
softening is **depth-weighted**, `(bestValue·depth + beta)/(depth+1)` — *not*
a fixed ratio at all — while qsearch's two blends are close to each other
(`0.456`/`0.544` split for stand pat, `0.481`/`0.519`-ish region for the
move-loop fail-high). Our own design and current code match SF here exactly
(design/07 §3.7.2 step 14 uses the depth-weighted formula at the main node's
natural exit; §3.8 steps 5/11 use SF's `(467,557)`/`(481,543)` ratios) — see
`src/search/mod.rs:1632-1637` (main-node depth-weighted softening) and
`:1927`, `:2098-2101` (qsearch blends), all present and SF-parity today.

**The one difference worth genuinely weighing**: Reckless's qsearch
stand-pat blend (`0.8256` toward beta) is markedly more aggressive than its
own move-loop fail-high blend (`0.5072`) — a deliberate-looking asymmetry
("a cutoff from the static eval alone is less trustworthy than one a real
child search produced, so pull it harder toward beta") that SF's two ratios
don't express (they're close to a 50/50 split in both cases). We don't have
this distinction in our own qsearch spec either (design/07 §3.8 steps 5 and
11 both use SF's near-symmetric ratios) — see §9 for a concrete, low-effort
retune idea.

**And the one place Reckless has NO softening at all where SF/we do**: the
ordinary main-search move-loop fail-high (the ordinary "a searched move beat
beta" exit, not an early-pruning-shortcut return) is returned **raw** —
there is no depth-weighted (or any) blend applied to `best_score` at the
bottom of `search()` (search.rs:1153-1176 falls straight through the history
updates into the TT store and `return best_score`). SF's/our uniform
node-exit softening (design/07 §3.7.2 step 14) only exists in Reckless at
the handful of *early-return* pruning sites listed above, never at the
node's natural bottom. This is a real, checkable structural difference, not
a transcription slip — worth confirming deliberately (not by omission) if we
ever consider dropping our own node-exit blend.

### 4.5 TT store on PV-node exit is a forced write (search.rs:1160-1162, transposition.rs:206-263)

`TranspositionTable::write`'s last parameter is `force: bool`
(transposition.rs:207-208); the normal replacement gate
(`if !force && key == entry_key && depth + 4 + 2·tt_pv <= entry.depth &&
same_age { return; }`, transposition.rs:253) is entirely **bypassed** when
`force` is true. The main node's own exit store passes `force = NODE::PV`
(search.rs:1161: `td.shared.tt.write(hash, depth, raw_eval, best_score,
bound, best_move, ply, tt_pv, NODE::PV)`), while every other write site in
the file (RFP return, TB probe, ProbCut, the mid-loop alpha-raise write, both
qsearch writes) passes a literal `false`. Net effect: **a PV (and Root) node
always lands its own result in the TT when it finishes, even if a deeper,
newer same-key entry already sits there** — trading a possibly-useful deep
entry for a guaranteed-fresh PV entry. Neither SF's `save()` (stockfish
notes §11.3, "overwrite test" — uniform for every node) nor our design's TT
store bullet (design/07 §3.7.2 step 14, "no non-PV/PV distinction in the
replacement test") has anything like this; TT replacement policy is owned by
**doc 08** for us, so this is that doc's subject, flagged here because it's
driven directly off the `NodeType` this note already had to characterize.

---

## 5. Quiescence search — `qsearch` (search.rs:1178-1362)

Same overall shape as SF/our design (repetition/draw/max-ply guards, TT
probe, stand pat unless in check, capture-only move loop, mate/fail-high
handling, `DEPTH_QS`-equivalent store), with several genuine simplifications.

### 5.1 What qsearch generates — one `MovePicker`, no separate evasion/qsearch pipelines

Unlike SF's four distinct `MovePicker` staging pipelines (MAIN / EVASION /
PROBCUT / QSEARCH, stockfish notes §8.1), Reckless has **exactly one**
five-stage state machine (`movepick.rs:10-16`: `HashMove → GenerateNoisy →
GoodNoisy → Quiet → BadNoisy`) used for the main search, qsearch, ProbCut,
*and* check evasions alike. Two things make this work:

- The underlying generator (`Board::generate_moves`, `board/movegen.rs:40-60`)
  is **check-aware at the bitboard-target level**, not via a separate
  `GenType::EVASIONS`: when `in_check()`, the target mask is narrowed to
  `between(king, checker) | checkers` (movegen.rs:53-54) — the *same*
  `append_noisy_moves`/`append_quiet_moves` calls the non-check case uses
  automatically only produce blocks/captures/king-moves. There is no
  evasion-specific move list at all.
- The noisy-move scorer gives check evasions an enormous flat bonus
  (`(200000 - 20000·piece_type) * in_check as i32`, `movepick.rs:139`) so
  that *within* the ordinary `GoodNoisy` stage, capturing evasions always
  sort above quiet evasions — reproducing SF's "capturing evasions always
  rank above quiet evasions" rule (stockfish notes §8.2) without a distinct
  evasion code path.
- `MovePicker::new(tt_move, threshold)`'s `skip_quiets` parameter (passed
  fresh on every `.next()` call, not baked into the picker at construction)
  is what qsearch uses to suppress the whole `Quiet` stage: qsearch calls
  `move_picker.next::<NODE>(td, skip_quiets(best_score), ply)` with
  `skip_quiets = |best_score| !in_check || !is_loss(best_score)`
  (search.rs:1290) — i.e. quiets are generated *only* when in check and no
  legal evasion has yet produced a real (non-`-INFINITE`) value, exactly
  SF's "no quiet-checks stage, but do search quiet evasions" rule
  (stockfish notes §6 step 4), expressed as a boolean closure over the
  running `best_score` instead of a compile-time-distinct stage set.

This is a genuinely reusable idea independent of anything else in this note:
**one staged picker, parameterized by a runtime `skip_quiets` flag and an
optional SEE `threshold`, replaces four separate pipelines.** That's doc 09's
subject for us (design/09-move-ordering-history.md governs `MovePicker`
staging) — flagged here because "what does qsearch generate" was explicitly
in scope and the honest answer is "the same generator as everything else,
with a runtime flag."

### 5.2 Per-move pruning is one `see()` call, not futility + window-SEE + a hard floor

SF's qsearch per-move rule (stockfish notes §6 step 5) is three separate
tests: a piece-value-table futility precheck (`futilityBase +
PieceValue[captured] <= alpha`), a SEE-vs-window check
(`see_ge(m, alpha - futilityBase)`), and an unconditional hard SEE floor
(`see_ge(m, -74)`) — plus a "not aimed at the previous move's destination"
carve-out that exempts recaptures from all of it. Reckless's equivalent
gating (search.rs:1295-1305), only reached once `best_score` isn't a loss,
does just two things per candidate: an LMP cutoff (past the second
non-checking candidate, stop trying more moves entirely) and a single SEE
call against the folded threshold `(alpha - eval)/8 -
min(|correction_value|, 68) - 74`.

Two collapses at once:

- **No recapture-square carve-out, no piece-value table, no separate
  futility precheck.** One `see()` call, gated by a single threshold that
  folds "how far behind is `alpha` relative to the static eval" (`(alpha -
  eval)/8`), "how uncertain is that eval" (`-min(|correction|, 68)`, an
  uncertainty discount capped at 68), and SF's flat SEE floor (`-74`, same
  constant) into one number. There's no need for a `PieceValue[captured]`
  table lookup in the hot loop at all, because `see()` itself already prices
  the capture — SF's separate futility precheck exists purely as a *cheap
  short-circuit before the expensive SEE call*; Reckless appears to accept
  paying for `see()` on every candidate instead (mitigated by the LMP rule
  below capping how many candidates there are in the first place).
- **Late-move pruning is a `break`, not a scoped `continue`.** SF's
  `moveCount > 2` rule (stockfish notes §6 step 5) only suppresses *that
  category* of move (non-check, non-recapture, non-promotion) and lets later
  captures of other categories through; Reckless's `break` at
  `move_count >= 3 && !is_direct_check(mv)` **exits the entire remaining move
  loop** the moment the third non-checking candidate comes up (since the
  picker yields captures in descending score order, everything after is
  assumed to be worse and is simply never tried — checking moves are the one
  exception, exempt from the counter entirely). This is a strictly blunter,
  cheaper rule: qsearch here tries at most ~2 non-check captures plus however
  many checks the picker turns up, full stop.

### 5.3 TT cutoff relaxes the PV exclusion by decisiveness, not by node type

SF's qsearch TT cutoff (and our design's, design/07 §3.8 step 4) is
**non-PV-only**, mirroring the main search's rule exactly. Reckless's
qsearch cutoff (search.rs:1225-1234) requires a valid stored value, the
bound pointing the right way for a cutoff — and, in place of a flat
non-PV-only gate, only `!NODE::PV || !is_decisive(tt_score)`.

i.e. a **PV** qsearch node *can* still return early off a stored TT value,
provided that value **isn't decisive** (not a mate/TB score) — only a
decisive stored score is withheld from a PV node (so a PV line never
silently truncates onto an old mate claim without re-verifying it). This is
strictly more permissive than SF/our current spec, and the rationale is
easy to state precisely: qsearch's own tree is already capture-bounded and
shallow, so letting a PV node reuse a merely-good-but-non-mate cached value
costs little accuracy and saves real work in long forced-capture sequences
that recur inside the PV. Also note: **the entry's depth is never even read
here** — qsearch's TT probe (search.rs:1212-1234) extracts `tt_score`/
`tt_bound`/`tt_pv` from the hit but has no local `tt_depth` at all, unlike
the main search (which does extract and compare `entry.depth`, search.rs:378,
387). Any stored entry — whether written by qsearch itself (at
`TtDepth::SOME`) or by a deeper main-search call — is treated as equally
trustworthy for a qsearch-level cutoff; SF's/our design's explicit
`tt_depth >= DEPTH_QS` guard (design/07 §3.8 step 4) is true unconditionally
at this granularity, so Reckless simply never writes the comparison down.

### 5.4 No narrow stalemate test

Our design's qsearch spec (design/07 §3.8 step 10) includes a deliberately
narrow, cheap stalemate detector for the classic "qsearch just won material
into a stalemate trap" pattern (no non-pawn material, all pawns blocked,
previous move captured ≥ a knight ⇒ pay for one legal-move-count check).
**Reckless has no such test anywhere in `qsearch`** — the only terminal
check is `if in_check && move_count == 0 { return mated_in(ply) }`
(search.rs:1334-1336); a non-check position that runs out of qsearch moves
simply falls through to the ordinary fail-high/TT-store path with whatever
`best_score` it already has (typically the stand-pat value), never checking
whether the side to move is actually stalemated. This is a real, accepted
accuracy gap in exchange for one fewer branch and no legality re-scan at
every quiet qsearch leaf — worth knowing is intentional-by-omission before
we consider ever dropping our own check for "simplicity."

### 5.5 Everything else matches SF/our spec closely

Stand pat, `futility_base`-equivalent, TT-miss-only eval-cache write, the
mate return, and the final TT store (bound `LOWER`/`UPPER`, never `EXACT`,
depth `TtDepth::SOME`, `is_pv` = the *probed* pv flag only, not OR'd with the
node's own PV-ness — search.rs:1356 passes `false` as the trailing `is_pv`
arg on every qsearch store, i.e. Reckless never marks a qsearch-stored entry
as "was PV" *even from a currently-PV node*, an even stricter version of the
"deliberately NOT OR'd" rule our design already calls out at design/07 §3.8
step 12) all line up with SF and with our current implementation.

---

## 6. The search `Stack` and per-ply state (stack.rs:1-80, thread.rs:190-241)

### 6.1 Layout: offset `+8`, boxed, self-referential via a raw pointer

`Stack` (stack.rs:5-8) is a fixed array of `StackEntry` sized `MAX_PLY + 16`
(`MAX_PLY = 240` here, types/mod.rs:26, so 256 entries), plus a second field
holding one inert "sentinel" continuation-history slab (a `[[i16; 64]; 13]`)
embedded directly in the same struct.

Indexing is `data[(index + 8) as usize]` (stack.rs:70-78) — an `+8` offset
where our design uses `+7` (design/07 §3.2.1; also a smaller total,
`MAX_PLY + 10` vs Reckless's `MAX_PLY + 16`, and Reckless's `MAX_PLY = 240`
vs our `246`). Functionally equivalent padding-for-negative-offset-reads
idea, different numbers.

The distinctive part: `Stack::new()` (stack.rs:15-23) builds a `Box<Self>`,
then takes a **raw pointer to the box's own embedded sentinel field**
(`&raw mut stack.sentinel`) and writes that pointer into *every* entry's
`conthist`/`contcorrhist` fields (both typed as a raw `*mut` pointer to the
slab shape) as their initial, always-safe-to-read value. `StackEntry` also
carries an explicit `unsafe impl Send` to let those raw pointers cross
thread boundaries.

This is a **self-referential struct**, made sound only by being heap-boxed
first (a fixed address that never moves for the box's lifetime) and by every
other write site (`make_move`, search.rs:1424-1439) overwriting these
pointers with real continuation-history slab addresses before they're ever
read for a real move — the sentinel is only ever read through padding slots
and after a null move (search.rs:590-593). This buys Reckless **one pointer
dereference** to reach a continuation-history slab from the stack, instead
of resolving a key through a table lookup.

**This is exactly the design our own doc rejected, on purpose** (design/07
§3.2.3: *"a Rust raw pointer here would poison the whole search with
`unsafe`... instead the stack stores a compact key that doc 09 resolves in
O(1)"* → `ContHistId(u16)`). Reading Reckless's version confirms the
rejected alternative works and is not unsound *as implemented* — but it also
requires `unsafe impl Send`, a `Box`-pinning discipline that has to be
re-verified by hand any time `Stack` is touched, and gives up the
compiler-checked guarantee our `ContHistId` gets for free. Nothing here
should change our decision; it's useful to have looked at the alternative
Reckless actually shipped and confirm ours is the better fit for D14a's
"minimize unsafe surface" stance.

### 6.2 `StackEntry` field list vs ours

Nine fields: the move played, the piece that moved, the (corrected) static
eval, the node's TT move, its sticky `tt_pv` flag, a move-count, an `i32`
reduction (the same "child reads its parent's LMR reduction back"
mechanism as our design's `reduction` field), and the pair of raw
continuation-history/continuation-correction-history pointers from §6.1.

Notably **absent** vs both SF's `Stack` and our design's `Stack`
(design/07 §3.2.3): `excluded_move` and `cutoff_cnt` are **not** fields on
`StackEntry` at all. Instead they live as two separate top-level
`ThreadData` fields, each typed as the same small generic array — one of
`i32` (the cutoff counts), one of `Move` (the excluded moves) — both sized
`MAX_PLY + 16` and indexed the identical `+8`-offset way as `Stack` itself.

`PlyArray<T, N>` (thread.rs:190-211) is a tiny **generic** newtype —
`Index<isize>`/`IndexMut<isize>` at the identical `+8` offset as `Stack`
itself, `Default` requiring `T: Copy + Default` — reused for both of these
per-ply scalars instead of growing the main `Stack` struct. This is a
genuinely nice, minimal piece of Rust idiom worth naming on its own: rather
than one large per-ply struct carrying every field anyone might need at any
ply (SF's approach, and currently ours), factor "a `+8`-offset fixed array"
into one small generic type and reuse it per auxiliary scalar that needs the
same indexing scheme but doesn't need to travel with the rest of the stack
entry. It costs one more field lookup indirection (`td.excluded[ply]`
vs `stack[ply].excluded_move`) in exchange for a smaller, more focused
`Stack`/`StackEntry` and zero duplication of the offset-indexing logic.

`ss.follow_pv` (our design's PV-path flag, design/07 §3.2.3) has no
counterpart in Reckless at all — nothing in `search.rs` gates any decision
on "did the parent follow last iteration's PV." That's plausibly folded into
`tt_pv`'s stickiness instead (a TT-hit-derived "was ever PV" bit serves a
similar "trust this path more" role), but it's a genuine behavioral
difference, not just a naming one, and worth keeping in mind if we ever
revisit whether `follow_pv` earns its keep.

---

## 7. Piece values used in ordering/SEE (types/piece.rs:138-146)

`Pawn 109, Knight 403, Bishop 435, Rook 679, Queen 1242` — roughly half of
SF's scale (`P208 N781 B825 R1276 Q2538`, stockfish notes §8.2), consistent
with a different (presumably NNUE-output-tied) internal centipawn scale.
Not a search-core concern per se, noted only because qsearch's SEE-threshold
formula (§5.2) and the noisy-move scorer (`movepick.rs:136-139`,
`14232 * captured.value() / 1024` ≈ `13.9 * captured.value()`, still roughly
MVV-only + history, matching SF's "pure MVV, no LVA term" observation)
both read directly off this table — doc 09/11 territory to dig into further.

---

## 8. Summary table — same / simpler / cleverer / missing

| Subsystem | SF | Our design (doc 07) | Reckless | Verdict |
|---|---|---|---|---|
| Node types | template param | sealed trait markers | same trait-marker idea | same |
| Aspiration delta | `5 + \|mss\|/10588` (+thread jitter) | matches SF | `23 + avg²/26394`, carries across MultiPV lines within a depth | simpler state, different tuning, riskier MultiPV carry-over |
| Optimism | single-thread `137·avg/(\|avg\|+81)` | matches SF | cross-thread max-tracked `113·best_avg/(\|best_avg\|+201)` | cleverer (lazy-SMP aware), P11 territory |
| Root sort | `stable_sort`, compound key | matches SF | `sort_by_key`, single key (stability comes from the `-INF` sentinel) | same effective behavior, less code |
| PV storage | parent-local `ss->pv` pointer | flat `PvLine` per ply (chosen) | full rectangular table, **duplicated per root move** | ours is tighter; Reckless trades memory for one reused type |
| Search node | many named steps (search.cpp) | explicit numbered steps (design) | one big function, same order | equally correct, less maintainable at scale |
| TT cutoff | depth-7 child re-verify, rule50≥96 | matches SF (deferred to doc 10) | no child re-verify, rule50≥90, bonus fires even when suppressed | simpler, slightly different edges |
| Node-exit softening | depth-weighted blend, every node | matches SF | **only at early-return pruning shortcuts**, never at the natural fail-high exit | missing at one site vs SF/us — verify deliberately |
| TT store on PV exit | uniform replacement test | matches SF | `force = NODE::PV` bypasses replacement entirely | cleverer (guarantees PV freshness), doc 08 territory |
| Qsearch movegen | 4 distinct staged pipelines | doc 09 plans SF-like staging | **one** staged picker + a runtime `skip_quiets` flag for everything | cleverer/simpler, doc 09 territory |
| Qsearch per-move pruning | futility table + window-SEE + hard floor + recapture carve-out | matches SF exactly | one `see()` call, one folded threshold, no recapture carve-out | simpler, cheaper, less precise |
| Qsearch LMP | `continue` (scoped to one move category) | matches SF | `break` (exits the whole remaining move loop) | blunter/cheaper |
| Qsearch TT cutoff at PV nodes | never | matches SF (never) | allowed when the stored value isn't decisive | more permissive, a real idea to weigh |
| Qsearch stalemate detector | narrow test | narrow test specified | **none** | accepted accuracy gap |
| Stack conthist ref | raw `PieceToHistory*` | `ContHistId(u16)` key, zero unsafe | boxed self-referential raw pointer, `unsafe impl Send` | ours is the safer fit for D14a; confirms the rejected alternative isn't unsound, just costlier to maintain |
| Per-ply excluded/cutoff state | fields on `Stack` | fields on `Stack` | separate generic `PlyArray<T,N>`, reused type | small, genuinely nice factoring idea |

---

## 9. Adapt to us

Ordered roughly by effort, cheapest first. None of these touch
movegen/make-unmake, board mutation, Zobrist, or attacks, so the D24 perft
ritual does **not** apply — but every item **does** need the `Bench:`
trailer discipline (D24/design/14 §3.2) since all of them can change node
counts — treat each as its own SPRT-gated commit, not a bundle.

1. **Bind the TT-substituted pruning eval into one local (`estimated_score`),
   once, at the point our design already substitutes it inline.**
   *Where*: design/07 §3.7.2 step 6 (the substitution itself is already
   specified; this is purely the "compute once, reuse at every pruning site"
   refactor Reckless does at search.rs:470-481). *Phase*: P9 (pruning waves
   in progress) — this is a non-functional refactor if applied to existing
   call sites verbatim (same formula, same order of evaluation), so it
   should reproduce the bench count bit-for-bit; confirm with check.sh on
   both sides before treating it as free. *Effort*: trivial (30 min).
   *Impact*: none on strength, small maintainability win, removes a
   footgun (re-deriving the substitution slightly differently at two sites
   by accident).

2. **Retune the qsearch stand-pat blend ratio separately from the qsearch
   move-loop fail-high blend**, following Reckless's asymmetry (0.8256 vs
   0.5072) as a hypothesis, not a value to copy verbatim. *Where*: design/07
   §3.8 steps 5 and 11 (`QS_STANDPAT_BLEND`/`QS_FAILHIGH_BLEND` in
   `search/params.rs`, currently SF-parity `(467,557,1024)`/`(481,543,1024)`
   — both close to 50/50). *Phase*: P9 (this is exactly the shape of
   "heuristic gated by a fixed-node proxy SPRT," design/14 §9.4) or a P9.5
   tuning pass once the pruning waves land. *Effort*: change two constants,
   run the existing qsearch blend tests (`src/search/mod.rs:3668`,
   `:3846`) plus a fixed-node proxy SPRT. *Impact*: speculative but
   plausible — Reckless's own SPSA presumably found this asymmetry for a
   reason (a stand-pat cutoff is strictly less trustworthy evidence than one
   a real child search produced), and it costs nothing to test.

3. **Consider relaxing the qsearch TT cutoff's PV exclusion from
   "never" to "unless decisive."** *Where*: design/07 §3.8 step 4 (currently
   "NonPV cutoff needs only..." — silently never fires at PV nodes at all).
   *Phase*: P9/P10 boundary — this is a genuine behavior change to spec
   before code, so it should land as a decisions.md-adjacent doc edit to
   design/07 first (small: one bullet, one guard condition), then an
   implementation commit with its own `Bench:` trailer and an SPRT (it
   changes qsearch node counts along any PV that revisits a transposed
   near-mate capture sequence — probably rare enough that the effect is
   small either way, which is exactly why it's worth measuring rather than
   guessing). *Effort*: small (a few hours including the doc edit and
   tests). *Impact*: likely small and net-neutral-to-positive; low risk
   given the decisive-score guard bounds the downside to "never truncate a
   mate line."

4. **Fold the qsearch futility-table + window-SEE + hard-floor triad into
   one SEE-threshold call**, Reckless-style, and separately decide whether
   to drop the recapture-square carve-out. *Where*: design/07 §3.8 step 7
   (the whole per-move rules bullet list). *Phase*: P9 — this is a
   functional change to an already-implemented, tested qsearch (the "P7 doc
   07 step 9: qsearch full spec" commit), so it must reproduce the *same*
   pruning decisions as today's three-part test or else change the node
   count and go through SPRT either way. Two sub-options worth trying
   independently: (a) collapse to one `see()` call with a folded threshold
   (drops the `PieceType` value-table dependency from the qsearch hot loop);
   (b) additionally drop the recapture-square exemption (simpler code, more
   aggressive pruning of recaptures — riskier, test separately from (a)).
   *Effort*: medium (rewrite one function, re-verify against the existing
   qsearch test suite at `src/search/mod.rs:3668-3897`, which will need new
   assertions for whatever the new single-threshold shape actually prunes).
   *Impact*: uncertain sign, likely a modest node-count/speed change either
   way — genuinely worth an SPRT rather than a guess, since Reckless is
   evidence the simplification is at least not ruinous at ~3833 SPCC.

5. **A generic `PlyArray<T, const N: usize>` type, reused for any future
   per-ply scalar that doesn't need to live on the main `Stack`.** *Where*:
   design/07 §3.2 (the `Stack`/indexing section) — not a change to the
   current field list, but a pattern to keep in the toolbox the next time a
   per-ply scalar (something like a future `excluded_move`-shaped need, or
   a per-ply diagnostic counter) comes up and doesn't obviously belong
   folded into `Stack` itself. *Phase*: opportunistic, no specific phase —
   pure Rust-idiom housekeeping, zero behavior change. *Effort*: trivial
   once needed. *Impact*: none on strength; keeps `Stack` from growing
   fields that only a few call sites read.

6. **Cross-thread "best average score seen so far, by depth" signal for
   optimism** (the `best_stats: [AtomicU32; MAX_MOVES]`, `fetch_max`-packed-
   by-depth scheme, search.rs:120-124, 164-167). *Where*: this is squarely
   **P11 (lazy SMP)** — our design doesn't have a threads doc section for
   this yet (design/13-threading-smp.md should gain a bullet once P11
   starts) and doc 07's optimism hook (design/07 references "eval-doc hook:
   optimism..." at §3.6) is currently single-thread-shaped, matching SF.
   *Effort*: small once P11's shared-state plumbing exists (one atomic
   array, one packed-compare-and-swap pattern already proven correct in
   Reckless). *Impact*: unclear in isolation — this is a lazy-SMP
   diversity/convergence aid, best evaluated as part of P11's own gauntlet,
   not bench-alone.

7. **Majority-vote soft-stop across lazy-SMP threads**, instead of "first
   thread to want to stop, stops everyone." *Where*: **P11** again — pairs
   with design/12 (time management)'s soft-stop policy once threads exist;
   today's single-threaded TM has no analogous decision to make. *Effort*:
   medium (an `AtomicUsize` vote counter, a retraction path when a thread's
   local signal improves — both already worked out in Reckless,
   search.rs:265-280). *Impact*: plausible reduction in the "one noisy
   thread stops the whole search early" failure mode SF's main-thread-only
   TM doesn't have to worry about but any true lazy-SMP scheme does; worth
   prototyping once P11 lands rather than guessing the 65% figure.

8. **Do NOT adopt**: the raw-pointer self-referential `Stack` (§6.1) and the
   full rectangular per-root-move `PrincipalVariationTable` (§3.2). Both are
   working code in a top-2 engine, which is exactly why they're worth having
   read closely — but both are strictly worse fits for our stated
   constraints (D14a's minimal-unsafe-surface stance for the first; our own
   already-better `PvLine` design for the second). Recorded here so a future
   pass doesn't rediscover them as "ideas" without the context of why we
   looked and passed.
