# Stockfish search — comprehensive notes

Source studied: Stockfish dev tree at commit `eca43a97` (2026-07-06), i.e. newer
than released SF 17. Files: `search.{h,cpp}`, `movepick.{h,cpp}`, `history.h`,
`tt.{h,cpp}`, `timeman.{h,cpp}`, `thread.{h,cpp}`, plus supporting parts of
`position.cpp`, `evaluate.cpp`, `types.h`, `uci.cpp`, `misc.h`, `memory.cpp`.
Line references are into `Stockfish/src/`.

Every numeric constant below is a **tuned value** (SPSA-fitted, retuned
constantly). Treat them as sane starting points, not gospel — what matters for
us is the *shape* of each formula and what it depends on.

**Units convention:** Stockfish keeps LMR reductions `r` in **1024ths of a
ply** (divide by 1024 before applying to depth). History values are ints
roughly in ±32k updated by a "gravity" formula (§9.2).

Two structural surprises vs. classic engine literature, up front:

- **Killer moves and countermove tables are gone.** Quiet ordering is entirely
  history-driven (butterfly + pawn-structure + continuation + low-ply
  histories, plus static threat/check terms).
- Returned scores are constantly **softened toward the bound** (weighted blends
  of `bestValue` and `beta`) rather than returned raw — deliberate
  search-instability damping.

---

## 1. Big picture

```
UCI "go"
  └─ ThreadPool::start_thinking            (thread.cpp:289)
       └─ main Worker::start_searching     (search.cpp:188)
            ├─ TimeManagement::init, tt.new_search()
            ├─ wake helper threads (lazy SMP — identical searchers, shared TT)
            └─ Worker::iterative_deepening (search.cpp:266)
                 └─ per depth, per MultiPV line: aspiration window loop
                      └─ search<Root>                (search.cpp:701)
                           └─ search<PV/NonPV>  ──┐  (recursive negamax, PVS)
                                └─ qsearch<PV/NonPV> (search.cpp:1596) at depth <= 0
            after stop: vote best thread → print bestmove
```

The search is a **principal variation search (PVS)** negamax: first move (and
PV re-searches) get the full `(alpha, beta)` window; every other move is first
tried with a zero-width window `(alpha, alpha+1)` at a **reduced** depth (LMR),
re-searched deeper/wider only when it unexpectedly beats alpha. Around that
skeleton hangs a large set of pruning/extension heuristics, all gated by depth,
node type, and history statistics.

Node types: template parameter `NodeType ∈ {Root, PV, NonPV}` plus a runtime
`cutNode` bool (expected fail-high node); `allNode = !PvNode && !cutNode`.
Invariant: NonPV nodes always have zero-width windows.

Per-ply search stack (`search.h:104-120`): `pv, continuationHistory,
continuationCorrectionHistory, ply, currentMove, excludedMove, staticEval,
statScore, moveCount, inCheck, ttPv, ttHit, followPV, cutoffCnt, reduction`.
Allocated as `stack[MAX_PLY + 10]` with `ss = stack + 7` so `(ss-7)…(ss+2)`
are always addressable without bounds checks.

---

## 2. Value semantics (types.h)

- `VALUE_MATE = 32000`, `VALUE_INFINITE = 32001`, `VALUE_NONE = 32002`,
  `VALUE_DRAW = 0`, `MAX_PLY = 246`.
- `mate_in(ply) = VALUE_MATE - ply`, `mated_in(ply) = -VALUE_MATE + ply`.
  Mate scores are "plies **from root**" while inside the search.
- Score bands, in increasing order of magnitude: normal evals <
  TB-win band (`VALUE_TB_WIN_IN_MAX_PLY …`) < mate band
  (`VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY`).
- Predicates `is_win / is_loss / is_decisive / is_mate` gate nearly every
  heuristic ("never prune/soften/adjust when a decisive score is involved").
- **TT conversion** (search.cpp:1852-1894): `value_to_tt(v, ply)` converts
  root-relative mates to node-relative (`v+ply` for wins, `v-ply` for losses)
  before storing; `value_from_tt(v, ply, rule50)` inverts it after probing,
  and additionally downgrades stale decisive scores: if
  `VALUE_MATE - v > 100 - rule50` (mate can't be delivered before the 50-move
  clock runs out) the score collapses to `VALUE_TB_WIN_IN_MAX_PLY - 1`
  (mirrored for losses, same test with `VALUE_TB` for TB scores).
- **Draw dither** (search.cpp:132): draws return
  `VALUE_DRAW - 1 + (nodes & 0x2)` — ±1 pseudo-random from the node counter,
  so repetition lines aren't all exactly comparable to 0 (avoids "3-fold
  blindness").
- Depth sentinels: `DEPTH_QS = 0`, `DEPTH_UNSEARCHED = -2` (eval-only TT
  writes), `DEPTH_NONE = -3`.

---

## 3. Iterative deepening (search.cpp:266-628)

Per new search each thread pre-ages its histories: `lowPlyHistory.fill(100)`,
`mainHistory *= 789/1024` (search.cpp:322-326).

Loop over `rootDepth = 1, 2, …` until MAX_PLY, stop flag, or depth limit
(depth limit respected by the main thread only). Each iteration halves the
PV-instability accumulator `totBestMoveChanges` and snapshots every root
move's `previousScore/previousPV`.

### 3.1 Aspiration windows (per MultiPV line, search.cpp:357-439)

- Initial half-width `delta = 5 + threadIdx % 8 + |meanSquaredScore| / 10588`
  — per-thread jitter (threads search slightly different windows: free lazy-SMP
  diversity) plus widening when the score has been large/volatile.
  `meanSquaredScore` is an EMA of `value·|value|` maintained at root.
- Window centered on the root move's `averageScore` (EMA `(value+avg)/2`).
- **Optimism** is set here (§10.3): `optimism[us] = 137·avg / (|avg| + 81)`,
  `optimism[them] = -optimism[us]` — the winning side evaluates optimistically.
- Search depth: `adjustedDepth = max(1, rootDepth - failedHighCnt
  - 3·(searchAgainCounter+1)/4)` — each consecutive fail-high re-search runs
  one ply shallower; `searchAgainCounter` grows when time management said
  "don't deepen" (§13.3). `rootDelta = beta - alpha` is stored and feeds the
  reduction formula.
- **Fail low**: `beta = alpha; alpha = max(bestValue - delta, -INF)`; reset
  `failedHighCnt`; **cancel `stopOnPonderhit`** (a dropping score revokes
  permission to stop early).
- **Fail high**: `alpha = max(beta - delta, alpha)` (alpha only backs off to
  the old beta — window kept as tight as possible); `beta = bestValue + delta`;
  `++failedHighCnt`.
- Widening per re-search: `delta += 44·delta/128` (≈ ×1.34).
- After every attempt root moves are `stable_sort`ed (non-PV moves carry
  `-VALUE_INFINITE`, stability preserves last iteration's order among them).

### 3.2 Cross-iteration bookkeeping & mate safety

- `lastBestMovePV/Score/Depth` track the best across iterations;
  `lastBestMoveDepth` = iteration at which the best move last changed (the
  time-management stability signal).
- **Forgotten mate**: if a previous iteration proved a mate and the current
  one comes back with a worse/bound score, roll back to the previous PV.
- **Aborted-loss hygiene**: an exact mated/TB-loss root score from a *stopped*
  search is untrusted — restore the previous iteration's move/score or mark
  it as a bound. MultiPV has extra guards so an aborted line can't spoil a
  proven loss in an earlier line.
- `limits.mate n`: stop once `VALUE_MATE - |score| <= 2n` after a full
  iteration.

---

## 4. The main search node, step by step (search.cpp:701-1587)

`search<NodeType>(pos, ss, alpha, beta, depth, cutNode)`. If `depth <= 0` →
qsearch. Order of operations at a node:

0. **Upcoming-repetition early draw** (non-root): if `alpha < VALUE_DRAW` and
   `pos.upcoming_repetition(ply)` (cheap Zobrist "cuckoo" detection of a
   forced repetition), raise alpha to the dithered draw value; return if
   `alpha >= beta`.
1. **Init**: `ss->inCheck`, `priorCapture`, `moveCount = 0`,
   `bestValue = -INF`, `maxValue = +INF` (TB cap). `ss->followPV` = we are on
   last iteration's PV path (root, or parent followed it and played the PV
   move) — disables IIR and PV-node quiet pruning below. Main thread checks
   time; PV nodes bump `selDepth`.
2. **Abort/draw**: stopped, `pos.is_draw(ply)`, or `ply >= MAX_PLY` → dithered
   draw (eval at MAX_PLY when not in check).
3. **Mate distance pruning**: `alpha = max(mated_in(ply), alpha)`,
   `beta = min(mate_in(ply+1), beta)`; return alpha if `alpha >= beta`.
4. **TT probe**: unpack `ttData` via `value_from_tt`. At root the "ttMove" is
   forced to the current PV move. `ss->ttPv = PvNode || (ttHit && ttData.is_pv)`
   — a sticky "was ever on the PV" flag, stored back to the TT.
5. **Static eval** (§10): in check → borrow `(ss-2)->staticEval` (no pruning
   happens anyway, jump to the move loop). Otherwise: reuse `ttData.eval` if
   valid else call `evaluate()`; apply correction history → `ss->staticEval`;
   then if the TT value can serve as a tighter bound
   (`bound & (ttValue > eval ? LOWER : UPPER)`) use `eval = ttValue` for the
   pruning decisions below. On a TT miss, immediately store an eval-only entry
   (`VALUE_NONE`, `BOUND_NONE`, `DEPTH_UNSEARCHED`, **uncorrected** eval) —
   the TT doubles as an eval cache.
   - `improving = ss->staticEval > (ss-2)->staticEval`;
     `opponentWorsening = ss->staticEval > -(ss-1)->staticEval`.
   - **Hindsight depth adjustment**: if the parent LMR-reduced us by ≥3 plies
     (`(ss-1)->reduction`) and the opponent isn't worsening → `depth++` (the
     reduction looks unjustified); if reduced ≥2 and
     `ss->staticEval + (ss-1)->staticEval > 173` → `depth--`.
6. **TT cutoff** (non-PV only, no excluded move): requires
   `ttDepth > depth - (ttValue <= beta)` (one ply of slack for upper bounds),
   bound matching the cutoff direction, and
   `cutNode == (ttValue >= beta) || depth > 4`. Extra safety:
   - never cut when `rule50_count() >= 96` (graph-history interaction);
   - at `depth >= 7` with a legal non-decisive ttMove, do/undo the ttMove and
     probe the **child's** TT entry — only cut if it doesn't contradict;
   - on a quiet-ttMove fail-high cutoff, give it history bonus
     `min(114·depth, 724)` and penalize the parent's early quiet move
     (`-2187` conthist if `(ss-1)->moveCount < 4`);
   - if the *only* thing blocking a cutoff was a wrong-direction bound and
     `depth > 5`: `ttWriter.penalize(1)` — age the useless entry.
7. **Syzygy TB probe** (piece count within cardinality, `rule50 == 0`, no
   castling): WDL → `±(VALUE_TB - ply)` or dithered draw; if the resulting
   bound already decides the node, store with depth `depth+6` and return; in
   PV nodes it instead tightens `bestValue`/`maxValue`.

   **If in check: skip straight to the move loop** (steps 8-12 all skipped).
8. **Eval-diff history nudge**: if the parent's move was a quiet non-check,
   grade it by the eval swing:
   `evalDiff = clamp(-(ss-1)->staticEval - ss->staticEval, -183, 180) + 62`;
   parent's move gets `mainHistory << 10·evalDiff` (and pawn history
   `<< 13·evalDiff` when fresh node / non-pawn mover / non-promotion).
9. **Razoring** (non-PV): if `eval < alpha - 465 - 300·depth²` → return
   `qsearch(alpha, beta)` directly. No depth cap.
10. **Reverse futility pruning** ("futility: child"): conditions
    `!ss->ttPv && depth < 17 && eval >= beta && (!ttMove || ttCapture)
    && !is_loss(beta) && !is_win(eval)`. Margin:
    `futilityMult = min(40 + 4·depth, 80)` (−20 if no ttHit);
    `margin = futilityMult·depth
              - (2934·improving + 343·opponentWorsening)·futilityMult/1024
              + |correctionValue|/182069`.
    If `eval - margin >= beta` return the blend `(716·beta + 308·eval)/1024`.
11. **Null move pruning** — note: **only at cutNodes** now. Conditions:
    `cutNode && ss->staticEval >= beta - 14·depth - 45·improving + 374
    && !excludedMove && we have non-pawn material && ply >= nmpMinPly
    && !is_loss(beta)`.
    - `R = 7 + depth/3` (flat + depth term; no eval-based term currently);
      search `(ss+1, -beta, -beta+1, depth-R)` after `do_null_move`.
    - On `nullValue >= beta` (and not a win score): return immediately if
      `nmpMinPly != 0 || depth < 16`; otherwise run a **verification search**
      at the same node (`(beta-1, beta)`, depth `depth-R`) with NMP disabled
      for plies `< ss->ply + 3·(depth-R)/4` via `nmpMinPly`; return only if
      verification also fails high.
12. **Internal iterative reduction**: if `!followPV && !allNode && depth >= 6
    && !ttMove` → `depth--`. (PV/cut nodes without a TT move get shallower
    instead of doing an internal search.)
13. **ProbCut**: `probCutBeta = beta + 214 - 59·improving`; requires
    `depth >= 3 && !is_decisive(beta)` and not
    (valid ttValue below probCutBeta). A MovePicker in ProbCut mode yields
    only captures/queen-promotions with `SEE >= probCutBeta - staticEval`;
    each is verified by qsearch at `(-probCutBeta, -probCutBeta+1)` and, if
    that holds and `probCutDepth = depth - 4 - improving > 0`, by a real
    search at that depth. On success: store `BOUND_LOWER` at
    `probCutDepth+1` and return `value - (probCutBeta - beta)`.
14. **Small in-check ProbCut** (runs even in check, just before the move
    loop): `probCutBeta = beta + 428`; if the TT holds
    `BOUND_LOWER, ttDepth >= depth-4, ttValue >= probCutBeta` (nothing
    decisive) → return `probCutBeta` immediately.

### 4.1 The move loop — per-move pruning (before making the move)

MovePicker supplies moves lazily (§8). Skip the excluded move, illegal moves,
and (at root) moves outside the current MultiPV band. `newDepth = depth - 1`.
Base reduction `r = reduction(improving, depth, moveCount, beta-alpha)` (§5),
`r += 1006` if `ss->ttPv`.

Pruning block requires `!rootNode && non_pawn_material(us) &&
!is_loss(bestValue)`:

- **Late move pruning**: once
  `moveCount >= (3 + depth²) / (2 - improving)` → `mp.skip_quiet_moves()`
  (later quiets never even get generated; captures still come).
- `lmrDepth = newDepth - r/1024` — the *predicted* post-LMR depth, used as
  the depth input to the margins below.
- **Captures/checks** (`captHist = captureHistory[pc][to][capturedType]`):
  - futility (non-checking, `lmrDepth < 7`): skip if
    `staticEval + 231 + 232·lmrDepth + PieceValue[captured]
     + 131·captHist/1024 <= alpha`;
  - SEE: skip if `!see_ge(move, -(175·depth + 34·captHist/1024))` — except
    when `alpha < 0` and the move sacrifices the last non-pawn piece
    (stalemate-trap guard).
- **Quiets** (only when `!followPV || !PvNode`):
  - `history = contHist(1) + contHist(2) + pawnHistory[pc][to]`;
    **continuation-history pruning**: skip if `history < -4313·depth`;
  - then `history += 2·mainHistory` and
    `lmrDepth += history / lmrDivisor[min(depth,16)-1]` (16-entry tuned table
    ≈ 2800-3600) — good history buys effective depth for the margins;
  - **futility (parent)**: if `!inCheck && lmrDepth < 12` and
    `staticEval + 40 + 138·(!bestMove) + 117·lmrDepth
     + 90·(staticEval > alpha) <= alpha` → skip (and let that futility value
    raise `bestValue` if larger);
  - **SEE**: skip if `!see_ge(move, -25·lmrDepth²)`.

### 4.2 Singular extensions / multicut (search.cpp:1218-1280)

Trigger: `move == ttMove`, `!rootNode`, no excluded move,
`depth >= 6 + ss->ttPv`, ttValue valid & non-decisive, `ttBound & LOWER`,
`ttDepth >= depth - 3`, and not `is_shuffling` (a rule50 ≥ 10 guard against
extending back-and-forth shuffle moves).

- `singularBeta = ttValue - (60 + 70·(ss->ttPv && !PvNode))·depth/59`;
  exclusion search at the **same ply** with `ss->excludedMove = ttMove`,
  window `(singularBeta-1, singularBeta)`, depth `newDepth/2`.
- `value < singularBeta` → ttMove is singular: extension = 1, 2, or 3
  depending on how far below (`doubleMargin`/`tripleMargin` are functions of
  PvNode, ttCapture, |correctionValue|, ttMoveHistory, ply>rootDepth), and
  the *whole node* gets `depth++`.
- `value >= beta` (non-decisive) → **multicut**: several moves beat beta;
  return `value` immediately and dock the scalar
  `ttMoveHistory << (-442 - 108·depth)`.
- Else negative extensions: `ttValue >= beta` → `extension = -3`;
  `cutNode` → `extension = -2`.

There are **no plain check extensions**. The only other depth bump: at the PV
re-search, if `move == ttMove` and its TT entry is deep/decisive, clamp
`newDepth = max(newDepth, 1)` (don't dive straight into qsearch).

### 4.3 Make move, finalize the reduction

`do_move` bumps the node counter, installs the child's continuation-history
pointers keyed `[inCheck][capture][piece][to]`, and prefetches the child's TT
line. Then `newDepth += extension` and `r` gets adjusted (all in 1024ths):

- `ss->ttPv`: `r -= 2766 + 1017·PvNode + 838·(ttValue > alpha)
  + (ttDepth >= depth)·(923 + 955·cutNode)`;
- flat: `r += 714`; `r -= 62·moveCount`; `r -= |correctionValue|/26131`;
- cutNode: `r += 3995 + 1059·(!ttMove)`;
- ttMove is a capture: `r += 1039`;
- child fail-high count: if `(ss+1)->cutoffCnt > 1`:
  `r += 236 + 1079·(cutoffCnt > 2) + 1143·allNode`; **else if**
  `move == ttMove`: `r = max(0, r - 2016)`;
- `ss->statScore` = captures: `809·PieceValue[captured]/128 + captHist`;
  quiets: `2·mainHistory + contHist(1) + contHist(2)`; then
  `r -= statScore·445/4096`;
- all-node scaling: `r += r·272/(256·depth + 285)`.

### 4.4 LMR search (search.cpp:1330-1361)

If `depth >= 2 && moveCount > 1`:

- `d = max(1, min(newDepth - r/1024, newDepth + 2)) + PvNode` — negative
  net reductions may search up to 2 plies *deeper*; PV nodes +1.
- Record `ss->reduction = newDepth - d` (the child reads it for its hindsight
  adjustment), zero-window search at depth `d` with `cutNode = true`.
- If the reduced search beats alpha: "**deeper/shallower**" —
  `newDepth += (value > bestValue + 52 && d < newDepth) - (value < bestValue + 9)`;
  if `newDepth > d`, re-search full-depth zero-window with `!cutNode`.
  The move's continuation histories always get a flat `+1415` bonus here.

Else if `!PvNode || moveCount > 1` (LMR conditions not met): full-depth
zero-window at `newDepth - (r > 5039) - (r > 5223 && newDepth > 2)` with
`!cutNode` (`r += 1085` first if no ttMove) — very high nominal reduction
still costs 1-2 plies.

**PV re-search**: for PV nodes, when `moveCount == 1 || value > alpha`, search
with the full window `(-beta, -alpha)` at `newDepth`.

### 4.5 Undo, bookkeeping (search.cpp:1392-1497)

- If stopped: return `VALUE_ZERO` immediately (untrusted subtree).
- Root: update the move's `effort` (nodes spent), `averageScore` and
  `meanSquaredScore` EMAs, score/bounds/PV; `++bestMoveChanges` when a
  non-first move becomes best (TM instability signal).
- **Equal-value dither**: near the leaves
  (`ply + 2 >= rootDepth && (nodes & 14) == 0`), an equal-to-best value is
  occasionally treated as strictly better — deliberate diversity.
- On `value > bestValue`: update; on `value > alpha`: set `bestMove`, extend
  `ss->pv`; on `value >= beta`: `ss->cutoffCnt += (extension < 2) || PvNode`,
  break. Otherwise (alpha raise, no cutoff): if `2 < depth < 13` and value
  non-decisive → `depth -= 2` (remaining siblings get a cheaper search);
  `alpha = value`.
- Non-best moves are recorded (up to 32 quiets + 32 captures) for maluses.

### 4.6 Node result (search.cpp:1499-1586)

- **Fail-high softening**: `bestValue = (bestValue·depth + beta)/(depth+1)`
  (when neither value decisive).
- No legal moves: `excludedMove ? alpha : inCheck ? mated_in(ply) : VALUE_DRAW`.
- If a bestMove exists: `update_all_stats` (§9.3); at non-PV nodes also
  `ttMoveHistory << (bestMove == ttMove ? 792 : -779)`.
- **Fail-low bonus for the parent's move** (no bestMove): if the previous move
  was quiet, it gets a bonus scaled by
  `bonusScale = max(0, -245 - (ss-1)->statScore/98 + min(59·depth, 430)
  + 191·((ss-1)->moveCount > 8) + 143·(!inCheck && bestValue <= staticEval-103)
  + 151·(!(ss-1)->inCheck && bestValue <= -(ss-1)->staticEval-78))`,
  `scaledBonus = min(141·depth - 82, 1472)·bonusScale`, distributed to
  conthist (×236/16384), mainHistory (×234/32768), pawn history (×322/8192).
  If it was a capture instead: flat `captureHistory << 901`.
- `bestValue = min(bestValue, maxValue)` (TB cap).
- **ttPv back-propagation**: on fail-low, `ss->ttPv |= (ss-1)->ttPv`.
- **TT store** (skipped for excluded-move searches and non-first MultiPV root
  lines): bound = LOWER on fail-high, EXACT for PV node with bestMove, else
  UPPER; depth = `depth` (or `depth+6` when `moveCount == 0`); value via
  `value_to_tt`; eval = the **raw, uncorrected** static eval.
- **Correction history update** (§10.2): if not in check, bestMove not a
  capture, and error direction consistent
  (`(bestValue > staticEval) == bool(bestMove)`):
  `bonus = clamp((bestValue - staticEval)·depth·(bestMove ? 12 : 18)/128, ±256)`
  distributed over pawn/minor/non-pawn/continuation correction tables.

---

## 5. The reduction function (search.cpp:693-694, 1830-1833)

Table (per thread): `reductions[i] = int(2834/128 · ln(i))` for
`i in 1..MAX_MOVES` — a single 1-D table indexed by depth *or* move number.

```
reduction(improving, depth, moveNumber, delta):
    scale = reductions[depth] * reductions[moveNumber]
    r = scale
        - delta * 617 / rootDelta          // wide local window vs root → reduce less
        + (!improving) * scale * 194/512   // not improving → ~38% more reduction
        + 1027
```

`r` is then adjusted by the long list in §4.3 and divided by 1024 at use.

---

## 6. Quiescence search (search.cpp:1596-1828)

`qsearch<PV|NonPV>` — no Root instantiation, no depth parameter (everything
stores/probes at `DEPTH_QS = 0`).

1. Upcoming-repetition + draw + MAX_PLY checks like the main search.
2. **TT probe**: non-PV cutoff needs only `ttDepth >= DEPTH_QS`, a valid
   value, and a matching bound.
3. **Stand pat** (not in check): corrected static eval exactly as in the main
   search (TT eval reused when valid; TT value substitutes when its bound
   allows). If `bestValue >= beta`: soften to
   `(467·bestValue + 557·beta)/1024`, store an eval-only `BOUND_LOWER` entry
   on a TT miss, return. Else `alpha = max(alpha, bestValue)` and
   `futilityBase = staticEval + 335`. In check: `bestValue = -INF` (no stand
   pat, all evasions searched).
4. **Moves**: MovePicker at `DEPTH_QS` yields captures/queen-promotions only —
   or all evasions when in check. **There is no "quiet checks" stage in
   qsearch anymore.** Only the ply-1 continuation history is passed.
5. **Per-move pruning** (only once `bestValue` is not a loss — so evasions in
   check aren't pruned until one legal escape has produced a real value); for
   non-checking, non-promotion moves not aimed at the previous move's square:
   - stop trying after 2 such moves (`moveCount > 2`);
   - futility: `futilityValue = futilityBase + PieceValue[captured]`; if
     `<= alpha` skip (and let it raise `bestValue`);
   - SEE vs window: skip if `!see_ge(move, alpha - futilityBase)`;
   - remaining non-captures are skipped entirely;
   - hard SEE floor for everything: skip if `!see_ge(move, -74)`.
6. Recurse with the full `(alpha, beta)`; standard updates.
7. **Mate**: no moves while in check → `mated_in(ply)`. A narrow stalemate
   test (no pawn pushes, no non-pawn material, previous capture ≥ knight)
   returns `VALUE_DRAW`.
8. Fail-high softening `(481·bestValue + 543·beta)/1024`; TT store with
   bound LOWER/UPPER (never EXACT in qsearch), depth `DEPTH_QS`.

---

## 7. Move generation staging (movegen)

`GenType`s: `CAPTURES` = captures **plus queen promotions** (even
non-capturing ones); `QUIETS` = non-captures **plus underpromotions**;
`EVASIONS` = check evasions (double check → king moves only);
`NON_EVASIONS`; `LEGAL`. `Position::capture_stage(m)` mirrors the split
(true for any capture or queen promotion) and is the predicate search uses to
classify moves consistently with the generator ("ttCapture", stats, etc.).

---

## 8. Move ordering — MovePicker (movepick.{h,cpp})

A **lazy staged generator**: fixed `ExtMove[MAX_MOVES]` buffer (`Move` +
`int value`), a stage counter, and segment pointers. `next_move()` emits one
pseudo-legal move at a time; the caller checks legality. If the search fails
high on the TT move, nothing is ever generated — the big win of staging.

### 8.1 Stages

```
Main search:  MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE → QUIET_INIT
              → GOOD_QUIET → BAD_CAPTURE → BAD_QUIET
Evasions:     EVASION_TT → EVASION_INIT → EVASION
ProbCut:      PROBCUT_TT → PROBCUT_INIT → PROBCUT
Qsearch:      QSEARCH_TT → QCAPTURE_INIT → QCAPTURE
```

Constructor: in check → evasion staging; else `depth > 0` → main staging,
`depth <= 0` → qsearch staging. The `*_TT` stage is skipped when the ttMove
isn't pseudo-legal (ProbCut additionally requires it to be a capture-stage
move). Emission order overall: **TT move, good captures, good quiets, bad
captures, bad quiets** — bad captures outrank bad quiets.

- `CAPTURE_INIT`: generate CAPTURES, score, **full sort**
  (`partial_insertion_sort` with limit `INT_MIN`).
- `GOOD_CAPTURE`: emit if `see_ge(move, -value/18)` — the good/bad split is a
  *score-dependent* SEE threshold: high-history/high-victim captures may lose
  a little material and still count as good; failures are swapped down into a
  bad-captures segment for later.
- `QUIET_INIT`: unless `skip_quiet_moves()` was called (LMP), generate
  QUIETS, score, and sort **only moves scoring ≥ `-3560·depth`** (deeper
  nodes sort more of the list; the tail stays unsorted).
- `GOOD_QUIET`: emit quiets with `value > -14000`; `BAD_QUIET` later emits
  the rest.
- `PROBCUT`: emit only captures with `see_ge(move, threshold)` where
  threshold = `probCutBeta - staticEval`.
- `select()` always skips the ttMove (already emitted) — dedup is a simple
  pointer comparison.

`partial_insertion_sort(begin, end, limit)`: one pass; elements with
`value >= limit` are insertion-sorted into a growing sorted prefix,
the rest left unsorted in the tail. Descending. (There's an AVX-512
16-at-a-time variant; scalar fallback.)

### 8.2 Scoring (movepick.cpp:194-261)

Piece values (types.h): **P 208, N 781, B 825, R 1276, Q 2538** (used for
both ordering and SEE).

**Captures**: `7·PieceValue[captured] + captureHistory[pc][to][capturedType]`
— pure MVV (no LVA term) plus history; the victim dominates but history
reorders within a victim class and drives the `-value/18` SEE gate.

**Quiets**:

```
value = 2·mainHistory[us][move.raw16]
      + 2·pawnHistory[pawnKey & mask][pc][to]
      + contHist(1) + contHist(2) + contHist(3) + contHist(4) + contHist(6)
```

(ply-5 is read for *updates* but not for ordering), then:

- **check bonus**: `+16384` if the destination is in
  `check_squares(pieceType)` and `see_ge(move, -75)`;
- **threat escape/entry**: with `threatByLesser[pt]` = squares attacked by
  strictly cheaper enemy pieces (N/B threatened by pawns; R by minors+pawns;
  Q by rooks and below), `value += 20·PieceValue[pt]·(from∈threat) -
  20·PieceValue[pt]·(to∈threat)` — fleeing a cheap attacker is a big bonus,
  stepping into one buries the move below even bad quiets;
- **low-ply history**: `+ 8·lowPlyHistory[ply][raw]/(1+ply)` for `ply < 5`.

**Evasions**: capturing evasions get `PieceValue[captured] + 2^28` (always
above quiet evasions); quiet evasions `mainHistory + contHist(1)`.

---

## 9. History tables and updates

### 9.1 The tables (history.h)

All are `StatsEntry<T, D>` grids; `D` bounds values in `[-D, D]` via gravity
(§9.2). NUMA-shared tables use relaxed atomics. Init values from
`Worker::clear()` (search.cpp:673-697).

| Table | D | Shape / indexing | Init |
|---|---|---|---|
| mainHistory (butterfly) | 7183 | `[color][65536]` — indexed by the **full 16-bit move**, not just from-to | fill(−5), decayed ×789/1024 each new search |
| lowPlyHistory | 7183 | `[5][65536]` — plies 0-4 only | fill(100) each `go` |
| captureHistory | 10692 | `[piece][to][capturedType]` | fill(−699) |
| continuationHistory | 30000 | `[inCheck][capture]` → `[piece][to]` → `PieceToHistory[piece][to]` | fill(−552) |
| pawnHistory | 8192 | `[pawnKey & 8191][piece][to]` (8192 slots/thread) | fill(−1262) |
| correction history (pawn/minor/nonPawnW/nonPawnB) | 1024 | one 65536-slot hash table, indexed by the matching specialized key `& mask`, then `[stm]` | fill(−6) |
| continuationCorrectionHistory | 1024 | `[piece][to]` → `[piece][to]`; read at (ss−2), (ss−4) | fill(5) |
| ttMoveHistory | 8192 | **one scalar per thread** | 0 |
| killers / countermoves | — | **removed entirely** | — |

### 9.2 The gravity operator

```
entry << bonus:
    b = clamp(bonus, -D, D)
    entry += b - entry·|b| / D
```

Self-decaying: entries asymptote to ±D and repeated same-sign bonuses have
diminishing effect. This one primitive drives every table. (Rust: a method on
`StatsEntry<const D: i32>`.)

### 9.3 update_all_stats — on a bestMove that raised alpha (search.cpp:1898)

```
bonus = min(134·depth - 79, 1572) + 382·(bestMove == ttMove) + (ss-1)->statScore/30
malus = min(1005·depth - 205, 2218)
if !PvNode: bonus += bonus·(quietsSearched + capturesSearched)/256
```

- Quiet best move → `update_quiet_histories(bonus·824/1024)`; every other
  searched quiet gets a **geometrically decaying malus** (start
  `malus·1136/1024`, ×956/1024 per successive move — earlier failures
  punished hardest).
- Capture best move → `captureHistory << bonus·1366/1024`. All non-best
  captures → `<< -malus·1518/1024`.
- Refutation malus: if the parent's move was quiet and was its first try
  (`(ss-1)->moveCount == 1 + (ss-1)->ttHit`), its continuation entries get
  `-malus·683/1024`.

`update_quiet_histories`: mainHistory ×1; lowPlyHistory ×663/1024 (ply < 5);
continuation ×820/1024; pawn history ×1038/1024 for bonuses but only
×525/1024 for maluses.

`update_continuation_histories`: offsets/weights
`{1:1040, 2:780, 3:300, 4:537, 5:129, 6:423}` (÷131072 with the bonus); in
check only offsets 1-2 update; a "consistency" multiplier
(`{96,113,101,105,127,121,126}`) grows with the count of already-positive
entries along the chain; offset 1 gets a flat `+71`.

Other write sites: TT-cutoff bonus `min(114·depth, 724)` (§4 step 6);
eval-diff nudge (§4 step 8); post-LMR flat `+1415` (§4.4); fail-low
parent bonuses (§4.6); correction history (§10.2).

---

## 10. Static evaluation, correction history, optimism

### 10.1 evaluate (evaluate.cpp:46-63)

`v` from NNUE: `nnue = psqt + positional`;
`complexity = |psqt - positional|`; `optimism += optimism·complexity/476`;
`nnue -= nnue·complexity/18236`;
`material = 534·pawnCount + nonPawnMaterial`;
`v = (nnue·(77871 + material) + optimism·(7191 + material)) / 77871`;
then rule-50 damping `v -= v·rule50/199` and clamping inside the non-TB
range. (For our engine pre-NNUE: the shape to keep is *eval blended with a
side-to-move optimism term that scales with material*.)

### 10.2 Correction history

Corrects the static eval by learned "eval was systematically off in
positions like this" deltas, keyed by structural hashes:

- read (search.cpp:83-105): `cv = 13345·pawnCorr + 9280·minorCorr
  + 11840·(nonPawnW + nonPawnB) + 8363·(contCorr(ss-2) + contCorr(ss-4))`;
  `corrected = clamp(raw + cv/131072, ±(TB range))`.
- write (§4.6): error `(bestValue - staticEval)` scaled by depth, clamped to
  ±256, distributed with per-table weights (pawn ×1, minor ×152/128,
  non-pawn ×186/128, cont(ss−2) ×136/128, cont(ss−4) ×68/128).
- `|correctionValue|` also serves as an **eval-uncertainty signal**: it widens
  the RFP margin, shrinks singular extension margins, and decreases LMR.

The corrected eval feeds all pruning decisions; the **raw** eval is what gets
stored in the TT.

### 10.3 Optimism

Set per aspiration window (§3.1): `optimism[us] = 137·avg/(|avg|+81)`,
negated for the opponent; passed into every `evaluate()` call. The side that
is winning at the root evaluates positions slightly optimistically —
sharpens conversion.

---

## 11. Transposition table (tt.{h,cpp})

### 11.1 Entry & cluster layout

**10 bytes per entry**, fields in read order: `key16` (low 16 bits of the
Zobrist key), `depth8` (stored as `d - DEPTH_NONE`, so 0 = empty — occupancy
check is just `bool(depth8)`), `genBound8`
(= `generation | bound << 5 | is_pv << 7`; 5 generation bits), `move16`,
`value16`, `eval16`. Every field is a relaxed atomic.

**Cluster = 3 entries + 2 bytes padding = 32 bytes** (`static_assert`), so
clusters divide the 64-byte cache line and one prefetch pulls the whole probe
target.

### 11.2 Probing

- Index: `table[mul_hi64(key, clusterCount)]` where
  `mul_hi64(a,b) = ((u128)a·b) >> 64` — maps a uniform 64-bit key onto
  `[0, clusterCount)` with one multiply, no modulo, no power-of-two
  requirement. The full key indexes; the *low* 16 bits are the in-cluster tag.
- Linear scan of 3 entries for `key16 == u16(key)`. Hit flag is
  `is_occupied()`, not just tag equality.
- Only 16 tag bits ⇒ genuine collisions (~1 per 2^16 probes). Accepted by
  design; search defends itself by re-validating `ttMove` with
  `pseudo_legal() && legal()` and checking `is_valid(ttValue)`.
- Miss: return the replacement victim and a blank `TTData`.

### 11.3 Replacement & aging

- `new_search()`: `++generation8 &= 0x1f` (cycle 32, bumped per `go`).
  Relative age = `(curGen - genBound8) & 0x1f` (clock arithmetic).
- **Victim on miss**: entry minimizing `depth8 - 8·relative_age` — one
  generation of staleness ≈ 8 plies of depth.
- **Overwrite test in save()** — write if any of:
  bound is EXACT; different key; `newDepth + 2·isPv > storedDepth - 4`
  (same position may refresh with up to ~4 less depth); entry is from an
  older search.
- **TT-move preservation**: a same-key write with a null move keeps the old
  move.
- **Secondary aging**: a rejected write against a deep non-EXACT *decisive*
  entry decrements its depth by 1 — slowly erodes stale mate-ish entries.
- `TTWriter.penalize(n)`: decrement depth8 (clamped at 0 = evicted); used by
  search on entries whose bound direction keeps mismatching the window.
- `hashfull`: sample the first 1000 clusters, count current-generation
  occupied entries, `/3` → permille.

### 11.4 SMP safety

One shared table, **lock-free and racy by design** — synchronization would
cost more than rare corrupted entries. Individual fields are relaxed atomics
(no torn fields, UB-free); the entry as a whole is *not* atomic, so a read
may be internally inconsistent. All consumers validate (legality check on
ttMove, `is_valid` on values, clamps on depth). The old "xor key with data"
trick is gone.

Rust mapping: `#[repr(C)]` struct of `AtomicU16/AtomicU8/AtomicI16` with
`Ordering::Relaxed` — exactly the sanctioned equivalent.

### 11.5 Memory

`clusterCount = MB·1024·1024/32`; allocation prefers huge pages (1 GiB
explicit or 2 MiB THP + madvise). `clear()` is parallelized: each thread
memsets a contiguous slice (NUMA first-touch placement).

---

## 12. Zobrist keys (position.cpp)

PRNG-seeded (`1070372`) tables:

- `psq[piece][square]`; **pawn keys on ranks 1/8 are zero** so promotion
  needs no "remove pawn from 8th" xor.
- `enpassant[file]` (8 keys); `castling[rights]` — **16 keys, one per 4-bit
  rights subset** (update = xor out old subset, xor in new).
- `side` xored when Black is to move; `noPawns` baseline for the pawn key.

**En passant is keyed only when a legal ep capture is actually possible**
(including discovered-pin checks) — matters for repetition detection and TT
correctness.

Specialized keys maintained incrementally in `StateInfo` alongside the main
key, each consumed by a specific table:

| key | hashes | used by |
|---|---|---|
| `key` | everything | TT, repetition/cuckoo |
| `pawnKey` | pawns only | pawn history, pawn correction history |
| `materialKey` | piece counts (psq[pc][8+i] trick) | Syzygy, endgame recognition |
| `minorPieceKey` | knights+bishops+? (pt <= BISHOP) | minor correction history |
| `nonPawnKey[color]` | per-color non-pawns incl. king | non-pawn correction history |

**Rule-50 folding**: `key()` xors in `make_key((rule50 - 14)/8)` once
`rule50 >= 14` — the TT key changes every 8 halfmove-clock steps, so
"same position, very different 50-move clock" gets distinct entries (GHI
mitigation; combined with the "no TT cutoff at rule50 ≥ 96" rule).

**Prefetch**: `do_move` computes the new key and prefetches the TT line (and
pawn/correction-history entries) *before* updating the board; the search's
`do_move` prefetches even earlier via a cheap key prediction that ignores
castling/ep/promotion.

---

## 13. Threading — lazy SMP (thread.{h,cpp})

### 13.1 Architecture

N identical searchers on the same position; **no tree splitting, no
synchronization** — threads diverge naturally (aspiration jitter
`threadIdx % 8`, TT races) and the shared TT propagates discoveries.

- Per-`Worker` (per thread): **all** histories, own copy of `rootPos` +
  `rootMoves` (own scores/PVs), relaxed-atomic `nodes/tbHits/
  bestMoveChanges` counters, reduction table, NNUE accumulator stack,
  `nmpMinPly`, limits copy. (Newest SF shares some history tables per NUMA
  node; for a first engine treat everything as per-thread.)
- Shared: the TT, `atomic_bool stop` and `increaseDepth`, options, network.
- Thread 0 = main thread: owns the real `SearchManager` (time management,
  UCI output, `check_time`); helpers get a no-op `NullSearchManager` and
  simply run `iterative_deepening` until they observe `stop`. This Null
  Object pattern means the search code has no `if is_main` scattering beyond
  `mainThread` blocks in the ID loop.
- Threads park on a per-thread `mutex + condvar` job loop (Rust: channel or
  `Mutex<Option<Job>> + Condvar`).

`start_thinking`: wait for previous search → clear stop flags → build root
move list (searchmoves filter, Syzygy root ranking) → per-thread setup jobs
(copy limits/rootMoves, zero counters) → wake **only the main thread**; the
main worker inits TM, bumps TT generation, wakes the helpers, searches, then:
busy-wait while pondering/infinite → `stop = true` → wait for helpers → vote
→ print `bestmove` (ponder move from PV or extracted from TT).

### 13.2 Best-thread voting (thread.cpp:350-401)

- `minScore` = min over threads' best scores; each thread votes
  `votes[its best move] += score - minScore + 14`.
- Pick: decisive (TB-win-or-better, non-bound) scores are protected — only a
  *shorter* mate/faster win displaces one; otherwise a candidate wins on
  strictly more votes, or equal votes and a longer PV (depth proxy).
- Skipped under `go depth` (bench reproducibility) and skill level.

### 13.3 Time management (timeman.cpp:46-135 + search.cpp:565-610)

**Allocation at `go`** (main thread): `mtg = movestogo ? min(movestogo, 50)
: 50` (scaled down below 1s); `timeLeft = max(1, time + inc·(mtg-1)
- moveOverhead·(2+mtg))`. For sudden-death/increment: fitted log formulas —
`optScale ≈ min(0.0121 + (ply + 3.23)^0.4687 · optConstant,
0.194·time/timeLeft) · originalTimeAdjust` with
`optConstant ≈ min(0.00299 + 0.000336·log10(scaledTime/1000), 0.0049)`;
`maxScale = min(6.87, maxConstant + ply/12.35)`. Then
`optimumTime = optScale·timeLeft` and
`maximumTime = max(optimum, min(0.81·time - moveOverhead,
maxScale·optimum))`. Ponder on: `optimum += optimum/4`. (`slowMover` is
gone.) `nodestime` mode replaces wall time by a node budget for reproducible
testing.

**Hard stop** — `check_time`, main thread, every 512 node-entries: stop when
`elapsed > maximum`, movetime reached, or aggregated node limit reached.
Never stops while `ponder` is set (converts to `stopOnPonderhit`).

**Soft stop** — between iterations, multiply `optimum` by four factors:

- `fallingEval = (11.87 + 2.21·(prevAvgScore - bestValue)
  + 1.0·(recentIterScore - bestValue))/100`, clamped [0.572, 1.708] — think
  longer when the score is dropping;
- `reduction = (1.48 + previousTimeReduction)/(2.157·timeReduction)` where
  `timeReduction ∈ [0.65, 1.55]` ramps with how many iterations the best
  move has been stable (and last move's savings are partially given back);
- `bestMoveInstability = 1.096 + 2.29·totBestMoveChanges/threads`
  (change counter summed over threads, halved each iteration);
- `highBestMoveEffort ∈ [0.71, 0.924]` — if ~80%+ of nodes already went into
  the best root move, stop sooner.

Stop when `elapsed > min(totalTime, maximum)`, or all PV lines are
`mate_in(3)`+, or root is `mated_in(2)`. One legal move: cap at ~562 ms.
Otherwise `increaseDepth = ponder || elapsed <= totalTime·0.50` — past half
budget, the next "iteration" re-searches near the same effective depth
(`searchAgainCounter`) instead of going deeper. A root **fail-low resets
`stopOnPonderhit`**.

### 13.4 MultiPV, ponder, limits

- MultiPV: for `pvIdx in 0..multiPV`, each line gets its own aspiration
  window; root moves before `pvIdx` are skipped in the root move loop;
  non-current lines carry `-VALUE_INFINITE` and stable sort preserves their
  order. Root moves are also banded by Syzygy `tbRank` (`pvFirst/pvLast`).
- Ponder: `go ponder` keeps the clock in the GUI's hands (`check_time` never
  stops); `ponderhit` clears the flag; if the budget was already burnt,
  `stopOnPonderhit` stops instantly. `bestmove` is never printed early —
  busy-wait if the loop ends while pondering/infinite.
- Limits: `depth` (main thread only, voting disabled), `nodes` (aggregated,
  checked in check_time), `movetime`, `mate n`, `infinite`, `searchmoves`,
  `perft`.

### 13.5 UCI info & score normalization

`info` lines per PV index after each completed line (throttled: extra
updates and currmove output only after 10M nodes): depth, seldepth, multipv,
score cp/mate (+ lowerbound/upperbound, optional wdl), time, nodes, nps,
hashfull, tbhits, pv. Internal values are normalized so **+100 cp ⇔ 50% win
probability** via a material-dependent fitted WDL model
(`to_cp(v) = 100·v/a(material)`); mate scores print `mate ±(plies+1)/2`.

---

## 14. What to implement, in what order (Rust roadmap)

The dependency-ordered skeleton, each step validated before the next
(fixed-depth self-play or node-count regression; perft already covers
movegen/make-unmake):

1. **Negamax + alpha-beta + qsearch(captures) + MVV ordering + mate distance
   pruning**, iterative deepening, `value_to_tt/from_tt` semantics from day
   one (even before the TT exists, keep mate scores ply-relative).
2. **TT** (§11): 10-byte entry, 3-entry 32-byte clusters, `mul_hi64` index,
   gen/bound/pv packing, the replacement rules. Single-threaded first; the
   relaxed-atomic layout costs nothing and is ready for SMP.
3. **PVS zero-window + aspiration windows** (§3.1).
4. **Move ordering** (§8): staged MovePicker (TT → good captures → quiets →
   bad captures), partial insertion sort, SEE (`see_ge` with the swap
   algorithm, pins, x-rays — §values P208/N781/B825/R1276/Q2538), butterfly
   history + capture history + the gravity update.
5. **The big pruning trio**: null move (+ verification), reverse futility,
   LMR with the log-table `reduction()` — then LMP, futility, SEE pruning,
   razoring, IIR roughly in that order.
6. **Continuation history** (+ per-ply conthist pointers installed in
   do_move) and history-driven LMR adjustments; killers/countermoves are
   *not needed* — SF removed them.
7. **Singular extensions + multicut** (needs a solid TT and stable search).
8. **Time management** (§13.3): optimum/maximum split, the four soft-stop
   factors, check_time every 512 nodes.
9. **Lazy SMP** (§13.1): Worker-owns-everything, shared TT + stop flag,
   voting. Rust note: the racy TT is exactly `AtomicU16/I16` fields with
   `Ordering::Relaxed`; everything else stays single-owner (no locks in the
   search path).
10. **Refinements, each worth real Elo but separable**: correction history
    (needs pawn/minor/non-pawn keys — wire those into `put_piece`/
    `remove_piece` alongside the main Zobrist key *now*, they're cheap),
    pawn history, low-ply history, ttMoveHistory, optimism, value softening
    blends, hindsight depth adjustments, ProbCut, upcoming-repetition
    cuckoo tables, Syzygy.

Board-layer prerequisites worth adopting immediately (they're hard to
retrofit): the specialized Zobrist keys (§12), ep-key-only-when-capturable,
zeroed promotion-rank pawn keys, rule-50 key folding, and TT prefetch hooks
in `do_move`.