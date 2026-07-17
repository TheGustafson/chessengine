# S0 — SF16-late mechanics audit (current-code verification, 2026-07-16)

Extends `06-sf14-sf16-search-patch-study.md` §6a/§6b with a line-by-line
verification pass against the **shipped** `src/search/mod.rs` (8252 lines) +
`params.rs` + `stack.rs` + `timeman.rs`, cross-checked against the reference
clone `stockfish_reference/Stockfish/src/search.cpp` at commit `eca43a97`
(2026-07-06). **Key finding on method:** the reference clone is the *same*
recent-master tree that `design/10` was written against (doc 10 §2 cites
`eca43a97`), so doc 10 is a faithful spec of current SF master, and every
mechanic doc 10 marks DEFERRED is a genuine gap that exists in current SF.
Ideas only — no SF code copied (GPLv3).

**SF-era column:** "SF16" = present by Stockfish 16 (in the 06 study's
sf_14..sf_16 mining, §2/§3). "post-SF16" = only in current master, OR an
SF16-era idea whose *current tuned form/constants* are post-SF16 refinements.

## 1. Verified inventory

Status: LIVE / PARTIAL / DEFERRED (params exist, unconsumed) / ABSENT.
Line refs are `src/search/mod.rs` unless noted. Elo est. from 06 study.

| Mechanic | Status | Where / evidence | SF-era | Elo est. |
|---|---|---|---|---|
| PVS + aspiration (avg-score centre, prev² widen) | LIVE | asp loop, `delta = 5 + \|mss\|/10588` (doc-comment 1683) | SF16 | — |
| TT probe + non-PV cutoff (depth slack, cutNode agree, rule50<96, depth≥7 child-verify, wrong-dir penalize) | LIVE | 2052–2151 (`tt_cut_base`/`base_cutoff` 2052–2062) | SF16 | — |
| Static eval 3-path + TT-value-as-better-bound + excluded-move reuse | LIVE | 1957–2004 | SF16 | — |
| **Correction history — read corrected eval + write update** | **LIVE** | fold 1985 (main) + 3639 (qsearch stand-pat); update 3375; commit `a8e16c5` (S1) | post-SF16 | +8–20 |
| improving / opponent_worsening + `improving \|= eval>=beta` | LIVE | 2004, 2017; late-OR present | SF16 | — |
| Hindsight ±depth (prior_reduction) | LIVE | N6 block ~2019–2050 | post-SF16 | — |
| Eval-diff parent nudge (N8) | LIVE | item12 test 8238; N8 site | SF16 | — |
| Razoring (N9, quadratic margin) | LIVE | 2238 (`eval < alpha - 465 - 300·d²`) | SF16 | — |
| Reverse futility (N10, blend-toward-beta return, corr term) | LIVE | 2243–2276; `rfp_prune_margin` 205–216 | SF16 | — |
| Null-move pruning + verification + `nmp_min_ply` | LIVE | 2298–…; field 757 | SF16 | — |
| Internal iterative reduction (N12) | **ABSENT** | reverted; no `depth--`, no `IIR_*` const; branch `wip/p9-item9-iir-followpv`. Ref has it: `depth>=6 && !ttMove && !followPV && !allNode` (search.cpp:1030) | SF16 | 0–8 |
| ProbCut (N13) + in-check small ProbCut (N14) | LIVE | 2372–2422; 2427 | SF16 | — |
| Singular extension + `depth++` | LIVE | 2742–2787 | SF16 | — |
| Multicut + negative extensions (−3/−2) + tt_move_history + is_shuffling | LIVE | 2788–2801; multicut 2788–2794; neg-ext 2795–2800 | SF16 | +15–40 (class) |
| double/triple extension margins (corr_adj, tmh, deep) | LIVE | 2772–2786; helpers 375–410 | post-SF16 | — |
| LMR core + reduction table + `ss.reduction` | LIVE | scout 2914–2943; `reduction()` params | SF16 | — |
| doDeeper / doShallower re-search ladder | LIVE | 2955–2984 (`do_deeper`/`do_shallower` 2956–2958) | SF16 | — |
| LMR post-scout conthist +1415; ttMove min-depth clamp | LIVE | 2977–2983; clamp at PV re-search | SF16/post | — |
| cutoffCnt **increment** on fail-high | LIVE | 3166 (`cutoff_cnt += extension<2 \|\| PV`) | SF16 | — |
| Reduction finalization: RED_FLAT, −MOVECOUNT, −corr/26131, cutNode, ttMove−2016, −statScore·445/4096 | LIVE | 2861–2878 | SF16 | — |
| **Reduction: ttPv rebate + entry bump** (RED_TTPV_*) | **DEFERRED** | consts 374–384 unused; ref search.cpp:1291 | post-SF16 | part of retune |
| **Reduction: tt_capture +1039** (RED_TTCAPTURE) | **DEFERRED** | const 386 unused; ref 1304 | SF16 | part of retune |
| **Reduction: cutoffCnt surcharge** (RED_CUTOFFCNT_*) | **DEFERRED** | consts 642–650 unused; ref 1308 | post-SF16 | part of retune |
| **Reduction: all_node scaling** (RED_ALLNODE_*) | **DEFERRED** | consts 389–393 unused; ref 1327 | post-SF16 | part of retune |
| **Sibling `depth -= 2` on quiet alpha-raise** | **DEFERRED** | note 3169–3172; consts SIBLING_DEPTH_* 634/636 unused; ref 1480 | SF16 | part of retune |
| **history-buys-lmr_depth** (`+2·main`, `lmrDepth += hist/LMR_DIVISOR`) | **DEFERRED** | note 2598; no LMR_DIVISOR const; ref 1191/1194 | post-SF16 | 0–5 |
| Continuation-history pruning (`hist < −4313·d`) | LIVE | N-loop; ref 1188 | SF16 | — |
| Capture futility + capture/check SEE pruning | LIVE | move-loop; helpers 219+ | SF16 | — |
| Parent futility + quiet SEE (`−25·lmrDepth²`) | LIVE | move-loop; ref 1196/1213 | SF16 | — |
| LMP / movecount pruning + skip_quiet_moves | LIVE | ref 1152 mirror | SF16 | — |
| Fail-high softening (main + qsearch) | LIVE | 3194+; qs 3835–3838 | SF16 | — |
| Equal-value dither / dithered draws | LIVE | 3135–3140 | SF16 | — |
| Fail-low parent bonus + capture bonus + ttPv backprop | LIVE | node-result block | SF16 | — |
| TT store (Lower/Exact/Upper, depth+6 no-moves, raw eval) | LIVE | 3328–3346 | SF16 | — |
| **qsearch: NEVER stores EXACT** | LIVE | 3843–3851 (Lower/Upper only) | SF16 | — |
| staticEval init to VALUE_NONE | LIVE | stack.rs:101,128 | SF16 | — |
| queen-promo counts as capture-stage (stats/ttCapture) | LIVE | movegen.rs:640–650 | SF16 | — |
| stat_bonus cap | LIVE | STAT_BONUS_* params 919+ | SF16 | — |
| **TM: fallingEval + stability/time_reduction + bestMoveInstability + nodesEffort** | **LIVE (all 4)** | soft_stop 4102–4116 | SF16 | — |
| **Optimism** (stm bias inside eval scale) | **ABSENT** | eval takes param but search passes `VALUE_ZERO` at every call (1898/1975/3578/3630); hook noted 1690; ref search.cpp:379 | SF16 (PR#3797) | +5–15 |
| **N1 upcoming-repetition / cuckoo early draw** | **DEFERRED** | not built (only comment 174); needs cuckoo tables | SF16 | — |
| Global SPSA retune of live margins vs new eval | N/A (retune) | 06 §6b item 1 | both eras | +15–40 |

## 2. Ranked shortlist — code gaps implementable without new data structures

Ranked by expected Elo given "gainer season" (eval moved +265 since these were
tuned/shelved). Items flagged **[retune-dominated]** deliver most of their value
through S2 SPSA of *existing* params, not new code.

1. **Un-defer the §3.7/§3.9 reduction+depth bundle** — consume the already-defined,
   currently-dead consts: cutoffCnt surcharge (RED_CUTOFFCNT_*), ttPv entry-bump
   (RED_TTPV_ENTRY) + rebate (RED_TTPV_*), tt_capture (RED_TTCAPTURE), all_node
   scaling (RED_ALLNODE_*), and sibling `depth -= 2`. Biggest shelved subsystem;
   the deferral reason was explicitly "PSQT eval over-reduces mate lines" — the
   NNUE-grade eval is exactly the precondition that lifts the block. **[partly
   retune-dominated]** — the constants are SF-tuned against NNUE and should
   transfer, but re-verify the mate suite and co-tune in S2.
2. **IIR narrow retest (N12)** — `depth>=6 && !ttMove && !followPV && !allNode →
   depth--`. Branch `wip/p9-item9-iir-followpv` exists; the reverted form was
   follow-PV-shaped and predates the eval jump. Cheapest possible; highest
   evidence (perennial SF gainer). No new data structures.
3. **history-buys-lmr_depth** — add the 16-entry `LMR_DIVISOR` **const table** and
   the `+2·main_history` term, then `lmr_depth += history / LMR_DIVISOR[idx]`.
   Distinct tree-*relaxing* heuristic (good-history quiets earn effective depth,
   sparing them from parent-futility/quiet-SEE). Const array, not a runtime
   structure.
4. **Optimism** — compute `optimism = 137·avg/(|avg|+81)` at aspiration setup and
   feed the (already-present) eval `optimism` parameter instead of `VALUE_ZERO`.
   Highest single-item external Elo (+~15) but couples with our hot eval-scale
   amplitude — needs the amplitude story before it will transfer.
5. **[retune, S2]** Global SPSA of the LIVE margins (RFP/futility/razoring/NMP/
   LMR bases/singular betas) vs the new eval. 06 §6b #1, biggest lever, but pure
   retune — belongs to S2, not S0 code.
6. **TT exact-bound cutoff aggressiveness** — minor post-SF16 refinement; audit
   whether our N6b under-cuts on EXACT entries. Low expected Elo; verify-only.
7. **[excluded — needs new data structure]** N1 upcoming-repetition/cuckoo early
   draw requires building the cuckoo tables (board layer). Real gap, but out of
   the "no new data structures" scope; sequence with a board-doc task.

## 3. Top-3 implementation sketches

### (1) Un-defer the reduction + sibling-depth bundle

**Hook points.** Reduction finalization at `mod.rs:2861–2878` (after `do_move`,
before the LMR scout) and the move-loop entry `r` (§3.5, add the ttPv bump where
`r` is first formed). Sibling depth-2 at `mod.rs:3169–3172` (the quiet
alpha-raise arm, currently a comment). All consts already live in `params.rs`
(374–393, 634–650) — no new params.

**Insertion order matters (SF parity):** in `search.cpp` the ttMove reduction
`r = max(0, r-2016)` is an **`else if`** after the cutoffCnt branch (search.cpp:
1308–1313). ARCHON currently applies it as a plain `if` (2870–2873) because
cutoffCnt is deferred. When cutoffCnt lands, the ttMove rebate MUST become
`else if` or both fire — this is the top correctness risk. Add: ttPv entry bump
`if ss.tt_pv { r += RED_TTPV_ENTRY }`, ttPv rebate block, `if tt_capture { r +=
RED_TTCAPTURE }`, the cutoffCnt/all_node blocks, and at 3172 the two-line
`if depth>2 && depth<13 && !is_decisive(value) { depth -= 2 }`.

**Params:** none new (all defined-but-unconsumed). SPSA will retune them in S2.

**SPRT plan.** This is a large tree-shaping change and WILL shift bench (the
deferral is what inflated depth-6 bench 350190→548978). Gate: (a) run the §5
mate + zugzwang suites at pinned depth FIRST — the whole reason for deferral was
mate-line over-reduction, so this is the go/no-go, not the SPRT; (b) fixed-node
proxy SPRT vs current incumbent to confirm no bug; (c) equal-time SPRT `[0,5]` at
10+0.1 with the unbalanced UHO book. Land in 2–3 sub-commits (cutoffCnt+all_node;
ttPv bump+rebate; tt_capture+sibling-depth) so a mate-suite regression bisects
cleanly. Risk: the constants were co-tuned as a set against NNUE eval + NNUE move
ordering; our move ordering is weaker, so expect to need the S2 co-tune before
this is net-positive at equal time — treat a small regression as "needs SPSA,"
not "revert."

### (2) IIR narrow retest (N12)

**Hook point.** Between N11b (`improving |= eval>=beta`, ~2007) and N13 ProbCut
(2372). Insert the exact SF-current form: `if !ss.follow_pv && !all_node &&
depth >= IIR_MIN_DEPTH && tt_data.mv == Move::NULL { depth -= 1; }` where
`all_node = !NT::PV && !cut_node`. NB the reference gate is **depth≥6** (search.cpp:
1030) — the 06 study's "depth≥9" was an older SF form; use 6.

**Params:** one new const `IIR_MIN_DEPTH = 6` in `params.rs`. `follow_pv` field
already exists (currently inert). No data structures.

**SPRT plan.** Cheap: cherry-pick `wip/p9-item9-iir-followpv` but replace its
follow-PV-shaped condition with the narrow cutnode/no-ttMove form above. Bench
will change. Gate: fixed-node proxy SPRT first (bug tripwire), then equal-time
`[0,3]` vs incumbent, unbalanced book. Risk is low (self-contained, well-guarded);
the only historical failure was the wrong *shape* pre-eval-jump. Keep the mate
suite in the gate (IIR touches depth on no-ttMove nodes).

### (3) Optimism

**Hook point.** Computation belongs at aspiration-window setup in the root driver
(where `average_score`/`avg` is already tracked for the window centre, doc-comment
1690). Store `self.optimism[us]` / `self.optimism[!us] = -optimism[us]` (the
`optimism: [Value; 2]` field is already specced, doc 10 §3.2), then thread it to
the four `evaluate(&self.board, &mut self.eval_ctx, OPTIMISM, ply)` call sites
(1898, 1975, 3578, 3630) in place of `VALUE_ZERO`.

**Params:** `OPTIMISM_NUM = 137`, `OPTIMISM_DEN_BASE = 81` (SF-current). No data
structures — the eval already accepts and applies the `optimism` scalar (eval/mod.rs
`evaluate(..., optimism, ...)`).

**SPRT plan.** Equal-time `[0,5]` at 10+0.1, unbalanced book. **Primary risk:**
optimism is applied *inside* eval scaling and our eval-scale amplitude differs
from SF's (our NNUE output distribution moved +265; the 137/81 shape was tuned to
SF's amplitude). Verify the applied-optimism magnitude in centipawns is sane on a
few positions before the match, and be ready to scale 137 to our amplitude — this
is the "needs the amplitude story" caveat from 06 §6b #6. Keep displayed evals
calibrated (optimism must not leak into `uci_score`).

## 4. Surprises from the inventory pass

- **TM is already complete.** All four soft-stop factors — fallingEval,
  stability/time_reduction, bestMoveInstability, and **nodesEffort** — are LIVE
  (soft_stop 4102–4116, with golden tests). 06 §6b item 4 lists "TM completeness
  (nodesEffort)" as if open; it is not. Remove it from the port list.
- **Correction history is fully wired** (S1, `a8e16c5`): corrected eval is read in
  both the main search (1985) and qsearch stand-pat (3639), and written back
  (3375). 06 §6a item 1 (the #2-ranked open item) is closed.
- **The single biggest remaining CODE lever is un-deferral, not a new port.** The
  richest shelved material — cutoffCnt/ttPv/all_node reductions + sibling depth-2 —
  is already implemented as dead params (`params.rs:374–393, 634–650`), shelved
  purely because PSQT eval over-reduced mate lines. Gainer season targets exactly
  these. This reframes S0: less "port SF16 mechanics we lack," more "re-enable the
  post-SF16 reductions the old eval couldn't support, then let S2 co-tune."
- **history-buys-lmr_depth IS in current master** (search.cpp:1191/1194) — I
  confirmed the reference clone still carries `history += 2·main; lmrDepth +=
  history/lmrDivisor`, so ARCHON's deferral of it is a real, still-current gap
  (not a since-removed SF idea).
- **Genuine ABSENT list is short:** only IIR (N12) and Optimism are true "SF16-era
  mechanic ARCHON lacks with no code present." Everything else open is either
  DEFERRED-with-params-ready or a pure retune.
- **One latent correctness trap:** when the cutoffCnt reduction lands, the ttMove
  rebate at 2870–2873 must move into an `else if` (SF parity, search.cpp:1312).
  Flagged in sketch (1).
