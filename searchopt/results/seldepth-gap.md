# SF's PV lines run 2.5 plies deeper than ours at equal nodes AND equal depth

2026-07-28, second session. **The first cross-engine statistic in this folder
that distinguishes and survives its own null.** Every previous one came back at
par (depth, EBF, static-eval MAE, within-engine self-consistency, cold/warm) or
died to an inert perturbation (argmax, head-to-head, node allocation).

## The measurement

150 plateau positions, `go nodes 110000`, `seldepth` and `depth` read straight
off the UCI info lines — same units, no frame to convert. Both engines use the
**identical** accounting: `if PvNode { selDepth = max(selDepth, ply+1) }` in
BOTH `search()` and `qsearch()`, reset per PV line (SF `search.cpp:762/1634`;
ours `src/search/mod.rs:2295/4366`). Verified before anything was built on it.

| arm | depth | seldepth | sel/nom | Δseldepth vs base |
|---|---|---|---|---|
| ARCHON base | 14.59 | 22.72 | 1.560 | — |
| ARCHON inert `RED_TABLE_NUM +1` | 14.47 | 22.64 | 1.570 | −0.08 ± 0.70 |
| ARCHON inert `RED_TABLE_NUM −1` | 14.49 | 22.64 | 1.563 | −0.08 ± 0.78 |
| **SF18** | 14.80 | **25.22** | **1.709** | **+2.50 ± 0.78** |

Same node budget, same nominal depth, **2.5 plies more selective depth**.

A six-arm inert null panel on the same statistic at fixed depth 12
(`RED_TABLE_NUM ±1`, `RFP_CORR_DIV −1`, `QHIST_PRUNE_MULT +1`,
`NMP_EVAL_BASE +1`, `SE_TRIPLE_DEEP +1`) spans **−0.03 .. +0.14**. The band is
tight; the gap is ~17× it.

## Where the plies come from — it is NOT qsearch

Mean `seldepth − depth` at fixed nominal depth, n=120:

| nominal depth | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| ARCHON | 3.45 | 1.79 | 2.73 | 3.03 | 3.90 | 5.02 | 6.31 |
| SF18 | 3.10 | 1.62 | 2.31 | 2.94 | 4.26 | 6.07 | **8.82** |
| gap | −0.35 | −0.17 | −0.42 | −0.09 | +0.36 | +1.05 | **+2.51** |

**At shallow depth OUR dive is deeper.** The gap is zero at depth 6 and then
compounds. That is an *extension* signature, and it switches on exactly at the
singular-extension gate (`depth >= 6 + ttPv`), not at any qsearch boundary.

## The frame hypothesis — mechanism-tested, and only ~26% true

Our internal eval frame is 0.538× SF's, and several extension-governing
constants are eval-frame quantities transcribed raw. In pawn terms ours are
~1.86× larger, and for all of them that direction means *fewer / shallower*
extensions:

* `singular_beta = tt.value − (SE_BETA_MULT + …)·depth / SE_BETA_DIV` — a larger
  margin puts `singular_beta` further below `tt.value`, so fewer moves are
  singular.
* `doDeeperSearch: value > bestValue + LMR_DEEPER_MARGIN` (52) fires less;
  `doShallowerSearch: value < bestValue + LMR_SHALLOWER_MARGIN` (9) fires more.
* `r -= |correction_value| / RED_CORR_DIV` — our correction values are learned
  in our own (smaller) eval units, so the reduction discount is under-applied.

Mechanism test at fixed depth 12, n=120, against the ±0.15 null band:

| arm | sel−nom | vs base |
|---|---|---|
| base | 6.31 | — |
| SE trigger frame-scaled (`SE_BETA_MULT` 60→32, `SE_BETA_TTPV_MULT` 70→38) | 6.91 | **+0.60** |
| `LMR_DEEPER_MARGIN` 52→28, `LMR_SHALLOWER_MARGIN` 9→5 | 6.61 | **+0.30** |
| `RED_CORR_DIV` 26131→14058 | 6.58 | **+0.27** |
| `RED_CORR_DIV` →7029 | 6.83 | +0.52 |
| **SE double/triple margins frame-scaled** | 6.42 | **+0.11 — inside the null band** |
| all three together | 6.10 | **−0.21** |

Three things, stated in the order that matters:

1. **The double/triple-margin half of the story is REFUTED.** It moves nothing.
   Only the singular *trigger* margin is live.
2. **The live parts recover ~0.6 of 2.51 plies — about 26%.** The remaining
   ~1.9 plies is unexplained and is NOT in: qsearch (we dive deeper shallow),
   IIR (exact SF parity, `depth >= 6`, `depth--`), the LMR depth clamp
   (`LMR_PV_DEPTH_BONUS` = SF's `+PvNode`, `RED_MAX_DEEPER` = SF's
   `newDepth+2`), the ttMove min-depth clamp, or the SE block's structure
   (including `depth += SE_NODE_DEEPEN`, SF's `depth++`).
3. **The three combined go NEGATIVE (−0.21).** Individually positive, jointly
   below base. Extensions and reductions interact and do not compose; this is
   the same ply-boundary chaos the ledger's OPEN list already records. Do not
   assume additivity when stacking these.

## Why this shape is worth chasing at all

Every tree-**widening** change with game evidence loses: `lmr30` −11.6 (n=628),
`EVAL_SCALE 2200` (narrower) −27.8, `corrgate` −17.1 (n=406). The width optimum
is pinned in both directions. An extension change is not a widening — it is a
**reallocation** of the same node budget toward critical lines, which is the
only shape left that can win.

## Games

`engines/runs/setrigger` — SE trigger margin frame-scaled, self-play, 30k
nodes, corpus openings. Arm-verified live before launch.
