# Ledger — established, refuted, open

One line per claim, with the number that decides it. **If a claim is not here, it is not
known.** Add to this file rather than to prose elsewhere.

---

## ESTABLISHED

| claim | deciding evidence |
|---|---|
| The search discards the right move too early; the move is valued correctly once forced | project owner's standing statement; everything measured on 2026-07-28 reduces to it |
| Eval is at par with SF or better | our static eval vs SF deep scores, slope **+0.803** vs SF's own static **+0.722**, n=2912 |
| Search constants are faithful to `eca43a97` | line-by-line: reduction table, reduction formula, statScore, 445/4096 coupling, 7183 clamp, bonus/malus, quiet ordering key |
| Global pruning/reduction constants are at a local optimum | no global knob beats an inert ±1 at equal nodes (RED_TABLE_NUM/BASE/CUTNODE/OFFSET, RFP, razor, cfut, pfut, qsfut, quiet weights) |
| Only ~46% of our losses are our defect | at 110k: SF blunders on **11/44** of our own loss positions; nulls 19/20/21 |
| Loss split (110k budget, null-bounded) | BOTH 14% · near-tie-for-both ~41% · warm-context ~14% · **OURS 32%** |
| On OURS we are **confidently wrong**, not uncertain | top1−top2 separation **55.3 cp (us) vs 10.5 cp (SF)**; near-tie rate 29% vs 64% |
| We need 2.3–18× SF's nodes on OURS | SF fixes all 14 at ≤110k; we need 250k×7, 500k×4, 2M×1, never×2 |
| Time management is not the cause | in-game depth == cold-110k depth, median gap **0**, every class |
| The warm **TT helps**; warm history/aspiration hurts | of 24 the cold search avoids, warm replay reproduces **4/24**, TT-cleared **6/24** |
| Tree shape: wide shallow, narrow deep | ours/SF width **1.29–1.45×** at d6–10, **0.90–0.92×** at d19–20, crossover ~d18 |
| P2's mechanism: a perpetual sets alpha and starves the win | `searchmoves={Qh3,Rxh7+}` → 52,407 nodes, no mate; `searchmoves={Qh3}` → mate at d18, 260,666 nodes |
| We have no contempt parameter (until now) | `value_draw` is `VALUE_DRAW − 1 + (nodes & 2)` — a ±1 dither about 0 |
| `RED_TABLE_OFFSET` (the reduction *tilt* knob) has never been moved from 0 | only the level `RED_TABLE_NUM` has ever been tuned |
| **A blunder = wp_loss ≥ 0.06 (~88 cp) from a position better than −2.5 pawns** | owner ruling: at worse than −2.5 vs SF18-class opposition the game was already gone, so the move is not why we lost. One-sided — throwing away a WON position still counts |
| Blunders are RARE in still-playable positions | **0.16%** of our moves at wp ≥ 0.10 (6 of 3,698 scored). At the 0.06 bar it is ~1.3% |
| 56% of our moves in lost games come from still-playable positions | 3,698 of 6,600 scored, `best_cp ≥ −250` |

## REFUTED — do not rebuild

| claim | what killed it |
|---|---|
| `RED_GIVES_CHECK` is a validated arm | **`RED_GIVES_CHECK = −512` (reduce checks MORE) clears the same null band.** The metric rewarded noise |
| "nodes to stop playing the blunder" is a sound metric | same control; *any* perturbation that shakes the search off the losing move scores well |
| Futility gates (RFP/pfut/cfut) block P1's mate | fixed-depth artifact; at **equal nodes** base and every arm mate at 100k |
| "Base never finds P1's mate" | my probe capped at **d12**; base mates at **d14 / 62k nodes** |
| P1's root rests on a "2 cp margin" | −282 was a fail-soft zero-window **bound**; bound-free it is **−682** at d8 |
| "Checking moves are never extended" | depth-9 artifact; SE applies only to the ttMove and fires on **8/33** eligible |
| "80.8% of TT hits carry no move" | misread `tt_move` (= "this row's move IS the tt move"); real figure **70%** at depth ≥6 |
| "Our `eval` prints the wrong perspective" | engine is correct and labels dynamically; my parser hardcoded it and double-flipped |
| "Quiet endgame losses are an EVAL failure" | deep search of both fixed leaves ranks them the **same** in both engines (d10/14/22/26) |
| "Our losses are disproportionately forcing" | 80% vs a **55% base rate**, p=0.081 |
| "P1 is not our defect" | read the 50k datapoint; at the in-game 110k SF is sound and we blunder — it **is** ours |
| LMR −30% helps | **−11.6 Elo** over 628 games; null at equal nodes |
| `EVAL_SCALE` tests the frame | it is a global pruning-aggressiveness knob; −27.8 ± 16.8 over 188 games |
| "`cp_loss ≥ 0` by construction" (my miner's docstring) | **11% of rows were negative**, median −12, worst −218. A FREE search shares its budget across all moves; a FORCED search spends all of it on one, so the played move can outscore the free search's own best. Both are now forced at the same budget |
| A wp threshold alone finds blunders | mate scores saturate the logistic, so **12 of 22 raw candidates were positions where SF merely saw a mate**. `win_prob` now clamps its input |

## MEASURED-NEUTRAL (game results)

| arm | result |
|---|---|
| `RFP_THREAT_MARGIN` 55→0 (an ARCHON original, never individually tested before) | **−3.5 ± 14.1**, n=1185 |
| LMR −30% (`RED_TABLE_NUM` 1984) | **−11.6 Elo**, n=628 |

## D48X TRANSITION (2026-07-28)

The canonical net is now **D48X** (`engines/canonical/`), and **the corpus below was
mined from D47V games** — it is stale until re-mined. Everything in ESTABLISHED that is a
statement about *our engine's behaviour on specific positions* must be re-measured; the
methodological entries (what instruments are admissible) carry over unchanged.

Verified working under D48X: `searchdbg.py` (traces, 5,938 rows on a sample position),
`uciprobe.NET` (resolves to `engines/canonical/net.cenn`), `mine_losses.py`,
`classify_losses.py`. `engines/bin/archon-trace` was rebuilt for `arch-d48x` — the old
D47V build failed with `acc_width 1280 != 768`, which is what an arch mismatch looks like.

A fresh corpus match (canonical vs SF18, same protocol as the D47V one so the two are
comparable) is running: `engines/runs/corpus-d48x/`.

## OPEN

- **Are we even losing to blunders?** Only **0.16%** of our moves in still-playable
  positions are blunders at wp ≥ 0.10 (and ~1.3% at 0.06). Against a −132 Elo opponent
  that points at **accumulated positional pressure** rather than discrete errors — a
  different failure mode than this folder is built to chase. Confirm on the full corpus
  before committing more effort to per-position walking. *(One partial-corpus number; do
  not promote it without the full run.)*

- **Which node discards the right move, on each of the 14 OURS positions.** Only P1 and
  P2 have been walked, and P1's walk produced a mechanism that was later refuted. This is
  the work.
- Why RGC-style arms are non-monotone (512 misses what 256 and 1024 find). Unexplained,
  and a hint the search is chaotic at ply-boundary granularity.
- `DRAW_CONTEMPT` exists (default 0) with a documented mechanism but **no admissible
  evidence**; its root effect died 0/6 under inert perturbation.
- D48X vs V: first game evidence on a training generation, running.
