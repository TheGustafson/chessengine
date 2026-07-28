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
| Only ~46% of our losses are our defect | at 110k: SF blunders on **11/44** of our own loss positions; nulls 19/20/21 (D47V) |
| Loss split (110k budget, null-bounded) | D47V: BOTH 14% · near-tie-for-both ~41% · warm-context ~14% · **OURS 32%**. **D48X re-derivation below supersedes the shares** |
| **D48X corpus (2026-07-28, full 404-game run)** | 5,379 still-playable moves scored over 173 lost games. Blunders (wp_loss ≥ 0.06): **25 = 0.46%**. Split **OURS 3 · BOTH 1 · NEITHER 21**. Provenance `d48x-1280-sb520-ema.cenn`, oracle 120k |
| **We are not losing to catastrophic discrete errors** | wp_loss ≥ 0.10: **6 of 5,379 (0.11%)**. wp_loss ≥ 0.20: **ZERO in the whole corpus.** Against a −132 Elo opponent. This answers the standing OPEN question on full data |
| The cold search already avoids most of our own blunders | **21 of 25** blunders are NEITHER — our cold 110k search does not play them. Points at in-game context (warm history/aspiration) or argmax instability, not a cold-search pruning gate. **PROVISIONAL: the classifier is one free-search argmax per engine at one budget, and has not been nulled** |
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
| **"top1−top2 is 55.3 cp us vs 10.5 SF" (the magnitude)** | **FRAME-CONTAMINATED, 2026-07-28.** `classify_losses.py` computed each gap inside one engine — which is why it looked safe — then PRINTED them side by side, and README/LEDGER read the ratio straight off that table. ARCHON prints raw internal values, SF divides by `a ≈ 385`, so ours is ~2.07× theirs. Real ratio is **2.54×, not 5.27×** (26.7 SF-frame cp vs 10.5). The *direction* survives; the magnitude was doubled. R2 violated by the tool itself — now converted at the boundary (`our_gap` is SF-frame, `our_gap_raw` retained) |
| "We are confidently wrong on OURS" (as a D48X fact) | not established on this corpus: frame-corrected, OURS is **70.0 vs 58.0 SF-frame cp — a 1.21× ratio at n=3**. NEITHER is **0.79×** (we are *less* separated than SF). Too few positions to conclude anything; the D47V 2.54× is the only version of this claim with real support |

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
comparable) ran as `engines/runs/corpus-d48x/`; the mine + classify **completed
2026-07-28** and its numbers are in ESTABLISHED above.

## OPEN

- **ANSWERED on the full corpus: we are not losing to blunders.** 25 blunders in 5,379
  still-playable moves (0.46%); 6 at wp ≥ 0.10 (0.11%); **zero** at wp ≥ 0.20. Only **3**
  are our cold search's defect. The partial-run figure (0.16% at wp ≥ 0.10) held up. This
  is real evidence that the −132 Elo is **accumulated positional pressure**, not discrete
  errors — a different failure mode than this folder is built to chase. Recorded, not
  acted on: the owner's standing direction is to work OURS first, then widen to the other
  losing positions.

- **Which node discards the right move, on each of the 3 D48X OURS positions.** This is
  the work. (The D47V-era 14 are stale — different net.) Then widen to NEITHER/BOTH.
  Index 0 `8/1r3kp1/R6p/1PR4P/6P1/8/p4P1K/r7 b - - 36 65`, played `b7b8`, best `a1b1`,
  wp_loss 0.068.

- **The OURS/NEITHER classifier has never been nulled.** The whole 3-vs-21 split rests on
  one free-search argmax per engine at one budget, and `walk.py`'s own docstring records
  that an inert ±1 flips ~25% of these positions. Until nulled, treat the split as
  provisional — `walk.py`'s 5-budget ladder is the stronger instrument and re-adjudicates
  each position as it is walked.
- Why RGC-style arms are non-monotone (512 misses what 256 and 1024 find). Unexplained,
  and a hint the search is chaotic at ply-boundary granularity.
- `DRAW_CONTEMPT` exists (default 0) with a documented mechanism but **no admissible
  evidence**; its root effect died 0/6 under inert perturbation.
- D48X vs V: first game evidence on a training generation, running.
