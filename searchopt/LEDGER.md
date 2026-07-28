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

## WHY WE LOSE — the holistic answer (2026-07-28, full 404-game corpus)

Full writeup: `searchopt/results/WHY-WE-LOSE-d48x.md`. Counted, not estimated.

| claim | deciding evidence |
|---|---|
| **We lose because we cannot convert, not because we blunder** | on the SAME 201 colour-paired UHO openings: we convert a won book **12%** (25W/172D/4L), SF converts **84%**; we salvage a lost book **16%**, SF **88%**. A +1 book is worth **92% to SF, 55% to us** |
| **Conversion is bimodal — tactics or nothing** | squandered wins (n=172) gain a median **+0.05** pawns over the book and **ZERO ever exceed +2.00**; converted wins (n=25) gain **+10.07** and all 25 exceed +2.00. We never grind +1.0 → +1.5 → +2.0 |
| **The advantage decays DISTRIBUTED, not at a point** | worst single move per game: median **0.25 pawns**, only **2 of 172** games contain any move costing ≥ 0.88 (the blunder bar). The single worst move carries a median **23%** of the total decline — 77% bleeds away below a quarter pawn per move |
| Game 87 audited move-by-move at 2M | largest single error in a 190 cp collapse is **32 cp**; at two moves our played move BEAT the free-search best. No locatable defect |
| Draw adjudication was NOT premature | in all 172, both engines independently reported ~0.00 in the last 3 moves (max +0.10); zero games where either still claimed ≥ +0.50 |
| Time management is not a loss cause (D48X) | **ZERO** losses on time or forfeit in 404 games; every loss is adjudication-on-eval or mate |
| Our eval is NOT the discriminator on these positions | at the in-game budget we match SF almost exactly on game 87's critical nodes: **+63 vs +64**, +78 vs +61, +98 vs +75 (SF cp). Both need ~8M nodes to see the draw; neither sees it at 110k |

### The two efficiency hypotheses, both KILLED by measurement (2026-07-28)

After the "no localizable defect" result, the remaining in-scope story was efficiency:
our moves are right in kind but short in depth (3/14 plateau positions flip to the 8M
oracle's move with more nodes; **0/14 never converge**, so it is not eval disagreement).
Two ways that could be true, both tested:

| hypothesis | measurement | verdict |
|---|---|---|
| We reach less DEPTH per node (the D47V "wide shallow" claim) | 70 plateau positions at 110k: ARCHON median depth **15.0**, SF18 **15.0**; mean 14.73 vs 14.47. We are shallower in 21/70, EQUAL in 20, **deeper in 29** | **REFUTED.** No depth-per-node deficit. The n=44 D47V tree-shape claim does not reproduce |
| Our SEARCH is slow (nps) | movetime 1000, equal contention: ARCHON+net **0.54 Mnps**, SF18 **1.03 Mnps** (1.92x) — but ARCHON on PSQT is **2.13 Mnps**, i.e. **2x SF18's total throughput**. **75% of our node time is eval** | **NOT A SEARCH DEFECT.** The search machinery is faster than SF's; the entire speed gap is net evaluation cost — the deliberate 1280-width tradeoff (out of scope) |

**Caveat on that 1.92x (owner, 2026-07-28): SF18's nps is INFLATED by its dual-net
design and the ratio overstates its per-node compute.** Verified in the checkout
(`src/evaluate.cpp:49`, Stockfish 18 / `cb3d4ee9`):

```cpp
bool Eval::use_smallnet(const Position& pos) { return std::abs(simple_eval(pos)) > 962; }
auto [psqt, positional] = smallNet ? networks.small.evaluate(...) : networks.big.evaluate(...);
if (smallNet && (std::abs(nnue) < 277))   // redo with the big net when it is close
```

SF runs a CHEAP net on every materially-lopsided position and pays for the big net only
near equality; we run the full 1280-wide net on every node. They are still genuinely
faster in wall-clock — adaptive eval cost is a real optimisation, not an accounting
artifact — but the like-for-like compute gap is smaller than 1.92x.

This makes the conclusion STRONGER, not weaker: if part of SF's nps is cheap evals, our
search at 2.13 Mnps compares even better against their search, and the bottleneck is even
more clearly our eval cost.

**Idea for whoever owns the net (NOT searchopt work, recorded so it is not lost):**
adaptive eval cost is a *conditioning axis* — precisely the shape A3 says is the only
thing that can still move anything, since every global level is already at a local
optimum. Era caveat: thresholds 962/277 are from the `cb3d4ee9` checkout, not our
constants reference `eca43a97`; the SHAPE is what carries over, not the values.

⇒ At equal nodes we reach equal depth and pick the deep oracle's move ~79% of the time.
The search machinery is not slow, not shallow, and not making locatable errors. **Every
measured deficit is in eval cost or eval quality**, both out of scope for this folder.

⇒ **Consequence for this folder.** The per-position "find the node that discards the right
move" program has essentially no material on D48X: 1 actionable blunder position in 404
games, and it resisted mechanism attribution. The Elo is in ~5 cp/move of distributed
drift across 172 unconverted wins. That is the failure mode the standing bar in CLAUDE.md
explicitly deprioritised as "leaves no trace to debug" — on this corpus it is the whole
story. **This is a finding for the owner to rule on, not for an agent to act on unilaterally.**

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
| **Root node ALLOCATION is a sound per-position statistic** | it is not. On OURS index 0 the `b7b8/a1b1` node ratio swings **8.1× → 0.1×** across a 6-arm inert ±1 panel (RED_BASE/RED_FLAT/RED_DELTA_MULT), and one inert null makes the engine play the RIGHT move. Same failure class as "nodes to stop playing the blunder" |
| **`searchmoves`-restricted head-to-head localises the defect** | it does not, at least not at one budget. Restricting the root to the two candidates flips the pick to the correct move — but that head-to-head argmax itself flips **2/6** under inert ±1, and restricting from 16 moves to 2 hands each ~8× the budget, which is a sufficient alternative explanation. This was P2's mechanism test; it does not survive its own control on a fresh position |

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

- **All 3 D48X OURS positions are WALKED (2026-07-28). Two are corpus false positives.**
  Verified at 4M nodes forced, both engines (`searchopt/results/*-d48x-ours*.md`):

  | idx | position | verdict |
  |---|---|---|
  | 0 | `8/1r3kp1/R6p/1PR4P/6P1/8/p4P1K/r7 b - - 36 65` | **REAL.** `a1b1` +0 (SF) / −13 (ours) vs `b7b8` −262/−264 — a 250 cp error, oracle confirmed by both engines, SF's free search plays `a1b1` at 4M. **No mechanism found: argmax, head-to-head and node-allocation instruments all died to their own inert nulls.** |
  | 1 | `r4r2/6pk/b3Qq1p/pp1B4/5PP1/7P/PP6/4R1K1 w - - 0 29` | **ORACLE ARTIFACT.** SF18 plays the "blunder" itself at 250k/500k/1M; its own forced score for the "best" move collapses −15 → −216 |
  | 2 | `5k2/pp3pb1/1q4p1/1B1p1bN1/1n1P3Q/1N2r3/PPR3P1/1K5R w - - 0 24` | **ORACLE ARTIFACT.** At 4M both engines rank the played move ABOVE the oracle's best (−519 vs −553 ours; −385 vs −409 SF) |

  ⇒ **The actionable D48X corpus is ONE position**, and that one resisted mechanism
  attribution. Next material must come from NEITHER (21) or a re-mined corpus.

- **The corpus needs an ORACLE-VERIFICATION GATE.** `mine_losses.py` scores at 120k and
  `classify_losses.py` classifies at 110k; both sit where SF's own preference is still
  unsettled, so a row can enter OURS purely because the oracle had not looked far enough.
  2 of 3 did. Any candidate's `best` must be re-checked forced at a budget well above the
  mining budget (4M settled all three here) before a walk is spent on it. One forced
  ladder per candidate — cheap next to an hour of tracing.

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
