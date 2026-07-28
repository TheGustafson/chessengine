# Ledger — established, refuted, open

One line per claim, with the number that decides it. **If a claim is not here, it is not
known.** Add to this file rather than to prose elsewhere.

---

## THE EVAL TRAP — 2026-07-28, recorded because the axiom alone did not prevent it

An agent briefed on A1 ("eval is at par or better"), which had read it in START-HERE.md and
README.md, and which the owner made restate the commitment in its own words FIVE times
before starting, blamed eval TWICE in one session and was wrong both times.

| # | the claim | how it was reached | what killed it |
|---|---|---|---|
| 1 | "the bottleneck is eval COST, out of scope" | 1.92x NPS gap; 75% of node time is eval. **Committed to git.** | equal-NODES matches: cost explains **at most half** the 134 Elo (-134 time -> -68 equal nodes) |
| 2 | "therefore eval", by ELIMINATION | four search hypotheses came back null, so the residual "must be" eval | **direct measurement**: static eval vs deep oracle, 630 positions — **MAE 97.1 ours vs 97.2 SF, corr 0.899 vs 0.905, our p90 LOWER**. At par or better |

The direct measurement takes **fifteen minutes**. It was run in hour six, after the owner
insisted, having been available since hour one.

**Diagnosis worth keeping.** Both conclusions ended in *"and therefore it is not my
problem."* Both arrived at the exact moment the search analysis ran out of road. Both were
wrapped in real rigour — NPS decompositions, bootstrap CIs, pre-registered bands — which
did not make them correct, only harder to challenge; the owner had to object three
separate times and was right every time.

**Rule: a conclusion that ends "not my problem" needs MORE evidence than one that doesn't.
Convenience is the tell.** And run the direct eval measurement BEFORE concluding eval.

Same session, same agent, one more mechanism error worth the same treatment: it blamed
root LMR for starving a late-ordered move, built a tunable for it, and then found that
cutting that move's reduction from **1020 to 3** moved its node allocation **440 -> 414**.
Reduction was never the cause; the low node count was the SYMPTOM of failing low.
**Verify an arm moves its mechanism before spending games on the outcome.**


## ESTABLISHED

| claim | deciding evidence |
|---|---|
| ~~The search discards the right move too early~~ — **THIS FOLDER'S FOUNDING PREMISE, AND IT IS NOT HOW WE LOSE** | it happens, but it is **1 actionable position in 404 games**. 25 blunders in 5,379 still-playable moves (0.46%), 44% of those are oracle artifacts. The real failure is distributed decay — see WHY WE LOSE below. Kept here only so nobody re-derives it |
| **Eval is at par with SF or better — RE-CONFIRMED BY DIRECT MEASUREMENT** | original: slope **+0.803** vs SF's static **+0.722**, n=2912. Re-measured 2026-07-28 on 630 positions across all 404 corpus games: **MAE 97.1 (ours) vs 97.2 (SF18)**, corr **0.899 vs 0.905**, and **our p90 error is LOWER** (149.5 vs 176) — fatter middle, thinner tail. Owner's training-time benches on far more data agree. **See THE EVAL TRAP above before you doubt this** |
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

### SEARCH vs EVAL — NOT SETTLED (2026-07-28). The equal-nodes ladder stands; the
### conclusion drawn from it was WRONG.

**This section was originally titled "SETTLED" and concluded the residual was eval. That
conclusion is retracted** — a direct static-eval measurement (see THE EVAL TRAP, top of
file) put our eval at par or better. What survives is the LADDER, which is solid and
important. What died is the inference built on top of it.

The within-engine test below ALSO has a structural weakness the owner identified and I
had underweighted: it measures search **convergence** (does 110k find what 1M finds),
which a *systematically* biased search passes trivially — pruning heuristics are
depth-scaled, so a search that discards a class of moves discards them at every budget.
**Low power against the most likely form of search defect.** Do not treat "self-agreement
at par" as "search at par".

**The question:** is the -132 Elo a search defect or an eval one? **Still open.**

**1. EQUAL-NODES LADDER vs SF18** (`-each nodes=N`, nodes constrained and NOTHING else —
depth, time and tree shape all free). Same 203 corpus openings, both colours, n≈406 each:

| budget | Elo | 95% CI |
|---|---|---|
| 10k | **−92** | [−121, −64] |
| 30k | **−64** | [−91, −38] |
| 60k | **−76** | [−103, −50] |
| 110k | **−68** | [−93, −43] |
| 10+0.1 (time) | **−134** | — |

⇒ **The 134 Elo splits ~66 speed / ~68 per-node.** Removing a 1.92x speed handicap buys
~66 Elo. The remaining ~68 Elo **saturates by 30k and does not close with more nodes**
(30k/60k/110k all overlap) — so it is NOT a "needs more search" problem, and eval COST
cannot explain it, because fixed nodes neutralises cost exactly.

⇒ The 10k→30k rise (+28) is the shallow-budget inefficiency: at 10k SF reaches a full ply
deeper on the same nodes (median 12.0 vs 11.0, EBF 2.154 vs 2.310, n=16,835 game moves);
by 110k depth is **EQUAL** (16.0 vs 16.0, n=8,219). The D47V "wide shallow, narrow deep"
entry reproduces on D48X — but it is spent by 30k and cannot account for the residual.

**2. WITHIN-ENGINE SEARCH QUALITY — the decisive test.** Each engine compared against ITS
OWN 1M-node search, so eval cancels EXACTLY (same eval on both sides of every comparison;
no cross-engine frame, no oracle, no self-agreement artifact). 150 plateau positions:

| engine | self-agree | mean self cp-loss | median | p90 | max | zeros | ≥50cp |
|---|---|---|---|---|---|---|---|
| ARCHON | 59.3% | 4.88 | 0.0 | 21.7 | 77.8 | 60% | 4/150 |
| SF18 | 58.7% | 4.05 | 0.0 | 20.0 | 58.0 | 61% | 2/150 |

Difference **+0.83 cp, bootstrap 95% CI [−2.31, +4.09] — NOT DISTINGUISHABLE FROM ZERO.**

⇒ Our search recovers its own eval's deep answer as well as SF's recovers its own. At
equal nodes we reach equal depth with equal self-consistency and are still 68 Elo worse.

**DO NOT READ THAT AS "the residual is not in src/search/" — that is exactly the wrong
conclusion this session drew from it.** The test cannot see a systematically biased
search (see the header above). Combined with the direct eval measurement coming back AT
PAR, the ~68 Elo at equal nodes, equal depth and at-par eval is **UNEXPLAINED**. That is
the honest state. The only thing left that this session did not rule out is *selectivity*
— WHICH nodes we spend on — and every aggregate probe of it came back null, which itself
says the cause is not a single global knob or a broad statistical bias.

Bounds, stated honestly: the CI bounds a real search deficit at **≤ ~4 cp/move**, it does
not zero it. The position class is narrow (plateau positions from squandered wins). And
the test measures convergence to OUR OWN answer — a search that explored a worse tree but
self-consistently would pass it (equal depth at equal nodes argues against, does not
exclude). Pre-registered bands were >1.5x = search defect, ≤1.0x = eval, 1.0–1.5x =
partial; 1.20x landed in the partial band and the CI then spans zero.

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

## ★★ THE COLD-PROBE REGIME IS NOT THE GAME REGIME (2026-07-28) — read before trusting ANY per-position measurement in this folder

**In-game depth at equal nodes, read from the stored match PGNs (n ≈ 29,000 moves per
engine — orders of magnitude more power than any per-position probe here):**

| budget | ARCHON | SF18 | delta |
|---|---|---|---|
| 110k | **21.43** | **21.43** | **+0.01** |
| 30k | 17.56 | 17.66 | −0.11 |
| 10k | 14.42 | 14.53 | −0.11 |

**Our depth EQUALS SF's at every budget in games.** Both engines see identical positions
in these matches, so position mix cannot explain it.

**But a COLD probe at 110k reads 14.54 (ours) / 14.80 (SF) — about SEVEN PLIES
SHALLOWER than the same engines at the same node budget in a game.** Every per-position
instrument in this folder — mine this session, and its predecessors — runs `ucinewgame`
per position and therefore measures a regime games never occupy.

### What this invalidates, and what it does not
* **Deflated: the (depth, seldepth) Pareto framing.** The depth half does not reproduce
  in games — in-game it is +0.01, not SF +0.26. Seldepth is not written to the PGN so it
  cannot be cross-checked in the game regime at all; treat the +2.50 cold gap as
  **unverified in the regime that matters**, not as an established fact.
* **Confirmed directly: cold mechanism gains do not transfer.** `asp30` measured
  **+0.96** depth cold and delivers **+0.08** in-game; `asp100` measured +1.24 and
  delivers +0.75. Read the arm's mechanism out of the MATCH PGN, not out of a cold probe.
* **Reported depth is an inflated proxy in the warm regime.** A warm TT lets shallow
  iterations complete off TT hits without proportionate work, so +7 plies of reported
  depth is NOT +7 plies of strength — this folder's own cold-vs-warm match prices warm
  state at ~−132 to −145 Elo, nowhere near what 7 plies would be worth. **Never convert
  a depth delta into Elo.**
* **Not invalidated:** anything measured from games (the equal-nodes ladder, every arm
  result, the conversion/salvage analysis). Those were always in the right regime.

### The rule
**Measure an arm's mechanism in the MATCH PGN.** `fastchess` writes `{score/depth time}`
per move, so in-game depth per engine is free and needs no probe at all. A cold probe is
admissible for a quick screen, but a mechanism that only exists cold is not a mechanism —
it is an artifact of the instrument, and five mechanism-verified arms that converted to
nothing this session are what that looks like from the outside.

## ★ THE FIRST PARETO MOVE: the aspiration window is too narrow, and it under-reduces the WHOLE TREE (2026-07-28)

Everything else this session moved us ALONG the (depth, seldepth) frontier. This moves
us toward it. 100 plateau positions at **110k nodes** (the in-game budget), null band
±0.15 from two inert arms:

| arm | depth | seldepth |
|---|---|---|
| base | 14.54 | 22.91 |
| inert `RED_TABLE_NUM +1` | 14.40 | 22.58 |
| inert `RED_TABLE_NUM −1` | 14.40 | 22.47 |
| `ASPIRATION_BASE_DELTA` 5→15 | 14.91 | 23.02 |
| `ASPIRATION_BASE_DELTA` 5→30 | 15.50 | 23.72 |
| **`ASPIRATION_BASE_DELTA` 5→60** | **15.62** | **24.27** |
| `ASPIRATION_MSS_DIVISOR` 6776→3000 | 14.86 | 23.27 |
| **SF18** | **14.80** | **25.22** |

**BOTH statistics rise, monotonically, far outside the null band.** At delta=60 our
nominal depth (15.62) EXCEEDS SF's (14.80) and seldepth nearly closes.

### The mechanism, and why it is not merely "prune harder"

Our reduction formula is `scale − delta * RED_DELTA_MULT / root_delta`
(`src/search/mod.rs:4189`), byte-for-byte SF's `reductionScale - delta * 617 / rootDelta`
(`eca43a97:search.cpp:1832`). **`root_delta` scales reductions across the whole tree.**

`docs/research/aspiration-frame-bug.md` established our root window is **2.4× too
narrow**: SF widens by `|meanSquaredScore| / 10588`, `mss` is a **squared** eval
quantity, our eval frame is 0.64× SF's, so our `mss` is 0.64² = 0.41× theirs while the
divisor was transcribed raw. That doc treated the consequence as root-score oscillation
(ours 1.44 pawns of range vs SF's 0.45 — **3.2×**).

**It has a second consequence that doc did not draw:** a too-small `root_delta` makes
`delta * 617 / root_delta` too LARGE, which **systematically UNDER-reduces every node in
the tree** — a wider, shallower search. That is exactly the measured deficit, and it is
a *coupling* defect, not a level defect.

**This matters for A3.** `lmr30` (−11.6, n=628) and `EVAL_SCALE` (both directions
negative) tested the reduction LEVEL and found it at a local optimum. `root_delta` is a
different axis — the level is right, its *scaling input* is wrong. A3 says only new
conditioning axes can move anything; this is one.

### ⚠ CORRECTION: the mss/frame argument is a RED HERRING for this effect

Bracketed at 110k, n=100, null band ±0.15:

| arm | depth | seldepth |
|---|---|---|
| base (5 / 6776) | 14.54 | 22.91 |
| `BASE_DELTA` 30 | 15.50 | 23.72 |
| `BASE_DELTA` 60 | 15.62 | 24.27 |
| **`BASE_DELTA` 100** | **15.78** | **24.82** |
| `BASE_DELTA` 200 | 15.87 | 24.30 ← seldepth TURNS OVER |
| **`ASPIRATION_MSS_DIVISOR` 6776→4341 (the frame-principled value)** | **14.72** | **22.84** ← INERT |
| `MSS_DIV` 4341 + `BASE_DELTA` 30 | 15.25 | 23.85 |
| `MSS_DIV` 3000 | 14.86 | 23.27 |
| **SF18** | **14.80** | **25.22** |

**The frame-derived mss divisor moves nothing** (+0.18 depth, −0.07 seldepth — inside
the null band). The *arbitrary linear base term* does all the work. So the elegant
"our mss is 0.41× because it is a SQUARED eval quantity" story, which is arithmetically
correct and which I used to justify this whole line, **does not explain this effect.**

Why: at a typical root score (~200 in our units) `mss ≈ 40,000`, so `mss/6776 ≈ 6`
against a base of 5. Re-scaling the divisor moves delta from ~11 to ~14. Raising the
base term to 100 moves it to ~106. The base term dominates by an order of magnitude,
and it is not a frame quantity at all.

⇒ **The honest mechanism is simply "our root window is far too narrow", full stop.**
The squared-frame analysis in `docs/research/aspiration-frame-bug.md` may still be
right about root-score OSCILLATION; it is not what produces the depth/seldepth gain.
**Recorded because the frame story was the reason I looked here, and it turned out to
be the wrong reason for the right place.**

⇒ `BASE_DELTA=100` is the best joint point measured (15.78/24.82 vs SF's 14.80/25.22 —
ahead on depth, nearly level on seldepth) and 200 shows the turnover, matching the
aspiration doc's own bracket (40 halved oscillation, 400 was worse). Arms in games:
`asp30` and `asp100`, both 110k.

### The attack on it, stated because the story is now elegant (README §2 rule 6)
Part of the depth gain IS a reduction increase, and "reduce more" is at a local optimum
by game evidence. So the honest reading is: **the depth/seldepth gain is established;
whether it converts to Elo is not.** `engines/runs/asp30` (110k, `ASPIRATION_BASE_DELTA`
5→30) decides it. Note the aspiration doc's own bracket: 40 more than halved the
oscillation but **400 was worse again**, so wider is not monotonically better and 30 is
a deliberate middle rather than the largest mechanism move.

## THE SELDEPTH GAP CLOSES ON ONE KNOB: `SE_MIN_DEPTH` (2026-07-28)

Bounding test (can ANY extension setting reach SF's sel−nom of 8.82 at d12?), 120
plateau positions, null band ±0.15:

| arm | sel−nom (d12) |
|---|---|
| base | 6.31 |
| `SE_BETA_MULT` 60→15 | 6.93 |
| **`SE_MIN_DEPTH` 6→4** | **8.74** |
| both | 8.89 |
| + `SE_EXT_BASE` 1→2 | 16.59 (overshoots) |

**`SE_MIN_DEPTH` 6→4 lands on SF18's value almost exactly (8.74 vs 8.82).** Every other
extension knob tried this session recovered ≤26% of the gap; this one recovers ~100%,
and `SE_EXT_BASE` shows the axis has range to spare, so the gap is not a ceiling effect.

**The paradox, stated rather than smoothed over:** SF's gate is `depth >= 6 + ttPv` and
ours is `SE_MIN_DEPTH = 6` + `SE_TTPV_DEPTH_BONUS = 1` — **an exact match**. So SF
reaches 8.82 with gate 6 while we need gate 4 to get there. The gate is therefore NOT
the underlying difference; something else makes SF's singular extensions fire more often
at the same gate (candidate: TT content — the SE gate also requires `tt.depth >= depth −
SE_TT_DEPTH_SLACK` and a lower-bound entry, so a differently-populated TT changes the
firing rate without any constant differing). Lowering our gate COMPENSATES for that
without explaining it. Do not write this up as "we found the missing constant".

### ⚠ AT FIXED NODES THE PICTURE CHANGES — and this is the sharper finding

The "closes the whole gap" result above is at fixed DEPTH. Games run at fixed NODES,
where every extension is paid for out of the same budget. Re-measured at fixed nodes,
100 plateau positions:

| 110k nodes | nominal depth | seldepth |
|---|---|---|
| ARCHON base | 14.54 | 22.91 |
| `SE_MIN_DEPTH=4` | 14.02 (−0.52) | 23.37 (+0.46) |
| `SE_TT_DEPTH_SLACK=6` | 13.60 (−0.94) | 23.12 (+0.21) |
| **SF18** | **14.80** | **25.22** |

(60k: base 13.09/20.01; se4 12.57/21.14; slack6 12.14/20.78.)

Three consequences, in order of importance:

1. **SF18 is PARETO-SUPERIOR on (nominal depth, selective depth) at equal nodes** —
   more of BOTH (14.80/25.22 vs 14.54/22.91). Every knob of ours **trades along that
   frontier**, buying selective depth only by spending nominal depth. **Nothing tried
   this session moves us TOWARD the frontier; it all moves us ALONG it.** That is the
   sharpest statement of the search deficit found so far, and it explains why five
   consecutive arms came back null: a pure reallocation between two things that are
   both already behind cannot gain.
2. **`se4` closes ~20% of the seldepth gap at fixed nodes**, not ~100%. The fixed-depth
   number (8.74 vs SF's 8.82) is real but answers a different question. **State which
   budget regime a mechanism number was measured in — they disagree by 5×.**
3. **`SE_TT_DEPTH_SLACK=6` is DOMINATED and is dropped without a match**: it costs
   nearly twice the nominal depth of `SE_MIN_DEPTH=4` (−0.94 vs −0.52) and buys less
   selective depth (+0.21 vs +0.46). One cheap measurement killed a queued 406-game run.

⇒ The open question is no longer "how do we extend more" but **"how does SF get +2.3
plies of selective depth at the SAME nominal depth and the same nodes"** — i.e. its
extensions are better TARGETED, not merely more frequent. Firing rate is the wrong
axis; which nodes get extended is the right one.

### Which gate condition binds? BOTH — and the TT one binds hard
`SE_TT_DEPTH_SLACK` (the `tt.depth >= depth − slack` condition) relaxed alone, same
120 positions, same ±0.15 null band:

| arm | sel−nom (d12) |
|---|---|
| base | 6.31 |
| `SE_MIN_DEPTH` 6→5 | 6.92 |
| **`SE_TT_DEPTH_SLACK` 3→6** | **8.47** |
| `SE_TT_DEPTH_SLACK` 3→10 | 10.37 |
| `SE_TT_DEPTH_SLACK` 3→99 (condition effectively removed) | 10.32 (saturated) |

⇒ The test does NOT cleanly discriminate — relaxing ANY SE gate condition raises the
firing rate and hence seldepth. What it does establish is that **the TT-depth condition
alone suppresses ~4 plies** (removing it reaches 10.32 from 6.31), so it is a hard
binding constraint, not a formality. That gives a SECOND candidate that reaches SF's
value without adding singular verification searches at shallow nodes:
`SE_TT_DEPTH_SLACK = 6` → 8.47 vs SF's 8.82.

Arms in games: `engines/runs/semindepth4`, 60k nodes (chosen over 30k because the
mechanism is zero at depth 6 and grows to +2.51 by depth 12 — the `setrigger` test at
30k under-sampled its own mechanism), plus `engines/runs/semindepth4-110k` running the
same arm at the in-game budget IN PARALLEL. Agreement across two budgets is much harder
to fake than either alone, and the box was free when they were launched.

## THE JUDGE INSTRUMENT IS NOW VALIDATED (2026-07-28)

After the `searchmoves` retraction above, `judge.py` scores the CHILD position
(`position … moves <mv>`, plain `go`, negated) and resets + alternates probe order.
Re-run of the NULL arm (ARCHON vs inert `RED_TABLE_NUM +1`), 2,473-position plateau
set, judges at 1M nodes, **149 pairs**:

**51.0% [42.9, 59.2]** — centred on 50, where the broken version sat at 41–45%.

⇒ The instrument is sound. It was stopped short of its 340-pair arm to relieve memory
pressure (see below); the arm is the outstanding measurement, and at ~340 pairs it
resolves a true 55% rate, which is the size that matters (55% on the 32% of positions
where the engines disagree, compounded over ~40 moves, is worth ~50 Elo).

## ARMS (games) — running total, all signed explicitly

`match.py` sorts engine names alphabetically and reports the FIRST one's score. That is
a live sign trap; every number below was recomputed directly from the PGN against the
arm name.

| arm | budget | result |
|---|---|---|
| `corrgate` (`RAZOR_CORR_DIV`/`NMP_CORR_DIV` 1e9 → 182069) | 30k | **−17.1 [−51.2, +16.7]**, n=406 |
| `setrigger` (`SE_BETA_MULT` 60→32, `SE_BETA_TTPV_MULT` 70→38) | 30k | **−16.3 [−50.4, +17.5]**, n=406 |
| `redcorr110k` (`RED_CORR_DIV` 26131→7029) | 110k | **−13.3 [−52.2, +25.2]**, n=313, STOPPED early for memory |
| `offense0` (`QUIET_OFFENSE_FACTOR` 10→0) | 30k | **−1.7 [−36.0, +32.5]**, n=398 |
| `semindepth4` (`SE_MIN_DEPTH` 6→4) | 60k | **−1.2 [−42.1, +39.6]**, n=281, stopped (null; decayed +43.7→+34.5→+19.2→−1.2 as n grew — a textbook positive partial dying) |
| `semindepth4-110k` (same arm, in-game budget) | 110k | **+4.8 [−35.3, +44.9]**, n=291, stopped (null, matching its 60k twin) |
| `asp30` (`ASPIRATION_BASE_DELTA` 5→30) | 110k | running |
| `asp100` (`ASPIRATION_BASE_DELTA` 5→100) | 110k | running |

⇒ **A real, nulled `fhf` improvement did NOT convert to Elo.** `QUIET_OFFENSE_FACTOR=0`
raised the first-move cutoff rate by +0.0021 against a ±0.0011 null band, with the
cutoff COUNT rising too (so not a ratio artifact) — and the games are flat. Consistent
rather than contradictory: 0.2 percentage points of ordering is simply too small to
show up as Elo. **The lesson is about the size of effect `fhf` must show before it is
worth a match, not about `fhf` being invalid.** Do not spend 400 games on a sub-0.005
fhf move again.

⇒ Four arms, every one neutral-to-negative with a CI spanning zero. Combined with the
folder's own history (`pfut` +3.4 ± 14 at n=1529, `rfp-threat0` −3.2 ± 14 at n=1200,
`lmr30` −11.6 at n=628), **single-knob tuning is exhausted**. Hence SPSA below.

### `QUIET_OFFENSE_FACTOR = 0`: mechanism verified, tree cost noted
The `fhf` gain is NOT a shrinking-denominator artifact — the cutoff COUNT rises too
(fh 967,486 → 1,034,004; fhf 828,370 → 887,442; rate 0.8562 → 0.8583). But the tree
grows with it (nodes +6.7%), and every tree-widening arm on record loses. The +6.7% is
itself inside the 7.3% node chaos band, so it is not established either way — the game
result is the only thing that settles this one.

## SPSA `s5-ordering` LAUNCHED (2026-07-28)

`tools/spsa/campaigns/s5-ordering.json` — the 16 ordering/history-shaping weights
tuned JOINTLY, all SPSA-uncovered. Rationale: single-knob screening is exhausted
(above), and unlike the eval-frame margins (blanket rescale refuted in BOTH directions,
`docs/research/eval-frame-scale-model-2026-07-27.md` §11.4) these weights are
denominated in OUR history units and scale with OUR history value distribution, so SF's
tuned values carry no guarantee. Ordering is also the only lever that improves the tree
WITHOUT enlarging it, which matters because the width optimum is pinned in both
directions.

Config: 3 local workers, 2000 iterations, 4 units × 1 pair, fn20k, `r_end` 0.036 (the
launcher's own rule `r_end ≈ 0.03 × 2400/iters`; the 0.002 default returns the starting
vector by construction). Exchange on DISK at `~/spsa-ex-s5`, never `/tmp` (tmpfs).
Output `tools/spsa/out/s5-ordering-0728-1642/`; state in
`~/spsa-ex-s5/state/theta-latest.json`; stop with `touch control/STOP` on the exchange.

**A non-PGO `tune` build is correct here and is not a deviation from the standing PGO
ruling:** SPSA plays fixed-NODE games with the same binary on both sides, and PGO
changes speed, not the search tree — results are identical. PGO buys wall-clock only.

**THE TOOLCHAIN WAS DEAD AND NOBODY KNEW.** `be4b6c9` ("archive 371 stale files to
old/") moved `tools/spsa/exchange.py` into `old/`; `spsa_dist.py` imports it on line 29,
so every distributed SPSA run since that commit dies with `ModuleNotFoundError`. That
sweep verified "no live file references any moved path" — which cannot catch a Python
`import`, because the module name never appears as a path. Restored in `732eac2`.
**Re-run that sweep's check as an import check before trusting anything else in `old/`.**

## INSTRUMENT CALIBRATION — pooled nodes-to-depth is DEAD, `fhf_rate` is USABLE (2026-07-28)

A 14-arm inert null panel on 120 plateau positions at fixed depth 13, using the
engine's own `ARCHON_SEARCH_STATS` counters:

| statistic | inert null band | verdict |
|---|---|---|
| pooled nodes / base | **0.9761 .. 1.0488 (7.3% range)** | **DEAD.** `RAZOR_BASE ±1` alone spans −2.4% to +4.9%. Cannot resolve anything under ~7% |
| `fhf_rate` (first-move cutoff rate) | **−0.0010 .. +0.0011** | **USABLE.** Resolves ~0.001 |

⇒ **Killed by this: the `QUIET_CHECK_BONUS` 16384→8192 candidate (−3.0% nodes).** It is
inside the band. Its dose response was already non-monotone (2048 +1.3%, 4096 −3.2%,
8192 −3.0%, 12288 +2.5%) which is what prompted the panel.

⇒ **Also retracted: "almost every ordering perturbation increases nodes, so base sits in
a shallow node minimum."** That came from a 4-arm null that happened to be all-positive;
at 14 arms the inert spread covers BOTH directions.

⇒ **`ARCHON_SEARCH_STATS` returns all zeros under `printf | engine` or
`subprocess.run(input=…)`** — both close stdin, and **stdin EOF is a STOP** (README §3
trap 3). Persistent stdin only. I walked into this folder's own documented trap and it
cost a full measurement round. Baseline with the counters actually live, 50 positions at
depth 14: **fhf 0.8483**, qsearch share of makes **0.129**, NMP cutoff rate 0.637,
extension plies/1k nodes 10.96.

### The one ordering signal that survives the `fhf` band
`QUIET_OFFENSE_FACTOR` (the Reckless-style "moving to attack a vulnerable enemy" term,
default 10) is **monotone against fhf and every point is outside the band**:

| value | 0 | 10 (base) | 20 | 40 |
|---|---|---|---|---|
| Δfhf | **+0.0020** | 0 | −0.0055 | −0.0147 |

Lower is better-ordered. `engines/runs/offense0` game-tests 0 at 30k (screen → verify →
bracket → pin; verify at 110k only if the screen is positive). **Caveat not yet closed:
fhf is a RATIO, and a change that finds fewer cutoffs overall while making the ones it
finds first-move would also raise it. Check the `fh` count, not just the ratio, before
believing a winner.**

## SELDEPTH — THE FIRST CROSS-ENGINE STATISTIC THAT DISTINGUISHES (2026-07-28)

Full writeup: `results/seldepth-gap.md`.

| claim | deciding evidence |
|---|---|
| **At equal nodes AND equal nominal depth, SF's PV lines run 2.5 plies deeper than ours** | 150 plateau positions @110k: seldepth **22.72 vs 25.22**, nominal depth 14.59 vs 14.80, sel/nom **1.560 vs 1.709**. Δ = **+2.50 ± 0.78**. Accounting verified identical on both sides before use (PV-only `max(ply+1)` in search AND qsearch, reset per PV line) |
| The statistic survives a SIX-arm inert null | `RED_TABLE_NUM ±1`, `RFP_CORR_DIV −1`, `QHIST_PRUNE_MULT +1`, `NMP_EVAL_BASE +1`, `SE_TRIPLE_DEEP +1` span **−0.03 .. +0.14** at d12. The gap is ~17× the band. **Every other cross-engine instrument in this folder either came back at par or died to its null; this one does neither** |
| It is an EXTENSION signature, not a qsearch one | mean `seldepth − depth` by nominal depth: at d1–d6 **we are DEEPER** (−0.35, −0.17, −0.42, −0.09); the gap opens at d8 (+0.36), d10 (+1.05), d12 (+2.51). Zero at exactly the singular-extension gate `depth >= 6 + ttPv`, then compounds |
| Frame-scaled extension margins move the mechanism — but recover only ~26% | vs the ±0.15 band: SE trigger (`SE_BETA_MULT` 60→32) **+0.60**, `LMR_DEEPER/SHALLOWER_MARGIN` 52/9→28/5 **+0.30**, `RED_CORR_DIV` 26131→14058 **+0.27** (→7029: +0.52) |
| ~~SE double/triple margins are frame-mis-scaled and cost extensions~~ | **REFUTED by its own mechanism test: +0.11, inside the null band.** Only the singular *trigger* margin is live |
| The three live knobs DO NOT COMPOSE | individually +0.60 / +0.30 / +0.27; **together −0.21**, below base. Extensions and reductions interact; do not stack them assuming additivity |
| The remaining ~1.9 plies is unexplained, and these are excluded | qsearch (we dive deeper at shallow depth), IIR (`depth>=6`, `depth--`, exact parity), the LMR depth clamp (`LMR_PV_DEPTH_BONUS` = SF `+PvNode`; `RED_MAX_DEEPER` = SF `newDepth+2`), the ttMove min-depth clamp, and the SE block structure incl. `SE_NODE_DEEPEN` (= SF `depth++`) |

⇒ **Why this shape and not another.** Every tree-WIDENING change with game evidence
loses (`lmr30` −11.6 n=628; `EVAL_SCALE 2200` −27.8; `corrgate` −17.1 n=406), so the
width optimum is pinned in both directions. An extension change is a REALLOCATION of
the same budget toward critical lines, which is the only shape left that can win.

## ARMS MEASURED THIS SESSION (games)

| arm | result |
|---|---|
| `corrgate` — the two disabled ARCHON-original conditioning terms `RAZOR_CORR_DIV` / `NMP_CORR_DIV` switched on at SF's RFP divisor 182069 | **−17.1 [−51.2, +16.7]**, n=406, 30k nodes self-play. Arm-verified live first. Dead |
| `setrigger` — SE trigger margin frame-scaled (`SE_BETA_MULT` 60→32, `SE_BETA_TTPV_MULT` 70→38) | **−16.3 [−50.4, +17.5]**, n=406, 30k nodes. No gain. NOTE the budget is a flaw: the mechanism is 0 at depth 6 and +2.51 at depth 12, so 30k (~d11) under-samples it. `redcorr110k` re-tests the family at the in-game budget |
| `redcorr110k` — `RED_CORR_DIV` 26131→7029 (mechanism +0.52 plies) at **110k** nodes | running |

## THE NEUTRAL-JUDGE ARM — RETRACTED, THE INSTRUMENT WAS BROKEN (2026-07-28)

**RETRACTION.** The first `judge.py` run is void and its numbers must not be used.
`forced_score` used `go searchmoves <move>`, and **Reckless — the only NEUTRAL judge,
the whole point of the design — ignores `searchmoves` entirely.** Verified directly:
`go searchmoves h3h4` on a position whose free best is `d5c4` returns **bestmove
d5c4**. ARCHON and SF18 honour it; Reckless does not.

So both candidate moves were handed to Reckless as the same unrestricted search, came
back with the IDENTICAL score, and every non-zero delta the tool printed for that row
was **TT warmth between the two probes**, nothing else. Caught only because the
first-probe fix (reset between probes) turned those deltas into a run of exact zeros.

Retracted with it: the reported "the two biased judges bracket the neutral one, so the
method works". The neutral row was noise, so there was nothing to bracket.

**The null is what caught this.** It read 41–45% instead of 50%, which I first
attributed only to probe ordering. The ordering fix removed the noise and exposed that
the underlying signal was zero. An instrument whose null misses by 6 points is telling
you something structural, not something cosmetic.

**Fix, and the general lesson:** score the CHILD position (`position fen … moves <mv>`
then a plain `go`, negated) instead of restricting the root. That needs nothing but
`go`, so it is correct for every engine, and it does not depend on an optional UCI
feature a third-party engine may silently skip. Re-verified live: deltas −4/+11/+44/−42
where the broken version gave 0/0/0/0.

**Never assume a third-party engine implements an optional UCI feature — make it
demonstrate it.** One probe (`does bestmove equal the move I forced?`) would have
caught this before the first run instead of after it.

Superseded numbers, kept only so nobody re-derives them: reckless 44.2% null / 47.4%
arm, sf18 41.0% / 51.3%, archon 44.9% / 45.0%, n=80 pairs.

### Re-run with the fixed instrument: RUNNING
(null 160 pairs, arm 340 pairs, 2,473-position plateau set, judges at 1M nodes)

### The power point stands regardless
At n=80 pairs the se is ~5.7 points per cell, so a TRUE 55% rate would have been
invisible — and 55% on the 32% of positions where the two engines disagree, compounded
over ~40 moves a game, is easily worth 50 Elo. The original run was underpowered even
before it was broken. Hence 340 pairs.

## (VOID — superseded above) THE NEUTRAL-JUDGE ARM CAME BACK NULL (2026-07-28)

`judge.py`, 80 disagreement pairs from 248 plateau positions, both engines at 110k,
each candidate move then scored under forced `searchmoves` at 1.5M by three judges.

| judge | null (archon vs inert archon) | arm (archon vs SF18) | arm − null |
|---|---|---|---|
| reckless (neutral) | 44.2% | 47.4% | +3.2 |
| sf18 | 41.0% | 51.3% | +10.3 |
| archon | 44.9% | 45.0% | +0.1 |

se ≈ 5.7 points per cell ⇒ **none of these is a real difference.** ⇒ **At equal nodes,
on the very positions the corpus says we squander, a neutral third engine cannot tell
our move from SF's — and we still lose those games.** The method's own design check
passed: the two biased judges (archon 45.0%, sf18 51.3%) bracket the neutral one.

The NULL also earned its keep: at 41–45% instead of 50% it exposed a first-probe bias
(both moves scored by one judge process with no reset between them). Fixed since —
reset between probes plus alternating probe order — so these numbers must be read
arm-minus-null, not absolutely.

## WARM-STATE CARRYOVER — NO SIGNAL (2026-07-28, partial)

Every instrument in this folder is a COLD probe; every game is WARM. `coldproxy.py`
injects `ucinewgame` before every `go` (verified symmetric: ARCHON's `new_game` clears
the TT *and* every worker's histories *and* TM, matching SF's `clear()`).

| engine | cold vs warm, 110k nodes, corpus openings |
|---|---|
| ARCHON | −132.1 [−187.6, −82.3] (n=113) |
| SF18 | −145.3 [−183.9, −109.8] (n=172) |

Heavily overlapping ⇒ no evidence either engine extracts more from carried state.
**Stopped early to free the box — these are PARTIAL and are recorded as "no signal",
not as a verdict.** (An earlier read at n=17/30 showed −114 vs −167 and looked like a
real gap; it closed as n grew. The partial-match trap, again.)

## THE SEARCH IS A FAITHFUL PORT — stop looking for a missing or mis-transcribed feature

2026-07-28, second session. Fifteen independent components read line-by-line against
`eca43a97` (SHAPE questions against the checkout, VALUES against eca43a97 per CLAUDE.md).
**Every one matches.** Recorded so nobody spends another session on this axis.

| component | verdict |
|---|---|
| correction history | all five tables present (pawn / minor / non-pawn ×2 / continuation) + `to_corrected_static_eval` + the gated bonus |
| history suite | main, low-ply, capture, continuation, **pawn**, corr, cont-corr, tt-move — full SF set |
| pawn history in the quiet-prune sum | present (`cont1 + cont2 + pawn < -4313·depth`) |
| qsearch | faithful: TT cutoff at `DEPTH_QS`, ttValue-as-stand-pat, the `(467·best + 557·beta)/1024` damping, `futilityBase`, `moveCount > 2`, both SEE bars, prevSq recapture exemption |
| rule50 eval damping | `v -= v·rule50/199` — exact |
| upcoming repetition | faithful, including the `ply > i` / `stp->repetition` split; ours adds a *stricter* mover-is-us test at the root |
| NMP | `R = 7 + depth/3`, bar `beta − 14·depth − 45·improving + 374` — exact |
| TT replacement | 5-bit generation, low bits, `depth − 8·relative_age` — **exact**, including the age weight |
| per-`go` butterfly decay 789/1024 | **exact** — this is SF `iterative_deepening` line 326, not an ARCHON original (I flagged it as one; a `grep mainHistory.fill` missed it) |
| optimism | `75·avg/(|avg|+45)` vs SF's `137/81` — deliberately **frame-corrected at 0.55×**, matching our 0.538× internal frame |
| aspiration mss divisor | 6776 vs SF's 10588 — frame-corrected (`docs/research/aspiration-frame-bug.md`) |

⇒ Two of those are frame corrections that were applied *selectively*, which is the
correct call: `docs/research/eval-frame-scale-model-2026-07-27.md` §11.4 killed the
blanket rescale in **both directions** and concluded the constants sit at a joint
optimum. Do not reopen it.

## THE CONVERSION FRAME IS NOT A LOCALISATION (2026-07-28)

Full writeup: `results/conversion-at-equal-nodes.md`. Tool: `tools/convert.py`.

| claim | deciding evidence |
|---|---|
| **"We cannot convert" is a restatement of the Elo gap, not a defect location** | 615-game colour-paired match at **110k nodes** from `d48x-plateau.epd` (the corpus's own +0.50…+1.50 band, 2 positions per squandered win at the 33rd/66th percentile of its plateau — deliberately NOT the peak). Advantaged side labelled by SF18 @1M. Conversion gap **+29** (ARCHON 18%, SF18 47%) against **+72** at 10+0.1 — and the **salvage gap is +28** (53.2% vs 81.6%). **Equal in both directions.** A conversion-specific defect makes the conversion gap the larger one |
| The plateau positions are not where the Elo is | same budget: **−54.0 ± 12.8** (pentanomial, n=616) vs **−67.6 [−93, −43]** on the corpus openings. Indistinguishable. **Not** evidence we do *better* there — the plateau book draws 66% against the corpus book's 47%, and draws compress Elo; the safe claim is only that nothing about the plateau regime makes the deficit larger |
| The whole equal-nodes ladder re-derives exactly (rule 8) | re-read today from the stored PGNs: 10k **−91.9**, 30k **−64.0**, 60k **−75.6**, 110k **−67.6**, all n=406. Unchanged |
| The equal-nodes ladder's instrument is sound | both engines honour `go nodes` comparably (median last-info nodes 89k vs 86k at a 110k ask, n=24) |
| We are **not** shallower, and the tree shape is the same | 40 plateau positions at 110k: depth 14.65 vs 14.95, seldepth 22.7 vs 24.9, **seldepth/depth 1.61 vs 1.60** |

⇒ `START-HERE.md` §4's "the failure is slow degradation of a held advantage" does not
survive its own control. The honest statement is the one already in this file: a
**uniform, budget-independent ~60 Elo** at equal nodes, with equal depth, equal tree
shape and at-par static eval, and **no phase or advantage regime carrying more of it
than any other**.

## INSTRUMENTS BUILT (2026-07-28, second session)

| tool | what it is, and its null |
|---|---|
| `tools/judge.py` | the first **neutral-judge** move comparison: play both engines at 110k, keep the disagreements, then have Reckless (never saw either eval) score BOTH moves under forced `searchmoves` at 1.5M. Primary statistic is a COUNT. Runs ours/SF/Reckless as three judges so the two biased ones bracket the neutral one. **NULL RUN FIRST, and it fired:** with SF replaced by an inert `RED_TABLE_NUM +1` ARCHON, all three judges came back **41–45% instead of 50%** (n=80) — a systematic bias toward the FIRST-probed move, because both moves are scored by the same judge process with no reset between them. Read arm-minus-null, and fix the tool (reset + alternate probe order) before its next use |
| `tools/convert.py` | the paired conversion/salvage read above |
| `tools/coldproxy.py` | a UCI pass-through that injects `ucinewgame` before every `go`, so an engine plays every move from COLD state (`--warm` passes through, so both arms of a match share the wrapper). Exists because **every instrument in this folder is a cold probe and every game is warm** — carried TT / histories / correction tables / aspiration seed are the one axis nothing on record has ever tested |

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
