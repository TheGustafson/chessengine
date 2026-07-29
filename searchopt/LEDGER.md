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

## ★★★ THE ARM DESIGN USED ALL SESSION IS UNDERPOWERED BY 3–10× (2026-07-28)

| n | 95% CI half-width |
|---|---|
| **406** (what I used for 8 arms) | **~24 Elo idealised; ~40–50 observed** |
| 1,000 | ~15 |
| 1,529 (this folder's `pfut-sprt-12k`) | ~12 |
| 4,000 | ~8 |
| 10,000 | ~5 |

**A good Stockfish search patch is worth +1 to +4 Elo. A strong one is +5 to +10.**

⇒ **Every "null" reported this session rules out only LARGE effects.** `corrgate` −17.1,
`setrigger` −16.3, `redcorr110k` −13.3, `offense0` −1.7, `semindepth4` −1.2/+4.8,
`asp30` +6.2 — not one of these had the resolution to distinguish "+3 Elo" from "0".
They are **not** evidence that these levers are dead; they are evidence the instrument
could not see them. Do not cite them as refutations of the underlying ideas.

⇒ **What survives from them:** `asp100` (−49.6 at n=155) is large enough to read, and
the DIRECTIONAL brackets (widening past ~30 hurts) are real because the effects there
are big. Everything in the ±25 band is simply unmeasured.

⇒ **Consequence for method — the arm-per-400-games workflow is unusable for real search
work.** Either run SPRT (sequential, stops early when the result is clear, which is what
`pfut-sprt` did) or budget n ≥ 1500 per arm. At fixed 30k nodes a game costs ~6s, so
1500 games at concurrency 4 is ~40 min — affordable, and roughly what this folder's
better-calibrated earlier arms already did (`pfut-sprt-12k` n=1529, `rfp-threat0`
n=1200). **My n=406 runs were a regression from the folder's own standard.**

⇒ **And it is the argument for SPSA.** SPSA never has to resolve one parameter against
noise: it aggregates gradient information over thousands of cheap paired comparisons and
moves all coordinates at once. For effects of +1 to +4 Elo that is the correct
instrument, and single-arm A/B is the wrong one.

### ⚠⚠ CORRECTION (2026-07-28, later): THE DAMPING HAS A QUANTUM, AND MY CEILING WAS BELOW IT

I wrote below that "any multiplicative factor on `optimum()` inherits this damping" and
killed the arm. **That conclusion was wrong, and the reason is arithmetic I had not yet
derived when I wrote it.**

**One extra ply costs 2^(1/2.02) = 1.409× nodes** (from the 2.02 ply/doubling measured in
games). A TM budget multiplier below ~1.41× therefore buys **nothing on average** — the
search cannot start an iteration it cannot finish, so the budget increase is quantised
away entirely.

| arm | factor across the balanced band | verdict |
|---|---|---|
| `TM_SCORE_C=256`, ceiling **1.25** | 1.13 – 1.25 | **entirely below the quantum — absorbed** |
| `TM_SCORE_C=1024`, ceiling 2.5 | 1.50 – 2.00 | above it; measured **+8%** share of clock |

**I set `TM_SCORE_CEIL = 1.25` by copying `unc_tm_factor`'s clamp, without checking it
against the cost of an iteration.** The knob could not work at C=256 by construction, and
the games correctly reported 0%. The correct statement is:

* **below ~1.41×: absorbed completely** (not "damped" — absorbed);
* **above it: works, but coarsely**, one ply at a time.

⇒ **The lever is NOT refuted — it was never tested at a workable magnitude.** The only
configuration whose mechanism was verified in games (C=1024, ceiling 2.5, +8% clock share
in the balanced band and flat elsewhere) has **no game result**. That is the arm worth
running.

⇒ **General rule, and it applies to `ARCHON_WDL_TM_C3` and `unc_tm_factor` too:** both are
clamped to [0.8, 1.25] and [0.8, 1.25]. **Both ceilings sit below the 1.41× quantum**, so
both would be absorbed even on a net that fed them. Any TM budget factor must be able to
reach ~1.5× to do anything at all.

### ⚠ THE MULTIPLICATIVE TM KNOB IS HEAVILY DAMPED — measured, and it kills the arm

`TM_SCORE_C` (landed, default 0, bit-identical) scales the soft-stop budget by
contestedness. Cold, it works exactly as designed: a balanced root goes 9163 → 10245 at
C=+256 and → 8081 at C=−256, with a winning root unchanged at 1153.

**In GAMES it barely moves anything.** `nodestime=150`, tc 20+0.2, mechanism read from the
match PGN (median, and share-of-own-clock, conditioned ex ante):

| setting | budget multiplier at a level root | in-game share of clock, \|eval\| < 0.25 |
|---|---|---|
| C=256 (ceiling 1.25) | 1.25× | **0.944** (median ratio 1.000 — nothing) |
| C=1024, ceiling 2.5 | **2.50×** | **1.081** (median ratio 1.064) |

⇒ **A 2.5× budget multiplier buys +8% of real time.** The two rows are consistent: at
1.25× the predicted effect is ~+1–2%, which is what "nothing" looks like.

⇒ **The budget is NOT the binding constraint.** The soft stop's other factors — stability,
effort, `fallingEval` — end the search before the budget matters, and a balanced quiet
position is exactly where the best move is stable and those factors fire first. **Any
multiplicative factor on `optimum()` inherits this damping**, which also predicts that
`ARCHON_WDL_TM_C3` and `unc_tm_factor` would be weak even on a net that fed them —
relevant to the NNUE study's R6.

⇒ **Consequence:** to move allocation by the ~20% the measurement calls for, the change has
to be to the STOP CONDITIONS (which factor dominates in quiet positions), not to the
budget they are compared against. That is a different and larger piece of work.

⇒ **The knob stays in at default 0** (exactly 1.0, bit-identical, gate green) as a
documented, measured dead end rather than a deletion — the damping number is the useful
artifact.

⇒ **Method note:** the SPRT was stopped at n=27 rather than run to completion, because the
mechanism check said the arm does not do the thing it was built to do. **Verify the
mechanism IN THE MATCH PGN before buying thousands of games**, not just in a cold probe —
the cold probe said +11.8% and the games said 0%.

## ★★ A PAIRED, ORDER-ALTERNATED DESIGN DOES NOT SURVIVE A THRASHING BOX (2026-07-28)

Same script, same paired ABBA ordering, same two binaries (old-pgo vs a PGO build of
merged main), measured twice:

| conditions | paired new/old | range | I reported |
|---|---|---|---|
| swap 100% full, another agent's gauntlet running | **0.9834** | [0.9626, 1.0004] | "a −2.9% REGRESSION" |
| load ~5, memory free | **1.2103** | [1.1813, 1.2585] | "+21%" |

**The two ranges DO NOT OVERLAP.** This is not sampling noise, and **pairing did not save
me**: order-alternation cancels *linear drift*, not a regime where one binary's memory
behaviour degrades differently under thrashing. I published the regression and retracted it.

⇒ **Rule: record `uptime` and `free -g` beside every wall-clock measurement, and discard
any taken while `available` < ~2 GB or swap is full.** Fixed-node/instruction counts are
contention-immune; cycles, NPS and SMP are not, and no amount of pairing fixes that.

⇒ This is the same failure family as THE COLD-PROBE REGIME above: an instrument that is
internally consistent and externally invalid because its *conditions* differ from the
conditions of the claim.

## ⛔ RETRACTED — BOTH OF MY NPS NUMBERS FOR THE PERF WORK (2026-07-28)

I reported, in order, **−2.9%** ("the merged perf work is a regression") and then **+23%**
("it is a large win"), and blamed the difference on box load. **Both are void and the load
explanation was wrong.** A low-load re-run of the first configuration still gave −1.6%,
while the other configuration gave +21% at the same load, on the same pair of binaries.

**The cause is the eval change, and it makes fixed-node NPS meaningless between these two
binaries.** Net-only fidelity panel, 800 positions (`tools/nnue/calib/scale-check-800.fens`,
`perf-artifacts-8/scripts/evaldiff.py`), old canonical vs studies 8+9:

| | ours | SF cp |
|---|---|---|
| identical | 129 / 800 (16.1%) | |
| mean signed delta | +0.376 | **+0.182** |
| mean \|delta\| | 2.179 | **1.053** |
| p99 / max \|delta\| | 8 / 11 | 3.86 / **5.31** |
| Pearson r | **0.999983** | |

**The eval differs on 84% of positions**, so at fixed nodes the two binaries **search
different trees** — different pruning, different node mix, different per-node cost. A
fixed-node NPS ratio between them is not a like-for-like workload, which is exactly why two
scripts differing only in node count (800k vs 600k) and panel size (20 vs 14) reversed the
sign, and why the ABSOLUTE numbers moved for BOTH binaries between scripts.

⇒ **Rule: never compare NPS at fixed nodes across an eval-changing arm.** Use instructions
per node on a matched panel (workload-controlled), or compare at fixed DEPTH, or measure
the outcome directly in equal-time games.

⇒ The eval change itself is small and unbiased — 1.05 SF cp mean against a mean |score| of
339, max 5.31, correlation 0.999983, and a signed mean of +0.18 vs a 1.05 absolute mean, so
it is perturbation rather than drift. It is consistent with the studies' own claim (0.14 +
0.94 SF cp for the two quantisation changes).

⇒ **The surviving speed evidence is perf study 8+9's own:** 7,116.6 → 6,429.9 instructions
per node (−9.65%) on a matched fixed-node panel with `instructions:u` spread ≤0.01%. That
is workload-controlled in the way my wall-clock comparison was not.

⇒ **Whether it converts to Elo is open** and is being answered directly by an equal-time
match (`engines/runs/s9-convert`, new vs old at 10+0.1), which needs neither NPS figure.

## ★ THE PARITY TARGET INCLUDES A FEATURE WE DO NOT HAVE (2026-07-28)

Every perf study prices ARCHON against **SF18's 4,329.6 instructions/node**. Verified in
the checkout at `cb3d4ee9` — the binary we actually play:

```
src/evaluate.cpp:49   use_smallnet(pos) { return abs(simple_eval(pos)) > 962; }
src/evaluate.cpp:62   smallNet ? networks.small.evaluate(...) : networks.big.evaluate(...)
src/evaluate.cpp:68   if (smallNet && abs(nnue) < 277)   // redo with the big net near equality
```

**SF18 runs a 128-wide net on any position more than ~2.5 pawns lopsided**, and pays for
the 1024-wide net only near equality. Its instructions/node is a **blend of a cheap path
and an expensive one**. We run the full 1280-wide net on every node.

⇒ **Part of the measured "gap" is not inefficiency — it is an architectural feature we
lack**, and no amount of optimising the expensive path can close it. Any statement of the
form "parity is not reachable by patch work" should say *this* rather than gesturing at
difficulty.

⇒ **`eca43a97` — our own constants reference — DELETED the small net** (`git grep
use_smallnet` returns nothing). SF later judged it not worth the complexity. So
benchmarking against SF18 specifically sets a bar SF no longer holds itself to.

⇒ **A version of it is available to us with NO retrain, and it is uncosted.** We already
have a PSQT-only path (the default build uses it), measured at **2.13 Mnps vs 0.54 Mnps
with the net**. Gating it on `|simple_eval| > T` with SF's near-equality recheck is the
same shape using machinery that exists. **The ceiling is a one-hour stub-and-diff:**
(a) what fraction of NODES trip the gate at various T — node-level, not root-level, since
deeper nodes are more lopsided; (b) instructions/node PSQT vs full net; (c) multiply;
(d) only then price the fidelity cost with `evaldiff.py`, exactly as study 8 priced its
quantisation items. **Not asserted to pay** — (a) could come back at 5% and kill it, and
gating eval on a cheap proxy risks losing more Elo than the speed buys.

## ★ INSTRUCTIONS → DEPTH CONVERTS AT ~0.45×, NOT 1:1 (2026-07-28)

Nobody in this chain had measured the end-to-end conversion; every perf study priced its
work in instructions per node and the Elo arithmetic assumed instructions ≈ time.

Equal-time match, `archon-s9-final` vs the pre-study-8 PGO, 10+0.1, canonical net both
sides (`engines/runs/s9-convert`), n=84 games / 4,695 moves per engine:

| | new | old |
|---|---|---|
| total s/game | 17.56 | 17.46 (**ratio 1.0057** — both spend their clock, no banking) |
| mean depth | 18.786 | 18.640 (+0.146 ply unpaired) |
| **paired depth delta** | **+0.110 ± 0.235 ply** | |

Studies 8+9 cut instructions/node by **9.65%**. At **2.02 ply/doubling** (measured in
games) the paired +0.110 ply implies **+3.8% effective speed** and the unpaired +0.146
implies **+5%**.

⇒ **instructions/node → wall-clock depth runs at ~0.4–0.5×, not 1:1.**

⇒ **Consequence for pricing perf work.** A −9.65% instruction cut is worth **+5 to +9
Elo**, not the ~+18 a 1:1 reading gives. For the standing 1.10×-SF target (a further
−26% instructions), ~0.45× conversion makes it **≈ +12 to +21 Elo**, not ~+45.

⇒ **Consequence for match design.** +5–9 Elo needs n≈1000 to become visible. A 400-game
match reading "+4.1 [−71, +80]" at n=84 is not evidence of failure — it is what a real
+5–9 Elo change looks like before it resolves. (See the power entry above.)

⇒ **Caveats, both real.** (a) The comparison spans an eval change — 84% of positions
differ, mean 1.05 SF cp — so the two binaries search different trees, and depth-at-equal-
time mixes speed with an eval-induced tree-shape change. Best available proxy, not a clean
one. (b) The CI still spans zero, though the point estimate has been stable across n=25 →
42 → 84 (+0.102, +0.099, +0.110).

⇒ **The banking hypothesis was tested and REFUTED:** the new binary does not leave clock
unspent (total-time ratio 1.0057), so the shortfall is not time management declining to
use the speed.

## PLY PER DOUBLING, AND WHAT A SPEEDUP IS ACTUALLY WORTH (2026-07-28)

Derived free from the stored equal-nodes ladder's IN-GAME depths (n≈29k moves per rung):

| step | doublings | Δ depth | ply/doubling |
|---|---|---|---|
| 10k → 30k | 1.585 | +3.14 | **1.98** |
| 30k → 110k | 1.874 | +3.87 | **2.06** |
| | | | **mean 2.02** |

⇒ **2.02 ply per node-doubling in games.** A cold EBF of ~1.6 predicts 1.47 — a 37%
disagreement, and one more instance of the cold and game regimes measuring different
things (see THE COLD-PROBE REGIME above).

⇒ **Conversion, with the constant's own uncertainty carried through:**

| Elo/doubling | Elo/ply | perf study 8+9's +0.101 ply |
|---|---|---|
| 120 (the folder's figure) | 59.3 | **+6.0 Elo** |
| 70 (the NNUE study notes the 95% CI includes zero) | 34.6 | **+3.5 Elo** |

⇒ **Method lesson, on my own test.** `engines/runs/s9-elo` runs SPRT `elo0=0 elo1=5`, and
the expected effect is +3.5…+6 — i.e. the bound sits ON the effect size. That is a badly
centred hypothesis: it needs thousands of games and can accept H0 while the change is
genuinely +4. **Derive the expected effect BEFORE choosing SPRT bounds.** For a change
with a strong positive prior and a small effect (a pure speedup), a fixed-n point estimate
with a CI is more informative than a hypothesis test.

⇒ **Independent corroboration of the perf work:** at equal time s9 reaches **+0.101 ply**
over canonical (n=1008 moves), against **+0.13** predicted from its ~9.7% instruction
saving. Prediction and measurement agree — the instruction savings do convert into depth.

## TIME ALLOCATION — the one lever fixed-node testing is structurally blind to (2026-07-28)

Every arm in this folder is tested at fixed nodes, which **cannot see time management at
all**. Read free from the 10+0.1 corpus match, conditioned **EX ANTE** on the engine's own
PREVIOUS eval (known before this move's search, so it cannot be an output of the time
spent — conditioning on the *current* eval is reverse causation and inflates the balanced
band from 0.823 to 0.917). Time as a multiple of each engine's OWN mean, so this is
allocation SHAPE, speed-free:

| prev \|eval\| | ARCHON | SF18 | ratio | n (a/s) |
|---|---|---|---|---|
| **< 0.25** | 0.559 | 0.679 | **0.823** | 2002 / 2692 |
| 0.25–0.75 | 1.014 | 1.015 | 0.999 | 2385 / 3241 |
| 0.75–1.5 | 1.249 | 1.223 | 1.021 | 7792 / 6538 |
| 1.5–3.0 | 1.011 | 0.920 | 1.099 | 3013 / 2941 |
| **> 3.0** | 0.546 | 0.819 | **0.667** | 2465 / 2249 |

⇒ **We under-spend by 18% in balanced positions — the exact band where our root churns
1.56× SF's and where the middlegame drift peaks — and by 33% when clearly winning.** The
conversion band (0.75–1.5) is at parity, so this is not a restatement of the conversion
story.

⇒ **Both engines under-spend in balanced positions** (0.56 / 0.68 of their own mean).
That is TM keying on best-move stability: a quiet position converges, the move stops
changing, and the soft stop fires. Nobody allocates on *balance*. **Spending more where
the score is near zero is a real deviation from SF and nobody's standard TM does it.**

⇒ **Two candidate explanations, and they are NOT the same lever — do not chain them:**
  1. *Independent TM defect* — our allocation shape is simply mis-set, and can be fixed
     directly.
  2. *Symptom of the churn* — our TM is SF-parity in code (design/12); its stability and
     `fallingEval` factors read our own scores, and our scores churn 1.24× more. Then the
     allocation difference is an OUTPUT of the instability, and fixing allocation treats
     the symptom.
  Distinguishing them needs a measurement, not an argument.

⇒ **DISCRIMINATED, free (2026-07-28).** Normalised time vs the engine's OWN previous
score movement (ex ante, |eval| ≤ 3), which is what a TM stability factor responds to:

| prev \|Δscore\| | ARCHON | SF18 |
|---|---|---|
| <0.02 | 0.880 | 0.928 |
| 0.02–0.06 | 1.023 | 1.049 |
| 0.06–0.15 | 1.047 | 1.013 |
| 0.15–0.40 | 1.017 | 0.998 |
| >0.40 | 0.978 | 0.846 |
| **escalation slope (>0.40 ÷ <0.02)** | **1.112×** | **0.911×** |

**The SYMPTOM hypothesis predicts the WRONG SIGN and is disfavoured.** We escalate *more*
on score movement than SF (1.112× vs 0.911×) and we churn *more* in balanced positions —
so if the allocation gap were merely churn flowing through SF-parity TM, we would spend
MORE time there. We spend **18% less**. Explanation 1 (an independent allocation shape)
is the better-supported one.

**Caveat on the proxy:** SF's TM keys on best-MOVE changes, which the PGN does not expose;
own-score movement is a proxy for that, not the mechanism itself. The sign argument
survives the caveat — it only needs our escalation not to be *flatter* than SF's, and it
is not.

**Incidentally: NEITHER engine escalates much on score instability, and SF actively
de-escalates (0.911×).** Whether "spend more when the score is unstable" is an unexploited
axis is a separate question this proxy cannot settle, but it is cheap to ask properly.

⇒ **CLOCK CONFOUND CONTROLLED — the effect strengthens (2026-07-28).** We average 0.274
s/move vs SF's 0.248, so time pressure was the obvious alternative explanation. Split by
phase:

| phase | balanced-band ratio | raw s/move ARCHON | SF18 |
|---|---|---|---|
| mid (plies 40–69) | **0.865** | 0.277 | 0.242 (**we use +14%**) |
| late (plies 70–99) | 0.952 | 0.156 | 0.175 (**we use −11%**) |

**The under-allocation is strongest in the midgame, where we spend MORE total time than
SF** — so it is not time pressure. Late, where we genuinely are squeezed, the ratio
returns toward parity. (Plies 10–39 are untestable: the UHO book makes them unbalanced by
construction, so the balanced band is empty.)

⇒ **Second, separate finding: our time curve is FRONT-LOADED relative to SF's.** We burn
14% more per move in the midgame and arrive 11% poorer late. At 10+0.1 with a 0.1s
increment that is a real cost in its own right, and it is independent of the balanced-band
shape.

### The two existing TM levers are INERT on D48X — measured, not assumed

`src/search/mod.rs` already implements two ARCHON-original soft-stop factors that do
exactly what the allocation measurement asks for, both default off:

* `ARCHON_UNC_TM_C2` — `1 + c2·(σ̂ − σ̄)/σ̄`, clamped [0.8, 1.25]: *unclear → think longer.*
* `ARCHON_WDL_TM_C3` — `1 − c3·decisiveness`, clamped [0.8, 1.0]: *a decided root earns
  less clock, banked for contested positions.*

**Both are inert on the canonical net.** `ARCHON_TM_TRACE` soft-stop sequences are
byte-identical across baseline / `C3=0.5` / `C2=0.5` (17 trace rows each, `opt=175`
throughout). D48X carries neither the σ̂ head nor a WDL sidecar, so `last_wdl()` returns
`None` and `root_sigma` stays 0.0 — the factors return exactly 1.0 by their own guard.

⇒ **Two fully-implemented engine consumers are waiting on net heads that do not exist.**
That is a concrete argument for a WDL/uncertainty sidecar and belongs in the NNUE
architecture study's cost/benefit — the consumer side is already built and free.

⇒ **The head-free alternative, which is what searchopt can actually build:** drive the same
factor from the SEARCH's own root score rather than a net head — e.g. scale the soft stop
by `|root score|` so a decided root earns less clock and the saving banks toward contested
moves. No net change, no retrain, and it uses information the root already has. This is
the direct implementation of the allocation finding, and `nodestime` makes it testable
**deterministically and contention-immune** (`src/timeman.rs` rewrites TM budgets to node
budgets), so it does NOT need the quiet box that equal-time testing needs.

⇒ **Blocked, and why:** testing this needs equal-TIME matches, which need a quiet box, and
a perf study is currently waiting on one. **It is the highest-value untested lever I have**
— it is orthogonal to everything tried this session, it points at the band both
independent localisations identified, and fixed-node testing is blind to it by
construction.

## THE 1.92× NPS GAP IS A COLD-PROBE ARTIFACT — in games it is ~1.4× (2026-07-28)

The ledger's speed figure (ARCHON+net 0.54 Mnps vs SF18 1.03 Mnps, **1.92×**) comes from
a cold `movetime` probe. Re-derived in the GAME regime, free: at `-each nodes=N` both
engines spend the same node budget per move, so the ratio of their per-move times in the
PGN **is** the inverse NPS ratio, measured in real games, with contention hitting both
symmetrically.

| match | concurrency | time ratio | ARCHON nps | SF18 nps | n moves |
|---|---|---|---|---|---|
| `plateau-110k` | 4 | **1.394×** | 1.196M | 1.667M | 35,821 |
| `equalnodes` (110k) | 12 | **1.487×** | 0.468M | 0.696M | 29,000 |
| `eqnodes-30000` | 12 | 1.447× | 0.545M | 0.789M | 27,693 |
| `eqnodes-10000` | 12 | 1.500× | 0.476M | 0.714M | 26,604 |
| cold `movetime` probe | ~1 | **1.92×** | 0.54M | 1.03M | — |

Three independent reasons the cold 1.92× is the outlier, not the truth:

1. **It contradicts its own trend.** Contention WIDENS the ratio (1.394× at concurrency 4
   → 1.487× at 12), so extrapolating toward a quiet box predicts ~1.35×, not 1.92×.
2. **Both engines are far faster in games than cold** (ours 1.196M vs 0.54M) — warm-TT
   nodes are cheaper. Same regime effect as the depth finding: cold probes are not the
   game distribution.
3. **Perf study 8's independent instructions/node ratio is 1.512×** (1.644× before its
   six changes), measured on a matched fixed-node panel with `instructions:u` spread
   0.0000%. That agrees with 1.39–1.49× and disagrees with 1.92×.

⇒ **Do not quote 1.92×.** The in-game speed gap is **~1.4×**, and the instruction-count
and wall-clock measurements now corroborate each other.

⇒ The empirical Elo split is **unaffected** — "~66 Elo speed / ~68 per-node" comes from
the −134 time vs −68 equal-nodes ladder, not from the NPS figure. What changes is the
*interpretation*: the speed handicap we are actually playing under is smaller than the
folder believed, so a given % of instruction savings is worth proportionally more of what
remains.

## OPENING HETEROGENEITY: UNTESTABLE ON A COLOUR-PAIRED BOOK (2026-07-28)

Question: is the −68 Elo uniform across openings, or concentrated in position types?
Attempted from the 110k equal-nodes PGN, 203 openings × 2 colour-swapped games.

**The test I built is invalid and the data said so.** Observed pair-score variance is
**0.547×** the value expected if the two games of a pair were independent — LESS
dispersed, not more. On a colour-paired **unbalanced** book the two games are strongly
**anti-correlated** by construction (the advantaged side usually scores), which is
exactly what the pairing is for. That anti-correlation swamps any opening-level signal,
so **a pair-variance test cannot detect opening heterogeneity on this book.** Recorded so
nobody re-derives it; testing it properly needs many games per opening, not two.

**What the pair-outcome distribution does show** (marginals W/D/L = 69/190/147,
p = 0.170/0.468/0.362):

| pair score | observed | expected if independent |
|---|---|---|
| 1.0 | 1 | 5.9 |
| 0.75 | 26 | 32.3 |
| 0.5 | 72 | 69.4 |
| **0.25** | **102** | 68.8 |
| 0.0 | **2** | 26.6 |

**Half of all opening pairs (102/203) score exactly 0.25 — lose one, draw one.** We
almost never lose both (2) and almost never win both (1). This reproduces AT EQUAL NODES
the corpus's "we lose the bad side and draw the good one" pattern (71% of pairs at
10+0.1), so it is not a speed artifact — it survives removing the 1.5–1.9× speed deficit.

## MOVE-TYPE ASYMMETRY: NOT ESTABLISHED — and the raw version reversed sign (2026-07-28)

Do we play different KINDS of moves than SF? SAN is free in the PGN. Middlegame band
(plies 20–79) of the 110k equal-nodes match, n ≈ 12,000 moves per engine.

**Raw**, both engines' middlegame moves: king moves **1.106×** (13.93% vs 12.60%, ~4σ)
and checks **0.881×** (6.53% vs 7.41%, ~4σ). Two clean four-sigma behavioural differences.

**Both are position artifacts.** The engines occupy different positions — being worse
*causes* defensive move types. Conditioning on the mover's OWN eval being within ±0.5
(balanced, both sides, frame-normalised at the boundary):

| class | raw ratio | conditioned | σ |
|---|---|---|---|
| K | 1.106 | 1.096 | **1.5** — gone |
| **CHECK** | **0.881** | **1.251** | 2.7 — **SIGN REVERSED** |
| Q | 0.988 | 1.221 | 2.8 |
| P / N / B / R / CAPTURE | — | 1.002 / 0.875 / 0.887 / 0.963 / 1.039 | ≤1.7 |

⇒ **Nothing is established.** The two survivors sit at 2.7–2.8σ with **eight classes
tested**, i.e. right at the Bonferroni line (z ≈ 2.73 for α = 0.05/8). Treat as noise.

⇒ **The lesson is the sign reversal.** "We give 12% fewer checks than SF" was a 4σ
result and is the exact opposite of the truth once position is controlled. Any
behavioural comparison between two engines PLAYING EACH OTHER is confounded by who is
winning, because the loser is handed different positions. **Condition on the eval band
before believing any move-distribution claim.**

## WHERE THE DRIFT GOES — decomposed at last (2026-07-28)

`START-HERE` §4 open question 1 was "Where does 5 cp/move actually go? Nobody has
decomposed it." Decomposed here from the stored equal-nodes PGNs, free, in the game
regime, using **SF18's own evals as the judge throughout** (one engine, one frame, no
cross-engine conversion), signs to ARCHON's POV, decided positions (|eval| ≥ 5) excluded:

| plies | mean Δ per move-pair @110k | @30k |
|---|---|---|
| 0–9 | −0.0119 | −0.0091 |
| 10–19 | −0.0149 | −0.0115 |
| 20–29 | −0.0184 | −0.0213 |
| **30–39** | **−0.0316** | −0.0173 |
| 40–49 | −0.0200 | **−0.0284** |
| 50–59 | −0.0263 | −0.0251 |
| 60–79 | ≈ −0.014 | ≈ −0.022 |
| 80–129 | −0.001 … −0.014 | −0.002 … −0.014 |
| 130+ | ≈ 0 / slightly positive | ≈ 0 |

n ≈ 1,000–2,000 per bucket.

⇒ **The bleed is a MIDDLEGAME phenomenon.** It is present from ply 0 but ~2.6× worse at
its peak, concentrated in plies ~20–80, and effectively over by ply ~100. Both budgets
reproduce the shape (the peak sits later at 30k). Accumulated across the peak stretch it
is ~0.6 pawns — the right order of magnitude to account for the corpus's "hold +1.0 and
lose it".

⇒ **Consistent with the volatility localisation**: excess root churn is 1.56× in balanced
positions, and the drift peaks in the middlegame. Middlegame + balanced is where both
independent in-regime measurements point.

⇒ **Caveat, stated not buried:** the near-zero drift after ply ~100 is partly a selection
effect — games still running that long are the drawish ones, and decided positions are
filtered out by the |eval| < 5 cut. The claim is about positions still in play, not about
endgame skill.

## ROOT-SCORE VOLATILITY RE-DERIVED IN THE GAME REGIME: 1.24×, not 3.2× (2026-07-28)

`docs/research/aspiration-frame-bug.md` headlines "**3.2× the oscillation**" (our root
score ranging 1.44 pawns over 14 iterations vs SF's 0.45). That is **one position,
searched cold**. Re-measured from the stored equal-nodes match PGNs — the game regime,
frame-normalised at the parse boundary, decided positions (|score| ≥ 5 pawns) excluded:

| budget | ARCHON | SF18 | ratio | n |
|---|---|---|---|---|
| 110k | 0.1271 | 0.1022 | **1.243×** | ~24k each |
| 30k | 0.1622 | 0.1326 | **1.223×** | ~23k each |

(mean |score change| between an engine's own consecutive moves, in SF-frame pawns.)

**The phenomenon is REAL and reproduces at both budgets** — our root genuinely churns
more than SF's, consistently. **But the magnitude is 1.24×, not 3.2×** — the doc
overstated it 2.6× off a single cold position, and that number was load-bearing for the
whole aspiration line, including my own arms this session.

⇒ Rule 8 in practice: the deciding number was free the whole time (it is in the PGN
annotations of matches that already existed) and had ~24,000× the sample of the number
actually quoted. **Before building on a per-position figure, check whether the same
quantity can be read out of a match that has already been played.**

### The arm's mechanism, checked IN THE GAME REGIME (free, from the arm's own PGN)

Mean |score change| between an engine's own consecutive moves, arm vs base, both ARCHON:

| arm, band | arm | base | ratio |
|---|---|---|---|
| `asp30`, **|s| < 0.25** | 0.0293 | 0.0335 | **0.877×** |
| `asp30`, 0.25–1.0 | 0.1189 | 0.1180 | 1.007× |
| `asp30`, 1.0–5.0 | 0.1681 | 0.1592 | 1.056× |
| `asp100`, |s| < 0.25 | 0.0270 | 0.0301 | 0.899× |
| `asp100`, 0.25–1.0 | 0.1391 | 0.1225 | **1.135×** |
| `asp100`, 1.0–5.0 | 0.1768 | 0.1606 | **1.101×** |

⇒ **`asp30` cuts churn 12.3% in exactly the band where our excess vs SF lives, and is
flat elsewhere.** A targeted effect, in the regime the test runs, costing zero engine
time — the arm's own 406-game PGN already contained it.

⇒ **`asp100` improves the same band but DEGRADES both others**, which is a
mechanism-level account of why it lost 49.6 Elo, and independently reproduces the
"too wide" end of the bracket.

#### The churn statistic, nulled (2026-07-28)
| check | result | what it means |
|---|---|---|
| within-match bootstrap on `asp30`'s ratio (2,000 resamples) | **0.877, 95% CI [0.814, 0.948]** | **excludes 1.0 — the reduction is real.** Arm and base play EACH OTHER in the same games, so this comparison is paired and opponent-controlled |
| base side of `asp30` vs base side of `asp100` — both unmodified ARCHON | 0.898 (10.2% apart) | **NOT a null for the above.** It confounds opponent: base faced a different engine in each match, hence a different position mix. It IS the null for CROSS-match comparisons |

⇒ Worth stating because I nearly talked myself out of a valid result with an invalid
null: a 10% cross-match spread does not impeach a 12% within-match paired effect. It
does confirm that the cross-match comparison declined below ("asp30 closes 85% of the
gap to SF's 0.0281") would have been noise-dominated — declining it was correct.

⇒ **Caveat, because the story is now elegant (README §2 rule 6):** the arm/base numbers
come from a SELF-PLAY match and the SF comparison from a different match against a
different opponent, so "asp30 closes ~85% of the gap to SF's 0.0281" would be a
CROSS-MATCH comparison of different position distributions and is NOT claimed.
And churn reduction is not strength — **the SPRT is the arbiter and it must not be read
early merely because the mechanism now looks tidy.**

### Where the excess volatility lives: BALANCED positions

Same PGNs, same frame handling, split two ways (n in the thousands per cell):

| |score| before the move | ARCHON | SF18 | ratio |
|---|---|---|---|---|
| **< 0.25 (balanced)** | 0.0439 | 0.0281 | **1.561×** |
| 0.25–1.0 | 0.1334 | 0.0998 | 1.336× |
| 1.0–5.0 | 0.1667 | 0.1507 | 1.107× |

| game phase (ply of later move) | ratio |
|---|---|
| opening (<20) | 1.152× |
| middle (20–59) | 1.278× |
| late (60+) | 1.234× |

⇒ **Flat by phase, strongly graded by score.** The excess is not an opening-book or
endgame artifact; it is concentrated in **near-equal positions**, where we churn 56%
more than SF and where games are actually decided. Once a game is decided (|s| > 1) we
are nearly as stable as SF (1.11×).

⇒ **It lands on the arm already under SPRT.** At |score| < 0.25 the `mss` term is ~0
(`mss ≈ score·|score|`), so our aspiration window collapses to the bare base of **5** —
exactly what `ASPIRATION_BASE_DELTA` 5→30 widens. The `asp30` SPRT was launched before
this was measured, so this is a post-hoc coherence, not the reason for the test;
**it does not license reading the SPRT early.**

⇒ Caveat kept explicit: churn is not automatically badness — a search that revises its
opinion may be revising it correctly. What is established is an ASYMMETRY vs SF in the
most decision-critical band, measured in the game regime with high n.

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
SHALLOWER than the same engines at the same node budget in a game.**

**⚠ SELF-CORRECTION on the CAUSE.** I first wrote that this 7-ply gap "isn't position
mix — it's TT warmth." **That was wrong.** A controlled test — the SAME self-play line
walked twice at 110k, `ucinewgame` before every move vs not, 320 searches each — prices
warmth on IDENTICAL positions at:

| | depth | seldepth |
|---|---|---|
| cold | 14.74 | 23.65 |
| warm | 15.67 (**+0.93**) | 27.11 (**+3.46**) |

Warmth is worth ~0.9 plies of depth, not 7. **Most of the cold-probe-vs-game gap is
POSITION MIX** — a game's move list is dominated by simpler middlegame and endgame
positions where depth is naturally far higher than on the hand-picked plateau set — plus
warmth accumulating over 80 moves rather than the 16 this test walked.

What survives, and it is the part that matters: **a cold probe on a curated position set
is not the game distribution, so a mechanism measured there need not appear in a match.**
And note `seldepth` is **3.7× more warmth-sensitive than depth** (+3.46 vs +0.93), which
matters because seldepth is the statistic the SF comparison rests on.

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
| `asp30` (`ASPIRATION_BASE_DELTA` 5→30) | 110k | **+10.3 [−23.5, +44.3]**, n=406 — best of the session, and UNRESOLVED: ±34 cannot separate +10 from 0. Escalated to SPRT |
| `asp30-sprt` (same arm, properly powered) | 110k | SPRT elo0=0 elo1=5 logistic, α=β=0.05, full 2.6M-opening UHO book, random order. Running |
| `asp100` (`ASPIRATION_BASE_DELTA` 5→100) | 110k | **−49.6 [−106.7, +4.8]**, n=155, stopped — large enough to read, and it confirms in GAMES the turnover the mechanism bracket predicted (and the aspiration doc's own "40 good, 400 worse") |
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

---

## CROSS-ENGINE PER-NODE LADDER (2026-07-28, third session) — ESTABLISHED

Same book (`d48x-corpus-openings.epd`, all 203 corpus openings), same 110k-node
budget, 406 games each. Fixed NODES, so these are contention-immune and were
run on a loaded box without prejudice.

| opponent @110k nodes | Elo | n |
|---|---|---|
| SF18 | **−68** | (prior rung) |
| **Reckless 0.9.0** | **−36.9 ± 18.4** | 406 |

Ptnml(0-2) vs Reckless: [3, 75, 88, 36, 1], WL/DD 1.44. Reckless's per-node
edge over us is roughly HALF SF18's. Both beat us with the clock removed, so
the deficit is not only speed — speed is a separate, already-measured 1.47×
(perf study 10).

**This number does not attribute.** It prices search+eval jointly. The split is
what the net-swap arm below is for; do not read either half off this row.

## EVAL-QUALITY PANEL, ARCHON RESTORED (2026-07-28)

`tools/nnue/eval_bench.py`, 1199 positions, oracle SF18 @50k, rescaled MAE in
SF-printed cp (lower better):

| SF-eca43a97 | SF18 | **ARCHON-D48X** | SF16 | Reckless |
|---|---|---|---|---|
| 54.6 | 57.4 | **57.7** | 61.2 | 73.4 |

ARCHON ties SF18 and leads the panel on sign agreement (89.7%). The ARCHON row
had been silently ABSENT from every prior run of this tool — three stacked
failures (retired default binary, a banner line counted as a value, and a cache
that persisted the failure so fixing the first two changed nothing). Fixed in
`ba28e78`.

**Do not read Reckless's last place as a net verdict.** The positions are from
OUR games and the oracle is an SF, so Reckless is judged on neither its own
positions nor its own family's oracle. This same metric ordered four ARCHON
nets spanning 86 Elo WRONG (2026-07-25). It says "right class", not "stronger".

## THE ARCHON/RECKLESS EVAL FRAME — MEASURED (2026-07-28)

Two independent instruments agree:

- Paired direct regression, n=184 positions: **ours = 0.7316 × theirs** (LS).
- Ratio of the two slopes fitted against an SF18@50k oracle over 1199
  positions: 0.373 / 0.511 = **0.730**.

Different data, different instrument, 0.2% apart. Use 0.7316 at any
ARCHON↔Reckless tool boundary; never hand-compare.

**But it is not a constant, and rule 7 caught it.** The median paired ratio is
0.554 with IQR [0.393, 0.739] — least squares is dominated by large-|eval|
positions, the median by small ones. The two nets differ in CURVATURE, not just
gain. Least squares is the right pick where pruning margins bite, but anything
converted this way is measured *through* the conversion.

Incidentally re-derives our own frame: slope 0.511 ⇒ our printed scores are
1.96× SF-printed, against the 2.07× on record.

## NET SWAP — THE WEIGHTS/SEARCH SPLIT (2026-07-28, RUNNING)

Branch `experiment/reckless-net-2` (firewalled, never merges). Reckless
v0.9.0's actual net evaluated inside ARCHON's search.

Correctness gate passed before a single game: our Rust port reproduces
`engines/bin/reckless`'s OWN `eval` output exactly — **0 mismatches / 50**,
debug and release; the Python reference it was ported from is 12/12 and 50/50.
Two structural predictions were confirmed before any weight was read (feature
count 66864; file size 63,266,880 bytes).

Control quality:
- ONE binary serves both arms; `$RECKLESS_NET` selects. Nothing else differs.
- The hook substitutes the INNER eval and then takes the same
  `apply_outer_shape` as every other path. The first version returned early and
  would have given the swapped arm no optimism blend, no material blend and no
  rule-50 damping — a second difference that would have been read as net
  quality. Caught on self-review, before the match.
- Refresh-only accumulator (3.3× slower: 214 knps vs 704). Irrelevant at fixed
  nodes, and it deletes the stale-accumulator bug class from the one
  measurement whose whole value is being believable.
- Arms verified to actually diverge: 2/8 differing best moves at 50k nodes.

**Reading rule, written BEFORE the result (pre-registration):** the swap arm
winning ⇒ our WEIGHTS are the weaker half and net training is the lever; losing
or tying ⇒ the weights are not the problem and the −36.9 is our SEARCH. Either
way the frame caveat above rides along.
