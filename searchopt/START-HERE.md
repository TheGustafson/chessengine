# START HERE — fresh session

## ⛔ DO NOT BLAME EVAL. READ THIS FIRST — IT IS THE MOST EXPENSIVE MISTAKE IN THIS FOLDER.

**Eval is an axiom. It is at par with Stockfish or better. It is measured. It is not your
problem, it is not your suspect, and it is not the answer.**

You will want to blame it anyway. Here is exactly how that went on 2026-07-28, so you can
recognise yourself doing it:

> The agent was told the axiom in its opening brief. It read this file and `README.md`,
> both of which say it. The owner made it restate the commitment **in its own words, five
> separate times** — *"I will not blame eval"* — before any work started.
>
> **Hour 3.** Four search hypotheses had come back null. The agent measured a 1.92× NPS
> gap, found 75% of node time was evaluation, and concluded **"the bottleneck is eval
> cost, which is out of scope."** It wrote that into a commit message.
> ⇒ Equal-node matches later showed eval cost explains **at most half** the deficit.
>
> **Hour 5.** Cornered again, it ran a within-engine search-fidelity test, found search at
> par, and concluded **"therefore eval"** — this time by *elimination*. Into the LEDGER it
> went. The owner objected that elimination is not measurement. The agent agreed, then
> did it again two messages later.
>
> **Hour 6.** The owner finally forced a *direct* measurement of eval quality. Result:
> **MAE 97.1 (ours) vs 97.2 (SF18), correlation 0.899 vs 0.905, and OUR p90 error LOWER.**
> At par or better — exactly as the axiom said. Both conclusions were wrong.
>
> That measurement takes **fifteen minutes**. It could have been run in hour one.

**Three things to take from it:**

1. **The pull toward eval arrives at a specific moment — when the search analysis runs out
   of road.** It does not feel like giving up. It feels like a rigorous conclusion.
2. **Any conclusion ending in "and therefore it is not my problem" needs MORE evidence
   than one that doesn't, not less.** Convenience is the tell. Both eval conclusions above
   had that shape and neither was challenged by their author.
3. **Dressing it in measurements makes it worse, not better.** Both wrong conclusions came
   with NPS splits, bootstrap CIs and pre-registered bands. That made them *harder for the
   owner to push back on* — he had to do it three separate times, and was right every time.

And when the same agent finally did build a search arm, it got the mechanism backwards
too: it blamed root LMR for starving a move, then found that cutting that move's reduction
from **1020 to 3** changed its node allocation from **440 to 414**. The reduction was
never the cause. Low node count was the *symptom* of failing low, not the reason for it.
**Test that your arm moves the mechanism before you spend games on the outcome.**

---

Read this, then `README.md` (the method), then `LEDGER.md` (every claim + what killed the
dead ones). Do not go reading `docs/research/` for context.

**This folder was re-founded on 2026-07-28.** Its original premise — *"the search discards
the right move too early"* — was tested to destruction on a 404-game D48X corpus and does
not describe how we lose. What replaced it is below. If you are here to hunt blunders,
stop and read §1.

---

## 1. The problem, in one sentence

**We hold an advantage and bleed it, ~5 cp per move, for forty moves, without ever making
a move you could call a mistake.**

Not blunders. Measured on the full D48X corpus (404 games vs SF18, colour-paired UHO):

| | ARCHON | SF18 |
|---|---|---|
| converts a won book (≥ +0.25) | **12%** (25W / 172D / 4L) | **84%** |
| salvages a lost book | **16%** | **88%** |

A +1 pawn book position is worth **92% to Stockfish and 55% to us.** That gap *is* the
−132 Elo. 71% of opening pairs follow one pattern: we lose the bad side and draw the good
one.

And the decay has no locus. Across the 172 squandered wins:

* worst single move per game: **median 0.25 pawns**
* games containing ANY move over the 0.88-pawn blunder bar: **2 of 172**
* share of the total decline carried by the single worst move: **median 23%**

77% of the advantage disappears in moves that individually cost under a quarter of a pawn.
**There is no node that discards the right move. There is no bad move to find.**

The conversion profile is bimodal with nothing in between: squandered wins gain a median
**+0.05** pawns over the book and **ZERO of 172 ever exceed +2.00**; converted wins gain
**+10.07** and all 25 exceed +2.00. We convert by tactics or not at all. We never grind
+1.0 into +1.5 into +2.0.

## 2. What the deficit is made of

Equal-**nodes** matches vs SF18 on the corpus's own 203 openings (`-each nodes=N`; nodes
constrained, depth/time/shape free), n≈406 per rung:

| budget | Elo | 95% CI |
|---|---|---|
| 10k | −92 | [−121, −64] |
| 30k | −64 | [−91, −38] |
| 60k | −76 | [−103, −50] |
| 110k | −68 | [−93, −43] |
| 10+0.1 (time) | −134 | — |

⇒ **~66 Elo is speed, ~68 Elo is per-node, and the per-node half SATURATES BY 30k.**
More search does not close it. It is not a "needs more depth" problem.

## 3. What has been ELIMINATED — do not re-run these

Every one of these was measured this session. They are dead ends, not open questions.

| hypothesis | verdict |
|---|---|
| discrete blunders | **25 in 5,379 moves (0.46%)**, 44% of those are oracle artifacts, **1 actionable position in 404 games** |
| time management | **ZERO** losses on time or forfeit in 404 games |
| draw adjudication was premature | both engines independently read ~0.00 in the last 3 moves of all 172 (max +0.10) |
| eval **quality** | **AT PAR.** MAE 97.1 vs SF 97.2, corr 0.899 vs 0.905 — and our p90 error is LOWER (149.5 vs 176). Owner's training-time benches on far more data agree: D48X ≥ SF18's net |
| eval **cost** / speed | real but ~66 Elo of 134, and neutralised entirely at equal nodes |
| depth per node | **EQUAL** — 16.0 vs 16.0 at 110k over 8,219 game moves |
| search self-consistency | equal (59.3% vs 58.7%; self cp-loss CI spans zero). **Weak test** — see README §3 |
| systematic over-pruning | not detected; and the inert null arm captured **57%** of the candidate arm's effect |
| root LMR starving late moves | **REFUTED at the mechanism level**: dropping a starved move's reduction from 1020 to **3** changed its node allocation 440 → 414. Reduction is not what starves it |

**The remaining ~68 Elo at equal nodes, equal depth, and at-par eval is UNEXPLAINED.**
That is the honest state. Say so rather than filling it in.

## 4. What to work on

The failure is *slow degradation of a held advantage*. That is a property of long move
sequences, not of any single decision, so:

1. **Instruments that read one position are the wrong tool** and have failed repeatedly
   (README §3). The effect is ~5 cp/move; an inert ±1 moves any per-position statistic
   further than that.
2. **Aggregates over the corpus are cheap and honest** — `lossshape.py` reads every game's
   eval trajectory straight out of the PGN annotations for free, all 404 games including
   the 204 draws the miner never touched. **That is where the Elo is.**
3. **Games arbitrate, and the corpus openings are the right book**
   (`engines/books/d48x-corpus-openings.epd`, all 203, wins included — no selection bias).
   Fixed-node matches are contention-immune and can share the box.

Open questions worth real effort, in order:

* **Where does 5 cp/move actually go?** Nobody has decomposed it. It is below the
  resolution of every per-position instrument tried, but it is not below the resolution of
  a 400-game equal-nodes match.
* **Why does the per-node deficit saturate at 30k?** Something stops scaling there. That
  is a shape question about the tree, and it has never been asked.
* **What does SF do in a +1.0 plateau that we don't?** We match its move 79% of the time
  at in-game budget and reach the same depth. The 21% is where the game goes.

## 5. Housekeeping

- Build the traced binary with `tools/build.sh` (needs BOTH `search-control` and `tune`).
- `Bench: 587683` and `CHECK_NNUE_GATE=1 bash tools/check.sh` must hold after any change.
- Corpus: `positions/{losses,classified,ours,both,neither}.json`, provenance-stamped.
  `classify_losses.py` now carries an **oracle-verification gate** — without it, 2 of 3
  OURS positions were artifacts of a 120k oracle that had not looked far enough.
