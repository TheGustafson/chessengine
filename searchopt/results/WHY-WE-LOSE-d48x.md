# Why we lose — the D48X 404-game corpus, holistically

2026-07-28. Supersedes the blunder framing for this corpus. Every number below is
counted from `engines/runs/corpus-d48x/games.pgn`; none is an estimate.

## The headline

**We do not lose to blunders. We lose because we cannot convert an advantage.**

On the SAME 201 UHO openings, played from both sides:

| | ARCHON | SF18 |
|---|---|---|
| converts a won book (>= +0.25) | **12%** (25W / 172D / 4L) | **84%** |
| salvages a lost book | **16%** | **88%** |

A +1 pawn book position is worth **92% to Stockfish and 55% to us**. That gap IS the
-132 Elo. 71% of opening pairs follow one pattern: we lose the bad side and draw the
good one.

## The mechanism, as far as the data goes

Comparing the 172 squandered wins against the 25 converted ones, on the GAIN over the
book advantage:

| | squandered (172) | converted (25) |
|---|---|---|
| book start | +0.98 | +1.00 |
| peak reached | +1.05 | +11.06 |
| **gain over book** | **+0.05** | **+10.07** |
| ever exceeded +2.00 | **0 games** | **25 games** |

**The distribution is bimodal with nothing in between.** Either the opponent collapses
tactically and we win by miles, or we gain a median of +0.05 pawns across 40+ moves and
draw. We never grind +1.0 into +1.5 into +2.0.

⇒ **ARCHON converts by tactics or not at all.** That is why the blunder hunt found
nothing: in a plateau there is no single bad move to find.

## Controls run (what this ISN'T)

* **Not adjudication artifact.** In all 172 squandered wins BOTH engines independently
  reported ~0.00 in the final 3 moves (max +0.10). Zero games where either still claimed
  >= +0.50. The wins were genuinely gone before the game was called.
* **Not time management.** ZERO losses on time or forfeit across 404 games; every loss is
  adjudication-on-eval or mate.
* **Not eval misvaluation** (and eval is an axiom anyway — this is stated because it was
  MEASURED, not assumed). On game 87's critical positions our eval at the in-game budget
  matches SF's almost exactly: +63 vs +64, +78 vs +61, +98 vs +75 (SF cp). Both engines
  need ~8M nodes to see the draw; neither sees it at 110k.
* **Not book imbalance.** Openings are colour-paired (203 distinct, 201 played twice), so
  imbalance cancels within a pair. The asymmetry is in what each engine does with it.

## Worked example — game 87 (we are White, book +1.08, drew by repetition)

SF's own in-game evals, White POV. No cliff; an 18-move bleed:

```
move 35: +1.94  (peak)
40-43:   +1.55 -> +1.14
44-51:   +0.98 +0.97 +0.93 +0.98 +0.93 +0.94 +1.01 +0.98   <- FLAT for 8 moves
52-54:   +0.67 -> +0.56 -> +0.40
58-62:   +0.36 -> +0.25 -> +0.15 -> +0.01
63-68:   Nf6+ Nd5+ Nf6+ Nd5+ Nc7+ Nd5+  -> threefold
```

At moves 23-25 we shuffle `Rg1-Rh1-Rg1-Rh1` while claiming +1.8. Through the 44-51
plateau our eval reads +1.0 to +1.28 SF-frame, i.e. it AGREES with SF's +0.93-1.01.
We valued it right and still made no progress. By move 63 our eval finally reads -0.01
and we take the perpetual because nothing better exists any more.

## Method notes worth keeping

* **In-game evals are not comparable to cold probes.** SF reported -0.01 at move 62 in
  game with a warm TT at depth 18; a COLD 110k search of the same position gives +64cp.
  Comparing an in-game annotation against a fresh probe measures TT warmth, not skill.
* The PGN's per-move annotations make all 404 games' trajectories free to analyse
  (`searchopt/tools/lossshape.py`) — including the 204 draws, which the miner never
  touched because it only scored LOST games. That is where the Elo actually is.
* FRAME: ARCHON's printed pawns are 2.07x SF's. `lossshape.py` normalises at the parse
  boundary. Once corrected the two engines agree closely (archon -2.61 -> -1.26 vs SF's
  -1.19 on game 87 move 8).

## What this does NOT establish

No search mechanism. "Cannot convert" is a description, not a defect location. The next
question is whether any single search decision in a plateau is wrong, or whether the
failure is genuinely distributed — see the game-87 2M move audit.
