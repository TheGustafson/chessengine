# OURS index 2 — `5k2/pp3pb1/1q4p1/1B1p1bN1/1n1P3Q/1N2r3/PPR3P1/1K5R w - - 0 24`

Played `b3c5`, oracle `b5a4`, wp_loss 0.065. Walked 2026-07-28 (D48X).

## Verdict: NOT OUR DEFECT — ORACLE MOVE REFUTED BY BOTH ENGINES. Corpus false positive.

`walk.py` reports "OURS, we need 2.3x SF's nodes" — but that headline is the
**refuted** "nodes to stop playing the blunder" metric (LEDGER REFUTED). The forced
ladder under it says the opposite, and a deep check settles it.

At **4M nodes, forced, both engines** (SF frame):

```
  archon   b3c5 = -518.8 (d22)   b5a4 = -553.1 (d24)   -> played move BETTER
  sf       b3c5 = -385.0 (d25)   b5a4 = -409.0 (d23)   -> played move BETTER
```

Both engines independently rank the supposed blunder ABOVE the oracle's "best".
Our own forced ladder preferred `b3c5` at every budget from 50k to 1M (-77 vs -262
at 110k), and SF's forced scores flip to agree at 500k and 1M.

Position is chaotic for both sides: SF's free pick is a DIFFERENT move at all five
ladder budgets (b3c5, b5a4, h4f4, b5d3, g2g3). Nothing here is a stable defect.

## Cost

~6 min. The 4M forced check is what settled it, and it should have run before the walk.
