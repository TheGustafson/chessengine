# OURS index 1 — `r4r2/6pk/b3Qq1p/pp1B4/5PP1/7P/PP6/4R1K1 w - - 0 29`

Played `e6f6`, oracle `e6e4`, wp_loss 0.086. Walked 2026-07-28 (D48X).

## Verdict: NOT OUR DEFECT — the ORACLE MOVE IS WRONG. Corpus false positive.

`walk.py`: *"budget to stop playing e6f6: ARCHON never, SF18 never."*

SF18 plays the supposed blunder `e6f6` **itself** at 250k, 500k and 1M. It prefers the
oracle move `e6e4` only at 50k-110k. Forced ladder (SF frame):

```
 budget |  A e6f6   A e6e4  A picks |  S e6f6   S e6e4  S picks
    50k |    -151      -33     e6e4 |    -138      -16     e6e4
   110k |    -162      -33     e6f6 |    -143      -20     e6e4
   250k |    -120     -223     e6f6 |    -143      -21     e6f6
   500k |    -153     -249     e6f6 |    -158      -15     e6f6
  1000k |    -160     -275     e6f6 |    -157     -216     e6f6
```

**SF's own forced score for `e6e4` collapses −15 (500k) → −216 (1M).** Ours collapses
earlier (−33 at 110k → −223 at 250k). Both engines eventually agree the "best" move is
bad. The descent then agrees at every ply 0-6 and diverges only at ply 7 by 7 cp.

## Why the corpus mislabelled it — worth generalising

Two independent budget-sensitive decisions stacked:

1. `mine_losses.py` scored the blunder with SF at **120k**, where SF still liked `e6e4`.
2. `classify_losses.py` assigned OURS at **110k**, where SF still liked `e6e4` and we
   did not.

Both budgets sit right at the boundary where SF's own preference flips. So a position
can enter OURS purely because the oracle had not yet seen far enough — no defect of ours
is involved. **This is an oracle-budget artifact, not a search defect**, and it is
invisible to any amount of tracing on our side.

⇒ Any future corpus should verify a candidate's `best` move at a budget well above the
one used to mine it before spending a walk on it. Cheap: one forced ladder.

## Cost

~4 min (walk.py alone; the ladder answered it before any tracing was needed — which is
exactly the "whose defect is this" gate earning its keep).
