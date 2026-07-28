# Pre-registered: are eval-based pruning margins mis-set for sparse positions?

2026-07-27, written **before** either arm is measured.

## The premise, measured

Across 30,229 move pairs from ARCHON-vs-SF18 games, our eval's disagreement
with SF18's is **unbiased at every material level** (median ≈ 0.00 pawns in all
buckets) but its **dispersion grows 4.7×** as the board empties:

| pieces | n | median | p90 disagreement |
|---|---:|---:|---:|
| ≤7 | 4,101 | +0.00 | **+3.17** |
| 8–12 | 6,151 | −0.00 | +1.80 |
| 13–20 | 8,969 | −0.01 | +1.04 |
| 21+ | 11,008 | −0.01 | +0.67 |

Independently, our real search misses (adjudicated at 256k nodes/move)
concentrate at low piece counts — 4 of 8 at ≤7 pieces against a 13.9% base
rate, p = 0.016.

## The hypothesis

Every eval-denominated pruning gate — RFP, razoring, qsearch futility, parent
and capture futility — compares a **noisy** quantity against a **fixed**
margin. Those margins were transcribed from Stockfish and are constant in
material. If eval precision degrades 4.7× as material comes off, then a margin
that is correctly tight at 21+ pieces is **far too tight at ≤7**, and the gate
prunes on noise exactly where the noise is largest.

Note what this is NOT: it is not a calibration or frame problem. The median
disagreement is zero everywhere, which is why five frame-rescale formulations
found nothing. This is a **precision** problem, and precision problems are not
fixed by rescaling — they are fixed by widening margins where precision is
poor, or by not pruning there at all.

## The prediction, and what falsifies it

Widen the eval-denominated margin families (`RFP`, `RAZOR`, `QSFUT`) by 1.5×
and screen the **same arms on two corpora**:

> **Predicted: a clear improvement on the SPARSE corpus (400 stratified
> endgame positions, ≤12 pieces) and neutral-or-worse on the GENERAL corpus
> (9,992 positions, all 21+ pieces).**

- **Helps on both** → not material-dependent, just a global mistune. Still
  useful, but the material story is wrong.
- **Helps on neither** → the hypothesis is wrong; eval noise in sparse
  positions is not reaching the pruning gates in a way that costs anything.
- **Helps only on sparse** → the margins should scale with material. Stockfish
  does not do this, so it would be an ARCHON original, and the natural
  implementation is a material term on the margin rather than a constant.

The differential is the point. Either corpus alone is uninterpretable: the
general corpus has been screened all day and shows nothing, and a sparse-only
result could just be a global mistune that nobody had tested.
