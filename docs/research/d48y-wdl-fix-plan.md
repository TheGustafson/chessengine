# D48Y — the WDL/target-transform fix, retrained

**User directive (2026-07-28, 8-hour autonomy window):** fix the WDL scaling —
apply the correct draw-aware transform, fitted PER CORPUS so all corpora align.
Then retrain D48X's architecture with it: **300 superbatches in one run**, usual
LR to sb240, gentle anneal to 0 from sb240→300. Verify it trained and
converged. Then test.

## Why (the measured case)

Three independent instruments say our WEIGHTS, not our search or features, are
the deficit — all in ARCHON's own search at fixed 110k nodes:

| weights | Elo vs D48X | n |
|---|---|---|
| Stockfish `eca43a97` | **+90.7 ± 16.9** | 380 |
| Reckless 0.9.0 (768-wide) | **+30.0 ± 18.0** | 406 |
| our net, de-framed (null) | −12.0 ± 18.2 | 406 |

and `sf-features` proved our HalfKA + FullThreats indices are IDENTICAL to
SF's (40/40 × 2 perspectives), so it is not the feature set.

**The defect.** The trainer builds `target = wdl*result + (1-wdl)*eval_t`.
`result` is an expected score (draw = 0.5). `eval_t` is
`sigmoid(score/400)` — a win probability with **no draw model**. The two
halves are different quantities and we average them. At +174 the eval half
says 0.607 where real outcomes score 0.75; the blended label is ~0.65. The
error is symmetric about 0.5, so it CANCELS IN THE MEAN and does not cancel in
the FUNCTION: every position is pulled toward 0.5 and the learned curve is
systematically flatter than reality. Invisible to rank-correlation tests
(ranking is scale-invariant), which is why our net looked fine on those.

**The fix already exists in the code**, unused: `features.py:545-552`,
`TrainConfig.target_offset` (`spec.py:309`), default `0.0` = the broken branch.
Measured on our own corpora, held out: Brier **0.0354 vs 0.0532**.

**The irony in D48X's own config:** `teacher_frame_offset: 163.0,
teacher_frame_scale: 76.9` — its teacher WAS trained in the calibrated frame,
and we mapped the teacher DOWN into the plain-sigmoid label frame to make them
agree. We converted the good frame into the bad one.

## Why per-corpus

The transform is not a property of positions; it is a property of the LABELLING
PROCESS and the POPULATION — label depth, who played the games out, and the
position mix all change what "score s" predicts about the outcome. Our own
measured fits disagree by >2x in scale (uho a=175.5 b=71.9; fishpack32
a=160.2 b=33.2). One global transform over a 12-corpus mix is a compromise that
is wrong for every component.

**Confound to rule out first:** a scale difference can also be a UNITS
difference (the `SFBINPACK_SCORE_DIV` frame bug class). The training study
tested frame-mixing across 11 corpora and called it a null (±13%), but a 2x
gap exceeds that null, so the fits get a units sanity check before use.

## Plan

1. **Fit** `(a_i, b_i)` per corpus by MLE, held out, bootstrapped
   (`tools/nnue/fit_target_transform.py`).
2. **Sanity-check** the spread: units confound vs genuine predictiveness.
3. **Thread per-corpus params** through `make_mixed_batches`. This is cheap
   because each batch is drawn from exactly ONE corpus (`data.py:95-130`), so
   the corpus index is known at map time.
4. **Verify calibration** post-transform: every corpus's targets should agree
   with observed outcomes (Brier per corpus, before vs after).
5. **Train** D48Y: same arch (`arch_hl 1280, arch_l2 16, arch_l3 32,
   out_skip`), 300sb, no teacher, LR flat to sb240 then anneal to 0 by sb300.
6. **Verify convergence** — TB telemetry armed at launch (standing rule), and
   the heldout MUST be computed in the ARM'S OWN FRAME. D48X's heldout is
   pinned at `wdl 0.3 / scale 400 / offset 0`, which would score a calibrated
   net as a regression BY CONSTRUCTION.
7. **Test** vs D48X: fixed-node screen first (attribution), then real-TC.

## State log

- (start) plan written; corpora verified present; GPU free.
