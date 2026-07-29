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

### State log — implementation complete, awaiting fits

- **Per-corpus transform wired and verified** (`51fe78e`). Design change forced
  by the code: `shuffle_chunk=128` pools records from SEVERAL corpora and
  shuffles them, so a batch is NOT single-corpus — params ride PER RECORD via
  an int32 `src` array carried through the same permutation/slicing in both the
  serial and threaded paths. Verified: per-record == scalar bit-exact;
  mixed-corpus batch splits correctly; `per_corpus_target=None` reproduces the
  old pipeline BIT-IDENTICALLY.
- **Two silent failures caught in `pump_train.py` before spending GPU:**
  1. It builds `TrainConfig` from an EXPLICIT field list, so `per_corpus_target`
     was being dropped — the run would have trained the OLD target while the
     config claimed the new one. Now threaded.
  2. `heldout_value_loss` used `TrainConfig()` defaults
     (`wdl 0.3 / scale 400 / offset 0`) — the uncalibrated frame. It would have
     scored a calibrated net as a REGRESSION BY CONSTRUCTION. Now takes the
     run's own transform via `heldout_target`.
- **Heldout corpus fitted**: `uho-band0-5` a=175.1 b=71.6, brier 0.03537 vs
  0.05317 (−33%), bootstrap a [173.6, 175.6] b [70.9, 72.0].
- **Fits so far** (all improve; `a` ~143–176, `b` ~52–91):
  datapv2 166.9/70.3 (−32%), farseerT75 142.8/91.2 (−15%),
  multinet 168.6/69.5 (−32%), t60t70 148.4/81.2 (−17%),
  wisr 167.6/52.4 (−37%).
- **Confound to state plainly when reporting:** D48X had a TEACHER for sb0–60;
  D48Y has none (user ruling). So D48Y-vs-D48X differs in two ways, not one —
  target transform AND teacher. Warm start is held identical
  (`d48x-warm-a3.bin`) and the seed is reused (220480001) so the batch schedule
  matches, which removes the other two obvious differences.

### State log — fits done, units confound RULED OUT, config built, smoke running

**Per-corpus fits (MLE, held out, bootstrapped).** Every corpus improves:

| corpus | a | b | a/b | brier now → fitted |
|---|---|---|---|---|
| uho-band75-100 | 175.5 | 71.9 | 2.44 | 0.05317 → 0.03541 (−33.4%) |
| fishpack32 | 160.2 | 33.1 | 4.84 | 0.04996 → 0.02336 (−53.2%) |
| farseerT74 | 164.5 | 117.7 | 1.40 | 0.08963 → 0.08004 (−10.7%) |
| datapv2 | 166.9 | 70.3 | 2.37 | 0.05188 → 0.03544 (−31.7%) |
| farseerT75 | 142.8 | 91.2 | 1.57 | 0.08625 → 0.07303 (−15.3%) |
| multinet | 168.6 | 69.5 | 2.43 | 0.05057 → 0.03422 (−32.3%) |
| t60t70 | 148.4 | 81.2 | 1.83 | 0.07335 → 0.06087 (−17.0%) |
| pylon | 146.5 | 95.0 | 1.54 | 0.08678 → 0.07539 (−13.1%) |
| wisr | 167.8 | 52.5 | 3.20 | 0.03671 → 0.02326 (−36.6%) |
| wrongnnue | 247.0 | 122.3 | 2.02 | 0.05426 → 0.04630 (−14.7%) |
| canon-tight-frame | 182.5 | 72.5 | 2.52 | 0.05075 → 0.03403 (−32.9%) |
| hardneg-frame | (no fit) | | | 0.013% of the mix; uho params used, immaterial |

**UNITS CONFOUND RULED OUT — the check the plan demanded.** A pure units
rescale by `k` maps `(a,b) → (k·a, k·b)`, so **a/b is invariant** under it.
Observed a/b spans **1.40 → 4.84** (3.46×), and **corr(a,b) = 0.377** where a
units bug would force ~+1.0. So these are genuine SHAPE differences in how
score predicts outcome per corpus — exactly what per-corpus fitting is for —
not a `SFBINPACK_SCORE_DIV`-class units bug. Safe to use.

**Config** `tools/nnue/runs/d48y_1280.json`, generated from the fits by script
(not by hand): same arch/warm/seed/mix as D48X, `distill_alpha 0` and teacher
keys removed, `superbatches 300`, `lr_ramp [[0,1e-3],[240,1e-3],[300,0]]`,
`heldout_target [175.1, 71.6]`.

**Smoke (3sb) launched** and the heldout fix is confirmed live in its log:
`[pump] heldout scored in the RUN frame a=175.1 b=71.6`.

### State log — D48Y LAUNCHED (300sb)

- Smoke (3sb) passed clean: `TRAIN EXIT 0`, value 0.009649 → 0.008460 →
  0.008394, ~190k pos/s. Log confirms the frame fix is live:
  `[pump] heldout scored in the RUN frame a=175.1 b=71.6`.
- **Full run launched.** `data/nnue/experiments/d48y-1280/train.log`, PID in
  `train.pid`, TB shim armed at launch → `data/nnue/tb/d48y-1280`.
  GPU 95% / 7.1 GB. sb1 value 0.009640 @ 189,822 pos/s.
- **ETA, measured not guessed:** 8192 × 3040 = 24.9M positions/superbatch ⇒
  ~131 s/sb ⇒ **300sb ≈ 11 h**. At the 8-hour mark expect ~sb 215–225, i.e.
  the anneal (sb240→300) will NOT have started yet. Running the full 300 rather
  than truncating, because the gentle anneal is the part that sets final
  quality and compressing it would change the requested experiment.
- **Do not misread the heldout.** D48Y's is scored at its own frame
  (a=175.1, b=71.6); D48X's logged 0.006350 is the UNCALIBRATED frame. Not
  comparable. Early D48Y heldout reads HIGH (smoke: 0.0141) because the warm
  trunk was trained in the old frame and is being scored against new-frame
  targets — expected, and it falls as the net adapts.
- **TM real-TC stopped at n=300, final −31.4 ± 18.9.** The question was already
  settled and the match was consuming CPU next to an 11-hour training run. The
  refutation stands: nodestime +52.6 ± 21.2 vs real clock −31.4 ± 18.9.
  `TM_SCORE_C` stays 0.
- Perf study 11 running in an isolated worktree against the single-net
  `stockfish-eca43a97` target.
