# Value-Foundation Research State (B4: why we can't beat net-0, and the fix)

> **CONTINUATION INVARIANT:** Read this file completely before every research
> loop; update it after every completed, rejected, or redirected trial. This
> file supersedes `policy-automl-state.md` as the ACTIVE log as of 2026-07-15
> (user-declared transformational point). The old file remains the historical
> archive and its header invariants REMAIN BINDING: the 5k-node bounded-label
> quarantine, the quality-first flywheel rule, the SF18 tiered-oracle plan,
> never reset/discard the worktree, D38 ledger rows for every
> net/corpus/match, local commits only (never push).
>
> **GOVERNING GOAL (user, 2026-07-15):** Find out WHY we struggle to train
> nets that beat net-0 — fast — then spend all effort improving models well
> beyond it. A world-class eval-only engine is the prerequisite for the D41
> research program. Roadmap doc: `archon-self-improvement-roadmap.md` (B4).
> **PROMOTION BAR (user, 2026-07-15 evening): candidates must beat net-0 at
> EQUAL TIME** — fixed-node screens are attribution instruments only; the
> gate is a 200-game 10+0.1 real-TC match (`tools/nnue/prod_tc_gate.sh`,
> width-specific fixed-node floors feed it). net-0 is far below top engines;
> the capability ladder beyond it: calibrated classic → exact HalfKA at
> scale on calibrated fishpack → multi-epoch schedules → self-play datagen
> flywheel + SPSA consumer retune.

Last updated: 2026-07-15 America/New_York

## The theory (ranked, with evidence)

**H1 — Scale/schedule starvation (dominant, prior ~0.6).** Every training run
in project history is 74 s – 15 min of GPU (≤ 747 M presentations; the
30-superbatch constant was never questioned). net-0 wins because 747 M over
its 25 M corpus = 30 epochs = converged; every bulk run was 0.3–3 epochs =
starved. Fits: fishpack classic1024 < classic256 at equal presentations
(classic undertrained-capacity signature); Lichess classic1024 −39 @20k off a
12-minute run. Community engines train tens of billions of presentations;
official SF trains world-class nets from the exact fishpack file we hold. Our
GPU does ~10 M pos/s on classic nets → a 12 B-presentation run costs ~2-4 h.
Compute was never the constraint; nobody turned the dial.

**H2 — Search-consumer coupling (real, magnitude unmeasured, prior ~0.25).**
Pruning margins (futility/razoring/NMP/aspiration) were tuned pre-NNUE and
only ever validated against net-0's output distribution (std ~613 cp, WDL-
sharpened). Bulk nets are flatter (std 257–371 cp); every cp-denominated
margin bites differently. Every Elo verdict flows through this one tuned
consumer. Counter-evidence: net-0 itself runs 2-3x "hot" vs the PSQT scale the
margins were tuned on and still gained +696 — so global scale alone is not
fatal (and the D40 1.65x rescale test hurt). The untested part is
*distribution shape* interaction, not global gain.

**H3 — Label heterogeneity / local coherence (real but secondary, ~0.1).**
Mixed engines/depths/budgets teach a jagged local surface (coherence audits
confirm), but Lichess classic1024 reached −39 @20k with worst-case
heterogeneous labels and zero outcomes → this is a lever, not a wall. The
bounded-label quarantine applies to the 5k-relabel corpora, NOT to
fishpack/Lichess.

**H4 — Pipeline/architecture bugs (exonerated, ~0.05).** net-0 seed-100
reproduction = exactly 50%; parity gates bit-exact everywhere. Stop looking.

## Preregistered probes (launched 2026-07-15)

### Probe A — one real training run (H1 test)

Campaign `value-scale-probe-a-v1` (config
`tools/nnue/configs/value-scale-probe-a-v1.json`, driver
`tools/nnue/value_matrix.py`), identical to the failed matrix-v2 recipe in
every respect EXCEPT the schedule: **480 superbatches ≈ 12.0 B presentations
(16x), full cosine decay to SB480, save every 30 SB.** Runs, in order:

1. fishpack classic256 (control cell; matrix-v2 anchor −163 @5k at SB30)
2. fishpack classic1024 (capacity cell; matrix-v2 anchor −215 @5k at SB30)
3. Lichess classic1024 (best-known bulk cell; anchor −100/−72/−39 @2k/5k/20k)

A watcher (`tools/nnue/probe_a_watch.py`) packs each intermediate checkpoint
and screens it 80 games @5k vs net-0 (per-width engine binaries, paired
openings, concurrency 2) → the **learning curve** is the deliverable.

Predictions: H1 true → curves still climbing at SB480, classic1024 overtakes
classic256, best cell approaches/crosses net-0; the SB30 point reproduces its
matrix-v2 anchor (built-in control). H1 false → curves plateau at ~anchor
level by SB120 with fully-decayed LR.

Decision rule: crossing or clearly converging toward ≥ net-0 ⇒ H1 confirmed —
stop diagnosing, start scaling (schedule → capacity → HalfKA → datagen
flywheel). Hard plateau ⇒ H1 dead for that source; weight shifts to H2/H3 per
Probe B/C.

Known risks accepted: SF-binpack reader must not be interrupted (resume
unsafe: skip_count ignored); cross-binary fixed-node screens for non-256
cells (historical net-10 glitch) — mitigated by same-binary SB-to-SB curve
comparisons and the SB30 anchor.

### Probe B — budget-ladder decomposition (H2 test)

`tools/nnue/probe_b_ladder.sh`: paired 80-game matches vs net-0 at **32 / 256
/ 2,000 / 20,000 fixed nodes** (balanced-256 book, seed 20260716,
concurrency 2) for the three strongest failed nets:

- `value-bulk-matrix-v2-lichess-classic1024.cenn` (1024-width binary)
- `value-bulk-d40-lichess-mid-v1-lichess-d40-full.cenn` (sota binary)
- `value-bulk-matrix-v2-binpack-classic256.cenn` (default binary, same-binary
  pair)

Interpretation: genuinely-worse net ⇒ loses at every budget, roughly
uniformly. Search-coupling ⇒ relatively strong at tiny budgets (pruning
barely acts at 32–256 nodes) and degrading as budget grows. If the coupling
signature appears ⇒ next act = per-net SPSA of the ~10 margin constants at
fixed nodes (D36 scaffold) and re-screen the "failed" nets before any more
data work.

### Probe C — label-semantics spot-audit (H3 guard, Codex's ask)

`tools/nnue/audit_label_semantics.py` (new): sample ~400 records each from
the Lichess canonical train.bin and fishpack32.binpack — decode the exact
bytes training consumed, reconstruct positions, fresh pinned SF18 searches at
fixed depth 12 (completed iterations; 0-bound regime per prior smoke), one
thread. Report sign agreement, Pearson/Spearman, slope, STM-vs-White POV
check, mate handling; JSON + markdown under `data/nnue/audits/`. Any semantic
defect ⇒ quarantine the corpus and re-audit every conclusion built on it
(including Probe A's cells on that source).

### Probe A2 (preregistered 2026-07-15 18:05, runs after Probe A cells finish)

**H3b — per-source target-scale miscalibration (new, from Probe C).** Fishpack
stored scores are SF INTERNAL units ≈ 3.0x SF18-UCI cp (Probe C OLS a=3.026,
R²=0.90); Lichess ≈ 1.28x. The trainer's sigmoid target scale (400) was
calibrated for net-0's UCI-scale SF16 labels, so every fishpack cell trained
against ~3x over-sharp sigmoid targets — the exact direction (over-sharp
targets) that the WDL-logit400 and scale-200 experiments already proved loses
engine strength. Test: fishpack classic256, 30 SB (matrix-v2 anchor recipe,
−163), with stored scores divided by ~3.03 (or equivalent sigmoid-scale
1212) — one ~12-minute GPU run + 80-game screen. Prediction: if H3b is
material, the rescaled cell clearly beats −163; combined with a long schedule
it may transform the fishpack picture entirely.

## Live status

- 2026-07-15: file created; probes A/B/C launching. Worktree committed
  through `a8856b1`; ledger backfilled (exps 0025-0034). No cloud spend
  authorized for the probes; everything local.
- 2026-07-15 ~17:47 ET: ALL THREE PROBES LAUNCHED (detached; pgids of
  probe_a_launch.sh / probe_a_watch.py / probe_b_ladder.sh ≈ 1055970-72).
  Prebuilt match binaries at `engines/bin/probe-{default,l1024,sota}`
  (x86-64-v3; sota + default both verified bench 548978). Probe A trains at
  8-12M pos/s on SF-binpack — a 480-SB cell is ~20-25 min, so GPU work
  finishes in ~1.5h and the watcher's 48 checkpoint screens are the long
  pole (~4-6h). First ladder instrument check: lichess-classic1024 @2k =
  -98.1 (matches the -100 matrix anchor, different seed/book — instrument
  consistent). Probe C runs as a subagent writing
  `tools/nnue/audit_label_semantics.py` + `data/nnue/audits/label-semantics-v1/`.

## Results log

- **2026-07-15 17:50 — Preregistration amendment (Probe A control).** The
  "SB30 checkpoint must reproduce the matrix-v2 anchor" control was
  mis-specified: matrix-v2's SB30 was the END of its cosine decay, while probe
  A's SB30 sits at near-full LR on a 480-SB horizon. Intermediate
  high-LR checkpoints screen far worse (observed: fishpack classic256 SB30 =
  −490 ± 82; cf. matrix-v1's undecayed −512 datapoint). The valid anchor
  comparison is probe A's FINAL SB480 (fully decayed) vs matrix-v2's final
  (−163/−215), at 16x presentations. Curve interpretation must weight the
  trend and the decayed endpoint, not raw intermediate levels.
- **2026-07-15 17:56 — PROBE B COMPLETE (80g/point, seed 20260716, paired
  balanced-256 book, vs net-0).** Elo ± SE by fixed-node budget:

  | net | n32 | n256 | n2k | n20k |
  |---|---|---|---|---|
  | lichess-classic1024 | −132 ± 39 | −26 ± 39 | −98 ± 40 | −48 ± 36 |
  | d40-lichess-mid-sb15 | −48 ± 36 | −75 ± 39 | −137 ± 40 | −39 ± 35 |
  | binpack-classic256 | −185 ± 41 | −112 ± 40 | −293 ± 52 | −293 ± 52 |

  (binpack 2k/20k identical scores verified as coincidence: distinct PGNs,
  correct node configs.) **Verdict: the H2 coupling signature — strong at
  eval-dominated tiny budgets, degrading as pruning margins bite — appears in
  NO net. All three lose clearly at n32, the near-eval-only regime, so their
  raw evaluations are genuinely weaker than net-0's; the negative screens
  were honest.** The fishpack net worsens sharply with depth (−185→−293),
  consistent with its terrible local coherence (156 cp mean residual)
  compounding under search. H2 demoted to a secondary factor (some
  budget-shape structure exists but is noise-level at 80g); per-net margin
  SPSA is DEPRIORITIZED. The binding question is now H1 vs H3 — Probe A's
  decayed endpoints and Probe C's audit decide.
- **2026-07-15 18:00 — PROBE C COMPLETE: BOTH SOURCES PASS — no semantic
  defect, no quarantine.** `tools/nnue/audit_label_semantics.py` (400
  rows/source, seed 20260715, FULL-file coverage on both corpora — fishpack
  sampled via chunk-header skip-scan across all 5477 chunks, one entry per
  selected chunk; lichess evenly-spaced record indices, stride 663152 —
  fresh SF18 `go depth 12` oracle, Threads=1, hash cleared per position, 2
  nice-10 workers; corpus sha256s re-verified == ledger rows; outputs
  `data/nnue/audits/label-semantics-v1/{lichess,fishpack}{.json,-summary.md}`).
  **Lichess train.bin PASS:** all 400 records decode white-to-move canonical
  (bulletformat flip verified: kings match ksq/opp_ksq^56; 0 structural
  faults); result byte uniformly 1 (exporter 0.5 placeholder, WDL=0 rule
  intact); core Pearson 0.946 / Spearman 0.974, sign agreement 96.9% (all 11
  disagreements |cp|<80 near-equality noise); OLS stored = 1.284·SF18cp + 5.0
  (R² 0.89) — stored cp ~28% hotter than depth-12 SF18-UCI, a calibration
  note (deeper heterogeneous browser-SF sources), not a defect; no mates,
  |stored| ≤ 1679 (2000 export cap intact). **fishpack32 PASS:** stm
  209w/191b; POV decisively stm-relative (core Pearson 0.948 stm vs −0.044
  white); sign agreement 99.7% (364/365); OLS stored = 3.026·SF18cp + 1.3
  (R² 0.90) — labels are SF *internal* units (~3x UCI cp, normalization ≈3.5
  attenuated by oracle noise); result −1/0/+1 = 90/232/78 stm-POV, coherent
  with scores (bucket mean stored −499/+21/+488). Oracle regime: 0/800
  bound-carrying finals, 0 timeouts, median 0.04 s / ~20-26k nodes per
  position. **Consequence: H3-as-semantic-defect is dead; both corpora mean
  what training assumed (mover/stm-POV, sign-correct, projection intact).
  H3 survives only as heterogeneity/scale (lichess 1.28x, fishpack 3.0x
  SF18-UCI-cp — per-source trainer scale/σ must match). Probe A/B
  conclusions on these sources stand; H1 is now the last hypothesis
  standing pending Probe A's decayed endpoints.**

- **2026-07-15 18:07 — PROBE A CELL 1 COMPLETE: fishpack classic256 @ 16x =
  −328 ± 55 (driver screen, fully decayed SB480) vs 1x anchor −163.** Full
  watcher curve (80g @5k, seed 20260715): SB30..270 flat in a noisy
  −407..−536 band at full LR (one −881 tail excursion at SB210, recovered;
  semantic probes sane throughout, no pack corruption; trainer loss smooth
  0.0154→0.0146 — loss saw nothing); late decay lifted it only to
  −269/−348/−328 at SB390/420/450. **Verdict: 16x presentations made this
  cell ~165 Elo WORSE than 1x. Pure H1 is DEAD for fishpack classic256.**
  Best explanation: H3b — more presentations fit the ~3x over-sharp targets
  more faithfully (semantic probes confirm amplitude shrinking with
  training: +N probe 405→342 cp from SB90→SB210). Note loss-vs-strength
  dissociation again: lowest running loss coincided with the −881 screen.
  Cell 2 (fishpack 1024) prediction under H3b: also fails to beat its −215
  anchor. Cell 3 (Lichess 1024, near-calibrated 1.28x labels) is now the
  decisive H1 test: if IT improves markedly with 16x while fishpack cells
  regress, the joint diagnosis is "H1 holds only with calibrated targets;
  fishpack needs the A2 rescale."

- **2026-07-15 18:40 — DIAGNOSIS CLOSED BY USER DECISION; probes terminated
  mid-cell-2.** User: remaining results (cell 2/3, A2) would not change the
  next action. All probe processes killed cleanly (campaign driver, cell-2
  trainer at ~SB300+, watcher, A2 runner, monitors); cell 2/3 and the A2
  anchor reruns are CANCELLED, their preregistrations void. **Final
  synthesis — why we struggled to train good nets:**
  1. NOT the pipeline (H4; seed-100 repro = 50.0%).
  2. NOT the search consumer (H2; Probe B: no coupling signature — every
     bulk net loses even at 32-node near-eval-only budgets).
  3. NOT a label-semantics defect (Probe C: both corpora byte-verified
     against pinned SF18; POV/sign/canonicalization exact).
  4. **Fishpack: target-scale miscalibration (H3b).** Stored scores are SF
     internal units ≈ 3.0x UCI cp; trainer sigmoid scale 400 assumes UCI cp
     ⇒ ~3x over-sharp targets. Probe A cell 1: 16x presentations made it
     WORSE (−328 vs −163) — faithful fitting of miscalibrated targets.
  5. **Scale alone fixes nothing when targets are miscalibrated; scale with
     calibrated targets remains the live (untested-at-endpoint) H1 lever**,
     with Lichess classic1024's −39 @20k off a 12-min run as the best
     supporting evidence.
  Actionable recipe going forward: calibrated targets (per-source score
  scale) + real outcomes/WDL + real schedule (multi-billion presentations)
  + capacity, screened with the local-coherence gate and multi-budget games.

### Production track (started 2026-07-15 18:40): prod-fishpack-cal-v1

The diagnosis phase is over; the GPU now trains PRODUCTION candidates.
Campaign `data/nnue/campaigns/prod-fishpack-cal-v1/`, bullet patched with the
env-gated `SFBINPACK_SCORE_DIV` (tools/nnue/patches/bullet-sfbinpack-score-div.patch):

1. `prod-fishpack-cal-v1-classic256`: fishpack, score/3.5, WDL 0.3, 480 SB
   full decay, seed 198273612 — the fast (~20 min) calibration read; its
   endpoint vs the miscalibrated anchors (−163 @1x, −328 @16x) measures H3b
   directly at scale.
2. `prod-fishpack-cal-v1-classic1024`: same recipe at width 1024 — the first
   real candidate. Endpoint screen 200g @5k vs net-0; if ≥ −50, run the
   2k/5k/20k curve + coherence audit before any promotion talk.

- **2026-07-15 19:05 — prod classic256 (÷3.5) endpoint = −470 ± 76: naive
  H3b calibration REFUTED; refined to H3b′ (consumer-amplitude matching).**
  The ÷3.5 "UCI calibration" overshot: the consumer's proven operating point
  is net-0's own (very hot) output distribution, not abstract UCI cp.
  Amplitude probes (+N, white knight up): net-0 = 1047, raw-fishpack-480SB
  net = 342 (screened −328), ÷3.5 net = 242 (screened −470) — strength has
  been MONOTONIC in output amplitude across all three. Fishpack labels are
  hotter than UCI but net-0's learned outputs are hotter still; dividing by
  3.5 made the coldest net yet. Test in flight: pack-time output rescale of
  the ÷3.5 net at k=1.75/2.33 (clip clamps 26/34 of 512 out-weights —
  partial compensation only), 80g @5k each. If Elo recovers toward −163,
  amplitude is causal and the durable fix is a training-side divisor ≈
  1.5–2.0 (or none) chosen to match net-0's output std, not label units.
  The classic1024 (÷3.5) cell continues training — its learned function is
  expected repackable/rescalable the same way.

## Resume instructions

1. Read this file + the takeover checkpoint of `policy-automl-state.md`.
2. Probe A state: `data/nnue/campaigns/value-scale-probe-a-v1/state.json` +
   watcher outputs under `.../matches/<net_id>/curve-sb*`. Do NOT restart an
   interrupted SF-binpack training run (resume unsafe) — record the
   interruption, keep completed checkpoints, and re-plan.
3. Probe B outputs: `data/nnue/campaigns/probe-b-ladder-v1/`.
4. Probe C outputs: `data/nnue/audits/label-semantics-v1*`.
5. After A–C: write the synthesis in Results log, update H1–H4 posteriors,
   pick the fork (scale / retune-consumer / label-quality), report to user.
