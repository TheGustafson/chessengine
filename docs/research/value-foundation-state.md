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

- **2026-07-15 19:35 — Thesis corrected (user challenge): asymmetric
  temperature tolerance, NOT co-adaptation, NOT a magic dataset.** Nothing
  was ever fit to sf-100: search margins are PSQT-era (P9), training
  constants are bullet defaults. The consistent story: (a) the consumer is
  HOT-TOLERANT (net-0 runs 2-3x hotter than the PSQT scale the margins were
  tuned on and won +696) but COLD-FRAGILE (over-pruning when eval swings sit
  inside fixed margins; k=1.75 output rescue bought +170 tonight); (b) eval
  temperature is set by data statistics under identical defaults — sf-100
  measured at **32.4% draws, |cp| median 168, std 468** (decisive-rich → hot
  net) vs fishpack streamed at ~60% draws (→ cold net, and amplitude decays
  the longer you train). Prediction: fishpack normalized into sf-100's
  marginals trains hot and competitive. Test = v2 ladder (div2.0 alone;
  div2.0+skip75) then v3 (`prod-fishpack-cal-v3-sf100frame`: div2.0 + skip75
  + draw-keep 32%), 30 SB vs the −163 anchor, auto-escalating to 480 SB +
  200g screen if ≥ −100. New trainer knobs: `NET0_RANDOM_SKIP_PCT`,
  `NET0_DRAW_KEEP_PCT` (both env-gated, default inert).

### Active experiments (preregistered 2026-07-15 ~20:05, post-reset)

**E1 — positions vs labels (user-designed).** Question: is fishpack held back
by its positions or its scores? Method: 5.93M fishpack positions (evenly
sampled across all 5,477 chunks, per-game cap via stride), relabeled with
SF16 using gen_sf_data.py's EXACT settings (engines/stockfish-datagen,
Threads=1, Hash=16, nodes=5000, White-POV UCI cp, quiet/check/mate/cap
filters, dedup; aspiration bounds intentionally NOT filtered — sf-100 itself
was made this way). Fishpack's own game results kept for the WDL 0.3 blend.
Then train with the exact net-0 recipe and screen ≥300g vs net-0.
OUTCOMES: works → positions fine, original fishpack scores (or their
consumption) were the problem, and this pipeline = unlimited sf-100-style
data; fails → scores were never the problem, suspect stream/loader
mechanics. Tools: sfbinpack-audit/src/bin/bulk_fens.rs,
tools/nnue/relabel_fishpack_sf16.py. Running: ~1,300 pos/s, ETA ~75 min.
Output: data/nnue/experiments/fishpack-sf16-relabel/.

**E2 — 4-net architecture matrix on sf-100 (user-directed).** Question: was
"bigger net on the known-good data" ever given a fair shot? (No: 512 got one
unreliable cross-binary +16; 1024 was schedule-starved at 747M.) Method:
classic256 (60 SB), classic1024 / king768x256 / d40-full (240 SB each ≈ 6B
presentations, 8x the old budget), net-0 recipe otherwise, WDL 0.3,
sf-100 = the exact data/nnue/train.bin artifact. Decision-grade 300g @5k
endpoint screens; 400g @20k for anything above −50. Checkpoints every 30 SB
retained for convergence inspection. Per user rule: if cells converge badly
on the KNOWN-GOOD data with generous schedules, the training method itself
goes under review. Campaign: data/nnue/campaigns/sf100-matrix-v1/.

- **2026-07-15 20:55 — E2 partials (decision-grade, 300g @5k vs net-0).**
  [OBSERVATION] classic256 @60SB: **+6.9 ± 18.0** (parity; 400g @20k curve
  −0.9) — instrument + recipe reproduce parity on the good data; 2x schedule
  neither helps nor hurts. classic1024 @240SB (6B presentations, 8x the old
  budget): **−68.0 ± 18.9** — did not reach the −50 curve floor.
  [HYPOTHESIS] 25M positions cannot feed 4x capacity (memorization/underfeed),
  and/or the single-cosine schedule shape remains wrong for width; either
  way, capacity is NOT a free win on sf-100 — more data (E1/E3 pipelines) is
  the prerequisite for capacity, consistent with official practice (stage-1
  corpora are billions of positions). king768x256 and d40 cells pending.
- **2026-07-15 20:52 — UHO prefix label audit (200 rows vs SF18 d12):
  PASS.** [OBSERVATION] stm-POV (opposite-POV r = −0.007), sign 98.25%, core
  Pearson 0.970, OLS slope **2.82x SF18-UCI** (internal-units family, like
  fishpack's 3.03). Draws 50.5%. Score std ≈ 883 internal ≈ 313 UCI —
  **matching sf-100's ≈ 312 UCI-equivalent** (the stage-1 family shares a
  temperature signature; fishpack at ~60% draws is the cold outlier).
  Report: data/nnue/audits/uho-label-semantics-v1/.

### E3 (preregistered 2026-07-15 20:55): data transplant — net-0's exact
recipe on the official stage-1 corpus

Question: does our full stack train a strong net from the LITERAL data family
official SF bootstraps on? Method: `nodes5000pv2_UHO.binpack` 10GiB prefix
(~4B+ positions), net-0's exact recipe (classic256, 30 SB, WDL 0.3, sigmoid
400) with exactly two declared adaptations: `SFBINPACK_SCORE_DIV=1.9` (maps
UHO's 2.82x units onto sf-100's SF16 frame ≈ 1.5x, where the recipe is
proven) and `NET0_RANDOM_SKIP_PCT=75` (their own rfs=3; approximates sf-100's
global pre-shuffle on a game-contiguous stream). 300g @5k screen vs net-0.
OUTCOMES: ≈ net-0 → trainer/recipe exonerated end-to-end on external data;
the whole official data lineage is usable and scaling (schedule, then
capacity, then stage-2 warm retrains) is unblocked. Clearly below → the
defect is on our side of the interface and now tightly localized (loss
shape/quantization/eval integration), since data, labels, units, and recipe
are all individually proven. Runs after E1 (GPU serial).

- **2026-07-15 23:40 — E2 d40 cell: parity gate FAILED; screened manually
  with caveat.** [OBSERVATION] After 240 SB on sf-100 (running loss
  0.000633), the packed CENN's runtime-vs-trainer max error was 0.355 logits
  (gate 0.2; mean 0.0124 passes; quantization passes). ~5x worse max
  deviation than the lichess-trained D40 packs (0.074). The matrix runner
  died on the gate; chain released manually, 300g screen running on the
  packed net with this deviation on record. [HYPOTHESIS] very low loss /
  large weights → saturation amplifying pack rounding; not investigated yet.
  E1 starts now (GPU free); ETAs: d40 screen ~00:05, E1 ~00:30, E3 ~00:55,
  size control ~01:10, E4 ~02:00.

- **2026-07-16 06:45 — overnight chain died at E3; cause found and fixed.**
  [OBSERVATION] E3's second run crashed at SB 28/30: bullet's
  SfBinpackLoader panics (`UnexpectedEof`, sfbinpack.rs:112 unwrap) on a
  binpack whose last chunk is byte-truncated — our UHO file was a raw 10GiB
  `head` prefix, so its tail ends mid-chunk. Clean EOF at a chunk boundary
  is handled (loader wraps for the next epoch); ragged EOF panics. Fix:
  truncated the file in place at the last complete `BINP` chunk boundary
  (10,736,935,551 bytes, 10,273 chunks, 482,689 tail bytes dropped). All
  downstream steps (size control, E4, E1 rescreen, D40 rescreen) were
  gate-chained behind E3 and never ran; relaunched everything as ONE serial
  script with no gate files: `tools/nnue/chain_v2.sh`
  (log `data/nnue/experiments/chain-v2.log`, results
  `data/nnue/experiments/chain-v2-results.log`). Ops rule reaffirmed: any
  byte-truncated binpack must be re-truncated to a chunk boundary before
  training.

- **2026-07-16 06:52 — E1 rescreen reproduces −463 exactly: the original
  was NOT contaminated.** [OBSERVATION] Fresh 300g @5k with a different
  fastchess seed (20260716): score 0.065, **−463.2 ± 38.7** — same to the
  decimal as the pkill-suspected run. E1 verdict stands: fishpack positions
  + SF16 5000-node labels + fishpack results, exact net-0 recipe, 4.06M
  rows → far below net-0. Also learned: these screens take ~20 s, not
  ~25 min (300 games @5k nodes, conc 4, classic256) — screening was never
  the bottleneck we scheduled around. Interpretation deferred to the size
  control (4M sf-100 rows, identical recipe): if it also craters, E1's
  deficit is mostly corpus size; if it holds near net-0, fishpack positions
  (or their fishpack-outcome WDL component) are the problem.

- **2026-07-16 06:54 — E3 RESULT: first net ever to beat net-0.**
  [OBSERVATION] `e3-uho-transplant-classic256` (net-0 exact recipe on the
  official stage-1 `nodes5000pv2_UHO` 10GiB prefix, SCORE_DIV 1.9,
  skip75, 30 SB ≈ 747M presentations ≈ one pass): 300g @5k vs net-0 =
  **+47.8 ± 18.7**, paired-bootstrap CI95 [+11.6, +85.0], score 0.568.
  Same architecture and engine as net-0 → no NPS tax; fixed-node ≈
  equal-time for this pair (equal-time gate still required before any
  promotion claim). [HYPOTHESIS → now well-supported] The stack (trainer,
  units bridge, filters, pack, engine integration) is exonerated
  end-to-end; the binding constraint was DATA. sf-100 (25M) was the best
  corpus we had; the official stage-1 family at 10GiB immediately clears
  it. Unblocked next steps, in order of expected value: (a) equal-time
  gate for e3 vs net-0; (b) scale: more UHO data (we hold the manifest
  for 287GB), longer schedules; (c) capacity on the bigger data
  (classic1024 failed on 25M — retry on UHO); (d) stage-2-style warm
  retrains on rich data from the e3/UHO base.

- **2026-07-16 06:55 — E1 size control: −189.3 ± 21.5. Positions-vs-labels
  decomposition complete.** [OBSERVATION] sf-100 truncated to E1's exact
  corpus size (4,064,476 rows, identical net-0 recipe/schedule):
  **−189.3 ± 21.5** vs net-0. Ladder at matched recipe: sf-100 25M ≈ 0
  (E2 classic256 +6.9 ± 18.0) → sf-100 4M = −189 ± 22 → fishpack-pos +
  SF16-labels 4M (E1) = −463 ± 39. [INTERPRETATION] Two real, separable
  effects: (1) corpus SIZE costs ~190 Elo at 4M vs 25M under this schedule
  (747M presentations over 4M rows = ~187 epochs of repetition); (2) with
  size and labels held fixed, fishpack positions + fishpack outcomes cost
  a further ~270 Elo vs SF self-play positions. The user's question is
  answered: it is BOTH, and neither is a units/pipeline artifact —
  position provenance matters at matched everything-else, and data volume
  matters steeply below ~10M rows. Consistent with E3: the winning move
  is more SF-self-play-shaped data, not fishpack repair.

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
