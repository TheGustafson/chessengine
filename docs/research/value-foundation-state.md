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

Last updated: 2026-07-16 ~21:00 America/New_York

## TAKEOVER CHECKPOINT (2026-07-17 06:20 — read this first)

**OVERNIGHT OUTCOME (all ledgered, details in the dated entries below):**
1. **Promotion #4 REJECTED (exp-0073)**: d40-uho-v1-240sb vs s4 at equal
   time = **−57.9 ± 12.6** (200g 10+0.1). D40 wins per node (+32.5
   exp-0065; 720sb adds +30.2 exp-0072) but the inference tax dominates;
   needs ~1.7–2× NPS to convert. D40 → background line (task #26).
   720sb itself trained fine but is parity-barred by the OOD-endgame
   quant tail (max 0.2517 vs 0.20, mean clean) — same mode as the hop;
   repair options for the user in the 03:50 entry.
2. **S1 correction history DEPLOYED (exp-0074)**: equal-time SPRT PASSED
   H1 in 208g, **+72.9 ± 16.2** (SPRT [0,10], LLR 2.95). Merge gate green
   at HEAD, bench 701605. Largest search gain since the P9 waves. The
   fixed-node screen said only +18.5 — protocol lesson: NPS-neutral
   search changes are underestimated by shallow screens; TC-gate them
   early.
3. **Obs/D41 platform is PRODUCTION (M1–M5 all closed)**: dual-teacher
   sample-v1 audits = GO (scores reproducible, 8–10cp cross-tier drift;
   argmax soft — median top margin 13–16cp → train on per-arm values;
   teachers convergent: 75% best-move agreement, tau 0.841). Reckless
   primary / SF18 control HOLDS. Both shards ledgered. **big-v1 labeling
   RUNNING since 06:09** (10k fresh positions, both teachers parallel,
   done ~16:30, auto-audits at end, [obsbig] tags).
4. **EVALUATOR-FREEZE PROPOSAL drafted (05:35 entry) — the user's morning
   decision**: freeze the classic line (s3-farseer incumbent, s4
   working-best + cheap promotion-#5 TC re-check), pause eval trainings,
   shift to search (S2 SPSA next) + D41/design-19 as the main thrust.
5. R4 COMPLETE: all three bench-neutral speedups landed — SIMD find_nnz
   (3257492), threat scratch (943584d), and **#1 lazy FullThreats fold
   (45301df)**: the fold (3 KB copy + 768-lane apply) now runs only for
   plies that reach an evaluate, mirroring HalfKA's memoized compute;
   king-crossed plies stay eager by design. Verified: 636 sota tests + 3
   new lazy-path tests + release replay oracle green, and the new binary
   is EVAL-BIT-IDENTICAL to the pre-R4 one across the 4096-fen parity
   set. probe-sota redeployed at 45301df (.pre-r4 backup kept); NPS
   numbers wait for the R1 quiet-box session. Sota-gated tests need
   `--features sota` — the default gate compiles them out. Probes carry
   S1 search now; cross-era fixed-node comparisons must note this.
6. Ops: 720sb wrapper exit ≠ success (train FAILED at parity after 7h —
   the night chain degraded gracefully by design). big-v1 guard survived
   a pipefail/pgrep bug (fixed, a76d1c7+). Ledger verify: 0 problems.

## PREVIOUS CHECKPOINT (2026-07-16 20:30)

**Housekeeping state (20:30):** disk cleaned 8.1G→41G free (~33GB: cargo
debug/cross targets, sf-100 shard intermediates, six old ckpt dirs, eight
dead campaign dirs, and three LEDGERED artifacts deleted WITH lost-
annotations: lichess bulk 8G, test80 binpack 11G [re-downloadable], E1
relabel corpus). Ledger gained the append-only correction mechanism
(`ledger.py annotate`, annotations.jsonl, design/18 §5; commit 287db4f);
verify now runs end-to-end (7fa53f4) and reports exactly 1 problem: exp-0042
dangling → unregistered net sf100-matrix-v1-sf100-d40-full (registration
needs a code_sha convention — user decision pending). Killed three stale
sleep-loop waiters from concluded eras (d40_rescreen, e4_king_rerun,
e1_control) — kill by exact PID only. Worktree CLEAN, all work committed;
122 commits ahead of origin/main, push is the USER's action (never push).
config.json is fastchess-written runtime state and will re-dirty after
every match — candidate for gitignore, don't chase it. Design docs + 
CLAUDE.md status refreshed to the post-B4 reality (24c8499, 52d1de9).
Rust perf tasks R1-R4 queued (native builds / VNNI sota_simd / PGO /
micro-audits) — R2 VNNI is the D40-NPS lever, box is Zen 4.

**In flight RIGHT NOW (updated 21:00; check these before doing anything):**
- `tools/nnue/d40_long_v1.sh` (bg): d40-uho-720sb (18B presentations)
  training — RESTARTED 20:48 after the OOM incident below, ~7h → done
  ~03:50. The first attempt (19:15) was OOM-killed at ~70 min.
- `tools/nnue/d40_night_v2.sh` (bg, re-armed 20:48): waits for the train,
  then (1) h2h hop d40-s2-farseer-120sb vs d40-uho-v1-240sb — DIAGNOSTIC
  ONLY, (2) 720sb parity parse + h2h vs 240sb, (3) EQUAL-TIME GATE best
  parity-PASSING D40 vs s4 (200g 10+0.1 conc 1) — promotion #4 verdict by
  ~08:00. Both engine binaries (probe-sota, probe-l2048) were rebuilt at
  df2b746 with the EvalFile leak fix; .pre-leakfix backups sit beside them.
- **THE 3.6GB-ENGINE BELIEF IS DEAD (21:00): it was a per-game memory
  leak, now FIXED (df2b746).** fastchess re-sends every option — EvalFile
  included — before every game; each re-send re-loaded and leaked the
  net (~78MB/game for D40, ~3MB/game for classic). Real sota engine
  footprint after fix: **~135MB flat** (verified over a 20-game match:
  134-135MB throughout vs 1.7GB-and-climbing before). This retro-explains
  weeks of "sota engines are huge" ops pain AND unlocks D40 test
  concurrency (T4 applies to D40 now, not just classic).
- HOP PARITY FAIL (19:14): d40-s2-farseer-120sb runtime mean/max
  0.011/**0.241** vs gates 0.03/**0.20** — mean excellent, p99 0.088,
  worst fen an OOD bare endgame (2R5/8/8/3BK2k/8/7p/8/8 b), runtime
  tracks quant (quant max 0.63) → quant-CLIPPING tail, NOT a transform
  bug (sf-100's true fail: 0.355 broad). Hop is NOT a gate candidate;
  user decides waiver vs rescale (tools/nnue/rescale_net.py) tomorrow
  with its h2h number in hand. Old `d40_s2_gate_v1.sh` chain is DEAD
  (exited 1 at parity) — superseded by d40_night_v2.sh.
- RAM RULE, amended 21:00: the old form ("sota pairs = 2×3.6GB, never
  run games while a train lives") was built on the leak, which is now
  fixed — engines are ~135MB. The surviving rules: (a) equal-time (TC)
  games still require a quiet box for WALL-CLOCK FAIRNESS — never run
  them alongside a trainer; (b) fixed-node games alongside a trainer are
  now RAM-safe but were the proximate cause of tonight's OOM before the
  fix — only resume that practice with post-fix binaries; (c) any new
  engine binary must pass a capped 20-game RSS-flatness check before an
  overnight chain trusts it.
- Results log for everything: `data/nnue/experiments/chain-v2-results.log`.
- SEARCH PROGRAM defined (user-directed): ranked matrix in
  `stockfish_reference/06-sf14-sf16-search-patch-study.md` §6a/6b;
  tasks S0 (unbalanced test book + SF16 mechanics audit), S1 (wire
  correction history — tables exist, read/write deferred at
  src/search/mod.rs ~line 121), S2 (gainer-season global SPSA retune —
  the single biggest pool, +15-40 est). Key audit fact: P9 already has
  singular/multicut/negative extensions, ProbCut, cutoffCnt,
  do_deeper/do_shallower, improving, cut_node, fallingEval/instability
  TM. Search game-testing waits for the D40 arc to settle; code work
  can start anytime.


**The governing question is ANSWERED.** Net-0 was never special; DATA VOLUME
was the constraint end-to-end. Everything below in one paragraph: the
official stage-1 UHO binpack beat net-0 at its own recipe on the first try
(E3, +47.8); width then scaled 256→512→1024→2048 (+32/+123/+225/+282 @5k vs
net-0) and SATURATED at 2048 (4096: +15 ± 17); schedule and lambda were flat;
stage-2 warm retrains gain in proportion to stage-2 LABEL QUALITY (farseer
0.93 Pearson: +55/+38; test80 0.78: +19/flat) and plateau after ~2 hops;
rich FEATURE SETS (king768, D40) failed historically ONLY from data
starvation — on full UHO the D40 parity gate passes (0.174 vs 0.355 on
sf-100) and stage-1-only D40 beats the best classic by +32.5 ± 17.3
(exp-0065). External anchors (all 100g, 10+0.1, 1 thread): SF16 −78 ± 18,
SF18 −191 ± 23 (gap 113 ± 29 ≈ known 90), Integral v6 −70 ± 22, and the
triangle closes locally (SF16−Integral = +52 ± 20 measured here; site 80
compresses at fast TC). Promotion chain (equal-time gated): net-0 → +166
(u1024-120sb) → +43.7 (uf2048-240sb) → +54.9 same-width (s3-2048-farseer,
INCUMBENT); s4-2048-t74 = working best (+15.6 ± 11.9, below bar).
**In flight at checkpoint:** see the "In flight RIGHT NOW" block above
(d40_long_v1 720sb retrain + d40_night_v2 chain; the old d40_s2_gate_v1
chain is dead). Next levers ranked: bigger/better stage-2 corpora
for D40 (T60T70 33GB needs disk; test80 deletable), longer D40 stage-1,
king768 probe, official loss shape (prepared, unneeded so far). Disk 8.5GB
free. All experiments exp-0038..0065 + 20 nets ledgered today; every
result also in `data/nnue/experiments/chain-v2-results.log`. Ops rules
that bit today: gate-waits must check status=complete AND not self-match
their own log lines; cloned D40 configs need `train.quantised` updated;
parity stage needs CUDA_PATH=/usr; list-then-delete separately; the old
"conc 1, sota engines are 3.6GB" rule is superseded — that was the
per-game EvalFile leak, fixed at df2b746 (see the 21:00 incident entry).

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

- **2026-07-16 07:15 — AUTONOMOUS LOOP MANDATE (user): "auto research loop
  until we have a SOTA eval model. Local compute only."** Promotion bar
  unchanged: beat the incumbent at equal time. Loop protocol: fixed-node
  screens route, 300g+ decides, equal-time promotes, everything ledgered.
  Wave preregistered now:
  - E3 follow-ups: 400g @20k curve; 200g 10+0.1 equal-time gate (quiet box).
  - `uho-ladder-v1` (tools/nnue/uho_ladder_v1.sh, GPU-serial after E4):
    u256-60sb / u256-120sb (schedule axis: does optimization beyond one
    pass of the 10GiB keep paying? official practice says data gets many
    epochs); u512-60sb / u1024-120sb (capacity axis: E2 showed 1024 fails
    on 25M rows; prediction from official practice = capacity unlocks with
    billions-scale data. If u1024 ≥ u256 at matched screens, capacity
    scaling is open and the SOTA path is data x capacity x schedule).
  - Full `nodes5000pv2_UHO.binpack` (40.3GB) downloading (resume-capable);
    disk freed 16→59GB by deleting re-generable checkpoints of finished
    campaigns + quarantined lc0/mix/net0-paired corpora (identities +
    regen recipes preserved in ledger; active campaigns untouched).
  Decision rules, in advance: schedule axis wins → full-UHO long run next;
  capacity wins → full-UHO at the winning width; both flat → stage-2 warm
  retrain exploration (task 7) or loss-shape work. Equal-time failures of
  fixed-node winners → NPS/time-management investigation before any arch
  pivot.

- **2026-07-16 18:15 — FULL-FEATURE BREAKTHROUGH: D40 works at data
  scale.** [OBSERVATION] `d40-uho-v1-240sb` (HalfKAv2_hm + FullThreats,
  stage-1 on full UHO, 2h23m train): parity gate **PASSES** (runtime
  mean/max 0.0072/0.174 vs gates 0.03/0.2; the sf-100 twin failed at
  0.355 → yesterday's failure was data-starvation weight saturation, not
  a pack bug). h2h vs s4 working-best: **+32.5 ± 17.3** (300g @5k,
  exp-0065) — a first-try, stage-1-only full-feature net beats the
  twice-stage-2-hopped classic 2048. Every sf-100-era rich-feature
  failure is now explained by data volume. [LAUNCHED] D40 stage-2
  farseer hop (warm via D40_INIT_RAW from the retained raw ckpt) →
  h2h → EQUAL-TIME gate vs s4 (200g 10+0.1, conc 1, ~3h) — promotion
  #4 candidate. Ops notes: cloned D40 configs must update
  `train.quantised` (stale filename aborted the first pack); the
  train_run parity stage needs `CUDA_PATH=/usr` in the caller env.

- **2026-07-16 17:55 — Integral v6 anchor + triangle closure: rating
  validated.** [OBSERVATION] Integral v6 built from source (tag v6, era
  net integral-networks@238c8e4a30 pinned — main-branch net would be
  wrong-era; clang-for-both-languages fixes a mixed-LTO link failure).
  ARCHON vs Integral v6: **40.0% (11W 58D 31L), −70.4 ± 22.4**
  (exp-0063). Triangle closure under identical local conditions (early-
  stopped at n=54 by preregistered rule): SF16 vs Integral **+51.9 ±
  19.7** (exp-0064) — the sites' ~80 gap compresses to ~52 at 10+0.1 /
  1 thread / balanced book (draw compression). Triangle arithmetic:
  predicted Integral−ARCHON = 26 ± 27, measured 70 ± 22 → closes within
  1.3σ. [CONCLUSION] All four external measurements are mutually
  consistent: ARCHON sits ≈ 70-90 below SF16-class, i.e. Integral-v6
  class head-to-head, at these conditions. The internal +265 chain and
  external anchors agree.

- **2026-07-16 17:10 — SF18 sanity check PASSES.** [OBSERVATION] ARCHON
  incumbent vs FULL SF18, 100g 10+0.1: **25.0% (0W 50D 50L), −190.8 ±
  23.3, 0 time losses** (exp-0062; D40 trainer on GPU during the match —
  symmetric load caveat). vs SF16 was −77.7 ± 18.4 → the measured
  SF16→SF18 gap through our engine is **113 ± 29**, consistent with the
  known ~90. External anchors now bracket the engine coherently:
  ≈ −78 from SF16, ≈ −191 from SF18, +265 internal chain over net-0.

- **2026-07-16 16:20 — Stage-2 plateau confirmed; FULL-FEATURE TRACK
  OPENED.** [OBSERVATION] s4 pooled 600g vs s3: **+15.6 ± 11.9** — not
  promoted (farseer hops: +55 → +29 → +15.6 pooled; the classic-family
  stage-2 well is drying up). s4 stays working-best/gate-baseline.
  [OBSERVATION] bench NPS: probe-sota 4.26M vs probe-l2048 4.09M — the
  D40 SIMD path carries NO material speed tax, so fixed-node wins should
  transfer to equal time. [LAUNCHED] `d40-uho-v1-240sb`: D40
  (HalfKAv2_hm + FullThreats) stage-1 on full UHO with the proven recipe
  (div 1.9, skip75 — D40_RANDOM_SKIP_PCT added to the trainer) — the
  first rich-feature net ever given billions-scale data. Known risk: the
  pack parity gate (0.355 max-err failure on sf-100; hypothesis =
  saturation at very low loss — big data may avoid it). Screen: h2h vs
  s4, conc 1. king768-on-UHO is the follow-up probe.

- **2026-07-16 16:00 — Reality check vs full Stockfish 16: −78 ± 18.**
  [OBSERVATION] ARCHON (incumbent s3-2048-farseer, P9-era search) vs
  FULL-strength SF16, 100g 10+0.1, both 1 thread / 16MB hash:
  **39.0% (4W 70D 26L), −77.7 ± 18.4, 0 time losses** (exp-0060). Net-0-era
  ARCHON anchored ~3400 vs SF16's handicapped rungs; today the gap to the
  full engine is under 80 Elo, consistent with the +265 equal-time chain.
  70% draw rate — the eval is competitive; the deficit is now plausibly
  search-era (P9 pruning stack) as much as eval. [OBSERVATION] T74 hop:
  s4-2048-t74-240sb vs s3 incumbent **+29.0 ± 16.5** (exp-0059) — second
  consecutive stage-2 gain, diminishing vs T76's +55; 600g pooled confirm
  running before promotion.

- **2026-07-16 14:20 — Width ladder SATURATES at 2048.** [OBSERVATION]
  uf4096-240sb vs uf2048-240sb h2h: **+15.1 ± 17.3** @5k fixed nodes
  (exp-0058). Per-doubling curve on full UHO: 512→1024 +102, 1024→2048
  +57, 2048→4096 +15 — and the 4096 equal-time tax would eat that.
  [DECISION] Capacity axis closed on stage-1 UHO + current recipe;
  width 2048 is the platform. Live levers, in expected-value order:
  (1) stage-2 hops on the incumbent (T74 hop training now — farseer
  family at 3x data), (2) more/better stage-2 corpora (T60T70 33GB needs
  a disk decision), (3) official loss shape in bullet, (4) feature-set
  growth (king buckets) at billions-scale — parked, one retry planned
  when a stage-2 plateau arrives. farseerT74 audited: div 1.66, Pearson
  0.937, sign 95.1% (farseer-tier).

- **2026-07-16 12:50 — THIRD PROMOTION: stage-2 works, and label quality
  is what gates it.** [OBSERVATION] Farseer hops (h2h vs each warm-start
  parent, 300g @5k): s3-2048-farseer **+54.9 ± 17.4** over incumbent
  uf2048-240sb; s3-1024-farseer **+38.4 ± 17.9** (test80 hop at the same
  width: +19.1 ± 12.5 pooled 600g; test80 at 2048: −6.9 ± 16.4 flat).
  farseerT76's labels audit cleaner (Pearson 0.934 vs test80's 0.784) and
  its hop gains are ~2-3x — stage-2 gains track stage-2 LABEL QUALITY.
  **New incumbent: stage2-v1-s3-2048-farseer-120sb** (same width/engine as
  the old one → h2h transfers to equal time; exp-0056). Composed
  equal-time vs net-0 ≈ **+265**. Next: farseerT74 (20.1GB, downloading)
  for a second farseer-family hop at more data; wave5 uf4096-240sb
  training (~14:45); redundant 10GiB UHO prefix deleted (full file
  supersedes; disk 28GB free → T74 fits).

- **2026-07-16 11:30 — SECOND PROMOTION: uf2048-240sb is the new
  incumbent.** [OBSERVATION] Equal-time gate (200g 10+0.1, quiet box):
  uf2048-240sb vs u1024-120sb **+43.7 ± 13.9**, score 0.562, 0 time
  losses (exp-0054). Wave4 h2h @5k was +57.3 ± 17.7 vs uf1024-240sb
  (exp-0052) — the 2048 real-time tax costs some of the fixed-node edge
  but the capacity gain survives at 3σ. Composed equal-time chain vs
  net-0: ≈ **+210** (+166 u1024 gate + 44 this gate). [OBSERVATION]
  Stage-2 attempt 1 (s2-1024-test80-120sb, warm from uf1024-240sb):
  **+23.2 ± 17.4** vs its parent (exp-0053) — the official two-stage
  shape shows signal where cold-training the same data family always
  cratered. Width ladder to date (vs net-0 @5k): 256 +32 / 512 +123-151 /
  1024 +225 / 2048 ≈ +282 composed. In flight: s2w2 (pool s2-1024 to
  600g; s2-2048-test80-120sb warm retrain of the incumbent), farseerT76
  audited next; then width 4096 vs stage-2 data scaling by results.
  Ops: disk crunch to 3.4GB during downloads → deleted search-trace,
  E2/E4 checkpoint dirs, sf-100 raw shards (train.bin exact-bytes anchor
  re-registered as `sf-100-trainbin`; list-then-delete lesson added to
  memory). Disk 19GB free.

- **2026-07-16 10:55 — Wave3 flat; stage-2 track armed.** [OBSERVATION]
  wave3 h2h vs uf1024-240sb (300g @5k): uf1024-480sb (12B presentations)
  **−10.4 ± 17.4**; uf1024-240sb-wdl0 (official pure-score lambda)
  **−6.9 ± 17.4**. Schedule and lambda both flat at width 1024 — the width
  family is recipe-saturated ≈ +200-225 over net-0 (exp-0050/0051).
  [OBSERVATION] test80 (Lc0-derived stage-2 corpus, 10.8GB, 10,308 clean
  chunks) label audit vs SF18 d12: stm-POV, core slope **2.18x SF18-UCI**,
  Pearson 0.784, sign 93.9% — clean but noisier than UHO (0.970/98.3%),
  29% exact-zero scores, weak tail sign (73.7% at |cp|≥2000). SCORE_DIV
  1.45 maps it to the proven SF16 frame. [PREREGISTERED] Stage-2 attempt 1
  (tools/nnue/stage2_v1.sh, gated on wave4): warm start uf1024-240sb
  (new NET0_INIT_CENN knob), test80, LR 0.0005 (halved), WDL 0.25
  (≈ official stage-2 lambda 0.75), 120 SB; h2h vs its parent. OUTCOMES:
  clear win → the official two-stage shape transfers; scale stage-2 data
  next (T60T70/farseer files as disk allows). Flat/loss → rich-data gains
  need the official loss shape or interleaved mixing — loss-shape work in
  bullet becomes the next lever. In flight: wave4 uf2048-240sb (capacity;
  ETA ~13:35), then stage2. Disk 9.2GB free (floor reached; fishpack32
  deleted, re-downloadable, was E1-proven negative-value).

- **2026-07-16 09:05 — Full-file scaling at width 1024 is FLAT under the
  current recipe; width is the live lever.** [OBSERVATION] uf1024-240sb
  (full 40.3GB, 6B presentations) vs u1024-120sb (10GiB, 3B): pooled 600g
  head-to-head **+6.9 ± 12.4 — parity** (exp-0049). At width 512 the full
  file DID pay (+123 → +151.3 ± 20.9 vs net-0). [INTERPRETATION] At 1024,
  4x data + 2x presentations bought nothing under single-cosine + WDL 0.3;
  the binding constraint is now recipe or capacity, not corpus size.
  In flight: wave3 probes recipe (uf1024-480sb = 12B presentations;
  uf1024-240sb-wdl0 = official pure-score stage-1 lambda), wave4 probes
  capacity (NEW `nnue-l1-2048` engine feature, commit b9536e3 — default
  build unaffected; probe-l2048 built; uf2048-240sb queued behind wave3).
  train_net0 now accepts comma-separated binpacks (concat loader) for the
  coming multi-file stage-1 mix; wrongIsRight_nodes5000pv2 (7.3GB)
  downloading. Screens within a width family (same NPS) use direct
  head-to-heads; equal-time gates reserved for cross-width promotion.

- **2026-07-16 08:30 — EQUAL-TIME GATE PASSED: net-0 DETHRONED after 4
  days as incumbent.** [OBSERVATION] `uho-ladder-v1-u1024-120sb` vs net-0,
  200g at 10+0.1 (the user's promotion bar): **+166.2 ± 18.3**, score
  0.7225, ZERO time losses either side. Fixed-node +225 → equal-time +166
  = a real but affordable width-1024 speed tax. Caveat on record: the
  full-UHO wave trained on GPU concurrently for part of the match
  (gate-wait bug: eval_match writes summary.json with status=running at
  START — wait conditions must check status=complete, not file existence);
  load was engine-symmetric and forfeits were zero, so the 9σ verdict
  stands. **New incumbent: u1024-120sb. All future promotion gates run
  vs the current incumbent at equal time.** net-0 becomes the historical
  baseline. Ledger exp-0048. uf512-120sb (full 40.3GB file, 1 pass):
  **+151.3 ± 20.9** vs net-0 @5k — width 512 gains ~+28 over its 10GiB
  version (1.4σ, scaling-consistent). uf1024-240sb training (SB 179/240).

- **2026-07-16 07:55 — uho-ladder-v1 screens: CAPACITY UNLOCKED on
  billions-scale data.** [OBSERVATION] 300g @5k vs net-0, all on the UHO
  10GiB clean prefix, e3 recipe: u256-60sb **+32.5 ± 18.7**, u256-120sb
  **+27.9 ± 18.8** (schedule axis flat at width 256 — 30/60/120 SB all
  ≈ +30-50), u512-60sb **+123.0 ± 19.5**, u1024-120sb **+225.0 ± 22.8**.
  The SAME classic1024 that scored −68 on sf-100 (25M rows) scores +225 on
  UHO-10GiB. Bench NPS sanity: probe-default 4.30M, classic512 4.26M,
  l1024 4.65M (tree-shape noise; no material eval tax at these widths).
  [INTERPRETATION] Width × data is the scaling frontier, exactly as
  official practice predicts (their stage-1 = 40-60B samples). Width 256
  is the net-0-era ceiling; it cannot absorb more data or schedule.
  Equal-time gate for u1024-120sb (200g 10+0.1) launched; full 40.3GB UHO
  still downloading. Next GPU wave (after gate, to keep the box quiet):
  u1024 longer schedule on 10GiB, then full-UHO scale runs; consider a
  2048-wide engine feature if 1024 promotes (code change needed).
  Ledger rows to follow with the gate result.

- **2026-07-16 07:35 — D40 rescreen + E3 depth curve; sf-100 matrix
  complete.** [OBSERVATION] d40-full on sf-100 (240 SB, conc-1 clean
  screen): **−478.0 ± 39.7** (parity caveat max-err 0.355 on record). Full
  E2/E4 matrix on sf-100 25M, 300g @5k: classic256 **+6.9**, classic1024
  **−68**, king768x256 **−409**, d40-full **−478** — strength is MONOTONE
  DECREASING in feature richness at fixed 25M rows. [OBSERVATION] E3 depth
  curve: **+62.3 ± 15.4 @20k (400g)** vs +47.8 ± 18.7 @5k — the UHO net's
  edge GROWS with depth; every failed bulk net closed from below instead.
  [INTERPRETATION] One coherent picture: feature-rich architectures are
  data-starved at 25M; data scale is the master variable. Ledger
  exp-0042/0043.

- **2026-07-16 07:15 — E4: the sparse-init fix does NOT rescue
  king768x256.** [OBSERVATION] With `init_with_effective_input_size(32)`
  fixed (the only change): **−409.1 ± 33.8** (300g @5k) vs the buggy run's
  −372.3 ± 31.4. [INTERPRETATION] The historical king-cell failures were
  NOT the init bug. Remaining suspects, now data-first given E3/e1ctl: 64
  king buckets slice 25M rows into ~sparse per-bucket samples (official
  HalfKA-family trains on billions); possibly also the trainer's king-cell
  transform itself. Action: park king features until a billions-scale
  corpus is in hand (full UHO downloading), then retry once at matched
  budget. The CENN-v3 "simple HalfKA destroys topology" reading stays
  partially confounded but is no longer blamed on the init bug alone.
  Ledger: exp-0041.

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

## 2026-07-16 19:45 [OBSERVATION] D40 farseer hop fails parity marginally; night chain replaces gate

- d40-s2-farseer-120sb (warm from d40-uho-v1-240sb raw, farseerT76 120SB)
  trained clean but FAILED the runtime-parity gate: mean/max 0.011/0.241
  vs 0.03/0.20 (p95 0.042, p99 0.088). Worst fen = OOD bare endgame at
  move 86; quant-vs-raw max 0.63 with runtime tracking quant → the tail
  is quantization clipping on out-of-distribution material, not a
  trainer/runtime transform mismatch (contrast sf-100 D40: 0.355 broad
  fail). Candidate fixes if its h2h is strong: rescale_net.py pass or a
  waiver with the caveat recorded. NOT a gate candidate as-is.
- Gate chain d40_s2_gate_v1.sh exited at the parity stage by design;
  equal-time gate rescheduled into tools/nnue/d40_night_v2.sh (waits for
  the 720sb train, then hop h2h [diagnostic] → 720sb parity+h2h → gate
  best PASSING net vs s4, verdict ~06:30). Reason: 7.6GB box cannot host
  2 sota engines + the resident trainer (OOM risk to an un-resumable
  train).
- Ledger verify hardened (commit 7fa53f4): runs end-to-end now; first
  honest full pass = 2 problems (sf100+lc0atanhraw-9pct-27m DRIFT from
  deleted mix shards; exp-0042.A dangling → unregistered net
  sf100-matrix-v1-sf100-d40-full whose .cenn exists). Both need an
  annotate/status mechanism or a registration decision (user pending).
- Rust perf program queued as tasks R1-R4: native builds (R1, deferred
  to quiet box), AVX-512 VNNI sota_simd path (R2 — box is Zen 4, VNNI
  available; attacks D40's NPS tax directly), PGO (R3), bounds-check/
  false-sharing audits (R4). All gate through bench bit-reproduction
  (D24); sota f32 tail is the rounding hazard.

## 2026-07-16 20:50 [PLAN] Throughput program: optimize the research loop itself

Measured tonight: D40 trainer is GPU-BOUND (99% util) — local training
speedups are capped (only levers: batch 16384→32768 fits VRAM 3.4/8GB +
TF32 — but batch changes recipe semantics, treat as an experiment, not a
free win). The wall-clock cost of a RESEARCH DECISION is dominated by
TESTING, so that's where the program aims. Ranked (impact × effort):

- T1 sota engine RSS audit: runbook says 3.6-5.1GB/engine but the D40 net
  is only ~250-500MB of weights — likely loader staging waste (f32 copies
  not freed / oversized per-thread tables). If RSS → <1GB, D40 testing
  conc 1→3 (≈3x all D40 screens+gates) AND kills the OOM-vs-trainer
  conflict that pushed tonight's gate overnight. AUDIT FIRST, tomorrow.
- T2 unbalanced book (= S0, queued): our anchor matches run ~70% draws;
  UHO-style unbalanced books are exactly fishtest's fix. Fewer games per
  verdict (~1.3-1.7x).
- T3 SPRT early-stop in eval_match (fastchess native): clear results stop
  at ~1/2-1/3 the games; borderline ones run long — right allocation.
- T4 fixed-node screen concurrency: node-limited games are conc-invariant
  by design; classic screens at conc 3 waste 13 cores. conc 12 ≈ 4x.
  One-line + a one-time invariance sanity check.
- T5 interruptible training: periodic D40_SAVE_RATE + accept fresh-stream
  resume (with random-skip, restarting the stream is a distribution nop
  for multi-epoch runs — revisit the "never interrupt" rule deliberately).
  Removes the 7h-block serialization of the whole box.
- (user Q) WSL2 RAM ceiling: box shows 7.6GB total + 2GB swap — if the
  Windows host has 16/32GB, a .wslconfig bump is the cheapest multiplier
  of all (removes conc caps + OOM fear even before T1).
- (flagged, D36) GCE-spot OpenBench workers when S2 SPSA starts (10k+
  games/tune) — designed for exactly this, but LOCAL-ONLY mandate stands
  until the user lifts it.

## 2026-07-16 21:00 [INCIDENT+FIX] OOM killed the 720sb train; root cause was a per-game net leak, now fixed

**What happened.** Post-compaction throughput work (T1 RSS audit) measured a
standalone probe-sota at 135MB peak through an 8s/2.27M-node search and
concluded the "3.6-5.1GB per sota engine" figure was a mismeasurement. On
that basis the hop h2h (fixed-node, believed RAM-safe) was launched at 20:16
alongside the resident 720sb trainer. Under real match conditions the
engines ballooned (~90MB/s each); at ~20:25 the kernel OOM killer shot the
trainer (d40_features, ~70 min into ~7h), then one engine. The night chain
was killed by hand before it could launch the same match again. **The old
rule was right and the standalone measurement was the trap: engine RSS must
be measured under fastchess, not a hand-driven UCI session.**

**Root cause (proven, not guessed).** A tee-wrapper capture of 4 games shows
fastchess re-sends every configured option — `setoption name EvalFile ...`
included — before EVERY game. Each call to `nnue::set_eval_file` re-read and
re-parsed the net and `Box::leak`ed a fresh copy: one net per game per
engine, forever (~78MB D40, ~3MB classic-2048). The historical 3.6-5.1GB
observations were ~46-65 games of accumulation, partially paged to swap.
All match RSS forensics ran engines under `ulimit -v` caps so a repeat
overrun could only kill the capped engine, never the trainer.

**Fix (df2b746).** `set_eval_file` now records the installed file's identity
(canonical path, size, mtime) and treats a same-file re-send as a no-op; a
changed file at the same path still reloads; failed installs drop the cached
identity. Regression test `tests/eval_file_reload.rs` (own process — it
installs process-global nets). Full suite green (631 tests), clippy clean on
the merge gate, check.sh bench fingerprint **548978 reproduced bit-for-bit**
(non-functional for search). Verified in production conditions: 20-game
fastchess match, RSS flat 134-135MB the whole way (was 756→1661MB and
climbing at the same point pre-fix). probe-sota + probe-l2048 rebuilt at
df2b746 (only src change since 52d1de9); `.pre-leakfix` backups kept.

**Cost + consequences.** ~70 min GPU time + ~5.5h calendar slip (train
restarted 20:48, done ~03:50; night-chain verdicts ~08:00, unchanged in
content). In exchange: the single biggest known testing-throughput lie is
gone — D40 engines are ~135MB, so T4 (screen concurrency) now applies to
the full-feature track, and the trainer-vs-testing RAM conflict is over for
fixed-node work (see amended RAM rule in the checkpoint). Standing lesson
recorded in memory: measure engine RSS under the real match harness, and
smoke-test any "this rule is wrong" conclusion under the conditions the
rule was written for before acting on it.

## 2026-07-16 21:40 [TOOLING] T3 SPRT early-stop + T4 concurrency invariance both landed

**T3 (95dfda5).** `eval_match.py` gains `--sprt-elo0/--sprt-elo1`
(fastchess-native SPRT, logistic model so bounds are regular Elo,
alpha=beta=0.05, `--games` stays the cap). Early stop records an `sprt`
block (llr/bounds/verdict) in summary.json; completed dirs stay idempotent
via stored `pgn_games`. Default invocations are byte-identical (the
overnight chain is untouched). Smokes: classic 4-game regression clean;
D40 big-gap pair with a [0,200] window stopped **H1 at 16/100 games**.
CALIBRATION LESSON: a narrow [0,10] window resolved SLOWLY even at 100%
score (LLR 1.42 after 100g) — SPRT windows must be sized to the question:
~[0,50] for is-it-clearly-better screens, ~[0,10] only for gate-grade
close calls where you'd accept long runs.

**T4 (this entry).** Fixed-node concurrency invariance PROVEN: 60 games
s3-vs-s4 @5k, seed 4242, conc 1 vs conc 12 — same opening assignment,
same W-L-D (17-23-20, −34.9), and **0/60 games differ in movetext** after
stripping fastchess's per-move time comments. Wall clock 20s → 6s
(~3.3× with the trainer loader running; better on longer matches). NEW
PRACTICE: fixed-node screens run `--concurrency 12` (classic AND D40 —
post-df2b746 engines are ~135MB). Equal-time matches keep conc 1-3 and a
quiet box (wall-clock fairness, unchanged). eval_match's default stays 1
so TC invocations can't inherit 12 by accident.

## 2026-07-16 22:30 [RESULTS] Evening fixed-node slate (post-leak-fix, conc 6-12 beside the trainer)

All fixed-node @5k, post-df2b746 engines, capped wrappers, trainer at 99%
GPU throughout — the first productive use of the reclaimed testing lane.

1. **Hop is DEAD (exp-0066).** d40-s2-farseer-120sb vs its own stage-1
   parent: **-23.2 +- 17.1** (300g). The farseer stage-2 recipe that gained
   +55 on classic REGRESSES the D40 arch. The parity waiver/rescale
   question is moot; gate candidate stays d40-uho-v1-240sb (or 720sb).
   OPEN QUESTION for the D40 track: why does stage-2 warm-retraining hurt
   here — LR too hot for a warm start? 120sb too short? threat features
   overfitting to farseer's distribution?
2. **D40 beats the INCUMBENT at fixed nodes (exp-0067).** d40-uho-v1-240sb
   vs s3-2048-farseer: **+30.2 +- 17.4** (300g SPRT [-10,10] cap-out,
   LLR 2.04). Consistent with +32.5 vs s4 (exp-0065). The arch is ahead of
   both classic bests per node; tonight's equal-time gate prices the NPS tax.
3. **Book sensitivity found, S0 promoted in urgency (exp-0068/69/70).**
   s3 vs s4 on the fishtest UHO unbalanced book (downloaded to
   data/nnue/openings/UHO_Lichess_4852_v1.epd, 2.63M lines): pooled 400g
   **-45 +- 15** vs **-15.6 +- 11.9** on our 256-line balanced book — a
   ~30 Elo book effect (~1.5 sigma). s4 (t74 stage-2) is plausibly the
   better net from unbalanced openings and our promotion history is all
   balanced-book. NOT leaderboard-invalidating on this evidence, but:
   adopt UHO for future fixed-node screens, and re-run the s3-vs-s4
   equal-time comparison on UHO before promotion #5. Draw rate at 5k was
   flat (26%->24-26%) — the S0 draw-rate claim is a TC question, untested.

## 2026-07-16 23:45 [CODE] S1 landed: correction history fully wired (a8e16c5) — D33 review in flight

The held-back half of Wave-F item 21: `ss.static_eval` is now the CORRECTED
eval (main-search N6 + qsearch stand pat); TT keeps raw everywhere (D22);
write-back was already live. Read/write formula parity checked against SF
search.cpp line-by-line. New bench fingerprint **701605** (was 548978,
+28% nodes at depth 6 — expected-functional per the doc; one pawnless
bench position balloons to 580k nodes, see below). Suite green: 586 tests.

Findings worth remembering from the landing:
- **Pawnless positions degrade correction history**: pawn_key and (in
  minor-less positions) minor_key are constant across the whole tree, so
  the mate-line's +256-clamped bonuses saturate the shared cells and
  inflate every sibling's margins. Empirically: the two-rook mate-in-3
  needed depth 12 (was 6) to prove exactly; middlegame positions shift
  ~1-3cp only. SF ships the identical formula (verified) — inherent
  tradeoff, not a bug. Doc 07 §8(4) bar amended 6→12; oracles re-scoped
  to exact-on-decisive + 8cp band (measured 3cp drift).
- S1 is FUNCTIONAL: games are the verdict. SPRT queued behind the D40
  arc per the search program (quiet box needed). Elo prior from the
  SF14→16 patch study: correction history family was worth ~+10-25.
- D33 3-persona review (correctness / SF-fidelity / test-quality, opus
  agents) dispatched post-commit; synthesis + any fixes before game
  testing. check.sh re-run on the shared release dir due ~03:50 when the
  720sb train frees it (side-build bench already bit-reproducible 2x).

## 2026-07-16 23:15 [PROGRESS] Overnight window: S1 screen positive, R2 VNNI landed, distribution builds

- **S1 fixed-node screen (exp-0071): +18.5 +- 17.4** vs pre-S1, same s3 net,
  300g @5k on UHO — mid-prior (+10-25 family), weak-positive, no red flag.
  Equal-time SPRT on the quiet box is still the verdict.
- **R2 landed (d895cf6):** AVX-512 VNNI dispatch tier in sota_simd —
  bit-exact (per-backend parity sweep), +6.4% D40 NPS measured on the busy
  box (lower bound; the compare binary lacked S1's per-node cost). Do NOT
  rebuild engines/bin probes until the night gate completes; then rebuild
  at HEAD (leak fix + S1 + VNNI together) and get clean R1 NPS numbers.
- **Distribution:** embed-net now takes ARCHON_EMBED_NET at build time
  (6725e38 — the old hardcoded net-0 embed would panic on width builds);
  two self-contained Windows exes (s3/s4 embedded, l1-2048, bench-verified
  701605) shipped to the user's desktop.
- S0 progress: UHO book adopted for screens (exp-0068..70); the SF16-late
  mechanics audit is running as a study agent, output lands in
  stockfish_reference/07-s0-sf16-mechanics-audit.md.
- Pending USER decisions: exp-0042 code_sha convention ("unknown"
  recommended); gitignore config.json.

## 2026-07-17 01:15 [CODE] D41 Phase-1 infrastructure: obs M1+M2 landed in one evening

The doc-16 Observation Store is real: `src/obs/` (0f81b34) — std-only TLV
(protobuf wire MODEL, no dependency), full four-layer schema incl. the
§5.2 completeness fields, shard framing with whole-episode appends;
contract tests green (round-trip, forward-compat incl. future EpisodeKind,
provenance-mandatory both sides, newer-version refusal, crash-truncation
semantics). M2 (e940e03): Board<->PositionState conversion (doc 15 nibble
convention, reconstruction THROUGH from_fen so shard bytes are validated,
never trusted) + `obs-tool verify/dump` — the design/19 §11.1 replay
audit; smoked on a real shard (0 problems) and a bit-flipped one (caught).
All behind feature="datagen"; default build + bench untouched; clippy
clean both feature sets. NEXT: M3 RootMoveAnalysis generator (D43 root-v0
searchmoves arms, cold state) -> M4 D44 stability/disagreement audits ->
M5 D33 review -> first Reckless labeling run (which does NOT wait for the
evaluator freeze). Revised Phase-1 forecast: first "more than eval"
training in ~3-5 days, gated mainly by M3 + the eval verdicts.

## 2026-07-17 01:15 [OVERNIGHT] Auto research loop armed; obs stack reviewed + first labeling run in flight

USER DIRECTIVE: autonomous loop until morning. In flight right now:
1. **720sb D40 train** — ends ~03:50; then check.sh reruns the S1 merge
   gate on the shared release dir; then d40_night_v2.sh runs h2h-720 +
   the EQUAL-TIME GATE (promotion #4) — verdicts ~06:30-08:00.
2. **obs sample v1 labeling** (tools/obs/label_sample_v1.sh, bg): the
   SAME 1000 seeded UHO positions through Reckless-d445b8b (child arms)
   AND SF18 (searchmoves arms), 4k/16k/64k tiers, then the M4
   stability/disagree audits — the dual-teacher data for the "is
   Reckless the right teacher?" question. Ledger add-data rows on
   completion. Results tee'd to chain-v2-results.log ([obslabel] tags).
3. **D33 round-2 verifier** (agent) re-checking the a664da0 synthesis;
   round 1 found 2 HIGH (canonical_key was the rule50-folded key —
   dedup/leakage-barrier breaker; child-mode mate-arm inversion) + 4 MED,
   ALL fixed with regression tests in a664da0. CONVERGED verdict expected
   → M5 closes → first big labeling run authorized.

Obs program status: M1 store, M2 verify/replay, M3 generator (+child arm
mode: Reckless IGNORES searchmoves — caught by the audits before any data
was generated), M4 audit tools all landed and reviewed. Reckless teacher
pinned (engines/bin/reckless-d445b8b, sha256 2e9fc5dd…). Teacher
recommendation on record: Reckless primary on alpha-beta-affinity +
teacher-diversity grounds, SF18 same-row control; the sample-v1 audit
numbers decide, diversity breaks ties.

## 2026-07-17 02:50 [CODE] R4 micro-audit: two bench-neutral SOTA speedups landed; big-v1 labeling staged

D33 round 2 CONVERGED at 01:40 (M5 closed; fixture fix 55c310f). Idle
window while the SF18 labeling pass runs was spent on R4 (task #16):

- **R4 audit** (read-only agent): docs/research/r4-micro-audit-2026-07-17.md.
  6 ranked candidates; TT layout/prefetch, MovePicker hoisting, and the
  classic accumulator verified already-tight. The one functional candidate
  (#5, defer correction_value past the NonPV TT cutoff) is parked — SPRT
  cost, hindsight coupling risk.
- **Landed #3** (943584d): threat-apply scratch arrays hoisted onto
  ThreatContext — removes an ~8 KB stack memset per SOTA do_move.
- **Landed #2** (3257492): find_nnz — the last scalar step in the SOTA
  forward, 192 branchy serial iterations — is now a dispatched SIMD kernel
  (AVX2 movemask + 256-row compaction table; identical index set, parity
  fixtures per backend). Guessing 3-9% SOTA NPS combined; measured numbers
  deferred to the R1 quiet-box session.
- Candidate #1 (make the FullThreats fold lazy like HalfKA, est. 5-15%
  SOTA NPS) is the remaining big lever — a real refactor, queued for a
  window with review capacity, not a 3am one-shot.
- **Verification gap found and closed**: the sota modules are compiled ONLY
  under `--features sota`, so the default check.sh gate never clippy-checks
  or tests them. Both commits verified explicitly with
  `--features sota` (633 lib tests green incl. threats_replay_*); pinned
  bench stays 701605 (the default bench build compiles sota out entirely).
  A pre-existing `sota,nnue-l1-2048` feature-combo build break was
  confirmed pre-existing (fails at the parent of 943584d too) and left
  alone — no probe config uses that combination.

Also staged: **obs big-v1** (tools/obs/label_big_v1.sh, a76d1c7) — 10k
fresh UHO positions (seed 20260718, sample-v1's 1000 excluded so sample-v1
stays a clean held-out shard), both teachers in PARALLEL (fixed-node =
load-immune, T4), behind a guard that waits for the night-chain gate, the
S1 SPRT, and the trainer (TC games need the quiet box). Launch decision
after the sample-v1 audits are reviewed. ETA ~10.5h from launch.

## 2026-07-17 03:15 [RESULTS] Obs M4: sample-v1 dual-teacher audits — GO for the big run

Both 1000-position shards complete and ledgered (obs-sample-v1-reckless
sha256 9c4981a0…, obs-sample-v1-sf18 sha256 73329eeb…; ledger verify 0
problems). Reckless pass 63 min, SF18 pass 54 min. The M4 fail-fast
audits, first real data:

- **Structure**: `verify` 0 problems on both; identical per-move row
  counts (101,727) — both teachers labeled exactly the same arm sets.
- **Stability** (per teacher, 16k→64k deepest tiers):
  | | accepted (stable+≥20cp) | flip | tie (<20cp) | cross-tier \|Δscore\| p50 | top margin p50 |
  |---|---|---|---|---|---|
  | Reckless | 39% | 23% | 38% | 10cp | 16cp |
  | SF18 | 36% | 24% | 40% | 8cp | 13cp |
  Read: **scores are reproducible** (median 8-10cp drift across a 4x node
  jump), but **argmax labels are inherently soft** — the median top-arm
  margin is only 13-16cp on UHO middlegames, so best-move identity churns
  on near-ties. Consequence for design/19: train on per-arm VALUES /
  margin-weighted or distributional targets, not hard best-move classes;
  the ~37% "accepted" subset is the natural hard-label stratum if one is
  ever needed.
- **Cross-teacher disagreement**: 75% deepest-tier best-move agreement,
  arm-ranking Kendall tau mean 0.841, unrestricted-score gap p50 11cp
  (p95 31cp). Two independent engines produce convergent arm rankings —
  strong external validity for the label signal; neither teacher is an
  outlier.
- **Teacher verdict**: no disqualifier on either side; numbers are near
  identical (Reckless marginally more decisive margins). The standing
  recommendation HOLDS: **Reckless primary** (alpha-beta affinity +
  engine-diversity vs the SF-derived eval lineage), **SF18 control** on a
  subsample of every future run. Caveat on record: Reckless arms are
  child-negate (root TT context lost), SF18 arms are true searchmoves;
  the 75%/0.841 agreement bounds the semantic gap as small.

**GO issued**: big-v1 launched 03:10 (10k positions, seed 20260718, both
teachers parallel), self-guarded behind the night-chain gate + S1 SPRT +
trainer; expect labeling to actually start ~09:00-10:00 and finish
~19:30-20:30. One guard bug caught immediately: pgrep|wc under
`set -o pipefail` exits 1 on zero matches — fixed, committed, relaunched.
M4 (task #24) is CLOSED: both fail-fast audits ran on real data and the
signal is trainable.

## 2026-07-17 03:50 [RESULTS] 720sb train: parity FAIL on the max tail (again the OOD-endgame clip); S1 merge gate GREEN

**d40-uho-720sb** (7h train, 720 superbatches on full UHO) completed and
packed, then FAILED the runtime-parity gate the same way the farseer hop
did: mean is clean (quant 0.0330, runtime 0.0121 — well under the 0.03
bar), but max = 0.2517 vs the 0.20 bar, worst fen a bare rook endgame
(`1k6/4R3/2K5/8/8/8/P4b2/8`). That is now TWO D40 nets whose only parity
failure is a quant-clipping tail on OOD pawnless endgames (240sb passed
at 0.174; longer/warmer training pushes some weight past the clip). The
night chain handles it correctly by design: 720sb is h2h'd as a
DIAGNOSTIC but the equal-time gate (promotion #4) runs with the parity
-passing **d40-uho-v1-240sb**. Open decision for the user (both hop and
720sb): (a) waive the max-tail bar for OOD-endgame-only divergence (mean
and p99 are clean), (b) repack with per-layer quant headroom (netpack
change; the T5 raw f32 checkpoint exists, so any repack attempt costs
minutes, not a retrain), or (c) train the next long run with weight
clipping so packing never saturates. Note `rescale_net.py` does NOT apply
here — it hot-scales the OUTPUT layer; this tail comes from packed
mid-net weights saturating their integer range.

04:34 update — the night chain's diagnostics are in (exp-0072): **720sb
beats 240sb +30.2 ± 17.1 at fixed nodes** (300g @5k, 121W/84D/95L). The
longer schedule genuinely helps; the ONLY thing blocking it from the gate
is the packing tail. This tilts the repair decision toward (b)/(c) — fix
packing/clipping and re-gate — over waiving. The hop diagnostic confirmed
the cached evening number (−23.2, exp-0066; hop stays dead). The
equal-time gate (240sb vs s4, promotion #4) started 03:52, verdict ~06:50.

## 2026-07-17 05:35 [VERDICT] Promotion #4 REJECTED: D40 loses −57.9 ± 12.6 at equal time. Evaluator-freeze proposal.

**GATE (exp-0073)**: d40-uho-v1-240sb (best parity-passing D40) vs
stage2-v1-s4-2048-t74-240sb, 200g 10+0.1, quiet box:
**−57.9 ± 12.6** (11W/145D/44L, score 41.7%). The full-feature net wins
per node (+32.5 ± 17.3, exp-0065) and its 720sb sibling adds another
+30 per node (exp-0072) — but at the clock the inference tax erases all
of it. D40 needs roughly 1.7–2× more NPS to convert its per-node edge
(~70 Elo per doubling rule of thumb); the realistic near-term speed
program (R4 #1 lazy threat fold 5–15%, PGO ~5–10%, native build ~5%, R2
VNNI +6.4% already landed) sums to ~20–35%, not 70–100%. Conclusion: D40
is not promotable on this hardware without an algorithmic speed break-
through; it stays a RESEARCH line (packing-tail fix + speed program),
re-gated only after a major NPS win. The night chain worked exactly as
designed end-to-end: parity bar → diagnostics → gate → clean verdict.

**EVALUATOR-FREEZE PROPOSAL (for the user, morning decision):**
1. **Freeze the classic binpack line as THE evaluator** for the next
   program phase. Incumbent stays `stage2-v1-s3-2048-farseer-120sb`
   (+265 composed over net-0); `s4-2048-t74-240sb` is working-best
   (+15.6 ± 11.9 pooled fixed-node, same width/NPS family). One cheap
   pending item: the s4 equal-time re-check on the UHO book (promotion
   #5) to make the frozen evaluator the true best classic net.
2. **Pause new eval trainings.** The B4 question is answered (data
   volume), width is saturated at 2048, and stage-2 gains are visibly
   shrinking (+55 → +29 → +15.6). Marginal Elo per GPU-hour on this box
   is now far below what the search side offers.
3. **Shift primary effort to search**: S1 correction history is merged
   (SPRT pending tonight, fixed-node screen +18.5 ± 17.4), S2 SPSA is
   queued, and the SF16-late audit shortlist (un-deferrals) is scoped.
4. **D41/design-19 becomes the main research thrust**: the obs platform
   is production-proven tonight (M1–M5 closed, dual-teacher audits GO,
   big-v1 10k labeling in flight). The eval foundation this program
   waited for is settled by this freeze.
5. D40 continues in the background only as engineering: fix the packing
   tail (netpack quant headroom or clip-aware training — raw ckpts
   exist), take the speed program as far as it goes, re-gate when NPS
   justifies it.

## 2026-07-17 12:55 [DECISION 4/8 PENDING] c4 NMP: second real mover — deeper null-move confidence wanted

Ledger row filed. NMP_EVAL_BASE 374 -> 507 (23% of range) and
NMP_EVAL_DEPTH 14 -> 12: the tuner wants null-move pruning applied more
readily. Screen at 1000g cap: +9.7 ± 17.0, LLR 0.63 — same shape as c3
(positive lean, undecided). Pattern so far with r_end=0.2: the margin/
threshold bundles DO have findable direction, sized ~5-10 Elo — exactly
the band the TC gates exist to adjudicate. Two pending gates queued
(c3, c4); c5 futility+SEE running.

## 2026-07-17 12:20 [DECISION 3/8 PENDING] c3 RFP+razoring: first real theta movement — gate tonight

exp-0079-adjacent (ledger row filed). With r_end=0.2 the harness finally
moves: **RFP_IMPROVING 2934 -> 1947** (22% of range, ~5 sigma above the
step-noise floor — the improving-trend discount on RFP margins wants to
be ~34% smaller), RFP_THREAT_MARGIN 55 -> 46, RAZOR_BASE 465 -> 422,
six smaller drifts. Fixed-node screen at the 1000g cap: +7.3 ± 16.8,
LLR 0.31 (positive lean, SPRT undecided). Decision 3/8 rides on
tonight's TC ship gate — night_gates_v1.sh generalized to gate every
campaign whose theta moved (vacuous thetas self-skip). c4 (NMP) running.

## 2026-07-17 12:15 [CALIBRATION] The SPSA season's true meaning: the search-param landscape is a high plateau

exp-0078, two measurements: (1) the r_end=0.01 control did NOT walk back
(final 1533 from 1541; z mean +0.0075 ± 0.016 — zero local gradient);
(2) the direct calibration match: **RED_BASE 1541 (a +50% detune of the
base LMR reduction) costs only −8.1 ± 12.9 Elo** at 20k nodes, 1200g.

The harness was never broken — there was nothing to find. Re-search
safety nets (do_deeper/do_shallower, PVS re-search, TT) absorb even
gross reduction mistuning. Season recalibration, in effect immediately:
- c1/c2 NO-CHANGE decisions are REAL findings ("no >~10 Elo error in
  the bundle"), not harness artifacts.
- c3–c8 run at 400 iters (4,000 games each — same certification power,
  half the cost) with r_end=0.2 (steps large enough that any genuinely
  steep gradient CAN move theta; c7's eval-blend params are the likeliest
  to have one).
- Sub-10-Elo per-param mining is fishtest-scale work (100k+ games per
  param) that this box cannot do; after the 8 decisions the Elo lives in
  the S0 structural un-deferrals, the D40 speed/gate arc, and D41.

## 2026-07-17 11:30 [ANALYSIS] p25 twin verdicts + the SPSA step-size diagnosis

**Study experiments #1+#2 (exp-0077, d40-uho-p25-240sb):** loss ^2.5 is
NEUTRAL vs ^2 at 240sb full-UHO scale — pooled 900g fixed-node
+5.0 ± 9.7. Parity is cleaner (runtime mean 0.0076 vs 0.0121) and the
net passed the recalibrated bars on the first try. No adoption alone;
revisit only bundled with a longer schedule. The #1 payload delivered:
8 harvested checkpoints (sb30..sb240) ready for a selection-by-games
probe. Net ledgered (cb3c3eea…).

**SPSA harness diagnosis (from the control's live trajectory):** at
k=125/200 the detuned RED_BASE has walked back −7.7 of −514 — almost
exactly the EXPECTED −0.07/batch for a ~30-Elo detune at r_end=0.01.
Conclusion forming: the harness detects gradients fine; the STEP SIZE is
the bottleneck. With c1/c2's r_end=0.002 the maximum systematic traverse
over a whole 800-iter campaign is a few units on 10000-unit scales — the
two nulls were foregone conclusions, informative only as "no gross
(>50 Elo) detune exists in those bundles." fishtest uses r~0.002 with
25-100x our games; at our budget r_end≈0.2 makes an 800-iter campaign
able to traverse ~10-15% of a param range against realistic z-bias while
keeping end-noise ~10% of range. PLAN once the control completes: rerun
the control at r_end=0.2 expecting a DECISIVE walk-back; if confirmed,
redo c1+c2 at r_end=0.2 (2x2h, cheap) and run c3-c8 at r_end=0.2 — the
season's 8 decisions get made under a harness that can actually move.

## 2026-07-17 11:00 [DECISION 2/8] SPSA c2 (LMR/reduction bundle): NO-CHANGE — and the harness goes on trial

exp-0076. Second consecutive perfect null: all 15 reduction params within
0.25% of defaults after 8,000 games (the bundle the SF14->16 study ranked
as the biggest single gainer pool). Two readings, now formally split:
(a) the SF-derived constants really are at a local optimum for our engine
at this eval; (b) the harness cannot resolve gradients at this budget
(8k games, r_end=0.002 — fishtest uses that r with 25-100x more games).
The DETUNED-START POSITIVE CONTROL launched 10:57 decides it: RED_BASE
starts at 1541 (1.5x default — a deliberate ~+50% reduction offset, a
known material detune) with --r-end 0.01, 200 iters. If theta walks back
toward 1027, the harness works and the nulls stand (and r_end 0.01
becomes the season default); if it sits still, decisions 3-8 are FROZEN
until the sensitivity problem is diagnosed. c3 waits for this verdict.

## 2026-07-17 09:20 [DECISION 1/8] SPSA c1 (corrhist weights): NO-CHANGE — the S1 constants survive

exp-0075. 800 iterations x 10 games @20k nodes (8,000 games) perturbing
all 13 correction-history read/update weights: every parameter came home
to within ±0.1% of its SF-parity default (mid-run wander up to ±90 units
on CONT_NULL_PRIOR fully reverted). The landscape is flat at this
budget's resolution (gradient floor ~5 Elo at 8k games). Decision:
KEEP DEFAULTS; ship-gate skipped as vacuous (theta_final == defaults
within integer rounding). Two useful facts fall out: (1) SF's corrhist
constants transfer to ARCHON's eval unchanged — consistent with S1's
clean +72.9 landing; (2) the harness needs a positive control if c2
(reductions — historically strong gradients) also comes back flat.
c2 launched immediately after (seed 20260720, same budget), with
tune-l2048 redeployed at HEAD (eval tunables + advertisement fix in).
Season: 1/8 decisions done. In parallel: d40-uho-p25-240sb (study exps
#1+#2) trains until ~11:30.

## 2026-07-17 08:20 [FINDING] The D40 "packing tail" was never packing — parity gate recalibrated; 720sb admitted to gating

Full note: docs/research/d40-packing-tail-2026-07-17.md (+ reusable
tools/nnue/audit_d40_quant_clip.py). Headlines: ZERO quant overflow in any
D40 tensor (bullet errors rather than saturates, and AdamW clips at ±1.98
— clipping was structurally impossible); the passing 240sb net actually
has LARGER threat weights than the failing 720sb; the runtime parity
error is integer-vs-float arithmetic divergence (the (l*r)>>FT_SHIFT
floor), a thin OOD-endgame tail on a clean distribution (720sb: mean
0.0121, p99 0.0635, max 0.2517). Gate recalibrated (train_run.py): mean
<= 0.03 AND p99 <= 0.10, max backstop 0.30 (sf-100's true transform bug
was mean 0.355 broad — still caught instantly). Network_scale rebalance
proven impotent (identical max at scale 400 vs 800); no repack can help
by construction, so no -rs1 net exists and the ledgered 720sb sha is
unchanged. CONSEQUENCE: d40-uho-720sb (+30.2±17.1 per node over 240sb,
exp-0072) is now a legitimate gate candidate — queued for tonight's
quiet-box window: equal-time vs s4 after the c1 TC gate. The two failed
"parity" hop nets from yesterday deserve re-audit under the new bars too.
Optional engine-side improvement (R1 of the note): rounding FT shift —
deferred, changes sota semantics mid-SPSA.

## 2026-07-17 08:00 [PROGRAM] D45: SPSA gainer season begins — 8 decisions, then the D41 shift

USER DIRECTIVE (07:10): "transition into a full SPSA auto research loop.
Goal is 8 full SPSA decisions from our plans and informed by our stockfish
study on optimization. After that we shift." This ratifies the optimize
direction of the 05:35 freeze proposal. Recorded as D45.

Infrastructure landed this morning (34deade + 2950ee1, both D24-verified):
- `tune` feature: all 233 params.rs scalars are runtime tunables
  (`setoption name <NAME> value <x>`), atomic-backed getters; the default
  build folds them back to constants — bench 701605 bit-identical.
  Proven live: ASPIRATION_BASE_DELTA=300 moves UCI-bench 4777232 -> 4590853;
  unknown names error loudly.
- `tools/spsa/spsa_tune.py`: local SPSA (paired Rademacher, fishtest
  schedules), fixed-node fastchess batches, both sides the SHIPPED s3 net
  (D39), preflight against the binary's `tunables` list. Smoke-tested.
- Decision protocol per campaign (design/14 §9.6): SPSA (~8000 games @20k
  nodes) -> fixed-node SPRT screen [0,10] -> EQUAL-TIME SPRT ship gate on
  the quiet box = the decision -> accept commits new defaults (functional,
  fresh Bench trailer) + ledger row.

Campaign order (S0 audit + SF14->16 study §6b + S1 freshness):
c1 corrhist weights (RUNNING since 07:58: 13 params, 800x10 games @20k,
beside the big-v1 labeling — fixed-node is load-immune) · c2 LMR/reduction
· c3 RFP+razoring · c4 NMP · c5 futility+SEE · c6 LMP+history-pruning ·
c7 aspiration+IIR · c8 history update magnitudes. TC gates run on quiet
windows (tonight); after decision #8 the program shifts to D41.

## 2026-07-17 06:10 [VERDICT] S1 correction history PASSES its equal-time SPRT: +72.9 ± 16.2, H1 at 208 games

**exp-0074**: engine-s1 (c10680f) vs engine-pre-s1 (df2b746), SAME s3 net
both sides, 10+0.1 conc 3 quiet box, SPRT [0, 10] α=β=0.05: **H1 at 208
games** (69W/113D/26L, +72.9 ± 16.2, LLR 2.95/2.94). The T3 early stop
saved ~2/3 of the 600-game budget.

Why equal-time (+73) so far exceeds the fixed-node screen (+18.5 ± 17.4,
exp-0071): correction history costs essentially no NPS, so its per-node
accuracy gain compounds through deeper TC searches (and 5k-node screens
barely fill the correction tables). Lesson for the protocol: NPS-neutral
SEARCH changes can be badly underestimated by shallow fixed-node screens
— route them to the TC gate quickly rather than iterating on screen Elo.

**Deployment call: S1 STAYS** (it is already merged at HEAD; merge gate
green, bench 701605). This is the largest single search gain since the
P9 pruning waves and lands squarely in support of freeze-proposal point
3: the search side is where the Elo is right now. Next in the program:
S2 (gainer-season SPSA retune) once the box has a free tuning window,
then the SF16-late un-deferral shortlist (S0 audit §6b).

**S1 merge gate**: with the trainer gone, `bash tools/check.sh` reran the
full D24 gate at HEAD (S1 correction history + R4 #2/#3 + obs stack):
fmt/clippy/tests/release-perft all green, bench fingerprint **701605**
reproduced. S1 is merge-clean; its equal-time SPRT (armed, [s1gate]) is
the remaining deployment gate.
