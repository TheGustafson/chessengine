# ARCHON Learned Search Allocation Research State

> **CONTINUATION INVARIANT:** Review this file, D41/doc 19, and
> `docs/research/archon-execution-plan.md`, and
> `docs/research/sf18-label-quality-plan.md` before every research loop. Update
> this file after every completed, rejected, or redirected
> trial. The legacy filename is retained so existing automation and compacted
> sessions find the same continuation record.
>
> **D41 GOVERNING INVARIANT:** External policy imitation is no longer the final
> objective. Lc0 and other strong engines provide positions and optional
> bootstrap/control labels. A slower, conservative ARCHON search provides
> native best-move, cutoff, error-risk, re-search, and cost labels. The value
> baseline is protected; learned authority expands only after calibrated
> fixed-node and equal-time evidence.
>
> **D42 VALUE-ORACLE INVARIANT:** When an external value label is needed, use
> official Stockfish 18 as the production bootstrap teacher. The newest compiled
> local Reckless release is now a sparse audit/disagreement control, not a
> required second label on every position; double labeling costs too much for a
> bootstrap corpus. Do not use Stockfish 16 except as a named historical scale
> and strength control. Freeze the selected value accumulator and head completely
> during the first control-head trials.
>
> **HIGH-SEVERITY DATA CONTRACT — SEARCH BOUNDS (2026-07-15):** Value labels
> generated with a hard 5,000-node limit are frequently aspiration-window
> bounds, not scalar values. In the audited cloud sample, 67.9% of Stockfish 18
> and 55.4% of Reckless labels were bounded. In the 5.66M-row local relabel,
> 43.0%/45.4% were bounded and only 38.2% were jointly exact. Treating these as
> exact targets caused lower held-label error while destroying playing strength
> (full protected HalfKA residual: 2-36-2 versus net-0 at 5k nodes). Never feed
> bounded values to a scalar loss. Every label shard and exporter must report
> lower/exact/upper counts and either reject bounds or use a censored inequality
> loss. Fixed depth 10 and 12 produced 0/100 bounds for both engines in a smoke
> test, proving completed iterations fix the semantic defect, but those depths
> are diagnostic only and are too shallow for the production value oracle.
> Production relabeling must use completed deep iterations plus score/best-move
> stability gates, escalating unstable or teacher-disagreement positions deeper.
> The two remaining cloud 5k Batch jobs and the local 5k relabeler were stopped;
> completed data is retained for policy, difficulty, and bound-aware research.
>
> **QUALITY-FIRST FLYWHEEL RULE (2026-07-15):** Throughput, cloud utilization,
> and apparent training progress are subordinate to label quality. No bulk
> generation or training may begin merely because a pipeline runs. Before a
> corpus is promoted, a small independently audited sample must establish: (1)
> complete and correctly reconstructed position state for the intended target,
> (2) exact score semantics and perspective, (3) explicit exact/lower/upper,
> WDL, mate, and missing-value handling, (4) sufficient search depth and
> score/best-move stability for oracle labels, (5) deterministic provenance and
> source alignment, (6) train/validation independence, and (7) useful held-out
> engine behavior at more than one search budget. Each promoted dataset must
> carry a machine-readable quality summary. Discovery of a semantic defect
> quarantines the dataset from affected losses and invalidates dependent
> conclusions until they are re-audited. Never lower a quality gate to keep
> workers busy.
>
> **STOCKFISH-18 HIGH-QUALITY ORACLE PLAN (2026-07-15):** Bulk labels must not
> come from one shallow fixed depth or a hard node-limit score. Use a tiered,
> resumable Stockfish-only oracle. Each position receives a cold natural
> iterative-deepening search with full FEN state (repetition history remains an
> explicit limitation), one thread, pinned binary/network/options, and a hard
> deterministic node ceiling. Capture completed iterations at deep checkpoints.
> A scalar search label is promoted only when the last three requested depths
> are exact, retain the same legal best move, and remain within a preregistered
> score span; mate labels require stable sign/status. Unstable positions escalate
> to a larger node/depth tier. Positions still unstable at the final tier are
> quarantined, not coerced into point labels. Store all completed iterations,
> nodes, WDL, PV move, depth/seldepth, bound type, FEN/source hashes, binary/net
> hashes, and rejection reason. First measure the stability/yield/cost frontier
> on a diverse local audit set, then run an independently verified pilot; only a
> passing pilot may unlock sharded cloud generation. Reckless is sampled only on
> a small stratified disagreement subset after the SF18 protocol is established.
>
> **LC0 DIAGNOSTIC INVARIANT:** There remains an unresolved systematic problem
> extracting generalizable value signal from the current Lc0 pipeline. The
> pivot does not erase it. Preserve the corrected corpus and controls, use the
> positions for ARCHON-native relabeling, and continue to isolate whether the
> deficit lies in targets, sampling/correlation, state representation, recipe,
> or evaluation/search interaction.

Last updated: 2026-07-15 America/New_York

## Active bulk value matrix — 2026-07-15

Per user direction, the loop has moved from tiny source-isolation trials to a
preregistered 4×2 bulk value screen, followed by combined-source training only
for survivors. The four cells are classic Chess768/256, classic Chess768/1024,
king-conditioned 64×Chess768/256, and exact HalfKAv2_hm+FullThreats D40. Exact
source semantics, matched budgets, early-stop rules, and the combined-loss mask
are recorded in `value-bulk-matrix-v1.md`.

All 17 Lichess Parquets are locally size/SHA verified and are being copied to
the project bucket before local space is reclaimed for the collection-aware
canonical stream. The complete `fishpack32.binpack` fetch is active. Direct
SF-binpack loading now compiles and runs for classic, king-conditioned HalfKA,
and exact HalfKAv2+FullThreats trainers; one-batch feature parity passed for the
new D40 path. The previous cold prefix classic256 run was a fail-fast rejection
at about -348 Elo, so the full matrix will pack/test early checkpoints and not
assume that more presentations cure semantic mismatch.

Bulk plumbing smoke passed: `export_lichess_bulk.py` carries FEN identity across
all batch/shard boundaries, vector-selects the source-ordered target, rejects
mate/extreme/structurally un-packable records, and streams directly into the
32-byte Bullet representation without a giant text intermediate. A 100K test
scanned 917,504 flattened rows, found 158,418 position groups, and wrote exactly
100,000 records; one-batch GPU loading at `WDL=0` passed. The exporter now
hashes its final artifact. The matrix runner is resumable and parameterized by
source/cell; it preserves separate candidate engine binaries for architecture-
specific runtime builds and applies 80-game fail-fast plus 2k/5k/20k curves.

Because the workstation's concurrent 40.6GB GCS upload throttled the 5.58GB
Hugging Face fetch, a single e2-standard-2 Spot Batch transfer was submitted as
`hf-fishpack32-20260715`, capped at $0.25 worst case. It downloads the exact
source, verifies LFS SHA-256
`04f90dcba242682cdb509e675ce590eff3676ea8b84a20e6baa8b5020f16ed16`,
and create-only publishes the file plus provenance sidecar to the Archon bucket.
Training remains local.

## Official Stockfish master-binpack checkpoint — 2026-07-15

The 13-file, 287,102,338,770-byte `official-stockfish/master-binpacks` Hugging
Face collection is now exposed through an exact remote manifest and selective
range/full-file fetcher; it is not bulk-mirrored onto the local disk. A 128 MiB
range from `fishpack32.binpack` yielded 127 complete chunks and 61,070,868
decoded positions with score, move, ply, rule-50, side-to-move game result, and
game continuity intact. Direct Bullet SF-binpack streaming support was added to
`train_net0.rs`, avoiding an expanded intermediate corpus. Full semantics,
licensing, audit numbers, and promotion gates are in
`official-stockfish-binpacks.md`.

The first net-0 fine-tune retained a 0.3 game-outcome blend. Its fourth
~1M-presentation checkpoint was about -61 Elo versus net-0 over 120 games at
5k nodes, compared with about -407 Elo for the 1M Lichess cp-only fine-tune.
The first checkpoint's 80-game screen was near even (+8.7 Elo, very wide
interval); later checkpoints weakened. A higher-confidence first-checkpoint
match and smaller effective-step sweep are active. This is not yet a promoted
network, but it isolates outcome-bearing official binpacks as a materially more
promising bootstrap stream than Lichess centipawn imitation alone.

Calibration update: the original ~1M-presentation checkpoint was -26 Elo over
400 games at 5k. Reducing the total effective step (`LR=3e-6`) produced -1.7
Elo at 5k and -10.4 at 20k, both statistically compatible with equality.
At the same step, outcome fractions 0.1/0.3/0.5/0.7 scored approximately
-0.9/-1.7/+7.8/+6.9 Elo at 5k over 400 games each; all intervals remain wide.
The 0.5 blend is the provisional leader and is being tested at 2k and 20k.
This is the first outcome-bearing external-data recipe to preserve net-0 and
show a positive point estimate, but it is not yet evidence of a real gain.

The current official nnue-pytorch defaults also use smart/WLD and soft-early
position filtering and expose 100M positions per epoch. Archon's initial quiet
filter is only a format compatibility baseline; sampling semantics must be
matched or experimentally justified before scaling.

The completed Lichess collection-level metadata audit covered all 17 verified
Parquets: 844,812,067 flattened rows with identical schemas. Nine of 16 tested
shard boundaries bisect a contiguous FEN group. Full-corpus readers
must carry boundary groups across files; independent per-file projection is a
silent target-selection bug. This contract is now enforced in the research
documentation before a large export is attempted.

## Lichess evaluations corpus — Kaggle v14

The CC0 Kaggle dataset `lichess/chess-evaluations`, version 14 (updated
2026-01-06), was fully acquired and size/SHA-256 verified under
`data/corpora/lichess-chess-evaluations/kaggle-v14/raw/`. The source release
contains 17 Parquet files, 40,612,747,732 bytes, 342,059,879 distinct chess
positions, and 844,812,067 denormalized evaluation/PV rows. Original files are
immutable; `tools/nnue/download_lichess_evaluations.py` records source metadata,
expected sizes, and per-file SHA-256 digests and resumes at file granularity.

This is a broad-pretraining/search-policy provenance tier, not SF18 gold. It
was produced by various browser Stockfish builds at heterogeneous depths and
node counts. Each position may have several searches and several MultiPV rows.
The source's recommended single-target projection is the first PV from the
highest-depth evaluation; Archon's projection must preserve original row order
and record the exact tie rule. `cp`/`mate` perspective, legal-PV behavior, and
the flattening/order assumptions must be independently checked before export.
The source FEN has only four fields: piece placement, active color, castling,
and en-passant. Rule-50, fullmove, and repetition state are unavailable, so the
same state boundary as other incomplete-history corpora applies. Search labels
may be used with explicit source semantics; they must never be represented as
controlled cold Stockfish 18 labels.

The first shard has 49,694,828 rows and the expected columns:
`fen:string`, `line:string`, `depth:uint8`, `knodes:int32`, `cp:int16`, and
`mate:int8`. Initial rows confirm contiguous groups containing multiple PVs and
multiple search records for one FEN. Collection schema, boundary behavior,
first-million legality, perspective, and SF18 static spot checks have passed;
full distributions, duplication, and deep-SF18 projection comparisons remain.

The first 100k-position canonical smoke has now passed the complete
Parquet-to-text-to-Bullet-to-CENN path. Its held-out Lichess-label MAE/Pearson
were 97.0 cp/0.792, better than net-0's 159.1 cp/0.625 on the same labels, so
the projected source signal is learnable. The random-initialized smoke lost
0-40 to net-0 at 5k nodes, as expected for only 100k unique positions; this
rejects the tiny model as a strength candidate but does not reject the corpus.
A 1M projection was then tested cold and as a protected low-rate net-0
fine-tune. Both fit held-out Lichess labels better while losing roughly 759 and
407 Elo respectively, and a zero-rate roundtrip proved initialization/packing
bit-exact. This rejects outcome-free Lichess cp imitation as a stand-alone
value recipe while retaining the corpus for coverage, policy/PV provenance,
hard-position mining, and SF18 relabeling. Full details are in
`docs/research/lichess-evaluations-corpus.md`.

The 16-position false-early-stability smoke also completed. Of labels accepted
at shallower tiers, 13/16 were value-stable at depths 26/28/30 and 11/16 were
joint value/best-move stable. This is strong evidence that apparent shallow
stability cannot be treated as gold without deeper validation. The precise
early-versus-deep score comparison still needs to be summarized before the
production SF18 floor is frozen.

## Pause checkpoint — 2026-07-14 20:23 America/New_York

Work is intentionally paused at the user's request. No relabel processes remain
running.

The resumable external-label tool now exists at
`tools/nnue/relabel_lc0_uci.py`. Its 16-row-per-side Reckless smoke passed:
atomic shards, deterministic affine source-row schedule, engine identity/hash,
score/best-move/depth/seldepth/node arrays, and exact resume validation all
worked. The manifest explicitly marks reconstructed records
`piece_state_bootstrap_only` because the normalized LC0 arrays lack castling,
en-passant, rule-50, and repetition state.

A full-corpus Reckless job was launched at 5K nodes, two nice-10 workers, and
4,096 rows/shard, then cleanly stopped after ~28 seconds for this pause. It had
written its manifest but no completed shard, so there is no ambiguous partial
data. Output path:
`data/lc0_nnue/relabel_reckless_d445b8bf_n5k/`. Resume with the exact command
encoded in that manifest; the intended target is all 50M normalized rows per
side in deterministic broad-coverage order. The original LC0 position arrays
remain canonical; relabel output stores only source IDs and new labels/search
diagnostics, not copied LC0 policy/result/eval metadata.

Near-term order on resume:

1. Resume the Reckless relabeler and verify the first completed production
   shards, throughput, score semantics, and restart behavior.
2. Add a completed-shard index/reader so value training can consume Reckless
   data while the full corpus continues in the background.
3. Specify and run small Reckless single-teacher value targets first, preserving
   the prevailing evaluator as the game control.
4. Relabel the identical deterministic source row IDs with pinned Stockfish 18,
   then calibrate scales and analyze disagreement before consensus training.
5. Promote a value representation only through evaluator-only games; freeze
   its complete accumulator and value head bit-identically.
6. Begin ARCHON-native move/cutoff labeling and frozen-value control-head
   experiments across multiple node budgets.

Plan correction from the user: item 6 is not serially blocked on items 1–5.
Search-control target semantics, telemetry, forced-counterfactual generation,
schema, and head/runtime scaffolding proceed as parallel Track B while external
value relabeling and frozen-accumulator selection proceed as Track A. Only final
head training/integration against the selected production accumulator is a hard
dependency. The previously omitted data-generation problem is now an explicit
G0C/B1–B3 milestone in `archon-execution-plan.md`.

D44 curriculum refinement: initial move-quality/search-complexity labels may
come from independent Reckless root arms. Generate stable best move, pairwise
ordering, score margin, PV stability, and per-move nodes at increasing budgets,
then rerun identical source row IDs with ARCHON and adapt. Keep producer engine
explicit. Do not interpret Reckless root outcomes as ARCHON-native cutoff,
reduction-risk, re-search, or deployed cost labels; those transition to ARCHON
internal observations/interventions. Promotion remains ARCHON engine games.

## GCP readiness checkpoint — 2026-07-14

Configured project: `archon-502500` (name Archon, lifecycle active, billing
enabled). Google Cloud SDK 576.0.0 is installed. Compute Engine and Cloud
Storage APIs are enabled; Batch and Artifact Registry APIs are not yet enabled.
There is no configured default region/zone, no bucket, and no VM. The project
has the default network only.

Observed regional headroom is sufficient for a CPU canary: general CPU quota
is 200 in `us-central1`, `us-east1`, `us-east4`, `us-west2` and 100 in
`us-west1`; instance quota is 24. N2 quota reaches 200 in the main eastern/central
regions, C2D 100, while C3 is currently 24. Each checked region has quota for
one L4 and one T4 but no A100. `PREEMPTIBLE_CPUS` reports zero; official Spot
rules say standard quota is used where separate preemptible quota has not been
granted, but a tiny Spot canary must verify this project-specific behavior.

Proposed cloud direction, not yet provisioned: use Google Cloud Batch task
arrays on Spot VMs for finite deterministic shard jobs, backed by immutable GCS
input/output shards and a digest-pinned container in Artifact Registry. A local
`tools/cloud/archon_cloud.py` controller should provide `doctor`, `benchmark`,
`plan`, `submit`, `status`, `reconcile`, `cancel`, and `gc`, defaulting to dry
run. Capacity discovery ranks quota/price/machine candidates but must attempt
allocation and fall through zones because Spot availability is not guaranteed
or reliably queryable. Task indices map to deterministic shards; workers write
atomic outputs directly to GCS and retries skip existing valid shards. No
worker-local disk is authoritative.

Before cloud scale: shard the canonical position-only corpus so workers fetch
small assigned inputs instead of every VM downloading the 6.4GB feature arrays;
create least-privilege controller/worker service accounts; add budget alerts and
script-side cost/wall-time/retry caps; run a one-task Spot canary; benchmark
labels-per-dollar across C4/C2D/N2-class candidates; then run a distributed
1M-label parity canary before the full corpus.

## Current goal and acceptance gates (D41)

Build an ARCHON-native search-action consequence model that improves the
strength–compute frontier of the strongest available frozen evaluator. The
first deployment target is safe and bounded: native move ordering/cutoff
prediction followed by a calibrated reduction-risk head allowed to correct
baseline LMR by at most `[-2, +2]`.

A candidate advances only when all relevant gates hold:

1. Oracle examples deterministically replay with complete position/search
   context, stable reference status, forced action, returned bound/score,
   re-search outcome, nodes/cycles, and artifact provenance.
2. The protected evaluator is bit-identical with control heads attached and
   disabled; semantic and runtime smoke tests pass.
3. Offline risk estimates are calibrated and reported by depth, node type, move
   class, source, and hard/OOD strata. Training aggregates cannot hide a failed
   tail.
4. Frozen-evaluator fixed-node games compare classical ordering, Lc0-policy
   control, and ARCHON-native control over several node budgets, paired
   openings, multiple seeds, and pair-bootstrap intervals.
5. Reduction control begins as a bounded correction with conservative fallback;
   uncertainty may spend more compute but may not authorize early hard pruning.
6. Promotion requires a reproducible positive Elo-versus-node-budget curve and
   positive equal-time evidence after NPS, invocation cost, re-search rate, and
   tactical regressions are counted.
7. Each data/model/oracle/controller/match artifact is entered into the D38
   ledger, and this state file records the promotion or rejection rationale.

## Immediate D41 execution loop

1. Identify and hash the strongest current evaluator; freeze it as the first
   control rather than assuming `net-0` remains strongest forever.
2. Define the minimal doc-16 node-context/counterfactual records and build a
   tiny deterministic ARCHON oracle relabeling smoke test.
3. Sample a small stratified slice of the existing Lc0 **positions**, including
   tactical/unstable and engine-disagreement strata; do not trust game-
   contiguous random rows as independent coverage.
4. Generate stable native final-best-move, pairwise, alpha-raise, and cutoff
   targets. Audit score perspective, bounds, search stability, legal PVs,
   castling, rule-50, and repetition state before scaling.
5. Train a cheap frozen-trunk/head baseline and compare it with Lc0 policy and
   classical ordering at multiple small fixed-node budgets.
6. If and only if the pipeline and engine signal survive, generate forced
   reduction counterfactuals, train calibrated error/re-search predictions, and
   test `clip(delta_R, -2, +2)`.
7. Add log-node/log-cycle prediction, then start the on-policy disagreement and
   failure-mining flywheel. Unify tasks only after each proves engine value.

## Superseded D40 policy goal (historical control)

Find a policy-informed configuration that is stronger than `net-0`, preferably
significantly so. Merely gaining Elo over a weak matched evaluator is no longer
success. A candidate is accepted only when all of these hold:

1. Trainer, CENN, and CPOL export/load smoke tests pass; policy scores are
   finite and the engine plays legal moves.
2. Semantic eval sanity passes (balanced start; material-winning signs).
3. **Strong base:** the policy-off candidate is at least near `net-0`, or uses
   `net-0` itself as the unchanged evaluator. A policy gain over an evaluator
   hundreds of Elo below net-0 does not qualify.
4. **Policy-only attribution:** same evaluator with policy enabled beats itself
   with policy disabled. Use paired openings and sweep/calibrate policy weight;
   reject settings whose confidence interval is clearly negative.
5. **Primary strength gate:** policy-on candidate beats `net-0`; target a
   statistically supported positive result, with **+30 Elo** as the initial
   definition of "significant" rather than a lucky point estimate.
6. Reconfirm the best setting on multiple seeds and time controls, then run the
   candidate and historical-reference gauntlets.

This remains useful as historical evidence and a frozen-policy upper-bound
control, but is not the governing optimization target. Its fixed-node-only rule
is superseded by D41's paired fixed-node **curve plus equal-time gate**.

## Platform fixes already made

- Corrected D40 FT layout, mixed 255/64 transformer scaling, and runtime SCReLU
  squares (commit `da75ce8`). Three-way trainer/packed/runtime parity passed.
- Replaced the incompatible legacy policy runtime. CPOL-v2 companions now use
  the active CENN's exact quantized HalfKA+FullThreats D40 trunk.
- Added bilinear and full CPOL packing/loading plus explicit `policy-eval` CLI.
- Added resumable `tools/nnue/d40_campaign.py`, 4x3 config, historical engines,
  per-job/per-match state, smoke mode, and Ordo aggregation.
- Full smoke: 13/13 jobs, 156/156 pairs, 312 games, no crash signatures.
- Production training initially OOM-killed at batch 8192 (host dense policy
  target prefetch). Corrected to batch 2048, 2 loader threads, 2000 batches/SB;
  same 102.4M examples/model. Stable RSS ~2.8GB.

## Generation 0: rejected (`policy_lambda = 1.0`)

Artifacts: `data/nnue/campaigns/d40-policy-4x3/`

- 12 unified-corpus models (4 seeds x eval/partial/full) trained for 102.4M
  examples each; one 49M Bullet-format eval reference also completed.
- All artifact and coarse semantic gates passed.
- Fail-fast games: r1 eval beat r1 partial 18-2 and r1 full 20-0 at 2k nodes.
- Isolation with CPOL disabled proved this was **eval destruction**, not the
  runtime policy: eval beat partial 35-4-1 and full 40-0.
- Root cause: policy CE ~3 versus eval MSE ~0.01; lambda 1 lets policy gradients
  dominate the shared trunk by roughly 300x. The original semantic gate was too
  coarse to detect positional-strength collapse.
- Production gauntlet stopped after 4 clean pairs. One fifth pair is marked
  `running` from the intentional interrupt and must be deleted/restarted.

## Lambda pilots (matched seed, 24.576M examples/model)

Artifacts: `data/nnue/campaigns/d40-policy-lambda-pilot/`

### lambda 0.01

- Partial CENN without CPOL: **+63 Elo** versus matched eval (100 games; result
  from eval perspective was -63). Eval survived and may have improved.
- Full CENN without CPOL: **-494 Elo** (100 games). Rejected for full head.
- Conclusion: direct 4096-logit head sends much larger shared-trunk gradients
  than rank-8 bilinear; a single lambda is not architecture-neutral.

### lambda 0.0001 (in progress)

- Partial CENN without CPOL: exactly 44-44-8 over 96 valid games (0 Elo).
  Four games had connection stalls at concurrency 6 and were excluded;
  FastChess hung in cleanup. Future D40 diagnostics/gauntlets use concurrency 2.
- Full CENN without CPOL: **-49 +/- 59 Elo** versus matched eval (100 games,
  41-55-4 from the candidate perspective). The interval includes parity, but
  the point estimate misses the +/-30 promotion band.
- Same-CENN partial CPOL attribution sweep began at 2k fixed nodes:
  `POLICY_W=1` scored 19-20-1 (**-9 Elo** point estimate); `POLICY_W=5`
  scored 14-23-3 (**-84 Elo**). The sweep was paused after these complete
  weights for the requested net-0 anchor check.

## Direct net-0 anchors (quick 40-game probes)

All use 2k fixed nodes, paired 8-ply openings, one thread, and concurrency 2.
These are coarse probes rather than precision SPRTs, but the gaps are decisive.

- Best LC0 dual-corpus pilot so far, partial lambda 0.01 CENN with CPOL off:
  **0-40 versus net-0**. Classical Elo is unbounded; its 95% score upper bound
  is roughly 7.2%, corresponding to worse than about **-445 Elo**.
- Generation-0 r1 unified-corpus eval-only CENN: **2-38**, **-512 Elo** point
  estimate versus net-0.
- Bullet/SF-corpus D40 reference: **3-37**, **-436 Elo** versus net-0. It is
  the strongest model from the new 13-model campaign, but still far below
  net-0.
- Earlier production `d40-c2-eval` control: **6-33-1**, **-285 +/- 193 Elo**
  versus net-0. This is much better than the campaign reference despite sharing
  the D40 feature/runtime family, strongly implicating training recipe/data or
  checkpoint duration rather than a pure D40 runtime failure.
- Interpretation: the current problem is broader than policy weighting. The
  D40 SF-corpus reference also trails badly, while the LC0 dual-target evaluator
  is weaker still. Before more large policy runs, isolate D40 architecture,
  training schedule/optimizer, eval scale, and fixed-node search interaction
  against net-0 and `d40-c2-eval`.

## Current hypothesis and next loop

Adam mostly normalizes a constant loss scale for policy-head-only weights, while
the small lambda protects the shared trunk where eval and policy gradients mix.
Therefore lambda 1e-4 may preserve eval while still learning useful heads.

The direct net-0 anchors change the preferred architecture: do not require the
policy experiment to replace the proven evaluator. The fastest clean route is
to retain `net-0` as evaluation and decouple policy inference into its own
trunk/head (or freeze a strong D40 trunk), preventing policy gradients from
damaging evaluation. Fixed-node tests isolate move-choice quality; real-time
tests follow only after a positive fixed-node result.

### LC0 weakness: known confounds to separate

- Do **not** use classic `net-1/2/3` as clean evidence about LC0: net-1 used the
  wrong display-Q mapping, net-2 transformed that bad label, and net-3 inverted
  an already clamped display value. The clean raw-Q `net-4`, with net-0's same
  architecture/schedule/WDL blend, recovered about 360 Elo and trails net-0 by
  **132 +/- 20 Elo** over 1000 games. The residual deficit is real but far
  smaller than the confounded 400-560 Elo headline.
- New unified D40 jobs saw 102.4M sample presentations and the campaign Bullet
  reference saw 49.2M, versus roughly 747M for net-0 and production d40-c2.
- The unified trainer uses pure transformed LC0 value targets, no 0.3 game-WDL
  blend, constant LR, and roughly one dataset pass. Net-0 uses the WDL blend,
  cosine decay, and about 30 passes over its 25M-position corpus.
- LC0 positions are dense self-play sequences with no per-game cap in the 100M
  dual corpus; net-0's SF corpus uses random balanced openings, dedup/filtering,
  and much lower within-game correlation.
- LC0 `best_q` transformed by `400*atanh(q)` is an MCTS action-value proxy, not
  a proven calibration to SF static/search centipawns. Median |cp| is only 42.
- The dual-corpus builder used an inferred "1858" map containing only 1771
  entries, missing all explicit promotions and 21 other actions, with index 97
  incorrectly duplicated as `e1c1`. It discarded any row with an unmapped
  active action: **16.37% of candidate positions**, systematically biasing out
  promotion/endgame positions. A tracked canonical generator now reproduces
  all 1858 moves exactly against Lc0 `src/neural/encoder.cc`; corrected 20k-row
  corpus smoke has `dropmap=0` and passes both color orientations.
- The old 16.37% metadata field was mislabeled "bijection failure"; it was the
  incomplete inferred-map drop rate, not evidence that omitted castling rights
  made those positions illegal.
- The unified corpus omitted game outcome entirely, so its eval trainer could
  not reproduce net-0/net-4's 0.3 WDL blend. The corrected builder now stores a
  compact STM-relative result target and the trainer defaults `D40_WDL=0.3`.
- The 100M corpus also had no per-game cap and dropped every position with more
  than 40 actions. Corrected sampling matches clean net-4 (skip opening/check,
  saturation, cap 40 evenly spaced positions/game) and retains high-mobility
  positions by renormalizing top-40 actions. On the 20k smoke, 19.9% required
  truncation but top-40 retained **99.39% mean policy mass**.

### Bilinear policy contract bug: fixed and legacy artifacts recovered

- Bullet matmul output is column-major (`linear = from + 64*to`), but the
  trainer fed a row-major target (`64*from + to`) before flattening. Thus every
  bilinear head learned the **transposed/reversed move matrix**. The previous
  symmetry-only verifier could not detect this because the wrong forward was
  still color/mirror symmetric.
- Before recovery, lambda-0.01 bilinear runtime had only ~4% teacher-top move
  accuracy and policy weight harmed Elo. Transposing the legacy graph cell at
  inference raises it to **23.35% top-1 / 17.66% quiet top-1** over 2k sampled
  corrected rows; generation-0 reaches **33.6% / 26.9%**. Existing CPOL files
  remain useful rather than needing deletion.
- CPOL-v2 byte 7 now explicitly marks bilinear layout: `0` legacy-transposed
  (runtime recovery), `1` corrected canonical. Full heads retain `0` because
  their direct affine output was already canonical.
- The corrected trainer transposes bilinear targets into Bullet's linear matrix
  layout. A 1M-presentation overfit smoke packed as layout 1 and achieved
  **84.6% teacher top-1, 76.8% quiet top-1, legal CE 1.51** through the actual
  runtime oracle. This proves dataset -> trainer -> quant -> CENN/CPOL -> Rust
  inference agreement.
- `tools/nnue/policy_rank.py` is now the mandatory pre-game rank gate. Training
  CE or symmetry alone is insufficient evidence.

### LC0 value calibration audit: factor-of-two target bug found

`tools/nnue/lc0_label_audit.py` is now a tracked, reproducible same-position
audit. On 1,000 corrected-smoke rows freshly searched by Stockfish at 5k nodes:

- Pearson LC0/SF = **0.890**, Spearman = **0.912**, median absolute error
  26.5cp, but MAE 110.6cp and p90 405cp because of hot tails;
- all 374 positions where both labels had |score| >= 50 agreed in sign;
- SF ~= `0.426 * stored_lc0_cp + 0.3` on this sample;
- a missing-castling proxy (king plus home rook, 259 rows) had *lower* MAE
  (55cp) than the other rows (130cp). Castling state is not the explanation
  for the residual LC0 weakness, though the input omission should still be
  documented;
- adding WDL=.3 cut outcome Brier error from 0.0678 to 0.0332, confirming that
  the previously missing WDL blend was material.

A second exact mapping bug follows from Lc0 semantics. `Q=P(win)-P(loss)`, so
expected score is `(1+Q)/2`. The corpus stores `cp=400*atanh(Q)`, but the old
trainer then used `sigmoid(cp/400) = sigmoid(atanh(Q))`; the correct inverse is
`sigmoid(2*atanh(Q)) = sigmoid(cp/200)`. The full 20k corrected smoke outcome
Brier sweep independently bottoms at denominator **200** (0.06227 value-only,
0.03051 with WDL=.3), versus 0.06757 / 0.03311 at the old 400. This factor of
two also affected clean classic net-4 and plausibly explains part of its
remaining ~132 Elo deficit.

`d40_policy.rs` now exposes and prints `D40_EVAL_TARGET_SCALE` and retains
WDL=.3. This is distinct from the engine's 400cp inference output scale. The
updated trainer passes CUDA release `cargo check`. Scale 200 was initially the
mathematical candidate, but the strength A/B below rejected it, so the empirical
default remains 400. Future controls must not conflate target scale with policy
architecture.

For a strict classic-net A/B, `tools/nnue/rescale_bullet_scores.py` now derives
a score-only copy of bulletformat corpora with hashes and transformation
provenance. `data/nnue/net4-qexact/{train,val}.bin` was produced from net-4's
exact 24,750,026/250,000 records with factor 2; 10k sampled records verified
that every non-score byte is bit-identical and every score is exactly doubled.
Ranges are -3962..3962, safely inside i16. Train this with net-0/net-4's exact
30-superbatch cosine/WDL=.3 recipe after the CPU sweeps finish, then compare it
to both net-4 and net-0. This is the clean test of whether the target bug
explains the remaining 132 Elo LC0 gap.

The classic factor-two control trained in unified exec session **26640** as
`net-11-lc0-qexact`: 30 superbatches x 1520 x 16384, cosine
0.001->0.000027, WDL=.3, hidden 256, two loader threads, checkpoints under
`data/nnue/net11_qexact_ckpt`. This matches net-4/net-0 except for the exact
score-only transform.

**Completed/rejected:** training finished in 74.8s at ~10.0M positions/s,
final running loss 0.026001. `net-11-lc0-qexact.cenn` packed with max output
weight 127, passed 78/78 bit-exact Bullet/runtime positions, and produced sane
material scores. But 2k-node 40-game probes were decisively unfavorable:

- versus net-4: 15W-24L-1D, **-79.5 +/-99.5 Elo**, LOS 4.7%;
- versus net-0: 7W-31L-2D, **-240.8 +/-119.5 Elo**, LOS 0%.

The point estimates compose with net-4's known ~-132 Elo gap. Exact expected-
score inversion makes the LC0 target too sharp/noisy for this static evaluator
and search; net-4's denominator-400 target acts as useful regularization. Do
not promote scale 200 or spend more games on net-11. WDL=.3 remains supported
by both Brier and prior strength evidence.

### Truly held-out policy rank gate

A separate corrected 20k-row holdout was built exclusively from raw tar index
20 (`data/lc0_nnue/policy_eval_ds_corrected_holdout`), disjoint from the first-
tar corrected smoke/training sample. It has 10k rows per side, `dropmap=0`, and
median |cp|=47. On the same deterministic 2,000-row sample, the teacher has
mean support 31.39, entropy 1.433, and best-move mass 0.568:

- corrected 20k overfit canonical bilinear: **7.45%** teacher top-1,
  7.10% quiet top-1, CE 5.045. Its earlier 84.6% was memorization and must not
  be used for games;
- legacy lambda-0.01 partial: **19.25%** top-1, 15.55% quiet, CE 3.559;
- generation-0 r1 partial after legacy-layout recovery: **24.75%** top-1,
  20.80% quiet, CE 3.309;
- generation-0 r1 full: **27.25%** top-1, 23.65% quiet, but CE 3.577. It ranks
  the best move most often yet is more poorly calibrated than r1 partial.

This reverses the initial artifact preference: the larger generation-0 heads
are stronger *policy teachers* even though their jointly trained evaluators are
unusable. The independent-trunk design lets us test those heads with net-0 and
is now the fastest route to the primary goal. Run both r1 partial and full
independent sweeps after the current lambda-0.01 screen.

All four generation-0 seeds were also compared on an identical 1,000-row
held-out sample. Partial heads: r1/r2/r3/r4 top-1 = 27.3/27.9/28.7/27.9%, CE
= 3.148/3.162/3.123/3.177; r3 is the best calibrated partial, while r2 has the
best quiet top-1 (22.4%). Full heads: top-1 = 30.9/30.8/30.3/29.6%, quiet =
23.9/25.3/23.7/23.6%; r2 is the best quiet full seed. Use r2/r3 for independent
seed confirmation if r1 produces a positive Elo region.

Generation-0 r1 bilinear independent-policy screen (net-0 unchanged on both
sides) completed weights 1/3/5/10/25/50. Policy-side results were +17/+17/+35/
-35/0/**+215** Elo point estimates. The weight-50 game score was
**13W-2L-5D (77.5%), +214.8 +/-76.4 Elo**. Unlike the legacy weight-5 spike,
this uses the substantially stronger held-out head and has a large margin, but
20 games remains a selection-biased pilot.

A 200-game confirmation is running in unified exec session **25123**, output
`data/nnue/campaigns/d40-policy-independent-net0-gen0-r1-partial-w50-confirm`,
with a new opening seed 20260715. It still uses the pre-depth-gate release binary
(quiet-only, all main depths), so it is a clean reproduction of the pilot.
Do not rebuild `target/release/chessengine` until this match finishes.
At 28 completed confirmation games the interim r1 policy score is 14W-9L-5D
(58.9%, roughly +63 Elo); encouraging, but continue all 200.

A separate r2 partial seed check is running in session **66625** at weight 50,
40 games, opening seed 20260716, output
`data/nnue/campaigns/d40-policy-independent-net0-gen0-r2-partial-w50`.

The legacy 43-line hand book is acceptable for smoke screens but repeats in a
200-game match. `tools/nnue/gen_balanced_openings.py` now generates deterministic
Stockfish-MultiPV balanced books with full provenance. A 256-position, 8-ply
book was generated at `data/nnue/openings/policy-balanced-256.epd`: 270 attempts,
14 rejected outside +/-120cp at 5k nodes, all 256 positions unique/legal/
nonterminal, SHA256 `14bb65c9...f167d5e`. Use this non-repeating book for the
depth-gated and final confirmations; regard the current 43-book match as a
strong intermediate gate whose naive error bar may be optimistic.

Corrected canonical rank-8 training is also running in unified exec session
**73322** against the now-immutable first 10M rows/side of the still-building
corpus. Configuration: lambda 1, WDL .3, empirical target scale 400, 25 x 2000
x 2048 = 102.4M presentations, seed r1, output
`data/nnue/campaigns/d40-policy-corrected-prefix10m/rank8`. An initial attempt
was stopped after <1 superbatch because the callback ignored Bullet's log-rate;
the trainer now actually emits every 100 batches and restarted from the same
seed. After completion: pack CENN/CPOL rank8, parity/sanity, then require the
disjoint tar-20 held-out rank gate before games.

The r1 full head scored 8W-5L-7D (+53 +/-65 Elo) at weight 1, but took roughly
an order of magnitude longer than bilinear. Its weight-3 run was deliberately
interrupted after 13/20 games to fail fast; preserve the partial PGN but do not
resume unless gated bilinear cannot confirm. Full's throughput is not viable.

Runtime observation: at 2k fixed nodes the bilinear independent-policy side is
already much slower than classic net-0, and the full 4096x768 affine head is
several times slower again (only 6 games after ~200 seconds under concurrent
load). Fixed-node Elo is the attribution gate, not sufficient deployment
evidence. `POLICY_MIN_DEPTH` has therefore been added (default 1 preserves
current main-search behavior) and the sweep driver accepts `--min-depth`.
After the ungated screens, rebuild once and test depth gates 2/4/6 plus direct
NPS/real-time controls. Do not claim success from fixed-node Elo if policy's
throughput cost erases it.

The next search variants are implemented but not yet built into the ongoing
ungated-match binary: `POLICY_CAPTURES=1` adds policy to capture ordering,
`POLICY_CAPTURE_W` tunes it separately, and one lazy `PolicyScores` object is
reused between capture and quiet scoring at a node. The sweep driver exposes
all controls. The trainer now accepts `D40_POLICY_RANK` 1..64 (default 8), and
`policy_pack.py --rank` writes the matching CPOL header/payload. Default rank-8
repacking is byte-identical to the canonical artifact; CUDA trainer check
passes. Corrected-corpus pilots should compare rank 8/16/32 before paying for
the prohibitively slow full head.

### Corrected corpus rebuild

The invalid 22GB derived corpus was removed only after preserving its metadata
under the campaign directory; raw Lc0 tars and trained artifacts remain intact.
The corrected 50M+50M rebuild is running from the preserved raw tars with
canonical map, WDL, decorrelation, and top-k retention.

Required isolation experiments: score the same retained positions with SF and
LC0 labels; categorize policy legality/orientation failures; match optimizer,
WDL, and sample-presentation schedules; then train crossed controls (LC0
positions+SF labels versus SF positions+matched targets).

Next actions:

1. Run direct anchors for `d40-c2-eval` and a longer Bullet D40 checkpoint (if
   distinct from the campaign reference) versus net-0; this is now the primary
   architecture/training control.
2. Resume the partial lambda-1e-4 CPOL weight sweep from weight 10 only after
   checking whether weights 1 and 5 already justify fail-fast rejection.
3. For every parity-passing head, run same-CENN CPOL-off vs CPOL-on weight sweep
   at `POLICY_W = 1, 5, 10, 25, 50, 100, 250` (start with 40 paired games,
   promote promising settings to >=200 games).
4. Inspect policy rank/target metrics, not just CE. Add top-k / legal-move rank
   validation on held-out corpus rows if weight sweeps are uniformly negative.
5. If bilinear underfits, try rank 16/32 and calibrated score normalization.
   If full is useful but too disruptive, freeze/detach the eval trunk or train
   the head after eval convergence (preferred clean experimental design).
6. Retrain only promoted configurations across four seeds; archive generation-0
   policy artifacts as rejected, preserving the eval and Bullet references.

## 2026-07-14 pre-compaction handoff (latest)

The user explicitly asked the AutoML loop to continue until a policy-informed
network beats net-0 and then announced an imminent context compaction. Resume
from this section first; do not restart discovery or treat the old 13-network
campaign as the success condition.

### Independent policy trunk implemented

`src/policy.rs` now supports `POLICY_TRUNK=<d40.cenn>`. `PolicyNet` owns an
optional `Arc<SotaNetwork>` loaded and validated by `sota_loader::load_v2`.
Policy scoring uses that independent trunk when present, otherwise it preserves
the generation-0 behavior of using the active D40 `EvalFile`. This enables the
critical clean test:

- evaluator on both sides: unchanged `data/nnue/nets/net-0.cenn`;
- policy-off side: no policy environment variables;
- policy-on side: `POLICY_NET=<head.cpol>`,
  `POLICY_TRUNK=<matching-d40.cenn>`, and swept `POLICY_W`;
- therefore any fixed-node gain is attributable to move ordering, not a weaker
  LC0-trained evaluator.

Environment-load failures now print a UCI `info string` diagnostic rather than
silently disabling policy. Existing CPOL files and the `policy-eval` CLI retain
their fallback behavior. Verification completed:

- `cargo test --features policy policy::tests`: 2/2 passed;
- `cargo build --release --features policy`: passed;
- UCI smoke with EvalFile=net-0 plus the independent legacy partial trunk/head
  reached `bestmove` normally and printed no load error.

Files were formatted with `cargo fmt`. The full feature suite still needs to be
rerun after research-critical changes settle.

### Active experiments at compaction boundary

1. Corrected 100M dual-corpus rebuild is still running in unified exec session
   **14039** (Python PID was 3160276). Last observed after 921 seconds:
   tar 9/86, white 6,171,212 / 50M, black 5,871,257 / 50M, `empty=0`,
   `topk=2,314,018`, `dropmap=0`, median |cp|=47, max |cp|=1981. Poll it with
   `write_stdin`/the saved session and do not launch GPU training before it
   completes and metadata/array sizes are validated.
2. Recovered legacy partial lambda-0.01 same-CENN policy-weight sweep is still
   running in unified exec session **99996**, output directory
   `data/nnue/campaigns/d40-policy-lambda-pilot/policy-weight-sweep-partial-l001-recovered-c1`.
   It uses 20 games/weight, 2k fixed nodes, concurrency 1, and tests
   weights 10,25,50,100,250. Completed observations (FastChess reports the
   policy-off engine first):
   - weight 10: off 10W-10L-0D, hence policy side 50%, neutral;
   - weight 25: off 6W-12L-2D, hence policy side 12W-6L-2D = 65%, point
     estimate roughly +108 Elo, but only 20 games and not yet evidence.
   - weight 50: off 8W-12L-0D, hence policy side 12W-8L = 60%, roughly
     +70 Elo, again only 20 games;
   - weight 100: off 9W-10L-1D, hence policy side 10W-9L-1D = 52.5%,
     roughly +17 Elo and statistically uninformative.
   - weight 250: off 10W-10L-0D, exactly neutral.
   All PGN counts were exactly 20 and session 99996 exited successfully. The
   hump at weights 25-50 is worth follow-up but none of these 20-game estimates
   is a strength claim.
3. Corpus building uses roughly one CPU core and ~11% RAM; a D40-vs-D40 match
   uses two very large engine processes (~20% RAM each). Keep gauntlets at
   concurrency 1 while the corpus build is alive to avoid previous connection
   stalls and memory pressure.

### Immediate resume sequence

1. Poll sessions 99996 and 14039. Preserve all results in this file.
2. Add a tracked, parameterized `tools/nnue/policy_sweep.py` instead of more
   ad-hoc shell loops. Required knobs: evaluator CENN, optional independent
   policy trunk, CPOL, weight list, games, nodes, concurrency, seed, openings,
   timeout, output directory. It must preflight the explicit policy oracle and
   UCI load/play, delete incomplete PGNs before retry (FastChess appends),
   require the exact requested game count, save command/log/JSON summary, and
   resume completed weights.
   **Completed:** the tracked driver now implements these requirements, plus
   policy-side Elo/SE estimates and per-weight resumable JSON. A 2-game/200-node
   independent-trunk smoke passed end-to-end with the exact requested PGN count.
3. Run the decisive independent-trunk sweep with net-0 on both sides, legacy
   `pilot-partial.cenn` + `pilot-partial.cpol`, initially 20 paired games at
   weights 1,5,10,25,50,100,250 and concurrency 1. Promote any positive region
   to at least 200 paired games. This is more important than finishing the old
   same-CENN sweep.
   **Running:** unified exec session **88329**, output
   `data/nnue/campaigns/d40-policy-independent-net0-legacy-partial-l001`.
   Results so far (policy side, 20 games each): weight 1 = 7W-8L-5D,
   -17 +/-69 Elo; weight 5 = 12W-6L-2D, +108 +/-79 Elo. Weight 5 is a
   promising but very noisy signal and must be promoted only if the neighboring
   weights agree or a >=200-game rerun confirms it.
   **Complete:** weights 10/25/50/100/250 were respectively -17/-17/-35/+35/
   -17 Elo point estimates, each with ~74-76 Elo SE. With both adjacent weights
   to 5 negative, the +108 at weight 5 is an isolated spike, not promotion
   evidence. Preserve it but prioritize stronger held-out heads.
4. After corrected corpus completion, verify all array lengths/dtypes, metadata
   counters, result balance, no unmapped actions, and run the updated
   `policy_rank.py` report including teacher entropy/best mass/uniform CE.
5. Train short corrected-layout rank-gated pilots before any long run. Prefer
   policy heads whose D40 trunk is used independently with net-0; use WDL=.3 for
   the auxiliary eval target, controlled schedules, and fail fast on held-out
   teacher top-1/legal CE before games.

Do not mistake the current +108 Elo 20-game point estimate for success. The
primary gate remains a reproducible policy-on win over net-0 with adequate game
count and multiple seeds/configuration confirmation.

## Automation/state locations

- Campaign config: `tools/nnue/configs/d40-policy-campaign.json`
- Campaign driver: `tools/nnue/d40_campaign.py`
- Production state: `data/nnue/campaigns/d40-policy-4x3/state.json`
- Smoke state: `data/nnue/campaigns/d40-policy-4x3-smoke/state.json`
- This file is the compact-safe research decision log; update it before moving
  to the next hypothesis.

## 2026-07-14 active AutoML loop (post-handoff)

The strongest generation-0 independent-policy result is the **run-1, rank-8** partial
head with `net-0` unchanged for evaluation and `POLICY_W=50`. The initial
20-game sweep scored 13-2-5 for the policy side (+215 Elo point estimate), so
it was promoted rather than accepted. Two confirmations are now in progress or
complete:

- seed 20260716, 40 games: policy **20-13-7**, score 58.75%, **+61.4 Elo**
  with estimated SE 51.2. This independently confirms the sign but not the
  initial effect magnitude.
- seed 20260715, 200-game run: at the latest completed-PGN snapshot, 94 games,
  now **complete** at policy **104-65-31**, score 59.75%, **+68.6 +/-23.0
  Elo**. This is a statistically credible fixed-node gain and, together with
  the independent 40-game seed, establishes that the policy signal is real.

Both confirmations use the old 43-line opening book and the pre-depth-gating
release binary. They establish fixed-node ordering signal, not deployable
wall-clock strength. Final evidence must use the new 256-position balanced book
without heavy opening repetition and must measure throughput/time-control
strength.

Active long-running work:

- corrected 100M corpus build: session 14039; at tar 29/86 it held 19,726,217
  white and 18,750,873 black positions, with `empty=0`, `dropmap=0`;
- corrected prefix-10M rank-8 training: session 73322, healthy at superbatch
  15/25, running combined loss near 3.04;
- 200-game run-1/rank-8 confirmation: session 25123 completed successfully.

Do not replace `target/release/chessengine` until the 200-game run finishes.
Then run the feature-policy tests/build and screen `POLICY_MIN_DEPTH` on the
balanced 256-position book. The corrected rank-8 artifact is promoted to games
only if its disjoint tar-20 policy-rank report equals or improves materially on
the generation-0 run-1/rank-8 baseline (about 24.75% top-1 on the prior 2K report).

### Corrected prefix-10M rank-8 result

Training completed in 11m53s (25 superbatches, 102.4M sampled examples), with
running combined loss falling from about 4.54 to 2.888. Packed artifacts:

- `data/nnue/campaigns/d40-policy-corrected-prefix10m/rank8/corrected-prefix10m-r8.cenn`
- `data/nnue/campaigns/d40-policy-corrected-prefix10m/rank8/corrected-prefix10m-r8.cpol`

CPOL load/oracle scores were finite. On 2,000 rows from the disjoint corrected
tar-20 holdout: top-1 26.10%, quiet top-1 21.00%, legal CE 3.0127, MRR 0.4266.
This clears the comparable generation-0 run-1/rank-8 baseline (24.75%, 20.8%, CE
about 3.309), so it is promoted to short independent-policy games. Its CENN
standalone semantic material scores were only +30/-27 and it is **not** an
evaluator candidate; use it solely as `POLICY_TRUNK` with net-0 unchanged.

### Throughput and new root-distance gate

At 50K requested nodes from startpos, the ungated net-0 search reported about
3.2M NPS. Minimum-depth policy gates 4/5/6/8 reported approximately
47K/83K/138K/305K NPS, respectively. This confirms the independent D40 refresh
dominates wall time even after depth gating.

Added `POLICY_MAX_PLY` plus `policy_sweep.py --max-ply`. With min-depth 1,
max-ply 0 was byte-for-byte equivalent to policy-off (the root search itself
does not use MovePicker), while max-ply 1/2/3 reported roughly 234K/145K/67K
NPS. Therefore max-ply 1 is the cheapest currently effective MovePicker gate,
but still has a large throughput tax. Screen its fixed-node retention before a
time-control test. A direct cached root-move prior may be the next architecture
if max-ply 1 loses too much wall-clock strength.

Verification after the depth/capture changes: full `cargo test --features
policy` passed (626 passed, 7 ignored, all integration tests passed); max-ply
targeted policy tests and isolated release build also passed. The match binary
in `target/release` remains untouched while the 200-game confirmation runs.

### Balanced-book efficiency pilots

Using the 256-position balanced opening book, 40 games each at 2K nodes:

- cached root-only generation-0 run-1/rank-8 policy: 20-18-2, **+17 Elo +/-54**;
  inconclusive but cheap enough to retain for a larger/time-control screen;
- interior max-ply-1 policy without the root prior: 13-18-9, **-44 Elo +/-49**;
  rejected. It loses most of the all-node benefit while still costing about
  14x throughput in the startpos benchmark.

The corrected prefix-10M rank-8 head is now running a balanced-book ungated
20-game weight screen at weights 25/50/100. Promote only a coherent positive
region; do not infer Elo from its better held-out rank alone.

## Adopted strength-compute research program

The protected-task joint objective has a **positive** anchor and a detached
teacher:

`L = L_eval + lambda_p*L_policy + lambda_a*(E_joint-stopgrad(E_teacher))^2`.

A negative anchor is forbidden: it rewards evaluator divergence and can make
the minimized objective unbounded. Evaluation is the primary task; policy is
accepted only when the final engine improves without violating eval parity.

The active curriculum is:

1. Freeze the strongest evaluator and train policy from its existing
   accumulator. Require bit-identical evaluation. Produce an Elo-vs-node-budget
   curve over paired openings, multiple policy seeds, and pair-level bootstrap
   confidence intervals rather than relying on one 2K-node result.
2. Measure baseline/policy node-scaling curves and infer the maximum affordable
   NPS ratio: if policy at N matches baseline at kN, require
   `NPS_base/NPS_policy <= k` for equal-time viability.
3. Test conditional inference: missing/untrusted TT move, depth, quiet-move
   count, PV status, flat history, policy entropy/confidence, and cache status.
   Keep captures and quiets separate. The decision criterion is expected
   descendant-node reduction versus inference cost.
4. Prefer a compact legal-move head: one small position embedding plus cheap
   move embeddings/factors (piece/source/destination/promotion/capture/tactical
   flags), with controllable low-rank interaction. Offline rank never promotes
   a model without fixed-node games.
5. After frozen validation, unfreeze only an adapter/reserved accumulator slice,
   use a 100-1000x smaller evaluator learning rate initially, retain the
   positive detached anchor, and log per-layer gradient norms/cosines. Compare
   GradNorm/uncertainty weighting/PCGrad only within this asymmetric contract.
6. Bootstrap from LC0, then fine-tune on Archon search telemetry: final-best,
   beta cutoff, refutation depth, subtree cost, iterative-depth stability,
   score margin, and cost-weighted pairwise ordering. The final target is a
   better strength-compute frontier, not imitation accuracy by itself.

Explicit LC0-evaluator gate: train a policy-free D40 evaluator on the corrected
corpus with WDL blending, using matched data/seed/schedule, then play fair
matches against net-0, net-4, and its policy sister with CPOL disabled. Start
on the stable corrected 10M-per-side prefix; repeat on the completed full corpus
if semantic and short-game gates pass. Current evidence does **not** show an
LC0-trained evaluator beating net-0 (best clean classic net-4 remains about
-132 +/-20 Elo), so this gate must not be reported as already unlocked.

Terminology correction: campaign `r1` means **experimental run 1**, not policy
rank 1. The confirmed +68.6 Elo generation-0 partial head is bilinear rank 8.

### Corrected rank-8 confirmation and factorized frozen head

The corrected-prefix-10M rank-8 independent D40 head's encouraging 20-game
weight-50 result did **not** replicate. Its precommitted 100-game balanced-book
confirmation finished 44 policy wins, 44 control wins, 12 draws: score 50.0%,
**0 Elo +/-32.76** (individual-game approximation). Treat the earlier +70 Elo
as small-sample noise and this corrected head as unproven. This does not erase
the separate generation-0 200-game +68.6 result, but it increases the need for
multiple seeds/books and pair-level bootstrap intervals.

Implemented CPOL kind 4, a frozen-classic factorized head:

`position = SCReLU(W_pos * [net0_stm; net0_ntm] + b_pos)`

`score(move) = b_move + position dot embedding_move`.

The evaluator and its incremental accumulators remain bit-identical. Runtime
computes the small position embedding once and scores only requested/generated
moves through shared `Arc` weights, rather than materialising 4,096 logits.
Trainer, packer, CPOL loader, policy oracle, formatting, and targeted policy
tests pass. The rank-32 smoke artifact is 606,368 bytes and produced finite
scores. Start-position 100K-node throughput was 2.78M NPS policy-off versus
373K NPS factorized, about 7.4x slower but roughly 25x faster than the prior
~15K-NPS classic bilinear head. All-node equal-time use still fails the likely
compute gate; selective invocation and lower ranks remain live if held-out
policy quality passes.

Two frozen-head platform bugs were then isolated:

1. The initial CENN feature-transformer importer transposed the weights into
   output-major order. Bullet sparse matrices are column-major and already
   match CENN's feature-major `[feature][neuron]` payload. Training loss could
   decrease against this scrambled deterministic accumulator, but the exported
   head was random when fed the real engine accumulator. The bad rank-32 pilot
   scored 3.30% held-out top-1; after correcting the import, the controlled
   rerun scored 6.95%.
2. The low-capacity factorized head was trained with a 4,096-way softmax even
   though runtime scores legal generated moves only. This wasted capacity on
   legality. The corrected sparse records contain the legal support (mean
   31.39 moves), so training now masks absent moves with -30 before softmax.
   The matched 20.48M-example masked pilot reached 10.70% held-out top-1,
   9.50% quiet top-1, and legal CE 3.3009 versus uniform 3.3991. This is real
   signal but below the corrected independent D40 head's 26.10% top-1, so it
   is not yet promoted to a large game test. Rank 64 is the next capacity
   probe; a weak marginal gain would identify the frozen eval accumulator as
   the bottleneck rather than rank.

Rank 64 confirmed diminishing returns: 11.15% held-out top-1, 9.45% quiet
top-1, and CE 3.2496, versus rank 32's 10.70%/9.50%/3.3009. Its 100K-node
throughput fell further to 207K NPS, so rank 32 is the frozen-head Pareto point.
At 2K fixed nodes, rank 32 weight 50 scored exactly 50% over 40 games (18 wins
each, 4 draws). A 20-game calibration at weights 250/500/1000 gave the same
52.5% point score at each setting, with enormous pair-bootstrap intervals;
there is no coherent positive region and no justification for a confirmation.
Reject all-node frozen-accumulator deployment: its modest offline signal has
not produced measured search efficiency and its ~7.4x NPS tax is prohibitive.
Retain the factorized runtime for a future policy adapter/shared accumulator.

`policy_sweep.py` now computes deterministic 20K-sample bootstrap intervals
over colour-swapped opening pairs and stores both score and Elo 95% intervals.
This is the required uncertainty unit for future node-budget curves.

The explicit LC0 evaluator gate is active: corrected prefix-10M, eval-only D40,
WDL 0.3, scale 400, campaign r1 seed 13893633, and matched 25x2000x2048
schedule. An accidental trainer-default seed run was caught and stopped before
one percent; the controlled run restarted from scratch. On completion, pack
CENN, run semantic checks, then fair policy-disabled games versus net-0.

### Corrected LC0 eval-only gate: failed, generalization bug isolated

The matched sequential prefix-10M eval-only run completed 102.4M presentations
in 8m29s and packed successfully. Semantic scores passed: start +42cp and
four-queen probes +1547/-1421. Nevertheless it lost **0-40** to net-0 at 2K
fixed nodes on the balanced book, all genuine chess/adjudication losses (no
crashes or forfeits). This is consistent with the earlier LC0 D40 collapse and
decisively answers the immediate gate: corrected LC0 eval-only is not superior
to net-0.

The disjoint tar-20 audit identifies training generalization, not teacher
quality or packing, as the current root:

- LC0 labels vs fresh Stockfish 5K: Pearson 0.916, Spearman 0.918;
- corrected D40 vs LC0: Pearson 0.531, target MSE 0.0280;
- corrected D40 vs Stockfish: Pearson 0.544, Spearman 0.470;
- net-0 vs Stockfish on the same rows: Pearson 0.657, Spearman 0.615.

On first-tar positions the corrected D40 reaches target MSE 0.0110 and LC0
Pearson 0.848; across random rows from the whole first 10M prefix it reaches
0.0128 and 0.826. Thus trainer loss agrees with runtime inference and the model
fits its training prefix, but fails on a same-run shard only a few hours later.
Both normalized source sides fail similarly, ruling out a hidden color/sign
bug. Artifact/export/runtime parity is not the explanation.

The D40 reader previously streamed 2K-row game/tar-contiguous chunks in file
order at constant LR and ended 12% into a sixth epoch. It now uses deterministic
1K-row chunk permutations, paired across normalized sides, with an independent
coprime traversal each epoch; ordinary 2K batches mix chunks. A matched shuffled
prefix-10M A/B is active. Promote it only on disjoint tar-20 metrics before any
games. The full 50M/side corpus remains required if shuffling alone cannot
bridge the cross-tar gap.

### Shuffled prefix-10M A/B: rejected before games

The exact-seed, exact-schedule shuffled run completed 102.4M presentations in
8m30s with final running loss 0.011375, essentially matching the sequential
run. Packing and semantic probes passed (start +55cp, four-queen probes
+1469/-1559). On a random sample from its own 10M/side training prefix it also
fit normally: LC0 Pearson 0.821 and blended-target MSE 0.01308, versus the
sequential run's 0.826 and 0.01279.

It did **not** generalize to the disjoint tar-20 shard: LC0 Pearson 0.512 and
target MSE 0.02832, versus 0.531 and 0.02800 sequential. Stockfish Pearson was
0.504 versus 0.544 sequential. Do not spend games on this candidate. Simple
chunk shuffling is ruled out as the fix; the failure is temporal/domain
coverage or a model/target weakness exposed by that shift. The next justified
evaluator trial is broad-corpus training across the completed 50M/side corpus,
with shuffled traversal and a decaying learning rate, followed by the same
disjoint audit before games.

The corrected broad corpus builder then completed all required rows:
50,000,000 white-normalized plus 50,000,000 black-normalized positions from
84/86 input tars. It reported zero empty-policy rows and zero canonical
policy-map drops; median/max absolute eval are 47/1981cp. Direct tail checks on
both sides found 1,000/1,000 populated rows, finite policy sums in
0.9995..1.0, and legal support sizes 3..40. Metadata records 19.09M top-k
truncations with mean retained mass 0.99394.

A full-coverage eval-only run is active as `corrected-full100m-cosine`: D40,
WDL=.3, scale=400, seed 13893633, shuffled chunks, 125x2000x2048 = 512M
presentations (about 5.12 passes), and cosine LR 0.001 -> 0.000027. This matches
the prefix experiment's presentations-per-row while exposing every broad-
corpus row. The trainer now supports `D40_SAVE_RATE`; a two-superbatch smoke
proved intermediate quantized checkpoints load/write correctly. This run saves
at superbatches 25/50/75/100 so disjoint audits can fail fast before the final
checkpoint. Do not play games until an intermediate/final checkpoint materially
improves the untouched-shard gate.

Important holdout correction: tar 20 is disjoint from prefix-10M but is part of
the new 100M training corpus, so it cannot gate the broad run. A new prospective
holdout was built from sorted source-tar indices 84:86 (2025-12-24 12:17 and
13:17), while the full corpus stopped after index 83. It contains 500K rows per
normalized side; tail validation found all rows populated, policy sums
0.9995..1.0, zero empty/map drops, and median/max |eval| 45/1981cp. On its fixed
1K-row audit, LC0 still agrees strongly with fresh SF5K (Pearson/Spearman
0.915/0.919). Net-0 reaches LC0 Pearson 0.663 and target MSE 0.02585; the old
prefix evaluator reaches only 0.569 and 0.03210. Use this tail holdout—not tar
20—for every full-corpus checkpoint gate.

At superbatch 25 (~102.4M presentations, ~0.5 corpus pass), the broad run
produced the first large offline generalization improvement: on the untouched
tail, LC0 Pearson 0.728 and SF Pearson 0.768 versus net-0's 0.663/0.694, with
blended-target MSE 0.02431 versus net-0's 0.02585. Semantic probes passed
(start -15cp, four-queen +1415/-1469). However LC0/SF Spearman were only
0.631/0.618 versus net-0's 0.665/0.680, and the mandatory engine screen exposed
that this half-pass checkpoint is not remotely ready: **1-19, -512 Elo point
estimate** at 2K fixed nodes. This is genuine progress on coverage but not on
chess strength. Do not mistake aggregate Pearson for the engine gate; continue
training and re-audit/re-screen later checkpoints only when rank/tactical
metrics improve.

Superbatch 50 (~one corpus pass) strengthened the same warning. Untouched-tail
metrics improved to LC0 Pearson/Spearman 0.774/0.684, target MSE 0.02181, and
SF Pearson/Spearman 0.794/0.671. Decisive SF sign agreement reached 0.917,
slightly above net-0's 0.914. Yet its 20-game fixed-node screen was still
**1W-18L-1D (-436 Elo point estimate)**. The failure is therefore not explained
by simple cross-tar coverage or aggregate static-label fit. Search strength
depends on locally coherent scores among parent/child and sibling positions,
tactical tails, and calibration under search; random-position correlation can
look excellent while those properties remain poor. Continue the scheduled run
to observe the learning curve, but prepare a crossed isolation experiment:
same LC0 positions with fresh deterministic SF/Archon value labels, retaining
LC0 visits only for policy. That cleanly separates position-distribution
difficulty from MCTS-Q label extractability.

The crossed isolation has started small: 25K deterministic random positions
per normalized side from the broad LC0 corpus are being relabeled by four
Stockfish workers at 5K nodes. The compact corpus preserves features, policy,
result, original LC0 eval, and source indices bit-for-bit and adds `eval_sf.npy`.
`d40_policy.rs` now accepts `D40_EVAL_NAME`, allowing exact-seed/schedule LC0-Q
versus SF-value sisters with no duplicated preprocessing. Compilation passed;
scale only if the SF sister shows a clear prospective-holdout advantage.

Strategic fallback/next stage agreed with the user: LC0 may serve as bootstrap
rather than final value teacher. The intended flywheel is LC0 policy
distillation -> Archon search telemetry/self-play -> search-native value and
move-order targets -> gated challenger promotion. Guard against self-confirming
collapse with a frozen multi-generation opponent/teacher pool, replay mixing,
fresh held-out openings, a persistent external anchor slice, and promotion by
paired engine games rather than self-training loss.

Do not prematurely conclude that epochs cannot help: superbatch 75 produced a
large strength jump. Untouched-tail LC0 Pearson/Spearman reached 0.798/0.694
and SF 0.807/0.685, now slightly above net-0 on both rank correlations as well
as Pearson; decisive SF sign agreement was 0.918. Its 20-game screen improved
from checkpoint 50's 1-18-1 (~-436 Elo) to **6-14 (~-147 Elo)**. This is still
a clear loss and a tiny sample, but it is a coherent learning curve. Finish the
scheduled run and test checkpoints 100/125; retain crossed relabeling because
it may reach the same local consistency with far less data/compute.

Superbatch 100 improved every untouched static metric again (LC0
Pearson/Spearman 0.813/0.706, SF 0.810/0.710, decisive SF sign 0.922). Its first
independent 20-game screen was 2W-16L-2D (~-301 Elo), but this used a different
opening seed. Repeating on checkpoint 75's exact paired openings produced the
same 6W-14L (~-147 Elo) result as checkpoint 75. Thus the apparent regression
was opening/sample variance, not evidence that the later checkpoint weakened.
Test final 125, then use common-opening confirmation if it merits it.

The 125-superbatch run completed in 43m41s. Final untouched-tail metrics were
LC0 Pearson/Spearman 0.813/0.715 and SF 0.811/0.716. On checkpoint 75's exact
20 paired openings it improved to 7W-12L-1D, **-88.7 Elo point estimate** with
a very wide pair-bootstrap interval. This is the first corrected LC0 D40 that
is plausibly within roughly 100 Elo of net-0, but it has not beaten net-0 and
needs a larger common-opening test before any parity claim.

### Crossed value-teacher isolation: first positive signal

The naive from-scratch 50K-position sisters were rejected: both 20M-parameter
D40s fit their tiny source sample and generalized near-randomly to future tars.
The experiment was repaired by initializing both sisters from the completed
LC0 evaluator and applying the same low-LR 10.24-pass fine-tune; only
`eval.npy` (LC0 Q) versus `eval_sf.npy` (fresh SF5K) differed. Quantized D40
initialization support was added and compiled. An over-specific startup assert
that required row zero to have 32 pieces was also corrected to accept valid
sampled positions.

On the untouched tail, SF fine-tuning moved SF Pearson 0.8109 -> 0.8122 and
decisive sign agreement 0.9187 -> 0.9235; the LC0 fine-tune moved decisive sign
the wrong way to 0.9169. Most importantly, the SF-label sister beat its exact
LC0-label control **22W-15L-3D over 40 games, +61 Elo point estimate**. The
pair-bootstrap 95% interval (-35,+158) crosses zero, so a 100-game independent
confirmation is active. If confirmed, scale the relabeled slice rather than
the learning rate or number of tiny-slice passes.

The independent 100-game confirmation **did not confirm** the point gain:
SF-label fine-tune 44W-49L-7D, -17 Elo, paired 95% interval approximately
-78..+42. Treat the 40-game result as noise. The very low-LR 50K random-position
fine-tune has no demonstrated engine benefit. Run one cheap matched sensitivity
at a stronger LR; if still null, do not scale random relabeling. Build the next
crossed set from search-local parent/child and sibling neighborhoods, because
local score coherence is the hypothesized missing property and random static
positions are a weak test of it.

The stronger-LR sensitivity was also null and overfit: SF-label versus
LC0-label 19W-17L-4D (+17 Elo, paired interval roughly -61..+98), while both
reduced prospective correlation relative to the base. Reject further passes or
larger LR on the 50K random slice. This small isolation was successful as a
fail-fast design but produced no evidence that random SF relabel fine-tuning
improves the engine. A 100-game final-full-LC0 versus net-0 confirmation is
active to establish the new evaluator baseline. The next teacher isolation
should directly sample search-local neighborhoods, or use a substantially
smaller evaluator whose sample complexity matches the relabel budget.

The 100-game final-full-LC0 confirmation decisively corrected the optimistic
20-game estimate: 21W-73L-6D, **-200 Elo**, paired 95% interval about
-282..-131. The corrected broad corpus and five passes recovered roughly 200
Elo versus the earlier ~-400/0-40 failures, but do not approach parity with
net-0. Never promote or characterize evaluator strength from the 20-game
screens; they are routing probes only.

### GCP durable-data factory bootstrap (2026-07-14)

GCP scope is intentionally narrow: generate durable, reusable datasets at CPU
scale and store them in GCS. Training stays on the local GPU. Smoke tests,
schema experiments, short-lived bootstrap labels, tuning data, and one-off
probes stay local. Future cloud submission must declare durable value, default
to dry-run, estimate worst-case cost, use Spot only, and enforce task/runtime
caps beneath a hard few-hundred-dollar monthly ceiling.

Provisioned keyless worker identity
`archon-batch-worker@archon-502500.iam.gserviceaccount.com`. Its effective
roles are Batch agent reporter and logging writer at project scope, Artifact
Registry reader on `archon-workers`, and Storage object viewer + creator on
`archon-main-bucket`. It cannot overwrite/delete objects, administer compute,
or train models. The active operator has Service Account User on this runtime
identity; no service-account key exists.

Created immutable Docker repository `archon-workers` in `us-east1`. Validated
and pushed worker image:

`us-east1-docker.pkg.dev/archon-502500/archon-workers/research-worker@sha256:55bb5dc258c98f3a928143d525ccef024fa600a115dd0eb8873bc45125089a24`

The image pins the Debian 13/Python 3.12 base by digest, all Python transitives
by hashes, and exact ARCHON, Reckless d445b8bf, Stockfish 18, net-0, and oracle
tool hashes. Debian 12 was rejected by smoke test because Stockfish required
GLIBC 2.38 and GLIBCXX 3.4.32. On the corrected image, artifact verification,
all three UCI identities, ARCHON+net-0 search, and the Reckless root-oracle
self-test pass. Reproducible infrastructure lives under `tools/cloud/`.

The first end-to-end cloud data canary then exercised Batch scheduling, Spot VM
provisioning, the custom worker identity, digest-pinned Artifact Registry pull,
Reckless labeling, Cloud Logging, immutable GCS creation, independent download,
and local validation. It used one `e2-standard-2` Spot task with a ten-minute
hard cap and one full-FEN forced-move position. The first attempt failed fast
because the submitted CLI value was the internal name `forced_child_position`
rather than the accepted `child`; logging exposed the exact argparse error and
the corrected no-retry rerun succeeded. This distinction must be covered by
the future submitter's local preflight.

Validated audit objects remain at
`gs://archon-main-bucket/smoke/batch-flow-20260714-2/`: a 1,428-byte manifest
(SHA-256 `a1f3b78f43f692a60acd357071104447c4b8d7178685a5be3d519776fbc00ff4`)
and a 2,481-byte shard (SHA-256
`22391009bae7c9e102f3d094b2323d9fed73f21a0fd84054068bdeacd70227d2`).
The record has complete FEN state except repetition history, pinned Reckless
SHA, one legal move/arm, low and high `forced_child_position` observations,
and stable-singleton acceptance. Both transient jobs were deleted and no VM
remained. Cold provisioning took roughly one minute, so production shard
duration should be tens of minutes to amortize startup while preserving
fine-grained Spot recovery.

### Durable dual-teacher corpus plan (2026-07-14)

The user expanded durable cloud scope. Preserve all local LC0 material (86 raw
tars, approximately 66GB; extracted datasets bring `data/lc0_nnue` to about
87GB), then paired-label approximately 101M selected existing positions with
pinned Reckless d445b8bf and Stockfish 18. Separately generate at least 100M
unique full-state positions from named Reckless self-play, SF18 self-play,
cross-play, and exploratory/disagreement strata, then attach the same paired
labels. Existing normalized rows remain explicitly
`piece_state_bootstrap_only`; native-play rows retain full game history/state.

Existing labeling will be one Batch task array of about 404 independent
250K-row recovery tasks, running at maximum safe Spot parallelism under family
and regional vCPU quota. Native game generation uses deterministic seed-range
tasks of at most 100K candidates each. A C2D/C3D/C4D 250K-row benchmark will
select measured throughput per dollar and 8-versus-16 worker SMT scaling. Bulk
gates are $50 for existing paired labels and $150 for the full native
generation/dedup/label campaign. The complete plan is
`docs/research/cloud-data-campaign.md`.

Local disk is 97% full with only about 8.3GB available. Upload original sources
directly (Batch cannot read local disk), stream position shards without a
second full copy, verify manifests/checksums/download samples, prove a cloud
consumer, and only then delete local raw/redundant arrays. Next implement the
versioned inventory/uploader and streaming 250K-row sharder, followed by the
GCS-aware paired-label worker and machine benchmark. Do not launch bulk compute
before those gates pass.

### Durable labeling campaign active (2026-07-15)

The raw archive is complete in GCS: 86 tars totaling 69,951,662,080 bytes,
plus a SHA-256 inventory. All 70 selected source-array objects are also present
(23,352,310,895 bytes). A real 5K-row cloud canary passed paired Reckless/SF18
labeling and structural validation.

The 250K-row, 5K-nodes-per-teacher benchmark selected
`c2d-highcpu-16` with 16 workers: 278.815 paired rows/s versus 180.681 for
`c3d-highcpu-16`. C2D output was deterministic across 8/16-worker runs and
byte-identical to C3D output. At the current six-VM C2D quota, 101M rows project
to about 16.8 wall hours and $18.4 Spot compute. The parameterized submitter
uses 404 tasks, parallelism six, a 1,200-second task cap, one retry, and refuses
the launch if retry-inclusive worst-case compute exceeds $50. C4D did no work:
its benchmark VM rejected `pd-balanced` during provisioning.

The 101M piece-state recovery sharder is active and resumable. Once its
404-shard manifest commits, submit the digest-pinned C2D Batch array to
`gs://archon-main-bucket/corpora/lc0-selected/v1/dual-labels-r5k-sf18-5k/`.

The historical net-0 corpus is now a second local paired-label campaign. Its
24,951,581 curated SF16 self-play source lines preserve FEN castling, en-passant,
and clocks but not repetition history, so the new schema is explicitly
`fen_state_no_repetition_history`. `relabel_fen_text_paired.py` stores stable
source-line/FEN-hash identity, original SF16 score/result, and full Reckless +
SF18 label records in immutable 250K-row chunks. A 64-row smoke and immediate
resume validation passed. Real 5K-node local throughput was 141/179/185/176
rows/s at 4/8/12/16 workers; use 12. The full local output is active at
`data/nnue/net0-paired-reckless-d445-sf18-n5k` with a projected 37.4-hour
runtime. The superseded local LC0 Reckless-only job was terminated cleanly and
its partial artifact retained.

The first local net-0 checkpoint contains 250,000 validated rows and ran at
192.99 paired rows/s. Its initial raw Pearson values looked alarmingly low
(SF16/SF18 0.347, Reckless/SF18 0.422), but this was a metric bug, not a label
bug: approximately 0.3--0.4% of new searches found mates represented by the
±30,000 mate-score convention. Ten direct re-searches across the shard matched
stored scores and moves byte-for-byte. Excluding mate rows gives SF16/SF18
Pearson/Spearman 0.954/0.992, SF16/Reckless 0.909/0.986, and Reckless/SF18
0.961/0.988. All future sanity reports must report mate incidence separately
and must not mix mate sentinels into ordinary CP correlation.

The 101M cloud manifest completed with exactly 404 shards and 101M rows;
first/middle/last downloads passed SHA-256, shape, source-index, and lineage
checks. The original six-way job was superseded before output when quota was
partially granted to 300 C2D vCPUs. One 18-way single-zone attempt showed
severe zone-C Spot scarcity, so the source manifest was partitioned by ordinal
modulo into three disjoint, hash-verified manifests (135/135/134 tasks) pinned
to `us-east1-b/c/d`, maximum six VMs each. Their individual retry-inclusive
cost caps are $16.47/$16.47/$16.35, preserving the global $49.29 ceiling.
Zone-D capacity began running while B/C remained in their normal provisioning
window. Dry-run fallbacks for the two stalled partitions are ready in
`us-central1` and `us-east4`; current official C2D-highcpu-16 Spot price there
is $0.158432/h and each partition's retry-inclusive cap is $14.26.

The hidden effective limiter was project-global `CPUS_ALL_REGIONS=32`, despite
the 300-vCPU regional/family grants. Global requests for 20K and 300 were
denied; incremental 64 and 96 requests were approved, while 128 was denied.
Stop at the effective 96-vCPU/six-worker limit. Zone D subsequently reached six
running workers. The first five cloud outputs published; the first two
independent full-download reconciliations passed all hashes, embedded source
records, teacher config, row/index counts, move encodings, and searched-node
checks. First measured production throughput was 268.9 paired rows/s.

Native generation implementation is now active under
`tools/cloud/native_game_generator.py`. Its deterministic gzip game records
retain standard-start complete move history, exact sampled FEN, rule-50 and
repetition count, legal/check context, 128-bit state key, full generator search
telemetry, engine/config identity, balanced cross-play colors, and true
MultiPV exploration. Local cross/exploratory/self-play smoke, exact replay,
immediate resume, and byte-for-byte regeneration all passed. A container-only
cross-filesystem `os.replace` failure was caught and fixed before push; mounted
first-write and retry then passed.

At 40 samples/game on the local pilot, generation throughput was 35.7/23.0/18.5
positions/core-second at 500/2K/5K move nodes. A conservative 15 positions/s at
5K projects 100M generation to about 1,852 core-hours and $21.18 expected C2D
Spot compute. The dry-run plan uses 4,000 25K-position tasks, 16 one-core tasks
packed per C2D-16 VM, disjoint stratum task ranges, and $61.00 retry-inclusive
worst case, leaving room under the $150 generation+dedup+label ceiling.

Validated native worker image:
`us-east1-docker.pkg.dev/archon-502500/archon-workers/data-worker@sha256:1787aefd90d65fb5a7f7c249b9663f8a422aebc3a5ded5730ed602e03c1015a6`.
A 100-position E2 Spot cloud publication canary is queued under
`archon-native-canary-v1`; it must pass GCS download/replay before any native
pilot wave is submitted.

### Cloud/data handoff checkpoint (2026-07-15 01:35 EDT)

The first native image above was superseded during preflight. The generator's
`config_hash` incorrectly included `task_index`, making every otherwise
identical task advertise a different campaign configuration. It now excludes
task identity, records task/seed separately, and includes the generator source
SHA-256. Two different task indices produced distinct deterministic outputs
with one shared config hash, and both replay-validated. The old queued canary
was deleted before publishing output.

The current verified worker image is:

`us-east1-docker.pkg.dev/archon-502500/archon-workers/data-worker@sha256:187dacdf4a95ed2333a1c2cefc97cc9eb11548399ca54ac0a93e8ceed43aa4b1`

Its in-container verifier passes 12 artifacts. Future native generation and
dedup submitters default to this digest. Corrected canary
`archon-native-canary-v2` remains scheduled behind the saturated 96-vCPU label
campaign; it has not published output yet.

Native reconciliation and global dedup are now parameterized:

- `reconcile_native_games.py` enforces exact task coverage, metadata/count/SHA
  identity, and optional full replay;
- `native_dedup.py` implements resumable hash-partition mappers, deterministic
  stable representative selection, reducers, and a coverage finalizer;
- `submit_native_dedup.py` emits digest-pinned Spot Batch arrays with retry and
  worst-cost caps.

Disjoint two-map smoke preserved all eight rows. A forced duplicate-source
test collapsed eight inputs to four unique rows, and the finalizer rejected
intentional overlapping mapper ranges. Representative 10M-pilot dry-run worst
cases are $0.18 for 16 mapper tasks and $1.46 for 64 reducers.

The LC0 zone-D production job currently has 29 succeeded, six running, and 99
pending tasks (7.25M published rows). Zone B/C remain scheduled behind quota;
no production task error is logged. A dashboard-observed 4/6 running interval
was a short shard handoff and returned to 6/6 on four consecutive polls. The
native E2 provisioning attempt stopped rather than remaining as a paid idle VM.

The local historical net-0 relabel is alive under PGID `2850611` with five
chunks / 1,250,000 rows at
`data/nnue/net0-paired-reckless-d445-sf18-n5k`.

The raw LC0 tar deletion gate was rerun immediately before cleanup. All 86 GCS
objects matched the local inventory by exact name and size, totaling
69,951,662,080 bytes; the cloud inventory was byte-identical; and first,
middle, and last cloud downloads matched SHA-256. With explicit user approval,
only `data/lc0_nnue/tars/*.tar` was deleted. Extracted arrays and all active
artifacts remain. Free disk increased from 7.5GB (97% used) to 73GB (70% used).
Evidence is in
`docs/research/artifacts/lc0-raw-tars-local-deletion-20260715.json`.

Always-next sequence after compaction/restart:

1. Poll/reconcile the three LC0 label jobs and corrected native canary; never
   duplicate an immutable completed object.
2. When the canary succeeds, download/hash/replay task 0 with
   `reconcile_native_games.py`.
3. Run one small packed multi-task v3 generation pilot and its GCS dedup
   map/reduce before the 10M native wave.
4. Once cloud work is autonomous, return foreground attention to paired value
   calibration, frozen evaluator selection, and the first Archon-native
   consequence-head experiment.

### Matched historical-corpus value isolation (2026-07-15 02:00 EDT)

Foreground network work resumed while both durable label campaigns continue.
The first six immutable local paired chunks were frozen by path and SHA as a
1,500,000-row audit snapshot. `paired_value_audit.py` passed every chunk and
found that ordinary (non-mate) SF18 and Reckless labels agree strongly
(Pearson/Spearman 0.963/0.988, median absolute difference 31 cp). Both are
less extreme than the historical SF16 labels: the direct SF18-on-SF16 slope
is 0.659 and Reckless-on-SF16 is 0.600 over the broad ordinary distribution.
Mate incidence is 0.304% SF18, 0.384% Reckless, and 0.554% for either teacher;
these sentinels are never mixed into CP statistics. SF18/Reckless best moves
agree on 45.0% of all rows at 5K nodes.

New parameterized tools now make the controlled path repeatable:

- `export_paired_value_corpora.py` verifies source/chunk hashes and row/FEN/
  score identity, applies one common representability mask, and uses one
  deterministic FEN-hash train/validation split for every teacher;
- `build_matched_value_bins.sh` converts and hard-checks row-identical Bullet
  binaries;
- `audit_value_nets.py` computes held-out cross-teacher target, outcome, and CP
  metrics through the actual engine evaluator;
- `rescale_value_text.py` creates explicit provenance for score-scale probes.

The frozen export retained 1,491,656 rows (1,342,575 train, 149,081 validation)
and excluded 8,344 rows (0.56%) because at least one of SF16, Reckless, or SF18
was outside Bullet's strict `abs(score) < 2500` contract. All three first binary
shards decoded with zero mismatches. Three matched 768->2x256->1 SCReLU nets
were trained from scratch for ten passes with identical WDL=.3, optimizer,
schedule, positions, and order. Final train loss was 0.01032 SF16, 0.01055
Reckless, and 0.00930 SF18. Every CENN pack passed the 78-FEN independent
bit-exact oracle and output-weight clip gate.

Engine screens corrected an initially attractive but incomplete result. At
2K nodes SF18 beat its matched SF16 sibling 37-21-2 (+95 Elo, paired interval
+17..+176), but at 500 nodes it scored 28-31-1 (-17, interval -101..+64) and
at 10K nodes 17-30-13 (-76, interval -154..-6). Therefore there is no general
SF18-over-SF16 claim from the tiny slice; the effect is budget-dependent and/or
screen-noisy. Reckless versus SF16 at 2K was exactly 28-28-4. SF18 versus the
full 25M-position net-0 at 2K was 6-50-4, approximately -325 Elo, an expected
sample-size/generalization gap rather than evidence that the new labels fail.
On the prospective 149,081 rows net-0 has the best outcome Brier (0.06375) and
SF18-target MSE (0.00842); the tiny SF18 net is 0.07238/0.01003.

A train-only central regression estimated `SF16 ~= -4.12 + 1.219 * SF18`.
The first explicit scale probe multiplies SF18 CP labels by 1.2189956 (only 9
rows clip). It restores evaluator prediction standard deviation from 533 cp to
618 cp, close to net-0's 613, and improves outcome Brier from 0.07238 to
0.06813 while, as expected, worsening loss against the unscaled SF18 target.
At 2K it scored 40-30-10 (+44 Elo, interval -26..+117) versus raw SF18; at 10K
34-32-14 (+9, interval -53..+70). This is directional but unproven. A direct
100-game 10K-node scale-corrected-SF18 versus matched-SF16 gate is active. If
ambiguous, stop tuning this 1.5M snapshot and rerun the frozen experiment on a
larger completed prefix.

Cloud status at this checkpoint: zone D has 35 succeeded, six running, and 93
pending (8.75M durable paired rows); zones B/C remain queued behind the global
96-vCPU limit, with no production error. Native canary v2 remains scheduled.
The local relabeler PGID 2850611 remains healthy at six chunks / 1.5M rows.

### First native-policy and root-arm loop (2026-07-15 02:25 EDT)

The feature-gated D41 `search_control` scaffold was independently reviewed and
its five focused safety tests pass. It remains shadow-only: no search call,
planner, or learned authority exists. Its checked seam rejects representation,
dimension, context, move, action, probability, cost, and non-finite mismatches
to explicit classical fallback.

`root_move_oracle.py` was extended to safely consume existing Bullet FEN text
with a deterministic bounded streaming sampler. A smoke caught a real causal
label bug: Reckless ignores `searchmoves`, but occasionally chooses the
requested move naturally, which made individual arms look supported and mixed
root-depth with forced-child-depth semantics. Auto mode now detects any fallback
within a position and recomputes every arm through one uniform forced-child
path. A generator-source SHA is now part of the manifest, preventing shards
made by old code from validating after logic changes. Repeated two-position
smokes passed resume and every row now has one execution mode.

On the same 16 deterministic full-FEN rows, Reckless d6/d8 produced 470 legal
arms, 0.987 ordinary-score low/high correlation, 34.5 cp MAE, but only 6/16
stable unique best moves. At d8/d10 it reached 0.989 correlation and 28.9 cp
MAE, with 8/16 stable unique and 10/16 stable best; its d10 best agreed with the
earlier d8 best on 10/16. Median high-budget best/second margin was only 17 cp.
These results reject a hard one-hot "correct move" interpretation at shallow
depth and motivate soft/pairwise targets that retain margins and instability.

ARCHON+net-0 true-searchmoves d6/d8 on the identical rows had 5/16 stable
unique labels, and its d8 best agreed with Reckless's d8 best on only 3/16.
This tiny shallow audit is not a teacher-strength verdict, but it decisively
requires producer-domain separation: Reckless labels may bootstrap generic
move quality; reduction, re-search, and final native control must be ARCHON
labels. Audits are reproducible through `audit_root_oracle.py`.

The much larger immediate bootstrap source is already present in every paired
5K-label shard: one Reckless and one SF18 best move. The frozen-classic policy
trainer now accepts a configurable sparse legal-support width (historical LC0
default remains 40; native pilot uses 64). `export_native_policy_npy.py`
normalizes side-to-move to White, mirrors black-to-move squares, verifies the
teacher move is legal, preserves the full deduplicated from-to legal support,
and emits disjoint skip-based splits.

A 200K-position Reckless train / next-20K validation pilot passed. Rank 8 on
the bit-identical frozen net-0 accumulator reached held-out top-1 9.33%, MRR
0.232, and legal CE 3.183 versus uniform CE 3.253. Rank 32 reached 10.12% top-1
and MRR 0.242 but overconfident CE 3.243. Existing LC0-trained frozen rank
32/64 heads scored 9.16%/9.05% top-1 on the same Reckless labels, so the native
head learned a small teacher-specific advantage but not a large one. A 40-game
2K-node routing screen at weight 50 was null: policy 17W-16L-7D, +9 Elo point
estimate, pair-bootstrap interval -108..+127. Do not tune or confirm this tiny
head.

The active discriminating experiment freezes the seven completed local chunks
and exports 800K positions per side (1.6M total) plus the next 50K per side for
validation. Train the same frozen-net0 rank-32 head. Proceed to games only if it
clearly exceeds both the 200K native head and LC0 frozen controls offline;
otherwise diagnose target noise/representation and move to soft independent-arm
targets rather than sweeping policy weights.

Correction immediately after that checkpoint: the 16-row dataset named
`archon-net0-val16-d6d8-v1` did **not** use net-0. The oracle had no general UCI
option plumbing and ARCHON's `EvalFile` default is empty, so it searched with
PSQT. Retain it only as a schema smoke and never use its 3/16 agreement result
as ARCHON+net-0 evidence. `root_move_oracle.py` now accepts repeatable
`--engine-option NAME=VALUE`, records the exact mapping in its config hash, and
the corrected ARCHON pilot uses an absolute explicit net-0 path. This was
caught before any scaled dataset or training.

The 1.6M-position frozen rank-32 head cleared its offline scale gate: on the
next 100K labels it reached 12.39% top-1, MRR 0.270, and legal CE 3.077 versus
10.12%/0.242/3.243 for the 200K head and about 9.1% top-1 for LC0 frozen
controls on the same target. Yet all-node 2K games were again null (17W-18L-5D,
-9 Elo point). Consumer inspection found a concrete integration bug:
`POLICY_ROOT=1` silently did nothing for classic frozen heads because root
sorting called their unavailable board-refresh path before refreshing the
protected accumulator. Root refresh now precedes policy sorting, classic heads
use `classic_policy_accumulators`, targeted tests pass, and a nodes=1 smoke
changed default `a2a3` to the head's top `d2d4`, proving the path executes.

The corrected root-only screen was +23 Elo (28W-24L-8D), but its independent
200-game confirmation was null: 88W-84L-28D, +7 Elo, pair interval -40..+53.
Reject root-only as too sparse. Near-root/high-depth (`ply<=2`, `depth>=4`)
was negative in its routing screen: 20W-28L-12D, -47 Elo, interval -120..+23.
Reject further invocation and weight tuning for hard 5K best-move imitation.
The model learns its teacher offline but has not improved Archon's consumer.

Foreground work therefore moved to consequence data. Root arms now also store
engine-reported search time and NPS in addition to nodes. A corrected explicit
net-0 two-position smoke passed. Active pilot:
`data/nnue/root-oracle/archon-net0-val128-d6d8-v2`, 128 deterministic full-FEN
positions, every legal move, independent cold d6/d8 `searchmoves` arms, four
workers, explicit absolute net-0 `EvalFile`. Audit stability, error labels,
nodes, and time before defining the first reduction/budget-risk tensor.

### First protected-latent consequence result (2026-07-15 03:58 EDT)

The valid explicit-net0 root corpus now contains 1,024 positions and 29,320
independent legal-move arms at cold depths 6 and 8. It has 437 stable-unique
positions plus three singletons; the same best move survives both depths in
47.5% of positions. Ordinary low/high scores correlate at 0.9866 with 87.1 cp
MAE, and the high-depth best/second margin is only 31 cp at the median. These
labels are useful as graded consequences, not as unquestioned one-hot truth.

`build_root_consequence_npz.py` materializes the protected bit-identical net-0
`[side-to-move; other-side]` accumulator, mover-normalized move features,
pre-action root context, score disagreement, tie-aware best-move, and log-cost
targets. Splitting is by source position, never by arm. On the 214-position /
6,219-arm prospective validation split, a nonlinear PCA-32 plus context
baseline improved 50-cp low/high disagreement AUC from context-only 0.549 to
0.637 (Brier 0.233), and best-move top-1 from 8.4% to 11.2% versus a 5.3%
tie-aware random baseline (MRR 0.261). This is real but modest evidence that
the frozen evaluator retains useful search-control signal.

Static root features did not explain subtree cost: nonlinear low-depth
log-node R2 was only 0.085 and high-depth R2 0.007, with median multiplicative
errors about 1.53x and 1.63x. Treat this as a data-contract diagnosis, not a
reason to increase model size. Cost and re-search behavior depend on the
alpha-beta state omitted by root independent arms: alpha/beta, node kind,
TT depth/bound/move, move index and ordering score, actual reduction, prior
best value, history, and whether the scout triggered a re-search. The next
experiment is therefore feature-gated, bounded, shadow-only observational LMR
telemetry from Archon's real search, followed by counterfactual reduction
arms on selected high-information nodes.

The oracle provenance boundary was hardened again: manifests now hash the
generator, engine binary, and every file-valued UCI option (notably EvalFile),
and each shard validates those hashes before work. A fresh one-position
explicit-net0 smoke passed after this change. Never resume an old output with a
changed generator; start a newly versioned dataset.

The scaled 1.6M hard-Reckless-best policy experiment is closed. Despite clear
offline learning (12.39% top-1), corrected root-only confirmation was +7 Elo
with a -40..+53 interval, and near-root use was negative. Do not sweep weights
or invocation thresholds on this target. It validates the pipeline and the
need for consequence/search-native targets, not engine strength.

Live data checkpoint: cloud zone D has 60/134 tasks succeeded, six running,
and 68 pending (15.0M durable paired rows); B/C and the native canary are
queued behind quota without production task failure. The local 12-worker
Reckless/SF18 relabel remains healthy with nine sealed 250K chunks (2.25M
rows) and the tenth in progress. Keep both running.

### Real-search observational telemetry (2026-07-15 03:25 EDT)

A feature-gated `search-control` telemetry seam now records a deterministic,
bounded sample of Archon's actual LMR actions. It is shadow-only and cannot
change reductions, ordering, pruning, or returned values. The hot search does
no I/O: the main worker buffers selected rows and appends them after `go`.
Each row contains the protected net-0 `[stm;ntm]` accumulator, node/move/search
context, actual reduction and scout depth, scout and final scores, separate
scout/LMR-research/PV-research node costs, alpha raise, and beta cutoff. Go
markers carry the root key, and collection manifests hash generator, engine,
net, source, and trace. Tracing is opt-in through `ARCHON_SEARCH_TRACE_PATH`;
normal builds and runs remain inert.

Safety and semantics gates passed: trace-off and trace-on on the same depth-8
position returned the identical 47 cp score, 9,373 nodes, and PV; the release
incremental-accumulator consistency oracle passed; a four-position collector
smoke produced one provenance marker per search and valid 512-value latents.
The audit correctly permits an invoked re-search to add zero nodes when it
returns from TT, while rejecting cost with no invocation. The v2 observational
schema is frozen for the completed corpus. A reviewed v3 adds full parent FEN,
side-to-move, and piece identity for cold counterfactual replay; FEN explicitly
does not restore repetition history.

The scaled v2 corpus is
`data/nnue/search-trace/archon-net0-val512-d8-v1`: 512 independent root
positions, 65,865 sampled real LMR actions, 26,388 internal position keys,
and 234 LMR re-search / 283 alpha-raise / 237 cutoff positives in the
11,609-row root-grouped validation split. Median effective reduction is two
plies; 90th percentile is four. The corpus, tensor, audit, manifest, and
baseline are persisted locally (about 198 MB before the baseline JSON).

On this adequately sized split, nonlinear classical context predicts the
classical controller's observational outcomes well: LMR re-search ROC-AUC
0.878/AP 0.213 at a 2.02% base rate; cutoff AUC 0.880/AP 0.201; alpha raise
AUC 0.865/AP 0.216. It predicts total log nodes at R2 0.332 with 1.20x median
multiplicative error, and scout log nodes at R2 0.369/1.28x. Adding PCA-32 of
the protected accumulator does not improve those behavior targets and mildly
degrades them. It does add a small amount of final score-gain signal (R2 0.083
versus -0.008 context-only). Do not interpret this as "the latent is useless":
these labels mostly ask the model to imitate decisions produced directly from
the same classical context. The discriminating target is whether an alternative
reduction gives the wrong bound, where chess-position information should matter.

Next: build and smoke the v3 full-FEN binary, collect a small v3 corpus, then
run `relabel_trace_counterfactual.py`. That replayer active-mines actual
re-searches, alpha/cutoff events, and near-window scouts and searches reductions
0..3 independently with explicit net-0. Its initial contract is intentionally
"cold replay": every action gets fresh TT/histories, and repetition history is
unavailable. Use it to validate the risk/cost learning path quickly, then move
the best targets into an exact in-engine counterfactual oracle rather than
pretending cold replay is causal parity with the encountered tree.

Updated data status at this checkpoint: cloud zone D is 68/134 succeeded with
six running (17.0M durable paired rows); local paired relabel has ten sealed
250K chunks (2.5M rows). Both remain healthy.

### First counterfactual reduction-risk pilot (2026-07-15 03:32 EDT)

V3 collection passed on 128 new roots and 12,993 observations. It records a
full six-field parent FEN in every row in addition to side-to-move and piece
identity; the audit found 5,148 internal position keys and a 2.36% actual LMR
re-search rate. Search behavior remains shadow-only. Exact source snapshots
were copied into the v2/v3 collection datasets and the counterfactual dataset,
and their SHA-256 values match the frozen manifests. Future collectors copy
their generator automatically.

The first cold replay dataset is
`data/nnue/search-counterfactual/archon-net0-val128-active128-r0r3-v1`.
It deliberately selected 128 actual in-tree re-searches as a danger-boundary
pilot and replayed their moves at reductions 0..3 with a fresh explicit-net0
engine per action. The 384 actions show a large compute/risk frontier:

- reduction 1: median 0.40x reference nodes, 42/128 (32.8%) wrong original
  window relation, median absolute score error 45 cp;
- reduction 2: median 0.161x nodes, 24/79 (30.4%) wrong relation, 59 cp;
- reduction 3: median 0.074x nodes, 18/49 (36.7%) wrong relation, 55 cp.

These high error rates are selection-biased by design and are not a claim
about all LMR moves. They prove the proposed labels are non-degenerate and
expose a strong resource tradeoff. On a root-grouped 79-action validation
split with only 16 wrong-window positives, nonlinear context/action reached
AUC 0.855 and AP 0.598; adding PCA-32 of the frozen net-0 latent improved this
to AUC 0.887 and AP 0.709. Latent+context predicted log node cost at R2 0.694
with about 1.38x median multiplicative error. This is the first direct positive
indication that protected chess features add risk information beyond search
context, but the sample is too small for an engine decision.

Immediate paired-bootstrap correction: that apparent latent lift is not
stable to the comparison model's regularization/hyperparameters. A harmonized
three-way fit (action-only, context, latent+context) on the same 12 validation
roots gave context AUC/AP 0.869/0.675 and latent+context 0.849/0.557. The paired
root-bootstrap latent-minus-context AUC delta was -0.020 with 95% interval
-0.108..+0.095; AP delta -0.118 with -0.311..+0.236. Therefore the pilot
validates the target and context signal, **not** incremental latent value.
Treat the earlier positive point estimate as ordinary tiny-sample/model
variance. The independent 512-root stratified replication is the decision
experiment.

The replayer now supports a deterministic stratified selection contract:
50% actual re-searches, 20% other alpha/cutoff events, 15% scouts within 50 cp
of alpha, and 15% uniform controls, with deficit fill. A 512-root v3 replication
from a disjoint source shard is active; follow it with at least 512 stratified
counterfactual observations, a root-grouped held-out baseline, and calibration.
Only if latent improvement replicates should an exact in-engine oracle and
trainable reduction-risk head become the next large investment.

Replication result: the disjoint 512-root v3 trace produced 57,948 actions.
The stratified replayer selected 512 observations from 295 roots (50% actual
re-search; remaining cutoff/near-window/uniform controls) and generated 1,608
action labels. Reduction 1 used median 0.452x reference nodes with 17.8%
wrong-window labels; reduction 2 used 0.181x with 20.0%; reduction 3 used
0.0848x with 23.1%. The frontier replicates, with lower error than the
danger-only pilot as expected.

On 52 held-out roots / 248 actions / 49 errors, a harmonized paired comparison
gave action-only AUC/AP 0.701/0.299, context+action 0.732/0.319, and protected
PCA-32+context+action 0.740/0.322. The latent-minus-context AUC delta was
+0.008 with root-bootstrap 95% interval -0.059..+0.074; AP delta +0.003 with
-0.077..+0.092. There is no established incremental classification value.
A separate smaller-tree baseline likewise slightly favored context for the
wrong-window class. The latent did improve continuous log absolute-score-error
R2 from 0.558 to 0.616, while context alone predicted log nodes strongly
(R2 0.782; latent was neutral). Calibration is poor and generic PCA/tree
models are not deployable heads.

Decision: stop scaling the cold replay as if more of the same were exact data.
It has validated non-degenerate risk/cost targets and the compute frontier,
but cold TT/histories and absent repetition prevent causal parity with the
encountered search. The next high-value engineering step is an exact
deterministic intervention oracle: rerun the original Threads=1 root search,
match a traced action by root search + `(position key, move, depth, ply,
move index, occurrence)`, override only its scout depth in a separate process,
and compare local and root outcomes across actions. This preserves the TT,
histories, ordering, and search prefix up to intervention. It is feature-gated,
explicit-env only, and must prove reference replay identity before generating
labels. After exact data exists, train the intended low-rank move-conditioned
head rather than drawing conclusions from PCA plus generic trees.

### Exact deterministic intervention oracle (2026-07-15 03:58 EDT)

The exact oracle is now implemented behind `search-control`. A fresh process
reruns the original Threads=1 root search and targets one action by position
key, raw move, node depth, ply, move index, and matching occurrence. It changes
only that action's scout depth and writes a local consequence record. With no
complete override environment it is inert; normal engine behavior and default
builds cannot invoke it. Focused occurrence and trace tests pass.

The reference identity gate is strong: requesting the recorded scout depth
must reproduce exactly the action's alpha/beta, scout/final scores, scout,
LMR-research, PV-research and total local nodes, plus the complete root score,
node count, and PV. A manual reference reproduced all fields and the 3,582-node
root bit-for-bit. Deepening that one scout from 1 to 2 preserved the root
result while reducing its local cost 8->6 and root cost 3,582->3,580 by
avoiding a re-search. An eight-observation smoke accepted 8/8 identities,
including two recovered at occurrence 1.

The first scaled exact corpus is
`data/nnue/search-counterfactual-exact/archon-net0-rep1-exact256-r0r3-v1`:
252/256 identities accepted (98.4%); 230 occurrence 0, 14 occurrence 1, seven
occurrence 2, one occurrence 3. Rejected identities have no action labels.
Among exact reduction-1 actions, 9.5% changed the reference window relation,
14.7% changed the final root best move, 42.5% changed the root score, and
43.7% consumed MORE root nodes than the reduction-0 reference. Median root
node ratio was exactly 1.00, p90 1.164, worst 5.14x. Reduction 2 changed the
best move 13.8% and used more nodes 51.3% of the time; reduction 3 changed the
best 15.5% and used more nodes 53.4%, with a 7.06x worst case. This is direct
evidence for the pivot's premise: local reduction does not monotonically save
global work, because re-searches and downstream tree changes dominate.

The 801-action exact pilot is still too small for a high-dimensional head.
Only 17 wrong-window and 17 best-change positives landed in its validation
split. Generic PCA/tree models overfit: action-only was the strongest
wrong-window baseline (AUC 0.874), while context and latent variants regressed.
Do not interpret that as an architectural verdict. An active 2,048-observation
exact campaign on the 512-root trace is the minimum next scale (expected about
6K actions); keep the engine binary and exact generator immutable until it
finishes. Then audit identity acceptance, train a move-conditioned low-rank
head with root-grouped validation, and compare it to action-only/context under
paired group bootstrap.
### Exact intervention scale gate and first protected-latent head (2026-07-15)

The 2,048-observation deterministic intervention campaign completed at
`data/nnue/search-counterfactual-exact/archon-net0-rep1-exact2048-r0r3-v1`.
It accepted 2,019/2,048 exact identities (98.58%); occurrence recovery was
1,838/134/30/17 for occurrences 0/1/2/3.  The audit found 6,503 labeled
actions from 429 unique root searches.  Reduction zero is exactly identical by
construction.  At reductions 1/2/3, wrong-window rates are 9.81/11.94/14.12%,
root-best-change rates 16.84/17.17/17.80%, and the intervention uses *more*
total root nodes 40.71/42.74/42.94% of the time.  P90 root-node ratios are
1.197/1.209/1.210 and maxima are 5.14x/5.64x/7.06x.  This is now strong causal
evidence for the pivot: a locally cheaper/deeper-reduced scout action often
causes costly global tree changes, so the useful target is action consequence,
not imitation of the classical reduction.

Added `fit_exact_lowrank.py`, a pure NumPy protected-accumulator research head
with root-grouped fit/calibration/held splits and matched action-only,
context/action, and rank-16 latent-by-context ablations.  On 1,426 held actions
from 83 held roots, the three-seed ensembles score as follows.  Action-only:
wrong-window AUC/AP .802/.290, best-change .743/.238, local-research .780/.656.
Context/action: .822/.281, .715/.229, and .867/.783 respectively; it also
improves log-local-node R2 .100 -> .302.  Context gives a paired-bootstrap
significant local-research lift (AUC +.088, 95% root CI +.041..+.128; AP +.120,
+.038..+.206), but not a demonstrated wrong-window/best-change lift.  The
low-rank latent head stopped after only 6--7 epochs and regressed badly:
wrong-window .687/.191, best-change .669/.226, local-research .815/.695.
Versus context it loses wrong-window AUC -.135 (CI -.201..-.065) and AP -.090
(CI -.151..-.024).  This is a rejected smoke architecture/data regime, not a
claim that position state lacks signal: 512 latent dimensions are being fit
from only 429 roots, and only 83 independent roots are held out.

Immediately after that result, a representation-boundary bug was found in the
research readers: traces intentionally store the raw signed accumulator, but
the first head and PCA baseline consumed it directly.  Net-0's actual consumer
applies SCReLU (`clamp(a,0,255)^2`); 52.9% of raw coordinates are negative and
should be inactive.  `fit_exact_lowrank.py` now explicitly applies configurable
SCReLU/CReLU/raw activation (default SCReLU), tensor metadata now says the
activation is the consumer's responsibility, and the classical exact baseline
was corrected likewise.  Treat the raw-latent low-rank loss above as a caught
pipeline smoke failure.  The corrected five-target, SCReLU-activated run is
complete.  Action-only/context/low-rank/residual held AUC/AP for primary scout
wrong-window are .772/.411, .810/.465, .722/.404, and .811/.467.  For local
re-search they are .782/.661, .863/.786, .787/.658, and .863/.786.  The dense
low-rank model still overfits and is rejected.  The protected residual behaves
correctly: it selects only 1/1/2 residual epochs across seeds and stays almost
identical to the frozen context head.  Residual minus context scout-risk AUC is
+.0002 (root-bootstrap CI -.0011..+.0017), AP +.0020 (-.0016..+.0053); no
latent lift is established.  Context minus action-only scout-risk AUC is +.038
(-.008..+.086) and AP +.048 (-.049..+.149), also not established on only 83
held roots, while the local-research lift remains clear.  Broader independent
root coverage is now the binding next experiment.

Cost tensors now also retain signed `log1p(action_root_nodes -
reference_root_nodes)`.  Absolute root nodes include a large identical prefix,
and log root-node ratio dilutes intervention cost by that prefix; the signed
delta is the exact global work saved or added by the one changed action and is
a better learning diagnostic.  Local subtree nodes, absolute root nodes, ratio,
and signed delta are all retained rather than choosing a convenient metric
after seeing results.  The expanded ten-target pipeline smoke passes.

Future head runs now fit conservative per-target Platt calibration only on the
inner root-grouped calibration partition, then report both raw and calibrated
held metrics including Brier score and 10-bin ECE.  This closes another gate
before planner authority: discrimination alone is insufficient for an
epsilon-constrained controller.  The final held roots remain untouched by
training, epoch selection, and calibration.

A second label-semantics audit found that `wrong_window_relation` describes the
post-verification final local result.  That is useful residual failure risk,
but it is not the proposed reduced scout's initial bound-error risk.  The exact
generator now records both explicitly, and old records can derive the scout
target without replay.  Against the full-depth final local reference, scout
wrong-window rates at reductions 0/1/2/3 are 7.33/16.05/21.44/22.85%, versus
post-research final wrong-window rates 0/9.81/11.94/14.12%.  Reduction zero can
still have a wrong initial scout because PV verification repairs it.  The v2
tensor and dynamic multi-head trainer now include `scout_wrong_window` as the
primary error-risk target while retaining final wrong-window, best-change,
score-change, re-search, and costs.  Do not conflate the two risk labels in
future reporting or planner design.

The shadow inference contract and governing doc 19 were updated before any
authority exists: one ambiguous `bound_error_probability` is now separate
`scout_bound_error_probability` and `residual_bound_error_probability`.
Scout risk predicts verification/re-search cost; residual risk is the safety
constraint after configured repair.  All five focused search-control contract
tests pass.  Only the debug test artifact was rebuilt; the release oracle
binary used by the active trace/exact lineage remains untouched.

Added fast root-paired bootstrap tooling in
`bootstrap_exact_head_compare.py`.  The initial sklearn-in-the-bootstrap
implementation was interrupted because repeated validation overhead was
needlessly slow; the replacement uses direct binary AUC/AP calculations and
finishes the full 10K replicates quickly.  Added `--max-per-root` to the exact
selector, included in its configuration hash, so future causal datasets can
trade repeated actions for independent position coverage.  A new 4,096-root
depth-8 v3 trace is active at
`data/nnue/search-trace/archon-net0-val4096-d8-fen-v3-rep2`, using a different
selection seed (small incidental overlap is possible).  Once complete, run a
capped (`--max-per-root 1` initially)
exact campaign and re-evaluate a protected residual/low-rank adapter.  Do not
scale the failed dense low-rank architecture on the 429-root corpus.

That broader trace completed with 470,938 validated v3 LMR observations from
exactly 4,096 unique roots and 184,843 unique internal position keys.  The
generator snapshot hash matches source.  A capped stratified selection produced
exactly 2,048 observations from 2,048 distinct roots with quotas 1,024
re-search / 409 alpha-or-cutoff / 307 near-window / 308 uniform.  The exact v2
campaign is active at
`data/nnue/search-counterfactual-exact/archon-net0-rep2-exact2048-rootcap1-r0r3-v2`
with four nice-10 workers; it passed 108 records in its first 30 seconds.  Keep
the pinned release engine and exact generator immutable until it completes,
then audit/build/train the calibrated ten-target ablations.

The root-capped exact v2 campaign completed in under four minutes with
2,003/2,048 accepted identities (97.80%) and 6,342 action rows.  Unlike the
prior corpus, every selected observation came from a distinct root; all 2,048
root ordinals are unique.  The causal rates replicate cleanly.  At reductions
1/2/3: scout error 16.28/22.93/24.48%, residual final-window error
11.13/13.33/13.90%, root-best change 16.67/17.13/16.24%, and *more* root nodes
41.64/39.84/39.98%.  P90 root-node ratios are 1.248/1.234/1.254; the worst
reduction-1 case is 8.23x.  The calibrated action/context/low-rank/protected-
residual ablation is now active on this higher-independent-sample tensor.

The calibrated unified ten-output prototype completed on 1,289 held action
rows from 402 held roots.  Action/context/low-rank/residual scout-risk AUC/AP:
.762/.336, .790/.337, .738/.288, .790/.335.  Context minus action scout AUC
+.030 has CI -.008..+.065 and AP actually changes -.023 (CI -.088..+.076): no
established primary-risk lift.  Local re-search does benefit from context:
AUC +.075 (CI +.041..+.107), AP +.101 (+.042..+.160).  Conversely context
damages post-verification wrong-window versus action-only: AUC -.111
(-.155..-.072), AP -.153 (-.238..-.081).  Protected residual again adds
essentially zero to context on every classification output (all AUC deltas
within about +/- .001) and selects only 1/2/1 epochs.  Global signed node-delta
R2 is negative for all modes; context predicts absolute root nodes R2 .138 and
local nodes .168 but not the causal work delta.

Important interpretation: this prototype shares one 32-wide hidden layer
across ten tasks, which prematurely violates the staged plan to prove each
target independently before unification.  The strong task-specific positive
and negative transfers make multi-task interference a live confound.  An
independent per-target action/context/SCReLU-PCA baseline is active now.  Do
not deploy or further scale the unified head unless independently successful
targets are later combined under an explicit no-regression gate.

Independent target results sharpen the diagnosis.  A 20-iteration per-target
tree baseline gives scout-risk action/context/PCA32+context AUC/AP
.751/.323, .803/.391, .775/.336.  Context AUC lift is established at +.0518
(root CI +.0147..+.0876); AP +.0684 still crosses zero (-.0277..+.1596).
Adding unsupervised protected-latent PCA trends worse: AUC -.0281
(-.0598..+.0035), AP -.0549 (-.1414..+.0519).  A full 512-coordinate,
inner-calibration-selected L2 logistic probe agrees and is stronger evidence
against a simple hidden direction: context AUC/AP .798/.369, full protected
latent+context .752/.311; latent AUC delta -.0454 has CI -.0811..-.0110.
Thus search context contains real Archon-native scout-risk signal, while the
frozen net-0 value accumulator has not yielded incremental signal through PCA,
linear probing, dense low-rank interaction, or protected residual adapters.

A true single-task nonlinear rerun removes multi-task interference entirely.
Action/context/low-rank/residual scout AUC/AP are .754/.332, .782/.353,
.752/.316, .782/.352.  The low-rank position interaction still stops at epoch
3/3/3; the residual stops 1/1/1 and clones context.  This is now a replicated
representation/architecture limitation, not merely loss interference.  The
next isolation tensor adds mover-normalized raw 12x64 board occupancy and will
compare board+context against protected-latent+context.  If raw board features
help, the protected value accumulator is discarding search-control information;
if neither helps, richer move-conditioned/child features or more causal data
are required before designing a separate control trunk.

Those two final small-data isolations also rejected cleanly.  A mover-normalized
12x64 board linear probe scores AUC/AP .758/.338 versus context alone
.798/.369; board+latent+context falls to .733/.313.  A board-by-context
rank-16 nonlinear head stops at epoch 2/2/2 and scores .752/.324; its protected
residual again stops almost immediately and matches context.  Replacing scalar
from/to with a 161-wide categorical move/action encoding (one-hot source,
destination, moving/captured/promotion piece, flags, reduction) does not rescue
the 2K-root regime: structured context .753/.345, latent interaction .759/.318,
residual .755/.348, all stopping in 2--4 epochs.  The simple 39-field context
head remains stronger.  Therefore do not add a separate trunk based on these
underpowered negatives; first scale independent causal roots by roughly an
order of magnitude, then rerun the same preregistered probes.

Campaign audit reminder: the historical 13-model D40 deliverable itself is
already trained and parity/smoke complete (13/13 models, 156/156 smoke pairs).
Its full production round robin was intentionally stopped after four pairs
because lambda=1 catastrophically destroyed evaluation; state explicitly says
not to treat completing that invalid gauntlet as the current success condition.
The governing D41 pivot and consequence work above supersede wasting games on
those rejected models, while their artifacts and partial production state stay
preserved.

The classical PCA+histogram-gradient baseline for the 2,048 exact campaign was
interrupted after severe CPU oversubscription (24 serial tree fits at 160
iterations were starving the trace collector); rerun it after the trace or
with a deliberately bounded iteration/thread budget.  The neural ablations
and bootstrap above are complete.  The
cloud LC0 zone-D job has 87/134 succeeded, six running, 41 pending = 21.75M
durable paired rows; B/C and native canary remain scheduled behind quota, with
no production task failure.  The local paired relabeler PGID 2850611 remains
healthy with 13 sealed 250K chunks = 3.25M rows.  Free local disk is about
67 GB.  Keep both untouched and monitored.

Later data poll: cloud zone D reached 98/134 succeeded, three running, 33
pending = 24.5M durable paired rows.  The drop from six to three is current
Spot capacity/task handoff state, not a failed task or paid idle instance; B/C
and native canary are `QUEUED`.  Local paired relabel reached 14 sealed chunks
= 3.5M rows.  Disk is 66 GB free.

The deferred classical diagnostic was rerun successfully at a bounded 40
boosting iterations after the neural job released CPU.  On final wrong-window,
action/context/SCReLU-PCA32+context AUC/AP are .806/.269, .836/.271, and
.812/.251; best-change .699/.216, .741/.261, .729/.220; local-research
.779/.653, .875/.795, .877/.801.  PCA latent does not improve causal-risk or
best-change prediction and gives only negligible local-research lift on this
small-root corpus.  Root-node-fraction R2 remains negative for every feature
set (-.014/-.043/-.036), confirming that global cost is the hardest current
target.  The baseline now includes the new scout-risk target for future runs;
the already completed JSON predates that one-line target addition, while the
neural v2 comparison above includes it.

### Return checkpoint (2026-07-15)

The durable cloud relabel campaign has advanced materially. Live GCP Batch
status in `us-east1` is: zone D `SUCCEEDED` with 134/134 tasks; zone B
`RUNNING` with 19 succeeded, 2 running, and 114 pending; zone C `RUNNING` with
18 succeeded, 4 running, and 113 pending. That is 171/404 production shards,
or 42.75M paired Reckless-d445/Stockfish-18 rows at 250K rows per shard, with
six active tasks and no production task failures. The earlier C4D benchmark
is the only failed listed job; it was an unsupported-machine benchmark, not a
production loss. C2D/C3D benchmarks and the native generation canary
succeeded.

Correction: the local historical net-0 relabel had finished only source 0 of
six, not the intended six-source corpus. The six raw sources contain exactly
24,951,581 lines; source 0 contains 4,157,515. Its 17 sealed resumable chunks
are valid, but calling the overall job complete was incorrect. The prior
process/session ended exactly at the source-0 boundary before source 1 began.
The same hash-pinned command has been resumed over all six sources with 12
workers; it first validates the existing source-0 chunks and will then label
the remaining 20,794,066 positions without duplicating accepted artifacts.
Current artifacts occupy about 91.3 MB and free local disk is about 70 GB.

The planned order-of-magnitude causal-coverage trace is being relaunched at
`data/nnue/search-trace/archon-net0-train16384-d8-fen-v3-rep3`, from the
distinct training source `sf16/train/shard-000005.txt`, 16,384 roots, depth 8,
seed 2764062979, trace sample-log2 4. The first launch left only the immutable
generator snapshot when its execution session was lost; that directory was
preserved with suffix `-launch-aborted` rather than deleted. The rerun is an
active unified exec session and must be monitored through initialization,
then audited before selecting the root-capped exact intervention campaign.
The preregistered next scientific gate remains unchanged: roughly 10x more
independent exact roots, followed by the same context, full-SCReLU linear,
low-rank/residual, raw-board, and structured-move probes. Only after that gate
should we decide whether the protected value representation is genuinely
insufficient and warrants a separate search-control trunk.

Operational correction to that trace launch: an older detached collector
eventually began writing after appearing empty, briefly overlapping a clean
restart pointed at the same path. The restart was stopped immediately, both
lineages were terminated, and the entire potentially shared output was
rejected rather than attempting to salvage rows. A four-root smoke in one
clean `uv` environment produced 650 rows in 9.5 seconds. The definitive
16,384-root run now has exactly one collector process and one engine process;
its observation stream is growing normally. Never accept either preserved
launch-aborted directory as corpus input.

Full cloud reconciliation against the 404-shard master manifest found 173
present immutable objects / 43.25M rows / 536,081,974 bytes, all under teacher
config hash `e83063f12b761804d2521494b72b355a4731ca4bf7b681fcab40d30aafcaf814`.
Downloaded ordinal samples 0, 191, and 401 passed CRC plus embedded source,
shape, count, searched-node, and move-encoding validation. Partial partition
manifests must not be passed to the shared-prefix reconciler because it
correctly sees objects from the other partitions as out of scope; use the
404-shard master manifest for live reconciliation.

The corrected native cloud canary is now closed, not merely job-successful.
`archon-native-canary-v2` published one exploratory task containing exactly
100 positions from six games. GCS reconciliation is complete with no missing
tasks; task 0 downloaded and replay-validated under config hash
`e98fd55a6c77ad13b6ae5a398b26be9f92b048b2a4f32a11ff49085e7ac8ffa0`.

### Governing post-causal roadmap (2026-07-15)

The user set the next research order. Finish the active causal experiment and
then continue directly, without the previously proposed planning pause, into an
extremely strong and efficient value-only Archon using Stockfish-style features
and Reckless-like incremental accumulators. Correctness, runtime/export parity,
inference and search efficiency, and classical heuristic quality are required
foundations, with Reckless and Stockfish 18 used as engineering/oracle anchors.
Train and iterate first on the complete local 24.95M historical corpus with
SF16/Reckless-d445/SF18 labels, then scale the proven recipe to durable cloud
data. The practical strength ambition is top-engine competitiveness, including
beating SF15 and plausibly SF16 in fair tests; SF18 may remain ahead but should
not win by an uninformative blowout. Only after that foundation is exhausted
should the value accumulator be frozen and new accumulators/adapters and
consequence heads be trained locally, then at full scale, toward the Archon
self-improvement flywheel. The detailed gates are persisted in
`docs/research/archon-self-improvement-roadmap.md` and govern after the current
experiment. Continue autonomously through B1--B4; the next requested checkpoint
is only after a very strong value-only network has been trained locally and
demonstrated with engine-level evidence.

### Autonomous B1--B4 execution resumed (2026-07-15)

The proposed intermediate planning review is explicitly obsolete. The next
user checkpoint is the completed B4 outcome: a locally trained value-only
network that has passed correctness, efficiency, and engine-strength gates.
At this resumption the corrected, root-capped exact intervention campaign has
3,454/16,000 immutable records after roughly five minutes of active work. Its
four nice-10 workers are healthy and no local paired-relabel workers remain;
the local paired relabel was intentionally paused at sealed chunk boundaries
to give the causal gate the machine. On completion, audit and preregistered
probes run immediately, followed directly by B1 correctness work. Do not
rebuild the pinned release engine while the exact labels are in flight.

### Phase-A causal gate complete; B1 active (2026-07-15)

The corrected exact campaign completed and audited cleanly at 16,000 records
from 16,000 unique roots: 15,756 accepted exact prefix identities and 244
identity rejections. The accepted tensor contains 49,878 exact action outcomes;
the held split has 9,946 rows, 1,639 initial scout-window errors, and 3,146
independent roots. Reduction 0 is an exact root identity by construction.

The scaled result confirms the small experiment and closes the representation
question for this target. Independent HGB context+action reaches ROC-AUC
0.80887/AP 0.42307 versus action-only 0.75420/0.34544. Root-paired 10,000-fold
bootstrap deltas are +0.05467 AUC (95% CI +0.04102..+0.06806) and +0.07763 AP
(+0.04769..+0.10768). Adding PCA-32 of the frozen net-0 accumulator is slightly
worse: -0.00313 AUC (-0.00645..+0.00023) and -0.00446 AP
(-0.02206..+0.01285). Full activated-latent linear, raw-board linear, board
plus latent, nonlinear low-rank latent, nonlinear raw-board, and structured
move encodings all fail to beat context alone. The strongest nonlinear
residual is effectively identical but slightly worse than its matched context
head: -0.00020 AUC (-0.00280..+0.00241) and -0.00427 AP
(-0.01311..+0.00413). Decision: retain search context/action data as the
essential first control representation; do not assume the protected value
latent or a raw-board adapter contributes reduction-risk signal. Richer
Archon-native labels may justify a dedicated adapter later, but Phase C must
prove that independently.

B1 began immediately. The focused D40 SOTA suite passes (39 tests plus the
large release replay), and a missing search-time debug accumulator oracle was
added along with sibling-reuse and HalfKA-only coverage. A concrete CENN-v2
contract bug was fixed: HalfKA-only containers were accepted by the loader but
the runtime indexed an empty threat transformer; the zero-threat channel is
now explicit in refresh and incremental paths. The trainer now hard-fails on
sparse-capacity overflow, supports full streaming feature/capacity scans,
configurable WDL/target scale/score multiplier, deterministic seed, and raw or
quantized initialization. A 10,000-row feature parity smoke passed exactly
(max HalfKA 32, max threats 53); a full 49M-row scan is active. The pinned
Bullet and bulletformat toolchains were restored durably under
`.archon-state/toolchains` at commits `cebc78a093d9` and `cb6a3602d70b`, and
the modified CUDA trainer compiles.

Cloud status during this transition: D is complete 134/134, B 34 succeeded/2
running/99 pending, C 47 succeeded/4 running/84 pending: 215/404 shards or
53.75M durable paired rows, with no production failures. The local paired
relabel remains intentionally paused at 21 sealed chunks / 5,157,515 rows
until the CPU-sensitive B1 gate permits resumption.

### B1 closed; B2 measured baseline (2026-07-15)

B1 is complete. The full sparse-feature scan verified all 49,452,091 corpus
rows plus self-check exactly; observed maxima are 32 HalfKA and 64 threat
features, within the now-enforced trainer capacities. The full SOTA all-target
Rust suite passes (627 library tests, 7 ignored, plus every integration target),
including the new search-time accumulator refresh oracle, sibling-reuse replay,
large threat replay, and HalfKA-only CENN-v2 coverage. The UCI golden was
corrected only for the intentional engine rename to ARCHON. Pinned Bullet and
bulletformat revisions build successfully from `.archon-state/toolchains`.

A deterministic 4,096-position parity sample was drawn from all 24,951,581
historical local rows at `data/nnue/parity/d40-local-4096.fens` (SHA-256
`1480e79ffb0e8ea99370724d19ddf212f5aae0531e80bac9efd686f96f56d323`).
The existing D40-C2 evaluator proves that the Rust packed runtime follows the
trainer's quantized forward pass closely (mean/max 0.00719/0.16356 logits;
p95/p99 0.02986/0.05515), but raw-to-quantized drift is too high for the B4
recipe (mean/max 0.18376/1.93345; p95/p99 0.57505/0.84988). Do not hide this
with a loose gate: improve quantization-aware training/scales or architecture.

B2's fixed-node baseline on the default 16-position suite at 50k nodes is
approximately 2.88M NPS for net-0 versus 0.506M NPS for D40-C2, a roughly
5.7x regression. The primary implementation gap is now concrete: SOTA HalfKA
updates, threat updates, activation/sparsity discovery, and the dense head are
scalar, whereas the Reckless reference fuses accumulator deltas and vectorizes
the forward path. B2 therefore proceeds by adding independently tested,
runtime-dispatched SIMD kernels and retaining only measured improvements.

### B2 inference gate and HalfKA promotion (2026-07-15)

The SOTA path now has runtime-dispatched, scalar-pinned AVX2 kernels for fused
HalfKA deltas, fused i8 threat deltas, pairwise activation, and the sparse L1
dot product. A correctness bug was found and fixed in the old sparse head: its
four-product group dot wrapped through i16, while Bullet's affine graph and
Reckless's dpbusd design accumulate in i32. Full-domain scalar/SIMD tests,
incremental/refresh replays, and the large release replay all pass after the
correction. Compact occupancy-neutral move and mutation primitives also replace
the worst first-cut FullThreats replay cases without changing refreshed values.

The optimized full D40 path improves from roughly 0.506M to 0.64M NPS on the
2.35M-node fixed suite. A matched eight-position 200k-node probe gives Archon
FullThreats a 597k geometric-mean NPS versus Reckless 458k and Stockfish 18
1.485M. Archon is already about 31% faster per searched node than the pinned
Rust reference on this workload, though Stockfish remains substantially faster.

The architecture isolation is decisive for the value foundation. Stripping
the threat transformer from the identical D40 head increases the suite to
about 0.98M NPS, proving FullThreats currently costs roughly 35% of throughput.
`D40_HALFKA_ONLY=1` now trains the canonical SF HalfKAv2_hm path while retaining
an identically zero compatibility tensor, and `train_run.py` can strip that
tensor into a loader-valid HalfKA-only CENN-v2. The 524,288-position smoke
trained at about 1.0M positions/sec and passed 4,096-position parity: raw-to-
quantized mean/max 0.01183/0.17212 logits and packed-runtime mean/max
0.00465/0.09371. This is dramatically better than the threat model's
0.18376/1.93345 raw quantization drift. Promote exact HalfKA first; FullThreats
must later prove enough equal-time Elo to repay both its NPS and quantization
cost.

Cloud background status at this gate is 233/404 immutable shards: D 134/134,
B 40 succeeded/2 running/93 pending, C 59 succeeded/4 running/72 pending, or
58.25M durable paired rows. Spot capacity/quota events continue but there are
no production task failures.

### First full HalfKA screen rejected; local-coherence gate added (2026-07-15)

The first full exact-HalfKAv2 local SF16 run is not a B4 candidate. It was
stopped after superbatch 15 because fail-fast games became decisively worse:
SB5 scored 4-34-2 against net-0 at 5k fixed nodes (about -338 Elo), and SB10
scored 2-37-1 (about -470 Elo). Output-scale probes did not rescue it. This is
not a sign/perspective error: Bulletformat canonicalizes black-to-move records
by flipping the board, swapping colors, and negating score/result, so the
canonical White perspective is side-to-move. Candidate losses were also
symmetric by game color.

The failure exposes why random held-position metrics cannot be the engine gate.
On 149,081 matched held positions, SB10 improved over net-0 against SF16
(target MSE .00608 versus .00874; Pearson .94643 versus .94619) and SF18, yet
lost catastrophically in search. Across every legal child of 1,000 sampled
positions, quiet-move local residual `-E(child)-E(parent)` had mean/p95/p99
absolute error 110/338/579 cp for SB10 versus 94/224/316 cp for net-0;
candidate-vs-net-0 residual disagreement was 101/293/529 cp. The degradation
affects every moving piece and is largest for queens and kings. Capture
transitions were only modestly worse, and side-to-move flip anti-consistency
remained high, isolating the main defect as quiet local score topology rather
than global calibration, sign, or capture-bucket crossings.

`tools/nnue/audit_local_coherence.py` is now the permanent pre-game checkpoint
gate. Next run the exact classic net-0 recipe with a pinned deterministic seed
on the same corpus. If it reproduces strength and coherence, the data and
trainer loop are exonerated and the D40 pairwise/bucketed head is the causal
architecture suspect. Only then introduce HalfKA incrementally into a simpler
SCReLU head or add explicit teacher/local-difference anchoring; do not tune
search heuristics around the rejected evaluator.

### Classic causal control passes; width screen active (2026-07-15)

The classic recipe control closes the isolation. A first run accidentally used
the same 24.70M SF16 records rebuilt in seed-3 order; it retained net-0-like
local coherence but screened about -70 Elo (12-20-8, paired interval
-168..+17). The historical `net-0` recipe actually consumed the seed-100
artifact at `data/nnue/train.bin`. Repeating all 30 superbatches on that exact
artifact scored 14-14-12, exactly 50%, against net-0 at 5k nodes. Its quiet
transition mean/p95/p99 are 81.2/208/303 cp versus net-0's 93.6/224/316, and
its mean residual disagreement with net-0 is only 17.9 cp. Export parity is
bit-exact against Bullet. The old CENN bytes do not reproduce exactly because
the original corpus byte hash was not recorded (a provenance defect), but the
engine behavior does.

Two orchestration bugs found by the smoke were fixed centrally: numeric
`--set` environment overrides are now stringified before `subprocess`, and the
classic output-weight verifier now reads the exact `l1w` tensor rather than a
tail window containing unrelated weights/padding. The trainer has an explicit
`NET0_SEED` and parameterized `NET0_HIDDEN`; the engine and packer now support a
mutually exclusive 1024-wide classic control. Its compile/tests and one-pass
pack/load smoke pass. The full 1024 run is active. Compare it to the 512-wide
SF control at fixed nodes and equal time with separate width-matched binaries;
do not promote from loss alone.

### Width and feature-expansion controls; calibrated continuation active (2026-07-15)

The classic 1024-wide control is rejected: it scored 12-25-3 against the
512-wide control (about -117 Elo, paired interval -203..-44) despite its lower
training loss. The local 24.95M corpus does not justify a fourfold classic
parameter increase. Width alone is not the missing capability.

A clean HalfKAv2-to-classic-SCReLU CENN-v3 trainer, packer, loader, incremental
runtime, and independent parity oracle were built to isolate the transformer
from D40's pairwise head. The original 768-wide random HalfKA model passed
4,096/4,096 packed-runtime parity but again damaged search as training
continued: roughly -636/-191/-179/-127/-319 Elo at SB1/2/4/5/10 in fail-fast
screens, with SB10 quiet residual 125/329/642 cp. This proves D40's pairwise
and bucketed tail was not the sole cause. A large king-conditioned transformer
trained from random initialization can fit static labels while destroying the
local topology alpha-beta consumes.

An attempted exact HalfKA expansion of net-0 exposed a representation fact:
SF HalfKA horizontally canonicalizes around the king, while Chess768/net-0 is
not constrained to horizontal symmetry. One HalfKA row therefore aliases two
distinct net-0 rows, making exact function preservation impossible. The bridge
was corrected to `64 perspective-relative king buckets x Chess768`, 49,152
inputs and a 256-wide classic head. This 12.6M-weight representation preserves
both horizontal orientations. Its zero-LR expansion is independently
bit-exact against its quantized weights and exactly equals net-0 on 4,096/4,096
positions. A paired 20-game fixed-node check scored exactly 50% pair by pair.
The runtime now reuses the proven AVX2 classic forward kernel. One 24.9M-row
pass at LR 1e-5 preserves coherence almost exactly (quiet residual
93.3/223/318 versus 93.6/224/316; residual disagreement 1.7/5/9 cp) but its
40-game screen was -26 Elo with a wide interval. LR 3e-5 increased tail drift
and screened -53 Elo. Do not escalate the king-conditioned branch until a
better label curriculum exists.

The classic trainer now supports a bit-exact `NET0_INIT` continuation and an
independent LR-decay horizon. A zero-LR round trip reproduces the shipped
net-0 CENN byte-for-byte (SHA-256
`60138d33b5fe1f1c6345e4d77a3c88636d75634b0f6583ffb133e12178709111`).
This is the protected local fine-tuning path.

The first matched-label isolation revealed a major scale trap. On the same
1.5M positions, ordinary SF18 and Reckless labels agree closely with each
other (SF18-on-Reckless slope about 1.005), but their slopes against historical
SF16 are only about 0.659 and 0.600. Raw SF18 continuation monotonically
shrinks net-0's output and loses games: epoch 2/4/10 screened about
-17/-44/-61 Elo at 5k nodes. This is not evidence that the newer teacher is
weaker; their UCI centipawn amplitudes are not directly interchangeable, while
Archon's search thresholds are tuned to net-0's SF16 scale.

Scale mappings were frozen from the matched training audit and materialized as
immutable derived corpora: SF18 x1.518258453 and Reckless x1.667383915, each
with source/output hashes and score-offset provenance. Ten-epoch net-0
continuations at LR 1e-4 preserve amplitude. The calibrated SF18 epoch-10
candidate screened +17 Elo (16-14-10) while calibrated Reckless screened -9
Elo (13-14-13); both intervals remain wide. A 200-game calibrated-SF18
confirmation at 5k nodes finished at 73-76-51, about -5 Elo with interval
-47..+37: calibrated continuation preserved net-0 but did not improve it.

The completed local paired subset was expanded from 1.5M to all 21 sealed
chunks: 5,157,515 input rows, 5,129,121 jointly non-mate/representable rows,
4,616,292 deterministic training rows and 512,829 held rows. The scale result
is stable on this larger sample: SF18/SF16 slope 0.66014, Reckless/SF16
0.60107, and SF18/Reckless 1.00526. Derived scaled targets use x1.51484 and
x1.66369 with immutable provenance. A 256-wide SF18 continuation remained
near parity across checkpoints.

A more promising protected-capacity experiment expands net-0 from 256 to 512
classic neurons. The old 256 neurons and output weights are copied exactly;
new transformer neurons receive deterministic small features but exactly zero
output weights, so the initial function remains bit-identical while the new
capacity can wake during training. The zero-LR 512 model equals net-0 on
4,096/4,096 values and every pair in a 20-game search check. On the 4.616M
scaled-SF18 corpus, epoch 6 screened +61 Elo in 40 games and confirmed at
86-74-40 over 200 games: +21 Elo, interval -23..+65, at 5k nodes. At 20k it
scored 27-31-42, about -14 Elo with interval -60..+31. The gain is therefore
not yet robust across search budgets. Its measured search NPS is 2.878M versus
2.927M for net-0 on 16 matched positions, only a 1.7% tax, so width itself is
not an equal-time blocker.

The resumable local dual relabel was restarted over all six historical sources
with 12 workers in unified session 2494. It first reconstructs the manifest by
validating existing immutable chunks (the manifest is temporarily partial
during this pass), then continues after the prior 5,157,515-row boundary.
Do not interrupt it merely because the live manifest temporarily lists fewer
than 21 chunks; verify artifacts/process state. Once new chunks accumulate,
repeat the protected 256/512 curriculum on increasing data without rerunning
the failed raw-scale recipes.

### SF18 WDL-target isolation rejected (2026-07-15)

Two direct attempts to extract a better target from Stockfish 18's reported WDL
distribution failed engine confirmation. `tools/nnue/derive_stockfish_wdl_bins.py`
materializes a Bullet score whose sigmoid is the SF18 WDL expectation while
preserving the original terminal-result byte. The converter now separates the
input-retention threshold from the output clamp, uses a defensible 0.5-per-mille
probability floor, and records complete source/output provenance.

The nominal `400 * logit(P(win) + 0.5 P(draw))` target was far too sharp for
net-0's established search scale: its checkpoints screened about -80, -70,
-98, and -137 Elo at epochs 2/4/6/10. A streaming regression over 4.87M central
records found SF16-on-WDL-logit400 slope 0.16330 with correlation 0.9351, giving
a temperature-matched scale of 65.3192. That calibrated continuation preserved
score amplitude but did not preserve strength: a tempting epoch-10 20-game
screen reversed in a 100-game confirmation, 29-45-26, about -56 Elo with a
paired interval of approximately -111..-3.5. Reject both WDL-target branches.
Global score scaling and direct WDL expectation are therefore insufficient;
short screens must not override confirmation matches.

The next protected-label experiment is a calibrated SF18/Reckless consensus.
If consensus continuation does not robustly beat net-0, prioritize a frozen
net-0 base plus trainable residual capacity so improved labels cannot corrupt
the evaluator topology already known to work in alpha-beta search.

### Calibrated teacher consensus preserves parity but adds no strength (2026-07-15)

`tools/nnue/blend_bullet_scores.py` now constructs an immutable score-only
blend of aligned Bullet corpora, verifies every non-score byte, and records
input/output hashes and normalized weights. The 4,616,292-row experiment used
an equal blend of SF18 x1.51484 and Reckless x1.66369. A protected 256-wide
net-0 continuation at LR 1e-5 screened approximately +26/0/-89/-35 Elo at
epochs 2/4/6/10, with no monotonic signal. Quiet residual disagreement from
net-0 at epoch 2 remained small (mean/p95/p99 about 7/19/31 cp), so this was a
clean label test rather than another catastrophic topology failure.

The apparently best epoch-2 checkpoint failed confirmation: 77-85-38 over 200
games at 5k fixed nodes, about -14 Elo with paired interval -60..+31. Reject the
consensus as an improvement and close the scalar-label-transform loop. Raw
SF18, calibrated SF18, calibrated Reckless, direct WDL, temperature-matched
WDL, and calibrated consensus have all failed to robustly improve an unfrozen
net-0. The next experiment must enforce architectural protection: bit-identical
frozen net-0 base plus an initially zero, separately trainable exact-HalfKA
residual. The combined evaluator must pass exact zero-residual value and search
gates before the residual receives any authority.

### Protected exact-HalfKA residual foundation passes (2026-07-15)

CENN-v4 now packages a validated CENN-v1 evaluator with a separately quantized
exact Stockfish HalfKAv2_hm residual. The engine maintains both accumulators and
adds the residual to the classic value before applying the outer evaluation
shape. `tools/nnue/bullet/halfka_residual.rs` trains the summed objective while
freezing all four base tensors by name. The small Bullet extension is preserved
as `tools/nnue/patches/bullet-frozen-weights.patch` with an idempotent installer;
the trainer asserts every frozen tensor remains bit-identical at every save.

The zero-residual proof passed all gates: frozen tensors exact, packed CENN-v4
equal to net-0 on 4,096/4,096 values, exact-HalfKA incremental replay equal to
fresh refresh across 48 randomized plies, and every pair in a 20-game fixed-node
match scored identically. This is the required protected-adapter invariant.
The current straightforward dual-accumulator implementation measures about
0.851M versus 1.023M NPS under the active relabel CPU contention, a 17% tax;
the absolute rates are contention-depressed but the matched ratio is useful.

On the existing 4.616M calibrated SF18/Reckless consensus rows, LR 1e-4
underfit (residual output weights only +/-1 after ten passes) and all 40-game
screens remained around parity. LR 1e-3 fit much more strongly, but local quiet
residual disagreement grew from mean/p95/p99 17/46/66 cp at epoch 2 to
41/109/157 at epoch 4, and the corresponding screens were about -9 and -26 Elo
with wide intervals. The protected base prevents catastrophic evaluator
corruption, but the current small matched corpus still has not yielded a useful
correction. Do not spend more cycles on scalar transforms of these same rows.

Cloud reconciliation currently shows 262/404 immutable label shards (about
65.5M rows) available, while the local resumable relabel reached chunk 22 / 
5,407,515 rows and remains active. The next scale gate should use a materially
larger, position-diverse subset of the cloud-native SF18/Reckless labels with
the protected residual, rather than repeatedly fitting the same 4.6M records.

### Protected HalfKA cloud-20M scale gate (2026-07-15)

Downloaded 80 evenly distributed completed cloud shards and added
`tools/nnue/export_cloud_value_bins.py`, which verifies the immutable position
hash, paired-label input hash, and exact `source_index` alignment before using
the official Bullet converter. A deterministic split-index bug was corrected
before the bulk export: the split key is now the manifest `global_start` plus
row order, not `global_start + source_index`. The resulting calibrated
SF18/Reckless consensus corpus has 19,794,850 train and 199,929 validation
records. Because the durable LC0 position corpus intentionally lacks game
outcomes, the result field is a neutral placeholder and this experiment trains
with WDL=0.

The protected exact-HalfKA residual trained for ten full passes at LR 3e-4 in
84 seconds (roughly 2.4--2.5M positions/s); the frozen net-0 tensors remained
bit-identical at every checkpoint. Loss fell from 0.010871 at pass 1 to
0.007786 at pass 10. The least disruptive checkpoint was pass 4, but at full
residual authority its quiet parent/child disagreement from net-0 was still
mean/p95/p99 51/132/190 cp. It improved a 30k-row cloud-label sample from 142
to 124 cp MAE while compressing score standard deviation from 313 to 257 cp.
This is another direct demonstration that shallow-teacher label fit is not an
engine-strength metric. The full-authority checkpoint lost 2-36-2 at 5k fixed
nodes, approximately -436 Elo, and is rejected.

`pack_protected_halfka.py` now supports a bounded `--residual-scale` by scaling
only the quantized residual output layer and bias. This preserves the learned
full-input direction while limiting its authority. The pass-4 10%/25%/50%
variants have quiet disagreement means 2.3/17.7/37.1 cp. Initial 40-game 5k
screens scored 18-14-8 (about +35 Elo, CI wide) for 10% and 15-15-10 (exact
parity) for 25%. This is a dose-response result, not yet a strength result.
Confirm the 10% candidate with more paired games and multiple node budgets;
remember that the current straightforward full-input runtime carries about a
17% NPS tax, so a tiny fixed-node gain cannot pass the final equal-time gate.

### Search-label quarantine and SF18-only quality pivot (2026-07-15)

The cloud and local 5K search-label jobs were stopped after the bound audit.
Completed 5K shards are quarantined from pointwise scalar-value losses. They
remain eligible only for policy/difficulty analysis or an explicitly censored
bound-aware objective. Reckless does not provide UCI WDL in this build: its
stored `[65535,65535,65535]` values are missing-value sentinels, not an equal
three-way distribution.

Static-teacher instrumentation was then audited. The original `evalraw` pilot
had ambiguous cross-engine semantics: Stockfish emitted its scaled static
evaluation through a position-dependent UCI-cp conversion, while Reckless
emitted raw NNUE internal units. Version 2 now emits named key/value components.
Stockfish records final internal value, final UCI cp, big-net PSQT/positional/sum
internal values, big-net UCI cp, and rule-50 state. Reckless records raw NNUE
units, a deterministic cold material/rule50-scaled value, and rule-50 state.
The production script defaults to Stockfish only; Reckless is optional. The
patches are preserved under `tools/nnue/patches/*-evalraw-v2.patch`, both engines
compile, and a mixed-state smoke proved rule-50 propagation, big-net
decomposition, and joint exclusion of in-check positions. The old 1M v1 static
pilot is not promoted; a 100-row v2 end-to-end relabel/export/resume smoke passed.

`tools/nnue/benchmark_search_label_stability.py` now measures exact completed
iterations and checkpoints atomically. Early depth-14/16/18 results already
showed material score drift despite exact bounds, confirming that a completed
shallow iteration is insufficient. A nominal depth-22 run exposed an extreme
cost tail, and then exposed that Reckless prioritizes a simultaneous UCI depth
limit over a node limit. The corrected benchmark sends only a hard node limit
and stops client-side only after receiving an *exact* completed target-depth
record; aspiration lower/upper records cannot trigger acceptance. Partial
pre-correction audit files are diagnostic only. Per user direction, the next
frontier run is Stockfish 18 only. Its output will determine tier depths, node
ceilings, stability tolerance, yield, and quarantine rate before any bulk job.

### SF18 high-quality oracle frontier measured (2026-07-15)

The governing plan is `docs/research/sf18-label-quality-plan.md`. A deterministic
4,096-candidate pool was selected across all 24,951,581 local full-FEN rows, then
stratified to 256 audit positions: 76 opening / 84 middlegame / 96 endgame and
76 balanced / 78 advantageous / 102 decisive, including 89 castling-rights,
both available en-passant, eight rule50>=50, and 35 late-game states. The source
corpus contains no in-check rows; that limitation is now explicit.

The stability schema v2 separates gold scalar value from gold policy. Tier 1
(18/20/22, 25M cap) yielded 165/256 gold values and 131/256 gold policies. Tier
2 (22/24/26, 100M cap) searched only 91 value failures and added 38 values. Tier
3 (26/28/30, 250M cap) searched the remaining 53 and added 16. Final yield was
219/256 gold values (85.55%), 178/256 policies (69.53%), and 37 quarantines.
The pipeline correctly rejected lower/upper aspiration records and missing
depth windows at node ceilings.

Total cost through all tiers was 5,176,308,830 final nodes and 2,251.14 local
CPU-seconds. Per original position: median 2.30M nodes, p95 121.7M, p99 277.5M,
max 358.4M, mean 8.79 seconds. Tier 3 alone cost 3.378B nodes for only 16 extra
values. That is cost evidence, not yet a production stopping rule. Per explicit
user guidance, do not cheap out on labels: audit a stratified sample of Tier-1
and Tier-2 accepted rows at depths 26/28/30 to measure false early stability.
Production depths are chosen from that evidence; budget changes corpus size,
not per-label integrity. Full-coverage
SF18 static components remain the broad auxiliary target. Deep search labels
are restricted to full-FEN sources; normalized LC0 piece-only records lack the
state required for honest deep search or rule50-scaled final evaluation.

The aggregate immutable report is
`data/nnue/quality/sf18-quality-audit-256-adaptive-summary-v1.json`. A fresh
16-position Tier-1 replay produced 0 semantic mismatches across depths,
seldepths, scores, mates, bounds, PV moves, nodes, hash fullness, WDL, and gate
decisions. Timing/NPS were correctly excluded from exact replay. Next: build the
chunked production/pilot labeler, audit duplicate/group splits, generate a small
immutable Tier1+2 pilot, and require protected-network engine evidence before
cloud bulk scale.

### Bulk public-corpus value matrix active (2026-07-15)

Per user direction, the next value screen is a preregistered four-by-two source
and architecture matrix (`docs/research/value-bulk-matrix-v1.md`). The four
cells are Chess768/256, Chess768/1024, king-conditioned 64xChess768/256, and
exact HalfKAv2_hm plus FullThreats D40. The initial v1 screen presented
249,036,800 accepted examples, but its superbatch-10 checkpoint was still at
LR 0.000757 on a 30-superbatch decay. Its classic256 result (about -512 Elo) is
retained only as an unfinished-training datapoint. The corrected v2 production
matrix completes all 30 superbatches: 747,110,400 accepted presentations per
cell with the LR fully decayed, followed by an 80-game 5k-node fail-fast match
and 2k/5k/20k confirmation curves for survivors. Combined training is gated on
source-specific survivors and must retain per-source loss masking.

The complete official Stockfish `fishpack32.binpack` is now durable in GCS and
local. It exactly matches the official LFS object: 5,579,166,857 bytes and
SHA-256 `04f90dcba242682cdb509e675ce590eff3676ea8b84a20e6baa8b5020f16ed16`.
The exhaustive audit found 5,477 complete chunks, 2,555,358,353 decoded
positions, 2,533,958,281 continuation pairs, genuine side-to-move outcomes,
and no truncated tail. The matrix may therefore consume it directly through
the SF-binpack shuffled loader with the standard quiet-position filter and
WDL=0.3.

The all-17-file Lichess v14 canonical export is complete. It scanned
844,812,067 flattened rows, formed 342,059,879 global FEN groups, and accepted
265,260,945 positions after the depth/node/mate/extreme-score and packing gates.
The exact 8,488,350,240-byte stream independently re-hashed to SHA-256
`b8939f7db39cc95186ea4bae1e779741887ed6c4d96149f835baa7a60541dc21`.
Cross-file FEN groups were carried correctly. Lichess has no game outcome, so
these runs are score-only with WDL=0; the neutral serialized result byte must
never be treated as a draw target. All local Parquets were removed only after
their exact GCS mirror was verified; the canonical training stream remains
local.

The first converged v2 Fishpack cell, Chess768/256, improved from the invalid
unfinished v1 screen (-512 Elo) to about -163 Elo after the full LR decay
(21-56-3 over 80 games at 5k nodes), narrowly missing the -150 survivor gate.
Chess768/1024 fit to lower loss but screened worse at about -215 Elo
(15-59-6), so plain width did not recover the missing engine signal. The
king-conditioned and D40 feature cells remain active; do not infer a corpus
verdict until those feature-isolation cells finish.

### Bulk matrix first results and D40 quarantine (2026-07-15)

The Fishpack side is complete and all four cells failed the -150 Elo survivor
gate at 5k fixed nodes: classic256 about -163, classic1024 about -215,
king-conditioned about -407, and D40 about -881 (0-79-1). The D40 result is
quarantined as a pipeline/recipe diagnostic rather than accepted as evidence
against the feature set. Raw, quantized, and packed-runtime inference agree,
the quantized tensors are not saturated, and exact trainer/runtime feature-set
parity passed, but the learned function itself is poor.

Matched held-position auditing makes the failure concrete. On 149,081 positions
the Fishpack D40 has SF16 Pearson 0.731 and blended-target MSE 0.02558, versus
net-0 at 0.946 and 0.00874. On 27,109 quiet legal transitions its local residual
mean/p95/p99 is 156.5/471/677.9 cp, versus net-0 at 93.6/224/316. The old
`d40-c2-eval` evaluated through the same current engine reaches SF16 Pearson
0.946 and MSE 0.00389, so the current D40 loader/runtime is capable of a much
better fitted function. Fishpack checkpoint audits at SB5/10/15/20/25/30 were
already poor at SB5 and improved only slightly, ruling out a late-training
collapse. Artifacts are under `data/nnue/audits/value-bulk-matrix-v2-*`.

An apparent SF-binpack restart bug was investigated and narrowed: the reader
does ignore `skip_count`, which makes resumed runs unsafe, but uninterrupted
training calls the reader once for the entire schedule and streams continuously
across superbatches. Thus the completed Fishpack jobs did not restart on every
superbatch. Fix or guard resume semantics before relying on an interrupted
SF-binpack continuation.

The Lichess source is materially better. Classic256 produced a confirmed curve
of approximately -193/-162/-176 Elo at 2k/5k/20k nodes. Classic1024 is the first
credible bulk candidate: approximately -100/-72/-39 Elo at 2k/5k/20k over 400
games per point; the 20k paired interval is about -74..-5. It remains weaker
than net-0 but approaches it as search deepens. The king-conditioned cell was
rejected at roughly -223 Elo in its 80-game 5k screen.

The automatic full Lichess D40 job was intentionally interrupted during SB1,
before a checkpoint, after the Fishpack smell-test failure was confirmed. A
five-superbatch Lichess pilot is active as
`value-bulk-d40-lichess-pilot-v1` (124,518,400 presentations, with the original
30-superbatch LR horizon). Before any full continuation it must pass static
held-label and legal-child coherence gates; only then run games. Do not combine
datasets until this D40 question is closed. The current best combined-training
candidate is Lichess classic1024, with Fishpack inclusion requiring explicit
per-source loss masking and evidence that it adds rather than dilutes signal.

### Takeover checkpoint: D40 staged diagnosis and why net-0 still leads (2026-07-15)

This is an intentional pause point. There are no training or match processes
left running. Read this whole file, then the last two sections in particular,
before resuming. The working tree is heavily dirty and contains the user's and
prior agents' work; never reset or discard it.

The five-superbatch Lichess D40 pilot completed at approximately -203 Elo
(17-59-4, 80 games, 5k nodes). It had excellent packed parity but only SF18
Pearson 0.87336 / blended-target MSE 0.01857 and quiet legal-transition
residual mean/p95/p99 166.5/418/601.9 cp. A staged continuation was therefore
implemented instead of paying for all 30 superbatches blindly:

- `tools/nnue/bullet/d40_features.rs` now supports
  `D40_START_SUPERBATCH` and `D40_INIT_RAW`; the absolute LR horizon remains
  `D40_LR_END_SUPERBATCH`. `tools/nnue/value_matrix.py` forwards both optional
  config keys. The SF-binpack reader still ignores skip count, so resumed
  SF-binpack jobs remain unsafe; uninterrupted jobs were not affected.
- `tools/nnue/configs/value-bulk-d40-lichess-mid-v1.json` loaded the SB5 raw
  weights, skipped 124,518,400 direct Bullet records, and ran absolute SB6-15.
  Adam moments reset at the boundary; this is a known approximation, not an
  exact optimizer resume.
- The SB15 candidate is
  `data/nnue/nets/value-bulk-d40-lichess-mid-v1-lichess-d40-full.cenn`.
  Raw/quant/packed-runtime parity remained excellent (quant mean/max error
  0.02069/0.28296 cp; packed runtime mean/max 0.00494/0.07390 cp).
- Its 80-game 5k screen improved dramatically to 27-44-9, about -74.97 Elo,
  paired 95% interval -152.4..-4.3. This rejects the notion that the original
  -881 result was an inherent limit of D40. The original recipe was grossly
  under-optimized for this large cold-start architecture.
- Static fit improved to SF18 Pearson 0.91018 / blended-target MSE 0.01396,
  SF16 Pearson 0.85583 / MSE 0.01794. Reports are
  `data/nnue/audits/value-bulk-d40-lichess-mid-v1-static.json` and
  `...-coherence.json`.
- Crucially, quiet-transition coherence got *worse* while static fit and games
  improved: mean/p95/p99 193.8/538/770.9 cp at SB15 versus
  166.5/418/601.9 at SB5 and 93.6/224/316 for net-0. Ordinary held-position
  loss is therefore not a sufficient model-selection metric. Do not continue
  SB16-30 automatically without resolving or explicitly controlling this.

The most important current explanation for the persistent net-0 gap is now a
target/search-contract mismatch, with a remaining possibility of an upstream
label semantic defect:

1. Exact net-0 seed-100 reproduction reached 50% against net-0, and the old
   `d40-c2-eval.cenn` reaches SF18 Pearson 0.9595 through the same current D40
   runtime. Trainer/export/runtime and the match harness can therefore produce
   and consume strong nets; a universal sign or packing failure is unlikely.
2. Net-0 used the small, coherent SF16 game corpus and `wdl=0.3`. Both new bulk
   sources differ. Lichess is isolated deep-search cp with no outcomes and must
   use WDL=0. Fishpack has its own score/outcome/search distribution. Dataset
   size does not compensate for inconsistent or consumer-misaligned targets.
3. A stride-101 audit of 2,626,346 canonical Lichess records found stored label
   mean/std +14.67/312.74 cp (all result bytes exactly neutral placeholders).
   The SB15 D40 prediction mean/std is +35.93/370.99 cp; net-0 is
   +7.13/612.74 cp on the matched held corpus. WDL training makes net-0 much
   more decisive. Archon's pruning/futility/aspiration margins consume absolute
   cp, so a net can improve correlation and cp MSE while violating the search
   controller's expected value scale.
4. The legal-neighbor audit shows that isolated-position objectives permit a
   jagged evaluation surface. Alpha-beta consumes score differences between
   parents, children, and siblings; net-0's original labels/recipe appear much
   more coherent in precisely that local topology.
5. Expanded HalfKA/threat spaces are cold-started and sparsely updated. Their
   optimization/curriculum is much slower than classic Chess768 and may amplify
   local jaggedness. The staged D40 result proves more training helps, but does
   not prove that simply training longer converges to the right search function.

A reversible output-scale causal test was completed before this pause.
`tools/nnue/rescale_net.py` now supports CENN-v2 by scaling only final bucketed
`l3_w/l3_b` tensors and recomputing the checksum; CENN-v1 behavior is retained.
The SB15 net was scaled 1.65x to roughly match net-0's held-score amplitude:
`data/nnue/nets/value-bulk-d40-lichess-mid-v1-lichess-d40-full-k1p65.cenn`.
Runtime output agreed with exact 1.65x scaling to 0.63 cp RMSE. On the identical
first 20 paired openings at 5k nodes, unscaled scored 14-21-5 (-61.4 Elo;
very wide interval) while 1.65x scored 11-24-5 (-117.2 Elo). Thus naive global
heating did not recover strength and was directionally worse in this small
screen. WDL's benefit is not reducible to multiplying every output; it likely
changes position-dependent calibration/ranking and the learned surface.
The first 80-game scale run was abandoned after two D40 engine stalls caused
fastchess to hang; all processes were cleaned up, then both 40-game controls
completed normally at concurrency four. Exact summaries are under
`data/nnue/campaigns/value-bulk-d40-lichess-mid-v1/matches/{d40-mid-k1p65,d40-mid-unscaled}`.
The automatic 400-game SB15 curve was deliberately interrupted at its first
2k point to prioritize this causal test. Campaign state now says
`interrupted_after_screen`; the trained artifacts are complete and must not be
retrained merely because the runner did not reach its terminal state update.

#### Recommended takeover order

1. Independently audit Lichess and Fishpack score perspective, score kind, and
   calibration against fresh pinned Stockfish 18 searches on reconstructed or
   source-preserved positions. Do not use exporter self-parity as the only
   proof. Specifically verify whether Lichess `cp` is the intended POV and
   whether taking the first row of each FEN group truly selects the documented
   deepest primary PV. Persist sampled row IDs/FENs and oracle outputs.
2. Make local coherence a mandatory checkpoint metric, but improve the metric:
   parent/child scores naturally change after a move, especially captures.
   Compare candidates against a strong teacher on the *same legal edges* and
   measure sibling ranking, quiet residual error versus teacher, tactical
   buckets, and calibration by score band—not only raw residual magnitude.
3. Run a minimal causal target experiment on classic256/1024 before another D40
   budget: coherent position groups or short trajectories, SF18 labels plus
   real outcome/WDL where trustworthy, and explicit score-only versus blended
   targets. Use identical initialization/presentations and small paired screens.
   The question is whether outcomes/local grouping recover search strength.
4. Do not naively combine Lichess and Fishpack. Combined training needs
   per-source WDL masks and evidence that Fishpack adds signal. The current best
   bulk candidate is Lichess classic1024: about -100/-72/-39 Elo at
   2k/5k/20k over 400 games per point.
5. If D40 resumes, prefer staged/protected initialization, lower LR, and
   coherence-aware checkpoint selection. Do not blindly run SB16-30. Consider
   initializing a protected/shared trunk from a strong evaluator, consistent
   with the long-term frozen-value architecture.
6. The strategic goal remains ARCHON: first establish an extremely strong,
   efficient value-only evaluator and tuned engine; then freeze/protect value
   computation and train Archon-native consequence heads for best move,
   cutoff, reduction risk, re-search, and cost. LC0/Lichess/Fishpack supply
   position coverage, while slow Archon/Reckless/SF18 searches supply explicit
   labels. Equal-time Elo is the ultimate gate.

Do not interpret the current result as “large data cannot beat net-0.” The
stronger conclusion is that net-0's small original corpus encoded a better
training/search contract than our bulk isolated-cp recipes. The next work is to
identify and reproduce that contract, not to spend more presentations on an
unverified target.
