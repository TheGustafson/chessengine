# Archon Self-Improvement Roadmap

Status: governing research sequence, recorded 2026-07-15.

## Objective

Build Archon into an efficient top-level classical engine with an excellent
value network, then protect that value capability while adding learned search
allocation and closing an Archon-native self-improvement flywheel.

The target is not merely a strong network in isolation. The engine, feature
updates, inference path, classical search, and learned controller must jointly
occupy a competitive strength-per-time frontier. A practical strength target
is to beat Stockfish 15 and plausibly Stockfish 16 under fair conditions, while
remaining competitive enough with Stockfish 18 that the match is informative
rather than a blowout. Beating Stockfish 18 is not a required gate.

## Phase A: Finish the causal information experiment

Complete the current independent-root search-intervention experiment before
changing the controller or committing to a new control representation:

1. Seal and audit the 16,384-root observational trace.
2. Generate exact deterministic counterfactual reductions with one sampled
   intervention per independent root.
3. Keep initial scout error, post-verification residual error, re-search,
   root-result changes, and global node cost distinct.
4. Rerun the preregistered context, protected-SCReLU latent, raw-board,
   low-rank/residual, and structured-move probes.
5. Use root-grouped held-out tests, calibration, and paired-root bootstrap
   intervals.
6. Persist the decision evidence, then continue directly into the strong
   value-engine foundation. Do not grant a learned head authority until its
   later Phase-C gates pass.

This experiment determines what richer data and representations the later
controller must retain. It is not allowed to delay the strong-value-engine
foundation after its decision gate is complete.

## Phase B: Build the strong value-engine foundation

### Dataset promotion gate

Data quality is a prerequisite for the flywheel, not a cleanup task after
scaling. Every new label family starts with a small audited sample. Bulk local
or cloud generation is forbidden until the sample has passed state-completeness,
score-semantics, bound/mate/missing-value, provenance/alignment, oracle-stability,
and held-out downstream checks. Search scores are not scalar value labels unless
the producing iteration completed with exact semantics; shallow completed
depths are diagnostic controls, not production oracles. Bounds must be rejected
from pointwise losses or modeled explicitly as censored inequalities. A failed
gate quarantines the affected corpus and all conclusions depending on it.

For the bootstrap value corpus, Stockfish 18 is the sole production teacher.
Reckless is reserved for sparse audits because requiring two engines on every
row roughly doubles cost without being necessary for bootstrap. Search labels
use a tiered stability oracle: exact completed iterations, stable best move and
score across the last three deep checkpoints, deterministic cost ceilings,
escalation of unresolved positions, and quarantine after the final failed tier.
The threshold and tiers must be selected from a local cost/yield frontier before
cloud scale. Static evaluator components may be retained as auxiliary targets,
but they must be named by their exact engine semantics and never conflated with
deep search values.

### B1. Correctness before strength claims

Establish end-to-end invariants for the value-only path:

- Stockfish-style position features and complete available state are encoded
  with documented perspective and scaling.
- Reckless-like incremental accumulators update correctly for ordinary moves,
  captures, promotions, en passant, castling, null moves, make/unmake, and
  refresh paths.
- Training activation, quantization, export, Rust loading, incremental runtime
  inference, and scalar/WDL interpretation agree within explicit tolerances.
- Mate sentinels are excluded from ordinary centipawn correlation metrics.
- Teacher score, WDL/result blending, side-to-move conversion, and rule-state
  limitations are explicit rather than inferred from filenames.
- Randomized long make/unmake sequences and large held corpora exercise parity,
  not only hand-selected positions.

No Elo result can override a failed semantic or parity gate.

### B2. Efficiency engineering

Use the latest compiled Reckless, Stockfish 18, and optimized Rust expectations
as references. Measure at least:

- accumulator refresh and delta-update latency;
- scalar evaluation latency and batched training throughput;
- bytes touched, cache behavior, alignment, SIMD utilization, and branch cost;
- search NPS with classical evaluation, value NNUE, and diagnostic heads off;
- make/unmake, move generation, TT probing, ordering, and quiescence cost;
- single-thread reproducible benchmarks first, then scaling behavior;
- release-build and target-feature identity for every comparison.

Profile before optimizing, preserve benchmark signatures, and require parity
tests around every low-level rewrite. Architectural convenience is not a valid
reason to accept a large per-node tax.

### B3. Classical search quality

Audit and tune the heuristic controller around the stable value network:

- aspiration, TT use, move ordering, history and continuation history;
- null-move pruning, futility/razoring, LMR, extensions, and quiescence;
- time management and repetition/rule-50 handling;
- tactical correctness, node-count regressions, and fixed-node behavior.

Use parameterized paired-opening gauntlets, multiple node/time budgets, fixed
seeds, and confidence intervals. Prefer broad sensitivity studies followed by
confirmation over tuning to one opponent or one budget. Revisit tuning after
material evaluator changes because search/evaluation interactions are real.

### B4. Local value-network curriculum

Start with the complete 24,951,581-position historical Stockfish-16 corpus as
it gains pinned Reckless-d445 and Stockfish-18 labels. Train value-only models
first. Isolate label targets and blends before combining them:

- original SF16 label;
- Reckless label;
- Stockfish 18 label;
- result/WDL information;
- teacher agreement, disagreement, mate, tactical, phase, and stability
  strata.

Run small deterministic smokes, then enough data to reject weak recipes
quickly. Promote only checkpoints that pass held prediction, runtime parity,
fixed-node games, and equal-time games. Iterate architecture, optimization,
features, search heuristics, and inference efficiency until the local-data
frontier is exhausted rather than declaring the first converged loss a final
network.

### B5. Cloud scale and top-engine gate

After the local recipe is stable, train on the full durable corpus:

- approximately 101M diverse LC0-sourced positions relabeled by Reckless and
  Stockfish 18;
- subsequently, strong-engine native positions from Reckless self-play,
  Stockfish-18 self-play, balanced cross-play, and exploratory/disagreement
  play, globally deduplicated and paired-labeled.

Cloud is used for durable data generation and storage; training remains on the
local GPU unless explicitly changed. Scale only recipes that have passed local
gates. Track an Elo-versus-time/node curve against historical Archon, Reckless,
Stockfish 15, Stockfish 16, and Stockfish 18. The foundation is ready when it
is both extremely efficient and strong enough that later controller gains are
meaningful at top-engine scale.

## Phase C: Protect value and add richer learned search control

Freeze the proven value accumulator and value output bit-identically. Then add
capacity gradually:

1. Train independent, calibrated Archon-native consequence targets locally.
2. Add dedicated adapters/accumulators only where the causal experiment shows
   missing information.
3. Prove each head independently before multitask unification.
4. Begin with ordering, verification, and bounded LMR corrections; hard pruning
   remains later.
5. Measure inference cost and selective invocation as part of every Elo claim.
6. Scale quickly to the full corpus only after local fixed-node and equal-time
   gates pass.

Evaluation remains the protected primary task. Joint training may be explored
later with detached-teacher output anchoring and asymmetric learning rates, but
must not silently erode the frozen-value upper bound.

## Phase D: Close the self-improvement flywheel

Once the engine/value foundation and bounded controller are strong:

1. Current Archon search and self-play generate states.
2. A slower, conservative Archon oracle generates value, cost, risk, and
   counterfactual search-action labels.
3. Disagreements, failures, uncertainty, re-searches, and action boundaries are
   preferentially mined.
4. New models deploy with bounded authority and deterministic fallback.
5. Fixed-node and equal-time games select models.
6. The deployed controller changes the visited-state distribution, which is
   captured, deduplicated, relabeled, and used for the next iteration.

Reckless and Stockfish remain useful external anchors, disagreement sources,
and regression opponents. The eventual improvement teacher is a slower,
stronger Archon operating on states produced by the current Archon.

## Non-negotiable experiment discipline

- Persist configuration, binary/network hashes, source lineage, seeds, and
  resumable progress.
- Smoke every new path and fail fast before bulk compute.
- Keep final held roots/openings untouched by training, selection, calibration,
  and tuning.
- Separate fixed-node search efficiency from equal-time strength and NPS tax.
- Do not infer engine value from imitation accuracy alone.
- Do not scale a target, architecture, or label blend merely because its loss
  decreases.
- Continue autonomously through Phase B4. The next user checkpoint is after a
  very strong local-data value-only network is demonstrated with engine-level
  evidence, not merely training loss.
