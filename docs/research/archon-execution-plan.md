# ARCHON Research Goal and Execution Plan

Status: Active D41 operating plan

Last updated: 2026-07-14 America/New_York

Read this file, `design/19-learned-search-allocation.md`, and
`docs/research/policy-automl-state.md` before beginning or resuming any research
loop. Update the state file after every trial. Update this plan only when the
goal, milestone definitions, or ordering changes.

## 1. North-star goal

Build an ARCHON-native learned search allocator that makes the strongest
available protected evaluator play stronger per unit of computation than the
classical search controller.

The final objective is **reproducible equal-time Elo**, not policy accuracy,
training loss, or node savings in isolation. The allocator must learn the
expected value, cost, and risk of proposed search actions from ARCHON-native
counterfactual searches and deploy through a constrained planner with
deterministic alpha–beta verification.

The aspirational outcome is a self-improving data flywheel whose successive
controllers generate more informative states and labels, producing continued
strength gains without depending on an external engine as the final teacher.

## 2. Goal hierarchy

### G0 — Trustworthy experimental platform

Before interpreting strength, prove that position state, search context,
reference labels, forced counterfactual actions, model export, runtime inference,
and match attribution are correct and reproducible.

Success means deterministic replay, complete provenance, adversarial label
audits, bit-identical protected evaluation, and fail-loud behavior for missing
or incompatible state. A G0 failure stops scale-up immediately.

The first value target uses paired labels from official Stockfish 18 and the
newest compiled local Reckless release. Their labels remain separate until
their score/WDL scales are calibrated; disagreement is retained as signal. The
selected value accumulator and head are then frozen completely for G1/G2.

### G1 — ARCHON-native move information has engine value

With one protected evaluator, train cheap heads for final-best-move, pairwise
ordering, alpha raise, and cutoff using deep ARCHON labels. Establish whether
they outperform classical ordering and the existing Lc0-policy control.

Success requires a positive Elo-versus-node-budget curve across multiple
budgets, not one favorable short match. This is the first scientific isolation
experiment and the immediate active objective.

### G0C — Search-control data and head infrastructure

This is a parallel foundation track, not downstream of value-model selection.
Define targets, collect observational telemetry, run forced counterfactual
searches, persist replayable records, and implement the head interfaces against
a versioned latent contract while the value track runs.

The only synchronization point is training or validating a production head
against the final selected accumulator. Data semantics, oracle machinery,
sampling, storage, placeholder-head tests, and engine-side bounded-control seams
must not wait for that selection.

Success means:

- every target has an exact event definition and perspective;
- observational behavior-policy outcomes are distinguishable from forced
  interventions;
- a stored node/context/action record deterministically replays;
- reduction/action forcing changes only the declared action;
- labels include correctness, re-search, and node/cycle cost;
- head inference has a stable latent/move/context/action interface, a dummy or
  frozen-test implementation, and a heads-disabled bit-parity gate.

### G2 — Learned reduction risk improves allocation

Train calibrated predictions of reference incompatibility and full-depth
re-search for forced move/reduction interventions. Apply only bounded LMR
corrections:

```text
R_final = R_baseline + clip(delta_R, -2, +2)
```

Success requires fewer nodes or stronger fixed-node play without a tactical,
bound-correctness, or re-search regression. High uncertainty falls back toward
more search.

### G3 — Cost-aware allocation survives real inference cost

Add expected log nodes and log cycles, then let the constrained planner choose
the cheapest action below its risk budget. Optimize selective invocation and
the legal-move head until search-efficiency gains survive NPS loss.

Success requires statistically supported equal-time improvement against the
same evaluator and classical controller, with invocation cost, nodes, cycles,
re-searches, and tactical outcomes reported.

### G4 — On-policy improvement flywheel

Deploy with bounded authority, sample the states the controller actually
visits, mine failures/disagreements/action boundaries, relabel with the slow
ARCHON oracle, and retrain.

Success requires at least two successive promoted generations and evidence
that newly mined data improves the controller's encountered distribution
without erasing performance on fixed longitudinal and OOD suites.

### G5 — Level-4/4.5 allocator

Unify independently proven targets, expand from bounded corrections to primary
allocation with heuristic fallback, and add node-level scheduling. Learned
memory and hard pruning remain later residual stages.

M11 is achieved when a unified candidate reproducibly improves both the
Elo-versus-node-budget frontier and equal-time Elo over the prevailing evaluator
and classical controller across multiple seeds.

## 3. Quantitative promotion standard

Pilots are for rejection and diagnosis; they never establish success.

A research result is promotable when:

1. Correctness and provenance gates are entirely green.
2. The effect replicates across at least two independent model/data seeds or
   independently labeled samples.
3. Paired-opening results include pair-level bootstrap confidence intervals.
4. The fixed-node curve covers at least three materially different budgets.
5. A claimed general gain is positive at two adjacent budgets and shows no
   material unexplained regression at the others.
6. Final promotion includes equal-time games, NPS/cycle measurements, tactical
   checks, and heads-disabled/classical-controller ablations.

The initial meaningful target is a **+30 Elo equal-time point estimate with a
confidence interval excluding zero** against the same protected evaluator and
classical controller. Any statistically supported positive equal-time gain is
scientifically important and may justify another flywheel generation; +30 is
the first significance target, not a reason to discard a real smaller gain.

## 4. Parallel execution plan

Run Track A and Track B concurrently. Within each track, always work on the
earliest incomplete gate. Neither track waits for the other until the explicit
synchronization gate.

### Track A — value representation and external relabeling

A1. **External oracle lock.** Pin and smoke-test official Stockfish 18 and the
   newest compiled local Reckless release by identity, revision, SHA-256, and
   label options. Stockfish 16 is prohibited except as a historical control.
A2. **Reckless-first corpus relabel.** Stream deterministic, resumable labels
   over the full existing LC0 position corpus. Keep the original corpus as the
   canonical position store and expose completed shards immediately.
A3. **Paired-oracle calibration.** Label identical row IDs with Stockfish 18,
   calibrate the two engines' value/WDL scales, retain disagreement, and test
   single-teacher/consensus/disagreement-aware targets.
A4. **Value bootstrap and baseline lock.** Identify, verify, hash, and ledger
   the strongest evaluator. Promote a new candidate only through evaluator-only
   games, then freeze its complete value accumulator and head. Preserve
   historical evaluator, classical-search, and Lc0-policy controls.

### Track B — search-control labels, data generation, and heads

B1. **Target contract.** Define exact final-best-move, pairwise, alpha-raise,
   cutoff, bound-error, re-search, node, and cycle labels, including score
   perspective and reference compatibility.
B2. **Reckless root bootstrap.** On deterministic source row IDs, independently
   search every legal move with the pinned Reckless build at two increasing
   budgets. Store stable best/pairwise labels, score margins, PV stability,
   per-move nodes, teacher identity, and acceptance status. This requires no
   final ARCHON accumulator and no invasive Reckless instrumentation.
B3. **ARCHON root adaptation.** Run the same root-arm protocol and identical row
   IDs with ARCHON. Measure teacher agreement/disagreement and adapt the move
   representation before treating it as native control.
B4. **Observational telemetry.** Instrument sampled normal ARCHON searches to
   capture full board/search context, behavior-policy actions, outcomes, and
   stable row identity without changing search behavior.
B5. **Counterfactual generator.** Build a slow deterministic oracle/replay tool
   that forces one move/action intervention at a time, holds all other search
   settings fixed, and records alternative reductions/extensions/policy tiers.
B6. **Head contract and scaffolding.** Define a versioned
   `position_latent + move + context + action -> consequences` interface.
   Implement feature-gated dummy/frozen-test heads, legal-move scoring, runtime
   disable/fallback, and bit-identical heads-off tests without waiting for final
   weights.
B7. **Small native dataset and model.** Generate a stratified smoke/pilot set,
   audit replay and leakage, then train the smallest useful move/cutoff head and
   reduction-risk baseline against a provisional frozen accumulator.

### Synchronization and engine gates

S1. **Frozen-value integration.** Bind the independently validated Track-B head
   contract to the selected Track-A accumulator. Freeze the complete accumulator
   and value head; retrain only adapters/heads.
S2. **Phase-1 games.** Compare classical ordering, Lc0 policy, and ARCHON-native
   control over several node budgets with paired openings and bootstrap
   intervals.
S3. **Bounded reduction risk.** Test calibrated `clip(delta_R, -2, +2)` control.
S4. **Cost and time.** Add selective invocation, scaling curves, and equal-time
   games.
S5. **Flywheel.** Mine on-policy failures/disagreements and repeat until G4/M11.

## 5. Always-next rule

At the start of every work session:

1. Read the last state entry and inspect running processes/artifacts.
2. Resume a valid interrupted experiment before starting a duplicate.
3. Otherwise select the earliest incomplete ordered-plan item.
4. Run the cheapest test capable of falsifying its current assumption.
5. If it fails, diagnose the failure and add a targeted control; do not scale.
6. If it passes, record artifacts and evidence, then advance exactly one gate.
7. Before stopping or compacting, update the state with results, uncertainties,
   current process IDs/artifacts, and the single next command or action.

When evidence is ambiguous, acquire the smallest additional sample that
distinguishes the leading explanations. Do not average incompatible conditions
or substitute a larger run for a missing control.

## 6. Fail-fast rules

Stop or reject a trial immediately for:

- incomplete/incorrect board or repetition state;
- score-perspective, bound, move-map, or forced-action mismatch;
- evaluator drift in a frozen-value experiment;
- train/test leakage or game-contiguous split contamination;
- non-finite outputs, illegal moves, artifact mismatch, or silent fallback;
- a clearly negative pilot effect with no evidence of a harness defect;
- inference cost already exceeding the measured node-equivalent budget by a
  margin that selective invocation cannot plausibly repair.

Do not launch a full corpus, long training run, or large gauntlet until the
preceding smoke and pilot gates are recorded green.

## 7. Protected invariants and non-goals

- Lc0 positions are valuable; Lc0 visit policy is not the final target.
- The unresolved Lc0-value generalization problem remains an explicit
  diagnostic subgoal, not an explanation to wave away.
- The strongest evaluator is protected and initially bit-identical.
- Shared computation is optional; uncontrolled gradient sharing is forbidden.
- The model predicts consequences; the planner chooses actions.
- Classical search remains behavior policy, fallback, and direct control.
- Learned ordering and bounded reductions precede irreversible pruning.
- Offline accuracy cannot promote a model without engine games.
- Fixed-node gains cannot ship without equal-time evidence.
- More training is not the default response to a failed causal isolation.
- GCP is a durable-data factory, not a training cluster. All training remains
  on the local GPU until explicitly reconsidered.
- Cloud jobs must produce reusable corpus artifacts. Smoke tests, schema
  experiments, one-off probes, tuning data, and disposable bootstrap labels
  remain local.
- Cloud submits are Spot-only, dry-run by default, resumable, cost-estimated,
  and bounded by task-count/runtime limits. A few hundred dollars per month is
  a hard ceiling, not a spending target.

## 8. Current checkpoint

Active objectives run in parallel:

- **Track A:** A2 — Reckless-first full-corpus relabel, then paired SF18
  calibration and value baseline selection.
- **Track B:** B1/B2 — exact search-control target definitions and a reusable
  independent-root-arm Reckless bootstrap; then identical-row ARCHON
  adaptation, native telemetry/counterfactual generation, and head scaffolding
  without waiting for the final value model.

Pinned binaries:

- Reckless: `reckless_reference/Reckless/target/release/reckless`, UCI
  `Reckless 0.10.0-dev-d445b8bf`, source `d445b8bfec58e4b1d94cdf1b6884b383561d4a60`,
  SHA-256 `8c4e36c012f4403f4441eead969a5e8dde182a2e45c69696170c85c11c2c66f9`.
- Stockfish: `engines/stockfish-18-src/src/stockfish`, UCI `Stockfish 18`,
  official tag `sf_18`, source `cb3d4ee9b47d0c5aae855b12379378ea1439675c`,
  AVX2 build SHA-256
  `174270346ae9ed600713d165fa745dfa87fc084e44907c8084767f2b905e86d3`.

The previous D40 policy campaign is historical evidence and control material.
No new direct-Lc0-policy campaign should begin unless it is explicitly designed
as a D41 baseline or ablation.

The next deliverable is the paired-oracle calibration and baseline-lock report:
label settings, target-scale comparison, disagreement strata, strongest
verified evaluator and lineage, frozen parity evidence, and exact classical and
Lc0-policy controls for the first ARCHON-native experiment.

Cloud data work is governed by `docs/research/cloud-data-campaign.md`. Durable
scope includes all local LC0 source material, paired Reckless/Stockfish-18
labels for approximately 101M selected existing rows, and a separate 100M
unique full-state corpus from Reckless/Stockfish play. Bulk labeling is a
high-parallelism Spot task array of approximately 404 independently resumable
250K-row tasks, not a monolithic process. Native generation is likewise split
by deterministic game-seed ranges. Representative price/performance benchmarks
and measured $50 existing-label / $150 native-campaign ceilings remain gates.
