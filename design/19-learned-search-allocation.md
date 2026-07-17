# 19 — ARCHON: learned counterfactual search allocation

Status: Living governing research roadmap (D41, 2026-07-14)

ARCHON learns how candidate search computations are likely to behave, then
allocates work under an explicit error-risk constraint. Strong external-engine
positions provide broad chess coverage; ARCHON's own conservative search
provides the final labels. The objective is not policy imitation. It is a
better strength–compute frontier for deterministic alpha–beta search.

---

## 1. Purpose and ownership

**This document owns:**

- The D41 research objective and phased program.
- The search-context and counterfactual-label contract consumed by training.
- The outputs of the node-control and legal-move consequence models.
- The generic constrained planner and gradual authority policy.
- The offline, fixed-node, and equal-time promotion gates.
- The iterative state → oracle label → train → deploy → failure-mine flywheel.

**This document does not own:**

- Base NNUE features, quantization, accumulator mechanics, or net serialization
  (doc 11).
- Observation-store physical schema and artifact provenance (docs 16 and 18).
- Search semantics and existing heuristic seams (docs 07, 09, 10, and 12).
- Match/SPRT mechanics (doc 14).
- Historical direct-policy experiments (doc 17 and
  `docs/research/policy-automl-state.md`).

Where an implementation needs a new field or seam, its owning document changes
before code. This roadmap specifies the research contract, not duplicate
component internals.

## 2. Governing decision and evidence

D41 is authoritative. It refines D34's search-integrated thesis after the D40
program established three important facts:

1. Learned move information can improve fixed-node search when paired with a
   sound evaluator.
2. Naive joint policy/value training can severely damage evaluation.
3. Lc0-trained value models have shown a systematic generalization deficit in
   the current pipeline, while random Stockfish relabel fine-tunes have not yet
   repaired it.

Therefore:

- Lc0 policy is an optional bootstrap teacher and experimental control, not the
  final objective.
- Lc0 games remain valuable as strategically rich state coverage.
- The protected evaluator and search-control models need not share all
  parameters.
- Engine games, not aggregate imitation loss, select deployable models.

The final teacher is a slower, more exhaustive ARCHON operating on states
encountered by the current ARCHON.

## 3. Core hypothesis and objective

A scalar evaluator discards information needed to allocate search. For a
position `x`, search context `c`, legal move `m`, and proposed search action
`a`, learn:

```text
Q(x, c, m, a) -> predicted search consequences
```

The minimum required outputs are:

```text
P(scout/reduced bound error before verification)
P(residual bound error after configured verification)
P(beta cutoff)
P(alpha raise)
P(full-depth re-search)
E[log(nodes)]
E[log(cycles)]
```

Later outputs may include final-best-move probability, PV stability, extension
value, candidate-set width, verification need, and the value of buying cheap
or full policy inference.

The controller chooses computation rather than merely choosing a move. For a
risk budget `epsilon`, the planner solves:

```text
a* = argmin_a E[cost(a)]
     subject to P(search error | x, c, m, a) < epsilon
```

The planner is deliberately chess-agnostic. Chess judgment lives in `Q`; the
planner performs resource optimization. There is no universal target class for
the “correct reduction,” because the best action depends on depth, node type,
window, uncertainty, time control, and tolerated risk.

The primary objective is maximum Elo per second. Fixed-node Elo measures search
efficiency; equal-time Elo decides whether that efficiency survives inference
cost.

## 4. Target architecture

### 4.1 Fast incremental position trunk

Reuse an NNUE-style accumulator:

```text
position -> incremental latent representation
```

The trunk may be the value evaluator's accumulator or a compatible incremental
representation. Sharing is justified only by measured compute savings and no
unacceptable loss of value or control quality.

### 4.2 Protected value head

Start from the strongest available evaluator, not a permanently privileged
historical net. Freeze it while search-control heads train. A frozen-control
experiment must keep evaluator outputs bit-identical on the parity corpus and
the engine's heads-disabled bench fingerprint unchanged.

Per D42 as amended (2026-07-17): the protected evaluator is selected from the
official-binpack-trained line by equal-time games on the UHO testing book —
B4 established that binpack data volume, not label source, was the evaluator
constraint, so the paired Reckless/SF18 labeling experiment is an ablation
and label-quality audit, not the selection path. Whatever wins must beat or
match the prevailing evaluator in fair evaluator-only games. After selection,
freeze the **entire value accumulator and value head**, not merely the final
scalar layer, for Phase 1 and the first reduction-risk experiment.

The default later decomposition is:

```text
shared incremental features
    ├── protected value adapter
    ├── node-control adapter
    └── move-allocation adapter
```

Unfreezing is staged. The evaluator learning rate begins 100–1,000 times below
the new-head learning rate, with detached teacher anchoring:

```text
L = L_eval + lambda_p * L_control
    + lambda_a * (E_joint - stopgrad(E_teacher))^2
```

Negative anchoring is forbidden. Per-layer gradient norms and cosine
similarities are recorded before shared capacity expands. GradNorm, uncertainty
weighting, or PCGrad may be tested, but none overrides evaluation's status as
the protected primary task.

### 4.3 Cheap node-control head

At eligible nodes, a small head predicts:

- evaluation uncertainty;
- candidate concentration or width;
- expected node cost;
- pruning danger and verification need;
- value of no, cheap, or full move-policy inference.

It decides whether additional neural control computation is worth purchasing.
Eligibility begins conservatively: PV/expected-PV nodes, meaningful remaining
depth, no trustworthy TT move, many quiet legal moves, flat classical ordering,
or uncached positions. Invocation rules are experimental inputs and must be
logged.

### 4.4 Legal-move consequence head

For generated legal moves only:

```text
position latent
+ move encoding
+ search context
+ proposed reduction / extension / policy tier
    -> consequences
```

The cheap structural default computes one small position embedding and scores
legal moves/actions using factorized move features: piece, source, destination,
promotion, capture, tactical flags, and search-action embedding. A low-rank
interaction is preferred over an unconstrained 4,096-class output unless
benchmarks prove the latter necessary:

```text
s(m | x) = b(m) + h(x)^T e(m)
```

This provides legal-move-only work, parameter sharing, controllable rank, and
cheap top-k ordering. Captures and quiet moves are evaluated separately because
classical tactical ordering may already solve most capture ordering.

### 4.5 Constrained planner

The first planner enumerates a small allowed action set and selects the cheapest
action whose upper confidence bound on error risk stays under `epsilon`.
Uncertainty or missing predictions fall back toward more search, never less.
Risk bounds may vary by PV/non-PV node and time control but are configuration,
not hidden learned state.

The planner initially controls only bounded corrections:

```text
R_final = R_baseline + clip(delta_R_theta, -2, +2)
```

It may spend computation before it may delete information. Hard pruning is not
authorized until reduction/verification calibration is stable in-distribution
and on hard/OOD suites.

## 5. Data contract

### 5.1 Position sources

Maintain an explicit mixture with source provenance:

- Lc0 game and analysis positions;
- ARCHON self-play positions;
- internal ARCHON search nodes;
- tactical, unstable, and engine-disagreement positions;
- perturbed and external positions for coverage and OOD evaluation.

Sampling weights are versioned. State-source splits are kept separate so a
large easy source cannot hide failure on the controller's actual distribution.
Game, opening, and source-family boundaries prevent train/test leakage.

### 5.2 Required board and search context

Each sampled node records enough state to reproduce the search decision:

- board state, side to move, castling, en-passant, rule-50, and repetition
  context;
- remaining depth, ply, alpha, beta, and node type;
- static evaluation and its location relative to the window;
- TT hit, stored depth, bound, and move confidence;
- legal-move count and tactical/quiet composition;
- classical ordering signals available at the decision point;
- previous PV, iterative-deepening score/best-move stability, and relevant
  re-search history;
- evaluator, engine, parameter, compiler, and hardware identities.

Missing context is not silently zero-filled. Schema presence bits and versions
distinguish unavailable fields from genuine zero values.

### 5.3 External value-oracle bootstrap (D42, amended 2026-07-17)

Demoted from the evaluator-selection path to an ablation + label-quality
audit (see §4.2 and D42's amendment); the mechanics below govern whenever
paired external labels are produced. When a non-ARCHON value label is
required, label identical positions with both:

- the newest compiled local Reckless release binary; and
- the official stable Stockfish 18 binary.

Dataset provenance pins path, UCI identity, source revision, SHA-256, options,
threads/hash, node/depth/time budget, and full search output. Stockfish 16 is a
historical `net-0` teacher/control only and is not an allowed silent fallback.

Store both engine outputs independently. Because engine centipawn and WDL
scales are not interchangeable, calibrate each against held-out game outcomes
or a declared common probability space before forming a consensus target.
Retain absolute and signed disagreement, bound/PV disagreement, and search
stability as labels and sampling strata. Train/evaluate single-teacher,
consensus, and disagreement-aware ablations so an ensemble cannot conceal a
bad oracle or incompatible scale.

### 5.4 ARCHON oracle reference search

The oracle is a deterministic, conservative ARCHON configuration with greater
depth/nodes and risky selectivity reduced or disabled. Every label records its
budget and completion status. A reference is accepted only if its score/bound
and best move meet a configured stability rule across deeper searches or budget
increments. Unstable references remain useful as an explicit hard/uncertain
class but cannot masquerade as ground truth.

Oracle validation includes replay from the recorded state, deterministic
repeatability, legal PVs, score-perspective checks, and comparison of a sample
against an independent strong engine to expose systematic oracle bugs.

### 5.5 Counterfactual actions and outcomes

For selected moves, test a subset of:

```text
reduction = 0, 1, 2, 3, ...
extension = 0 or 1
policy tier = none, cheap, full
```

Record:

- returned score and bound;
- compatibility with the stable reference result;
- the proposed scout/reduced bound relation before verification;
- residual bound compatibility after configured verification/re-search;
- alpha raise and beta cutoff;
- full-depth re-search requirement and its result;
- final-best-move status and score margin;
- nodes, CPU cycles, wall time, and inference counts;
- resulting PV and its stability.

The record is an intervention: the proposed action must be forced while other
search settings remain fixed. Labels derived only from what the behavior policy
happened to do are observational and must be marked as such.

### 5.6 Counterfactual sampling policy

Do not exhaust every action for every move. Spend oracle compute on:

- model uncertainty and weak calibration regions;
- model/classical-controller disagreement;
- unexpected fail-highs, re-searches, and incorrectly reduced moves;
- tactical and engine-disagreement positions;
- policy-use disagreements;
- action boundaries where adjacent reductions have similar predicted utility;
- underrepresented context/source strata.

Retain a random counterfactual sample to measure selection bias and provide a
stable longitudinal test set.

## 6. Training flywheel

```text
1. Current ARCHON searches and self-plays.
2. Sampler retains important encountered nodes plus a random control sample.
3. Slow ARCHON oracle runs reference and counterfactual searches.
4. Consequence models train with source/context-stratified validation.
5. A candidate deploys with bounded authority.
6. Candidate games expose its new state distribution.
7. Failures, disagreements, uncertainty, and boundary cases are mined.
8. Those states are relabeled and the cycle repeats.
```

Every loop is a D38 lineage: behavior engine, evaluator, controller, position
mixture, oracle configuration, dataset, model, offline report, match report, and
promotion/rejection decision are content-addressed. A static corpus is never
assumed sufficient after the controller changes the visited distribution.

## 7. Phased research program

### Phase 1 — ARCHON-native move targets

Begin with D44's teacher curriculum. Independently search every legal root move
with pinned Reckless at increasing budgets to bootstrap stable-best, pairwise
ordering, score-margin, PV-stability, and root-complexity targets. Then rerun
identical source row IDs with ARCHON and adapt the move representation to its
search. Contextual cutoff is native-only once actual ARCHON windows are used.
Train final-best-move probability, pairwise ordering, and later cutoff
probability. Compare directly with Lc0 policy and classical ordering while the
evaluator remains frozen.

Every row carries `producer_engine`. Reckless and ARCHON labels may coexist for
engine-agnostic move quality only through explicit teacher/domain ablations;
they are never pooled blindly. Reduction, re-search, bound-error, and deployed
cost heads use ARCHON-native targets.

Fail-fast order:

1. Tiny end-to-end deterministic smoke corpus.
2. Reckless independent-root-arm stability and repeatability audit.
3. Same-row ARCHON root adaptation and teacher-disagreement audit.
4. Label audit and leakage checks.
5. Small model with calibration/ranking metrics.
6. Fixed-node games at several shallow budgets.
7. Scale data/model only after a positive engine signal or a diagnosed,
   repairable failure.

The output is an Elo-versus-node-budget curve, not one Elo number.

### Phase 2 — Reduction-risk model

For each sampled move/reduction, predict reference incompatibility and
full-depth re-search. Deploy only `clip(delta_R, -2, +2)` corrections to current
LMR. Initially, uncertain predictions reduce less or fall back to baseline.

### Phase 3 — Cost model

Predict expected log nodes and log cycles for move/action pairs. Calibrate node
and cycle estimates separately; nodes isolate search efficiency while cycles
capture inference and machine cost. The planner can now select actions by both
correctness and cost.

### Phase 4 — Unified Level-4 allocator

Jointly expose best-move, cutoff, reduction-risk curve, extension value,
subtree cost, and policy-inference value through a search-control adapter.
Unification occurs only after each target independently demonstrates engine
value. Protected-value constraints remain active.

### Phase 5 — Constrained Level-4.5 node planner

Add candidate width, broad-versus-narrow preference, total uncertainty,
verification budget, and policy-compute tier. The planner builds a conditional
node schedule from predicted consequences.

### Phase 6 — Limited learned memory

Only after the static allocator converges, add a small recurrent state updated
from sibling outcomes. It may represent within-node policy miscalibration,
unexpectedly broad candidate sets, repeated re-searches, or rising tactical
instability. Memory is a residual improvement, never the foundation or sole
source of irreversible pruning.

## 8. Evaluation and promotion gates

Every phase passes all three gates.

### 8.1 Offline prediction

- Reliability diagrams and expected calibration error for bound error,
  cutoff, alpha raise, and re-search.
- Cost MAE/RMSE on log nodes and log cycles, plus tail-error reporting.
- Ranking, top-k, and pairwise metrics where applicable.
- Metrics split by depth, node type, move class, source, and action.
- Dedicated tactical, unstable, disagreement, and OOD suites.
- Baselines: constant predictor, classical heuristic, Lc0 policy where
  applicable, and ablations without search context.

Offline metrics may reject a model but cannot promote it by themselves.

### 8.2 Fixed-node engine tests

- Frozen evaluator clone versus evaluator plus controller.
- Paired openings and pair-level bootstrap confidence intervals.
- Multiple policy/model seeds.
- Several fixed-node budgets spanning shallow to deeper search.
- Heads disabled, classical controller, and bounded-controller ablations.
- Invocation count, re-search rate, tactical regression rate, and nodes saved.

Policy/control value is reported as an Elo-versus-node-budget curve. Also fit a
baseline scaling curve: if the controller at `N` nodes matches baseline at
`kN`, then `k` is its node-equivalent multiplier.

### 8.3 Equal-time engine tests

Measure NPS loss, latency/cycles per invocation and node class, invocation rate,
total nodes saved, re-searches, and equal-time Elo. A controller with multiplier
`k` can tolerate approximately a `k`-fold node-rate penalty before erasing its
fixed-node advantage; measured games remain authoritative.

A candidate promotes only when it reproducibly improves the prevailing
evaluator/controller at equal time without a material tactical or stability
regression. Initial pilots may use fixed nodes to fail fast, but fixed-node gain
alone is not shipment.

## 9. Safety and rollout

Authority expands in this order:

```text
observe/shadow only
-> move ordering and policy-tier selection
-> heuristic + bounded neural reduction/extension correction
-> neural allocation with heuristic fallback
-> neural allocation as primary controller
-> selective hard pruning after calibration and verification evidence
```

The classical controller remains a behavior policy, source of states, fallback,
and experimental control. Every learned feature has a runtime disable switch.
Out-of-range, non-finite, missing-context, version-mismatch, or high-uncertainty
outputs take the conservative fallback. Search remains deterministic under a
fixed model/configuration and retains alpha–beta verification.

## 10. Immediate execution path

1. Settle the eval foundation (in flight, 2026-07-16): the D40 equal-time
   gate, the 720sb scale point, and the s3-vs-s4 re-check at equal time on
   the UHO book. Select the protected evaluator from the binpack-trained
   line by equal-time games (D42 as amended), then freeze its complete
   accumulator and value head bit-identically.
2. Pin the newest local Reckless release and official Stockfish 18 binaries
   for the D44 curriculum and the demoted paired-label ablation; label a
   small common calibration set and audit score/WDL compatibility.
3. Treat the current Lc0 corpus as positions; preserve its original labels only
   as bootstrap features and controls.
4. Define the minimal reproducible node-context and oracle-output records in
   doc 16, including castling, rule-50, and repetition state.
5. Build a tiny ARCHON oracle relabeling smoke pipeline with deterministic
   replay and label audits.
6. Generate ARCHON-native final-best-move, pairwise, alpha-raise, and cutoff
   targets on a small stratified sample.
7. Compare a frozen-head ARCHON-native model against Lc0 policy and classical
   ordering over multiple fixed-node budgets.
8. Add forced counterfactual reductions and train the first calibrated
   reduction-risk head.
9. Test bounded LMR corrections, then add log-node/log-cycle cost prediction.
10. Begin on-policy failure/disagreement mining and iterative relabeling.
11. Unify heads only after independent engine-value evidence; optimize equal
    time throughout.

## 11. Acceptance criteria

The roadmap is correctly instantiated when:

1. An oracle-labeled example can be deterministically replayed from stored
   position and context, and every counterfactual action has explicit outcome,
   cost, provenance, and completion status.
2. The frozen evaluator is bit-identical with control heads attached and
   disabled.
3. Phase-1 targets are evaluated against both Lc0 policy and classical ordering
   on held-out source-stratified data and in fixed-node games.
4. The first reduction-risk model is calibrated, has a conservative fallback,
   and cannot alter baseline LMR by more than two plies.
5. Every promotion report includes offline strata, an Elo-versus-node-budget
   curve, pair-bootstrap intervals, multiple seeds, NPS/inference costs, and an
   equal-time result.
6. Dataset/model/controller/oracle lineage is complete in the experimental
   ledger.
7. The first M11 candidate shows reproducible positive fixed-node and equal-time
   strength against the same prevailing evaluator with the classical controller
   as its direct control.

## 12. Principal risks and required diagnostics

- **Oracle bias or instability:** use deeper-budget stability, independent
  engine spot checks, conservative selectivity, and explicit uncertain labels.
- **Counterfactual confounding:** force exactly one action while holding the
  remaining search configuration constant; distinguish interventions from
  observations.
- **Distribution shift:** retain random controls and continuously relabel states
  reached by the deployed controller.
- **Risk miscalibration in rare tactics:** stratified hard suites, uncertainty-
  conservative fallback, and no early hard pruning.
- **Value degradation:** frozen first, detached positive anchoring later,
  asymmetric learning rates, gradient diagnostics, and evaluator-only games.
- **Inference cost erases node savings:** selective invocation, factorized legal-
  move scoring, node-equivalent scaling curves, and equal-time gates.
- **Search-context leakage:** time/order-sensitive fields must represent only
  information available when the action is selected.
- **Self-reinforcing blind spots:** mix external and disagreement states with
  on-policy states; never train solely on the current controller's successes.

The research bet is concise: learn the expected value, cost, and risk of each
possible search computation, then let a constrained planner allocate work while
deterministic alpha–beta retains final verification.
