# Stockfish 18 High-Quality Bootstrap Label Plan

Status: governing plan, begun 2026-07-15.

## Objective

Create durable, semantics-explicit Stockfish 18 labels that can bootstrap a
strong Archon value network without poisoning the later self-improvement
flywheel. Quality is the hard constraint. Throughput and cloud utilization are
optimized only after the label contract passes local audits.

Stockfish 18 is the only production bootstrap teacher. Reckless may later label
a small stratified audit sample to measure teacher disagreement, but it is not
on the per-position critical path.

## Two label tiers, never conflated

### Full-coverage static labels

Use the patched Stockfish 18 `evalraw` v2 protocol and store separately:

- final static internal value with zero optimism;
- final value converted through Stockfish's position-dependent UCI-cp model;
- big-network PSQT and positional components;
- their unscaled internal sum and UCI-cp conversion;
- parsed rule-50 state and validity status.

These are cheap, deterministic evaluator labels, not search truth. They are
appropriate for broad pretraining and auxiliary losses. Field names and
metadata must preserve this distinction.

### Stable deep-search labels

Use single-thread cold Stockfish searches and capture completed iterative-
deepening checkpoints. These are expensive gold labels for value correction,
policy, hard-position mining, and oracle validation.

A search record is never accepted merely because a process returned a score.
Bounds, incomplete aspiration retries, missing target depths, illegal PV moves,
and unstable semantics are explicit rejection states.

## Position-state boundary

Full-FEN sources preserve side to move, castling, en passant, rule-50, and move
number; repetition history is unavailable and must remain declared. They are
eligible for both static and deep-search labels.

The normalized LC0 piece arrays omit castling, en passant, rule-50, and
repetition state. They are not eligible for deep-search oracle labels or
Stockfish's rule50-scaled final evaluation. They remain valuable for
piece-state-compatible big-NNUE component labels, representation pretraining,
policy bootstrap, and state coverage. No reconstructed default state may be
presented as the original game state.

## Gold-label gates

Every deep-search row stores every requested completed iteration, not only the
final number. Each iteration includes depth, seldepth, score, mate distance,
bound type, WDL, best move, nodes, NPS/time diagnostics, and hash fullness.

Promotion decisions are separate:

- `gold_value`: requested terminal window present; every record exact; either
  non-mate score span is at most the preregistered tolerance or mate status/sign
  is stable.
- `gold_policy`: `gold_value` plus one legal best move retained across the
  terminal window.
- `gold_joint`: both gates pass.
- `quarantined`: final tier fails; retain the trajectory and reason, but do not
  feed it to a pointwise target.

The initial scalar tolerance is 25 UCI cp across three completed checkpoints.
It may change only after a larger audit and must be versioned, never edited in
place. WDL stability is recorded separately and may support a future WDL-only
target, but saturated WDL does not retroactively validate an unstable scalar
score.

## Measured adaptive tiers

The first 16-position smoke established the provisional frontier, then a frozen
256-position stratified audit across all six local full-FEN shards measured it:

1. Tier 1: depths 18/20/22, 25M-node hard ceiling.
2. Tier 2: unresolved rows only, depths 22/24/26, 100M-node ceiling.
3. Tier 3: unresolved rows only, depths 26/28/30, 250M-node ceiling.
4. Quarantine rows unresolved after the configured final tier.

On 256 positions, Tier 1 produced 165 gold values (64.5%) and 131 gold policies
(51.2%) for 522M final nodes / 349 CPU-seconds. Tier 2 processed only the 91
value failures and added 38 gold values, reaching 203/256 (79.3%), for another
1.276B nodes / 554 seconds. Tier 3 processed the remaining 53 and added only 16
gold values, reaching 219/256 (85.5%), but cost another 3.378B nodes / 1,348
seconds. Thirty-seven positions remained quarantined.

The cost curve alone does **not** authorize Tier 1 plus Tier 2 for production.
Before fixing a default, a stratified sample of labels accepted at Tier 1 and
Tier 2 must be searched through depths 26/28/30. This measures false stability:
a score can look stable over 18/20/22 and drift again later. If that rate is not
negligible, the production floor moves deeper even if yield-per-node worsens.
Budget controls how many positions receive gold labels, never whether an
individual label receives the search required by the validated contract. Tier
3's poor marginal yield remains useful cost evidence, not permission to cheapen
labels.

The aggregate audit is
`data/nnue/quality/sf18-quality-audit-256-adaptive-summary-v1.json`. Cumulative
per-source cost was median 2.30M nodes, p95 121.7M, p99 277.5M, max 358.4M;
mean wall time was 8.79 seconds including all three tiers. A 16-position replay
reproduced every semantic iteration field and decision exactly, with timing/NPS
explicitly excluded as nondeterministic diagnostics.

## Required audit before scale

1. Select at least 256 deterministic hash-ranked positions across all six
   local full-FEN shards, stratified by phase, score magnitude, tactical
   instability, rule-50, castling rights, and checks.
2. Run all adaptive tiers and report value, policy, and joint yield separately.
3. Measure node/time percentiles and total projected CPU cost, including the
   escalation tail.
4. Rerun a deterministic verification sample from scratch and require exact
   equality of completed iteration records.
5. Cross-check FEN hash, source line/hash, perspective, legal PV, score/mate,
   WDL total, rule state, binary hash, embedded-network identity, and options.
6. Audit train/validation grouping and duplicate/transposition leakage.
7. Generate a small immutable pilot, train a protected candidate, and require
   held-out calibration plus multi-budget engine evidence before bulk scale.

## Production architecture

The labeler is resumable, content-addressed, and tier-aware. Each immutable
shard includes its source lineage, label protocol version, Stockfish binary
hash, network identity, UCI options, tier, thresholds, quality counts, and
quarantine reasons. Atomic writes and resume validation are mandatory.

Cloud workers process independent shards. Escalation manifests contain only
unresolved source IDs, so expensive tiers do not repeat accepted positions.
Spot preemption is expected: completed immutable shards survive; partial shards
are discarded and recomputed. No worker may silently change search limits or
accept the last bounded/incomplete record when a ceiling is reached.

Items 1--5 above now pass for the 256-position audit, including semantic replay,
except that early-pass false stability remains a required depth-crossing audit.
Items 6--7 also remain hard gates before bulk generation.

## Scale decision

Bulk generation begins only when the 256+ audit, deterministic replay, pilot
training, and downstream engine gates pass. The likely durable corpus is:

- static SF18 components on every state-compatible position;
- adaptive gold deep-search labels on a selected full-state subset;
- all rejected trajectories retained for later bound-aware, uncertainty, and
  search-allocation research.

This creates a high-quality bootstrap without pretending that incomplete LC0
state or shallow search scores are ground truth. Later flywheel iterations will
replace the external oracle with deeper Archon-native labels on states produced
by the current Archon.
