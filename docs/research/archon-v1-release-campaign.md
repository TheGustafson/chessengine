# ARCHON v1 public-release campaign (user directive, 2026-07-23)

The net this campaign produces is **the first public release of ARCHON**.
Master sequence (user's numbering), each phase gated by the one before:

## 1. Unlock JAX  [IN FLIGHT]
Kernel agent working the sparse-FT GPU bottleneck (Pallas embedding-bag /
segment-sum / bf16; gate ≥168k pos/s @ batch 8192). **USER RULING: there
is NO fallback — "I don't accept that we can't fix it. We must." A park
verdict means escalation (new approaches, more agents, custom CUDA if
needed), not a bullet detour.** Also ruled: NO early external anchoring —
the release bar is "very good," and the campaign runs internal-gated
until phase 7's battery.
`jax-workstream-plan.md` governs. Data acquisition chain running in
parallel (wrongIsRight + T60T70wIsRightFarseer + farseerT74/75; D46 gates
before any training use).

## 2. Stage-3 on JAX, trained INTELLIGENTLY
Everything learned this week goes in:
- **Schedule**: WSD/anneal-to-FLOOR (never zero mid-campaign — doctrine in
  policy-roadmap §LR); length adaptive via the held-out instrument
  (Elo-predictor, validated ±4), not pre-committed.
- **λ**: share-locked (~1.0) or λ/4-class — the twin proved λ is a free
  accelerator; policy dominance was an accident, make it a knob.
- **Ops**: orbax full-state checkpoints (no more fresh-Adam transients),
  telemetry armed at launch (standing rule), heldout evals per checkpoint.
- **Data**: warm from d47c-stage2-sb240; fold the NEW packs per the
  absorption hypothesis + B4 (coverage first); consider FT-sparsity
  regularization (+12-14% NPS lever, density-linear — study #6 #1 pick).
- Gates: netpack byte-parity + engine oracle + scale check + heldout curve.

## 4. Profile the stage-3 net
Reprofile agent pattern (study #6 template): hot paths, density probe
(avg-nnz now a NET-level property that moves NPS), deployment ladder.

## 5. Policy-head finetune experiments (the head bank, frozen trunk)
Multiple experiments per policy-roadmap §2/§4: Lc0 soft targets (§2d,
data download is cheap), bucketed specialists (§2c), hindsight min-over-K
(§2b) if soft top-2 disappoints, trunk-sufficiency read (§3). Gates: SF18
argmax probe + ON/OFF ablation (the sidecar must EARN deployment — it
currently buys 0 Elo and +6.2% tree).

## 6. Settle the net; decide stage-4 (the SHIP ANNEAL)
Pick the champion config (net + head set). If held-out still falling or
the anneal was floored: stage-4 = LR→~0 ship anneal (doctrine: annealing
to zero is a SHIPPING action — this is the one place it belongs).

## 6b. SMALL SPSA (user amendment, 2026-07-23): TM + a compact
high-leverage constant set, BEFORE the optimization/profiling phase —
fixes the measured clock bleed so phases 7-9 measure true. The
comprehensive tune stays at the end (phase 9), run exactly once.
**USER MANDATE (05:5x): the small SPSA MUST address the poor time
management** (measured: move-1 burns ~19s of a 60s clock; constants are
SF-fitted for SF's search, never tuned for ours). **Measurement
protocol: PRE-SPSA equal-time (10+0.1) match vs SF16 and a POST-SPSA
one — same net, same binary, only the tuned constants differ; the pair
isolates the SPSA delta. Matches run on an idle box (equal-time is
corrupt under contention) WHILE the SPSA/TM code is being written —
pre-measure launches the moment stage-3 training exits.**

## 7. Test, profile, optimize → final optimization agent
Full gate battery (fixed-node, equal-time, external anchors on the
optimized binary), reprofile, then a FINAL optimization agent (study #7)
over the shipping binary; implement its findings under the bit-parity
oracle discipline (the +30.9% arc is the template).

## 9. Comprehensive SPSA — "we go hard"
TM constants first (measured bleed; params:: runtime-tunable), then the
full search-constant retune co-adapted to the SHIPPING eval+NPS profile
(the S2 program; constants are PSQT-era stale). Local SPSA at scale;
this is the last Elo pass before release.

## Release engineering (pinned prerequisites, from earlier decisions)
- WASM + ARM/NEON targets (user: required before public release).
- Windows cross-compile → CCRL/CEGT listability (external anchors forever).
- License decision, README/docs pass, reproducibility statement (ledger
  provides provenance), embedded-or-downloaded default net decision
  (EvalFile UX — a GUI user must not get silent PSQT).
- Version: ARCHON 1.0. The bench trailer, perft suite, and full gate
  battery green on the release tag.
