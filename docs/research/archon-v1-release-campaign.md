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

## 6. THE PUMP CHAIN (user amendment 2026-07-23 ~07:0x: "we need the
## warm pump… insert a stage 5 and 6 before release")
The warm-pump pattern is validated (stage-2 +60; stage-3 tracking the
same arc); each stage breaks the annealed minimum open with an LR pump
+ ~40% rehearsal mix so NEW coverage restructures the net (B4: coverage
is the killer; low-LR fine-tuning is a proven dead-end). Training
stages before release:
- **Stage-4 (next pump): fold T60T70wIsRightFarseer + farseerT75** —
  their D46 unit audits are now ON THE CRITICAL PATH (run post-stage-3,
  never during equal-time measurement). RECIPE PINNED (user: "don't
  waste time on that AB. Just guess", 2026-07-23 ~08:0x): **peak LR
  5e-4** (stage-2's proven pump height; stage-4's new-coverage fraction
  is large and wants the stronger kick; EMA + floor-anneal insure the
  deeper bounce), warm from stage-3 sb240, EMA shadow ON (the shippable
  net), 1-2sb LR warmup ramp, λ_sparse 0.001, same stretched-cosine
  floor-anneal (end ~0.2x), 240sb, mix = T60T70+T75+rehearsal (exact
  weights set after audits size the packs). sb240 rule: NO extension —
  gate, hand off warm, next pump absorbs any end-slope.
- **Stages 5+6 SUPERSEDED → THE UNION RUN (user ruling 2026-07-23
  ~20:4x: "shuffle everything from every stage together and run one
  big stage over all of it… Let's do it. Set it up carefully").**
  Rationale: the staged pumps paid the restructuring bounce at every
  boundary and judged progress on confounded mid-chain gates; the
  union run pays the bounce once, runs a true WSD stable phase over
  the iid-shuffled union of ALL audited corpora, and folds the ship
  anneal in as the tail. This IS the pump-until-saturation plan with
  the saturation check moved inside one run.
  - **Leg A (plateau)**: warm = stage-4 endpoint-battery winner;
    constant LR 5e-4 (lr_end_superbatch=0), 2sb warmup, EMA 0.99984,
    λ_pol 0.00125 unmasked, λ_sparse 0.001; mix = PROPORTIONAL-TO-ROWS
    over the full union (existing 5 corpora + the stage-5 data-chain
    output + canon-tight-frame/hardneg-frame rescales) — proportional
    weighting = a genuine epoch of the union and satisfies the small-
    corpus epoch guard by construction. Generous sb cap (~720),
    extended by resume while the instrument still improves.
  - **Saturation instrument**: heldout+EMA per 10sb (in-run) + every
    30sb a VALUE-ONLY 100g screen (cpol stripped BOTH sides — the
    2026-07-23 sidecar finding) of the EMA checkpoint vs the fixed
    stage-4 winner. Trigger the anneal when the screen curve gains
    <~10 Elo over a trailing ~90sb window AND heldout-EMA is flat —
    judged with the no-verdicts-from-weak-tests rule, not automated.
  - **Leg B (SHIP ANNEAL)**: new config warm-started from the chosen
    plateau checkpoint; cosine lr_end_superbatch = leg length
    (~120-180sb, ~15-20%% of plateau length), warmup 0 → lands at the
    0.027x floor ≈ LR→~0. Settle-the-net = endpoint battery on the
    anneal's EMA-vs-live pair; winner is the release value net.
  - House data: SF18-relabel (canon-tight, DIV 0.674 frame fix) +
    hardneg (DIV 0.699, audit ann-0032) ride at proportional weight;
    their upweighted use returns in T11 (EDA seed), not here. test80
    stays waived (Lc0 atanh labels; nonlinear remap = future option).
SEQUENCING NOTE: the TM-SPSA (6b) + its PRE/POST SF16 equal-time
bracket must share ONE net — they slide to after the pump chain
settles; equal-time anything requires an idle box, so SPSA/brackets
and GPU training stages serialize. No early external anchoring
(standing ruling) — the SF16 bracket exists to isolate the SPSA delta,
not to anchor.

## 5b. T11 — NET-ALONE DISAGREEMENT PROGRAM (user, 2026-07-23 ~08:4x:
## "formally add this to the master plan pre launch. I see a lot of
## latent potential in this idea")
On the settled champion trunk (after the pump chain), IN the release:
1. **Residual EDA**: forward-sweep the corpora (search values already
   stored ⇒ labels ~free at 1.26M pos/s); bucket worst residuals by
   phase/material/structure/king-state. Systematic clusters → v2
   input/arch hypotheses; irreducible clusters → search's job.
2. **Disagreement head**: frozen-trunk head on the residual target;
   deploys through the dormant D40_UNC_HEAD sidecar seam. Gate: ON/OFF
   ablation (heads must EARN deployment).
3. **Allocation consumption** (D41/design-19 resumes here): deeper
   search where predicted disagreement is high (extension/reduction/TM
   modulation). Gate: SPRT. ORDERING: must land (or be dropped) BEFORE
   phase 9's comprehensive SPSA so the final constants co-adapt with it.
Full mechanics: arch-ablation-suite.md §T11. EDA caveat: v1 labels
conflate net-error with SF-frame difference — fine for the head, but
arch conclusions want an own-search oracle slice (self-play flywheel).

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
