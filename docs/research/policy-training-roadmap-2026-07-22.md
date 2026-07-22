# Policy-head training roadmap (post-D47c)

*2026-07-22. Context: d47c-1280-240sb is mid-flight — the first net with the
D47 shared-bottleneck 1858-class policy head, trained jointly at constant
λ=0.005 (measured policy≈value parity at init). This doc captures where policy
training goes AFTER the current run, prompted by the question: would policy
benefit from being trained independently at some point? Short answer: yes —
but as a separate training PHASE on the shared trunk, not a separate network.*

## 1. The framing: joint forms the trunk, independent specializes the head

The D47 design bets that value and policy share representation and that policy
inference rides the value forward pass almost free (selected-row int8 MACs on
the already-computed 128-d bottleneck). Nothing here revisits that. The open
question is only **where the policy head's weights come from**:

- **Joint (current)**: one run, λ-weighted sum, both heads shape the trunk.
  Multi-task training may regularize the trunk; or it may tax value. The
  λ=0 twin (same seed/data, policy off) is the attribution instrument —
  compare final value component + fixed-node Elo, d47c vs twin.
- **Independent (proposed follow-on)**: freeze a finished trunk, retrain ONLY
  the policy head — embedding E[1858×128] + bias[1858] ≈ 240k params — on
  policy-specialized data. No backprop through the ~107M FT params: tiny
  memory, minutes-to-hours not 8h, and value output is bit-identical by
  construction (the trunk never moves). This makes policy-data iteration
  cheap and completely decoupled from the value grind.

A fully separate policy network (own accumulators/trunk) is REJECTED: it
doubles per-node inference where ordering runs and forfeits the shared-
bottleneck efficiency that makes deployment viable at all.

## 2. The strongest argument for a separate phase: data-appetite mismatch

The D46 scalar-value contract quiet-filters training rows hard — correct and
mandatory for scalar labels, but it deletes exactly the rows where move
ordering earns its keep. Policy is consumed *inside the search tree*, which is
disproportionately noisy/tactical; best-move labels (unlike scalar evals) are
perfectly well-defined on non-quiet positions. Joint training forces both
heads to eat the value head's filtered diet. Independent head training can
feed policy what it actually needs:

- **Unfiltered best-move rows** from existing corpora (the quiet filter holds
  back rows for VALUE reasons; the move labels on the rejected rows are fine).
- **Hard-negative mining**: positions where ARCHON's search chose a move the
  SF18 oracle rejects (`mine_archon_losses.sh` / `mine_archon_reckless.sh`
  infrastructure already exists for the value analogue).
- **Deep-teacher labels**: SF18 tiered protocol
  (`docs/research/sf18-label-quality-plan.md`) — deep best-move labels are
  much cheaper than deep scalar labels (only the argmax must be right).
- **Soft targets** (parked option C): multiPV teacher → distribution over
  legal moves instead of one-hot. Softmax-CE already supports it; needs a
  target-format extension (top-k ids + weights per row) in the loader.

## 2b. Two-move policy via hindsight loss (user idea, 2026-07-22 — NO new data)

Adapt the head to propose TWO moves, scored so the loss rewards either being
the teacher move — the classic **multiple-choice / hindsight (min-over-K)
loss**: two embedding matrices E₁,E₂ (≈240k params each), per-position loss
= min(CE₁, CE₂) + a small load-balance term (known failure mode: one head
hogs every sample). Only the closer head gets gradient → heads SPECIALIZE
(tactical vs positional candidate in ambiguous positions) instead of one
calibrated distribution hedging. Under the 128-d bottleneck constraint this
reallocates capacity toward exactly what AB ordering consumes: "was one of
my first two tries right." Same one-hot labels — zero new data.

Sequenced cheapest-first:
1. **FREE measurement first**: the current CE head already ranks all 1858
   classes — read its top-2/top-3 coverage (needs a tiny engine top-k print,
   post-run; calibrated expectation ≈30% top-2 at pol_ce 2.58 vs 20% top-1).
   Hindsight training only pays if it beats CE's own top-2 meaningfully.
2. If warranted: **frozen-trunk head finetune** (§4 machinery) with min-CE —
   minutes-to-hours, zero value risk, no full retrain.
3. **Slot design + ablation**: move #1 keeps the empty-TT slot; move #2 goes
   AFTER good captures / before quiets (safe variant) — ON/OFF ablate like
   sb30. Note for joint training later: min-CE shrinks pol_ce's scale ⇒ λ
   would need rebalancing; in head-only finetunes this doesn't arise.

## 2c. Bucketed policy specialists (user idea, 2026-07-22 — route on piece count)

Per-bucket policy embeddings E_b[1858×128] routed on piece count, mirroring
the value head's 8 output buckets. **Free at inference** (selected-row MACs
with a bucket offset — zero NPS), ~1.9M params at 8 buckets, and the trainer
already has the machinery (the value tail's `select(buckets)` applied to a
per-bucket polmove affine). Case is arguably STRONGER than for value: move
priors shift drastically by phase (king activity, pawn pushes, piece play).
Warm-start every E_b from the trained shared E ⇒ runnable as a frozen-trunk
head finetune, no new data. Known cost: bucket data imbalance (endgame
buckets see less data — value head already tolerates this). Composes with
§2b: buckets × hindsight hypotheses. T5-adjacent: queens-on/off may be a
better routing axis for POLICY than piece count (tactical-regime flip) —
cheap to A/B at head-finetune cost.

## 2d. Lc0 soft policy targets (user idea, 2026-07-22 morning)

The single richest FREE policy-data upgrade: Lc0 training chunks carry full
visit distributions (soft targets from ~800-visit PUCT) — and **our 1858
policy vocab IS Lc0's move encoding**, so their policy vectors map 1:1 onto
our head. Precedent: official SF has trained on rescored Lc0 data since
SF15 (provenance is community-standard). Plan, cheapest-first:
1. **Frozen-trunk head finetune on Lc0 soft CE** (§4 route 1, cached
   activations — CPU-scale, runnable DURING stage-2). This is the clean
   version of "introduce their policy late in training": after the trunk.
2. Gates: SF18 top-1 probe (deployment consumes ARGMAX — Lc0's argmax ≠ SF
   best; watch for calibration-up/argmax-down) + ON/OFF ordering ablation.
   Mitigation if argmax slips: mixed loss (Lc0 soft + SF one-hot).
3. Mask Lc0 VALUE labels entirely (own-net Q/WDL, wrong scale; rescore
   later if ever needed).
4. If soft wins → wire soft-CE + mixed curriculum into the next cold joint
   run (d47d); loader needs the top-k-ids+weights row extension (§2).
5. Bridge to `hybrid-ab-puct-search.md` H3: same loader/loss later consumes
   OUR root-PUCT visit distributions from self-play datagen.
Domain-shift note: their self-play position distribution ≠ our UHO/AB tree;
the frozen-trunk route sidesteps half of this (trunk stays on our data).
Data: storage.lczero.org (slice of a recent run, ~50-100M rows to start;
public HTTP download, disk is ample).

## 3. The prerequisite experiment: trunk sufficiency (run after sb240 + twin)

Both finished trunks — d47c (joint) and the λ=0 twin (value-only, same seed,
same data) — get an identical frozen-trunk head finetune on the same policy
data. Compare top-1 accuracy vs the SF18@50k teacher (canon400.sf18 probe +
a larger held-out set for tighter error bars than ±1.8%).

| Outcome | Reading | Consequence |
|---|---|---|
| twin-trunk head ≈ joint-trunk head | trunk features are policy-sufficient for free | all future policy work = cheap head finetunes; λ can go to 0 in value runs (no tug-of-war, no share drift to babysit) |
| joint-trunk head clearly better | joint loss buys real policy representation | keep λ in the trunk recipe; STILL finish with a head-finetune polish pass on tactical data |
| both heads plateau at low accuracy | 128-d bottleneck or trunk is the ceiling | revisit branch point (wider bottleneck / branch before L1) before buying more data |

Also worth reading off the same experiment: how far head-only training pushes
accuracy above the joint run's own number — that delta is the "specialized
data + dedicated optimization" dividend, measured at zero risk to value.

## 4. Mechanics of a frozen-trunk head finetune

Two implementation routes, cheapest first:

1. **Offline on cached activations (preferred).** Dump `(h128 bottleneck,
   legal-move ids, teacher move_id)` tuples once per position — via the
   engine's `last_l2_for` path or a trainer eval pass — then train the head
   as a standalone masked-softmax logistic regression (numpy/torch, CPU or
   tiny GPU; the whole model is 240k params). No bullet involvement, no
   12GB buffers, no OOM risk, iterate on data recipes in minutes. Quantize
   to the CPL1 sidecar with the existing netpack path (D40_CPOL_OUT).
2. **In bullet with the trunk frozen** (lr=0 / no-grad on non-head params).
   Heavier, but exercises the exact production graph; use it to confirm the
   offline result before shipping a sidecar.

Either way the deliverable is just a new `.cpol` beside the SAME `.cenn` —
the deploy story (sidecar auto-load, Zobrist-stamped activation stash,
ordering fills the empty-TT slot) is already built and does not change.

## 5. Joint-training refinements (the other branch, if joint wins §3)

- **λ anneal/decay**: constant λ demonstrably drifts policy-dominant (share
  0.81→1.58 by sb10, still rising at decreasing rate). If the sb30+ Elo says
  the drift hurts, the first fix is λ ∝ value-component (share-locked), or a
  cosine decay to ~λ/4 — user has pre-approved "lambda decay can come later."
- **Two-stage**: value-only pretrain → short joint finetune (policy share
  high, LR low). Cheaper than full joint if joint's benefit concentrates
  late; risks eval-search-style co-adaptation issues — test, don't assume.

## 5b. The λ tax and presentation count (stage-2 decision rule)

Joint training dilutes value's share of the trunk gradient (~37% at the
observed share plateau ~1.7), so a schedule calibrated for value-only
training is in expectation slightly short for joint. Compensation = more
presentations — via a **stage-2 warm retrain** (the most validated recipe we
own: it built the incumbent chain), NOT by stretching a pinned schedule or
raw-repeating epochs (B4: coverage beats epochs; stage-2 can add fresh data).

Decision rule at sb240, two free measurements:
1. **Twin gap**: d47c vs λ=0 twin at identical presentations — the value
   component + fixed-node Elo gap IS the tax, measured exactly.
2. **End slope**: `loss/value_component` still visibly falling at sb240 →
   undertrained; flat since ~sb180 → tax absorbed, extension buys little.

Either signal positive → launch `d47c-stage2`: warm start from sb240, LOW
λ (~λ/4, not zero — at λ=0 the trunk drifts under frozen policy rows and
accuracy silently rots; small λ keeps the head aligned while value takes
~full gradient). This is "lambda decay can come later" materialized across
stages.

## 6. Downstream consumers beyond first-move ordering (later, D41-adjacent)

Once policy accuracy is credible, the same logits can inform more than the
TT-slot fill: policy-aware LMR (reduce less for high-prior moves),
prior-scaled history seeding in the MovePicker, and eventually the design/19
learned-search-allocation program (P15/M11) gets a native prior signal.
Ordering-first is deliberate: it is the lowest-risk consumer (wrong policy =
one wasted first move, not a pruning error). The radical endpoint of this
ladder — a hybrid AB/PUCT search where a rich head (priors + WDL/σ) drives
both an AB core and a root-PUCT allocator, plus the visit-distribution
datagen flywheel — is pinned with its own experiment ladder (H0–H3) in
`hybrid-ab-puct-search.md`.

## 7. Sequencing

1. d47c → sb240, gate_d47.sh (learnability-first readouts).
2. λ=0 twin auto-fires on train exit → three-way attribution
   (incumbent / twin / d47c) per the live-log plan.
3. **Trunk-sufficiency head finetune (§3)** on both trunks — the decision
   fork for everything above.
4. Then either the cheap head-iteration flywheel (§2 data, §4 route 1) or
   λ-schedule work (§5), per the fork.

Success metrics throughout: top-1 vs SF18@50k (bigger probe set), fixed-node
Elo with ordering ON vs OFF (same net, sidecar removed = bit-identical
control), and node-count deltas at equal depth (ordering quality shows up as
tree shrinkage before it shows up as Elo).
