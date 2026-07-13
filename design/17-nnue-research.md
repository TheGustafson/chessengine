# 17 — NNUE research: the search-integrated network

Status: Living (research roadmap — designed now, built incrementally post-P13
behind Tier-B triggers, like doc 16)

This document is the roadmap for the engine's differentiating bet (D34). A
conventional NNUE is a value function trained on deep-search labels and then
**bolted onto a hand-tuned search** — the two are developed separately and
never talk to each other. Our bet is different: a **search-integrated,
shared-trunk, multi-head network** in which evaluation *and search control* are
**co-trained** from the observation store (D29), so the two co-evolve. Two
flagship thrusts carry the bet:

- **Thrust I — uncertainty coupling.** The network predicts *how much a search
  will move the eval* at a node (the deep-search residual). That one signal
  drives extensions, reductions/pruning, and time management — the engine
  spends nodes where they change the answer.
- **Thrust II — learned policy & search control.** A policy/control head on the
  *same trunk* guides move ordering, reductions, and pruning — moving us off
  D20's hand-tuned constants toward learned control, safely and gradually.

The thesis of why this can **beat Stockfish** despite SF's enormous
eval-tuning lead: at fixed time, strength is *decision quality per
node-microsecond*, not centipawn accuracy per position. A smaller, cleaner
engine that **searches smarter** — allocating effort by learned uncertainty and
learned policy — can out-*decide* a bigger engine that evaluates marginally
better but searches by hand-tuned heuristics. That is the lever this doc is
built to pull. It is aspirational and genuinely research (§9 lists the ways it
can fail), but it is the explicit goal, and the data platform (D29) exists
precisely so the signals these heads need are preserved rather than thrown away.

---

## 1. Purpose and scope

**This document owns:**

- The **research thesis** (§3.1): search-integration (not raw eval strength) as
  the lever to beat a stronger-evaluating opponent, and the compute-budget
  framing that governs every idea here.
- The **shared-trunk multi-head architecture** (§3.2): one accumulator, many
  cheap heads (value, WDL, uncertainty, policy, control-correction), and the
  amortization argument that makes the extra heads affordable in-tree.
- **Thrust I — uncertainty coupling** (§3.3) and **Thrust II — policy & learned
  search control** (§3.4), each developed to first-experiment-and-metric
  resolution.
- The **supporting thrusts** at lower resolution (§3.5): cascaded/cost-aware
  evaluation, mixture-of-experts routing, and threat/relational input features
  (the one near-term item, into P13's first net).
- The **corpus → target → experiment map** (§3.6): which doc-16 corpus/layer
  supplies each head's labels, the first experiment, the online metric, the
  risk, and the per-node cost class.
- The **compute-budget ranking discipline** (§3.7): the governing constraint —
  rank ideas by expected decision-quality gain per node-microsecond, measured
  against P13's *real* inference nps.
- The **staging & triggers** (§3.8): build-now vs designed-deferred for each
  thrust; nearly all are Tier-B / post-P13.

**This document does NOT own:**

- The **base NNUE architecture, quantization, net format, and training recipe**
  — doc 11 §3.4. The trunk and heads *live in* doc 11's network; this doc adds
  heads and their training targets and says how each couples to search.
- The **observation-store schema, corpora, and provenance** — doc 16. This doc
  *consumes* doc 16's five corpora and four record layers as label sources; it
  defines no storage.
- The **datagen bootstrap and curation** — doc 15 (and the threat-feature
  record amendment §3.2.3 there that Thrust-adjacent features need).
- The **search sites the heads inform** — the extension/reduction/pruning
  points (doc 10), MovePicker ordering (doc 09), the ID loop and node types
  (doc 07), time management (doc 12). The heads *emit signals into* those
  seams; the seams are owned there.
- The **inference/threading mechanics** — doc 11 §3.4.6 (SIMD/quant) and doc 13
  (per-worker). This doc states the *budget*; those docs meet it.
- The **ML training/model code** for the heads — discardable, non-engine
  research tooling (doc 16 §3.5.1); may use any deps. This doc defines the data
  contract and the engine-side inference/coupling only.

## 2. References

- `design/decisions.md` — **D34** (this direction — the authoritative decision
  this doc elaborates), **D29** (research data platform — the corpora/layers
  that supply every target here), **D28** (two-tier bootstrap: the heads train
  on Tier-B clean data), **D24** (the `Bench:` fingerprint — heads-off must not
  move it), **D22/D21** (the `evaluate` seam and eval roadmap the value head
  extends), **D20** (the hand-tuned search constants the control head learns to
  *correct*), **D27** (ideas only, never copied source), **D14/D15** (unsafe /
  zero-dep budgets the engine-side inference respects).
- `design/11-evaluation.md` §3.4 — the base NNUE. The trunk (accumulator) and
  every head are additions to that network; §3.4.6 (SIMD/quant) is the nps
  budget these heads spend.
- `design/16-research-data-platform.md` — the label sources: the four record
  layers (§3.2), the five corpora (§3.6.1), and the five research-target
  categories (§3.6.2) map directly onto §3.6 here.
- `design/15-nnue-datagen.md` — the bootstrap; §3.2.3 is the record amendment
  the threat features (§3.5) require.
- `design/10 / 09 / 07 / 12` — the search-control, ordering, node-type, and
  time-management seams the heads feed.
- `design/13-threading-smp.md` — per-worker inference; the heads run on the same
  accumulator the search thread already maintains.
- No new external references. Research model code is discardable tooling (doc 16
  §3.5.1); the engine crate stays zero-dep (D15).

## 3. Design

### 3.1 Thesis: search-integration as the lever to beat a stronger evaluator

Stockfish has a multi-year, thousands-of-core-years lead in *eval tuning*. We
will not beat it at centipawn-accuracy-per-position, and we should stop trying
to. The bet is orthogonal: **at fixed time, playing strength is decision quality
per node-microsecond.** Two engines that evaluate a position equally well but
allocate their node budget differently can differ by hundreds of Elo. SF
allocates by *hand-tuned* heuristics (its LMR table, futility margins, singular
rules — our D20 transcriptions). We propose to allocate by **learned** signals
derived from our own search's behaviour:

- spend depth where the eval is *uncertain* (Thrust I), and
- order/reduce/prune by a *learned policy* rather than history heuristics alone
  (Thrust II).

The claim is not that this is easy — it is that it is the *only* lever where a
clean, well-instrumented engine has an advantage over a giant hand-tuned one,
because the training signal (our search's internal decisions and their
outcomes) is something we can generate and SF cannot hand us (doc 16 §3.8). The
whole engine — the D29 platform, the P9 heuristics we can now supervise, the
Tier-B self-play flywheel — is arranged to make this signal cheap to collect.

Corollary that governs the whole doc: **because these signals are consumed
*inside the search tree*, cost is everything** (§3.7). A perfect uncertainty
estimate that costs 5 µs/node is worthless; the same estimate amortized to
~0 µs on an accumulator we already compute is transformative.

### 3.2 The shared-trunk multi-head network

The base NNUE (doc 11) already computes an **accumulator** — the perspective
feature-transformer output — once per node, maintained incrementally across
make/unmake. That accumulator is the single most expensive thing the eval does.
The architecture here is: **compute the trunk once, read it with many small
heads.**

```
        incremental accumulator  (the trunk — doc 11, paid for already)
                    │
   ┌──────────┬─────┴─────┬───────────┬─────────────────┐
   ▼          ▼           ▼           ▼                 ▼
 VALUE      WDL       UNCERTAINTY   POLICY        CONTROL-CORRECTION
 (cp)     (w/d/l)   (Δ residual)  (move prior)  (Δ to hand-tuned LMR/prune)
  doc11    doc11      Thrust I      Thrust II          Thrust II
```

Each head is a small affine (± one hidden layer) on the shared trunk — cheap in
both compute and parameters relative to the trunk. This is the crucial
practical fact: **a policy or uncertainty head is nearly free because the
trunk it reads is a sunk cost.** The heads are trained *jointly* from the
observation store (one forward pass, multiple losses, §5), which also
regularizes the trunk (multi-task learning tends to improve the value head, not
just add capabilities). Heads live behind net-format flags so a net that lacks
a head degrades to value-only, and a *disabled* head is compiled/branched out
so the D24 bench fingerprint is unmoved (§6).

Design intent: the value/WDL heads ship first (P13, doc 11). Uncertainty,
policy, and control-correction are added to the *same trunk* as Tier-B data
arrives — additive, not a rearchitecture.

### 3.3 Thrust I — uncertainty coupling

#### 3.3.1 The signal (free label)

For every node the engine searches, doc 16 already captures the **static eval**
(Layer 2 `static_eval`) and the **deep-search result** (Layer 3 `final_score`).
Their difference is the **deep-search residual** — how wrong the cheap static
eval turned out to be once searched. That residual, aggregated over the store,
is a *free* supervised label: no extra generation, it falls out of data we
collect for value training. The **uncertainty head** predicts the residual's
magnitude `û(pos) ≈ E[ |static_eval − deep_score| ]` (and optionally its sign /
a coarse distribution — §9f).

#### 3.3.2 The three couplings (one scalar, engine-wide effect)

`û` is a per-node estimate of "how much does searching this actually matter,"
and it plugs into three search sites:

1. **Extensions (doc 10).** Extend nodes with high `û` — the positions where the
   static picture is most misleading and depth pays. A *learned* extension
   trigger, replacing/augmenting the hand-tuned singular and check extensions.
2. **Reductions & pruning (doc 10).** Reduce and prune *harder* where `û` is low
   — confident, quiet positions the eval already reads correctly. This is the
   high-volume win: most nodes are low-uncertainty, and safely spending less on
   them frees depth for the few that matter.
3. **Time management (doc 12).** At the root, allocate clock by the uncertainty
   (and its change across ID iterations) of the *best* and *contending* moves —
   think longer when the top moves are genuinely unclear, move fast when they
   are not. This is a principled replacement for the best-move-stability
   heuristic P10 uses.

#### 3.3.3 Why it is the right first thrust

Cheapest and lowest-risk of the novel set: (a) the label is free (§3.3.1); (b)
the head is one affine on the shared trunk (§3.2), ~0 µs amortized; (c) it is
**auxiliary** — it never changes the returned value, only *where the search
spends*, so a bad head costs Elo but cannot corrupt eval or produce illegal
play; (d) it degrades gracefully — with the head off, the couplings fall back to
the hand-tuned D20 behaviour.

#### 3.3.4 First experiment and metric

Train the uncertainty head jointly with the value net on the Tier-B value
corpus (residual is already there). Wire coupling **#2 only** first (reduce more
where `û` low), behind a params flag. Metric: fixed-node proxy + real 10+0.1
SPRT vs the value-only baseline; success = node reduction at equal strength, or
Elo at equal nodes. Then add #1 (extensions), then #3 (TM), each SPRT-gated.

### 3.4 Thrust II — learned policy & search control

> **Status update (2026-07-13).** Thrust II is now driven by **D40** (design/11
> §12): the SOTA net trains a **triple-net grid** — {eval-only, eval+**small**
> policy (rank-8 bilinear from×to), eval+**full** policy (4096 from×to)} — sharing
> one trunk architecture, so the policy head is developed with a permanent
> eval-only control. **First in-search result is in and it is NEGATIVE for the
> obvious use:** an offline-trained bilinear policy (Lc0 target, 32% top-1) blended
> *additively* into MovePicker quiet ordering gave a net-neutral tree on 400
> realistic positions (geomean ~1.0) — it helps endgames but hurts open/middlegame
> where search time is spent (see the memory note `policy-integration-status`). So
> the naive "policy → quiet ordering" hook is parked; the live hypotheses for the
> policy heads are **policy-as-pruning/reduction-confidence** and **policy-only-where-
> history-is-cold** (§3.4.2's control-correction framing), tested on the D40 nets.
> Neither SF nor Reckless has a policy head — this remains genuinely novel (D34).

#### 3.4.1 Policy head (move prior, amortized)

A **policy head** on the shared trunk emits a prior over moves (or a score per
move-from-square/to-square factorization to keep it small). Targets come from
doc 16's **Root Move Analysis** corpus (every legal move analysed → which was
best, the ranking, the regret) and **Internal Search** corpus (which move caused
the cutoff). Uses:

- **Move ordering (doc 09).** Blend the policy prior into MovePicker scoring
  alongside history — a learned prior is strongest exactly early in a game/line
  where history is cold.
- **Policy-weighted reductions (doc 10).** Reduce low-prior moves more (they are
  a-priori unlikely to be best) — the LMR analogue of AlphaZero's policy
  focusing MCTS.

The usual objection ("a policy net is too slow for alpha-beta's node rate") is
answered by §3.2: the prior is a small read on an accumulator we already
maintain, evaluated once per node and shared across all its moves — not a fresh
network per move.

#### 3.4.2 Control-correction (the safe path off hand-tuning)

Rather than *replace* the P9 LMR/pruning with a per-node net (dangerous: a net
in the hottest loop wrecks nps, and a wrong learned reduction destabilizes
search), learn a **correction** to the hand-tuned decision, supervised by doc
16's **Counterfactual** corpus (reduced-vs-full and pruned-vs-unpruned pairs —
the pair *is* the label for "was that reduction/prune safe here?"). The control
head emits a small Δ that adjusts the D20 formula and **falls back** to it when
the head is absent or unconfident. This mirrors the engine's existing
*correction-history* idea (doc 09) — a learned nudge on a sound default — lifted
from eval to search control. It is the incremental, reversible route toward the
learned search the thesis wants, without betting the search's stability on a
net in one step.

#### 3.4.3 The per-node cost discipline (hard constraint)

Everything in Thrust II runs *inside the tree*. Non-negotiable: the policy read
is once-per-node on the shared trunk; the control Δ is a tiny affine or a table
indexed by cheap features; neither may add a second network evaluation per move.
Any in-tree head that cannot be amortized to near-zero µs/node against P13's
measured inference nps (§3.7) is rejected regardless of signal quality.

#### 3.4.4 First experiments and metrics

Policy: train the head on Root Move Analysis, wire it into MovePicker scoring
first (ordering only — no reductions yet). Metric: node reduction at equal depth
(better ordering ⇒ more cutoffs on the first move), then SPRT. Control: start
with the *single* highest-value counterfactual axis (LMR-depth correction),
SPRT-gated, fallback-safe.

### 3.5 Supporting thrusts (lower resolution)

- **Cascaded / cost-aware evaluation.** A cheap net on most nodes; an expensive
  net only where a **learned gate** predicts it flips the decision — supervised
  by doc 16's cheap-vs-expensive Counterfactual axis. Principled "lazy eval."
  Real upside on nps-bound hardware; risk is the gate's own cost. Deferred
  behind Thrusts I–II proving the flywheel.
- **Mixture-of-experts / learned routing.** NNUE output buckets (by material)
  are already a *static* MoE; generalize to *learned* routing over specialized
  experts (phase / pawn-structure / king-safety regimes), supervised by the
  different-experts Counterfactual axis. Heaviest (memory, routing, training);
  last.
- **Threat / relational input features (near-term).** The one buildable-now
  item: relational attacker/victim planes into a second accumulator (Reckless
  study, `reckless_reference/11-nnue-features-threats.md`) — a *proven* eval
  upgrade that belongs in P13's first strong net. Needs the doc 15 §3.2.3
  record amendment. Input-representation SOTA, independent of the head research
  above but sharing the trunk.

### 3.6 Corpus → target → experiment map

Every head is a *cheap derivation* from a doc-16 corpus/layer (D29). "Cost"
is the per-node in-tree evaluation cost class.

| Head / target | doc-16 source | First experiment | Online metric | Cost | Risk |
|---|---|---|---|---|---|
| Value / WDL | Layers 1+3 + episode WDL (Self-Play) | P13 baseline net | SPRT vs PSQT | trunk (paid) | low (known) |
| **Uncertainty** (Thrust I) | Layer 2 vs Layer 3 residual (free) | reduce-more-when-confident | nodes @ equal Elo | ~0 (amortized) | low (auxiliary) |
| **Policy** (Thrust II) | Root Move Analysis + Internal Search (L3/L4) | prior into MovePicker | cutoffs on 1st move | ~0 (amortized) | med (must beat history) |
| **Control-correction** (Thrust II) | Counterfactual pairs (L4) | LMR-depth Δ, fallback-safe | SPRT | tiny (Δ affine) | med-high (stability) |
| Cascade gate | cheap-vs-expensive Counterfactual | route to big net when it flips | nps @ equal Elo | gate cost | med (break-even) |
| MoE router | different-experts Counterfactual | learned bucket routing | SPRT | router + experts | high (complexity) |
| Threat features | (feature-set, needs doc 15 §3.2.3) | second accumulator | SPRT vs no-threats | trunk (paid) | low (proven) |

### 3.7 The compute-budget ranking discipline (governing rule)

Rank every idea by **expected decision-quality gain ÷ per-node microseconds**,
measured against P13's *real* measured inference nps on the target hardware —
not against theory. Consequences that are binding, not advisory:

- In-tree heads (policy, control, cascade gate) must be **amortized on the
  shared trunk or be tiny**; a head that needs a second network evaluation per
  node is rejected.
- Auxiliary heads that only *steer* the search (uncertainty) are preferred first
  — they cannot corrupt correctness, only spend.
- The SIMD/quantization efficiency (doc 11 §3.4.6) is not a separate workstream
  from this research: it is the **budget** that decides how much of this doc is
  affordable. Every µs shaved off the trunk is a µs a head can spend. Re-rank
  §3.6 once P13 inference nps is measured.

### 3.8 Staging & triggers (build-now vs deferred)

Like doc 16 §3.7: designed in full now, built behind the research that consumes
each piece. Almost everything is Tier-B / post-P13.

| Item | Class | Trigger |
|---|---|---|
| This roadmap + D34 | **BUILD-NOW** | this doc |
| Threat/relational input features | Near-term | P13 first strong net (+ doc 15 §3.2.3 amendment) |
| Shared-trunk multi-head *scaffold* (value+WDL heads) | P13 | doc 11 Phase C net |
| Uncertainty head + coupling #2 | Deferred | Tier-B value corpus exists (residual is free) — the FIRST novel thrust |
| Uncertainty couplings #1, #3 | Deferred | uncertainty head proven |
| Policy head + ordering use | Deferred | Root Move Analysis corpus (doc 16) |
| Policy-weighted reductions | Deferred | policy ordering proven |
| Control-correction (LMR-depth Δ) | Deferred | Counterfactual corpus (doc 16) |
| Cascade gate | Deferred | cheap-vs-expensive Counterfactual + a second net |
| MoE router | Deferred | experts exist + different-experts Counterfactual |

## 4. Implementation plan

Commit-sized; nearly all deferred with triggers. No engine code lands from this
doc beyond what its owning docs (11/16) build; this doc sequences the research.

1. **BUILD-NOW (this doc):** ratify the thesis (§3.1), the multi-head
   architecture (§3.2), both flagship thrusts (§3.3–3.4), the corpus map (§3.6),
   the budget discipline (§3.7), the staging (§3.8). Pin **D34**; index in
   `design/README.md`; add a master-plan research-track note.
2. **Near-term (P13, doc 11/15):** ship the value/WDL net with the shared-trunk
   scaffold and (stretch) threat features — so heads are *additive* later.
3. **Deferred (Tier-B value data):** the **uncertainty head** + coupling #2
   (§3.3.4) — the first novel result, because its label is free.
4. **Deferred (Root Move Analysis):** the **policy head** + MovePicker use
   (§3.4.4).
5. **Deferred (Counterfactual):** **control-correction** (§3.4.2), fallback-safe,
   one axis at a time.
6. **Deferred:** the uncertainty extensions/TM couplings, cascade, MoE — each
   gated on the prior proving out.

## 5. Testing and validation

- **Offline head quality** before any game: uncertainty — residual-prediction
  calibration (predicted vs actual `|static−deep|` on the frozen holdout, doc 16
  §3.7); policy — top-1/top-k best-move accuracy and cutoff-move recall;
  control — does the predicted Δ agree with the counterfactual ground truth.
- **Online, SPRT-gated** (D25): every coupling is a search-behaviour change and
  is proven in games (fixed-node proxy + real 10+0.1) vs the value-only +
  hand-tuned-search baseline. A head that is offline-good but online-neutral is
  not shipped.
- **Bench-neutrality (D24):** with a head disabled, the `bench` fingerprint is
  bit-identical to a build without it — heads and couplings are behind
  net-format flags / cold branches (§6).
- **Honest holdout:** all offline metrics use doc 16 §3.7's frozen, unbiased
  holdout reserved by canonical key, so head quality is not measured on training
  positions.
- **Multi-head training doesn't degrade value:** the joint-trained value head
  must SPRT ≥ a value-only net (multi-task should help, not hurt — §9d).
- **Axis hygiene (D39):** rank nets by **fixed-node** strength (it isolates net
  quality from inference speed); real-TC is a ship gate, never a net-quality
  read. Log every net's `nps` (D38) so a real-TC result decomposes into
  net-quality + speed Elo. Any arch/quant change → **fixed-node screen →
  production inference → real-TC**, never judged on a first-cut inference path;
  and **SPSA-tune on the shipped net, not net-0** (tuning is eval-coupled). Full
  protocol + worked example: `design/14` §3.11.

## 6. Performance / budget notes

- **Trunk is the sunk cost.** The accumulator is computed once per node and
  maintained incrementally (doc 11); every head is a small read on it. The
  design lives or dies on keeping heads *reads*, not *networks* (§3.7).
- **Heads-off is bench-neutral.** Head inference and each search coupling sit
  behind net-format capability flags and params gates; the tournament net that
  lacks a head, or has a coupling disabled, runs the doc-11 value path
  unperturbed (D24).
- **SIMD/quant is the enabler.** Doc 11 §3.4.6's inference efficiency sets the
  µs/node budget the heads spend; this doc's §3.7 ranking is re-run against the
  *measured* nps, not projected. Uncertainty (auxiliary, ~0 µs) is affordable
  essentially always; in-tree policy/control are affordable only if amortized.
- **Training compute:** the heads add loss terms, not a separate pipeline — one
  forward pass, multiple losses, on the existing trainer (doc 11 §3.4.6 /
  discardable tooling). The 4060-class GPU handles the trunk + heads for a first
  net; large cascade/MoE sweeps want the bigger machine (doc 16 §9).

## 7. Interactions

- **doc 11 (base net).** The trunk and every head are additions to doc 11's
  network; doc 11 owns arch/quant/format/training, this doc owns *which heads
  exist, what they learn, and how they couple to search.*
- **doc 16 (data platform).** Supplies every label: uncertainty ⟵ Layer 2/3
  residual; policy ⟵ Root Move Analysis + Internal Search (L3/L4); control ⟵
  Counterfactual (L4); cascade/MoE ⟵ Counterfactual axes. This doc is the
  primary *consumer* of doc 16's deferred corpora and the concrete reason they
  are worth building.
- **doc 10 (search heuristics).** Uncertainty and control heads emit signals
  into the extension/reduction/pruning sites doc 10 specifies; control-correction
  is a learned Δ on D20's constants — the closing of the loop doc 16 §7 names.
- **doc 09 (ordering).** The policy prior blends into MovePicker scoring.
- **doc 12 (time management).** Uncertainty coupling #3 replaces the
  best-move-stability clock heuristic with a learned uncertainty allocation.
- **doc 07 / 13.** Node types and the ID loop provide the coupling sites; the
  heads run per-worker on the accumulator the search thread already maintains.
- **D34 / D29 / D28.** This doc is the D34 spec; it consumes the D29 platform and
  trains on D28 Tier-B clean data.

## 8. Acceptance criteria (D26)

The research direction is **correctly designed** when:

1. The thesis (§3.1) — search-integration as the lever, and the
   decision-quality-per-µs framing — is stated, with the honest claim that this,
   not eval accuracy, is where we can beat a stronger evaluator.
2. The shared-trunk multi-head architecture (§3.2) and its amortization argument
   are concrete (heads are reads on a sunk-cost trunk; heads-off is bench-neutral).
3. Both flagship thrusts (§3.3, §3.4) are developed to a **first experiment with
   an online metric**, with uncertainty justified as the low-risk first mover
   and control-correction justified as the safe route off hand-tuning.
4. Every target is mapped to its doc-16 corpus/layer and a first experiment
   (§3.6 table), so nothing depends on data the platform did not preserve.
5. The compute-budget ranking discipline (§3.7) is stated as binding, with
   SIMD/quant named as the budget, not a side quest.
6. The staging table (§3.8) classifies every thrust build-now/deferred with a
   trigger, and the build-now set is exactly `{ this roadmap+D34, and — into
   P13 — the shared-trunk scaffold + threat features }`.
7. The doc does not re-own doc 11 (net) or doc 16 (store) material.

Each **thrust** is *done* when its head, trained on the relevant Tier-B corpus,
**SPRT-beats** the value-only + hand-tuned-search baseline at equal hardware
(§5) — and the program's north-star acceptance is an SPRT win vs a
contemporary Stockfish at equal time control on equal hardware (§3.1; a
post-P13 milestone to be added to the master plan when the first thrust lands).

## 9. Open questions (the real research risks)

1. **Does the residual head generalize or memorize?** `|static−deep|` may be
   dominated by tactics the net cannot see without searching — the head might
   just relearn "is there a hanging piece." Test on the holdout across phases; a
   head that only fires on material threats is weaker than one that captures
   positional uncertainty.
2. **Can a policy prior beat history heuristics enough to pay its cost?** History
   (doc 09) is already very strong and free. The policy must add ordering value
   *beyond* history, most plausibly in cold-history early positions — quantify
   the marginal cutoff improvement before wiring reductions to it.
3. **Control-correction stability.** Learned reductions can destabilize search
   (a wrong reduction hides a tactic; its cost surfaces far away — the
   credit-assignment problem, §9g). The fallback-safe Δ mitigates but does not
   eliminate this; needs the zugzwang/mate suites plus long SPRT.
4. **Joint-training interference.** Multiple heads sharing a trunk can fight over
   capacity (loss-weighting problem). Does the value head stay ≥ value-only? If
   heads degrade value, they must be separate trunks (losing the amortization) —
   a real risk to the whole cost argument.
5. **Per-node break-even on real hardware.** Every in-tree head's affordability
   is unknown until P13 inference nps is measured; §3.7's ranking is provisional
   until then.
6. **Uncertainty representation.** Magnitude scalar vs signed residual vs a
   coarse distribution (quantiles) — richer is more useful for TM but costlier
   to train/store; decide when the head is first built.
7. **Search-control credit assignment.** A pruning/reduction decision's true
   cost/benefit shows up plies away and games later; the Counterfactual pair
   gives a *local* label, but the global effect is what matters. How well local
   counterfactual supervision predicts global Elo is the deepest open question
   of Thrust II.

## 10. Integral study inputs (D37, 2026-07-12)

Integral (`integral_reference/`, top-6 solo-dev engine) built its eval with
almost exactly our planned pipeline (`bullet` + iterative HCE→NNUE→self-play
retrain = D28/D34) and reached top-6 — a direct validation. The sharpened
lessons for our P13 (ideas only; Integral is all-rights-reserved, D27/D37):

- **Get the full loop working cheap-and-early.** datagen → train → embed → bench
  end-to-end at a *weak* first architecture, BEFORE polishing — the sharpened
  form of D35. Integral's expensive phase was the ~6-month *architecture* search;
  once the arch froze and the net embedded, retrains dropped to ~13 days each
  (Zekrom v2→v7 in 67 days). Lesson: don't over-engineer the first net's shape;
  make the pipeline solid, then iterate architectures behind it.
- **Pure self-bootstrap reaches top-tier.** Integral's first net distilled from
  its OWN HCE self-play; every net since retrained on the prior net's self-play,
  no foreign teacher (its own-HCE data beat lichess-HCE data in an A/B). Confirms
  D28's clean-provenance path and that the SF-teacher (D28a) + Reckless-net A/B
  (D35) are optional insurance, not load-bearing.
- **Architecture reference data point** (not a prescription): Integral's net is
  `(768×12 factorized → 1536)×2 → (16 → 32 → 1)×8` — horizontally mirrored, 12
  factorized king input buckets, L1=1536, an L2/L3 head, 8 output buckets. A
  concrete "what a top-6 net shape looks like" anchor for our `design/11` §3.4.1
  arch choice; our first net stays smaller/simpler (D35) and grows toward this.
- **The DATAGEN-flag footgun** (audit into acceptance criteria): Integral strips
  search-only score adjustments (rule-50 / optimism terms) from the *training
  labels* under a datagen build flag — our labels must not bake in search-only
  eval tweaks either. Fold into `design/11` §3.6 / `design/16` acceptance.
- **P13 is a perpetual research thread**, not a one-shot: Integral was still
  shipping new nets 19 months / 7 nets after the first — matches the D34 framing.
