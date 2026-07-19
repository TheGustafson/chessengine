# H-AUX adversarial study: train-time auxiliary search targets on an eval-only net

Date: 2026-07-19. Type: commissioned adversarial evaluation of ONE hypothesis
(owner mandate: falsify hard, but truthfully; no foregone conclusion). Author
model: **Fable (claude-fable-5)** — stated per the owner's explicit request.
Method: read the prior study and its reframe, then audit the hypothesis against
the ledger, the training code, the design suite, and the multi-task literature.
Citations: `exp-NNNN` / net-ids = `ledger/experiments.jsonl` /
`ledger/nets.jsonl`; `PAS` = `docs/research/policy-automl-state.md`; `VFS` =
`docs/research/value-foundation-state.md`; `file:line` = source. Measured
numbers are labeled measured; reasoning is labeled reasoning. No code was run
(an SF16 equal-time match owns the box; this is analysis-only by design).

**Executive verdict in three sentences.** H-AUX survives falsification: nothing
in the record refutes it, the one experiment in our ledger that actually has
its shape (the λ=0.01 co-training pilot, deployed eval-only) came out
**sign-positive (+63 ± ~65, PAS:408-411)**, and the prior study's §2.5 — the
only evidence ever cited against the family — measurably does not bear on this
formulation (§5 below). It is not a free win: the honest expected effect at
our current data scale is **+5..+20 Elo, with ~35–45% probability of clearing
+10** — a real but modest lever whose main virtues are that T = 1 exactly, the
λ→0 limit bounds the downside, and (the load-bearing new fact of this study)
**the auxiliary target is already present in every training row we own**
(`entry.mv` in the official binpacks), so the decisive experiment needs zero
new data and ~1–2 box-days. Run EXP-AUX-1 (§7); precommit the kill conditions
(§6.4); do not book the Elo before the A/B.

---

## 1. The hypothesis, steelmanned properly

H-AUX: co-train the D40 eval trunk with an auxiliary head predicting a
search-derived move-preference target, weighted λ against the scalar value
loss; **drop the head at deployment** → the shipped net is architecturally
identical to a value-only net, T = 1 exactly, zero NPS tax. The bet: the aux
gradient shapes the trunk toward search-relevant structure that the value loss
alone does not find, yielding a strictly better eval at equal inference cost.

Steelman components, strongest first:

1. **The inference-cost inequality is genuinely irrelevant.** The prior
   study's spine (`k > T`, `T = 1 + f·c_i/c_n`) governs inference-time
   consultation. Here f = 0 at deployment. The reframe doc is simply correct
   on this point; no argument in the prior study's §1–§2 transfers. (One
   nuance for honesty: two D40 nets with different weights are not exactly
   NPS-identical — the NNZ-sparse L1 (`sota_head.rs:119-123`) makes node cost
   depend on *activation* sparsity, which is weight-dependent. Measured
   spread between sibling D40 nets has been ~0–2%, second-order; the
   equal-time gate catches it anyway.)

2. **The λ→0 limit contains the control.** A λ sweep is a superset of the
   value-only recipe. Unlike every inference-time formulation — where the
   mechanism itself costs NPS before it helps — the worst case here is
   bounded by training compute plus one bad point on a dose-response curve.
   This asymmetry alone justifies a cheaper evidence bar than the helm ever
   deserved.

3. **Our own design suite already made this bet and never tested it.**
   `design/17` §3.2 states the heads are "trained *jointly* … which also
   regularizes the trunk (multi-task learning tends to improve the value
   head)" and §5 pins the acceptance criterion "the joint-trained value head
   must SPRT ≥ a value-only net (multi-task should help, not hurt)". That
   criterion was written before the D40 campaign, was exercised only at
   λ = 1.0 on starved nets (where it failed for the known gradient-scale
   reason, PAS:398-400), and was never run at the data scale that produces
   winning nets. H-AUX is not a new program; it is an unexecuted acceptance
   test of the existing one.

4. **The signal exists and lives in this trunk family.** Gen-0 proved
   move-preference is learnable at 100M rows (24.8% disjoint-holdout top-1 /
   33.6% on the corrected-row sample, PAS:512, 599-601; +68.6 ± 23 fixed-node
   as an ordering prior, exp-0029), and d41-alloc-v11-c
   proved the *frozen value-trained* D40 trunk supports a policy head at
   0.706 rank-corr / 0.258 top-1 vs the 0.181 cold-classical bar
   (nets.jsonl:41). The two tasks are demonstrably correlated in
   representation space — the precondition for positive transfer.

5. **The deployment story matches how the program already promotes.** An
   eval-only candidate rides the existing fixed-node screen → equal-time
   gate pipeline unchanged (VFS promotion bar). No new runtime code, no
   bench-fingerprint risk, no D24 exposure.

---

## 2. Q1 — Eval headroom: is there anything left for a better representation to unlock?

**Verdict: real headroom exists; it is data-, recipe-, and arch-limited, not
information-ceiling-limited. The precondition for H-AUX is satisfied — but the
same evidence says most headroom is reachable by cheaper, proven levers, which
sets H-AUX's opportunity-cost bar.**

Evidence, all from our ledger:

- **Data was the binding constraint (B4, answered).** Every 25M–265M-row
  challenger lost to net-0; the official binpacks at 3–9B presentations won
  decisively (+265 composed at equal time; CLAUDE.md status, nets.jsonl:20-31).
  A net family that gains hundreds of Elo from data volume alone is nowhere
  near an information ceiling.
- **Feature-set headroom is real and measured:** D40 full-feature beats the
  best classic net at fixed nodes (+32.5 ± 17.3, exp-0065; +30.2 ± 17.4 vs the
  actual incumbent, exp-0067). Representation choices still move the needle.
- **But the current recipes are saturating on their own axes:** width ladder
  saturated at 2048 (+102/+57/+15 per doubling, nets.jsonl:33); D40 stage-1
  saturated at 720sb (−9.3 ± 15.4 for +360sb, exp-0100); stage-2 hops dead on
  D40 (exp-0066). **The marginal Elo of more value-only compute under current
  recipes is measured ≈ 0.** This is important for H-AUX's opportunity cost:
  the GPU-hours EXP-AUX-1 costs are hours that currently buy nothing on the
  saturated axes.
- **External calibration:** ARCHON is −78 ± 18 vs SF16 and −191 ± 23 vs SF18
  at 10+0.1. SF's own scalar NNUE line has kept improving for five years by
  data + arch alone. A ~3400-anchored eval is not near the ceiling of what a
  scalar value net extracts; SF's is several rungs above ours on the same
  deployment economics.

Is *search-aware structure* part of the remaining headroom? Partly, by this
argument (reasoning): the stage-2 gains came from deeper/better labels
(farseer +55 on classic) — i.e., pushing the static output toward deep-search
values from the *target* side. H-AUX aims at the same asymptote from the
*representation* side: a leaf eval that internally encodes "a strong quiet
continuation exists here" is the same commodity as initiative/threat quality,
which is exactly where FullThreats already pays. The two mechanisms plausibly
overlap, which deflates H-AUX's marginal effect below what either would do
alone — an honest reason to expect the low end of the range.

---

## 3. Q2 — What multi-task theory and the precedents actually predict here

The literature, read without cherry-picking, gives H-AUX a conditional yes
with known sign-control variables:

**For (in-domain, large-data, strong-main-task):**

- **AlphaGo Zero's dual network** (Silver et al. 2017): merging policy and
  value into one trunk *slightly reduced move-prediction accuracy but reduced
  value error* and gained ~600 Elo over separate nets, explicitly attributed
  to "the dual objective regularises the network to a common representation."
  This is the canonical in-domain datum that policy co-training improves the
  VALUE function in the large-data regime — not merely that a shared trunk is
  cheaper. (The +600 confounds compute-sharing; the value-MSE reduction does
  not.)
- **KataGo** (Wu 2019, "Accelerating Self-Play Learning in Go"): auxiliary
  targets (ownership, score distribution, opponent-reply policy) produced
  large training-efficiency gains and final-strength improvement at equal
  compute, on a strong, data-rich main task. The aux heads' *deployment* role
  is minor-to-none; the benefit was in the shaped representation. Closest
  published precedent for "search/game-structure aux target improves the
  value net."
- **DeepMind searchless chess** (Ruoss et al. 2024): per-move action-value
  prediction from position alone is learnable at scale and dominates
  state-value + argmax for playing strength — direct evidence that per-move
  targets carry structure a state-value net does not spontaneously acquire,
  in chess, from search-derived (SF oracle) labels.
- **Our own record** (§6.2 below): the λ dose-response pilot is sign-positive
  at the right λ, and the frozen-trunk decodability result (0.258 top-1)
  proves the tasks share representation.

**Against (the honest weights on the other pan):**

- **The regime effect is real and adverse.** The MTL literature (Standley et
  al. 2020; Fifty et al. 2021; gradient-surgery line) consistently finds
  auxiliary benefit shrinking as main-task data grows, with sign flips under
  task interference. Our value task moved from starved (25M rows) to rich
  (3–9B presentations) precisely last week. The AGZ/KataGo positives are on
  *outcome-noise-dominated* value targets (game result z), where aux
  denoising is most valuable; our value target is a low-noise deep-search
  score. Expect the top of the literature's effect range to be unavailable.
- **The shallow-trunk objection (the strongest theoretical headwind).** AGZ,
  KataGo, and the searchless-chess nets are deep residual/transformer stacks
  with abundant shared computation; "representation shaping" has many layers
  to act on. A D40 net is one linear feature transform + pairwise activation
  feeding a tiny head. Shaping "the representation" here means re-weighting
  first-layer feature columns through a 768-d bottleneck that the value loss
  at 3–9B presentations already constrains tightly. The channel exists (the
  bottleneck is where the policy head reads too, and gen-0 proves the
  bottleneck *can* carry both) but it is narrower than in every published
  positive. This is the main reason my expected effect is +5..+20 and not
  AGZ-shaped.
- **Interference is measured, not hypothetical, in our own ledger** — λ = 1.0
  destroyed eval (300× gradient dominance, PAS:398-400); the 4096-logit full
  head was value-negative at every λ tried (PAS:412, 421-423). Sign depends
  on head architecture and λ; the sweep must respect that.

Do chess-eval and search-policy help or compete? Our data says: **correlated
enough to share a trunk (0.258 top-1 from a frozen value trunk), coupled
enough to destroy each other at λ = 1, and separable enough that a rank-8
head at λ = 0.01 left value intact-or-better.** That is exactly the profile of
an auxiliary task worth dose-tuning, per the theory.

---

## 4. Q3 — Target choice is the crux

The prior study's negative evidence is all about *unlearnable or already-cheap
targets*; the reframe is right that the learnable one was never plugged into
co-training. Adversarial ranking of the candidates:

| target | learnable? (evidence) | available at value-data scale? | expected sign as aux |
|---|---|---|---|
| **Best-move CE (search's played move)** | yes — gen-0 24.8% disjoint-holdout top-1; v11-c 0.258 off a frozen trunk | **YES — `entry.mv` is in every sfbinpack row we already train on** (`d40_attn.rs:209-218` uses it for filtering today); stage-1 = SF-search move @5k nodes, farseer = Lc0-derived play | positive-or-null at tuned λ; the ONLY target with full-scale row-aligned availability |
| Per-move teacher value / margin | yes, moderately — d41-alloc-v1 rank-corr 0.604 (nets.jsonl:40) | no — gold-v2-teachers ≈ 180k positions × 2 teachers (VFS:109-114); 4 orders of magnitude short; bounded-label quarantine applies to the *values* | positive in theory (richest per-row signal; searchless-chess evidence) but data-starved → confident-noise risk, the exact exp-0092 failure mode |
| Lc0 policy distribution (soft CE) | yes — the gen-0 corpus | 100M rows, corrected builder exists; but a *different corpus* than the value rows → confounds the A/B and mixes label provenance | positive plausible; use later, not for the decisive test |
| VOC / signed node-delta | **no — negative R² at every feature set, 16k roots** (prior study §2.5) | n/a | **negative at any material λ**: an unlearnable target's gradients are noise injected into the trunk |
| Reduction-risk / bound-error | learnable but already cheap-scalar-saturated (AUC 0.809 from 39 context scalars; latent adds ≤ 0) | n/a | ~zero: the signal the trunk could add is measured ≈ 0 |

Three conclusions:

1. **Yes, the sign of H-AUX depends on the target.** Aux benefit requires a
   learnable target correlated with the main task. VOC fails learnability;
   reduction-risk fails incrementality; move-preference passes both. The
   prior study tested exactly the two failing rows and none of the passing
   ones — the reframe's central charge is confirmed.
2. **Best-move CE is the decisive-experiment target**, not because it is the
   richest, but because it is the only one available *row-for-row on the same
   3–9B-presentation stream that trains the value net* — no new data, no
   corpus confound, no quarantine exposure (the *move choice* is an argmax,
   not a bounded scalar; the bound quarantine binds values, not choices).
   This fact was missed by both prior docs (the reframe still says "may need
   more/better data"). It collapses H-AUX's cost from "build a per-move
   corpus" to "read a field we already parse."
3. **Which target best shapes the eval toward positions where scalar value is
   currently wrong?** CE has a useful built-in property: its loss (and hence
   its gradient pressure on the trunk) concentrates where the move
   distribution is hard to predict — sharp, tactical, search-sensitive
   positions — which is where eval-vs-search disagreement lives. Per-move
   value regression would target this even more directly but is data-barred
   (above). An explicit disagreement-weighted aux (upweight rows with large
   |label − net output|) is a v2 ablation, not v1.

Note the current stage-1 filter keeps only quiet, non-check rows
(`d40_attn.rs:209-218`), so the aux becomes "predict the search's chosen
*quiet* move" — precisely the structure (initiative, threats, piece activity)
that plausibly overlaps value headroom, and the same regime where gen-0's
ordering gain lived.

---

## 5. Q4 — Does the prior study's §2.5 bear on H-AUX? Rigorous adjudication

**Claim under test:** "the 16k-root counterfactual campaign showed the value
latent adds nothing → co-training can't add anything either."

Three independent logical gaps, then the honest steelman:

1. **Probing ≠ co-training.** §2.5 asked whether a *frozen, value-trained*
   latent linearly (and nonlinearly, via adapters) improves prediction of
   search targets. H-AUX asks whether *gradient pressure from a search
   target during training* improves the latent for VALUE. These are
   different directions of inference: the probe measures what value training
   *happened to encode*; H-AUX changes what training encodes. A null probe
   is fully consistent with a positive co-training effect (the aux adds
   structure the value loss never sought) — and equally consistent with a
   null one. The probe simply does not discriminate.
2. **Wrong targets.** §2.5's targets were VOC (unlearnable, negative R² from
   *any* features — including the raw board, which provably contains all
   position information; the target itself is noise-dominated by re-search
   cascades) and scout-bound-error (saturated by search-context scalars —
   window, TT state — that no *position* representation could carry). H-AUX's
   target is move preference, for which the same trunk family measurably
   carries signal (0.258 top-1 frozen; 24.8% disjoint-holdout co-trained).
   Generalizing from
   the two failing targets to the passing one is the exact error class the
   owner flagged.
3. **Wrong artifact scale.** The §2.5 probes rode net-0's latent and
   336k-row/16k-root supervision — pre-B4 artifacts. The B4 resolution showed
   this program's negatives at small data do not transfer to large data.

**The steelman (the strongest pessimistic reading that survives):** §2.5 does
establish one thing relevant here — *value training alone does not
spontaneously encode search-allocation structure* (else the probe would have
found it). Two live interpretations:

- *Pessimistic:* value-optimal and search-optimal representations are nearly
  orthogonal in this trunk → co-training buys pure capacity competition, and
  the best λ is ≈ 0 (null result).
- *Optimistic:* value training leaves search structure unlearned, so the aux
  contributes genuinely new features; if any of them also reduce value error
  (threat/initiative substructure), the effect is positive.

The orthogonality reading is directly contradicted for the H-AUX target by
d41-alloc-v11-c: the frozen value trunk decodes move preference at 0.258
top-1 vs 0.181 cold baseline — the representations overlap substantially. So
the surviving content of §2.5 against H-AUX is narrow: **it bounds the hope
that co-training would *additionally* make the eval trunk a good free input
for future inference-time allocation heads (VOC/reduction).** No claim about
the value-improvement channel survives. Resolution: **§2.5 is inadmissible
against H-AUX's core bet; it correctly kills only the dual-use-trunk side
hope, which H-AUX does not need.**

---

## 6. Q5 — Falsification: the strongest case against, and what the ledger already says

### 6.1 The strongest case that H-AUX fails

1. **Regime shift.** The aux literature's benefits concentrate where the main
   task is data-starved or label-noisy. Ours no longer is: 3–9B
   presentations of low-noise search scores. The marginal information of
   ~4–5 bits of move identity per row may be fully redundant once the value
   loss has seen the *child positions themselves* elsewhere in the corpus
   (they occur as later plies of the same games). In the limit, best-move CE
   on the parent adds only a cross-move consistency constraint — and maybe
   the value loss at 30+ effective epochs already enforces it implicitly.
2. **Shallow trunk.** One linear layer + pairwise activation is the least
   favorable known architecture for representation-shaping arguments (§3).
   All published positives are deep nets.
3. **Interference is real here.** λ = 1 destroyed eval; the full head was
   negative at every λ. The useful window may be narrow, λ-fragile, and
   data-scale-dependent — a multi-train tuning cost for a small prize.
4. **Too-small-to-gate.** If the true effect is +5, it is real, free Elo —
   and unprovable at reasonable game budgets (±17 at 300g @5k). The program
   could burn box-days confirming noise. (Counter: +5 undetectable also
   means −5 undetectable; the promotion bar simply won't promote a null, and
   the option cost was one training run.)
5. **Opportunity cost.** The confirmed levers right now are speed (+28.8 ±
   10.3 equal-time from opt#1 alone, exp-0101) and search (S1 corrhist
   +72.9 ± 16.2, exp-0074). — Counter-argument from the same ledger: those
   are CPU-side engineering; the GPU axis is *saturated* (exp-0100: more
   stage-1 compute = −9.3; stage-2 dead on D40, exp-0066). EXP-AUX-1 spends
   otherwise-idle GPU capacity on the only unexplored training-side lever
   with a positive prior. The opportunity cost argument, checked against
   what the GPU can currently buy, is weak.

### 6.2 What the ledger ALREADY says — the evidence the prior study never surfaced

The mandate's question "did any net trained with a policy head have
better/worse VALUE than its value-only sibling?" has an answer, and it is the
single most H-AUX-relevant table in the record (all: matched arch/data/seed
twins, CPOL disabled = eval-only deployment; PAS:388-427):

| λ (policy weight) | head | eval-only vs value-only twin | reading |
|---|---|---|---|
| 1.0 | rank-8 bilinear | catastrophic (eval beat it 35-4-1) | policy CE ~3 vs eval MSE ~0.01 → ~300× gradient dominance |
| 1.0 | full 4096 | catastrophic (40-0) | worse still |
| **0.01** | **rank-8 bilinear** | **+63 (100g; ±~65)** — "eval survived and may have improved" | **the H-AUX-shaped positive** |
| 0.01 | full 4096 | −494 | head gradient scale decides sign; λ is not architecture-neutral |
| 1e-4 | rank-8 bilinear | 0 exactly (44-44-8) | dose → 0, effect → 0; consistent with a real dose-response |
| 1e-4 | full 4096 | −49 ± 59 | ~null |

This is a coherent inverted-U in λ for the small head and monotone-negative
for the big head — mechanistically exactly what auxiliary-task theory
predicts, in our own data. Honest caveats, stated at full strength: 100
games (the +63 is ~1σ); the twins were grossly undertrained (24.576M
examples, −285..−512 vs net-0 as a family) — i.e., measured in the starved
regime where aux benefits are inflated; the corpus was the pre-correction LC0
dual corpus. **This evidence is suggestive, not probative. It sets the prior;
it cannot substitute for the A/B at winning data scale.** The prior study
cited none of it.

Also on record, for completeness of the against-case: every *deployment*-side
policy result (exp-0086 −9.3, exp-0092 −29.6, root-only nulls) is about
inference-time use and carries zero weight here, in either direction.

### 6.3 What evidence would kill H-AUX

- **Primary kill:** EXP-AUX-1's λ sweep {3e-3, 1e-2, 3e-2} at 240sb, each
  pooled to 600g @5k vs the existing value-only control, produces no point
  ≥ +10 and no coherent positive dose-response region → H-AUX is dead at
  this arch/data scale. (One λ ≤ 0 is not a kill; the pilot says the window
  is narrow.)
- **Secondary kill (ship-relevance):** a 240sb positive that evaporates when
  re-run on the 720sb incumbent recipe → starved-regime artifact; H-AUX
  survives as a science result about aux-vs-data-scale but dies as an
  ARCHON lever.
- **Non-kill to guard against:** a single-seed +15 at 240sb is NOT a
  confirmation — D40 cross-seed variance is unmeasured. Seed-2 replication
  is mandatory before any promotion claim (§7).

### 6.4 Adversarial bottom line

The falsification case reduces to "the benefit probably shrank below our
noise floor when the data got big, and the trunk is too shallow for the
mechanism." That is a *quantitative* objection — a probability statement, not
a structural impossibility like k > T. Structural kills (unlearnable target,
inference tax, measured absence of signal) do not apply to this formulation.
When the strongest available objection is "the effect might be small," and
the test costs 1–2 box-days of otherwise-saturated GPU with the control
already trained and ledgered, the correct move under this program's own
methodology is to run the test, with precommitted kill conditions and honest
priors — not to book the win, and not to close the door by analogy to a
different formulation's death.

---

## 7. Q6 — The surviving design and the decisive experiment (EXP-AUX-1)

### 7.1 Design

- **Trunk/arch:** exact `d40-uho-v1-240sb` recipe (nets.jsonl:35; config
  `data/nnue/experiments/…` via `train_run.py`), unchanged. Same width, same
  features, same schedule, same seed, same stage-1 UHO stream, same filter.
- **Aux target:** best-move CE. Extend the sfbinpack path with a custom datum
  carrying the canonical stm-relative (from, to) of `entry.mv` — the field
  the filter already parses (`d40_attn.rs:209-218`). One-hot CE over the
  4096 from×to logits, label smoothing 0.05. No new data, no exporter, no
  quarantine surface (move choice is an argmax, not a bounded scalar).
- **Aux head:** rank-8 bilinear from/to over the shared 768-d post-activation
  trunk — the exact gen-0 shape (`policy.rs` CPOL kind-1 mirror;
  `sota_head.rs::policy_trunk` is the trainer-runtime contract). Chosen
  because its gradient scale has a measured dose-response (§6.2) and it is
  the proven-trainable shape. Trainer work = merge `d40_policy.rs`'s head +
  loss machinery into the `d40_features.rs`/`d40_attn.rs` sfbinpack
  ValueTrainer (the NPZ reader in `d40_policy.rs:193-330` shows the custom
  DataReader pattern). Estimated 1–2 days of trainer engineering; nothing in
  the engine changes.
- **λ:** first shot λ = 0.01 (the pilot's positive), with per-step
  policy/value gradient-norm ratio logged (design/19 §4.2 discipline). If
  the screen is ambiguous, sweep {3e-3, 3e-2}. Loss-scale note: CE(≈3) vs
  MSE(≈0.01) means λ = 0.01 ≈ 3:1 raw loss ratio; Adam normalizes head-local
  scale but not the shared-trunk mix — the logged ratio, not λ itself, is
  the quantity to report.
- **Aux layer tap:** the 768 trunk activation (option a). Tapping deeper (L2)
  shapes only the tiny tail; the headroom, parameters, and the pilot's
  evidence all live in the FT.
- **Deployment:** pack with the standard CENN-v2 packer — which has no
  policy slot, so the shipped artifact is eval-only *by construction* —
  then the standard runtime parity gate (0.03/0.20 bars).

### 7.2 The A/B and gates

- **A (control): already exists** — `d40-uho-v1-240sb`, parity-passed,
  ledgered. Zero cost.
- **B: one training run** of the twin + aux (240sb ≈ ⅓ of the 720sb
  overnight run — a few GPU-hours).
- **Gate 1 (screen):** B vs A, 300g @ 5k fixed nodes, policy-balanced book,
  the standard harness. Identical arch → fixed-node ≈ equal-time modulo the
  ~0–2% sparsity note (§1.1). Decision rule, precommitted:
  - point ≥ +15 → train B-seed2; if seed2 point > 0, pool and proceed to
    Gate 2;
  - point in (−15, +15) → run the two flanking λs before judging (the pilot
    says the window is narrow — one λ point is not a verdict);
  - point ≤ −15 → H-AUX badly weakened at this scale; write the negative to
    the ledger and stop unless the gradient-ratio log shows λ was mis-scaled.
- **Gate 2 (confirm + ship path):** pooled ≥ 900g fixed-node (resolves +10
  at ~95%), then apply the winning λ to the **720sb incumbent recipe** (the
  secondary-kill test, §6.3), then the standard 200g 10+0.1
  `prod_tc_gate.sh` — the promotion bar unchanged (VFS:24-27).
- **Ablations (only after a Gate-1 positive):** target swap (per-move-value
  regression mixed at low rate from gold-v2-teachers' 180k×2-teacher arms —
  the data exists, VFS:109-114, values subject to the exactness quarantine),
  λ dose-response completion, head-rank 4/16.

### 7.3 Cost, power, expected effect — stated before the result

- **Cost:** 1–2 days trainer work (one-time; also unblocks any future
  aux-target study) + ~4–6 GPU-hours per net + 300–900 games per arm.
  Runs entirely behind the current match queue; respects the 7GB-OOM rule
  (bullet training RSS is the known ~2.8GB profile).
- **Expected effect (honest priors, reasoning):** at 240sb (mid-data,
  pre-saturation — 720sb beats 240sb by +30, so this scale still has
  optimization slack for the aux to exploit): **+5..+20 fixed-node at the
  best λ; P(≥ +10) ≈ 35–45%; P(coherent positive λ-region) ≈ 50–55%;
  P(survives at 720sb ≥ +10) ≈ 20–30%.** Downside at λ = 0.01 bounded by
  the pilot's dose-response: worst plausible ≈ −20.
- **Signal vs noise:** 300g @5k → ±17–19. The Gate-1 thresholds are set so
  that no decision is made inside the noise band on one point; the pooled
  900g requirement before any promotion claim is what separates this
  protocol from the exp-0029→corrected-head replication failure (PAS:956-964).

### 7.4 Sequencing honesty

H-AUX is a lever-A (eval quality) experiment with a novel mechanism — it does
not touch the deprioritized helm line (lever D), does not contend with the
speed program for CPU, and does not need the D36 fleet. Per the north-star
ranking it queues behind nothing currently using the GPU. It should NOT
displace: the opt#2 equal-time match readout, the PGO rebuild, or the
label-value deep-relabel A/B — all of which are further along their own
evidence chains.

---

## 8. Verdict

1. **H-AUX survives adversarial review.** The only evidence ever cited
   against the family (§2.5) does not bear on it; the ledger's one
   experiment of the right shape is sign-positive; the in-domain precedents
   (AGZ dual-objective value-error reduction, KataGo aux targets) are
   positive in the large-data regime; and our own design suite (design/17
   §3.2/§5) predicted the effect and pinned an acceptance criterion that has
   never been executed at winning data scale.
2. **It is not a big lever.** The regime shift to 3–9B presentations and the
   shallow-trunk mechanism cap the honest expectation at +5..+20 Elo — a
   rung on the same ladder as speed and label quality, not a path-to-#1
   bet. Anyone selling it as more is overclaiming; anyone closing it by
   analogy to the per-node helm's death is underclaiming.
3. **The economics are unusually good for a test:** T = 1 by construction,
   the control is already trained, the target is already in every training
   row (`entry.mv` — the fact both prior docs missed), the GPU axis is
   otherwise saturated, and λ→0 bounds the downside. Run EXP-AUX-1 with the
   precommitted kill conditions in §6.3/§7.2, and let the dose-response
   curve — not another essay — settle it.
