# Learned search augmentation, comprehensive — the full space, mapped and ranked

Date: 2026-07-19. Type: commissioned comprehensive study (owner mandate: the
two prior studies were each fed one pre-chosen formulation and stopped there;
this one maps the ENTIRE space, goes deep on the cells they excluded, and
invents the ones nobody named). Author model: **Fable (claude-fable-5)** —
stated per the owner's explicit request.

Method: read the three predecessor docs
(`learned-search-augmentation-first-principles-2026-07-19.md` = "study 1",
`d41-auxiliary-task-adversarial-2026-07-19.md` = "study 2",
`d41-auxiliary-task-reframe-2026-07-19.md`), the live state
(`value-foundation-state.md`), the three profiling rounds, design/19/17/12,
and the actual code (`policy.rs`, `movepick.rs`, `search/mod.rs`,
`timeman.rs`, `sota_head.rs`, `history.rs`, `search_control.rs`,
`search_trace.rs`); audit every measured constant against the ledger
(through exp-0103); then rebuild the design space from scratch instead of
inheriting either prior study's framing. Measured numbers are labeled
measured; reasoning is labeled reasoned. No code was modified; no games were
run (the SF18 relabel owns the CPU; this is analysis by design).

**Executive summary in five sentences.** Study 1 correctly killed per-node
fresh-forward inference (k > T fails by ~100×) and study 2 correctly
licensed the train-time auxiliary (T = 1); but between those two poles lies
a lattice of cost classes that neither study priced, and the cheapest of
them — **extra output columns on the eval forward the engine already pays
for** — has T ≈ 1.000 by construction, self-generated labels, shipped
precedent in Stockfish and KataGo, and multiple crude incumbents to beat
(fixed margins, telemetry-only time management). The second-cheapest —
**the incremental policy accumulator, re-derived here with an at-use
nonlinearity that study 1's version lacked and priced against the round-2
profile at T ≈ 1.08–1.15** — is the only vehicle that could ever ship a
per-node learned prior, and its go/no-gate remains the unrun K1 decay
curve. The distillation cell (root-frozen prior) is refined here into a
sharper mechanism — **inject the prior as the cold-start initializer of the
history stack, decaying as the tables warm** — which aims the signal at
exactly the regime where gen-0's +68.6 was earned. Non-ordering uses split
cleanly: eval-uncertainty gating is live and cheap (it is the flagship
output-column consumer), learned time management is mostly a retuning
problem because the incumbent is a full SF-current transplant, learned
reduction is measured-dead in neural form and alive only as formula terms,
and signed eval-correction collapses into either better eval training or
correction history. The ranked bets, argued in §5: (1) the multi-output
head tail (uncertainty + WDL + best-move-gap columns) feeding TM, margins,
and aspiration; (2) K1→K2′ (decay curve, then the cold-history distilled
prior); (3) EXP-AUX-1 extended to a three-arm co-training twin with a
keep-the-head sidecar so a positive feeds bet 2 for free.

---

## 0. What the prior studies settled, and the exact holes this study fills

Settled by measurement, not revisited except to re-price:

- **S-1. Per-node fresh forward inference is dead.** T = 130–210 as
  deployed (gen-0), T = 7.4 for the cheapest true shared-trunk head, gain
  non-retainable under every structural gate, required k ≈ 1.4–1.6 at 2k
  nodes and falling with depth (study 1 §2.2–2.3; PAS:870–1011). Nothing
  below reopens this.
- **S-2. The per-(node,move) reduction-risk latent is empty and the causal
  VOC target is unlearnable.** 16k exact roots: context+action AUC
  0.809/AP 0.423; the value latent adds −0.003 AUC (CI −0.006..+0.000);
  signed global node-delta has negative R² for every feature set; 40–43% of
  deeper reductions cost MORE root nodes (PAS:2139–2162, 1784–2004).
- **S-3. The allocation landscape near the incumbent is a plateau.**
  +50% LMR detune = −8 ± 13 (exp-0078); the D45 SPSA season shipped 0/8.
- **S-4. The train-time auxiliary (head dropped) is licensed and specified**
  (study 2's EXP-AUX-1; λ dose-response measured at PAS:388–427).
- **S-5. The local Elo/NPS slope is steep**: +28.8 ± 10.3 ET Elo for
  +13–15% NPS (exp-0101, bit-identical binaries) ≈ 2.1 Elo per 1% NPS ≈
  ~150 (±50) Elo per node-doubling at 10+0.1 — every 1% of throughput an
  idea consumes costs ~2 Elo before it helps, and every 1% of effective-node
  gain pays ~2 Elo.

The holes — what neither study analyzed, in the owner's order:

1. A **used-at-inference** search head in any cheap form (study 1 called the
   incremental form a "long shot" and spent one paragraph; study 2 excluded
   used heads by construction).
2. **Non-ordering outputs**: TM/budget, reduction/extension modulation,
   eval-uncertainty, eval-correction — each has a different (f, c_i,
   incumbent, headroom) tuple and none was priced.
3. **Dual-objective architectures where both heads ship.**
4. **Paradigm hybrids** on this hardware (RTX 4060Ti idle at play time + 16
   CPU threads), costed rather than waved off.
5. **Distill-then-compile** analyzed past the one-line K2 sketch: what
   actually survives compilation, and the mechanism that maximizes survival.
6. Formulations nobody named.

---

## 1. The completed cost lattice — the piece both studies were missing

Study 1's classes were A (constants), B (tables), C (cheap head on the live
accumulator), D (fresh forward). That lattice has a hole at the cheap end
and no entry for off-CPU or train-time forms. The complete lattice, priced
against the round-2 profile (sota opt1+2: 6.1k cycles/node, eval = 71% of
cycles, 0.34 evals/node, quiet-scoring confined to the ~10%-of-cycles
MovePicker; profiling-study-2026-07-19 §2):

| class | form | marginal cost at deployment | T (all relevant sites) | who missed it |
|---|---|---|---|---|
| **T1** | train-time only (aux loss, loss re-weighting); artifact byte-identical in shape | zero | **1.000 exactly** | study 2 owns this |
| **E0** | **extra output columns on the eval forward already being paid** (e.g., a variance column off L3, a WDL triple, a best-move-gap scalar) | ~30–100 FLOPs per eval on top of a ~4,300-cycle eval (71% × 6.1k); reasoned ≤ 0.1–0.3% NPS | **≈ 1.001–1.003** | **both studies** |
| A | compiled constants / new formula terms (fishtest praxis) | zero | 1.000 | mapped (R3) |
| B | table lookups: offline-distilled tables, or a per-`go`/per-iteration materialized table | one array read per consultation; a root forward is ~1 ms against a 100–600 ms move | 1.002–1.01 | mapped (R1/K2) |
| **C-inc** | **incrementally-maintained policy/aux accumulator lanes** (linear-in-features pre-activations, nonlinearity applied at use) | maintenance rides the existing fold kernels; priced in §3.1.2 | **≈ 1.08–1.15** (sharpened from study 1's 1.1–1.3) | study 1 named it, never priced or designed it |
| C-head | fresh head over the live accumulator per consultation | measured: rank-32 factorized = 6.4 node-equivalents | **7.4 measured** all-node | closed (S-1) |
| D | fresh trunk refresh + head per consultation | measured 150–200 node-equivalents | **130–210 measured** | closed (S-1) |
| **G** | **GPU-async**: inference off the search's critical path (4060Ti idle during play) | zero CPU cycles; cost = staleness + integration + result-application (class B on arrival) | ≈ 1.00 CPU-side | both studies ("different engine" wave-off) |

Two structural consequences the completed lattice exposes:

- **The inequality k > T is nearly moot at the cheap end.** For E0/T1/A/B/G
  the denominator is ~1; the binding constraints become **label
  availability, incumbent strength, and consumer headroom** — questions
  about signal, not cost. The entire program to date spent its games in the
  C-head/D rows and its analysis on the cost question; the cheap rows'
  question was never asked in game form.
- **Consultation-site frequency is as important as class.** The measured
  anatomy says quiet-move scoring happens at a minority of nodes (qsearch —
  the majority of nodes — never scores quiets; interior TT/capture cutoffs
  skip the quiet stage), eval happens 0.34×/node, root decisions ~1 per
  2·10⁵ nodes, and time-management decisions ~1 per iteration. A form's
  true tax is class-cost × its site's frequency; §2's map prices each cell
  at its actual site.

---

## 2. The full map: decision site × signal × cheapest viable class

Sites extend study 1's D1–D9 with three it omitted. For each: the cheapest
class that could carry a learned signal there, the incumbent to beat, and
the verdict. Deep analysis for every live or contested cell follows in §3.

| # | decision site | incumbent (audited in code) | cheapest viable class | verdict |
|---|---|---|---|---|
| D1 | total time for this move | **full SF-current TM transplant** — fitted opt/max formulas + 4-factor soft stop (fallingEval, stability reduction, instability, effort; `timeman.rs`, `search/mod.rs:4319–4394`) | A (refit constants) / E0 (add position-content difficulty input) | **LIVE, modest** — §3.2.1 |
| D2 | root budget split / easy-move / per-root-move effort | effort tracked (`rootmoves`), consumed only by the soft-stop effort factor; no nodesEffort-style per-move shaping | B (root table) + A | LIVE, small — §3.2.1 |
| D3 | move ordering at a node | TT move → MVV/SEE captures → 7-plane history sum + threat/offense terms (`movepick.rs:590–630`) | B (distilled cold-start prior, §3.1.3) or C-inc (§3.1.2) | **LIVE, the two flagship ordering vehicles** |
| D4 | reduction amount (LMR) | log-table + ~15 context adjustments incl. statScore, corrections (`search/mod.rs:3014–3032`) | A only (context terms; S-2 killed the neural forms) | fleet-gated formula work — §3.2.2 |
| D5 | extensions (singular, check; SE verification search) | margin formulas; SE runs a real exclusion search | E0 (uncertainty column modulates margins) / A | LIVE as an E0 consumer, small — §3.2.2 |
| D6 | prune outright (RFP, futility, LMP, razoring, NMP, SEE margins) | depth-indexed constant margins + improving flag + corrhist-adjusted eval | **E0 (uncertainty-scaled margins)** | **LIVE — flagship consumer** — §3.2.3 |
| D7 | aspiration window / re-search schedule | fixed width formula + widening schedule | E0 + A | LIVE, small — §3.2.3 |
| D8 | qsearch entry / stand-pat / delta margins | margins | E0 (only at fresh-eval nodes) | LIVE, rides D6 |
| D9 | what to remember (TT replacement, history update sizes, gravity) | aging/gravity formulas | A/B | open, ceiling a few Elo; fold into fleet SPSA — tombstone-adjacent |
| D10 | **eval-compute routing** (which net to run per node — the smallnet gate, profiling-r3 §4.2) | none yet; SF ships a material-threshold router | A (threshold) — a *learned* router is over-engineering a 1-bit decision | occupied by the speed program; learned form tombstoned |
| D11 | **thread allocation (lazy SMP: what helpers search)** | lazy-SMP redundancy (P11) | E0-fed scheduler (helpers verify flagged-risky subtrees) | genuinely novel, post-fleet/post-SMP-era — §3.4.3 |
| D12 | **training-data allocation** (which rows/weights the value net trains on) | uniform-ish binpack sampling + filters | T1 (search-impact-weighted loss) | invented cell, cheap probe — §3.6.1 |

Signals (what a head could output), with their label status:

| signal | label source | learnability status |
|---|---|---|
| move preference (best-move CE / pairwise) | `entry.mv` in every binpack row (study 2's find); gen-0 corpus | **proven** (24.8% top-1 at 100M rows; 0.258 off the frozen 720sb trunk) |
| value uncertainty / |deep − static| | **self-generated**: the net's own residual vs the training score; heteroscedastic NLL needs zero new data | proven-in-class elsewhere (KataGo score-stddev; searchless-chess distributional); unprobed here |
| WDL triple / draw-sharpness | binpack outcome + score (trainer already has WDL blending, `train_run.py` NET0_WDL) | trivial |
| best-move margin / policy sharpness | from the policy head's own logits (gap top1−top2) | free by-product of any policy aux |
| reduction bound-error | counterfactual oracle | learnable but **saturated by 39 context scalars** (S-2) |
| VOC / signed node-delta | exact oracle | **unlearnable** (negative R², S-2) |
| subtree cost (log-nodes) | in-search capture / oracle | context R² ≈ 0.17–0.30 local; global delta unlearnable; weak |
| TM stop/extend correctness | telemetry replay ("did more time change the move/result") | unprobed; labels cheap from obs platform |

---

## 3. Deep cells

### 3.1 The used-at-inference multi-head, made cheap (owner gap 1)

#### 3.1.1 E0 — output columns on the forward you already pay for (the class both studies missed)

**Mechanism.** The D40 head ends in `propagate_l3`: 32 f32 FMAs → one
scalar, per-bucket (`sota_head.rs:180–205`). Any additional per-position
scalar the search wants is one more 32-weight column against the same L3
activations (or L2 tap): value-variance, WDL triple, policy-sharpness,
"depth-needed" — each ~32–40 FLOPs on an eval that costs ~4,300 cycles.
The signal arrives **wherever eval is computed, for free** — no gating
question, no consultation-rate question, no anticorrelation problem,
because the carrier (the eval forward) is already paid at exactly the
nodes that matter.

**Cost model.** f = 0.34/node (eval rate), c_i ≈ 10–30 cycles ⇒
T ≈ 1.001–1.003 (reasoned from instruction counts; verifiable to
bit-identical-nodes standard by benching a net with dummy columns). One
real engineering wrinkle, flagged honestly: TT-cached evals skip the
forward, so the columns exist only at fresh-eval visits. The 10-byte TT
entry (32-byte 3-entry clusters) has no spare byte; storing a quantized
uncertainty would cost cluster geometry. Resolution: consumers that need
per-node coverage (futility margins) use the column only when the eval was
computed this visit and fall back to the depth-constant otherwise —
futility/RFP consume `ss.static_eval` computed at node entry, so coverage
is high where it matters; consumers at root/PV (TM, aspiration) always
have a fresh forward available.

**Why this is the flagship cell.** Compare its economics to everything the
program has tested: gen-0 paid 150–200 node-equivalents per consultation
for its signal; an output column pays ~0.005 node-equivalents. The k > T
inequality that killed three campaigns simply does not bind here. The
binding questions are (a) is the signal learnable (yes-in-class: KataGo's
score-stddev head trains routinely; heteroscedastic regression on our own
residuals needs no new data), and (b) does any consumer have headroom.
Consumer audit against the code:

1. **Pruning margins (D6).** RFP/futility/razoring margins are
   depth-linear constants modulated by `improving` and corrhist. A
   position-conditioned error bar lets margins be tight where the eval is
   reliable and loose where it is not — precisely the "when to trust the
   eval" formulation the owner asked for. Incumbent strength: the margins
   are SF-fitted to SF's eval-error statistics, never to D40's; the
   corrhist stack corrects the *mean* error online but carries no
   *spread* estimate.
2. **Time management (D1).** The soft stop's four factors are pure
   telemetry (score trajectory, best-move stability, effort) — **zero
   position content**. A root uncertainty/sharpness input is complementary
   information, not a re-weighting of what TM already sees.
3. **Aspiration width (D7).** Currently a fixed function of depth/prev
   score. Predicted root-value spread is the natural width prior.
4. **Singular-extension margins (D5).** SE margins gate a real exclusion
   search; uncertainty-scaled margins spend verification where the eval is
   shaky.
5. **NMP eval-margin and qsearch stand-pat cushion (D6/D8).** Same shape.
6. **WDL/drawishness column**: contempt-style steering and TM shortening
   in dead-drawn positions; KataGo/Lc0 precedent.

**Reference precedent (reasoned from public knowledge, not copied code):**
SF derives complexity/optimism-style scalars from its eval internals and
feeds them into eval blending and search margins; it also ships a dual-net
router — the strongest engine already consumes eval-side-band signals at
E0 cost. KataGo's score-stddev aux head is used at search time for search
control. Lc0's WDL head steers endgame play. The class is shipped practice
at the frontier; ARCHON's version differs only in which trunk carries it.

**Adversarial case, at full strength.** (a) Corrhist overlap: S1 landed
+72.9 internal precisely by correcting eval online; the *mean* channel is
taken, and if error spread correlates strongly with |correction|, the
marginal is small. (b) The margins are already implicitly tuned to average
eval error — position-conditioning the spread may be worth little on the
plateau (S-3 says detuning costs little, which cuts both ways: it also says
margin *improvements* buy little). (c) The variance a shallow trunk can
express may reduce to "phase/material/tension," which `improving`, depth,
and the material buckets already proxy. (d) Each consumer needs 1–2
coefficients tuned — at one-box resolution (±13) a +5 per-consumer effect
is invisible; this cell may be real but unprovable locally (the honest
counter: wire 2–3 consumers at once and gate the *package*).

**Open vs settled.** Entirely open — no experiment in the ledger touches
it. Nothing in S-2 bears on it: the counterfactual campaign probed
per-(node,move) reduction risk against search-context features; per-position
value-error spread was never a target, and the probe direction (frozen
latent → target) says nothing about a co-trained column (study 2 §5's
argument applies verbatim).

**Decisive experiment — EXP-UNC-1 (specified for pre-registration).**
Train the 240sb twin (control exists, ledgered) with the value head
emitting (mean, log-variance) under heteroscedastic NLL (Adam-safe; label =
the row's own score; optional second aux: WDL triple at the trainer's
existing blending machinery). Deploy with variance wired to exactly two
consumers behind env-gated coefficients: (i) futility/RFP margin scale
`margin *= clamp(1 + c1·(σ̂ − σ̄)/σ̄, 0.7, 1.5)`; (ii) soft-stop
`total_time *= (1 + c2·normalized-root-σ̂)`. Gate 1: value parity of the
twin (the NLL mean must not lose to the MSE control at fixed nodes —
kill if ≤ −15). Gate 2: consumers on vs off, same net, 600g fixed-node +
300g 10+0.1. Expected effect (reasoned prior): **+3..+15 internal at the
package level; P(≥ +10) ≈ 25–35%; P(value-parity holds) ≈ 70%**. Cost:
~1 day trainer work (shared with EXP-AUX-1's plumbing), one 240sb train,
~1 box-day of games. Kill conditions precommitted: value regression at
Gate 1, or no positive consumer configuration in a 2×2 coefficient probe.

#### 3.1.2 C-inc — the incremental policy accumulator, actually designed and priced (study 1's §4.4, taken seriously)

**Mechanism, precisely.** The gen-0 head computes from/to embeddings by
projecting the *post-activation* trunk (768-d pairwise-multiplied FT,
`policy_trunk`, `sota_head.rs:94–98`) — that projection is why every
consultation costs a fresh forward. Move the projection **before** the
nonlinearity and it becomes linear in the accumulator, hence linear in
features, hence maintainable by the same add/sub column deltas as the
accumulator itself:

```
F_pre[p] = W_f · acc[p]   (rank r × 64 squares = 64r lanes, i16, per perspective)
T_pre[p] = W_t · acc[p]   (64r lanes)
```

Because `acc = Σ_features A[:, i]`, precompute `W_f·A` and `W_t·A` as
per-feature policy rows folded by the existing `i16_deltas_into` kernels.
At consultation (quiet scoring in MovePicker only):

```
score(m) = Σ_k act(F_pre[stm][from·r + k]) · act(T_pre[stm][to·r + k])
```

with `act` = clipped-ReLU or SCReLU **applied at use time, per lane** —
the exact NNUE trick (accumulate linearly, activate at read). This is the
design improvement over study 1's sketch, which conceded "linear-in-
features embeddings, no nonlinearity" and therefore weak expressiveness:
at-use lane activation restores one full nonlinear stage. The resulting
head class sits strictly between "linear prior" and gen-0's
two-nonlinearity head; gen-0's gain will still not transfer 1:1 (it saw
products of *activated* sums; this sees activated sums only — one stage,
not two), but the family is the same one NNUE evaluation itself proved
sufficient for a much harder function.

**Cost model, priced against the round-2 profile (reasoned from measured
shares).** Maintenance: HalfKA folds cost 8.9% of cycles for 768 i16 lanes
per perspective per feature. Policy adds 2×64r lanes per perspective. At
r = 8: 1,024 extra lanes = ×2.33 the HalfKA fold volume ⇒ **+11.8% of
cycles ⇒ ≈ −10.6% NPS**. At r = 4: ≈ −5.9%. Tap **HalfKA features only** —
extending to the threat channel multiplies the 17.1% threat-fold block by
the same ×2.33 (+23% cycles) and is excluded. Finny-refresh and snapshot
costs scale proportionally (small). Consultation: an r-wide dot + two lane
activations per scored quiet ≈ a history-plane read; quiet scoring lives
inside the 10%-of-cycles MovePicker and qsearch never pays it ⇒ ≤ +1–2%.
**Total T ≈ 1.08 (r=4) – 1.15 (r=8), all-node, no gating** — the
configuration that actually held the +68.6, at last affordable. Required
k ≈ 1.08–1.15. At the measured ~150 Elo/doubling local slope, k/T = 1.05
pays ≈ +10 ET Elo; k/T = 0.95 costs the same. Knife-edge, exactly as
study 1 said — but now with a concrete design whose tax is at the *bottom*
of study 1's 1.1–1.3 band and whose expressiveness is higher than the
version that band was quoted for.

**What is genuinely open.** (a) K1 — does the per-node ordering signal
survive at depth? The +68.6 was measured at 2k nodes; nobody has run
8k/32k/128k. If it decays below k ≈ 1.15 by 32k, this cell dies with it.
(b) Does the one-nonlinearity family retain enough of gen-0's gain? Cheap
offline probe: train the pre-activation head on the existing gen-0 corpus
(the 100M-row corpus and trainer both exist, `d40_policy.rs`) and compare
holdout top-1/rank-corr against gen-0's 24.8%. If the pre-activation head
lands within ~80% of it, the family carries the cargo. (c) Trainer/export/
quantization engineering: 2–4 weeks, the most delicate in the program
(make/unmake plumbing through `sota_accumulator.rs`, netpack rev, parity
gates) — priced, not hand-waved.

**Adversarial case.** The k-ceiling argument (study 1 §1.3): the warm
stack at 2·10⁵ nodes/move has 20× more online training signal than at 2k;
first-move-cutoff rates in mature engines sit ~90%+; the residual ordering
headroom at real depth may be k ≤ 1.05 — under this design's own tax even
at r=4. Additionally the maintenance tax is paid at EVERY node including
the ~46%-of-pushes-never-folded (lazy eval) — no, correction: policy lanes
ride the same lazy fold schedule as the eval accumulator (fold-on-demand at
quiet-scoring time), so laziness transfers; but quiet-scoring happens at
nodes whose eval may never be folded, forcing policy-only folds — the true
maintenance rate sits between the eval-fold rate (0.45/node) and the
quiet-scoring rate. The T band above brackets this; a bench with dummy
lanes settles it in an hour.

**Sequencing.** Do not build on spec (study 1 was right). The gate chain:
**K1 (decay curve) → offline pre-activation-head retention probe → 1-hour
dummy-lane NPS bench → build.** All three gates cost < 3 box-days combined
and each can kill the cell before the expensive step.

#### 3.1.3 B — distill-then-compile, refined: the prior as the history stack's cold-start (K2 → K2′)

**Mechanism.** Study 1's K2: one forward per root, materialize the 64×64
from/to logit table, add `w·table[from][to]` to quiet scores everywhere —
pay position-conditioning once, accept staleness. The refinement (K2′),
motivated by *where the evidence says the signal lives*: gen-0's +68.6 was
earned at 2k nodes — cold histories, ~6 ID iterations; the root-only
ordering tests were null (+17±54, +7 null) because at the root the stack
is warmest; the gating ladder showed the gain lives at shallow/cold nodes.
So don't add the prior uniformly — **inject it where the stack is cold and
retire it as the stack warms**:

- blend weight per plane: `w_eff = w0 / (1 + n_updates(plane)/h)` — the
  prior dominates a zero-information history and vanishes against a warm
  one (h ≈ a few hundred updates, tunable);
- optionally seed continuation-history planes at allocation with per-piece
  destination tables (192 entries distilled from the same head), which is
  cheaper than per-move blending and touches the strongest ordering plane;
- re-materialize the table at each ID iteration (~1 ms each, ≤ 12/move) so
  staleness is bounded by one iteration's PV drift, not the whole search.

Cost: class B — one lookup + one shift per scored quiet, table refresh
amortized to ~nothing. T ≈ 1.002–1.01 (bench-verifiable; must show < 1%).

**What survives compilation — the owner's gap 5 question, made
quantitative.** The distillation loses (a) position-conditioning below the
root (staleness) and (b) nothing else — the head's full resolution at the
root survives. So the open question is exactly: how fast do the head's
logits decorrelate along search paths? This is measurable offline TODAY
with existing tools (`policy-eval` CLI, `uci.rs:102`): sample ~200 roots,
walk each PV/sampled path to ply 1..12, run the head at each node, compute
rank-correlation of the child's logits (restricted to its legal quiets)
against the root table. Call this **E-K2a (the staleness curve)**; ~1 quiet
box-hour, zero training. Prediction (reasoned): rank-corr ≈ 0.8+ at ply 2,
decaying to ~0.4–0.6 by ply 8 for quiet openings — if it holds ≥ 0.5 over
the plies where most interior quiet-scoring happens, the distilled prior
retains most of its ordering value; if it collapses by ply 4, K2′ needs
the per-iteration refresh at minimum and the cell shrinks.

**Adversarial case.** (a) The signal may be inherently position-local —
gen-0's Elo might come precisely from the nodes where the local position
differs most from the root (tactical sub-lines), which the table misses by
construction; E-K2a measures exactly this. (b) Double-counting: the history
stack *learns during the search* much of what the prior knows; injecting it
as cold-start may only accelerate convergence the stack achieves by
iteration 3 anyway — worth little at 2·10⁵ nodes/move (the honest reason
to expect fixed-node retention to shrink at equal time). (c) The blend
adds a tunable (w0, h) surface at one-box resolution — same provability
problem as every small effect.

**Decisive chain: E-K2a (offline, 1 hour) → K2 as specified in study 1
(fixed-node + ET, ~1 box-day) → K2′ cold-start variant (30 lines in
`movepick.rs`/`history.rs`, same harness).** Expected: study 1's 25–40%
of retaining > +10 fixed-node upgrades to ~35–50% conditional on a healthy
E-K2a curve; equal-time expectation +0..+15.

#### 3.1.4 Consult-where-valuable — why gating is a dominated strategy (upgraded tombstone)

The owner's 1(c). Two independent kills, one old, one new:

- Measured: every structural gate (root-only, max-ply, min-depth 4–8)
  destroyed the gain before it rescued the cost (S-1), because
  affordability and marginal value are anticorrelated across the tree.
- Structural (new): gating is only *needed* when the head is expensive; an
  ambiguity-triggered gate (consult when top-2 history scores are within ε
  — never tried, and better-aimed than depth gates because ambiguous+cold
  is where the prior helps) still pays the full per-invocation cost at
  ~10–30% of quiet nodes ⇒ T ≈ 1 + 0.2·0.3·150 ≈ 10 for class-D heads:
  dead on arrival. If instead the head is cheap (E0/C-inc), no gate is
  needed. **There is no (head-cost, gate) pair where gating converts a
  dead cell to a live one** — the gate variable is dominated everywhere.
  The one surviving gated form is "consult only at f ≈ once-per-move
  sites" — which is just the root/TM cells by another name.

### 3.2 Non-ordering uses of a search head (owner gap 2)

#### 3.2.1 Learned time/node budgeting — incumbent audit corrects study 1's premise

Study 1 rated learned TM its highest-probability bet (40–50%) partly on
"TM heuristics are the least fishtest-polished part of most engines; our
design/12 implementation is basic." **Code audit: that premise is wrong
for ARCHON.** `timeman.rs` is a complete SF-current transplant — fitted
opt/max scale curves, per-game adjust, and the full four-factor soft stop
(fallingEval, stability time_reduction, cross-thread instability, best-move
effort), constants pinned by golden tests. The incumbent here is *SF's own
fitted TM*, i.e., one of the most fishtest-polished artifacts in existence
— fitted to SF's tree, not ours, which is the genuinely open residue.

Re-derived decomposition of the cell:

- **(a) Refit the existing factor constants to ARCHON's tree** — cost
  class A, fleet SPSA work, no new inputs. This is where most of the
  literature's "TM patches keep landing" Elo actually lives. Expected
  +3–10 internal; not differentiated research.
- **(b) Add missing telemetry mechanics** (nodesEffort-style per-root-move
  effort shaping, easy-move fast stops): porting work already queued in
  the search program; class A.
- **(c) The learned residue: position-content difficulty.** The current
  factors see only *telemetry*. A root-level content signal — E0
  uncertainty/sharpness columns, or a G-class GPU forecast — is the only
  input class the incumbent cannot already express. This is EXP-UNC-1
  consumer (ii), not a separate program.
- **(d) A fully-learned stop rule** (offline classifier on "did further
  search change the move/result", replay-evaluated, then distilled to a
  formula): the clean research version; labels are cheap (obs platform
  captures the telemetry; 20k games of logs suffice); but its headroom is
  bounded by (a)+(c) — reasoned expected marginal over a refit incumbent:
  +2–8. Run it only if (c) shows the content signal is real.

Verdict: LIVE but demoted from study 1's ranking — it is mostly a
*retuning* cell wearing a research costume; the novel part is exactly the
E0 column input. Probability the full package (a–c) is worth ≥ +10
internal: ~40%; the learned-specific part alone: ~20%.

#### 3.2.2 Learned reduction/extension modulation — measured-dead in neural form; the surviving slivers

Settled (S-2, S-3): no neural per-(node,move) reduction head — latent adds
nothing over 39 context scalars for bound-error; the causal cost target is
unlearnable; the landscape is a plateau. Two slivers survive, both
non-neural or E0:

- **Context-interaction formula terms** (study 1's R3): take the 2–3
  strongest interactions from the 16k-root campaign into `params.rs` LMR
  terms, let fleet SPSA set coefficients. Class A; bounded by the plateau;
  +5–20 internal at fleet scale; not differentiated.
- **SE/verification economics** (new): singular extension runs a real
  exclusion search per candidate — a genuine per-node compute purchase
  with a measurable outcome (did it fail low?). A predictor could skip
  predicted-obvious verifications or trigger ones the margin misses. Cost:
  E0/A inputs only (TT margin, depth, uncertainty column) at f ≈ SE-candidacy
  rate (rare, deep). Adversarial: SF-lineage margins already encode exactly
  this trade and were tuned over ~10⁸ games; the exclusion search's cost is
  modest (reduced depth); expected ceiling +0–8. Status: open, cheap to
  probe offline from the obs platform's logged SE decisions — run only as
  a by-product of a search-telemetry logging pass, never as its own
  program.

#### 3.2.3 Eval-uncertainty / when-to-trust-the-eval — the strongest non-ordering formulation

Fully analyzed as §3.1.1 (it is the flagship E0 output). One addition on
the "gates search effort" framing the owner used: uncertainty can gate in
two directions — **spend more search where the eval is unreliable**
(extensions/margins/TM) and **trust the eval more where it is reliable**
(prune harder, stand pat earlier). The second direction is where the Elo
per node lives (pruning = the tree shrinks); the first is where safety
lives. A single column feeds both through opposite-signed coefficients;
the SPSA'd signs themselves are diagnostic of where the headroom is.

#### 3.2.4 Eval-correction in tactical positions — the collapse argument

Formulation: a head predicting the *signed* residual (deep-search value −
static eval), consulted to correct the eval where the scalar is wrong.
Adversarial decomposition kills it cleanly:

- The part of the residual predictable from the **position** is, by
  definition, an improvement to the eval function itself — adding it to
  the output IS training a better eval on the same labels. It cannot beat
  "just train the eval better" (same loss, same data), and the eval is
  trained to 3–9B presentations. Rebuttal considered: quantization/capacity
  limits mean a small correction head COULD carry what the main head
  cannot — but then it is just more eval capacity in a strange shape, and
  width is measured-saturated (S-3 analog: 2048 plateau; 768-trunk D40
  wins per node).
- The part predictable from **search context** (window, TT state, node
  type) is corrhist's exact domain — online, position-keyed, already
  shipped at +72.9 internal (exp-0074). A trained offline twin of corrhist
  would fight an incumbent that adapts to the current game online.
- The part predictable from **neither** is noise.

**Tombstone: collapses into (better eval training) ⊕ (corrhist), both
occupied.** The *unsigned* residual (spread) does NOT collapse — that is
the uncertainty column. This is the precise boundary between the dead and
live halves of "when the eval is wrong."

### 3.3 True dual-objective multi-head where both ship (owner gap 3)

The AlphaZero/KataGo shape adapted honestly to a 768-d one-layer trunk:

**Architecture.** Shared FT/threat trunk → (value tail, unchanged) +
(uncertainty/WDL columns, E0) + (policy head: rank-8 bilinear at train
time). Deployment: value + columns ship in the CENN; the policy head ships
**only in compiled form** — K2′ table materialization (B) or, post-K1,
C-inc lanes. Nothing per-node-expensive ever ships; "both heads ship"
means both *signals* reach the search, each at its own cheapest class.

**Does co-training help eval, search output, both, or neither — and at
what capacity?** The evidence, assembled:

- Help to eval: the λ = 0.01 rank-8 twin is the only right-shaped datum
  (+63 ± ~65, starved regime — suggestive, not probative; study 2 §6.2's
  inverted-U). Expected at 3–9B presentations: +5..+20, P(≥+10) ≈ 35–45%
  (study 2's numbers stand; nothing here revises them).
- Help to the policy head: co-training with the value loss regularizes the
  policy against its own small-data noise (v1.1's failure mode was
  confident noise at 336k rows); with `entry.mv` the policy target rides
  the same 3–9B stream, so the data-starvation that killed v1.1 is gone.
  Reasoned: the co-trained head should meet or beat the frozen-trunk
  0.258 top-1 at far better calibration — which directly raises K2′'s
  distillation quality. **This coupling is the point of both-ship: the
  aux twin (bet for eval Elo) and the distillation source (bet for
  ordering Elo) are the same training run.**
- Capacity: the 768-d bottleneck is tight (the value loss at 3–9B
  presentations constrains it hard; width saturated at 2048-classic /
  768-D40) — the honest reason to cap expectations below every published
  deep-net positive. The trunk can carry both (0.258 decodes off the
  frozen value trunk), but the co-benefit channel is one linear layer
  wide.
- The λ = 1.0 catastrophe and the 4096-head's monotone negativity bound
  the interference risk and pin the recipe: small head, λ ≈ 0.01,
  gradient-ratio logging (design/19 §4.2 discipline).

**Design consequence — extend EXP-AUX-1 to three arms, one trainer
change:** A = value-only control (exists); B = +policy aux (study 2's
spec, `entry.mv` CE, rank-8, λ = 0.01); C = +policy aux +
uncertainty/WDL columns (EXP-UNC-1's tail). Pack B/C with the aux head
bytes in a **sidecar file** (CPOL already exists as the container format)
so any Gate-1 positive immediately feeds E-K2a/K2′ with a
*better-calibrated* head than gen-0 — the distillation bet inherits the
co-training bet's artifact for free. Marginal cost of arm C over B: ~0.

### 3.4 Paradigm hybrids on this hardware (owner gap 4) — costed, not waved off

The box: RTX 4060Ti (~22 TFLOPS fp16 dense, 8–16 GB) idle during play; 16
CPU threads (play/tests currently 1-thread; lazy SMP exists from P11).

#### 3.4.1 Full paradigm switch (PUCT/MCTS, GPU inference) — tombstone with numbers

What it would take: a policy+value net at Lc0 scale (months of training we
cannot do locally, or adopting Lc0's — a different project), a batched
PUCT searcher (~2–4 months solo), abandonment of the entire AB asset
(TT/histories/pruning stack — the −25.5±28/node-vs-Reckless search).
What it would buy: reasoned from public cross-lists, Lc0-class engines on
a 4060Ti at ~10+0.1 land at best around SF16-class and typically below
strong CPU AB on 8–16 threads at short TC; the paradigm pays on big GPUs
and long TC. On THIS box it is a lateral-to-negative move with total asset
write-off. **Dead: not by k > T, but by opportunity cost and hardware
regime.**

#### 3.4.2 GPU as an asynchronous root oracle — the one live hybrid

**Mechanism.** During the search of move N, the GPU — otherwise idle —
evaluates a batch built from the root: the root position and its top-k
children (k ≈ 20–40, one make_move each), through a model with **no CPU
budget constraint** (a 10–100M-param policy/value/uncertainty net; even a
distilled SF-teacher transformer). Results land between ID iterations
(never on the critical path; latency 5–50 ms is irrelevant at that
cadence) and are applied at class-B cost: (i) refresh the K2′ root prior
table with far higher quality than the CPU head could afford; (ii) a
root-difficulty/uncertainty forecast into the soft stop; (iii) per-root-
move budget priors (D2). Optionally, ponder-time and opponent-time
batches pre-warm the next move's table.

**Cost model.** CPU-side T ≈ 1.00 (one table swap + a few scalars per
iteration). The real costs are staleness (bounded: the advice is *about
the root*, refreshed every iteration), engineering (a GPU inference
sidecar process + IPC — substantial but bounded, ~2–3 weeks), and
**advice redundancy**: by iteration 10+ the search's own returns dominate
any static forecast of the same root — the oracle's window of usefulness
is early iterations (where the table seeds cold structures — convergent
with K2′'s mechanism) and TM (where the incumbent has zero content
signal at any iteration).

**Adversarial case.** Root-only ordering tested null twice (+17±54, +7
null) — the direct precedent says root-granularity ordering advice adds
little; the rebuttals are (a) those tests used weak/cheap heads, not an
unconstrained GPU model, and (b) the oracle's primary consumers here are
TM/budget/cold-start, not root ordering. Still, the null precedent caps
the prior. Also: every Elo must clear the same one-box measurement fog as
everything else.

**Verdict: OPEN, medium prize (+5–20 reasoned), medium-high effort. The
correct sequencing is parasitic: build it only if K2′ shows the table
mechanism pays with the cheap head — the GPU then upgrades table quality
with zero new deployment mechanics.** Decisive pre-experiment shared with
K2′ (E-K2a/K2); the GPU-specific increment is justified by K2′'s effect
size, not on spec.

#### 3.4.3 Lazy-SMP-aware allocation — the post-fleet novel cell

At Threads = 16, lazy SMP buys ~1.5–2× effective via redundant
diversified search. Helpers are half-idle capacity by design — the only
place in the system where compute is nearly free at the margin. A learned
scheduler could aim them: verify reductions flagged risky (uncertainty
column high, SE-adjacent), pre-search the TM-predicted next root move,
deepen PV alternatives with high policy mass. History: smart splitting
(YBWC et al.) lost to lazy at scale because coordination overhead beat
the gains — but the loss was measured with *heuristic* risk estimates;
learned flags are a different input class. Honest status: post-M9 (SMP
validation is itself unfinished — the +279 4-thread number is PSQT-era),
post-fleet, unpriceable today. Recorded as the genuinely novel cell it
is, with a one-line gate: revisit when SMP is re-validated in the NNUE
era and an E0 risk column exists.

### 3.5 Distillation depth — what survives compilation (owner gap 5)

Consolidated from §3.1.3 plus the general principle: compilation loses
position-conditioning below the granularity of the compiled form, nothing
else. The forms, in increasing retained conditioning per unit runtime
cost:

| compiled form | conditioning granularity | runtime cost | retained-signal question |
|---|---|---|---|
| formula terms / SPSA'd constants (A) | none (global) | 0 | settled: this is fishtest praxis; retains only population-average structure |
| static offline tables (B): from/to, piece×to, conthist-init priors | none (global), but move-structured | 1 lookup | open; the piece×to marginal of gen-0's head is measurable offline in minutes |
| per-root materialized table (K2/K2′) | root position | 1 lookup + 1 ms/root | **E-K2a measures exactly this** (the staleness curve) |
| per-iteration re-materialized | root, refreshed | ≤ 12 ms/move | bounded staleness; the practical K2′ form |
| per-node incremental lanes (C-inc) | full per-node | T ≈ 1.08–1.15 | retention of gen-0's gain through one-nonlinearity family; offline-probeable on the existing corpus |
| GPU-async table (G) | root, unconstrained model capacity | ~0 CPU | K2′'s question × a better teacher |

The through-line: **every rung's decisive measurement is offline and
cheap**, because the head, corpus, and `policy-eval` CLI all exist. The
program never ran any of them — it jumped from "expensive per-node" to
"give up" without walking this ladder. Walking it is days, not weeks.

### 3.6 Invented cells neither study named (owner gap 6)

#### 3.6.1 Search-impact-weighted value training (T1)

Weight each training row's value loss by an estimate of how much eval
error *at that kind of node* propagates to search outcomes (proxies:
in-search visit frequency of similar positions from the obs platform;
|score| proximity to typical windows; quietness). Precedent-adjacent: SF's
trainer already applies score-based skip rules — a crude static version.
T = 1; needs only a sampling-weight hook in the trainer. Adversarial: at
30+ effective epochs over 3–9B presentations the value loss is already
dominated by exactly the positions search visits (the binpacks ARE
search-visited distributions) — the reweighting may be a no-op; and every
"data quality" intervention we tried is 0-for-~15. Expected: +0–10,
P(≥+10) ≈ 10–15%. One-line verdict: a cheap rider on the next training
campaign, never its own program.

#### 3.6.2 Learned memory-policy (D9)

TT replacement/aging and history-gravity weights are formulas; a learned
version is class A/B with a few Elo of ceiling. Fold into fleet SPSA;
independent tombstone otherwise.

#### 3.6.3 Aspiration/re-search policy (D7)

Width and widening schedule from (depth, prev-score, stability, σ̂-column).
Class A + E0 input. Small (+2–8 reasoned); implemented as EXP-UNC-1's
third consumer if the first two show life.

#### 3.6.4 Learned smallnet router (D10)

The dual-net gate itself is queued speed work (profiling-r3 §4.2). A
*learned* router over a 1-bit route with a material-threshold incumbent
shipped by SF is over-engineering: the router must run before eval (cheap
features only), so its expressiveness is capped at ~the threshold it
would replace. Tombstone.

---

## 4. The ranked map

EV = (expected internal ET Elo | success) × P(success) × tractability
discount (one-box measurability, engineering weeks, gate dependencies).
"internal" Elo deflates ~2–3× externally (exp-0091 lesson) — uniformly, so
ranking is unaffected.

| rank | cell | class | E[Elo|+] | P(≥ +10 int.) | tractability | EV verdict |
|---|---|---|---|---|---|---|
| 1 | **E0 output columns + 2–3 consumers** (uncertainty→margins, →TM; WDL) — EXP-UNC-1 | E0 | +8..+20 | 25–35% | HIGH (twin train + ~100 LoC + games) | **RUN** |
| 2 | **K1 decay curve → E-K2a staleness curve → K2/K2′ cold-history prior** | B | +5..+20 | 25–40% (post-E-K2a: 35–50%) | HIGH (all artifacts exist; zero training) | **RUN — the measurement debt** |
| 3 | **EXP-AUX-1 three-arm** (value / +policy-aux / +policy+columns, sidecar head) | T1 (+feeds 2) | +5..+20 | 35–45% | HIGH (study 2's spec + one arm) | **RUN (GPU otherwise saturated)** |
| 4 | C-inc incremental policy accumulator (post-K1, post-retention-probe) | C-inc | +10..+30 | ~15% unconditional; ~40% if K1 holds k ≥ 1.15 @32k | MED-LOW (2–4 wk delicate) | GATED — the only big-prize per-node vehicle |
| 5 | GPU async root oracle (upgrade of K2′) | G | +5..+20 | ~20% | MED (sidecar + IPC) | GATED on K2′ positive |
| 6 | TM refit + nodesEffort + content input | A/E0 | +5..+15 | ~40% pkg | MED (mostly fleet) | fold into search/SPSA program + bet 1 |
| 7 | Context-interaction LMR terms (R3) | A | +5..+20 | ~30% | fleet-gated | queue behind D36 |
| 8 | SE-verification predictor | A/E0 | +0..+8 | ~15% | LOW-cost probe | by-product only |
| 9 | Search-impact-weighted loss | T1 | +0..+10 | 10–15% | trivial rider | rider only |
| 10 | SMP verification scheduler | E0-fed | unknown | unpriceable | post-M9/fleet | recorded, parked |
| — | learned TM stop-rule (standalone), memory-policy, aspiration-policy (standalone), smallnet router | A/B | small | — | — | absorbed into 1/6/7 or tombstoned |

Dead cells (tombstones, killing reason):

- **Per-node fresh forward (any size, any gate)** — measured T = 7.4–210
  vs k ≤ ~1.6-and-falling; gain non-retainable under every gate (S-1).
- **VOC-gated consultation** — the causal target has negative R² at every
  feature set, 16k roots (S-2); and gating is dominated even with a
  learnable gate (§3.1.4).
- **Neural reduction/extension head** — latent adds ≤ 0 over 39 context
  scalars, replicated at three scales (S-2).
- **Signed eval-correction head** — collapses into eval training ⊕
  corrhist, both occupied (§3.2.4).
- **Helm data-scale-up for the per-node prior** — the inequality is
  indifferent to the fixed-node number (study 1 §2.4); d41-gold-v2 stays
  cancelled.
- **Full MCTS/PUCT paradigm switch on this box** — hardware regime +
  total asset write-off (§3.4.1).
- **Learned capture ordering** — MVV/SEE+capthist is near-solved;
  in-search capture labels are 99.6% bounds; policy-capture screens never
  showed a region.
- **Learned smallnet router** — 1-bit decision, threshold incumbent,
  pre-eval feature cap (§3.6.4).
- **Low-rank threat factorization** (adjacent bet) — measured: rank-256
  holds 51% of energy; the table is full-rank (profiling-r3 §3).

---

## 5. The top three bets, argued

### Bet 1 — EXP-UNC-1: the multi-output head tail (highest EV; the class both prior studies missed)

The argument, compressed: **the entire negative record of this program
lives in cost classes C-head and D; the entire positive record of the
frontier's learned-signal consumption at inference (SF complexity/optimism
/dual-net routing, KataGo score-stddev, Lc0 WDL) lives in class E0 — and
ARCHON has never trained or wired a single E0 output.** The class has
T ≈ 1.000 by construction (the forward is already paid; a column is ~30
FLOPs), labels that are free by construction (the net's own residual under
NLL; WDL from the same rows), a trunk that demonstrably carries side-band
signal (0.258 top-1 policy decodes off the frozen value trunk), and four
consumer seams whose incumbents are constants or telemetry-only. None of
S-1/S-2/S-3 bears on it: S-1 is about paying for forwards (none added);
S-2 probed per-(node,move) reduction targets against search-context
features (this is a per-position value-spread target, never probed); S-3
says margin *levels* sit on a plateau — it says nothing about
position-conditioning the margins, which is a new axis, not a re-level.

The honest risk profile: this is a +3..+15-internal-class package in a
program whose one-box resolution is ±13 — the same provability trap as
everything sub-15. Mitigations built into the design: consumers are
wired as a package behind one gate (not per-consumer), the twin's value
parity is a hard pre-gate, and a null is cheap (the trainer plumbing is
shared with bet 3, so the marginal cost of the columns is near zero).
Worst case: one 240sb train + one box-day of games buys a clean negative
that closes the class. Best case: the first learned-at-inference signal
in the engine's history to survive an equal-time gate — at a cost class
the k > T literature says is the only one that generically survives.

### Bet 2 — K1 → E-K2a → K2/K2′: pay the measurement debt, then compile the prior into the cold stack

K1 (the 2k/8k/32k/128k decay curve of the one artifact that ever worked)
remains, as study 1 said, the single most information-dense unrun
experiment — it prices every per-node vehicle including bet 4, and it is
pure measurement (~1–2 box-days, no training). This study adds two things:
**E-K2a** (the staleness curve: rank-correlation of the gen-0 head's
logits along search paths vs the root table; ~1 box-hour with the existing
`policy-eval` CLI) which converts K2's 25–40% prior into a measured
go/no-go before any games; and **K2′** (inject the table as the history
stack's cold-start prior with warmth-decaying weight, plus conthist
seeding, plus per-iteration refresh) which aims the signal at the exact
regime — cold planes, early iterations — where the +68.6 was measured to
live and where the root-only nulls say the plain root blend cannot reach.
The chain is strictly ordered by cost: 1 hour → 1 day → 2 days, each
stage able to kill the next. A K2′ positive additionally unlocks bet 5
(GPU oracle) as a pure quality upgrade with zero new deployment mechanics.

### Bet 3 — EXP-AUX-1, three-arm, with the sidecar that makes both-ship real

Study 2's design stands unchanged (target `entry.mv` CE on the same 3–9B
stream, rank-8, λ = 0.01, precommitted kills). This study's amendments:
(a) add arm C (+uncertainty/WDL columns) so bets 1 and 3 share one
trainer change and one control; (b) pack every arm's aux head into a CPOL
sidecar so a positive is simultaneously the *distillation source* for
bet 2's K2′ — replacing gen-0's 2026-era head with a better-calibrated
one trained on the winning data at zero extra cost. This is the concrete
form of "multi-head where both genuinely ship": eval improvement rides
T1; the search output ships compiled (B), not consulted raw (C-head/D).
Expected effect and kill conditions: study 2 §7.3 verbatim.

**Sequencing across bets (box-aware):** bets 1+3 are one merged GPU
campaign (three arms, shared control, GPU otherwise measured-saturated);
bet 2's chain is CPU-light and fits quiet windows (E-K2a needs one quiet
hour; K1 needs the box between matches). None of this displaces the speed
tail, the label-value pilot verdict, or the fleet stand-up — which remain
the program's confirmed levers (path-to-1 study); this portfolio is the
*differentiated-research* track running behind them.

---

## 6. What this study adds over studies 1–2 (delta ledger, for the record)

1. **The E0 cost class** — output columns on the paid-for forward; T ≈ 1
   by construction; the flagship bet. Neither prior study priced it.
2. **The C-inc design, completed**: pre-activation projection + at-use
   lane nonlinearity (the NNUE trick applied to policy, restoring the
   nonlinear stage study 1's sketch conceded away), priced against the
   round-2 profile at T ≈ 1.08–1.15 with a threat-channel exclusion rule
   and a three-gate kill chain that costs < 3 box-days before any build.
3. **K2′** — the cold-history-initializer mechanism, aligning the
   distilled prior with the measured location of the gen-0 gain, plus
   **E-K2a**, the offline staleness curve that prices distillation before
   games.
4. **The incumbent audit correcting study 1's TM premise** (full
   SF-current transplant, not "basic"), splitting the TM cell into refit
   (A) vs content-input (E0) with honest small numbers.
5. **The collapse argument** cleanly separating dead signed-eval-correction
   from live unsigned-uncertainty.
6. **The gating-dominance argument** (no (head-cost, gate) pair converts a
   dead cell), upgrading S-1's empirical kill to a structural one.
7. **Costed hybrid verdicts**: the GPU-async root oracle as the one live
   hybrid (parasitic on K2′), the paradigm switch tombstoned with
   hardware-regime numbers, and the SMP verification scheduler recorded
   as a real but currently unpriceable novel cell.
8. **Three invented cells** (search-impact-weighted loss, SE-verification
   predictor, learned memory-policy) priced honestly as riders, not
   programs.
9. **The both-ship synthesis**: one three-arm training campaign that
   serves the eval bet, the uncertainty bet, and the distillation source
   simultaneously — the concrete architecture in which eval and search
   heads co-train AND both reach the engine, each at its own viable cost
   class.

The research-contribution framing of study 1 §5 survives intact and is
strengthened: the viability-frontier paper gains a measured E0 row (the
class that generically survives as unit cost → 0) and, if bet 1 or bet 2
lands, its positive headline — a learned allocator beating a mature
hand-tuned stack at equal wall-clock — at the cost class the framework
predicts is the only place it could happen.
