# Learned search augmentation, from first principles — will it ever pay?

Date: 2026-07-19. Type: commissioned first-principles study (owner mandate:
"this is bigger than chess — if it isn't going to work, I want to know; if it
will, I want to know how"). Method: formalize the idea from the ground up, map
its full design space, derive the economic inequality that governs it, plug in
ARCHON's measured numbers, and try hard to falsify every formulation. This is
NOT a review of the D41 program's results; the results are used as measurements
of the idea's parameters. Citations: `exp-NNNN` = `ledger/experiments.jsonl`;
`PAS:L` = `docs/research/policy-automl-state.md` line; `file:line` = source.
Where a number is measured I say measured; where I am reasoning I say so.

**Executive verdict in three sentences.** Learned *per-node* search
augmentation — the formulation ARCHON has been testing — is economically dead
on CPU alpha-beta, not marginally but by two orders of magnitude, and our own
data is sufficient to prove it; the deep reason is structural (the
affordability of inference and the informativeness of the prior are
anticorrelated across the tree), not an engineering shortfall. However, the
*idea* — a learned allocator beating hand-tuned heuristics at equal compute —
is not dead; it survives in exactly the cells of the design space where
inference cost is amortized to ~zero (root-level decisions, distillation into
tables/constants, and one unexplored incremental-computation trick), which is
also precisely where alpha-beta engines *already* deploy learned allocators
without calling them that. The decisive next experiment costs about two days
and is specified in §4.1/§6.

---

## 1. Formalization and the design-space map

### 1.1 What a search allocator actually decides

An alpha-beta search under budget B (nodes or time) is a stream of allocation
decisions. Enumerated concretely, with where they live in ARCHON:

| # | decision | granularity | current controller |
|---|----------|-------------|--------------------|
| D1 | total time for this move | per root move (game level) | hand-tuned TM (`timeman`, design/12) |
| D2 | budget split across root moves (aspiration, easy-move stop, effort) | per root move | hand heuristics + RootMoves sorting |
| D3 | move *ordering* at a node | per (node, move) | TT move → captures (MVV/SEE) → killers-equivalents → history stack (butterfly/pawn/continuation/low-ply, `movepick.rs:590-630`) |
| D4 | reduction amount (LMR) | per (node, move) | log-formula base table + ~15 statScore/context adjustments (`params.rs`) |
| D5 | extension amount (singular, check, …) | per (node, move) | margin formulas |
| D6 | prune outright (futility, LMP, SEE, razoring, NMP) | per (node, move) | depth-indexed margin formulas |
| D7 | re-search / verification (PVS window, do_deeper) | per node | window logic |
| D8 | qsearch entry/stand-pat | per node | margins |
| D9 | what to remember (TT replacement, history update sizes) | per node | aging/gravity formulas |

Every one of these is a candidate site for learning. The question is never
"can a model predict something useful here" — it usually can — but "does the
predicted improvement, converted to Elo, exceed the Elo cost of computing the
prediction plus the opportunity cost of the nodes not searched."

### 1.2 The overlooked fact: alpha-beta already runs learned allocators

This reframing matters for everything downstream. The history stack IS a
learned allocator: it is trained online (gravity updates from observed
cutoffs), conditioned on move features (piece×to, continuation context), and
consulted at ~zero cost (a handful of array lookups per scored move,
`movepick.rs:590-596`). The SPSA/fishtest-tuned constants in D4–D8 are a
learned allocator: trained offline by black-box optimization over ~10^8 games,
then *compiled into the binary* at zero inference cost. S1 correction history
(+72.9 ± 16.2 internal, exp-0074) — the single biggest recent search win — is
another table-based online learner.

So the honest form of the D41 question is NOT "learned vs hand-tuned." It is:
**does a high-capacity, position-conditioned representation (an NN forward)
add allocation signal over the low-capacity learned allocators already
present (tables + tuned formulas), net of the inference cost that the
low-capacity forms do not pay?** Both the numerator (marginal signal *over a
warm table stack*, not over nothing) and the denominator (inference cost paid
per decision) are harder than the naive framing suggests. Every measurement
below should be read against this bar.

### 1.3 The central inequality, derived

Let:

- `c_n` = baseline cost of one node (ARCHON sota: ~6.1k cycles measured,
  profiling-study-2026-07-19 §4; net-0 classic era: ~1.5k cycles at 3.2M NPS);
- `c_i` = cost of one allocator inference, in node-equivalents (`c_i/c_n`);
- `f`   = allocator invocations per node (the consultation rate);
- `T`   = throughput tax = `1 + f · c_i/c_n` (NPS_off / NPS_on);
- `k`   = node-equivalent multiplier of the allocator's decisions: allocator
  at N nodes plays like baseline at kN (design/19 §8.2's own metric);
- `s(N)` = Elo per node-doubling at operating point N (steep when shallow,
  ~50–150 at 2k nodes; flatter deep, ~30–70 at 10^5+; ARCHON's measured
  *local* equal-time slope is ~2.1 Elo per +1% NPS at 10+0.1, exp-0101).

At equal wall-clock the allocator engine searches `N/T` nodes worth `k·N/T`
baseline-equivalent nodes. Therefore:

```
Equal-time gain (Elo) ≈ s(N) · log2(k / T)      →      wins iff  k > T
```

Three structural consequences, before any measurement:

1. **The amortization law.** `T − 1 = f · c_i/c_n`. You can pay a huge `c_i`
   if `f` is tiny (root decisions: f ≈ 1 per 2·10^5 nodes ⇒ even a 1ms
   inference is T ≈ 1.005), or a tiny `c_i` often (history lookups). What you
   cannot do is pay NN-forward-scale `c_i` at per-node `f`: with
   `c_i/c_n ≈ 5–200` (measured below) and `f ≈ 0.1–1`, T lands in 1.5–200,
   and no ordering improvement can deliver k of that size (§1.4).
2. **The k-ceiling for ordering.** Knuth-Moro: perfect ordering searches
   ~2·b^(d/2) vs b^d for none. But mature engines already sit near the
   perfect-ordering regime at the nodes that matter — TT move present at
   revisited nodes via iterative deepening, first-move-cutoff rates ~90%+
   (standard across engines; reasoning, not ARCHON-measured), and LMR pushes
   effective branching below the theoretical perfect-ordering bound. The
   *residual* k available from better ordering on top of a warm stack is
   small — plausibly 1.1–1.5 shallow (cold histories, few ID iterations) and
   shrinking toward 1.0–1.15 at real depth. Our own exp-0078 measures the
   flatness of the allocation landscape directly: detuning the base LMR
   reduction by +50% costs only −8 ± 13 Elo. A landscape that flat around the
   incumbent has little room for ANY reallocator, learned or not.
3. **The unit-cost law (the cross-domain key, §5).** Everything above scales
   with `c_i/c_n`. Alpha-beta chess has the cheapest search unit of any
   domain where learned allocation has been tried — a node is ~1–6k cycles,
   *microseconds*. In NN-MCTS the unit is a full net forward (ms-scale on
   GPU); in LLM inference-time compute the unit is an entire generation
   (seconds). Chess alpha-beta on CPU is therefore close to the *worst-case
   testbed* for this idea — a fact that cuts both ways (§5.3).

### 1.4 The design-space map with viability calls

Cells = decision site × cost class. Cost classes: **A** compiled constants
(ρ=0), **B** table lookup (ρ≈0, online- or offline-learned), **C** cheap head
riding an *already-maintained* representation (ρ small), **D** fresh NN
forward per invocation (ρ huge).

| cell | example | T (derived/measured) | required k | viability for CPU alpha-beta |
|------|---------|----------------------|-----------|------------------------------|
| D1–D2 × D (root) | learned time manager / root budget allocator; ONE forward per move | 1.001–1.01 | +1–2 Elo gross | **VIABLE on cost, always.** Question is pure headroom (§4.2) |
| D3 × B (distilled) | learned prior *compiled* to a from/to bias table at the root, consulted at all nodes like a history plane | ~1.002 | +1–2 Elo | **VIABLE on cost; the untested cell** (§4.1) |
| D4–D6 × A | fit LMR/futility formulas to a rich offline model's outputs, or SPSA at fleet scale | 1.000 | anything > 0 | **VIABLE; this is "smarter fishtest"** — bounded by the plateau (exp-0078), needs D36 fleet to verify |
| D3 × C (incremental) | pre-activation policy embeddings maintained like the eval accumulator; per-quiet cost ≈ a history lookup, maintenance ~10–30% NPS | 1.1–1.3 | k ≥ 1.1–1.3 | **KNIFE-EDGE** — the only per-node design whose arithmetic even closes (§4.4) |
| D3 × C (gated head) | factorized head on the live accumulator, consulted at 5–15% of nodes | 1.1–1.5 (derived from measured c_i) | k ≥ 1.1–1.5 | measured to fail: gating destroys the gain before it rescues the cost (§2.3) |
| D3 × D | fresh trunk refresh + bilinear head per node (gen-0, helm v1.x as implemented) | **7.4–200 measured** | k ≥ 7.4–200 | **DEAD, measured** (§2.2) |
| D4 × C/D | learned reduction-risk head per (node,move) | same as D3×C/D | same | DEAD in D form; C form redundant — context features carry the signal and those are what formulas already consume (§2.5) |
| VOC-gated anything | consult the model only where value-of-computation is high | needs VOC prediction | — | **DEAD at current predictability: the causal compute-consequence target has NEGATIVE R²** (§2.5) |
| D9 × B | learned TT replacement/history-update weights | ~1.0 | tiny | viable on cost, unexplored, ceiling probably a few Elo |

The map's summary: **every cell with ρ ≈ 0 is viable and largely occupied by
existing practice; every cell with ρ > node-cost is dead; the frontier is the
thin band where a learned signal can be made ~free** (distillation,
incremental maintenance, root amortization).

---

## 2. The inequality with ARCHON's numbers plugged in

### 2.1 The measured constants

- Baseline NPS (net-0 classic, the eval used in every policy game test):
  **3.2M** single-thread (PAS:L878). ARCHON sota today: 860k–959k
  (profiling r2 §4); cycles/node 6.1k; **eval = 71% of cycles**.
- Equal-time Elo/NPS slope: **+28.8 ± 10.3 Elo for +13–15% NPS** at 10+0.1
  (exp-0101, bit-identical binaries) ⇒ ~**2.1 Elo per 1% NPS** locally —
  about 2× the usual rule because the eval is expensive per node. Every 1% of
  NPS an allocator consumes costs ~2 Elo at equal time before it has helped.
- Gen-0 rank-8 bilinear invocation (the +68.6 artifact): a **full D40 trunk
  refresh** (`policy.rs:246-252` calls `refresh_both` — from scratch, not
  incremental) plus two 512×768 scalar f32 affines (`policy.rs:345-371`)
  ≈ 0.8–1M scalar ops ≈ **150–200 node-equivalents per invocation**.
  Measured consequence: all-node consultation ran at **~15–25k NPS vs 3.2M
  baseline, T ≈ 130–210** (PAS:L878, L977 "prior ~15K-NPS classic bilinear").
- Cheapest deployed variant: classic-factorized rank-32 head riding the
  *real, incrementally-maintained* net-0 accumulator (true shared trunk,
  `policy.rs:280-328`): **373k NPS vs 2.78M off ⇒ T = 7.4** at all-node
  consultation (PAS:L975-981). Rank-64: 207k NPS. So even a genuine
  head-only design costs ~6.4 node-equivalents per invocation — because a
  16k-MAC scalar head is *several nodes' worth of work* when a node is 1.5k
  cycles. Sharing the trunk is necessary but nowhere near sufficient.
- Depth-gated consultation (min-depth 4/5/6/8): NPS 47k/83k/138k/305k
  (PAS:L878-884) — still T ≈ 10–68, because "depth ≥ 4" is not rare enough
  and the invocation cost is monstrous.
- Doc 19's addendum estimated "~2% NPS at rank 8 on the shared trunk"
  (design/19 addendum, consultation economics). The implemented reality
  measured 7.4× at rank 32 all-node; even rank-8 factorized at 10%
  consultation derives to ~4–15% NPS, not 2%. The addendum's economics were
  optimistic by roughly an order of magnitude against every measured
  configuration. This audit matters: the shared-trunk amortization story
  (question 5d of the mandate) does NOT arithmetically deliver "near-free"
  on this engine — the *head*, not the trunk, is the binding cost, and the
  head cannot shrink much below ~1 node-equivalent while staying
  position-conditioned at f32.

### 2.2 The crux artifact dissected: +68.6 at fixed nodes vs the tax

exp-0029: gen-0 rank-8 bilinear from/to head, net-0 evaluator unchanged on
both sides, +68.6 ± 23.0 over 200 games at 2k fixed nodes, independently
+61.4 (40g, second seed). This is a real, replicated, correctly-attributed
per-node ordering gain — the strongest evidence anywhere in our record that
a learned prior CAN improve per-node decisions over the warm history stack.

Convert it. At 2k nodes/move the self-scaling slope is steep — 100–150
Elo/doubling is the generous range (reasoning from standard scaling data;
our own tiny-net budget screens swing consistently with this). So +68.6 Elo
≈ **k ≈ 1.4–1.6**, at most 2.0 under the most generous slope. Equal-time
viability requires T ≤ k, i.e. the entire learned apparatus must cost
**≤ 40–60% NPS**. Its measured cost was **T ≈ 130–210**. The gap is a factor
of ~100, and — this is the important part — it is not an implementation gap:

- Suppose a perfect engineer makes the invocation 30× cheaper (incremental
  trunk sharing + SIMD head): T ≈ 5–7. Still fails by 4×.
- Suppose additionally consultation is gated to 10% of nodes with ZERO loss
  of the gain: T ≈ 1.4–1.6. Now it *exactly breaks even* — assuming two
  miracles and zero gain decay at depth.
- And the gain does decay with depth. That is the k-ceiling argument (§1.3
  #2), and the record supports it: exp-0092's post-mortem is explicit that
  "the warm in-search picker sits far above" the cold-ordering bar the helm
  was clearing offline. At 2k nodes histories are cold and there have been
  ~6 ID iterations; at the real 10+0.1 operating point (~2·10^5 nodes/move)
  the free stack has 20× more training signal per game position. Nobody has
  measured the k(N) decay curve for exp-0029 — a critical omission (§3.4) —
  but every proxy says it shrinks.

**Conclusion of the arithmetic: the fixed-node/equal-time gap is not an NPS
"tax" to be engineered away; it is a 100× structural deficit against a
gain whose ceiling is ~1.5× and falling with depth.**

### 2.3 The gating experiments measured the deep structural problem

Every attempt to keep the gain while shedding the cost failed, and the
pattern is the finding:

- Root-only ordering (one forward per `go`, ~free): gen-0 head **+17 ± 54**
  (40g, inconclusive-null, PAS:L894-899); the stronger-offline native-label
  head **+7, CI −40..+53 over 200 games — null** (PAS:L1583-1584). No
  root-only configuration ever confirmed positive.
- max-ply ≤ 1 (root + first reply): **−44 ± 49** while still paying ~14×
  throughput (PAS:L900-902).
- min-depth gates 4–8: kept 10–68× cost, screens never recovered the gain
  (PAS:L884-886, exp-0029 notes: "POLICY_MIN_DEPTH/root-only gates all
  failed to retain the gain cheaply").
- All-node factorized rank-32 (T = 7.4): exactly 50% over 40 games at every
  weight tried; "no coherent positive region" (PAS:L1003-1011).

Why this pattern is structural, not bad luck: **the affordability of
inference and its marginal value are anticorrelated across the tree.** The
few nodes where you can afford a big inference (root, high-depth, PV) are
exactly the nodes where the free stack is already strongest — TT move from
the previous iteration, warm continuation history, IID — so the prior's
marginal signal is smallest there. The numerous shallow nodes where the free
stack is cold and the prior genuinely helps (where exp-0029's Elo actually
lived, as the gating results reveal) are exactly where per-invocation cost
can never be repaid, because each such node's subtree is tiny. A per-node
learned prior on CPU alpha-beta is therefore squeezed from both ends — this
is the single most important sentence in this study, and it was *measured*,
not conjectured: the gating ladder above is precisely an experimental scan
of gain-retention vs affordability, and it found no crossing point.

### 2.4 What −29.6 (helm v1.1) does and does not prove

exp-0092: the D41-native helm head (frozen 720sb trunk, pairwise + all-tier
training, clears every offline bar: 0.706 held-out rank-corr, 0.258 top-1 vs
0.181 cold-classical) scored **−29.6 ± 12.9 at 2k fixed nodes** — negative
before any speed accounting, worse than v1 (−9.3 ± 25.5, exp-0086), with the
POLICY_W sweep monotone worse at higher weights.

What it proves: (a) offline ranking above a *cold* classical baseline is
irrelevant — the deployed competitor is the warm picker, a moving,
position-specific target that the offline metric does not even see; (b) a
sharper head trained on insufficient data (336k rows vs gen-0's 100M)
injects *confident* noise — blending a below-picker prior into a
well-calibrated ordering actively subtracts Elo.

What it does NOT prove: that native consequence-trained priors can't clear
the picker at fixed nodes. Gen-0 proved the signal exists at 100M-row scale.
The v1.1 failure is a data-scale failure of one pipeline, honestly diagnosed
in the ledger row. If someone pays for 10–30× more rows, the fixed-node
number could plausibly go positive again — **but §2.2's arithmetic is
indifferent to this**: even a reproduced +68 fixed-node gain is ~100× short
of equal-time viability in the per-node formulation. This is why "more data
for the helm" is not the binding question and why further helm *training*
spend is correctly frozen.

### 2.5 The counterfactual campaign quietly falsified the program's two core premises

The D41 exact-intervention oracle work (PAS:L1784-2004, L2139-2162) is the
most scientifically valuable thing the program produced, and its results are
brutal for the "learned VOC allocator" thesis:

1. **Premise "the scalar evaluator discards information needed to allocate
   search" (design/19 §3): not confirmed — at scale, refuted for this
   trunk.** Across 16,000 exact deterministic interventions (49,878 action
   outcomes), predicting scout bound-error: cheap context+action features
   reach AUC 0.809/AP 0.423; adding the protected net-0 accumulator in ANY
   form — PCA-32 (Δ AUC −0.003, CI −0.006..+0.000), full 512-d activated
   linear probe, low-rank interaction, protected residual adapter, raw
   12×64 board — **adds nothing or hurts**, replicated at three scales
   (429 → 2,048 → 16,000 roots). The position latent that carries a
   3400-Elo evaluation does not carry incremental reduction-risk signal
   beyond 39 hand-picked context scalars. Those context scalars are exactly
   what the existing LMR formula and histories already consume.
2. **Premise "predict cost, then optimize it": the causal cost target is
   currently unlearnable.** The signed global node-delta of an intervention
   (the true value-of-computation quantity a planner needs) has **negative
   R² for every feature set tried**, and 40–43% of deeper reductions cost
   *more* total root nodes than the reference (re-search cascades dominate).
   VOC-gated inference (formulation (c) in the mandate) requires exactly
   this prediction; it is dead until someone demonstrates the target is
   predictable at all.

Positive residue: context features DO carry real risk signal (AUC lift
+0.055 over action-only, CI +0.041..+0.068). But a context-only model is
cost-class A/B — a better LMR formula — which is the fishtest cell, not the
neural cell. The campaign, read plainly, says: **the learnable part of
per-node allocation is the part that is already cheap, and the part that
needed the NN is not there (for this trunk/labels/scale).**

### 2.6 What the steep Elo/NPS slope and 70% eval share imply

Two opposing effects, worth stating precisely:

- The expensive eval RAISES `c_n`, which *lowers* the relative tax of any
  fixed-cost allocator (a 16k-MAC head is ~2.6 node-equivalents against
  net-0's 1.5k-cycle nodes but only ~1 against sota's 6.1k). Eval-heavy
  engines are marginally friendlier territory for learned allocation, and a
  hypothetical future 10×-bigger eval would be friendlier still.
- The same expensiveness makes each searched node more valuable — the
  measured 2.1 Elo/1% slope (exp-0101) — so foregone throughput is punished
  at double the normal rate.

Net: the two roughly cancel in the inequality (both scale with `c_n`), and
neither rescues a formulation that is 100× underwater. But the first effect
is why the incremental-accumulator design (§4.4) is knife-edge rather than
absurd on this engine, and why it would have been absurd on a PSQT engine.

---

## 3. Falsification: the strongest case against, and what survives it

### 3.1 The strongest case that this is a dead end for CPU alpha-beta

1. **The headroom theorem.** Ordering/reduction headroom over a warm mature
   stack is bounded small (k ≤ ~1.15–1.5 depending on depth; §1.3), and our
   flattest direct measurement of the allocation landscape (exp-0078:
   +50% LMR detune ⇒ −8 ± 13) says the incumbent sits on a high plateau.
   You cannot buy much k at ANY price, and per-node inference is the most
   expensive price on the menu.
2. **The cost floor.** A position-conditioned forward cannot go below a few
   hundred cycles without ceasing to be position-conditioned; a node is
   1.5–6k cycles. So `c_i/c_n ≥ ~0.1` even in fantasy hardware-friendly
   form, and realistically ≥ 1. With per-node f, T ≥ 1.1–2 forever. Combined
   with (1): the inequality generically fails in the per-node cell.
3. **Revealed preference of the frontier.** A decade of fishtest (~10^8
   games/yr of free validation, a community desperate for Elo), plus every
   top CPU engine (SF, Reckless, Berserk, Integral, Viridithas...),
   converged on: NN for eval, tables + tuned formulas for allocation. Zero
   adopted per-node learned inference for search control. Lc0 — the one
   engine family built around a learned allocator — had to abandon
   alpha-beta for PUCT, abandon CPU for GPU batching, and accept 1000×
   fewer nodes to make the economics work. The market has run this
   experiment at a scale we never will.
4. **Our own record.** Two positive-gross artifacts ever (exp-0029 +68.6
   fixed-node; the +32.5/node D40 *eval* which is a different lever), zero
   net-positive; every cost-reduction gate destroyed the gain
   (§2.3); the native-label pipeline's premises falsified at scale (§2.5);
   the honest v1.1 play test negative before tax (exp-0092).

### 3.2 Does the existing evidence already kill the per-node formulation?

**Yes, to the standard of "stop spending here."** The kill is the
conjunction: (measured T = 7.4 for the cheapest true-shared-trunk variant)
× (measured gain retention ≈ 0 under every gate that reduces T) × (measured
k-ceiling flatness, exp-0078) × (measured absence of latent signal beyond
context for the reduction target, 16k roots). Any one leg could be an
artifact; four independent legs pointing the same way, on an engine whose
eval/search/speed levers all measure normally, is a verdict. The only
missing measurement for a *formal* kill is the k(N) decay curve (§3.4),
which the program's own design doc demanded (design/19 §8.2, "the output is
an Elo-versus-node-budget curve, not one Elo number") and never ran for the
one artifact that ever worked. I recommend running it as the closing
experiment, not because the verdict is in doubt, but because it converts
"dead by arithmetic" into "dead by direct measurement" — which is what a
paper needs (§5).

### 3.3 What the evidence does NOT kill (the honest steelman)

1. **The signal exists.** exp-0029 is replicated: a learned prior beat the
   warm picker per-node at shallow depth. Any claim that "hand-tuned
   heuristics already extract all allocation signal" is FALSE at 2k nodes.
   The open question is only its magnitude at real depth and its price.
2. **The NNUE precedent — the strongest counterargument to §3.1.** For
   ~15 years the identical community consensus said "NN evaluation is too
   slow for alpha-beta on CPU" — the same inequality, same revealed
   preference, same market efficiency. It flipped in one move when the
   incremental-accumulator trick collapsed `c_i/c_n` from ~100 to ~2.
   Frontier consensus about cost floors is consensus about *known
   representations*, not about the idea. The per-node formulation is dead
   unless someone finds "the accumulator trick for policy" — and there is a
   concrete candidate shape (§4.4). Assigning this zero probability would
   repeat the exact error the NNUE skeptics made.
3. **The native-consequence pipeline never got its data.** v1.1 trained on
   336k rows against a target whose one success needed 100M. The premises
   in §2.5 were falsified *for the frozen value trunk at current label
   scale* — a dedicated control representation trained on 100× more causal
   data could still find latent signal context features miss. I judge this
   unlikely (the context features are suspiciously complete for the
   reduction target — they include the window, TT state, and ordering
   signals that *generate* the behavior) but it is not excluded.
4. **The coarse cells were never tested.** Root TM, distilled tables, and
   learned root budget allocation have zero games played against them in
   our ledger. Nothing about the per-node failure transfers to them — their
   T ≈ 1.00.

### 3.4 The two measurements that settle it definitively

- **K1 — the budget-decay curve (kills or licenses the signal claim).**
  Re-run exp-0029's exact configuration (gen-0 r1 rank-8, net-0 both sides,
  paired openings, 200+ games/point) at fixed nodes 2k / 8k / 32k / 128k.
  Prediction from this study: the +68.6 decays monotonically and is < +15
  by 32k. If instead it *holds* at 128k, the per-node signal is genuinely
  depth-robust, the k-ceiling argument is wrong, and the economics
  re-open at T ≤ 1.3 (then §4.4 becomes a priority instead of a long-shot).
  Cost: ~1–2 box-days (the 128k point dominates). This is the single most
  information-dense experiment available to the program.
- **K2 — the free-application test (kills or licenses the distillation
  cell).** Take the SAME gen-0 head. At each root, run ONE forward,
  materialize the 64×64 from/to logit table, and add `w·table[from][to]`
  into quiet ordering at EVERY node of that search (a stale-position prior
  — one lookup per scored quiet, identical cost class to a history plane;
  ~30 lines in `movepick.rs`). Measure fixed-node AND 10+0.1 equal-time vs
  policy-off, plus bench-NPS delta (must be < 1%). This is different from
  the failed root-only test (which ordered only the root move list) and
  from max-ply gating (which paid full inference near the root): it applies
  the *signal* everywhere while paying the *cost* once. If the +68.6
  retains even ±20% here, we have the first learned allocator in history to
  beat a mature hand-tuned stack at equal time, for free, and the research
  claim in §5 flips positive. If it nulls, the signal is inherently
  position-local (the prior must see the *current* node, which per §2.3 can
  never be afforded) — and the per-node formulation is closed with a
  publishable mechanism, not just a cost complaint.

Both experiments reuse existing artifacts and harnesses. Neither requires
training anything.

---

## 4. If it can work — the ranked formulations

Ranked by P(beats the inequality) × size of prize × cost to find out.

### 4.1 R1 — Distill-then-compile: root-frozen prior table (cost class B)

The theory: pay for position conditioning once per move (~1ms against a
100–600ms move budget, T ≈ 1.002), accept staleness across the subtree.
At 10+0.1 the root position and a depth-30 leaf differ, but the plies that
dominate ordering decisions (near-root, high-depth nodes) differ little.
Refinement ladder if K2 (§3.4) shows signal: re-materialize the table at
each ID iteration; per-piece-type destination tables (192 entries) blended
into continuation history initialization; two-table blend (root + previous
PV midpoint). Each step stays in cost class B.
**Minimal experiment: K2 above.** Prize if it works: +5–25 ET Elo and the
research result. Probability: 25–40% of retaining something > +10 at fixed
nodes (the staleness question is genuinely open); maybe half that at equal
time and real depth.

### 4.2 R2 — Learned root allocation: time management + budget shaping (cost class B/D-at-root)

One inference per move decides: stop-early confidence, extend-on-instability,
aspiration width, and (later) per-root-move node budgets. The free stack
here is thin — TM heuristics are the *least* fishtest-polished part of most
engines (SF's own TM patches routinely land double-digit fishtest gains
elsewhere in history; our design/12 implementation is basic; nodesEffort is
still on the queue). Inputs are search telemetry (score trajectory,
best-move stability, node distribution — all already in RootMoves/obs), not
position features, so this can even be cost-class-A (a fitted formula).
**Minimal experiment:** log 20k games of TM telemetry (the obs platform
already captures it), fit a stop/extend classifier offline against
"did further search change the move / result", replay-evaluate, then a
2000-game 10+0.1 SPRT of formula-TM vs learned-TM. Prize: +5–15 ET Elo
(literature-consistent; reasoning). Probability of positive: ~40–50%, the
highest in the portfolio, because the incumbent is weak and the cost is
zero. Note: this is also the ONE cell where our result would transfer
almost directly to the LLM budget-allocation analogue (§5.2).

### 4.3 R3 — Distill-to-constants at fleet scale (cost class A)

Use the counterfactual/consequence data (whose context-signal is real,
AUC 0.81) to *shape functional forms* — e.g., add the 2–3 most predictive
context interactions from the 16k-root campaign as new terms in the LMR
formula — then let fleet SPSA (D36) set the coefficients. This is "learned
search augmentation" in the only form the frontier has ever accepted.
Bounded by the plateau (exp-0078) to maybe +10–30 internal, deflating 2–3×
externally; needs the fleet to even resolve. Not a differentiated research
result — it is fishtest praxis — but it is where the context-signal finding
cashes out. Do it when D36 is live, as part of the queued SPSA season.

### 4.4 R4 — The long-shot: an incremental policy accumulator (cost class C, the "NNUE trick for ordering")

The only theoretically coherent per-node design. The bilinear head's
expensive step is projecting the (post-activation) trunk to from/to
embeddings. Move the projection BEFORE the nonlinearity:
`F(pos) = W_f · a(pos)`, where `a` is the raw accumulator — then F is
*linear in features* and maintainable by the same add/sub column deltas as
the accumulator itself (a rank-8 from/to pair adds 2×512 i16 lanes per
changed feature ≈ +30–60% of current accumulator-update cost ≈ ~10–25% NPS
[derived, not measured]; per-quiet score = rank-8 dot ≈ a history lookup).
T ≈ 1.1–1.3 at ALL-node consultation with zero gating — the configuration
that actually held the +68.6. Required k ≈ 1.1–1.3: exactly at the
plausible ordering-headroom ceiling. Expressiveness is lower (linear-in-
features embeddings, no cross-feature nonlinearity), so the gen-0 gain will
not transfer 1:1.
**Do not build this on spec.** It is 2–4 weeks of delicate engineering
(quantization, make/unmake plumbing, trainer support) for a knife-edge
prize. Its go/no-go gate is K1: if the per-node signal survives at 32k+
nodes (k ≥ 1.2 at depth), R4 is the vehicle that could ship it, and it
becomes the most interesting engineering project in the program. If K1
shows decay, R4 dies with it and nothing was spent.

### 4.5 Dead cells — do not reopen without new physics

Per-node fresh forwards at any size (measured T ≥ 7.4, §2.2); VOC-gated
consultation (VOC target unlearnable, §2.5); depth/ply-gated heavy heads
(anticorrelation, §2.3); helm training-data scale-up aimed at the per-node
MovePicker prior (the inequality is indifferent to the fixed-node number,
§2.4). The queued d41-gold-v2 train and any successor whose deployment
story is "per-node prior, W-blended into the picker" should stay cancelled.

---

## 5. The cross-domain research contribution

### 5.1 The transferable formalization

The portable objects from this study are the inequality `k > T` with
`T = 1 + f·c_i/c_n`, the **k-ceiling** (headroom over the incumbent
allocator, which for mature systems is a warm, online-learned stack — not
"no allocator"), and the **affordability–informativeness anticorrelation**
(§2.3), which I have not seen stated in the literature and which our gating
ladder measured directly. Define the *viability index* of a domain:
`V = (c_unit / c_allocator) × A`, where A is the number of units one
allocator decision influences. Learned allocation pays where V >> 1/(k−1).

| domain | c_unit | typical V | verdict predicted by the framework | observed |
|---|---|---|---|---|
| CPU alpha-beta, per-node | ~1–6k cycles | 0.01–0.2 | dead | dead (this study) |
| CPU alpha-beta, root/compiled | full move search | 10^4–10^6 | viable | universal practice (TM, SPSA, histories) |
| NN-MCTS (AlphaZero/Lc0) | full net forward | ~10^2 (policy head marginal on shared forward ≈ free) | viable | works — and note the policy there is not an *ordering hint*, it is the primary selectivity mechanism replacing ~3 orders of magnitude of expansion |
| LLM inference-time compute (budget/routing/stop rules) | full generation, seconds | 10^4+ | trivially viable on cost; headroom is the question | consistent with published positives (adaptive best-of-n, learned verifiers/routing, speculative-decoding policies) |
| Anytime planning / sim-based robotics | expensive rollout | 10^2–10^4 | viable | scattered positives |

The framework's non-obvious prediction, confirmed here: as a search's unit
cost falls, learned allocation does not disappear — it *migrates down the
cost ladder* into tables and compiled constants, and the learned artifact's
capacity is capped by the inference budget at its deployment site, NOT by
the available signal. Chess alpha-beta is the terminal case: the units are
so cheap that only ρ≈0 representations survive, and that is exactly what we
observe in every frontier engine.

### 5.2 Is there a crisp publishable claim?

Yes — one honest paper, two possible headlines depending on K1/K2:

- **Negative headline:** "When does learned compute allocation pay?
  A measured viability frontier in the worst-case regime." Contributions:
  the V-framework; the anticorrelation mechanism with the gating-ladder
  measurement; the k(N) decay curve (K1); the 16k-root causal result that
  the value latent adds no reduction-risk signal over search context and
  that the causal compute-delta is unpredictable (negative R² — a genuinely
  surprising, useful negative for the VOC literature); the 100× inequality
  audit of a real +68-fixed-node artifact. This is a solid
  workshop-to-mid-venue paper (AAAI/IJCAI search track, or the
  reasoning-budget workshops), valuable precisely because everyone's
  intuition ("just gate the inference to important nodes!") is refuted
  mechanistically, and because the LLM inference-compute community is
  currently rediscovering these tradeoffs with V ≈ 10^4 and no theory of
  where the frontier sits.
- **Positive headline (requires K2 or R2 to land):** "A learned allocator
  beating a mature hand-tuned search stack at equal wall-clock" — to my
  knowledge unclaimed for alpha-beta at any engine strength that matters.
  Even a +10 ET Elo root-distilled prior or learned TM on a 3400-Elo
  engine, with pre-registered NPS accounting, would be the first clean
  instance, and chess's Elo instrument makes it rigorous in a way no LLM
  benchmark currently is.

Why chess alpha-beta is a GOOD testbed despite being worst-case: the
measurement instrument (paired-opening Elo, fixed-node isolation,
bit-reproducible searches, an incumbent tuned by 10^8 games) is the
cleanest available anywhere for "net of cost" claims — LLM inference-time
studies almost never charge the allocator's own cost or compare against a
*tuned* heuristic incumbent. Why it is a BAD testbed: a negative here
bounds only the low-V regime; it must be framed as the frontier's boundary,
never as "learned allocation doesn't work."

### 5.3 The minimal rigorous cross-domain experiment

K1 + K2 (§3.4), pre-registered, plus one V-ladder replication: repeat the
same protocol on an MCTS-with-NNUE-rollouts toy (V ≈ 10^2) to show the
frontier crossing within one codebase. That triptych — dead at V ≈ 0.1,
knife-edge at V ≈ 10, alive at V ≈ 10^2 — is the paper's figure 1, and
ARCHON's infrastructure (obs platform, exact-intervention oracle, ledger)
is already most of the methods section.

---

## 6. VERDICT

**(i) Does learned search augmentation help ARCHON reach #1?**

- **Per-node formulation (the helm as deployed): NO — definitively, on our
  own measurements.** k ≤ ~1.5 shallow and decaying, T ≥ 7.4 at the
  cheapest true-shared-trunk implementation, gain non-retainable under
  every gate, latent adds no reduction-risk signal over context at 16k
  causal roots, VOC target unpredictable. Stop all helm training spend and
  all-node deployment work permanently unless K1 falsifies the decay
  prediction. This is stronger than the prior study's "correctly parked":
  parked implies waiting; the per-node cell is *closed*.
- **Coarse/compiled formulations: CONDITIONAL YES, modest size.** R2
  (learned TM, ~40–50% to be worth +5–15 ET) and R1/K2 (root-distilled
  prior, ~25–40% to be worth +5–25 ET) are cheap, unblocked by data scale,
  and their combined expected value (~+10–20 ET Elo) is real but not a
  path-to-#1 lever on its own — it is one more rung of the same ladder as
  speed and eval work. R4 (incremental policy accumulator) is the only
  formulation that could ever be worth >+30 and it is gated on K1.
- Sequencing: K1+K2 are ~3 box-days total, need no training, no fleet, and
  produce a publishable curve either way. They can run whenever the box is
  free without violating the "don't tunnel on the helm" ranking — they are
  measurement, not development.

**(ii) Independent of chess, is there a research contribution?** YES —
provided the program stops framing it as "make the helm win" and reframes
it as "measure the viability frontier of learned compute allocation, in the
domain with the best instruments and the worst economics." The negative
result is already 80% collected (this study assembles it); K1/K2 complete
it; a K2/R2 positive upgrades it from cartography to a first. Chess was
never going to be where learned allocation wins big — it is where the claim
can be *proven* to the standard other domains can't reach.

**The single most decisive next experiment: K1 — the Elo-vs-node-budget
decay curve of the gen-0 rank-8 head (2k/8k/32k/128k fixed nodes, paired,
200+ games/point), immediately followed by K2 — the root-frozen prior table
at equal time.** K1 decides whether ANY per-node vehicle (including R4) has
a cargo worth carrying; K2 decides whether the distillation cell — the last
cheap door — is open. Every other decision in this program is downstream of
those two numbers.
