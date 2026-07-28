# Post-D40 research direction (governing plan)

**Status:** the enshrined research program *after* the D40 eval foundation
stabilizes (user, 2026-07-20). This is a FRAMEWORK + a menu of axes, deliberately
NOT a fixed recipe — pick the highest-EV cell each loop, gate honestly, let results
choose the next. It sits above the specific docs it points to (wdl-head-design,
archon-loss-profile, d41-auxiliary-task-adversarial, transformer-head-exploration,
design/19).

## The governing principle
We pay for one EXPENSIVE deep SF18 search per training position and capture a rich
4-layer superset (design/16 `.ceobs`), but the shipped net distills exactly ONE
number out of it (the cp). The program has two axes, pursued together and gated the
same way:
- **Axis A — exploit ALL the captured data** (more targets, more output heads).
- **Axis B — search the ARCHITECTURE space** we have never explored (depth, width,
  accumulator design, features, activations, heads).
Neither is the biggest lever — eval-quality via data DENSITY on ARCHON's weak
positions (hard negatives) is (archon-loss-profile-2026-07-20.md). A + B are the
*polish + explore* layer once the eval foundation is strong. Do not over-invest
before it is.

---

## Axis A — use all the SF18 data

Signal inventory (what one deep search already gives us; `.ceobs` L1–L4):

| signal | captured in our shards? | net use it enables |
|---|---|---|
| `final_score` (cp) | ✅ used | value head (today) |
| game result (WDL) | via FEN-join | WDL blend target (done) + WDL **output** head |
| `best_move` | ✅ captured, UNUSED | best-move-CE head (aux / policy) |
| `id_snapshots` (eval-vs-depth trajectory) | ✅ captured, UNUSED | **decisiveness/uncertainty** head (LABELLED, not self-derived); **search-augmentation** signal `final − shallow`; **time-to-depth** for TM |
| `pv` | ✅ captured | policy consistency / deeper targets |
| per-move L4 (value, ordering, effort, pruning) | ❌ reference-only skipped it (35× relabel) | full policy head; learned pruning/extensions |

Framework (not an order):
1. **Every signal → a candidate target or output head.** The heads plug into the
   already-implemented "extra-output head + sidecar + env-gated consumers" machinery
   (see wdl-head-design Part A).
2. **For each head, test BOTH co-trained (tuned λ) AND frozen-trunk** — we have NOT
   proven either wins; co-training may improve the eval itself (AGZ/KataGo/our +63
   policy pilot), frozen can't hurt it. Sweep λ, log the value/aux gradient-norm
   ratio, keep the λ→0 control.
3. **Cheapest first: best-move and trajectory are already captured** — using them is
   an EXPORT problem, not a re-label problem. The L4 per-move policy is the one real
   re-labeling fork (35×) — defer until the cheap heads prove the multi-head thesis.
4. **Outputs earn their keep at inference** (WDL/decisiveness → TM + optimism +
   margin scaling; ARCHON's TM has zero position content today = the most
   complementary consumer) OR at training (aux heads that shape the trunk, dropped
   at deploy, T=1). Classify each head as used-output vs aux-only and gate accordingly.

---

## Axis B — architecture ablations (the space we have never searched)

The D40 arch (HalfKA+FullThreats → 768 acc → per-bucket 768→16→32→1, ×8 buckets,
pairwise-SCReLU) was chosen by porting Reckless, not by ablation. Width saturated
at 2048 on the OLD data/recipe — re-open on the corrected canonical data. Menu of
single-variable ablations (each from a fixed baseline, one knob at a time):

- **Depth:** add hidden layer(s) to the head (e.g. 768→32→32→1, or a 4th layer);
  deeper trunk before the accumulator bottleneck.
- **Width:** wider L1/L2/L3; wider accumulator (768→1024/1536/2048); re-run the
  width ladder on the canonical data (the saturation was recipe-specific).
- **Dual / multiple accumulators:** a second accumulator (e.g. a cheap HalfKA-only
  side accumulator — the smallnet-gate infra already maintains a second
  `HalfkaContext`), or separate accumulators per feature family fused late.
- **Output buckets:** count (8 → 16/32) and bucketing scheme (material vs phase vs
  learned); per-bucket vs shared tails.
- **Features (input space):** king-bucket factorization (#27, zero inference cost,
  the single most under-exploited eval lever); FullThreats on/off / variants;
  HalfKAv2_hm variants; new relational features; feature dropout.
- **Activations:** SCReLU vs CReLU vs pairwise-mul variants; where pairwise applies.
- **Residual / skip connections** into the tiny head.
- **Quantization:** the i16-FT / i8-L1 / f32-tail scheme — clip points, unified
  quant (kills the i32 activate tax), int8 vs int16 tradeoffs.
- **Heads as arch (Axis A ∩ B):** attention/transformer head off the accumulator
  (transformer-head-exploration.md); shared-trunk multi-head shapes.
- **Speed-as-arch:** the smallnet gate (route decided nodes to a cheap net) — an
  arch change that buys NPS → equal-time Elo.

Framework: fixed a baseline recipe (the canonical-data control), change ONE knob,
fixed-node screen → equal-time gate. Track a small standing table of ablation
results so the space is mapped, not re-guessed. **The concrete, runnable test matrix
lives in `arch-ablation-suite.md`** (T0 king-bucket factorization #27 first, then
head-shape, width re-sweep, activation, second routing axis, feature-contribution,
learned-router MoE, multiple accumulators, trunk-mixing — configs, priors, run order).

---

## Governing methodology (the rules that keep this honest)
1. **One variable at a time**; fixed-node screen (isolation) → equal-time gate (ship
   bar). Fixed-node ≈ equal-time only for zero-cost changes; heavier arches must earn
   their NPS tax at equal time.
2. **Pair every eval change with a search retune** (co-adaptation — the −582 scale
   bug and the ft50k −196 both came from an eval whose calibration the old search
   couldn't consume). Keep ONE scale (this session's SCORE_MULT lesson).
3. **Test competing designs; don't crown one from a single/confounded result**
   (the EXP-UNC-1 over-generalization correction). "We haven't proven anything yet"
   is a valid status; state what an experiment *narrowly* showed.
4. **Attribute Elo honestly by gate**; internal gains deflate 2–3× externally; track
   vs the SF16 anchor, not the internal ladder.
5. **The hard-negative flywheel regenerates per net** — a stronger net loses on new
   positions; re-mine (SF18 AND Reckless opponents) after each promotion.
6. **Ship-form parity + the D24 bench discipline** apply to every arch change.

---

## Loose sequencing (results re-order this)
1. **Eval foundation + data density** (canonical SF18 net → Phase-D hard negatives)
   — the biggest lever, in flight. Everything below waits on a strong eval.
2. **Cheap captured-data heads** — best-move (aux), trajectory→decisiveness (TM /
   uncertainty), WDL output. Export-only, both co-train + frozen tested.
3. **Architecture ablations** — factorization #27 and the width/depth re-ladder
   first (cheap/proven-shaped), then accumulator/feature/activation ablations.
4. **Expensive forks** — L4 per-move policy relabel; transformer heads; smallnet gate
   trained net. Only after 2–3 pay.
5. **D41 learned search allocation** (design/19) — the search-augmentation signal
   (`final − shallow`) is captured; this is where it gets consumed.

Track everything in the ledger + value-foundation-state.md; this doc is the map.
