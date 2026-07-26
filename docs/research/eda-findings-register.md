# EDA Findings Register — what we learned, what to do, what it's worth

**Built 2026-07-26** from the nnue-pytorch trainer study
(`stockfish_reference/09-nnue-pytorch-trainer-study.md`) and the 2026-07-25/26
exploratory analysis (`jax-workstream-plan.md`, the `★` sections).

**How to read the columns.** *EV* is expected Elo unless stated otherwise;
`?` means genuinely unknown, not "small". *Confidence* is about the FINDING,
not the fix. *Cost* is wall-clock including the games needed to verify.
Anything marked **MEASURED** has data behind it in this repo; everything else
is mechanism-based reasoning and should be treated as a hypothesis.

**The governing caveat, established twice tonight:** static-eval accuracy
does NOT rank nets by playing strength (across four ARCHON nets spanning 86
Elo it ordered them *wrong*). Every eval-quality item below is therefore
"plausibly worth Elo", never "worth Elo". Only games decide.

---

## A. ON THE v1 CRITICAL PATH (do before release)

| # | Finding | Action | EV | Cost | Conf |
|---|---|---|---|---|---|
| A1 | **FT permutation is free and lossless.** SF's `ftperm.py` maximizes all-zero 4-groups; our `find_nnz` skips exactly those groups. We have no equivalent. | Build it. Gather activations ~1M positions → greedy 2-swap/3-cycle → apply to FT columns + L1 rows with the paired `x+HL/2` extension → assert bit-identical eval. | 0 to +10 (pure NPS) | ~1 day | high (it works for SF) |
| A2 | **A/B first: occupancy may not drive our NPS.** D47M at ~15% active groups vs champion ~25% benchmarked IDENTICALLY. MEASURED. | Permute ONE net, measure NPS before/after on an idle box. Decides whether A1 is worth finishing. | decision-only | ~1 h | med (3 positions, loaded box) |
| A3 | **Dead-neuron count is free with A1.** Always-dead accumulator entries are prunable at ZERO capacity cost and cut BOTH the update and the matmul — unlike activation sparsity, which cannot touch the accumulator. | Same gather pass reports per-neuron firing rates. If ~10% never fire, prune the width. | 0 to +15 | +0 (bundled) | med |
| A4 | **We have no engine↔trainer parity gate.** SF's `cross_check_eval.py` reports per-position error percentiles + worst-offender FENs. | Port the idea into our settle battery next to `net_scale_check.sh`. | 0 Elo, bug insurance | ~2 h | high |

## B. V2 NET PROGRAM (from-scratch retrain; cannot be retrofitted)

Ordered by EV/cost. Items B1–B3 are the ones I would not skip.

| # | Finding | Action | EV | Cost | Conf |
|---|---|---|---|---|---|
| B1 | **★ Target transform is badly miscalibrated.** `σ(s/400)` vs the mixture-fitted `a=163, b=76.9`: Brier 0.0626 → 0.0465 (−26%). Our target says 0.65 where the outcome data says 0.80, and we blend 30% outcome into it — so the two halves of every target have been disagreeing all campaign. MEASURED, MLE, held out, bootstrapped, 12 corpora. | Use the fitted double sigmoid. **ALREADY IMPLEMENTED** (`target_offset`, commit a9dd7e8; `offset=0` is bit-exact backward compatible; 6 verification checks pass). Needs the 3-arm training test. | +5 to +25 | 6 h (3 short arms + screens) | high (finding) / med (Elo) |
| B2 | **Uniform ±1.98 weight clip is binding almost everywhere.** MEASURED on a live net: `l0_pstw`, `l0_threatw`, `polmovew`, `polmoveb` all pinned at 1.98 — including tensors exported as **f32**, where there is no quantization reason to clip, and the **i16 FT**, whose real ceiling is ~128. SF derives a clip per layer from that layer's own scale. | Per-tensor clips matched to export encoding. Drop entirely on f32 tensors; widen on the i16 FT; keep ±1.98 only for i8@64. | +5 to +20 (capacity) | 1 h + A/B | high (finding) / med (Elo) |
| B3 | **Our batch composition is frozen forever.** Contiguous 8192-record windows from pre-shuffled corpora ⇒ record *i* and *i+1* share a batch on every epoch of every leg, permanently. SF reshuffles at position granularity through a 1M-entry buffer every pass. | Position-level shuffle buffer in the loader. | +0 to +15 | ~4 h | high (finding) |
| B4 | **λ is constant from step 0; SF ramps it** (1.0→0.75, i.e. outcome weight 0→0.25). Early training then never depends on outcome calibration. Pairs naturally with B1. | Ramp `wdl` 0 → 0.3. | +0 to +10 | 1 h | med |
| B5 | **`pow_exp` 2.0 vs SF's 2.5** (their docs say 2.6 also worked). Higher exponent buys precision at the cost of outlier accuracy — precisely the axis where we trail SF18. | Screen 2.5. | +0 to +10 | 2 h | med |
| B6 | **Threats quantized 4× coarser than SF** (64 vs 256), then rescaled ×255/64 at runtime. ~100–160 active threat features ⇒ ≈20 quanta of accumulated rounding noise. Only 0.095% of our threat weights exceed SF's ±0.496 clip. | Raise threat quant to 128/255 + training-time clip on `l0_threatw`. Deploy-format change. | +0 to +15 | ~1 day | med |
| B7 | **No gradient clipping, no NaN guard, and we decay biases** (SF: global norm 2.0, `TerminateOnNaN`, biases never decayed, weight decay 0 by default). | Add all three. | ~0 Elo, stability | 1 h | high |
| B8 | **Data filters we lack:** WLD-coherence *probabilistic* thinning (respects "don't filter away good data"), soft ply ramp instead of our hard `ply≥16` cut, `random_fen_skipping` for intra-game decorrelation, placeholder-zero detection. | Port into the export tools. | +0 to +15 | ~1 day | med |
| B9 | **Cosine-to-floor needs `lr_end_superbatch` committed up front** — the knob that has repeatedly bitten the pump chain. SF's per-epoch exponential γ is schedule-length-agnostic and composes across warm-start legs. | Exponential decay for chained legs. | ~0 Elo, ergonomics | 1 h | high |
| B10 | **Bucket init:** SF copies bucket 0 into all 8 layer-stack buckets and inits PSQT from HCE piece values; we draw 8 independent randoms and zero-init PSQT. | Match. | ~0 (we warm-start) | 30 min | high |
| B11 | **No QAT.** Our i8 tensors quantize at 1/64 = 0.0156 against a p99 magnitude ~0.24 — ~6% relative error, uncompensated. SF simulates it with straight-through estimators by default. | Exactly one experiment. Do not build a program around it. | ? | ~1 day | low |
| B12 | ~~Mate/TB remap instead of hard drop~~ | **DOWNGRADED.** Our own analysis killed the rationale: the excluded positions aren't in the test set (canon clamps at 2499), and blowout precision demonstrably doesn't help — restricting to decision-relevant positions *flips* the SF18 gap in our favor. | ~0 | — | high |

## C. POST-RELEASE RESEARCH

| # | Finding | Action | EV | Cost | Conf |
|---|---|---|---|---|---|
| C1 | **FT-sparsity penalty is unjustified by evidence.** Its only articulated benefit (NPS) does not appear in measurement; its cost is demonstrated (it destroyed the policy substrate in D47M when unopposed). SF has no activation penalty at all. NOT the same as "wrong" — it may regularize, untested. | Two short legs, λ 0.001 vs 0, comparing heldout + screens + properly-measured NPS on an idle box. | +0 to +15 (capacity) | 8 h | med |
| C2 | **The penalty is also mis-shaped.** It charges per nonzero *activation*; the engine charges per nonzero *4-group*. And it penalizes *magnitude*, shrinking large useful activations — a pure capacity tax. | If C1 says keep it, reshape as group lasso over the deploy groups. | +0 to +10 | 2 h | high (shape) |
| C3 | **Our eval is more regime-dependent than SF18's.** A single global affine fit costs us more than it costs them (canon gap +6.3 overall vs +2.1 within SF's big-net regime). MEASURED. | B1 is the hypothesized fix; test whether a per-regime affine fit still beats a global one afterwards. | diagnostic | +0 (bundled with B1) | med |
| C4 | **We have complementary knowledge to SF18.** corr(our error, SF18 error) = 0.747 vs SF16's 0.813; a 50/50 rescaled ensemble scores 60.6 MAE, **beating SF18's 61.9**. MEASURED. | Research only — points at what a distillation target or a second head could capture. | ? | — | high |
| C5 | **SF ships two nets with a material-based switch** (`use_smallnet`: `\|simple_eval\| > 962`, big net re-check if `\|nnue\| < 277`). We have no equivalent. | Consider a small fast net for decided positions. | ? (speed) | ~1 wk | high (finding) |
| C6 | Multi-head training levers left unswept: anchor strength (λ_pol 0.002 was intuition), alternation frequency, output-space (KL) anchoring vs label CE, separate head LR, gradient surgery, dedicated capacity. | The post-release multi-head queue. | ? | — | — |
| C7 | **From-scratch joint training at scale** — every joint experiment so far fine-tuned a value-saturated trunk. | Flywheel era. | ? | — | — |

## D. DATA-QUALITY ACTIONS (cheap, do opportunistically)

| # | Finding | Action | EV | Cost | Conf |
|---|---|---|---|---|---|
| D1 | **`wrongnnue` is a frame outlier.** Fitted `a=246.8` against 143–183 for every other corpus (2.6% of the mix). MEASURED. | Re-audit its score frame before the V2 build. | avoids a data bug | 1 h | med |
| D2 | **The per-corpus transform fit is a better frame-alignment diagnostic than our DIV audits.** `a` encodes score scale and is stable at ~165 across 11/12 corpora — evidence D46 harmonization *worked*; `b` encodes conversion sharpness and legitimately varies with draw rate (33–118). | Make `fit_target_transform.py` a standing gate for every new corpus: `a` far from ~165 ⇒ scale bug. | avoids data bugs | done (tool exists) | high |
| D3 | **canon-v1 `corpus.quiet.txt` labels are WHITE-POV**, and its `wdl` column is all 0.5 (no outcomes). Reading it as stm collapses every evaluator to Pearson ~0.04. | Documented; `eval_bench` has `--labels-white-pov` and a <0.30-correlation guard. **Open assumption:** the .txt→.bin conversion must apply the flip (our nets work, so it evidently does) — worth a 5-minute verification. | avoids a silent catastrophe | 5 min | high |

## E. PROCESS / DISCIPLINE (free, already paid for in mistakes)

- **Sanity-check against a known quantity BEFORE the sophisticated analysis.** A three-position material probe killed a confident wrong conclusion about eval calibration that had survived a full regression analysis. Two frame errors compounded: misread `polyfit` direction, then compared against SF's *UI-normalized* cp when the question was about *internal* units.
- **100-game screens never carry narrative weight.** 200+ or silence. The d47j "value tax" was invisible for a full day because 100g bars are ±30.
- **When every method scores near-random, suspect the data, not the methods.** Six evaluators at Pearson 0.04 meant a POV bug, not six bad evaluators. Now a guard in `eval_bench`.
- **Restrict to the regime that matters before concluding.** "SF18 is 10% better" became "we win where it decides moves" under the decision-relevant restriction.
- **An observed correlation from one configuration is not a law** ([[mechanism-over-correlation]]) — write the mechanism first, then check whether the observation follows necessarily or only under the config that was run.
- **Never run heavy CPU/RAM jobs beside a trainer**; if unavoidable, cap them in a `systemd-run --scope -p MemoryMax=…` cgroup so they die instead of the trainer.

---

## Recommended order

1. **Finish v1 on the current line.** Nothing above blocks it. A1–A4 slot into the
   existing profiling/optimization phase; A2 gates A1.
2. **Run the B1 test as soon as the GPU frees** (post-D47Q). It is implemented,
   verified, backward-compatible, and it is the largest measured defect we have
   found in the training recipe.
3. **B2 next** — same shape of finding (measured, cheap, capacity-recovering).
4. **Bundle B3–B10 into the V2 from-scratch build**, since they cannot be
   retrofitted into the D47 warm chain without voiding its comparability.
5. **C1 before any future run keeps paying the sparsity penalty by default.**
6. D1–D3 opportunistically; E is free and permanent.
