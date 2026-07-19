# Data-quality curriculum strategy (2026-07-19)

As we accumulate training data from sources of differing label quality (cheap
5k-node binpack labels, SF18@50k deep relabels, mined error positions, future
tablebase/deeper oracles), we need a principled way to combine them. This doc is
the strategy, grounded in our own empirical findings + Stockfish's proven recipe.

## 0. THE CORE STRATEGY (2026-07-19): targeted relabel + active-learning flywheel

**Reframe (user):** don't relabel to *improve* labels — relabel to *fix the
wrong ones*, pull them out of the bulk first-pass, and put them exclusively in
the higher tiers. Progressively weed the problematic positions out of the bulk
until what remains is certified-clean.

**Why this is the whole ballgame.** Our disagreement analysis (123k matched
positions) found SF18@50k and the 5k labels agree on the winner **99.8%** of the
time. Economically that means **blanket relabeling is ~99.8% wasted** — for
almost every position the cheap label already gives the right verdict. ALL the
value is in the small wrong tail. So this is a data-TARGETING problem, not a
data-VOLUME problem. Relabel ~5–15% (the wrong tail), capture most of the value,
fit the 96-vCPU quota, spend ~$8 not ~$64. (`partition_fen_manifest.py` was
literally built for "the disagreement-selected subset"; blanket c1/c2 was a
deviation — this returns to the original design intent.)

**Solving the chicken-and-egg (which 5k labels are wrong, without deep-labeling
first):** we are already generating the answer key. c1 = 20M positions with BOTH
5k and 50k labels → a supervised training set for a cheap **wrongness predictor**:
net-free features (`|cp_5k|`, eval-vs-WDL disagreement, ply/phase, in-check,
capture) + the net's eval + the EXP-UNC-1 uncertainty head, regressed onto actual
`|cp_5k − cp_50k|` / winner-flip. Apply it to the un-relabeled 168M to SELECT the
suspect subset. Free corpus signals (eval-vs-outcome, obs aspiration-BOUND flags)
and the behavioral signal (mined Archon-loss positions) are additional inputs.
This UNIFIES mining + EXP-UNC-1 + obs bound-flags as inputs to one selector.

**"Wrong" = impact-weighted, not raw |Δcp|.** Most disagreement is the deep
search being more *moderate* (calibration, trains fine). Select on TRAINING
IMPACT: winner-flips and large errors in the near-balanced/decisive band where the
target is most sensitive. 5k=+500/50k=+490 → worthless; 5k=+150/50k=−60 → gold.

**The flywheel (= "slowly become confident"):** each round — train on the
certified-clean bulk + corrected/upsampled high tiers → score all bulk positions
for wrongness → deep-relabel the top suspects → MOVE them out of bulk into the
high tier with correct labels → repeat. Bulk gets monotonically cleaner (a
durable compounding asset); the net trains on clean labels THROUGHOUT (not
clean-bulk + noisy-hard-positions as today). Converges when few new suspects surface.

**Adjustment by area:** LABELING — cascade 5k → wrongness-score → deep-relabel
only the impactful-suspect tail (depth scales with selected value; endgames →
tablebase later). TRAINING — first pass on CLEANED bulk (drop/downweight the
flagged+relabeled), reintroduce them via upsampled high tier. CLOUD — let c1
finish (it's the answer-key), DO NOT submit c2 as blanket; replace with a targeted
relabel of the selected subset (~10–20M, fits quota, ~$8). [User decision pending.]

**Risks:** over-cleaning the bulk (strips hard positions) → re-inject upsampled +
validate on held-out hard positions. Proxy imperfection → ensemble signals +
MEASURE recall on the c1 answer-key before trusting at scale. Circularity (net
picks its own targets) → keep net-free signals primary, iterate so blind spots shift.

**Immediate validation — DONE 2026-07-19 (tools/nnue/wrongness_calibrate.py, 222k
matched positions). RESULT: cheap net-free features CANNOT target the impactful
errors.** outcome_mismatch/near_zero/abs_cp5k/ply all had recall@k ≈ random (lift
~1x) or worse for the impactful-wrong set. `abs_cp5k` predicts *total* |Δcp| almost
perfectly (r=0.98) but 0% of impactful errors — i.e. it finds the harmless
extreme-position calibration tail, not the near-balance corrections that matter.
Two consequences that REVISE §0's optimism:
- The "small wrong tail" premise is shaky: winner-flips are 0.13% (tiny, likely
  irrelevant); impactful near-balance corrections (|Δcp|>100, |cp50k|<250) are
  **33%** — a THIRD of positions, not a 5–15% slice. Targeting can't be "relabel a
  tiny fraction," and cheap features can't isolate even that third.
- BUT 33% of positions carry big corrections in the training-SENSITIVE near-balance
  band (where the sigmoid loss is steepest) — a real argument the labels matter,
  which the 99.8% winner-agreement had hidden. The mix gate sizes whether it
  translates to Elo.
Revised targeting plan: cheap-feature selection is OUT. Viable paths: (a) NET-based
signals (incumbent eval + EXP-UNC-1 uncertainty — a forward pass, still far cheaper
than deep search; UNTESTED, the natural next calibration); (b) a medium-depth (25k)
screen cascade (relabel-cheap-everything → deep only the flagged); (c) behavioral
selection via mining. If none give strong recall, the honest fallback is that
label-depth wants broad application (large fraction), not surgical targeting.

The sections below (two-axis framing, mixing mechanics, curriculum staging,
finetune nuance) are the SUPPORTING detail for how the tiers are trained once
selected.

## Two axes, not one quality ladder

The naive framing is a single "low→high quality" ladder. Reality has **two
orthogonal axes**, and conflating them wastes effort:

1. **Label quality** — the depth/accuracy of the eval on a position (5k → 50k →
   tablebase). **This is a MODEST lever for us.** The disagreement analysis
   (123k matched positions, 2026-07-19) found SF18@50k vs the original 5k labels
   agree on the *winner* 99.8% of the time (r=0.93); they differ in
   *magnitude/calibration* (mean |cp| 515→189), not verdict. A curriculum built
   purely on label depth has a low ceiling.
2. **Position value** — how *informative* a position is (random UHO → error-mined
   → curated). **This is the SHARPER lever.** Mined Archon-loss positions aren't
   better-labeled, they are positions where our eval is *provably wrong* (we
   thought we were fine, then lost). Weight effort here over chasing label depth.

## The proven backbone (do this, it already won)

Broad-cheap-first → refine is exactly Stockfish's official two-stage recipe
(nodes5000pv2 stage-1 → farseer stage-2 warm retrain, LR halved) and our own
winning chain (u1024 → uf2048 → s3-farseer). The B4 finding was that data
*volume* was the constraint. Lead with the abundant cheap corpus to build the
representation — NOT because cheap data is "easier" (it's noisier + more extreme),
but because it's abundant. Then sharpen.

## The failure mode: a POORLY-TUNED small finetune (not an inevitable one)

Correction (user pushback, 2026-07-19): a small pure high-quality finetune does
NOT *have* to overfit/forget. My −636 Elo disaster was a *badly-tuned* finetune —
LR 5e-4 for 6 SB, which on 5.69M is **~26 epochs** at a high LR (that's a lot of
hard passes, not a gentle polish). Catastrophic forgetting is a RISK controlled
by LR + number of presentations + how far the finetune distribution sits from the
base; with a low LR and few presentations you can find a non-forgetting regime.

The honest tension is a **narrow sweet spot**: gentle-enough-not-to-forget ↔
strong-enough-to-matter. Too gentle → the net barely moves → no benefit; too hard
→ it forgets. The sweet spot exists but narrows as base/finetune distributions
diverge, and each tuning iteration costs a train + gate. Two wrinkles for us:
- **Payoff vs tuning cost.** For *label depth* the payoff is modest (99.8%
  winner-agreement) → the sweet-spot search may cost more than it returns. For
  higher-value data (mined errors, future tablebase) the payoff is larger → the
  tuning is worth it.
- **Trainer granularity.** 1 superbatch ≈ 1520 batches ≈ several epochs on a small
  set, so a *truly* gentle finetune needs a smaller batches-per-superbatch (check
  the D40 knob) to dose sub-epoch presentations.

So the mixed/upsampled refine (below) is preferred not because pure finetune MUST
fail, but because it reaches the same goal without walking the LR tightrope. A
gentle pure finetune stays in the toolkit — TEST it (a train + gate is cheap)
rather than assume either way. Both approaches must still be **gated on a real
game** (the −636 passed a 3-position sanity check).

## Non-negotiable mechanics

- **Scale-harmonize before mixing.** Sources are on different cp scales: binpack
  needs ÷1.9 (SFBINPACK_SCORE_DIV) to reach UCI-cp; fleet relabels are already raw
  SF18 UCI-cp (verified: recompute ratio 1.000, trainer mult=1.0/scale=400).
  Mixing raw → contradictory targets. Everything to ONE scale (UCI-cp) first.
- **Substitute, don't concatenate, on overlaps.** The good relabels are the SAME
  FENs as the cheap ones (fleet = first 20M of the binpack). Concatenating feeds
  the net both labels for one position and dilutes the signal. The clean version
  *replaces* the cheap label in place (FEN-keyed).
- **Gate every stage with a real fixed-node game on a large-corpus net.** Sanity
  fens don't catch a broken net (the −636 passed them).
- **Upsample high-quality + mined positions 2–4×** at the refine stage — the right
  knob (matches the "upsample at later epochs" instinct).

## Concrete staged pipeline

1. **Stage 1 — representation:** broad cheap data (24.7M binpack). *Have it — the
   incumbent d40-uho-720sb.*
2. **Stage 2 — refine (the active work):** warm-retrain (LR/2) on a
   **scale-harmonized, label-SUBSTITUTED** corpus where SF18@50k replaces the
   cheap labels for covered positions, kept large by the binpack base, with the
   high-quality + mined positions upsampled 2–4×. Gate vs incumbent (fixed-node).
3. **No tiny pure finetune.** Any "polish" is a low-LR pass over a *large*
   high-quality-*weighted* mix, never a small pure one.

## Calibration point (pending)

`fleet-mix-v1` (warm-start incumbent → continue on 30.4M = 24.7M binpack + 5.7M
fleet-relabel, CONCATENATED = the diluted first cut) is training now; its gate is
the first read on whether the label-quality axis is worth a curriculum at all.
**If null → pour effort into mining more informative positions, not label depth.**
The clean label-SUBSTITUTED Stage-2 is the follow-up if the diluted cut shows signal.
