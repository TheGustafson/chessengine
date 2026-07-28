# D41 reframe: learned search augmentation as a TRAIN-TIME auxiliary task (2026-07-19)

**Origin:** user critique of `learned-search-augmentation-first-principles-2026-07-19.md`.
The first Fable study is rigorous but **wrongly scoped**: it steelmanned the
*inference-time per-node* formulation and closed it correctly, then treated the
whole program as closed. It never evaluated the formulation that matters most —
and that the user flagged: **learned search structure used only at TRAIN time to
shape an eval network that is deployed EVAL-ONLY at inference.**

## Why the first study does not touch this

Its entire falsification is the cost inequality `k > T`, where
`T = 1 + f·c_i/c_n` is the *inference* tax. The train-time-auxiliary formulation
has **T = 1 exactly** (the auxiliary head is dropped at deployment; inference
cost is identical to today's net). So the `k > T` argument — the whole spine of
the study — **does not apply**.

The closest evidence it offers is §2.5 (the counterfactual campaign), but that
does not falsify the auxiliary idea:
1. It **probed a FROZEN representation** ("does the net-0 latent help predict
   reduction-risk / VOC?") — it did NOT **co-train** (add an auxiliary loss and
   let it shape the trunk). Multi-task regularization can help the main task even
   when the auxiliary target is noisy or already-cheap; probing-a-frozen-net says
   nothing about that.
2. It picked the two **provably-hardest targets** — VOC-as-signed-node-delta
   (negative R², genuinely unlearnable) and scout-bound-error (already
   cheap-scalar-predictable). But we KNOW a learnable search signal exists:
   **gen-0 at 100M rows learned move-ordering to +68.6 fixed-node** (exp-0029).
   A **policy / move-preference / per-move search-value** auxiliary target has
   real signal; the study never connected that to co-training.
3. It was measured on **our trunk, our labels, our (small) data** — exactly the
   "not near optimal / bad data" caveats the user raised.

## The corrected hypothesis (H-AUX)

Co-train the D40 eval trunk with an **auxiliary head predicting the teacher's
deep-search move preference / per-move value** (the LEARNABLE signal, NOT VOC),
weighted as a regularizer alongside the scalar value loss. **Deploy eval-only**
(drop the aux head → byte-identical inference path, T = 1). The bet: the
auxiliary task forces the trunk to encode search-relevant structure, yielding a
**strictly better eval at zero NPS cost** — Elo with no tax, and the research
result the user cares about (a search-aware representation that helps at equal
compute).

This is the standard "auxiliary-task / representation-learning" bet (cf. how
AlphaZero's policy head co-trains the value trunk), applied to a CPU alpha-beta
NNUE where the policy head is *discarded* at inference.

## The experiment

- **Data:** teacher (SF18/deep) per-move values or best-move over a real
  position set — the label-value / obs-Tier-F pipeline already produces the
  root-analysis per-move data; may need more/better data (the user's point).
- **Two trained nets, identical arch/data/schedule except the aux loss:**
  A = value-only (control); B = value + auxiliary policy/search-value head.
- **Deploy both eval-only** (aux head dropped from B).
- **Gate:** B vs A at **equal nodes** (isolates eval quality) AND **equal time**
  (10+0.1, the ship metric). Bit-identity is NOT expected (different weights);
  this is a real A/B, so ≥+10 equal-time with SE control is the bar.
- **Ablations:** aux weight λ sweep; target choice (best-move CE vs per-move
  value regression vs margin); which layer the aux head taps.

## Cost / gating

- On-box, GPU training (no cloud). A few box-days for the A/B + ablations.
- Independent of the speed program and the inference-time helm (which stays
  closed). Does NOT need the D36 fleet for the first read (fixed-node + a local
  equal-time gate), though the fleet would tighten it.
- Sits alongside the strategic sequencing: it's an EVAL-quality lever (the plan's
  #1) with a novel mechanism, not the deprioritized inference-time helm.

## Next step (recommended)

One tight **Fable study scoped specifically to H-AUX** — steelman it, design the
auxiliary target + loss + which teacher signal + data requirement, and predict
whether representation-shaping can plausibly move a 3400-Elo eval (or whether
the eval is already near the value-information ceiling). THEN spend training
cycles. Do not let the first study's (correct, but narrow) inference-time
verdict foreclose this — they are different bets.
