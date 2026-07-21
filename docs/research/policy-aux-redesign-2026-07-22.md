# Policy I/O redesign — plan and options (2026-07-22, ~02:30)

**Trigger (user):** "We can't train a net with this shitty encoding. Stop now
and rethink all inputs and outputs from scratch." The pol1/pol4 runs were
KILLED mid-train (pol1 at ~sb65, no checkpoint — SAVE_RATE=240 — ~1.3h GPU
sunk). The WDL sweep (encoding-unaffected: clean 3-way softmax) was allowed to
take the GPU (`haux-wdl006-240sb` training; kill it too if a full stop is
preferred — nothing depends on it).

The killed config carried three KNOWN structural handicaps (marginals-only
output, cross-half mass competition, rank-32 attachment) — running it anyway
was a no-weak-tests violation in spirit. This document is the from-scratch
rethink, scoped honestly.

## 1. Scope: what the evidence says to keep vs rethink

| Surface | Evidence | Verdict |
|---|---|---|
| Value INPUTS (HalfKAv2_hm + FullThreats) | +696 vs PSQT; +265 composed; beats classic features at fixed nodes (exp-0065); D45a | **KEEP.** Rethinking discards the most-validated asset. Open questions (1536 width, threats NPS) stay parked as data-driven follow-ups. |
| Value OUTPUT (8-bucket scalar) | the shipping engine | **KEEP.** |
| WDL / σ outputs | clean 3-way softmax / live consumer | **KEEP** (sweep running / deployed). |
| POLICY AUX output | zero positive results ever (exp-0121 −14±23 confounded; tonight's config killed pre-readout) | **REDESIGN — this is the rethink target.** |
| Policy-specific inputs | never explored | In scope for options C/D below. |

## 2. Design axes (from tonight's review)

1. **Attachment/capacity:** hl3 (32-d, rank-32 ceiling — the killed config) →
   hl1/ft (768-d pairwise accumulator, fully expressive) → a small dedicated
   policy tower off the accumulator (capacity isolated from the value path).
2. **Output structure:** per-half marginal softmaxes → conditional
   P(to | trunk, from) (teacher-forced binding) → low-rank joint
   `s(m|x)=b(m)+h(x)ᵀe(m)` (design/19 §4.4's deployed form) → full 4096-class
   joint (needs sparse targets or smaller batch).
3. **Target richness — our unique data moat:** one-hot best move (crudest) →
   top-k SOFT targets from the Tier-R obs shards' per-move ARM VALUES
   (softmax over arm cp at temperature τ). We are one of the few shops with
   per-move search values at scale; one-hot throws that away.
4. **Loss hygiene:** LEGAL-MOVE MASKING — compute CE only over legal moves
   (the chessengine crate is already linked into the trainer; movegen at
   train time is feasible, measure throughput). Removes the dominant target
   noise (63/64 squares are illegal-from most positions).
5. **Instrumentation FIRST:** per-component loss logging (trainer-core change,
   scheduled daytime) is a PREREQUISITE for interpreting any new sweep — no
   more single-fused-loss blindness. λ-schedule remains an open question.
6. **Purpose split:** aux-shaping head (drop at deploy — optimize for trunk
   pressure) vs deployed ordering head (design/19 low-rank — optimize for
   inference cost). Different optima; never conflate again.

## 3. Options

**A. "Clean marginals+" (cheap rebuild, ~1 day):** per-half softmaxes +
ft-attach + legal-move masking + one-hot targets + component logging. Fixes
every KNOWN defect of the killed config with minimal new machinery. λ pair
re-derived after masking (CE scale changes: ln(~30 legal) not ln(128)).
*Proves:* whether clean policy pressure helps the trunk at all.

**B. "Binding" (+half day on A):** add conditional P(to | trunk, from) via
teacher-forced concat. *Proves:* whether from↔to binding adds shaping value.

**C. "Soft-target policy" (the moat, ~2-3 days):** A(+B) trained on top-k
soft targets from Tier-R arm data. Needs a target side-channel (the 2 spare
ChessBoard bytes can't carry a distribution): an aligned auxiliary target
file + custom loader, or a migration to a wider record format for the policy
corpus. *Proves:* whether search-informed targets beat best-move imitation —
the first real step toward the D41 thesis.

**D. "Search-informed multihead" (the D41 vision pulled forward, ~1-2 weeks):**
value + WDL + soft policy + margin/cost heads co-trained on Tier-R data with
the full instrument stack. The user's stated north star. High machinery; only
sensible after A/C validate the pipeline shape.

## 4. Recommended plan

1. **Now:** WDL sweep finishes on GPU (unaffected); policy work goes dark
   until instruments + design land. No more handicapped runs.
2. **Daytime task 1:** per-component loss logging (trainer core, tested).
3. **Daytime task 2:** Option A implementation + smokes (graph builds, legal
   masking throughput measured, engagement checkpoint-diff).
4. **A λ-pair on GPU** (~5h) gated + divergence-probed → first CLEAN policy
   readout.
5. **In parallel on CPU:** Option C's target pipeline (export arm
   distributions, aligned-target loader design).
6. **Then:** C runs; B folded into C's config; D scoped from C's results.
7. Every step under the existing validity machinery (scale-check gates,
   divergence floor, traps register, runbook interpretation rules).

## 5. Decision points (user)

1. Scope confirmation: keep value I/O as-is (recommended) or include it in
   the rethink?
2. WDL sweep: let it finish (recommended — clean head, produces the sidecar
   for the TM consumer) or full stop?
3. Sequence: A→C→D (recommended) or jump straight to C / D?
4. GPU budget for the A pair before C lands.

## RESOLUTION (2026-07-22 ~03:30) — expert spec adopted; options collapse

The user supplied the definitive spec: **complete categorical moves over the
compact 1,858-class vocabulary (1792 queen/knight-geometry from→to + 66
underpromotions), policy branch off the concatenated accumulators with an
H≈128-256 projection, per-move weight rows `score(m) = W[m]·h + b[m]`,
legal-rows-only evaluation, uint16 move_id labels.** This is design/19 §4.4's
low-rank form concretized with the Lc0 vocabulary. All factored from/to
encodings (joint-128, per-half marginals, conditional P(to|from)) are DEAD.
Option A/B collapse; C becomes "soft targets over the 1858 vocab"; D unchanged.

**Adopted ARCHON deviation:** IDs are computed in the trunk-normalized frame
(rank-flip + HalfKAv2_hm file-mirror) — Lc0 rank-flips only, but our
accumulators mirror by king file and unmirrored IDs against mirrored features
re-create the exp-0121 orientation tax. Documented in-module; symmetric under
the vocabulary.

**DONE tonight:** `src/eval/nnue/policy_vocab.rs` (c735588) — the single
shared implementation (engine crate, trainer imports it like halfka_index),
5/5 tests incl. every-legal-move-maps through the frame function on tactical
fixtures. Clippy-clean.

**REMAINING (daytime, in order):**
1. Loader: pack oriented from/to + promo code (extra[2] 2 bits) — the loader
   crate stays chessengine-free; the trainer example (which has the dep)
   computes move_id from the packed bytes via policy_vocab.
2. Trainer head: policy_projection (hl1→H, CReLU) + dense 1858 affine;
   additive legal-mask input (VRAM to measure: ~122MB/tensor at batch 16384 —
   halve batch if tight); softmax-CE vs one-hot move_id target; λ re-derived
   for the masked CE scale (ln(~30) ≈ 3.4).
3. Legal-mask generation in the mapper (Board reconstruction from the stored
   stm-canonicalized ChessBoard; movegen ~2-4µs/pos; the every-target-is-legal
   sampled assert = the frame tripwire).
4. Per-component loss logging (trainer core) BEFORE any λ sweep.
5. Smokes: build/engage/assert; then the λ pair on GPU.
6. Deployed ordering head (later): selected-row int8 inference per the spec —
   design/19 §4.4 unchanged.
