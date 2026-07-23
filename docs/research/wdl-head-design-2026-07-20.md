# WDL output head — precise implementation state + design (2026-07-20, restated)

Goal: give the net an extra output (win/draw/loss probabilities) that SEARCH and
TIME-MANAGEMENT consume. This doc states EXACTLY what already exists in code, what
does not, and how the pieces connect — so nobody re-derives or re-builds what's
there.

## TL;DR
A full "extra-output head + sidecar + env-gated search/TM consumers" pipeline is
**already implemented and wired** (from the EXP-UNC-1 attempt), but it is (a)
shaped for ONE scalar (an uncertainty value σ̂), not 3-way WDL, and (b) **completely
dormant** — no trained net ever produced a useful extra output, and every consumer
defaults OFF (coefficient 0 → bit-identical to today). So this is ~90% plumbing-
reuse, not a from-scratch build. What's genuinely missing is small and specific.

---

## PART A — what is ALREADY IMPLEMENTED (and default-inert)

All of this exists on main today, all env-gated, all bit-identical when the env
coefficients are unset (the default).

### A1. Engine: an optional extra per-bucket output head
`src/eval/nnue/sota_head.rs`:
- `l3_var_weights/l3_var_biases: Option<Box<[f32]>>` (line 84-86) — an OPTIONAL
  second head, `L3 → BUCKETS` (one extra scalar per output bucket). `None` ⇒ the
  net has no extra output ("variance-less"), and every consumer no-ops.
- `propagate_l3_var(l2, bucket)` (line 225) — computes that scalar off the SAME
  shared `l2` activation the eval head already produced (~tens of FLOPs, T≈1).
- `forward()` returns `(value, Option<σ̂>)`.

### A2. Engine: sidecar loader (keeps the head OUT of the shipped net)
`src/eval/nnue/sota_loader.rs:431`: `load_uncertainty_sidecar` reads the extra
head's weights from a file named by env `D40_UNC_HEAD` (flat LE f32). It is NOT in
the CENN format → the default build with no sidecar is byte-identical to today.
Called once at net-load.

### A3. Engine: the accessor
`src/eval/mod.rs:140`: `EvalContext::last_sigma() -> Option<f32>` → the extra
scalar for the last-evaluated position (or `None`).

### A4. Search/TM: the σ̄ EMA + the two consumers — WIRED, not just defined
`src/search/mod.rs`:
- Per-worker running mean `sigma_bar`, updated at each node that produces a fresh
  σ̂ (line ~2224-2228, EMA α=2⁻¹⁰, env `ARCHON_UNC_EMA_SHIFT`).
- **Consumer 1 (RFP/futility margins):** `unc_margin_scale(σ̂, σ̄, c1)` (line 278,
  called ~2541) → `margin *= clamp(1 + c1·(σ̂−σ̄)/σ̄, 0.7, 1.5)`. High σ̂ ⇒ bigger
  margin ⇒ prune less; low σ̂ ⇒ prune more. Coeff env `ARCHON_UNC_RFP_C1`, default 0.
- **Consumer 2 (TM soft-stop):** `unc_tm_factor(root_σ̂, σ̄, c2)` (line 294, called
  ~1635) → think longer at a sharp/uncertain root. Coeff env `ARCHON_UNC_TM_C2`,
  default 0.

### A5. Engine: the optimism hook (SF-style) already exists
`src/eval/mod.rs:260`: `evaluate(board, ctx, optimism, ply)` takes an `optimism`
value (currently computed by SEARCH at aspiration-window setup) and has a
complexity/optimism blend (inert while complexity=0). A net-derived W-prob could
feed this hook — but nothing does today.

### A6. Trainer: an extra head with the FROZEN-TRUNK recipe (the correct one)
`tools/nnue/bullet/d40_features.rs:713-738`: behind `D40_HETERO_NLL`, a parallel
`l3_var` head (`L3 → NUM_BUCKETS`) exposed as a SAVED output (so it lands in the
raw checkpoint). The intended-safe path freezes the trunk via `D40_INIT_RAW` (load
the finished eval, then ONLY l3_var trains — "freezing isolates it perfectly").
Plus `tools/nnue/extract_unc_head.py` (sidecar extractor from the raw ckpt).

---

## PART B — how it works end-to-end (the data flow)
```
TRAIN:  finished eval net ──freeze trunk (D40_INIT_RAW)──▶ train ONLY the extra head
                                                              │ raw ckpt (has l3_var)
        extract_unc_head.py ◀─────────────────────────────────┘
              │ sidecar file (flat f32)
LOAD:   engine net-load ──D40_UNC_HEAD=<sidecar>──▶ head gets l3_var weights
RUN:    sota forward ──▶ (eval, extra) ──▶ EvalContext::last_sigma()
              │
        search: update σ̄ EMA ──▶ unc_margin_scale (RFP/futility) + unc_tm_factor (TM)
              │  (only if ARCHON_UNC_RFP_C1 / _TM_C2 != 0)
        SPSA-tune c1,c2 ──▶ fixed-node gate ──▶ equal-time gate
```
Default (no sidecar, coefficients 0): the extra head is `None`, `last_sigma()` is
`None`, consumers return 1.0 → **bit-identical to shipping today**.

---

## PART C — what is NOT implemented (what building the WDL head actually costs)

The scaffolding above is for ONE uncertainty scalar. To make it a real **WDL**
output the deltas are:

1. **Widen the head to 3 outputs.** Today `l3_var` is `L3 → BUCKETS` (1/bucket).
   WDL needs `L3 → BUCKETS×3` (W/D/L logits per bucket) + softmax. Engine head +
   sidecar layout + extractor all need the ×3. (Small, mechanical.)
2. **Change the training loss.** Today it's variance NLL. WDL is **softmax
   cross-entropy on the real game result** (0/0.5/1 → L/D/W). We already have the
   real WDL per row (the WDL-join in the fulldist corpus). Reuse the frozen-trunk
   recipe (A6) — do NOT co-train (see Part D).
3. **Map WDL → the consumer scalars.** The consumers eat a scalar. Derive it from
   the WDL head: e.g. decisiveness `|W−L|` or draw-prob `D` feeds the *existing*
   `unc_margin_scale`/`unc_tm_factor` unchanged. (Optionally add an **optimism
   consumer** feeding `evaluate(optimism)` from W-prob — that IS new code.)
4. **THE ACTUAL MISSING PIECE: a trained net whose extra head is useful.** The σ̂
   attempt never produced one — the co-trained NLL net was catastrophic (Part D),
   and a validated frozen σ̂ net was never gated. So today there is **zero deployed
   benefit**; the pipeline is dormant. A WDL head needs its own train + parity +
   SPSA-tuned c1/c2 + gate before it does anything.

---

## PART D — co-train vs frozen: BOTH untested, test both (do NOT over-read EXP-UNC-1)
Co-training is NOT proven bad. What EXP-UNC-1 showed is narrow and confounded:
- It used a **heteroscedastic-NLL** variance head, loss `0.5·(exp(−s)·(mean−t)²+s)`.
  The `exp(−s)` term **multiplicatively reweights the MEAN's own gradient** per
  example → a bad early variance estimate distorts what the mean learns. That's a
  property of *that loss*, not of co-training. Result −179 (exp-0105).
- The "detach" fallback also failed (−116, exp-0106) — but on a **bullet tooling
  bug** (`with_no_grad` didn't isolate the gradient). A confound, not a law.

Evidence the OTHER way (co-training can HELP the eval): our own policy pilot
λ=0.01 deployed eval-only = **+63** (PAS:408-411); AlphaGo Zero (co-train
policy+value REDUCED value error); KataGo (aux co-training improved the value net).
Multi-task learning's whole point is the main head getting better — freezing
forgoes that upside.

Why WDL is a different case from EXP-UNC-1: **softmax-CE is ADDITIVE** — combined
loss `eval_MSE + λ·WDL_CE`, gradients ADD; the WDL head does NOT reweight the eval
gradient the way NLL's `exp(−s)` did. At small λ it's a gentle regularizer; at λ=1
it dominates (the known policy-λ=1 failure) — a **tunable scale** problem, not
structural mean-distortion. So the one failure we have likely does not apply here.

**Two options, both untested — TEST BOTH, assume neither:**
1. **Co-trained WDL head (tuned λ, additive CE)** — potentially the HIGHER-value
   path: may improve the eval itself, not just add an output (AGZ/KataGo/our +63).
   Risk (λ-fragility/interference) is real but measurable: λ sweep {3e-3,1e-2,3e-2},
   log per-step policy/value gradient-norm ratio, λ→0 limit contains the control.
2. **Frozen-trunk head** (A6) — conservative: zero eval risk, but forgoes any trunk
   improvement. Use as fallback, or when only the consumable output is wanted.
Gate each vs the value-only control; the co-trained one is the more interesting
experiment, not a thing to skip because a *different-loss* head once failed.

---

## PART E — honest value + sequencing
- **Value: single-digits to low-double-digits Elo**, mostly via **TM** (ARCHON's TM
  has ZERO position content today — a decisiveness/draw signal is complementary,
  not redundant). Optimism/margin scaling are secondary. This does NOT close the
  −129 vs SF16 gap — that's eval quality (hard-negatives + a stronger net).
- **Note we already use WDL as a training TARGET** (the 0.3 blend, ~+19 Elo). This
  head is the separate "WDL as a consumed OUTPUT" idea.
- **Sequencing:** queued behind the eval foundation (fulldist → Phase-D net). The
  frozen-trunk head is low-risk (can't hurt the eval), so it can slot in as the
  first D41 step once a good eval net exists — reusing ~all of Part A.
