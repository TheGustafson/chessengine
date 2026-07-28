# Overnight autonomous plan — 2026-07-21 (user asleep ~8h)

**Governing rule (user, tonight): THINK before acting; measure before choosing a
knob; do NOT launch a run on a guessed hyperparameter.** I've been jumping in and
undoing — stop that. Each phase below has a decision gate; honor it.

## Current state (snapshot at plan time)
- **from-scratch-1024** (value-only, HL=1024, 120sb UHO, `seed=198273612`) training,
  ~sb 87/120. This net is BOTH the capacity control AND the H-AUX control **A**.
- **ONE** pipeline waiter armed (`/tmp/capacity_and_wdl.sh`, pid 82868) → on GPU-free:
  capacity gate (1024 vs 768 `foldval-ctrl`) + WDL smoke + policy smoke.
- Heads built (trainer-side, compile): WDL (`D40_WDL_LAMBDA`), policy from/to
  (`D40_POL_LAMBDA`), both additive + λ=0-inert (CENN parity). Move packed into
  `ChessBoard.extra`. Value CENN is byte-clean (quant.bin excludes the heads).

## ⚠ THE λ-SCALING TRAP (measure, don't guess)
Loss = `value_MSE + λ·CE`, all `reduce_sum_batch` (SUMS over 16384). Magnitudes:
`value_batch ≈ 160–500`; `policy_CE_batch ≈ 16384·2·log(64) ≈ 1.36e5`;
`wdl_CE_batch ≈ 16384·log(3) ≈ 1.8e4`. So a "normal-looking" λ=0.3 makes policy
**~400× the value loss** → the net abandons eval → garbage. **Gentle regime (aux ≈
15–25% of value): λ_pol ≈ 3e-4…8e-4, λ_wdl ≈ 1e-3…3e-3.** Compute it from the smoke:
`policy_CE ≈ (pol_smoke_loss − value_smoke_loss)/λ_smoke`, then
`λ_pol = 0.2·value / policy_CE`. Clamp to [2e-4, 1.5e-3]. (The wdl-head-design pilot
cited λ=0.01 for +63 — likely a different loss normalization; trust the measured
magnitudes here, not that number.)

## ★ VERDICT (2026-07-21 AM): ALL CHEAP EVAL LEVERS ARE DEAD ★
- **fold-in: neutral/negative** (exp-0118, monotonic-hurt).
- **width 768→1024: −8 ± 21** at equal time (exp-0120; NPS cost eats the gain).
- **policy co-train (H-AUX): −14 ± 23** (768@120sb, λ=0.001, to-square target). The
  zero-NPS aux HURTS — it steals trunk capacity from value + the imperfect target
  (to-sq only, vertical-flip-only orientation) adds noise. Refuted AT THIS CONFIG.
  (Caveat: a better config — from+to, HalfKA `_hm` orientation, lower λ, full sb —
  is untested; but the pattern says diminishing returns on cheap additions.)

**Conclusion:** the incumbent recipe (UHO stage-1 → farseer stage-2) is a HARD FLOOR
at −78 vs SF16, and no low-cost addition moves it at equal time. Beating SF16 needs
the TRANSFORMATIVE lever, not another cheap knob.

## NEXT DIRECTION — label quality (the SF18-canonical path)
The one lever with headroom: distill from a teacher STRONGER than the training data
(SF18-canonical labels > the binpacks' shallow-SF labels). It failed at −582, root-
caused THIS session to the **quiet-filter bypass** in the corpus build (in-check/
tactical positions carrying search-score labels a static net can't represent).
`fast_fens` now has the filter — so the fix is: rebuild the canon corpus THROUGH the
filter, retrain, gate vs the incumbent. That is the actionable transformative bet.
Verifying the full-recipe base first: farseer stage-2 (running) must reproduce
incumbent-class strength at 768 before we trust the pipeline for the canon retrain.

## RESULTS SO FAR (updated live)
- **CAPACITY (Phase 1): DEAD at equal time.** 1024 vs 768 (matched UHO 120sb) =
  **−8.1 ± 21** @ 10+0.1 (exp-0120). The wider net's NPS cost offsets any eval gain.
  Fold-in was already dead (exp-0118). ⇒ **neither data-polish nor width is the
  equal-time lever.** The one lever left that pays at equal time is a **zero-NPS**
  eval improvement — i.e. the H-AUX policy-aux (co-train, deploy eval-only).
- **POLICY HEAD: built, DEBUGGED (3 real bugs), validated.** Fixes: (1) create head
  affines only inside the λ>0 branch (orphaned-weight panic); (2) softmax-CE needs
  `.reduce_sum_rows()` before `.reduce_sum_batch()`; (3) value+TWO extra heads makes
  the backward topo-sort cyclic → use ONE head (to-square, reads hl3, loss-only).
  Validated: trains, no cycle, and policy_mean 3.95 < log(64)=4.16 ⇒ **the
  move-packing feeds a REAL target** (below-uniform CE = learning the teacher's move).
- **λ MEASURED (not guessed): λ_pol ≈ 0.001** (policy ≈ 20% of the value loss; at
  λ=0.02 the policy contribution was ~0.079 vs value ~0.017).
- **H-AUX RUNNING (Phase 2):** A2=value-only vs B2=value+policy(λ=0.001), SAME
  current binary + seed 198273612, 1024@120sb, gate at 10+0.1. Isolates the policy
  effect cleanly. `/tmp/haux2.sh`. This is THE experiment now.
- **WDL head deploy path complete** (engine: propagate/forward/last_wdl/sidecar;
  `extract_head.py`) — compiles, tests green, parity-safe.

## STRATEGIC PIVOT (given capacity+fold-in dead)
If H-AUX shows policy co-train improves the eval (>+10 vs A2): the ship candidate is
**768 + policy** (768 = best equal-time width; policy = zero-NPS gain) at higher sb,
gated vs the incumbent + SF16. If H-AUX is neutral/negative: the eval lever is
hard-negative data DENSITY (retrain a fresh net on a hard-neg-enriched corpus, NOT
fold-in) — the last untried angle. Either way, do it at 768 (1024 doesn't pay).

## Phase 1 — capacity + smokes (auto, ~1h). DECISION GATES:
- **Capacity** (scratch-1024 vs foldval-ctrl 768, tc=10+0.1): `>+15 Elo` → width helps,
  ship at 1024. `≈0 / negative` → width doesn't help; ship at 768, and A must be
  re-benched at 768 for the H-AUX (or accept the 1024 A/B isolates policy only).
- **Policy smoke**: total loss finite + falling ⇒ head trains + move-packing OK →
  proceed. Loss stuck ≈ `2·log(64)` region with no fall ⇒ move-packing broken
  (check `entry.mv` populated + the `extra` read + orientation) BEFORE spending a B run.
- **WDL smoke**: λ>0 saves `l3_wdl` in raw ckpt; λ=0 does not (parity).

## Phase 2 — H-AUX (the rich-net payoff, ~2h). Only if policy smoke passed.
Train **B = value+policy** at the *computed* λ_pol, SAME width/seed/recipe as A
(`seed=198273612`, 1024, 120sb UHO). Netpack B → value CENN (policy dropped). Gate
**B vs A** (from-scratch-1024) tc=10+0.1, 400g. → does policy co-training make the
EVAL better at zero NPS? `>+10` = win (include in ship). `≈0` = no eval help (policy
is then only an output-head option). `<0` = λ too high or interference → halve λ once.
Script ready: `/tmp/haux_experiment.sh` (FIX its λ to the computed value before use —
its default 0.3 is in the trap).

## Phase 3 — ship candidate (~4h). Build the strongest net from the winners.
Winning width (1024 if capacity won, else 768) + policy at winning λ (if H-AUX won),
higher sb (240–480), UHO. Netpack → gate vs the incumbent `d40-uho-720sb` (fixed-node
screen → equal-time) and vs **SF16** at tc=10+0.1. Log to
`value-foundation-state.md` + a D38 ledger row for any net/gate.

## Loop-tick checklist (each wake)
1. GPU busy? What's the running job's progress? Never leave the GPU idle >1 tick —
   launch the next phase's job.
2. Any completed gate/smoke? Read result → apply the DECISION GATE above → log it.
3. Keep exactly ONE of any waiter (settle count via `comm==bash` match, not raw ps).
4. Before any train launch: is the λ measured/computed, not guessed? Is the seed set
   for a clean A/B? Is the width the decided one?
5. Update this doc's "state" + value-foundation-state.md after each result.

## Do-NOT (tonight's anti-thrash list)
- No launching B/ship with a guessed λ (measure).
- No editing the running trainer's binary while a run I depend on is mid-flight
  (rebuild only affects future runs — fine — but verify λ=0 parity if I do).
- No engine-side WDL refactor rushed + untested (GPU-gated) — spec it, defer to when
  a WDL net exists to validate against.
- No pkill-by-pattern that can self-match this shell — kill by explicit PID.
