# EXP-UNC-1 consumer design (Gate-2 wiring) — 2026-07-19

Design for the search-side consumers of the value-uncertainty column σ̂
(`EvalContext::last_sigma()` → `SotaContext::last_sigma()`). Written while the
Integral match holds the box (no builds); implement + build + test after Gate 1
passes and the variance-bearing net exists.

## σ̄ — running EMA, not a calibrated constant (the key call)

The study formula `margin *= clamp(1 + c1·(σ̂−σ̄)/σ̄, 0.7, 1.5)` needs a reference
σ̄. A **calibrated constant** would need an offline pass over the variance net
(which doesn't exist until the Gate-2 retrain) and would drift with the position
distribution. Instead maintain a **per-worker running EMA**:

```
σ̄ ← σ̄ + (σ̂ − σ̄) >> EMA_SHIFT     // EMA_SHIFT ≈ 10 (α ≈ 1/1024), f32
```

updated at every node that produces a σ̂ (fresh SOTA eval). Seed σ̄ lazily to the
first σ̂ (so the ratio is 0 on the first node → no effect until it warms). This
is self-calibrating: the moment the variance net loads, the consumers work with
no separate calibration step, and σ̄ tracks the game/phase distribution. The
relative signal `(σ̂−σ̄)/σ̄` is dimensionless, so σ̂ living in sigmoid (WDL) space
is fine.

State lives on the `Worker`/`Stack` (per-thread), reset at `refresh_root`. `None`
σ̂ (variance-less net, or TT-cached eval that skipped the forward) → the consumer
uses its unscaled incumbent value → **bit-identical to today when off**.

## Consumer 1 — RFP / futility margin scale (D6)

Site: `rfp_prune_margin` result at its call site (search/mod.rs ~214 + the N10
prune), and `futility_base = static_eval + 335`. Scale the margin:

```
factor = clamp(1.0 + UNC_RFP_C1 · (σ̂ − σ̄)/σ̄, 0.7, 1.5)   // f32, σ̂ present
margin = (margin as f32 · factor) as Value
```

- σ̂ HIGH (unreliable eval) → factor > 1 → LARGER margin → prune LESS (safety).
- σ̂ LOW (reliable eval)   → factor < 1 → SMALLER margin → prune MORE (the Elo,
  the tree shrinks — study §3.2.3: "trust the eval more where reliable").
- `UNC_RFP_C1` env-gated (`ARCHON_UNC_RFP_C1`), **default 0.0 → no change**. The
  SPSA'd sign is itself diagnostic of where the headroom is.

Keep the scaling in a pure helper so a unit test pins its sign/magnitude (mirror
`rfp_threat_coupling_*`).

## Consumer 2 — TM soft-stop scale (D1)

Site: the ID-loop soft stop (uci.rs / search_control.rs). ARCHON's TM has ZERO
position content today (pure SF telemetry transplant), so root σ̂ is complementary
info, not a re-weight. At the ROOT only, using the root position's σ̂:

```
total_time ·= clamp(1.0 + UNC_TM_C2 · (σ̂_root − σ̄)/σ̄, 0.8, 1.25)
```

- σ̂_root HIGH (sharp/unclear root) → think LONGER.
- `UNC_TM_C2` env-gated (`ARCHON_UNC_TM_C2`), **default 0.0 → no change**.
- Use the σ̂ from the root fresh eval (always computed at ply 0); do not let a
  child σ̂ leak in.

## Gate 2 protocol (study §3.1.1)

Twin net (variance head) with consumers ON vs the same twin with consumers OFF:
**600g fixed-node + 300g 10+0.1**, a 2×2 (c1,c2) probe. Kill if no positive
config. σ̄-EMA warmup makes early plies neutral, so no calibration run needed.

## STATUS 2026-07-19: BOTH CONSUMERS BUILT + Integral milestone

- Consumer 1 (RFP margin) committed 20897a7; consumer 2 (TM soft-stop) committed
  74a5104. Both cfg(sota)+env-gated, bit-identical off (sota bench 147691), unit-
  tested, 639 sota tests + clippy clean. `EvalContext::last_sigma` = ebe4f80.
- The EXP-UNC-1 *code* is COMPLETE; the rest is gated on Gate 1's verdict.
- Integral: ARCHON +24.4±32.2 (exp-0104), from −70. logvar fix build OK.

## Post-Gate-1 execution checklist

1. Report + ledger the Integral result (DONE). logvar-verify auto: post-twin
   smoke confirms l3_varw/l3_varb are in the raw ckpt (the fix works).
2. Twin finishes → Gate-1 harness auto-fires (value parity vs control, kill ≤−15).
3. If Gate 1 PASSES: **Gate-2 RETRAIN** — rerun `train_run.py --config
   data/nnue/experiments/d40-uho-unc-v1/config.json` with the logvar-fixed
   checkout trainer (already staged/built) so l3_var is saved (same seed →
   identical mean → Gate 1 carries over). Then:
   `extract_unc_head.py <retrain final-raw.bin> data/nnue/nets/d40-uho-unc-240sb.sidecar --validate data/nnue/nets/d40-uho-unc-240sb.cenn`.
4. **Gate 2** — the variance net (CENN) + `D40_UNC_HEAD=<sidecar>`, a 2×2 probe
   over the two env coeffs, each vs the same net with consumers OFF, at fixed-node
   AND 10+0.1:
   - `(0,0)` baseline (consumers off) ; `ARCHON_UNC_RFP_C1=<c1>` ;
     `ARCHON_UNC_TM_C2=<c2>` ; both. Seed c1≈0.3, c2≈0.3, then refine.
   Kill if no config is positive. If Gate 1 FAILS: kill EXP-UNC-1 (consumers stay
   dormant — reusable for EXP-AUX-1 / any future uncertainty signal).
