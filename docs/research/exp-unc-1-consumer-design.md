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

## Post-match / post-Gate-1 execution checklist

1. Resume relabel (match trap does it); report + ledger the Integral result.
2. Build-verify the logvar fix: copy tracked trainer → checkout, build, 2-SB
   smoke with D40_HETERO_NLL=1, confirm the raw ckpt now has l3_varw/l3_varb
   (`extract_unc_head.py <raw> <side> --validate <cenn>` passes the transpose).
3. Train finishes → Gate-1 harness auto-fires (value parity vs control).
4. If Gate 1 PASSES: Gate-2 retrain (same config + the logvar-save fix) → extract
   sidecar → wire consumers 1+2 (this doc) + `EvalContext::last_sigma` (already
   written) → build/clippy/test → Gate 2. If Gate 1 FAILS: kill EXP-UNC-1.
