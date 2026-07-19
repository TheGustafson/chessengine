# N-gate (t-shirt S/M/L) adaptive eval — faithful research plan (2026-07-19)

User-directed: "let current net size be medium; we test t-shirt sizes" + "begin a
faithful auto research loop … faithful attention to get it to work well."

## The idea
Route eval compute by position **difficulty**, spending cheap evals on easy nodes
and expensive-accurate evals where it matters:

| tier | net | routed when | cost |
|---|---|---|---|
| **SMALL** | HalfKA-only (no FullThreats) | DECIDED — `\|material\| ≥ mat_gate` **and** `\|small_eval\| ≥ thresh` | ~0.5–0.6× medium |
| **MEDIUM** | current D40 (HalfKA + FullThreats) | default | 1× |
| **LARGE** | full-feature, more accurate | CRITICAL — medium net's σ̂ ≥ `sigma_gate` | ≥ 2× medium |

**Why σ̂ routes the LARGE tier (the synthesis with EXP-UNC-1):** `\|eval\|` can't
distinguish a dead-drawn ~0.0 from a razor-sharp ~0.0; the **uncertainty head σ̂
can**. High σ̂ = "the medium net is unsure here" = exactly where a more-accurate
(expensive) net earns its cost. So EXP-UNC-1's variance head becomes the tier
router — arguably a more natural consumer of σ̂ than the RFP/TM scaling.

## Infra (SHIPPED — commits 62102ca, f165749, 37336d7)
`SotaContext` carries the medium stacks + optional SMALL (HalfKA) + optional LARGE
(HalfKA+threats) stacks, all lazy + in lockstep. Env: `ARCHON_SMALLNET` /
`_THRESH` / `_MAT_GATE`; `ARCHON_LARGENET` / `_SIGMA` (+ `D40_UNC_HEAD` sidecar so
the medium net produces σ̂). Default off ⇒ **bit-identical** (sota 469205, default
depth-6 701605). Verified: small routes decided, large escalates on σ̂, both
transparent when the tier net equals the medium.

## The FAITHFUL caveat (do not oversell)
A tier only helps if its net has the right cost/accuracy for its job:
- **SMALL is a real lever now:** the cheap net is genuinely cheaper (drops the
  46 MB threat channel), and speed converts to equal-time Elo on the D40 arch
  (opt#1 = +28.8 for +14% NPS). This is the immediately-testable win.
- **LARGE only helps if the large net is genuinely MORE ACCURATE than the
  medium** — and our one-box results say width is saturated (L1-2048 null) and
  eval is data-limited (`sf16-gap-diagnosis`, `training-methods-study`). So a
  same-data wider net will NOT help. The honest large-tier candidates are:
  (a) an **ensemble** of D40 nets (variance reduction; limited by correlated
  errors but buildable now), (b) a genuinely **wider** net (needs a generic-L1
  runtime refactor — the accumulator/head are 768-typed; deferred), or (c) a
  **beyond-SF16 fleet net** (the real payoff — the gate is its deployment
  vehicle). The loop tests (a) faithfully and reports; (b)/(c) are follow-ons.

## The auto research loop (faithful protocol)
Baseline is always **uniform-medium** (all tiers off) — bit-identical eval, so a
positive result is the routing, not a net swap. Candidate = medium + armed tier.
fastchess has no per-engine env ⇒ a wrapper arms the candidate.

- **Attribution (fixed-node, load-immune — runs on a busy box):** node count +
  NPS per cell → does the gate actually change eval cost / tree shape?
- **Ship metric (equal-time 10+0.1 — needs a QUIET box):** Elo vs uniform-medium.
  The loop SIGSTOPs the relabel for these arms and SIGCONTs after, so the timing
  is clean.
- **Phase 1 — SMALL sweep:** `thresh ∈ {300,500,700}` × `mat_gate ∈ {0,300,500}`.
  Success = any cell ≥ +8 Elo equal-time (2σ) vs uniform. Refine around the best.
- **Phase 2 — LARGE sweep:** requires a more-accurate large net (ensemble first).
  `sigma_gate` swept over the σ̂ distribution's upper quantiles (route the top
  ~5–20% most-uncertain nodes). Success = ≥ +8 Elo equal-time. Honest null =
  "one-box has no more-accurate net; large tier waits for the fleet."
- **Journal** every cell (params, NPS, node count, Elo±err) to
  `data/nnue/experiments/n-gate/journal.md`; ledger the winners (D38).

Abort discipline: if the SMALL sweep is flat at equal time across the grid AND
NPS clearly rose, the cheap net is too weak (its decided-position evals disagree
with the medium) → retrain the cheap net longer / distilled from the medium, do
not tune thresholds blindly.
