# ARCHON perf study #32 — halving the HalfKA feature-transformer row to i8

Date: 2026-07-30. Instruments: `tools/perf32/`. Target named by study 31 §4.3 as
the largest single unclaimed item in the engine: **−463 cycles/node, ≈ +17 Elo**,
because 52% of our load-stall cycles are FT weight rows and 61% of those are the
HalfKA table.

---

## 0. THE PASS BAR, PRE-REGISTERED

*Written and committed before the first eval delta was measured.*

The change is a pure-speed change. It does not improve the eval; the eval can
only get worse. So the question is whether the fidelity loss is small compared
to the +17 Elo the speed is worth, and the bar has to be set in the units the
engine actually consumes — the final NNUE score, in Stockfish's printed-cp
frame (ours is 2.07× that; `uciprobe.to_sf_frame`).

**PASS requires all five, on the 800-FEN `scale-check-800` battery:**

| statistic | bar |
|---|---|
| mean signed Δ | within ±0.5 SF cp |
| median \|Δ\| | ≤ 1.0 SF cp |
| p95 \|Δ\| | ≤ 3.0 SF cp |
| p99 \|Δ\| | ≤ 5.0 SF cp |
| Spearman ρ vs the reference net | ≥ 0.9990 |

**Why those numbers, and not tighter or looser.** Three anchors, all from this
project's own record:

1. **Perf study 8 §4.6 shipped a deliberate eval approximation at a measured
   mean 0.14 SF cp** — the threat channel's 255/64 frame ratio carried as 4, a
   0.39% gain change — and re-pinned the NNUE gate for it. That is the
   precedent for "an accepted eval perturbation".
2. **The threat channel already carries 5.58% relative weight quantisation
   error** and 86% of all accumulator quantisation noise
   (`tools/nnue/requant_threats.py`, 2026-07-29), against the 2.0–2.8% the
   schemes below cost on HalfKA. We are proposing to add *less* noise than the
   engine already runs on.
3. **Games arbitrate.** The offline bar is a screen, not the verdict. Anything
   that passes goes to a ≥1466-game fixed-node screen (±10 Elo), which is the
   number that decides whether it lands.

A scheme that fails the bar is not the end of the study: the fallback ladder is
per-column scales → clipped per-column scales → mixed precision → asymmetric
zero-point → a short QAT fine-tune. Each is a real deliverable on its own.
