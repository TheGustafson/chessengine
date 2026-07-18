# Flywheel v1 — first self-improvement cycle spec (draft 2026-07-18 00:45)

Goal: one complete data→train→gate cycle for the D41 helm program on the
D40-class architecture (D45a), using only proven components. This spec is
the checklist; the live log stays `value-foundation-state.md`.

## Why now

- Helm v1/v1.1 established the bottleneck is DATA VOLUME (exp-0086: null at
  336k rows; exp-0092: −29.6±12.9 despite clearing every offline bar). The
  one confirmed positive (gen-0, +68.6 fixed-node) sat at ~100M rows.
- The harvest machinery is now measured: gold-v1 parallel labeling =
  **11–12k positions/hour/teacher** (12.5–13.9× serial, 2026-07-18);
  in-search capture v0 merged (2acc736, ~2% overhead, node-identical).
- The speed program (R1 native + R3 PGO, this night) multiplies every
  games/hour and labels/hour number below.

## Cycle definition (one turn of the wheel)

1. **Harvest A — root-analysis gold labeling** (teacher-quality targets):
   `label_gold_v1.sh <N> 4` on fresh FEN pools (UHO-derived, dedup vs all
   prior pools by canonical key). Budget: overnight run ≈ 100k
   positions/teacher ≈ ~3.4M rows (34 arms/position avg).
2. **Harvest B — in-search capture** (self-observation targets): fixed-node
   matches/datagen with ObsCapturePath set (NEVER in TC fairness games).
   Every screening match we run anyway becomes training data. Yield to be
   measured in cycle 1 (knobs: MinDepth 6, SampleBits 4, PerGameCap 20000).
3. **Train**: d41_alloc11 recipe (frozen 720sb trunk, pairwise RankNet,
   `export-train --all-tiers`, early stopping). One arch change queued
   behind data scale: king-bucket factorization (top pending training exp).
4. **Offline bars** (necessary, NOT sufficient — proven twice): held-out
   rank corr / top-1 vs the 18.1% cold-classical baseline, position-id
   split.
5. **ONE play test** (no knob sweeps pre-scale): 600g fixed-node vs plain
   build, net-0 eval both sides, then equal-time only if fixed-node is
   clearly positive. Ledger row regardless of outcome.

## Volume targets (from the exp-0086/0092 vs gen-0 bracket)

| rung | rows | source mix | verdict gate |
|---|---|---|---|
| done | 336k | big-v1 | play-negative (exp-0092) |
| cycle 1 | ~4M | big-v1 + gold-v1 + one overnight gold run + capture | play test may stay negative — the point is the slope |
| cycle 2 | ~15–40M | + capture harvests from all screens + 2nd gold week | offline slope decides whether 100M is needed |
| gen-0 ref | ~100M | (SF-distilled policy era) | the known-positive scale |

## Hard rules carried forward

- 5k-hard-limit search labels stay quarantined from scalar losses; capture
  records are ORDERING/allocation targets, not eval targets.
- Every corpus/net/match → D38 ledger row; every FEN pool dedup'd and
  seeded reproducibly.
- Fixed-node harvests coexist (T4 load-immunity); TC gates get a quiet box.
- Offline bars never justify a ship claim — only game tests do.

## Open items before cycle 1 starts

- [ ] Speed matrix (v3/native/PGO × D40/classic) — running tonight.
- [ ] Capture yield measurement: rows/game at the standard screen config
  (first fixed-node screen with ObsCapturePath set, then obs-tool export).
- [ ] FEN pool #2: 100k fresh openings-derived positions, dedup vs
  obs-gold-v1-fens + big-v1 pools (seed 20260800).
- [ ] Decide harvest-B blend weighting (teacher rows vs capture rows) —
  start uniform, measure.
