# Milestones — curated reproducible strong points

The versions worth running back to (not every commit — the ones that mark a
strength step or a capability). Each = git SHA + net + anchored strength + how
to reproduce. Reproduce a build with `tools/ledger/ledger.py reproduce <id>`.

| Date | Tag | git SHA | Net | Bench (d6) | Anchored strength | Note |
|---|---|---|---|---|---|---|
| 2026-07-10 | M4 / P6 (TT) | `~P6` | — (PSQT) | — | SF16 UCI_Elo 2000–2200 band | TT landed; WAC 154/200 @100ms |
| 2026-07-11 | P9 pruning | `80dc33d` | — (PSQT) | 350190 (d3 era) | +208 Elo vs pre-wave-C | all pruning waves A–G |
| 2026-07-11 | P11 SMP | `bbbb423` | — (PSQT) | 548978 | ~2720 single-core; +279 Elo @4t | **classical engine complete** |
| 2026-07-12 | net-0 (first NNUE) | `fedd159` | `net-0` (sha256:60138d33…) | 548978 (PSQT default) | *pending step-4 SPRT vs PSQT* | `768×2×256×1` SCReLU, SF-distill |

## How to read strength
- **Anchored strength** = gauntlet vs SF16 `UCI_Elo` rungs (approximate,
  non-linear anchor) — see `docs/strength-log.md` and `experiments.jsonl`
  (`exp-0003`).
- **Bench (d6)** = the D24 depth-6 fingerprint (`bench 16 1 6 default depth`);
  reproducible only with that commit's era's depth (pre-P6 trailers were d3).
- Nets are content-addressed in `nets.jsonl`; the `.cenn` bytes are gitignored
  (fetch by hash from the store).

## Pending
- **net-0 vs PSQT** — the P13 "it works" gate (step 4). When it lands, add the
  `exp-` row + fill net-0's `elo_vs_parent` and this table's strength cell.
- The net series (net-1, net-2, …) each gets a row once SPRT-gated vs its parent.
