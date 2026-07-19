# SF16 gap diagnosis — WHERE the −41.5 comes from (2026-07-19)

Data-driven autopsy of the equal-time (10+0.1) ARCHON-vs-SF16 match
`data/nnue/experiments/pgo-vs-sf16/games.pgn` (463 games, the exp-0103 −41.5±12.3
result). Every move carries `{eval/depth time}` from both engines, so we can
attribute the deficit to a phase / mechanism, not just count outcomes.
Scripts: scratchpad `sf16_loss_diag.py`, `sf16_depth_cmp.py`.

## Headline: the SF16 gap is EVAL QUALITY at ~equal depth — not speed, not TM.

**Depth at equal time (the decisive measurement):**
| phase | ARCHON median depth | SF16 median depth | gap |
|---|---|---|---|
| opening (≤ move 15) | 17 | 18 | +1 |
| middlegame (16–40) | 16 | 17 | +1 |
| endgame (> 40) | 15 | 17 | +2 |
| **overall** | **16** | **17** | **+1** |

Same mean time/move (0.26 s both). **SF16 out-searches ARCHON by only ~1 ply.**
A gap that persists at ~equal depth is an **evaluation** gap: SF16 judges each
node better than ARCHON's D40 net. (ARCHON is actually time-*efficient* — median
0.1 s/move to depth 16 vs SF16's 0.2 s to 17 — so it is not slow per depth.)

**Corroborating loss anatomy (160 losses):**
- **Not time trouble:** 0 time forfeits (350 adjudication / 113 normal). The
  decisive blunder move took a median **0.7 s at depth 18** — a *considered* move
  that was wrong, not a scramble. → not a TM or move-overhead problem.
- **Not eval-blind:** 0 losses where ARCHON's eval never flagged the trouble;
  only 5/160 where ARCHON stayed optimistic (>0) while already lost (median
  overoptimism −1.7). ARCHON *knows* when it is lost — it just gets there.
- Losses cluster early (decisive swing median fullmove 10, most material still
  on) — but see the book caveat below; this is largely the UHO book, not a
  distinct "early" weakness.

## Two things the data CANNOT tell us (methodology guards)

1. **Per-color split is a book artifact, not an ARCHON bias.** Raw split is 50%
   as White vs 38% as Black — but with an *unbalanced* book (UHO) this is
   mathematically forced and symmetric between engines: on a color-reversed pair,
   `SF16-Black score = 1 − ARCHON-White score` by construction, so the split
   measures the book's White-advantage, not any ARCHON color weakness. NNUE is
   architecturally color-symmetric (perspective HalfKA) anyway. **Do not chase a
   "Black weakness."**
2. **"Losses start early" is partly the book** dropping both engines into sharp
   unbalanced positions where the worse side loses fast. Not a clean ARCHON
   signal.

## Strategic implications (honest)

- **Beating SF16 (+30) requires an eval BETTER than SF16's, at ~equal depth.**
  ARCHON's D40 net is distilled from SF16 → it is a slightly-lossy copy that
  loses at equal depth. To *exceed* SF16 needs training targets/coverage beyond
  SF16 itself (SF18/deeper labels, more data, self-play) + the compute to consume
  them. This is exactly the value-foundation thesis and re-affirms the **D36
  distributed-test/training FLEET as the binding (transformational) lever.**
- **Speed levers matter MORE than a naive depth-count suggests (correction
  2026-07-19).** The −41.5 has two components: (a) a per-node eval-quality deficit
  (SF16 judges each node slightly better) and (b) a **depth/inference-tax deficit**
  — the D40 net (HalfKA+FullThreats+multi-layer head) is heavy per eval (it LOST
  the equal-time gate to the lighter classic net by −57.9 on tax alone, exp-0073),
  which is a real chunk of why ARCHON is +1 ply behind SF16. Speed attacks (b)
  directly, and the empirical Elo-per-NPS is high: opt#1's +13-15% NPS bought
  **+28.8 Elo** (exp-0101) — ~2 Elo per 1% NPS, FAR above what "+0.1-0.2 ply"
  depth-math predicts (extra nodes buy ordering fills, qsearch, and TM headroom,
  not just nominal depth). So the smallnet gate (+8-15% NPS if the cheap net is
  cheap enough) plausibly banks **~+8 to +30 Elo**, not "a few" — its exact
  vs-SF16 value is capped by (a) and is what the equal-time gate sweep measures.
  Bottom line: speed closes the depth/tax component (meaningful, opt#1-proven);
  it does NOT touch the per-node eval-quality component (a) — that is the fleet.
  Both are real; neither alone closes −41.5.
- **EXP-UNC-1 (eval-uncertainty consumers) is likewise incremental** — it makes
  search *use* the eval better; it does not make the eval better than SF16's.
- **The endgame is the one phase where ARCHON is ~2 ply shallower** — a small,
  concrete place the smallnet gate (cheap eval on decided/endgame nodes → more
  endgame depth) could actually help disproportionately. Worth checking in the
  gate sweep.

## Refinement: the gap is FINE-GRAINED eval, not blunders (blindspot mining)

`tools/analysis/sf16_eval_blindspots.py` mines positions where ARCHON's eval was
roughly equal/slightly-better (E_a ∈ [−0.5, +1.5]) right after its move but
LOSING (≤ −1.0) after SF16's single reply — genuine "thought it was fine, wasn't"
oversights (won-position eval-compression filtered out).

**Only 57 such blindspots in 26,892 ARCHON moves (~0.2%).** ARCHON's eval is
almost never grossly wrong. So the −41.5 is NOT blunder-driven — it is a
consistent, small per-node eval disadvantage (SF16 judges each node ~a fraction
of a pawn better, accumulating over the game), exactly what "distilled-from-SF16,
slightly lossy" predicts. Corollary: **you cannot close the gap by patching a
handful of blindspots; the whole eval has to get uniformly sharper** → data +
compute (fleet), not a targeted fix.

Where the rare blindspots DO occur: 44% opening / 37% endgame; the worst cases
are sharp positions where ARCHON UNDER-rates an enemy king attack (e.g.
`7k/7p/1pn3p1/p1pnpr2/1P2N1q1/P1PP1pPr/5P1P/1R1Q1NRK b` — ARCHON +1.20, actually
−3.5). A weak (N=57) hint that king-safety/attack in sharp middlegames is the
softest spot. Saved as a regression suite:
`data/nnue/testsuites/archon-eval-blindspots.epd` (a better net should rate these
nearer the SF16-implied value).

## What this changes

Nothing already in flight is wasted, and the one-box program is worth MORE than
first stated (see the speed correction above): speed (smallnet gate, ~+8–30 if
the cheap net is cheap enough) + consumers (EXP-UNC-1, +3–15) plausibly recover a
real chunk of −41.5 — best case to roughly parity, worst case single digits. But
they attack the depth/tax component, not the per-node eval-quality component:
**crossing SF16 to +30 still needs the fleet + better-than-SF16 training data,**
because that is the only lever on component (a). So the honest sequence is: bank
the one-box speed/consumer Elo (in flight) to close toward parity, then the fleet
to cross. Surface the fleet to the user as the decisive lever; the local pipeline
banks the approach.
