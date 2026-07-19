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
- **Speed levers are INCREMENTAL for the SF16 goal, not decisive.** opt#1
  (+13-15% NPS) gave +28.8 Elo in self-play (ARCHON vs slower ARCHON), and speed
  still compounds generally — but against SF16 the depth gap is already only
  +1 ply, so +8-15% NPS (~+0.1-0.2 ply) will NOT close a gap that survives at
  equal depth. **The smallnet gate is still worth finishing for general Elo, but
  it is not the SF16-closer** — I had over-framed it. Fair value: a few Elo, not
  "closes the −41.5."
- **EXP-UNC-1 (eval-uncertainty consumers) is likewise incremental** — it makes
  search *use* the eval better; it does not make the eval better than SF16's.
- **The endgame is the one phase where ARCHON is ~2 ply shallower** — a small,
  concrete place the smallnet gate (cheap eval on decided/endgame nodes → more
  endgame depth) could actually help disproportionately. Worth checking in the
  gate sweep.

## What this changes

Nothing already in flight is wasted (all incremental gains compound), but it
reorders the *narrative*: the one-box program (speed + consumers) buys single-
to-low-double-digit Elo; **the −41.5 → +30 jump needs the fleet + better-than-SF16
training data.** Surface the fleet to the user as the highest-leverage next step
whenever local levers are exhausted.
