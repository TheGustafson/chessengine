# TRANSITION — 2026-07-28 (loss-forensics session)

Read this first, then `docs/research/LOSS-FORENSICS-PLAYBOOK.md` (the method,
the tools, the traps, a literal 16-rung WHY ladder), then the standing bar at
the top of `CLAUDE.md`.

---

## 1. THE STANDING BAR (user, 2026-07-28) — read before triaging anything

> **Work backwards from where we want to be — a top-tier engine — not forward
> from where we are. Top-tier engines almost never blunder big moves. Do not
> accept blunders; root them out with aggressive investigation.**
>
> **And they are the CHEAPEST defects to fix**, because each leaves a complete
> trace: a position, the move played, the move that was right, a measurable
> gap, and a reproducible search you can bisect gate by gate. Cumulative drift
> leaves nothing to load into a debugger. Rank by **tractability × Elo**, not
> Elo share.

This corrects a real mistake I made and must not be repeated: I computed that
blunders cost ~5.5 Elo against ~32 Elo of per-move drift and used the ratio to
deprioritise them. Wrong three ways — the detector had ~9% recall, centipawns
are not additive into outcomes, and tractability was ignored entirely.

---

## 2. LIVE STATE

| thing | state |
|---|---|
| **D48X training** | **COMPLETE**, `TRAIN EXIT 0`, sb520/520, heldout 0.006350. Artifacts `data/nnue/experiments/d48x-1280/d48x-1280-sb520-ema-quant.bin` (+40 GB of checkpoints worth pruning) |
| **gift-corpus harvest** | RUNNING — 5,981 moves from lost games, oracle-scored. Output `$SCRATCH/gift_corpus.json` + `.fens`, sorted by win-probability lost |
| **TM fix @ 10+0.1** | RUNNING, 58/400. Arm `FALLING_EVAL_PREV_AVG_COEFF=3.45, FALLING_EVAL_ITER_COEFF=1.5625` |
| **forensics agent** | RUNNING (Opus 5, background). Task: 2 more losses of the game-18 mode + a fix for all three |
| git | all work committed on `wip/class4-material-eval` |

Match binaries **rebuilt and verified** with the new default (bench 587683):
`archon-d47v-pgo`, `archon-d47v-tune-pgo`, `archon-d48x-pgo`.

---

## 3. SHIPPED

**`ASPIRATION_MSS_DIVISOR: 10588 → 6776`** (commit `f34e4e0`, `Bench: 587683`,
pin updated in `check.sh` + `CLAUDE.md`). The only Elo-relevant change of the
session, and it shipped on a derivation rather than a measurement.

`meanSquaredScore` is a **squared** eval quantity; our frame is 0.64× SF18's,
so our mss is 0.41× theirs while we divided by the same transcribed constant —
the adaptive window was 1.56× too narrow. Full write-up:
`docs/research/aspiration-frame-bug.md`.

Note what was NOT shipped: a 4000-position screen preferred **3000**, which has
a catastrophic tail the screen's mean cannot see (depth 6→8 costs 3,193 nodes
at 6776 and **25,554** at 3000). Caught by
`lmr_shrinks_the_effective_branching_factor`. **The derivation produced the
safe value; the empirical screen produced the dangerous one.**

---

## 4. THE GENERAL RULE THAT FOUND IT

A transcribed constant scales by **r^k**, k set by its **dimensional role**:

```
k = +1   divisor of a SQUARED eval quantity
k = -1   coefficient converting eval into another unit (history, time)
k =  0   a ratio, or a converter already in our frame
```

Five previous frame campaigns scaled **linearly** and were structurally
incapable of finding this. `tools/frame_audit.py` now carries a `CROSS_UNIT`
table plus a guard for any `params::` constant appearing beside an eval-derived
identifier that is unclassified — it found `SEE_THRESH_MULT` on its first run.
All four affected constants are `f64`/`i32`/`i64`, never `Value`.

---

## 5. FAILURE MODES (measured, 120-game match vs SF18)

| mode | share | status |
|---|---|---|
| Aspiration root instability | — | **FIXED** (mss) |
| Depth not reached in time | — | mechanism confirmed; Elo under test |
| Eval gates hide a refutation | ~1/3 of errors | **OPEN** — agent working it |
| Residual eval error | ~2/3 | OPEN, least understood |

**Blunder rate, detector-independent** (SF18 @120k `searchmoves` over every
move in lost games, 260 sampled): **1.5% of moves lose ≥100 cp ≈ one per lost
game**; 0.40 game-equivalents of win probability per lost game. The own-eval
detector found 5 in 120 games — **~9% recall**.

**Time management:** we spend 9.98 s of a 10 s bank before move 33 (SF18: 8.78);
by move 40 we hold 0.49 s to their 2.33 s; 47.3% of our moves are under one
second against their 25.2%. Grind losses fall 22.5→16.7 per 100 at equal nodes
while tactical losses do not move — two instruments, one phenomenon.

### Game 18, fully traced (the worked example)
`8/k2bN3/Pn2p3/R4p1B/7P/1rp3P1/5K2/8 w` — we play Be2, Rc5 is right.
1. Refutation is Black's quiet **Bb5**, deflecting the rook so the c-pawn queens.
2. Handed the move, **our own search scores it +442 at d12, +605 at d20** —
   better than SF18's +298. Not the net, not depth.
3. LMR is not the cause (`RED_TABLE_NUM=1000` does not help).
4. **RFP is** — the only gate whose removal flips the game position.
5. Frame-free: our eval sits **1.43 shared razoring margins** below alpha where
   SF18 sits at **0.76**. Both engines use RAZOR_BASE 465.

---

## 6. REFUTED — do not re-run without a new mechanism

- Eval-frame **linear** rescaling, five formulations. Median disagreement with
  SF18 is −0.02 pawns over 30k move pairs: no bias to correct.
- `HINDSIGHT_EVAL_SUM=111`: +0.03±0.15 at full corpus, −17.4 Elo in games.
- `EVALDIFF_*_MULT` ×1.5625 (eval→history): 0.0 Elo over 140 games.
- **`RFP_THREAT_MARGIN=0`**: fixes game 18 but +0.23 cp and **+7.5% nodes** at
  full corpus. The D30 relaxation earns its place; global blunting is dead.
- `SEE_THRESH_MULT` 1024→1600: cp neutral-to-worse at +4.7% nodes.
- "Our net under-rates compensation": refuted at n=250 — we are *more*
  optimistic (−1.01 vs −1.59 margins). Game 18 is an outlier against trend.
- Tablebases as the endgame explanation: SF18 played with `tbhits 0`.
- Sparse-endgame concentration of errors: an artifact of a **pawn**-threshold
  blunder detector; dissolves in win probability.

**D48X net:** better than d47v *and* SF18 on the net-only suite (Spearman
0.8004, iso-MAE 105.7), NPS **0.845×**, equal-node **+8.9±17**, equal-time
**−12.6±17**. Not a gain at equal time.

---

## 7. TOOLS BUILT THIS SESSION

| tool | purpose |
|---|---|
| `tools/rematch.py` | ARCHON vs X, both legs, net-live guard, chains match_report + loss_positions; self-play A/B with `--opponent-options` |
| `tools/blunder_forensics.py` | own-eval collapses → search_miss / eval_instability / already_lost. **Win-probability threshold, not pawns.** ~9% recall — use the oracle audit instead |
| `tools/depth_to_find.py` | settle-depth per engine vs depth reached in game; separates search shape from depth-in-time |
| `tools/deep_oracle.py` | uncensored `searchmoves` oracle over any FEN list |
| `$SCRATCH/harvest.py` | **the gift corpus** — every ≥100 cp error in every lost game, oracle-scored, ranked by WP lost |
| `screen.py --oracle-audit` | tier strength in cp; `--reference` + `--budgets` = eval-vs-search decomposition |
| `frame_audit.py` | type-derived + `CROSS_UNIT`, exits non-zero on an unclassified constant |
| `match_report.py --tc` | clock story (reconstructed remaining time) |

---

## 8. INSTRUMENT TRAPS (each cost hours)

- **Net-live guard** — never remove it. A net-less binary invalidated a day, twice.
- **Screen power**: sem ~0.23 cp at 4000 positions, ~0.15 at 9992; effects are
  0.2–0.3 cp. `--limit 4000` is a coin flip. Use `--limit 0` to decide.
- **Deterministic vs wall-clock**: fixed-node/bench/fhf are load-immune (nice 19
  beside anything); NPS and timed games need a quiet box. I once read NPS at
  load 21 and got 0.755× where the truth was 0.845×.
- **`oracle-obs.json` is 100% 21+ pieces** — zero endgames.
- **Loss/sparse corpora are mate-score dominated** (±20000): use top-1 rates,
  never absolute means. Symptom: ±50–85 cp error bars.
- **`sf_a(material)` clamps to [17,78]** — invalid in a 12-point endgame. Prefer
  the frame-free shared-margin comparison.
- **MultiPV only unhides ROOT children.** For pruning anywhere in the tree, use
  all-eval-gates-off. MultiPV misclassified the very failure it was meant to find.
- **Elo error from the real W/D/L split**, not the 0.25 shortcut (halves it).
- **Two runs of one config read +27.9 and −14.7.** Pool before believing.
- **`build_pgo.sh` takes a NAME**, not a path; its `cp` can hit `Text file
  busy` and `set -e` will silently abort the rest of the chain. Verify the
  default landed by **behaviour** (`tunables` / bench), not timestamp.

---

## 9. NEXT

1. **Work the gift corpus top-down** — it is the session's main asset. Each row
   is a debuggable defect. Validate any fix on **three** independent losses of
   one mode; one position is a patch.
2. Finish the TM Elo at 10+0.1 (mechanism already confirmed: the arm holds ~1 s
   more clock at move 40, time-trouble moves 32.8%→25.4%).
3. Split the residual ~2/3 bucket into depth-fixable vs eval-fixable — the
   probe was written (`$SCRATCH/depthsplit.py`) and never completed.
4. For mode 3, the untested hypothesis: RFP's premise is "nothing of ours is
   attacked ⇒ calm", and `all_threats()` is an **attack map** blind to a passed
   pawn near promotion. Gate the relaxation on that rather than removing it.
   **Hypothesis, not a finding** — the node I checked did not even satisfy
   `no_own_threat`.
