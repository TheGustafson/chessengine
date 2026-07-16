# Official nnue-pytorch training recipes (wiki + master-net PRs, read 2026-07-15)

Sources: github.com/official-stockfish/nnue-pytorch/wiki (Home, Basic training
procedure easy_train.py + train.py, Training datasets, Training examples,
Training infrastructure) and Stockfish PRs #3927 (SFNNv4), #4020 (SFNNv5),
#4100. Read during the B4 value-foundation investigation; distilled here per
repo convention (ideas only, no code).

## The canonical two-stage recipe (every documented master net)

**Stage 1 — from scratch, on SHALLOW fixed-budget SF self-play:**
- Data: `nodes5000pv2_UHO.binpack` (SF self-play, **fixed 5000 nodes/move**,
  UHO openings) or depth-9 gensfen data. Wiki: depth 9 / nodes 5000 "remains
  optimal" for this stage.
- `--lambda=1.0` (PURE score; no outcome blend in stage 1 for SFNNv4/v5).
- LR 8.75e-4, `--gamma=0.992` applied PER EPOCH (slow exponential decay),
  batch 16384, `--random-fen-skipping 3` (~75% skipped).
- 400–600 epochs × 100M positions/epoch = **40–60 BILLION samples**.

**Stage 2 — warm retrain of the stage-1 net on RICH data:**
- Data: Lc0-derived mixes (T60/T70 + Farseer etc.), or interleaved
  Leela+DFRC (`Leela-dfrc_n5000.binpack`).
- `--resume-from-model` (weights only), **LR halved** to 4.375e-4,
  gamma 0.995, 800 epochs = **80B samples**.
- Lambda 1.0 fixed (SFNNv4) or **scheduled 1.0 → 0.75** (SFNNv5+, linear
  over the retrain; later work uses a win-rate-model loss).

**Explicit wiki warnings:**
- "Usually best to FIRST train a network with datasets generated with
  Stockfish (depth 9, nodes 5000), and THEN retrain an already good network
  using various Lc0-derived datasets."
- "Training solely on the Lc0-derived datasets doesn't produce as good
  results."
- On what makes a good dataset: "We don't really know."

## Filters (train.py defaults)

- `smart-fen-skipping` (ON): skip positions whose best move is a capture
  (+ in-check class) — "positions with tactical captures are not very well
  understood by the network."
- `wld-fen-skipping` (ON): skip positions "where the evaluation doesn't
  correlate with the end game result" — an eval/outcome consistency filter.
- `random-fen-skipping 3`: stochastic ~75% skip for stream decorrelation.
- Max skipping rate exists as a knob (SFNNv5 reduced 15 → 10).

## Selection/testing discipline

- Nets are game-tested DURING training (`run_games.py`, c-chess-cli, e.g.
  20k nodes/move, 200 games/round, explore factor 1.5) and ranked by ordo
  Elo; `delete_bad_nets.py` prunes by Elo. Loss is plotted but selection is
  by GAMES. "Training exhibits variance; multiple runs needed for meaningful
  comparisons" (they run ~16 parallel runs and pick winners).
- Typical "maturity" ~400 epochs; competitive after ~100.

## Direct implications for ARCHON (hypothesis-grade, preregistered in
docs/research/value-foundation-state.md before E1/E2 results)

1. sf-100 (25M, SF16 self-play, fixed 5000 nodes/move) is EXACTLY the
   official stage-1 data shape. "Why did net-0 work first try" has a
   documented answer: that data shape is what the field bootstraps on.
2. Every large ARCHON bulk failure cold-trained stage-2-style data (deep
   heterogeneous Lichess evals, mixed-provenance fishpack) from scratch —
   the pattern the wiki explicitly warns against. Our one staged warm
   continuation (D40 SB6-15) — the closest thing to their stage 2 — was
   also our largest single improvement (−881 → −75).
3. Scale gap: official stage 1+2 ≈ 120B samples with per-epoch exponential
   decay and an LR reset between stages; ARCHON's largest run = 12B with a
   single cosine.
4. Their default filters include an eval/outcome-consistency skip
   (wld-fen-skipping) and ~75% random skip; ARCHON gained equivalents only
   on 2026-07-15 (`NET0_RANDOM_SKIP_PCT`, `NET0_DRAW_KEEP_PCT`; no
   wld-consistency filter yet).
5. Their in-training checkpoint selection is by game Elo, never loss —
   institutionalizing what ARCHON learned empirically (loss does not
   predict strength).
6. Lambda practice: stage 1 pure-score (1.0), stage 2 annealed to 0.75.
   ARCHON's fixed WDL 0.3 (≈ lambda 0.7) is more outcome-weighted than
   either official stage.
