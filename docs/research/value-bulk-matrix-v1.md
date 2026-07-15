# ARCHON Bulk Value Matrix v1

Status: v1 early screen superseded by converged v2, 2026-07-15.

## Question

Can broad official-Stockfish game data or broad Lichess analysis-board data
train an Archon evaluator that is competitive with net-0, and which accumulator
capacity/feature family benefits most? The matrix is an architecture and source
screen, not final hyperparameter optimization.

## Sources

1. `official-stockfish/master-binpacks:fishpack32.binpack`, complete file,
   pinned by the Hugging Face LFS SHA-256 in the captured manifest. Preserve
   score, side-to-move result, rule-50, game continuity, and filename
   provenance. This is official Stockfish training data, not named SF18 gold.
2. `lichess/chess-evaluations` Kaggle v14, all 17 verified Parquets. Select one
   target per contiguous FEN group using the first PV of the maximum
   `(depth, knodes)` evaluation. Carry the nine groups known to cross shard
   boundaries. Exclude mate-only, invalid/identifiable-Chess960, and extreme
   score records. Missing rule-50/repetition and missing outcome remain explicit.

No neutral placeholder may enter an outcome-weighted loss. Lichess-only runs
use `WDL=0`; binpack runs retain genuine game outcomes. A combined run may use
outcome-bearing binpack batches and cp-only Lichess batches, but must apply the
loss mask per source rather than interpreting neutral placeholders as draws.

## Four architecture cells per source

| Cell | Accumulator/features | Width/head | Purpose |
|---|---|---|---|
| A | Chess768 | classic SCReLU 256 | net-0-compatible control |
| B | Chess768 | classic SCReLU 1024 | isolate capacity |
| C | king-conditioned 64×Chess768 | classic SCReLU 256 | isolate king-conditioned accumulator value |
| D | exact HalfKAv2_hm + FullThreats | 768, 8 bucket D40 head | production-feature candidate |

Cell B intentionally changes width without feature semantics. Cell C changes
accumulator features while retaining a simple head. Cell D tests the richer
Stockfish/Reckless-inspired feature and bucketed-head path. All packed artifacts
must pass trainer/quantized/runtime parity appropriate to their format.

## Budget and stopping

- Run one-batch parser/feature/pack smoke before every new source/architecture
  combination.
- Use matched accepted presentations within a source for the first comparison;
  record effective rows, cycles over source data, LR schedule, result blend,
  score scale, filters, and seed.
- Pack early checkpoints. First strength screen: 80 paired games at 5k nodes.
  Stop immediately for crashes, semantic failures, or a clearly catastrophic
  result (approximately below -150 Elo).
- Survivors receive 400 paired games at 2k, 5k, and 20k nodes. A single positive
  point estimate does not pass; inspect the curve and pair bootstrap intervals.
- Only source-specific survivors may enter combined training. Combined training
  must preserve per-source target semantics and compare against the best
  source-only sister architecture, not only net-0.

The original automation stopped at superbatch 10 while retaining a
30-superbatch cosine-decay horizon. Its classic256 checkpoint was therefore
still at LR 0.000757 and lost about 512 Elo; this is retained as an unfinished
early-training datapoint, not interpreted as a corpus result. The production
matrix is `value-bulk-matrix-v2`: 30 superbatches, a completed 30-superbatch
decay, 747,110,400 accepted presentations per cell, and checkpoints every five
superbatches. All survivor decisions are made from these converged checkpoints.

## Known controls

The first cold classic256 run on a 61.07M-position prefix, using the historical
net-0 schedule, lost about 348 Elo to net-0 in an 80-game 5k screen. Therefore
scale alone is not assumed to solve target/sampling mismatch. The complete-file
matrix proceeds because the user requested the broad screen, but every cell is
fail-fast and early checkpoints are retained. A low-step net-0 fine-tune could
preserve strength, yet its apparent +8 Elo at 5k became -31 Elo at 2k and -9 Elo
at 20k; it is rejected by the curve gate.

## Storage

The immutable 40.6GB Parquet source is copied to
`gs://archon-main-bucket/corpora/lichess-chess-evaluations/kaggle-v14/raw/`
and verified before local originals are removed. Local disk then holds the
canonical selected Bullet stream, the chosen complete binpack, active
checkpoints, and promoted nets. Disposable raw checkpoints are deleted after
packing/parity to avoid silently exhausting the training host.
