# TEMPO

A UCI chess engine in Rust, built from an empty repository toward
Stockfish-class strength — with every design decision documented, every
strength claim measured in games, and every net and corpus tracked in a
content-addressed provenance ledger.

## What it is

- **Search**: iterative-deepening alpha-beta with the modern pruning canon
  (TT, null move, razoring/RFP, ProbCut, LMR/LMP, SEE pruning, singular
  extensions, upcoming-repetition detection via cuckoo tables), on a
  bitboard + mailbox hybrid board with fancy-magic sliding attacks.
- **Evaluation**: NNUE — a 768-wide HalfKA feature transformer extended
  with a threat-event channel, int8 quantization with per-column scales,
  lazy threat folding, and a multi-head tail (value, WDL, variance, and a
  search-disagreement head) in our own `CENN` container format that
  carries per-net metadata (including the output-bucket layout).
- **Training**: an in-repo JAX stack — data generation, deep-search
  relabeling, per-corpus target-transform and teacher calibration,
  performance-triggered distillation annealing — feeding a self-mining
  data flywheel: the engine plays, its games are relabeled by a deeper
  oracle, and the result trains the next net.
- **Method**: games arbitrate. Screens are screens, verdicts need power;
  the experimental ledger (`ledger/`) records every net, corpus, and
  match with a regeneration recipe. The design suite (`design/`) is the
  contract the code is checked against.

## Build

```sh
cargo build --release            # portable binary
RUSTFLAGS="-C target-cpu=native" cargo build --release   # machine-local
cargo test                       # full test tier incl. perft oracle
```

The engine speaks UCI (`./target/release/chessengine`); `bench`,
`perft <depth> [fen]`, and `d` are available as argv subcommands. NNUE
weights load via `setoption name EvalFile` — release nets ship separately
with a net card documenting their full training lineage.

## Provenance and attribution

TEMPO's search architecture follows the ideas of the open-source
state of the art. Concretely:

- **Stockfish** (GPLv3): the search-heuristic canon and many tunable
  starting values were adopted at a documented upstream revision; the
  transcription is disclosed in the design documents, and TEMPO is
  licensed GPLv3 accordingly. No Stockfish code is copied.
- **Reckless** (AGPLv3) and other engines are studied as idea sources
  only, with attributed design notes in-repo.
- **Training data**: portions of the training corpus lineage derive from
  Leela Chess Zero and official Stockfish data (ODbL); nets trained on
  them are produced works and carry an attribution notice. The
  distillation teacher used in parts of the lineage is a CC0-licensed
  network from `official-stockfish/networks`.

## License

GPL-3.0-or-later. See `LICENSE`.
