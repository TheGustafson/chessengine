# Stockfish study notes — index

Notes written while studying the Stockfish source in `Stockfish/src/`.
One markdown file per topic, listed here as they're written.

Purpose: when we're unsure how to do something optimally, read how
Stockfish does it, distill the idea into a note here, then implement it
in Rust our own way. **Never copy code** — Stockfish is GPLv3; we take
ideas and measurements, not source.

## Notes

- [00-design-implications.md](00-design-implications.md) — distilled action
  list: what the study means for our Rust design, ordered by roadmap
- [01-bitboards-attacks-move-encoding.md](01-bitboards-attacks-move-encoding.md)
  — Move u16 layout, XOR square/piece tricks, fancy magic bitboards (no PEXT
  in slider path!), precomputed tables (~1 MB total), Carry-Rippler + epoch
  magic search
- [02-position-state-make-unmake.md](02-position-state-make-unmake.md) —
  StateInfo copied/derived split, Vec-stack instead of pointer chain, Zobrist
  tricks (zeroed promo ranks, material count slots), castling-rights mask per
  square, incremental checkers/blockers/pinners, do_move order + prefetch
- [03-movegen-movepick-history.md](03-movegen-movepick-history.md) —
  monomorphized GenType generation, target-mask trick, bulk pawn shifts, lazy
  legality, staged MovePicker + partial insertion sort, gravity history
  update rule, killers are gone
- [04-tt-search-structures.md](04-tt-search-structures.md) — 10-byte TTEntry,
  32-byte 3-entry clusters, mul_hi64 indexing, relaxed-atomic lock-free TT,
  generation aging, large pages, search Stack/RootMove, threading layout
- [stockfish_search_notes.md](stockfish_search_notes.md) — comprehensive
  review of the search **algorithms**: iterative deepening + aspiration
  windows, the main negamax node step-by-step (every pruning/reduction/
  extension technique with current constants), qsearch, MovePicker scoring +
  all history tables and update formulas, SEE, correction history/optimism,
  TT replacement policy, Zobrist key family, lazy SMP + thread voting, time
  management, and a dependency-ordered Rust implementation roadmap

## Quick map of Stockfish source

| File | What it covers |
|---|---|
| `types.h` | Square/Piece/Move encodings, Value, Score |
| `bitboard.{h,cpp}` | Bitboard utilities, magic bitboard init |
| `attacks.{h,cpp}` | Attack table generation |
| `position.{h,cpp}` | Board state, make/unmake, StateInfo, Zobrist |
| `movegen.{h,cpp}` | Staged legal/pseudo-legal move generation |
| `movepick.{h,cpp}` | Move ordering (TT move, captures, killers, history) |
| `search.{h,cpp}` | Alpha-beta search, pruning, extensions |
| `evaluate.{h,cpp}` + `nnue/` | Evaluation (NNUE) |
| `tt.{h,cpp}` | Transposition table |
| `uci.{h,cpp}`, `engine.{h,cpp}` | UCI protocol and engine glue |
| `perft.h` | Perft driver |
- `05-nnue-pytorch-training-recipes.md` — official two-stage training recipe (stage-1 shallow 5000-node self-play from scratch; stage-2 warm retrain on rich data at halved LR), filters (random/smart/wld-fen-skipping), 40-120B-sample scale, Elo-based net selection; read 2026-07-15 for the B4 investigation.
- `06-sf14-sf16-search-patch-study.md` — full mining of sf_14..sf_16 (480 commits, ~210 qualifying non-NNUE patches) + fishtest methodology + PR discussions: what classes of search change kept winning (extensions, cutnode/cutoffCnt reductions, re-search ladder, history-fed pruning, optimism, TM-by-uncertainty), what got simplified away, "gainer season" after evaluator jumps, and the ranked ARCHON port list; mined 2026-07-16 during the D41-prep study.
- `07-s0-sf16-mechanics-audit.md` — S0 current-code verification pass over `src/search/mod.rs` vs the `eca43a97` reference (the same tree doc 10 specs): a LIVE/PARTIAL/DEFERRED/ABSENT inventory table with line refs and SF16-vs-post-SF16 attribution, an 8-item ranked gap shortlist (reduction-bundle un-deferral, IIR, history-buys-lmr, optimism), and top-3 implementation sketches. Findings: correction history + all 4 TM factors are LIVE; only IIR and Optimism are truly ABSENT; the biggest code lever is re-enabling the shelved post-SF16 reductions now that the eval jumped.
