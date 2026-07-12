# Reckless study notes — index

Notes written while studying the **Reckless** chess engine source in
`reckless_reference/Reckless/`. One markdown file per topic, listed here.

Purpose: Reckless is an independently-developed, top-tier Rust engine — a
second data point (distinct from Stockfish) on how to build a Stockfish-class
engine idiomatically in Rust. When a design question has more than one good
answer, seeing how a *different* strong Rust engine resolved it is worth as
much as the SF reference. We distill the idea into a note here, then implement
it in Rust our own way. The payoff — a prioritized, roadmap-ordered list of
what to adapt — lives in **[00-design-implications.md](00-design-implications.md)**.

## ⚠ Provenance & licensing warning (read before touching this material)

- **Repo:** `codedeliveryservice/Reckless`, vendored at `v0.10.0-dev`
  (edition 2024), gitignored under `reckless_reference/Reckless/`.
- **Strength:** ~3833 SPCC, #2 at time of study.
- **License:** **AGPLv3.** This is *stronger copyleft than Stockfish's GPLv3*
  — even network/SaaS use of derived code triggers source-disclosure
  obligations. Per **D27**, we take **ideas, algorithms, constants, and
  formulas only** (facts are not copyrightable); we **never copy code** and we
  **never ship Reckless network files**. Every note reproduces formulas as
  hand-transcribed prose/arithmetic in our own words, with file:line citations
  back to the vendored source for re-verification — not as pasted code.
- **Scale caveat:** Reckless's internal piece/eval values (P 109, N 403, B 435,
  R 679, Q 1242) are ~½ of ours/SF's (P 208 … Q 2538). Every raw constant
  comparison in these notes implicitly carries a ~2× factor.

## 👉 The payoff

**[00-design-implications.md](00-design-implications.md)** — the prioritized,
roadmap-ordered action list: what to adapt, grouped by phase (P9 pruning NOW /
P10 time mgmt / P11 SMP+NUMA / P13 NNUE / cross-cutting Rust-idiom · TT ·
build/PGO · tooling), each with the exact `design/NN` §section it touches,
rough effort & impact, and a status tag `[novel | already-planned |
already-have | conflicts-Dxx]`. Leads with an executive summary of the
highest-value takeaways.

## Notes

- [01-core-types-move-encoding.md](01-core-types-move-encoding.md) — `Move`
  kind bits that bake in capture/queen-promo-ness (zero board access), the
  lean 3-key Zobrist family (full/pawn/non_pawn[2]), shared rule50 bucket for
  GHI + correction history, ArrayVec+score MoveList, pervasive `transmute`
  enum↔int, cubic `to_cp`.
- [02-attacks-magics-setwise.md](02-attacks-magics-setwise.md) — build-time
  magic-table codegen (`build.rs` + `include!`, no self-verification),
  `setwise.rs` whole-army Kogge-Stone attack fills, the precomputed per-node
  opponent threat map, directional `ray_pass` as the sole pin-legality table.
- [03-board-makemove-fen.md](03-board-makemove-fen.md) — current-state-as-field
  (ancestors in a Vec), fused per-node threat map, enemy-occupancy-only sniper
  detection, 3-event `BoardObserver` NNUE hook, uniform Chess960/Shredder-FEN
  castling-rook scan, shipped cuckoo repetition.
- [04-movegen-see.md](04-movegen-see.md) — fully-legal-in-one-pass movegen
  (no lazy filter), SEE with a single running balance + `-1` recurrence,
  promotions/EP evaluated in SEE, on-pin-ray recapture, branchless LVA,
  history-scaled quadratic SEE-pruning margins.
- [05-movepick-history.md](05-movepick-history.md) — unified 5-stage
  MovePicker, linear-scan selection, root re-scoring mid-iteration, threat-
  indexed butterfly/noisy history, symmetric "offense" quiet term, rule50-
  bucketed correction history, zero-init tables, no per-go decay, mmap+hugepage
  tables.
- [06-transposition-table.md](06-transposition-table.md) — cluster-level packed
  SWAR tag word (8-byte entries), non-atomic access under `unsafe impl Sync`,
  PV-node `force`-overwrite, halved age weight (4 vs 8), mmap+`MADV_HUGEPAGE`
  allocation, GHI folding from clock 0.
- [07-search-node-pruning.md](07-search-node-pruning.md) — threat-coupled RFP,
  cutoffCnt/TT-capture-coupled NMP, TT-narrowed null window, dynamic ProbCut
  re-verify, bound-type-conditioned singular margin + LDSE, `ilog2(depth)`-only
  LMR base with history substituting for move-number, `lerp` blends.
- [08-search-id-aspiration-qsearch.md](08-search-id-aspiration-qsearch.md) —
  one unified 5-stage picker for all node types, single-`see()`-call qsearch
  pruning, cross-thread optimism via packed atomic, forced PV-node TT write,
  asymmetric qsearch blend ratios, self-referential boxed Stack, majority-vote
  soft stop.
- [09-uci-time-management.md](09-uci-time-management.md) — Limits-as-tagged-enum,
  independent closed-form soft/hard bounds, layered overhead incl. movetime,
  `completed_depth==0` hard-stop guard, node-bitmask poll cadence, quorum soft
  stop, vote-based bestmove, to_cp-without-WDL, `define!` const/SPSA macro.
- [10-threading-numa-smp.md](10-threading-numa-smp.md) — three-tier state model,
  generalized closure job dispatch, cross-thread optimism fusion, 65% quorum
  soft stop, reduction-formula thread-id jitter, NUMA worth-it gate + generic
  replication registry, listener-thread UCI responsiveness.
- [11-nnue-features-threats.md](11-nnue-features-threats.md) — dual feature
  transformers (PST + threats) fused additively into one accumulator, relational
  threat features (66,864-wide), split-accumulator pairwise-multiply activation,
  int8-sparse-L1→f32-tail quantization, non-uniform king/output buckets.
- [12-nnue-accumulator-psq.md](12-nnue-accumulator-psq.md) — ply-indexed flat
  accumulator stack, lazy-metadata-replay vs eager-observer update disciplines,
  gap-search with bucket-crossing bailout, `split_at_mut(i+1)` update, Finny-
  table refresh cache, king-relative bucket+mirror via one XOR.
- [13-nnue-inference-simd.md](13-nnue-inference-simd.md) — hybrid int8/f32
  quantization, mulhi SCReLU trick, cfg-gated free-function SIMD abstraction
  (compile-time dispatch), bit-exactness discipline, movemask+LUT/hardware-
  compress NNZ detection, stable-Rust `asm!` escape hatch.
- [14-tablebases-tooling-build.md](14-tablebases-tooling-build.md) — Syzygy via
  vendored Fathom FFI (feature-gated), two-tier TB probing, const-generic bench
  output, `is_legal_perft` differential oracle, per-tier PGO with SDE training,
  cross-microarch bench-determinism CI, wasm front-end.

## Quick map of Reckless source

| Module / path | What it covers |
|---|---|
| `src/types/{square,color,piece,bitboard,moves,movelist,score,castling,zobrist,keys}.rs` | Core enums (transmute-built), `Move` kind bits, `ArrayVec` MoveList, 3-key Zobrist family, score bands |
| `build/{build,attacks,magics,maps}.rs` | Build-time bake of all leaper + fancy-magic attack tables (codegen → `include!`) |
| `src/lookup.rs` | Runtime attack-table accessors (generated), cuckoo repetition table |
| `src/setwise.rs` | Whole-army Kogge-Stone setwise attack fills (AVX2/AVX-512 + scalar fallback) |
| `src/board.rs` | Board (state-as-field + Vec of ancestors), `update_threats`, `is_legal`, `attackers_to`, slider blockers, hash/rule50 bucket |
| `src/board/{makemove,movegen,see,parser,tests}.rs` | make/undo + observer hooks, fully-legal one-pass generation, SEE, FEN/Shredder parser, perft+hash tests |
| `src/movepick.rs` | Unified 5-stage MovePicker, good/bad-noisy SEE split, threat/offense quiet scoring |
| `src/history.rs` | Quiet/noisy/continuation/correction histories, gravity update, `HugeBox` mmap allocator |
| `src/search.rs` | ID + aspiration, negamax node (all pruning/reductions/extensions), qsearch, hygiene |
| `src/parameters.rs` | `define!` const-vs-SPSA-tunable macro scaffold (declared, unpopulated) |
| `src/transposition.rs` | 8-byte entries, cluster SWAR tag word, PV force-overwrite, mmap hugepages |
| `src/thread.rs`, `src/threadpool.rs`, `src/numa.rs` | ThreadData, closure job dispatch, NUMA replication registry + worth-it gate |
| `src/time.rs` | Closed-form soft/hard bounds, node-bitmask poll cadence, quorum soft stop |
| `src/uci.rs` | Two-thread UCI loop, options `match`, vote-based bestmove, listener thread |
| `src/evaluation.rs` | `correct_eval` (material-weighted optimism blend + rule50 damp) |
| `src/nnue.rs` + `src/nnue/**` | Dual PST+threat accumulators, ply-indexed stack, Finny cache, hybrid-quant forward pass, per-ISA SIMD kernels |
| `src/tb.rs`, `deps/Fathom/` | Syzygy via vendored Fathom C library (MIT), feature-gated FFI |
| `src/tools/{bench,perft,speedtest}.rs`, `src/misc.rs` | Bench/perft/speedtest, `is_legal_perft` oracle, dbg counters |
| `src/wasm.rs` | cfg(wasm32) browser front-end (Web Worker threading via raw pointers) |
| `build/build.rs`, `Makefile`, `.github/workflows/*` | Release profile, per-tier PGO, cross-microarch bench-determinism CI |
