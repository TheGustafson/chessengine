# Integral study notes — index

Notes written while studying the **Integral** chess engine source in
`integral_reference/integral/`. One markdown file per topic, listed here.

Purpose: Integral is an independently-developed, **solo**, top-6 /
TCEC-competitive C++ engine — a third data point (after Stockfish and Reckless)
on how a Stockfish-class engine is actually built, and — uniquely among the
three — a clean case study in **how a single hobbyist reached the top tier**.
The study therefore has two payoffs: the usual code/technique adaptation list,
and, more importantly for this project, a **success-factor analysis** of what
made a small solo effort succeed. We distill ideas into notes here, then
implement them in Rust our own way. The two payoffs are linked below.

## ⚠ Provenance & licensing warning (read before touching this material)

- **Repo:** `aronpetko/integral`, C++ (C++20/23), ~10,500 LOC (excl.
  third-party), gitignored under `integral_reference/integral/`.
- **Strength:** top-6 / TCEC-competitive at time of study; ~98.5% one solo
  developer (Aron Petkovski — 659 of 669 commits; 10 from ~3 community
  contributors).
- **License: NONE.** There is **no LICENSE file** at the repo root, which means
  **all rights reserved by default** — this is *stronger* than Stockfish's
  GPLv3 or Reckless's AGPLv3. Per **D27** (applied here even more strictly), we
  take **ideas, algorithms, constants, and formulas only** (facts are not
  copyrightable); we **never copy code** and we **never ship or network any
  Integral artifact** (including its nets, which stay gitignored per D35's
  firewalling rule if ever loaded for a dev-only A/B). Every note reproduces
  mechanisms as hand-transcribed prose/pseudocode in our own words, with
  file:line or commit-hash citations back to the vendored source for
  re-verification — not as pasted code.
- **Scale caveat:** Integral's internal eval scale (QA=255, L1_QUANT=128, eval
  scale 200) and its ~126 SPSA-tuned constants are its own; **compare formula
  shapes, not raw numbers.**

## 👉 The payoff

**[00-success-factors.md](00-success-factors.md) — THE HEADLINE.** The ranked,
evidence-backed answer to *what made Integral succeed as a small solo project*:
distributed OpenBench-style test compute (the dominant lever), the SPSA
tuning cadence, the disciplined one-idea-per-PR / `Bench:`-trailered rhythm, the
iterative HCE→NNUE data strategy, and the shared ecosystem — each with the
in-repo evidence and a concrete **lesson for us**, emphasizing what we are NOT
yet leveraging (distributed testing, SPSA scaffold) and should. Leads with an
executive summary.

**[00-design-implications.md](00-design-implications.md)** — the code/technique
adaptation list: prioritized, roadmap-ordered (P9 search / P10 TM / P11 SMP-NUMA
/ P13 NNUE / cross-cutting), each with the exact `design/NN` §section it
touches, rough effort & impact, and a status tag `[novel | already-planned |
already-have | conflicts-Dxx]`. Compares Integral vs Reckless vs SF vs us. Leads
with the highest-value technique takeaways.

## Notes

- [01-board-movegen-magics.md](01-board-movegen-magics.md) — bitboard+mailbox
  hybrid board with **fused** place/remove+Zobrist mutation; **copy-full-state
  undo** (no incremental unmake — polar opposite of our D8); same
  generate-then-lazy-legal movegen as SF/us but pre-filters pinned knights and
  does *not* restrict single-check evasions at generation time; **plain/dense
  magics** (~2.25 MB); the one studied engine that **ships PEXT** (opt-in,
  CPU-family-gated to exclude slow Zen1/2 — confirms our D12).
- [02-search-node.md](02-search-node.md) — the SF/our N0–N14 node skeleton but
  consistently the **cheaper, less-hardened variant** (looser TT-cutoff,
  shallower IIR gate, SF-classic 2-D LMR table, SF-flat ProbCut); independently
  removed check extensions and adopted symmetric aspiration widening we've
  shipped from day one; keeps NMP eval-reduction and killer moves; the standout
  gap is its **SPSA/OpenBench tuning pipeline** we lack.
- [03-move-ordering-history.md](03-move-ordering-history.md) — one 8-stage
  MovePicker; **still uses 2 killer moves** feeding LMR (counter-evidence to
  D17); the novel **`eval_complexity`** distrust signal (`|corrected−raw
  static eval|` → RFP + LMR); pawn/major/non-pawn correction keys (dropped
  minor — opposite of SF/us); unified shared gravity constant; threat context
  *and* additive threat bonus both present.
- [04-nnue-architecture.md](04-nnue-architecture.md) — `(768×12 factorized king
  buckets → 1536)×2 → (16→32→1)×8`; **split-halves pairwise-multiply activation
  (FT_SHIFT=9, identical to Reckless)**; output-bucket formula
  `(popcount−2)/4` an **exact match to our design/11 §3.4.1**; 8-month
  incremental scaling timeline (output buckets before king buckets); **zero
  threat features** (top-6 without them).
- [05-nnue-inference-accumulator.md](05-nnue-inference-accumulator.md) — int8-
  sparse-FT→f32-tail hybrid (dpbusd + NNZ), two-tier RawNetwork/Network layout
  with a **build-time `preprocess` binary**, ply-indexed lazy accumulator with
  **eager DirtyPiece** (matches us, contra Reckless), Finny refresh cache, **no
  accumulator work on null moves**; a real **VNNI-flag-never-wired-to-fast-path
  bug** flagged; profile-guided SPARSE_PERMUTE (speculative).
- [06-smp-numa-scaling.md](06-smp-numa-scaling.md) — the **minimalist** lazy-SMP
  (fused Searcher, one cyclic Barrier, no voting, no jitter, no NUMA for
  10 months); the "scales well" reputation predated NUMA code; a well-evidenced
  ~2.5-month NUMA first-touch **bug-hunt** — one abandoned false start (an
  `emplace_back` data race, apparently caught pre-merge, not shipped) plus a
  barrier-lifetime bug our borrow checker would reject that did land briefly;
  **zero threading tests** — direct evidence our TSan/loom plan earns its keep.
- [07-datagen-training.md](07-datagen-training.md) — node-budgeted self-play with
  private TT/RNG; **pure self-bootstrap** (own HCE → NNUE, no foreign teacher);
  book + organic opening diversity; **record-everything-filter-downstream**
  binpack path (validates raising our P_cap); a **DATAGEN build flag** that
  strips search-only score adjustments from training labels; conservative draw
  adjudication; **zero committed provenance metadata** (our meta.json is better).
- [08-timeman-uci-tuning-build.md](08-timeman-uci-tuning-build.md) — simpler TM
  (no ply term, no movestogo; Alexandria-borrowed then SPSA-refit 6×); a
  fixed 4096-slot node-effort array; **deliberately minimal UCI**; the
  **`TUNABLE`/`TUNABLE_STEP` SPSA scaffold** (126 tunables, OpenBench print
  format) — the biggest tooling gap vs us; 6-target build matrix; 47% `Bench:`-
  trailer rate confirming our D24 line.
- [09-dev-history-progression.md](09-dev-history-progression.md) — **the
  669-commit development story** (SUCCESS-FACTOR): ~98.5% solo; a sharp
  three-era discipline curve; the OpenBench-wiring night as the real inflection
  point; only 1 explicit revert on `main` (survivorship bias); HCE-as-thin-bridge → NNUE; 7 SPSA
  campaigns; **no design docs at all** (a counter-lesson, not a license to cut
  ours); the S-curve of gains.
- [10-ecosystem-force-multipliers.md](10-ecosystem-force-multipliers.md) — **the
  ecosystem arithmetic** (SUCCESS-FACTOR): in-repo OpenBench fingerprints; the
  SPSA scale; the error-bar math proving several-million-games is impossible on
  one desktop (⇒ distributed testing is the dominant, ~half-the-story lever);
  the shared idea/plumbing pool (CPW, Alexandria, bullet, Fathom, UHO book,
  lichess-bot); the individual-vs-ecosystem split.
- [11-nnue-success-story.md](11-nnue-success-story.md) — **the eval data story**
  (SUCCESS-FACTOR / P13): pure self-referential net lineage (own HCE bootstrap,
  each net retrained on the prior's self-play); an `own_hce_data` vs
  `lichess_hce_data` A/B that chose self-play; ~945 branches as the unit of
  experimentation; the DATAGEN decision-policy-vs-eval-accuracy split; net-swap-
  as-one-line-commit; P13 as a perpetual, no-finish-line research thread.
- [99-further-study.md](99-further-study.md) — **coverage/accuracy audit**:
  confirms subsystem + success-factor coverage against this index; records
  the corrections applied to `06`/`07`/`09`/`00-success-factors.md` after a
  git-log spot-check (the NUMA-saga "shipped for ten weeks" claim didn't
  hold up — the commit was never merged to `main`); a standing caution
  about `--all` vs `main`-only git citations for future study passes.

## Quick map of Integral source

| Module / path | What it covers |
|---|---|
| `src/chess/{bitboard.h,board.h,board.cc,move.{h,cc},move_gen.{h,cc},fen.cc}` | Bitboard+mailbox board with fused place/remove+Zobrist, copy-full-state undo, generate-then-lazy-legal movegen, packed u16 move |
| `src/magics/{attacks.h,attacks.cc,entry.h,magic_finder.{h,cc},precomputed.h}` | Plain/dense magic tables; brute-force offline magic finder; late opt-in PEXT (CPU-family-gated) |
| `src/engine/search/{search.cc,search.h,stack.h,constants.h}` | ID + aspiration, the negamax node (all pruning/reductions/extensions), qsearch, killer moves, `eval_complexity`, material-phase scaling, Syzygy probe gate |
| `src/engine/search/move_picker.{h,cc}` | One 8-stage MovePicker (2 mode flags), blended history score, threat-context + threat-bonus quiet scoring |
| `src/engine/search/history/{history,bonus,quiet_history,capture_history,continuation_history,continuation_entries,pawn_history,correction_history}.h` | Butterfly/capture/continuation/pawn histories, unified gravity constant, total-score continuation gravity, pawn/major/non-pawn correction |
| `src/engine/evaluation/nnue/{nnue.h,nnue.cc,accumulator.h,sparse.h}` | 1536-wide dual-perspective FT, split-halves pairwise-multiply activation, NNZ dpbusd, ply-indexed lazy accumulator + Finny cache, 12 king / 8 output buckets |
| `shared/nnue/definitions.h`, `shared/simd.h`, `shared/multi_array.h` | Net constants (QA/L1_QUANT/FT_SHIFT/scale), SIMD kernels, `MultiArray` |
| `preprocess/net_processing.cc` + `preprocess/CMakeLists.txt` | Standalone build-time net-layout transpose/permute (quantize vs SIMD-layout split) |
| `src/data_gen/{data_gen.h,data_gen.cc}`, `src/data_gen/format/{format.h,fens.h,binpack.h}` | Node-budgeted self-play, book/organic opening diversity, FEN + binpack formatters behind one `OutputFormatter` seam |
| `src/tuner/spsa.h` | `TUNABLE`/`TUNABLE_STEP` macro pair — zero-cost `constexpr` normally, UCI-option + OpenBench-format print under a build flag |
| `src/engine/uci/uci.cc`, `src/main.cc` | Minimal UCI + declarative command DSL; the `argv bench` OpenBench calling convention |
| `src/engine/search/syzygy/{syzygy.h,syzygy.cc}`, `third-party/fathom` | Syzygy via vendored Fathom (MIT), generic always-on in-search probe |
| `src/engine/search/{transpo.cc, thread pool in search.h}`, `src/utils/barrier.h` | Lock-free TT; fused Searcher pool; one cyclic Barrier (no per-thread addressability); first-touch NUMA |
| `src/utils/{types.h,list.h}`, `third-party/{fmt,incbin}` | Core enums, `List`, formatting, net embedding |
| `src/tests/{bench_test.cc,perft_test.cc}`, `Makefile`, `CMakeLists.txt`, `README.md` | Shared 49-FEN OpenBench bench suite, perft suite, 6-target build matrix, net auto-download by `NETWORK_NAME`, arch string |
