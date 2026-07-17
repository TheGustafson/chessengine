# R4 micro-audit — hot search/eval paths (2026-07-17)

Read-only performance micro-audit of ARCHON's hot search + eval paths. No source
was modified; nothing was built or benched (a GPU trainer holds `target/`).
Candidates are ranked by `(expected NPS gain × confidence) / risk`.

**Verification convention.** A *pure speedup* must reproduce the pinned
fingerprint **bit-for-bit (bench 701605)** — check.sh on both sides of the
change. A *functional* change (alters which nodes are searched) states a new
count and must clear an SPRT. Each candidate below is tagged.

Scope note: the SOTA (D40) full-feature eval is the shipping evaluator, so its
per-move accumulator maintenance dominates the eval hot path. Several of the
biggest levers live there. The classic PSQT/CENN-v1 paths are already tight.

---

## Ranked candidates

| # | Site | Cost today | Proposed change | Effect (guess) | Functional? | How to measure |
|---|------|-----------|-----------------|----------------|-------------|----------------|
| 1 | `sota_threats.rs:556-612` (`ThreatContext::apply`), driven from `search/mod.rs:3906-3909` via `eval/mod.rs:206-216` | FullThreats accumulator is folded **eagerly at every `do_move`**: a ~3 KB parent→child copy (`:573`) + per-perspective resolve + `apply_i8_deltas` (`:609`), even for children that cut off via TT before ever calling `evaluate`. HalfKA, by contrast, is lazy (`sota_accumulator.rs:61-64` just records the dirty record). | Make the 768-lane fold lazy like HalfKA: at `do_move` store the resolved `(adds, subs)` index lists per ply/perspective; defer the child-copy + `apply_i8_deltas` to `SotaContext::evaluate`, folding from the nearest computed ancestor. | Large for the SOTA path — removes the biggest fixed per-made-move eval cost on the fraction of children that never evaluate (TT cutoffs, immediately-refuted moves). Guess 5-15% SOTA NPS. | **No** if the eval is bit-identical → bench-neutral (701605); it only moves *when* work happens. | Build `--features sota`; confirm bench unchanged; A/B NPS at fixed depth with the shipping net. |
| 2 | `sota_head.rs:117-131` (`find_nnz`), called `:210` | Scalar 192-iteration serial scan with a data-dependent branch per 4-byte group, run once per SOTA eval to compact the ~85-95%-sparse FT activation. The branch mispredicts on the sparse pattern. | Vectorize: load 16/32 FT bytes, `movemask` the nonzero lanes, and compact group indices via a precomputed shuffle/`vpcompress` table (the SF/Reckless `vec_nnz` trick). Produces the identical nnz set. | Medium — `find_nnz` is comparable to the `l1_dot` it feeds because it is 192 serial branchy iterations. Guess 2-6% SOTA NPS. | **No** — identical nnz set ⇒ bit-identical eval ⇒ bench-neutral. | Unit-test new nnz == scalar nnz over random FT; confirm bench 701605; A/B NPS. |
| 3 | `sota_threats.rs:581-584` inside the per-perspective loop `:574-610` | Two `[0usize; DELTA_CAP]` (`DELTA_CAP == 256`, `:88`) scratch arrays are declared *inside* the perspective loop → ~8 KB of stack re-zeroing per made move (2 perspectives × adds+subs × 2 KB). Only the runtime prefix is read, so LLVM cannot drop the memset. | Hoist to persistent `ThreatContext` fields with explicit `len`s (the existing `DeltaBuf` pattern, `:90-127`) — reset the length, never zero the storage. | Low-medium — removes an ~8 KB memset per SOTA `do_move`. Guess 1-3% SOTA NPS. | **No** — functional-neutral; bench-neutral (701605). | Confirm bench unchanged; A/B NPS `--features sota`. |
| 4 | Quiet-pruning gate `search/mod.rs:2687-2690` vs `stat_score` `:3444-3445` (called `:2817`) | For a quiet that survives the shallow-pruning block, the two continuation-history slab reads `cont_ref(stack[cur-1])[pci][to]` and `cont_ref(stack[cur-2])[pci][to]` are done **twice**: once in the continuation-history-pruning gate, once in `stat_score`. Each read is an indexed load into a 1.5 KB slab (likely a cache-line touch). | Compute the offset-1/offset-2 cont-history pair once in the gate and thread the value into `stat_score` (or hoist both reads above the gate). | Low — 2 redundant slab loads per surviving quiet at shallow nodes. Guess <1% NPS. | **No** — same arithmetic ⇒ bit-identical ⇒ bench-neutral (701605). | check.sh both sides (must match 701605). |
| 5 | Main search: `correction_value` `search/mod.rs:1876-1878` precedes the NonPV TT cutoff `:2063-2149` | `correction_value` (4 masked loads into the 1 MB `corr` table + 2 into `cont_corr`, `history.rs:632-665`) runs on **every** main-search node, including NonPV TT-cutoff nodes. Qsearch already defers it past its own TT cutoff (`:3635` is after the cutoff at `:3604`). The main-search ordering is deliberate (hindsight `:2018-2027` reads `static_eval` to adjust `depth`, which the cutoff `:2055` then uses). | Compute `correction_value`/`static_eval` lazily so a TT-cutoff node where hindsight does not fire (the common case — parent did not hard-reduce) skips the correction loads, matching SF's "cutoff before eval" order. | Medium potential (TT-cutoff nodes are a large fraction of interior nodes) but low confidence it survives the hindsight coupling cleanly. | **Yes** — reordering changes hindsight's `depth` input on the nodes where it *does* fire; SPRT-gated, not bench-neutral. | Proxy SPRT at fixed nodes; then equal-time SPRT. High risk of subtle regression. |
| 6 | `do_move` `search/mod.rs:3886` vs caller `:2542` | `capture_stage(m)` is recomputed in `do_move` (`:3886`, for the `ContHistId`) though the main-search caller already holds `is_capture` from `:2542`. | Pass `is_capture` into `do_move` (add a param; qsearch computes it locally). | Negligible — `capture_stage` is a single bitboard `contains` for the common case (`movegen.rs:640-650`). | **No** — bench-neutral (701605). | Not worth measuring alone. |

---

## Detail on the top candidates

### 1. FullThreats is eager where HalfKA is lazy (SOTA path)

`SotaContext::push` (`sota_context.rs:66-69`) calls `halfka.push` (records the
dirty record, O(1), lazy — `sota_accumulator.rs:61-64`) **and**
`threats.apply_at` (eager — `sota_threats.rs:617-621` → `apply` `:556-612`).
Every made move therefore pays, up front:

- `let mut child = parent;` (`:573`) — a copy of `[[i16; 768]; 2]` ≈ 3 KB;
- `build_deltas` (`:646-743`) reading the post-move board (magic-slider,
  knight, pawn attack lookups per touched square);
- per perspective: `threat_index` (`sota_features.rs:324-348`) per raw event +
  an O(add×sub) linear dedup (`:591`, `:599`) + `apply_i8_deltas` (`:609`);
- `self.stack.push(child)` (`:611`).

The architectural reason it is eager is real: the delta list is derived from the
post-move board, which is gone once sibling subtrees mutate the board in place.
But that constrains only the **resolve** step (`build_deltas` + `threat_index`)
to run at `do_move` time. The 768-lane accumulator write (`apply_i8_deltas`) and
the 3 KB child copy can be deferred: store the resolved `(adds, subs)` per
ply/perspective and fold forward lazily at `evaluate` time from the nearest
computed ancestor — exactly the HalfKA `compute` pattern
(`sota_accumulator.rs:73-103`). Children that cut off before `evaluate` (TT
cutoffs, quick fail-lows, reduced scouts that immediately re-probe) then never
pay the fold. The eval result is unchanged, so this is bench-neutral and its
only gate is an NPS A/B. It is the largest single lever here but also the
biggest refactor: the sibling-overwrite/`truncate` semantics
(`apply_at` `:617-621`) and the undo path (`:627-633`) must be preserved, and
the debug oracle (`sota_context.rs:90-108`) must stay green.

### 2. `find_nnz` is a scalar serial scan

`find_nnz` (`sota_head.rs:117-131`) is the one clearly-scalar step left in the
otherwise AVX-512-VNNI SOTA forward. It runs 192 iterations, each OR-ing four
bytes and taking a data-dependent branch, to compact the sparse FT activation
before `l1_dot`. Because the nonzero pattern is data-dependent the branch
mispredicts often, so this can rival the `l1_dot` (`:141`) it feeds even though
`l1_dot` only touches the ~10-15% nonzero groups. The standard fix (SF/Reckless
`vec_nnz`) processes 16/32 bytes at once, forms a nonzero bitmask with
`movemask`, and appends group indices through a precomputed compaction table.
Output is the identical index set, so bit-exact and bench-neutral. New unsafe
SIMD ⇒ needs a scalar-parity unit test (the file already has the pattern:
`sota_simd.rs:559-642`).

### 3. Threat scratch arrays re-zeroed per made move

`apply` (`sota_threats.rs:581-584`) declares `adds`/`subs` as `[0usize; 256]`
inside the perspective loop. Two 2 KB arrays × 2 perspectives = ~8 KB zeroed per
SOTA `do_move`. Only `&adds[..add_len]` / `&subs[..sub_len]` are ever read, but
`add_len`/`sub_len` are runtime values, so LLVM cannot elide the initializing
memset. The `DeltaBuf` field (`:90-127`) already demonstrates the fix: a
persistent buffer whose `clear` (`:118-121`) only resets a length. Move
`adds`/`subs` onto `ThreatContext` the same way. Functional-neutral, trivial,
low risk — a good warm-up before tackling #1.

---

## Not worth it / verified clean (so the next reader doesn't re-derive)

- **Correction-history at qsearch stand-pat — no redundant recompute (the R4
  question).** `qsearch` computes `correction_value` exactly once
  (`search/mod.rs:3635`), inside the `!in_check` block and **after** the qsearch
  TT cutoff (`:3604`), for its own position. `search` computes it once per node
  (`:1876`). There is no double-computation within a node and no sharing to be
  had between a node and its qsearch child (different positions). All four
  position keys it reads — `pawn_key`/`minor_key`/`non_pawn_key(W/B)` — are O(1)
  incremental reads off the top `StateInfo` (`board.rs:533-547`), not
  recomputes. The lookup cost itself (4 masked loads into the 1 MB `corr` table
  + 2 into `cont_corr`, `history.rs:632-665`) is inherent to the SF-parity
  correction design. The *only* asymmetry worth knowing is that the **main
  search** spends this before its TT cutoff while qsearch spends it after — that
  is candidate #5, not a qsearch bug.

- **TT probe layout / prefetch timing.** Already correct: one cluster ==
  32 bytes, `align(32)`, never straddles a line (`tt.rs:298-305`); probe scans 3
  entries in order and returns the first tag match (`:472-505`); `mul_hi64`
  indexing avoids division (`:411-413`); the child cluster is prefetched
  mid-make via the `TtPrefetch` hook the instant the key is final
  (`search/mod.rs:432-446`, `board.rs:377-399`). No reordering opportunity found.

- **Classic (CENN-v1) accumulator forward.** `NnueContext::evaluate` already
  hoists the `KERNELS` LazyLock deref out of the per-node path
  (`accumulator.rs:121-124`), folds forward with a disjoint `split_at_mut` and
  **no** per-ply accumulator memcpy (the kernel writes straight into `dst`,
  `:176-186`), and routes every move-kind delta shape through the SIMD seam
  (`:245-273`). Tight.

- **`MovePicker` hoisting.** `init_quiets` (`movepick.rs:508-649`) already
  resolves the five continuation slabs, the pawn slot, the butterfly/low-ply
  planes, the threat/offense tiers, and the slider `tables()` deref **once** per
  node (`:551-566`), not per move. `init_captures` (`:456-490`) likewise. No
  per-move recomputation of node-constants found. A TT-move cutoff still does
  zero generation/scoring/sorting (`:352-355`).

- **Allocation in the move loop.** None on the hot path: generation uses a stack
  `MoveList`; the picker owns its `[ExtMove; 256]` buffer; the threat
  `DeltaBuf` is persistent. The only per-move heap-ish cost is the incremental
  accumulator stack `push` (candidate #1), which is a `Vec` write within the
  64-entry reserve, not a realloc.

- **Reordering the `evaluate` net-dispatch chain** (`eval/mod.rs:265-313`). The
  active net is the same on every call, so the `if let Some` chain is
  perfectly predicted; reordering saves nothing measurable.

- **`capture_stage` double-compute** (candidate #6): listed for completeness but
  it is a single bitboard test; not worth a signature-affecting refactor on its
  own.

- **Branch hints / cold-path extraction.** The `#[cfg(feature = "policy")]` and
  `#[cfg(feature = "search-control")]` blocks throughout the move loop and
  picker are compiled out of the shipping build entirely — no runtime cost, no
  `#[cold]`/`likely` opportunity that the compiler isn't already handling on the
  hot arms.
