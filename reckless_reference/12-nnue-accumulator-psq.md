# NNUE accumulator, PSQ feature transformer, incremental update

Source: `src/nnue.rs`, `src/nnue/accumulator.rs`, `src/nnue/accumulator/psq.rs`
(primary), plus `src/nnue/accumulator/threats*.rs`, `src/board/makemove.rs`,
`src/search.rs`, `src/evaluation.rs`, `src/board.rs` (`BoardObserver`) for how
the pieces wire together. Reckless v0.10.0-dev, AGPLv3 — ideas and formulas
only, no code reproduced (D27-equivalent).

## 0. Vocabulary trap up front

Reckless names its **primary NNUE input-feature accumulator**
`PstAccumulator` / "Pst" (`nnue/accumulator/psq.rs`). Despite the name, this
is **not** a Stockfish-style separate PSQT side-output blended with the
network output — there is no such shortcut anywhere in `Parameters`. "Pst"
here just means "piece-square feature" — it's the thing Stockfish calls the
*feature transformer accumulator* (HalfKA-ish). The task brief's framing
("PSQ side-output... how it combines with the net") does not have a
Reckless analogue: Reckless is a single forward network, FT → L1 → L2 → L3,
with **no PSQT bypass term**. Read the rest of this note with that
substitution in mind: "PSQ" = "the main feature-transformer accumulator",
full stop.

## 1. Architecture shape (`src/nnue.rs:82-123`)

Constants (exact, `nnue.rs:82-111`):

```text
NETWORK_SCALE = 380
INPUT_BUCKETS = 10        OUTPUT_BUCKETS = 8
L1_SIZE = 768             L2_SIZE = 16      L3_SIZE = 32
FT_QUANT = 255            L1_QUANT = 64
FT_SHIFT = 9  (feature-transformer post-clip shift, AVX-512 uses u32, else i32)
DEQUANT_MULTIPLIER = (1 << FT_SHIFT) / (FT_QUANT * FT_QUANT * L1_QUANT) as f32
```

`Parameters` (`nnue.rs:356-367`) holds, in order: a **threat** feature-weight
table `[[i8; 768]; 66864]` (i8, not i16 — see §4), the primary piece-square
weight table `[[i16; 768]; INPUT_BUCKETS*768]` = 7680 rows, an `i16[768]`
bias row, then `l1_weights: [[i8; L2*L1]; OUTPUT_BUCKETS]`,
`l1_biases/l2_weights/l2_biases/l3_weights/l3_biases` all **f32**. So only
the FT→L1 boundary is integer-quantized (i16 accumulator, i8 L1 weights,
classic SF-style `mulhi`+shift dequant, `DEQUANT_MULTIPLIER`/`FT_SHIFT`
above); L1→L2→L3 (16→32→1) run in plain f32 — those layers are cheap enough
that quantizing them buys nothing. Two independent 768-wide accumulators
feed the FT output: the primary "Pst" one and a parallel "threats" one
(§4); their concatenated, activated outputs are what L1 consumes.

Forward pass per node (`nnue.rs:286-301`, `output_transformer`):
1. `bucket = OUTPUT_BUCKETS_LAYOUT[popcount(all occupancy)]` — 8 buckets by
   total piece count (`nnue.rs:113-123`: bucket 0 for ≤8 pieces ... bucket 7
   for 29-32), selecting which of the 8 `l1/l2/l3` weight sets to use.
2. `activate_ft` (SIMD, `forward/{scalar,vectorized}.rs`) clips/squares the
   two 768-wide accumulators for the side-to-move-relative pair.
3. `find_nnz` looks up a **precomputed 256-entry nonzero-bit table**
   (`SparseEntry{indexes:[u16;8], count}`, built once in `Network::new`,
   `nnue.rs:144-157`) to skip zero lanes of the clipped FT output before the
   dense 768×16 L1 matvec — the classic "NNZ sparse propagate" trick: after
   ReLU-family clipping a large fraction of FT outputs are exactly zero, so
   the L1 layer only walks the nonzero columns.
4. `propagate_l1/l2/l3` (f32 from L1 on) produce a scalar; final score is
   `(l3_out * NETWORK_SCALE) as i32`.

This is a materially deeper/wider net than our own Phase C v1 plan (design
doc 11 §3.4.3: single hidden layer, 768→256→1). Reckless is effectively a
live example of the doc's own stated "upgrade path (L1 512, buckets...)" —
see §6 below.

## 2. The accumulator stack and how it rides make/unmake

### 2.1 Storage: a flat, ply-indexed array (not a linked StateInfo chain)

`Network` (`nnue.rs:132-167`) owns `pst_stack: Box<[PstAccumulator]>` and
`threat_stack: Box<[ThreatAccumulator]>`, each **preallocated to `MAX_PLY`**
(240, `types/mod.rs:26`) and indexed by a plain `usize` cursor `self.index`
— not by `Board`'s own `state_stack` length (which includes UCI-replayed
game history, matching exactly the concern our doc 11 §3.4.4 flags: "never
by `board.states.len()`, which includes UCI-replayed history"). `push`/`pop`
just do `self.index += 1` / `-= 1` (`nnue.rs:169-185`); no computation
happens on push, and **`pop` does zero work at all** — decrementing the
index is sufficient because the popped slot's ancestor entries are still
sitting untouched in the array.

### 2.2 Two-tier hook strategy: lazy metadata vs. eager observer calls

The wiring in `search.rs:1424-1444`:

```text
make_move(td, ply, mv):
    td.nnue.push(mv, &td.board)      // BEFORE board mutation
    td.board.make_move(mv, &mut td.nnue)   // Board::make_move<T: BoardObserver>

undo_move(td, mv):
    td.nnue.pop()                    // no recompute
    td.board.undo_move(mv)           // no observer at all on undo
```

`Network::push` (`nnue.rs:169-181`) is called **while the board is still
pre-move** and just snapshots three fields into the new slot's delta:
`mv`, `piece = board.piece_on(mv.from())`, `captured = board.piece_on(mv.to())`.
That's the entire "dirty piece" record for the *Pst* accumulator — nothing
else is written at make-time; the actual add/sub feature indices are
re-derived later, lazily, inside `PstAccumulator::update` by re-pattern-
matching on `mv.kind()` (`psq.rs:105-138`): normal move → 1 add/1 sub;
capture/promotion-capture → 1 add/2 subs (own move + captured-piece removal,
en-passant capture square computed as `to ^ 8`); castling → 2 adds/2 subs
(king + rook, rook squares from `board.get_castling_rook`).

Meanwhile `Board::make_move` *does* drive a `BoardObserver` trait
(`board.rs:597-609`) with three granular callbacks —
`on_piece_change`/`on_piece_move`/`on_piece_mutate` — fired inline as each
placement mutation happens (`makemove.rs:32-122`; e.g. castling fires
`on_piece_change` for the rook's removal, `on_piece_move` for the king, then
`on_piece_change` for the rook's arrival — three calls for one move).
`Network`'s `impl BoardObserver` (`nnue.rs:342-354`) forwards each callback
straight into `push_threats_on_move/_on_mutate/_on_change`
(`threats.rs`/`accumulator.rs` — see §4), which append `ThreatDelta` records
to the **current** stack slot's delta list *during* `make_move` itself.

So the two accumulators use genuinely different update disciplines for the
same seam: the primary FT accumulator is **fully lazy** (3 scalar fields
written at push, real work deferred to `evaluate()`, re-derived from move
metadata) while the threats accumulator is **eagerly recorded, lazily
applied** (per-placement-mutation callbacks fire during `make_move`, but the
SIMD add/sub over the 768-wide vector still only happens later, at
`evaluate()`, via the same lazy-chain mechanism as §2.3). The reason: the
Pst delta is fully reconstructable from `(mv, piece, captured)` after the
fact, so no observer plumbing is needed; the threat feature set is a
function of the *board's occupancy at each intermediate placement step*
(who's now attacking what), which is only observable while the mutation is
happening, not reconstructable from before/after board snapshots alone.

`Board::undo_move` (`makemove.rs:124-145`) takes **no observer argument at
all** — unmake is placement-only; nothing NNUE-shaped is invoked, matching
`Network::pop`'s no-op. This mirrors our own design doc 11 §3.4.4's
observation that Stockfish gets its freshness guarantee "structurally"
(accumulator lives inside `StateInfo`, fresh per `do_move`); Reckless gets
the same guarantee by **overwriting `pst_stack[index].accurate = [false;
2]`** and stamping a fresh `delta` on every `push`, i.e. explicit
invalidation over a reused array slot, functionally identical to our own
plan's "the explicit invalidation *is* the freshness guarantee" for a
ply-indexed stack.

### 2.3 Lazy gap-fill at `evaluate()` time

`Network::evaluate` (`nnue.rs:197-218`) is the only place real accumulator
work happens. For each POV independently: if the top-of-stack slot is
already `accurate[pov]`, skip. Otherwise:

- `can_update_pst(pov)` (`nnue.rs:242-262`) walks `self.index` backward
  looking for the nearest ancestor slot with `accurate[pov] == true`. While
  walking, if it passes a ply whose recorded delta was **the POV's own king
  moving across a king-bucket or kingside/queenside mirror boundary**
  (`from.is_kingside() != to.is_kingside() || INPUT_BUCKETS_LAYOUT[from] !=
  INPUT_BUCKETS_LAYOUT[to]`), it aborts and returns `None` — the chain
  cannot be incrementally repaired across a king-relative reindexing, so a
  **full refresh** is forced instead. Otherwise it returns `Some(ancestor)`.
  Note: the backward loop provably always terminates at ply 0, which
  `evaluate` asserts is always `accurate` (root refresh guarantee).
- If `Some(k)`: `update_pst_accumulator` (`nnue.rs:220-229`) walks *forward*
  from `k` to `index`, applying one ply's delta at a time via
  `PstAccumulator::update`. The idiomatic Rust trick worth lifting verbatim
  as a pattern (not code): `if let (prev, [current, ..]) =
  stack.split_at_mut(i + 1) { current.update(&prev[i], ...) }` —
  `split_at_mut` at `i+1` yields a disjoint immutable `prev[..=i]` and a
  mutable `current` head at `i+1`, letting the borrow checker prove
  simultaneous read of ply `i` and write of ply `i+1` inside the *same*
  boxed slice, with zero unsafe and zero extra allocation. This is directly
  reusable for our own planned `AccumulatorEntry` stack walk (doc 11
  §3.4.4) instead of index-juggling with raw pointers or `RefCell`.
- If `None`: full `refresh(board, pov, cache, parameters)` (§3).

Threats use the identical shape (`can_update_threats`/
`update_threat_accumulator`, `nnue.rs:231-284`), except the trigger for
"must refresh" checks only the kingside/queenside mirror crossing, not
input-bucket crossing (threat features aren't king-bucketed — only mirrored,
per `threat_index`'s `mirrored` flag, §4).

**Cost consequence, directly relevant to our own doc's stated pitfall**
(doc 11 §6, "Lazy-update pathology: a long walk-back replays many plies"):
Reckless bounds this not by measuring replay depth but structurally — the
*only* thing that forces a full, expensive refresh is a king crossing a
bucket/mirror boundary; everything else is an unbounded-length but
*cheap-per-ply* incremental chain (one SIMD pass over 768 values per ply).
The expensive case is exactly rehabilitated by the Finny-table cache (§3),
which turns "full refresh" into "diff against the last cached position for
this bucket", not "recompute from the empty board".

## 3. Refresh and the Finny-table cache (`psq.rs:33-67`, `accumulator.rs:13-41`)

`AccumulatorCache::entries: Box<[[[CacheEntry; INPUT_BUCKETS]; 2]; 2]>` — a
flat 2 (pov) × 2 (kingside/queenside mirror) × 10 (input bucket) = 40-slot
cache, each `CacheEntry` holding the last-known 768-wide accumulator values
for that (pov, mirror, bucket) combination **plus** the exact piece/color
bitboards of the board that produced them (`pieces: [Bitboard; 6]`,
`colors: [Bitboard; 2]`). This is the "Finny table" technique (named after
the online handle that popularized it in the NNUE community; also since
adopted into Stockfish itself as `AccumulatorCaches`): instead of
recomputing a bucket's accumulator from the biases up every time a king
crosses into it, keep a standing cache of "the last board state we computed
this bucket for" and only apply the **symmetric-difference** of pieces
between then and now.

`PstAccumulator::refresh` (`psq.rs:33-67`):
1. Locate the cache slot: `cache.entries[pov][king.is_kingside()]
   [INPUT_BUCKETS_LAYOUT[king ^ (56*pov)]]`.
2. For each `(color, piece_type)`, diff the *current* board's bitboard
   against the *cached entry's* stored bitboard for that piece/color:
   `to_add = pieces & !(cached_pieces & cached_colors)`,
   `to_sub = !pieces & (cached_pieces & cached_colors)`. This can be a
   handful of squares even when the "reference" position the cache last
   held is many moves removed from the current one — the point of the
   cache is that it's usually a *recent, similar* position (same king
   bucket visited repeatedly across a search tree), not that it's the
   immediately preceding ply.
3. `apply_changes` (`psq.rs:170-202`, unsafe, register-blocked SIMD):
   processes the 768-wide vector in blocks of `8 registers × SIMD lane
   width`, loading each block once, folding in every add/sub feature's
   contribution while the partial sums stay in registers, then storing
   once — amortizing load/store bandwidth over however many add/sub
   features apply (up to 64 each, `ArrayVec<PstFeature, 64>`). This is
   the standard heavier-weight refresh kernel that pays for itself only
   because a refresh may touch many features at once (unlike the 1-2
   feature `apply_delta` used for per-move incremental updates, §2.3,
   which doesn't need register blocking).
4. The cache entry's stored bitboards/values are overwritten to match the
   *current* board — so the cache always reflects "the last position this
   bucket slot was computed for", ready to serve the next refresh with a
   cheap diff.

An AVX-512VBMI2-specific fast path (`psq.rs:80-103`) replaces the per-square
Rust loop with `_mm512_maskz_compress_epi8` + a broadcast-XOR to bulk-build
the feature-index list for a whole bitboard in one shot — a nice
architecture-specific micro-optimization not relevant to correctness, only
noted for completeness.

Note: the cache is **not** part of the search-ply stack. It's a single
long-lived structure per `Network` (so per search thread), reused across
the entire search tree and across moves within a game — that persistence
across unrelated branches, not just within one line, is the actual payoff
of a Finny table over a plain "recompute from bias" refresh.

## 4. King-relative bucket selection and mirroring

`INPUT_BUCKETS_LAYOUT: [u8; 64]` (`nnue.rs:101-111`), reproduced as a
**formula description, not the array**: rank 1 assigns 4 distinct buckets
0-3 by king file, mirrored (a/h→0, b/g→1, c/f→2, d/e→3); rank 2 assigns
4 more buckets 4-7 the same way; rank 3 collapses to one bucket (8); ranks
4-8 collapse to a single bucket (9). So granularity is finest exactly where
king safety differs most sharply (back-rank castled-king positions) and
coarsest where the king rarely visits mid-opening/middlegame (this
correlates with "buckets are a way to spend accumulator-table capacity
where the position space actually varies, not uniformly") — an asymmetric,
hand-designed bucketing scheme rather than "8×8 by square" or "N buckets by
symmetric rank/file division".

Mirroring: `king.is_kingside()` (file ≥ e) selects a horizontal mirror so
the same 10 buckets serve both halves of the board (files a-d and e-h share
bucket indices by construction — the array is already horizontally
symmetric, e.g. rank 1 reads `0,1,2,3,3,2,1,0`). The combined transform is
one XOR: `flip = (7 * is_kingside) ^ (56 * (pov == Black))` — bit 7 (file
flip, mirror) XORed with bit 56 (rank flip, POV-relative), applied
uniformly to the king square (for bucket lookup — harmless since the bucket
table is file-symmetric so the mirror component is a no-op there) **and**
to every piece square when building a feature index
(`pst_index`, `psq.rs:204-211`):

```text
pst_index(color, piece_type, square, king, pov):
    flip = (7 * is_kingside(king)) xor (56 * (pov == Black))
    bucket = INPUT_BUCKETS_LAYOUT[king xor flip]
    return bucket*768 + 384*(color != pov) + 64*piece_type + (square xor flip)
```

(384 = 6 piece types × 64 squares per color-perspective half; 768 = both
halves; total feature space = 10 buckets × 768 = 7680, matching the
`ft_piece_weights` row count exactly.) Reusing one combined XOR mask for
both the bucket lookup and every feature square — rather than computing POV
flip and mirror flip separately and applying them at different points — is
a small but genuinely clean simplification enabled entirely by the bucket
table happening to be file-symmetric.

The parallel **threat** feature space (`threats/threat_index.rs`) is a
separate, much larger and more elaborate indexing scheme: 66,864 total
threat-feature rows, built from a precomputed `PIECE_PAIR_LOOKUP[12][12]`
(attacker piece × attacked piece, with `excluded`/`semi_excluded` flags for
meaningless or symmetric-redundant pairs), a per-attacker `PIECE_TARGET_COUNT`
weighting (6/10/8/8/10/8 for P/N/B/R/Q/K — how many "target classes" that
attacker type distinguishes), and an `ATTACK_INDEX_LOOKUP[piece][from][to]`
giving the rank of `to` among `from`'s attacked squares. Every occupied
square's attacks-that-land-on-an-occupied-square (`attacks(piece, sq, occ) &
occ`, i.e. both genuine threats *and* defended-own-piece relationships) get
folded into this index (`threats.rs:74-90`, `threat_index`, only mirrored,
not bucketed — a single flip flag, no `INPUT_BUCKETS_LAYOUT` lookup). This
is the layer of the note that is hardest to reverse-engineer exactly from
reading alone; treat the constant `66864` and the exact pairing algebra as
"there is a real formula here, go re-read `threat_index.rs:1-118` before
relying on the specific numbers" rather than something to reconstruct from
this note.

## 5. What's DISTINCTIVE

**Vs. Stockfish** (per our `stockfish_reference/` notes and design doc 11's
citations of SF's NNUE shape — SF's public nets: feature transformer L1,
dual-perspective accumulators, sparse-input layers, SqrClippedReLU):

- **A second, entirely separate "threats" accumulator/feature-set run in
  parallel with the classic piece-square one.** This is not present in
  Stockfish's published architecture at all (as best our prior SF study
  captured it) — it's Reckless's own architectural idea: augmenting
  HalfKA-ish inputs with attacker/defender relationship features. This is
  the single most novel thing in this subsystem and the one item in this
  note we'd flag as "worth prototyping ourselves, later, as an experiment"
  rather than "worth adopting wholesale now."
- **Two disciplines for the same seam**: eager per-mutation observer
  callbacks for threats vs. fully lazy metadata-replay for the primary
  accumulator (§2.2). Stockfish (and our own doc 11 plan) use one uniform
  `DirtyPiece` record for everything; Reckless shows that when a feature
  set needs finer-grained information than "which squares changed",
  eager-record/lazy-apply is a workable middle ground rather than forcing
  either "fully eager compute" or "fully reconstructable from move
  metadata".
- **NNZ sparse L1 propagation** (precomputed 256-entry nonzero-bit table,
  `nnue.rs:144-157`) and a **deep-ish cascade** (FT→L1(16)→L2(32)→L3(1))
  with **8 output buckets by piece count** — more elaborate than the
  "single hidden layer" architectures our stockfish notes describe as SF's
  historical starting point, though broadly in line with where modern
  top-tier NNUE nets (SF included, in recent versions) have converged.
  Nothing here is unique to Reckless; it's "current state of the art,
  implemented".
- **Finny-table refresh cache** keyed by (pov, mirror, bucket) rather than
  by king square directly — a coarser cache (40 slots total) than some
  implementations that key per-square, trading cache-hit specificity for a
  much smaller, more cache-resident table. Whether 40 slots vs. ~128
  (2×64) matters in practice would need measurement; flagged as a concrete,
  cheap design parameter to just copy the *shape* of (small, flat array;
  not the exact dimensions) when we get there.

**Vs. our engine / our own design doc 11:**

- We currently have **no NNUE code at all** (Phase A PSQT only, computed
  from scratch every call, explicitly *not* incrementally — design doc 11
  §3.2.4's "Phase A computes... from scratch (decision)"). Everything in
  this note is comparison against our *plan* (doc 11 §3.4), not existing
  code.
- Our plan's shape is **already extremely close** to Reckless's mechanism
  at the structural level: a `Box<[AccumulatorEntry]>` stack indexed by
  **search ply** (not board-state-stack length) with an explicit
  `computed: [bool; 2]` flag and walk-back-to-nearest-computed-ancestor
  logic (doc 11 §3.4.4) is, point for point, the same design as Reckless's
  `pst_stack`/`index`/`accurate`/`can_update_pst`. This convergence is
  reassuring: it's independent confirmation (from a top-2 engine, not just
  from reading Stockfish) that the ply-indexed-array-instead-of-linked-
  StateInfo approach is sound. Our doc even explicitly names the exact
  upgrade path Reckless has already taken: "a refresh is never *forced* by
  a king move... that starts only with king buckets... which is also when
  a refresh-cache (per-bucket cached accumulator + diff) earns its keep" —
  Reckless is a working instance of exactly that future step.
  Our own `Aligned64<T>` wrapper (doc 11 §3.4.6) is the same
  `#[repr(align(64))]` pattern as Reckless's `Aligned<T>` (`nnue.rs:429-453`)
  almost verbatim in intent (not code) — independent convergence again.
- **Unsafe discipline differs.** Our D14 restricts NNUE unsafe to
  SIMD-intrinsic kernels behind one dispatch site with one `// SAFETY`
  comment (doc 11 §3.4.6: "the single dispatch call site carries the one
  ... comment"). Reckless scatters `unsafe` more liberally through the
  accumulator layer itself — raw-pointer walks in `apply_delta`/
  `apply_changes` (`psq.rs:140-202`), an `unsafe fn update` on
  `ThreatAccumulator` (`threats.rs:141`), and a raw `std::mem::transmute`
  of an `include_bytes!` blob straight into the `Parameters` struct with
  **no format validation, no version tag, no checksum**
  (`nnue.rs:369-384`) — the embedded net and the compiled struct layout
  must simply agree, enforced by nothing but the build pipeline producing
  both together. Our own planned `loader.rs` (doc 11 §3.4.5, magic +
  version + dims + FNV-1a checksum, "never panic on user-supplied files")
  is strictly more defensive; that's a deliberate choice on our side, not
  something to walk back, but worth naming explicitly as a place we're
  choosing safety margin Reckless doesn't bother with (Reckless has no
  `EvalFile`-equivalent UCI option either — one net, baked in at build
  time, no runtime loading path at all).
- **`DirtyPiece` payload shape differs slightly.** Our planned
  `DirtyPiece{adds:[(Piece,Square);2], subs:[(Piece,Square);2], n_add,
  n_sub}` pre-materializes the exact squares/pieces at hook time (populated
  while `make_move_with` already has those locals in scope). Reckless's
  `PstDelta{mv, piece, captured}` is smaller (fits comfortably in a few
  bytes) and re-derives the specific add/sub squares later by re-matching
  on `mv.kind()` inside `update()`. Both are valid; ours trades a few extra
  stored bytes per stack slot for simpler `update()` logic (no
  re-pattern-matching of move kind at every lazy-chain step); Reckless
  trades a smaller per-slot footprint for redoing that dispatch on every
  incremental step. Given `MAX_PLY` slots this is a few hundred bytes
  either way — not worth changing our plan over, just worth knowing it's a
  real, deliberate tradeoff point, not an oversight on either side.
- Reckless's `ArrayVec<T, N>` fixed-capacity stack vector
  (`types/arrayvec.rs`) used throughout the accumulator code for add/sub
  lists is the same idiom as our own `MoveList` (`Deref`-to-slice,
  fixed-array-backed, no heap allocation in hot paths) — same convention,
  independently arrived at; nothing to change, just confirmation our
  existing "no allocation in movegen/search" house style extends naturally
  to NNUE accumulator work.
- Reckless has a `piece_contribution`/`eval_with_bucket` debug pair wired
  into its `uci.rs` (`nnue.rs:303-339`, `uci.rs:352,372`) — evaluate with
  and without a given piece to report its marginal contribution, and force
  a specific output bucket — a rough analogue of our planned `trace.rs`
  "term-by-term breakdown for the eval CLI/UCI command" (doc 11 §3.7),
  though implemented as two full-refresh-and-evaluate round trips (fine for
  a debug command, not something to design the hot path around).

## 6. Adapt to us

Ordered roughly by how directly actionable each idea is against our current
plan (design doc 11, phase P13).

1. **Ply-indexed flat accumulator stack, not board-state-length indexed.**
   *Doc/phase*: design doc 11 §3.4.4 (`NnueContext.stack`), P13 step 11.
   *What*: confirms (doesn't change) our plan — `Box<[AccumulatorEntry]>`
   sized `MAX_PLY` + margin, cursor by search ply, `computed[bool;2]` flags,
   walk-back-to-nearest-computed on read. This note is mostly evidence our
   plan is already right; no design change needed, but cite Reckless as a
   second independent implementation of the same shape when justifying it
   in review.
   *Effort*: none (already planned). *Impact*: confidence, not code.

2. **The `split_at_mut(i+1)` adjacent-slot update pattern.**
   *Doc/phase*: design doc 11 §3.4.4's lazy update walk; P13 step 11
   implementation detail.
   *What*: when writing `update_pst_accumulator`-equivalent code (walking
   forward from the nearest computed ancestor, reading slot `i` and writing
   slot `i+1` of the *same* boxed slice), use `if let (prev, [current, ..])
   = stack.split_at_mut(i + 1)` instead of reaching for unsafe pointer
   arithmetic or `RefCell`/`Cell` wrapping. Directly reusable, zero
   translation cost, fits our "no unsafe unless D14 budget" rule for free.
   *Effort*: trivial (a few lines, decided at write-time). *Impact*: small
   but removes a temptation to reach for unsafe or interior mutability in
   the accumulator walk.

3. **Finny-table refresh cache, once we adopt king buckets.**
   *Doc/phase*: design doc 11 §3.4.4 (explicitly named as future work:
   "which is also when a refresh-cache... earns its keep") — this is a
   post-v1 upgrade step, not part of the initial P13 net.
   *What*: a small flat cache (Reckless: 2×2×10 = 40 `CacheEntry` slots,
   each a full accumulator + the piece/color bitboards it was computed
   from) keyed by (pov, mirror, king-bucket), diffed against the *current*
   board rather than recomputed from bias every time a king crosses a
   bucket boundary. Concrete, copyable *shape* (small array of {accumulator
   + snapshot bitboards}, diff-against-cached-snapshot algorithm); exact
   dimensions (10 buckets, that specific asymmetric rank/file layout) are a
   net-architecture decision we'd make when we design our own king-bucket
   scheme, not something to copy verbatim.
   *Effort*: medium (new cache struct + diff logic + register-blocked SIMD
   apply kernel) — but only relevant once we have king buckets at all,
   which is itself a later net-architecture decision.
   *Impact*: potentially large once buckets exist (turns "recompute 768
   feature adds from empty" into "diff a handful of features against a
   recent snapshot") — but zero relevance to our unbucketed v1 net.

4. **NNZ sparse L1 propagation, once we grow past a single hidden layer.**
   *Doc/phase*: design doc 11 §3.4 "upgrade path (L1 512, buckets...)" —
   post-v1, whenever we add an L2 (our v1 is 768→256→1, no L1/L2/L3
   cascade, so there's no sparse-matvec to skip yet).
   *What*: precompute a 256-entry table (one per possible byte of
   nonzero-lane bitmask) listing which lanes are nonzero, use it to skip
   zeroed (clipped) FT-output columns in the dense matvec into the next
   layer. Cheap to build once (`Network::new`-equivalent), meaningfully
   cuts the matvec's real work once FT outputs are sparse after clipping.
   *Effort*: small once we have a second dense layer to skip work in.
   *Impact*: throughput of the forward pass at that future point; no
   correctness risk (it's a work-skipping optimization over an identical
   mathematical result, testable by comparing against the dense path
   bit-for-bit).

5. **Eager-record/lazy-apply as a third pattern for future auxiliary
   features**, alongside our currently-planned single uniform `DirtyPiece`.
   *Doc/phase*: design doc 11 §3.4.4 (the `on_dirty` hook design) — relevant
   if/when we ever add a second NNUE input plane beyond plain piece-square
   (e.g., if we experiment with Reckless's "threat" idea, item 6 below, or
   any other feature needing finer granularity than before/after squares).
   *What*: keep the existing plan (one `MakeMoveHooks::on_dirty` method,
   one `DirtyPiece` payload) for the primary accumulator; if a future
   feature set needs per-sub-mutation visibility that `(mv, piece,
   captured)` can't reconstruct, add a *second* hook method
   (`on_piece_placed`/`on_piece_removed`-style, mirroring Reckless's
   `BoardObserver`) rather than trying to force everything through one
   record shape. Not needed now — flagged so we don't reflexively reject
   a second hook method later as "inconsistent with the DirtyPiece plan"
   when it may genuinely be the right tool for a specific feature.
   *Effort*: zero now; small later (one more default-no-op trait method,
   same bench-neutrality argument as doc 11 step 9 already makes for
   `on_dirty`).
   *Impact*: design-flexibility only; no immediate code.

6. **A parallel "threat" input feature set — flagged as a research idea,
   not a near-term port.**
   *Doc/phase*: design doc 11 §3.4 (net architecture, post-v1); would first
   need our own datagen/training loop (design docs 15/16) working, since
   it changes the net's input dimension and requires retraining, not just
   an inference-side change.
   *What*: augment the classic piece-square/HalfKA-style input with a
   second accumulator over (attacker piece, attacked/defended piece,
   from-square, attack-rank-among-targets) tuples, run through its own
   weight table and summed with the primary accumulator before L1. This is
   Reckless's most novel idea in this file and plausibly a real source of
   playing strength (a top-2 engine ships it), but it roughly doubles
   accumulator-maintenance complexity (a second lazy-update chain, a
   second set of observer-style hooks) and is a genuine architecture
   research question (does *our* net benefit the same way, at our
   parameter budget?) rather than a known-good port.
   *Effort*: large (new feature-index scheme, new accumulator, retraining,
   new hook plumbing) — explicitly a "later, if NNUE strength plateaus and
   we want another architectural lever" item, not P13-v1 scope.
   *Impact*: unknown until tried; Reckless's own strength suggests
   "possibly significant", but the honest answer is we don't have data on
   it for our net specifically.

7. **Debug/introspection pair (`piece_contribution`, forced-output-bucket
   eval) as inspiration for our `trace.rs`.**
   *Doc/phase*: design doc 11 §3.7 (`eval/trace.rs`, "term-by-term
   breakdown for the eval CLI/UCI command").
   *What*: when we build `trace.rs` for Phase C, consider adding a
   "marginal contribution of piece on square X" mode (evaluate with vs.
   without a piece, via a scratch board clone + full refresh) alongside
   whatever term-by-term breakdown the doc already specifies for Phase
   A/B. Purely a diagnostics nicety, not hot-path relevant.
   *Effort*: trivial once `trace.rs` and the accumulator refresh path both
   exist. *Impact*: developer-experience only.
