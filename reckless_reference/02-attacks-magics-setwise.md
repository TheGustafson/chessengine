# Attack generation in Reckless: build-time magics, lookup, setwise

Source: `build/build.rs`, `build/attacks.rs`, `build/magics.rs`, `build/maps.rs`,
`src/lookup.rs`, `src/setwise.rs`, plus consumers in `src/board.rs`,
`src/board/movegen.rs`, `src/board/see.rs`, `src/movepick.rs`,
`src/types/bitboard.rs`, `src/types/square.rs`. Version studied: v0.10.0-dev
(AGPLv3 — ideas only, no code reproduced; see the license note at the end).

Read together with `stockfish_reference/01-bitboards-attacks-move-encoding.md`
(SF's `attacks.{h,cpp}`) and our own `design/03-attack-tables.md` /
`src/attacks.rs`. The short version: **Reckless bakes the entire fancy-magic
attack table at Cargo build time and hardcodes the magic constants**, whereas
both Stockfish and our engine search for magics and populate the table once at
process startup. On top of that, Reckless adds a capability neither SF nor we
have: a **setwise ("fill") attack generator** that computes the union of
attacks from a whole bitboard of same-type pieces in one shot, vectorized over
AVX2, and uses it to precompute a full per-node opponent "threat map" that
both movegen legality and move ordering consult in O(1).

## 1. The build-time pipeline (the big structural difference)

Cargo's `build = "build/build.rs"` (`Cargo.toml:6`) mechanism runs
`build/build.rs` as an ordinary host-target Rust program *before* the crate
itself compiles. Reckless uses it for several codegen jobs (network path env
vars, compiler/version strings, optional Syzygy FFI bindings) and, relevant
here, for the entire attack-table machinery (`generate_attack_maps`,
`build/build.rs:68-73`).

Three private modules live under `build/` and are compiled *only* into the
build-script binary, never into the shipped engine:

- `build/attacks.rs` — a **naive**, on-the-fly attack generator: directional
  bit-shifts (`shift_dir`, masking off the wrapping file only — rank overflow
  is left to fall off the top/bottom of the `u64`, exactly the comment at
  `build/attacks.rs:1-4` says), a ray walker that stops at the first blocker
  (`generate_slide`, `build/attacks.rs:68-79`), and `generate_ray` for
  line/between-style geometry (`build/attacks.rs:57-66`). This is the
  correctness oracle — structurally the same idea as our `sliding_attack`/
  `safe_destination` and SF's `sliding_attack`, just written with explicit
  per-direction shift amounts instead of a shared step-table.
- `build/magics.rs` — two `const` arrays, `ROOK_MAGICS`/`BISHOP_MAGICS: [MagicEntry; 64]`,
  each entry `{ mask, magic, shift, offset }` (`build/magics.rs:1-3`), with
  **literal hardcoded values** for every square — `ROOK_MAP_SIZE = 102_400`,
  `BISHOP_MAP_SIZE = 5_248` (`build/magics.rs:73,143`). There is no PRNG, no
  Carry-Rippler search, no epoch-array verification loop anywhere in the
  shipped repository. The magics were evidently found once (by some external
  or historical process not present in this checkout) and simply pasted in as
  data.
- `build/maps.rs` — takes those fixed magics and, for each square, enumerates
  every occupancy subset of the mask with the standard Carry-Rippler trick
  (`occupancies = occupancies.wrapping_sub(mask) & mask`, `build/maps.rs:61`,
  same formula as SF and our doc §3.6 step 3) and writes the true attack set
  into a flat `Vec<u64>` at `magic_index(occ, entry)` (`build/maps.rs:50-66,
  72-76`) — the classic `((occ & mask) * magic) >> shift + offset` formula,
  byte-for-byte the same shape as ours and SF's. It also builds the leaper
  tables (`generate_king_map`/`generate_knight_map`, king/knight only —
  pawns get their own two-row table) and the geometry tables described in §3.

`build/build.rs::write` (`build/build.rs:75-99`) then calls each generator and
`writeln!`s the *result*, not the algorithm, into a generated file: a
`static NAME: [Type; N] = [...]` literal built from Rust's `{:?}` (`Debug`)
formatting of the array, one `write_map!` invocation per table (`DIAGONALS`,
`KING_MAP`, `KNIGHT_MAP`, `PAWN_MAP`, `RAYPASS`, `BETWEEN`, `ROOK_MAP`,
`BISHOP_MAP`, `ROOK_MAGICS`, `BISHOP_MAGICS`, plus a duplicate `MagicEntry`
struct def so the generated file is self-contained). The file lands at
`$OUT_DIR/lookup.rs` and `src/lookup.rs:3` pulls it in with
`include!(concat!(env!("OUT_DIR"), "/lookup.rs"))`.

Net effect: **the ~841 KB rook+bishop attack table, plus every leaper/geometry
table, is compiled straight into `.rodata` as source-level literals**. There
is no `LazyLock`, no runtime initialization function, and — this is the
trade-off's sharp edge — **no self-verification against the naive oracle at
either build time or run time**. `generate_slide`/`generate_ray` exist only to
*populate* the baked tables during the build; nothing ever re-checks a baked
entry against them again. Compare this directly to our
`design/03-attack-tables.md` §3.9, "Alternative documented but not chosen:
hardcoded magic constants" — Reckless is living proof of exactly the
alternative we wrote up and declined, including the exact trade-offs we
predicted (zero startup cost and no `LazyLock`, versus losing the "a subtle
table bug cannot survive a run" property and taking on generated-source-text
bloat). Nothing here suggests our §3.9 call was wrong; it's just useful to see
a top-tier engine choosing the other fork.

## 2. `src/lookup.rs`: the runtime accessor layer

Thin and entirely `unsafe`-indexed. Every accessor does `unsafe { *TABLE
.get_unchecked(idx) }` (`src/lookup.rs:70,86,90,94,102,107,111,119`) — no
bounds checks anywhere on the hot path, unlike our current safe-indexing
default (`design/03-attack-tables.md` §3.8, deferred behind D14(c) pending
profiling). The magic lookup is the same three-op formula as ours and SF's:

```text
index = ((occ & mask) * magic) >> shift + offset      // src/lookup.rs:127-131
```

`attacks(piece, square, occ)` (`src/lookup.rs:73-83`) dispatches by
`PieceType`, queen = rook∪bishop OR, exactly SF/our shape. One extra piece of
state shares this file for unrelated reasons: a Zobrist-cuckoo table for
repetition/upcoming-repetition detection (`CUCKOO`/`A`/`B`: three raw
`static mut` arrays of `0x2000` entries, filled once by `init_cuckoo` via
open-addressing with two hash functions `h1`/`h2`, `src/lookup.rs:5-59`). It's
plumbing for a different subsystem (repetition detection, not attack
generation) but is worth flagging structurally: Reckless's `lookup` module
mixes "precomputed attack tables" and "cuckoo hash for repetition" in one
file/namespace, where our design keeps attack tables (`design/03`) and
position-state/repetition (`design/04`) in separate modules. Also notable
purely as a Rust-safety data point: this is genuine unguarded `unsafe fn` +
`static mut`, initialized once from a single-threaded `initialize()` call —
the kind of pattern our conventions (newtypes, no unsafe without a measured
reason) would avoid; not a bug, just a looser safety posture than ours.

## 3. Geometry tables actually shipped — a smaller set than SF/ours

Reckless ships noticeably fewer geometry tables than Stockfish or our design
doc, and gets away with it by choosing single-purpose formulas:

| Table | Shape | Reckless builder | Our/SF equivalent |
|---|---|---|---|
| `KING_MAP`, `KNIGHT_MAP` | `[u64;64]` | `maps::generate_king_map/knight_map` (`build/maps.rs:3-9`) | `KING_ATTACKS`/`KNIGHT_ATTACKS` — same |
| `PAWN_MAP` | `[[u64;64];2]` | `maps::generate_pawn_map` (`build/maps.rs:28-33`) | `PAWN_ATTACKS` — same |
| `RAYPASS` | `[[u64;64];64]` | `maps::generate_rays_map`, non-`between` variant of `generate_ray` (`build/maps.rs:24-26`, `build/attacks.rs:57-66`) | our `RAY_PASS` — same defining shape |
| `BETWEEN` | `[[u64;64];64]` | `maps::generate_between_map` (`build/maps.rs:20-22`) | our `BETWEEN` (also OR'd with the target square) — same |
| `DIAGONALS` | `[[u64;64];2]` (**not** 64×64) | `maps::generate_diagonal_tables`, unblocked ±9 and ±7 rays (`build/maps.rs:35-40`) | nothing directly equivalent — see below |
| — no `LINE` table | | | we build one (`design/03` §3.4) |
| — no `DISTANCE` table | | | we build one (4 KB, Chebyshev) |

Two simplifications stand out:

- **No `LINE`, no `aligned()`.** Pin-move legality in Reckless is
  `self.pinned(stm).contains(from) && !ray_pass(king, from).contains(to)` ⇒
  illegal (`src/board.rs:380`, and again for castling-rook pins at
  `src/board.rs:370`). `ray_pass(s1, s2)` is *directional* — the ray from `s1`
  exclusive, through `s2`, to the edge on `s2`'s far side (§3 defining formula
  in `build/attacks.rs:57-66`, same construction as our `RAY_PASS`) — so it
  only accepts squares on the king→piece→beyond side, never the opposite side
  of the king. One table does double duty for both SF's classic "ray-pass"
  use (discovered-attack bookkeeping) and pin legality, where SF and our
  design instead reach for the *symmetric* `LINE` table (`aligned(from, to,
  ksq)`, our `design/05-movegen.md` §3.8). This works safely for Reckless
  specifically because `Board::is_legal` (`src/board.rs:349`) is a **from-scratch
  validator** called directly on a raw candidate move (TT-move validation at
  `src/movepick.rs:49`, an alternate perft mode at `src/tools/perft.rs:117`) —
  it cannot assume the move already respects blocker occupancy, so it needs
  the *directional* check to correctly reject a hypothetical "move to the far
  side of the king" candidate that a real move generator would never have
  produced (the king itself blocks that ray). Our `legal(m)` gets away with
  the symmetric, bidirectional `aligned()` check only because it is documented
  and enforced to run *after* `pseudo_legal(m)` (`design/05-movegen.md` §3.8,
  §3.10) — pseudo-legal generation already stops a slider at the king, so the
  "wrong side" square is never a candidate `to` in practice. Both are correct;
  Reckless's version is correct *standalone*, ours is correct *by
  composition*. See "Adapt to us" item 4 for what this suggests we could
  tighten.
- **No `DISTANCE` table.** Chebyshev distance is instead a tiny runtime
  function, `Square::distance_from` = `max(|Δfile|, |Δrank|)`
  (`src/types/square.rs:63-66`), computed on demand rather than a precomputed
  4 KB `[[u8;64];64]`. A legitimate "just compute it" alternative to our
  table — it's only ever a couple of subtractions and a max, and is called
  rarely enough (not in the movegen inner loop) that a table buys nothing.
- **`DIAGONALS`/`relative_diagonal`** (`src/lookup.rs:69-71`) is a small,
  specific trick: index 0 is the unblocked "/"-diagonal (±9) through a
  square, index 1 the "\"-diagonal (±7) (`build/maps.rs:35-40`). It is
  indexed by a `Color` value used purely as a `{0,1}` selector, not as an
  actual side — the one caller, pinned-pawn-capture generation, builds
  `[relative_diagonal(stm, king_sq), relative_diagonal(!stm, king_sq)]`
  (`src/board/movegen.rs:146`) to get "the two king-diagonals" as a 2-element
  pin-mask array for diagonal pawn captures, without needing a per-move
  `aligned`/`line` lookup. Repurposing the `Color` enum as a generic boolean
  index is a little loose typewise but costs nothing and reads fine at the
  call site.

## 4. No PEXT, same table sizes as SF and us

Table sizes are the textbook fancy-magic numbers and match both SF and our
own doc exactly: rook `0x19000 = 102,400` entries / bishop `0x1480 = 5,248`
entries (`build/magics.rs:73,143`; our `design/03-attack-tables.md` §3.5 cites
the identical numbers). Nothing distinctive here — this is just what "edges
excluded from the mask" converges to for every fancy-magic implementation.
Reckless has no PEXT/BMI2 path at all in the files studied; the only
CPU-feature branching in this subsystem is the AVX2/AVX512 SIMD in
`setwise.rs` (§6) and an AVX512VBMI2 move-list-write trick (§7), both
unrelated to slider *indexing*. This matches our own §3.8 note that current
Stockfish also dropped PEXT from its default path in favor of the plain
multiply-and-shift magic. `Makefile:8-13,49-53` (`x64-check`) shows the
project cares about x86-64-v1..v4 buildability, including `+gfni,+avx512bw,
+avx512vl,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512bitalg` — all consumed
by the NNUE SIMD and the move-list trick, not by attack indexing.

## 5. `setwise.rs`: parallel-prefix ("fill") attacks for a whole army at once

This is the one genuinely new capability relative to both Stockfish and our
engine. "Setwise" here means: given a **bitboard of several same-type
pieces** (not a single square) plus the board occupancy, compute the union of
every one of those pieces' attacks in a handful of whole-bitboard operations,
instead of looping per-piece and doing N separate magic lookups.

- **Pawns and knights** need no fill — a piece's reach is one step, so the
  setwise version is just the leaper shift pattern applied to the *whole*
  source bitboard at once (`pawn_attacks_setwise`, `knight_attacks_setwise`,
  `src/setwise.rs:14-38`): mask off the two wrap-prone files, shift the whole
  bitboard by each of the 2 (pawn) or 8 (knight) step directions, OR
  together. Nothing algorithmically new — same trick our own
  `pawn_attacks_bb` already does for pawns (`design/03-attack-tables.md`
  §3.3); Reckless just extends the same idea to knights.
- **Bishops and rooks** need the real trick: a **Kogge-Stone-style occluded
  fill**, computed in our own words as:

  ```text
  gen  = starting piece bitboard (the whole set, one shot)
  pro  = empty squares, masked per direction to forbid file/rank wraparound
  for dist in [1, 2, 4]:                    // doubling reach: 1+2+4 = 7
      gen |= pro & shift(gen, dist)
      pro &= shift(pro, dist)
  attacks = shift(gen, 1) & wrap_mask       // one more step off the fill, masked
  ```

  run independently for each of the piece's 2 (rook: N/S or E/W collapse to
  4 total; see below) or diagonal directions, then unioned. This is the
  classic "Dumb7Fill"/Kogge-Stone occluded-fill algorithm (well known from
  chess programming folklore) rather than anything magic-index based — it
  trades one magic lookup per piece for O(log 8) shift/AND/OR steps applied
  to the *whole* bitboard regardless of how many pieces are set.
- **The AVX2 vectorization is the actual novelty.** `bishop_attacks_setwise`/
  `rook_attacks_setwise` (`src/setwise.rs:52-104`) run all 4 directions of a
  slider type *simultaneously*, one direction per 64-bit lane of a single
  `__m256i`: the starting bitboard is broadcast into all 4 lanes
  (`_mm256_set1_epi64x`), and a per-direction edge mask is packed into a
  second `__m256i` (`_mm256_set_epi64x`, four different file/rank exclusion
  masks, one per lane). The fill loop above becomes 3 rounds of
  `_mm256_and/or_si256` plus one `shiftv` call per round, where `shiftv<A,B,C,D>`
  (`src/setwise.rs:106-117`) shifts each of the 4 lanes by its *own*,
  possibly-negative, compile-time constant amount by OR-ing a variable
  left-shift (`_mm256_sllv_epi64`) with a variable right-shift by the negated
  amounts (`_mm256_srlv_epi64`) — since exactly one of the two shifts is
  non-degenerate per lane (a negative shift-left-by-huge-count is zero on
  x86, so only the correctly-signed term survives), one call handles all 4
  independent per-direction shift directions at once. Where AVX512F/AVX512VL
  is available, the same helper collapses to a single **rotate**
  (`_mm256_rolv_epi64`, `src/setwise.rs:119-127`) — cheaper than shift+OR,
  correct here specifically because the wrap-mask applied right after
  discards whatever garbage a rotate carries around the 64-bit lane, so
  "rotate then mask" is observably identical to "shift then mask" for this
  use. `fold_to_bitboard` (`src/setwise.rs:129-137`) collapses the 4 lanes
  back to one `u64` with two 128-bit ORs and an extract. Compile-time gated:
  a plain scalar fallback (`#[cfg(not(target_feature = "avx2"))]`, `src/setwise.rs:
  40-50,73-83`) just loops over each source square, does an ordinary
  single-square magic lookup (`lookup::bishop_attacks`/`rook_attacks`), and
  ORs the results — i.e., on a non-AVX2 target, "setwise" degrades to exactly
  the loop we'd write by hand with our own `attacks()`, no algorithmic gain.

## 6. What the setwise attacks are *for*: a precomputed whole-board threat map

This is the architecturally interesting part, and the reason `setwise.rs`
exists at all. `Board::update_threats` (`src/board.rs:431-461`) runs once per
node, right after a move is made, and computes, per opposing piece type, the
**union of every attack that piece type currently makes anywhere on the
board**:

```text
occ_no_king = occupancy with the side-to-move's own king square cleared
              (lets enemy sliders "see through" the king — see below)
threats[Pawn]   = pawn_attacks_setwise(enemy pawns, enemy color)
threats[Knight] = knight_attacks_setwise(enemy knights)
threats[Bishop] = bishop_attacks_setwise(enemy bishops, occ_no_king)
threats[Rook]   = rook_attacks_setwise(enemy rooks, occ_no_king)
threats[Queen]  = bishop_attacks_setwise(enemy queens, occ_no_king)
                | rook_attacks_setwise(enemy queens, occ_no_king)
threats[King]   = king_attacks(enemy king square)
all_threats     = OR of all six
```

stored directly in `StateInfo` (`piece_threats: [Bitboard; PieceType::NUM]`,
`all_threats: Bitboard`, `src/board.rs:35-36`). The king-exclusion trick
(`src/board.rs:432-439`, comment preserved in spirit) is worth restating
precisely: computing enemy slider attacks with *our own* king removed from
occupancy lets a slider's ray "pass through" where the king stands, which is
exactly the extra reach needed to correctly forbid the king from stepping
*along* the checking ray — a well-known idea (SF's `pos.attackers_to` for king
moves does the same occupancy trick, just per-square rather than precomputed
setwise) but the comment there is worth citing since it flags that this
changes the *bitboard* without changing legal-move behavior (squares behind
the king are never legal targets anyway).

Two, quite different, consumers read this precomputed map:

1. **Legality, O(1) instead of a query.** King steps: `(king_attacks(from) &
   !self.all_threats()).contains(to)` (`src/board.rs:375-376`); castling:
   `(castling_threat[kind] & self.all_threats()).is_empty()`
   (`src/board.rs:369`); non-king legality still falls back to a genuine
   attacker query only for the check/pin/en-passant cases that need
   occupancy *at the destination* (`attackers_to`, `src/board.rs:340-346`, and
   the ad hoc en-passant discovered-check probe starting at `src/board.rs:389`,
   both single-square, not setwise). The distinctive move is specifically for
   *king* legality and castling: instead of asking "is square X attacked?"
   once per candidate king destination (what SF's `pos.legal` and our
   `Board::legal`'s `attackers_to_exist` branch both do,
   `design/05-movegen.md` §3.8, `_ if from == ksq => !self
   .attackers_to_exist(to, occ ^ from, them)`), Reckless answers *every*
   king-destination legality question for the whole node with one bitmask
   test against a value computed once. The cost moves from "up to 8 attacker
   queries per node, only if the king actually has 8 pseudo-legal steps" to
   "always compute one whole-board threat map, then O(1) per candidate" — a
   different amortization trade that only pays off because the same map is
   *also* reused for move ordering (next point), otherwise it would just be
   moving the same total work earlier.
2. **Move-ordering features, for free.** `movepick.rs`'s quiet-move scoring
   builds `threatened`/`offense` bitboard arrays straight out of the same
   `piece_threats` (`src/movepick.rs:150-186`): "is my piece of type X
   currently standing on a square attacked by a lower/equal-ranked enemy
   piece" (`threatened`, weighted `-8875` in the score formula) and "does
   this move land on a square where I'd be attacking an enemy piece that
   isn't already defended" (`offense`, weighted `+3446`), on top of the
   existing history/continuation-history terms. Nothing conceptually new
   versus the "is this move a threat/escape" heuristics chess engines have
   used for a long time — the notable part is that it falls out essentially
   free of the setwise map that legality already computed, rather than being
   its own bespoke per-move attacker query.

Do not confuse this with `nnue/accumulator/threats*` — that is a *separate*,
incrementally-maintained set of NNUE input features (threat encodings for the
network), unrelated to `setwise.rs`/`update_threats`, out of scope for this
note (attack generation, not NNUE features).

Neither Stockfish nor our engine currently computes anything like
`all_threats`/`piece_threats` — both stay purely lazy/on-demand
(`attackers_to`/`attackers_to_exist`, queried per candidate). This is the
single most transferable idea in the file.

## 7. A tangential but related bulk technique: vectorized move-list writes

Not attack generation, but directly downstream of a setwise-shaped bitboard
(a piece's `to`-square set) and worth a one-line flag since it shows the same
"operate on the whole bitboard, not bit-by-bit" philosophy extends into
movegen: `MoveList::push_setwise` has an `avx512vbmi2` path
(`src/types/movelist.rs:40-67`) that expands a 32/64-bit `to`-square mask
directly into up to 32 packed `Move`s per 512-bit register in one masked
store, instead of the scalar `for to in to_bb { push(...) }` loop
(`src/types/movelist.rs:33-38`) used everywhere else. Server-CPU-only
(`avx512vbmi2`), and out of this note's scope, but the same instinct as
`setwise.rs`: prefer one wide instruction over an N-iteration bit loop
whenever N could be large.

## 8. Rust-idiom differences worth naming (not corrections, just data points)

- `Bitboard` implements `Iterator` directly (`src/types/bitboard.rs:68`,
  presumably a pop-lsb `next`), where our convention deliberately keeps sets
  out of `Iterator` and gives them `IntoIterator` + a dedicated iterator type
  (`CLAUDE.md` "Conventions"). Different, already a considered choice on our
  side — no action, just noted for contrast.
- Every hot lookup in `lookup.rs` uses `get_unchecked` unconditionally; we
  ship safe indexing until a profile justifies `unsafe` (D14(c)). Reckless
  takes the opposite default everywhere in this module.
- `Square` is a 65-variant `#[repr(u8)]` enum built via `unsafe {
  mem::transmute(value) }` in its constructor (`src/types/square.rs:30-33`),
  versus our newtype-over-`u8` with an `assert!`-guarded `const fn new`
  (`src/types.rs:222-225`) — same invariant, different mechanism; ours needs
  no `unsafe` for the equivalent guarantee.

## Adapt to us

Ranked roughly by impact; each names the owning doc/section and rough phase.

1. **Setwise (whole-army) slider/leaper attacks as a new `attacks` primitive**
   — a scalar `rook_attacks_bb(pieces: Bitboard, occ: Bitboard) -> Bitboard`
   (loop + OR of per-square magic lookups) plus, if profiling ever justifies
   it, an AVX2-vectorized fill version. Doc: `design/03-attack-tables.md` §1
   currently defers "set-wise movegen shift helpers beyond `pawn_attacks_bb`"
   to movegen — this would need either a scope amendment there or to land
   directly in `movegen`/`eval`. Phase: infra, usable whenever; the payoff is
   really idea 2. Effort: low for the scalar version (an afternoon); medium-high
   for a hand-vectorized AVX2 fill (a day-plus, needs its own criterion bench
   and correctness cross-check against the naive/magic path). Impact:
   moderate on its own, enabling for idea 2.

2. **Precompute a per-node opponent threat map and use it for king/castling
   legality (the core idea of §6).** Concretely: after `make_move`, compute
   `all_threats` (union of setwise attacks of every enemy piece type, with the
   side-to-move's king excluded from occupancy exactly as Reckless does) as a
   `StateInfo`-resident field, then change the king-move and castling arms of
   `Board::legal` (`design/05-movegen.md` §3.8) from per-candidate
   `attackers_to_exist` calls to a single `Bitboard` AND/contains test. Doc:
   `design/04-position-state.md` (new `StateInfo` field + incremental
   maintenance cost — needs its own accounting, since it's now eagerly
   computed every node, not just for check detection) and
   `design/05-movegen.md` §3.8. Phase: touches `movegen`/`board` mutation, so
   the full D24 perft ritual applies before it can land, and — because it's
   search-behavior-adjacent — the `Bench:` trailer discipline applies too
   (must be node-count-neutral if done correctly: it changes *cost*
   distribution, not the *set* of legal moves, so perft counts and search
   node counts at fixed depth should reproduce bit-for-bit; any change means a
   bug, exactly like the "attack tables are functionally invisible" rule in
   `design/03-attack-tables.md` §5.4). Effort: medium-high (careful
   correctness work + mandatory perft/bench verification). Impact: uncertain
   until measured — likely a small nps win in positions with many king
   pseudo-moves or many pinned pieces (removes repeated slider queries), but
   the bigger payoff is idea 3 riding on the same computed value.

3. **Threat-aware move-ordering features** (Reckless's `threatened`/`offense`
   arrays, §6 point 2) built from the map in idea 2: "is the piece on this
   move's `from` square currently hanging", "does this move land on/create an
   attack on an undefended enemy piece". Doc: `design/09-move-ordering-history.md`
   §3.4 (table catalogue) / §3.7 (scoring formula) — a new orthogonal
   ordering term alongside the existing butterfly/capture/continuation/pawn/
   low-ply history stack (P8, already landed, D17). Phase: P9 (pruning/move-
   ordering is squarely in scope for the current wave) or immediately after
   idea 2 lands. Effort: low once idea 2 exists (the bitboards are already
   computed; this is scoring-formula plumbing + weight tuning). Impact: worth
   a fixed-node proxy SPRT per `design/14-performance-testing.md` §9.4 before
   committing to a real-TC run — Reckless's own score formula gives these
   terms non-trivial weights (`-8875`/`+3446` alongside history terms in the
   thousands), suggesting real signal, but our history stack already differs
   enough (no killers, D17) that the exact weights won't transfer — only the
   feature *shape* should be ported, then re-tuned.

4. **Harden `Board::legal`'s pin-move check to be directional (`ray_pass`-
   style) instead of symmetric (`aligned`/`LINE`-style), matching §3's
   analysis.** We already build `RAY_PASS` (`design/03-attack-tables.md`
   §3.4); the fix is swapping which table the pin-legality clause in
   `design/05-movegen.md` §3.8 consults, from `aligned(from, to, ksq)` (LINE-
   backed, bidirectional) to `ray_pass(ksq, from).contains(to)` (directional).
   Currently harmless *only* because `legal()` is documented to run strictly
   after `pseudo_legal()` everywhere it's called (§3's analysis) — this
   change would make `legal()` correct standalone, which matters if a future
   caller (e.g., a from-scratch validator analogous to Reckless's `is_legal`,
   useful for datagen/fuzzing raw moves) ever calls it without that
   precondition. Doc: `design/05-movegen.md` §3.8 (note the change and why in
   the doc per the "docs follow decisions" rule if adopted, since it doesn't
   change *decisions.md* but does change a specified formula). Phase: no
   fixed phase — a correctness-hardening cleanup, cheap enough to fold into
   any movegen touch, but touches `movegen` so D24 perft applies (should be
   perfectly perft-neutral: same legal moves, same table already built,
   different predicate over it). Effort: very low (one-line swap + rationale
   in the doc + full perft run to confirm no behavior change). Impact: low
   probability of ever mattering given current call sites, but essentially
   free to fix and removes a documented "only safe by composition" caveat.

5. **Build-time-baked magic tables (build.rs codegen + `include!`), i.e. the
   mechanism Reckless actually uses (§1), matched against our own
   `design/03-attack-tables.md` §3.9 "alternative not chosen."** Not a
   recommendation to switch — our stated rejection reasons (loses the
   per-launch self-verification against the naive oracle; ~841 KB of
   generated source text) are exactly the trade-offs Reckless is actually
   living with (no verification anywhere in its repo, confirmed in §1). This
   entry is here only as confirmation that our §3.9 call was a real,
   deliberate fork in the road and not an oversight — worth revisiting only
   if the ~21 ms startup search (`design/03-attack-tables.md` §6) ever
   actually shows up as a problem (e.g., extremely short-TC gauntlet/SPRT
   harnesses launching many processes). Phase: none scheduled. Effort:
   low-medium if ever undertaken (mechanical, well-trodden Cargo pattern).
   Impact: low/negative expected (worse verification posture) unless startup
   latency becomes a measured problem.

## License note

Reckless is AGPLv3 (`reckless_reference/Reckless/LICENSE`). Everything above
is description and independently-restated formulas/pseudocode, not copied
source; file:line references are pointers for a future reader to re-find the
original, not excerpts. No Reckless code should be pasted into our engine —
any of the ideas above are to be reimplemented from scratch against our own
types and design docs.
