# Integral: board representation, move generation, magics/PEXT

Source: `integral_reference/integral/src/chess/{bitboard,board,move,move_gen,fen}.{h,cc}`,
`integral_reference/integral/src/magics/{attacks,entry,magic_finder,precomputed}.{h,cc}`,
`integral_reference/integral/shared/multi_array.h`, `integral_reference/integral/src/utils/{types,list}.h`,
plus the git history of those paths (`git -C integral_reference/integral log --oneline`).
Version studied: the checked-out clone, HEAD at 669 commits, PR #208 (BMI2) at
`839c3edc`. **No license file in the repo → all rights reserved.** Everything
below is prose/pseudocode written from scratch after reading the source; no
Integral code is reproduced. File:line and commit-hash citations are pointers
for a future reader to re-find the original, not excerpts.

Read together with `stockfish_reference/01-bitboards-attacks-move-encoding.md`,
`reckless_reference/02-attacks-magics-setwise.md`, `reckless_reference/03-board-makemove-fen.md`,
`reckless_reference/04-movegen-see.md`, and our own `design/02-core-types.md`,
`design/03-attack-tables.md`, `design/04-position-state.md`, `design/05-movegen.md`.

## 1. Board representation

`BoardState` (`src/chess/board.h:87-278`) is a flat, copyable struct — not a
`Board`-owned incremental stack entry the way our `StateInfo` is (design/04
§3). It holds, in one aggregate:

- `piece_bbs: [BitBoard; 6]` (indexed by `PieceType`, `kNone` excluded —
  `kNumPieceTypes = kNone = 6`, `src/utils/types.h:24-33`) and
  `side_bbs: [BitBoard; 2]` — the same two-dimension-collapsed-to-two-arrays
  bitboard shape as ours (`pieces[color][piece_type]` vs. their
  `piece_bbs[piece_type] & side_bbs[color]` computed on demand via accessors
  like `Pawns(Color)`, `board.h:205-251`) and as SF's `byTypeBB`/`byColorBB`.
  No combined `[color][piece_type]` table — every per-color-per-type query is
  an AND of the two arrays, done at every call site (`Pawns(us)`,
  `Knights(us)`, …), not cached.
- `piece_on_square: [PieceType; 64]` — a mailbox, exactly parallel to our
  `mailbox: [Option<Piece>; 64]`, except color is a *separate* lookup
  (`GetPieceColor`, `board.h:169-173`, tests `side_bbs[White].IsSet` then
  `side_bbs[Black].IsSet`) rather than baked into one `Option<Piece>` per
  square the way ours is. `GetPieceAndColor` (`board.h:179-182`) does
  `piece*2 + color` to get a single 0..11 "colored piece" index — used
  directly as the Zobrist piece-table index (`zobrist::pieces[colored_piece][square]`,
  `board.h:113` etc.) and matches the classic SF/our convention of one
  combined piece-color axis for the hash tables.
- Five Zobrist-family keys, same idea as our five-key family (design/04 §3.3):
  `zobrist_key` (full), `pawn_key`, `minor_key`, `major_key`,
  `non_pawn_keys: [U64; 2]` (per-color). No dedicated king-only or
  castling/ep-only key beyond what's folded into `zobrist_key` — compare our
  design's ep/castling/side keys, which Integral instead XORs directly into
  the *single* `zobrist_key` inline in `MakeMove`/`SetFromFen`
  (`board.cc:239-246,267-269,272`), not into their own key streams. Keys are
  maintained incrementally inside `PlacePiece`/`RemovePiece`
  (`board.h:103-167`, template `bool update_key = true` parameter to allow a
  key-less fast path) — i.e., **placement mutation and key maintenance are
  fused in one call**, unlike our explicit split (`put_piece`/`remove_piece`
  are placement-only; Zobrist deltas live in `make_move`, CLAUDE.md "Core
  encoding decisions"). Integral's fusion is simpler to write but means any
  future placement-only helper (e.g. a `set_from_fen`-only path that doesn't
  want key churn) must thread the `update_key` template bool through by hand;
  ours keeps that concern out of the mutator entirely.
- `checkers`, `pinned: [BitBoard; 2]`, `threats`, `threatened_by: [BitBoard; 6]`,
  `check_zones: [BitBoard; 4]` — all derived, all recomputed from scratch
  every `MakeMove`/`MakeNullMove` via `CalculateThreats`/`CalculateKingThreats`
  (`board.cc:511-593`), not incrementally patched. See §3.5 below — this is
  directly comparable to our own D30 threat-map decision.

**The undo model is "copy-full-state", not incremental make/unmake.**
`Board::MakeMove` (`board.cc:204-281`) pushes a *copy* of the entire
`BoardState` onto `history_: List<BoardState, 2048>` (`board.h:347`) before
mutating `state_` in place; `UndoMove` (`board.cc:283-286`) simply
`state_ = history_.PopBack()`. There is no move-reversal logic at all — no
"put the captured piece back", no "restore the pawn that was en-passant'd";
the entire struct (six bitboard-ish arrays, a 64-byte mailbox, ~10 scalar
fields, 4+ Zobrist keys, `threatened_by`/`check_zones`) is byte-copied twice
per ply (once to push, once to pop). `BoardState` is roughly 300+ bytes
(`piece_bbs` 48 B + `side_bbs` 16 B + `piece_on_square` 64 B + `threatened_by`
48 B + `check_zones` 32 B + keys/scalars ~70 B), so a max-depth 2048-ply
`history_` is ~600 KB+ of stack per `Board` object — this is the "no unmake
code, more memory/copy bandwidth" fork in the road, the mirror image of our
own incremental `StateInfo` stack (D8) and of SF's incremental make/unmake.
Integral's choice trades a large, fixed, node-independent memory/bandwidth
cost for **zero unmake bugs by construction** (nothing to get backwards —
undo is definitionally correct if make was). Given lazy SMP (our P11, just
landed) multiplies this cost by thread count, this is a real reason *not* to
adopt the pattern wholesale, but it's worth naming explicitly as the
alternative our design deliberately didn't take.

`CastleRights` (`board.h:23-85`) is a `U8` bitmask with per-color/per-side
masks (`kMasks[color][kingside/queenside]`), same shape as our
`CastlingRights` newtype-over-bits; nothing new here.

## 2. Move encoding

`Move` (`src/chess/move.h`) is a packed `U16`, same overall shape as ours
(SF-style) but with the two 2-bit fields in the **opposite order**:

```text
Integral:  bits 0-5 from | bits 6-11 to | bits 12-13 MoveType | bits 14-15 PromotionType
Ours:      bits 0-5 from | bits 6-11 to | bits 12-13 promotion piece | bits 14-15 kind
```

`MoveType` is `{Normal, Promotion, Castle, EnPassant}` (`move.h:19-24`);
`PromotionType` is `{Knight=0, Bishop=1, Rook=2, Queen=3}` (`types.h:35-40`) —
the promotion-piece *values* match ours exactly (0=knight…3=queen per
CLAUDE.md), only the field order in the packed word differs, which is
functionally irrelevant (both are internal-only encodings, never serialized
raw). Castling is encoded as the king's own two-square move (e1g1-style),
identical to our D2. `Move::IsCapture`/`IsNoisy`/`IsEnPassant` (`move.cc:68-78`)
are the same small helper trio ours has via `Move`/`Board` methods.

## 3. Move generation

### 3.1 Leaper/pawn tables

`kKnightMasks`, `kKingMasks`, `kPawnAttackMasks` (`move_gen.cc:9-57,125-127`)
are `constexpr`-built `[BitBoard; 64]` (and `[2][64]` for pawns) tables —
same shape and same "build with compile-time shift-and-mask" technique as our
`attacks::knight_attacks`/`king_attacks`/`pawn_attacks` (design/03 §3.2-3.3).
Nothing new: every engine in this comparison set (SF, Reckless, Integral, us)
builds these identically.

### 3.2 Slider lookup dispatch

`BishopMoves`/`RookMoves` (`move_gen.cc:176-184`) just forward to
`magics::attacks::Get{Bishop,Rook}AttackIndex` + a table index — see §4.
`QueenMoves` is `BishopMoves | RookMoves`, same "no queen magic table, OR the
two" convention as SF/Reckless/us.

### 3.3 Pseudo-legal generation, then a *separate* legality predicate

`GenerateMoves<MoveGenType>` (`move_gen.cc:448-506`) and
`Board::IsMoveLegal` (`board.cc:129-202`) are architecturally the same
two-phase split as SF's `generate<LEGAL>` decomposition and our own doc 05 §3
contract (`generate` produces pseudo-legal, `Board::legal` filters lazily,
called per move as the search actually visits it — confirmed at the call
sites `search.cc:375,817,893` and `move_picker.cc:179`, which generate once
per node then call `IsMoveLegal` per popped move, exactly our `MovePicker` /
`generate` + `Board::legal` shape). This is convergent, not novel — SF,
Reckless, Integral, and we all do "generate broad, filter lazily, only for
moves actually tried."

Where Integral's generator differs from SF's and ours in two concrete ways:

1. **Pinned knights are excluded at generation time, not filtered
   afterward.** The one micro-optimization Integral applies:
   `for (Square from : state.Knights(state.turn) & ~state.pinned[state.turn])`
   (`move_gen.cc:474`) — every *other* piece type (bishop/rook/queen/pawn) is
   generated unconditionally and left to `IsMoveLegal`'s pin/ray check
   (`board.cc:186-191`, same "not pinned, or moving along the king-piece ray"
   test as our `Board::legal` arm 4, `movegen.rs:535-537`). This is provably
   safe *only* for knights (a pinned knight's step is never colinear with the
   pin ray, so it can never be legal) and saves, per pinned knight, up to 8
   generated moves + 8 wasted legality probes + wasted move-ordering scoring
   that would otherwise be thrown away one-by-one. We do not currently do
   this (our knight loop in `movegen.rs:184` iterates
   `board.pieces(us, PieceType::Knight)` unfiltered) — see §5 item 1.
2. **Single-check evasions are NOT restricted to the block/capture ray at
   generation time.** Only the *double*-check case is special-cased
   (`move_gen.cc:460-469`: king moves only). For a *single* checker,
   `GenerateMoves` falls straight through into the ordinary
   knight/bishop/rook/queen/pawn loops with `targets` = "empty squares" ∪
   "enemy pieces" — the same targets as a non-check position — and relies
   entirely on `IsMoveLegal`'s tail (`board.cc:193-201`: "if not in check, or
   we can take the checking piece … return RayBetween(king, checker).IsSet(to)")
   to reject every generated move that doesn't block or capture. This is
   **strictly less restrictive at generation time** than SF's/our `EVASIONS`
   generation type, which ANDs every non-king piece's targets with
   `between(king, checker) ∪ {checker}` *during* generation (design/05 §3.4,
   `movegen.rs:198-217,301,313,332,347-351`) — our move lists in check are
   never bloated with doomed non-blocking moves in the first place. This is
   not something to adopt; it's a place where our design is already ahead of
   Integral's, and a useful data point that "generate everything, filter
   lazily" is a viable simplicity/performance trade even for a top-10 engine
   — just not the one we picked, and ours is strictly better here.

### 3.4 Checkers/pins/check-zones: the standard SF x-ray algorithm

`CalculateKingThreats` (`board.cc:548-593`) computes `checkers` and
`pinned[us]` with the textbook "super-piece from the king square, treating
only enemy pieces as blockers" x-ray technique:

```text
xraying = sliding_attacks_from(king_sq, occupied = enemy_pieces_only) & (enemy bishops/rooks/queens)
for each enemy slider s in xraying:
    between = our_pieces & ray_between(king_sq, s)
    if between is empty:       s is a direct checker
    elif |between| == 1:       that one piece is pinned
```

(`board.cc:566-582`, using `GetSlidingAttackersTo(state, king_sq, their_pieces, them)`
as the "friendly-transparent" super-piece query.) This is exactly the same
algorithm SF uses and the same one our `blockers_for_king`/pin detection uses
(design/04 §3.5) — confirmatory parity, not a new idea.

`check_zones[pt]` (`board.cc:584-592`) is SF's `checkSquares[pt]` concept by
another name: for each piece type, the set of squares from which *that* piece
type would give check to the opponent's king right now, recomputed every
node from the opponent king's square. `MoveGivesDirectCheck` (`board.cc:468-483`)
then becomes an O(1) `check_zones[piece].IsSet(to)` test instead of
re-deriving check status by simulating the move. **We already have the
identical mechanism** — `check_squares: [Bitboard; 6]` in `src/board.rs:185`,
populated at `board.rs:565-571`, consumed by `gives_check` at `board.rs:622-634`
— so this is pure confirmation we're aligned with both SF and Integral here,
no action needed.

### 3.5 Threat map: recomputed from scratch every node (3-way convergence)

`CalculateThreats` (`board.cc:511-546`) builds `threatened_by[pt]` (union of
attacks of every enemy piece of type `pt`) and `threats` (their union) by
looping per-piece and calling the scalar `PawnAttacks`/`KnightMoves`/
`BishopMoves`/`RookMoves` lookups directly, OR-ing results together — called
unconditionally at the end of *every* `MakeMove` and `MakeNullMove`
(`board.cc:277,306`), i.e., **never maintained incrementally**, always thrown
away and rebuilt. This is the same architectural bet Reckless makes with its
AVX2-vectorized `update_threats`/`setwise.rs` machinery
(`reckless_reference/02-attacks-magics-setwise.md` §6) — Integral's version is
the scalar, unvectorized cousin (a plain loop of single-square magic lookups
per enemy piece, no Kogge-Stone fill, no SIMD) — and matches what our own
design already commits to for the newly-landed Wave-G threat map (D30,
`design/10-pruning-extensions.md` "Wave G", `design/decisions.md` D30): "once
per `make_move`/`make_null_move`, build `all_threats` + `piece_threats[pt]`…
via whole-army setwise attack fills." **Three independently-written
codebases (Reckless, Integral, us) converge on "recompute the whole-board
threat map from scratch every node rather than maintain it incrementally"** —
strong evidence this is simply the right trade for this feature (incremental
maintenance across arbitrary from/to/capture/promotion deltas is fiddly and
bug-prone; a full recompute is cheap enough per node not to matter). If we
ever want the *vectorized* version of this idea, Reckless's setwise fill is
the model to study — Integral's is the plain scalar loop, offering nothing
beyond what we already do.

### 3.6 `GetAttackersTo`: no short-circuiting

`GetAttackersTo` (`move_gen.cc:233-263`) unconditionally computes all five
piece-type attack bitboards (pawn, knight, bishop, rook, king) and ORs them
before the final `& state.Occupied(attacker)` mask — no early-exit once a hit
is found. Our `attackers_to_exist` (`movegen.rs:448-461`) short-circuits via
`||` across the same five terms, sliders first (design/05 §3.8 rationale:
sliders answer the common king-safety probes most often). Minor, but worth
noting: Integral's function *always* returns the full attacker bitboard (it
has to — it's used both as a boolean check and to enumerate attackers, e.g.
`GetOpponentWinningCaptures`), so an early-exit boolean twin doesn't apply to
every call site the way ours does; not directly comparable 1:1, just noted.

### 3.7 `fen.cc`: no ep-legality validation

`fen::StringToBoard` (`fen.cc:70-77`) records whatever en-passant square the
FEN string states, unconditionally, with no check that a legal en-passant
capture actually exists from that square (contrast our D9 "legality-exact FEN
ep validation," CLAUDE.md status). A hand-crafted or malformed FEN with a
"phantom" ep square would hash and behave as if a real ep opportunity
existed. Not a correctness bug for normal play (FENs from legal games always
have a real ep opportunity when the field is set) but a smaller robustness
margin than ours for adversarial/fuzzed FEN input — noted for completeness,
not something to adopt.

## 4. Magics and PEXT

### 4.1 Plain (dense) magic bitboards, not fancy

`MagicEntry { mask, magic, shift }` (`src/magics/entry.h:8-12`) plus two
**fixed-size, per-square-dense** tables:

```text
kBishopAttacks: [[BitBoard; 512]; 64]     // 512 = 2^9, worst-case bishop relevant bits
kRookAttacks:   [[BitBoard; 4096]; 64]    // 4096 = 2^12, worst-case rook relevant bits (corner)
```

(`src/magics/attacks.h:9-18`, using the generic `MultiArray<T, Dims...>`
nested-`std::array` helper, `shared/multi_array.h`). **Every square's
subtable is allocated at the worst-case size for its piece type**, regardless
of how few relevant bits that particular square actually needs — this is the
"plain" magic scheme, as opposed to the "fancy" scheme both Stockfish and our
own `design/03-attack-tables.md` §3.5 use, where every square's subtable is
sized to exactly `2^popcount(mask)` and packed contiguously into one shared
array via a per-square offset (our doc's own words: "as opposed to 'plain'
magics where every square gets a worst-case-sized 4096-entry block" —
`design/03-attack-tables.md:350-351` — Integral is *the* worked example of
that alternative). Net memory: 64×4096×8 B = 2 MiB (rook) + 64×512×8 B =
256 KiB (bishop) ≈ 2.25 MiB total, versus our fancy layout's ~841 KiB
(102,400 rook + 5,248 bishop entries, `design/03-attack-tables.md:9-11` /
`reckless_reference/02-attacks-magics-setwise.md` §4, same numbers for
Reckless). On modern multi-MiB L3 caches this size difference is unlikely to
move nps much either way (this exact point — fancy magics mattered far more
in the 512 KB–1 MB L2-cache era than on today's hardware — is well-known
chess-programming folklore); nothing here suggests switching our approach,
it just situates Integral as the "plain and simple, memory is cheap" fork
among this comparison set (SF: fancy; Reckless: fancy, build-time-baked;
us: fancy, `LazyLock`-searched at startup; Integral: plain, generated once at
process load via a global-variable initializer).

### 4.2 Magic search: brute-force random, no epoch-array optimization

`magics::finder::FindMagic` (`src/magics/magic_finder.cc:32-68`) is the
textbook approach: enumerate every blocker subset of the square's mask via
the Carry-Rippler trick (`subset = (subset - 1) & moves`, same formula our
own doc and Reckless's `build/maps.rs` use), generate the true attack set for
each subset with a naive ray-walker (`SlidingAttacks<Dir>`,
`attacks.cc:35-48`, stops at the first blocker per direction — structurally
identical to SF's `sliding_attack` and our own doc's oracle), then try
candidate magics — up to 1,000,000 attempts per square, `and`-ing three
64-bit `mt19937_64` draws together per candidate (`random_magic_number() &
random_magic_number() & random_magic_number()`, `magic_finder.cc:58-59`) —
the same "sparse random" trick (fewer set bits → better magic candidates,
well-known folklore) our own `design/03-attack-tables.md` §3.6 popcount
prefilter is built on, just without the popcount pre-filter step itself
(Integral relies on the 3-AND alone to produce sufficiently sparse
candidates). `TryMagic` (`magic_finder.cc:13-30`) allocates a fresh
`std::vector<BitBoard>` of `1 << shift`-inverse size and memsets it via
default-construction on *every single candidate attempt* — no epoch/generation
array to avoid the repeated clear the way our own doc's §3.6 step 3 epoch
trick does (`design/03-attack-tables.md:487-530`). Since good magics are
found within a few thousand tries in practice, this is a one-time,
offline-feeling cost (the magics in `precomputed.h` are baked as `constexpr`
data, `magic_finder.cc`/`GenerateMagics` is a generator tool, not part of the
shipped engine's runtime path) — a correctness/simplicity-over-micro-efficiency
choice that's fine precisely because it never runs in the shipped binary.

### 4.3 PEXT: opt-in, gated by a CPU-family exclusion list — PR #208, `839c3edc`

Timeline (from `git log`, all dates from `git show -s --format=%ci`):

| Date | Commit | Milestone |
|---|---|---|
| 2023-10-25 | `8d6ed33b` | initial commit |
| 2024-03-16 | `6da6e3da` | "implemented magic bitboards" (first magic table, ~4.5 months in) |
| 2024-03-16 | `c8d3453c` | "piece bitboard rehaul" (same day — mailbox/bitboard split solidified) |
| 2024-07-23 | `c4f136e3` | "Move Generation/Make Move Rehaul" (PR #18; also the point the tree reorganizes into `src/chess/`, `src/engine/`, `src/magics/` — the layout studied here) |
| 2025-06-19 | `839c3edc` | "Utilize BMI2 instructions for attack table lookups" (PR #208, `Bench: 2021017`) |

So PEXT arrived **~20 months and 208 PRs after project start**, as a small,
self-contained, four-file diff (`839c3edc`, +115/−48 lines total) — added
long after the engine was already strong, not part of the original design.
The change itself:

- `attacks.cc` grows `GetBishopAttackIndex`/`GetRookAttackIndex`
  (`attacks.cc:124-140`) as the single seam both the table-generation code
  and `move_gen.cc`'s `BishopMoves`/`RookMoves` now call through:
  ```text
  fn attack_index(square, occupied):
      if USE_PEXT:  return pext(occupied, mask[square])          // BMI2 intrinsic
      else:         return ((occupied & mask[square]) * magic[square]) >> shift[square]
  ```
  Both branches feed the *same* dense per-square table (§4.1) — PEXT changes
  only how the index is computed, never the table's shape or size, since
  PEXT's compressed-bits output range is `[0, 2^popcount(mask))`, identical
  to the magic-multiply's index range.
- The interesting part is entirely in `CMakeLists.txt` (`:87-152`). Instead
  of a single "have BMI2 → use PEXT" test, it **excludes specific CPU
  families known to have slow (microcoded) `PEXT`/`PDEP`**: Zen 1 and Zen 2
  are refused PEXT even when the compiler reports `__BMI2__` available,
  *except* a hardcoded carve-out for "AMD EPYC 7B" detected via `lscpu` on
  Linux (almost certainly because that's the specific cloud CPU their
  OpenBench-style test fleet runs on, and they'd measured it to be fine
  there specifically — `Makefile:8-13` `x86_64_bmi2` build target and the
  `BUILD_AVX2_BMI2` CMake profile, `-march=haswell -mbmi2 -DUSE_PEXT`,
  further confirm a curated per-arch build matrix, not just a native-only
  toggle). Zen 3/Zen 4 and all non-AMD (i.e., Intel) targets get PEXT
  unconditionally. This is precisely the well-known "PEXT/PDEP are
  microcoded and slow pre-Zen-3 on AMD" fact — and it is **the exact same
  fact our own `design/03-attack-tables.md` §3.8 ("Why not PEXT") already
  cites** as one reason we don't use PEXT at all: "avoids the catastrophic
  microcoded-PEXT penalty on Zen 1/2." Integral's CMake logic is essentially
  a worked, shipped implementation of the caveat our doc names in prose —
  useful independent confirmation that the concern is real and specific
  (a top-10 engine wrote 45 lines of CMake specifically to route around it)
  rather than theoretical.
- Net effect for the comparison set: **SF dropped PEXT years ago and now
  ships plain-magic-only** (confirmed, `stockfish_reference/01-bitboards-attacks-move-encoding.md:37`).
  **Reckless never had a PEXT path** (`reckless_reference/02-attacks-magics-setwise.md`
  §4). **We deliberately declined PEXT** (D12, same Zen-1/2 rationale).
  **Integral is the one engine here that actively ships PEXT**, but only as
  an *opt-in, CPU-gated* alternative alongside the same plain-magic fallback
  everyone else uses unconditionally — not a wholesale "PEXT is better,
  switch to it" story, but a "PEXT is a small win on the CPUs where it's
  fast, so gate it narrowly and keep the safe universal default" story. That
  nuance (rather than a blanket recommendation either way) is the useful
  takeaway.

## 5. Adapt to us

1. **Pre-filter pinned knights out of full-legal move generation** — a
   knight pinned to its own king can never have a legal move (no knight step
   is colinear with any ray), so iterating `board.pieces(us, Knight) &
   !board.blockers_for_king(us)` instead of the unfiltered set (currently
   `movegen.rs:184`) skips a doomed `Board::legal` probe and a wasted
   `MovePicker` score for every generated destination of a pinned knight, at
   the cost of one extra `Bitboard` AND per node. Target: `design/05-movegen.md`
   §3 (the per-piece-type generation loop) — note the change and its
   provable-safety argument in the doc per the "docs follow decisions" rule.
   Touches `movegen`, so the full D24 perft ritual applies; should be
   perfectly perft- and bench-neutral (same legal-move set, cheaper path to
   it) — any node-count change at fixed depth would flag a bug. Effort: very
   low (one-line change + a unit test asserting a pinned knight never
   appears as a `from` square in `generate_legal`'s output). Impact: small
   and situational (only pays off in positions with a pinned knight — not
   rare enough to ignore, common enough not to be free lunch every node),
   but essentially free to add.

2. **If a PEXT-gated attack-index variant is ever explored, gate it on CPU
   *family* exclusion (Zen 1/2), not merely on `target_feature = "bmi2"`.**
   Our D16 fast-build profile (`RUSTFLAGS="-C target-cpu=x86-64-v3"`) already
   implies BMI2 is *architecturally* present (x86-64-v3 requires BMI1+BMI2),
   but Zen 1/2 CPUs do support the BMI2 *instruction* — just with a slow
   microcoded `pext`/`pdep` — so a naive `#[cfg(target_feature = "bmi2")]`
   PEXT path would silently regress on exactly the hardware class our own
   §3.8 already worries about, exactly as Integral's CMake logic guards
   against with its explicit Zen-1/2 exclusion + Linux `lscpu`-based
   EPYC-7B carve-out (`CMakeLists.txt:120-152`). This is not a recommendation
   to add PEXT — our documented reasons for declining it (D12) stand — but
   if it's ever revisited (e.g., a dedicated `x86-64-v4`/Znver4 build
   profile), the gate needs to be a runtime/build-time CPU-family check
   (`raw_cpuid` crate or manual `CPUID` leaf 1/0x80000001 family/model
   decode), not a blanket feature-flag test. Target: `design/03-attack-tables.md`
   §3.8 (append the caveat, cite this note), D12. Effort: low if ever
   undertaken (mechanical `cfg`/runtime-branch plumbing, needs its own
   correctness cross-check against the existing magic path via `cargo test`
   at every square + a criterion bench, same rigor as the existing magic
   init). Impact: speculative and likely small (single-digit-percent nps at
   best on Intel/Zen3+, per general chess-programming consensus that plain
   multiply-magic and PEXT are close on modern hardware) — low priority,
   parking-lot item, not a recommended near-term investment.

3. **Cross-validation, no action required:** Integral's from-scratch,
   per-node `CalculateThreats` (§3.5) is a third independent confirmation
   (after Reckless's vectorized `update_threats`) that "recompute the
   whole-board opponent threat map every node" beats incremental
   maintenance for this specific feature. Our D30 Wave-G threat map already
   made this call; this note is supporting evidence to cite in
   `design/10-pruning-extensions.md` "Wave G" if a future reader questions
   the non-incremental design, not a new task.

4. **Considered and declined: copy-full-`BoardState` undo instead of
   incremental make/unmake.** Integral's `history_: List<BoardState, 2048>`
   (§1) trades a large fixed per-thread memory/bandwidth cost (~300+ B ×
   2048 ≈ 600 KiB+ per search thread, multiplied by worker count under our
   just-landed P11 lazy SMP) for "undo can never be wrong, because it isn't
   computed — it's restored." Our D8 `StateInfo` stack + incremental Zobrist
   maintenance is the opposite, deliberate choice (smaller footprint, but
   `unmake_move` must be provably the exact inverse of `make_move` for every
   move kind — exactly what `assert_consistent` + the D24 perft ritual exist
   to guarantee). Nothing here suggests reconsidering D8; recorded because a
   future reader comparing the two designs should know the alternative was
   examined, not overlooked, and why the memory/thread-count math argues
   against it now that lazy SMP is live.

5. **No action, parity confirmed:** `check_zones`/`MoveGivesDirectCheck`
   (§3.4) is the same SF `checkSquares[pt]` idea we already ship as
   `check_squares`/`gives_check` (`src/board.rs:185,565-571,622-634`) — cited
   here only so a future reader doesn't mistake it for something to port.
