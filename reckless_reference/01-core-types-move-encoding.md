# Core types & move encoding (Reckless)

Source: `reckless_reference/Reckless/src/types/{mod,bitboard,square,color,piece,
moves,movelist,arrayvec,score,castling,zobrist,keys}.rs`, Reckless commit
`d445b8b` (tagged `0.10.0-dev`, 2026-07-11), Rust edition 2024, **AGPLv3**.

License reminder (mirrors our D27): everything below is *my own prose and
pseudocode*, distilled from reading the source. No Reckless code is quoted
beyond the handful of single-line formulas called out explicitly as such. Do
not paste this note's pseudocode blocks back into `src/` as if they were a
port of the original text — reimplement from the idea, in our own style.

Reckless is single-author, NNUE-native from day one (no HCE phase), and reads
throughout like a hand-tuned, SIMD-friendly rewrite of Stockfish's core types
with several independent simplifications and a few outright cleverer bit
tricks. It leans on `unsafe { mem::transmute }` far more than our house style
allows (D14) — flagged everywhere it matters below, not just once.

## Square (`square.rs`)

Same little-endian-rank-file mapping as ours and SF (a1=0 … h8=63), but the
representation is different in kind: `Square` is a **65-variant fieldless
enum** (`A1..H8`, plus `#[default] None = 64`), not a `u8`-wrapping struct.
Every int→enum conversion (`new`, `file`, `rank`) goes through
`unsafe { mem::transmute }` (square.rs:31-47); the reverse direction
(enum→int) is the safe, ordinary `self as u8`. This is *why* the type has to
be an enum at all — Reckless wants the compiler-visible variant names
(`Square::E4`, exhaustive `match`) but pays for it with unsafe on every
"build a square from a number" path. Our `Square` sidesteps this entirely: a
plain `{ index: u8 }` struct with `const fn new` guarded by a runtime
`assert!`, so building a square from an arbitrary `u8` is always safe Rust
(`src/types.rs:222-225`), just not `const`-panic-free on garbage input (which
is the intended failure mode — see design/02 §3.4).

Other notables:
- `Square::shift(offset: i8)` (square.rs:49-54) is an *unchecked* raw offset
  add guarded only by a `debug_assert!` range check — no wrap detection at
  all (contrast our `try_offset`, design/02 §3.4, which additionally rejects
  any step whose Chebyshev distance exceeds 2, catching file-wrap). Reckless
  accepts the wrap risk at every call site instead of centralizing the guard.
- `distance_from` (square.rs:63-67) is Chebyshev distance via two `abs()`
  calls — same formula as our `Square::distance` (design/02 §3.4), computed
  fresh rather than table-backed either way.
- `flip_rank = self ^ 56` (square.rs:69-71) — identical XOR trick to ours
  (`flip_rank`), and `relative_to` (square.rs:56-61) is a `match` rather than
  our branchless `index ^ (color as u8 * 56)` (design/02 §3.4) — Reckless
  spends a branch where we spend a multiply-by-0-or-1.
- `impl<T> Index<Square> for [T]` is safe (`&self[square as usize]`,
  square.rs:128-134) but **`IndexMut<Square>` is `unsafe { get_unchecked_mut
  }`** (square.rs:136-140) — an asymmetry: shared-reference table lookups by
  `Square` are bounds-checked, mutable ones are not. A stray `Square::None`
  (discriminant 64) reaching a 64-entry table through the mutable path is
  silent UB, while the same value on the immutable path panics loudly. This
  asymmetry is a real risk surface unique to this scheme (see "notable
  differences" below); our own indexing helpers should not copy it.
- `TryFrom<&str>` (square.rs:82-96) parses algebraic notation directly by
  byte-pattern match on `[a-h][1-8]`, same shape as our `FromStr` (design/02),
  no meaningful difference.

## Color (`color.rs`)

`#[repr(u8)] enum { White = 0, Black = 1 }`, `Not` flips (color.rs:23-32),
`Index<Color>`/`IndexMut<Color>` sugar for `[T]` (color.rs:34-46). Functionally
identical to our `Color` (design/02); the only difference is the blanket
`Index<Color> for [T]` operator-overload — see the "Adapt to us" item on this
pattern, it recurs for every newtype in this file.

## Piece / PieceType (`piece.rs`)

The one place Reckless's encoding is **structurally different from both SF
and us**, not just differently coded.

- SF packs color in **bit 3** of an 4-bit-ish piece byte: `W_PAWN=1..W_KING=6`,
  `B_PAWN=9..B_KING=14` (see `stockfish_reference/01-bitboards-...md`).
- We use a plain 2-field struct `Piece { color: Color, piece_type: PieceType }`
  with a *derived* flat index `color as usize * 6 + piece_type as usize`
  (`src/types.rs:106-108`, design/02 §3.6) — White P..K = 0..6, Black P..K =
  6..12 (block layout).
- Reckless **interleaves** color in **bit 0**:
  `Piece::new(color, pt) = (pt << 1) | color` (piece.rs:49-51). So
  `WhitePawn=0, BlackPawn=1, WhiteKnight=2, BlackKnight=3, …` — same-type
  pieces of opposite color are always *adjacent* array slots, not 6 apart.
  Extraction is two more `transmute`s: `color = self & 1`, `piece_type =
  self >> 1` (piece.rs:59-65). A 13th variant, `#[default] None` (index 12,
  never a valid `(color, piece_type)` pair), plays the "empty square" role
  that we give to `Option<Piece>` in the mailbox (`mailbox: [Option<Piece>;
  64]`, per CLAUDE.md) — Reckless's mailbox is presumably `[Piece; 64]` with
  `Piece::None` as the empty sentinel (no `Option` wrapper needed, but no
  compiler-enforced "must handle the empty case" either).
- `PieceType` (piece.rs:120-129) mirrors this: 6 real variants plus a 7th
  `None` sentinel, again built via `unsafe transmute(value as u8)`
  (piece.rs:134-136).
- `PieceType::value()` (piece.rs:138-148) gives P 109, N 403, B 435, R 679,
  Q 1242 — roughly **half our SEE scale** (ours: 208/781/825/1276/2538,
  SF-pinned per D11/D20) but at nearly the *same relative ratios* (N/P≈3.7
  both, B/P≈4.0 both, R/P≈6.1–6.2 both; only Q/P drifts, 11.4 vs 12.2).
  Reads as an independently retuned absolute scale on the same skeleton, not
  a different valuation model — a useful external data point for whenever
  we retune D11 away from the SF defaults ourselves.
- `TryFrom<char>`/`TryInto<char>` (piece.rs:68-98) round-trip through the
  literal string `"PpNnBbRrQqKk"` (index = position in the string) — a neat,
  allocation-free one-liner alternative to our `match`-based `to_char`/
  `from_char` (design/02), worth 5 minutes to consider, not worth churning
  working code for.

## Bitboard (`bitboard.rs`)

`#[repr(transparent)] struct Bitboard(pub u64)` — public field, same
newtype-over-u64 shape as ours, but three differences:

1. **`Bitboard` directly `impl Iterator`** (bitboard.rs:68-80): `next()` pops
   the LSB and clears it (`self.0 &= self.0 - 1`) exactly like our
   `pop_lsb`-style iterator, but there is no separate iterator *type* — the
   set doubles as its own single-use iterator (consumed by value, since it's
   `Copy` a `for sq in bb` loop silently iterates over a *copy*, never
   mutating the caller's `bb`). Our project deliberately does the opposite
   (CLAUDE.md: "Sets don't implement `Iterator`; they implement
   `IntoIterator` with a separate iterator struct" — see `Bitboard`/
   `BitboardIter`, `src/bitboard.rs:148-178`) specifically to keep "the set"
   and "an in-progress walk over the set" as distinct types. Reckless's way
   is fewer lines; ours is the documented house style and should not change.
2. **Generic signed-shift primitive, unmasked**: `shift(offset: i8)`
   (bitboard.rs:55-57) is just `<< offset` or `>> -offset` — no file-wrap
   masking at all. Every call site is responsible for pre-masking
   (`(bb & !FileH).shift(9)`, e.g. `setwise.rs:20,30-37`, `board/movegen.rs:
   107,126,131`). Our `Bitboard::shift(d: Direction)` (`src/bitboard.rs:
   109-124`, design/02 §3.5) bakes the correct `NOT_FILE_A`/`NOT_FILE_H` mask
   into each of the 8 match arms so a caller *cannot* forget it. Reckless
   trades a small amount of per-call-site risk for a leaner primitive that
   composes with hand-tuned setwise leaper generation (below); we trade a
   few more lines in one place for a shift that is correct by construction
   everywhere it's used. Ours is the safer default; worth keeping.
3. **Setwise leaper generation** (`setwise.rs`, referenced from
   `bitboard.rs`'s `shift`): knight attacks for an entire bitboard of knights
   are computed as 8 shift+mask+OR terms over pre-masked file bands (`not_a`,
   `not_ab`, `not_gh`, `not_h`), not via a per-square table lookup in a loop.
   Bishop/pawn "attacked-by-any-piece-in-this-set" bitboards are built the
   same way (`setwise.rs:20`). This is a genuine, distinct technique from our
   per-square fancy-magic/leaper-table approach — good for "which squares
   does this whole army threaten" bulk queries (eval king-safety/threat
   masks), not obviously better for per-move legal generation where we
   already hit one piece at a time.

Constants worth noting: `LIGHT_SQUARES = 0x55AA55AA55AA55AA`, `CORNERS =
0x8100000000000081`, `LEVER_RANKS` (the two ranks where pawns first threaten
enemy pawns), `THIRD_RANK`/`SIXTH_RANK` (double-push landing ranks), and
`BOTH_HOME_ROWS` (bitboard.rs:13-20) — all are the kind of derived
mini-tables that recur across engines; ours has the equivalent set
(`FILE_A/H`, `RANK_1/8`, `FILES`/`RANKS`, `src/bitboard.rs:10-40`) but not
`LEVER_RANKS`/`CORNERS`/`LIGHT_SQUARES` yet — cheap to add if a pawn-structure
or bishop-color eval term wants them (design/11, not yet built).

## Move & `MoveKind` (`moves.rs`) — the headline difference

Both engines pack a move into one `u16`: 6 bits `from`, 6 bits `to`, then 4
more bits. SF and we spend those 4 bits as **two independent 2-bit fields**
(promotion-piece, kind ∈ {Normal, Promotion, EnPassant, Castling}) — capture-
ness is *extrinsic*, answered by asking the board. Reckless spends the same 4
bits as **one enumeration with 12 populated values out of 16**
(moves.rs:18-35), and gets capture-ness and the noisy/quiet split *for free*
from the bit pattern alone, no board access required:

```
Normal            = 0b0000        DoublePush = 0b0001   Castling = 0b0010
Capture           = 0b0100        EnPassant  = 0b0101
PromotionN..Q     = 0b1000..1011  (low 2 bits: N=0,B=1,R=2,Q=3)
PromotionCaptureN..Q = 0b1100..1111 (same low-2-bit piece encoding)
```

The design exploits three bit positions:
- **bit 2 (`0b100`) ⇔ "this move captures."** True for `Capture`,
  `EnPassant`, and all four `PromotionCapture*` — false for everything else,
  including plain promotions. `is_capture()` is `self.0 & (1<<14) != 0`
  (moves.rs:78-80) — one AND, zero board lookups, always in sync with the
  move because it was set once at generation time.
- **bit 3 (`0b1000`) ⇔ "this move promotes."** `is_promotion()` is the same
  shape one bit up (moves.rs:82-84); combined with `kind() & 3`
  (moves.rs:102-105) that recovers the promoted `PieceType` without a lookup
  table, same trick we already use (design/02 §3.7).
- **`is_noisy()` is a single masked comparison, and it is genuinely clever**
  (moves.rs:70-72, comment preserved as "sneaky bit-twiddling" in spirit —
  quoted verbatim as the one formula worth citing exactly):
  `(kind as u8 & 7) > Castling as u8` (i.e. `> 2`). Masking off the
  promotion bit (bit 3) and comparing the remaining 3 bits against `Castling
  = 2` sorts all 12 kinds into "noisy" (any capture, or a **queen**
  promotion) vs "quiet" (everything else, including plain **under**-
  promotions) — because `PromotionN/B/R` mask down to `0/1/2` (≤ 2, quiet)
  while `PromotionQ` masks to `3` (> 2, noisy), and every `*Capture*` kind
  masks to ≥ 4. One AND, one compare, zero branches, zero board access, and
  it reproduces exactly the "captures + queen promotions are tactical,
  underpromotions are not" rule every engine wants for qsearch/history
  gating.
- **`is_special()` = `kind & 0b1011 != 0`** (moves.rs:74-76) picks out bits
  0, 1, 3 — i.e. "does `make_move` need non-generic bookkeeping" (double
  push sets an ep square; castling also moves a rook; any promotion changes
  the piece type). Notably bit 2 (plain capture) is *excluded* — an ordinary
  capture needs no bookkeeping beyond "there was a piece on `to`," which
  `make_move` already handles unconditionally.
- `capture_sq()` (moves.rs:90-92): `to() ^ (is_en_passant() as u8 * 8)` — the
  captured pawn's square is `to()` itself for every ordinary capture, and
  `to() XOR 8` (one rank toward the mover) for en passant, computed
  branchlessly by multiplying the boolean flag into the XOR mask. Small, but
  a nice one-liner over the usual `if en_passant { to - up } else { to }`.
- `to_uci()` (moves.rs:132-153) special-cases Chess960: when
  `board.is_frc()` and the move is castling, it re-encodes as
  "king-captures-rook" (UCI's Chess960 castling convention) by asking the
  board for the actual rook square — confirms Reckless ships Chess960/FRC
  support end to end (see also `castling.rs` below). Our master plan lists
  Chess960 as a still-open item (`design/00-master-plan.md:260`).
- `to_tb_move()` (moves.rs:107-130, `#[cfg(feature = "syzygy")]`) packs a
  move into the Syzygy tablebase probe format: `(from << 6) | to`, plus
  `(promo & 0x7) << 12` where the *tablebase's own* promotion encoding is
  Q=1, R=2, B=3, N=4 (note: different numbering from Reckless's own
  in-house `MoveKind` promotion bits, and from ours — this is the external
  Syzygy wire format, not a Reckless invention, but the exact constant is
  worth having on file for whenever we implement Syzygy, an open item per
  `design/00-master-plan.md:390`).

**The correctness caveat this trades in**: because capture-ness is baked
into the move's bits at generation time rather than derived from the current
board, a `Move` is only self-describing *relative to the position it was
generated against*. Our extrinsic `Board::is_capture(m)`
(`src/movegen.rs:654-660`) can never disagree with the board because it
always re-asks it; Reckless's intrinsic `is_capture()` could in principle go
stale if a `Move` (e.g. a raw `u16` pulled from the TT) were interpreted
against a *different* board than the one that produced it. In practice this
is a non-issue as long as every such move is first run through a
pseudo-legality check against the current board (our own D19 TT-move
validator does exactly this) — but if we ever adopt the "bake capture into
kind" trick, the validator must also confirm the capture bit still matches
current occupancy at `to`, not just that `from`/`to`/piece are consistent.

## `MoveList` / `MoveEntry` / `ArrayVec` (`movelist.rs`, `arrayvec.rs`)

Reckless's `MoveList` is not a bare array of `Move` — it's
`ArrayVec<MoveEntry, 256>` where `MoveEntry { mv: Move, score: i32 }`
(movelist.rs:5-14) — i.e. **every** move list, including plain movegen/perft
call sites that never touch move ordering, carries a 4-byte (really 8-byte,
with padding) score slot per move, always initialized to 0 and usually
unused until the picker fills it in. Contrast our split (design/02 §3.8,
design/09 §3.1): `MoveList` stays a bare `[Move; 256]` (2 bytes/slot,
`src/moves.rs:132-143`), and only the search's own `MovePicker` buffer uses
the paired type (`ExtMove { mv: Move, value: i32 }`,
`src/movepick.rs:58-63`, `#[repr(C)]`, asserted 8 bytes/align 4). Same idea,
opposite layering choice — ours keeps the plain generator/perft path lean
and pays the pair-type cost only where scores are actually used; Reckless
pays it everywhere for the uniformity of one type.

`ArrayVec<T, N>` (arrayvec.rs) is a hand-rolled fixed-capacity vector over
`[MaybeUninit<T>; N]`, `Copy`-only, with `push`/`swap_remove`/`iter` all
built on `unsafe` (`get_unchecked`, `assume_init`, `ptr::copy`,
`slice::from_raw_parts`) — arrayvec.rs:24-71. Our `MoveList` needs none of
this: `Move` is a `Copy` `u16` cheap enough that the array is simply
*fully initialized* up front to `Move::NULL` (`src/moves.rs:138-143`) and
`len` tracks the logical size — zero `unsafe`, at the cost of one 512-byte
memset per list that we'd pay anyway for any fixed-size buffer. Given our
D14 policy (`unsafe` is rationed to one sanctioned prefetch intrinsic), the
memset-and-track-`len` shape is the right tradeoff for us; Reckless's
`MaybeUninit` version only earns its unsafe if move-list construction is hot
enough that skipping the memset matters (unmeasured here) — and it does
double duty as the substrate for the AVX-512 fast paths below, which is
probably the real reason it exists.

**AVX-512 VBMI2 bulk push** (`#[cfg(target_feature = "avx512vbmi2")]`,
movelist.rs:40-67, 76-106, arrayvec.rs:82-106): on capable hardware,
`push_setwise`/`push_pawns_setwise` skip the per-destination-square Rust loop
entirely. They build a 32-lane `__m512i` template of "pre-filled" `Move`s (16
bits each, so 32 fit in 512 bits), OR/ADD in the shared `from`/`kind` bits
once, then use `_mm512_maskz_compress_epi16` to compact only the *set* lanes
(driven directly by the 32/64-bit half of the destination bitboard as an AVX
mask) into the buffer and bump `len` by the popcount — turning "generate up
to 32 moves from one piece" into a handful of vector instructions with no
branches per destination square. `MAX_MOVES = 256` (not the tight 218) exists
*specifically* so this 16-or-32-at-a-time compressed store never overruns
the buffer (`mod.rs:28-31`, comment preserved: "Padding added because...
writes 16 moves at a time"). This is a legitimately advanced, narrow-audience
technique (needs `avx512vbmi2`, a rare target even among x86-64-v3/v4
machines) — interesting to know about, not something to chase before NNUE
and lazy SMP (P13/P11) are done; our own `MAX_MOVES = 256` already has the
same headroom over 218 for unrelated reasons (design/02).

## Score (`score.rs`)

Constants: `NONE=32002, INFINITE=32001, MATE=32000, MATE_IN_MAX = MATE -
MAX_PLY`. Reckless's `MAX_PLY = 240` (`mod.rs:26`) vs ours `246`
(`src/types.rs:458`) — a small, inconsequential numeric difference, but worth
knowing the two aren't the same constant if ever cross-referencing mate
distances. `TB_WIN = MATE_IN_MAX - 1`, `TB_WIN_IN_MAX = TB_WIN - MAX_PLY`
(score.rs:16-17) — same three-band shape as our
`VALUE_MATE`/`VALUE_MATE_IN_MAX_PLY`/`VALUE_TB`/`VALUE_TB_WIN_IN_MAX_PLY`
ladder (design/02 §3.2), just without separately *named* negative-mirror
constants — `is_win`/`is_loss` (score.rs:42-48) negate the same constant
in-line instead of pre-computing `TB_LOSS_IN_MAX`. Functionally identical
predicate structure to ours.

`draw(td)` (score.rs:20-22): `(td.nodes() % 5) as i32 - 2`, a five-way
±2 dither keyed off the node counter — wider and one integer-division
costlier than our node-count draw dither, `VALUE_DRAW - 1 + (nodes & 0x2)`
(`src/search/mod.rs:177`, ±1, one AND, matches current SF exactly per
design/07 §3.1). Reckless deliberately diverges from SF here; whether ±1 or
±2 (or something else) is better is an empirical SPRT question, not
something to copy blind.

`normalize_to_cp(score, board)` (score.rs:58-70): converts an internal score
to "pawns=100" centipawns via a phase-dependent cubic fit — material count
(1/3/3/5/9-weighted, clamped to `[16, 78]`, normalized to `v ∈ [0,1]`) feeds
`normalization = -285.1 v³ + 642.5 v² - 455.5 v + 464.8`, then
`round(100 * score / normalization)`. This is the exact shape (material-
dependent polynomial normalization, not a flat divide-by-a-constant) that our
own `to_cp` seam is already pinned to adopt once phase C lands
(`design/12-uci-time-management.md:924-931`, currently identity pending
NNUE) — Reckless's coefficients are a concrete, independently-fitted data
point for that day, not something to transcribe verbatim (their net's output
scale won't match ours).

## Castling (`castling.rs`)

`CastlingKind` is a 4-variant bitmask enum (`WhiteKingside=1,
WhiteQueenside=2, BlackKingside=4, BlackQueenside=8`, castling.rs:7-12) with
`KINDS: [[Self;2];2]` giving `[color][queenside=0/kingside=1]` lookup
(castling.rs:15-16) — and `Castling { raw: u8 }` (castling.rs:43-45) is a
bare wrapper with `is_allowed`/`raw`, plus `Index<Castling> for [T]`
(castling.rs:90-96) so a 16-entry table can be indexed by the *whole* rights
byte directly (`table[castling]`). This is structurally the same idea as our
`CastlingRights(u8)` newtype with `WHITE_KINGSIDE=0b0001` etc.
(`src/board.rs:10-18`) and the same "index a 16-slot table by the raw 4-bit
mask" trick our Zobrist castling keys already use (design/04 §3.4) — no
functional gap either direction. `Castling::to_string` (castling.rs:56-88)
derives Chess960 (`board.is_frc()`) shredder-FEN-style castling letters
(`A`/`a` + rook file) on demand rather than storing them — confirms (again)
that Reckless's Chess960 support runs all the way through FEN output, not
just move parsing.

## Zobrist keys (`zobrist.rs`, `keys.rs`)

One flat `[u64; 865]` table, filled by a **SplitMix64** stream (not
xorshift64\*, which is what SF and we use — design/04 §3.4, D20): seed
`0xFFAA_B58C_5833_FE89`, golden-ratio increment `0x9E37_79B9_7F4A_7C15`,
the classic SplitMix64 avalanche (`z ^= z>>30; z *= 0xBF58...; z ^= z>>27;
z *= 0x94D0...; z ^= z>>31`) applied to a running `state += increment`
counter (zobrist.rs:12-30) — a fine, independent choice of PRNG, not
something to adopt just because it's different (our xorshift64\* stream is
pinned to match SF's exact seed/algorithm per D20, which is the point).

The **865** slots split as `pieces[12][64]` (768) + `en_passant[64]` (one
per *square*, not per *file* — see below) + `castling[16]` + `side[1]` +
`fiftymove_clock[16]` = 768+64+16+1+16 = 865, `unsafe transmute`d in one shot
from a flat array into the `Zobrist` struct (zobrist.rs:29) — the one place
in this file where the `unsafe` is doing real, defensible work (a
compile-time reinterpret of a `[u64; 865]` into a `#[repr(Rust)]` struct of
matching total layout), unlike the enum-conversion transmutes elsewhere.

**`en_passant` is square-indexed (64 slots), not file-indexed (8 slots)**
like SF's `enpassant[FILE_NB]` and ours (`Tables::en_passant: [Key; 8]`,
design/04 §3.4) — functionally fine (the ep target square's rank is always
implied by side to move, so file-only hashing loses no information) but
8× more memory for no behavioral gain; a simplification we already made
correctly and shouldn't undo.

**The key *family* is smaller than ours.** `Keys { full, pawn, non_pawn:
[u64;2] }` (keys.rs:4-8) — three tracked components (main + pawn +
per-color non-pawn) versus our five-key family (`key`/`pawn_key`/
`material_key`/`minor_key`/`non_pawn_key[2]`, design/04 §3.4, D9). Reckless
has **no material key and no minor-piece key** at all. This isn't an
oversight — it exactly matches what its correction-history stack actually
needs: `SharedCorrectionHistory { pawn: CorrectionHistory, non_pawn:
[CorrectionHistory; 2] }` plus a separate pointer-indexed (not
zobrist-keyed) 2-ply/4-ply *continuation* correction history
(`thread.rs:145-147`, `search.rs:1367-1409`) — there is simply no
material- or minor-piece-keyed correction table to back with a key. Our own
correction-history stack already carries per-table weights over
pawn/minor/… (design/09 §3.10) — the five-key family earns its keep for us,
it just means Reckless's `Keys` type is a case of "build exactly the keys
your consumers use," worth re-checking on our side whenever a new correction
table is proposed (don't add a key nobody reads).

**`toggle_castling` is called twice per move, bracketing the mutation**
(`board/makemove.rs:96-98`): once with the *old* `Castling` (XORs it out of
`full`), then the rights byte is ANDed down by the from/to
castling-rights-clearing masks, then `toggle_castling` again with the *new*
value (XORs it back in). Same "old XOR new" effect as our single-call
`castling[old] ^ castling[new]` bookkeeping (design/04 §3.4 trick 2), just
expressed as two separate one-argument toggles instead of one two-argument
XOR — a style choice, not a different algorithm.

**GHI mitigation and correction-history bucketing share one table.** The
16-slot `fiftymove_clock` Zobrist array does double duty:
`hash()` XORs `ZOBRIST.fiftymove_clock[bucket]` into the TT-facing key to
mitigate graph-history interaction (`board.rs:78-84`, bucket =
`(clock.saturating_sub(8) / 8).min(15)`), and the *exact same*
`fiftymove_clock_bucket()` function is reused as the bucket index into the
pawn/non-pawn **correction histories** (`search.rs:1366-1394`) — one bucket
function, two independent consumers. This is a genuinely nice piece of
economy: a single "how stale/deep into the 50-move count are we" signal
feeds both the classic SF-style key-folding trick *and* a rule50-dependent
axis of the correction-history read/write, rather than inventing two
separate bucketing schemes. Our own `fold_rule50`/`fold_rule50_after`
(`src/zobrist.rs:176-197`) only serves the key-folding purpose today; our
correction history (design/09 §3.10) currently has no rule50 axis at all.
See "Adapt to us" below — this is a concrete, cheap idea to evaluate.

## Distinctive vs Stockfish and vs us — summary

| Aspect | Stockfish | Us | Reckless |
|---|---|---|---|
| `Square` repr | packed int | `struct{index:u8}`, safe `new` | 65-variant enum, `unsafe transmute` both ways in, safe `as u8` out |
| `Piece` color bit | bit 3 (`pc^8` flips color) | separate `Color`/`PieceType` fields, no packing | bit 0 (interleaved, `(pt<<1)\|color`) |
| Move capture info | extrinsic (board lookup) | extrinsic (`Board::is_capture`, board lookup) | **intrinsic** — baked into `MoveKind`'s bit pattern |
| Noisy/quiet test | kind==Promotion&&Q \|\| board capture | same shape (`capture_stage`, board lookup) | one bitmask+compare, no board access |
| Zobrist key family | key+pawn+material+minor+nonPawn×2 (current SF) | same 5+1 (D9) | key+pawn+nonPawn×2 only (3+1) |
| Zobrist PRNG | xorshift64\* | xorshift64\* (SF-matched, D20) | SplitMix64 |
| `en_passant` keys | 8 (by file) | 8 (by file) | 64 (by square) |
| GHI key folding | rule50 bucket | rule50 bucket (`fold_rule50`) | rule50 bucket, **also reused for corrhist bucketing** |
| MoveList element | `Move` only | `Move` only (`ExtMove` only in the picker) | `(Move, score)` pair everywhere |
| Bitboard iteration | — | `IntoIterator` + separate iterator struct (house style) | `Bitboard` is its own `Iterator` |
| `unsafe` budget | n/a | one sanctioned prefetch intrinsic (D14) | pervasive — every enum int-conversion, `ArrayVec`, AVX-512 paths |
| Chess960/FRC | supported | open item (`design/00`:260) | supported end to end (castling, UCI, Zobrist letters) |
| Syzygy TB | supported | open item (`design/00`:390) | supported behind `feature="syzygy"` |

## Adapt to us

- **Bake "is capture" (and "is a queen promotion") into the move encoding
  itself**, the same way `MoveKind`'s bit layout does — e.g. reserve one bit
  of our existing 2-bit `kind` + 2-bit promo field pair (or accept a wider
  encoding) so `Move::is_noisy()`/`is_capture()` become pure bit tests with
  no `Board` argument. Touches `design/02-core-types.md` §3.7 (`Move`
  encoding) and `design/05-movegen.md` (every `push` call site must supply
  the right kind at generation time) and ripples into `design/09`
  (`MovePicker` scoring, which today calls `board.is_capture`/`capture_stage`
  per candidate). **Effort: medium** (touches movegen, TT-move validation,
  qsearch gating, SEE call sites — a real encoding change, not additive).
  **Impact: likely small-but-free speedup** in move-ordering/qsearch hot
  loops (removes a board dependency from the single most frequently asked
  question in search) — worth a fixed-node proxy SPRT (doc 14 §9.4) before
  committing, since it's a correctness-sensitive encoding change gated by
  the D24 perft ritual.
- **Reuse the rule50/GHI bucket as a correction-history axis.** Our
  `fold_rule50` bucketing (`src/zobrist.rs`) already computes "how deep into
  the 50-move count is this position"; Reckless shows that the *same*
  function can double as a correction-history bucket dimension (alongside
  the existing pawn/minor/non-pawn/material axes, design/09 §3.10) for
  cheap. **Effort: low** (one new array dimension + one bucket lookup,
  no new Zobrist keys needed — we already compute the bucket for TT
  folding). **Impact: unknown, plausible small Elo** — needs its own
  proxy-SPRT slot in the P9 pruning-wave / P8 history-tuning backlog before
  landing; flag as an experiment, not a guaranteed win.
- **Blanket `Index<T>`/`IndexMut<T>` impls for our own newtypes over `[U]`**
  (`impl<U> Index<Square> for [U] { fn index(&self, s: Square) -> &U { &self[s.index() as usize] } }`,
  and the same for `Piece`, `PieceType`, `Color`). Every one of Reckless's
  core types gets this (square.rs:128-140, piece.rs:106-118&151-163,
  color.rs:34-46, castling.rs:28-40&90-96) and it turns every
  `table[x.index() as usize]` in our attack tables, PSQT tables, and history
  tables into plain `table[x]`. **Effort: very low** (a handful of trait
  impls, purely additive, no behavior change — safe to land any time).
  **Impact: readability only**, but free and touches every module that
  indexes by these types (`attacks`, `eval`, `history`), so it compounds.
  Keep the *mutable* impl bounds-checked (`&mut self[idx]`, not
  `get_unchecked_mut`) — Reckless's asymmetric safe-read/unsafe-write split
  (square.rs:128-140) is a footgun (a stray out-of-range index silently
  corrupts memory on the write path while the read path would have
  panicked), not a pattern worth inheriting under our D14 discipline.
- **`normalize_to_cp`'s cubic material-weighted normalization** is a
  concrete, ready-to-crib *shape* (not constants) for `design/12`'s
  currently-identity `to_cp` seam once eval phase C (NNUE) lands and we fit
  our own coefficients from our own net's output distribution
  (`design/12-uci-time-management.md` §"WDL model timing", D21). **Effort:
  low once NNUE exists** (curve-fit against our own eval outputs across a
  representative game corpus). **Impact: UCI-facing score display only**
  (`cp` values reported to a GUI/operator), not search-strength-affecting —
  low priority, correctness/UX polish for P13.
- **Setwise leaper attack generation** (`setwise.rs`'s shift-and-mask
  knight/pawn attacks for a whole bitboard at once) is worth a look for
  eval's threat/king-safety bitboards (`piece_threats`-style aggregates)
  once `design/11` eval phase B/C wants "all squares attacked by all pieces
  of type X" in bulk, rather than looping per-piece through the fancy-magic
  tables. **Effort: medium** (new bitwise formulas per piece type, each
  needs its own correctness proof against the naive per-square oracle, D24-
  adjacent discipline). **Impact: speculative** — only matters if profiling
  ever shows per-piece attack-table lookups as a hot spot in eval, not
  movegen (movegen already iterates per-piece for other reasons, e.g. move
  encoding).
- **Not recommended to port**: the pervasive `unsafe transmute` int↔enum
  pattern across `Square`/`Piece`/`PieceType`/`MoveKind`, `ArrayVec`'s
  `MaybeUninit`-backed storage, and the AVX-512 VBMI2 move-list fast paths.
  All three buy Reckless real things (compact enum ergonomics, one fewer
  memset, real SIMD throughput on rare hardware) but each also directly
  contradicts a standing project decision (D14's unsafe rationing, and the
  general "safe Rust unless proven necessary" stance) for a benefit we
  haven't measured a need for. Revisit only if profiling ever puts movegen's
  list-construction cost on the critical path, and even then, quantify
  first — our current `[Move; 256]` + `len` approach is simple, safe, and
  has not been shown to be a bottleneck.
