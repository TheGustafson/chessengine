# 02 — Core types: `Value`, `Depth`, extensions to existing types

Status: Reviewed v1

This document specifies the search-oriented core type layer: the `Value` and
`Depth` aliases with their full constant sets and predicates, mate-score
arithmetic (ply-relative from day one, including the TT conversion pair),
`Direction` and safe `Square` arithmetic, color-generic `Bitboard` shift
helpers, flat piece indexing, the raw 16-bit `Move` accessor pair needed by
history tables and the transposition table, the `ExtMove` representation
contract, and the SEE piece-value constants. Everything here is a pure
addition to the existing modules (`types`, `bitboard`, `moves`) — no existing
public API changes — and every constant is the current Stockfish value,
recorded verbatim as a starting point (D20).

## 1. Purpose and scope

This doc owns:

- `Value` and `Depth` type aliases, all their constants (mate/TB bands,
  sentinels, `MAX_PLY`), and the predicates and conversions that operate on
  bare values: `is_valid`, `is_win`, `is_loss`, `is_decisive`, `is_mate`,
  `is_mated`, `mate_in`, `mated_in`, `value_to_tt`, `value_from_tt` (D5, D6).
- `Direction` and `Square` arithmetic: checked stepping, the unchecked-variant
  policy, XOR flip transforms, distance helpers.
- `Bitboard` extensions: `shift` by `Direction`, color-generic pawn push and
  pawn attack helpers, `more_than_one`.
- `Piece::index()` flat 0..12 indexing (D4).
- `Move::raw16()` / `Move::from_raw16()` — the full 16-bit encoding as a
  first-class integer for butterfly-history indexing and TT storage.
- The `ExtMove` representation contract: `#[repr(C)]`, 8 bytes, size-asserted
  (D7). Its behavior and buffer management are NOT owned here.
- SEE piece-value constants (D11).
- Exact file placement for each addition.

This doc does NOT own (with the owning doc, filenames tentative pending the
suite index — see Open questions):

- Attack tables, `BETWEEN`/`LINE`, magic bitboards → the attack-tables doc
  (planned `design/03-attack-tables.md`) (D12).
- `StateInfo`, Zobrist keys, make/unmake → the board-state doc (planned
  `design/04-board-state.md`) (D8, D9, D10).
- Move generation and perft → the movegen doc (planned
  `design/05-movegen-perft.md`).
- The TT entry layout and replacement policy that *consume* `value_to_tt` /
  `value_from_tt` / `Move::raw16` → the TT doc (D19). This doc defines the
  conversion functions themselves because they are pure value arithmetic,
  unit-testable with no TT present.
- MovePicker staging, history tables, and everything `ExtMove` *does* → the
  movepick/history doc (D7, D17).
- SEE (`Board::see_ge`) itself → board-state or its own SEE doc (D11); only
  the piece-value constants live here.
- UCI score formatting (`mate N` output) → the UCI doc.

## 2. References

- `stockfish_reference/stockfish_search_notes.md` §2 (value semantics,
  sentinels, TT conversion, draw dither), §9.1 (butterfly history indexed by
  the full 16-bit move).
- `stockfish_reference/01-bitboards-attacks-move-encoding.md` (move encoding,
  XOR square transforms, `more_than_one`, piece values).
- `stockfish_reference/03-movegen-movepick-history.md` (ExtMove = move +
  score, exactly 8 bytes, lives in the picker).
- SF source verified for exact constants: `Stockfish/src/types.h` (Value and
  Depth constants at lines 116-241, Direction enum at 257-266, `mate_in` /
  `mated_in` at 380-382, piece values at 196-200), `Stockfish/src/search.cpp`
  (`value_to_tt` / `value_from_tt` at 1849-1895, `value_draw` at 132),
  `Stockfish/src/bitboard.h` (shift/pawn-attack templates at 110-134),
  `Stockfish/src/movegen.h` (ExtMove).
- Existing APIs inventoried from `src/types.rs`, `src/bitboard.rs`,
  `src/moves.rs`, `src/board.rs`, `src/lib.rs`.

## 3. Design

### 3.1 Inventory: what already exists

`src/types.rs`:
- `PieceType` (`#[repr(u8)]`, `Pawn=0 .. King=5`, `COUNT=6`, `ALL`,
  FEN char conversions).
- `Color` (`#[repr(u8)]`, `White=0, Black=1`, `COUNT`, `ALL`, `flip()`).
- `Piece { color, piece_type }` (a two-field struct, not a packed byte —
  D4 keeps this; no 16-slot SF-style packing).
- `Square` (private `index: u8`, invariant `< 64` enforced by an `assert!` in
  `Square::new`; `from_coords`, `index()`, `file()`, `rank()`, `all()`,
  named constants `Square::A1..H8`, `Display`/`FromStr`). Mapping is
  little-endian rank-file, a1=0 .. h8=63 (D3).

`src/bitboard.rs`:
- `Bitboard(u64)` newtype: `EMPTY`, `FULL`, `FILE_A/H`, `RANK_1/8`,
  `FILES`/`RANKS` arrays, `from_u64`/`to_u64`, `from_square`, `is_empty`,
  `count`, `contains`, `lsb`, `pop_lsb`, `squares()` iterator +
  `IntoIterator`/`FromIterator`, full bit-op operator set, `Shl`/`Shr` by
  `u32`, board-diagram `Display`.

`src/moves.rs`:
- `Move(u16)`: bits 0-5 from, 6-11 to, 12-13 promo piece − knight, 14-15 kind
  (D2). Constructors `normal`, `promotion`, `en_passant`, `castling`;
  accessors `from()`, `to()`, `kind()`, `promotion_piece()`; `Move::NULL`
  (a1a1 = 0) as the none-sentinel; UCI `Display`. `MoveKind` enum.
- `MoveList`: `[Move; MAX_MOVES]` + len, `MAX_MOVES = 256`, derefs to
  `[Move]`. Stays thin — no scores in it (D7).

`src/board.rs`: `Board` with `pieces(color, piece_type)`, `occupancy(color)`,
`all_occupancy()`, `piece_on(sq)`, `put_piece`/`remove_piece`,
`side_to_move()`, `castling_rights()`, `en_passant_square()`,
`halfmove_clock()`, `fullmove_number()`; `CastlingRights` 4-bit set.

Everything below is additive.

### 3.2 `Value`: alias, constants, bands (src/types.rs)

Per (D5), `Value` is a plain alias — search formulas are dense arithmetic and
newtype ceremony buys nothing here:

```rust
/// A search/eval score. Positive is good for the side to move.
/// All in-search values lie in (-VALUE_NONE, VALUE_NONE] and MUST fit an
/// i16 — the TT stores value16/eval16 (D19).
pub type Value = i32;

pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_MATE: Value = 32000;
pub const VALUE_INFINITE: Value = 32001; // strictly > any real score; window bound only
pub const VALUE_NONE: Value = 32002;     // "no value" sentinel (TT empty fields)

/// Maximum search ply. usize because its primary job is sizing arrays
/// (search stack, ply-indexed tables); cast to Value in score arithmetic (D6).
pub const MAX_PLY: usize = 246;

// Mate band: mate_in(ply)/mated_in(ply) for ply <= MAX_PLY.
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value; // 31754
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;      // -31754

// Tablebase band: sits directly below the mate band.
pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;                  // 31753
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY as Value; // 31507
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;   // -31507
```

All constants are SF-current (verified in `types.h`); derived values are
spelled out in comments so a transcription error is caught by the constants
unit test (§5). The number line, positive side (negatives mirror):

```
0 ..........(normal evals).......... 31506 | 31507 ...(TB band)... 31753 | 31754 ...(mate band)... 32000 | 32001 INF | 32002 NONE
                                            VALUE_TB_WIN_IN_MAX_PLY  VALUE_TB  VALUE_MATE_IN_MAX_PLY   VALUE_MATE
```

Why the bands matter: nearly every search heuristic is gated on "is this
score decisive?" — decisive scores must never be pruned, softened, averaged,
or corrected (search notes §2). Encoding the bands as ranges makes those
gates single comparisons.

Even though we start with no tablebase probing, the TB band is defined from
day one: the band layout determines `is_win`/`is_loss` thresholds and the
`value_from_tt` downgrade target, and retrofitting it would shift every
threshold (same reasoning as D9 for Zobrist keys).

#### Predicates

Free functions in `types` (D5 allows free functions or a trait; free
functions win because they are `const fn`-able today and mirror how the
formulas read in the search literature):

```rust
pub const fn is_valid(v: Value) -> bool { v != VALUE_NONE }

/// TB win or better (includes mates).
pub const fn is_win(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// TB loss or worse (includes being mated).
pub const fn is_loss(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v <= VALUE_TB_LOSS_IN_MAX_PLY
}

pub const fn is_decisive(v: Value) -> bool { is_win(v) || is_loss(v) }

/// Strictly a mate-for-us score (not merely a TB win).
pub const fn is_mate(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v >= VALUE_MATE_IN_MAX_PLY
}

/// Strictly a mated score.
pub const fn is_mated(v: Value) -> bool {
    debug_assert!(is_valid(v));
    v <= VALUE_MATED_IN_MAX_PLY
}
```

Semantics note (matches SF): `is_win` is "TB-win or better", NOT "mate only";
`is_mate` is the narrow mate-band test. Callers that mean "any decisive
score" use `is_decisive`.

#### Mate-score arithmetic

Ply-relative from day one (D5). Inside the search, mate scores mean "plies
from the ROOT":

```rust
/// Score for delivering mate at `ply` plies from the root.
pub const fn mate_in(ply: i32) -> Value { VALUE_MATE - ply }

/// Score for being mated at `ply` plies from the root.
pub const fn mated_in(ply: i32) -> Value { -VALUE_MATE + ply }
```

`mate_in(0) == VALUE_MATE`, `mate_in(MAX_PLY as i32) == VALUE_MATE_IN_MAX_PLY`
— the band is exactly the reachable range. Note `mated_in(p) == -mate_in(p)`.

#### TT conversion pair

The TT stores mate/TB scores relative to the CURRENT node, not the root
(otherwise the same position reached at different plies would disagree).
These are pure value functions, so they live here and are unit-tested here;
the TT doc consumes them at store/probe time.

```rust
/// Root-relative -> node-relative, applied before a TT store.
pub const fn value_to_tt(v: Value, ply: i32) -> Value {
    debug_assert!(is_valid(v));
    if is_win(v) { v + ply } else if is_loss(v) { v - ply } else { v }
}

/// Node-relative -> root-relative, applied after a TT probe. Additionally
/// downgrades decisive scores that the 50-move clock has made unreachable
/// (graph-history interaction): if the mate/TB conversion needs more plies
/// than remain on the clock, collapse to just below/above the TB band.
pub const fn value_from_tt(v: Value, ply: i32, rule50: i32) -> Value {
    if !is_valid(v) {
        return VALUE_NONE;
    }
    if is_win(v) {
        // 100 plies = the 50-move rule; rule50 already used up.
        if is_mate(v) && VALUE_MATE - v > 100 - rule50 {
            return VALUE_TB_WIN_IN_MAX_PLY - 1; // stale mate score
        }
        if VALUE_TB - v > 100 - rule50 {
            return VALUE_TB_WIN_IN_MAX_PLY - 1; // stale TB score
        }
        return v - ply;
    }
    if is_loss(v) {
        if is_mated(v) && VALUE_MATE + v > 100 - rule50 {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        if VALUE_TB + v > 100 - rule50 {
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        return v + ply;
    }
    v
}
```

The downgrade target `VALUE_TB_WIN_IN_MAX_PLY - 1` (= 31506) is deliberate:
the highest possible NON-decisive score, so the stale score still dominates
normal evals but no longer trips the "never prune decisive scores" gates.
(Formula and constant per SF `search.cpp:1860-1893`, SF-current, D20.)

Related but NOT owned here: the draw-dither `VALUE_DRAW - 1 + (nodes & 0x2)`
(±1 pseudo-random from the node counter, avoids 3-fold blindness) is a search
function using search state — the search doc owns it. Same for UCI mate
display `(VALUE_MATE - v + 1) / 2`.

### 3.3 `Depth` and ply (src/types.rs)

Per (D6):

```rust
/// Remaining search depth in plies. i32 because sentinels are negative and
/// depth arithmetic (reductions, extensions) freely goes through zero.
pub type Depth = i32;

/// Depth of a quiescence-search TT entry.
pub const DEPTH_QS: Depth = 0;
/// TT entries written without any search (standalone eval save).
/// Must compare below every real search depth including DEPTH_QS.
pub const DEPTH_UNSEARCHED: Depth = -2;
/// "No depth" marker for empty/invalid TT entries; below DEPTH_UNSEARCHED.
pub const DEPTH_NONE: Depth = -3;
```

Conventions fixed here, enforced by review:

- `depth: Depth` = remaining depth. May be fractional-free negative only via
  the sentinels above; a real search never recurses with depth < 0 (qsearch
  is entered instead).
- `ply: usize` when indexing arrays (search stack, `lowPlyHistory`), `i32`
  inside value arithmetic (`mate_in(ply as i32)`). Casts happen at the use
  site, not by storing two copies.
- The TT packs depth into a `u8` as `depth - DEPTH_NONE_OFFSET` where the
  offset equals `DEPTH_NONE` (so representable depths are `-3..=252`, and
  `MAX_PLY = 246` fits). That packing is TT-internal and owned by the TT doc;
  it is noted here only as the constraint that keeps `MAX_PLY <= 252`.

### 3.4 `Direction` and `Square` arithmetic (src/types.rs)

#### The `Direction` enum

```rust
/// A one-square step, encoded as the board-index delta (a1=0 mapping, D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Direction {
    North = 8,
    NorthEast = 9,
    East = 1,
    SouthEast = -7,
    South = -8,
    SouthWest = -9,
    West = -1,
    NorthWest = 7,
}

impl Direction {
    pub const ALL: [Direction; 8] = [
        Direction::North, Direction::NorthEast, Direction::East,
        Direction::SouthEast, Direction::South, Direction::SouthWest,
        Direction::West, Direction::NorthWest,
    ];

    /// The raw index delta.
    pub const fn offset(self) -> i8 { self as i8 }
}

impl Color {
    /// The forward pawn-push direction for this color.
    pub const fn pawn_push(self) -> Direction {
        match self {
            Color::White => Direction::North,
            Color::Black => Direction::South,
        }
    }
}
```

Knight steps (±17, ±15, ±10, ±6) are NOT `Direction`s — they are raw `i8`
step values used only by table builders, which is why the checked stepping
API below takes `i8`, not `Direction`.

#### Checked stepping: `Square::try_offset`

The single safe primitive for table building (leaper tables, rays, between/
line). It must reject two distinct failure modes: leaving the 0..64 range,
and *wrapping* across the a/h edge (which stays in range but teleports to the
other side of the board). The wrap test is a Chebyshev-distance bound: every
legitimate single step (queen direction: distance 1; knight: distance 2) moves
at most 2 files/ranks, while any wrap jumps at least 6 files.

```rust
impl Square {
    /// File distance |file(a) - file(b)|.
    pub const fn file_distance(self, other: Square) -> u8 {
        self.file().abs_diff(other.file())
    }

    /// Rank distance |rank(a) - rank(b)|.
    pub const fn rank_distance(self, other: Square) -> u8 {
        self.rank().abs_diff(other.rank())
    }

    /// Chebyshev distance: king moves between the squares.
    /// (The attack-tables doc may later back this with a 64x64 table; the
    /// const fn stays as the reference implementation.)
    pub const fn distance(self, other: Square) -> u8 {
        let f = self.file_distance(other);
        let r = self.rank_distance(other);
        if f > r { f } else { r }
    }

    /// Step by a raw index delta. None if the destination is off-board OR
    /// the step wrapped around a board edge (Chebyshev distance > 2).
    pub const fn try_offset(self, step: i8) -> Option<Square> {
        let to = self.index() as i8 + step;
        if to < 0 || to >= 64 {
            return None;
        }
        let to = Square::new(to as u8);
        if self.distance(to) > 2 { None } else { Some(to) }
    }

    /// Step by a Direction. None if off-board or wrapped.
    pub const fn shifted(self, d: Direction) -> Option<Square> {
        self.try_offset(d.offset())
    }
}
```

Note the distance-2 bound admits the double pawn push (±16, distance 2,
cannot wrap) and knight steps, and rejects every single-step wrap. Callers
that iterate a slider ray apply `shifted` one step at a time, so distance per
call is always 1 and the bound is never a false negative.

#### Unchecked variant policy (D14)

Hot paths (movegen recovering a pawn's origin from a shifted bitboard, SEE
walking attack rays) know by construction that the step is valid. Policy:

```rust
impl Square {
    /// `self + d` where the CALLER guarantees the destination exists and
    /// does not wrap (e.g. derived from an already-masked bitboard shift).
    /// Range escapes still panic via Square::new's assert; file wrap is NOT
    /// detected here.
    pub const fn offset(self, d: Direction) -> Square {
        Square::new((self.index() as i8 + d.offset()) as u8)
    }
}
```

- This is safe Rust: an out-of-range result panics in `Square::new`
  (invariant preserved), a wrapped result is a caller bug that the perft
  suite catches (D24).
- We do NOT introduce a `debug_assert`-only or `unsafe` constructor now.
  If a profile ever shows `Square::new`'s compare-and-branch in movegen as
  measurable, the relaxation is: a private `const fn from_index_unchecked`
  with `debug_assert!` (still no `unsafe` — a bad `Square` can only cause a
  bounds-check panic downstream, not UB), plus a comment recording the
  measurement per (D14).

#### XOR transforms

Branchless coordinate flips (facts from SF `types.h`, standard bitboard
lore):

```rust
impl Square {
    /// Vertical mirror: a1 <-> a8.
    pub const fn flip_rank(self) -> Square { Square::new(self.index() ^ 56) }

    /// Horizontal mirror: a1 <-> h1.
    pub const fn flip_file(self) -> Square { Square::new(self.index() ^ 7) }

    /// The square from `color`'s point of view: identity for White,
    /// vertical mirror for Black. E2.relative_to(Black) == E7.
    pub const fn relative_to(self, color: Color) -> Square {
        Square::new(self.index() ^ (color as u8 * 56))
    }

    /// Rank from `color`'s point of view (0 = own back rank).
    pub const fn relative_rank(self, color: Color) -> u8 {
        self.rank() ^ (color as u8 * 7)
    }
}
```

These exploit `Color::White as u8 == 0`, `Color::Black as u8 == 1` (already
guaranteed by the existing `#[repr(u8)]` enum) — no branches. PSQT lookup,
promotion-rank tests, and pawn-structure code all key off `relative_rank`.

### 3.5 `Bitboard` extensions (src/bitboard.rs)

```rust
impl Bitboard {
    /// True iff at least two bits are set. One AND + compare — cheaper and
    /// more explicit than count() >= 2. Primary use: double-check detection
    /// (evasion generation), "lone attacker" tests in SEE.
    pub const fn more_than_one(self) -> bool {
        (self.0 & self.0.wrapping_sub(1)) != 0
    }

    /// Shift every set square one step in `d`. Squares that would leave the
    /// board vanish; east/west components mask the wrapping file first.
    pub const fn shift(self, d: Direction) -> Bitboard {
        const NOT_FILE_A: u64 = !Bitboard::FILE_A.to_u64(); // 0xFEFE_FEFE_FEFE_FEFE
        const NOT_FILE_H: u64 = !Bitboard::FILE_H.to_u64(); // 0x7F7F_7F7F_7F7F_7F7F
        Bitboard::from_u64(match d {
            Direction::North => self.0 << 8,
            Direction::South => self.0 >> 8,
            Direction::East => (self.0 & NOT_FILE_H) << 1,
            Direction::West => (self.0 & NOT_FILE_A) >> 1,
            Direction::NorthEast => (self.0 & NOT_FILE_H) << 9,
            Direction::NorthWest => (self.0 & NOT_FILE_A) << 7,
            Direction::SouthEast => (self.0 & NOT_FILE_H) >> 7,
            Direction::SouthWest => (self.0 & NOT_FILE_A) >> 9,
        })
    }

    /// Single pawn pushes for `color` pawns in `self`.
    pub const fn pawn_pushes(self, color: Color) -> Bitboard {
        self.shift(color.pawn_push())
    }

    /// Squares attacked by `color` pawns in `self` (both diagonals, bulk).
    pub const fn pawn_attacks(self, color: Color) -> Bitboard {
        match color {
            Color::White => Bitboard::from_u64(
                self.shift(Direction::NorthEast).0 | self.shift(Direction::NorthWest).0,
            ),
            Color::Black => Bitboard::from_u64(
                self.shift(Direction::SouthEast).0 | self.shift(Direction::SouthWest).0,
            ),
        }
    }
}
```

Design notes:

- SF makes `shift` a template over the direction so the match is resolved at
  compile time. Rust's equivalent (enum const generics) is unstable, but a
  `match` on a literal enum variant constant-folds identically under
  `opt-level=3`: `bb.shift(Direction::NorthEast)` compiles to one AND + one
  SHL. Movegen call sites therefore always pass a literal or a value the
  compiler can prove constant (e.g. after matching on `Color`). This is a
  documented expectation, checked once via asm inspection (§6).
- `wrapping_sub` in `more_than_one` keeps the function panic-free on
  `EMPTY` in debug builds (`0u64 - 1` would otherwise be an overflow panic);
  `u64::MAX & 0 == 0` gives the right answer.
- `pawn_pushes` does not mask occupancy — movegen ANDs with `!occupied`
  itself, because double pushes need the intermediate emptiness test anyway.
- The internal `.0` field accesses are fine inside `bitboard.rs`; operator
  traits (`BitOr` etc.) are not `const fn`, so const helpers use the raw u64.

### 3.6 `Piece::index()` — flat 0..12 (src/types.rs)

Per (D4), storage stays `[[Bitboard; 6]; 2]`; flat indexing exists only as a
projection for piece-indexed tables (capture history `[piece][to][captured]`,
continuation history `[piece][to]`, NNUE feature indexing later):

```rust
impl Piece {
    /// Number of distinct (color, piece_type) pairs.
    pub const COUNT: usize = 12;

    /// Flat table index: White P,N,B,R,Q,K = 0..6, Black P..K = 6..12.
    pub const fn index(self) -> usize {
        self.color as usize * PieceType::COUNT + self.piece_type as usize
    }
}
```

We deliberately do NOT adopt SF's 16-slot packed piece (bit 3 = color, gap
slots wasted): our tables are `[T; 12]` — denser, and the multiply is free
against a table lookup. No `from_index` inverse until something needs it.

### 3.7 `Move::raw16()` / `Move::from_raw16()` (src/moves.rs)

The butterfly main history is `[Color::COUNT][65536]`, indexed by the FULL
16-bit move value — not just from/to — so that promotions to different pieces
and the castling flag get distinct history slots (search notes §9.1). The TT
stores the best move as a raw 16-bit field (`move16`, D19). Both need the
encoding as an integer:

```rust
impl Move {
    /// The full 16-bit encoding. Index into butterfly history
    /// [color][65536]; the TT's move16 field.
    pub const fn raw16(self) -> u16 { self.0 }

    /// Rebuild a Move from raw bits (TT probe path). Every u16 decodes to a
    /// structurally valid Move (from/to are 6-bit fields, all kind values
    /// are meaningful), so this is total. SEMANTIC validity is the caller's
    /// job: TT consumers must run pseudo-legality + legality checks on any
    /// probed move before playing it (D19).
    pub const fn from_raw16(bits: u16) -> Move { Move(bits) }
}
```

Both are trivially `const fn` and stay inside `moves.rs` where the field is
visible. Sentinel bookkeeping (fact-check against D2): `Move::NULL` (0,
a1→a1) is the none-sentinel. When null-move pruning needs a DISTINCT
"null move was played here" marker, the reserved pattern is another
from==to encoding, canonically `Move::from_raw16(65)` (b1→b1); defining that
constant is owned by the search doc, but the raw16 pair above is what makes
it expressible.

### 3.8 `ExtMove`: representation contract only (D7)

`ExtMove` lives in the future `src/movepick.rs`; its staging/partial-sort
behavior is the movepick doc's business. This doc fixes only the repr so the
movepick doc and any sort-performance reasoning can rely on it:

```rust
// src/movepick.rs (future). Repr contract fixed by design/02-core-types.md.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ExtMove {
    pub mv: Move,   // 2 bytes; 2 bytes padding follow
    pub value: i32, // move-ordering score. NOT a Value: it mixes history
                    // sums and MVV bonuses on scales far beyond eval range.
}

const _: () = assert!(std::mem::size_of::<ExtMove>() == 8);
const _: () = assert!(std::mem::align_of::<ExtMove>() == 4);
```

Rationale for the asserts: the picker's insertion sort swaps `ExtMove`s in
its `[ExtMove; MAX_MOVES]` buffer millions of times per second; 8-byte
elements move as single u64 register loads/stores. `#[repr(C)]` pins the
layout so the size can't silently grow if field order changes. `MoveList`
stays score-free (D7) — scoring happens only inside the picker's own buffer.

### 3.9 SEE piece values (src/types.rs)

Per (D11), with values recorded as SF-current starting points (D20):

```rust
pub const PAWN_VALUE: Value = 208;
pub const KNIGHT_VALUE: Value = 781;
pub const BISHOP_VALUE: Value = 825;
pub const ROOK_VALUE: Value = 1276;
pub const QUEEN_VALUE: Value = 2538;

/// Exchange values indexed by PieceType as usize. King = 0: SEE sequences
/// terminate before a king is ever "captured", and a 0 keeps loops total.
pub const PIECE_VALUES: [Value; PieceType::COUNT] = [
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, VALUE_ZERO,
];

impl PieceType {
    /// Exchange (SEE) value of this piece type.
    pub const fn value(self) -> Value { PIECE_VALUES[self as usize] }
}
```

Constraint inherited from SF (documented so tuning doesn't break it): the
values are chosen such that summed non-pawn material uniquely identifies
material configurations — `Board`'s `non_pawn_material` (board-state doc)
doubles as a material signature for eval/time-management heuristics. Any
retuning must preserve that property.

Placement rationale vs (D20): D20 centralizes *search* tunables in
`search/params.rs`. Piece values are consumed below the search layer (SEE on
`Board`, capture ordering in movepick, phase A material eval), so they belong
with `Value` in `types.rs`, exactly as SF keeps them in `types.h`. They are
still named constants, greppable, and listed in the tuning inventory.

### 3.10 File placement summary

| Addition | File | Section |
|---|---|---|
| `Value`, `Depth` aliases + all constants | `src/types.rs` | 3.2, 3.3 |
| `is_valid/is_win/is_loss/is_decisive/is_mate/is_mated` | `src/types.rs` | 3.2 |
| `mate_in/mated_in`, `value_to_tt/value_from_tt` | `src/types.rs` | 3.2 |
| `MAX_PLY` | `src/types.rs` | 3.2 |
| `Direction`, `Color::pawn_push` | `src/types.rs` | 3.4 |
| `Square::{try_offset, shifted, offset, distance, file_distance, rank_distance, flip_rank, flip_file, relative_to, relative_rank}` | `src/types.rs` | 3.4 |
| `Bitboard::{shift, pawn_pushes, pawn_attacks, more_than_one}` | `src/bitboard.rs` | 3.5 |
| `Piece::{COUNT, index}` | `src/types.rs` | 3.6 |
| `Move::{raw16, from_raw16}` | `src/moves.rs` | 3.7 |
| `ExtMove` + size/align asserts | `src/movepick.rs` (future, with the movepick doc) | 3.8 |
| `PIECE_VALUES`, `PieceType::value()` | `src/types.rs` | 3.9 |

`src/lib.rs` re-exports gain exactly one name: `Direction`
(`pub use types::{Color, Direction, Piece, PieceType, Square};`). Constants
and value predicates stay under the `types` path — search code will
`use crate::types::*;` locally rather than polluting the crate root with
`is_win` etc.

## 4. Implementation plan

Each step is one commit; `cargo test` and `cargo clippy --all-targets` green
is implied in every done-condition.

1. **Value/Depth layer.** Add aliases, all constants of §3.2/§3.3, the six
   predicates, `mate_in`/`mated_in`, `value_to_tt`/`value_from_tt`, with doc
   comments and the unit tests of §5 items 1-4.
   Done when: constant-table test asserting the literal derived values
   (31754/31753/31507) passes; i16-fit static assert compiles.
2. **Piece values.** `PAWN_VALUE..QUEEN_VALUE`, `PIECE_VALUES`,
   `PieceType::value()`; §5 item 5 tests.
   Done when: `PieceType::Queen.value() == 2538` style assertions pass and
   the array/individual constants cross-check test passes.
3. **Direction + Square arithmetic.** `Direction`, `Color::pawn_push`,
   `try_offset`/`shifted`/`offset`, distances, XOR flips; §5 items 6-8 tests.
   Done when: exhaustive 64-square × 8-direction property test matches the
   coordinate-space reference model.
4. **Bitboard helpers.** `shift`, `pawn_pushes`, `pawn_attacks`,
   `more_than_one`; §5 items 9-10 tests.
   Done when: bulk-shift-vs-per-square-`shifted` equivalence test passes on
   randomized bitboards.
5. **Move raw accessors + Piece::index.** `raw16`/`from_raw16`,
   `Piece::{COUNT, index}`; §5 items 11-12 tests. Update `lib.rs` re-exports
   (add `Direction`).
   Done when: u16 round-trip and total-decoding tests pass; the 12 flat
   indices are distinct and in range.
6. **(Deferred, tracked)** `ExtMove` with its size/align asserts lands with
   the first commit of the movepick module, not now.
   Done when: the const asserts in §3.8 compile in `src/movepick.rs`.

No step touches existing code paths, so no behavioral gate applies yet; from
step 3 onward these primitives become load-bearing for the attack-tables and
movegen docs, after which the perft suite (D24) is the change gate for any
edit to them.

## 5. Testing and validation

Unit tests (in-module `#[cfg(test)]`, existing style: named square constants,
no struct literals):

1. **Constant table.** Assert every derived constant against its literal:
   `VALUE_MATE_IN_MAX_PLY == 31754`, `VALUE_MATED_IN_MAX_PLY == -31754`,
   `VALUE_TB == 31753`, `VALUE_TB_WIN_IN_MAX_PLY == 31507`,
   `VALUE_TB_LOSS_IN_MAX_PLY == -31507`; ordering chain
   `VALUE_TB_WIN_IN_MAX_PLY < VALUE_TB < VALUE_MATE_IN_MAX_PLY < VALUE_MATE
   < VALUE_INFINITE < VALUE_NONE`. Compile-time
   `const _: () = assert!(VALUE_NONE <= i16::MAX as i32);` (TT i16 storage,
   D19).
2. **Predicates.** Band-edge cases: `is_win(VALUE_TB_WIN_IN_MAX_PLY)`,
   `!is_win(VALUE_TB_WIN_IN_MAX_PLY - 1)`, `is_mate(mate_in(MAX_PLY as i32))`,
   `!is_mate(VALUE_TB)`, `is_decisive` symmetric on ± values,
   `!is_decisive(VALUE_DRAW)`, `!is_decisive(1000)`;
   `mated_in(p) == -mate_in(p)` for several p.
3. **TT conversion round-trip.** For v in {`mate_in(50)`, `mated_in(45)`,
   `VALUE_TB - 50`, `-(VALUE_TB - 50)`, 0, 130, -2500} and ply in
   {0, 1, 5, 40}: `value_from_tt(value_to_tt(v, ply), ply, 0) == v`.
   (The decisive test values have distance-from-root >= 40, unlike the
   originally drafted `mate_in(3)`/`VALUE_TB - 5`: a root-relative decisive
   score with distance k is only ever stored at plies `<= k` — a node at
   ply p only sees mates/TB wins at k >= p — and `value_to_tt` ADDS ply, so
   storing deeper overshoots the value's own band. `value_to_tt(mate_in(3),
   5)` lands exactly on `VALUE_NONE` (32002) and reads back as the
   sentinel; `value_to_tt(VALUE_TB - 5, 40)` = 31788 lands in the mate band
   where the rule-50 mate guard fires (apparent mate distance 212 > 100)
   and downgrades instead of round-tripping. The latter is asserted in
   item 4.)
4. **TT downgrade.** `value_from_tt(value_to_tt(mate_in(10), 4), 4, 95)`
   collapses to `VALUE_TB_WIN_IN_MAX_PLY - 1` (mate needs ~31990 > 5 plies of
   clock); mirrored loss case gives `VALUE_TB_LOSS_IN_MAX_PLY + 1`; stale TB
   scores with the band intact but the clock expired (`VALUE_TB - 5` at
   rule50 = 98) downgrade the same way, both signs; the out-of-band store
   from item 3's note (`VALUE_TB - 5` stored at ply 40) reads back as
   `VALUE_TB_WIN_IN_MAX_PLY - 1`; `value_from_tt(VALUE_NONE, p, r) ==
   VALUE_NONE`.
5. **Piece values.** `PIECE_VALUES` agrees with the individual constants;
   `PieceType::King.value() == 0`; values strictly increase P<N<B<R<Q.
6. **Square stepping, exhaustive.** For all 64 squares × 8 directions,
   compare `sq.shifted(d)` against a coordinate-space model (compute
   file/rank deltas as i8, rebuild with `from_coords` when in range) — this
   catches both wrap and range bugs. Spot checks: `H4.shifted(East) == None`,
   `H4.shifted(NorthEast) == None`, `A1.shifted(SouthWest) == None`,
   `E4.shifted(North) == Some(E5)`.
7. **Knight steps + wrap.** `Square::G1.try_offset(17) == Some(Square::H3)`;
   `Square::H1.try_offset(10) == None` (wraps to b3, distance 6);
   `Square::A8.try_offset(15) == None` (off-board); double push
   `Square::E2.try_offset(16) == Some(Square::E4)`.
8. **XOR transforms.** `A1.flip_rank() == A8`, `A1.flip_file() == H1`,
   `E2.relative_to(Color::Black) == E7`,
   `E2.relative_to(Color::White) == E2`,
   `relative_rank`: `E2` is rank 1 for White, rank 6 for Black; involution
   `s.flip_rank().flip_rank() == s` over all squares.
9. **Bitboard shift.** `RANK_1.shift(North) == RANKS[1]`;
   `FILE_H.shift(East) == EMPTY`; `FULL.shift(East) == !FILE_A`;
   equivalence property: for 1000 SplitMix64-generated random bitboards (tiny
   inline PRNG in the test module — no dev-dependency needed, D15) and all 8
   directions, `bb.shift(d)` equals the union of `sq.shifted(d)` over
   `bb.squares()`.
10. **Pawn helpers + more_than_one.**
    `pawn_pushes`/`pawn_attacks` for both colors against per-square models;
    edge case: a-file white pawn attacks only NE. `more_than_one`: false for
    `EMPTY` and every single square, true for any 2-bit board and `FULL`.
11. **Move raw16.** Round-trip `from_raw16(m.raw16()) == m` for one move of
    each kind including all four promotion pieces; totality:
    `Move::from_raw16(0xFFFF)` decodes without panic and reports
    `MoveKind::Castling` with from/to == h8 (structural validity only);
    distinct kinds on the same from/to yield distinct raw16 values.
12. **Piece::index.** All 12 `(color, piece_type)` pairs give distinct
    indices covering exactly 0..12; `White/Pawn == 0`, `Black/King == 11`.

Regression strategy: these are leaf primitives with exhaustive/property
tests, so the component-level gate is `cargo test` + clippy. They become
perft-gated (D24) transitively the moment movegen consumes them: any later
edit to `shift`/`try_offset`/`offset` requires the full perft suite to pass.
The bench oracle (D24) applies from the search milestone onward; nothing in
this doc may ever change bench node counts once search exists, because none
of it is heuristic — a bench delta from this layer means a bug.

## 6. Performance notes

Targets:

- Every function in this doc must inline to branch-free straight-line code
  (or one predictable branch for the `Square::new` assert): `shift` with a
  literal direction = AND+SHL (2 instr), XOR transforms = 1 instr,
  `more_than_one` = LEA/AND+SETNE, predicates = CMP+SETcc.
- Zero memory traffic: no statics, no tables in this layer.

Pitfalls and mitigations:

- **Runtime-variant `shift` in a hot loop** becomes a jump table. Movegen
  must branch on `Color`/direction OUTSIDE its per-square loops so each
  `shift` call site sees a constant. Verified once by inspecting
  `cargo rustc --release -- --emit asm` (or `cargo asm` if installed as a
  tool, not a dependency) for a movegen-like function: expect no `jmp`
  through a table for the pawn-push path.
- **Unit confusion** (`Value` vs `Depth` vs ply) is the cost of aliases over
  newtypes (D5). Conventions: parameters are named `value/v/eval`,
  `depth`, `ply`; formulas never mix them without an explicit cast; reviews
  enforce it. If this ever bites badly, a newtype migration is mechanical
  but is NOT planned.
- **i32 overflow headroom**: `VALUE_INFINITE + MAX_PLY`-scale sums are ~3e4,
  five orders below i32 limits; history-scale products (~3e4 × ~3e4) still
  fit. The search doc owns keeping its formula intermediates in range.
- **`Square::new` assert in `offset`**: a compare+branch that is
  never-taken and thus ~free; policy for removing it is fixed in §3.4 (D14)
  — profile first, `debug_assert` second, never `unsafe`.
- **`ExtMove` at 8 bytes** keeps picker sorts register-only; the const
  asserts make any accidental growth a compile error.

What to measure (later, once perft/bench exist): perft nps before/after
movegen adopts these helpers (expect: no measurable delta vs hand-inlined
masks, because it IS the same code after inlining); TT probe path cost of
`value_from_tt` (expect: negligible, branch-predictable).

## 7. Interactions

Depends on: nothing — this is the foundation layer. Complies with
`design/decisions.md` (D2-D7, D11, D14, D20).

Feeds (tentative filenames, see Open questions):

- `design/03-attack-tables.md`: `Direction`, `Square::try_offset` (table
  builders), `distance` reference impl, `Bitboard::shift`.
- `design/04-board-state.md`: `Value` for `non_pawn_material` (uses
  `PIECE_VALUES`), `Piece::index` unused there but `relative_rank` feeds
  promotion/ep logic; SEE (`see_ge`, D11) consumes `PIECE_VALUES` and
  `more_than_one`.
- `design/05-movegen-perft.md`: `pawn_pushes`/`pawn_attacks` bulk shifts,
  `Square::offset` unchecked stepping, `Color::pawn_push`.
- TT doc: `value_to_tt`/`value_from_tt`, `Move::{raw16, from_raw16}`,
  `DEPTH_QS/UNSEARCHED/NONE`, the i16-fit constraint, `is_valid` (D19).
- Search doc: the whole `Value` band system and predicates, `mate_in`/
  `mated_in`, `MAX_PLY` (stack size `MAX_PLY + 10`, offset +7, D18), the
  reserved null-move sentinel pattern (§3.7).
- Movepick/history doc: `ExtMove` contract (§3.8), `Piece::index`,
  `Move::raw16` butterfly indexing, `PieceType::value()` for MVV ordering.
- Eval doc: `Value`, piece values (phase A material), `relative_rank`/PSQT
  mirroring via `relative_to`.
- UCI doc: mate-distance display derives from `VALUE_MATE` and
  `is_decisive`.

## 8. Acceptance criteria (D26)

This component is done when all of the following hold:

1. All additions of §3.10 exist in the stated files with doc comments on
   every public item; `src/movepick.rs`'s `ExtMove` asserts are explicitly
   deferred to the movepick milestone and tracked there.
2. The constant-table unit test pins the exact values: 32000 / 32001 / 32002
   / 0 / 246 / 31754 / −31754 / 31753 / 31507 / −31507 / 0 / −2 / −3 / 208 /
   781 / 825 / 1276 / 2538 — and passes.
3. Compile-time asserts in place: `VALUE_NONE` fits i16; (deferred with
   movepick) `size_of::<ExtMove>() == 8`.
4. All §5 tests pass under `cargo test`; the exhaustive 64×8 stepping test
   and the randomized bulk-shift equivalence test are present, not just spot
   checks.
5. `cargo clippy --all-targets` is clean; zero new runtime dependencies
   (D15); no `unsafe` blocks introduced (D14).
6. Const-evaluability demonstrated: a test declares
   `let _stack = [0u8; MAX_PLY + 10];` and a `const` computed via
   `Bitboard::shift` / `mate_in`, proving downstream docs can build const
   tables on these primitives.
7. Existing public APIs and all pre-existing tests are untouched and green
   (pure-addition property, D1).
8. `is_win`/`is_loss` gate semantics ("TB-win or better", not "mate only")
   are documented on the functions themselves, since misreading them corrupts
   every pruning gate written later.

## 9. Open questions

1. **Sibling doc filenames.** Only `decisions.md` exists in `design/` at
   time of writing; the filenames cited in §1/§7 (`03-attack-tables.md`,
   `04-board-state.md`, `05-movegen-perft.md`, and the TT/search/movepick
   docs) are tentative and must be reconciled when the suite index lands.
   No design content hinges on the names.
2. **`ValueExt` trait later?** D5 permits a trait instead of free functions.
   Decided: free functions now (const-fn friendly, formula-shaped call
   sites). If method syntax (`v.is_win()`) proves nicer once search code
   exists, adding a thin trait over the same free functions is
   non-breaking; revisit at the search milestone.
3. **Where `distance` ultimately lives.** The const fn here is the reference
   implementation; the attack-tables doc may add a 64×64 lookup if profiling
   favors it (SF uses a table). The API is identical either way, so this is
   an implementation detail deferred to that doc.
