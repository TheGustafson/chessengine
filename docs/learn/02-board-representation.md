# 02 — Board representation

A chessboard is 8×8 = 64 squares, each empty or holding one of twelve kinds of
piece (six types × two colors). That is a tiny amount of information — you could
store it in 64 bytes. The interesting question is not *whether* you can
represent a board, but how to represent it so that the operations an engine
does billions of times — "which squares does this bishop attack?", "is this
square defended?", "generate all pawn moves" — are as close to free as
possible. The answer, universal among strong engines, is the **bitboard**. This
file builds it from first principles, then explains why ARCHON pairs it with a
second, redundant representation.

## Squares as numbers 0–63

First we need a name for each square that a computer likes. ARCHON numbers the
squares 0 to 63 using the **little-endian rank-file** mapping (often "LERF"):

- **a1 = 0**, b1 = 1, …, h1 = 7 (the first rank, left to right),
- a2 = 8, …, h2 = 15 (the second rank),
- … up to a8 = 56, …, **h8 = 63**.

A **file** is a column (a–h); a **rank** is a row (1–8). The mapping is just
`index = rank * 8 + file`, with rank and file each 0–7. This choice has a payoff
that justifies it: because the file is the low three bits and the rank is the
next three bits, you extract them with a mask and a shift, no division:

```rust
pub const fn file(self) -> u8 { self.index & 0b111 }  // low 3 bits
pub const fn rank(self) -> u8 { self.index >> 3 }      // next 3 bits
```

(From `src/types.rs`. `const fn` means these can run at compile time, so tables
built from them cost nothing at runtime — a recurring theme.) ARCHON wraps the
raw index in a `Square` newtype whose constructor asserts `index < 64`, so an
out-of-range square cannot exist. The named constants `Square::E4`,
`Square::A1`, and so on make the code and tests readable. "Little-endian" here
means square 0 is the least-significant end and matches bit 0 of a 64-bit word —
which is the whole point, as we are about to see.

Why this mapping and not, say, a8 = 0 counting downward (which reads more like a
printed diagram)? Because LERF makes bit arithmetic line up with board geometry:
shifting a number left by 8 moves every piece up one rank, shifting by 1 moves
east one file, and so on. Several engines use the a8 = 0 convention instead;
ARCHON commits to LERF and never wavers, because inconsistency here is a
bottomless source of bugs (`design/02` pins it as decision D3).

## Bitboards: a board is a set of squares

Here is the key idea. A 64-bit unsigned integer has exactly 64 bits. A
chessboard has exactly 64 squares. So **one `u64` can represent any subset of
squares**: bit `i` is 1 if square `i` is "in the set," 0 if not. That integer is
a **bitboard**.

ARCHON's is a newtype over `u64` (from `src/bitboard.rs`):

```rust
pub struct Bitboard(u64);
```

What subset would you want to store? All of them. A real board is represented as
a *collection* of bitboards, one per kind of thing you want to ask about:

- the set of squares with a white pawn,
- the set with a black knight,
- the set of all white pieces,
- the set of all occupied squares,

and so on. To reconstruct the full position you keep one bitboard per
(color, piece-type) pair — twelve of them — plus a couple of aggregate boards.
More on that layout below.

The reason this representation is transformative is that **set operations become
single machine instructions.** If `white_pawns` and `black_pieces` are
bitboards, then:

- **union** (squares in either set) is bitwise OR: `white_pawns | black_pieces`
- **intersection** (squares in both) is bitwise AND: `white_pawns & black_pieces`
- **difference** (in the first but not the second) is `white_pawns & !black_pieces`
- **complement** (every square *not* in the set) is bitwise NOT: `!occupied`

Each of these operates on all 64 squares at once, in one CPU instruction. That
is the magic: geometry becomes arithmetic, and arithmetic on 64-bit words is
what CPUs are fastest at. "Which of my pawns are attacked by an enemy piece?"
becomes `my_pawns & enemy_attacks` — one AND. "Which empty squares can this
knight move to?" becomes `knight_attacks & !my_pieces`. Loops over squares
collapse into bit tricks.

`src/bitboard.rs` implements Rust's operator traits (`BitOr`, `BitAnd`,
`BitXor`, `Not`, and their assigning cousins) on the `Bitboard` newtype so you
write `a | b` and `a & !b` directly and the type system keeps you from
accidentally mixing a bitboard up with some other `u64`. The idiom is: expose
*behavior* (set operations) and hide the *representation* (the raw integer),
which is a theme in this codebase — the field inside `Bitboard(u64)` is private.

### A picture

Bitboards are easier to believe once you see one printed. ARCHON's `Display` for
`Bitboard` prints rank 8 at the top down to rank 1, with `X` for a set bit. Here
is the set of squares a knight on d4 attacks:

```
. . . . . . . .
. . . . . . . .
. . X . X . . .
. X . . . X . .
. . . . . . . .      the knight is on d4 (not shown);
. X . . . X . .      X marks each square it attacks
. . X . X . . .
. . . . . . . .
```

As a `u64` this is one specific 64-bit number, and ARCHON has it precomputed for
every one of the 64 possible knight squares. File 03 is all about how such
attack sets are computed; here we care only that the *result* is a bitboard you
can AND and OR with everything else.

## Three bit tricks you will see everywhere

Three primitive operations on bitboards show up constantly, and modern CPUs have
dedicated instructions for all three.

**Population count ("popcount"): how many squares are in the set?** This is
`u64::count_ones()`, one instruction (`POPCNT`). ARCHON exposes it as
`Bitboard::count()`. "How many pawns do I have?" is `my_pawns.count()`. It also
powers a cheap "are there at least two bits set?" test,
`more_than_one()`, implemented as `(x & x.wrapping_sub(1)) != 0` — a classic
trick where `x & (x-1)` clears the lowest set bit, so the result is nonzero iff
a second bit remains. That test drives double-check detection (a king attacked
by two pieces at once) without a full popcount.

**Least-significant bit: which is the lowest square in the set?** `u64`'s
`trailing_zeros()` counts the zeros below the lowest 1-bit, which is exactly its
square index. `Bitboard::lsb()` returns it as a `Square` (or `None` if the board
is empty).

**Pop the LSB: remove and return the lowest square.** This is how you *iterate*
over a bitboard. Take the lowest set square, then clear it with the same
`x &= x - 1` trick, and repeat until the board is empty:

```rust
pub fn pop_lsb(&mut self) -> Option<Square> {
    let sq = self.lsb()?;
    self.0 &= self.0 - 1;   // clear the lowest set bit
    Some(sq)
}
```

This is the bitboard equivalent of a `for` loop over pieces. To visit every
white knight, you loop `pop_lsb` on the white-knight bitboard; each iteration
hands you one square, in ascending index order, and the loop runs exactly as
many times as there are knights. ARCHON wraps this in an iterator so you can
write the idiomatic `for sq in bitboard { ... }` — the type implements
`IntoIterator` (with an explicit `BitboardIter` struct) rather than `Iterator`
directly, because a set is not itself a sequence; it *yields* one. That
distinction (sets provide `IntoIterator`, not `Iterator`) is a small Rust-idiom
choice ARCHON makes deliberately.

One more family of one-liners: **directional shifts.** Because of the LERF
mapping, `bitboard << 8` moves every square up one rank and `>> 8` moves down
one rank. East/west shifts (`<< 1`, `>> 1`) are almost as simple, but they need
a mask to stop a piece on the h-file from "wrapping" around to the a-file of the
next rank — an artifact of squares being laid out in one long 64-bit line.
`Bitboard::shift(direction)` handles the eight compass directions with exactly
one AND (to kill wrap) plus one shift each. From these, bulk pawn moves fall out
for free: every white pawn's single push is `pawns << 8`, and every white pawn's
two diagonal attacks are two shifts OR'd together (`pawn_attacks` in
`src/bitboard.rs`). No per-pawn loop at all — the whole pawn army moves in a
couple of instructions.

## The mailbox: the other half of the hybrid

Bitboards are wonderful for "which squares have property X?" But they are
clumsy for the opposite question: "**what piece, if any, is on this one
square?**" With bitboards alone you would have to test up to twelve boards to
find out — is d5 a white pawn? a black pawn? a white knight? … That query comes
up constantly (every capture needs to know what it captured, every move needs to
know what is moving), and testing twelve boards for it is silly.

So ARCHON keeps a **second, redundant representation** alongside the bitboards: a
plain array of 64 slots, each holding `Some(piece)` or `None`. This is called a
**mailbox** (each square is a "mailbox" you can peek into). Looking up "what is
on d5?" is now a single array index — O(1), one memory load:

```rust
mailbox: [Option<Piece>; Square::COUNT],   // Square::COUNT == 64
```

(From the `Board` struct in `src/board.rs`.) Keeping two representations of the
same fact is a classic space-for-time trade, and it means both must be updated
in lockstep whenever a piece moves. ARCHON enforces this by funneling *all*
placement changes through two private methods, `put_piece` and `remove_piece`
(plus an internal `move_piece` fast path), which update every layer — the
relevant piece bitboard, the color-occupancy bitboard, the all-occupancy
bitboard, and the mailbox — together. Nothing else is allowed to touch the raw
fields. This is the "expose behavior, not representation" rule again, now
protecting a cross-cutting invariant: the bitboards and the mailbox can never
disagree, because there is exactly one code path that mutates them.

This "hybrid" design — bitboards as the primary representation, a mailbox as a
redundant O(1) index — is the pinned decision D4 in `design/02`, and it is
standard in strong bitboard engines. You get the fast set algebra of bitboards
*and* the fast point lookup of a mailbox, at the cost of a little extra memory
and the discipline of updating both.

## The full board layout

Putting it together, here is the heart of ARCHON's `Board` (from `src/board.rs`,
lightly trimmed):

```rust
pub struct Board {
    // Primary representation: bitboards.
    pieces: [[Bitboard; 6]; 2],       // pieces[color][piece_type]
    color_occupancy: [Bitboard; 2],   // all pieces of each color
    all_occupancy: Bitboard,          // every occupied square

    // Redundant mailbox for O(1) piece lookup by square.
    mailbox: [Option<Piece>; 64],

    side_to_move: Color,
    // ... castling geometry, and the state stack (file 04) ...
}
```

Walk the fields:

- `pieces[color][piece_type]` is the core: twelve bitboards, one per
  (color, type). `pieces[White][Pawn]` is the set of white-pawn squares.
- `color_occupancy[color]` is the union of a color's six piece bitboards —
  redundant, but computing it once and keeping it saves six ORs on every query
  that needs "all white pieces."
- `all_occupancy` is the union of both colors — needed constantly, because
  sliding-piece attacks depend on *which squares are occupied* (file 03).
- `mailbox` is the point-lookup index described above.
- `side_to_move` records whose turn it is — a single bit of information that
  flips every ply.

The redundant boards (`color_occupancy`, `all_occupancy`) are yet another
space-for-time trade, and they are maintained by the same `put_piece` /
`remove_piece` chokepoint, so they too can never drift out of sync.

## Encoding pieces and colors

The last piece of vocabulary is how a "piece" is named. From `src/types.rs`:

- `Color` is an enum, `White` or `Black`, with `White as usize == 0` and
  `Black == 1` so a color can index a two-element array directly.
- `PieceType` is an enum of the six types in a fixed order: `Pawn, Knight,
  Bishop, Rook, Queen, King`, with discriminants 0–5, again so a type indexes a
  six-element array.
- `Piece` bundles a `Color` and a `PieceType`.

The order of `PieceType` is not arbitrary. It runs from least to most valuable
(pawn up to queen, then king), which lets several algorithms — most importantly
the "capture with your least valuable attacker" logic of move ordering (file 09)
and static exchange evaluation — simply iterate in enum order. `Piece` also
offers a flat `index()` in 0–11 (`color * 6 + type`), used for tables that want a
single dimension per piece — history tables, and later the NNUE feature indices.

Colors carry a little behavior too. `Color::flip()` swaps White and Black (used
every time the side to move changes), and `Color::pawn_push()` returns the
forward direction for that color's pawns (North for White, South for Black) — so
the bulk pawn-move code in `src/bitboard.rs` is written once, color-agnostic,
and asks the color which way "forward" is.

A note on why ARCHON stores `[[Bitboard; 6]; 2]` rather than, say, Stockfish's
packed 16-value piece encoding: with only twelve real pieces, a dense `[T; 12]`
or `[[T; 6]; 2]` table is simpler and just as cache-friendly, and it keeps the
indexing obvious. `design/02` records this as a small deliberate divergence from
the reference engine — an example of the project's stance that idiomatic clarity
wins unless a benchmark says otherwise.

## Why all this matters for what comes next

Everything downstream is expressed in this vocabulary. File 03's move generation
is a series of bitboard AND/OR/shift operations against precomputed attack sets.
File 04's Zobrist hashing XORs a random key per (piece, square) — indexed
exactly by the encodings above. The evaluation in file 08 loops over
`board.pieces(color, type)` and pops squares. When file 09 talks about a
"history table indexed by piece and to-square," the `Piece::index()` and
`Square::index()` you just met are the indices. The board representation is not a
detail you set up and forget; it is the substrate whose shape dictates how every
later algorithm is written.

## What you now know

You know that ARCHON numbers squares 0–63 in little-endian rank-file order so
that file and rank are a mask and a shift, and so that bit arithmetic mirrors
board geometry. You understand a bitboard as a `u64` interpreted as a set of
squares, where union/intersection/difference/complement are single bitwise
instructions, and you have met the three workhorse bit tricks — popcount, LSB,
and pop-LSB (the bitboard's `for` loop). You know why ARCHON *also* keeps a
64-slot mailbox: to answer "what is on this square?" in O(1), a question
bitboards answer badly. You have seen the full `Board` layout — twelve piece
bitboards, two occupancy aggregates, the mailbox, side to move — and the
`put_piece`/`remove_piece` chokepoint that keeps the redundant layers eternally
in sync. And you know how pieces and colors are encoded as small array-indexable
enums.

**Next:** `03-move-generation.md` — with the board as bitboards, we can finally
answer the question move generation is built on: which squares does a piece
attack? For knights and kings that is a lookup; for bishops, rooks, and queens,
whose reach depends on what is in the way, it requires one of the most
beautiful hacks in the field — magic bitboards.
