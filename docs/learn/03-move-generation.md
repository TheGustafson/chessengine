# 03 — Move generation

Search calls move generation more than any other subsystem: every node in the
tree needs its list of legal moves. So this code has to be both *fast* and
*flawlessly correct* — a single wrong move count silently poisons the entire
search. This file builds move generation up in layers: attack sets for the
simple pieces, the blocker problem for sliding pieces, the magic-bitboard trick
that solves it (this is the wall most readers hit — we go slowly), the
distinction between pseudo-legal and legal moves, staged generation, and
finally **perft**, the test that engine authors treat as sacred.

## The core question: what does a piece attack?

Move generation reduces to one primitive: given a piece on square `s` and the
current occupancy (which squares hold *any* piece), which squares does it
**attack**? "Attack" means "could capture on, or move to if empty." Once you can
compute the attack set as a bitboard, generating moves is bookkeeping: AND the
attack set with "not my own pieces" to get legal destinations, split into
captures and quiet moves, and emit them.

The pieces divide into two classes:

- **Leapers** — knight and king — jump to a fixed set of squares regardless of
  what is in the way. A knight on d4 attacks the same eight squares whether the
  board is empty or crowded (it can be blocked only in the sense that it cannot
  land on its own pieces, which is handled later by the "not my pieces" mask).
- **Sliders** — bishop, rook, queen — move along rays until something stops
  them. A rook on d4 attacks up the d-file until it hits the first piece; that
  piece and everything beyond it change the attack set. Sliders' reach *depends
  on occupancy*, and that dependence is the whole difficulty.

Pawns are a special case (they capture differently than they move, they promote,
they have the two-square first move and en passant), handled with the bulk-shift
tricks from file 02. We focus here on leapers and — the hard part — sliders.

## Leapers: just precompute everything

For a knight, there are only 64 possible squares, and each has a fixed attack
set. So compute all 64 attack bitboards once, store them in a 64-entry table,
and every future query is a single array lookup. Same for the king. ARCHON
builds these at **compile time** with `const fn` code (`src/attacks.rs`): a
`KNIGHT_ATTACKS: [Bitboard; 64]` table and a `KING_ATTACKS: [Bitboard; 64]`
table baked straight into the binary's read-only data. Querying a knight's
attacks is `KNIGHT_ATTACKS[square]` — no computation at runtime at all.

Building each entry is a small loop over the piece's step-offsets (a knight has
eight: ±17, ±15, ±10, ±6 in the 0–63 numbering) with an edge check so a step off
the board or across a file boundary is dropped rather than wrapping. That edge
check is the same "don't let it wrap around the h-file" concern from file 02,
handled here by verifying the destination stays within two files of the origin.
The result is validated against a naive reference generator (more on that
philosophy under perft). Leapers, then, are trivial: the answer is a table. The
sliders are where it gets interesting.

## The blocker problem

Consider a rook on a1 (square 0). On an empty board it attacks the whole first
rank and the whole a-file — 14 squares. Now put a pawn on a4. Suddenly the rook
attacks a2, a3, a4 (it can capture the pawn) but *not* a5, a6, a7, a8 — the pawn
blocks the ray. Put another pawn on d1 and the rank attack is cut at d1 too.

So a rook's attack set is a function of two things: its square, and the set of
**blockers** — occupied squares along its rays. The rook has up to 14 relevant
ray squares. Each could be occupied or empty. That is up to `2^14 = 16384`
possible blocker configurations per square. Even so, of those 14 squares, the
ones on the board's edge never matter as blockers: a piece on a8 (the far end of
the file) does not change *which* squares the rook attacks along that ray,
because there is nothing beyond a8 for it to block. Dropping edge squares leaves
at most 12 **relevant** blocker bits for a rook (fewer for rooks not in a corner)
and at most 9 for a bishop.

The naive approach: for a given square and a given occupancy, walk each ray
outward, square by square, stopping at the first blocker. `src/attacks.rs` has
exactly this as `sliding_attack` — and it is the module's **correctness oracle**,
not its hot path. Walking rays is correct but slow (a loop with a branch per
square, unpredictable to the CPU). Sliders are queried constantly in search and
evaluation; we need the attack set in *one lookup*, like the leapers. But we
cannot precompute a table indexed by "square × full 64-bit occupancy" — that
would need `2^64` entries per square. The relevant blockers are far fewer (≤ 12
bits), but they are *scattered* across the 64-bit occupancy word, not packed
into a neat 12-bit index. We need a way to gather those scattered bits into a
small dense index. That is what magic bitboards do.

## Magic bitboards, from the ground up

Here is the problem stated as pure bit manipulation. We have:

- a **mask** for square `s`: a bitboard with 1s on exactly the relevant blocker
  squares (the rays, minus edges). For a rook on a1 the mask is b1–g1 and a2–a7
  (12 bits).
- the current **occupancy**, a 64-bit board.

`occupancy & mask` isolates just the relevant blocker bits — but they are still
sitting at their original, scattered positions within the 64-bit word (bits 1–6
for the rank, bits 8, 16, 24, … for the file, in the a1 example). We want to turn
that sparse pattern into a small integer 0…2^12−1 that we can use to index a
table. If we had such an index, we could precompute, for each square, a table
mapping every blocker pattern to its attack bitboard, and the query would be:
mask, index, look up.

The trick — and it genuinely feels like a trick the first time — is a single
**multiply**. Multiplying the masked occupancy by a carefully chosen 64-bit
constant (the "magic number") shuffles and overlaps the scattered bits so that
the ones we care about land, collision-free, in the *top* bits of the product.
Then we shift the product right to bring those top bits down to the bottom,
yielding a dense index. In one line:

```
index = ((occupancy & mask) * magic) >> (64 - number_of_relevant_bits)
```

Why does multiplying help? A multiply by a constant is, bit by bit, a sum of
left-shifted copies of the operand (that is literally what long multiplication
is in binary). With the right constant, each of the scattered blocker bits gets
shifted to a distinct top-bit position and they add up without stepping on each
other — a **perfect hash** of the blocker pattern into a compact index. There is
no formula for the magic constant; you *search* for one that happens to be
collision-free for that square's mask. More precisely, we allow *benign*
collisions: two different blocker patterns are allowed to map to the same index
as long as they have the *same* attack set (which happens often — many blocker
configurations beyond the first blocker on each ray produce identical attacks).
That tolerance is exactly what lets the tables stay small.

ARCHON's `Magic` struct (in `src/attacks.rs`) packs the four things a lookup
needs:

```rust
struct Magic {
    mask: u64,     // relevant-blocker mask for this square
    magic: u64,    // the multiplier found by search
    offset: u32,   // where this square's sub-table starts in the shared array
    shift: u32,    // 64 - popcount(mask)
}
```

and the lookup itself is precisely the one-liner above:

```rust
fn index(&self, occupied: Bitboard) -> usize {
    self.offset as usize
        + ((((occupied.to_u64() & self.mask).wrapping_mul(self.magic)) >> self.shift) as usize)
}
```

`wrapping_mul` is Rust's "multiply and let it overflow, that is intended" —
magic multiplication deliberately overflows the 64-bit word; the discarded high
bits are part of the shuffle. The `offset` field is why this is called **"fancy"**
magic: rather than give every square a worst-case-sized (4096-entry) table,
ARCHON packs all 64 sub-tables back-to-back into one shared array, each sized to
its own `2^popcount(mask)`. Rook tables total 102,400 entries and bishop tables
5,248 — together about 841 KB — instead of the megabytes plain magics would
waste. Storing an `offset` into the shared array (rather than a raw pointer, as
some engines do) keeps the struct free of `unsafe` self-reference and a few bytes
smaller.

### Finding the magics

Where do the magic numbers come from? A search, run once at startup behind a
`LazyLock` (Rust's thread-safe lazy initializer, so the tables are built exactly
once on first use). For each square, ARCHON (`init_magics` in `src/attacks.rs`):

1. Computes the relevant-blocker mask (rays minus edges).
2. Enumerates **every** blocker subset of that mask, recording the true attack
   set for each (computed with the naive ray-walker). The enumeration uses the
   *Carry-Rippler* trick, `next = (subset - mask) & mask`, which cleverly steps
   through all `2^k` subsets of a mask in sequence — a well-known idiom worth
   recognizing when you see it.
3. Draws random candidate constants and tests each: does it map every enumerated
   blocker subset to a table slot that is either empty or already holds the same
   attack set (a benign collision)? The candidates are drawn *sparse* (the AND of
   three random words, so only ~8 bits set) because good magics empirically have
   few bits, and sampling sparse makes the search converge orders of magnitude
   faster.
4. On the first collision-free candidate, that square's sub-table is now fully
   and correctly filled — verified against the oracle *by construction* — and the
   magic is recorded.

This runs in about 21 ms for both sliders combined and is deterministic (fixed
random seeds), so every launch re-derives and thereby re-verifies the same
tables. A queen, by the way, needs no magic of its own: a queen attacks the union
of what a rook and a bishop would from its square, so `queen_attacks` is just
`rook_attacks | bishop_attacks` (two lookups OR'd).

The upshot: after ~21 ms of startup, "which squares does this rook attack given
this occupancy?" is a mask, a multiply, a shift, and an array load — a handful of
instructions, no branches, no ray walking. That is the speed move generation and
evaluation are built on.

## Pseudo-legal vs legal

Attack sets get us most of the way to moves, but not all the way, because of a
rule we have ignored: **you may not leave your own king in check**. A move that
is fine by the piece's movement rules can still be illegal because it exposes the
king.

Engines split this into two notions:

- A **pseudo-legal** move obeys the moving piece's rules (a bishop moves
  diagonally, a pawn captures diagonally, castling squares are clear) but *might*
  leave the king in check.
- A **legal** move is pseudo-legal *and* does not leave the mover's king
  attacked.

Only three kinds of pseudo-legal move can actually be illegal, which is what
makes the legality filter cheap:

1. **Moving a pinned piece off its pin line.** A **pin** is when one of your
   pieces stands between your king and an enemy slider; moving it (except along
   the pin line) exposes the king.
2. **A king move into an attacked square** (including castling through or into
   check).
3. **En passant**, which can uncover a horizontal pin in a rare way that no other
   move can (removing *two* pawns from one rank at once).

Every *other* pseudo-legal move is legal by construction. ARCHON exploits this:
its `generate_legal` (in `src/movegen.rs`) generates all pseudo-legal moves, then
only re-checks the suspects — moves of a pinned piece, king moves, and en
passant — with a full `board.legal(m)` test; everything else is passed through
untouched. To know which pieces are pinned, the board precomputes, whenever the
position changes, a `blockers_for_king` bitboard (the pieces standing between a
king and enemy sliders) and the `checkers` (enemy pieces currently giving check).
File 04 covers where that precomputed check/pin information lives.

When the king *is* in check, the rules narrow sharply: you must get out of check
this move. So ARCHON generates **evasions** instead — moves that block the check,
capture the checker, or move the king to safety — and if the king is in **double
check** (attacked by two pieces at once, detectable by the `more_than_one` test
from file 02), only king moves can possibly help, so it generates just those.

## Staged generation

Search rarely wants "all legal moves" as an undifferentiated list. It wants the
*promising* moves first, so pruning (files 05, 09, 10) can cut off after
examining only a few. So ARCHON generates moves in **stages**, parameterized by a
`GenType` (in `src/movegen.rs`):

- `CAPTURES` — capturing moves (and queen promotions, which are capture-like in
  their value),
- `QUIETS` — non-capturing moves (and underpromotions),
- `EVASIONS` — the only legal replies when in check,
- `NON_EVASIONS` — the disjoint union of captures and quiets, used when not in
  check.

`generate::<GT>` is a Rust *const generic*: the stage is a compile-time
parameter, so the compiler produces a separate, specialized copy of the generator
for each stage with the stage-specific branches folded away — the flexibility of
a parameter with the speed of hand-written variants. The search hot path does not
even call `generate_legal`; it uses a `MovePicker` (file 09) that requests
captures first, checks them lazily for legality only when about to play them, and
often never generates the quiet moves at all because a cutoff arrived first. That
laziness — never generating what you can prove you will not need — is a recurring
performance principle.

## Perft: the correctness oracle

How do you know your move generator is correct? A subtle bug — a missing en
passant capture, a castling right not revoked, a pin mishandled — can lower the
node count by a fraction of a percent and be invisible in games while quietly
corrupting the search. You need an exact, mechanical test. That test is
**perft** (short for "performance test," a historical misnomer — it is a
correctness test).

Perft counts leaf nodes of the legal move tree to a fixed depth. `perft(n)` from
the starting position is: make every legal move, recurse to `perft(n-1)`, sum,
with `perft(0) = 1`. The counts are famous, exact, and independently verified
across the whole chess-programming community:

| Depth | Perft from the start position |
|-------|-------------------------------|
| 1     | 20                            |
| 2     | 400                           |
| 3     | 8,902                         |
| 4     | 197,281                       |
| 5     | 4,865,609                     |

If your engine reports 8,901 or 8,903 at depth 3, you have a bug — full stop. The
value of perft is that it is *unforgiving and precise*: the number is either
exactly right or it is not, and the community-known counts across a suite of
tricky positions (castling, promotions, en-passant pins, double checks) exercise
every corner of the rules. ARCHON keeps a 23-position suite in
`tests/data/perft.epd` and its `perft` function in `src/perft.rs`.

ARCHON actually runs perft in two modes. `perft` bulk-counts fast (at depth 1 it
just returns the legal move-list length, no make/unmake). `perft_checked` is
paranoid: after every single move it runs a full board-invariant oracle
(`assert_consistent`) that recomputes the position hash and derived state from
scratch and compares, verifies the check prediction against an independent
recompute, and after unmaking confirms the board is bit-for-bit identical to
before. That is the difference between "the count is right" and "every internal
data structure is provably consistent at every node."

Because move generation and make/unmake are so unforgiving, ARCHON pins a ritual
(decision **D24**, in `CLAUDE.md`): any change touching move generation, board
mutation, the Zobrist hashing, or the attack tables must pass perft — the fast
tier automatically in `cargo test`, and a full release-mode tier (over a billion
nodes) before it can land. There is also a divide tool (`divide` in
`src/perft.rs`, plus `tools/perft_diff.py`) that reports per-root-move subtree
counts, so when a count is wrong you bisect it: compare against a reference,
descend into the one root move whose subtree count differs, and repeat until you
find the exact position and move the bug hides in. Treating perft as sacred — as
a precondition for landing any relevant change, not an afterthought — is what
lets everyone downstream *trust* that the moves are right and spend their
debugging energy elsewhere.

## What you now know

You know that move generation reduces to one question — which squares does a
piece attack given the occupancy — and that leapers answer it with a precomputed
64-entry table while sliders cannot, because their reach depends on blockers. You
understand the blocker problem and its solution: mask the occupancy to the
relevant squares, multiply by a searched-for magic constant that perfectly (up to
benign collisions) hashes the scattered blocker bits into a compact index, shift
down, and look up — the magic-bitboard trick, packed "fancy" into one shared
841 KB table. You know the difference between pseudo-legal and legal moves, that
only pinned-piece moves, king moves, and en passant can be illegal (so the
legality filter is cheap), and that being in check switches generation to
evasions. You have seen staged generation via const generics and lazy
capture-first move picking, and you understand perft as the exact, sacred
correctness oracle that ARCHON pins behind the D24 ritual.

**Next:** `04-state-zobrist-and-repetition.md` — search makes and unmakes
millions of moves per second, and each one must update not just the pieces but a
whole ledger of derived state: castling rights, the fifty-move clock, and above
all a hash of the position used to detect repetitions and to key the
transposition table. We look at how that state is tracked incrementally and at
the elegant randomized hashing that makes it possible.
