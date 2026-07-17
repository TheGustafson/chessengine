# 04 — State, Zobrist hashing, and repetition

Search walks the game tree by *making* a move to descend into a child position
and *unmaking* it to come back up, millions of times a second. Making a move is
more than sliding a piece: castling rights may be lost, the en-passant square may
appear or vanish, the fifty-move clock ticks or resets, and — crucially — a
64-bit fingerprint of the whole position must be updated so the engine can
recognize repeated positions and cache search results. This file explains how
ARCHON tracks all of that incrementally, and the randomized hashing scheme
(**Zobrist hashing**) that makes an incremental position fingerprint possible.

## Make/unmake vs copy-make

There are two ways to get from a position to a child and back.

**Copy-make:** keep the current board, copy it wholesale, apply the move to the
copy, and recurse on the copy. To go back up, just drop the copy — the original
was never touched. Simple and bug-resistant, but it copies the entire board
(bitboards, mailbox, all the derived state) at every node. At the branching
factor and depth an engine reaches, that copying dominates.

**Make/unmake:** keep *one* board and mutate it in place. `make_move` applies the
move; `unmake_move` reverses it exactly, restoring the previous position. No
copy per node — but every field the move touched must be perfectly restored, or
the corruption propagates up the tree.

ARCHON uses **make/unmake** (decision D8) — the standard choice in strong engines
— but with a twist that captures most of copy-make's safety. The trick is to
separate "the pieces" from "the derived bookkeeping." The piece placement is
mutated and un-mutated in place (cheap: a move touches at most a few squares).
The bookkeeping — castling rights, clocks, the position hash, precomputed check
and pin information — lives in a **stack of state records**, one per ply. Making a
move *pushes* a new record; unmaking *pops* it. Popping is O(1) and instantly
restores every derived field, because the parent's record is sitting intact
underneath. So `unmake_move` is, in ARCHON's words, "dumb-cheap": flip the side to
move, reverse the piece placement, pop the state. All the derived state comes
back for free.

## The state stack

From `src/board.rs`, the board owns:

```rust
states: Vec<StateInfo>,   // never empty; one entry per ply
```

`make_move` pushes a `StateInfo`, `unmake_move` pops one, and the vector is
pre-sized past the maximum search depth so mid-search pushes never trigger a
reallocation (a latency spike you do not want inside the hot loop). The top of
the stack is the current position's derived state; the public accessors
(`board.rule50_count()`, `board.checkers()`, and so on) all delegate to it.

What is in a `StateInfo`? Two groups (`src/board.rs`):

- A **`CopiedState`** block — the fields that are *carried over* from the parent
  and then lightly edited: the several Zobrist keys (below), castling rights, the
  fifty-move clock (`rule50`), the en-passant square, per-color non-pawn material.
  These are `Copy`; `make_move`'s first act is literally `new.copied =
  parent.copied`, ARCHON's safe spelling of Stockfish's "memcpy the struct
  prefix." Most of these fields change little or not at all per move, so copying
  and then editing is cheaper than recomputing.
- The **recomputed** fields — the position hash `key`, and the derived
  check/pin data: `checkers` (enemy pieces checking the side to move),
  `blockers_for_king` and `pinners` (the pin structure from file 03),
  `check_squares` (from which squares each piece type would give check), and a
  Wave-G "threat map" used by later move ordering. These are rebuilt each ply
  because they depend on the exact new placement.

This split — carry-and-edit the cheap stuff, recompute the position-dependent
stuff — is the core of the design. The reason the derived check/pin data lives
*here*, precomputed once per position rather than recomputed inside move
generation, is that file 03's legality filter and the search's pruning both
consult it many times per node; computing it once at make-time and stashing it in
the state record amortizes the cost.

## Zobrist hashing: a fingerprint you can update incrementally

Now the centerpiece. The engine constantly needs to answer: "have I seen this
exact position before?" — to detect draws by repetition, and to look a position
up in the transposition table (file 07). A position is a lot of state (piece
placement, side to move, castling rights, en-passant square), so comparing full
positions is slow and awkward as a hash-table key. We want a single 64-bit number
that identifies a position with negligible collision risk, and — this is the hard
requirement — that we can **update incrementally** as pieces move, without
rescanning the board.

**Zobrist hashing** delivers exactly this. The idea, due to Albert Zobrist:

1. Before the program runs, generate a table of random 64-bit numbers, one for
   every (piece, square) combination: a random key for "white pawn on e2," a
   different random key for "black knight on f6," and so on — 12 × 64 keys. Plus a
   key for "it is Black's turn," keys for each castling-rights configuration, and
   a key per en-passant file.
2. The hash of a position is the **XOR** of the keys for every fact true in that
   position: XOR together the (piece, square) key for each piece on the board, the
   side-to-move key if it is Black's move, the castling key, and the en-passant
   key if applicable.

Why XOR? Because XOR is its own inverse and is commutative and associative. Those
three properties are exactly what make the hash *incrementally updatable*.
Suppose the hash is `H` and a white pawn moves from e2 to e4. The two facts that
changed are "white pawn on e2" (now false) and "white pawn on e4" (now true). To
update the hash you XOR in *both* keys:

```
H' = H  XOR  key[white pawn][e2]  XOR  key[white pawn][e4]  XOR  key[side to move]
```

XOR-ing `key[white pawn][e2]` a second time *cancels* it (`x XOR x = 0`),
removing the pawn from e2; XOR-ing `key[white pawn][e4]` *adds* it. The
side-to-move key flips whose turn it is. A capture XORs out the captured piece
too; castling XORs the rook's two squares as well as the king's; a promotion XORs
out the pawn and XORs in the new piece. Each move is a handful of XORs against the
random table — no board scan. That is the magic that makes a whole-position
fingerprint affordable to maintain at every one of millions of nodes.

Because the whole thing is just XORs of fixed random constants, ARCHON builds the
key tables at **compile time** (`src/zobrist.rs`, a `const fn` fed by a small
seeded pseudo-random generator) and stores them as a single static `ZOBRIST`
table. Nothing about the keys is computed at runtime; only the running XOR is.

### Why does it work? The collision math

The keys are random, so the hash of a position is effectively a random 64-bit
value, and two *different* positions collide only if their random values happen
to coincide. With 64-bit hashes there are `2^64 ≈ 1.8 × 10^19` possible values.
How worried should we be?

This is the birthday problem. If you hash `n` distinct positions with a
`b`-bit hash, the probability that *some* pair collides is approximately
`n² / 2^(b+1)`. Search a billion nodes (`n = 10^9`) with 64-bit keys:

```
    n² / 2^65  ≈  10^18 / 3.7×10^19  ≈  0.027
```

Even at a billion distinct positions the chance of *any* collision is a few
percent, and a single collision usually just means one transposition-table entry
is briefly wrong — which the search's own validation (file 07) catches or
tolerates. The 64-bit hash is not a guarantee of uniqueness; it is a probabilistic
one strong enough that collisions are a non-issue in practice. (The
transposition table also stores a second, independent check on the key, so an
actual collision has to fool two tests, not one.)

### From-scratch computation as the oracle

Incremental hashing is efficient but error-prone: forget one XOR — say, on a
subtle castling-rights change — and the running hash silently diverges from the
true hash of the position, poisoning repetition detection and the TT. So ARCHON
keeps a from-scratch computation (scan the board, XOR everything) used in exactly
two places: once at setup (`from_fen`), to seed the running key; and in the
debug/test invariant oracle (`assert_consistent`), which recomputes the key from
scratch after every move in debug builds and asserts it matches the incrementally
maintained one. The moment an incremental XOR is wrong, a debug perft run trips.
This is the same "cheap incremental path plus an exact oracle that verifies it"
pattern as perft in file 03.

## The five-key family

ARCHON does not maintain *one* Zobrist key but **five** (a Stockfish-derived
design, decision D9), all updated by the same XOR mechanism from the same table:

- **`key`** — the full position hash, described above. Used for repetition
  detection and as the transposition-table key.
- **`pawn_key`** — the XOR of only the pawn keys (kings go in the non-pawn key
  below, a small divergence from Stockfish's convention). Pawn structure changes
  slowly and drives a lot of evaluation, so a separate pawn hash lets a future
  pawn-evaluation cache be keyed on just the pawns.
- **`material_key`** — a hash of just the *counts* of each piece type, keying
  material-configuration tables and endgame knowledge.
- **`minor_key`** and the per-color **`non_pawn_key`s** — further specialized
  hashes (knights+bishops; each color's non-pawn pieces) that later evaluation
  and correction-history features key on.

Why five? Because different evaluation components care about different subsets of
the position, and each wants its own cache keyed on just its subset — so that, for
example, two positions with identical pawns but different piece placement hit the
same pawn-cache entry. Maintaining them all incrementally costs a few extra XORs
per move and is near-impossible to retrofit later (every piece of code that
touches placement would need updating), so ARCHON wires all five in from day one
even though the later features that consume them arrive much later. The material
key uses a neat trick to avoid a separate table: it reuses unused square-slots of
the piece-square keys as "count slots" (`zobrist::Tables::material`), since the
two hash spaces never mix.

There are a couple of subtleties the five-key design bakes in that are worth
noting because they show the level of care hashing demands:

- **Promotion ranks are zeroed for pawns.** A pawn can never legally stand on the
  first or eighth rank, so those pawn keys are set to zero. That means a promotion
  needs no "clear the pawn at the destination" XOR — there was never a pawn-key
  there. A small optimization that also makes the promotion update simpler.
- **Castling is keyed by the whole 4-bit rights mask,** not per individual right.
  Any change to castling rights is `castling[old_mask] XOR castling[new_mask]` —
  two XORs, no per-right bookkeeping.
- **The en-passant key is XOR'd in only when an en-passant capture is actually
  legal,** not merely when a pawn made a two-square move. Two positions that
  differ only in a phantom en-passant square that no pawn can actually exploit
  must hash *identically*, or repetition detection breaks. Getting this exactly
  right (validating that a real en-passant capture exists) is a classic source of
  hard bugs, and ARCHON handles it in the FEN parser and in `make_move`'s
  double-push logic.

One more wrinkle: the key exposed to the transposition table is not the raw key
but a **rule50-folded** version. As the fifty-move clock climbs toward a draw,
ARCHON perturbs the exposed key in buckets (`fold_rule50` in `src/zobrist.rs`), so
that two positions with identical pieces but very different draw-clock values do
not share a TT entry — because their true values differ near the fifty-move
boundary. Repetition detection, by contrast, compares the *raw* keys. This is a
subtle mitigation for the "graph-history interaction" problem, where a position's
value depends on history (how close the draw clock is) that a pure position hash
would ignore.

## Repetition and the fifty-move rule

The chess rules include two ways a game draws that depend on *history*, not just
the current placement, and both lean on the Zobrist key.

**The fifty-move rule:** if 50 full moves (100 ply) pass with no pawn move and no
capture, either player may claim a draw. ARCHON tracks this with the `rule50`
counter in each `StateInfo`: it increments every ply and resets to zero on any
capture or pawn move. When it exceeds 99, the position is a draw — with one
guard: a checkmate delivered *on* the hundredth ply still wins, because checkmate
outranks the clock. So the draw claim stands only if the side to move is not in
checkmate (`is_draw` in `src/board.rs` checks for a legal move when in check).

**Threefold repetition:** if the same position occurs three times, it is a draw.
Detecting this is where the Zobrist key earns its keep. After each move, ARCHON
scans back through the state stack comparing the new `key` against earlier keys.
It does not scan the whole history — only back to the last *irreversible* move (a
capture or pawn move resets the counters), because a position before an
irreversible move can never recur. The scan window is
`min(rule50, plies_from_null)` and steps by two (a position can only repeat with
the same side to move). When it finds a match, it records the repetition in the
state as a signed distance: a positive value for the first repeat, negative if
that earlier occurrence was *itself* a repeat (meaning we are now at a threefold).

The search then treats repetition with a subtlety (`is_repetition` in
`src/board.rs`): a position that repeats an earlier one *inside the current
search tree* can be scored as a draw immediately (even on the second occurrence,
not waiting for a literal third), because if the search is willing to walk into a
repetition it is effectively conceding the draw. This "twofold inside the tree
counts" rule is a standard and important optimization — it lets the search prune
repetition draws early — encoded in that signed-distance representation. The
`plies_from_null` bound exists because of *null moves* (a search device covered
in file 10 where a side "passes"); a null move creates a boundary the repetition
scan must not cross, since positions on either side are not reachable from each
other by real play.

## Putting it together: one make_move

Here is the whole sequence, conceptually, when `make_move` runs:

```
1. Copy the parent's CopiedState block (rights, clocks, keys, material).
2. Increment rule50 (reset it later if this is a capture or pawn move).
3. Reverse-XOR the old en-passant / castling keys that are about to change.
4. Move the piece(s) on the bitboards + mailbox (the one placement chokepoint),
   XOR-ing each (piece, square) key into the running keys as it goes.
   - captures also remove the victim (extra XOR, material update, rule50 reset)
   - promotions swap pawn for piece; castling moves king and rook
5. Flip side to move; XOR the side key.
6. Set the new en-passant square (and XOR its key) only if a capture is possible.
7. Recompute check/pin/threat info for the new side to move.
8. Scan for repetition; record it in the new state.
9. Push the new StateInfo. (In debug builds, assert_consistent verifies
   every key from scratch.)
```

`unmake_move` is the easy direction: flip the side, reverse the placement, pop
the state. Everything derived — all five keys, the check info, the clocks — comes
back instantly because it was sitting in the parent's record all along. That
asymmetry (careful make, trivial unmake) is the payoff of the state-stack design.

## What you now know

You know why ARCHON uses make/unmake rather than copy-make, and how the state
stack makes unmaking O(1): the pieces are mutated in place while all derived
bookkeeping is pushed and popped as `StateInfo` records, split into a cheap
carried-over block and recomputed check/pin/hash fields. You understand Zobrist
hashing from first principles — a table of random per-fact keys, XOR'd together
to fingerprint a position, and incrementally updatable precisely because XOR is
its own inverse — and you have seen the birthday-problem math that makes 64-bit
collisions a non-issue. You know why ARCHON keeps five specialized keys, the
en-passant and castling and promotion-rank subtleties that hashing demands, and
that an exact from-scratch recompute stands guard over the incremental path. And
you understand how the fifty-move clock and threefold repetition — the two
history-dependent draws — are detected using those keys, including the
"twofold-inside-the-tree is a draw" search optimization.

**Next:** `05-minimax-and-alpha-beta.md` — with the board, moves, and state
machinery in place, we can finally build the search. We start from minimax,
reformulate it as negamax, and then meet alpha-beta pruning — the single idea
that, given good move ordering, lets an engine search twice as deep for the same
work.
