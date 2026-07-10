# What the Stockfish study means for our Rust design

Distilled action list from notes 01-04. Ordered by when we'll hit it.

## Attack tables (next up)

1. Leaper tables (`knight/king/pawn attacks`, between/line) as `const` tables
   via `const fn` — Rust can do at compile time what SF does at startup.
2. Sliders: **fancy magic bitboards** (packed shared table, ~841 KB), magics
   found at startup with Carry-Rippler subset enumeration + epoch
   verification, stored behind `OnceLock` (or pre-generated into source).
   Current SF does NOT use PEXT for slider indexing — plain magic
   multiply/shift is the reference implementation.
3. `#[repr(align(64))]` magic structs; build with `target-cpu=x86-64-v3`+.

## Make/unmake (right after)

4. `StateInfo` split into `CopiedState` (Copy: keys, castling, rule50,
   pliesFromNull, ep) + derived fields (key, checkers, blockers, pinners,
   checkSquares, captured, repetition). History = `Vec<StateInfo>` stack,
   not pointer chain.
5. Zobrist with SF's tricks: castling keyed by whole 4-bit mask; promotion
   ranks zeroed for pawn psq; material key via count slots. Maintain
   pawnKey/nonPawnKey/minorKey fields from the start (correction history
   uses them later).
6. `castling_rights_mask: [u8; 64]`, revocation =
   `rights &= !(mask[from] | mask[to])`.
7. Castling encoded as king-captures-own-rook internally (Chess960-proof);
   `gives_check` computed by caller and passed into make_move.
8. Repetition field per state; scan window `min(rule50, plies_from_null)`.

## Movegen

9. Pseudo-legal generation, monomorphized over GenType (const generics),
   single target-mask parameter, bulk pawn shifts. Lazy legality: only
   pinned-from / king / ep moves need the full check.
10. Keep `MoveList` as is; `ExtMove` (move+score, 8 bytes) only inside the
    future MovePicker.
11. Perft validates everything before any search work.

## Later (search phase)

12. TT: 10-byte entries, 3-per-32-byte cluster, `mul_hi64` indexing, relaxed
    atomics per field, generation aging, large pages via madvise.
13. Staged MovePicker state machine + partial insertion sort.
14. Histories: one generic gravity-update fn parameterized by limit D;
    butterfly + capture + continuation first, correction later.
15. Mate scores stored node-relative in TT (value_to_tt/from_tt).

## Performance philosophy checks

- Redundant representations are fine (mailbox + bitboards + counts) as long
  as mutation is centralized (our put/remove_piece).
- Branchless idioms: XOR flips (`^56`, `^7`, `^8`), mask tricks,
  `b & (b-1)`, `b & -b`.
- Prefetch (TT, histories) mid-make_move hides latency — add once TT exists.
- Accept benign races (relaxed atomics) in shared heuristic tables; never
  lock the hot path.
