# Move generation, MovePicker staging, history tables

Source: `movegen.{h,cpp}`, `movepick.{h,cpp}`, `history.h`.

## ExtMove and MoveList (movegen.h:39-69)

- `ExtMove` = Move (u16) + `int value` score, padded to exactly **8 bytes**
  (static_asserted). Order/compare by value only.
- `MoveList<GenType>` = fixed `Move[256]` on the stack + end pointer;
  generators take a raw cursor and return the advanced cursor
  (`*list++ = m`). No heap, no bounds checks in the hot loop. Scoring happens
  later in MovePicker's own `ExtMove[256]` buffer — the plain list stays
  2 bytes/move.

## Generation is monomorphized per GenType (movegen.cpp)

`GenType`: CAPTURES, QUIETS, EVASIONS, NON_EVASIONS, LEGAL. Explicit template
instantiation per type + `if constexpr` inside compiles each variant down to
a small branch-free routine (the CAPTURES build contains zero quiet-push
code). **Rust equivalent: const-generic parameter**
(`fn generate<const GT: GenType>` via a u8 const param or one monomorphic fn
per type) — same effect.

### The target-mask trick

One `target: Bitboard` parameterizes everything; attack sets are ANDed with:
- CAPTURES → enemy pieces
- QUIETS → empty squares
- NON_EVASIONS → not own pieces
- EVASIONS → `between_bb(ksq, checker)` (which includes the checker square) —
  block or capture

### Pseudo-legal + lazy legality (movegen.cpp:270-288)

Base generators are pseudo-legal. `generate<LEGAL>` only calls the expensive
`pos.legal(m)` for moves that CAN be illegal: from-square is a pinned piece
(`blockers_for_king & own`), king moves, or en passant. Everything else is
legal by construction. Removal is swap-with-last (order doesn't matter — it
gets sorted later anyway). The search itself never uses LEGAL in the hot
path: MovePicker emits pseudo-legal and search checks legality only for the
move it actually plays.

Evasions: double check (`more_than_one(checkers)`) => only king moves
generated. The evasion generator does NOT test attacked-square safety for
king moves — the legality filter does.

### Bulk pawn shifts (movegen.cpp:105-184)

Never loop per pawn:
- single pushes: `shift<Up>(pawns) & empty`
- double pushes: `shift<Up>(singles & rank3) & empty` (both-empty for free)
- captures: `shift<UpRight>(pawns) & enemies`, same for UpLeft
- from-square recovered as `to - offset` since the whole board shifted
  uniformly.
- Promotions handled separately from a `pawnsOn7` split; queen promotions are
  classed with CAPTURES, under-promotions with QUIETS.
- Under EVASIONS, `enemies = checkers` so pawn captures only target the
  checking piece.

(There's an AVX-512 `splat_pawn_moves` that emits up to 8 moves per vector
store — ignore for now, scalar pop_lsb loop is the baseline.)

## MovePicker: staged generation (movepick.{h,cpp})

One `ExtMove moves[256]` buffer + cursor pointers carve it into regions.
Emits one pseudo-legal move per `next_move()` call. Stages:

**Main search**: TT move (no generation at all!) → generate+score CAPTURES,
full sort → GOOD_CAPTURE (SEE-passing; failures swapped to a bad-captures
region at the buffer front) → generate+score QUIETS, partial sort with limit
`-3560 * depth` → GOOD_QUIET (score > -14000) → BAD_CAPTURE (replay deferred
region) → BAD_QUIET.

**Evasions**: TT → generate all evasions, score, full sort → emit.
**Qsearch**: TT → captures only. **ProbCut**: TT → captures with
`see_ge(threshold)`.

Details worth keeping:
- Constructor does `stage += !(ttm && pos.pseudo_legal(ttm))` — branchless
  skip of the TT stage when there's no usable TT move.
- **Partial insertion sort** with a limit: only elements with
  `value >= limit` are sorted to the front; the tail stays unsorted because a
  beta cutoff usually happens in the first few moves. Sorting work scales
  with what's actually consumed.
- `skip_quiet_moves()`: late-move pruning can abandon quiet generation
  entirely.
- Why staged wins: most nodes cut off on the TT move or first good capture —
  generating and scoring 30+ quiets against ~6 history tables each would be
  pure waste.

### Scoring (movepick.cpp:194-261)

- Captures: `captureHistory[pc][to][victim_type] + 7 * PieceValue[victim]`
  (MVV + history; NO attacker term — SEE handles that).
- Good/bad capture split is dynamic: `see_ge(m, -value/18)` — a capture with
  great history tolerates slightly losing SEE.
- Quiets: `2*mainHistory + 2*pawnHistory + continuationHistory[0,1,2,3,5]`
  + 16384 check bonus (if SEE >= -75) + threat-evasion term (bonus for moving
  off a square attacked by a cheaper piece, penalty for moving onto one)
  + low-ply history near root.
- Evasions: captures get `PieceValue[victim] + (1 << 28)` (huge constant so
  all captures outrank all quiet evasions).

## History tables (history.h)

**One update rule for everything** ("gravity"):
```
entry += clamp(bonus, -D, D) - entry * abs(clamp(bonus, -D, D)) / D
```
Self-saturating in [-D, D], exponentially decays old signal. Implement once
in Rust, generic over D: `fn update(entry: &mut i16, bonus: i32, const D: i32)`.

| Table | Index | D | Size |
|---|---|---|---|
| ButterflyHistory (main) | [color][from-to 16-bit] | 7183 | 256 KB |
| LowPlyHistory | [ply<5][from-to] | 7183 | 640 KB |
| CapturePieceToHistory | [piece][to][victim_type] | 10692 | 16 KB |
| PieceToHistory | [piece][to] | 30000 | 2 KB |
| ContinuationHistory | [prev piece][prev to] → PieceToHistory | 30000 | 2 MB |
| PawnHistory | [pawnKey & mask][piece][to] | 8192 | large, heap |
| CorrectionHistories | various, keyed by pawn/minor/nonpawn keys | 1024 | — |

- ContinuationHistory is a table OF tables: outer index = previous move's
  (piece, to); the search stack stores **pointers to the inner tables** for
  plies 1,2,3,4,6 back, so scoring a quiet is 6 array reads.
- Killers are GONE in current SF — history-based ordering replaced them.
- Hash-indexed tables (pawn/correction) are power-of-two sized: `key & (n-1)`.
- Thread-shared tables use relaxed atomics (i16); per-thread ones are plain.

## Rust implications

- Movegen: `fn generate<const GT: u8>(pos: &Board, list: &mut MoveList)`
  with the target-mask pattern; our MoveList already matches ([Move; 256]).
- We'll need `ExtMove`-equivalent only inside the picker; keep MoveList thin.
- Legality model: store blockers/pinners in state (see note 02), generate
  pseudo-legal, filter lazily. Perft uses generate<LEGAL> semantics.
- MovePicker = explicit enum state machine over one `[ExtMove; 256]` buffer
  with index-based regions (indices, not pointers — borrow-checker friendly
  and just as fast).
- Implement partial_insertion_sort exactly; don't reach for sort_unstable.
