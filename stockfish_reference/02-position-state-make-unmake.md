# Position, StateInfo, make/unmake, Zobrist

Source: `position.h`, `position.cpp`.

## StateInfo: the undo record (position.h:45-67)

Split into two blocks; the boundary is the `key` field.

**Block A — copied on do_move** (one `memcpy` up to `offsetof(StateInfo, key)`):
`materialKey`, `pawnKey`, `minorPieceKey`, `nonPawnKey[2]`,
`nonPawnMaterial[2]`, `castlingRights`, `rule50`, `pliesFromNull`, `epSquare`.

**Block B — recomputed after the move**: `key`, `checkersBB`, `previous`,
`blockersForKing[2]`, `pinners[2]`, `checkSquares[8]`, `capturedPiece`,
`repetition`.

The partial-memcpy relies on declaration order. Rust equivalent: split into
two structs (`CopiedState` + `DerivedState`) so "copy the prefix" is
`copied: prev.copied` (a plain Copy) and the derived part is constructed
fresh — same performance, no offsetof fragility.

## State history ownership

- `StateInfo.previous` raw pointers form the chain; search passes a stack
  local into each `do_move`. Game history lives in a
  `std::deque<StateInfo>` (chosen because deque growth never invalidates
  element addresses).
- **Rust way**: ply-indexed `Vec<StateInfo>` owned by the search/game — push
  on make, pop on unmake, index back for repetition scans. No
  self-referential pointers needed.

## Zobrist (position.cpp:48-162)

Tables: `psq[16][64]`, `enpassant[8]` (by file), `castling[16]` (indexed by
the whole rights mask, so a rights change is two XORs, not per-right),
`side`, `noPawns`.

Tricks:
- `psq[W_PAWN][rank8]` and `psq[B_PAWN][rank1]` are **zeroed**: pawns can't
  sit there, so promotion just XORs the new piece at `to` without clearing a
  pawn key.
- `materialKey` reuses psq rows 8..15 as "count slots":
  XOR `psq[pc][8 + count]` per piece added/removed — no separate table.
- Separate incremental keys: `pawnKey`, `nonPawnKey[color]`, `minorPieceKey`,
  `materialKey` — these feed correction histories and eval caches. Worth
  wiring the *fields* early even if unused at first.
- Exposed `key()` is `adjust_key50(st->key)`: perturbed once rule50 >= 14 so
  positions nearing the 50-move draw hash differently.
- Cuckoo tables (`cuckoo[8192]`) for `upcoming_repetition` — detects that a
  reversible move could repeat, used for early draw scoring. (Advanced;
  defer, but the key algebra `psq[pc][s1]^psq[pc][s2]^side` is why
  move-delta keys are cheap.)

## Incremental check info (position.cpp:467-480, 612-637)

Recomputed at the end of every do_move (`set_check_info`):
- `blockersForKing[c]` + `pinners[~c]` via `slider_blockers`: find "snipers"
  (enemy sliders that would hit the king on empty board), then any single
  piece on `between_bb(ksq, sniper)` is a blocker; friendly blocker => sniper
  is a pinner. Blockers include enemy pieces (discovered-check candidates).
- `checkSquares[pt]`: squares from which each piece type checks the ENEMY
  king => `gives_check(m)` is O(1) for most moves.
- `checkersBB` set in do_move as
  `givesCheck ? attackers_to(their king) & us : 0` — trusts the caller's
  `givesCheck` to skip the attack query entirely on quiet moves.

Consumers: `legal()` (pins), `see_ge()` (pinned attackers may not count),
`gives_check`, evasion generation.

## do_move order of operations (position.cpp:807-1062)

1. `k = st->key ^ side`; memcpy state prefix; link `previous`; advance `st`.
2. `++gamePly, ++rule50, ++pliesFromNull`.
3. Castling first (encoded as **king captures own rook** — Chess960-proof;
   `do_castling` removes both pieces before placing either).
4. Captures: ep-adjust the capture square; update pawn/nonpawn/minor/material
   keys; `rule50 = 0`.
5. XOR moved piece; clear old ep key; castling rights via mask (below);
   new ep square only if an enemy pawn could legally capture (checks
   discovered-check legality using previous blockers).
6. Promotion: swap piece, fix all keys.
7. **Prefetch TT entry with the new key**, then store `st->key`, then
   prefetch history entries.
8. Mutate board (remove/move/put piece), set `capturedPiece`, `checkersBB`,
   flip side, `set_check_info()`, compute `repetition`.

`undo_move` is deliberately dumb-cheap: restore board, pop `st`. No key or
check-info recomputation ever on the way back up.

`do_null_move`: full-struct copy, toggle ep+side keys only.

## Castling rights mask trick (position.cpp:449-463, 919)

`castlingRightsMask[64]`: per-square bits revoked when that square is
touched. King and rook start squares get their bits; all others 0. Then one
branchless line handles every case:
`rights &= ~(mask[from] | mask[to])` (moving king, moving rook, capturing
rook). Also precomputed: `castlingPath[right]` (squares that must be empty)
and `castlingRookSquare[right]`.

## Repetition (position.cpp:1040-1054, 1518-1544)

- Scan window = `min(rule50, pliesFromNull)`; walk back 2 plies at a time
  comparing keys.
- `repetition` field: ply-distance to previous occurrence; negative if that
  occurrence was itself a repeat (=> 3-fold). `is_repetition(ply)` treats a
  repeat inside the search tree (`< ply`) as an immediate draw.
- 50-move draw has a stalemate guard (not a draw if checkmated on the spot).

## Performance summary

- Redundancy: mailbox `board[64]` + `byTypeBB[8]` + `byColorBB[2]` +
  `pieceCount[16]`; `byTypeBB[ALL_PIECES]` is total occupancy. (We have the
  same minus pieceCount — add counts when eval needs them.)
- `gives_check` passed INTO do_move by the caller (search already knows it).
- TT + history prefetches issued mid-do_move to hide latency.
- NNUE DirtyPiece / DirtyThreats incremental tracking exists throughout
  do_move — ignore until eval, but it's why put/remove are centralized.

## Rust implications

- `make_move(&mut self, m: Move, gives_check: bool)` + `unmake_move(&mut self, m: Move)`
  with a `Vec<StateInfo>` stack; StateInfo split into Copied/Derived structs.
- Implement `castling_rights_mask: [CastlingRights; 64]` on Board setup.
- Zobrist as `static` tables generated by a const-fn PRNG (or once at init);
  adopt the zeroed-promotion-rank and material-count-slot tricks.
- Keep `checkers`, `blockers_for_king`, `pinners`, `check_squares` in
  StateInfo from day one — legal movegen and SEE need them immediately.
