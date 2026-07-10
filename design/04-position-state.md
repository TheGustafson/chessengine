# 04 — Board state, Zobrist keys, make/unmake, SEE

Status: Draft v1

This document specifies the evolution of `Board` from a static position
container into the search-ready core of the engine: a split `StateInfo` undo
record on a ply-indexed stack (D8), the full five-key Zobrist family with
every Stockfish hashing trick wired from day one (D9), `make_move` /
`unmake_move` / null-move mirroring Stockfish's `do_move` order of operations
(D10), incremental check information (`checkers`, `blockers_for_king`,
`pinners`, `check_squares`), repetition and draw detection, static exchange
evaluation (`see_ge`, D11), a debug-only consistency oracle, and the prefetch
hook seam that the TT and history layers will plug into later. Everything
here is expressed as original Rust sketches and prose; constants and formulas
are taken from current Stockfish as citable facts (D20, D27).

## 1. Purpose and scope

This document owns:

- The `StateInfo` / `CopiedState` structs and the `Vec<StateInfo>` history
  owned by `Board` (D8), including the migration of `castling_rights`,
  `en_passant_square`, and `halfmove_clock` off the `Board` struct into the
  state stack.
- The `zobrist` module: `Key`, the PRNG, all tables, and every keying rule
  (five keys, promotion-rank zeroing, count-slot material key, ep-only-when-
  legal, whole-mask castling key, rule50 folding) (D9).
- `make_move(m, gives_check)`, `unmake_move(m)`, `make_null_move`,
  `unmake_null_move` (D10), the castling helper, and the internal
  `move_piece` fast path.
- Check bookkeeping: `slider_blockers`, `set_check_info`, `gives_check(m)`,
  and the `attackers_to(sq, occupied)` query they share.
- Draw machinery: the `repetition` field, `is_draw(ply)`, `is_repetition`,
  `has_repeated`, and — phase-tagged as a later refinement — the cuckoo
  tables behind `upcoming_repetition(ply)`.
- `see_ge(m, threshold)` with pinned-piece handling and x-ray occupancy
  updates (D11).
- Board setup extensions: `castling_rights_mask[64]`, `castling_path`,
  `castling_rook_square`, en-passant validation at FEN load, `game_ply`.
- The debug-only `assert_consistent` extension (full key recompute).
- Prefetch hook points inside `make_move` (specified now, wired when the TT
  and history tables exist).

This document does NOT own:

- Attack tables — leaper tables, magics, `BETWEEN`, `LINE`, distance
  (design/03-attack-tables.md, D12). Everything here consumes them.
- Move generation, `legal(m)`, `pseudo_legal(m)`, perft
  (design/05-movegen-perft.md). Those predicates read the state this doc
  maintains (blockers, checkers) but live with movegen.
- The TT (probing, replacement, `rule50 >= 96` cutoff guard) and search draw
  handling policy (design docs for TT and search). We only provide the keys,
  `is_draw`, and the prefetch seam.
- Evaluation and NNUE dirty-piece tracking (D21); the hook seam defined here
  in §3.10 is where it will attach, but its content is out of scope.

## 2. References

- `stockfish_reference/02-position-state-make-unmake.md` — StateInfo split,
  do_move ordering, castling mask trick, repetition encoding.
- `stockfish_reference/stockfish_search_notes.md` §12 — Zobrist key family,
  rule50 folding, prefetch strategy.
- `stockfish_reference/00-design-implications.md` items 4–8.
- Verified against SF source (read-only, GPL — ideas only, D27):
  - `Stockfish/src/position.h` — StateInfo layout, `adjust_key50`, inline
    accessors.
  - `Stockfish/src/position.cpp` — `Position::init` (Zobrist + cuckoo),
    `set_state`, `set_check_info`, `update_slider_blockers`, `do_move`,
    `undo_move`, `do_castling`, `do_null_move`, `gives_check`, `see_ge`,
    `is_draw`, `has_repeated`, `upcoming_repetition`, `prefetch_key`.
  - `Stockfish/src/types.h` — piece values, `make_key`.
  - `Stockfish/src/misc.h` — xorshift64* PRNG.
- Our code: `src/board.rs`, `src/types.rs`, `src/moves.rs`, `src/fen.rs`.

## 3. Design

### 3.1 New core aliases and helpers

```rust
// types.rs additions
pub type Key = u64;                      // Zobrist hash

impl Piece {
    /// Flat 0..12 index: color * 6 + piece_type (D4). Used by Zobrist psq
    /// rows and, later, history tables.
    pub const fn index(self) -> usize {
        self.color as usize * 6 + self.piece_type as usize
    }
}

impl CastlingRights {
    /// The raw 4-bit mask, for indexing zobrist::CASTLING[0..16].
    pub const fn bits(self) -> u8 { self.0 }
}
```

SEE / material piece values (D11, SF-current `types.h` values, D20):

```rust
// board.rs (or a small `values` section in types.rs)
/// Indexed by PieceType as usize; King = 0 (never traded in SEE arithmetic).
pub const PIECE_VALUE: [Value; 6] = [208, 781, 825, 1276, 2538, 0];
//                                    P    N    B    R     Q    K
```

`Value` itself (`pub type Value = i32`, D5) is owned by the core-types design
doc; we only consume it.

### 3.2 StateInfo: the split undo record (D8)

Stockfish copies a struct prefix with `memcpy` up to `offsetof(StateInfo,
key)`. We express the same thing safely as two structs. The five Zobrist
keys in `CopiedState` are: `pawn_key`, `material_key`, `minor_key`, and
`non_pawn_key` for each color. The main `key` is derived (recomputed by
every `make_move` from the parent's key), so it lives with the derived
fields, exactly mirroring SF's block boundary.

```rust
/// Fields carried over from the previous state and then incrementally
/// edited by make_move. Plain Copy — this is our "memcpy the prefix".
#[derive(Clone, Copy)]
pub struct CopiedState {
    pub pawn_key: Key,             // pawns only, both colors
    pub material_key: Key,         // piece counts via count-slot trick
    pub minor_key: Key,            // knights + bishops, both colors
    pub non_pawn_key: [Key; 2],    // per-color non-pawns INCLUDING the king
    pub non_pawn_material: [Value; 2], // sum of PIECE_VALUE for N,B,R,Q
    pub castling_rights: CastlingRights,
    pub rule50: i32,               // halfmove clock
    pub plies_from_null: i32,
    pub ep_square: Option<Square>, // set ONLY when a legal ep capture exists
}

/// One ply of undo/search state. Copy so make_null_move can clone the
/// whole record cheaply (SF memcpy's the full struct there).
#[derive(Clone, Copy)]
pub struct StateInfo {
    pub copied: CopiedState,

    // Recomputed from scratch by every make_move:
    pub key: Key,                       // full position hash
    pub checkers: Bitboard,             // enemy pieces checking stm's king
    pub blockers_for_king: [Bitboard; 2], // per king-owner color; both colors' pieces
    pub pinners: [Bitboard; 2],         // pinners[c] pin AGAINST color !c's king
    pub check_squares: [Bitboard; 6],   // per PieceType; from-squares that check the enemy king
    pub captured: Option<Piece>,        // piece captured by the move into this state
    pub repetition: i32,                // 0 / +distance / -distance (see §3.8)
}
```

Size estimate: `CopiedState` ≈ 64 bytes, `StateInfo` ≈ 168–176 bytes with
alignment — about three cache lines, same ballpark as SF. Assert an upper
bound in a test (`size_of::<StateInfo>() <= 192`) so field creep gets
noticed.

### 3.3 Board evolution

```rust
pub struct Board {
    // Placement (unchanged):
    pieces: [[Bitboard; PieceType::COUNT]; Color::COUNT],
    color_occupancy: [Bitboard; Color::COUNT],
    all_occupancy: Bitboard,
    mailbox: [Option<Piece>; Square::COUNT],

    side_to_move: Color,
    game_ply: i32,                       // halfmoves since the game start position

    // Castling geometry, fixed at setup (Chess960-ready shape, D10):
    castling_rights_mask: [CastlingRights; Square::COUNT],
    castling_rook_square: [Square; 4],   // indexed by right bit position 0..4
    castling_path: [Bitboard; 4],        // squares that must be empty

    // Ply-indexed history; NEVER empty (root state pushed at setup) (D8).
    states: Vec<StateInfo>,
}
```

Changes from today's `board.rs`:

- `castling_rights`, `en_passant_square`, `halfmove_clock`,
  `fullmove_number` fields are removed; the public accessors keep their
  signatures (D1) and delegate:

```rust
impl Board {
    fn st(&self) -> &StateInfo { self.states.last().expect("state stack never empty") }

    pub fn castling_rights(&self) -> CastlingRights { self.st().copied.castling_rights }
    pub fn en_passant_square(&self) -> Option<Square> { self.st().copied.ep_square }
    pub fn halfmove_clock(&self) -> u32 { self.st().copied.rule50 as u32 }
    /// Canonical rule50 accessor for search/TT/pruning callers (matches the
    /// design/01-architecture.md §3.3.4 seam). `halfmove_clock()` above is
    /// the D1-preserved FEN-facing alias; both read the same state field.
    pub fn rule50_count(&self) -> u32 { self.st().copied.rule50 as u32 }
    pub fn fullmove_number(&self) -> u32 {
        (1 + (self.game_ply - (self.side_to_move == Color::Black) as i32) / 2) as u32
    }
    pub fn game_ply(&self) -> i32 { self.game_ply }

    // New state accessors:
    pub fn key(&self) -> Key { self.adjust_key50(self.st().key) }        // §3.5
    pub fn raw_key(&self) -> Key { self.st().key }
    pub fn pawn_key(&self) -> Key { self.st().copied.pawn_key }
    pub fn material_key(&self) -> Key { self.st().copied.material_key }
    pub fn minor_key(&self) -> Key { self.st().copied.minor_key }
    pub fn non_pawn_key(&self, c: Color) -> Key { self.st().copied.non_pawn_key[c as usize] }
    pub fn non_pawn_material(&self, c: Color) -> Value { self.st().copied.non_pawn_material[c as usize] }
    pub fn checkers(&self) -> Bitboard { self.st().checkers }
    pub fn blockers_for_king(&self, c: Color) -> Bitboard { self.st().blockers_for_king[c as usize] }
    pub fn pinners(&self, c: Color) -> Bitboard { self.st().pinners[c as usize] }
    pub fn check_squares(&self, pt: PieceType) -> Bitboard { self.st().check_squares[pt as usize] }
    pub fn captured_piece(&self) -> Option<Piece> { self.st().captured }
    pub fn king_square(&self, c: Color) -> Square {
        self.pieces(c, PieceType::King).lsb().expect("king must exist")
    }
}
```

- FEN parsing (`from_fen`) sets `game_ply = max(2 * (fullmove as i32 - 1), 0)
  + (side_to_move == Color::Black) as i32`. The `max(..., 0)` clamp mirrors
  SF `set` and absorbs the common incorrect FENs with fullmove counter 0
  (emitted by some tools): without it `game_ply` goes to -2/-1 and
  `fullmove_number()`'s `as u32` cast wraps to garbage in re-emitted FENs.
  `from_fen` then builds the castling geometry (§3.7), validates the ep square
  (§3.6), then calls `init_state()` — the from-scratch analogue of SF's
  `set_state` — and pushes the root `StateInfo` with `rule50` = the FEN
  halfmove clock and `plies_from_null = 0`. Since `plies_from_null` counts
  at most the plies actually recorded on the stack, the repetition scan
  window `min(rule50, plies_from_null)` can never index before the root
  state. UCI `position ... moves ...` replays moves through `make_move`
  onto this stack, so game-history repetitions are visible to the search
  for free.
- `states` is created with `Vec::with_capacity(MAX_PLY as usize + 64)` and
  `make_move` pushes/`unmake_move` pops. Reallocation is *correct* (no
  pointers into the Vec, unlike SF's `previous` chain) but we reserve to
  keep it off the hot path. `Board: Clone` clones the history, which is
  exactly what per-thread Worker copies need (D8, D23).
- `put_piece` / `remove_piece` stay the only placement mutators, plus a new
  internal fast path used by make/unmake:

```rust
impl Board {
    /// Move a piece between squares in one pass (from-bit ^ to-bit on three
    /// bitboards + two mailbox writes). Destination must be empty.
    pub(crate) fn move_piece(&mut self, from: Square, to: Square) { ... }
}
```

**Where hashing lives.** CLAUDE.md sketched "Zobrist hooks into
`put_piece`/`remove_piece`"; we deliberately do NOT do that. The five-key
family updates differently per piece type and per move kind, and the
ep/castling/side keys never route through placement at all — SF computes all
key deltas explicitly inside `do_move`, and so do we. `put_piece` /
`remove_piece` / `move_piece` stay placement-only; the from-scratch
`init_state()` and the debug oracle (§3.11) are the only full-board key
computations. (Called out in Open questions since it amends a CLAUDE.md
hint, not a decision.)

### 3.4 The zobrist module (D9)

All constants below are SF-current facts (D20).

```rust
// zobrist.rs
pub type Key = u64; // re-exported from types

pub struct Tables {
    /// psq[Piece::index()][square]. Count slots (§ material key) reuse
    /// square indices 8..=17 of each row — separate hash spaces never mix.
    pub psq: [[Key; 64]; 12],
    pub en_passant: [Key; 8],   // by FILE of the ep square
    pub castling: [Key; 16],    // by the WHOLE 4-bit rights mask
    pub side: Key,              // xored in when Black is to move
    pub no_pawns: Key,          // pawn_key baseline for a pawnless board
}

pub static ZOBRIST: Tables = build_tables(); // const-evaluated
```

PRNG: xorshift64\* exactly as SF uses (seed and constants are facts;
our draw order differs, so our keys won't equal SF's — nothing requires
that). Implementation-wise this is the shared `attacks::Prng` of doc 03
§3.6, upgraded to `const fn` (stable since const `&mut self`) so
`build_tables` const-evaluates — no local tuple-threading helper needed:

```rust
let mut rng = Prng::new(1070372); // SF's seed for the Zobrist table stream
// every key is one rng.rand64() draw
```

`build_tables` is a `const fn`: fill `psq` (color-major, piece-type, then
square a1..h8), then `en_passant[0..8]`, `castling[0..16]`, `side`,
`no_pawns` — then apply the tricks:

1. **Zeroed pawn promotion ranks**: `psq[white_pawn_row][56..64] = 0` and
   `psq[black_pawn_row][0..8] = 0`. Pawns can never stand there, so a
   promotion XORs only the new piece at `to` — no pawn-clear XOR needed,
   and the pawn_key from/to update degenerates to a from-only update.
2. **Castling by whole mask**: any rights change is two XORs
   (`castling[old_mask] ^ castling[new_mask]`), never per-right.
   `castling[0]` is a real random key (not zero) — harmless and simpler.
3. **Material key via count slots**: no separate table. Invariant:

   `material_key = XOR over pieces pc, i in 0..count(pc) of psq[pc][8 + i]`

   Adding a piece whose pre-add count is `c`: `^= psq[pc][8 + c]`.
   Removing one whose post-removal count is `c`: `^= psq[pc][8 + c]`.
   `c` reaches 9: promotions legally allow 10 knights/bishops/rooks (or 9
   queens) of one color, so slots 8..=17 are live (pawns stop at 15). The
   full 64-entry psq row covers this with margin — SF relies on the same.
   Do NOT assert `count <= 8` or size a dedicated 16-slot region: that
   would pass every perft position in the suite (none exceeds count 8) yet
   panic or corrupt the material key in legal promotion-heavy games. If a
   debug assert is wanted, bound the slot argument: `c <= 9` for non-pawns,
   `c <= 7` for pawns (board counts <= 10 and <= 8 respectively). Reusing
   real square keys is fine because the material key is never combined
   with the main key.

Rule50 folding (GHI mitigation) — the exposed `key()` perturbs the raw key
once the halfmove clock nears the 50-move rule, in 8-step buckets starting
at 14:

```rust
/// SF: make_key(seed) = seed * 6364136223846793005 + 1442695040888963407
const fn make_key(seed: u64) -> Key {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

impl Board {
    fn adjust_key50(&self, k: Key) -> Key {
        let r = self.st().copied.rule50;
        if r < 14 { k } else { k ^ make_key(((r - 14) / 8) as u64) }
    }
}
```

So rule50 0..=13 → raw key; 14..=21 → `k ^ make_key(0)`; 22..=29 →
`k ^ make_key(1)`; etc. Repetition detection compares RAW keys (`st.key`)
— folding applies only to the exposed `key()` that feeds the TT. A second
variant with threshold 13 (`adjust_key50_after`, modeling "parent rule50 +
1" for quiet non-pawn moves) exists EXCLUSIVELY for the speculative
`prefetch_key` prediction of §3.10; inside `make_move` the child clock is
already exact, so hook A folds with the standard threshold 14 (§3.8 step 9).

`pawn_key` is initialized to `no_pawns` (not 0) so that "no pawns at all"
hashes distinctly — pawn correction history depends on it later.

### 3.5 Check info: slider_blockers and set_check_info

`attackers_to` — the shared attack query (consumed here, by movegen, and by
SEE):

```rust
impl Board {
    /// All pieces of BOTH colors attacking `sq` under the given occupancy.
    pub fn attackers_to(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        // rook_attacks(sq, occupied)   & (rooks | queens)
        // | bishop_attacks(sq, occupied) & (bishops | queens)
        // | pawn_attacks(White, sq) & black pawns   (attack symmetry trick)
        // | pawn_attacks(Black, sq) & white pawns
        // | knight_attacks(sq) & knights
        // | king_attacks(sq)   & kings
    }
}
```

`slider_blockers` — computes, for the king of color `c`:
`blockers_for_king[c]` (pieces of EITHER color that are the sole piece
between an enemy sniper and the king; enemy ones are discovered-check
candidates) and `pinners[!c]` (snipers whose single blocker is friendly to
`c`, i.e. genuinely pinned):

Algorithm (SF `update_slider_blockers`):

1. `ksq = king_square(c)`.
2. `snipers = ((rook_attacks(ksq, EMPTY) & (rooks|queens)) |
   (bishop_attacks(ksq, EMPTY) & (bishops|queens))) & occupancy(!c)`
   — enemy sliders that would hit the king on an empty board.
3. `occ = all_occupancy ^ snipers` (snipers don't block each other).
4. For each sniper `s`: `b = BETWEEN[ksq][s] & occ`. If `b` has exactly one
   bit: `blockers |= b`; and if `b & occupancy(c)` is non-empty,
   `pinners_against_c |= s`.

`set_check_info(st: &mut StateInfo)` — called at the end of every
make/null-move and at setup, AFTER `side_to_move` has been flipped:

1. `slider_blockers` for both colors → `st.blockers_for_king`, `st.pinners`.
2. `ksq = king_square(!side_to_move)` (the king the new stm may check):
   - `check_squares[Pawn]   = pawn_attacks(!stm, ksq)` (a pawn of color
     `!stm` on `ksq` attacks exactly the squares from which a stm pawn
     checks `ksq`),
   - `check_squares[Knight] = knight_attacks(ksq)`,
   - `check_squares[Bishop] = bishop_attacks(ksq, all_occupancy)`,
   - `check_squares[Rook]   = rook_attacks(ksq, all_occupancy)`,
   - `check_squares[Queen]  = Bishop | Rook entries`,
   - `check_squares[King]   = EMPTY`.

### 3.6 gives_check(m) — O(1) for the common cases

Caller-supplied `gives_check` (D10) is produced by this method; search calls
it once per move before `make_move`.

```rust
impl Board {
    /// Does this pseudo-legal move of side_to_move give check?
    pub fn gives_check(&self, m: Move) -> bool { ... }
}
```

Logic (SF `gives_check`):

1. `pt = piece_on(from).piece_type`. **Direct check**: if
   `check_squares(pt).contains(to)` → true.
2. **Discovered check**: if `blockers_for_king(!stm).contains(from)` → true
   unless the piece moves along the king line
   (`LINE[from][to]` contains the enemy king square); castling is always
   true here (the rook lands with tempo while the king vacates a blocker
   square).
3. Otherwise by kind:
   - `Normal` → false.
   - `Promotion` → does the promoted piece attack the enemy king from `to`
     with occupancy `all ^ from` (the pawn has left)?
   - `EnPassant` → the one case steps 1–2 miss: discovered check *through
     the captured pawn*. `capsq = Square::from_coords(to.file(),
     from.rank())`; `occ = (all ^ from ^ capsq) | to`; check whether any
     stm rook/queen or bishop/queen attacks the enemy king under `occ`.
   - `Castling` → `check_squares(Rook).contains(rto)` where `rto` is the
     rook destination (F1/D1 relative to stm).

### 3.7 Castling geometry (rights mask, paths, rook squares)

At FEN setup, for each granted right, `set_castling_right(color, rook_from)`:

- `right` = kingside if `rook_from.file() > king_from.file()` else queenside
  (Chess960-proof comparison; standard chess today, D10).
- `castling_rights_mask[king_from] |= right` (both of that color's rights
  accumulate on the king square), `castling_rights_mask[rook_from] |= right`.
  All other squares stay `NONE`.
- `castling_rook_square[right.index()] = rook_from` where `right.index()` is
  the bit position 0..4 (WK=0, WQ=1, BK=2, BQ=3).
- With `kto`/`rto` the standard destinations (G1/F1 or C1/D1 relative):
  `castling_path[i] = (BETWEEN[rook_from][rto] | BETWEEN[king_from][kto]
  | rto_bb | kto_bb) & !(king_from_bb | rook_from_bb)` — the squares that
  must be empty. (SF folds the endpoints via inclusive `between_bb`; either
  formulation, the set is: every square the king or rook crosses or lands
  on, minus the two start squares.)

Then rights revocation in `make_move` is one branchless line covering king
moves, rook moves, and rook captures:

```rust
let touched = self.castling_rights_mask[from.index() as usize].bits()
            | self.castling_rights_mask[to.index() as usize].bits();
// rights &= !touched  (two XORs of zobrist::CASTLING around it — §3.8 step 7)
```

FEN-time ep validation (SF `set`): accept the parsed ep square only if all
hold, else store `None`:

a. a stm pawn attacks the ep square;
b. an enemy pawn stands on the square in front of it (the double-pushed
   pawn);
c. the ep square and the square behind it (the pawn's origin) are empty;
d. at least one candidate capturing pawn can play the capture without
   leaving its own king in check: simulate `occ = all ^ target ^ ep_bb ^
   candidate` (captured pawn off, capturing pawn moved to ep square) and
   require no enemy attacker (other than the removed target) hits our king.

This matters because the ep file is hashed: a spurious ep square would make
identical positions hash differently and break repetition detection and the
TT. Consequence: `to_fen` output drops unusable ep squares (SF-compatible
behavior); the fen round-trip tests get updated accordingly.

### 3.8 make_move — full order of operations (D10)

```rust
impl Board {
    /// `m` must be legal; `gives_check` must equal self.gives_check(m).
    pub fn make_move(&mut self, m: Move, gives_check: bool) {
        self.make_move_with(m, gives_check, &mut NoHooks);
    }
    pub fn unmake_move(&mut self, m: Move);
    pub fn make_null_move(&mut self);
    pub fn unmake_null_move(&mut self);
}
```

`make_move_with` (hook seam in §3.10) builds a local `StateInfo` and pushes
it at the end; `self.states.last()` therefore refers to the PREVIOUS state
throughout, which the ep-legality step needs. Steps, mirroring SF `do_move`:

1. `let prev = self.st();` (borrow ends before mutation); start the new key:
   `k = prev.key ^ ZOBRIST.side`. Create
   `let mut st = StateInfo { copied: prev.copied, ..zeroed derived };`
   — this assignment IS the "memcpy the prefix" (D8).
2. Counters: `self.game_ply += 1; st.copied.rule50 += 1;
   st.copied.plies_from_null += 1;`
3. Identify actors: `us = side_to_move`, `them = us.flip()`, `from/to =
   m.from()/m.to()`, `pc = piece_on(from)`,
   `captured = if m.kind() == EnPassant { Some(them pawn) } else
   { piece_on(to) }`. Note our castling encoding is king→G1/C1 (D2), NOT
   SF's king-captures-rook, so `piece_on(to)` is `None` for castling and no
   special-casing of `captured` is needed — but assert
   `captured.piece_type != King` in debug.
4. **Castling** (`m.kind() == Castling`): `kingside = to.file() >
   from.file()`; `rfrom = castling_rook_square[...]` (H1/A1 relative today),
   `rto` = F1/D1 relative. Board update Chess960-proof: **remove both**
   (king at `from`, rook at `rfrom`) **then place both** (king at `to`, rook
   at `rto`) — with standard squares this never overlaps, but the shape is
   960-ready (D10). Keys:
   `k ^= psq[rook][rfrom] ^ psq[rook][rto]` and
   `st.copied.non_pawn_key[us] ^=` the same pair.
5. **Capture** (`captured == Some(cap)`; mutually exclusive with 4):
   `capsq = to`, or for ep `to` shifted one rank toward `us`'s home
   (`to - 8` for White, `to + 8` for Black).
   - pawn captured: `st.copied.pawn_key ^= psq[cap][capsq]`;
   - else: `st.copied.non_pawn_material[them] -= PIECE_VALUE[cap.pt]`;
     `st.copied.non_pawn_key[them] ^= psq[cap][capsq]`; and if knight or
     bishop, `st.copied.minor_key ^= psq[cap][capsq]`.
   - `k ^= psq[cap][capsq]`;
     `st.copied.material_key ^= psq[cap][8 + count(cap) - 1]` where
     `count` is the PRE-mutation popcount (slot index = post-removal count;
     we defer ALL board mutation to step 10, unlike SF which removes the ep
     pawn early for piece-list reasons we don't have).
   - `st.copied.rule50 = 0`.
6. Mover key: `k ^= psq[pc][from] ^ psq[pc][to]`. (For castling this is the
   king pair; for promotions `psq[pawn][to]` is zero by construction, so
   this line needs no promotion special case in `k`.)
   Clear stale ep: if `st.copied.ep_square` is `Some(ep)`,
   `k ^= ZOBRIST.en_passant[ep.file()]; st.copied.ep_square = None;`
7. Castling rights: `k ^= castling[rights.bits()]`, apply the mask
   revocation (§3.7), `k ^= castling[new_rights.bits()]`.
8. **Pawn extras** (`pc.piece_type == Pawn`):
   - **Double push** (`from.index() ^ to.index() == 16`): candidate
     `ep = to` shifted one rank back toward `us`'s home. Set it ONLY if a
     legal ep capture will exist (D9), using the cheap exact test on the
     PREVIOUS state's blockers (SF do_move):
     - `pawns = pawn_attacks(us, ep) & pieces(them, Pawn)` — candidates.
       None → no ep.
     - `ksq = king_square(them)`; `not_blockers =
       !prev.blockers_for_king[them]`.
     - `no_discovery = not_blockers.contains(from) || from.file() ==
       ksq.file()` — if the double push itself delivers discovered check,
       ep can never be the legal reply.
     - Legal capture exists iff `no_discovery && (pawns & (not_blockers |
       LINE[ep][ksq]))` is non-empty — a candidate pawn either isn't a
       blocker of its own king, or captures along its pin line. (Using the
       *previous* blockers also catches the horizontal both-pawns-removed
       pin: before the double push only one pawn sat between king and rook,
       so the capturer was registered as a blocker.)
     - If set: `st.copied.ep_square = Some(ep); k ^=
       ZOBRIST.en_passant[ep.file()]`.
   - **Promotion**: `promo = m.promotion_piece()` piece of `us`.
     `k ^= psq[promo][to]` (no pawn-clear thanks to the zeroed rank);
     `st.copied.material_key ^= psq[promo][8 + count(promo)]
     ^ psq[pawn_us][8 + count(pawn_us) - 1]` (pre-mutation counts);
     `st.copied.non_pawn_key[us] ^= psq[promo][to]`; if minor,
     `st.copied.minor_key ^= psq[promo][to]`;
     `st.copied.non_pawn_material[us] += PIECE_VALUE[promo.pt]`.
   - Always: `st.copied.pawn_key ^= psq[pawn_us][from] ^ psq[pawn_us][to]`
     (the `to` term is zero on the promotion rank); `st.copied.rule50 = 0`.
   Non-pawn mover instead updates `non_pawn_key[us]` (and `minor_key` for
   N/B) with the from/to pair.
9. `st.key = k`. **Prefetch hook point A** — the TT cluster key for the
   child position: fold with the standard threshold-14 rule on the
   IN-FLIGHT state's clock, which is already exact here (incremented in
   step 2, zeroed by steps 5/8):
   `hooks.on_key(if st.copied.rule50 < 14 { k } else
   { k ^ make_key(((st.copied.rule50 - 14) / 8) as u64) })`.
   Do NOT call `adjust_key50_after` here: it reads the PARENT's clock via
   `self.st()` (the push happens at step 14) and models "parent rule50 +
   1", which is only valid for quiet non-pawn moves — after any capture or
   pawn move the child's clock is 0, so at parent clocks >= 13 it would
   fold a key whose true TT probe key is raw. It exists solely for the
   speculative `prefetch_key` of §3.10. (SF: `do_move` prefetches
   `tt->first_entry(adjust_key50(k))` on the already-updated child rule50;
   `adjust_key50<true>` appears only in `prefetch_key`.) **Hook point B**
   right after: history/correction-entry prefetches keyed by the
   just-finalized specialized keys.
10. **Board mutation** (skipped for castling, done in step 4): remove the
    captured piece at `capsq` if any; then `move_piece(from, to)`, except
    promotions do `remove_piece(from); put_piece(promo, to)`.
11. `st.captured = captured;`
    `st.checkers = if gives_check { self.attackers_to(king_square(them),
    all_occupancy) & occupancy(us) } else { Bitboard::EMPTY };` — trusting
    the caller keeps quiet moves free of any attack query (debug_assert it
    against a from-scratch check).
12. `self.side_to_move = them;` then `set_check_info(&mut st)` (§3.5).
13. **Repetition** (§3.9): `st.repetition = 0`; `end = min(rule50,
    plies_from_null)`; if `end >= 4`, with `n = self.states.len()` (the new
    state's future index), scan `i = 4, 6, 8, ... <= end` comparing
    `self.states[n - i].key == st.key`; on first hit set `st.repetition =
    if states[n - i].repetition != 0 { -(i as i32) } else { i as i32 }` and
    stop.
14. `self.states.push(st);` (debug builds: `self.assert_consistent()`).

`unmake_move(m)` — deliberately dumb-cheap; no key or check-info work:

1. `let st = self.states.pop()` (assert the root state remains);
   `self.side_to_move = self.side_to_move.flip()`; `us = side_to_move`;
   `self.game_ply -= 1`.
2. Castling: remove king at `to` and rook at `rto`, place king at `from` and
   rook at `rfrom` (remove-both-place-both again).
3. Else: promotions do `remove_piece(to); put_piece(pawn_us, from)`;
   otherwise `move_piece(to, from)`. Then if `st.captured == Some(cap)`,
   `put_piece(cap, capsq)` with `capsq` recomputed from the move kind.

Null moves (SF `do_null_move`/`undo_null_move`):

- `make_null_move`: requires `checkers().is_empty()`. Copy the ENTIRE top
  `StateInfo` (both blocks — this is why `StateInfo: Copy`). Then: clear ep
  (with its key XOR), `key ^= ZOBRIST.side`, `plies_from_null = 0`,
  `captured = None`, `repetition = 0`, flip `side_to_move`,
  `set_check_info`, push. `rule50` and `game_ply` are intentionally NOT
  changed (SF behavior; the `plies_from_null = 0` window reset is what
  isolates repetition scans across the null boundary).
- `unmake_null_move`: pop + flip side.

### 3.9 Draws and repetition

`repetition` encoding (SF): `0` = no earlier occurrence inside the reachable
window; `+d` = the same raw key occurred `d` plies ago (first repeat);
`-d` = it occurred `d` plies ago AND that occurrence was itself a repeat —
i.e. at least threefold. The scan window is `min(rule50, plies_from_null)`
because captures/pawn moves and null moves both make earlier positions
unreachable.

```rust
impl Board {
    /// Draw by 50-move rule or repetition, relative to search ply.
    pub fn is_draw(&self, ply: i32) -> bool {
        let st = self.st();
        // 50-move rule, guarded: a mating move on the 100th halfmove wins.
        if st.copied.rule50 > 99
            && (st.checkers.is_empty() || self.has_legal_moves()) {
            return true;
        }
        self.is_repetition(ply)
    }

    /// Draw score if repeated once strictly inside the search tree,
    /// or twice anywhere (negative encoding is always < ply).
    pub fn is_repetition(&self, ply: i32) -> bool {
        let r = self.st().repetition;
        r != 0 && r < ply
    }

    /// Any repetition since the last irreversible move? (search bookkeeping)
    pub fn has_repeated(&self) -> bool {
        // Walk from the top state backwards while end >= 4, end -= 1 each
        // step, checking each state's repetition field for nonzero.
        // end starts at min(rule50, plies_from_null).
    }
}
```

`has_legal_moves()` is provided by movegen (design/05-movegen-perft.md);
until it lands, `is_draw` ships gated on `checkers().is_empty()` only, with
a TODO tied to that doc.

**upcoming_repetition(ply) — specified now, phase-tagged LATER** (build
after search exists; it's an early-draw-scoring refinement, not a
correctness requirement). Marcel van Kervinck's cuckoo scheme, SF-exact:

- Tables: `cuckoo: [Key; 8192]`, `cuckoo_move: [Move; 8192]`. Hash slots:
  `H1(k) = k & 0x1FFF`, `H2(k) = (k >> 16) & 0x1FFF`.
- Fill: for each of the 12 pieces EXCEPT pawns, for each square pair
  `s1 < s2` with `s2` in the piece's empty-board attack set from `s1`:
  `move_key = psq[pc][s1] ^ psq[pc][s2] ^ side`, insert
  `(move_key, Move::normal(s1, s2))` by cuckoo displacement (swap into
  `H1`, evictee goes to its alternate slot, repeat). SF asserts exactly
  **3668** entries. Needs empty-board slider attacks → built after the
  attacks module initializes (a `LazyLock` alongside it, D12).
- Query: `end = min(rule50, plies_from_null)`; if `end < 3` → false. Walk
  `i = 3, 5, 7, ... <= end` maintaining `other`, the XOR of consecutive
  key-deltas: when `other == 0`, the last two plies form a returnable
  "move"; look up `move_key = current_key ^ states[n - i].key` in the
  cuckoo tables; if found and `BETWEEN[s1][s2]` (exclusive of the occupied
  endpoint) is empty of pieces, a reversible move can force the repeat —
  return true if `i < ply` (inside the tree), or if the earlier state
  already had `repetition != 0`.

### 3.10 Prefetch hook points (specified now, wired later)

D10 fixes `make_move(&mut self, m: Move, gives_check: bool)`, so prefetch
targets can't be parameters of the public method. Seam: a monomorphized hook
trait; `NoHooks` compiles to nothing. The same seam later carries NNUE
dirty-piece recording (D21) — one reason placement mutation stays
centralized.

```rust
pub trait MakeMoveHooks {
    /// Hook A: full child key available (rule50-adjusted); board mutation
    /// still in flight. TT will prefetch its cluster here.
    #[inline] fn on_key(&mut self, _key: Key) {}
    /// Hook B: specialized keys final. Histories/correction tables
    /// prefetch their entries here.
    #[inline] fn on_state(&mut self, _copied: &CopiedState) {}
}
pub struct NoHooks;
impl MakeMoveHooks for NoHooks {}

impl Board {
    pub fn make_move_with<H: MakeMoveHooks>(&mut self, m: Move,
                                            gives_check: bool, hooks: &mut H);
}
```

Additionally, `prefetch_key(m)` — SF's cheap child-key *prediction* used by
search to prefetch even before calling `make_move` (ignores
castling/ep/promotion details):

```rust
impl Board {
    /// Approximate child key: side ^ mover from/to ^ captured@to.
    /// Exact for normal quiet/capture moves.
    pub fn prefetch_key(&self, m: Move) -> Key {
        let pc = ...; let captured = self.piece_on(m.to());
        let mut k = self.raw_key() ^ ZOBRIST.side
                  ^ psq(pc, m.from()) ^ psq(pc, m.to())
                  ^ captured.map_or(0, |c| psq(c, m.to()));
        // rule50 resets on capture/pawn move → no fold; otherwise fold with
        // the after-move threshold 13 (models rule50 + 1 >= 14).
        if captured.is_some() || pc.piece_type == PieceType::Pawn { k }
        else { self.adjust_key50_after(k) }
    }
}
```

The actual `prefetch` intrinsic (`_mm_prefetch` on x86_64, no-op elsewhere)
lives in the TT design's scope; it is unsafe-budget item (a) of D14 and gets
a `// SAFETY: prefetch has no memory effects` comment.

### 3.11 SEE: see_ge (D11)

Board method (needs occupancy, pinners/blockers, attack queries):

```rust
impl Board {
    /// True iff the static exchange value of `m` is >= threshold.
    /// Null-window swap algorithm; values from PIECE_VALUE (D11).
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool;
}
```

Algorithm (SF `see_ge`, expressed originally):

1. Non-`Normal` kinds (promotion, ep, castling) short-circuit:
   `return 0 >= threshold`.
2. `swap = value(piece_on(to)) - threshold` (empty `to` counts 0). If
   `swap < 0` → false (even winning the target free doesn't reach the bar).
3. `swap = value(piece_on(from)) - swap`. If `swap <= 0` → true (opponent
   recapturing our mover still leaves us at/above the bar).
4. `occupied = all ^ from_bb ^ to_bb` — clear the mover's square AND toggle
   `to` (for a capture this removes the victim; for a quiet move it marks
   `to` occupied). The `to`-toggle is what keeps the pinner-liveness test in
   step 6 honest when the first move captures a pinner.
   `stm = side_to_move`; `attackers = attackers_to(to, occupied)`;
   `res = true` (meaning: the side that played `m` currently meets the
   threshold).
5. Loop: `stm = stm.flip(); attackers &= occupied;`
   `my_attackers = attackers & occupancy(stm)`; if empty → break (stm has
   no recapture; `res` stands).
6. **Pinned-piece handling**: while any of stm's opponents' pinners are
   still on the board (`pinners(!stm) & occupied` non-empty), stm's pinned
   pieces may not join: `my_attackers &= !blockers_for_king(stm)`; if that
   empties the set → break. (Approximation: once every pinner has been
   captured in the exchange, formerly pinned pieces participate freely.)
7. `res = !res;` then take stm's least valuable attacker, in order
   P → N → B → R → Q → K:
   - Pawn: `swap = 208 - swap`; if `swap < res as i32` → break. Remove that
     pawn from `occupied` (lsb of the pawn subset) and reveal diagonal
     x-rays: `attackers |= bishop_attacks(to, occupied) & (bishops|queens)`.
   - Knight: `swap = 781 - swap`; bar check; remove (no x-rays through a
     knight).
   - Bishop: `swap = 825 - swap`; bar check; remove; add diagonal x-rays.
   - Rook: `swap = 1276 - swap`; bar check; remove; add orthogonal x-rays
     (`rook_attacks(to, occupied) & (rooks|queens)`).
   - Queen: `swap = 2538 - swap`; remove; add BOTH x-ray sets. (No bar
     check needed: a queen recapture after anything but a king always
     clears it — SF asserts instead of testing.)
   - King: the king may only "recapture" if the opponent has no attackers
     left: `return if (attackers & !occupancy(stm)).is_empty() { res }
     else { !res }`.
   The bar check `swap < res as i32` is the null-window trick: `res` ∈
   {0, 1} acts as the margin the current capturer must clear; failing it
   ends the sequence with the already-flipped `res` as the answer.
8. On loop exit: `return res`.

Consumers: qsearch pruning, MovePicker capture staging, probcut, history
malus decisions — all in later docs. The pinner data it reads is exactly
what §3.5 maintains, which is why SEE lives on `Board`.

### 3.12 Consistency validation

Extend the existing `assert_consistent` (currently `#[cfg(test)]`,
bitboards-vs-mailbox only) to `#[cfg(any(test, debug_assertions))]` and add:

- Full from-scratch recompute of all five specialized keys + the raw main
  key (shared code with `init_state`) compared field-for-field against the
  incremental values — this is the drift detector for every XOR in §3.8.
- `checkers` recomputed via `attackers_to(king_square(stm), all) &
  occupancy(!stm)` and compared.
- `non_pawn_material` recomputed from popcounts × `PIECE_VALUE`.
- Structural sanity: exactly one king per color; no pawns on ranks 1/8;
  castling rights imply king and rook on their recorded squares; ep square
  (if set) on the correct relative rank with the double-pushed pawn in
  front; `states.len() >= 1`; `rule50 >= 0`; blockers/pinners recomputed
  and compared.

Debug `make_move`/`unmake_move` call it after mutation (step 14), so a
debug-mode perft run exercises the oracle at every node (D24). Release
builds compile it out entirely.

## 4. Implementation plan

Ordered, commit-sized; each lands only with `cargo test` + clippy clean.
Steps 1–3 need nothing from the attacks module; 4+ depend on
design/03-attack-tables.md.

1. **types: `Key`, `Piece::index`, `CastlingRights::bits`** (+ tests:
   index range 0..12, bits round-trip). Done: compiles, tests pass.
2. **zobrist module**: const-fn xorshift64* PRNG (seed 1070372), `Tables`,
   promotion-rank zeroing, `make_key`, rule50-fold helpers. Done: tests
   verify determinism across calls, zeroed rows exactly `psq[wp][56..64]` /
   `psq[bp][0..8]`, all other entries nonzero and pairwise-distinct
   (spot-check), 16 castling keys, fold bucket boundaries (13→raw, 14 and
   21 same bucket, 22 next).
3. **StateInfo split + Board migration**: `CopiedState`/`StateInfo`, `states:
   Vec`, `game_ply`, accessor delegation, `init_state()` computing all keys
   + material from scratch (checkers/blockers stubbed empty until step 4),
   FEN plumbing incl. `game_ply` mapping and castling geometry arrays.
   Done: existing fen/board tests pass (ep validation still permissive),
   `from_fen(START_FEN).key()` is stable across runs, size assertion on
   `StateInfo`.
4. **attackers_to + slider_blockers + set_check_info** (first consumer of
   attack tables). Wire into `init_state`; implement FEN ep validation
   (§3.7) now that attack queries exist. Done: fixture tests for pins,
   discovered-check candidates (enemy blockers), check_squares on known
   positions; spurious-ep FENs round-trip to `-`.
5. **make/unmake, normal + capture moves only** (`move_piece` helper, steps
   1–3, 5–7, 9–12, 14 of §3.8; unmake). Done: scripted sequences restore
   the board byte-identically; incremental keys equal `init_state`
   recompute after every step.
6. **make/unmake: ep, promotion, castling** (steps 4, 8) + `gives_check`.
   Done: fixture tests in §5 pass, incl. ep-legality corner cases.
7. **Null move + repetition field + is_draw/has_repeated** (is_draw's
   legal-move guard behind a movegen-availability TODO). Done: repetition
   encoding tests (+d/-d), window clipping by captures and null moves.
8. **Debug assert_consistent extension** wired into make/unmake. Done: all
   move-making tests pass in debug with the oracle live.
9. **see_ge**. Done: SEE fixture suite passes.
10. **Hook seam + prefetch_key** (`MakeMoveHooks`, `NoHooks`,
    `make_move_with`). Done: `make_move` delegates; bench-neutral by
    construction (monomorphized empty hooks); prefetch_key equals the real
    child key on normal moves in a property test.
11. **(gate, with design/05-movegen-perft.md)** perft suite green in debug
    (oracle on) and release (D24).
12. **(later phase, after search lands)** cuckoo tables +
    `upcoming_repetition` per §3.9, verified by the 3668-entry count and
    equivalence tests against brute-force "exists a legal move into
    is_draw".

## 5. Testing and validation

Unit tests (all `Board::empty()`+`put_piece` or `from_fen`, per CLAUDE.md):

- **Key invariance (the big one)**: after every make in a scripted (later:
  randomized, once movegen exists) walk, all six keys equal a from-scratch
  `init_state` recompute; after every unmake, `Board` compares equal to a
  pre-move clone (derive `PartialEq` including states' top or compare
  field-wise ignoring Vec capacity).
- **Transposition invariance**: two move orders reaching the same position
  produce identical raw keys (e.g. knight-shuffle rejoins); differing side
  to move / castling rights / ep availability produce different keys.
- **EP hashing cases**: double push with no adjacent enemy pawn → no ep key;
  with adjacent pawn → keyed; ep capture forbidden by vertical pin of the
  capturer → not keyed; the horizontal both-pawns-off-the-rank pin (e.g.
  white Kh5/Pf5, black Ra5, after ...e7e5) → not keyed; capture legal along
  a diagonal pin line → keyed; double push delivering discovered check →
  not keyed. FEN-load equivalents for §3.7 a–d.
- **FEN fullmove-0 clamp**: `from_fen` on a FEN suffixed `... 0 0` (both
  side-to-move variants) yields `game_ply == 0` (White) / `1` (Black),
  `fullmove_number() == 1` with no u32 wrap, and `to_fen` re-emits a legal
  counter of 1.
- **gives_check**: fixtures for direct, discovered (incl. moving along the
  king line → false), promotion check through the vacated from-square, ep
  double-discovery through the captured pawn, castling rook check; each
  cross-checked against `make_move` + recomputed checkers.
- **Castling**: rights revocation via mask for king move, each rook move,
  each rook capture (incl. capturing a never-moved rook on its home
  square); remove-both-place-both round trip; castling keys.
- **Rule50 folding**: same position with clocks 13 vs 14 vs 21 vs 22 —
  `key()` equal/unequal per bucket; raw keys always equal; repetition still
  detected across differing clocks.
- **Repetition**: shuffle sequence produces `+d` then `-d`; `is_draw(ply)`
  true for in-tree twofold, false for a root-side twofold at `ply` 0 unless
  threefold; window cut by a capture and by a null move; `has_repeated`.
- **Null move**: key = old ^ side ^ (ep key if any); unmake restores;
  `plies_from_null` resets.
- **SEE fixtures** (values are D11 numbers):
  - `1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1`, `e1e5` (RxP,
    undefended): `see_ge(m, 208)` true, `see_ge(m, 209)` false.
  - `1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1`, `d3e5`
    (NxP, defended): loses N for P → `see_ge(m, 208 - 781)` true,
    `see_ge(m, 208 - 781 + 1)` false.
  - Pinned-attacker case: a defender pinned to its king must not count
    until its pinner is captured (construct with §3.5 fixtures).
  - X-ray case: battery (rook behind rook) recapture counted via occupancy
    update.
  - Non-normal kinds: ep/promotion/castling return `0 >= threshold`.
- **assert_consistent** self-test: corrupt a key on purpose in a test and
  assert the oracle panics (validates the validator).

Gates (D24): once design/05-movegen-perft.md lands, the perft suite is the
make/unmake oracle — startpos perft(6) = 119,060,324; Kiwipete perft(5) =
193,690,690; the standard positions 3–6 — run in debug (assert_consistent
at every node) and release. No change to this module lands afterwards
without the suite passing. Once `bench` exists, "non-functional" changes
here must reproduce the bench node count exactly.

## 6. Performance notes

Targets and expectations:

- `make_move` + `unmake_move` must stay allocation-free (Vec pre-reserved;
  push of a ~176-byte Copy struct is a straight memcpy — same cost shape as
  SF's prefix-memcpy + field writes).
- Quiet non-pawn move fast path touches: one `CopiedState` copy, ~6 XORs,
  castling-mask lookup, `move_piece` (3 bitboard XORs + 2 mailbox writes),
  `set_check_info` (2 × slider_blockers + 4 attack lookups) and a
  repetition scan bounded by `min(rule50, plies_from_null)`. No
  `attackers_to` on quiet moves — that's the point of caller-supplied
  `gives_check` and the checkers shortcut.
- Measure with perft (release, `target-cpu=x86-64-v3`, D16): perft is
  make/unmake + movegen with nothing else, so it's our microbenchmark.
  Target: within ~2× of Stockfish's `perft 6` wall time on the same
  machine at parity of legality strategy; track nodes/sec in the bench
  notes from the first working build.
- Pitfalls:
  - Vec growth mid-search: reserve `MAX_PLY + 64` beyond the game-history
    length at `from_fen`/`ucinewgame`; growth is correct but shows up as
    latency spikes.
  - `Option<Piece>` in the mailbox: `Piece` has no niche, so
    `Option<Piece>` is 2 bytes — fine; but avoid `unwrap()` chains in the
    hot path (use `debug_assert` + direct destructuring).
  - Popcount-based piece counts in material-key updates run only on
    captures/promotions; if profiling ever shows them, add a
    `piece_count: [u8; 12]` cache (Open question 2) — do not pre-add it.
  - Bounds checks on `states[n - i]` in the repetition scan: index
    arithmetic on a local slice reference lets LLVM elide most; only
    revisit with `get_unchecked` after a profile per D14(c).
  - Don't recompute `set_check_info` in `unmake_move` — restoring the
    board and popping the state is the entire function; resist symmetry.
- Prefetch (hooks A/B) historically buys a few percent NPS once the TT is
  large; wire and measure when the TT doc lands — the seam must be
  bench-neutral until then (empty monomorphized hooks).

## 7. Interactions

- **Depends on** design/03-attack-tables.md (D12): pawn/knight/king attack
  tables, magic slider attacks, `BETWEEN`, `LINE` — consumed by
  `attackers_to`, `slider_blockers`, `set_check_info`, `gives_check`,
  `see_ge`, ep validation, and (later) cuckoo table construction.
- **Depends on** the core-types design doc for `Value` and its constants
  (D5); this doc adds `Key`, `Piece::index`, `PIECE_VALUE`.
- **Feeds** design/05-movegen-perft.md: movegen reads `checkers`,
  `blockers_for_king`, `pinners`, ep square, castling paths/rook squares;
  `legal(m)`/`pseudo_legal(m)` live there but are pure consumers of this
  state. Perft is this doc's acceptance oracle (D24).
- **Feeds** the TT design doc (D19): `key()` (rule50-folded) is the probe
  key; hook A is the TT prefetch point; `prefetch_key` supports the
  earlier search-side prefetch; the "no TT cutoff at rule50 ≥ 96" guard
  belongs there.
- **Feeds** the search design doc (D17/D18): `make_move(m, gives_check)`
  call contract, `gives_check(m)`, `is_draw(ply)`, `has_repeated`,
  `upcoming_repetition(ply)`, null-move API, `non_pawn_material` (null-move
  guard), mate/ply arithmetic via `game_ply`.
- **Feeds** the movepick/history design doc: `see_ge` for capture staging;
  `pawn_key`/`minor_key`/`non_pawn_key` for correction histories (hook B).
- **Feeds** the eval/NNUE design doc (D21/D22): the `MakeMoveHooks` seam is
  where dirty-piece accumulation attaches; `non_pawn_material` and
  `material_key` for phase/endgame logic.
- Sibling filenames other than 03/05 are per the suite plan and may shift;
  the topic mapping above is the contract.

## 8. Acceptance criteria (D26)

This component is done when all of the following hold:

1. `cargo test` green with all §5 unit tests present; `cargo clippy
   --all-targets` clean; zero `unsafe` in this component (the prefetch
   intrinsic itself belongs to the TT phase).
2. Key-invariance property holds over at least 10,000 positions of random
   legal walks (once movegen lands): incremental keys ≡ from-scratch
   recompute for all six keys, and unmake restores an equal `Board`.
3. Full perft suite exact in BOTH release and debug-with-oracle builds
   (startpos 6, Kiwipete 5, positions 3–6 at standard depths) — shared
   gate with design/05-movegen-perft.md (D24).
4. En-passant keying is legality-exact: the §5 ep corner cases (vertical
   pin, horizontal double-removal pin, discovered-check double push, FEN
   spurious-ep) all pass.
5. `see_ge` passes the fixture suite including pinned-attacker and x-ray
   cases, using exactly the D11 values.
6. `is_draw`/`is_repetition`/`has_repeated` match the SF semantics tests
   (encoding, windows, ply-relativity), and repetition compares raw keys
   while `key()` folds rule50 at ≥ 14 in 8-step buckets.
7. `StateInfo` is `Copy`, ≤ 192 bytes, and `make_move` performs no heap
   allocation (asserted by a test using a counting allocator, or by
   pre-reserve + capacity assertion).
8. The hook seam exists with `NoHooks` and demonstrably identical perft
   NPS (±noise) to a build without the seam.
9. Cuckoo/`upcoming_repetition` is NOT required for done — it is
   phase-tagged to the search-refinement phase; its spec (§3.9) is the
   contract when it lands (table count 3668 as the build-time assert).

## 9. Open questions

1. **CLAUDE.md hint amended**: Zobrist does NOT hook into
   `put_piece`/`remove_piece` (contra the "will hook in there" note);
   key deltas are computed explicitly in `make_move`, SF-style, because
   the five-key family and ep/castling/side keys don't map onto placement
   ops. The centralized mutators remain the seam for NNUE dirty pieces
   instead. CLAUDE.md should be updated when this lands. (Not a
   decisions.md deviation — no decision mandates the hook.)
2. **Piece-count cache** (`[u8; 12]`): deferred; popcount on the
   capture/promotion paths until a profile or eval (D21-A material terms)
   asks for it.
3. **`is_draw` legal-move guard ordering**: CLOSED in P3 — the rule50
   branch now consults `movegen::has_legal_moves` (design/05-movegen.md
   §3.9): in check on a full clock the draw stands unless it is checkmate.
4. **Sibling doc filenames**: references to the core-types, TT, search,
   movepick, and eval docs use topic names; exact filenames to be
   reconciled when the suite is assembled.
