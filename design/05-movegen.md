# Design 05 — Move generation

Status: Reviewed v1

This document specifies `src/movegen.rs`: type-parameterized pseudo-legal move
generation (`CAPTURES`, `QUIETS`, `EVASIONS`, `NON_EVASIONS` and a `LEGAL`
wrapper), the lazy legality model (`Board::legal`), the full TT-move validator
(`Board::pseudo_legal`), and the generator-consistent capture predicate
(`Board::capture_stage`). The design mirrors current Stockfish semantics
exactly — including the subtle CAPTURES/QUIETS promotion split that the
MovePicker's staged generation and the search's statistics depend on — while
expressing everything through our existing types (`Board`, `Bitboard`, `Move`,
`MoveList`) and the `attacks` module. Perft is this component's correctness
oracle (D24); the perft algorithm itself lives in the dedicated `perft`
module owned by `design/06-perft-validation.md` (D13) — this doc restates
only the known-count gates movegen must pass (§5.3), while the bench/nps
harness lives in the performance doc.

## 1. Purpose and scope

**This document owns:**

- The `movegen` module (D13): `generate::<GT>`, `generate_legal`, the
  `GenType` constants, and the internal pawn/piece/king/castling generators.
- `Board::legal(m)` — the pseudo-legal → legal filter.
- `Board::pseudo_legal(m)` — the from-scratch validator for TT moves
  (every check spelled out in §3.10).
- `Board::capture_stage(m)` and `Board::is_capture(m)` — the predicates the
  search and MovePicker use to classify moves consistently with the generator.
- `Board::attackers_to_exist(sq, occ, c)` — the early-exit boolean variant of
  `attackers_to` that legality testing hammers (§3.8;
  `design/04-position-state.md` provides the bitboard-returning
  `attackers_to`).
- `has_legal_moves()` — the mate/stalemate probe that
  `design/04-position-state.md`'s draw/mate logic expects from this module.
- One small `MoveList` extension: `swap_remove` (used by `generate_legal`).
- The perft-count *gate* movegen must pass (§5.3 — a restatement of
  `design/06-perft-validation.md`'s suite, which is normative).

**This document explicitly does NOT own:**

- Perft — per D13 it is its own module, `src/perft.rs`. The algorithm (bulk
  counting), `perft_checked`, `divide(board, depth) -> DivideResult`
  (per-root-move breakdown), the full validation suite, and the test
  tiers/budgets: `design/06-perft-validation.md` §3.1–§3.5. The `bench`
  subcommand, criterion harness, and nps targets are owned by
  `design/14-performance-testing.md`.
- Attack tables — `knight_attacks`, `king_attacks`, `pawn_attacks`,
  `pawn_attacks_bb`, slider `attacks(pt, sq, occ)`, `between`, `line`,
  `aligned`: `design/03-attack-tables.md` (D12).
- `Direction`, `Square::offset`, `Bitboard::shift`, `more_than_one`,
  `Move::raw16`/`from_raw16`: `design/02-core-types.md`.
- `StateInfo` derived fields (`checkers`, `blockers_for_king`, `pinners`,
  `check_squares`), `attackers_to`, `king_square`, make/unmake,
  `gives_check`, SEE (`see_ge`, D11), and the en-passant-square recording
  rule (D8, D9, D10): `design/04-position-state.md`.
- Move *ordering*, `ExtMove`, staged emission, history scoring:
  `design/09-movepick-history.md` (D7, D17).

## 2. References

- `stockfish_reference/03-movegen-movepick-history.md` — GenType
  monomorphization, target-mask trick, bulk pawn shifts, lazy legality,
  MoveList/ExtMove split.
- `stockfish_reference/stockfish_search_notes.md` §7 — CAPTURES/QUIETS
  semantics (queen promotions vs. underpromotions) and `capture_stage`'s role
  in search (`ttCapture`, stats classification).
- Verified against SF source (ideas and constants only, never code — D27):
  - `Stockfish/src/movegen.cpp` — `make_promotions`, `generate_pawn_moves`,
    `generate_moves`, `generate_all`, `generate<LEGAL>` (lines 86–288).
  - `Stockfish/src/position.cpp` — `legal` (660–698), `pseudo_legal`
    (703–753), en-passant recording in `set` (388–418) and `do_move`
    (920–948), `attackers_to` (640–657).
  - `Stockfish/src/position.h` — `capture` and `capture_stage` (346–372).
  - `Stockfish/src/movepick.cpp` `score()` (190–262) — confirms the picker
    generates into a plain move list and copies into its own `ExtMove` buffer
    while scoring, so `MoveList` stays 2 bytes/move as-is.

## 3. Design

### 3.1 GenType and entry points

Stockfish monomorphizes one generator template per `GenType`; the Rust
equivalent is a `const`-generic parameter. Arbitrary-enum const parameters
(`adt_const_params`) are not stable, so the parameter is a `u8` with named
constants (§9 Q3):

```rust
// src/movegen.rs
pub mod gen_type {
    pub const CAPTURES: u8 = 0;     // captures + queen promotions (§3.2)
    pub const QUIETS: u8 = 1;       // non-captures + underpromotions
    pub const EVASIONS: u8 = 2;     // check evasions (side to move in check)
    pub const NON_EVASIONS: u8 = 3; // all pseudo-legal moves (not in check)
}

/// Append all pseudo-legal moves of the given type to `list`.
/// Does NOT clear `list` first (callers compose sub-ranges).
/// debug_asserts that GT == EVASIONS if and only if the mover is in check.
pub fn generate<const GT: u8>(board: &Board, list: &mut MoveList);

/// Append all fully legal moves (EVASIONS or NON_EVASIONS + legality filter).
pub fn generate_legal(board: &Board, list: &mut MoveList);

// Ergonomic wrappers (thin, #[inline]):
pub fn generate_captures(board: &Board, list: &mut MoveList);
pub fn generate_quiets(board: &Board, list: &mut MoveList);
pub fn generate_evasions(board: &Board, list: &mut MoveList);
pub fn generate_non_evasions(board: &Board, list: &mut MoveList);
```

`generate` dispatches once on the side to move into a color-monomorphized
worker, so pawn directions and rank masks are compile-time constants exactly
as in SF's `template<Color Us>`:

```rust
pub fn generate<const GT: u8>(board: &Board, list: &mut MoveList) {
    debug_assert_eq!(GT == gen_type::EVASIONS, !board.checkers().is_empty());
    match board.side_to_move() {
        Color::White => generate_all::<GT, true>(board, list),
        Color::Black => generate_all::<GT, false>(board, list),
    }
}
```

Monomorphization note: inside the workers, conditions like
`if GT != gen_type::CAPTURES { ... }` are ordinary `if`s over a const
parameter; each of the 4×2 instantiations compiles to a straight-line routine
with the dead branches removed (verified by inspecting `--emit=asm` once, then
trusted). This is the exact effect of SF's `if constexpr`.

### 3.2 Generation semantics per type (the SF split, exactly)

| GenType        | Emits                                                                                              |
|----------------|-----------------------------------------------------------------------------------------------------|
| `CAPTURES`     | All captures (incl. en passant), **all capturing promotions (Q/R/B/N)**, **non-capturing queen promotions** |
| `QUIETS`       | All non-captures except promotions, castling, **non-capturing underpromotions (R/B/N)**              |
| `EVASIONS`     | King moves to non-own squares; if single check: captures of the checker, interpositions, and all promotions landing on the block/capture ray |
| `NON_EVASIONS` | Everything pseudo-legal = `CAPTURES` ⊎ `QUIETS` (disjoint union, a tested invariant, §5.1)            |

Promotion split in full (this is what `make_promotions` in SF encodes; the
table is the normative spec for our `push_promotions`, §3.4):

| Promotion kind          | Queen emitted by                          | R/B/N emitted by                          |
|-------------------------|-------------------------------------------|-------------------------------------------|
| capturing (diagonal)    | CAPTURES, EVASIONS, NON_EVASIONS          | CAPTURES, EVASIONS, NON_EVASIONS          |
| non-capturing (push)    | CAPTURES, EVASIONS, NON_EVASIONS          | QUIETS, EVASIONS, NON_EVASIONS            |

So: `QUIETS` never emits a capturing promotion; `CAPTURES` never emits a
non-capturing underpromotion. Under `EVASIONS` the capture targets are
restricted to the checker and push-promotions to the blocking ray (§3.4, §3.6).

**`Board::capture_stage(m)`** must mirror this split exactly, because the
MovePicker relies on it to avoid emitting a move twice across the
capture/quiet stages, and the search uses it for `ttCapture` and stats
classification:

```rust
impl Board {
    /// True iff `m` is generated by the CAPTURES stage: any capture,
    /// or any queen promotion (even non-capturing).
    pub fn capture_stage(&self, m: Move) -> bool {
        match m.kind() {
            MoveKind::Normal => self.all_occupancy().contains(m.to()),
            MoveKind::Promotion => self.all_occupancy().contains(m.to())
                || m.promotion_piece() == Some(PieceType::Queen),
            MoveKind::EnPassant => true,
            MoveKind::Castling => false,
        }
    }

    /// True iff `m` actually captures a piece (material changes).
    pub fn is_capture(&self, m: Move) -> bool {
        match m.kind() {
            MoveKind::Normal | MoveKind::Promotion => {
                self.all_occupancy().contains(m.to())
            }
            MoveKind::EnPassant => true,
            MoveKind::Castling => false,
        }
    }
}
```

Precondition for both: `m` is pseudo-legal in `self` (side to move owns the
piece on `m.from()`); on arbitrary moves the answer is meaningless but must
not panic.

### 3.3 The single target mask

One `Bitboard` parameterizes every non-king generator; attack sets are ANDed
with it:

| GenType        | `target`                                                    |
|----------------|-------------------------------------------------------------|
| `CAPTURES`     | `board.occupancy(them)`                                     |
| `QUIETS`       | `!board.all_occupancy()`                                    |
| `NON_EVASIONS` | `!board.occupancy(us)`                                      |
| `EVASIONS`     | `attacks::between(ksq, checker_sq)` — **includes** the checker square (block or capture); `checker_sq = board.checkers().lsb().unwrap()` |

The `attacks::between(s1, s2)` convention (strictly-between OR'd with `s2`)
is fixed in `design/03-attack-tables.md` §3.4 precisely so this line needs no
`| checker_bb` fixup.

Top-level worker:

```rust
fn generate_all<const GT: u8, const WHITE: bool>(board: &Board, list: &mut MoveList) {
    use gen_type::*;
    let us = if WHITE { Color::White } else { Color::Black };
    let ksq = board.king_square(us);

    // Double check: only king moves can be legal — skip all non-king work.
    if GT != EVASIONS || !board.checkers().more_than_one() {
        let target = match GT {
            EVASIONS => attacks::between(ksq, board.checkers().lsb().unwrap()),
            NON_EVASIONS => !board.occupancy(us),
            CAPTURES => board.occupancy(us.flip()),
            _ /* QUIETS */ => !board.all_occupancy(),
        };
        pawn_moves::<GT, WHITE>(board, target, list);
        let occ = board.all_occupancy();
        piece_moves(board.pieces(us, PieceType::Knight), target,
                    |from| attacks::knight_attacks(from), list);
        piece_moves(board.pieces(us, PieceType::Bishop), target,
                    |from| attacks::bishop_attacks(from, occ), list);
        piece_moves(board.pieces(us, PieceType::Rook), target,
                    |from| attacks::rook_attacks(from, occ), list);
        piece_moves(board.pieces(us, PieceType::Queen), target,
                    |from| attacks::queen_attacks(from, occ), list);
    }

    // King moves: under EVASIONS the king may go anywhere not our own
    // (fleeing is not restricted to the block/capture ray).
    let king_target = match GT {
        EVASIONS | NON_EVASIONS => !board.occupancy(us),
        CAPTURES => board.occupancy(us.flip()),
        _ /* QUIETS */ => !board.all_occupancy(),
    };
    for to in attacks::king_attacks(ksq) & king_target {
        list.push(Move::normal(ksq, to));
    }

    if GT == QUIETS || GT == NON_EVASIONS {
        castling_moves::<WHITE>(board, list);
    }
}
```

Notes:

- Under double check (`more_than_one`, from `design/02-core-types.md`:
  `bits & (bits - 1) != 0`), only the king block runs — SF's exact shortcut.
- King evasion moves are NOT safety-tested here; `Board::legal` does that
  (with the king removed from occupancy so sliders x-ray through the vacated
  square, §3.8).
- `generate` *appends*; it never clears. `generate_legal` and the MovePicker
  compose ranges by remembering `list.len()` before the call.

### 3.4 Pawn generation — bulk shifts

Never loop per pawn: shift the whole pawn set and recover the from-square as
`to − offset`, since every pawn moved by the same offset. All shift helpers
come from `Bitboard::shift(Direction)` (`design/02-core-types.md`); their
formulas, restated with constants because the from-recovery depends on them:

| Direction  | offset | shift formula                    |
|------------|--------|----------------------------------|
| North      | +8     | `bb << 8`                        |
| South      | −8     | `bb >> 8`                        |
| NorthEast  | +9     | `(bb & !FILE_H) << 9`            |
| NorthWest  | +7     | `(bb & !FILE_A) << 7`            |
| SouthEast  | −7     | `(bb & !FILE_H) >> 7`            |
| SouthWest  | −9     | `(bb & !FILE_A) >> 9`            |

Per-color compile-time constants (white / black):

- `UP` = North / South; `UP_RIGHT` = NorthEast / SouthWest;
  `UP_LEFT` = NorthWest / SouthEast; `DOWN` = the opposite of `UP`.
- `RANK_7_REL` = `Bitboard::RANKS[6]` (0x00FF_0000_0000_0000) /
  `RANKS[1]` (0x0000_0000_0000_FF00) — pawns about to promote.
- `RANK_3_REL` = `Bitboard::RANKS[2]` (0x0000_0000_00FF_0000) /
  `RANKS[5]` (0x0000_FF00_0000_0000) — the double-push relay rank.

```rust
fn pawn_moves<const GT: u8, const WHITE: bool>(
    board: &Board, target: Bitboard, list: &mut MoveList,
) {
    use gen_type::*;
    let us = if WHITE { Color::White } else { Color::Black };
    let them = us.flip();
    let up      = if WHITE { Direction::North } else { Direction::South };
    let up_right = if WHITE { Direction::NorthEast } else { Direction::SouthWest };
    let up_left  = if WHITE { Direction::NorthWest } else { Direction::SouthEast };
    let down    = if WHITE { Direction::South } else { Direction::North };
    let rank7 = if WHITE { Bitboard::RANKS[6] } else { Bitboard::RANKS[1] };
    let rank3 = if WHITE { Bitboard::RANKS[2] } else { Bitboard::RANKS[5] };

    let empty = !board.all_occupancy();
    // EVASIONS: pawn captures may only take the checking piece.
    let enemies = if GT == EVASIONS { board.checkers() } else { board.occupancy(them) };
    let pawns = board.pieces(us, PieceType::Pawn);
    let on7 = pawns & rank7;
    let not7 = pawns & !rank7;

    // --- 1. Single and double pushes (everything except CAPTURES) ---
    if GT != CAPTURES {
        let mut b1 = not7.shift(up) & empty;
        // Rank-3 relay trick: a double push exists iff the single push landed
        // on relative rank 3 AND one more shift is also empty — both-empty
        // tested with two ANDs, zero per-pawn work.
        let mut b2 = (b1 & rank3).shift(up) & empty;
        if GT == EVASIONS {              // only interpositions count
            b1 &= target;
            b2 &= target;
        }
        for to in b1 { list.push(Move::normal(to.offset(down), to)); }
        for to in b2 { list.push(Move::normal(to.offset(down).offset(down), to)); }
    }

    // --- 2. Promotions (from-square recovery: to − direction) ---
    if !on7.is_empty() {
        if GT != QUIETS {                // capturing promotions
            for to in on7.shift(up_right) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_right)), to);
            }
            for to in on7.shift(up_left) & enemies {
                push_promotions::<GT, true>(list, to.offset(opposite(up_left)), to);
            }
        }
        let mut b3 = on7.shift(up) & empty;   // push promotions
        if GT == EVASIONS { b3 &= target; }   // must land on the block ray
        for to in b3 {
            push_promotions::<GT, false>(list, to.offset(down), to);
        }
    }

    // --- 3. Ordinary and en-passant captures (not for QUIETS) ---
    if GT != QUIETS {
        for to in not7.shift(up_right) & enemies {
            list.push(Move::normal(to.offset(opposite(up_right)), to));
        }
        for to in not7.shift(up_left) & enemies {
            list.push(Move::normal(to.offset(opposite(up_left)), to));
        }
        if let Some(ep) = board.en_passant_square() {
            // EVASIONS: if the square *behind* the ep square (one further
            // `up` from our view — the double-push origin) lies on the
            // checking ray, the check is a discovered slider check through
            // that square, and an ep capture can never resolve it. Skip.
            if GT == EVASIONS && target.contains(ep.offset(up)) {
                return;
            }
            // Our pawns attacking the ep square = pawn attacks FROM ep as
            // if an enemy pawn stood there (standard reversal trick).
            for from in not7 & attacks::pawn_attacks(them, ep) {
                list.push(Move::en_passant(from, ep));
            }
        }
    }
}
```

`push_promotions` — the normative encoding of the §3.2 table (queen first,
then R, B, N, matching SF's emission order for bench determinism):

```rust
#[inline(always)]
fn push_promotions<const GT: u8, const CAPTURE: bool>(
    list: &mut MoveList, from: Square, to: Square,
) {
    use gen_type::*;
    let all = GT == EVASIONS || GT == NON_EVASIONS;
    if GT == CAPTURES || all {
        list.push(Move::promotion(from, to, PieceType::Queen));
    }
    if (GT == CAPTURES && CAPTURE) || (GT == QUIETS && !CAPTURE) || all {
        list.push(Move::promotion(from, to, PieceType::Rook));
        list.push(Move::promotion(from, to, PieceType::Bishop));
        list.push(Move::promotion(from, to, PieceType::Knight));
    }
}
```

From-square recovery safety: every `to` in a shifted-and-masked bitboard has
a valid, non-wrapping origin by construction, so `Square::offset` (safe;
panics only on a genuine bug, per `design/02-core-types.md` §3.5 policy) is
correct here. `opposite(d)` is the direction negation; if
`Direction::opposite` does not land in doc 02, movegen keeps a private
4-line helper.

En-passant invariant: per D9 (owned by design/04-position-state.md), the ep square is
only recorded when at least one *legal* ep capture exists (SF verifies this
both on FEN load and in `do_move`). Once that normalization lands, movegen
adds `debug_assert!(!(not7 & attacks::pawn_attacks(them, ep)).is_empty())`.
Until then the capture loop simply finds no capturers and emits nothing —
correct either way (§4 step 7).

### 3.5 Piece-loop generation

One shared inner loop, monomorphized per call site via `impl Fn` (each call
gets its own copy with the attack lookup inlined — the Rust idiom for SF's
`template<PieceType Pt>`):

```rust
#[inline(always)]
fn piece_moves(
    pieces: Bitboard,
    target: Bitboard,
    attack: impl Fn(Square) -> Bitboard,
    list: &mut MoveList,
) {
    for from in pieces {
        for to in attack(from) & target {
            list.push(Move::normal(from, to));
        }
    }
}
```

Occupancy is captured once in `generate_all` (§3.3) and moved into the slider
closures; leaper closures ignore it. King and pawn moves never go through
this path (king: single square, no loop needed; pawns: §3.4).

### 3.6 Evasion generation

Fully covered by the pieces above; collected here as the normative rules:

1. `generate::<EVASIONS>` must only be called when `checkers()` is non-empty
   (debug_assert in `generate`).
2. Double check (`checkers().more_than_one()`): king moves only.
3. Single check: non-king target = `between(ksq, checker)` **including** the
   checker — block or capture in one mask. Pawn captures use
   `enemies = checkers()`; pawn pushes and push-promotions are ANDed with the
   target; ep is skipped when the check is a discovered slider check through
   the double-push origin (§3.4).
4. King evasions use `!occupancy(us)` and are NOT safety-filtered here;
   `Board::legal` rejects moves along the checker's x-ray (it removes the
   king's from-square from occupancy).
5. Castling is never generated under EVASIONS (or CAPTURES).

The pseudo-legal evasion list may therefore contain illegal moves in exactly
three categories — pinned-piece interpositions, unsafe king steps, illegal
ep — all caught by the §3.9 filter or by the search's per-move `legal` call.

### 3.7 Castling generation

Generated only for `QUIETS` and `NON_EVASIONS` (a castle never captures and
never evades — the mover is not in check in both contexts). Our encoding is
the king's two-square move, `Move::castling(E1, G1)` etc. (D2, D10; UCI
notation). Conditions checked at generation time:

1. The right is present in `board.castling_rights()`.
2. The path between king and rook is empty (rights guarantee king/rook
   placement; make/unmake maintains rights on any king/rook move or capture —
   `design/04-position-state.md`).

Attacked-square testing is deliberately deferred to `Board::legal` (§3.8),
matching SF ("delayed at a later time: now!").

Constants (standard chess; Chess960 deferred per D10):

| Right           | Move            | Path-empty mask (`u64`)       | Squares |
|-----------------|-----------------|-------------------------------|---------|
| White kingside  | `E1 → G1`       | `0x0000_0000_0000_0060`       | f1, g1  |
| White queenside | `E1 → C1`       | `0x0000_0000_0000_000E`       | b1, c1, d1 |
| Black kingside  | `E8 → G8`       | `0x6000_0000_0000_0000`       | f8, g8  |
| Black queenside | `E8 → C8`       | `0x0E00_0000_0000_0000`       | b8, c8, d8 |

```rust
fn castling_moves<const WHITE: bool>(board: &Board, list: &mut MoveList) {
    // (right, path, king_from, king_to) per color, from the table above
    for (right, path, from, to) in castle_specs::<WHITE>() {
        if board.castling_rights().contains(right)
            && (board.all_occupancy() & path).is_empty()
        {
            list.push(Move::castling(from, to));
        }
    }
}
```

Small ergonomic addition to `CastlingRights` (board.rs):
`pub const fn kingside(color: Color) -> CastlingRights` and
`pub const fn queenside(color: Color) -> CastlingRights`, mapping to the four
existing constants.

### 3.8 Legality: `Board::legal(m)`

Precondition: `m` is pseudo-legal for the side to move (generated by §3.1–3.7
or validated by §3.10). Returns whether playing `m` leaves our king
unattacked. Uses only current state — no make/unmake.

```rust
impl Board {
    pub fn legal(&self, m: Move) -> bool {
        let us = self.side_to_move();
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let ksq = self.king_square(us);

        match m.kind() {
            // 1. Castling: the two squares the king crosses/lands on must be
            //    unattacked. from (e1/e8) is not tested — castling is never
            //    generated nor pseudo_legal-validated while in check.
            MoveKind::Castling => {
                let kingside = to > from;   // g-file destination
                let tested: [Square; 2] = /* per color/side from §3.7 table:
                    kingside → [F1, G1] / [F8, G8]
                    queenside → [D1, C1] / [D8, C8] */;
                !tested.iter().any(|&s|
                    self.attackers_to_exist(s, self.all_occupancy(), them))
                // Chess960 (deferred, D10): additionally verify the moving
                // rook was not shielding the king from a slider.
            }

            // 2. En passant: the only move that clears TWO squares at once.
            //    Rebuild occupancy and probe sliders from the king square.
            MoveKind::EnPassant => {
                let down = /* South for White mover, North for Black */;
                let capsq = to.offset(down);
                let occ = (self.all_occupancy()
                    ^ Bitboard::from_square(from)
                    ^ Bitboard::from_square(capsq))
                    | Bitboard::from_square(to);
                (attacks::rook_attacks(ksq, occ)
                    & (self.pieces(them, PieceType::Rook)
                       | self.pieces(them, PieceType::Queen))).is_empty()
                && (attacks::bishop_attacks(ksq, occ)
                    & (self.pieces(them, PieceType::Bishop)
                       | self.pieces(them, PieceType::Queen))).is_empty()
            }

            // 3. King steps: destination must be safe with the king REMOVED
            //    from occupancy, so a checking slider's ray extends through
            //    the vacated square (fleeing along the ray is illegal).
            _ if from == ksq => !self.attackers_to_exist(
                to, self.all_occupancy() ^ Bitboard::from_square(from), them),

            // 4. Everything else: legal iff not pinned, or moving along the
            //    pin ray (toward or away from the king).
            _ => !self.blockers_for_king(us).contains(from)
                || attacks::aligned(from, to, ksq),
        }
    }
}
```

`Board::attackers_to_exist(sq, occ, c)` (owned here; the bitboard-returning
`attackers_to` lives in `design/04-position-state.md` §3.5): true iff any of

- `rook_attacks(sq, occ) & (c rooks | c queens)`
- `bishop_attacks(sq, occ) & (c bishops | c queens)`
- `pawn_attacks(c.flip(), sq) & c pawns` (reversal trick)
- `knight_attacks(sq) & c knights`
- `king_attacks(sq) & c king`

is non-empty, with early exit between terms.

Deviation note (recorded, not silent — §9 Q1): current SF has **no**
en-passant branch in `legal()` at all; it relies on the ep square being
recorded only when a fully legal ep capture exists (position.cpp 388–418 and
920–948, our D9), plus the pin test for multi-capturer cases. We keep the
classic occupancy-swap test because it makes `legal` self-contained and
correct independent of the recording invariant (defense-in-depth for the
horizontally-double-removed-pawn case, e.g. `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3`),
at the cost of two slider probes on a rare move type.

### 3.9 `generate_legal` — lazy filtering with swap-remove

Only three kinds of pseudo-legal moves can be illegal: moves of a pinned
piece, king moves, and en passant. Everything else is legal by construction
and is not tested — that is the entire point of the lazy model. Removal is
swap-with-last; order does not matter because consumers either sort
(MovePicker) or count (perft).

```rust
pub fn generate_legal(board: &Board, list: &mut MoveList) {
    let start = list.len();
    if board.checkers().is_empty() {
        generate::<{ gen_type::NON_EVASIONS }>(board, list);
    } else {
        generate::<{ gen_type::EVASIONS }>(board, list);
    }

    let us = board.side_to_move();
    let pinned = board.blockers_for_king(us) & board.occupancy(us);
    let ksq = board.king_square(us);

    let mut i = start;
    while i < list.len() {
        let m = list[i];
        let suspect = pinned.contains(m.from())
            || m.from() == ksq
            || m.kind() == MoveKind::EnPassant;
        if suspect && !board.legal(m) {
            list.swap_remove(i);        // do not advance i
        } else {
            i += 1;
        }
    }
}
```

`blockers_for_king(us)` contains blockers of *both* colors (any single piece
between a slider and the king); intersecting with `occupancy(us)` yields the
pinned set — an enemy blocker is the enemy's problem.

The search hot path never calls `generate_legal`: the MovePicker emits
pseudo-legal moves and the search calls `board.legal(m)` only on the move it
is about to play (`design/09-movepick-history.md`). `generate_legal` is for
perft, root-move-list setup, UCI `position ... moves` parsing, and
mate/stalemate detection.

For the latter, an early-exit probe (consumed by
`design/04-position-state.md`'s draw/mate logic and the UCI layer) avoids
building the full list:

```rust
/// True if the side to move has at least one legal move. Same generation +
/// lazy-filter logic as generate_legal, but returns on the first survivor.
pub fn has_legal_moves(board: &Board) -> bool;
```

New `MoveList` method (src/moves.rs), the only MoveList change — the type
stays a thin `[Move; 256]` + len, per the reference note and D7 (`ExtMove`
lives in the picker, not here):

```rust
impl MoveList {
    /// O(1) unordered removal: overwrite index `i` with the last move and
    /// shrink by one. Panics if `i >= len` (index invariant, matches Vec).
    pub fn swap_remove(&mut self, i: usize) -> Move;
}
```

### 3.10 `Board::pseudo_legal(m)` — the TT-move validator

TT moves are raw `u16`s that may be corrupted by data races or key aliasing
(D19); before use they must be proven pseudo-legal in the *current* position
(then `legal()` is applied separately). The validator must accept exactly the
moves that `generate::<EVASIONS|NON_EVASIONS>` would emit. Every check:

```rust
impl Board {
    pub fn pseudo_legal(&self, m: Move) -> bool {
        let us = self.side_to_move();
        let them = us.flip();
        let (from, to) = (m.from(), m.to());

        // 1. Non-normal kinds (promotion / ep / castling) are rare: validate
        //    by containment in a fresh generation. This inherits every rule
        //    (promotion geometry, ep square match, castling rights + empty
        //    path) with zero duplicated logic. Legality is NOT included.
        if m.kind() != MoveKind::Normal {
            let mut list = MoveList::new();
            if self.checkers().is_empty() {
                generate::<{ gen_type::NON_EVASIONS }>(self, &mut list);
            } else {
                generate::<{ gen_type::EVASIONS }>(self, &mut list);
            }
            return list.contains(&m);
        }

        // 2. Bit hygiene: a Normal move must have zero promotion bits, or it
        //    could never equal a generated move (and would corrupt stats
        //    keyed by the raw encoding). Rejects garbage u16s. Also rejects
        //    Move::NULL (from == to fails checks 3–5 in every case).
        if m.raw16() & 0x3000 != 0 {
            return false;
        }

        // 3. A piece of ours must sit on `from`.
        let Some(pc) = self.piece_on(from) else { return false };
        if pc.color != us {
            return false;
        }

        // 4. `to` must not hold one of our pieces.
        if self.occupancy(us).contains(to) {
            return false;
        }

        // 5a. Pawns: the destination cannot be a promotion rank (kind Normal),
        //     and the move must be a capture, single push, or double push.
        if pc.piece_type == PieceType::Pawn {
            if (Bitboard::RANK_1 | Bitboard::RANK_8).contains(to) {
                return false;
            }
            let up = /* +8 White / -8 Black, as Direction */;
            let start_rank = if us == Color::White { 1 } else { 6 };
            let is_capture = attacks::pawn_attacks(us, from).contains(to)
                && self.occupancy(them).contains(to);
            let is_single = from.shifted(up) == Some(to)
                && !self.all_occupancy().contains(to);
            let is_double = from.rank() == start_rank
                && from.shifted(up).and_then(|s| s.shifted(up)) == Some(to)
                && !self.all_occupancy().contains(to)
                && !self.all_occupancy().contains(to.offset(/* down */));
            if !(is_capture || is_single || is_double) {
                return false;
            }
        }
        // 5b. Other pieces: `to` must be attacked from `from` on the current
        //     occupancy (handles king steps too; `attacks` dispatches N/B/R/Q/K).
        else if !attacks::attacks(pc.piece_type, from, self.all_occupancy())
            .contains(to)
        {
            return false;
        }

        // 6. In check, a Normal move is pseudo-legal only if the evasion
        //    generator would emit it (blocks, checker captures, king flights).
        if !self.checkers().is_empty() {
            let mut list = MoveList::new();
            generate::<{ gen_type::EVASIONS }>(self, &mut list);
            return list.contains(&m);
        }

        true
    }
}
```

Cost profile: one call per node for the TT move; the slow containment paths
trigger only for non-normal TT moves or in-check nodes — same trade SF makes.
`list.contains(&m)` is the slice method via `Deref<Target = [Move]>`; `Move`
is `PartialEq` on the full 16 bits, so kind and promotion piece must match
exactly.

Search usage contract (consumed by `design/07-search-core.md` /
`design/08-transposition-table.md`): a TT move is used only if
`board.pseudo_legal(tt_move) && board.legal(tt_move)`; the MovePicker's
`stage += !usable_tt_move` trick keys off the same predicate pair.

### 3.11 Module layout and prerequisites

```
src/movegen.rs
├── pub mod gen_type                  // GT constants (§3.1)
├── pub fn generate<const GT: u8>     // + 4 named wrappers
├── pub fn generate_legal
├── fn generate_all<GT, WHITE>        // §3.3
├── fn pawn_moves<GT, WHITE>          // §3.4
├── fn push_promotions<GT, CAPTURE>   // §3.4
├── fn piece_moves                    // §3.5
├── fn castling_moves<WHITE>          // §3.7
└── impl Board { legal, pseudo_legal, capture_stage, is_capture }  // §3.2/3.8/3.10
```

Perft is deliberately NOT here: it lives in its own module `src/perft.rs`
(D13), specified in `design/06-perft-validation.md`, and consumes
`generate_legal` from this module.

`lib.rs` adds `pub mod movegen;`. The `impl Board` block lives in movegen.rs
(same crate, different module — idiomatic and keeps generator-coupled
semantics next to the generator; `pseudo_legal` literally calls `generate`).

Prerequisite APIs (must exist first; see §7):

- `design/02-core-types.md`: `Direction`, `Square::offset` / `Square::shifted`,
  `Bitboard::shift`, `Bitboard::more_than_one`, `Move::raw16`/`from_raw16`.
- `design/03-attack-tables.md`: `attacks::{knight_attacks, king_attacks,
  pawn_attacks, bishop_attacks, rook_attacks, queen_attacks, attacks,
  between, line, aligned}`.
- Board-state doc: `Board::{checkers, blockers_for_king, king_square,
  attackers_to_exist}` (derived `StateInfo` fields, D8); later
  `make_move`/`unmake_move` for full-depth perft, and the D9 ep-recording
  rule.

Out of scope, noted for the record: SF's AVX-512 `splat_pawn_moves` /
`splat_moves` (vectorized emission of up to 8/32 moves per store,
`movegen.cpp` 38–64) — the scalar pop-lsb loop is our baseline; revisit only
after a profile shows generation on the critical path, gated by the unsafe
budget (D14).

## 4. Implementation plan

Each step is one commit-sized unit; "done" always includes `cargo test` green
and `cargo clippy --all-targets` clean.

1. **`MoveList::swap_remove` + `CastlingRights::{kingside, queenside}`.**
   Unit tests: swap_remove middle/last/single element, panic on bad index.
   *Done:* tests pass; `MoveList` still `Deref`s to `[Move]` unchanged.
2. **Module skeleton + non-pawn generation.** `gen_type`, `generate`,
   wrappers, `generate_all` with knight/bishop/rook/queen/king loops and the
   four target masks; pawn/castling/evasion bodies stubbed with `todo!()`
   only if unreachable in tests, otherwise land empty. Tests on pawnless
   FENs with hand-counted moves per GenType (e.g. lone rook + kings:
   CAPTURES/QUIETS counts).
   *Done:* counts match hand calculations; CAPTURES ⊎ QUIETS == NON_EVASIONS
   on the test positions.
3. **Pawn generation.** Pushes with the rank-3 relay, diagonal captures,
   promotion split per §3.2 table, ep captures. Tests: promotion-rich FENs
   (each of the four contexts in the §3.2 table asserted by membership),
   double-push blocking, ep emission.
   *Done:* per-type membership tests pass; disjoint-union property holds on
   pawn-heavy positions.
4. **Evasions.** Target mask, checker-restricted pawn captures, double-check
   king-only, ep discovered-check skip. Tests: single-check block/capture
   sets, double-check king-only assertion, the two ep-evasion FENs (§5.2).
   *Done:* evasion tests pass; `debug_assert` on EVASIONS↔in-check wired.
5. **Castling generation.** §3.7 table. Tests: rights present/absent, each
   path-occupancy square individually blocked.
   *Done:* Kiwipete emits both white castles in QUIETS.
6. **`Board::legal`, `generate_legal`, `capture_stage`, `is_capture`.**
   Requires checkers/blockers from design/04-position-state.md.
   *Done:* **perft(1) gate** — `generate_legal` length matches the depth-1
   column of §5.3 for all six positions (no make/unmake needed yet).
7. **`Board::pseudo_legal`.** Plus the 64k-raw-move fuzz test (§5.2). Enable
   the ep-capturer `debug_assert` (§3.4) once design/04-position-state.md's D9 ep
   normalization has landed; otherwise leave a `TODO(D9)`.
   *Done:* fuzz agreement on all curated positions.
8. **Full-perft gate** (after make/unmake and `src/perft.rs` land — the
   perft implementation itself is `design/06-perft-validation.md`'s plan).
   Run the full-depth suite; criterion baseline recorded per
   `design/14-performance-testing.md`.
   *Done:* every count in §5.3 exact (D24).

## 5. Testing and validation

### 5.1 Property tests (every position in the corpus)

Corpus: the six perft positions (§5.3) + positions reached by applying their
divide moves (once make/unmake exists) + any bench-suite positions.

1. **Disjoint union:** when not in check,
   `set(CAPTURES) ∪ set(QUIETS) == set(NON_EVASIONS)`,
   `set(CAPTURES) ∩ set(QUIETS) == ∅`, and lengths add up (no duplicates
   within a type either).
2. **Predicate/generator consistency:** every move from `CAPTURES` satisfies
   `capture_stage(m) == true`; every move from `QUIETS` satisfies
   `capture_stage(m) == false`. (This is the invariant the MovePicker's
   two-stage dedup silently depends on — test it, don't assume it.)
3. **Evasion subset:** in check, every `generate_legal` move appears in
   `EVASIONS`, and every `EVASIONS` move that survives `legal()` appears in
   `generate_legal`.
4. **Validator agreement (64k fuzz):** for each corpus position, iterate all
   65,536 raw u16 values; `board.pseudo_legal(Move::from_raw16(v))` must
   equal membership of that move in the generated pseudo-legal set
   (EVASIONS when in check, else NON_EVASIONS). This is the direct unit test
   of the TT-corruption defense and is cheap (< 1 ms/position).
5. **Legality oracle (after make/unmake):** for every pseudo-legal move,
   `legal(m)` must equal "make `m`, king not attacked, unmake" — the slow
   reference implementation lives only in the test.

### 5.2 Targeted regression positions

Membership assertions (exact counts verified against `stockfish perft` as
external tooling during test authoring, D15/D25-style: trust games/tools, not
eyeballs):

- **Horizontal ep double-removal** — `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1`
  (not in check; the queen's rank-4 ray is blocked by the two pawns): `e4d3`
  is emitted by `NON_EVASIONS` pseudo-legally but must be **absent** from
  `generate_legal` — capturing removes both pawns from rank 4 and exposes
  the king on a4.
- **Ep as evasion (checker is the double-pushed pawn)** —
  `8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1`: `e4d3` **present** in
  `generate_legal`.
- **Ep discovered-check skip** — position where the double push gives a
  discovered slider check through its origin square (construct per §3.4
  comment); assert no ep move is generated under EVASIONS.
- **Double check** — e.g. a rook+bishop double check; assert every generated
  evasion has `from == king_square`.
- **Castling through check / out of check / path blocked** — variants of
  Kiwipete with an added attacker on f1/g1; assert generation still emits
  the castle (path empty) but `generate_legal` drops it.
- **Underpromotion split** — `n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1`
  (classic promotion position): assert exact QUIETS/CAPTURES partition of
  the 24 depth-1 moves per the §3.2 table.
- **Pinned-piece filter** — absolute pin where the pinned piece may slide
  along the ray (both directions legal) but not off it.

### 5.3 Perft gates (D24 — non-negotiable)

`perft` is defined in `design/06-perft-validation.md` (module `src/perft.rs`,
D13): bulk-counting `perft(board, depth) -> u64`, the debug-checked
`perft_checked`, and `divide(board, depth) -> DivideResult` (per-root-move
breakdown, SF `go perft` style) — see that doc's §3.1–§3.3 for the
algorithm and API and §3.4 for the full validation suite (a superset of the
table below, including the tricky ep-pin/castling/promotion positions). The
CLI/bench integration is `design/14-performance-testing.md`'s. What this
doc owns is the gate: movegen is not done until the counts below match.

Known counts (standard published values; all must match exactly):

| Position (FEN) | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|
| startpos | 20 | 400 | 8,902 | 197,281 | 4,865,609 | 119,060,324 |
| Kiwipete `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 48 | 2,039 | 97,862 | 4,085,603 | 193,690,690 | 8,031,647,685 |
| Pos 3 `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -` | 14 | 191 | 2,812 | 43,238 | 674,624 | 11,030,083 |
| Pos 4 `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 6 | 264 | 9,467 | 422,333 | 15,833,292 | 706,045,033 |
| Pos 5 `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 44 | 1,486 | 62,379 | 2,103,487 | 89,941,194 | — |
| Pos 6 `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 46 | 2,079 | 89,890 | 3,894,594 | 164,075,551 | — |

Test tiers (fast debug tier in plain `cargo test`, deep depths `#[ignore]`d
and run via `cargo test --release -- --ignored` in CI and before any movegen
or make/unmake change lands) are owned by `design/06-perft-validation.md`
§3.5 (D24). Pos 3 exercises ep pins, Pos 4/5 promotion storms and
castling-rights edge cases, Pos 6 quiet middlegame breadth.

### 5.4 Regression strategy

- Any change to movegen, `legal`, `pseudo_legal`, or make/unmake requires the
  full perft tier before merge (D24).
- Emission *order* is part of bench reproducibility (D24): tests pin the
  order class (pushes → promotions → captures → ep within pawns; N→B→R→Q→K;
  castles last; Q→R→B→N within promotions). A deliberate order change must
  note the expected bench-node change.
- The 64k pseudo_legal fuzz runs in the default test suite forever — it is
  the cheapest strong invariant we have.

## 6. Performance notes

Targets (owned by `design/14-performance-testing.md`, restated):
bulk-counted perft ≥ 100 Mnps floor / 250 Mnps target (startpos d6, Kiwipete
d5); full make/unmake perft ≥ 20/50 Mnps. Movegen must never be the reason
search misses SF-class per-node cost.

What to measure (criterion micro-benches + perft macro):

- `generate::<NON_EVASIONS>` / `<CAPTURES>` on the six perft positions
  (ns/position and moves/µs).
- Bulk perft nps; compare against `./stockfish bench ... perft` on the same
  machine for a sanity ratio.

Known pitfalls, in expected order of impact:

1. **`MoveList::new()` zero-initializes 512 bytes** (`[Move::NULL; 256]`).
   SF leaves its buffer uninitialized. Mitigations, in preference order:
   (a) reuse — perft and search keep one `MoveList` per ply and `clear()`
   (len = 0) instead of constructing; (b) verify LLVM elides the memset when
   the list is immediately filled; (c) only if a profile blames it,
   a `MaybeUninit` interior per D14 with a documented measurement. Start
   with (a); do not reach for (c) speculatively.
2. **`Square::new`'s hard `assert!`** runs inside `pop_lsb` and
   `Square::offset` on every emitted move. Policy per
   `design/02-core-types.md`: keep the safe version until a profile shows
   the bounds check; the perft suite is the safety net for any later
   `debug_assert` downgrade (D14).
3. **`MoveList::push`'s overflow `assert!`**: one predictable branch per
   move; keep (256 is a real bound — the known maximum is 218 legal moves —
   but a movegen bug overflowing silently would be far worse than the
   branch).
4. **Monomorphization bloat**: 4 GenTypes × 2 colors ≈ 8 instantiations of a
   small function family — negligible; do not merge them back into runtime
   branches, the per-type dead-code elimination is the whole trick.
5. **Occupancy/target recomputation**: hoist `all_occupancy()` and the
   closures once per `generate_all` (§3.3); slider closures capture it by
   value (`Bitboard` is `Copy`).
6. **AVX-512 splat** (SF's `USE_AVX512ICL` path): out of scope now; the
   scalar loop is the baseline on x86-64-v3 (D16). Revisit only with a
   profile and per D14.

## 7. Interactions

Depends on (must land first):

- `design/02-core-types.md` — `Direction`, `Square::offset/shifted`,
  `Bitboard::shift`/`more_than_one`, `Move::raw16`/`from_raw16` (§3.11).
- `design/03-attack-tables.md` — all attack/geometry queries; the
  `between(s1, s2)`-includes-`s2` convention that makes the evasion target a
  single lookup (§3.3).
- `design/04-position-state.md` (board state / make-unmake doc) — `checkers`,
  `blockers_for_king`, `king_square`, `attackers_to_exist` (D8); the D9
  ep-recording invariant (§3.4, §3.8); `make_move`/`unmake_move` for
  full-depth perft; `gives_check` (D10) is consumed by search, not movegen.

Feeds:

- `design/09-movepick-history.md` — staged generation calls
  `generate::<CAPTURES>` then `generate::<QUIETS>` (or `<EVASIONS>`) into a
  scratch `MoveList` and scores into its own `ExtMove` buffer;
  `capture_stage` powers stage dedup; `pseudo_legal` powers the branchless
  TT-stage skip.
- `design/07-search-core.md` / `design/08-transposition-table.md` — TT-move
  validation contract (`pseudo_legal` + `legal`), `capture_stage` for
  `ttCapture`/stats, per-played-move `legal` check.
- `design/06-perft-validation.md` — `src/perft.rs` (D13) consumes
  `generate_legal` (and the make/unmake machinery) and owns the perft
  algorithm, `divide`, and validation suite; this doc's §5.3 gate is a
  restatement of that suite's standard positions.
- `design/14-performance-testing.md` — the bench and criterion harnesses;
  nps targets defined there.
- The UCI doc — `generate_legal` for `position ... moves` parsing and
  mate/stalemate detection at the root.

(Sibling filenames follow the suite index; where a sibling lands under a
different name, the topic reference governs — see §9 Q2.)

## 8. Acceptance criteria (D26)

Movegen is done when all of the following hold:

1. `generate::<GT>` implemented for all four types + `generate_legal`, with
   semantics matching the §3.2/§3.3 tables; the §5.1 property tests
   (disjoint union, capture_stage consistency, evasion subset, 64k
   pseudo_legal fuzz) pass on the full corpus.
2. **Perft(1)** matches §5.3 for all six positions using `generate_legal`
   alone (pre-make/unmake milestone).
3. **Full perft** matches every §5.3 count exactly (using
   `design/06-perft-validation.md`'s `perft` with
   `design/04-position-state.md`'s make/unmake); test tiers wired per that
   doc's §3.5 (D24).
4. `Board::pseudo_legal` accepts exactly the generated pseudo-legal set on
   every corpus position (64k fuzz), including in-check and
   promotion/ep/castling-rich positions.
5. `Board::legal` verified against the make/unmake oracle (§5.1.5) on the
   corpus with zero disagreements.
6. All §5.2 targeted regressions encoded as unit tests.
7. Zero `unsafe`, zero heap allocation in any generation path
   (`MoveList` on stack; no `Vec` anywhere in this module — `divide`'s
   `Vec` lives in `src/perft.rs`), `cargo clippy --all-targets` clean.
8. Criterion baseline for `generate` and bulk perft recorded; bulk perft
   ≥ 100 Mnps floor on the reference machine (target 250 Mnps per
   `design/14-performance-testing.md`).

## 9. Open questions

1. **Ep legality: occupancy-swap vs. D9 invariant.** Current SF dropped the
   ep branch from `legal()` and instead never records an ep square unless a
   legal ep capture exists. We keep the self-contained occupancy-swap test
   (§3.8) as defense-in-depth. Once design/04-position-state.md's ep-recording rule
   is implemented and perft-proven, we may delete the branch for exact SF
   parity — decision deferred until then; behavior is identical either way.
2. **Sibling filenames.** References to on-disk docs
   (`design/04-position-state.md`, `design/07-search-core.md`, etc.) are
   verified; `design/09-movepick-history.md` had not landed at writing time
   and is cited by expected name. Note also that some siblings cite this doc
   as `design/05-movegen-perft.md` — the actual file is
   `design/05-movegen.md`, and the perft algorithm is NOT here: it lives in
   `src/perft.rs` per D13, specified in `design/06-perft-validation.md`
   (§5.3 here keeps only the gate counts). Topic references govern;
   reconcile names in a docs-sync pass.
3. **GenType representation.** Chosen: `u8` const generic with named
   constants (stable Rust today). If `adt_const_params` stabilizes, switching
   to `enum GenType` as the const parameter is a mechanical cleanup —
   revisit opportunistically, not on the critical path.
4. **`MoveList::new()` init cost.** Decision deferred to a profile (D14):
   reuse-per-ply is the planned mitigation; `MaybeUninit` only with a
   recorded measurement (§6.1).
5. **When to enable the ep-capturer `debug_assert`** (§3.4): blocked on
   `design/04-position-state.md`'s D9 normalization of FEN-loaded ep squares;
   tracked as `TODO(D9)` in code until then.
