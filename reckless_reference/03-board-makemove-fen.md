# Board representation, make/unmake, FEN (Reckless)

Source: `src/board.rs`, `src/board/makemove.rs`, `src/board/parser.rs`.
Followed into: `src/types/keys.rs`, `src/types/zobrist.rs`,
`src/types/castling.rs`, `src/types/piece.rs`, `src/types/moves.rs`,
`src/lookup.rs` (cuckoo tables), `src/setwise.rs` (parallel attack
generation), `src/board/movegen.rs` (castling generation), `src/nnue.rs`
(the one `BoardObserver` impl), `src/search.rs` / `src/movepick.rs` (who
actually consumes `all_threats`/`material`/the specialized keys). Reckless
is AGPLv3 — everything below is prose/pseudocode from reading the code, not
copied source; formulas/constants are quoted narrowly where exactness
matters, with file:line so a future reader can re-find the real thing.

Compare against: `stockfish_reference/02-position-state-make-unmake.md`
(read first — most of what Reckless does is recognizably the SF algorithm)
and our own `src/board.rs` (`CopiedState`/`StateInfo`/`Board`,
`make_move_with`, `unmake_move`) plus `src/zobrist.rs`, governed by
`design/04-position-state.md` (D8/D9/D10/D11).

## 1. Board representation

`Board` (`board.rs:44-56`): bitboard-primary + mailbox, same shape as SF and
as us —

- `pieces: [Bitboard; PieceType::NUM]` — **by piece type only**, not
  `[color][piece_type]`. Reckless gets the per-color slice by ANDing with
  `colors[color]` at the call site (`colored_pieces`, `board.rs:193-199`).
  We store `pieces: [[Bitboard; PieceType::COUNT]; Color::COUNT]` (color
  outer) and get "all queens" via an OR of both colors. Neither is more
  correct; Reckless's shape is the direct SF layout
  (`byTypeBB`/`byColorBB`), ours is transposed. Six bitboards vs. twelve,
  same information.
- `colors: [Bitboard; Color::NUM]` — matches our `color_occupancy`; no
  separate `all_occupancy` bitboard — `occupancies()` is computed on demand
  as `colors(White) | colors(Black)` (`board.rs:189-191`) rather than
  maintained incrementally. We maintain `all_occupancy` incrementally in
  `put_piece`/`remove_piece`. Reckless's choice trades one OR per query for
  one fewer bitboard to keep in sync — plausible either way; worth a note,
  not obviously a bug either direction.
- `mailbox: [Piece; Square::NUM]` — **not** `[Option<Piece>; 64]`. `Piece`
  has a 13th variant `None` (`types/piece.rs:8-25`) that doubles as the
  "empty square" sentinel, so `piece_on(sq)` returns a bare `Piece` and call
  sites compare `== Piece::None` instead of matching `Option`. See §7 for
  why this is the one place I'd flag Reckless as *less* Rust-idiomatic than
  us.
- `state: InternalState` + `state_stack: Vec<InternalState>` — see §2; the
  interesting structural difference is that the CURRENT state is a bare
  struct field, not `Vec::last()`.
- Castling geometry fixed at setup, `board.rs:51-54`:
  `castling_rights: [u8; 64]` (our `castling_rights_mask`),
  `castling_path: [Bitboard; 16]`, `castling_threat: [Bitboard; 16]`
  (occupancy-must-be-empty and attacked-must-be-empty sets, one per
  `CastlingKind` **index 0..16**, not 0..4 — see §5), `castling_rooks:
  [Square; 16]`.
- `halfmove_number: usize` replaces separate `side_to_move` +
  `fullmove_number` fields: `side_to_move()` is `Color::new(halfmove_number
  & 1)` and `fullmove_number()` is `halfmove_number / 2`
  (`board.rs:67-73`). We keep `side_to_move: Color` and `game_ply: i32`
  as two fields; Reckless folds them into one counter and derives both.
  Slightly denser, no real behavior difference (our `game_ply` IS
  `halfmove_number` under a different name).
- `frc: bool` — a live Chess960/FRC flag threaded through move printing and
  castling-rights `Display` (`board.rs:63-65`, `types/castling.rs:74-82`).
  We defer Chess960 entirely (D10); see §5.

No piece-count cache (`[u8; 16]`, the thing SF has and our design doc 04 §9
open-question 2 defers) — Reckless recomputes `self.pieces(pt).popcount()`
on the material/promotion paths in `make_move` (`makemove.rs:69,93` call
`.value()` on the already-known captured/promoted piece, and the *count*
for the material-key equivalent — except Reckless has no material-key at
all, see §3 — so this never comes up as a hot path there). Same conclusion
as our open question: not worth a cache until profiling asks for it.

## 2. State: `InternalState` + the "current state is a field" trick

`InternalState` (`board.rs:26-41`) is Reckless's `StateInfo`, but it is
**not split** into a copied/derived pair the way SF's `memcpy`-prefix or our
`CopiedState`/`StateInfo` are. It's one flat `#[derive(Copy, Clone,
Default)]` struct with every field — keys, `en_passant`, `castling`,
`fiftymove_clock`, `material`, `plies_from_null`, `repetition`,
`captured`, **and** `piece_threats`/`all_threats`/`pinned`/`pinners`/
`checkers`/`checking_squares` — all copied wholesale on every move
(`increment_stack` at `makemove.rs:5-15` does `state_stack.push(self.state)`,
a full-struct `Copy`). There's no "copy only the prefix" optimization at
all; the derived fields (threats, checkers, pins) get overwritten by
`update_threats()` a few lines later in `make_move` regardless of what was
copied. Simpler than SF's memcpy-prefix trick and than our
`CopiedState`/`StateInfo` split, at the cost of copying ~8 extra
`Bitboard`s (64 bytes) per move that are about to be discarded anyway. This
is a real "simpler, not obviously worse" data point: our split exists
because SF's memcpy semantics demanded it; in a value-copy language a flat
`Copy` struct is arguably just as clean, and the wasted copy of a few
bitboards is cheap next to the `update_threats` recompute that immediately
follows (§4).

**The distinctive part**: `Board.state: InternalState` is a plain field, and
`Board.state_stack: Vec<InternalState>` holds only the ANCESTORS, never the
current position. `make_move`'s `increment_stack` pushes the (about to be
stale) current state onto the stack, then keeps mutating `self.state`
in place into the new position (`makemove.rs:5-15,38`); `undo_move` pops the
stack back into `self.state` (`makemove.rs:144`). Every accessor
(`board.rs:79-166`, `en_passant()`, `castling()`, `checkers()`, ...) is
therefore a direct field read (`self.state.foo`), never an index or a
`Vec::last()`. We instead keep an ever-growing `states: Vec<StateInfo>`
where the current position IS the last element, so every read goes through
`self.st()` → `self.states.last().expect(...)` (`board.rs:327-333` in our
code) — one more pointer hop and (checked in debug) a length compare, on
every single state read in the hottest code in the engine. See §8 for
whether this is worth porting.

## 3. Zobrist keys: `Keys` — three scalars, not five

`types/keys.rs:4-8`:

```
struct Keys { full: u64, pawn: u64, non_pawn: [u64; 2] }
```

Reckless carries **no material key and no minor-piece key** — just `full`,
`pawn`, and `non_pawn[color]`. `Keys::toggle(piece, sq)` XORs `psq[piece][sq]`
into `full` always, and additionally into `pawn` or `non_pawn[color]`
depending on `piece.piece_type()` (`keys.rs:23-32`). Confirmed by
consumption: `search.rs:1369-1394` builds correction-history lookups purely
from `pawn_key()` and `non_pawn_key(White)`/`non_pawn_key(Black)` — there is
no material-imbalance or minor-piece correction history in this top-2
engine at all. Our design (doc 04 §3.2, following SF) carries **five** keys
(`pawn_key`, `material_key`, `minor_key`, `non_pawn_key[2]`) specifically so
those axes are available if we ever want material/minor correction
histories. Useful calibration: a ~3830-Elo engine ships with the *smaller*
key family and gets by with two correction-history axes (pawn, non-pawn per
color) — material/minor correction history is not obviously required for
top strength, though it's cheap for us to keep since we already pay for it.

`Zobrist` table build (`types/zobrist.rs:12-30`): a `const` block, SplitMix64
(explicit constants `SEED = 0xFFAA_B58C_5833_FE89`, `INCREMENT =
0x9E37_79B9_7F4A_7C15`, mixing constants `0xBF58476D1CE4E5B9` /
`0x94D049BB133111EB`), filling a flat `[u64; 865]` then `transmute`d into
the `Zobrist { pieces: [[u64;64];12], en_passant: [u64;64], castling:
[u64;16], side, fiftymove_clock: [u64;16] }` struct — same const-eval spirit
as our `zobrist::build_tables` (xorshift64\* over a `Prng`), different PRNG
and a `transmute` instead of field-by-field assignment (an unsafe most
readers would avoid; see §7). No zeroed-promotion-rank trick and no
material-count-slot trick — because there's no material key to speed up and
promotions already only touch `non_pawn`/`full`, the "avoid a pawn-key clear
XOR on promotion" saving SF/we rely on (doc 04 §3.4 trick 1) isn't needed
either, since Reckless's pawn key is *removed* via the normal capture path
(`makemove.rs:918-919`) — there is no separate "pawn leaves the promotion
rank" case to special-case away.

**Fifty-move-clock hashing — a different mechanism for the same GHI
mitigation.** Reckless's exposed `hash()` (`board.rs:79-84`) is
`self.state.keys.full() ^ ZOBRIST.fiftymove_clock[bucket]` where `bucket =
(fiftymove_clock.saturating_sub(8) / 8).min(15)` (`board.rs:75-77`) — a
16-way bucket that is **live from clock 0** (bucket 0 covers clocks 0..=15,
bucket 1 covers 16..=23, etc., 8-wide after the first). Our `fold_rule50`
(`zobrist.rs:176-182`) instead hashes **raw** for clocks 0..=13 and only
starts XORing `make_key(bucket)` once the clock reaches 14 (also 8-wide
buckets, `make_key` an LCG-style `wrapping_mul`/`wrapping_add` avalanche of
a small bucket index, not a table lookup). Same *purpose* — perturb the
TT-facing key as the clock climbs so long, drawish repeated-position
searches don't confuse the TT across different distances from the 50-move
mark — different mechanism (table XOR from move 0 vs. threshold-gated LCG
XOR starting near the danger zone). Raw keys (`state.keys.full()` /
`raw_key()`) are used for repetition detection in both engines; the folding
is TT-probe-only in both. `key_after(mv)` (`board.rs:86-106`) is Reckless's
speculative-child-key predictor, functionally identical to our planned
`prefetch_key` (doc 04 §3.10): xor side, mover psq delta, captured psq if
any, then fold with the predicted post-move clock (0 on capture/pawn move,
else `clock + 1`).

## 4. Check info + threats: one fused per-node recompute

This is the single biggest structural difference from Stockfish (and from
our own doc-04-following code). SF/we compute, per move: `checkers`,
`blockers_for_king[2]`, `pinners[2]`, `check_squares[6]` via
`slider_blockers` + `set_check_info` (SF's classic "snipers on an empty
board, then count real blockers between king and each sniper" algorithm —
stockfish_reference/02 §"Incremental check info"). Reckless's
`update_threats()` (`board.rs:431-502`) computes **all of that plus a full
board-wide "every square attacked by the side not to move" bitboard**, in
one pass, called unconditionally at the end of every `make_move` and
`make_null_move` (`makemove.rs:21,100`):

- `piece_threats[pt]` for every non-king enemy piece type, via **setwise**
  (whole-bitboard-at-once) attack generation — `pawn_attacks_setwise`,
  `knight_attacks_setwise` are pure parallel bit-shifts (Kogge-Stone-style:
  `(bb & !FILE_H).shift(9) | (bb & !FILE_A).shift(7)` for white pawn
  attacks, `setwise.rs:14-19`); `bishop_attacks_setwise`/
  `rook_attacks_setwise` fall back to a per-square loop through the magic
  tables without AVX2, but under `#[cfg(target_feature = "avx2")]` do a
  **4-lane vectorized occluded fill** — one `__m256i` carrying all four
  diagonal (or orthogonal) directions, doubling the ray length by
  self-`OR`ing with a shifted copy each iteration (`setwise.rs:48-64`,
  `shiftv::<-9,-7,7,9>`) — the whole diagonal-attack set for an arbitrary
  number of bishops/queens in ~4 SIMD ops instead of N magic lookups.
- `checking_squares[pt]` (our `check_squares`) for the SIDE NOT TO MOVE's
  king, same idea as SF.
- `checkers`/`pinned`/`pinners` via a single combined loop over both colors
  (`board.rs:469-501`): the "snipers" step uses
  `bishop_attacks(king, colors(!color)) & colors(!color) & diagonal_sliders`
  — i.e. slider attacks computed with **only the enemy occupancy as
  blockers** (friendly pieces transparent), restricted to enemy diagonal/
  orthogonal piece types. This finds exactly the same snipers as SF's
  "attack on an empty board" trick, but *without* SF's extra `occ =
  all_occupancy ^ snipers` correction step for the case of two enemy
  sliders on the same ray — because using real enemy occupancy as the
  blocker set means a *farther* enemy slider is already blocked by a
  *nearer* one of the same color and simply never enters the sniper set to
  begin with. One fewer special case for the same result; worth adopting
  in prose even if we keep SF's literal formulation (design/04 §3.5
  already documents SF's `all_occupancy ^ snipers` step precisely because
  of this edge case — Reckless shows it's avoidable by construction).
- The king itself is excluded from `occupancies` while building threats
  (`board.rs:432-441`, explicit comment) so that sliding pieces "see
  through" the king's own square — the standard "can the king step along
  the checking ray" fix, done once for the whole threat map instead of
  per-candidate-move.

**Why pay for a full attack map every node?** `is_legal` (`board.rs:349-421`)
never calls `attackers_to` for king moves — it tests
`(king_attacks(from) & !self.all_threats()).contains(to)` (`board.rs:376`),
a single bitboard AND, because every enemy-attacked square is already known.
SF's `legal()`, by contrast, queries `attackers_to(to, occupied ^ ksq)`
per candidate king destination. And `all_threats()`/`piece_threats(pt)` feed
move-ordering history directly — see §6. So the setwise recompute isn't
just check-info bookkeeping; it's replacing N per-destination attack queries
(movegen legality) and feeding a whole new history dimension (move
ordering) with one node-entry cost. Our design keeps SF's
on-demand-`attackers_to` shape (design/05-movegen.md, `Board::legal`
lives in movegen next to the generator, queries per move); this is the
clearest "different strategy, not a bug" finding in the whole note.

## 5. `make_move`/`undo_move`: order of operations

`makemove.rs:32-122`. Close to SF/our order but flattened (no
copied/derived split to respect) and reordered where that split doesn't
constrain it:

1. `increment_stack()`: bump `halfmove_number`, push current `state` onto
   `state_stack`, toggle side key, clear `repetition`, clear a stale ep
   square (and its key) — i.e. steps 1+2 of SF's `do_move` and the ep-clear
   of SF's step 5, done up front instead of after the mover XOR.
2. Snapshot `captured = piece_on(to)` (before any mutation) and bump
   `plies_from_null`.
3. Fiftymove clock: 0 on any `Capture` kind or pawn move, else `+1`
   (`makemove.rs:44-48`) — note this test runs BEFORE looking at
   en-passant/castling specially, because `mv.kind() == Capture` already
   covers normal captures and the pawn-piece-type test covers quiet pawn
   moves, double pushes, promotions, AND en passant (en passant's `MoveKind`
   is a distinct `EnPassant` value, not `Capture`, but the mover is always
   a pawn so the `piece.piece_type() == Pawn` arm still fires).
4. **Castling** (`makemove.rs:50-60`): remove rook, remove king,
   place king, place rook — always remove-both-then-place-both, so
   overlapping FRC king/rook start squares are handled with no special
   case, exactly the SF/our-doc-04-§3.8 pattern. Rook's key delta computed
   inline and XORed into `k` and `non_pawn_key[us]`.
5. **Capture** branch (`makemove.rs:61-84`, mutually exclusive with
   castling): remove mover from `from`, remove captured from `to`, place
   mover at `to`; subtract captured's value from `material`. Plain quiet
   moves and en-passant share the `else` arm: move mover, then if
   `is_en_passant()` remove the piece at `to ^ 8` (identical little-endian
   rank-file XOR trick to ours — same square-index convention, see
   CLAUDE.md), else if `is_double_push()` set `en_passant = to ^ 8` and XOR
   its key — **unconditionally**, no legality pre-check.
6. **Promotion** (`makemove.rs:86-94`): remove pawn from `to`, place the
   promoted piece, adjust `material` by `promo.value() - Pawn.value()`.
7. **Castling rights**: `castling.raw &= castling_rights[from] &
   castling_rights[to]` (`makemove.rs:97`) — an AND-of-masks instead of our
   (and SF's) "OR the touched bits, then subtract" — same result, phrased
   as "keep only rights whose mask bit survives at both squares" rather
   than "revoke the touched bits." Key XORed around it exactly like ours
   (old castling key out, new one in).
8. `update_threats()` + `validate_en_passant()` (`makemove.rs:100-101`) —
   the full recompute from §4, called for EVERY move (not gated on whether
   check/pin info could have changed), followed by an **unconditional**
   post-hoc en-passant legality check (`board.rs:526-554`): try every
   candidate capturing pawn, check whether removing it + the captured pawn
   exposes the king to a rook/bishop/queen; only keep the ep square (and
   its key) if at least one capture is truly legal. This is the same D9
   ep-validation contract we already enforce, but done in `make_move`
   *after* setting the ep square provisionally, rather than SF/our approach
   of testing legality once during `make_move`, using the PARENT state's
   blockers, before ever writing the key (design/04 §3.8 step 8). Both are
   correct; ours does the exact-and-cheap test with information already in
   hand (previous blockers) and avoids ever hashing a since-retracted ep
   key, Reckless re-derives it via a small loop over candidate pawns using
   the NEW threats/board state and un-hashes the key if it turns out
   illegal. Worth a note only because it means Reckless can (rarely) XOR an
   ep key in and back out again within the same `make_move` call — never
   externally observable, just a few wasted cycles on the moves where a
   double push's ep square isn't actually capturable.
9. Repetition scan (`makemove.rs:103-121`): identical algorithm and
   encoding to ours/SF — window `min(plies_from_null, fiftymove_clock)`,
   step by 2 plies, `+i`/`-i` first-repeat/at-least-threefold encoding.

`undo_move` (`makemove.rs:124-145`): "dumb-cheap," same philosophy as SF/us
— no key or check-info recomputation, just place pieces back (castling:
remove both, place both; promotion: remove promoted piece, place a pawn;
capture: place the stored `captured_piece()` back at `capture_sq()`), then
`self.state = self.state_stack.pop().unwrap()` restores everything else
(including threats/checkers/pins) in one struct assignment. Because
`InternalState` isn't split, there's no separate "restore the copied prefix,
recompute nothing else" — the whole state, derived fields included, comes
back from the stack in one move. Our `unmake_move` is the same shape
(`board.rs:1092-1135`) modulo the copied/derived split.

`make_null_move`/`undo_null_move` (`makemove.rs:17-27`): `increment_stack`
(full state copy + side/ep key work), zero `plies_from_null`, clear
`captured`, `update_threats()` again — i.e. Reckless pays the full setwise
recompute on a null move too, where SF/we can skip re-deriving
`checkers`/`blockers`/`pinners` because a null move by definition leaves
them trivially known (the side who didn't move can't newly check itself).
Given Reckless's `update_threats` also rebuilds the whole-board attacked-
squares map (which DOES change identity of "the moving side" even though no
piece moved), the full recompute may be necessary for them where it isn't
for us — but it's the same per-node cost as a real move, so `make_null_move`
is not cheaper than `make_move` in Reckless, unlike in SF/our design.

## 6. Threats feed move-ordering history directly

Not board-module code, but the entire reason `update_threats` exists at the
cost it does: `all_threats()`/`piece_threats(pt)` are read by `movepick.rs`
(lines 129-159, building "is this square currently attacked" bucket
indices) and `search.rs` (quiet-history and noisy-history `get`/`update`
calls all take `td.board.all_threats()` as an extra key, e.g.
`search.rs:768-771,1081-1102`). This is a **threat-conditioned butterfly/
noisy history** — the same `[piece][to]`-style tables SF/we already have,
but bucketed additionally by whether relevant squares are attacked in the
CURRENT position, not just by the move itself. Our doc 09 §3.4 table
catalogue (butterfly, low-ply, capture, continuation, pawn, correction) has
no threat axis at all. See §8.

## 7. Rust-style notes

- **`Piece::None` sentinel vs. `Option<Piece>`.** Reckless folds "no piece"
  into a 13th `Piece` discriminant (`types/piece.rs:8-25`) so `mailbox` is
  `[Piece; 64]` and call sites write `if piece != Piece::None`. We keep
  `mailbox: [Option<Piece>; 64]` (CLAUDE.md's explicit convention: "expose
  behavior, not representation"). `Option<Piece>` gets Rust's niche
  optimization for free here (12 used discriminants leave spare byte
  patterns), so there is no size cost to being idiomatic — this is a case
  where Reckless reproduces a C++-style sentinel pattern in Rust for no
  compensating benefit; ours is the more idiomatic choice CLAUDE.md already
  mandates.
- **Liberal `unsafe`/`transmute` vs. our near-zero-unsafe budget.**
  `Piece::new`/`Piece::from_index` (`types/piece.rs:49-57`), `Move::kind()`
  (`types/moves.rs:52-54`), the Zobrist table build
  (`types/zobrist.rs:29`), and the cuckoo tables (`lookup.rs:5-7`, `static
  mut` globals) all reach for `unsafe { transmute }` or `unsafe { static
  mut }` where a match/array-index would do the same job safely. 23 of
  Reckless's source files touch `unsafe`, 25 `transmute` call sites
  project-wide. Our D14 budget is intentionally narrow (prefetch
  intrinsics, SIMD in NNUE/eval, profiler-justified unchecked indexing) and
  our `board.rs`/`types.rs`/`moves.rs` currently contain **zero** `unsafe`.
  Not a criticism of Reckless's approach (it's a legitimate, common
  Rust-chess-engine style — "we know the numeric ranges, transmute is
  faster to write than a match"), but confirms our stricter budget is a
  deliberate stylistic choice, not something the reference engine's
  performance depends on.
- **Move encoding: capture is DATA, not derived.** Reckless's `MoveKind`
  (`types/moves.rs:18-35`) is the classic chess-programming-wiki 4-bit
  scheme where bit 2 (`0b0100`) means "capture" and bit 3 (`0b1000`) means
  "promotion" directly, so `is_capture()`/`is_promotion()` are single bit
  tests on the raw `u16` (`types/moves.rs:78-84`) — no board access needed,
  and en passant (`0b0101`) is a "capture" by this flag even though the
  destination square is empty. Our `Move` (D2, SF-style, `moves.rs:1-99`)
  encodes only from/to/promotion-piece/kind (Normal/Promotion/EnPassant/
  Castling) and `is_capture` is a **board query**
  (`Board::is_capture(m)`/`capture_stage`, living in movegen next to the
  generator) — matches SF exactly, and is pinned by D2. Not a defect (a
  board reference is always in hand at every call site that currently
  cares), just worth knowing the two schemes trade "self-describing move,
  4 extra encoding bits' worth of design space used up" against "move stays
  minimal, classification always costs a board lookup." D2 is pinned; this
  is recorded for context, not as a change proposal.
- **FEN parsing strictness.** Reckless's `from_fen` (`parser.rs:24-66`)
  explicitly documents itself as lenient ("not very strict... the GUI's
  responsibility") and backs that with `unwrap_or_default()` on ep square,
  fiftymove clock, and fullmove number — a malformed clock silently becomes
  0 rather than erroring. It does return `Result<Self, ParseFenError>`, but
  the error enum only has three variants (`MissingPlacementData`,
  `InvalidPieceType`, `InvalidActiveColor`) — most malformed input is
  silently absorbed. Our `FenError` (`fen.rs:11-27`) is deliberately
  stricter and richer (`FieldCount`, `RankCount`, `InvalidPiece`,
  `RankWidth{rank,width}`, `SideToMove`, `Castling`, `EnPassant`, `Clock`,
  each with a `Display` message) and matches CLAUDE.md's "fallible parsing
  returns `Result`, never panics" convention more thoroughly. This is one
  place our engine is already ahead — no action item, just confirmation.

## 8. Castling geometry + Chess960/FRC — Reckless ships it, we've deferred it

Reckless's castling is Chess960-general from the ground up (`frc: bool`
flag threaded through `Display`/UCI move printing, `types/castling.rs:74-87`)
while ours is explicitly "Chess960-ready SHAPE, standard chess only" (D10).
The concrete algorithm worth banking for when D10 gets revisited:

- `Board::set_castling` (`parser.rs:68-97`) parses BOTH classic `KQkq` and
  Shredder-FEN file-letter (`A`-`H`) notations with one code path: for a
  file-letter right, `search_step = letter - 'A' - king_from.file()` (the
  signed direction from the king toward that file); for the classic `K`/`Q`
  letters, `search_step` is just overwritten to a fixed `RIGHT`/`LEFT`
  direction. Either way, the ACTUAL rook square is then found by scanning:
  `ray_pass(king_from, king_from.shift(search_step)) &
  colored_pieces(color, Rook)`, taking the nearest hit (`.lsb()`)
  (`parser.rs:85-86`). One function handles standard chess, Shredder-FEN,
  and arbitrary Chess960 starts uniformly — worth reusing the *shape* of
  this even for standard-only support now, since it removes the hardcoded
  "rook always starts on A1/H1/A8/H8" assumption our current `fen.rs`
  makes (`fen.rs:121-130`: fixed `Square::H1`/`A1`/`H8`/`A8` per right).
- `castling_path`/`castling_threat` are indexed by the **whole 4-bit right
  mask 0..16** in Reckless (`board.rs:52-53`), not just the 4 single-right
  slots 0..4 our doc 04 §3.7 uses — meaning Reckless can look up "path for
  both of white's rights combined" in one table read where we'd union two
  single-right bitboards. A minor indexing convenience, not a correctness
  difference (`castling_path[Castling]` uses `Index<Castling>`,
  `types/castling.rs:90-96`, indexing by the raw rights byte).

## 9. Repetition, draws, `upcoming_repetition` (cuckoo) — matches our deferred plan closely

`is_draw`/`draw_by_repetition`/`has_repeated`/`draw_by_fifty_move_rule`
(`board.rs:269-290`) are essentially line-for-line the same algorithm as our
design doc 04 §3.9 and SF: `repetition` field with the `+d`/`-d`
first-repeat/at-least-threefold encoding, `min(rule50, plies_from_null)`
scan window, a stalemate/checkmate guard on the 100-halfmove claim
(`self.checkers().is_empty() || has_legal_moves()`, matching our own guard
exactly). `draw_by_material()` (`board.rs:242-267`) is a KPK/insufficient-
material-family check keyed on piece counts and light/dark-square bishop
color — a heuristic our design doesn't currently specify at the board
level (worth checking whether it belongs in board or eval for us; Reckless
puts it on `Board` since it only reads piece bitboards/counts).

`upcoming_repetition` (`board.rs:298-338`) is a **working implementation**
of exactly the van Kervinck cuckoo scheme our design doc 04 §3.9 already
specifies as "specified now, phase-tagged LATER." Confirms the plan is
sound and gives a second real data point for the hash-slot constants:
Reckless's `h1(h) = (h >> 32) & 0x1FFF`, `h2(h) = (h >> 48) & 0x1FFF`
(`lookup.rs:49-55`) — slicing the TOP 13 bits of each half — versus SF's
(and our documented) `H1(k) = k & 0x1FFF`, `H2(k) = (k >> 16) & 0x1FFF` —
the bottom bits. Both are just hash functions over a 64-bit avalanche-mixed
key; either bit range works, but it's a concrete reminder that the exact
slice is arbitrary and doesn't need to match SF bit-for-bit when we
implement it (only internal consistency between fill and query matters).
Table size matches: `CUCKOO/A/B: [_; 0x2000]` = 8192 slots
(`lookup.rs:5-7`), same as our planned `cuckoo: [Key; 8192]`. Fill algorithm
is the same cuckoo-displacement insert loop (swap into `H1`, evictee goes
to its alternate slot, repeat until a `None` square pair lands)
(`lookup.rs:15-47`).

## 10. NNUE hook shape (context for our future P13 seam)

Not this note's core topic, but directly relevant to design/04 §3.10's "the
same seam later carries NNUE dirty-piece recording (D21)": Reckless's
`BoardObserver` trait (`board.rs:597-609`) is the concrete shape a shipped
top engine uses for this. Three primitive events, called inline at every
placement mutation in `make_move` (`makemove.rs:52-91`):

- `on_piece_change(board, piece, sq, add: bool)` — a piece appeared or
  vanished at `sq` (used for the rook half of castling, and the mover's
  `from`-square vacate).
- `on_piece_move(board, piece, from, to)` — a plain slide (quiet moves, and
  the mover's king/piece leg of captures... actually captures fire
  `on_piece_change`+`on_piece_mutate`, not `on_piece_move` — the mover's
  `from`-vacate is `on_piece_change(piece, from, false)` and the `to`-square
  swap is `on_piece_mutate`; `on_piece_move` is reserved for the no-capture,
  no-promotion case).
- `on_piece_mutate(board, old_piece, new_piece, sq)` — the occupant of `sq`
  CHANGED IDENTITY (captured piece replaced by mover; pawn replaced by
  promoted piece) — exactly the granularity an NNUE accumulator wants
  (a single "sub old feature, add new feature" call instead of a naive
  remove-then-add pair), confirmed by the one real implementor,
  `nnue.rs:342-354` (`impl BoardObserver for Network`). Our current
  `MakeMoveHooks` trait only exposes hash-key hook points (`on_key`/
  `on_state`, design/04 §3.10) — no piece-level events yet, since NNUE is
  P13. When we get there, this three-event shape (add/remove, slide,
  same-square-mutate) is a good concrete reference for shaping the
  eventual dirty-piece hook rather than inventing the SubAdd/SubSubAddAdd
  taxonomy from scratch.

## Adapt to us

1. **Cache the current state as a plain field; keep only ancestors in the
   Vec.** `design/04-position-state.md` §3.2/§3.3 (D8). Every hot-path state
   read (`self.st()`) currently pays a `Vec::last().expect(...)` indirection;
   Reckless's `self.state: InternalState` field + `state_stack: Vec<..>` of
   ancestors turns every read into a direct field access. Behaviorally
   equivalent to D8's "ply-indexed `Vec<StateInfo>` owned by `Board`," but
   the literal storage shape changes (current state duplicated out of the
   Vec), so this would want a one-line D8 wording amendment alongside the
   code change, not a silent refactor. **Effort: small (mechanical
   refactor of `st()`/`st_mut()` and the push/pop sites in
   `make_move_with`/`unmake_move`). Impact: unknown until profiled** —
   worth a quick perf-counter check (or just diffing the bench fingerprint
   node/sec, not node count, since this is a pure speed change) before
   committing to it; if `self.states.last()` already compiles down to a
   branch-free load (plausible — LLVM can often prove the Vec is
   non-empty right after a push), there's nothing to gain.

2. **Port `update_threats`'s "friendly-transparent, enemy-real" sniper
   trick into our documented `slider_blockers` prose.** `design/04-
   position-state.md` §3.5. Reckless computes snipers as
   `bishop_attacks(king, colors(enemy)) & colors(enemy) & diagonal_sliders`
   (occupancy = enemy-only, so enemy pieces correctly block each other)
   instead of SF's "attack on an EMPTY board, then separately subtract
   snipers from occupancy to avoid one sniper masking another." Same
   result, one fewer bitboard op, no separate correction step needed. Purely
   a documentation/implementation-detail change to `slider_blockers` (no
   behavior change, no bench impact expected beyond a possibly-faster
   `checkers`/`pinners` computation). **Effort: small. Impact: small
   (removes one bitboard AND per call, simplifies the algorithm SF/we
   currently document with the extra correction step).**

3. **Consider a threat-conditioned quiet/noisy history axis.**
   `design/09-move-ordering-history.md` §3.4 (table catalogue) / §3.9
   (update functions), consuming a new board-level "all squares attacked by
   the side not to move" bitboard that would live in `design/04-position-
   state.md` §3.5 next to `checkers`/`blockers_for_king`. This is new scope
   beyond what either doc currently specifies (would need a decisions.md-
   or doc-level discussion before implementation, per the suite's process
   rules, not just a code change) — flagging it because a top-2 engine
   spends a real per-node cost (full setwise attack-map recompute, §4) to
   get this, and the current P9 pruning-wave / move-ordering work is the
   natural place to evaluate it. If our own `attackers_to`-based per-move
   legality (design/05) stays as-is, the threats bitboard would need to be
   computed SEPARATELY for history purposes (not free the way it is for
   Reckless, which uses it for legality too) — that changes the cost/benefit
   math materially, so this should be SPRT-tested as its own heuristic
   (doc 14 §9.4 fixed-node proxy) rather than assumed to pay for itself.
   **Effort: medium (new board-level bitboard, new history table axis,
   design doc update). Impact: uncertain — untested on our engine, and the
   "is_legal gets it for free" argument that makes it cheap for Reckless
   doesn't automatically apply to us.**

4. **Reuse the FEN castling-rook-detection algorithm's SHAPE when D10
   (Chess960) is revisited.** `design/04-position-state.md` §3.7,
   `design/05-movegen.md` (deferred Chess960 notes), D10. Reckless's
   single scan-in-a-direction-from-the-king routine
   (`parser.rs:68-97`) handles classic `KQkq`, Shredder-FEN, and arbitrary
   Chess960 starts with one code path, rather than our current hardcoded
   `H1`/`A1`/`H8`/`A8` assumption (`fen.rs:121-130`). Not urgent (D10
   explicitly defers Chess960), but worth keeping as the reference shape
   for that future doc revision rather than re-deriving it from scratch.
   **Effort: small when D10 is revisited. Impact: N/A until then (no
   strength effect, pure feature completeness).**

5. **NNUE dirty-piece hook: three-event shape, not five.** `design/04-
   position-state.md` §3.10 (seam) and `design/11-evaluation.md` §3.4
   (Phase C, incremental accumulators), roadmap P13. When we extend
   `MakeMoveHooks` with piece-level events for NNUE, Reckless's
   `BoardObserver` (`on_piece_change`/`on_piece_move`/`on_piece_mutate`,
   `board.rs:597-609`) is a clean, already-proven-in-production reference
   for the event granularity (add/remove with a bool flag, slide, and
   same-square-identity-change) that maps directly onto accumulator
   sub/add operations. **Effort: N/A now (P13 is far out), but worth
   citing in doc 04/11 as prior art when that seam is actually designed.
   Impact: potentially saves a design iteration by starting from a
   validated event shape instead of inventing one.**

6. **Non-action confirmations worth recording in doc 04's open questions.**
   No material key / no minor key is sufficient for a top-2 engine (§3) —
   mild evidence our five-key family is "ahead of need," not wrong; keep it
   since it's already paid for, but don't feel obligated to build
   material/minor correction histories just because the fields exist. Our
   stricter `FenError` and near-zero `unsafe` budget are already the more
   idiomatic/robust choice versus this reference (§7) — no change needed,
   just confirmation the design docs' conventions are paying off relative
   to a real competitor.
