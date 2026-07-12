# Move generation & SEE (Reckless)

Source: `src/board/movegen.rs`, `src/board/see.rs`, `src/board/tests.rs`,
plus what they lean on: `src/board.rs` (threats/pins/checkers state,
`is_legal`, `attackers_to`), `src/setwise.rs` (vectorized setwise attack
fills), `src/types/{movelist,moves,piece,castling}.rs`, and the two call
sites that drive generation/SEE at runtime, `src/movepick.rs` and
`src/search.rs`. Reckless is AGPLv3 — everything below is paraphrase +
my-own-words pseudocode; no source is reproduced beyond the handful of
single-line formulas called out explicitly.

## 0. The one big architectural fact: full threat/pin/check state is
recomputed once per node, not queried per candidate move

This is the thing that shapes everything else in this note, so it goes
first. `Board::state` (`board.rs:42-56`) carries, alongside the copyable
`InternalState`:

- `piece_threats: [Bitboard; PieceType::NUM]` — for **every** piece type,
  the union of all squares that type (of the non-side-to-move) attacks
- `all_threats: Bitboard` — the union of the six `piece_threats`
- `pinned: [Bitboard; 2]`, `pinners: [Bitboard; 2]` — **both colors**, not
  just the side to move
- `checkers: Bitboard`, `checking_squares: [Bitboard; PieceType::NUM]` — the
  latter is "if a piece of this type stood here, would it check the enemy
  king", used for a cheap `is_direct_check` probe

All of this is thrown away and recomputed from scratch by
`Board::update_threats` (`board.rs:431-501`) once per `make_move` /
`make_null_move` (`board/makemove.rs:21,100`) — not incrementally maintained
through the move, a full recompute every ply. The per-piece-type threat
bitboards are built with **setwise** attack generators (`setwise.rs`):
`pawn_attacks_setwise`/`knight_attacks_setwise` are closed-form
shift-and-mask formulas over the whole pawn/knight bitboard at once, and
`bishop_attacks_setwise`/`rook_attacks_setwise` use a Kogge-Stone-style
generate/propagate sliding fill (`generate |= propagate & shift(generate)`,
doubling the shift each round) with an AVX2 4-lane vectorized variant
(`setwise.rs:52-104`, one lane per ray direction, `_mm256_sllv_epi64` +
`_mm256_srlv_epi64` to shift both ways in one op) and a scalar
per-square-`bishop_attacks`-loop fallback when AVX2 isn't available. The
same function also derives `pinned`/`pinners`/`checkers` for **both** kings
in one pass: for each color it takes the enemy diagonal/orthogonal sliders
that attack the king through enemy-only occupancy, and for each such slider
counts blockers on the `between` ray — zero blockers means it's a checker
(only recorded when `color == stm`, an unpinned discovered-check pattern
can't exist for the side not to move), one blocker means a pin, two+ means
nothing.

The payoff: **king-move legality during generation is a single mask AND
against a precomputed bitboard**, not a per-candidate-square attacker
search. `generate_moves` (`movegen.rs:47`) does
`king_attacks(king_sq) & !self.all_threats() & kind_target` directly — no
loop, no `is_square_attacked` call per candidate destination.
`collect_castling` (`movegen.rs:93-104`) checks the castling-threat squares
the same way (`self.castling_threat[kind] & self.all_threats()`).
`Board::is_legal` (`board.rs:349-414`, see §4) reads the same bitboards to
answer full legality in O(1) for every move kind, including castling and en
passant. And `is_direct_check` (`board.rs:427-429`) answers "does this move
give check" by testing `checking_squares[piece_type].contains(to)` — no
`gives_check` computation at all, just a lookup (the doc comment admits
"roughly 90–95% accurate... does not account for discovered checks,
promotions, en passant, or checks delivered via castling" — a deliberately
approximate O(1) proxy used only for move-ordering heuristics like LMP
gating, `search.rs:1297`).

**Consequence for the generator itself: `generate_all_moves` emits fully
legal moves in one pass**, with the sole exception of pinned
non-king/non-castling pieces, which the generator restricts inline (see
§2) rather than filtering after the fact. There is no separate
pseudo-legal→legal filter step anywhere in the hot path — contrast this
with both Stockfish and our engine, where movegen produces pseudo-legal
moves and a *lazy* legality check (`Board::legal`/`pos.legal()`) is run
only on the move actually about to be searched
(`stockfish_reference/03-movegen-movepick-history.md` "Pseudo-legal + lazy
legality"; our `generate_legal`/`Board::legal` in
`src/movegen.rs:92-155,475-539`). Reckless can afford full legality at
generation time cheaply *because* it already paid for `all_threats`/
`pinned`/`checkers` once per node regardless of movegen; we (and SF) never
compute a full king-danger bitboard, only `checkers` + `blockers_for_king`,
so filtering king moves needs a live attacker query per candidate instead.

## 1. Move generation (`movegen.rs`)

Two-way split, not SF's four-way `GenType`: `MovegenKind::Quiet` /
`MovegenKind::Noisy` (`movegen.rs:8-12`). `append_all_moves` just calls
`append_noisy_moves` then `append_quiet_moves` (`movegen.rs:27-38`) — noisy
first, matching the staged MovePicker's consumption order. The
noisy/quiet split follows the same convention as SF and us: captures +
non-capturing **queen** promotions are noisy; non-captures + non-capturing
**under**-promotions are quiet (`movegen.rs:135-141` puts `PromotionR/B/N`
in the quiet branch, `movegen.rs:140-141` puts `PromotionQ` in the noisy
branch).

`generate_moves` (`movegen.rs:40-76`) pseudocode, in my own words:

```
fn generate_moves(list, kind):
    kind_target = if Quiet { empty squares } else { enemy occupancy }
    move_kind   = if Quiet { Normal } else { Capture }

    push king moves: king_attacks(king_sq) & !all_threats & kind_target
    if in double check: return   # only king moves are ever legal

    target = if in_check { between(king, checker) | checker } else { ALL }
    pinned = self.pinned(stm)

    collect_pawn_moves(target, pinned, kind)     # target NOT yet &= kind_target here!
    target &= kind_target

    for each of knight/bishop/rook/queen:
        collect(target, pieces, move_kind, pinned, attacks_fn)

    if Quiet: collect_castling()
```

Two things worth flagging:

- **The pawn call happens before `target &= kind_target`** — the code
  comment literally says `//broken noisy/quiet boundary`
  (`movegen.rs:57`). `collect_pawn_moves` has to redo its own
  capture/quiet split internally (it re-derives `target & self.colors(!stm)`
  for the noisy branch at `movegen.rs:143`) because the shared `target`
  it's handed hasn't been intersected with `kind_target` yet — only the
  check-evasion restriction has been applied at that point. It's a
  self-acknowledged wart in an otherwise clean function, worth remembering
  as a "even a ~3800-rated engine ships an ugly seam here" data point.
- **One generic `collect` closure services knights, bishops, rooks, and
  queens** (`movegen.rs:78-91`): unpinned pieces of that type get
  `attacks(from) & target` directly; pinned pieces of that type get
  `attacks(from) & target & ray_pass(king_sq, from)` — `ray_pass` is the
  full two-directional line through both squares (not just the segment
  between them), so this is exactly "restrict to the pin ray" expressed as
  one extra AND, no per-piece-type duplication. Knights get the same
  treatment via the *same* closure but with an all-or-nothing pin mask,
  since `ray_pass` restricted to a knight's L-shaped attack set is always
  empty — the pinned branch naturally yields zero moves for a pinned
  knight without a special case.

Pawn handling (`collect_pawn_moves`, `movegen.rs:118-154`) has two cute
micro-optimizations for the pin case that avoid a per-pawn `ray_pass` call
entirely:

- **Pushes**: a pinned pawn is only allowed into the push set if it sits on
  the *same file as the king*
  (`pawns & (!pinned | Bitboard::file(king_sq.file()))`, `movegen.rs:126`).
  This is a sound shortcut, not an approximation: if a pawn is pinned *and*
  shares a file with its king, the pin ray connecting them can only be
  vertical (a diagonal or horizontal ray can't pass through two squares on
  the same file unless they coincide), so any push — which by definition
  stays on that same file — cannot leave the ray. No need to compute or
  compare the actual pin direction.
- **Diagonal captures**: the two possible capture directions
  (up-left/up-right) each get a *fixed* pin mask computed once per call,
  outside any per-pawn loop —
  `relative_diagonal(stm, king_sq)` and `relative_diagonal(!stm, king_sq)`
  (`movegen.rs:146`) — because a king square has exactly two diagonals
  through it, and a pawn's capture direction must match one of them to stay
  on a diagonal pin ray. Pawns pinned off that specific diagonal are masked
  out per direction (`movegen.rs:150`) instead of calling `ray_pass`
  per-pawn.
- **En passant** (`collect_pawn_captures`, `movegen.rs:106-116`) gets *no*
  pin-ray filtering at generation time at all — it's validated after the
  fact, once per node, by `Board::validate_en_passant`
  (`board.rs:528-554`), which is called right after `update_threats` in
  `make_move` (`board/makemove.rs:100-101`) and simply clears the ep square
  (and its Zobrist key) if any of stm's pawns capturing it would expose a
  horizontal/discovered check on the king. This moves the classically
  fiddly "two pawns leave the rank at once" ep-pin case out of the
  generator and into a single post-move sweep, at the cost of doing it on
  *every* move with a live ep square regardless of whether a capture is
  even generated this ply.

Castling (`collect_castling`, `movegen.rs:93-104`): fully Chess960/FRC-ready
today (unlike our engine, which has deliberately deferred Chess960 support
per D10 — see design/00-master-plan.md and design/04 §"Chess960-ready
shape"). `castling_path`/`castling_threat`/`castling_rooks` are
per-`CastlingKind` arrays (`board.rs:52-54`) fixed at setup; a castle is
legal iff the path is empty, none of the threat squares are in
`all_threats()`, and the rook itself isn't pinned
(`!self.pinned(stm).contains(self.castling_rooks[kind])`). `board/tests.rs`
carries three Chess960 perft fixtures and a Shredder-FEN round-trip test
(`tests.rs:49-74`) exercising this.

## 2. `Move` and `MoveList` encoding

`Move` is a `u16` like ours, but the 16 bits are split from(0-5)/to(6-11)/
**kind(12-15)**, a full 4-bit `MoveKind` enum with 12 named variants
(Normal, DoublePush, Castling, Capture, EnPassant, then the eight
promotion×capture combinations) rather than our (and SF's) 2-bit kind +
2-bit promotion-piece split (`types/moves.rs:18-35`). This buys some neat
one-instruction predicates directly off the raw bits: `is_capture()` is bit
14, `is_promotion()` is bit 15 (`types/moves.rs:78-84`) — no match needed —
and `is_noisy()` is a single comparison, `(kind & 7) > Castling as u8`
(`types/moves.rs:70-72`, exploiting that every noisy `MoveKind` numerically
sorts above `Castling` once the top promotion/capture bits are masked out).
The cost is a `MoveKind` with 12 variants instead of 4×2, and
`promo_piece_type()` has to reconstruct the piece from `kind & 3`
(`types/moves.rs:102-105`) instead of reading a dedicated 2-bit field.

`MoveList` (`types/movelist.rs`) stores `MoveEntry { mv: Move, score: i32 }`
— **always**, even during raw perft generation where the score is dead
weight (`push` sets `score: 0` unconditionally, `movelist.rs:29-31`).
Contrast SF and us: `MoveList`/our list stores bare 2-byte `Move`s, and only
the MovePicker's own scoring buffer pairs them with a score
(`stockfish_reference/03-movegen-movepick-history.md` "ExtMove... Scoring
happens later in MovePicker's own buffer"; our `src/moves.rs:135-151`
derefs to `[Move]`). Reckless trades a little memory density (256×8 bytes
vs 256×2) for one unified list type everywhere instead of a
lean-list/scored-buffer two-tier system. `push_setwise`/
`push_pawns_setwise` additionally carry an AVX-512VBMI2 SIMD path
(`movelist.rs:40-67,76-106`) that builds up to 32 `Move`s per instruction
via a broadcast-template-plus-OR trick and a byte-permute "splat" primitive,
falling back to a plain per-bit loop otherwise — a level of
platform-specific micro-optimization neither Stockfish nor we currently
attempt (our target-cpu baseline is `x86-64-v3`/AVX2, D16 — AVX-512 isn't
in scope).

## 3. TT-move legality (`Board::is_legal`, `board.rs:349-414`)

Used exactly once at runtime: `MovePicker::next` validates the hash move
with `td.board.is_legal(self.tt_move)` before ever generating anything
(`movepick.rs:49`) — the same role our `Board::pseudo_legal` +
`Board::legal` pair plays for the TT move. The interesting difference: this
IS full legality, computed from scratch per move without calling the
generator at all, branching on move kind and reading straight off
`pinned`/`checkers`/`all_threats`/`checking`-style bitboards (e.g. a
non-special, non-king move is legal iff `!pinned(stm).contains(from) ||
ray_pass(king, from).contains(to)`, `board.rs:379-386`; a king step is
legal iff `king_attacks(from).contains(to) && !all_threats().contains(to)`
combined with a same-color/capture-consistency check, `board.rs:373-376`).
Our `pseudo_legal` takes a different shortcut for the fiddly kinds — for
anything other than `Normal` it literally calls `generate::<GT>` and checks
list-containment (`src/movegen.rs:558-566`, "inherits every rule... with
zero duplicated logic") — i.e. we lean on "regenerate and compare" where
Reckless leans on "recompute the legality rules directly." Both are
defensible; ours trades a full regeneration (rare path, only hit for
promotion/ep/castling TT moves) for zero logic duplication, Reckless's
avoids the regeneration but must keep `is_legal`'s hand-rolled rules in
sync with the generator's by hand.

## 4. Check evasions

Same top-level shape as SF and us: `checkers().is_multiple()` (Reckless
calls it that; we call it `more_than_one`) means only king moves are legal,
full stop (`movegen.rs:49-51`); single check restricts the target to
`between(king, checker) | checkers` (checker square included by convention
— same as our doc 05/06 `between` semantics, `movegen.rs:53-54`). The
structural difference is that Reckless does **not** have a dedicated
`EVASIONS` `GenType` the way SF/we do (4 enumerated
CAPTURES/QUIETS/EVASIONS/NON_EVASIONS combinations, one monomorphized
routine each). Instead it factors the "am I in check" restriction and the
"captures vs quiets" restriction as two *independent* masks
(`target` for check-safety, `kind_target` for capture/quiet), ANDed
together inline in the one `generate_moves` function
(`movegen.rs:53-59`). This is a genuinely different decomposition: 2
orthogonal concerns multiplied together instead of 4 enumerated cases — a
plausibly more DRY structure, undercut in practice by the very
pawn-ordering wart noted in §1 (the two masks aren't actually applied at
the same point for pawns).

## 5. SEE (`board/see.rs`)

Signature: `see(&self, mv: Move, threshold: i32) -> bool`. Piece values
(`types/piece.rs:138-147`): **P 109, N 403, B 435, R 679, Q 1242, K 0** — a
compact, apparently independently-tuned scale, not Stockfish's classic
values. Compare our inherited-from-SF D20 scale, P 208 / N 781 / B 825 /
R 1276 / Q 2538 / K 0 (`src/types.rs:636-654`): the minor/rook ratios are
close (N/P ≈ 3.7 both, B/P ≈ 4.0 both, R/P ≈ 6.2–6.3 both) but Q/P differs
more (11.4 vs 12.2) and the whole scale sits at roughly half our magnitude
— consistent with an engine that has run its own SPSA/texel-style SEE
retuning rather than carrying SF's historical constants forward untouched.

### 5.1 Special-cased move kinds — more precise than SF/us, not less

- **Castling always passes**, unconditionally, regardless of `threshold`
  (`see.rs:13-15`) — stronger than "return `0 >= threshold`"; a
  `see(castling_move, +500)` call would still return `true`. In practice
  this is close to moot: castling is never classified as noisy
  (`is_noisy()` excludes it, `types/moves.rs:70-72`), so the MovePicker's
  `td.board.see(entry.mv, threshold)` gate (`movepick.rs:59`) never runs
  against it anyway.
- **Promotions and en passant are fully run through the real exchange
  simulation** — this is the interesting deviation. `move_value` (`see.rs:
  91-99`) computes `captured.value() + (promo.value() - Pawn.value())` when
  the move promotes, and the very first "worst case" deduction
  (`see.rs:24`) subtracts the **promoted piece's** value rather than the
  pawn's — correct, because after the promotion the piece standing on `to`
  really is a queen/rook/etc., so if it gets recaptured for free the mover
  loses a queen, not a pawn. En passant clears the actual captured-pawn
  square from the occupancy bitboard before the exchange loop starts
  (`see.rs:34-36`). Contrast: our `see_ge` (and Stockfish's) special-cases
  *every* non-`Normal` `MoveKind` — promotion, en passant, **and**
  castling alike — to a flat `return 0 >= threshold`
  (`src/board.rs:1271-1276`, design/04-position-state.md §3.11 step 1,
  explicitly documented as mirroring Stockfish's own behavior). That means
  our SEE currently treats a queen-promoting capture as if it had *zero*
  material value for pruning/ordering purposes — a deliberate,
  SF-inherited simplification, but a real accuracy gap on promotion-heavy
  tactics (7th-rank pushes, promotion races) that Reckless does not have.

### 5.2 The exchange loop: one running balance instead of a swap+flag pair

Reckless's core loop (`see.rs:47-84`) tracks a single signed `balance`,
folds the `threshold` in exactly once at the very start
(`balance = move_value(mv) - threshold`, `see.rs:18`), and updates it every
capture with one recurrence — worth quoting exactly since the "-1" is easy
to get wrong by eye:

    balance = -balance - 1 - attacker.value()

(`see.rs:71`). The loop exits the instant `balance >= 0` (this ply's
recapture has already locked in the threshold, no need to keep
negamax-ing, `see.rs:72-74`), or the instant the side to move has no legal
recapture left (`see.rs:55-57`), or immediately if a king would have to
"recapture" into an attacked square (`see.rs:61-64`, checked *before* the
capture is made — the king is simply never allowed to be the used
attacker unless the opponent has nothing left to hit back with). The final
verdict is read off by comparing the loop's ending side-to-move against the
position's actual side to move (`stm != self.side_to_move()`, `see.rs:88`)
— **no separate boolean result flag at all.**

Our `see_ge` (design/04 §3.11, `src/board.rs:1271-1396`) is the classic
Stockfish null-window swap: `swap` also folds the threshold in once, but a
*separate* `res: bool` flips every ply and is compared against `swap` as an
explicit strict/non-strict bar (`swap < res as Value`), plus a
special-cased `debug_assert` for queen recaptures (a queen recapture after
anything but a king is asserted to always clear the bar rather than
tested). Both formulations are the same mathematics — a single-ply
early-exit negamax over material balance — expressed with different
bookkeeping: Reckless folds the strict/non-strict tie-break into a
constant `-1` baked into the recurrence itself (opaque without re-deriving
it, but one fewer live variable and no queen special case); ours carries
an explicit `res` bit that documents the tie-break at each read site. A
genuinely "different but equivalent" implementation choice, not a
correctness difference.

### 5.3 Pinned-attacker handling — Reckless is measurably more precise here

Both engines exclude a would-be attacker if it's pinned to its own king
while an enemy pinner of that king is still on the board (mid-exchange, a
formerly-pinned piece becomes free to join once its pinner has itself been
captured). Our version (design/04 §3.11 step 6, `src/board.rs:1334-1339`
in the code) excludes **all** of `blockers_for_king(stm)` unconditionally
whenever a live pinner remains — documented explicitly as "Approximation:
...an approximation shared with Stockfish." Reckless computes a strictly
more accurate answer for one case: a pinned piece **recapturing exactly
along its own pin ray** is still legal (it never leaves the ray connecting
its king to the pinner), so it should still be allowed to participate.
Reckless precomputes, once per `see` call, the ray from each king through
the target square (`king_rays = [ray_pass(king(White), to),
ray_pass(king(Black), to)]`, `see.rs:44-45`) and only excludes pinned
pieces **not** on that ray:

    our_attackers &= !(pinned(stm) & !king_rays[stm])

(`see.rs:52`, paraphrased). This is a real, if narrow, accuracy edge our
SEE lacks: a rook pinned to its king along a file, asked to recapture on a
square that same file, is wrongly excluded by our (and Stockfish's)
approximation but correctly included by Reckless's.

### 5.4 Least-valuable-attacker selection: branchless bitmask + `trailing_zeros`

Reckless picks the cheapest attacker type with a 6-bit presence mask —
one unconditional non-emptiness test per piece type (P, N, B, R, Q, K, in
value order), OR'd together with the pawn test in bit 0, and
`trailing_zeros()` on the resulting byte directly yields the enum index of
the lightest piece present (`see.rs:126-138`, scalar fallback; an
AVX-512F variant does the same six tests as one `_mm512_test_epi64_mask`
vector instruction, `see.rs:101-124`, `target_feature = "avx512f"`). The
type is found first; the actual attacking square is recovered afterward
with a second, separate intersection (`see.rs:67`). Our `see_ge`
(`src/board.rs:1345-1352`) fuses both steps into one short-circuiting
`find_map` over `PieceType::ALL`, returning `(piece_type, square)`
together on the first hit. Both are O(1)-ish (six piece types, small
constant), but the two shapes have different branch-predictor profiles —
Reckless's mask computes all six tests unconditionally then does one
`trailing_zeros`; ours short-circuits but has a data-dependent branch per
iteration. Worth benchmarking, not assuming.

## 6. How SEE gets used (context, not this doc's core topic but directly
adjacent)

- **Movepick good/bad-noisy split** (`movepick.rs:59-64`): the SEE
  threshold a capture must clear is `-entry.score / 47 + 116`, where
  `entry.score` is the move's **full** ordering score — MVV term +
  noisy-history + queen-promotion bonus + in-check bonus
  (`movepick.rs:132-141`) — not just the captured piece's raw value.
  Contrast Stockfish's and our `-value / 18`
  (`stockfish_reference/03-movegen-movepick-history.md` "dynamic... a
  capture with great history tolerates slightly losing SEE";
  `src/movepick.rs:396`, `GOOD_CAPTURE_SEE_DIVISOR`) — ours scales the bar
  only by the captured piece's value, Reckless scales it by the whole
  ordering score (history and bonuses included).
- **Main-search SEE pruning** (`search.rs:823-830`) uses a threshold that's
  quadratic in `depth` **and** history-scaled for *both* the quiet and the
  noisy branch: quiet `min(0, -12·depth² + 56·depth - 27·history/1024 +
  27)`; noisy `min(0, -7·depth² - 36·depth - 39·history/1024 + 14)`. Our
  current (in-progress, design/10 §3.5, P9 Wave B) margins:
  capture/check margin is *linear* in depth plus a history term
  (`175·depth + 34·capt_hist/1024`, `src/search/mod.rs:290-302`), quiet
  margin is quadratic but **not** history-scaled
  (`-25·lmr_depth²`, `src/search/mod.rs:1451`). Reckless's quiet branch
  folding a history term in is something ours doesn't currently do.
- **Qsearch SEE pruning** (`search.rs:1301-1302`) folds in a
  correction-history-magnitude term: `(alpha - eval)/8 -
  min(68, |correction_value|) - 74` — the more the static eval has needed
  correcting recently, the more permissive (less pruning) the SEE bar
  becomes, treating correction magnitude as an uncertainty signal. Our
  qsearch (design/07 §3.8) currently does a window-relative SEE check
  (`see_ge(m, alpha - futility_base)`) plus a separate flat hard floor
  (`QS_SEE_FLOOR`, `src/search/mod.rs:2026,2040`) with no correction-based
  term.

## 7. Movegen/SEE test approach — noticeably thinner than ours

`board/tests.rs` (140 lines total) is essentially: a perft macro over 6
standard + 3 Chess960 FEN fixtures (depths chosen per position, the usual
chessprogramming.org tables, `tests.rs:12-53`); an incremental-hash-vs.
-recompute consistency check run through the same perft tree
(`hash_perft`, `tests.rs:82-111`); one Shredder-FEN round-trip test; and two
small edge-case regression tests (missing optional FEN fields, fifty-move
clock saturation at `u8::MAX`). There is no dedicated SEE fixture suite (no
"QxP undefended / NxP defended" style unit tests anywhere in the crate —
SEE correctness appears to be trusted to perft-adjacent sanity plus
whatever the SPRT gate on playing strength catches), no fuzz test of
`is_legal`/`is_legal`-equivalent over the full `u16` move space, and no
`assert_consistent`-style full-invariant oracle wired into every
make/unmake in debug builds. Our engine's test posture is heavier on all
three fronts: `Board::pseudo_legal` has a 64k-encoding fuzz test (design/05
§4 step 7), `see_ge` has a documented fixture suite tied to design/04 §4
step 9 (undefended/defended capture pairs at exact threshold boundaries),
and `assert_consistent` runs at every node in every debug build (design/04
§3.12, D24). Not an "adapt to us" item — this is a place our existing
practice is already ahead — but worth recording as a genuine, measured
difference in engineering posture between a top-2 competitive engine and
ours: Reckless appears to lean much more heavily on perft + SPRT as the
correctness net than on white-box unit fixtures for movegen/SEE
specifically.

## Adapt to us

1. **Quiet SEE-pruning margin: add a history term.** Reckless's quiet
   branch is `min(0, -12·depth² + 56·depth - 27·history/1024 + 27)`; ours
   (design/10-pruning-extensions.md §3.5, currently landing as part of
   P9 Wave B) is `-25·lmr_depth²` with no history scaling at all. Folding
   in the existing quiet-history score (already computed for move
   ordering, `src/history.rs`) the same way our capture-SEE margin already
   does (`175·depth + 34·capt_hist/1024`) is a small, well-scoped, directly
   testable change — one new term in an existing formula, no new data
   structures. Effort: low (an afternoon + SPRT). Impact: unclear but
   plausible, given it's exactly the kind of margin Wave B is already
   tuning; gate it the same way as every other Wave B heuristic (fixed-node
   proxy SPRT per design/14 §9.4, then a real-TC SPRT).

2. **Evaluate promotions and en passant through the real SEE exchange
   instead of a flat `0 >= threshold` shortcut.** Our `see_ge`
   (`src/board.rs:1271-1276`, design/04-position-state.md §3.11 step 1)
   currently mirrors Stockfish's blanket non-`Normal` shortcut exactly, on
   purpose. Reckless shows this is not load-bearing: promotions get a
   `move_value` that folds in the promotion gain and deducts the
   *promoted* piece's value on the worst-case branch, and en passant just
   clears the right occupancy bit before an otherwise-ordinary exchange
   loop — a handful of lines, no new algorithm. This directly improves
   move-ordering/pruning precision on promotion-race and 7th-rank-push
   positions, which is exactly the kind of tactic our current qsearch/
   pruning stack (design/10) leans on SEE for. Effort: low-to-moderate
   (touches a `Bench:`-relevant function — every existing SEE consumer's
   node counts could shift, so this needs the D24-adjacent discipline of
   treating it as a functional change, new bench trailer, and an SPRT
   before it's called a strength win rather than just a "more correct"
   change). Impact: plausibly small-but-real Elo, moderate confidence.

3. **Pinned-attacker precision in `see_ge`: allow on-ray recapture.**
   Reckless's `our_attackers &= !(pinned(stm) & !king_rays[stm])` (§5.3
   above) is a strictly more accurate rule than our and Stockfish's
   blanket "exclude every blocker while a pinner remains"
   (design/04 §3.11 step 6, explicitly flagged there as a shared SF
   approximation). The fix is one extra precomputed ray
   (`ray_pass(king_square(stm), m.to())`) and a mask tweak — cheap to
   implement, cheap to verify with a targeted fixture (pinned rook,
   recapture along the pin file). Effort: low. Impact: very likely
   Elo-neutral in practice (the situation is rare) but strictly more
   correct, and a good candidate to fold into whatever pass eventually
   revisits `see_ge`'s "approximation" comment — worth doing opportunistically
   rather than as a standalone SPRT-gated change, unless bench-neutrality
   can't be established, in which case treat it like item 2.

4. **Qsearch SEE margin: fold in a correction-magnitude term.** Reckless's
   `(alpha - eval)/8 - min(68, |correction_value|) - 74` treats a large
   recent eval-correction as an "our static eval is less trustworthy right
   now" signal and loosens the SEE pruning bar accordingly. Our qsearch SEE
   floor (design/07 §3.8) doesn't currently read correction-history
   magnitude at all. This is a small, well-motivated idea (uncertainty-
   aware pruning) but touches the same correction-history machinery
   several other of our search heuristics already use — a good candidate
   to batch with a broader "does correction magnitude modulate other
   pruning margins" pass rather than land in isolation. Effort: low
   (mechanically) but needs the SPRT budget shared with related
   experiments. Impact: speculative, worth a fixed-node proxy run first.

5. **Least-valuable-attacker: branchless bitmask + `trailing_zeros`.**
   Reckless's scalar six-bit presence mask (`see.rs:126-138`) is a clean,
   generalizable idiom for "which of N ordered categories is present" that
   could replace our `find_map`-over-`PieceType::ALL` in `see_ge`
   (`src/board.rs:1345-1352`). This is a pure micro-optimization — it must
   reproduce the exact same decisions and therefore the exact same node
   counts (our CLAUDE.md merge-gate rule: a non-functional speedup must
   match the old `Bench:` count bit-for-bit). Effort: low (a self-contained
   refactor + `tools/check.sh` on both sides to confirm the count doesn't
   move). Impact: small, uncertain nps win — worth a quick criterion
   microbenchmark on `see_ge` in isolation before touching the real
   function, since the AVX-512 half of Reckless's version isn't reachable
   under our `x86-64-v3` baseline (D16) and the scalar mask's win over a
   short-circuiting `find_map` over only six items is not obvious without
   measuring.

6. **`is_noisy()` as a single bit-comparison, not a `match`.** Reckless's
   `MoveKind` packs kind into 4 bits with capture/promotion as the literal
   top two bits (`types/moves.rs:18-35,78-84`), making `is_capture`/
   `is_promotion`/`is_noisy` single-instruction bit tests. Our `Move`
   already splits 2-bit kind + 2-bit promotion piece (D-pinned encoding,
   `CLAUDE.md` "Move encoding") and our `is_noisy`-equivalent predicates
   are presumably already cheap enough that this isn't worth a re-encode —
   flagging as a **considered-and-declined** idea: the encoding is a pinned
   core decision (design/02-core-types.md), re-litigating it for a
   probably-negligible instruction-count win on an already-fast predicate
   is not worth the churn. Recorded for completeness, not recommended.

7. **Not recommended: full `all_threats`-style precomputed threat map for
   king-move legality.** This is the single biggest architectural lever in
   Reckless's movegen (§0), and it is tempting, but adopting it wholesale
   would mean recomputing a full 6-piece-type setwise attack fill every
   `make_move`/`unmake_move` (ours currently computes only `checkers` +
   `blockers_for_king`, far cheaper) purely to make king-move legality O(1)
   instead of a live attacker query. Given our current pseudo-legal +
   lazy-`legal()` architecture (matching Stockfish, `design/05-movegen.md`
   §3.8-3.9) already defers the live attacker query to the one move that's
   actually played rather than every generated king move, the wholesale
   swap is unlikely to be a net win for us without also restructuring the
   MovePicker/search staging around "trust the generator's legality," and
   it would touch `movegen`/`board` mutation broadly enough to require the
   full D24 perft ritual plus an SPRT. File under "understood, deliberately
   not porting" rather than a near-term action item — but worth
   remembering as prior art if a future profiling pass ever shows
   `Board::legal`/king-move filtering as a hot spot.

8. **Chess960/FRC support shape validates our own deferred design.** Not
   a new idea — Reckless's `castling_path`/`castling_threat`/
   `castling_rooks: [Bitboard; 16]`/`[Square; 16]` arrays (`board.rs:52-54`)
   are exactly the shape our own castling geometry was already built to be
   "Chess960-ready" for per D10 (design/04-position-state.md, "Chess960
   geometry, fixed at setup"). Seeing a shipping top engine use the same
   per-kind precomputed path/threat/rook-square arrays is a useful
   confidence check that our deferred-but-prepared design (D10) is on a
   sound path when P-whatever eventually picks Chess960 back up — no
   action needed now.
