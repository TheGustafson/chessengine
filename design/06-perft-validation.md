# 06 — Perft and correctness validation

Status: Draft v1

Perft is the ground-truth oracle for the move generator and make/unmake: it
counts every legal move path to a fixed depth and compares against exactly
known totals, so a single missing en-passant pin check or stale castling
right shows up as a wrong integer, not a mystery Elo loss. This document
specifies our perft implementation (with bulk counting at the horizon),
perft divide for debugging, the exact position suite and node counts we gate
on (D24), the tiered test organization (fast debug subset in `cargo test`,
full and extended release tiers as ignored tests), the `bench` subcommand
whose total node count is the functional-change fingerprint for the whole
engine lifetime, a differential-testing tool that bisects disagreements
against a reference engine over UCI, and the debug-build invariant-checking
strategy (board consistency, Zobrist recomputation, unmake round-trips).
Every node count in this document was cross-checked against python-chess
1.11.2 (an independent legal-move implementation) or is a published
consensus value for the standard positions; counts are facts and are cited
exactly (D27).

## 1. Purpose and scope

This document owns:

- The `perft` module (D13): plain perft with bulk counting, perft divide,
  and the debug-checked perft variant.
- The validation position suite: FENs, depths, expected node counts, and
  which tier each depth belongs to.
- Test organization and runtime budgets for the perft tiers.
- The `bench` subcommand (D24): argument grammar, default position list,
  output format, determinism rules, and the `Bench:` commit-message
  convention.
- The differential-testing tool `tools/perft_diff.py` (tooling only, D15).
- The debug-build invariant-checking strategy: which invariants are checked,
  where `assert_consistent` is called, and the Zobrist recomputation
  cross-check.

This document does NOT own:

- Move generation itself — `generate_legal` and friends belong to
  design/05-movegen.md. Perft is a *consumer* of the LEGAL generator.
- `make_move`/`unmake_move`, `StateInfo`, and the five Zobrist keys — owned
  by the position-state design doc (design/04-position-state.md, see §7).
  Perft validates them but does not define them.
- The UCI loop and the `position`/`go` command parsing — owned by the UCI
  design doc. This doc only specifies the `go perft N` semantics, the
  `bench` subcommand behavior, and their output formats, because external
  tooling parses them.
- Search itself — the bench *depth* mode needs a search to exist; until
  then bench runs in perft mode (§3.6).
- SPRT/strength testing — that is D25 territory (search/testing docs).

## 2. References

- `stockfish_reference/02-position-state-make-unmake.md` — StateInfo split,
  incremental keys, make/unmake ordering: the machinery perft validates.
- `stockfish_reference/03-movegen-movepick-history.md` — pseudo-legal +
  lazy-legality generation model; "Perft uses generate<LEGAL> semantics".
- `stockfish_reference/00-design-implications.md` — item 11: "Perft
  validates everything before any search work."
- `stockfish_reference/Stockfish/src/perft.h` — SF's perft shape: bulk
  counting at the horizon (`leaf = depth == 2` → child movelist size), root
  prints per-move counts.
- `stockfish_reference/Stockfish/src/benchmark.cpp` — the default bench FEN
  list (49 standard-chess entries after dropping the two Chess960 entries
  and `setoption` lines), argument defaults `16 1 13 default depth`, and
  the corrected-time benchmark variant (which we do not adopt).
- `stockfish_reference/Stockfish/src/uci.cpp` — `bench` accumulates nodes
  across positions and prints the `Total time / Nodes searched /
  Nodes/second` trailer to stderr; `go perft` prints per-move counts and a
  `Nodes searched:` line.
- Chess Programming Wiki "Perft Results" — published deep counts for the
  standard six positions (startpos, Kiwipete, positions 3–6).

## 3. Design

### 3.1 Core perft with bulk counting

Module `src/perft.rs` (D13). Perft counts *leaf nodes*: the number of
distinct legal move sequences of exactly `depth` plies. Rules that terminate
games without terminating move sequences are ignored by definition —
**no 50-move, repetition, or insufficient-material cutoffs in perft**. A
naive "return draw" branch here corrupts counts; the halfmove clock is
carried through make/unmake but never consulted.

The one algorithmic trick worth having: **bulk counting at the horizon**. At
`depth == 1` the answer is just the length of the legal move list — no
make/unmake needed. Since leaf-parents dominate the tree, this cuts
make/unmake calls by roughly the branching factor (~30× on middlegame
positions) and is the difference between a perft that runs the full suite in
seconds versus minutes. Stockfish expresses this as `leaf = (depth == 2)`
and taking the child's movelist size; recursing into a `depth == 1` call
that returns the list length is equivalent (one extra call frame, which is
noise in Rust).

```rust
// src/perft.rs
use crate::board::Board;
use crate::movegen; // design/05-movegen.md owns generate_legal
use crate::moves::MoveList;

/// Count all legal move paths of exactly `depth` plies from `board`.
/// perft(0) == 1 by convention.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    if depth == 1 {
        return list.len() as u64; // bulk counting: no make/unmake at horizon
    }
    let mut nodes = 0;
    for m in &list {
        let gives_check = board.gives_check(m); // D10: caller supplies it
        board.make_move(m, gives_check);
        nodes += perft(board, depth - 1);
        board.unmake_move(m);
    }
    nodes
}
```

Notes:

- `&mut Board` with make/unmake, not clone-per-node: this exercises exactly
  the code path search will use (D8 history push/pop included). Perft is a
  test of make/unmake as much as of movegen.
- Perft deliberately routes through `gives_check(m)` so the O(1)
  check-detection machinery (check_squares, discovered checks) is validated
  before search ever relies on it. The debug-checked variant (§3.3)
  cross-checks the prediction.
- `for m in &list` binds `m: Move` *by value*: `&MoveList` implements
  `IntoIterator<Item = Move>` (`iter().copied()` over the
  `Deref<Target = [Move]>` slice — already in today's `moves.rs`), and
  `Move` is `Copy`, so it feeds `gives_check`/`make_move`/`unmake_move`
  (which take `Move` by value, D10) with no reference friction.
- Stack use per frame is one `MoveList` (520 bytes) plus locals; at suite
  depths (≤ 8) this is trivially fine on the default stack.
- No transposition-table acceleration ("hashed perft"). Rejected: the
  oracle must not depend on the Zobrist keys it is itself validating, and
  exactness beats speed here. If we ever want faster deep perft we
  parallelize at the root instead (§9).

### 3.2 Perft divide

Divide is perft's debugging face: per-root-move subtree counts. When a total
disagrees with the table, divide tells you *which move* to descend into; a
few iterations of divide-descend pinpoint the exact position where the move
list is wrong. The differential script (§3.7) automates this loop.

```rust
/// One root move and the size of its subtree.
pub struct DivideEntry {
    pub mv: Move,
    pub nodes: u64,
}

/// Per-root-move counts plus the total. Entries in generation order.
pub struct DivideResult {
    pub entries: Vec<DivideEntry>,
    pub total: u64,
}

pub fn divide(board: &mut Board, depth: u32) -> DivideResult {
    debug_assert!(depth >= 1);
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    let mut entries = Vec::with_capacity(list.len());
    let mut total = 0;
    for m in &list {
        let nodes = if depth == 1 {
            1
        } else {
            let gives_check = board.gives_check(m);
            board.make_move(m, gives_check);
            let n = perft(board, depth - 1);
            board.unmake_move(m);
            n
        };
        entries.push(DivideEntry { mv: m, nodes });
        total += nodes;
    }
    DivideResult { entries, total }
}
```

Text output (via a `Display` impl or a UCI-layer printer) copies Stockfish's
format exactly, because `tools/perft_diff.py` parses both engines with one
parser — moves in UCI long algebraic (our `Move: Display` already does
this, including the promotion suffix), then a blank line and the total:

```
a2a3: 44
b2b3: 42
...
e2e4: 44

Nodes searched: 2039
```

Exposed at the UCI layer as `go perft <depth>` (matching SF), and as a CLI
convenience `chessengine perft <depth> [fen...]` for use before the UCI
loop exists. Output goes to stdout.

### 3.3 Debug-checked perft and invariant strategy

A second entry point, `perft_checked`, runs the same traversal but verifies
engine invariants at every node. This is where make/unmake bugs die young.
It is compiled always but intended for debug/test builds; the fast tests run
it at shallow depth (§3.5).

Invariants checked, in order, after each `make_move`:

1. **Layer consistency**: bitboards ↔ mailbox ↔ occupancy agreement.
   `Board::assert_consistent` is promoted from `#[cfg(test)]` to
   `#[cfg(any(test, debug_assertions))]` so `perft_checked` can call it from
   a debug binary, not just unit tests. (Behavior unchanged in release.)
2. **Zobrist recomputation**: all five incremental keys (D9) — `key`,
   `pawn_key`, `material_key`, `minor_key`, `non_pawn_key[2]` — must equal a
   from-scratch recomputation. The Zobrist module must therefore expose a
   slow, obviously-correct `zobrist::compute_from_scratch(&Board) -> KeySet`,
   where `KeySet` is a small `PartialEq` struct of exactly those key values,
   mirrored by a `Board::keys()` accessor. (`KeySet` is distinct from the
   random-number table `zobrist::Tables` behind the `ZOBRIST` static —
   design/04-position-state.md §3.4 owns both.) The computation is the same
   one `from_fen`/`init_state` setup uses. The recomputation must
   reproduce the SF key rules exactly, or the cross-check itself lies:
   castling keyed by the whole 4-bit mask, ep file keyed **only when a legal
   ep capture exists**, zeroed pawn-psq on promotion ranks, material key via
   count slots. The rule50 fold (>= 14) applies to the *exposed* key, not
   the stored one — compare stored keys pre-fold.
3. **gives_check agreement**: the `gives_check` the caller computed before
   `make_move` must equal `!checkers.is_empty()` after it. Because make
   trusts the flag to set `checkers` cheaply (reference note 02), this
   check must recompute checkers independently:
   `board.attackers_to(ksq, occ) & board.occupancy(them)`.
4. **Derived-state recomputation**: stored `checkers`, `blockers_for_king`,
   `pinners` equal fresh recomputations via the slow path.

And after each `unmake_move`:

5. **Round-trip identity**: the board equals a snapshot cloned before
   `make_move` — piece placement, side to move, castling rights, ep square,
   clocks, all five keys, and state-stack depth. `Board` derives
   `PartialEq`, so this is one comparison. That derive requires `StateInfo`
   and `CopiedState` (design/04-position-state.md §3.2) to carry
   `PartialEq` alongside `Clone, Copy` — every field is comparable (keys,
   bitboards, `Option<Piece>`, fixed arrays), and `Vec<StateInfo>`'s
   derived comparison is by length and elements, so capacity differences
   never matter. Doc 04 §5's unmake round-trip test uses the same
   comparison. If the derived impl ever grows fields that legitimately
   differ (caches), both docs' tests fall back to comparing a
   `(fen, keys)` projection instead.

```rust
/// Perft that validates board invariants at every node. ~10-50x slower
/// than `perft`; used by tests and for bug hunts, never for timing.
pub fn perft_checked(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for m in &list {
        let snapshot = board.clone();
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);

        board.assert_consistent();
        let fresh = zobrist::compute_from_scratch(board);
        assert_eq!(board.keys(), fresh, "incremental key drift after {m}");
        assert_eq!(gives_check, !board.checkers().is_empty(),
                   "gives_check mispredicted for {m}");
        board.assert_derived_state(); // checkers/blockers/pinners recompute

        nodes += perft_checked(board, depth - 1);
        board.unmake_move(m);
        assert_eq!(*board, snapshot, "unmake_move failed to restore after {m}");
    }
    nodes
}
```

Call-site strategy for `assert_consistent` outside perft: `make_move` and
`unmake_move` themselves carry only cheap `debug_assert!`s (piece present on
`from`, destination sanity) — the full O(64) consistency scan lives in
`perft_checked` and in board-mutating unit tests (per CLAUDE.md), so debug
search remains usable while debug perft stays maximally paranoid.

### 3.4 The validation suite

One const table drives every tier. Counts marked ✔ were verified for this
document with python-chess 1.11.2; deeper counts are the published CPW
consensus values.

```rust
/// A perft regression case. `counts` ascending by depth and covers the
/// fast + full tiers; `extended` holds the multi-billion-node depths.
pub struct PerftCase {
    pub name: &'static str,
    pub fen: &'static str,
    pub counts: &'static [(u32, u64)],
    /// Depths <= fast_depth run in debug `cargo test`.
    pub fast_depth: u32,
    pub extended: &'static [(u32, u64)],
}

pub const PERFT_SUITE: &[PerftCase] = &[ /* tables below */ ];
```

**Standard positions** (CPW "Perft Results" set):

| name | FEN | depth: nodes | fast ≤ |
|---|---|---|---|
| startpos | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 1: 20 ✔, 2: 400 ✔, 3: 8 902 ✔, 4: 197 281 ✔, 5: 4 865 609 ✔, 6: 119 060 324; ext 7: 3 195 901 860 | 4 |
| kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1` | 1: 48 ✔, 2: 2 039 ✔, 3: 97 862 ✔, 4: 4 085 603 ✔, 5: 193 690 690; ext 6: 8 031 647 685 | 3 |
| cpw3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 1: 14 ✔, 2: 191 ✔, 3: 2 812 ✔, 4: 43 238 ✔, 5: 674 624 ✔, 6: 11 030 083 ✔, 7: 178 633 661 | 5 |
| cpw4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 1: 6 ✔, 2: 264 ✔, 3: 9 467 ✔, 4: 422 333 ✔, 5: 15 833 292 ✔, 6: 706 045 033 | 4 |
| cpw5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 1: 44 ✔, 2: 1 486 ✔, 3: 62 379 ✔, 4: 2 103 487 ✔, 5: 89 941 194 | 3 |
| cpw6 | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 1: 46 ✔, 2: 2 079 ✔, 3: 89 890 ✔, 4: 3 894 594 ✔, 5: 164 075 551; ext 6: 6 923 051 137 | 3 |

Kiwipete alone covers castling through/into attack, ep, promotions,
discovered and double checks in one position; cpw4 covers *both* sides'
rights revoked by rook capture and promotion-with-capture out of a
half-open position; cpw5 is the classic make/unmake killer (promoted piece
must un-promote cleanly).

**Tricky positions.** The first block is the well-known TalkChess/Sedlak
set; `ep-pin-rank` is the Peter Ellis Jones rank-pin position;
`double-check` and the exact `castle-prevented` ladder were
generated/verified with python-chess for this document. Every count below
is ✔-verified at every listed depth.

| name | FEN | what it catches | depth: nodes | fast ≤ |
|---|---|---|---|---|
| ep-pin-h | `3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1` | ep capture illegal: *both* pawns leave rank 5, exposing Ka5 to Rh5 (ep-discovered-pin) | 1: 18, 2: 92, 3: 1 670, 4: 10 138, 5: 185 429, 6: 1 134 888 | 4 |
| ep-pin-rank | `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1` | ep target set but exd3 illegal (Qh4–Ka4 rank pin through both pawns) | 1: 6, 2: 136, 3: 863, 4: 20 471, 5: 117 741, 6: 2 822 114 | 4 |
| ep-pin-diag | `8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1` | after d2d4, cxd3 ep would self-expose along a2–e6 diagonal | 1: 13, 2: 102, 3: 1 266, 4: 10 276, 5: 135 655, 6: 1 015 133 | 4 |
| ep-gives-check | `8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1` | legal ep capture that gives check; ep interaction with evasions | 1: 15, 2: 126, 3: 1 928, 4: 13 931, 5: 206 379, 6: 1 440 467 | 4 |
| castle-rights | `r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1` | rights revoked when a *rook is captured* (…Bxh1, …Bxa1, Qxh1) — the `mask[from] \| mask[to]` trick | 1: 26, 2: 1 141, 3: 27 826, 4: 1 274 206 | 3 |
| castle-prevented | `r3k2r/8/3Q4/8/8/8/8/R3K2R b KQkq - 0 1` | castling blocked by attacked transit squares while rights persist | 1: 20, 2: 872, 3: 16 068, 4: 665 077 | 3 |
| castle-check-k | `5k2/8/8/8/8/8/8/4K2R w K - 0 1` | O-O delivering check; rook path vs king path | 1: 15, 2: 66, 3: 1 198, 4: 6 399, 5: 120 330, 6: 661 072 | 4 |
| castle-check-q | `3k4/8/8/8/8/8/8/R3K3 w Q - 0 1` | O-O-O delivering check; b1 not in king's path | 1: 16, 2: 71, 3: 1 286, 4: 7 418, 5: 141 077, 6: 803 711 | 4 |
| promo-storm | `n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1` | mass promotions + under-promotions + promo-captures, both colors | 1: 24, 2: 496, 3: 9 483, 4: 182 838, 5: 3 605 103, 6: 71 179 139 | 4 |
| promo-out-of-check | `2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1` | promotion (exf8) as a check evasion | 1: 11, 2: 133, 3: 1 442, 4: 19 174, 5: 266 199, 6: 3 821 001 | 4 |
| promo-give-check | `4k3/1P6/8/8/8/8/K7/8 w - - 0 1` | promotion delivering check (gives_check on promotions) | 1: 9, 2: 40, 3: 472, 4: 2 661, 5: 38 983, 6: 217 342 | 5 |
| underpromo-check | `8/P1k5/K7/8/8/8/8/8 w - - 0 1` | under-promotion delivering check | 1: 6, 2: 27, 3: 273, 4: 1 329, 5: 18 135, 6: 92 683 | 5 |
| double-check | `4k3/8/3N4/8/8/8/8/4RK2 b - - 0 1` | double check (Nd6 + Re1): only king moves legal at the root | 1: 3, 2: 69, 3: 212, 4: 4 755, 5: 23 842, 6: 550 752 | 5 |
| discovered-check | `8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1` | discovered checks and evasions with knight forks | 1: 29, 2: 165, 3: 5 160, 4: 31 961, 5: 1 004 658 | 4 |
| self-stalemate | `K1k5/8/P7/8/8/8/8/8 w - - 0 1` | near-stalemate corner; zero-move nodes counted correctly | 1: 2, 2: 6, 3: 13, 4: 63, 5: 382, 6: 2 217 | 6 |
| stale-checkmate | `8/k1P5/8/1K6/8/8/8/8 w - - 0 1` | stalemate and checkmate leaves mixed | 1: 10, 2: 25, 3: 268, 4: 926, 5: 10 857, 6: 43 261, 7: 567 584 | 5 |
| double-checkmate2 | `8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1` | mating nets: forced-mate leaves at shallow depth | 1: 37, 2: 183, 3: 6 559, 4: 23 527 | 4 |

Perft counts do not depend on generation order, so the table is stable under
any movegen refactor — exactly the property an oracle needs.

### 3.5 Test organization and runtime budgets

Three tiers, all driven off `PERFT_SUITE`:

1. **Fast tier — debug, runs in plain `cargo test`.** In `src/perft.rs`
   unit tests. All positions at depths ≤ `fast_depth` (≈ 2.1 M nodes total),
   plus `perft_checked` on every position at `max(1, fast_depth - 1)`
   (≈ 90 K nodes with full invariant checking). Budget: **< 10 s debug**;
   expected ~1–3 s (debug bulk-counted perft runs at several Mnps).
2. **Full tier — release, ignored test.** `tests/perft_suite.rs`:

   ```rust
   #[test]
   #[ignore = "release perft suite: cargo test --release -- --ignored"]
   fn perft_full_suite() { /* every (depth, count) in every case's `counts` */ }
   ```

   Total ≈ 1.6 G nodes (dominated by cpw4 d6 = 706 M, kiwipete d5 = 194 M,
   cpw3 d7 = 179 M, cpw6 d5 = 164 M, startpos d6 = 119 M). Budget:
   **< 30 s** at the ≥ 100 Mnps target (§6); expected 6–15 s.
3. **Extended tier — release, separately named ignored test**
   (`perft_extended`): startpos d7 (3.20 G), kiwipete d6 (8.03 G), cpw6 d6
   (6.92 G) ≈ 18 G nodes. Budget: **< 4 min**. Run before releases and after
   any movegen/make-unmake rewrite; not part of the routine gate.

The D24 gate, concretely: any change touching `movegen`, `board` mutation,
`zobrist`, or `attacks` must pass tier 1 locally (automatic via
`cargo test`) **and** tier 2 (`cargo test --release -- --ignored perft`)
before landing. Tier 3 at maintainer discretion per above. Record this
ritual in CLAUDE.md when the module lands.

### 3.6 The bench subcommand (D24)

Bench is the *search* regression oracle: a fixed position list searched to a
fixed depth on one thread produces a total node count that is a pure
function of the engine's functional behavior. Any eval/search/ordering
change moves it; any refactor that claims to be non-functional must
reproduce it bit-for-bit. This is the Stockfish workflow and we adopt it
wholesale, including the commit-message convention: **every functional
commit ends with a `Bench: <total-nodes>` trailer; non-functional engine
commits state `No functional change`** (verified by running bench on both
sides). Once search lands, the trailer joins `cargo test`/clippy in the
pre-commit ritual.

Invocation (argv subcommand, handled before the UCI loop; also available as
a `bench` command inside the loop), argument order and defaults exactly as
SF so muscle memory and scripts transfer:

```
chessengine bench [ttMB=16] [threads=1] [limit=13] [fens=default] [type=depth]
```

```rust
// src/bench.rs
pub enum FenSource { Default, Current, File(std::path::PathBuf) }
pub enum BenchLimit { Depth(i32), Perft(u32), Nodes(u64), Movetime(u64) }

pub struct BenchParams {
    pub tt_mb: usize,
    pub threads: usize,
    pub limit: BenchLimit,
    pub fens: FenSource,
}

pub struct BenchResult { pub nodes: u64, pub elapsed_ms: u64 }

/// Run the bench: one ucinewgame, then for each entry set the position
/// (entries may carry "moves ..." suffixes) and search/perft it,
/// accumulating node counts. Diagnostics to stderr, per SF.
pub fn run(params: &BenchParams /*, &mut Engine */) -> BenchResult;
```

Semantics pinned down for determinism:

- `ucinewgame` once at the start; the TT is **not** cleared between
  positions (SF behavior — TT carry-over is part of the fingerprint).
- Default mode is `depth 13`, single thread: node count is exactly
  reproducible run-to-run and machine-to-machine. `movetime` mode exists
  for throughput measurement only and is never a fingerprint.
- `type=perft` runs `perft(limit)` on each position instead of a search —
  this makes bench useful from day one, before search exists, and doubles
  as a whole-suite movegen throughput measure (`bench 16 1 5 default
  perft`).
- Nodes accumulated from the engine's per-search node counters (later: sum
  of per-worker relaxed counters, D23 — with threads=1 this is exact and
  deterministic; bench with threads > 1 is legal but its count is not a
  fingerprint).
- Output: per-position `Position: i/N (fen)` lines to **stderr**, and the
  SF-format trailer to stderr so stdout stays a clean UCI stream:

```
===========================
Total time (ms) : 1234
Nodes searched  : 2049514
Nodes/second    : 1660903
```

`Nodes searched` is THE number: the `Bench:` trailer in commit messages, and
what `tools/check_bench.sh` re-verifies (build release, run bench, compare
against the trailer of HEAD's commit message; exit nonzero on mismatch).

**Default position list**: we adopt Stockfish's current 49-entry standard
chess bench list verbatim (positions are facts; provenance:
`benchmark.cpp`, Chess960 entries dropped since 960 is deferred per D10).
The list is deliberately diverse: openings/middlegames, closed positions,
the two synthetic maximum-threat positions, 5/6/7-man endgames, and mate/
stalemate positions (which exercise "search with zero or one legal move" —
the UCI doc must handle `bestmove` with no legal moves). Three entries carry
`moves ...` suffixes, so the bench runner feeds entries through the same
`position fen ... [moves ...]` parser the UCI loop uses. Stored as:

```rust
pub const BENCH_POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "1r6/1P4bk/3qr1p1/N6p/3pp2P/6R1/3Q1PP1/1R4K1 w - - 1 42",
    "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    "K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - -",
    "7k/7P/6K1/8/3B4/8/8/8 b - -",
];
```

(The last two entries have no clock fields; our `from_fen` already accepts
4–6 fields.)

### 3.7 Differential testing vs a reference engine

`tools/perft_diff.py` — Python 3 stdlib only (it is tooling, not a
dependency, D15). It drives two UCI engines as subprocesses — ours and a
reference (any engine supporting `go perft` with SF-style output; path via
`--ref` or `$PERFT_REF`) — and **bisects** any disagreement down to a
single position and move list:

```
diff(fen, moves, depth):
    ours = divide(our,  fen, moves, depth)   # dict move -> count
    ref  = divide(ref,  fen, moves, depth)
    if ours == ref: return AGREE
    missing = ref.keys()  - ours.keys()      # we fail to generate
    extra   = ours.keys() - ref.keys()       # we generate an illegal move
    if missing or extra or depth == 1:
        report(fen, moves, missing, extra, count_deltas); return
    pick any m with ours[m] != ref[m]
    diff(fen, moves + [m], depth - 1)        # descend into the divergence
```

The report is a *minimal reproducer*: a FEN plus a move sequence reaching
the exact position where the two engines' legal-move sets or counts differ —
paste it into a unit test and fix. Engine protocol needed (owned by the UCI
doc, required by this tool): `uci`, `isready`, `position fen F [moves ...]`,
`go perft N` with `move: count` lines and a `Nodes searched:` total.

Modes:

- `--suite` — run every `PERFT_SUITE` entry at its full-tier depths.
- `--fen FEN --depth N` — single position.
- `--walk N --depth D [--seed S]` — random-walk fuzzing: from startpos (or
  `--fen`), repeatedly diff at depth D, then play one random legal move
  (sampled from the reference's divide keys, seeded for reproducibility),
  for N plies or until a divergence. This explores positions the fixed
  suite never reaches (long rule50 runs, weird material).

The script is a debugging and pre-merge confidence tool; the *gate* remains
the exact-count suite (§3.5), which needs no external binary.

### 3.8 Rejected alternatives

- **Hashed/TT perft**: faster deep perft, but the oracle would depend on
  Zobrist correctness — the very thing under test. Rejected.
- **Clone-per-node perft** (no unmake): simpler, but it would stop testing
  `unmake_move`, which search relies on absolutely. Rejected; the checked
  variant's snapshot comparison gives us the best of both.
- **Legality by post-hoc king-capture check** (pseudo-legal make + "was king
  left in check"): masks bugs in the pin/blocker machinery that search's
  `legal(m)` uses. Perft must go through the same `generate_legal` path as
  the engine (reference note 03).

## 4. Implementation plan

Prerequisites: attack tables, Zobrist keys, make/unmake, and LEGAL movegen
landed (§7). Steps are commit-sized; each names its done-condition.

1. **`perft` module with bulk counting + fast-tier tests.** `perft()`,
   `PerftCase`, `PERFT_SUITE` with the standard six positions, unit tests at
   `fast_depth`. Done when: `cargo test` (debug) passes with startpos d1–4,
   kiwipete d1–3, cpw3–6 fast depths, in < 10 s.
2. **Tricky positions + full suite tables.** Add the 17 tricky cases and
   the `extended` arrays; add `tests/perft_suite.rs` with the ignored
   `perft_full_suite` and `perft_extended` tests. Done when:
   `cargo test --release -- --ignored perft_full_suite` is green in < 30 s
   on the dev machine, and `perft_extended` has passed once (runtime
   recorded in the commit message).
3. **Checked perft + invariant plumbing.** Promote `assert_consistent` to
   `#[cfg(any(test, debug_assertions))]`; add
   `zobrist::compute_from_scratch`, `Board::assert_derived_state`,
   `perft_checked`, and the checked mini-tier test. Done when: checked tier
   passes AND a deliberately injected bug (e.g. skip the ep-pin legality
   test in movegen, or drop the castling-mask update on rook capture) is
   caught by the checked/fast tier — do this mutation test once, note the
   result in the commit, revert.
4. **`divide` + CLI/UCI surface.** `divide()`, `DivideResult` printing in
   SF format, `go perft N` in the UCI loop and the `chessengine perft`
   convenience subcommand. Done when: output for kiwipete d2 matches SF's
   own `go perft 2` per-move counts line-for-line (order may differ; sort
   before diffing).
5. **`tools/perft_diff.py`.** Subprocess driver, bisection, suite/fen/walk
   modes. Done when: with a reference engine configured, `--suite` agrees
   everywhere, and re-injecting the step-3 mutation yields a minimal
   reproducer FEN automatically.
6. **`bench` subcommand, perft mode.** `src/bench.rs`, `BENCH_POSITIONS`,
   argument parsing with SF defaults, stderr trailer format. Done when:
   `chessengine bench 16 1 5 default perft` prints a deterministic
   `Nodes searched` twice in a row and matches a python-chess computed sum
   once (record it).
7. **`bench` depth mode + `Bench:` convention** (blocked on search doc).
   Wire `go depth` search into bench, add `tools/check_bench.sh`, document
   the trailer rule in CLAUDE.md. Done when: two consecutive
   `bench` runs report identical node counts, a functional search tweak
   changes the count, and `check_bench.sh` fails a commit whose trailer
   doesn't match.

## 5. Testing and validation

- **Unit tests** (`src/perft.rs`): fast tier (§3.5); `perft(0) == 1`;
  `divide` total equals `perft` for kiwipete d3; divide entries sum to the
  total; `perft_checked == perft` on the mini tier (the counts, not just
  absence of panics).
- **Table self-checks**: a test asserting `counts` are strictly ascending in
  depth and `fast_depth` appears in `counts` — guards table typos.
- **Integration tests** (`tests/perft_suite.rs`): full and extended tiers as
  ignored tests, exact counts from §3.4. These ARE the D24 gate.
- **Mutation testing** (one-off, step 3/5): verify the suite actually
  catches the classic bug classes it claims to (ep pin, castling-rights on
  rook capture, under-promotion, double-check evasions) by breaking each
  once and watching the specific position fail.
- **Bench regression**: `tools/check_bench.sh` ties every functional commit
  to its fingerprint; `bench ... perft` mode gives a movegen-only
  fingerprint usable before search exists.
- **Differential fuzzing**: `perft_diff.py --walk` runs (e.g. 1000 plies,
  depth 4) before declaring movegen done — catches suite blind spots.
- **Regression strategy**: when any perft/differential failure is fixed,
  its minimal position enters `PERFT_SUITE` as a new tricky case with
  python-chess-verified counts. The suite only grows.

## 6. Performance notes

- **Targets** (release profile per D16, `target-cpu=x86-64-v3`,
  single thread): bulk-counted perft ≥ **100 Mnps** on kiwipete d5
  (hard acceptance floor), ~250+ Mnps stretch once magics and movegen are
  tuned. startpos d6 ≈ 1 s, full tier < 30 s, extended < 4 min all follow
  from the floor.
- **What to measure**: `bench 16 1 5 default perft` (whole-list nps) and
  kiwipete d5/d6 wall time. Track in commit messages when touching movegen;
  criterion (dev-dep, D15) optional for micro-level movegen benches.
- **Pitfalls**:
  - Forgetting bulk counting costs ~30× — if perft is mysteriously slow,
    check this first.
  - `MoveList::push` asserts capacity; in release this bound check is real
    but negligible next to generation cost. Do NOT unsafe-index it without
    the D14 profile ritual.
  - `gives_check` in the perft loop must be the O(1) check_squares path;
    an attackers_to-based fallback doubles perft cost and hides the fact
    that search will inherit the slow path.
  - History `Vec<StateInfo>` (D8) reallocation: `Board` should reserve
    ~`MAX_PLY` states up front so perft/search never grow it mid-run.
  - Debug-tier runtime creep: keep the fast tier's total node budget
    ≈ 2 M; when adding regression cases prefer depth ladders whose fast
    depth stays ≤ ~50 K nodes.
- Perft speed is a *movegen* benchmark, not a goal in itself: past the
  floor, effort goes to search, not perft records.

## 7. Interactions

Topical ownership is the binding reference if filenames ever shift.

- **Depends on**: design/decisions.md (D8–D15, D24, D26, D27);
  design/05-movegen.md (`generate_legal`, legality model, `gives_check`
  home); design/03-attack-tables.md (magics must init before any
  generation); design/04-position-state.md (Zobrist tables and the five
  keys, `compute_from_scratch`, `make_move`/`unmake_move`, `StateInfo`,
  derived check info that `perft_checked` cross-checks).
- **Feeds**: design/12-uci-time-management.md (`go perft`, `position ...
  moves`, `bench` command wiring, zero-legal-move searches);
  design/07-search-core.md (bench depth mode, node accounting, `Bench:`
  trailer discipline — the transposition table itself is
  design/08-transposition-table.md and stays out of perft, §3.8);
  design/13-threading-smp.md (D23 — per-worker node counters that bench
  aggregates); every future functional change, via the bench fingerprint.
- **Updates**: CLAUDE.md gains the perft/bench commands and the
  commit-trailer convention once steps 1–7 land.

## 8. Acceptance criteria (D26)

1. `cargo test` (debug) runs the fast tier + checked mini-tier green in
   < 10 s, with every count exactly matching §3.4.
2. `cargo test --release -- --ignored perft_full_suite` green in < 30 s;
   every listed full-tier count exact.
3. `perft_extended` (startpos d7, kiwipete d6, cpw6 d6) has passed at least
   once, with runtime recorded.
4. Bulk-counted kiwipete d5 ≥ 100 Mnps in the release configuration
   (measured and recorded, D16).
5. `go perft N` / `chessengine perft` output is SF-format
   (`move: count` lines + `Nodes searched: N`) and is successfully parsed
   by `tools/perft_diff.py` against a reference engine, with `--suite`
   fully agreeing.
6. The one-off mutation test (step 3) demonstrably caught an injected
   ep-pin bug and an injected castling-rights bug via specific suite
   positions.
7. `chessengine bench` (perft mode pre-search; depth mode post-search)
   prints the SF-format trailer and produces byte-identical
   `Nodes searched` across two consecutive runs at `threads=1`.
8. The `Bench:` / `No functional change` commit-trailer convention and
   `tools/check_bench.sh` are in place and documented in CLAUDE.md
   (criterion applies from the first search commit onward).

## 9. Open questions

1. **Bench list custody**: we adopt SF's current 49-position default list
   verbatim as a battle-tested starting point (D20 spirit). Once our
   eval/search diverges substantially (NNUE phase), revisit whether a
   list curated for *our* engine's blind spots fingerprints better. Low
   stakes — the fingerprint only needs to be fixed and diverse.
2. **Root-parallel perft for the extended tier**: an easy
   `std::thread::scope` split over root moves would cut the extended tier
   to seconds with zero determinism loss (counts are order-independent).
   Deferred until the single-threaded floor is met; not worth complexity
   before threading (D23) lands.
3. **`castle-prevented` provenance**: circulated copies of the TalkChess
   list pair this FEN with 1 720 476 at depth 4; independent computation
   (python-chess) gives 665 077, which we adopt. If our engine also
   produces 665 077 at d4 (and the rest of the ladder matches), treat the
   circulated figure as a transcription artifact and close this question.
4. **CI**: whether to add a GitHub Actions job running the fast tier per
   push and the full tier nightly — pending a decision on hosted CI for
   this repo at all. The local D24 ritual does not wait for it.
