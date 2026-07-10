# chessengine

A UCI chess engine in Rust, aiming at Stockfish-class architecture. This is
also a Rust learning project: **prefer the idiomatic Rust way** over a
transliteration of C++ patterns, and briefly explain non-obvious idioms when
introducing them.

## Design documents (read before implementing anything)

`design/` holds the reviewed design suite that governs all implementation:

- `design/decisions.md` — pinned decisions D1-D27. **Decisions change here
  first; docs follow; code follows docs.**
- `design/00-master-plan.md` — phase ordering (P0-P14), milestone gates,
  current phase status.
- `design/README.md` — index and process rules for the suite.
- `design/01-*.md` … `design/14-*.md` — one contract per component. Check
  code against the owning doc's acceptance criteria (§8) before calling a
  phase done.

When implementing a component, read its design doc first and follow it; if
reality forces a deviation, update the doc (and decisions.md if a pinned
decision changes) in the same commit.

## Commands

- `cargo test` — run all tests (the default check after any change)
- `cargo clippy --all-targets` — must stay clean
- `cargo run` — the UCI loop (P0 subset: `uci`/`isready`/`d`/`quit`;
  argv subcommands `bench`, `perft <depth> [fen]`, `d`)
- `bash tools/check.sh` — the full merge gate (fmt, clippy, tests, release
  perft/bench once those phases land)
- **Fast builds** (D16, design/01 §3.7) — the supported configuration for
  strength testing, bench signatures, and SPRT; SPRT pairs must use
  identical flags:
  `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`
  (or `target-cpu=native` for a machine-local binary). Default builds stay
  portable — we deliberately don't commit these flags in `.cargo/config.toml`.
- Profiling: `RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes"
  cargo build --profile profiling` (release codegen + line tables for
  perf/flamegraph)

## Layout

Library crate + thin binary. Modules (re-exported at the crate root by
`src/lib.rs`):

- `types` — `Color`, `PieceType`, `Piece`, `Square` (+ `Display`/`FromStr`)
- `bitboard` — `Bitboard` newtype over `u64`, set ops, iteration, constants
- `board` — `Board` (hybrid bitboards + mailbox), `CastlingRights`
- `fen` — `Board::from_fen`/`to_fen`, `FenError`, `START_FEN`
- `moves` — `Move` (packed `u16`), `MoveKind`, `MoveList` (stack-allocated)
- `attacks` — const-built leaper/geometry tables (`knight_attacks`,
  `pawn_attacks`, `between`, `line`, `aligned`, ...) + fancy magic slider
  lookups behind `LazyLock` (`attacks(pt, sq, occ)`; hoist `tables()` in hot
  loops). Criterion baseline: `benches/attacks.rs` → `docs/perf-baseline.md`.

## Core encoding decisions (do not change casually)

- **Square mapping**: little-endian rank-file. a1 = 0, b1 = 1, ..., h8 = 63;
  `file = index & 7`, `rank = index >> 3`.
- **Board representation**: bitboards are primary (`pieces[color][piece_type]`,
  per-color occupancy, all occupancy); a redundant `mailbox: [Option<Piece>; 64]`
  gives O(1) piece-on-square. `put_piece`/`remove_piece` are the ONLY ways to
  mutate placement — they keep every layer in sync, and Zobrist hashing will
  hook in there too.
- **Move encoding** (Stockfish-style, must fit u16): bits 0-5 from, 6-11 to,
  12-13 promotion piece (0 = knight ... 3 = queen), 14-15 kind
  (normal/promotion/en-passant/castling). Castling is the king's move (e1g1).

## Conventions

- Newtypes with private fields; constructors enforce invariants
  (`Square::new` asserts index < 64). Expose behavior, not representation.
- No heap allocation in anything that will be hot (movegen, search):
  `MoveList` is a fixed array + len that `Deref`s to `[Move]`.
- Sets don't implement `Iterator`; they implement `IntoIterator` with a
  separate iterator struct (see `Bitboard`/`BitboardIter`).
- Text conversions go through `Display`/`FromStr` with real error types that
  implement `std::error::Error`. Fallible parsing returns `Result`, never
  panics.
- `const fn` and `const` tables wherever possible; zero runtime dependencies
  so far — keep it that way unless there's a strong reason.
- Tests: build positions with `Board::empty()` + `put_piece` or `from_fen`,
  never struct literals. Board-mutating tests should call
  `board.assert_consistent()` (test-only) to cross-check bitboards vs mailbox.
  Use the named square constants (`Square::E4`) for readability.

## Stockfish reference

`stockfish_reference/Stockfish/` holds a clone of Stockfish (gitignored).
When unsure how to do something optimally, read the corresponding C++ source,
then write a distilled markdown note in `stockfish_reference/` (tracked; index
in `NOTES.md`) so the insight isn't lost. **Never copy Stockfish code** — it's
GPLv3; take ideas, reimplement idiomatically in Rust.

## Status / roadmap

Done: types, bitboards, board + FEN, `Move`/`MoveList`; attack tables (P1:
const leapers + geometry, fancy magic sliders, exhaustively validated
against the naive ray oracle, criterion-benchmarked).

Next (in rough dependency order): make/unmake with a `StateInfo` undo stack,
Zobrist keys wired into `put_piece`/`remove_piece`, perft validation, UCI
loop, search, evaluation. Perft is the correctness oracle — no
movegen/make-unmake change lands without matching known perft counts.
