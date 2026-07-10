# Design decisions record (authoritative)

Every design doc in this directory MUST comply with these decisions. If a doc
needs to deviate, it must call the deviation out explicitly in its Open
Questions section rather than silently contradict this file.

Status: living document. Decisions are numbered for citation (D1, D2, ...).

## Project goal

A UCI chess engine in Rust with Stockfish-class architecture and, ultimately,
strength. Two constraints in permanent tension, resolved in this order:
1. **Correctness** (perft-verified, bench-reproducible) beats everything.
2. **Performance**: our data structures must never be the reason we can't
   match Stockfish's per-node costs.
3. **The Rust way**: safe, idiomatic Rust preferred; `unsafe` only with a
   measured, documented justification (D14).

## Naming and existing code (do not churn)

- **D1**: The position type is `Board` (not `Position`). Existing modules
  `types`, `bitboard`, `board`, `fen`, `moves` keep their names and public
  APIs; extensions add to them.
- **D2**: `Move` keeps OUR bit layout: bits 0-5 = from, 6-11 = to, 12-13 =
  promotion piece minus knight, 14-15 = kind (normal/promotion/ep/castling).
  This is the reverse of Stockfish's from/to order — functionally equivalent,
  not worth churning. `Move::NULL` (from==to==a1) is the none-sentinel; a
  distinct null-move sentinel, if ever needed, uses another from==to pattern.
- **D3**: Square mapping stays little-endian rank-file (a1=0 ... h8=63).
- **D4**: Piece storage stays `pieces: [[Bitboard; 6]; 2]` indexed by
  `(Color, PieceType)`. When flat piece indexing is needed (history tables),
  add `Piece::index(self) -> usize` returning `color as usize * 6 + piece_type
  as usize` (0..12) — we do NOT adopt Stockfish's 16-slot packed Piece.

## New core types

- **D5**: `Value` is a type alias `pub type Value = i32` (module `types`),
  NOT a newtype. Rationale: search formulas are dense arithmetic; newtype
  ceremony buys little here. Constants mirror Stockfish semantics:
  `VALUE_MATE = 32000`, `VALUE_INFINITE = 32001`, `VALUE_NONE = 32002`,
  `VALUE_DRAW = 0`, `MAX_PLY = 246`, TB/mate bands, and predicates
  `is_win / is_loss / is_decisive / is_mate` as free functions or a `ValueExt`
  trait. Mate scores are ply-relative from day one (`mate_in(ply)`,
  `mated_in(ply)`, `value_to_tt`, `value_from_tt`).
- **D6**: `Depth` is `pub type Depth = i32` (negative sentinels needed:
  `DEPTH_QS = 0`, `DEPTH_UNSEARCHED = -2`, `DEPTH_NONE = -3`). Ply is `usize`
  where it indexes arrays, `i32` in value arithmetic.
- **D7**: `ExtMove { mv: Move, value: i32 }` with `#[repr(C)]`, asserted
  8 bytes, lives in the MovePicker layer, not in `MoveList`.

## Board / state architecture

- **D8**: `StateInfo` is split into two structs: `CopiedState` (Copy: the five
  Zobrist keys, non-pawn material, castling rights, rule50, plies_from_null,
  ep square) and derived fields (checkers, blockers, pinners, check_squares,
  captured piece, repetition). History is a ply-indexed `Vec<StateInfo>`
  **owned by `Board`** — no pointer chains. `make_move` pushes, `unmake_move`
  pops. `Board::clone()` therefore clones history; search threads each own
  their Board copy.
- **D9**: Wire ALL five Zobrist keys from the start: `key`, `pawn_key`,
  `material_key`, `minor_key`, `non_pawn_key[2]`, with Stockfish's tricks:
  zeroed pawn psq on promotion ranks, castling keyed by the whole 4-bit mask,
  material key via count-slots, ep key only when a legal ep capture exists,
  rule50 key folding at >= 14. These are near-impossible to retrofit.
- **D10**: `make_move(&mut self, m: Move, gives_check: bool)` — caller
  supplies gives_check. Castling is encoded internally as king-to-castling-
  destination (our current Move encoding); the make/unmake implementation
  must still handle Chess960-style rook handling internally
  (remove-both-then-place-both), but Chess960 support itself is deferred.
- **D11**: SEE (`see_ge`) is a `Board` method (it needs occupancy, pinners,
  attack queries). SEE piece values: P 208, N 781, B 825, R 1276, Q 2538.

## Attack tables

- **D12**: Leaper tables (knight/king/pawn), `BETWEEN`, `LINE`, distance:
  built at compile time via `const fn`. Slider attacks: **fancy magic
  bitboards** (packed shared table, Carry-Rippler enumeration, epoch-array
  verification, hardcoded per-rank PRNG seeds), initialized once at startup
  behind `std::sync::LazyLock<AttackTables>`. No PEXT variant (current SF
  dropped it for sliders). Revisit hardcoding found magic constants later
  only if startup time ever matters.

## Module map (single crate, lib + bin)

- **D13**: One crate. Planned new modules as they land:
  `attacks` (leapers + magics), `zobrist`, `movegen`, `perft`, `see` (or in
  `board`), `eval` (later a dir), `search`, `tt`, `movepick`, `history`,
  `timeman`, `uci`, `threads`, `bench`. `main.rs` stays a thin UCI entry.

## Unsafe, dependencies, toolchain

- **D14**: Unsafe budget: allowed only for (a) prefetch intrinsics, (b) SIMD
  intrinsics in NNUE/eval, (c) proven-hot unchecked indexing AFTER a profile
  shows the bounds check and a comment records the measurement. Each unsafe
  block gets a `// SAFETY:` comment. TT relaxed atomics are SAFE Rust.
- **D15**: Zero runtime dependencies. Dev-dependencies allowed (e.g.
  criterion). External tools (cutechess-cli/fastchess, bullet NNUE trainer)
  are tooling, not dependencies.
- **D16**: Edition 2024, current stable toolchain. Release profile:
  `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` for release builds
  used in strength testing; document `RUSTFLAGS=-C target-cpu=x86-64-v3` (or
  native) as the supported fast configuration.

## Search architecture

- **D17**: PVS negamax with aspiration windows, staged MovePicker,
  history-driven ordering. **No killer/countermove tables, ever** — modern SF
  removed them; we skip that detour entirely.
- **D18**: Search stack = fixed array `[Stack; MAX_PLY + 10]` with the search
  operating at offset +7 so `ss-7 ... ss+2` reads are always in bounds
  (Stockfish's padding trick, expressed as safe indexing on a slice).
- **D19**: TT: 10-byte entries (key16, depth8, gen_bound8, move16, value16,
  eval16), all fields relaxed atomics, 3-entry 32-byte clusters, `mul_hi64`
  indexing, 5-bit generations, SF replacement rules. Lock-free and racy by
  design; every consumer validates what it reads (pseudo_legal+legal on
  ttMove, is_valid on values).
- **D20**: Tuned constants: adopt Stockfish's current values as starting
  points, recorded verbatim in the design docs with the formula SHAPES
  emphasized. We keep a single `search/params.rs` (or similar) so constants
  are greppable and tunable; no magic numbers inline without a name.

## Evaluation roadmap

- **D21**: Three phases. (A) Material + tapered PSQT — enough to validate
  search machinery. (B) Hand-crafted eval with texel tuning (own tuner) —
  intermediate strength. (C) **NNUE** — required for the end goal; implement
  inference with incremental accumulators (dirty-piece tracking hooks in
  make_move), quantized SIMD with scalar fallback; train our own net with
  external tooling (bullet) — we do NOT ship Stockfish's GPL nets.
- **D22**: Eval interface is a single `evaluate(&Board, ...) -> Value` seam so
  phases A/B/C swap without touching search. Correction history corrects the
  static eval inside search (raw eval is what the TT stores).

## Threading

- **D23**: Lazy SMP. Per-thread `Worker` owns everything (Board copy,
  histories, root moves); shared state is exactly: the TT, `AtomicBool` stop
  flags, and options. **Persistent parked worker threads + `Arc<TT>`**
  (amended 2026-07-09 from scope+`&TT`: persistent threads match SF's pool
  reuse across `go` commands and avoid re-spawn cost; docs 01/13 specify the
  mechanism). Consequences: the TT generation counter is interior-mutable
  (`AtomicU8`, relaxed) so `new_search(&self)` works through the shared
  handle; `resize`/`clear` run on the UCI thread between searches via
  `Arc::get_mut`. No locks on the search path; node counters are per-worker
  relaxed atomics aggregated on read. Single-threaded first; the data layout
  is SMP-ready from the start.

## Validation gates (non-negotiable)

- **D24**: No movegen/make-unmake change lands without the perft suite
  passing. A `bench` subcommand (fixed position suite → total nodes + nps)
  is the search regression oracle: functional changes must report the bench
  node count; "non-functional" changes must not alter it.
- **D25**: Strength claims come from games only: fastchess/cutechess SPRT
  vs. the previous version (and reference engines), never from eyeballing.
- **D26**: Every design doc gets an acceptance-criteria section; a phase is
  done when its criteria are met, not when its code exists.

## Licensing

- **D27**: We take ideas, algorithms, constants, and formulas from Stockfish
  (facts are not copyrightable); we never copy code or ship SF network files.
  Notes in `stockfish_reference/` document provenance.
