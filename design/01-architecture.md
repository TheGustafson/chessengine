# 01 — Architecture, module layout, and conventions

Status: Reviewed v1

This document is the top-level map of the engine: the single-crate layout
(D13), every planned module with its responsibility and public seam, the
end-to-end data flow from UCI stdin to attack tables and back, and the
crate-wide policies — error handling, `unsafe` (D14), dependencies (D15),
build profiles (D16), and the `params.rs` convention for tuned constants
(D20). Component-level algorithms (magics, make/unmake, search, TT, ...) are
deliberately sketched only at the *signature* level here; each has its own
design doc (§7). Everything in this document complies with
`design/decisions.md`; deviations, if any, appear only in §9.

---

## 1. Purpose and scope

**This doc owns:**

- The crate/binary split, the module inventory, and each module's public
  boundary (what it exports, what it may depend on).
- The repo layout and how `design/`, `stockfish_reference/`, and `src/`
  relate to each other.
- Crate-wide conventions: error handling, `debug_assert` policy, `unsafe`
  policy application, dependency policy, build profiles and flags, the tuned
  constants (`params.rs`) convention, naming, and re-export rules.
- The end-to-end data flow: which module calls which, what is shared between
  threads, and where the hot path is.

**This doc explicitly does NOT own** (each pointer is the owning doc):

- Core value/depth/score types, mate helpers, `Move::raw16` →
  `design/02-core-types.md`
- Attack-table construction and magic bitboards → `design/03-attack-tables.md`
- `StateInfo`, make/unmake, Zobrist keys, SEE, draw rules →
  `design/04-position-state.md`
- Move generation → `design/05-movegen.md`
- Perft, validation tiers, and the `bench` subcommand →
  `design/06-perft-validation.md`
- Search core (iterative deepening, aspiration, PVS skeleton, stack) →
  `design/07-search-core.md`
- Transposition table internals → `design/08-transposition-table.md`
- MovePicker staging and history tables → `design/09-move-ordering-history.md`
- Pruning, extensions, and reductions → `design/10-pruning-extensions.md`
- Evaluation (phases A/B/C per D21, including NNUE) →
  `design/11-evaluation.md`
- UCI protocol details, options, and time management →
  `design/12-uci-time-management.md`
- Threading / lazy SMP → `design/13-threading-smp.md`
- Performance measurement and testing protocol (bench discipline, SPRT) →
  `design/14-performance-testing.md`

Signatures shown here for those components are *seams*: the owning doc may
refine parameter lists, but a change to a seam shown here must be reflected
back into this doc in the same commit. **Where this doc and a component doc
disagree on a detail inside the component's scope, the component doc wins**
and this doc is the one that gets corrected.

---

## 2. References

- `stockfish_reference/00-design-implications.md` — distilled action list;
  the module ordering below follows its roadmap.
- `stockfish_reference/01-bitboards-attacks-move-encoding.md` — table sizes,
  `Value` constants, move encoding (backs the `types`/`attacks` seams).
- `stockfish_reference/02-position-state-make-unmake.md` — StateInfo split,
  history-as-Vec, `make_move` contract (backs the `board` seam).
- `stockfish_reference/03-movegen-movepick-history.md` — GenType
  monomorphization, ExtMove, staged picking (backs `movegen`/`movepick`/
  `history` seams).
- `stockfish_reference/04-tt-search-structures.md` — TT layout, search
  Stack/RootMove, threading data layout (backs `tt`/`search`/`threads`
  seams).
- `stockfish_reference/stockfish_search_notes.md` — the full search
  algorithm survey at SF commit `eca43a97` (2026-07-06); all tuned constants
  cited in this doc suite trace to that commit.
- SF source (read-only, never copied, per D27):
  `stockfish_reference/Stockfish/src/types.h` (value bands, piece values),
  `engine.cpp`/`uci.cpp` (engine glue shape), `thread.{h,cpp}` (worker
  layout), `benchmark.cpp` (bench suite concept).

---

## 3. Design

### 3.1 Crate layout and repo layout

One crate: a library plus a thin binary (D13). Everything testable lives in
the library; `main.rs` only dispatches. The library is what unit tests,
integration tests, and (dev-only) criterion benches link against.

```
chessengine/
├── Cargo.toml               # zero [dependencies]; profiles in §3.7
├── CLAUDE.md                # working conventions (kept in sync with this doc)
├── design/                  # THE build plan — decisions.md is authoritative
│   ├── decisions.md
│   ├── 01-architecture.md   # this file
│   └── 02..14-*.md          # per-component designs (see §7)
├── stockfish_reference/
│   ├── NOTES.md             # index of study notes (tracked)
│   ├── 0X-*.md, stockfish_search_notes.md
│   └── Stockfish/           # GPL clone, gitignored, READ-ONLY reference
├── src/
│   ├── lib.rs               # module tree, re-exports, #![deny(unsafe_code)]
│   ├── main.rs              # thin bin: one line, delegates to uci::run()
│   ├── types.rs             # Color/PieceType/Piece/Square + Value/Depth (§3.3.1)
│   ├── bitboard.rs          # Bitboard + iteration (exists)
│   ├── moves.rs             # Move/MoveKind/MoveList (exists)
│   ├── board.rs             # Board, CastlingRights (exists; grows make/unmake)
│   ├── fen.rs               # FEN parse/print (exists)
│   ├── attacks.rs           # leaper const tables + fancy magics (D12)
│   ├── zobrist.rs           # the five key families (D9)
│   ├── movegen.rs           # pseudo-legal staged generation
│   ├── perft.rs             # perft/divide drivers
│   ├── see.rs               # impl Board { fn see_ge } (D11)
│   ├── eval.rs              # evaluate() seam (D22); becomes eval/ in phase B/C
│   ├── tt.rs                # transposition table (D19) + prefetch wrapper
│   ├── movepick.rs          # staged MovePicker, ExtMove (D7)
│   ├── history.rs           # gravity StatEntry + all history tables
│   ├── search/
│   │   ├── mod.rs           # iterative deepening, PVS node, qsearch, Stack
│   │   └── params.rs        # ALL tuned search constants (D20, §3.8)
│   ├── timeman.rs           # TimeManager + Limits (doc 12 §3.1)
│   ├── uci.rs               # stdin loop, command parsing, info output
│   ├── threads.rs           # ThreadPool, Worker, PoolShared (D23)
│   └── bench.rs             # fixed-suite bench (D24 regression oracle)
├── tests/
│   └── perft.rs             # integration perft suite (the movegen gate)
└── benches/                 # optional criterion micro-benches (dev-dep only)
```

Rules:

- Module names and boundaries follow D13 exactly. A module may be promoted
  from `foo.rs` to `foo/mod.rs + submodules` when it grows (e.g. `board/`,
  `eval/`), but its public API must not change in the same commit as the
  file move (no churn hidden inside reorganizations).
- `lib.rs` re-exports the vocabulary types at the crate root (as today:
  `Board`, `Bitboard`, `Move`, `MoveList`, `Square`, ...). Subsystem types
  (`TranspositionTable`, `MovePicker`, `Worker`) are used through their
  module path — they are not vocabulary.
- Dependency direction is acyclic and points "downward" only:

```
uci → threads → search → { movepick, tt, eval, timeman }
                movepick → { movegen, history, see }
                movegen  → { board, attacks }
                see      → { board, attacks }
                board    → { attacks, zobrist, bitboard, types, moves }
                perft/bench → { movegen, board } / { search stack }
```

  `types`, `bitboard`, `moves` are leaves; nothing below `board` may depend
  on anything above it. `eval` sees only `&Board` (+ its own tables) per D22.

- `main.rs` stays a one-line delegator (doc 12 §3.1 owns the entry point):

```rust
fn main() {
    chessengine::uci::run();
}
```

  `uci::run()` checks `std::env::args()` first: the argv subcommands
  `bench [...]` (doc 06 §3.6) and `perft <depth> [fen...]` (doc 06 §3.2)
  run to completion before the stdin loop would start; with no subcommand
  it enters the blocking UCI loop, which also accepts `bench`, `perft N`,
  and `d` as in-session commands (doc 12 §3.3).

### 3.2 End-to-end data flow

```
            stdin (GUI)
               │ one line per command
               ▼
┌─ UCI thread (uci::run) ─────────────────────────────────────────────┐
│  token dispatch, one cmd_* per command (doc 12 §3.3)                │
│  "position ..." : Board::from_fen + replay moves via make_move      │
│  "setoption ..." : update Options (Hash/Threads only while idle)    │
│  "go ..."       : pool.start_thinking(&board, limits, &tt)          │
│  "stop"         : pool.stop()  — relaxed store to the shared flag   │
└──────────────┬──────────────────────────────────────────────────────┘
               ▼
┌─ ThreadPool (threads) ───────────────────────────────────────────────┐
│  N persistent Workers parked on Mutex<Job> + Condvar loops (doc 13)  │
│  SHARED (and only this, per D23 as pinned by doc 13 §3.4):           │
│    Arc<TT> (relaxed atomics; handed per Search job)                  │
│    Arc<PoolShared> { stop/ponder/increase_depth AtomicBools, shells }│
│    per-thread atomic Counters (nodes, ...) aggregated on read        │
│  PER WORKER: Board clone (owns its StateInfo history Vec),           │
│    all history tables, RootMoves; Options stay on the UCI thread     │
│    (per-`go` state arrives via the Setup job)                        │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼  (main worker also owns TimeManager + UCI info output)
     search::iterative_deepening
               │ per depth / MultiPV line: aspiration window loop
               ▼
     search::search::<NodeType>  (PVS negamax) ⇄ search::qsearch
        │            ▲                │
        │ TT probe / │ TT store       │ next_move()
        ▼            │                ▼
       tt (shared) ──┘          movepick::MovePicker (staged)
        ▲                          │ generate::<GT>()      │ scores
        │ prefetch                 ▼                       ▼
        │                    movegen ──────────────► history tables
        │                          │ pseudo-legal moves   (per worker)
        │                          ▼
        └──────────── board::make_move / unmake_move ◄── legality, gives_check
                           │ put/remove_piece keep bitboards+mailbox;
                           │ key deltas computed inline in make_move (doc 04)
                           ▼
                attacks (leapers + magics)      eval::evaluate(&Board, &mut ctx, optimism)
                between/line/pinners queries         └─ called at eval points
```

Narrative of one search node (the hot path — panic-free per §3.4):

1. `search` probes the TT (`tt::probe`), possibly cutting off.
2. It builds a `MovePicker` with the TT move and its history-table refs.
3. `next_move()` lazily runs `movegen::generate::<GT>` into an internal
   `[ExtMove; MAX_MOVES]` buffer, scores with history tables, and yields
   pseudo-legal moves; `board.legal(m)` filters at play time.
4. `board.make_move(m, gives_check)` pushes a `StateInfo`, computes all five
   Zobrist key deltas inline (`put_piece`/`remove_piece` stay
   placement-only, doc 04 §3.3), recomputes checkers/pinners, and prefetches
   the child's TT cluster through the `MakeMoveHooks` seam.
5. Recurse; `board.unmake_move(m)` pops. Scores flow back up; TT stores on
   the way out; history updates on cutoffs.

The TT is the *only* mutable state shared during search (plus the stop flag
and aggregated node counters); everything else is worker-owned (D23).

### 3.3 Module inventory and public seams

Each entry: one-paragraph responsibility, then the boundary signatures.
Existing APIs (types/bitboard/board/fen/moves) are kept verbatim (D1).

#### 3.3.1 `types` (exists; grows)

Vocabulary: `Color`, `PieceType`, `Piece`, `Square` with `Display`/`FromStr`
and const constructors. Grows the search vocabulary per D5/D6: `Value` and
`Depth` type aliases with Stockfish-semantics constants and ply-relative
mate helpers, plus `Piece::index` for flat 0..12 table indexing (D4) and the
D11 piece values (`PAWN_VALUE..QUEEN_VALUE`, `types::PIECE_VALUES`,
`PieceType::value()` — doc 02 §3.9 owns the placement rationale). These
constants are facts from SF `types.h` and are our permanent semantics, not
tunables — they do NOT live in `params.rs`. Full layer design:
`design/02-core-types.md`.

```rust
pub type Value = i32;                                   // D5: alias, not newtype
pub const VALUE_DRAW: Value = 0;
pub const VALUE_MATE: Value = 32000;
pub const VALUE_INFINITE: Value = 32001;
pub const VALUE_NONE: Value = 32002;
pub const MAX_PLY: usize = 246;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value;  // 31754
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;
pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;                   // 31753
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY as Value;  // 31507
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY;

pub const fn mate_in(ply: i32) -> Value  { VALUE_MATE - ply }
pub const fn mated_in(ply: i32) -> Value { -VALUE_MATE + ply }
pub const fn is_valid(v: Value) -> bool  { v != VALUE_NONE }
pub const fn is_win(v: Value) -> bool    { v >= VALUE_TB_WIN_IN_MAX_PLY }
pub const fn is_loss(v: Value) -> bool   { v <= VALUE_TB_LOSS_IN_MAX_PLY }
pub const fn is_decisive(v: Value) -> bool { is_win(v) || is_loss(v) }

pub type Depth = i32;                                   // D6
pub const DEPTH_QS: Depth = 0;
pub const DEPTH_UNSEARCHED: Depth = -2;
pub const DEPTH_NONE: Depth = -3;

impl Piece {
    /// Flat 0..12 index: color * 6 + piece_type (D4). For history tables.
    pub const fn index(self) -> usize;
}
```

`value_to_tt(v, ply: i32)` / `value_from_tt(v, ply: i32, rule50: i32)`
(mate-score ply adjustment, D5) live here in `types` beside the mate helpers
— `design/02-core-types.md` §3.2 owns and unit-tests them; the TT and search
merely consume them. The *ply-relative convention* is global from day one.
Ply is `usize` where it indexes arrays, `i32` in value arithmetic (D6);
casts happen at the use site (`mate_in(ply as i32)`).

#### 3.3.2 `bitboard` (exists, frozen API)

`Bitboard` newtype over `u64`: set ops, `lsb`/`pop_lsb`, `IntoIterator` via
`BitboardIter`, file/rank constants. Only planned additions: directional
`shift::<const DIR>()` helpers for bulk pawn moves (owned by
`design/05-movegen.md`) and `more_than_one(self) -> bool`
(`b & (b-1) != 0`, the double-check test).

#### 3.3.3 `moves` (exists, frozen API)

`Move` packed u16 with OUR bit layout (D2), `MoveKind`, stack-allocated
`MoveList` (`[Move; 256]` + len, derefs to `[Move]`). Planned additions
(owned by `design/02-core-types.md` §3.7): `Move::raw16(self) -> u16` /
`Move::from_raw16(bits: u16) -> Move` (`pub const fn`; butterfly-history
indexing and TT move16 round-trips use the full 16-bit pattern) and
`Move::is_null(self) -> bool` (`== Move::NULL`).

#### 3.3.4 `board` (exists; the growth center)

Hybrid bitboards + mailbox position with `put_piece`/`remove_piece` as the
only placement mutators. The mutators stay *placement-only*: the five-key
Zobrist deltas (D9) are computed inline inside `make_move`, mirroring SF's
`do_move` (doc 04 §3.3 owns that decision), and `make_move` delegates to a
monomorphized `make_move_with::<H: MakeMoveHooks>` seam that carries TT
prefetch now and NNUE dirty-piece recording later. Grows: the `StateInfo`
history stack owned by the Board (D8), `make_move`/`unmake_move` (D10), null
moves, attack/legality/check queries, repetition and draw detection, and key
accessors. Full design in `design/04-position-state.md`; the seam other
modules program against:

```rust
impl Board {
    // Mutation (hot path; infallible by contract: m must be legal)
    pub fn make_move(&mut self, m: Move, gives_check: bool);   // D10
    pub fn unmake_move(&mut self, m: Move);
    pub fn make_null_move(&mut self);
    pub fn unmake_null_move(&mut self);

    // Keys (rule50-folded main key per D9)
    pub fn key(&self) -> u64;
    pub fn pawn_key(&self) -> u64;
    pub fn material_key(&self) -> u64;
    pub fn minor_key(&self) -> u64;
    pub fn non_pawn_key(&self, c: Color) -> u64;

    // Check & pin state (from the current StateInfo)
    pub fn checkers(&self) -> Bitboard;
    pub fn blockers_for_king(&self, c: Color) -> Bitboard;
    pub fn pinners(&self, c: Color) -> Bitboard;
    pub fn check_squares(&self, pt: PieceType) -> Bitboard;

    // Queries
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard;
    pub fn legal(&self, m: Move) -> bool;        // m must be pseudo-legal
    pub fn pseudo_legal(&self, m: Move) -> bool; // validates TT/external moves
    pub fn gives_check(&self, m: Move) -> bool;
    pub fn capture_stage(&self, m: Move) -> bool; // capture or queen promo
    pub fn is_draw(&self, ply: i32) -> bool;      // repetition + 50-move
    pub fn non_pawn_material(&self, c: Color) -> Value;
    pub fn rule50_count(&self) -> u32;
    pub fn ply_from_root(&self) -> usize;         // len of state stack
}
```

`Board: Clone + Send` — cloning copies the history Vec (D8); each Worker
clones the root Board once per `go`.

#### 3.3.5 `fen` (exists, frozen API)

`Board::from_fen`/`to_fen`, `FenError`, `START_FEN`, `Board::startpos()`.
Only change: `from_fen` will additionally initialize the first `StateInfo`
(keys, checkers, ...) once `board` grows state — same signature.

#### 3.3.6 `attacks` (new)

All precomputed attack knowledge. Leaper tables (pawn/knight/king),
`BETWEEN` (inclusive of the second square — directly usable as
block-or-capture mask), `LINE`, and Chebyshev distance are `const` tables
built by `const fn` at compile time. Slider attacks are fancy magic
bitboards — packed shared table (~841 KB), Carry-Rippler enumeration, epoch
verification, per-rank seeds — initialized once behind
`std::sync::LazyLock<AttackTables>` (D12). Design:
`design/03-attack-tables.md`.

```rust
pub fn pawn_attacks(c: Color, sq: Square) -> Bitboard;     // const table
pub fn knight_attacks(sq: Square) -> Bitboard;             // const table
pub fn king_attacks(sq: Square) -> Bitboard;               // const table
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard; // magic
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard;   // magic
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard;
pub fn attacks(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard; // not Pawn
pub fn between(a: Square, b: Square) -> Bitboard;  // strictly-between | {b}
pub fn line(a: Square, b: Square) -> Bitboard;     // full line or EMPTY
pub fn aligned(a: Square, b: Square, c: Square) -> bool;
pub fn distance(a: Square, b: Square) -> u32;
/// Accessor for the LazyLock'd tables (doc 03 §3.7); calling it once at
/// startup forces initialization, and hot loops may hoist the reference.
pub fn tables() -> &'static AttackTables;
```

#### 3.3.7 `zobrist` (new)

The five key families (D9) as a static generated by a `const fn` PRNG —
xorshift64\* seeded with 1070372, the same generator shape as
`attacks::Prng` (doc 03 §3.6); our key *values* need not match Stockfish's —
only internal consistency matters. Encodes the SF tricks as data: pawn psq
zeroed on ranks 1/8, castling keyed by the whole 4-bit mask, material-key
count slots reusing psq square indices 8..=17 of each row, `no_pawns` as the
pawn-key baseline. Rule50 folding is a `Board` helper (`adjust_key50`, doc
04 §3.5), not a zobrist export. Consumed only by `board`. Design:
`design/04-position-state.md` §3.4.

```rust
pub type Key = u64;                // re-exported from types

pub struct Tables {
    pub psq: [[Key; 64]; 12],      // indexed by Piece::index(); slots 8..=17
                                   //   of each row double as count slots
    pub en_passant: [Key; 8],      // by FILE of the ep square
    pub castling: [Key; 16],       // by the WHOLE 4-bit rights mask
    pub side: Key,
    pub no_pawns: Key,
}
pub static ZOBRIST: Tables = build_tables(); // const-evaluated
```

#### 3.3.8 `movegen` (new)

Pseudo-legal move generation, monomorphized per generation type with the
single target-mask parameter and bulk pawn shifts. `GenType` semantics
follow SF: `CAPTURES` includes queen promotions, `QUIETS` includes
underpromotions, `EVASIONS` for in-check, `NON_EVASIONS`, and a `LEGAL`
wrapper that filters lazily (only pinned/king/ep moves get the full check).
Because `adt_const_params` is not stable, the const parameter is a `u8` with
named constants. Design: `design/05-movegen.md`.

```rust
pub mod gen_type {
    pub const CAPTURES: u8 = 0;
    pub const QUIETS: u8 = 1;
    pub const EVASIONS: u8 = 2;
    pub const NON_EVASIONS: u8 = 3;
}
pub fn generate<const GT: u8>(board: &Board, list: &mut MoveList);
pub fn generate_legal(board: &Board, list: &mut MoveList);
```

#### 3.3.9 `perft` (new)

The correctness oracle (D24). Bulk-counting perft (`generate_legal` at
leaf-1), an invariant-checking `perft_checked` variant, and a `divide` for
debugging. Called from the UCI loop (`go perft N` / `perft N`), from the
argv convenience subcommand `chessengine perft <depth> [fen...]` (§3.1),
and from `tests/perft.rs`. Design: `design/06-perft-validation.md`.

```rust
pub fn perft(board: &mut Board, depth: u32) -> u64;
pub fn perft_checked(board: &mut Board, depth: u32) -> u64; // + invariants
pub struct DivideEntry { pub mv: Move, pub nodes: u64 }
pub struct DivideResult { pub entries: Vec<DivideEntry>, pub total: u64 }
pub fn divide(board: &mut Board, depth: u32) -> DivideResult;
```

#### 3.3.10 `see` (new)

Static exchange evaluation as a `Board` method (D11) in its own file:
`src/see.rs` contains `impl Board { pub fn see_ge(...) }` — satisfies both
"SEE is a Board method" (D11) and "see is a module" (D13). Swap algorithm
with x-rays and pinned-piece exclusion. SEE piece values (SF-current,
`types.h`): P 208, N 781, B 825, R 1276, Q 2538 — defined as
`types::PIECE_VALUES` with `PieceType::value()` (doc 02 §3.9 owns the
placement; eval phase A and MVV ordering consume the same array). Design:
`design/04-position-state.md`.

```rust
impl Board {
    /// True if the exchange sequence started by m nets >= threshold.
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool;
}
```

#### 3.3.11 `eval` (new; later a directory)

The single evaluation seam (D22). Phase A: material + tapered PSQT. Phase B:
hand-crafted terms + texel tuning. Phase C: NNUE with incremental
accumulators. Search never knows which phase is behind the function.
Optimism is threaded through from day one (0 until the search sets it);
correction history adjusts the *returned* value inside search, never inside
`evaluate` (raw eval is what the TT caches). Design:
`design/11-evaluation.md` (NNUE: its §3.4).

```rust
pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value) -> Value;  // D22
```

`EvalContext` is the per-thread mutable eval state (Phase B pawn-structure
hash, Phase C NNUE accumulator stack), owned by each `Worker` (§3.3.16, D23)
and threaded through from day one: it cannot hide behind `&Board` without
interior mutability or `thread_local!`, both banned on the search path
(design/13 §3.12). Phase A ignores it. Borrow-wise the call splits disjoint
`Worker` fields — `evaluate(&self.board, &mut self.eval_ctx, optimism)`.

#### 3.3.12 `tt` (new)

The shared transposition table (D19): 10-byte entries (key16, depth8,
gen_bound8, move16, value16, eval16), every field a relaxed atomic, 3-entry
32-byte clusters, `mul_hi64` indexing, 5-bit generations, SF replacement
policy. Lock-free and racy by design; all consumers validate what they read.
Also owns the crate's only prefetch wrapper (sanctioned unsafe, §3.5).
`value_to_tt`/`value_from_tt` live in `types` (doc 02 §3.2, see §3.3.1) —
this module only consumes them. Design: `design/08-transposition-table.md`.

```rust
pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    generation8: AtomicU8,   // interior-mutable: new_search takes &self (D23)
}
pub struct TtData {
    pub mv: Move, pub value: Value, pub eval: Value,
    pub depth: Depth, pub bound: Bound, pub is_pv: bool,
}
pub struct TtWriter<'tt> { /* &'tt TtEntry chosen by probe */ }
/// `hit` is load-bearing: a tag-matching but unoccupied entry returns
/// hit == false with a writer into that same slot (doc 08 §3.5) —
/// an Option cannot express that, hence a struct.
pub struct Probe<'tt> { pub hit: bool, pub data: TtData, pub writer: TtWriter<'tt> }

impl TranspositionTable {
    pub fn resize(&mut self, mb: usize) -> usize; // &mut: UCI thread only,
    pub fn clear(&mut self);                      //   while idle, via
                                                  //   Arc::get_mut (§3.3.16)
    pub fn new_search(&self);                 // ++generation (5-bit wrap),
                                              //   relaxed AtomicU8 store
    pub fn generation(&self) -> u8;
    pub fn probe(&self, key: u64) -> Probe<'_>;
    pub fn prefetch(&self, key: u64);
    pub fn hashfull(&self, max_age: u8) -> u32;   // permille
}
impl TtWriter<'_> {
    pub fn write(&self, key: u64, value: Value, is_pv: bool, bound: Bound,
                 depth: Depth, mv: Move, eval: Value, generation: u8);
}
```

#### 3.3.13 `movepick` (new)

The staged move-ordering state machine over one `[ExtMove; MAX_MOVES]`
buffer with index-based regions (indices, not pointers). Emits one
pseudo-legal move per `next_move()`; if the search cuts off on the TT move,
nothing is ever generated. Stages per SF: TT → good captures (SEE-gated) →
good quiets (partial insertion sort) → bad captures → bad quiets; separate
stagings for evasions, qsearch, ProbCut. `ExtMove { mv: Move, value: i32 }`
is `#[repr(C)]`, asserted 8 bytes, and lives here, not in `MoveList` (D7).
No killer/countermove stages, ever (D17). The picker **holds no borrows**:
`&Board` and `&Histories` are passed into each call, so search can freely
make/unmake and update histories between calls (doc 09 §3.5 owns this
resolution of SF's `const Position&` member). Design:
`design/09-move-ordering-history.md`.

```rust
#[repr(C)]
pub struct ExtMove { pub mv: Move, pub value: i32 }
const _: () = assert!(std::mem::size_of::<ExtMove>() == 8);

pub struct MovePicker { /* stage, ttm, depth, ply, cursors,
                           cont_hist: [ContHistId; 6], buf — owned, no refs */ }
impl MovePicker {
    /// Main search (depth > 0), qsearch (depth <= 0), evasions (in check).
    pub fn new(board: &Board, tt_move: Move, depth: Depth, ply: usize,
               cont_hist: [ContHistId; 6]) -> Self;
    /// ProbCut: captures with see_ge(threshold).
    pub fn new_probcut(board: &Board, tt_move: Move, threshold: Value) -> Self;
    pub fn next_move(&mut self, board: &Board, hist: &Histories) -> Option<Move>;
    pub fn skip_quiet_moves(&mut self);          // LMP hook
}
```

#### 3.3.14 `history` (new)

All heuristic tables and the single gravity update primitive
(`entry += clamp(b,±D) - entry*|clamp(b,±D)|/D`), generic over the bound D.
Butterfly (D=7183), capture (D=10692), continuation (D=30000), pawn
(D=8192), low-ply (D=7183), correction (D=1024) — the D bounds are
SF-current structural const generics living in `history.rs` (doc 09 §3.4);
the scoring/update *multipliers* are named constants in `search/params.rs`
(§3.8). Big tables are heap-boxed inside `Histories`, which is per-Worker
(no sharing, no atomics, per D23). Design:
`design/09-move-ordering-history.md`.

```rust
pub struct StatEntry<const D: i32>(i16);
impl<const D: i32> StatEntry<D> {
    pub fn get(self) -> i32;
    pub fn update(&mut self, bonus: i32);   // the gravity rule
}
pub struct Histories { /* Box'ed tables: main, low_ply, capture, cont,
                          pawn, corr, cont_corr + tt_move scalar */ }
impl Histories {
    pub fn new() -> Box<Self>;      // tables never transit the stack
    pub fn clear(&mut self);        // per-`ucinewgame` init (fill) values
    pub fn new_search(&mut self);   // each `go`: main decay 789/1024, ...
}
```

#### 3.3.15 `search` (new; `mod.rs` + `params.rs`)

Iterative deepening with aspiration windows, the PVS negamax node, qsearch,
and all pruning/extension/reduction machinery (skeleton owned by
`design/07-search-core.md`, techniques by `design/10-pruning-extensions.md`).
The per-ply stack is a worker-owned fixed array
`Box<[Stack; STACK_SIZE]>` (`STACK_SIZE = MAX_PLY + 10`) and the search runs
at offset `SS_OFFSET = 7` so `ss-7 ..= ss+2` are always in bounds as safe
slice indexing (D18). `Stack` is `Copy` and small (target ≤ 40 bytes, doc 07
§3.2.3); **PV lines do NOT live on the stack** — they live in a worker-owned
per-ply table `pv_table: Box<[PvLine; MAX_PLY + 1]>` of fixed arrays,
updated by borrow-splitting `update_pv` (doc 07 §3.3), so nothing on the
search path allocates. Continuation-history "pointers" are `ContHistId` flat
indices (doc 09 §3.2), not references — borrow-checker-friendly, same speed.
No killers (D17). `Limits` is defined in `timeman` (doc 12 §3.6 owns it, see
§3.3.17); search consumes a per-`go` copy.

```rust
pub const SS_OFFSET: usize = 7;              // D18
pub const STACK_SIZE: usize = MAX_PLY + 10;  // 256

#[derive(Clone, Copy)]
pub struct Stack {
    pub ply: i32,
    pub current_move: Move,       // Move::NULL when none (D2)
    pub excluded_move: Move,      // singular-search exclusion
    pub static_eval: Value,       // corrected eval; VALUE_NONE when unset
    pub stat_score: i32,
    pub move_count: i32,
    pub in_check: bool,
    pub tt_pv: bool,              // sticky "was ever on the PV"
    pub tt_hit: bool,
    pub cutoff_cnt: i32,
    pub reduction: i32,           // 1024ths of a ply
    pub follow_pv: bool,          // "on last iteration's PV path"
    pub cont_hist: ContHistId,    // doc 09 §3.2 owns the representation
}

pub struct RootMove {             // doc 07 §3.4 owns the full definition
    pub pv: Vec<Move>,            // root is cold; Vec is fine here
    pub previous_pv: Vec<Move>,
    pub score: Value, pub previous_score: Value, pub uci_score: Value,
    pub average_score: Value, pub mean_squared_score: i64,
    pub score_lowerbound: bool, pub score_upperbound: bool,
    pub previous_score_exact: bool,
    pub sel_depth: i32, pub effort: u64,
}

impl Worker {                     // Worker itself lives in threads (§3.3.16)
    /// Driven from Worker::start_searching (doc 13 §3.7).
    pub fn iterative_deepening(&mut self);
}
```

`search/params.rs` is specified in §3.8.

#### 3.3.16 `threads` (new)

Lazy SMP (D23). N **long-lived worker threads** parked on per-thread
`Mutex<Job> + Condvar` loops, created once per `setoption Threads` / engine
start and woken with jobs — SF's lifecycle, pinned by doc 13 §3.4 as D23's
mechanism (`std::thread::scope` + `&TT` remains the documented fallback,
not chosen). Shared state is exactly: the TT behind `Arc<TT>` (handed to
each worker *per Search job*, never stored), the `Arc<PoolShared>` flag
block (stop/ponder/increase_depth `AtomicBool`s + the thread shells), and
per-thread atomic `Counters` aggregated on read. TT resize/clear require
`&mut` and happen only between searches, on the UCI thread, via
`Arc::get_mut` (all job `Arc` clones are dropped while workers are idle);
the per-`go` generation bump goes through `new_search(&self)` over an
interior-mutable `AtomicU8` (doc 08 §3.9). Main worker (index 0) owns the
`SearchManager` (TimeManager + reporting) and prints UCI output; helpers
just search until `stop`. Best-thread voting at the end. Design:
`design/13-threading-smp.md`.

```rust
pub enum Job {
    Setup(Box<SetupJob>),   // install per-go state on the worker's thread
    Search { tt: Arc<TT> }, // run Worker::start_searching(&tt)
    Clear,                  // ucinewgame
    Exit,
}
pub struct PoolShared {
    pub shells: Vec<Arc<ThreadShell>>,  // index == thread_idx; 0 = main
    pub stop: AtomicBool,
    pub increase_depth: AtomicBool,
    pub ponder: AtomicBool,
}
pub struct Worker {
    pub thread_idx: usize,
    pub root_board: Board,              // clone incl. history Vec (D8)
    pub root_moves: Vec<RootMove>,
    pub limits: Limits,                 // per-go copy
    pub histories: Box<Histories>,      // ~20 MiB, persists across `go`
    pub reductions: Box<[i32; MAX_MOVES]>, // built in Worker::new (§3.8)
    pub counters: Arc<Counters>,        // nodes, ... — relaxed atomics
    pub pool: Arc<PoolShared>,
    pub manager: Option<Box<SearchManager>>, // main worker only
    /* per-search scalars, stack array, pv_table, ... (doc 13 §3.2) */
}
pub struct ThreadPool { /* Arc<PoolShared> + join handles */ }
impl ThreadPool {
    pub fn set(&mut self, n: usize);             // (re)create pool, idle only
    pub fn start_thinking(&mut self, board: &Board, limits: Limits,
                          tt: &Arc<TT>);         // non-blocking
    pub fn stop(&self);
    pub fn ponderhit(&self);
    pub fn wait_for_search_finished(&self);      // waits on main worker only
    pub fn clear(&self);                         // ucinewgame
    pub fn nodes_searched(&self) -> u64;         // Σ shells[i].counters.nodes
}
```

#### 3.3.17 `timeman` (new)

Time allocation (optimum/maximum split with SF's fitted formulas) and the
soft/hard stop conditions; consumed by the main worker only. Also home of
the `Limits` type shared by `uci`, `search`, and `threads` (doc 12 §3.6
owns it): times are `i64` milliseconds with 0-as-absent (SF convention —
required so nodestime can rewrite them in place, and tolerant of negative
`wtime`). Design: `design/12-uci-time-management.md`.

```rust
pub struct Limits {
    pub time: [i64; 2],       // wtime/btime ms, 0 = not given (SF convention)
    pub inc: [i64; 2],        // winc/binc
    pub movestogo: i32,       // 0 = sudden death / not given
    pub depth: Option<Depth>,
    pub nodes: Option<u64>,
    pub movetime: Option<i64>,
    pub mate: Option<i32>,
    pub infinite: bool,
    pub ponder: bool,
    pub searchmoves: Vec<Move>,   // empty = all root moves
    pub perft: Option<u32>,       // go perft d (dev)
    pub start_time: Instant,      // stamped when parsing begins
}

pub struct TimeManager { /* start_time: Instant, optimum_ms, maximum_ms: i64,
                            nodestime state */ }
impl TimeManager {
    /// May rewrite limits.time/inc under nodestime (doc 12 §3.11).
    pub fn init(&mut self, limits: &mut Limits, us: Color, ply: i32,
                options: &Options, persist: &mut TimePersist);
    pub fn optimum(&self) -> i64;   // ms (or nodes under nodestime)
    pub fn maximum(&self) -> i64;
    pub fn elapsed(&self, nodes: impl Fn() -> u64) -> i64;  // TM-elapsed
    pub fn elapsed_ms(&self) -> i64;                        // wall clock
}
```

#### 3.3.18 `uci` (new)

The blocking stdin loop and protocol glue: per-command token dispatch
(doc 12 §3.3 — no monolithic command enum; each `cmd_*` parses its own
tail), `position`/`go` handling, `setoption` bookkeeping, `info` line
formatting (score normalization lives here), and the engine-extension
commands `bench`, `perft N`, `d` (display). Owns the `Options` registry and
the `Engine` glue aggregate (pool + `Arc<TranspositionTable>` + options +
session board). Never panics on input: unrecognized/malformed commands
print `info string ...` and are ignored (GUIs send garbage; a dead engine
loses on time); `debug` and `register` are consumed silently per spec.
Design: `design/12-uci-time-management.md`.

```rust
pub fn run();   // argv subcommands (§3.1), then the blocking stdin loop

pub struct Engine {               // doc 12 §3.1 owns the glue
    pub pool: ThreadPool,
    pub tt: Arc<TranspositionTable>,
    pub options: Options,
    pub board: Board,             // session position set by `position`
}

/// Resolve UCI move text against the current position (handles castling,
/// promotion, ep encodings). None = not a legal move here.
pub fn move_from_uci(board: &Board, token: &str) -> Option<Move>;
/// Registration-ordered option table (Hash, Threads, MultiPV,
/// Move Overhead, Ponder, ...) — doc 12 §3.5.
pub struct Options { /* entries: Vec<OptionEntry> */ }
```

#### 3.3.19 `bench` (new)

The regression oracle (D24): a fixed suite of FENs searched at a fixed depth
with a fixed Hash/Threads=1 configuration, printing per-position and total
node counts + NPS. The total node count is the *bench signature* every
functional commit must report; non-functional commits must not change it.
Suite content, argument grammar, and output format:
`design/06-perft-validation.md` §3.6; measurement discipline and targets:
`design/14-performance-testing.md`.

```rust
// Types normative in design/06-perft-validation.md §3.6 (BenchParams
// carries tt_mb, threads, limit: BenchLimit, fens: FenSource).
pub struct BenchResult { pub nodes: u64, pub elapsed_ms: u64 }
pub fn run(params: &BenchParams) -> BenchResult;
```

### 3.4 Error-handling strategy

Three zones with different rules:

1. **Parse boundaries — `Result` with real error types.** Everything that
   converts text to engine types returns `Result<_, E>` where
   `E: std::error::Error` (existing: `FenError`, `ParseSquareError`,
   `ParseCastlingRightsError`). The UCI layer's per-command parsers (doc 12
   §3.3) follow the same rule internally and surface failures as absorbed
   `info string` replies, not process exits. No panicking parsers, no
   `unwrap` on external input. This is already the convention in
   `fen.rs`/`types.rs` — keep it.

2. **The UCI loop — absorb, never die.** A chess engine that exits on bad
   input forfeits games. `uci::run` catches every parse error, replies
   `info string unknown command: ...`, and continues. I/O errors on stdin
   (EOF) terminate cleanly like `quit`. The only intentional process exits
   are `quit` and fatal startup failures (e.g. TT allocation failure —
   print a diagnostic to stderr, exit nonzero).

3. **Hot paths (movegen, make/unmake, search, eval, TT) — panic-free by
   construction, checked by `debug_assert!`.** These functions are total
   over their stated preconditions ("`m` is legal", "`sq` came from a
   bitboard of this position") and those preconditions are enforced by the
   type system where possible (`Square` can't be out of range; `Move`
   getters can't fail) and by `debug_assert!` where not. No `Result`, no
   `Option` unwrapping on values that can't be `None` in correct code —
   restructure so the invariant is visible to the compiler instead (e.g.
   iterate bitboards rather than index-and-check).

`debug_assert` policy, concretely:

- **Cheap invariant, cold code** → plain `assert!` (e.g. `Square::new`,
  option parsing). Cost is irrelevant; failures are user-visible bugs.
- **Cheap invariant, hot code** → `debug_assert!` (e.g. "moved piece belongs
  to side to move" inside `make_move`). Perft and the test suite run in
  debug regularly, so these fire in practice.
- **Expensive whole-structure validation** → dedicated `#[cfg(test)]` or
  debug-only functions called from tests and the perft harness
  (`Board::assert_consistent` is the model; a future
  `assert_state_consistent` will cross-check incremental keys against
  from-scratch recomputation).
- Existing hard `assert!(self.len < MAX_MOVES)` in `MoveList::push` stays
  until movegen is perft-verified; downgrading it to `debug_assert!` is a
  measured decision later (§9).

Release builds use `panic = "abort"` (D16): any panic in a worker thread
kills the process instead of unwinding into a poisoned half-alive engine.
That is the correct behavior for an engine — a panic means a corrupted
search; there is nothing sane to recover to. Consequently we never rely on
`catch_unwind`, and `Mutex` poisoning is a non-concern in release.

### 3.5 Unsafe policy application (D14)

Crate root: `#![deny(unsafe_code)]` in `lib.rs`. Modules with a sanctioned
exemption opt back in with a module-level `#![allow(unsafe_code)]` plus a
header comment naming the D14 clause they use. `deny` (not `forbid`) is
chosen precisely so the exemption is possible but must be visible and
grep-able (`grep -rn "allow(unsafe_code)" src/` enumerates every exempted
module; `grep -rn "unsafe" src/` every block).

Sanctioned sites and where they live:

| D14 clause | Module | Shape |
|---|---|---|
| (a) prefetch intrinsics | `tt` (only) | `TranspositionTable::prefetch` wrapping `_mm_prefetch` on x86_64, no-op elsewhere; later reused for history prefetch via a tiny `pub(crate) fn prefetch_ptr<T>(p: *const T)` in `tt` |
| (b) SIMD in eval/NNUE | `eval/` (phase C) | quantized inference kernels with a scalar safe fallback selected at compile time |
| (c) proven-hot unchecked indexing | case-by-case | only AFTER a profile shows the bounds check; the `// SAFETY:` comment must cite the measurement (what was measured, when, how much it saved) |

Everything else is safe Rust. Explicitly safe by design: TT relaxed atomics
(D19 — `AtomicU16/AtomicU8/AtomicI16` with `Ordering::Relaxed` are safe
code), the search stack offset trick (D18 — safe slice indexing on a
fixed-size array), and `LazyLock` attack tables (D12).

### 3.6 Dependency policy (D15)

- `[dependencies]` in `Cargo.toml` stays **empty**. Std-only. The check is
  objective: `cargo tree --edges normal` prints exactly one line
  (`chessengine v...`).
- `[dev-dependencies]` allowed where they pay for themselves; expected:
  `criterion` for micro-benches. Nothing that generates code into the
  library.
- External *tools* (not dependencies): `fastchess`/`cutechess-cli` for SPRT
  (D25), `bullet` for NNUE training (D21). They never appear in the build
  graph.
- Anything platform-specific (huge pages via `madvise` for the TT, D19
  territory) is done through `std` where possible; if it ever genuinely
  needs `libc`, that is a decisions.md amendment first, not a quiet Cargo
  edit.

### 3.7 Build profiles and flags (D16)

`Cargo.toml`:

```toml
[package]
name = "chessengine"
version = "0.1.0"
edition = "2024"

[dependencies]
# (intentionally empty — D15)

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"

# Profiling builds: release codegen + symbols. `perf` / flamegraphs read this.
[profile.profiling]
inherits = "release"
debug = "line-tables-only"   # symbols for perf/flamegraph; no codegen change
```

Notes:

- `panic = "abort"` is ignored by Cargo for `test`/`bench` profiles, so
  `cargo test` and criterion keep working unchanged.
- **The documented fast configuration** (used for all strength testing,
  bench signatures, and SPRT runs):

  ```sh
  RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
  # or, for a machine-local binary:
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  ```

  `x86-64-v3` guarantees POPCNT/BMI2/AVX2 so `count_ones`,
  `trailing_zeros`, and the magic-index arithmetic compile to single
  instructions. We deliberately do **not** commit a `.cargo/config.toml`
  with `target-cpu`: default builds stay portable; the fast flags are an
  explicit, documented step (this section + CLAUDE.md).
- Profiling runs additionally want frame pointers:
  `RUSTFLAGS="-C target-cpu=x86-64-v3 -C force-frame-pointers=yes" cargo build --profile profiling`.
- Bench-signature discipline (D24) assumes a fixed build recipe: node counts
  are compared across commits built with the same profile + RUSTFLAGS.

**Supported platforms.** Two tiers, so a "competitive" engine's testers and
GUI users (who largely live on Windows/macOS and increasingly aarch64) are
not silently broken by Linux-only assumptions:

- **Tier 1 — Linux x86-64 (x86-64-v3)**: the development and strength-
  testing platform; all performance targets (§6), profiling recipes, and
  bench/SPRT discipline apply here.
- **Tier 2 — Windows and macOS x86-64, and aarch64 (Apple Silicon /
  Linux)**: must build, pass `cargo test`, and play legal games; no
  performance targets. The code is written portably by construction —
  prefetch is cfg-gated to a no-op off x86_64 (§3.5), the TT uses portable
  `Vec` allocation, and everything else is std-only — and the objective
  gate in §5 keeps it that way.

### 3.8 `params.rs` — tuned constants convention (D20)

All *tuned* search constants live in **`src/search/params.rs`** as named
`pub const` items — no magic numbers inline, anywhere. "Tuned" means:
Stockfish arrived at the value by SPSA/fishtest and we adopt it as a
starting point. Structural constants (bit widths, `VALUE_MATE`, array
sizes) are semantics, not tunables — they live with their types.

Conventions:

- **Provenance header**: the file opens with a comment pinning the source:
  `// Values are SF-current starting points from Stockfish commit eca43a97
  (2026-07-06) unless noted. See stockfish_reference/stockfish_search_notes.md.`
  Individual constants only carry extra comments when they deviate from
  that baseline or need shape context.
- **Naming**: `TECHNIQUE_ROLE` screaming-snake, grouped by technique in the
  order of `design/07-search-core.md`/`design/10-pruning-extensions.md`
  sections. The *formula shape* is documented
  at the use site in `search`, citing the design doc section; the constant
  documents only its role.
- **Tables too**: derived tables are regenerable from named constants so a
  tuner can move them. Integer tables (the 16-entry LMR-divisor table) are
  plain `const`. The `ln`-based reduction table **cannot be `const` on
  stable** (`f64::ln` is not const-callable — E0015; the stabilized const
  float methods cover only trivial ops): `params.rs` holds the named scale
  constants plus a runtime builder `pub fn build_reductions() -> [i32;
  MAX_MOVES]`, called once from `Worker::new` into a worker-owned
  `reductions: Box<[i32; MAX_MOVES]>` (docs 10 §3.3, 13 §3.2) —
  deterministic, once per worker, no `LazyLock` needed.
- **Tunability**: constants are plain `pub const` for now (zero-cost). If we
  later add SPSA support, `params.rs` is the single file a
  `#[cfg(feature = "tune")]` mechanism would rewrite — that is the point of
  centralizing.

Excerpt of the intended shape (values are facts from SF `eca43a97`;
constant names and types follow the owning inventories in docs 07/10 —
this excerpt amended to match them on 2026-07-10, when P0 seeded the
file):

```rust
//! Tuned search parameters. SF-current starting points (D20),
//! Stockfish commit eca43a97 (2026-07-06).

use crate::types::{Depth, Value};

// NOTE on what does NOT live here: piece values (P 208 .. Q 2538) are
// semantics in types::PIECE_VALUES (doc 02 §3.9); history gravity bounds
// (7183/10692/30000/8192/1024) are structural const generics in
// history.rs (doc 09 §3.4). Only their *multipliers* are tunables here.

// --- Aspiration windows (doc 07 §3.6; search.cpp) ---
pub const ASPIRATION_BASE_DELTA: Value = 5;      // + thread jitter + |mss|/10588
pub const ASPIRATION_MSS_DIVISOR: i64 = 10588;
pub const ASPIRATION_WIDEN_NUM: Value = 44;      // delta += delta * 44 / 128
pub const ASPIRATION_WIDEN_DEN: Value = 128;

// --- Reduction table (doc 10 §3.3): reductions[i] = (2834/128) * ln(i),
//     i in 1..MAX_MOVES. Built at RUNTIME (f64::ln is not const on
//     stable), once per Worker.
pub const RED_TABLE_NUM: i32 = 2834;
pub const RED_TABLE_DEN: i32 = 128;
pub fn build_reductions() -> [i32; MAX_MOVES] { /* runtime ln loop */ }

// --- Razoring (doc 10 §3.1): qsearch if eval < alpha - 465 - 300*depth^2 ---
pub const RAZOR_BASE: Value = 465;
pub const RAZOR_DEPTH_MULT: Value = 300;
// ... (full inventory owned by designs 07, 09, and 10)
```

The rule of thumb for reviewers: **any bare numeric literal in
`search/`, `movepick.rs`, or `see.rs` that a tuner could move is a defect**
— it must be a named `params` constant.

### 3.9 How design docs, reference notes, and code relate

Three layers with a strict flow:

1. **`design/decisions.md`** — the constitution. Numbered, terse, binding.
   Docs and code cite decision numbers. Changing a decision is its own
   commit with rationale.
2. **`design/NN-topic.md`** — the build plan, one doc per component,
   numbered in rough dependency order (§7 lists them). Each contains the
   full algorithmic design, constants, implementation steps, and acceptance
   criteria (D26). *Docs lead code*: when implementation forces a deviation,
   the doc is updated in the same commit — a doc that disagrees with the
   code is a bug in the doc.
3. **`stockfish_reference/*.md`** — study notes: provenance for every idea
   and constant we adopted, with SF file/line references, indexed by
   `NOTES.md`. Notes are *descriptive* (what SF does), design docs are
   *prescriptive* (what we build). When unsure during implementation: read
   the design doc → if insufficient, the note → if still unsure, the SF
   source itself, then extend the note with what was learned. Constants
   cited anywhere trace to SF commit `eca43a97` (2026-07-06) unless stated
   otherwise. We never copy SF code; ideas, constants, and formulas only
   (D27) — the clone under `stockfish_reference/Stockfish/` is gitignored
   and read-only.

CLAUDE.md remains the day-to-day working conventions summary; where it and
this doc overlap, they must agree, and this doc carries the detail.

---

## 4. Implementation plan

Commit-sized, in order. (This plan covers only the architecture scaffolding;
components land under their own docs' plans.)

1. **Build profiles + fast-config docs.** Add the `[profile.release]` and
   `[profile.profiling]` blocks from §3.7 to `Cargo.toml`; document the
   RUSTFLAGS recipes in CLAUDE.md.
   *Done when:* `cargo build --release` and
   `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release` succeed;
   `cargo test` and `cargo clippy --all-targets` still pass clean.
2. **Crate hygiene.** Add `#![deny(unsafe_code)]` and a crate-level doc
   comment (goal, decision-record pointer) to `lib.rs`.
   *Done when:* clippy clean; `grep -c "deny(unsafe_code)" src/lib.rs` = 1.
3. **`types` growth (D4/D5/D6/D11).** Add `Value`/`Depth` aliases, the
   constant bands, `mate_in`/`mated_in`/`is_*` predicates,
   `value_to_tt`/`value_from_tt`, `types::PIECE_VALUES` +
   `PieceType::value()`, and `Piece::index`, with unit tests (band ordering,
   predicate edges at `VALUE_TB_WIN_IN_MAX_PLY`, TT roundtrips,
   `Piece::index` covers 0..12 bijectively) — details per
   `design/02-core-types.md`.
   *Done when:* tests pass; constants match §3.3.1 exactly.
4. **`search/params.rs` seed.** Create the module with the provenance
   header and the first search constants (aspiration/razoring shape of
   §3.8). Piece values land in `types` (step 3); history gravity bounds are
   const generics in `history.rs` (doc 09 §3.4) — neither goes here.
   *Done when:* file exists with header convention; `cargo doc` renders it;
   no other module defines a tuned constant.
5. **Thin-bin + UCI skeleton.** `main.rs` per §3.1 (one-line delegator);
   `uci::run` handling `uci`/`isready`/`quit`/unknown-command absorption
   (full protocol lands with `design/12-uci-time-management.md`).
   *Done when:* `printf 'uci\nisready\nquit\n' | cargo run -q` prints
   `id name ...`, `uciok`, `readyok` and exits 0; garbage lines produce
   `info string` and do not exit.
6. **Module stubs only as their docs land** — we do NOT pre-create empty
   files for `attacks`, `tt`, etc.; each arrives with its component per its
   own doc's plan (next: `design/03-attack-tables.md`).
   *Done when:* n/a (policy step, recorded here).

---

## 5. Testing and validation

- **Unit tests** (this doc's scope): `types` value-band tests (step 3);
  UCI command-dispatch tests including malformed input (surface owned by
  `design/12-uci-time-management.md` §5); the step-5 smoke test.
- **Layout/policy gates** (cheap, objective, run in CI or by hand):
  - `cargo test` and `cargo clippy --all-targets` clean (CLAUDE.md gate).
  - `cargo tree --edges normal | wc -l` == 1 (D15).
  - `grep -rn "unsafe" src/ | grep -v "SAFETY:" | grep -v deny` finds
    nothing unexplained (D14).
  - Size pins compiled as `const _: () = assert!(...)`:
    `size_of::<Move>() == 2` (exists as test — promote to const assert),
    `size_of::<ExtMove>() == 8` (D7), TT cluster `== 32` (D19) as those
    types land.
  - Tier-2 platform gate (§3.7): `cargo check --target
    x86_64-pc-windows-msvc` and `cargo check --target aarch64-apple-darwin`
    (or equivalent installed cross-targets) stay green — catches
    Linux-/x86-only assumptions at compile time.
- **Cross-cutting gates owned elsewhere but anchored here** (D24): the
  perft suite (`tests/perft.rs`) gates every movegen/make-unmake commit;
  the `bench` node-count signature gates every search-affecting commit.
  Definitions live in `design/06-perft-validation.md` and
  `design/14-performance-testing.md`; the *policy* — no merge without them
  — is architectural and restated here on purpose.
- **Regression strategy for this component**: architecture regressions are
  API/layout drift. The seam signatures in §3.3 are the contract; a PR that
  changes one must edit this doc in the same commit (checked in review;
  greppable because signatures here are real Rust).

---

## 6. Performance notes

- **Targets**: startup (attack-table init) well under 100 ms (D12 allows
  revisiting hardcoded magics if ever exceeded); perft(startpos) throughput
  in the hundreds of millions of nodes/s with bulk counting (gate in
  design/06); search NPS tracked per-commit via `bench` from the first
  search commit onward. Per-node cost parity with Stockfish's data
  structures is the standing constraint (project goal #2).
- **What to measure, from day one**: bench NPS (release + documented
  RUSTFLAGS, single thread), perft NPS, and — when investigating — `perf
  stat` cache-miss/branch-miss rates on the `profiling` build.
- **Pitfalls this architecture already guards against**:
  - *Bounds checks*: fixed-size arrays indexed by `Square::index() as usize`
    (0..64 by construction) are elidable; where the optimizer fails, D14(c)
    applies — measure first.
  - *LazyLock deref cost*: `attacks::*` free functions deref a `LazyLock`
    per call. LLVM hoists the initialized check in hot loops, but movegen
    should bind `let tables = attacks::tables();` once per generation call
    if profiles show otherwise (seam allows adding such an accessor without
    breaking callers).
  - *Allocation on the hot path*: forbidden by convention (CLAUDE.md).
    `MoveList`/ExtMove buffers are stack arrays; PV storage is the
    worker-owned fixed `pv_table` of `PvLine` arrays (doc 07 §3.3 — no
    per-ply Vecs); `StateInfo` Vec is `reserve`d to `MAX_PLY + game length`
    so `make_move` never reallocates mid-search.
  - *False sharing*: per-Worker data is worker-owned (D23); the only shared
    hot memory is the TT, whose 32-byte clusters are half-cache-line by
    design (D19).
  - *fat LTO + codegen-units=1*: makes cross-module `#[inline]` hints mostly
    unnecessary in release — do not scatter `#[inline(always)]`; add only
    with a measurement, like unsafe.
- **Build-flag sensitivity**: node-count signatures are flag-independent
  (search is deterministic single-threaded), but NPS claims are only
  comparable at identical profile + RUSTFLAGS (§3.7).

---

## 7. Interactions

This doc depends on `design/decisions.md` and feeds every other doc. The
suite, in rough build order (numbers = filenames):

| Doc | Owns | Consumes from this doc |
|---|---|---|
| `02-core-types.md` | Value/Depth bands, mate helpers, `value_to/from_tt`, `PIECE_VALUES`, `Move::raw16` | `types`/`moves` seams (§3.3.1/3) |
| `03-attack-tables.md` | leaper const tables, magics, BETWEEN/LINE | `attacks` seam (§3.3.6), unsafe/dep policy |
| `04-position-state.md` | StateInfo, make/unmake, Zobrist, SEE, draw rules | `board`/`zobrist`/`see` seams (§3.3.4/7/10) |
| `05-movegen.md` | GenType generation, legality predicates | `movegen` seam (§3.3.8), MoveList |
| `06-perft-validation.md` | perft/divide, validation tiers, bench suite + CLI (D24) | `perft`/`bench` seams (§3.3.9/19), gates in §5 |
| `07-search-core.md` | ID, aspiration, PVS node, qsearch, Stack/PvLine/RootMove | `search` seam (§3.3.15), params.rs (§3.8), Stack (D18) |
| `08-transposition-table.md` | entry/cluster layout, replacement, prefetch, huge pages | `tt` seam (§3.3.12), unsafe clause (a) |
| `09-move-ordering-history.md` | picker stages, scoring, gravity tables, ContHistId | `movepick`/`history` seams (§3.3.13/14) |
| `10-pruning-extensions.md` | pruning/extension/reduction techniques + constants | `search` seam (§3.3.15), params.rs (§3.8) |
| `11-evaluation.md` | phases A/B/C, tapered PSQT, texel tuning, NNUE (§3.4) | `evaluate` seam (§3.3.11), unsafe clause (b) |
| `12-uci-time-management.md` | protocol, options, Engine glue, Limits, TM formulas | `uci`/`timeman` seams (§3.3.17/18), error zones (§3.4) |
| `13-threading-smp.md` | pool lifecycle, job dispatch, voting, TT-resize via `Arc::get_mut` | `threads` seam (§3.3.16), data-flow (§3.2) |
| `14-performance-testing.md` | measurement hierarchy, bench discipline, SPRT procedure (D24/D25) | perf targets (§6), gates in §5 |

---

## 8. Acceptance criteria (D26)

The architecture component is done when all of the following hold:

1. `Cargo.toml` contains the §3.7 profile blocks verbatim; the fast
   configuration is documented in CLAUDE.md; `cargo build --release` with
   `-C target-cpu=x86-64-v3` succeeds.
2. `[dependencies]` is empty and `cargo tree --edges normal` shows only the
   crate itself.
3. `lib.rs` carries `#![deny(unsafe_code)]`; `grep -rn unsafe src/` returns
   zero hits (until a D14-sanctioned site lands with its `// SAFETY:` and
   module-level allow).
4. `types` exports `Value`, `Depth`, the §3.3.1 constants and predicates,
   `PIECE_VALUES`, and `Piece::index`, all unit-tested.
5. `src/search/params.rs` exists with the provenance header; a repo-wide
   grep finds no tuned numeric literal outside it in search-layer code
   (vacuously true until search lands, enforced thereafter).
6. The binary handles the argv `bench`/`perft` subcommands per §3.1 and
   survives the `uci`/`isready`/garbage/`quit` smoke test with exit code 0.
7. `main.rs` is ≤ 30 lines; all logic is in the library and reachable from
   `cargo test`.
8. Every module that exists matches its §3.3 seam (signatures compile as
   written or the doc has been amended in the same commit).
9. `cargo test` and `cargo clippy --all-targets` are clean at every commit
   touched by this plan.

---

## 9. Open questions

1. **`MoveList::push` hard `assert!`**: downgrade to `debug_assert!` once
   the perft suite passes (the 218-move bound makes overflow unreachable
   from legal generation)? Decide with a bench measurement when movegen
   lands; until then the assert stays.
2. **Committing found magic constants**: D12 keeps startup search + LazyLock
   and defers hardcoding. Revisit only if measured init time exceeds the
   100 ms startup target on the test machine.
3. **TT sharing/resize mechanism — resolved.** An earlier draft of this doc
   sketched workers inside one `std::thread::scope` with
   `RwLock<TranspositionTable>` (D23's literal wording).
   `design/13-threading-smp.md` §3.4 owns the final call and chose
   persistent parked threads + `Arc<TT>` handed per Search job, with
   resize/clear via `Arc::get_mut` between searches and an
   interior-mutable generation (`new_search(&self)`, doc 08 §3.9); §3.2 and
   §3.3.16 here follow it. D23's mechanism wording is amended accordingly
   (recorded in doc 13; `thread::scope` stays documented as the fallback).
4. **Eval weights vs `params.rs`**: decided here as: `search/params.rs` is
   search-scoped; eval weights (PSQT tables, texel-tuned terms, NNUE nets)
   are data owned by `eval` under `design/11-evaluation.md` conventions.
   Flagged
   in case decisions.md wants to promote this to a numbered decision.
