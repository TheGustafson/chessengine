# 12 — UCI protocol and time management

Status: Reviewed v1

This document specifies the engine's entire user-visible surface: the UCI
protocol implementation (`uci` module), the developer command extensions
(`bench`, `perft`, `d`, `eval`), the options table, move text conversion,
`info` output and its throttling, MultiPV root mechanics as seen from the
protocol, and the complete time-management system (`timeman` module):
allocation at `go`, the hard stop in `check_time`, the four soft-stop
factors between iterations, pondering, and the `nodestime` reproducibility
mode. All tuned constants are Stockfish-current values adopted as starting
points (D20); everything is expressed as original prose and Rust sketches,
never copied code (D27).

## 1. Purpose and scope

This doc owns:

- `src/uci.rs`: the stdin command loop, command grammar and dispatch, the
  options table and `setoption` semantics, move parsing/formatting in UCI
  long algebraic notation, `info`/`bestmove` output formatting and
  throttling, and the `go`→`bestmove` session lifecycle including the
  exactly-once `bestmove` contract.
- `src/timeman.rs`: `TimeManager` — allocation formulas (`optimum`/
  `maximum`), elapsed-time queries, `nodestime` accounting.
- The soft-stop/hard-stop *policy* (formulas, constants, where they hook
  into the iterative-deepening loop) and the `Limits` type shared by both
  modules.
- The command *surface* of the dev extensions (`bench`, `perft N`, `d`,
  `eval`) — what they print and how they parse.

This doc explicitly does NOT own:

- The iterative-deepening loop, aspiration windows, and root-move
  bookkeeping mechanics (`RootMove` internals, `bestMoveChanges`,
  `effort` accounting) — owned by the search design doc; this doc
  specifies only the hooks it needs.
- Thread pool construction, worker wake/join machinery, and best-thread
  voting — owned by the threading design doc (D23); this doc specifies the
  contract the UCI loop relies on (stop flag, exactly-once `bestmove`,
  `wait_for_search_finished`).
- TT sizing/clearing/`hashfull` — owned by the TT design doc; the `Hash`
  option and `hashfull` info field defined here call into it.
- The bench position suite and signature definition (D24) — owned by the
  bench design doc; the `bench` command surface is defined here.
- Perft internals — owned by the movegen/perft doc.
- Evaluation trace output — owned by the eval doc.

(Sibling docs are referenced by topic in §7; exact filenames per the final
suite index — see Open Questions.)

## 2. References

- `stockfish_reference/stockfish_search_notes.md` §13.1 (threading
  architecture, start_thinking sequence), §13.3 (time management), §13.4
  (MultiPV, ponder, limits), §13.5 (UCI info and score normalization).
- Stockfish source (read-only reference, verified 2026 master):
  - `src/timeman.cpp` — allocation formulas and every constant in §3.7.
  - `src/uci.cpp` — command loop, `parse_limits`, move formatting
    (castling/promotion), WDL model shape, info-line field order.
  - `src/search.cpp` — soft-stop factors (lines ~560-615), `check_time`
    (~2035-2061), busy-wait and bestmove emission (~208-260), currmove
    throttle (`NODES_LIMIT_OUTPUT = 10'000'000`), `searchAgainCounter`
    and `adjustedDepth`, mate-limit stop, root fail-low clearing
    `stopOnPonderhit`.
  - `src/engine.cpp` — options table (names, defaults, ranges),
    `set_ponderhit`.
  - `src/thread.cpp` — `start_thinking` reset sequence, per-game TM state
    defaults (`previousTimeReduction = 0.85`, `originalTimeAdjust = -1`).
- Existing APIs in `src/`: `Board` (D1), `Move`/`MoveList` (D2),
  `Color`/`Square`/`PieceType`, `Board::from_fen`/`START_FEN`.

## 3. Design

### 3.1 Module layout and ownership

```rust
// src/uci.rs   — protocol loop, parsing, formatting, options
// src/timeman.rs — TimeManager + Limits
// src/main.rs  — stays thin:
fn main() {
    chessengine::uci::run();
}
```

`uci::run()` owns an `Engine` value — the glue aggregate, defined HERE
(this doc owns the glue; the parts come from their owning docs):

```rust
pub struct Engine {
    pub pool: ThreadPool,             // threading doc §3.5; owns Arc<PoolShared>
    pub tt: Arc<TranspositionTable>,  // TT doc; cloned into each Search job
    pub options: Options,             // §3.5
    pub board: Board,                 // the "session" position set by `position`
}

impl Engine {
    pub fn new_game(&mut self);  // ucinewgame: Clear job to every worker
                                 //   (histories + the main worker's TimePersist,
                                 //   §3.7) + TT clear
    pub fn wait_for_search_finished(&self); // delegates to pool (threading doc)
    pub fn quit(&mut self);      // stop + join workers (ThreadPool::drop)
}
```

`TimePersist` (§3.7) does NOT live in `Engine`: it is main-worker state
inside `SearchManager` (§3.8), matching the threading doc. The UCI thread
is the *main OS thread*; search always happens on worker threads, so stdin
is never blocked by a running search.

Time is handled as `i64` milliseconds throughout `timeman` (like SF's
`TimePoint`), converting from `std::time::Instant` only at the measurement
point. This keeps every formula in plain integer/float arithmetic and
makes the `nodestime` substitution (§3.11) trivial.

### 3.2 Threading architecture and the bestmove contract

Roles (consistent with D23 and the threading doc):

- **UCI thread (main)**: reads stdin line by line, dispatches commands.
  Never computes; never prints `bestmove`.
- **Main search worker (thread 0)**: runs time management, `check_time`,
  all `info` output, and prints `bestmove` — exactly once per `go`.
- **Helper workers**: search until they observe `stop`; no I/O.

Shared flags (the only cross-thread mutable search-control state besides
the TT). They live in the threading doc's `PoolShared` (§3.5 there) —
that struct and its naming are owned by the threading doc; earlier drafts
of this doc called the flag block "Signals", a name that no longer exists:

```rust
/// Threading doc's PoolShared (excerpt: the flag block). All flag
/// accesses Relaxed: pure flags; no data is published through them.
pub struct PoolShared {
    // shells: Vec<Arc<ThreadShell>>, ... (threading doc §3.5)
    pub stop: AtomicBool,           // set by "stop"/"quit"/hard+soft stops
    pub ponder: AtomicBool,         // true from `go ponder` until "ponderhit"/"stop"
    pub increase_depth: AtomicBool, // soft-stop budget signal (§3.9)
}
```

`stop` and `ponder` are written by the UCI thread and polled by workers;
`increase_depth` is written by the main worker and read at the top of each
iteration by all workers. `stopOnPonderhit` is NOT shared: it is a plain
`bool` field of the main worker's `SearchManager`, because every reader
and writer after search start is the main worker itself (§3.10).

**Session lifecycle** (mirrors SF `start_thinking` / `start_searching`):

1. `go` on the UCI thread: wait for any previous search to finish (a
   well-behaved GUI never overlaps, but we tolerate it); parse `Limits`
   with `start_time` stamped at the *first* token of the line; set the
   three `PoolShared` flags: `stop = false`, `ponder = limits.ponder`,
   `increase_depth = true`; build the root move list (legal moves,
   filtered by `searchmoves` if present); ship limits/root moves to each
   worker via per-thread `Setup` jobs, which run *on the workers' own
   threads* — the main worker's `Setup` additionally resets
   `stop_on_ponderhit = false` and `calls_cnt = 0` there (threading doc
   §3.6 step 4), so `SearchManager` is never written cross-thread; wake
   **only the main worker**; return to reading stdin.
2. Main worker: `time_manager.init(...)` (§3.7); `tt.new_search()`
   (generation bump, TT doc); if the root move list is empty, emit
   `info depth 0 score mate 0` (in check) or `info depth 0 score cp 0`
   (stalemate) and `bestmove (none)`, and finish. Otherwise wake helpers
   and run `iterative_deepening`.
3. When the ID loop returns (depth cap, node/mate/movetime limit, or soft
   stop) the main worker MUST NOT print `bestmove` yet if the GUI still
   owns the clock:

   ```rust
   // The bestmove-exactly-once busy-wait rule: in ponder or infinite
   // mode, bestmove may only be printed after "stop" or "ponderhit".
   while !pool.stop.load(Relaxed)
       && (pool.ponder.load(Relaxed) || limits.infinite)
   {
       std::hint::spin_loop();
   }
   ```

4. Then: `stop = true`; join helpers; best-thread vote (threading doc;
   skipped under `go depth` for bench reproducibility, D24); record
   `best_previous_score` / `best_previous_average_score`; print the final
   PV if the last iteration's PV was never sent; print
   `bestmove <m> [ponder <m2>]`, with the ponder move taken from `pv[1]`
   or, if the PV has length 1, probed from the TT after making `pv[0]`
   (only if the resulting position is not an immediate draw and the TT
   move is legal).

**Contract summary** (GUIs depend on every clause):

- Exactly one `bestmove` per `go`, always — even when stopped instantly,
  even with zero legal moves (`bestmove (none)`).
- Never print `bestmove` while `ponder` is set or `infinite` is active
  and no `stop` has arrived.
- `stop` → `bestmove` promptly (bounded by one `check_time` interval plus
  join time).
- `ponderhit` → the search continues as a normal timed search; if the
  time budget was already exhausted while pondering (`stop_on_ponderhit`),
  it stops at the next `check_time` poll.

**Commands during search.** `stop`, `ponderhit`, `quit`, `isready` are
handled immediately by the UCI thread (`isready` → `readyok` right away —
it is a liveness ping, not a barrier, during search). `setoption`,
`position`, `ucinewgame`, `go`, and all dev commands first call
`engine.wait_for_search_finished()`. To keep that wait *bounded*, a
blocking command that arrives while a search is active is treated as an
**implicit `stop`**: the UCI thread sets `pool.stop = true` first, the
pending `bestmove` flushes normally (the exactly-once contract is
untouched), then the wait returns and the command applies. Without this
rule, `go infinite` (or an un-hit ponder search) followed by an illegal
but real mid-search `position` would deadlock: the search ends only on
`stop`, which the blocked UCI thread would never read. SF accepts that
hang; we deliberately don't — GUIs are not supposed to send these
mid-search at all, but a robust engine survives ones that do.

### 3.3 Command grammar and dispatch

Input is processed line by line; tokens are `split_whitespace()`. Unknown
top-level commands print
`info string Unknown command: '<line>'` and continue. Blank lines are
ignored. EOF on stdin is treated as `quit` (critical for
cutechess/fastchess which close the pipe).

| Command | Grammar | Response |
|---|---|---|
| `uci` | — | `id name ARCHON <version>`, `id author <author>`, one `option` line per table entry (§3.5), then `uciok`; `ARCHON` is the D41 public engine name while the internal Cargo package/binary may remain `chessengine` |
| `isready` | — | `readyok` |
| `setoption` | `setoption name <id...> [value <x...>]` | none (errors as `info string`) |
| `ucinewgame` | — | none; clears TT, histories, per-game TM state (§3.7) |
| `position` | `position (startpos \| fen <f1..f6>) [moves <m>...]` | none |
| `go` | `go [<param>...]` — see §3.6 | `info` lines, then exactly one `bestmove` |
| `stop` | — | (search prints `bestmove`) |
| `ponderhit` | — | none (search continues, now on our clock) |
| `debug` | `debug [on \| off]` | none (accepted per spec; we have no debug mode) |
| `register` | `register later \| register ...` | none (no registration required) |
| `quit` | — | process exits |
| `bench` | `bench [ttSize [threads [limit [fenFile [limitType]]]]]` | bench report + signature (bench doc) |
| `perft` | `perft <depth>` (also `go perft <depth>`) | per-move split counts + total (perft doc) |
| `d` | — | board diagram, FEN, Zobrist key, checkers |
| `eval` | — | static-eval trace (eval doc) |

Notes on parsing:

- `setoption`: the option *name* may contain spaces — collect tokens after
  `name` until the literal token `value` or end of line; the *value* is
  the rest of the line (whitespace-joined). Name matching is
  case-insensitive (SF's options map is case-insensitive).
- `position fen`: collect tokens until the literal `moves` or end of line
  and join them as the FEN string; `Board::from_fen` already accepts 4-6
  fields. `position startpos` uses `Board::startpos()`.
- `position ... moves m1 m2 ...`: parse and apply moves *atomically* — on
  a scratch `Board`, parse each token against the legal move list of the
  evolving position (§3.4) and `make_move` it; if any token fails, print
  `info string Invalid move '<tok>' in position command` and leave the
  engine's position unchanged. (SF exits the process here; we prefer to
  stay alive — a deliberate, documented divergence for robustness.)
  Applying the full move list through `make_move` builds the ply-indexed
  `StateInfo` history inside `Board` (D8), which is exactly what
  repetition detection and rule-50 need.
- `go` parameter tokens may appear in any order except `searchmoves`,
  which consumes all remaining tokens (SF behavior; matches how GUIs emit
  it). Unknown `go` tokens are skipped silently (protocol robustness).
- `debug` and `register` are consumed silently: the UCI spec requires
  engines to accept them, and routing them to the unknown-command branch
  would spam `info string` at GUIs that send them. (SF prints
  unknown-command here; deliberate conformance divergence.)
- `stop` and `ponderhit` with no active search are harmless no-ops: the
  flag stores hit an idle pool, and the next `go` resets both flags
  (§3.2 step 1).

Dispatch sketch:

```rust
pub fn run() {
    let mut engine = Engine::new(); // startpos, default options
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };            // EOF/error => quit
        if !dispatch(&mut engine, &line) { break }     // false => quit
    }
    engine.quit(); // stop + join workers
}

fn dispatch(engine: &mut Engine, line: &str) -> bool {
    let mut tokens = line.split_whitespace();
    match tokens.next() {
        Some("uci") => cmd_uci(engine),
        Some("isready") => println!("readyok"),
        Some("setoption") => cmd_setoption(engine, tokens),
        Some("ucinewgame") => engine.new_game(),
        Some("position") => cmd_position(engine, tokens),
        Some("go") => cmd_go(engine, tokens),
        Some("stop") => engine.pool.stop(),           // no-op when idle (§3.3 notes)
        Some("ponderhit") => engine.pool.ponderhit(), // no-op when idle
        Some("quit") => { engine.pool.stop(); return false }
        Some("debug") => { /* accepted per spec; we have no debug mode */ }
        Some("register") => { /* no registration required; consumed */ }
        Some("bench") => cmd_bench(engine, tokens),
        Some("perft") => cmd_perft(engine, tokens),
        Some("d") => cmd_display(engine),
        Some("eval") => cmd_eval(engine),
        Some(other) => println!("info string Unknown command: '{other}'"),
        None => {}
    }
    true
}
```

### 3.4 Move parsing and formatting (UCI long algebraic)

Formatting: our `Move` `Display` impl already emits UCI long algebraic —
`from` + `to` + optional lowercase promotion letter (`e2e4`, `e7e8q`).
Castling needs no translation because our internal encoding is already
the king-to-castling-square form (`e1g1`, `e1c1`) per D2/D10 — unlike
engines that encode king-takes-rook internally, we have nothing to remap
until Chess960 lands (deferred; when it does, 960 mode emits king-to-rook
square). En passant prints as the capturing pawn's from/to, which is
already correct.

Two sentinel spellings (SF conventions):

```rust
/// Move -> UCI text. Move::NULL is our none-sentinel (D2).
pub fn move_to_uci(m: Move) -> String {
    if m == Move::NULL { "(none)".to_string() } else { m.to_string() }
}
```

(`0000` — the null-move spelling — is reserved for a future distinct
null-move sentinel per D2; it never appears in a PV, so v1 never emits
it.)

Parsing goes through move *generation*, not text analysis: a token is a
legal move iff it formats identically to one of the position's legal
moves. This makes the formatter the single source of truth and gets
promotion/ep/castling disambiguation for free:

```rust
/// Parse a UCI move token against the legal moves of `board`.
/// Case-insensitive (GUIs occasionally send E2E4 or e7e8Q).
pub fn move_from_uci(board: &Board, token: &str) -> Option<Move> {
    let token = token.to_ascii_lowercase();
    legal_moves(board) // movegen doc: MoveList of fully legal moves
        .iter()
        .copied()
        .find(|m| m.to_string() == token)
}
```

Cost is O(legal moves) with a String per candidate; only executed on
`position`/`searchmoves` parsing, never in search. If profiling ever
cares, compare against a stack-formatted `[u8; 5]`, but this is not hot.

### 3.5 Options table

```rust
pub struct Options {
    entries: Vec<OptionEntry>, // registration order == `uci` print order
}

pub struct OptionEntry {
    pub name: &'static str,
    pub kind: OptionKind,
    pub value: OptionValue, // current value
}

pub enum OptionKind {
    Check { default: bool },
    Spin { default: i64, min: i64, max: i64 },
    Button,
    Text { default: &'static str },
}
```

v1 table (defaults/ranges are SF-current where we share the option):

| Name | Type | Default | Range | Effect on set |
|---|---|---|---|---|
| `Debug Log File` | string | "" (empty) | — | close any open log; if non-empty, open (append) and tee all stdin/stdout traffic (below) |
| `Hash` | spin | 16 | 1 .. max (TT doc; MiB) | resize + clear TT |
| `Threads` | spin | 1 | 1 .. `max(4·available_parallelism, 512)` (§9.6, D31) | rebuild thread pool |
| `Clear Hash` | button | — | — | clear TT + histories |
| `MultiPV` | spin | 1 | 1 .. 256 (`MAX_MOVES`) | read at next `go` |
| `Ponder` | check | false | — | TM adds +25% optimum (§3.7); GUI drives actual pondering via `go ponder` |
| `Move Overhead` | spin | 10 | 0 .. 5000 (ms) | read at next TM init |
| `nodestime` | spin | 0 | 0 .. 10000 (nodes per ms) | read at next TM init (§3.11) |
| `Minimal` | check | false | — | low-verbosity report mode (§3.13, D31): suppress per-iteration `info`, print one summary before `bestmove` |

**`Debug Log File`** is the protocol debugging tool (SF ships the same
option): when set, every line read from stdin and every line written to
stdout is appended to the file, prefixed with a direction marker (`<<`
in, `>>` out) and a monotonic timestamp (ms since the logger opened).
It hooks the two existing choke points — the `run()` read loop and the
single-`println!` output path (§3.13) — so info lines land in the log
atomically. The log is opened/closed on `setoption`; open or write
failures degrade silently (an `info string` at open time at most —
logging must never crash or stall the engine). This is the tool for
diagnosing time losses and GUI-specific failures from fastchess/
cutechess runs, where the raw dialogue is otherwise invisible.

Deferred (documented so the table shape anticipates them): `SyzygyPath` /
`SyzygyProbeDepth` / `Syzygy50MoveRule` / `SyzygyProbeLimit` (tablebase
doc), `UCI_ShowWDL` (needs our own WDL model, §3.14), `UCI_Chess960`,
`Skill Level` / `UCI_LimitStrength` / `UCI_Elo` (we may never want
these), and `EvalFile` (phase C, D21; the eval doc §3.4.5 requires it
from this layer): string, default `<internal>` = the embedded net; on
set, `wait_for_search_finished()`, then load and validate the CENN file
via the eval doc's loader; on failure print
`info string ERROR: could not load <path>` and keep the current net;
bench signatures are only comparable at the same net (D24).

`uci` prints each entry as
`option name <name> type <type> [default <d>] [min <lo> max <hi>]`.

`setoption` semantics:

1. `engine.wait_for_search_finished()` — options never change under a
   running search. Per the implicit-stop rule (§3.2), a `setoption` sent
   mid-search first sets `stop` (the pending `bestmove` flushes), then
   waits and applies — the wait is always bounded. The commands a GUI
   legitimately sends mid-search (`stop`, `ponderhit`, `isready`) are
   handled before this point in dispatch, and GUIs serialize on
   `isready` anyway.
2. Case-insensitive lookup; unknown name →
   `info string No such option: '<name>'`.
3. Spin values are range-clamped after parse; parse failure → error
   string, value unchanged.
4. Side effects run synchronously (Hash resize, Threads rebuild), so a
   following `isready` naturally answers only when they're done — that is
   exactly the sync point GUIs rely on.

Typed accessors (`options.hash_mb()`, `options.threads()`,
`options.multipv()`, `options.ponder()`, `options.move_overhead_ms()`,
`options.nodestime()`) hide the table from the rest of the engine.

### 3.6 `go` limits

```rust
/// Everything `go` can ask for. Times in milliseconds.
#[derive(Clone)]
pub struct Limits {
    pub time: [i64; 2],       // wtime/btime, 0 = not given (SF convention)
    pub inc: [i64; 2],        // winc/binc
    pub movestogo: i32,       // 0 = sudden death / not given
    pub depth: Option<Depth>, // go depth d
    pub nodes: Option<u64>,   // go nodes n
    pub movetime: Option<i64>,// go movetime ms
    pub mate: Option<i32>,    // go mate n (n = moves, not plies)
    pub infinite: bool,
    pub ponder: bool,
    pub searchmoves: Vec<Move>,   // empty = all root moves
    pub perft: Option<u32>,       // go perft d (dev)
    pub start_time: Instant,      // stamped when parsing begins
}

impl Limits {
    /// Clock-based time management is active when the GUI sent EITHER
    /// clock — side-independent, exactly SF's predicate (search.h). A GUI
    /// that sends only the opponent's clock (`go btime ...` with white to
    /// move) thus still gets a TM-bounded search, never an unbounded one
    /// (see §3.7 step 1 for what init does in that case).
    /// depth/nodes/movetime/mate/infinite modes bypass TM.
    pub fn use_time_management(&self) -> bool {
        self.time[Color::White as usize] != 0
            || self.time[Color::Black as usize] != 0
    }
}
```

Semantics of each limit (verified against SF):

- **depth d**: the *main worker's* ID loop stops iterating once
  `root_depth >= d`; helpers run until they observe `stop` (set when the
  main worker finishes). Best-thread voting is disabled under a depth
  limit so `bench` node counts are reproducible (D24).
- **nodes n**: aggregated over all workers (relaxed per-worker counters,
  summed on read, D23), enforced in `check_time`; the check cadence
  tightens to `min(512, n / 1024)` node-entries so tiny node budgets are
  honored (§3.8).
- **movetime t**: hard wall in `check_time` (`elapsed >= t`); soft-stop
  factors do not apply.
- **mate n**: after each *completed* iteration, stop when
  `rootMoves[0].score` is a mate score and
  `VALUE_MATE - score.abs() <= 2 * n` (mate in ≤ n full moves proven).
- **infinite**: never self-stop; `bestmove` only after `stop` (busy-wait
  rule, §3.2).
- **searchmoves m1 m2 ...**: root move whitelist; parsed with
  `move_from_uci`; unparseable tokens are ignored with an `info string`
  warning. If the filter leaves *zero* root moves in a position that has
  legal moves (every token unparseable or illegal), the filter is ignored
  entirely — search all legal moves and print
  `info string Invalid searchmoves list ignored` — so bad GUI input can
  never turn a playable position into `bestmove (none)`. MultiPV is
  clamped to the filtered root-move count.
- Bare `go` (no parameters): all fields empty → no TM, no limits → the ID
  loop runs to `MAX_PLY` and then prints `bestmove` (matches SF; GUIs
  that want analysis send `go infinite`).
- `wtime/btime` may be negative in GUI lag situations; parse into i64 and
  let `timeLeft`'s `max(1, ...)` clamp handle it.

### 3.7 Time management: allocation at `go`

All constants in this section are SF-current fitted values adopted
verbatim as starting points (D20); they live in `search/params.rs` (or
`timeman.rs` consts) under greppable names, never inline.

```rust
pub struct TimeManager {
    start_time: Instant,
    optimum_ms: i64,
    maximum_ms: i64,
    use_nodes_time: bool,
    /// nodestime budget; -1 = not yet initialized this game (§3.11).
    available_nodes: i64,
}

/// Per-GAME state that survives across `go` calls; reset by ucinewgame.
pub struct TimePersist {
    pub original_time_adjust: f64,      // -1.0 = unset
    pub previous_time_reduction: f64,   // 0.85 initial
    pub best_previous_score: Value,          // VALUE_INFINITE initial
    pub best_previous_average_score: Value,  // VALUE_INFINITE initial
}

impl TimeManager {
    /// Compute optimum/maximum for this move. `ply` = game ply
    /// (2*(fullmove_number-1) + (stm == Black) as i32).
    /// May rewrite limits.time/inc under nodestime (§3.11).
    pub fn init(&mut self, limits: &mut Limits, us: Color, ply: i32,
                options: &Options, persist: &mut TimePersist);

    pub fn optimum(&self) -> i64;   // ms (or nodes under nodestime)
    pub fn maximum(&self) -> i64;
    /// TM-elapsed: nodes searched under nodestime, else wall ms.
    /// `Fn`, not `FnOnce`: the counter closure is only invoked under
    /// nodestime, and check_time may also need it for the node limit
    /// (§3.8) — same call, possibly twice, possibly zero times.
    pub fn elapsed(&self, nodes: impl Fn() -> u64) -> i64;
    /// Wall-clock ms, always (info lines / nps).
    pub fn elapsed_ms(&self) -> i64;
    pub fn advance_nodes_time(&mut self, used_minus_inc: i64);
}
```

`init` algorithm (numbers exact from `timeman.cpp`):

1. Reset `optimum_ms = maximum_ms = 0`, stamp
   `start_time = limits.start_time`, latch `use_nodes_time`. If
   `limits.time[us] == 0` → done (movetime/depth/nodes/infinite handled
   elsewhere; SF's init has the same per-side early return). Note the
   asymmetry with the side-independent `use_time_management()` (§3.6):
   when the GUI sent only the *opponent's* clock, TM stays active but
   init returned here, so `maximum() == 0` and the first `check_time`
   poll stops the search — a missing own clock in a timed game yields an
   instant (shallow) answer, never an unbounded search. (SF gets the
   same outcome via zero-initialized members; we reset explicitly so the
   behavior is per-`go` deterministic and testable, §5.)
2. `move_overhead = options.move_overhead_ms()`. Apply the nodestime
   conversion if enabled (§3.11), producing `scale_factor` (1 in normal
   mode) and `scaled_time = time / scale_factor` — "the time as a human
   would see it", used in comparisons against absolute constants.
3. Move horizon:
   - `mtg = if movestogo > 0 { min(movestogo, 50) } else { 50 }`
   - if `scaled_time < 1000` ms: `mtg = (scaled_time as f64 * 0.05) as i32`
     (gradually shrink the horizon below one second on the clock; note
     `mtg` may become 0 — the formulas below stay well-defined in f64,
     and `timeLeft`'s clamp handles the integer part).
4. `time_left = max(1, time + inc*(mtg-1) - move_overhead*(2 + mtg))`
   (i64 math; the `2 +` buys one extra move of overhead safety margin).
5. **Sudden death / increment** (`movestogo == 0`) — fitted log formulas:
   - once per game (first `init` with `original_time_adjust < 0`):
     `original_time_adjust = 0.3272 * log10(time_left) - 0.4141`
   - `log_time_in_sec = log10(scaled_time / 1000.0)`
   - `opt_constant = min(0.0029869 + 0.00033554 * log_time_in_sec, 0.004905)`
   - `max_constant = max(3.3744 + 3.0608 * log_time_in_sec, 3.1441)`
   - `opt_scale = min(0.012112 + (ply + 3.22713).powf(0.46866) * opt_constant,`
     `              0.19404 * time / time_left) * original_time_adjust`
   - `max_scale = min(6.873, max_constant + ply / 12.352)`
6. **Repeating TC** (`movestogo > 0`):
   - `opt_scale = min((0.88 + ply / 116.4) / mtg, 0.88 * time / time_left)`
   - `max_scale = 1.3 + 0.11 * mtg`
7. Results:
   - `optimum_ms = max(1, (opt_scale * time_left) as i64)`
   - `maximum_ms = max(optimum_ms,`
     `  min(0.8097 * time - move_overhead, max_scale * optimum) as i64)`
     — never budget more than ~81% of the remaining clock for one move.
8. If the `Ponder` *option* is on (regardless of whether this `go` is a
   ponder search): `optimum_ms += optimum_ms / 4` — pondering recoups
   time on average, so think 25% longer.

Formula shapes to remember when tuning (D20): `opt_scale` grows slowly
with game ply (a power curve saturated by `opt_constant`, which itself
grows with the log of the clock) and is capped at ~19.4% of the current
clock per move; `max_scale` allows `maximum` to be up to ~6.9× `optimum`
early, tapering via the absolute 81%-of-clock cap.

`ucinewgame` resets `TimePersist` to
`{ original_time_adjust: -1.0, previous_time_reduction: 0.85,
best_previous_score: VALUE_INFINITE,
best_previous_average_score: VALUE_INFINITE }` and
`available_nodes = -1`; these two "first move of the game" defaults
measurably affect move-1 time usage (SF resets them in exactly this way).

### 3.8 Hard stop: `check_time`

Runs on the **main worker only**, at every node entry, gated by a
countdown so the clock syscall happens every 512 node-entries:

```rust
pub struct SearchManager {              // main worker only (Null-object
    pub tm: TimeManager,                // pattern: helpers have none)
    pub persist: TimePersist,
    pub calls_cnt: i32,
    pub stop_on_ponderhit: bool,        // plain bool: main-worker-local
    pub iter_value: [Value; 4],         // scores from the last 4 iters
    pub iter_idx: usize,                // both re-seeded per `go` (§3.9)
    pub previous_time_reduction: f64,
    // ... info-throttling state (§3.13)
}

impl SearchManager {
    // `total_nodes` is an aggregation closure (Σ per-worker counters,
    // threading doc §3.10). `Fn`, not `FnOnce`: it is invoked lazily —
    // once inside `elapsed` under nodestime, once for the node-limit
    // check — and not at all when neither applies, which keeps the
    // aggregation off the common path (§6).
    pub fn check_time(&mut self, limits: &Limits, pool: &PoolShared,
                      total_nodes: impl Fn() -> u64) {
        self.calls_cnt -= 1;
        if self.calls_cnt > 0 { return; }
        // With a node limit, poll at >= 0.1% of the budget.
        self.calls_cnt = match limits.nodes {
            Some(n) => (n / 1024).min(512) as i32,
            None => 512,
        };

        // The GUI owns the clock while pondering: NEVER stop here.
        if pool.ponder.load(Relaxed) { return; }

        let elapsed = self.tm.elapsed(&total_nodes); // nodes under nodestime
        let tm_stop = limits.use_time_management()
            && (elapsed > self.tm.maximum() || self.stop_on_ponderhit);
        let movetime_stop = limits.movetime.is_some_and(|t| elapsed >= t);
        let nodes_stop = limits.nodes.is_some_and(|n| total_nodes() >= n);
        if tm_stop || movetime_stop || nodes_stop {
            pool.stop.store(true, Relaxed);
        }
    }
}
```

Key properties: `maximum` is the only *hard* clock wall; the ponder guard
means a pondering search can only be ended by the GUI; and
`stop_on_ponderhit` converts "budget exhausted during ponder" into an
instant stop at the first poll after `ponderhit` (§3.10). The 512-entry
cadence bounds clock-read overhead to ~1 syscall per 512 nodes while
keeping stop latency well under a millisecond at realistic NPS.

**P10 implementation note (the §10 robustness bundle, landed).** The
shipped `check_time` (`src/search/mod.rs`) folds in the Reckless-study
hardening, so it diverges from the sketch above in four bench-neutral ways
(all inert under fixed-depth/fixed-node search — the stop conditions never
fire there — so the `bench` fingerprint is unmoved, D24):

- **Bitmask cadence.** The `calls_cnt` countdown field is replaced by a
  bitmask on the per-worker node counter: the real check runs when `nodes &
  mask == mask`. `mask = CHECK_TIME_CALLS − 1` (511) by default; under
  `go nodes n` it tightens to the largest power of two `≤ min(512, n/1024)`
  (a mask of 0 polls every node), so tiny node budgets are still honored.
  This lets the added qsearch-entry poll share one cadence for free — no
  double-decrement.
- **`completed_depth == 0` guard.** No hard stop of any kind (clock,
  `movetime`, or `nodes`) fires before the first ID iteration has fully
  completed, guaranteeing a real — if shallow — `bestmove` under an absurdly
  small budget. `completed_depth` is set to `root_depth` at the end of each
  un-aborted iteration.
- **Flat epsilon shave** (`CHECK_TIME_EPSILON_MS`, 15 in TM units) is
  subtracted from the *wall-clock* bounds only — `maximum()` and `movetime`
  — so we stop a few ms early to cover the unwind-and-print tail and cut
  fast-TC forfeit risk. The `nodes` budget is left unshaved (an exactness
  bound, not a clock forfeit risk).
- **qsearch-entry re-poll.** `check_time` is also called at qsearch entry,
  bounding stop latency under a long forcing-capture dive; qsearch itself
  does not abort — the enclosing node's step-11 stop check discards the
  untrusted subtree the moment the dive returns.

### 3.9 Soft stop: between-iterations decision

At the end of each completed ID iteration, the main worker decides
whether starting another iteration is worth it. Runs only when
`limits.use_time_management() && !stop && !stop_on_ponderhit`.

Inputs maintained by the search doc's ID loop:

- `best_value` — this iteration's score; `iter_value` ring stores the
  scores of the previous four iterations
  (`iter_value[iter_idx]` is the value from 4 iterations ago at read
  time; after the decision: `iter_value[iter_idx] = best_value;
  iter_idx = (iter_idx + 1) & 3`). **Seeding is per-`go`, never carried
  over**: at the top of the main worker's ID loop, `iter_idx = 0` and
  all four entries are filled with `persist.best_previous_score`, or
  with `VALUE_DRAW` when that is still `VALUE_INFINITE` (first search of
  the game) — exactly SF's `iterValue.fill(...)` in
  `iterative_deepening`. This makes fallingEval well-defined from
  iteration 1 and the golden tests of §5 writable.
- `last_best_move_depth` — the iteration at which the current best root
  move first became best.
- `tot_best_move_changes: f64` — per-worker `bestMoveChanges` counters
  (incremented when the best root move flips mid-iteration), summed into
  the main worker's accumulator each iteration, then the accumulator is
  **divided by 2.5 at the start of every iteration** (exponential decay;
  SF halves it — see the depth-redenomination note below).
- `rootMoves[0].effort` — nodes searched under the best root move.

The four factors (constants from `search.cpp`, with two deliberate
2026-07-26 deviations; `lerp(x, x0, x1, y0, y1)` is linear interpolation):

> **Depth-redenomination (2026-07-26).** Factors 2 and 3 take inputs
> denominated in ITERATIONS, and SF's constants assume SF's iteration
> counts. Measured on 60 shared middlegames at 3 s/move: we complete ~3.3
> fewer iterations than SF16 at equal time (depth 24.2 vs 27.5) while the
> ply of the last best-move change (12.7 vs 12.3) and root flips per search
> (4.08 vs 4.05) are EQUAL — we are slower, not less decisive. Unadjusted,
> the same wobble therefore reads as "unstable" (stability input 11.6 vs
> 15.2) and each flip decays ~2^3.3 ≈ 10× less by the stop decision,
> which front-loaded the clock (real 2+1 game: 108 s of base gone by move
> 15 vs SF16's 54 s). Two compensations, both derived from the measured
> deficit, both pinned in `params.rs` with the derivation: the stability
> lerp window shifts down 3 ([5,18] → [2,15]) and the flip decay divisor
> becomes 2.5 (`d^11.6 = 2^15.2 ⇒ d ≈ 2.48`). Replaying the probe game
> with real draining clocks reproduced 53.6 s by move 15 (SF16: 54.1 s).

1. **fallingEval** — think longer when the score is dropping:
   `fallingEval = (11.87 + 2.21*(best_previous_average_score - best_value)`
   `              + 1.0*(iter_value[iter_idx] - best_value)) / 100.0`
   clamped to `[0.572, 1.708]`.
2. **reduction** (best-move stability) — save time when the best move has
   been stable for many iterations, partially giving back last move's
   savings:
   `time_reduction = clamp(lerp(root_depth - last_best_move_depth,`
   `                            2.0, 15.0, 0.65, 1.55), 0.65, 1.55)`
   (SF's window is [5,18]; ours is shifted by the depth-redenomination
   above)
   `reduction = (1.48 + previous_time_reduction) / (2.157 * time_reduction)`
   (`previous_time_reduction` = last move's `time_reduction`, initial
   0.85; **held constant for the whole search** — only the denominator's
   `time_reduction` is recomputed each iteration. The running
   `time_reduction` seeds to `1.0` at each `go` (SF `search.cpp:279`) and is
   stored back into persist as the next move's `previous_time_reduction`
   only at search end, never mid-search).
3. **bestMoveInstability** — think longer when the best move keeps
   flipping:
   `best_move_instability = 1.096 + 2.29 * tot_best_move_changes / threads`
4. **highBestMoveEffort** — stop sooner when the lion's share of nodes
   already went into the best move (its refutation isn't coming):
   `nodes_effort = rootMoves[0].effort * 100_000 / max(1, total_nodes)`
   (per-100k units, i.e. 79219 ≈ 79.2%)
   `high_best_move_effort = clamp(lerp(nodes_effort as f64,`
   `    79_219.0, 101_822.0, 0.924, 0.71), 0.71, 0.924)`

Decision:

```text
total_time = tm.optimum() * falling_eval * reduction
           * best_move_instability * high_best_move_effort
if root has exactly ONE legal move:
    total_time = min(561.7, total_time)          # ms; viewer-experience cap
elapsed = tm.elapsed(total_nodes)

if elapsed > min(total_time, tm.maximum())
   or rootMoves[multiPV-1].score >= mate_in(3)   # every PV line mates in <= 3 plies
   or rootMoves[0].score == mated_in(2):         # we are mated in 2 plies
    if ponder: stop_on_ponderhit = true          # keep pondering; GUI owns clock
    else:      pool.stop = true
else:
    pool.increase_depth = ponder or elapsed <= total_time * 0.50
```

**P10 landed (step 8).** The four factors, both mate early-stops, and the
`increase_depth`/`search_again_counter` damping ship now
(`Worker::soft_stop`, `src/search/mod.rs`), each factor a pure function
with a golden test pinning its §3.9 constants (params in
`search/params.rs`). One v1 restriction: `stop_on_ponderhit` is the ponder
state machine (step 9), which is **deferred to P11** — so while pondering,
`soft_stop` simply never self-stops and leaves `increase_depth = true`
(the GUI owns the clock until `ponderhit`/`stop`), rather than latching
`stop_on_ponderhit`. Everything is gated behind `use_time_management()`, so
fixed-depth/fixed-node search (bench, D24) is untouched: the factor math
never runs, the `iter_value` ring is never read, and `search_again_counter`
stays 0, leaving `adjusted_depth` at its full value.

**`increase_depth` / `searchAgainCounter`**: when the soft budget is more
than half spent, the next iteration should not honestly deepen — at the
top of each iteration every worker does
`if !increase_depth { search_again_counter += 1 }` and searches at
`adjusted_depth = max(1, root_depth - failed_high_cnt`
`                     - 3 * (search_again_counter + 1) / 4)`
so late "iterations" re-search near the same effective depth, refreshing
the PV and TT instead of starting a deeper pass that would be aborted
mid-way. (The `adjusted_depth` formula lives in the search doc's
aspiration loop; recorded here because TM drives it.)

**Root fail-low revokes `stopOnPonderhit`**: in the aspiration re-search
loop, when the root search fails low (`best_value <= alpha`), the main
worker sets `stop_on_ponderhit = false`. Rationale: a fail-low means the
intended move may be refuted — the one situation where burning past the
soft budget after a ponderhit is clearly right.

**Mate-limit stop** (`go mate n`): checked after each completed
iteration, independent of TM: stop when `rootMoves[0].score` is decisive
mate and `VALUE_MATE - |score| <= 2n`.

### 3.10 Ponder state machine

States of a `go` session: `Searching { ponder: bool }` →
(`ponderhit` | `stop`) → `Finishing` → `bestmove` printed → `Idle`.

- `go ... ponder`: `pool.ponder = true` before the search starts. The
  clock fields in `go ponder` are the GUI's clocks *as if the pondered
  move was played*; TM init runs normally so budgets are ready the moment
  `ponderhit` arrives.
- While `ponder` is set: `check_time` returns early (never stops), and
  the soft stop converts its verdict into `stop_on_ponderhit = true`
  instead of stopping. The search thus runs indefinitely on GUI time.
- `ponderhit` (GUI: the opponent played the expected move):
  `pool.ponder = false`. Consequences, in order of observation by the
  main worker: the busy-wait (if the ID loop already returned) exits and
  `bestmove` flows; otherwise `check_time` now sees
  `stop_on_ponderhit == true` (budget already burnt while pondering) and
  stops at the next poll, or — if budget remains — the search simply
  continues as a normal timed search.
- `stop` (GUI: pondered move was NOT played, or analysis over): stop
  immediately and print `bestmove` for the pondered position (the GUI
  discards it).
- The `Ponder` UCI *option* only feeds the +25% optimum bump (§3.7 step
  8); actual pondering is always GUI-initiated via `go ponder`.

The ponder move printed with `bestmove` is `pv[1]` when available; with a
1-ply PV the TT probe fallback (§3.2 step 4) fills it in when possible —
without a ponder move, a `Ponder`-on GUI has nothing to ponder, wasting
half the feature's Elo.

### 3.11 `nodestime` mode (reproducible time controls)

`setoption name nodestime value N` (N = nodes per millisecond) replaces
wall-clock time with a node budget — bit-reproducible "time" for testing
TM changes (D24/D25 tooling):

- At the first `init` of a game: `available_nodes = N * limits.time[us]`.
- Every `init`: `limits.time[us] = available_nodes`,
  `limits.inc[us] *= N`, `move_overhead *= N`, `scale_factor = N` (so
  `scaled_time` still means human milliseconds).
- `tm.elapsed(...)` returns *nodes searched so far* instead of wall ms —
  every comparison against `optimum`/`maximum`/`total_time` transparently
  becomes a node comparison.
- After the search: `available_nodes = max(0,`
  `available_nodes - (nodes_searched - inc_in_nodes))` — consumption is
  booked and the increment credited, exactly like a real clock.
- Caveat (SF warns the same): N must be far below the engine's real NPS
  or real-world runs will lose on time; this is a testing mode.
- `elapsed_ms()` (wall clock) still feeds `info ... time/nps` so logs
  remain human-readable.

### 3.12 MultiPV mechanics at root

Owned jointly with the search doc; the protocol-visible rules:

- `multi_pv = min(options.multipv(), root_moves.len())`, re-read at every
  `go`.
- Each iteration runs an outer loop `for pv_idx in 0..multi_pv`; each PV
  line gets its **own aspiration window** seeded from that root move's
  running `average_score` (window sizing in the search doc; SF-current
  seed: `delta = 5 + thread_idx % 8 + |mean_squared_score| / 10588`).
- Root move loop skips moves outside `root_moves[pv_idx..pv_last]` — the
  first `pv_idx` moves are this iteration's already-finished PV lines.
  (`pv_first/pv_last` banding by Syzygy `tb_rank` arrives with TB
  support; until then `pv_last = root_moves.len()`.)
- At iteration start, every root move's `score` is reset to
  `-VALUE_INFINITE` after saving `previous_score`; only searched lines
  get real scores. Sorting MUST be **stable** (Rust `slice::sort_by` is)
  so unsearched moves keep last iteration's order behind the searched
  ones: after each aspiration round,
  `root_moves[pv_idx..pv_last].sort_by(cmp)`; after a line completes,
  `root_moves[pv_first..=pv_idx].sort_by(cmp)`. The comparator orders by
  `(score, previous_score)` descending.

```rust
pub struct RootMove {
    pub pv: Vec<Move>,            // pv[0] is the root move
    pub score: Value,             // -VALUE_INFINITE when unsearched
    pub previous_score: Value,
    pub average_score: Value,     // running blend across iterations
    pub mean_squared_score: i64,
    pub uci_score: Value,         // what info lines print
    pub score_lowerbound: bool,   // fail-high in progress
    pub score_upperbound: bool,   // fail-low in progress
    pub sel_depth: i32,
    pub effort: u64,              // nodes spent under this move (§3.9)
    pub tb_rank: i32,             // Syzygy, later
}
```

- Aborted-search hygiene (search doc owns the details; protocol
  consequence recorded here): exact mated/TB-loss scores from an
  *aborted* line are not trusted — they are rolled back to the previous
  iteration's score or republished with a bound flag, so the GUI never
  sees a fake forced loss.
- `searchmoves` composes naturally: the root move list is filtered before
  MultiPV clamping.

### 3.13 `info` output and throttling

One `info` line per PV index, emitted after each completed PV line.
Field order (SF's, kept for GUI-parser compatibility):

```text
info depth <d> seldepth <sd> multipv <k> score <cp X | mate Y>
     [lowerbound|upperbound] [wdl <w> <d> <l>]
     nodes <n> nps <n> hashfull <permille> tbhits <n> time <ms>
     pv <m1> <m2> ...
```

- `depth` = root iteration depth; `seldepth` = per-line max ply reached
  (reset per iteration per line).
- `score` from `uci_score` with bound suffix from the flags (§3.12);
  formatting in §3.14. `wdl` only when `UCI_ShowWDL` exists and is on
  (deferred).
- `nodes` aggregated over workers; `nps = nodes * 1000 / max(1, time)`;
  `time` = wall-clock `elapsed_ms()` (never node-time); `hashfull` =
  TT permille estimate (TT doc); `tbhits` aggregated (0 until Syzygy).
- Emission is ONE `println!` per line from a pre-built `String` — a
  single atomic write keeps concurrent-looking output impossible (only
  the main worker prints during search, but `info string` from the UCI
  thread must not interleave mid-line).

Currmove lines, during the root move loop on the main worker:

```text
info depth <d> currmove <move> currmovenumber <n>   # n = move_count + pv_idx
```

**Throttling rules** (avoid flooding GUIs — SF-current):

- `NODES_LIMIT_OUTPUT = 10_000_000` (name it `INFO_THROTTLE_NODES`):
  - `currmove` lines are printed **only after** total nodes exceed it
    (SF once used "after 3 seconds"; the node form is the current rule
    and what we adopt — it's also nodestime-friendly).
  - Aspiration fail-high/fail-low intermediate updates (re-publishing the
    PV with a bound before a re-search) are printed only when
    `multi_pv == 1` and nodes exceed the threshold.
  - Completed PV lines below the threshold are printed only when the last
    line of the iteration finishes (`pv_idx + 1 == multi_pv`); above it,
    after every line.
- On final `bestmove` emission, if the last iteration's PV was never sent
  (or the vote picked another thread's line), print it first — the GUI's
  last-seen PV must match the bestmove.

### 3.14 Score formatting and the normalization hook

```rust
pub enum UciScore { Cp(i32), Mate(i32) } // Mate counts full MOVES, signed

pub fn format_score(v: Value, board: &Board) -> UciScore {
    if is_mate(v) {
        let plies = VALUE_MATE - v;          // mate FOR us in `plies` plies
        UciScore::Mate((plies + 1) / 2)      // round up: our mating move counts
    } else if is_mated(v) {
        let plies = VALUE_MATE + v;          // mated in `plies` plies
        UciScore::Mate(-(plies / 2))         // truncating, matching SF exactly
    } else {
        UciScore::Cp(to_cp(v, board))
    }
}
```

Mate convention: positive `mate M` = we mate in M moves; SF computes
`(plies > 0 ? plies + 1 : plies) / 2` on the *signed* ply count with
truncating division — equivalent to the two branches above (Rust's `/`
truncates like C++). Encapsulated in one function with unit tests over
the full mate band, both signs, odd and even plies.

**Normalization (deferred, hook documented)**: SF normalizes internal
values so that **+100 cp ⇔ 50% win probability**, via a WDL model
`P(win) = 1 / (1 + e^{(a - v)/b})` where `a` and `b` are cubic
polynomials in a material count (`pawns + 3*minors + 5*rooks + 9*queens`,
clamped to `[17, 78]`, anchored at 58), fitted on fishtest LTC data;
`to_cp(v) = round(100 * v / a(material))`, and the optional
`wdl <w> <d> <l>` triple (permille) comes from the same model evaluated
at `v` and `-v`. The *coefficients* are fits to Stockfish's own eval and
are meaningless for ours — so we do NOT adopt them: phases A/B print raw
internal values (`to_cp(v, _) = v` — our eval is already roughly
centipawn-scaled), and once our NNUE is trained we fit our own
`(a, b)` polynomials from self-play data and enable `UCI_ShowWDL`. The
single seam is `to_cp(v: Value, board: &Board) -> i32`; nothing else in
the engine may convert values to centipawns.

### 3.15 Dev extensions

- `bench [ttSize] [threads] [limit] [fenFile] [limitType]` — defaults
  `16 1 13 default depth` (SF-compatible invocation shape). Runs the
  fixed suite via the normal `go` path, prints per-position headers to
  stderr, and ends with the three-line report (total time, **nodes
  searched** = the bench signature (D24), nps). Owned by the bench doc.
- `perft N` / `go perft N` — from the current position, print
  `<move>: <count>` per root move plus `Nodes searched: <total>` (perft
  doc; the divide format is what every debugging workflow diffs against).
- `d` — the `Board` `Display` diagram plus `Fen: <fen>`,
  `Key: <16-hex-digits>` (Zobrist, D9) and `Checkers: <squares>`.
- `eval` — static evaluation trace of the current position (eval doc).

All dev commands `wait_for_search_finished()` first (§3.2).

**P10 landed (step 12).** In-session `perft N` / `go perft N` (perft divide
from the session position), `bench [...]` (the fixed suite on its own
worker/TT, diagnostics to stderr), and `eval` are wired into the dispatch
alongside `d`. `eval` prints the Phase A static value (material + PSQT) from
the side-to-move's perspective — the rich per-term trace lands with the
eval doc's later phases, and `to_cp` is still the identity (§3.14). The
`bench current`-position source (design/06 §3.6) is not yet plumbed to the
session board: the in-session command answers `info string` rather than
hitting its `todo!()`. Also landed: step 11's post-search `nodestime`
accounting (`advance_nodes_time`), booking consumed nodes minus the
increment credit after every clock-based move so the node budget carries
across the game — two identical scripted `nodestime` games are
bit-deterministic (§5).

## 4. Implementation plan

Each step is one commit-sized unit with a done-condition. Steps 1-3 need
only the existing core types; steps 4+ interleave with the search and
threading docs' milestones.

1. **`timeman.rs`: `Limits`, `TimeManager::init`, elapsed queries.** Pure
   functions + `TimePersist`; all constants named in one place (D20).
   Done: golden unit tests for §3.7 (fixed inputs → hand-computed
   optimum/maximum), property tests (optimum ≤ maximum; maximum ≤
   0.8097·time − overhead whenever that cap binds; ponder bump = +25%).
2. **UCI skeleton**: `run()` loop, `uci`/`isready`/`quit`/`d`/`position`
   (startpos/fen, no moves yet), unknown-command handling, EOF-as-quit.
   Done: scripted stdin session produces byte-exact expected transcript
   (integration test spawning the binary).
3. **Move text layer**: `move_to_uci`, `move_from_uci` (needs legal
   movegen), `position ... moves ...` atomic application. Done:
   round-trip tests over every legal move in castling/ep/promotion FENs;
   bad-move rejection leaves position unchanged.
4. **Options table + `setoption`**: table type, printing in `uci`,
   case-insensitive set, clamping, `Debug Log File`/`Hash`/`Clear Hash`/
   `Threads`/`MultiPV`/`Ponder`/`Move Overhead`/`nodestime` wired to
   their consumers. Done: `setoption` round-trip test; `uci` output lists
   all options with correct defaults/ranges.
5. **`go` end-to-end, single-threaded**: `parse_limits`, search on one
   worker thread, `PoolShared.stop`, `stop`/`quit` handling, exactly-once
   `bestmove` incl. `(none)` cases; `movetime`/`depth`/`nodes`/`infinite`
   limits via `check_time` (hard stops only). Done: engine plays a full
   fixed-movetime game under cutechess without stalls, illegal moves, or
   duplicate/missing bestmove (log-grepped).
6. **`info` emission + throttling**: per-iteration PV lines, currmove
   after `INFO_THROTTLE_NODES`, single-write lines. Done: GUI (or
   scripted session) shows well-formed info; a `go nodes 20000000` run
   emits currmove lines only late.
7. **Clock TM wired**: `TimeManager` driving `check_time`
   (optimum/maximum), the v1 between-iterations soft stop — the plain
   `elapsed > optimum` comparison (§3.9's `total_time` with no factors
   yet) with the single-reply cap, per the master plan's P7 scope note —
   and `ucinewgame` persist reset. Done: 100 games at 10s+0.1 sudden
   death with zero time forfeits (D25 tooling).
8. **Soft stop completion**: the four factors (multiplying into step 7's
   plain-optimum budget), mate-score early stops,
   `increase_depth`/`search_again_counter` (hooks: `best_move_changes`,
   `effort`, `iter_value` from the search doc).
   Done: bench signature unchanged (fixed-depth bench is TM-independent,
   D24); TC self-play vs step-7 build is non-regressing (SPRT, D25).
9. **Ponder**: `go ponder`, `ponderhit`, busy-wait rule,
   `stop_on_ponderhit`, fail-low revocation, +25% optimum. Done:
   cutechess match with ponder=on completes; log grep confirms exactly
   one bestmove per go and no bestmove-before-ponderhit.
10. **MultiPV**: root loop skipping, per-line aspiration, stable-sort
    invariants, clamping vs `searchmoves`. Done: `MultiPV=3 go depth 12`
    prints 3 lines per iteration, multipv 1..3, weakly decreasing exact
    scores; `bench` (MultiPV=1) signature unchanged.
11. **`nodestime`**: budget conversion, node-elapsed, post-search
    accounting. Done: two identical scripted games with
    `nodestime 1000` produce identical move sequences and identical
    node counts.
12. **Dev surface completion**: `bench`, `perft N`, `eval` wired to their
    owners. Done: `bench` prints the signature; `perft 6` from startpos
    matches the known 119,060,324.

## 5. Testing and validation

- **Unit — timeman**: golden tests for `init` (e.g. `time=60000,
  inc=600, ply=20, overhead=10, movestogo=0` → assert exact
  optimum/maximum computed from §3.7); branch coverage: movestogo branch,
  sub-second `mtg` shrink, `time_left` clamp at 1, ponder bump,
  `original_time_adjust` computed exactly once per game, missing own
  clock (`time[us] == 0`, opponent clock set → `use_time_management()`
  true, `optimum == maximum == 0`, §3.7 step 1). Property tests:
  `1 <= optimum <= maximum` (whenever our clock was given); monotonicity
  of optimum in `time`.
- **Unit — soft stop**: factor functions tested at their clamp edges
  (fallingEval at ±extreme score drops; timeReduction at stability 0, 5,
  18, 30; highBestMoveEffort at 0%, 79.2%, 101.8% effort); single-reply
  cap; decision truth table (elapsed vs min(total, maximum), mate
  conditions, ponder → stopOnPonderhit); `iter_value`/`iter_idx` per-`go`
  seeding (all four entries = `best_previous_score`, or `VALUE_DRAW` on
  the first search of a game, §3.9 — never stale values from the
  previous `go`).
- **Unit — text**: move round-trips (castling e1g1/e1c1, ep, all four
  promotions, case-insensitivity); score formatting across the mate band
  including negative odd/even plies; `(none)` handling; FEN error paths.
- **Unit — parser**: `setoption` with spaced names/values; `go` token
  soup in random order; `searchmoves` consuming the tail; unknown tokens
  skipped; `position fen` with 4- and 6-field FENs; `debug` and
  `register` consumed silently (no unknown-command output, §3.3);
  `stop`/`ponderhit` on an idle engine are no-ops.
- **Integration (spawn the binary, scripted stdin/stdout)**:
  1. `uci`→`uciok` with full options block; `isready`→`readyok`.
  2. `position startpos moves e2e4 e7e5` + `d` → expected FEN.
  3. `go depth 4` → info lines then exactly one bestmove.
  4. `go infinite`, 200ms sleep, `stop` → exactly one bestmove, after
     `stop` only.
  5. `go ponder wtime.. btime..`, 100ms, `ponderhit` → bestmove follows;
     variant with `stop` instead.
  6. `go movetime 100` → bestmove within 100ms + overhead tolerance.
  7. Mated/stalemated positions → `bestmove (none)` with the
     `info depth 0` line.
  8. Implicit stop (§3.2): `go infinite`, 100ms sleep, `position
     startpos` → exactly one bestmove flushes, then the new position
     applies (`d` confirms); no hang.
  9. `go searchmoves` with only invalid tokens in a position with legal
     moves → `info string Invalid searchmoves list ignored` (§3.6) and a
     normal bestmove, never `(none)`.
  10. `setoption name Debug Log File value <tmp>` + a short scripted
     session → the log contains every stdin/stdout line with direction
     markers and timestamps (§3.5); unset closes the file.
- **Perft/bench gates (D24)**: `perft` suite green before/after every
  step touching move text or position handling; bench signature recorded
  in every commit message from step 5 on; TM-only patches must not move
  it (bench is fixed-depth).
- **Regression under real clocks (D25)**: fastchess runs at 10+0.1 and
  60+0.6, ponder off and on; acceptance thresholds in §8. Time-forfeit
  count is the primary metric for this doc's components; Elo deltas
  belong to the search doc's changes.
- **nodestime determinism test** in CI: scripted 20-move game twice,
  byte-identical output required.

## 6. Performance notes

- **`check_time` cadence**: one `Instant::now()` (+ a nodes aggregation
  under a node limit) per 512 node-entries ≈ per ~25µs at 20 Mnps —
  negligible (<0.1%), while worst-case stop latency stays ≪ 1ms. Never
  read the clock per node.
- **Stop polling**: `pool.stop.load(Relaxed)` per node is a plain L1
  load; do not "optimize" it behind the calls counter — prompt stop
  observation by helpers matters for `stop` latency.
- **Output**: build each info line in one `String`, one `println!`
  (single locked write + flush). Un-throttled output is a real GUI
  killer (SF added the 10M-node gate because Fritz 19 hangs); keep the
  throttle in place even for short searches.
- **Busy-wait** after search-end during ponder/infinite burns one core
  with `spin_loop` hints — same trade SF makes; the window is typically
  milliseconds. Revisit with a condvar park only if profiling of long
  analysis sessions ever shows it mattering (Open Questions).
- **Move Overhead default 10ms** assumes local pipes. Document in the
  README that lag-prone setups (network GUIs, containers) should raise
  it; every timeLeft formula already prices in `overhead*(2+mtg)`.
- **What to measure** when tuning TM (D20/D25): time-forfeit rate (must
  be 0), average clock utilization per game phase, Elo of factor tweaks
  via SPRT at two TCs (TM changes are notoriously TC-sensitive), and
  `go movetime` adherence (elapsed − budget ≤ one check interval).

## 7. Interactions

(Filenames per the final suite index; topics are binding, numbers
provisional — see Open Questions.)

- **Depends on**:
  - Movegen/perft doc — `legal_moves(&Board)` for move parsing,
    `searchmoves`, root-move building, gives_check for applying
    `position` moves (D10); the perft driver behind `perft N`.
  - Board/state doc — `Board::from_fen`, `make_move` + owned `StateInfo`
    history (D8) so `position ... moves` establishes repetition/rule-50
    context; Zobrist key for `d` output (D9).
  - Search doc — ID-loop hooks consumed by §3.9 (`best_move_changes`,
    `effort`, `iter_value`, `last_best_move_depth`,
    `search_again_counter`/`adjusted_depth`, aspiration fail-low event),
    `RootMove` ownership, seldepth, mate-limit check placement.
  - Threading doc (D23) — `ThreadPool` and `PoolShared` (home of the
    stop/ponder/increase_depth flags, §3.2), `Setup` jobs, wake/join,
    `wait_for_search_finished`, best-thread voting and its `go depth`
    exemption, aggregated node/tbhit counters. (The `Engine` glue
    aggregate itself is defined here, §3.1.)
  - TT doc — `Hash` option resize/clear, `Clear Hash`, `hashfull()`,
    `new_search()` generation bump, ponder-move TT probe.
  - Eval doc (D21/D22) — `eval` command trace; later the WDL fit feeding
    `to_cp`.
  - Bench doc (D24) — suite, signature, `bench` argument handling.
- **Feeds**: every doc's acceptance testing (bench/perft run through this
  command surface); the threading doc's `start_thinking` sequence is
  co-specified here (§3.2); tablebase doc will extend the options table,
  root-move banding (`pv_first/pv_last`), and `tbhits`.

## 8. Acceptance criteria (D26)

The component is done when all of the following hold:

1. **Protocol conformance**: the scripted integration suite of §5 passes
   byte-exact; `uci` lists exactly the §3.5 v1 options with correct
   types/defaults/ranges.
2. **bestmove contract**: across all integration tests and a 1000-game
   fastchess run, `grep -c bestmove` equals the number of `go` commands;
   no bestmove precedes its `stop`/`ponderhit` in infinite/ponder
   sessions.
3. **Zero time forfeits** in 1000 games at 10+0.1 and 300 games at
   60+0.6 (ponder off), and 300 games at 10+0.1 with ponder on, against
   any opponent, with `Move Overhead = 10` on local pipes.
4. **TM math**: timeman golden tests pin every constant of §3.7; soft
   stop factor tests pin every constant of §3.9; the constants live in
   one params module (D20) and appear nowhere else.
5. **MultiPV**: `MultiPV=k` emits exactly `min(k, legal)` lines per
   iteration with `multipv 1..=k`, exact scores weakly decreasing, and
   order stability across iterations verified by test.
6. **nodestime determinism**: the CI determinism test (§5) is green.
7. **Dev surface**: `perft 6` startpos = 119,060,324 via the command;
   `bench` prints a stable signature across two identical runs (D24);
   `d` and `eval` produce their owners' specified output.
8. **Options behave**: `setoption name Hash value 256` changes `hashfull`
   dynamics measurably; `Threads value 4` scales nps in `bench` >2×
   on 4 cores; both applied only between searches.
9. **Clean build gates**: `cargo test` and `cargo clippy --all-targets`
   clean; no runtime dependencies added (D15).

## 9. Open questions

1. **Sibling doc filenames**: §7 references sibling docs by topic; the
   suite's final numbered filenames must be substituted once the index is
   fixed.
2. **Busy-wait vs parking**: we adopt SF's busy-wait for the
   search-finished-while-ponder/infinite window (§3.2). If long analysis
   sessions show meaningful idle burn, switch to `Condvar` parking — the
   contract is unaffected.
3. **`position` error handling deviation**: SF terminates the process on
   an invalid `position` command; we keep the old position and print
   `info string` (§3.3). Deliberate robustness choice — flagging since it
   diverges from the reference behavior.
4. **currmove throttle basis**: we adopt the SF-current 10M-node gate;
   the task brief also mentions the older 3-second wall rule. Node-based
   is nodestime-friendly and current; revisit only if GUI UX complains on
   very slow hardware.
5. **WDL model timing**: `to_cp` stays identity until phase C (D21) plus
   a self-play WDL fit; decide then whether `UCI_ShowWDL` ships with the
   first NNUE release or later.
6. **Threads option upper bound** — RESOLVED in P10 (D31): the provisional
   flat 1024 is replaced by the hardware-scaled `max(4·available_parallelism,
   512)` (`uci::max_threads`), falling back to 512 when the core count is
   unavailable. The Hash upper bound stays the TT doc's 64-bit cap.

## 10. Reckless study inputs (D31, 2026-07-11)

P10 candidates from the Reckless study (`reckless_reference/09`, summary
`00-design-implications.md` §P10); ideas only, safe-Rust (D27):
- **Robustness bundle (§3.8) — LANDED in P10.** A `completed_depth==0`
  hard-stop guard (guarantee a real bestmove under a tiny budget); re-poll
  `check_time` at qsearch entry (bound stop latency in long forcing dives);
  a flat epsilon shave (`CHECK_TIME_EPSILON_MS`) on the wall-clock bounds
  including `movetime` (cut fast-TC forfeit risk); node-counter bitmask poll
  cadence (`nodes & mask`, mask `= CHECK_TIME_CALLS − 1`, tightened under a
  node limit). See the §3.8 implementation note for the shipped shape.
- **Options — LANDED in P10.** `Threads` upper bound
  `max(4·available_parallelism, 512)` (§9.6, `uci::max_threads`) resolves the
  flat-1024 open question with a hardware-scaled bound; the `Minimal` check
  option (§3.13) gates every per-iteration `info` line, leaving one summary
  before `bestmove` for high-frequency self-play — pure I/O, bench-neutral.
