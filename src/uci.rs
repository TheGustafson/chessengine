//! UCI entry point: argv subcommand dispatch, then the blocking stdin loop.
//!
//! Contracts: `design/01-architecture.md` §3.1/§3.3.18 (the `run()` seam;
//! argv subcommands run to completion before the loop) and
//! `design/12-uci-time-management.md` §3.3–§3.6 (command grammar, move
//! text, options table, `go` limits) plus its robustness rules: never
//! panic on input, unknown commands answer `info string`, EOF on stdin is
//! `quit` — GUIs close the pipe, and a dead engine loses on time.
//!
//! P5 state (doc 12 §4 steps 1–6): everything from P4 — `position`
//! (startpos/fen + atomic move application), the move-text layer
//! (`move_to_uci`/`move_from_uci`), the §3.5 options table with full
//! `setoption` semantics (including the `Debug Log File` stdin/stdout
//! tee), `ucinewgame` per-game TM reset, `go`-limit parsing into
//! [`Limits`] — plus the live `go` session (doc 12 §3.2): the search runs
//! on its own thread while this thread keeps reading stdin, `stop` flips a
//! shared flag, blocking commands apply the implicit-stop rule, and the
//! worker prints `info` lines (§3.13 v1 scope) and the exactly-once
//! `bestmove` — after `stop`/`ponderhit` only, in infinite/ponder mode
//! (the busy-wait rule). wtime/btime go through
//! [`TimeManager::init`](crate::timeman::TimeManager::init) with
//! the `maximum` hard wall enforced in the search's `check_time`; the
//! soft-stop factors and real pondering are doc 12 steps 8–9. The
//! in-session `bench`/`perft`/`go perft`/`eval` dev commands land with doc
//! 12 step 12. The argv subcommands are live: `perft <depth> [fen]` prints
//! a real SF-format divide (P3, design/06 §3.2) and `bench ... perft` runs
//! perft-mode bench.
//!
//! P6 state (doc 08 §4 steps 7–8): [`Engine`] owns the shared
//! `Arc<TranspositionTable>` (D23); `setoption name Hash` resizes it
//! between searches via `Arc::get_mut` — fallible per doc 08 §3.13,
//! allocation shortfall answers
//! `info string Failed to allocate <req> MB hash, using <got> MB` and the
//! engine keeps playing — while `Clear Hash` and `ucinewgame` clear it.
//! Each `go` clones the handle into its search thread ([`run_search`]
//! holds it for the job's duration and bumps `tt.new_search()` once, doc
//! 13 §3.4/§3.7 B — dropped at job end, so the idle UCI thread stays the
//! sole holder and `Arc::get_mut` keeps working), the worker probes and
//! stores through it (doc 07's call sites), and completed PV lines carry
//! `info hashfull` (§3.13).

use crate::bench;
use crate::board::Board;
use crate::movegen;
use crate::moves::{MAX_MOVES, Move, MoveList};
use crate::perft;
use crate::search::{SearchInfo, Worker};
use crate::timeman::{Limits, TimePersist};
use crate::tt::{MAX_HASH_MB, MIN_HASH_MB, TranspositionTable};
use crate::types::{Color, VALUE_INFINITE, VALUE_MATE, Value, is_mate, is_mated};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::str::{FromStr, SplitWhitespace};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Instant;

/// `id name` text (doc 12 §3.3): crate name + version from the manifest,
/// so the handshake can never drift from `Cargo.toml`.
const NAME: &str = concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"));
/// `id author` text, likewise from the manifest `authors` field.
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// The program front door (design/01 §3.1): argv subcommands (`bench`,
/// `perft <depth> [fen...]`, `d`) run to completion; with no subcommand
/// this blocks on the stdin UCI loop until `quit`/EOF.
pub fn run() {
    // Force the magic attack tables before any command runs (design/03
    // §3.7): the ~21 ms magic search never happens mid-search, and the
    // LazyLock branch is always-taken from here on. `main` is a one-line
    // delegator to this function, so this IS startup.
    crate::attacks::tables();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => cmd_bench_argv(&args[1..]),
        Some("perft") => cmd_perft_argv(&args[1..]),
        // `d`: board diagram + FEN + key + checkers (design/12 §3.15).
        // Outside a UCI session the "current" position is the start
        // position.
        Some("d") => {
            let _ = print_board(&mut io::stdout().lock(), &Board::startpos());
        }
        Some(other) => exit_usage(format_args!(
            "unknown subcommand: '{other}' (expected bench, perft, or d)"
        )),
        None => {
            let stdin = io::stdin();
            // Deliberately NOT StdoutLock: the search worker prints from
            // its own thread, and a lock held by this thread would block
            // it forever. [`SharedOut`]'s mutex is the line-atomicity
            // guarantee instead (doc 12 §3.13).
            let out = SharedOut::new(io::stdout());
            uci_loop(stdin.lock(), &out);
        }
    }
}

/// Report an argv usage error on stderr and exit 1.
///
/// `process::exit` skips destructors — fine at the program front door:
/// nothing below `run()` holds state that needs unwinding, and stderr is
/// unbuffered. Inside the UCI *loop* errors never exit (doc 12 §3.3).
fn exit_usage(msg: fmt::Arguments<'_>) -> ! {
    eprintln!("{msg}");
    std::process::exit(1)
}

/// `chessengine bench [ttMB] [threads] [limit] [fens] [type]`
/// (design/06 §3.6).
fn cmd_bench_argv(args: &[String]) {
    match bench::BenchParams::parse(args) {
        Ok(params) => {
            bench::run(&params);
        }
        Err(e) => exit_usage(format_args!("bench: {e}")),
    }
}

/// `chessengine perft <depth> [fen...]` — the CLI convenience face of
/// perft divide (design/06 §3.2); the FEN may span several argv tokens.
fn cmd_perft_argv(args: &[String]) {
    // divide requires depth >= 1 (design/06 §3.2); reject 0 here rather
    // than panic on external input (design/01 §3.4 error zones).
    let depth = args.first().and_then(|tok| tok.parse::<u32>().ok());
    let Some(depth @ 1..) = depth else {
        exit_usage(format_args!(
            "usage: chessengine perft <depth> [fen...] (depth >= 1)"
        ));
    };
    let mut board = if args.len() > 1 {
        match Board::from_fen(&args[1..].join(" ")) {
            Ok(board) => board,
            Err(e) => exit_usage(format_args!("perft: invalid FEN: {e}")),
        }
    } else {
        Board::startpos()
    };
    print!("{}", perft::divide(&mut board, depth));
}

// ---------------------------------------------------------------------------
// Move text (design/12 §3.4)
// ---------------------------------------------------------------------------

/// `Move` -> UCI long algebraic text. `Move::NULL` is our none-sentinel
/// (D2) and prints as `(none)`, the SF convention for `bestmove` with no
/// legal moves. Everything else defers to the `Display` impl (from + to +
/// lowercase promotion letter); castling needs no translation because our
/// internal encoding is already the king-to-castling-square form (D2/D10).
pub fn move_to_uci(m: Move) -> String {
    if m == Move::NULL {
        "(none)".to_string()
    } else {
        m.to_string()
    }
}

/// Parse a UCI move token against the legal moves of `board`:
/// a token is a legal move iff it formats identically to one of them, so
/// the formatter is the single source of truth and promotion/ep/castling
/// disambiguation comes for free. Case-insensitive (GUIs occasionally send
/// `E2E4` or `e7e8Q`). O(legal moves) with a `String` per candidate — only
/// runs on `position`/`searchmoves` parsing, never in search.
pub fn move_from_uci(board: &Board, token: &str) -> Option<Move> {
    let token = token.to_ascii_lowercase();
    let mut list = MoveList::new();
    movegen::generate_legal(board, &mut list);
    list.iter().copied().find(|m| m.to_string() == token)
}

// ---------------------------------------------------------------------------
// Options table (design/12 §3.5)
// ---------------------------------------------------------------------------

/// `Hash` spin range in MiB, straight from the owner (doc 08 §3.13:
/// SF's 64-bit cap; the range is a sanity bound on the option, not a
/// promise the allocation succeeds — `resize` is fallible).
const HASH_MB_MIN: i64 = MIN_HASH_MB as i64;
const HASH_MB_MAX: i64 = MAX_HASH_MB as i64;

/// What kind of option an entry is, with its default and (for spins) the
/// clamping range. The kind is immutable after registration; only
/// [`OptionValue`] changes on `setoption`.
#[derive(Clone, Copy, Debug)]
pub enum OptionKind {
    Check { default: bool },
    Spin { default: i64, min: i64, max: i64 },
    Button,
    Text { default: &'static str },
}

/// Current value of an option. Buttons carry no value — their effect is
/// the side effect of pressing them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionValue {
    Check(bool),
    Spin(i64),
    Button,
    Text(String),
}

/// One registered option: prints itself as the `uci`-handshake line.
#[derive(Clone, Debug)]
pub struct OptionEntry {
    pub name: &'static str,
    pub kind: OptionKind,
    pub value: OptionValue,
}

impl fmt::Display for OptionEntry {
    /// `option name <name> type <type> [default <d>] [min <lo> max <hi>]`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option name {} type ", self.name)?;
        match self.kind {
            OptionKind::Check { default } => write!(f, "check default {default}"),
            OptionKind::Spin { default, min, max } => {
                write!(f, "spin default {default} min {min} max {max}")
            }
            OptionKind::Button => write!(f, "button"),
            // `<empty>` is the GUI-conventional spelling of an empty
            // string default (SF prints the same).
            OptionKind::Text { default } => {
                let shown = if default.is_empty() {
                    "<empty>"
                } else {
                    default
                };
                write!(f, "string default {shown}")
            }
        }
    }
}

/// The options table (doc 12 §3.5). Registration order == `uci` print
/// order; lookup is case-insensitive. Typed accessors hide the table from
/// the rest of the engine. `Clone` because each `go` session ships a
/// snapshot to the search thread (`setoption` never changes options under
/// a running search, §3.5, so a snapshot is always current).
#[derive(Clone)]
pub struct Options {
    entries: Vec<OptionEntry>,
}

impl Options {
    /// The v1 table with SF-current defaults/ranges (D20). Consumers:
    /// `Move Overhead`/`nodestime`/`Ponder` feed [`TimeManager::init`];
    /// `Debug Log File` drives the I/O tee in this module; `Hash`/`Clear
    /// Hash` act on the TT (P6), `Threads` rebuilds the pool (P8),
    /// `MultiPV` is read at each `go` (P5+) — until those land the values
    /// are stored and parroted, nothing more.
    pub fn new() -> Options {
        let register = |name, kind| OptionEntry {
            name,
            kind,
            value: match kind {
                OptionKind::Check { default } => OptionValue::Check(default),
                OptionKind::Spin { default, .. } => OptionValue::Spin(default),
                OptionKind::Button => OptionValue::Button,
                OptionKind::Text { default } => OptionValue::Text(default.to_string()),
            },
        };
        Options {
            entries: vec![
                register("Debug Log File", OptionKind::Text { default: "" }),
                register(
                    "Hash",
                    OptionKind::Spin {
                        default: 16,
                        min: HASH_MB_MIN,
                        max: HASH_MB_MAX,
                    },
                ),
                register(
                    "Threads",
                    OptionKind::Spin {
                        default: 1,
                        min: 1,
                        max: 1024,
                    },
                ),
                register("Clear Hash", OptionKind::Button),
                register(
                    "MultiPV",
                    OptionKind::Spin {
                        default: 1,
                        min: 1,
                        max: MAX_MOVES as i64,
                    },
                ),
                register("Ponder", OptionKind::Check { default: false }),
                register(
                    "Move Overhead",
                    OptionKind::Spin {
                        default: 10,
                        min: 0,
                        max: 5000,
                    },
                ),
                register(
                    "nodestime",
                    OptionKind::Spin {
                        default: 0,
                        min: 0,
                        max: 10000,
                    },
                ),
            ],
        }
    }

    /// Registration-order iteration (the `uci` print order).
    pub fn iter(&self) -> impl Iterator<Item = &OptionEntry> {
        self.entries.iter()
    }

    /// `setoption` core (doc 12 §3.5): case-insensitive lookup, spin
    /// values range-clamped after parse, parse failure leaves the value
    /// unchanged. Returns the CANONICAL name on success (callers key side
    /// effects off it); the `Err` string is ready for `info string`.
    pub fn set(&mut self, name: &str, value: Option<&str>) -> Result<&'static str, String> {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.name.eq_ignore_ascii_case(name))
        else {
            return Err(format!("No such option: '{name}'"));
        };
        match entry.kind {
            OptionKind::Button => {} // no value; pressing IS the effect
            OptionKind::Check { .. } => {
                let v = value.ok_or_else(|| format!("No value for option '{}'", entry.name))?;
                entry.value = match v {
                    "true" => OptionValue::Check(true),
                    "false" => OptionValue::Check(false),
                    _ => {
                        return Err(format!(
                            "Invalid value '{v}' for check option '{}'",
                            entry.name
                        ));
                    }
                };
            }
            OptionKind::Spin { min, max, .. } => {
                let v = value.ok_or_else(|| format!("No value for option '{}'", entry.name))?;
                let parsed: i64 = v
                    .parse()
                    .map_err(|_| format!("Invalid value '{v}' for spin option '{}'", entry.name))?;
                entry.value = OptionValue::Spin(parsed.clamp(min, max));
            }
            OptionKind::Text { .. } => {
                // Missing value and the `<empty>` sentinel both mean the
                // empty string (that is how a GUI unsets a string option).
                let v = match value {
                    None | Some("<empty>") => "",
                    Some(v) => v,
                };
                entry.value = OptionValue::Text(v.to_string());
            }
        }
        Ok(entry.name)
    }

    fn spin(&self, name: &str) -> i64 {
        match self
            .entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| &e.value)
        {
            Some(&OptionValue::Spin(v)) => v,
            _ => unreachable!("{name} is a registered spin option"),
        }
    }

    pub fn hash_mb(&self) -> i64 {
        self.spin("Hash")
    }

    pub fn threads(&self) -> i64 {
        self.spin("Threads")
    }

    pub fn multipv(&self) -> i64 {
        self.spin("MultiPV")
    }

    pub fn ponder(&self) -> bool {
        match self
            .entries
            .iter()
            .find(|e| e.name == "Ponder")
            .map(|e| &e.value)
        {
            Some(&OptionValue::Check(v)) => v,
            _ => unreachable!("Ponder is a registered check option"),
        }
    }

    pub fn move_overhead_ms(&self) -> i64 {
        self.spin("Move Overhead")
    }

    pub fn nodestime(&self) -> i64 {
        self.spin("nodestime")
    }

    pub fn debug_log_file(&self) -> &str {
        match self
            .entries
            .iter()
            .find(|e| e.name == "Debug Log File")
            .map(|e| &e.value)
        {
            Some(OptionValue::Text(v)) => v,
            _ => unreachable!("Debug Log File is a registered string option"),
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options::new()
    }
}

// ---------------------------------------------------------------------------
// Debug Log File tee (design/12 §3.5)
// ---------------------------------------------------------------------------

/// The open `Debug Log File`: every stdin/stdout line is appended with a
/// direction marker (`<<` in, `>>` out) and a monotonic ms timestamp since
/// the logger opened. Write failures degrade silently — logging must never
/// crash or stall the engine.
struct DebugLog {
    file: File,
    opened: Instant,
}

impl DebugLog {
    fn open(path: &str) -> io::Result<DebugLog> {
        let file = File::options().create(true).append(true).open(path)?;
        Ok(DebugLog {
            file,
            opened: Instant::now(),
        })
    }

    fn log_line(&mut self, direction: &str, line: &str) {
        let ms = self.opened.elapsed().as_millis();
        let _ = writeln!(self.file, "[{ms:>8}] {direction} {line}");
    }
}

/// `Write` adapter over the engine's stdout that tees complete output
/// lines into the debug log. All protocol output flows through one of
/// these, so the log captures the whole dialogue at the two existing choke
/// points (the read loop logs input; this wrapper logs output).
struct UciOut<W: Write> {
    out: W,
    log: Option<DebugLog>,
    /// Output bytes of the current (not yet newline-terminated) line.
    partial: Vec<u8>,
}

impl<W: Write> UciOut<W> {
    fn new(out: W) -> UciOut<W> {
        UciOut {
            out,
            log: None,
            partial: Vec::new(),
        }
    }

    fn log_input(&mut self, line: &str) {
        if let Some(log) = &mut self.log {
            log.log_line("<<", line);
        }
    }

    /// Reconfigure logging per the option value: close any open log, then
    /// open (append) the new path if non-empty. Open failure degrades to
    /// an `info string` and no logging (doc 12 §3.5).
    fn set_log_path(&mut self, path: &str) -> io::Result<()> {
        self.log = None;
        self.partial.clear();
        if path.is_empty() {
            return Ok(());
        }
        match DebugLog::open(path) {
            Ok(log) => {
                self.log = Some(log);
                Ok(())
            }
            Err(e) => writeln!(self, "info string Could not open debug log '{path}': {e}"),
        }
    }
}

impl<W: Write> Write for UciOut<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.log.is_some() {
            for &byte in buf {
                if byte == b'\n' {
                    let line = String::from_utf8_lossy(&self.partial).into_owned();
                    if let Some(log) = &mut self.log {
                        log.log_line(">>", &line);
                    }
                    self.partial.clear();
                } else {
                    self.partial.push(byte);
                }
            }
        }
        self.out.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Line-atomic shared output (doc 12 §3.13): the UCI thread and the search
/// worker both print through one mutex-guarded [`UciOut`], so lines from
/// concurrent writers can never interleave mid-line, and the `Debug Log
/// File` tee sees ALL stdout traffic — search output included. `Clone`
/// shares the sink (an `Arc` under the hood); the manual impl avoids a
/// spurious `W: Clone` bound.
struct SharedOut<W: Write>(Arc<Mutex<UciOut<W>>>);

impl<W: Write> Clone for SharedOut<W> {
    fn clone(&self) -> SharedOut<W> {
        SharedOut(Arc::clone(&self.0))
    }
}

impl<W: Write> SharedOut<W> {
    fn new(out: W) -> SharedOut<W> {
        SharedOut(Arc::new(Mutex::new(UciOut::new(out))))
    }

    /// Lock for one line (or one short multi-line block, e.g. the `uci`
    /// handshake). A poisoned mutex — some writer panicked mid-line —
    /// degrades to the inner value: a protocol loop that must never die
    /// (doc 12 §3.3) prefers possibly-torn log output over propagating
    /// the panic.
    fn lock(&self) -> MutexGuard<'_, UciOut<W>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Recover the raw sink; callable only when every search thread has
    /// been joined (test harness use).
    #[cfg(test)]
    fn into_inner(self) -> W {
        let mutex = Arc::try_unwrap(self.0)
            .ok()
            .expect("all SharedOut clones dropped (searches joined)");
        mutex
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .out
    }
}

// ---------------------------------------------------------------------------
// Engine — the session glue aggregate (design/12 §3.1, P5/P6 subset)
// ---------------------------------------------------------------------------

/// The glue aggregate owned by the UCI loop (doc 12 §3.1). P5/P6 subset:
/// the session position, the options table, the shared transposition
/// table, and the single search worker with its per-game TM state. The
/// real `ThreadPool` (with `PoolShared`) joins in P8; [`Session`] and
/// [`SearchState`] are its v1 stand-ins with the same contract: stdin is
/// never blocked by a running search, and `bestmove` is printed exactly
/// once per `go`, by the search thread.
pub struct Engine {
    board: Board,
    options: Options,
    /// The one large shared-mutable structure (D23): search sessions will
    /// clone this handle into their workers (doc 08 §4 step 7 / P8 pool);
    /// between searches the UCI thread is the only holder, so
    /// [`Engine::tt_mut`] gets `&mut` for resize/clear via `Arc::get_mut`
    /// (doc 13 §3.6) with zero locking on the probe path.
    tt: Arc<TranspositionTable>,
    /// The idle-state search state; `None` exactly while a search thread
    /// owns it (it travels there and comes back through the join handle).
    state: Option<SearchState>,
    /// The most recent `go` session — possibly still searching, possibly
    /// finished-but-not-yet-joined. Joined (and `state` restored) by
    /// [`Engine::wait_for_search_finished`].
    session: Option<Session>,
}

/// Everything one search session owns: the [`Worker`] (board copy, search
/// stack, root moves, and the per-`go` `TimeManager`) plus the per-GAME
/// [`TimePersist`]. This is v1's stand-in for the threading doc's
/// main-worker `SearchManager` (doc 12 §3.8) — deliberately NOT `Engine`
/// state (§3.1), so the migration to the pool is a move, not a redesign.
struct SearchState {
    worker: Worker,
    persist: TimePersist,
}

impl SearchState {
    fn new() -> SearchState {
        SearchState {
            worker: Worker::new(Board::startpos()),
            persist: TimePersist::default(),
        }
    }
}

/// One `go`'s cross-thread handles — the v1 stand-in for the threading
/// doc's `PoolShared` flag block (doc 12 §3.2). Both flags are pure
/// signals (no data published through them), hence `Relaxed` everywhere.
struct Session {
    /// Set by `stop`/`quit`/blocking commands (implicit stop) and by the
    /// search's own hard limits in `check_time`.
    stop: Arc<AtomicBool>,
    /// True from `go ponder` until `ponderhit`/`stop`: while set, the
    /// busy-wait in [`run_search`] withholds `bestmove` (§3.2 step 3).
    ponder: Arc<AtomicBool>,
    /// Returns the [`SearchState`] to the UCI thread at join.
    handle: thread::JoinHandle<SearchState>,
}

impl Engine {
    pub fn new() -> Engine {
        let options = Options::new();
        // The startup default allocation (Hash default, 16 MB). Doc 01
        // §3.4 permits this one to be fatal; `resize`'s halving fallback
        // makes even it survive anything short of total memory exhaustion.
        let tt = Arc::new(TranspositionTable::new(options.hash_mb() as usize));
        Engine {
            board: Board::startpos(),
            options,
            tt,
            state: Some(SearchState::new()),
            session: None,
        }
    }

    /// Exclusive TT access for resize/clear, legal only between searches
    /// (doc 13 §3.6, D23). After [`Engine::wait_for_search_finished`] the
    /// UCI thread holds the only strong reference (sessions clone the
    /// handle and drop it at join), so `Arc::get_mut` cannot fail; a
    /// `None` here would be an engine bug, never a protocol-input error.
    fn tt_mut(&mut self) -> &mut TranspositionTable {
        Arc::get_mut(&mut self.tt).expect("no search running: UCI thread holds the only TT handle")
    }

    /// `ucinewgame` (doc 12 §3.3/§3.7): clear the TT (doc 08 §3.14) and
    /// reset the per-game TM state (the "first move of the game" defaults
    /// measurably affect move-1 time usage) and the `nodestime` budget.
    /// Deliberately does NOT touch the session position — GUIs send
    /// `position` separately. History clears join with doc 09.
    pub fn new_game(&mut self) {
        self.wait_for_search_finished();
        self.tt_mut().clear();
        let state = self.state.as_mut().expect("engine idle after wait");
        state.persist = TimePersist::default();
        state.worker.tm.new_game();
    }

    /// Join any running search, first applying the implicit-stop rule
    /// (doc 12 §3.2): a blocking command that arrives mid-search sets
    /// `stop`, the pending `bestmove` flushes normally (the exactly-once
    /// contract is untouched), and the wait is therefore always bounded —
    /// `go infinite` followed by a mid-search `position` can never
    /// deadlock. No-op when idle.
    pub fn wait_for_search_finished(&mut self) {
        if let Some(session) = self.session.take() {
            session.stop.store(true, Ordering::Relaxed);
            self.state = Some(session.handle.join().unwrap_or_else(|_| {
                // A panicked worker must not kill the UCI session ("never
                // panic on input", §3.3): restart from fresh state. The
                // per-game TM state is lost — acceptable in a path that
                // indicates a bug anyway (the panic went to stderr).
                SearchState::new()
            }));
        }
    }

    /// `stop`: flip the shared flag; the search thread prints its
    /// `bestmove` and finishes (§3.2). Harmless no-op when idle (§3.3).
    pub fn stop_search(&self) {
        if let Some(session) = &self.session {
            session.stop.store(true, Ordering::Relaxed);
        }
    }

    /// `ponderhit`: the GUI played the expected move — clear the ponder
    /// flag so the busy-wait releases and the search runs on our clock
    /// (§3.10 v1 subset; the full ponder state machine is doc 12 step 9).
    /// Harmless no-op when idle (§3.3).
    pub fn ponderhit(&self) {
        if let Some(session) = &self.session {
            session.ponder.store(false, Ordering::Relaxed);
        }
    }

    /// Test-only: join the search WITHOUT the implicit stop, letting a
    /// self-terminating `go` (depth/nodes/movetime) run to natural
    /// completion so its output is deterministic. Would hang on
    /// `go infinite` — deliberate, tests must stop those explicitly.
    #[cfg(test)]
    fn join_search(&mut self) {
        if let Some(session) = self.session.take() {
            self.state = Some(session.handle.join().unwrap_or_else(|_| SearchState::new()));
        }
    }
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

// ---------------------------------------------------------------------------
// The command loop (design/12 §3.3)
// ---------------------------------------------------------------------------

/// The blocking UCI loop, generic over the byte streams (`impl BufRead` /
/// [`SharedOut`]) so unit tests drive it with in-memory buffers while
/// `run()` passes stdin/stdout. `W: Send + 'static` because each `go`
/// clones the shared sink into its search thread.
///
/// Loop-exit rules (doc 12 §3.3): `quit`, stdin EOF, or a read error all
/// leave the loop; a write error means the GUI is gone, same thing. No
/// input line may panic. On exit, any running search is stopped and
/// joined so its `bestmove` flushes before the process ends.
fn uci_loop<W: Write + Send + 'static>(input: impl BufRead, out: &SharedOut<W>) {
    let mut engine = Engine::new();
    for line in input.lines() {
        let Ok(line) = line else { break };
        out.lock().log_input(&line);
        match dispatch(&mut engine, &line, out) {
            Ok(true) => {}
            Ok(false) | Err(_) => break,
        }
        // Piped stdout is block-buffered in Rust: flush per command or a
        // GUI waiting on `uciok` hangs forever.
        if out.lock().flush().is_err() {
            break;
        }
    }
    // `quit`/EOF: stop + join (doc 12 §3.2) — the exactly-once `bestmove`
    // contract holds even for a search interrupted by process exit.
    engine.wait_for_search_finished();
    let _ = out.lock().flush();
}

/// Handle one input line (the doc 12 §3.3 dispatch shape).
/// Returns `Ok(false)` exactly when the loop should end (`quit`).
///
/// Commands legal DURING a search (`stop`, `ponderhit`, `isready` — a
/// liveness ping, not a barrier) are handled without waiting; every other
/// state-touching command first goes through
/// [`Engine::wait_for_search_finished`] and its implicit-stop rule
/// (§3.2), so the wait is always bounded.
fn dispatch<W: Write + Send + 'static>(
    engine: &mut Engine,
    line: &str,
    out: &SharedOut<W>,
) -> io::Result<bool> {
    let mut tokens = line.split_whitespace();
    match tokens.next() {
        Some("uci") => {
            // One lock for the whole handshake block: GUI parsers may
            // assume the id/option/uciok lines arrive uninterrupted.
            let mut out = out.lock();
            writeln!(out, "id name {NAME}")?;
            writeln!(out, "id author {AUTHOR}")?;
            for entry in engine.options.iter() {
                writeln!(out, "{entry}")?;
            }
            writeln!(out, "uciok")?;
        }
        Some("isready") => writeln!(out.lock(), "readyok")?,
        Some("setoption") => {
            engine.wait_for_search_finished();
            cmd_setoption(engine, tokens, &mut *out.lock())?;
        }
        Some("ucinewgame") => engine.new_game(), // waits internally
        Some("position") => {
            engine.wait_for_search_finished();
            cmd_position(engine, tokens, &mut *out.lock())?;
        }
        Some("go") => cmd_go(engine, tokens, out)?,
        Some("stop") => engine.stop_search(),
        Some("ponderhit") => engine.ponderhit(),
        Some("d") => {
            engine.wait_for_search_finished(); // dev commands wait (§3.15)
            print_board(&mut *out.lock(), &engine.board)?;
        }
        Some("quit") => return Ok(false),
        // Accepted-but-ignored per the UCI spec (doc 12 §3.3): routing
        // these through the unknown branch would spam GUIs that send them.
        Some("debug" | "register") => {}
        // Everything else is absorbed, never fatal (doc 12 §3.3).
        Some(_) => writeln!(out.lock(), "info string Unknown command: '{}'", line.trim())?,
        None => {} // blank line
    }
    Ok(true)
}

/// `setoption name <id...> [value <x...>]` (doc 12 §3.5): the name may
/// contain spaces — tokens after `name` up to the literal `value` (or end
/// of line); the value is the rest of the line, whitespace-joined. All
/// errors answer `info string` and change nothing.
fn cmd_setoption<W: Write>(
    engine: &mut Engine,
    mut tokens: SplitWhitespace,
    out: &mut UciOut<W>,
) -> io::Result<()> {
    if tokens.next() != Some("name") {
        return writeln!(out, "info string Malformed setoption (expected 'name')");
    }
    let mut name_tokens: Vec<&str> = Vec::new();
    let mut value: Option<String> = None;
    while let Some(tok) = tokens.next() {
        if tok == "value" {
            value = Some(tokens.collect::<Vec<_>>().join(" "));
            break;
        }
        name_tokens.push(tok);
    }
    let name = name_tokens.join(" ");
    match engine.options.set(&name, value.as_deref()) {
        Err(msg) => writeln!(out, "info string {msg}"),
        // Side effects, keyed off the canonical name. They run
        // synchronously so a following `isready` is the GUI's sync point
        // (§3.5); the searches-must-be-finished precondition is trivially
        // true until P5.
        Ok("Debug Log File") => {
            let path = engine.options.debug_log_file().to_string();
            out.set_log_path(&path)
        }
        // `Hash`: resize + clear (doc 12 §3.5 table). The stored option
        // value is already range-clamped, and `resize` is fallible
        // (doc 08 §3.13): on allocation shortfall it fell back through
        // halved sizes — announce the achieved size and keep playing,
        // never abort mid-session.
        Ok("Hash") => {
            let requested = engine.options.hash_mb() as usize;
            let got = engine.tt_mut().resize(requested);
            if got == requested {
                Ok(())
            } else {
                writeln!(
                    out,
                    "info string Failed to allocate {requested} MB hash, using {got} MB"
                )
            }
        }
        Ok("Clear Hash") => {
            // Clears the TT (doc 08 §3.14); histories join with doc 09.
            engine.tt_mut().clear();
            Ok(())
        }
        // `Threads` rebuilds the pool (P8). Other values are stored
        // above; nothing further yet.
        Ok(_) => Ok(()),
    }
}

/// `position (startpos | fen <f1..f6>) [moves <m>...]` (doc 12 §3.3).
///
/// Moves are parsed against the legal move list of the evolving position
/// and applied ATOMICALLY on a scratch board: if any token fails, the
/// engine's position is left unchanged (SF exits the process here; we
/// prefer to stay alive — a documented robustness divergence, §9.3).
/// Applying the whole list through `make_move` builds the ply-indexed
/// `StateInfo` history inside `Board` (D8) — exactly what repetition
/// detection and rule-50 need.
fn cmd_position<W: Write>(
    engine: &mut Engine,
    mut tokens: SplitWhitespace,
    out: &mut UciOut<W>,
) -> io::Result<()> {
    let mut board = match tokens.next() {
        Some("startpos") => {
            match tokens.next() {
                None | Some("moves") => {}
                Some(other) => {
                    return writeln!(
                        out,
                        "info string Malformed position command at '{other}' (expected 'moves')"
                    );
                }
            }
            Board::startpos()
        }
        Some("fen") => {
            // Collect tokens up to the literal `moves` (which take_while
            // also consumes) or end of line; `from_fen` accepts 4-6
            // fields.
            let fen = (&mut tokens)
                .take_while(|&tok| tok != "moves")
                .collect::<Vec<_>>()
                .join(" ");
            match Board::from_fen(&fen) {
                Ok(board) => board,
                Err(e) => {
                    return writeln!(
                        out,
                        "info string Invalid FEN '{fen}' in position command: {e}"
                    );
                }
            }
        }
        _ => {
            return writeln!(
                out,
                "info string Malformed position command (expected 'startpos' or 'fen')"
            );
        }
    };
    // Whatever remains is the move list.
    for tok in tokens {
        let Some(m) = move_from_uci(&board, tok) else {
            return writeln!(out, "info string Invalid move '{tok}' in position command");
        };
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
    }
    engine.board = board;
    Ok(())
}

/// `go [<param>...]` (doc 12 §3.6, §3.2 step 1): finish any previous
/// session (implicit stop — a well-behaved GUI never overlaps, but we
/// tolerate it), parse the limits, and hand the search state to a fresh
/// worker thread. This thread returns to reading stdin immediately; the
/// worker prints every `info` line and the exactly-once `bestmove`.
fn cmd_go<W: Write + Send + 'static>(
    engine: &mut Engine,
    tokens: SplitWhitespace,
    out: &SharedOut<W>,
) -> io::Result<()> {
    engine.wait_for_search_finished();
    let limits = parse_go(&engine.board, tokens, &mut *out.lock())?;
    // `go perft <d>` is a dev command (doc 12 §3.15), wired with the rest
    // of the in-session dev surface (step 12) — say so rather than
    // launching a search the GUI did not ask for.
    if limits.perft.is_some() {
        return writeln!(
            out.lock(),
            "info string go perft is not implemented yet (dev commands land with doc 12 step 12)"
        );
    }
    let mut state = engine.state.take().expect("engine idle after wait");
    state.worker.board = engine.board.clone(); // history Vec included (D8)
    state.worker.limits = limits;
    // Fresh flags per `go` (§3.2 step 1): stop = false, ponder from the
    // limits. (`increase_depth` joins with the soft stop, step 8.) The
    // worker shares the ponder flag: its `check_time` ponder guard must
    // observe `ponderhit` (doc 12 §3.8).
    let stop = Arc::new(AtomicBool::new(false));
    let ponder = Arc::new(AtomicBool::new(state.worker.limits.ponder));
    state.worker.ponder = Arc::clone(&ponder);
    let spawned = {
        let stop = Arc::clone(&stop);
        let ponder = Arc::clone(&ponder);
        let options = engine.options.clone();
        // The session's TT handle (D23; doc 13 §3.4's per-job carry):
        // cloned per `go`, held by the search thread for the job's
        // duration, dropped when `run_search` returns — between searches
        // the UCI thread is again the only holder, so `Engine::tt_mut`'s
        // `Arc::get_mut` keeps working.
        let tt = Arc::clone(&engine.tt);
        let out = out.clone();
        thread::Builder::new()
            .name("search-main".to_string())
            .spawn(move || run_search(state, &stop, &ponder, &options, &tt, &out))
    };
    match spawned {
        Ok(handle) => {
            engine.session = Some(Session {
                stop,
                ponder,
                handle,
            });
            Ok(())
        }
        Err(e) => {
            // Spawn failure (resource exhaustion): the state moved into
            // the dropped closure, so rebuild it, and honor the
            // exactly-once contract with the best answer available.
            engine.state = Some(SearchState::new());
            let mut legal = MoveList::new();
            movegen::generate_legal(&engine.board, &mut legal);
            let m = legal.first().copied().unwrap_or(Move::NULL);
            let mut out = out.lock();
            writeln!(out, "info string Could not start search thread: {e}")?;
            writeln!(out, "bestmove {}", move_to_uci(m))
        }
    }
}

/// One `go` session, on the search thread (doc 12 §3.2 steps 2–4): time
/// management init, iterative deepening with `info` emission (§3.13),
/// the busy-wait rule for infinite/ponder searches, then the
/// exactly-once `bestmove`. Returns the [`SearchState`] to the UCI
/// thread through the join handle.
///
/// Output errors are swallowed (`let _`): if the GUI closed the pipe
/// there is nobody left to tell, and the session still has to unwind
/// cleanly (§3.3).
fn run_search<W: Write>(
    mut state: SearchState,
    stop: &AtomicBool,
    ponder: &AtomicBool,
    options: &Options,
    tt: &TranspositionTable,
    out: &SharedOut<W>,
) -> SearchState {
    let worker = &mut state.worker;
    // §3.2 step 2 — TM init on the search thread: turns wtime/btime into
    // the optimum/maximum budgets `check_time` enforces, and may rewrite
    // the clock fields under nodestime. (The post-search
    // `advance_nodes_time` accounting lands with doc 12 step 11.)
    let mut limits = worker.limits.clone();
    let (us, game_ply) = (worker.board.side_to_move(), worker.board.game_ply());
    worker
        .tm
        .init(&mut limits, us, game_ply, options, &mut state.persist);
    worker.limits = limits;
    let start_time = worker.limits.start_time;

    // The single per-`go` TT generation bump, after TM init and before
    // the search starts (doc 08 §3.9; doc 13 §3.7 B's ordering).
    tt.new_search();

    // §3.13: one pre-built String, one locked write per info line. The
    // PV head of the last line sent drives the final flush below.
    let mut last_sent: Option<Move> = None;
    let multi_pv = 1usize; // fixed until the MultiPV option lands (step 10)
    worker.iterative_deepening(tt, stop, &mut |info| {
        // Throttle (§3.13): below INFO_THROTTLE_NODES only the
        // iteration's LAST MultiPV line prints — at MultiPV 1 that is
        // every completed line, so nothing is ever suppressed yet.
        if info.nodes <= INFO_THROTTLE_NODES && info.multipv != multi_pv {
            return;
        }
        let elapsed_ms = start_time.elapsed().as_millis() as i64;
        let mut out = out.lock();
        let _ = writeln!(out, "{}", format_info(info, elapsed_ms));
        let _ = out.flush();
        last_sent = info.pv.first().copied();
    });

    if worker.root_moves.is_empty() {
        // §3.2 step 2: no legal moves — mate (in check) or stalemate.
        let score = if worker.board.checkers().is_empty() {
            "cp 0"
        } else {
            "mate 0"
        };
        let mut out = out.lock();
        let _ = writeln!(out, "info depth 0 score {score}");
        let _ = out.flush();
    }

    // §3.2 step 3 — the bestmove-exactly-once busy-wait rule: in ponder
    // or infinite mode, bestmove may only be printed after "stop" or
    // "ponderhit". Burns one core for the (typically tiny) window — SF
    // makes the same trade (doc 12 §6, Open Question 2).
    while !stop.load(Ordering::Relaxed)
        && (ponder.load(Ordering::Relaxed) || worker.limits.infinite)
    {
        std::hint::spin_loop();
    }
    stop.store(true, Ordering::Relaxed);

    // §3.2 step 4 — final PV flush + bestmove in one locked block, so the
    // pair can never interleave with UCI-thread output.
    let mut out = out.lock();
    if let Some(best) = worker.best_move() {
        let rm = &worker.root_moves[0];
        // §3.13: the GUI's last-seen PV must match the bestmove. A
        // mid-iteration stop can promote a root move whose line was never
        // sent (or stop before the first report); republish — unless the
        // front move has no trusted score at all (stopped before any
        // depth-1 result), in which case there is nothing to say.
        if last_sent != Some(best) && rm.uci_score != -VALUE_INFINITE {
            let elapsed_ms = start_time.elapsed().as_millis() as i64;
            let info = SearchInfo {
                depth: worker.root_depth.max(1),
                sel_depth: rm.sel_depth,
                multipv: 1,
                score: rm.uci_score,
                lowerbound: rm.score_lowerbound,
                upperbound: rm.score_upperbound,
                nodes: worker.nodes,
                hashfull: tt.hashfull(0),
                pv: &rm.pv,
            };
            let _ = writeln!(out, "{}", format_info(&info, elapsed_ms));
        }
        // Ponder move = pv[1] when the line is long enough; for a 1-ply
        // PV, fall back to probing the TT after making pv[0] (doc 12
        // §3.2 step 4).
        let ponder_mv = worker
            .root_pv()
            .get(1)
            .copied()
            .or_else(|| ponder_from_tt(tt, &worker.board, best));
        match ponder_mv {
            Some(p) => {
                let _ = writeln!(out, "bestmove {best} ponder {p}");
            }
            None => {
                let _ = writeln!(out, "bestmove {best}");
            }
        }
    } else {
        let _ = writeln!(out, "bestmove (none)");
    }
    let _ = out.flush();
    drop(out);
    state
}

/// The 1-ply-PV ponder fallback (doc 12 §3.2 step 4): make the best move
/// on a scratch board and probe the TT there; offer the stored move as
/// the ponder move only if the resulting position is not an immediate
/// draw and the move is legal there — the doc 08 §3.5 validation
/// contract, discharged by legal-list membership: a collided or garbage
/// move simply matches nothing. Cold path (once per `go`), so the clone
/// and the O(legal) scan are fine.
fn ponder_from_tt(tt: &TranspositionTable, board: &Board, best: Move) -> Option<Move> {
    let mut b = board.clone();
    b.make_move(best, b.gives_check(best));
    if b.is_draw(1) {
        return None;
    }
    let probe = tt.probe(b.key());
    if !probe.hit || probe.data.mv == Move::NULL {
        return None;
    }
    let mut legal = MoveList::new();
    movegen::generate_legal(&b, &mut legal);
    legal.contains(&probe.data.mv).then_some(probe.data.mv)
}

// ---------------------------------------------------------------------------
// info formatting (design/12 §3.13–§3.14)
// ---------------------------------------------------------------------------

/// design/12 §3.13's `NODES_LIMIT_OUTPUT`, under the doc's preferred
/// name: `currmove` lines and intermediate aspiration bound updates wait
/// for this many nodes, and completed PV lines below it print only when
/// the iteration's last MultiPV line completes. At v1's MultiPV = 1 the
/// PV rule always prints; `currmove` emission itself lands with the
/// MovePicker's root loop (docs 09/12 step 6+).
const INFO_THROTTLE_NODES: u64 = 10_000_000;

/// Format one completed PV line (doc 12 §3.13, v1 field scope): depth,
/// seldepth, score (with bound suffix), nodes, nps, hashfull (the TT
/// permille estimate, doc 08 §3.10), wall-clock time, pv — the doc's SF
/// field order, minus the deferred `multipv`/`wdl`/`tbhits`. One `String`
/// so the caller can emit it as a single write.
fn format_info(info: &SearchInfo, elapsed_ms: i64) -> String {
    use std::fmt::Write as _; // writing to a String cannot fail
    let time = elapsed_ms.max(0);
    let nps = info.nodes.saturating_mul(1000) / time.max(1) as u64;
    let mut line = format!(
        "info depth {} seldepth {} score {}",
        info.depth,
        info.sel_depth,
        format_score(info.score)
    );
    if info.lowerbound {
        line.push_str(" lowerbound");
    } else if info.upperbound {
        line.push_str(" upperbound");
    }
    let _ = write!(
        line,
        " nodes {} nps {nps} hashfull {} time {time} pv",
        info.nodes, info.hashfull
    );
    for m in info.pv {
        let _ = write!(line, " {m}");
    }
    line
}

/// A score as UCI prints it (doc 12 §3.14). `Mate` counts full MOVES,
/// signed: positive "we mate in M", negative "we are mated in M".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UciScore {
    Cp(Value),
    Mate(i32),
}

impl fmt::Display for UciScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            UciScore::Cp(v) => write!(f, "cp {v}"),
            UciScore::Mate(m) => write!(f, "mate {m}"),
        }
    }
}

/// Split a ply-relative internal value into the UCI `cp`/`mate` form
/// (doc 12 §3.14): mate plies convert to full moves — rounding up on the
/// winning side (our mating move counts), truncating on the losing side,
/// matching SF exactly (Rust `/` truncates like C++). Outside the mate
/// bands the value passes through as centipawns: `to_cp` is the identity
/// while eval phase A/B prints raw internal values; the WDL normalization
/// seam replaces it in phase C (§3.14).
pub fn format_score(v: Value) -> UciScore {
    debug_assert!(crate::types::is_valid(v) && v.abs() != VALUE_INFINITE);
    if is_mate(v) {
        let plies = VALUE_MATE - v;
        UciScore::Mate((plies + 1) / 2)
    } else if is_mated(v) {
        let plies = VALUE_MATE + v;
        UciScore::Mate(-(plies / 2))
    } else {
        UciScore::Cp(v)
    }
}

/// Parse the `go` parameter tokens into [`Limits`] (doc 12 §3.6):
/// parameters in any order, except `searchmoves` which consumes all
/// remaining tokens (SF behavior; matches how GUIs emit it). Unknown or
/// malformed tokens are skipped silently (protocol robustness); bad
/// `searchmoves` tokens warn via `info string` and are dropped —
/// root-move building decides later whether to ignore the whole filter
/// (§3.6). `start_time` is stamped at entry.
fn parse_go<W: Write>(
    board: &Board,
    mut tokens: SplitWhitespace,
    out: &mut UciOut<W>,
) -> io::Result<Limits> {
    /// Parse the next token as a number; `None` (and the token still
    /// consumed) when missing or malformed — mirroring how SF's stream
    /// extraction leaves the field alone.
    fn next_num<T: FromStr>(tokens: &mut SplitWhitespace) -> Option<T> {
        tokens.next().and_then(|tok| tok.parse().ok())
    }

    let mut limits = Limits::new();
    while let Some(tok) = tokens.next() {
        match tok {
            "searchmoves" => {
                let mut any_token = false;
                for tok in tokens.by_ref() {
                    any_token = true;
                    match move_from_uci(board, tok) {
                        Some(m) => limits.searchmoves.push(m),
                        None => {
                            writeln!(out, "info string Invalid searchmoves token '{tok}' ignored")?
                        }
                    }
                }
                // §3.6: every token unparseable/illegal in a position
                // that has legal moves — root-move building will ignore
                // the filter wholesale; tell the GUI why it gets a
                // normal bestmove instead of `(none)`.
                if any_token && limits.searchmoves.is_empty() {
                    let mut legal = MoveList::new();
                    movegen::generate_legal(board, &mut legal);
                    if !legal.is_empty() {
                        writeln!(out, "info string Invalid searchmoves list ignored")?;
                    }
                }
            }
            // wtime/btime may be negative in GUI lag situations; parse
            // into i64 and let TM's max(1, ...) clamps handle it (§3.6).
            "wtime" => {
                if let Some(v) = next_num(&mut tokens) {
                    limits.time[Color::White as usize] = v;
                }
            }
            "btime" => {
                if let Some(v) = next_num(&mut tokens) {
                    limits.time[Color::Black as usize] = v;
                }
            }
            "winc" => {
                if let Some(v) = next_num(&mut tokens) {
                    limits.inc[Color::White as usize] = v;
                }
            }
            "binc" => {
                if let Some(v) = next_num(&mut tokens) {
                    limits.inc[Color::Black as usize] = v;
                }
            }
            "movestogo" => {
                if let Some(v) = next_num(&mut tokens) {
                    limits.movestogo = v;
                }
            }
            "depth" => limits.depth = next_num(&mut tokens),
            "nodes" => limits.nodes = next_num(&mut tokens),
            "movetime" => limits.movetime = next_num(&mut tokens),
            "mate" => limits.mate = next_num(&mut tokens),
            "perft" => limits.perft = next_num(&mut tokens),
            "infinite" => limits.infinite = true,
            "ponder" => limits.ponder = true,
            _ => {} // unknown `go` tokens are skipped silently (§3.6)
        }
    }
    Ok(limits)
}

/// `d` output (doc 12 §3.15): diagram, `Fen:`, `Key:` (16 hex digits,
/// D9), and `Checkers:` (the checking pieces' squares).
fn print_board(out: &mut impl Write, board: &Board) -> io::Result<()> {
    writeln!(out, "{board}")?;
    writeln!(out)?;
    writeln!(out, "Fen: {}", board.to_fen())?;
    writeln!(out, "Key: {:016X}", board.key())?;
    let checkers: Vec<String> = board
        .checkers()
        .into_iter()
        .map(|sq| sq.to_string())
        .collect();
    writeln!(out, "Checkers: {}", checkers.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;
    use crate::types::Square;

    /// Drive the loop over a scripted stdin, returning captured stdout.
    /// The loop's exit path joins any search thread, so the capture is
    /// complete and the `Arc` unwrap in `into_inner` cannot fail.
    fn session(input: &str) -> String {
        let out = SharedOut::new(Vec::new());
        uci_loop(input.as_bytes(), &out);
        String::from_utf8(out.into_inner()).expect("engine output is UTF-8")
    }

    /// Run one command against a fresh engine, returning (engine, output).
    fn command(line: &str) -> (Engine, String) {
        let mut engine = Engine::new();
        let out = command_on(&mut engine, line);
        (engine, out)
    }

    /// Run one command against an existing engine, returning its output.
    /// Joins (without stopping) any search the command started, so a
    /// self-terminating `go` line's whole `info`/`bestmove` stream is
    /// captured deterministically after natural completion.
    fn command_on(engine: &mut Engine, line: &str) -> String {
        let out = SharedOut::new(Vec::new());
        dispatch(engine, line, &out).expect("Vec sink never fails");
        engine.join_search();
        String::from_utf8(out.into_inner()).expect("engine output is UTF-8")
    }

    fn parse_limits(fen: &str, params: &str) -> (Limits, String) {
        let board = Board::from_fen(fen).expect("test FEN");
        let mut out = UciOut::new(Vec::new());
        let limits =
            parse_go(&board, params.split_whitespace(), &mut out).expect("Vec sink never fails");
        (limits, String::from_utf8(out.out).unwrap())
    }

    #[test]
    fn uci_handshake() {
        // doc 12 §3.3: id name, id author, the options block, then uciok.
        let out = session("uci\nquit\n");
        assert!(out.contains("id name chessengine "), "{out}");
        assert!(out.contains("id author "), "{out}");
        let id_at = out.find("id name").unwrap();
        let uciok_at = out.find("uciok\n").unwrap();
        assert!(id_at < uciok_at, "uciok must follow the id lines: {out}");
    }

    #[test]
    fn uci_lists_the_v1_options_verbatim() {
        // doc 12 §3.5/§8.1: exactly the v1 table, registration order,
        // correct types/defaults/ranges.
        let out = session("uci\nquit\n");
        let expected = "\
option name Debug Log File type string default <empty>
option name Hash type spin default 16 min 1 max 33554432
option name Threads type spin default 1 min 1 max 1024
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 256
option name Ponder type check default false
option name Move Overhead type spin default 10 min 0 max 5000
option name nodestime type spin default 0 min 0 max 10000
uciok
";
        assert!(out.ends_with(expected), "{out}");
        assert_eq!(out.matches("option name").count(), 8, "{out}");
    }

    #[test]
    fn isready_answers_readyok() {
        assert!(session("isready\nquit\n").contains("readyok\n"));
    }

    #[test]
    fn d_prints_diagram_fen_key_checkers() {
        let out = session("d\nquit\n");
        assert!(out.contains(&format!("Fen: {START_FEN}")), "{out}");
        let key = Board::startpos().key();
        assert!(out.contains(&format!("Key: {key:016X}")), "{out}");
        assert!(out.contains("Checkers: \n"), "{out}");
    }

    #[test]
    fn d_shows_checkers_squares() {
        let (mut engine, _) = command("position fen 4k3/4R3/8/8/8/8/8/4K3 b - - 0 1");
        let out = command_on(&mut engine, "d");
        assert!(out.contains("Checkers: e7\n"), "{out}");
    }

    #[test]
    fn quit_stops_reading() {
        // The second `uci` must never be answered.
        let out = session("uci\nquit\nuci\n");
        assert_eq!(out.matches("uciok").count(), 1, "{out}");
    }

    #[test]
    fn eof_is_quit() {
        // doc 12 §3.3: EOF == quit (cutechess/fastchess close the pipe).
        // The script has no `quit`; the loop must still terminate cleanly.
        assert!(session("isready\n").contains("readyok\n"));
    }

    #[test]
    fn garbage_is_absorbed_never_fatal() {
        // design/01 §4 step 5: garbage lines produce `info string` and the
        // loop keeps serving commands afterwards.
        let out = session("banana split\nuci\n###\nisready\nquit\n");
        assert!(
            out.contains("info string Unknown command: 'banana split'"),
            "{out}"
        );
        assert!(out.contains("uciok\n"), "{out}");
        assert!(out.contains("readyok\n"), "{out}");
    }

    #[test]
    fn debug_and_register_are_consumed_silently() {
        // doc 12 §3.3: spec commands we don't implement produce NO output
        // (no unknown-command spam at GUIs that send them).
        assert_eq!(session("debug on\nregister later\nquit\n"), "");
    }

    #[test]
    fn stop_and_ponderhit_are_idle_noops() {
        // doc 12 §3.3: harmless with no active search — and silent.
        assert_eq!(session("stop\nponderhit\nucinewgame\nquit\n"), "");
    }

    #[test]
    fn blank_lines_are_ignored() {
        let out = session("\n   \n\t\nisready\nquit\n");
        assert!(!out.contains("Unknown command"), "{out}");
        assert!(out.contains("readyok\n"), "{out}");
    }

    #[test]
    fn no_panic_on_junk_torrent() {
        // "Never panic on any input" — near-miss commands, dev commands
        // that exist in the grammar but not yet in the loop (they must
        // absorb, not hit todo! stubs), non-ASCII, control characters, and
        // malformed-but-real P4 commands.
        let junk = "uciready\nucii\nquitt quit\nd d d\nположение\n\u{1}\u{7f}\n\
                    go depth 4\nposition startpos moves e2e4\n\
                    position fen banana\nposition moves e2e4\n\
                    setoption name Hash value banana\nsetoption Hash 16\n\
                    setoption name\ngo wtime\ngo searchmoves zzz\n\
                    stop\nponderhit\nucinewgame\neval\nbench\nperft 1\n\
                    go perft 5\nquit\n";
        let out = session(junk);
        // ... and the loop stayed alive through all of it up to `quit`.
        assert!(
            out.contains("info string Unknown command: 'perft 1'"),
            "{out}"
        );
    }

    // -- move text (doc 12 §3.4; §5 "Unit — text") ------------------------

    /// Every legal move of `fen` must round-trip through the UCI text
    /// layer, including uppercase input.
    fn assert_roundtrips(fen: &str) {
        let board = Board::from_fen(fen).expect("test FEN");
        let mut list = MoveList::new();
        movegen::generate_legal(&board, &mut list);
        assert!(!list.is_empty(), "test position has no legal moves: {fen}");
        for &m in list.iter() {
            let text = move_to_uci(m);
            assert_eq!(move_from_uci(&board, &text), Some(m), "{fen} {text}");
            let upper = text.to_ascii_uppercase();
            assert_eq!(move_from_uci(&board, &upper), Some(m), "{fen} {upper}");
        }
    }

    #[test]
    fn move_text_roundtrips_castling_ep_promotions() {
        // All four castlings available.
        assert_roundtrips("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_roundtrips("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        // En passant (d5xe6 and f5xe6).
        assert_roundtrips("4k3/8/8/3PpP2/8/8/8/4K3 w - e6 0 2");
        // All four promotions, push and capture.
        assert_roundtrips("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        assert_roundtrips("4k3/8/8/8/8/8/7p/4K1N1 b - - 0 1");
    }

    #[test]
    fn move_text_sentinels_and_rejections() {
        let board = Board::startpos();
        assert_eq!(move_to_uci(Move::NULL), "(none)");
        // Promotion piece must match a generated move exactly.
        assert_eq!(
            move_to_uci(Move::promotion(
                Square::A7,
                Square::A8,
                crate::types::PieceType::Queen
            )),
            "a7a8q"
        );
        for bad in ["", "e2e5", "e7e5", "0000", "(none)", "e2e4q", "banana"] {
            assert_eq!(move_from_uci(&board, bad), None, "{bad:?}");
        }
    }

    // -- position (doc 12 §3.3; §5 integration item 2) ---------------------

    #[test]
    fn position_startpos_moves_reaches_expected_fen() {
        let (engine, out) = command("position startpos moves e2e4 e7e5");
        assert_eq!(out, "", "position produces no output on success");
        assert_eq!(
            engine.board.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn position_moves_build_history_for_repetition() {
        // Four reversible knight moves: the start position repeats, and
        // the board history (D8) must know (doc 12 §3.3).
        let (engine, _) =
            command("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
        assert!(engine.board.has_repeated());
    }

    #[test]
    fn position_fen_with_and_without_moves() {
        const KIWIPETE: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let (engine, _) = command(&format!("position fen {KIWIPETE}"));
        assert_eq!(engine.board.to_fen(), KIWIPETE);
        // With moves; both castlings through the text layer.
        let (engine, out) = command(&format!("position fen {KIWIPETE} moves e1g1 e8g8"));
        assert_eq!(out, "");
        assert!(
            engine.board.to_fen().starts_with("r4rk1/p1ppqpb1/"),
            "{}",
            engine.board.to_fen()
        );
        // 4-field FEN (from_fen accepts 4-6 fields; clocks default 0/1).
        let (engine, out) = command("position fen 4k3/8/8/8/8/8/8/4K2R w K -");
        assert_eq!(out, "");
        assert_eq!(engine.board.to_fen(), "4k3/8/8/8/8/8/8/4K2R w K - 0 1");
    }

    #[test]
    fn position_ep_square_appears_when_capturable() {
        let (engine, _) = command("position startpos moves e2e4 d7d5 e4e5 f7f5");
        assert_eq!(
            engine.board.to_fen(),
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        );
    }

    #[test]
    fn position_invalid_move_is_atomic() {
        let mut engine = Engine::new();
        command_on(&mut engine, "position startpos moves d2d4");
        let before = engine.board.to_fen();
        // Second e2e4 is illegal (square empty after the first): the whole
        // command must be rejected, leaving the d2d4 position.
        let out = command_on(&mut engine, "position startpos moves e2e4 e2e4");
        assert!(
            out.contains("info string Invalid move 'e2e4' in position command"),
            "{out}"
        );
        assert_eq!(engine.board.to_fen(), before);
    }

    #[test]
    fn position_invalid_fen_is_rejected() {
        let mut engine = Engine::new();
        command_on(&mut engine, "position startpos moves d2d4");
        let before = engine.board.to_fen();
        let out = command_on(&mut engine, "position fen banana split");
        assert!(out.contains("info string Invalid FEN"), "{out}");
        assert_eq!(engine.board.to_fen(), before);
    }

    // -- setoption (doc 12 §3.5; §5 "Unit — parser") -----------------------

    #[test]
    fn setoption_roundtrips_every_consumer_accessor() {
        let mut engine = Engine::new();
        for line in [
            "setoption name Hash value 256",
            "setoption name Threads value 4",
            "setoption name MultiPV value 3",
            "setoption name Ponder value true",
            "setoption name Move Overhead value 100",
            "setoption name nodestime value 1000",
        ] {
            assert_eq!(command_on(&mut engine, line), "", "{line}");
        }
        assert_eq!(engine.options.hash_mb(), 256);
        assert_eq!(engine.options.threads(), 4);
        assert_eq!(engine.options.multipv(), 3);
        assert!(engine.options.ponder());
        assert_eq!(engine.options.move_overhead_ms(), 100);
        assert_eq!(engine.options.nodestime(), 1000);
    }

    #[test]
    fn setoption_is_case_insensitive_and_clamps() {
        let mut options = Options::new();
        assert_eq!(options.set("hash", Some("0")), Ok("Hash"));
        assert_eq!(options.hash_mb(), 1); // clamped up to min
        assert_eq!(options.set("HASH", Some("99999999999")), Ok("Hash"));
        assert_eq!(options.hash_mb(), 33_554_432); // clamped down to max
        assert_eq!(
            options.set("move overhead", Some("42")),
            Ok("Move Overhead")
        );
        assert_eq!(options.move_overhead_ms(), 42);
    }

    #[test]
    fn setoption_errors_leave_values_unchanged() {
        let mut options = Options::new();
        assert!(options.set("Hash", Some("banana")).is_err());
        assert_eq!(options.hash_mb(), 16);
        assert!(options.set("Hash", None).is_err());
        assert!(options.set("Ponder", Some("yes")).is_err());
        assert!(!options.ponder());
        assert!(options.set("No Such Option", Some("1")).is_err());
        // Buttons need no value and never error on one.
        assert_eq!(options.set("Clear Hash", None), Ok("Clear Hash"));
        assert_eq!(options.set("Clear Hash", Some("now")), Ok("Clear Hash"));
    }

    #[test]
    fn setoption_error_paths_answer_info_string() {
        let (_, out) = command("setoption name Does Not Exist value 1");
        assert!(
            out.contains("info string No such option: 'Does Not Exist'"),
            "{out}"
        );
        let (_, out) = command("setoption value 5");
        assert!(out.contains("info string Malformed setoption"), "{out}");
        let (_, out) = command("setoption name Hash value banana");
        assert!(out.contains("info string Invalid value 'banana'"), "{out}");
    }

    // -- Hash / Clear Hash TT wiring (doc 08 §4 step 8; doc 12 §3.5) --------

    /// Plant a recognizable entry in the engine's TT through the public
    /// probe/write API (search integration is doc 08 step 7; until then
    /// the tests are the TT's only writer).
    fn plant_tt_entry(engine: &Engine, key: u64) {
        use crate::tt::Bound;
        engine.tt.probe(key).writer.write(
            key,
            42,
            false,
            Bound::Exact,
            5,
            Move::NULL,
            7,
            engine.tt.generation(),
        );
        assert!(engine.tt.probe(key).hit, "planted entry must be probeable");
    }

    #[test]
    fn setoption_hash_resizes_and_clears_the_tt() {
        let mut engine = Engine::new();
        assert_eq!(
            engine.tt.size_bytes(),
            16 * 1024 * 1024,
            "default Hash is 16 MB"
        );
        let key = 0xFACE_FEED_0000_0001;
        plant_tt_entry(&engine, key);
        let out = command_on(&mut engine, "setoption name Hash value 4");
        assert_eq!(out, "", "successful resize is silent");
        assert_eq!(engine.tt.size_bytes(), 4 * 1024 * 1024);
        assert!(!engine.tt.probe(key).hit, "resize implies clear");
        assert_eq!(engine.tt.generation(), 0);
    }

    #[test]
    fn setoption_clear_hash_clears_the_tt_without_resizing() {
        let mut engine = Engine::new();
        engine.tt.new_search();
        let key = 0xFACE_FEED_0000_0002;
        plant_tt_entry(&engine, key);
        let out = command_on(&mut engine, "setoption name Clear Hash");
        assert_eq!(out, "");
        assert!(!engine.tt.probe(key).hit, "Clear Hash empties the table");
        assert_eq!(engine.tt.generation(), 0, "clear resets the age");
        assert_eq!(engine.tt.size_bytes(), 16 * 1024 * 1024, "size untouched");
    }

    #[test]
    fn ucinewgame_clears_the_tt() {
        // doc 12 §3.3 command table: `ucinewgame` clears the TT (histories
        // join with doc 09; per-game TM reset asserted separately below).
        let mut engine = Engine::new();
        let key = 0xFACE_FEED_0000_0003;
        plant_tt_entry(&engine, key);
        command_on(&mut engine, "ucinewgame");
        assert!(!engine.tt.probe(key).hit);
    }

    #[test]
    fn mid_search_hash_setoption_implicitly_stops_then_applies() {
        // doc 12 §3.5 step 1: a mid-search `setoption` first stops the
        // search (bestmove flushes exactly once), THEN resizes — the
        // Arc::get_mut path must never race a live session.
        let out = session("go infinite\nsetoption name Hash value 2\nisready\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
        assert!(out.contains("readyok\n"), "{out}");
        assert!(!out.contains("Failed to allocate"), "{out}");
    }

    #[test]
    fn setoption_spaced_string_value_roundtrips() {
        let mut options = Options::new();
        options
            .set("Debug Log File", Some("logs of the engine.txt"))
            .unwrap();
        assert_eq!(options.debug_log_file(), "logs of the engine.txt");
        options.set("Debug Log File", Some("<empty>")).unwrap();
        assert_eq!(options.debug_log_file(), "");
    }

    #[test]
    fn debug_log_file_tees_both_directions() {
        let path = std::env::temp_dir().join(format!(
            "chessengine_uci_log_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path_str = path.to_str().expect("temp path is UTF-8");
        let script = format!(
            "setoption name Debug Log File value {path_str}\n\
             isready\n\
             setoption name Debug Log File value\n\
             uci\nquit\n"
        );
        let out = session(&script);
        assert!(out.contains("readyok\n"), "{out}");
        let log = std::fs::read_to_string(&path).expect("debug log was created");
        let _ = std::fs::remove_file(&path);
        // Both directions, marked and timestamped (doc 12 §3.5).
        assert!(log.contains("<< isready"), "{log}");
        assert!(log.contains(">> readyok"), "{log}");
        // The unsetting line itself is still logged; the log closes after.
        assert!(
            log.contains("<< setoption name Debug Log File value"),
            "{log}"
        );
        assert!(!log.contains("uciok"), "log must close on unset: {log}");
        assert!(log.lines().all(|l| l.starts_with('[')), "{log}");
    }

    // -- go parsing (doc 12 §3.6; §5 "Unit — parser") ----------------------

    #[test]
    fn go_token_soup_parses_in_any_order() {
        let (limits, out) = parse_limits(
            START_FEN,
            "btime 2000 ponder wtime 1000 movestogo 5 binc 20 depth 8 \
             winc 10 nodes 5000 movetime 300 mate 3 infinite",
        );
        assert_eq!(out, "");
        assert_eq!(limits.time[Color::White as usize], 1000);
        assert_eq!(limits.time[Color::Black as usize], 2000);
        assert_eq!(limits.inc[Color::White as usize], 10);
        assert_eq!(limits.inc[Color::Black as usize], 20);
        assert_eq!(limits.movestogo, 5);
        assert_eq!(limits.depth, Some(8));
        assert_eq!(limits.nodes, Some(5000));
        assert_eq!(limits.movetime, Some(300));
        assert_eq!(limits.mate, Some(3));
        assert!(limits.infinite);
        assert!(limits.ponder);
        assert!(limits.use_time_management());
    }

    #[test]
    fn go_bare_and_unknown_tokens() {
        let (limits, out) = parse_limits(START_FEN, "");
        assert_eq!(out, "");
        assert!(!limits.use_time_management());
        // Unknown tokens are skipped silently; known ones still land.
        let (limits, out) = parse_limits(START_FEN, "frobnicate depth 3 blorp");
        assert_eq!(out, "");
        assert_eq!(limits.depth, Some(3));
        // Negative clocks parse (GUI lag); TM clamps later.
        let (limits, _) = parse_limits(START_FEN, "wtime -50 btime 100");
        assert_eq!(limits.time[Color::White as usize], -50);
        // Malformed values are dropped, parsing continues.
        let (limits, out) = parse_limits(START_FEN, "wtime banana btime 100");
        assert_eq!(out, "");
        assert_eq!(limits.time[Color::White as usize], 0);
        assert_eq!(limits.time[Color::Black as usize], 100);
        // `go perft` parses as a dev limit (§3.15; wired in step 12).
        let (limits, _) = parse_limits(START_FEN, "perft 5");
        assert_eq!(limits.perft, Some(5));
    }

    #[test]
    fn go_searchmoves_consumes_the_tail() {
        let (limits, out) = parse_limits(START_FEN, "depth 5 searchmoves e2e4 d2d4 depth 9");
        // `depth 9` after searchmoves is a (failed) move token, NOT a
        // parameter (SF behavior).
        assert_eq!(limits.depth, Some(5));
        assert_eq!(limits.searchmoves.len(), 2);
        assert_eq!(move_to_uci(limits.searchmoves[0]), "e2e4");
        assert_eq!(move_to_uci(limits.searchmoves[1]), "d2d4");
        assert!(out.contains("info string Invalid searchmoves token 'depth' ignored"));
        assert!(out.contains("info string Invalid searchmoves token '9' ignored"));
    }

    // -- ucinewgame ---------------------------------------------------------

    #[test]
    fn ucinewgame_resets_tm_state_not_position() {
        let mut engine = Engine::new();
        command_on(&mut engine, "position startpos moves e2e4");
        engine.state.as_mut().unwrap().persist.original_time_adjust = 1.5;
        command_on(&mut engine, "ucinewgame");
        assert_eq!(
            engine.state.as_ref().unwrap().persist.original_time_adjust,
            -1.0
        );
        // The session position is untouched (GUIs send `position` next).
        assert!(engine.board.to_fen().contains("4P3"));
    }

    // -- the go session (doc 12 §3.2; §4 steps 5-6) --------------------------

    /// Count `bestmove` lines in a transcript.
    fn bestmoves(out: &str) -> usize {
        out.lines().filter(|l| l.starts_with("bestmove ")).count()
    }

    #[test]
    fn go_depth_emits_info_lines_then_exactly_one_bestmove() {
        // Mate-in-1: the completed depth-2 search must find d1d8 with an
        // exact mate score, and the transcript shape is deterministic.
        let mut engine = Engine::new();
        command_on(
            &mut engine,
            "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        );
        let out = command_on(&mut engine, "go depth 2");
        assert_eq!(bestmoves(&out), 1, "{out}");
        let bestmove_line = out.lines().last().unwrap();
        assert!(bestmove_line.starts_with("bestmove d1d8"), "{out}");
        // §3.13 v1 fields, in order, one line per completed iteration.
        let infos: Vec<&str> = out.lines().filter(|l| l.starts_with("info ")).collect();
        assert_eq!(infos.len(), 2, "{out}");
        assert!(infos[0].starts_with("info depth 1 seldepth "), "{out}");
        assert!(infos[1].starts_with("info depth 2 seldepth "), "{out}");
        for info in &infos {
            assert!(info.contains(" score mate 1"), "{info}");
            for field in [" nodes ", " nps ", " hashfull ", " time ", " pv "] {
                assert!(info.contains(field), "missing{field}in: {info}");
            }
            assert!(info.ends_with(" pv d1d8"), "{info}");
        }
    }

    #[test]
    fn go_mate_self_terminates_on_the_mate_limit() {
        // doc 12 §3.9's mate-limit stop end-to-end: `go mate 1` rides
        // `Limits::mate` from parse_go into the ID loop's §3.5 check
        // (doc 07). No depth/time/node limit here, so only the mate rule
        // can end the session — a regression hangs rather than passes.
        let mut engine = Engine::new();
        command_on(
            &mut engine,
            "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        );
        let out = command_on(&mut engine, "go mate 1");
        assert_eq!(bestmoves(&out), 1, "{out}");
        assert!(
            out.lines().last().unwrap().starts_with("bestmove d1d8"),
            "{out}"
        );
        assert!(out.contains(" score mate 1"), "{out}");
    }

    #[test]
    fn ponder_from_tt_offers_only_validated_moves() {
        // doc 12 §3.2 step 4: the 1-ply-PV fallback probes the TT after
        // making the best move and must validate everything it reads
        // (doc 08 §3.5).
        use crate::tt::Bound;
        let tt = TranspositionTable::new(MIN_HASH_MB);
        let board = Board::startpos();
        let best = move_from_uci(&board, "e2e4").unwrap();
        let mut after = board.clone();
        after.make_move(best, after.gives_check(best));
        let key = after.key();

        // Nothing stored: no ponder move.
        assert_eq!(ponder_from_tt(&tt, &board, best), None);

        // A legal stored reply is offered.
        let reply = move_from_uci(&after, "c7c5").unwrap();
        tt.probe(key)
            .writer
            .write(key, 10, false, Bound::Exact, 3, reply, 0, tt.generation());
        assert_eq!(ponder_from_tt(&tt, &board, best), Some(reply));

        // A garbage/collided move (illegal after 1.e4: it moves a WHITE
        // pawn with black to play) is dropped by the validation contract.
        let foreign = Move::normal(Square::A2, Square::A3);
        tt.probe(key)
            .writer
            .write(key, 10, false, Bound::Exact, 4, foreign, 0, tt.generation());
        assert_eq!(ponder_from_tt(&tt, &board, best), None);
    }

    #[test]
    fn ponder_from_tt_refuses_immediate_draws() {
        // The "not an immediate draw" guard (doc 12 §3.2 step 4): the
        // best move lands on the 100th quiet halfmove, so the resulting
        // position is a 50-move draw — no pondering into a dead game.
        use crate::tt::Bound;
        let tt = TranspositionTable::new(MIN_HASH_MB);
        let board = Board::from_fen("k7/8/2K5/8/8/8/8/7R w - - 99 80").unwrap();
        let best = move_from_uci(&board, "h1h2").unwrap();
        let mut after = board.clone();
        after.make_move(best, after.gives_check(best));
        let key = after.key();
        let reply = move_from_uci(&after, "a8b8").unwrap();
        tt.probe(key)
            .writer
            .write(key, 0, false, Bound::Exact, 3, reply, 0, tt.generation());
        assert_eq!(ponder_from_tt(&tt, &board, best), None);
    }

    #[test]
    fn bestmove_carries_a_ponder_move_when_the_pv_is_long_enough() {
        let mut engine = Engine::new();
        let out = command_on(&mut engine, "go depth 3");
        assert_eq!(bestmoves(&out), 1, "{out}");
        let last = out.lines().last().unwrap();
        assert!(last.contains(" ponder "), "depth-3 startpos PV >= 2: {out}");
        // Both moves parse against the evolving position.
        let tokens: Vec<&str> = last.split_whitespace().collect();
        let mut board = Board::startpos();
        let best = move_from_uci(&board, tokens[1]).expect("legal bestmove");
        board.make_move(best, board.gives_check(best));
        assert!(move_from_uci(&board, tokens[3]).is_some(), "{last}");
    }

    #[test]
    fn mated_root_answers_info_depth_0_and_bestmove_none() {
        // Fool's mate: white is checkmated (doc 12 §3.2 step 2, §5 item 7).
        let mut engine = Engine::new();
        command_on(
            &mut engine,
            "position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        );
        let out = command_on(&mut engine, "go depth 3");
        assert_eq!(out, "info depth 0 score mate 0\nbestmove (none)\n");
    }

    #[test]
    fn stalemated_root_answers_info_depth_0_and_bestmove_none() {
        let mut engine = Engine::new();
        command_on(&mut engine, "position fen k7/8/1K6/8/8/8/7Q/8 b - - 0 1");
        let out = command_on(&mut engine, "go depth 3");
        assert_eq!(out, "info depth 0 score cp 0\nbestmove (none)\n");
    }

    #[test]
    fn go_infinite_prints_bestmove_only_after_stop() {
        // Scripted: `stop` releases the busy-wait; exactly one bestmove.
        let out = session("go infinite\nstop\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn quit_implicitly_stops_an_infinite_search_and_flushes_bestmove() {
        let out = session("go infinite\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn eof_implicitly_stops_a_search_and_flushes_bestmove() {
        // No quit at all: the loop's exit path must stop + join (§3.2).
        let out = session("go infinite\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn blocking_commands_implicitly_stop_a_running_search() {
        // §3.2/§5 item 8: a mid-search `position` flushes the pending
        // bestmove, then applies; `d` (also blocking) proves it applied.
        let out = session("go infinite\nposition startpos moves e2e4\nd\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
        assert!(
            out.contains("Fen: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            "{out}"
        );
        // The bestmove flushed before the new position's diagram printed.
        assert!(
            out.find("bestmove").unwrap() < out.find("Fen:").unwrap(),
            "{out}"
        );
    }

    #[test]
    fn two_go_cycles_answer_exactly_one_bestmove_each() {
        // Distinct mate-in-1 positions prove the position change between
        // searches applied (doc 12 §4 steps 5-6 done-conditions).
        let mut engine = Engine::new();
        command_on(
            &mut engine,
            "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        );
        let first = command_on(&mut engine, "go depth 2");
        assert_eq!(bestmoves(&first), 1, "{first}");
        assert!(first.contains("bestmove d1d8"), "{first}");

        command_on(
            &mut engine,
            "position fen 3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
        );
        let second = command_on(&mut engine, "go depth 2");
        assert_eq!(bestmoves(&second), 1, "{second}");
        assert!(second.contains("bestmove d8d1"), "{second}");
    }

    #[test]
    fn ponderhit_releases_a_finished_ponder_search() {
        // `go ponder depth 1` finishes searching, then busy-waits; the
        // ponderhit must release exactly one bestmove without a hang
        // (§3.2 step 3 — the full ponder machine is step 9).
        let out = session("go ponder depth 1\nponderhit\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn timed_ponder_search_stops_only_after_ponderhit() {
        // The §3.8 ponder guard: the expired movetime cannot stop the
        // pondering search, so the session hinges on `ponderhit` flipping
        // the shared flag — then the next check_time poll stops it and
        // exactly one bestmove flows (§3.10 v1 subset).
        let out = session("go ponder movetime 1\nponderhit\nquit\n");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn isready_answers_during_a_search() {
        // `isready` is a liveness ping (§3.2): it must not wait for the
        // infinite search — the readyok arrives while the search runs.
        let out = session("go infinite\nisready\nstop\nquit\n");
        assert!(out.contains("readyok\n"), "{out}");
        assert_eq!(bestmoves(&out), 1, "{out}");
    }

    #[test]
    fn go_perft_answers_not_implemented() {
        let (_, out) = command("go perft 5");
        assert!(
            out.contains("info string go perft is not implemented yet"),
            "{out}"
        );
        assert_eq!(bestmoves(&out), 0, "{out}");
    }

    #[test]
    fn unusable_searchmoves_warns_and_still_searches() {
        // §3.6/§5 item 9: only-invalid searchmoves → warn, search all
        // moves, never a fake `bestmove (none)`. (`depth` before
        // `searchmoves` — the latter consumes the whole tail, §3.6.)
        let mut engine = Engine::new();
        let out = command_on(&mut engine, "go depth 1 searchmoves e2e5");
        assert!(
            out.contains("info string Invalid searchmoves token 'e2e5' ignored"),
            "{out}"
        );
        assert!(
            out.contains("info string Invalid searchmoves list ignored"),
            "{out}"
        );
        assert_eq!(bestmoves(&out), 1, "{out}");
        assert!(!out.contains("bestmove (none)"), "{out}");
    }

    // -- score/info formatting (doc 12 §3.14; §5 "Unit — text") -------------

    #[test]
    fn format_score_covers_the_mate_band_both_signs() {
        use crate::types::{mate_in, mated_in};
        // Winning side rounds up (our mating move counts).
        assert_eq!(format_score(mate_in(1)), UciScore::Mate(1));
        assert_eq!(format_score(mate_in(2)), UciScore::Mate(1));
        assert_eq!(format_score(mate_in(3)), UciScore::Mate(2));
        assert_eq!(format_score(mate_in(4)), UciScore::Mate(2));
        assert_eq!(format_score(mate_in(5)), UciScore::Mate(3));
        // Losing side truncates, matching SF exactly.
        assert_eq!(format_score(mated_in(0)), UciScore::Mate(0));
        assert_eq!(format_score(mated_in(1)), UciScore::Mate(0));
        assert_eq!(format_score(mated_in(2)), UciScore::Mate(-1));
        assert_eq!(format_score(mated_in(3)), UciScore::Mate(-1));
        assert_eq!(format_score(mated_in(4)), UciScore::Mate(-2));
        // Band edges stay mates; everything below is centipawns.
        use crate::types::{MAX_PLY, VALUE_MATE_IN_MAX_PLY};
        assert_eq!(
            format_score(VALUE_MATE_IN_MAX_PLY),
            UciScore::Mate((MAX_PLY as i32 + 1) / 2)
        );
        assert_eq!(format_score(0), UciScore::Cp(0));
        assert_eq!(format_score(123), UciScore::Cp(123));
        assert_eq!(format_score(-4000), UciScore::Cp(-4000));
    }

    #[test]
    fn uci_score_displays_the_wdl_less_form() {
        assert_eq!(UciScore::Cp(-31).to_string(), "cp -31");
        assert_eq!(UciScore::Mate(-3).to_string(), "mate -3");
    }

    #[test]
    fn format_info_orders_the_v1_fields() {
        let pv = [
            Move::normal(Square::E2, Square::E4),
            Move::normal(Square::E7, Square::E5),
        ];
        let info = SearchInfo {
            depth: 7,
            sel_depth: 12,
            multipv: 1,
            score: 42,
            lowerbound: false,
            upperbound: false,
            nodes: 4000,
            hashfull: 13,
            pv: &pv,
        };
        assert_eq!(
            format_info(&info, 2000),
            "info depth 7 seldepth 12 score cp 42 nodes 4000 nps 2000 hashfull 13 time 2000 \
             pv e2e4 e7e5"
        );
        // Zero elapsed must not divide by zero; bounds add their suffix.
        let bound = SearchInfo {
            lowerbound: true,
            ..info
        };
        assert_eq!(
            format_info(&bound, 0),
            "info depth 7 seldepth 12 score cp 42 lowerbound nodes 4000 nps 4000000 hashfull 13 \
             time 0 pv e2e4 e7e5"
        );
    }
}
