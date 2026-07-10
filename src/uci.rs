//! UCI entry point: argv subcommand dispatch, then the blocking stdin loop.
//!
//! Contracts: `design/01-architecture.md` §3.1/§3.3.18 (the `run()` seam;
//! argv subcommands run to completion before the loop) and
//! `design/12-uci-time-management.md` §3.3–§3.6 (command grammar, move
//! text, options table, `go` limits) plus its robustness rules: never
//! panic on input, unknown commands answer `info string`, EOF on stdin is
//! `quit` — GUIs close the pipe, and a dead engine loses on time.
//!
//! P4 state (doc 12 §4 steps 1–4): `position` (startpos/fen + atomic move
//! application), the move-text layer (`move_to_uci`/`move_from_uci`), the
//! §3.5 options table with full `setoption` semantics (including the
//! `Debug Log File` stdin/stdout tee), `ucinewgame` per-game TM reset, and
//! `go`-limit parsing into [`Limits`]. Running a search — and with it the
//! real `go`/`stop`/`bestmove` session — lands in P5 (doc 12 step 5); the
//! in-session `bench`/`perft`/`go perft`/`eval` dev commands land with doc
//! 12 step 12. The argv subcommands are live: `perft <depth> [fen]` prints
//! a real SF-format divide (P3, design/06 §3.2) and `bench ... perft` runs
//! perft-mode bench.

use crate::bench;
use crate::board::Board;
use crate::movegen;
use crate::moves::{MAX_MOVES, Move, MoveList};
use crate::perft;
use crate::timeman::{Limits, TimeManager, TimePersist};
use crate::types::Color;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::str::{FromStr, SplitWhitespace};
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
            let mut stdout = io::stdout().lock();
            uci_loop(stdin.lock(), &mut stdout);
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

/// `Hash` spin range in MiB. Owned by the TT design doc (§3.13:
/// `MIN_HASH_MB`/`MAX_HASH_MB`, SF's 64-bit cap); mirrored here until the
/// `tt` module lands (P6) and these become re-exports.
const HASH_MB_MIN: i64 = 1;
const HASH_MB_MAX: i64 = 33_554_432;

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
/// the rest of the engine.
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

// ---------------------------------------------------------------------------
// Engine — the session glue aggregate (design/12 §3.1, P4 subset)
// ---------------------------------------------------------------------------

/// The glue aggregate owned by the UCI loop (doc 12 §3.1). P4 subset: the
/// session position, the options table, and the per-game TM state. The
/// `ThreadPool` and `Arc<TranspositionTable>` join in P5/P6; when they do,
/// `TimeManager`/`TimePersist` migrate into the main worker's
/// `SearchManager` (doc 12 §3.8) and `new_game` grows the TT/history
/// clears.
pub struct Engine {
    board: Board,
    options: Options,
    time_manager: TimeManager,
    persist: TimePersist,
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            board: Board::startpos(),
            options: Options::new(),
            time_manager: TimeManager::new(),
            persist: TimePersist::default(),
        }
    }

    /// `ucinewgame` (doc 12 §3.3/§3.7): reset the per-game TM state (the
    /// "first move of the game" defaults measurably affect move-1 time
    /// usage) and the `nodestime` budget. Deliberately does NOT touch the
    /// session position — GUIs send `position` separately. TT + history
    /// clears join in P5/P6.
    pub fn new_game(&mut self) {
        self.persist = TimePersist::default();
        self.time_manager.new_game();
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
/// `impl Write`) so unit tests drive it with in-memory buffers while
/// `run()` passes locked stdin/stdout.
///
/// Loop-exit rules (doc 12 §3.3): `quit`, stdin EOF, or a read error all
/// leave the loop; a write error means the GUI is gone, same thing. No
/// input line may panic.
fn uci_loop(input: impl BufRead, out: &mut impl Write) {
    let mut engine = Engine::new();
    let mut out = UciOut::new(out);
    for line in input.lines() {
        let Ok(line) = line else { break };
        out.log_input(&line);
        match dispatch(&mut engine, &line, &mut out) {
            Ok(true) => {}
            Ok(false) | Err(_) => break,
        }
        // Piped stdout is block-buffered in Rust: flush per command or a
        // GUI waiting on `uciok` hangs forever.
        if out.flush().is_err() {
            break;
        }
    }
    let _ = out.flush();
}

/// Handle one input line (the doc 12 §3.3 dispatch shape).
/// Returns `Ok(false)` exactly when the loop should end (`quit`).
fn dispatch<W: Write>(engine: &mut Engine, line: &str, out: &mut UciOut<W>) -> io::Result<bool> {
    let mut tokens = line.split_whitespace();
    match tokens.next() {
        Some("uci") => {
            writeln!(out, "id name {NAME}")?;
            writeln!(out, "id author {AUTHOR}")?;
            for entry in engine.options.iter() {
                writeln!(out, "{entry}")?;
            }
            writeln!(out, "uciok")?;
        }
        // Trivially ready until searches run on worker threads (P5); it
        // stays an immediate answer even then (liveness ping, §3.2).
        Some("isready") => writeln!(out, "readyok")?,
        Some("setoption") => cmd_setoption(engine, tokens, out)?,
        Some("ucinewgame") => engine.new_game(),
        Some("position") => cmd_position(engine, tokens, out)?,
        Some("go") => cmd_go(engine, tokens, out)?,
        // Harmless no-ops while no search runs (§3.3); they become
        // PoolShared flag stores in P5.
        Some("stop" | "ponderhit") => {}
        Some("d") => print_board(out, &engine.board)?,
        Some("quit") => return Ok(false),
        // Accepted-but-ignored per the UCI spec (doc 12 §3.3): routing
        // these through the unknown branch would spam GUIs that send them.
        Some("debug" | "register") => {}
        // Everything else is absorbed, never fatal (doc 12 §3.3).
        Some(_) => writeln!(out, "info string Unknown command: '{}'", line.trim())?,
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
        // `Clear Hash` clears the TT + histories once those exist (P6);
        // `Hash` resizes the TT (P6); `Threads` rebuilds the pool (P8).
        // Their values are stored above; nothing further yet.
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

/// `go [<param>...]` (doc 12 §3.6). P4 parses the limits — the search
/// session that consumes them is doc 12 step 5 (P5), so until then the
/// engine says so instead of leaving the GUI waiting for a `bestmove`
/// that cannot come.
fn cmd_go<W: Write>(
    engine: &mut Engine,
    tokens: SplitWhitespace,
    out: &mut UciOut<W>,
) -> io::Result<()> {
    let limits = parse_go(&engine.board, tokens, out)?;
    let _ = limits; // consumed by the search session from P5 on
    writeln!(
        out,
        "info string go is not implemented yet (search lands in P5)"
    )
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
                for tok in tokens.by_ref() {
                    match move_from_uci(board, tok) {
                        Some(m) => limits.searchmoves.push(m),
                        None => {
                            writeln!(out, "info string Invalid searchmoves token '{tok}' ignored")?
                        }
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
    fn session(input: &str) -> String {
        let mut out = Vec::new();
        uci_loop(input.as_bytes(), &mut out);
        String::from_utf8(out).expect("engine output is UTF-8")
    }

    /// Run one command against a fresh engine, returning (engine, output).
    fn command(line: &str) -> (Engine, String) {
        let mut engine = Engine::new();
        let out = command_on(&mut engine, line);
        (engine, out)
    }

    /// Run one command against an existing engine, returning its output.
    fn command_on(engine: &mut Engine, line: &str) -> String {
        let mut out = UciOut::new(Vec::new());
        dispatch(engine, line, &mut out).expect("Vec sink never fails");
        String::from_utf8(out.out).expect("engine output is UTF-8")
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

    #[test]
    fn go_command_answers_not_implemented() {
        let (_, out) = command("go depth 4");
        assert!(
            out.contains("info string go is not implemented yet"),
            "{out}"
        );
    }

    // -- ucinewgame ---------------------------------------------------------

    #[test]
    fn ucinewgame_resets_tm_state_not_position() {
        let mut engine = Engine::new();
        command_on(&mut engine, "position startpos moves e2e4");
        engine.persist.original_time_adjust = 1.5;
        command_on(&mut engine, "ucinewgame");
        assert_eq!(engine.persist.original_time_adjust, -1.0);
        // The session position is untouched (GUIs send `position` next).
        assert!(engine.board.to_fen().contains("4P3"));
    }
}
