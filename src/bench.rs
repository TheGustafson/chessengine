//! The `bench` subcommand — the search regression oracle (D24).
//!
//! Specification: `design/06-perft-validation.md` §3.6 owns the argument
//! grammar, the compiled-in position suite, the types below, and the
//! output format; `design/14-performance-testing.md` §3.2 owns the
//! measurement policy that makes the numbers evidence (the `Bench:`
//! commit-trailer fingerprint).
//!
//! Grammar (SF argument order, so muscle memory and scripts transfer):
//!
//! ```text
//! chessengine bench [ttMB=16] [threads=1] [limit=13] [fens=default] [type=depth]
//! ```
//!
//! Search modes (`type=depth/nodes/movetime`, P5 / design/06 §4 step 7)
//! run one `ucinewgame`-fresh [`Worker`] session across the whole suite —
//! a deterministic single-threaded search per position. Since P6 the
//! session owns a `ttMB`-sized [`TranspositionTable`] that deliberately
//! PERSISTS across positions with one generation bump per position — the
//! TT carry-over is part of the fingerprint (design/06 §3.6, design/14
//! §3.2 rule 3); `threads` becomes meaningful with SMP. Depth and nodes
//! limits never read the wall clock (design/14 §3.2 determinism rule 4),
//! so their totals are exact fingerprints; `movetime` is an nps probe
//! only, never a fingerprint.
//!
//! Output contract (design/06 §3.6, all on stderr so stdout stays a clean
//! UCI stream): a `Position: i/N (entry)` line per position followed by
//! that position's `Nodes searched: n` count (perft mode: the same line
//! SF's per-position perft emits, format per design/06 §3.2; search
//! modes: the position's search node count), then the fixed four-line
//! trailer. The footer's `Nodes searched` total is THE fingerprint: the
//! `Bench: <total-nodes>` commit trailer (design/14 §3.2) that
//! `tools/check.sh` extracts and reruns. Everything except the
//! `Total time (ms)` and `Nodes/second` lines is a pure function of the
//! suite — byte-identical across runs (design/14 §3.2 determinism; doc 06
//! §8 criterion 7; doc 07 §8.5).
//!
//! Cross-checked fingerprints for the default suite in perft mode, from
//! python-chess 1.11.2 (an independent legal-move implementation; recorded
//! 2026-07-10 per design/06 §4 step 6): depth 1 = 1,380; depth 2 = 44,275;
//! depth 3 = 1,839,507.

use crate::board::Board;
use crate::movegen;
use crate::moves::MoveList;
use crate::perft;
use crate::search::Worker;
use crate::timeman::{Limits, TimePersist};
use crate::tt::TranspositionTable;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// Where the bench positions come from (design/06 §3.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenSource {
    /// The compiled-in [`BENCH_POSITIONS`] suite.
    Default,
    /// The UCI session's current position (in-loop `bench` only; P4).
    Current,
    /// One FEN per line from a file.
    File(PathBuf),
}

/// What limits each per-position run (design/06 §3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchLimit {
    Depth(i32),
    Perft(u32),
    Nodes(u64),
    Movetime(u64),
}

/// Parsed bench invocation (design/06 §3.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchParams {
    pub tt_mb: usize,
    pub threads: usize,
    pub limit: BenchLimit,
    pub fens: FenSource,
    /// Optional NNUE net (design/11 §3.4.5, P13 step 4): the 6th positional
    /// arg. `None` ⇒ PSQT (the default; the depth-6 fingerprint invocation
    /// `bench 16 1 6 default depth` has no 6th arg, so it stays bit-identical).
    /// `Some(path)` loads it as the active net before the run for the NNUE-on
    /// node count.
    pub net: Option<String>,
}

/// SF defaults: `16 1 13 default depth` (design/06 §3.6), NNUE off.
impl Default for BenchParams {
    fn default() -> Self {
        BenchParams {
            tt_mb: 16,
            threads: 1,
            limit: BenchLimit::Depth(13),
            fens: FenSource::Default,
            net: None,
        }
    }
}

/// Total nodes (THE fingerprint number) and elapsed wall time.
pub struct BenchResult {
    pub nodes: u64,
    pub elapsed_ms: u64,
}

/// Malformed `bench` arguments. Fallible parsing returns `Result`, never
/// panics (crate convention; design/14 §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchArgsError {
    /// A positional numeric field did not parse.
    BadNumber { field: &'static str, token: String },
    /// The `type` field was not depth/perft/nodes/movetime.
    UnknownLimitType(String),
    /// The limit number does not fit the range its `type` requires.
    LimitOutOfRange {
        limit_type: &'static str,
        value: u64,
    },
}

impl fmt::Display for BenchArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchArgsError::BadNumber { field, token } => {
                write!(f, "invalid {field}: '{token}'")
            }
            BenchArgsError::UnknownLimitType(token) => write!(
                f,
                "unknown limit type '{token}' (expected depth, perft, nodes, or movetime)"
            ),
            BenchArgsError::LimitOutOfRange { limit_type, value } => {
                write!(f, "limit {value} out of range for type {limit_type}")
            }
        }
    }
}

impl Error for BenchArgsError {}

impl BenchParams {
    /// Parse the positional argument tail of
    /// `bench [ttMB] [threads] [limit] [fens] [type]` (design/06 §3.6).
    ///
    /// Missing fields take the SF defaults `16 1 13 default depth`; the
    /// limit number is interpreted according to the (later) `type` field.
    /// Extra arguments beyond the five are ignored, mirroring SF's
    /// tolerance of trailing tokens.
    pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<BenchParams, BenchArgsError> {
        let arg = |i: usize| args.get(i).map(S::as_ref);

        let tt_mb = match arg(0) {
            Some(tok) => tok.parse().map_err(|_| BenchArgsError::BadNumber {
                field: "ttMB",
                token: tok.to_string(),
            })?,
            None => 16,
        };
        let threads = match arg(1) {
            Some(tok) => tok.parse().map_err(|_| BenchArgsError::BadNumber {
                field: "threads",
                token: tok.to_string(),
            })?,
            None => 1,
        };
        let limit_value: u64 = match arg(2) {
            Some(tok) => tok.parse().map_err(|_| BenchArgsError::BadNumber {
                field: "limit",
                token: tok.to_string(),
            })?,
            None => 13,
        };
        let fens = match arg(3) {
            None | Some("default") => FenSource::Default,
            Some("current") => FenSource::Current,
            Some(path) => FenSource::File(PathBuf::from(path)),
        };
        let limit = match arg(4).unwrap_or("depth") {
            "depth" => BenchLimit::Depth(i32::try_from(limit_value).map_err(|_| {
                BenchArgsError::LimitOutOfRange {
                    limit_type: "depth",
                    value: limit_value,
                }
            })?),
            "perft" => BenchLimit::Perft(u32::try_from(limit_value).map_err(|_| {
                BenchArgsError::LimitOutOfRange {
                    limit_type: "perft",
                    value: limit_value,
                }
            })?),
            "nodes" => BenchLimit::Nodes(limit_value),
            "movetime" => BenchLimit::Movetime(limit_value),
            other => return Err(BenchArgsError::UnknownLimitType(other.to_string())),
        };
        // Optional 6th positional arg: an NNUE net path (design/11 §3.4.5).
        // Absent ⇒ PSQT, so the pinned `bench 16 1 6 default depth` fingerprint
        // is unaffected.
        let net = arg(5).map(str::to_string);

        Ok(BenchParams {
            tt_mb,
            threads,
            limit,
            fens,
            net,
        })
    }
}

/// Run the bench: one `ucinewgame`, then for each entry set the position
/// (entries may carry `moves ...` suffixes) and search/perft it,
/// accumulating node counts. Diagnostics go to stderr, per SF, so stdout
/// stays a clean UCI stream (design/06 §3.6).
pub fn run(params: &BenchParams) -> BenchResult {
    // A failed stderr write means the world is gone; panicking here matches
    // what `eprintln!` would do.
    run_to(params, &mut io::stderr().lock()).expect("bench: writing diagnostics to stderr failed")
}

/// Materialize the position list for a [`FenSource`] (design/06 §3.6):
/// the compiled-in suite, or one entry per non-blank line of a file
/// (surrounding whitespace trimmed; entries may carry `moves` suffixes).
fn suite_entries(fens: &FenSource) -> Vec<String> {
    match fens {
        FenSource::Default => BENCH_POSITIONS.iter().map(|s| s.to_string()).collect(),
        FenSource::File(path) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read fen file {}: {e}", path.display()))
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        // Reachable only from the future in-session `bench` dev command
        // (design/12 UCI dev surface); the argv front door has no session
        // position. Parsing stays forward-compatible (see the parser test).
        FenSource::Current => {
            todo!("in-session bench on the session position (design/12 dev surface)")
        }
    }
}

/// The writer-generic engine behind [`run`], mirroring `uci_loop`'s
/// stream-injection pattern so tests can capture and byte-compare the
/// exact output the binary would print.
fn run_to(params: &BenchParams, out: &mut impl Write) -> io::Result<BenchResult> {
    let entries = suite_entries(&params.fens);

    // `ucinewgame` semantics, once for the whole run (design/06 §3.6): a
    // fresh Worker and a fresh `params.tt_mb`-MiB table are the search
    // session the loop below reuses across positions — the TT deliberately
    // PERSISTS across positions (carry-over is part of the fingerprint,
    // design/14 §3.2 rule 3), aged by one `new_search` bump per position
    // in [`search_entry`]; histories (doc 09) will reset here too. Perft
    // mode never touches either.
    let mut worker = Worker::new(Board::startpos());
    // Install the active NNUE net (set by the front door from `params.net`, or
    // by a prior `setoption EvalFile`) into this worker; `None` ⇒ PSQT, so the
    // default fingerprint invocation is untouched (design/11 §3.4.5).
    worker.refresh_eval_net();
    let tt = TranspositionTable::new(params.tt_mb);
    let start = Instant::now();
    let mut nodes: u64 = 0;

    for (i, entry) in entries.iter().enumerate() {
        writeln!(out, "Position: {}/{} ({entry})", i + 1, entries.len())?;
        let count = match params.limit {
            BenchLimit::Perft(depth) => {
                let mut board = board_from_entry(entry);
                perft::perft(&mut board, depth)
            }
            BenchLimit::Depth(d) => {
                let mut limits = Limits::new();
                limits.depth = Some(d);
                search_entry(&mut worker, &tt, entry, limits)
            }
            BenchLimit::Nodes(n) => {
                let mut limits = Limits::new();
                limits.nodes = Some(n);
                search_entry(&mut worker, &tt, entry, limits)
            }
            BenchLimit::Movetime(t) => {
                let mut limits = Limits::new();
                limits.movetime = Some(i64::try_from(t).unwrap_or(i64::MAX));
                search_entry(&mut worker, &tt, entry, limits)
            }
        };
        // Per-position count, in the SF `go perft` trailer format
        // (design/06 §3.2) — part of the fingerprint section.
        writeln!(out, "Nodes searched: {count}")?;
        nodes += count;
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    // The SF-format trailer (design/06 §3.6). `Nodes searched` is THE
    // number: the `Bench:` commit trailer (design/14 §3.2).
    writeln!(out, "===========================")?;
    writeln!(out, "Total time (ms) : {elapsed_ms}")?;
    writeln!(out, "Nodes searched  : {nodes}")?;
    writeln!(
        out,
        "Nodes/second    : {}",
        nodes * 1000 / elapsed_ms.max(1)
    )?;

    // Move-ordering QUALITY census (instrument builds only). It writes its
    // own FILE, never stdout or stderr — the fingerprint this function prints
    // to stderr is byte-compared by `spawned_bench_*_runs_are_byte_identical`.
    #[cfg(feature = "census70")]
    crate::census70::report(nodes);

    Ok(BenchResult { nodes, elapsed_ms })
}

/// One search-mode bench position (design/06 §3.6): load the entry into
/// the session worker, bump the shared table's generation (the per-`go`
/// `new_search` the session layer owns, doc 08 §3.9 / doc 13 §3.7 B) and
/// run a full `go` under `limits` to its natural end, returning the
/// position's node count (the per-`go` counter the iterative-deepening
/// driver resets and `Worker::do_move` bumps — the design/14 §3.2 rule-5
/// semantics, so bench just sums per-position counters). Info reports are
/// discarded: their time/nps fields are wall-clock noise, and everything
/// the fingerprint needs is in the count. The stop flag stays quiet —
/// nobody else holds it, and depth/nodes limits terminate by themselves —
/// so depth/nodes runs are deterministic by construction (no clock read
/// anywhere on their path). Mate/stalemate entries return immediately
/// with 0 nodes, exercising the zero-legal-move seam the suite exists to
/// cover.
fn search_entry(worker: &mut Worker, tt: &TranspositionTable, entry: &str, limits: Limits) -> u64 {
    worker.board = board_from_entry(entry);
    worker.limits = limits;
    tt.new_search();
    let stop = AtomicBool::new(false);
    // Fixed depth/nodes bench: TM is inert, so a throwaway per-entry
    // persist keeps the fingerprint independent of persist state (D24).
    let mut persist = TimePersist::default();
    worker.iterative_deepening(tt, &stop, &mut persist, &mut |_| {});
    worker.nodes
}

/// Set up a bench entry: a FEN, optionally followed by `moves m1 m2 ...`
/// to be replayed through the same parser the UCI `position` command uses
/// (design/06 §3.6).
fn board_from_entry(entry: &str) -> Board {
    let fen = entry.split(" moves").next().unwrap_or(entry);
    let mut board =
        Board::from_fen(fen).unwrap_or_else(|e| panic!("invalid bench FEN '{entry}': {e}"));
    // Replay each UCI move token against the legal move list — the same
    // match-by-notation rule the UCI `position ... moves` parser uses
    // (design/06 §3.6), so promotion suffixes and castling (king's
    // two-square move, D2) resolve for free via `Move: Display`.
    for token in entry[fen.len()..]
        .strip_prefix(" moves")
        .unwrap_or("")
        .split_whitespace()
    {
        let mut list = MoveList::new();
        movegen::generate_legal(&board, &mut list);
        let m = list
            .iter()
            .copied()
            .find(|m| m.to_string() == token)
            .unwrap_or_else(|| panic!("illegal move '{token}' in bench entry '{entry}'"));
        let gives_check = board.gives_check(m);
        board.make_move(m, gives_check);
    }
    board
}

/// The default bench suite: Stockfish's current 49-entry standard-chess
/// bench list, adopted verbatim (positions are facts, D27; the two
/// Chess960 entries are dropped since 960 is deferred, D10). Provenance:
/// `stockfish_reference/Stockfish/src/benchmark.cpp`, reproduced in
/// design/06 §3.6. The list is FROZEN: changing it invalidates every
/// historical bench fingerprint (design/14 §3.2).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_no_args_gives_sf_defaults() {
        // design/06 §3.6: bench [ttMB=16] [threads=1] [limit=13]
        // [fens=default] [type=depth].
        let params = BenchParams::parse::<&str>(&[]).unwrap();
        assert_eq!(params, BenchParams::default());
        assert_eq!(params.tt_mb, 16);
        assert_eq!(params.threads, 1);
        assert_eq!(params.limit, BenchLimit::Depth(13));
        assert_eq!(params.fens, FenSource::Default);
    }

    #[test]
    fn parse_positional_overrides() {
        // The perft-mode movegen fingerprint invocation from design/06 §3.6.
        let params = BenchParams::parse(&["16", "1", "5", "default", "perft"]).unwrap();
        assert_eq!(params.limit, BenchLimit::Perft(5));

        let params = BenchParams::parse(&["64", "2", "10"]).unwrap();
        assert_eq!(params.tt_mb, 64);
        assert_eq!(params.threads, 2);
        assert_eq!(params.limit, BenchLimit::Depth(10));

        let params = BenchParams::parse(&["16", "1", "50000", "current", "nodes"]).unwrap();
        assert_eq!(params.limit, BenchLimit::Nodes(50_000));
        assert_eq!(params.fens, FenSource::Current);

        let params = BenchParams::parse(&["16", "1", "1000", "suite.fens", "movetime"]).unwrap();
        assert_eq!(params.limit, BenchLimit::Movetime(1000));
        assert_eq!(params.fens, FenSource::File(PathBuf::from("suite.fens")));
    }

    #[test]
    fn parse_malformed_input_is_err_never_panics() {
        assert!(matches!(
            BenchParams::parse(&["banana"]),
            Err(BenchArgsError::BadNumber { field: "ttMB", .. })
        ));
        assert!(matches!(
            BenchParams::parse(&["16", "-1"]),
            Err(BenchArgsError::BadNumber {
                field: "threads",
                ..
            })
        ));
        assert!(matches!(
            BenchParams::parse(&["16", "1", "deep"]),
            Err(BenchArgsError::BadNumber { field: "limit", .. })
        ));
        assert!(matches!(
            BenchParams::parse(&["16", "1", "13", "default", "bogus"]),
            Err(BenchArgsError::UnknownLimitType(_))
        ));
        // Limit number too large for a depth/perft limit.
        assert!(matches!(
            BenchParams::parse(&["16", "1", "99999999999", "default", "perft"]),
            Err(BenchArgsError::LimitOutOfRange { .. })
        ));
    }

    #[test]
    fn suite_has_49_entries_and_three_moves_suffixes() {
        // design/06 §3.6 / design/14 §5: 49 standard-chess entries, three
        // of which carry `moves ...` suffixes.
        assert_eq!(BENCH_POSITIONS.len(), 49);
        let with_moves = BENCH_POSITIONS
            .iter()
            .filter(|e| e.contains(" moves "))
            .count();
        assert_eq!(with_moves, 3);
    }

    #[test]
    fn every_suite_fen_parses() {
        // design/14 §5 suite integrity (the `moves` replay half of the
        // check needs movegen and lands with P3/P4). from_fen accepts the
        // two clockless 4-field entries.
        for entry in BENCH_POSITIONS {
            let fen = entry.split(" moves").next().unwrap();
            assert!(
                Board::from_fen(fen).is_ok(),
                "bench suite FEN failed to parse: {entry}"
            );
        }
    }

    #[test]
    fn bench_perft_mode_smoke() {
        // design/06 §4 step 6: perft-mode bench runs the whole suite,
        // including the three `moves ...` replay entries. Depth 1 keeps
        // this cheap; 1,380 is the python-chess 1.11.2 cross-checked sum
        // of legal-move counts over all 49 entries (recorded 2026-07-10).
        let params = BenchParams::parse(&["16", "1", "1", "default", "perft"]).unwrap();
        let result = run_to(&params, &mut io::sink()).unwrap();
        assert_eq!(result.nodes, 1_380);
    }

    #[test]
    fn suite_entries_default_is_the_frozen_49() {
        assert_eq!(suite_entries(&FenSource::Default), BENCH_POSITIONS);
    }

    #[test]
    fn suite_entries_file_takes_one_entry_per_nonblank_trimmed_line() {
        // design/06 §3.6: `fens=<path>` reads one entry per line; entries
        // may carry `moves ...` suffixes just like the compiled-in suite.
        let path = std::env::temp_dir().join(format!(
            "chessengine_bench_fens_{}_{:?}.txt",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(
            &path,
            "\n  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4 c7c5\n\
             \t \nK1k5/8/P7/8/8/8/8/8 w - - 0 1  \n\n",
        )
        .unwrap();

        let source = FenSource::File(path.clone());
        let entries = suite_entries(&source);
        assert_eq!(entries.len(), 2, "blank lines skipped, entries trimmed");
        assert!(entries[0].starts_with("rnbqkbnr/") && entries[0].ends_with("moves e2e4 c7c5"));

        // Run it: python-chess 1.11.2 says 30 legal moves after 1.e4 c5
        // and 2 in the self-stalemate corner (recorded 2026-07-10).
        let params = BenchParams {
            fens: source,
            limit: BenchLimit::Perft(1),
            ..BenchParams::default()
        };
        let mut out = Vec::new();
        let result = run_to(&params, &mut out).unwrap();
        assert_eq!(result.nodes, 32);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Position: 1/2 (rnbqkbnr/"));
        assert!(text.contains("Position: 2/2 (K1k5/"));

        std::fs::remove_file(&path).ok();
    }

    /// The fingerprint section: every output line except the two that
    /// legitimately vary run-to-run (wall time and the nps derived from
    /// it) — design/14 §3.2 determinism policy.
    fn fingerprint_section(output: &str) -> Vec<&str> {
        output
            .lines()
            .filter(|l| !l.starts_with("Total time (ms) :") && !l.starts_with("Nodes/second    :"))
            .collect()
    }

    #[test]
    fn perft_output_matches_the_doc_06_format() {
        // design/06 §3.6 output contract, checked byte-for-byte at the
        // line level: `Position: i/N (entry)` + per-position count per
        // entry, then the fixed four-line trailer.
        let params = BenchParams::parse(&["16", "1", "1", "default", "perft"]).unwrap();
        let mut out = Vec::new();
        run_to(&params, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2 * 49 + 4);
        assert_eq!(
            lines[0],
            "Position: 1/49 (rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1)"
        );
        assert_eq!(lines[1], "Nodes searched: 20");
        assert!(lines[96].starts_with("Position: 49/49 (7k/"));
        assert_eq!(lines[98], "===========================");
        assert!(lines[99].starts_with("Total time (ms) : "));
        assert_eq!(lines[100], "Nodes searched  : 1380");
        assert!(lines[101].starts_with("Nodes/second    : "));
    }

    #[test]
    fn perft_fingerprint_is_byte_identical_across_runs_in_process() {
        // doc 06 §8 criterion 7 / design/14 §3.2: the fingerprint section
        // is a pure function of the suite — two runs, identical bytes.
        let params = BenchParams::parse(&["16", "1", "1", "default", "perft"]).unwrap();
        let mut first = Vec::new();
        let mut second = Vec::new();
        let a = run_to(&params, &mut first).unwrap();
        let b = run_to(&params, &mut second).unwrap();
        assert_eq!(a.nodes, b.nodes);
        let first = String::from_utf8(first).unwrap();
        let second = String::from_utf8(second).unwrap();
        assert_eq!(fingerprint_section(&first), fingerprint_section(&second));
    }

    /// Locate the engine binary Cargo builds alongside this test
    /// (`target/<profile>/chessengine`); unit tests run from
    /// `target/<profile>/deps/`, so it lives one directory up.
    fn engine_binary() -> std::path::PathBuf {
        let mut path = std::env::current_exe().expect("test binary path");
        path.pop(); // the test executable's name
        if path.ends_with("deps") {
            path.pop();
        }
        path.push(format!("chessengine{}", std::env::consts::EXE_SUFFIX));
        assert!(
            path.exists(),
            "engine binary not found at {} — run the full `cargo test` (not --lib) \
             so Cargo builds the bin target",
            path.display()
        );
        path
    }

    #[test]
    fn bench_depth_mode_searches_each_entry_and_counts_nodes() {
        // design/06 §4 step 7: search-mode bench. A two-entry file suite
        // keeps the trees tiny — debug builds run the full board
        // invariant oracle at every node — while still covering a `moves`
        // replay entry and the zero-legal-move seam the compiled-in suite
        // exists to exercise (design/06 §3.6).
        let path = std::env::temp_dir().join(format!(
            "chessengine_bench_depth_{}_{:?}.txt",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(
            &path,
            "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1 moves b1c2\n\
             8/8/8/8/8/6k1/6p1/6K1 w - -\n",
        )
        .unwrap();

        let params = BenchParams {
            fens: FenSource::File(path.clone()),
            limit: BenchLimit::Depth(2),
            ..BenchParams::default()
        };
        let mut out = Vec::new();
        let result = run_to(&params, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();

        // Same output shape as perft mode: Position + per-position count
        // per entry, then the fixed four-line trailer.
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2 * 2 + 4);
        assert!(lines[0].starts_with("Position: 1/2 (8/8/8/8/5kp1/"));
        assert!(lines[1].starts_with("Nodes searched: "));
        // Entry 2 is stalemate (white to move, no legal moves): the
        // search returns immediately and reports zero nodes.
        assert_eq!(lines[3], "Nodes searched: 0");
        assert!(result.nodes > 0, "entry 1 must actually search");
        assert_eq!(lines[6], format!("Nodes searched  : {}", result.nodes));
        // The footer total is the sum of the per-position counts
        // (design/14 §3.2 rule 5: bench just sums per-position counters).
        let per_position: u64 = [lines[1], lines[3]]
            .iter()
            .map(|l| l.strip_prefix("Nodes searched: ").unwrap())
            .map(|n| n.parse::<u64>().unwrap())
            .sum();
        assert_eq!(per_position, result.nodes);

        // Determinism in-process (doc 07 §8.5): identical totals and a
        // byte-identical fingerprint section across two runs.
        let mut second = Vec::new();
        let again = run_to(&params, &mut second).unwrap();
        assert_eq!(result.nodes, again.nodes);
        let second = String::from_utf8(second).unwrap();
        assert_eq!(fingerprint_section(&text), fingerprint_section(&second));

        std::fs::remove_file(&path).ok();
    }

    /// design/14 §4 step 7 — THE bench determinism test: the default
    /// suite searched twice in-process must produce identical node totals
    /// (and byte-identical fingerprint sections). Release-gated, the
    /// doc's "release-gated or `--ignored` in debug" option: a run spends
    /// ~335M nodes at depth 6 — ~34 s each in release, unusable in debug.
    /// The depth matches `tools/check.sh`'s pinned fingerprint invocation
    /// `bench 16 1 6 default depth` (P5 pinned 3; the P6 exit audit
    /// raised it to 6 once the TT flattened the curve), and both move
    /// toward the SF-default depth 13 as P7+ shrink the tree.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: two ~34 s suite runs (design/14 §4 step 7)"
    )]
    fn bench_depth_fingerprint_is_deterministic_over_the_default_suite() {
        let params = BenchParams::parse(&["16", "1", "6", "default", "depth"]).unwrap();
        let mut first = Vec::new();
        let mut second = Vec::new();
        let a = run_to(&params, &mut first).unwrap();
        let b = run_to(&params, &mut second).unwrap();
        assert_eq!(
            a.nodes, b.nodes,
            "the bench fingerprint must reproduce (D24)"
        );
        let first = String::from_utf8(first).unwrap();
        let second = String::from_utf8(second).unwrap();
        assert_eq!(fingerprint_section(&first), fingerprint_section(&second));
    }

    #[test]
    fn spawned_bench_depth_runs_are_byte_identical() {
        // doc 06 §4 step 7 / doc 07 §8.5 end to end: two consecutive
        // search-mode bench invocations of the real binary print the same
        // fingerprint section, and stdout stays a clean UCI stream. A
        // small file suite keeps this debug-friendly (`cargo test` spawns
        // the DEBUG binary — the default suite belongs to the
        // release-gated in-process test above and to tools/check.sh).
        let bin = engine_binary();
        let path = std::env::temp_dir().join(format!(
            "chessengine_bench_depth_spawn_{}_{:?}.txt",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(
            &path,
            "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1\n\
             K1k5/8/P7/8/8/8/8/8 w - - 0 1 moves a8a7\n",
        )
        .unwrap();

        let run_once = || {
            let out = std::process::Command::new(&bin)
                .args(["bench", "16", "1", "3", path.to_str().unwrap(), "depth"])
                .output()
                .expect("failed to spawn the engine binary");
            assert!(out.status.success(), "bench exited with {}", out.status);
            assert!(
                out.stdout.is_empty(),
                "bench must keep stdout a clean UCI stream (design/06 §3.6)"
            );
            String::from_utf8(out.stderr).expect("bench output is UTF-8")
        };

        let first = run_once();
        let second = run_once();
        assert_eq!(
            fingerprint_section(&first),
            fingerprint_section(&second),
            "bench depth fingerprint must be byte-identical across runs"
        );

        // Format spot-checks on the spawned output.
        let lines: Vec<&str> = first.lines().collect();
        assert_eq!(lines.len(), 2 * 2 + 4);
        assert!(lines[1].starts_with("Nodes searched: "));
        assert_ne!(lines[1], "Nodes searched: 0", "entry 1 really searched");
        assert_eq!(lines[4], "===========================");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn spawned_bench_perft_runs_are_byte_identical() {
        // doc 06 §4 step 6 done-condition, end to end: two consecutive
        // `chessengine bench 16 1 2 default perft` invocations print the
        // same fingerprint section (everything but the time/nps lines),
        // stdout stays a clean UCI stream, and the total matches the
        // python-chess 1.11.2 cross-checked sum (44,275 at depth 2,
        // recorded 2026-07-10).
        let bin = engine_binary();
        let run_once = || {
            let out = std::process::Command::new(&bin)
                .args(["bench", "16", "1", "2", "default", "perft"])
                .output()
                .expect("failed to spawn the engine binary");
            assert!(out.status.success(), "bench exited with {}", out.status);
            assert!(
                out.stdout.is_empty(),
                "bench must keep stdout a clean UCI stream (design/06 §3.6)"
            );
            String::from_utf8(out.stderr).expect("bench output is UTF-8")
        };

        let first = run_once();
        let second = run_once();
        assert_eq!(
            fingerprint_section(&first),
            fingerprint_section(&second),
            "bench perft fingerprint must be byte-identical across runs"
        );

        // Documented format spot-checks on the spawned output.
        let lines: Vec<&str> = first.lines().collect();
        assert_eq!(lines.len(), 2 * 49 + 4);
        assert_eq!(
            lines[0],
            "Position: 1/49 (rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1)"
        );
        assert_eq!(lines[1], "Nodes searched: 400");
        assert_eq!(lines[98], "===========================");
        assert_eq!(lines[100], "Nodes searched  : 44275");
    }
}
