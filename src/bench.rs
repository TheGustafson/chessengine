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
//! Until P5 lands search, only `type=perft` has a code path (design/06
//! §3.6: "this makes bench useful from day one, before search exists"),
//! and that path in turn panics in the P3 movegen stub.

use crate::board::Board;
use crate::perft;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;
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
}

/// SF defaults: `16 1 13 default depth` (design/06 §3.6).
impl Default for BenchParams {
    fn default() -> Self {
        BenchParams {
            tt_mb: 16,
            threads: 1,
            limit: BenchLimit::Depth(13),
            fens: FenSource::Default,
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

        Ok(BenchParams {
            tt_mb,
            threads,
            limit,
            fens,
        })
    }
}

/// Run the bench: one `ucinewgame`, then for each entry set the position
/// (entries may carry `moves ...` suffixes) and search/perft it,
/// accumulating node counts. Diagnostics go to stderr, per SF, so stdout
/// stays a clean UCI stream (design/06 §3.6).
pub fn run(params: &BenchParams) -> BenchResult {
    let entries: Vec<String> = match &params.fens {
        FenSource::Default => BENCH_POSITIONS.iter().map(|s| s.to_string()).collect(),
        FenSource::File(path) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read fen file {}: {e}", path.display()))
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        FenSource::Current => todo!("P4: UCI loop (bench on the session position)"),
    };

    // P5: `ucinewgame` semantics here — allocate the TT at params.tt_mb,
    // clear histories; the TT then persists across positions (design/06
    // §3.6 determinism rules).
    let start = Instant::now();
    let mut nodes: u64 = 0;

    for (i, entry) in entries.iter().enumerate() {
        eprintln!("Position: {}/{} ({entry})", i + 1, entries.len());
        nodes += match params.limit {
            BenchLimit::Perft(depth) => {
                let mut board = board_from_entry(entry);
                perft::perft(&mut board, depth)
            }
            BenchLimit::Depth(_) | BenchLimit::Nodes(_) | BenchLimit::Movetime(_) => {
                todo!("P5: search")
            }
        };
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    // The SF-format trailer (design/06 §3.6). `Nodes searched` is THE
    // number: the `Bench:` commit trailer (design/14 §3.2).
    eprintln!("===========================");
    eprintln!("Total time (ms) : {elapsed_ms}");
    eprintln!("Nodes searched  : {nodes}");
    eprintln!("Nodes/second    : {}", nodes * 1000 / elapsed_ms.max(1));

    BenchResult { nodes, elapsed_ms }
}

/// Set up a bench entry: a FEN, optionally followed by `moves m1 m2 ...`
/// to be replayed through the same parser the UCI `position` command uses
/// (design/06 §3.6).
fn board_from_entry(entry: &str) -> Board {
    let fen = entry.split(" moves").next().unwrap_or(entry);
    let board = Board::from_fen(fen).unwrap_or_else(|e| panic!("invalid bench FEN '{entry}': {e}"));
    if entry.len() != fen.len() {
        // Replaying the moves suffix needs legal movegen + make_move.
        todo!("P3: movegen (replay 'moves ...' suffixes)")
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
    #[ignore = "RED: un-ignore when P3 movegen lands"]
    fn bench_perft_mode_smoke() {
        // design/06 §4 step 6: perft-mode bench runs the whole suite.
        // Depth 1 keeps this cheap; the exact total becomes a known count
        // once recorded per the plan (python-chess cross-check).
        let result = run(&BenchParams::parse(&["16", "1", "1", "default", "perft"]).unwrap());
        assert!(result.nodes > 0);
    }
}
