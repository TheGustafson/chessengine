//! UCI entry point: argv subcommand dispatch, then the blocking stdin loop.
//!
//! Contracts: `design/01-architecture.md` §3.1/§3.3.18 (the `run()` seam;
//! argv subcommands run to completion before the loop) and
//! `design/12-uci-time-management.md` §3.3 (command grammar and the
//! robustness rules: never panic on input, unknown commands answer
//! `info string`, EOF on stdin is `quit` — GUIs close the pipe, and a dead
//! engine loses on time).
//!
//! P0 skeleton (design/01 §4 step 5): the loop answers `uci`, `isready`,
//! `d`, and `quit`, consumes `debug`/`register` silently per spec, and
//! absorbs every other line — including `position`/`go`/`setoption`, which
//! P4/P5 wire to the full doc 12 grammar — with `info string`. In-session
//! `bench`/`perft` (design/01 §3.1) also wait for P3/P4: today their
//! implementations stop at `todo!` movegen stubs, and the loop must never
//! panic; their argv forms below run as far as their phases allow.

use crate::bench;
use crate::board::Board;
use crate::perft;
use std::fmt;
use std::io::{self, BufRead, Write};

/// `id name` text (doc 12 §3.3): crate name + version from the manifest,
/// so the handshake can never drift from `Cargo.toml`.
const NAME: &str = concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"));
/// `id author` text, likewise from the manifest `authors` field.
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// The program front door (design/01 §3.1): argv subcommands (`bench`,
/// `perft <depth> [fen...]`, `d`) run to completion; with no subcommand
/// this blocks on the stdin UCI loop until `quit`/EOF.
pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => cmd_bench_argv(&args[1..]),
        Some("perft") => cmd_perft_argv(&args[1..]),
        // `d`: board diagram + FEN (design/12 §3.15; Zobrist key and
        // checkers join the output once P2 position-state lands). Outside
        // a UCI session the "current" position is the start position.
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

/// The blocking UCI loop, generic over the byte streams (`impl BufRead` /
/// `impl Write`) so unit tests drive it with in-memory buffers while
/// `run()` passes locked stdin/stdout.
///
/// Loop-exit rules (doc 12 §3.3): `quit`, stdin EOF, or a read error all
/// leave the loop; a write error means the GUI is gone, same thing. No
/// input line may panic.
fn uci_loop(input: impl BufRead, out: &mut impl Write) {
    // The session position; `position` starts mutating it in P4 — until
    // then `d` always shows the start position.
    let board = Board::startpos();
    for line in input.lines() {
        let Ok(line) = line else { break };
        match dispatch(&board, &line, out) {
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

/// Handle one input line (the doc 12 §3.3 dispatch shape, P0 subset).
/// Returns `Ok(false)` exactly when the loop should end (`quit`).
fn dispatch(board: &Board, line: &str, out: &mut impl Write) -> io::Result<bool> {
    let mut tokens = line.split_whitespace();
    match tokens.next() {
        Some("uci") => {
            writeln!(out, "id name {NAME}")?;
            writeln!(out, "id author {AUTHOR}")?;
            // One `option` line per registered option goes here once the
            // Options table lands (P4, doc 12 §3.5).
            writeln!(out, "uciok")?;
        }
        // Trivially ready until searches run on worker threads (P4).
        Some("isready") => writeln!(out, "readyok")?,
        Some("d") => print_board(out, board)?,
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

/// `d` output: diagram + `Fen:` line (doc 12 §3.15; the `Key:` and
/// `Checkers:` lines join once P2 lands Zobrist and check info).
fn print_board(out: &mut impl Write, board: &Board) -> io::Result<()> {
    writeln!(out, "{board}")?;
    writeln!(out)?;
    writeln!(out, "Fen: {}", board.to_fen())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_FEN;

    /// Drive the loop over a scripted stdin, returning captured stdout.
    fn session(input: &str) -> String {
        let mut out = Vec::new();
        uci_loop(input.as_bytes(), &mut out);
        String::from_utf8(out).expect("engine output is UTF-8")
    }

    #[test]
    fn uci_handshake() {
        // doc 12 §3.3: id name, id author, then uciok.
        let out = session("uci\nquit\n");
        assert!(out.contains("id name chessengine "), "{out}");
        assert!(out.contains("id author "), "{out}");
        let id_at = out.find("id name").unwrap();
        let uciok_at = out.find("uciok\n").unwrap();
        assert!(id_at < uciok_at, "uciok must follow the id lines: {out}");
    }

    #[test]
    fn isready_answers_readyok() {
        assert!(session("isready\nquit\n").contains("readyok\n"));
    }

    #[test]
    fn d_prints_diagram_and_fen() {
        let out = session("d\nquit\n");
        assert!(out.contains(&format!("Fen: {START_FEN}")), "{out}");
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
    fn blank_lines_are_ignored() {
        let out = session("\n   \n\t\nisready\nquit\n");
        assert!(!out.contains("Unknown command"), "{out}");
        assert!(out.contains("readyok\n"), "{out}");
    }

    #[test]
    fn no_panic_on_junk_torrent() {
        // "Never panic on any input" — near-miss commands, P4/P5 commands
        // that exist in the grammar but not in the P0 subset (they must
        // absorb, not hit todo! stubs), non-ASCII, and control characters.
        let junk = "uciready\nucii\nquitt quit\nd d d\nположение\n\u{1}\u{7f}\n\
                    go depth 4\nposition startpos moves e2e4\n\
                    setoption name Hash value 16\nstop\nponderhit\n\
                    ucinewgame\neval\nbench\nperft 1\ngo perft 5\nquit\n";
        let out = session(junk);
        // ... and the loop stayed alive through all of it up to `quit`.
        assert!(
            out.contains("info string Unknown command: 'perft 1'"),
            "{out}"
        );
    }
}
