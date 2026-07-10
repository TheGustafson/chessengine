//! Thin binary entry point (design/01 §3.1).
//!
//! Argv subcommands run to completion before any UCI loop would start:
//! `bench [...]` (design/06 §3.6), `perft <depth> [fen...]` (design/06
//! §3.2), and `d` (design/12 §3.15). With no subcommand this becomes the
//! blocking UCI loop once P4 lands `uci::run()`; until then it prints the
//! start position. Real dispatch logic lives in the library.

use chessengine::{Board, bench, perft};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => cmd_bench(&args[1..]),
        Some("perft") => cmd_perft(&args[1..]),
        // `d`: board diagram + FEN (design/12 §3.15; Zobrist key and
        // checkers join the output once P2 position-state lands). No UCI
        // session yet, so the "current" position is the start position.
        Some("d") | None => {
            print_board(&Board::startpos());
            if args.is_empty() {
                println!("\n(UCI loop lands in P4; subcommands: bench, perft, d)");
            }
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand: '{other}' (expected bench, perft, or d)");
            ExitCode::FAILURE
        }
    }
}

fn print_board(board: &Board) {
    println!("{board}");
    println!("\nFen: {}", board.to_fen());
}

fn cmd_bench(args: &[String]) -> ExitCode {
    match bench::BenchParams::parse(args) {
        Ok(params) => {
            bench::run(&params);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bench: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `chessengine perft <depth> [fen...]` — the CLI convenience face of
/// perft divide (design/06 §3.2); FEN may span several argv tokens.
fn cmd_perft(args: &[String]) -> ExitCode {
    // divide requires depth >= 1 (design/06 §3.2); reject 0 here rather
    // than panic on external input (design/01 §3.4 error zones).
    let depth = args.first().and_then(|tok| tok.parse::<u32>().ok());
    let Some(depth @ 1..) = depth else {
        eprintln!("usage: chessengine perft <depth> [fen...] (depth >= 1)");
        return ExitCode::FAILURE;
    };
    let mut board = if args.len() > 1 {
        match Board::from_fen(&args[1..].join(" ")) {
            Ok(board) => board,
            Err(e) => {
                eprintln!("perft: invalid FEN: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        Board::startpos()
    };
    print!("{}", perft::divide(&mut board, depth));
    ExitCode::SUCCESS
}
