//! UCI smoke test over the real binary (design/01 §4 step 5; master plan
//! §7 item 4): spawn `chessengine`, script stdin, and assert the handshake
//! answers plus garbage absorption with exit code 0 — the P0 exit gate
//! "the `uciok` smoke test exits 0" (design/00 §3 P0).
//!
//! `CARGO_BIN_EXE_chessengine` is set by Cargo when integration tests
//! build alongside a binary target: it holds the absolute path of the
//! freshly built binary, so there is no target-dir path guessing.

use chessengine::START_FEN;
use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Run the binary with `args`, feed `input` to stdin, close the pipe
/// (EOF), and collect everything.
fn run_engine(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_chessengine"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn chessengine");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(input.as_bytes())
        .expect("write scripted stdin");
    // The taken ChildStdin dropped above => the engine sees EOF after the
    // script (doc 12 §3.3 treats that as quit).
    child.wait_with_output().expect("collect engine output")
}

#[test]
fn uci_handshake_survives_garbage_and_exits_zero() {
    let out = run_engine(
        &[],
        "banana split\nuci\ngo depth 4\nisready\n%%%###\nquit\n",
    );
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout is UTF-8");
    assert!(stdout.contains("uciok"), "missing uciok in:\n{stdout}");
    assert!(stdout.contains("readyok"), "missing readyok in:\n{stdout}");
    // Garbage produced info string lines and did not kill the loop: the
    // post-garbage `isready` was still answered, in order.
    assert!(
        stdout.contains("info string Unknown command: 'banana split'"),
        "garbage line not absorbed:\n{stdout}"
    );
    let uciok_at = stdout.find("uciok").unwrap();
    let readyok_at = stdout.find("readyok").unwrap();
    assert!(uciok_at < readyok_at, "answers out of order:\n{stdout}");
}

#[test]
fn eof_without_quit_exits_zero() {
    // doc 12 §3.3: EOF on stdin is `quit` — cutechess/fastchess just close
    // the pipe, and a hung engine here would hang the whole tournament.
    let out = run_engine(&[], "uci\nisready\n");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("uciok"), "{stdout}");
    assert!(stdout.contains("readyok"), "{stdout}");
}

#[test]
fn argv_d_prints_startpos() {
    // The argv subcommands (design/01 §3.1) must survive the move of
    // main.rs dispatch into uci::run().
    let out = run_engine(&["d"], "");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(&format!("Fen: {START_FEN}")), "{stdout}");
}

#[test]
fn argv_errors_exit_nonzero() {
    // Usage errors are argv-time errors and exit 1 (unlike in-loop
    // garbage, which must never exit — design/12 §3.3).
    for args in [
        &["frobnicate"][..],      // unknown subcommand
        &["perft"][..],           // missing depth
        &["perft", "0"][..],      // divide requires depth >= 1
        &["bench", "banana"][..], // malformed ttMB field
    ] {
        let out = run_engine(args, "");
        assert!(
            !out.status.success(),
            "expected nonzero exit for argv {args:?}"
        );
    }
}
