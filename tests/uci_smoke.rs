//! UCI smoke test over the real binary (design/01 §4 step 5; master plan
//! §7 item 4): spawn `chessengine`, script stdin, and assert the handshake
//! answers plus garbage absorption with exit code 0 — the P0 exit gate
//! "the `uciok` smoke test exits 0" (design/00 §3 P0). From P5 on this
//! file also carries the doc 12 §5 integration items for the live `go`
//! session: exactly-once `bestmove`, `go infinite`/`stop` promptness,
//! `go movetime` clock adherence, multi-`go` sessions, and the
//! `bestmove (none)` terminals — every session must exit 0 with no panic
//! on stderr (worker-thread panics do NOT fail the exit code, so stderr
//! is checked explicitly).
//!
//! `CARGO_BIN_EXE_chessengine` is set by Cargo when integration tests
//! build alongside a binary target: it holds the absolute path of the
//! freshly built binary, so there is no target-dir path guessing.

use chessengine::{Board, MoveList, START_FEN, movegen};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

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
fn handshake_transcript_is_byte_exact() {
    // doc 12 §4 step 2 done-condition: a scripted session produces a
    // byte-exact transcript, options block included (§3.5 v1 table,
    // registration order, correct defaults/ranges — acceptance §8.1).
    let out = run_engine(&[], "uci\nisready\nquit\n");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout is UTF-8");
    let expected = format!(
        "id name {} {}\n\
         id author {}\n\
         option name Debug Log File type string default <empty>\n\
         option name Hash type spin default 16 min 1 max 33554432\n\
         option name Threads type spin default 1 min 1 max 1024\n\
         option name Clear Hash type button\n\
         option name MultiPV type spin default 1 min 1 max 256\n\
         option name Ponder type check default false\n\
         option name Move Overhead type spin default 10 min 0 max 5000\n\
         option name nodestime type spin default 0 min 0 max 10000\n\
         uciok\n\
         readyok\n",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_AUTHORS"),
    );
    assert_eq!(stdout, expected);
}

#[test]
fn position_moves_then_d_shows_expected_fen() {
    // doc 12 §5 integration item 2: `position startpos moves e2e4 e7e5`
    // + `d` reaches the expected FEN through the real binary.
    let out = run_engine(&[], "position startpos moves e2e4 e7e5\nd\nquit\n");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Fen: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        "{stdout}"
    );
}

#[test]
fn position_with_illegal_move_is_rejected_atomically() {
    // doc 12 §3.3: any bad token rejects the WHOLE command with an
    // `info string`, leaving the previous session position in place (SF
    // exits the process here; we stay alive — documented divergence §9.3).
    let out = run_engine(
        &[],
        "position startpos moves d2d4\n\
         position startpos moves e2e4 e2e4\n\
         d\nquit\n",
    );
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("info string Invalid move 'e2e4' in position command"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Fen: rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"),
        "the d2d4 position must survive the failed command:\n{stdout}"
    );
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

// ---------------------------------------------------------------------------
// The live go session (doc 12 §4 steps 5-6; §5 integration items 3-8)
// ---------------------------------------------------------------------------

/// A timestamped engine output line, as observed by the reader thread.
type Lines = Arc<Mutex<Vec<(Instant, String)>>>;

/// An interactive engine session: stdin stays open between commands so
/// tests can interleave input with the search's own timing (a scripted
/// one-shot pipe cannot — the very next line would implicitly stop the
/// search, doc 12 §3.2). A reader thread timestamps every stdout line as
/// it arrives, so ordering assertions ("no bestmove before stop") work.
struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Lines,
    stderr: Arc<Mutex<String>>,
    readers: Vec<JoinHandle<()>>,
}

impl Session {
    fn spawn() -> Session {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chessengine"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn chessengine");
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr_pipe = child.stderr.take().expect("stderr is piped");

        let lines: Lines = Arc::new(Mutex::new(Vec::new()));
        let stdout_lines = Arc::clone(&lines);
        let stdout_reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                stdout_lines.lock().unwrap().push((Instant::now(), line));
            }
        });
        let stderr = Arc::new(Mutex::new(String::new()));
        let stderr_buf = Arc::clone(&stderr);
        let stderr_reader = thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr_pipe).read_to_string(&mut buf);
            *stderr_buf.lock().unwrap() = buf;
        });

        Session {
            child,
            stdin: Some(stdin),
            lines,
            stderr,
            readers: vec![stdout_reader, stderr_reader],
        }
    }

    /// Send one command line; the write must succeed (the engine is alive).
    fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("session still open");
        writeln!(stdin, "{line}").expect("write to engine stdin");
        stdin.flush().expect("flush engine stdin");
    }

    /// Block until some output line satisfies `pred` (or fail after
    /// `timeout`), returning its arrival timestamp. Timeouts are generous:
    /// these tests assert protocol behavior, not scheduler latency.
    fn wait_for(&self, pred: impl Fn(&str) -> bool, timeout: Duration) -> Instant {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(&(at, _)) = self.lines.lock().unwrap().iter().find(|(_, l)| pred(l)) {
                return at;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for engine output; transcript so far:\n{}",
                self.transcript()
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn transcript(&self) -> String {
        let lines = self.lines.lock().unwrap();
        lines
            .iter()
            .map(|(_, l)| l.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Close stdin (EOF == quit, doc 12 §3.3) and reap the process with a
    /// hard timeout so a hung engine fails the test instead of the suite.
    /// Returns (exit status, timestamped stdout lines, stderr).
    fn finish(mut self) -> (ExitStatus, Vec<(Instant, String)>, String) {
        drop(self.stdin.take()); // EOF
        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("wait on engine") {
                break status;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!(
                    "engine did not exit after EOF; transcript:\n{}",
                    self.transcript()
                );
            }
            thread::sleep(Duration::from_millis(5));
        };
        for reader in self.readers {
            reader.join().expect("reader thread");
        }
        let lines = std::mem::take(&mut *self.lines.lock().unwrap());
        let stderr = std::mem::take(&mut *self.stderr.lock().unwrap());
        (status, lines, stderr)
    }
}

/// Every session must exit 0 with no panic on stderr (a panicking search
/// thread would NOT change the exit code — stderr is the only witness).
fn assert_clean_exit(status: ExitStatus, stderr: &str) {
    assert!(status.success(), "engine exited with {status}: {stderr}");
    assert!(!stderr.contains("panicked"), "worker panicked:\n{stderr}");
}

fn bestmove_lines(lines: &[(Instant, String)]) -> Vec<&(Instant, String)> {
    lines
        .iter()
        .filter(|(_, l)| l.starts_with("bestmove "))
        .collect()
}

/// The `bestmove` token must be a legal move of `fen`.
fn assert_legal_bestmove(line: &str, fen: &str) {
    let token = line
        .split_whitespace()
        .nth(1)
        .unwrap_or_else(|| panic!("malformed bestmove line: {line}"));
    let board = Board::from_fen(fen).expect("test FEN");
    let mut legal = MoveList::new();
    movegen::generate_legal(&board, &mut legal);
    assert!(
        legal.iter().any(|m| m.to_string() == token),
        "bestmove {token} is not legal in {fen}"
    );
}

#[test]
fn go_depth_completes_with_exactly_one_legal_bestmove() {
    // doc 12 §5 item 3: info lines, then exactly one bestmove — with the
    // search left to finish on its own (interactive session; a scripted
    // `quit` would implicitly stop it mid-iteration).
    let mut s = Session::spawn();
    s.send("position startpos");
    s.send("go depth 3");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(20));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(bests.len(), 1, "exactly one bestmove per go");
    assert_legal_bestmove(&bests[0].1, START_FEN);
    // Monotone completed-iteration reports, each with the §3.13 v1 fields.
    let depths: Vec<&str> = lines
        .iter()
        .filter(|(_, l)| l.starts_with("info depth "))
        .map(|(_, l)| l.split_whitespace().nth(2).unwrap())
        .collect();
    assert_eq!(depths, ["1", "2", "3"], "one info line per iteration");
    for (_, line) in lines.iter().filter(|(_, l)| l.starts_with("info depth ")) {
        for field in [
            " seldepth ",
            " score ",
            " nodes ",
            " nps ",
            " time ",
            " pv ",
        ] {
            assert!(line.contains(field), "missing{field}in: {line}");
        }
    }
}

#[test]
fn go_infinite_bestmove_comes_only_after_stop_and_promptly() {
    // doc 12 §5 item 4 + the task's promptness bar: 100 ms of infinite
    // search, then `stop` → exactly one bestmove, only after the stop,
    // within a comfortable latency bound (§3.2: bounded by one check_time
    // interval plus join time — microseconds; 2 s absorbs CI scheduling).
    let mut s = Session::spawn();
    s.send("position startpos");
    s.send("go infinite");
    thread::sleep(Duration::from_millis(100));
    let stop_at = Instant::now();
    s.send("stop");
    let seen_at = s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(5));
    assert!(
        seen_at.duration_since(stop_at) < Duration::from_secs(2),
        "bestmove not prompt after stop: {:?}",
        seen_at.duration_since(stop_at)
    );
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(bests.len(), 1, "exactly one bestmove per go");
    assert!(
        bests[0].0 >= stop_at,
        "bestmove observed before stop was sent (busy-wait rule, §3.2)"
    );
    assert_legal_bestmove(&bests[0].1, START_FEN);
}

#[test]
fn go_movetime_respects_the_clock_within_tolerance() {
    // doc 12 §5 item 6: `go movetime 200` answers no earlier than the
    // budget (nothing else can stop this search) and not much later
    // (elapsed − budget ≤ one check interval, §6 — the wide upper bound
    // is CI-scheduler slack, not engine latency).
    let mut s = Session::spawn();
    s.send("position startpos");
    let go_at = Instant::now();
    s.send("go movetime 200");
    let seen_at = s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(5));
    let elapsed = seen_at.duration_since(go_at);
    assert!(
        elapsed >= Duration::from_millis(180),
        "bestmove after {elapsed:?} — movetime 200 ended too early"
    );
    assert!(
        elapsed <= Duration::from_millis(1500),
        "bestmove after {elapsed:?} — movetime 200 overshot"
    );
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    assert_eq!(bestmove_lines(&lines).len(), 1);
}

#[test]
fn two_go_cycles_and_a_position_change_in_one_session() {
    // Task bar: two go cycles in one session; the position change between
    // them applies. Distinct forced mates make the bestmoves prove it.
    const WHITE_MATES: &str = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
    const BLACK_MATES: &str = "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1";
    let mut s = Session::spawn();
    s.send(&format!("position fen {WHITE_MATES}"));
    s.send("go depth 3");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(20));
    s.send(&format!("position fen {BLACK_MATES}"));
    s.send("go depth 3");
    let all_bestmoves = |lines: &[(Instant, String)]| {
        lines
            .iter()
            .filter(|(_, l)| l.starts_with("bestmove"))
            .count()
    };
    s.wait_for(
        move |l| l.starts_with("bestmove d8d1"),
        Duration::from_secs(20),
    );
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    assert_eq!(all_bestmoves(&lines), 2, "one bestmove per go");
    let bests = bestmove_lines(&lines);
    assert!(bests[0].1.starts_with("bestmove d1d8"), "{}", bests[0].1);
    assert!(bests[1].1.starts_with("bestmove d8d1"), "{}", bests[1].1);
}

#[test]
fn terminal_roots_answer_info_depth_0_and_bestmove_none() {
    // doc 12 §5 item 7: mated and stalemated roots. Scripted pipes are
    // fine here — the answer is immediate and deterministic.
    const MATED: &str = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
    let out = run_engine(&[], &format!("position fen {MATED}\ngo depth 3\nquit\n"));
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("info depth 0 score mate 0\n"), "{stdout}");
    assert_eq!(stdout.matches("bestmove").count(), 1, "{stdout}");
    assert!(stdout.contains("bestmove (none)\n"), "{stdout}");

    const STALEMATED: &str = "k7/8/1K6/8/8/8/7Q/8 b - - 0 1";
    let out = run_engine(
        &[],
        &format!("position fen {STALEMATED}\ngo depth 3\nquit\n"),
    );
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("info depth 0 score cp 0\n"), "{stdout}");
    assert_eq!(stdout.matches("bestmove").count(), 1, "{stdout}");
    assert!(stdout.contains("bestmove (none)\n"), "{stdout}");
}

#[test]
fn blocking_position_during_infinite_search_implicitly_stops_it() {
    // doc 12 §5 item 8: `go infinite`, then a mid-search `position` —
    // the pending bestmove flushes (exactly once), the new position
    // applies (`d` confirms), and nothing hangs.
    let mut s = Session::spawn();
    s.send("go infinite");
    thread::sleep(Duration::from_millis(100));
    s.send("position startpos moves e2e4");
    s.send("d");
    s.wait_for(|l| l.starts_with("Fen: "), Duration::from_secs(5));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    assert_eq!(
        bestmove_lines(&lines).len(),
        1,
        "implicit stop flushes bestmove"
    );
    let fen_line = lines.iter().find(|(_, l)| l.starts_with("Fen: ")).unwrap();
    assert_eq!(
        fen_line.1,
        "Fen: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    );
    // The bestmove flushed before the new position's diagram printed.
    let best_at = bestmove_lines(&lines)[0].0;
    assert!(best_at <= fen_line.0, "bestmove must precede the d output");
}

#[test]
fn isready_answers_immediately_during_a_search() {
    // doc 12 §3.2: isready is a liveness ping, not a barrier — it must be
    // answered while the infinite search keeps running.
    let mut s = Session::spawn();
    s.send("go infinite");
    thread::sleep(Duration::from_millis(50));
    s.send("isready");
    let ready_at = s.wait_for(|l| l == "readyok", Duration::from_secs(5));
    // No bestmove may have been observed before readyok (nothing stopped).
    {
        let lines = s.lines.lock().unwrap();
        assert!(
            !lines
                .iter()
                .any(|(at, l)| l.starts_with("bestmove") && *at <= ready_at),
            "search must still be running at readyok"
        );
    }
    s.send("stop");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(5));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    assert_eq!(bestmove_lines(&lines).len(), 1);
}
