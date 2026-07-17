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
    // Under `datagen` the four ObsCapture options (design/16 §3.5.5) follow
    // EvalFile; the default (tournament/bench) binary must not carry them.
    let obs_block = if cfg!(feature = "datagen") {
        "option name ObsCapturePath type string default <empty>\n\
         option name ObsCaptureMinDepth type spin default 6 min 0 max 246\n\
         option name ObsCaptureSampleBits type spin default 4 min 0 max 32\n\
         option name ObsCapturePerGameCap type spin default 20000 min 0 max 100000000\n"
    } else {
        ""
    };
    let expected = format!(
        "id name {} {}\n\
         id author {}\n\
         option name Debug Log File type string default <empty>\n\
         option name Hash type spin default 16 min 1 max 33554432\n\
         option name Threads type spin default 1 min 1 max {threads_max}\n\
         option name Clear Hash type button\n\
         option name MultiPV type spin default 1 min 1 max 256\n\
         option name Ponder type check default false\n\
         option name Move Overhead type spin default 10 min 0 max 5000\n\
         option name nodestime type spin default 0 min 0 max 10000\n\
         option name Minimal type check default false\n\
         option name EvalFile type string default <empty>\n\
         {obs_block}\
         uciok\n\
         readyok\n",
        "ARCHON",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_AUTHORS"),
        // The Threads ceiling is hardware-scaled (doc 12 §9.6): compute the
        // same value the engine registers so this stays byte-exact anywhere.
        threads_max = chessengine::uci::max_threads(),
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

/// Drop the wall-clock `nps <x>` and `time <x>` tokens from an info line so
/// two `nodestime` runs (which differ only in real time) compare equal.
fn strip_walltime(line: &str) -> String {
    let toks: Vec<&str> = line.split_whitespace().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if (toks[i] == "nps" || toks[i] == "time") && i + 1 < toks.len() {
            i += 2;
        } else {
            out.push(toks[i]);
            i += 1;
        }
    }
    out.join(" ")
}

#[test]
fn nodestime_games_are_deterministic() {
    // doc 12 §5 / step 11: the node clock removes wall-clock from every stop
    // decision, so two identical scripted `nodestime` games produce
    // identical move sequences AND node counts. Fixed positions (chosen
    // moves, not the engine's) let the cross-move `advance_nodes_time`
    // budget carry across three `go`s; wall-clock nps/time are stripped.
    fn transcript() -> Vec<String> {
        let mut s = Session::spawn();
        s.send("setoption name nodestime value 1000");
        s.send("ucinewgame");
        for (i, pos) in [
            "position startpos",
            "position startpos moves e2e4",
            "position startpos moves e2e4 e7e5",
        ]
        .iter()
        .enumerate()
        {
            s.send(pos);
            s.send("go wtime 3000 btime 3000");
            // Wait for THIS move's bestmove before the next position, so the
            // next `position`'s implicit stop never truncates this search.
            wait_for_bestmove_n(&s, i + 1, Duration::from_secs(30));
        }
        let (status, lines, stderr) = s.finish();
        assert_clean_exit(status, &stderr);
        lines
            .iter()
            .map(|(_, l)| l.as_str())
            .filter(|l| l.starts_with("info depth ") || l.starts_with("bestmove "))
            .map(strip_walltime)
            .collect()
    }
    let a = transcript();
    let b = transcript();
    assert!(!a.is_empty(), "the game produced info/bestmove output");
    assert_eq!(a, b, "nodestime games must be bit-deterministic");
}

#[test]
fn perft_command_matches_known_counts() {
    // doc 12 §5 / step 12: `perft N` and `go perft N` both run divide from
    // the session position and print the known node totals.
    let out = run_engine(
        &[],
        "position startpos\nperft 4\ngo perft 3\nposition startpos moves e2e4\nperft 3\nquit\n",
    );
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Nodes searched: 197281"),
        "perft 4 startpos total wrong:\n{stdout}"
    );
    assert!(
        stdout.contains("Nodes searched: 8902"),
        "go perft 3 startpos total wrong:\n{stdout}"
    );
    // After 1.e4, perft 3 is 13160 — proves it runs from the session position.
    assert!(
        stdout.contains("Nodes searched: 13160"),
        "perft 3 after e2e4 total wrong:\n{stdout}"
    );
}

#[test]
fn eval_command_prints_a_static_evaluation() {
    // doc 12 §5 / step 12: `eval` prints the Phase A static evaluation of
    // the session position. Symmetric start position -> exactly 0.
    let out = run_engine(&[], "position startpos\neval\nquit\n");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Final evaluation: 0 "),
        "eval of the symmetric start position must be 0:\n{stdout}"
    );
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

/// Block until the session has produced `n` bestmove lines; returns the
/// n-th line's (arrival timestamp, move token).
fn wait_for_bestmove_n(s: &Session, n: usize, timeout: Duration) -> (Instant, String) {
    let deadline = Instant::now() + timeout;
    loop {
        {
            let lines = s.lines.lock().unwrap();
            let mut bests = lines.iter().filter(|(_, l)| l.starts_with("bestmove "));
            if let Some((at, line)) = bests.nth(n - 1) {
                let token = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_else(|| panic!("malformed bestmove line: {line}"))
                    .to_string();
                return (*at, token);
            }
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for bestmove #{n}; transcript:\n{}",
            s.transcript()
        );
        thread::sleep(Duration::from_millis(2));
    }
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
fn multipv_emits_three_ordered_lines_per_iteration() {
    // doc 12 §3.12/§8.5: MultiPV=3 `go depth 6` prints 3 info lines per
    // iteration, multipv 1..3, with weakly-decreasing exact scores. The
    // final iteration (depth 6) is inspected — a completed, non-aborted
    // block whose three lines are searched exactly.
    let mut s = Session::spawn();
    s.send("setoption name MultiPV value 3");
    s.send("position startpos");
    s.send("go depth 6");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(20));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    // The depth-6 block: three lines, multipv 1/2/3 in order.
    let depth6: Vec<&str> = lines
        .iter()
        .map(|(_, l)| l.as_str())
        .filter(|l| l.starts_with("info depth 6 "))
        .collect();
    assert_eq!(
        depth6.len(),
        3,
        "MultiPV=3 emits 3 lines at depth 6:\n{depth6:?}"
    );
    let mut scores = Vec::new();
    for (i, line) in depth6.iter().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let mpv = toks[toks.iter().position(|&t| t == "multipv").unwrap() + 1];
        assert_eq!(mpv, (i + 1).to_string(), "multipv index in order: {line}");
        // score is `cp N` (no mate at the start position, low depth).
        let sp = toks.iter().position(|&t| t == "score").unwrap();
        assert_eq!(toks[sp + 1], "cp", "expected a cp score: {line}");
        scores.push(toks[sp + 2].parse::<i64>().expect("cp value"));
    }
    assert!(
        scores[0] >= scores[1] && scores[1] >= scores[2],
        "exact scores must weakly decrease across multipv 1..3: {scores:?}"
    );
}

#[test]
fn minimal_report_mode_prints_one_summary_line() {
    // doc 12 §3.13 / D31: `Minimal` suppresses the per-iteration info lines
    // and prints exactly one summary line before `bestmove` (log-volume
    // relief for high-frequency self-play). Pure I/O — no search change.
    let mut s = Session::spawn();
    s.send("setoption name Minimal value true");
    s.send("position startpos");
    s.send("go depth 8");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(20));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let infos = lines
        .iter()
        .filter(|(_, l)| l.starts_with("info depth "))
        .count();
    assert_eq!(
        infos,
        1,
        "Minimal prints exactly one summary info line, not one per iteration:\n{}",
        s_transcript(&lines)
    );
    assert_eq!(bestmove_lines(&lines).len(), 1, "exactly one bestmove");
}

/// Join timestamped lines for an assertion message.
fn s_transcript(lines: &[(Instant, String)]) -> String {
    lines
        .iter()
        .map(|(_, l)| l.as_str())
        .collect::<Vec<_>>()
        .join("\n")
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
fn stop_before_any_go_is_a_quiet_noop() {
    // doc 07 §8.9 / P5 exit: `stop` with no search running must not hang,
    // must emit no bestmove, and must leave the session fully usable.
    let mut s = Session::spawn();
    s.send("stop");
    s.send("isready");
    s.wait_for(|l| l == "readyok", Duration::from_secs(5));
    s.send("position startpos");
    s.send("go depth 2");
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(20));
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(bests.len(), 1, "an idle stop must not produce a bestmove");
    assert_legal_bestmove(&bests[0].1, START_FEN);
}

#[test]
fn rapid_stop_after_go_always_yields_one_legal_bestmove() {
    // doc 07 §8.9 / P5 exit: `stop` racing the search startup — no delay
    // between `go infinite` and `stop`, repeated in one session to shake
    // the thread-spawn-vs-stop-flag race. Every cycle must produce exactly
    // one legal bestmove (never zero — the flag may win the race and abort
    // iteration 1, but the driver still owes a move), and never two.
    let mut s = Session::spawn();
    s.send("position startpos");
    const CYCLES: usize = 5;
    for want in 1..=CYCLES {
        s.send("go infinite");
        s.send("stop");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let n = {
                let lines = s.lines.lock().unwrap();
                lines
                    .iter()
                    .filter(|(_, l)| l.starts_with("bestmove "))
                    .count()
            };
            if n >= want {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for bestmove #{want}; transcript:\n{}",
                s.transcript()
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(
        bests.len(),
        CYCLES,
        "exactly one bestmove per go/stop cycle"
    );
    for (_, line) in bests {
        assert_legal_bestmove(line, START_FEN);
    }
}

#[test]
fn stop_during_deep_search_is_prompt_and_exactly_once() {
    // doc 07 §8.9 / P5 exit: a stop landing mid-tree in a deep search.
    // `go depth 30` is unreachable at P5 speeds, so the search is deep in
    // some iteration ≥ 6 when the stop arrives; the reply must still be
    // prompt (bounded by the check_time cadence, §3.9) and exactly once.
    let mut s = Session::spawn();
    s.send("position startpos");
    s.send("go depth 30");
    s.wait_for(|l| l.starts_with("info depth 5 "), Duration::from_secs(60));
    let stop_at = Instant::now();
    s.send("stop");
    let seen_at = s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(5));
    assert!(
        seen_at.duration_since(stop_at) < Duration::from_secs(2),
        "bestmove not prompt after mid-iteration stop: {:?}",
        seen_at.duration_since(stop_at)
    );
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(bests.len(), 1, "exactly one bestmove per go");
    assert!(
        bests[0].0 >= stop_at,
        "bestmove observed before stop (infinite-depth search self-stopped?)"
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
fn go_with_clocks_returns_promptly_and_spends_near_optimum() {
    // doc 12 step 7: the wtime/btime clock path is live end to end. With
    // 10 s on the clock the TM allocates optimum ≈ 150 ms and maximum
    // ≈ 1 s (§3.7 at ply 0); the v1 soft stop (§3.9 subset) ends the
    // search at the first iteration boundary past optimum. So the answer
    // must take at least ~optimum (nothing stops a clock search sooner)
    // and land well under both the hard wall + slack and the 10 s clock.
    // Bounds are loose on purpose: they assert TM behavior, not
    // CI-scheduler latency.
    let mut s = Session::spawn();
    s.send("position startpos");
    let go_at = Instant::now();
    s.send("go wtime 10000 btime 10000");
    let seen_at = s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(8));
    let elapsed = seen_at.duration_since(go_at);
    assert!(
        elapsed >= Duration::from_millis(50),
        "bestmove after {elapsed:?} — a clock search cannot stop before optimum"
    );
    assert!(
        elapsed <= Duration::from_secs(3),
        "bestmove after {elapsed:?} — nowhere near the ~150 ms optimum"
    );
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    let bests = bestmove_lines(&lines);
    assert_eq!(bests.len(), 1, "exactly one bestmove per go");
    assert_legal_bestmove(&bests[0].1, START_FEN);
}

#[test]
fn clock_game_of_20_moves_never_forfeits_at_10_plus_01() {
    // doc 12 step 7's done-condition scaled to a test: a 10+0.1 game
    // simulation with real clock bookkeeping. GUI-style accounting:
    // elapsed runs from the `go` send to the bestmove arrival, so pipe
    // and parse overhead count against the engine (Move Overhead's job
    // to price in). Every move is validated against an independent
    // board; overspending the remaining clock on any move is a forfeit
    // and fails the test.
    const INC_MS: i64 = 100;
    let mut clocks: [i64; 2] = [10_000, 10_000]; // [white, black] ms
    let mut moves: Vec<String> = Vec::new();
    let mut board = Board::startpos();
    let mut s = Session::spawn();
    for ply in 0..40usize {
        let stm = ply % 2;
        if moves.is_empty() {
            s.send("position startpos");
        } else {
            s.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let go_at = Instant::now();
        s.send(&format!(
            "go wtime {} btime {} winc {INC_MS} binc {INC_MS}",
            clocks[0], clocks[1]
        ));
        let (seen_at, token) = wait_for_bestmove_n(&s, ply + 1, Duration::from_secs(15));
        if token == "(none)" {
            break; // checkmate/stalemate ended the game early
        }
        let elapsed = seen_at.duration_since(go_at).as_millis() as i64;
        assert!(
            elapsed <= clocks[stm],
            "time forfeit at ply {ply}: spent {elapsed} ms with {} ms left",
            clocks[stm]
        );
        clocks[stm] = clocks[stm] - elapsed + INC_MS;
        // Replay the move on the reference board: legality proof, and
        // keeps the `position ... moves` line honest for the next ply.
        let mut legal = MoveList::new();
        movegen::generate_legal(&board, &mut legal);
        let mv = legal
            .iter()
            .copied()
            .find(|m| m.to_string() == token)
            .unwrap_or_else(|| panic!("illegal bestmove {token} at ply {ply}: {}", board.to_fen()));
        board.make_move(mv, board.gives_check(mv));
        moves.push(token);
    }
    let (status, lines, stderr) = s.finish();
    assert_clean_exit(status, &stderr);
    // Exactly one bestmove per `go`: the count equals the plies played,
    // plus one for a terminal `(none)` if the game ended early.
    let bests = bestmove_lines(&lines).len();
    assert!(
        bests == moves.len() || bests == moves.len() + 1,
        "{bests} bestmoves for {} plies",
        moves.len()
    );
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
