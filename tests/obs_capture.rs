//! End-to-end in-search observation capture (design/16 §3.5.5, D45a).
//!
//! Drives the real `chessengine` binary over UCI with `ObsCapture*` options set,
//! then reads the produced shard back through the store's own `ShardReader` and
//! the `obs-tool verify` binary. The whole file compiles to nothing without the
//! `datagen` feature (the options + recorder are feature-gated).
#![cfg(feature = "datagen")]

use chessengine::obs::{EpisodeKind, ShardReader};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// A minimal interactive UCI session: send commands, wait for output lines.
/// Unlike a one-shot stdin pipe, this lets a `go nodes N` search RUN TO
/// COMPLETION (we wait for its `bestmove`) before EOF, instead of the `quit`
/// racing in and aborting it.
struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Arc<Mutex<Vec<String>>>,
    reader: Option<JoinHandle<()>>,
}

impl Session {
    fn spawn() -> Session {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chessengine"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn chessengine");
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&lines);
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                sink.lock().unwrap().push(line);
            }
        });
        Session {
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
        }
    }

    fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("session open");
        writeln!(stdin, "{line}").expect("write stdin");
        stdin.flush().expect("flush stdin");
    }

    /// Block until an output line satisfies `pred`, or fail after `timeout`.
    fn wait_for(&self, pred: impl Fn(&str) -> bool, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            if self.lines.lock().unwrap().iter().any(|l| pred(l)) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out; transcript:\n{}",
                self.lines.lock().unwrap().join("\n")
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Block until at least `n` `bestmove` lines have been produced.
    fn wait_for_bestmove_count(&self, n: usize, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            let count = self
                .lines
                .lock()
                .unwrap()
                .iter()
                .filter(|l| l.starts_with("bestmove "))
                .count();
            if count >= n {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {n} bestmove(s); transcript:\n{}",
                self.lines.lock().unwrap().join("\n")
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// EOF (== quit) then reap; returns the collected stdout lines. Closing
    /// stdin triggers the recorder's final flush on engine drop.
    fn finish(mut self) -> Vec<String> {
        drop(self.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if self.child.try_wait().expect("wait").is_some() {
                break;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!("engine did not exit after EOF");
            }
            thread::sleep(Duration::from_millis(5));
        }
        if let Some(reader) = self.reader.take() {
            reader.join().expect("reader thread");
        }
        std::mem::take(&mut *self.lines.lock().unwrap())
    }
}

fn temp(name: &str) -> PathBuf {
    let mut base = std::env::temp_dir();
    base.push(format!(
        "archon-obs-it-{name}-{}-{:?}.ceobs",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&base);
    base
}

fn obs_tool_bin() -> PathBuf {
    // obs-tool lives beside chessengine in the same target dir.
    Path::new(env!("CARGO_BIN_EXE_chessengine")).with_file_name("obs-tool")
}

/// The last `nodes N` reported across info lines (monotonic → final count).
fn last_nodes(lines: &[String]) -> u64 {
    lines
        .iter()
        .rev()
        .find_map(|l| {
            if !l.starts_with("info") {
                return None;
            }
            let toks: Vec<&str> = l.split_whitespace().collect();
            let i = toks.iter().position(|&t| t == "nodes")?;
            toks.get(i + 1)?.parse().ok()
        })
        .expect("an info line carrying a node count")
}

fn bestmove(lines: &[String]) -> String {
    lines
        .iter()
        .find_map(|l| {
            l.strip_prefix("bestmove ")
                .map(|rest| rest.split_whitespace().next().unwrap_or("").to_string())
        })
        .expect("a bestmove line")
}

/// Read every episode of a shard (fails on any decode error).
fn read_episodes(path: &Path) -> Vec<chessengine::obs::Episode> {
    let mut reader = ShardReader::open(path).expect("open shard");
    let mut out = Vec::new();
    while let Some(episode) = reader.next_episode().expect("decode episode") {
        out.push(episode);
    }
    out
}

/// `obs-tool verify` must exit 0 (built under datagen alongside the engine).
fn assert_obs_tool_verify(path: &Path) {
    let bin = obs_tool_bin();
    assert!(bin.exists(), "obs-tool binary missing at {}", bin.display());
    let output = Command::new(bin)
        .arg("verify")
        .arg(path)
        .output()
        .expect("run obs-tool verify");
    assert!(
        output.status.success(),
        "obs-tool verify failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("0 problems"),
        "obs-tool verify did not report 0 problems: {}",
        String::from_utf8_lossy(&output.stdout),
    );
}

const NODES: &str = "go nodes 50000";

/// Capture must be OBSERVATION-ONLY: with capture on (max sampling) vs off, the
/// SAME search returns identical node count and bestmove. This is the core
/// safety property — the sample gate is a pure function of the position key, so
/// nothing about the search changes.
#[test]
fn capture_does_not_perturb_the_search() {
    // Off.
    let mut off = Session::spawn();
    off.send("position startpos");
    off.send(NODES);
    off.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(30));
    let off_lines = off.finish();

    // On, maximum capture: sample_bits 0 (every key hits), min_depth 2.
    let path = temp("identity");
    let mut on = Session::spawn();
    on.send(&format!(
        "setoption name ObsCapturePath value {}",
        path.display()
    ));
    on.send("setoption name ObsCaptureMinDepth value 2");
    on.send("setoption name ObsCaptureSampleBits value 0");
    on.send("position startpos");
    on.send(NODES);
    on.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(60));
    let on_lines = on.finish();

    assert_eq!(
        last_nodes(&off_lines),
        last_nodes(&on_lines),
        "node count changed with capture enabled"
    );
    assert_eq!(
        bestmove(&off_lines),
        bestmove(&on_lines),
        "bestmove changed with capture enabled"
    );
    std::fs::remove_file(&path).ok();
}

/// A shard written during a real `go nodes` search decodes through ShardReader
/// AND passes `obs-tool verify`, carrying InSearchCapture episodes with
/// per-move rows and full per-record position identity.
#[test]
fn shard_from_a_real_search_verifies() {
    let path = temp("verify");
    let mut s = Session::spawn();
    s.send(&format!(
        "setoption name ObsCapturePath value {}",
        path.display()
    ));
    s.send("setoption name ObsCaptureMinDepth value 4");
    s.send("setoption name ObsCaptureSampleBits value 2");
    s.send("position startpos");
    s.send(NODES);
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(60));
    s.finish();

    let episodes = read_episodes(&path);
    assert_eq!(episodes.len(), 1, "one game -> one episode");
    let episode = &episodes[0];
    assert_eq!(episode.kind, EpisodeKind::InSearchCapture);
    assert!(episode.game_result.is_none());
    let records = &episode.records;
    assert!(!records.is_empty(), "the search captured no nodes");
    // Every record: full identity, empty path, at least one searched-move row.
    let mut total_rows = 0usize;
    for record in records {
        assert_ne!(
            record.position.canonical_key, 0,
            "canonical_key is identity"
        );
        assert!(
            record.position.path_from_episode_root.is_empty(),
            "InSearchCapture records carry an empty path"
        );
        assert!(
            !record.moves.is_empty(),
            "an interior captured node searched at least one move"
        );
        assert!(record.context.depth >= 4, "min_depth respected");
        total_rows += record.moves.len();
    }
    assert!(total_rows > 0, "no per-move rows captured");

    assert_obs_tool_verify(&path);
    std::fs::remove_file(&path).ok();
}

/// The per-game cap bounds the buffer and drops are COUNTED (no silent loss):
/// a tiny cap yields exactly `cap` records and a positive drop count.
#[test]
fn per_game_cap_is_respected_with_drop_counting() {
    let path = temp("cap");
    let mut s = Session::spawn();
    s.send(&format!(
        "setoption name ObsCapturePath value {}",
        path.display()
    ));
    s.send("setoption name ObsCaptureMinDepth value 2");
    s.send("setoption name ObsCaptureSampleBits value 0"); // capture every node
    s.send("setoption name ObsCapturePerGameCap value 5");
    s.send("position startpos");
    s.send(NODES);
    s.wait_for(|l| l.starts_with("bestmove "), Duration::from_secs(60));
    s.finish();

    let episodes = read_episodes(&path);
    assert_eq!(episodes.len(), 1);
    let episode = &episodes[0];
    assert_eq!(episode.records.len(), 5, "cap bounds the record count");
    assert!(
        episode.dropped_records > 0,
        "over-cap sampled nodes must be counted as drops"
    );
    assert_obs_tool_verify(&path);
    std::fs::remove_file(&path).ok();
}

/// `ucinewgame` is the game boundary: two games (with a `ucinewgame` between)
/// produce exactly two episodes, ids 0 and 1, the last flushed on engine drop.
#[test]
fn two_games_flush_two_episodes() {
    let path = temp("framing");
    let mut s = Session::spawn();
    s.send(&format!(
        "setoption name ObsCapturePath value {}",
        path.display()
    ));
    s.send("setoption name ObsCaptureMinDepth value 2");
    s.send("setoption name ObsCaptureSampleBits value 2");

    // Game 1.
    s.send("ucinewgame");
    s.send("position startpos");
    s.send("go nodes 30000");
    s.wait_for_bestmove_count(1, Duration::from_secs(60));

    // Game 2 (a different opening so the positions differ).
    s.send("ucinewgame");
    s.send("position startpos moves e2e4 c7c5");
    s.send("go nodes 30000");
    s.wait_for_bestmove_count(2, Duration::from_secs(60));
    s.finish();

    let episodes = read_episodes(&path);
    assert_eq!(episodes.len(), 2, "two games -> two episodes");
    assert_eq!(episodes[0].episode_id, 0);
    assert_eq!(episodes[1].episode_id, 1);
    assert_eq!(episodes[0].kind, EpisodeKind::InSearchCapture);
    assert_eq!(episodes[1].kind, EpisodeKind::InSearchCapture);
    assert!(!episodes[0].records.is_empty());
    assert!(!episodes[1].records.is_empty());
    assert_obs_tool_verify(&path);
    std::fs::remove_file(&path).ok();
}
