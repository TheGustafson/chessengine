//! `root-analysis` — the D43/D44 RootMoveAnalysis generator, root v0 (obs M3).
//!
//! For every input position and node-budget tier, drives a pinned UCI engine
//! over (a) one unrestricted search and (b) one independent `searchmoves`
//! arm per legal root move, and writes the results as RootMoveAnalysis
//! episodes into an Observation-Store shard (design/16 §3.5, doc 19 §7
//! Phase 1). D43's root-v0 authorization is exactly this shape: independent
//! root arms with cold state, before internal intervention instrumentation.
//!
//!   root-analysis --engine <bin> --fens <file> --out <shard.ceobs> \
//!     [--nodes 1000,4000,16000] [--uci-option Name=Value]... \
//!     [--seed N] [--created-unix N] [--arm-timeout-secs 60] \
//!     [--arm-mode searchmoves|child] [--reference-only]
//!
//! `--reference-only` (Tier-F): emit ONE Observation per tier — the position,
//! the unrestricted reference search's Layer-3 SearchObservations (final
//! score, bound, pv, completion, and the per-depth id trajectory), and
//! provenance — SKIPPING the per-legal-move arm loop. This is the "free
//! layers always" capture: a production relabel routed through obs gets the
//! eval trajectory + provenance for ~the cost of a scalar relabel, while the
//! scalar-cp corpus stays a cheap `obs-tool export-train` view (data-quality
//! strategy, Dimension 1). The full per-move arms (Tier-R, ~35× cost) are
//! reserved for high-value positions.
//!
//! Arm modes: `searchmoves` restricts the root search to one move (spec
//! shape; ARCHON supports it). `child` searches the position AFTER the move
//! at the same budget and NEGATES the score — fully independent child
//! searches for engines that silently ignore searchmoves (Reckless). The
//! mode is pinned in provenance.search_config; arms from different modes
//! are not comparable rows.
//!
//! Cold state: `ucinewgame` + `isready` before EVERY `go` (the per-arm reset
//! D43 requires; the M4 repeatability audit empirically confirms it). Engine
//! identity is pinned in provenance: path, `id name`, binary sha256 (via
//! sha256sum; recorded as unavailable if the tool is missing), options, and
//! the exact budget set. Score convention: centipawns as reported; `mate n`
//! maps to ±(31000 − n) so shorter mates order higher — a documented,
//! monotone encoding, not engine-internal Value parity.
//!
//! Episode layout: one episode per position (replay_root = the position);
//! one Observation per tier (context.node_budget = tier, window ±32000,
//! Layer 3 = the unrestricted search); one PerMoveObservation per legal move
//! (evidence = Reference, returned_score/nodes/pv-head from the arm,
//! completion = Complete or Stopped on timeout). The event_hash_chain folds
//! every arm result so replays can locate the first divergence.

use chessengine::board::Board;
use chessengine::movegen::generate_legal;
use chessengine::moves::{Move, MoveList};
use chessengine::obs::convert::position_from_board;
use chessengine::obs::{
    Bound, Completion, Episode, EpisodeKind, EvidenceKind, IdSnapshot, OBS_FORMAT_VERSION,
    Observation, PerMoveObservation, Provenance, SearchContext, SearchObservations, ShardHeader,
    ShardWriter,
};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, ExitCode, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

const SCORE_WINDOW: i32 = 32_000;
const MATE_BASE: i32 = 31_000;

#[derive(Clone, Copy, PartialEq)]
enum ArmMode {
    /// `go nodes N searchmoves <m>` — requires engine support (ARCHON has it).
    Searchmoves,
    /// Search the position AFTER the move and negate: fully independent
    /// child searches, for engines that ignore searchmoves (Reckless does —
    /// found by the M4 smoke: all arms returned the unrestricted result).
    Child,
}

struct Config {
    engine: String,
    fens: String,
    out: String,
    nodes: Vec<u64>,
    options: Vec<(String, String)>,
    seed: u64,
    created_unix: Option<u64>,
    arm_timeout: Duration,
    arm_mode: ArmMode,
    /// Tier-F (`--reference-only`): emit ONLY the unrestricted reference
    /// search per tier (position + Layer-3 SearchObservations incl. the id
    /// trajectory + provenance), SKIPPING the ~35× per-legal-move arm loop.
    /// This is the "free layers always" capture (data-quality strategy
    /// Dimension 1): a production relabel routed through obs gets the eval
    /// trajectory + provenance for ~the cost of a scalar relabel, and the
    /// scalar-cp corpus stays a cheap `obs-tool export-train` view.
    reference_only: bool,
    /// Warm mode (`--warm`): do NOT `ucinewgame` between positions, so the
    /// transposition table persists across consecutive FENs. When the input is
    /// GAME-ORDERED (position N+1 is one move off N), their search trees overlap
    /// and reuse work — ~1.5-2x throughput at a reduced node budget, or better
    /// labels at the same budget. TRADE-OFF: labels become history-dependent (a
    /// function of the arrival path, not the position alone), so only use with
    /// game-ordered input where that is acceptable (decision 2026-07-19). Cold
    /// (default) stays independent/reproducible.
    warm: bool,
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        engine: String::new(),
        fens: String::new(),
        out: String::new(),
        nodes: vec![1_000, 4_000, 16_000],
        options: Vec::new(),
        seed: 0,
        created_unix: None,
        arm_timeout: Duration::from_secs(60),
        arm_mode: ArmMode::Searchmoves,
        reference_only: false,
        warm: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match flag.as_str() {
            "--engine" => config.engine = value("--engine")?,
            "--fens" => config.fens = value("--fens")?,
            "--out" => config.out = value("--out")?,
            "--nodes" => {
                config.nodes = value("--nodes")?
                    .split(',')
                    .map(|n| {
                        n.trim()
                            .parse()
                            .map_err(|_| format!("bad node count '{n}'"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--uci-option" => {
                let pair = value("--uci-option")?;
                let (name, val) = pair
                    .split_once('=')
                    .ok_or(format!("--uci-option wants Name=Value, got '{pair}'"))?;
                config.options.push((name.into(), val.into()));
            }
            "--seed" => config.seed = value("--seed")?.parse().map_err(|_| "bad --seed")?,
            "--created-unix" => {
                config.created_unix = Some(
                    value("--created-unix")?
                        .parse()
                        .map_err(|_| "bad --created-unix")?,
                );
            }
            "--arm-mode" => {
                config.arm_mode = match value("--arm-mode")?.as_str() {
                    "searchmoves" => ArmMode::Searchmoves,
                    "child" => ArmMode::Child,
                    other => return Err(format!("--arm-mode searchmoves|child, got '{other}'")),
                };
            }
            "--arm-timeout-secs" => {
                config.arm_timeout = Duration::from_secs(
                    value("--arm-timeout-secs")?
                        .parse()
                        .map_err(|_| "bad timeout")?,
                );
            }
            "--reference-only" => config.reference_only = true,
            "--warm" => config.warm = true,
            other => return Err(format!("unknown flag '{other}'")),
        }
    }
    if config.engine.is_empty() || config.fens.is_empty() || config.out.is_empty() {
        return Err("required: --engine <bin> --fens <file> --out <shard>".into());
    }
    if config.nodes.is_empty() {
        return Err("--nodes must list at least one tier".into());
    }
    Ok(config)
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("root-analysis: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run(&config) {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("root-analysis: {error}");
            ExitCode::FAILURE
        }
    }
}

// ─── UCI engine driver ─────────────────────────────────────────────────────

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    id_name: String,
    /// When true, skip the per-position `ucinewgame` so the TT persists
    /// (warm mode; see Config::warm).
    warm: bool,
}

/// One completed search: the last-seen info fields at bestmove time, plus
/// (when requested) the per-depth id trajectory built from the info lines.
#[derive(Default, Clone)]
struct SearchResult {
    score: i32,
    depth: i32,
    seldepth: i32,
    nodes: u64,
    time_ms: u64,
    pv: Vec<u16>,
    best: u16,
    timed_out: bool,
    /// One entry per iterative-deepening depth (last summary line wins for a
    /// depth that re-searched under aspiration). Empty unless the caller asked
    /// `search(.., record_trajectory=true, ..)` — only the unrestricted
    /// reference search records it; per-move arms stay point observations.
    id_snapshots: Vec<IdSnapshot>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Any early-return `?` path must not orphan a still-searching child
        // (D33 round 1). kill/wait on an already-exited child just errors,
        // which is fine; the reader thread ends when stdout closes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Engine {
    fn spawn(config: &Config) -> Result<Self, String> {
        let mut child = Command::new(&config.engine)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawning {}: {e}", config.engine))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        // Reader thread: blocking line reads become channel messages so every
        // wait below can carry the arm timeout.
        let (sender, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let mut engine = Engine {
            child,
            stdin,
            lines,
            id_name: String::new(),
            warm: config.warm,
        };
        engine.send("uci")?;
        let deadline = Duration::from_secs(10);
        loop {
            let line = engine
                .recv(deadline)?
                .ok_or("engine died during uci handshake")?;
            if let Some(name) = line.strip_prefix("id name ") {
                engine.id_name = name.trim().to_string();
            }
            if line.trim() == "uciok" {
                break;
            }
        }
        for (name, value) in &config.options {
            engine.send(&format!("setoption name {name} value {value}"))?;
        }
        engine.sync(deadline)?;
        Ok(engine)
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.stdin, "{line}").map_err(|e| format!("engine stdin: {e}"))
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<String>, String> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Ok(Some(line)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err("engine stdout closed".into()),
        }
    }

    fn sync(&mut self, timeout: Duration) -> Result<(), String> {
        self.send("isready")?;
        loop {
            match self.recv(timeout)? {
                Some(line) if line.trim() == "readyok" => return Ok(()),
                Some(_) => continue,
                None => return Err("engine unresponsive to isready".into()),
            }
        }
    }

    /// Cold-state search: `ucinewgame` + sync, then `go nodes N` (optionally
    /// restricted to one root move). On timeout the result is marked and the
    /// caller must discard this engine (its search is still running).
    fn search(
        &mut self,
        fen: &str,
        nodes: u64,
        searchmove: Option<&str>,
        record_trajectory: bool,
        timeout: Duration,
    ) -> Result<SearchResult, String> {
        // Cold (default): reset the TT before every position so each label is an
        // independent function of the position. Warm (`--warm`): skip the reset so
        // the TT carries over from the previous (game-ordered) position and its
        // overlapping tree reuses work — faster / better, but history-dependent.
        if !self.warm {
            self.send("ucinewgame")?;
            self.sync(timeout)?;
        }
        self.send(&format!("position fen {fen}"))?;
        match searchmove {
            Some(mv) => self.send(&format!("go nodes {nodes} searchmoves {mv}"))?,
            None => self.send(&format!("go nodes {nodes}"))?,
        }
        let mut result = SearchResult::default();
        loop {
            let Some(line) = self.recv(timeout)? else {
                result.timed_out = true;
                return Ok(result);
            };
            let tokens: Vec<&str> = line.split_whitespace().collect();
            match tokens.first() {
                Some(&"info") => {
                    // A summary info line (one carrying a `pv`) is the depth's
                    // final word; fold it into the trajectory, overwriting an
                    // earlier line for the same depth (aspiration re-search).
                    if parse_info(&tokens, &mut result) && record_trajectory && result.depth > 0 {
                        let snap = IdSnapshot {
                            depth: result.depth,
                            score: result.score,
                            best: result.pv.first().copied().unwrap_or(0),
                            nodes: result.nodes,
                            time_ms: result.time_ms,
                            pv: result.pv.clone(),
                        };
                        match result.id_snapshots.last_mut() {
                            Some(last) if last.depth == snap.depth => *last = snap,
                            _ => result.id_snapshots.push(snap),
                        }
                    }
                }
                Some(&"bestmove") => {
                    if let Some(best) = tokens.get(1) {
                        result.best = parse_uci_move(best).unwrap_or(0);
                    }
                    return Ok(result);
                }
                _ => {}
            }
        }
    }

    fn quit(mut self) {
        let _ = self.send("quit");
        // Drop reaps; sending quit first lets the engine exit cleanly.
    }

    fn kill(self) {
        // Drop kills + reaps.
    }
}

/// Fold one `info` line into `result`. Returns true when the line carried a
/// `pv` — the marker of a depth-summary line, used by the caller to snapshot
/// the id trajectory.
fn parse_info(tokens: &[&str], result: &mut SearchResult) -> bool {
    let mut had_pv = false;
    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if let Some(d) = tokens.get(i + 1).and_then(|t| t.parse().ok()) {
                    result.depth = d;
                }
                i += 2;
            }
            "seldepth" => {
                if let Some(d) = tokens.get(i + 1).and_then(|t| t.parse().ok()) {
                    result.seldepth = d;
                }
                i += 2;
            }
            "nodes" => {
                if let Some(n) = tokens.get(i + 1).and_then(|t| t.parse().ok()) {
                    result.nodes = n;
                }
                i += 2;
            }
            "time" => {
                if let Some(t) = tokens.get(i + 1).and_then(|t| t.parse().ok()) {
                    result.time_ms = t;
                }
                i += 2;
            }
            "score" => {
                match (
                    tokens.get(i + 1),
                    tokens.get(i + 2).and_then(|t| t.parse::<i32>().ok()),
                ) {
                    (Some(&"cp"), Some(value)) => result.score = value,
                    (Some(&"mate"), Some(moves)) => {
                        result.score = if moves >= 0 {
                            MATE_BASE - moves
                        } else {
                            -MATE_BASE - moves
                        };
                    }
                    _ => {}
                }
                i += 3;
            }
            "pv" => {
                result.pv = tokens[i + 1..]
                    .iter()
                    .map_while(|t| parse_uci_move(t))
                    .collect();
                had_pv = true;
                break; // pv is last on the line
            }
            _ => i += 1,
        }
    }
    had_pv
}

/// UCI move text → our packed u16, via the position-independent subset
/// (from/to/promotion). Castling arrives as the king's two-square move in
/// UCI, which matches our encoding's from/to; the KIND bits differ from a
/// real `Move`, so these u16s are for RECORDING only — the store's replay
/// path uses legal-move matching by from/to/promo, not raw equality.
fn parse_uci_move(text: &str) -> Option<u16> {
    let bytes = text.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    let sq = |file: u8, rank: u8| -> Option<u16> {
        if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
            return None;
        }
        Some(u16::from(rank - b'1') * 8 + u16::from(file - b'a'))
    };
    let from = sq(bytes[0], bytes[1])?;
    let to = sq(bytes[2], bytes[3])?;
    let promo = match bytes.get(4).map(u8::to_ascii_lowercase) {
        Some(b'n') => 0u16,
        Some(b'b') => 1,
        Some(b'r') => 2,
        Some(b'q') => 3,
        Some(_) => return None,
        None => 0,
    };
    let kind = if bytes.len() >= 5 { 1u16 } else { 0 }; // promotion kind bit
    Some((kind << 14) | (promo << 12) | (to << 6) | from)
}

// ─── Provenance helpers ────────────────────────────────────────────────────

fn sha256_of(path: &str) -> String {
    Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8(o.stdout)
                .ok()?
                .split_whitespace()
                .next()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "sha256-unavailable".into())
}

fn hardware_identity(engine_threads: u32) -> String {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1).map(|v| v.trim().to_string()))
        })
        .unwrap_or_else(|| "unknown-cpu".into());
    format!("{model}, {engine_threads} engine thread(s)")
}

fn fnv1a64(seed: u64, data: &[u8]) -> u64 {
    let mut hash = if seed == 0 {
        0xcbf2_9ce4_8422_2325
    } else {
        seed
    };
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ─── Main run ──────────────────────────────────────────────────────────────

/// Match an engine-reported move (positional from/to/promo only) to the
/// canonical legal `Move`'s raw16, so Layer-3 best_move and Layer-4 arm ids
/// share one encoding even for castling/en-passant.
fn canonicalize(reported: u16, legal: &MoveList) -> u16 {
    const POSITIONAL: u16 = 0x3fff; // from|to|promo bits; kind bits excluded
    legal
        .iter()
        .find(|m| {
            let raw = m.raw16();
            (raw & 0x0fff) == (reported & 0x0fff)
                && (raw >> 14 != 1 || (raw & POSITIONAL) == (reported & POSITIONAL))
        })
        .map(|m| m.raw16())
        .unwrap_or(reported)
}

fn run(config: &Config) -> Result<String, String> {
    let fens: Vec<String> = std::fs::read_to_string(&config.fens)
        .map_err(|e| format!("reading {}: {e}", config.fens))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if fens.is_empty() {
        return Err("no positions in the fens file".into());
    }

    let mut engine = Engine::spawn(config)?;
    let options_text = config
        .options
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    let nodes_text = config
        .nodes
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mode_text = match config.arm_mode {
        ArmMode::Searchmoves => "searchmoves",
        ArmMode::Child => "child-negate",
    };
    let search_config = format!(
        "cold ucinewgame per go; arm-mode {mode_text}; nodes tiers {nodes_text}; \
         options [{options_text}]"
    );
    // The searching engine's net identity (D33 round 1): an ARCHON arm
    // selects its net via `--uci-option EvalFile=…`, which must surface in
    // the §3.4 net_version field, not only inside the config string.
    let net_version = config
        .options
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("EvalFile"))
        .map(|(_, value)| value.clone());
    let engine_threads: u32 = config
        .options
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("Threads"))
        .and_then(|(_, value)| value.parse().ok())
        .unwrap_or(1);
    let provenance = Provenance {
        engine_version: format!("root-analysis v0 (obs M3) @ {}", env!("CARGO_PKG_VERSION")),
        engine_bench_sig: 0,
        net_version,
        // For RootMoveAnalysis shards teacher_identity IS D44's
        // producer_engine: single --engine per shard, resolved per
        // observation via Episode.provenance_ref (interned per §3.4).
        teacher_identity: Some(format!(
            "{} | uci id '{}' | sha256:{}",
            config.engine,
            engine.id_name,
            sha256_of(&config.engine)
        )),
        search_config_hash: fnv1a64(0, search_config.as_bytes()),
        search_config,
        generator_version: "root-analysis v0".into(),
        rng_seed: config.seed,
        dataset_source: "root_move_analysis".into(),
        opening_source: config.fens.clone(),
        hardware_identity: hardware_identity(engine_threads),
    };
    let header = ShardHeader {
        format_version: OBS_FORMAT_VERSION,
        created_unix: config.created_unix.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        }),
        provenance: vec![provenance],
    };
    let mut writer = ShardWriter::create(&config.out, &header)
        .map_err(|e| format!("creating {}: {e}", config.out))?;

    let (mut episodes, mut arms_total, mut timeouts) = (0u64, 0u64, 0u64);
    for (index, fen) in fens.iter().enumerate() {
        let board = Board::from_fen(fen).map_err(|e| format!("fens line {}: {e}", index + 1))?;
        let mut legal = MoveList::new();
        generate_legal(&board, &mut legal);
        if legal.is_empty() {
            eprintln!("skipping terminal position at line {}", index + 1);
            continue;
        }
        let canonical_fen = board.to_fen();
        let root_state = position_from_board(&board, 0, vec![]);
        let mut records = Vec::with_capacity(config.nodes.len());
        let mut hash_chain = Vec::new();

        for &tier in &config.nodes {
            // Unrestricted reference search for this tier (records the id
            // trajectory — the free Tier-F layer that rides along with any
            // relabel routed through obs).
            let full = engine.search(&canonical_fen, tier, None, true, config.arm_timeout)?;
            if full.timed_out {
                timeouts += 1;
                engine = respawn(engine, config)?;
            }

            // Tier-F (`--reference-only`) captures ONLY the reference search;
            // an empty arm set skips the ~35× per-legal-move cost.
            let arm_moves: &[Move] = if config.reference_only { &[] } else { &legal };
            let mut moves_out = Vec::with_capacity(arm_moves.len());
            for mv in arm_moves.iter() {
                let text = mv.to_string();
                let arm = match config.arm_mode {
                    ArmMode::Searchmoves => {
                        engine.search(&canonical_fen, tier, Some(&text), false, config.arm_timeout)?
                    }
                    ArmMode::Child => {
                        let mut child = board.clone();
                        let gives_check = child.gives_check(*mv);
                        child.make_move(*mv, gives_check);
                        if let Some(score) = terminal_child_score(&child) {
                            SearchResult {
                                score,
                                ..Default::default()
                            }
                        } else {
                            let mut result = engine.search(
                                &child.to_fen(),
                                tier,
                                None,
                                false,
                                config.arm_timeout,
                            )?;
                            // The child search scores from the OPPONENT's
                            // side; the ±(MATE_BASE − n) encoding negates
                            // symmetrically (and `mate 0` cannot occur on a
                            // non-terminal child).
                            result.score = -result.score;
                            result
                        }
                    }
                };
                arms_total += 1;
                if arm.timed_out {
                    timeouts += 1;
                    engine = respawn(engine, config)?;
                }
                let mut chained = fnv1a64(*hash_chain.last().unwrap_or(&0), text.as_bytes());
                chained = fnv1a64(chained, &arm.score.to_le_bytes());
                chained = fnv1a64(chained, &arm.nodes.to_le_bytes());
                hash_chain.push(chained);
                moves_out.push(PerMoveObservation {
                    mv: mv.raw16(),
                    order_index: moves_out.len() as u16,
                    returned_score: arm.score,
                    nodes: arm.nodes,
                    searched_depth: arm.depth,
                    evidence: EvidenceKind::Reference,
                    completion: if arm.timed_out {
                        Completion::Stopped
                    } else {
                        Completion::Complete
                    },
                    ..Default::default()
                });
            }

            records.push(Observation {
                position: root_state.clone(),
                context: SearchContext {
                    depth: full.depth,
                    node_budget: Some(tier),
                    alpha: -SCORE_WINDOW,
                    beta: SCORE_WINDOW,
                    alpha_before_move: -SCORE_WINDOW,
                    beta_before_move: SCORE_WINDOW,
                    legal_move_count: legal.len() as u8,
                    capture_move_count: legal.iter().filter(|&&m| board.is_capture(m)).count()
                        as u8,
                    ..Default::default()
                },
                search: SearchObservations {
                    best_move: canonicalize(full.best, &legal),
                    pv: full.pv.clone(),
                    final_score: full.score,
                    bound: Bound::Exact,
                    nodes: full.nodes,
                    seldepth: full.seldepth,
                    id_snapshots: full.id_snapshots.clone(),
                    completion: if full.timed_out {
                        Completion::Stopped
                    } else {
                        Completion::Complete
                    },
                },
                moves: moves_out,
            });
        }

        let episode = Episode {
            kind: EpisodeKind::RootMoveAnalysis,
            episode_id: index as u64,
            provenance_ref: 0,
            game_result: None,
            cf_axis: None,
            records,
            replay_root: root_state,
            root_reversible_keys: vec![],
            event_hash_chain: hash_chain,
            dropped_records: 0,
        };
        writer
            .append(&episode)
            .map_err(|e| format!("appending episode {index}: {e}"))?;
        episodes += 1;
    }

    writer
        .finish()
        .map_err(|e| format!("finishing shard: {e}"))?;
    engine.quit();
    let mode = if config.reference_only {
        "reference-only"
    } else {
        "full-arms"
    };
    Ok(format!(
        "root-analysis: {mode}; {episodes} episode(s), {arms_total} arm(s), \
         {timeouts} timeout(s) -> {}",
        config.out
    ))
}

/// The arm score when a child position is TERMINAL, from the ARM-MOVER's
/// perspective: a mate-delivering move is a win (+MATE_BASE), a stalemating
/// move a draw (0). Asking an engine instead would return `mate 0` —
/// "the side to move IS mated" — which the win-positive score parse would
/// invert into the worst possible arm label (D33 round 1, HIGH).
fn terminal_child_score(child: &Board) -> Option<i32> {
    let mut legal = MoveList::new();
    generate_legal(child, &mut legal);
    if !legal.is_empty() {
        return None;
    }
    Some(if child.checkers().is_empty() {
        0
    } else {
        MATE_BASE
    })
}

fn respawn(engine: Engine, config: &Config) -> Result<Engine, String> {
    engine.kill();
    Engine::spawn(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chessengine::types::Square;

    /// The D33 round-1 HIGH repro: Rb8# from 6k1/5ppp/.../1R4K1. The child
    /// after Rb8 is checkmate; the arm must score +MATE_BASE (pre-fix the
    /// engine's `mate 0` reply was negated into -MATE_BASE, ranking the
    /// mating move dead last).
    #[test]
    fn mate_delivering_child_arm_scores_as_a_win() {
        let board = Board::from_fen("6k1/5ppp/8/8/8/8/8/1R4K1 w - - 0 1").unwrap();
        let mut legal = MoveList::new();
        generate_legal(&board, &mut legal);
        let rb8 = *legal
            .iter()
            .find(|m| m.from() == Square::B1 && m.to() == Square::B8)
            .expect("Rb8 legal");
        let mut child = board.clone();
        let gives_check = child.gives_check(rb8);
        child.make_move(rb8, gives_check);
        assert_eq!(terminal_child_score(&child), Some(MATE_BASE));
    }

    #[test]
    fn stalemating_child_arm_scores_as_a_draw() {
        // Qf3xf7 stalemates the bare king (the qsearch stalemate fixture).
        let board = Board::from_fen("7k/5r2/8/8/8/5Q2/8/1K6 w - - 0 1").unwrap();
        let mut legal = MoveList::new();
        generate_legal(&board, &mut legal);
        let qxf7 = *legal
            .iter()
            .find(|m| m.from() == Square::F3 && m.to() == Square::F7)
            .expect("Qxf7 legal");
        let mut child = board.clone();
        let gives_check = child.gives_check(qxf7);
        child.make_move(qxf7, gives_check);
        assert_eq!(terminal_child_score(&child), Some(0));
    }

    #[test]
    fn non_terminal_child_defers_to_the_engine() {
        let mut child = Board::startpos();
        let mut legal = MoveList::new();
        generate_legal(&child, &mut legal);
        let e4 = *legal
            .iter()
            .find(|m| m.from() == Square::E2 && m.to() == Square::E4)
            .unwrap();
        let gives_check = child.gives_check(e4);
        child.make_move(e4, gives_check);
        assert_eq!(terminal_child_score(&child), None);
    }

    #[test]
    fn mate_scores_parse_sign_correctly_and_case_insensitive_promotions() {
        let mut r = SearchResult::default();
        parse_info(&["info", "score", "mate", "3"], &mut r);
        assert_eq!(r.score, MATE_BASE - 3);
        parse_info(&["info", "score", "mate", "-2"], &mut r);
        assert_eq!(r.score, -MATE_BASE + 2);
        assert_eq!(parse_uci_move("e7e8q"), parse_uci_move("e7e8Q"));
        assert!(parse_uci_move("(none)").is_none());
    }

    #[test]
    fn parse_info_reports_pv_lines_and_parses_time_for_the_trajectory() {
        let mut r = SearchResult::default();
        // A depth-summary line (carries a pv) → had_pv true, time captured.
        let had = parse_info(
            &[
                "info", "depth", "7", "seldepth", "9", "score", "cp", "34", "nodes", "12000",
                "time", "42", "pv", "e2e4", "e7e5",
            ],
            &mut r,
        );
        assert!(had, "a line with a pv must report had_pv");
        assert_eq!(r.depth, 7);
        assert_eq!(r.time_ms, 42);
        assert_eq!(r.nodes, 12000);
        assert_eq!(r.pv.len(), 2);
        // A currmove line (no pv) → not a snapshot boundary.
        let had2 = parse_info(
            &["info", "depth", "7", "currmove", "d2d4", "currmovenumber", "2"],
            &mut r,
        );
        assert!(!had2, "a currmove line has no pv and must not snapshot");
    }
}
