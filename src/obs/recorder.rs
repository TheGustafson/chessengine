//! In-search observation capture (design/16 §3.5.5, D45a).
//!
//! The engine-side recorder: a per-`Worker` buffer that samples interior
//! main-search nodes DURING real games and flushes one `InSearchCapture`
//! episode per game into an Observation-Store shard. It is observation-only —
//! the sample gate is a pure function of the position key, so capture never
//! consumes a node, changes timing, or perturbs any search decision (identical
//! node counts / bestmove / score with capture on or off). The whole path is
//! behind `feature = "datagen"`, so the tournament/bench binary compiles it out
//! and the D24 default-build fingerprint is untouched (§6 bench neutrality).
//!
//! Lifecycle: created on the first `go` after `ObsCapturePath` is set (main
//! worker, `Threads == 1`); the buffer accumulates across a game's `go`s;
//! `flush_game` writes one episode on `ucinewgame` and on engine drop/quit.

use super::convert::position_from_board;
use super::schema::{
    Bound, Completion, Episode, EpisodeKind, EvidenceKind, NodeType, OBS_FORMAT_VERSION,
    Observation, PerMoveObservation, Provenance, SearchContext, SearchObservations, ShardHeader,
};
use super::shard::ShardWriter;
use crate::board::Board;
use crate::movegen::generate_legal;
use crate::moves::MoveList;
use crate::types::{Depth, Key, Value};
use std::io;
use std::path::PathBuf;

/// The scalar config the recorder is built from (assembled by the worker from
/// its UCI options snapshot so the `obs` module stays free of a `uci`
/// dependency).
pub struct ObsRecorderConfig {
    pub path: PathBuf,
    pub min_depth: Depth,
    pub sample_bits: u32,
    pub per_game_cap: usize,
    pub engine_version: String,
    pub search_config: String,
    pub net_version: Option<String>,
    pub hardware_identity: String,
}

/// Per-node scratch, built at the node's move-loop entry and consumed at node
/// exit. Snapshots the fields that mutate during the loop (depth via singular
/// extension, alpha via move updates) so the recorded context is the ENTRY
/// window, and accumulates one Layer-4 row per move actually searched.
pub struct ObsNodeCapture {
    entry_nodes: u64,
    depth: Depth,
    alpha: Value,
    beta: Value,
    moves: Vec<PerMoveObservation>,
}

impl ObsNodeCapture {
    pub fn new(entry_nodes: u64, depth: Depth, alpha: Value, beta: Value) -> ObsNodeCapture {
        ObsNodeCapture {
            entry_nodes,
            depth,
            alpha,
            beta,
            moves: Vec::new(),
        }
    }

    /// Append one Layer-4 row for a move actually searched at this node.
    #[inline]
    pub fn push_move(&mut self, row: PerMoveObservation) {
        self.moves.push(row);
    }

    /// Whether any move was actually searched. A sampled node that searched
    /// none (terminal, or every move pruned before make) carries no v0 Layer-4
    /// data and is not recorded.
    #[inline]
    pub fn searched_any(&self) -> bool {
        !self.moves.is_empty()
    }
}

/// Everything the recorder needs to reconstruct Layers 2/3 at node exit that it
/// cannot read from the board alone. Assembled at the call site (which knows the
/// `NodeType` const-generic and the loop-local outcome scalars).
pub struct NodeExit {
    pub best_value: Value,
    pub best_move: u16,
    pub bound: Bound,
    pub node_type: NodeType,
    pub static_eval: Value,
    pub ply: u16,
    pub tt_hit: bool,
    pub tt_depth: Depth,
    pub tt_bound: Bound,
    pub tt_value: Value,
    pub tt_move: u16,
    pub correction_value: Value,
    pub node_budget: Option<u64>,
}

/// The per-`Worker` in-search recorder. Owns the shard writer, the bounded
/// per-game buffer, and the running episode/drop counters.
pub struct ObsRecorder {
    min_depth: Depth,
    /// `(1 << sample_bits) - 1`; a node is sampled iff `raw_key & mask == 0`.
    sample_mask: u64,
    per_game_cap: usize,
    writer: ShardWriter,
    /// Fixed-capacity buffer, reserved once to `per_game_cap` and reused across
    /// games (`flush_game` clears but keeps capacity).
    records: Vec<Observation>,
    /// Sampled records this game the cap refused (no silent loss).
    dropped: u64,
    next_episode_id: u64,
}

impl ObsRecorder {
    /// Open the shard and write its header (provenance). Fails loudly if the
    /// path is unwritable or provenance is incomplete (the writer enforces
    /// §3.4). Sample bits are clamped to a sane `0..=63`.
    pub fn enable(config: ObsRecorderConfig) -> io::Result<ObsRecorder> {
        if let Some(parent) = config.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let sample_bits = config.sample_bits.min(63);
        let sample_mask = (1u64 << sample_bits) - 1;
        let search_config_hash = fnv1a64(config.search_config.as_bytes());
        let provenance = Provenance {
            engine_version: config.engine_version,
            engine_bench_sig: 0, // a datagen+net binary is not the pinned fingerprint
            net_version: config.net_version,
            teacher_identity: None,
            search_config: config.search_config,
            search_config_hash,
            generator_version: "in-search-capture v0".into(),
            rng_seed: 0, // capture is RNG-free (position-keyed sampling)
            dataset_source: "in_search_capture".into(),
            opening_source: "uci-in-game".into(),
            hardware_identity: config.hardware_identity,
        };
        let header = ShardHeader {
            format_version: OBS_FORMAT_VERSION,
            created_unix: now_unix(),
            provenance: vec![provenance],
        };
        let writer = ShardWriter::create(&config.path, &header)?;
        let per_game_cap = config.per_game_cap;
        Ok(ObsRecorder {
            min_depth: config.min_depth,
            sample_mask,
            per_game_cap,
            writer,
            records: Vec::with_capacity(per_game_cap.min(1 << 20)),
            dropped: 0,
            next_episode_id: 0,
        })
    }

    #[inline]
    pub fn min_depth(&self) -> Depth {
        self.min_depth
    }

    /// Deterministic, position-keyed sample gate (no RNG, no timing).
    #[inline]
    pub fn sample_hit(&self, raw_key: Key) -> bool {
        raw_key & self.sample_mask == 0
    }

    /// Whether the per-game buffer still has room for another record.
    #[inline]
    pub fn has_room(&self) -> bool {
        self.records.len() < self.per_game_cap
    }

    /// A sampled node the cap refused before collection began.
    #[inline]
    pub fn note_dropped(&mut self) {
        self.dropped += 1;
    }

    /// Push a completed node observation, respecting the per-game cap. A
    /// refused record is counted, never silently lost.
    #[inline]
    pub fn try_push(&mut self, observation: Observation) {
        if self.records.len() < self.per_game_cap {
            self.records.push(observation);
        } else {
            self.dropped += 1;
        }
    }

    /// Flush the current game's buffer as one `InSearchCapture` episode, then
    /// reset for the next game. An empty buffer flushes nothing (so N games map
    /// to N episodes). Each flushed episode is made crash-durable.
    pub fn flush_game(&mut self) -> io::Result<()> {
        if self.records.is_empty() {
            // Nothing searched-and-sampled this game; do not emit an empty
            // episode. A leftover drop count with no records is impossible
            // (drops require a full buffer), but reset defensively anyway.
            self.dropped = 0;
            return Ok(());
        }
        let episode = Episode {
            kind: EpisodeKind::InSearchCapture,
            episode_id: self.next_episode_id,
            provenance_ref: 0,
            game_result: None,
            cf_axis: None,
            records: std::mem::take(&mut self.records),
            replay_root: Default::default(),
            root_reversible_keys: Vec::new(),
            event_hash_chain: Vec::new(),
            dropped_records: self.dropped,
        };
        self.writer.append(&episode)?;
        self.writer.flush()?;
        // Reuse the buffer's capacity for the next game (§3.5.5 fixed-capacity).
        self.records = episode.records;
        self.records.clear();
        self.dropped = 0;
        self.next_episode_id += 1;
        Ok(())
    }
}

impl Drop for ObsRecorder {
    fn drop(&mut self) {
        // The final game (no trailing `ucinewgame` before `quit`) flushes here.
        if let Err(error) = self.flush_game() {
            eprintln!("ARCHON obs capture: final flush failed: {error}");
        }
    }
}

/// Build the Layer-1/2/3 wrapper for a captured node and hand it to the
/// recorder. `board` MUST be at the node's own position (every made move undone)
/// — true at the search node-exit call site. Legal/capture counts come from a
/// legal-move generation done only here, on the rare sampled node.
pub fn finish_node(
    board: &Board,
    total_nodes: u64,
    recorder: &mut ObsRecorder,
    cap: ObsNodeCapture,
    exit: NodeExit,
) {
    let mut legal = MoveList::new();
    generate_legal(board, &mut legal);
    let capture_count = legal.iter().filter(|&&m| board.is_capture(m)).count();
    let observation = Observation {
        position: position_from_board(board, 0, Vec::new()),
        context: SearchContext {
            depth: cap.depth,
            node_budget: exit.node_budget,
            alpha: cap.alpha,
            beta: cap.beta,
            node_type: exit.node_type,
            static_eval: exit.static_eval,
            tt_hit: exit.tt_hit,
            tt_depth: exit.tt_depth,
            tt_bound: exit.tt_bound,
            tt_value: exit.tt_value,
            tt_move: exit.tt_move,
            ply: exit.ply,
            alpha_before_move: cap.alpha,
            beta_before_move: cap.beta,
            correction_value: exit.correction_value,
            legal_move_count: legal.len().min(u8::MAX as usize) as u8,
            capture_move_count: capture_count.min(u8::MAX as usize) as u8,
            ..Default::default()
        },
        search: SearchObservations {
            best_move: exit.best_move,
            final_score: exit.best_value,
            bound: exit.bound,
            nodes: total_nodes.saturating_sub(cap.entry_nodes),
            completion: Completion::Complete,
            ..Default::default()
        },
        moves: cap.moves,
    };
    recorder.try_push(observation);
}

/// One-move-row helper so the search call site stays declarative. `Prune::None`
/// + `Observational` on every v0 row (only searched moves are rowed).
#[allow(clippy::too_many_arguments)]
pub fn move_row(
    mv: u16,
    order_index: u16,
    searched_depth: Depth,
    reduction_plies: i32,
    extension: Depth,
    returned_score: Value,
    nodes: u64,
    raised_alpha: bool,
    caused_cutoff: bool,
    researched: bool,
) -> PerMoveObservation {
    PerMoveObservation {
        mv,
        order_index,
        searched_depth,
        reduction: reduction_plies.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        extension: extension.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        returned_score,
        nodes,
        raised_alpha,
        caused_cutoff,
        researched,
        evidence: EvidenceKind::Observational,
        completion: Completion::Complete,
        ..Default::default()
    }
}

/// A short CPU/thread identity for provenance (machine-relative labels demand
/// it, doc 19 §5.2). Best-effort: `unknown-cpu` if `/proc/cpuinfo` is absent.
pub fn hardware_identity() -> String {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1).map(|v| v.trim().to_string()))
        })
        .unwrap_or_else(|| "unknown-cpu".into());
    format!("{model}, 1 engine thread (in-search capture)")
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(path: PathBuf) -> ObsRecorderConfig {
        ObsRecorderConfig {
            path,
            min_depth: 6,
            sample_bits: 4,
            per_game_cap: 3,
            engine_version: "archon test".into(),
            search_config: "Threads=1 Hash=16".into(),
            net_version: None,
            hardware_identity: "test-cpu".into(),
        }
    }

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "archon-obs-rec-{name}-{}.ceobs",
            std::process::id()
        ))
    }

    #[test]
    fn sample_gate_is_pure_key_function() {
        let rec = ObsRecorder::enable(config(temp("gate"))).unwrap();
        // sample_bits 4 => mask 15 => keys divisible by 16 hit.
        assert!(rec.sample_hit(0));
        assert!(rec.sample_hit(16));
        assert!(rec.sample_hit(0xFFFF_FFFF_FFFF_FFF0));
        assert!(!rec.sample_hit(1));
        assert!(!rec.sample_hit(15));
        std::fs::remove_file(temp("gate")).ok();
    }

    #[test]
    fn per_game_cap_bounds_buffer_and_counts_drops() {
        let path = temp("cap");
        let mut rec = ObsRecorder::enable(config(path.clone())).unwrap();
        // cap = 3: push 5, expect 3 kept + 2 dropped, then one episode.
        for _ in 0..5 {
            rec.try_push(Observation::default());
        }
        assert_eq!(rec.records.len(), 3);
        assert_eq!(rec.dropped, 2);
        rec.flush_game().unwrap();
        assert_eq!(rec.records.len(), 0);
        assert_eq!(rec.dropped, 0);
        drop(rec);

        // The written episode carries the drop count and exactly 3 records.
        let mut reader = super::super::ShardReader::open(&path).unwrap();
        let episode = reader.next_episode().unwrap().unwrap();
        assert_eq!(episode.kind, EpisodeKind::InSearchCapture);
        assert_eq!(episode.records.len(), 3);
        assert_eq!(episode.dropped_records, 2);
        assert!(reader.next_episode().unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_game_flushes_nothing_two_games_two_episodes() {
        let path = temp("framing");
        let mut rec = ObsRecorder::enable(config(path.clone())).unwrap();
        // Game 0: empty -> no episode.
        rec.flush_game().unwrap();
        // Game 1: one record.
        rec.try_push(Observation::default());
        rec.flush_game().unwrap();
        // Game 2: two records.
        rec.try_push(Observation::default());
        rec.try_push(Observation::default());
        rec.flush_game().unwrap();
        drop(rec);

        let mut reader = super::super::ShardReader::open(&path).unwrap();
        let ep0 = reader.next_episode().unwrap().unwrap();
        let ep1 = reader.next_episode().unwrap().unwrap();
        assert!(reader.next_episode().unwrap().is_none());
        assert_eq!(ep0.episode_id, 0);
        assert_eq!(ep0.records.len(), 1);
        assert_eq!(ep1.episode_id, 1);
        assert_eq!(ep1.records.len(), 2);
        std::fs::remove_file(&path).ok();
    }
}
