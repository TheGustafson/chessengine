//! `obs-tool` — audit and inspect Observation-Store shards (design/16 §5,
//! obs M2). Build: `cargo build --release --features datagen --bin obs-tool`.
//!
//!   obs-tool verify <shard.ceobs>...   full structural + replay audit
//!   obs-tool dump   <shard.ceobs> [n]  human summary of the first n episodes
//!
//! verify checks, per shard: header + provenance presence; per episode:
//! provenance_ref validity, replay-root reconstruction through FEN
//! validation, stored canonical_key vs the recomputed Zobrist key, and — for
//! every observation — legal replay of `path_from_episode_root` from the
//! replay root, with the destination's identity (occ, stm, canonical key)
//! matching the stored Layer 1. Exit 0 only if every check passes: a shard
//! that verifies can be deterministically replayed (design/19 §11.1).

use chessengine::board::Board;
use chessengine::obs::convert::{board_from_position, fen_from_position, replay_path};
use chessengine::obs::{
    Bound, Completion, Episode, EpisodeKind, Observation, PositionState, Provenance, ShardReader,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "verify" && !rest.is_empty() => verify(rest),
        Some((cmd, rest)) if cmd == "dump" && !rest.is_empty() => dump(rest),
        Some((cmd, rest)) if cmd == "stability" && !rest.is_empty() => stability(rest),
        Some((cmd, rest)) if cmd == "disagree" && rest.len() == 2 => disagree(rest),
        Some((cmd, rest)) if cmd == "export" && !rest.is_empty() => export(rest),
        Some((cmd, rest)) if cmd == "export-train" && rest.len() >= 2 => export_train(rest),
        Some((cmd, rest)) if cmd == "export-value" && rest.len() >= 2 => export_value(rest),
        Some((cmd, rest)) if cmd == "rank" && !rest.is_empty() => rank(rest),
        _ => {
            eprintln!("usage: obs-tool verify <shard.ceobs>...");
            eprintln!("       obs-tool dump <shard.ceobs> [episodes]");
            eprintln!("       obs-tool stability <shard.ceobs> [tie-margin-cp (default 20)]");
            eprintln!("       obs-tool disagree <shardA.ceobs> <shardB.ceobs>");
            eprintln!("       obs-tool export <shard.ceobs>   (per-arm JSONL on stdout)");
            eprintln!(
                "       obs-tool export-train <out-dir> <shard.ceobs>... [--all-tiers]  (D40-class per-move policy corpus, NumPy SoA)"
            );
            eprintln!(
                "       obs-tool export-value <out.txt> <shard.ceobs>... [--all-tiers] [--no-filter]  (scalar value corpus: FEN | cp | wdl; trainer-parity quiet filter ON by default, --no-filter = audit-only raw export)"
            );
            eprintln!("       obs-tool rank <shard.ceobs>     (classical-ordering baseline)");
            ExitCode::FAILURE
        }
    }
}

// ─── M4: D44 fail-fast audits ──────────────────────────────────────────────

/// The design/19 §5.4 stability rule over a RootMoveAnalysis shard: a
/// position's reference is ACCEPTED when the unrestricted best move agrees
/// between the two deepest tiers AND the top-tier margin (best arm minus
/// runner-up) clears the tie threshold. Everything else is reported as the
/// explicit unstable/tie class — usable as hard-case strata, never as
/// ground truth.
fn stability(args: &[String]) -> ExitCode {
    let path = &args[0];
    let tie_margin: i32 = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(20);
    let episodes = match load_root_episodes(path) {
        Ok(episodes) => episodes,
        Err(error) => {
            eprintln!("obs-tool: {error}");
            return ExitCode::FAILURE;
        }
    };
    let (mut accepted, mut best_flip, mut tie, mut too_few_tiers) = (0u64, 0u64, 0u64, 0u64);
    let mut margins = Vec::new();
    let mut tier_score_drift = Vec::new();
    let mut arm_vs_full_mismatch = 0u64;

    let mut incomplete_refs = 0u64;
    let mut stopped_arms = 0u64;
    for episode in &episodes {
        if episode.records.len() < 2 {
            too_few_tiers += 1;
            continue;
        }
        let deep = &episode.records[episode.records.len() - 1];
        let prev = &episode.records[episode.records.len() - 2];
        // A partial reference must never drive an ACCEPT (D33 round 1): its
        // best move/score are whatever arrived before the timeout.
        if deep.search.completion != chessengine::obs::Completion::Complete
            || prev.search.completion != chessengine::obs::Completion::Complete
        {
            incomplete_refs += 1;
            continue;
        }

        // Margin between the two best COMPLETE arms at the deepest tier
        // (Stopped arms carry partial scores — recorded, never aggregated).
        stopped_arms += deep
            .moves
            .iter()
            .filter(|m| m.completion != chessengine::obs::Completion::Complete)
            .count() as u64;
        let mut arm_scores: Vec<(i32, u16)> = deep
            .moves
            .iter()
            .filter(|m| m.completion == chessengine::obs::Completion::Complete)
            .map(|m| (m.returned_score, m.mv))
            .collect();
        arm_scores.sort_by_key(|&(score, _)| std::cmp::Reverse(score));
        let margin = match arm_scores.as_slice() {
            [(top, _), (second, _), ..] => top - second,
            _ => i32::MAX, // single legal move: trivially stable
        };
        margins.push(margin);

        // Does the top arm agree with the unrestricted search's best move?
        if let Some(&(_, top_mv)) = arm_scores.first()
            && top_mv != deep.search.best_move
        {
            arm_vs_full_mismatch += 1;
        }

        // Score drift of the unrestricted search between the deepest tiers.
        tier_score_drift.push((deep.search.final_score - prev.search.final_score).abs());

        if deep.search.best_move != prev.search.best_move {
            best_flip += 1;
        } else if margin < tie_margin {
            tie += 1;
        } else {
            accepted += 1;
        }
    }

    let total = accepted + best_flip + tie;
    println!("stability audit: {path}");
    println!(
        "  positions: {} ({} skipped: <2 tiers; {} skipped: incomplete reference; \
         {} Stopped arms excluded)",
        episodes.len(),
        too_few_tiers,
        incomplete_refs,
        stopped_arms
    );
    if total > 0 {
        println!(
            "  ACCEPTED (stable best + margin >= {tie_margin}cp): {accepted} ({:.0}%)",
            100.0 * accepted as f64 / total as f64
        );
        println!(
            "  best-move flip across deepest tiers:              {best_flip} ({:.0}%)",
            100.0 * best_flip as f64 / total as f64
        );
        println!(
            "  tie (margin < {tie_margin}cp):                             {tie} ({:.0}%)",
            100.0 * tie as f64 / total as f64
        );
        println!("  top-arm vs unrestricted-best mismatch:            {arm_vs_full_mismatch}");
        println!(
            "  top-tier margin cp: {}   |Δscore| across tiers cp: {}",
            quantiles(&mut margins),
            quantiles(&mut tier_score_drift)
        );
    }
    ExitCode::SUCCESS
}

/// Teacher-disagreement report between two shards over the SAME positions
/// (same fens file, same tier count): best-move agreement at the deepest
/// tier, unrestricted-score deltas, and arm-ranking correlation (Kendall
/// tau) — the design/19 §5.3 disagreement strata.
fn disagree(args: &[String]) -> ExitCode {
    let (a, b) = (&args[0], &args[1]);
    let (episodes_a, episodes_b) = match (load_root_episodes(a), load_root_episodes(b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("obs-tool: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut pairs = 0u64;
    let mut best_agree = 0u64;
    let mut score_delta = Vec::new();
    let mut taus = Vec::new();
    for (ea, eb) in episodes_a.iter().zip(&episodes_b) {
        if ea.replay_root.canonical_key != eb.replay_root.canonical_key {
            eprintln!(
                "obs-tool: episode {} roots differ — shards are not row-aligned",
                ea.episode_id
            );
            return ExitCode::FAILURE;
        }
        let (Some(da), Some(db)) = (ea.records.last(), eb.records.last()) else {
            continue;
        };
        if da.search.completion != chessengine::obs::Completion::Complete
            || db.search.completion != chessengine::obs::Completion::Complete
        {
            continue;
        }
        pairs += 1;
        if da.search.best_move == db.search.best_move {
            best_agree += 1;
        }
        score_delta.push((da.search.final_score - db.search.final_score).abs());
        taus.push(kendall_tau(&da.moves, &db.moves));
    }
    println!("disagreement audit: {a} vs {b}");
    println!("  aligned positions: {pairs}");
    if pairs > 0 {
        println!(
            "  deepest-tier best-move agreement: {best_agree} ({:.0}%)",
            100.0 * best_agree as f64 / pairs as f64
        );
        println!(
            "  |Δ unrestricted score| cp: {}",
            quantiles(&mut score_delta)
        );
        let mean_tau = taus.iter().sum::<f64>() / taus.len() as f64;
        println!("  arm-ranking Kendall tau: mean {mean_tau:.3}");
    }
    ExitCode::SUCCESS
}

/// Flat per-arm JSONL for the Derived layer (design/16 §3.6): one line per
/// (position, tier, legal move) through the store's OWN decoder — python
/// tooling consumes this instead of growing a second TLV implementation.
/// Escaping is trivial by construction: every emitted string is a FEN, a
/// UCI move, or an enum name (no quotes/backslashes possible).
fn export(args: &[String]) -> ExitCode {
    use chessengine::moves::Move;
    let episodes = match load_root_episodes(&args[0]) {
        Ok(episodes) => episodes,
        Err(error) => {
            eprintln!("obs-tool: {error}");
            return ExitCode::FAILURE;
        }
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    use std::io::Write;
    for episode in &episodes {
        let fen = match fen_from_position(&episode.replay_root) {
            Ok(fen) => fen,
            Err(error) => {
                eprintln!("obs-tool: episode {}: {error}", episode.episode_id);
                return ExitCode::FAILURE;
            }
        };
        for record in &episode.records {
            let tier = record.context.node_budget.unwrap_or(0);
            for arm in &record.moves {
                let mv = Move::from_raw16(arm.mv);
                if writeln!(
                    out,
                    "{{\"episode\":{},\"fen\":\"{}\",\"tier\":{},\"mv\":\"{}\",\
                     \"score\":{},\"nodes\":{},\"depth\":{},\"complete\":{},\
                     \"is_best\":{},\"ref_score\":{},\"ref_nodes\":{},\"ref_complete\":{},\
                     \"legal\":{},\"captures\":{}}}",
                    episode.episode_id,
                    fen,
                    tier,
                    mv,
                    arm.returned_score,
                    arm.nodes,
                    arm.searched_depth,
                    arm.completion == chessengine::obs::Completion::Complete,
                    arm.mv == record.search.best_move,
                    record.search.final_score,
                    record.search.nodes,
                    record.search.completion == chessengine::obs::Completion::Complete,
                    record.context.legal_move_count,
                    record.context.capture_move_count
                )
                .is_err()
                {
                    return ExitCode::FAILURE; // broken pipe: downstream closed
                }
            }
        }
    }
    ExitCode::SUCCESS
}

// ─── export-train: obs shards → D40-class value/allocation training corpus ───
//
// The Phase-1 target (design/19 §7, §3): learn per-move search consequences
// from the teacher's independent root-arm searches. The M4 audit
// (value-foundation-state.md 03:15) found argmax labels SOFT (median top-arm
// margin 13-16cp) but per-arm SCORES reproducible (8-10cp cross-tier drift) —
// so we emit per-arm VALUES and value-MARGINS, not hard best-move classes.
//
// Format: a NumPy structure-of-arrays directory, the same on-disk primitive
// the D40 policy trainer streams (`d40_policy.rs`'s `NpyFile`/`SideFiles` read
// exactly this: NumPy v1 `.npy`, one file per column). DELTA vs d40_policy's
// schema (documented in manifest.json + here):
//   * one row per (position, legal arm) instead of one row per position with a
//     40-wide sparse policy block — value/margin targets are per-arm and the
//     legal-move count is unbounded, so a fixed policy cap would truncate arms;
//   * targets are `value` (per-arm cp) + `margin` (vs the best arm) instead of
//     softmax policy probabilities; `is_ref_best`/`is_max_value` are optional
//     soft hard-labels (design/19 §7 final-best-move);
//   * no white/black side split (that was an Lc0-data artifact) — an explicit
//     `stm` column replaces it; `feats` are ABSOLUTE and stay compatible with
//     d40_policy's `board_from_features` ((colour*6+pt)*64+sq, -1 padded);
//   * added provenance columns `tier` (node budget), `teacher_id`, and `key`
//     (canonical position identity, for dedup + leakage-safe train/test split).
// The writer is versioned: any column-set change bumps TRAIN_WRITER.

const TRAIN_SCHEMA: &str = "archon.obs-train.rootmove-value-v1";
const TRAIN_WRITER: &str = "obs-tool export-train v1";
const MAX_TEACHERS: usize = u8::MAX as usize + 1;

/// (teacher_id, identity, row count), in stable id order.
type PerTeacherCounts = Vec<(u8, String, u64)>;
/// (node budget, row count), ascending.
type TierHistogram = Vec<(u32, u64)>;

/// Interned per-shard producer identity (design/19 §7 `producer_engine`):
/// mixed-teacher corpora stay auditable because every row carries a
/// `teacher_id` into this table and the manifest pins the full provenance.
#[derive(Clone)]
struct TeacherInfo {
    teacher_id: u8,
    identity: String,
    engine_version: String,
    net_version: Option<String>,
    search_config: String,
    search_config_hash: u64,
    generator_version: String,
    dataset_source: String,
    opening_source: String,
    hardware_identity: String,
}

fn teacher_from_prov(prov: &Provenance) -> TeacherInfo {
    // teacher_identity IS D44's producer_engine for RootMoveAnalysis shards
    // (root_analysis.rs pins it); fall back to a stable composite only if a
    // shard predates that field.
    let identity = prov
        .teacher_identity
        .clone()
        .unwrap_or_else(|| format!("{}|{}", prov.engine_version, prov.dataset_source));
    TeacherInfo {
        teacher_id: 0,
        identity,
        engine_version: prov.engine_version.clone(),
        net_version: prov.net_version.clone(),
        search_config: prov.search_config.clone(),
        search_config_hash: prov.search_config_hash,
        generator_version: prov.generator_version.clone(),
        dataset_source: prov.dataset_source.clone(),
        opening_source: prov.opening_source.clone(),
        hardware_identity: prov.hardware_identity.clone(),
    }
}

/// One emitted training example: a single Complete arm of a position's chosen
/// deepest-Complete tier. `Copy` so the SoA writer can stream columns.
#[derive(Clone, Copy)]
struct TrainRow {
    position_id: u32,
    key: u64,
    stm: u8,
    feats: [i16; 32],
    nmoves: u16,
    move_idx: i16,
    mv_raw16: u16,
    value: i32,
    margin: i32,
    is_ref_best: u8,
    is_max_value: u8,
    tier: u32,
    ref_score: i32,
    teacher_id: u8,
    arm_nodes: u32,
    arm_depth: i16,
}

#[derive(Default)]
struct Stats {
    positions_in: u64,
    positions_out: u64,
    arms_out: u64,
    arms_skipped_stopped: u64,
    dedup_drops: u64,
    skip_no_complete_reference: u64,
    skip_no_complete_arms: u64,
    skip_bad_position: u64,
    non_rootmove_episodes: u64,
    ref_best_missing: u64,
}

struct CorpusBuilder {
    rows: Vec<TrainRow>,
    seen_keys: std::collections::HashSet<u64>,
    next_position_id: u32,
    stats: Stats,
    teachers: Vec<TeacherInfo>,
    teacher_index: std::collections::HashMap<String, u8>,
}

impl CorpusBuilder {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            seen_keys: std::collections::HashSet::new(),
            next_position_id: 0,
            stats: Stats::default(),
            teachers: Vec::new(),
            teacher_index: std::collections::HashMap::new(),
        }
    }

    /// Intern a producer identity → stable `teacher_id`; identical teachers
    /// across shards share one id so a merged corpus stays coherent.
    fn intern_teacher(&mut self, prov: &Provenance) -> Result<u8, String> {
        let info = teacher_from_prov(prov);
        if let Some(&id) = self.teacher_index.get(&info.identity) {
            return Ok(id);
        }
        if self.teachers.len() >= MAX_TEACHERS {
            return Err(format!("more than {MAX_TEACHERS} distinct teachers"));
        }
        let id = self.teachers.len() as u8;
        self.teacher_index.insert(info.identity.clone(), id);
        self.teachers.push(TeacherInfo {
            teacher_id: id,
            ..info
        });
        Ok(id)
    }

    /// Fold one episode into the corpus (deepest-tier only — the default,
    /// backward-compatible behaviour). Kept as a thin wrapper so the unit tests
    /// and any legacy caller stay unchanged. (Only exercised from tests now that
    /// the CLI path calls `add_episode_tiered`.)
    #[cfg_attr(not(test), allow(dead_code))]
    fn add_episode(&mut self, episode: &Episode, teacher_id: u8) {
        self.add_episode_tiered(episode, teacher_id, false);
    }

    /// Fold one episode into the corpus. Non-RootMoveAnalysis episodes and
    /// positions without a Complete reference/arm are counted and skipped;
    /// duplicate canonical keys (across every input shard) drop with
    /// first-occurrence-wins.
    ///
    /// `all_tiers` controls tier selection (D41 v1.1 anti-overfit data lever):
    ///   * `false` — emit ONLY the deepest tier whose unrestricted reference is
    ///     Complete AND which has ≥1 Complete arm (design/19 §5.4). The historic
    ///     export-train behaviour.
    ///   * `true` — emit EVERY such Complete tier, each row tagged with that
    ///     tier's node budget and its margin computed vs THAT tier's own best
    ///     arm. All tiers of one position share a single `position_id`/`key`, so
    ///     the leakage-safe train/test split (by key) still keeps every tier of
    ///     a position on one side; the trainer groups ranking by (key, tier), so
    ///     shallow/deep searches stay separate ranking problems.
    fn add_episode_tiered(&mut self, episode: &Episode, teacher_id: u8, all_tiers: bool) {
        match episode.kind {
            EpisodeKind::RootMoveAnalysis => {}
            // In-search capture (design/16 §3.5.5): each RECORD is its own
            // sampled interior node (full position key, empty replay path), so
            // it folds record-as-position — not the shared-replay-root +
            // per-tier shape the RootMoveAnalysis path below assumes.
            EpisodeKind::InSearchCapture => {
                self.add_insearch_episode(episode, teacher_id);
                return;
            }
            _ => {
                self.stats.non_rootmove_episodes += 1;
                return;
            }
        }
        self.stats.positions_in += 1;
        let key = episode.replay_root.canonical_key;
        if !self.seen_keys.insert(key) {
            self.stats.dedup_drops += 1;
            return;
        }

        // Every tier whose unrestricted reference is Complete AND which has at
        // least one Complete arm, plus the single deepest such tier.
        let mut qualifying: Vec<&Observation> = Vec::new();
        let mut deepest: Option<&Observation> = None;
        let mut any_complete_ref = false;
        for record in &episode.records {
            if record.search.completion != Completion::Complete {
                continue;
            }
            any_complete_ref = true;
            if !record
                .moves
                .iter()
                .any(|m| m.completion == Completion::Complete)
            {
                continue;
            }
            qualifying.push(record);
            let budget = record.context.node_budget.unwrap_or(0);
            if deepest.is_none_or(|c| budget > c.context.node_budget.unwrap_or(0)) {
                deepest = Some(record);
            }
        }
        if qualifying.is_empty() {
            if any_complete_ref {
                self.stats.skip_no_complete_arms += 1;
            } else {
                self.stats.skip_no_complete_reference += 1;
            }
            return;
        }

        let feats = match source_features(&episode.replay_root) {
            Ok(feats) => feats,
            Err(_) => {
                self.stats.skip_bad_position += 1;
                return;
            }
        };
        let stm = episode.replay_root.stm;

        let position_id = self.next_position_id;
        self.next_position_id += 1;
        self.stats.positions_out += 1;

        if all_tiers {
            // Ascending tier order keeps a position's rows tier-contiguous.
            qualifying.sort_by_key(|r| r.context.node_budget.unwrap_or(0));
            for record in qualifying {
                self.emit_record(record, &feats, position_id, key, stm, teacher_id);
            }
        } else {
            self.emit_record(deepest.unwrap(), &feats, position_id, key, stm, teacher_id);
        }
    }

    /// Fold one in-search-capture episode (design/16 §3.5.5). Each record is an
    /// independently-keyed interior search node, so we iterate records-as-
    /// positions (features from the record's own position, not a shared replay
    /// root). QUARANTINE-SAFE label gate: the recorder stamps every capture
    /// `Completion::Complete`, so completion is NOT an exactness signal here —
    /// the node's `bound` is. Only `Bound::Exact` (PV) nodes carry a true
    /// evaluation; cut/all nodes are aspiration bounds and are dropped (the same
    /// bound-quarantine our 5k-node datagen labels sit under). NOTE: even at an
    /// exact node, non-best arms' `returned_score` are scout-search bounds, so
    /// the emitted margins are ordering-grade, not clean scalar values — train
    /// this as a pairwise/ranking signal, not an L2 value target.
    fn add_insearch_episode(&mut self, episode: &Episode, teacher_id: u8) {
        for record in &episode.records {
            self.stats.positions_in += 1;
            if record.search.bound != Bound::Exact {
                self.stats.skip_no_complete_reference += 1;
                continue;
            }
            if !record
                .moves
                .iter()
                .any(|m| m.completion == Completion::Complete)
            {
                self.stats.skip_no_complete_arms += 1;
                continue;
            }
            let key = record.position.canonical_key;
            if !self.seen_keys.insert(key) {
                self.stats.dedup_drops += 1;
                continue;
            }
            let feats = match source_features(&record.position) {
                Ok(feats) => feats,
                Err(_) => {
                    self.stats.skip_bad_position += 1;
                    continue;
                }
            };
            let stm = record.position.stm;
            let position_id = self.next_position_id;
            self.next_position_id += 1;
            self.stats.positions_out += 1;
            self.emit_record(record, &feats, position_id, key, stm, teacher_id);
        }
    }

    /// Emit every Complete arm of one tier (record) as a `TrainRow`, with the
    /// per-tier best arm defining margin 0 and the per-tier `is_max_value`.
    fn emit_record(
        &mut self,
        record: &Observation,
        feats: &[i16; 32],
        position_id: u32,
        key: u64,
        stm: u8,
        teacher_id: u8,
    ) {
        let complete: Vec<&_> = record
            .moves
            .iter()
            .filter(|m| m.completion == Completion::Complete)
            .collect();
        self.stats.arms_skipped_stopped += (record.moves.len() - complete.len()) as u64;
        // Caller guarantees a non-empty Complete set for this record.
        let best_value = complete.iter().map(|m| m.returned_score).max().unwrap();
        let tier = record.context.node_budget.unwrap_or(0) as u32;
        let ref_score = record.search.final_score;
        let ref_best = record.search.best_move;
        let nmoves = complete.len() as u16;

        let mut max_taken = false; // exactly one is_max_value per tier (ties → first)
        let mut ref_best_seen = false;
        for arm in complete {
            let is_max = arm.returned_score == best_value && !max_taken;
            max_taken |= is_max;
            let is_ref_best = arm.mv == ref_best;
            ref_best_seen |= is_ref_best;
            let from = arm.mv & 0x3f;
            let to = (arm.mv >> 6) & 0x3f;
            self.rows.push(TrainRow {
                position_id,
                key,
                stm,
                feats: *feats,
                nmoves,
                move_idx: (from * 64 + to) as i16,
                mv_raw16: arm.mv,
                value: arm.returned_score,
                margin: arm.returned_score - best_value,
                is_ref_best: u8::from(is_ref_best),
                is_max_value: u8::from(is_max),
                tier,
                ref_score,
                teacher_id,
                arm_nodes: arm.nodes.min(u64::from(u32::MAX)) as u32,
                arm_depth: arm
                    .searched_depth
                    .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                    as i16,
            });
            self.stats.arms_out += 1;
        }
        if !ref_best_seen {
            self.stats.ref_best_missing += 1;
        }
    }
}

/// The 32 absolute source features of a stored position, `board_from_features`
/// compatible: reconstruct THROUGH `Board::from_fen` (full validation), then
/// one `(colour*6+pt)*64+sq` index per occupied square in ascending order,
/// `-1` padded. A position with >32 pieces (impossible in legal chess) errors
/// rather than overflow.
fn source_features(position: &PositionState) -> Result<[i16; 32], String> {
    let board = board_from_position(position)?;
    let mut feats = [-1i16; 32];
    for (n, sq) in board.all_occupancy().into_iter().enumerate() {
        if n >= 32 {
            return Err("position has more than 32 pieces".into());
        }
        let piece = board.piece_on(sq).expect("occupied square");
        let feat = (u16::from(piece.color as u8) * 6 + u16::from(piece.piece_type as u8)) * 64
            + u16::from(sq.index());
        feats[n] = feat as i16;
    }
    Ok(feats)
}

// ─── NumPy v1 .npy writer (std-only; the exact layout d40_policy.rs reads) ───

/// A NumPy v1.0 `.npy` header: magic, version, u16 header length, then a
/// dict literal padded with spaces so the data section starts 64-byte aligned
/// and terminated with `\n`. Mirrors `NpyFile::open` in `d40_policy.rs`.
fn npy_header(descr: &str, shape: &str) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape}), }}");
    let unpadded = 10 + dict.len() + 1; // 6 magic + 2 version + 2 length + dict + '\n'
    let pad = (64 - (unpadded % 64)) % 64;
    let mut body = dict.into_bytes();
    body.resize(body.len() + pad, b' ');
    body.push(b'\n');
    let header_len = u16::try_from(body.len()).expect("npy header fits u16");
    let mut out = Vec::with_capacity(10 + body.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn write_npy(
    dir: &std::path::Path,
    name: &str,
    descr: &str,
    shape: &str,
    data: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(dir.join(name))?);
    file.write_all(&npy_header(descr, shape))?;
    file.write_all(data)?;
    file.flush()
}

/// Minimal JSON string literal (quote + escape) — the manifest is the only
/// place strings can carry arbitrary teacher/path bytes.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Write the whole SoA corpus (one `.npy` per column) plus `manifest.json`.
#[allow(clippy::too_many_arguments)] // cohesive SoA-corpus writer; args are the columns' provenance
fn write_corpus(
    dir: &std::path::Path,
    rows: &[TrainRow],
    teachers: &[TeacherInfo],
    stats: &Stats,
    sources: &[String],
    per_teacher: &[(u8, String, u64)],
    tier_hist: &[(u32, u64)],
    all_tiers: bool,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let n = rows.len();
    let shape1 = format!("{n},");

    // Scalar columns.
    let mut buf = Vec::with_capacity(n * 4);
    for r in rows {
        buf.extend_from_slice(&r.position_id.to_le_bytes());
    }
    write_npy(dir, "position_id.npy", "<u4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.key.to_le_bytes());
    }
    write_npy(dir, "key.npy", "<u8", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.push(r.stm);
    }
    write_npy(dir, "stm.npy", "|u1", &shape1, &buf)?;

    // feats: (N, 32) i16, C-order.
    buf.clear();
    for r in rows {
        for &f in &r.feats {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    write_npy(dir, "feats.npy", "<i2", &format!("{n}, 32"), &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.nmoves.to_le_bytes());
    }
    write_npy(dir, "nmoves.npy", "<u2", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.move_idx.to_le_bytes());
    }
    write_npy(dir, "move_idx.npy", "<i2", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.mv_raw16.to_le_bytes());
    }
    write_npy(dir, "mv_raw16.npy", "<u2", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.value.to_le_bytes());
    }
    write_npy(dir, "value.npy", "<i4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.margin.to_le_bytes());
    }
    write_npy(dir, "margin.npy", "<i4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.push(r.is_ref_best);
    }
    write_npy(dir, "is_ref_best.npy", "|u1", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.push(r.is_max_value);
    }
    write_npy(dir, "is_max_value.npy", "|u1", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.tier.to_le_bytes());
    }
    write_npy(dir, "tier.npy", "<u4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.ref_score.to_le_bytes());
    }
    write_npy(dir, "ref_score.npy", "<i4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.push(r.teacher_id);
    }
    write_npy(dir, "teacher_id.npy", "|u1", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.arm_nodes.to_le_bytes());
    }
    write_npy(dir, "arm_nodes.npy", "<u4", &shape1, &buf)?;

    buf.clear();
    for r in rows {
        buf.extend_from_slice(&r.arm_depth.to_le_bytes());
    }
    write_npy(dir, "arm_depth.npy", "<i2", &shape1, &buf)?;

    write_manifest(
        dir,
        rows,
        teachers,
        stats,
        sources,
        per_teacher,
        tier_hist,
        all_tiers,
    )
}

#[allow(clippy::too_many_arguments)] // mirrors write_corpus's provenance columns
fn write_manifest(
    dir: &std::path::Path,
    rows: &[TrainRow],
    teachers: &[TeacherInfo],
    stats: &Stats,
    sources: &[String],
    per_teacher: &[(u8, String, u64)],
    tier_hist: &[(u32, u64)],
    all_tiers: bool,
) -> std::io::Result<()> {
    let created_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let columns = [
        (
            "position_id",
            "<u4",
            "dense position id; arms of one position are contiguous and share it",
        ),
        (
            "key",
            "<u8",
            "canonical_key (unfolded Zobrist position identity) for dedup + leakage-safe split",
        ),
        ("stm", "|u1", "side to move (0=white,1=black)"),
        (
            "feats",
            "<i2 [N,32]",
            "absolute source features (colour*6+pt)*64+sq, -1 padded; d40 board_from_features compatible",
        ),
        (
            "nmoves",
            "<u2",
            "count of Complete arms emitted for this position (root complexity)",
        ),
        (
            "move_idx",
            "<i2",
            "from*64+to in 0..4095 (d40 policy-index convention)",
        ),
        (
            "mv_raw16",
            "<u2",
            "full packed Move (kind|promo|to|from); lossless move identity",
        ),
        (
            "value",
            "<i4",
            "PRIMARY per-arm value cp (returned_score at the chosen tier)",
        ),
        (
            "margin",
            "<i4",
            "PRIMARY value - best_arm_value (<=0; best arm = 0)",
        ),
        (
            "is_ref_best",
            "|u1",
            "1 if arm == unrestricted reference best move (soft best-move label)",
        ),
        (
            "is_max_value",
            "|u1",
            "1 for the max-value arm (ties -> first)",
        ),
        (
            "tier",
            "<u4",
            "node budget of the chosen deepest-Complete tier",
        ),
        (
            "ref_score",
            "<i4",
            "unrestricted reference final score cp at the chosen tier",
        ),
        (
            "teacher_id",
            "|u1",
            "index into `teachers` (producer_engine, design/19 §7)",
        ),
        (
            "arm_nodes",
            "<u4",
            "nodes the arm search spent (Phase-3 cost signal)",
        ),
        ("arm_depth", "<i2", "depth the arm search reached"),
    ];
    let columns_json = columns
        .iter()
        .map(|(name, dtype, doc)| {
            format!(
                "{}: {{\"dtype\": {}, \"doc\": {}}}",
                json_escape(name),
                json_escape(dtype),
                json_escape(doc)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let teachers_json = teachers
        .iter()
        .map(|t| {
            let net = t
                .net_version
                .as_deref()
                .map_or_else(|| "null".to_string(), json_escape);
            format!(
                "{{\"teacher_id\": {}, \"identity\": {}, \"engine_version\": {}, \"net_version\": {}, \
                 \"search_config\": {}, \"search_config_hash\": {}, \"generator_version\": {}, \
                 \"dataset_source\": {}, \"opening_source\": {}, \"hardware_identity\": {}}}",
                t.teacher_id,
                json_escape(&t.identity),
                json_escape(&t.engine_version),
                net,
                json_escape(&t.search_config),
                t.search_config_hash,
                json_escape(&t.generator_version),
                json_escape(&t.dataset_source),
                json_escape(&t.opening_source),
                json_escape(&t.hardware_identity),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let per_teacher_json = per_teacher
        .iter()
        .map(|(id, name, rows)| {
            format!(
                "{{\"teacher_id\": {id}, \"identity\": {}, \"rows\": {rows}}}",
                json_escape(name)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let tier_hist_json = tier_hist
        .iter()
        .map(|(tier, count)| format!("\"{tier}\": {count}"))
        .collect::<Vec<_>>()
        .join(", ");

    let sources_json = sources
        .iter()
        .map(|s| json_escape(s))
        .collect::<Vec<_>>()
        .join(", ");

    let tier_selection = if all_tiers {
        "\"all Complete tiers (Complete reference AND >=1 Complete arm); margin per-tier; --all-tiers\""
    } else {
        "\"deepest tier with Complete reference AND >=1 Complete arm\""
    };
    let manifest = format!(
        "{{\n  \"schema\": {schema},\n  \"writer\": {writer},\n  \"created_unix\": {created},\n  \
         \"rows\": {rows_len},\n  \"row_granularity\": \"per (position, legal Complete arm)\",\n  \
         \"tier_selection\": {tier_selection},\n  \
         \"value_encoding\": \"centipawns as the teacher reported; mate mapped to +-(31000-n) (root_analysis.rs)\",\n  \
         \"margin_definition\": \"value - max(Complete arm value); best arm = 0, worse arms negative\",\n  \
         \"targets\": {{\"primary\": [\"value\", \"margin\"], \"soft_best_move\": [\"is_ref_best\", \"is_max_value\"]}},\n  \
         \"stats\": {{\"positions_in\": {p_in}, \"positions_out\": {p_out}, \"arms_out\": {a_out}, \
         \"arms_skipped_stopped\": {a_stop}, \"dedup_drops\": {dedup}, \
         \"skip_no_complete_reference\": {s_ref}, \"skip_no_complete_arms\": {s_arms}, \
         \"skip_bad_position\": {s_bad}, \"non_rootmove_episodes\": {non_rma}, \
         \"ref_best_missing\": {rbm}}},\n  \
         \"per_teacher\": [{per_teacher_json}],\n  \
         \"tier_histogram\": {{{tier_hist_json}}},\n  \
         \"columns\": {{{columns_json}}},\n  \
         \"teachers\": [{teachers_json}],\n  \
         \"source_shards\": [{sources_json}]\n}}\n",
        schema = json_escape(TRAIN_SCHEMA),
        writer = json_escape(TRAIN_WRITER),
        created = created_unix,
        rows_len = rows.len(),
        p_in = stats.positions_in,
        p_out = stats.positions_out,
        a_out = stats.arms_out,
        a_stop = stats.arms_skipped_stopped,
        dedup = stats.dedup_drops,
        s_ref = stats.skip_no_complete_reference,
        s_arms = stats.skip_no_complete_arms,
        s_bad = stats.skip_bad_position,
        non_rma = stats.non_rootmove_episodes,
        rbm = stats.ref_best_missing,
    );
    std::fs::write(dir.join("manifest.json"), manifest)
}

/// Per-teacher row counts and tier histogram, derived from the emitted rows in
/// stable id/tier order.
fn corpus_aggregates(
    rows: &[TrainRow],
    teachers: &[TeacherInfo],
) -> (PerTeacherCounts, TierHistogram) {
    let mut per_teacher: PerTeacherCounts = teachers
        .iter()
        .map(|t| (t.teacher_id, t.identity.clone(), 0))
        .collect();
    let mut tiers: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    for row in rows {
        if let Some(entry) = per_teacher
            .iter_mut()
            .find(|(id, _, _)| *id == row.teacher_id)
        {
            entry.2 += 1;
        }
        *tiers.entry(row.tier).or_insert(0) += 1;
    }
    (per_teacher, tiers.into_iter().collect())
}

fn export_train(args: &[String]) -> ExitCode {
    // `--all-tiers` (D41 v1.1): emit every Complete tier per position, not just
    // the deepest. Flag may appear anywhere; the rest are out-dir + shards.
    let all_tiers = args.iter().any(|a| a == "--all-tiers");
    let positional: Vec<&String> = args
        .iter()
        .filter(|a| a.as_str() != "--all-tiers")
        .collect();
    if positional.len() < 2 {
        eprintln!("obs-tool: export-train needs <out-dir> <shard.ceobs>... [--all-tiers]");
        return ExitCode::FAILURE;
    }
    let out_dir = std::path::PathBuf::from(positional[0]);
    let shards: Vec<String> = positional[1..].iter().map(|s| (*s).clone()).collect();
    let mut builder = CorpusBuilder::new();

    // Deterministic order: shards in argument order, episodes in stream order,
    // arms in stored order. dedup by canonical_key across every shard.
    for path in &shards {
        let mut reader = match ShardReader::open(path) {
            Ok(reader) => reader,
            Err(error) => {
                eprintln!("obs-tool: {path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        let provenance = reader.header().provenance.clone();
        loop {
            let episode = match reader.next_episode() {
                Ok(Some(episode)) => episode,
                Ok(None) => break,
                Err(error) => {
                    eprintln!("obs-tool: {path}: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let Some(prov) = provenance.get(episode.provenance_ref as usize) else {
                eprintln!(
                    "obs-tool: {path}: episode {} provenance_ref {} out of range",
                    episode.episode_id, episode.provenance_ref
                );
                return ExitCode::FAILURE;
            };
            let teacher_id = match builder.intern_teacher(prov) {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("obs-tool: {path}: {error}");
                    return ExitCode::FAILURE;
                }
            };
            builder.add_episode_tiered(&episode, teacher_id, all_tiers);
        }
    }

    if builder.rows.is_empty() {
        eprintln!("obs-tool: export-train produced 0 rows (no usable positions)");
        return ExitCode::FAILURE;
    }

    let (per_teacher, tier_hist) = corpus_aggregates(&builder.rows, &builder.teachers);
    if let Err(error) = write_corpus(
        &out_dir,
        &builder.rows,
        &builder.teachers,
        &builder.stats,
        &shards,
        &per_teacher,
        &tier_hist,
        all_tiers,
    ) {
        eprintln!("obs-tool: writing {}: {error}", out_dir.display());
        return ExitCode::FAILURE;
    }

    print_train_stats(&out_dir, &builder.stats, &per_teacher, &tier_hist);
    ExitCode::SUCCESS
}

fn print_train_stats(
    out_dir: &std::path::Path,
    stats: &Stats,
    per_teacher: &[(u8, String, u64)],
    tier_hist: &[(u32, u64)],
) {
    println!(
        "export-train -> {} (NumPy SoA + manifest.json)",
        out_dir.display()
    );
    println!("  positions in (RootMoveAnalysis): {}", stats.positions_in);
    println!("  positions out:                   {}", stats.positions_out);
    println!("  arms out (rows):                 {}", stats.arms_out);
    println!(
        "  arms skipped (Stopped in tier):  {}",
        stats.arms_skipped_stopped
    );
    println!("  skip reasons:");
    println!("    dedup (canonical_key seen):    {}", stats.dedup_drops);
    println!(
        "    no Complete reference:         {}",
        stats.skip_no_complete_reference
    );
    println!(
        "    Complete ref but no arms:      {}",
        stats.skip_no_complete_arms
    );
    println!(
        "    unreconstructable position:    {}",
        stats.skip_bad_position
    );
    println!(
        "    non-RootMoveAnalysis episodes: {}",
        stats.non_rootmove_episodes
    );
    println!(
        "  positions missing ref-best arm:  {}",
        stats.ref_best_missing
    );
    println!("  per-teacher rows:");
    for (id, identity, rows) in per_teacher {
        println!("    [{id}] {rows} rows  ({identity})");
    }
    print!("  tier histogram (rows):");
    for (tier, count) in tier_hist {
        print!(" {tier}={count}");
    }
    println!();
}

// ─── export-value: obs shards → scalar value corpus (the Tier-F view) ────────
//
// The scalar-value counterpart to export-train's per-move policy corpus. Where
// export-train needs Complete ARMS (Tier-R/rich shards), export-value needs
// only the unrestricted REFERENCE search — so it consumes Tier-F
// (`root-analysis --reference-only`) shards, which carry no arms. Output is the
// text schema the bullet corpus pipeline already ingests (tools/nnue/
// build_corpus.sh → bf-convert): one `FEN | cp | wdl` line per accepted
// position, cp WHITE-relative and clamped to ±2500 (mates collapse to the cap,
// matching relabel_depth.py). QUARANTINE (CLAUDE.md / design/19 §5.4): only a
// Bound::Exact + Completion::Complete reference emits a row — aspiration bounds
// and partial (timed-out) searches never feed a scalar value loss. WDL is a
// 0.5 placeholder (a teacher relabel has no game result, identical to
// relabel_depth's default for FEN-only inputs); export-time WDL derivation is a
// documented future lever (data-schema doc §extensions).

const VALUE_SCORE_CAP: i32 = 2500;

/// stm-relative UCI score → White-relative cp, clamped to ±cap (which also
/// collapses the ±MATE_BASE mate encoding that root-analysis records down to
/// ±cap, exactly as relabel_depth.py caps mates).
fn value_cp_white(stm_score: i32, stm: u8) -> i32 {
    let white = if stm == 0 { stm_score } else { -stm_score };
    white.clamp(-VALUE_SCORE_CAP, VALUE_SCORE_CAP)
}

/// True only for an accepted scalar-value reference (PV-exact + completed).
fn value_reference_ok(obs: &Observation) -> bool {
    obs.search.bound == Bound::Exact && obs.search.completion == Completion::Complete
}

/// Why `--filter-sfbinpack` dropped a row (one counter per criterion, in the
/// trainer's short-circuit order — the summary line is how a silent
/// filter-coverage regression gets noticed).
#[derive(Default)]
struct SfbFilterStats {
    ply: u64,
    check: u64,
    score: u64,
    kind: u64,
    capture: u64,
    bad_board: u64,
}

impl SfbFilterStats {
    fn total(&self) -> u64 {
        self.ply + self.check + self.score + self.kind + self.capture + self.bad_board
    }
}

/// The `--filter-sfbinpack` predicate: the EXACT five criteria the trainer's
/// `sfbinpack_filter` (d40_features.rs) applies to binpack entries at load
/// time, so a text-corpus export reaches the loss with the same distribution
/// a binpack does. Skipping this filter is how ~22% best-move-capture
/// positions poisoned every canon run while "quiet-filtered" runs removed
/// only the in-check ~9% (diagnosis 2026-07-21, delta 2). Returns
/// `Some(drop-reason updater)` if the row must be dropped.
///
/// Criteria (trainer order): ply ≥ 16, side to move not in check,
/// |raw score| ≤ 10000, best move `MoveKind::Normal` (no promo/ep/castling),
/// best-move destination empty (non-capture). The score is tested on the RAW
/// stm-relative value BEFORE `value_cp_white`'s ±2500 clamp — a mate label
/// collapsed to ±2500 first would silently pass the ≤10000 criterion.
fn sfbinpack_parity_drop(
    obs: &Observation,
    position: &PositionState,
) -> Option<fn(&mut SfbFilterStats)> {
    use chessengine::moves::{Move, MoveKind};
    if position.game_ply < 16 {
        return Some(|s| s.ply += 1);
    }
    let Ok(board) = checked_board(position) else {
        return Some(|s| s.bad_board += 1);
    };
    if !board.checkers().is_empty() {
        return Some(|s| s.check += 1);
    }
    if obs.search.final_score.unsigned_abs() > 10_000 {
        return Some(|s| s.score += 1);
    }
    let mv = Move::from_raw16(obs.search.best_move);
    if !mv.is_ok() || mv.kind() != MoveKind::Normal {
        return Some(|s| s.kind += 1);
    }
    if board.is_capture(mv) {
        return Some(|s| s.capture += 1);
    }
    None
}

fn export_value(args: &[String]) -> ExitCode {
    // `--all-tiers`: emit every accepted tier per position (default: only the
    // deepest accepted reference — one clean label per position).
    //
    // The trainer-parity quiet filter (`sfbinpack_parity_drop`) is ON BY
    // DEFAULT (2026-07-21): two corpus poisonings in a row came from
    // unfiltered exports feeding a scalar value loss (the canon −582 and the
    // hardneg aux — docs/research/sf18-canon-data-diagnosis-2026-07-21.md), so
    // the safe path is now the default path. `--no-filter` opts out for
    // NON-TRAINING uses only (audits, label studies, key extraction) and warns
    // loudly; `--filter-sfbinpack` is accepted as an explicit no-op for
    // self-documenting call sites.
    let all_tiers = args.iter().any(|a| a == "--all-tiers");
    let no_filter = args.iter().any(|a| a == "--no-filter");
    let sfb_filter = !no_filter;
    if no_filter {
        eprintln!(
            "obs-tool: export-value --no-filter: output is NOT training-grade for a scalar \
             value loss (in-check/capture-best/mate-scale labels retained). Audit use only."
        );
    }
    let positional: Vec<&String> = args
        .iter()
        .filter(|a| {
            a.as_str() != "--all-tiers"
                && a.as_str() != "--filter-sfbinpack"
                && a.as_str() != "--no-filter"
        })
        .collect();
    if positional.len() < 2 {
        eprintln!(
            "obs-tool: export-value needs <out.txt> <shard.ceobs>... [--all-tiers] [--no-filter]"
        );
        return ExitCode::FAILURE;
    }
    let out_path = std::path::PathBuf::from(positional[0]);
    let shards: Vec<String> = positional[1..].iter().map(|s| (*s).clone()).collect();

    let mut lines: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let (mut positions_in, mut skip_quarantined, mut skip_bad_fen, mut dedup_drops, mut skip_kind) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut filter_stats = SfbFilterStats::default();

    // Emit one `FEN | cp | wdl` line for a chosen reference observation.
    // Returns false only on a bad FEN; a filter drop is recorded in
    // `filter_stats` by the caller-supplied stats and counts as handled.
    let emit = |obs: &Observation,
                position: &PositionState,
                lines: &mut Vec<String>,
                filter_stats: &mut SfbFilterStats|
     -> bool {
        if sfb_filter && let Some(bump) = sfbinpack_parity_drop(obs, position) {
            bump(filter_stats);
            return true;
        }
        let fen = match fen_from_position(position) {
            Ok(fen) => fen,
            Err(_) => return false,
        };
        let cp = value_cp_white(obs.search.final_score, position.stm);
        lines.push(format!("{fen} | {cp} | 0.5"));
        true
    };

    for path in &shards {
        let mut reader = match ShardReader::open(path) {
            Ok(reader) => reader,
            Err(error) => {
                eprintln!("obs-tool: {path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        loop {
            let episode = match reader.next_episode() {
                Ok(Some(episode)) => episode,
                Ok(None) => break,
                Err(error) => {
                    eprintln!("obs-tool: {path}: {error}");
                    return ExitCode::FAILURE;
                }
            };
            match episode.kind {
                EpisodeKind::RootMoveAnalysis => {
                    // One position (replay_root), one row per accepted tier.
                    positions_in += 1;
                    let key = episode.replay_root.canonical_key;
                    if !seen.insert(key) {
                        dedup_drops += 1;
                        continue;
                    }
                    let accepted: Vec<&Observation> = episode
                        .records
                        .iter()
                        .filter(|o| value_reference_ok(o))
                        .collect();
                    if accepted.is_empty() {
                        skip_quarantined += 1;
                        continue;
                    }
                    let chosen: Vec<&Observation> = if all_tiers {
                        accepted
                    } else {
                        // Deepest accepted tier by node budget.
                        vec![
                            *accepted
                                .iter()
                                .max_by_key(|o| o.context.node_budget.unwrap_or(0))
                                .unwrap(),
                        ]
                    };
                    for obs in chosen {
                        if !emit(obs, &episode.replay_root, &mut lines, &mut filter_stats) {
                            skip_bad_fen += 1;
                        }
                    }
                }
                EpisodeKind::InSearchCapture => {
                    // Each record is an independently-keyed interior node.
                    for obs in &episode.records {
                        positions_in += 1;
                        if !value_reference_ok(obs) {
                            skip_quarantined += 1;
                            continue;
                        }
                        let key = obs.position.canonical_key;
                        if !seen.insert(key) {
                            dedup_drops += 1;
                            continue;
                        }
                        if !emit(obs, &obs.position, &mut lines, &mut filter_stats) {
                            skip_bad_fen += 1;
                        }
                    }
                }
                _ => skip_kind += 1,
            }
        }
    }

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        eprintln!("obs-tool: creating {}: {error}", parent.display());
        return ExitCode::FAILURE;
    }
    let body = if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    };
    if let Err(error) = std::fs::write(&out_path, body) {
        eprintln!("obs-tool: writing {}: {error}", out_path.display());
        return ExitCode::FAILURE;
    }
    eprintln!(
        "export-value: {} row(s) -> {} (positions_in {positions_in}, quarantined {skip_quarantined}, \
         dedup {dedup_drops}, bad_fen {skip_bad_fen}, non-value-kind {skip_kind})",
        lines.len(),
        out_path.display()
    );
    if sfb_filter {
        let f = &filter_stats;
        eprintln!(
            "export-value: sfbinpack-parity filter dropped {} (ply<16 {}, in-check {}, |score|>10000 {}, \
             non-normal-best {}, capture-best {}, bad_board {})",
            f.total(),
            f.ply,
            f.check,
            f.score,
            f.kind,
            f.capture,
            f.bad_board
        );
    }
    ExitCode::SUCCESS
}

/// The design/19 §8.1 CLASSICAL-ORDERING baseline: where does the teacher's
/// reference best move sit in TEMPO's own MovePicker order (fresh
/// histories, no TT move — the cold ordering a Phase-1 model must beat)?
/// Only complete deepest-tier references count; unstable positions are the
/// model's hard class, not the baseline's.
fn rank(args: &[String]) -> ExitCode {
    use chessengine::history::{ContHistId, Histories};
    use chessengine::movepick::MovePicker;
    use chessengine::moves::Move;
    let episodes = match load_root_episodes(&args[0]) {
        Ok(episodes) => episodes,
        Err(error) => {
            eprintln!("obs-tool: {error}");
            return ExitCode::FAILURE;
        }
    };
    let histories = Histories::new(); // fresh: the cold-start ordering
    let (mut scored, mut skipped) = (0u64, 0u64);
    let (mut top1, mut top3, mut top5) = (0u64, 0u64, 0u64);
    let mut rank_sum = 0u64;
    for episode in &episodes {
        let Some(deep) = episode.records.last() else {
            skipped += 1;
            continue;
        };
        if deep.search.completion != chessengine::obs::Completion::Complete {
            skipped += 1;
            continue;
        }
        let board = match checked_board(&episode.replay_root) {
            Ok(board) => board,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let best = Move::from_raw16(deep.search.best_move);
        let mut picker = MovePicker::new(&board, Move::NULL, 1, 0, [ContHistId::SENTINEL; 6]);
        let mut position = 0u64;
        let mut found = None;
        while let Some(mv) = picker.next_move(&board, &histories) {
            if !board.legal(mv) {
                continue;
            }
            position += 1;
            if mv == best {
                found = Some(position);
                break;
            }
        }
        let Some(rank) = found else {
            skipped += 1; // reference best not produced (shouldn't happen)
            continue;
        };
        scored += 1;
        rank_sum += rank;
        if rank <= 1 {
            top1 += 1;
        }
        if rank <= 3 {
            top3 += 1;
        }
        if rank <= 5 {
            top5 += 1;
        }
    }
    println!("classical-ordering baseline: {}", args[0]);
    println!("  scored positions: {scored} (skipped {skipped})");
    if scored > 0 {
        println!(
            "  teacher-best in classical order: top-1 {:.1}%  top-3 {:.1}%  top-5 {:.1}%  mean rank {:.2}",
            100.0 * top1 as f64 / scored as f64,
            100.0 * top3 as f64 / scored as f64,
            100.0 * top5 as f64 / scored as f64,
            rank_sum as f64 / scored as f64
        );
    }
    ExitCode::SUCCESS
}

fn load_root_episodes(path: &str) -> Result<Vec<Episode>, String> {
    let mut reader = ShardReader::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut episodes = Vec::new();
    loop {
        match reader.next_episode() {
            Ok(Some(episode)) => {
                if episode.kind == chessengine::obs::EpisodeKind::RootMoveAnalysis {
                    episodes.push(episode);
                }
            }
            Ok(None) => break,
            Err(error) => return Err(format!("{path}: {error}")),
        }
    }
    if episodes.is_empty() {
        return Err(format!("{path}: no RootMoveAnalysis episodes"));
    }
    Ok(episodes)
}

/// p25/p50/p75/p95 of an integer sample, as a compact string.
fn quantiles(sample: &mut [i32]) -> String {
    if sample.is_empty() {
        return "n/a".into();
    }
    sample.sort_unstable();
    let at = |p: usize| sample[(sample.len() - 1) * p / 100];
    format!(
        "p25 {} / p50 {} / p75 {} / p95 {} (n={})",
        at(25),
        at(50),
        at(75),
        at(95),
        sample.len()
    )
}

/// Kendall tau between two engines' arm scores over the same move set.
/// Tau-b (tie-aware): ties are everywhere at low node budgets, and the
/// tau-a denominator would bias |tau| toward 0 exactly there (D33 round 1).
fn kendall_tau(
    a: &[chessengine::obs::PerMoveObservation],
    b: &[chessengine::obs::PerMoveObservation],
) -> f64 {
    use chessengine::obs::Completion;
    let scores_b: std::collections::HashMap<u16, i32> = b
        .iter()
        .filter(|m| m.completion == Completion::Complete)
        .map(|m| (m.mv, m.returned_score))
        .collect();
    let rows: Vec<(i32, i32)> = a
        .iter()
        .filter(|m| m.completion == Completion::Complete)
        .filter_map(|m| scores_b.get(&m.mv).map(|&sb| (m.returned_score, sb)))
        .collect();
    if rows.len() < 2 {
        return 1.0;
    }
    let (mut concordant, mut discordant) = (0i64, 0i64);
    let (mut ties_a, mut ties_b) = (0i64, 0i64);
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let da = rows[i].0 - rows[j].0;
            let db = rows[i].1 - rows[j].1;
            if da == 0 {
                ties_a += 1;
            }
            if db == 0 {
                ties_b += 1;
            }
            match (da.signum() * db.signum()).signum() {
                1 => concordant += 1,
                -1 => discordant += 1,
                _ => {}
            }
        }
    }
    let n = rows.len() as i64;
    let n0 = n * (n - 1) / 2;
    let denominator = (((n0 - ties_a) as f64) * ((n0 - ties_b) as f64)).sqrt();
    if denominator == 0.0 {
        return 0.0; // one side fully tied: correlation undefined, report 0
    }
    (concordant - discordant) as f64 / denominator
}

fn verify(paths: &[String]) -> ExitCode {
    let mut problems = 0usize;
    for path in paths {
        problems += verify_shard(path);
    }
    if problems == 0 {
        println!("\nobs-tool verify: 0 problems");
        ExitCode::SUCCESS
    } else {
        println!("\nobs-tool verify: {problems} problem(s)");
        ExitCode::FAILURE
    }
}

fn verify_shard(path: &str) -> usize {
    let mut problems = 0usize;
    let mut fail = |message: String| {
        println!("  PROBLEM  {message}");
    };
    let mut reader = match ShardReader::open(path) {
        Ok(reader) => reader,
        Err(error) => {
            fail(format!("{path}: unreadable shard: {error}"));
            return 1;
        }
    };
    let provenance_count = reader.header().provenance.len();
    println!(
        "{path}: format v{}, {} provenance block(s)",
        reader.header().format_version,
        provenance_count
    );

    let (mut episodes, mut records, mut arms) = (0u64, 0u64, 0u64);
    loop {
        let episode = match reader.next_episode() {
            Ok(Some(episode)) => episode,
            Ok(None) => break,
            Err(error) => {
                fail(format!("{path}: episode {episodes}: undecodable: {error}"));
                problems += 1;
                break; // stream position is unrecoverable past a bad frame
            }
        };
        episodes += 1;
        records += episode.records.len() as u64;
        arms += episode
            .records
            .iter()
            .map(|r| r.moves.len() as u64)
            .sum::<u64>();
        problems += verify_episode(&episode, provenance_count, &mut fail);
    }
    println!("  {episodes} episode(s), {records} record(s), {arms} per-move row(s)");
    problems
}

fn verify_episode(
    episode: &Episode,
    provenance_count: usize,
    fail: &mut impl FnMut(String),
) -> usize {
    let id = episode.episode_id;
    let mut problems = 0usize;

    if episode.provenance_ref as usize >= provenance_count {
        fail(format!(
            "episode {id}: provenance_ref {} out of range ({provenance_count} blocks)",
            episode.provenance_ref
        ));
        problems += 1;
    }

    // In-search capture (design/16 §3.5.5): each record carries FULL position
    // identity (`canonical_key = raw_key`) with an EMPTY path — there is no
    // shared replay root to reconstruct/replay from. Verify each record's
    // position in isolation (reconstruct through FEN validation, recompute the
    // key) instead of the RootMoveAnalysis replay-root + path-replay path.
    if episode.kind == EpisodeKind::InSearchCapture {
        for (index, record) in episode.records.iter().enumerate() {
            if !record.position.path_from_episode_root.is_empty() {
                fail(format!(
                    "episode {id} record {index}: InSearchCapture must carry an empty path"
                ));
                problems += 1;
            }
            if let Err(error) = checked_board(&record.position) {
                fail(format!("episode {id} record {index}: {error}"));
                problems += 1;
            }
        }
        return problems;
    }

    let root = match checked_board(&episode.replay_root) {
        Ok(board) => board,
        Err(error) => {
            fail(format!("episode {id}: replay root: {error}"));
            return problems + 1;
        }
    };

    for (index, record) in episode.records.iter().enumerate() {
        if let Err(error) = verify_record(&root, record) {
            fail(format!("episode {id} record {index}: {error}"));
            problems += 1;
        }
    }
    problems
}

/// Reconstruct a stored position and confirm its stored identity is the
/// truth (recomputed key), not just self-consistent bytes.
fn checked_board(position: &PositionState) -> Result<Board, String> {
    let board = board_from_position(position)?;
    // raw_key: the unfolded placement+stm+castling+ep identity (the folded
    // board.key() would agree with a mis-keyed writer — D33 round 1, HIGH).
    if board.raw_key() != position.canonical_key {
        return Err(format!(
            "stored canonical_key {:#018x} != recomputed {:#018x}",
            position.canonical_key,
            board.raw_key()
        ));
    }
    Ok(board)
}

/// Replay the record's path from the episode root, requiring every step to
/// be legal and the destination to be the stored position.
fn verify_record(root: &Board, record: &chessengine::obs::Observation) -> Result<(), String> {
    let board = replay_path(root, &record.position.path_from_episode_root)?;
    let stored = &record.position;
    if board.raw_key() != stored.canonical_key {
        return Err(format!(
            "replayed key {:#018x} != stored canonical_key {:#018x}",
            board.raw_key(),
            stored.canonical_key
        ));
    }
    // Belt and braces beyond the key: exact occupancy + side to move.
    let replayed_occ: u64 = board
        .all_occupancy()
        .into_iter()
        .fold(0u64, |acc, sq| acc | (1u64 << sq.index()));
    if replayed_occ != stored.occ {
        return Err("replayed occupancy != stored occ".into());
    }
    if board.side_to_move() as u8 != stored.stm {
        return Err("replayed stm != stored stm".into());
    }
    Ok(())
}

fn dump(args: &[String]) -> ExitCode {
    let path = &args[0];
    let limit: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(3);
    let mut reader = match ShardReader::open(path) {
        Ok(reader) => reader,
        Err(error) => {
            eprintln!("obs-tool: {error}");
            return ExitCode::FAILURE;
        }
    };
    for (index, provenance) in reader.header().provenance.iter().enumerate() {
        println!(
            "provenance[{index}]: engine {} | gen {} | source {} | seed {} | hw {}",
            provenance.engine_version,
            provenance.generator_version,
            provenance.dataset_source,
            provenance.rng_seed,
            provenance.hardware_identity
        );
    }
    let mut shown = 0usize;
    while shown < limit {
        match reader.next_episode() {
            Ok(Some(episode)) => {
                shown += 1;
                let fen = fen_from_position(&episode.replay_root)
                    .unwrap_or_else(|error| format!("<invalid: {error}>"));
                println!(
                    "episode {} kind {:?} prov {} records {} | root {}",
                    episode.episode_id,
                    episode.kind,
                    episode.provenance_ref,
                    episode.records.len(),
                    fen
                );
                for record in episode.records.iter().take(4) {
                    println!(
                        "  depth {} window [{}, {}] eval {} best {:#06x} score {} nodes {} arms {}",
                        record.context.depth,
                        record.context.alpha,
                        record.context.beta,
                        record.context.static_eval,
                        record.search.best_move,
                        record.search.final_score,
                        record.search.nodes,
                        record.moves.len()
                    );
                }
            }
            Ok(None) => break,
            Err(error) => {
                eprintln!("obs-tool: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use chessengine::obs::convert::position_from_board;
    use chessengine::obs::{Completion, PerMoveObservation, SearchContext, SearchObservations};

    fn arm(mv: u16, score: i32, completion: Completion) -> PerMoveObservation {
        PerMoveObservation {
            mv,
            returned_score: score,
            completion,
            ..Default::default()
        }
    }

    #[test]
    fn export_value_cp_is_white_relative_and_capped() {
        // White to move (stm 0): score already White-relative.
        assert_eq!(value_cp_white(340, 0), 340);
        // Black to move (stm 1): a stm-relative +good-for-black flips negative.
        assert_eq!(value_cp_white(340, 1), -340);
        assert_eq!(value_cp_white(-340, 1), 340);
        // Mate encoding (±MATE_BASE-ish) collapses to ±cap, either side.
        assert_eq!(value_cp_white(31000 - 5, 0), VALUE_SCORE_CAP);
        assert_eq!(value_cp_white(31000 - 5, 1), -VALUE_SCORE_CAP);
        assert_eq!(value_cp_white(-(31000 - 5), 0), -VALUE_SCORE_CAP);
        // Ordinary large-but-sub-cap value is preserved.
        assert_eq!(value_cp_white(2000, 0), 2000);
    }

    /// `--filter-sfbinpack` must reproduce the trainer's five criteria
    /// exactly — this is the delta-2 regression tripwire (a canon corpus
    /// that skips any criterion re-poisons the loss with tactical labels).
    #[test]
    fn export_value_sfbinpack_filter_matches_trainer_criteria() {
        use chessengine::moves::Move;
        use chessengine::obs::convert::position_from_board;
        use chessengine::types::Square;

        // A quiet middlegame position, White to move, e2e4 available & quiet.
        let board =
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 3")
                .expect("valid fen");
        let mut position = position_from_board(&board, 0, Vec::new());
        position.game_ply = 20; // past the ply>=16 gate
        let mut obs = Observation {
            search: SearchObservations {
                bound: Bound::Exact,
                completion: Completion::Complete,
                final_score: 35,
                best_move: Move::normal(Square::E2, Square::E4).raw16(),
                ..Default::default()
            },
            ..Default::default()
        };

        // Baseline: every criterion satisfied -> no drop.
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_none(),
            "quiet normal row is kept"
        );

        // ply < 16 -> dropped.
        position.game_ply = 15;
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_some(),
            "ply<16 dropped"
        );
        position.game_ply = 20;

        // |raw score| > 10000 (a mate-ish label that the ±2500 clamp would
        // otherwise hide) -> dropped.
        obs.search.final_score = 30_995;
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_some(),
            "mate-scale score dropped"
        );
        obs.search.final_score = 35;

        // Capture best move -> dropped (Nxe5).
        obs.search.best_move = Move::normal(Square::F3, Square::E5).raw16();
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_some(),
            "capture best-move dropped"
        );

        // Non-Normal kind (promotion encoding) -> dropped.
        obs.search.best_move =
            Move::promotion(Square::E2, Square::E4, chessengine::types::PieceType::Queen).raw16();
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_some(),
            "non-normal best-move dropped"
        );

        // Null best move -> dropped, not emitted.
        obs.search.best_move = 0;
        assert!(
            sfbinpack_parity_drop(&obs, &position).is_some(),
            "null best-move dropped"
        );
        obs.search.best_move = Move::normal(Square::E2, Square::E4).raw16();

        // Side to move in check -> dropped (scholar's-mate-ish check).
        let checked =
            Board::from_fen("rnbqkbnr/ppp2Qpp/3p4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 3")
                .expect("valid fen");
        let mut checked_pos = position_from_board(&checked, 0, Vec::new());
        checked_pos.game_ply = 20;
        assert!(
            sfbinpack_parity_drop(&obs, &checked_pos).is_some(),
            "in-check dropped"
        );

        // Stats route to the right counters.
        let mut stats = SfbFilterStats::default();
        position.game_ply = 3;
        sfbinpack_parity_drop(&obs, &position).expect("drop")(&mut stats);
        assert_eq!((stats.ply, stats.total()), (1, 1));
    }

    #[test]
    fn export_value_quarantines_bounded_and_incomplete_references() {
        let mut obs = Observation {
            search: SearchObservations {
                bound: Bound::Exact,
                completion: Completion::Complete,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(value_reference_ok(&obs), "exact+complete is accepted");
        obs.search.bound = Bound::Lower; // an aspiration bound
        assert!(!value_reference_ok(&obs), "a bounded label is quarantined");
        obs.search.bound = Bound::Exact;
        obs.search.completion = Completion::Stopped; // timed out
        assert!(!value_reference_ok(&obs), "a partial search is quarantined");
    }

    /// Tau-b with heavy ties must not deflate toward zero (D33 round 1: the
    /// tau-a denominator counted tied pairs against agreement), and Stopped
    /// arms must be excluded from the statistic entirely.
    #[test]
    fn kendall_is_tie_aware_and_skips_stopped_arms() {
        // Perfect agreement on the comparable pairs, with one tied pair on
        // each side: tau-b must be exactly 1.0 (tau-a would report 5/6).
        let a = vec![
            arm(1, 100, Completion::Complete),
            arm(2, 50, Completion::Complete),
            arm(3, 50, Completion::Complete),
            arm(4, 10, Completion::Complete),
        ];
        let b = vec![
            arm(1, 90, Completion::Complete),
            arm(2, 40, Completion::Complete),
            arm(3, 40, Completion::Complete),
            arm(4, 5, Completion::Complete),
        ];
        assert!((kendall_tau(&a, &b) - 1.0).abs() < 1e-12);

        // A Stopped arm with a garbage score must not perturb the result.
        let mut poisoned = a.clone();
        poisoned.push(arm(5, -30_000, Completion::Stopped));
        let mut b_with = b.clone();
        b_with.push(arm(5, 25_000, Completion::Stopped));
        assert!((kendall_tau(&poisoned, &b_with) - 1.0).abs() < 1e-12);

        // Fully tied on one side: undefined, reported as 0.
        let flat = vec![
            arm(1, 7, Completion::Complete),
            arm(2, 7, Completion::Complete),
        ];
        assert_eq!(kendall_tau(&flat, &b[..2]), 0.0);
    }

    // ─── export-train fixtures ───────────────────────────────────────────

    fn prov(name: &str) -> Provenance {
        Provenance {
            engine_version: "test-engine".into(),
            generator_version: "test".into(),
            search_config: "test".into(),
            dataset_source: "test".into(),
            opening_source: "test".into(),
            hardware_identity: "test".into(),
            teacher_identity: Some(name.into()),
            ..Default::default()
        }
    }

    fn record(
        budget: u64,
        ref_completion: Completion,
        ref_score: i32,
        best: u16,
        arms: Vec<PerMoveObservation>,
    ) -> Observation {
        Observation {
            position: PositionState::default(),
            context: SearchContext {
                node_budget: Some(budget),
                ..Default::default()
            },
            search: SearchObservations {
                best_move: best,
                final_score: ref_score,
                completion: ref_completion,
                ..Default::default()
            },
            moves: arms,
        }
    }

    fn episode(root_fen: &str, id: u64, records: Vec<Observation>) -> Episode {
        let board = Board::from_fen(root_fen).unwrap();
        Episode {
            kind: EpisodeKind::RootMoveAnalysis,
            episode_id: id,
            provenance_ref: 0,
            records,
            replay_root: position_from_board(&board, 0, vec![]),
            ..Default::default()
        }
    }

    /// The synthetic fixture: deepest Complete tier is the value source, its
    /// Stopped arm is dropped, values/margins/best-flags are exact.
    #[test]
    fn export_train_emits_exact_value_margin_rows() {
        let mut builder = CorpusBuilder::new();
        let teacher = builder.intern_teacher(&prov("reckless")).unwrap();
        let (arm_a, arm_b, arm_c) = (0x0001u16, 0x0002u16, 0x0003u16);
        let ep = episode(
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            0,
            vec![
                record(
                    1000,
                    Completion::Complete,
                    10,
                    arm_a,
                    vec![
                        arm(arm_a, 8, Completion::Complete),
                        arm(arm_b, 5, Completion::Complete),
                    ],
                ),
                record(
                    4000,
                    Completion::Complete,
                    12,
                    arm_a,
                    vec![
                        arm(arm_a, 20, Completion::Complete),
                        arm(arm_b, 7, Completion::Complete),
                        arm(arm_c, -3, Completion::Stopped),
                    ],
                ),
            ],
        );
        builder.add_episode(&ep, teacher);

        assert_eq!(builder.rows.len(), 2, "two Complete arms; Stopped dropped");
        let (a, b) = (builder.rows[0], builder.rows[1]);
        // Deepest tier (4000) is the value source, not tier 1000.
        assert_eq!(a.tier, 4000);
        assert_eq!(a.ref_score, 12);
        // Arm A: max value + reference best.
        assert_eq!(a.mv_raw16, arm_a);
        assert_eq!(a.value, 20);
        assert_eq!(a.margin, 0);
        assert_eq!(a.is_max_value, 1);
        assert_eq!(a.is_ref_best, 1);
        assert_eq!(a.move_idx, 64); // raw16 1 -> from 1, to 0 -> 1*64+0
        // Arm B: value + margin vs best.
        assert_eq!(b.value, 7);
        assert_eq!(b.margin, 7 - 20);
        assert_eq!(b.is_max_value, 0);
        assert_eq!(b.is_ref_best, 0);
        assert_eq!(b.move_idx, 128); // raw16 2 -> from 2, to 0
        // Shared position metadata + absolute feats (a1 white K, h8 black K).
        assert_eq!(a.position_id, 0);
        assert_eq!(a.nmoves, 2);
        assert_eq!(a.teacher_id, teacher);
        assert_eq!(a.stm, 0);
        assert_eq!(a.feats[0], 320); // (0*6+5)*64 + 0
        assert_eq!(a.feats[1], 767); // (1*6+5)*64 + 63
        assert_eq!(a.feats[2], -1); // padded
        // Stats.
        assert_eq!(builder.stats.positions_in, 1);
        assert_eq!(builder.stats.positions_out, 1);
        assert_eq!(builder.stats.arms_out, 2);
        assert_eq!(builder.stats.arms_skipped_stopped, 1);
        assert_eq!(builder.stats.ref_best_missing, 0);
    }

    /// Same canonical_key across two shards: first occurrence wins, the second
    /// drops, and the drop is counted (not silently merged).
    #[test]
    fn export_train_dedups_by_canonical_key_across_shards() {
        let mut builder = CorpusBuilder::new();
        let t1 = builder.intern_teacher(&prov("reckless")).unwrap();
        let t2 = builder.intern_teacher(&prov("sf18")).unwrap();
        let make = || {
            episode(
                "7k/8/8/8/8/8/8/K7 w - - 0 1",
                0,
                vec![record(
                    4000,
                    Completion::Complete,
                    5,
                    0x0001,
                    vec![
                        arm(0x0001, 10, Completion::Complete),
                        arm(0x0002, -4, Completion::Complete),
                    ],
                )],
            )
        };
        builder.add_episode(&make(), t1); // shard A
        builder.add_episode(&make(), t2); // shard B: identical key

        assert_eq!(builder.stats.dedup_drops, 1);
        assert_eq!(builder.rows.len(), 2, "only shard A's two arms survive");
        assert!(builder.rows.iter().all(|r| r.teacher_id == t1));
        assert_eq!(builder.stats.positions_in, 2);
        assert_eq!(builder.stats.positions_out, 1);
    }

    /// A Stopped reference is not a value source: skip if none complete, but
    /// fall back to the deepest COMPLETE tier when a shallower one exists.
    #[test]
    fn export_train_completion_filter_and_tier_fallback() {
        let mut builder = CorpusBuilder::new();
        let teacher = builder.intern_teacher(&prov("reckless")).unwrap();

        // Only tier's reference Stopped -> skipped entirely.
        let stopped_only = episode(
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            0,
            vec![record(
                4000,
                Completion::Stopped,
                0,
                0x0001,
                vec![arm(0x0001, 10, Completion::Complete)],
            )],
        );
        builder.add_episode(&stopped_only, teacher);
        assert_eq!(builder.rows.len(), 0);
        assert_eq!(builder.stats.skip_no_complete_reference, 1);
        assert_eq!(builder.stats.positions_out, 0);

        // Deepest tier Stopped, shallower Complete -> use the shallower one.
        let fallback = episode(
            "k7/8/8/8/8/8/8/7K w - - 0 1",
            1,
            vec![
                record(
                    4000,
                    Completion::Complete,
                    3,
                    0x0001,
                    vec![
                        arm(0x0001, 9, Completion::Complete),
                        arm(0x0002, 1, Completion::Complete),
                    ],
                ),
                record(
                    16000,
                    Completion::Stopped,
                    0,
                    0x0001,
                    vec![arm(0x0001, 99, Completion::Complete)],
                ),
            ],
        );
        builder.add_episode(&fallback, teacher);
        assert_eq!(builder.rows.len(), 2);
        assert!(
            builder.rows.iter().all(|r| r.tier == 4000),
            "fell back to the deepest Complete tier, not the Stopped one"
        );
    }

    /// `--all-tiers` emits every Complete tier (not just the deepest), each
    /// tier's arms tagged with its budget and margined vs its own best arm;
    /// all tiers of one position share a single `position_id`.
    #[test]
    fn export_train_all_tiers_emits_every_complete_tier() {
        let ep = || {
            episode(
                "7k/8/8/8/8/8/8/K7 w - - 0 1",
                0,
                vec![
                    record(
                        4000,
                        Completion::Complete,
                        10,
                        0x0001,
                        vec![
                            arm(0x0001, 10, Completion::Complete),
                            arm(0x0002, -4, Completion::Complete),
                        ],
                    ),
                    record(
                        16000,
                        Completion::Complete,
                        20,
                        0x0001,
                        vec![
                            arm(0x0001, 20, Completion::Complete),
                            arm(0x0002, 5, Completion::Complete),
                        ],
                    ),
                ],
            )
        };

        // Deepest-only (default): just the 16000 tier's two arms.
        let mut deep = CorpusBuilder::new();
        let td = deep.intern_teacher(&prov("reckless")).unwrap();
        deep.add_episode_tiered(&ep(), td, false);
        assert_eq!(deep.rows.len(), 2);
        assert!(deep.rows.iter().all(|r| r.tier == 16000));
        assert_eq!(deep.stats.positions_out, 1);

        // All-tiers: both tiers, one position_id, per-tier best margin 0.
        let mut all = CorpusBuilder::new();
        let ta = all.intern_teacher(&prov("reckless")).unwrap();
        all.add_episode_tiered(&ep(), ta, true);
        assert_eq!(all.rows.len(), 4, "both Complete tiers emitted");
        assert_eq!(all.stats.positions_out, 1, "one position across tiers");
        assert!(all.rows.iter().all(|r| r.position_id == 0));
        let tiers: std::collections::BTreeSet<u32> = all.rows.iter().map(|r| r.tier).collect();
        assert_eq!(tiers, [4000, 16000].into_iter().collect());
        for tier in [4000u32, 16000] {
            let n_best = all
                .rows
                .iter()
                .filter(|r| r.tier == tier && r.margin == 0)
                .count();
            let n_max = all
                .rows
                .iter()
                .filter(|r| r.tier == tier && r.is_max_value == 1)
                .count();
            assert_eq!(n_best, 1, "exactly one margin-0 (best) arm per tier");
            assert_eq!(n_max, 1, "exactly one is_max_value per tier");
        }
        // Per-tier margins are vs THAT tier's best (-4-10=-14; 5-20=-15).
        let m4 = all
            .rows
            .iter()
            .find(|r| r.tier == 4000 && r.mv_raw16 == 0x0002)
            .unwrap()
            .margin;
        let m16 = all
            .rows
            .iter()
            .find(|r| r.tier == 16000 && r.mv_raw16 == 0x0002)
            .unwrap()
            .margin;
        assert_eq!(m4, -14);
        assert_eq!(m16, -15);
    }

    /// The `.npy` writer produces the exact layout `d40_policy.rs` reads
    /// (magic, u16 header length, data at 10+header_len) and round-trips the
    /// column payloads; the manifest names the versioned schema.
    #[test]
    fn export_train_npy_columns_round_trip() {
        fn read_npy_i32(path: &std::path::Path) -> Vec<i32> {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(&bytes[..6], b"\x93NUMPY");
            assert_eq!(bytes[6], 1, "npy v1");
            let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            bytes[10 + header_len..]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }

        let mut builder = CorpusBuilder::new();
        let teacher = builder.intern_teacher(&prov("reckless")).unwrap();
        let ep = episode(
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            0,
            vec![record(
                4000,
                Completion::Complete,
                12,
                0x0001,
                vec![
                    arm(0x0001, 20, Completion::Complete),
                    arm(0x0002, 7, Completion::Complete),
                ],
            )],
        );
        builder.add_episode(&ep, teacher);
        let (per_teacher, tier_hist) = corpus_aggregates(&builder.rows, &builder.teachers);
        let dir = std::env::temp_dir().join(format!("archon-train-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_corpus(
            &dir,
            &builder.rows,
            &builder.teachers,
            &builder.stats,
            &["synthetic".to_string()],
            &per_teacher,
            &tier_hist,
            false,
        )
        .unwrap();

        assert_eq!(read_npy_i32(&dir.join("value.npy")), vec![20, 7]);
        assert_eq!(read_npy_i32(&dir.join("margin.npy")), vec![0, -13]);
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(manifest.contains("archon.obs-train.rootmove-value-v1"));
        assert!(manifest.contains("\"rows\": 2"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
