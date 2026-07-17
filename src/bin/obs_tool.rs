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
use chessengine::obs::{Episode, PositionState, ShardReader};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "verify" && !rest.is_empty() => verify(rest),
        Some((cmd, rest)) if cmd == "dump" && !rest.is_empty() => dump(rest),
        Some((cmd, rest)) if cmd == "stability" && !rest.is_empty() => stability(rest),
        Some((cmd, rest)) if cmd == "disagree" && rest.len() == 2 => disagree(rest),
        Some((cmd, rest)) if cmd == "export" && !rest.is_empty() => export(rest),
        Some((cmd, rest)) if cmd == "rank" && !rest.is_empty() => rank(rest),
        _ => {
            eprintln!("usage: obs-tool verify <shard.ceobs>...");
            eprintln!("       obs-tool dump <shard.ceobs> [episodes]");
            eprintln!("       obs-tool stability <shard.ceobs> [tie-margin-cp (default 20)]");
            eprintln!("       obs-tool disagree <shardA.ceobs> <shardB.ceobs>");
            eprintln!("       obs-tool export <shard.ceobs>   (per-arm JSONL on stdout)");
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

/// The design/19 §8.1 CLASSICAL-ORDERING baseline: where does the teacher's
/// reference best move sit in ARCHON's own MovePicker order (fresh
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
    use chessengine::obs::{Completion, PerMoveObservation};

    fn arm(mv: u16, score: i32, completion: Completion) -> PerMoveObservation {
        PerMoveObservation {
            mv,
            returned_score: score,
            completion,
            ..Default::default()
        }
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
}
