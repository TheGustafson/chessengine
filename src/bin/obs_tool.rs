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
use chessengine::movegen::generate_legal;
use chessengine::moves::{Move, MoveList};
use chessengine::obs::convert::{board_from_position, fen_from_position};
use chessengine::obs::{Episode, PositionState, ShardReader};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "verify" && !rest.is_empty() => verify(rest),
        Some((cmd, rest)) if cmd == "dump" && !rest.is_empty() => dump(rest),
        _ => {
            eprintln!("usage: obs-tool verify <shard.ceobs>...");
            eprintln!("       obs-tool dump <shard.ceobs> [episodes]");
            ExitCode::FAILURE
        }
    }
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
    if board.key() != position.canonical_key {
        return Err(format!(
            "stored canonical_key {:#018x} != recomputed {:#018x}",
            position.canonical_key,
            board.key()
        ));
    }
    Ok(board)
}

/// Replay the record's path from the episode root, requiring every step to
/// be legal and the destination to be the stored position.
fn verify_record(root: &Board, record: &chessengine::obs::Observation) -> Result<(), String> {
    let mut board = root.clone();
    for (ply, &raw) in record.position.path_from_episode_root.iter().enumerate() {
        let mv = Move::from_raw16(raw);
        let mut legal = MoveList::new();
        generate_legal(&board, &mut legal);
        if !legal.contains(&mv) {
            return Err(format!("path ply {ply}: move {raw:#06x} is not legal here"));
        }
        let gives_check = board.gives_check(mv);
        board.make_move(mv, gives_check);
    }
    let stored = &record.position;
    if board.key() != stored.canonical_key {
        return Err(format!(
            "replayed key {:#018x} != stored canonical_key {:#018x}",
            board.key(),
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
