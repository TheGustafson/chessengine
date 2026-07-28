// Probe C helper: sample entries from an SF binpack file with FULL-FILE
// coverage, without decoding every chunk.
//
//   sample FILE COUNT SEED
//
// Pass 1 scans only the 8-byte chunk headers ("BINP" + u32le payload size),
// seeking over payloads, so the whole multi-GB file is indexed in ~1 syscall
// per chunk. Pass 2 decodes COUNT chunks at evenly-spaced chunk indices and
// emits ONE entry per selected chunk (deterministic splitmix64-seeded index
// within the chunk), guaranteeing samples come from many different games
// (binpack is game-contiguous within chains; distinct chunks are far apart).
//
// Output: JSONL on stdout — first a {"type":"meta",...} line, then one
// {"type":"row",...} line per sample. Read-only; discardable scaffold.
use std::{
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    process::ExitCode,
};

use sfbinpack::parse_chunk;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(path), Some(count), Some(seed)) = (
        args.next(),
        args.next().and_then(|t| t.parse::<usize>().ok()),
        args.next().and_then(|t| t.parse::<u64>().ok()),
    ) else {
        eprintln!("usage: sample FILE COUNT SEED");
        return ExitCode::from(2);
    };

    let mut file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("open {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

    // Pass 1: index chunk (payload_offset, payload_size) pairs.
    let mut chunks: Vec<(u64, u32)> = Vec::new();
    let mut pos = 0_u64;
    let mut tail_note: Option<String> = None;
    let mut header = [0_u8; 8];
    while pos + 8 <= bytes {
        if let Err(error) = file
            .seek(SeekFrom::Start(pos))
            .and_then(|_| file.read_exact(&mut header))
        {
            tail_note = Some(format!("header read error at {pos}: {error}"));
            break;
        }
        if &header[..4] != b"BINP" {
            tail_note = Some(format!("bad magic at offset {pos}"));
            break;
        }
        let size = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if pos + 8 + u64::from(size) > bytes {
            tail_note = Some(format!(
                "truncated final chunk at offset {pos} (declared {size}, file ends)"
            ));
            break;
        }
        chunks.push((pos + 8, size));
        pos += 8 + u64::from(size);
    }
    if tail_note.is_none() && pos != bytes {
        tail_note = Some(format!("{} unparsed trailing bytes", bytes - pos));
    }
    let n_chunks = chunks.len();
    if n_chunks == 0 {
        eprintln!("no complete chunks in {path}");
        return ExitCode::FAILURE;
    }

    println!(
        "{{\"type\":\"meta\",\"file\":\"{}\",\"bytes\":{},\"complete_chunks\":{},\"indexed_payload_end\":{},\"tail_note\":{},\"count\":{},\"seed\":{},\"selection\":\"chunk[floor(i*n_chunks/count)] (round-robin if count>n_chunks); entry=splitmix64(seed^chunk_index^(i<<32)) % entries\"}}",
        json_escape(&path),
        bytes,
        n_chunks,
        pos,
        tail_note
            .as_ref()
            .map(|t| format!("\"{}\"", json_escape(t)))
            .unwrap_or_else(|| "null".to_string()),
        count,
        seed,
    );

    let mut payload = Vec::new();
    for i in 0..count {
        let ci = if count <= n_chunks {
            i * n_chunks / count
        } else {
            i % n_chunks
        };
        let (offset, size) = chunks[ci];
        payload.resize(size as usize, 0);
        if let Err(error) = file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(&mut payload))
        {
            eprintln!("chunk {ci} read error: {error}");
            return ExitCode::FAILURE;
        }
        let entries = parse_chunk(&payload);
        if entries.is_empty() {
            eprintln!("chunk {ci} decoded to zero entries");
            return ExitCode::FAILURE;
        }
        let ei = (splitmix64(seed ^ (ci as u64) ^ ((i as u64) << 32)) % entries.len() as u64)
            as usize;
        let entry = &entries[ei];
        let stm_white = entry.pos.side_to_move().ordinal() == 0;
        let fen = match entry.pos.fen() {
            Ok(fen) => fen,
            Err(error) => {
                eprintln!("chunk {ci} entry {ei}: fen error: {error:?}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "{{\"type\":\"row\",\"sample_index\":{},\"chunk_index\":{},\"chunk_offset\":{},\"entries_in_chunk\":{},\"entry_index\":{},\"fen\":\"{}\",\"move_uci\":\"{}\",\"score_stm\":{},\"ply\":{},\"result_stm\":{},\"rule50\":{},\"stm\":\"{}\",\"in_check\":{}}}",
            i,
            ci,
            offset,
            entries.len(),
            ei,
            json_escape(&fen),
            entry.mv.as_uci(),
            entry.score,
            entry.ply,
            entry.result,
            entry.pos.rule50_counter(),
            if stm_white { "w" } else { "b" },
            entry.pos.is_checked(entry.pos.side_to_move()),
        );
    }
    ExitCode::SUCCESS
}
