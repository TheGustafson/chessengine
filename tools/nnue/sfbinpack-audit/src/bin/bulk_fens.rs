// Bulk FEN extractor for the positions-vs-labels experiment
// (docs/research/value-foundation-state.md): emit COUNT candidate positions
// from an SF binpack, evenly spread across every chunk, as TSV on stdout:
//
//     FEN \t result_white \t ply
//
// result_white is 1.0/0.5/0.0 from White's perspective (binpack stores it
// stm-relative). Within a chunk, entries are picked at a fixed stride large
// enough that consecutive picks land in different games (games are 50-150
// contiguous entries; strides here are hundreds+), which enforces the
// sf-100 datagen's per-game position cap organically. Read-only scaffold.
//
//     bulk_fens FILE COUNT
use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    process::ExitCode,
};

use sfbinpack::parse_chunk;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(path), Some(count)) = (args.next(), args.next().and_then(|t| t.parse::<usize>().ok()))
    else {
        eprintln!("usage: bulk_fens FILE COUNT");
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

    // Pass 1: index chunks by scanning 8-byte headers only.
    let mut chunks: Vec<(u64, u32)> = Vec::new();
    let mut pos = 0_u64;
    let mut header = [0_u8; 8];
    while pos + 8 <= bytes {
        if file.seek(SeekFrom::Start(pos)).is_err() || file.read_exact(&mut header).is_err() {
            break;
        }
        if &header[0..4] != b"BINP" {
            eprintln!("bad chunk magic at offset {pos}");
            return ExitCode::FAILURE;
        }
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if pos + 8 + u64::from(size) > bytes {
            break; // incomplete tail
        }
        chunks.push((pos + 8, size));
        pos += 8 + u64::from(size);
    }
    if chunks.is_empty() {
        eprintln!("no complete chunks in {path}");
        return ExitCode::FAILURE;
    }
    let per_chunk = count.div_ceil(chunks.len());
    eprintln!(
        "indexed {} chunks; emitting ~{per_chunk} positions/chunk toward {count}",
        chunks.len()
    );

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut emitted = 0_usize;
    for &(offset, size) in &chunks {
        if emitted >= count {
            break;
        }
        let mut payload = vec![0_u8; size as usize];
        if file.seek(SeekFrom::Start(offset)).is_err() || file.read_exact(&mut payload).is_err() {
            eprintln!("chunk read error at {offset}");
            return ExitCode::FAILURE;
        }
        let entries = parse_chunk(&payload);
        if entries.is_empty() {
            continue;
        }
        let stride = (entries.len() / per_chunk).max(200);
        let mut index = stride / 2;
        while index < entries.len() && emitted < count {
            let entry = &entries[index];
            index += stride;
            let fen = match entry.pos.fen() {
                Ok(fen) => fen,
                Err(_) => continue,
            };
            // result is stm-relative (-1/0/1); convert to White-relative text.
            let white_result = if entry.pos.side_to_move().ordinal() == 0 {
                entry.result
            } else {
                -entry.result
            };
            let result_str = match white_result {
                1 => "1.0",
                0 => "0.5",
                -1 => "0.0",
                _ => continue,
            };
            if writeln!(out, "{fen}\t{result_str}\t{}", entry.ply).is_err() {
                return ExitCode::FAILURE;
            }
            emitted += 1;
        }
    }
    eprintln!("emitted {emitted} candidate positions");
    ExitCode::SUCCESS
}
