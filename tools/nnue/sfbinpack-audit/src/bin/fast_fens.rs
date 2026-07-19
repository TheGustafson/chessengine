// Fast FEN extractor for the SF18-relabel fleet. Emits ~COUNT positions spread
// across the binpack as TSV 'FEN\tresult_white\tply' on stdout.
//
// Key difference vs `bulk_fens`: bulk_fens decodes EVERY chunk of the 40 GB file
// to draw an even sample, so a 20M extract costs the same as decoding all ~2B
// positions (~45 min). This instead **selects every Kth chunk** and decodes ONLY
// those (~COUNT positions worth), emitting all of each selected chunk — same
// spread across the file, but ~1/K of the decode work. K is derived from the
// first chunk's entry count. Chunk headers are walked with cheap explicit reads
// (not an mmap page-fault storm). Order is not preserved (irrelevant for relabel).
//
//     fast_fens FILE COUNT
use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    process::ExitCode,
};

use sfbinpack::parse_chunk;

fn read_chunk(file: &mut File, offset: u64, size: usize) -> Option<Vec<u8>> {
    let mut payload = vec![0u8; size];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut payload).ok()?;
    Some(payload)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(path), Some(count)) =
        (args.next(), args.next().and_then(|t| t.parse::<usize>().ok()))
    else {
        eprintln!("usage: fast_fens FILE COUNT  (TSV 'FEN\\tresult_white\\tply' to stdout)");
        return ExitCode::from(2);
    };

    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

    // Index chunks by reading only the 8-byte headers, seeking past payloads.
    let mut chunks: Vec<(u64, usize)> = Vec::new();
    let mut pos = 0u64;
    let mut header = [0u8; 8];
    while pos + 8 <= bytes {
        if file.seek(SeekFrom::Start(pos)).is_err() || file.read_exact(&mut header).is_err() {
            break;
        }
        if &header[0..4] != b"BINP" {
            eprintln!("bad chunk magic at offset {pos}");
            return ExitCode::FAILURE;
        }
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if pos + 8 + size as u64 > bytes {
            break;
        }
        chunks.push((pos + 8, size));
        pos += 8 + size as u64;
    }
    if chunks.is_empty() {
        eprintln!("no complete chunks in {path}");
        return ExitCode::FAILURE;
    }

    // Estimate entries/chunk from the first chunk -> select every Kth chunk so we
    // decode only ~COUNT positions instead of the whole file.
    let first = read_chunk(&mut file, chunks[0].0, chunks[0].1)
        .map(|p| parse_chunk(&p).len())
        .unwrap_or(0)
        .max(1);
    let total_est = first.saturating_mul(chunks.len());
    let stride_chunks = (total_est / count.max(1)).max(1);
    eprintln!(
        "indexed {} chunks (~{first} entries/chunk, ~{total_est} total); every {stride_chunks}th chunk toward {count}",
        chunks.len()
    );

    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut emitted = 0usize;
    let mut idx = 0usize;
    while idx < chunks.len() && emitted < count {
        let (offset, size) = chunks[idx];
        idx += stride_chunks;
        let Some(payload) = read_chunk(&mut file, offset, size) else { continue };
        for entry in parse_chunk(&payload) {
            if emitted >= count {
                break;
            }
            let Ok(fen) = entry.pos.fen() else { continue };
            let stm_white = entry.pos.side_to_move().ordinal() == 0;
            let white_result = if stm_white { entry.result } else { -entry.result };
            let result_str = match white_result {
                1 => "1.0",
                0 => "0.5",
                -1 => "0.0",
                _ => continue,
            };
            // White-relative original (5k-node) label, for label-quality comparison
            // against the fleet's SF18@50k relabel (which is also White-relative).
            let white_score = if stm_white { entry.score } else { -entry.score };
            if writeln!(out, "{fen}\t{result_str}\t{}\t{white_score}", entry.ply).is_err() {
                return ExitCode::FAILURE;
            }
            emitted += 1;
        }
    }
    eprintln!("emitted {emitted} positions");
    ExitCode::SUCCESS
}
