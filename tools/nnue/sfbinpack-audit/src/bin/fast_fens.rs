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
//     fast_fens FILE COUNT [filter|nofilter] [keys.txt]
//
// KEYS MODE (2026-07-21): a 4th arg names a file of normalized-FEN keys
// ("board stm castling ep", one per line — e.g. cut -f1 canon.sorted). The
// tool then decodes EVERY chunk (COUNT still caps emissions; pass 0 for
// unlimited) and emits only entries whose normalized key is in the set,
// FNV-hashed to u64 (~16 B/key in RAM; a 77M-key set ≈ 1.2 GB). Built for the
// full-coverage WDL join: two stratified extractions of the same binpack
// overlap only partially (measured 35.6%), so joining a relabel corpus
// against a sampled extraction silently drops rows — keys mode recovers ~all
// of them in one full pass. Hash collisions only ever ADD a spurious row,
// which the downstream sort+join re-verifies against the full key anyway.
use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    process::ExitCode,
};

use sfbinpack::chess::piecetype::PieceType;
use sfbinpack::chess::r#move::MoveType;
use sfbinpack::parse_chunk;

/// The exact data-quality gate the proven sfbinpack training path enforces at
/// load time (`tools/nnue/bullet/d40_features.rs` `sfbinpack_filter`): a *static*
/// NNUE cannot represent a search score on an in-check / capture-pending / very
/// early position (the label describes the tactical resolution, not the board),
/// so those become irreducible high-magnitude noise. Corpus builds that dumped
/// every position bypassed this and poisoned the eval surface exactly where
/// search visits (Fable diagnosis 2026-07-20). Replicated here so materialised
/// corpora keep recipe parity.
fn passes_quiet_filter(entry: &sfbinpack::TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn read_chunk(file: &mut File, offset: u64, size: usize) -> Option<Vec<u8>> {
    let mut payload = vec![0u8; size];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut payload).ok()?;
    Some(payload)
}

/// FNV-1a over a normalized-FEN key ("board stm castling ep"). Matches nothing
/// else in the codebase — keys-mode is self-contained (writer and reader are
/// both this function), so the only contract is internal consistency.
fn fnv1a(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// First four space-separated fields of a FEN (drop the clocks so keys match
/// across a relabel export and a binpack extraction).
fn normalized_key(fen: &str) -> &str {
    let mut spaces = 0usize;
    for (i, b) in fen.bytes().enumerate() {
        if b == b' ' {
            spaces += 1;
            if spaces == 4 {
                return &fen[..i];
            }
        }
    }
    fen
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(path), Some(count)) =
        (args.next(), args.next().and_then(|t| t.parse::<usize>().ok()))
    else {
        eprintln!(
            "usage: fast_fens FILE COUNT [filter|nofilter] [keys.txt]  (TSV 'FEN\\tresult_white\\tply\\tscore' to stdout)\n  \
             filter: apply the sfbinpack quiet-position gate (ply>=16, not-in-check, |score|<=10000, best-move-normal-non-capture)\n  \
             keys.txt: normalized-FEN keys ('board stm castling ep' per line) -> full scan, emit only matches; COUNT=0 = no cap"
        );
        return ExitCode::from(2);
    };
    // The filter mode is EXPLICIT-REQUIRED (2026-07-21): an implicit default
    // here is how the 53%-poisoned extractions happened — a caller who doesn't
    // say which distribution they want shouldn't get one silently. `filter` =
    // trainer-parity quiet gate (training-grade extraction); `nofilter` = raw
    // (WDL-join key coverage, audits — NOT for direct training use).
    let filter = match args.next().as_deref() {
        Some("filter") => true,
        Some("nofilter") => false,
        other => {
            eprintln!(
                "fast_fens: third arg must be 'filter' or 'nofilter' (got {other:?}) — \
                 state the distribution you want explicitly"
            );
            return ExitCode::from(2);
        }
    };
    if filter {
        eprintln!("quiet filter ON: dropping in-check / capture-pending / early-ply positions");
    } else {
        eprintln!("quiet filter OFF: raw extraction — NOT training-grade for a scalar value loss");
    }
    // Optional keys file switches to full-scan lookup mode (see header).
    let keys: Option<std::collections::HashSet<u64>> = match args.next() {
        Some(keys_path) => {
            let text = match std::fs::read_to_string(&keys_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("read keys {keys_path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let set: std::collections::HashSet<u64> =
                text.lines().filter(|l| !l.is_empty()).map(fnv1a).collect();
            eprintln!("keys mode: {} keys loaded from {keys_path} (full scan)", set.len());
            Some(set)
        }
        None => None,
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
    // keys mode scans every chunk (the whole point is coverage); COUNT=0 lifts
    // the emission cap in either mode.
    let cap = if count == 0 { usize::MAX } else { count };
    let stride_chunks =
        if keys.is_some() { 1 } else { (total_est / count.max(1)).max(1) };
    eprintln!(
        "indexed {} chunks (~{first} entries/chunk, ~{total_est} total); every {stride_chunks}th chunk toward {count}",
        chunks.len()
    );

    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut emitted = 0usize;
    let mut skipped = 0usize;
    let mut idx = 0usize;
    while idx < chunks.len() && emitted < cap {
        let (offset, size) = chunks[idx];
        idx += stride_chunks;
        if let Some(set) = &keys
            && idx.is_multiple_of(2048)
        {
            eprintln!("  scan {idx}/{} chunks, {emitted} matched of {} keys", chunks.len(), set.len());
        }
        let Some(payload) = read_chunk(&mut file, offset, size) else { continue };
        for entry in parse_chunk(&payload) {
            if emitted >= cap {
                break;
            }
            if filter && !passes_quiet_filter(&entry) {
                skipped += 1;
                continue;
            }
            let Ok(fen) = entry.pos.fen() else { continue };
            if let Some(set) = &keys
                && !set.contains(&fnv1a(normalized_key(&fen)))
            {
                continue;
            }
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
    if filter {
        let seen = emitted + skipped;
        let pct = if seen > 0 { 100.0 * skipped as f64 / seen as f64 } else { 0.0 };
        eprintln!("emitted {emitted} positions ({skipped} dropped by quiet filter = {pct:.1}%)");
    } else {
        eprintln!("emitted {emitted} positions");
    }
    ExitCode::SUCCESS
}
