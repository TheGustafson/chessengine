//! canon_filter — apply the sfbinpack QUIET filter to the SF18-canonical value
//! corpus (`FEN | cp | wdl` lines on stdin → filtered lines on stdout).
//!
//! ★ SUPERSEDED for shard-based corpora: `obs-tool export-value
//! --filter-sfbinpack` applies the FULL trainer-parity filter (ply≥16,
//! !in-check, |raw score|≤10000 pre-clamp, best-move Normal non-capture) at
//! export, where the `.ceobs` best_move is available. Use that whenever shards
//! exist. This tool is for text-only corpora with no shards behind them, and it
//! is a PARTIAL filter: it enforces only "side-to-move not in check", plus a
//! `|cp| <= 10000` test that is a NO-OP on export-value output (the export
//! clamps cp to ±2500 first, so mate-scale labels arrive pre-collapsed and
//! pass). It misses capture-best and clamp-hidden mate labels.
//!
//!   canon_filter < corpus.txt > corpus.quiet.txt

use chessengine::Board;
use std::io::{self, BufRead, BufWriter, Write};

fn main() {
    let stdin = io::stdin();
    let mut out = BufWriter::new(io::stdout().lock());
    let (mut kept, mut d_check, mut d_cp, mut d_parse) = (0u64, 0u64, 0u64, 0u64);
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let mut it = line.split(" | ");
        let Some(fen) = it.next() else { continue };
        let cp: i32 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        if cp.abs() > 10_000 {
            d_cp += 1;
            continue;
        }
        match Board::from_fen(fen) {
            Ok(b) if b.checkers().is_empty() => {
                writeln!(out, "{line}").expect("write");
                kept += 1;
            }
            Ok(_) => d_check += 1,
            Err(_) => d_parse += 1,
        }
    }
    let total = kept + d_check + d_cp + d_parse;
    eprintln!(
        "canon_filter: kept={kept} ({:.1}%) dropped_incheck={d_check} dropped_cp={d_cp} dropped_parse={d_parse} of {total}",
        100.0 * kept as f64 / total.max(1) as f64
    );
}
