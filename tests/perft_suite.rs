//! Integration perft suite — the D24 movegen/make-unmake gate
//! (design/06 §3.5, tiers 2-3, plus a fast-tier shape; the tier split
//! here is data-driven, see the node-cap notes below).
//!
//! The fast tier runs in plain `cargo test`; the full and extended tiers
//! keep permanent release-tier ignores and are invoked selectively by name
//! (design/06 §3.5, TESTING.md).
//!
//! Data and loader come from the shared fixture module:
//! `tests/data/perft.epd` parsed by `common::load_perft_epd()` into
//! `common::PerftRecord { fen, counts: Vec<(depth, nodes)> }`.

mod common;

use chessengine::perft::perft;

/// Fast-tier cut: design/06 §3.4 pins a per-case `fast_depth`; with the
/// suite as EPD data the equivalent data-driven rule is a per-entry node
/// cap. 700_000 admits the largest fast-tier entry in the doc's tables
/// (cpw3 d5 = 674,624) and keeps the tier's total near the ~2.1 M-node
/// budget of design/06 §3.5 tier 1.
const FAST_NODE_CAP: u64 = 700_000;

/// Full/extended cut: the largest full-tier count is cpw4 d6 =
/// 706,045,033; the extended tier starts in the billions (design/06
/// §3.4-3.5).
const FULL_NODE_CAP: u64 = 1_000_000_000;

/// Run every `(depth, expected)` entry selected by `keep`, asserting the
/// exact known counts from tests/data/perft.epd.
fn run_tier(keep: impl Fn(u64) -> bool) {
    let records = common::load_perft_epd().unwrap_or_else(|e| panic!("{e}"));
    assert!(!records.is_empty(), "perft.epd contains no cases");
    for record in &records {
        let mut board = common::board(&record.fen);
        for &(depth, expected) in &record.counts {
            if !keep(expected) {
                continue;
            }
            assert_eq!(
                perft(&mut board, depth),
                expected,
                "perft mismatch: fen '{}' depth {depth}",
                record.fen
            );
        }
    }
}

/// Fast tier: cheap enough for plain `cargo test`.
#[test]
fn perft_fast_tier() {
    run_tier(|nodes| nodes <= FAST_NODE_CAP);
}

/// Tier 2 (full): every count up to the full-tier cap, release only
/// (design/06 §3.5: `cargo test --release -- --ignored perft_full_suite`,
/// budget < 30 s).
#[test]
#[ignore = "full tier: cargo test --release -- --ignored perft_full_suite"]
fn perft_full_suite() {
    run_tier(|nodes| nodes <= FULL_NODE_CAP);
}

/// Tier 3 (extended): the multi-billion-node depths, if present in the
/// EPD (design/06 §3.5: release only, budget < 4 min, run before releases
/// and after movegen/make-unmake rewrites).
#[test]
#[ignore = "extended tier: cargo test --release -- --ignored perft_extended"]
fn perft_extended() {
    run_tier(|nodes| nodes > FULL_NODE_CAP);
}
