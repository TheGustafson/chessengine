//! Criterion macro-bench for bulk-counted perft (design/14 §4 step 3,
//! design/06 §6). Never a merge gate; baselines are recorded in
//! `docs/perf-baseline.md`. Run with the documented fast configuration:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench --bench perft
//! ```
//!
//! Both benchmarks use `Throughput::Elements(<known node count>)`, so the
//! criterion "thrpt" column reads directly in nodes/sec — compare it
//! against the D24 targets (≥ 100 Mnps floor, 250 Mnps stretch, doc 06
//! §6). The counts are the perft-suite oracle values; each iteration
//! asserts them, so a movegen regression fails the bench instead of
//! producing a fast wrong number.

use std::hint::black_box;

use chessengine::perft::perft;
use chessengine::{Board, START_FEN};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// One perft benchmark: `fen` at `depth`, throughput = the oracle count.
fn bench_perft(c: &mut Criterion, name: &str, fen: &str, depth: u32, nodes: u64) {
    let mut board = Board::from_fen(fen).expect("bench FEN is valid");
    let mut g = c.benchmark_group("perft");
    g.throughput(Throughput::Elements(nodes));
    // Whole-tree iterations are ~10-15 ms; a smaller sample count keeps the
    // bench in the seconds range without hurting the point estimate.
    g.sample_size(30);
    g.bench_function(name, |b| {
        b.iter(|| {
            let count = perft(black_box(&mut board), depth);
            assert_eq!(count, nodes, "perft regression inside the bench");
            count
        })
    });
    g.finish();
}

/// design/14 §4 step 3: startpos d5 and kiwipete d4 (≈ 4.9 M / 4.1 M
/// nodes — big enough to average over the whole tree shape, small enough
/// for criterion's default measurement window).
fn bench_bulk_perft(c: &mut Criterion) {
    bench_perft(c, "startpos_d5", START_FEN, 5, 4_865_609);
    bench_perft(c, "kiwipete_d4", KIWIPETE, 4, 4_085_603);
}

criterion_group!(benches, bench_bulk_perft);
criterion_main!(benches);
