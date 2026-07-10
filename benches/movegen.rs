//! Criterion micro-bench for move generation (design/14 §4 step 4, doc 05
//! §6). Never a merge gate; baselines are recorded in
//! `docs/perf-baseline.md`. Run with the documented fast configuration:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench --bench movegen
//! ```
//!
//! What it measures, per doc 05 §6: `generate::<NON_EVASIONS>` and
//! `generate::<CAPTURES>` over the six standard perft positions (one
//! iteration = all six, so ns/position = iteration time ÷ 6), plus
//! `generate_legal` on the same six — the legality-filter overhead perft
//! pays on top of raw generation. `Throughput::Elements` is the total
//! move count each pass emits, so the criterion "thrpt" column reads in
//! moves/sec (Melem/s = moves/µs).

use std::hint::black_box;

use chessengine::movegen::{generate_captures, generate_legal, generate_non_evasions};
use chessengine::{Board, MoveList, START_FEN};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// The six standard perft positions (design/05 §5.3 / design/06 §3.4).
const POSITIONS: &[&str] = &[
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

fn boards() -> Vec<Board> {
    POSITIONS
        .iter()
        .map(|fen| Board::from_fen(fen).expect("bench FEN is valid"))
        .collect()
}

/// One benchmark: run `generator` on all six boards into a reused
/// `MoveList` (the doc 05 §6.1 reuse-and-`clear` pattern), summing lengths
/// so no generation is dead code. Throughput = total moves per pass.
fn bench_generation(c: &mut Criterion, name: &str, generator: impl Fn(&Board, &mut MoveList)) {
    let boards = boards();
    // Deterministic move total for the throughput denominator.
    let total: usize = boards
        .iter()
        .map(|b| {
            let mut list = MoveList::new();
            generator(b, &mut list);
            list.len()
        })
        .sum();

    let mut g = c.benchmark_group("movegen");
    g.throughput(Throughput::Elements(total as u64));
    g.bench_function(name, |b| {
        let mut list = MoveList::new();
        b.iter(|| {
            let mut moves = 0usize;
            for board in black_box(&boards) {
                list.clear();
                generator(board, &mut list);
                moves += list.len();
            }
            moves
        })
    });
    g.finish();
}

fn bench_movegen(c: &mut Criterion) {
    bench_generation(c, "non_evasions_x6", generate_non_evasions);
    bench_generation(c, "captures_x6", generate_captures);
    bench_generation(c, "legal_x6", generate_legal);
}

criterion_group!(benches, bench_movegen);
criterion_main!(benches);
