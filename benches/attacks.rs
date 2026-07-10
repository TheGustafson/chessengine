//! Criterion micro-bench for the `attacks` module (design/03 §6, design/14
//! §3.3). Never a merge gate; baselines are recorded in
//! `docs/perf-baseline.md`. Run with the documented fast configuration:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench --bench attacks
//! ```
//!
//! What it measures, per doc 03 §6 (throughput mode — a hot-cache stream of
//! pipelined lookups, which is how movegen/SEE/eval will issue them):
//!
//! - **`*_hoisted`** — rook/bishop/queen magic lookups through a hoisted
//!   `&'static AttackTables` handle, the hot-loop pattern of doc 03 §3.7.
//!   Targets: ≤ 5 ns per rook/bishop lookup, ≤ 10 ns per queen lookup.
//! - **`*_free_fn`** — the same through the convenience free functions,
//!   which re-deref the `LazyLock` (atomic acquire load + branch) on every
//!   call. The delta vs. the hoisted rows is the measured cost that the
//!   hoisting guidance rests on.
//! - **Init time** — the full magic search, timed once around the first
//!   `tables()` touch and printed before the criterion runs (a `LazyLock`
//!   can only initialize once per process, so this is a one-shot number,
//!   not a statistic). Target: ≤ 25 ms (doc 03 §6).

use std::hint::black_box;
use std::time::Instant;

use chessengine::attacks::{self, AttackTables};
use chessengine::{Bitboard, Square};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group};

/// (square, occupancy) pairs per iteration: 1024 × 16 B = 16 KB, so the
/// stream itself stays cache-resident and the numbers price the table path
/// (magic metadata load, multiply, dependent table load, bounds check) —
/// not stream misses. Doc 03 §6 pitfall 5: cold-cache lookups are ~3-4x
/// worse by design; this bench is deliberately the hot-cache measurement.
const STREAM_LEN: usize = 1024;

/// SplitMix64 — deterministic occupancies without a `rand` dependency (D15;
/// same pattern as the unit tests in `src/attacks.rs` and `src/bitboard.rs`).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Mixed-density random occupancy stream — uniform (~32 bits set), sparse
/// (~8 bits, endgame-like), and dense (~48 bits) in rotation, mirroring the
/// doc 03 §5.3 validation mix. Off-mask bits are the norm here, exercising
/// the `& mask` step exactly as full-board occupancy does in real lookups.
fn occupancy_stream() -> Vec<(Square, Bitboard)> {
    let mut rng = SplitMix64(0x5EED_A77A_C4B3_9C41);
    (0..STREAM_LEN)
        .map(|i| {
            let sq = Square::new((rng.next_u64() % 64) as u8);
            let occ = match i % 3 {
                0 => rng.next_u64(),
                1 => rng.next_u64() & rng.next_u64() & rng.next_u64(),
                _ => rng.next_u64() | rng.next_u64(),
            };
            (sq, Bitboard::from_u64(occ))
        })
        .collect()
}

/// One benchmark: fold `lookup` over the whole stream, OR-accumulating so
/// no lookup is dead code. With `Throughput::Elements(STREAM_LEN)` criterion
/// reports elements/sec; ns per lookup = iteration time / `STREAM_LEN`.
fn bench_lookup(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    stream: &[(Square, Bitboard)],
    lookup: impl Fn(Square, Bitboard) -> Bitboard,
) {
    g.bench_function(name, |b| {
        b.iter(|| {
            let mut acc = Bitboard::EMPTY;
            for &(sq, occ) in black_box(stream) {
                acc |= lookup(sq, occ);
            }
            acc
        })
    });
}

fn bench_attacks(c: &mut Criterion) {
    let stream = occupancy_stream();
    let tables: &'static AttackTables = attacks::tables();

    let mut g = c.benchmark_group("attacks");
    g.throughput(Throughput::Elements(STREAM_LEN as u64));

    // Hoisted handle: the doc 03 §3.7 hot-loop pattern.
    bench_lookup(&mut g, "rook_hoisted", &stream, |sq, occ| {
        tables.rook_attacks(sq, occ)
    });
    bench_lookup(&mut g, "bishop_hoisted", &stream, |sq, occ| {
        tables.bishop_attacks(sq, occ)
    });
    bench_lookup(&mut g, "queen_hoisted", &stream, |sq, occ| {
        tables.queen_attacks(sq, occ)
    });

    // Free functions: one LazyLock deref per call.
    bench_lookup(&mut g, "rook_free_fn", &stream, attacks::rook_attacks);
    bench_lookup(&mut g, "bishop_free_fn", &stream, attacks::bishop_attacks);
    bench_lookup(&mut g, "queen_free_fn", &stream, attacks::queen_attacks);

    g.finish();
}

criterion_group!(benches, bench_attacks);

// Hand-rolled `criterion_main!` so the one-shot init measurement runs before
// anything else can force the LazyLock.
fn main() {
    let start = Instant::now();
    attacks::tables();
    println!(
        "attacks::tables() init (full magic search, one-shot): {:.1?}  \
         [doc 03 §6 target: <= 25 ms release]",
        start.elapsed()
    );

    benches();
    Criterion::default().configure_from_args().final_summary();
}
