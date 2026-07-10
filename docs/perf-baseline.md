# Criterion performance baselines

Recorded numbers from `cargo bench` (design/14 §3.3): micro-bench baselines
are written down here whenever a component stabilizes, one section per
machine. They are **never** asserted in CI or used as a merge gate — shared
and virtualized runners make timing assertions flaky (design/14 §5); nps
gates are human gates on a known machine.

Protocol: the documented fast configuration
(`RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench`, CLAUDE.md), quiet
machine, ≥ 2 consecutive runs agreeing within a few percent, criterion
point estimates recorded. When a component changes materially, append a
new dated subsection — never overwrite history.

## Machine: AMD Ryzen 7 7700X (WSL2)

WSL2 (Linux 5.15 kernel), rustc 1.92.0, criterion 0.8.2,
`RUSTFLAGS="-C target-cpu=x86-64-v3"`, bench profile = release
(fat LTO, codegen-units = 1). WSL2 caveat (design/14 §6): elapsed-time
medians are trustworthy, hardware perf counters are not; expect a few
percent run-to-run noise and some criterion outliers.

### attacks — 2026-07-10, P1 (`benches/attacks.rs`, contract: design/03 §6)

1024 (square, occupancy) pairs per iteration, mixed-density random stream
(uniform/sparse/dense in rotation), hot cache, throughput mode.
ns/lookup = criterion point estimate ÷ 1024. Two consecutive runs agreed
within ±3.5%; both point estimates shown, ns/lookup from the better run.

| Benchmark              | run 1     | run 2     | ns/lookup | target (doc 03 §6) |
|------------------------|-----------|-----------|-----------|--------------------|
| attacks/rook_hoisted   | 1.102 µs  | 1.100 µs  | 1.07      | ≤ 5 ns             |
| attacks/bishop_hoisted | 923 ns    | 903 ns    | 0.88      | ≤ 5 ns             |
| attacks/queen_hoisted  | 1.794 µs  | 1.717 µs  | 1.68      | ≤ 10 ns            |
| attacks/rook_free_fn   | 1.162 µs  | 1.187 µs  | 1.13      | —                  |
| attacks/bishop_free_fn | 966 ns    | 939 ns    | 0.92      | —                  |
| attacks/queen_free_fn  | 1.899 µs  | 1.859 µs  | 1.82      | —                  |

- **All doc 03 §6 targets met with ~4x headroom** (hoisted hot lookups:
  ~1.1 ns rook, ~0.9 ns bishop, ~1.7 ns queen).
- **LazyLock deref price** (free_fn vs hoisted rows): ≈ +0.04–0.10
  ns/lookup, i.e. ~4–8% on this throughput measurement — real but small.
  The doc 03 §3.7 hoisting guidance stands; Worker-cached
  `&'static AttackTables` handles (D23) will remove it from the search
  path entirely.
- **Magic-search init** (one-shot, printed by the bench before criterion
  runs, since a `LazyLock` initializes once per process): **20.9–21.1 ms**
  — matches the ~21 ms measured once at P1 step 4 (doc 03 §6 target
  ≤ 25 ms).
