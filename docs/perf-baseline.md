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

### movegen + perft — 2026-07-10, P3/M2 (`benches/movegen.rs`, `benches/perft.rs`; contracts: design/05 §6, design/14 §4 steps 3–4)

Same machine/config as above (rustc 1.92.0, criterion 0.8.2,
`RUSTFLAGS="-C target-cpu=x86-64-v3"`, bench profile = release).
Throughput mode: the perft rows use `Throughput::Elements(<oracle node
count>)` so thrpt reads directly in nodes/s; the movegen rows use the
total moves emitted per pass over the six standard perft positions
(ns/position = time ÷ 6).

| Benchmark               | run 1 (time / thrpt)   | run 2 (time / thrpt)   | derived            | target                        |
|-------------------------|------------------------|------------------------|--------------------|-------------------------------|
| perft/startpos_d5       | 17.34 ms / 280.5 Mnps  | 17.39 ms / 279.8 Mnps  | 280 Mnps bulk      | ≥ 100 Mnps floor / 250 target |
| perft/kiwipete_d4       | 13.76 ms / 297.0 Mnps  | 13.78 ms / 296.6 Mnps  | 297 Mnps bulk      | ≥ 100 Mnps floor / 250 target |
| movegen/non_evasions_x6 | 171.6 ns / 1.236 G/s   | 172.6 ns / 1.228 G/s   | ≈ 28.7 ns/position | —                             |
| movegen/captures_x6     | 60.4 ns / 364 Mmoves/s | 61.1 ns / 360 Mmoves/s | ≈ 10.1 ns/position | —                             |
| movegen/legal_x6        | 361.4 ns / 492.6 M/s   | 361.1 ns / 493.0 M/s   | ≈ 60 ns/position   | —                             |

- Three perft runs were taken; the middle one was a +10 % WSL2 excursion
  on `startpos_d5` (kiwipete_d4 unaffected), so the recorded pair is runs
  1 and 3, which agree within 0.3 %. The movegen rows are two consecutive
  runs, all within ±3 %.
- **Both perft rows clear the D24 floor (100 Mnps) and the 250 Mnps
  stretch** (design/06 §6, design/05 §8.8).
- **Kiwipete d5 bulk perft, D16 record (doc 06 §8.4)**: the release binary
  (`chessengine perft 5 <kiwipete>`, same RUSTFLAGS) finishes 193,690,690
  nodes in 0.61 / 0.62 / 0.67 s wall = 289–318 Mnps *including* process
  startup and the ~21 ms magic init; the in-process M2 gate measurement
  (commit c8e30dd) was 303–359 Mnps, median 339.
- `generate_legal` costs ≈ 2.1× raw `NON_EVASIONS` generation on this mix
  (the lazy legality filter + suspect testing, doc 05 §3.9) while still
  emitting ~0.5 G legal moves/s.

### fixed-node SPRT proxy — 2026-07-11, P9 Wave A (design/14 §3.8 / §9 Q4)

Calibration of the per-move node budget `N` for the fixed-node game proxy
that gates each pruning heuristic (`go nodes N`, driven by fastchess
`tc=inf nodes=N` via `NODES=N tools/sprt.sh`). Same machine/config
(x86-64-v3, release, 8 physical / 16 logical cores). `bench 16 1 6 default
depth` measured **5.82 Mnps** on this tree.

**Determinism (the whole point):** `go nodes N` reads no wall clock on any
node-count path (design/14 §3.2), so it is bit-reproducible and
hardware-independent. Verified — identical bestmove *and* final node count
on repeat (`go nodes 250000`): startpos → `g1f3`/60297, kiwipete →
`e2a6`/125258, R+P endgame → `b4f4`/133554, twice each. Stop overshoot is
bounded by `min(N/1024, 512)` nodes (`check_time_reset`).

**Real 10+0.1 anchor** (this build, self-play, 380 plies driving both
clocks): mean ≈ 1.0–1.4 M, median ≈ 0.45 M nodes/move (mean skewed by
max-time tactical moves; the info-line node figure understates the true
per-move total ≈ think-time × nps). Matching that absolute budget would cost
~15–20 s/game — too slow for a cheap proxy.

**Throughput vs budget** (48-game self-vs-self batches, 16-way, fixed nodes
tolerate SMT):

| N (nodes/move) | games/s | amortized s/game | ~equiv think-time | ~depth |
|----------------|---------|------------------|-------------------|--------|
| 100 000        | 2.27    | 0.44             | 18 ms             | 6–7    |
| **200 000**    | **1.45**| **0.69**         | **36 ms**         | **7–8**|
| 250 000        | 1.06    | 0.94             | 45 ms             | 7–9    |
| 300 000        | 0.93    | 1.08             | 55 ms             | 8–9    |

**Chosen: `N = 200_000`** — ~1/7 of the real 10+0.1 per-move budget (the
standard fast-STC fraction), still deep enough (~7–8) that RFP/NMP/LMR fire
in an STC-representative regime, and cheap enough that a 400-game batch is
~4–5 min and a large-effect pruning SPRT terminates in ~3–5 min. `sprt.sh`
self-vs-self smoke at N=200 000 returned Elo −0.00 / LLR −0.00 over 20 games
(sanity: a proxy of a binary against itself is ~0 Elo). `NODES=100000`
is the faster/rougher setting; `NODES=400000` a higher-fidelity confirm. The
proxy gates acceptance within a wave; the milestone gate still uses real
10+0.1 (design/14 §3.8).
