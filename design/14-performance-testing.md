# 14 — Performance engineering and strength testing

Status: Reviewed v1

This document defines how we *know* the engine is fast and strong: the
measurement hierarchy (SPRT games > `bench` > tactical suite > criterion
micro-benches), the measurement policy that makes `bench` the node-count
fingerprint and nps oracle for every functional change (D24) — the
subcommand itself (CLI grammar, 49-position suite, output format) is
specified in the perft/validation doc §3.6 — realistic performance targets
per development phase with their assumptions stated, the profiling toolkit
and a triage workflow for nps regressions, the release compile configuration
(D16), the SPRT/gauntlet methodology for strength claims (D25), the
tier-1.5 tactical sanity harness, the per-commit regression checklist, and
the Rust-specific performance pitfalls (bounds checks, implicit clones,
`LazyLock` overhead) that can silently cost us the per-node cost parity we
need. Nothing in this doc adds a runtime dependency (D15); everything
heavier than criterion is external tooling.

---

## 1. Purpose and scope

**This doc owns:**

- The `bench` *measurement policy*: fingerprint discipline, determinism
  requirements, and the `Bench:` commit-trailer protocol (D24). The
  subcommand itself — CLI grammar, the compiled-in 49-position suite,
  `src/bench.rs` types, and output format — is owned by the perft/validation
  doc §3.6; this doc states what must hold for its numbers to be evidence.
- The tier-1.5 tactical sanity harness (`tools/tactics.sh`, §3.1/§3.9).
- Criterion micro-benchmark suite layout and what each bench measures (D15).
- Performance targets: perft nps, search nps per eval phase, TT probe
  latency, with explicit hardware assumptions.
- Profiling workflow: `perf stat` counter set, flamegraphs, cachegrind,
  bounds-check auditing, and the ordered triage list for regressions.
- Compile configuration: release profile flags, `target-cpu` policy, and the
  optional PGO pipeline (D16).
- Strength testing methodology: fastchess/cutechess SPRT parameters, opening
  books, gauntlet ladder, Elo milestones (D25).
- The regression protocol checklist every change must pass.

**This doc does NOT own:**

- The `bench` subcommand's implementation: argument grammar, the position
  suite content (the 49 FENs, printed verbatim there), the `src/bench.rs`
  API (`BenchParams`, `BenchLimit`, `FenSource`, `BenchResult`, `run`), and
  the output format — all owned by the perft/validation doc §3.6.
- The implementation of search, movegen, perft, TT, eval, or threads
  themselves — only how they are *measured*. Internals live in their own
  docs: the perft doc (perft algorithm and known-count tables), the search
  doc (node counting semantics), the TT doc (huge pages, prefetch,
  `hashfull`), the threads doc (per-worker node counters and aggregation),
  and the UCI doc (how `bench` is dispatched from `main`). See §7 for the
  dependency list.
- NNUE SIMD implementation details (eval phase C doc); we only set its nps
  targets here.
- Texel tuning / net training pipelines (eval docs).

---

## 2. References

- `stockfish_reference/stockfish_search_notes.md` — §11.5 (TT memory, huge
  pages, parallel clear), §13.2 (thread voting disabled under `go depth` for
  bench reproducibility), §13.3 (`nodestime` as a reproducible-time device),
  §14 (dependency-ordered roadmap this doc's milestones attach to).
- `stockfish_reference/00-design-implications.md` — performance philosophy
  (centralized mutation, branchless idioms, prefetch, benign races).
- SF source (read-only, ideas only, D27):
  - `Stockfish/src/benchmark.cpp` — `bench` defaults: TT 16 MB, 1 thread,
    depth 13, a fixed `Defaults` list of 51 FENs — 49 standard chess plus 2
    Chess960 (openings, middlegames, endgames, 5/6/7-man TB positions,
    mates/stalemates); we adopt the 49 standard entries (960 deferred per
    D10), list printed in the perft/validation doc §3.6; argument order
    `ttSize threads limit fenFile limitType`.
  - `Stockfish/src/tt.cpp`, `misc.cpp`, `memory.cpp` — hashfull sampling,
    large-page allocation (what our dTLB measurements will validate).
  - `Stockfish/src/Makefile` — the `profile-build` (PGO) flow we mirror with
    `cargo-pgo`.
- External methodology (tooling, not dependencies, D15): fishtest's SPRT
  practice (normalized-Elo bounds, pentanomial statistics, UHO books),
  fastchess and cutechess-cli documentation.
- Our code: `src/board.rs` (`Board`, `put_piece`/`remove_piece`),
  `src/moves.rs` (`Move`, `MoveList`, `MAX_MOVES`), `src/fen.rs`
  (`Board::from_fen`, `START_FEN`), `src/bitboard.rs`.

---

## 3. Design

### 3.1 The measurement hierarchy

Three tiers, each answering a different question. Using the wrong tier is
the most common testing mistake, so the rule is explicit:

| Tier | Tool | Question it answers | Never use it for |
|---|---|---|---|
| 1 | SPRT games (fastchess) | "Is the engine *stronger*?" | speed-only claims |
| 1.5 | tactical EPD suite (`tools/tactics.sh`) | "Does the engine still *find* tactics and mates?" | merge gating or Elo claims |
| 2 | `bench` subcommand | "Did search behavior change? Did nps change?" | strength claims |
| 3 | criterion micro-benches | "Did this one function get faster?" | whole-engine claims |

Corollaries: a search heuristic that loses nps can still gain Elo (tier 1
decides); a 5% movegen speedup (tier 3) must show up in tier 2 nps before it
is claimed; tier 2 node-count changes classify a patch as *functional*,
which mandates tier 1 for anything touching search/eval.

**Tier 1.5 — the tactical sanity harness.** The bespoke mate-in-N and
zugzwang sets in the search and pruning docs are tiny and targeted; nothing
else in the hierarchy runs a *published* tactical corpus, yet an
over-aggressive pruning guard that stops finding mates only in rare
tactical positions is exactly the regression SPRT is too slow to attribute
and bench node counts can't see. `tools/tactics.sh` feeds an EPD suite —
WAC (Win At Chess, 300 positions) plus a mate-in-2..5 set, downloaded by
the script, never vendored (D15) — through `position fen` + `go movetime`
(or `go depth`), scores `bm`/`am` hits, and appends the score to
`docs/strength-log.md`. It is **not a merge gate** (single-position
tactics are noisy and some WAC solutions are contested), but it is
**required before milestone declarations (M2 and later, §3.8)**: the
baseline score is tracked over time and an unexplained drop blocks the
milestone until attributed (§3.9).

A speed rule of thumb for prioritization (fact from engine-testing
folklore, treat as approximate): a 2x search speedup is worth roughly
**+50 to +70 Elo** at short TC, less at long TC. So a measured +10% nps with
identical bench nodes ≈ +7–10 Elo — usually worth an SPRT-free merge if the
node count is provably unchanged, per the protocol in §3.9.

### 3.2 The `bench` measurement policy (the oracle, D24)

`bench` is a CLI subcommand of the engine binary (dispatched in `main.rs`
before entering the UCI loop, and also accepted as a command inside the UCI
loop). Its **specification lives in the perft/validation doc §3.6**, which
owns the SF-shaped argument grammar
(`bench [ttMB=16] [threads=1] [limit=13] [fens=default] [type=depth]`),
the `src/bench.rs` types (`BenchParams`, `BenchLimit`, `FenSource`,
`BenchResult { nodes, elapsed_ms }`, `fn run(&BenchParams) -> BenchResult`),
the compiled-in position suite (printed verbatim there), and the output
format (per-position lines plus the fixed footer, to stderr so stdout stays
a clean UCI stream). This doc deliberately does not duplicate that spec;
it owns the *policy* that makes bench's numbers evidence. The footer, for
reference:

```
===========================
Total time (ms) : 4213
Nodes searched  : 3187513
Nodes/second    : 756590
```

`Nodes searched` is the **fingerprint**: the exact sum of nodes over the
suite at the default settings. Two builds of the same source must produce
the same number; a patch that changes it is *functional* by definition.

**Position suite.** Stockfish's standard-chess default bench list — **49
FENs**, i.e. SF's 51-entry `Defaults` list minus its two Chess960 entries
(960 deferred per D10); FEN strings are chess positions, i.e. facts, not
copyrightable code (D27). The list itself is `BENCH_POSITIONS` in the
perft/validation doc §3.6. Its coverage is deliberately broad: quiet
openings, tactical middlegames, fortress-ish endgames, 5/6/7-man positions,
mates and stalemates (the last exercise mate-score plumbing and `MoveList`
edge cases). The suite is **frozen**: changing it invalidates all
historical fingerprints, so edits require a design-doc update and a clearly
labeled commit.

**Determinism requirements** (each is a hard invariant, tested in §5):

1. Bench always runs the search with a deterministic limit (`depth`,
   `nodes`, or `perft` — `movetime` is allowed for nps smoke tests but never
   for fingerprints).
2. Single main search thread by default; when `threads > 1`, bench is only
   an nps/scaling probe, never a fingerprint (lazy SMP is
   nondeterministic by design, D23). Thread voting is disabled under a
   depth limit (mirrors SF §13.2).
3. `ucinewgame` semantics at the start of the run (TT allocated at `ttMB`
   and cleared, histories reset); the TT deliberately **persists across the
   suite's positions** within one run — deterministic single-threaded, and
   it makes the fingerprint sensitive to TT replacement changes, which we
   want.
4. No wall-clock reads on any code path that affects node counts. The draw
   dither (`nodes & 0x2`) and equal-value dither key off the node counter —
   deterministic. No `HashMap`/`HashSet` (randomized hashing) anywhere in
   search-affecting code; fixed arrays only.
5. Node counting semantics: the counter increments exactly once per
   `make_move` in search/qsearch (defined precisely in the search doc);
   bench just sums per-position counters.

(No `src/bench.rs` sketch here on purpose — the types and `run` signature
are normative in the perft/validation doc §3.6; a second copy would drift.)

**Commit trailer protocol.** Every commit that can affect search behavior
carries a trailer line, exactly:

```
Bench: 3187513
```

- Functional patches (node count changes): trailer states the *new* count.
- Intended-non-functional patches (refactors, speedups): trailer states the
  unchanged count; a reviewer (or `tools/check.sh`, §3.9) reruns bench and
  rejects the commit if the number moved.
- Commits that cannot affect search (docs, UCI I/O, build scripts) may omit
  the trailer.

`bench` also doubles as the **perft comparison harness**:
`chessengine bench 16 1 6 default perft` runs perft 6 on the whole suite,
which is directly comparable to `./stockfish bench 16 1 6 default perft` on
the same machine (§3.5).

### 3.3 Criterion micro-benchmarks (D15)

Criterion is our only dev-dependency; benches live in `benches/` with
`harness = false`. Micro-benches are for *directed* optimization work and
for catching per-component regressions early — they never gate commits
(too noisy on shared machines) but their numbers get recorded in
`docs/perf-baseline.md` whenever a component stabilizes.

```toml
# Cargo.toml additions
[dev-dependencies]
criterion = "0.8.2"   # pinned current at implementation time (P1)

[[bench]]
name = "attacks"
harness = false

[[bench]]
name = "perft"
harness = false

[[bench]]
name = "movegen"
harness = false

[[bench]]
name = "eval"
harness = false

[[bench]]
name = "tt"
harness = false
```

The benches and their exact measurements (`benches/attacks.rs` is owned by
the attack-tables doc 03 §6 — hot rook/bishop/queen magic lookups on a
mixed-density occupancy stream, the free-function-vs-hoisted-`tables()`
LazyLock delta, and the one-shot magic-search init time; it landed first,
in P1, and brought the criterion scaffolding with it):

- **`benches/perft.rs`** — `perft(startpos, 5)` (4,865,609 nodes) and
  `perft(kiwipete, 4)` (4,085,603 nodes; Kiwipete =
  `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -`).
  Reported as throughput (nodes/sec) via `Throughput::Elements`. This is the
  make/unmake + movegen parity indicator.
- **`benches/movegen.rs`** — generation only: fill a `MoveList` for each of
  ~10 fixed FENs (open middlegame, closed middlegame, check evasions,
  promotion-heavy, endgame), separately for the captures generator and the
  full generator. Metric: ns/position and moves/sec.
- **`benches/eval.rs`** — `evaluate(&board)` over the 49 bench positions.
  Metric: ns/call. For NNUE later: one *refresh* (full accumulator rebuild)
  vs one *incremental* update+eval — the ratio is the health check for
  dirty-piece tracking.
- **`benches/tt.rs`** — TT probe/store loops over a 16 MB and a 1 GB table
  with a xorshift key stream: (a) hot loop (working set ≪ LLC), (b) cold
  loop (uniform random over the whole table), (c) cold with `prefetch`
  issued N iterations ahead. Metric: ns/probe. This bench is what validates
  the huge-pages work in the TT doc.

Skeleton showing the conventions (black_box on all inputs, throughput
units):

```rust
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use chessengine::{Board, perft::perft};

const KIWIPETE: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

fn bench_perft(c: &mut Criterion) {
    let mut g = c.benchmark_group("perft");
    g.throughput(Throughput::Elements(4_085_603));
    g.bench_function("kiwipete_d4", |b| {
        let mut board = Board::from_fen(KIWIPETE).unwrap();
        b.iter(|| perft(black_box(&mut board), 4))
    });
    g.finish();
}

criterion_group!(benches, bench_perft);
criterion_main!(benches);
```

Note: Cargo ignores the `panic` setting for `test`/`bench` profiles, so
`panic = "abort"` in `[profile.release]` (§3.7) does not break `cargo test`
or criterion.

### 3.4 Target ladder

**Stated assumptions** (all targets below assume these; re-derive on other
hardware): single thread; x86-64-v3 (AVX2/BMI2) build; modern desktop core
(Zen 4/5 or Raptor-Lake class, ~4.5–5.5 GHz boost); DDR4/DDR5; TT = 16 MB
for bench; transparent huge pages available but not assumed; quiet machine,
performance governor. WSL2 (our current dev box) typically measures within
a few percent of native for CPU-bound work but has unreliable access to
hardware perf counters — see §3.6.

| Metric | Floor (investigate below) | Target | Parity / stretch |
|---|---|---|---|
| perft, bulk-counted (leaf level counts moves without making them), startpos d6 / Kiwipete d5 | 100 Mnps | 250 Mnps | 400+ Mnps |
| perft, full make/unmake at every node incl. leaves | 20 Mnps | 50 Mnps | 80 Mnps |
| Search nps, eval phase A (material + tapered PSQT, incremental) | 1.5 Mnps | 2–4 Mnps | — |
| Search nps, eval phase B (HCE) | 1 Mnps | 1.5–2.5 Mnps | — |
| Search nps, eval phase C (NNUE, AVX2, incremental accumulators) | 0.5 Mnps | 1–2 Mnps | ≥ 70% of local SF bench nps |
| NNUE scalar fallback | — | ≥ 25–35% of the SIMD path | — |
| TT probe, hot in cache | — | < 10 ns | — |
| TT probe, cold (DRAM) | — | 70–100 ns raw; mostly hidden by prefetch issued in `make_move` | — |
| PSQT incremental eval | — | 5–20 ns/call | — |

Interpretation notes:

- **Perft is the make/unmake parity indicator.** Bulk counting measures
  movegen almost purely; the full-make variant adds make/unmake and state
  maintenance (all five Zobrist keys, D9, plus checkers/pins recompute).
  A large gap between our full-make perft and SF's on the same machine means
  our `make_move` is doing something expensive per node — find it before
  building search on top.
- **Search nps falls as the engine matures.** More eval work per node and
  more TT/history traffic per node both cost nps while gaining Elo. That is
  why nps targets are *per eval phase* and why tier 1 owns strength claims.
- Whole-run counter expectations on `bench` (for §3.6 triage): IPC ≥ 1.3
  (typical 1.8–2.5); branch-miss rate on branches < 6% (search is
  inherently branchy); LLC misses concentrated in TT probes and (later)
  NNUE weights; dTLB-load-miss rate collapses (> 10x) once the TT sits on
  2 MB huge pages — even a 16 MB TT (4096 × 4 KB pages) exceeds typical
  dTLB coverage, so expect +2–5% nps from huge pages at 16 MB and more at
  1 GB+ tables.

### 3.5 Comparing against a local Stockfish fairly

Never compare against published SF numbers from other machines. Protocol:

1. Build SF from `stockfish_reference/Stockfish/src` on this machine:
   `make -j profile-build ARCH=x86-64-v3` (PGO, same ISA baseline we
   document for ourselves; use `ARCH=native` only if we also build ours with
   `target-cpu=native`).
2. Same environment for both: performance governor, idle machine, pinned to
   the same physical core (`taskset -c 2 ...`), 5 runs, take the median.
3. Like for like:
   - Movegen/make parity: `./stockfish bench 16 1 6 default perft` vs
     `chessengine bench 16 1 6 default perft`. This is the *only* directly
     comparable raw-speed number early on.
   - Search nps: `./stockfish bench` (depth 13) vs `chessengine bench`.
     Until phase C, our nps being *higher* than SF's is expected and means
     nothing (PSQT is ~100x cheaper than dual-net NNUE); the comparison
     becomes meaningful in phase C, where the target is ≥ 70% of SF's bench
     nps on the same machine.
   - Never compare node *counts* across engines — different searches,
     meaningless.
4. For search changes that alter tree shape, use **time-to-depth** on the
   bench suite (elapsed to finish depth 13) rather than nps: nps can rise
   while time-to-depth worsens.

### 3.6 Profiling toolkit and triage workflow

**Counter set** (the ones that have historically mattered for engines):

```bash
perf stat -e task-clock,cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses \
  taskset -c 2 ./target/release/chessengine bench
```

Read in this order: **IPC** (instructions/cycles) tells you whether the
machine stalls; **instructions** tells you whether the *work* grew (a
regression with equal instructions but lower IPC is a memory/branch
problem; equal IPC but more instructions is codegen/algorithmic);
**branch-miss rate** for search/movepick changes; **LLC-load-misses** for
TT/NNUE-sized data; **dTLB-load-misses** before/after huge pages (the TT
doc's acceptance test).

**Flamegraphs**: build with symbols but full optimization via a dedicated
profile, then:

```bash
cargo flamegraph --profile profiling --bin chessengine -- bench
# or manually:
perf record -g --call-graph dwarf ./target/profiling/chessengine bench
perf report --hierarchy
```

Sanity anchor for a healthy search build: expect eval + movegen + make/
unmake + TT probe to dominate; any single "utility" function (a `Display`,
an allocator frame, `LazyLock::deref`) visible at ≥ 1% is a finding.

**Cachegrind** (cache-line-level attribution; ~50x slowdown so shrink the
run, and it works on WSL2 where `perf` counters may not):

```bash
valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
  ./target/profiling/chessengine bench 16 1 8
cg_annotate cachegrind.out.*
```

Use it to answer layout questions: does `TTEntry` traffic stay within one
64-byte line per probe (it must — 3×10-byte cluster + padding = 32 bytes,
D19); does `ExtMove` scoring walk linearly (D7).

**Bounds-check audit** (see also §3.10): whole-binary metric plus targeted
disassembly:

```bash
# ambient metric — track over time, investigate jumps:
objdump -d target/release/chessengine | grep -c panic_bounds_check
# targeted (cargo-show-asm):
cargo asm --release --lib "chessengine::movegen::generate" | grep -A2 panic
```

**Triage order when bench nps regresses** (stop at the first hit):

1. Reproduce: 3 bench runs on a quiet, pinned core. < 1% deltas are noise;
   ignore.
2. Did the **node count** change? If yes the patch is functional — nps alone
   is the wrong lens; measure time-to-depth and take it to SPRT.
3. `perf stat` diff vs the previous commit: instructions↑ → inspect codegen
   (new bounds checks, missing inlining, an accidental `clone`); IPC↓ with
   instructions flat → memory or branches; check LLC and dTLB counters.
4. Flamegraph diff: any new frame ≥ 1% is the suspect.
5. Suspect memory layout → cachegrind on the hot structure.
6. Suspect codegen → `cargo asm` the hot function, compare against the
   previous commit's output.

### 3.7 Compile configuration (D16) and PGO

```toml
# Cargo.toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"          # ignored by test/bench profiles, so cargo test still works

[profile.profiling]
inherits = "release"
debug = "line-tables-only"   # symbols for perf/flamegraph; no codegen change
```

- The supported fast configuration is documented (not hardcoded) as
  `RUSTFLAGS="-C target-cpu=x86-64-v3"`; local strength testing may use
  `target-cpu=native`. Binaries used in any SPRT pair must be built with
  identical flags.
- Startup builds the magic tables at runtime behind `LazyLock` (D12), so
  compile-time cost stays sane; no build-script codegen needed for perf.
- **PGO** (optional, later, after NNUE lands — it moves the most when
  branchy cold/hot splits exist): via `cargo-pgo`:

```bash
cargo pgo build                       # instrumented build
./target/x86_64-*/release/chessengine bench 64 1 15   # training workload
cargo pgo optimize build              # final optimized build
```

Expected gain: **low single-digit percent nps** (SF's own profile-build is
worth ~5–10% for C++; Rust with fat LTO already captures much of that).
Rule: PGO'd binaries are fine for gauntlets but the *baseline and candidate
must both be PGO'd or both not* — never mix.

### 3.8 Strength testing (D25)

**Harness**: fastchess (primary; maintained, pentanomial-aware SPRT),
cutechess-cli (fallback; logistic-Elo SPRT only). Both are external tools
(D15).

**SPRT parameters** — the standard gain test for any functional
search/eval change:

- Hypotheses: `elo0 = 0`, `elo1 = 5` in **normalized Elo** (nElo),
  `alpha = 0.05`, `beta = 0.05`, `model = normalized`.
- LLR bounds: accept H1 when `LLR ≥ ln((1-β)/α) = ln(19) ≈ 2.94`; accept H0
  when `LLR ≤ ln(β/(1-α)) = -ln(19) ≈ -2.94`. fastchess computes the GSPRT
  LLR from pentanomial game-pair counts; we just read the verdict.
- **Simplifications / non-regressions**: bounds `[-5, 0]` (accept if we are
  not more than ~5 nElo weaker).
- TC: **10+0.1** (short) for iteration; confirm larger architectural
  changes at **60+0.6** before declaring a milestone. Concession to local
  hardware: our 10+0.1 plays the role fishtest's STC plays for SF.
- Openings: **UHO book** (e.g. `UHO_Lichess_4852_v1.epd` — the
  fishtest-style unbalanced human openings book), `order=random`,
  `-repeat` so each opening is played with colors reversed (pentanomial
  pairs).
- Both engines: `Threads=1`, `Hash=16` (STC) / `Hash=64` (LTC), identical
  build flags.

```bash
fastchess \
  -engine cmd=./candidate name=candidate \
  -engine cmd=./baseline  name=baseline \
  -each proto=uci tc=10+0.1 option.Hash=16 option.Threads=1 \
  -openings file=books/UHO_Lichess_4852_v1.epd format=epd order=random \
  -repeat -games 2 -rounds 25000 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 model=normalized \
  -draw movenumber=34 movecount=8 score=20 \
  -resign movecount=3 score=600 \
  -concurrency 6 -recover -ratinginterval 10
```

**Honest testing on limited local cores:**

- `concurrency = physical cores - 1` for *timed* games; never count SMT
  siblings — hyperthread contention corrupts clocks and biases results.
  Leave one core for the OS.
- **Fixed-node games as the cheap proxy**: `tc=inf nodes=50000` (cutechess
  syntax; fastchess `-each nodes=50000`) removes clock jitter entirely, so
  SMT and background load stop mattering and all logical cores can play.
  Use for eval and search-shape changes; *never* for time-management
  changes (TM is unexercised) and confirm anything surprising with a timed
  run before merging.
- Error-bar honesty (approximate, logistic, draw-heavy TC): ~±44 Elo at 100
  games, ~±14 at 1,000, ~±4.4 at 10,000 (95% CI). Elo from score:
  `elo = -400·log10(1/s - 1)` where `s` is the match score. Corollary: a
  200-game eyeball run proves nothing inside ±30 Elo — SPRT or it didn't
  happen (D25).

**Gauntlet ladder** (calibration only — SPRT vs. our own previous version
is the merge gate; gauntlets tell us *where we are*). Reference opponents
with stable public versions, approximate CCRL-blitz anchors (all "~"):

| Plan phase (roadmap §14 of search notes) | Feature state | Suggested opponents | Milestone (CCRL-blitz-equivalent) |
|---|---|---|---|
| M1: roadmap steps 1–2 (negamax + qsearch + TT), eval phase A (PSQT) | ID, MVV ordering, mate-distance | TSCP (~1700), a low Stash tag | ≥ 1900 |
| M2: steps 3–5 (PVS, aspiration, staged MovePicker, SEE, NMP/RFP/LMR/LMP/futility/razoring/IIR) | still eval A | Vice (~2000), mid Stash tags | ≥ 2450 |
| M3: steps 6–8 (conthist, singular/multicut, time management) | still eval A | Fruit 2.1 (~2700), higher Stash | ≥ 2750 |
| M4: eval phase B (HCE + texel tuning) | full search | ~2900-class engines | ≥ 2950 |
| M5: eval phase C (NNUE) + step 9 (lazy SMP) + step 10 refinements | complete | recent Stash tags, older SF releases | ≥ 3200, then chase parity |

Stash is the recommended ladder backbone: ~30 tagged releases spanning
roughly 1300–3400 CCRL, so there is always an opponent within ±200 Elo of
us (matches in the 45–65% score band have the smallest Elo error). Run each
rung as a gauntlet (us vs 3–4 opponents, ≥ 1000 games total, UHO book) and
compute ratings with `ordo`.

### 3.9 Regression protocol — the per-change checklist

Codified in `tools/check.sh`; a change is mergeable when:

1. `cargo fmt --check` — clean.
2. `cargo clippy --all-targets -- -D warnings` — clean (CLAUDE.md).
3. `cargo test` — green (includes board-consistency and all unit tests).
4. **Perft suite** green (release-mode test over the standard positions;
   mandatory for any movegen/make-unmake/attack-table change, D24).
5. `chessengine bench` run; node count recorded in the commit message as a
   `Bench: <nodes>` trailer (§3.2). Non-functional claims are verified by
   rerunning bench on the reviewed tree.
6. Search/eval functional changes: SPRT result (`[0, 5]` gain or `[-5, 0]`
   simplification) linked or summarized in the commit body (D25).
7. Speed claims (> 1% nps): 5-run median bench nps before/after on a pinned
   core, plus the relevant `perf stat` counters, in the commit body.
8. **Milestone declarations only** (M2 and later, §3.8 ladder — not a
   per-commit gate): `tools/tactics.sh` has been run (§3.1 tier 1.5) and
   the tactical-suite score recorded in `docs/strength-log.md`; an
   unexplained drop against the tracked baseline blocks the milestone
   until attributed.

```bash
#!/usr/bin/env bash
# tools/check.sh — the gate. Fails loudly, runs everything.
set -euo pipefail
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --release perft          # full perft suite in release
cargo build --release
./target/release/chessengine bench | tail -n 4   # paste 'Nodes searched' into the trailer
```

### 3.10 Rust-specific performance pitfalls

Ranked by how likely they are to silently cost us node-cost parity:

1. **Bounds checks in hot loops.** Indexing `[T; N]` with a runtime `usize`
   emits a check unless the compiler proves the range. Mitigations in
   preference order: (a) iterate (`for sq in bb` via `BitboardIter` — no
   indexing at all); (b) index with values whose type proves the range —
   `Square::index()` returns `u8 < 64` by constructor invariant, so
   `table[sq.index() as usize]` on a `[T; 64]` is *usually* elided, but
   **verify, don't assume** (`cargo asm`/`objdump` metric, §3.6); (c) for
   2-D tables prefer `[[T; 64]; 12]` indexed by `Piece::index()` (D4) over
   flattened manual arithmetic the compiler can't see through; (d) only
   after a profile shows a check on a hot path: `get_unchecked` with a
   `// SAFETY:` comment citing the measurement (D14).
2. **Implicit `Board` clones.** `Board` is ~250 bytes of arrays *plus*,
   once D8 lands, a `Vec<StateInfo>` history — so `Board::clone()`
   **allocates**. One accidental `.clone()` per node is catastrophic.
   Rules: search takes `&mut Board` and strictly pairs
   `make_move`/`unmake_move`; `Clone` stays (threads need per-worker copies
   at search start, D23) but a periodic `grep -n "\.clone()" src/` audit of
   hot modules is part of search-code review; helper APIs never take
   `Board` by value.
3. **`LazyLock` deref in hot code.** Every `*ATTACK_TABLES` deref is an
   atomic load + branch (D12). Hoist once per search (or per movegen call)
   into `let atk: &'static AttackTables = &ATTACK_TABLES;` and pass the
   reference down, or expose `#[inline] fn attacks() -> &'static
   AttackTables` and hold the result across loops. Force initialization at
   startup (`LazyLock::force`) so first-probe latency never lands inside a
   timed search.
4. **`inline` discipline.** `#[inline(always)]` only for tiny leaf
   accessors (bitboard ops, `Square`/`Move` getters, gravity update) —
   these are already candidates, so this mostly guards against
   `#[inline(never)]`-by-accident across crate-internal calls before LTO.
   Never `inline(always)` anything with a loop or a cold branch (icache
   pressure). Mark error/IO paths `#[cold]`/`#[inline(never)]`.
5. **Branchless patterns.** Prefer arithmetic on `bool as i32` for the
   formula-shaped code (`margin -= 2934 * improving as i32 * fut_mult /
   1024`), XOR square flips (`sq ^ 56`), `b & (b-1)` / `b & b.wrapping_neg()`
   for bit tricks, and `Value::max/min` clamps over `if` chains. Check the
   generated code before claiming a win — LLVM often does this rewrite
   itself, and "branchless" that defeats a 95%-predictable branch is a
   pessimization. `perf stat` branch-miss rate arbitrates.
6. **`Vec` growth mid-search.** The `StateInfo` history (D8) must
   `Vec::with_capacity(MAX_PLY + 64)` at `Board` construction/reset so
   `make_move` never reallocates (a realloc invalidates any cached
   pointers and costs a syscall-ish hiccup at the worst time).
7. **`Option<Piece>` mailbox in hot paths** is fine (niche-packed to 2
   bytes, branch predicts well), but don't `.unwrap()` on paths where the
   invariant is structural — use `debug_assert!` + explicit match so
   release code carries no panic branch beyond the necessary one.
8. **Iterator adapter chains in movegen/search inner loops**: acceptable
   only when they compile to the same loop (verify once with `cargo asm`);
   when in doubt in the hottest loops, write the plain loop — this is the
   one place CLAUDE.md's "prefer idiomatic" yields to measurement.

---

## 4. Implementation plan

Ordered, commit-sized; each step names its done-condition.

1. **Release + profiling profiles.** Add `[profile.release]`
   (`lto="fat"`, `codegen-units=1`, `panic="abort"`) and
   `[profile.profiling]` to `Cargo.toml`; document
   `RUSTFLAGS="-C target-cpu=x86-64-v3"` in CLAUDE.md.
   *Done:* release build succeeds; `cargo test` unaffected.
2. **`tools/check.sh`.** fmt + clippy + test (+ perft/bench lines commented
   until those exist). *Done:* script exits non-zero on any injected
   failure.
3. **Criterion scaffolding + perft bench** (lands with the perft module;
   the scaffolding itself — dev-dep, first `[[bench]]`, and
   `docs/perf-baseline.md` — landed early with `benches/attacks.rs` per the
   attack-tables doc 03 §4 step 5). Add `benches/perft.rs` with startpos-d5
   and Kiwipete-d4. *Done:* `cargo bench perft` runs; two consecutive runs
   agree within ±2%; numbers recorded in `docs/perf-baseline.md`.
4. **`benches/movegen.rs`** (lands with movegen). *Done:* runs; baseline
   recorded.
5. **`bench` fingerprint v1.** The subcommand itself (parser, 49-FEN
   suite, output format) is implemented per the perft/validation doc §3.6
   and its plan — perft mode first, depth mode with search v1 + TT; this
   step is the fingerprint policy going live. *Done:* three consecutive
   `chessengine bench` runs print identical `Nodes searched`;
   `bench 16 1 6 default perft` totals match the perft suite's known
   counts.
6. **Bench trailer protocol.** CLAUDE.md gains the `Bench:` trailer rule
   and the functional/non-functional definitions; `tools/check.sh` gains
   the bench line. *Done:* first search commit carries a verified trailer.
7. **Determinism test.** `#[test]` (release-gated or `--ignored` in debug)
   that runs the bench suite twice at depth 6 in-process and asserts equal
   node totals. *Done:* test in CI set; deliberately breaking determinism
   (e.g. seeding from time) fails it.
8. **`benches/eval.rs` and `benches/tt.rs`** (land with eval phase A and
   the TT). *Done:* run; baselines recorded, including hot/cold/prefetch
   TT probe split.
9. **`tools/profile.sh`.** The `perf stat` counter set, `perf record` +
   flamegraph recipes, cachegrind fallback, bounds-check `objdump` metric;
   documents WSL2 counter limitations. *Done:* produces a flamegraph SVG of
   `bench` on this machine.
10. **Strength harness.** `tools/sprt.sh` (the §3.8 fastchess invocation
    with baseline/candidate args), `books/` download instructions, pinned
    opponent binaries under `engines/` (gitignored). *Done:* a 200-game
    smoke SPRT between two builds of the same source terminates ~50% and
    reports LLR.
11. **`tools/gauntlet.sh` + ordo; `tools/tactics.sh`.** Gauntlet configs
    for the current milestone rung; the tier-1.5 tactical runner (§3.1) —
    EPD download (WAC + mate-in-2..5, script-fetched, not vendored, D15),
    `bm`/`am` scoring, strength-log line. *Done:* first M1 gauntlet
    produces an ordo rating with error bars, and a first tactics baseline
    score, both recorded in `docs/strength-log.md`.
12. **PGO pipeline** (deferred until phase C). `cargo-pgo` recipe in
    `tools/`. *Done:* PGO build runs bench with identical node count and
    the nps delta is recorded.

---

## 5. Testing and validation

- **Unit tests** (`src/bench.rs`, module owned by the perft/validation doc
  §3.6): argument parsing — defaults (`16/1/13/default/depth`), each
  positional override, fen-file path, malformed input returns `Err` (never
  panics, per conventions); suite integrity — 49 entries, every entry
  (including the three carrying `moves ...` suffixes) parses via the same
  `position fen ... [moves ...]` path the UCI loop uses.
- **Determinism test** (plan step 7): equal node totals across two
  in-process runs; this is the regression net for accidental
  nondeterminism (time-seeded anything, `HashMap` iteration, uninitialized
  memory reads).
- **Perft cross-check**: `bench ... perft` totals must equal the perft
  suite's known counts — ties the bench plumbing to the movegen oracle
  (D24).
- **No golden node count in tests.** The fingerprint lives in commit
  trailers, not in a `#[test]` — otherwise every functional patch would
  edit a test constant and the gate would train people to update it
  blindly. Verification of "non-functional" claims is `tools/check.sh` +
  review, not CI.
- **Criterion baselines** are recorded (`docs/perf-baseline.md`, one table
  per machine) but never asserted in CI: shared/virtualized runners make
  timing assertions flaky. nps gates are human gates on a known machine.
- **Strength validation**: every merge-gating SPRT result gets one line in
  `docs/strength-log.md` (date, commit, bounds, W-D-L pairs, LLR verdict) —
  the audit trail behind any Elo claim (D25).

---

## 6. Performance notes

- The targets ladder and counter expectations live in §3.4; the triage
  order in §3.6. Summary of *what to measure first* for each subsystem as
  it lands: movegen → bulk perft nps; make/unmake + Zobrist → full-make
  perft nps and the gap vs bulk; TT → cold-probe ns and dTLB misses
  pre/post huge pages, plus prefetch coverage (cold-probe cost should
  mostly vanish from flamegraphs once `make_move` prefetches, per the
  TT/board docs); eval A → ns/call and its share of bench cycles (should
  be < 15% for PSQT); NNUE → refresh-to-incremental ratio and SIMD-to-
  scalar ratio.
- **Measurement hygiene** (applies to every number in this doc): pinned
  core, performance governor (`cpupower frequency-set -g performance`),
  idle machine, ≥ 3 runs (5 for claims), median not mean, < 1% is noise.
  On WSL2, prefer elapsed-time medians and cachegrind; treat `perf`
  hardware counters as available-if-lucky and validate any surprising
  counter reading on native Linux before acting on it.
- **The one nps trap**: nps is *not* a fitness function. Optimizing nps by
  making nodes dumber (skipping history updates, thinning eval) is
  strictly worse under SPRT. nps only matters at fixed node semantics —
  which is exactly what the bench fingerprint pins down.

---

## 7. Interactions

Filenames per the design-suite index; adjust numbers if the index shifts
(see Open questions).

**Depends on:**

- Movegen/perft doc — perft entry points and known-count tables that
  `bench ... perft` and `benches/perft.rs` call; **owns the `bench`
  subcommand itself** (§3.6 there: argument grammar, the 49-position
  suite, `src/bench.rs` types, output format) whose numbers this doc's
  fingerprint/determinism policy governs.
- Board/make-unmake doc — node-counting hook (`make_move`), history-`Vec`
  capacity rule (§3.10 item 6), prefetch site this doc measures.
- TT doc — probe/store API for `benches/tt.rs`, huge-pages work validated
  by the dTLB measurements here, `hashfull` for bench output.
- Search doc — `go depth/nodes` limits bench drives; the node counter's
  exact semantics; deterministic-under-depth guarantees.
- Eval docs (phases A/B/C) — `evaluate` seam (D22) for `benches/eval.rs`;
  phase C SIMD paths whose targets are set here.
- Threads doc — per-worker relaxed node counters and aggregation that bench
  reads; the rule that multi-thread bench is nps-only.
- UCI doc — argv dispatch of `bench` before the UCI loop and as an in-loop
  command; `ucinewgame` semantics bench reuses.

**Feeds:**

- Every other doc's §8 acceptance criteria reference the gates defined
  here (perft suite, bench fingerprint, SPRT bounds, criterion baselines).
- decisions.md D24/D25/D26 are operationalized by this doc; CLAUDE.md gets
  the `Bench:` trailer and `tools/check.sh` workflow (plan steps 2, 6).

---

## 8. Acceptance criteria (D26)

This component is done when all of the following hold:

1. `chessengine bench` exists as specified in the perft/validation doc
   §3.6 (SF-shaped argument contract, defaults 16/1/13/default/depth,
   49-position suite, fixed footer), and the fingerprint policy of §3.2 is
   in force.
2. Three consecutive default bench runs on one machine produce **identical**
   `Nodes searched`; the in-process determinism test passes.
3. `bench 16 1 6 default perft` totals equal the known perft counts for the
   suite.
4. Criterion benches for perft, movegen, eval, and TT probe run via
   `cargo bench`; `docs/perf-baseline.md` holds a filled baseline table for
   the primary dev machine, including hot/cold/prefetched TT probe numbers.
5. `[profile.release]` matches D16 and `[profile.profiling]` exists;
   CLAUDE.md documents the `target-cpu` policy.
6. `tools/check.sh` runs fmt/clippy/test/perft/bench and is referenced from
   CLAUDE.md; all search-affecting commits since its adoption carry a
   `Bench:` trailer, and at least one intended-non-functional commit has
   been verified by rerun.
7. `tools/sprt.sh` has executed at least one real SPRT to termination
   (either verdict) with the §3.8 parameters, logged in
   `docs/strength-log.md`; `tools/gauntlet.sh` has produced one ordo-rated
   gauntlet for the current milestone rung; `tools/tactics.sh` has produced
   a baseline tactical-suite score (§3.1 tier 1.5) in
   `docs/strength-log.md`.
8. `tools/profile.sh` reproduces a flamegraph and the `perf stat` counter
   set for bench on a native-Linux machine (or documents the WSL2
   limitation and the cachegrind fallback with a real cachegrind report).
9. The bounds-check binary metric (`objdump | grep -c panic_bounds_check`)
   is recorded in the baseline doc with per-hot-function audits noted for
   movegen and the search inner loop.

---

## 9. Open questions

1. **Cross-doc filenames**: §7 references sibling docs by topic because the
   suite index (final `NN-name.md` filenames) is being authored in
   parallel; reconcile the references once the index is frozen.
2. **Bench suite provenance** — resolved: we adopt SF's standard-chess
   default list, 49 FENs after dropping its two Chess960 entries (positions
   are facts, D27; 960 deferred per D10); the list is printed in the
   perft/validation doc §3.6. Revisit only if our engine's feature set
   makes some positions redundant; either way the list freezes at bench v1.
3. **SPRT model on the fallback harness**: fastchess `model=normalized`
   (nElo) is the primary; cutechess-cli only offers logistic-Elo SPRT. If
   we're ever forced onto cutechess, translate bounds to roughly
   `[0, 3]` logistic at our draw rates — needs a one-time calibration
   against fastchess on the same pair of builds.
4. **Fixed-node budget for proxy games** — resolved (P9 Wave A): calibrated
   to **`N = 200_000` nodes/move** for this build (~5.5 Mnps). `go nodes N`
   is deterministic and hardware-independent (node counting reads no clock,
   §3.2) — verified identical bestmove + node count on repeat. 200 k ≈ 36 ms
   equivalent think-time ≈ ~1/7 of the real 10+0.1 per-move budget (measured
   median ~0.45 M, mean ~1.0–1.4 M nodes/move), reaching ~depth 7–8 so
   Wave-A pruning fires in an STC-representative regime; throughput ~1.45
   games/s (16-way, SMT-tolerant) keeps a per-heuristic SPRT to minutes. Full
   arithmetic, throughput table, and the exact `tools/sprt.sh` invocation are
   in TESTING.md and docs/perf-baseline.md. The proxy gates acceptance
   *within* a wave; the milestone gate still uses real 10+0.1 (§3.8), so a
   known +X Elo change is re-confirmed timed before the wave is declared.

## 10. Multi-agent code-review loop (D33)

Before behavioral/game testing of a substantial batched or one-shot
implementation, run an iterative multi-persona review loop to catch the bugs
unit tests miss — a silent search bug poisons everything downstream (NNUE
labels, SMP search), so cheap adversarial scrutiny is worth it before expensive
game time. **Reviewers are READ-ONLY; only the adjust agent mutates.**

Each **round**:

1. **N parallel reviewer subagents** (default 3), each a distinct persona,
   review the batch (the git range vs the pre-batch base) and return structured
   findings — `{file, line, severity: high|med|low|nit, category, claim,
   suggested_fix}`:
   - **A — correctness & safety.** Mate/draw score integrity; PVS/zero-window
     invariants; pruning/extension guards (`is_loss`/`is_win`/decisive/PV/
     `excluded_move`); singular-search recursion safety; sign / off-by-one /
     margin errors; TT-bound interactions; and bench-node-neutrality of any
     pure-infrastructure commit (e.g. the D30 threat map).
   - **B — reference fidelity.** Diff each heuristic against the proven formulas
     in `stockfish_reference/` (esp. `stockfish_search_notes.md`) and
     `reckless_reference/` — exact constants, divisors, thresholds, operation
     order, guard conditions; flag transcription drift. (Ideas only — D27.)
   - **C — Rust idiom / perf / test quality.** Safe idiomatic Rust (no needless
     `unsafe`, D14/D19); `params.rs` discipline (D20); no heap in hot paths;
     borrow-safety; clippy-clean; readability — AND whether the unit tests
     actually pin the behavior, including the *disarm* conditions.
2. **One synthesis/adjust subagent** ingests all findings, VERIFIES each against
   the code (discarding false positives with a recorded reason), applies the
   warranted code + test fixes, re-runs the correctness gate (fmt / clippy -D
   warnings / `cargo test` debug+release / `tools/check.sh` / the ignored perft
   tier + differential oracle / `selfplay_smoke.py`), and commits
   `"<phase> review round N fixes: <summary>"` (Bench + Co-Authored-By trailers,
   no push).

**Repeat** until a round yields no material (high/med, verified) finding —
**convergence** — or a ~4-round cap trips (report if it does). **Then** game
testing (the SPRT), interpreted per its owning decision (e.g. D32's tripwire
rule for the P9 C-G batch).

**Shape:** a Workflow — `parallel(reviewers)` barrier → adjust agent → loop
while material findings remain. Persona count/mix scales with the change's size
and risk; the three above are the default for search/eval code. Emit the
per-round finding counts so convergence is visible.
