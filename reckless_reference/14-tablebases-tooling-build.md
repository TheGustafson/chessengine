# Reckless — tablebases, tooling, build/PGO, misc

Source: `reckless_reference/Reckless` (v0.10.0-dev, AGPLv3 — ideas only, D27
discipline: prose + own-words pseudocode, no copied blocks, at most one
short formula/constant quoted verbatim below). Primary files: `src/tb.rs`,
`src/tools/{bench,perft,speedtest,mod}.rs`, `src/misc.rs`, `src/lib.rs`,
`src/wasm.rs`, `Cargo.toml`, `build/build.rs`, `Makefile`, `.cargo/config.toml`,
`README.md`, `.github/workflows/{architectures,pgo,release,check_bench,rust,
games,webhook}.yml`. Supporting: `deps/Fathom/{tbprobe.h,tbprobe.c,LICENSE}`
(Fathom is a *separate* MIT-licensed vendored C library, not Reckless's own
code — its header/API can be named more freely than Reckless's own AGPL
source, but I still only describe the API shape, not paste it).

Cross-refs into our own tree: `design/02-core-types.md` §"mate/TB score
bands", `design/10-pruning-extensions.md` §N7 and §9 open question 2,
`design/13-threading-smp.md` §3.10 (`Counters`), `design/14-performance-testing.md`
§3.2/3.7, `design/decisions.md` D14/D15/D16/D24, `CLAUDE.md` fast-build section,
`tools/check.sh`.

---

## 1. Syzygy tablebase integration (Fathom FFI)

### 1.1 How the C library gets into the binary

Fathom (`deps/Fathom/tbprobe.{c,h}`, MIT license, ~70 KB of C) is vendored
as source, not linked as a system library. `build/build.rs:37-56` compiles
it with the `cc` build-dependency (`cc = "1.2.19"`, pinned to `clang`
explicitly, `-O3`, a few `-Wno-*` flags to silence upstream C warnings)
into a static lib named `fathom`, then runs `bindgen` (`0.71.1`) over
`tbprobe.h` to autogenerate the `extern "C"` FFI surface into
`src/bindings.rs` — a **generated, gitignored** file (`.gitignore:8`
lists `bindings.rs`), so nothing hand-written ever drifts from the C
header. This whole step is skipped for `target_arch = "wasm32"`
(`build.rs:22-25`) — TB support simply doesn't exist in the wasm build.

Everything is gated behind a Cargo feature: `[features] default =
["syzygy"]; syzygy = []` (`Cargo.toml:11-13`), and `src/lib.rs:30-35`
conditionally compiles both `mod tb` and `mod bindings` on that feature.
`make no-syzygy` (`Makefile:29-30`) builds with `--no-default-features`
specifically to drop the Clang/Fathom dependency for people who don't
want (or can't satisfy) a C toolchain requirement.

**Distinctive vs us:** this is the one place in Reckless's whole tree that
needs a C compiler and a build-dependency graph (`cc`, `bindgen`, and a
runtime `libc = "0.2.175"` dependency for FFI-compatible types). Everything
else in their engine is pure safe-ish Rust with a handful of narrowly
scoped `unsafe` blocks (relaxed atomics, raw-pointer wasm dispatch, the TB
FFI calls themselves). **Vs our D15** ("zero runtime dependencies... external
tools are tooling, not dependencies") this is a direct conflict if we ever
adopted the same approach: a vendored-and-compiled C library that ships
*inside* the final binary is a runtime dependency in every sense that
matters (build-toolchain requirement, FFI unsafe surface, binary size),
not "tooling" in the SPRT-harness sense the decision was written for.

### 1.2 Two probe entry points, two different contracts

Fathom exposes (see `deps/Fathom/tbprobe.h:207-274` for the documented
contracts, prose only, no code copied):

- **`tb_probe_wdl`** — cheap, thread-safe, meant to be called from
  *every* search thread at *every* qualifying interior node. Its
  simplified wrapper form hard-requires `rule50 == 0` and no castling
  rights (fails otherwise) — i.e. it only proves a result for the
  probeable position itself, no distance-to-zero bookkeeping.
- **`tb_probe_root_dtz`** / **`tb_probe_root_wdl`** — heavier,
  **not** thread-safe, meant to be called exactly once at the root before
  the parallel search fans out. These accept the *real* rule50 clock and a
  repetition flag, and instead of returning one verdict they fill in a
  whole `TbRootMoves` array: every legal root move gets a `tbRank` and
  `tbScore`, so the engine can pick the move that is DTZ-optimal (fastest
  provable conversion), not merely WDL-preserving.

Reckless uses both, for different jobs: `tb::probe` (`src/tb.rs:40-66`)
wraps the cheap WDL probe for interior nodes; `tb::rank_rootmoves`
(`src/tb.rs:68-160`) wraps the two root APIs to rank the actual root move
list once per search.

**Own-words pseudocode for the interior probe:**

```
fn probe(board) -> Option<Win|Loss|Draw>:
    ep = board.en_passant_square & 0x3F        // 0 if none, matches Fathom's convention
    code = fathom::tb_probe_wdl(white_bb, black_bb, K_bb, Q_bb, R_bb, B_bb, N_bb, P_bb,
                                 rule50=0, castling=0, ep, board.side_to_move == White)
    match code { WIN -> Some(Win), LOSS -> Some(Loss), DRAW -> Some(Draw), _ -> None }
```

Note the `_ -> None` arm: Fathom's *unwrapped* probe can also return
`CURSED_WIN`/`BLESSED_LOSS` (a win/loss only by grace of the 50-move
counter being close to reset), but Reckless's thin wrapper never
constructs those variants and just treats them as "no information" at
interior nodes — it silently skips scoring the node from TB and lets
normal search continue. This is a **deliberate simplification vs
Stockfish**, which explicitly assigns cursed/blessed positions a small
non-zero "drawScore" at every probed node. Reckless only handles the
cursed/blessed distinction **at the root** (see §1.4), where Fathom's DTZ
ranking already encodes it in the rank magnitude.

### 1.3 Score encoding — already isomorphic to our own design

Reckless folds TB scores into the *same* magnitude-banded scale as mate
scores (`src/types/score.rs:4-18`): `MATE = 32000`, `MATE_IN_MAX = MATE -
MAX_PLY`, `TB_WIN = MATE_IN_MAX - 1`, `TB_WIN_IN_MAX = TB_WIN - MAX_PLY`
(with `MAX_PLY = 240` that's `TB_WIN ≈ 31759`, `TB_WIN_IN_MAX ≈ 31519`).
`tb_win_in(ply) = TB_WIN - ply`, `tb_loss_in(ply) = -TB_WIN + ply` — exactly
the same ply-relative-to-root shape as `mate_in`/`mated_in`, just one rung
lower on the scale. `is_win`/`is_loss` treat "TB win or better" as one
band (`score >= TB_WIN_IN_MAX`), so every place in the engine that already
special-cases decisive scores (TT storage/retrieval, UCI mate-vs-cp
formatting, aspiration re-search) gets TB scores "for free," no separate
enum or side channel needed. `transposition.rs:309-345` (`score_from_tt`)
folds the *same* stale-score-downgrade logic for both bands: a stored win
whose implied mate/TB distance can't survive the current 50-move clock
(`TB_WIN - score > 100 - halfmove_clock`) is clamped down to
`TB_WIN_IN_MAX - 1` rather than trusted verbatim.

**Not distinctive at all vs us** — this is worth flagging precisely
*because* it isn't a gap: `design/02-core-types.md` already specs
`VALUE_MATE = 32000`, `VALUE_TB = VALUE_MATE_IN_MAX_PLY - 1`,
`VALUE_TB_WIN_IN_MAX_PLY`, `is_win`/`is_loss`/`is_decisive`, and a
`value_to_tt`/`value_from_tt` pair with the identical stale-mate/stale-TB
downgrade shape, all before this study. Reckless independently converged
on the same band layout (which itself descends from Stockfish's
`VALUE_TB` convention) — strong external validation that our doc 02 design
is right. Nothing to change here; just confirmation.

### 1.4 Interior-node integration (search.rs:409-444)

Gated by: not a root node, not an excluded-move (singular-extension)
search, the shared `stop_probing_tb` flag not set, `fiftymove_clock() ==
0`, no castling rights left, and `popcount(occupancies) <= tb::size()`
(the cardinality Fathom's loaded set actually covers, cached from
`TB_LARGEST` at `SyzygyPath` load time). On a hit: increment a per-thread
`tb_hits` counter, map the WDL verdict to a bound (`Win → tb_win_in(ply)` +
lower bound, `Loss → tb_loss_in(ply)` + upper bound, `Draw → 0` + exact),
then:

- if the bound already decides the node against the window (exact, or a
  lower bound ≥ beta, or an upper bound ≤ alpha) — **store into the TT at
  `min(depth + 6, MAX_PLY - 1)`** (a depth *bonus* over the actual search
  depth, since a TB verdict is worth more than a normal leaf) with a null
  move, and return immediately without expanding the node further;
- otherwise, in a PV node, still **tighten `alpha` (win case) or a
  `max_score` ceiling (loss case)** so the remainder of the node's own
  search is bounded by the now-proven truth, even though it can't cut off
  outright.

**This is close to a direct implementation of what our own doc 10 §N7
already speced as plumbing** ("if the implied bound already decides the
node... store at depth+6 and return; in PV nodes tighten best_value/
max_value instead... we keep max_value and the final clamp wired from day
one so Syzygy can drop in later" — `design/10-pruning-extensions.md:352-360`).
Reckless is essentially the reference implementation of exactly that
shape. One added subtlety worth folding into our doc if we ever activate
N7 for real: Reckless's depth-bonus constant is `+6` plies, applied via
`.min(MAX_PLY - 1)` to avoid overflowing the TT's depth field — a small,
concrete number we didn't have pinned yet.

### 1.5 Root-level DTZ/WDL ranking and the repetition reconciliation trick

`threadpool.rs:162-167`: before the parallel search fans out, if the root
position has no castling rights, is within TB cardinality, and is not
already a draw by the engine's own detector, thread 0 calls
`tb::rank_rootmoves` then `tb::normalize_draw_ranks` — **before any
worker thread is spawned**, consistent with Fathom's "root APIs are not
thread-safe" contract.

**`rank_rootmoves` (own-words):**

```
fn rank_rootmoves(td):
    build a C-shaped TbRootMoves buffer, one slot per current root move
    (converting our Move to Fathom's u16 TbMove encoding)

    if fathom::tb_probe_root_dtz(<position>, real_rule50, ep, turn,
                                   has_repeated, &mut buffer) succeeds:
        copy tbScore/tbRank back into our RootMove list (matched by move equality)
        sort root moves descending by tbRank
        root_in_tb = true; stop_probing_tb = true   // DTZ is exact — trust it fully
        return

    // DTZ tables absent/incomplete for this position — fall back to WDL-only ranking
    if fathom::tb_probe_root_wdl(<position>, real_rule50, ep, turn, &mut buffer) succeeds:
        copy scores/ranks back, sort descending
        root_in_tb = true
        // only keep spending time on TB probes deeper in the tree if we're
        // actually winning without a forced-mate distance to lean on —
        // a draw or loss is already the final practical answer
        stop_probing_tb = (best_root_tb_score <= 0)
```

The `stop_probing_tb <= 0` heuristic is a nice one-line cost/benefit call:
once the root outcome is *known*, further interior-node TB probes only
earn their keep if they help the search actually *find* the win faster
(mating technique, avoiding 50-move drift); if the truth is a draw or a
loss, more probing buys nothing practical, so switch it off and let the
freed node budget go to normal search.

**`normalize_draw_ranks` (own-words)** — the defensive step that isn't in
Fathom at all, and is worth calling out on its own: for every ranked root
move, actually **make the move on the real board and ask the engine's own
draw detector** (repetition history + 50-move rule, `board.is_draw(1)`)
whether the resulting position is a draw; if so, force that move's
`tb_rank`/`tb_score` down to a plain draw (0), overriding whatever Fathom
said. **Why this matters:** DTZ/WDL tables only see the tablebase position
itself — they have no idea a "winning" continuation actually repeats a
position from *earlier in this specific game*. Without this reconciliation
step the engine could report a winning score while the move it's about to
play is a real, rules-enforced draw. This is a genuinely useful technique
independent of Fathom/Syzygy specifics: **any oracle that scores positions
in isolation needs to be cross-checked against game-history-dependent draw
rules before its verdict is trusted.**

### 1.6 UCI score reporting (`thread.rs:298-375`)

When `root_in_tb` is set, `print_uci_info` distinguishes a "cursed"
result — `tb_rank` in `(0, 900)` or `(-900, 0)`, i.e. a win/loss that only
holds courtesy of the 50-move-rule reset window, per Fathom's own rank-band
convention — and reports it as a plain `cp 0` (since practical play will
likely see it drift to a real draw), versus a clean win/loss, which
substitutes the TB's own score for the search window's bound/score and
turns off the `upperbound`/`lowerbound` flags (a TB verdict is exact truth,
not a windowed guess). For decisive-but-not-mate TB scores it also remaps
into a distinct, very large centipawn band rather than a `mate N` string —
the one formula I'll quote verbatim since the exact constant is the point:
`cp = 20_000 - Score::TB_WIN + |score|` (`thread.rs:344`) — a GUI-facing
convention (a huge `cp` number distinguishes "TB win, not yet a forced
mate" from both ordinary eval and genuine `mate N`), mirroring Stockfish's
own `VALUE_TB` cp-inflation convention for GUIs.

`tb_hits` is a per-thread sharded counter (`Counter` — an array of
cache-line-sized atomics, one shard per thread, summed on read; the exact
same shape as their `nodes` counter) incremented on every successful
interior probe and surfaced as the UCI `tbhits` info field
(`thread.rs:78-102` for `Counter`, `:365` for the info line).

**Already anticipated in our own design:** `design/13-threading-smp.md:603-606`
already specs a `Counters` struct with `nodes`, `tb_hits`, and
`best_move_changes` as three cache-line-isolated `AtomicU64`s — i.e. we
already planned the exact same counter shape for a `tb_hits` field years
before deciding whether to build TB support at all. Wiring real TB
probing in later is a small increment on top of already-planned plumbing.

### 1.7 A wart worth noting, not copying

`tb::size()`/`initialize()` cache `TB_LARGEST` in a bare `static mut usize`
(`src/tb.rs:20,27,36-38`), guarded only by the informal invariant that
`initialize` runs once, synchronously, from the UCI/main thread, strictly
before any search thread reads `size()`. It works in practice but is a
real "unsafe mutable static" with no atomic/lock backing — the kind of
thing our own D14 unsafe budget (prefetch intrinsics, SIMD intrinsics,
profile-justified unchecked indexing) would not authorize as written. If
we ever build the equivalent, an `AtomicUsize` or `OnceLock<usize>` costs
nothing and removes the wart entirely.

---

## 2. Tooling: `bench` / `perft` / `speedtest`

### 2.1 `bench.rs` — the fingerprint tool

`src/tools/bench.rs:20-71` compiles in a fixed 50-position FEN suite
(quiet openings through 7-man endgames through three Chess960 setups),
`DEFAULT_DEPTH = 12`, `DEFAULT_HASH = 16` MiB, `DEFAULT_THREADS = 1`
(`:73-75`). The `bench<const PRETTY: bool>(args)` function
(`:77-128`) is monomorphized twice via a **const-generic bool** rather than
a runtime branch or two near-duplicate functions: `Mode::Uci` (typed at
the interactive UCI/console prompt) calls `bench::<true>` and gets a
per-position table plus totals; `Mode::Cli` (invoked as `reckless bench`
from argv, which is what CI's `check_bench.yml` parses) calls
`bench::<false>` and gets exactly one machine-parseable line,
`Bench: {nodes} nodes {nps} nps`. Same loop body, zero runtime cost for
the branch, compiler picks the print statements out at compile time — a
clean use of const generics for "verbose vs terse" output variants that
share all their logic.

**Distinctive vs us:** our own `bench` subcommand (design/06 §3.6, D24)
already does the single-fingerprint-line-plus-optional-verbosity job, so
this is confirmation of a shape we already have, not a new idea — except
the *mechanism* (const-generic bool monomorphizing two print paths off one
loop) is a nicer Rust idiom than a runtime `if pretty` branch threaded
through the loop, worth adopting verbatim as a technique regardless of TB.

### 2.2 `perft.rs` — three variants, one of them a strong differential oracle

- `perft` (`:13-48`) — verbose per-root-move divide with per-move and
  total nps, SF-`divide`-shaped.
- `simple_perft` (`:50-67`) — bare divide, no timing, for scripting.
- `is_legal_perft` (`:69-122`) — the interesting one. Instead of driving
  the tree with the normal move generator, it enumerates **every possible
  16-bit move encoding**, `0..=u16::MAX`, throws out the encodings whose
  top nibble corresponds to a promotion/move-kind combination that can
  never legally occur (own-words: three reserved bit patterns are
  skipped), `transmute`s each survivor into a `Move`, and keeps it only if
  `mv.is_present() && board.is_legal(mv)`. That predicate-only move list
  then drives its own perft, independent of the real move generator
  entirely. It's a **legality-predicate-only alternate movegen used purely
  as a differential oracle**: if this perft's counts ever diverge from the
  normal generator's counts, either the generator is missing/over-producing
  moves or `is_legal` disagrees with it — either way, a bug the normal
  perft suite alone wouldn't catch (it never questions whether the
  generator's *candidate set* is complete; this brute-forces the entire
  16-bit space instead of trusting the generator to enumerate candidates
  at all).

**Distinctive vs us:** CLAUDE.md's P3 summary already describes "the
TT-move validator (`pseudo_legal` + 64k fuzz)" — i.e. we already fuzz the
*entire* 16-bit move space through the legality predicate for TT-move
sanitization. Reckless takes the identical fuzz and additionally wires it
into a full alternate perft, cross-checked against the real generator's
node counts. We have the harness already; extending it to drive a perft
comparison is a small, high-confidence addition (see Adapt to us).

### 2.3 `speedtest.rs` — self-documenting throughput probe

`src/tools/speedtest.rs:70-122`: runs the *same* 50-position suite (see
§2.4) at a fixed wall-clock budget per position (`duration_secs * 1000 /
50`), with an explicit 3-position warmup pass first (`WARMUP_POSITIONS_COUNT
= 3`, `:68`) to let the TT/branch-predictor/cache state settle before the
timed measurement — avoids a cold-start skew on the very first position.
Before running, it prints `env!("ENGINE_VERSION")`,
`env!("COMPILER_VERSION")`, `env!("COMPILER_TARGET")`,
`env!("COMPILER_FEATURES")` (`:78-81`) — all baked in at compile time by
`build.rs` (see §3.2) — so a pasted speedtest result is self-describing:
anyone reading a report knows the exact rustc version, target triple, and
`-C target-feature` set the binary was built with, without having to ask.

### 2.4 A smell worth *not* copying

The 50-entry `POSITIONS` FEN list is duplicated **verbatim** between
`tools/bench.rs:20-71` and `tools/speedtest.rs:11-62` — two independent
copies of the same const array in two files, no shared module. This is a
place where **our own design is already better**: doc 06 §3.6 owns a
single frozen `BENCH_POSITIONS` list that everything else references, by
design, specifically to avoid exactly this kind of drift risk. Nothing to
adopt here — just noted so a future reader doesn't mistake the DRY
violation for something to imitate.

---

## 3. Build system: profile, build-script codegen, target-cpu policy, wasm

### 3.1 Release profile — already the same as ours

`Cargo.toml:21-24`: `lto = "fat"`, `panic = "abort"`, `codegen-units = 1`.
Byte-for-byte the same shape as our own D16 / `design/01-architecture.md`
§3.7 / `design/14-performance-testing.md` §3.7 release profile. No new
idea here — strong outside confirmation that a top-2-strength engine
converged on the identical release-profile recipe we already pinned.

### 3.2 Build-script codegen (`build/build.rs`)

Four independent jobs, worth naming individually since each is a small,
reusable pattern:

1. **Attack/magic table codegen** (`:68-99`) — writes a `lookup.rs` into
   `OUT_DIR` containing the fully materialized leaper/ray/magic tables as
   `static` array literals, generated by ordinary Rust functions
   (`maps::generate_*`, `magics::ROOK_MAGICS`/`BISHOP_MAGICS`) run *at
   build time* rather than either hand-written consts or a `LazyLock`
   computed at process startup. This is a genuine difference from our own
   approach (`design/01-architecture.md` deliberately keeps magic-table
   construction behind `LazyLock<AttackTables>` computed once at process
   start, D12) — Reckless instead pays the one-time cost at `cargo build`
   time and ships the tables as literal static data, trading build-time
   for a startup-time win (no first-call `LazyLock` init latency, ever,
   and the tables are `.rodata`, not heap/`OnceLock`-guarded).
2. **Compiler/version provenance** (`:116-148`) — shells out to `rustc
   --version`, reads `TARGET`/`CARGO_CFG_TARGET_FEATURE` env vars, and
   `git rev-parse --short=8 HEAD`, baking all of it into `env!()`-visible
   constants (`COMPILER_VERSION`, `COMPILER_TARGET`, `COMPILER_FEATURES`,
   `ENGINE_VERSION = "{version}-{sha}"`). Surfaced via the `compiler` UCI
   command (`uci.rs:171-175`) and at the top of `speedtest` output (§2.3).
3. **Network download** (`:101-114`) — if no `EVALFILE` env var and no
   local `networks/<name>.nnue` file exists, `curl`s the pinned NNUE net
   from a GitHub release URL at build time. Not relevant to us before
   P13, but the *pattern* (a build.rs that fetches a large pinned binary
   artifact only if missing, keyed by an override env var) is directly
   reusable for our own NNUE bootstrap distribution once P13 lands.
4. **Syzygy FFI codegen** (§1.1).

### 3.3 `target-cpu` policy and the three-tier binary convention

`.cargo/config.toml` pins `target-cpu=native` per-target as the
*default* for anyone running `cargo build` locally (including
`wasm32-unknown-unknown`, which instead pins a fixed wasm feature set:
`+atomics,+bulk-memory,+simd128,+relaxed-simd`, plus linker flags for
shared-memory + explicit TLS symbol exports — see §3.4). `Makefile:4`
documents the same `native` default with an override hook via `RUSTFLAGS`.

For **releases** (`release.yml`) and **CI compile smoke-checks**
(`architectures.yml`), the policy shifts to three fixed, portable tiers,
each a genuinely separate static build:

- `generic`: `-C target-cpu=x86-64` (baseline v1, runs anywhere)
- `avx2`: `-C target-cpu=x86-64-v3`
- `avx512`: `-C target-cpu=x86-64-v4 -C target-feature=+gfni,+avx512bw,
  +avx512vl,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512bitalg` (an
  explicit curated AVX-512 feature set, not just `-v4`'s default subset)

No runtime CPU dispatch / function multiversioning anywhere — same
philosophy as our own D16 (one documented `RUSTFLAGS` convention, no fat
binary). The distinctive part is doing this **three times over, fully
separately, including three independent PGO runs** (§4.1) — not a new
idea vs our single-tier D16 policy, but a concrete template for what a
future multi-tier release process would look like once we start shipping
binaries to non-developers.

### 3.4 The wasm build

`src/lib.rs:37-38` / `wasm.rs` gate a second, thin "front end" behind
`cfg(target_arch = "wasm32")`, symmetrical to the native UCI loop: same
core `board`/`search`/`nnue`/`threadpool` modules, wrapped by a
`#[wasm_bindgen] struct Engine` exposing `set_position`, `make_move`,
`go_uci`/`go_movetime`, `evaluate`, `last_bestmove`/`last_score`, etc.
(`wasm.rs:15-113`). Threading is the hard part: wasm32 has no `std::thread`
in this configuration, so multi-threaded search is implemented by handing
JavaScript a **raw pointer** to each helper thread's `ThreadData`
(`js_sys::Array` of pointers cast to `u32`, `threadpool.rs:124-133`) and
letting a JS-side dispatch callback (`WASM_DISPATCH`, a `thread_local!
RefCell<Option<Function>>`, `thread.rs:10-17`) spin up real Web Workers
that each call back into a `run_helper_thread(ptr, thread_count)` free
function (`wasm.rs:121-126`), which unsafely reconstitutes the pointer
into a `&mut ThreadData` and calls the normal `search::start`. Progress
lines (`info depth ...`) are routed back to JS through a second
thread-local callback (`WASM_CALLBACK`) rather than `println!`, since
there's no stdout in a browser. Build recipe: nightly toolchain +
`-Z build-std=panic_abort,std` (to get a `panic=abort` std for wasm),
`wasm-bindgen --target web`, then `wasm-opt -O3 --enable-simd
--enable-threads --enable-relaxed-simd` as a post-link pass
(`Makefile:38-44`).

**Not on our roadmap** (nothing in `design/00-master-plan.md` mentions a
wasm/browser target), but the pattern — one shared core, two thin
front-ends selected by `cfg(target_arch)`, native `std::thread` swapped
for a JS-orchestrated Web Worker pool with raw-pointer handoff — is the
standard playbook if we ever wanted a browser demo, and it's a clean,
low-risk generalization of our own "library crate + thin binary" split
(CLAUDE.md) to a second front-end rather than a competing architecture.

---

## 4. CI: PGO pipeline and cross-microarchitecture bench determinism

### 4.1 Per-target PGO, SDE-assisted training on commodity CI hardware

`pgo.yml` and `release.yml` both drive `cargo-pgo` (a wrapper around
LLVM's instrument → train → optimize PGO flow): `cargo pgo instrument`,
then `cargo pgo run -- bench` as the training workload (their own `bench`
subcommand doubles as the PGO training corpus — no separate training set
needed), then `cargo pgo optimize`. The distinctive part is in
`release.yml:66-98`: this three-step pipeline runs **once per release
target** (generic/avx2/avx512 × linux/windows, plus native macOS) — you
cannot reuse one profile across ISA levels, since the instrumented binary
must actually execute AVX-512 code paths to produce a useful AVX-512
profile. For the avx2/avx512 Linux and Windows targets specifically, the
*training run itself* executes under **Intel SDE** (Software Development
Emulator: `"$SDE_PATH/sde64" -bdw`/`-icx -- cargo pgo run -- bench`,
`release.yml:72-85`) — i.e. CI hardware that may not physically support
AVX-512 can still produce a valid AVX-512 PGO profile by emulating the
instructions during the training run. This turns "we need real AVX-512
silicon to PGO-optimize the AVX-512 release" into "we need SDE," which
runs on any x86-64 CI box.

### 4.2 `check_bench.yml` — bench determinism across ISA levels, not just across commits

This is the most transferable idea in the whole study. Reckless's version
of our own `Bench:` trailer discipline: parse the last commits' messages
for a `Bench: N` trailer, rebuild, run `bench`, and fail if the measured
count doesn't match — same convergent idea as our D24 (own-words: both
projects independently invented "the commit message carries the expected
node count, CI recomputes it and fails on drift"). **The extension we
don't have**: `check_bench.yml` runs this check under **five different
Intel SDE microarchitecture emulations** (`-nhm` Nehalem-class generic,
`-hsw` Haswell, `-skx` Skylake-X server, `-icl`/`-icx` Ice Lake
client/server) *plus* a sixth native run on real Apple Silicon
(`check-bench-arm` job, `:88-128`, confirming the NEON code path compiles
and reproduces the same trailer natively, since SDE doesn't emulate ARM).
Every one of these must reproduce the **exact same node count** as the
trailer. This catches a bug class our current single-native-runner CI
cannot: search behavior that is accidentally ISA-dependent (e.g. an eval
or move-ordering computation whose result depends on SIMD lane width,
reduction order, or FMA contraction/reassociation differing between AVX2
and AVX-512 codegen at the same optimization level). It's low-value for
us **today** (pure-integer PSQT eval has no such reassociation hazard),
but becomes directly relevant the moment NNUE (P13) introduces any
floating point or width-sensitive horizontal-sum code — exactly the class
of bug this check is built to catch before it ships.

### 4.3 `architectures.yml` — compile-only smoke matrix, a cheaper prerequisite check

Separate from §4.2: this workflow only runs `cargo check` (not `bench`)
across the four x86-64 tiers (v1–v4, with the same curated AVX-512
feature list) on both Linux and Windows, plus a macOS `native` check and a
dedicated `wasm32` job. It answers "does the code even compile under each
target-feature gate" (catches `#[cfg(target_feature = "...")]` gating
mistakes, e.g. an AVX-512 intrinsic accidentally reachable from a
`v3`-gated path) without paying for a full bench run — a cheap, fast-fail
prerequisite that would normally run before the much more expensive SDE
bench-determinism gate.

### 4.4 `games.yml` — a fastchess self-play smoke test as its own CI job

Plays the engine against itself for a handful of very short (`tc=8+0.08`)
games from a fixed opening book with `-concurrency 4`, using **debug
builds** (`cmd=../target/debug/reckless`), and fails the job if the game
log contains `illegal move`, `disconnect`, or `stall`. This is *not* a
strength test (no SPRT, no Elo claim) — it's a protocol-and-crash smoke
test: does the engine ever emit an illegal move, hang, or crash under
real GUI-adjacent conditions, on every push/PR. Distinct from and
cheaper than a full SPRT run, and running it on **debug** builds is
deliberate — debug assertions (their equivalent of our
`assert_consistent` oracle, likely) are compiled in and will abort loudly
on the first invariant violation rather than silently producing an
illegal move.

---

## 5. `misc.rs` — an always-on, opt-in-by-callsite instrumentation channel

`src/misc.rs` (`#![allow(dead_code)]`) implements the same job as
Stockfish's `dbg_hit_on`/`dbg_mean_of`/`dbg_stdev_of`/`dbg_correl_of`
family (`stockfish_reference/Stockfish/src/misc.cpp:341-383`) as two fixed
32-slot arrays of atomics (`HITS`, `STATS`, `:8-11`), each slot a small
`Wrapper` holding three `AtomicI64` counters plus a `Mutex<Vec<i64>>` of
raw samples. `dbg_hit(condition, slot)` (`:91-99`) is the cheap path:
bump a total, bump a hit count if true, print `Hits/Total/Rate%` at the
end of a search (`dbg_print`, called after every `bestmove` in `uci.rs:250`
and after `bench`/`speedtest`). `dbg_stats(value, slot)` (`:101-110`) is
the richer path: accumulates count/sum/sum-of-squares for a running
mean/stddev, **and** pushes the raw value into the slot's `Vec` so
`dbg_print` can additionally report median, min, max, and the **Gini mean
difference** (`:50-71` — a robust dispersion measure, an average of all
pairwise absolute differences, less sensitive to outliers than stddev) —
strictly more diagnostic power than Stockfish's mean/stdev-only version,
at the cost of an unbounded per-slot `Vec` and a lock on every sample.

**A quirk worth noting, not copying:** `HITS` and `STATS` share the same
`Wrapper::add` method, whose "push to the sample vector" branch is keyed
on `index == 1` (`:26-33`) — the convention `dbg_stats` uses for "this is
the value being accumulated." But `dbg_hit`'s hit-branch *also* calls
`add(1, 1)` on a true condition, which means every single hit-counter
increment silently pushes an unused `1_i64` into a growing, mutex-guarded
`Vec` that `dbg_print` never reads back for `HITS` slots (only `STATS`
slots get their median/GMD/min/max printed). It's harmless (no incorrect
output) but a real, avoidable cost leak from sharing one struct across two
call conventions that don't actually need the same fields. If we ever
build the equivalent, keep the "cheap hit-rate counter" and "expensive
distribution-sample recorder" as two distinct types rather than one
struct with a magic index.

**Const-array-of-non-`Copy`-statics idiom worth adopting regardless of
TB/misc specifics:** `[const { Wrapper::new() }; SLOTS]` (`:10-11`,
also used for `[const { AtomicU32::new(0) }; MAX_MOVES]` in
`thread.rs`'s `SharedContext::default`) is the stable "inline const in an
array-repeat expression" idiom (stable since Rust 1.79) for initializing a
fixed-size array of a type that is neither `Copy` nor has a `const`
`Default` — here a struct containing a `Mutex`. It sidesteps the older
`MaybeUninit`-array-plus-manual-drop dance or a runtime `Vec`-then-
`try_into` for exactly this case, with zero unsafe. Directly usable
anywhere we need a fixed array of non-`Copy` statics (e.g. per-square or
per-slot atomics/locks) without reaching for `LazyLock` or unsafe.

---

## Adapt to us

Ordered roughly by effort, cheapest first.

1. **Const-generic verbose/terse output split for `bench`.**
   `bench<const PRETTY: bool>` (§2.1) instead of a runtime flag threaded
   through the print loop. *Touches:* design/06 §3.6 (`src/bench.rs`
   `BenchParams`/`run`). *Phase:* any time (pure refactor, no `Bench:`
   trailer impact since it's output-formatting only). *Effort:* trivial
   (an hour). *Impact:* small code-quality win, zero behavior change.

2. **`[const { T::new() }; N]` idiom for fixed arrays of non-`Copy`
   statics.** (§5, last paragraph.) *Touches:* nowhere specific yet — a
   generally-useful idiom to reach for instead of `LazyLock`/unsafe
   whenever we need e.g. per-slot atomics or locks in a fixed array.
   *Phase:* opportunistic, any phase. *Effort:* trivial. *Impact:* small
   simplification wherever it applies; mostly a "know this exists" note
   for future code.

3. **Bake compiler/target/feature provenance into the binary via
   `build.rs` + a `compiler` command.** (§2.3, §3.2 item 2.) *Touches:*
   `design/14-performance-testing.md` §3.6/§3.7 (profiling/triage
   workflow) and our `bench`/future `speedtest`-equivalent output; maybe a
   new UCI/CLI `compiler` command. *Phase:* any time, useful now (P9) for
   triaging nps regressions and even more useful once we start caring
   about x86-64-v3 vs v4 dispatch or PGO. *Effort:* small (a `build.rs`
   `env!` shim plus one UCI branch, an afternoon). *Impact:* medium —
   makes every bench/report/bug-report self-documenting about exactly
   which `RUSTFLAGS`/toolchain produced it, removing a whole class of "what
   flags did you build with" triage friction.

4. **Extend the existing 64k move-encoding fuzz into a differential
   perft oracle** (`is_legal_perft`, §2.2). We already fuzz the full
   16-bit move space through `pseudo_legal`/`legal` (CLAUDE.md P3 status).
   Wiring that same fuzzed, legality-predicate-only move list into a perft
   comparison against the real generator's counts is a small addition on
   top of infrastructure we already have. *Touches:* design/06 §3.5 (perft
   tiers) / §9 (open questions) — candidates as a new perft tier or an
   extension of the existing fuzz test. *Phase:* opportunistic, fits
   anywhere in the ongoing D24 perft ritual, no dependency on later
   phases. *Effort:* small–medium (a day: build the predicate-only move
   enumerator, wire it into the existing perft harness, decide whether it
   gates fast/full/extended tier). *Impact:* medium — a strictly stronger
   differential check on movegen/legality agreement than what we run
   today, catching a bug class (generator produces a strict subset/superset
   of what the predicate accepts) the current suite doesn't specifically
   target.

5. **Cross-microarchitecture bench-determinism CI gate via Intel SDE**
   (§4.2). Emulate 3-5 x86-64 microarchitectures (a baseline v1, an AVX2
   one, an AVX-512 one at minimum) and assert the `Bench:` trailer node
   count is identical across all of them, on top of the existing
   single-native-runner check. *Touches:* `design/14-performance-testing.md`
   §3.2/§3.9 (bench policy, merge-gate checklist) and `tools/check.sh`
   (a CI-only addition, not a local pre-commit step). *Phase:* low
   priority now (pure-integer PSQT eval has no known ISA-reassociation
   hazard); becomes genuinely valuable once P13 (NNUE) lands, since SIMD
   width/lane-order differences in a forward pass are exactly the bug
   class this catches before it reaches an SPRT run. *Effort:* medium (CI
   setup: Intel SDE tool install, a matrix job, teaching the check to
   parse our own `Bench:` trailer — the parsing logic is a near-identical
   twin of `check_bench.yml`'s `git log` grep). *Impact:* medium-high but
   deferred — prevents an entire class of "works on my machine, silently
   wrong on the release binary" bugs, specifically timed to matter most
   right when NNUE SIMD code lands.

6. **Fastchess self-play smoke test as its own fast CI job** (§4.4,
   distinct from full SPRT). A handful of very short self-play games on
   every push, checking only for illegal moves/disconnects/stalls (not
   Elo). *Touches:* `design/14-performance-testing.md` §3.8 (strength
   testing) as a new, cheaper *tier 0* sitting below tier 1.5
   (tactics) — a protocol/crash smoke test, not a strength claim.
   *Phase:* any time; most useful once P10 lands full UCI (time
   management edge cases — stalls/timeouts under real GUI-style clocks are
   exactly what this would catch that unit tests and bench don't).
   *Effort:* small (reuse our existing `tools/gauntlet.sh`/fastchess
   setup with a tiny game count and a log-grep for failure strings).
   *Impact:* medium — cheap, fast, catches a different bug class (protocol
   violations, hangs) than SPRT or bench ever would, on every commit
   instead of only when someone remembers to run a gauntlet.

7. **Per-target PGO with SDE-assisted training for higher ISA tiers**
   (§4.1). If/when we ship prebuilt binaries and adopt PGO (already
   flagged as "optional, later, after NNUE" in
   `design/14-performance-testing.md` §3.7), train each target tier
   (generic/v3/v4) *separately*, and use Intel SDE to execute the AVX-512
   training run on CI hardware that doesn't physically have AVX-512.
   *Touches:* `design/14-performance-testing.md` §3.7 (PGO pipeline) and
   `design/00-master-plan.md` P14 (tuning/parity chase) if we ever start
   distributing prebuilt binaries. *Phase:* P14 at the earliest (matches
   our own doc's "after NNUE" sequencing). *Effort:* medium (multi-job CI
   matrix, SDE integration, `cargo-pgo` wiring — mechanical once the
   single-target PGO flow already works). *Impact:* the profile-per-tier
   correctness matters more than a big nps number: reusing one tier's
   profile for another risks training on cold paths that are actually hot
   under the other tier's codegen.

8. **Should we build Syzygy tablebase support? Assessment: not now,
   revisit post-P13, and scope it as its own decision + doc when we do.**
   The scoring/plumbing side is essentially already designed —
   `design/02-core-types.md`'s `VALUE_TB` band and
   `design/10-pruning-extensions.md`'s N7 interior-probe shape already
   match Reckless's implementation almost line-for-line (§1.3, §1.4), and
   `design/13-threading-smp.md` already has a `tb_hits` counter slot
   waiting (§1.6). What's genuinely missing is (a) the root-level DTZ
   ranking + the `normalize_draw_ranks`-style reconciliation against our
   own repetition detector (§1.5 — not yet speced anywhere), and (b) most
   importantly, **the dependency question**: real Syzygy support needs a
   vendored C library (Fathom, MIT — fine to vendor), a C compiler at
   build time (`cc` + `clang`), a `bindgen` build-dependency, and a
   runtime `libc` dependency, plus a new, larger class of `unsafe` FFI
   surface than anything D14 currently authorizes. That's a direct
   conflict with D15 as worded today, so shipping it — even opt-in — would
   need a new pinned decision (own doc + a `decisions.md` entry, "decisions
   change here first") explicitly carving out an FFI/build-dependency
   exception scoped to a **non-default** `syzygy`-style Cargo feature
   (mirroring Reckless's own feature gate, but defaulting *off* rather
   than *on* so our default `cargo build`/`cargo test` path stays exactly
   as zero-dependency as D15 intends). Given current roadmap priorities
   (P9 pruning waves → P10 time-mgmt/UCI → P11 lazy SMP → P13 NNUE) and
   that TB's Elo contribution matters most at very high playing strength
   (multi-thousand-Elo engines converting known-won endgames perfectly,
   not a ~2000-2200-class engine's main lever), this is correctly
   deferred — doc 10 §9's existing "decide during Wave E" stance is right;
   I'd push the actual decision even later, to after P13, since NNUE will
   move the Elo needle far more per unit effort, and by then the
   already-matching plumbing in doc 02/doc 10 means the search-side
   integration itself would be a small, low-risk addition once the
   dependency question is separately and deliberately resolved.
   *Touches:* a future `design/17-tablebases.md` (or an appendix to doc
   10, per the open question already on record) + a new D-numbered
   decision. *Phase:* not before P13; revisit then. *Effort:* the FFI/build
   plumbing itself is medium (Fathom vendoring + bindgen + feature-gating,
   all precedented by Reckless); the search-side integration is small
   given how much is already speced. *Impact:* likely low-to-medium Elo at
   our current strength tier, rising as strength rises — correctly a
   late-phase item, not a P9/P10 distraction.
