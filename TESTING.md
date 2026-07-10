# Testing and evaluation guide

How we know the engine is correct, fast, and strong — and which tool answers
which question. This file is the map; the binding contracts live in the
design suite, chiefly `design/06-perft-validation.md` (perft + bench spec)
and `design/14-performance-testing.md` (measurement policy + strength
testing). Where this guide and a design doc disagree, the design doc wins.

The framework exists **before** the engine components it tests: every phase
starts against an oracle that is already in place (see "The RED protocol").

## The oracle hierarchy

Three oracles, in the order they gate development. Using the wrong tier for
a claim is the classic testing mistake — design/14 §3.1 has the full table.

| Oracle | Question it answers | Spec |
|---|---|---|
| **Perft suite** — exact node counts over the 23 positions in `tests/data/perft.epd` | "Do movegen and make/unmake produce exactly the legal move tree?" | design/06 §3.4–3.5 |
| **Bench fingerprint** — `chessengine bench` total node count | "Did search *behavior* change? Did nps change?" | design/06 §3.6, design/14 §3.2 |
| **SPRT games** — fastchess self-play with statistical stopping | "Is the engine *stronger*?" | design/14 §3.8 |

Rules of thumb (design/14 §3.1):

- Perft is the ground truth for anything touching `movegen`, `board`
  mutation, `zobrist`, or `attacks` — no such change lands without the suite
  passing (D24).
- A patch that changes the bench node count is *functional* by definition
  and needs SPRT if it touches search/eval; a "non-functional" refactor must
  reproduce the fingerprint bit-for-bit.
- Strength claims come from games only, never from eyeballing or nps (D25).
- Below these sit tier 1.5 (tactical EPD sanity, design/14 §3.1) and tier 3
  (criterion micro-benches, design/14 §3.3) — neither ever gates a merge.

## Test tiers and how to run them

```sh
cargo test                     # the default check after ANY change (< 10 s)
cargo test --release           # same non-ignored set, release codegen (CI runs both)
cargo test --release -- --ignored perft_full_suite   # full perft tier (D24 gate)
cargo test --release -- --ignored perft_extended     # extended tier (releases/rewrites)
tools/check.sh                 # the whole merge gate, in order
python3 tools/verify_perft.py --suite   # re-verify the perft oracle data itself
cargo bench                    # criterion micro-benches (never a gate; baselines in docs/perf-baseline.md)
```

- **`cargo test` (debug)** — unit tests, property tests, and (once P3 lands)
  the fast perft tier (design/06 §3.5 tier 1, budget < 10 s). Must be green
  at all times, on every commit.
- **Full perft tier** — release-mode, permanently `#[ignore]`d test in
  `tests/` run explicitly by name (budget < 30 s, design/06 §3.5 tier 2).
  Mandatory for movegen/make-unmake/attack-table changes.
- **Extended tier** — ~18 G nodes, < 4 min; before releases and after any
  movegen/make-unmake rewrite only (tier 3 of the perft suite).
- **`tools/check.sh`** — codifies the per-change checklist of design/14
  §3.9: fmt, clippy (`-D warnings`), `cargo test`, the release perft tier,
  and the bench fingerprint. Steps whose components haven't landed yet print
  `SKIP: ... TODO(Pn)` and activate automatically once the binary supports
  them. Run it before every merge; CI runs the hosted subset
  (`.github/workflows/ci.yml`).
- Don't run a bare `cargo test -- --ignored`: it would execute *every*
  ignored test, including RED tests for unbuilt components (below), which
  fail by design. Ignored tests are always invoked selectively by name.

## The RED protocol (test-first phases)

Tests for components that don't exist yet land **compiling but ignored**:

```rust
#[ignore = "RED: un-ignore when P3 movegen lands"]
```

- The marker names the phase (`Pn`, per `design/00-master-plan.md` §3) and
  the component. Because the tests are ignored, `cargo test` and CI stay
  green while the oracle waits.
- **Starting a phase means removing its RED markers.** That is the failing
  bar you code against: un-ignore, watch the tests fail, implement until
  green. Never delete or weaken a RED test to get there.
- Find the current backlog: `grep -rn "RED: un-ignore" src tests benches`.
- Distinguish RED ignores from *permanent* ignores (e.g. the release-only
  perft tiers, whose ignore reason states the command that runs them): a
  permanent ignore survives its phase; a RED marker must not.

## Debugging a wrong perft count

When a perft total disagrees with the table (design/06 §3.4):

1. **Rule out the oracle data** — `tools/verify_perft.py` recomputes
   `tests/data/perft.epd` with python-chess (an independent legal-move
   implementation, `pip install chess`; D27: counts are facts, cited
   exactly). It works today and is how new counts get verified *before*
   they may gate the engine:

   ```sh
   python3 tools/verify_perft.py --suite                      # verify all counts <= 1M nodes
   python3 tools/verify_perft.py --suite --max-nodes 20000000 # raise the per-count budget
   python3 tools/verify_perft.py --suite --line 60            # verify one record (raw file line)
   python3 tools/verify_perft.py "FEN" N                      # trusted count for any position
   python3 tools/verify_perft.py "FEN" N --divide             # true per-root-move counts (SF format)
   ```

   The positional `"FEN" N` mode is the movegen-debugging workhorse: at any
   position where you suspect your divide counts, it prints the true total,
   and `--divide` the true per-move split (sorted by UCI move text — sort
   engine output before diffing).
2. **Divide** — `perft::divide` / `chessengine perft <depth> [fen]` / UCI
   `go perft <depth>` print per-root-move subtree counts (design/06 §3.2).
   Descend into the move whose count is off (re-checking against
   `verify_perft.py "FEN" N --divide`); a few iterations pinpoint the broken
   position.
3. **`tools/perft_diff.py`** *(lands with P3, contract: design/06 §3.7)* —
   automates the divide-descend loop by differential testing against a
   reference engine over UCI. Point it at any engine supporting `go perft`
   with SF-style output via `--ref` or `$PERFT_REF` (e.g. a locally built
   Stockfish). Modes: `--suite` (whole validation suite),
   `--fen FEN --depth N` (one position), `--walk N --depth D [--seed S]`
   (random-walk fuzzing). It bisects any disagreement to a minimal
   reproducer FEN + move sequence — paste it into a unit test.
4. **Checked perft** — `perft_checked` re-verifies board consistency, all
   five Zobrist keys against from-scratch recomputation, `gives_check`
   predictions, and unmake round-trips at every node (design/06 §3.3). Slow;
   for bug hunts, never timing.
5. **Regression rule**: every fixed divergence enters `tests/data/perft.epd`
   as a new tricky case with `verify_perft.py`-verified counts (design/06
   §5). The suite only grows.

## Property-based tests

Property tests (round-trips like FEN/`Display` parse-print, encoding
invariants, set-operation laws) use **proptest** as a dev-dependency (D15
allows dev-deps; the runtime dependency count stays zero). They run under
plain `cargo test` — no separate command.

When proptest finds a failing case it shrinks it and persists the seed in
`proptest-regressions/`. **Those files are committed**: they turn a
one-in-a-million counterexample into a deterministic regression test that
replays first on every future run. Never delete a regression entry to make
a test pass — fix the code, or (if the property itself was wrong) fix the
property in the same commit that explains why.

## The `Bench:` commit trailer (D24)

Once search lands (P5), every commit that can affect search behavior ends
with a trailer line, exactly:

```
Bench: 3187513
```

- The number is `Nodes searched` from the pinned fingerprint invocation
  `chessengine bench 16 1 3 default depth` (the SF-shaped default is 16 MB
  TT, 1 thread, depth 13 — design/06 §3.6 — but the P5 skeleton pins the
  depth at 3 until the TT lands; TODO(P6) in tools/check.sh tracks raising
  it). `tools/check.sh` prints the footer for you.
- *Functional* patches (the count changes) state the **new** count.
  *Intended-non-functional* patches state the unchanged count — reviewers
  and tooling verify by rerunning bench and reject the commit if the number
  moved. Commits that cannot affect search (docs, build scripts, UCI I/O)
  may omit the trailer. Full policy, determinism requirements, and the
  frozen 49-position suite: design/14 §3.2.

## Strength testing (SPRT, gauntlets, tactics)

Elo claims are made only by games under SPRT (D25); the full methodology,
honest-concurrency rules, and the gauntlet/rating ladder are design/14 §3.8.
The harness (design/14 §4 steps 10–11) is live; every run that supports a
claim gets one row in `docs/strength-log.md` (appended by the tools).

```sh
tools/sprt.sh CANDIDATE BASELINE        # the merge gate for functional search/eval changes
tools/gauntlet.sh [ENGINE]              # calibration vs reference opponents + ordo rating
tools/tactics.sh [ENGINE] [SUITE]       # tier-1.5 tactical sanity (never a gate)
tools/tactics.sh --fetch-wac            # download WAC + mate-in-2..5 into engines/suites/
```

- **`tools/sprt.sh`** — fastchess SPRT with the design/14 §3.8 parameters:
  bounds `[0, 5]` nElo for gains (`ELO0=-5 ELO1=0` for simplifications),
  `model=normalized`, 10+0.1 default TC, paired games with colors reversed,
  concurrency capped at physical cores − 1. `NODES=50000` switches to
  fixed-node proxy games (immune to machine load; never valid for
  time-management changes). Both binaries must be built with identical
  RUSTFLAGS (D16).
- **`tools/gauntlet.sh`** — the engine vs calibrated Stockfish rungs
  (`UCI_LimitStrength` + `UCI_Elo`, coarse anchors) and/or real reference
  engines under `engines/` via `OPPONENTS="name=path;..."`; ratings with
  error bars from ordo over the produced PGN.
- **`tools/tactics.sh`** (wrapper over `tools/tactics.py`, python-chess) —
  scores `bm`/`am` hits over an EPD suite. Default suite:
  `tests/data/tactics-smoke.epd`, 28 committed positions, every one
  machine-verified (the repo's exhaustively-verified mate suite, replayed
  famous miniatures, composed motifs — all re-checked with Stockfish 16 at
  depth 24 before commit). Published corpora (WAC revised, mate-in-2..5)
  are fetched with `--fetch-wac`, never vendored (D15). Milestone rule
  (design/14 §3.9 item 8): run with `--log` before declaring M2+; an
  unexplained drop against the logged baseline blocks the milestone.

**External tooling** (all under gitignored `engines/`, install notes in the
script headers): fastchess v1.8.0 (`engines/bin/fastchess`, release tarball
from github.com/Disservin/fastchess — cutechess-cli is not packaged for
this distro), ordo 1.2.6 (`engines/bin/ordo`, built from
github.com/michiguel/Ordo), Stockfish 16 (`apt-get install stockfish`,
`/usr/games/stockfish`, UCI_Elo range 1320–3190). Openings: the committed
43-line balanced book `tests/data/openings/openings-8ply.epd` (regenerate
with `tools/gen_openings.py`; every line legality-checked by python-chess);
serious SPRT runs should prefer an UHO book (download pointer in
`tools/sprt.sh`).

## What CI runs (and what stays local)

`.github/workflows/ci.yml`, single job per push/PR: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test` (debug), and
`cargo test --release` (non-ignored set). RED tests are ignored, so CI is
green throughout test-first development.

Deliberately **not** in CI: the ignored perft tiers (until P3; then a
nightly job is the candidate home, design/06 §9.4), bench fingerprint
verification (needs a known machine and the commit trailer), criterion
baselines and any timing assertion (flaky on shared runners — design/14
§5), and SPRT. Those live in `tools/check.sh` and the human checklist of
design/14 §3.9.
