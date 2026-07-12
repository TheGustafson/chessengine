# Success factors: the ecosystem and force multipliers (Integral)

Source: `integral_reference/integral/` — the full 669-commit git history
(`git log`, Oct 2023 → Apr 2026), `README.md`, `Makefile`, `CMakeLists.txt`,
`src/tuner/spsa.h`, `src/engine/search/constants.h`, `src/data_gen/`,
`third-party/` (`fathom`, `fmt`, `incbin`), `.clang-tidy`/`.clang-format`.

**Provenance/licensing note (read before touching this file further):**
Integral has **no license file** at all → all-rights-reserved by default,
strictly stronger than Stockfish's GPLv3 or Reckless's AGPLv3 (D27 applies
here even harder). Nothing below reproduces Integral source. Every code
fact is stated as prose (file:line or commit hash to re-find it), parameter
*names/counts* (not copyrightable expression), or my own paraphrase of a
mechanism. No snippet in this note is copy-pasted from Integral.

Compare against: `stockfish_reference/NOTES.md` (SF pioneered mass
distributed testing — Fishtest, ~2013), `reckless_reference/09-uci-time-management.md`
§"SPSA/tunable-constant scaffold" (Reckless's own testing-infra footprint),
our `design/14-performance-testing.md` §3.8 (D25) and `design/decisions.md`
D24/D25/D28/D29.

---

## 0. The headline answer, up front

Integral is a **~10,500-line** (`src/**/*.cc,*.h`, third-party excluded),
**effectively solo** (659 of 669 commits under one author's various git
identities; the remaining 10 commits from ~3 outside contributors — §5) C++
engine that went from "initial commit" to a top-6, TCEC-competitive engine
carrying a lone college student's name in about 30 months of *elapsed* time
and, by every visible signal, **far less wall-clock compute than that would
require on one machine.** The evidence in this repo's own history says the
gap was closed by three external, free, shared systems that the developer
plugged into rather than built:

1. **Distributed SPRT/SPSA testing** (OpenBench-shaped — confirmed in-repo,
   §1). This is the dominant lever. A single "LTC SPSA (100k games, 12.5k
   iterations)" session (commit `8babfda9`, §3) is 1-3 orders of magnitude
   more game-compute than a hobbyist's desktop can supply in the ~2-3 month
   cadence these sessions actually ran at (quantified in §6). Without a
   volunteer compute pool, the *statistical validation step itself* — not
   the idea, not the code — is the bottleneck, and it is the one Integral's
   own commit log shows was solved externally.
2. **`bullet`** (jw1912's NNUE trainer, credited by name in `README.md`)
   removed the *other* classic bottleneck (GPU-hours to train a
   competitive net) by being fast enough to retrain on one consumer GPU per
   iteration. This is the one leg **we already have** (D28: RTX 4060,
   same trainer).
3. **A shared, public idea/technique pool** — the Chess Programming Wiki,
   and, concretely, *other engine authors reading and reusing each other's
   commit histories* (§5: named cross-engine contributors, "Using
   Alexandria's soft limit calculations", "SF-style aspiration window
   adjustments"). This is necessary but **not sufficient by itself** — see
   §7 for why.

The individual still supplied the taste to pick which of hundreds of
possible experiments to spend a scarce test-slot on, and 30 months of
disciplined engineering (tunable-parameter scaffolding, `Bench:`
node-count trailers, PR-per-idea hygiene, `.clang-tidy`/`.clang-format`).
But the *raw compute* that turned "a plausible idea" into "a proven +N Elo
commit" over and over, 230+ times, was not self-supplied. That is the
single most transferable lesson for us (§8).

---

## 1. OpenBench fingerprints, found directly in the repo

Three independent pieces of hard evidence, all still in the tree today:

- **An explicit comment naming it.** `src/engine/uci/uci.cc:248` — the
  branch that special-cases `argv[1] == "bench"` to run the fixed-position
  bench suite and exit immediately (no UCI handshake) carries the comment
  "OpenBench requires the bench command to be parsed from the command
  line." This is not a guess; the codebase says so.
- **The bench output format matches OpenBench's parser exactly.**
  `src/tests/bench_test.cc:63-82` (`BenchSuite`) runs a fixed, 49-position
  FEN suite (embedded verbatim in the source, `kBenchFens`,
  `bench_test.cc:8-60`) at a pinned depth (`kDefaultBenchDepth = 12`,
  `tests.h:18`) and prints a single line of the shape `"<nodes> nodes <nps>
  nps"`. That is the exact string OpenBench's harness regexes out of stdout
  to fingerprint a build — this is the C++ analogue of our own `bench`
  subcommand and `Bench:` trailer (D24, `design/14` §3.2), and it is a
  convention that *originates* with OpenBench/Fishtest-style tooling, not
  something Integral invented independently.
- **The `Makefile`'s target names are OpenBench's contract, not a personal
  choice.** `Makefile:5` (top) defines exactly six build targets — `native,
  vnni512, avx512, avx2_bmi2, avx2, sse41_popcnt` — matching the
  micro-architecture buckets OpenBench uses to pick a build recipe for
  whatever CPU a volunteer's worker happens to report. `CMakeLists.txt:86-91`
  defines the matching `-march=...` flag sets. A hobbyist building for
  themselves has no reason to maintain six named architecture targets;
  a project whose binaries are compiled *by other people's machines,
  automatically, based on what CPU they report* has every reason to.

**When this was wired up, precisely:** commit `5f45ebc6` ("ob support
stuff", 2024-04-23) changes `main()`'s signature to accept `argc`/`argv`
so the executable can be invoked as `integral bench 12` directly from a
shell (`src/main.cc`, diff in that commit) — i.e., exactly the calling
convention an automated test-cluster needs and a human at a UCI prompt
does not. The `bench`/`test`/perft-suite commands themselves land one
commit earlier the same day-ish window, commit `54b86b59` ("lmr and 3fold
pvs (+450 elo), bench", 2024-04-23) and are hardened over the next few
commits (`a9fa8b0c`, `317cbc37`, `5d9f3361`, `0e6ca441` — the commit
subjects literally read "bench fixes", "bench again", "bench WTFFF",
"ok last bench?", visible groaning about getting the fingerprint
deterministic). **This is 6 months after the initial commit** (2023-10-25)
and squarely inside the intense Feb-Sep 2024 sprint that took the engine
from "plays legal chess" to "top-6" (commit-count-by-month in §2) — i.e.,
OpenBench access was wired in *early*, well before the engine was strong,
not bolted on after success. Plugging into the distributed-testing
ecosystem looks like a precondition for the climb, not a reward for it.

---

## 2. Timeline, from the commit graph itself

```
2023-10-25   initial commit (basic move gen)
2023-11      1 commit, then a ~3-month gap (idle)
2024-02      dev resumes; 19 commits
2024-03      54 commits  \
2024-04      79 commits   |  the sprint: negamax, TT, LMR/PVS (`54b86b59`,
2024-05      96 commits   |  "+450 elo"), bench/OB wiring (`5f45ebc6`),
2024-06      74 commits  /   CPW PSQTs (`1345776a`)
2024-07      71 commits      PR-numbered workflow + Bench: trailer begin
                              formalizing (commit `20e5da17`, PR #7, 07-09)
2024-08 .. 2025-08   steady 10-70 commits/month: the "Wave"-style
                     incremental-with-SPRT-gate era; 7 documented
                     100k-scale LTC SPSA retunes (§4)
2025-09 .. 2026-04   commits taper to 0-8/month (maintenance cadence,
                     matches a "shipped, now polishing" project, not
                     abandonment — last commit 2026-04-02, PR #238)
```
(Raw counts: `git log --format=%ci | cut -c1-7 | sort | uniq -c`.)

Two things this timeline makes concrete:

1. **The strength climb (Feb-Sep 2024, ~380 commits, ~7 months) coincides
   exactly with the window OpenBench access, the PR/Bench-trailer
   discipline, and the SPSA scaffold all came online.** The infrastructure
   wasn't a late add-on; it was contemporaneous with — arguably a
   precondition for — the fast part of the climb.
2. **A ~3-month dead period right after the initial commit** (Nov 2023-Jan
   2024) is itself a data point: even with unlimited personal motivation, a
   solo hobbyist's calendar has gaps a funded team's doesn't. The project
   still reached top-6 despite this — evidence that the *rate-limiting*
   resource wasn't developer-hours so much as *validated-idea throughput*,
   which is exactly what a shared test cluster buys you (you can queue an
   idea and come back to the verdict whenever your gap ends).

---

## 3. The PR + `Bench:` trailer workflow, and how it hardened over three weeks

- Commit `d0fc3734` (Yoshie2000, 2024-06-17, "Correction history indexed by
  pawn hash") already tacks a node count onto the end of the subject line
  ("... Bench: 1159053") — an informal, ad hoc version of the convention.
- Commit `20e5da17` (2024-07-09, "Defended pawns and history pruning (#7)")
  is the first commit with (a) a GitHub PR number in the subject **and**
  (b) `Bench: <n>` as its own trailer line in the commit body. From here on
  the two conventions travel together: **214 of 669 commits** carry a
  `(#NNN)` PR reference (`git log --oneline | grep -cE '#[0-9]+\)'`), and
  **313 of 669** carry a `Bench:` line somewhere in the body
  (`git log --format=%B | grep -c '^Bench:'`).
- This is functionally identical to our own D24 `Bench:`-trailer rule
  (`CLAUDE.md`, `design/14` §3.2) and to Stockfish's own convention — it did
  not spring from nowhere; it is the community-standard way an
  OpenBench-style pipeline lets a maintainer (or a bot) sanity-check that a
  merged PR didn't silently change node counts (a "functional" vs.
  "non-functional" distinction we also draw). Seeing an *independent* solo
  C++ project converge on the identical trailer convention is good outside
  confirmation that this is genuinely load-bearing tooling, not a
  house-style quirk of our own project.
- **The discipline is not perfectly rigid** — some commits (`fd0709c4`,
  `26d6a69e`, `f6fb1547`, `8a2dde8d`, `4b5bca0c`, …) are just `Bench: N` with
  no other message, i.e., bench-only "sanity churn" commits, and several
  merge commits carry a trailing `Bench:` too. Real projects are messier
  than the idealized process description — worth remembering when we hold
  ourselves to D24's letter.

---

## 4. SPSA at the scale the task brief predicted — verified in-repo

`src/tuner/spsa.h` implements a `TUNABLE`/`TUNABLE_STEP` macro pair,
compiled two ways: in a normal build both expand to a plain `constexpr`
(zero runtime cost — identical to our own `pub const` params,
`design/10-pruning-extensions.md` note on `search/params.rs`); behind an
`SPSA_TUNE` build flag, each one instead registers a UCI option (so an
external tuner can rewrite it over stdin at runtime) and — critically —
`spsa.h` also prints one line per tunable in the exact five/six-field
shape `name, type, value, min, max, step, learning_rate` when a second flag
(`PRINT_SPSA_INPUTS`) is set. **That print format is not incidental** — it
is the literal paste-into-the-tuner format that OpenBench's SPSA page (and
the OpenBench-adjacent `chess-tuning-tools`/`weather-factory` scripts many
engines use around it) expects verbatim, field-for-field. An engine only
prints its tunables in that exact shape if it expects to hand them straight
to that pipeline.

Scale: **126 tunables** total (`grep -c TUNABLE` across the 9 files that
declare them: `constants.h` 76, `move_picker.cc` 14, `time_mgmt.cc` 10,
`evaluation.h` 7, `history.h` 5, `bonus.h` 5, `correction_history.h` 4,
`spsa.h` 4, `pawn_history.h` 1) spanning LMR bases/divisors, aspiration
window growth/delta, six-plus futility/reverse-futility margins, history
bonus/malus shapes, time-management soft/hard-limit scalers, and eval
term weights.

**Seven distinct large-batch retuning sessions are named directly in commit
subjects**, each with an explicit game count and iteration count (grep
`-iE SPSA` over every commit message):

| Commit / PR | Date | Scale |
|---|---|---|
| `407d2150` PR #13 | 2024-07-16 | "LTC SPSA Tune" (no count given — earliest) |
| PR #97 | 2024-10-20 | 25,000 iterations, **100,000 games** |
| PR #144 | 2024-12-12 | "LTC SPSA Tune" |
| PR #164 | 2025-01-15 | 9,375 iterations, 75,000 games |
| PR #180 | 2025-03-25 | 9,375 iterations, 75,000 games |
| PR #204 | 2025-06-16 | 12,000 iterations, 98,000 games |
| PR #220 | 2025-08-05 | 12,500 iterations, **100,000 games** |

Cadence: roughly every 1.5-3 months, each session on the order of
**75,000-100,000 full games at LTC** (long time control — the kind of game
that itself takes many times longer per game than STC). This is the exact
"100k-game LTC SPSA" the task brief hypothesized, and it is not a one-off —
it recurred **seven times** over 13 months. §6 works out why that number is
implausible on one desktop at that cadence.

---

## 5. The shared idea/technique pool, and direct people-overlap

**Idea provenance, named in Integral's own commit subjects** (facts, not
code — safe to state plainly):

- `1345776a` (2024-05-18) "CPW Piece Square Tables" — the pre-NNUE
  hand-crafted eval literally started from Chess Programming Wiki table
  values, the same community reference table nearly every hobbyist engine
  (including early SF, and our own bootstrap PSQT) bootstraps from.
- `3418de6b` "Using Alexandria's soft limit calculations (43 elo LTC, 68
  elo STC)" — a time-management formula taken from a *different* named
  open engine (Alexandria) and reported with its own measured Elo gain.
- `1568901f` (PR #234) "Use SF-style aspiration window adjustments" —
  Stockfish's own aspiration-window re-widening shape, adopted directly.
  (Consistent with SF's GPLv3 permitting derivative code — but notably
  Integral's own commit message frames it as "style," i.e., a re-derivation,
  matching our own D27 discipline of taking ideas/formulas, not text.)

**Direct people overlap** — `git log --format="%an <%ae>" | sort | uniq -c`
shows 659 of 669 commits under Aron Petkovski's several git identities, and
exactly **10 commits from what looks like 3 distinct outside people** (4
git identities — `xu-shawn` and `Shawn Xu` use different emails,
`50402888+xu-shawn@users.noreply.github.com` vs. `xu107288696@gmail.com`,
but given identical surname/timing this reads as one person's early
personal-email commit vs. later GitHub-noreply PR commits, not two people):

- `xu-shawn` / `Shawn Xu` — 8 commits total (`82e8bad7`, `ceeeae44`,
  `14f91f33`, `72ded2e4`, `22db2a9d`, `1246f30a`, `0c598482`, `99fcfe0c`),
  spanning 2024-05 through 2025-05: continuation-history tweaks, an
  aspiration window tweak, an `.editorconfig` addition, a MultiArray
  refactor. `xu-shawn` is a recognizable name in the open
  engine-programming community (active around Stockfish/CPW circles).
- `Yoshie2000` — 1 commit (`d0fc3734`, 2024-06-17, correction history
  indexed by pawn hash) — an idea that subsequently shows up, independently
  re-derived, in several *other* engines' pawn-correction-history designs,
  i.e., a two-way flow (an outside dev drops a technique into Integral,
  and the technique itself is understood community-wide).
- `Jonathan Hallström` — 1 commit (`45e0e036`, 2025-04-29, "Finny table
  fused updates") — NNUE accumulator-cache optimization from another
  recognized community contributor.

**Reading this against our two other studies**: `reckless_reference/09-uci-time-management.md`
records that Reckless itself has an *unpopulated* fishtest-style
SPSA-tunable-constant scaffold ("`define!` macro... declared but
unpopulated in this snapshot") — i.e., even the #2 Rust engine's public
snapshot shows it plugged into **Stockfish's own Fishtest** infrastructure
for tuning rather than standing up a separate instance. Stockfish itself is
the historical origin of this whole model: Fishtest (launched ~2013) was
the first mass, volunteer-donated-CPU, continuously-running SPRT harness in
this hobby, and both OpenBench (built later, deliberately lighter-weight
and easier for a small engine to self-host or share) and the SPSA-print
convention in `spsa.h` are direct intellectual descendants of it. The
three engines we've now studied (SF, Reckless, Integral) triangulate on the
**same** external testing lineage — nobody re-invented this wheel.

---

## 6. Quantifying the multiplier — why one desktop doesn't explain the evidence

Our own `design/14-performance-testing.md` §3.8 already derives the
relevant statistics (this is generic SPRT/Elo arithmetic, not anything
borrowed from Integral): approximate 95%-CI error bars of **±44 Elo at 100
games, ±14 at 1,000, ±4.4 at 10,000** (error scales ∝ 1/√N). Solving that
scaling relation for the Elo deltas Integral's own commit log claims:

- To resolve a claimed **"~2 elo"** gain (commit `8af5a61c`) at the same
  confidence needs roughly `N ≈ 100 · (44/2)² ≈ 48,000` games.
- A claimed **"5 elo"** gain (`895078a2` is "2 elo", others like `76969f82`
  "5 elo", `00a4286a` "4 elo") needs `N ≈ 100·(44/5)² ≈ 7,700` games.
- Integral's commit log has **51 commits** with an explicit "N elo" claim
  in the subject line alone (`git log --oneline | grep -ciE elo`), the
  overwhelming majority in the single-to-low-double-digit range — each one,
  if it is to be a real SPRT verdict rather than an eyeballed guess (and
  D25-style discipline, which the `Bench:`/PR convention strongly implies,
  says it should be), individually costs **thousands to tens of thousands
  of games**.
- On top of that: 214 PR-numbered commits, most gated by at least one STC
  SPRT (and larger ones by an LTC confirmation, mirroring our own §3.8
  two-tier practice) — even a conservative 3,000-10,000 games/PR average
  puts total *per-PR* SPRT compute at **0.6-2 million games**, before
  counting the SPSA sessions in §4 separately: the five sessions with a
  disclosed count (PR #97 100k + #164 75k + #180 75k + #204 98k + #220
  100k) alone sum to **≈448,000 games**, and that's undercounting — two
  more named sessions (PR #13, #144) gave no count at all.

**Total plausible lower bound: several million completed games over the
project's active life.** At LTC (each game many times the wall-clock of an
STC game — our own harness treats LTC as the 60+0.6-class confirmation
tier, `design/14` §3.8), even a generous single desktop (say, 16-32
threads dedicated *full time*, nothing else running) doing nothing but
play STC games nonstop would need low tens of thousands of wall-clock
hours to produce that many games — multiple **years** of 24/7 dedication,
not the observed 1.5-3 month cadence *between* SPSA sessions alone (each
of which, at LTC, is itself the more expensive kind of game). A hobbyist
does not have a spare always-on 32-thread server exclusively reserved for
chess for two years, competing with nothing else (compiling, personal use,
electricity cost, thermal/reliability failures). The numbers only close if
the games are farmed out across many contributed machines running in
parallel — i.e., distributed testing. This is inference, not a
document Integral wrote saying "we used N machines" — but the in-repo
OpenBench fingerprints (§1) plus this arithmetic together make the
distributed-testing explanation the only one consistent with both the
evidence and ordinary hobbyist resource constraints.

---

## 7. Comparison table — three engines, one lineage

| | Stockfish | Reckless | Integral | Us (current) |
|---|---|---|---|---|
| Testing infra | Fishtest (self-originated, ~2013, thousands of volunteer nodes, oldest/largest instance) | Uses external Fishtest-style SPSA scaffold, unpopulated in studied snapshot (`reckless_reference/09` §"SPSA/tunable-constant scaffold") | OpenBench-shaped (bench cmd + arch-named Makefile targets + SPSA print format, all in-repo, §1) | Single local machine, `fastchess` SPRT (`tools/sprt.sh`), **no** distributed harness (design/14 §3.8 says so explicitly: "concession to local hardware") |
| SPSA / auto-tuning | Yes, at massive scale (origin of the practice) | Scaffolded, not exercised in the studied snapshot | Yes — 126 tunables, 7 named 75k-100k-game sessions (§4) | Not yet built (planned P14, `design/00-master-plan.md` P14 row; `design/10` notes params are "pending SPSA") |
| NNUE trainer | Own (`nnue-pytorch`, then largely superseded internally) | Not covered in our snapshot | `bullet` (external, credited by name) | `bullet` (planned, D28) — **same trainer, already decided** |
| Opening book | UHO-class books (their own `books` releases; other engines including Integral reuse them) | n/a in this snapshot | UHO_Lichess_4852_v1 (README, external) | UHO_Lichess_4852_v1 already referenced in `tools/sprt.sh` comments — same book |
| Tablebases | Own Syzygy-adjacent probing | n/a in this snapshot | Fathom (vendored, MIT-equivalent, multi-author: de Man/basil00/Dart) | Not yet integrated |
| Net distribution | Shipped in-repo / release assets | n/a | Auto-downloaded from a separate GitHub Releases repo at CMake-configure time (`CMakeLists.txt:23-45`, `NETWORK_NAME`/`NNUE_URL`) — a network *upgrade* is a one-line `NETWORK_NAME` diff (see PR #212, "Switch to Zekrom v7 network") | Open question in our own roadmap (`design/00-master-plan.md` P13 row: "Net binary distribution (release assets vs git LFS); decided") |
| Team size | Large, distributed, decades of accreted contributors | Small team/solo (per our prior study) | **Effectively one person** (659/669 commits), ~3 drive-by external contributors (10 commits total, §5) | One person (+ AI pair-programming) |
| Code size | ~100k+ LOC | Mid-size (per our prior study) | **~10,500 LOC** (excl. third-party) | Comparable order of magnitude, growing |

**Reading across the row:** every engine we've studied — including
Stockfish, the one with the resources to have built its *own* testing
infrastructure from scratch — ends up on the same short list of shared
external systems (a Fishtest-lineage distributed tester, a UHO-class book,
a Syzygy-format tablebase prober, and, for the NNUE-era engines, `bullet`).
The "top-6 by one person" result is not an outlier that skipped the
infrastructure step; it's an engine that plugged into *all four* legs
essentially immediately (§1-§2) rather than building any of them.

---

## 8. Individual vs. ecosystem — the actual split

Framed as a rough capital allocation, what did the ecosystem supply for
free vs. what did the individual have to bring?

**Supplied by the ecosystem (free, external, load-bearing):**
- The compute to *validate* an idea at reasonable statistical confidence
  (§6) — this is the resource that, in the absence of the ecosystem, would
  have hard-capped Integral's Elo progress at whatever a single desktop's
  games/month could confirm (roughly: only the largest, ≥15-20 Elo, changes
  would ever clear the noise floor at affordable N — the entire tail of
  2-8 Elo micro-optimizations in the commit log, which is most of the 51
  Elo-tagged commits, would have been economically untestable alone).
- The NNUE training loop's GPU-hour cost, via `bullet`'s efficiency (this
  leg turns out to be comparatively cheap even solo, §0 point 2 — it is
  *not* the scarce resource the way test-compute is).
- A pre-validated grab-bag of techniques (CPW, other engines' public
  commit histories, direct patches from community members) that shrinks
  the *search space* of ideas worth spending a precious test-slot on.
- Off-the-shelf, permissively-licensed infrastructure code the developer
  didn't have to write or debug at all: `fathom` (Syzygy probing, MIT-style,
  multi-author), `fmt` (MIT) for all formatted output, `incbin` (public
  domain-ish) for embedding the net binary. None of this is chess-specific
  differentiation; all of it is necessary plumbing the community had
  already solved.
- A public rating ladder with zero infrastructure cost to the developer:
  the README's "Play against Integral on Lichess" line implies the
  `lichess-bot` framework (a free, third-party, config-file-only bridge
  from any UCI engine to Lichess's rated bot pool) and, per the task
  brief, a TCEC berth — both external, volunteer-run stages that a solo
  dev gets to walk onto for free, rather than needing to build a rating
  system or tournament infrastructure themselves.

**Supplied by the individual (not free, not fungible):**
- The taste to decide which of many plausible experiments is worth a
  scarce SPRT slot — the ecosystem validates ideas, it doesn't generate
  them for free; someone still has to read Stockfish's/Reckless's/CPW's
  material, form a hypothesis specific to *this* engine's current
  weaknesses, and implement it correctly enough to test at all.
- ~30 months of unglamorous engineering discipline: the tunable-constant
  scaffold (`spsa.h`) had to be *built* before it could be filled by
  external SPSA compute; the `Bench:`/PR hygiene had to be *maintained*
  commit after commit; `.clang-tidy`/`.clang-format` had to be set up and
  kept clean.
- The judgment to know *when* a technique from another engine actually
  transfers (Alexandria's soft-limit formula measured its own Elo gain
  when re-tested in Integral's own move-ordering/pruning context, per the
  commit message — a straight copy without re-validation would not have
  been trusted).

**Verdict:** the compute leg (§1, §4, §6) is the single most decisive,
most clearly *external*, and most clearly *irreplaceable-by-effort-alone*
factor — a solo developer cannot compensate for a missing volunteer test
fleet by simply working more hours, because the bottleneck is wall-clock
game-count, not developer attention. The idea-pool leg is real but only a
multiplier on individual judgment, not a substitute for it (plenty of
engines read the same CPW pages and don't reach top-6). Our best estimate,
stated plainly as a judgment call rather than a measured fact: **on the
order of half or more of "how did one person reach top-6" is "they had
free access to a testing fleet that would otherwise have cost tens of
thousands of dollars and years of wall-clock time to replicate alone"** —
the individual supplied the ideas and the discipline to keep 30 months of
solo engineering coherent, but the ecosystem supplied the one resource
(statistically confident validation at scale) that no amount of individual
skill can substitute for.

---

## What this means for us as a small project

Concrete, ordered by leverage-per-effort. Cross-referenced to our own
`design/`, `CLAUDE.md`, and `tools/`.

1. **Set up an OpenBench-shaped testing pipeline — this is the highest-
   leverage single item in this whole note.** We already emit the
   equivalent artifact (`bench` subcommand + `Bench:` trailer, D24,
   `design/14` §3.2) and already run local SPRT (`tools/sprt.sh`, D25). The
   gap is entirely the *distribution* layer: (a) confirm our `bench`
   argv-subcommand prints in the OpenBench-parseable `"<n> nodes <n> nps"`
   shape (ours currently reports via `docs/perf-baseline.md`/check.sh — verify
   the exact stdout string an OpenBench worker would regex against); (b)
   stand up (self-hosted, e.g. the open-source OpenBench Django app) or
   join an existing community OpenBench instance once we're ready to
   accept public traffic; (c) mirror Integral's Makefile pattern — named,
   micro-architecture-tagged build targets, not just our current
   `RUSTFLAGS=target-cpu=x86-64-v3` single fast-build recipe (D16,
   `design/01` §3.7) — so a volunteer's worker can build the exact binary
   its detected CPU wants. Impact: unlocks every other item below; without
   it we remain bottlenecked at "however many games one local machine can
   run," which §6's arithmetic says caps us out of resolving anything
   below roughly 10-15 Elo per change at affordable N.
2. **Add the SPSA scaffold now, even before we have distributed compute to
   feed it** — cheap and already on our roadmap (`design/00-master-plan.md`
   P14 row, `design/10` note "SPSA-able" params file). Integral's pattern
   (`spsa.h`: a `TUNABLE`/`TUNABLE_STEP` macro pair, zero-cost in normal
   builds, UCI-option-backed + stdout-printed-in-tuner-format under a
   feature flag) is a clean, idiomatic-to-port shape (our `Reckless` study,
   `reckless_reference/09` §"SPSA/tunable-constant scaffold", already
   flagged the analogous Rust pattern as "Adapt #10" — this note adds a
   second independent confirmation that the shape is worth building early,
   *before* the distributed compute exists, so the params file is populated
   and ready the day testing capacity arrives). Do this as a `cargo`
   feature (`spsa`), not a runtime flag, per Reckless's own pattern.
3. **Adopt the "net is a separate small download, not a repo blob" pattern**
   for NNUE once P13 lands. Integral's `CMakeLists.txt:23-45` auto-downloads
   the pinned network from a *separate* GitHub Releases repo
   (`aronpetko/integral-networks`) keyed by a single `NETWORK_NAME` string
   (`CMakeLists.txt:23`), so shipping a new net is a one-line diff + a
   release upload, and the
   main repo never bloats with binary history. This directly answers our
   own open question in `design/00-master-plan.md` P13 row ("Net binary
   distribution (release assets vs git LFS)") — recommend release-assets
   over git-LFS, mirroring this exact mechanism.
4. **Reuse Fathom for Syzygy, don't write a prober.** MIT-equivalent,
   multi-author-maintained (Ronald de Man — the TB format's own creator —
   plus community patches), used by a large fraction of the field.
   `design/00-master-plan.md`/roadmap has no Syzygy line item yet; when TB
   support is scheduled, budget it as "wrap an existing safe-Rust or FFI
   binding to Fathom/a Fathom-equivalent," not "design a probing format,"
   consistent with our own D27-style stance of reusing infra rather than
   reinventing chess-programming commodity plumbing.
5. **We already made the right call on the book and the trainer — keep
   both.** `tools/sprt.sh` already documents `UHO_Lichess_4852_v1.epd` as
   the serious-run book (same one Integral's README names), and D28 already
   pins `bullet` as our trainer. No change needed; this is confirmation,
   not a gap — worth noting in `design/14` or `design/15` as "independently
   corroborated by the Integral study" so a future reader doesn't second-
   guess a decision that two unrelated top engines converged on.
6. **Budget for the statistics, not just the idea, when scoping a change.**
   Use §6's error-bar math explicitly in our own process: before starting
   any change we expect to be worth single-digit Elo, write down the
   game-count that change will need to clear the noise floor (our own
   `design/14` §3.8 table already has the constants; we're only missing the
   habit of applying them prospectively) — and if that number is not
   affordable on our current single-machine harness, that is itself the
   argument for item 1, not a reason to skip testing the idea.
7. **Treat the multi-agent review loop (D33) as our stand-in for "another
   engine dev reads your PR"** until we have real outside contributors.
   Integral's small but real trickle of drive-by contributions from named
   community members (xu-shawn, Yoshie2000, Jonathan Hallström — §5) is a
   benefit we can't manufacture without a public, credible, actively-tested
   engine to attract them — but our existing A/B/C review persona loop
   (`design/14` §10) is explicitly designed to approximate the "a second
   set of eyes from a different vantage point" effect in the meantime; keep
   investing in it rather than treating it as a placeholder.
8. **Don't overweight the idea-sharing leg relative to the compute leg
   when prioritizing our own limited time.** We already read Stockfish and
   Reckless deeply (this note is itself proof the idea-pool leg is easy
   for us to access — three studies deep already). The leg we have
   *nothing* built for is distributed validation compute (item 1). If
   forced to choose where to spend the next unit of infrastructure effort,
   this study's arithmetic (§6) says: testing throughput before technique
   breadth — we are not short on known techniques (our own `design/10`
   backlog already exceeds what we've had test-budget to land), we are
   short on the game-count to confirm them cheaply.
