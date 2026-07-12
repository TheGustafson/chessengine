# Integral — time management, UCI, the in-tree SPSA tuner, build/target dispatch

Source studied: Integral (C++20, no explicit license => all rights reserved,
**never copy code** — distilled to prose/pseudocode only) at the clone under
`integral_reference/integral`, HEAD `3f15ba6b` (2026-04-02), 669 commits back
to the initial commit (2023-10-25). Files read in full: `src/engine/search/
time_mgmt.h` (161 lines) + `.cc` (278 lines); `src/engine/uci/uci.h` (106),
`uci.cc` (262), `option.h` (123), `command.h` (271), `reporter.h` (218);
`src/tuner/spsa.h` (86); `src/engine/search/constants.h` (105, representative
of every file that declares tunables); `Makefile` (101) + `CMakeLists.txt`
(210) + `preprocess/CMakeLists.txt` (44) + `preprocess/net_processing.cc`
(112). Git history mined with `git log`/`git show`/`git branch -r` — the
**946 remote branches** and the SPSA-commit timeline are primary sources for
the process/success-factor half of this note, per the task brief. Every
citation is `<file>:<line>` or a commit hash against that tree; no Integral
source is reproduced anywhere below.

Compared throughout against `stockfish_reference/` (`timeman.cpp` notes),
`reckless_reference/09-uci-time-management.md` (Reckless's TM/UCI, already a
close analog — cited inline rather than re-derived), and our own
`design/12-uci-time-management.md` + `src/timeman.rs` (644 lines, a faithful
SF-formula port, D20) + `design/10-pruning-extensions.md` §9 (the still-open
"`tune` cfg mechanism for SPSA" question, `design/00-master-plan.md` P14 row).

**Top-line framing**: Integral's time manager is a from-scratch, independently
fitted heuristic (borrowed once, early, from another open engine —
"Alexandria's soft limit calculations" — then re-tuned by SPSA six more times
over two years) rather than SF's formula ported verbatim the way ours is. Its
UCI surface is deliberately smaller than SF's or our design's (no `movestogo`,
`searchmoves`, `ponder`, WDL) but its *tuning* infrastructure is the most
mature thing in this whole codebase: a ~15-line macro turns any constant
into a zero-cost `constexpr` in production and a live UCI-settable,
OpenBench-SPSA-ready parameter in a special build, with **122 such constants**
declared across the engine at HEAD. The git history around that macro and the
`bench`/`Bench:` convention is direct, load-bearing evidence for the "how did
a solo dev reach top-10" question the user cares most about — see §6.

---

## 1. Time management — `TimedLimiter` (`time_mgmt.h:87-121`, `.cc:78-184`)

### 1.1 Shape: a strategy-object stack, not one struct

`TimeManagement` (`time_mgmt.h:123-158`) owns three **pre-allocated, reused**
limiter objects — `DepthLimiter`, `NodeLimiter`, `TimedLimiter`
(`time_mgmt.cc:187-193`) — and rebuilds a `Vec`-like `active_limiters_` list of
raw pointers into whichever subset applies to the current `go` (`time_mgmt.h:
157`, `ConfigureLimiters`, `.cc:204-225`): `depth > 0` activates the depth
limiter, `nodes/soft_nodes > 0` activates the node limiter, `move_time > 0 ||
time_left > 0` activates the timed limiter — **all three can be active
simultaneously** (a `go depth 10 nodes 5000000 wtime 60000` runs all three
checks every call and stops on whichever fires first), which is closer to our
own flat-`Limits`-struct design (design/12 §3.6, all axes freely combined)
than to Reckless's mutually-exclusive `Limits` enum
(`reckless_reference/09` §2) — a second independent engine converging on "GUIs
really do combine axes, don't force a choice."

Every limiter implements the same four-method interface
(`ShouldStop`/`TimesUp`/`Start`/`Stop`/`GetSearchDepth`/`Update`,
`time_mgmt.h:29-44`) — classic OO polymorphism (virtual dispatch through a
`vector<TimeLimiter*>`), the one place in this codebase that reads as a
direct C++-idiom carryover rather than something to imitate in Rust; our
own `TimeManager` (a single concrete struct, `src/timeman.rs:219-226`) gets
the same "one code path handles every limit kind" property for free via
plain `Option`/zero-sentinel fields and ordinary `if` chains, with no
vtable and no heap-allocated trait objects — the more idiomatic Rust shape
for a closed, small set of limiter kinds (an enum-of-checks, if we ever
wanted the OO split, would still beat `dyn TimeLimiter`).

### 1.2 The soft-stop formula — `TimedLimiter::ShouldStop` (`.cc:98-144`)

In prose: below depth 6, just compare elapsed time to the hard limit (no
formula needed yet — not enough iterations to have a stability signal). From
depth 6 on, multiply the base `allocated_time_` by three independent
multiplicative factors and stop once elapsed time reaches that product:

```
stability_factor  = stabilityBase − stabilitySlope × bestMoveStabilityCount   (capped at 10 reps)
scoreChangeFactor  = clamp(scoreChangeBase
                            + searchScoreCoeff  × max(0, score[d-3] − score[d])
                            + prevScoreCoeff    × max(0, previousSearchScore − score[d]),
                           scoreChangeMin, scoreChangeMax)
nodeCountFactor    = max(nodeFactorBase, (1 − nodesUnderBestMove/totalNodes) × nodeFactorSlope + nodeFactorIntercept)

stop when elapsed >= allocatedTime × stabilityFactor × scoreChangeFactor × nodeCountFactor
```

Three named ideas, each independently well-known in the engine-programming
community but assembled into Integral's own specific shape:

1. **Best-move stability** (`.cc:107-116`) — a plain counter, capped at 10,
   of how many consecutive iterations kept the same root best move; more
   stability ⇒ smaller `stability_factor` ⇒ stop sooner. Functionally the
   same idea as SF's/our `reduction`/`previous_time_reduction` mechanism
   (design/12 §3.9 factor 2) but computed from a raw repeat-count rather
   than SF's `lerp` over `(root_depth − last_best_move_depth)` — simpler,
   no cross-`go` persistence at all (the counter resets every search,
   `.cc:83-84` in the constructor and implicitly via `previous_best_move_`
   staying `NullMove` at `Start()`).
2. **Score-change factor** (`.cc:118-133`) — widens time only when the score
   is *falling* (`search_score_diff`/`previous_score_diff` are only counted
   when positive, i.e. when depth-over-depth or move-over-move the score
   got worse) — the "first search of the game" branch doubles the intra-
   search diff and zeroes the cross-move diff, since there is no previous
   move to compare against yet (`.cc:122-126`) — a small, explicit special
   case for move 1, the same kind of thing our own `TimePersist::default()`
   (`src/timeman.rs:199-209`) exists to seed cleanly.
3. **Node-count-effort factor** (`.cc:135-140`) — the fraction of *this
   search's* total nodes that were spent on moves **other than** the current
   best move; the more effort went elsewhere, the less confident the engine
   is, so it's allowed more time. This is the same underlying signal as SF's
   `rootMoves[0].effort`/`nodes_effort` (design/12 §3.9 factor 4,
   `high_best_move_effort`) computed the other way around (fraction spent
   *not* on the best move vs. fraction spent *on* it) — same information,
   inverted sign convention.

**Per-move node accounting** (`time_mgmt.h:99, 120`; `.cc:94-96`): a fixed
`std::array<U64, 4096>` indexed by `move.GetData() & 4095` — a from-move
16-bit packed encoding masked into a 4096-slot table, not a `HashMap<Move,
u64>`. Collisions between two different root moves that happen to hash into
the same bucket are accepted as harmless noise (root move counts differ by
enough, in practice, that a rare collision doesn't flip the stop decision) —
a genuinely reusable trick for a small project that doesn't want a heap
allocation or a real hash map on this hot a path: **a fixed power-of-two
array plus a mask of the move's own bit pattern, in lieu of a hash map**,
reset once per search (`Start()`, `.cc:150-153`) not once per iteration, so
it accumulates node counts across the whole iterative-deepening loop the
same way SF's/our root-move `effort` field does (design/12's `RootMove.
effort`, `src/timeman.rs`-adjacent search-side field, not this file).

### 1.3 The bound formulas — `CalculateLimits` (`.cc:163-177`)

```
overhead   = MoveOverhead UCI option
if move_time != 0:  hard_limit = move_time − overhead;  return    // exact, no soft limit at all

total_time    = max(1, time_left + 50×increment − 50×overhead)
allocated_time = min(time_left × 0.4193, total_time × 0.0575)
hard_limit     = max(1, min(time_left × 0.9221 − overhead, allocated_time × 5.928) − 10)
```

Two structural differences from SF's/our formula (design/12 §3.7):

- **No `movestogo` branch at all.** Every clock-based game is treated as
  sudden-death-with-increment; there is no analog of SF's/our repeating-time-
  control formula (`MTG_OPT_BASE`/`MTG_MAX_SCALE_BASE`, `src/timeman.rs:
  332-337`). Consistent with §2's finding that Integral's `go` parser never
  reads a `movestogo` token at all (§2.2) — the omission is total, not just
  in this one function.
- **No ply/game-phase term.** SF's/our formula grows `opt_scale` with
  `(ply + const)^0.469` and grows `max_scale` with `ply/12.352` (`src/
  timeman.rs:327-331`) — a whole extra dimension (how far into the game are
  we) that Integral's `CalculateLimits` doesn't consult at all: the only
  inputs are `time_left`, `increment`, and the `MoveOverhead` option. The
  entire "spend a bit more as the game goes long" story that SF encodes as
  an explicit `ply` term, Integral instead gets indirectly, at the *iteration*
  level, only through the soft-stop's score/stability/node factors above —
  a materially simpler bound computation, at the cost of not distinguishing
  a move-1 60+0.6 search from a move-80 60+0.6 search at the *allocation*
  step the way SF/we do.
- **The `50×increment / 50×overhead` and `0.9221`/`5.928`/`−10` constants**
  read as the residue of "Using Alexandria's soft limit calculations" (commit
  `3418de6b`, 2024-06-05 — **43 Elo LTC / 68 Elo STC**, the single largest
  named TM Elo gain in the whole history) re-fitted six times since by SPSA
  — i.e. the *shape* came from another open engine (openly credited in the
  commit message, matching our own SF/Reckless-study norm, D27/D31), the
  *numbers* are now purely Integral's own. A clean example of "borrow the
  shape, refit the constants" as a legitimate, commonly-practiced way to
  bootstrap a subsystem rather than deriving it from first principles.

### 1.4 A time-management subsystem needs real iteration, even at the top

Grepping the file's own commit log surfaces at least three dedicated
bug-fix commits after the formula above had already shipped and been SPSA-
tuned: `b9eead25` "Fix time loss", `96e2c1ee` "Fix time loss again", and PR
`001f52bb` "Time loss fixes (#152)" (2024-12-19) — plus a full rewrite,
`33465b9a` "Different time management (#98)" (2024-10-21, `Bench: 2466379`),
and the earlier "`3c2015fc` TM Move Stability Scaling (**7 Elo STC**)"
(2024-06-23) that first introduced the stability counter this section
describes. **Even a top-6 engine's time manager needed a full rewrite plus
three separate forfeit-bugfix passes over its first 14 months** — worth
internalizing as a realistic budget line, not a sign anyone did it wrong the
first time.

### 1.5 All ten TM constants are currently *frozen*

Every `TUNABLE(...)` in `time_mgmt.cc:8-17` (`kStabilityBase` through
`kNodeFactorIntercept`) is declared with `disabled = true` at HEAD — meaning
none of the ten TM constants are included in the *current* SPSA rotation (see
§3.3): they were tuned to their present values by an earlier LTC run and then
explicitly excluded going forward, presumably because further TM-specific
tuning needs real wall-clock games (not the fixed-node proxy the rest of the
engine's constants can share) and is comparatively expensive to keep
re-running every cycle. **"Freeze a subsystem's constants once its own
dedicated tuning round has converged, to save the shared iteration budget for
whatever hasn't converged yet"** is a real, visible practice here (§6 has the
general form of this lesson).

---

## 2. UCI protocol surface

### 2.1 A declarative command framework, not a big `match`

`command.h` (271 lines) builds a small reusable DSL: `Argument` pairs a name
with an `ArgumentType` (`kOptional`/`kRequired`) and a `CommandInputProcessor`
— `NoInputProcessor` (flag-only, e.g. `infinite`), `LimitedInputProcessor<N>`
(fixed token count, e.g. `depth <d>`), or `UnlimitedInputProcessor` (rest of
the line, e.g. `moves ...`) (`command.h:20-63`). A `Command` holds an ordered
or unordered bag of `Argument`s and a `CommandHandler` closure; `Listener::
Listen` (`uci.h:27-43`) tokenizes each stdin line, looks the leading token up
in a `commands_` map, and calls `ProcessLine` then `Execute` (`command.h:
131-151`). Every UCI verb — `position`, `go`, `datagen`, `stop`, `ucinewgame`,
`eval`, `print`, `setoption`, `test`, `bench`, `uci`, `isready`, `quit` — is
one `listener.RegisterCommand(...)` call with an inline lambda handler
(`uci.cc:42-229`), not a hand-written parser per verb.

This buys real extensibility as the command set grew over 669 commits —
`test see|perft`, `bench [depth]`, and a whole self-contained `datagen
soft_limit hard_limit games threads min_moves max_moves out [book]` command
(`uci.cc:124-146`, wired straight into the engine's own self-play data
generator, no external driver needed) all reuse the same argument-processor
vocabulary rather than each writing its own token loop — at the cost of ~270
lines of framework a from-scratch `match`-based dispatcher (ours, design/12
§3.3, or Reckless's, `reckless_reference/09` §8) doesn't need. Our own
design's plain `match` (design/12 §3.3 dispatch sketch) is the right call for
our smaller, more UCI-standard command set; Integral's investment paid off
specifically *because* it kept adding bespoke, non-standard commands
(`datagen`, `test`, `permute`) that benefit from a shared declarative
vocabulary.

**A concrete robustness lesson already learned the hard way here**: commit
`30505782` "Made UCI go arguments not enforcing order" (2024-07-10, `Bench:
3315231`) flipped `go` from `CommandType::kOrdered` to `kUnordered` — the
initial version required GUIs to send `go` fields in a fixed sequence, which
broke against a real GUI that didn't. Our own design already specifies `go`
tokens as order-free (design/12 §3.3 notes: "`go` parameter tokens may appear
in any order except `searchmoves`") — a place we're already ahead by design,
not by having to discover the bug in production first.

### 2.2 What's missing, and why it's probably deliberate

Grepping the whole source tree for `movestogo`, `searchmoves`, and `ponder`
(as a command or option) returns **zero matches** anywhere in `src/`. The `go`
command (`uci.cc:67-122`) parses exactly: `perft`, `infinite`, `movetime`,
`depth`, `nodes`, `soft_nodes`, `wtime`/`winc`/`btime`/`binc` — nothing else,
silently reporting "unknown argument" for anything not on that list
(`command.h:238-248`, `ProcessUnorderedArguments`). The option table
(`uci.cc:20-35`) is seven entries: `Hash`, `Threads`, `MultiPV`,
`MoveOverhead`, `Minimal`, `SyzygyPath`, `SyzygyProbeDepth` — no `Ponder`
option either, and (per `preprocess`, §4) no runtime `EvalFile` option at
all — the NNUE net is compiled in, full stop, not swappable without a
rebuild.

None of this reads as unfinished work; it reads as **ruthless scope-matching
to what OpenBench/SPRT testing actually exercises**. `cutechess-cli`/
`fastchess` gauntlets are run almost exclusively at fixed time+increment
(`tc=X+Y`), never classical `movestogo` tournaments; SPRT test pairs never
send `searchmoves`; and pondering is off by default in essentially every
automated test harness (it burns CPU on the "wrong" side's clock, which
distributed OpenBench workers can't account for fairly). A solo dev's UCI
surface only needs to be *complete enough for the loop that actually gates
merges* — every hour spent on `ponder`'s state machine or `searchmoves`'
plumbing is an hour not spent on the next SPRT-worthy search idea. Our own
design deliberately goes further (full `ponder`, `searchmoves`, `nodestime`,
a `Debug Log File`, design/12 §3.5/§3.10) because reaching literal SF-parity
completeness is an explicit project goal (D-numbered decisions, "Stockfish-
class architecture") — a choice this comparison validates as a genuine
trade-off, not a mistake on Integral's side.

### 2.3 Two reporters: UCI-format and a colored terminal mode

`reporter.h` defines an abstract `ReportInfo` with two concrete
implementations selected once per search (`search.cc:76-81`) by whether
stdin looks like a real UCI session: `UCIReportInfo` (`reporter.h:113-148`)
prints the standard `info depth ... pv ...` line; `PrettyReportInfo`
(`reporter.h:150-214`) instead prints a hand-tuned, column-aligned, ANSI-
colored line (HSV-interpolated PV-move coloring by position, red/green/
yellow score coloring, `reporter.h:58-110`) meant for a human staring at a
terminal directly — i.e. running the raw binary (not through a GUI) gets a
genuinely nicer developer experience for free, gated purely on `uci::
reporter::using_uci` (`reporter.h:17`, flipped true the moment the `uci`
command itself is ever received, `uci.cc:217`). Purely a quality-of-life
feature — bench-neutral, zero effect on search behavior — but a real one:
a solo dev spends a lot of personal time staring at exactly this output
while debugging by hand.

The `Minimal` check option (`uci.cc:28`) gates every per-iteration `info`
print, leaving one summary line right before `bestmove`
(`search.cc:146, 153-179`) — **the identical name and identical
semantics** as the `Minimal` option we already adopted from the Reckless
study and landed in P10 (design/12 §3.13, D31 note in
`design/12-uci-time-management.md` §10). Two independent top engines
converging on the same option name and the same behavior is a good signal
this is genuinely the standard shape for "quiet mode," not a coincidence
worth re-deriving.

---

## 3. The SPSA tuner — headline: 100k games / 12.5k iterations at LTC (#220)

### 3.1 The macro: one definition, two build modes, zero per-constant cost

`src/tuner/spsa.h` (86 lines) is the entire tuning framework. A single line
at the top, `// #define SPSA_TUNE` (`spsa.h:8`, commented out), is the whole
on/off switch — a developer manually uncomments it, rebuilds, runs the SPSA
sweep externally, then re-comments it and rebuilds normally. There is no
build-system flag, no CMake option, no environment variable: **the entire
"am I building a tuning binary or a release binary" decision is one
commented-out preprocessor line**, which is about as low-ceremony as this
kind of infrastructure can get.

The macro itself, in our own words (not Integral's code):

```
// production (SPSA_TUNE undefined):
TUNABLE(name, value, min, max, disabled)  =>  a plain compile-time constant
                                               equal to `value`; min/max/
                                               disabled are inert, purely
                                               documentation for the tuning
                                               build. Zero runtime cost.

// tuning build (SPSA_TUNE defined):
TUNABLE(name, value, min, max, disabled)  =>  a namespace-scope object whose
                                               constructor (running before
                                               main(), ordinary C++ static-
                                               init order) does two things
                                               unless `disabled`:
                                                 1. registers a UCI option
                                                    named `name` (an integer
                                                    "spin" option if T=int,
                                                    a "string" option
                                                    encoding the double as
                                                    text if T=double) whose
                                                    setoption callback writes
                                                    straight back into the
                                                    object's own stored value;
                                                 2. prints one CSV-ish line
                                                    to stdout:
                                                    "name, {int|float}, value,
                                                     min, max, step,
                                                     learning_rate"
                                               The object supports an
                                               implicit conversion to T, so
                                               every call site that reads
                                               `kSomeConstant` is unchanged
                                               between the two build modes.
```

`TUNABLE_STEP` is the same thing with an explicit, hand-chosen SPSA step
size; plain `TUNABLE` defaults the step to `(max − min) / 20` (`spsa.h:13`) —
i.e. "20 steps across the declared range" is the engine's own default
heuristic for constants where a hand-picked step wasn't worth the thought,
reserving `TUNABLE_STEP` for cases where 1/20th of the range is a bad SPSA
step (small-integer depth thresholds where that could round to a useless
zero step, or centipawn margins where the natural granularity is domain-
specific). The learning rate is a single global constant,
`kLearningRate = 0.002` (`spsa.h:18`), shared by every parameter in the
engine — no per-parameter learning-rate tuning at all, just the one range/20
step heuristic doing double duty as the de-facto per-parameter scale
normalization.

Because the print happens at **static-initialization time** (before `main`
even runs, since every `TUNABLE(...)` call site is a namespace-scope
declaration — `constants.h`, `move_picker.cc`, `evaluation.h`, four
`history/*.h` files, `time_mgmt.cc`), simply launching the tuning-mode binary
— no special flag, no UCI command — dumps every active tunable's SPSA config
line to stdout immediately. That print format (`name, type, value, min, max,
step, learning_rate`) is exactly the input OpenBench's built-in SPSA tuner
expects to build its sweep configuration from (strongly corroborated by the
adjacent, explicit comment "`// OpenBench requires the bench command to be
parsed from the command line`", `uci.cc:248`, sitting right next to the
`argv[1] == "bench"` handling every OpenBench-integrated engine needs) —
i.e. the entire tuning-mode-binary-dump-to-stdout mechanism exists
*specifically* so an external OpenBench worker fleet can auto-generate its
SPSA job config without any hand-written parameter list living outside the
source tree.

### 3.2 122 tunables, 27 currently frozen

`grep`-counting `^TUNABLE`/`^TUNABLE_STEP` at HEAD: **122 total declarations**
(73 `TUNABLE`, 49 `TUNABLE_STEP`) spread across search constants
(`constants.h`, 76 of them — LMR bases/divisors, aspiration window
delta/growth/score-div, reverse-futility/razoring/NMP/IIR/LMP/futility/SEE/
history-pruning/singular-extension margins, a material-scaling constant),
move-picker ordering weights (`move_picker.cc`), history bonus/malus curves
(`history/bonus.h`, `history/correction_history.h`, `history/pawn_history.h`),
static-eval-history blending (`evaluation.h`), and — now frozen — the ten
time-management constants above. Of the 122, **27 currently carry
`disabled=true`** — mostly integer depth-gate thresholds (`kAspWindowDepth`,
`kRevFutDepth`, `kRazoringDepth`, `kNmpRedBase`/`kNmpRedDiv`, `kIirDepth`,
`kLmpBase`, `kFutPruneDepth`, `kHistPruneDepth`, `kLmrHistDiv`/
`kLmrCaptHistDiv`, `kSeDepth`, `kProbcutDepth`, `constants.h:13,28,37,42-43,
46,53,55,64,70-71,78,97`) plus all ten TM constants — i.e. **the SPSA
parameter space is actively curated, not "tune literally everything all the
time."** Commits like `0b32de80` "Make `kLmpBase` a tunable int" (2025-05-29,
widening its declared range from a narrow `1.5..5.0` float to a coarser
`1..10` int just three weeks before the next big LTC run, `#204`) and
`260cc289` "Cleanup + tunables" (2025-07-30, one week before `#220`) show a
consistent small-prep-commit-before-big-SPSA-run rhythm: adjust what's
in/out of scope and its declared range, *then* spend the big compute budget.

### 3.3 The timeline — six LTC retunes over roughly 19 months

| Date | Commit/PR | Scale | Notes |
|---|---|---|---|
| 2024-06-01 | `0e2475f3` | — | Tuner added (originally texel-style HCE weight fitting, not yet SPSA) |
| 2024-06-05 | `3418de6b` | — | "Using Alexandria's soft limit calculations" — **43 Elo LTC / 68 Elo STC**, TM formula borrowed from another open engine |
| 2024-06-19 | `007f50d7` | — | SPSA macro reworked to auto-convert floats (scaled-int trick → direct string-encoded doubles) |
| 2024-06-23 | `3c2015fc` | — | TM move-stability scaling added — **7 Elo STC** |
| 2024-07-11 | `03ad1a4a`/`a7982c83` | — | "Make search parameters tunable" — the current macro shape lands across `constants.h`, `evaluation.h`, `history/*` |
| 2024-07-16 | `407d2150` (#13) | first LTC SPSA | first full-engine batched retune under the new macro |
| 2024-10-20 | `5348ae81` (#97) | **100k games, 25k iterations** | |
| 2024-12-12 | `2ce10ccd` (#144) | LTC SPSA | |
| 2025-01-15 | `170f8c3f` (#164) | **75k games, 9,375 iterations** | paired with a fractional-LMR search change in the same PR |
| 2025-03-25 | `84a3e833` (#180) | **75k games, 9,375 iterations** | |
| 2025-06-16 | `276b58be` (#204) | **98k games, 12k iterations** | |
| 2025-08-05 | `8babfda9` (#220) | **100k games, 12.5k iterations** | the task's headline; touches ~114 lines across `constants.h` + 6 more files in one commit |

Each of these full retunes is a **single commit** touching every currently-
active tunable across search *and* eval at once (`8babfda9`'s diffstat:
`evaluation.{cc,h}`, `constants.h` (114 lines), `history/bonus.h`,
`correction_history.h`, `history/pawn_history.h`, `move_picker.cc` — eight
files, one PR, `Bench: 1836807`) — i.e. **all-parameters-at-once SPSA**, not
one-heuristic-at-a-time, run as a single long job and landed as one bulk
value-replacement commit. This is a structurally different rhythm from our
own P9 Wave A/B (one heuristic, one SPRT, one commit — design/10 §9.4/design/
14 §9.4) and much closer to what our own P9 Waves C–G batching (D32) already
anticipates for the *implementation* side; SPSA retuning is the natural
*next* step after a wave of new heuristics lands, exactly the shape
`design/10-pruning-extensions.md` §9 item 1 already names as the open
question ("a `tune` cfg feature that swaps consts for statics + UCI options,
deferred until we actually tune" — `design/00-master-plan.md` P14 row: "`tune`
cfg mechanism for SPSA (consts → statics + UCI options)").

### 3.4 The `Bench:` trailer — direct, independent validation of our own D24 convention

313 of Integral's 669 commits (47%) carry a `Bench: <nodes>` trailer in the
commit body, starting with PR `#7` ("Defended pawns and history pruning,"
2024-07-09) — i.e. **the exact same convention as our own D24 `Bench:`
trailer** (CLAUDE.md, `design/14` §3.2): a fixed `bench` invocation's node
count, pasted into every commit that can affect search behavior, so a
reviewer (or OpenBench's own commit-message parser) can catch an
accidentally-nondeterministic or untested change before it ever reaches an
SPRT queue. This began the moment the project moved to GitHub PRs (`#7` is
the seventh numbered PR) — i.e. **the trailer convention and the PR-based
workflow arrived together**, strongly suggesting the trailer exists
specifically because OpenBench (or a bot wrapping it) enforces "the PR body
must state the bench count, and it must match what `./engine bench` actually
prints on this commit" as a precondition for queuing a test. Finding this
independently, in a completely separate codebase's git history, is a strong
validation that our own D24/`tools/check.sh` discipline (already
independently designed before this study) is not a bespoke invention — it's
the load-bearing convention every OpenBench-tested engine needs, and we
already have it.

---

## 4. Build system and target dispatch

### 4.1 The Makefile/CMake matrix: OpenBench's arch-suffix contract

`Makefile:80-100` exposes exactly six named targets — `native`, `vnni512`,
`avx512`, `avx2_bmi2`, `avx2`, `sse41_popcnt` (plus `debug`) — each just
setting a `BUILD_TYPE` variable and re-invoking the generic `all` target,
which configures CMake with `-D$(BUILD_TYPE)=ON` (`Makefile:33-46`) and then
copies the resulting binary to a target-suffixed name (`integral_x86_64_
popcnt`, etc., `Makefile:57-74`). `CMakeLists.txt:86-158` maps each of those
options to a concrete `-march=`/`-mtune=`/feature-define combination —
`znver4`/VNNI, `x86-64-v4`/skylake-avx512, `haswell`+`-mbmi2`, a deliberately
*non*-BMI2 `bdver4`/`znver2` tune for plain AVX2, and `nehalem`/`sandybridge`
for the oldest SSE4.1+POPCNT baseline. This is boilerplate any OpenBench-
tested C++ engine needs: OpenBench's distributed worker fleet is
heterogeneous (workers volunteer whatever CPU they own), and its config
system expects engines to ship a small, standard set of pre-built
architecture variants named by a known convention so it can pick the fastest
one each worker actually supports — the target names here (`vnni512`,
`avx512`, `avx2_bmi2`, `avx2`, `sse41_popcnt`) are exactly that standard set,
not a bespoke design choice.

**A genuinely interesting piece of hard-won hardware knowledge**, in the
`BUILD_NATIVE` auto-detection branch (`CMakeLists.txt:110-158`): after
detecting `-march=native`'s `__BMI2__` define, the build does **not**
automatically enable `-DUSE_PEXT` (their PEXT-based magic-bitboard/NNUE-
permutation fast path) just because the CPU claims BMI2 support — it further
checks for Zen 1/Zen 2 (`__znver1`/`__znver2`) and disables PEXT there,
because `pdep`/`pext` are microcoded and *slow* on those specific AMD cores
(a widely-known chess-engine-community fact, independently rediscovered here
in build-config form), **except** a special-cased AMD EPYC 7B (detected via
parsing `lscpu` output at configure time, `CMakeLists.txt:140-152`), which
apparently has a fast-enough PEXT despite being Zen 2. Zen 3/4 get PEXT
unconditionally (fast there). This level of "know your target hardware's
actual instruction latency, not just its claimed feature bits" is exactly
the kind of nuance worth remembering if/when we ever consider a PEXT-based
magic-bitboard alternative to our current fancy-magic `LazyLock` tables
(`src/attacks.rs`) — we currently sidestep the whole question by not using
PEXT at all, which this evidence suggests is a reasonable default rather
than a missed optimization, given how CPU-model-dependent PEXT's payoff is.

### 4.2 NNUE net embedding: a target-matched preprocessing step, then `INCBIN`

Relevant to build/target-dispatch even though the network architecture
itself is a P13/design-17 topic: `CMakeLists.txt:30-48` auto-downloads the
named release net (`zekrom-v7.nnue`) from a GitHub Releases URL if no
`EVALFILE` override is given; a small separate `preprocess` executable
(`preprocess/net_processing.cc`, built via its own nested `CMakeLists.txt`
that mirrors the *same* six `BUILD_*` architecture flags, `preprocess/
CMakeLists.txt:5-12`) then transposes the L1/L2 weight matrices and, under
SIMD builds, permutes 128-bit blocks of the int16 feature weights/biases into
the exact register order its `DPBUSD`-style int8 accumulation kernel expects
(`net_processing.cc:14-37, 53-69`) — i.e. **the raw trained net is packed
differently depending on which architecture target is being built**, as a
CMake custom-command build step (`CMakeLists.txt:70-83`) that runs *before*
the main compile (`add_dependencies(integral run_preprocess)`,
`CMakeLists.txt:211`) and whose output is then embedded directly into the
binary via `INCBIN(EVAL, EVALFILE)` (`nnue.cc:21`) — the same public-domain
compile-time-embed trick Stockfish and most other NNUE engines use, so
there's no runtime file I/O for the default net at all (and, per §2.2, no
UCI option to swap it without recompiling either).

### 4.3 What we already match, or exceed

- **LTO**: Integral enables IPO/LTO conditionally on `check_ipo_supported()`
  for Release only (`CMakeLists.txt:170-177`). Our `Cargo.toml` already pins
  `lto = "fat"` + `codegen-units = 1` unconditionally in `[profile.release]`
  — equivalent or stronger, no action needed.
- **Fast-build target-cpu flags**: Integral's answer to "build for this
  machine specifically" is `-march=native` (`CXXFLAGS_NATIVE`); ours is the
  documented `RUSTFLAGS="-C target-cpu=x86-64-v3"` fast-build recipe (D16,
  CLAUDE.md) — same idea, and we additionally keep the *default* build
  portable on purpose (Integral's default `make` invocation is *also*
  `native`, i.e. its default binary is **not** portable across machines the
  way ours is — a real difference worth naming: Integral optimizes for "fast
  on the box that built it," we optimize for "safe to hand the default binary
  to someone else," with fast builds as an opt-in extra).
- **Multi-target dispatch vs. Rust idiom**: Integral's whole compile-time
  matrix exists because C++ picks one fixed instruction-set target for an
  entire translation unit at compile time — there is no ergonomic way to ship
  one binary that runs the AVX-512 path on capable CPUs and falls back
  elsewhere without hand-written dispatch tables. Rust's `is_x86_feature_
  detected!` macro (paired with `#[target_feature(enable = "...")]` functions)
  makes *runtime* multiversioning inside a single binary comparatively
  ergonomic — which is why most Rust engines on OpenBench (per the wider
  community, not this codebase) ship one binary with internal runtime
  dispatch rather than Integral's six separately-built artifacts. Not a gap
  to close; a place our language already gives us a cleaner answer to the
  same underlying problem, worth remembering explicitly rather than porting
  Integral's Makefile shape wholesale if/when P13's NNUE inference needs
  SIMD-tier fallbacks.
- **No committed CI/merge-gate script at all.** `find` across the whole
  Integral tree turns up zero `.py`/`.sh` scripts and no `.github/workflows`
  in this clone's history (`git log --all --diff-filter=A -- '.github/*'`
  returns nothing) — all local-side rigor is manual (`test see|perft`,
  `bench` run by hand) plus whatever OpenBench itself enforces externally.
  Our own `tools/check.sh` (fmt/clippy/tests/perft ritual/bench fingerprint,
  one command, D24) is **meaningfully more automated locally** than anything
  visible in Integral's tree — see §6 for why that's still consistent with
  Integral's success rather than evidence check.sh is overkill.

---

## 5. Comparison summary

| Subsystem | Stockfish | Our design (doc 12) | Reckless | Integral | Verdict |
|---|---|---|---|---|---|
| TM formula origin | first-party, ply/log10-fitted | ported verbatim (D20) | own fit, no ply term | borrowed once ("Alexandria"), refit 6× by SPSA | three different "who owns the formula" answers |
| `movestogo` | yes | yes | yes (`Cyclic` variant) | **absent** | Integral is the outlier — sudden-death-only |
| `searchmoves`/`ponder` | yes/yes | designed (P10)/designed (P10) | absent/absent | absent/absent | two engines skip both; matches "test-harness-driven scope" |
| Per-move node accounting | `RootMove.effort`, exact per-move | same (planned) | none persisted | fixed 4096-slot masked array (approximate, collision-tolerant) | a genuinely cheap, reusable structure for us |
| Command dispatch | big `switch`/string match | plain `match` (design/12 §3.3) | plain `match` | declarative Argument/Command DSL (`command.h`) | DSL pays off only once command count grows past ~10 bespoke verbs |
| Tunable-constant scaffold | external (fishtest) | none yet (open question, design/10 §9) | `define!` macro, **unpopulated** in the studied snapshot | **122 live tunables**, in continuous 6-round SPSA rotation since 2024-07 | Integral is the only one of the four actually running this at scale today |
| `Bench:` trailer | — | D24, `tools/check.sh` | not confirmed | 313/669 commits (47%), since the first PR | independent convergence — validates our own convention |
| Local CI/merge-gate automation | extensive | `tools/check.sh` (fmt/clippy/tests/perft/bench, one command) | not surveyed | **none found** — manual `test`/`bench` + external OpenBench only | we invest more locally; Integral invests entirely in the external gate |
| Build target matrix | ARCH= make targets | single portable + opt-in fast build | not surveyed | 6 named CMake/Makefile targets, OpenBench-shaped | boilerplate for C++/OpenBench; Rust's runtime dispatch is a cleaner alternative |
| NNUE net delivery | `EvalFile` UCI option, swappable | designed (P13, D21) | n/a (no NNUE yet) | compile-time `INCBIN`, auto-downloaded by CMake, **not** swappable at runtime | a real completeness gap on Integral's side worth not copying |

---

## 6. What this means for us as a small project

This is the part the user cares most about. Ranked roughly by leverage.

1. **Set up OpenBench (or confirm access to a shared instance) before P9's
   pruning waves are fully spent, not after.** Every piece of tuning
   infrastructure this note describes (the `spsa.h` macro, the `Bench:`
   trailer, the `bench` argv subcommand, the six-target build matrix) exists
   *because* Integral is wired into OpenBench, and the wiring is
   deliberately cheap: a `bench` subcommand that prints "Nodes searched"
   deterministically (`uci.cc:248-253`), and a commit-message convention a
   bot can grep. **We already have both halves** — `cargo run -- bench`
   (CLAUDE.md) and the D24 `Bench:` trailer — so the remaining work is
   purely "point an OpenBench instance at our repo," not "build the
   plumbing first." This is the single highest-leverage thing this note
   surfaces: the infrastructure cost is already sunk on our side.

2. **Build the `tune` cfg mechanism now, even before we have SPSA compute
   lined up** (`design/10-pruning-extensions.md` §9 item 1, `design/00-
   master-plan.md` P14 row). Integral's macro is genuinely minimal — one
   `#ifdef`, one templated wrapper struct, ~86 lines total — and the payoff
   (122 tunables, zero per-constant boilerplate, zero production cost) is
   large enough that building it speculatively, ahead of actually having
   games to spend on tuning, costs little and removes a blocker the moment
   compute *does* show up. A safe-Rust equivalent needs an `AtomicI64`/
   bit-cast cell (or a `once_cell`-free interior-mutability wrapper) behind
   the same "const in release, live cell + UCI option in a `tune` feature"
   split — bounded, well-scoped work, not a research project.

3. **Run SPSA in large, infrequent, all-parameters-at-once batches, not
   continuously.** Integral's six-retune-in-19-months cadence (§3.3), each
   one a single 75-100k-game LTC job touching 60-120 constants at once,
   is a deliberate economy-of-scale choice: SPSA's per-iteration cost is
   dominated by fixed overhead (game setup, engine restart) that doesn't
   scale with parameter count, so tuning 100 constants at once is barely
   more expensive than tuning 10, and running the sweep only after a batch
   of *new* heuristics (or a search-shape rewrite) accumulates avoids paying
   that fixed overhead per-heuristic. This maps directly onto our own D32
   batching decision (Waves C-G implemented together, one bug-tripwire SPRT,
   *then* tune later) — Integral's history is direct evidence that "batch
   the implementation, batch the tune, less often but bigger" scales to a
   top-10 result, not just a shortcut we're taking under time pressure.

4. **Curate the tunable set explicitly — freeze what's converged, gate what's
   new.** 27 of 122 constants are currently excluded from SPSA (§3.2), and
   there's a visible small-prep-commit ritual (widen a range, flip a
   `disabled` flag) in the run-up to each big retune. When we build our own
   `tune` scaffold, plan for the same discipline from day one: a params.rs
   convention that tracks "currently in the SPSA rotation" as a first-class,
   reviewable property (not just min/max), so the rotation doesn't silently
   balloon to "everything, forever" and dilute the iteration budget.

5. **Match UCI completeness ambition to what actually gets tested, and say so
   explicitly when we choose otherwise.** Integral's missing `movestogo`/
   `searchmoves`/`ponder`/WDL/runtime-`EvalFile` isn't sloppiness — it's a
   correct read that none of those get exercised by the SPRT loop that
   actually decides whether an idea ships. We've made the opposite, equally
   legitimate choice (full SF-parity completeness is a stated project goal),
   but it's worth periodically asking, for any *not-yet-built* UCI feature,
   "does anything in our own SPRT/gauntlet/bench loop actually exercise
   this?" before spending a phase's budget on it — the honest answer for
   `Ponder`'s full state machine, for instance, is "no automated test we run
   today would notice if it were subtly wrong," which is a real prioritization
   signal, not an argument to drop it (design/12 already treats it as
   correctly deferred to P10, not cut).

6. **Openly borrowing a formula's *shape* from another open engine, refitting
   its constants, and crediting it in the commit message ("Using Alexandria's
   soft limit calculations") is normal, successful, top-10 practice** — not
   a shortcut smaller projects take to catch up. This directly validates the
   whole premise of our own SF/Reckless/Integral study series (D27/D31/this
   task): reading other engines' *ideas*, re-deriving them in our own words
   and our own fitted numbers, and citing the source, is exactly how a top
   engine's own author operates, not a workaround for not being Stockfish.

7. **A solo/small-team project does not need a heavy local CI apparatus to
   reach the top — but it does need iron discipline about *never* merging a
   behavior change without going through the external test gate.** Integral
   has no committed CI config and no merge-gate script in this tree, yet
   pushed roughly one experiment branch a day for two and a half years
   (946 remote branches, §7 below), each — per the visible `Bench:`-trailer
   convention — expected to carry a reproducible bench signature before it
   could be queued for an SPRT. The lesson isn't "stop investing in
   `tools/check.sh`" (our heavier local automation is a legitimate,
   complementary choice, and catches classes of bug — clippy lints, perft
   regressions, fmt drift — OpenBench SPRT never will); it's that **the
   external test gate, used without exception, is the part that cannot be
   skipped**, and everything else (local scripts, review loops, doc rigor)
   is in service of making sure only good candidates ever reach that gate,
   not a substitute for it.

8. **946 remote branches over ~2.5 years is roughly one throwaway experiment
   branch per day** (`git branch -r` on the clone; names range from
   descriptive — `tt_clusters`, `threat_corrhist`, `static_eval_bonus_fix` —
   to visibly informal — `aspergers`, `asprin`, `testbruh`, `this_shit_
   again`), each apparently pushed to GitHub specifically so an OpenBench
   instance can test it against `main`, kept around whether it passed or
   not. The volume itself is the success factor: a solo dev's actual
   throughput on a "top-10 engine" trajectory looks like *hundreds* of small,
   cheaply-abandoned attempts, not a handful of big deliberate pushes — the
   low ceremony of "branch, push, let the bot test it, merge or walk away, no
   cleanup required" is what makes that volume sustainable for one person.
   Adopting a similarly disposable branch-per-idea habit (rather than
   batching several ideas into one longer-lived feature branch) is a cheap,
   directly copyable process change, independent of anything to do with
   OpenBench specifically.

9. **Version deliberately and visibly** (`git tag`: `v1` 2024-05-18 through
   `v7` 2025-02-14, roughly every 1-3 months while the engine was young,
   slowing as it matured; `-dev` suffix on the in-progress name between
   tags, e.g. `"Integral v6.1.3-dev"`, `uci.h:17` at HEAD reads
   `"Integral v7.0.0-dev"`) and **ship a public-facing artifact** (a Lichess
   bot, `IntegralBot`, linked directly from the README). Neither materially
   affects engine strength, but both create the feedback loop (real games
   against real opponents outside the SPRT sandbox, a public README that
   invites outside compilation/contribution) that likely helped sustain
   almost three years of solo motivation — worth treating "visible external
   footprint" as part of the success story, not just an engineering
   afterthought.
