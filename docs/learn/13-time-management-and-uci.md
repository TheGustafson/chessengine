# 13 — Time management and UCI: talking to the world and spending the clock

So far the engine has been a closed world: a board, a search, an
evaluation. But an engine is useless unless something can *drive* it — a
graphical interface, a tournament manager, a testing harness — and unless
it can manage the one resource it never has enough of: **time on the
clock.** This file covers both, because they are two halves of the same
surface. The protocol (**UCI**) is how the outside world tells the engine
"here is a position, think for this long, what is your move?" Time
management is how the engine turns "this long" into a concrete decision
about when to stop searching. And at the end, both connect to a theme that
runs through file 14: **determinism** — the discipline that makes the
engine's behavior reproducible enough to test.

The owning design document is `design/12-uci-time-management.md`
(`src/uci.rs`, `src/timeman.rs`).

## 1. UCI: the protocol as a state machine

**UCI** stands for *Universal Chess Interface*. It is a deliberately
simple, text-based, line-oriented protocol: the GUI (or test harness)
writes commands to the engine's standard input, and the engine writes
replies to its standard output. No binary framing, no network sockets —
just lines of text. You can drive an engine by hand from a terminal, and
people do, because it makes debugging trivial.

The essential commands form a small vocabulary:

```
   GUI → engine                          engine → GUI
   ─────────────────                     ─────────────────
   uci                                   id name ARCHON ...
                                         option name Hash type spin ...
                                         uciok
   isready                               readyok
   ucinewgame                            (nothing)
   position startpos moves e2e4 e7e5     (nothing)
   go wtime 60000 btime 60000 ...        info depth 1 ... 
                                         info depth 2 ...
                                         bestmove g1f3 ponder b8c6
   stop                                  bestmove ...
   quit                                  (process exits)
```

Read it as a **state machine**. The engine idles, waiting for commands.
`uci` triggers a handshake: the engine announces its name, lists every
**option** it supports (each as an `option` line the GUI parses to build a
settings dialog), and finishes with `uciok`. `isready`/`readyok` is a
liveness ping the GUI uses to synchronize — it means "have you finished
processing what I sent?" `position` sets up the board (either from the
standard start position or a FEN string, optionally followed by a list of
moves to apply). Then `go` starts a search with a set of limits, the engine
streams `info` lines as it thinks, and eventually emits exactly one
`bestmove`. `stop` cuts a search short; `quit` ends the process.

### 1.1 The three roles and the exactly-once contract

ARCHON runs this as three cooperating roles (`src/uci.rs`):

- The **UCI thread** is the main OS thread. It does nothing but read
  stdin line by line and dispatch. It *never* searches and *never* prints
  `bestmove`. This is deliberate — because the reader thread never blocks
  on computation, a `stop` command is always read promptly even while a
  search is grinding away on other threads.
- The **main search worker** runs the search, manages the clock, prints all
  the `info` lines, and prints `bestmove` — exactly once per `go`.
- **Helper workers** (file on threading) search in parallel and do no I/O.

The phrase "exactly once" is the load-bearing part of the whole protocol,
and it is harder than it sounds. GUIs depend absolutely on receiving
**one** `bestmove` per `go`: zero means the engine hung, two means the
protocol desynchronized and the GUI may play a stale move. ARCHON
guarantees it in every case — even when the position has no legal moves
(it prints `bestmove (none)`), even when `stop` arrives a microsecond after
the search started, even under the trickier modes below. The whole session
lifecycle in `src/uci.rs` is built around never violating this contract.

### 1.2 Parsing that goes through move generation

One small design choice deserves a mention because it recurs as a theme:
ARCHON parses move text by **generating** the legal moves and finding the
one that formats to the same string, rather than by parsing the text
directly. A token like `e7e8q` is a legal move if and only if it is the
UCI spelling of one of the position's legal moves. This makes the move
*formatter* the single source of truth and gets all the fiddly cases —
promotion suffixes, castling notation, en passant — correct for free.
Slow, but it only runs when setting up a position, never in search.

### 1.3 Infinite, ponder, and the busy-wait

Two `go` modes complicate the exactly-once contract. `go infinite` means
"analyze until I say stop" — used for analysis. `go ponder` means "think on
my time about the move you *expect* me to face next" — the engine searches
during the *opponent's* clock, a free lunch if it guesses the opponent's
move. In both modes the engine may reach the end of its search but **must
not** print `bestmove` yet, because the GUI still owns the clock. ARCHON
handles this with a short busy-wait loop: after the search finishes, spin
until `stop` (or `ponderhit`) arrives, *then* emit the move. Only after
that signal does the one guaranteed `bestmove` flow out.

## 2. The time-allocation problem

Now the harder half. A `go` command in a real game carries the clock
state: `wtime`/`btime` (milliseconds left for White/Black), `winc`/`binc`
(the increment added after each move), and sometimes `movestogo` (moves
until the next time control). From this, the engine must answer a
deceptively simple question: **how long should I think about *this* move?**

### 2.1 The core budget

The naive answer — "divide my remaining time by the number of moves left"
— is close but crude. If you have 60 seconds and expect 30 more moves, a
first cut is 2 seconds per move. Two refinements matter immediately:

- **The increment is nearly free time.** If you get `+1s` after every move,
  you can spend close to that increment every move without your main clock
  ever dropping — so the increment folds into the per-move budget.
- **The horizon is uncertain.** You do not actually know how many moves
  remain. ARCHON uses a fixed horizon of ~50 moves when `movestogo` is not
  given (sudden-death or increment time controls), and shrinks that horizon
  as the clock gets low so it does not gamble a big fraction of a nearly
  empty clock on one move.

ARCHON's actual formula (in `src/timeman.rs`, transcribed from Stockfish's
fitted constants) is more elaborate — a logarithmic curve in the remaining
time and the game ply, capped so it never budgets more than ~19% of the
current clock on a single move — but the *shape* is the intuition above:
increment-aware, horizon-bounded, and self-limiting when time is short.

### 2.2 Soft bound and hard bound

The crucial idea is that the engine computes **two** time limits, not one:

- **`optimum`** — the *soft* target. "Under normal circumstances, aim to
  finish around here."
- **`maximum`** — the *hard* wall. "No matter what, never exceed this."

Why two? Because the engine cannot stop a search at an arbitrary instant
and trust the answer. It searches by **iterative deepening** (file 06):
depth 1, then depth 2, then depth 3, each iteration producing a complete,
trustworthy result. A half-finished iteration is *not* trustworthy — it may
have looked at only the first few root moves. So the natural place to stop
is **between iterations**, when the last completed depth gives a clean
answer.

This is exactly what the two bounds encode. Between iterations, the engine
checks the **soft** bound: "have I spent my target time? If so, don't start
another iteration — I'd only abort it mid-way." The **hard** bound is
checked *during* the search, at every few hundred nodes, as an emergency
brake: if a single iteration runs catastrophically long (a deep forcing
sequence), stop *now*, mid-iteration, because blowing past the hard wall
risks losing on time — the worst possible outcome. The hard stop is the
only thing that can interrupt a running iteration; the soft stop only ever
declines to start a new one.

A subtle refinement lives in the *hard*-bound checker (`check_time` in
`src/search/mod.rs`): it never fires before the **first** iteration
completes. Under an absurdly small budget the engine still returns a real —
if shallow — move rather than nothing, preserving the exactly-once
contract. And it shaves a few milliseconds off the wall-clock bounds to
cover the time it takes to unwind the search and actually print the move.

## 3. Stability and instability: thinking longer when it matters

A fixed soft bound would be leaving Elo on the table, because not all moves
deserve equal time. Some positions are obvious (recapture the queen);
others are critical and unsettled (the whole game hinges on the next
choice). ARCHON *modulates* the soft target using signals from the search
itself. It multiplies the `optimum` budget by four factors, each a small
scalar near `1.0`:

1. **Falling eval → think longer.** If the score has been *dropping* over
   the last few iterations, something is going wrong and more depth might
   find a save. The `fallingEval` factor scales time up when the current
   score is below recent ones.

2. **Best-move instability → think longer.** Every iteration, the engine
   notes whether its top choice *changed* from the previous iteration. If
   the best move keeps flipping — depth 8 says `Nf3`, depth 9 says `d4`,
   depth 10 says `Nf3` again — the position is unsettled and deserves more
   time to converge. The `bestMoveInstability` factor scales up with how
   often the best move has changed.

3. **Best-move stability → think less.** The mirror image: if the same move
   has been best for many iterations running, the engine is confident, and
   it *gives back* some of its budget — banking time for a harder move
   later. This is the `reduction` factor.

4. **High best-move effort → stop sooner.** If the search has already poured
   the overwhelming majority of its nodes into the current best move (its
   rivals were dismissed cheaply), the refutation is not coming; think less.

The intuition unifying all four: **spend time where the outcome is in
doubt.** A stable, confident position is done quickly; a falling,
flip-flopping one gets every millisecond the hard bound allows. These are
the time-management analogue of file 10's "spend nodes where they matter" —
same philosophy, applied to the clock instead of the tree.

## 4. Fixed-node and fixed-depth modes, and why testing needs them

`go` supports limits other than the clock, and they are essential — not for
playing, but for *testing and reproducibility*:

- **`go depth d`** — search to exactly depth `d` and stop, ignoring the
  clock entirely.
- **`go nodes n`** — search exactly `n` nodes and stop.
- **`go movetime t`** — think for exactly `t` milliseconds (a hard wall,
  with the soft-stop factors disabled).

Why do these matter so much? Because **wall-clock time is not
reproducible.** Run the same `go wtime ...` twice and you will search a
slightly different number of nodes each time — the CPU was a hair busier,
the OS scheduled differently, the clock ticked over at a different node.
That is fine for playing but poison for testing: if you cannot reproduce a
run, you cannot tell whether a change altered the engine's *behavior* or
just its *luck*.

**Fixed-node and fixed-depth modes remove time from the equation entirely.**
`go nodes 50000` searches the same 50,000 nodes, visits the same
positions, and — if the engine is deterministic — makes the same move,
every time, on any hardware. File 14 leans on this constantly: fixed-node
matches let you compare two networks' *quality per node* without the
confound of one being faster to run, and they are immune to a busy test
machine, so all the CPU cores can play games at once. There is even a
`nodestime` mode that substitutes a node budget for the wall clock, giving
bit-reproducible *time management* for testing the time manager itself.

The catch, which file 14 develops: a fixed-node comparison deliberately
*hides* the speed axis. A network that is stronger per node but slower to
evaluate can win a fixed-node test and *lose* a real-clock game. So
fixed-node is the research microscope; a real time-control match is the
ship gate. Both are needed, for different questions.

## 5. Determinism and the bench fingerprint

Which brings us to the discipline that ties this file to the rest of the
engine's engineering culture: **reproducibility as a first-class invariant.**

ARCHON has a command, `bench`, that runs the search over a fixed suite of
positions to a fixed depth and prints, among other things, a single number:
`Nodes searched`. Because the search is deterministic under a fixed-depth
limit — same positions, same depth, same code, same result — that node
count is a **fingerprint** of the engine's exact search behavior. The
canonical invocation is pinned:

```
   ./target/release/chessengine bench 16 1 6 default depth
```

and at the time of writing the default build produces exactly **701605**
nodes. Every commit that can affect search behavior carries this number in
its commit message as a `Bench: <nodes>` trailer.

The power of this convention is that it makes an otherwise-fuzzy question
sharp. Suppose you refactor the move generator and claim it is "just a
cleanup, no behavior change." How do you *prove* it? Run bench before and
after: if the number is bit-for-bit identical, the search visited exactly
the same tree, so the refactor was genuinely behavior-neutral. If the
number *changed*, the patch is **functional** by definition — it altered
what the engine does — and must be validated by playing games (file 14), no
matter how innocent it looked. A single number, and it collapses "did this
change behavior?" from an argument into a measurement.

This is why so much of the engine is built around determinism: the
avoidance of unsynchronized shared state on the single-thread path, the
"one correct reference, several implementations validated against it"
pattern for SIMD and move generation, the fixed-node testing modes. It all
serves the same end — making the engine's behavior a *reproducible
function* of its code, so that when the numbers move you know it was the
code that moved them. Determinism is not an aesthetic preference here; it
is the foundation the entire testing methodology stands on.

(Note the careful scoping: the fingerprint is reproducible for *fixed-depth*
bench. A parallel search across multiple threads is intentionally *not*
bit-reproducible — threads race through the shared transposition table in a
nondeterministic order — which is exactly why bench pins one thread and a
fixed depth. Reproducibility is arranged where it is needed for testing,
and deliberately traded away where parallel speed is worth more.)

## What you now know

- **UCI** is a line-oriented text protocol driven as a **state machine**:
  `uci`/`uciok` handshake, `isready`/`readyok` sync, `position`, `go`,
  streamed `info` lines, and exactly one `bestmove` per `go` — a contract
  ARCHON upholds in every case, including `infinite` and `ponder`.
- Time allocation produces a **soft bound** (`optimum`, aim to finish here,
  checked *between iterations*) and a **hard bound** (`maximum`, never
  exceed, checked *during* the search) from the clock, increment, and move
  horizon.
- The soft target is modulated by **stability/instability** signals —
  falling eval and a flip-flopping best move *extend* time; a stable best
  move *gives it back* — so the engine spends the clock where the outcome
  is in doubt.
- **Fixed-node** and **fixed-depth** modes strip out wall-clock
  nondeterminism, which is what makes controlled testing possible; the
  trade-off is that fixed-node deliberately hides the inference-speed axis.
- The **bench fingerprint** (`Bench: 701605`) is a deterministic node count
  that classifies any patch as behavior-neutral or functional, anchoring
  the whole engine's reproducibility discipline.

We keep invoking "play games to find out if it's really stronger." It is
finally time to make that rigorous. The next file is about the statistics
of chess testing: why "it looks stronger" is almost always wrong, how Elo
and error bars really work, and the sequential test (SPRT) that lets you
reach a verdict in a third of the games — the machinery behind every
"+X Elo" claim in this series. Continue to `14-testing-and-statistics.md`.
