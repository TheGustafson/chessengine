# Reckless — UCI protocol & time management

Source studied: Reckless (Rust, edition 2024, AGPLv3) at the clone under
`reckless_reference/Reckless`, tag/branch `v0.10.0-dev` (~3833 SPCC). Files
read in full: `src/uci.rs` (548 lines), `src/time.rs` (103 lines). Followed
into: `src/search.rs` (the iterative-deepening driver that calls the time
manager and prints `info`), `src/thread.rs` (`ThreadData`, `SharedContext`,
`Counter`, `Status`, `print_uci_info`), `src/threadpool.rs`
(`execute_searches`, the persistent-worker dispatch `go` blocks on), and
`src/parameters.rs` (the const/tunable macro). Every citation below is
`<file>:<line>` against that tree — **no Reckless source is reproduced**;
formulas are re-derived in prose/pseudocode, and a handful of single
constants are quoted verbatim where exactness matters (permitted, D27).

Read alongside `stockfish_reference/stockfish_search_notes.md` §13.3–§13.5
(SF dev `eca43a97`) throughout — this note assumes that one and calls out
deltas rather than re-deriving shared ground. Compared against our own
`design/12-uci-time-management.md` (Reviewed v1, 1177 lines) and the current
`src/timeman.rs` (591 lines) / `src/uci.rs` (2193 lines) — P7 has already
landed the basic clock-TM subset (commit `d2fd281`, "clock TM live — deepen
only while the optimum lasts"); the four-factor soft stop, ponder,
`searchmoves`, `nodestime` end-to-end, and MultiPV are scaffolded but pinned
at v1 defaults, explicitly deferred to **P10**.

The ID-loop-internal half of this story (the five-factor soft-stop
multiplier, the `pv_stability`/`eval_stability` counters, the
forgotten-mate/aborted-loss hygiene) is already covered from the
`search.rs` side in `reckless_reference/08-search-id-aspiration-qsearch.md`
§2.3–§2.4; this note covers it again only from the `TimeManager`/protocol
side that note explicitly put out of its scope, and otherwise points there
rather than re-deriving.

**Top-line framing**: Reckless's entire UCI+TM surface (651 lines across the
two files) is dramatically smaller than Stockfish's (`uci.cpp` + `timeman.cpp`
+ the `engine.cpp` options table) and smaller than our own *design doc*
(1177 lines of prescriptive spec for the same scope) — not because Reckless
is unfinished, but because it deliberately doesn't implement several
protocol features SF and our design treat as required (ponder, `searchmoves`,
`nodestime`, WDL output, a debug log). Where it does implement a feature, the
mechanism is consistently simpler than SF's and, in one place (the
multi-threaded soft-stop), genuinely more sophisticated than SF's.

---

## 1. Feature-surface comparison

| Feature | Stockfish | Our design (doc 12) | Reckless |
|---|---|---|---|
| `go` axes | all, freely combined | all, freely combined | one `Limits` enum variant at a time (§2) |
| Clock model | movestogo / sudden-death+inc, ply-driven log formulas | same (adopted from SF) | fullmove-driven exponential (Fischer) / flat division (Cyclic) |
| Ponder | yes | designed, not yet wired (P10) | **absent** — no `go ponder`, no `Ponder` option |
| `searchmoves` | yes | designed, not yet wired (P10) | **absent** |
| `nodestime` | yes | implemented (`src/timeman.rs`) | **absent** |
| MultiPV | yes | designed, pinned at 1 (P10) | yes, vote-aware (§7) |
| `UCI_ShowWDL` | yes (own fitted model) | deferred to NNUE (D21) | **absent** — own `normalize_to_cp` only, no win/draw/loss triple (§10) |
| `Debug Log File` | no | designed (§3.5) | **absent** |
| Single-reply time cap | yes (561.7 ms) | implemented (`ONE_REPLY_TOTAL_TIME_CAP`) | **absent** |
| Opportunistic "found a mate, stop early" during a normal timed search | yes | designed (P10) | **absent** (only explicit `go mate n` self-stops, §5) |
| Soft-stop is multi-threaded-vote-aware | no (main thread decides alone) | not yet (single-thread today) | **yes** — genuinely ahead here (§5, §7) |
| `UCI_Chess960` | yes | deferred | yes (`settings.frc`) |
| SPSA/tunable-constant scaffold | fishtest-external | not yet | macro-based, declared but unpopulated in this snapshot (§11) |

---

## 2. `Limits`: a sum type instead of a struct-of-optionals

```
enum Limits { Infinite, Depth(i32), Time(u64), Nodes(u64), Mate(u64),
              Fischer(main_ms, inc_ms), Cyclic(main_ms, inc_ms, movestogo) }
```
(`time.rs:10-18`.)

`parse_limits` (`uci.rs:389-432`) walks the `go` tokens two at a time
(`chunks(2)`), color-filtering `wtime/btime`/`winc/binc` into local
`main`/`inc` accumulators and `movestogo` into a third; the moment it sees
`depth`/`movetime`/`nodes`/`mate` with a positive value it **returns
immediately** with that variant, discarding every other token in the line
(`uci.rs:404-408`). Only if none of those four appeared does it fall through
to build `Fischer`/`Cyclic`/`Infinite` from the clock accumulators.

This is the textbook "make invalid states unrepresentable" idiom: a
`Limits::Depth(5)` value cannot simultaneously carry a stray `movetime`,
whereas our `Limits` (design/12 §3.6, `src/timeman.rs:96-120`) is a flat
struct with `Option`/zero-sentinel fields for every axis at once — closer to
SF's C-struct `LimitsType`, and deliberately so: a real GUI legitimately
combines the *clock* fields with each other (`wtime`+`btime`+`winc`+`binc`+
`movestogo`), and our own tooling combines `nodes` with other axes for
fixed-node proxy SPRTs (design/14 §9.4). Reckless's enum can represent the
clock combination fine (that's one variant, `Fischer`/`Cyclic`) but genuinely
**cannot** represent "stop at whichever of depth 20 or 5,000,000 nodes comes
first" — first-token-order wins outright and the rest of the line is
silently dropped. In practice GUIs essentially never combine
depth/movetime/nodes/mate with *each other*, so this is a low-risk
simplification, not a bug we've seen anyone hit — but it's a real
expressiveness trade our flat struct doesn't have to make (§"Adapt to us" #5
proposes a narrower version of the same idiom that doesn't lose this).

`TimeManager` itself (`time.rs:22-28`) has **no persisted per-game state at
all** — no analogue of our/SF's `TimePersist` (`original_time_adjust`,
`previous_time_reduction`, `best_previous_average_score`). The only
cross-`go` memory is a single centipawn scalar,
`ThreadData::previous_best_score` (`thread.rs:235`, default `0` — not a
`VALUE_INFINITE` sentinel needing a first-move special case, since `0`
already reads as "assume roughly level," which is exactly what SF/our design
fall back to for move 1 via an explicit branch, doc 12 §3.9). `ThreadData`
is recreated from scratch by `ThreadPool::clear`/`set_count`
(`threadpool.rs:73-80`) on `ucinewgame` (`uci.rs:177-186`), which is how
`previous_best_score` resets to 0 without any dedicated "persist" type to
reset.

---

## 3. Bound formulas — `TimeManager::new` (`time.rs:31-69`)

Both bounds are computed **once**, at construction, directly from the raw
clock — unlike SF/ours, where `maximum` is derived as a multiple of
`optimum` (`maximum ≈ max_scale · optimum`, capped by a clock fraction),
Reckless computes `soft`/`hard` as two **independent** closed-form functions
of the same remaining time. Whichever fires first, at runtime, wins — there
is no code that requires one to be a function of the other.

### 3.1 Fischer (sudden death / increment) — `time.rs:40-49`

```
soft_scale(fullmove) = 0.0594 − 0.0492 · e^(−0.0386 · fullmove)
hard_scale            = 0.7281                              (constant)

soft_bound = soft_scale · (main − overhead) + 0.75 · inc
hard_bound = hard_scale · (main − overhead) + 0.75 · inc

soft = min(soft_bound, main − overhead)
hard = min(hard_bound, main − overhead)
```

`soft_scale` is a saturating exponential: ~1.02% of the remaining clock at
move 1 (`e^0 ≈ 1`: `0.0594 − 0.0492 = 0.0102`), rising smoothly toward a
5.94% asymptote as the game runs long. This single curve stands in for the
whole of SF's/our `opt_scale`/`opt_constant`/`original_time_adjust`/
`time_left`-horizon-projection stack (design/12 §3.7 steps 3–7) — no
`movestogo`-horizon fiction, no `log10(time)`-fitted correction term, no
per-game "compute once" cached constant. `hard_scale` has **no game-phase
term at all** — a flat 72.81% of the remaining (overhead-adjusted) clock,
always, versus SF's/our `max_scale = min(6.873, max_constant + ply/12.352)`
(itself a function of `log10(time)` and ply, and then multiplied onto
`optimum`, not onto raw time).

One consequence worth naming: because `soft_scale` grows with the game but
`hard_scale` doesn't, the `hard/soft` ratio is **not constant** — roughly
`0.7281/0.0102 ≈ 71×` at move 1, shrinking to `0.7281/0.0594 ≈ 12.3×` late
game. Early on, the two bounds are computed as independent (large)
multiples of each other purely as an artifact of the curve shapes, not by
design intent the way SF's `max_scale` formula is.

`fullmove` is `board.fullmove_number()` (`uci.rs:190`) — the **full-move**
counter (increments once per Black move), not the **game ply** SF/our design
use (`ply = 2·(fullmove−1) + (stm == Black)`, design/12 §3.7). White's and
Black's move within the same full-move number therefore get the *identical*
`soft_scale` value; SF/our design distinguish them (twice the temporal
resolution). A minor precision difference, likely invisible in practice.

### 3.2 Cyclic (`movestogo`) — `time.rs:50-56`

```
base = main / movestogo + 0.75 · inc
soft = min(1.0 · base, main + inc)
hard = min(5.0 · base, main + inc)
```

Plain "divide the remaining bank evenly across the moves left," 0.75 of the
increment folded in, capped at a flat `5×` for the hard bound. No
sub-one-second horizon shrink (SF/ours shrink `mtg` itself below 1000 ms of
clock, design/12 §3.7 step 3) — an extremely low clock under `movestogo`
divides by whatever `movestogo` was last reported, with no protective floor
against a degenerate small quotient. (A `movestogo=0` from a GUI would divide
by zero — `f64` semantics make this `+inf`, harmlessly absorbed by the
`.min(main + inc)` cap rather than panicking, but it is not a case Reckless
guards against explicitly the way a `.max(1)` clamp would; UCI GUIs never
send `movestogo 0` in practice.)

### 3.3 `Time` / `Nodes` / `Depth` / `Mate` / `Infinite` — `time.rs:36-39, 57-61, 75-94`

- **`Time(ms)`** (i.e. `go movetime`): `soft = hard = ms`, both then get the
  flat internal shave below (§3.4) — so `go movetime 100` actually enforces
  a hard wall at **~85 ms**, not 100. A deliberate uniform undershoot margin
  applied to *every* bound, including `movetime`. Our own `check_time`
  treats `movetime` as an exact wall (`elapsed >= t`, design/12 §3.6/§3.8) —
  no analogous shave (see "Adapt to us" #4).
- **`Nodes(n)`**: `soft`/`hard` fall back to `u64::MAX` (irrelevant, `_` arm
  of the match at construction, `time.rs:57-60`); the real limit is
  re-checked directly against `n` from `self.limits` every time
  `soft_limit`/`check_time` runs (`time.rs:78, 91`), not via the precomputed
  `Duration` fields.
- **`Depth`/`Mate`/`Infinite`**: same `u64::MAX` placeholder — these are
  enforced elsewhere entirely: `Depth` in the ID loop's own per-iteration
  guard (`search.rs:79-85`, main thread only); `Mate` after each *completed*
  iteration (`search.rs:237-243`, same `Score::MATE − |score| <= 2·n` test as
  SF/ours); `Infinite` never self-stops — see §6 for how `bestmove` is still
  never printed early without an explicit busy-wait state machine.

### 3.4 Overhead, layered twice — `time.rs:20, 66-68`; `uci.rs:34` (Settings default)

A flat `TIME_OVERHEAD_MS = 15` is subtracted from **both** `soft_bound` and
`hard_bound` unconditionally at the very end of `new()`
(`soft.saturating_sub(TIME_OVERHEAD_MS)`, same for hard) — regardless of
which `Limits` variant produced them, including `Time`/`movetime`. This
stacks with the separate `MoveOverhead` UCI option (`Settings` default
**100 ms**, `uci.rs:34`), which is subtracted from `main` *before* the
Fischer/Cyclic formulas run (`main.saturating_sub(move_overhead)`,
`time.rs:44-48, 51`). Net default overhead ≈ **115 ms** (100 + 15) for
clock-based modes, vs our/SF's single-layer `Move Overhead` default of
**10 ms** (design/12 §3.5) — Reckless ships a far more conservative default
cushion, and uniquely, applies a piece of it (the fixed 15 ms) even to modes
our design currently leaves completely unpadded (`movetime`, `nodes`).

---

## 4. Hard stop — `TimeManager::check_time` (`time.rs:84-94`)

```
fn check_time(td):
    if td.completed_depth == 0: return false      // never before iteration 1 finishes
    match limits:
        Infinite | Depth(_) | Mate(_) -> false
        Nodes(n)                       -> aggregate_nodes() > n
        _ (Time/Fischer/Cyclic)        -> (td.nodes() & 2047 == 2047)
                                            && elapsed() >= hard_bound
```

Two properties worth calling out precisely:

1. **`completed_depth == 0` guard.** No hard stop of any kind can fire
   before the *first* iterative-deepening iteration has fully completed —
   not for `movetime`, not for `nodes`, not for the clock wall. Under an
   absurdly small budget (`go movetime 1`, or a 1-node-per-check nodestime
   equivalent if Reckless had one), the engine still finishes depth 1 before
   it will ever consider stopping, guaranteeing a real (if shallow)
   `bestmove` rather than one built from an aborted, possibly-empty search.
   Our own `check_time` (design/12 §3.8, `src/search/mod.rs:2175-2202`) has
   **no equivalent guard** — `tm_stop`/`movetime_stop`/`nodes_stop` can all
   fire from the very first poll, including mid-iteration-1, if `maximum()`/
   `movetime`/`nodes` is small enough. This is a real, cheap, checkable
   hardening gap (see "Adapt to us" #1).
2. **Cadence via bitmask on the per-thread node counter**, not a separate
   decrementing field: `td.nodes() & 2047 == 2047` reuses the counter each
   thread already increments on every `make_move`
   (`nodes.increment(id)`, `search.rs:1433`) rather than maintaining a
   dedicated countdown (our/SF's `calls_cnt`, reloaded to 512 or
   `(nodes_limit/1024).min(512)`, design/12 §3.8). One fewer struct field,
   at the cost of a fixed granularity — every **2048** node-entries (SF/ours
   default to every **512**, 4× finer) — and only for the main thread
   (`td.id == 0` at both call sites).

`check_time` is polled at **two** call sites: the main negamax entry
(`search.rs:336`) *and* qsearch's own entry (`search.rs:1199`) — the hard
wall is re-checked at qsearch nodes too, bounding worst-case stop latency
even under a long forcing-capture qsearch dive. Our own `check_time` call
lives only in the main search function, with an explicit comment that
"qsearch relies on these checks above it" (`src/search/mod.rs:960-961`) — a
deliberate scoping choice, not an oversight, but one Reckless's evidence
suggests is worth revisiting (see "Adapt to us" #2).

---

## 5. Soft stop — `TimeManager::soft_limit` + the multiplier closure

```
fn soft_limit(td, multiplier):
    match limits:
        Infinite | Depth(_) | Mate(_) -> false
        Nodes(n)                       -> aggregate_nodes() >= n
        Time(ms)                       -> elapsed() >= ms
        _ (Fischer/Cyclic)             -> elapsed() >= soft_bound × multiplier()
```
(`time.rs:75-82`.) Note this does **not** clamp against `hard_bound` the way
SF/our design explicitly compute `elapsed > min(total_time, maximum)`
(design/12 §3.9) — the two mechanisms simply run independently and whichever
fires first (via its own, separately-polled, call site) stops the search.
Harmless in practice: if the soft decision says "keep going" while actually
past the hard wall, the very next `check_time` poll (§4) catches it a
fraction of a second later.

The `multiplier` argument is a **closure** built fresh each iteration in
`search::start` (`search.rs:245-263`) and passed by reference into
`soft_limit` — the TM layer owns *when* to evaluate it, the ID loop owns
*what* it computes, with neither side needing the other's internals. The
five factors it multiplies (node-fraction "effort," score-trend, PV
stability, eval stability, best-move-change count via `ln_1p`) are already
covered in depth from the `search.rs` side in
`reckless_reference/08-search-id-aspiration-qsearch.md` §2.3 — that note
also covers the forgotten-mate/aborted-loss hygiene (§2.4), which
independently matches the mechanism our own P7 hygiene-pass commit
(`8a7e476`, `forgotten_mate`/`aborted_loss` in `src/search/mod.rs:729-765`)
already implements — a convergent validation, not a gap, worth recording
once here rather than re-deriving: a top-2 engine protects its `bestmove`
output from exactly the same two failure modes we independently guarded
against.

**The genuinely novel piece, from the TM side**: soft-stop under multiple
threads is a **majority vote**, not "first thread to want to stop, stops
everyone" (`search.rs:265-280`, `SharedContext::soft_stop_votes:
AtomicUsize`, `thread.rs:163`). Every thread running under
`use_time_management()` (all of them, not just thread 0 — `soft_limit` takes
no `id == 0` guard, unlike `check_time`) independently evaluates its own
`multiplier()` against its own `root_moves[0]`/`nodes()`/stability counters,
and casts (or later retracts, via `fetch_sub`) exactly one vote:

```
majority = ceil(65% × thread_count)
if my_soft_limit_says_stop():
    if I haven't already voted: votes += 1; remember I voted
    if votes >= majority: Status = STOPPED
elif I previously voted:
    votes -= 1; forget that I voted
```

With `thread_count == 1` (the common default), `majority = ceil(0.65) = 1`,
so a single thread's own vote alone reaches the threshold immediately — the
mechanism degenerates to exactly the ordinary single-thread soft-stop with
zero behavior change, which is presumably why it was safe to make the
default path. Only under lazy SMP (`Threads > 1`) does the supermajority
requirement matter, damping the case where one helper thread — searching
the same position from a different depth/effort profile — would otherwise
unilaterally end everyone else's search. This is a structurally different
lazy-SMP time-management shape from SF's "only the main thread's opinion
counts, helpers just watch `stop`" (stockfish_reference/stockfish_search_notes.md
§13.3) and from our own design (design/12 §3.2: `PoolShared.stop`/
`increase_depth` are written only by the main worker). Flagged already in
`reckless_reference/08` §2.3 from the ID-loop side; recorded here from the
`TimeManager`/vote-counter side for whoever implements P11 (see "Adapt to
us" #6, which is the P10/P11-boundary companion of `08`'s item 7).

---

## 6. The UCI command loop: two threads, not three roles

Reckless splits I/O across exactly two OS threads, and folds SF's/our
three-role model (dumb UI thread / main search worker / helper workers) into
two:

- **`spawn_listener`** (`uci.rs:113-149`): reads stdin in a tight loop.
  `isready` is answered inline (`println!("readyok")`, never queued);
  `stop`/`quit` set the shared `Status` atomic **directly**, from this
  thread, regardless of what the other thread is doing (`quit` additionally
  forwards a `"quit"` message so the other thread eventually unwinds and
  exits, `uci.rs:131-134`). **Every other command is forwarded to the
  channel only if `Status != RUNNING`** (`uci.rs:140-143`) — a `position`,
  `setoption`, or `go` arriving while a search is in flight is **silently
  dropped**, not queued and not applied later, following the UCI spec's
  "ignore commands unexpected in the current state" clause literally.
- **`message_loop`** (`uci.rs:41-111`): the thread that actually executes
  commands, including `go`. Critically, `go` (`uci.rs:188-251`) calls
  `ThreadPool::execute_searches`, which is **fully synchronous** — it spawns
  the search onto the persistent worker pool via `std::thread::scope` and
  then **blocks on `handler.join()` for every thread** (`threadpool.rs:
  199-201`) before `go` can return. `message_loop` is therefore completely
  unresponsive to new commands for the entire duration of a search — which
  is fine, because `spawn_listener` (a separate OS thread) is the one
  reading stdin live and has already special-cased the three commands that
  must always be responsive.

This gets the "never print `bestmove` before it's allowed" invariant
**for free**, structurally, rather than via an explicit busy-wait state
machine: under `Limits::Infinite`, the ID loop has no self-stop condition at
all (`soft_limit`/`check_time` both return `false` for that variant), so
every thread's `search::start` simply keeps deepening — usually forever in
practice — until the listener thread's `stop` handler flips `Status`, at
which point the next recursive `search()` call anywhere in the tree observes
`STOPPED` (`search.rs:315-317`) and unwinds, the ID loop's own `if status ==
STOPPED { break }` fires (`search.rs:233-235`), and *only then* can
`execute_searches`'s `.join()` return and `go` proceed to print `bestmove`.
The `while status != STOPPED { spin_loop() }` block that *does* exist
(`search.rs:287-291`) only matters in the degenerate case of `MAX_PLY`
exhaustion before a `stop` arrives — for `Infinite`/ponder-shaped modes in
practice, the "busy-wait" is simply "the loop hasn't been told to stop yet,"
with no separate mechanism needed.

**Deliberate divergence from our design, worth being explicit about**: our
`§3.2` "implicit stop" rule (design/12) makes a `setoption`/`position`/`go`
arriving mid-search trigger a `stop` first, so it can still apply once the
search flushes — chosen specifically so a misbehaving GUI that sends these
mid-search doesn't get silently ignored or (per SF) hang the engine.
Reckless's drop-on-arrival policy is simpler code but trusts the GUI more:
a `position startpos` sent while a `go infinite` is running is not queued,
not applied, and not even flagged — it is gone. This is a place our design
is already **more robust** than Reckless's, not something to adopt (recorded
so a future reader doesn't mistake the omission for an oversight worth
copying).

---

## 7. `go`'s `bestmove`: a weighted vote across threads, not "thread 0 wins"

After `execute_searches` returns (`uci.rs:194-249`):

- Empty root moves → `bestmove (none)` immediately (mirrors SF/our
  `info depth 0` + `bestmove (none)` path, design/12 §3.2 step 2, though
  Reckless's `print_uci_no_move` — `thread.rs:376-382` — prints only the
  bare `info depth 0 score cp 0`/`score mate 0` line, with no separate
  `bestmove` call site duplication needed).
- Otherwise: `min_score` = the minimum `root_moves[0].score` across all
  threads; each thread casts `vote_value = (its_score − min_score + 10) ×
  its_completed_depth` onto a `HashMap<&Move, i32>` keyed by that thread's
  best move (`uci.rs:199-204`).
- Thread 0 is the default winner; a challenger thread displaces it only if
  (in order, `uci.rs:211-236`): thread 0's score is a decisive win and the
  challenger's is *strictly better*; thread 0's score is a decisive loss and
  the challenger's is *strictly less bad*; the challenger has **any**
  decisive score at all (win or loss) while thread 0 has neither; or,
  falling through to the ordinary case, the challenger isn't itself a loss
  and wins on `vote_value` (ties broken by the raw `vote_value` magnitude,
  which folds in depth).
- Voting is skipped entirely when the search was `Limits::Depth`-limited
  (bench reproducibility, matching SF/design/12's exemption) **or** when
  `MultiPV > 1` (`uci.rs:209`) — Reckless doesn't attempt cross-thread
  MultiPV reconciliation at all.
- If the vote picked a thread other than 0, `go` explicitly prints that
  thread's final `info` line before `bestmove` (`uci.rs:244-247`) — the same
  "the GUI's last-seen PV must match the printed bestmove" contract our
  design states explicitly (§3.13) — necessary here because helper threads
  run with `Report::None` and have never printed anything during the search.

This is a concrete, different worked alternative to SF's best-thread voting
(`votes[best move] += score − minScore + 14`,
stockfish_reference/stockfish_search_notes.md §13.2) that our own design/12
§7 currently defers entirely to "the threading doc" without a formula —
worth keeping alongside SF's as two independently-arrived-at answers to the
same question when design/13 gets its P11 pass (see "Adapt to us" #7).

---

## 8. Options table: direct pattern match, no typed registry

`uci()` (`uci.rs:151-169`) prints a fixed, hand-written list of `option
name ... type ...` lines (`Hash`, `Threads`, `MoveOverhead`, `Minimal`,
`Clear Hash`, `UCI_Chess960`, `MultiPV`, `SyzygyPath` behind a feature flag,
plus SPSA-tunable parameters behind another). `set_option`
(`uci.rs:286-329`) is a single `match` on the token slice, one arm per
option name, mutating either a plain `Settings` struct field (`uci.rs:22-27`:
`frc`, `multi_pv`, `move_overhead`, `report`) or reaching into
`shared`/`threads` directly (TT resize/clear, thread-pool resize). There is
**no** `OptionEntry`/`Options` registry type, no typed accessors, no
range-clamping helper reused across entries (each arm does its own
`v.parse().unwrap()` or `.unwrap_or_default()`) — noticeably less structure
than our own `Options`/`OptionEntry`/`OptionKind` table (design/12 §3.5,
`src/uci.rs:190-434` in our tree) or SF's options map. This buys Reckless
brevity at the cost of duplicated parsing boilerplate per option and no
single place that enumerates "every option and its type" other than the
`uci()` print function itself (which must be kept in sync with `set_option`
by hand — our registry-driven table generates both from one source of
truth, and is the more maintainable shape as the option count grows).

---

## 9. `position`: non-atomic move application

`position` (`uci.rs:253-284`) walks tokens, and for the `moves` tail simply
calls `make_uci_move` once per token (`uci.rs:279-284`), which looks the
token up against the *current* legal move list and applies it **if found**,
silently doing nothing otherwise — crucially, **it keeps processing the
remaining tokens against whatever board state resulted**, rather than
aborting or rolling back. A `position startpos moves e2e4 bogus e7e5`
therefore applies `e2e4`, skips `bogus`, and *still* applies `e7e5` against
the post-`e2e4` position (exercised directly by
`uci.rs` test `test_position_invalid_move_ignored`). Our design's
`position ... moves` handling is explicitly **atomic**: any unparseable
token leaves the whole command a no-op, the engine's position unchanged
(design/12 §3.3, chosen as "a deliberate, documented divergence for
robustness" from SF, which exits the process on the same input). Reckless
takes a third position on the same fork: neither SF's fatal exit nor our
all-or-nothing rollback, just "skip the bad token, keep going" — simpler
than either, but capable of silently landing the engine on a position the
GUI never intended if a single token in a long `moves` list is malformed
(e.g. truncated by a transport glitch). Not proposed as an adoption target;
recorded as the third data point on a fork our design already made a
considered choice about.

---

## 10. `info` output and score normalization

`ThreadData::print_uci_info` (`thread.rs:298-374`) emits one line per
MultiPV index: `depth seldepth multipv score nodes time nps hashfull tbhits
pv`. Two differences from SF's field order/content (design/12 §3.13 adopts
SF's order verbatim): **no `wdl` field ever** (Reckless has no WDL model at
all, only the scalar `cp` conversion below), and **no explicit
`currmove`/`currmovenumber` output at any throttle threshold** — grepping
`thread.rs`/`search.rs` turns up no `currmove` string; Reckless simply
doesn't implement that particular `info` sub-line SF and our design both
specify (design/12 §3.13). The `10_000_000`-node throttle for extra
aspiration-fail-high/low intermediate updates (`search.rs:175, 228`) is the
same threshold value our design names `INFO_THROTTLE_NODES`
(design/12 §3.13) — but Reckless writes the literal `10_000_000` inline at
both call sites rather than through a named constant, which is exactly the
kind of "bare tunable literal in search-layer code" our own conventions
(D20, design/01) treat as a review defect — a place we already hold
ourselves to a higher bar.

**Score normalization** (`types/score.rs:58-70`, `normalize_to_cp`) already
does what our design defers to "once our NNUE is trained" (design/12 §3.14):
a from-scratch cubic polynomial in material
(`pawns + 3·minors + 5·rooks + 9·queens`, clamped to `[16, 78]`, `/58.0` —
note Reckless's floor is **16**, not SF's **17**), evaluated as
`−285.1v³ + 642.5v² − 455.5v + 464.8`, with `cp = round(100 · score /
normalization)`. This is Reckless's **own** fit (not SF's coefficients,
correctly — SF's fit is meaningless for a different engine's eval, exactly
the reasoning our design already states for why we don't adopt SF's `(a,b)`
polynomials verbatim), and it ships with **no accompanying `wdl` triple** —
`+100 cp` is calibrated to "50% win probability, per Reckless's own
material-scaled eval," full stop, no separate win/draw/loss percentage
output at all. A smaller, cheaper deliverable than SF's/our eventual
`UCI_ShowWDL` plan, and one concrete existence proof that shipping
*normalization without WDL* is a viable, real-world middle ground worth
considering as an interim step (see "Adapt to us" #8).

`Report` (`search.rs:26-30`: `Full`/`Minimal`/`None`) gates verbosity per
search: helper threads always run `Report::None` (never print anything,
`threadpool.rs:193`); the `Minimal` UCI option
(`setoption name Minimal value true`, `uci.rs:288-292`) suppresses every
per-iteration `info` line and prints exactly **one** summary line right
before `bestmove` (`search.rs:293-295`) instead. A cheap verbosity knob with
no search-behavior effect at all — see "Adapt to us" #9.

---

## 11. The tunable-constant macro (`parameters.rs:1-30`)

A `define! { type name: value; ... }` macro list, expanded two different
ways by a `spsa` Cargo feature:

- **Without** the feature: each entry becomes a plain `pub const fn name() ->
  type { value }` — zero runtime cost, effectively identical to our own
  `search/params.rs` convention (D20: "tuned search constants live only in
  `search/params.rs`, named, never inline").
- **With** the feature: the same declarations instead expand to a
  `static mut` behind an unsafe getter, plus one generic
  `set_parameter(name: &str, value: &str)` and `print_options()` pair driven
  by `stringify!(name)` — i.e. every constant in the list becomes a UCI
  `setoption`-tunable string option, with **zero per-constant boilerplate**,
  for the cost of one build-time feature switch.

Caveat worth being explicit about: in this cloned snapshot the macro is
declared but **not invoked anywhere** — no constants (not `time.rs`'s, not
any search constant) are actually routed through `define!` yet. The
exportable idea is the *macro pattern itself* (single source-of-truth list
→ free `const fn` in production, free SPSA surface in a tuning build), not
a specific constant list to copy. Relevant whenever we build an
OpenBench/SPSA-style automated tuning pass over our own pinned constants
(design/14, D20) — see "Adapt to us" #10.

---

## 12. Summary — same / simpler / cleverer / missing, vs Stockfish and vs us

| Subsystem | Stockfish | Our design (doc 12) | Reckless | Verdict |
|---|---|---|---|---|
| `Limits` shape | flat struct, all fields | flat struct, all fields | tagged enum, one axis at a time | more idiomatic, less expressive across non-clock axes |
| Clock formula | ply/log10-fitted, `time_left` horizon projection | matches SF | fullmove-driven exponential (soft) / flat constant (hard) | much simpler, coarser resolution (fullmove not ply) |
| Bound coupling | `maximum` = f(`optimum`, ply) | matches SF | `soft`/`hard` independent functions of raw time | different architecture, same practical effect |
| Overhead | one option, one layer | matches SF (10 ms default) | option (100 ms) + fixed internal shave (15 ms), applied even to `movetime` | more conservative default, extra safety margin we lack for `movetime` |
| Hard-stop guard vs empty search | none named explicitly | none yet | `completed_depth == 0` blocks any hard stop before iteration 1 | a real hardening idea we lack (Adapt #1) |
| Hard-stop cadence | 512-node countdown field | matches SF | 2048-node bitmask on the existing counter, no extra field | simpler mechanism, 4× coarser |
| Hard-stop in qsearch | implied via shared node cadence | polled only in main search (deliberate) | explicitly re-polled at qsearch entry | a real hardening idea we lack (Adapt #2) |
| Soft-stop factor stack | 4 factors, ply/iterValue-history-based | matches SF | 5 factors, purely per-iteration counters, no cross-`go` persistence | different set, less carried state (see `reckless_reference/08` §2.3) |
| Soft-stop across threads | main thread alone decides | not yet built | majority vote (⌈65%⌉), retractable | genuinely more sophisticated than SF (Adapt #6) |
| Best-thread/bestmove selection | vote by `score − min + 14` | deferred to threading doc | vote by `score − min + 10, × depth`, plus decisive-score overrides | a concrete alternative formula worth having on file (Adapt #7) |
| Ponder | full state machine | designed, not yet wired | absent entirely | a whole feature we're ahead on once P10 lands it |
| `searchmoves` | yes | designed | absent | ditto |
| `nodestime` | yes | implemented | absent | ditto |
| WDL output | fitted model + `wdl` triple | deferred to NNUE | own material-cubic `to_cp`, no `wdl` triple at all | validates "normalize without WDL" as a real interim step (Adapt #8) |
| Options registry | typed map | typed `Options`/`OptionEntry` table | one hand-written `match`, no registry type | ours is more maintainable at scale |
| `position ... moves` on bad token | process exit | atomic no-op, position unchanged | skip token, keep applying the rest | three different considered answers to the same fork |
| Command loop under active search | main thread does I/O, never blocked | implicit-stop-then-apply for mid-search commands | stdin-listener thread never blocked; mid-search commands silently dropped | ours is the more GUI-robust choice, deliberately |
| Debug/log tooling | none in SF core | `Debug Log File` option designed | absent | a feature we're ahead on once P10 lands it |
| Tunable-constant scaffold | external (fishtest) | none yet | `define!` macro (unpopulated in this snapshot) | a reusable pattern for whenever we build SPSA tooling (Adapt #10) |

---

## Adapt to us

Ordered roughly by effort, cheapest first. Every item that touches search
*behavior* (changes what gets stopped when) needs its own `Bench:` trailer
discipline (D24/design/14 §3.2) and, if it changes node counts, its own
SPRT — treat each as its own commit, not a bundle. Items that are pure
protocol surface (no search-behavior change) are bench-neutral by
construction and noted as such.

1. **Guard `check_time` so no hard stop can fire before the first completed
   iteration.** *Where*: design/12 §3.8 (`check_time`), `src/search/mod.rs`'s
   `check_time` (~line 2175) and the iterative-deepening driver's
   `root_depth`/completed-iteration tracking. Add a boolean (or reuse
   `root_depth > 0 && <iteration i finished>`) so `tm_stop`/`movetime_stop`/
   `nodes_stop` are all suppressed until depth 1 has actually completed —
   Reckless's `td.completed_depth == 0` early return (`time.rs:85-87`).
   *Phase*: P10 (TM completion). *Effort*: trivial (one field, one guard
   clause). *Impact*: correctness hardening for extreme budgets
   (`movetime`/`nodes` values too small to finish depth 1) — currently such
   a search could abort with no real best move computed; changes behavior
   only in that edge, so it still needs a `Bench:` trailer and a quick SPRT
   sanity check even though normal-budget games shouldn't be affected.

2. **Re-poll `check_time` at qsearch's own entry, not only at the main
   search node above it.** *Where*: design/12 §3.8 (cadence — extend it to
   name qsearch explicitly), qsearch's entry in `src/search/mod.rs` (the
   function documented at line ~1828). Add the same `self.check_time(stop)`
   call already made in the main search node, at the top of qsearch, gated
   identically. *Phase*: P10. *Effort*: trivial (one call, already-existing
   function). *Impact*: bounds worst-case stop latency under a long
   forcing-capture qsearch dive; changes node counts in principle (an
   aborted qsearch returns a different, "untrusted" value the same way an
   aborted main-search node already does) so it needs the `Bench:`
   trailer/SPRT treatment, but should be behavior-neutral except right at a
   stop boundary.

3. **Shave a small fixed epsilon off every enforced bound, including
   `movetime`, not just clock-based `maximum`.** *Where*: design/12 §3.6/§3.7
   (note the exception explicitly: "movetime t: hard wall... elapsed >= t")
   and §3.8's `check_time`. Introduce a small internal constant (a few ms,
   separate from the `Move Overhead` option since it should apply
   universally, including to `movetime`/`nodes` runs) subtracted from
   `maximum()` and compared against for `movetime` too — Reckless's flat
   `TIME_OVERHEAD_MS = 15` (`time.rs:20, 66-68`) applied unconditionally.
   *Phase*: P10. *Effort*: small (one named constant in `params.rs`, wire it
   into `check_time`'s three stop conditions). *Impact*: directly reduces
   time-forfeit risk (design/12 §8's "zero time forfeits" acceptance gate),
   most relevant at the bullet-ish self-play time controls our own
   D25 tooling exercises, where the unwind-and-print tail after `stop` is a
   larger fraction of the total budget.

4. **A `Minimal`-style low-verbosity report mode.** *Where*: design/12 §3.13
   (`info` output) — add `option name Minimal type check default false`;
   gate every per-iteration `info` emission on it, printing one final
   summary line right before `bestmove` when it's set. *Phase*: P10 (rides
   along with the rest of the options-table work). *Effort*: small (one
   option, a few `if` gates around existing print call sites — no search
   logic touched, bench-neutral, no `Bench:` trailer needed since it's pure
   I/O). *Impact*: operational quality-of-life for our own D25 self-play
   tooling (sprt.sh/gauntlet.sh log volume across thousands of games), not
   a strength change.

5. **A narrower "clock model" sum type nested inside our existing flat
   `Limits`**, without giving up the ability to combine `nodes`/`depth`/etc.
   the way real tooling requires. *Where*: design/12 §3.6/§3.7,
   `src/timeman.rs`. Today, `init` already branches on `limits.movestogo ==
   0` (`timeman.rs:291`) to pick sudden-death-vs-repeating math via an `i32`
   sentinel; wrap just that choice in a small enum
   (`ClockModel::SuddenDeath { movestogo_none } | ClockModel::Cyclic
   { movestogo }`, or similar) so the branch becomes a compiler-checked
   match instead of an integer comparison, while `depth`/`nodes`/`movetime`/
   `mate`/`infinite`/`ponder`/`searchmoves` stay exactly as they are today
   (all independently combinable, unlike Reckless's `Limits`). *Phase*: P10,
   internal refactor only. *Effort*: small-medium (a non-behavior-changing
   type change plus updating the two formula branches and their tests) —
   must reproduce the existing golden-test values bit-for-bit (verify with
   `check.sh` on both sides per the refactor discipline in CLAUDE.md).
   *Impact*: none on strength, a genuine "prefer the idiomatic Rust way"
   win (per CLAUDE.md's stated project goal) without Reckless's
   expressiveness trade-off.

6. **Majority-vote soft-stop across lazy-SMP threads, from the
   `TimeManager` side.** *Where*: design/12 §3.9 (soft stop) crossed with
   design/13 (threading, P11) — already flagged from the `search.rs` side
   in `reckless_reference/08` §2.3/item 7; this note's complement is the
   `TimeManager`-side shape: `soft_limit` takes no `id == 0` guard (every
   thread evaluates its own bound independently, `time.rs:75-82`), while
   `check_time` (the hard wall) stays main-thread-only
   (`search.rs:336, 1199`) — i.e. the hard wall is centrally enforced but
   the soft, "is it worth continuing" advisory is genuinely democratic.
   *Phase*: P11 (lazy SMP), pairs with completing P10's soft-stop factor
   stack first (today's single-threaded TM has nothing to vote about).
   *Effort*: medium (`AtomicUsize` vote counter + retraction path, both
   already worked out in Reckless, `search.rs:265-280`). *Impact*: plausible
   fix for the "one noisy helper thread stops everyone early" failure mode a
   naive lazy-SMP soft stop would otherwise have; worth prototyping against
   our own gauntlet once P11 lands rather than guessing the 65% figure.

7. **Record Reckless's best-thread vote formula as a second concrete
   candidate for design/13's still-unspecified "best-thread voting"
   section**, alongside SF's (already in
   `stockfish_reference/stockfish_search_notes.md` §13.2). *Where*: design/12
   §7 currently just references "the threading doc" without a formula;
   design/13 (P11) is where the pick actually needs to be made. Reckless's
   shape (`uci.rs:199-241`): `vote = (score − min_score + 10) × completed_depth`
   per thread's best move, with explicit decisive-score override branches
   ahead of the vote comparison, and voting skipped under `Limits::Depth`
   *or* `MultiPV > 1`. *Phase*: P11. *Effort*: none yet (this is a
   spec-writing input, not code) — the actual formula choice and its tests
   land with P11's design/13 draft. *Impact*: gives whoever writes that
   section two independently-tuned real-world answers to blend or choose
   between, rather than inventing one from scratch.

8. **Consider shipping `to_cp` normalization (once we have a self-play
   fitted polynomial, D21) *without* a `wdl` triple as a valid interim
   step**, rather than treating "normalization" and "`UCI_ShowWDL`" as one
   bundled deliverable. *Where*: design/12 §3.14 (open question 5 already
   asks "decide then whether `UCI_ShowWDL` ships with the first NNUE release
   or later") — Reckless (`types/score.rs:58-70`) is a real, shipped example
   of normalization-without-WDL at ~3833 SPCC. *Phase*: whenever P13 (NNUE)
   reaches the point of fitting our own `(a, b)`/polynomial coefficients.
   *Effort*: none now — this resolves an already-open question rather than
   adding new scope. *Impact*: lets us ship calibrated `cp` scores sooner
   without also having to validate a full WDL model in the same release.

9. **Do not adopt as-is**: Reckless's silently-dropped mid-search
   `position`/`setoption`/`go` (§6) and its non-atomic `position ... moves`
   partial application (§9). Both are working code in a top-2 engine and
   worth having read closely, but both are strictly less robust than the
   choices our design already made deliberately (design/12 §3.2's implicit
   stop, §3.3's atomic move-list application) — recorded here so a future
   pass doesn't mistake either omission for an idea worth copying.

10. **File away the `define!` const/tunable macro pattern** for whenever we
    build an SPSA/OpenBench-style automated tuning pass over
    `search/params.rs`. *Where*: design/01 (the `params.rs` convention, D20)
    gains a note once we build tuning tooling; not owned by design/12, cited
    here only because it was found while reading `time.rs`'s constant list.
    *Phase*: post-P10, whenever automated tuning becomes a priority (P9's
    SPRT-gated waves are the natural precursor). *Effort*: medium (the
    macro itself, `unsafe`-review for whatever the mutable-parameter storage
    turns out to be — `static mut` needs care, an `AtomicI64`/bit-cast
    wrapper may be the safer Rust-idiomatic equivalent) — genuinely bounded,
    not started from zero, since Reckless's macro shape is a complete,
    working reference. *Impact*: high **if and when** we start SPSA-tuning
    (zero-cost in production builds, free tunability in a tuning build) but
    zero until then — not worth building speculatively.
