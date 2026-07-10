# 13 — Threading: lazy SMP in safe Rust

Status: Draft v1

This document specifies the engine's multithreading layer: a lazy-SMP design
(D23) in which N identical searchers run on private copies of the root
position, diverge naturally through per-thread aspiration jitter and shared-TT
races, and elect a best move by voting when the search stops. Every piece of
mutable search state is single-owner inside a per-thread `Worker`; the shared
surface is deliberately tiny (the transposition table, three atomic flags, and
per-thread atomic counters). The whole layer is safe Rust — no `unsafe`, no
locks on the per-node search path, and no `Rc`/`RefCell` anywhere near search.
Threads are long-lived and parked between searches, mirroring Stockfish's
lifecycle — this is now D23's pinned mechanism (decisions.md was amended
2026-07-09 to persistent parked threads + `Arc<TT>`); `std::thread::scope`
remains documented as the fallback but is not chosen (§3.4).

---

## 1. Purpose and scope

**This document owns:**

- The `threads` module (D13): `ThreadPool`, `Worker` (as a container — its
  search *fields* are enumerated here, their algorithms live in the search
  doc), the per-thread job loop, and the wake/park protocol.
- The `start_thinking` sequence — everything between UCI `go` and the
  `bestmove` line, at the orchestration level.
- Main-thread-only responsibilities and the `SearchManager` null-object
  pattern.
- Best-thread voting.
- Cross-thread node/tbhit/bestMoveChanges aggregation.
- The Send/Sync story and the definitive list of accepted vs. forbidden races.

**This document does not own** (pointer is to the design doc covering that
topic; see §7 for the full dependency list):

- The TT's internal atomic layout, replacement policy, and generation aging —
  the TT doc (D19). This doc only consumes `&TT` as a `Sync` value and calls
  `tt.new_search()` — which must therefore be callable through `&TT`
  (§3.7 B).
- The iterative-deepening loop, aspiration windows, and every search
  heuristic — the search doc. This doc only specifies which of their inputs
  are per-thread (`thread_idx`, `nmp_min_ply`, histories) and which are
  shared flags.
- Time-management formulas (`optimum`/`maximum`, soft-stop factors) — the
  time-management doc. This doc only places `TimeManager` inside
  `SearchManager` and defines *when* `check_time` runs.
- UCI parsing, `Limits` construction, `info`/`bestmove` formatting — the UCI
  doc. This doc defines the handoff (`start_thinking(board, limits, tt)`).
- History table shapes and init/decay constants — the movepick/history doc.
  This doc only requires that `Worker::clear()` exists and that histories are
  plain (non-atomic) per-thread data.
- NUMA awareness — explicitly deferred (§3.15).

---

## 2. References

- `stockfish_reference/stockfish_search_notes.md` §13 (threading — lazy SMP),
  §13.2 (voting), §13.3 (time management interplay), §3.1 (aspiration jitter),
  §11.4 (TT SMP safety).
- `stockfish_reference/04-tt-search-structures.md` — threading data layout,
  relaxed-atomic TT mapping to Rust.
- Stockfish source (verified for this doc, read-only, D27):
  - `src/thread.h` — `Thread` (idle loop, per-thread mutex+condvar,
    `searching`/`exit` flags), `ThreadPool` (`stop`, `increaseDepth`
    atomics, `accumulate` over per-worker relaxed counters).
  - `src/thread.cpp` — constructor parks the thread before use;
    `run_custom_job` protocol; `start_thinking` sequence (lines 289-348);
    `get_best_thread` voting (lines 350-401); `clear()` resets
    `previousTimeReduction = 0.85` etc.
  - `src/search.h` / `search.cpp` — `Worker` field inventory,
    `ISearchManager`/`NullSearchManager` null-object, main-thread blocks in
    `iterative_deepening`, `check_time` cadence.
- `design/decisions.md` — D5, D6, D8, D13, D14, D15, D19, D20, D23, D24,
  D26, D27.

All numeric constants below are SF-current tuned values recorded as starting
points (D20); they are facts, not copied code.

---

## 3. Design

### 3.1 Architecture overview and the shared-state inventory

Lazy SMP: no tree splitting, no work queues, no synchronization during search.
Each thread runs the full iterative-deepening search on its own copy of the
root position. Divergence between threads comes from three sources only:

1. **Aspiration jitter**: initial half-window
   `delta = 5 + thread_idx % 8 + |mean_squared_score| / 10588` — the
   `thread_idx % 8` term makes threads search slightly different windows
   (formula owned by the search doc; the `thread_idx` plumbing is owned here).
2. **TT races**: threads read each other's discoveries through the shared TT
   and get cut/guided differently depending on arrival order.
3. **Per-thread histories**: history tables are never shared, so ordering
   diverges as searches diverge.

The **complete** shared-state inventory — everything two threads may touch
concurrently — is:

| Shared item | Type | Writers | Readers |
|---|---|---|---|
| Transposition table | `Arc<TT>` (all fields relaxed atomics, D19) | all workers | all workers |
| `stop` | `AtomicBool` | main worker, UCI thread | all workers (every node) |
| `increase_depth` | `AtomicBool` | main worker | all workers (per ID iteration) |
| `ponder` | `AtomicBool` | UCI thread (`ponderhit`/`stop`) | main worker |
| Per-thread `Counters` (nodes, tb_hits, best_move_changes) | `AtomicU64` ×3, cache-line aligned | owning worker (and main zeroes `best_move_changes`) | main worker, UCI thread |
| Per-thread `SearchReport` | `Mutex<Option<SearchReport>>` | owning worker (once, at search end) | main worker (voting, after helpers park) |
| Per-thread job slot | `Mutex<ThreadState>` + `Condvar` | pool + owning worker | pool + owning worker |

Everything else — `Board`, `RootMove` list, all history tables, the reduction
table, `nmp_min_ply`, the `Limits` copy, the search stack — is **single-owner
inside `Worker`** and never observable from another thread (D23). There is no
`Mutex` anywhere on the per-node path; the mutexes above are touched only at
job boundaries (a handful of times per `go`).

### 3.2 The `Worker` struct

One `Worker` per thread, **owned by its thread** (moved into the thread's
closure at spawn, never shared). Field inventory:

```rust
// src/threads.rs (search-algorithm fields elaborated in the search doc)

pub struct Worker {
    pub thread_idx: usize,              // 0 = main; feeds aspiration jitter

    // ---- Root state, refreshed by every Setup job ----
    pub root_board: Board,              // clone incl. StateInfo history (D8),
                                        //   so repetition detection sees the game
    pub root_moves: Vec<RootMove>,      // own scores/PVs; RootMove owned by search doc
    pub limits: Limits,                 // per-go copy (UCI doc owns the type)

    // ---- Per-search scalars ----
    pub root_depth: Depth,
    pub completed_depth: Depth,
    pub sel_depth: i32,
    pub nmp_min_ply: i32,               // null-move verification exclusion window
    pub optimism: [Value; Color::COUNT],

    // ---- Heavy single-owner tables ----
    pub histories: Box<Histories>,      // main/lowPly/capture/continuation/pawn/
                                        //   correction + ttMoveHistory scalar;
                                        //   ~20 MiB (doc 09 §3.4 owns the
                                        //   per-table breakdown) — boxed,
                                        //   persists across `go`, reset only
                                        //   by Worker::clear()
    pub reductions: Box<[i32; MAX_MOVES]>, // reductions[i] = (2834/128 · ln i) as i32
                                           //   (SF-current, D20); built in Worker::new

    // ---- Cross-thread-visible handles ----
    pub counters: Arc<Counters>,        // this thread's atomic counters (see §3.10)
    pub pool: Arc<PoolShared>,          // stop/increase_depth/ponder + all shells
                                        //   (main uses it to wake helpers & aggregate)

    // ---- Main-thread-only state (None for helpers) ----
    pub manager: Option<Box<SearchManager>>,
}
```

Notes:

- `Board::clone()` clones the ply-indexed `Vec<StateInfo>` history (D8), so
  each worker's repetition detection is self-contained. The clone happens once
  per `go` per thread — negligible.
- The reduction table is per-worker even though its contents are identical
  across threads. Rationale: 1 KB per thread is free, it matches SF, and it
  keeps the read NUMA-local if/when we bind threads later. Revisit only if
  someone objects to the duplication (§9).
- `histories` persist across `go` calls within a game (the search doc
  specifies the per-`go` decay/refresh); `Worker::clear()` — invoked by the
  `Clear` job on `ucinewgame` — resets them to their init values (history doc
  owns the fill constants).
- The counters are *handles* (`Arc`) rather than plain fields precisely so
  the pool can read them while the worker is exclusively borrowed by its
  thread (§3.10).

### 3.3 `SearchManager` and the null-object pattern

Main-thread-only responsibilities, gathered in one struct so the search code
stays branch-light:

- Time management: `TimeManager` init at `go`, soft-stop between iterations,
  hard-stop in `check_time`.
- `check_time` cadence: a `calls_cnt` countdown decremented on node entry;
  when it hits 0, reset to `if limits.nodes > 0 { min(512, limits.nodes / 1024) } else { 512 }`
  (SF-current) and perform the elapsed/node-limit checks.
- UCI output: `info depth ...` per completed PV, throttled `currmove` /
  extra-upperbound updates (after 10M nodes, per UCI doc), the final
  `bestmove`.
- Cross-move bookkeeping: `best_previous_score`,
  `best_previous_average_score`, `previous_time_reduction` (reset to `0.85`
  by `clear()`, SF-current), `original_time_adjust`, `stop_on_ponderhit`,
  `tot_best_move_changes` (halved each iteration).

```rust
pub struct SearchManager {
    pub tm: TimeManager,                    // timeman doc
    pub calls_cnt: i32,
    pub stop_on_ponderhit: bool,            // plain bool: main-worker-only
    pub best_previous_score: Value,
    pub best_previous_average_score: Value,
    pub previous_time_reduction: f64,
    pub original_time_adjust: f64,
    pub tot_best_move_changes: f64,
    pub first_root_info_sent: bool,         // output throttling state (UCI doc)
}
```

Rust's equivalent of SF's `ISearchManager`/`NullSearchManager` virtual pair is
`Option<Box<SearchManager>>` plus access helpers:

```rust
impl Worker {
    #[inline]
    fn is_main(&self) -> bool { self.manager.is_some() }

    /// Run `f` on the manager iff this is the main thread; no-op otherwise.
    #[inline]
    fn with_manager(&mut self, f: impl FnOnce(&mut SearchManager, /* rest of */ &mut WorkerCtx)) { ... }
}
```

Why `Option` and not a trait object: a `None` check is a single perfectly
predicted branch and the compiler can see through it; a `dyn` call is an
indirect call it cannot. The search code contains a small number of
`if let Some(m) = &mut self.manager` blocks exactly where SF has
`if (mainThread)` blocks (ID-loop bookkeeping, `check_time`, info output) —
nowhere on the per-move hot path. Helpers pay one boxed-pointer of storage
and zero per-node cost.

Note an improvement over SF made possible by our job protocol: SF's UCI thread
writes `stopOnPonderhit` and `ponder` into the main worker's manager from
outside. We instead reset `stop_on_ponderhit` inside the `Setup` job (which
runs *on* the worker thread), so `SearchManager` is never touched
cross-thread and needs no atomics; only `ponder` stays shared (the UCI thread
must flip it mid-search on `ponderhit`), and it lives in `PoolShared`.

### 3.4 Thread lifecycle: persistent parked threads (chosen) vs `std::thread::scope`

**Decision: long-lived threads with a per-thread job loop (Mutex + Condvar),
exactly SF's lifecycle.** Threads are created once per `setoption Threads` /
engine start, park in an idle loop, and are woken with jobs. They die only on
`quit` or when the pool is resized.

This is D23's pinned mechanism (decisions.md amended 2026-07-09: persistent
parked threads + `Arc<TT>`, interior-mutable TT generation, resize/clear via
`Arc::get_mut` between searches), keeping D23's substance (per-thread
`Worker` owns everything; shared state is exactly the TT, atomic flags, and
per-thread atomic counters; no locks on the search path).

Justification:

1. **UCI requires `go` to be non-blocking.** The UCI thread must return to
   reading stdin immediately (to receive `stop`/`ponderhit`), so *someone*
   other than the UCI thread must run the search. With persistent threads,
   `start_thinking` just wakes the main worker and returns. A scoped design
   would need to spawn a fresh "search master" thread per `go` to host the
   scope — so it doesn't even remove the cross-'go' thread machinery, it just
   re-creates it every move.
2. **Spawn latency.** Thread creation costs tens of microseconds per thread
   plus first-touch page faults; at bullet/hyperbullet controls
   (`movetime 10`-class searches during SPRT, D25) spawning 8+ threads per
   move is measurable noise. Parked threads wake in ~1 µs.
3. **State residency.** Histories (megabytes per thread) persist across moves
   within a game by design; with persistent threads they also stay
   cache/TLB-warm and, later, NUMA-local (a worker's tables are allocated and
   cleared by the thread that uses them — first-touch placement for free).
4. **Generality.** The same job loop later serves parallel TT clear and any
   other "run this on thread i" need (SF's `run_custom_job`), without new
   machinery.

**The documented alternative** — `std::thread::scope`, the mechanism D23
literally names: the pool owns `Vec<Worker>`; each `go` spawns a master
thread which opens a scope, spawns N-1 helpers borrowing `&mut workers[i]`
and `&TT` directly (no `Arc` at all, lifetimes prove everything), and joins
them at scope exit. It is genuinely
idiomatic and *simpler to prove correct* (scope join is a compiler-checked
happens-before). We reject it for the reasons above, but it remains the
fallback if the job loop ever fights the borrow checker in an ugly way: the
Worker/shared-state split specified here works unchanged under scope — only
`Arc<PoolShared>` handles become plain references.

**The job loop.** Per-thread shell and protocol (compare SF `Thread::idle_loop`
/ `run_custom_job`, reimplemented, not copied):

```rust
pub enum Job {
    Setup(Box<SetupJob>),        // install per-go state (runs on the worker's thread)
    Search { tt: Arc<TT> },      // run Worker::start_searching(&tt)
    Clear,                       // Worker::clear() — ucinewgame
    Exit,                        // leave the loop; thread terminates
}

pub struct SetupJob {
    pub board: Board,            // clone with game history (D8)
    pub root_moves: Vec<RootMove>,
    pub limits: Limits,
}

struct ThreadState { job: Option<Job>, busy: bool }

pub struct ThreadShell {
    state: Mutex<ThreadState>,
    cv: Condvar,                          // one condvar, both directions
    pub counters: Counters,               // §3.10, 64-byte aligned
    pub report: Mutex<Option<SearchReport>>, // §3.8
}
```

```rust
// Runs on the spawned thread. Worker is constructed HERE (first-touch local).
fn thread_main(idx: usize, pool: Arc<PoolShared>) {
    let mut worker = Worker::new(idx, pool.clone());
    let shell = &pool.shells[idx];
    loop {
        let job = {
            let mut st = shell.state.lock().unwrap();
            st.busy = false;
            shell.cv.notify_all();                    // wake wait_idle() callers
            while st.job.is_none() {
                st = shell.cv.wait(st).unwrap();
            }
            st.job.take().unwrap()                    // busy stays true (set by submitter)
        };
        match job {
            Job::Exit => return,
            Job::Setup(s) => worker.setup(*s),
            Job::Search { tt } => worker.start_searching(&tt),
            Job::Clear => worker.clear(),
        }
    }
}

impl ThreadShell {
    /// Hand a job to this thread; blocks until it is idle first.
    pub fn submit(&self, job: Job) {
        let mut st = self.state.lock().unwrap();
        while st.busy || st.job.is_some() {
            st = self.cv.wait(st).unwrap();
        }
        st.job = Some(job);
        st.busy = true;
        self.cv.notify_all();
    }

    /// Block until this thread has finished its current job (if any).
    pub fn wait_idle(&self) {
        let mut st = self.state.lock().unwrap();
        while st.busy || st.job.is_some() {
            st = self.cv.wait(st).unwrap();
        }
    }
}
```

Protocol invariants (mirror SF's `searching` flag semantics):

- `busy == true` from submission until the worker re-parks; `wait_idle`
  observes the *completion* of the job, and the mutex hand-off gives the
  happens-before edge that publishes everything the job wrote (reports,
  root-move scores) to whoever waited.
- A job is only ever submitted to an idle thread (`submit` waits), so the
  `Option<Job>` slot never holds two jobs.
- **Stack size**: search recursion reaches `MAX_PLY` with fat frames; spawn
  via `std::thread::Builder::new().stack_size(8 * 1024 * 1024)` (SF forces
  8 MiB likewise). The main *process* thread only runs UCI parsing.

`Job::Search` carries `Arc<TT>` **per job** rather than the worker storing it
permanently. This makes `setoption name Hash` trivially safe: when all
threads are idle, no `Arc<TT>` clone exists outside the UCI thread, so
`Arc::get_mut` yields `&mut TT` for resize/reallocation with zero locking on
the probe path (the TT doc owns resize itself; this doc guarantees the
exclusivity window). Cost: one atomic refcount bump per thread per `go` —
nothing.

### 3.5 `ThreadPool` and `PoolShared`

```rust
pub struct PoolShared {
    pub shells: Vec<Arc<ThreadShell>>,   // index == thread_idx; 0 = main worker
    pub stop: AtomicBool,
    pub increase_depth: AtomicBool,
    pub ponder: AtomicBool,
}

pub struct ThreadPool {
    shared: Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// (Re)create the pool with `n >= 1` threads. Waits for any running
    /// search, exits and joins old threads first.
    pub fn set(&mut self, n: usize);

    /// UCI `go`. Non-blocking: returns once the main worker is woken.
    pub fn start_thinking(&mut self, board: &Board, limits: Limits, tt: &Arc<TT>);

    pub fn stop(&self)          { self.shared.stop.store(true, Ordering::Relaxed); }
    pub fn ponderhit(&self)     { self.shared.ponder.store(false, Ordering::Relaxed); }
    pub fn wait_for_search_finished(&self);   // wait_idle on the MAIN thread only
    pub fn clear(&self);        // ucinewgame: Job::Clear to every thread, wait all

    pub fn nodes_searched(&self) -> u64;      // Σ shells[i].counters.nodes
    pub fn tb_hits(&self) -> u64;
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // stop → wait main idle → submit Job::Exit to all → join all.
    }
}
```

`ThreadPool` is owned by the UCI thread. The main worker (thread 0) reaches
its siblings through its own `Arc<PoolShared>` clone — that is how it wakes
helpers, waits for them, aggregates counters, and reads reports. Only thread 0
ever exercises those powers; helpers hold the same `Arc` and simply never use
it (enforced by the `manager: None` gate, same as SF).

`wait_for_search_finished` waits **only on the main worker**: the main
worker's own epilogue (§3.7 step E) already waits for all helpers, so main
being idle implies everyone is idle. This matches SF and keeps `submit`'s
"only to idle threads" invariant trivial.

### 3.6 The `start_thinking` sequence (UCI thread)

Verbatim orchestration order (SF `thread.cpp:289-348`, re-expressed):

1. **Wait for the previous search**: `shells[0].wait_idle()`. (GUIs send
   `stop` first; if they don't, we block — same as SF.)
2. **Clear control flags**: `stop.store(false)`, `increase_depth.store(true)`,
   `ponder.store(limits.ponder)`.
3. **Build the root move list once**: generate legal moves on `board`
   (movegen doc); if `limits.searchmoves` is non-empty, keep exactly the
   legal ones, *in searchmoves order*; else all legal moves in generation
   order. Wrap each as `RootMove::new(mv)` (score `-VALUE_INFINITE`,
   `previous_score = -VALUE_INFINITE`, empty PV seeded with `mv`). The same
   list (same order) is cloned to every thread — divergence comes from
   search, not from root shuffling.
4. **Per-thread setup**: for every `i`, `shells[i].submit(Job::Setup(...))`
   with clones of the board, root moves, and limits. `Worker::setup` (on the
   worker's own thread) installs them and resets:
   `nodes/tb_hits/best_move_changes → 0` (relaxed stores),
   `nmp_min_ply = 0`, `root_depth = completed_depth = 0`, `sel_depth = 0`,
   `report = None`; on the main worker additionally
   `manager.stop_on_ponderhit = false`, `manager.calls_cnt = 0`.
5. **Wait for all setups**: `wait_idle` on every shell.
6. **Wake the main thread only**: `shells[0].submit(Job::Search { tt })`.
   Return to the UCI read loop immediately.

Helpers are *not* woken here — the main worker wakes them (next section)
**after** it has bumped the TT generation and initialized time management, so
no helper can ever observe a stale generation or race the TM init.

### 3.7 Main worker orchestration (`Worker::start_searching`, thread 0)

A. **No legal moves**: if `root_moves.is_empty()`, skip to step D with a
   synthetic `mate 0 / stalemate` info line; helpers are never woken.
B. **Init**: `manager.tm.init(&limits, side_to_move, game_ply)` (timeman doc);
   `tt.new_search()` — the **single** per-`go` generation bump (D19).
   The main worker only ever holds the shared handle (`Job::Search { tt:
   Arc<TT> }` → `start_searching(&tt)`), and the UCI thread and pool retain
   `Arc` clones throughout, so a `&mut self` bump is unobtainable here.
   `new_search` therefore **must take `&self`**: the TT doc's generation
   counter is interior-mutable — a relaxed `AtomicU8` store here, a relaxed
   load in `tt.generation()` (already read once per node, so zero added
   hot-path cost). The step-C `submit` mutex hand-off publishes the bump, so
   no helper can observe a stale generation.
C. **Wake helpers**: for `i in 1..n`, `shells[i].submit(Job::Search { tt })`
   — each helper runs the identical `iterative_deepening()`; then main runs
   `iterative_deepening()` itself.
D. **Never answer early**: after ID returns, busy-wait while
   `!stop && (ponder || limits.infinite)`:

   ```rust
   while !pool.stop.load(Ordering::Relaxed)
       && (pool.ponder.load(Ordering::Relaxed) || self.limits.infinite)
   {
       std::thread::yield_now();   // deliberate busy wait, as in SF
   }
   ```

   Two ways out: UCI `stop` sets the flag, or `ponderhit` clears `ponder`
   and the loop falls through (the budget was already consumed, so we answer
   immediately — the `ponderhit`-during-ID case, where the search keeps
   going on the now-live clock, is handled by `check_time` +
   `stop_on_ponderhit` inside ID and never reaches this loop still
   pondering).
E. **Stop and collect**: `pool.stop.store(true, Ordering::Relaxed)`; for
   `i in 1..n`, `shells[i].wait_idle()` — every worker (main included) wrote
   its `SearchReport` into `shell.report` as the last act of its search job.
F. **Vote** (§3.8) → winning report.
G. **Emit**: if the winner differs from main's own last output, print one
   final `info` line from the winner's report; then
   `bestmove <pv[0]> [ponder <pv[1]>]`. If the winner's PV has length 1, try
   TT ponder extraction: play `pv[0]` on a scratch board, probe the TT,
   validate the stored move `pseudo_legal && legal` (D19 discipline), and use
   it as the ponder move if valid. Update `manager.best_previous_score` /
   `best_previous_average_score` from the winner for the next move's TM.

Helper behavior during the search: helpers run ID until they observe `stop`;
they ignore `limits.depth` (main-only, per SF — with voting disabled under
`go depth`, their extra depth is harmless) and never evaluate time or node
limits themselves — the node limit is enforced by the main thread's
`check_time` against the *aggregated* count (§3.10).

`increase_depth`: written by main between iterations
(`increase_depth = ponder || elapsed <= total_time * 0.50`, constant owned by
the timeman doc); read by **every** worker's ID loop — when false, a worker
increments its private `search_again_counter` instead of deepening
effectively (`adjusted_depth` shrinks; search doc owns the formula). That is
why it is a shared atomic and not manager state.

### 3.8 Best-thread voting

Inputs: one `SearchReport` per thread, published under `shell.report` before
parking:

```rust
pub struct SearchReport {
    pub best: RootMove,        // rootMoves[0]: score, bounds, full PV
    pub completed_depth: Depth,
    pub sel_depth: i32,
}
```

Algorithm (SF `get_best_thread`, re-expressed; constants SF-current, D20):

1. **Skip conditions**: voting is disabled and thread 0's report wins
   directly when `n_threads == 1` **or** `limits.depth` is set (fixed-depth
   bench must be reproducible, D24).
2. `min_score = min over threads of report.best.score`.
3. Each thread votes for its own best move:
   `votes[report.best.pv[0]] += (report.best.score - min_score + 14)`
   — the `+14` tempo term means even the weakest thread's opinion counts;
   identical moves accumulate. (`votes: HashMap<Move, i64>` — heap use is
   fine here, we are strictly after the search.)
4. Scan threads with `best = 0` as the incumbent; for each candidate `i`:
   - A report is **decisive** iff
     `score != -VALUE_INFINITE && is_decisive(score) && !score_is_bound`
     (an aborted iteration may leave a bound-flagged win/loss — those are
     untrusted; `is_decisive` is the TB-win-band predicate from the value
     layer, D5).
   - If the incumbent is decisive: only a decisive candidate with strictly
     larger `|score|` displaces it — i.e. only a **shorter mate** (or faster
     TB win / longer mated-in defense) wins; vote counts are ignored
     entirely.
   - Otherwise the candidate wins if it is decisive, **or** if its score is
     not a loss (`!is_loss(score)` — this also excludes helpers that never
     finished depth 1, whose score is `-VALUE_INFINITE`) and it has strictly
     more votes, **or** equal votes and a **longer PV** (depth proxy
     tiebreak).

```rust
fn best_thread(reports: &[SearchReport]) -> usize {
    debug_assert!(reports.len() > 1);
    let min_score = reports.iter().map(|r| r.best.score).min().unwrap();
    let mut votes: HashMap<Move, i64> = HashMap::new();
    for r in reports {
        *votes.entry(r.best.pv[0]).or_insert(0) +=
            (r.best.score - min_score + VOTE_TEMPO) as i64;   // VOTE_TEMPO = 14
    }
    let mut best = 0;
    for i in 1..reports.len() {
        let (b, c) = (&reports[best].best, &reports[i].best);
        let b_dec = is_trusted_decisive(b);
        let c_dec = is_trusted_decisive(c);
        if b_dec {
            if c_dec && c.score.abs() > b.score.abs() { best = i; }
        } else if c_dec
            || (!is_loss(c.score)
                && (votes[&c.pv[0]] > votes[&b.pv[0]]
                    || (votes[&c.pv[0]] == votes[&b.pv[0]]
                        && c.pv.len() > b.pv.len())))
        { best = i; }
    }
    best
}
```

`VOTE_TEMPO = 14` lives in `search/params.rs` (D20). SF additionally disables
voting under Skill Level; we have no skill feature.

### 3.9 Search diversity — what we deliberately do NOT synchronize

For the record, because it is tempting to "fix": threads do not share
histories, do not partition root moves, do not communicate scores during the
search, and do not synchronize iteration depths. SF experiments show the
naive-looking full duplication plus TT osmosis beats coordinated schemes.
The only inter-thread influence during search is the TT and the two flags.

### 3.10 Node / tbhit / bestMoveChanges aggregation

Per-thread counters, cache-line-isolated so helpers never false-share:

```rust
#[repr(align(64))]
pub struct Counters {
    pub nodes: AtomicU64,
    pub tb_hits: AtomicU64,
    pub best_move_changes: AtomicU64,
}
```

- **Increment (owner only)**: single-writer, so the cheap non-RMW form is
  correct and preferred:
  `c.nodes.store(c.nodes.load(Relaxed) + 1, Relaxed)` — plain load+store, no
  `lock xadd`; wrapped as `Counters::bump_nodes(&self)`. `nodes` increments
  in `make_move`-per-node position entry (search doc), `tb_hits` on syzygy
  probes (later), `best_move_changes` when a non-first root move becomes
  best.
- **Aggregate (any thread)**:
  `pool.nodes_searched() = Σ shells[i].counters.nodes.load(Relaxed)` — used
  by the main worker's `check_time` (node limit), by `info ... nodes/nps`
  lines, and by `bench`. Sums are approximate while threads run (relaxed,
  by design) and exact after `wait_idle` (mutex ordering publishes final
  values).
- **`best_move_changes` harvest**: once per ID iteration the main worker
  executes `tot_best_move_changes = tot_best_move_changes / 2 + Σ_i
  shells[i].counters.best_move_changes.swap(0, Relaxed)` and feeds the
  time-manager instability factor
  (`bestMoveInstability = 1.096 + 2.29 · totBestMoveChanges / n_threads`,
  constants owned by the timeman doc). This is the one place main *writes*
  another thread's counter; `swap(0)` is atomic so no increments are lost,
  and staleness is harmless (it feeds a heuristic).

### 3.11 Stop flag and memory-ordering discipline

- `stop` is checked by every worker **once per node** (a relaxed `AtomicBool`
  load of a cache line that is shared-read-only until the single store —
  effectively free) and inside the aspiration and ID loops. On observing
  `stop`, a node returns immediately (unwinding with untrusted `VALUE_ZERO`
  scores; the search doc owns the aborted-score hygiene).
- Setters: main worker (`check_time` hard limit, soft-stop between
  iterations, `mate`/`nodes` limits) and the UCI thread (`stop` command,
  pool drop).
- **Ordering: `Relaxed` everywhere for all three flags.** Justification: the
  flags carry no data — they only *request* behavior (stop deepening / stop
  searching / keep waiting). Every actual data handoff (root move scores,
  reports, counters' final values) is published by the `ThreadShell` mutex
  hand-off in `wait_idle`, which is a full happens-before edge. A stale flag
  read costs at most a few extra nodes, never correctness. (SF uses relaxed
  loads on the hot path and defaulted stores; we make both explicit.)
- **Latency budget**: from `stop.store(true)` to all threads parked must be
  well under time-management noise — target < 5 ms typical, < 50 ms hard
  (§5). A node takes ~1 µs, so the per-node check bounds latency by the
  deepest unwind path.

### 3.12 Send/Sync analysis (exhaustive)

Rule zero: **no `unsafe impl Send/Sync` anywhere in this layer** (D14 — TT
atomics are safe Rust; threading adds nothing unsafe). If a type doesn't
auto-derive the marker we need, the design is wrong, not the compiler.

| Type | Send | Sync | Why / role |
|---|---|---|---|
| `TT` | auto | auto — every field is `AtomicU16/U8/I16` in clusters, plus the `AtomicU8` generation counter (§3.7 B); interior mutability via atomics is `Sync` by construction (D19) | shared via `Arc<TT>`; the *only* shared mutable search data |
| `PoolShared` | auto | auto — `Vec<Arc<ThreadShell>>` (never mutated after build) + three `AtomicBool` | one `Arc` per thread + one in the pool |
| `ThreadShell` | auto | auto — `Mutex`, `Condvar`, `Counters` (atomics), `Mutex<Option<SearchReport>>` | job slot + published results |
| `Counters` | auto | auto — three `AtomicU64` | cross-thread reads while owner runs |
| `SearchReport` | auto (`Vec<Move>`, plain ints) | n/a — only ever behind the `Mutex` | end-of-search handoff |
| `Worker` | auto — `Board` (arrays + `Vec<StateInfo>`), `Vec<RootMove>`, `Box<Histories>` (plain `i16/i32` arrays), `Arc<_>` handles (`Send+Sync` targets), `Option<Box<SearchManager>>` | **not required and not relied upon** — a `Worker` is moved into its thread once and never shared | the single-owner container |
| `SearchManager` | auto — `TimeManager` (plain data + `Instant`), scalars | n/a — main-worker-only | null-object payload |
| `Job` / `SetupJob` | auto — `Board`/`Vec`/`Limits`/`Arc<TT>` | n/a — moved through the mutex | job payloads |
| `Board` | auto (D8: no pointers, history is an owned `Vec`) | n/a | cloned per worker |

Forbidden types anywhere in `threads`/`search`/`tt` module graphs: `Rc`,
`RefCell`, `Cell`, raw pointers, `thread_local!` for search state. This is
compiler-enforced the moment a `Worker` moves into `thread::spawn` (`Rc`
would fail `Send`) and additionally pinned by a test:

```rust
#[cfg(test)]
mod marker_asserts {
    use super::*;
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    #[test]
    fn shared_types_are_sync() {
        sync::<TT>(); sync::<PoolShared>(); sync::<ThreadShell>(); sync::<Counters>();
        send::<Worker>(); send::<Job>(); send::<SearchReport>();
    }
}
```

### 3.13 Races: accepted vs forbidden

**Accepted (by design, value-level only — never data races in the memory
model):**

1. **TT field races** (D19): two threads write the same entry's fields
   concurrently; a reader may see a *torn entry* — value from one write, move
   from another — or a 16-bit tag collision. Every field access is a relaxed
   atomic, so this is defined behavior; every consumer validates
   (`pseudo_legal && legal` on ttMove, `is_valid` on values, depth clamps).
   ThreadSanitizer sees atomics, so these are TSan-clean.
2. **Stale flag reads** (`stop`, `increase_depth`, `ponder`): a thread may
   search a few extra nodes after `stop` is set. Bounded, harmless.
3. **Approximate counter sums**: `nodes_searched()` mid-search can undercount
   in-flight increments; `best_move_changes.swap(0)` may race an increment
   into the *next* harvest. Both feed heuristics/reporting only.

**Forbidden (must be impossible by construction, and TSan-verified):**

- Any non-atomic shared mutation: histories, root moves, boards, the search
  stack, `SearchManager`, reduction tables are single-owner — a second
  thread must have *no path* to them, not merely a promise not to use it.
- Reading another worker's `RootMove` scores/PVs during the search (SF's
  layout technically allows main to peek; ours does not — the only window is
  the post-`wait_idle` `SearchReport`).
- Torn `u64` counter updates from two writers (single-writer discipline; the
  one exception, `best_move_changes`, uses atomic `swap`).
- Any lock acquisition on the per-node path.

The crisp invariant: **ThreadSanitizer must be 100% clean.** Because even our
"accepted" races go through atomics, anything TSan flags is a real bug, with
no allowlist to maintain.

### 3.14 Bench determinism (D24)

`bench` (bench doc) runs the fixed position suite through this exact
`start_thinking` path with `Threads=1`, a fixed Hash, and `go depth N`:

- one thread ⇒ no helpers woken, no TT races, no jitter (thread 0's
  `thread_idx % 8 == 0`);
- `limits.depth` set ⇒ voting disabled (§3.8 step 1);
- fixed depth ⇒ `check_time` never fires a stop; draw dither depends only on
  the deterministic node counter.

Therefore the total node count is bit-reproducible run-to-run — the
regression oracle. Multi-threaded bench is *not* deterministic (TT arrival
order) and is used only as a sanity/hang check, never as a functional gate.

### 3.15 NUMA — explicitly deferred

Out of scope for v1: thread-to-node binding, per-node shared histories,
NUMA-replicated NNUE weights, node-aware TT clear. The design keeps the door
open at zero cost: workers already allocate and clear their own tables on
their own thread (first-touch), the reduction table is per-worker, and the
job loop can carry a future `Job::Custom` for node-pinned work. Nothing else
in this doc changes when NUMA lands.

---

## 4. Implementation plan

Ordered, commit-sized; each step keeps `cargo test` and `cargo clippy
--all-targets` green (CLAUDE.md).

1. **`threads` module skeleton**: `Counters`, `ThreadState`, `ThreadShell`,
   `Job` (with `Setup`/`Search`/`Clear`/`Exit`), `PoolShared`, `thread_main`
   idle loop, `ThreadPool::{set, drop}` with 8 MiB stacks.
   *Done when*: unit tests prove park→submit→run→park and clean `Exit`/join
   for n ∈ {1, 4, 16}; marker-trait test passes; pool drop never hangs
   (test under a timeout).
2. **`Worker` container + `SearchManager` null-object**: move the existing
   single-threaded search state (search doc) into `Worker`; add
   `manager: Option<Box<SearchManager>>` and convert main-only code to
   `if let` blocks; add `Worker::{new, setup, clear}`.
   *Done when*: single-threaded search runs entirely through
   `Worker::start_searching`; bench node count unchanged from the pre-refactor
   baseline (D24 non-functional gate).
3. **`start_thinking` at `Threads=1`**: wire UCI `go` → pool (steps §3.6),
   main-worker orchestration (§3.7) minus helper wake, `stop` command,
   `wait_for_search_finished`, `bestmove` emission moved into the main
   worker.
   *Done when*: full UCI session (position/go/stop/quit) works; `go infinite`
   + `stop` returns a bestmove; bench node count still identical.
4. **Helpers**: wake/wait in §3.7 C/E, `Arc<TT>` handed per Search job,
   counter aggregation (`nodes_searched`, `info nps`), `best_move_changes`
   harvest.
   *Done when*: `Threads=4` searches finish and print legal bestmoves over a
   500-search smoke loop with zero hangs; `Threads=1` bench unchanged.
5. **Voting**: `SearchReport` publication, `best_thread` with decisive
   protection and tiebreaks, disabled under `go depth` / single thread;
   `VOTE_TEMPO` into `search/params.rs`.
   *Done when*: the §5 voting unit-test matrix passes; multi-thread `go
   depth` always answers with thread 0's move.
6. **Ponder + infinite discipline**: `ponder` flag, `ponderhit` handler,
   `stop_on_ponderhit` inside the manager, busy-wait loop, TT ponder-move
   extraction.
   *Done when*: `go ponder` → `ponderhit` continues and answers; `go ponder`
   → `stop` answers immediately with the current best; never emits
   `bestmove` while ponder/infinite are live.
7. **Race hardening pass**: TSan stress run (below), stop-latency test,
   docs of measured numbers in the commit message.
   *Done when*: acceptance criteria §8 all check.

---

## 5. Testing and validation

**Unit tests (in `src/threads.rs`):**

- Job-loop protocol: submit to idle thread runs the job exactly once;
  `wait_idle` returns only after completion (job flips a value read after
  the wait); `Exit` terminates; double-`set` recreates the pool.
- Setup propagation: a `Setup` job installs board/limits/root list; a
  position with history clones repetition state (make a 2-fold at the UCI
  level, verify the worker's board sees it).
- Counters: single-writer bump + cross-thread aggregate equals expected;
  `best_move_changes` `swap` harvest loses no counts under a hammering
  writer thread.
- Voting matrix (pure function, synthetic reports):
  (a) unanimous → that move; (b) 3 threads × move A at score 10 vs 1 thread
  × move B at score 30 → A (votes accumulate); (c) incumbent mate-in-5 vs
  candidate mate-in-3 → candidate (larger `|score|`); (d) incumbent
  mate-in-3 vs candidate with huge votes but normal score → incumbent
  (decisive protection); (e) bound-flagged win does not count as decisive;
  (f) equal votes → longer PV wins; (g) `-VALUE_INFINITE` helper never wins;
  (h) `limits.depth` set → thread 0 regardless.
- Marker-trait assertions (§3.12).

**Integration gates:**

- **Bench determinism (D24)**: `bench` at `Threads=1` twice → identical total
  node counts; this is the permanent regression oracle for every future
  threading change ("non-functional" = node count unchanged).
- **Multi-thread sanity**: `bench` at `Threads=4` completes (node count not
  asserted), bestmoves legal; repeated 500× short-`movetime` loop with
  interleaved `stop`/`isready`/new `position` — no hang, no panic (run under
  a global timeout).
- **Stop-latency test**: start `go infinite` on a middlegame position with
  N=cores threads; sleep 100 ms; `stop()`; measure until
  `wait_for_search_finished` returns. Assert < 50 ms (target < 5 ms); record
  the typical number.
- **Ponder tests** as in plan step 6.

**Sanitizers / model checking:**

- **ThreadSanitizer**: nightly-only run,
  `RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test --target x86_64-unknown-linux-gnu`
  plus the 500-search stress binary. Must be clean — our accepted races are
  atomic-mediated and invisible to TSan (§3.13). Not in default CI (nightly +
  slow); run before any threading-touching merge.
- **Miri**: whole-engine runs are impractical (interpreter speed), and this
  layer has no `unsafe` for Miri to catch; still, `cargo miri test
  threads::` on the job-loop unit tests is cheap and catches protocol UB if
  `unsafe` ever creeps in nearby. Optional.
- **loom** (dev-dependency, allowed by D15): a loom model of
  `submit`/`wait_idle`/`thread_main` exhaustively checks the condvar protocol
  for lost wakeups/deadlocks. Small, high value for the one hand-rolled
  concurrency primitive we have. Recommended in plan step 1.

**Strength validation (D25)**: threading claims (e.g., "8 threads gains X
Elo over 1 at fixed time") come only from fastchess/cutechess games, never
from nps.

---

## 6. Performance notes

**Targets:**

- Per-thread nps retention ≥ 90% at 8 threads vs 1 thread on the bench suite
  (i.e., `nps(8) ≥ 0.9 × 8 × nps(1)` on a machine with ≥ 8 physical cores).
  Lazy SMP has no algorithmic sync; retention losses point at false sharing
  or memory bandwidth, not design.
- Zero mutex operations during search (verify: `perf record` shows no
  `pthread_mutex` frames inside the search call tree).
- Stop latency < 5 ms typical (§5).

**Pitfalls to design against (and measure):**

- **False sharing**: `Counters` is `#[repr(align(64))]`; shells are separate
  `Arc` allocations. If profiling shows ping-ponging on the shell line
  (counters + mutex share a shell), split `Counters` into its own `Arc`.
- **RMW on counters**: use the single-writer load+store bump, not
  `fetch_add` (saves a locked op per node ≈ 1-2% at high nps).
- **`stop` line contention**: it's read-shared and written once — MESI keeps
  it in every L1 in shared state; no action needed, but don't co-locate it
  with anything frequently written (it lives with the other two rarely
  written flags only).
- **TT contention is the real scaling wall**: hashfull and memory bandwidth
  dominate multi-thread behavior — owned by the TT doc (huge pages), but
  measured here: track `nps(N)/N` and `hashfull` while scaling N.
- **Arc indirection**: `worker.counters` and `worker.pool` are one pointer
  deref on hot paths (node bump, stop check) — same cost as SF's member
  access through `this`; do not "optimize" into raw pointers (D14).
- **What to measure per change**: bench nps at 1 thread (regression),
  nps scaling at 2/4/8, stop latency, time-to-move jitter at
  `movetime 100`.

---

## 7. Interactions

Sibling docs are referenced by topic slug (`design/NN-<slug>.md`, numbering
per the suite index):

- **Depends on:**
  - `*-tt.md` — `TT` type (`Sync` via atomics, D19), `tt.new_search()`
    (must be `&self` — relaxed `AtomicU8` generation, §3.7 B),
    resize-when-idle contract (§3.4), prefetch.
  - `*-search*.md` — `Worker`'s search fields and `iterative_deepening`;
    `RootMove`; aspiration jitter consuming `thread_idx`; aborted-score
    hygiene on `stop`; `search_again_counter` consuming `increase_depth`.
  - `*-timeman*.md` — `TimeManager` inside `SearchManager`; `check_time`
    hard stop; soft-stop factors consuming `tot_best_move_changes`;
    `increase_depth` write rule.
  - `*-uci*.md` — `Limits` type, `go`/`stop`/`ponderhit`/`ucinewgame`/
    `setoption Threads|Hash` handlers calling into `ThreadPool`; `info`
    formatting used by the manager.
  - `*-movepick*.md` / `*-history*.md` — `Histories` aggregate,
    `Worker::clear()` init values, per-`go` decay.
  - `*-movegen*.md` — legal root-move generation for §3.6 step 3.
- **Feeds:**
  - `*-bench.md` — bench runs through `start_thinking` at `Threads=1`
    (§3.14) and reads `nodes_searched()`.
  - Future NUMA work (§3.15) and parallel TT clear (`Job::Custom`).

---

## 8. Acceptance criteria (D26)

The threading component is done when all of the following hold:

1. `Threads=1` bench node count is bit-identical across runs and identical
   to the pre-threading single-thread baseline (the refactor into
   `Worker`/pool is non-functional, D24).
2. `Threads=N` (N = 2, 4, 8): 500 consecutive short searches with random
   interleavings of `stop`, `isready`, `position`, `go ponder`/`ponderhit`
   complete with legal bestmoves, no hangs, no panics.
3. Voting unit-test matrix (§5) passes, including decisive-score protection
   and PV-length tiebreak; `go depth` multi-thread always returns thread 0's
   move.
4. Stop latency measured < 50 ms hard bound (< 5 ms typical) at N=cores.
5. ThreadSanitizer stress run is clean (zero reports).
6. `perf` profile of a 8-thread search shows no lock/condvar symbols inside
   the search call tree, and nps retention ≥ 90% per thread at 8 threads on
   the reference machine.
7. Pool lifecycle: `setoption name Threads` resizes with all state cleanly
   rebuilt; `setoption name Hash` succeeds via `Arc::get_mut` while idle;
   `quit` and `ThreadPool::drop` join all threads promptly.
8. No `unsafe` in the `threads` module; marker-trait tests compile; grep for
   `Rc<`/`RefCell` in `threads`/`search`/`tt` returns nothing.
9. A fixed-time SPRT (D25) of `Threads=4` vs `Threads=1` shows the expected
   large positive Elo gap (sanity that SMP actually helps; exact bounds set
   at test time).

---

## 9. Open questions

1. **RESOLVED (2026-07-09): D23 amended to this doc's mechanism.**
   decisions.md D23 now pins long-lived parked threads (`Mutex<Job>` +
   `Condvar`) with `Arc<TT>`, interior-mutable TT generation
   (`new_search(&self)` over `AtomicU8`, see doc 08 §3.9), and `setoption
   name Hash` resize via `Arc::get_mut` while all threads are idle.
   Doc 08 §3.9 has been updated to match; doc 01 §3.3.16 alignment is
   tracked in that doc's review findings.
2. **Sibling doc filenames**: this doc references peers by topic slug
   (`*-tt.md`, etc.); reconcile to exact filenames once the suite index is
   frozen. No design impact.
3. **Reduction table placement**: per-worker (chosen, SF-parity, NUMA-ready)
   vs a single `LazyLock` global (saves 1 KB/thread). Revisit only if a
   profile ever cares; the search doc should read it via
   `worker.reductions`, keeping the swap trivial.
4. **loom in dev-dependencies**: recommended for the job-loop model (§5) —
   confirm we accept the dev-dep (allowed by D15) or drop to hand-written
   stress tests only.
5. **Busy-wait during ponder/infinite** (§3.7 D): chosen for SF parity and
   simplicity with `yield_now`; if idle-CPU during long ponders becomes a
   complaint, replace with a condvar park woken by `stop()`/`ponderhit()` —
   contained change, same semantics.
6. **`Job::Custom(Box<dyn FnOnce(&mut Worker) + Send>)`**: deferred until
   parallel TT clear / NUMA needs it; the enum keeps the variant slot open.
