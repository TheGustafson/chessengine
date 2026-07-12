# Integral: Lazy SMP + NUMA — the "scales very well" standout

Source: `integral_reference/integral/src/engine/search/search.{h,cc}`
(`Thread`, `Searcher`), `integral_reference/integral/src/utils/barrier.h`
(`Barrier`), `integral_reference/integral/src/engine/search/transpo.cc`
(parallel `Clear`), `integral_reference/integral/src/engine/uci/uci.cc`
(command wiring), plus the full commit history of those paths
(`git -C integral_reference/integral log --oneline --all -- src/engine/search
src/utils/barrier.h`). Version studied: the checked-out clone, HEAD at 669
commits (dated 2023-10-25 .. 2026-04-02). **No license file in the repo →
all rights reserved.** Everything below is prose/pseudocode written from
scratch after reading the source and history; no Integral code is
reproduced — file:line and commit-hash citations are pointers for a future
reader to re-find the original, not excerpts.

Read together with `stockfish_reference/stockfish_search_notes.md` §13,
`reckless_reference/10-threading-numa-smp.md` (a full parallel study of a
second engine's lazy-SMP+NUMA implementation — cited throughout below for a
three-way comparison), and our own `design/13-threading-smp.md` +
`src/threads.rs` (P11, just landed).

---

## 1. The headline finding, up front

Integral's README says Lazy SMP "has been proven to scale very well at
higher thread counts compared to other alpha-beta chess engines." Having
read the whole threading subsystem and its git history, the honest
conclusion is: **the scaling reputation is not, primarily, a NUMA story.**
The git-archaeology in §6 shows the "scales very well" line was added to
the README on **2024-09-28** (commit `09d22d0e`) — about two months after
Lazy SMP first landed and a full **ten months before any NUMA-specific
code existed at all** (the first NUMA commit is `e61b7c89`, 2025-05-18).
Whatever makes Integral scale well was already true in September 2024, from
an architecture with:

- no thread pinning, no topology discovery, no per-NUMA-node data
  replication (contrast Reckless's `numa.rs`, `reckless_reference/10` §8);
- no thread-index aspiration jitter (no `thread_idx % 8` term, unlike SF and
  our own doc 13 §3.1) and no reduction-formula thread-diversity seed
  (unlike Reckless's `(nodes + thread_id*C) % 128` jitter, `reckless_reference/10`
  §6) — **no deliberate inter-thread diversity mechanism of any kind** (§5);
- no best-thread voting at all — the main thread's own result always wins,
  unconditionally (§4);
- a genuinely minimal shared-state surface: one lock-free TT plus two
  `atomic_bool`s (`stop_`, `quit_`) plus a handful of hand-rolled
  synchronization barriers, touched only at `go`/`stop` boundaries, never
  per-node.

In other words: Integral's scaling story reads as "do the classic lazy-SMP
recipe with an unusually small, clean, lock-free hot path, and don't add
any of the extra machinery (voting, jitter, quorum soft-stop, NUMA
replication) that other engines layer on top" — the opposite of "NUMA
awareness is the secret sauce." The NUMA first-touch work that *does* exist
(§6) was added roughly a year later, explicitly for TCEC/CCC-scale
hardware, is much smaller in scope than Reckless's NUMA subsystem, and (per
its own commit history) took a false start plus three commits, spanning
about three months from the false start, to get right — including one
false start whose diff contains a real concurrency bug, but which the
`git`-ancestry evidence says was never actually merged to `main` (§6.2:
`git merge-base --is-ancestor` fails, and its PR number is a clean gap in
`main`'s merged sequence) — a caught-before-shipping story, corrected here
from an earlier draft of this note that had it living on `main` for
10 weeks. That whole saga is a rich case study in its own right, and
is the part of this note most directly relevant to "how did a solo dev
pull this off" — see §9.

---

## 2. Core data structures

### 2.1 `Thread` — the per-thread state (`search.h:103-154`)

```text
struct alignas(64) Thread {
    id: u32                       // 0 == main thread (IsMainThread())
    board: Board                  // owned; refreshed every go via SetBoard
    history: History              // all move-ordering + correction tables
    stack: Stack                  // per-ply search stack
    nodes_searched: atomic<u64>   // owner-writes-with-fetch_add (see §5.3)
    scores: [Score; kMaxSearchDepth + 1]   // per-iteration best score, for TM
    previous_score: Score
    root_depth, sel_depth: u16
    tb_hits: atomic<u64>
    pv_move_idx: i32              // current MultiPV line
    root_moves: RootMoveList
    nmp_min_ply: u16
}
```

This is a close structural match to our own `Worker` (design/13 §3.2) and
to Reckless's `ThreadData` (`reckless_reference/10` §2.2) — one
struct-per-thread, single-owner, holding board + histories + stack + root
moves. Three differences worth flagging precisely:

- **`alignas(64)` on the whole struct**, not on a small isolated counters
  sub-struct. Our design cache-line-isolates only the tiny `Counters` (3
  atomics, `#[repr(align(64))]`, `src/threads.rs:47`); Reckless does the
  same (`Counter` per-shard alignment, `reckless_reference/10` §2.3).
  Integral instead aligns the *entire* multi-kilobyte `Thread` object.
  Since each `Thread` is a separate heap allocation (`unique_ptr<Thread>`,
  not an array element), this is a coarser, blunter false-sharing guard
  than ours — it protects the *first* cache line of each allocation from
  sharing with an adjacent unrelated allocation, but doesn't isolate
  `nodes_searched`/`tb_hits` from the rest of the *same* thread's hot
  fields the way our dedicated `Counters` struct does. Landed in isolation
  (`d6209d1a`, "Align search thread data", 2025-04-16) as part of the
  April 2025 TCEC-scaling push (§6.1) — i.e., added five years^H^H months
  after Lazy SMP itself, presumably after profiling showed it mattered at
  higher thread counts.
- **No `manager`/`SearchManager` split.** Main-only state
  (`report_info`, the soft-timeout check) is computed *inline* inside
  `IterativeDeepening` via `if (thread.IsMainThread())` branches
  (`search.cc:77,149,153,197`), not factored into a separate struct the way
  our `Option<Box<SearchManager>>` (design/13 §3.3) or SF's
  `ISearchManager`/`NullSearchManager` null-object are. Because Integral's
  main-only state is small (one `unique_ptr<ReportInfo>` local variable,
  not a persistent struct), this works fine for them but wouldn't scale to
  our `SearchManager`'s larger field inventory (`TimeManager`,
  `stop_on_ponderhit`, cross-move bookkeeping) without becoming unreadable.
- **`nodes_searched`/`tb_hits` are genuine `std::atomic<u64>` incremented
  with `fetch_add`** (`search.cc:404,610,833,1044`), not the cheaper
  single-writer load-then-store our doc 13 §6 and Reckless's `Counter`
  (`reckless_reference/10` §2.3) both use. Since each counter is written
  only by its owning thread (exactly the same single-writer situation we
  and Reckless optimize), this is a real, if small, unforced cost — a
  `lock xadd` on every node instead of a plain store. Worth noting as a
  place where **our existing implementation already does better than a
  top-6 engine**, not something to adopt.

### 2.2 `Searcher` — the pool + shared state (`search.h:156-220`)

```text
class Searcher {
    board_: &mut Board                       // the ONE shared position (§7)
    time_mgmt_: TimeManagement
    stop_, quit_: atomic<bool>
    stop_barrier_, start_barrier_,
      search_end_barrier_, thread_init_barrier_: Barrier   // §3
    stop_mutex_, thread_stopped_mutex_: Mutex
    searching_threads_: atomic<i32>
    thread_stopped_signal_: Condvar
    threads_: Vec<Box<Thread>>
    raw_threads_: Vec<std::thread>
    transposition_table_: TranspositionTable
}
```

There is **no separate `ThreadPool` type** — `Searcher` fuses "the search
algorithm" and "the thread pool" into one class, unlike SF (`Thread`+
`ThreadPool` are separate), Reckless (`ThreadData`+`ThreadPool` separate,
`reckless_reference/10` §2-3), and our own `Worker`+`ThreadPool` split
(design/13 §3.2/§3.5). This is a real simplicity win for a project this
size — no `PoolShared`/`Arc` indirection at all, because there's only ever
one `Searcher` instance owning everything directly.

### 2.3 `Barrier` — the one hand-rolled primitive (`utils/barrier.h`, 57 lines)

A **reusable, cyclic** barrier (like C++20 `std::barrier`, which the
codebase predates using or chooses not to depend on): `expected` parties
call `ArriveAndWait()`; the last arriver resets the counter and bumps a
`phase_` counter under a `notify_all`, and everyone else's wait predicate
is keyed off `phase_` advancing — so the barrier can be reused indefinitely
across "generations" without a separate reset call in the common case
(`Reset()` exists only for changing `expected` itself, e.g. on
`SetThreadCount`, and asserts the barrier is currently fully drained first).
This single ~50-line primitive is the *entire* synchronization mechanism
for the whole thread pool — no per-thread job queue, no channel, no
per-thread mutex+condvar pair. Compare:

- **Us** (design/13 §3.4): per-thread `Mutex<ThreadState{job,busy}> +
  Condvar`, one job slot addressable independently per thread.
- **Reckless** (`reckless_reference/10` §3.1-3.2): a zero-capacity `mpsc`
  rendezvous channel per thread for dispatch, plus a second
  `Mutex<bool>+Condvar` for completion — two primitives per thread, but
  still independently addressable (a job can be sent to thread *i* alone).
- **Integral**: **no per-thread addressability at all.** Every rendezvous
  is a **global, symmetric, all-or-nothing barrier** — you cannot send work
  to "just thread 3"; every `go` wakes literally every thread with the same
  action (search the same `board_`) at the same moment. This is the
  cheapest possible mechanism for a pool whose only job type ever *is*
  "everyone search," but it is why Integral has no `Job::Custom`-equivalent
  anywhere (§8) — the barrier protocol has no vocabulary for "run this one
  closure on this one thread."

---

## 3. Thread lifecycle: the double-barrier handshake

### 3.1 The worker loop (`Searcher::Run`, `search.cc:1292-1308`)

```text
fn run(thread):
    loop {
        stop_barrier.arrive_and_wait()     // "I have fully finished; ready for next go"
        start_barrier.arrive_and_wait()    // blocks until the UCI thread also arrives
        if quit { return }
        thread.reset()
        thread.set_board(shared_board)     // copies Searcher::board_ into thread.board
        iterative_deepening::<Regular>(thread)
    }
```

Every one of the `count` worker OS threads (**including thread 0, the
"main" search thread** — it is a real, separate OS thread from the UCI/stdin
thread, same as SF and our own design, unlike Reckless where "main" runs on
the message-loop thread itself) runs this identical loop.

### 3.2 `Start` (UCI thread, on `go`) — `search.cc:1380-1397`

```text
fn start(time_config):
    if searching_threads.load() > 0 { return }     // §7: silently DROPS a `go` mid-search
    stop_barrier.arrive_and_wait()                 // handshake #1 (releases parked workers)
    stop.store(false)
    time_mgmt.set_config(time_config); time_mgmt.start()
    searching_threads.store(thread_count)
    start_barrier.arrive_and_wait()                // handshake #2 (releases workers into search)
```

The two-barrier handshake is a neat piece of ordering logic worth spelling
out, because it is *not* obviously safe on first read: after handshake #1
releases everyone (workers *and* the UCI thread, symmetric arrival), the
worker threads race ahead into their loop body and immediately block again
at `start_barrier` (since it needs `count+1` arrivals and only `count`
workers have reached it yet) — so no worker can proceed into
`thread.reset()`/`set_board()`/`iterative_deepening` until the UCI thread
*also* reaches handshake #2, by which point it has already stored
`stop=false`, initialized `time_mgmt_`, and set `searching_threads_`. The
barrier itself supplies the happens-before edge (§3.11 in our own doc's
terms) — no separate atomic-ordering reasoning is needed beyond "a barrier
release is a full synchronization point for every party." This is a
strictly simpler way to get the same "publish setup before waking workers"
guarantee our design gets from a `Setup` job + per-shell `wait_idle`
(design/13 §3.6 steps 4-6) or Reckless gets from blocking on
`execute_searches`'s own scope join (`reckless_reference/10` §3.3) — at the
cost of losing per-thread addressability (§2.3) and of every worker
literally copying the *same* `board_` reference (§7), rather than each
being handed its own already-prepared `SetupJob`.

### 3.3 End of search (`IterativeDeepening`'s epilogue, `search.cc:189-217`)

```text
send_stopped_signal():
    { lock(thread_stopped_mutex); --searching_threads; notify_all() }
    search_end_barrier.arrive_and_wait()     // ALL threads rendezvous here, main included

if thread.is_main():
    if time_mgmt.is_infinite() { busy_wait_on(!stop) }     // no ponder (§10) — just infinite
    lock(stop_mutex); stop.store(true)
    send_stopped_signal()
    transposition_table.age()
    if regular_search { print "bestmove <thread.root_moves[0].move>" }
else:
    send_stopped_signal()
```

Two things worth naming precisely:

- **`search_end_barrier_` has no `+1`** — it is sized to exactly
  `thread_count` (`search.cc:1351` in the current `SetThreadCount`), unlike
  `stop_barrier_`/`start_barrier_` which are `count+1` (to include the UCI
  thread). Only worker threads ever call `ArriveAndWait` on it; it exists
  purely so that **the main thread's `bestmove` print happens after every
  helper has also finished its own iteration** — a lightweight analogue of
  our "wait for all helpers before voting" step (design/13 §3.7 step E),
  except here there is nothing to *vote on* (§4).
- **`WaitForThreads()` (the `stop`-command handler's blocking wait,
  `search.cc:1310-1317,1433-1436`) only waits on the `searching_threads_`
  counter reaching zero — not on `search_end_barrier_` itself.** Since the
  counter is decremented *before* a thread calls `search_end_barrier_.ArriveAndWait()`,
  the very last thread to decrement can wake the UCI thread's `Stop()` call
  before that same thread has actually completed its own barrier
  rendezvous. This is **not a bug**: the next `Start()` call's own
  `stop_barrier_.ArriveAndWait()` will simply block until every worker
  genuinely finishes looping back around, so the raciness is absorbed by
  the next synchronization point rather than causing any observable
  incorrectness — a nice small illustration of the same "accepted
  approximate ordering, corrected at the next real handoff" philosophy our
  own doc 13 §3.13 documents for `nodes_searched()`/`best_move_changes`.

---

## 4. Best-thread "voting": there isn't any

This is the single most surprising finding relative to both reference
engines. SF's `get_best_thread` (`stockfish_reference/stockfish_search_notes.md`
§13.2) and Reckless's depth-weighted vote (`reckless_reference/10` §7) both
implement real cross-thread comparison logic — decisive-score protection,
vote accumulation, PV-length tiebreaks — and our own design/13 §3.8
specifies a close reimplementation of SF's algorithm. **Integral has none
of this.** `IterativeDeepening`'s epilogue (§3.3) prints
`thread.root_moves[0].move` **only when `thread.IsMainThread()`** — helper
threads' final root-move lists are read by *nobody*, ever. The engine's
answer to "which thread's result do we trust" is simply **"thread 0's,
unconditionally, always"** — the other `N-1` threads exist purely to feed
the shared TT with more search coverage; their own conclusions are
discarded outright.

This is a real, deliberate simplification (not an oversight — there is
nowhere in the 669-commit history where a voting mechanism was added and
later removed; it was never there). It trades away the modest Elo that SF
and Reckless report from voting (the classic argument: at high thread
counts, a helper occasionally reaches a better conclusion than main by
sheer luck of TT arrival order, and letting it "win" recovers a little bit
of strength) for **zero extra code, zero extra synchronization at the end
of search, and one less place to get the decisive-score-protection logic
wrong.** Given that the "scales very well" reputation was earned by
September 2024 with this exact policy already in place, it's reasonable to
conclude voting is a small refinement on top of a healthy lazy-SMP
foundation, not a prerequisite for good scaling — worth keeping in mind
when judging how much engineering effort our own §3.8 voting logic is
"owed" relative to other threading work.

---

## 5. Thread diversity: TT races are the *only* mechanism

### 5.1 No aspiration jitter

`IterativeDeepening`'s aspiration window (`search.cc:83-88`) computes
`window = kAspWindowDelta + average_score^2 / kAspWindowScoreDiv` from
**only** the running average score — there is no `thread.id` term
anywhere in the formula, unlike SF's `threadIdx % 8` addend
(`stockfish_reference/stockfish_search_notes.md` §13.1) which our own doc
13 §3.1 also specifies. Every thread, at the same score, computes the
*identical* initial window.

### 5.2 No reduction-formula jitter either

Reckless, having dropped thread-indexed aspiration jitter, moved the
diversity seed one layer deeper into the LMR reduction formula itself
(`(nodes + thread_id*C) % 128`, `reckless_reference/10` §6). Integral's
`CalculateLMR`/`GenerateLateMoveReductionTable` (`search.cc:20-46`) is a
plain `base + ln(depth)*ln(moves)/divisor` table, shared read-only by every
thread (`tables::kLateMoveReduction`, a namespace-level global, not
per-thread) — no thread-id or node-count perturbation term at all.

### 5.3 No RNG-seeded divergence

`utils/random.h` exists (`thread_local std::mt19937_64`, fixed seed
`0x1333317`) but a grep of the search module shows it is not called from
anywhere in `search.cc`/`move_picker.cc` — it is used elsewhere (Zobrist
init, datagen book-move sampling), not for search-thread diversification.

### 5.4 What's left: pure TT-arrival-order divergence

With items 5.1-5.3 all absent, the **entire** mechanism by which
Integral's N threads ever produce different trees is: they start from
byte-identical state (same board, same empty history at game start, same
window, same reduction table) and diverge purely because real wall-clock
scheduling differences mean thread A's write to a TT cluster lands before
or after thread B's read of the same cluster, cascading into different
cutoffs, different move-ordering feedback (since each thread's *own*
history tables then update differently based on which moves it actually
searched), and — as histories diverge — increasingly different trees
deeper into the game tree. This is the same "TT osmosis" mechanism that SF,
Reckless, and our own doc 13 §3.9 all rely on as *one of several* sources
of diversity; Integral relies on it as the **only** source, and evidently
that alone is enough to get useful multi-threaded search gains at the
thread counts Integral has been tested at. This is a useful data point for
prioritization: our own §3.1 jitter formula and the Reckless-inspired
"cross-thread optimism fusion"/"reduction jitter" ideas already flagged in
our doc 13 §10 are refinements on top of a mechanism that isn't strictly
required to get *some* scaling — they should still be evaluated by SPRT on
their own merits (per doc 13 §11 process), not assumed necessary by
analogy to what "real" engines do.

---

## 6. The NUMA first-touch saga — an abandoned first attempt, then a three-commit, two-day fix

### 6.1 The build-up: TCEC/CCC-driven scaling work (April 2025)

Four commits cluster tightly in April 2025, all in service of high-core-count
tournament hardware:

- `df71df67`, **"Multithreaded TT clearing (#186)"**, 2025-04-10 — commit
  message literally says *"Intended for TCEC/CCC"*. `TranspositionTable::Clear`
  (`transpo.cc:82-101`) changed signature to take a thread count and spawns
  that many **short-lived, throwaway** `std::thread`s, each `memset`-ing an
  equal contiguous shard of the cluster array, then joins them all. Purely a
  wall-clock-speed measure (parallel memory bandwidth for a big `Hash`
  clear) — it does **not** attempt to make those shards NUMA-local to the
  *search* threads that will later probe them (the clearing threads are
  freshly spawned each call, not the persistent pool), which is a
  reasonable simplification given the TT is one global, un-replicated
  resource that every thread on every node probes anyway (§9 in Reckless's
  study: only NUMA-*replicated* resources benefit from node-local clearing).
- `d6209d1a`, **"Align search thread data"**, 2025-04-16 — `alignas(64)` on
  `Thread` (§2.1).
- `23e035b3`, **"Increase `Threads` option maximum value"**, 2025-04-20 —
  raises the UCI `Threads` ceiling from 256 to 512, presumably because a
  target machine had more logical cores than the old cap allowed.

### 6.2 The false start — a race condition disguised as a NUMA fix, apparently caught pre-merge (`e61b7c89`, "#198", 2025-05-18)

The *idea* (correct, and exactly what our own design already does, §7): stop
constructing each `Thread` object on the UCI/setup thread and instead
construct it **inside the lambda that runs on the newly spawned OS
thread**, so the very first page-touching writes to that thread's `board`/
`history`/`stack` happen from the CPU (and NUMA node) that will actually
run the search — first-touch placement, for free, matching Linux's
default `MPOL_DEFAULT` policy of "a page is physically allocated on the
node of the CPU that first faults it in."

The *implementation*, however, has each spawned thread call
`threads_.emplace_back(std::make_unique<Thread>(...))` on the **shared**
`std::vector<std::unique_ptr<Thread>>` — from `count` different OS threads,
**concurrently, with no lock**. `std::vector::emplace_back` is not safe to
call from multiple threads on the same vector instance: if the vector needs
to grow, one thread's reallocation can invalidate the memory another thread
is mid-write into, and even without reallocation, concurrent unsynchronized
writes to the vector's internal size/pointer bookkeeping are a data race by
definition. This is a real, latent concurrency bug — introduced *while
fixing* a locality issue. **Correction:** an earlier draft of this note
claimed this commit "shipped and stayed live on `main` for exactly ten
weeks" until superseded by `aa10d228`. That claim does not survive a
git-ancestry check: `git merge-base --is-ancestor e61b7c89 main` fails
(exit 1), `git log --oneline main` never mentions `e61b7c89` or its "#198"
PR title, and — most tellingly — `main`'s merged-PR-number sequence around
that date runs `#197, #199, #200, #201, ...` with **`#198` conspicuously
absent**. The commit object is only reachable today via two unrelated,
never-merged branches (`origin/capture_text`, `origin/cfix`), not via any
NUMA-named branch. The most consistent reading of this evidence: PR #198
was opened (and bench-tested locally/pre-CI, hence the real `Bench:`
trailer) but never merged — the race condition described above is a real
bug *in a proposed diff*, not a bug that ever ran live on `main`. Whether it
was caught by review, by a stress run, or simply abandoned in favor of a
cleaner rewrite 2.5 months later is not evidenced either way; what the
history does rule out is "it shipped for 10 weeks." (Also present in this
same unmerged commit: removing the previous approach of storing
`std::thread raw_thread` as a *field inside* `Thread` itself, in favor of a
parallel `raw_threads_` vector — a separation-of-concerns cleanup riding
along with the (buggy, unmerged) locality fix; `main` picks up the
`raw_threads_` split independently in `aa10d228`, §6.3.)

### 6.3 The first fix to actually land on `main`, with a fresh lifetime hazard (`aa10d228`, "#218", 2025-08-01)

About 2.5 months after the abandoned `#198` attempt, a proper fix: `threads_.resize(count)` (pre-sizing the
vector on the calling thread, so no concurrent reallocation can occur) is
done *before* any thread is spawned; each spawned thread then writes only
to its own distinct index (`threads_[i] = make_unique<Thread>(i)`), which
is race-free because distinct vector slots are distinct memory and no two
threads write the same one. A local `Barrier thread_init_barrier(count+1)`
synchronizes "every slot is constructed" before any thread proceeds into
`Run()`.

But this local barrier is itself a hazard: it is a **stack-local object in
`SetThreadCount`**, captured **by reference** into every spawned thread's
closure (`[&, i]`). Once the *last* required party (the UCI/calling thread,
completing the `count+1`-th arrival) returns from its own
`ArriveAndWait()` call, `SetThreadCount` returns and the local
`thread_init_barrier` is destroyed — while other worker threads may still
be inside their own `ArriveAndWait()` call on that same object (mid
wake-up, reacquiring the internal mutex, or simply not yet having returned
from the call), which is a use-after-scope access to a destroyed
`std::mutex`/`std::condition_variable`. Whether this ever manifested
observably is not evidenced in the history, but the *shape* of the bug —
and the fact that it was fixed literally the next day — is a strong
signal it was found quickly, likely by a stress run at a thread count high
enough to make the race window non-negligible.

### 6.4 Two more same-day fixes — cleanup, and the *real* second half of the fix (2025-08-02)

- `22162dd9`, **"Potential NUMA fix"**: promotes `thread_init_barrier` from
  a local variable to a persistent member field `thread_init_barrier_`
  (`search.h`), closing exactly the lifetime hazard in §6.3 — the barrier
  now lives as long as `Searcher` itself, so no thread can ever reference it
  past its destruction. Also drops an accidental `raw_threads_.resize(count)`
  that (combined with the loop's own `emplace_back` calls) would have left
  `count` extra default-constructed (never-joinable, harmless-but-wasteful)
  `std::thread` placeholders sitting in the vector.
- `0d55bc02`, **"Another NUMA fix (#219)"** — the commit that actually
  matters most, and is easy to miss if you only skim commit titles: it
  moves `thread.Reset()` and `thread.SetBoard(board_)` **out of `Start()`**
  (where the *UCI thread* was calling them, once per `go`, on every
  thread's `Board`/history state) and **into the top of each worker's own
  `Run()` loop** (§3.1), so those big per-`go` writes — re-zeroing the
  search stack, re-deriving the whole NNUE accumulator stack from the new
  position — now happen on the thread that will actually use the result,
  every single move, not just at pool-construction time. This is the
  deeper half of the NUMA fix: §6.3's first-touch correction only protects
  the *one-time* initial allocation; without this second change, **every
  subsequent `go`** would have had the UCI thread's CPU dirty pages that
  first-touch had placed on a different node, re-introducing the exact
  remote-access penalty the whole effort was for. It is the per-move hot
  path, not the one-time startup path, that this commit actually fixes —
  and it is the one instance where "NUMA-aware" and "who is allowed to
  mutate a `Thread`'s own state" turn out to be the *same* design question.

### 6.5 Reading the saga as a whole

One abandoned false start (`e61b7c89`, "#198" — never merged to `main`,
§6.2) plus three commits that did land on `main` (`aa10d228` + `22162dd9` +
`0d55bc02`, the latter two a tight two-day cluster right after `aa10d228`)
span **about 2.5 months end to end** (false start 2025-05-18, final fix
2025-08-02; under four months measured from the April 2025 TCEC push that
motivated the whole effort) — corrected here from an earlier "fifteen
months" claim in this note, which did not survive a git-log date check
(§1's own "three months" was the accurate figure all along), and further
corrected to reflect that only 3 of these 4 commits are actually part of
`main`'s history (§6.2) — to land what our own design
achieved *by construction, at zero extra cost*, from the very first
commit of `src/threads.rs`: because `thread_main` (`src/threads.rs:271-319`)
already builds `Worker::new_in_pool(...)` **inside** the spawned closure,
and every per-`go` `Setup` job (`worker.setup(*s)`, called *on the worker's
own thread* by the job-loop dispatch, `src/threads.rs:290`) reinstalls
board/limits/histories from inside that same thread — both halves of
Integral's fix (construction-time locality, §6.3, *and* per-`go`
re-touch locality, §6.4) are simply how our design already works, and this
was true before P11 ever ran a single multi-threaded search. See §11 item
1 for the concrete "nothing to adopt, just confirm" note this justifies.

---

## 7. What's shared vs. per-thread, and one real correctness gap

| Item | Owner / access |
|---|---|
| `TranspositionTable` | shared, lock-free atomics (not studied in depth here — see a future TT-focused note if one exists) |
| `stop_`, `quit_` | `atomic<bool>`, relaxed-ish (see below) |
| `searching_threads_` | `atomic<i32>`, decremented by each thread at search end |
| `board_` (**the position itself**) | a single `Board` object, owned by the *caller* of `Searcher`'s constructor (i.e. by `AcceptCommands`'s local `Board board`, `uci.cc:240-243`), referenced by `Searcher` and by every `Thread::SetBoard` copy |
| Everything else (`history`, `stack`, `root_moves`, `nodes_searched`, …) | single-owner inside each `Thread`, per §2.1 |

The one genuine gap, found by reading rather than by the engine's own tests
(there are none for this): **`board_` is mutated directly by the `position`
UCI command handler (`uci.cc:48-64`, `board.SetFromFen(...)`/
`board.MakeMove(...)`) with no check that a search is not currently
running**, and **`Searcher::NewGame` (`search.cc:1438-1447`, the
`ucinewgame` handler) mutates every `Thread::history`/`stack` directly with
no `Stop()`/drain first either.** Both assume the GUI obeys the UCI
convention of sending `stop` before `position`/`ucinewgame` during an
active search — the same assumption SF makes — but unlike SF, Integral's
own `Start()` doesn't defensively wait for a previous search to fully drain
before proceeding (§3.2: it just silently *drops* a second `go`, `search.cc:1380-1383`,
rather than queuing behind the first the way our own
`ThreadPool::start_thinking`'s step 1 `wait_for_search_finished()` does,
design/13 §3.6 step 1). A misbehaving or racy GUI/harness that sends
`position`/`go` back-to-back without an intervening `stop` — plausible in
an automated SPRT/gauntlet harness under load — could hit a real, if
narrow, data race on `board_`, or could have its second `go` silently
ignored. Nothing to adopt; flagged because our own design already handles
both cases correctly (design/13 §3.6 step 1 blocks-and-waits rather than
drops; `ThreadPool::clear()`, `src/threads.rs:493-501`, always
`wait_for_search_finished()` before touching any per-thread state).

---

## 8. No `Job::Custom` equivalent — and why that's fine for Integral

Our design/13 §9 (open question 6) and the Reckless study (`reckless_reference/10`
§3.2, §7 item 4) both converge on "a generalized `Job::Custom(closure)`
dispatch is the natural shape for *every* pool operation, not just search."
Integral has no such thing, and structurally *can't* cheaply add one within
the barrier model (§2.3): the only two actions the pool can ever be told to
do are "everyone search `board_` together" (via `Start`) and "everyone
exit" (via `QuitThreads`, itself just another `stop_barrier`/`start_barrier`
round-trip with `quit_` checked after, `search.cc:1319-1333`). There is no
"run this on thread 3 only" primitive, and nothing in the codebase ever
needs one — even the multithreaded TT clear (§6.1) sidesteps the persistent
pool entirely and spawns its own throwaway threads instead of dispatching
through `threads_`/`raw_threads_`. For an engine whose only two pool
operations really are "search" and "exit," the barrier model is *simpler*
than a job-queue/closure model would be, at the cost of that generality —
a legitimate trade for Integral's scope, but not one we should copy, since
we already have concrete near-term uses for `Job::Custom` (parallel TT
clear *on the persistent pool*, for exactly the NUMA-locality reason §6.1
explains Integral's own ad-hoc clear doesn't get; the `run_on_main` test
hook already built on it, `src/threads.rs:511-519,596-623`).

---

## 9. No ponder support (converging with Reckless, diverging from SF and us)

A repo-wide search turns up **zero** occurrences of "ponder" anywhere in
Integral's source. Like Reckless (`reckless_reference/10` §4), Integral
skips `go ponder`/`ponderhit` entirely — `IterativeDeepening`'s only
"don't answer yet" condition is `time_mgmt_.IsInfinite()` (busy-wait while
`!stop`, `search.cc:197-200`), with no separate ponder flag or
`stop_on_ponderhit` state machine anywhere. Between the two *later*,
smaller-team/solo top engines we've now studied in depth, **both** dropped
ponder; only Stockfish (the shared ancestor everyone's UCI conventions
trace back to) still carries it. This is circumstantial but consistent
evidence that ponder is complexity that doesn't move the needle on the
metric that actually decides engine rankings (fixed-movetime/depth SPRT and
rating-list games, which don't ponder) — it matters only for how the
engine feels in a GUI used by a human. Nothing here argues for *removing*
our own ponder support (design/13 §3.7 step D, already built and tested for
P11) — it's a real UCI feature and we already paid for it — but it's worth
recording as a data point next time similar-scope "is X worth its
complexity" questions come up: two-for-two among the non-SF engines we've
studied said no.

---

## 10. Testing discipline: no threading-specific verification at all

`src/tests/` contains exactly three suites: `perft_test.cc`, `see_test.cc`,
`bench_test.cc` — nothing that exercises the thread pool, the barrier
protocol, or concurrent access under a sanitizer. A search of `Makefile`/
`CMakeLists.txt` turns up no `-fsanitize=thread`/`address`/`undefined`
build target at all. This matters directly for interpreting §6: the
ten-week-live vector-race bug (§6.2) and the next-day lifetime hazard
(§6.3) were **not** caught by any automated concurrency check — Integral
has none — and almost certainly not by the project's standard
patch-acceptance testing either, because that testing (like virtually all
top engines', via OpenBench-style SPRT: confirmed by an explicit
`// OpenBench requires the bench command to be parsed from the command
line` comment, `uci.cc:248`) runs Elo-gated games **at `Threads=1`** by
convention, for the same reason every engine's SPRT pipeline does: it
maximizes games-per-CPU-hour and Elo is (to first approximation)
thread-count-invariant for *quality-of-search* questions. **A
single-threaded SPRT gate structurally cannot catch a multi-threaded-only
concurrency bug** — it would need a dedicated high-thread-count stress/hang/
race pass, which this history shows Integral doesn't run as a matter of
course; these bugs were evidently found only once someone (very plausibly
in the specific run-up to a TCEC/CCC event, §6.1) actually exercised the
engine at high thread counts on real hardware. This is the sharpest,
most directly actionable lesson from the whole NUMA saga for us: **our
existing design/13 §5 test plan (ThreadSanitizer stress run, loom model of
the job-loop protocol, the 500-search interleaving smoke test, the explicit
panic-recovery tests already in `src/threads.rs`) is not a "nice to have"
matching what a top-6 engine does — it is *more* rigorous than what the
top-6 engine we studied actually has, and this history is direct evidence
of the kind of bug that gap leaves open.** Keep paying for it.

---

## 11. Comparison summary

**vs. Stockfish** (`stockfish_reference/stockfish_search_notes.md` §13):
same non-blocking `go` / persistent-thread-pool shape and same "TT + a
couple of atomics" shared-state philosophy, but Integral has no
`SearchManager`/null-object split, no best-thread voting, no thread-index
aspiration jitter, and (until 2025-08) no NUMA first-touch discipline at
all — a leaner subset of SF's design in almost every dimension except the
barrier-vs-mutex mechanism, which is genuinely different from both SF's and
our own per-thread job-queue approach.

**vs. Reckless** (`reckless_reference/10-threading-numa-smp.md`, all
sections): Reckless is the maximalist of the three studied engines here —
a three-tier (global/per-NUMA-node/per-thread) state model, real
`sched_setaffinity` pinning gated by a measured "is it worth it" formula,
cross-thread optimism fusion via `fetch_max`, a 65%-quorum soft-stop vote,
depth-weighted best-thread voting, and a generalized closure-based job
dispatch (at the cost of one `unsafe` transmute). Integral is the
minimalist: none of the above except a narrow, mechanically different
first-touch fix that took three tries to land correctly, no voting at all,
and no diversity mechanism beyond TT races. Both are reported as
excellent-scaling engines — strong evidence that the *maximal* Reckless
feature set is not a precondition for good lazy-SMP scaling, only a set of
further refinements on top of a healthy minimal foundation both engines
share (lock-free TT, no locks on the hot path, per-thread histories, clean
first-touch construction).

**vs. our `design/13` + `src/threads.rs` (P11)**: our design already
independently arrived at (a) constructing per-thread state on the thread
that will use it (`thread_main`, matching Integral's *eventual*, hard-won
fix, §6.5), (b) a persistent parked-thread pool with non-blocking `go`
(matching both SF and Integral, unlike Reckless's blocking model), (c) the
single-writer load-then-store counter-bump optimization Integral doesn't
use (§2.1), and (d) a `wait_for_search_finished`/`clear()` discipline that
never touches per-thread state without draining first — precisely the gap
found in Integral's `board_`/`NewGame` handling (§7). Where our design is
*more* elaborate than Integral's: `SearchManager` null-object split, full
best-thread voting with decisive-score protection, per-thread job
addressability (`Job::Custom`), and a real automated concurrency-testing
plan (§10). Where Integral is simpler and that simplicity looks earned
rather than accidental: no voting, no thread-index jitter, one barrier
primitive instead of a job queue. Nothing in this note argues for removing
any of our existing machinery — it argues that our current design choices
were, independently, the *correct calls* on every point where Integral's
history shows it initially got something wrong and had to redo it.

---

## Adapt to us

Ordered roughly by effort/confidence, cheapest and most-validated first.
Items that touch search *behavior* need the fixed-node proxy-SPRT gate
(doc 14 §9.4) before being trusted; items that are pure validation of
already-shipped design need no gate at all.

1. **No action needed — confirm and move on.** Integral's entire NUMA
   first-touch saga (§6) is a hard-won (one abandoned attempt, then a
   three-commit fix landing over ~2.5 months) arrival at exactly what
   `src/threads.rs`'s `thread_main` +
   `Worker::setup` already does from day one of P11: construct per-thread
   state on the thread that uses it, and re-touch that state *from the
   same thread* on every subsequent `go` (not from the UCI thread). Record
   this note as the citation the next time design/13 §3.15 ("NUMA —
   explicitly deferred") is revisited, so a future reader doesn't
   re-discover the need for this the hard way. Touches: design/13 §3.15
   (add a citation), no code change. Phase: documentation-only, now.
   Effort: trivial. Impact: none (already have it) — risk-avoidance value
   only.

2. **Treat §10 as a standing argument for keeping our TSan/loom/stress
   test plan, not trimming it.** No code change; if design/13 §5's testing
   plan is ever proposed for descoping "since it hasn't found anything
   yet," this note is the concrete counter-evidence: a top-6 engine with no
   such testing produced a real vector-race bug (in a PR that, per §6.2's
   correction, was apparently caught before merge rather than shipped) and
   did ship a barrier-lifetime bug for one day, both invisible in principle
   to its standard (single-threaded, Elo-gated) SPRT pipeline. Touches:
   design/13 §5 (cite this note in the rationale), design/14 (process doc,
   if it discusses test-suite trimming). Phase: ongoing discipline, no
   phase. Effort: trivial. Impact: risk-avoidance.

3. **Consider whether `alignas`-the-whole-struct is worth a look for
   `Worker`, in addition to our existing `Counters` isolation.** Integral
   added `alignas(64)` to the entire `Thread` struct only in April 2025,
   as part of a *measured* TCEC-scaling push (i.e., they didn't have it
   from day one either, and added it once it started mattering at higher
   thread counts) — our `Worker` doesn't currently carry a struct-level
   alignment attribute. Low priority (our hot counters are already isolated
   into their own aligned `Counters`, which is the part that matters most;
   this would only help if profiling at high thread counts ever shows
   cross-`Worker` false sharing on other fields). Touches: `src/threads.rs`
   / `src/search`'s `Worker` definition. Phase: P11 hardening, opportunistic
   — only if a future profile at 8+ threads shows it. Effort: trivial (one
   attribute). Impact: unmeasured, likely tiny; not worth a dedicated SPRT,
   just a profile-triggered follow-up.

4. **Do not adopt: dropping best-thread voting, thread-index jitter, or
   ponder.** Recorded explicitly as *non*-adoptions, because the temptation
   after reading §4/§5/§9 might be "a top-6 engine doesn't bother, so
   maybe we're over-engineering." The counter-argument: Integral's
   simplicity there is consistent with *not yet having measured* whether
   adding them would help (there's no history of anyone trying and
   reverting them, only of them never existing) — it is evidence that
   they're not *required* for good scaling, not evidence that they're
   worthless. Our own doc 13 §3.8 voting and §3.1 jitter are already
   shipped and working; the Reckless-derived refinements in doc 13 §10
   remain open, gated proxy-SPRT candidates exactly as already recorded
   there. This note doesn't change that gating, only removes "but does a
   top-6 engine even need this" as a reason to skip evaluating them.

5. **If we ever build a parallel TT clear (design/13 §3.15's `Job::Custom`
   use-case), prefer dispatching it through the *persistent* pool, not
   ad-hoc threads.** Integral's own parallel clear (§6.1, `transpo.cc:82-101`)
   spawns fresh throwaway threads each time rather than using
   `threads_`/`raw_threads_` — reasonable for them only because the TT is
   never NUMA-replicated in the first place, so there's no locality to
   preserve. If we ever *do* NUMA-replicate anything (design/13 §3.15,
   currently inert), a parallel clear must run on the actual persistent
   worker threads (via `Job::Custom`) so the clearing writes land on the
   same node the subsequent search reads from — Integral's shortcut is not
   safe to copy in a world where locality is being deliberately managed.
   Touches: design/13 §3.15 (note this constraint for whenever parallel
   clear is actually built). Phase: post-P11, NUMA-gated (per doc 13 §10
   item 7's "mechanism vs. behavior" split already recorded from the
   Reckless study). Effort: n/a (documentation constraint only, for now).

---

## File:line / commit quick index

- `search.h`: `Thread` 103-154 (142-153 field list, 121-123 `IsMainThread`);
  `RootMove`/`RootMoveList` 28-101; `Searcher` 156-220 (209-219 the shared
  state: `board_`, four `Barrier`s, `searching_threads_`, `threads_`/
  `raw_threads_`, `transposition_table_`).
- `search.cc`: `IterativeDeepening` 64-218 (77-88 `report_info`/aspiration
  setup with no thread-id term, 189-217 the `SendStoppedSignal`/main-thread
  epilogue and its *absence* of voting); `PVSearch` node-entry time-check
  494-500 (`static thread_local int counter`, main-thread-gated, mask
  `& 4095`); node-counter `fetch_add` sites 404, 610, 833, 1044;
  `Searcher::Run` 1292-1308; `WaitForThreads` 1310-1317; `QuitThreads`
  1319-1333; `ShouldQuit` 1335-1337; `SetThreadCount` 1339-1378 (the
  first-touch construction loop, post-fix); `Start` 1380-1397 (1381-1383
  the silent-drop-on-concurrent-`go` guard); `DataGenStart` 1399-1417,
  `Bench` 1419-1431 (both bypass the pool/barriers, direct
  `unique_ptr<Thread>` — matches our own "detached `Worker`" bench path,
  design/13 §11); `Stop` 1433-1436; `NewGame` 1438-1447 (no drain-first
  guard, §7); `GetNodesSearched`/`GetTbHits` 1453-1465; `ResizeHash`
  1467-1470.
- `transpo.cc`: `Clear(int num_threads)` 82-101 (throwaway-thread parallel
  memset, no persistent-pool/locality tie-in).
- `utils/barrier.h`: whole file, 57 lines (`ArriveAndWait` 25-43,
  `Reset` 17-23 with its "must be fully drained" assert).
- `uci.cc`: `AcceptCommands` 239-261 (single blocking `listener.Listen()`
  read loop, no separate listener thread — `go` is non-blocking by
  construction of `Start()`'s barrier handshake, so none is needed, unlike
  Reckless's workaround); `position` handler 43-65 (mutates `board_` with
  no active-search guard, §7); `go` handler 67-121 (121 `searcher.Start(...)`;
  no `ponder` argument anywhere, §9); `stop` 148-156; `ucinewgame` 157-159;
  `Threads`/`Hash` options 20-25.
- Commits (chronological): `05a0f454` "Thread pool foundation (#25)"
  2024-07-25; `45740fe6` "Remove spin locking when idle (#29)" 2024-07-29
  (introduces `Barrier`); `080d346f` "Lazy SMP (#30)" 2024-07-29,
  same day; `48887c01` "Fix threads not stopping correctly (#44)"
  2024-08-28; `09d22d0e` "Update README (#87)" 2024-09-28 (adds the
  "scales very well" claim — §1); `f0d7de96` "SMP fixes + cleanups"
  2024-09-24; `df71df67` "Multithreaded TT clearing (#186)" 2025-04-10
  ("Intended for TCEC/CCC"); `d6209d1a` "Align search thread data"
  2025-04-16; `23e035b3` "Increase `Threads` option maximum value"
  2025-04-20; `e61b7c89` "NUMA First Touch Policy for Thread objects
  (#198)" 2025-05-18 (§6.2, the vector-race attempt); `aa10d228` "NUMA
  First Touch Policy for `Thread` objects (#218)" 2025-08-01 (§6.3, the
  local-barrier-lifetime attempt); `22162dd9` "Potential NUMA fix"
  2025-08-02 (§6.4a); `0d55bc02` "Another NUMA fix (#219)" 2025-08-02
  (§6.4b, the per-`go` re-touch fix — the deeper half).
