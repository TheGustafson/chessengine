# Threading, thread pool & NUMA — lazy SMP in Reckless

Source: `src/thread.rs`, `src/threadpool.rs`, `src/numa.rs` (primary), plus
`src/search.rs` (thread-visible fields/flags in use), `src/uci.rs` (pool
lifecycle, best-move voting, the stdin listener), `src/time.rs`
(per-thread time checks), `src/history.rs` (the one table that is actually
NUMA-shared), `src/nnue.rs` (the other one). Reckless v0.10.0-dev, AGPLv3 —
ideas, formulas, and file:line pointers only; no code reproduced
(D27-equivalent). Compared throughout against
`stockfish_reference/stockfish_search_notes.md` §13 and
`stockfish_reference/04-tt-search-structures.md` ("Threading data layout"),
and against our own `design/13-threading-smp.md` (which already anticipates
much of this and explicitly defers NUMA at §3.15).

---

## 1. The big picture: a three-tier state model, not two

Both Stockfish's classic story and our design doc's inventory (§3.1) present
threading state as a simple two-way split: **shared** (TT, flags, options)
vs. **per-thread** (everything else). Reckless's shipped code has a **third
tier** wedged in between: state shared **per NUMA node** among the sibling
threads bound to that node, but replicated **across** nodes. Concretely:

| Tier | Lives in | Examples | Mutation |
|---|---|---|---|
| Global-shared (1 instance, all threads) | `SharedContext` (`thread.rs:156-188`) | TT, `Status`, node/tbhit counters, `soft_stop_votes`, `best_stats: [AtomicU32; MAX_MOVES]` | relaxed atomics |
| Per-NUMA-node-shared (1 instance per node, shared by that node's threads) | `Arc<T>` handed out by `NumaReplicated<T>` (`numa.rs:330-375`) | `SharedCorrectionHistory` (pawn + non-pawn correction tables, `thread.rs:144-154`), NNUE `ParametersHandle` (`nnue.rs:387-427`) | relaxed atomics (history) / read-only (weights) |
| Per-thread (1 instance per thread, never observed by another thread) | `ThreadData` (`thread.rs:213-243`) | `Board`, `root_moves`, `pv_table`, noisy/quiet/pawn/continuation history, `stack`, `time_manager`, NNUE accumulator | plain, single-owner |

This is exactly the shape our own §9.5 (doc 09) and §3.15 (doc 13) open
questions describe as "the SMP-phase option we explicitly defer" — Reckless
is the concrete, working answer to that deferral. One nuance worth pinning
precisely: Reckless shares **only** the correction-history family per NUMA
node (`thread.rs:145-148`: `pawn` + `non_pawn: [CorrectionHistory; 2]`).
Butterfly/capture/continuation/pawn-*quiet* histories all stay fully
per-thread inside `ThreadData` (`thread.rs:223-227`), even under NUMA
replication. That is narrower than what our notes record current Stockfish
doing ("SF shares pawn/correction/continuation histories per NUMA node",
doc 09 §9.5) — Reckless made a more conservative cut of *which* tables are
worth the cross-node synchronization cost.

---

## 2. Core data structures

### 2.1 `SharedContext` — the global-shared surface (`thread.rs:156-188`)

One instance, `Arc<SharedContext>`, owned jointly by the UCI/message loop and
every `ThreadData`. Fields: `tt`, `status: Status`, `nodes`/`tb_hits:
Counter`, `stop_probing_tb`/`root_in_tb: AtomicBool`, `soft_stop_votes:
AtomicUsize`, `best_stats: [AtomicU32; MAX_MOVES]`, `history` and
`parameters` (both `Arc<NumaReplicated<_>>`), and `numa_context:
Arc<NumaReplicationContext>`. Constructing it builds a fresh
`NumaReplicationContext` from `NumaConfig::from_system()` — topology
discovery happens once, at process/engine-reset time, not per search.

### 2.2 `ThreadData` — per-thread state (`thread.rs:213-284`)

Mirrors our planned `Worker` almost field-for-field: an `id: usize` (0 =
main — used the same way our doc uses `thread_idx`), a clone of the
`Arc<SharedContext>`, an `Arc<SharedCorrectionHistory>` handle obtained once
at construction via the NUMA token, its own `Board`, `TimeManager`, boxed
`Stack`, NNUE `Network`, `root_moves`, PV table, all four *other* history
tables, `optimism: [i32; 2]`, root/completed depth, `nmp_min_ply`, MultiPV
bookkeeping (`multi_pv`/`pv_index`/`pv_start`/`pv_end`), and a
`Box<dyn UciWriter>` for output. `ThreadData::new` is where the NUMA handle
resolution happens: `shared.history.get(numa_token)` and
`shared.parameters.get(numa_token)` (`thread.rs:246-249`) — a cheap `Arc`
clone out of a `RwLock<Vec<Arc<T>>>` indexed by the token's node index.

A small, generically useful type worth noting: `PlyArray<T, N>`
(`thread.rs:190-211`) indexes with a **signed** `isize` and a fixed `+8`
offset, so per-ply arrays (`cutoff_count`, `excluded`) can be indexed at
negative plies (needed for continuation-history look-back before ply 0)
without a separate bias constant scattered at every call site — the offset
lives once, in the `Index`/`IndexMut` impls.

### 2.3 `Counter` — single-writer sharded atomics (`thread.rs:73-113`)

```text
Counter { shards: Box<[AlignedAtomicU64]> }   // one #[repr(align(64))] shard per thread slot
increment(id):  shards[id] = shards[id].load(Relaxed) + 1   // plain load+store, NOT fetch_add
aggregate():    sum of all shards, loaded Relaxed
```

Two things worth flagging precisely:

1. **The shard count is `ThreadPool::available_threads()`** (up to 512, see
   §3.5), not the *current* thread count — so resizing the pool never
   needs to reallocate `Counter`; it's sized once for the worst case and
   indexed by `td.id` (`nodes.increment(td.id)` at `search.rs:1433`,
   `tb_hits.increment(td.id)` at `search.rs:419`).
2. **`increment` is a plain load-then-store, not `fetch_add`.** Because
   each shard is only ever written by the one thread whose `id` it belongs
   to, the non-atomic-RMW form is correct and cheaper (no `lock xadd`).
   This is *exactly* the optimization our own doc 13 §6 ("Pitfalls") already
   prescribes ("use the single-writer load+store bump, not `fetch_add`") —
   Reckless independently arrived at the same answer, which is a good
   confidence signal, not a new idea.

One minor oddity worth a footnote rather than an action item: `Counter`
carries an explicit `unsafe impl Sync for Counter {}` (`thread.rs:82`) even
though every field (`AlignedAtomicU64` wrapping an `AtomicU64`) is already
auto-`Sync` — the `unsafe` appears redundant. Not something to copy; if
anything it is a small counter-example to keep in mind for our own "no
unsafe in threads" bar (doc 13 §3.12 rule zero) — the discipline of *never*
reaching for `unsafe impl Send/Sync` and trusting the compiler to tell you
when a real problem exists is worth keeping as strict as we've written it.

### 2.4 `Status` (`thread.rs:115-142`)

A trivial `AtomicUsize` wrapper with two named constants
(`STOPPED = 0`, `RUNNING = 1`), `get` = `Acquire` load, `set` = `Release`
store. This is Reckless's equivalent of our `stop: AtomicBool` — one flag
instead of a bool, but the same role. Notably it is **not** `Relaxed`
(unlike our doc's explicit "Relaxed everywhere for all three flags"
argument, §3.11) — Reckless spends an acquire/release pair on every check.
Given `Status::get()` is read once per search node (`search.rs:315`), this
is a measurable (if small) cost our `Relaxed` choice avoids; nothing to
adopt here, just a data point that our reasoning in §3.11 holds up against
a real shipped alternative.

---

## 3. Thread pool lifecycle

### 3.1 Persistent worker threads + a rendezvous channel (`threadpool.rs:221-338`)

Reckless's `WorkerThread` is a real, long-lived `std::thread::JoinHandle`
created once (`make_worker_thread`, `threadpool.rs:318-333`) that loops on
`receiver.recv()` from a **zero-capacity** `mpsc::sync_channel(0)`
(`make_work_channel`, `threadpool.rs:249-257`) — a rendezvous channel: the
sender's `send` blocks until the worker is actually ready to receive, so
"submit a job" and "the thread is idle" are the same event, with no separate
busy-flag to check (contrast with our design's explicit `ThreadState{ job,
busy }` + `Condvar` pair, doc 13 §3.4, which needs the flag because a plain
`Mutex<Option<Job>>` can't block the *submitter* until the *worker* drains
it). A second, independent `Mutex<bool> + Condvar` pair (`completion_signal`,
`threadpool.rs:240,251`) signals **job completion** back to the submitter.
So each worker thread has two rendezvous points: one for "here is work" (the
channel itself), one for "work is done" (the completion condvar) —
functionally equivalent to our single job-slot protocol, split into two
primitives instead of one guarded struct.

**Termination is implicit**: dropping the channel's `Sender` (done in
`WorkerThread::join`, `threadpool.rs:228-231`) makes the worker's
`receiver.recv()` return `Err`, and the loop simply ends
(`threadpool.rs:321-330`). There is no explicit `Job::Exit` variant to
match on — closing the channel *is* the exit signal. Small but genuinely
nice idiom: one fewer enum case, one fewer `match` arm, and the invariant
("no more jobs will ever be sent") is expressed by a type-level action
(drop the sender) rather than by convention (send a magic value).

### 3.2 The generalized job type: "any job is a boxed closure"

Reckless does not have a `Job` enum with `Setup`/`Search`/`Clear` variants
the way our doc 13 §3.4 sketches. Instead, `WorkSender` carries
`Box<dyn FnOnce() + Send>` (`threadpool.rs:239`) — **every** dispatch,
whether it's "run the search," "construct this thread's `ThreadData` while
pinned to NUMA node N," or (implicitly, via the same mechanism) any future
"run this on thread i" need, is the *same* primitive: package a closure,
send it, join. This is precisely the generalization our doc 13 §9 (open
question 6) names as a deferred nice-to-have (`Job::Custom(Box<dyn
FnOnce(&mut Worker) + Send>)`) — Reckless demonstrates it is not a rare
extra, it is the *only* dispatch mechanism the pool has, used constantly
(every `set_count`/`clear` call constructs `ThreadData` through it, per
worker, in parallel — `threadpool.rs:341-379`).

**The catch: doing this without `Arc`-wrapping every piece of per-`go` state
costs one `unsafe` transmute.** `ScopeExt::spawn_into`
(`threadpool.rs:283-315`) lets a closure borrow data with lifetime `'scope`
from an enclosing `std::thread::scope`, then **erases that lifetime to
`'static`** with `std::mem::transmute` (`threadpool.rs:296-298`) before
sending it through the (necessarily `'static`) channel to the persistent
worker thread. The safety argument is a comment, not a compiler proof:
*"This file is structured such that threads never hold the data longer than
is permissible."* A `ReceiverHandle` is returned whose `Drop` **asserts** it
was actually joined (`threadpool.rs:264-281`) — a runtime tripwire, not a
static guarantee (it would not catch a `std::mem::forget`'d handle). The
surrounding `std::thread::scope(|scope| { ...; for handler in handlers {
handler.join(); } })` in `execute_searches` (`threadpool.rs:147-202`) does
always join before returning, so in practice the invariant holds — but the
proof lives in code discipline, not in the type system.

This is the one place in the threading layer that costs real `unsafe`, and
it is worth being explicit that **our design's "zero unsafe in `threads`"
bar (doc 13 §3.12 rule zero) is strictly stronger than what Reckless holds
itself to.** The good news: the *safe* half of doc 13's own deferred idea —
`Job::Custom(Box<dyn FnOnce(&mut Worker) + Send + 'static>)`, where the
closure receives `&mut Worker` as a **parameter** at call time (inside
`thread_main`, which already owns the `Worker` for its whole lifetime)
rather than **capturing** a borrow of it at spawn time — gets the exact
same "any job is a closure" generality with no transmute at all, because
nothing with a non-`'static` lifetime needs to cross the channel. See §7
(Adapt to us) for the concrete recommendation to promote this from
"deferred" to "do at P11."

### 3.3 `execute_searches` — the per-`go` orchestration (`threadpool.rs:82-203`)

Order of operations, native (non-wasm) path:

1. `shared.tt.increment_age()` — the TT generation bump, done by the
   **calling thread** (the UCI/message-loop thread), *before* any worker is
   dispatched. This is safe in Reckless's model specifically *because*
   `execute_searches` itself blocks until the previous search fully
   returned (§4) — there is no risk of bumping the generation while a
   stale `Arc<TT>` clone from a still-running previous search exists. This
   is a nice contrast worth internalizing: our design's more elaborate
   "generation counter must be interior-mutable, bumped via `&self` inside
   `start_searching`" story (doc 13 §3.7 step B, decisions.md D23
   amendment) is specifically the price of our **non-blocking** `go` — a
   thread could otherwise still be unwinding from the previous search when
   the age bump needs to happen. Reckless's simpler bump-before-dispatch
   is only correct because Reckless's `go` blocks (§4). Good validation
   that our extra complexity there is earning its keep, not gratuitous.
2. Reset shared counters (`nodes`/`tb_hits` `.reset()`), `soft_stop_votes`
   to 0, `status` to `RUNNING`, and seed **every** `best_stats[i]` entry
   from the main thread's `previous_best_score` (`threadpool.rs:88-94`) —
   see §5.1 for what that array is for.
3. Open one `std::thread::scope`, split `(main, helpers)` from both the
   `ThreadData` vector and the `WorkerThread` vector in lock-step, install
   fresh root state (board clone, freshly generated legal root moves, TB
   root ranking for the main thread only) into `t1`, clone that *exact*
   root-move list (same order) into every helper, then `spawn_into` the
   main thread's closure (which calls `search::start` then sets
   `Status::STOPPED` on return) and one closure per helper (`search::start`
   with `Report::None`), then join all.

Two small details worth citing exactly: `root_in_tb`/`stop_probing_tb` are
reset to `false`/`false` on **only** the main thread's path before its
closure runs (`threadpool.rs:155-156`) — helpers read them but never reset
them (single-writer-then-read discipline, same shape as our TT-generation
story). And syzygy root ranking (`tb::rank_rootmoves` /
`normalize_draw_ranks`) runs once, on the main thread, before the root-move
list is cloned to helpers — so helpers inherit ranked `tb_rank` fields for
free, rather than each re-deriving them.

### 3.4 Pool resize / clear semantics (`threadpool.rs:39-80`)

- `ThreadPool::new`: exactly **one** worker + one `ThreadData` (matches the
  UCI default `Threads = 1`); registers thread count 1 with the NUMA
  context.
- `set_count(n)`: clamp to `[1, available_threads()]`; tell the NUMA context
  the new thread count (which may trigger every registered `NumaReplicated`
  resource to re-replicate, §5.5); **fully tear down and rebuild**: join and
  drop every old `WorkerThread`, spawn `n` brand-new ones, drop every old
  `ThreadData`, reconstruct all `n` from scratch. There is no incremental
  add/remove — changing `Threads` mid-session always pays a full rebuild
  (and loses all warm history state), which is simple and matches the
  expectation that `Threads` changes are rare, not per-move.
- `clear()` (used by `ucinewgame`): keeps the existing `WorkerThread`s
  (no respawn) but rebuilds `ThreadData` for all of them — a lighter reset
  that matches our planned `Job::Clear`.

### 3.5 `Threads` UCI upper bound (`threadpool.rs:30-37`)

```text
available_threads() = max(4 * available_parallelism(), 512)   // 512 if available_parallelism() fails
```

This is a direct, ready-to-use answer to our own doc 12 §9.6 open question
("Threads option upper bound (1024) ... provisional"): rather than a flat
hardcoded ceiling (Stockfish hardcodes 1024 regardless of the machine),
Reckless scales the bound with detected hardware (4× logical cores) with a
512 floor so small machines still get a generous allowance for
oversubscription/testing. Landed deliberately per Reckless's own history
(commit titled "Adjust thread limit based on available logical CPU cores"),
not an accident.

---

## 4. UCI-thread responsiveness: a listener thread instead of non-blocking `go`

Our doc 13 spends real weight (§3.4 point 1, §3.6, §3.7 step D) justifying
persistent parked threads on the grounds that "**`go` must be non-blocking**
so the UCI thread can keep reading stdin for `stop`." Reckless's actual UCI
loop (`uci.rs:38-111`) takes a different, simpler shape: `go(...)`
**calls `execute_searches` synchronously and blocks** the message-loop
thread until the whole `std::thread::scope` (§3.3) joins. Responsiveness
during that block comes from a **second, dedicated thread**
(`spawn_listener`, `uci.rs:113-149`) that does nothing but
`stdin().read_line()` in a loop and special-cases exactly three commands —
`isready` (prints immediately), `stop`, `quit` — by writing directly to
`shared.status`/re-sending `"quit"`, **without ever touching the message
loop's channel**. Every *other* line is forwarded through an `mpsc` channel
to the message loop only when `status != RUNNING` (i.e. dropped silently
while a search is in flight, per the UCI spec's own allowance that
"commands unexpected in the current state should be ignored").

So the architecture is really three roles, not our doc's two:

```text
stdin-listener thread:  read_line loop; stop/isready/quit handled inline
                        against shared atomics; everything else forwarded
                        via channel, but only when not RUNNING
message-loop thread:    reads from that channel; dispatches go/position/
                        setoption/etc.; BLOCKS inside `go`
N persistent workers:   the actual search, dispatched via §3.1/§3.2
```

This sidesteps essentially all of our doc's non-blocking-`go` machinery
(no `Setup` job to install root state before waking, no busy-wait-while-
pondering discipline for the message loop specifically, no need for
`bestmove` to be printed from inside a worker thread — it's printed by the
message-loop thread right after `execute_searches` returns). The trade-off
is real, though: **Reckless has no `ponder` support at all** (confirmed —
no occurrence of "ponder" anywhere in the source), and the blocking design
is part of why that is easy to leave out; a `go ponder` / `ponderhit` /
"never answer while pondering" story is exactly the kind of thing that
would force the listener thread to start reasoning about search state,
eroding the simplicity. Since we already support ponder in our design (doc
13 §3.7 step D, §5 test plan), the non-blocking architecture is the correct
choice for us, not a needless complication — this is presented here as a
validated alternative for a ponder-free engine, not as a suggested rewrite.

---

## 5. Cross-thread signals beyond the TT

Both Stockfish and our doc 13 (§3.9, "what we deliberately do NOT
synchronize") treat the TT plus two or three tiny flags as the *entire*
inter-thread channel during a lazy-SMP search. Reckless adds two more,
small, purpose-built shared signals that are worth understanding in detail
because they are genuinely novel relative to both references.

### 5.1 `best_stats`: cross-thread optimism fusion via atomic `fetch_max`

`SharedContext::best_stats` is `[AtomicU32; MAX_MOVES]`, one slot per
MultiPV index (`thread.rs:164`). At the top of every aspiration-window
attempt (`search.rs:120-124`), before searching, each thread reads its
slot, blends it with its own running average, and derives an "optimism"
bias fed into evaluation:

```text
packed = best_stats[pv_index].load()            // (depth << 16) | (score + 32768)
best_avg = ((packed & 0xffff) as i32 - 32768 + own_average) / 2
optimism[side_to_move]  = 113 * best_avg / (|best_avg| + 201)
optimism[other_side]    = -optimism[side_to_move]
```

And once a thread's own aspiration attempt *succeeds* (score lands strictly
inside the window), it publishes back with a **`fetch_max`**, not a plain
store (`search.rs:164-167`):

```text
best_stats[pv_index].fetch_max((this_depth << 16) | (this_average + 32768))
```

Packing `depth` into the high 16 bits means `fetch_max` naturally prefers
whichever thread has reached the **greatest depth so far**, and only
compares scores among threads tied on depth — a single atomic register
acting as a lock-free "best conclusion any thread has reached at this
MultiPV index" register, self-cleaning every `go` (reseeded from
`previous_best_score` at the start of `execute_searches`,
`threadpool.rs:92-94`). Reckless's own history names this precisely
("Use highest depth thread to adjust optimism") — it was added and kept,
i.e. it measured as worth it.

This is a strictly bigger idea than "share the TT": it is deliberate,
narrow, cheap (4 bytes × MultiPV, not NUMA-replicated at all — small enough
that cross-node traffic doesn't matter) cross-pollination of an
*evaluation-bias* signal across the whole lazy-SMP pool, on top of the
classic TT-only osmosis. Worth the exact formula, and worth noting it is
**not** NUMA-replicated (unlike NNUE weights/correction history) — a good
illustration of the underlying principle: replicate what is large and/or
frequently written enough for cross-node latency to matter; a `MAX_MOVES`-
sized `u32` array touched a few hundred times per `go` is not that.

### 5.2 `soft_stop_votes`: consensus soft-stop across *all* threads, not just main

Both Stockfish and our doc 13 model the soft-stop decision (the "should we
start another ID iteration" check) as **main-thread-only**: main computes
the four instability factors, decides, and if so sets the shared `stop`.
Reckless's soft-stop is instead a **live, retractable quorum vote cast by
every thread**, main and helpers alike (`search.rs:265-284`, no `id == 0`
guard anywhere in this block — contrast with the *hard*-stop check at
`search.rs:336`, which *is* gated to `id == 0`). Each thread evaluates its
own `time_manager.soft_limit(td, multiplier)` using **its own** per-thread
stats (its own `pv_stability`, `eval_stability`, its own node count vs. its
own `root_moves[0].nodes`, its own `best_move_changes`) — there is no
cross-thread aggregation of these inputs at all:

```text
if this_thread_wants_to_stop() {
    if not already_voted {
        already_voted = true
        votes = soft_stop_votes.fetch_add(1) + 1
        if votes >= ceil(thread_count * 0.65) { status = STOPPED }
    }
} else if already_voted {
    already_voted = false
    soft_stop_votes.fetch_sub(1)
}
```

The vote is **retractable**: a thread that voted to stop but then, on the
next ID iteration, no longer meets its own soft-limit condition (e.g. the
score just dropped and `eval_stability` reset) takes its vote back before
the next thread's check. Global stop only fires when **65% of threads
simultaneously agree** it's fine to stop. This is a materially different
philosophy from "one thread manages the clock, others just search until
told to stop" — it makes soft-stop itself a distributed, self-correcting
consensus computed from N independent (and by construction, differently-
diverged, since each thread's search tree differs) opinions rather than one
canonical opinion. Reckless's own history names this too ("Implement
collective thread voting for soft stop") — again landed and kept.

---

## 6. Thread-diversity seed: reduction jitter instead of aspiration jitter

Our notes on Stockfish (and our own doc 13 §3.1) record the classic
lazy-SMP diversity trick as living in the **aspiration window**: initial
half-width includes a `thread_idx % 8` term so different threads search
slightly different windows from move 1. Reckless's aspiration delta
(`search.rs:98,115`) has **no thread-index term at all** — every thread
computes the same initial `delta` from the same `average` value. Instead,
the `id`-seeded diversity lives one level deeper, inside the **LMR
reduction formula itself**, added once in the zero-window branch and once
in the full-depth-search branch (`search.rs:895` and `search.rs:957`):

```text
reduction += ((this_thread_node_count + thread_id * C) mod 128) - K
```

with `C = 27, K = 59` in one branch and `C = 26, K = 56` in the other — a
small (roughly ±60, against reduction terms in the hundreds-to-thousands
range in the same fixed-point units) pseudo-random nudge that is seeded by
**both** the thread's own ever-growing node counter (so it changes every
node) **and** the thread id (so two threads at the same local node count
still diverge). Two things stacked into one cheap expression: it smooths
the otherwise-quantized integer reduction ladder (a known technique for
avoiding "staircase" artifacts in LMR), and it is — as far as these sources
show — Reckless's *only* deliberate SMP-diversity seed. (Natural divergence
still also comes from TT arrival-order races and the `best_stats`/
`soft_stop_votes` signals in §5, same as any lazy-SMP engine.)

Note this formula itself belongs to the pruning/reductions doc (ours would
be design/10), not this one — it's flagged here purely for the `thread_id`
plumbing, which is this doc's business.

---

## 7. Best-thread voting (`uci.rs:188-249`)

Runs on the message-loop thread after `execute_searches` returns (so, after
every worker has already returned from `search::start` and — for the main
thread — set `Status::STOPPED`; no explicit "wait for reports" step is
needed because `execute_searches`'s own `scope` join already guarantees
every thread finished).

```text
min_score = min over all threads of root_moves[0].score
vote_value(thread) = (thread.root_moves[0].score - min_score + 10) * thread.completed_depth
votes[move] = sum of vote_value(thread) over threads whose best move == move

if fixed-depth search, or MultiPV > 1:
    winner = thread 0                                   // no voting at all
else:
    winner = thread 0
    for each other thread `c`:
        if winner is a trusted win:      c replaces winner only if c is a trusted win with a *larger* score
        elif winner is a trusted loss:   c replaces winner only if c is not −∞ and has a *smaller* score
        elif c is decisive (and valid):  c replaces winner
        else:                            c replaces winner if c is not a loss and
                                          (votes[c] > votes[winner], or tie broken by vote_value)
```

Differences worth naming precisely against our doc 13 §3.8 (which mirrors
Stockfish's `get_best_thread`, flat `score - min + 14` votes):

- **Votes are weighted by `completed_depth`**, not flat. A thread that only
  reached depth 12 counts for less than one that reached depth 20 at the
  same score margin. Our doc's formula has no depth term at all — this is
  a concrete, cheap variant worth an isolated experiment (§8).
- **Voting is disabled outright under `MultiPV > 1`**, in addition to fixed
  depth. Our doc 13 §3.8 step 1 only names `n_threads == 1` and
  `limits.depth` — the MultiPV case isn't mentioned. Worth adding
  explicitly: voting across multiple simultaneously-tracked PV lines has no
  clean semantics, and Reckless's blanket "thread 0 always wins" sidesteps
  the question entirely.
- The decisive-protection check is written as two explicit branches (best
  is a win → only a bigger score wins; best is a loss → only a smaller
  score wins) rather than a single `is_trusted_decisive` predicate with a
  `|score|` comparison — same effect, different shape. One thing to
  double-check if we ever compare bit-for-bit: Reckless's vote loop reads
  `root_moves[0].score` directly and does not appear to exclude
  bound-flagged (aborted-iteration) mate scores the way our doc's
  `is_trusted_decisive` explicitly does (doc 13 §3.8 step 4) — our stricter
  check is likely the safer one to keep as specified.

---

## 8. The NUMA subsystem (`numa.rs`, all 385 lines)

### 8.1 Topology discovery (`numa.rs:66-256`)

`NumaConfig::from_system()` is Linux-only (other platforms silently get
`NumaConfig::default()`, a single node holding every reported logical CPU,
`numa.rs:66-74`). On Linux it reads `/sys/devices/system/node/online` for
the list of node ids, then `/sys/devices/system/node/node{N}/cpulist` per
node (`numa.rs:238-251`), **intersected with the process's own CPU affinity
mask** obtained via `sched_getaffinity` (`numa.rs:30-46`, cached in a
`LazyLock<BTreeSet<CpuIndex>>`) — so a process already confined by
`taskset`/a cgroup only ever sees the CPUs it's actually allowed to use, not
the whole machine's topology. Any read failure (missing sysfs, permission,
non-Linux) falls back to the single-node default; empty nodes are pruned
(`remove_empty_numa_nodes`). All of this runs once, at `SharedContext`
construction — never on a hot path.

### 8.2 The "is it even worth it" gate (`numa.rs:106-120`)

```text
requires_memory_replication()  =  more than one NUMA node exists
suggests_binding_threads(n)    =  requires_memory_replication()
                                   and n > 1
                                   and ( n > largest_node_cpu_count / 2
                                         or n >= 4 * sufficient_node_count )
    where a node counts as "sufficient" if its CPU count is > 60% of the
    largest node's CPU count
```

This is the mechanism that makes NUMA-awareness **cost nothing** on the
overwhelmingly common case (a single-socket desktop/laptop, or a
multi-socket box asked for only a handful of threads): both flags collapse
to `false`, every `NumaReplicatedAccessToken` is node 0, and
`distribute_threads_among_numa_nodes` degenerates to `vec![0; n]`
(`numa.rs:122-124`). Binding/replication only engages once the thread count
is large enough, relative to the actual node sizes, that spreading threads
across sockets and giving each socket its own copy of hot data plausibly
pays for itself. Nothing here is guessed per-run — it's a pure function of
already-known topology and the requested thread count, computed once per
`set_count`/`clear`.

### 8.3 Distributing threads across nodes (`numa.rs:122-147`)

A simple greedy proportional-fill assignment, one thread at a time: for
each thread to place, pick the node minimizing `(occupied_on_node + 1) /
node_cpu_count` — i.e. the node currently *least full, relative to its own
size*. This correctly load-balances across **asymmetric** topologies (a
16-core node and an 8-core node get threads in roughly 2:1 proportion, not
split evenly), with no special-casing beyond the one formula.

### 8.4 Pinning mechanics (`numa.rs:149-190`)

`bind_current_thread_to_numa_node`: build a `cpu_set_t` mask from that
node's CPU set, `sched_setaffinity(0, ...)`, then an immediate
`sched_yield()` (`numa.rs:154-168`) — the yield right after the affinity
change is a small, concrete, worth-remembering detail: it nudges the
scheduler to actually migrate the calling thread *before* the next line of
code runs, so the very next heap allocation on that thread is much more
likely to physically land on the target node's RAM (first-touch). Node-
bound allocation itself (`execute_on_numa_node`, `numa.rs:174-190`) spawns a
short-lived thread, pins it, runs the allocator closure, joins, and hands
the result back over a channel — a disposable helper thread whose only job
is to make one allocation happen on the right node.

### 8.5 The generic replication mechanism (`numa.rs:16-22, 285-385`)

```text
trait NumaReplicable:
    fn allocate() -> Arc<Self>                    // node-bound instance
    fn allocate_shared() -> Option<Arc<Self>>      // optional cheap fallback (default: None)

NumaReplicationContext:
    holds the NumaConfig, the current thread count, and a registry
    (Vec<Arc<dyn NumaReplicatedBase>>) of every replicated resource that
    has ever asked to be tracked

NumaReplicated<T: NumaReplicable>:
    holds Vec<Arc<T>>, one entry per NUMA node currently in use
    get(token) = clone the Arc at token's node index
    replicate_instances():
        if suggests_binding_threads(current_thread_count):
            one T::allocate() per node, each built by a thread pinned to
            that node (first-touch)
        elif T::allocate_shared() returns Some(shared):
            use that single cheap instance for every thread
        else:
            one T::allocate() built on node 0
```

`NumaReplicationContext::set_thread_count` (`numa.rs:313-323`) is an
**observer pattern**: any registered `NumaReplicated<T>` gets told
`on_numa_config_changed()` and re-runs `replicate_instances()` whenever the
thread count actually changes. This means a resource only has to implement
two small methods (`allocate`, optionally `allocate_shared`) and register
itself once; it never has to know *when* to re-replicate — the context
tells it. The one dynamic-dispatch cost (`Vec<Arc<dyn NumaReplicatedBase>>`)
is paid only at pool-reconfiguration time (a UCI `setoption Threads` /
`ucinewgame` event), never per search, per node, or per anything hot — a
textbook "trait objects for rare/administrative operations, monomorphized
code for hot paths" split.

### 8.6 What actually gets replicated, and the trick that makes the common
case free

- **`SharedCorrectionHistory`** (`thread.rs:144-154`): implements only
  `allocate()` (fresh zeroed tables), no `allocate_shared` override — so in
  the non-NUMA common case it still goes through `allocate_on_node(0)`
  (one throwaway pinned-allocation thread at replication time, negligible).
  Its internal storage is `Box<[[[AtomicI16; 65536]; 2]; 16]>`
  (`history.rs:159-161`), and — critically — **`update` is a plain relaxed
  load-then-store, not a compare-and-swap loop**, even though this table is
  genuinely written concurrently by every sibling thread sharing a node
  (`history.rs:174-178`). This is the *same* "tolerate a rare lost update,
  the statistic self-corrects" philosophy our own TT design already commits
  to (design/08, D19) — applied here to a second, non-TT shared table.
  Good confirmation that the pattern generalizes cleanly: any table you
  decide to share across threads (node-local or global) can reuse the
  TT's relaxed-atomics-and-shrug idiom rather than needing a lock or CAS
  loop, as long as the update formula (gravity-style, bounded) tolerates
  an occasional torn read/write.
- **NNUE `ParametersHandle`** (`nnue.rs:387-427`): a small enum,
  `Embedded(&'static Parameters)` or `Owned(Arc<Parameters>)`, behind a
  `Deref`. It **does** override `allocate_shared()` to just wrap the
  already-`'static` embedded network reference in an `Arc` — **zero bytes
  copied** in the common (non-NUMA) case, since the compiled-in network is
  already sitting in read-only static memory shared by every thread for
  free. Only when real cross-node replication is warranted does `allocate()`
  pay the cost of a full heap copy of the (multi-megabyte) weight struct,
  made via an uninitialized-`Box` + `copy_nonoverlapping` dance
  (`nnue.rs:375-384`) specifically to avoid transiting the huge struct
  through a stack frame — and that copy happens *while the allocating
  thread is pinned to the target node* (§8.4), so the copy's physical pages
  land locally.
- **Everything else stays per-thread**, and gets first-touch placement
  *for free*, with no `NumaReplicated` wrapper at all — because
  `ThreadData::new` is called from *inside* the (potentially node-pinned)
  worker-thread closure (`threadpool.rs:341-379`: bind, *then* construct).
  Every per-thread heavy table (continuation history, quiet/noisy/pawn
  history, the PV table, the NNUE accumulator stack) is simply allocated by
  the thread that will use it, which already happens to be running on the
  correct node. This is precisely what our own doc 13 §3.2/§3.4 already
  predicts ("a worker's tables are allocated and cleared by the thread that
  uses them — first-touch placement for free") — strong, independent
  confirmation that our planned "construct `Worker` inside the spawned
  thread's closure" ordering is the right call and needs no NUMA-specific
  code to benefit from it later.

### 8.7 A non-Linux / wasm aside

The whole NUMA layer degrades to a single implicit node on non-Linux
targets (`numa.rs:87-89`), and the crate additionally compiles to `wasm32`
with a parallel, from-scratch thread-pool implementation
(`thread.rs:9-18`, `threadpool.rs:96-144,381-398`) that has no real OS
threads at all — work is dispatched to JS Web Workers through a
`wasm-bindgen` callback and awaited with a spin-loop on a shared
`WORKERS_REMAINING` counter. Interesting as evidence the abstraction
(`ThreadPool`/`ThreadData` as platform-agnostic containers) holds up across
a genuinely different execution model, but not something we need — noted
for completeness, not for adoption.

---

## 9. Distinctive vs Stockfish (summary)

- **Third state tier** (per-NUMA-node-shared, not just global-shared vs.
  per-thread) is present in current Stockfish too (per our own doc 09 §9.5
  / doc 04 notes: "SF shares pawn/correction/continuation histories per
  NUMA node"), but Reckless's cut is **narrower** — only correction history,
  not continuation or pawn history.
- **No `thread_idx % 8` aspiration jitter** — diversity is instead seeded
  inside the LMR reduction formula (§6), a different layer entirely.
- **Soft-stop is a live 65%-quorum vote across every thread** (§5.2), not a
  main-thread-only decision — this is not present in Stockfish's model at
  all as far as our notes record.
- **Cross-thread optimism fusion via a packed `(depth, score)` atomic
  `fetch_max`** (§5.1) — again, not part of the classic lazy-SMP recipe;
  a genuine Reckless-original addition on top of "just share the TT."
  Best-thread votes are **depth-weighted** (§7), where Stockfish's/our
  doc's formula is flat.
- **The generalized "job = boxed closure" dispatch** (§3.2) is more elegant
  and more heavily used than Stockfish's `run_custom_job` (used for a
  handful of special cases in SF); Reckless makes it the *only* dispatch
  path, at the cost of one `unsafe` transmute to make the lifetimes work
  with persistent threads.
- **UCI responsiveness via a privileged listener thread + blocking `go`**
  (§4) rather than SF's non-blocking `start_thinking` + main-thread
  busy-wait-while-pondering — simpler, but only viable because Reckless has
  no ponder support.

## 10. Distinctive vs our design/13 (and 09 §9.5, 12 §9.6) (summary)

- Our `available_threads()`/`Threads` upper-bound open question (doc 12
  §9.6) has a concrete, working answer in `threadpool.rs:30-37` (§3.5).
- Our deferred NUMA section (doc 13 §3.15) and deferred NUMA-shared-history
  question (doc 09 §9.5) both have a complete, working reference
  implementation to measure against (§8) — including the specific
  "is it even worth it" gate (§8.2) we hadn't designed a formula for yet.
- Our deferred `Job::Custom` idea (doc 13 §9, open question 6) is validated
  as high-value and heavily used by Reckless (§3.2) — but Reckless's
  version needs `unsafe`; ours doesn't have to (§7 below).
- Our single-writer counter-bump optimization (doc 13 §6) is independently
  confirmed correct and worth keeping (§2.3).
- Our `Relaxed`-everywhere flag-ordering argument (doc 13 §3.11) is *not*
  what Reckless does (`Status` uses Acquire/Release, §2.4) — a data point
  in our favor, not a gap.
- Two mechanisms Reckless has that our doc doesn't mention at all: the
  cross-thread optimism signal (§5.1) and the multi-thread soft-stop quorum
  (§5.2) — both genuinely new territory for us, flagged for design
  discussion in §11 below, not silently adopted.

---

## 11. Adapt to us

Ordered roughly by effort, cheapest first. Every item that touches search
*behavior* (not just plumbing) needs the fixed-node proxy-SPRT gate
(doc 14 §9.4) before it's considered validated, per our own process.

1. **`Threads` upper bound formula.** Replace the provisional flat `1024`
   (doc 12 §9.6) with `max(4 * available_parallelism, 512)`. Touches: doc
   12 §9.6, doc 13 (wherever `available_threads()`-equivalent lives).
   Phase: P11. Effort: trivial (one function, no behavior change to search
   itself, no SPRT needed — purely a UCI-option ceiling). Impact: small UX
   correctness win (scales with hardware instead of a magic constant).

2. **Channel-close-as-exit instead of `Job::Exit`.** If we end up
   implementing the job dispatch via a channel (rather than
   `Mutex<Option<Job>> + Condvar` as currently drafted, doc 13 §3.4), drop
   the sender to terminate the worker instead of adding an explicit `Exit`
   variant. Touches: doc 13 §3.4. Phase: P11. Effort: trivial. Impact: one
   fewer enum case; purely stylistic.

3. **Hard-stop cadence via a bitmask on the node counter, split from the
   node-limit check.** Replace the planned `calls_cnt` countdown field
   (doc 13 §3.3) with `this_thread_nodes() & 2047 == 2047` for the
   wall-clock cadence, and keep the aggregate node-limit check as a
   *separate*, unconditional branch (evaluated every node only when
   `Limits::Nodes` is the active mode) rather than folding both into one
   adaptive reset formula. Touches: doc 13 §3.3 ("check_time cadence"),
   doc 12 (node-limit handling). Phase: P11. Effort: trivial (removes a
   mutable field). Impact: small simplification; no behavior change if the
   node-limit branch is kept separate as described.

4. **Promote `Job::Custom` from "deferred" to "build at P11," using the
   safe signature.** Our doc 13 §9 open question 6 already names
   `Job::Custom(Box<dyn FnOnce(&mut Worker) + Send>)` as a possible future
   addition; Reckless shows this is not a rare extra but the natural,
   constantly-used shape for *all* dispatch (search, per-thread
   construction, and later parallel TT clear / NUMA work all become the
   same primitive). The key adaptation: have the closure **receive**
   `&mut Worker` as a parameter when `thread_main` invokes it (since
   `thread_main` already owns the `Worker` for the thread's whole
   lifetime), rather than **capturing** a borrow of `Worker`/root state at
   spawn time the way Reckless's `spawn_into` does — this gets the same
   generality with zero `unsafe`, preserving doc 13 §3.12 rule zero.
   Touches: doc 13 §3.4, §9 open question 6. Phase: P11 (step 1, pool
   skeleton). Effort: small (one enum variant + one call site change from
   a fixed `Job::Setup/Search/Clear` enum to `Custom` covering all three).
   Impact: cleaner code, and it's the natural vehicle for parallel TT clear
   and any future NUMA-bound-construction work, so it pays for itself
   later even if it saves nothing today.

5. **Add "MultiPV > 1 disables voting" to the best-thread voting spec.**
   Concrete gap in doc 13 §3.8 step 1 (currently only names
   `n_threads == 1` and `limits.depth`). Touches: doc 13 §3.8. Phase: P11.
   Effort: trivial (one extra condition + one test-matrix case). Impact:
   avoids ill-defined semantics; essentially free correctness/clarity fix.

6. **Depth-weighted best-thread votes as an experiment.** Try
   `vote_value = (score - min_score + VOTE_TEMPO) * completed_depth` in
   place of the current flat formula (doc 13 §3.8). Touches: doc 13 §3.8,
   `search/params.rs` (`VOTE_TEMPO`). Phase: P11. Effort: small (one
   multiply, update the voting unit-test matrix). Impact: unproven for us —
   flag as a proxy-SPRT candidate once we have real multi-thread games to
   test with, not a drop-in assumption.

7. **NUMA subsystem: port the *mechanics*, gate the *behavior* behind
   measured need.** Concretely reusable pieces, all cheap and inert until
   `suggests_binding_threads` returns true: topology discovery via sysfs +
   affinity intersection with single-node fallback (§8.1), the
   worth-it gate formula (§8.2), the proportional-fill node distribution
   (§8.3), and the generic `NumaReplicable`/`NumaReplicated`/
   `NumaReplicationContext` observer-registration mechanism (§8.5) as a
   *design pattern* even before real multi-socket pinning lands — it cleanly
   expresses "shared vs. per-thread" as a runtime policy per table instead
   of a hardwired choice. **Caveat to flag honestly**: `sched_getaffinity`/
   `sched_setaffinity` need either the `libc` crate or raw syscalls, which
   is a real tension with our stated "zero runtime dependencies so far"
   convention (CLAUDE.md) — recommend treating full pinning as opportunistic
   (build it only once we have SPRT access to genuine multi-socket
   hardware and can measure a win), while the topology-detection +
   replication-registry *pattern* can land dependency-light and inert on
   everyone's single-socket dev machine. Touches: doc 13 §3.15 (currently
   "explicitly deferred" — this turns it into a scoped, gated plan rather
   than a blank deferral), doc 09 §9.5. Phase: P11 (mechanism) / post-P11
   (actual pinning, need-gated). Effort: medium (self-contained module,
   no hot-path interaction) for the mechanism; small-to-medium for
   pinning itself if/when a target machine justifies it. Impact: zero until
   multi-socket hardware is in the loop; potentially real on it (this is
   exactly what Reckless's own "Overhaul NUMA support" commit landed for).

8. **Pilot global-shared correction history *without* NUMA, as a cheaper
   first step.** Doc 09 §9.5 already flags "SF/Reckless share
   pawn/correction/continuation-family tables per NUMA node; we ship
   per-thread, revisit at the SMP-scaling phase." Since most of our own
   SPRT/dev hardware is single-socket (where "per NUMA node" collapses to
   "global"), we can test the *value* of sharing correction history across
   *all* threads (one instance, `AtomicI16` entries, same relaxed
   load-then-store gravity update as `history.rs:174-178` — reusing the
   exact tolerated-race idiom our own TT already commits to, D19) as a
   standalone P11 experiment, fully decoupled from real NUMA-topology code.
   Only split it per-node later if we ever have multi-socket SPRT capacity
   and the global-shared version already proved out. Touches: doc 09 §9.5,
   §3.4 (table catalogue: `corr` table becomes optionally-shared), doc 13
   (needs the thread-count-aware allocation seam). Phase: P11. Effort:
   small-medium (mostly changing `StatEntry`'s storage for the correction
   table to atomic and adding one shared-vs-per-thread switch). Impact:
   unknown, worth an isolated fixed-node/fixed-time SPRT — this is exactly
   the kind of thing doc 14 §9.4's proxy-SPRT gate exists for.

9. **Cross-thread optimism fusion (`best_stats`-style), once multi-thread.**
   Extend our already-planned single-thread optimism formula (doc 11
   "Optimism", cross-referenced from doc 07/13's aspiration-window setup)
   with a small shared `[AtomicU32; MAX_MOVES]` (or per-MultiPV-index)
   register holding a packed `(depth, score)` value, updated via
   `fetch_max` and blended into each thread's own average the way §5.1
   describes. Not NUMA-replicated — small enough that it doesn't need to
   be. Touches: doc 11 (optimism formula), doc 13 (needs the new shared
   array + `thread_count`/MultiPV plumbing). Phase: P11 (meaningless before
   multiple threads exist). Effort: small (one atomic array + one
   `fetch_max` + one blend). Impact: unproven for our constants, but
   Reckless's own commit history ("Use highest depth thread to adjust
   optimism") suggests it measured as a real, kept improvement for them —
   good proxy-SPRT candidate at P11.

10. **Soft-stop-by-quorum as a design discussion, not yet a commitment.**
    The biggest, riskiest idea here: replace (or augment) our planned
    main-thread-only soft-stop decision (doc 12, doc 13 §3.7 step D) with
    a live, retractable vote across all threads using **each thread's own**
    per-thread stability stats, gated by a majority threshold (Reckless
    uses 65%). This is a bigger design change than the others — it means
    every thread needs its own full soft-stop-factor computation (not just
    main), and it changes what "stop" means from "one authority decided"
    to "a supermajority independently agrees." Touches: doc 12 (soft-stop
    factors, currently main-thread-scoped), doc 13 §3.7 step D, §3.10.
    Phase: P11, but treat as a research question first — recommend writing
    up the trade-off in doc 12/13 as an open question with this note as the
    reference, then validating with a fixed-time SPRT (`Threads=N`,
    quorum vs. main-only) before committing either way. Effort: medium
    (touches the soft-stop contract in two docs and needs new per-thread
    state). Impact: unclear — could reduce variance/forfeits at high
    thread counts, or could just add complexity for a similar outcome;
    this is exactly the kind of claim doc 13 §8 criterion 9 ("fixed-time
    SPRT... shows the expected positive Elo gap") already commits us to
    measuring rather than assuming.

---

## 12. File:line quick index

- `thread.rs`: `Counter` 73-113 (82 the redundant `unsafe impl Sync`,
  93-95 load+store bump, 104-112 sized to `available_threads()`);
  `Status` 115-142; `SharedCorrectionHistory` 144-154; `SharedContext`
  156-188; `PlyArray` 190-211; `ThreadData` 213-284 (246-249 NUMA-token
  resolution); `RootMove` 386-417; `PrincipalVariationTable` 419-457.
- `threadpool.rs`: `available_threads` 30-37; `ThreadPool::new` 39-46;
  `set_count` 48-59; `clear` 73-80; `execute_searches` 82-203 (86 age
  bump, 92-94 `best_stats` reseed, 146-202 native scope/spawn/join);
  `WorkerThread`/`join` 220-232; `WorkSender`/`WorkReceiver`/
  `make_work_channel` 237-257; `ReceiverHandle` 259-281; `ScopeExt::
  spawn_into` (the `unsafe transmute`) 283-315; `make_worker_thread`
  317-333; `make_thread_data` (NUMA binding + first-touch `ThreadData`
  construction) 340-379; wasm32 alternates 96-144, 381-398.
- `numa.rs`: `NumaReplicable` 16-22; affinity/`SYSTEM_THREADS` 27-46;
  `NumaConfig` 59-256 (106-120 `suggests_binding_threads`, 122-147
  `distribute_threads_among_numa_nodes`, 149-172 pin + `sched_yield`,
  174-190 `execute_on_numa_node`, 215-256 sysfs parsing); `parse_cpu_indices`
  258-279; `NumaReplicationContext` 290-328; `NumaReplicated<T>` 330-385.
- `nnue.rs`: `Parameters` 356-367; `embedded`/`allocate_owned` 369-384;
  `ParametersHandle` + `NumaReplicable` impl 387-427.
- `history.rs`: `HugeBox` 1-13 (mmap + `MADV_HUGEPAGE`); `CorrectionHistory`
  159-189 (170-178 relaxed load-then-store `update`, the racy-but-tolerated
  gravity write).
- `search.rs`: `id == 0` gates 79, 237, 336, 1199 (hard-stop / depth-limit /
  mate-limit are main-only); soft-stop vote block 265-284 (no `id` gate);
  `best_stats` read/optimism 120-124, write 164-167; reduction jitter 895,
  957; `nodes`/`tb_hits` per-thread increment 419, 1433.
- `time.rs`: `soft_limit` 75-82 (per-thread multiplier input); `check_time`
  84-94 (`nodes() & 2047 == 2047` cadence).
- `uci.rs`: `message_loop` 38-111; `spawn_listener` 113-149; `go` +
  best-thread voting 188-249.
