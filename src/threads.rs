//! Threading — lazy SMP in safe Rust (`design/13-threading-smp.md`).
//!
//! A pool of long-lived, parked worker threads (D23). Each thread owns a
//! [`Worker`] (its board copy, search stack, histories, and — for thread 0
//! only — the main-worker [`SearchManager`], `design/07`), constructed on the
//! thread itself for first-touch locality (§3.4). Threads park in
//! [`thread_main`]'s idle loop and are woken with [`Job`]s through a per-thread
//! [`ThreadShell`] (`Mutex<ThreadState>` + `Condvar`, exactly SF's
//! `searching`-flag protocol re-expressed in safe Rust).
//!
//! The shared surface is deliberately tiny (§3.1): the transposition table
//! (`Arc<TT>`, all-atomic, D19), three [`AtomicBool`] flags in [`PoolShared`]
//! (`stop`/`increase_depth`/`ponder`), and per-thread atomic [`Counters`].
//! Everything mutable a search touches — board, root moves, histories,
//! reduction table — is single-owner inside the `Worker` and is never
//! observable from another thread (D23). **No `unsafe`, no `Rc`/`RefCell`, and
//! no lock on the per-node search path** (§3.11–§3.13): the mutexes here are
//! touched only at job boundaries (a handful of times per `go`).

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::board::Board;
use crate::search::{SearchReport, UciSink, Worker};
use crate::timeman::Limits;
use crate::tt::TranspositionTable;
use crate::uci::Options;

/// Per-thread stack size (§3.4). Search recursion reaches `MAX_PLY` with fat
/// frames; SF forces 8 MiB likewise. The main *process* thread only parses
/// UCI, so its default stack is untouched.
pub const WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Counters (§3.10)
// ---------------------------------------------------------------------------

/// Per-thread atomic counters, cache-line isolated so helpers never
/// false-share (§3.10, §6). `nodes`/`tb_hits` are written by their owning
/// worker only, so the cheap single-writer load+store bump is correct and
/// preferred (no `lock xadd`); they are read by any thread. `best_move_changes`
/// is the one counter the main worker also *writes* (an atomic `swap(0)`
/// harvest), so its owner bump uses `fetch_add` — a genuine atomic RMW — and
/// the harvest then loses no increment even when the two race (§3.10).
#[repr(align(64))]
pub struct Counters {
    pub nodes: AtomicU64,
    pub tb_hits: AtomicU64,
    pub best_move_changes: AtomicU64,
}

impl Counters {
    pub fn new() -> Counters {
        Counters {
            nodes: AtomicU64::new(0),
            tb_hits: AtomicU64::new(0),
            best_move_changes: AtomicU64::new(0),
        }
    }

    /// Zero every counter (each `Setup`, §3.6 step 4). Owner-only, so plain
    /// relaxed stores suffice.
    pub fn reset(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.tb_hits.store(0, Ordering::Relaxed);
        self.best_move_changes.store(0, Ordering::Relaxed);
    }

    /// Bump `best_move_changes` (§3.10): a non-first root move became best this
    /// iteration. Unlike `nodes`/`tb_hits`, the main worker *also* writes this
    /// counter — it harvests every thread's count with an atomic `swap(0)` — so
    /// the owner bump must be a real atomic RMW (`fetch_add`). A plain
    /// load+store could interleave with the harvest and lose or double-count
    /// (owner loads `v`; main `swap(0)`s away `v`; owner stores `v+1`,
    /// re-seeding the already-harvested `v`); `fetch_add` vs `swap` are both
    /// RMWs on one location, so every increment is accounted exactly once. The
    /// extra locked op is negligible: this fires at most once per root move per
    /// iteration, never on the per-node path.
    pub fn bump_best_move_changes(&self) {
        self.best_move_changes.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for Counters {
    fn default() -> Counters {
        Counters::new()
    }
}

// ---------------------------------------------------------------------------
// Jobs (§3.4)
// ---------------------------------------------------------------------------

/// Per-`go` state installed on the worker's own thread by a [`Job::Setup`]
/// (§3.6 step 4). Board is cloned WITH its game history (D8) so each worker's
/// repetition detection is self-contained.
pub struct SetupJob {
    pub board: Board,
    pub limits: Limits,
    /// The full options snapshot (main uses it for `TimeManager::init`;
    /// helpers ignore it). Small and `Send`, cloned once per worker per `go`.
    pub options: Options,
    /// Type-erased output sink (main worker only writes through it, §3.7 G).
    pub output: Arc<dyn UciSink>,
}

/// A unit of work handed to a parked thread (§3.4). `Custom` is the
/// zero-`unsafe` generalized dispatch (§10, Reckless idea): the closure takes
/// the worker as a *call-time* `&mut` parameter rather than capturing a borrow,
/// so it needs no lifetime gymnastics and stays `Send`.
pub enum Job {
    /// Install per-`go` state (runs on the worker's thread).
    Setup(Box<SetupJob>),
    /// Run [`Worker::start_searching`] with the shared table handle. Carried
    /// per job (not stored) so `setoption Hash` can `Arc::get_mut` the table
    /// while all threads are idle (§3.4).
    Search { tt: Arc<TranspositionTable> },
    /// `ucinewgame`: [`Worker::clear`].
    Clear,
    /// Leave the loop; the thread terminates.
    Exit,
    /// Run an arbitrary one-shot on this worker (parallel TT clear / NUMA / a
    /// test hook). Boxed `FnOnce(&mut Worker) + Send` (§10).
    Custom(Box<dyn FnOnce(&mut Worker) + Send>),
}

// ---------------------------------------------------------------------------
// ThreadShell — the job slot + published results (§3.4)
// ---------------------------------------------------------------------------

struct ThreadState {
    job: Option<Job>,
    busy: bool,
}

/// One thread's rendezvous point: the job slot (`Mutex<ThreadState>` +
/// `Condvar`, both directions on one condvar), its atomic counters, and the
/// slot the worker publishes its [`SearchReport`] into for voting (§3.8).
pub struct ThreadShell {
    state: Mutex<ThreadState>,
    cv: Condvar,
    pub counters: Arc<Counters>,
    pub report: Mutex<Option<SearchReport>>,
}

impl ThreadShell {
    fn new(counters: Arc<Counters>) -> ThreadShell {
        ThreadShell {
            state: Mutex::new(ThreadState {
                job: None,
                busy: false,
            }),
            cv: Condvar::new(),
            counters,
            report: Mutex::new(None),
        }
    }

    /// Hand a job to this thread; blocks until it is idle first, so the
    /// `Option<Job>` slot never holds two jobs (§3.4 invariant). The mutex
    /// hand-off publishes everything the submitter set up.
    pub fn submit(&self, job: Job) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.busy || st.job.is_some() {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = Some(job);
        st.busy = true;
        self.cv.notify_all();
    }

    /// Block until this thread has finished its current job (if any). The
    /// mutex hand-off is the happens-before edge that publishes the job's
    /// writes (reports, root-move scores, final counters) to the waiter
    /// (§3.4, §3.11).
    pub fn wait_idle(&self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.busy || st.job.is_some() {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
}

// ---------------------------------------------------------------------------
// PoolShared — the whole shared surface reachable from a Worker (§3.5)
// ---------------------------------------------------------------------------

/// The shared state every worker can reach through its `Arc<PoolShared>`
/// clone: the per-thread shells (index == `thread_idx`, 0 == main) and the
/// three control flags. All `Relaxed` — they carry no data, only requests;
/// every real handoff goes through a shell mutex (§3.11).
pub struct PoolShared {
    pub shells: Vec<Arc<ThreadShell>>,
    pub stop: AtomicBool,
    pub increase_depth: AtomicBool,
    pub ponder: AtomicBool,
}

impl PoolShared {
    fn new(n: usize) -> Arc<PoolShared> {
        let shells = (0..n)
            .map(|_| Arc::new(ThreadShell::new(Arc::new(Counters::new()))))
            .collect();
        Arc::new(PoolShared {
            shells,
            stop: AtomicBool::new(false),
            increase_depth: AtomicBool::new(true),
            ponder: AtomicBool::new(false),
        })
    }

    /// A 1-thread shared block for a detached [`Worker`] (bench, unit tests):
    /// the worker's own shell shares `counters`, so `nodes_searched` still
    /// aggregates correctly for it.
    pub fn detached(counters: Arc<Counters>) -> Arc<PoolShared> {
        Arc::new(PoolShared {
            shells: vec![Arc::new(ThreadShell::new(counters))],
            stop: AtomicBool::new(false),
            increase_depth: AtomicBool::new(true),
            ponder: AtomicBool::new(false),
        })
    }

    /// Σ over threads of the live node counters (§3.10). Approximate while
    /// threads run (relaxed), exact after `wait_idle`.
    pub fn nodes_searched(&self) -> u64 {
        self.shells
            .iter()
            .map(|s| s.counters.nodes.load(Ordering::Relaxed))
            .sum()
    }

    /// Σ over threads of the tablebase-hit counters (§3.10).
    pub fn tb_hits(&self) -> u64 {
        self.shells
            .iter()
            .map(|s| s.counters.tb_hits.load(Ordering::Relaxed))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// thread_main — the idle loop (§3.4)
// ---------------------------------------------------------------------------

/// Runs on each spawned thread. The `Worker` is constructed HERE so its heavy
/// tables get first-touch-local pages (§3.4). The loop parks until a job
/// arrives, runs it, then marks itself idle and re-parks; `Exit` breaks out
/// and the thread ends.
///
/// `busy` is set true by the submitter (with the job) and cleared here ONLY
/// after the job has fully run — never before taking one. An earlier design
/// (doc §3.4) cleared `busy` at the top of the loop; that races a `submit`
/// that lands before the worker's first park (the submitter set `busy = true`,
/// then the worker's first iteration clobbered it back to `false` and ran the
/// job with `busy == false`, so `wait_idle` returned before completion). The
/// fix keeps `busy == true` for the whole submit→complete window.
///
/// A panicking job must never wedge the pool: the whole `busy` clear + notify
/// must run on unwind too, or a dead thread would leave `busy == true` forever
/// and every future `wait_idle` (and thus `shutdown`/`Drop`, and every next
/// `go`) would block for good — the pre-P11 code recovered from this explicitly
/// (§3.3). The job body therefore runs inside `catch_unwind`: the default panic
/// hook still prints the message + backtrace, but the unwind stops here, the
/// thread releases `busy`, wakes waiters, and re-parks. A job that panicked may
/// leave its `Worker` in an arbitrary (but always *safe* — this is all safe
/// Rust) state; the next `Setup`/`Clear` reinstalls the fields a search reads,
/// so the thread stays serviceable.
fn thread_main(idx: usize, pool: Arc<PoolShared>) {
    let counters = Arc::clone(&pool.shells[idx].counters);
    let mut worker = Worker::new_in_pool(idx, counters, Arc::clone(&pool));
    let shell = Arc::clone(&pool.shells[idx]);
    loop {
        let job = {
            let mut st = shell.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.job.is_none() {
                st = shell.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            // `busy` was set true by the submitter alongside the job; it stays
            // true through execution.
            st.job.take().expect("woke with a job")
        };
        if matches!(job, Job::Exit) {
            return;
        }
        // Run the job; a panic is caught so the idle handoff below still runs.
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| match job {
            Job::Setup(s) => worker.setup(*s),
            Job::Search { tt } => worker.start_searching(&tt),
            Job::Clear => worker.clear(),
            Job::Custom(f) => f(&mut worker),
            Job::Exit => unreachable!("Exit handled before dispatch"),
        }));
        // A panicked job must not leave the REST of the pool running. The main
        // worker's `start_searching` runs its stop-and-join epilogue (search
        // `main_search` step E: `pool.stop.store(true)` + wait every helper)
        // INSIDE the job, so a panic before that point would swallow the stop
        // and leave the helpers searching — still holding their per-`go`
        // `Arc<TranspositionTable>` clones — while `wait_for_search_finished`
        // already observes the parked (panicked) main. A following
        // `setoption Hash`/`ucinewgame`/`Clear Hash` would then reach
        // `Arc::get_mut(&mut tt)` (uci `tt_mut`) against those live clones and
        // panic a SECOND time on the UCI read loop (breaking "never panic on
        // input"). Requesting `stop` here makes every helper wind down and drop
        // its clone promptly; the all-shells wait in `wait_for_search_finished`
        // then closes the get_mut race for good. Harmless for a non-`Search`
        // job or a between-`go` panic (the next `start_thinking` clears `stop`).
        if outcome.is_err() {
            pool.stop.store(true, Ordering::Relaxed);
        }
        // Job complete (or panicked): announce idle and wake any
        // `wait_idle`/`submit` waiter.
        let mut st = shell.state.lock().unwrap_or_else(|e| e.into_inner());
        st.busy = false;
        shell.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// ThreadPool (§3.5)
// ---------------------------------------------------------------------------

/// The pool of parked search threads, owned by the UCI thread. `start_thinking`
/// is non-blocking (§3.6); all cross-`go` state lives in the workers (inside
/// their threads) and in `PoolShared`.
pub struct ThreadPool {
    shared: Arc<PoolShared>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a pool with `n >= 1` threads, all parked and ready.
    pub fn new(n: usize) -> ThreadPool {
        let mut pool = ThreadPool {
            shared: PoolShared::new(0),
            handles: Vec::new(),
        };
        pool.spawn(n.max(1));
        pool
    }

    fn spawn(&mut self, n: usize) {
        let shared = PoolShared::new(n);
        let mut handles = Vec::with_capacity(n);
        for idx in 0..n {
            let pool = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("search-{idx}"))
                .stack_size(WORKER_STACK_SIZE)
                .spawn(move || thread_main(idx, pool))
                .expect("spawn search worker thread");
            handles.push(handle);
        }
        self.shared = shared;
        self.handles = handles;
    }

    /// (Re)create the pool with `n >= 1` threads (`setoption Threads`). Waits
    /// for any running search, exits and joins the old threads first, then
    /// spawns the new set. All per-thread state (histories, TM) is rebuilt.
    pub fn set(&mut self, n: usize) {
        let n = n.max(1);
        if n == self.size() {
            return;
        }
        self.shutdown();
        self.spawn(n);
    }

    /// Number of threads currently in the pool.
    pub fn size(&self) -> usize {
        self.shared.shells.len()
    }

    /// Read-only handle to the shared block (flags + counters), for the UCI
    /// layer's `stop`/`ponderhit` and mid-search `info` node aggregation.
    pub fn shared(&self) -> &Arc<PoolShared> {
        &self.shared
    }

    /// UCI `go` (§3.6). Non-blocking: waits for the previous search, clears
    /// the flags, installs per-`go` state on every thread, then wakes ONLY the
    /// main thread and returns. The main worker wakes the helpers itself
    /// (§3.7 C), after bumping the TT generation.
    pub fn start_thinking(
        &self,
        board: &Board,
        limits: Limits,
        tt: &Arc<TranspositionTable>,
        options: Options,
        output: Arc<dyn UciSink>,
    ) {
        // Step 1: wait for the previous search (main idle ⇒ all idle, §3.5).
        self.shared.shells[0].wait_idle();
        // Step 2: clear control flags.
        self.shared.stop.store(false, Ordering::Relaxed);
        self.shared.increase_depth.store(true, Ordering::Relaxed);
        self.shared.ponder.store(limits.ponder, Ordering::Relaxed);
        // Steps 3–4: install per-`go` state on every thread. Root-move
        // construction is deterministic from board + searchmoves, so each
        // worker rebuilds the identical list in `iterative_deepening` (design
        // deviation from §3.6 step 3's "clone one list" — equivalent order,
        // one fewer clone; noted in design/13 §3.6).
        for shell in &self.shared.shells {
            shell.submit(Job::Setup(Box::new(SetupJob {
                board: board.clone(),
                limits: limits.clone(),
                options: options.clone(),
                output: Arc::clone(&output),
            })));
        }
        // Step 5: wait for all setups (so main sees every helper parked-idle
        // when it wakes them).
        for shell in &self.shared.shells {
            shell.wait_idle();
        }
        // Step 6: wake ONLY the main thread; return immediately.
        self.shared.shells[0].submit(Job::Search { tt: Arc::clone(tt) });
    }

    /// `stop`: request every worker to finish (§3.10). Relaxed — a data-free
    /// signal.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    /// `ponderhit`: the pondered move was played — clear the ponder flag so
    /// the main worker's clock goes live and its busy-wait releases (§3.7 D).
    pub fn ponderhit(&self) {
        self.shared.ponder.store(false, Ordering::Relaxed);
    }

    /// Wait until the search is fully finished (§3.5). In normal operation the
    /// main worker's epilogue (`main_search` step E) stops and joins every
    /// helper before it parks, so main-idle already implies all-idle — but if
    /// the main worker's job PANICS before that epilogue, its helpers keep
    /// running (and holding `Arc<TranspositionTable>` clones) after main parks.
    /// Waiting on EVERY shell — not just `shells[0]` — guarantees that once
    /// this returns no worker is busy and every per-`go` TT clone has been
    /// dropped, so the UCI thread's `Arc::get_mut(&mut tt)` (uci `tt_mut`)
    /// cannot race a live helper. `thread_main` sets `stop` on a caught panic
    /// so those helpers actually wind down (bounded wait, never a hang). Cheap
    /// in the common case: the helpers are already parked, each extra
    /// `wait_idle` returns at once; at Threads=1 it is exactly the old
    /// single-shell wait.
    pub fn wait_for_search_finished(&self) {
        for shell in &self.shared.shells {
            shell.wait_idle();
        }
    }

    /// `ucinewgame`: reset every worker (histories, TM). Waits for the search
    /// to finish first, then runs `Clear` on all threads.
    pub fn clear(&self) {
        self.wait_for_search_finished();
        for shell in &self.shared.shells {
            shell.submit(Job::Clear);
        }
        for shell in &self.shared.shells {
            shell.wait_idle();
        }
    }

    pub fn nodes_searched(&self) -> u64 {
        self.shared.nodes_searched()
    }

    pub fn tb_hits(&self) -> u64 {
        self.shared.tb_hits()
    }

    /// Test hook: run `f` on the main worker and wait for completion. Uses the
    /// `Job::Custom` dispatch to reach the otherwise-encapsulated worker state
    /// (e.g. asserting a per-game field was reset by `ucinewgame`).
    #[cfg(test)]
    pub fn run_on_main<F: FnOnce(&mut Worker) + Send + 'static>(&self, f: F) {
        self.wait_for_search_finished();
        self.shared.shells[0].submit(Job::Custom(Box::new(f)));
        self.shared.shells[0].wait_idle();
    }

    /// Stop, wait for main idle, then Exit + join every thread. Shared by
    /// [`ThreadPool::set`] and `Drop`; must never hang (a joined-thread test
    /// runs under a timeout).
    fn shutdown(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.shells[0].wait_idle();
        for shell in &self.shared.shells {
            shell.submit(Job::Exit);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    // ---- Send/Sync marker assertions (§3.12) ----
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn shared_types_are_sync() {
        assert_sync::<TranspositionTable>();
        assert_sync::<PoolShared>();
        assert_sync::<ThreadShell>();
        assert_sync::<Counters>();
        assert_send::<Worker>();
        assert_send::<Job>();
        assert_send::<SearchReport>();
        assert_send::<Counters>();
    }

    /// Run `f` on a fresh detached thread and fail if it does not finish
    /// within `limit` — every pool test must be hang-proof (§5).
    fn with_timeout(limit: Duration, f: impl FnOnce() + Send + 'static) {
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let d2 = Arc::clone(&done);
        let h = std::thread::spawn(move || {
            f();
            let (m, cv) = &*d2;
            *m.lock().unwrap() = true;
            cv.notify_all();
        });
        let (m, cv) = &*done;
        let mut guard = m.lock().unwrap();
        let start = Instant::now();
        while !*guard {
            let remaining = limit.checked_sub(start.elapsed());
            match remaining {
                None => panic!("pool operation did not finish within {limit:?} (hang)"),
                Some(r) => {
                    let (g, to) = cv.wait_timeout(guard, r).unwrap();
                    guard = g;
                    if to.timed_out() && !*guard {
                        panic!("pool operation did not finish within {limit:?} (hang)");
                    }
                }
            }
        }
        drop(guard);
        h.join().unwrap();
    }

    #[test]
    fn park_submit_run_park_via_custom() {
        with_timeout(Duration::from_secs(10), || {
            let pool = ThreadPool::new(4);
            let counter = Arc::new(AtomicUsize::new(0));
            // Submit a Custom job to every thread; each bumps a shared count.
            for shell in &pool.shared.shells {
                let c = Arc::clone(&counter);
                shell.submit(Job::Custom(Box::new(move |_w| {
                    c.fetch_add(1, Ordering::SeqCst);
                })));
            }
            for shell in &pool.shared.shells {
                shell.wait_idle();
            }
            assert_eq!(counter.load(Ordering::SeqCst), 4);
            // The threads re-parked: a second round runs cleanly.
            for shell in &pool.shared.shells {
                let c = Arc::clone(&counter);
                shell.submit(Job::Custom(Box::new(move |_w| {
                    c.fetch_add(1, Ordering::SeqCst);
                })));
            }
            for shell in &pool.shared.shells {
                shell.wait_idle();
            }
            assert_eq!(counter.load(Ordering::SeqCst), 8);
        });
    }

    #[test]
    fn clean_exit_and_join_for_various_sizes() {
        for &n in &[1usize, 4, 16] {
            with_timeout(Duration::from_secs(10), move || {
                let pool = ThreadPool::new(n);
                assert_eq!(pool.size(), n);
                // Dropping the pool must Exit + join all threads promptly.
                drop(pool);
            });
        }
    }

    #[test]
    fn resize_rebuilds_the_pool() {
        with_timeout(Duration::from_secs(10), || {
            let mut pool = ThreadPool::new(2);
            assert_eq!(pool.size(), 2);
            pool.set(8);
            assert_eq!(pool.size(), 8);
            pool.set(1);
            assert_eq!(pool.size(), 1);
        });
    }

    #[test]
    fn wait_idle_observes_job_completion() {
        with_timeout(Duration::from_secs(10), || {
            let pool = ThreadPool::new(1);
            let flag = Arc::new(AtomicUsize::new(0));
            let f = Arc::clone(&flag);
            pool.shared.shells[0].submit(Job::Custom(Box::new(move |_w| {
                std::thread::sleep(Duration::from_millis(20));
                f.store(42, Ordering::SeqCst);
            })));
            // Not yet observed necessarily; wait_idle must publish the write.
            pool.shared.shells[0].wait_idle();
            assert_eq!(flag.load(Ordering::SeqCst), 42);
        });
    }

    #[test]
    fn counters_aggregate_across_threads() {
        with_timeout(Duration::from_secs(10), || {
            let pool = ThreadPool::new(4);
            for shell in &pool.shared.shells {
                let c = Arc::clone(&shell.counters);
                shell.submit(Job::Custom(Box::new(move |_w| {
                    // Single-writer load+store bump, as on the search path.
                    for _ in 0..1000 {
                        c.nodes
                            .store(c.nodes.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
                    }
                })));
            }
            for shell in &pool.shared.shells {
                shell.wait_idle();
            }
            assert_eq!(pool.nodes_searched(), 4000);
        });
    }

    /// §5 acceptance: the `best_move_changes` harvest loses no count under a
    /// hammering writer. The owner bump is `fetch_add` (an atomic RMW) and the
    /// main worker harvests with `swap(0)`, so every increment is accounted
    /// exactly once — accumulated harvest + residual must equal N. A plain
    /// load+store bump would fail this (it can lose or double-count against the
    /// concurrent `swap`).
    #[test]
    fn best_move_changes_harvest_loses_no_counts() {
        with_timeout(Duration::from_secs(10), || {
            const N: u64 = 200_000;
            let counters = Arc::new(Counters::new());
            let writer = Arc::clone(&counters);
            let h = std::thread::spawn(move || {
                for _ in 0..N {
                    writer.bump_best_move_changes();
                }
            });
            let mut harvested = 0u64;
            while !h.is_finished() {
                harvested += counters.best_move_changes.swap(0, Ordering::Relaxed);
            }
            h.join().unwrap();
            // Final sweep for whatever the writer stored after the last swap.
            harvested += counters.best_move_changes.swap(0, Ordering::Relaxed);
            assert_eq!(harvested, N, "every bump must survive the racing harvest");
        });
    }

    /// A panicking job must not wedge the pool (§3.3): after a `Custom` job
    /// panics, its thread must release `busy`, re-park, and still service the
    /// next job — `wait_idle` must return (not hang) and later work must run.
    /// The pre-fix code let the unwind kill the thread with `busy == true`, so
    /// every subsequent `wait_idle`/`shutdown`/`go` blocked forever.
    #[test]
    fn a_panicking_job_does_not_wedge_the_pool() {
        with_timeout(Duration::from_secs(10), || {
            // The panicking job runs the default hook (stderr backtrace); mute
            // it for this test so the output stays clean, then restore.
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));

            let pool = ThreadPool::new(2);
            // Panic on the MAIN thread's shell — the worst case, since main
            // idle ⇒ all idle drives shutdown/next-go waits.
            pool.shared.shells[0].submit(Job::Custom(Box::new(|_w| {
                panic!("boom");
            })));
            // Must return despite the panic (the fix clears `busy` on unwind).
            pool.shared.shells[0].wait_idle();

            std::panic::set_hook(prev);

            // The thread survived and re-parked: a normal job still runs.
            let ran = Arc::new(AtomicUsize::new(0));
            let r = Arc::clone(&ran);
            pool.shared.shells[0].submit(Job::Custom(Box::new(move |_w| {
                r.store(7, Ordering::SeqCst);
            })));
            pool.shared.shells[0].wait_idle();
            assert_eq!(ran.load(Ordering::SeqCst), 7, "pool must survive a panic");
            // Drop must also not hang (Exit + join on a live thread).
        });
    }

    /// Round-2 lifecycle fix (§3.5, review): a panic in the MAIN worker's job
    /// must not leave the helpers running. The main worker's stop-and-join
    /// epilogue lives INSIDE its job, so a panic before it never sets `stop`;
    /// `thread_main` therefore sets `stop` on any caught unwind, and
    /// `wait_for_search_finished` waits on EVERY shell. Together they guarantee
    /// that once the wait returns no helper is still busy (holding an
    /// `Arc<TT>` clone) — the window that would let the UCI thread's
    /// `Arc::get_mut` in `tt_mut` panic a second time. Here two helpers spin
    /// like a live search until they see `stop`; only the panic-set `stop` can
    /// release them, and only the all-shells wait can observe it.
    #[test]
    fn a_main_panic_stops_helpers_so_the_all_shells_wait_joins() {
        with_timeout(Duration::from_secs(10), || {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));

            let pool = ThreadPool::new(3);
            pool.shared.stop.store(false, Ordering::Relaxed);
            // Helpers busy-loop until they observe `stop` — a live search that
            // never stops itself, exactly the case a main panic must not strand.
            for shell in pool.shared.shells.iter().skip(1) {
                let shared = Arc::clone(&pool.shared);
                shell.submit(Job::Custom(Box::new(move |_w| {
                    while !shared.stop.load(Ordering::Relaxed) {
                        std::thread::yield_now();
                    }
                })));
            }
            // Main panics before it could ever set `stop` on its own.
            pool.shared.shells[0].submit(Job::Custom(Box::new(|_w| {
                panic!("boom");
            })));

            // Returns only once the panic-set `stop` has released both helpers
            // and every shell has parked — a hang here fails via `with_timeout`.
            pool.wait_for_search_finished();
            std::panic::set_hook(prev);

            assert!(
                pool.shared.stop.load(Ordering::Relaxed),
                "a caught panic must request stop so helpers wind down"
            );
            // Every shell is serviceable again (proves they re-parked, not hung).
            let ran = Arc::new(AtomicUsize::new(0));
            for shell in &pool.shared.shells {
                let r = Arc::clone(&ran);
                shell.submit(Job::Custom(Box::new(move |_w| {
                    r.fetch_add(1, Ordering::SeqCst);
                })));
            }
            for shell in &pool.shared.shells {
                shell.wait_idle();
            }
            assert_eq!(ran.load(Ordering::SeqCst), 3, "all shells survived");
        });
    }

    /// A `UciSink` that drops everything — a `Setup` needs one, but this test
    /// only inspects worker board state, never output.
    struct NullSink;
    impl UciSink for NullSink {
        fn emit_info(&self, _: &crate::search::SearchInfo, _: i64) {}
        fn emit_raw(&self, _: &str) {}
        fn emit_bestmove(
            &self,
            _: Option<(crate::search::SearchInfo, i64)>,
            _: crate::moves::Move,
            _: Option<crate::moves::Move>,
        ) {
        }
        fn emit_bestmove_none(&self) {}
    }

    /// §5 setup-propagation: a `Job::Setup` clones the board WITH its game
    /// history (D8, §3.2), so a pooled worker's `root_board` sees the same
    /// repetition state the session board does — the per-thread repetition
    /// detection is self-contained. Verified at the worker level via
    /// `run_on_main`, which the session-level test at `uci.rs` cannot reach.
    #[test]
    fn setup_propagates_repetition_history_to_the_worker() {
        use crate::moves::Move;
        use crate::types::Square;
        with_timeout(Duration::from_secs(10), || {
            // Four reversible knight moves each side: the start position
            // repeats twofold (same shuffle as the uci.rs session test).
            let shuffle = [
                (Square::G1, Square::F3),
                (Square::G8, Square::F6),
                (Square::F3, Square::G1),
                (Square::F6, Square::G8),
                (Square::G1, Square::F3),
                (Square::G8, Square::F6),
                (Square::F3, Square::G1),
                (Square::F6, Square::G8),
            ];
            let mut board = Board::startpos();
            for (from, to) in shuffle {
                let m = Move::normal(from, to);
                board.make_move(m, board.gives_check(m));
            }
            assert!(board.has_repeated(), "session board must see the twofold");

            let pool = ThreadPool::new(2);
            pool.shared.shells[0].submit(Job::Setup(Box::new(SetupJob {
                board,
                limits: Limits::new(),
                options: Options::new(),
                output: Arc::new(NullSink),
            })));
            pool.shared.shells[0].wait_idle();
            // The worker's own cloned board must also know the twofold.
            pool.run_on_main(|w| {
                assert!(
                    w.board.has_repeated(),
                    "the worker's cloned board lost its repetition history"
                );
            });
        });
    }
}
