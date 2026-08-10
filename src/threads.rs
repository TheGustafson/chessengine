//! Threading — lazy SMP in safe Rust (`design/13-threading-smp.md`).
//!
//! A pool of long-lived, parked worker threads (D23). Each thread owns a
//! [`Worker`] (thread 0 also owns the main-worker `SearchManager`),
//! constructed on the thread itself for first-touch locality. Threads park in
//! [`thread_main`]'s idle loop and are woken with [`Job`]s through a
//! per-thread [`ThreadShell`] (mirrors SF's `searching`-flag protocol).
//!
//! The shared surface is deliberately tiny: the TT (`Arc<TT>`, all-atomic),
//! three [`AtomicBool`] flags in [`PoolShared`], and per-thread atomic
//! [`Counters`]. Everything else a search mutates is single-owner inside the
//! `Worker`. **No `unsafe` and no lock on the per-node search path**: the
//! mutexes here are touched only at job boundaries.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::board::Board;
use crate::search::{SearchReport, UciSink, Worker};
use crate::timeman::Limits;
use crate::tt::TranspositionTable;
use crate::uci::Options;

/// Per-thread stack size: search recursion reaches `MAX_PLY` with fat
/// frames (SF forces 8 MiB likewise).
pub const WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Counters (§3.10)
// ---------------------------------------------------------------------------

/// Per-thread atomic counters, cache-line isolated so helpers never
/// false-share. `nodes`/`tb_hits` are written by their owning worker only,
/// so the cheap single-writer load+store bump is correct; `best_move_changes`
/// is also written by the main worker (a `swap(0)` harvest), so its owner
/// bump must be a real atomic RMW (`fetch_add`).
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

    /// Zero every counter (each `Setup`). Owner-only; relaxed stores suffice.
    pub fn reset(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.tb_hits.store(0, Ordering::Relaxed);
        self.best_move_changes.store(0, Ordering::Relaxed);
    }

    /// Bump `best_move_changes`: a non-first root move became best this
    /// iteration. NOTE: must stay `fetch_add` — the main worker harvests with
    /// `swap(0)`, and a plain load+store bump can lose or re-seed an already
    /// harvested count. Fires at most once per root move per iteration.
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

/// Per-`go` state installed on the worker's own thread by a [`Job::Setup`].
/// Board is cloned WITH its game history so each worker's repetition
/// detection is self-contained.
pub struct SetupJob {
    pub board: Board,
    pub limits: Limits,
    /// Options snapshot (main uses it for `TimeManager::init`; helpers ignore).
    pub options: Options,
    /// Type-erased output sink (only the main worker writes through it).
    pub output: Arc<dyn UciSink>,
}

/// A unit of work handed to a parked thread. `Custom`'s closure takes the
/// worker as a *call-time* `&mut` parameter rather than capturing a borrow,
/// so it needs no lifetime gymnastics and stays `Send`.
pub enum Job {
    /// Install per-`go` state (runs on the worker's thread).
    Setup(Box<SetupJob>),
    /// Run [`Worker::start_searching`]. The TT handle is carried per job (not
    /// stored) so `setoption Hash` can `Arc::get_mut` while all are idle.
    Search { tt: Arc<TranspositionTable> },
    /// `ucinewgame`: [`Worker::clear`].
    Clear,
    /// Leave the loop; the thread terminates.
    Exit,
    /// Run an arbitrary one-shot on this worker.
    Custom(Box<dyn FnOnce(&mut Worker) + Send>),
}

// ---------------------------------------------------------------------------
// ThreadShell — the job slot + published results (§3.4)
// ---------------------------------------------------------------------------

struct ThreadState {
    job: Option<Job>,
    busy: bool,
}

/// One thread's rendezvous point: the job slot (mutex + condvar), its atomic
/// counters, and the slot the worker publishes its [`SearchReport`] into.
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

    /// Hand a job to this thread; blocks until it is idle first, so the slot
    /// never holds two jobs. The mutex hand-off publishes the submitter's
    /// writes.
    pub fn submit(&self, job: Job) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.busy || st.job.is_some() {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = Some(job);
        st.busy = true;
        self.cv.notify_all();
    }

    /// Block until this thread has finished its current job (if any); the
    /// mutex hand-off publishes the job's writes to the waiter.
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

/// The shared state every worker can reach: the per-thread shells (index ==
/// `thread_idx`, 0 == main) and the three control flags. All `Relaxed` —
/// they carry no data, only requests; real handoffs go through shell mutexes.
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

    /// A 1-thread shared block for a detached [`Worker`] (bench, unit tests).
    pub fn detached(counters: Arc<Counters>) -> Arc<PoolShared> {
        Arc::new(PoolShared {
            shells: vec![Arc::new(ThreadShell::new(counters))],
            stop: AtomicBool::new(false),
            increase_depth: AtomicBool::new(true),
            ponder: AtomicBool::new(false),
        })
    }

    /// Σ over threads of the live node counters: approximate while threads
    /// run (relaxed), exact after `wait_idle`.
    pub fn nodes_searched(&self) -> u64 {
        self.shells
            .iter()
            .map(|s| s.counters.nodes.load(Ordering::Relaxed))
            .sum()
    }

    /// Σ over threads of the tablebase-hit counters.
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
/// tables get first-touch-local pages. The loop parks until a job arrives,
/// runs it, marks itself idle, and re-parks; `Exit` ends the thread.
///
/// NOTE: `busy` is set true by the submitter (with the job) and cleared here
/// ONLY after the job has fully run — clearing it at the top of the loop
/// races a `submit` landing before the first park, and `wait_idle` would
/// return before completion.
///
/// NOTE: the job body runs inside `catch_unwind` — a panicking job must
/// still clear `busy` and notify, or every future `wait_idle` (and thus
/// `shutdown`/`Drop`/next `go`) blocks forever. A panicked `Worker` may be
/// in an arbitrary safe state; the next `Setup`/`Clear` reinstalls what a
/// search reads.
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
        // NOTE: a panicked main job must not strand the helpers. Main's
        // stop-and-join epilogue runs INSIDE its job, so a panic before it
        // would leave helpers searching (holding per-`go` `Arc<TT>` clones)
        // and a later `Arc::get_mut` on the TT would panic the UCI loop.
        // Requesting `stop` here makes every helper wind down; harmless for
        // non-`Search` jobs (the next `start_thinking` clears `stop`).
        if outcome.is_err() {
            pool.stop.store(true, Ordering::Relaxed);
        }
        // Job complete (or panicked): announce idle and wake waiters.
        let mut st = shell.state.lock().unwrap_or_else(|e| e.into_inner());
        st.busy = false;
        shell.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// ThreadPool (§3.5)
// ---------------------------------------------------------------------------

/// The pool of parked search threads, owned by the UCI thread.
/// `start_thinking` is non-blocking; all cross-`go` state lives in the
/// workers and in `PoolShared`.
pub struct ThreadPool {
    shared: Arc<PoolShared>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a pool with `n >= 1` threads, all parked. A spawn failure here
    /// is at *startup*, so it is fatal.
    pub fn new(n: usize) -> ThreadPool {
        let (shared, handles) =
            Self::build(n.max(1)).expect("spawn search worker threads at startup");
        ThreadPool { shared, handles }
    }

    /// Spawn `n >= 1` parked worker threads WITHOUT installing them on
    /// `self`. On any OS spawn failure the threads already created are
    /// cleanly `Exit`ed and joined and the `io::Error` is returned — no
    /// half-built pool ever escapes, and `set` can degrade gracefully.
    fn build(n: usize) -> std::io::Result<(Arc<PoolShared>, Vec<JoinHandle<()>>)> {
        let shared = PoolShared::new(n);
        let mut handles = Vec::with_capacity(n);
        for idx in 0..n {
            let pool = Arc::clone(&shared);
            match std::thread::Builder::new()
                .name(format!("search-{idx}"))
                .stack_size(WORKER_STACK_SIZE)
                .spawn(move || thread_main(idx, pool))
            {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    // Roll back: `Job::Exit` + join reaps every spawned thread.
                    for shell in shared.shells.iter().take(handles.len()) {
                        shell.submit(Job::Exit);
                    }
                    for h in handles.drain(..) {
                        let _ = h.join();
                    }
                    return Err(e);
                }
            }
        }
        Ok((shared, handles))
    }

    /// (Re)create the pool with `n >= 1` threads (`setoption Threads`).
    /// Builds the NEW set FIRST, then shuts down and swaps out the old — a
    /// spawn failure returns `Err` with the existing pool untouched. Cost: a
    /// brief `old + new` thread overlap. Per-thread state is rebuilt fresh.
    pub fn set(&mut self, n: usize) -> std::io::Result<()> {
        let n = n.max(1);
        if n == self.size() {
            return Ok(());
        }
        let (shared, handles) = Self::build(n)?;
        self.shutdown();
        self.shared = shared;
        self.handles = handles;
        Ok(())
    }

    /// Number of threads currently in the pool.
    pub fn size(&self) -> usize {
        self.shared.shells.len()
    }

    /// Read-only handle to the shared block (flags + counters).
    pub fn shared(&self) -> &Arc<PoolShared> {
        &self.shared
    }

    /// UCI `go`. Non-blocking: waits for the previous search, clears the
    /// flags, installs per-`go` state on every thread, then wakes ONLY the
    /// main thread and returns; the main worker wakes the helpers itself.
    pub fn start_thinking(
        &self,
        board: &Board,
        limits: Limits,
        tt: &Arc<TranspositionTable>,
        options: Options,
        output: Arc<dyn UciSink>,
    ) {
        // Step 1: wait on EVERY shell, not just main. NOTE: "main idle ⇒ all
        // idle" is FALSE after a main-job panic — a stranded helper must be
        // reaped BEFORE step 2 clears `stop`, or it never stops and step 3's
        // `submit` to its still-busy shell wedges the UCI thread.
        self.wait_for_search_finished();
        // Step 2: clear control flags.
        self.shared.stop.store(false, Ordering::Relaxed);
        self.shared.increase_depth.store(true, Ordering::Relaxed);
        self.shared.ponder.store(limits.ponder, Ordering::Relaxed);
        // Steps 3–4: install per-`go` state on every thread. Root-move
        // construction is deterministic from board + searchmoves, so each
        // worker rebuilds the identical list itself (design/13 §3.6).
        for shell in &self.shared.shells {
            shell.submit(Job::Setup(Box::new(SetupJob {
                board: board.clone(),
                limits: limits.clone(),
                options: options.clone(),
                output: Arc::clone(&output),
            })));
        }
        // Step 5: wait for all setups.
        for shell in &self.shared.shells {
            shell.wait_idle();
        }
        // Step 6: wake ONLY the main thread; return immediately.
        self.shared.shells[0].submit(Job::Search { tt: Arc::clone(tt) });
    }

    /// `stop`: request every worker to finish. Relaxed — a data-free signal.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    /// `ponderhit`: clear the ponder flag so the main worker's clock goes
    /// live and its busy-wait releases.
    pub fn ponderhit(&self) {
        self.shared.ponder.store(false, Ordering::Relaxed);
    }

    /// Wait until the search is fully finished. NOTE: waits on EVERY shell,
    /// not just main — after a main-job panic the helpers can outlive main,
    /// and only the all-shells wait guarantees every per-`go` TT clone is
    /// dropped before the UCI thread's `Arc::get_mut`. Bounded because
    /// `thread_main` sets `stop` on a caught panic; cheap when already parked.
    pub fn wait_for_search_finished(&self) {
        for shell in &self.shared.shells {
            shell.wait_idle();
        }
    }

    /// `ucinewgame`: wait for the search, then run `Clear` on all threads.
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

    /// Test hook: run `f` on the main worker (via `Job::Custom`) and wait.
    #[cfg(test)]
    pub fn run_on_main<F: FnOnce(&mut Worker) + Send + 'static>(&self, f: F) {
        self.wait_for_search_finished();
        self.shared.shells[0].submit(Job::Custom(Box::new(f)));
        self.shared.shells[0].wait_idle();
    }

    /// Stop, wait for main idle, then Exit + join every thread. Shared by
    /// [`ThreadPool::set`] and `Drop`; must never hang.
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
    /// within `limit` — every pool test must be hang-proof.
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
            pool.set(8).expect("grow pool");
            assert_eq!(pool.size(), 8);
            pool.set(1).expect("shrink pool");
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

    /// The `best_move_changes` harvest loses no count under a hammering
    /// writer: accumulated harvest + residual must equal N. A plain
    /// load+store bump would fail this.
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

    /// A panicking job must not wedge the pool: its thread must release
    /// `busy`, re-park, and still service the next job.
    #[test]
    fn a_panicking_job_does_not_wedge_the_pool() {
        with_timeout(Duration::from_secs(10), || {
            // Mute the default panic hook for clean test output, then restore.
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));

            let pool = ThreadPool::new(2);
            // Panic on the MAIN thread's shell — the worst case.
            pool.shared.shells[0].submit(Job::Custom(Box::new(|_w| {
                panic!("boom");
            })));
            // Must return despite the panic (`busy` cleared on unwind).
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
        });
    }

    /// A panic in the MAIN worker's job must not leave the helpers running:
    /// `thread_main` sets `stop` on a caught unwind and
    /// `wait_for_search_finished` waits on EVERY shell. Two helpers spin like
    /// a live search until they see `stop`; only the panic-set `stop` can
    /// release them.
    #[test]
    fn a_main_panic_stops_helpers_so_the_all_shells_wait_joins() {
        with_timeout(Duration::from_secs(10), || {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));

            let pool = ThreadPool::new(3);
            pool.shared.stop.store(false, Ordering::Relaxed);
            // Helpers busy-loop until they observe `stop`.
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

            // Returns only once the panic-set `stop` released both helpers.
            pool.wait_for_search_finished();
            std::panic::set_hook(prev);

            assert!(
                pool.shared.stop.load(Ordering::Relaxed),
                "a caught panic must request stop so helpers wind down"
            );
            // Every shell is serviceable again.
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

    /// A `UciSink` that drops everything.
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

    /// A `Job::Setup` clones the board WITH its game history, so a pooled
    /// worker sees the same repetition state the session board does.
    #[test]
    fn setup_propagates_repetition_history_to_the_worker() {
        use crate::moves::Move;
        use crate::types::Square;
        with_timeout(Duration::from_secs(10), || {
            // Four reversible knight moves each side: startpos repeats twofold.
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
