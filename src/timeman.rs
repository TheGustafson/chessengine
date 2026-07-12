//! Time management: the `Limits` type shared with the UCI layer, and the
//! `TimeManager` allocation math run at every `go` (design/12 §3.6, §3.7,
//! §3.11).
//!
//! `init` turns the GUI clock into two millisecond budgets: `optimum` (the
//! soft target the iterative-deepening loop compares against between
//! iterations, doc 12 §3.9) and `maximum` (the hard wall `check_time`
//! enforces mid-search, doc 12 §3.8). All time is `i64` milliseconds,
//! converted from `std::time::Instant` only at the measurement points, so
//! every formula is plain integer/float arithmetic and the `nodestime`
//! substitution (milliseconds become node counts) is transparent.
//!
//! Every tuned constant is an SF-current fitted value adopted verbatim as a
//! starting point (D20), named here and nowhere else. The formula SHAPES
//! matter more than the numbers when tuning: `opt_scale` grows slowly with
//! game ply (a power curve saturated by `opt_constant`, which itself grows
//! with the log of the clock) capped at ~19.4% of the current clock per
//! move; `max_scale` allows `maximum` up to ~6.9x `optimum` early, tapering
//! via the absolute ~81%-of-clock cap.

use crate::moves::Move;
use crate::types::{Color, Depth, VALUE_INFINITE, Value};
use crate::uci::Options;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Allocation constants (design/12 §3.7; D20 — exact from SF's timeman.cpp)
// ---------------------------------------------------------------------------

/// Maximum move horizon: never plan for more than 50 further moves.
const MOVE_HORIZON: i32 = 50;
/// Sub-second horizon shrink: below one second on the clock the horizon
/// becomes `scaled_time * 0.05` moves (may reach 0; the formulas stay
/// well-defined, §3.7 step 3).
const SUBSECOND_HORIZON: f64 = 0.05;
/// `time_left` prices in `move_overhead * (2 + mtg)`: one extra move of
/// overhead as a safety margin.
const OVERHEAD_MARGIN_MOVES: i64 = 2;

/// `original_time_adjust = 0.3272 * log10(time_left) - 0.4141`, computed
/// once per game at the first timed `init` (§3.7 step 5).
const ORIGINAL_TIME_ADJUST_LOG_COEFF: f64 = 0.3272;
const ORIGINAL_TIME_ADJUST_OFFSET: f64 = 0.4141;

/// `opt_constant = min(0.0029869 + 0.00033554 * log10(time_in_sec), 0.004905)`
const OPT_CONSTANT_BASE: f64 = 0.0029869;
const OPT_CONSTANT_LOG_COEFF: f64 = 0.00033554;
const OPT_CONSTANT_CAP: f64 = 0.004905;

/// `max_constant = max(3.3744 + 3.0608 * log10(time_in_sec), 3.1441)`
const MAX_CONSTANT_BASE: f64 = 3.3744;
const MAX_CONSTANT_LOG_COEFF: f64 = 3.0608;
const MAX_CONSTANT_FLOOR: f64 = 3.1441;

/// Sudden-death `opt_scale`: `min(0.012112 + (ply + 3.22713)^0.46866 *
/// opt_constant, 0.19404 * time / time_left) * original_time_adjust`.
const OPT_SCALE_BASE: f64 = 0.012112;
const OPT_SCALE_PLY_OFFSET: f64 = 3.22713;
const OPT_SCALE_PLY_POW: f64 = 0.46866;
const OPT_SCALE_CLOCK_CAP: f64 = 0.19404;

/// Sudden-death `max_scale = min(6.873, max_constant + ply / 12.352)`.
const MAX_SCALE_CAP: f64 = 6.873;
const MAX_SCALE_PLY_DIV: f64 = 12.352;

/// Repeating-TC `opt_scale = min((0.88 + ply / 116.4) / mtg,
/// 0.88 * time / time_left)`.
const MTG_OPT_BASE: f64 = 0.88;
const MTG_OPT_PLY_DIV: f64 = 116.4;
const MTG_OPT_CLOCK_CAP: f64 = 0.88;

/// Repeating-TC `max_scale = 1.3 + 0.11 * mtg`.
const MTG_MAX_SCALE_BASE: f64 = 1.3;
const MTG_MAX_SCALE_PER_MOVE: f64 = 0.11;

/// Never budget more than ~81% of the remaining clock for one move:
/// `maximum <= 0.8097 * time - move_overhead`.
const MAX_CLOCK_FRACTION: f64 = 0.8097;

/// `Ponder` option bump: think `optimum / 4` longer, since pondering
/// recoups time on average (§3.7 step 8).
const PONDER_OPTIMUM_BONUS_DIV: i64 = 4;

/// Initial `previous_time_reduction` (per-game default; SF's
/// `previousTimeReduction = 0.85` in `thread.cpp`). `pub(crate)` so the
/// search worker's working copy (doc 12 §3.9 `reduction` factor) can share
/// the same seed.
pub(crate) const PREVIOUS_TIME_REDUCTION_INIT: f64 = 0.85;

/// Per-`go` seed for the running `time_reduction` (SF's local
/// `double timeReduction = 1` at `search.cpp:279`). It is recomputed by the
/// soft stop each iteration and written back to `previous_time_reduction`
/// at search end; seeding to 1.0 means a `go` whose factor block never runs
/// (fixed depth/nodes) leaves the *next* move's numerator at 1.0, not a
/// stale persisted seed. `pub(crate)` for the search worker's working copy.
pub(crate) const TIME_REDUCTION_INIT: f64 = 1.0;

// ---------------------------------------------------------------------------
// Limits — everything `go` can ask for (design/12 §3.6)
// ---------------------------------------------------------------------------

/// Everything `go` can ask for. Times in milliseconds. Zero in
/// `time`/`inc`/`movestogo` means "not given" (SF convention); the `Option`
/// fields distinguish absent limits without sentinels.
#[derive(Clone, Debug)]
pub struct Limits {
    /// `wtime`/`btime`, indexed by `Color as usize`; 0 = not given. May be
    /// rewritten to a node budget by `TimeManager::init` under `nodestime`.
    pub time: [i64; 2],
    /// `winc`/`binc`, indexed by `Color as usize`.
    pub inc: [i64; 2],
    /// `movestogo`; 0 = sudden death / not given.
    pub movestogo: i32,
    /// `nodestime` (nodes-per-ms) flag, set by [`TimeManager::init`] ONLY
    /// when it actually converted the side-to-move's own clock to a node
    /// budget this move (SF's `limits.npmsec`, `search.h`/`timeman.cpp`):
    /// non-zero iff `nodestime > 0` AND the own clock was given (so `init`
    /// ran past its missing-clock early return). The post-search node
    /// accounting gates on this — NOT on `use_time_management()`, which is
    /// true even when only the OPPONENT's clock was sent, a corner where
    /// `init` early-returns and the conversion never happened (§3.11).
    /// Reset to 0 by [`Limits::new`] every `go`.
    pub npmsec: i64,
    /// `go depth d`
    pub depth: Option<Depth>,
    /// `go nodes n`
    pub nodes: Option<u64>,
    /// `go movetime ms`
    pub movetime: Option<i64>,
    /// `go mate n` (n = full moves, not plies)
    pub mate: Option<i32>,
    pub infinite: bool,
    pub ponder: bool,
    /// Root-move whitelist; empty = all root moves.
    pub searchmoves: Vec<Move>,
    /// `go perft d` (dev command, doc 12 §3.15)
    pub perft: Option<u32>,
    /// Stamped when parsing of the `go` line begins.
    pub start_time: Instant,
}

impl Limits {
    /// Fresh, empty limits (a bare `go`), stamped `start_time` now.
    pub fn new() -> Limits {
        Limits {
            time: [0; 2],
            inc: [0; 2],
            movestogo: 0,
            npmsec: 0,
            depth: None,
            nodes: None,
            movetime: None,
            mate: None,
            infinite: false,
            ponder: false,
            searchmoves: Vec::new(),
            perft: None,
            start_time: Instant::now(),
        }
    }

    /// Clock-based time management is active when the GUI sent EITHER
    /// clock — side-independent, exactly SF's predicate. A GUI that sends
    /// only the opponent's clock still gets a TM-bounded search, never an
    /// unbounded one (`init` returns early in that case, leaving
    /// `maximum() == 0`, so the first `check_time` poll stops — §3.7 step
    /// 1). depth/nodes/movetime/mate/infinite modes bypass TM.
    pub fn use_time_management(&self) -> bool {
        self.time[Color::White as usize] != 0 || self.time[Color::Black as usize] != 0
    }
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::new()
    }
}

// ---------------------------------------------------------------------------
// TimePersist — per-GAME state surviving across `go` calls (design/12 §3.7)
// ---------------------------------------------------------------------------

/// Per-game time-management state; reset by `ucinewgame`. Lives in the main
/// worker's `SearchManager` once the thread pool lands (doc 12 §3.8/§3.1);
/// until P5 the UCI `Engine` holds it directly.
#[derive(Clone, Debug)]
pub struct TimePersist {
    /// Sudden-death log correction, computed at the FIRST timed `init` of a
    /// game and reused for the rest of it. `-1.0` = unset.
    pub original_time_adjust: f64,
    /// Last move's soft-stop `time_reduction` (doc 12 §3.9 factor 2).
    pub previous_time_reduction: f64,
    /// `rootMoves[0]` score of the previous search; seeds `iter_value`.
    pub best_previous_score: Value,
    /// Running-average score of the previous search; feeds `fallingEval`.
    pub best_previous_average_score: Value,
}

impl Default for TimePersist {
    /// The "first move of the game" defaults (§3.7): these measurably
    /// affect move-1 time usage, so `ucinewgame` must restore exactly this.
    fn default() -> TimePersist {
        TimePersist {
            original_time_adjust: -1.0,
            previous_time_reduction: PREVIOUS_TIME_REDUCTION_INIT,
            best_previous_score: VALUE_INFINITE,
            best_previous_average_score: VALUE_INFINITE,
        }
    }
}

// ---------------------------------------------------------------------------
// TimeManager (design/12 §3.7, §3.11)
// ---------------------------------------------------------------------------

/// Per-`go` allocation state: computes `optimum`/`maximum` once at `init`
/// and answers elapsed-time queries for `check_time` and the soft stop.
#[derive(Clone, Debug)]
pub struct TimeManager {
    start_time: Instant,
    optimum_ms: i64,
    maximum_ms: i64,
    use_nodes_time: bool,
    /// `nodestime` budget; `-1` = not yet initialized this game (§3.11).
    available_nodes: i64,
}

impl TimeManager {
    pub fn new() -> TimeManager {
        TimeManager {
            start_time: Instant::now(),
            optimum_ms: 0,
            maximum_ms: 0,
            use_nodes_time: false,
            available_nodes: -1,
        }
    }

    /// `ucinewgame` reset (§3.7): forget the node budget so the next timed
    /// `init` re-derives it from the fresh clock.
    pub fn new_game(&mut self) {
        self.available_nodes = -1;
    }

    /// Compute `optimum`/`maximum` for this move. `ply` = game ply
    /// (`2 * (fullmove_number - 1) + (stm == Black) as i32`, i.e.
    /// `Board::game_ply`). May rewrite `limits.time`/`inc` under
    /// `nodestime` (§3.11). Numbers exact from `timeman.cpp` via the design
    /// doc; the step numbers below are design/12 §3.7's.
    pub fn init(
        &mut self,
        limits: &mut Limits,
        us: Color,
        ply: i32,
        options: &Options,
        persist: &mut TimePersist,
    ) {
        // Step 1: explicit per-`go` reset so behavior is deterministic and
        // testable even on the early return (missing own clock in a timed
        // game => optimum == maximum == 0 => first check_time poll stops).
        self.start_time = limits.start_time;
        self.optimum_ms = 0;
        self.maximum_ms = 0;
        let npmsec = options.nodestime();
        self.use_nodes_time = npmsec > 0;

        let us = us as usize;
        if limits.time[us] == 0 {
            return; // movetime/depth/nodes/infinite are handled elsewhere
        }

        // Step 2: overhead, and the nodestime conversion (§3.11) — replace
        // wall milliseconds with a node budget so "time" becomes
        // bit-reproducible. `scaled_time` stays in human milliseconds for
        // comparisons against absolute constants.
        let mut move_overhead = options.move_overhead_ms();
        if self.use_nodes_time {
            if self.available_nodes == -1 {
                // Only once at game start (time is in msec here).
                self.available_nodes = npmsec * limits.time[us];
            }
            limits.time[us] = self.available_nodes;
            limits.inc[us] *= npmsec;
            // SF timeman.cpp:79 — the post-search accounting gate. Set here,
            // AFTER the missing-own-clock early return, so it stays 0 in the
            // opponent-clock-only corner where no conversion happened.
            limits.npmsec = npmsec;
            move_overhead *= npmsec;
        }
        let scale_factor: i64 = if self.use_nodes_time { npmsec } else { 1 };
        let time = limits.time[us];
        let inc = limits.inc[us];
        let scaled_time = time / scale_factor;

        // Step 3: move horizon. Below one second on the clock the horizon
        // shrinks toward 0 (`mtg` may BE 0: the f64 formulas below stay
        // well-defined — the division yields inf, the min()-cap catches it,
        // and `time_left`'s clamp handles the integer part).
        let mut mtg: i32 = if limits.movestogo > 0 {
            limits.movestogo.min(MOVE_HORIZON)
        } else {
            MOVE_HORIZON
        };
        if scaled_time < 1000 {
            mtg = (scaled_time as f64 * SUBSECOND_HORIZON) as i32;
        }

        // Step 4: the time we can realistically distribute over the
        // horizon. `max(1)` also absorbs negative GUI-lag clocks.
        let time_left = (time + inc * (i64::from(mtg) - 1)
            - move_overhead * (OVERHEAD_MARGIN_MOVES + i64::from(mtg)))
        .max(1);
        let time_left_f = time_left as f64;

        let (opt_scale, max_scale): (f64, f64);
        if limits.movestogo == 0 {
            // Step 5: sudden death / increment — fitted log formulas.
            if persist.original_time_adjust < 0.0 {
                persist.original_time_adjust = ORIGINAL_TIME_ADJUST_LOG_COEFF * time_left_f.log10()
                    - ORIGINAL_TIME_ADJUST_OFFSET;
            }
            let log_time_in_sec = (scaled_time as f64 / 1000.0).log10();
            let opt_constant = (OPT_CONSTANT_BASE + OPT_CONSTANT_LOG_COEFF * log_time_in_sec)
                .min(OPT_CONSTANT_CAP);
            let max_constant = (MAX_CONSTANT_BASE + MAX_CONSTANT_LOG_COEFF * log_time_in_sec)
                .max(MAX_CONSTANT_FLOOR);
            opt_scale = (OPT_SCALE_BASE
                + (f64::from(ply) + OPT_SCALE_PLY_OFFSET).powf(OPT_SCALE_PLY_POW) * opt_constant)
                .min(OPT_SCALE_CLOCK_CAP * time as f64 / time_left_f)
                * persist.original_time_adjust;
            max_scale = MAX_SCALE_CAP.min(max_constant + f64::from(ply) / MAX_SCALE_PLY_DIV);
        } else {
            // Step 6: repeating time control.
            opt_scale = ((MTG_OPT_BASE + f64::from(ply) / MTG_OPT_PLY_DIV) / f64::from(mtg))
                .min(MTG_OPT_CLOCK_CAP * time as f64 / time_left_f);
            max_scale = MTG_MAX_SCALE_BASE + MTG_MAX_SCALE_PER_MOVE * f64::from(mtg);
        }

        // Step 7: budgets. (Rust f64->i64 `as` saturates, so even the
        // degenerate -inf/inf corners of the formulas stay panic-free.)
        self.optimum_ms = ((opt_scale * time_left_f) as i64).max(1);
        self.maximum_ms = ((MAX_CLOCK_FRACTION * time as f64 - move_overhead as f64)
            .min(max_scale * self.optimum_ms as f64) as i64)
            .max(self.optimum_ms);

        // Step 8: if the Ponder OPTION is on (regardless of whether this
        // `go` is a ponder search), think 25% longer — pondering recoups
        // time on average. Applied last, after `maximum` (SF order), so the
        // bumped optimum may exceed maximum.
        if options.ponder() {
            self.optimum_ms += self.optimum_ms / PONDER_OPTIMUM_BONUS_DIV;
        }
    }

    /// Soft time budget in ms (or nodes under `nodestime`).
    pub fn optimum(&self) -> i64 {
        self.optimum_ms
    }

    /// Hard time wall in ms (or nodes under `nodestime`).
    pub fn maximum(&self) -> i64 {
        self.maximum_ms
    }

    /// TM-elapsed: nodes searched under `nodestime`, else wall ms. `Fn`,
    /// not `FnOnce`: the counter closure is invoked only under `nodestime`,
    /// and `check_time` may need it again for the node limit — same call,
    /// possibly twice, possibly zero times (doc 12 §3.8).
    pub fn elapsed(&self, nodes: impl Fn() -> u64) -> i64 {
        if self.use_nodes_time {
            nodes() as i64
        } else {
            self.elapsed_ms()
        }
    }

    /// Wall-clock ms since the `go` arrived — always, even under
    /// `nodestime` (feeds `info ... time`/`nps`, which stay human-readable).
    pub fn elapsed_ms(&self) -> i64 {
        self.start_time.elapsed().as_millis() as i64
    }

    /// Post-search `nodestime` accounting (§3.11): book the consumed nodes
    /// and credit the increment. `used_minus_inc` = nodes searched minus
    /// the increment converted to nodes.
    pub fn advance_nodes_time(&mut self, used_minus_inc: i64) {
        debug_assert!(self.use_nodes_time);
        self.available_nodes = (self.available_nodes - used_minus_inc).max(0);
    }
}

impl Default for TimeManager {
    fn default() -> TimeManager {
        TimeManager::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Golden expectations are hand-derived from design/12 §3.7 and
    /// cross-checked with an independent Python implementation of the same
    /// formulas (double precision, identical operation order).
    fn init_tm(
        time: i64,
        inc: i64,
        movestogo: i32,
        ply: i32,
        options: &Options,
        persist: &mut TimePersist,
    ) -> (TimeManager, Limits) {
        let mut limits = Limits::new();
        limits.time[Color::White as usize] = time;
        limits.inc[Color::White as usize] = inc;
        limits.movestogo = movestogo;
        let mut tm = TimeManager::new();
        tm.init(&mut limits, Color::White, ply, options, persist);
        (tm, limits)
    }

    fn default_options() -> Options {
        Options::new() // Move Overhead 10, Ponder false, nodestime 0
    }

    #[test]
    fn golden_sudden_death() {
        // time=60000 inc=600 ply=20 overhead=10 movestogo=0 (§5 example):
        // mtg=50, time_left=88880, OTA=0.3272*log10(88880)-0.4141.
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(60000, 600, 0, 20, &default_options(), &mut persist);
        assert_eq!(tm.optimum(), 2973);
        assert_eq!(tm.maximum(), 20433); // 6.873 * 2973 caps below 81% clock
        assert!((persist.original_time_adjust - 1.205_148_683_829_215_8).abs() < 1e-12);
    }

    #[test]
    fn golden_repeating_tc() {
        // time=180000 movestogo=40 ply=10: mtg=40, time_left=179580,
        // opt_scale=(0.88+10/116.4)/40, max_scale=1.3+0.11*40=5.7.
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(180_000, 0, 40, 10, &default_options(), &mut persist);
        assert_eq!(tm.optimum(), 4336);
        assert_eq!(tm.maximum(), 24715);
        // The movestogo branch never computes the sudden-death adjust.
        assert_eq!(persist.original_time_adjust, -1.0);
    }

    #[test]
    fn golden_subsecond_horizon_shrink() {
        // time=750 ply=30: scaled_time<1000 => mtg=(750*0.05)=37,
        // time_left=750-10*39=360.
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(750, 0, 0, 30, &default_options(), &mut persist);
        assert_eq!(tm.optimum(), 4);
        assert_eq!(tm.maximum(), 22);
    }

    #[test]
    fn golden_time_left_clamps_at_one() {
        // time=10: mtg=0, time_left=max(1, 10-10*2)=1, OTA=-0.4141 (<0) so
        // the optimum floor of 1 binds; the 81% cap is negative so the
        // maximum floor (optimum) binds too.
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(10, 0, 0, 0, &default_options(), &mut persist);
        assert_eq!(tm.optimum(), 1);
        assert_eq!(tm.maximum(), 1);
    }

    #[test]
    fn golden_ponder_option_bumps_optimum_by_quarter() {
        let mut options = default_options();
        options.set("Ponder", Some("true")).unwrap();
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(60000, 600, 0, 20, &options, &mut persist);
        // Base optimum 2973 (golden_sudden_death) + 2973/4 = 743.
        assert_eq!(tm.optimum(), 2973 + 2973 / 4);
        assert_eq!(tm.maximum(), 20433); // bump applies after maximum (SF order)
    }

    #[test]
    fn original_time_adjust_computed_once_per_game() {
        let options = default_options();
        let mut persist = TimePersist::default();
        let (_, _) = init_tm(60000, 600, 0, 20, &options, &mut persist);
        let first = persist.original_time_adjust;
        // Second `go` of the same game, different clock: OTA is reused, not
        // recomputed (fresh persist would give 1.1466...).
        let (tm, _) = init_tm(30000, 600, 0, 22, &options, &mut persist);
        assert_eq!(persist.original_time_adjust, first);
        assert_eq!(tm.optimum(), 1981);
        assert_eq!(tm.maximum(), 13615);
        // ucinewgame resets it (§3.7).
        persist = TimePersist::default();
        assert_eq!(persist.original_time_adjust, -1.0);
    }

    #[test]
    fn missing_own_clock_yields_zero_budgets() {
        // GUI sent only the opponent's clock: TM stays active
        // (side-independent predicate) but init returns early, so the first
        // check_time poll stops the search (§3.7 step 1).
        let mut limits = Limits::new();
        limits.time[Color::Black as usize] = 5000;
        assert!(limits.use_time_management());
        let mut tm = TimeManager::new();
        let mut persist = TimePersist::default();
        tm.init(
            &mut limits,
            Color::White,
            0,
            &default_options(),
            &mut persist,
        );
        assert_eq!(tm.optimum(), 0);
        assert_eq!(tm.maximum(), 0);
        assert_eq!(persist.original_time_adjust, -1.0);
    }

    #[test]
    fn nodestime_converts_clock_to_node_budget() {
        // nodestime=1000, time=60000ms: available_nodes=60M once at game
        // start; limits.time is rewritten to nodes; scaled_time stays 60000
        // human ms; budgets are node counts.
        let mut options = default_options();
        options.set("nodestime", Some("1000")).unwrap();
        let mut persist = TimePersist::default();
        let (tm, limits) = init_tm(60000, 600, 0, 20, &options, &mut persist);
        assert_eq!(limits.time[Color::White as usize], 60_000_000);
        assert_eq!(limits.inc[Color::White as usize], 600_000);
        assert_eq!(tm.optimum(), 5_395_671);
        assert_eq!(tm.maximum(), 37_084_446);
        // elapsed() now counts nodes, elapsed_ms() stays wall clock.
        assert_eq!(tm.elapsed(|| 12345), 12345);
        assert!(tm.elapsed_ms() >= 0);
    }

    #[test]
    fn nodestime_budget_advances_and_survives_within_a_game() {
        let mut options = default_options();
        options.set("nodestime", Some("1000")).unwrap();
        let mut persist = TimePersist::default();
        let mut limits = Limits::new();
        limits.time[Color::White as usize] = 60000;
        let mut tm = TimeManager::new();
        tm.init(&mut limits, Color::White, 0, &options, &mut persist);
        // Book a search of 5M nodes with a 1M-node increment credit.
        tm.advance_nodes_time(5_000_000 - 1_000_000);
        let mut limits2 = Limits::new();
        limits2.time[Color::White as usize] = 57500; // GUI clock, ignored now
        tm.init(&mut limits2, Color::White, 2, &options, &mut persist);
        assert_eq!(limits2.time[Color::White as usize], 56_000_000);
        // ucinewgame forgets the budget; the next init re-derives it.
        tm.new_game();
        let mut limits3 = Limits::new();
        limits3.time[Color::White as usize] = 60000;
        tm.init(&mut limits3, Color::White, 0, &options, &mut persist);
        assert_eq!(limits3.time[Color::White as usize], 60_000_000);
    }

    #[test]
    fn nodestime_opponent_clock_only_leaves_budget_untouched() {
        // §3.11 corner (SF timeman.cpp:79 / search.cpp:232): nodestime ON but
        // only the OPPONENT's clock was sent (engine is White, GUI sends
        // `btime` only). `init` early-returns before the conversion, so it
        // must leave `limits.npmsec == 0` — the post-search accounting gate —
        // and NOT seed `available_nodes`. Gating on `use_time_management()`
        // (true here, Black's clock is set) would corrupt the game-long node
        // budget to 0 and forfeit the rest of the game.
        let mut options = default_options();
        options.set("nodestime", Some("1000")).unwrap();
        let mut persist = TimePersist::default();
        let mut limits = Limits::new();
        limits.time[Color::Black as usize] = 60000; // opponent clock only
        let mut tm = TimeManager::new();
        tm.init(&mut limits, Color::White, 0, &options, &mut persist);
        // The gate SF (and our uci.rs) checks stays 0: no conversion ran.
        assert_eq!(limits.npmsec, 0);
        assert!(limits.use_time_management()); // but TM IS active (Black clock)
        // Next move White's clock arrives; the budget must still derive fresh
        // from the full clock (available_nodes was NOT clobbered to 0).
        let mut limits2 = Limits::new();
        limits2.time[Color::White as usize] = 60000;
        tm.init(&mut limits2, Color::White, 2, &options, &mut persist);
        assert_eq!(limits2.time[Color::White as usize], 60_000_000);
        assert_eq!(limits2.npmsec, 1000);
    }

    #[test]
    fn elapsed_uses_wall_clock_without_nodestime() {
        let mut persist = TimePersist::default();
        let (tm, _) = init_tm(60000, 0, 0, 0, &default_options(), &mut persist);
        // The nodes closure must not be consulted in wall-clock mode.
        let ms = tm.elapsed(|| panic!("nodes closure invoked without nodestime"));
        assert!(ms >= 0);
    }

    #[test]
    fn bare_limits_use_no_time_management() {
        let limits = Limits::new();
        assert!(!limits.use_time_management());
        assert!(limits.depth.is_none() && limits.nodes.is_none());
        assert!(!limits.infinite && !limits.ponder);
        assert!(limits.searchmoves.is_empty());
    }

    proptest! {
        /// §5 property: whenever our clock was given,
        /// `1 <= optimum <= maximum`, and the ~81%-of-clock cap is never
        /// exceeded except by the optimum floor itself.
        #[test]
        fn optimum_and_maximum_are_ordered(
            time in 1i64..=10_000_000,
            inc in 0i64..=100_000,
            movestogo in 0i32..=50,
            ply in 0i32..=500,
        ) {
            let mut persist = TimePersist::default();
            let (tm, _) = init_tm(time, inc, movestogo, ply, &default_options(), &mut persist);
            prop_assert!(tm.optimum() >= 1);
            prop_assert!(tm.optimum() <= tm.maximum());
            let clock_cap = (MAX_CLOCK_FRACTION * time as f64 - 10.0) as i64;
            prop_assert!(tm.maximum() <= tm.optimum().max(clock_cap));
        }

        /// §5 property: optimum is monotone (non-decreasing) in the clock,
        /// away from the sub-second horizon switch (fresh game each time).
        #[test]
        fn optimum_is_monotone_in_time(
            time in 1000i64..=1_000_000,
            delta in 0i64..=1_000_000,
            inc in 0i64..=10_000,
            ply in 0i32..=300,
        ) {
            let mut p1 = TimePersist::default();
            let mut p2 = TimePersist::default();
            let (tm1, _) = init_tm(time, inc, 0, ply, &default_options(), &mut p1);
            let (tm2, _) = init_tm(time + delta, inc, 0, ply, &default_options(), &mut p2);
            prop_assert!(tm1.optimum() <= tm2.optimum());
        }
    }
}
