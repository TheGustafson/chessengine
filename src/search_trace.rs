//! Bounded, feature-gated observational search telemetry.
//!
//! This is deliberately not part of the search controller. It samples the
//! classical controller's actual LMR decisions and records their consequences
//! after the search is complete. The search path performs no I/O; rows are
//! buffered on the main worker and appended at the end of `go`.

use crate::eval::nnue::L1;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

pub const SCHEMA: &str = "archon.search-lmr-observation.v3";

#[derive(Debug)]
pub struct LmrObservation {
    pub position_key: u64,
    /// Full parent state for cold counterfactual replay. FEN preserves pieces,
    /// side, castling, en-passant, and rule-50; repetition history is not
    /// representable and is explicitly absent from this bootstrap contract.
    pub fen: String,
    pub latent: Box<[i16; L1 * 2]>,
    pub raw_move: u16,
    pub side_to_move: u8,
    pub moving_piece: u8,
    pub captured_piece: u8,
    pub promotion_piece: u8,
    pub depth: i32,
    pub ply: u16,
    pub alpha: i32,
    pub beta: i32,
    pub static_eval: i32,
    pub best_value_before: i32,
    pub node_kind: u8,
    pub in_check: bool,
    pub improving: bool,
    pub cut_node: bool,
    pub tt_hit: bool,
    pub tt_depth: i32,
    pub tt_bound: u8,
    pub tt_move: bool,
    pub move_index: u16,
    pub capture: bool,
    pub gives_check: bool,
    pub stat_score: i32,
    pub base_reduction: i32,
    pub final_reduction: i32,
    pub extension: i32,
    pub nominal_child_depth: i32,
    pub scout_depth: i32,
    pub scout_score: i32,
    pub final_score: i32,
    pub scout_nodes: u64,
    pub research_nodes: u64,
    pub pv_research_nodes: u64,
    pub total_nodes: u64,
    pub lmr_research: bool,
    pub pv_research: bool,
    pub alpha_raise: bool,
    pub beta_cutoff: bool,
}

#[derive(Debug)]
pub struct SearchTrace {
    path: PathBuf,
    root_position_key: u64,
    sample_log2: u8,
    min_depth: i32,
    max_rows: usize,
    /// Debugger mode: omit the 1024-value latent from the output. The latent is
    /// only meaningful to the D41 allocation datagen; for walking a failing
    /// search it is pure noise and ~2 kB of JSON per row.
    no_latent: bool,
    /// Debugger mode: also record `move_count == 1`. The datagen deliberately
    /// skips the first move (it is never LMR-reduced), but when debugging a
    /// blunder the first move is exactly the one we are arguing with.
    all_moves: bool,
    /// Debug counter: how many times the sampler was consulted at all.
    considered: std::cell::Cell<u64>,
    rows: Vec<LmrObservation>,
}

impl SearchTrace {
    /// Enable tracing only when an explicit output path is provided.
    pub fn from_env(root_position_key: u64) -> Option<Self> {
        let path = std::env::var_os("ARCHON_SEARCH_TRACE_PATH")?;
        let sample_log2 = env_parse("ARCHON_SEARCH_TRACE_SAMPLE_LOG2", 10u8).min(63);
        let min_depth = env_parse("ARCHON_SEARCH_TRACE_MIN_DEPTH", 3i32).max(2);
        let max_rows = env_parse("ARCHON_SEARCH_TRACE_MAX_ROWS", 4096usize).clamp(1, 1_000_000);
        Some(Self {
            path: PathBuf::from(path),
            root_position_key,
            sample_log2,
            min_depth,
            max_rows,
            no_latent: std::env::var_os("ARCHON_SEARCH_TRACE_NO_LATENT").is_some(),
            all_moves: std::env::var_os("ARCHON_SEARCH_TRACE_ALL_MOVES").is_some(),
            considered: std::cell::Cell::new(0),
            rows: Vec::with_capacity(max_rows.min(4096)),
        })
    }

    /// Whether `move_count == 1` should be recorded (debugger mode).
    #[inline]
    pub fn all_moves(&self) -> bool {
        self.all_moves
    }

    #[inline]
    pub fn should_sample(&self, key: u64, raw_move: u16, depth: i32, ply: usize) -> bool {
        self.considered.set(self.considered.get() + 1);
        if depth < self.min_depth || self.rows.len() >= self.max_rows {
            return false;
        }
        let mut x = key
            ^ (u64::from(raw_move) << 17)
            ^ ((depth as u64) << 49)
            ^ (ply as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        // SplitMix64 finalizer: deterministic and insensitive to traversal
        // adjacency, unlike sampling on the global node counter.
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        x & ((1u64 << self.sample_log2).wrapping_sub(1)) == 0
    }

    #[inline]
    pub fn push(&mut self, row: LmrObservation) {
        if self.rows.len() < self.max_rows {
            self.rows.push(row);
        }
    }

    pub fn flush(&mut self) -> io::Result<usize> {
        let count = self.rows.len();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let empty = !self.path.exists() || self.path.metadata()?.len() == 0;
        let mut out = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        if empty {
            writeln!(
                out,
                "{{\"schema\":\"{SCHEMA}\",\"kind\":\"header\",\"latent_dim\":{},\"sample_log2\":{},\"min_depth\":{},\"max_rows_per_go\":{}}}",
                L1 * 2,
                self.sample_log2,
                self.min_depth,
                self.max_rows,
            )?;
        }
        writeln!(
            out,
            "{{\"schema\":\"{SCHEMA}\",\"kind\":\"go\",\"root_position_key\":{},\"rows\":{count},\"considered\":{}}}",
            self.root_position_key,
            self.considered.get(),
        )?;
        for row in self.rows.drain(..) {
            write!(
                out,
                "{{\"schema\":\"{SCHEMA}\",\"kind\":\"row\",\"position_key\":{},\"fen\":\"{}\",\"latent\":[",
                row.position_key, row.fen,
            )?;
            if !self.no_latent {
                for (i, value) in row.latent.iter().enumerate() {
                    if i != 0 {
                        out.write_all(b",")?;
                    }
                    write!(out, "{value}")?;
                }
            }
            writeln!(
                out,
                "],\"raw_move\":{},\"side_to_move\":{},\"moving_piece\":{},\"captured_piece\":{},\"promotion_piece\":{},\"depth\":{},\"ply\":{},\"alpha\":{},\"beta\":{},\"static_eval\":{},\"best_value_before\":{},\"node_kind\":{},\"in_check\":{},\"improving\":{},\"cut_node\":{},\"tt_hit\":{},\"tt_depth\":{},\"tt_bound\":{},\"tt_move\":{},\"move_index\":{},\"capture\":{},\"gives_check\":{},\"stat_score\":{},\"base_reduction\":{},\"final_reduction\":{},\"extension\":{},\"nominal_child_depth\":{},\"scout_depth\":{},\"scout_score\":{},\"final_score\":{},\"scout_nodes\":{},\"research_nodes\":{},\"pv_research_nodes\":{},\"total_nodes\":{},\"lmr_research\":{},\"pv_research\":{},\"alpha_raise\":{},\"beta_cutoff\":{}}}",
                row.raw_move,
                row.side_to_move,
                row.moving_piece,
                row.captured_piece,
                row.promotion_piece,
                row.depth,
                row.ply,
                row.alpha,
                row.beta,
                row.static_eval,
                row.best_value_before,
                row.node_kind,
                row.in_check,
                row.improving,
                row.cut_node,
                row.tt_hit,
                row.tt_depth,
                row.tt_bound,
                row.tt_move,
                row.move_index,
                row.capture,
                row.gives_check,
                row.stat_score,
                row.base_reduction,
                row.final_reduction,
                row.extension,
                row.nominal_child_depth,
                row.scout_depth,
                row.scout_score,
                row.final_score,
                row.scout_nodes,
                row.research_nodes,
                row.pv_research_nodes,
                row.total_nodes,
                row.lmr_research,
                row.pv_research,
                row.alpha_raise,
                row.beta_cutoff,
            )?;
        }
        out.flush()?;
        Ok(count)
    }
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sampler_respects_zero_log2() {
        let trace = SearchTrace {
            path: PathBuf::from("unused"),
            root_position_key: 7,
            sample_log2: 0,
            min_depth: 3,
            max_rows: 2,
            rows: Vec::new(),
        };
        assert!(trace.should_sample(1, 2, 3, 4));
        assert!(!trace.should_sample(1, 2, 2, 4));
    }
}
