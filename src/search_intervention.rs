//! Explicit single-action LMR intervention for deterministic oracle replay.
//!
//! This module is compiled only with `search-control` and is inert unless all
//! `ARCHON_LMR_OVERRIDE_*` variables are present. It is not a playing option:
//! each process may alter one matched scout depth and writes one result file.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::moves::Move;
use crate::types::{Depth, Key, Value};

pub const SCHEMA: &str = "archon.lmr-intervention.v1";

#[derive(Debug, Clone, Copy)]
pub struct InterventionOutcome {
    pub alpha: Value,
    pub beta: Value,
    pub classical_scout_depth: Depth,
    pub applied_scout_depth: Depth,
    pub scout_score: Value,
    pub final_score: Value,
    pub scout_nodes: u64,
    pub research_nodes: u64,
    pub pv_research_nodes: u64,
    pub total_nodes: u64,
    pub lmr_research: bool,
    pub pv_research: bool,
}

#[derive(Debug)]
pub struct LmrIntervention {
    key: Key,
    raw_move: u16,
    depth: Depth,
    ply: usize,
    move_index: i32,
    occurrence: u32,
    requested_scout_depth: Depth,
    result_path: PathBuf,
    matching_occurrences: u32,
    applied: bool,
    outcome: Option<InterventionOutcome>,
}

impl LmrIntervention {
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(result_path) = std::env::var_os("ARCHON_LMR_OVERRIDE_RESULT") else {
            return Ok(None);
        };
        let key_text = required("ARCHON_LMR_OVERRIDE_KEY")?;
        let key = if let Some(hex) = key_text.strip_prefix("0x") {
            u64::from_str_radix(hex, 16)
        } else {
            key_text.parse()
        }
        .map_err(|_| "invalid ARCHON_LMR_OVERRIDE_KEY".to_string())?;
        Ok(Some(Self {
            key,
            raw_move: parse("ARCHON_LMR_OVERRIDE_MOVE")?,
            depth: parse("ARCHON_LMR_OVERRIDE_DEPTH")?,
            ply: parse("ARCHON_LMR_OVERRIDE_PLY")?,
            move_index: parse("ARCHON_LMR_OVERRIDE_MOVE_INDEX")?,
            occurrence: parse("ARCHON_LMR_OVERRIDE_OCCURRENCE")?,
            requested_scout_depth: parse("ARCHON_LMR_OVERRIDE_SCOUT_DEPTH")?,
            result_path: PathBuf::from(result_path),
            matching_occurrences: 0,
            applied: false,
            outcome: None,
        }))
    }

    /// Return the requested scout depth only for the selected occurrence.
    #[inline]
    pub fn request(
        &mut self,
        key: Key,
        mv: Move,
        depth: Depth,
        ply: usize,
        move_index: i32,
    ) -> Option<Depth> {
        if self.applied
            || key != self.key
            || mv.raw16() != self.raw_move
            || depth != self.depth
            || ply != self.ply
            || move_index != self.move_index
        {
            return None;
        }
        let occurrence = self.matching_occurrences;
        self.matching_occurrences += 1;
        if occurrence != self.occurrence || self.requested_scout_depth < 1 {
            return None;
        }
        self.applied = true;
        Some(self.requested_scout_depth)
    }

    pub fn finish(&mut self, outcome: InterventionOutcome) {
        if self.applied && self.outcome.is_none() {
            self.outcome = Some(outcome);
        }
    }

    pub fn flush(&self) -> io::Result<()> {
        if let Some(parent) = self.result_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = File::create(&self.result_path)?;
        write!(
            out,
            "{{\"schema\":\"{SCHEMA}\",\"target\":{{\"key\":{},\"raw_move\":{},\"depth\":{},\"ply\":{},\"move_index\":{},\"occurrence\":{},\"requested_scout_depth\":{}}},\"matching_occurrences\":{},\"applied\":{}",
            self.key,
            self.raw_move,
            self.depth,
            self.ply,
            self.move_index,
            self.occurrence,
            self.requested_scout_depth,
            self.matching_occurrences,
            self.applied,
        )?;
        if let Some(row) = self.outcome {
            write!(
                out,
                ",\"outcome\":{{\"alpha\":{},\"beta\":{},\"classical_scout_depth\":{},\"applied_scout_depth\":{},\"scout_score\":{},\"final_score\":{},\"scout_nodes\":{},\"research_nodes\":{},\"pv_research_nodes\":{},\"total_nodes\":{},\"lmr_research\":{},\"pv_research\":{}}}",
                row.alpha,
                row.beta,
                row.classical_scout_depth,
                row.applied_scout_depth,
                row.scout_score,
                row.final_score,
                row.scout_nodes,
                row.research_nodes,
                row.pv_research_nodes,
                row.total_nodes,
                row.lmr_research,
                row.pv_research,
            )?;
        }
        writeln!(out, "}}")?;
        out.flush()
    }
}

fn required(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("missing {name}"))
}

fn parse<T: std::str::FromStr>(name: &str) -> Result<T, String> {
    required(name)?
        .parse()
        .map_err(|_| format!("invalid {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occurrence_match_is_single_use() {
        let mut intervention = LmrIntervention {
            key: 7,
            raw_move: 9,
            depth: 6,
            ply: 3,
            move_index: 4,
            occurrence: 1,
            requested_scout_depth: 2,
            result_path: PathBuf::from("unused"),
            matching_occurrences: 0,
            applied: false,
            outcome: None,
        };
        let mv = Move::from_raw16(9);
        assert_eq!(intervention.request(7, mv, 6, 3, 4), None);
        assert_eq!(intervention.request(7, mv, 6, 3, 4), Some(2));
        assert_eq!(intervention.request(7, mv, 6, 3, 4), None);
    }
}
