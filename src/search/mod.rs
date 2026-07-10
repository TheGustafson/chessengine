//! Search — iterative deepening driver, PVS nodes, quiescence
//! (`design/07-search-core.md`) and the pruning/extension heuristics
//! (`design/10-pruning-extensions.md`).
//!
//! Landed so far (design/07 §4 steps 1–3): the [`params`] constants module
//! (P0, design/01 §4 step 4), the per-ply search [`stack`] with PV lines
//! (§3.2–§3.3, D18), and [`rootmoves`] bookkeeping (§3.4). The `Worker`,
//! negamax/qsearch skeleton, and ID loop are the remaining P5 steps.

pub mod params;
pub mod rootmoves;
pub mod stack;

pub use rootmoves::{RootMove, RootMoves, sort_descending};
pub use stack::{ContHistId, PvLine, SS_OFFSET, STACK_SIZE, Stack, init_stack, ss, update_pv};
