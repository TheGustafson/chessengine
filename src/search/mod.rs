//! Search — iterative deepening driver, PVS nodes, quiescence
//! (`design/07-search-core.md`) and the pruning/extension heuristics
//! (`design/10-pruning-extensions.md`).
//!
//! P0 seeds only the [`params`] constants module (design/01 §4 step 4);
//! the `Worker`, search stack, and node loop land in P5 per design/07 §4.

pub mod params;
