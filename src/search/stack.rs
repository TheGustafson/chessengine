//! The per-ply search stack and PV storage (design/07 §3.2–§3.3, D18).
//!
//! One fixed `[Stack; STACK_SIZE]` array per `Worker`, with the search
//! operating at offset [`SS_OFFSET`] so relative reads from `ss-7` through
//! `ss+2` never leave the array:
//!
//! - `ss-7 … ss-1` padding below the root: continuation-history updates
//!   for the move at `ss-1` read back to `ss-6`; one more slot of slack
//!   matches Stockfish.
//! - `ss+2` above: each node zeroes its grandchild's `cutoff_cnt` on entry.
//!
//! We do NOT mimic Stockfish's `Stack*` pointer walk. Search functions
//! receive `ply` and index the worker-owned array directly — each access is
//! a short-lived, bounds-checked borrow (design/07 §3.2.2; `get_unchecked`
//! only after a profile justifies it, D14). Inside a node the common style
//! hoists the slot index once:
//!
//! ```text
//! let cur = ss(ply);
//! self.stack[cur].in_check = in_check;
//! let prev_move = self.stack[cur - 1].current_move;
//! self.stack[cur + 2].cutoff_cnt = 0;
//! ```
//!
//! PV storage is per-ply fixed lines ([`PvLine`]), worker-owned — option 3
//! of design/07 §3.3 — spliced by the borrow-splitting [`update_pv`].

use crate::history::ContHistId;
use crate::moves::Move;
use crate::types::{MAX_PLY, VALUE_NONE, Value};

/// Offset of the root's stack slot: `ss-7 … ss-1` padding lives below it.
pub const SS_OFFSET: usize = 7;

/// Length of the per-worker stack array (256 = `MAX_PLY + 10`): room for
/// the 7 padding slots below the root and `ss+2` reads at `ply == MAX_PLY`.
pub const STACK_SIZE: usize = MAX_PLY + 10;

/// Index of the stack slot for a given ply.
///
/// Bounds (design/07 §3.2.1): `ply <= MAX_PLY` ⇒
/// `ss(ply) + 2 <= 255 < STACK_SIZE` and `ss(ply) - 7 >= 0`.
#[inline]
pub const fn ss(ply: usize) -> usize {
    ply + SS_OFFSET
}

// The continuation-history key `ContHistId` — the `Copy`, one-load-resolvable
// key resolving to one `[Piece × Square]` slab of the conthist tensor
// (design/07 §3.2.3) — is owned by `crate::history` (design/09 §3.2) and
// imported above. The stack only stores it, defaults it to the inert
// `SENTINEL`, and compares it for equality.

/// One ply's worth of search state (design/07 §3.2.3).
///
/// Deliberately absent vs Stockfish: a `pv` pointer (per-ply [`PvLine`]s
/// remove the need, §3.3) and killers (D17). There is also no separate
/// `cont_corr_hist` field — the correction history's outer index is derived
/// from `cont_hist` at its use sites (doc 09 §3.9).
///
/// Must stay `Copy` and small: ≤ 40 bytes, so the whole per-worker stack is
/// ~10 KB (asserted below).
#[derive(Clone, Copy)]
pub struct Stack {
    /// [`Move::NULL`] when none (D2).
    pub current_move: Move,
    /// Singular-search exclusion (doc 10 writes).
    pub excluded_move: Move,
    /// Corrected eval; [`VALUE_NONE`] when unset.
    pub static_eval: Value,
    /// The move's history score (doc 10 consumes).
    pub stat_score: i32,
    pub move_count: i32,
    pub in_check: bool,
    /// Sticky "was ever on the PV" (stored to the TT).
    pub tt_pv: bool,
    pub tt_hit: bool,
    /// Fail-highs seen at this ply (LMR input, doc 10).
    pub cutoff_cnt: i32,
    /// LMR reduction applied to the child, in 1024ths of a ply (doc 10
    /// writes, the child reads it back).
    pub reduction: i32,
    /// Continuation-history key; see [`ContHistId`].
    pub cont_hist: ContHistId,
}

// Design/07 §3.2.3: `Stack` must stay small enough that the whole
// per-worker array is ~10 KB.
const _: () = assert!(size_of::<Stack>() <= 40);
const _: () = assert!(STACK_SIZE == 256);

impl Default for Stack {
    fn default() -> Self {
        Stack {
            current_move: Move::NULL,
            excluded_move: Move::NULL,
            static_eval: VALUE_NONE,
            stat_score: 0,
            move_count: 0,
            in_check: false,
            tt_pv: false,
            tt_hit: false,
            cutoff_cnt: 0,
            reduction: 0,
            cont_hist: ContHistId::SENTINEL,
        }
    }
}

/// Reset a worker's stack at the top of `iterative_deepening`
/// (design/07 §3.2.4): default-fill everything, and install the inert
/// continuation-history sentinel and "no eval" in the padding slots below
/// the root. (Search code receives `ply` as an argument everywhere; the
/// stack slot does not carry a copy.)
///
/// A single 10 KB fill per `go` — cheap, and never done inside the tree.
pub fn init_stack(stack: &mut [Stack; STACK_SIZE]) {
    stack.fill(Stack::default());
    // Padding slots below the root: reads through them must be valid but
    // inert. (`Default` already guarantees this; restated per the §3.2.4
    // contract so it survives any future `Default` change.)
    for slot in &mut stack[..SS_OFFSET] {
        slot.cont_hist = ContHistId::SENTINEL;
        slot.static_eval = VALUE_NONE;
    }
}

/// One principal-variation line: a fixed array of moves plus a length
/// (design/07 §3.3). The worker owns `MAX_PLY + 1` of these, indexed by the
/// same `ply` the stack uses; only PV/Root nodes ever touch them.
#[derive(Clone)]
pub struct PvLine {
    moves: [Move; MAX_PLY + 1],
    len: usize,
}

impl PvLine {
    pub const fn new() -> Self {
        PvLine {
            moves: [Move::NULL; MAX_PLY + 1],
            len: 0,
        }
    }

    /// O(1): only the length is reset — never memset the 494-byte move
    /// array (design/07 §6).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    /// `self = [first] ++ child`.
    pub fn load(&mut self, first: Move, child: &PvLine) {
        self.moves[0] = first;
        self.moves[1..=child.len].copy_from_slice(&child.moves[..child.len]);
        self.len = child.len + 1;
    }
}

impl Default for PvLine {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrow-splitting PV update: `pv_table[ply] = [m] ++ pv_table[ply + 1]`.
///
/// Called only in the `value > alpha` branch of a PV node, which in PVS
/// always follows a full-window PV-typed child search — so
/// `pv_table[ply + 1]` is fresh, never stale data from a zero-window child
/// (zero-window children are NonPV and never write it). See the invariants
/// in design/07 §3.3.
pub fn update_pv(pv_table: &mut [PvLine], ply: usize, m: Move) {
    let (head, tail) = pv_table.split_at_mut(ply + 1);
    head[ply].load(m, &tail[0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    fn new_stack() -> Box<[Stack; STACK_SIZE]> {
        Box::new([Stack::default(); STACK_SIZE])
    }

    // Distinct, recognizable move for a given ply.
    fn move_for(ply: usize) -> Move {
        Move::normal(
            Square::new((ply % 64) as u8),
            Square::new(((ply + 1) % 64) as u8),
        )
    }

    fn moves_eq(a: &[Move], b: &[Move]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
    }

    // --- Stack indexing (design/07 §5: offsets in bounds at both extremes) ---

    #[test]
    fn ss_offsets_in_bounds_at_ply_zero_and_max_ply() {
        // The full relative window the search ever touches, at both extremes.
        assert_eq!(ss(0) - 7, 0);
        assert_eq!(ss(MAX_PLY) + 2, STACK_SIZE - 1);

        // Exercise the actual array accesses: every slot in ss-7..=ss+2 at
        // ply 0 and ply MAX_PLY must be indexable without panicking.
        let mut stack = new_stack();
        for ply in [0, MAX_PLY] {
            let cur = ss(ply);
            for slot in cur - 7..=cur + 2 {
                stack[slot].cutoff_cnt = 1; // write
                assert_eq!(stack[slot].cutoff_cnt, 1); // read back
            }
        }
    }

    #[test]
    fn init_stack_pads_below_root() {
        let mut stack = new_stack();
        // Dirty everything first so init provably resets it.
        for slot in stack.iter_mut() {
            slot.static_eval = 123;
            slot.current_move = move_for(5);
            slot.tt_pv = true;
        }
        init_stack(&mut stack);

        // Padding slots below the root carry the inert sentinel and no eval.
        for slot in &stack[..SS_OFFSET] {
            assert!(slot.cont_hist == ContHistId::SENTINEL);
            assert_eq!(slot.static_eval, VALUE_NONE);
        }
        // Default-fill state: no stale moves or flags anywhere.
        for slot in stack.iter() {
            assert!(slot.current_move == Move::NULL);
            assert!(slot.excluded_move == Move::NULL);
            assert!(!slot.tt_pv);
        }
    }

    #[test]
    fn stack_is_small() {
        // Design/07 §3.2.3: ≤ 40 bytes so the whole stack is ~10 KB.
        assert!(size_of::<Stack>() <= 40);
        assert!(size_of::<[Stack; STACK_SIZE]>() <= 10 * 1024);
    }

    // --- PvLine (design/07 §5: update/extraction round-trips) ---

    #[test]
    fn pv_line_load_splices_first_plus_child() {
        let mut child = PvLine::new();
        child.load(move_for(1), &PvLine::new()); // child = [m1]
        let mut parent = PvLine::new();
        parent.load(move_for(0), &child); // parent = [m0] ++ [m1]

        assert!(moves_eq(parent.as_slice(), &[move_for(0), move_for(1)]));
    }

    #[test]
    fn pv_line_clear_empties_without_touching_moves() {
        let mut line = PvLine::new();
        line.load(move_for(3), &PvLine::new());
        assert_eq!(line.as_slice().len(), 1);
        line.clear();
        assert!(line.as_slice().is_empty());
        // A cleared line splices as an empty child.
        let mut parent = PvLine::new();
        parent.load(move_for(2), &line);
        assert!(moves_eq(parent.as_slice(), &[move_for(2)]));
    }

    #[test]
    fn pv_line_load_at_full_capacity() {
        // A child of maximal length MAX_PLY still fits: the parent line
        // becomes MAX_PLY + 1 moves, exactly the array size.
        let mut child = PvLine::new();
        for ply in (1..=MAX_PLY).rev() {
            let grandchild = child.clone();
            child.load(move_for(ply), &grandchild);
        }
        assert_eq!(child.as_slice().len(), MAX_PLY);
        let mut root = PvLine::new();
        root.load(move_for(0), &child);
        assert_eq!(root.as_slice().len(), MAX_PLY + 1);
        assert!(root.as_slice()[0] == move_for(0));
        assert!(root.as_slice()[MAX_PLY] == move_for(MAX_PLY));
    }

    #[test]
    fn update_pv_round_trip_through_scripted_tree() {
        // Simulate a 4-ply PV being backed up from the leaf: each node
        // clears its line on entry and, on value > alpha, splices its move
        // in front of the child's line.
        const DEPTH: usize = 4;
        let mut pv_table = vec![PvLine::new(); MAX_PLY + 1];
        // Poison the child rows to prove update_pv overwrites, not appends.
        for row in &mut pv_table[..DEPTH] {
            row.load(move_for(63), &PvLine::new());
        }
        pv_table[DEPTH].clear(); // leaf's child row
        for ply in (0..DEPTH).rev() {
            pv_table[ply].clear(); // node entry
            update_pv(&mut pv_table, ply, move_for(ply)); // value > alpha
        }
        // Root PV extraction equals the concatenation of chosen moves.
        let want: Vec<Move> = (0..DEPTH).map(move_for).collect();
        assert!(moves_eq(pv_table[0].as_slice(), &want));
        assert!(moves_eq(pv_table[1].as_slice(), &want[1..]));
    }
}
