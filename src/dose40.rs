//! Perf study 40's **dose amplifier** for `partial_insertion_sort`
//! (`--features dosesort40`, never shipped, never gated).
//!
//! # Why this exists
//!
//! Perf study 34 built a branchless insertion window (`--features
//! insertwin34`), verified it node-identical, and measured it exactly:
//! **+0.702% instructions for −0.530% branches and −4% of all engine branch
//! mispredicts**. That is an instructions-for-mispredicts trade, and only
//! `cycles:u` adjudicates a trade. Its own op-count model prices the trade at
//! ≈ **−0.17% of cycles** — and six independent nulls (studies 33–39) put the
//! `cycles:u` floor on this box at **±1–9% per paired round**. The arm is an
//! order of magnitude below the instrument. Study 34 said so honestly and
//! shipped the arm OFF; study 38 hit the same wall on a different site.
//!
//! # What it does
//!
//! Every `partial_insertion_sort` call first replays `P40_DOSE` **real earlier
//! inputs** through the same sort, on scratch, and throws the results away. The
//! search tree is untouched (verified: node count and eval hash identical at
//! every dose), so a dose series measures the same quantity at (D+1)× its size
//! and the slope regresses back to one call.
//!
//! Two design points are load-bearing, both paid for by earlier studies:
//!
//! * **Real earlier inputs, from a ring — not the current buffer repeated.**
//!   Repeating one buffer D times hands the branch predictor the identical
//!   outcome sequence D times in a row, which it learns; the walk's mispredicts
//!   would collapse and the amplifier would understate exactly the thing the
//!   arm is buying. A ring of previously-seen buffers gives every repetition a
//!   different real input, so a replay costs what a real call costs. The
//!   mispredict counter checks this rather than assuming it (study 40 §3.2).
//! * **The dose is a RUNTIME knob, so one binary serves the whole series** and
//!   code layout is differenced away by construction (study 39 handover §10.1).
//!
//! The ring is filled with the buffer as it arrives — *before* the real sort —
//! so replayed inputs are unsorted, exactly like live ones.

use std::cell::RefCell;

use crate::movepick::{ExtMove, partial_insertion_sort_inner};
use crate::moves::Move;

const EMPTY: ExtMove = ExtMove::new(Move::NULL, 0);

/// Slots in the replay ring. Bigger than any dose used, so consecutive
/// repetitions never see the same slot twice within a call.
const RING: usize = 32;
/// Longest buffer the ring stores. The mean scored-quiet list is 22.6 elements
/// (study 34 §3) and captures are 1.5; a call longer than this is simply not
/// captured, which costs the amplifier a little fidelity in the far tail and
/// costs it identically in both arms.
const SLOT: usize = 96;

struct Slot {
    buf: [ExtMove; SLOT],
    len: usize,
    limit: i32,
}

struct Ring {
    slots: Vec<Slot>,
    scratch: [ExtMove; SLOT],
    cursor: usize,
}

thread_local! {
    static RING_TLS: RefCell<Ring> = RefCell::new(Ring {
        slots: (0..RING)
            .map(|_| Slot { buf: [EMPTY; SLOT], len: 0, limit: i32::MIN })
            .collect(),
        scratch: [EMPTY; SLOT],
        cursor: 0,
    });
}

/// Dose, read once from `P40_DOSE` (default 0 — an inert build).
fn dose() -> usize {
    use std::sync::OnceLock;
    static DOSE: OnceLock<usize> = OnceLock::new();
    *DOSE.get_or_init(|| {
        std::env::var("P40_DOSE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Replay `dose()` earlier inputs through the sort, then record this one.
///
/// Called from `partial_insertion_sort` BEFORE the real sort, so `moves` is
/// still in generation order.
pub(crate) fn replay(moves: &[ExtMove], limit: i32) {
    let d = dose();
    if d == 0 {
        return;
    }
    RING_TLS.with(|r| {
        let r = &mut *r.borrow_mut();
        for k in 0..d {
            let i = (r.cursor + k) % RING;
            let (len, lim) = (r.slots[i].len, r.slots[i].limit);
            if len == 0 {
                continue;
            }
            r.scratch[..len].copy_from_slice(&r.slots[i].buf[..len]);
            partial_insertion_sort_inner(&mut r.scratch[..len], lim);
            // The result escapes, so no store is dead and nothing folds away.
            std::hint::black_box(&r.scratch);
        }
        if moves.len() <= SLOT {
            let i = r.cursor % RING;
            r.slots[i].buf[..moves.len()].copy_from_slice(moves);
            r.slots[i].len = moves.len();
            r.slots[i].limit = limit;
        }
        r.cursor = r.cursor.wrapping_add(1);
    });
}
