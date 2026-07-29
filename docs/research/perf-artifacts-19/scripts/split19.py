#!/usr/bin/env python3
"""perf study 19 — the MEASUREMENT-ONLY stage split of `at_full`.

Study 17 item 1b's second half, still owed after study 18 and now the top-ranked
open block.  `at_full` is 470.82 i/n self at 0.4151 calls/node = 1134
instructions per evaluated ply, and it inlines FOUR distinct stages that nobody
has ever priced separately:

    1. `compute`        — the per-perspective ancestor walk + the child loop
    2. `build_deltas`   — event generation (its callee `record_square_touch` is
                          ALREADY a separate symbol, so this prices the replay
                          orchestration only)
    3. `resolve_both` / `resolve_into` — per-event index resolution, both
                          perspectives, plus the psqt row accumulate
    4. the fold         — the empty-list chunked copy, the psqt bucket adds, and
                          the loop around `apply_i8_deltas_into` (also already a
                          separate symbol)

This patch adds `#[inline(never)]` to 1-3 and extracts 4 into a free function.
IT IS NOT A SHIPPING CHANGE — it exists to make callgrind able to see the
boundary.  Revert with `git checkout -- src/eval/nnue/sota_threats.rs`.

CAVEAT, stated up front and the same one study 18 §4.1 carried for its Stockfish
`noinline` rebuild: an outlined stage pays call/prologue overhead the inlined
build never paid, and `fold_one` below takes seven arguments (over the SysV
six-register limit study 17 §5.1c priced at 14.2 i/n).  The bound on the total
distortion is the WHOLE-BINARY delta between the base and split profiles; quote
that alongside the per-stage numbers.

Usage:  python3 split19.py [--revert]
"""

import pathlib
import subprocess
import sys

SRC = pathlib.Path("src/eval/nnue/sota_threats.rs")

# (anchor, replacement) — each anchor must appear EXACTLY once.
EDITS = [
    # 1. compute
    (
        "    fn compute(&mut self, ply: usize) {",
        "    #[inline(never)]\n    fn compute(&mut self, ply: usize) {",
    ),
    # 2. build_deltas
    (
        "fn build_deltas(buf: &mut DeltaBuf, tab: &AttackTables, dp: &DirtyPiece, post: &ThreatPosition) {",
        "#[inline(never)]\nfn build_deltas(buf: &mut DeltaBuf, tab: &AttackTables, dp: &DirtyPiece, post: &ThreatPosition) {",
    ),
    # 3a. resolve_both
    (
        "#[allow(clippy::too_many_arguments)]\nfn resolve_both(",
        "#[allow(clippy::too_many_arguments)]\n#[inline(never)]\nfn resolve_both(",
    ),
    # 3b. resolve_into
    (
        "fn resolve_into(\n    events: &[ThreatDelta],",
        "#[inline(never)]\nfn resolve_into(\n    events: &[ThreatDelta],",
    ),
]

# 4. the fold — extracted wholesale out of `compute`'s child loop.
FOLD_OLD = """            for (p, &needs_fold) in need.iter().enumerate() {
                if !needs_fold {
                    continue;
                }
                let (add_len, sub_len) = lens[p];
                let (before, after) = self.stack.split_at_mut(child);
                let parent = &before[child - 1];
                let entry = &mut after[0];
                if add_len == 0 && sub_len == 0 {"""

FOLD_NEW = """            for (p, &needs_fold) in need.iter().enumerate() {
                if !needs_fold {
                    continue;
                }
                let (add_len, sub_len) = lens[p];
                let (before, after) = self.stack.split_at_mut(child);
                let parent = &before[child - 1];
                let entry = &mut after[0];
                fold_one(
                    entry,
                    parent,
                    weights,
                    &self.add_scratch[p][..add_len],
                    &self.sub_scratch[p][..sub_len],
                    &psqt_deltas[p],
                    p,
                );
                continue;
                #[allow(unreachable_code)]
                if add_len == 0 && sub_len == 0 {"""

FOLD_FN = """
/// STUDY 19 MEASUREMENT SPLIT ONLY — the phase-2 fold, outlined so callgrind can
/// price it apart from `compute`'s walk and the two generation stages.
#[inline(never)]
#[allow(clippy::too_many_arguments, dead_code)]
fn fold_one(
    entry: &mut ThreatEntry,
    parent: &ThreatEntry,
    weights: &[i8],
    adds: &[usize],
    subs: &[usize],
    psqt_delta: &[i32; BUCKETS],
    p: usize,
) {
    if adds.is_empty() && subs.is_empty() {
        const { assert!(ACC.is_multiple_of(32)) };
        for (d, s) in entry.acc[p]
            .chunks_exact_mut(32)
            .zip(parent.acc[p].chunks_exact(32))
        {
            d.copy_from_slice(s);
        }
        entry.psqt[p] = parent.psqt[p];
        entry.fold[p] = Fold::Done;
        return;
    }
    apply_i8_deltas_into(&mut entry.acc[p], &parent.acc[p], weights, adds, subs);
    for (dst, (&src, &d)) in entry.psqt[p]
        .iter_mut()
        .zip(parent.psqt[p].iter().zip(psqt_delta.iter()))
    {
        *dst = src + d;
    }
    entry.fold[p] = Fold::Done;
}
"""


def main():
    if "--revert" in sys.argv:
        subprocess.run(["git", "checkout", "--", str(SRC)], check=True)
        print("reverted", SRC)
        return
    text = SRC.read_text()
    for old, new in EDITS:
        n = text.count(old)
        assert n == 1, f"anchor appears {n} times, expected 1:\n{old[:80]}"
        text = text.replace(old, new)
    assert text.count(FOLD_OLD) == 1, "fold anchor not unique"
    text = text.replace(FOLD_OLD, FOLD_NEW)
    # Append the outlined fold beside `discharge_refresh`.
    marker = "\n/// Resolve the perspective-agnostic raw `events` to one perspective's"
    assert text.count(marker) == 1
    text = text.replace(marker, FOLD_FN + marker)
    SRC.write_text(text)
    print(f"applied 5 stage splits to {SRC}")


if __name__ == "__main__":
    main()
