#!/usr/bin/env python3
"""Scale the tail of a run spec's `lr_ramp` by a constant factor.

WHY A TOOL AND NOT A ONE-OFF EDIT
    This is the third time an owner has asked for "+25% LR from superbatch N"
    mid-run. Done by hand it is an easy operation to get subtly wrong in ways
    that no error message reports: scaling from the wrong superbatch, reshaping
    the decay instead of scaling it, or clobbering the terminal anneal so the
    run never actually reaches LR 0. All three are silent — training continues
    and the damage only shows up in a gate 100 superbatches later.

WHAT IT PRESERVES
    The ramp is a list of [superbatch, lr] pairs. Scaling multiplies the lr of
    every entry at or after `--from-sb` by `--factor` and touches nothing else,
    so the DECAY PROFILE is unchanged — the curve keeps its shape and just sits
    higher. The terminal anneal to true zero survives for free because
    0 * factor == 0, which is the property that makes scaling the right
    operation here rather than re-deriving a new cosine.

    Entries BEFORE `--from-sb` are never touched: they are history, they
    describe superbatches the run has already trained through, and rewriting
    them would make the spec disagree with what actually happened.

CHECKS (all fatal, because each one has a silent failure mode)
    * `--from-sb` must exist as an exact key in the ramp. A resume lands on a
      real checkpoint superbatch, and asking to scale from a superbatch that is
      not a ramp entry means the caller has the wrong number.
    * the join must be the only discontinuity: the ratio between the last
      untouched entry and the first scaled one is reported so the caller sees
      the actual jump it is introducing.
    * a zero-valued tail entry must stay zero.

USAGE
    python3 tools/ops/train/scale_lr_tail.py SPEC.json --from-sb 700 --factor 1.25
    ... --dry-run     print the change and write nothing
"""

import argparse
import json
import pathlib
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=pathlib.Path)
    ap.add_argument("--from-sb", type=int, required=True)
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default="", help="appended to the spec's _lr_bump field")
    a = ap.parse_args()

    spec = json.loads(a.spec.read_text())
    ramp = spec.get("lr_ramp")
    if not ramp:
        print("FATAL: spec has no lr_ramp", file=sys.stderr)
        return 2

    keys = [int(sb) for sb, _ in ramp]
    if a.from_sb not in keys:
        near = min(keys, key=lambda k: abs(k - a.from_sb))
        print(
            f"FATAL: sb{a.from_sb} is not a ramp entry (nearest is sb{near}). "
            "Scale from a real checkpoint superbatch.",
            file=sys.stderr,
        )
        return 2

    before = [(sb, lr) for sb, lr in ramp if sb < a.from_sb]
    after = [(sb, lr) for sb, lr in ramp if sb >= a.from_sb]

    last_old = before[-1][1] if before else None
    first_old = after[0][1]
    first_new = first_old * a.factor

    scaled = [[sb, lr * a.factor] for sb, lr in after]

    # the terminal anneal must still land on exactly zero
    zeros_before = sum(1 for _, lr in after if lr == 0.0)
    zeros_after = sum(1 for _, lr in scaled if lr == 0.0)
    if zeros_before != zeros_after:
        print("FATAL: scaling changed the number of zero-LR entries", file=sys.stderr)
        return 2

    print(f"spec        {a.spec}")
    print(f"ramp        {len(ramp)} entries, sb{keys[0]}..sb{keys[-1]}")
    print(f"scaling     {len(after)} entries from sb{a.from_sb} by x{a.factor}")
    if last_old is not None:
        print(f"join        sb{before[-1][0]} {last_old:.6e} -> sb{a.from_sb} {first_new:.6e}"
              f"   (step x{first_new / last_old:.4f})")
    print(f"first       sb{a.from_sb}  {first_old:.6e} -> {first_new:.6e}")
    print(f"last        sb{scaled[-1][0]}  {after[-1][1]:.6e} -> {scaled[-1][1]:.6e}")
    print(f"zero tail   {zeros_after} entries at exactly 0.0 (preserved)")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    bak = a.spec.with_suffix(a.spec.suffix + f".pre-sb{a.from_sb}-x{a.factor}")
    shutil.copy2(a.spec, bak)

    spec["lr_ramp"] = [[sb, lr] for sb, lr in before] + scaled
    prev = spec.get("_lr_bump", "")
    spec["_lr_bump"] = (prev + " | " if prev else "") + (
        a.note or f"x{a.factor} from sb{a.from_sb}: remaining cosine SCALED, "
        "not reshaped; terminal anneal to true zero preserved."
    )
    a.spec.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nwrote {a.spec}  (backup {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
