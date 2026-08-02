#!/usr/bin/env python3
"""Transplant a 1280-wide HalfKA+FullThreats checkpoint into a 768-wide
HalfKA-ONLY warm start.

PURPOSE
    Produce the warm checkpoint for the owner's "fastest net" experiment:
    drop the FullThreats channel entirely and narrow the accumulator 1280 -> 768,
    keeping the learned HalfKA function rather than starting from scratch.

USAGE
    uv run --no-project --with numpy==2.5.1 python jax/tools/transplant_hka768.py \
        IN-raw.bin OUT-raw.bin [--width 768] [--rank head|ft|product] [--dry-run]

WHAT IT DOES
    drop   l0_threatw   (60,720 x W)  and  l0_psqt_thw (60,720 x 8)
           -> 72.9% of the input rows, ~78M of ~108M parameters
    slice  every tensor carrying the accumulator axis, on ONE shared column
           index set, so the head keeps reading the columns it was trained on
    keep   tensors with no accumulator axis (l0_psqt_hkw, l2w/l2b, polmove*)

WHICH 768 COLUMNS — the only real decision here
    A column is worth keeping in proportion to how much it influences the
    OUTPUT, not how large its feature weights happen to be. Three rankings:

      head     ||l1w[:, col, :]||  summed over buckets  (DEFAULT)
               How much the head actually reads that accumulator unit. A column
               the head ignores is dead no matter how big its FT weights are.
      ft       ||l0_pstw[:, col]||
               How much HalfKA signal the column carries. Blind to whether the
               head uses it.
      product  head x ft
               Both, for when they disagree.

    `head` is the default because the failure mode we care about is keeping a
    high-norm column the head has learned to ignore.

    NB: this is an INITIALISATION, not a distillation. The 768 net retrains from
    here, so the ranking sets the starting point rather than the final function.
    Do not over-tune it; do check that the three rankings disagree less than the
    subsequent training moves things.

WHY NOT JUST TAKE THE FIRST 768
    Accumulator columns are not ordered by importance -- nothing in training
    imposes that -- so the first 768 is an arbitrary sample. Ranked selection
    costs one pass and is strictly better-informed. `--rank first` is available
    as a NULL ARM: if ranked selection is not better than arbitrary selection
    after training, the ranking was noise and should be dropped.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archon_jax import checkpoint as ck  # noqa: E402

DROP = ("l0_threatw", "l0_psqt_thw")


def pick_columns(params: dict[str, np.ndarray], width: int, how: str) -> np.ndarray:
    """Return `width` accumulator column indices, ascending."""
    acc = params["l0_pstb"].shape[-1]
    if width > acc:
        raise SystemExit(f"width {width} > source accumulator {acc}; this tool narrows only")
    if how == "first":
        return np.arange(width)

    l1w = params["l1w"]                       # (buckets, acc, L2) in some order
    axis = [i for i, d in enumerate(l1w.shape) if d == acc]
    if len(axis) != 1:
        raise SystemExit(f"cannot locate the accumulator axis in l1w{l1w.shape} (acc={acc})")
    a = axis[0]
    head = np.sqrt((l1w.astype(np.float64) ** 2).sum(
        axis=tuple(i for i in range(l1w.ndim) if i != a)))

    pst = params["l0_pstw"].astype(np.float32)
    pax = [i for i, d in enumerate(pst.shape) if d == acc]
    ft = np.sqrt((pst.astype(np.float64) ** 2).sum(
        axis=tuple(i for i in range(pst.ndim) if i != pax[0])))

    score = {"head": head, "ft": ft, "product": head * ft}[how]
    keep = np.sort(np.argsort(-score)[:width])
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--rank", default="head", choices=["head", "ft", "product", "first"])
    ap.add_argument("--spec", default="../tools/nnue/runs/d49_sfblend_500.json",
                    help="run spec of the SOURCE checkpoint (supplies its arch/shapes)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # Shapes come from the SOURCE arch, never guessed: the accumulator axis is
    # axis 1 on l0_pstw (HALFKA, hl) but axis 0 on l1w (hl, buckets*l2), so a
    # positional assumption would silently slice the wrong dimension.
    import json
    from archon_jax.spec import ArchSpec
    src_arch = ArchSpec.from_run_spec(json.load(open(a.spec)))
    shapes = src_arch.param_shapes()
    params = ck.open_raw(a.src, shapes=shapes, mmap=False)
    acc = src_arch.hl
    print(f"source: {os.path.basename(a.src)}  acc={acc}  tensors={len(params)}  spec={os.path.basename(a.spec)}")

    keep = pick_columns(params, a.width, a.rank)
    print(f"ranking='{a.rank}'  keeping {len(keep)}/{acc} columns")

    # Report how much the rankings agree -- if they disagree wildly the choice
    # matters more than this docstring claims and deserves a real experiment.
    if a.rank != "first":
        others = {h: pick_columns(params, a.width, h) for h in ("head", "ft", "product") if h != a.rank}
        for h, o in others.items():
            ov = len(np.intersect1d(keep, o)) / len(keep)
            print(f"  overlap with '{h}': {ov:5.1%}")

    out: dict[str, np.ndarray] = {}
    dropped = sliced = kept = 0
    for name, arr in params.items():
        if name in DROP:
            dropped += arr.size
            continue
        if acc in arr.shape:
            if arr.shape.count(acc) > 1:
                raise SystemExit(f"{name}{arr.shape}: two axes equal acc={acc}; "
                                 "slicing axis is ambiguous, fix this tool explicitly")
            ax = arr.shape.index(acc)
            out[name] = np.take(arr, keep, axis=ax)
            sliced += 1
        else:
            out[name] = arr
            kept += 1

    before = sum(v.size for v in params.values())
    after = sum(v.size for v in out.values())
    print(f"  dropped {dropped:,} params ({dropped/before:.1%})   sliced {sliced} tensors   kept {kept}")
    print(f"  {before:,} -> {after:,} params  ({after/before:.3f}x)")

    if a.dry_run:
        print("  --dry-run: nothing written")
        return
    ck.write_raw(a.dst, out)
    print(f"  wrote {a.dst}  ({os.path.getsize(a.dst)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
