#!/usr/bin/env python3
"""perf study 36 — derive the fold-census tables from a `foldcount` dump.

    python3 tools/perf36/census36.py docs/research/perf-artifacts-36/census36-foldcount.txt

Prints per-node call/row rates, the two shape histograms, the weight-row byte
rate, and a reproduction check against perf study 29's three published control
numbers (refresh rate/node 0.16102, refresh rows/node 0.9132, rows/fold 2.275).
Study 32 reproduced all three within 0.9% post-merge; a disagreement larger
than that is a bug to chase, not a finding.
"""

import re
import sys

CONTROLS = {
    "refresh rate/node": 0.16102,
    "refresh rows/node": 0.9132,
    "rows/fold (hka_into)": 2.275,
}


def main() -> None:
    path = sys.argv[1]
    nodes = 0
    width = 0
    sites: dict[str, list[int]] = {}
    hka_shape: dict[str, int] = {}
    thr_shape: dict[int, int] = {}
    for line in open(path):
        if m := re.search(r"nodes=(\d+) acc_width=(\d+)", line):
            nodes, width = int(m.group(1)), int(m.group(2))
        if m := re.search(r"(\w+)\.calls=(\d+) \1\.rows=(\d+)", line):
            sites[m.group(1)] = [int(m.group(2)), int(m.group(3))]
        if m := re.search(r"hka_shape\.(\w+)=(\d+)", line):
            hka_shape[m.group(1)] = int(m.group(2))
        if m := re.search(r"thr_shape\.(\d+)=(\d+)", line):
            thr_shape[int(m.group(1))] = int(m.group(2))

    print(f"nodes={nodes}  acc_width={width}\n")
    print(f"{'site':<14}{'calls/node':>12}{'rows/node':>12}{'rows/call':>12}")
    rows_total = 0
    for k, (calls, rows) in sites.items():
        rows_total += rows
        rc = rows / calls if calls else 0.0
        print(f"{k:<14}{calls / nodes:12.4f}{rows / nodes:12.4f}{rc:12.3f}")
    print(f"\nweight rows/node = {rows_total / nodes:.3f}"
          f"  ->  {rows_total / nodes * width / 1024:.2f} KiB/node")

    got = {
        "refresh rate/node": sites["hka_dual"][0] / nodes,
        "refresh rows/node": sites["hka_dual"][1] / nodes,
        "rows/fold (hka_into)": sites["hka_into"][1] / sites["hka_into"][0],
    }
    print("\nstudy 29 control reproduction")
    for k, want in CONTROLS.items():
        dev = 100.0 * (got[k] - want) / want
        print(f"  {k:<24} published {want:<9} mine {got[k]:.5f}  dev {dev:+.2f}%")

    ht = sum(hka_shape.values())
    print("\nHalfKA into-fold shape (adds, subs)")
    for k in sorted(hka_shape):
        print(f"  {k:<6} {100 * hka_shape[k] / ht:6.2f}%   n={hka_shape[k]}")
    for k in ("1a1s", "1a2s", "2a1s", "2a2s"):
        if k not in hka_shape:
            print(f"  {k:<6}   0.00%   n=0   <- shaped instantiation never used")

    tt = sum(thr_shape.values())
    cum = 0
    print(f"\nthreat fold shape (total rows), calls={tt}")
    for k in sorted(thr_shape):
        cum += thr_shape[k]
        print(f"  {k:2d} rows {100 * thr_shape[k] / tt:6.2f}%   cum {100 * cum / tt:6.2f}%")
    print(f"  mean rows/call = {sum(k * v for k, v in thr_shape.items()) / tt:.3f}")


if __name__ == "__main__":
    main()
