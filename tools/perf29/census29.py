#!/usr/bin/env python3
"""perf study 29 -- turn the exact FOLDCOUNT tallies into the LOAD CENSUS.

Two factors, both exact:

  * calls and weight ROWS per site, counted by the instrumented build
    (integers, not samples); and
  * loads per call, from the kernels' static shape, verified by disassembly:
    every kernel walks the accumulator in `ACC / LANES` blocks and issues
    exactly ONE memory read per (block, stream) with no spill and no reload.

The unit is the `L1-dcache-loads` event (`ls_dispatch.ld_dispatch`). ZEN 4
DOUBLE-PUMPS A 512-BIT LOAD: measured 2.0015 events per `vmovdqu64` and 1.0016
per `vmovdqu` (`tools/perf29` calibration, 1e8 issued loads each). So a
64-byte accumulator/i16-row block costs 2 events and a 32-byte i8-row block
costs 1. Getting this wrong halves the whole census.

Usage: python3 census29.py <census-file> [total-loads-per-node]
"""

import re
import sys

ACC = 1280
LANES = 32              # i16 lanes per zmm
BLOCKS = ACC // LANES   # 40 accumulator blocks per call
EV_ZMM = 2              # L1d load events per 512-bit load  (measured)
EV_YMM = 1              # ... per 256-bit load               (measured)

# events per call for the accumulator STREAM read, and per weight ROW.
ACC_EV = BLOCKS * EV_ZMM          # 80  -- 2560 B of accumulator
ROW_I16_EV = BLOCKS * EV_ZMM      # 80  -- 2560 B HalfKA row
ROW_I8_EV = BLOCKS * EV_YMM       # 40  -- 1280 B FullThreats row
ACTIVATE_EV = 4 * BLOCKS * EV_ZMM  # 320 -- four [i16; ACC] arrays


def main():
    txt = open(sys.argv[1]).read()
    tot_loads = float(sys.argv[2]) if len(sys.argv) > 2 else None
    g = dict(re.findall(r"([a-z_]+\.(?:calls|rows))=(\d+)", txt))
    nodes = int(re.search(r"FOLDCOUNT nodes=(\d+)", txt).group(1))
    v = {k: int(n) / nodes for k, n in g.items()}

    rows = [
        # site, calls/node, rows/node, acc-events/call, row-events/row
        ("hka_into   (HalfKA fold)", v["hka_into.calls"], v["hka_into.rows"], ACC_EV, ROW_I16_EV),
        ("hka_dual   (Finny warm)", v["hka_dual.calls"], v["hka_dual.rows"], ACC_EV, ROW_I16_EV),
        ("hka_inplace(Finny cold)", v["hka_inplace.calls"], v["hka_inplace.rows"], ACC_EV, ROW_I16_EV),
        ("thr_into   (threat fold)", v["thr_into.calls"], v["thr_into.rows"], ACC_EV, ROW_I8_EV),
        ("thr_copy   (empty-delta)", v["thr_copy.calls"], 0.0, ACC_EV, 0),
        ("activate   (head merge)", v["activate.calls"], 0.0, ACTIVATE_EV, 0),
    ]
    print(f"nodes={nodes}  ACC={ACC}  blocks/call={BLOCKS}  "
          f"events: acc-read={ACC_EV} i16-row={ROW_I16_EV} i8-row={ROW_I8_EV} activate={ACTIVATE_EV}")
    print(f"{'site':28s}{'calls/n':>10s}{'rows/n':>10s}{'rows/call':>11s}"
          f"{'ACC-STREAM':>12s}{'WEIGHTS':>11s}{'total':>10s}")
    ta = tw = 0.0
    for name, c, r, ae, re_ in rows:
        a, w = c * ae, r * re_
        ta += a
        tw += w
        rc = r / c if c else 0.0
        print(f"{name:28s}{c:10.4f}{r:10.4f}{rc:11.3f}{a:12.2f}{w:11.2f}{a + w:10.2f}")
    print(f"{'TOTAL':28s}{'':10s}{'':10s}{'':11s}{ta:12.2f}{tw:11.2f}{ta + tw:10.2f}")
    print(f"\naccumulator stream = {100 * ta / (ta + tw):.1f}% of the accumulator layer")
    folds_a = ta - v["activate.calls"] * ACTIVATE_EV
    print(f"  ... excluding `activate`: {100 * folds_a / (folds_a + tw):.1f}% "
          f"({folds_a:.1f} of {folds_a + tw:.1f})")
    if tot_loads:
        print(f"\nengine total = {tot_loads:.1f} loads/node")
        print(f"  accumulator stream {ta:8.1f}  = {100 * ta / tot_loads:5.2f}%")
        print(f"  weight rows        {tw:8.1f}  = {100 * tw / tot_loads:5.2f}%")
        print(f"  everything else    {tot_loads - ta - tw:8.1f}  = "
              f"{100 * (tot_loads - ta - tw) / tot_loads:5.2f}%")

    # ---- what FUSION removes -------------------------------------------
    # One i16[ACC] per perspective carrying both channels: the threat fold's
    # own accumulator read/write disappears (its rows fold into the SAME
    # register tile as the HalfKA rows), and `activate` reads two arrays
    # instead of four. Weight-row traffic is untouched -- the same rows are
    # still fetched, just into one pass.
    fused_folds = v["hka_into.calls"] * ACC_EV + v["hka_dual.calls"] * ACC_EV
    now_folds = folds_a
    fused_act = v["activate.calls"] * 2 * BLOCKS * EV_ZMM
    now_act = v["activate.calls"] * ACTIVATE_EV
    print("\n--- fusion, at the SAME width 1280 ---")
    print(f"  fold acc-stream   {now_folds:8.1f} -> {fused_folds:8.1f}   "
          f"saving {now_folds - fused_folds:7.1f}")
    print(f"  activate          {now_act:8.1f} -> {fused_act:8.1f}   "
          f"saving {now_act - fused_act:7.1f}")
    tot_save = (now_folds - fused_folds) + (now_act - fused_act)
    print(f"  TOTAL             {tot_save:8.1f} loads/node removed")
    if tot_loads:
        print(f"                    = {100 * tot_save / tot_loads:.2f}% of all engine loads")


if __name__ == "__main__":
    main()
