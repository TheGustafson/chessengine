#!/usr/bin/env python3
"""Paired wall-clock speed panel for ARCHON perf work.

WHY THIS EXISTS
---------------
`instructions:u` is our gate because it has a measured null of 0.000 -- but it
is NOT the deliverable. Measured end-to-end on studies 52-56 (2026-07-31):

    non-PGO instructions   -4.163%   <- what studies report
    PGO instructions       -2.686%   <- native+PGO absorbs 38%
    PGO wall-clock         -1.020%   <- instructions overstate time 2.63x
    DERIVED Elo            +1.77     <- 120 Elo/doubling (exp-0154)

A -4.163% headline is worth under two Elo. This tool measures the wall-clock
step so a study can quote its real value instead of its headline.

THE DESIGN, AND WHY EACH PIECE IS THERE
---------------------------------------
1. NULL AND ARM ARE INTERLEAVED WITHIN EACH ROUND. Every round times all three
   binaries back to back, so the null and the arm see the SAME contention. An
   earlier null-then-arm design had its null drift -0.132% -> +0.972% between
   phases as an unrelated build started; interleaved, the null landed at
   p = 1.0000 (20/40 negative) on a box that was NOT idle.
2. BLOCKED ROTATION of which binary runs first, so position in the round cannot
   alias onto a binary. Not free shuf (unbalances position), not plain rotation
   (freezes adjacency).
3. 40 ROUNDS BY DEFAULT. cycles/wall sigma on this box is ~1.5-5% per run. Perf
   study 57 measured a ~1% contrast reading -0.790% and FAILING its sign test at
   24 rounds (17/24, p=0.0639), then -1.228% at p=0.0034 at 39. Twenty-three
   rounds resolve the MEDIAN; the SIGN TEST needs ~40.
4. MEDIAN-ONLY LICENCE, PRE-REGISTERED HERE IN THE TOOL. Across four panels the
   median null stayed in [-0.132%, +0.972%] while the mean null reached -2.068%
   with sd 6.3-9.5, wrecked by single contention rounds. Deciding the statistic
   after seeing the data is the failure this file exists to prevent.

A LOADED BOX SILENTLY RUINS THIS PANEL -- SO IT IS CHECKED
---------------------------------------------------------
cycles and wall-clock are NOT load-robust. Measured 2026-07-31: the same binary
read 17.95 B cycles on a quiet box and 26.92 B (+49.9%) while a match and a build
ran; instructions:u over the same pair drifted 0.08%. Panels run under load have
produced a byte-identical null that moved FURTHER than the arm.

This script now samples system CPU busy-fraction before the first round and after
the last, and prints both. If either exceeds QUIET_THRESHOLD it says so loudly --
a panel that was not quiet end-to-end should be VOIDED AND RE-RUN, not salvaged.
Interleaving the null within each round (below) makes the panel robust to steady
load, but not to load that ARRIVES or LEAVES mid-run.

STARTUP IS NOT A GAME WIN -- SUBTRACT IT
----------------------------------------
The 5M-node bench frame charges process startup ONCE PER RUN. An arm that cuts
load-time work (net load, quantise_halfka) therefore reads as a wall-clock win
that a real game -- where startup is amortised over minutes of search -- never
sees. Measured 2026-07-31 on the studies 52-58 pair: the 1-node delta was
-261.4 M cycles against -526.0 M for the whole frame, i.e. HALF the apparent gain
was startup. Perf study 59's own arms were 94% startup and it retracted its Elo.

Pass --startup-frame to run the same panel at 1 node; subtract that result from
the full-frame result and quote the residual as the game-relevant number.

WHAT LICENSES THE ELO CONVERSION
--------------------------------
Only run --elo when the two binaries are behaviourally IDENTICAL, i.e. the perf
work was gated node-identical. Verify with a fixed-node match: ours returned
Ptnml [0, 0, 750, 0, 0] over 1500 games -- every pair split exactly 1-1 -- so the
speed ratio is the ENTIRE difference between the binaries. Without that, a
speed->Elo constant is measuring strength changes too, and the number is void.

Report the result as DERIVED, never as measured.

USAGE
-----
    python3 tools/wallclock_panel.py --base <bin> --arm <bin> [--rounds 40] [--elo]

Both binaries should be built the same way (both PGO or both not; same
target-cpu). build_pgo.sh honours PGO_CPU if you need to hold target-cpu fixed --
mixing x86-64-v3 with native silently conflates two changes.
"""
import argparse, os, shutil, statistics as st, subprocess, sys, tempfile, time
from math import comb, log2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NET  = os.path.join(ROOT, "engines/canonical/net.cenn")
DEFAULT_FENS = os.path.join(ROOT, "tools/nnue/calib/wl50.fens")

QUIET_THRESHOLD = 0.15  # fraction of total CPU busy with anything other than us


def cpu_busy(window=1.0):
    """System-wide non-idle CPU fraction over `window` seconds.

    Sampled when this script is NOT benching, so anything it sees is somebody
    else's work. /proc/stat field 4 is idle, field 5 is iowait.
    """
    def snap():
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        return sum(v), v[3] + v[4]
    t0, i0 = snap()
    time.sleep(window)
    t1, i1 = snap()
    dt, di = t1 - t0, i1 - i0
    return 0.0 if dt <= 0 else max(0.0, 1.0 - di / dt)


def sign_p(k, n):
    """Two-sided sign test."""
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="reference binary")
    ap.add_argument("--arm",  required=True, help="candidate binary")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--nodes", default="100000")
    ap.add_argument("--net",  default=DEFAULT_NET)
    ap.add_argument("--fens", default=DEFAULT_FENS)
    ap.add_argument("--startup-frame", action="store_true",
                    help="run at 1 node instead of --nodes, to measure the fixed "
                         "startup component; subtract from the full-frame result")
    ap.add_argument("--elo", action="store_true",
                    help="convert to DERIVED Elo; only valid if the binaries are "
                         "node-identical (see module docstring)")
    a = ap.parse_args()

    net = os.path.realpath(a.net)
    nodes = "1" if a.startup_frame else a.nodes
    cmd = ["bench", "16", "1", nodes, a.fens, "nodes", net]
    if a.startup_frame:
        print("STARTUP FRAME (1 node): this measures fixed per-process cost only.")
        print("Subtract it from the full-frame result; games never see it.\n")

    # The null arm is a byte-identical copy of base: any reading it produces is
    # pure instrument noise, by construction.
    tmp = tempfile.mkdtemp(prefix="wcpanel-")
    null = os.path.join(tmp, "null_bin")
    shutil.copy2(a.base, null)

    def t(b):
        s = time.perf_counter()
        subprocess.run([b] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return time.perf_counter() - s

    order = [(a.base, null, a.arm), (null, a.arm, a.base), (a.arm, a.base, null)]
    nulls, arms = [], []
    busy_before = cpu_busy()
    try:
        for r in range(a.rounds):
            res = {b: t(b) for b in order[r % 3]}
            nulls.append(100 * (res[null]   - res[a.base]) / res[a.base])
            arms .append(100 * (res[a.arm]  - res[a.base]) / res[a.base])
            if (r + 1) % 10 == 0:
                print(f"  ...{r + 1}/{a.rounds}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    busy_after = cpu_busy()
    print(f"\nbox CPU busy (excluding this panel): before {100*busy_before:.1f}%, "
          f"after {100*busy_after:.1f}%")
    if max(busy_before, busy_after) > QUIET_THRESHOLD:
        print(f"*** NOT QUIET: exceeds {100*QUIET_THRESHOLD:.0f}%. This panel should be "
              f"VOIDED and re-run on an idle box, not salvaged. ***")

    def rep(v, tag):
        m, sd = st.median(v), st.stdev(v)
        neg = sum(1 for x in v if x < 0)
        print(f"{tag}: median {m:+.3f}%  sd {sd:.3f}  "
              f"sign {neg}/{len(v)} neg  p={sign_p(max(neg, len(v)-neg), len(v)):.4f}")
        return m

    print()
    nm = rep(nulls, "NULL (byte-identical)")
    am = rep(arms,  "ARM                  ")
    print()

    # Pre-registered licence: median-only, null must be under half the arm.
    if abs(nm) >= abs(am) / 2:
        print(f"NOT LICENSED: null median {nm:+.3f}% vs arm {am:+.3f}%.")
        print("The instrument cannot resolve this arm under current load. "
              "Re-run on a quieter box or raise --rounds; do NOT quote the arm.")
        return 1

    print(f"LICENSED (pre-registered median-only: |{nm:.3f}| < |{am:.3f}|/2)")
    print(f"  wall-clock : {am:+.3f}%")
    if a.elo:
        speed = 1 / (1 + am / 100)
        print(f"  speed      : {speed:.4f}x")
        print(f"  DERIVED Elo: {120 * log2(speed):+.2f}  (120/doubling, exp-0154)")
        print("  ^ DERIVED, not measured. Valid only if the binaries are "
              "node-identical -- confirm with a fixed-node match.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
