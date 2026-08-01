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

A BEFORE/AFTER CHECK CANNOT SEE A JOB THAT STARTS AND ENDS BETWEEN THEM. Perf
study 63's first panel ran 18:59-20:11 with a before/after process census that
saw only the GPU trainer, while a concurrent session on the same box ran a
400-game fixed-node match and a wall-clock smoke test of its own inside that
window (`d990a1bd`, `4fecb7c1`). Sample DURING the panel when you can.

TWO DEFECTS THE NULL CAUGHT, FIXED IN PERF STUDY 63
---------------------------------------------------
A 400-round PGO panel had the byte-identical null read +0.646% at p < 1e-4, in
every rotation phase and every quarter, with the three position medians inside
0.25% of each other -- so not position, not drift. Two asymmetries, both in this
file, both now fixed and both documented at their code:

1. The null copy went to /tmp, which is TMPFS on this box while the binaries
   under test are on the NVMe. The copy now lands beside the base binary.
2. Runs were NOT pinned, while every counter panel in the 45-62 chain pins to
   one physical core. Unpinned per-round sd was 3.58% against 0.98-1.13% for
   study 62's panels. `--cpu` now defaults to core 2; `--cpu -1` restores the
   old behaviour.

The lesson is the ordinary one: the null is not a formality, it is the only
part of the harness that can fail loudly, and when it fails the answer is to
FIX THE INSTRUMENT, not to re-run hoping for a quieter box.

RAW ROUNDS ARE THE ARTIFACT (--dump, perf study 63)
---------------------------------------------------
Until study 63 a panel left behind five printed lines, so nothing downstream
could recover the round-to-round sd, re-check the median, or size the NEXT
panel. That matters because the round count a panel needs scales as the INVERSE
SQUARE of the contrast: the 40-round default is calibrated for the ~1% arms of
studies 52-58, and study 63's arm was -0.32% of instructions, needing 400. The
only way to pre-register a round count honestly is to have the previous panel's
sd on disk. `--dump` writes one TSV row per round (absolute seconds for base,
null and arm) beside the summary. Park it under docs/research/perf-artifacts-NN/,
never on /tmp -- that is tmpfs.

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
    ap.add_argument("--cpu", type=int, default=2,
                    help="physical core to pin every run to (default 2, the "
                         "core every counter panel in this series uses); pass "
                         "-1 to leave runs unpinned as before study 63")
    ap.add_argument("--dump", default=None,
                    help="write the per-round raw timings to this TSV so the "
                         "panel is auditable and its sd is recoverable; see "
                         "RAW ROUNDS ARE THE ARTIFACT in the module docstring")
    a = ap.parse_args()

    if a.cpu is not None and a.cpu < 0:
        a.cpu = None
    net = os.path.realpath(a.net)
    nodes = "1" if a.startup_frame else a.nodes
    cmd = ["bench", "16", "1", nodes, a.fens, "nodes", net]
    if a.startup_frame:
        print("STARTUP FRAME (1 node): this measures fixed per-process cost only.")
        print("Subtract it from the full-frame result; games never see it.\n")

    # The null arm is a byte-identical copy of base: any reading it produces is
    # pure instrument noise, by construction.
    #
    # PERF STUDY 63 -- THE COPY MUST LIVE ON THE SAME FILESYSTEM AS THE BASE.
    # `tempfile.mkdtemp()` lands in $TMPDIR, i.e. /tmp, which on this box is
    # TMPFS while the binaries under test sit on the NVMe. Executing an ELF out
    # of tmpfs is not the same as executing it out of the ext4 page cache (this
    # /tmp is mounted `huge=advise`, so the two differ in whether the text
    # mapping can be backed by huge pages, and tmpfs pages are swap-backed).
    # Measured 2026-07-31, 400 PGO rounds: the byte-identical null read
    # **+0.646%, 125/400 negative, p < 1e-4** -- the harness resolved a
    # difference between two identical binaries more confidently than it
    # resolved the arm, in EVERY rotation phase (+0.407 / +0.661 / +0.864) and
    # EVERY quarter of the run, with the three position medians within 0.25% of
    # each other. It is an asymmetry between the null and the two real arms,
    # built into the tool, and it had been there since the tool was written.
    tmp = tempfile.mkdtemp(prefix=".wcpanel-",
                           dir=os.path.dirname(os.path.abspath(a.base)))
    null = os.path.join(tmp, "null_bin")
    shutil.copy2(a.base, null)

    # PERF STUDY 63 -- PIN, exactly as every counter panel in the 45-62 chain
    # does (`panel61.sh`'s `taskset -c "$CPU"`). Unpinned, a run migrates across
    # a box carrying a GPU trainer at 1.44 CPUs and a browser, and the 400-round
    # panel above measured a per-round sd of **3.58%** -- against 0.98-1.13% for
    # study 62's panels -- which is 3.6x more noise than the arms this tool
    # exists to resolve. The counter panels have pinned since study 45; the
    # wall-clock arbiter never did.
    pin = ["taskset", "-c", str(a.cpu)] if a.cpu is not None else []

    def t(b):
        s = time.perf_counter()
        subprocess.run(pin + [b] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return time.perf_counter() - s


    # PERF STUDY 60's TRAP, IN THIS TOOL TOO (found 2026-07-31 after it voided
    # two published Elo figures). `engines/canonical/net.cenn` is a symlink into
    # gitignored `data/`, so it DANGLES in every fresh git worktree -- and this
    # script's default --net resolves relative to its own location, i.e. to the
    # worktree it is running from. With a dangling net the engine prints
    # "reading NNUE file: No such file" and EXITS HAVING SEARCHED ZERO NODES, in
    # ~0.02 s. The panel then happily reports a median over three near-instant
    # exits. Refuse instead: the net must be readable AND a probe run must
    # actually search the frame.
    if not os.path.isfile(net):
        sys.exit(f"REFUSING: --net '{net}' is missing or a dangling symlink.\n"
                 f"Pass a resolved absolute path (readlink -f from the main "
                 f"checkout); a fresh worktree's copy dangles.")
    probe = subprocess.run([a.base] + cmd, capture_output=True, text=True)
    if "Nodes searched" not in probe.stdout:
        sys.exit(f"REFUSING: probe run of '{a.base}' searched no nodes.\n"
                 f"{probe.stdout.strip()[-300:]}\n{probe.stderr.strip()[-300:]}")

    order = [(a.base, null, a.arm), (null, a.arm, a.base), (a.arm, a.base, null)]
    nulls, arms, raw = [], [], []
    busy_before = cpu_busy()
    try:
        for r in range(a.rounds):
            res = {b: t(b) for b in order[r % 3]}
            nulls.append(100 * (res[null]   - res[a.base]) / res[a.base])
            arms .append(100 * (res[a.arm]  - res[a.base]) / res[a.base])
            raw.append((r, res[a.base], res[null], res[a.arm]))
            if (r + 1) % 10 == 0:
                print(f"  ...{r + 1}/{a.rounds}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if a.dump:
            with open(a.dump, "w") as fh:
                fh.write("# base=%s arm=%s nodes=%s rounds=%d\n"
                         % (a.base, a.arm, nodes, a.rounds))
                fh.write("round\tt_base\tt_null\tt_arm\n")
                for r, tb, tn, ta in raw:
                    fh.write(f"{r}\t{tb:.6f}\t{tn:.6f}\t{ta:.6f}\n")
            print(f"raw rounds -> {a.dump}")

    # PERF STUDY 63 -- a before/after census CANNOT see a job that starts AND
    # ends between the two checks; one did exactly that inside study 63's first
    # panel. The base binary's own per-round time is sampled throughout, so a
    # regime shift mid-run shows up as a half-vs-half difference.
    base_t = [x[1] for x in raw]          # study 63 already records per-round timings
    if len(base_t) >= 8:
        h = len(base_t) // 2
        m1, m2 = st.median(base_t[:h]), st.median(base_t[h:])
        drift = 100 * (m2 - m1) / m1
        print(f"\nmid-run stability (base binary): first half {m1:.3f}s, "
              f"second half {m2:.3f}s, drift {drift:+.2f}%")
        if abs(drift) > 2.0:
            print("*** REGIME SHIFT mid-run: load arrived or left between the "
                  "before/after checks. Treat this panel as VOID. ***")

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
