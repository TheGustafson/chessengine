#!/usr/bin/env python3
"""Training dashboard shim: bullet stdout -> TensorBoard tfevents (design/11 §11.1).

bullet is a Rust trainer with no native dashboard: it prints one line per
superbatch to stdout. This ~read-a-line/write-a-scalar shim tees that stream
(so the user still sees training live) and mirrors the numbers into tfevents so
`tensorboard --logdir tools/nnue/runs` shows live curves.

Parsed line (bullet crates/trainer/src/run/logger.rs), ANSI-colored:
    superbatch N | time Ts | running loss L | P pos/sec | total time TTs
plus:      LR dropped to X            (StepLR / any drop)
Preamble:  LR Scheduler : start X ... (initial LR)

NOTE (bullet limitation, not ours): bullet does NOT emit a validation loss —
`test_set` support prints "Validation data not currently implemented!"
(crates/bullet_lib/src/value.rs). So the live curve is TRAIN loss + LR +
throughput only. The real "am I getting stronger" signal for a chess NNUE is
net-vs-net Elo, tracked separately in net_registry.csv (SPRT column, step 4).

Usage (pipe the trainer's stdout through it):
    <trainer> 2>&1 | uv run python tools/nnue/dashboard.py \
        --run net-0 --logdir tools/nnue/runs

Tooling layer only (design/16 §3.5.1; the engine crate stays zero-dep, D15).
"""

import argparse
import re
import sys

import tensorflow as tf

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# "superbatch 12 | time 1.3s | running loss 0.041234 | 4200000 pos/sec | ..."
SB = re.compile(
    r"superbatch\s+(\d+)\s*\|\s*time\s*([\d.]+)s\s*\|\s*running loss\s*([\d.eE+-]+)\s*"
    r"\|\s*(\d+)\s*pos/sec"
)
LR_DROP = re.compile(r"LR dropped to\s*([\d.eE+-]+)")
# Preamble, e.g. "LR Scheduler : start at 0.001 and cosine decay to ..." (or
# StepLR's "start 0.001 gamma ..."); grab the first LR after "start".
LR_START = re.compile(r"LR Scheduler\s*:\s*start(?:\s+at)?\s*([\d.eE+-]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name (subdir under logdir)")
    ap.add_argument("--logdir", default="tools/nnue/runs", help="tfevents root")
    args = ap.parse_args()

    logdir = f"{args.logdir}/{args.run}"
    writer = tf.summary.create_file_writer(logdir)
    print(f"[dashboard] logging tfevents to {logdir}", file=sys.stderr)

    cur_lr = None
    last_sb = 0
    with writer.as_default():
        for raw in sys.stdin:
            line = ANSI.sub("", raw)
            sys.stdout.write(raw)  # tee the live stream through unchanged
            sys.stdout.flush()

            m = LR_START.search(line)
            if m and cur_lr is None:
                cur_lr = float(m.group(1))
            if m := LR_DROP.search(line):
                cur_lr = float(m.group(1))

            if m := SB.search(line):
                sb = int(m.group(1))
                loss = float(m.group(3))
                pps = int(m.group(4))
                tf.summary.scalar("train/loss", loss, step=sb)
                tf.summary.scalar("train/pos_per_sec", pps, step=sb)
                if cur_lr is not None:
                    tf.summary.scalar("train/lr", cur_lr, step=sb)
                tf.summary.flush()
                last_sb = sb

    print(f"[dashboard] done — {last_sb} superbatches logged to {logdir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
