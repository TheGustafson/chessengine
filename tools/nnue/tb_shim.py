#!/usr/bin/env python3
"""Tail a bullet trainer log into TensorBoard scalars, live.

    uv run --no-project --with tensorboardX python tools/nnue/tb_shim.py \
        /tmp/d47_train.log data/nnue/tb/d47-v1/train

Emits:
    loss/superbatch      running loss per superbatch (the headline curve)
    loss/batch           fine-grained loss (per ~128-batch print)
    throughput/pos_per_sec
    time/superbatch_sec  wall seconds per superbatch (contention shows here)
    lr/value             the decayed learning rate (parsed from LR drops)
Follows across truncation/recreation (relaunches), strips ANSI codes.
"""
import os
import re
import sys
import time

from tensorboardX import SummaryWriter

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
SB = re.compile(
    r"superbatch (\d+) \| time ([\d.]+)s \| running loss ([\d.]+|NaN) \| (\d+) pos/sec"
)
BATCH = re.compile(r"sb (\d+) batch (\d+): loss ([\d.]+|NaN)")
LR = re.compile(r"LR dropped to ([\d.eE+-]+)")
# pol_ce is OPTIONAL: value-only runs (twins, pure-value recipes) print
# "components sb N | value X" with no policy field — they must still chart.
COMP = re.compile(r"components sb (\d+) \| value ([\d.]+)(?: \| pol_ce ([\d.]+))?")
LAMBDA = re.compile(r"pol_lambda=([\d.eE+-]+)")
# stage-3 (JAX) extras: the heldout instrument + sparsity telemetry
HELDOUT = re.compile(r"heldout sb (\d+) \| value ([\d.]+)(?: \| ema ([\d.]+))?")
FT = re.compile(r"ft_l1 ([\d.]+) \| ft_density ([\d.]+)")
BPS = 3040  # batches per superbatch (D40_BPS of the current run)


def follow(path):
    fh, ino = None, None
    while True:
        try:
            st = os.stat(path)
            if fh is None or st.st_ino != ino or st.st_size < fh.tell():
                if fh:
                    fh.close()
                fh = open(path, errors="replace")
                ino = st.st_ino
        except FileNotFoundError:
            time.sleep(2)
            continue
        line = fh.readline()
        if line:
            yield line
        else:
            time.sleep(2)


def main():
    log_path, tb_dir = sys.argv[1], sys.argv[2]
    writer = SummaryWriter(tb_dir)
    last_sb, n_b = 0, 0
    pol_lambda = 0.0
    for raw in follow(log_path):
        line = ANSI.sub("", raw)
        m = SB.search(line)
        if m:
            sb, secs, loss, pps = (
                int(m.group(1)),
                float(m.group(2)),
                m.group(3),
                int(m.group(4)),
            )
            last_sb = sb
            if loss != "NaN":
                writer.add_scalar("loss/superbatch", float(loss), sb)
            writer.add_scalar("throughput/pos_per_sec", pps, sb)
            writer.add_scalar("time/superbatch_sec", secs, sb)
            writer.flush()
            continue
        m = BATCH.search(line)
        if m:
            sb, b, loss = int(m.group(1)), int(m.group(2)), m.group(3)
            if loss != "NaN":
                writer.add_scalar("loss/batch", float(loss), (sb - 1) * BPS + b)
            n_b += 1
            if n_b % 50 == 0:
                writer.flush()
            continue
        m = LR.search(line)
        if m:
            writer.add_scalar("lr/value", float(m.group(1)), last_sb)
            writer.flush()
            continue
        m = LAMBDA.search(line)
        if m:
            pol_lambda = float(m.group(1))
            continue
        m = COMP.search(line)
        if m:
            sb, v = int(m.group(1)), float(m.group(2))
            writer.add_scalar("loss/value_component", v, sb)
            if m.group(3) is not None:
                ce = float(m.group(3))
                writer.add_scalar("loss/policy_ce", ce, sb)
                if v > 0 and pol_lambda > 0:
                    writer.add_scalar("loss/policy_share_of_value", pol_lambda * ce / v, sb)
            mft = FT.search(line)
            if mft:
                writer.add_scalar("ft/l1", float(mft.group(1)), sb)
                writer.add_scalar("ft/density", float(mft.group(2)), sb)
            writer.flush()
            continue
        m = HELDOUT.search(line)
        if m:
            writer.add_scalar("loss/heldout", float(m.group(2)), int(m.group(1)))
            if m.group(3) is not None:
                writer.add_scalar("loss/heldout_ema", float(m.group(3)), int(m.group(1)))
            writer.flush()


if __name__ == "__main__":
    main()
