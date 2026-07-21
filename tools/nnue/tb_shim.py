#!/usr/bin/env python3
"""Tail a bullet trainer log into TensorBoard scalars, live.

    uv run --no-project --with tensorboardX python tools/nnue/tb_shim.py \
        /tmp/d47_train.log data/nnue/tb/d47-v1

Parses the per-superbatch summary lines:
    superbatch N | time Ts | running loss L | P pos/sec | total time ...
and the per-batch progress prints (finer loss trace, logged per 128 batches):
    sb N batch B: loss L
Follows across truncation/recreation (relaunches), strips ANSI codes.
"""
import os
import re
import sys
import time

from tensorboardX import SummaryWriter

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
SB = re.compile(r"superbatch (\d+) \| time [\d.]+s \| running loss ([\d.]+|NaN) \| (\d+) pos/sec")
BATCH = re.compile(r"sb (\d+) batch (\d+): loss ([\d.]+|NaN)")
BPS = 1520  # batches per superbatch (D40_BPS)


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
    n_sb = n_b = 0
    for raw in follow(log_path):
        line = ANSI.sub("", raw)
        m = SB.search(line)
        if m:
            sb, loss, pps = int(m.group(1)), m.group(2), int(m.group(3))
            if loss != "NaN":
                writer.add_scalar("loss/superbatch", float(loss), sb)
            writer.add_scalar("throughput/pos_per_sec", pps, sb)
            writer.flush()
            n_sb += 1
            continue
        m = BATCH.search(line)
        if m:
            sb, b, loss = int(m.group(1)), int(m.group(2)), m.group(3)
            if loss != "NaN":
                writer.add_scalar("loss/batch", float(loss), (sb - 1) * BPS + b)
            n_b += 1
            if n_b % 50 == 0:
                writer.flush()


if __name__ == "__main__":
    main()
