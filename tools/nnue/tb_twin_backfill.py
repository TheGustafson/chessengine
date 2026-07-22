#!/usr/bin/env python3
"""One-shot TensorBoard backfill for the d47c twin (value-only) run.

The live shim tails only the main run's log, and its components regex
requires a pol_ce field the twin never prints — so the twin had no tb
curves. This parses the twin log + the watcher log and emits a clean
run dir. Last occurrence per superbatch wins (supersedes the killed
first launch and the pre-OOM sb161-172 duplicates).

    uv run --no-project --with tensorboardX python tools/nnue/tb_twin_backfill.py \
        [/tmp/d47c_twin.log] [/tmp/d47c_ckpt.log] [data/nnue/tb/d47c/twin]

Re-run guidance: delete the out dir first for a clean single curve.
"""
import re
import sys

from tensorboardX import SummaryWriter

twin_log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d47c_twin.log"
ckpt_log = sys.argv[2] if len(sys.argv) > 2 else "/tmp/d47c_ckpt.log"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "data/nnue/tb/d47c/twin"

COMP = re.compile(r"components sb (\d+) \| value ([\d.]+)")
ELO = re.compile(r"\[twin sb(\d+)\] ELO (-?[\d.]+) \+/- ([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

value_by_sb = {}
with open(twin_log, errors="replace") as fh:
    for line in fh:
        m = COMP.search(ANSI.sub("", line))
        if m:
            value_by_sb[int(m.group(1))] = float(m.group(2))

elo_by_sb = {}
with open(ckpt_log, errors="replace") as fh:
    for line in fh:
        m = ELO.search(line)
        if m:
            elo_by_sb[int(m.group(1))] = float(m.group(2))

writer = SummaryWriter(out_dir)
for sb in sorted(value_by_sb):
    writer.add_scalar("loss/value_component", value_by_sb[sb], sb)
    writer.add_scalar("loss/superbatch", value_by_sb[sb], sb)
for sb in sorted(elo_by_sb):
    writer.add_scalar("elo/fixed_node_vs_incumbent", elo_by_sb[sb], sb)
writer.flush()
print(f"backfilled {len(value_by_sb)} value points + {len(elo_by_sb)} elo points -> {out_dir}")
