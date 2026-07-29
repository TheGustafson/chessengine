#!/bin/sh
# Run a heavy job (build, match, sweep) so the kernel's OOM killer picks IT
# rather than a long-running trainer.
#
# WHY: every process in this session's tree inherits oom_score_adj=200, and the
# trainer is the largest RSS among them -- so the global OOM killer chooses the
# TRAINER every time. Three training runs died that way (2026-07-29).
#
# An unprivileged process may RAISE its own oom_score_adj but never lower it,
# so we cannot protect the trainer; we make everything else MORE killable.
# Children inherit the value at fork, so cargo/rustc/fastchess all get it.
#
# `systemd-run -p OOMScoreAdjust=` does NOT work for --user --scope: verified
# 2026-07-29, the build still read 200.
#
#   tools/oomlast.sh cargo build --release ...
#   systemd-run --user --scope -p MemoryMax=5G tools/oomlast.sh <cmd>
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || echo "oomlast: WARN could not raise oom_score_adj" >&2
exec "$@"
