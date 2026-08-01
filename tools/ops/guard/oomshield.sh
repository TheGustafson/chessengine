#!/usr/bin/env bash
# Keeps the D49 trainer the LAST thing the kernel OOM killer picks.
#
# Why this exists (2026-07-31 22:53): the kernel OOM killer killed d49y.service
# and the watchdog had to restart it, losing 3 superbatches. MemoryLow=4G did NOT
# prevent it -- MemoryLow guards against RECLAIM, and has no bearing on OOM-killer
# victim selection, which scores by RSS x oom_score_adj.
#
# Root cause: everything launched from this harness inherits oom_score_adj=200
# (the claude process carries 200), so the trainer is born MORE attractive to the
# OOM killer than a default process. Lowering it needs CAP_SYS_RESOURCE, which we
# do not have. Raising other processes is always permitted -- so we raise every
# disposable job to 900, which achieves the same ordering from the other side.
set -uo pipefail
SELF=$$
while :; do
  for pat in 'fastchess' 'target-sota/release/chessengine' 'engines/bin/archon' \
             '\.claude/worktrees/.*/target.*/chessengine' 'stockfish'; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$p" = "$SELF" ] && continue
      tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q shell-snapshots && continue
      cur=$(cat "/proc/$p/oom_score_adj" 2>/dev/null) || continue
      [ "${cur:-0}" -lt 900 ] && echo 900 > "/proc/$p/oom_score_adj" 2>/dev/null
    done
  done
  sleep 20
done
