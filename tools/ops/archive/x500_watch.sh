#!/usr/bin/env bash
# Progress watchdog for d49x500. The restart loop is correct but has no
# instant-failure guard: a trainer that dies in 1.5s (as happened at 08:54 when
# ARCHON_JAX_ALLOW_GPU was missing) would relaunch forever while the unit still
# reported "active" -- the healthy-while-dead pattern that bit three times today.
#
# This watches the WORK (the superbatch counter), not the unit's status.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
sb_now() { grep -oE 'components sb [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }
last=$(sb_now); stuck=0
while :; do
  sleep 600                                  # 10 min >> a ~2.4 min superbatch
  systemctl --user is-active --quiet d49x500 2>/dev/null || {
      echo "[x500watch] d49x500 is NOT ACTIVE $(date -Is)" >> "$LOG"; continue; }
  cur=$(sb_now)
  if [ -n "$cur" ] && [ -n "$last" ] && [ "$cur" -le "$last" ]; then
    stuck=$((stuck+1))
    echo "[x500watch] no progress for ${stuck}0 min (sb${cur}) $(date -Is)" >> "$LOG"
    [ "$stuck" -ge 3 ] && {
      echo "[x500watch] *** STALLED 30+ min at sb${cur} -- last 5 journal lines ***" >> "$LOG"
      journalctl --user -u d49x500 --no-pager 2>/dev/null | tail -5 >> "$LOG"; stuck=0; }
  else
    stuck=0
  fi
  last=$cur
done
