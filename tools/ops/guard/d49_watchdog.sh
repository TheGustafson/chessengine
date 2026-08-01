#!/bin/bash
# Restarts the D49 cyclical continuation if it dies before sb1700.
# v2: PROGRESS-GATED. A restart is only allowed if the run has advanced since the
# last one. Without that, a persistent failure (a bad resume, a missing
# checkpoint, an OOM that recurs) would be relaunched every 100 s for 55 hours --
# which is exactly what the sb700 resume bug would have produced had it not been
# caught. Three restarts with no progress = stop and log loudly.
LOG=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
S=/home/nick/.local/share/archon-d49
sb_now() { grep -oE 'components sb [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }
while systemctl --user is-active d49x.service >/dev/null 2>&1; do sleep 60; done
stall=0; last_seen=-1
for i in $(seq 1 2000); do
  sleep 100
  cur=$(sb_now); cur=${cur:-0}
  [ "$cur" -ge 1699 ] && { echo "[wd] reached sb$cur -- COMPLETE $(date -Is)" >> "$LOG"; exit 0; }
  systemctl --user is-active d49y.service >/dev/null 2>&1 && { last_seen=$cur; stall=0; continue; }
  if [ "$cur" -le "$last_seen" ]; then
    stall=$((stall+1))
    if [ "$stall" -ge 3 ]; then
      echo "[wd] ***HALTED*** 3 restarts with NO progress past sb$cur. Not restarting again." >> "$LOG"
      echo "[wd] Investigate before relaunching: likely a resume/checkpoint/config fault, not transient." >> "$LOG"
      exit 1
    fi
  else
    stall=0
  fi
  last_seen=$cur
  echo "[wd] d49y inactive at sb$cur (stall=$stall) -- RESTARTING $(date -Is)" >> "$LOG"
  systemctl --user reset-failed d49y.service 2>/dev/null
  systemd-run --user --unit=d49y --quiet bash "$S/d49y_launch.sh"
  sleep 120
done
