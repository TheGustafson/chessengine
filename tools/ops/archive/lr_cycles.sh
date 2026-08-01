#!/usr/bin/env bash
# LR JUMP -> ANNEAL experiment (owner, 2026-08-01).
#   Stop D49 at the next export, SAVE that checkpoint as the anchor, then run
#   three cycles: warm-start at 5x the base LR, decay linearly to 0 over 30 sb,
#   checkpoint, repeat.
#
#   anchor  sb1540  (base LR there 3.5837e-6; 5x = 1.7919e-5)
#   cycle 1 sb1540 -> sb1570
#   cycle 2 sb1570 -> sb1600
#   cycle 3 sb1600 -> sb1630
#
# Each cycle uses the SAME peak (1.7919e-5). "5x the LR" cannot mean 5x the
# ENDING lr for cycles 2-3 because that is 0, so the treatment is repeated
# identically -- which is also what isolates the cycling effect.
set -uo pipefail
R=/home/nick/projects/chessengine
D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
SHARE=/home/nick/.local/share/archon-d49
cd "$R"

newest() { ls -t "$D"/*-ema-raw.bin 2>/dev/null | head -1 | grep -oE 'sb[0-9]+' | tr -d 'sb'; }

# ---- 1. wait for the sb1540 export -----------------------------------------
ANCHOR_DONE=1
echo "[lrc] waiting for the sb1540 export $(date -Is)" >> "$LOG"
for _ in $(seq 1 240); do
  n=$(newest); [ -n "$n" ] && [ "$n" -ge 1540 ] && break
  sleep 30
done
n=$(newest)
echo "[lrc] export sb${n} present; stopping the base run $(date -Is)" >> "$LOG"

# ---- 2. stop the trainer and its watchdog ----------------------------------
systemctl --user stop d49wd 2>/dev/null      # stop the watchdog FIRST or it restarts d49y
sleep 3
systemctl --user stop d49y 2>/dev/null
sleep 10

# ---- 3. PROTECT the anchor from the reaper ---------------------------------
# ckpt_reaper globs d49-1280-sb*-raw.bin and keeps only multiples of 100 plus
# the latest 4, so sb1540 WOULD be deleted. A different prefix is invisible to it.
for suf in ema-raw ema-quant raw quant; do
  s="$D/d49-1280-sb${n}-${suf}.bin"
  [ -f "$s" ] && cp -n "$s" "$D/d49anchor-sb${n}-${suf}.bin" 2>/dev/null
done
echo "[lrc] anchor saved: d49anchor-sb${n}-*.bin ($(ls -1 "$D"/d49anchor-sb${n}-*.bin 2>/dev/null | wc -l) files)" >> "$LOG"

# ---- 4. three jump->anneal cycles ------------------------------------------
starts=(1540 1570 1600); ends=(1570 1600 1630)
for i in 0 1 2; do
  c=$((i+1)); a=${starts[$i]}; b=${ends[$i]}
  echo "[lrc] cycle ${c}: sb${a} -> sb${b}, lr 1.7919e-5 -> 0 $(date -Is)" >> "$LOG"
  systemd-run --user --unit="d49lrc${c}" --working-directory=/home/nick/projects/chessengine/jax \
    --property=MemoryMax=9G --property=MemorySwapMax=0 \
    /usr/bin/env PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
      uv run --no-project --with numpy==2.5.1 \
      --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
      python tools/pump_train.py "../tools/nnue/runs/d49_lrcycle${c}_1280.json" "$a" \
    >/dev/null 2>&1
  # wait for it to reach the cycle end
  sleep 45   # let it actually start before judging it
  if ! systemctl --user is-active --quiet "d49lrc${c}" 2>/dev/null; then
    echo "[lrc] cycle ${c} FAILED TO START -- aborting the whole run" >> "$LOG"
    journalctl --user -u "d49lrc${c}" --no-pager 2>/dev/null | tail -5 >> "$LOG"
    exit 1
  fi
  for _ in $(seq 1 480); do
    if ! systemctl --user is-active --quiet "d49lrc${c}" 2>/dev/null; then
      m=$(newest)
      if [ -z "$m" ] || [ "$m" -lt "$b" ]; then
        echo "[lrc] cycle ${c} DIED at sb${m:-?} before reaching sb${b} -- aborting" >> "$LOG"
        exit 1
      fi
      break
    fi
    m=$(newest); [ -n "$m" ] && [ "$m" -ge "$b" ] && break
    sleep 30
  done
  systemctl --user stop "d49lrc${c}" 2>/dev/null; sleep 8
  m=$(newest)
  for suf in ema-raw ema-quant raw quant; do
    s="$D/d49-1280-sb${m}-${suf}.bin"
    [ -f "$s" ] && cp -n "$s" "$D/d49lrc${c}-sb${m}-${suf}.bin" 2>/dev/null
  done
  echo "[lrc] cycle ${c} done at sb${m}; checkpoint saved d49lrc${c}-sb${m}-*.bin $(date -Is)" >> "$LOG"
done
echo "[lrc] ALL CYCLES COMPLETE $(date -Is)" >> "$LOG"
