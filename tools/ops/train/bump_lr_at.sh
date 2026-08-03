#!/usr/bin/env bash
# tools/ops/train/bump_lr_at.sh — at superbatch N, stop the trainer, scale the
# remaining LR ramp, and resume.
#
# USAGE
#   AT_SB=700 FACTOR=1.25 UNIT=d50wtrain \
#     bash tools/ops/train/bump_lr_at.sh
#
#   Environment:
#     AT_SB    superbatch to act on                (REQUIRED)
#     FACTOR   LR multiplier for the tail          (default 1.25)
#     UNIT     systemd --user unit of the trainer  (default d50wtrain)
#     SPEC     run spec to edit    (default tools/nnue/runs/d50_w768.json)
#     EXP_DIR  checkpoint dir      (default read from the spec's `out`)
#     PREFIX   checkpoint prefix   (default read from the spec's `net_id`)
#
# WHY THIS IS A SCRIPT AND NOT THREE COMMANDS
#   Three things here are timing- or integrity-sensitive and each has a silent
#   failure mode that has already cost this project a run:
#
#   1. THE STOP IS A RACE. The launcher resumes from the HIGHEST
#      `-sb<N>-ema-raw.bin` on disk, and checkpoints land every 2 superbatches
#      (~3.5 min apart). Stopping "around sb700" by hand can easily resume from
#      sb702 or sb704, which then disagrees with the superbatch the LR was
#      scaled from. So this reads the ACTUAL resume point back off the disk
#      after the stop and scales from that, rather than from the number the
#      caller asked for.
#
#   2. STOPPING MID-WRITE TRUNCATES A CHECKPOINT. On 2026-08-02 a trainer was
#      stopped while it was serialising and left a 203 MB file where 261 MB was
#      expected — 10 of 21 tensors. It looked fine: the file existed and was
#      non-empty, which is all anything checked. Here the new checkpoint's size
#      is compared against the previous one and a mismatch is FATAL.
#
#   3. A FAILED RESTART IS INVISIBLE. `systemctl start` succeeds the instant
#      the unit is queued, so a trainer that dies during JAX init reports
#      success. This waits and confirms the superbatch counter has actually
#      advanced past the resume point before declaring done.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

AT_SB=${AT_SB:?AT_SB required}
FACTOR=${FACTOR:-1.25}
UNIT=${UNIT:-d50wtrain}
SPEC=${SPEC:-$ARCHON_ROOT/tools/nnue/runs/d50_w768.json}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/bump_lr.log}; export OPS_LOG

PREFIX=${PREFIX:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['net_id'])" "$SPEC")}
EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['out'])" "$SPEC")}

log "bump_lr_at: unit=$UNIT at=sb$AT_SB factor=$FACTOR prefix=$PREFIX"
log "  spec=$SPEC"
log "  ckpt=$EXP_DIR"

target="$EXP_DIR/${PREFIX}-sb${AT_SB}-ema-raw.bin"
prev="$EXP_DIR/${PREFIX}-sb$((AT_SB - 2))-ema-raw.bin"

# --- 1. wait for the checkpoint -------------------------------------------
log "waiting for $(basename "$target")"
for _ in $(seq 1 2000); do
    [ -s "$target" ] && break
    systemctl --user is-active --quiet "$UNIT" || { log "FATAL: $UNIT died while waiting"; exit 1; }
    sleep 30
done
[ -s "$target" ] || { log "FATAL: sb$AT_SB never appeared"; exit 1; }

# Let the writer finish. Size-stability is the check, not a fixed sleep: the
# file appears the moment serialisation STARTS.
last=0
for _ in $(seq 1 40); do
    sz=$(stat -c %s "$target" 2>/dev/null || echo 0)
    [ "$sz" -gt 0 ] && [ "$sz" -eq "$last" ] && break
    last=$sz; sleep 5
done
log "checkpoint settled at ${last} bytes"

# --- 2. verify integrity against the previous checkpoint -------------------
if [ -s "$prev" ]; then
    psz=$(stat -c %s "$prev")
    if [ "$last" -ne "$psz" ]; then
        log "FATAL: size mismatch — sb$AT_SB is $last B, sb$((AT_SB - 2)) is $psz B (truncated write?)"
        exit 3
    fi
    log "integrity ok: matches sb$((AT_SB - 2)) exactly ($psz B)"
else
    log "WARN: no sb$((AT_SB - 2)) to compare against; skipping size check"
fi

# --- 3. stop the trainer ---------------------------------------------------
log "stopping $UNIT"
systemctl --user stop "$UNIT"
sleep 5

# --- 4. read the ACTUAL resume point back off the disk ---------------------
RESUME=$(ls "$EXP_DIR/${PREFIX}"-sb*-ema-raw.bin 2>/dev/null \
         | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)
[ -n "$RESUME" ] || { log "FATAL: no checkpoint found to resume from"; exit 1; }
if [ "$RESUME" != "$AT_SB" ]; then
    log "NOTE: latest checkpoint is sb$RESUME, not sb$AT_SB — scaling from sb$RESUME so the"
    log "      ramp and the resume point agree (asking for sb$AT_SB would leave $((RESUME - AT_SB)) sb at the old LR)"
fi

# --- 5. scale the ramp tail ------------------------------------------------
python3 "$ARCHON_ROOT/tools/ops/train/scale_lr_tail.py" "$SPEC" \
    --from-sb "$RESUME" --factor "$FACTOR" \
    --note "x${FACTOR} from sb${RESUME} (owner request at sb${AT_SB})" 2>&1 | tee -a "$OPS_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { log "FATAL: LR scale failed (rc=$rc) — trainer left STOPPED, spec unchanged"; exit 4; }

# --- 6. restart and confirm it actually trains -----------------------------
log "restarting $UNIT from sb$RESUME"
systemctl --user start "$UNIT"
for _ in $(seq 1 40); do
    sleep 30
    systemctl --user is-active --quiet "$UNIT" || { log "FATAL: $UNIT failed to stay up"; exit 5; }
    cur=$(grep -oE '^components sb [0-9]+' "$EXP_DIR/cyc.log" 2>/dev/null | tail -1 | awk '{print $3}')
    if [ -n "$cur" ] && [ "$cur" -gt "$RESUME" ]; then
        log "DONE: training past sb$cur at the boosted LR"
        exit 0
    fi
done
log "WARN: unit is up but the superbatch counter has not passed sb$RESUME yet — check manually"
exit 0
