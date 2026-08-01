#!/usr/bin/env bash
# tools/ops/guard/watchdog.sh — restart a training unit if it dies, but only
# while it is actually making progress.
#
# PURPOSE
#   Long runs get killed by things outside our control (six OOM kills in one
#   day, most caused by an unrelated `npm run build`). This restarts the unit so
#   the cost of each event stays ~1 superbatch — while refusing to relaunch a
#   run that is failing persistently.
#
# USAGE
#   systemd-run --user --unit=d49wd -p MemoryMax=128M \
#       /path/to/tools/ops/guard/watchdog.sh
#
#   Environment:
#     UNIT         systemd unit to watch/restart   (default d49train)
#     LAUNCHER     script the unit runs   (default tools/ops/train/d49_launch.sh)
#     PAYLOAD      regex identifying the real work  (default pump_train)
#     EXP_DIR      where cyc.log lives  (default data/nnue/experiments/d49-1280)
#     TARGET_SB    stop watching once reached   (default 0 = watch forever)
#     MAX_STALL    consecutive no-progress restarts before halting (default 3)
#     PERIOD       seconds between checks (default 100)
#
# THE TWO RULES THIS ENCODES
#
#   1. PROGRESS-GATED RESTARTS. A restart is allowed only if the run advanced
#      since the last one. Without that, a persistent fault — a bad resume, a
#      missing checkpoint, a recurring OOM — gets relaunched every 100 s for
#      days. MAX_STALL consecutive no-progress restarts and it HALTS loudly
#      instead, because at that point the fault is structural and a human needs
#      to look. Restarting a broken run forever is worse than stopping.
#
#   2. LIVENESS IS THE PAYLOAD, NOT `is-active`. The launcher is a
#      `while :; do ... done` loop service, so `systemctl is-active` stays true
#      forever — including after the payload exits and even after it completes.
#      Reading is-active as "the work is running" starved four separate things
#      in one day. We ask whether the PAYLOAD process exists.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

UNIT=${UNIT:-d49train}
LAUNCHER=${LAUNCHER:-$ARCHON_ROOT/tools/ops/train/d49_launch.sh}
PAYLOAD=${PAYLOAD:-pump_train}
EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/data/nnue/experiments/d49-1280}
TARGET_SB=${TARGET_SB:-0}
MAX_STALL=${MAX_STALL:-3}
PERIOD=${PERIOD:-100}
OPS_LOG="$EXP_DIR/cyc.log"; export OPS_LOG

sb_now() {
    grep -oE 'components sb [0-9]+' "$OPS_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'
}

log "watchdog: unit=$UNIT payload='$PAYLOAD' target=${TARGET_SB:-none} max_stall=$MAX_STALL"

stall=0
last_seen=-1
while :; do
    sleep "$PERIOD"
    cur=$(sb_now); cur=${cur:-0}

    if [ "$TARGET_SB" -gt 0 ] && [ "$cur" -ge "$TARGET_SB" ]; then
        log "watchdog: reached sb${cur} >= target sb${TARGET_SB} — COMPLETE"
        exit 0
    fi

    # Liveness = the payload exists. NOT is-active (see rule 2 above).
    if unit_running "$UNIT" "$PAYLOAD"; then
        last_seen=$cur
        stall=0
        continue
    fi

    if [ "$cur" -le "$last_seen" ]; then
        stall=$((stall + 1))
        if [ "$stall" -ge "$MAX_STALL" ]; then
            log "*** HALTED *** ${MAX_STALL} restarts with NO progress past sb${cur}."
            log "*** This is a structural fault (resume / checkpoint / config), not transient."
            log "*** Investigate before relaunching; the watchdog will NOT restart again."
            exit 1
        fi
    else
        stall=0
    fi
    last_seen=$cur

    log "watchdog: '$UNIT' payload absent at sb${cur} (stall=${stall}/${MAX_STALL}) — RESTARTING"
    systemctl --user reset-failed "$UNIT" 2>/dev/null
    systemd-run --user --unit="$UNIT" --quiet -p MemorySwapMax=0 "$LAUNCHER" \
        || log "watchdog: relaunch of '$UNIT' FAILED"
    # GPU memory release after a kill takes longer than it looks (>30 s measured).
    sleep 120
done
