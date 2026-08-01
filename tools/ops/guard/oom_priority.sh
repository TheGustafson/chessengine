#!/usr/bin/env bash
# tools/ops/guard/oom_priority.sh — make the trainer the LAST thing the kernel
# OOM killer picks, by raising everything disposable above it.
#
# PURPOSE
#   Protect long training runs on a 15 GB box that has been OOM-killed six times
#   in a day. Writes to /proc/<pid>/oom_score_adj only. **Kills nothing, ever.**
#
# USAGE
#   systemd-run --user --unit=oompriority -p MemoryMax=128M \
#       /path/to/tools/ops/guard/oom_priority.sh
#
#   Environment:
#     PROTECT   regex of cmdlines to NEVER raise  (default 'pump_train')
#     ADJ       score to set on disposable work   (default 1000, the max)
#     PERIOD    seconds between sweeps            (default 15)
#
# WHY RAISING OTHERS, RATHER THAN LOWERING THE TRAINER
#   Everything launched from this harness inherits `oom_score_adj=200`, so the
#   trainer is BORN more attractive to the OOM killer than a default process
#   (0). Lowering it requires CAP_SYS_RESOURCE, which we do not have. Raising
#   other processes is always permitted and achieves the same ordering from the
#   other side.
#
#   Note `MemoryLow=` does NOT help here: it guards against RECLAIM and has no
#   bearing on OOM-killer victim selection, which scores by RSS x oom_score_adj.
#   That was tried (MemoryLow=4G) and the trainer was killed anyway.
#
# WHY STRUCTURAL MATCHING, NOT A TOOL LIST
#   This merges three earlier scripts (oomguard/oomshield/train_guard) that each
#   solved the same problem and each lapsed differently:
#     v1 matched only compilers      -> lapsed whenever no build was running.
#     v2 added a fixed tool-name list -> lapsed the moment a study used a NEW
#        tool (reprgap/probes.py held 6.9 GB at adj=200 while the trainer also
#        sat at 200).
#   So we match STRUCTURALLY: any process whose cmdline references this repo
#   (tools/, worktrees/, engines/bin/), plus compilers, plus the node/npm family
#   — which is what actually killed the trainer three times on 2026-08-01, and
#   which no repo-scoped pattern would ever have caught.
#
# WHAT IS NEVER TOUCHED
#   * the trainer itself (PROTECT regex)
#   * wrapper shells (`shell-snapshots` — every harness command sources one, so
#     matching them is the self-match trap that makes pgrep hit its own caller)
#   * the harness process

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

PROTECT=${PROTECT:-'pump_train'}
ADJ=${ADJ:-1000}
PERIOD=${PERIOD:-15}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/oom_priority.log}; export OPS_LOG

log "oom_priority: adj=$ADJ protect='$PROTECT' period=${PERIOD}s root=$ARCHON_ROOT"
log "self oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null)"

raised_total=0
while :; do
    # Candidates: compilers by name, anything referencing this repo, and the
    # node/npm family (external builds are the measured top cause of our OOMs).
    for p in $(pgrep -x 'rustc|cargo|cc1|ld' 2>/dev/null; \
               pgrep -f "$ARCHON_ROOT" 2>/dev/null; \
               pgrep -f 'node-MainThread|npm |npm$|esbuild|webpack|vite' 2>/dev/null); do
        [ "$p" = "$$" ] && continue
        [ "$p" = "$PPID" ] && continue
        cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        [ -z "$cmd" ] && continue

        # never these
        case "$cmd" in
            *shell-snapshots*)  continue;;   # wrapper shell (self-match trap)
            *claude*)           continue;;   # the harness
            *oom_priority*)     continue;;   # ourselves
        esac
        echo "$cmd" | grep -qE "$PROTECT" && continue

        cur=$(cat "/proc/$p/oom_score_adj" 2>/dev/null) || continue
        if [ "${cur:-0}" -lt "$ADJ" ]; then
            if echo "$ADJ" > "/proc/$p/oom_score_adj" 2>/dev/null; then
                raised_total=$((raised_total + 1))
                # log only the first few per process class to avoid log spam
                [ "$raised_total" -le 50 ] && log "raised pid $p ($cur -> $ADJ): ${cmd:0:70}"
            fi
        fi
    done
    sleep "$PERIOD"
done
