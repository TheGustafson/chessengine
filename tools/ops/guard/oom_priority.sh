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
# RSS floor (kB) for the exe-based heap sweep. Only processes big enough to
# decide an OOM are worth reclassifying; 200 MB skips the long tail.
MIN_RSS_KB=${MIN_RSS_KB:-200000}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/oom_priority.log}; export OPS_LOG

log "oom_priority: adj=$ADJ protect='$PROTECT' period=${PERIOD}s root=$ARCHON_ROOT"
log "self oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null)"

raised_total=0
while :; do
    # Candidates: compilers by name, anything referencing this repo, and the
    # node/npm family (external builds are the measured top cause of our OOMs).
    # `-x` matches the process NAME (comm); `-f` matches the full cmdline. The
    # node family has to be caught by NAME: a node process's cmdline is
    # `/usr/bin/node /path/to/eslint ...`, so the thread name `node-MainThread`
    # that `ps -o comm` prints appears in no cmdline at all. Matching it with
    # `-f` therefore matched exactly one thing — this loop's own pgrep argv,
    # which contains the literal string — while every real node process stayed
    # at adj=200. That is the self-match trap this file's own header warns
    # about, and it silently disarmed the guard against the measured top cause
    # of our OOMs for as long as the pattern existed.
    #
    # NAME AND CMDLINE ARE BOTH FORGEABLE; /proc/<pid>/exe IS NOT.
    # A process can rewrite its own argv (`process.title`), and comm is capped
    # at 15 chars. Next.js does exactly this: the dev server presents as
    # `next-server (v16.2.3)` with a cmdline of the same string and nothing
    # else, so it matches NO name pattern and NO cmdline pattern while holding
    # 1.13 GB at adj=200. `readlink /proc/<pid>/exe` still says /usr/bin/node.
    # So candidates are ALSO drawn from a heap sweep: any process above
    # MIN_RSS_KB whose exe is an interpreter/toolchain binary. The RSS floor
    # keeps this cheap and points it at the only processes that can decide an
    # OOM anyway.
    heap_candidates=""
    for sp in /proc/[0-9]*; do
        hp=${sp#/proc/}
        rss=$( { awk '/^VmRSS:/{print $2}' "$sp/status"; } 2>/dev/null ) || continue
        [ -n "$rss" ] && [ "$rss" -ge "$MIN_RSS_KB" ] || continue
        exe=$(readlink "$sp/exe" 2>/dev/null) || continue
        case "${exe##*/}" in
            node|npm|npx|deno|bun|esbuild|rustc|cargo|cc1|cc1plus|ld|ld.lld|lld)
                heap_candidates="$heap_candidates $hp";;
        esac
    done

    for p in $(pgrep -x 'rustc|cargo|cc1|ld' 2>/dev/null; \
               pgrep -x 'node|npm|npx|esbuild' 2>/dev/null; \
               pgrep -f "$ARCHON_ROOT" 2>/dev/null; \
               pgrep -f 'npm |npm$|esbuild|webpack|vite' 2>/dev/null; \
               echo $heap_candidates); do
        [ "$p" = "$$" ] && continue
        [ "$p" = "$PPID" ] && continue
        # A pid from pgrep can exit before we read it. `2>/dev/null` on `tr`
        # does NOT silence this: the failing part is the SHELL's own `<`
        # redirection, whose diagnostic is emitted before tr ever runs. Putting
        # the redirect inside a subshell that redirects its own stderr is what
        # actually suppresses it (this logged a spurious "No such file or
        # directory" every few minutes otherwise).
        cmd=$( { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null ) || continue
        [ -z "$cmd" ] && continue

        # never these
        case "$cmd" in
            *shell-snapshots*)  continue;;   # wrapper shell (self-match trap)
            *claude*)           continue;;   # the harness
            *codex*)            continue;;   # the owner's other agent harness
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
