#!/usr/bin/env bash
# tools/ops/lib/common.sh — shared helpers for every script under tools/ops/.
#
# PURPOSE
#   One place for the four things that broke repeatedly on 2026-08-01. Every
#   helper here exists because its absence cost a run, a match, or a verdict.
#   Source it from the first lines of any ops script:
#
#       source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
#
#   It sets `set -uo pipefail` and cd's to ARCHON_ROOT, so scripts start from a
#   known state and never depend on the caller's working directory.
#
# WHAT IT PROVIDES
#   ARCHON_ROOT            absolute path to the primary checkout
#   ARCHON_D49             the durable state/log dir (ext4, survives reboot)
#   need_mem MB            exit non-zero unless MemAvailable >= MB
#   hold_for_mem MB [S]    block until MemAvailable >= MB (polls, never fails)
#   safe_pids PATTERN      pids matching PATTERN, excluding our own shell
#   safe_kill PATTERN      kill those pids (never our own caller)
#   unit_running UNIT      true only if UNIT's PAYLOAD is alive (see WHY below)
#   wait_file PATH [S] [N] block until PATH exists and is non-empty
#   log MSG                timestamped line to stderr and $OPS_LOG if set
#
# WHY EACH ONE EXISTS (receipts, all 2026-08-01)
#   ARCHON_ROOT — scripts cd'd to a hardcoded path or relied on cwd. A PGO
#     build ran with --working-directory=chessengine-mainwt, wrote its binary
#     there, and the SPRT that consumed it cd'd to the primary checkout and
#     failed with "candidate missing". The same split also broke a merge gate,
#     a net path, a fastchess lookup, sprt.sh, and the trainer's launcher —
#     SEVEN incidents from one root cause.
#   need_mem / hold_for_mem — the box is 15 GB and OOM-killed six times.
#   safe_kill — `pkill -f PATTERN` matches the shell running the very command
#     that typed the pattern, so it kills its own caller; the turn dies with a
#     bare exit 143 and no output. The bracket dodge ('fastches[s]') fails the
#     moment the same command line also mentions the plain name. Excluding $$
#     and $PPID is ALSO insufficient for DETECTION, because a pgrep inside
#     $(...) runs in a subshell whose pid is neither — so we additionally drop
#     anything whose cmdline mentions shell-snapshots, which every harness-run
#     command sources.
#   unit_running — `systemctl is-active` on a `while :; do ... done` loop
#     service is ALWAYS "active", even when its payload has exited. That single
#     misreading starved four separate things in one day: a frame arm, an A/B
#     twice, and the entire build->match chain. NEVER ask is-active whether work
#     is done; ask whether the payload process exists.
#
# CONVENTIONS FOR SCRIPTS IN THIS TREE
#   - Long-running work is a systemd transient SERVICE (systemd-run --user,
#     WITHOUT --scope). A --scope dies with the session; that is what killed a
#     5880-game screen at game 1043.
#   - Builds get -p MemoryMax=<cap> -p MemorySwapMax=0 and CARGO_BUILD_JOBS=2.
#   - Anything that can be waited on waits on an ARTIFACT (a file, a pid), never
#     on is-active.

set -uo pipefail

# --- roots ------------------------------------------------------------------
# Resolved from THIS FILE's location, never from cwd: tools/ops/lib/ -> ../../..
ARCHON_ROOT="${ARCHON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export ARCHON_ROOT
# ARCHON_MAIN — the checkout that owns the big shared artifacts (binaries,
# books, nets). Normally identical to ARCHON_ROOT; they diverge when a script
# runs from a git WORKTREE, which shares the object store but NOT untracked
# directories. In one such worktree `data/` was a symlink back to the primary
# checkout while `engines/` was a real, nearly-empty directory — so a script
# found nets and missed every binary, and an equal-node Stockfish match failed
# with "our engine not executable" and "stockfish not executable" in one run.
#
# Rule: CODE comes from ARCHON_ROOT (the tree under test); BINARIES, BOOKS and
# NETS come via asset(), which falls back to ARCHON_MAIN.
ARCHON_MAIN="${ARCHON_MAIN:-$(git -C "$ARCHON_ROOT" rev-parse --git-common-dir 2>/dev/null | sed 's|/\.git$||')}"
[ -d "$ARCHON_MAIN" ] || ARCHON_MAIN="$ARCHON_ROOT"
export ARCHON_MAIN

# asset RELPATH — echo the first existing absolute path, preferring
# ARCHON_ROOT then ARCHON_MAIN. Returns non-zero (and echoes nothing) if absent,
# so callers can `asset x || asset y` to express a fallback chain.
asset() {
    local rel=${1:?asset RELPATH} d
    case "$rel" in /*) [ -e "$rel" ] && { echo "$rel"; return 0; }; return 1;; esac
    for d in "$ARCHON_ROOT" "$ARCHON_MAIN"; do
        [ -e "$d/$rel" ] && { echo "$d/$rel"; return 0; }
    done
    return 1
}

ARCHON_D49="${ARCHON_D49:-/home/nick/.local/share/archon-d49}"
export ARCHON_D49
mkdir -p "$ARCHON_D49"
cd "$ARCHON_ROOT" || { echo "common.sh: cannot cd to ARCHON_ROOT=$ARCHON_ROOT" >&2; exit 9; }

# --- logging ----------------------------------------------------------------
log() {
    local m="[$(date -Is)] $*"
    echo "$m" >&2
    [ -n "${OPS_LOG:-}" ] && echo "$m" >> "$OPS_LOG"
    return 0
}

# --- memory -----------------------------------------------------------------
mem_avail_mb() { awk '/^MemAvailable/{print int($2/1024)}' /proc/meminfo; }

# need_mem MB — hard gate. Use before any build.
need_mem() {
    local want=${1:?need_mem MB} have; have=$(mem_avail_mb)
    if [ "$have" -lt "$want" ]; then
        log "REFUSE: MemAvailable ${have}MB < ${want}MB required"
        return 1
    fi
    log "memory ok: ${have}MB available (needed ${want}MB)"
    return 0
}

# hold_for_mem MB [SLEEP] — soft gate; blocks instead of failing. For loops that
# should pause rather than die when the box is busy.
hold_for_mem() {
    local want=${1:?hold_for_mem MB} nap=${2:-60} have
    while :; do
        have=$(mem_avail_mb)
        [ "$have" -ge "$want" ] && return 0
        log "holding: ${have}MB < ${want}MB"
        sleep "$nap"
    done
}

# --- process handling -------------------------------------------------------
# safe_pids PATTERN — pids matching PATTERN, minus anything that is us.
# Drops $$ / $PPID AND any process whose cmdline mentions shell-snapshots
# (every harness-run command sources one), which is what makes this safe to
# call from inside $(...) where $$ is a subshell pid.
safe_pids() {
    local pat=${1:?safe_pids PATTERN} p
    pgrep -f "$pat" 2>/dev/null | while read -r p; do
        [ "$p" = "$$" ] && continue
        [ "$p" = "$PPID" ] && continue
        tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q shell-snapshots && continue
        echo "$p"
    done
}

safe_kill() {
    local pat=${1:?safe_kill PATTERN} pids
    pids=$(safe_pids "$pat")
    [ -z "$pids" ] && { log "safe_kill: nothing matches '$pat'"; return 0; }
    log "safe_kill: killing $(echo "$pids" | tr '\n' ' ')"
    echo "$pids" | xargs -r kill
}

# unit_running UNIT [PAYLOAD_PATTERN] — is the unit's WORK actually running?
# Falls back to is-active only when no payload pattern is given, and says so.
unit_running() {
    local unit=${1:?unit_running UNIT} pat=${2:-}
    [ "$(systemctl --user is-active "$unit" 2>/dev/null)" = "active" ] || return 1
    [ -z "$pat" ] && return 0                      # no payload known: is-active it is
    [ -n "$(safe_pids "$pat")" ]                   # loop-service-safe
}

# --- waiting ----------------------------------------------------------------
# wait_file PATH [SLEEP] [TRIES] — block until PATH exists and is non-empty.
# Preferred over any is-active poll: an artifact on disk cannot lie.
wait_file() {
    local path=${1:?wait_file PATH} nap=${2:-30} tries=${3:-240} i=0
    while [ "$i" -lt "$tries" ]; do
        [ -s "$path" ] && { log "wait_file: $path present"; return 0; }
        sleep "$nap"; i=$((i+1))
    done
    log "wait_file: TIMEOUT waiting for $path"
    return 1
}
