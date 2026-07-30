#!/usr/bin/env bash
# Stop any running perf24 measurement, BY PID.
#
# Never `pkill -f` (CLAUDE.md): it matches the shell running the very command
# that typed the pattern. `pgrep` + an explicit `shell-snapshots` filter is the
# form that is safe for BOTH killing and detection -- a `$$`/`$PPID` filter is
# not, because a `$(...)` subshell's pid is neither.
for pat in 'renull24.sh' 'pair24.sh' 'perf24/bin/'; do
    pgrep -f "$pat" | while read -r p; do
        cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case $cmd in *shell-snapshots*|*stop24*) continue ;; esac
        echo "kill $p  $cmd"
        kill "$p" 2>/dev/null || true
    done
done
echo "stop24: done"
