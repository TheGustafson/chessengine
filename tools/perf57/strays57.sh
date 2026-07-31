#!/usr/bin/env bash
# perf study 57 — list this study's stray measurement processes, EXCLUDING the
# shell that is running this very command.
#
# Never `pkill -f` (CLAUDE.md, user instruction 2026-07-27): the pattern
# matches the caller's own command line. The `shell-snapshots` filter is the
# reliable self-exclusion — a `pgrep` inside `$(...)` runs in a subshell whose
# pid is neither `$$` nor `$PPID`, so those two are not enough on their own.
set -uo pipefail
pat=${1:-bin57/}
pgrep -f "$pat" | while read -r p; do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
  case "$cmd" in
    *shell-snapshots*) continue ;;
    *strays57*) continue ;;
  esac
  printf '%s\t%s\n' "$p" "$cmd"
done
