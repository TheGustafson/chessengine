#!/usr/bin/env bash
# perf study 41 — is anything from this study still running?
#
#   bash tools/perf41/live41.sh [pattern]
#
# CLAUDE.md's detection rule: a bare `pgrep -f` also matches the shell that
# typed the pattern, and the `$$`/`$PPID` filter is not enough because a
# `$(...)` runs in a subshell whose pid is neither. Every harness-run command
# sources `shell-snapshots`, so that is the signature to drop.
set -euo pipefail
PAT=${1:-panel40.sh|census41.sh|chessengine bench}
{ pgrep -f "$PAT" || true; } | while read -r p; do
  cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
  case "$cl" in *shell-snapshots*) continue ;; esac
  [ -n "$cl" ] && printf '%s  %.90s\n' "$p" "$cl"
done
echo "[live41] done"
