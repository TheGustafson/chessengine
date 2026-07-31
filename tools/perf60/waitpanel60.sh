#!/usr/bin/env bash
# perf study 60 — block until a panel file has `rounds * labels` records.
#   bash tools/perf60/waitpanel60.sh <panel-file> <expected-records>
# Exists so a turn can be ended while a panel runs (the standing "background
# waits, not blocking" rule) without holding the shell on a bare `sleep`.
set -euo pipefail
cd "$(dirname "$0")/../.."
F=${1:?panel file}
N=${2:?expected record count}
# `grep -c` on an EMPTY file prints 0 *and* exits 1, so a `|| echo 0` fallback
# appends a second line and the `[ -lt ]` then sees "0\n0" and dies. Count with
# grep alone and treat a missing file as zero explicitly.
while :; do
  if [ -f "$F" ]; then n=$(grep -c '^###' "$F" || true); else n=0; fi
  [ "${n:-0}" -ge "$N" ] && break
  sleep 15
done
echo "panel complete: $F ($N records)"
