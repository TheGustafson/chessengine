#!/usr/bin/env bash
# perf study 34 — block until the panel log reaches N rows.
#   bash tools/perf34/waitpanel34.sh <rows> [file]
set -euo pipefail
cd "$(dirname "$0")/../.."
WANT=${1:?rows}
FILE=${2:-docs/research/perf-artifacts-34/panel.txt}
while true; do
  n=$(wc -l < "$FILE")
  if [ "$n" -ge "$WANT" ]; then
    echo "panel has $n rows"
    exit 0
  fi
  sleep 20
done
