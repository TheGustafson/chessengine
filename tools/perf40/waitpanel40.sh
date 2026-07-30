#!/usr/bin/env bash
# perf study 40 — block until a panel log has at least N rows.
#   bash tools/perf40/waitpanel40.sh <panel.txt> <rows>
set -euo pipefail
cd "$(dirname "$0")/../.."
F=${1:?panel}
N=${2:?rows}
while [ "$(grep -c . "$F" 2>/dev/null || echo 0)" -lt "$N" ]; do sleep 15; done
echo "[waitpanel40] $F has $(grep -c . "$F") rows"
