#!/usr/bin/env bash
# perf study 36 — block until a panel log has the expected number of rows.
#
#   bash tools/perf36/waitpanel36.sh <panel.txt> <rows>
#
# Descends from tools/perf35/waitpanel35.sh. A background until-loop rather
# than a foreground sleep: a long job must never hold a turn hostage.
set -euo pipefail
cd "$(dirname "$0")/../.."
PANEL=${1:?panel}
WANT=${2:?rows}
while [ "$(grep -c . "$PANEL" 2>/dev/null || echo 0)" -lt "$WANT" ]; do
  sleep 15
done
echo "[waitpanel36] $PANEL reached $WANT rows"
