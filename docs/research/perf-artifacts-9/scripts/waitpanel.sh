#!/usr/bin/env bash
# Block until the named pstat8 log has at least N data rows, then print DONE.
set -euo pipefail
FILE=${1:?log file}
WANT=${2:?row count}
while [ "$(grep -c '^round=' "$FILE" || echo 0)" -lt "$WANT" ]; do
  sleep 10
done
echo PANEL_DONE
