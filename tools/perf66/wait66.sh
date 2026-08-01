#!/usr/bin/env bash
# perf study 66 — block until a panel file has reached its expected row count.
#   bash tools/perf66/wait66.sh <file> <rows>
set -euo pipefail
F=${1:?file}
N=${2:?rows}
while [ "$(wc -l < "$F")" -lt "$N" ]; do sleep 15; done
echo "ready: $F"
