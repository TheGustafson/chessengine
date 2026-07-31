#!/usr/bin/env bash
# perf study 59 — block until a panel file has `n` round markers.
#   bash tools/perf59/waitpanel59.sh <file> <n>
set -euo pipefail
f=${1:?panel file}
n=${2:?rounds}
until [ "$(grep -c '^### round' "$f")" -ge "$n" ]; do sleep 20; done
echo "[waitpanel59] $f reached $n markers"
