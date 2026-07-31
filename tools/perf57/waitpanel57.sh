#!/usr/bin/env bash
# perf study 57 — block until a panel file has `n` round markers.
#   bash tools/perf57/waitpanel57.sh <file> <n>
set -euo pipefail
f=${1:?panel file}
n=${2:?rounds}
until [ "$(grep -c '^### round' "$f")" -ge "$n" ]; do sleep 20; done
echo "[waitpanel57] $f reached $n markers"
