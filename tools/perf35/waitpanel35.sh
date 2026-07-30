#!/usr/bin/env bash
# perf study 35 -- block until the panel has written <n> lines.
#   bash tools/perf35/waitpanel35.sh <lines> [panel-path]
set -euo pipefail
cd "$(dirname "$0")/../.."
WANT=${1:?lines}
P=${2:-docs/research/perf-artifacts-35/panel.txt}
until [ "$(wc -l < "$P")" -ge "$WANT" ]; do sleep 20; done
wc -l "$P"
