#!/usr/bin/env bash
# Fresh null at the CURRENT noise level + a replicate of the p1 arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "=== FRESH NULL (bit-identical source, n=8) ==="
bash tools/perf24/pair24.sh base nullB 8 bench16 | tail -6
echo "=== p1 REPLICATE (n=10) ==="
bash tools/perf24/pair24.sh a1b p1 10 bench16 | tail -6
echo "=== a1b REPLICATE vs base (n=8) ==="
bash tools/perf24/pair24.sh base a1b 8 bench16 | tail -6
