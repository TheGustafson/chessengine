#!/usr/bin/env bash
# Block until the SMP curve has all 60 rows, then report.
set -euo pipefail
F=docs/research/perf-artifacts-9/smp-uci.txt
while [ "$(grep -c '^round=' "$F")" -lt 60 ]; do sleep 15; done
echo SMP_CURVE_COMPLETE
