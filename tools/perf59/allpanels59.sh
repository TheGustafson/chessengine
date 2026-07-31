#!/usr/bin/env bash
# perf study 59 — the four panels, run SEQUENTIALLY on one core.
#
# They are sequential on purpose: `panel59.sh` pins to cpu1 and two panels on
# one core measure each other. `flock` on the output path stops the study-57
# accident (two jobs, one file) but it cannot stop two jobs from one shell.
set -euo pipefail
cd "$(dirname "$0")/../.."
A=docs/research/perf-artifacts-59
mkdir -p "$A"

# 1. The gate instrument: native, non-PGO, exact counters.
P59_PANEL=$A/panel-native.txt bash tools/perf59/panel59.sh 24 base null lutp final

# 2. The SHIPPED binary: native + PGO. 39 rounds, because study 57 Sec 9.3's
#    calibration says a ~1% cycles effect needs ~40 paired rounds for the sign
#    test and this study has a cycles claim to make.
P59_PANEL=$A/panel-pgo.txt bash tools/perf59/panel59.sh 39 base-pgo nullpgo-pgo final-pgo

# 3. The portable rung: x86-64-v3, non-PGO. The whole point of the study's
#    headline rule — an arm's value is a property of the target-cpu.
P59_PANEL=$A/panel-v3.txt bash tools/perf59/panel59.sh 24 base-v3 null-v3 lutp-v3 final-v3

# 4. The startup frame (1 node): where the quantiser arm actually lives.
P59_BUDGET=1 P59_PANEL=$A/panel-startup.txt \
  bash tools/perf59/panel59.sh 39 base null final
echo "[allpanels59] done"
