#!/usr/bin/env bash
# perf study 40 — per-CPU busy fraction over a short window.
#
#   bash tools/perf40/cpubusy40.sh [seconds]
#
# The panel pins the bench to ONE logical CPU, so the thing that decides its
# cycle noise is what runs on that CPU's SMT SIBLING: two threads on one Zen 4
# core share the front end, the schedulers and L1/L2. This prints both halves of
# every core so a panel can be pinned to the quietest pair.
set -euo pipefail
W=${1:-3}
awk '/^cpu[0-9]/{print $1, $2+$3+$4+$6+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat >/tmp/cpubusy40.a
sleep "$W"
awk '/^cpu[0-9]/{print $1, $2+$3+$4+$6+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat >/tmp/cpubusy40.b
paste /tmp/cpubusy40.a /tmp/cpubusy40.b | awk '{
  busy = $5 - $2; tot = $6 - $3;
  printf "%-7s %5.1f%%\n", $1, tot ? 100*busy/tot : 0
}'
