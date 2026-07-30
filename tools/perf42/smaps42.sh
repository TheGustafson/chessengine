#!/usr/bin/env bash
# perf study 42 -- what BACKS the transposition table.
#
#   bash tools/perf42/smaps42.sh <hashMB> [binary] [budget]
#
# Runs a bench long enough to sample /proc/<pid>/smaps, and prints every anon
# mapping >= 4 MB with its AnonHugePages line. The TT is the mapping whose
# size matches <hashMB>; whether the kernel gave it 2 MB pages decides how many
# dTLB entries a probe costs, and THP is `always` on this box -- which only
# helps a mapping the allocator happened to place on a 2 MB boundary.
set -euo pipefail
cd "$(dirname "$0")/../.."
MB=${1:-16}
BIN=${2:-bin34/b42}
BUDGET=${3:-400000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
taskset -c 2 "./$BIN" bench "$MB" 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>&1 &
pid=$!
sleep 3
awk -v want=$((MB * 1024)) '
  /^[0-9a-f]+-[0-9a-f]+ / { hdr=$1" "$2; sz=""; thp="" }
  /^Size:/ { sz=$2 }
  /^AnonHugePages:/ {
      thp=$2
      if (sz+0 >= 4096)
        printf "%-40s size=%8d kB  AnonHugePages=%8d kB  %s\n", hdr, sz, thp, (sz+0==want ? "<== TT-sized" : "")
  }' "/proc/$pid/smaps"
echo "--- totals ---"
grep -E "^VmRSS|^HugetlbPages" "/proc/$pid/status" || true
awk '/^AnonHugePages:/ { t += $2 } END { print "AnonHugePages total: " t " kB" }' "/proc/$pid/smaps"
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
