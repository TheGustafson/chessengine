#!/usr/bin/env bash
# perf study 37 — what did the kernel actually DO with the huge-page hint?
#
#   bash tools/perf37/vmflags37.sh <binary>
#
# `AnonHugePages: 0` in smaps_rollup has two very different causes and they
# need different fixes: the advice never reached the VMA (no `hg` in VmFlags),
# or it reached it and the kernel declined to honour it (`hg` present,
# AnonHugePages still 0 — fragmentation). This prints, for every mapping over
# 20 MB, the size, the huge-page bytes, and the VmFlags line that distinguishes
# the two.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
NET=engines/canonical/net.cenn

fifo=$(mktemp -u)
mkfifo "$fifo"
"$BIN" < "$fifo" > /dev/null 2>&1 &
pid=$!
exec 9> "$fifo"
printf 'uci\nsetoption name EvalFile value %s\nisready\n' "$NET" >&9
sleep 6
awk '
  /^[0-9a-f]+-[0-9a-f]+ / { addr = $1 }
  /^Size:/                { sz = $2 }
  /^AnonHugePages:/       { ahp = $2 }
  /^VmFlags:/             { if (sz + 0 > 20000) printf "%-30s size=%9s kB  AnonHuge=%8s kB  %s\n", addr, sz, ahp, $0 }
' "/proc/$pid/smaps"
grep -H . /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
printf 'quit\n' >&9
exec 9>&-
wait "$pid" 2>/dev/null || true
rm -f "$fifo"
