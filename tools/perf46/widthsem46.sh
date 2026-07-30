#!/usr/bin/env bash
# perf study 46 -- CLOSE study 44's assumption A4.
#
# Study 44 could not say whether `fp_ops_retired_by_width.pack_512_uops_retired`
# counts ONE event per 512-bit instruction or TWO (Zen 4 executes 512-bit ops on
# a 256-bit datapath, so a "double-pumped halves" reading is plausible). It
# bracketed its port-pressure verdict at both readings rather than assert one.
#
# The bracket is closable by measurement: run a KNOWN number of 512-bit
# instructions and read the counter.  `ilp` prints its own exact op count, so
# the ratio counter/ops is the answer directly:
#
#     ratio ~ 1.0  -> one event per 512-bit INSTRUCTION
#     ratio ~ 2.0  -> one event per double-pumped HALF
#
# The 256-bit arm is the control: pack_256 against the same loop shape must read
# ~1.0, since a 256-bit op is not double-pumped on this machine.
#
#   bash tools/perf46/widthsem46.sh [cpu]
set -euo pipefail
CPU=${1:-0}
ILP=${P46_ILP:-/tmp/ilp}
if [ ! -x "$ILP" ]; then
  echo "[widthsem46] building $ILP from tools/perf46/gen_ilp.py" >&2
  python3 "$(dirname "$0")/gen_ilp.py" /tmp/ilp.c
  gcc -O2 -mavx512f -mavx512vnni -mavx512bw -mavx512vl -o "$ILP" /tmp/ilp.c
fi

probe() { # insn nch iters event
  local out ops ev
  out=$(perf stat -x, -e "$4:u" -- taskset -c "$CPU" "$ILP" "$1" "$2" "$3" 2>&1)
  ops=$(echo "$out" | grep -E '^[0-9]+$' | tail -1)
  ev=$(echo "$out" | awk -F, -v e="$4:u" '$3==e{print $1}')
  awk -v e="$ev" -v o="$ops" 'BEGIN{printf "%14s  ops=%-12s counter=%-14s ratio=%.4f\n", "", o, e, e/o}'
}

echo "=== 512-bit: pack_512_uops_retired per 512-bit INSTRUCTION ==="
for insn in vfmadd231ps vpdpbusd vpaddd; do
  printf '%-14s\n' "$insn"
  probe "$insn" 8 6250000 fp_ops_retired_by_width.pack_512_uops_retired
done

echo "=== 256-bit CONTROL: pack_256 must read ~1.00 (no double pump) ==="
for insn in vfmadd231ps_y vpdpbusd_y; do
  printf '%-14s\n' "$insn"
  probe "$insn" 8 6250000 fp_ops_retired_by_width.pack_256_uops_retired
done
