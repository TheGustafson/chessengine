#!/usr/bin/env bash
# perf study 15 — memcpy/memmove census by CALLER (study 14 §3's instrument,
# wrapped so it is one command).
#
#   bash memspy15.sh <label> [nodes-per-position]
#
# LD_PRELOADs a memcpy/memmove interposer that records every call's size and
# `__builtin_return_address(0)` minus the load base, then resolves each return
# address against the binary's own disassembly. Call counts and byte totals are
# deterministic workload facts, not samples.
set -euo pipefail
LABEL=${1:?label}
NPP=${2:-3000}
SCR=docs/research/perf-artifacts-15/scripts
OUT=docs/research/perf-artifacts-15/memspy-$LABEL.txt
BIN=target-fast15/release/chessengine
SO=$(mktemp -u /tmp/memspy15-XXXX.so)
gcc -O2 -shared -fPIC -o "$SO" "$SCR/memspy.c"

raw=$(mktemp)
MEMSPY_OUT="$raw" LD_PRELOAD="$SO" nice -n 19 taskset -c 6 \
  "$BIN" bench 16 1 "$NPP" tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > "$raw.eng" 2>&1 || true
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$raw.eng" | tail -1)

objdump -d --demangle "$BIN" 2>/dev/null | grep -E '^[0-9a-f]+ <' \
  | sed 's/[<>:]//g' > "$raw.syms"

python3 - "$raw" "$raw.syms" "$nodes" "$LABEL" <<'PY' | tee "$OUT"
import sys
raw, symf, nodes, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
syms = []
for line in open(symf):
    a, _, name = line.strip().partition(' ')
    try:
        syms.append((int(a, 16), name))
    except ValueError:
        pass
syms.sort()
import bisect
starts = [s[0] for s in syms]
rows = []
for line in open(raw):
    n, b, off = line.split()
    off = int(off, 16)
    i = bisect.bisect_right(starts, off) - 1
    who = f"{syms[i][1]} +0x{off - starts[i]:x}" if i >= 0 else f"0x{off:x}"
    rows.append((int(n), int(b), who))
rows.sort(reverse=True)
tot_n = sum(r[0] for r in rows)
tot_b = sum(r[1] for r in rows)
print(f"# memspy15 label={label} nodes={nodes:,} total_calls={tot_n:,} total_bytes={tot_b:,}"
      f" calls_per_node={tot_n/nodes:.4f} bytes_per_node={tot_b/nodes:.1f}")
print(f"{'calls':>9} {'bytes':>14} {'size':>8} {'per_node':>9}  caller")
for n, b, who in rows:
    if n < 20:
        continue
    print(f"{n:>9,} {b:>14,} {b/n:>8.1f} {n/nodes:>9.4f}  {who}")
PY
rm -f "$raw" "$raw.eng" "$raw.syms" "$SO"
