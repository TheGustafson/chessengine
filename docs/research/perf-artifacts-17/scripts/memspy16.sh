#!/usr/bin/env bash
# perf study 16 — memcpy/memmove/MEMSET census by CALLER.
#
#   bash memspy16.sh <label> [nodes-per-position]
#
# study 15's instrument plus the hook it was missing. LD_PRELOADs an interposer
# that records every call's size and `__builtin_return_address(0)` minus the
# load base, then resolves each return address against the binary's own
# disassembly. Call counts and byte totals are deterministic workload facts,
# not samples.
set -euo pipefail
LABEL=${1:?label}
NPP=${2:-3000}
SCR=docs/research/perf-artifacts-16/scripts
OUT=docs/research/perf-artifacts-16/memspy-$LABEL.txt
BIN=${MEMSPY_BIN:-target-fast16/release/chessengine}
SO=$(mktemp -u /tmp/memspy16-XXXX.so)
gcc -O2 -shared -fPIC -o "$SO" "$SCR/memspy.c"

raw=$(mktemp)
MEMSPY_OUT="$raw" LD_PRELOAD="$SO" nice -n 19 taskset -c 6 \
  "$BIN" bench 16 1 "$NPP" tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > "$raw.eng" 2>&1 || true
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$raw.eng" | tail -1)

objdump -d --demangle "$BIN" 2>/dev/null | grep -E '^[0-9a-f]+ <' \
  | sed 's/[<>:]//g' > "$raw.syms"

python3 - "$raw" "$raw.syms" "$nodes" "$LABEL" <<'PY' | tee "$OUT"
import sys, bisect
raw, symf, nodes, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
syms = []
for line in open(symf):
    a, _, name = line.strip().partition(' ')
    try:
        syms.append((int(a, 16), name))
    except ValueError:
        pass
syms.sort()
starts = [s[0] for s in syms]
rows = []
for line in open(raw):
    parts = line.split()
    n, b, off = int(parts[0]), int(parts[1]), int(parts[2], 16)
    kind = parts[3] if len(parts) > 3 else "?"
    i = bisect.bisect_right(starts, off) - 1
    who = f"{syms[i][1]} +0x{off - starts[i]:x}" if i >= 0 else f"0x{off:x}"
    rows.append((n, b, kind, who))
rows.sort(reverse=True)
tot_n = sum(r[0] for r in rows)
tot_b = sum(r[1] for r in rows)
print(f"# memspy16 label={label} nodes={nodes:,} total_calls={tot_n:,} total_bytes={tot_b:,}"
      f" calls_per_node={tot_n/nodes:.4f} bytes_per_node={tot_b/nodes:.1f}")
for k in ("memcpy", "memmove", "memset"):
    kn = sum(r[0] for r in rows if r[2] == k)
    kb = sum(r[1] for r in rows if r[2] == k)
    print(f"#   {k:>8}: calls={kn:>10,} bytes={kb:>14,} per_node={kn/nodes:.4f}")
print(f"{'calls':>9} {'bytes':>14} {'size':>8} {'per_node':>9}  {'kind':<8} caller")
for n, b, kind, who in rows:
    if n < 20:
        continue
    print(f"{n:>9,} {b:>14,} {b/n:>8.1f} {n/nodes:>9.4f}  {kind:<8} {who}")
PY
rm -f "$raw" "$raw.eng" "$raw.syms" "$SO"
