#!/usr/bin/env bash
# perf study 56 -- did the arm actually delete the bounds checks it claims?
# Prints the total panic-edge count of each binary, from tools/perf56/census56.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
for b in "$@"; do
  printf '%-12s ' "$b"
  python3 tools/perf56/census56.py "bin56/$b" 0 2>/dev/null | head -1
done
