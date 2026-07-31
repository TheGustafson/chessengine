#!/usr/bin/env bash
# perf study 60 — the identity gate for one arm binary in bin60/.
#   bash tools/perf60/ident60.sh <label>...
# Prints the node total and the wl50 eval hash; both must match the pins, and a
# memory arm that changes either is not a memory arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=${P60_NET:-engines/canonical/net.cenn}
FENS=${P60_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P60_BIN:-bin60}
NODEPIN=5001562
HASHPIN=2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4
rc=0
for lab in "$@"; do
  # The bench summary goes to STDERR (the per-position lines go to stdout).
  nodes=$("./$BIN/$lab" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null \
          | sed -n 's/^Nodes searched  : //p' | tail -1)
  hash=$("./$BIN/$lab" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)
  ok=OK
  [ "$nodes" = "$NODEPIN" ] || { ok="NODES-DIFFER"; rc=1; }
  [ "$hash"  = "$HASHPIN"  ] || { ok="EVAL-DIFFERS"; rc=1; }
  printf '%-12s nodes=%-9s hash=%s  %s\n' "$lab" "${nodes:-NONE}" "${hash:0:16}" "$ok"
done
exit $rc
