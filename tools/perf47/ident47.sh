#!/usr/bin/env bash
# perf study 44 — the FRAME GATE. Run it on every arm before any panel.
#
#   bash tools/perf46/ident44.sh <label>...
#
# A cache hint computes no value the search depends on, so every arm in this
# study is node-identical BY CONSTRUCTION. That is exactly why it is checked:
# "by construction" has been wrong before in this repo (study 29's first
# `accpin` was supposed to be load-count-neutral and leaked 249 loads/node to
# CSE), and a silent tree change would turn a timing panel into a comparison of
# two different searches.
#
# Checks, per label:
#   1. node count at the study frame  == 5001562
#   2. wl50 eval hash                 == 2e1ede95...79c4  (the NNUE gate pin)
#   3. d13 node count                 == 1739588          (the NNUE gate pin)
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=${P47_NET:-engines/canonical/net.cenn}
FENS=${P47_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P47_BIN:-bin47}
CPU=${P47_CPU:-2}
WANT_NODES=5001562
WANT_HASH=2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4
WANT_D13=1739588
rc=0
for label in "$@"; do
  # `bench` reports on STDERR -- stdout is reserved for the fingerprint, which
  # is exactly why the eval hash below can be piped straight from stdout.
  n=$(taskset -c "$CPU" "./$BIN/$label" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null \
      | sed -n 's/^Nodes searched  : //p' | tail -1)
  d13=$(taskset -c "$CPU" "./$BIN/$label" bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null \
      | sed -n 's/^Nodes searched  : //p' | tail -1)
  # Same spelling `tools/check.sh` uses for its own pin, so the two cannot drift.
  h=$("./$BIN/$label" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)
  s_n=OK; s_d=OK; s_h=OK
  [ "$n" = "$WANT_NODES" ] || { s_n="MISMATCH(want $WANT_NODES)"; rc=1; }
  [ "$d13" = "$WANT_D13" ] || { s_d="MISMATCH(want $WANT_D13)"; rc=1; }
  [ "$h" = "$WANT_HASH" ] || { s_h="MISMATCH"; rc=1; }
  printf '%-10s nodes=%-9s %-28s d13=%-9s %-28s hash=%.16s %s\n' \
    "$label" "$n" "$s_n" "$d13" "$s_d" "$h" "$s_h"
done
exit $rc
