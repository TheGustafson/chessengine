#!/usr/bin/env bash
# perf study 37 — the PAIRED within-round panel, with a SELECTABLE event set.
#
#   bash tools/perf37/panel37.sh <rounds> <label>...
#   P37_EV=...  P37_PANEL=...  P37_BUDGET=...  P37_CPU=...
#
# Descends from tools/perf34/panel34.sh and keeps its two load-bearing
# properties: every label runs once per round with the order rotating, and
# `med37.py` differences each label against the SAME ROUND's first label before
# medianing (study 29 nearly shipped a -1.67% result that was position-in-round;
# rotation alone does not fix that).
#
# What is new is the event set. Zen 4 has six general counters, so the study's
# TLB panel cannot ride along with study 34's five-event set without
# multiplexing — and a multiplexed count is an extrapolation, which is exactly
# the kind of instrument this project has been burned by. So the event list is
# a parameter and the panels are run separately:
#
#   EV_CORE  instructions,cycles,branches,mispredicts,L1d loads   (study 34's)
#   EV_TLB   page walks + the 4k/2m reload split                  (this study's)
#
# Rows are emitted as `key=value` pairs so `med37.py` needs no per-event regex.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P37_CPU:-2}
BUDGET=${P37_BUDGET:-100000}
EV=${P37_EV:-instructions:u,cycles:u,branches:u,ex_ret_brn_misp:u,L1-dcache-loads:u}
OUT=${P37_PANEL:-docs/research/perf-artifacts-37/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for r in $(seq "$ROUNDS"); do
  for label in "$@"; do
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin34/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      # perf -x, columns: value,unit,event,run,pct[,...]. A "<not counted>"
      # value lands as a non-numeric field, which is dropped rather than
      # silently averaged in as zero.
      $1 ~ /^[0-9]+$/ && $3 != "" { printf "%s=%.6f ", $3, $1 / n }
      END { print "" }
    ' "$tmp/stat" | sed "s|^|r$r $label nodes=$nodes |" | tee -a "$OUT"
  done
  set -- "${@:2}" "$1"
done
