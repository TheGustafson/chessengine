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
  # DEFECT FIX (perf study 43, 2026-07-30): this was a cyclic rotation,
  # `set -- "${@:2}" "$1"`. Rotation changes only the STARTING POINT --
  # relative order is invariant under it, so every label keeps the same
  # predecessor in every round and any carryover from "which arm ran
  # immediately before" never averages out, at any number of rounds.
  # Measured: a BYTE-IDENTICAL null read -0.5700% cycles, 13/16 rounds
  # negative. Shuffling breaks the adjacency, as panel32/39/43 already do.
  # BLOCKED LATIN SQUARE (perf studies 43 + 44, 2026-07-30). Two defects were
  # found in this reordering, one per study, and they are OPPOSITE:
  #   * cyclic rotation alone (the original) balances POSITION exactly but
  #     freezes ADJACENCY -- every arm keeps the same predecessor forever.
  #     Study 43: a byte-identical null read -0.5700% cycles, 13/16 rounds.
  #   * free `shuf` alone breaks adjacency but leaves POSITION unbalanced --
  #     simulated spread 0.83 slots (p95 1.35) at 5 arms x 20 rounds.
  #     Study 44: a byte-identical null read +2.36% cycles with the
  #     reference seated at mean position 0.77 of 1.50.
  # Rotating WITHIN a block of N and reshuffling the base BETWEEN blocks
  # gets both: position spread 0.000, and 13.7 of 20 distinct adjacency
  # pairs against rotation's 5.0 (simulated, same conditions).
  set -- "${@:2}" "$1"
  if (( r % $# == 0 )); then
    set -- $(printf '%s\n' "$@" | shuf | tr '\n' ' ')
  fi
done
