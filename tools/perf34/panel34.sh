#!/usr/bin/env bash
# perf study 34 — the PAIRED within-round panel.
#
#   bash tools/perf34/panel34.sh <rounds> <ref-label> <label>...
#
# Every label runs once per round, the order rotating within the round, and
# `med34.py` differences each label against the REFERENCE RUN OF THE SAME ROUND
# before taking a median. Study 29 nearly shipped a -1.67% cycles result at
# n=13 that was position-in-round; rotating the order alone does not fix that,
# because rotation still compares a label's round-r slot against another
# label's round-r' slot. Pairing inside the round does.
#
# `instructions:u` is exact (null 1e-6% — study 23 §2.2) but the GATE is
# cycles (study 20 was -5.99% cycles with instructions UP 0.19%), so both are
# recorded and both are reported.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P34_CPU:-2}
BUDGET=${P34_BUDGET:-100000}
EV=instructions:u,cycles:u,branches:u,ex_ret_brn_misp:u,L1-dcache-loads:u
OUT=${P34_PANEL:-docs/research/perf-artifacts-34/panel.txt}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p docs/research/perf-artifacts-34

for r in $(seq "$ROUNDS"); do
  for label in "$@"; do
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin34/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $3~/^instructions/{i=$1} $3~/^cycles/{c=$1} $3~/^branches/{b=$1}
      $3~/^ex_ret_brn_misp/{m=$1} $3~/^L1-dcache-loads/{ld=$1}
      END{printf "r%-3d %-10s nodes=%-9d i/n=%9.3f cyc/n=%9.3f br/n=%8.3f misp/n=%7.4f loads/n=%9.3f\n",
          r, l, n, i/n, c/n, b/n, m/n, ld/n}
    ' "$tmp/stat" | tee -a "$OUT"
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
