#!/usr/bin/env bash
# perf study 35 -- the cycles panel with the SMT SIBLING PINNED BUSY.
#
#   bash tools/perf35/panelsmt35.sh <rounds> <label>...
#
# WHY. `panel35.sh` measured a +/-20.3% round-to-round spread on `cycles:u` for
# the CONTROL and a +1.585% / +1.650% paired median for two BYTE-IDENTICAL
# binaries -- against an arm whose mechanism predicts ~0.4%. Two things were
# wrong with it, and both are fixed here:
#
#  1. THE SMT SIBLING. The measured run is pinned to one CPU, but nothing was
#     pinned to that CPU's hyperthread twin, so every run rolled a die: share
#     the core with a trainer thread (IPC down ~35%) or have it to itself.
#     That is bimodal noise no amount of averaging removes quickly. Here a
#     spinner -- the SAME binary running the SAME bench, so the contention has a
#     representative instruction mix -- is pinned to the twin for the WHOLE
#     panel. Absolute cycles/node come out inflated; the ARM-TO-ARM RATIO,
#     which is the only thing this study reads, comes out clean. An occupied
#     core pair also repels the kernel's load balancer, which helps twice.
#
#  2. `nice -n 19`. Studies 29/31/33 measured at the LOWEST priority, to be
#     polite to the trainer. That makes the measurement maximally preemptible
#     by exactly the noise it is trying to see through. The measured run is
#     nice 0 here; the spinner stays at 19, so the pair still yields to the
#     trainer as a whole.
#
# The spinner is killed by PID from a trap -- never `pkill -f`, which matches
# the shell that typed the pattern (CLAUDE.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P35_CPU:-2}
SIB=${P35_SIB:-10}
BUDGET=${P35_BUDGET:-100000}
EV=instructions:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,ex_no_retire.load_not_complete:u
OUT=${P35_PANEL:-docs/research/perf-artifacts-35/panel-smt.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d)

SPIN=""
cleanup() {
  [ -n "$SPIN" ] && kill "$SPIN" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

# The sibling load: one very long bench, restarted if it ever finishes.
nice -n 19 taskset -c "$SIB" bash -c \
  "while :; do ./bin35/base bench 16 1 100000000 $FENS nodes $NET >/dev/null 2>&1; done" &
SPIN=$!
sleep 3   # let it get going so round 1 sees the same core state as round 16

for r in $(seq "$ROUNDS"); do
  for label in "$@"; do
    perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin35/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $3~/^instructions/{i=$1} $3~/^cycles/{c=$1}
      $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
      $3~/^ex_no_retire.load_not_complete/{st=$1}
      END{printf "r%-3d %-12s nodes=%-9d i/n=%8.2f cyc/n=%8.2f loads/n=%9.3f miss/n=%8.3f rate=%.3f%% stall/n=%8.2f\n",
          r, l, n, i/n, c/n, ld/n, ms/n, 100*ms/ld, st/n}
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
