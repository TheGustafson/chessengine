#!/usr/bin/env bash
# perf study 38 — is the threat channel's 2560-byte `memcpy` a `rep movsb`?
#
#   bash tools/perf38/repmovsb38.sh <rounds> <label>
#
# The question matters because the answer decides which counter can see the
# copy at all: a `rep movsb` retires as **ONE instruction** however many bytes
# it moves, so `instructions:u` — the counter this study run trusts because its
# null is exactly 0.000% — would be nearly blind to it, while the load counter
# still sees every microcoded load.
#
# READ THE DEFAULT BEFORE ASSUMING IT. On this box
# `/lib64/ld-linux-x86-64.so.2 --list-tunables` reports
# `glibc.cpu.x86_rep_movsb_threshold: 0x1800000` — **24 MiB**, not the 2 KiB
# the folklore says — so a 2560-byte copy never went near `rep movsb` and the
# default path is already the vectorised one. This study's first run of this
# script raised the threshold and measured a null, which was inert by
# construction and is recorded as such in the write-up.
#
# So the arm goes the OTHER way: force `rep movsb` ON by dropping the threshold
# below the copy size. That both settles the mechanism and prices the
# vectorised path against the microcoded one, with no rebuild and no code
# change — the SAME binary, one environment variable.
#
#   dflt   default tunables (24 MiB threshold) -> vectorised `more_8x_vec` loop
#   rep    threshold 512 B                     -> `rep movsb`
#
# Rows alternate within a round so the pair is drift-matched, as in `panel37.sh`.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}
LABEL=${2:-base}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P38_CPU:-2}
BUDGET=${P38_BUDGET:-100000}
EV=${P38_EV:-instructions:u,branches:u,L1-dcache-loads:u}
OUT=${P38_PANEL:-docs/research/perf-artifacts-38/panel-repmovsb.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

run() { # $1 = arm name, $2.. = env assignments
  local arm=$1; shift
  env "$@" nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
    taskset -c "$CPU" "./bin34/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  local nodes
  nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
  awk -F, -v l="$arm" -v r="$R" -v n="$nodes" '
    $1 ~ /^[0-9]+$/ && $3 != "" { printf "%s=%.6f ", $3, $1 / n }
    END { print "" }
  ' "$tmp/stat" | sed "s|^|r$R $arm nodes=$nodes |" | tee -a "$OUT"
}

for R in $(seq "$ROUNDS"); do
  if [ $((R % 2)) -eq 1 ]; then
    run dflt GLIBC_TUNABLES=
    run rep  GLIBC_TUNABLES=glibc.cpu.x86_rep_movsb_threshold=512
  else
    run rep  GLIBC_TUNABLES=glibc.cpu.x86_rep_movsb_threshold=512
    run dflt GLIBC_TUNABLES=
  fi
done
