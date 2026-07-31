#!/usr/bin/env bash
# perf study 60 — the exact-counter panel, BLOCKED LATIN SQUARE ordering.
#
#   bash tools/perf60/panel60.sh <rounds> <label>...
#
# Inherited verbatim from `tools/perf58/panel58.sh` (55/48/45 before it),
# including both of study 57 §2's guards: `perf stat -o` writes to a PRIVATE
# temp file per run so two jobs cannot interleave records, and an `flock` on
# the output path makes a second panel refuse.
#
# STUDY 60 NOTE — CACHE COUNTERS NEED THEIR OWN NULL. On this box
# `L1-dcache-loads:u` FAILED its null at n=24 (study 58 §6) and PASSED at n=39
# (§7.1); `ls_stlf` fails outright. The default event list therefore always
# carries `instructions:u` (null 0.000) alongside whatever cache counter is
# being licensed, and every panel in this study runs a byte-identical null in
# the rotation. A cache number without its null in the same panel is not
# evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
LABELS=("$@")
NLAB=${#LABELS[@]}
CPU=${P60_CPU:-1}
NET=${P60_NET:-engines/canonical/net.cenn}
FENS=${P60_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P60_BIN:-bin60}
BUDGET=${P60_BUDGET:-100000}
EV=${P60_EV:-instructions:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u}
OUT=${P60_PANEL:-docs/research/perf-artifacts-60/panel.txt}
mkdir -p "$(dirname "$OUT")"
exec 9>"$OUT.lock"
flock -n 9 || { echo "[panel60] REFUSING: another panel owns $OUT" >&2; exit 1; }
: > "$OUT"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
base=("${LABELS[@]}")
for r in $(seq 0 $((ROUNDS - 1))); do
  k=$((r % NLAB))
  if [ "$k" -eq 0 ]; then
    mapfile -t base < <(printf '%s\n' "${LABELS[@]}" | shuf)
  fi
  order=()
  for i in $(seq 0 $((NLAB - 1))); do
    order+=("${base[$(((i + k) % NLAB))]}")
  done
  for pos in $(seq 0 $((NLAB - 1))); do
    lab=${order[$pos]}
    perf stat -x, -e "$EV" -o "$tmp/stat" -- taskset -c "$CPU" "./$BIN/$lab" \
      bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    {
      echo "### round=$r pos=$pos label=$lab nodes=${nodes:-NONE}"
      grep -E '^[0-9]+,' "$tmp/stat"
    } >> "$OUT"
  done
done
echo "[panel60] -> $OUT"
