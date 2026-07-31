#!/usr/bin/env bash
# perf study 57 — the exact-counter panel, BLOCKED LATIN SQUARE ordering.
#
#   bash tools/perf57/panel57.sh <rounds> <label>...
#
# Descends from `tools/perf55/exact55.sh` (48/45 before that). Two ordering
# defects are designed out and both were paid for by earlier studies:
#   * a cyclic rotation balances POSITION but freezes ADJACENCY — study 43's
#     byte-identical null read −0.5700% cycles at 13/16 rounds;
#   * a free `shuf` breaks adjacency but leaves POSITION unbalanced — study
#     44's byte-identical null read +2.363% cycles with the reference seated
#     at mean position 0.77 of 1.50.
# Rotating WITHIN a block of NLAB and reshuffling the base BETWEEN blocks gets
# both.
#
# ⚠ THE INTERLEAVING DEFECT, AND WHY THE OUTPUT FILE IS NOW WRITE-ONCE-PER-RUN
#
# Study 55's shape appends BOTH the round marker and `perf stat`'s CSV to one
# accumulating file, with the engine's own stderr going to the same file.
# That is correct for one job and silently catastrophic for two: a stray panel
# left running from an earlier launch appends to the same file, the parser
# assigns its records to the other job's markers, and every contrast is
# scrambled 50/50 — which reads, seductively, as "the counter is bimodal".
# That is exactly what happened here on 2026-07-31 and it cost this study an
# hour and a retracted finding (§2).
#
# Two guards, both cheap:
#   1. `perf stat -o` writes to a PRIVATE temp file per run; only parsed
#      numbers reach the panel file, so two jobs cannot interleave records.
#   2. a lock file: a second panel writing the same output path refuses.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
LABELS=("$@")
NLAB=${#LABELS[@]}
CPU=${P57_CPU:-1}
NET=${P57_NET:-engines/canonical/net.cenn}
FENS=${P57_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P57_BIN:-bin57}
BUDGET=${P57_BUDGET:-100000}
EV=${P57_EV:-instructions:u,cycles:u,branches:u,L1-dcache-loads:u,ls_dispatch.ld_dispatch:u}
OUT=${P57_PANEL:-docs/research/perf-artifacts-57/panel.txt}
mkdir -p "$(dirname "$OUT")"
exec 9>"$OUT.lock"
flock -n 9 || { echo "[panel57] REFUSING: another panel owns $OUT" >&2; exit 1; }
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
echo "[panel57] -> $OUT"
