#!/usr/bin/env bash
# perf study 39 — ONE panel carrying the additive AND the subtractive probe,
# which is exactly what study 29's ranked item #5 asked for and neither study
# did.
#
#   bash tools/perf39/panel39.sh <rounds> <spec>...
#
# A spec is  label=binary  or  label=binary:VAR=VAL,VAR=VAL
# so the four ADDITIVE arms are the SAME BYTES driven by `P39_DOSE`/`P39_RING`
# (see `sota_simd::probe::addcopy`). That removes code layout as a confounder
# from every comparison inside the additive family — study 29 §8.6 needed two
# separate nulls precisely because its arms were different binaries.
#
# FRAME (the coordinator's 2026-07-30 recipe, and studies 29/31's):
#   * fixed NODE budget, never fixed depth — fixed depth lets the trees differ,
#     so "per node" silently compares different nodes;
#   * order RANDOMISED inside each round, not merely rotated: study 25 §0.2
#     measured that a fixed-order round-robin cancels monotone drift but not
#     curvature (its fixed-order null came back +2.06%, its randomised one
#     −1.93%);
#   * `ex_no_retire.load_not_complete` is the currency — a COUNT, not a timing,
#     so it tolerates box contention far better than `cycles:u`, which is
#     quoted only next to its null.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P39_CPU:-3}
BUDGET=${P39_BUDGET:-100000}
EV=instructions:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,ex_no_retire.load_not_complete:u
OUT=${P39_PANEL:-docs/research/perf-artifacts-39/panel.txt}
mkdir -p docs/research/perf-artifacts-39
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

SPECS=("$@")
echo "# panel39 $(date -Is) cpu=$CPU budget=$BUDGET arms=${SPECS[*]}" >> "$OUT"

for r in $(seq "$ROUNDS"); do
  for idx in $(seq 0 $(( ${#SPECS[@]} - 1 )) | shuf); do
    spec=${SPECS[$idx]}
    label=${spec%%=*}
    rest=${spec#*=}
    bin=${rest%%:*}
    envs=""
    if [ "$rest" != "$bin" ]; then envs=$(printf '%s' "${rest#*:}" | tr ',' ' '); fi
    # shellcheck disable=SC2086
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      env $envs taskset -c "$CPU" "./bin39/$bin" \
      bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>"$tmp/bench"
    nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $3~/^instructions/{i=$1} $3~/^cycles/{c=$1}
      $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
      $3~/^ex_no_retire.load_not_complete/{st=$1}
      END{printf "%d %s %d %.4f %.4f %.4f %.4f %.4f\n", r, l, n, i/n, c/n, ld/n, ms/n, st/n}
    ' "$tmp/stat" | tee -a "$OUT"
  done
  printf 'panel39: round %s/%s\n' "$r" "$ROUNDS" >&2
done
