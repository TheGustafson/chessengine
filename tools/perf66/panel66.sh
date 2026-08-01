#!/usr/bin/env bash
# perf study 66 — the multi-mode counter panel.
#
# ONE binary, N data modes selected by environment variable. Because the modes
# share a `.text` byte for byte, every difference the panel reports is
# attributable to the LAYOUT of the weight table alone and the box's load
# cancels — which is the whole reason study 65's negative result was believable
# and is why this study copies its shape.
#
# The order is SHUFFLED inside every round (perf study 43 §6.2: cyclic rotation
# preserves adjacency and gave a byte-identical null a −0.57 % bias).
#
#   bash tools/perf66/panel66.sh <binary> <rounds> [mode ...]
#
# A mode is `label=ENV=VALUE`, e.g. `pack1=ARCHON_PACK=1`. The FIRST mode is the
# reference every other is quoted against. Default modes are the packing arms.
# Rows land in $P66_OUT as  round<TAB>label<TAB>counter<TAB>value.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
ROUNDS=${2:-6}
shift 2 || true
MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
  MODES=(narrow0=ARCHON_NARROW=0 pack0=ARCHON_PACK=0 pack1=ARCHON_PACK=1)
fi
CPU=${P66_CPU:-2}
OUT=${P66_OUT:-/var/tmp/perf66/panel.tsv}
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
REF=${MODES[0]%%=*}

EVENTS=instructions:u,task-clock,L1-dcache-load-misses,l2_cache_req_stat.dc_access_in_l2,l2_cache_req_stat.ic_dc_miss_in_l2,l2_pf_hit_l2.all

for r in $(seq 1 "$ROUNDS"); do
  for spec in $(printf '%s\n' "${MODES[@]}" | shuf); do
    label=${spec%%=*}
    env_var=${spec#*=}
    perf stat -x, -e "$EVENTS" -o /var/tmp/perf66/.stat \
      env "$env_var" taskset -c "$CPU" "$BIN" \
        bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn \
        > /var/tmp/perf66/.bench 2>&1
    nodes=$(awk '/Nodes searched *:/{n=$NF} END{print n}' /var/tmp/perf66/.bench)
    if [ "$nodes" != "5001560" ]; then
      echo "[panel66] NODE IDENTITY BROKEN in $label: $nodes" >&2
      exit 1
    fi
    awk -F, -v r="$r" -v m="$label" 'NF>2 && $1 ~ /^[0-9.]+$/ {print r"\t"m"\t"$3"\t"$1}' \
      /var/tmp/perf66/.stat >> "$OUT"
    printf '.' >&2
  done
done
echo >&2
python3 tools/perf66/med66.py "$OUT" "$REF"
