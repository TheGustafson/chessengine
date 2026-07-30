#!/usr/bin/env bash
# Recode POC phase-1 speed frame (commissioning brief, 2026-07-30):
# same binary, two eval-identical nets that differ ONLY in threat-table size —
# arm A "scatter" (77.7 MB table + identity TRM1) vs arm B "compress"
# (R'-row table + real TRM1). Fixed NODE budget, alternating arm order,
# cycles:u + instructions:u per run.
#
# Frame validity gates (checked by the report script):
#   - node totals identical across arms and rounds (eval-identical nets);
#   - instructions:u matched to ~0.02% (same work, only cache behaviour free);
#   - then cycles rank separation is the verdict.
#
# Usage: recode_speed_ab.sh <binary> <netA> <netB> <rounds> <outdir> [nodes]
set -euo pipefail
BIN=$1; NETA=$2; NETB=$3; ROUNDS=$4; OUT=$5; NODES=${6:-100000}
FENS=tools/nnue/calib/wl50.fens
mkdir -p "$OUT"

run() { # run <tag> <net> <round>
    local tag=$1 net=$2 r=$3
    perf stat -e cycles:u,instructions:u -x, -o "$OUT/$tag.r$r.perf" \
        "$BIN" bench 16 1 "$NODES" "$FENS" nodes "$net" \
        2> "$OUT/$tag.r$r.bench" >/dev/null
    grep '^Nodes searched' "$OUT/$tag.r$r.bench" | awk '{print $4}' \
        > "$OUT/$tag.r$r.nodes"
}

# Pre-flight: the two nets must be eval-identical (same forward pass on wl50).
ha=$("$BIN" nnue-eval "$NETA" < "$FENS" | sha256sum | cut -d' ' -f1)
hb=$("$BIN" nnue-eval "$NETB" < "$FENS" | sha256sum | cut -d' ' -f1)
echo "evalhash A=$ha" | tee "$OUT/preflight.txt"
echo "evalhash B=$hb" | tee -a "$OUT/preflight.txt"
if [ "$ha" != "$hb" ]; then
    echo "FATAL: arms are not eval-identical — the frame is void" >&2
    exit 1
fi

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then
        run A "$NETA" "$r"; run B "$NETB" "$r"
    else
        run B "$NETB" "$r"; run A "$NETA" "$r"
    fi
    echo "round $r done"
done
echo "runs complete: $OUT"
