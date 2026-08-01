#!/usr/bin/env bash
# perf study 67 — per-symbol profile of ONE binary on the study frame.
#   bash tools/perf67/prof67.sh <binary> <event> <outfile> [period]
# Per-SYMBOL shares of instructions:u are valid on this box (study 58 §8);
# per-INSTRUCTION shares of the same counter are NOT (study 36 §2.3) — use IBS.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}; EV=${2:?event}; OUT=${3:?out}; PER=${4:-2000003}
CPU=${P67_CPU:-2}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
taskset -c "$CPU" perf record -q -e "$EV" -c "$PER" -o "$tmp/p.data" -- \
  "$BIN" bench 16 1 100000 "$FENS" nodes "$NET" >/dev/null 2>&1
perf report -i "$tmp/p.data" --stdio --no-children -q --percent-limit 0.15 2>/dev/null > "$OUT"
echo "[prof67] $EV -> $OUT"
head -30 "$OUT"
