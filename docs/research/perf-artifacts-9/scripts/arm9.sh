#!/usr/bin/env bash
# perf study 9 — build ONE arm as a PGO binary, run both identity pins, and
# take the deterministic instruction measurement. Study 8's arm8.sh with the
# study-9 pins and output path.
#
#   bash arm9.sh <name>
#
# Pins are study 8's post-§4.6 values; an arm that intentionally moves one
# says so in its commit and updates tools/check.sh in the same commit.
set -euo pipefail
NAME=${1:?arm name}
FEAT=sota,arch-d48x
NET=engines/canonical/net.cenn
OUT=docs/research/perf-artifacts-9/arms.txt
EVAL_PIN=5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38
NODE_PIN=1922947

free -g | awk 'NR==2 {if ($7 < 3) {print "REFUSING: only " $7 " GB available (15 GB box)"; exit 1}}'

CARGO_BUILD_JOBS=4 bash tools/build_pgo.sh "$FEAT" "archon-s9-$NAME" "$NET" > "logs/build-$NAME.log" 2>&1
BIN=engines/bin/archon-s9-$NAME

hash=$("$BIN" nnue-eval "$NET" < tools/nnue/calib/wl50.fens | sha256sum | cut -d' ' -f1)
nodes=$(taskset -c 4 "$BIN" bench 16 1 13 tools/nnue/calib/wl50.fens depth "$NET" 2>&1 >/dev/null \
        | grep -oP 'Nodes searched\s*:\s*\K[0-9]+' | tail -1)

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,cycles:u,branches:u -o "$tmp.perf" -- \
  taskset -c 4 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes "$NET" > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
bnodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
nps=$(grep -oP 'Nodes/second\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

evalok=OK; [ "$hash" = "$EVAL_PIN" ] || evalok="**CHANGED**"
nodeok=OK; [ "$nodes" = "$NODE_PIN" ] || nodeok="**CHANGED**"

line="arm=$NAME eval_hash=${hash:0:16} eval=$evalok d13_nodes=$nodes node=$nodeok bench_nodes=$bnodes instructions=$ins cycles=$cyc branches=$brs nps=$nps"
echo "$line" | tee -a "$OUT"
