#!/usr/bin/env bash
# perf study 8 — build ONE arm as a PGO binary and run the full gate + the
# deterministic instruction measurement, in that order.
#
#   bash arm8.sh <name>
#
# Prints, and appends to docs/research/perf-artifacts-8/arms.txt:
#   * the wl50 eval hash          (must equal the D48X pin for a bit-exact arm)
#   * the wl50 d13 node total     (must equal 1972607 for a node-identical arm)
#   * instructions:u at a fixed 5,001,xxx-node tree (bench 16 1 100000 ... nodes)
#
# instructions:u is the load-immune instrument: it reproduces to ~0.00001% on
# an identical tree, so a sub-1% arm is still resolvable on a busy box.
# Wall-clock is recorded but is a band, never a claim (studies 3/5/7).
set -euo pipefail
NAME=${1:?arm name}
FEAT=sota,arch-d48x
NET=engines/canonical/net.cenn
OUT=docs/research/perf-artifacts-8/arms.txt
EVAL_PIN=d82caf8aa4e3cca8ea9be5dc9918e192733d14b76f0b2006c0da02b36e9f8163
NODE_PIN=1972607

free -g | awk 'NR==2 {if ($7 < 3) {print "REFUSING: only " $7 " GB available (15 GB box, a PGO build beside a trainer OOMed it on 2026-07-27)"; exit 1}}'

# Cap the build's parallelism: 15 GB is the binding resource on this box and an
# unbounded `cargo build -j` beside a running match OOMed it on 2026-07-27.
CARGO_BUILD_JOBS=4 bash tools/build_pgo.sh "$FEAT" "archon-s8-$NAME" "$NET" > "logs/build-$NAME.log" 2>&1
BIN=engines/bin/archon-s8-$NAME

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
