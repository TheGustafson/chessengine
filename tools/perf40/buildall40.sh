#!/usr/bin/env bash
# perf study 40 — build every labelled binary this study measures, one at a
# time (the box hosts a D49 trainer; two fat-LTO builds at once is how a
# previous study OOMed the machine).
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf34/build34.sh b40 "sota,arch-d48x"
bash tools/perf34/build34.sh w40 "sota,arch-d48x,insertwin34"
bash tools/perf34/build34.sh n38 "sota,arch-d48x,null38"
echo "[buildall40] done"
