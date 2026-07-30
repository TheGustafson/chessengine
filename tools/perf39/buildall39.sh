#!/usr/bin/env bash
# perf study 39 — the whole arm set, built one at a time so two fat-LTO
# linkers are never resident together (the box has 15 GB and a live trainer).
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf39/build39.sh addcopy sota,arch-d48x,probe-addcopy
bash tools/perf39/build39.sh actpin  sota,arch-d48x,probe-actpin
bash tools/perf39/build39.sh dstpin  sota,arch-d48x,probe-dstpin
bash tools/perf39/build39.sh accpin  sota,arch-d48x,probe-accpin
echo "[buildall39] done"
md5sum bin39/*
