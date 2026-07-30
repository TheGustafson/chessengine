#!/usr/bin/env bash
# perf study 39 — third build wave: the `P39_SRC=live` addcopy and the
# `thr_refresh` foldcount census.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf39/build39.sh addcopy sota,arch-d48x,probe-addcopy
cp bin39/addcopy bin39/addcopy-null
bash tools/perf39/build39.sh fcbase sota,arch-d48x,foldcount
echo "[build3] done"
