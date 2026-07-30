#!/usr/bin/env bash
# perf study 39 — second build wave: the `livepin` rebuild of actpin plus the
# two `foldcount` census binaries that settle whether the SUBTRACTIVE probes
# changed the workload rather than only its locality.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf39/build39.sh actpin   sota,arch-d48x,probe-actpin
bash tools/perf39/build39.sh fcbase   sota,arch-d48x,foldcount
bash tools/perf39/build39.sh fcactpin sota,arch-d48x,foldcount,probe-actpin
echo "[build2] done"
