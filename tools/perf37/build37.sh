#!/usr/bin/env bash
# perf study 37 — build one ATTRIBUTE variant of the threat fold.
#
#   bash tools/perf37/build37.sh <base|nocold|noinl|noboth>
#
# Applies the variant with `patch37.py`, hands off to study 34's `build34.sh`
# (which owns the memory discipline: MemoryMax=6G scope, -j2, >=3000 MB gate),
# and ALWAYS restores the shipped source, even on failure — a study that leaves
# a patched tree behind is how the next arm silently measures two changes.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?base|nocold|noinl|noboth}
trap 'python3 tools/perf37/patch37.py base' EXIT
python3 tools/perf37/patch37.py "$LABEL"
bash tools/perf34/build34.sh "$LABEL"
