#!/usr/bin/env bash
# perf study 56 -- build a batch of labelled arms, skipping any already built.
# Each line of the batch file is "<label> [feature,list]".
#   bash tools/perf56/batch56.sh tools/perf56/batches/<name>.txt
set -euo pipefail
cd "$(dirname "$0")/../.."
BATCH=${1:?batch file}
while read -r label feats; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue ;; esac
  if [ -f "bin56/$label" ] && [ "${P56_FORCE:-0}" != 1 ]; then
    md5=$(objdump -d --section=.text "bin56/$label" | grep -v 'file format' | md5sum | cut -c1-8)
    echo "[batch56] skip $label (exists, .text md5 $md5)"
    continue
  fi
  bash tools/perf56/build56.sh "$label" "${feats:-}" 2>&1 | grep -E '^\[build56\] ->'
done < "$BATCH"
