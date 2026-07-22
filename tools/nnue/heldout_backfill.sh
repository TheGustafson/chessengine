#!/usr/bin/env bash
# Backfill held-out curves over a run's saved quant checkpoints (retroactive —
# SAVE_RATE=1 runs have every sb on disk). Emits TSV to stdout and appends to
# the run's heldout.tsv. GPU user — run when the GPU is free (post-run) or
# accept trainer contention consciously.
#   bash tools/nnue/heldout_backfill.sh <ckpt_dir> <prefix> <pol 0|1> [stride]
# e.g. bash tools/nnue/heldout_backfill.sh \
#        data/nnue/experiments/d47c-1280-240sb/ckpt d47c-1280-240sb 1 10
set -euo pipefail
cd "$(dirname "$0")/../.."
DIR="${1:?ckpt dir}"
PREFIX="${2:?ckpt prefix}"
POL="${3:?0|1}"
STRIDE="${4:-10}"
TSV="$DIR/../heldout.tsv"
for Q in $(ls "$DIR"/"$PREFIX"-sb*-quant.bin | sort -V); do
  SB=$(basename "$Q" | grep -oE 'sb[0-9]+' | grep -oE '[0-9]+')
  [ $((SB % STRIDE)) -eq 0 ] || continue
  LINE=$(bash tools/nnue/heldout_eval.sh "$Q" quant "$POL" | tail -1)
  echo -e "$SB\t$LINE" | tee -a "$TSV"
done
echo "backfill complete -> $TSV"
