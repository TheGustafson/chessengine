#!/usr/bin/env bash
# Deep-label the hard-negative fens with SF18 @ 50k nodes, in parallel, RESUMABLE.
# Runs on the CPU while the base net trains on the GPU. root-analysis is single-
# threaded per process, so we split the fens into many small chunks and run P at
# a time; each chunk writes <chunk>.ceobs + <chunk>.done, and a chunk with a
# .done marker is skipped (survives process boundaries — just re-run to resume).
#
#   deeplabel_hardneg.sh [parallelism] [nodes] [nchunks]
set -uo pipefail
cd "$(dirname "$0")/../.."
P="${1:-8}"; NODES="${2:-50000}"; NCHUNKS="${3:-64}"
# TSV/WORK env-overridable so the same resumable labeler serves the Reckless-vs-
# ARCHON hard-negs (2026-07-20) as well as the original SF18-vs-ARCHON set.
TSV="${TSV:-data/nnue/hardneg/all.tsv}"
WORK="${WORK:-data/nnue/hardneg/labeled}"
SF18=engines/bin/stockfish18
RA=target/release/root-analysis
mkdir -p "$WORK/chunks"
say(){ echo "[deeplabel $(date -u +%H:%M:%S)] $*"; }

# fens (dedup already done upstream by pgn_extract_all)
[ -s "$WORK/all.fens" ] || cut -f1 "$TSV" > "$WORK/all.fens"
total=$(wc -l < "$WORK/all.fens"); say "total $total fens, $NCHUNKS chunks, P=$P, nodes=$NODES"
# split into NCHUNKS (only once)
if ! ls "$WORK/chunks"/fen_* >/dev/null 2>&1; then
  split -n l/$NCHUNKS -d -a 3 "$WORK/all.fens" "$WORK/chunks/fen_"
fi

label_one() {
  local f="$1" nodes="$2" ra="$3" sf="$4"
  local out="${f/fen_/ceobs_}.ceobs" done="${f/fen_/done_}.done"
  [ -f "$done" ] && return 0
  "$ra" --engine "$sf" --fens "$f" --nodes "$nodes" --reference-only --out "$out" >/dev/null 2>&1 \
    && touch "$done"
}
export -f label_one

ls "$WORK/chunks"/fen_* | xargs -P "$P" -I{} bash -c 'label_one "$@"' _ {} "$NODES" "$RA" "$SF18"

ndone=$(ls "$WORK/chunks"/done_*.done 2>/dev/null | wc -l)
say "chunks done: $ndone/$NCHUNKS"
if [ "$ndone" -eq "$NCHUNKS" ]; then touch "$WORK/DONE"; say "ALL DEEP-LABELING DONE"; fi
