#!/usr/bin/env bash
# Label-value pilot (2026-07-18, user-greenlit): does deeper (SF18 @100k) vs
# cheap (SF18 @5k) value labels on the SAME 2M training-distribution positions
# produce a stronger net? Chunked + RESUMABLE deep relabel (an 8h write-at-end
# job losing everything on a crash is unacceptable): each 200k chunk relabels
# independently and marks .done; rerun this script to resume. Then the survivor
# FENs are relabeled cheap (line-aligned), calibrated, and built into two
# train.bins that differ ONLY in the cp column. Trains + A/B are launched
# separately (GPU-gated, after the warm-continue frees the accelerator).
set -uo pipefail
cd "$(dirname "$0")/../.."
SF=engines/stockfish-18-src/src/stockfish
EXP=data/nnue/experiments/label-depth
W=${1:-10}           # SF18 workers (10 = RAM-safe beside a bullet train)
DEEP=${2:-100000}    # deep node budget
CONV=tools/nnue/bf-convert/target/release/nnue_convert
say() { echo "[labelval $(date +%H:%M:%S)] $*"; }

# --- phase 1: chunked deep relabel (resumable) ---
for ch in "$EXP"/chunks/chunk_*.txt; do
  base=$(basename "$ch" .txt)
  out="$EXP/chunks/${base}_deep.txt"
  if [ -f "$EXP/chunks/${base}.done" ]; then say "skip $base (done)"; continue; fi
  say "deep relabel $base @ ${DEEP}n, $W workers"
  if uv run python tools/nnue/relabel_depth.py --input "$ch" --sf "$SF" \
       --nodes "$DEEP" --curate --workers "$W" --out "$out"; then
    touch "$EXP/chunks/${base}.done"
  else
    say "FAILED $base — rerun to resume"; exit 1
  fi
done

# --- phase 2: concat deep + survivor FENs ---
say "concat deep chunks"
mkdir -p "$EXP/deep_raw" "$EXP/cheap_raw"
cat "$EXP"/chunks/chunk_*_deep.txt > "$EXP/deep_raw/shard_000.txt"
cut -d'|' -f1 "$EXP/deep_raw/shard_000.txt" | sed 's/ *$//' > "$EXP/survivors.txt"
say "survivors: $(wc -l < "$EXP/survivors.txt")"

# --- phase 3: cheap relabel on the SAME survivor FENs (line-aligned) ---
say "cheap relabel survivors @ 5000n"
uv run python tools/nnue/relabel_depth.py --input "$EXP/survivors.txt" --sf "$SF" \
  --nodes 5000 --workers "$W" --out "$EXP/cheap_raw/shard_000.txt"

# --- phase 4: full-scale calibration (free corroborator) ---
say "calibration"
uv run python tools/nnue/paired_label_calibration.py \
  "$EXP/cheap_raw/shard_000.txt" "$EXP/deep_raw/shard_000.txt" | tee "$EXP/full_calibration.txt"

# --- phase 5: build two train.bins (converter set; shuffle OFF => identical order) ---
say "build corpora"
for ARM in deep cheap; do
  BULLETFORMAT_CONVERT="$CONV convert" \
    bash tools/nnue/build_corpus.sh --raw "$EXP/${ARM}_raw" --out "$EXP/${ARM}" --require-convert --val-pct 1 \
    || { say "corpus build FAILED ($ARM)"; exit 1; }
done
say "PILOT RELABEL+CORPORA DONE — ready for trains (GPU): $EXP/{deep,cheap}/train.bin"
