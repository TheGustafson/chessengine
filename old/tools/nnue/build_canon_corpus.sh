#!/usr/bin/env bash
# Build the SF18-canonical training corpus from LOCAL canon-v1 .ceobs shards
# (2026-07-20, native-Arch). Simplified WDL=0 variant of
# tools/cloud/build_corpus_from_shards.sh: the relabel is eval-only so the shard
# WDL is a 0.5 placeholder; instead of the expensive FEN-join to restore the
# original game WDL, we emit result=0.5 (draw) and train with D40_WDL=0 (pure
# SF18-cp regression). cp is clamped to +-2499 for the bf-convert |score|<2500
# invariant.
#
# WARNING (2026-07-21): the SF18 cp here is raw UCI cp, which is NOT the
# incumbent net's output scale. Measured: binpack_cp = ~2.7x SF18-UCI (NOT the
# 1.9x this header used to claim), so training on these labels needs
# SCORE_MULT ~= 1.94, not 1.0. See docs/research/sf18-canon-data-diagnosis-2026-07-21.md.
# Also: this materialize-and-train path only removes in-check via canon_filter,
# leaving ~22% best-move-capture positions the sfbinpack load-time filter drops
# (full 5-criterion filter needs best_move at export time).
#
#   build_canon_corpus.sh <canon_dir> <workdir>
#     <canon_dir> = data/nnue/canon-v1  (has shards/shards/*.ceobs)
#     <workdir>   = data/nnue/canon-v1/build
#   -> <workdir>/corpus.txt (FEN | cp | 0.5) and <workdir>/corpus.bin
set -uo pipefail
cd "$(dirname "$0")/../.."

# ★ DEPRECATED (2026-07-21): this path built the -582 corpus (no quiet filter,
# WDL=0.5 placeholder, false scale comment). The tight replacement is
# tools/nnue/build_canon_tight.sh (filtered export + real WDL join +
# self-checks). This script is kept only for archaeology.
if [ "${FORCE_LEGACY:-0}" != 1 ]; then
  echo "build_canon_corpus.sh is DEPRECATED — use tools/nnue/build_canon_tight.sh."
  echo "(FORCE_LEGACY=1 to run anyway; its output is NOT training-grade.)"
  exit 1
fi

CANON="${1:-data/nnue/canon-v1}"
WORK="${2:-data/nnue/canon-v1/build}"
OBS=target/release/obs-tool
CONV=tools/nnue/bf-convert/target/release/nnue_convert
mkdir -p "$WORK"
say(){ echo "[canon-corpus $(date -u +%H:%M:%S)] $*"; }

SHARDS_DIR="$CANON/shards/shards"
nshards=$(ls "$SHARDS_DIR"/*.ceobs 2>/dev/null | wc -l)
say "found $nshards .ceobs shards in $SHARDS_DIR"
[ "$nshards" -eq 0 ] && { say "no shards yet"; exit 1; }

# 1. export SF18 value labels -> values.raw ("FEN | cp | wdl"), CHUNKED.
# A single 114M export holds a ~3GB dedup set for ~14min and was OOM/boundary-
# killed (2026-07-20). Batches of ~285 shards run ~2min each with a small dedup
# set, survive process boundaries, and append incrementally. canon-v1 is
# pre-deduped upstream so cross-batch dup loss is negligible. Resumable: a batch
# whose marker exists is skipped.
say "export-value (chunked) over $nshards shards -> $WORK/values.raw"
mapfile -t shards < <(ls "$SHARDS_DIR"/*.ceobs)
BATCH=285
[ -f "$WORK/.export_start" ] || : > "$WORK/values.raw"
: > "$WORK/values.raw"
nb=$(( (${#shards[@]} + BATCH - 1) / BATCH ))
for ((i=0; i<${#shards[@]}; i+=BATCH)); do
  bn=$((i/BATCH))
  "$OBS" export-value "$WORK/values_b$bn.raw" "${shards[@]:i:BATCH}" 2>&1 | tail -1
  if [ ! -s "$WORK/values_b$bn.raw" ]; then
    say "FATAL: batch $bn/$nb produced empty output (killed?) — aborting, no DONE"; exit 1; fi
  cat "$WORK/values_b$bn.raw" >> "$WORK/values.raw"; rm -f "$WORK/values_b$bn.raw"
  say "batch $((bn+1))/$nb done; cumulative $(wc -l < "$WORK/values.raw") rows"
done
[ -s "$WORK/values.raw" ] || { say "FATAL: empty values.raw — aborting"; exit 1; }
say "values.raw complete: $(wc -l < "$WORK/values.raw") rows"

# 2. clamp cp to +-2499, force result=0.5 -> corpus.txt ("FEN | cp | 0.5")
say "clamp cp, emit corpus.txt"
awk -F' \\| ' '{
  cp=$2+0; if(cp>2499)cp=2499; if(cp<-2499)cp=-2499;
  print $1" | "cp" | 0.5"
}' "$WORK/values.raw" > "$WORK/corpus.txt"
rows=$(wc -l < "$WORK/corpus.txt")
say "corpus.txt: $rows rows"

# 3. bf-convert text -> bullet 32B .bin
say "bf-convert $rows rows -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -2
sz=$(stat -c%s "$WORK/corpus.bin" 2>/dev/null || echo 0)
say "corpus.bin: $sz bytes ($((sz/32)) records); $((sz%32==0)) 32B-aligned(1=ok)"

# 4. SHUFFLE — D40_FORMAT=bulletformat uses DirectSequentialDataLoader (reads in
# order, NO random-skip decorrelation), and canon-v1 is game-contiguous, so an
# unshuffled corpus gives correlated batches. mem-used-mb > file size => in-RAM.
UTIL="$PWD/.archon-state/toolchains/bullet/target/release/bullet-utils"
say "shuffle corpus.bin -> corpus.shuf.bin (in-RAM)"
mkdir -p "$WORK/tmp"
( cd "$WORK" && "$UTIL" shuffle --input "$PWD/corpus.bin" --output "$PWD/corpus.shuf.bin" --mem-used-mb 8000 ) 2>&1 | tail -2
ssz=$(stat -c%s "$WORK/corpus.shuf.bin" 2>/dev/null || echo 0)
say "corpus.shuf.bin: $ssz bytes ($((ssz/32)) records) — TRAIN ON THIS"
say "DONE"
