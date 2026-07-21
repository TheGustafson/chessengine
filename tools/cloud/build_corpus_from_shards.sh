#!/usr/bin/env bash
# Incremental corpus builder for the SF18-relabel fleet: pull the .ceobs shards
# that have landed in GCS, export their SF18 value labels, MERGE the original
# binpack WDL back (relabel is eval-only -> WDL=0.5; we restore the real outcome
# by FEN join), and bf-convert to a bullet .bin ready to train — so we can train
# on the first millions while the fleet keeps generating. Resumable: skips shards
# already pulled. Safe to re-run as more shards complete.
#
#   build_corpus_from_shards.sh <gcs_shards_prefix> <orig_fen_wdl_score_tsv> <workdir>
#
# <orig...tsv> is the bulk_fens output (FEN\twdl\tscore) for the SAME positions —
# the source of the real WDL. Outputs <workdir>/corpus.txt (FEN|SF18cp|origWDL)
# and <workdir>/corpus.bin.
set -uo pipefail
cd "$(dirname "$0")/../.."
PREFIX="${1:?gs://.../shards prefix}"; ORIG="${2:?orig FEN\twdl\tscore tsv}"; WORK="${3:?workdir}"
OBS=target/release/obs-tool
CONV=tools/nnue/bf-convert/target/release/nnue_convert
mkdir -p "$WORK/ceobs"
say(){ echo "[corpus $(date -u +%H:%M:%S)] $*"; }

# Parallel incremental pull: gcloud storage rsync only fetches shards not already
# local and runs many transfers at once (serial `cp` was ~4.6s/shard -> hours for
# thousands). Within one job the .ceobs basenames are unique, so a flat mirror is
# collision-free. Rebuild only when the shard count actually grew.
say "rsync shards from $PREFIX/shards/"
gcloud storage rsync "$PREFIX/shards/" "$WORK/ceobs/" >/dev/null 2>&1
have=$(ls "$WORK/ceobs"/*.ceobs 2>/dev/null | wc -l)
say "have $have .ceobs shards locally"
[ "$have" -eq 0 ] && { say "no shards landed yet"; exit 0; }
last=$(cat "$WORK/.last_count" 2>/dev/null || echo 0)
if [ "$have" -eq "$last" ] && [ -s "$WORK/corpus.bin" ]; then
    say "no new shards ($have); corpus.bin current"; exit 0
fi

# Export value labels from ALL pulled shards (one export handles many shards).
# 2026-07-21: export-value now applies the trainer-parity quiet filter BY
# DEFAULT (the unfiltered export from this very script fed the -582 corpus).
# The explicit flag documents intent; the summary line shows the drop split —
# expect ~26-30% and investigate outside that band (data-recipe-contract.md).
say "export-value over $(ls "$WORK/ceobs"/*.ceobs 2>/dev/null | wc -l) shards (trainer-parity filter)"
"$OBS" export-value "$WORK/values.raw" "$WORK"/ceobs/*.ceobs --filter-sfbinpack 2>&1 | tail -2

# Merge the real WDL by FEN — memory-safe disk sort+join. A Python FEN->wdl
# dict would need ~15GB for the full 188M corpus; this box has ~7GB. TAB is the
# join delimiter so the whole space-containing FEN is a single key field. orig
# is deduped on FEN (-u -k1,1: one wdl per position); the sorted orig is cached.
say "merging original WDL by FEN (disk sort+join)"
ORIGSORT="$WORK/orig.sorted"
if [ ! -s "$ORIGSORT" ] || [ "$ORIG" -nt "$ORIGSORT" ]; then
    say "  sorting orig (cached after first run)"
    LC_ALL=C sort -t$'\t' -k1,1 -u -S 512M --parallel=4 -o "$ORIGSORT" <(cut -f1,2 "$ORIG")
fi
# values.raw rows are "FEN | cp | wdl"; keep FEN\tcp, sort on the FEN field.
awk -F' \\| ' '{print $1"\t"$2}' "$WORK/values.raw" \
  | LC_ALL=C sort -t$'\t' -k1,1 -S 512M --parallel=4 > "$WORK/values.sorted"
LC_ALL=C join -t$'\t' -1 1 -2 1 "$WORK/values.sorted" "$ORIGSORT" \
  | awk -F'\t' '{print $1" | "$2" | "$3}' > "$WORK/corpus.txt"
merged=$(wc -l < "$WORK/corpus.txt"); vals=$(wc -l < "$WORK/values.sorted")
say "  merged $merged rows (of $vals exported; unmatched dropped)"

rows=$(wc -l < "$WORK/corpus.txt")
say "bf-convert $rows rows -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -1
echo "$have" > "$WORK/.last_count"
say "DONE: $WORK/corpus.bin ($rows positions). Train on it now; re-run as more shards land."
