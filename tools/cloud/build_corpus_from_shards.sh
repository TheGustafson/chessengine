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
PULLED="$WORK/pulled.list"; touch "$PULLED"
say(){ echo "[corpus $(date -u +%H:%M:%S)] $*"; }

say "listing shards under $PREFIX"
mapfile -t remote < <(gcloud storage ls -r "$PREFIX/" 2>/dev/null | grep "\.ceobs$")
new=0
for uri in "${remote[@]}"; do
    base=$(basename "$uri")
    grep -qxF "$base" "$PULLED" && continue
    if gcloud storage cp "$uri" "$WORK/ceobs/$base" >/dev/null 2>&1; then
        echo "$base" >> "$PULLED"; new=$((new+1))
    fi
done
say "pulled $new new shards ($(wc -l < "$PULLED") total)"
[ "$new" -eq 0 ] && [ -f "$WORK/corpus.bin" ] && { say "nothing new; corpus.bin current"; exit 0; }

# Export value labels from ALL pulled shards (one export handles many shards).
say "export-value over $(ls "$WORK/ceobs"/*.ceobs 2>/dev/null | wc -l) shards"
"$OBS" export-value "$WORK/values.raw" "$WORK"/ceobs/*.ceobs 2>&1 | tail -1

# Merge the real WDL by FEN (relabel is eval-only). Load orig FEN->WDL once.
say "merging original WDL by FEN"
python3 - "$ORIG" "$WORK/values.raw" "$WORK/corpus.txt" <<'PY'
import sys
orig={}
with open(sys.argv[1]) as f:
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)>=2: orig[p[0]]=p[1]
n=miss=0
with open(sys.argv[2]) as fin, open(sys.argv[3],"w") as out:
    for line in fin:
        parts=[x.strip() for x in line.rstrip("\n").split("|")]
        if len(parts)<3: continue
        fen,cp=parts[0],parts[1]
        wdl=orig.get(fen)
        if wdl is None: miss+=1; continue
        out.write(f"{fen} | {cp} | {wdl}\n"); n+=1
print(f"  merged {n} rows, {miss} unmatched")
PY

rows=$(wc -l < "$WORK/corpus.txt")
say "bf-convert $rows rows -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -1
say "DONE: $WORK/corpus.bin ($rows positions). Train on it now; re-run as more shards land."
