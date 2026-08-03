#!/usr/bin/env bash
# tools/ops/cloud/fetch_shards.sh — campaign progress, and bring the shards home.
#
# USAGE
#   bash tools/ops/cloud/fetch_shards.sh --dest gs://bucket/campaigns/x --status
#   bash tools/ops/cloud/fetch_shards.sh --dest gs://bucket/campaigns/x --into DIR
#
# VERIFY ON ARRIVAL, NOT ON DEPARTURE
#   Each shard.json carries the sha256 the worker computed over the corpus it
#   wrote. This recomputes it locally after download and refuses to count a
#   shard whose bytes do not match. That is the only check that covers the whole
#   path — a corrupted or truncated object looks perfectly fine to a listing,
#   and `fishpack32.shuf.bin` is already in the training mix with a hash that
#   never matched because it was registered mid-write.
#
#   Downloads land in a .part file and are renamed only after the hash passes,
#   so an interrupted fetch cannot leave a shard that later looks complete.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

DEST=""; INTO=""; STATUS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST=$2; shift 2;;
        --into) INTO=$2; shift 2;;
        --status) STATUS=1; shift;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done
[ -n "$DEST" ] || { log "FATAL: --dest required"; exit 2; }

ls_shards() {
    case "$DEST" in
        gs://*) gcloud storage ls "$DEST/" 2>/dev/null | grep -oE 'shard-[0-9]+' | sort -u ;;
        *)      ls -d "$DEST"/shard-* 2>/dev/null | xargs -r -n1 basename ;;
    esac
}
get() {  # get REMOTE LOCAL
    case "$1" in
        gs://*) gcloud storage cp "$1" "$2" >/dev/null 2>&1 ;;
        *)      cp "$1" "$2" 2>/dev/null ;;
    esac
}

SHARDS=$(ls_shards)
N=$(printf '%s\n' "$SHARDS" | grep -c . || true)
[ "$N" -gt 0 ] || { log "no shards at $DEST"; exit 0; }

if [ "$STATUS" = "1" ]; then
    log "$N shard(s) published at $DEST"
    tmp=$(mktemp -d); total=0
    for s in $SHARDS; do
        get "$DEST/$s/shard.json" "$tmp/$s.json" || continue
        r=$(python3 -c "import json;print(json.load(open('$tmp/$s.json')).get('records',0))" 2>/dev/null || echo 0)
        total=$((total + r))
    done
    rm -rf "$tmp"
    log "  total records: $total"
    exit 0
fi

[ -n "$INTO" ] || { log "FATAL: --into DIR required (or use --status)"; exit 2; }
mkdir -p "$INTO"

ok=0; bad=0; skipped=0; total=0
for s in $SHARDS; do
    d=$INTO/$s; mkdir -p "$d"
    if [ -s "$d/corpus.bin" ] && [ -s "$d/shard.json" ]; then
        r=$(python3 -c "import json;print(json.load(open('$d/shard.json')).get('records',0))" 2>/dev/null || echo 0)
        total=$((total + r)); skipped=$((skipped + 1)); continue
    fi
    get "$DEST/$s/shard.json" "$d/shard.json" || { log "  $s: no shard.json, skipping"; continue; }
    get "$DEST/$s/games.pgn.zst" "$d/games.pgn.zst" || true
    get "$DEST/$s/corpus.bin" "$d/corpus.bin.part" || { log "  $s: corpus fetch FAILED"; bad=$((bad+1)); continue; }

    want=$(python3 -c "import json;print(json.load(open('$d/shard.json')).get('sha256') or '')")
    got=$(sha256sum "$d/corpus.bin.part" | cut -d' ' -f1)
    if [ -n "$want" ] && [ "$want" != "$got" ]; then
        log "  $s: SHA MISMATCH — want ${want:0:16}… got ${got:0:16}…  (kept as .part)"
        bad=$((bad + 1)); continue
    fi
    mv "$d/corpus.bin.part" "$d/corpus.bin"
    r=$(python3 -c "import json;print(json.load(open('$d/shard.json')).get('records',0))")
    total=$((total + r)); ok=$((ok + 1))
done

log "fetched $ok, already had $skipped, FAILED $bad — $total records in $INTO"
[ "$bad" -eq 0 ] || exit 1
log "next: .venv/bin/python tools/ops/data/merge_shards.py $INTO --out <corpus>.shuf.bin"
