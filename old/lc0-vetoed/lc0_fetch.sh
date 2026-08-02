#!/usr/bin/env bash
# tools/ops/data/lc0_fetch.sh — fetch Lc0 training tars, newest-first, with a manifest.
#
# PURPOSE
#   Acquire the first 1 B positions of Lc0 test91 data (see
#   docs/research/lc0-acquisition-plan-1B.md). Newest-first, resumable, and every
#   file recorded so the NEXT slice can continue with no overlap and no holes.
#
# USAGE
#   systemd-run --user --unit=lc0fetch -p MemoryMax=1G \
#       /path/to/tools/ops/data/lc0_fetch.sh
#
#   Environment:
#     RUN          archive subdir            (default test91)
#     TARGET_GB    stop after this much      (default 208 -> ~1 B positions)
#     MIN_FREE_GB  refuse/stop below this    (default 60)
#     DEST         where tars land           (default data/nnue/lc0/tars)
#
# WHY NEWEST-FIRST
#   test91 is live and still growing; its newest data comes from its strongest
#   net. Walking backwards also makes "what do we have" a single date range
#   rather than a scattered set.
#
# THE MANIFEST IS THE POINT
#   data/nnue/lc0/manifest.tsv records name, bytes, sha256, and status for every
#   tar. Without it a second download either re-fetches what we hold or leaves a
#   hole, and neither is detectable after the fact.
#
# DISK
#   Stops cleanly at MIN_FREE_GB. This box runs a GPU trainer and a match; filling
#   the volume would take both down. 208 GB into 281 GB free leaves ~73 GB, which
#   is why the floor is checked EVERY file rather than once at the start.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

RUN=${RUN:-test91}
TARGET_GB=${TARGET_GB:-208}
MIN_FREE_GB=${MIN_FREE_GB:-60}
DEST=${DEST:-$ARCHON_ROOT/data/nnue/lc0/tars}
BASE="https://storage.lczero.org/files/training_data/${RUN}"
MAN="$ARCHON_ROOT/data/nnue/lc0/manifest.tsv"
OPS_LOG=${OPS_LOG:-$ARCHON_D49/lc0_fetch.log}; export OPS_LOG

mkdir -p "$DEST" "$(dirname "$MAN")"
[ -s "$MAN" ] || printf 'tar_name\tbytes\tsha256\tstatus\n' > "$MAN"

log "lc0_fetch: run=$RUN target=${TARGET_GB}GB floor=${MIN_FREE_GB}GB dest=$DEST"

LIST=$(mktemp)
curl -sS --max-time 120 "$BASE/" | grep -oE "training-run[0-9]+-${RUN}-[0-9]{8}-[0-9]{4}\.tar" \
    | sort -u > "$LIST"
TOTAL=$(wc -l < "$LIST")
[ "$TOTAL" -gt 0 ] || { log "FATAL: archive listing empty"; exit 2; }
log "  archive has $TOTAL tars; taking newest-first"

got_gb=0
while read -r name; do
    grep -qF "$name" "$MAN" && continue          # already have it
    free_gb=$(df --output=avail -BG "$DEST" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        log "STOP: free ${free_gb}GB < floor ${MIN_FREE_GB}GB after ${got_gb}GB"
        break
    fi
    if [ "$got_gb" -ge "$TARGET_GB" ]; then
        log "DONE: reached target ${TARGET_GB}GB"
        break
    fi
    out="$DEST/$name"
    if ! curl -sS --fail --max-time 900 -C - -o "$out" "$BASE/$name"; then
        log "  FAILED $name (kept going)"
        printf '%s\t0\t-\tfailed\n' "$name" >> "$MAN"
        rm -f "$out"; continue
    fi
    b=$(stat -c %s "$out"); h=$(sha256sum "$out" | cut -d' ' -f1)
    printf '%s\t%s\t%s\tok\n' "$name" "$b" "$h" >> "$MAN"
    got_gb=$((got_gb + b/1000000000))
    n=$(grep -c 'ok$' "$MAN")
    [ $((n % 25)) -eq 0 ] && log "  $n tars, ~${got_gb}GB, free $(df --output=avail -BG "$DEST" | tail -1 | tr -d ' ')"
done < <(tac "$LIST")

rm -f "$LIST"
OK=$(grep -c 'ok$' "$MAN"); GB=$(awk -F'\t' '$4=="ok"{s+=$2} END{printf "%.1f", s/1e9}' "$MAN")
log "lc0_fetch finished: $OK tars, ${GB} GB"
log "  oldest held: $(awk -F'\t' '$4=="ok"{print $1}' "$MAN" | sort | head -1)"
log "  newest held: $(awk -F'\t' '$4=="ok"{print $1}' "$MAN" | sort | tail -1)"
log "  NEXT SLICE continues BACKWARDS from the oldest held tar."
