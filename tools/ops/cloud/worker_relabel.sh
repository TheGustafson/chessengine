#!/usr/bin/env bash
# tools/ops/cloud/worker_relabel.sh — relabel ONE shard of a bulletformat
# corpus with fixed-node SF searches. Sibling of worker_shard.sh: same
# DEST-may-be-local property, same idempotency, same st_put semantics.
#
# USAGE (Batch sets BATCH_TASK_INDEX; locally pass TASK_INDEX)
#   TASK_INDEX=0 CAMPAIGN=relabel-fishpack-v1 \
#   SRC=gs://archon-main-bucket/corpora/fishpack32.shuf.bin  (or a local path)
#   DEST=gs://archon-main-bucket/relabel/fishpack-v1         (or a local dir)
#   SHARD_POS=4000000 NODES=5000 CONC=16 bash worker_relabel.sh
#
# The shard covers records [TASK_INDEX*SHARD_POS, +SHARD_POS) of SRC, fetched
# as a byte range — no pre-split objects. Output: shard-NNNN.bin (records with
# ONLY the score bytes replaced; castle-possible rows dropped) + stats json.
# DETERMINISM: CONC is part of the campaign config (see relabel_worker.py);
# a retry with the same env reproduces the shard byte-for-byte.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ARCHON_ROOT=${ARCHON_ROOT:-$HERE}
TASK_INDEX=${TASK_INDEX:-${BATCH_TASK_INDEX:-0}}
CAMPAIGN=${CAMPAIGN:?CAMPAIGN required}
SRC=${SRC:?SRC required (gs://... or local .shuf.bin)}
DEST=${DEST:?DEST required}
SHARD_POS=${SHARD_POS:-4000000}
NODES=${NODES:-5000}
CONC=${CONC:-16}
ENGINE=${ENGINE:-$ARCHON_ROOT/engines/bin/stockfish-eca43a97}
WORKDIR=${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/relabel.XXXXXX")}
SHARD=$(printf 'shard-%05d' "$TASK_INDEX")
say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

st_exists() { case "$1" in gs://*) gcloud storage ls "$1" >/dev/null 2>&1 ;; *) [ -s "$1" ] ;; esac; }
st_put() {
    case "$2" in
        gs://*)
            if gcloud storage ls "$2" >/dev/null 2>&1; then
                local want got
                want=$(openssl dgst -md5 -binary "$1" | base64)
                got=$(gcloud storage objects describe "$2" --format="value(md5_hash)" 2>/dev/null | tr -d "\r")
                [ -z "$got" ] && { say "FATAL: $2 exists, md5 unreadable"; return 1; }
                [ "$want" != "$got" ] && { say "FATAL: $2 exists with DIFFERENT bytes"; return 1; }
                say "  already present: $(basename "$2")"; return 0
            fi
            gcloud storage cp "$1" "$2" ;;
        *) mkdir -p "$(dirname "$2")"; cp "$1" "$2" ;;
    esac
}

# --- idempotency: artifact LAST is the commit point ------------------------
if st_exists "$DEST/$SHARD.bin" && st_exists "$DEST/$SHARD.json"; then
    say "$SHARD already complete at $DEST — nothing to do"; exit 0
fi

# --- ranged fetch of exactly this shard's records --------------------------
START=$(( TASK_INDEX * SHARD_POS * 32 ))
END=$(( START + SHARD_POS * 32 - 1 ))          # inclusive byte range
case "$SRC" in
    gs://*)
        # The last shard is usually partial: clamp END to the object's final
        # byte, or `gcloud storage cat -r` errors "Download not completed"
        # after streaming everything that exists (task 403 of the fishpack-v1
        # campaign died on this, 2026-08-09).
        SIZE=$(gcloud storage objects describe "$SRC" --format='value(size)') \
            && [ -n "$SIZE" ] || { say "FATAL: cannot stat $SRC"; exit 3; }
        [ "$START" -lt "$SIZE" ] || { say "FATAL: empty slice (past EOF? task $TASK_INDEX)"; exit 4; }
        [ "$END" -ge "$SIZE" ] && END=$((SIZE - 1))
        gcloud storage cat -r "$START-$END" "$SRC" > "$WORKDIR/slice.bin" \
            || { say "FATAL: ranged read failed"; exit 3; } ;;
    *)
        dd if="$SRC" of="$WORKDIR/slice.bin" bs=32 skip=$(( TASK_INDEX * SHARD_POS )) \
           count="$SHARD_POS" status=none || { say "FATAL: dd failed"; exit 3; } ;;
esac
BYTES=$(stat -c%s "$WORKDIR/slice.bin")
[ "$BYTES" -gt 0 ] || { say "FATAL: empty slice (past EOF? task $TASK_INDEX)"; exit 4; }
say "$SHARD: $((BYTES / 32)) records, labeling at ${NODES}n x${CONC}"

python3 "$HERE/relabel_worker.py" "$WORKDIR/slice.bin" "$WORKDIR/out.bin" \
    "$WORKDIR/stats.json" "$ENGINE" "$NODES" "$CONC" \
    || { say "FATAL: labeler failed"; exit 5; }

st_put "$WORKDIR/stats.json" "$DEST/$SHARD.json" || exit 6
st_put "$WORKDIR/out.bin"    "$DEST/$SHARD.bin"  || exit 6
say "$SHARD done"
rm -rf "$WORKDIR"
