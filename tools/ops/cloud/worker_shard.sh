#!/usr/bin/env bash
# tools/ops/cloud/worker_shard.sh — produce ONE campaign shard. The whole cloud
# worker; also runnable locally, which is how it is tested.
#
# USAGE (Batch sets BATCH_TASK_INDEX; locally pass TASK_INDEX)
#   TASK_INDEX=0 DEST=gs://archon-main-bucket/campaigns/oracle-1b \
#   CAMPAIGN=oracle-1b SHARD_POS=2000000 NODES=5000 CONC=16 \
#     bash worker_shard.sh
#
#   DEST may be a gs:// URI OR A LOCAL DIRECTORY. That is not a convenience —
#   it is what makes this script testable without any cloud access at all. The
#   old cloud workers reached for GCS directly and could only be exercised by
#   submitting a job, so the first real test of the code was also the first
#   time money was spent.
#
# IDEMPOTENT BY CONSTRUCTION
#   A Spot task can be preempted at any moment, and Batch will re-run it. So
#   the first thing this does is ask whether its shard already exists at DEST;
#   if the artifact and its shard.json are both there it exits 0 immediately.
#   Uploads publish the artifact LAST, after shard.json, so a half-finished
#   upload can never look complete: the presence of corpus.bin is the
#   commit point.
#
# WHAT IT DOES NOT DO
#   No merging, no shuffling across shards, no ledger writes. Those need the
#   whole corpus and happen locally, once, in merge_shards.py. A worker that
#   only ever appends immutable, self-describing shards cannot corrupt the
#   campaign by dying at a bad moment.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ARCHON_ROOT=${ARCHON_ROOT:-$HERE}
export PYTHONPATH="$HERE/vendor:${PYTHONPATH:-}"

TASK_INDEX=${TASK_INDEX:-${BATCH_TASK_INDEX:-0}}
CAMPAIGN=${CAMPAIGN:?CAMPAIGN required}
DEST=${DEST:?DEST required (gs://... or a local dir)}
SHARD_POS=${SHARD_POS:-2000000}
NODES=${NODES:-5000}
PERGAME=${PERGAME:-80}
CONC=${CONC:-16}
KEPT_PER_GAME=${KEPT_PER_GAME:-57.5}
WORKDIR=${WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/shard.XXXXXX")}

SHARD=$(printf 'shard-%04d' "$TASK_INDEX")
say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

# --- storage backend -------------------------------------------------------
# One indirection so gs:// and a local path are interchangeable.
st_exists() {
    case "$1" in
        gs://*) gcloud storage ls "$1" >/dev/null 2>&1 ;;
        *)      [ -s "$1" ] ;;
    esac
}
# st_put LOCAL REMOTE — upload, tolerating an object that is already there.
#
# THE WORKER IDENTITY HOLDS objectCreator, NOT objectAdmin: it may CREATE an
# object and may not OVERWRITE one. That is a deliberate safety property (a
# runaway task cannot destroy a corpus) and it breaks the naive retry path. A
# Spot task preempted after uploading shard.json but before corpus.bin comes
# back, finds the shard incomplete, regenerates — and then gets 403 re-uploading
# shard.json, failing forever on a shard it could have finished.
#
# So: if the object already exists, verify our bytes match what is there and
# treat it as done. A genuine mismatch is loud, because that means two tasks
# produced different data for the same shard and something is wrong upstream.
st_put() {  # st_put LOCAL REMOTE
    case "$2" in
        gs://*)
            if gcloud storage ls "$2" >/dev/null 2>&1; then
                # Compare MD5, not SHA256. GCS stores crc32c and md5 for an
                # object; there is no server-side sha256, and `gcloud storage
                # hash` hashes LOCAL files. Asking it for a remote sha256
                # silently returned nothing, which made the mismatch branch
                # unreachable and turned this into "assume it matches".
                local want got
                want=$(openssl dgst -md5 -binary "$1" | base64)
                got=$(gcloud storage objects describe "$2" --format="value(md5_hash)" 2>/dev/null | tr -d "\r")
                if [ -z "$got" ]; then
                    say "FATAL: $2 exists but its md5 could not be read; refusing to assume it matches"
                    return 1
                fi
                if [ "$want" != "$got" ]; then
                    say "FATAL: $2 exists with DIFFERENT bytes (want $want got $got)"
                    return 1
                fi
                say "  already present, skipping: $(basename "$2")"
                return 0
            fi
            gcloud storage cp "$1" "$2" >/dev/null 2>&1
            ;;
        *)      mkdir -p "$(dirname "$2")" && cp "$1" "$2" ;;
    esac
}

say "$SHARD task_index=$TASK_INDEX campaign=$CAMPAIGN dest=$DEST"

if st_exists "$DEST/$SHARD/corpus.bin" && st_exists "$DEST/$SHARD/shard.json"; then
    say "$SHARD already published — nothing to do"
    exit 0
fi

# --- preflight: fail loudly here, not 40 minutes into a match --------------
for b in engines/bin/stockfish-eca43a97 engines/bin/fastchess \
         tools/nnue/bf-convert/target/release/nnue_convert; do
    [ -x "$HERE/$b" ] || { say "FATAL: missing or non-executable $b"; exit 2; }
    # Test the DYNAMIC LINKS, not an exit code. The thing that can actually be
    # wrong on a fresh base image is a missing libstdc++6 (Stockfish and
    # fastchess are both dynamically linked C++), and `ldd` reports that
    # directly. Probing with `--help` or a `quit` on stdin instead conflates it
    # with CLI convention: nnue_convert exits non-zero on a usage error, so a
    # perfectly working binary failed this check on its first run.
    # SIGILL (exit 132) means the binary needs CPU instructions this machine
    # does not have. An `ldd` check cannot see it — links resolve fine — and
    # this preflight originally DID execute each binary until I replaced that
    # with ldd because nnue_convert exits non-zero on a usage error. Removing
    # execution removed the only check that would have caught an AVX-512 binary
    # on a Zen 3 VM, which is what the second canary died of. So execute again,
    # but judge ONLY on 132 and ignore ordinary non-zero exits.
    "$HERE/$b" --version >/dev/null 2>&1 || true
    if [ $? -eq 132 ] || { "$HERE/$b" --help >/dev/null 2>&1; [ $? -eq 132 ]; }; then
        say "FATAL: $b died with SIGILL — built for a CPU this machine is not"
        say "  this host: $(grep -m1 '^flags' /proc/cpuinfo | grep -o 'avx512[a-z]*' | tr '\n' ' ')"
        exit 2
    fi
    if ldd "$HERE/$b" 2>/dev/null | grep -q 'not found'; then
        say "FATAL: $b has unresolved shared libraries on this image"
        ldd "$HERE/$b" 2>&1 | grep 'not found' | sed 's/^/    /'
        say "  the base image needs libstdc++6 (use python:3.12, not -slim, or apt-get install it)"
        exit 2
    fi
done
python3 -c 'import chess' 2>/dev/null || { say "FATAL: vendored python-chess not importable"; exit 2; }
say "preflight ok"

BOOK=$HERE/engines/books/UHO_Lichess_4852_v1.epd
BOOKLINES=$(wc -l < "$BOOK")
GAMES=$(python3 -c "print(int($SHARD_POS / $KEPT_PER_GAME))")

# Book slice: identical derivation to gen_campaign.sh, so a cloud shard and a
# local shard with the same campaign name and index mine the SAME openings and
# are directly comparable.
BASE=$(( 0x$(printf '%s' "$CAMPAIGN" | sha256sum | cut -c1-8) % BOOKLINES ))
SPAN=$(( GAMES + 16 ))
BOOKSTART=$(( (BASE + TASK_INDEX * SPAN) % (BOOKLINES - SPAN) + 1 ))
say "$SHARD ~$GAMES games at $NODES nodes, book line $BOOKSTART, conc=$CONC"

SDIR=$WORKDIR/$SHARD
mkdir -p "$SDIR"

# PY=python3: no virtualenv here; python-chess is vendored onto PYTHONPATH.
#
# SHUFFLE=0 IS DELIBERATE AND IT IS WHY NUMPY IS NOT NEEDED CLOUD-SIDE.
# The per-shard shuffle is redundant work: merge_shards.py performs a two-pass
# GLOBAL external shuffle across every shard when the campaign comes home, and
# shuffling within a shard first changes nothing about that result. Dropping it
# removes the only cloud-side numpy dependency — which matters, because numpy
# ships compiled extensions and could not be vendored by directory copy the way
# pure-Python python-chess can.
#
# The artifact is therefore named corpus.bin, NOT corpus.shuf.bin. That is the
# honest name: anyone who picks a shard up without merging must be able to see
# from the filename that its rows are still in game order.
MODE=oracle GAMES=$GAMES GAMENODES=$NODES PERGAME=$PERGAME CONC=$CONC \
INLINE=1 MIN_ROWS=1000 SHUFFLE=0 TAG="$CAMPAIGN/$SHARD" OUTDIR="$SDIR" \
PY=python3 ARCHON_ROOT="$HERE" ARCHON_MAIN="$HERE" \
    bash "$HERE/tools/ops/data/run_pipeline.sh" 2>&1 | sed 's/^/    /'
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || { say "FATAL: pipeline rc=$rc"; exit 3; }
[ -s "$SDIR/corpus.bin" ] || { say "FATAL: no corpus produced"; exit 3; }

# --- merge metadata, compress the provenance source ------------------------
# The payload manifest is the cloud equivalent of a git head: outside a
# checkout there is no commit to record, but there IS an exact set of hashed
# files that produced this shard. Carry it.
python3 - "$SDIR" "$CAMPAIGN" "$SHARD" "$TASK_INDEX" "$BOOKSTART" "$HERE" <<'PYEOF'
import hashlib, json, pathlib, sys
d, campaign, shard, idx, bookstart, here = sys.argv[1:7]
d = pathlib.Path(d)
def load(n):
    try: return json.loads((d / n).read_text())
    except Exception: return None
out = dict(campaign=campaign, shard=shard, task_index=int(idx),
           book_start=int(bookstart),
           manifest=load("manifest.json"), extract=load("positions.tsv.stats.json"),
           label=load("labelled.tsv.stats.json"), corpus=load("corpus.meta.json"),
           census=load("census.json"))
pm = pathlib.Path(here) / "payload_manifest.json"
if pm.is_file():
    out["payload_manifest_sha256"] = hashlib.sha256(pm.read_bytes()).hexdigest()
out["records"] = (out.get("corpus") or {}).get("records", 0)
out["sha256"] = (out.get("corpus") or {}).get("sha256")
(d / "shard.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["records"])
PYEOF

if [ -s "$SDIR/games.pgn" ]; then
    zstd -19 -q --rm "$SDIR/games.pgn" -o "$SDIR/games.pgn.zst" 2>/dev/null \
        || gzip -9 "$SDIR/games.pgn"
fi

# --- publish: artifact LAST, so its presence is the commit point -----------
st_put "$SDIR/shard.json" "$DEST/$SHARD/shard.json" || { say "FATAL: upload shard.json"; exit 4; }
for f in games.pgn.zst games.pgn.gz; do
    [ -s "$SDIR/$f" ] && { st_put "$SDIR/$f" "$DEST/$SHARD/$f" || { say "FATAL: upload $f"; exit 4; }; }
done
st_put "$SDIR/corpus.bin" "$DEST/$SHARD/corpus.bin" || { say "FATAL: upload corpus"; exit 4; }

N=$(python3 -c "import json;print(json.load(open('$SDIR/shard.json'))['records'])")
say "$SHARD PUBLISHED $N records -> $DEST/$SHARD/"
rm -rf "$WORKDIR"
