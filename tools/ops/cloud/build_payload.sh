#!/usr/bin/env bash
# tools/ops/cloud/build_payload.sh — assemble the ONE artifact a cloud worker needs.
#
# USAGE
#   bash tools/ops/cloud/build_payload.sh [--out DIR]
#
# WHY A PAYLOAD TARBALL AND NOT A CONTAINER IMAGE
#   The previous cloud generation shipped a custom Docker image through Artifact
#   Registry: build_worker_image.py, an allow-list build context, a pinned
#   manifest, a repository, and a push. That machinery is the single largest
#   piece of the old system and it is the piece that rots fastest — the manifest
#   still in the tree pins `net-0.cenn` and three engine SHAs its own `_note`
#   records as already drifted at build time.
#
#   Everything that image bought is available for less:
#     * reproducibility  -> the tarball is sha256-pinned; the base image is a
#       PUBLIC python image pinned by digest. Two hashes instead of a registry.
#     * dependency isolation -> python-chess is PURE PYTHON (verified: zero .so
#       files), so it is vendored by directory copy and reached via PYTHONPATH.
#       No pip, no network install, nothing to resolve at task start. numpy is
#       avoided cloud-side by running the worker with SHUFFLE=0 — the only two
#       users of it, shuffle_records.py and merge_shards.py, are both subsumed
#       by the global shuffle that runs locally at merge time. (numpy ships
#       compiled extensions, so unlike python-chess it could NOT be vendored by
#       directory copy; the first draft of this file wrongly claimed it simply
#       was not needed, and stage 4 failed on the shuffler.)
#     * a place to put binaries -> they go in the tarball.
#
#   The result: no Docker daemon, no registry, no image build in the loop. The
#   payload can be built and tested on this box today, which the image could not.
#
# WHY IT SHIPS THE SAME SCRIPTS THE LOCAL CAMPAIGN RUNS
#   The old system had a SEPARATE cloud generator (native_game_generator.py, a
#   multipv/temperature sampler) that produced different positions with
#   different WDL semantics from anything local. Two generators means two
#   corpora that cannot be compared or pooled, and no way to tell which
#   difference in a trained net came from the data and which from the
#   generator. This ships tools/ops/data/ verbatim, so a cloud shard and a local
#   shard are the same computation with a different book slice.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

OUT=${1:-$ARCHON_ROOT/data/cloud/payload}
[ "$OUT" = "--out" ] && OUT=${2:?--out needs a dir}
mkdir -p "$OUT"
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/archon-payload.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

log "building cloud payload -> $OUT"

# The payload MIRRORS REPO PATHS exactly. It is a minimal checkout, not a new
# layout: `asset()` and every script resolve paths relative to ARCHON_ROOT, so
# `engines/bin/stockfish-eca43a97` must be at that path or gen_games.sh in
# oracle mode resolves an empty ENGINE and dies with "engine not executable:".
# Measured the hard way — the first payload used bin/ and book/ and failed in
# stage 1. A payload that renames anything is a second layout to keep in sync.
mkdir -p "$STAGE/engines/bin" "$STAGE/engines/books" "$STAGE/tools/ops/data" \
         "$STAGE/tools/ops/lib" "$STAGE/tools/nnue/bf-convert/target/release" \
         "$STAGE/tools/forensics" "$STAGE/vendor"

# --- binaries --------------------------------------------------------------
# PORTABLE builds, not the ones this box runs. Our local binaries are compiled
# -march=native on a Zen 4 CPU WITH AVX-512; GCP c2d is Zen 3 and has none, so
# they die instantly with SIGILL. That is exactly what killed the second canary:
# `Illegal instruction (core dumped)`, rc=132, fastchess gone in under a second.
# Rebuilt at engines/bin/portable/ with -march=x86-64-v3 (fastchess) and
# ARCH=x86-64-avx2 (Stockfish eca43a97).
#
# NOTE this is the one class of defect a local `docker run` CANNOT catch: the
# container uses the HOST cpu, so an AVX-512 binary works perfectly in a
# container on this box and fails on any machine without it.
for pair in \
    "engines/bin/portable/stockfish-eca43a97:engines/bin/stockfish-eca43a97" \
    "engines/bin/portable/fastchess:engines/bin/fastchess" \
    "tools/nnue/bf-convert/target/release/nnue_convert:tools/nnue/bf-convert/target/release/nnue_convert"
do
    src=${pair%%:*}; dst=${pair#*:}
    [ -f "$ARCHON_ROOT/$src" ] || { log "FATAL: missing $src"; exit 2; }
    # Fail HERE, not on a cloud VM. %zmm is the giveaway for AVX-512; match on
    # that specifically — a looser pattern (kmov) false-positives on AVX2's
    # vpmaskmovq and would reject a perfectly portable binary.
    if [ "${ALLOW_AVX512:-0}" != "1" ] \
       && objdump -d "$ARCHON_ROOT/$src" 2>/dev/null | grep -q '%zmm'; then
        log "FATAL: $src contains AVX-512 (%zmm). GCP c2d is Zen 3 and will SIGILL."
        log "  rebuild portable, or set ALLOW_AVX512=1 for an AVX-512 machine family."
        exit 2
    fi
    install -D -m755 "$ARCHON_ROOT/$src" "$STAGE/$dst"
done

# --- opening book ----------------------------------------------------------
# 167 MB, the largest single item. It ships whole rather than pre-sliced because
# a task's slice is chosen from its OWN task index at run time; pre-slicing
# would move that decision into the build and make the payload task-specific.
install -D -m644 "$ARCHON_ROOT/engines/books/UHO_Lichess_4852_v1.epd" \
    "$STAGE/engines/books/UHO_Lichess_4852_v1.epd"

# --- pipeline (the SAME scripts the local campaign runs) --------------------
for f in gen_games.sh pgn_positions.py label_inline.py label_positions.py \
         export_corpus.sh run_pipeline.sh census.py; do
    install -D -m755 "$ARCHON_ROOT/tools/ops/data/$f" "$STAGE/tools/ops/data/$f"
done
install -D -m644 "$ARCHON_ROOT/tools/ops/lib/common.sh" "$STAGE/tools/ops/lib/common.sh"
# label_inline.py IMPORTS d46_verdict from label_positions.py rather than
# copying it, so the two stages cannot drift apart on the filter that decides
# what enters the corpus. The cost of that choice is label_positions' own
# module-level `import uciprobe`, which must therefore exist here too even
# though the inline path never opens an engine. Worth it: a duplicated filter
# that silently diverges is a far worse failure than one extra stdlib-only file.
install -D -m644 "$ARCHON_ROOT/tools/forensics/uciprobe.py" "$STAGE/tools/forensics/uciprobe.py"
# The record shuffler. Its absence used to be a SILENT SKIP in export_corpus.sh
# (now fatal), which would have published unshuffled cloud shards.
install -D -m644 "$ARCHON_ROOT/tools/nnue/shuffle_records.py" "$STAGE/tools/nnue/shuffle_records.py"
install -D -m644 "$ARCHON_ROOT/tools/nnue/calib/wl50.fens" "$STAGE/tools/nnue/calib/wl50.fens" 2>/dev/null || true

# --- vendored python-chess (pure Python, no pip anywhere) ------------------
CHESS=$(ls -d "$ARCHON_ROOT"/.venv/lib/python*/site-packages/chess 2>/dev/null | head -1)
[ -d "$CHESS" ] || { log "FATAL: python-chess not found in .venv"; exit 2; }
if find "$CHESS" -name '*.so' | grep -q .; then
    log "FATAL: python-chess has compiled extensions here; vendoring by copy is unsafe"
    exit 2
fi
cp -r "$CHESS" "$STAGE/vendor/chess"
rm -rf "$STAGE/vendor/chess/__pycache__"

install -D -m755 "$(dirname "${BASH_SOURCE[0]}")/worker_shard.sh" "$STAGE/worker_shard.sh"
# Relabel campaign worker (2026-08-07): sibling of worker_shard.sh, pure-stdlib
# python labeler. Same payload, so one artifact serves both campaign types.
install -D -m755 "$(dirname "${BASH_SOURCE[0]}")/worker_relabel.sh" "$STAGE/worker_relabel.sh"
install -D -m644 "$(dirname "${BASH_SOURCE[0]}")/relabel_worker.py" "$STAGE/relabel_worker.py"

# --- manifest: every file with its hash, so drift is detectable -------------
python3 - "$STAGE" > "$STAGE/payload_manifest.json" <<'PYEOF'
import hashlib, json, os, sys
root = sys.argv[1]
files = {}
for dirpath, _, names in os.walk(root):
    for n in names:
        p = os.path.join(dirpath, n)
        rel = os.path.relpath(p, root)
        if rel == "payload_manifest.json":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        files[rel] = dict(sha256=h.hexdigest(), bytes=os.path.getsize(p))
json.dump(dict(schema="archon.cloud-payload.v1", files=files,
               total_bytes=sum(v["bytes"] for v in files.values())),
          sys.stdout, indent=2, sort_keys=True)
PYEOF

TAR=$OUT/payload.tar.gz
tar -C "$STAGE" -czf "$TAR" .
SHA=$(sha256sum "$TAR" | cut -d' ' -f1)
SIZE=$(stat -c%s "$TAR")
cp "$STAGE/payload_manifest.json" "$OUT/payload_manifest.json"
python3 - "$OUT/payload.meta.json" "$SHA" "$SIZE" "$TAR" <<'PYEOF'
import json, subprocess, sys, datetime
out, sha, size, tar = sys.argv[1:5]
json.dump(dict(schema="archon.cloud-payload-meta.v1", artifact=tar,
               sha256=sha, bytes=int(size),
               built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               git_head=subprocess.run(["git","rev-parse","HEAD"],
                                       capture_output=True, text=True).stdout.strip()),
          open(out, "w"), indent=2)
PYEOF

log "payload  $TAR"
log "  bytes  $SIZE  ($(numfmt --to=iec "$SIZE" 2>/dev/null || echo "$SIZE"))"
log "  sha256 $SHA"
log "  files  $(python3 -c "import json;print(len(json.load(open('$OUT/payload_manifest.json'))['files']))")"
