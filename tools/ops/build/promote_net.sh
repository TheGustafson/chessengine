#!/usr/bin/env bash
# tools/ops/build/promote_net.sh — promote a packed net to CANONICAL, with all
# the bookkeeping that must move with it.
#
# PURPOSE
#   Flipping `engines/canonical/net.cenn` is not one action, it is seven, and
#   missing any of them leaves the repo in a state where the gate checks the
#   wrong thing. This does them in order and REFUSES rather than half-finishing.
#
# USAGE
#   bash tools/ops/build/promote_net.sh <packed.cenn> <net-id> [--elo "..."] [--dry-run]
#
#   bash tools/ops/build/promote_net.sh \
#       data/nnue/tmp/canon/d49sfb-sb1758-756.cenn d49sfb-sb1758-s756 \
#       --elo "+30.4 +/- 11.0 vs D48X over 1500 games"
#
# WHAT IT DOES, IN ORDER
#   1. copy the packed net into data/nnue/nets/screens/<net-id>.cenn (durable;
#      tmp/ is scratch and gets reaped)
#   2. repoint engines/canonical/net.cenn at it
#   3. RE-MEASURE both NNUE pins against the NEW net — a net change moves the
#      eval hash AND the d13 node total, unlike a search change which moves only
#      the node counts
#   4. rewrite those pins in tools/check.sh
#   5. rewrite the CURRENT pin line in CLAUDE.md (see the warning below)
#   6. run the full gate to prove the new pins
#   7. add a ledger row (D38) so the net's lineage is provable
#
# ⚠ THE CLAUDE.md TRAP THIS EXISTS TO PREVENT
#   `859a442` updated check.sh correctly but wrote its CLAUDE.md edit onto the
#   PRECEDING-pin line. The doc then named a superseded gate for a day and
#   silently overwrote an earlier study's real hash. **When you move a pin, the
#   CURRENT slot must move, not the history.** This script edits by anchored
#   pattern and verifies the old value is GONE from the current slot afterwards.
#
# ⚠ WHAT IT DOES NOT DO
#   It does not choose the pack SCALE. `network_scale` is a free multiplier with
#   a measured interior optimum, and that optimum was fitted for the OUTGOING
#   net — a new net may want a different one. Sweep it BEFORE promoting if the
#   margin is close; this script promotes what you hand it.
#
#   It also does not update engines/canonical/archon (the release BINARY). A net
#   promotion and an engine promotion are separate decisions.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

SRC=${1:?usage: promote_net.sh <packed.cenn> <net-id> [--elo "..."] [--dry-run]}
NET_ID=${2:?net-id required (becomes the filename and the ledger id)}
shift 2
ELO=""; DRY=0; NEW_ARCH=""; NEW_ENGINE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --elo) ELO=$2; shift 2;;
        --arch) NEW_ARCH=$2; shift 2;;
        --engine) NEW_ENGINE=$2; shift 2;;
        --dry-run) DRY=1; shift;;
        *) log "unknown arg: $1"; exit 2;;
    esac
done

case "$SRC" in /*) ;; *) SRC="$ARCHON_ROOT/$SRC";; esac
[ -s "$SRC" ] || { log "FATAL: packed net not found: $SRC"; exit 2; }
DEST="$ARCHON_ROOT/data/nnue/nets/screens/${NET_ID}.cenn"
ARCH=$(cat "$ARCHON_ROOT/engines/canonical/arch.txt")
OLD_ARCH="$ARCH"
[ -n "$NEW_ARCH" ] && ARCH="$NEW_ARCH"

# ⚠ AN ARCH CHANGE IS A THREE-WAY COUPLING, not a one-file edit.
#   arch.txt  net.cenn  archon(binary)  must all move together.
# A .cenn records its acc_width and the loader REFUSES a mismatch, so promoting a
# 768 net while engines/canonical/archon is still a 1280 build leaves a canonical
# ENGINE that cannot load the canonical NET -- and nothing else in the tree
# checks that pair. Refuse rather than produce it.
if [ "$ARCH" != "$OLD_ARCH" ]; then
    log "ARCH CHANGE: $OLD_ARCH -> $ARCH"
    [ -n "$NEW_ENGINE" ] || {
        log "FATAL: --arch requires --engine <binary built with the NEW arch>."
        log "       Build it first:  bash tools/ops/build/pgo.sh <name> '$ARCH' <net>"
        log "       Otherwise the canonical engine cannot load the canonical net."
        exit 2
    }
    case "$NEW_ENGINE" in /*) ;; *) NEW_ENGINE="$ARCHON_ROOT/$NEW_ENGINE";; esac
    [ -x "$NEW_ENGINE" ] || { log "FATAL: --engine not executable: $NEW_ENGINE"; exit 2; }
fi
OLD=$(readlink -f "$ARCHON_ROOT/engines/canonical/net.cenn")
OPS_LOG=${OPS_LOG:-$ARCHON_D49/promote.log}; export OPS_LOG

log "promote: $(basename "$SRC") -> canonical as '${NET_ID}'"
log "  outgoing: $(basename "$OLD")"
log "  arch:     $ARCH"
[ "$DRY" = "1" ] && { log "--dry-run: stopping before any mutation"; exit 0; }

# 1-2. place and repoint
cp -n "$SRC" "$DEST" 2>/dev/null || cp "$SRC" "$DEST"
ln -sfn "../../data/nnue/nets/screens/${NET_ID}.cenn" "$ARCHON_ROOT/engines/canonical/net.cenn"
log "  canonical -> $(basename "$(readlink -f "$ARCHON_ROOT/engines/canonical/net.cenn")")"

if [ "$ARCH" != "$OLD_ARCH" ]; then
    printf '%s\n' "$ARCH" > "$ARCHON_ROOT/engines/canonical/arch.txt"
    cp "$NEW_ENGINE" "$ARCHON_ROOT/engines/bin/$(basename "$NEW_ENGINE")" 2>/dev/null || true
    ln -sfn "../bin/$(basename "$NEW_ENGINE")" "$ARCHON_ROOT/engines/canonical/archon"
    log "  arch.txt -> $ARCH"
    log "  archon   -> $(basename "$NEW_ENGINE")"
    # PROVE the canonical pair actually loads before anything downstream trusts it
    if "$ARCHON_ROOT/engines/canonical/archon" nnue-eval \
         "$(readlink -f "$ARCHON_ROOT/engines/canonical/net.cenn")" \
         < tools/nnue/calib/wl50.fens 2>&1 | grep -qiE 'ERROR|mismatch|cannot'; then
        log "FATAL: canonical engine cannot load canonical net after the arch change"
        exit 6
    fi
    log "  canonical engine + net load OK"
fi

# 3. re-measure BOTH pins against the NEW net
need_mem 2500 || exit 1
log "  rebuilding to re-measure pins (a net change moves the eval hash too)"
cargo build --release --features "$ARCH" --target-dir target/nnue-gate --quiet 2>&1 | tail -3
BIN=./target/nnue-gate/release/chessengine
NEWHASH=$($BIN nnue-eval "$(readlink -f engines/canonical/net.cenn)" < tools/nnue/calib/wl50.fens | sha256sum | cut -d' ' -f1)
NEWD13=$($BIN bench 16 1 13 tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn 2>&1 \
         | grep -E 'Nodes searched' | tail -1 | grep -oE '[0-9]+$')
OLDHASH=$(grep -oE '[0-9a-f]{64}' tools/check.sh | head -1)
OLDD13=$(grep -oE 'NNUE_NODE_PIN-[0-9]+' tools/check.sh | grep -oE '[0-9]+')
log "  eval hash : ${OLDHASH:0:8}… -> ${NEWHASH:0:8}…"
log "  d13 nodes : ${OLDD13} -> ${NEWD13}"
[ -n "$NEWD13" ] || { log "FATAL: could not measure d13"; exit 3; }

# 4. check.sh
sed -i "s/${OLDHASH}/${NEWHASH}/g" tools/check.sh
sed -i "s/NNUE_NODE_PIN-${OLDD13}/NNUE_NODE_PIN-${NEWD13}/" tools/check.sh

# 5. CLAUDE.md CURRENT slot only, then PROVE the old value left it
sed -i "s/\*\*d13 node total ${OLDD13}\*\*/**d13 node total ${NEWD13}**/" CLAUDE.md
# CLAUDE.md writes the hash TRUNCATED ("2e1ede95…79c4"), so a full-64-char sed
# silently misses it and the doc keeps naming the superseded net.
OLDT="${OLDHASH:0:8}…${OLDHASH: -4}"; NEWT="${NEWHASH:0:8}…${NEWHASH: -4}"
sed -i "s/${OLDT}/${NEWT}/g" CLAUDE.md
# NB: take the LAST field, not `grep -oE '[0-9]+'` -- that also matches the "13"
# inside "d13" and the comparison can then never succeed. This exact bug made the
# guard refuse a promotion whose edit had in fact landed (2026-08-02).
CUR=$(grep -oE 'd13 node total [0-9]+' CLAUDE.md | head -1 | awk '{print $NF}')
if [ "$CUR" != "$NEWD13" ]; then
    log "FATAL: CLAUDE.md current pin slot did not move (still ${CUR}). Fix by hand;"
    log "       do NOT edit a history line — see the header warning."
    exit 4
fi
log "  CLAUDE.md current slot verified at ${NEWD13}"

# 6. prove it
log "  running the full gate"
CHECK_NNUE_GATE=1 bash tools/check.sh > "$ARCHON_D49/promote_gate.log" 2>&1
rc=$?
tail -4 "$ARCHON_D49/promote_gate.log" | sed 's/^/    /'
[ "$rc" -eq 0 ] || { log "FATAL: gate FAILED after promotion (rc=$rc) — revert before committing"; exit 5; }

# 7. ledger (D38): identity is the artifact sha256; every row carries a recipe
python3 tools/ledger/ledger.py add-net \
    --id "$NET_ID" --file "$DEST" --arch "$ARCH" \
    --provenance "promoted to canonical $(date -Is)" \
    ${ELO:+--elo-vs-parent "$ELO"} \
    --notes "canonical promotion; eval hash ${NEWHASH:0:16}, d13 ${NEWD13}" \
    2>&1 | tail -2 | sed 's/^/    /'

log "PROMOTED. Commit tools/check.sh + CLAUDE.md + engines/canonical/net.cenn + ledger/ together."
