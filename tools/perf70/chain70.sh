#!/usr/bin/env bash
# tools/perf70/chain70.sh — run a long match as SHORT SEGMENTS that survive
# being OOM-killed, and pool them.
#
# Why: `tools/oomlast.sh` sets oom_score_adj=1000 BY DESIGN so the GPU trainer
# outlives everything else (standing rule). On a memory-tight box that makes a
# long match the kernel's first victim, and fastchess cannot resume — study 70
# lost 2,000 games and then 80 and then 16 that way. Segments bound the loss:
# each one prints its own `Ptnml(0-2)` when it finishes, and
# `tools/perf70/pool_sprt.py` pools the segments pentanomially, so a kill costs
# at most one segment instead of the whole run.
#
# It also WAITS for headroom rather than piling on: a segment only starts when
# `free -m` reports at least MIN_AVAIL MB available.
#
#   tools/perf70/chain70.sh CAND BASE [tag] [segments] [rounds-per-segment]
#
# Pool with:
#   tools/perf70/pool_sprt.py engines/runs/<tag>-*/sprt.out
set -uo pipefail
cd "$(dirname "$0")/../.."

CAND=${1:?usage: chain70.sh CAND BASE [tag] [segments] [rounds]}
BASE=${2:?}
TAG=${3:-s70-chain}
SEGMENTS=${4:-40}
ROUNDS=${5:-125}

MIN_AVAIL=${MIN_AVAIL:-2500}
CONC=${CONCURRENCY:-2}
NODES=${NODES:-25000}
BOOK=${BOOK:-engines/books/UHO_Lichess_4852_v1.epd}

for i in $(seq 1 "$SEGMENTS"); do
    # Wait for headroom. `available`, not `free`: reclaimable page cache is
    # not pressure, and this box always has a lot of it.
    for _ in $(seq 1 120); do
        avail=$(free -m | awk '/^Mem:/ {print $7}')
        [ "$avail" -ge "$MIN_AVAIL" ] && break
        sleep 30
    done
    seg=$(printf '%s-%03d' "$TAG" "$i")
    printf '=== segment %s (avail %s MB) ===\n' "$seg" "${avail:-?}"
    # `-sprt` bounds are left at the default; a segment is not meant to stop
    # on its own verdict, the POOL decides. ROUNDS is what ends it.
    CONCURRENCY="$CONC" NODES="$NODES" BOOK="$BOOK" ROUNDS="$ROUNDS" \
        OUT="engines/runs/$seg" \
        ./tools/oomlast.sh bash tools/sprt.sh "$CAND" "$BASE" > /dev/null 2>&1
    if grep -q "Ptnml" "engines/runs/$seg/sprt.out" 2> /dev/null; then
        printf '  ok: %s\n' "$(grep -m1 'Games:' "engines/runs/$seg/sprt.out")"
    else
        printf '  KILLED before its first report — segment lost\n'
    fi
    python3 tools/perf70/pool_sprt.py engines/runs/"$TAG"-*/sprt.out 2> /dev/null | tail -2
done
