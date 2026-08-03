#!/usr/bin/env bash
# tools/ops/data/gen_campaign.sh — run oracle-vs-oracle data generation at scale.
#
# USAGE
#   systemd-run --user --unit=dgcampaign \
#     --working-directory=/home/nick/projects/chessengine \
#     -E TARGET=100000000 -E CONC=6 \
#     /home/nick/projects/chessengine/tools/ops/data/gen_campaign.sh
#
#   Environment:
#     TARGET     kept positions to produce        (default 10000000)
#     SHARD_POS  kept positions per shard         (default 2000000)
#     NODES      per-move budget, both sides      (default 5000)
#     PERGAME    positions sampled per game       (default 80)
#     CONC       fastchess concurrency            (default 4)
#     NAME       campaign name / dir              (default oracle-<date>)
#     KEEP_PGN   1 = keep games.pgn.zst           (default 1)
#
# WHAT IT PRODUCES  data/nnue/selfplay/campaigns/<NAME>/
#     campaign.json          running totals + per-shard index
#     shard-NNNN/
#         corpus.shuf.bin    the artifact
#         games.pgn.zst      compressed provenance source (KEEP_PGN=1)
#         shard.json         manifest + every stage's stats, merged
#
# ---------------------------------------------------------------------------
# WHY THE ON-DISK LAYOUT IS THREE FILES AND NOT TEN
#
#   The four-stage pipeline writes 0.45 GB per 1M kept positions of which
#   0.03 GB is the deliverable — 15x write amplification. That structure buys
#   stage-boundary resume, which mattered when labelling was 2.95 of 5.85
#   core-h. With in-game labels (INLINE=1) labelling is ~0 and GENERATION IS
#   100 % OF THE COST, so the boundary it protected no longer needs protecting.
#
#   Left alone the layout does not fit the job: 1.03 B positions (mix weight
#   0.10) is 464 GB against 266 GB free. Reaping the intermediates at the
#   export boundary takes it to 0.05 GB per 1M => 52 GB. 9x.
#
#   The compressed PGN is the one intermediate worth keeping. It is the only
#   re-derivable source: a different per-game cap, a different filter, or a
#   re-label at any depth can all be produced from it WITHOUT replaying a
#   single game — and replaying games is now the entire bill. zstd-19 gets
#   4.3x on this data (measured), so it costs 19 GB per billion positions.
#
# WHY SHARDS
#   A 24-hour single run that dies at hour 23 loses 23 hours. Shards commit
#   ~2M positions at a time, and a restart skips every shard that already has
#   its artifact. Book slices are derived from the shard index so no two shards
#   ever mine the same openings.
# ---------------------------------------------------------------------------

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

TARGET=${TARGET:-10000000}
SHARD_POS=${SHARD_POS:-2000000}
NODES=${NODES:-5000}
PERGAME=${PERGAME:-80}
CONC=${CONC:-4}
NAME=${NAME:-oracle-$(date +%Y%m%d)}
KEEP_PGN=${KEEP_PGN:-1}

ROOT=$ARCHON_ROOT/data/nnue/selfplay/campaigns/$NAME
mkdir -p "$ROOT"
OPS_LOG=${OPS_LOG:-$ROOT/campaign.log}; export OPS_LOG

# Kept positions per game, measured on the 60-game oracle smoke at 5k/5k:
# 74.7 candidates/game -> 57.5 kept after the D46 filter (23.0 % drop).
# Used only to size a shard's game count; the real count is whatever comes out.
KEPT_PER_GAME=${KEPT_PER_GAME:-57.5}
GAMES_PER_SHARD=$(python3 -c "print(int($SHARD_POS / $KEPT_PER_GAME))")
NSHARDS=$(python3 -c "import math;print(math.ceil($TARGET / $SHARD_POS))")

BOOK=$(asset engines/books/UHO_Lichess_4852_v1.epd)
BOOKLINES=$(wc -l < "$BOOK")

log "CAMPAIGN $NAME"
log "  target=$TARGET kept positions in $NSHARDS shards of ~$SHARD_POS"
log "  ~$GAMES_PER_SHARD games/shard at $NODES nodes, pergame=$PERGAME, conc=$CONC"
log "  book=$(basename "$BOOK") $BOOKLINES lines"
log "  out=$ROOT"

# Each shard needs a disjoint book slice. GAMES_PER_SHARD openings each, laid
# end to end from a campaign-specific offset so two campaigns do not collide
# either. Wraps if a campaign is large enough to exhaust the book (2.6 M lines
# = ~150 M positions at this yield), and logs when it does.
BASE=$(( 0x$(echo -n "$NAME" | sha256sum | cut -c1-8) % BOOKLINES ))

total_kept=0
completed=0
for i in $(seq 0 $((NSHARDS - 1))); do
    SHARD=$(printf 'shard-%04d' "$i")
    SDIR=$ROOT/$SHARD

    if [ -s "$SDIR/corpus.shuf.bin" ] && [ -s "$SDIR/shard.json" ]; then
        n=$(python3 -c "import json;print(json.load(open('$SDIR/shard.json')).get('records',0))" 2>/dev/null || echo 0)
        total_kept=$((total_kept + n)); completed=$((completed + 1))
        log "$SHARD: done ($n records), skipping"
        continue
    fi

    span=$(( GAMES_PER_SHARD + 16 ))
    start=$(( (BASE + i * span) % (BOOKLINES - span) + 1 ))
    [ $(( BASE + i * span )) -ge $(( BOOKLINES - span )) ] && \
        log "  NOTE: book wrapped at shard $i — openings repeat from here"

    log "$SHARD: generating (~$GAMES_PER_SHARD games, book line $start)"
    # A shard holds CONC concurrent games x 2 SF processes; refuse rather than
    # join an OOM cascade on a box that has been OOM-killed repeatedly.
    hold_for_mem $(( 800 + CONC * 250 )) 120

    MODE=oracle GAMES=$GAMES_PER_SHARD GAMENODES=$NODES PERGAME=$PERGAME \
    CONC=$CONC INLINE=1 MIN_ROWS=1000 TAG="campaigns/$NAME/$SHARD" \
    BOOKSTART=$start \
        bash "$ARCHON_ROOT/tools/ops/data/run_pipeline.sh" >>"$ROOT/campaign.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$SDIR/corpus.shuf.bin" ]; then
        log "$SHARD: FAILED (rc=$rc) — leaving intermediates for diagnosis, stopping"
        exit 4
    fi

    # --- merge every stage's metadata into ONE file, then reap ---------------
    python3 - "$SDIR" <<'PYEOF'
import json, pathlib, sys, os
d = pathlib.Path(sys.argv[1])
def load(name):
    p = d / name
    try: return json.loads(p.read_text())
    except Exception: return None
out = dict(
    manifest = load("manifest.json"),
    extract  = load("positions.tsv.stats.json"),
    label    = load("labelled.tsv.stats.json"),
    corpus   = load("corpus.meta.json"),
    census   = load("census.json"),
)
out["records"] = (out.get("corpus") or {}).get("records", 0)
out["sha256"]  = (out.get("corpus") or {}).get("sha256")
(d / "shard.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["records"])
PYEOF

    if [ "$KEEP_PGN" = "1" ] && [ -s "$SDIR/games.pgn" ]; then
        zstd -19 -q --rm "$SDIR/games.pgn" -o "$SDIR/games.pgn.zst" 2>/dev/null \
            || gzip -9 "$SDIR/games.pgn"
    else
        rm -f "$SDIR/games.pgn"
    fi
    # Everything else is re-derivable from games.pgn.zst at zero game cost.
    rm -f "$SDIR"/positions.tsv "$SDIR"/positions.tsv.stats.json \
          "$SDIR"/labelled.tsv "$SDIR"/labelled.tsv.dropped.tsv \
          "$SDIR"/labelled.tsv.stats.json "$SDIR"/corpus.txt \
          "$SDIR"/corpus.txt.projstats.json "$SDIR"/corpus.bin \
          "$SDIR"/manifest.json "$SDIR"/corpus.meta.json "$SDIR"/census.json \
          "$SDIR"/fastchess.out "$SDIR"/fastchess.log

    n=$(python3 -c "import json;print(json.load(open('$SDIR/shard.json')).get('records',0))")
    total_kept=$((total_kept + n)); completed=$((completed + 1))
    du_s=$(du -sh "$SDIR" 2>/dev/null | cut -f1)
    log "$SHARD: +$n records (total $total_kept / $TARGET), shard on disk $du_s"

    python3 - "$ROOT" "$NAME" "$TARGET" "$total_kept" "$completed" "$NSHARDS" <<'PYEOF'
import json, pathlib, sys
root, name, target, kept, done, n = sys.argv[1:7]
p = pathlib.Path(root) / "campaign.json"
p.write_text(json.dumps(dict(name=name, target=int(target), kept=int(kept),
                             shards_done=int(done), shards_total=int(n)),
                        indent=2) + "\n")
PYEOF

    [ "$total_kept" -ge "$TARGET" ] && { log "TARGET REACHED at $total_kept"; break; }
done

log "CAMPAIGN COMPLETE: $total_kept kept positions in $completed shards"
log "  merge:  python3 tools/ops/data/merge_shards.py $ROOT --out <corpus>.shuf.bin"
log "  ledger: python3 tools/ledger/ledger.py add-data ... (identity = merged sha256)"
