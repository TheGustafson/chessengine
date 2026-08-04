#!/usr/bin/env bash
# tools/ops/data/gen_games.sh — generate ARCHON games as NNUE training material.
#
# PURPOSE
#   Stage 1 of the self-play data pipeline: play games and keep the PGN. The
#   PGN is the PROVENANCE ARTIFACT — every later stage (position extraction,
#   labelling, export) traces back to a (run-id, game-number, ply) triple, so a
#   bad label can always be walked back to the game that produced it. That
#   traceability is the thing the D38 ledger cannot give us today: it proves a
#   corpus came from a recipe, not which position came from which game.
#
# MODES (the three the owner asked for)
#   self   ARCHON vs ARCHON            — cheapest positions/hour, our own distribution
#   sf18   ARCHON vs engines/bin/stockfish18
#   sf16   ARCHON vs engines/bin/stockfish16
#   oracle ORACLE vs ORACLE (eca43a97)  — every ply is a position the LABELLING
#          engine itself searched, so label density per game is 100 %. Highest
#          label quality and the strongest WDL, at the cost of sampling
#          Stockfish's position distribution rather than our own.
#
# USAGE
#   MODE=self  GAMES=200 NODES=25000 bash tools/ops/data/gen_games.sh
#   MODE=sf18  GAMES=200 TAG=pilot   bash tools/ops/data/gen_games.sh
#
#   Environment:
#     MODE      self | sf18 | sf16 | oracle      (default self)
#     GAMES     total games                      (default 200)
#     NODES     per-move node budget, both sides (default 25000)
#     CONC      fastchess concurrency            (default 4)
#     BOOK      opening book                     (default UHO_Lichess_4852_v1.epd)
#     BOOKSTART 1-based book line to start at    (default random by SRAND)
#     SRAND     book randomisation seed          (default derived from TAG)
#     ADJ       1 = adjudicate, 0 = play out     (default 1)
#     TAG       run label                        (default <mode>-<date>)
#     OUTDIR    default data/nnue/selfplay/<TAG>
#
# PRODUCES  $OUTDIR/games.pgn  +  $OUTDIR/manifest.json (full provenance)
#
# ---------------------------------------------------------------------------
# WHY THE FLAGS ARE WHAT THEY ARE — each of these was a decision, not a default.
#
#   nodes=N tc=inf (NOT a real clock).  Node-limited play is reproducible and
#     LOAD-IMMUNE. This box carries a GPU trainer and a 1500-game match; a
#     wall-clock datagen run would silently produce weaker games whenever the
#     box got busy, and the WDL labels would inherit that variation. Nodes cost
#     nothing in fairness here because both sides get the same budget.
#
#   -repeat / -games 2 ONLY in the vs-Stockfish modes.  With two IDENTICAL
#     deterministic engines, playing an opening with colours reversed produces
#     the SAME GAME. MEASURED 2026-08-02, not assumed: 8 games under
#     `-games 2 -rounds 4 -repeat` yielded 4 distinct (opening, move-list)
#     signatures — every pair bit-identical, 127/140/87/143 plies. So in `self`
#     mode -repeat is an exact 2x waste of CPU and contributes ZERO new
#     positions. In sf18/sf16 mode the two engines differ, colour balance is
#     real, and the pairing is worth it.
#
#   -srand + start=  give a REPRODUCIBLE, NON-OVERLAPPING book slice, so two
#     runs of this script never mine the same openings twice.
#
#   ADJUDICATION IS A FRAME TRAP, and this is why resign is twosided.
#     fastchess reads the resign/draw threshold against each engine's PRINTED
#     score, and OUR printed scores are ~2.07x Stockfish's printed cp
#     (uciprobe.ARCHON_PER_SF_PRINTED). A one-sided `score=1000` therefore fires
#     for ARCHON at ~483 SF-cp and for Stockfish at 1000 SF-cp — a systematic,
#     colour-correlated truncation bias that would land straight in the WDL
#     column. `twosided=true` requires BOTH engines to agree before the game is
#     called, which removes the asymmetry. The draw rule is likewise two-sided
#     by construction (it needs both scores inside the band).
#     Thresholds are quoted in ARCHON frame and halved for a pure-SF pair.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

MODE=${MODE:-self}
GAMES=${GAMES:-200}
NODES=${NODES:-25000}
CONC=${CONC:-4}
ADJ=${ADJ:-1}
TAG=${TAG:-${MODE}-$(date +%Y%m%d-%H%M%S)}
BOOK=${BOOK:-$(asset engines/books/UHO_Lichess_4852_v1.epd)}
OUTDIR=${OUTDIR:-$ARCHON_ROOT/data/nnue/selfplay/$TAG}
FASTCHESS=${FASTCHESS:-$(asset engines/bin/fastchess)}
ENGINE=${ENGINE:-$(asset engines/canonical/archon)}
NET=${NET:-$(asset engines/canonical/net.cenn)}

# SRAND: deterministic from TAG unless given, so a rerun of the same TAG mines
# the same openings and a new TAG mines different ones.
SRAND=${SRAND:-$(( 0x$(echo -n "$TAG" | sha256sum | cut -c1-8) % 2000000000 ))}

case "$MODE" in
    self) OPP=$ENGINE ; OPPNET=$NET ;;
    sf18) OPP=$(asset engines/bin/stockfish18) ; OPPNET="" ;;
    sf16) OPP=$(asset engines/bin/stockfish16) ; OPPNET="" ;;
    # ORACLE vs ORACLE. Both sides are the labelling engine, so EVERY ply of
    # every game is a position the oracle itself searched — the maximum-density
    # source of oracle-quality labels with a real result attached.
    oracle) ENGINE=$(asset engines/bin/stockfish-eca43a97) ; OPP=$ENGINE ; OPPNET="" ; NET="" ;;
    *) log "FATAL: MODE must be self|sf18|sf16|oracle (got '$MODE')"; exit 2 ;;
esac

[ -x "$ENGINE" ]    || { log "FATAL: engine not executable: $ENGINE"; exit 2; }
[ -x "$OPP" ]       || { log "FATAL: opponent not executable: $OPP"; exit 2; }
# oracle mode has no ARCHON side, so no .cenn to check.
[ "$MODE" = oracle ] || [ -s "$NET" ] || { log "FATAL: net missing: $NET"; exit 2; }
[ -x "$FASTCHESS" ] || { log "FATAL: fastchess missing"; exit 2; }
[ -s "$BOOK" ]      || { log "FATAL: book missing: $BOOK"; exit 2; }

# Each concurrent game holds two engines; ARCHON with a 100 MB net plus hash is
# the heavy side. Refuse rather than join an OOM cascade — this box has been
# OOM-killed six times in two days.
need_mem $(( 500 + CONC * 350 )) || exit 3

mkdir -p "$OUTDIR"
PGN="$OUTDIR/games.pgn"
OPS_LOG=${OPS_LOG:-$OUTDIR/gen.log}; export OPS_LOG

# Book slice: BOOKSTART lets successive runs take disjoint regions of a
# 2.6M-line book. Default derives from SRAND so it is reproducible but spread.
BOOKLINES=$(wc -l < "$BOOK")
BOOKSTART=${BOOKSTART:-$(( SRAND % (BOOKLINES - GAMES - 10) + 1 ))}

# Pairing: see the -repeat note in the header. `oracle` joins `self` here for
# the same measured reason — two identical deterministic engines replay the
# same game with colours reversed, so -repeat is an exact 2x CPU waste that
# contributes zero new positions.
if [ "$MODE" = "self" ] || [ "$MODE" = "oracle" ]; then
    PAIRING=(-games 1 -rounds "$GAMES")
else
    PAIRING=(-games 2 -rounds $(( (GAMES + 1) / 2 )) -repeat)
fi

# Adjudication thresholds are in the PLAYERS' PRINTED FRAME. ARCHON now
# normalises its printed score into STOCKFISH'S frame at the UCI boundary
# (`uci.rs::to_cp`), so every mode shares ONE set of constants and the
# two-frame split this block used to carry is gone. Left as it was, the
# ARCHON-frame 1600 would be a 2.07x LOOSER bar than intended for every
# ARCHON-playing mode — resign firing far later, longer games, and a corpus
# with a different endgame distribution that still looked identically
# generated.
#
# 773 SF-cp is exactly what the old ARCHON-frame 1600 always meant, kept rather
# than rounded so corpora from before and after the normalisation stay
# distributionally comparable.
ADJARGS=()
if [ "$ADJ" = "1" ]; then
    RESIGN_SCORE=773; DRAW_SCORE=10           # SF printed frame, ALL modes
    ADJARGS=(
        -resign movecount=4 score="$RESIGN_SCORE" twosided=true
        -draw   movenumber=40 movecount=8 score="$DRAW_SCORE"
        -maxmoves 250
    )
fi

OPPNAME=$(basename "$OPP")
[ "$MODE" != "self" ] && [ "$MODE" != "oracle" ] && OPPNAME=$(echo "uci" | "$OPP" 2>/dev/null | grep -m1 -i '^id name' | cut -d' ' -f3- | tr ' ' '_')

log "GEN mode=$MODE games=$GAMES nodes=$NODES conc=$CONC adj=$ADJ tag=$TAG"
log "  book=$(basename "$BOOK") start=$BOOKSTART srand=$SRAND"
log "  out=$PGN"

# --- provenance manifest, written BEFORE the run so a crashed run is still
# --- attributable. Hashes are of the exact bytes that played the games.
python3 - "$OUTDIR/manifest.json" <<PYEOF
import hashlib, json, subprocess, sys, os, datetime
def _git_head():
    try:
        return subprocess.run(["git","-C","$ARCHON_ROOT","rev-parse","HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except (FileNotFoundError, OSError):
        return None
def sha(p):
    if not p or not os.path.exists(p): return None
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
m = dict(
    tag="$TAG", mode="$MODE", generated_utc=datetime.datetime.utcnow().isoformat()+"Z",
    games=$GAMES, nodes=$NODES, concurrency=$CONC, adjudication=bool($ADJ),
    engine=dict(path="$ENGINE", sha256=sha("$ENGINE"),
                net="$NET", net_sha256=sha("$NET")),
    opponent=dict(mode="$MODE", name="$OPPNAME", path="$OPP", sha256=sha("$OPP")),
    book=dict(path="$BOOK", sha256=sha("$BOOK"), start=$BOOKSTART, srand=$SRAND,
              lines=$BOOKLINES),
    fastchess=subprocess.run(["$FASTCHESS","--version"],capture_output=True,
                             text=True).stdout.strip().split("\n")[0],
    # OPTIONAL field, and it must not be able to kill the manifest. Outside a
    # git checkout `rev-parse` merely fails; with no git binary at all —
    # e.g. a cloud worker running from the payload tarball — subprocess.run
    # RAISES FileNotFoundError, the whole heredoc dies, and manifest.json is
    # never written. That took down stage 3, which reads the manifest to
    # confirm the oracle played both sides. Provenance degrades to null here;
    # it does not take the run with it.
    git_head=_git_head(),
)
json.dump(m, open(sys.argv[1],"w"), indent=2)
print("manifest ->", sys.argv[1])
PYEOF

if [ "$MODE" = "oracle" ]; then
    # Both sides are the oracle binary and neither takes a .cenn. Distinct
    # names so the PGN records which side moved.
    ENGARGS=(-engine cmd="$ENGINE" name=oracleA proto=uci option.Threads=1)
    OPPARGS=(-engine cmd="$OPP"    name=oracleB proto=uci option.Threads=1)
elif [ "$MODE" = "self" ]; then
    ENGARGS=(-engine cmd="$ENGINE" name=archon  proto=uci option.EvalFile="$(readlink -f "$NET")" option.Threads=1)
    OPPARGS=(-engine cmd="$OPP"    name=archon2 proto=uci option.EvalFile="$(readlink -f "$NET")" option.Threads=1)
else
    ENGARGS=(-engine cmd="$ENGINE" name=archon proto=uci option.EvalFile="$(readlink -f "$NET")" option.Threads=1)
    OPPARGS=(-engine cmd="$OPP" name="$OPPNAME" proto=uci option.Threads=1)
fi

START=$(date +%s)
"$FASTCHESS" \
  "${ENGARGS[@]}" \
  "${OPPARGS[@]}" \
  -each nodes="$NODES" tc=inf option.Hash=64 \
  -openings file="$BOOK" format=epd order=random start="$BOOKSTART" \
  "${PAIRING[@]}" \
  "${ADJARGS[@]}" \
  -srand "$SRAND" \
  -concurrency "$CONC" \
  -pgnout file="$PGN" notation=uci nodes=true \
  -recover -ratinginterval 0 -log file="$OUTDIR/fastchess.log" level=warn \
  > "$OUTDIR/fastchess.out" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))

NG=$(grep -c '^\[Event ' "$PGN" 2>/dev/null || echo 0)
log "done rc=$RC games=$NG elapsed=${ELAPSED}s rate=$(python3 -c "print(f'{$NG*3600/max($ELAPSED,1):.0f}')") games/h"

python3 - "$OUTDIR/manifest.json" "$NG" "$ELAPSED" <<'PYEOF'
import json, sys
p = sys.argv[1]; m = json.load(open(p))
m["result"] = dict(games_written=int(sys.argv[2]), elapsed_s=int(sys.argv[3]),
                   games_per_hour=round(int(sys.argv[2])*3600/max(int(sys.argv[3]),1),1))
json.dump(m, open(p,"w"), indent=2)
PYEOF

grep -A5 'Results of' "$OUTDIR/fastchess.out" 2>/dev/null | tail -6 | tee -a "$OPS_LOG"
exit $RC
