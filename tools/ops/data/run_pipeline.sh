#!/usr/bin/env bash
# tools/ops/data/run_pipeline.sh — the whole self-play corpus, end to end.
#
# PURPOSE
#   One entry point that runs stages 1-4 for one MODE and leaves a trainer-ready
#   corpus plus the provenance to explain every row in it. This is the script to
#   launch when the operating point is agreed; the study tools beside it are how
#   that operating point was chosen.
#
# USAGE
#   MODE=self GAMES=20000 bash tools/ops/data/run_pipeline.sh
#
#   Long runs belong in a transient SERVICE, not a scope (a scope dies with the
#   session and once killed a 5880-game screen at game 1043):
#     systemd-run --user --unit=dgself -p MemoryMax=6G -p MemorySwapMax=0 \
#       --working-directory=/home/nick/projects/chessengine \
#       -E MODE=self -E GAMES=20000 \
#       /home/nick/projects/chessengine/tools/ops/data/run_pipeline.sh
#
#   Environment (defaults are the measured operating point — see
#   docs/research/selfplay-data-design-2026-08-02.md):
#     MODE       self | sf18 | sf16        (default self)
#     GAMES      games to play             (default 2000)
#     GAMENODES  per-move budget in play   (default 25000)
#     ORACLE     labelling engine          (default engines/bin/stockfish18)
#     LABELNODES oracle budget per label   (default 25000)
#     PERGAME    positions sampled/game    (default 14)
#     CONC       fastchess concurrency     (default 3)
#     WORKERS    label workers             (default 5)
#     TAG        run label                 (default <mode>-<timestamp>)
#
# PRODUCES  data/nnue/selfplay/<TAG>/
#     games.pgn  manifest.json  positions.tsv  labelled.tsv
#     corpus.txt  corpus.bin  corpus.shuf.bin  corpus.meta.json
#
# STAGE BOUNDARIES ARE RESUMABLE: each stage skips if its output exists, so a
# killed run is restarted with the same command and picks up where it stopped.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

MODE=${MODE:-self}
GAMES=${GAMES:-2000}
GAMENODES=${GAMENODES:-25000}
ORACLE=${ORACLE:-$(asset engines/bin/stockfish18)}
LABELNODES=${LABELNODES:-25000}
PERGAME=${PERGAME:-14}
CONC=${CONC:-3}
WORKERS=${WORKERS:-5}
TAG=${TAG:-${MODE}-$(date +%Y%m%d-%H%M%S)}
DIR=$ARCHON_ROOT/data/nnue/selfplay/$TAG
PY=$ARCHON_ROOT/.venv/bin/python

mkdir -p "$DIR"
OPS_LOG=${OPS_LOG:-$DIR/pipeline.log}; export OPS_LOG
die() { log "FATAL: $*"; exit 1; }

log "PIPELINE tag=$TAG mode=$MODE games=$GAMES gamenodes=$GAMENODES"
log "  oracle=$(basename "$ORACLE") labelnodes=$LABELNODES pergame=$PERGAME"

# --- stage 1: play -----------------------------------------------------------
if [ ! -s "$DIR/games.pgn" ]; then
    MODE=$MODE GAMES=$GAMES NODES=$GAMENODES CONC=$CONC TAG=$TAG OUTDIR=$DIR \
        bash "$ARCHON_ROOT/tools/ops/data/gen_games.sh" || die "stage 1 failed"
else
    log "stage 1: games.pgn exists, skipping"
fi
[ -s "$DIR/games.pgn" ] || die "no games produced"

# --- stage 2: extract --------------------------------------------------------
if [ ! -s "$DIR/positions.tsv" ]; then
    "$PY" "$ARCHON_ROOT/tools/ops/data/pgn_positions.py" "$DIR/games.pgn" \
        --out "$DIR/positions.tsv" --per-game "$PERGAME" || die "stage 2 failed"
else
    log "stage 2: positions.tsv exists, skipping"
fi

# --- stage 3: label + D46 filter --------------------------------------------
if [ ! -s "$DIR/labelled.tsv" ]; then
    hold_for_mem 2500 60
    "$PY" "$ARCHON_ROOT/tools/ops/data/label_positions.py" "$DIR/positions.tsv" \
        --out "$DIR/labelled.tsv" --oracle "$ORACLE" --nodes "$LABELNODES" \
        --workers "$WORKERS" || die "stage 3 failed"
else
    log "stage 3: labelled.tsv exists, skipping"
fi

# --- stage 4: export ---------------------------------------------------------
if [ ! -s "$DIR/corpus.shuf.bin" ] && [ ! -s "$DIR/corpus.bin" ]; then
    bash "$ARCHON_ROOT/tools/ops/data/export_corpus.sh" "$DIR/labelled.tsv" \
        --out "$DIR/corpus" || die "stage 4 failed"
else
    log "stage 4: corpus exists, skipping"
fi

# --- census + next steps -----------------------------------------------------
"$PY" "$ARCHON_ROOT/tools/ops/data/census.py" "$DIR/labelled.tsv" \
    --json "$DIR/census.json" | tee -a "$OPS_LOG"

log "COMPLETE -> $DIR"
log "REMAINING MANUAL STEPS, both required before training:"
log "  1. D38 ledger row:  python3 tools/ledger/ledger.py add-data ... "
log "     (identity is the corpus sha256, already in corpus.meta.json)"
log "  2. SCORE_MULT: labels are raw ${ORACLE##*/} UCI cp. Per the data"
log "     contract §2 that is MULT 1.42 at WDL=0.3 / 1.94 at WDL=0 -- and the"
log "     AUTHORITY is tools/nnue/net_scale_check.sh on the first trained net,"
log "     not this comment. It is a hard pre-game gate; run it."
