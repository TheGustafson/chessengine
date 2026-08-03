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
#     ORACLE     labelling engine          (default engines/bin/stockfish-eca43a97)
#     LABELNODES oracle budget per label   (default 10000)
#     PERGAME    positions sampled/game    (default 80)
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
# These three are the STUDY'S RECOMMENDED OPERATING POINT, not the values the
# pilots happened to run at. They diverged once already: the header claimed
# "defaults are the measured operating point" while the code defaulted to
# stockfish18 / 25000 / 14 — i.e. the pilot settings, kept because they were
# typed first. That matters most for ORACLE, because the study went and
# measured the two candidates head to head and found eca43a97 is
# **+47.78 +/- 25.58 Elo stronger than stockfish18 at exactly this node
# budget**, at equal cost (112.7 vs 107.3 pos/s). A production run launched
# without explicit env would silently have labelled with the weaker engine, and
# nothing downstream would ever have said so.
ORACLE=${ORACLE:-$(asset engines/bin/stockfish-eca43a97)}
LABELNODES=${LABELNODES:-10000}   # §1.1: 5k-25k all defensible; 10k recommended
PERGAME=${PERGAME:-80}            # §6.1: largest cap that keeps the WDL mix unbiased
CONC=${CONC:-3}
WORKERS=${WORKERS:-5}
TAG=${TAG:-${MODE}-$(date +%Y%m%d-%H%M%S)}
# OUTDIR wins if given. This script used to compute DIR itself and forward
# OUTDIR only to stage 1, so a caller that set OUTDIR got its GAMES in one place
# and its CORPUS in another — and the caller's "did a corpus appear?" check then
# failed on a run that had actually succeeded. The cloud worker needs the whole
# run inside a scratch dir it can delete, so the whole run has to honour it.
DIR=${OUTDIR:-$ARCHON_ROOT/data/nnue/selfplay/$TAG}
# Interpreter is overridable so this script is not tied to THIS box. The cloud
# worker has no .venv — it runs system python3 with python-chess vendored onto
# PYTHONPATH — and a hardcoded path here is what made stage 2 die there.
PY=${PY:-$ARCHON_ROOT/.venv/bin/python}

mkdir -p "$DIR"
OPS_LOG=${OPS_LOG:-$DIR/pipeline.log}; export OPS_LOG
die() { log "FATAL: $*"; exit 1; }

log "PIPELINE tag=$TAG mode=$MODE games=$GAMES gamenodes=$GAMENODES"
log "  oracle=$(basename "$ORACLE") labelnodes=$LABELNODES pergame=$PERGAME"

# --- stage 1: play -----------------------------------------------------------
# EXISTENCE IS NOT COMPLETENESS. fastchess cannot resume, so a killed run leaves
# a PARTIAL games.pgn — and skipping on `-s` alone silently accepts however many
# games happened to finish. Measured: a campaign shard was stopped mid-generation
# to raise concurrency and restarted; stage 1 skipped, 18,742 of 34,782 games
# survived, and the shard came out at 54% of its target size with nothing
# reporting a problem. The per-game yield model was fine (58.04 kept/game against
# a 57.5 estimate); the game COUNT was silently halved.
#
# So count the games and only skip a PGN that actually holds them all.
if [ -s "$DIR/games.pgn" ]; then
    HAVE=$(grep -c '^\[Event' "$DIR/games.pgn" 2>/dev/null || echo 0)
    if [ "$HAVE" -lt "$GAMES" ]; then
        log "stage 1: games.pgn has $HAVE/$GAMES games (partial) — regenerating"
        rm -f "$DIR/games.pgn"
        # Downstream outputs derived from the partial PGN are now stale.
        rm -f "$DIR/positions.tsv" "$DIR/positions.tsv.stats.json" \
              "$DIR/labelled.tsv" "$DIR/labelled.tsv.stats.json" \
              "$DIR/labelled.tsv.dropped.tsv" "$DIR/corpus.txt" "$DIR/corpus.bin" \
              "$DIR/corpus.shuf.bin" "$DIR/corpus.meta.json"
    fi
fi
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
# INLINE=1 uses the score the oracle already produced during play instead of
# re-searching every position. Only valid when the ORACLE PLAYED BOTH SIDES
# (MODE=oracle); label_inline.py reads the manifest and refuses otherwise, so
# this cannot silently mislabel a mixed-engine run. It removes the whole
# labelling stage: 5.85 -> 2.90 core-h per 1M kept, measured.
if [ ! -s "$DIR/labelled.tsv" ]; then
    if [ "${INLINE:-0}" = "1" ]; then
        "$PY" "$ARCHON_ROOT/tools/ops/data/label_inline.py" "$DIR" \
            --out "$DIR/labelled.tsv" || die "stage 3 (inline) failed"
    else
        hold_for_mem 2500 60
        "$PY" "$ARCHON_ROOT/tools/ops/data/label_positions.py" "$DIR/positions.tsv" \
            --out "$DIR/labelled.tsv" --oracle "$ORACLE" --nodes "$LABELNODES" \
            --workers "$WORKERS" || die "stage 3 failed"
    fi
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
