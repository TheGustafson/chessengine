#!/usr/bin/env bash
# tools/ops/match/vs_stockfish.sh — where are we against Stockfish.
#
# PURPOSE
#   The external anchor. Plays our engine (with a chosen net) against a
#   Stockfish binary, either at EQUAL NODES (quality per node, load-immune) or
#   at a REAL CLOCK (who actually wins, charges for speed). Ladder-position
#   measurement only — the merge gate is always our own previous version
#   (tools/ops/match/sprt.sh).
#
# USAGE
#   # equal nodes (default):
#   NET=data/nnue/tmp/canon/sfb-sb1648-756.cenn bash tools/ops/match/vs_stockfish.sh
#
#   # real clock:
#   TC=2+0.1 GAMES=150 SF=engines/bin/stockfish16 bash tools/ops/match/vs_stockfish.sh
#
#   Environment:
#     ENGINE   our binary   (default target-sota/release/chessengine, else canonical)
#     NET      .cenn to load                 (default engines/canonical/net.cenn)
#     SF       opponent binary               (default engines/bin/stockfish18)
#     TC       real clock, e.g. 2+0.1        (unset => equal-node mode)
#     NODES    per-move budget in node mode  (default 25000)
#     GAMES    default 1000   CONC default 3
#     BOOK     default engines/books/UHO_Lichess_4852_v1.epd
#
# PRODUCES
#   $ARCHON_D49/vs-sf-<net>-<tc-or-nodes>.out  plus a summary in the log.
#
# HOW TO READ THE RESULT
#
#   1. A NODE IS NOT A UNIVERSAL UNIT. Equal-node removes SPEED, which is the
#      point, but does not make the engines equivalent per node: different
#      pruning means each engine's node buys a different amount of tree. Read it
#      as "quality per node", never as shippable strength.
#
#   2. EQUAL-NODE AT A SHALLOW BUDGET FLATTERS US. Stockfish is tuned for far
#      deeper searches; 25k nodes is not its regime. If a deficit narrows as
#      nodes rise it was a shallow-search artifact — sweep NODES before quoting
#      any "our search is ahead" claim.
#
#   3. REAL-CLOCK RESULTS ARE LOAD-SENSITIVE. Equal-node is immune to a busy
#      box; a TC run is not. This machine usually carries a GPU trainer. Both
#      engines suffer, so the comparison stays fair, but the absolute number
#      reads low. Never run two TC matches concurrently.
#
#   4. THE SCORES ARE IN DIFFERENT FRAMES. Our printed scores are ~2.07x
#      Stockfish's printed cp. Never hand-compare eval output from the two
#      logs — only the game result is frame-free.
#
#   5. SMALL n IS A GLANCE, NOT A VERDICT. The bar is 383/sqrt(n): 150 games is
#      +/-31 Elo. Fine for placing yourself on a ladder, useless for ranking two
#      things that are close.
#
# WHY THIS TEST IS WORTH RUNNING
#   Swapping SF's NET into OUR search at equal nodes measured +90.7 Elo on a
#   provably identical 83,248-input feature space, locating the deficit in the
#   weights. This is the complementary measurement: our net in our search
#   against the whole opposing engine, so the gap includes their search too.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

# Binaries/books/nets are SHARED assets: resolve via asset(), which prefers
# ARCHON_ROOT then falls back to ARCHON_MAIN. A worktree has its own (nearly
# empty) engines/ dir, so $ARCHON_ROOT/engines/... silently misses.
ENGINE=${ENGINE:-$(asset target-sota/release/chessengine || asset engines/canonical/archon)}
NET=${NET:-$(asset engines/canonical/net.cenn)}
SF=${SF:-$(asset engines/bin/stockfish18)}
TC=${TC:-}
NODES=${NODES:-25000}
GAMES=${GAMES:-1000}
CONC=${CONC:-3}
BOOK=${BOOK:-$(asset engines/books/UHO_Lichess_4852_v1.epd)}
FASTCHESS=${FASTCHESS:-$(asset engines/bin/fastchess)}

ENGINE=$(asset "$ENGINE" || echo "$ENGINE")
NET=$(asset "$NET" || echo "$NET")
SF=$(asset "$SF" || echo "$SF")

[ -x "$ENGINE" ]    || { log "FATAL: our engine not executable: $ENGINE"; exit 2; }
[ -x "$SF" ]        || { log "FATAL: stockfish not executable: $SF"; exit 2; }
[ -s "$NET" ]       || { log "FATAL: net not found: $NET"; exit 2; }
[ -x "$FASTCHESS" ] || { log "FATAL: fastchess not found: $FASTCHESS"; exit 2; }
[ -s "$BOOK" ]      || { log "FATAL: book not found: $BOOK"; exit 2; }

TAG=$(basename "$NET" .cenn)
SFTAG=$(basename "$SF")
OUT="$ARCHON_D49/vs-${SFTAG}-${TAG}-${TC:-${NODES}n}.out"
OPS_LOG=${OPS_LOG:-$ARCHON_D49/vs_stockfish.log}; export OPS_LOG
SFNAME=$(echo "uci" | "$SF" 2>/dev/null | grep -m1 -i '^id name' | cut -d' ' -f3-)

log "ANCHOR: $(basename "$ENGINE") + $(basename "$NET")  vs  ${SFNAME:-$SFTAG}"
if [ -n "$TC" ]; then
    log "  TC=$TC (real clock, both sides)  games=$GAMES  conc=$CONC"
    log "  NB: real-clock results are LOAD-SENSITIVE; bar at n=$GAMES is +/-$(python3 -c "print(f'{383/($GAMES**0.5):.0f}')") Elo"
    MODE=(tc="$TC")
else
    log "  nodes=$NODES (both sides)  games=$GAMES  conc=$CONC"
    log "  NB: equal-node measures quality PER NODE, not shippable strength"
    MODE=(nodes="$NODES" tc=inf)
fi

"$FASTCHESS" \
  -engine cmd="$ENGINE" name=archon proto=uci option.EvalFile="$(readlink -f "$NET")" \
  -engine cmd="$SF" name="${SFNAME:-$SFTAG}" proto=uci \
  -each "${MODE[@]}" \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds $((GAMES / 2)) -repeat -concurrency "$CONC" \
  -recover -ratinginterval 25 > "$OUT" 2>&1

{ echo "=== vs ${SFNAME:-$SFTAG} (${TC:-${NODES}n}): $(basename "$NET") ==="
  grep -A5 'Results of' "$OUT" | tail -5; } | tee -a "$OPS_LOG"
log "done -> $OUT"
