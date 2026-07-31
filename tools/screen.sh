#!/usr/bin/env bash
# High-power fixed-node net screen — the instrument our net work has been missing.
#
# WHY THIS EXISTS (2026-07-29). Every net decision tonight was read off 120-game
# screens returning +/-33..53 Elo, which cannot resolve the effects we care about.
# D48Y2's whole trend (sb90 -43.7, sb170 -11.6, sb190 -40.7) sits inside one
# error bar. The training-gap study filed this as G0: "our net-screening
# instrument is ~100x underpowered, and the fix is cheap" — Reckless screens
# EVERY net at 25,000 fixed nodes over ~40,000 games (+/-2 Elo) before it ever
# reaches a time control.
#
# The bar is CALIBRATED AGAINST OUR OWN OBSERVED SCREENS, not a textbook rule:
# our 120-game runs returned ~+/-35, so k = 35*sqrt(120) = 383 and
#
#     bar ~= 383 / sqrt(n)
#
# Cross-check: that predicts +/-1.9 at n=40,000, and Reckless report +/-2.
# (A generic "1600/sqrt(n)" rule I tried first was wrong by 4.2x. The data won.)
#
#     n =  1,470  ->  +/-10 Elo
#     n =  5,880  ->  +/- 5 Elo     <- default here
#     n = 16,333  ->  +/- 3 Elo
#
# Fixed NODES, so the result is contention-immune and concurrency is free.
#
# Usage:
#   tools/screen.sh CAND.cenn REF.cenn [GAMES] [NODES]
#   GAMES default 5880 (+/-5 Elo), NODES default 25000 (Reckless's screen budget)
#
# Both sides run the SAME engine binary with different EvalFile, so the net is
# the only variable. EvalFile is passed EXPLICITLY: it defaults to <empty>, and
# an empty EvalFile silently evaluates PSQT (~2.13 Mnps instead of ~1.0) — which
# would measure nothing at all.
set -euo pipefail

CAND="${1:?candidate .cenn}"
REF="${2:?reference .cenn}"
GAMES="${3:-5880}"
NODES="${4:-25000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -r "$CAND" ] || { echo "error: candidate not readable: $CAND" >&2; exit 1; }
[ -r "$REF"  ] || { echo "error: reference not readable: $REF" >&2; exit 1; }

ENGINE="${ENGINE:-$ROOT/engines/canonical/archon}"
BOOK="${BOOK:-$ROOT/engines/books/UHO_Lichess_4852_v1.epd}"
FASTCHESS="${FASTCHESS:-$ROOT/engines/bin/fastchess}"
HASH="${HASH:-16}"
# CONCURRENCY IS MEMORY-BOUND, NOT CPU-BOUND — learned the hard way (2026-07-29,
# oom-killed at conc=14 under MemoryMax=4G). Each engine process loads the net
# into PRIVATE heap, so the cost is
#
#     2 * CONC * sizeof(net)   ~=  2 * CONC * 138 MB  for a 1280 net
#
# i.e. conc=14 wants ~3.9 GB of nets before hash or overhead. Derive the cap
# from free memory and the actual net size rather than from core count, and take
# the min with cores-2.
_netmb=$(( ( $(stat -c %s "$CAND") + $(stat -c %s "$REF") ) / 2 / 1048576 + 1 ))
_freemb=$(free -m | awk 'NR==2 {print $7}')
# keep 1.5 GB headroom, budget net+hash per process pair
_memcap=$(( (_freemb - 1536) / (2 * (_netmb + ${HASH:-16})) ))
CONC="${CONC:-$(( _memcap < $(nproc) - 2 ? _memcap : $(nproc) - 2 ))}"
[ "$CONC" -ge 1 ] || { echo "error: not enough free memory to run even one pair" >&2; exit 1; }

# ---------------------------------------------------------------------------
# LOAD GUARD (2026-07-30). A screen whose engine cannot READ a net does not
# crash -- the EvalFile load fails, NNUE silently switches off, and that side
# plays on PSQT. It then reports a perfectly well-formed result with tight error
# bars. Measured: D49 packed as CENN-v5 (netpack with D40_THREAT_QUANT set)
# against engines/canonical/archon, which predates CEN5:
#     "ERROR EvalFile: CENN bad magic: [67,69,78,53]"  ([C,E,N,5])
#     -> 1466 games, 0 wins, 1461 losses, FIVE draws, Elo -1107.
# The giveaway was the draw count, not the Elo. Never rely on noticing that.
# Both nets must LOAD in the chosen ENGINE before a single game is played.
for _net in "$CAND" "$REF"; do
    _probe=$(printf 'uci\nsetoption name EvalFile value %s\nisready\nquit\n' \
                    "$(readlink -f "$_net")" | "$ENGINE" 2>&1)
    if printf '%s' "$_probe" | grep -qi 'ERROR EvalFile'; then
        echo "error: $ENGINE cannot load $_net" >&2
        printf '%s\n' "$_probe" | grep -i 'ERROR EvalFile' >&2
        echo "  -> that side would play on PSQT and the match would be meaningless." >&2
        echo "  -> use an ENGINE built from a tree that supports this container." >&2
        exit 1
    fi
    printf '%s' "$_probe" | grep -qi 'info string eval: NNUE' || {
        echo "error: $ENGINE did not report NNUE active for $_net" >&2; exit 1; }
done

OUT="${OUT:-$ROOT/engines/runs/screen-$(basename "${CAND%.cenn}")}"
ROUNDS=$(( GAMES / 2 ))

mkdir -p "$OUT"
python3 - "$GAMES" <<'PY'
import math, sys
n = int(sys.argv[1])
print(f"screen: {n} games -> expected bar ~+/-{383/math.sqrt(n):.1f} Elo "
      f"(calibrated k=383 from our own 120-game screens)")
PY
echo "candidate : $CAND"
echo "reference : $REF"
echo "nodes=$NODES concurrency=$CONC rounds=$ROUNDS  out=$OUT"

exec "$FASTCHESS" \
    -engine cmd="$ENGINE" name=CAND option.Hash="$HASH" option.Threads=1 \
        option.EvalFile="$(readlink -f "$CAND")" \
    -engine cmd="$ENGINE" name=REF  option.Hash="$HASH" option.Threads=1 \
        option.EvalFile="$(readlink -f "$REF")" \
    -each proto=uci nodes="$NODES" \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds "$ROUNDS" -maxmoves 250 \
    -concurrency "$CONC" -recover -ratinginterval 500 \
    -pgnout file="$OUT/games.pgn" \
    -log file="$OUT/fastchess.log" level=warn \
    2>&1 | tee "$OUT/run.out"
