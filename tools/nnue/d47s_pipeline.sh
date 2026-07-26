#!/usr/bin/env bash
# d47s_pipeline.sh — MASTER-PLAN v5.5 steps 1→3 for the finished D47S run,
# one command per stage (staged 2026-07-26 while D47S trained; see the
# value-foundation-state 04:0x plan entry).
#
#   bash tools/nnue/d47s_pipeline.sh pack       # quant -> .cenn + nocpol copy
#   bash tools/nnue/d47s_pipeline.sh calibrate  # HARD GATE: network_scale
#   bash tools/nnue/d47s_pipeline.sh q1         # 300g fixed-node vs D47P (value-only)
#   bash tools/nnue/d47s_pipeline.sh et         # 200g 10+0.1 vs D47P (IDLE BOX)
#   bash tools/nnue/d47s_pipeline.sh sf         # sf_baseline.sh both legs (IDLE BOX)
#
# ORDER IS LOAD-BEARING: calibrate before ANY game (user hard gate — the
# v1.5 target transform changed what the output MEANS; every cp margin
# reads the old scale). q1 before et (cheap verdict first). sf last.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?pack|calibrate|q1|et|sf}"
EXP=data/nnue/experiments/d47s-v15
ID=d47s-v15
CENN=data/nnue/nets/${ID}.cenn
NOCPOL=data/nnue/nets/screens/${ID}-nocpol.cenn
REF_NET=data/nnue/nets/screens/d47p-vrepair2-sb120-ema-nocpol.cenn
BOOK=engines/books/UHO_Lichess_4852_v1.epd

case "$STAGE" in
pack)
  Q=$(ls -t "$EXP"/${ID}-sb240-ema-quant.bin "$EXP"/${ID}-final-quant.bin 2>/dev/null | head -1)
  [ -n "${Q:-}" ] || { echo "no final quant checkpoint in $EXP"; exit 1; }
  echo "packing $Q"
  .venv/bin/python tools/nnue/netpack.py "$Q" "$CENN" d40
  cp "$CENN" "$NOCPOL"        # screens copy: NO .cpol sibling -> policy off
  ls -la "$CENN" "$NOCPOL"
  ;;
calibrate)
  bash tools/nnue/calibrate_network_scale.sh "$CENN" "$REF_NET" 5
  # keep the screens copy in sync with the patched header
  cp "$CENN" "$NOCPOL"
  bash tools/nnue/net_scale_check.sh "$NOCPOL" "$REF_NET" engines/bin/archon-d47 | tail -2
  ;;
q1)
  OUT=data/nnue/experiments/settle-d47s-q1; mkdir -p "$OUT"
  engines/bin/fastchess \
    -engine cmd=engines/bin/archon-d47 name=d47s "option.EvalFile=$NOCPOL" \
    -engine cmd=engines/bin/archon-d47 name=d47p "option.EvalFile=$REF_NET" \
    -each proto=uci tc=inf nodes=50000 option.Threads=1 option.Hash=16 \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds 150 -concurrency 4 -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -srand 20260726 -pgnout file="$OUT/games.pgn" notation=lan \
    -log file="$OUT/fastchess.log" level=warn 2>&1 | tail -22 | tee "$OUT/stdout.log"
  ;;
et)
  OUT=data/nnue/experiments/settle-d47s-et; mkdir -p "$OUT"
  engines/bin/fastchess \
    -engine cmd=engines/bin/archon-d47 name=d47s "option.EvalFile=$NOCPOL" \
    -engine cmd=engines/bin/archon-d47 name=d47p "option.EvalFile=$REF_NET" \
    -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds 100 -concurrency 4 -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -srand 20260726 -pgnout file="$OUT/games.pgn" notation=lan \
    -log file="$OUT/fastchess.log" level=warn 2>&1 | tail -22 | tee "$OUT/stdout.log"
  ;;
sf)
  bash tools/nnue/sf_baseline.sh "$CENN" d47s-post-v15 200 4
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
