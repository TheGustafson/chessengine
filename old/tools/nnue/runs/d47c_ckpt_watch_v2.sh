#!/usr/bin/env bash
# d47c self-measuring checkpoints v2 — RAM-tight edition (from sb30).
# Changes vs v1: Elo matches at concurrency 1 (games are ~ms at 5k nodes;
# parallelism was costing ~1.6GB we don't have), guard lowered to 1.2GB with
# a 12x60s retry window instead of a single skip, and the Elo step is
# resume-safe (skips if summary.json already exists).
set -uo pipefail
cd /home/nick/projects/chessengine
CKPT=data/nnue/experiments/d47c-1280-240sb/ckpt
PACK_ENV=(D40_ACC=1280 D40_L2=128 D40_L3=32 D40_BUCKETS=8 D40_HALFKA_ONLY=0 D40_POLICY=1)
INC=data/nnue/nets/d40-uho-v1-240sb.cenn
tb_emit() {
  uv run --no-project --with tensorboardX python - "$1" "$2" "$3" <<'PYEOF'
import sys
from tensorboardX import SummaryWriter
w = SummaryWriter("data/nnue/tb/d47c/checkpoints")
w.add_scalar(sys.argv[2], float(sys.argv[3]), int(sys.argv[1]))
w.flush()
PYEOF
}
avail_gb() { awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo; }
for sb in $(seq 30 5 240); do
  Q="$CKPT/d47c-1280-240sb-sb${sb}-quant.bin"
  until [ -s "$Q" ]; do sleep 60; done
  sleep 8
  NET="data/nnue/nets/d47c-sb${sb}.cenn"
  [ -s "$NET" ] || env "${PACK_ENV[@]}" uv run --no-project --with numpy python tools/nnue/netpack.py "$Q" "$NET" d40 >/dev/null 2>&1
  ACC=$(ENG=engines/bin/archon-d47 bash tools/nnue/policy_accuracy_probe.sh "$NET" 2>/dev/null | grep -oE '= [0-9.]+%' | grep -oE '[0-9.]+' | head -1 || echo "")
  [ -n "$ACC" ] && tb_emit "$sb" "policy/top1_accuracy_pct" "$ACC" && echo "[d47c sb${sb}] top1 ${ACC}%"
  SCALE=$(REF_ENGINE=engines/bin/archon-sota-native bash tools/nnue/net_scale_check.sh "$NET" "$INC" engines/bin/archon-d47 2>/dev/null | grep -oE 'median\(cand/ref\)=[0-9.]+' | grep -oE '[0-9.]+$' || echo "")
  [ -n "$SCALE" ] && tb_emit "$sb" "value/scale_vs_incumbent" "$SCALE"
  if [ $((sb % 30)) -eq 0 ]; then
    OUTDIR="data/nnue/experiments/ckpt-elo-d47c-sb${sb}"
    if [ -s "$OUTDIR/summary.json" ]; then
      echo "[d47c sb${sb}] elo already measured, skipping"
    else
      RAN=0
      for try in $(seq 1 12); do
        if awk "BEGIN{exit !($(avail_gb) >= 1.2)}"; then
          python3 tools/nnue/eval_match.py \
            --candidate "$NET" --baseline "$INC" \
            --nodes 5000 --games 100 --concurrency 1 \
            --candidate-engine engines/bin/archon-d47 --baseline-engine engines/bin/archon-sota-native \
            --openings engines/books/UHO_Lichess_4852_v1.epd \
            --output "$OUTDIR" >/dev/null 2>&1 || true
          RAN=1; break
        fi
        sleep 60
      done
      if [ "$RAN" = 1 ] && [ -s "$OUTDIR/summary.json" ]; then
        read -r ELO ERR W D L <<< "$(python3 -c "import json;d=json.load(open('$OUTDIR/summary.json'));print(d['candidate_elo'],d['candidate_elo_se'],d.get('candidate_wins','?'),d.get('draws','?'),d.get('candidate_losses','?'))")"
        tb_emit "$sb" "elo/fixed_node_vs_incumbent" "$ELO"
        echo "[d47c sb${sb}] ELO $(printf '%.1f' "$ELO") +/- $(printf '%.1f' "$ERR")  W-D-L ${W}-${D}-${L} (100g fixed-node 5k)"
      else
        echo "[d47c sb${sb}] elo SKIPPED (RAM $(avail_gb)G after 12 retries)"
      fi
    fi
  fi
done
echo "[d47c] ALL CHECKPOINTS MEASURED (v2)"
