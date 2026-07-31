#!/usr/bin/env bash
# Sequential probe chain, memory-gated: ONE heavy pass at a time, each pass
# waits for >= 3000 MB available (coordinator budget 2026-07-30; box is 15 GB
# with the D49 trainer live and swap exhausted).
set -u
cd "$(dirname "$0")/../../.."
LOG=data/nnue/nets/screens/reprgap
# coordinator envelope 2026-07-30: <=4 threads, <=3 GB, sequential passes
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
gate() {
  while [ "$(free -m | awk 'NR==2{print $7}')" -lt 3000 ]; do
    echo "[gate] available < 3000 MB, waiting"; sleep 30
  done
}
# wait for the TSV extraction/targets pipeline if it is still running
until grep -q 'TSV-PIPELINE-DONE' "$LOG/tsv-pipeline.log" 2>/dev/null; do
  sleep 20
done
for step in "gram mix" "gram tsvfilt_tr" "report mix heldout" \
            "report tsvfilt_tr tsvfilt_ev" "report tsvfilt_tr tsvraw"; do
  # idempotent: skip a gram whose output exists
  set -- $step
  if [ "$1" = gram ] && [ -s "$LOG/gram-$2.npz" ]; then
    echo "[chain] skip $step (exists)"; continue
  fi
  gate
  echo "[chain] $step"
  nice -n 10 .venv/bin/python tools/nnue/reprgap/probes.py $step \
    >> "$LOG/probe-chain.log" 2>&1 || { echo "[chain] FAILED: $step"; exit 1; }
done
echo PROBE-CHAIN-DONE
