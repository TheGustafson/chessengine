#!/usr/bin/env bash
# THE one-script distributed SPSA launcher (user directive 2026-07-26:
# "just have a script we run"). Local mode works TODAY (no GCP); gs://
# mode is the same command with a bucket URI once access exists —
# workers join from anywhere and mix freely with local ones.
#
#   bash tools/spsa/run_spsa_dist.sh <campaign.json> <exchange-uri> \
#        <engine> <net.cenn> [local_workers] [iters] [units] [pairs] [nodes]
#
#   # tonight-style local run, 8 workers, stage-A campaign:
#   bash tools/spsa/run_spsa_dist.sh tools/spsa/campaigns/s2-stageA.json \
#        file:///tmp/spsa-ex engines/bin/archon-d47v-tune \
#        /tmp/d47v-a4b2.cenn 8
#
#   # later, GCP: same but gs://archon-spsa/stageA and local_workers=0;
#   # spot VMs run worker.py via gcp/worker_startup.sh (see gcp/RUNBOOK.md).
#
# Engine must be a tune-feature build (exposes `tunables` + option.NAME):
#   bash tools/build_pgo.sh sota,arch-d47v,tune archon-d47v-tune <net>
# (PGO per the standing ruling; fn tuning is load-immune either way.)
set -eu
cd "$(dirname "$0")/../.."
CFG=${1:?campaign.json}
EX=${2:?exchange uri (file:///dir or gs://bucket/prefix)}
ENGINE=${3:?tune-feature engine build}
NET=${4:?eval file both sides play}
NWORK=${5:-8}
ITERS=${6:-400}
UNITS=${7:-8}
PAIRS=${8:-4}
NODES=${9:-20000}
# Exchange poll period. The 3s worker / 2s coordinator defaults are sized for
# a gs:// exchange (per-request cost, rate limits). On a file:// exchange a
# poll is a stat(), so the defaults just add ~3s of dead time to every
# iteration — at 8-game iterations that is a third of the wall clock.
POLL=${10:-3.0}
# Step size. This belongs at LAUNCH time, not in the campaign file, because
# the right value depends on the ITERATION BUDGET rather than on which
# parameters you are tuning: total movement as a fraction of a parameter's
# range is ~180 * r_end * E[z], independent of range width. Fishtest reaches
# ~13% of range at ~30k iterations with the 0.002 default; an overnight run
# here is ~2400 iterations, so it needs ~0.025-0.03 to explore comparably,
# and the 0.002 default would return the starting vector. Rule of thumb:
# r_end ~= 0.03 * (2400 / iters). Empty = leave the campaign/default alone.
R_END=${11:-}
# Real time control INSTEAD of fixed nodes. Empty = fixed nodes (the default,
# and the right choice while anything else is using the box). Set it only on a
# QUIET box: real TC is load-sensitive, which is exactly the property fixed
# nodes was chosen for. Why it matters at all — we tune at fn20k (~12 ply) and
# play at ~20, and many tuned constants are depth-scaled, so a fixed-node null
# may only mean "right for depth 12".
TC=${12:-}
NAME=$(python3 -c "import json,sys; print(json.load(open('$CFG'))['name'])")
OUT=tools/spsa/out/$NAME-$(date +%m%d-%H%M)
mkdir -p "$OUT"
case "$EX" in file://*) mkdir -p "${EX#file://}";; esac

# RESUME / DIRTY-EXCHANGE GUARD (added 2026-07-28 after this bit us).
# spsa_dist.py only resumes when passed --resume (spsa_dist.py:101). This
# launcher never passed it, so "just restart the coordinator" silently began a
# NEW run at k=1 -- and because the exchange still held the previous run's
# unit/result files, the fresh coordinator CONSUMED THEM: 1,014 stale results
# replayed at 11.8 games/s (impossible for real games), each attributed to a
# perturbation vector that did not produce it. Garbage gradients, no error, and
# state/theta-latest.json overwritten so the real vector was lost too.
RESUME_FLAG=""
if [ -e "${EX#file://}/state/theta-latest.json" ]; then
  RESUME_FLAG="--resume"
  echo "[run] RESUMING from ${EX#file://}/state/theta-latest.json"
elif [ -n "$(ls -A "${EX#file://}/results" 2>/dev/null)" ]; then
  echo "[run] REFUSING: $EX has results but no state/theta-latest.json." >&2
  echo "[run] A fresh run on a dirty exchange replays stale results as if they" >&2
  echo "[run] were fresh. Clear it first:  rm -rf ${EX#file://}" >&2
  exit 1
fi

echo "[run] coordinator -> $OUT (exchange $EX)"
nohup python3 tools/spsa/spsa_dist.py --config "$CFG" --engine "$ENGINE" \
  --eval-file "$NET" --exchange "$EX" --out "$OUT" --iters "$ITERS" \
  --units "$UNITS" --pairs "$PAIRS" --nodes "$NODES" \
  --book engines/books/UHO_Lichess_4852_v1.epd --poll "$POLL" ${RESUME_FLAG} \
  ${R_END:+--r-end "$R_END"} ${TC:+--tc "$TC"} \
  > "$OUT/coordinator.log" 2>&1 &
echo $! > "$OUT/coordinator.pid"

for i in $(seq 1 "$NWORK"); do
  # Caches hold the engine + net + book (~90MB each). On this box /tmp is a
  # RAM-backed tmpfs, so N worker caches there are N x 90MB of memory taken
  # from the trainer — keep them on disk.
  nohup nice -n 10 python3 tools/spsa/worker.py --exchange "$EX" \
    --id "local-$i" --cache "$HOME/.cache/spsa-workers/w$i" --poll "$POLL" \
    > "$OUT/worker-$i.log" 2>&1 &
  echo $! >> "$OUT/workers.pid"
done
echo "[run] $NWORK local workers up. Watch: tail -f $OUT/coordinator.log"
echo "[run] stop early: touch STOP on the exchange (control/STOP)"
