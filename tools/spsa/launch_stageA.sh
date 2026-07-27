#!/usr/bin/env bash
# Stage-A SPSA, designed to run CONCURRENTLY with D48W training (user
# ruling 2026-07-27: V net + PGO binary; "be mindful not to OOM anything
# or slow down training too much"):
#   - fn20k games (load-immune both directions; GPU untouched)
#   - concurrency 6 workers (trainer keeps its 4 CPU workers + headroom)
#   - whole thing under a systemd unit with MemoryMax=4G
#   - PGO'd tune binary built here if missing (~6 min, niced)
# ~200 iters x 8 units x 4 pairs = 12.8k games ≈ 2.5-3.5h.
set -eu
cd "$(dirname "$0")/../.."
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
ENGINE=engines/bin/archon-d47v-tune-pgo
[ -e "$ENGINE" ] || {
  echo "[stageA] building PGO tune binary (~6 min)"
  nice -n 19 bash tools/build_pgo.sh sota,arch-d47v,tune archon-d47v-tune-pgo "$NET"
}
systemd-run --user --unit=archon-spsa-stageA --collect \
  --working-directory="$PWD" -p MemoryHigh=3G -p MemoryMax=4G \
  /bin/bash -c "bash tools/spsa/run_spsa_dist.sh \
    tools/spsa/campaigns/s2-stageA.json file:///tmp/spsa-stageA-ex \
    $ENGINE $NET 6 200 8 4 20000 && sleep infinity" \
  || true
echo "[stageA] unit archon-spsa-stageA up; watch tools/spsa/out/s2-stageA-*/coordinator.log"
echo "[stageA] verify gate afterwards: 10+0.1 SPRT vs untuned before anything lands in params.rs"
