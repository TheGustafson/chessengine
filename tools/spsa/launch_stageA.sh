#!/usr/bin/env bash
# Stage-A SPSA, designed to run CONCURRENTLY with the D48X training run
# (user ruling 2026-07-27: V net + PGO binary; "be mindful not to OOM
# anything or slow down training too much"):
#   - fn20k games (load-immune in both directions; GPU untouched)
#   - 4 workers, one unit each per iteration (units == workers, so no
#     worker sits idle waiting for a straggler to finish a second unit)
#   - whole thing under a systemd unit with MemoryMax=3G. The box runs the
#     trainer at ~12GB of a 15GB machine, so this is the binding constraint,
#     not CPU: 8 engine processes x ~150MB (84MB net + 16MB hash) fits.
#   - exchange lives on DISK, not /tmp: /tmp here is a 7.7G tmpfs, i.e. RAM.
#     The coordinator stages the engine + fastchess + book into the exchange
#     and then writes two JSON files per unit forever, so an exchange under
#     /tmp both competes with the trainer for memory and dies on a full
#     tmpfs (Errno 122, observed 2026-07-27 01:0x).
#   - PGO'd tune binary built here if missing (~6 min, niced behind training)
#
# SHAPE (2026-07-27): 8 games/iteration x 2400 iterations ~= 19k games in
# ~5-6h. SPSA is a stochastic approximation — it converges on the NUMBER of
# gradient steps, not on the precision of any single one (Fishtest tunes at
# 2 games/iteration). So for a fixed game budget, many noisy iterations beat
# few precise ones, and the per-iteration overhead is what caps us: hence
# --poll 0.4 on the file:// exchange.
set -eu
cd "$(dirname "$0")/../.."
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
ENGINE=engines/bin/archon-d47v-tune-pgo
[ -e "$ENGINE" ] || {
  echo "[stageA] building PGO tune binary (~6 min, niced behind the trainer)"
  nice -n 19 bash tools/build_pgo.sh sota,arch-d47v,tune archon-d47v-tune-pgo "$NET"
}
systemd-run --user --unit=archon-spsa-stageA --collect \
  --working-directory="$PWD" -p MemoryHigh=2500M -p MemoryMax=3G \
  /bin/bash -c "bash tools/spsa/run_spsa_dist.sh \
    tools/spsa/campaigns/s2-stageA.json file://$HOME/.cache/archon-spsa-stageA \
    $ENGINE $NET 4 2400 4 1 20000 0.4 && sleep infinity" \
  || true
echo "[stageA] unit archon-spsa-stageA up; watch tools/spsa/out/s2-stageA-*/coordinator.log"
echo "[stageA] verify gate afterwards: 10+0.1 SPRT vs untuned before anything lands in params.rs"
