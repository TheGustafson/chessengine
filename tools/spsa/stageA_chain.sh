#!/usr/bin/env bash
# Self-chaining tail for the overnight SPSA block, so the box never sits
# idle waiting for a human (harness monitors die with the session; a
# systemd unit does not).
#
# Waits for stage A to write theta_final.json, then runs the 10+0.1 verify
# and leaves the result in the log. It deliberately does NOT edit
# params.rs — landing a vector is a judgement call on the margin, and the
# whole point of the gate is that a human reads it.
#
# Covers the failure path too: if the stage-A unit dies without producing
# a final vector, say so loudly rather than waiting forever in silence.
set -u
cd /home/nick/projects/chessengine
LOG=/tmp/spsa_chain.log
# Pin the run we are waiting on AT ARM TIME. Globbing for any
# theta_final.json matches finished runs from previous nights — the first
# arming did exactly that and started verifying a stale vector from a smoke
# run three hours earlier.
RUN=${1:-$(ls -dt tools/spsa/out/s2-stageA-* 2>/dev/null | grep -v aborted | head -1)}
{
echo "════ $(date '+%F %H:%M') chain armed on $RUN ════"
while :; do
  F="$RUN/theta_final.json"
  [ -e "$F" ] && { echo "════ $(date '+%F %H:%M') stage A finished: $F ════"; break; }
  if ! systemctl --user is-active --quiet archon-spsa-stageA; then
    echo "════ $(date '+%F %H:%M') stage-A unit is GONE with no theta_final ════"
    echo "The interim vector is still usable: verifying that instead."
    break
  fi
  sleep 120
done
echo "════ $(date '+%F %H:%M') verify: tuned vs default, 600g @ 10+0.1 ════"
if [ -e "$RUN/theta_final.json" ]; then
  bash tools/spsa/verify_theta.sh "$RUN/theta_final.json"
else
  bash tools/spsa/verify_theta.sh
fi

# Stage A2 next rather than leaving the box idle. It is independent of stage
# A's outcome — different parameters, started from the compiled defaults
# either way — so it is safe to run before a human has judged A's margin.
# A2 is the core LMR reduction shape, the larger Elo pool of the two.
echo "════ $(date '+%F %H:%M') launching stage A2 (core LMR shape) ════"
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
ENGINE=engines/bin/archon-d47v-tune-pgo
rm -rf "$HOME/.cache/archon-spsa-stageA2"
bash tools/spsa/run_spsa_dist.sh tools/spsa/campaigns/s2-stageA2.json \
  "file://$HOME/.cache/archon-spsa-stageA2" "$ENGINE" "$NET" 4 2400 4 1 20000 0.4
A2RUN=$(ls -dt tools/spsa/out/s2-stageA2-* 2>/dev/null | head -1)
echo "════ $(date '+%F %H:%M') stage A2 running in $A2RUN ════"
while [ ! -e "$A2RUN/theta_final.json" ]; do
  pgrep -f 'spsa_dis[t].py' >/dev/null || {
    echo "════ $(date '+%F %H:%M') A2 coordinator gone; verifying interim ════"
    break
  }
  sleep 120
done
echo "════ $(date '+%F %H:%M') verify A2: tuned vs default, 600g @ 10+0.1 ════"
if [ -e "$A2RUN/theta_final.json" ]; then
  bash tools/spsa/verify_theta.sh "$A2RUN/theta_final.json"
else
  bash tools/spsa/verify_theta.sh "$HOME/.cache/archon-spsa-stageA2/state/theta-latest.json"
fi
echo "════ $(date '+%F %H:%M') SPSA BLOCK DONE (A + A2 both verified) ════"
} >> "$LOG" 2>&1
