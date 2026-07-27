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
echo "════ $(date '+%F %H:%M') SPSA BLOCK DONE ════"
} >> "$LOG" 2>&1
