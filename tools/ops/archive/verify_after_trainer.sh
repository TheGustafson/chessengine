#!/bin/bash
# Wait for d49train to go INACTIVE (coordinator's hard rule: no builds while
# it is active), then run the consolidated tier-0/0b verification:
#   1. cargo fmt --check + cargo test (dev)
#   2. the three pins: default d6 bench / sota d13 wl50 / eval hash
#   3. CHECK_NNUE_GATE=1 tools/check.sh (the full merge gate)
# Polls every 60s; aborts after 4h of waiting.
set -u
cd /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a || exit 9
NET=/home/nick/projects/chessengine/data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn
FENS=tools/nnue/calib/wl50.fens

for _ in $(seq 1 240); do
    state=$(systemctl --user is-active d49train 2>/dev/null)
    [ "$state" != "active" ] && break
    sleep 60
done
state=$(systemctl --user is-active d49train 2>/dev/null)
if [ "$state" = "active" ]; then
    echo "TIMEOUT: trainer still active after 4h; no builds run"
    exit 8
fi
echo "== trainer state: $state; starting verification $(date -Is)"

echo "== cargo fmt --check"
cargo fmt --check || exit 1
echo "== cargo test (dev)"
cargo test --quiet 2>&1 | grep -E 'test result|FAILED|error\[' | head -20
cargo test --quiet > /dev/null 2>&1 || { echo "TESTS FAILED"; exit 2; }

echo "== default release build"
cargo build --release --quiet || exit 3
echo "== PIN 1: bench 16 1 6 default depth"
./target/release/chessengine bench 16 1 6 default depth 2>&1 >/dev/null | tail -3

echo "== sota,arch-d48x release build"
cargo build --release --features sota,arch-d48x --target-dir target/nnue-gate --quiet 2>/dev/null || exit 4
echo "== PIN 2: d13 wl50 node total"
./target/nnue-gate/release/chessengine bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | tail -3
echo "== PIN 3: eval hash"
./target/nnue-gate/release/chessengine nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1

echo "== full gate: CHECK_NNUE_GATE=1 tools/check.sh"
CHECK_NNUE_GATE=1 NNUE_GATE_NET="$NET" bash tools/check.sh
rc=$?
echo "== check.sh exit: $rc  $(date -Is)"
exit $rc
