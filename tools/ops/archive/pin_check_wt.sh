#!/bin/bash
# Pin check for the agent worktree tree: default d6 bench + sota d13 + eval hash.
set -u
cd /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a || exit 9
NET=/home/nick/projects/chessengine/data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn
FENS=tools/nnue/calib/wl50.fens

echo "== default release build"
cargo build --release --quiet || exit 1
echo "== bench 16 1 6 default depth"
./target/release/chessengine bench 16 1 6 default depth 2>&1 >/dev/null | tail -4
echo "== sota,arch-d48x release build"
cargo build --release --features sota,arch-d48x --target-dir target/nnue-gate --quiet 2>/dev/null || exit 2
echo "== d13 wl50 node total"
./target/nnue-gate/release/chessengine bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | tail -4
echo "== eval hash"
./target/nnue-gate/release/chessengine nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1
echo "== done"
