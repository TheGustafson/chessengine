#!/bin/bash
# PGO baseline for the simplification baskets, built from the exported
# verified tip (8dd54ba4) so the agent worktree stays editable.
set -u
cd /home/nick/.cache/archon-simplify/tip-src || exit 9
NET=/home/nick/projects/chessengine/data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn
export PGO_PROF_DIR=/home/nick/.cache/archon-simplify/pgo-prof
mkdir -p "$PGO_PROF_DIR"
bash tools/build_pgo.sh sota,arch-d48x archon-simp-base "$NET" || exit 1
install -m755 engines/bin/archon-simp-base \
    /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a/engines/bin/archon-simp-base || exit 2
echo "BASE_PGO_DONE $(date -Is)"
