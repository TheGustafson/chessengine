#!/bin/bash
cd /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a || exit 9
cargo check --all-targets 2>&1 | grep -E '^error' | head -10
echo LIGHT_CHECK_DONE
