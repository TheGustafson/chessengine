#!/bin/bash
cd /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a || exit 9
rc=0
for f in "sota" "sota,arch-d47v" "sota,arch-d48w" "sota,arch-d48x" "sota,arch-d48x,narrow65" "sota,arch-hka2048" "sota,tune" "datagen" "search-control" "sota,arch-d48x,search-stats" "sota,arch-d48x,pgonull69" "sota,arch-d48x,census70" "sota,arch-d48x,arm70"; do
    echo "== $f"
    cargo check --quiet --all-targets --features "$f" 2>&1 | grep -E '^error' | head -5 && rc=1
done
echo "MATRIX_DONE rc=$rc"
