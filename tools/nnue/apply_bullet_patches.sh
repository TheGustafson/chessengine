#!/usr/bin/env bash
set -euo pipefail

BULLET_DIR="${1:-.archon-state/toolchains/bullet}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$SCRIPT_DIR/patches/bullet-frozen-weights.patch"

if git -C "$BULLET_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "bullet frozen-weight patch already applied"
elif git -C "$BULLET_DIR" apply --check "$PATCH"; then
    git -C "$BULLET_DIR" apply "$PATCH"
    echo "applied bullet frozen-weight patch"
else
    echo "bullet checkout is incompatible with $PATCH" >&2
    exit 1
fi
