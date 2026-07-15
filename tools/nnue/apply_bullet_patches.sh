#!/usr/bin/env bash
set -euo pipefail

BULLET_DIR="${1:-.archon-state/toolchains/bullet}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

apply_one() {
    local patch="$SCRIPT_DIR/patches/$1"
    if git -C "$BULLET_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        echo "$1 already applied"
    elif git -C "$BULLET_DIR" apply --check "$patch"; then
        git -C "$BULLET_DIR" apply "$patch"
        echo "applied $1"
    else
        echo "bullet checkout is incompatible with $patch" >&2
        exit 1
    fi
}

apply_one bullet-frozen-weights.patch
apply_one bullet-sfbinpack-score-div.patch
