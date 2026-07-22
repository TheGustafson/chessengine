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
# bullet-sfbinpack-combined.patch SUPERSEDES bullet-sfbinpack-score-div.patch
# (same file: score-div env rescale + the D41 extra-move packing's _hm-orient
# target frame, 2026-07-21). The old patch is kept for archaeology only.
apply_one bullet-sfbinpack-combined.patch
apply_one bullet-trainer-metrics.patch
