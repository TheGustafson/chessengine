#!/usr/bin/env bash
# perf study 66 — replay this branch's four commits WITHOUT the `bin66/` arm
# binaries, which a `git add -A` swept in (14 MB). `bin65/` and every earlier
# study's arm directory is gitignored; 66's was not, and the omission has to be
# fixed before the branch is merged, not after.
#
# Messages are preserved with `git commit -C`, so the history reads identically.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${1:?base commit}
shift
COMMITS=("$@")
git reset --hard "$BASE"
for c in "${COMMITS[@]}"; do
  git cherry-pick -n "$c"
  git rm -r --cached --ignore-unmatch -q bin66 || true
  git commit -q -C "$c"
done
git log --oneline "$BASE"..HEAD
