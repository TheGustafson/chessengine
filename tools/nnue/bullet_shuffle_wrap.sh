#!/usr/bin/env bash
# build_corpus.sh adapter: BULLET_UTILS_SHUFFLE receives <in.bin> <out.bin>.
# (Tracked replacement for the vanished scratchpad shuffle_wrap.sh.)
set -euo pipefail
exec "$(dirname "$0")/../../.archon-state/toolchains/bullet/target/release/bullet-utils" \
  shuffle --input "$1" --output "$2" --mem-used-mb 2048
