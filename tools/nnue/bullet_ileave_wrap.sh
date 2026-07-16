#!/usr/bin/env bash
# build_corpus.sh adapter: BULLET_UTILS_ILEAVE receives <out.bin> <in1.bin>...
set -euo pipefail
out="$1"; shift
exec "$(dirname "$0")/../../.archon-state/toolchains/bullet/target/release/bullet-utils" \
  interleave "$@" --output "$out"
