#!/usr/bin/env bash
# perf study 17 — build Stockfish `eca43a97` for the AVX2 callgrind frame.
#
# WHY. `engines/bin/stockfish-eca43a97` is an AVX-512 binary (1,309 `zmm`
# references) and valgrind cannot execute AVX-512 — it dies with an illegal
# opcode inside SF's static-init constructor before `main`.  A callgrind
# comparison of the two engines' make/unmake needs BOTH sides in a frame
# valgrind can run, so SF is rebuilt at ARCH=x86-64-avx2, exactly as ARCHON is
# built `-C target-cpu=x86-64-v3` for the same reason.
#
# The source is a read-only `git archive` of `eca43a97` (CLAUDE.md's pinned
# constants revision — NOT the checkout, which sits at the older `sf_18` tag).
# The net is exported out of the shipped AVX-512 binary with SF's own
# `export_net`, and its sha256 matches `EvalFileDefaultName` exactly, so the
# rebuild is bit-identical in weights to the binary every previous study used.
#
# Build discipline: own cgroup, no swap, oomlast.  `-j2`.
set -euo pipefail
DIR=${1:-/home/nick/archon-sf17-eca43a97}
ARCH=${2:-x86-64-avx2}
free -m | awk 'NR==2 {if ($7 < 2600) {print "REFUSING: only " $7 " MB available (need 2600)"; exit 1}}'
cd "$DIR/src"
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  /home/nick/projects/chessengine/tools/oomlast.sh \
  nice -n 10 make -j2 ARCH="$ARCH" build
ls -la "$DIR/src/stockfish"
