#!/usr/bin/env bash
# Read a file from the Stockfish reference clone at the PINNED revision
# eca43a97 (CLAUDE.md: never the checkout, never a tag).  Read-only.
#   bash sfshow.sh src/position.cpp
set -euo pipefail
REV=${REV:-eca43a97}
exec git -C stockfish_reference/Stockfish show "$REV:${1:?path}"
