#!/usr/bin/env bash
# tools/tactics.sh — tier-1.5 tactical sanity harness (design/14 §3.1, §4 step 11).
#
# Feeds an EPD suite through the engine and scores bm/am hits. NOT a merge
# gate (single-position tactics are noisy); required before milestone
# declarations (M2+), where the score is tracked in docs/strength-log.md and
# an unexplained drop blocks the milestone (design/14 §3.9 item 8).
#
# Usage:
#   tools/tactics.sh [ENGINE] [SUITE] [tactics.py args...]
#   tools/tactics.sh --fetch-wac
#
#   ENGINE  defaults to target/release/chessengine
#   SUITE   defaults to tests/data/tactics-smoke.epd (committed, 28 verified
#           positions). Published corpora are fetched, never vendored (D15):
#           `--fetch-wac` downloads WAC (revised) + mate-in-2..5 into
#           engines/suites/ (gitignored).
#   Extra args go to tools/tactics.py: --movetime MS | --depth D, --hash MB,
#   --log (append score to docs/strength-log.md), -v.
#
# Examples:
#   tools/tactics.sh                                        # smoke, 1 s/pos
#   tools/tactics.sh target/release/chessengine tests/data/tactics-smoke.epd --movetime 200 --log
#   tools/tactics.sh --fetch-wac
#   tools/tactics.sh target/release/chessengine engines/suites/wac-revised.epd --log
#
# Requires: python3 + python-chess (pip install chess) — the same dependency
# tools/verify_perft.py already uses.
set -euo pipefail
cd "$(dirname "$0")/.."

# Suite mirror: TerjeKir/EngineTests (the Weiss author's public test-suite
# collection) — plain EPD with bm/id opcodes, stable raw URLs.
WAC_BASE="https://raw.githubusercontent.com/TerjeKir/EngineTests/master/testfiles"

if [ "${1:-}" = "--fetch-wac" ]; then
    mkdir -p engines/suites
    for f in wac-revised.epd mate_in_2.epd mate_in_3.epd mate_in_4.epd mate_in_5.epd; do
        echo "fetching $f ..."
        curl -sfL -o "engines/suites/$f" "$WAC_BASE/$f"
    done
    wc -l engines/suites/*.epd
    exit 0
fi

ENGINE="${1:-target/release/chessengine}"
[ $# -ge 1 ] && shift
if [ ! -x "$ENGINE" ]; then
    echo "error: engine binary '$ENGINE' not found/executable (build with: cargo build --release)" >&2
    exit 1
fi

exec python3 tools/tactics.py "$ENGINE" "$@"
