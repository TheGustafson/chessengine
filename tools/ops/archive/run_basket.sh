#!/bin/bash
# Launch one games-gated basket match: CAND vs BASE, fn25k, segmented,
# pooled non-inferiority SPRT elo0=-3 elo1=+1.
#
#   run_basket.sh CAND BASE TAG [SEGMENTS] [ROUNDS]
#
# Rules honoured:
# - refuses to run while d49train is active (that is a BUILD-free rule, but a
#   match is also memory pressure; matches wait for the trainer too unless
#   FORCE_WITH_TRAINER=1)
# - stops d49xml for the window and RESTARTS it afterwards (rearm_xml.sh)
# - chain70.sh already waits for MIN_AVAIL and pools per segment
set -u
cd /home/nick/projects/chessengine/.claude/worktrees/agent-abed05f2117dec14a || exit 9
CAND=${1:?}; BASE=${2:?}; TAG=${3:?}; SEGMENTS=${4:-16}; ROUNDS=${5:-125}

if [ "${FORCE_WITH_TRAINER:-0}" != "1" ]; then
    while [ "$(systemctl --user is-active d49train 2>/dev/null)" = "active" ]; do
        sleep 60
    done
fi

xml_was_active=0
if [ "$(systemctl --user is-active d49xml 2>/dev/null)" = "active" ]; then
    xml_was_active=1
    systemctl --user stop d49xml
    echo "d49xml stopped for the match window"
fi

MIN_AVAIL=2500 CONCURRENCY=2 NODES=25000 \
    BOOK=/home/nick/projects/chessengine/engines/books/UHO_Lichess_4852_v1.epd \
    bash tools/perf70/chain70.sh "$CAND" "$BASE" "$TAG" "$SEGMENTS" "$ROUNDS"

python3 tools/perf70/pool_sprt.py engines/runs/"$TAG"-*/sprt.out || true

if [ "$xml_was_active" = "1" ]; then
    bash /home/nick/.local/share/archon-d49/rearm_xml.sh 2>/dev/null \
        || systemctl --user start d49xml 2>/dev/null \
        || echo "NOTE: could not restart d49xml -- restart it manually"
    echo "d49xml restart attempted"
fi
