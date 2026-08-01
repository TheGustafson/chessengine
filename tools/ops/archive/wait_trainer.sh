#!/bin/bash
until [ "$(systemctl --user is-active d49train 2>/dev/null)" != "active" ]; do sleep 30; done
echo "TRAINER DONE: $(systemctl --user is-active d49train 2>/dev/null)"
tail -2 /home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
