#!/usr/bin/env bash
cd /home/nick/projects/chessengine
for m in self sf18 sf16; do
  MODE=$m GAMES=150 NODES=25000 CONC=2 TAG=pilot-$m bash tools/ops/data/gen_games.sh
done
