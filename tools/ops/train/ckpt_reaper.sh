#!/bin/bash
# Keeps D49's checkpoint dir bounded for the rest of the unattended run.
# KEEPS: sb400 and sb700 (the two 5880-game measured baselines), every 100sb
# (analysis ladder), and the newest 4 pairs (resume safety -- the launcher
# discovers its resume point from the newest complete pair, so this must never
# leave fewer than one).
# Growth is ~866 MB per 10sb; 880sb remain => ~76 GB unchecked.
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
while :; do
  latest=$(ls "$D"/d49-1280-sb*-raw.bin 2>/dev/null | grep -v ema \
           | sed -E 's/.*-sb([0-9]+)-raw\.bin/\1/' | sort -n | tail -4 | tr '\n' ' ')
  [ -z "$latest" ] && { sleep 600; continue; }
  for f in "$D"/d49-1280-sb*-raw.bin "$D"/d49-1280-sb*-quant.bin; do
    [ -f "$f" ] || continue
    n=$(basename "$f" | sed -E 's/.*-sb([0-9]+)-(ema-)?(raw|quant)\.bin/\1/')
    [ "$n" = "400" ] || [ "$n" = "700" ] && continue
    [ $((n % 100)) -eq 0 ] && continue
    keep=0; for k in $latest; do [ "$n" = "$k" ] && keep=1; done
    [ $keep -eq 1 ] && continue
    rm -f "$f"
  done
  sleep 900
done
