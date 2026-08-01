#!/bin/bash
while :; do
  st=$(systemctl --user is-active simplify-verify2 2>/dev/null)
  if [ "$st" != "active" ]; then
    echo "simplify-verify2 unit: $st"
    grep -E 'Nodes searched|^[0-9a-f]{64}$|check.sh exit|FAILED|TIMEOUT|^== ' /home/nick/.cache/archon-simplify/verify2.log 2>/dev/null | tail -25
    break
  fi
  sleep 45
done
