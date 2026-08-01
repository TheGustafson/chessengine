#!/bin/bash
while :; do
  st=$(systemctl --user is-active simplify-verify 2>/dev/null)
  if [ "$st" != "active" ]; then
    echo "simplify-verify unit: $st"
    tail -20 /home/nick/.cache/archon-simplify/verify.log
    break
  fi
  sleep 60
done
