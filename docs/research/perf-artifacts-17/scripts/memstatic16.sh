#!/usr/bin/env bash
# perf study 16 — STATIC census of libc mem* calls, by enclosing symbol.
#
#   bash memstatic16.sh [binary]
#
# study 15 §10's one-liner, made a tool. This is the instrument with no filter:
# it does not need the site to execute, so it finds a memset on a path the
# profile never samples AND one on a path the interposer sees only once. Pair it
# with memspy16.sh (which supplies the RATE) -- static says "it is there",
# dynamic says "it costs".
set -euo pipefail
BIN=${1:-target-fast16/release/chessengine}
objdump -d --demangle "$BIN" \
  | awk '
      /^[0-9a-f]+ </ { s = $0; sub(/^[0-9a-f]+ </, "", s); sub(/>:$/, "", s) }
      /call.*<mem(set|cpy|move)@/ {
        fn = $NF; gsub(/[<>]/, "", fn); sub(/@.*/, "", fn)
        # the immediate before the call is usually the size in %edx/%rdx; report
        # the raw instruction so the size is visible
        printf "%-8s %s\n", fn, s
      }' \
  | sort | uniq -c | sort -rn
