#!/usr/bin/env bash
# perf study 40 — md5 of a binary's `.text` alone.
#
#   bash tools/perf40/textmd5.sh <binary>...
#
# Study 35 found that adding a doc comment changes a binary's sha256 (panic
# `Location` line numbers live in `.rodata`), so a whole-file hash cannot tell
# "the code is the same" from "the code moved". `.text` can.
set -euo pipefail
cd "$(dirname "$0")/../.."
for b in "$@"; do
  s=$(objcopy -O binary --only-section=.text "$b" /dev/stdout | md5sum | cut -d' ' -f1)
  n=$(objcopy -O binary --only-section=.text "$b" /dev/stdout | wc -c)
  printf '%-8s %s  %s bytes of .text\n' "$(basename "$b")" "$s" "$n"
done
