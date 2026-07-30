#!/usr/bin/env bash
# perf study 40 — the arm ladder, built one at a time (the box hosts a D49
# trainer; `build34.sh` caps each build at a 6 G cgroup with -j2).
#
#   ARM B, the ~26 i/n: a 2x2 over {machinery present/absent} x {copy inline/call}
#     b40   arm      : main, alias taken            (already built)
#     n38   null38   : machinery, alias never taken (already built)
#     na40  noalias40: no machinery at all          <- reconstructs pre-study-38
#     nac40 noalias40+callcopy40                    <- codegen pinned to the call
#     n38c  null38+callcopy40                       <- codegen pinned to the call
#
#   ARM A, insertwin34: one binary per sort algorithm, dose from P40_DOSE
#     ds40  dosesort40                 : the walk
#     dsw40 dosesort40+insertwin34     : the branchless window
#
#   plus b40r, a REBUILD of plain main after this study's source edits, so the
#   `.text` md5 can be compared against the binary the panels measured
#   (study 35: a doc comment alone changes a binary's sha256).
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf34/build34.sh b40r  "sota,arch-d48x"
bash tools/perf34/build34.sh na40  "sota,arch-d48x,noalias40"
bash tools/perf34/build34.sh nac40 "sota,arch-d48x,noalias40,callcopy40"
bash tools/perf34/build34.sh n38c  "sota,arch-d48x,null38,callcopy40"
bash tools/perf34/build34.sh ds40  "sota,arch-d48x,dosesort40"
bash tools/perf34/build34.sh dsw40 "sota,arch-d48x,dosesort40,insertwin34"
echo "[buildarms40] done"
