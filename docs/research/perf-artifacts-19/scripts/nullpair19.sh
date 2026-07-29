#!/usr/bin/env bash
# perf study 19 — NULL FOR THE PAIRED INSTRUMENT (rule 1, run before any arm).
#
# Builds the SAME source twice under different names and pairs them. A working
# instrument returns cycles and cache-miss ratios of 1.000 with a spread smaller
# than any delta this study intends to claim. If it does not, the second metric
# axis is not measurable here and must be reported as such rather than quoted.
set -euo pipefail
bash docs/research/perf-artifacts-19/scripts/probe19.sh nullpair-A
touch src/lib.rs
bash docs/research/perf-artifacts-19/scripts/probe19.sh nullpair-B
bash docs/research/perf-artifacts-19/scripts/pair19.sh nullpair-A nullpair-B 8
