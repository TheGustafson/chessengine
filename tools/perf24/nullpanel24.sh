#!/usr/bin/env bash
# tools/perf24/nullpanel24.sh -- the paired-cycles null, three replicates.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash tools/perf24/pair24.sh base nullB 6 bench16
bash tools/perf24/pair24.sh nullA nullB 6 bench16
bash tools/perf24/pair24.sh base nullA 10 bench16
