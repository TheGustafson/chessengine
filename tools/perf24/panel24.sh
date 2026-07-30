#!/usr/bin/env bash
# tools/perf24/panel24.sh <frame> <label>...   -- probe several labels in a row
set -euo pipefail
cd "$(dirname "$0")/../.."
frame=$1; shift
for l in "$@"; do bash tools/perf24/probe24.sh "$l" "$frame"; done
