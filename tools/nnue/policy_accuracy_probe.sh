#!/usr/bin/env bash
# D47 learnability probe: top-1 policy accuracy vs SF18@50k teacher moves.
#   bash tools/nnue/policy_accuracy_probe.sh <net.cenn>   (needs sibling .cpol)
# Teacher set: diag/canon400.sf18 (fen \t cp \t bestmove, 401 rows).
set -uo pipefail
cd "$(dirname "$0")/../.."
NET="${1:?net.cenn with sibling .cpol}"
TEACH=data/nnue/canon-v1/build/diag/canon400.sf18
ENG="${ENG:-engines/bin/archon-d47}"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
cut -f1 "$TEACH" > "$T/fens"
"$ENG" nnue-eval "$NET" < "$T/fens" 2>/dev/null > "$T/out"
paste "$T/out" "$TEACH" | awk -F'\t' '
  NF>=4 && $2 != "" {
    n++
    if ($2 == $5) hit++
  }
  END {
    if (n==0) { print "[pol-probe] NO POLICY OUTPUT (sidecar missing?)"; exit 1 }
    printf "[pol-probe] top-1 accuracy: %d/%d = %.1f%%  (random-legal baseline ~3-4%%)\n", hit, n, 100*hit/n
  }'
