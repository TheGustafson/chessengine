#!/usr/bin/env bash
# Empirically calibrate a net's network_scale so its cp output lands on the
# INCUMBENT's scale (user ruling 2026-07-26: "we MUST adjust for scale (it is
# a big gotcha everytime) before we eval this thing").
#
# Why empirical: a changed target transform (v1.5's double sigmoid) has no
# clean closed-form inverse, and every cp-denominated search margin —
# futility, RFP, delta, aspiration — is calibrated to the incumbent's scale.
# net_scale_check.sh reports median(cand/ref); we drive that to ~1.000.
#
#   bash tools/nnue/calibrate_network_scale.sh <cand.cenn> [ref.cenn] [iters]
set -uo pipefail
cd "$(dirname "$0")/../.."
CAND="${1:?candidate .cenn}"
REF="${2:-data/nnue/nets/screens/d47h-anneal-sb150-nocpol.cenn}"
ITERS="${3:-4}"
ENG=engines/bin/archon-d47
ratio_of() {
  REF_ENGINE=$ENG bash tools/nnue/net_scale_check.sh "$1" "$REF" "$ENG" 2>/dev/null \
    | grep -oE 'median\(cand/ref\)=[0-9.]+' | grep -oE '[0-9.]+$'
}
cur=$(python3 -c "
import struct,sys
print(struct.unpack('<i', open(sys.argv[1],'rb').read()[44:48])[0])" "$CAND")
echo "start: network_scale=$cur"
for i in $(seq 1 "$ITERS"); do
  r=$(ratio_of "$CAND")
  [ -z "$r" ] && { echo "scale check FAILED"; exit 1; }
  echo "  iter $i: median(cand/ref) = $r  (want ~1.000)"
  within=$(awk "BEGIN{d=$r-1; if(d<0)d=-d; print (d<0.02)?1:0}")
  [ "$within" = 1 ] && { echo "CONVERGED at network_scale=$cur (ratio $r)"; exit 0; }
  cur=$(awk "BEGIN{printf \"%d\", $cur / $r + 0.5}")
  python3 tools/nnue/rescale_net_header.py "$CAND" "$CAND.tmp" "$cur" >/dev/null
  mv -f "$CAND.tmp" "$CAND"
  echo "    -> patched network_scale=$cur"
done
echo "WARNING: did not converge in $ITERS iters; last scale=$cur"
exit 2
