#!/usr/bin/env bash
# tools/ops/train/anneal.sh — final LR anneal from a gated checkpoint.
#
# PURPOSE
#   Take the checkpoint that passed its gate and decay the learning rate to ~0
#   over a short tail, so the weights settle instead of stopping mid-stride.
#   This is the step before promoting a net to canonical.
#
# USAGE
#   START_SB=1720 bash tools/ops/train/anneal.sh          # 30 sb anneal
#   START_SB=1720 ANNEAL_SB=40 bash tools/ops/train/anneal.sh
#
#   Environment:
#     START_SB    checkpoint to anneal from (REQUIRED — the one that passed)
#     ANNEAL_SB   length of the anneal      (default 30)
#     LR_START    LR to decay from          (default: read from the run's ramp)
#     RUN_SPEC    template                  (default d49_sfblend_anneal.json)
#
# WHY alpha IS HELD CONSTANT
#   The anneal settles the SAME objective the run was trained on. Changing
#   distill_alpha and the LR together would confound "the anneal helped" with
#   "less teacher helped", and neither would be attributable.
#
# WHY LINEAR-TO-ZERO RATHER THAN ANOTHER COSINE
#   A cosine tail never actually reaches zero, so the last superbatches still
#   move the weights. The point here is to STOP moving them.
#
# NB: an earlier LR jump->anneal EXPERIMENT was a measured NULL
#   (-7.41 +/- 10.76 over 1500 games, three cycles vs anchor). That tested
#   REPEATED jump-then-anneal cycles mid-run. It says nothing about a single
#   terminal anneal, which is standard practice — do not cite it as evidence
#   against this step.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
START_SB=${START_SB:?START_SB required — the checkpoint that passed its gate}
ANNEAL_SB=${ANNEAL_SB:-30}
RUN_SPEC=${RUN_SPEC:-d49_sfblend_anneal.json}
SPEC="$ARCHON_ROOT/tools/nnue/runs/$RUN_SPEC"
END=$((START_SB + ANNEAL_SB))
LR_START=${LR_START:-$(python3 -c "
import json;r=json.load(open('$ARCHON_ROOT/tools/nnue/runs/d49_sfblend_500.json'))['lr_ramp']
print(min((abs(sb-$START_SB),lr) for sb,lr in r)[1])")}
python3 - "$SPEC" "$START_SB" "$END" "$LR_START" <<'PY'
import json,sys
p,s,e,lr0=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
c=json.load(open(p)); n=e-s
c['superbatches']=e; c['lr_end_superbatch']=e; c['lr']=lr0
c['lr_ramp']=[[sb, lr0*max(0.0,(e-sb)/n)] for sb in range(s,e+1,2)]
json.dump(c,open(p,'w'),indent=2)
print(f"  anneal sb{s} -> sb{e}: lr {lr0:.4e} -> 0 linearly ({n} sb)")
PY
log "launching anneal from sb${START_SB} to sb${END}"
RUN_SPEC="$RUN_SPEC" CKPT_GLOB=d49sfb exec "$ARCHON_ROOT/tools/ops/train/d49_launch.sh"
