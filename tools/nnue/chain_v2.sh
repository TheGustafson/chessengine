#!/usr/bin/env bash
# Serial relaunch of the interrupted overnight chain (2026-07-16 morning).
# No gate files, no parallelism: each step runs to completion before the next.
# All RESULT lines also append to $R for the synthesis pass.
#
#   1. E1 rescreen   — clean 300g screen of the fishpack-pos+SF16-label net
#   2. E3            — net-0 recipe on official UHO prefix (tail-truncated fix)
#   3. e1ctl         — sf-100 truncated to E1's corpus size (positions-vs-size)
#   4. E4            — king768x256 rerun with the sparse-init fix
#   5. D40 rescreen  — concurrency 1 (RAM: one 3.6GB engine pair max)
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$R"; }
result() { # result <tag> <summary.json>
  python3 - "$1" "$2" <<'EOF' | tee -a "$R"
import json, sys
tag, path = sys.argv[1], sys.argv[2]
s = json.load(open(path))
print(f"[{tag}] RESULT: score={s['candidate_score']:.3f} elo={s['candidate_elo']:.1f} se={s['candidate_elo_se']:.1f}")
EOF
}

# ---- 1. E1 rescreen -------------------------------------------------------
E1=data/nnue/experiments/fishpack-sf16-relabel
if [ ! -f "$E1/screen-n5k/summary.json" ]; then
  say "step 1/5: E1 rescreen (300g @5k, conc 4)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/e1-fishpackpos-sf16labels-classic256.cenn \
    --baseline data/nnue/nets/net-0.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260716 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$E1/screen-n5k" || say "step 1 FAILED"
fi
[ -f "$E1/screen-n5k/summary.json" ] && result e1-rescreen "$E1/screen-n5k/summary.json"

# ---- 2. E3: official UHO prefix, net-0 recipe -----------------------------
E3=data/nnue/experiments/uho-transplant
if [ ! -f data/nnue/nets/e3-uho-transplant-classic256.cenn ]; then
  say "step 2/5: E3 train (binpack tail truncated at clean chunk boundary)"
  uv run python tools/nnue/train_run.py --config "$E3/e3-config.json" \
    --only preflight,train,verify,pack,sanity >> "$E3/run.log" 2>&1 || say "step 2 train FAILED (see $E3/run.log)"
fi
if [ -f data/nnue/nets/e3-uho-transplant-classic256.cenn ] && [ ! -f "$E3/screen-n5k/summary.json" ]; then
  say "step 2/5: E3 screen (300g @5k, conc 4)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/e3-uho-transplant-classic256.cenn \
    --baseline data/nnue/nets/net-0.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$E3/screen-n5k" || say "step 2 screen FAILED"
fi
[ -f "$E3/screen-n5k/summary.json" ] && result e3-uho "$E3/screen-n5k/summary.json"

# ---- 3. e1ctl: sf-100 prefix, 4,064,476 records ---------------------------
EC=data/nnue/experiments/e1-size-control
mkdir -p "$EC"
if [ ! -f data/nnue/nets/e1ctl-sf100-4m-classic256.cenn ]; then
  say "step 3/5: e1ctl train (sf-100 head, 4,064,476 records)"
  head -c $((4064476 * 32)) data/nnue/train.bin > "$EC/train.bin"
  python3 - <<'EOF'
import json
cfg = json.load(open('data/nnue/campaigns/sf100-matrix-v1/resolved-configs/sf100-matrix-v1-sf100-classic256.json'))
cfg['net_id'] = 'e1ctl-sf100-4m-classic256'
cfg['notes'] = 'E1 size control: sf-100 prefix truncated to 4,064,476 records (E1 corpus size), net-0 recipe.'
cfg['corpus'] = {'id': 'sf-100-prefix-4m', 'raw_dir': None,
                 'train_bin': 'data/nnue/experiments/e1-size-control/train.bin'}
cfg['train']['checkpoint_dir'] = '/root/projects/learnrust/chessengine/data/nnue/experiments/e1-size-control/ckpt'
env = cfg['train']['env']
env['NET0_DATA'] = 'data/nnue/experiments/e1-size-control/train.bin'
env['NET0_FORMAT'] = 'bulletformat'
env['NET0_SUPERBATCHES'] = '30'
env['NET0_LR_END_SB'] = '30'
env['NET0_SAVE_RATE'] = '30'
json.dump(cfg, open('data/nnue/experiments/e1-size-control/config.json', 'w'), indent=2)
EOF
  uv run python tools/nnue/train_run.py --config "$EC/config.json" \
    --only preflight,train,verify,pack,sanity >> "$EC/run.log" 2>&1 || say "step 3 train FAILED (see $EC/run.log)"
fi
if [ -f data/nnue/nets/e1ctl-sf100-4m-classic256.cenn ] && [ ! -f "$EC/screen-n5k/summary.json" ]; then
  say "step 3/5: e1ctl screen (300g @5k, conc 4)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/e1ctl-sf100-4m-classic256.cenn \
    --baseline data/nnue/nets/net-0.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
    --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
    --output "$EC/screen-n5k" || say "step 3 screen FAILED"
fi
[ -f "$EC/screen-n5k/summary.json" ] && result e1ctl "$EC/screen-n5k/summary.json"

# ---- 4. E4: king768x256 with the sparse-init fix --------------------------
E4=data/nnue/campaigns/sf100-king-fixedinit-v1
if [ ! -f "$E4/DONE" ]; then
  say "step 4/5: E4 king768x256 rerun (init fix; value_matrix trains + screens)"
  python3 - <<'EOF'
import json
cfg = json.load(open('tools/nnue/configs/sf100-matrix-v1-king768x256.json'))
cfg['campaign_id'] = 'sf100-king-fixedinit-v1'
json.dump(cfg, open('tools/nnue/configs/sf100-king-fixedinit-v1.json', 'w'), indent=2)
EOF
  uv run python tools/nnue/value_matrix.py \
    --config tools/nnue/configs/sf100-king-fixedinit-v1.json \
    --source sf100 --cell king768x256 >> "$E4/run.log" 2>&1 \
    && touch "$E4/DONE" || say "step 4 FAILED (see $E4/run.log)"
fi
S4=$(find "$E4" -name summary.json | head -1)
[ -n "$S4" ] && result e4-king-fixedinit "$S4"

# ---- 5. D40 rescreen (concurrency 1) --------------------------------------
M=data/nnue/campaigns/sf100-matrix-v1
if [ ! -f "$M/matches/d40-manual-screen-n5k/summary.json" ]; then
  say "step 5/5: D40 rescreen (300g @5k, conc 1 — RAM)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/sf100-matrix-v1-sf100-d40-full.cenn \
    --baseline data/nnue/nets/net-0.cenn \
    --games 300 --nodes 5000 --concurrency 1 --seed 20260715 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-default \
    --output "$M/matches/d40-manual-screen-n5k" || say "step 5 FAILED"
fi
[ -f "$M/matches/d40-manual-screen-n5k/summary.json" ] && result d40-sf100 "$M/matches/d40-manual-screen-n5k/summary.json"

say "CHAIN COMPLETE"
