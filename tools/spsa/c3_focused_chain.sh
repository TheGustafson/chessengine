#!/usr/bin/env bash
# Chain: when c2-reduction finishes, launch a FOCUSED follow-up campaign.
#
# Detached by design (nohup + setsid). Harness state — crons, monitors, the
# session itself — does not survive a reset, but a detached process does, so
# anything that must outlive the session has to launch itself.
#
# WHY A FOCUSED FOLLOW-UP RATHER THAN s2-stageB (41 params at fn20k)
#
# Two measurements from 2026-07-27, both in tools/spsa/leverage_screen.py's
# header and the commit that added it.
#
# 1. DILUTION. SPSA perturbs every parameter at once, so each one's gradient
#    carries cross-term noise from the other n-1. Widening a campaign costs
#    every parameter already in it. Going 15 -> 6 buys ~1.7x on each.
#
# 2. REGIME. fn20k is about depth 12; 10+0.1 at ~1.2M nps and ~265 ms/move
#    reaches about depth 18. We are tuning six plies below where the games
#    are played, and the leverage RANKING does not even survive four plies
#    (Spearman +0.20 for d6 vs d10). Reduction parameters are depth-scaled by
#    construction, so this is partly real rather than pure noise — which is
#    exactly why it matters.
#
# So: fewer parameters, more nodes. NODES=40000 is about depth 13, twice the
# cost per game. The arithmetic behind that specific choice, with c2 as the
# yardstick (2400 iters, 15 params, SNR ~ sqrt(K)/sqrt(n-1) ~ 13.1):
#
#   nodes   depth  games in ~5h   iters   n=6 SNR    regime
#   20000    ~12       12500      1560      17.6     6 plies low
#   40000    ~13        6200       780      12.5     5 plies low   <- chosen
#   60000    ~14        4150       520      10.2     4 plies low
#
# 40k holds roughly c2's per-parameter SNR while moving one ply closer. It is
# a compromise and it is not obviously optimal: the honest answer is real-TC
# SPSA, which is the RIGHT regime outright, but that is load-sensitive and
# the box is not quiet until the trainer finishes (~21:50). Revisit then.
#
# Parameter choice is c2's own output: the top N by |sigma| from
# drift_report.py, i.e. displacement measured against each parameter's own
# random-walk scale. Those are the dimensions where 19,200 games actually saw
# a gradient; refining them at a better depth is worth more than opening 41
# new ones at the wrong one.
#
# WHY c3 STARTS FROM ENGINE DEFAULTS RATHER THAN WARM-STARTING FROM c2's THETA
#
# spsa_dist has no warm-start hook — preflight reads theta from the binary's
# compiled `tunables` — and that turns out to be the better experiment anyway.
# Our real problem is not finding parameter values, it is BELIEVING them: a
# 200-game verification match has a ~13 Elo error bar and cannot resolve the
# few-Elo effects SPSA produces, which is exactly why stage A's verify came
# back 0.00 +- 13.14 and told us nothing.
#
# Two independent campaigns starting from the same defaults, run at different
# depths, give us something a single campaign cannot: REPLICATION. If c2 (d12)
# and c3 (d13) independently push a parameter the same direction, that is real
# evidence about the sign, gathered from ~26,000 games total rather than from a
# match we cannot afford to make conclusive. If they disagree, the parameter is
# depth-sensitive or noise, and either way we should not ship it. Warm-starting
# would destroy that by construction — c3 would inherit c2's answer instead of
# testing it.
#
# One confound to keep honest: c2 ran on the pre-upcoming-repetition engine and
# c3 will run on the post one (the tune-PGO binary was rebuilt 2026-07-27 for
# the loader fix and picked the feature up). Upcoming-repetition only fires in
# positions with an available draw cycle, so it is unlikely to move reduction
# optima much — but a disagreement between c2 and c3 has two possible causes,
# not one, and should not be read as depth-sensitivity without checking.
set -eu
cd "$(dirname "$0")/../.."

C2_OUT=${1:?c2 run dir, e.g. tools/spsa/out/c2-reduction-0727-0827}
NKEEP=${2:-6}
NODES=${3:-40000}
ITERS=${4:-780}
ENGINE=engines/bin/archon-d47v-tune-pgo
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
LOG=tools/spsa/out/c3-chain.log

exec >>"$LOG" 2>&1
printf '\n===== c3 chain armed %s =====\n' "$(date -Is)"
printf 'waiting on %s/theta_final.json\n' "$C2_OUT"

# Poll cheaply. This is a detached chain, not a foreground wait.
while [ ! -f "$C2_OUT/theta_final.json" ]; do
    # If the coordinator died without finishing, stop rather than spin for
    # ever: no process AND no result means the run is gone.
    if ! pgrep -f 'spsa_dist.py' >/dev/null 2>&1; then
        printf 'c2 coordinator is gone and there is no theta_final.json — aborting %s\n' "$(date -Is)"
        exit 1
    fi
    sleep 120
done
printf 'c2 finished %s\n' "$(date -Is)"

python3 tools/spsa/drift_report.py "$C2_OUT" || true

# Build the focused campaign from c2's drift. Keeping the ORIGINAL min/max
# and c_end: the leverage screen showed we cannot justify rescaling c_end
# from single-depth numbers, so the ranges stay as they were.
python3 - "$C2_OUT" "$NKEEP" <<'PY'
import json, re, subprocess, sys
run, nkeep = sys.argv[1], int(sys.argv[2])
out = subprocess.run(["python3", "tools/spsa/drift_report.py", run],
                     text=True, capture_output=True).stdout
sigma = {}
for ln in out.splitlines():
    m = re.match(r'^([A-Z][A-Z0-9_]+)\s+.*\s+([\d.]+)\s*$', ln)
    if m:
        sigma[m.group(1)] = float(m.group(2))
if not sigma:
    raise SystemExit("could not parse drift_report output — not launching")

ranked = sorted(sigma, key=lambda n: -sigma[n])[:nkeep]
best = max(sigma.values())
print(f"[c3] top |sigma|: " + ", ".join(f"{n}={sigma[n]:.1f}" for n in ranked))
if best < 1.5:
    # Not fatal. It means c2 saw no strong gradient anywhere, so the follow-up
    # is a refinement of weak candidates rather than of proven ones — worth
    # saying out loud in the log instead of quietly proceeding.
    print(f"[c3] WARNING: best sigma is only {best:.1f}; c2 found no strong "
          f"gradient. Treat c3's output as exploratory.")

base = {p["name"]: p for p in
        json.load(open("tools/spsa/campaigns/c2-reduction.json"))["params"]}
cfg = {"name": "c3-focused",
       "note": "top-|sigma| subset of c2-reduction, retuned nearer the "
               "deployment depth; see c3_focused_chain.sh for the reasoning",
       "params": [base[n] for n in ranked]}
json.dump(cfg, open("tools/spsa/campaigns/c3-focused.json", "w"), indent=2)
print(f"[c3] wrote tools/spsa/campaigns/c3-focused.json with {len(ranked)} params")
PY

EX=file://$HOME/.cache/spsa-ex-c3
rm -rf "${EX#file://}"
printf 'launching c3-focused: nodes=%s iters=%s %s\n' "$NODES" "$ITERS" "$(date -Is)"
bash tools/spsa/run_spsa_dist.sh tools/spsa/campaigns/c3-focused.json \
    "$EX" "$ENGINE" "$NET" 4 "$ITERS" 4 1 "$NODES" 0.4 0.03
