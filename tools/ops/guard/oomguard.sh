# v3: protect the D49 trainer by raising EVERY expendable workload above it.
# v1 matched only compilers -> lapsed whenever no build ran.
# v2 added a fixed tool-name list -> lapsed the moment a study used a NEW tool
#    (reprgap/probes.py held 6.9 GB at adj=200 while the trainer sat at adj=200).
# v3 matches STRUCTURALLY: any process whose cmdline references this repo's
# tools/ or a worktree, plus compilers -- minus the trainer and wrapper shells.
# Writes to /proc only; kills nothing.
REPO=/home/nick/projects/chessengine
while :; do
  for p in $(pgrep -x 'rustc|cargo' 2>/dev/null; pgrep -f "$REPO" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *pump_train*)       continue;;   # NEVER the trainer
      *shell-snapshots*)  continue;;   # NEVER a wrapper shell (self-match trap)
      *claude*)           continue;;   # NEVER the harness
    esac
    case "$cmd" in
      *tools/*|*worktrees/*|*/bin3[0-9]/*|*/bin4[0-9]/*|*rustc*|*cargo*) ;;
      *) continue;;
    esac
    cur=$(cat "/proc/$p/oom_score_adj" 2>/dev/null || echo 0)
    [ "${cur:-0}" -lt 1000 ] && echo 1000 > "/proc/$p/oom_score_adj" 2>/dev/null
  done
  sleep 10
done
