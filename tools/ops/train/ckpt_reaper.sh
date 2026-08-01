#!/usr/bin/env bash
# tools/ops/train/ckpt_reaper.sh — bound a training run's checkpoint directory.
#
# PURPOSE
#   Training writes a checkpoint every `export_every` superbatches and never
#   deletes one. At ~433 MB each that outruns the disk on any long run. This
#   prunes the ones nothing will need again, on a schedule, forever.
#
# USAGE
#   systemd-run --user --unit=ckptreaper -p MemoryMax=256M \
#       /path/to/tools/ops/train/ckpt_reaper.sh
#
#   Environment:
#     EXP_DIR      checkpoint dir       (default data/nnue/experiments/d49-1280)
#     CKPT_PREFIX  restrict to one run  (default: EVERY prefix found in EXP_DIR)
#     KEEP_NEWEST  resume-safety margin (default 4)
#     KEEP_EVERY   analysis ladder      (default 100; 0 disables)
#     KEEP_ALWAYS  sb numbers never deleted (default "400 700")
#     PERIOD       seconds between sweeps   (default 900)
#     DRY_RUN=1    log what it would remove, delete nothing
#
# RETENTION — a checkpoint survives if ANY holds:
#     * it is a SYMLINK                       (a deliberate named reference)
#     * its superbatch is in KEEP_ALWAYS      (measured baselines cited in results)
#     * its superbatch is a multiple of KEEP_EVERY   (analysis ladder)
#     * it is among the newest KEEP_NEWEST for its prefix  (resume safety)
#
# WHY SYMLINKS ARE NEVER REAPED
#   A symlink in this directory is a human-created alias for a checkpoint that
#   something else depends on by NAME — e.g. `d49sfb-sb1630-ema-raw.bin` is the
#   warm point that the teacher-blend result (+17.15 +/- 10.70) is measured
#   against, and that the owed alpha=0 control must start from. It points at a
#   different run's file (`d49lrc3-...`), so no prefix or modulo rule would have
#   protected it: a dry run showed this reaper about to delete it. Regular files
#   accumulate; symlinks are intent. Never reap intent.
#
# WHY IT IS PREFIX-AGNOSTIC — the bug this version fixes
#   The previous version hardcoded the glob `d49-1280-sb*`. When the run was
#   renamed `d49sfb-*` for the teacher continuation, the reaper silently matched
#   NOTHING for the live run while still looking healthy: 23 checkpoints
#   accumulated unreaped, and the remaining 470 superbatches would have written
#   ~101 GB against 107 GB free. A prefix hardcoded in a reaper is a disk-full
#   waiting for a rename. It now discovers prefixes from the directory itself.
#
# SAFETY
#   KEEP_NEWEST must be >= 1. The launcher discovers its resume point from the
#   newest checkpoint, so reaping it would restart the run from scratch; the
#   script refuses to start otherwise.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/data/nnue/experiments/d49-1280}
CKPT_PREFIX=${CKPT_PREFIX:-}
KEEP_NEWEST=${KEEP_NEWEST:-4}
KEEP_EVERY=${KEEP_EVERY:-100}
KEEP_ALWAYS=${KEEP_ALWAYS:-"400 700"}
PERIOD=${PERIOD:-900}
DRY_RUN=${DRY_RUN:-0}
OPS_LOG="$EXP_DIR/cyc.log"; export OPS_LOG

[ "$KEEP_NEWEST" -ge 1 ] || { log "FATAL: KEEP_NEWEST=$KEEP_NEWEST would reap the resume point"; exit 2; }
[ -d "$EXP_DIR" ]        || { log "FATAL: EXP_DIR not found: $EXP_DIR"; exit 2; }

# Discover every checkpoint prefix present, so a run rename cannot blind us.
prefixes() {
    if [ -n "$CKPT_PREFIX" ]; then echo "$CKPT_PREFIX"; return; fi
    ls "$EXP_DIR"/*-sb*.bin 2>/dev/null | xargs -r -n1 basename \
        | sed -E 's/-sb[0-9]+-.*$//' | sort -u
}

log "ckpt_reaper: dir=$EXP_DIR keep_newest=$KEEP_NEWEST every=$KEEP_EVERY always='$KEEP_ALWAYS' dry=$DRY_RUN"

while :; do
    for pfx in $(prefixes); do
        newest=$(ls "$EXP_DIR/$pfx"-sb*.bin 2>/dev/null \
                 | sed -E 's/.*-sb([0-9]+)-.*/\1/' | sort -un | tail -"$KEEP_NEWEST" | tr '\n' ' ')
        [ -z "$newest" ] && continue
        freed=0
        for f in "$EXP_DIR/$pfx"-sb*.bin; do
            [ -f "$f" ] || continue
            # A symlink is a named reference something depends on. Never reap it.
            [ -L "$f" ] && continue
            n=$(basename "$f" | sed -E 's/.*-sb([0-9]+)-.*/\1/')
            case "$n" in ''|*[!0-9]*) continue;; esac

            keep=0
            for k in $KEEP_ALWAYS; do [ "$n" = "$k" ] && keep=1; done
            if [ "$keep" -eq 0 ] && [ "$KEEP_EVERY" -gt 0 ] && [ $((n % KEEP_EVERY)) -eq 0 ]; then keep=1; fi
            for k in $newest; do [ "$n" = "$k" ] && keep=1; done
            [ "$keep" -eq 1 ] && continue

            sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
            if [ "$DRY_RUN" = "1" ]; then
                log "would reap $(basename "$f") ($((sz/1024/1024))MB)"
            else
                rm -f "$f" && freed=$((freed + sz))
            fi
        done
        [ "$freed" -gt 0 ] && log "reaped $((freed/1024/1024))MB from '$pfx' (kept sb: $newest)"
    done
    sleep "$PERIOD"
done
