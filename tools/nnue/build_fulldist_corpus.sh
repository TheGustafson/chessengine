#!/usr/bin/env bash
# Build the FULL-DISTRIBUTION SF18 corpus (2026-07-20). Fixes exp-0112: the
# canon disagreement-subset net can't play normal positions (loses in ~5 moves).
# This mixes the FULL UHO distribution with SF18 cp substituted on the canon
# (relabeled) positions + REAL game WDL from the binpack — the substitute_labels
# idea, done memory-safe via disk sort+join (109M SF18 labels won't fit RAM on 15GB).
#
# ★ SCALE BUG (found 2026-07-21, parked path — fix before any reuse): the
# original claim "score5k/1.9 = SF18 scale" is FALSE. Measured K =
# binpack/SF18-UCI = 2.64-2.76, so control units are binpack/1.9 and the unit-
# correct lift for SF18 cp is ×K/1.9 ≈ 1.42 — NOT the ×1.94 baked into the awk
# below (1.94 is the WDL=0 OUTPUT-matching constant from exp-0111, a different
# thing). As written, relabeled rows land ~1.37× hotter than their score5k/1.9
# neighbors: a WITHIN-CORPUS scale inconsistency on top of everything else.
# See docs/research/sf18-canon-data-diagnosis-2026-07-21.md + train_canon.sh's
# derivation header. Change 1.94 -> 1.42 in the join awk if this path revives.
#
#   build_fulldist_corpus.sh
# Inputs:  data/nnue/binpacks/uho_frs.tsv        (fast_fens: FEN\tresult\tply\tscore5k, White-rel)
#          data/nnue/canon-v1/build/corpus.txt   (FEN | cp | 0.5 ; SF18 cp, White-rel)
# Output:  data/nnue/fulldist/build/corpus.shuf.bin  (train on this)
# NOTE: verify score5k is White-relative in fast_fens.rs before trusting (a STM-rel
#       score would sign-flip half the non-relabeled positions).
set -uo pipefail
cd "$(dirname "$0")/../.."
# UHO_FRS/WORK are env-overridable so the quiet-filtered twin (Fable diagnosis
# 2026-07-20) can be built from a filtered fast_fens extract into a separate dir
# without disturbing the original unfiltered corpus.
UHO_FRS="${UHO_FRS:-data/nnue/binpacks/uho_frs.tsv}"
CANON_TXT=data/nnue/canon-v1/build/corpus.txt
WORK="${WORK:-data/nnue/fulldist/build}"
CONV=tools/nnue/bf-convert/target/release/nnue_convert
UTIL=.archon-state/toolchains/bullet/target/release/bullet-utils
mkdir -p "$WORK" "$WORK/sorttmp"
export TMPDIR="$WORK/sorttmp"   # /tmp is a small tmpfs (RAM); sort 188M rows needs big-disk temp
say(){ echo "[fulldist $(date -u +%H:%M:%S)] $*"; }
for f in "$UHO_FRS" "$CANON_TXT"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done

# join key = normalized FEN (board stm castling ep; drop clocks so relabel/binpack FENs match)
say "sort canon SF18 labels by normalized-FEN key ($(wc -l < "$CANON_TXT") rows)"
awk -F' \\| ' '{n=split($1,a," "); print a[1]" "a[2]" "a[3]" "a[4]"\t"$2}' "$CANON_TXT" \
  | LC_ALL=C sort -t$'\t' -k1,1 -S 1G -T "$WORK/sorttmp" --parallel=4 -o "$WORK/canon.sorted"

say "sort UHO (key \t fullFEN \t result \t score5k) ($(wc -l < "$UHO_FRS") rows)"
awk -F'\t' '{n=split($1,a," "); print a[1]" "a[2]" "a[3]" "a[4]"\t"$1"\t"$2"\t"$4}' "$UHO_FRS" \
  | LC_ALL=C sort -t$'\t' -k1,1 -S 2G -T "$WORK/sorttmp" --parallel=4 -o "$WORK/uho.sorted"

say "left-join UHO + canon -> corpus.txt (SF18 cp where relabeled, else score5k/1.9)"
LC_ALL=C join -t$'\t' -1 1 -2 1 -a 1 -e 'NA' -o '1.2,1.3,1.4,2.2' "$WORK/uho.sorted" "$WORK/canon.sorted" \
  | awk -F'\t' '{
      fen=$1; result=$2; s5k=$3; sf18=$4;
      cp = (sf18=="NA") ? int(s5k/1.9) : int(sf18*1.42);  # both on CONTROL scale: score5k/1.9 and SF18*K/1.9 with measured K=2.7 => x1.42 (2026-07-21 fix; the old x1.94 made relabeled rows 1.37x hot). Train SCORE_MULT=1.0.
      if(cp>2499)cp=2499; if(cp<-2499)cp=-2499;
      print fen" | "cp" | "result
    }' > "$WORK/corpus.txt"
rows=$(wc -l < "$WORK/corpus.txt"); relabeled=$(LC_ALL=C join -t$'\t' -1 1 -2 1 -o '1.1' "$WORK/uho.sorted" "$WORK/canon.sorted" | wc -l)
say "corpus.txt: $rows rows ($relabeled matched SF18-relabeled)"

say "bf-convert -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -1
say "shuffle -> corpus.shuf.bin (in-RAM)"
UTIL_ABS="$PWD/$UTIL"; mkdir -p "$WORK/tmp"
( cd "$WORK" && "$UTIL_ABS" shuffle --input "$PWD/corpus.bin" --output "$PWD/corpus.shuf.bin" --mem-used-mb 8000 ) 2>&1 | tail -1
ssz=$(stat -c%s "$WORK/corpus.shuf.bin" 2>/dev/null || echo 0)
say "DONE: corpus.shuf.bin $((ssz/32)) records — train: NET_ID=fulldist-sf18-240sb SB=240 SCORE_MULT=1.94 WDL... via a WDL-enabled train"
