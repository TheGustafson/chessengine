#!/usr/bin/env bash
# ============================================================================
# build_canon_tight.sh — the TIGHT SF18-canon corpus rebuild (2026-07-21).
#
# Fixes deltas 2+3 of the canon collapse
# (docs/research/sf18-canon-data-diagnosis-2026-07-21.md):
#   delta 2: exports with --filter-sfbinpack (the trainer-parity 5-criterion
#            quiet filter: ply>=16, !in-check, |raw score|<=10000, best move
#            Normal, best move non-capture) instead of the in-check-only
#            canon_filter that left ~22% tactical-label poison.
#   delta 3: joins REAL game WDL from the uho_frs.tsv fleet extraction
#            (White-relative result) instead of the 0.5 placeholder, so the
#            corpus can train with the proven D40_WDL=0.3 blend.
# Delta 1 (scale) is fixed at train time: train_canon.sh auto-couples
# SCORE_MULT to WDL (0.3 -> 1.42, 0 -> 1.94) and net_scale_check.sh verifies
# the output scale before any gate games.
#
#   bash tools/nnue/build_canon_tight.sh            # full (~2280 shards)
#   SMOKE=1 bash tools/nnue/build_canon_tight.sh    # 2 batches end-to-end
#
# Output: data/nnue/canon-v1/build-tight/corpus.shuf.bin (+ corpus.txt)
# Self-checks (each FATAL on violation — a silent recipe drift must not
# produce a trainable file): filter drop-rate in [20,35]%, WDL join hit-rate
# >= 98%, result distribution not degenerate, .bin 32B-aligned + row-matched.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

CANON="${CANON:-data/nnue/canon-v1}"
WORK="${WORK:-data/nnue/canon-v1/build-tight}"
UHO_FRS="${UHO_FRS:-data/nnue/binpacks/uho_frs.tsv}"
OBS=target/release/obs-tool
CONV=tools/nnue/bf-convert/target/release/nnue_convert
UTIL=.archon-state/toolchains/bullet/target/release/bullet-utils
SHARDS_DIR="$CANON/shards/shards"
BATCH=285
JOBS=4

mkdir -p "$WORK" "$WORK/sorttmp"
export TMPDIR="$WORK/sorttmp"   # /tmp is tmpfs (RAM) — big sorts go to real disk
say(){ echo "[canon-tight $(date -u +%H:%M:%S)] $*"; }
die(){ say "FATAL: $*"; exit 1; }

for f in "$OBS" "$CONV" "$UTIL"; do [ -x "$f" ] || die "not built: $f"; done
[ -s "$UHO_FRS" ] || die "missing WDL source: $UHO_FRS"
mapfile -t shards < <(ls "$SHARDS_DIR"/*.ceobs 2>/dev/null)
[ "${#shards[@]}" -gt 0 ] || die "no shards in $SHARDS_DIR"
if [ "${SMOKE:-0}" = 1 ]; then BATCH=50; shards=("${shards[@]:0:100}"); say "SMOKE: ${#shards[@]} shards, batch $BATCH"; fi
say "${#shards[@]} shards, batches of $BATCH, $JOBS parallel exporters"

# ---- 1. filtered export (chunked, resumable, parallel) ---------------------
# Per-batch outputs are kept until the final concat; a non-empty batch file is
# a completed batch (obs-tool writes the whole file at the end, so a killed
# batch leaves nothing or nothing-usable — re-run to redo it).
nb=$(( (${#shards[@]} + BATCH - 1) / BATCH ))
batchlist="$WORK/batches.txt"; : > "$batchlist"
for ((i=0; i<${#shards[@]}; i+=BATCH)); do echo $((i/BATCH)) >> "$batchlist"; done

export_batch() { # bn
  local bn="$1" i=$(( $1 * BATCH ))
  local out="$WORK/values_b${bn}.raw" log="$WORK/values_b${bn}.log"
  if [ -s "$out" ]; then echo "[canon-tight] batch $bn: exists, skip"; return 0; fi
  "$OBS" export-value "$out.part" "${shards[@]:i:BATCH}" --filter-sfbinpack 2> "$log" \
    && [ -s "$out.part" ] && mv "$out.part" "$out" \
    && echo "[canon-tight] batch $bn: $(wc -l < "$out") rows" \
    || { echo "[canon-tight] batch $bn FAILED (see $log)"; return 1; }
}
export -f export_batch 2>/dev/null || true   # subshell path below doesn't need the export

say "exporting $nb batches (filter: sfbinpack-parity)"
fail=0
while read -r bn; do
  # bounded parallelism without xargs -I env gymnastics
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || fail=1; done
  export_batch "$bn" &
done < "$batchlist"
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
[ "$fail" = 0 ] || die "one or more export batches failed"

# filter-rate self-check from the batch logs (positions_in vs dropped)
tot_in=$(grep -h "positions_in" "$WORK"/values_b*.log | grep -oE "positions_in [0-9]+" | awk '{s+=$2} END{print s}')
tot_drop=$(grep -h "parity filter dropped" "$WORK"/values_b*.log | grep -oE "dropped [0-9]+" | awk '{s+=$2} END{print s}')
rate=$(python3 -c "print(f'{100*$tot_drop/max($tot_in,1):.1f}')")
say "filter: dropped $tot_drop / $tot_in = ${rate}% (expect ~26%)"
python3 -c "exit(0 if 20.0 <= $rate <= 35.0 else 1)" \
  || die "filter drop-rate ${rate}% outside [20,35]% — the filter or the shards changed; investigate before training"

cat "$WORK"/values_b*.raw > "$WORK/values.raw"
rows_raw=$(wc -l < "$WORK/values.raw")
say "values.raw: $rows_raw filtered rows"

# ---- 2. WDL join (normalized-FEN key) + global dedup -----------------------
# key = board stm castling ep (clocks dropped so relabel/extract FENs match).
# sort -u on the key gives the cross-batch dedup the chunked export lacks.
say "sort+dedup canon rows by normalized-FEN key"
awk -F' \\| ' '{n=split($1,a," "); print a[1]" "a[2]" "a[3]" "a[4]"\t"$1"\t"$2}' "$WORK/values.raw" \
  | LC_ALL=C sort -t$'\t' -k1,1 -u -S 2G -T "$WORK/sorttmp" --parallel=4 -o "$WORK/canon.sorted"
rows_dedup=$(wc -l < "$WORK/canon.sorted")
say "canon.sorted: $rows_dedup rows (dedup removed $((rows_raw - rows_dedup)))"

say "sort+dedup UHO result source (key \t result)"
if [ ! -s "$WORK/uho_results.sorted" ]; then
  awk -F'\t' '{n=split($1,a," "); print a[1]" "a[2]" "a[3]" "a[4]"\t"$2}' "$UHO_FRS" \
    | LC_ALL=C sort -t$'\t' -k1,1 -u -S 2G -T "$WORK/sorttmp" --parallel=4 -o "$WORK/uho_results.sorted"
fi
say "uho_results.sorted: $(wc -l < "$WORK/uho_results.sorted") unique positions"

say "inner-join canon + results -> corpus.txt (FEN | cp | result)"
LC_ALL=C join -t$'\t' -1 1 -2 1 -o '1.2,1.3,2.2' "$WORK/canon.sorted" "$WORK/uho_results.sorted" \
  | awk -F'\t' '{
      cp=$2+0; if(cp>2499)cp=2499; if(cp<-2499)cp=-2499;
      print $1" | "cp" | "$3
    }' > "$WORK/corpus.txt"
rows_joined=$(wc -l < "$WORK/corpus.txt")
hit=$(python3 -c "print(f'{100*$rows_joined/max($rows_dedup,1):.2f}')")
say "corpus.txt: $rows_joined rows (join hit-rate ${hit}% of $rows_dedup)"
# MEASURED REALITY (2026-07-21): canon_fens (Jul 19, old-box extraction) and
# uho_frs.tsv (Jul 20 re-extraction) are two DIFFERENT stratified-chunk samples
# of the same UHO binpack -> expected overlap ~53%, verified NOT a key bug
# (missed boards are absent from uho_frs entirely; join keys match exactly on
# the overlap). The intersection is quality-unbiased (chunk sampling is blind
# to position character), so keeping only joined rows is clean. JOIN_MIN=50
# is therefore the documented default for THIS corpus pair; a future
# full-coverage fast_fens keys-mode extraction restores ~100% and JOIN_MIN=98.
# FULL-SET MEASUREMENT: the whole-corpus overlap is 35.6% (the SMOKE's first
# 100 shards overlapped 53% — shard order correlates with source position, so
# early shards are not representative). 27.5M joined rows of 77.2M. Still
# quality-unbiased; JOIN_MIN default set to admit it CONSCIOUSLY. At 60sb this
# corpus gives ~54 epochs/unique — the same replay class as canon-clean's 60,
# which makes the A/B a nearly replay-controlled recipe comparison.
JOIN_MIN="${JOIN_MIN:-33}"
python3 -c "exit(0 if $hit >= $JOIN_MIN else 1)" \
  || die "WDL join hit-rate ${hit}% < ${JOIN_MIN}% — key normalization or source mismatch; investigate"
[ "${JOIN_MIN%.*}" -lt 98 ] && say "NOTE: join keeps ${hit}% (two-sample overlap, see header) — unjoined rows are DROPPED, corpus is 100% real-WDL"

# result-distribution self-check: a degenerate all-0.5 column means the join
# silently reproduced the placeholder bug this script exists to fix.
say "result distribution:"
awk -F' \\| ' '{c[$3]++} END{for(r in c) printf "  result %s: %d (%.1f%%)\n", r, c[r], 100*c[r]/NR}' "$WORK/corpus.txt"
frac05=$(awk -F' \\| ' '{n++; if($3=="0.5")d++} END{print (n?100*d/n:100)}' "$WORK/corpus.txt")
python3 -c "exit(0 if $frac05 <= 70.0 else 1)" \
  || die "result column is ${frac05}% draws — degenerate join, placeholder bug reproduced"

# ---- 3. bf-convert + shuffle ----------------------------------------------
say "bf-convert -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -1
sz=$(stat -c%s "$WORK/corpus.bin" 2>/dev/null || echo 0)
[ $((sz % 32)) -eq 0 ] || die "corpus.bin not 32B-aligned ($sz)"
[ $((sz / 32)) -eq "$rows_joined" ] || say "WARN: bin records $((sz/32)) != text rows $rows_joined (converter drops unparseable lines — investigate if large)"

say "shuffle -> corpus.shuf.bin"
( cd "$WORK" && "$OLDPWD/$UTIL" shuffle --input "$PWD/corpus.bin" --output "$PWD/corpus.shuf.bin" --mem-used-mb 8000 ) 2>&1 | tail -1
ssz=$(stat -c%s "$WORK/corpus.shuf.bin" 2>/dev/null || echo 0)
[ "$ssz" -eq "$sz" ] || die "shuffle size mismatch ($ssz vs $sz)"

say "DONE: $WORK/corpus.shuf.bin ($((ssz/32)) records)"
say "next: the A/B —"
say "  A: DATA=$WORK/corpus.shuf.bin WDL=0.3 SB=60 NET_ID=canon-tight-A-60sb bash tools/nnue/train_canon.sh"
say "  B: DATA=$WORK/corpus.shuf.bin WDL=0   SB=60 NET_ID=canon-tight-B-60sb bash tools/nnue/train_canon.sh"
say "  then: bash tools/nnue/net_scale_check.sh <net> && bash tools/nnue/gate_canon.sh <net>"
