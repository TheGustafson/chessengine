#!/usr/bin/env bash
# Build the hard-negative training corpus from deep-labeled ceobs (SF18 cp) +
# the REAL game WDL (from the mined games). Unlike the canon relabel, these
# positions HAVE real WDL, so we train with WDL>0 (the fix for the WDL=0 disaster).
#   values.raw  = "FEN | cp | 0.5"  (obs-tool export-value; WDL is placeholder)
#   all.tsv     = "FEN\twdl"        (real White-relative game result)
#   -> disk sort+join by FEN -> corpus.txt "FEN | clamp(cp) | real_wdl" -> .bin -> shuffle
set -uo pipefail
cd "$(dirname "$0")/../.."
LAB=data/nnue/hardneg/labeled
WORK=data/nnue/hardneg/corpus
OBS=target/release/obs-tool
CONV=tools/nnue/bf-convert/target/release/nnue_convert
UTIL=.archon-state/toolchains/bullet/target/release/bullet-utils
mkdir -p "$WORK/tmp"
say(){ echo "[hardneg-corpus $(date -u +%H:%M:%S)] $*"; }

say "export-value over $(ls "$LAB"/chunks/ceobs_*.ceobs 2>/dev/null | wc -l) ceobs chunks (trainer-parity filter)"
# --filter-sfbinpack (2026-07-21): the full 5-criterion quiet filter at export.
# The earlier corpus had NO filter (9.17% in-check poison, the -582 mechanism);
# the "clean" re-filter used canon_filter = in-check only, leaving capture-best
# tactical labels a static eval can't represent. Hard-negs are mined from
# losses, so expect a LARGE capture-best drop here — that is the filter working,
# not data loss: tactical refutations belong to search, not the value target.
"$OBS" export-value "$WORK/values.raw" "$LAB"/chunks/ceobs_*.ceobs --filter-sfbinpack 2>&1 | tail -2
[ -s "$WORK/values.raw" ] || { say "FATAL: empty values.raw"; exit 1; }

# join real WDL from all.tsv by FEN (disk sort+join; TAB delimiter so FEN is one key)
say "join real WDL by FEN"
LC_ALL=C sort -t$'\t' -k1,1 -u -S 512M --parallel=4 data/nnue/hardneg/all.tsv > "$WORK/wdl.sorted"
awk -F' \\| ' '{print $1"\t"$2}' "$WORK/values.raw" | LC_ALL=C sort -t$'\t' -k1,1 -S 512M --parallel=4 > "$WORK/vals.sorted"
LC_ALL=C join -t$'\t' -1 1 -2 1 "$WORK/vals.sorted" "$WORK/wdl.sorted" \
  | awk -F'\t' '{cp=$2+0; if(cp>2499)cp=2499; if(cp<-2499)cp=-2499; print $1" | "cp" | "$3}' > "$WORK/corpus.txt"
rows=$(wc -l < "$WORK/corpus.txt"); vals=$(wc -l < "$WORK/vals.sorted")
say "joined $rows rows (of $vals; unmatched dropped)"
[ "$rows" -gt 0 ] || { say "FATAL: 0 joined rows"; exit 1; }

say "bf-convert -> corpus.bin"
"$CONV" convert "$WORK/corpus.txt" "$WORK/corpus.bin" 2>&1 | tail -1
say "shuffle -> corpus.shuf.bin"
( cd "$WORK" && "$OLDPWD/$UTIL" shuffle --input "$PWD/corpus.bin" --output "$PWD/corpus.shuf.bin" --mem-used-mb 4000 ) 2>&1 | tail -1
say "DONE: corpus.shuf.bin $(stat -c%s "$WORK/corpus.shuf.bin" 2>/dev/null) bytes ($(( $(stat -c%s "$WORK/corpus.shuf.bin")/32 )) records)"
touch "$WORK/DONE"
