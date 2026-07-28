#!/usr/bin/env bash
# Convert corpora produced by export_paired_value_corpora.py to matched Bullet bins.
set -euo pipefail

ROOT="${1:?usage: build_matched_value_bins.sh EXPORT_ROOT [CONVERTER]}"
CONVERTER="${2:-${BULLETFORMAT_CONVERTER:-}}"
[[ -n "$CONVERTER" && -x "$CONVERTER" ]] || {
  echo "converter is missing or not executable: $CONVERTER" >&2
  exit 2
}
[[ -f "$ROOT/manifest.json" ]] || { echo "missing export manifest: $ROOT/manifest.json" >&2; exit 2; }

for teacher in sf16 reckless stockfish18; do
  out="$ROOT/$teacher/bin"
  mkdir -p "$out/train" "$out/val"
  for split in train val; do
    mapfile -t texts < <(find "$ROOT/$teacher/$split" -maxdepth 1 -name 'shard-*.txt' | sort)
    [[ ${#texts[@]} -gt 0 ]] || { echo "no text shards for $teacher/$split" >&2; exit 2; }
    bins=()
    for text in "${texts[@]}"; do
      bin="$out/$split/$(basename "${text%.txt}.bin")"
      "$CONVERTER" convert "$text" "$bin"
      rows=$(wc -l < "$text")
      bytes=$(stat -c%s "$bin")
      [[ "$bytes" -eq $((rows * 32)) ]] || {
        echo "binary size mismatch: $bin ($bytes != $((rows * 32)))" >&2
        exit 1
      }
      bins+=("$bin")
    done
    cat "${bins[@]}" > "$ROOT/$teacher/$split.bin"
  done
  train_bytes=$(stat -c%s "$ROOT/$teacher/train.bin")
  val_bytes=$(stat -c%s "$ROOT/$teacher/val.bin")
  echo "$teacher train=$((train_bytes / 32)) val=$((val_bytes / 32))"
done

# Matching is a hard invariant, not merely a line-count check.
for split in train val; do
  reference=$(stat -c%s "$ROOT/sf16/$split.bin")
  for teacher in reckless stockfish18; do
    actual=$(stat -c%s "$ROOT/$teacher/$split.bin")
    [[ "$actual" -eq "$reference" ]] || { echo "$split size mismatch for $teacher" >&2; exit 1; }
  done
done
