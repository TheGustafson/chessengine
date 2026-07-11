#!/usr/bin/env bash
# Assemble a bullet-trainable NNUE corpus from gen_sf_data.py text shards.
#
# design/15-nnue-datagen.md §3.2.2 pipeline + §3.6 storage/meta:
#   raw/*.txt  --bulletformat convert-->  bin/*.bin  (32-byte ChessBoard)
#              --bullet-utils shuffle-->  shuffled/*.bin
#              --bullet-utils interleave-> train.bin  (+ carved val.bin ~1%)
#   + meta.json (regenerability: SF rev/sha/ARCH, repo rev, node budget, every
#     filter/adjudication threshold, seeds, per-shard counts, piece/phase
#     histograms, timestamp -- §3.6).
#
# We do NOT hand-pack the 32-byte record (§3.2.1): the binary is produced by
# bulletformat's own converter, located via env. The exact convert/shuffle/
# interleave CLI is PINNED against the vendored bullet+bulletformat commit; this
# script parametrises it:
#
#   BULLETFORMAT_CONVERT  full command; receives:  <in.txt> <out.bin>
#   BULLET_UTILS_SHUFFLE  full command; receives:  <in.bin> <out.bin>
#   BULLET_UTILS_ILEAVE   full command; receives:  <out.bin> <in1.bin>...
#
# The working converter + the full build/train recipe are documented and pinned
# (bulletformat 1.8.0, bullet cebc78a) in tools/nnue/TRAINING.md; the converter
# reference source is tools/nnue/bf_convert.rs. Validated end-to-end on the
# shard_43 smoke corpus 2026-07-11 (raw text -> 32B bin -> GPU-trained net).
# Typical wiring:
#   BULLETFORMAT_CONVERT="$SCRATCH/nnue_convert/target/release/nnue_convert convert" \
#     tools/nnue/build_corpus.sh --seed 43 --require-convert
#
# When those are unset the script still validates the raw shards and writes a
# complete meta.json, marking the binary corpus PENDING (the smoke path for
# this workflow -- the bullet-ingestion gate is design/15 step 4 / P13). Pass
# --require-convert to hard-fail instead.
#
# Usage:
#   tools/nnue/build_corpus.sh --seed 1 [--raw data/nnue/raw] [--out data/nnue] \
#       [--sf engines/stockfish-datagen] [--val-pct 1] [--require-convert]
set -euo pipefail

RAW=data/nnue/raw
OUT=data/nnue
SF=engines/stockfish-datagen
SEED=""
VAL_PCT=1
REQUIRE_CONVERT=0
REC=32  # bytes per bulletformat::ChessBoard record

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw) RAW="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sf) SF="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --val-pct) VAL_PCT="$2"; shift 2 ;;
    --require-convert) REQUIRE_CONVERT=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

[[ -d "$RAW" ]] || { echo "no raw dir: $RAW" >&2; exit 2; }
mapfile -t SHARDS < <(find "$RAW" -maxdepth 1 -name 'shard_*.txt' | sort)
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "no shard_*.txt in $RAW" >&2; exit 2; }

mkdir -p "$OUT/bin" "$OUT/shuffled"

# --- provenance for meta.json -----------------------------------------------
sf_sha="$( [[ -f "$SF" ]] && sha256sum "$SF" | awk '{print $1}' || echo "MISSING" )"
sf_rev="$(git -C stockfish_reference/Stockfish rev-parse HEAD 2>/dev/null || echo unknown)"
sf_arch="$( [[ -f "$SF" ]] && "$SF" compiler 2>/dev/null | awk -F': *' '/Compilation architecture/{print $2; exit}' || echo unknown )"
repo_rev="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

total_lines=0
declare -a per_shard_counts
for s in "${SHARDS[@]}"; do
  c=$(wc -l < "$s")
  per_shard_counts+=("$(basename "$s"):$c")
  total_lines=$((total_lines + c))
done
echo "[build_corpus] ${#SHARDS[@]} shards, $total_lines text lines total"

# --- binary conversion (optional external tools) ----------------------------
BIN_STATUS="pending"
train_count=0
val_count=0
if [[ -n "${BULLETFORMAT_CONVERT:-}" ]]; then
  echo "[build_corpus] converting text -> 32B bin via: $BULLETFORMAT_CONVERT"
  bins=()
  for s in "${SHARDS[@]}"; do
    b="$OUT/bin/$(basename "${s%.txt}.bin")"
    # shellcheck disable=SC2086
    $BULLETFORMAT_CONVERT "$s" "$b"
    sz=$(stat -c%s "$b")
    [[ $((sz % REC)) -eq 0 ]] || { echo "bin $b not a multiple of ${REC}B" >&2; exit 1; }
    recs=$((sz / REC))
    lc=$(wc -l < "$s")
    [[ "$recs" -eq "$lc" ]] || { echo "record/line mismatch $b: $recs != $lc" >&2; exit 1; }
    bins+=("$b")
  done
  # shuffle (per shard) then interleave
  shufd=()
  for b in "${bins[@]}"; do
    o="$OUT/shuffled/$(basename "$b")"
    if [[ -n "${BULLET_UTILS_SHUFFLE:-}" ]]; then
      # shellcheck disable=SC2086
      $BULLET_UTILS_SHUFFLE "$b" "$o"
    else
      cp "$b" "$o"
    fi
    shufd+=("$o")
  done
  full="$OUT/train_full.bin"
  if [[ -n "${BULLET_UTILS_ILEAVE:-}" ]]; then
    # shellcheck disable=SC2086
    $BULLET_UTILS_ILEAVE "$full" "${shufd[@]}"
  else
    cat "${shufd[@]}" > "$full"
  fi
  # carve val.bin (~VAL_PCT%) on a 32-byte record boundary -- pure slicing of
  # fixed-size records, no packing knowledge needed.
  tot=$(( $(stat -c%s "$full") / REC ))
  val_count=$(( tot * VAL_PCT / 100 ))
  train_count=$(( tot - val_count ))
  head -c $(( train_count * REC )) "$full" > "$OUT/train.bin"
  tail -c $(( val_count * REC )) "$full" > "$OUT/val.bin"
  rm -f "$full"
  BIN_STATUS="built"
  echo "[build_corpus] train.bin=$train_count  val.bin=$val_count records"
else
  msg="[build_corpus] BULLETFORMAT_CONVERT unset -> binary corpus PENDING (design/15 step 4 / P13)."
  if [[ "$REQUIRE_CONVERT" -eq 1 ]]; then
    echo "$msg" >&2
    echo "  set BULLETFORMAT_CONVERT (and optionally BULLET_UTILS_SHUFFLE / _ILEAVE)" >&2
    exit 3
  fi
  echo "$msg"
  echo "       (raw text validated + meta.json written; run validate_sample.py separately)"
fi

# --- meta.json (§3.6) -------------------------------------------------------
python3 - "$OUT" "$RAW" "$SEED" "$sf_sha" "$sf_rev" "$sf_arch" "$repo_rev" \
             "$ts" "$total_lines" "$BIN_STATUS" "$train_count" "$val_count" <<'PY'
import glob, json, os, sys
out, raw, seed, sf_sha, sf_rev, sf_arch, repo_rev, ts, total, binst, trn, val = sys.argv[1:13]

# aggregate the per-worker stats sidecars written by gen_sf_data.py
agg = {}
pc = {}
ph = {}
shard_counts = {}
thresholds_seen = None
for sp in sorted(glob.glob(os.path.join(raw, "shard_*.stats.json"))):
    d = json.load(open(sp))
    name = os.path.basename(sp).replace(".stats.json", "")
    shard_counts[name] = d.get("positions_written", 0)
    for k, v in d.items():
        if k == "piece_hist":
            for kk, vv in v.items():
                pc[kk] = pc.get(kk, 0) + vv
        elif k == "phase_hist":
            for kk, vv in v.items():
                ph[kk] = ph.get(kk, 0) + vv
        elif isinstance(v, int):
            agg[k] = agg.get(k, 0) + v

meta = {
    "schema": "design/15 bulletformat ChessBoard / Chess768 text-interchange",
    "timestamp_utc": ts,
    "seed": int(seed) if seed else None,
    "teacher_stockfish": {
        "git_rev": sf_rev,
        "binary_sha256": sf_sha,
        "arch": sf_arch,
        "note": "GPLv3 scaffold, gitignored, never shipped (D27/D28)",
    },
    "generating_repo_git_rev": repo_rev,
    "bench_signature": "N/A (Tier-A labels are SF's, not our engine's; D24 Bench applies to Tier-B self-play)",
    "corpus": {
        "raw_text_lines": int(total),
        "binary_status": binst,
        "train_records": int(trn),
        "val_records": int(val),
        "per_shard_positions": shard_counts,
    },
    "run_stats": agg,
    "piece_count_histogram": {k: pc[k] for k in sorted(pc, key=int)},
    "phase_histogram": {k: ph[k] for k in sorted(ph, key=int)},
    "format": {
        "record_bytes": 32,
        "input_type": "Chess768 (768 = 2x6x64, doc-11 §3.4.1)",
        "score": "White-relative UCI cp on disk; bullet re-canonicalises to STM (§3.2.4)",
        "result": "White-relative 1.0/0.5/0.0",
        "train_recommended": "--scale 400 --wdl 0.3 (doc-11 lambda=0.7 eval; §3.2.5)",
    },
}
path = os.path.join(out, "meta.json")
json.dump(meta, open(path, "w"), indent=2)
open(path, "a").write("\n")
print(f"[build_corpus] wrote {path}")
PY

echo "[build_corpus] done (binary_status=$BIN_STATUS)"
