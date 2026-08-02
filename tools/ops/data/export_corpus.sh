#!/usr/bin/env bash
# tools/ops/data/export_corpus.sh — stage 4: labelled positions -> trainer bytes.
#
# PURPOSE
#   Turn stage-3 output into the 32-byte bulletformat records the trainer eats,
#   with the D46 §5 self-checks wired in so the builder DIES on recipe drift
#   instead of silently shipping a poisoned corpus. Both 2026-07 corpus
#   poisonings were silent recipe drift, not bad labels.
#
# USAGE
#   bash tools/ops/data/export_corpus.sh <labelled.tsv> [<labelled.tsv> ...] \
#        --out data/nnue/selfplay/<tag>/corpus
#   env: SHUFFLE=1|0 (default 1)   MIN_ROWS=1000   ALLOW_DEGENERATE=0
#
# PRODUCES  <out>.txt  <out>.bin  <out>.shuf.bin  <out>.meta.json
#
# ---------------------------------------------------------------------------
# THE WDL COLUMN IS REAL, AND THAT IS THIS PIPELINE'S MAIN STRUCTURAL ADVANTAGE.
#   The shard pipeline had to INNER-JOIN a result column onto positions labelled
#   from a different sample, and `build_canon_tight.sh` documents the cost: it
#   ships with JOIN_MIN=33 because the measured whole-corpus overlap was 35.6 %
#   (27.5M rows joined out of 77.2M). Two thirds of the labelled positions had
#   no result and were thrown away.
#   Here the result rides WITH the position from the game that produced it, so
#   the join hit-rate is 100 % BY CONSTRUCTION. The self-check below asserts
#   that rather than assuming it: a missing result is a bug, not a row to drop.
#
# CLAMP ORDER
#   Stage 3 already applied |raw| <= 10000 BEFORE clamping to +/-2499, so this
#   stage only has to preserve the invariant. It re-asserts |cp| <= 2499 and
#   dies rather than clamping again — a second clamp here would be exactly the
#   "clamp hides the magnitude criterion" bug that put 86k mate labels into an
#   aux corpus.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

CONV=${CONV:-$(asset tools/nnue/bf-convert/target/release/nnue_convert)}
SHUF=${SHUF:-$(asset tools/nnue/shuffle_records.py)}
SHUFFLE=${SHUFFLE:-1}
MIN_ROWS=${MIN_ROWS:-1000}
ALLOW_DEGENERATE=${ALLOW_DEGENERATE:-0}

INPUTS=(); OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=$2; shift 2 ;;
        *) INPUTS+=("$1"); shift ;;
    esac
done
[ -n "$OUT" ] || { log "FATAL: --out required"; exit 2; }
[ ${#INPUTS[@]} -gt 0 ] || { log "FATAL: no input TSVs"; exit 2; }
[ -x "$CONV" ] || { log "FATAL: nnue_convert missing: $CONV"; exit 2; }

mkdir -p "$(dirname "$OUT")"
OPS_LOG=${OPS_LOG:-$(dirname "$OUT")/export.log}; export OPS_LOG
die() { log "FATAL: $*"; exit 1; }

log "EXPORT ${#INPUTS[@]} input(s) -> $OUT"

# --- 1. text projection: FEN | cp | wdl -------------------------------------
# Column order is stage 3's: fen, label_cp, raw_cp, best, result, ply, ...
# Rows missing a result are a JOIN FAILURE and are counted, not silently kept.
python3 - "$OUT.txt" "${INPUTS[@]}" <<'PYEOF'
import sys, json, collections
out_path, inputs = sys.argv[1], sys.argv[2:]
n_in = n_out = n_nores = n_nocp = n_badclamp = n_unfiltered = 0
res_counts = collections.Counter()
frames = set()
with open(out_path, "w") as out:
    for p in inputs:
        with open(p) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(hdr)}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < len(hdr):
                    continue
                n_in += 1
                fen = f[ix["fen"]]
                cp = f[ix["label_cp"]]
                res = f[ix["result"]]
                # An audit-mode (--no-filter) file carries rows the D46 filter
                # REJECTED. Exporting one would put tactical labels into a
                # static-eval loss -- the exact 27.5%/32% poisoning the contract
                # exists to prevent. Refuse the whole file, loudly.
                if "drop_reason" in ix and f[ix["drop_reason"]].strip():
                    n_unfiltered += 1
                    continue
                if res not in ("1.0", "0.5", "0.0"):
                    n_nores += 1
                    continue
                if cp == "":
                    n_nocp += 1
                    continue
                v = int(cp)
                if abs(v) > 2499:
                    n_badclamp += 1
                    continue
                out.write(f"{fen} | {v} | {res}\n")
                res_counts[res] += 1
                n_out += 1
stats = dict(rows_in=n_in, rows_out=n_out, missing_result=n_nores,
             missing_cp=n_nocp, over_clamp=n_badclamp,
             d46_rejected_rows=n_unfiltered,
             results={k: v for k, v in res_counts.items()})
json.dump(stats, open(out_path + ".projstats.json", "w"), indent=2)
print(json.dumps(stats))
PYEOF
[ $? -eq 0 ] || die "text projection failed"

PS="$OUT.txt.projstats.json"
ROWS=$(python3 -c "import json;print(json.load(open('$PS'))['rows_out'])")
NORES=$(python3 -c "import json;print(json.load(open('$PS'))['missing_result'])")
NOCP=$(python3 -c "import json;print(json.load(open('$PS'))['missing_cp'])")
OVER=$(python3 -c "import json;print(json.load(open('$PS'))['over_clamp'])")
IN=$(python3 -c "import json;print(json.load(open('$PS'))['rows_in'])")

log "  rows_in=$IN rows_out=$ROWS missing_result=$NORES missing_cp=$NOCP over_clamp=$OVER"

# --- 2. self-checks (contract §5) -------------------------------------------
[ "$ROWS" -ge "$MIN_ROWS" ] || die "only $ROWS rows (< MIN_ROWS=$MIN_ROWS)"

# (b) WDL join hit-rate. 100% is the design target here, not an aspiration.
HIT=$(python3 -c "print(f'{100.0*$ROWS/max($IN,1):.2f}')")
python3 -c "exit(0 if $NORES == 0 else 1)" || \
    die "$NORES row(s) have no game result. In this pipeline the result rides
     WITH the position, so a miss is a STAGE-2/3 BUG (a lost game_id or a
     truncated PGN), not an acceptable join loss. Fix upstream."
[ "$OVER" -eq 0 ] || die "$OVER row(s) exceed the +/-2499 clamp — stage 3 did not clamp"
REJ=$(python3 -c "import json;print(json.load(open('$PS'))['d46_rejected_rows'])")
[ "$REJ" -eq 0 ] || die "$REJ row(s) carry a D46 drop_reason. This input was
     produced with --no-filter (AUDIT ONLY). Re-label without it; audit output
     must never become training data."
log "  WDL hit-rate ${HIT}% (design target 100%: result carried from the game)"

# (c) result distribution non-degenerate
python3 - "$PS" "$ALLOW_DEGENERATE" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); allow = sys.argv[2] == "1"
tot = s["rows_out"]; r = s["results"]
worst = max(r.values()) / max(tot, 1) * 100 if r else 100
for k in ("1.0", "0.5", "0.0"):
    print(f"  result {k}: {r.get(k,0)} ({100*r.get(k,0)/max(tot,1):.1f}%)")
if worst > 70.0 and not allow:
    print(f"FATAL: degenerate result distribution ({worst:.1f}% one value). "
          f"Reference UHO extraction is ~W42/D47/L11 White-relative. "
          f"Set ALLOW_DEGENERATE=1 only if you know why.", file=sys.stderr)
    sys.exit(1)
PYEOF
[ $? -eq 0 ] || die "degenerate result distribution"

# --- 3. convert + alignment check -------------------------------------------
"$CONV" convert "$OUT.txt" "$OUT.bin" >/dev/null 2>&1 || die "nnue_convert failed"
SZ=$(stat -c%s "$OUT.bin")
[ $((SZ % 32)) -eq 0 ] || die "corpus.bin not 32B-aligned ($SZ)"
NREC=$((SZ / 32))
[ "$NREC" -eq "$ROWS" ] || die "bin records $NREC != text rows $ROWS (silent parse drops)"
log "  converted: $NREC records, 32B-aligned, text/bin match"

# --- 4. global shuffle (the JAX loader reads contiguous windows) -------------
FINAL="$OUT.bin"
if [ "$SHUFFLE" = "1" ] && [ -s "$SHUF" ]; then
    need_mem 2000 || die "not enough memory to shuffle"
    .venv/bin/python "$SHUF" "$OUT.bin" "$OUT.shuf.bin" >/dev/null 2>&1 \
        || die "shuffle failed"
    SSZ=$(stat -c%s "$OUT.shuf.bin")
    [ "$SSZ" -eq "$SZ" ] || die "shuffle size mismatch ($SSZ != $SZ)"
    FINAL="$OUT.shuf.bin"
    log "  shuffled -> $FINAL"
fi

# --- 5. meta / provenance ----------------------------------------------------
python3 - "$OUT.meta.json" "$PS" "$NREC" "$FINAL" "${INPUTS[@]}" <<'PYEOF'
import json, sys, os, hashlib, datetime
meta_p, ps, nrec, final = sys.argv[1:5]
inputs = sys.argv[5:]
def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
srcs = []
for p in inputs:
    st = p + ".stats.json"
    srcs.append(dict(tsv=p, label_stats=json.load(open(st)) if os.path.exists(st) else None))
json.dump(dict(
    created_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    records=int(nrec), artifact=final, sha256=sha(final),
    projection=json.load(open(ps)), sources=srcs,
    contract="D46: filter at label layer (stage 3), clamp after filter, real WDL",
), open(meta_p, "w"), indent=2)
print("meta ->", meta_p)
PYEOF

log "done: $FINAL ($NREC records)"
log "NEXT: register a D38 ledger row (tools/ledger/ledger.py add-data) before training"
