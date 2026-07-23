#!/usr/bin/env bash
# perf_study4_bench.sh — post-tournament quiet-window measurements for
# docs/research/perf-study-4-eval-2026-07-22.md (§7).
#
# Confirms/refutes the study's top claims:
#   C1  threat channel share + latency-vs-bandwidth verdict at d47c dims
#   C2  head cost share + codegen-width audit (f32 tail, l1_dot, dispatch)
#   C3  policy-machinery overhead when the picker does not consume it
#   C4  T10 smallnet gate fire-rate inputs (offline histogram; optional patch)
#   B   clean NPS ladder re-baseline + THP/dTLB environment audit
#
# DO NOT run while any match/tournament/training job is live. The script
# refuses to start if it smells contention.
#
# Usage:
#   bash tools/perf_study4_bench.sh [all|env|build|ladder|c1|c2|c3|c4|svd]
# Defaults to "all". Outputs land in $OUT (printed at start); a summary is
# appended to $OUT/summary.md as sections complete.
#
# Optional env:
#   NET=...        d47c CENN path      (default: newest data/nnue/nets/d47c-*.cenn)
#   NET768=...     768-era incumbent   (default: data/nnue/nets/d40-uho-720sb.cenn)
#   DEPTH=12       bench depth for perf lanes
#   CORE=14        pinned core (SMT sibling should stay idle)
#   PERF=perf      perf binary (native Arch: plain `perf` works)
#
# ── C4 optional counter patch (apply manually, rebuild, run c4, revert) ──
# The true in-tree gate fire-rate needs one counter. 25-line shape:
#   * sota_context.rs: add `evals: u64, gated: u64` to SotaContext; in
#     evaluate(): evals += 1 up front; inside the smallnet early-return
#     branch: gated += 1. Behind `ARCHON_SMALLNET_STATS=1`, print
#     "smallnet_stats evals=<n> gated=<n>" from a Drop impl (stderr).
#   * Run with ARCHON_SMALLNET=<the d47c net itself> ARCHON_SMALLNET_THRESH=T
#     ARCHON_SMALLNET_MAT_GATE=M — routing changes values (the stand-in
#     smallnet zeroes threats), so use it ONLY to count, never to gate Elo.
#   * Sweep (M,T) ∈ {400,600,800,1000}×{300,600}; report gated/evals.
# Without the patch, c4 still produces the offline (game-position-weighted)
# material-imbalance histogram, which lower-bounds p.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${1:-all}"
DEPTH="${DEPTH:-12}"
CORE="${CORE:-14}"
PERF="${PERF:-perf}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$REPO/data/perf/study4-$STAMP}"
mkdir -p "$OUT"
SUM="$OUT/summary.md"

NET="${NET:-$(ls -t data/nnue/nets/d47c-*.cenn 2>/dev/null | head -1 || true)}"
NET768="${NET768:-data/nnue/nets/d40-uho-720sb.cenn}"

log() { echo "[study4] $*" | tee -a "$SUM"; }
sec() { printf '\n## %s\n\n' "$*" >> "$SUM"; echo "=== $* ==="; }

# ── guards ──────────────────────────────────────────────────────────────
guard() {
  if pgrep -x fastchess >/dev/null 2>&1; then
    echo "REFUSING: fastchess is running (tournament live)."; exit 1
  fi
  if pgrep -f "run_rr_tournament|gauntlet.sh|sprt.sh" >/dev/null 2>&1; then
    echo "REFUSING: a match harness is running."; exit 1
  fi
  if pgrep -f "bullet|d47c_train|trainer" >/dev/null 2>&1; then
    echo "WARNING: a trainer-looking process is alive — verify before trusting numbers."
  fi
  local load; load="$(cut -d' ' -f1 /proc/loadavg)"
  awk -v l="$load" 'BEGIN{ if (l > 1.5) { exit 1 } }' \
    || { echo "REFUSING: loadavg $load > 1.5 — box not quiet."; exit 1; }
  [ -n "$NET" ] && [ -f "$NET" ] || { echo "REFUSING: NET not found ($NET)"; exit 1; }
}

# median-of-3 NPS from the pinned fingerprint bench; echoes "nodes nps"
bench3() { # $1 binary, $2.. bench args
  local bin="$1"; shift
  local best_nodes="" nps_list=()
  for _ in 1 2 3; do
    local out
    out="$(taskset -c "$CORE" "$bin" bench 16 1 "$DEPTH" default depth "$@" 2>&1 | tail -20)"
    local nodes nps
    nodes="$(echo "$out" | awk '/Nodes searched/{print $NF}' | tail -1)"
    nps="$(echo "$out" | awk '/Nodes\/second|nps/{print $NF}' | tail -1)"
    best_nodes="$nodes"; nps_list+=("$nps")
  done
  local median
  median="$(printf '%s\n' "${nps_list[@]}" | sort -n | sed -n 2p)"
  echo "$best_nodes $median (${nps_list[*]})"
}

# ── env: environment + THP/TLB audit (claim B / item #8) ───────────────
do_env() {
  sec "environment"
  { uname -a
    grep -m1 'model name' /proc/cpuinfo
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
    echo "THP defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo n/a)"
    echo "NET=$NET ($(stat -c%s "$NET" 2>/dev/null || echo '?') bytes)"
    echo "NET768=$NET768"
    "$PERF" --version || echo "perf MISSING — install linux-tools; C1/C2/C3 degraded to wall-clock"
  } | tee -a "$SUM"
}

# ── build: the two lanes (allowed in the quiet window) ──────────────────
do_build() {
  sec "builds"
  RUSTFLAGS="-C target-cpu=native" cargo build --release --features sota 2>&1 | tail -2
  cp target/release/chessengine "$OUT/archon-rel"
  RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
    cargo build --profile profiling --features sota 2>&1 | tail -2
  cp target/profiling/chessengine "$OUT/archon-prof"
  log "binaries: $OUT/archon-rel, $OUT/archon-prof"
}

# ── ladder: clean NPS re-baseline (claim B) ─────────────────────────────
do_ladder() {
  sec "NPS ladder (pinned c$CORE, median-of-3, depth $DEPTH)"
  local BIN="$OUT/archon-rel"
  log "d47c (+cpol if present): $(bench3 "$BIN" "$NET")"
  # cpol-less lane: point at a sidecar-free copy of the same net
  local nocpol="$OUT/net-nocpol.cenn"
  ln -sf "$(realpath "$NET")" "$nocpol"
  log "d47c (no cpol):          $(bench3 "$BIN" "$nocpol")"
  [ -f "$NET768" ] && log "incumbent-768:           $(bench3 "$BIN" "$NET768")"
  log "PSQT (no net):           $(bench3 "$BIN")"
  # external anchors (optional, same positions comparison is in the studies)
  for e in engines/bin/stockfish18 engines/bin/stockfish16; do
    [ -x "$e" ] && log "$(basename "$e") bench: $(taskset -c "$CORE" "$e" bench 2>&1 | grep -Ei 'nodes/second|nps' | tail -1)"
  done
}

# ── C1: threat-channel share + latency/bandwidth verdict ────────────────
do_c1() {
  sec "C1 threat channel (perf record/stat, d47c vs 768)"
  local BIN="$OUT/archon-prof"
  for lane in d47c:"$NET" net768:"$NET768"; do
    local name="${lane%%:*}" net="${lane#*:}"
    [ -f "$net" ] || { log "skip $name (missing $net)"; continue; }
    "$PERF" record -F 999 -g --call-graph fp -o "$OUT/c1-$name.perf.data" -- \
      taskset -c "$CORE" "$BIN" bench 16 1 "$DEPTH" default depth "$net" >/dev/null
    "$PERF" report -i "$OUT/c1-$name.perf.data" --stdio --no-children --percent-limit 0.5 \
      > "$OUT/c1-$name.report.txt" 2>/dev/null
    {
      echo "--- $name top eval symbols:"
      grep -E 'i8_deltas|i16_deltas|build_deltas|record_square_touch|resolve|threat_index|compute|activate|l1_dot|find_nnz|evaluate|finny|memmove|malloc|free' \
        "$OUT/c1-$name.report.txt" | head -25
    } | tee -a "$SUM"
    "$PERF" stat -o "$OUT/c1-$name.stat.txt" \
      -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses -- \
      taskset -c "$CORE" "$BIN" bench 16 1 "$DEPTH" default depth "$net" >/dev/null 2>&1 || true
    tee -a "$SUM" < "$OUT/c1-$name.stat.txt" | tail -14
  done
  log "VERDICT INPUTS: compare i8_deltas share d47c vs 768 (study EST: 1.4-1.6x);"
  log "LLC-load-miss rate — if d47c >> 768 lane, the path went bandwidth-bound (re-opens int4)."
  log "dTLB-load-misses — if large and THP!=always, item #8 (madvise hugepages) fires."
}

# ── C2: head share + codegen audit ──────────────────────────────────────
do_c2() {
  sec "C2 head codegen audit (objdump on the release binary)"
  local BIN="$OUT/archon-rel" ASM="$OUT/c2-evaluate.asm"
  objdump -d --no-show-raw-insn "$BIN" > "$OUT/c2-full.asm"
  # carve out the SotaContext::evaluate + head neighborhood
  awk '/<.*SotaContext.*evaluate.*>:/,/^$/' "$OUT/c2-full.asm" > "$ASM" || true
  {
    echo "FMA census (whole binary): scalar_ss=$(grep -c 'vfmadd[0-9]*ss' "$OUT/c2-full.asm" || true)" \
         "xmm_ps=$(grep 'vfmadd[0-9]*ps' "$OUT/c2-full.asm" | grep -c xmm || true)" \
         "ymm_ps=$(grep 'vfmadd[0-9]*ps' "$OUT/c2-full.asm" | grep -c ymm || true)" \
         "zmm_ps=$(grep 'vfmadd[0-9]*ps' "$OUT/c2-full.asm" | grep -c zmm || true)"
    echo "vpdpbusd sites: $(grep -c vpdpbusd "$OUT/c2-full.asm" || true)"
    echo "vpcompressw sites (expect 0 until item #2d): $(grep -c vpcompressw "$OUT/c2-full.asm" || true)"
    echo "LazyLock KERNELS check pattern (expect >0 until #2c): $(grep -c 'KERNELS' "$OUT/c2-full.asm" || true)"
  } | tee -a "$SUM"
  log "READ: if xmm_ps dominates ymm+zmm in the evaluate region, the 128x32 f32 tail"
  log "is still width-stuck (round-2 F4) => item #2a's explicit zmm kernel pays ~2-4x there."
}

# ── C3: policy machinery overhead ───────────────────────────────────────
do_c3() {
  sec "C3 policy overhead (.cpol present vs absent)"
  local BIN="$OUT/archon-prof"
  local cpol="${NET%.cenn}.cpol"
  if [ ! -f "$cpol" ]; then log "no .cpol beside $NET — C3 skipped"; return; fi
  # lane A: sidecar visible (symlink net+cpol into OUT); lane B: net only
  ln -sf "$(realpath "$NET")"  "$OUT/pol.cenn";  ln -sf "$(realpath "$cpol")" "$OUT/pol.cpol"
  ln -sf "$(realpath "$NET")"  "$OUT/nopol.cenn"
  for lane in pol nopol; do
    "$PERF" record -F 999 -g --call-graph fp -o "$OUT/c3-$lane.perf.data" -- \
      taskset -c "$CORE" "$BIN" bench 16 1 "$DEPTH" default depth "$OUT/$lane.cenn" >/dev/null
    "$PERF" report -i "$OUT/c3-$lane.perf.data" --stdio --no-children --percent-limit 0.1 \
      > "$OUT/c3-$lane.report.txt" 2>/dev/null
    {
      echo "--- $lane: nodes/nps: $(bench3 "$OUT/archon-rel" "$OUT/$lane.cenn")"
      echo "--- $lane: alloc + policy symbols:"
      grep -E 'malloc|free|row_score|best_move|generate_legal|forward_full|Box' \
        "$OUT/c3-$lane.report.txt" | head -15
    } | tee -a "$SUM"
  done
  log "READ: trees differ (policy fills picker slots) — attribute by profile share,"
  log "not NPS delta. Study EST: stash(Box+copy) 1-2% + consult 2-5% in the pol lane."
}

# ── C4: T10 gate fire-rate inputs ───────────────────────────────────────
do_c4() {
  sec "C4 material-imbalance histogram (game-weighted lower bound for p)"
  local SAMPLE
  SAMPLE="$(ls -t data/nnue/corpora/canon*/*.bin data/nnue/*.bin 2>/dev/null | head -1 || true)"
  if [ -z "$SAMPLE" ]; then log "no local binpack sample found — set SAMPLE= and rerun c4"; return; fi
  python3 - "$SAMPLE" <<'EOF' | tee -a "$OUT/summary.md"
import sys, struct, collections
# bulletformat ChessBoard: 32 bytes/record — occ u64, pieces 16B (nibbles), score i16, result u8, ksq u8, oksq u8, pad
# material imbalance from the nibble-packed piece list; values: P100 N316 B328 R493 Q982 (SF-ish cp class)
VAL = {0:100, 1:316, 2:328, 3:493, 4:982, 5:0}
path = sys.argv[1]
hist = collections.Counter(); n = 0
with open(path, 'rb') as f:
    while n < 2_000_000:
        rec = f.read(32)
        if len(rec) < 32: break
        occ = struct.unpack('<Q', rec[:8])[0]
        pieces = rec[8:24]
        mat = 0; idx = 0
        for i in range(64):
            if occ >> i & 1:
                nib = (pieces[idx // 2] >> (4 * (idx % 2))) & 0xF; idx += 1
                pt, black = nib & 7, nib >> 3
                v = VAL.get(pt if pt < 6 else 5, 0)
                mat += -v if black else v
        a = abs(mat); n += 1
        for t in (400, 600, 800, 1000):
            if a >= t: hist[t] += 1
print(f"sample={path} n={n}")
for t in (400, 600, 800, 1000):
    print(f"  |material| >= {t}cp: {hist[t]/max(n,1):.3%}")
print("NOTE: game-position-weighted (a LOWER bound for search-tree p in decided phases);")
print("the true in-tree rate needs the ARCHON_SMALLNET_STATS counter patch (header).")
EOF
  log "If the counter patch is applied: rerun a bench + startpos/UHO-game 'go depth 15'"
  log "with ARCHON_SMALLNET=<net> THRESH/MAT_GATE sweeps and record gated/evals here."
}

# ── svd: 3-minute re-probe of the 1280-wide threat table (checkbox) ─────
do_svd() {
  sec "SVD re-probe (d47c threat table — expect flat spectrum, keep low-rank dead)"
  log "reuse scratchpad/svd_probe.py from study 3 against $NET (offsets per sota_loader.rs);"
  log "if rank-256 energy is still ~50%, the low-rank kill stands at 1280 too."
}

guard
do_env
case "$MODE" in
  env)    ;;
  build)  do_build ;;
  ladder) do_build; do_ladder ;;
  c1)     do_build; do_c1 ;;
  c2)     do_build; do_c2 ;;
  c3)     do_build; do_c3 ;;
  c4)     do_c4 ;;
  svd)    do_svd ;;
  all)    do_build; do_ladder; do_c1; do_c2; do_c3; do_c4; do_svd ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

log "DONE — outputs in $OUT (summary: $SUM)"
log "Seed docs/perf-baseline.md's NNUE section from the ladder + C1 stat lines."
