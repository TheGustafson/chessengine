# Stage-3 TRAIN EXIT runbook (pinned ~08:3x 2026-07-23)

Execute in order at `[stage3] TRAIN EXIT 0` (~11:00). Box rules: builds
allowed again until stage-4 launches; conc-1 for any match; no equal-time
anything until the box is otherwise idle.

## 1. Builds + verification (~30 min)
- `bash tools/check.sh` on main — MUST reproduce `Bench: 701605`
  (the timeman tunables promotion f96fdab compiles to plain consts
  without the tune feature; any drift = bug in the promotion).
- Tune build sanity: `cargo build --release --features "sota tune"` +
  `./target/release/chessengine tunables | grep -c .` (expect ~28 TM
  params among them); tune build is for SPSA GAMES ONLY, never ships.
- Rebuild native engine only if check.sh touched anything (it won't).

## 2. Endpoint gates on d47d-stage3-sb240 (~1.5h, overlaps §3)
- Pack sb240 (+ PSQ1 section rides along) via watcher PACK_ENV; scale
  check vs stage-2 (archon-d47 both sides).
- 300g fixed-node 5k vs stage2-base-nocpol (eval_match, conc 1) —
  the promotion-trajectory number. Heldout final from the log.
- Ledger: add-net d47d-stage3-sb240 (recipe: driver+corpora rows,
  λ=0.001, mix .40/.35/.25, warm=stage-2-sb240) + add-exp for the 300g.
- NO equal-time, NO SF16 anchor (slides to post-chain per rulings).

## 3. Disk cleanup (frees ~45G; needed BEFORE stage-4 dumps)
- d47c-stage2 per-sb quants: keep sb{30,60,...,240} + final; delete the
  other ~210 files (~30G). Raws stay (only 2 exist).
- d47d-stage3 checkpoints: keep sb{60,120,180,210,240} quant+raw,
  delete the rest (~10G). jax-s2/s3 smoke artifacts: delete (~3G).
- THEN fetch wrongNNUE_02_d9 (5.4G, was disk-guard-blocked).

## 4. D46 audits (T60T70, T75; ~1h, 2 SF workers, nice)
- `uv run tools/nnue/audit_label_semantics.py --source fishpack
  --fishpack data/nnue/binpacks/raw/T60T70wIsRightFarseer.binpack
  --count 400` → OLS slope = native divisor → training DIV =
  1.9 × (native/2.82). Same for farseerT75. Ledger-annotate both rows.
- Mixed-source pack caveat: if T60T70's fit is bimodal/low-Pearson
  (it blends Lc0+wrongIsRight+farseer), fall back to per-band DIV or
  exclude the pack from stage-4 (T75 alone + rehearsal is a valid mix).

## 5. Conversions (band-sized for disk; ~2h)
- T60T70: D40_CONVERT_DUMP, band = D40_RANDOM_SKIP_PCT trick does NOT
  do upper bounds — use D40_HELDOUT_BAND=75:87 instead (12% band,
  trainer-parity filter, ~. . . G) — NOTE: heldout-band mode REPLACES the
  skip test and hits the dump path; requires D40_HELDOUT_DUMP not
  CONVERT (single-pass!): use D40_CONVERT_DUMP + a NEW env
  D40_CONVERT_BAND=75:87 (5-line addition to d40_features.rs: reuse
  HELDOUT_BAND parsing for the convert keep-test) — implement in §1's
  build window. DIV from §4.
- T75: same, band 75:87. Shuffle both with --unlink-input; ledger rows.
- Mix weights (pinned): T60T70 .30 / T75 .30 / UHO .20 / fishpack .10 /
  T74 .10 — new-coverage-forward with rehearsal floor; adjust only if
  §4 excludes a pack.

## 6. Stage-4 launch (target ~15:00-16:00)
- stage3_train.py generalization: WARM=stage-3-sb240-raw, lr=5e-4,
  warmup 2sb ramp, ema_decay=0.99984, new MIX, superbatches 240,
  lr_end_superbatch=331. Telemetry + monitor + watcher (v2 protocol,
  baseline stays stage2-base-nocpol for cross-stage comparability;
  add a stage-3-sb240 baseline screen at sb120+).
- Build silence resumes at launch.
