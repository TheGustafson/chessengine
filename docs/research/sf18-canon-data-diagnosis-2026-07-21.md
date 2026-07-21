# SF18-canon data forensics — independent diagnosis (2026-07-21)

Independent first-principles audit of the SF18-canonical training data
(`data/nnue/canon-v1/build/`), commissioned to answer: *what is actually wrong
with the canon data that makes nets converge to low loss but play
catastrophically?* All prior conclusions were treated as unverified hypotheses
and re-derived. Probe scripts + raw outputs: `data/nnue/canon-v1/build/diag/`
(small, reproducible; the multi-GB slice files were deleted after use).

## Verdict

**The canon data itself is sound. Every integrity property I could test —
label fidelity, POV, scale-as-labeled, FEN legality, text→binary encoding,
alignment, shuffle, duplication, distribution shape — is clean.** The
catastrophic results (−582/−564/−463/−400) are not caused by a broken label or
a broken encoding. They are caused by the **training-recipe delta** between
the canon text-corpus path and the proven sfbinpack streaming path: the canon
path silently changed FOUR training-relevant variables at once, and no
diagnostic run to date fixed more than one of them at a time — which is why
each individual fix "failed" and was wrongly ruled out in sequence.

The four deltas (each with direct evidence below):

| # | Variable | Healthy sfbinpack recipe | Every canon run | Status |
|---|----------|--------------------------|-----------------|--------|
| 1 | Target scale | binpack cp ÷ 1.9 (net-facing ≈ 2× UCI cp) | raw SF18 UCI cp ⇒ **0.51×** the expected scale (SCORE_MULT=1.0 runs) | proven, quantified; ×1.94 fixes it (exp-0111) but alone recovered only ~+120 |
| 2 | Quiet filter | 5 criteria at load: ply≥16, not-in-check, \|score\|≤10000, best-move Normal, best-move non-capture (**drops 26.9%** measured) | base runs: none; "clean" run: in-check only (9.17%) — **~22% best-move-capture positions remain** | proven present; **never actually fixed in any canon run** — exp-0123's "filter ruled out" is invalid |
| 3 | WDL blend | WDL=0.3 with real game results | WDL=0 forced (corpus result byte is a 0.5 placeholder) | unresolved confound; present in every catastrophic run, absent in every healthy one; never tested in isolation |
| 4 | Unique volume / epochs | ~14.4B-entry binpack streamed, <0.5 epoch (+75% random skip) | 99.7M uniques × **60 epochs** (240sb) | supported as contributor (memorization signature in the loss curve); exp-0115 vs 0116 puts this factor at roughly −143 vs −463 |

None of these is a property of the SF18 labels. "The canon data is broken" is
the wrong frame; "the canon *recipe* diverges from the proven recipe in four
uncontrolled ways" is the right one.

## What I verified as CLEAN (the data-integrity case)

### 1. Labels are exactly what they claim: SF18@50k UCI cp, white-relative, on the right FENs

Re-ran a fresh `engines/bin/stockfish18` at `go nodes 50000` on 401 positions
sampled uniformly across `corpus.txt` (every ~274k-th row) and compared to the
stored labels (rerun cp converted stm→white, clamped ±2499 like the build):

- slope(label on rerun) = **0.960**, sign agreement (|cp|>50) = **99.6%**
- |diff|: p50 = **10 cp**, p90 = 42, p99 = 115
- Only 3/401 rows differ >400 cp — all three are mate-horizon positions where
  a warm-hash 50k search and my cold 50k search resolve differently (e.g. one
  found the mate, the other returned −691). Genuine search variance, not label
  corruption.

This single probe simultaneously rules out: sign flip, POV flip, scale
corruption of the stored numbers, FEN↔score misalignment, and oracle
misconfiguration. r = 0.904 overall (dragged by the 3 tail rows).

### 2. POV chain is consistent end-to-end

- Corpus text is **white-relative** as documented: corr(cp, white-material) =
  **+0.704 for white-to-move rows and +0.698 for black-to-move rows** (55k-row
  strided sample; stm-relative labels would flip sign on the black rows).
  `obs-tool export-value` converts SF18's stm-relative UCI score with
  `value_cp_white` (src/bin/obs_tool.rs:1182), unit-tested.
- bulletformat 1.8.0's `FromStr` (the converter bf-convert delegates to)
  expects white-relative text and **negates score + flips result for
  black-to-move** while canonicalizing the board to STM perspective (crate
  source chess.rs:240-243) — matching what the trainer's stm-relative
  `sigmoid(score/400)` target requires.

### 3. Text→binary encoding is bit-faithful and aligned

- Row counts: `corpus.txt` 109,706,426 lines == `corpus.bin` 109,706,426
  records (3,510,605,632 B / 32); `corpus.quiet.txt` 99,671,803 ==
  `corpus.quiet.bin` records. Zero parse drops (the converter drops
  unparseable lines, which would desync — none did).
- Decoded and cross-checked **232k records at three offsets** (head of both
  corpora, records 50.0M–50.1M, final 100k): **0 mismatches** in score
  (incl. the stm negation), result byte, occupancy popcount, and both
  king-square fields (incl. the `^56` conventions).

### 4. Shuffle is real

Lag-1 autocorrelation of popcount(occ): raw `corpus.quiet.bin` **0.958**
(game-contiguous, as expected) → `corpus.quiet.shuf.bin` / `corpus.shuf.bin`
**0.000–0.003** at offsets 0 / 40M / 90M. (Bonus confirmation of correct STM
encoding: the *raw* file's score lag-1 autocorr is **−0.67** — consecutive
plies alternate side to move, so a correct stm-relative score alternates
sign.)

### 5. No duplication, legal FENs, sane distribution

- **0 duplicate** board-keys (board+stm+castling+ep) within each of three
  contiguous 2M-row slices (head/mid/tail) and 0 within the 55k strided
  sample. The corpus is thoroughly deduped; a memorize-the-duplicates story is
  out.
- 0 illegal FENs in 55k (python-chess `is_valid`).
- stm split 50.2/49.8; |cp| p50 = 92, p90 = 491; clamp mass at ±2499 = 0.84%;
  in-check 9.17% (raw) → **0** in quiet (canon_filter verified working on the
  criteria it implements).

### 6. The distribution is NOT anomalous vs the known-good binpack

Decoded 800 entries spread across `nodes5000pv2_UHO.binpack` (chunk-indexed
sampler) and compared:

| metric | canon-v1 | UHO binpack (raw) |
|---|---|---|
| piece count p10/p50/p90 | 6 / 14 / 26 | 6 / 13 / 25 |
| ply p50 (canon: 2×fullmove) | ~68 | 68 |
| in-check | 9.17% | 8.0% |
| early positions | 0.8% (fullmove≤8) | ~p10 at ply 26 |

The endgame-heavy, opening-free shape that looked pathological is **the same
shape the winning data has**. Also, the canon slice is *not* adversarially
hard for the incumbent: incumbent-vs-SF18(×1.98) disagreement is statistically
identical on canon positions (r 0.914, p50 76 cp, 6.0% >400 cp) and random UHO
positions (r 0.907, p50 75 cp, 6.4% >400 cp). The "disagreement-selected
subset is a pathological distribution" story is not supported — consistent
with fulldist (full UHO distribution) failing at −662 (exp-0114).

## What is actually broken (the recipe deltas, with evidence)

### Delta 1 — target scale: the pinned "canon cp == binpack_cp/1.9" claim is FALSE

Measured directly, positions from the UHO binpack re-searched by SF18@50k:
**binpack score = 2.76× SF18-UCI cp** (regression slope; median ratio 2.64,
IQR 2.24–3.10, n=798, r=0.94) — not 1.9×. Net-facing measurement on canon
positions: **median(label / incumbent-net eval) = 0.506** (n=401; the
incumbent's effective output scale ≈ 2× UCI cp — the extra sharpening beyond
2.76/1.9≈1.45 plausibly comes from the WDL=0.3 blend pushing decided-position
targets toward 0/1). So training with SCORE_MULT=1.0 teaches a net whose
output is half what the search's margins/prunes are co-adapted to — exactly
the 0.516× measured on the first canon net, and why SCORE_MULT=1.94 restored
0.988× (exp-0111). **Runs affected by this un-fixed:** canon-v1-base (−582)
and canon-clean (−400, exp-0123 — it kept SCORE_MULT=1.0). The build/train
script headers assert the false claim and should be corrected.

### Delta 2 — the quiet filter was never fully applied to any canon run

Measured on the 401-row probe (SF18 bestmove + python-chess classification):
**20.4% best-move-capture, 7.2% in-check, 0.2% promo ⇒ the sfbinpack
load-time filter would drop 26.9%** (matches the 26% fast_fens smoke number).
`canon_filter` implements only in-check + |cp|>10000 (the export text lacks
the move), so `corpus.quiet.*` still contains **~22% positions whose 50k-node
label prices a tactical resolution a static eval cannot represent** — the
exact mechanism root-caused on 2026-07-20 for fulldist (MSE tails → search
minimaxes into the warped region). **exp-0123 did not test the filter
hypothesis it claims to rule out**: it removed the *smaller* poison component
(in-check) while keeping the larger one (captures), *and* kept the scale bug.
The shards contain `SearchObservations.best_move`, so the full filter is
implementable at export time.

### Delta 3 — WDL=0 is an unresolved confound present in every catastrophic run

The corpus result byte is a constant 0.5 placeholder (no game-WDL join), which
*forces* WDL=0. Every healthy net (incumbent, repro −143) trained WDL=0.3 with
real results; every catastrophic net (canon ×4, fulldist ×2, fishpack-wdl0)
trained WDL=0. The one "isolation" (exp-0110, −848 on fishpack) is invalid —
fishpack is a scale-broken source (per exp-0036 and the 07-20 retraction).
WDL=0 has never been tested on clean data at correct scale. I could not test
it statically (it needs a training run); it must be closed by the experiment
below, not assumed either way.

### Delta 4 — 60-epoch replay of 100M uniques vs billions-streamed-once

240sb = 5.98B presentations = **60 epochs** over the 99.7M quiet corpus. The
healthy path streams a ~14.4B-entry binpack (38,546 chunks × ~370k entries)
with 75% random skip — under half an epoch. Loss curves: canon-clean fell
0.00844 → **0.000512** and was *still declining* at sb240 (memorization
signature), vs the healthy repro's 0.0213 → 0.00754. Caveat: the raw 15×
gap is **not** pure memorization evidence — WDL=0.3 targets carry an
irreducible game-result noise floor that WDL=0 targets lack, so the loss
scales aren't comparable. The honest statement: "low training loss" is partly
an artifact of the target change (delta 3) and partly multi-epoch fitting;
exp-0115/0116 (−143 streamed vs −463 materialized-137M, same budget, filter
applied) bounds the volume/epoch term as large.

### Why "low loss + terrible play" is not a paradox here

The net minimizes exactly the loss it is given, on the distribution it is
given. With deltas 1–2 the *minimum itself* is poisoned: a perfect fit outputs
half-scale evals (breaking every co-adapted search margin) and prices ~22% of
quiet-looking positions at their tactical resolution (warping the eval surface
precisely where search stands pat). With deltas 3–4 the loss number is
additionally incomparable (no result-noise floor) and inflated downward by
epoch-replay. Low loss is what a *faithfully learned wrong objective* looks
like — no data corruption required, and none exists.

## Prior beliefs: confirmed / refuted

- **Confirmed:** labels are white-relative SF18 UCI cp (exp-0036-class audit
  now done for canon itself); eval-scale bug real and ≈0.5× (quantified two
  independent ways); trainer/netpack/engine/gate exonerated (repro −143);
  in-check fraction 9.17%; quiet rebuild correctly removed it.
- **Refuted:** "canon cp == binpack/1.9" (it's ≈ binpack/2.7; net-facing
  ≈ ×0.51 — use SCORE_MULT≈1.94–2.0); "exp-0123 rules out the quiet filter"
  (it tested 2/5 criteria with the scale bug still present); "disagreement
  distribution is the definitive bug" (canon's phase profile and
  incumbent-disagreement profile are indistinguishable from UHO's; fulldist
  −662 already contradicted it); "the SF18-canon DATA is broken" (all nine
  integrity probes clean).
- **Left open (by design of the evidence, not laziness):** how the −400..−582
  residual splits between deltas 2, 3, 4 once scale is fixed. That split is
  not measurable statically; it is exactly what the experiment below measures.

## Confidence

- Data integrity (labels faithful, encoding correct, no dup/misalignment):
  **high** — direct, redundant, quantitative probes.
- Scale claim false / ×1.94 needed: **high** — two independent measurements
  agree with each other and with exp-0111's net-side result.
- "Recipe delta, not data" as the root-cause frame: **high** — every
  catastrophic run carries ≥2 un-fixed deltas; the one run with zero deltas
  (repro) is healthy on the same box/trainer/budget.
- Relative weight of filter vs WDL vs volume in the residual: **genuinely
  unknown** — deliberately deferred to the experiment.

## Minimal fix + smallest confirming experiment (GPU, when free)

**Fix (small, all at build/export layer):**
1. Extend `obs-tool export-value` to emit (or filter on) the shard's
   `best_move`: drop best-move-capture/promo and ply<16 alongside the existing
   in-check drop ⇒ full 5-criterion parity (~26% drop, ~81M rows).
2. Join real game WDL (the mined.tsv / fleet path already carries
   white-relative results) so the result byte is real ⇒ enables WDL=0.3.
3. Train with `SCORE_MULT=1.94` (and fix the false scale comment in
   `build_canon_corpus.sh` / `train_canon.sh`).

**Experiment (one 60sb A/B, ~1.5h GPU total + two 200-game gates):**
- **Cell A:** fully-filtered, WDL-joined canon corpus; SCORE_MULT=1.94,
  WDL=0.3, 60sb → `gate_canon.sh` vs `d40-uho-v1-240sb`.
- **Cell B (closes the WDL confound):** same corpus, WDL=0, all else equal.
- Predictions: diagnosis right ⇒ A lands repro-class (≈−150±40), i.e. the
  canon data trains a normal net once the recipe matches. A ≈ −400 ⇒ the
  residual is the volume/epoch term ⇒ stop materializing: use the canon corpus
  as `D40_MIX_DATA` fold-in aux (5–15%) on the streamed sfbinpack base (the
  mechanism already exists in the trainer). B collapses while A is healthy ⇒
  WDL=0 confirmed as a first-class killer; B ≈ A ⇒ WDL was innocent all along.
  Either way the four-way confound is finally cut.

## Probe inventory (all CPU-only, seconds–minutes, live GPU run untouched)

| Probe | Method | Result |
|---|---|---|
| Label fidelity | SF18@50k rerun, 401 stratified rows | slope 0.960, p50 diff 10cp, 99.6% sign |
| POV | corr(cp, white material) by stm, 55k rows | +0.704 / +0.698 (white-relative) |
| Encoding | struct-decode 232k records @ 3 offsets vs text | 0 mismatches |
| Alignment | wc -l vs filesize/32, both corpora | exact |
| Shuffle | lag-1 autocorr popcount/score, 3 offsets | 0.958 → 0.000 |
| Duplicates | sort -u on 3×2M slices + strided | 0 |
| Units ratio | 800 UHO binpack entries vs SF18@50k | binpack = 2.64–2.76 × UCI |
| Net-facing scale | 401 labels vs incumbent nnue-eval | 0.506 |
| Tactical fraction | SF18 bestmove classification | 20.4% capture; 26.9% filter-drop |
| Hardness bias | incumbent-vs-SF18 diff, canon vs UHO | identical (6.0% vs 6.4% >400cp) |
| Loss curves | /tmp/canon_clean.log, /tmp/repro_sfbinpack.log | 0.000512 (falling) vs 0.00754 |
