# ARCHON's eval weakness profile vs SF16 + SF18-label value (2026-07-20)

Two investigations run while the fulldist SF18 net trained. Both inform the
eval-improvement strategy beyond generic label depth.

## 1. SF18@50k labels are a CALIBRATION improvement, not a verdict change

On ~1.4M matched positions (canon SF18 cp vs the binpack 5k score / 1.9, both on
control scale):
- **Winner-agreement 97.5%** (same sign or both ~0), correlation **r=0.93**.
- SF18 is **~30% less extreme**: mean |cp| 147 (SF18) vs 209 (5k/1.9). Deeper
  search → less extreme, better-calibrated evals.
- median |diff| 27cp, p90 218cp; 16.6% differ by >150cp (a tail).

**Implication:** SF18 labels mostly fix MAGNITUDE/calibration on a minority of
positions, not the winner. → the fulldist net (only 26% SF18-relabeled) should be
a MODEST win over the control, not a big one (matches the epic's "label depth is a
small lever"). The real label lever is **upsampling the SF18 positions** to
concentrate the signal (currently diluted 4× at 26%).

## 2. ARCHON loses to SF16 by gradual POSITIONAL drift, mid→endgame, material-even

Analyzed the native equal-time −129 games (178 ARCHON losses; 37 with a clear
eval-collapse ≥−0.5 → ≤−1.5, via python-chess board replay):
- **Phase at collapse: 62% middlegame (14-23 pc), 27% endgame (<14), 11% opening.**
- **Median fullmove# at collapse = 39** (late middlegame into endgame).
- **Material at collapse: EVEN** (median 0, mean +0.1); **68% ARCHON is
  material-even-or-up** → outplayed positionally, not blundered/down material.
- King exposure moderate (2.76/12) — present, not the dominant factor.

**Implication:** ARCHON's eval deficit is **quiet positional judgment of
middlegame→endgame positions** — it drifts even while material-even. NOT tactics,
NOT material, NOT acute king safety. Consistent with the 86%-gradual-loss finding
(native-box-regression-2026-07-20.md), now localized.

## Strategy takeaway

Generic SF18 label depth is a small lever. The −129 gap lives in subtle positional
judgment of mid/endgame positions where ARCHON is slowly outplayed. The highest-
leverage eval work is therefore:
1. **Hard negatives (Phase D)** — the ARCHON-vs-SF18 loss positions
   (`data/nnue/hardneg/all.tsv`, 2M) ARE these positions; deep-SF18-label them +
   upsample into training to directly densify ARCHON's weak spots.
   **PROFILE CONFIRMED:** hardneg is 49% endgame / 28% mid / 22% open, 60%
   material-even — matches the loss profile. **PIPELINE VALIDATED (pilot 2026-07-20):**
   `root-analysis --engine engines/bin/stockfish18 --fens X --nodes 50000
   --reference-only --out Y.ceobs` works locally; **35 ms/pos** → full 2M ≈ 2-3h at
   8-10 parallel instances (shard the fens; leave ~6 cores for a running train's
   loader). Then `obs-tool export-value` → FEN|cp, join the WDL from all.tsv,
   upsample, fold into the fulldist corpus, retrain. NOT started (awaiting the
   fulldist gate + protecting the current GPU train).
   **★ ALREADY DONE (found 2026-07-20): the 2M deep-label is COMPLETE** —
   `data/nnue/hardneg/labeled/chunks/*.ceobs` (64 chunks, ~1.5GB, SF18@50k) from a
   prior session, and exported to `data/nnue/hardneg/labeled/hardneg.vals` (1.93M
   `FEN|cp|0.5`, mean|cp| 356, 5.4% decisive — sharp because they're ARCHON's real
   losing positions). REMAINING (all fast/local, after the fulldist gate): scale SF18
   cp ×1.94 (→control scale, same as fulldist), join real WDL from
   `data/nnue/hardneg/all.tsv` by FEN, UPSAMPLE ~2-4×, concatenate into the fulldist
   `corpus.txt` → bf-convert → shuffle → retrain (SCORE_MULT=1.0, WDL=0.3). So the
   Phase-D net is ~1 corpus-assembly step + 1 train away — the expensive labeling is
   banked.
   **★★ FRAGMENT BUILT 2026-07-20: `data/nnue/hardneg/labeled/hn_cp_wdl.txt` =
   1.93M `FEN | SF18cp×1.94(control-scale) | real_WDL`** (WDL 31/31/38 L/D/W). To
   build the Phase-D net: `for i in $(seq N); do cat hn_cp_wdl.txt; done >> a copy of
   the fulldist corpus.txt` (upsample N≈4-8), then bf-convert → shuffle → train
   NET_ID=fulldist-hardneg-240sb SCORE_MULT=1.0 WDL=0.3. Gate vs fulldist-sf18-240sb
   AND vs the control. Everything but the final assemble+train is done.
2. **SF18-position upsampling** in the canonical corpus (deferred opt #2).
3. King-bucket factorization #27 (zero-cost eval-quality) as an orthogonal add.
Endgame coverage is already good (canon-v1 = 37.5% endgame); the mid→endgame
positional gap is a data-density + calibration problem, which (1)+(2) target.
