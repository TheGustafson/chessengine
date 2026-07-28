# Session 2026-07-27c — net-live re-screens + four SF-parity structural toggles

Continuation of the scale/tuning investigation on a NET-LIVE binary
(`target/qtrace`, `tune,sota,arch-d47v`, `info string eval: NNUE(sota-v2)`
confirmed; no-op arm read exactly 0.00 twice, once per binary build).

## Instrument state

- Ledger quarantined: all 37 pre-fix rows were PSQT-regime (top1 ~0.44 vs
  ~0.66 net-live) → `screen-results-PSQT-INVALID-2026-07-27.jsonl`.
- Net-live full-corpus control baseline (9,992 positions, 50k nodes):
  **mean cp loss 5.31 / 4.38 / 4.57 by tier, top1 66.7%, blunder 0.1%,
  p90 16** — the numbers every later delta is measured against.
- Determinism spot check: control row reproduces exactly across runs and
  across a rebuild (4.84/4.18/4.23 at n=1600, twice).

## The PSQT-era candidates, re-screened net-live (n=1600, paired)

| arm | cp@4k/16k/64k | nodes | verdict |
|---|---|---|---|
| QUIET_LOW_PLY_WEIGHT=12 | +0.28/+0.25/+0.28 (±~0.25) | +4.2% | **dead** — worse on all tiers AND more nodes |
| NMP_EVAL_BASE=168 | +0.38/+0.23/+0.36 | −11.5% | quality loss all tiers; node saving priced below |
| RED_TABLE_NUM=2700 | +0.30/−0.14/+0.17 | +11.7% | mixed sign; full-corpus verdict pending |
| QS_FUTILITY_MARGIN=60 | +0.23/+0.01/+0.19 | +5.1% | no gain, more nodes — dead |

Coordinator's concurrent n=4000 draws (same seed, different subset) read the
opposite sign on red2700/nmp168/qsfut60 — a sampling difference, settled by
the full-corpus runs recorded in the ledger.

**Full-corpus verdict (n=9992): RED_TABLE_NUM=2700 is NULL** —
+0.10/−0.01/−0.02 (±0.10-0.11) at +11.7% nodes, i.e. strictly worse at a real
time control. Both exploratory reads (+0.17 mine, −0.45 coordinator's) were
sampling noise around ~0; the PSQT-era "+2.7 Elo net" claim is dead. A clean
demonstration of the regression-to-the-mean rule: n=1600 and n=4000 disagreed
in SIGN on a true effect of ~0.0 cp.

## Four SF-parity structural toggles (commit c11f26e, identity-off defaults)

Found by three-reviewer side-by-side reading vs `eca43a97` (D33). Screened
net-live at n=1600:

| toggle | cp@4k/16k/64k | top1 | nodes | verdict |
|---|---|---|---|---|
| FOLLOW_PV_MODE=1 | +0.04/−0.06/**−0.40±0.26** | **68.3% (+1.6)** | +12.2% | candidate — full corpus + budgets pending |
| QS_UPREP=1 | +0.47/+0.20/+0.23 | 66.5% | +0.6% | worse on all tiers — stays off |
| RED_TTPV_TTMISS_REBATE=1 | +0.44/+0.30/+0.34 | 66.4% | +5.3% | worse on all tiers — stays off |
| PROBCUT_IN_VERIF=1 | (pending) | | | |

The two rejections are informative: per-mechanism SF parity is NOT free Elo —
our tree/eval interact with each mechanism differently than SF's do, and both
parity restorations measurably hurt. Structure has to be screened like any
constant.

## INSTRUMENT CORRECTION (coordinator, mid-session): oracle saturation

The 64k-node oracle tier saturates: the control's cp-per-doubling collapses
from ~2.0 (3-12k budgets) to ~0.3 at 50k→100k, because a 50k-node arm has
only 0.36 doublings of headroom under its own judge. Consequences applied
retroactively here:

- 50k-budget deltas (all rows above) are COMPRESSED — magnitudes are lower
  bounds; significance is pessimistic.
- Tiers SHALLOWER than the arm's budget score "agreement with a weaker
  judge" and are ignored from now on (the all-three-tiers sign rule was
  biased toward weak arms).
- My fpv budget scan (25k/100k/200k vs the 64k judge) is INVALID above 64k;
  its 200k "+0.24" flip cannot carry a rejection. The earlier in-session
  "budget scan fails fpv" call is WITHDRAWN — fpv's verdict rests on the
  full-corpus 50k row and, if that holds, an SPRT.
- Pricing rule (order of magnitude): ~2 cp/doubling where the judge can see
  and ~120 Elo/doubling ⇒ 1 cp ≈ 60 Elo. fpv: −0.40 cp ≈ +24 Elo vs
  +12.2% nodes ≈ −20 Elo — marginal either way; only games can settle it.

## Pre-registered decision rule for FOLLOW_PV_MODE (written before the data)

Promote `fpv` to SPRT only if ALL of:
1. full-corpus (n=9992) cp@64k delta ≤ −0.15 (better) and |delta| ≥ 1 paired sem;
2. the budget scan (25k/100k/200k) does NOT show the gain collapsing toward
   zero or flipping at 200k (tuned-for-shallow trap);
3. the node cost stays ≤ ~15% (at ~120 Elo/doubling, +12% nodes ≈ −20 Elo to
   overcome; a deep-tier cp gain that prices below that is not worth a SPRT).

SPRT if promoted: same binary both sides, `option.FOLLOW_PV_MODE=1` vs
default, 10+0.1, Hash=16, Threads=1, UHO book, [0, 5] bounds — after the
loss-mining match releases the timed slot.

## FINAL VERDICTS (full corpus, n=9992, 50k-node arms, 64k tier only)

| arm | cp@64k | nodes | net price (~60 Elo/cp, 120 Elo/dbl) | verdict |
|---|---|---|---|---|
| RED_TABLE_NUM=2700 | −0.02±0.11 | +11.7% | ≈ −20 Elo | **dead** (PSQT-era "+2.7 Elo" was noise) |
| NMP_EVAL_BASE=168 | −0.06±0.11 | **−11.5%** | ≈ **+14±7 Elo** | **SPRT running** (fixed 300k nodes/move, [0,5]) |
| FOLLOW_PV_MODE=1 | −0.09±0.10 | +12.2% | ≈ −15 Elo | **not promoted** — n=1600's −0.40 shrank 4× at full corpus; fails pre-registered rule 1. Wiring stays (identity default) for a future joint retune: SF runs this mechanism inside constants co-tuned with it. |
| QUIET_LOW_PLY_WEIGHT=12 | +0.28±0.26 (n=1600) | +4.2% | worse both axes | dead |
| QS_FUTILITY_MARGIN=60 | +0.19±0.26 (n=1600) | +5.1% | worse both axes | dead |
| QS_UPREP=1 | +0.23±0.27 (n=1600) | +0.6% | — | parity rejected |
| RED_TTPV_TTMISS_REBATE=1 | +0.34±0.26 (n=1600) | +5.3% | — | parity rejected |
| PROBCUT_IN_VERIF=1 | ±0.00 exactly | +0.0% | — | de facto no-op in BOTH engines (stored-value clause blocks it); toggle deleted |

Caveats that stand over the whole table: the 64k oracle saturates a 50k arm
(0.36 doublings of headroom) — all magnitudes are compressed lower bounds; and
nmp168's shallower-tier gradient (+0.34@4k, +0.11@16k) is consistent with a
real-but-compressed quality cost, which is exactly what the running SPRT
integrates properly.

The SPRT: `engines/runs/lossmine-0727/sprt-nmp168.log`, frozen binary
`engines/bin/archon-sprt-c4350d3` (= commit c4350d3 qtrace build), fixed
300k nodes/move so it is load-immune and needs no timed slot, elo0=0 elo1=5.

## Also this session

- 17 more implicit-coefficient tunables at identity (commit 2e84cbe):
  qsearch return blends (6), NMP verify fraction (2), singular negative
  extensions (2), LMR entry gate (2), LMRDEPTH main-history 64/32 split (2),
  RED_TABLE_OFFSET, ASPIRATION_FAILHIGH_DEPTH_MULT, FH_SOFTEN_BETA_W,
  SIBLING_DEPTH_STEP. Identity proven: pinned bench 445174 bit-identical +
  net-live 800-fen d10 bench 8080707 bit-identical.
- FRAME_S documented as a DISCONNECTED knob (zero call sites; commit 6c7da4d)
  so no SPSA campaign sweeps a guaranteed null.
- Loss-mining match vs SF18 running (10+0.1, UHO): ~30% score, taxonomy so
  far ~51% tactical / 49% grind, median disagreement at the losing move
  −4.7 pawns. Corpus + tiered oracle build to follow.

## Deferred (found, not yet acted on)

- Movepick/history tunable surface (cont-offset weight arrays, history init
  and decay constants, evasion weights, low-ply divisor) — arrays need
  scalar-ization; coordinator's audit says geometry is faithful, so these are
  pure surface, low prior.
- `nodestime` elapsed uses only the local worker's counter (real bug, but no
  harness uses nodestime).
- MultiPV>1: aborted-line PV repair quirk; searched-list joint capacity 32
  (SF splits 32/32).
- TM frame-sensitive couplings (falling-eval coefficients consume our-frame
  cp deltas against SF-frame-tuned thresholds) — only visible in timed games,
  not in fixed-node screens.
