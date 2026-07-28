# Search-tuning audit — 2026-07-27

One day of reasoned ablation on the search parameters. Recorded as a ledger of
claims and verdicts rather than a narrative, because the useful output is
which hypotheses are dead and why.

## The ledger

| claim | verdict | killed/supported by |
|---|---|---|
| eval-frame global rescale pays | **REFUTED** | 10/10 tactical arms worse; the algebraic dual also lost |
| eval-frame per-family pays | **REFUTED** | PRUNE +3.5±31.7, HISTMG +9.3±17.7 |
| class 4 (material-vs-eval) alone pays | **REFUTED** | wired at identity and swept: neutral-to-worse |
| our constants have drifted from Stockfish | **REFUTED** | all 260 screened vs `eca43a97`; 2 hits, both known ARCHON originals |
| a constant-bearing mechanism is missing | **REFUTED** | 2 absences, both deliberate (Syzygy, skill level) |
| razoring is net-negative | **REFUTED** | full corpus −0.13/+0.07/+0.09 at **+0.1%** nodes |
| razoring is good, we have the wrong values | **REFUTED** | same measurement — it is inert either way |
| `IIR_MIN_DEPTH 8` helps | **REFUTED** | full corpus +0.20/+0.05/+0.29 (was −1.50 on biased samples) |
| `RFP ×1.25` helps | **REFUTED** | full corpus −0.10/−0.07/+0.06 |
| correction-adaptive LMR beats uniform loosening | **REFUTED** | at matched node cost, identical |
| the curated combo is worth +32 tactics | **REFUTED** | SPRT 211 games, 45.75%, −29.6±31.9 |
| gate-class predicts direction | **UNPROVEN** | 6/7, but scored on biased samples; re-running |
| `RED_TABLE_NUM 2700` helps | **MARGINAL** | full corpus −0.29±0.25 → net **+2.7 Elo** after node cost |
| `noRED+IIR8` is cheaper at equal quality | **SUPPORTED** | full corpus −0.25±0.25 **and −14.6% nodes** |
| contribution ≠ leverage | **SUPPORTED** | LMP shapes 23–47% of the tree with one near-inert parameter |
| 1 cp ≈ 79 Elo | **SUPPORTED** | −1.52 cp per node doubling × 120 Elo per doubling |

## What the refutations say together

Every hypothesis that our search **parameters** are wrong has died. The
constants are faithful transcriptions of a recent Stockfish master, their
effective scale is fine, and they sit close to a joint optimum: nothing moved
more than ~0.3 cp at full power, and the one arm with a real quality gain
(`RED2700`, +22.9 Elo) repays almost all of it in nodes (−20.2) for a net
**+2.7 Elo**.

The only thing that reliably works is finding configurations that are
**cheaper at equal quality**. `noRED+IIR8` is −14.6% nodes at −0.25 cp, which
is ~+27 Elo of speed plus ~+20 of quality — and the dominant term is a
deterministic node count rather than a noisy estimate.

That independently re-derives the gap decomposition this program started
from: the −78 Elo against SF16 splits ~−33 eval quality and ~−45 speed
(`where-the-sf16-gap-actually-is-2026-07-27.md`). **A day of careful
parameter work found no third bucket.** The levers remain the net (D48X) and
instruction count, not tuning.

## What was built, and why it outlasts the findings

- **17 implicit-×1 tunables** — terms carrying an invisible coefficient of 1,
  structurally invisible to design/01 §3.8's "bare numeric literal" rule.
  Includes the whole TT replacement policy, which had no tunables at all.
- **Two novel search mechanisms**, off by default: `RAZOR_CORR_DIV` and
  `NMP_CORR_DIV`, eval-uncertainty coupling for the two node-level gates that
  lacked it.
- **`tools/screen.py` + `docs/research/SCREENING.md`** — the protocol, with
  an SF18 oracle over 9,992 positions scoring every legal move at three node
  tiers, paired standard errors, a power table, budget scaling, tail
  statistics, and seeded random subsampling.

## The methodological result, which is the most transferable thing here

**Three instrument errors surfaced in one day**, each producing confident,
consistent, plausible numbers:

1. a **censoring** MultiPV oracle that floored losses exactly where a config
   behaved worst;
2. **unpaired** standard errors, hiding that 1–3 cp deltas at n=300 were
   noise;
3. **prefix subsampling**, which over-weighted hard positions — *bias*, not
   noise, so re-running could never have revealed it.

Not one was caught by reasoning harder about the measurement. Every one was
caught by **measuring the same claim a second way** — a second oracle, a
paired statistic, a matched-cost control, a full-corpus re-run. The screening
protocol now encodes each of those as a rule.
