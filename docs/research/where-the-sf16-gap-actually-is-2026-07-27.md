# Where the SF16 gap actually is — a decomposition that closes

Date: 2026-07-27. No new measurements: this assembles numbers already in
`perf-study-7-sf-side-by-side-2026-07-27.md`, `docs/strength-log.md` and the
value-foundation log into one arithmetic that turns out to close exactly. It
exists because the standing framing — "we lag behind in search fit" — is not
what the evidence says, and the difference changes where effort should go.

## The inputs (all previously measured, none re-derived here)

| quantity | ARCHON | SF16 | SF18 | source |
|---|---:|---:|---:|---|
| nodes to reach depth 14 (wl50) | **3.07 M** | 4.71 M | 3.20 M | perf study 7 §1.2 |
| effective branching factor | **1.40** | — | 1.51 | perf study 7 §1.2 |
| cycles / node (matched 15 M-node bench) | 4,588 | **2,302** | 3,401 | perf study 7 §1 |
| tactical score at EQUAL 50k nodes | 2,356 | 2,375 | 2,327 | strength-log 2026-07-27 |
| match Elo at fixed 50k nodes | — | **−33** | −133 | exp-0147, exp-0149..53 |
| Elo per speed doubling | 120 | | | exp-0154 |

## The arithmetic

Time to reach a given depth is `nodes-to-depth × cycles-per-node`. Our tree is
**smaller** than both engines', which partly cancels our slower nodes:

```
ARCHON  3.07M × 4,588 = 1.41e10
SF16    4.71M × 2,302 = 1.08e10   ->  SF16 is 1.30x faster to depth 14
SF18    3.20M × 3,401 = 1.09e10   ->  SF18 is 1.29x faster to depth 14
```

At 120 Elo/doubling, 1.30× is worth **45 Elo**. So:

```
vs SF16:  eval quality at equal nodes  -33
          speed (1.30x effective)      -45
          predicted equal-time         -78
          MEASURED equal-time anchor   -78 +/- 18   <- closes
```

The same model gives −178 vs SF18 against a −191±23 anchor measured on an
older net — consistent, though not the clean hit SF16 is.

## What this says

**1. The search is not the deficit.** Our tree is smaller than SF16's by 35%
and smaller than SF18's, our EBF is better (1.40 vs 1.51), and at equal nodes
we are tactically **level with SF16 and ahead of SF18**. Pruning, ordering and
extensions are not where the loss is. This is consistent with the separate
finding that design/10's three remaining "DEFERRED" search items were stale —
the N0–N14 program is materially complete.

**2. "We are 2× slower than SF16" overstates the speed problem by a lot.**
Per node we are 1.99× slower. But we need 35% fewer nodes, so the *effective*
deficit to equal depth is only **1.30×**. Search efficiency has already clawed
back most of the raw throughput gap. Quoting the per-node ratio as the speed
problem double-counts a deficit the tree already pays back.

**3. The gap is 42% eval, 58% speed.** −33 Elo of it is evaluation quality at
equal nodes, and since tactics are level at equal nodes, that −33 lives in
**quiet/positional** judgement — exactly what the D48X training run and the
quiet-stratum panels target. The other −45 is throughput, addressable in
instructions per node.

**4. It also prices speed work.** At 120 Elo/doubling, **1% fewer instructions
is worth ~1.7 Elo**. Today's landed work — perf-impl-1's −5.65% instructions
plus the loader bulk-read — is therefore worth roughly +10 Elo on its own
terms, and the remaining cheap stack in perf study 7 §7 (items 5–7, ~+3–4.5%)
is another ~+6.

## Caveats, stated rather than buried

- Nodes-to-depth-14 is a 50-position measure on one FEN set; a different mix
  would move it.
- Cycles/node came off a contended box and is quoted by the study as a band;
  the instruction ratio (deterministic) is the firmer number and would give a
  slightly different speed factor.
- 120 Elo/doubling is itself one experiment (exp-0154), not a law.
- The exact agreement at −78 is more precise than the inputs deserve. Three
  quantities each carrying ±10% cannot really produce a number good to 1 Elo;
  treat the *decomposition* as sound and the exactness as partly luck.
- Tactical suites are not positional strength. "Level with SF16 tactically"
  does not mean level in play — the −33 says we are not.

## What follows

Ranked by the arithmetic above, not by appetite:

1. **Quiet/positional eval** owns −33 Elo. That is the D48X run and its
   successors; it needs no new argument.
2. **Instructions per node** owns −45 Elo, and is the only lever with a known
   conversion rate (~1.7 Elo per 1%). Perf study 7 §7 items 5–7 are days of
   work for ~+6 Elo; the PGO training corpus (two positions for the whole
   NNUE path) is untested and cheap to test.
3. **Search parameter tuning** (the SPSA campaigns) is worth running because
   it is nearly free in engineer-time, but it should not be described as
   closing the SF gap. On this evidence its headroom is small.
4. **New search features** are the weakest lever available. We are already
   ahead on tree quality; adding mechanics fights for the smallest share.
