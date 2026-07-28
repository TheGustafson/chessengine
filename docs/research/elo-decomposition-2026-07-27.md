# Where the 191 Elo to SF18 actually lives — and a pre-registered prediction

2026-07-27. Written **before** the equal-time / equal-nodes rematch runs, so
the prediction below is falsifiable rather than fitted after the fact.

## The inputs (both already measured, neither by me)

External anchors, 100 games, 10+0.1 (`CLAUDE.md`):

| opponent | ARCHON |
|---|---|
| SF16 | −78 ± 18 |
| SF18 | −191 ± 23 |
| Integral v6 | −70 ± 22 |

NPS, same-round pinned matched bench (`perf-study-7`, §1.1):

| engine | NPS | vs ARCHON |
|---|---:|---:|
| ARCHON | 1.125 M | — |
| SF18 | 1.558 M | 1.33–1.38× |
| SF16 | 2.264 M | 2.01× |

The SF18 deficit grows with depth to ~1.44–1.48× on the movetime instrument,
so 1.33× is the floor and ~1.45× the working figure.

## The decomposition

A node-count ratio converts to Elo through the speed-doubling constant, which
this project measured for itself: **~120 Elo per doubling** (exp-0154, the
1.51× ≈ +73 Elo calibration).

```
SF18:  log2(1.40) = 0.485 doublings  ->  ~58 Elo of pure speed handicap
SF16:  log2(2.01) = 1.007 doublings  ->  ~121 Elo of pure speed handicap
```

Subtract the speed handicap from the observed gap to get the per-node
decision-quality difference:

| opponent | observed | speed | **per-node quality** |
|---|---:|---:|---:|
| SF18 | −191 | −58 | **−133** |
| SF16 | −78 | −121 | **+43** |

Two things fall out, and the second is the one that matters.

**We are already better than SF16 per node, by ~43 Elo.** We overcome a full
doubling of speed deficit and still finish only 78 Elo behind. That is the net
doing its job, and it is a real achievement rather than a rounding artifact.

**Against SF18, speed is roughly 30% of the gap and decision quality is 70%.**
The implied claim — SF18 is ~176 Elo better per node than SF16 — is consistent
with the outside world, where SF18 is well ahead of SF16 at equal time *and*
slower per node, so its equal-node margin must be larger still.

## What this means for tunables

Blunt version, and it is the honest one: **search-parameter tuning has a
bounded ceiling here, and the bound is not large.**

- Closing the *entire* NPS gap to SF18 buys ~58 Elo, and no realistic set of
  parameter changes closes all of it. Node-saving tunables are worth tens of
  Elo, not hundreds.
- The other ~133 Elo is per-node decision quality. Our search is a verified
  faithful transcription of `eca43a97` — stage machine, ordering terms, history
  geometry, and 42 eval-denominated constants all check out — so that 133 is
  mostly **eval**, i.e. the net.

Which is exactly what the project's own history says: net-0 was +696 over
PSQT, the width ladder and the data-volume finding moved hundreds, and D48X is
training right now. The tunables are worth grinding — tens of Elo are still
Elo, and they are what can move today — but they should not be mistaken for
the main lever, and no tuning campaign should be justified by the 191.

It also re-ranks my own queue. Anything whose entire benefit is fewer nodes is
competing for a ~58 Elo pool that we can only partly capture. Anything that
improves the *choice* at a given node competes for a ~133 Elo pool. Prefer the
latter when both are available at similar cost.

## The pre-registered prediction

The rematch plays the same match twice — equal time and equal nodes. Equal
nodes removes the speed term by fiat, so:

> **Predicted equal-nodes result: ARCHON −133 Elo vs SF18, with a working
> range of −115 to −150** (the range reflects 1.33× vs 1.48× on the NPS
> deficit and the ±23 on the anchor).

Falsification, stated in advance:

- **Better than −100** — speed was a bigger share than study 7's NPS implies,
  or the equal-node protocol flatters us (e.g. our node accounting differs from
  Stockfish's, which would be an instrument bug worth finding). Either way the
  node-saving branch is worth more than this note claims.
- **Worse than −170** — speed contributes almost nothing, the ~120 Elo per
  doubling constant does not transfer to this comparison, and NPS work is close
  to worthless for Elo. Retire it from the queue.
- **Anywhere in −115 to −150** — the decomposition holds, and the eval is
  confirmed as the dominant term.

One caveat I want on the record before the number arrives: equal-nodes matches
are not a clean instrument for a speed decomposition, because the two engines
count nodes differently and neither counts the *work per node* that the other
does. Treat a result inside the band as consistent with the decomposition, not
as proof of it, and treat a result outside the band as a reason to check the
instrument before rewriting the theory.
