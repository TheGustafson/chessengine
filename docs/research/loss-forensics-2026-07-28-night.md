# Loss forensics, night of 2026-07-28 — the population defect

Companion to `LOSS-FORENSICS-PLAYBOOK.md` (method) and
`TRANSITION-2026-07-28.md` (prior state). Read the two REFUTED banners in the
playbook first: `bbc28fd` and `b22c9c9` retract the gate-attribution mode.

Everything below was measured tonight. Where a number contradicts an earlier
one, the earlier one is named and withdrawn.

---

## 1. THE HEADLINE — our tuning instrument samples the wrong population

`tools/nnue/calib/oracle-obs.json` is the corpus behind **every tunable screen
this project has ever run**. It is 9,992 positions, **100% of them 21+ pieces**,
and the FENs sit around move 6–8. Against the errors we actually make in games
we lost:

| pieces | share of errors | share of WIN PROBABILITY lost | share of oracle |
|---|---|---|---|
| ≤ 7 | 66.7% | 36.8% | **0%** |
| 8–12 | 18.3% | 8.4% | **0%** |
| 13–20 | 11.1% | 36.9% | **0%** |
| 21+ | 4.0% | 17.9% | **100%** |

And by move number: **zero errors before move 21**; 87% of win-probability loss
after move 33; 64% after move 46.

So the screen measures the phase that holds ~18% of the damage and is blind to
the other 82% — and the phase where we demonstrably make *no* mistakes at all.

The count column overstates the endgame share (a pawn-threshold detector
inflates sparse positions — a trap already on the refuted list), which is why
the decision rests on the win-probability column. It survives the correction.

**This is not merely low power — the screen can be BLIND to a large real
effect**, and the case is on file: `HINDSIGHT_EVAL_SUM=111` came back
**+0.03 ± 0.15 cp** at the full 9,992 positions — a flat nothing — and then
lost **17.4 Elo** in games. (Precision matters here: that is blindness, not
inversion. The screen's honest verdict was "no effect" on a change worth −17
Elo. The playbook records the separate reason it was ever believed — a 4000-
position run read −0.32 ± 0.23 and 1.4σ was mistaken for a find.) A corpus that
cannot see a 17-Elo change is not a weak instrument, it is the wrong
instrument. It also reframes every dry campaign —
139 of 303 tunables were screened, all of them on openings, and the other 164
were never screened at all.

**Fixed:** `tools/forensics/build_oracle.py` samples positions from our own
games uniformly across each game, so phases appear in the proportion we play
them, and scores every legal move at the same 4k/16k/64k tiers, so
`screen.py --oracle` takes it directly. Positions come from **all** games, not
only losses: conditioning the population on the outcome would bias every future
screen toward whatever we happened to be doing while losing.

---

## 2. THE CLOCK — real, quantified, and NOT the binding constraint

Per-game clock reconstruction over the 120-game rematch (10+0.1):

| move | ARCHON left | SF18 left | diff |
|---|---|---|---|
| 10 | 8.65 | 8.63 | +0.02 |
| 25 | 4.30 | 5.01 | −0.71 |
| 33 | 2.38 | 3.67 | −1.29 |
| 40 | 1.28 | 2.76 | −1.48 |
| 45 | 0.80 | 2.30 | −1.50 |

47.3% of our moves are played with under a second left, against their 25.2%.
Spend as a fraction of the clock in hand runs 1.6× (moves 21–30), 2.35×
(31–40), 2.67× (41–50) of theirs, and by move 41 our *absolute* seconds fall
below theirs — 0.163 s to their 0.184 s. We arrive at move 45 with 35% of
Stockfish's clock, and search 1.5–2.5 plies shallower from move 41 on.

**But this does not explain the blunders, and I initially said it did.** Two
things refute the causal story:

- At moves 41–50 Stockfish gains **1.5 plies from 13% more time**. No branching
  factor converts 13% into 1.5 plies. Most of that gap is not the clock.
- On game 18 Stockfish plays the right move at **~25k nodes**; we spent
  0.049 s ≈ 50–70k nodes in the game. We had a budget at which the reference is
  already correct. More time would not have saved it.

The clock deficit is real and worth fixing on its own terms. It is not the
mechanism behind the errors.

---

## 3. TREE SHAPE — measured at equal nodes, the gap is small

Depth reached at a fixed 200k nodes, Hash 16 (as in the match):

| pieces | ARCHON | SF18 | gap |
|---|---|---|---|
| 21+ | 16.37 | 16.23 | **+0.13** |
| 13–20 | 17.30 | 17.37 | −0.07 |
| 8–12 | 16.90 | 17.50 | **−0.60** |
| ≤7 | 19.53 | 19.97 | −0.43 |

We are *deeper* than Stockfish in the opening at equal nodes and 0.4–0.6 plies
shallower in endgames. That accounts for roughly a third of the in-game
endgame depth gap. The remainder is NPS and clock.

---

## 4. WHAT SURVIVES — the search-efficiency gap

Like-for-like MultiPV-1 ladders on game 18: **Stockfish commits to `a5c5` at
depth 5 / 618 nodes and never leaves it. We hold `h5e2` from depth 6 to 20 and
switch at depth 21 / 410,613 nodes.** The forensics agent's node-budget
measurement puts it at 25k vs 600–800k. Either way the deficit is one to two
orders of magnitude in nodes, on a position where:

- our own **static eval already prefers the right move**, by 108 cp in our
  frame against Stockfish's 31 — the net is not the problem here;
- no pruning gate is responsible (§5);
- the clock is not responsible (§2).

That leaves move ordering, extensions, and near-equal-move resolution. The
agent measured first-move cutoff at **89.4% against Stockfish's 93–95%**. That
is the surviving hypothesis and the next target.

---

## 4b. GAME 18, closed out — it is null-window root promotion, not knowledge

Two measurements finished this position off, and both went against the story I
was building.

**The eval is not merely adequate here, it is better than Stockfish's.**
Ranking all 20 of Black's replies to Be2 against deep truth (SF @400k):

| | Spearman vs deep truth | its value for the refutation `d7b5` |
|---|---|---|
| **ARCHON static** | **0.618** | **+45** (correctly positive for Black) |
| SF18 static | 0.179 | −71 (wrong sign) |

Both statics pick `c3c2` as Black's best; the truth is `d7b5`. Neither net
understands the deflection — but ours ranks the field three times better and
gets the key move's sign right. **No eval fix is available here, because our
eval is already the better of the two.**

**And Stockfish does not "know" the answer either.** Reading both candidates
out of ONE tree at MultiPV 3:

| depth | ARCHON a5c5 / h5e2 | SF18 a5c5 / h5e2 |
|---|---|---|
| 6 | −1 / **+63** | −12 / −25 |
| 10 | −32 / **+1** | −6 / **+3** |
| 12 | −25 / **−1** | −5 / **+3** |

**Stockfish's own full-window values prefer `h5e2` from depth 10 on** — the
losing move — while its single-PV search picks `a5c5` at depth 5 and never
leaves it, at every node budget from 25k up. So SF's correctness here is not
superior evaluation and not deeper resolution. In single-PV the second root
move is scouted with a null window around alpha and must EXCEED it to be
promoted; SF's scout proves `h5e2 <= alpha` and ours promotes it.

The localization is therefore precise and narrow: **root-move promotion under
the null window.** Not eval, not the gates (§5), not the clock (§2), not
movepick structure — our stage list is an exact match to SF's at `eca43a97`,
and neither engine has killers, so D17 stands.

This is a localization, not a fix, and one position cannot support a change.
It goes on the list for corpus-v2 to confirm or kill.

## 5. GATES — refuted, including my own attribution

`gate_depth.py` on game 18 with defaults read out of the binary said parent
futility cost 12 plies of discovery and 27× the nodes, where RFP-off made the
position *worse*. That overturned the standing "RFP is the only gate that flips
game 18" claim. Then a null panel overturned mine:

| arm | 25k | 50k | 100k | 200k | 400k | 800k |
|---|---|---|---|---|---|---|
| control | h5e2 | h5e2 | h5e2 | h5e2 | a5c5 | a5c5 |
| pfut-off | a5c5 | a5c5 | a5c5 | a5c5 | a5c5 | a5c5 |
| **null `PFUT_BASE` 40→41** | h5e2 | **a5c5** | **a5c5** | h5e2 | h5e2 | a5c5 |
| null `RAZOR_BASE` 465→464 | h5e2 | h5e2 | h5e2 | a5c5 | a5c5 | a5c5 |

A **±1 change to an unrelated constant** also "fixes" the position. The argmax
here is chaotic (the agent measured a 7.9–14.9% null flip rate against 30.9%
for gates-off), and its aggregate node-matched control **inverts** the whole
gates-off effect: +24.1 ± 6.3 depth-matched, **−25.7 ± 5.1 node-matched**,
monotone in the node ratio. Pruning is doing its job.

`pfut-off` is still the only arm correct at every budget, and no null matches
that. It is a curiosity, not a finding: **one position cannot support a
mechanism claim on a chaotic argmax.**

---

## 5b. THE ONE ELO LEAD — parent futility prunes too much, and my sign was wrong

Full corpus (n=9,992, openings — see §1 for why that matters), arms at 12,500
nodes:

| arm | what it does to the margin | cp@16k | nodes | blunder rate |
|---|---|---|---|---|
| control | — | 6.32 | — | 0.2% |
| `pf64c` constants ×0.64 | prune MORE | **+0.19 ± 0.15** | −1.8% | 0.3% |
| `pf64all` all four ×0.64 | mixed | −0.24 ± 0.15 | −0.8% | 0.2% |
| `pfutbase120` base ×3 (agent) | prune less | −0.38 ± 0.14 | +3.0% | — |
| `pfmult64` ply coeff ×0.64 | prune less when `lmr_depth<0` | −0.44 ± 0.15 | +4.4% | 0.2% |
| **`pfinv` constants ×1.56** | **prune less** | **−0.58 ± 0.14** | +4.4% | **0.1%** |

Five arms, four of them independently derived, and they order themselves
perfectly by how much they RELAX the bound. The one arm that tightens it
(`pf64c`) is the only one that hurts, and it is also the only one that fattens
the tail (0.3%). The best arm halves the blunder rate, 0.2% → 0.1%, which is
the property the screen's own trailer says to prefer: the mean is mostly tail,
and a blunder converts a game where a 2 cp inaccuracy does not.

**`pfinv` was built as a falsification control for my own hypothesis, and it
won.** I derived from the frame that our margins were too LARGE and should be
scaled by 0.64; the data says they are too SMALL. The dimensional argument
picked the right constants to move and the wrong direction — worth recording
precisely, because the same rule shipped the aspiration fix and will be reached
for again. It locates candidates; it does not predict signs.

At the screen's own conversion (~60 Elo per cp, and `log2(1+node_cost)·120`
against) `pfinv` is nominally ~35 − 7.5 ≈ **+27 Elo**. Do not believe that
number yet:

1. It is measured on the openings corpus (§1), which is the wrong population
   and is the very instrument this document exists to indict.
2. Relaxation cannot be monotone — gates-off is worse once node-matched — so
   the optimum is bracketed but not located. `c1.56 / c2.5 / c4 / c6 / off` is
   running.
3. `hind111` is the standing reminder that a screen number is not Elo.

Sequence before it can be promoted: bracket → re-screen on the phase corpus →
SPRT. **PFUT_\* are among the 164 of 303 tunables in no campaign at all**,
which is the most likely reason a 4σ effect was still lying around.

## 6. INSTRUMENT BUGS FOUND TONIGHT (five, all mine)

1. **Bound lines read as scores.** `uciprobe` matched `score cp X upperbound`
   as an exact value. Harmless for a depth ladder; fatal for a node-limited
   search, whose last iteration is cut off mid-window. A position worth −228
   read as −40.
2. **Cold vs warm TT.** `harvest.py` analyses unrestricted-then-forced back to
   back, so the forced search inherits a warm TT: 212 cp warm, 0 cold, same
   budget. Its cp values are order-dependent. This produced a false retraction
   ("the corpus is blind to game 18") which is itself now retracted.
3. **A hand-typed control.** Three of eight gate "defaults" were invented
   (`NMP_R_BASE` 4658 for 7, `RED_TABLE_NUM` 1155 for 2834, `LMP_BASE` 3387 for
   3). Every delta in that table was meaningless while looking quantitative.
   Defaults now come from `<engine> tunables`. **A control you typed is not a
   control.**
4. **MultiPV confound.** Stockfish's ladder was run at MultiPV 6 and compared
   against our MultiPV 1. MultiPV gives every root move a full window and
   surfaces refutations plies earlier. That manufactured most of an "11-ply
   gap".
5. **Cumulating per-move means.** Averaging time-spent at move N across games
   of different lengths and summing gives a *negative* time-left column. Clocks
   must be simulated per game.

The pattern in 1, 2 and 3: each instrument's sensitivity or baseline was set by
convenience rather than checked against a known value. `harvest2.py` now
refuses to run until it re-detects game 18.

---

## 7. STATE

| thing | state |
|---|---|
| tm10c (frame-scaled `FALLING_EVAL_*`, 10+0.1) | arm 51.6% at 321 games = **+10.8 ± 13.5 Elo**. Not significant. Arm holds *less* clock and searches shallower — it does **not** validate the §2 story |
| `pf64all` (all four PFUT terms ×0.64) | **−0.24 ± 0.15 cp @16k** on the *openings* corpus, −0.8% nodes. Needs re-screening on the phase corpus |
| `pf64c` (constants only) | +0.19 ± 0.15 — worse. The four terms want scaling together or not at all |
| phase oracle | building, 3,350 positions, 12.5/25/25/37.5% across the four buckets |
| corpus-v2 | 700 games vs SF18, chained behind tm10c |

**Blocking constraint (the agent's finding, and it is right):** of 19
adjudicated candidates in the 120-game match, only **two** had an oracle stable
across 400k/1.6M/8M nodes. One match does not contain three clean cases of any
mode. More games is the prerequisite for the three-independent-losses rule, not
more analysis of these.
