# 08 — Classical evaluation

Search propagates leaf values; evaluation *produces* them. Every position where
the search stops (after quiescence, file 06) gets handed to `evaluate`, which
returns one number — how good is this for the side to move? — and the entire tree
is built on those numbers. This file opens the classical, hand-built kind of
evaluation: material, piece placement, and game phase, combined by tuned tables
into a single score. ARCHON ships this as its baseline (the "PSQT" evaluator,
worth roughly 2720 Elo on its own), and understanding it is the necessary setup
for files 11–12, where a neural network replaces it. We also confront the ceiling
that hand-built evaluation hits — the reason the whole field moved to learned
evaluation. Ground truth: `src/eval/psqt.rs` and `src/eval/mod.rs`.

## What evaluation must produce

Recall the contract from files 01 and 05. `evaluate` returns a `Value` (an
integer) that is:

- **From the side-to-move's perspective** (positive = the player about to move is
  doing well) — the negamax convention.
- **Never called in check** — search resolves checks first, and evaluating a
  king in check is meaningless. ARCHON debug-asserts this.
- **Strictly inside the normal range** — never in the mate or tablebase bands,
  which are reserved for *proven* results, not estimates.

Everything below is about how one number gets computed from a static position.
The number is measured in ARCHON's internal centipawn-like units, where the
mental model "a pawn is worth a few hundred, bigger is better" holds.

## Material: the first-order term, and why the values are not obvious

The single most important fact about a position is who has more material. The
classic beginner's piece values — pawn 1, knight 3, bishop 3, rook 5, queen 9 —
are a scaled-down version of what an engine uses. Multiply by ~100 and you have a
first evaluation: sum your pieces' values, subtract the opponent's, done. This
alone plays a recognizable game of chess, because most of the time the side with
more material is winning.

But the values are *not constants of nature*, and that is the first subtle point.
A bishop is worth more than a knight in open positions and less in closed ones. A
rook grows in value as the board empties. Two bishops together ("the bishop pair")
are worth more than two individual bishop-values. A knight on the rim is worth
noticeably less than one in the center. So "the value of a knight" is a fiction;
what exists is "the value of *this* knight, on *this* square, in *this* position."
Hand-built evaluation approximates that reality with two devices: a base material
value, and a per-square correction. It also, in fact, uses *two different* value
scales for two different purposes — a point worth flagging because it confuses
people reading the code:

- The **exchange values** in `src/types.rs` (pawn 208, knight 781, bishop 825,
  rook 1276, queen 2538) are used by static exchange evaluation (file 04) and
  pruning to answer "does this capture sequence win material?" They are chosen so
  that a sum of non-pawn material *uniquely identifies* a material configuration —
  a technical property those heuristics rely on.
- The **evaluation material values** baked into the piece-square tables
  (`MG_VALUE = [82, 337, 365, 477, 1025, …]` in `src/eval/psqt.rs`) are the ones
  that actually contribute to the position score.

They differ because they answer different questions; do not expect them to match.

## Piece-square tables

The per-square correction is a **piece-square table** (PST, or "PSQT"): for each
piece type, a table of 64 numbers, one per square, saying how much better or worse
that piece is on that square than its baseline. A knight on d5 might get +20, a
knight on a1 might get −50. Pawns get bonuses for advancing, kings get bonuses for
hiding in the corner (in the middlegame) — the tables encode a great deal of
positional wisdom as a simple lookup.

The beauty of PSTs is that they are *free* at runtime — a table index — and yet
capture most of the low-hanging positional knowledge. Evaluation with material +
PSTs is: for each piece, look up `base_value + pst[square]`, sum with sign by
color. About 32 table adds for a full board, which ARCHON notes is negligible next
to the cost of `make_move` and move generation, so it recomputes the whole thing
from scratch on every call rather than maintaining it incrementally
(`psqt_eval` in `src/eval/psqt.rs`).

ARCHON's tables are the **PeSTO** tables (by Ronald Friederich), a well-known
public set that were themselves *tuned* rather than hand-guessed — more on tuning
below. The project transcribes the *values* (not code — the licensing firewall
again) and folds the material value directly into each table entry, so the
evaluation loop is a single "look up and add" with no separate material step.

A small but instructive implementation trap lives here. The published PeSTO tables
are laid out "board-top-down" (index 0 = a8), while ARCHON's square mapping is
little-endian (index 0 = a1, from file 02). So the transcription flips every table
vertically with `sq ^ 56` (XOR-ing a square index with 56 mirrors its rank). Get
this flip wrong and every piece is evaluated as if it were on the wrong rank — a
bug that would pass casual testing but wreck play. ARCHON guards it with a
color-flip symmetry test suite: a white pawn on e4 must score exactly what a black
pawn on e5 scores (mirror images), and the start position must evaluate to exactly
zero (perfectly symmetric). Those tests are the tripwire for a botched flip.

Also note there is a *black* perspective to handle: the tables are written from
White's point of view, so a black piece looks up its **rank-mirrored** square
(`sq.relative_to(color)` in the code — White reads the square as-is, Black mirrors
the ranks). A black knight on d5 should score like a white knight on d4, and the
mirror achieves that with one operation.

## Tapered evaluation: the game changes as pieces leave

Here is a problem PSTs alone cannot solve. The *right* square for a piece depends
on the phase of the game. In the **middlegame**, with queens and rooks on the
board, the king wants to hide behind its pawns in the corner — exposing it is
suicide. In the **endgame**, with few pieces left, the king becomes a fighting
piece and wants to march to the center. One PST cannot express "corner is good"
and "center is good" simultaneously; they contradict.

The fix is **tapered evaluation**: keep *two* sets of tables — a middlegame set and
an endgame set — evaluate the position with both, and blend the two results by how
far the game has progressed. ARCHON's PeSTO tables come in exactly this pairing
(`MG_PESTO` and `EG_PESTO`, plus `MG_VALUE`/`EG_VALUE` in `src/eval/psqt.rs`). The
king PST, for instance, rewards the corner in the middlegame table and the center
in the endgame table; the blend does the rest.

"How far the game has progressed" is captured by a **game phase** number computed
from the remaining non-pawn material: each knight/bishop/rook/queen contributes a
weight (`GAME_PHASE_INC`), summing to a maximum at the start of the game (all pieces
present) and shrinking toward zero as pieces are traded off. The blend is a linear
interpolation:

```
        phase · mg_score  +  (phase_max − phase) · eg_score
score = ─────────────────────────────────────────────────────
                           phase_max
```

(This is `taper` in `src/eval/score.rs`.) At the opening, `phase = phase_max`, so
`score = mg_score` — pure middlegame tables. In a bare-pieces endgame,
`phase = 0`, so `score = eg_score` — pure endgame tables. In between, a smooth
mix. This gradual crossfade is why a piece "changes value" as the game
simplifies, without any discrete "now it's the endgame" switch. It is worth
seeing this in a concrete number: `psqt_eval`'s own test computes a white knight
on f3 as middlegame 354, endgame 278, and with one knight on the board the phase
is 1 out of 24, giving `(354·1 + 278·23)/24 = 281` — heavily weighted toward the
endgame value because almost all the material is gone.

## Tempo, and terms ARCHON's baseline omits

A few small evaluation ideas round out the classical picture:

- **Tempo.** The side to move has a small inherent advantage — it gets to act
  first. Many engines add a fixed "tempo bonus" of ~15–30 units to the side to
  move. ARCHON's PeSTO baseline *omits* it, because the PeSTO tables were tuned
  without one; adding it would double-count. (A worked-in tempo term is the kind
  of thing later tuning would introduce.)

- **The outer shape.** ARCHON wraps the raw material+PST inner score in a thin
  outer transform (`apply_outer_shape` in `src/eval/mod.rs`) shared by *all* eval
  phases, including the neural nets later. It does two things: it blends in an
  **optimism** term supplied by the search (a mild "the side to move is probably
  doing a bit better than the static score suggests near a fresh aspiration
  window") weighted by how much material is on the board, and it **damps the score
  toward zero as the fifty-move clock climbs** — because a position that is one
  reversible move from a draw is, in truth, closer to a draw than its material
  suggests. The result is clamped to stay strictly inside the non-mate range. This
  outer shape is deliberately factored out so search never has to know which
  evaluator is underneath.

- **Dead-draw detection.** Before any of the above, `evaluate` checks for
  insufficient mating material (king vs king, king and one minor vs king, king and
  two knights vs king — configurations where checkmate is impossible) and returns
  an exact draw score, short-circuiting the whole pipeline. This is cheap (it reads
  cached material counts) and prevents the engine from chasing a phantom advantage
  in a position no one can win.

## The concepts a full HCE engine adds

ARCHON's shipped classical evaluator is material + tapered PSTs + the outer shape —
deliberately lean, because it is a *fallback* behind the neural net, not the final
product. But the historical "hand-crafted evaluation" (HCE) engines that dominated
before ~2018 layered many more terms on top, and it is worth knowing the vocabulary,
because these concepts are exactly what the neural network in file 11 *learns to
approximate* without being told:

- **King safety** — count and weight the attackers near the enemy king, the holes
  in its pawn shield, open files pointing at it. Often the single largest positional
  term, and notoriously hard to tune: a king-safety bug can make an engine either
  recklessly sacrificial or paralyzed with fear.
- **Mobility** — how many squares each piece can move to. A cramped position with
  few legal moves is usually worse; a bishop with a long open diagonal is worth more
  than one hemmed in by its own pawns. Cheap to compute from the attack bitboards of
  file 03.
- **Pawn structure** — doubled pawns (two on one file, a weakness), isolated pawns
  (no friendly pawn on adjacent files, hard to defend), passed pawns (no enemy pawn
  can stop them promoting, very valuable), backward pawns, pawn chains. Pawn
  structure changes slowly, which is exactly why file 04 keeps a separate `pawn_key`
  — so a pawn-evaluation cache can be reused across positions with the same pawns.
- **Piece-specific terms** — rooks on open files, knights on protected outposts,
  bad bishops blocked by their own pawns, and so on.

Each is a hand-written formula with hand-chosen weights, and a strong HCE engine had
hundreds of such weights.

## Tuning, and the ceiling of hand-built evaluation

How do you set hundreds of weights? Not by guessing — by **tuning**. The standard
method, "Texel tuning," treats the weights as parameters of a model and fits them to
data: take millions of positions with known game outcomes, define a loss (how well
the evaluation's score, squashed through a sigmoid into a win-probability, predicts
the actual result), and adjust the weights by gradient descent to minimize it. The
PeSTO tables ARCHON uses were produced this way. Tuning turns evaluation from an art
into an optimization problem, and it is a large part of why late-era HCE engines
were so strong.

But tuning a hand-built evaluation runs into a hard ceiling, and this is the
punchline that sets up the rest of the series. The problem is **the functional
form**, not the weights. A hand-crafted evaluation is a *sum of human-conceived
terms*: material + king safety + mobility + pawn structure + … . Tuning finds the
best weights *for that fixed set of terms*, but it cannot invent a term a human did
not think to write. And chess positional understanding is full of interactions no
tidy formula captures: this knight is strong *because* that pawn is fixed *and* the
enemy bishop is the wrong color *and* the king is committed to the other side.
Enumerating and hand-coding every such interaction is hopeless — there are too many,
and most are subtle conjunctions of features.

Two limitations in particular:

1. **Linearity.** Most HCE terms combine additively (weighted sums). But real
   positional value is riddled with *nonlinear* interactions — feature A matters only
   when feature B is present. A weighted sum cannot represent "A AND B but not A
   alone" without an explicit A-and-B term, and there are combinatorially many such
   conjunctions.
2. **Human feature engineering.** Every term must be conceived, coded, and debugged
   by a person. The space of useful features is far larger than any team can
   hand-build, and the marginal new term yields less and less as the obvious ones
   are exhausted.

The escape, which the whole top of the field took around 2018–2020, is to stop
hand-designing the functional form and instead *learn* it: feed a neural network the
raw board (which pieces are on which squares, relative to each king) and let gradient
descent discover both the features and their nonlinear combination. That is
**NNUE** — "efficiently updatable neural network" evaluation — and it is worth,
against a strong HCE baseline, hundreds of Elo. ARCHON's own history makes the point
concretely: its first NNUE net beat the classical PSQT evaluator of this file by
about +696 Elo — a near-total win rate. The classical evaluation did not become
wrong; it became the *fallback*, kept behind the same frozen `evaluate` seam so the
network can be swapped in without search noticing. Files 11 and 12 are that story.

## What you now know

You know that evaluation collapses a static position into one side-to-move-relative
number, and that material is the first-order term — but that piece values are not
constants: they depend on square, phase, and position, which is why ARCHON uses two
different value scales (exchange values for SEE, evaluation values baked into the
tables). You understand piece-square tables as a per-square positional correction
(with the `sq ^ 56` transcription flip and the color mirror as real gotchas), and
tapered evaluation as the middlegame/endgame table blend driven by a material-based
game phase — the mechanism by which a piece's value smoothly changes as the board
simplifies. You have seen the outer shape (optimism, fifty-move damping,
dead-draw detection) that wraps every ARCHON evaluator, and the vocabulary of the
richer terms — king safety, mobility, pawn structure — that full HCE engines added.
And you understand the ceiling: tuning optimizes the weights of a fixed, human-chosen,
largely linear functional form, but cannot invent the nonlinear feature interactions
that real chess understanding requires — which is exactly why the field, and ARCHON,
moved to learned evaluation.

**Next:** `09-move-ordering-and-history.md` — file 05 showed that alpha-beta's whole
payoff depends on trying good moves first, and files 06–07 showed the transposition
table feeding best-moves forward. Now we look at the rest of the ordering machinery:
the MovePicker that decides what to try next, and the history tables that learn,
during a single search, which quiet moves tend to be good.
