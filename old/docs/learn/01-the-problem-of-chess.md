# 01 — The problem of chess

Chess has simple rules and a colossal number of consequences. That gap — trivial
to describe, impossibly large to enumerate — is the entire engineering problem.
This file lays out the problem precisely, introduces the two ideas that tame it
(search and evaluation), and defines the vocabulary the rest of the series
leans on. No code yet; this is the map before the territory.

## The game tree

Start from a position. List every legal move. Each move leads to a new
position, from which the opponent has their own list of legal moves, and so on.
This branching structure is the **game tree**: nodes are positions, edges are
moves, and a path from the root down to a leaf is one possible continuation of
the game.

```
                     (start position)
                    /       |        \
                 e4        d4         Nf3      ... ~20 legal first moves
                /  \       /  \
              e5   c5    d5   Nf6   ... ~20 replies to each
             / \   ...
           Nf3  ...
```

A single ply — one move by one side — is one level of this tree. (Chess people
say **ply** for a half-move so there is no ambiguity: "one move" colloquially
means one move by *each* side, i.e. two ply.) A game that lasts 40 moves is 80
ply deep.

### Branching factor

The **branching factor** `b` is the average number of legal moves in a
position. In chess it is roughly **35**. It is near 20 in the opening, swells
into the 30s and 40s in a busy middlegame, and shrinks toward single digits in
simple endgames. The theoretical maximum number of legal moves in any one
position is 218 — which is why ARCHON's move list is sized at 256 slots (see
`MAX_MOVES` in `src/moves.rs`), comfortably above the worst case.

Why does `b ≈ 35` matter so much? Because the number of leaves at depth `d`
grows like `b^d`. Plug in numbers:

| Depth `d` (ply) | Approx. positions `35^d` |
|-----------------|--------------------------|
| 1               | 35                       |
| 2               | 1,225                    |
| 3               | ~43,000                  |
| 4               | ~1.5 million             |
| 6               | ~1.8 billion             |
| 10              | ~2.7 × 10¹⁵              |
| 20              | ~4 × 10³⁰               |

A strong engine routinely reaches an effective depth of 20+ ply. The full tree
at depth 20 has more nodes than there are seconds since the Big Bang. **You
cannot enumerate it.** Not with a faster computer, not with a bigger data
center — the growth is exponential and the exponent is large. The entire game
of chess (all games, from the start) is estimated at somewhere around 10¹²⁰
positions, a number with no physical meaning. Brute force is off the table
before you write a line of code.

This is the central fact. Everything an engine does is a strategy for looking
at a vanishingly small slice of this tree and still choosing well.

It is worth pausing on *why a faster computer does not rescue you*, because the
instinct is strong. Suppose today's hardware lets you search depth 12 in a
second. Moore's law, when it was still healthy, doubled compute roughly every
two years. But each extra ply of chess costs a factor of `√b ≈ 6` more work
(that square root is the payoff of file 05's pruning; without it, a factor of
35). So *doubling* your compute buys you less than half a ply — you would need
about 36× more compute for two extra ply. To brute-force your way from depth 12
to depth 20 would take a factor of roughly `6^8 ≈ 1.7 million` more compute, or
decades of hardware progress, to gain eight ply that a good algorithm gets for
free by pruning and evaluating. Exponential search spaces laugh at linear (or
even exponential-but-slow) hardware growth. The leverage is entirely in the
algorithm, which is why this series spends far more time on search and
evaluation than on any low-level speed trick.

This also explains why chess is *unsolved* while simpler games are solved.
Tic-tac-toe (a few thousand positions) and checkers (famously solved in 2007,
after 18 years of computation, to a draw under perfect play) have small enough
trees to conquer with enough cleverness and time. Chess, at 10¹²⁰ positions,
is not in the same universe — and it never will be. An engine does not solve
chess; it plays it well by looking at almost none of it.

## The two escapes

There are exactly two ways to avoid exploring the whole tree, and every engine
uses both:

1. **Don't search all the way to the end. Stop early and *estimate*.** Instead
   of playing every line out to checkmate, look ahead some fixed number of ply,
   then guess how good the resulting positions are. That guess is
   **evaluation**.
2. **Don't look at every branch. Prove some branches can't matter and *skip*
   them.** Most moves in most positions are bad, and you can often prove a move
   is bad without examining it fully. That pruning is what **search**
   algorithms do.

The rest of the series is, at bottom, an elaboration of these two escapes.
Files 05–07, 09, and 10 are about search and pruning. Files 08, 11, and 12 are
about evaluation. Files 02–04 build the machinery both depend on. The
interesting part is that the two escapes reinforce each other: better
evaluation makes pruning safer, and better pruning lets you search deeper,
which makes evaluation matter less. We will see this loop again and again.

## Minimax: the logic of a two-player game

Before pruning, we need to know what we are even computing. Chess is a
**zero-sum, perfect-information, two-player game**: what is good for me is
exactly bad for you (zero-sum), nothing is hidden (perfect information), and we
alternate turns. In such a game there is a clean, non-negotiable definition of
the best move, called **minimax**.

The idea: assign every leaf a number, where a large positive number means "good
for White" and a large negative number means "good for Black." Now propagate
values up the tree. At a node where it is White's turn to move, White will pick
the child with the **maximum** value — White is trying to maximize the score. At
a node where it is Black's turn, Black picks the **minimum**. Alternate max, min,
max, min up the levels; the value that reaches the root is the value of the
position under best play, and the move that achieved it at the root is the best
move.

A tiny worked example. Suppose it is White to move, we look two ply ahead, and
the leaf evaluations (White's perspective) come out like this:

```
                White to move (MAX)
               /                  \
          Black to move        Black to move
            (MIN)                 (MIN)
           /     \               /     \
        +3       -1           +2       +2
```

Black, moving second, minimizes. Under the left branch Black picks −1 (the
smaller of +3, −1). Under the right branch Black picks +2. Now White,
maximizing, compares −1 (left) against +2 (right) and chooses the right branch,
value +2. Note that the tempting +3 leaf is a mirage: Black would never let
White reach it, because Black gets to choose and prefers −1. Minimax bakes in
the assumption that the opponent plays well.

The word "minimax" describes the alternating max/min pattern. In the next file
of the search arc (05) we replace it with **negamax**, an equivalent
reformulation that exploits the zero-sum symmetry so we only ever write the
"maximize" case — a small trick that removes a whole class of sign bugs. But
the semantics are identical to what you just saw.

## Search meets evaluation: the decomposition

Minimax as stated wants leaf values, and true leaf values only exist at the end
of the game (win = +∞, draw = 0, loss = −∞). Since we cannot search to the end,
we cut the search off at some depth and call **evaluation** on the non-terminal
positions there, treating its estimate as if it were a true leaf value. So the
engine is really two components bolted together:

- **Search** decides *which* positions to look at and propagates values through
  the tree (minimax, plus all the pruning tricks).
- **Evaluation** looks at *one* position and returns a single number estimating
  who stands better and by how much.

This decomposition is so fundamental that ARCHON pins it as a hard interface.
The evaluation entry point is one function, `evaluate`, in `src/eval/mod.rs`,
and its signature is deliberately frozen across three eras of the engine
(hand-tuned tables, then a neural net, then a bigger neural net) so that search
never has to change when evaluation is swapped underneath it. Search calls
`evaluate`; it does not care how the number is produced. That clean seam is why
files 08, 11, and 12 can talk about evaluation almost independently of the
search files.

Two conventions from that seam are worth stating now because they recur
everywhere:

- **Scores are from the side-to-move's perspective.** A positive number means
  "the player about to move is doing well," whichever color that is. This is
  the negamax convention (file 05) and it is what lets the code write one
  maximizing case instead of two.
- **Scores are in "centipawns," roughly.** The unit is historically hundredths
  of a pawn: +100 means "up about one pawn." ARCHON's internal units are
  Stockfish-derived and not literally 1/100 of a pawn — a pawn's exchange value
  is 208 internal units, not 100 (see `PAWN_VALUE` in `src/types.rs`) — but the
  mental model "bigger number = better, a pawn is worth a few hundred" holds.
  Special sentinel values are reserved at the extremes: a forced checkmate is
  scored near ±32000 (`VALUE_MATE` in `src/types.rs`), far outside the range any
  evaluation of a normal position can produce, so "I found a mate" is
  unambiguous.

## The shape of a modern engine

With search and evaluation as the two pillars, a real engine needs a few more
load-bearing components. Here is the whole machine in one view; every box is a
file or two later in the series.

```
        ┌──────────────────────────────────────────────┐
        │  UCI protocol  (talk to the GUI / clock)      │  file 13
        └───────────────┬──────────────────────────────┘
                        │ "search this position, you have N ms"
        ┌───────────────▼──────────────────────────────┐
        │  SEARCH  (iterative deepening + alpha-beta +  │  files 05, 06, 07,
        │   pruning + move ordering + transposition TT) │        09, 10
        └───┬───────────────────────────┬──────────────┘
            │ needs legal moves          │ needs a leaf score
   ┌────────▼─────────┐         ┌────────▼─────────┐
   │ MOVE GENERATION  │ file 03 │   EVALUATION      │ files 08, 11, 12
   │ (bitboards,      │         │ (PSQT / NNUE)     │
   │  magic lookups)  │         └───────────────────┘
   └────────┬─────────┘
   ┌────────▼─────────┐
   │ BOARD + STATE    │ files 02, 04
   │ (make/unmake,    │
   │  Zobrist hashing)│
   └──────────────────┘
```

Read bottom to top:

- **Board and state** (files 02, 04) hold the position and can *make* a move to
  reach a child and *unmake* it to return — the workhorse operation of search,
  called millions of times per second.
- **Move generation** (file 03) produces the legal moves from a position. It has
  to be fast and exactly correct: a single missing or spurious move is a
  catastrophic bug that silently corrupts the search.
- **Evaluation** (files 08, 11, 12) scores a leaf. Historically a sum of
  hand-tuned terms (material, piece placement, king safety); in the modern era a
  small neural network. ARCHON has both, behind that one frozen `evaluate` seam.
- **Search** (files 05–07, 09, 10) is the brain: it drives the tree walk,
  prunes, caches, orders moves, and manages depth. This is where most of the
  playing strength is won or lost.
- **UCI** (file 13) is the thin protocol layer that connects the engine to a
  graphical front-end or a tournament manager: it receives positions and clock
  information and reports the engine's chosen move and its thinking.

Everything is in service of one query, issued over and over: *given this
position and this time budget, what is the best move?*

## What "Elo" measures

We keep saying "stronger." What is the unit? It is **Elo**, a rating system
(named after Arpad Elo) that turns head-to-head results into a single number per
player. The only thing Elo directly encodes is an *expected score* between two
players from their rating difference:

```
                       1
    E_A  =  ───────────────────────
              1 + 10^((R_B − R_A)/400)
```

`E_A` is the fraction of points player A is expected to score (a win is 1, a
draw is ½). The 400 is a convention. The consequences worth memorizing:

- A **0** rating difference predicts a **50%** score — a coin flip.
- A **+200** difference predicts about **76%**.
- A **+400** difference predicts about **91%**.

Elo is *relative*: it only measures who beats whom and by how much. The absolute
numbers are anchored to a reference pool. When this project says something like
"the classical evaluation is worth about 2720 Elo" or "net-0 was +696 Elo over
the hand-tuned tables," those are differences measured by playing thousands of
games between versions and fitting the results to the formula above. A "+696"
improvement means the new version scores about 99% against the old one — a rout,
in chess terms, which is why such a jump is a headline event.

Two practical points that file 14 develops. First, because Elo is statistical,
measuring a *small* improvement (say +5 Elo, a genuinely useful gain at the top)
requires *many* games to distinguish from noise — often tens of thousands. The
reason is that a single game's result is nearly a coin flip between near-equal
players: to tell a 51% player from a 50% player with confidence, you need enough
games that the ±1% signal rises above the random scatter, and the number of
games grows roughly as the *inverse square* of the effect size. Halve the Elo
gain you are trying to detect and you quadruple the games needed. This is why
engine development is as much an exercise in statistics as in programming, and
why file 14 is devoted to the machinery (SPRT) that decides when enough games
have been played. Second, Elo differences are only comparable under fixed
conditions: same time control, same hardware, same opponent pool. Change the
clock and the numbers move — a change that helps at fast time controls can hurt
at slow ones. When we quote gains, they are always "at equal time" or "at fixed
nodes," and file 14 explains why that distinction matters.

## What you now know

You know why chess cannot be solved by brute force: the game tree branches by
about 35 and grows like `35^d`, which outruns any conceivable amount of compute
within a handful of ply. You know the two escapes every engine uses — stop early
and *evaluate*, and prove branches irrelevant and *prune* — and that they
reinforce each other. You understand minimax as the definition of best play in a
zero-sum game, and the clean search/evaluation decomposition that ARCHON pins as
a frozen `evaluate` seam. You have the whole-engine block diagram in your head,
and you know that "strength" is measured in Elo, a relative scale where +400
means winning nine games in ten.

**Next:** `02-board-representation.md` — before we can generate moves or search,
we need to represent a chessboard in a way a computer can manipulate at hundreds
of millions of operations per second. That means thinking of a board not as a
grid but as a set of 64-bit integers.
