# 05 — Minimax and alpha-beta

This is where the engine gets its brain. We have a board, legal moves, and (soon)
an evaluation that scores a position. Search is the algorithm that puts them
together: walk the game tree, propagate values by the minimax logic of file 01,
and — the key to making it practical — *prove that most branches cannot affect
the answer and skip them*. That proof is **alpha-beta pruning**, and understanding
it deeply is understanding how a chess engine can look twenty moves ahead. We
build it from minimax, reformulate to the sign-trick form ARCHON actually uses,
walk a concrete pruning example by hand, and then meet the refinements — the
window as a contract, principal variation search, and mate scoring.

## From minimax to negamax

Recall minimax from file 01: propagate leaf values up the tree, maximizing at
your own nodes and minimizing at the opponent's. Written directly, that is two
mutually recursive cases — a `max` routine and a `min` routine — with mirror-image
logic and twice the surface area for sign bugs.

**Negamax** collapses the two into one using the zero-sum symmetry. The
observation: a position's value to the player to move is the *negation* of its
value to the opponent. If I evaluate every position from the perspective of
*whoever is about to move*, then at each node I always want to *maximize*, and to
get a child's value from my perspective I just negate the value the child
computed from the opponent's perspective. One case, one sign flip:

```
value(node) = max over legal moves m of  ( − value(child after m) )
```

The leaf evaluation must obey the same convention — return the score from the
side-to-move's perspective — which is exactly the contract file 01 pinned on
ARCHON's `evaluate`. Negamax is not a different algorithm from minimax; it is the
same computation written to exploit the symmetry, and every serious engine uses
it because the single-case code is dramatically less bug-prone. All the search
code in `src/search/mod.rs` is negamax; whenever you see a recursive call
negated, `-search(...)`, that is the sign trick.

## The problem negamax alone does not solve

Plain negamax still visits every node: `b^d` of them. It computes the exact
minimax value, but it does no pruning, so it buys us nothing over brute force. We
need to skip work. The insight that lets us is almost obvious once stated, and
completely transformative:

> Once you know a move is good enough to refute a line, you do not need to know
> *how* good the refutation is. You just need to know the line is refuted.

Concretely: suppose you are evaluating your options, and you have already found a
move that guarantees you a score of at least +2. Now you start examining a second
move, and partway through the opponent's replies you discover the opponent has a
reply that holds you to +1. You can **stop examining that second move
immediately.** You already have a +2 option; a move that lets the opponent get you
to +1 is worse, and it does not matter whether the opponent could actually hold
you to +1 or to −5 — the move is already rejected. The opponent's remaining
replies to that second move need never be generated. That is the cut.

## Alpha, beta, and the window

To turn that insight into an algorithm we thread two bounds through the recursion.
At every node, the search carries a **window** `[alpha, beta]`:

- **alpha** is the best score the side to move has *already guaranteed* somewhere
  — a floor. "I can already get at least alpha; only tell me about moves that beat
  it."
- **beta** is the best the *opponent* will allow — a ceiling, imposed from above.
  "If I find a move scoring beta or more, my parent will never let this position
  happen, so I can stop."

The window is a *contract* between a node and its parent: "I promise to return a
value that is useful for decisions inside `(alpha, beta)`; outside that range I
only promise a bound, not the exact number." This contract is what makes pruning
sound, and it is worth stating precisely because it governs everything that
follows, including the transposition table in file 07.

The algorithm (negamax with alpha-beta):

```
search(node, alpha, beta):
    if leaf: return evaluate(node)
    for each move m:
        score = − search(child(m), −beta, −alpha)   # note the swap and negate
        if score ≥ beta:
            return score          # FAIL-HIGH: too good; parent won't allow it. Cut.
        if score > alpha:
            alpha = score         # new floor; this move is the best so far
    return alpha                  # FAIL-LOW or exact, depending
```

Two details make it click. First, the recursive call passes `(−beta, −alpha)` and
negates the result — because the child sees the world from the opponent's
perspective, so our floor is their ceiling and vice versa, with signs flipped. The
sign gymnastics are exactly the negamax convention applied to the window. Second,
the `score ≥ beta` test is the cut: the moment a move meets or beats beta, we
return without looking at the remaining moves. That early return is where the `b^d`
work disappears.

### Fail-high and fail-low

Two outcomes of a node need names, because the rest of the engine speaks in them:

- **Fail-high** (`score ≥ beta`): the node found a move so good the opponent will
  avoid this whole position. We return a **lower bound** — "the true value is *at
  least* this" — because we cut before checking every move, so the true value
  could be even higher; we just did not bother to find out.
- **Fail-low** (no move exceeded alpha): every move was disappointing; nothing beat
  the floor we already had. We return an **upper bound** — "the true value is *at
  most* this."
- **Exact**: some move landed strictly inside the window (beat alpha without
  reaching beta). Then alpha ends up an exact value and this is a **principal
  variation** node — a node on the current best line.

Those three cases — lower bound, upper bound, exact — are precisely the `Bound`
enum ARCHON stores in the transposition table (`Bound::Lower`, `Bound::Upper`,
`Bound::Exact` in `src/tt.rs`). A stored search result is only meaningful together
with its bound type, because a fail-high value is not the position's value — it is
merely a proof that the value is at least that much. File 07 returns to this; for
now, hold onto the idea that *a value without its bound type is a lie*.

## Why move ordering changes the complexity

Here is the payoff, and the single most important quantitative fact about search.
With **perfect move ordering** — if at every node you happen to try the best move
first — alpha-beta reduces the number of leaves examined from `b^d` to roughly
`b^(d/2)`. The effective branching factor drops from `b` to `√b`.

Intuition for the square root: to *prove* a move is best you must examine all of
your own moves (you cannot skip one, or you might miss a better one) — that is a
full `b`-way fan-out at "your" levels. But to *refute* each of the opponent's
replies, you only need to find *one* good enough counter — the first refutation
cuts the rest. So the tree fans out fully on your levels and minimally on the
opponent's, alternating, and the two effects multiply to `b^(d/2)`.

What does `√b` buy? For `b ≈ 35`, `√35 ≈ 5.9`. So alpha-beta with good ordering
searches to depth `d` for about the cost plain minimax pays for depth `d/2`.
Restated: **for the same work, alpha-beta doubles your search depth.** In a game
where each extra ply of depth is worth tens of Elo, doubling the reachable depth
is the difference between a toy and a real engine.

But note the condition: *good move ordering*. The `b^(d/2)` bound assumes you try
the best move first at every node. Try the *worst* move first everywhere and you
get no cuts at all — back to `b^d`. So the value of alpha-beta is entirely
contingent on ordering moves well, and that is why an enormous fraction of
engine effort (all of file 09, plus the transposition table's "try the previously
best move first," plus staged capture-first generation from file 03) goes into
making the best move appear early. Search and move ordering are not separable
concerns; ordering is what makes search fast.

## A worked example

Let us prune a small tree by hand. White to move (MAX), two ply deep, branching
factor 3 at the root, and leaf values from White's perspective. We search
left-to-right, starting with the maximally open window `[−∞, +∞]`.

```
                    ROOT  (White, MAX)      alpha starts −∞
                 /        |          \
               A          B           C
             (Black,    (Black,     (Black,
              MIN)       MIN)        MIN)
            / | \       / | \       / | \
           3  5  6     2  ?  ?     ...
```

**Move A.** We search its children (Black to move, so Black minimizes). Black's
options under A evaluate to 3, 5, 6; Black picks the minimum, **3**. So A is worth
3 to White. Back at the root, alpha becomes 3: White can already guarantee 3.

**Move B.** Now we search B's children with the knowledge alpha = 3 — i.e., we
only care about B if it can beat 3. B's first child evaluates to **2**. Black, who
minimizes at B, will take *at most* 2 (the other children can only pull it lower).
But 2 < 3 = alpha. So B is already worse than A *no matter what its other children
are*. We **prune** B's remaining children — never generate them, never evaluate
them. This is the cut: from White's viewpoint, B fails low, and its exact value
(2 or less) is irrelevant.

**Move C.** Search normally, again with alpha = 3 as the floor, cutting any child
that drops C below 3.

We skipped B's second and third subtrees entirely. In a real tree those subtrees
are themselves deep, so the savings compound multiplicatively at every level.
That is alpha-beta: the more good moves you find early (here, move A giving a
strong floor immediately), the more of everything else you get to throw away.

## Principal variation search (the zero-window trick)

ARCHON does not run plain alpha-beta. It runs **principal variation search**
(PVS), also called "scout," which sharpens the pruning further using the same
insight one level deeper. The reasoning:

After you have searched the first move at a node and it looks best, you *believe*
the remaining moves are worse — you are just verifying it. Verifying "this move is
worse than the best so far" is a **yes/no** question, and a yes/no question can be
answered with a much narrower, cheaper search than "exactly how good is this
move?"

So PVS searches the first move with the full window `[alpha, beta]` to get an
exact value, then searches every later move with a **zero-window** (also
"null-window") `[alpha, alpha+1]` — a window one unit wide. A zero-window search
cannot return an exact value (there is no room between the bounds); it can only
answer "is this move worse than alpha (fail-low, as expected) or does it somehow
beat alpha (fail-high)?" Because the window is so tight, the search cuts almost
immediately — zero-window searches are cheap. The bet is that *almost all* later
moves fail low quickly (confirming the first move was best), and only rarely does
one surprise you by failing high.

When a later move *does* fail high against the zero window, the bet was wrong: this
move might actually be better, and the cheap search cannot tell you by how much. So
you **re-search** it with the full window to get its real value. That re-search is
wasted work — you searched the move twice. PVS pays off only when re-searches are
rare, which is why it is coupled so tightly to move ordering: if your first move is
usually best (good ordering), zero-window scouts confirm it cheaply and
re-searches almost never happen; if your ordering is bad, you pay for constant
re-searches and PVS loses. In a well-ordered search, PVS is a clear win, and it is
what ARCHON's move loop implements — the header of `src/search/mod.rs` describes
the exact three-call structure: a full-window search for the first move, a
zero-window scout for the rest, and a full-window re-search for any scout that
beats alpha. Files 06, 09, and 10 all build on this scaffold: aspiration windows
(file 06) apply the same "search with a narrow window, widen on surprise" idea at
the very root, and reductions (file 10) shrink the *depth* of those scout searches
too.

## Mate scores and the ply adjustment

One special value class needs care: **checkmate**. A checkmate is not a normal
evaluation; it is a terminal outcome, and the search must not only recognize it but
prefer a *faster* mate to a slower one (mate in 3 is better than mate in 7 — you
want to actually finish the game). ARCHON encodes this with the mate band near the
extremes of the value range.

`VALUE_MATE` is 32000 (in `src/types.rs`), far above any static evaluation of a
normal position. A checkmate delivered right now scores `VALUE_MATE`; being mated
scores `−VALUE_MATE`. But to prefer faster mates, ARCHON stores mates as
"`VALUE_MATE` minus the number of ply to reach it": `mate_in(ply) = VALUE_MATE −
ply`. So mate in 1 ply scores 31999, mate in 3 ply scores 31997, and the search,
maximizing, naturally prefers the higher score — the *shorter* mate. The mirror
holds for being mated: `mated_in(ply) = −VALUE_MATE + ply`, and the search prefers
the *longer* defense (delay the mate as long as possible).

This creates a subtlety at the transposition table boundary. A mate score is
"distance from *this* node," but the TT stores results that get reused at
different depths in the tree. A mate-in-2-from-here stored and then retrieved 4
ply deeper is a mate-in-6-from-the-root, or whatever. So ARCHON adjusts mate
scores on the way into and out of the TT: `value_to_tt` converts a "distance from
root" score to "distance from this node" before storing, and `value_from_tt`
converts back on retrieval (both in `src/types.rs`), by adding or subtracting the
node's ply. Any value that is not in the mate band passes through unchanged. This
ply-correcting dance is easy to get subtly wrong and is a classic engine-bug
habitat; ARCHON isolates it in those two functions so the arithmetic lives in
exactly one place. There is a parallel "TB band" just below the mate band for
tablebase-proven results, which behaves the same way; you can treat it as a second
mate-like class for now.

## What you now know

You know negamax as the single-case, sign-flipping reformulation of minimax that
every serious engine uses, and why plain negamax alone saves nothing. You
understand alpha-beta pruning from its core insight — a refuted move needs no exact
value — through the `[alpha, beta]` window as a contract, the fail-high /
fail-low / exact trichotomy (which is exactly the TT's `Bound` type), and the
central quantitative fact that good move ordering turns `b^d` into `b^(d/2)`,
doubling reachable depth. You have pruned a tree by hand and seen where the savings
come from. You know PVS/zero-window search as the refinement ARCHON actually runs —
cheap yes/no scouts for all but the first move, re-searching only on surprise, and
therefore utterly dependent on good ordering. And you understand mate scoring as
`VALUE_MATE − ply` to prefer faster mates, plus the ply adjustment that keeps mate
scores honest across the transposition table.

**Next:** `06-iterative-deepening-quiescence-aspiration.md` — the alpha-beta of
this file is the skeleton, but three refinements make it actually work: searching
depths 1, 2, 3, … in turn rather than jumping to the target (which, surprisingly,
is *faster*), extending the search at the leaves to avoid being fooled by pending
captures, and narrowing the root window to a guess. Those three turn a correct
search into a strong one.
